"""Local access to the prepared story corpus.

This module deliberately owns no retrieval or generation resources.  It combines
the source-record store with the local ordering, glossary, and summary cache so
callers that only need corpus data do not have to construct ``StoryQueryEngine``.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .indexer.source_store import DEFAULT_SOURCE_DB_PATH, SourceRecordStore
from .lexical import glossary_aliases_for
from .story_order import StoryOrder, default_story_order, load_story_order


class CorpusSourceStore(Protocol):
    """Minimal source-store surface required by the read-only corpus facade."""

    def iter_scenes(self) -> list[dict[str, Any]]: ...

    def get_scene(self, file_path: str, scene_index: int) -> dict[str, Any] | None: ...

    def max_story_order(self, *, arc_id: str, episode: int | None = None) -> int | None: ...


@dataclass(frozen=True)
class SummaryRecord:
    """A cached summary and its cache metadata, without generation semantics."""

    cache_key: str
    summary: str
    fingerprint: str
    schema_version: str
    inputs: dict[str, Any]
    metadata: dict[str, str]


@dataclass(frozen=True)
class SceneReference:
    """A scene locator pinned to the corpus revision that produced it."""

    revision: str
    file_path: str
    scene_index: int
    scene_id: str


class StoryCorpus:
    """Read source records and local corpus metadata without query dependencies.

    The service intentionally does not validate cache freshness or generate a
    replacement summary.  ``SummaryRecord.inputs`` and ``fingerprint`` are
    exposed so a later preparation step can perform that validation explicitly.
    """

    def __init__(
        self,
        source_store: CorpusSourceStore,
        *,
        story_order: StoryOrder | None = None,
        glossary: dict[str, dict[str, str]] | None = None,
        summaries: dict[str, Any] | None = None,
        revision: str | None = None,
    ) -> None:
        self.source_store = source_store
        self.story_order = story_order or default_story_order()
        self.glossary = glossary or {}
        self._summaries = summaries or {}
        self.revision = revision

    @classmethod
    def from_prepared_dir(cls, prepared_dir: str | Path) -> StoryCorpus:
        """Open one prepared directory after validating its shared revision."""
        directory, catalog = validate_prepared_corpus(prepared_dir)
        story_order = _story_order_from_catalog(catalog)
        glossary = _glossary_from_catalog(catalog)
        return cls(
            SourceRecordStore(directory / "source_records.db"),
            story_order=story_order,
            glossary=glossary,
            summaries={
                str(record["cache_key"]): record
                for record in catalog.get("summaries", [])
                if isinstance(record, dict) and isinstance(record.get("cache_key"), str)
            },
            revision=str(catalog["revision"]),
        )

    @classmethod
    def from_publication(cls, publication_dir: str | Path) -> StoryCorpus:
        """Open the generation named by ``current.json`` and pin it for this reader."""
        generation, expected_revision = resolve_publication_pointer(publication_dir)
        try:
            corpus = cls.from_prepared_dir(generation)
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError("Published corpus generation is invalid") from exc
        if corpus.revision != expected_revision:
            raise ValueError("Published corpus pointer revision does not match its generation")
        return corpus

    @classmethod
    def from_paths(
        cls,
        *,
        source_db: str | Path = DEFAULT_SOURCE_DB_PATH,
        story_order_file: str | Path | None = None,
        glossary_file: str | Path | None = None,
        summary_cache_file: str | Path | None = None,
    ) -> StoryCorpus:
        """Open local inputs, initializing the source DB and tolerating absent JSON files."""
        story_order = (
            load_story_order(story_order_file) if story_order_file is not None else default_story_order()
        )
        glossary = _load_optional_json_object(glossary_file, "Glossary") if glossary_file else {}
        summaries = (
            _load_optional_json_object(summary_cache_file, "Summary cache")
            if summary_cache_file
            else {}
        )
        return cls(
            SourceRecordStore(source_db),
            story_order=story_order,
            glossary=_as_glossary(glossary),
            summaries=summaries,
        )

    @property
    def summary_cache(self) -> dict[str, Any]:
        """Return the loaded cache mapping for callers doing their own validation."""
        return self._summaries

    def iter_scenes(self) -> list[dict[str, Any]]:
        """Return persisted source scenes in deterministic story order."""
        return self.source_store.iter_scenes()

    def get_scene(self, file_path: str, scene_index: int) -> dict[str, Any] | None:
        return self.source_store.get_scene(file_path, scene_index)

    def scene_reference(self, file_path: str, scene_index: int) -> SceneReference | None:
        """Create a revision-bound reference for a currently readable scene."""
        scene = self.get_scene(file_path, scene_index)
        if scene is None:
            return None
        if self.revision is None:
            raise ValueError("Scene references require a corpus revision")
        return SceneReference(
            revision=self.revision,
            file_path=str(scene["file_path"]),
            scene_index=int(scene["scene_index"]),
            scene_id=str(scene["scene_id"]),
        )

    def get_scene_by_reference(self, reference: SceneReference) -> dict[str, Any]:
        """Read a scene only when its reference belongs to this revision."""
        if self.revision is None or reference.revision != self.revision:
            raise ValueError("Scene reference belongs to a different corpus revision")
        scene = self.get_scene(reference.file_path, reference.scene_index)
        if scene is None or str(scene["scene_id"]) != reference.scene_id:
            raise ValueError("Scene reference is stale for this corpus revision")
        return scene

    def max_story_order(self, *, arc_id: str, episode: int | None = None) -> int | None:
        return self.source_store.max_story_order(arc_id=arc_id, episode=episode)

    def get_summary(self, cache_key: str) -> SummaryRecord | None:
        raw = self._summaries.get(cache_key)
        if not isinstance(raw, dict):
            return None
        summary = raw.get("summary")
        if not isinstance(summary, str):
            return None
        inputs = raw.get("inputs")
        metadata = _summary_metadata(cache_key)
        raw_metadata = raw.get("metadata")
        if isinstance(raw_metadata, dict):
            metadata.update({str(key): str(value) for key, value in raw_metadata.items()})
        return SummaryRecord(
            cache_key=cache_key,
            summary=summary,
            fingerprint=_string_value(raw.get("fingerprint")),
            schema_version=_string_value(raw.get("schema_version")),
            inputs=dict(inputs) if isinstance(inputs, dict) else {},
            metadata=metadata,
        )

    def iter_summaries(self) -> list[SummaryRecord]:
        """Return readable cached summaries and retained metadata in stable key order."""
        records = [
            record
            for key in sorted(self._summaries)
            if (record := self.get_summary(key)) is not None
        ]
        return records

    def lookup_glossary(self, term: str) -> dict[str, str] | None:
        """Resolve a canonical glossary term, translation, or generated alias."""
        folded = term.casefold()
        for category, terms in self.glossary.items():
            for canonical, translation in terms.items():
                if any(alias == term or alias.casefold() == folded for alias in glossary_aliases_for(canonical, translation)):
                    return {
                        "category": category,
                        "term": canonical,
                        "translation": translation,
                    }
        return None


def _load_json_object(path: str | Path, label: str) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"{label} not found: {file_path}")
    with file_path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def _load_optional_json_object(path: str | Path, label: str) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    return _load_json_object(file_path, label)


def _as_glossary(value: dict[str, Any]) -> dict[str, dict[str, str]]:
    return {
        category: terms
        for category, terms in value.items()
        if isinstance(category, str)
        and isinstance(terms, dict)
        and all(isinstance(key, str) and isinstance(term, str) for key, term in terms.items())
    }


def _glossary_from_catalog(catalog: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Load an embedded glossary, leaving older catalogs metadata-unavailable."""
    if "glossary" not in catalog:
        if isinstance(catalog.get("schema_version"), int) and catalog["schema_version"] >= 5:
            raise ValueError("Prepared corpus glossary metadata is missing")
        return {}
    value = catalog["glossary"]
    if not isinstance(value, dict):
        raise ValueError("Prepared corpus glossary metadata is invalid")
    glossary = _as_glossary(value)
    if len(glossary) != len(value):
        raise ValueError("Prepared corpus glossary metadata is invalid")
    return glossary


def _story_order_from_catalog(catalog: dict[str, Any]) -> StoryOrder:
    """Reconstruct embedded ordering metadata, falling back for legacy catalogs."""
    raw = catalog.get("story_order")
    is_current = isinstance(catalog.get("schema_version"), int) and catalog["schema_version"] >= 5
    if raw is None and not is_current:
        revision_inputs = catalog.get("revision_inputs")
        if isinstance(revision_inputs, dict):
            raw = revision_inputs.get("story_order")
    if not isinstance(raw, dict):
        if is_current:
            raise ValueError("Prepared corpus story-order metadata is missing")
        return default_story_order()
    positions = raw.get("positions")
    if not isinstance(positions, dict):
        raise ValueError("Prepared corpus story-order metadata is invalid")
    try:
        chronological = _order_positions(positions["chronological"], 2)
        summary = _order_positions(positions["summary"], 2)
        parts = _part_positions(positions["parts"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Prepared corpus story-order metadata is invalid") from exc
    return StoryOrder(
        chronological_positions=chronological,
        summary_positions=summary,
        part_positions=parts,
    )


def _order_positions(value: Any, key_size: int) -> dict[tuple[str, str], tuple[int, int]]:
    if not isinstance(value, list):
        raise TypeError("order positions must be a list")
    result: dict[tuple[str, str], tuple[int, int]] = {}
    for item in value:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not isinstance(item[0], list)
            or len(item[0]) != key_size
            or not all(isinstance(part, str) for part in item[0])
            or not isinstance(item[1], list)
            or len(item[1]) != 2
            or not all(isinstance(position, int) for position in item[1])
        ):
            raise ValueError("invalid order position")
        result[tuple(item[0])] = tuple(item[1])  # type: ignore[arg-type]
    return result


def _part_positions(value: Any) -> dict[tuple[str, str, str], dict[str, int]]:
    if not isinstance(value, list):
        raise TypeError("part positions must be a list")
    result: dict[tuple[str, str, str], dict[str, int]] = {}
    for item in value:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not isinstance(item[0], list)
            or len(item[0]) != 3
            or not all(isinstance(part, str) for part in item[0])
            or not isinstance(item[1], list)
        ):
            raise ValueError("invalid part position")
        entries: dict[str, int] = {}
        for part in item[1]:
            if (
                not isinstance(part, list)
                or len(part) != 2
                or not isinstance(part[0], str)
                or not isinstance(part[1], int)
            ):
                raise ValueError("invalid part position")
            entries[part[0]] = part[1]
        result[tuple(item[0])] = entries  # type: ignore[arg-type]
    return result


def _string_value(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _summary_metadata(cache_key: str) -> dict[str, str]:
    parts = cache_key.split("|")
    if parts and parts[0] == "YEAR" and len(parts) == 2:
        return {"tier": "Year", "arc_id": parts[1]}
    if parts and parts[0] == "EPISODE" and len(parts) == 4:
        return {
            "tier": "Episode",
            "arc_id": parts[1],
            "story_type": parts[2],
            "episode_name": parts[3],
        }
    if len(parts) == 4:
        return {
            "tier": "Part",
            "arc_id": parts[0],
            "story_type": parts[1],
            "episode_name": parts[2],
            "part_name": parts[3],
        }
    return {"tier": "Unknown"}


def validate_prepared_corpus(prepared_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    """Validate the catalog and both local indexes share one published revision."""
    directory = Path(prepared_dir).resolve()
    catalog_path = directory / "catalog.json"
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Prepared corpus catalog not found: {catalog_path}")
    try:
        catalog = _load_json_object(catalog_path, "Prepared corpus catalog")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Prepared corpus catalog is invalid: {catalog_path}") from exc
    revision = catalog.get("revision")
    if not isinstance(revision, str) or not revision:
        raise ValueError("Prepared corpus catalog has no revision")
    for name in ("source_records.db", "lexical_index.db"):
        if not (directory / name).is_file():
            raise FileNotFoundError(f"Prepared corpus artifact not found: {directory / name}")
    source_metadata = _read_metadata_read_only(directory / "source_records.db")
    lexical_metadata = _read_metadata_read_only(directory / "lexical_index.db")
    expected = {"revision": revision, "revision_inputs": catalog.get("revision_inputs")}
    if source_metadata != expected or lexical_metadata != expected:
        raise ValueError("Prepared corpus artifacts do not share the catalog revision")
    return directory, catalog


def resolve_publication_pointer(publication_dir: str | Path) -> tuple[Path, str]:
    root = Path(publication_dir).resolve()
    pointer_path = root / "current.json"
    try:
        pointer = _load_json_object(pointer_path, "Published corpus pointer")
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Published corpus pointer is invalid: {pointer_path}") from exc
    relative = pointer.get("path")
    revision = pointer.get("revision")
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or relative != Path(relative).as_posix()
        or not isinstance(revision, str)
    ):
        raise ValueError("Published corpus pointer has invalid path or revision")
    generation = (root / relative).resolve()
    if root not in generation.parents:
        raise ValueError("Published corpus pointer escapes its publication directory")
    return generation, revision


def _read_metadata_read_only(path: Path) -> dict[str, Any]:
    """Read corpus metadata without creating, migrating, or modifying a database."""
    if not path.is_file():
        raise FileNotFoundError(f"Prepared corpus artifact not found: {path}")
    uri = f"{path.as_uri()}?mode=ro"
    try:
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            rows = connection.execute("SELECT key, value_json FROM corpus_metadata").fetchall()
    except (OSError, sqlite3.Error, json.JSONDecodeError) as exc:
        raise ValueError(f"Prepared corpus metadata is unreadable: {path}") from exc
    try:
        return {str(key): json.loads(str(value)) for key, value in rows}
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Prepared corpus metadata is invalid: {path}") from exc
