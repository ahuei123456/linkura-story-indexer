"""Build local, generation-free corpus artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any
from uuid import uuid4

from ..corpus import (
    StoryCorpus,
    SummaryRecord,
    resolve_publication_pointer,
    validate_prepared_corpus,
)
from ..lexical import LexicalIndex, glossary_alias_groups
from ..models.story import StoryNode
from ..story_order import DEFAULT_STORY_ORDER_PATH, StoryOrder, load_story_order
from .chunker import (
    CHUNKER_VERSION,
    MAX_CHUNK_CHARS,
    MIN_USEFUL_CHARS,
    TARGET_CHUNK_CHARS,
    build_retrieval_chunks,
)
from .corpus import PreparedCorpus, prepare_corpus
from .manifest import (
    RAW_EVIDENCE_SCHEMA_VERSION,
    SUMMARY_CACHE_SCHEMA_VERSION,
    hash_text,
    stable_hash,
)
from .parser import PARSER_VERSION
from .source_store import SourceRecordStore

CATALOG_SCHEMA_VERSION = 7
SOURCE_SCHEMA_VERSION = RAW_EVIDENCE_SCHEMA_VERSION
LEXICAL_SCHEMA_VERSION = 1


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_snapshot(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256_bytes(path)
        for path in sorted(root.rglob("*.md"), key=lambda item: str(item))
    }


def _file_input(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"present": False}
    resolved = Path(path).resolve()
    return {
        "present": resolved.exists(),
        "path": str(resolved),
        "sha256": _sha256_bytes(resolved) if resolved.exists() else None,
    }


def _story_order_input(
    story_order: StoryOrder,
    path: str | Path | None,
    *,
    file_input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "file": file_input if file_input is not None else _file_input(path),
        "positions": {
            "chronological": sorted(
                (list(key), list(value)) for key, value in story_order.chronological_positions.items()
            ),
            "summary": sorted(
                (list(key), list(value)) for key, value in story_order.summary_positions.items()
            ),
            "parts": sorted(
                (list(key), sorted(value.items()))
                for key, value in story_order.part_positions.items()
            ),
        },
    }


@dataclass(frozen=True)
class CorpusRevision:
    revision: str
    inputs: dict[str, Any]


def _revision_inputs(
    prepared: PreparedCorpus,
    *,
    story_order_file: str | Path | None,
    glossary_file: str | Path | None,
    summary_cache_file: str | Path | None,
    source_hashes: dict[str, str],
    file_inputs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    file_inputs = file_inputs or {}
    return {
        "source_files": source_hashes,
        "parser_version": PARSER_VERSION,
        "source_schema_version": SOURCE_SCHEMA_VERSION,
        "catalog_schema_version": CATALOG_SCHEMA_VERSION,
        "story_order": _story_order_input(
            prepared.story_order,
            story_order_file,
            file_input=file_inputs.get("story_order"),
        ),
        "glossary": file_inputs.get("glossary", _file_input(glossary_file)),
        "chunker": {
            "version": CHUNKER_VERSION,
            "min_chars": MIN_USEFUL_CHARS,
            "target_chars": TARGET_CHUNK_CHARS,
            "max_chars": MAX_CHUNK_CHARS,
        },
        "lexical": {"schema_version": LEXICAL_SCHEMA_VERSION, "tokenizer_version": "1"},
        "summary_cache": file_inputs.get("summary_cache", _file_input(summary_cache_file)),
    }


def _corpus_revision(prepared: PreparedCorpus, **kwargs: Any) -> CorpusRevision:
    inputs = _revision_inputs(prepared, source_hashes=kwargs.pop("source_hashes"), **kwargs)
    digest = hashlib.sha256(_stable_json(inputs).encode("utf-8")).hexdigest()
    return CorpusRevision(digest, inputs)


def _node_id(node: StoryNode) -> str:
    metadata = node.metadata
    return node.metadata.chunk_id or f"chunk:{metadata.parent_part_id}:{metadata.scene_start}-{metadata.scene_end}"


def _lexical_text(node: StoryNode, glossary: dict[str, dict[str, str]]) -> str:
    metadata = node.metadata
    matching = [
        alias
        for group in glossary_alias_groups(glossary)
        if any(alias in node.text or alias in metadata.detected_speakers for alias in group)
        for alias in group
    ]
    header = " ".join(
        [
            metadata.arc_id,
            metadata.story_type,
            metadata.episode_name,
            metadata.part_name,
            *metadata.detected_speakers,
            *matching,
        ]
    )
    return f"{header}\n{node.text}"


def _summary_metadata(
    cache_key: str,
    record: SummaryRecord,
    *,
    source_status: str = "unverified",
    rolling_status: str = "unverified",
    aggregate_status: str = "unverified",
) -> dict[str, Any]:
    tier = record.metadata.get("tier", "Unknown")
    level = {"Year": 1, "Episode": 2, "Part": 3}.get(tier, 0)
    freshness = "stale" if "stale" in (source_status, rolling_status, aggregate_status) else "unverified"
    return {
        "summary_level": level,
        **record.metadata,
        "cache_key": cache_key,
        "provenance": "summary_cache",
        "freshness": freshness,
        "source_freshness": source_status,
        "rolling_freshness": rolling_status,
        "aggregate_freshness": aggregate_status,
        "freshness_scope": "tracked_source_files",
    }


def _summary_predecessor_key(
    cache_key: str,
    record: SummaryRecord,
    part_keys: list[str],
) -> str | None:
    """Return the preceding part summary from the parsed corpus sequence."""
    level = record.inputs.get("level")
    if level == "part":
        if cache_key not in part_keys:
            return None
        index = part_keys.index(cache_key)
        return part_keys[index - 1] if index else ""
    else:
        # Aggregate child hashes and their tier-specific ordering remain future work.
        return None


def _rolling_freshness(
    cache_key: str,
    record: SummaryRecord,
    records: dict[str, SummaryRecord],
    part_keys: list[str],
) -> str:
    """Validate only part-level predecessor hashes; aggregate tiers stay unknown."""
    if (
        record.schema_version != SUMMARY_CACHE_SCHEMA_VERSION
        or record.fingerprint != stable_hash(record.inputs)
        or "previous_summary_hash" not in record.inputs
    ):
        return "unverified"
    previous_hash = record.inputs["previous_summary_hash"]
    if not isinstance(previous_hash, str) or (
        previous_hash
        and (
            len(previous_hash) != 64
            or any(char not in "0123456789abcdef" for char in previous_hash)
        )
    ):
        return "unverified"
    predecessor_key = _summary_predecessor_key(cache_key, record, part_keys)
    if predecessor_key is None:
        return "unverified"
    if predecessor_key == "":
        return "rolling_context_current" if previous_hash == "" else "stale"
    predecessor = records.get(predecessor_key)
    if predecessor is None or not predecessor.summary:
        return "unverified"
    expected = _trim_previous_summary_context(predecessor.summary)
    expected_hash = hash_text(expected) if expected else ""
    return "rolling_context_current" if previous_hash == expected_hash else "stale"


def _aggregate_child_keys(prepared: PreparedCorpus, record: SummaryRecord) -> list[str] | None:
    """Return the producer's ordered child keys for one aggregate summary."""
    level = record.inputs.get("level")
    if level not in ("episode", "year"):
        return None
    parts = record.cache_key.split("|")
    if level == "episode":
        if len(parts) != 4 or parts[0] != "EPISODE":
            return None
        episode = tuple(parts[1:])
        keys = {
            (
                node.metadata.arc_id,
                node.metadata.story_type,
                node.metadata.episode_name,
                node.metadata.part_name,
            )
            for node in prepared.raw_nodes
            if (
                node.metadata.arc_id,
                node.metadata.story_type,
                node.metadata.episode_name,
            ) == episode
        }
        ordered = sorted(keys, key=lambda key: prepared.story_order.part_key(*key))
        return ["|".join(key) for key in ordered]
    if len(parts) != 2 or parts[0] != "YEAR":
        return None
    arc_id = parts[1]
    episodes = {
        (node.metadata.arc_id, node.metadata.story_type, node.metadata.episode_name)
        for node in prepared.raw_nodes
        if node.metadata.arc_id == arc_id
    }
    ordered_episodes = sorted(episodes, key=lambda key: prepared.story_order.summary_episode_key(*key))
    return ["EPISODE|" + "|".join(key) for key in ordered_episodes]


def _aggregate_freshness(
    prepared: PreparedCorpus,
    record: SummaryRecord,
    records: dict[str, SummaryRecord],
) -> str:
    """Validate an aggregate's direct child hashes and producer ordering only."""
    if (
        record.schema_version != SUMMARY_CACHE_SCHEMA_VERSION
        or record.fingerprint != stable_hash(record.inputs)
        or record.inputs.get("level") not in ("episode", "year")
    ):
        return "unverified"
    child_hashes = record.inputs.get("child_summary_hashes")
    child_keys = _aggregate_child_keys(prepared, record)
    if (
        child_keys is None
        or not child_keys
        or not isinstance(child_hashes, list)
        or len(child_hashes) != len(child_keys)
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
            for value in child_hashes
        )
    ):
        return "unverified"
    children = [records.get(key) for key in child_keys]
    if any(child is None or not child.summary for child in children):
        return "unverified"
    expected = [hash_text(child.summary) for child in children if child is not None]
    return "aggregate_children_current" if child_hashes == expected else "stale"


def _trim_previous_summary_context(summary: str) -> str | None:
    known_labels = {
        "Overview",
        "Key Events",
        "Character Developments",
        "Continuity Facts",
        "Important Terms",
        "Part Index",
        "Episode Arc",
        "Relationship / Unit Developments",
        "Episode Index",
        "Character Trajectories",
        "Unit / Club State",
    }
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in summary.splitlines():
        label = line.strip()[:-1] if line.strip().endswith(":") else ""
        if label in known_labels:
            current = label if label in {"Overview", "Continuity Facts"} else None
            if current is not None:
                sections.setdefault(label, [])
        elif current is not None:
            sections[current].append(line)
    parts = []
    for label in ("Overview", "Continuity Facts"):
        if sections.get(label):
            content = "\n".join(sections[label]).strip()
            if content:
                parts.append(f"{label}:\n{content}")
    return "\n\n".join(parts) if parts else None


def _summary_freshness(
    record: SummaryRecord, source_hashes: dict[str, str], story_root: Path
) -> str:
    """Compare listed raw-file hashes; this does not establish summary freshness."""
    if record.schema_version != SUMMARY_CACHE_SCHEMA_VERSION or record.fingerprint != stable_hash(record.inputs):
        return "unverified"
    tracked = record.inputs.get("source_file_hashes")
    if not isinstance(tracked, dict) or not tracked or not all(
        isinstance(path, str) and bool(path)
        and isinstance(digest, str) and len(digest) == 64
        and all(char in "0123456789abcdef" for char in digest)
        for path, digest in tracked.items()
    ):
        return "unverified"
    normalized = {item.replace("\\", "/"): digest for item, digest in source_hashes.items()}
    mapped = {}
    for path, digest in tracked.items():
        key = _source_dependency_key(path, story_root)
        if key is None:
            return "unverified"
        if key in mapped and mapped[key] != digest:
            return "unverified"
        mapped[key] = digest
    return "tracked_sources_current" if all(normalized.get(key) == digest for key, digest in mapped.items()) else "stale"


def _part_summary_keys(prepared: PreparedCorpus) -> list[str]:
    parts = {
        (
            node.metadata.arc_id,
            node.metadata.story_type,
            node.metadata.episode_name,
            node.metadata.part_name,
        )
        for node in prepared.raw_nodes
    }
    ordered = sorted(
        parts,
        key=lambda key: (
            prepared.story_order.summary_episode_key(*key[:3]),
            prepared.story_order.part_key(*key),
        ),
    )
    return ["|".join(key) for key in ordered]


def _source_dependency_key(path: str, story_root: Path) -> str | None:
    """Map cache paths without reading dependencies outside the configured root."""
    root = story_root.resolve()
    candidate = Path(path.replace("\\", "/"))
    # A Windows absolute path cannot be interpreted on a different platform.
    if PureWindowsPath(path).drive and not candidate.is_absolute():
        return None
    try:
        return candidate.resolve().relative_to(root).as_posix()
    except ValueError:
        if candidate.is_absolute() or candidate.anchor or ".." in candidate.parts:
            return None
    # Support root-relative records and the common '<root-name>/...' spelling.
    if candidate.parts and candidate.parts[0] == root.name:
        candidate = Path(*candidate.parts[1:])
    try:
        return (root / candidate).resolve().relative_to(root).as_posix()
    except ValueError:
        return None


def _write_catalog(
    path: Path,
    prepared: PreparedCorpus,
    corpus: StoryCorpus,
    hashes: dict[str, str],
    revision: CorpusRevision,
) -> None:
    summaries = corpus.iter_summaries()
    summary_records = {summary.cache_key: summary for summary in summaries}
    part_keys = _part_summary_keys(prepared)
    catalog = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "revision": revision.revision,
        "revision_inputs": revision.inputs,
        # Keep the exact prepared metadata beside the revision so readers do
        # not depend on mutable or deleted input files.
        "story_order": revision.inputs["story_order"],
        "glossary": corpus.glossary,
        "source_root": str(prepared.story_root.resolve()),
        "source_file_hashes": hashes,
        "source_file_count": len(prepared.files),
        "scene_count": len(prepared.raw_nodes),
        "summary_count": len(summaries),
        "summaries": [
            {
                "cache_key": summary.cache_key,
                "summary": summary.summary,
                "fingerprint": summary.fingerprint,
                "schema_version": summary.schema_version,
                "inputs": summary.inputs,
                "metadata": _summary_metadata(
                    summary.cache_key,
                    summary,
                    source_status=_summary_freshness(summary, hashes, prepared.story_root),
                    rolling_status=_rolling_freshness(
                        summary.cache_key, summary, summary_records, part_keys
                    ),
                    aggregate_status=_aggregate_freshness(prepared, summary, summary_records),
                ),
            }
            for summary in summaries
        ],
    }
    path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_artifacts(
    stage: Path,
    prepared: PreparedCorpus,
    *,
    glossary_file: str | Path | None,
    summary_cache_file: str | Path | None,
    revision: CorpusRevision,
    source_hashes: dict[str, str],
) -> None:
    chunks = build_retrieval_chunks(prepared.raw_nodes)
    source_store = SourceRecordStore(stage / "source_records.db")
    source_store.replace_all(prepared.raw_nodes, chunks)
    source_store.set_metadata({"revision": revision.revision, "revision_inputs": revision.inputs})
    corpus = StoryCorpus.from_paths(
        source_db=stage / "source_records.db",
        glossary_file=glossary_file,
        summary_cache_file=summary_cache_file,
    )
    corpus.story_order = prepared.story_order
    lexical = LexicalIndex(stage / "lexical_index.db")
    ids = [_node_id(chunk) for chunk in chunks]
    lexical.upsert_records(
        ids=ids,
        documents=[chunk.text for chunk in chunks],
        metadatas=[chunk.metadata.model_dump() | {"summary_level": chunk.summary_level} for chunk in chunks],
        search_texts=[_lexical_text(chunk, corpus.glossary) for chunk in chunks],
    )
    summaries = corpus.iter_summaries()
    summary_records = {summary.cache_key: summary for summary in summaries}
    part_keys = _part_summary_keys(prepared)
    if summaries:
        summary_ids = [f"summary:{summary.cache_key}" for summary in summaries]
        lexical.upsert_records(
            ids=summary_ids,
            documents=[summary.summary for summary in summaries],
            metadatas=[
                    _summary_metadata(
                        summary.cache_key,
                        summary,
                        source_status=_summary_freshness(summary, source_hashes, prepared.story_root),
                        rolling_status=_rolling_freshness(
                            summary.cache_key, summary, summary_records, part_keys
                        ),
                        aggregate_status=_aggregate_freshness(prepared, summary, summary_records),
                    )
                for summary in summaries
            ],
            search_texts=[summary.summary for summary in summaries],
        )
    lexical.set_metadata({"revision": revision.revision, "revision_inputs": revision.inputs})
    _write_catalog(stage / "catalog.json", prepared, corpus, source_hashes, revision)


def prepare_corpus_artifacts(
    story_root: str | Path,
    output_dir: str | Path,
    *,
    story_order_file: str | Path | None = None,
    glossary_file: str | Path | None = None,
    summary_cache_file: str | Path | None = None,
    replace: bool = False,
) -> Path:
    """Prepare artifacts, optionally publishing a new immutable revision."""
    destination = Path(output_dir).resolve()
    if destination.exists() and not replace:
        raise FileExistsError(f"Output directory already exists: {destination}")
    if replace:
        if not destination.is_dir():
            raise NotADirectoryError(f"Replacement target is not a directory: {destination}")
        validate_prepared_corpus(destination)
        if (destination / "current.json").exists():
            generation, pointer_revision = resolve_publication_pointer(destination)
            try:
                _, catalog = validate_prepared_corpus(generation)
            except (FileNotFoundError, ValueError) as exc:
                raise ValueError("Published corpus pointer generation is invalid") from exc
            if catalog["revision"] != pointer_revision:
                raise ValueError("Published corpus pointer revision does not match its generation")
    root = Path(story_root)
    order_input_path = story_order_file or DEFAULT_STORY_ORDER_PATH
    initial_sources = _source_snapshot(root)
    initial_file_inputs = {
        "story_order": _file_input(order_input_path),
        "glossary": _file_input(glossary_file),
        "summary_cache": _file_input(summary_cache_file),
    }
    story_order = load_story_order(order_input_path, story_root=root)
    prepared = prepare_corpus(root, story_order=story_order)
    revision = _corpus_revision(
        prepared,
        story_order_file=order_input_path,
        glossary_file=glossary_file,
        summary_cache_file=summary_cache_file,
        source_hashes=initial_sources,
        file_inputs=initial_file_inputs,
    )
    parent = destination.parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    # Inherit the output parent's ACL; tempfile's private Windows ACL can make
    # a published directory unreadable to other local tools.
    stage = parent / f".{destination.name}-{uuid4().hex}"
    stage.mkdir()
    try:
        _build_artifacts(
            stage,
            prepared,
            glossary_file=glossary_file,
            summary_cache_file=summary_cache_file,
            revision=revision,
            source_hashes=initial_sources,
        )
        current_sources = _source_snapshot(root)
        current_inputs = _revision_inputs(
            prepared,
            story_order_file=order_input_path,
            glossary_file=glossary_file,
            summary_cache_file=summary_cache_file,
            source_hashes=current_sources,
        )
        if current_inputs != revision.inputs:
            raise ValueError("Corpus inputs changed during preparation; nothing was published")
        if not replace:
            stage.rename(destination)
            return destination
        revisions = destination / ".revisions"
        resolved_revisions = revisions.resolve()
        if destination not in resolved_revisions.parents:
            raise ValueError("Replacement revisions directory escapes its output directory")
        revisions.mkdir(exist_ok=True)
        generation = revisions / f"{revision.revision}-{uuid4().hex}"
        stage.rename(generation)
        pointer = {"path": generation.relative_to(destination).as_posix(), "revision": revision.revision}
        pointer_stage = destination / f".current-{uuid4().hex}.json.tmp"
        try:
            pointer_stage.write_text(json.dumps(pointer, sort_keys=True), encoding="utf-8")
            pointer_stage.replace(destination / "current.json")
        finally:
            pointer_stage.unlink(missing_ok=True)
    except BaseException:
        resolved_parent = parent.resolve()
        if stage.parent != resolved_parent:
            raise RuntimeError(f"Refusing to clean staging path outside output parent: {stage}")
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return destination
