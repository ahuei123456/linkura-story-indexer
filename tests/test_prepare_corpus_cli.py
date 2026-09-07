import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from linkura_story_indexer import cli, database
from linkura_story_indexer.corpus import StoryCorpus, SummaryRecord
from linkura_story_indexer.indexer import preparation, summarizer
from linkura_story_indexer.indexer.manifest import hash_text, stable_hash
from linkura_story_indexer.indexer.preparation import (
    _aggregate_freshness,
    _rolling_freshness,
    _trim_previous_summary_context,
)
from linkura_story_indexer.indexer.source_store import SourceRecordStore
from linkura_story_indexer.indexer.summarizer import trim_previous_summary_context
from linkura_story_indexer.lexical import LexicalIndex


def test_aggregate_freshness_tracks_ordered_direct_child_hashes(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    for part, text in (("1", "first"), ("2", "second")):
        path = root / "103" / "episode" / f"{part}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    order = preparation.load_story_order(_order_file(tmp_path), story_root=root)
    prepared = preparation.prepare_corpus(root, story_order=order)
    first = SummaryRecord(
        cache_key="103|Other|episode|1",
        summary="part one",
        fingerprint="",
        schema_version="1",
        inputs={"level": "part"},
        metadata={},
    )
    second = SummaryRecord(
        cache_key="103|Other|episode|2",
        summary="part two",
        fingerprint="",
        schema_version="1",
        inputs={"level": "part"},
        metadata={},
    )
    inputs = {"level": "episode", "child_summary_hashes": [hash_text(first.summary), hash_text(second.summary)]}
    aggregate = SummaryRecord(
        cache_key="EPISODE|103|Other|episode",
        summary="episode",
        fingerprint=stable_hash(inputs),
        schema_version="1",
        inputs=inputs,
        metadata={},
    )
    records = {first.cache_key: first, second.cache_key: second, aggregate.cache_key: aggregate}
    assert _aggregate_freshness(prepared, aggregate, records) == "aggregate_children_current"
    records[first.cache_key] = replace(first, summary="edited")
    assert _aggregate_freshness(prepared, aggregate, records) == "stale"
    malformed_inputs = {"level": ["episode"], "child_summary_hashes": []}
    malformed = replace(aggregate, fingerprint=stable_hash(malformed_inputs), inputs=malformed_inputs)
    assert _aggregate_freshness(prepared, malformed, records) == "unverified"
    empty_inputs = {"level": "episode", "child_summary_hashes": []}
    empty = replace(aggregate, fingerprint=stable_hash(empty_inputs), inputs=empty_inputs)
    assert _aggregate_freshness(prepared, empty, records) == "unverified"
    episode_inputs = {"level": "episode", "child_summary_hashes": []}
    episode = replace(aggregate, summary="episode", inputs=episode_inputs, fingerprint=stable_hash(episode_inputs))
    records[episode.cache_key] = episode
    year_inputs = {"level": "year", "child_summary_hashes": [hash_text(episode.summary)]}
    year = replace(
        aggregate,
        cache_key="YEAR|103",
        summary="year",
        inputs=year_inputs,
        fingerprint=stable_hash(year_inputs),
    )
    records[year.cache_key] = year
    assert _aggregate_freshness(prepared, year, records) == "aggregate_children_current"


def test_aggregate_freshness_is_unverified_when_a_current_child_is_missing(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    for part in ("1", "2"):
        path = root / "103" / "episode" / f"{part}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(part, encoding="utf-8")
    order = preparation.load_story_order(_order_file(tmp_path), story_root=root)
    prepared = preparation.prepare_corpus(root, story_order=order)
    inputs = {"level": "episode", "child_summary_hashes": ["0" * 64, "1" * 64]}
    aggregate = SummaryRecord(
        cache_key="EPISODE|103|Other|episode",
        summary="episode",
        fingerprint=stable_hash(inputs),
        schema_version="1",
        inputs=inputs,
        metadata={},
    )
    assert _aggregate_freshness(prepared, aggregate, {aggregate.cache_key: aggregate}) == "unverified"


@pytest.mark.parametrize(
    ("case", "episode_a_status"),
    [("current", "aggregate_children_current"), ("stale", "stale"), ("missing", "unverified")],
)
def test_aggregate_freshness_agrees_across_prepared_readers_and_pinned_order(
    tmp_path: Path, case: str, episode_a_status: str
) -> None:
    root = tmp_path / "corpus"
    main_episode = "\u7b2c1\u8a71 Main"
    for story_type, episode, parts in (
        ("Main", main_episode, ("1", "2")),
        ("Other", "Episode B", ("1",)),
    ):
        for part in parts:
            path = root / "103" / episode / f"{part}.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{story_type}-{episode}-{part}", encoding="utf-8")
    order_file = tmp_path / "order.yaml"
    order_file.write_text(
        """
chronological_order:
  - story_type: Main
    arcs: ["103"]
  - story_type: Other
    arcs: ["103"]
summary_order:
  - story_type: Other
    arcs: ["103"]
  - story_type: Main
    arcs: ["103"]
part_order_overrides:
  - story_type: Main
    arc_id: "103"
    episode_name: "\u7b2c1\u8a71 Main"
    parts: ["2", "1"]
""",
        encoding="utf-8",
    )
    part_one = f"103|Main|{main_episode}|1"
    part_two = f"103|Main|{main_episode}|2"
    other_part = "103|Other|Episode B|1"
    episode_a = f"EPISODE|103|Main|{main_episode}"
    episode_b = "EPISODE|103|Other|Episode B"
    part_records = {
        key: SummaryRecord(
            cache_key=key,
            summary=f"summary:{key}",
            fingerprint="",
            schema_version="1",
            inputs={"level": "part"},
            metadata={},
        )
        for key in (part_one, part_two, other_part)
    }
    episode_a_inputs = {
        "level": "episode",
        "child_summary_hashes": [
            hash_text(part_records[part_two].summary),
            hash_text(part_records[part_one].summary),
        ],
    }
    episode_b_inputs = {
        "level": "episode",
        "child_summary_hashes": [hash_text(part_records[other_part].summary)],
    }
    episode_records = {
        episode_a: SummaryRecord(
            cache_key=episode_a,
            summary="summary:episode-a",
            fingerprint=stable_hash(episode_a_inputs),
            schema_version="1",
            inputs=episode_a_inputs,
            metadata={},
        ),
        episode_b: SummaryRecord(
            cache_key=episode_b,
            summary="summary:episode-b",
            fingerprint=stable_hash(episode_b_inputs),
            schema_version="1",
            inputs=episode_b_inputs,
            metadata={},
        ),
    }
    year_inputs = {
        "level": "year",
        "child_summary_hashes": [
            hash_text(episode_records[episode_b].summary),
            hash_text(episode_records[episode_a].summary),
        ],
    }
    records = {
        **part_records,
        **episode_records,
        "YEAR|103": SummaryRecord(
            cache_key="YEAR|103",
            summary="summary:year",
            fingerprint=stable_hash(year_inputs),
            schema_version="1",
            inputs=year_inputs,
            metadata={},
        ),
    }
    if case == "stale":
        records[part_two] = replace(records[part_two], summary="edited part summary")
    elif case == "missing":
        del records[part_two]
    cache = tmp_path / "summaries.json"
    cache.write_text(
        json.dumps(
            {
                key: {
                    "summary": record.summary,
                    "fingerprint": record.fingerprint,
                    "schema_version": record.schema_version,
                    "inputs": record.inputs,
                }
                for key, record in records.items()
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "prepared"
    preparation.prepare_corpus_artifacts(root, output, story_order_file=order_file, summary_cache_file=cache)
    catalog = json.loads((output / "catalog.json").read_text(encoding="utf-8"))
    catalog_status = {
        record["cache_key"]: record["metadata"]["aggregate_freshness"]
        for record in catalog["summaries"]
        if record["cache_key"] in {episode_a, episode_b, "YEAR|103"}
    }
    lexical = LexicalIndex(output / "lexical_index.db")
    lexical_status = {
        metadata["cache_key"]: metadata["aggregate_freshness"]
        for _, metadata in lexical.search("summary", n_results=10)
        if metadata.get("cache_key") in {episode_a, episode_b, "YEAR|103"}
    }
    reader = StoryCorpus.from_prepared_dir(output)
    reader_status = {
        key: summary.metadata["aggregate_freshness"]
        for key in (episode_a, episode_b, "YEAR|103")
        if (summary := reader.get_summary(key)) is not None
    }
    expected = {
        episode_a: episode_a_status,
        episode_b: "aggregate_children_current",
        "YEAR|103": "aggregate_children_current",
    }
    assert catalog_status == lexical_status == reader_status == expected
    freshness = {
        record["cache_key"]: record["metadata"]["freshness"]
        for record in catalog["summaries"]
        if record["cache_key"] in expected
    }
    assert freshness[episode_a] == ("stale" if case == "stale" else "unverified")
    assert freshness[episode_b] == "unverified"
    assert freshness["YEAR|103"] == "unverified"


def test_aggregate_freshness_stays_pinned_across_replacement(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    first = root / "103" / "episode" / "1.md"
    second = root / "103" / "episode" / "2.md"
    first.parent.mkdir(parents=True)
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    first_key = "103|Other|episode|1"
    second_key = "103|Other|episode|2"
    episode_key = "EPISODE|103|Other|episode"
    part_records = {
        first_key: "first summary",
        second_key: "second summary",
    }
    cache = tmp_path / "cache.json"

    def write_cache(*, refresh_aggregate: bool = True) -> None:
        episode_inputs = {
            "level": "episode",
            "child_summary_hashes": [
                hash_text(part_records[first_key]),
                hash_text(part_records[second_key]),
            ],
        }
        if not refresh_aggregate:
            episode_inputs["child_summary_hashes"][1] = hash_text("second summary")
        cache.write_text(
            json.dumps(
                {
                    key: {
                        "summary": summary,
                        "schema_version": "1",
                        "inputs": {"level": "part"},
                        "fingerprint": stable_hash({"level": "part"}),
                    }
                    for key, summary in part_records.items()
                }
                | {
                    episode_key: {
                        "summary": "episode summary",
                        "schema_version": "1",
                        "inputs": episode_inputs,
                        "fingerprint": stable_hash(episode_inputs),
                    }
                }
            ),
            encoding="utf-8",
        )

    order = _order_file(tmp_path)
    output = tmp_path / "published"
    write_cache()
    preparation.prepare_corpus_artifacts(
        root, output, story_order_file=order, summary_cache_file=cache
    )
    preparation.prepare_corpus_artifacts(
        root, output, story_order_file=order, summary_cache_file=cache, replace=True
    )
    pinned = StoryCorpus.from_publication(output)

    def aggregate_status(directory: Path) -> tuple[str, str, str]:
        catalog = json.loads((directory / "catalog.json").read_text(encoding="utf-8"))
        catalog_status = next(
            item["metadata"]["aggregate_freshness"]
            for item in catalog["summaries"]
            if item["cache_key"] == episode_key
        )
        lexical_status = LexicalIndex(directory / "lexical_index.db").search(
            "episode summary", n_results=1, where={"cache_key": episode_key}
        )[0][1]["aggregate_freshness"]
        reader_summary = StoryCorpus.from_prepared_dir(directory).get_summary(episode_key)
        assert reader_summary is not None
        return catalog_status, lexical_status, reader_summary.metadata["aggregate_freshness"]

    initial_generation = output / json.loads((output / "current.json").read_text(encoding="utf-8"))["path"]
    assert aggregate_status(initial_generation) == (
        "aggregate_children_current",
        "aggregate_children_current",
        "aggregate_children_current",
    )

    part_records[second_key] = "edited second summary"
    write_cache(refresh_aggregate=False)
    preparation.prepare_corpus_artifacts(
        root, output, story_order_file=order, summary_cache_file=cache, replace=True
    )
    pinned_summary = pinned.get_summary(episode_key)
    assert pinned_summary is not None
    assert pinned_summary.metadata["aggregate_freshness"] == "aggregate_children_current"
    current = StoryCorpus.from_publication(output)
    current_summary = current.get_summary(episode_key)
    assert current_summary is not None
    assert current_summary.metadata["aggregate_freshness"] == "stale"
    current_generation = output / json.loads((output / "current.json").read_text(encoding="utf-8"))["path"]
    assert aggregate_status(current_generation) == ("stale", "stale", "stale")


@pytest.mark.parametrize(
    ("mutate", "schema", "fingerprint", "expected"),
    [
        (lambda inputs: {**inputs, "child_summary_hashes": ["bad"]}, "1", None, "unverified"),
        (lambda inputs: {**inputs, "child_summary_hashes": []}, "1", None, "unverified"),
        (lambda inputs: inputs, "1", "invalid", "unverified"),
        (lambda inputs: inputs, "unsupported", None, "unverified"),
    ],
)
def test_aggregate_freshness_rejects_malformed_or_missing_dependencies(
    tmp_path: Path, mutate, schema: str, fingerprint: str | None, expected: str
) -> None:
    root = tmp_path / "corpus"
    source = root / "103" / "Other" / "episode" / "1.md"
    source.parent.mkdir(parents=True)
    source.write_text("source", encoding="utf-8")
    order = _order_file(tmp_path)
    child_key = "103|Other|episode|1"
    child_summary = "child summary"
    base_inputs = {"level": "episode", "child_summary_hashes": [hash_text(child_summary)]}
    inputs = mutate(base_inputs)
    aggregate = {
        "summary": "aggregate",
        "schema_version": schema,
        "inputs": inputs,
        "fingerprint": fingerprint or stable_hash(inputs),
    }
    cache = tmp_path / "summaries.json"
    cache.write_text(
        json.dumps(
            {
                child_key: {
                    "summary": child_summary,
                    "schema_version": "1",
                    "inputs": {"level": "part"},
                    "fingerprint": stable_hash({"level": "part"}),
                },
                "EPISODE|103|Other|episode": aggregate,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "prepared"
    preparation.prepare_corpus_artifacts(root, output, story_order_file=order, summary_cache_file=cache)
    summaries = json.loads((output / "catalog.json").read_text(encoding="utf-8"))["summaries"]
    metadata = next(
        record["metadata"]
        for record in summaries
        if record["cache_key"] == "EPISODE|103|Other|episode"
    )
    assert metadata["aggregate_freshness"] == expected
def _order_file(path: Path) -> Path:
    result = path / "order.yaml"
    result.write_text(
        """
chronological_order:
  - story_type: Other
    arcs: [\"103\"]
summary_order:
  - story_type: Other
    arcs: [\"103\"]
""",
        encoding="utf-8",
    )
    return result


def test_prepare_corpus_cli_builds_local_artifacts_without_external_services(
    tmp_path: Path, monkeypatch
) -> None:
    story_root = tmp_path / "corpus"
    story_file = story_root / "103" / "Main" / "Episode" / "1.md"
    story_file.parent.mkdir(parents=True)
    story_file.write_text("花帆: Hello local corpus", encoding="utf-8")
    glossary = tmp_path / "glossary.json"
    glossary.write_text(json.dumps({"characters": {"花帆": "Kaho"}}), encoding="utf-8")
    summaries = tmp_path / "summaries.json"
    summaries.write_text(
        json.dumps(
            {
                "103|Main|Episode|1": {
                    "summary": "A cached summary",
                    "fingerprint": "source-fingerprint",
                    "schema_version": "3",
                    "inputs": {"location": "Episode/1"},
                }
            }
        ),
        encoding="utf-8",
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("prepare-corpus must not construct external services")

    monkeypatch.setattr(cli, "initialize_ingest_settings", forbidden)
    monkeypatch.setattr(cli, "initialize_generation_settings", forbidden)
    monkeypatch.setattr(cli, "HierarchicalSummarizer", forbidden)
    monkeypatch.setattr(cli, "embed_texts", forbidden)
    monkeypatch.setattr(cli, "StoryQueryEngine", forbidden)
    monkeypatch.setattr("linkura_story_indexer.database.create_generation_model", forbidden)
    monkeypatch.setattr("linkura_story_indexer.database.create_generation_text_agent", forbidden)
    monkeypatch.setattr("linkura_story_indexer.database.genai.Client", forbidden)
    monkeypatch.setattr("linkura_story_indexer.database.chromadb.PersistentClient", forbidden)

    output = tmp_path / "prepared"
    result = CliRunner().invoke(
        cli.app,
        [
            "prepare-corpus",
            "--story-dir",
            str(story_root),
            "--output-dir",
            str(output),
            "--story-order",
            str(_order_file(tmp_path)),
            "--glossary",
            str(glossary),
            "--summaries",
            str(summaries),
        ],
    )

    assert result.exit_code == 0, f"{result.stdout}\n{result.exception!r}"
    assert (output / "catalog.json").exists()
    assert len(SourceRecordStore(output / "source_records.db").iter_scenes()) == 1
    lexical = LexicalIndex(output / "lexical_index.db")
    chunk_id = next(record_id for record_id in lexical.list_ids() if record_id.startswith("chunk:"))
    assert lexical.list_ids() == {chunk_id, "summary:103|Main|Episode|1"}
    assert lexical.search("Kaho", n_results=1)[0][1]["chunk_id"] == chunk_id
    catalog = json.loads((output / "catalog.json").read_text(encoding="utf-8"))
    assert catalog["summary_count"] == 1
    assert catalog["summaries"][0]["fingerprint"] == "source-fingerprint"
    assert catalog["summaries"][0]["metadata"]["freshness"] == "unverified"


def test_prepare_summary_freshness_tracks_source_edits_and_deletions(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    source = root / "103" / "episode" / "1.md"
    source.parent.mkdir(parents=True)
    source.write_text("stable", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    cache = tmp_path / "summaries.json"
    cache.write_text(
        json.dumps(
            {
                "103|Other|episode|1": {
                    "summary": "cached",
                    "schema_version": "1",
                    "inputs": {"source_file_hashes": {"103/episode/1.md": digest}},
                    "fingerprint": stable_hash(
                        {"source_file_hashes": {"103/episode/1.md": digest}}
                    ),
                }
            }
        ),
        encoding="utf-8",
    )
    order = _order_file(tmp_path)
    output = tmp_path / "prepared"
    preparation.prepare_corpus_artifacts(root, output, story_order_file=order, summary_cache_file=cache)
    catalog = json.loads((output / "catalog.json").read_text(encoding="utf-8"))
    assert catalog["summaries"][0]["metadata"]["source_freshness"] == "tracked_sources_current"
    source.write_text("changed", encoding="utf-8")
    changed = tmp_path / "changed"
    preparation.prepare_corpus_artifacts(root, changed, story_order_file=order, summary_cache_file=cache)
    changed_catalog = json.loads((changed / "catalog.json").read_text(encoding="utf-8"))
    assert changed_catalog["summaries"][0]["metadata"]["freshness"] == "stale"
    assert changed_catalog["summaries"][0]["metadata"]["source_freshness"] == "stale"
    source.unlink()
    (source.parent / "2.md").write_text("remaining raw scene", encoding="utf-8")
    deleted = tmp_path / "deleted"
    preparation.prepare_corpus_artifacts(root, deleted, story_order_file=order, summary_cache_file=cache)
    reader = StoryCorpus.from_prepared_dir(deleted)
    summary = reader.get_summary("103|Other|episode|1")
    assert summary is not None
    assert summary.metadata["source_freshness"] == "stale"
    assert reader.iter_scenes()[0]["text"] == "remaining raw scene"


@pytest.mark.parametrize("path_style", ["relative", "root_prefixed", "absolute", "backslash", "cwd_relative"])
def test_summary_source_path_styles_are_mapped_portably(tmp_path: Path, monkeypatch, path_style: str) -> None:
    root = tmp_path / "data" / "corpus"
    source = root / "103" / "episode" / "nested" / "1.md"
    source.parent.mkdir(parents=True)
    source.write_text("stable", encoding="utf-8")
    relative = "103/episode/nested/1.md"
    cache_path = {"relative": relative, "root_prefixed": f"corpus/{relative}", "absolute": str(source), "backslash": relative.replace("/", "\\"), "cwd_relative": f"data/corpus/{relative}"}[path_style]
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    inputs = {"source_file_hashes": {cache_path: digest}}
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"103|Other|episode|1": {"summary": "cached", "schema_version": "1", "inputs": inputs, "fingerprint": stable_hash(inputs)}}), encoding="utf-8")
    if path_style in {"root_prefixed", "cwd_relative"}:
        monkeypatch.chdir(tmp_path)
    output = tmp_path / f"prepared-{path_style}"
    preparation.prepare_corpus_artifacts(root, output, story_order_file=_order_file(tmp_path), summary_cache_file=cache)
    catalog = json.loads((output / "catalog.json").read_text(encoding="utf-8"))
    assert catalog["summaries"][0]["metadata"]["source_freshness"] == "tracked_sources_current"


@pytest.mark.parametrize("inputs, schema", [({}, "1"), ({"source_file_hashes": {"x.md": "x"}}, "2"), ({"source_file_hashes": {"x.md": "x"}}, "1")])
def test_summary_integrity_or_missing_dependencies_stay_unverified(tmp_path: Path, inputs, schema: str) -> None:
    root = tmp_path / "corpus"
    source = root / "103" / "Main" / "episode" / "1.md"
    source.parent.mkdir(parents=True)
    source.write_text("stable", encoding="utf-8")
    fingerprint = "bad" if inputs and schema == "1" else stable_hash(inputs)
    record = {"summary": "cached", "schema_version": schema, "inputs": inputs, "fingerprint": fingerprint}
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"103|Other|episode|1": record}), encoding="utf-8")
    output = tmp_path / "prepared"
    preparation.prepare_corpus_artifacts(root, output, story_order_file=_order_file(tmp_path), summary_cache_file=cache)
    metadata = json.loads((output / "catalog.json").read_text(encoding="utf-8"))["summaries"][0]["metadata"]
    assert metadata["freshness"] == "unverified"
    assert metadata["source_freshness"] == "unverified"


def test_rolling_dependency_remains_overall_unverified_and_metadata_agrees(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    source = root / "103" / "episode" / "1.md"
    source.parent.mkdir(parents=True)
    source.write_text("stable", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    inputs = {"source_file_hashes": {"103/episode/1.md": digest}, "previous_summary_hash": "prior"}
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"103|Other|episode|1": {"summary": "cached", "schema_version": "1", "inputs": inputs, "fingerprint": stable_hash(inputs)}}), encoding="utf-8")
    output = tmp_path / "prepared"
    preparation.prepare_corpus_artifacts(root, output, story_order_file=_order_file(tmp_path), summary_cache_file=cache)
    catalog_record = json.loads((output / "catalog.json").read_text(encoding="utf-8"))["summaries"][0]["metadata"]
    lexical = LexicalIndex(output / "lexical_index.db")
    lexical_record = lexical.search("cached", n_results=1)[0][1]
    corpus_record = StoryCorpus.from_prepared_dir(output).get_summary("103|Other|episode|1")
    assert corpus_record is not None
    assert catalog_record["freshness"] == lexical_record["freshness"] == corpus_record.metadata["freshness"] == "unverified"
    assert catalog_record["source_freshness"] == lexical_record["source_freshness"] == corpus_record.metadata["source_freshness"] == "tracked_sources_current"


def test_rolling_predecessor_freshness_is_scoped_and_tracks_trimmed_context(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    first = root / "103" / "episode" / "1.md"
    second = root / "103" / "episode" / "2.md"
    first.parent.mkdir(parents=True)
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    first_summary = "Overview:\nFirst overview\n\nContinuity Facts:\nFirst continuity\n\nIgnored: nope"
    first_context = trim_previous_summary_context(first_summary)
    assert first_context is not None
    first_inputs = {
        "level": "part",
        "source_file_hashes": {"103/episode/1.md": hashlib.sha256(first.read_bytes()).hexdigest()},
        "previous_summary_hash": "",
    }
    second_inputs = {
        "level": "part",
        "source_file_hashes": {"103/episode/2.md": hashlib.sha256(second.read_bytes()).hexdigest()},
        "previous_summary_hash": hash_text(first_context),
    }
    cache = tmp_path / "cache.json"
    cache.write_text(
        json.dumps(
            {
                "103|Other|episode|1": {
                    "summary": first_summary,
                    "schema_version": "1",
                    "inputs": first_inputs,
                    "fingerprint": stable_hash(first_inputs),
                },
                "103|Other|episode|2": {
                    "summary": "Overview:\nSecond overview",
                    "schema_version": "1",
                    "inputs": second_inputs,
                    "fingerprint": stable_hash(second_inputs),
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "prepared"
    preparation.prepare_corpus_artifacts(root, output, story_order_file=_order_file(tmp_path), summary_cache_file=cache)
    catalog = json.loads((output / "catalog.json").read_text(encoding="utf-8"))
    metadata = {item["cache_key"]: item["metadata"] for item in catalog["summaries"]}
    assert metadata["103|Other|episode|1"]["rolling_freshness"] == "rolling_context_current"
    assert metadata["103|Other|episode|2"]["rolling_freshness"] == "rolling_context_current"
    assert metadata["103|Other|episode|2"]["freshness"] == "unverified"

    second_inputs["previous_summary_hash"] = "0" * 64
    cache.write_text(
        json.dumps({
            "103|Other|episode|1": {
                "summary": first_summary,
                "schema_version": "1",
                "inputs": first_inputs,
                "fingerprint": stable_hash(first_inputs),
            },
            "103|Other|episode|2": {
                "summary": "Overview:\nSecond overview",
                "schema_version": "1",
                "inputs": second_inputs,
                "fingerprint": stable_hash(second_inputs),
            },
        }),
        encoding="utf-8",
    )
    stale = tmp_path / "stale"
    preparation.prepare_corpus_artifacts(root, stale, story_order_file=_order_file(tmp_path), summary_cache_file=cache)
    stale_catalog = json.loads((stale / "catalog.json").read_text(encoding="utf-8"))
    stale_metadata = {item["cache_key"]: item["metadata"] for item in stale_catalog["summaries"]}
    assert stale_metadata["103|Other|episode|2"]["rolling_freshness"] == "stale"


def test_rolling_freshness_agrees_across_lexical_and_pinned_publication_readers(
    tmp_path: Path,
) -> None:
    root = tmp_path / "corpus"
    first = root / "103" / "episode" / "1.md"
    second = root / "103" / "episode" / "2.md"
    first.parent.mkdir(parents=True)
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    first_summary = "Overview:\nFirst overview\n\nContinuity Facts:\nFirst continuity"
    first_context = trim_previous_summary_context(first_summary)
    assert first_context is not None
    first_inputs = {
        "level": "part",
        "source_file_hashes": {"103/episode/1.md": hashlib.sha256(first.read_bytes()).hexdigest()},
        "previous_summary_hash": "",
    }
    second_inputs = {
        "level": "part",
        "source_file_hashes": {"103/episode/2.md": hashlib.sha256(second.read_bytes()).hexdigest()},
        "previous_summary_hash": hash_text(first_context),
    }
    cache = tmp_path / "cache.json"

    def write_cache() -> None:
        cache.write_text(
            json.dumps(
                {
                    "103|Other|episode|1": {
                        "summary": first_summary,
                        "schema_version": "1",
                        "inputs": first_inputs,
                        "fingerprint": stable_hash(first_inputs),
                    },
                    "103|Other|episode|2": {
                        "summary": "Overview:\nSecond overview",
                        "schema_version": "1",
                        "inputs": second_inputs,
                        "fingerprint": stable_hash(second_inputs),
                    },
                }
            ),
            encoding="utf-8",
        )

    write_cache()
    order = _order_file(tmp_path)
    output = tmp_path / "published"
    preparation.prepare_corpus_artifacts(root, output, story_order_file=order, summary_cache_file=cache)
    preparation.prepare_corpus_artifacts(
        root, output, story_order_file=order, summary_cache_file=cache, replace=True
    )
    pinned = StoryCorpus.from_publication(output)

    def rolling_metadata(directory: Path) -> tuple[str, str, str, str]:
        catalog = json.loads((directory / "catalog.json").read_text(encoding="utf-8"))
        catalog_status = {
            item["cache_key"]: item["metadata"]["rolling_freshness"]
            for item in catalog["summaries"]
        }["103|Other|episode|2"]
        lexical_status = LexicalIndex(directory / "lexical_index.db").search(
            "Second overview", n_results=1, where={"cache_key": "103|Other|episode|2"}
        )[0][1]["rolling_freshness"]
        reader = StoryCorpus.from_prepared_dir(directory)
        reader_summary = reader.get_summary("103|Other|episode|2")
        assert reader_summary is not None
        return (
            catalog_status,
            lexical_status,
            reader_summary.metadata["rolling_freshness"],
            reader_summary.metadata["freshness"],
        )

    initial_generation = output / json.loads((output / "current.json").read_text(encoding="utf-8"))["path"]
    assert rolling_metadata(initial_generation) == (
        "rolling_context_current",
        "rolling_context_current",
        "rolling_context_current",
        "unverified",
    )

    second_inputs["previous_summary_hash"] = "0" * 64
    write_cache()
    preparation.prepare_corpus_artifacts(
        root, output, story_order_file=order, summary_cache_file=cache, replace=True
    )
    pinned_summary = pinned.get_summary("103|Other|episode|2")
    assert pinned_summary is not None
    assert pinned_summary.metadata["rolling_freshness"] == "rolling_context_current"
    current = StoryCorpus.from_publication(output)
    current_summary = current.get_summary("103|Other|episode|2")
    assert current_summary is not None
    assert current_summary.metadata["rolling_freshness"] == "stale"
    current_generation = output / json.loads((output / "current.json").read_text(encoding="utf-8"))["path"]
    assert rolling_metadata(current_generation) == (
        "stale",
        "stale",
        "stale",
        "stale",
    )
    assert pinned.iter_scenes()[0]["text"] == current.iter_scenes()[0]["text"] == "first"


def test_rolling_freshness_rejects_missing_middle_and_malformed_dependencies() -> None:
    first = SummaryRecord(
        cache_key="103|Other|episode|1",
        summary="Overview:\nFirst",
        fingerprint="",
        schema_version="1",
        inputs={"level": "part", "previous_summary_hash": ""},
        metadata={},
    )
    third_inputs = {"level": "part", "previous_summary_hash": "0" * 64}
    third = SummaryRecord(
        cache_key="104|Other|episode|1",
        summary="Overview:\nThird",
        fingerprint=stable_hash(third_inputs),
        schema_version="1",
        inputs=third_inputs,
        metadata={},
    )
    records = {first.cache_key: first, third.cache_key: third}
    assert _rolling_freshness(
        third.cache_key,
        third,
        records,
        [first.cache_key, "103|Other|episode|2", third.cache_key],
    ) == "unverified"

    malformed_inputs = {"level": "part", "previous_summary_hash": "not-a-hash"}
    malformed = SummaryRecord(
        cache_key="103|Other|episode|2",
        summary="Overview:\nSecond",
        fingerprint=stable_hash(malformed_inputs),
        schema_version="1",
        inputs=malformed_inputs,
        metadata={},
    )
    assert _rolling_freshness(
        malformed.cache_key,
        malformed,
        {first.cache_key: first, malformed.cache_key: malformed},
        [first.cache_key, malformed.cache_key],
    ) == "unverified"

    invalid_fingerprint = SummaryRecord(
        cache_key="103|Other|episode|2",
        summary="Overview:\nSecond",
        fingerprint="bad",
        schema_version="1",
        inputs={"level": "part", "previous_summary_hash": ""},
        metadata={},
    )
    assert _rolling_freshness(
        invalid_fingerprint.cache_key,
        invalid_fingerprint,
        {invalid_fingerprint.cache_key: invalid_fingerprint},
        [invalid_fingerprint.cache_key],
    ) == "unverified"


def test_rolling_part_order_crosses_episode_boundaries_and_skips_blank_sections() -> None:
    previous = "Overview:\n\nContinuity Facts:\n\nKey Events:\nignored"
    assert _trim_previous_summary_context(previous) is None
    inputs = {"level": "part", "previous_summary_hash": hash_text("Overview:\nUseful")}
    record = SummaryRecord(
        cache_key="103|Other|episode-2|1",
        summary="Overview:\nUseful",
        fingerprint=stable_hash(inputs),
        schema_version="1",
        inputs=inputs,
        metadata={},
    )
    predecessor = SummaryRecord(
        cache_key="103|Other|episode-1|2",
        summary="Overview:\nUseful",
        fingerprint="",
        schema_version="1",
        inputs={"level": "part", "previous_summary_hash": ""},
        metadata={},
    )
    assert _rolling_freshness(
        record.cache_key,
        record,
        {predecessor.cache_key: predecessor, record.cache_key: record},
        [predecessor.cache_key, record.cache_key],
    ) == "rolling_context_current"


def test_prepare_corpus_cli_rejects_existing_destination(tmp_path: Path) -> None:
    story_root = tmp_path / "corpus"
    story_root.mkdir()
    output = tmp_path / "prepared"
    output.mkdir()
    result = CliRunner().invoke(
        cli.app,
        ["prepare-corpus", "--story-dir", str(story_root), "--output-dir", str(output)],
    )
    assert result.exit_code == 1
    assert "already exists" in result.stdout


def test_prepare_corpus_replace_publishes_pointer_and_retains_legacy_reader(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    source = root / "103" / "episode" / "1.md"
    source.parent.mkdir(parents=True)
    source.write_text("old scene", encoding="utf-8")
    order = _order_file(tmp_path)
    output = tmp_path / "prepared"
    preparation.prepare_corpus_artifacts(root, output, story_order_file=order)
    legacy = StoryCorpus.from_prepared_dir(output)

    source.write_text("new scene", encoding="utf-8")
    preparation.prepare_corpus_artifacts(root, output, story_order_file=order, replace=True)

    published = StoryCorpus.from_publication(output)
    assert legacy.iter_scenes()[0]["text"] == "old scene"
    assert published.iter_scenes()[0]["text"] == "new scene"
    assert (output / "current.json").is_file()
    assert len(list((output / ".revisions").iterdir())) == 1


@pytest.mark.parametrize("provider", ["google", "openai"])
def test_summarize_corpus_uses_generation_only_provider_and_reuses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    root = tmp_path / "corpus"
    source = root / "103" / "episode" / "1.md"
    source.parent.mkdir(parents=True)
    source.write_text("A short scene", encoding="utf-8")
    cache = tmp_path / "summaries.json"
    order = _order_file(tmp_path)
    monkeypatch.setenv("LINKURA_INGEST_PROVIDER", provider)
    if provider == "google":
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-google-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("LINKURA_INGEST_MODEL", "fake-generation-model")
    calls: list[str] = []

    class FakeAgent:
        def run_sync(self, prompt: str) -> SimpleNamespace:
            calls.append(prompt)
            return SimpleNamespace(
                output=(
                    "Overview:\nA generated summary. " + prompt[-20:] + "\n\n"
                    "Continuity Facts:\nThe scene is recorded."
                )
            )

    monkeypatch.setattr(summarizer, "create_generation_text_agent", lambda _: FakeAgent())
    for name in (
        "embed_texts",
        "get_chroma_collection",
        "StoryQueryEngine",
        "initialize_ingest_settings",
        "initialize_query_settings",
    ):
        monkeypatch.setattr(cli, name, lambda *args, _name=name, **kwargs: (_ for _ in ()).throw(
            AssertionError(f"{_name} must not be constructed")
        ))
    monkeypatch.setattr(
        database.chromadb,
        "PersistentClient",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Chroma must not be constructed")),
    )
    monkeypatch.setattr(
        database.genai,
        "Client",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("genai client must not be constructed")),
    )

    cache.write_text(json.dumps({"obsolete": {"summary": "remove me"}}), encoding="utf-8")

    cli.summarize_corpus_command(
        story_dir=str(root),
        cache_file=str(cache),
        story_order_file=str(order),
        glossary_file=None,
    )
    first_bytes = cache.read_bytes()
    assert len(calls) == 3
    records = json.loads(first_bytes)
    assert {"103|Other|episode|1", "EPISODE|103|Other|episode", "YEAR|103"} <= records.keys()
    assert records["103|Other|episode|1"]["inputs"]["chat_model"] == "fake-generation-model"
    assert "obsolete" not in records

    prepared = tmp_path / "prepared"
    preparation.prepare_corpus_artifacts(
        root, prepared, story_order_file=order, summary_cache_file=cache
    )
    catalog = json.loads((prepared / "catalog.json").read_text(encoding="utf-8"))
    assert catalog["summaries"][0]["metadata"]["source_freshness"] == "tracked_sources_current"

    cli.summarize_corpus_command(
        story_dir=str(root),
        cache_file=str(cache),
        story_order_file=str(order),
        glossary_file=None,
    )
    assert len(calls) == 3
    assert cache.read_bytes() == first_bytes

    source.write_text("Changed scene", encoding="utf-8")
    monkeypatch.setenv("LINKURA_INGEST_MODEL", "changed-model")
    calls.clear()
    def fail_after_one(_: str) -> FakeAgent:
        if len(calls) >= 1:
            raise RuntimeError("generation failed")
        return FakeAgent()

    monkeypatch.setattr(summarizer, "create_generation_text_agent", fail_after_one)
    with pytest.raises(RuntimeError, match="generation failed"):
        cli.summarize_corpus_command(
            story_dir=str(root),
            cache_file=str(cache),
            story_order_file=str(order),
            glossary_file=None,
        )
    assert cache.read_bytes() == first_bytes
    assert list(tmp_path.glob(".summaries.json.*.tmp")) == []


def test_prepare_replace_drops_renamed_and_resplit_source_records(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    source = root / "103" / "episode" / "1.md"
    source.parent.mkdir(parents=True)
    source.write_text("old scene", encoding="utf-8")
    output = tmp_path / "prepared"
    preparation.prepare_corpus_artifacts(root, output, story_order_file=_order_file(tmp_path))
    preparation.prepare_corpus_artifacts(
        root, output, story_order_file=_order_file(tmp_path), replace=True
    )
    pinned = StoryCorpus.from_publication(output)
    reference = pinned.scene_reference(str(source), 0)
    assert reference is not None

    source.unlink()
    replacement = source.with_name("renamed.md")
    replacement.write_text("first scene\n---\nsecond scene", encoding="utf-8")
    preparation.prepare_corpus_artifacts(
        root, output, story_order_file=_order_file(tmp_path), replace=True
    )

    reader = StoryCorpus.from_publication(output)
    pointer = json.loads((output / "current.json").read_text(encoding="utf-8"))
    generation = output / pointer["path"]
    scenes = reader.iter_scenes()
    assert [(scene["file_path"], scene["scene_index"], scene["text"]) for scene in scenes] == [
        (str(replacement), 0, "first scene"),
        (str(replacement), 1, "second scene"),
    ]
    assert reader.get_scene(str(source), 0) is None
    assert pinned.get_scene_by_reference(reference)["text"] == "old scene"
    with pytest.raises(ValueError, match="different corpus revision"):
        reader.get_scene_by_reference(reference)
    lexical = LexicalIndex(generation / "lexical_index.db")
    assert all("old scene" not in document for document, _ in lexical.search("old scene", n_results=10))
    assert lexical.search("first scene", n_results=10)


def test_prepared_reader_uses_embedded_order_and_glossary_after_inputs_deleted(
    tmp_path: Path,
) -> None:
    root = tmp_path / "corpus"
    source = root / "103" / "episode" / "1.md"
    source.parent.mkdir(parents=True)
    source.write_text("花帆 scene", encoding="utf-8")
    order = tmp_path / "order.yaml"
    order.write_text(
        "chronological_order:\n  - story_type: Other\n    arcs: [\"103\"]\n"
        "summary_order:\n  - story_type: Other\n    arcs: [\"103\"]\n",
        encoding="utf-8",
    )
    glossary = tmp_path / "glossary.json"
    glossary.write_text(json.dumps({"characters": {"花帆": "Kaho Old"}}), encoding="utf-8")
    output = tmp_path / "prepared"
    preparation.prepare_corpus_artifacts(root, output, story_order_file=order, glossary_file=glossary)
    order.unlink()
    glossary.unlink()

    reader = StoryCorpus.from_prepared_dir(output)
    assert reader.story_order.chronological_positions[("Other", "103")] == (0, 0)
    assert reader.lookup_glossary("Kaho") == {
        "category": "characters",
        "term": "花帆",
        "translation": "Kaho Old",
    }


def test_replacement_pins_glossary_snapshot_per_reader(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    source = root / "103" / "episode" / "1.md"
    source.parent.mkdir(parents=True)
    source.write_text("花帆 scene", encoding="utf-8")
    order = _order_file(tmp_path)
    glossary = tmp_path / "glossary.json"
    glossary.write_text(json.dumps({"characters": {"花帆": "Kaho Old"}}), encoding="utf-8")
    output = tmp_path / "published"
    preparation.prepare_corpus_artifacts(root, output, story_order_file=order, glossary_file=glossary)
    preparation.prepare_corpus_artifacts(
        root, output, story_order_file=order, glossary_file=glossary, replace=True
    )
    old_reader = StoryCorpus.from_publication(output)
    glossary.write_text(json.dumps({"characters": {"花帆": "Kaho New"}}), encoding="utf-8")
    preparation.prepare_corpus_artifacts(
        root, output, story_order_file=order, glossary_file=glossary, replace=True
    )

    old_match = old_reader.lookup_glossary("Kaho")
    assert old_match is not None
    assert old_match["translation"] == "Kaho Old"
    new_reader = StoryCorpus.from_publication(output)
    new_match = new_reader.lookup_glossary("Kaho")
    assert new_match is not None
    assert new_match["translation"] == "Kaho New"


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_replace_pointer_failure_preserves_previous_pointer_and_cleans_temp(
    tmp_path: Path, monkeypatch, failure
) -> None:
    root = tmp_path / "corpus"
    source = root / "103" / "episode" / "1.md"
    source.parent.mkdir(parents=True)
    source.write_text("stable", encoding="utf-8")
    output = tmp_path / "prepared"
    order = _order_file(tmp_path)
    preparation.prepare_corpus_artifacts(root, output, story_order_file=order)
    preparation.prepare_corpus_artifacts(root, output, story_order_file=order, replace=True)
    pointer_before = (output / "current.json").read_bytes()
    old_reader = StoryCorpus.from_publication(output)
    source.write_text("unpublished", encoding="utf-8")
    original_replace = Path.replace

    def fail_pointer(self: Path, target: Path):
        if self.name.startswith(".current-"):
            raise failure("pointer failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_pointer)
    with pytest.raises(failure, match="pointer failure"):
        preparation.prepare_corpus_artifacts(root, output, story_order_file=order, replace=True)
    assert (output / "current.json").read_bytes() == pointer_before
    assert old_reader.iter_scenes()[0]["text"] == "stable"
    assert not list(output.glob(".current-*.json.tmp"))


def test_replace_after_deletion_keeps_open_reader_on_old_generation(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    source = root / "103" / "episode" / "1.md"
    source.parent.mkdir(parents=True)
    source.write_text("retained", encoding="utf-8")
    output = tmp_path / "prepared"
    order = _order_file(tmp_path)
    preparation.prepare_corpus_artifacts(root, output, story_order_file=order)
    preparation.prepare_corpus_artifacts(root, output, story_order_file=order, replace=True)
    old_reader = StoryCorpus.from_publication(output)
    source.unlink()
    replacement = root / "103" / "episode" / "2.md"
    replacement.write_text("replacement", encoding="utf-8")
    preparation.prepare_corpus_artifacts(root, output, story_order_file=order, replace=True)
    assert old_reader.iter_scenes()[0]["text"] == "retained"
    assert StoryCorpus.from_publication(output).iter_scenes()[0]["text"] == "replacement"


@pytest.mark.parametrize("mutation", ["edit", "delete"])
def test_replace_pins_summary_source_freshness_per_reader(
    tmp_path: Path, mutation: str
) -> None:
    root = tmp_path / "corpus"
    source = root / "103" / "episode" / "1.md"
    source.parent.mkdir(parents=True)
    source.write_text("old scene", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    cache = tmp_path / "summaries.json"
    inputs = {"source_file_hashes": {"103/episode/1.md": digest}}
    cache.write_text(
        json.dumps(
            {
                "103|Other|episode|1": {
                    "summary": "cached",
                    "schema_version": "1",
                    "inputs": inputs,
                    "fingerprint": stable_hash(inputs),
                }
            }
        ),
        encoding="utf-8",
    )
    order = _order_file(tmp_path)
    output = tmp_path / "published"
    preparation.prepare_corpus_artifacts(
        root, output, story_order_file=order, summary_cache_file=cache
    )
    preparation.prepare_corpus_artifacts(
        root, output, story_order_file=order, summary_cache_file=cache, replace=True
    )
    old_reader = StoryCorpus.from_publication(output)
    old_summary = old_reader.get_summary("103|Other|episode|1")
    assert old_summary is not None
    assert old_summary.metadata["source_freshness"] == "tracked_sources_current"

    if mutation == "edit":
        source.write_text("edited scene", encoding="utf-8")
    else:
        source.unlink()
        (source.parent / "2.md").write_text("remaining scene", encoding="utf-8")
    preparation.prepare_corpus_artifacts(
        root, output, story_order_file=order, summary_cache_file=cache, replace=True
    )

    pinned_summary = old_reader.get_summary("103|Other|episode|1")
    current_reader = StoryCorpus.from_publication(output)
    current_summary = current_reader.get_summary("103|Other|episode|1")
    assert pinned_summary is not None
    assert current_summary is not None
    assert pinned_summary.metadata["source_freshness"] == "tracked_sources_current"
    assert current_summary.metadata["source_freshness"] == "stale"
    assert old_reader.iter_scenes()[0]["text"] == "old scene"
    assert {scene["text"] for scene in current_reader.iter_scenes()} == {
        "edited scene" if mutation == "edit" else "remaining scene"
    }


@pytest.mark.parametrize("removal", ["empty", "deleted"])
def test_summary_removal_replacement_keeps_pinned_summary_and_current_raw_access(
    tmp_path: Path, removal: str
) -> None:
    root = tmp_path / "corpus"
    source = root / "103" / "episode" / "1.md"
    source.parent.mkdir(parents=True)
    source.write_text("scene", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    inputs = {"source_file_hashes": {"103/episode/1.md": digest}}
    cache = tmp_path / "summaries.json"
    cache.write_text(
        json.dumps(
            {
                "103|Other|episode|1": {
                    "summary": "cached summary",
                    "schema_version": "1",
                    "inputs": inputs,
                    "fingerprint": stable_hash(inputs),
                }
            }
        ),
        encoding="utf-8",
    )
    order = _order_file(tmp_path)
    output = tmp_path / "published"
    preparation.prepare_corpus_artifacts(
        root, output, story_order_file=order, summary_cache_file=cache
    )
    preparation.prepare_corpus_artifacts(
        root, output, story_order_file=order, summary_cache_file=cache, replace=True
    )
    pinned = StoryCorpus.from_publication(output)
    pinned_summary = pinned.get_summary("103|Other|episode|1")
    assert pinned_summary is not None
    assert pinned_summary.summary == "cached summary"
    initial_revision = pinned.revision

    if removal == "empty":
        cache.write_text("{}", encoding="utf-8")
    else:
        cache.unlink()
    preparation.prepare_corpus_artifacts(
        root, output, story_order_file=order, summary_cache_file=cache, replace=True
    )

    current = StoryCorpus.from_publication(output)
    assert current.revision != initial_revision
    assert pinned.get_summary("103|Other|episode|1") is not None
    assert current.get_summary("103|Other|episode|1") is None
    assert current.iter_scenes()[0]["text"] == "scene"
    pointer = json.loads((output / "current.json").read_text(encoding="utf-8"))
    generation = output / pointer["path"]
    catalog = json.loads((generation / "catalog.json").read_text(encoding="utf-8"))
    assert catalog["summary_count"] == 0
    if removal == "deleted":
        assert catalog["revision_inputs"]["summary_cache"] == {
            "present": False,
            "path": str(cache.resolve()),
            "sha256": None,
        }
    assert LexicalIndex(generation / "lexical_index.db").search(
        "cached summary", n_results=5, where={"cache_key": "103|Other|episode|1"}
    ) == []


def test_prepare_corpus_cli_replace_and_active_pointer_validation(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    source = root / "103" / "episode" / "1.md"
    source.parent.mkdir(parents=True)
    source.write_text("scene", encoding="utf-8")
    output = tmp_path / "prepared"
    order = _order_file(tmp_path)
    preparation.prepare_corpus_artifacts(root, output, story_order_file=order)
    first = CliRunner().invoke(
        cli.app,
        ["prepare-corpus", "--story-dir", str(root), "--output-dir", str(output), "--story-order", str(order), "--replace"],
    )
    assert first.exit_code == 0, f"{first.stdout}\n{first.exception!r}"
    active_pointer = json.loads((output / "current.json").read_text(encoding="utf-8"))
    (output / "current.json").write_text("{malformed", encoding="utf-8")
    with pytest.raises(ValueError, match="pointer"):
        preparation.prepare_corpus_artifacts(root, output, story_order_file=order, replace=True)
    pointer = {"path": ".revisions/missing", "revision": "wrong"}
    (output / "current.json").write_text(json.dumps(pointer), encoding="utf-8")
    with pytest.raises(ValueError, match="pointer|generation"):
        preparation.prepare_corpus_artifacts(root, output, story_order_file=order, replace=True)
    active_pointer["revision"] = "wrong"
    (output / "current.json").write_text(json.dumps(active_pointer), encoding="utf-8")
    with pytest.raises(ValueError, match="revision does not match"):
        preparation.prepare_corpus_artifacts(root, output, story_order_file=order, replace=True)
    with pytest.raises(ValueError, match="revision does not match"):
        StoryCorpus.from_publication(output)


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_prepare_corpus_cleans_interrupted_staging(tmp_path: Path, monkeypatch, failure) -> None:
    root = tmp_path / "corpus"
    source = root / "103" / "episode" / "1.md"
    source.parent.mkdir(parents=True)
    source.write_text("A local scene", encoding="utf-8")
    output = tmp_path / "prepared"

    def interrupt_catalog(*args, **kwargs):
        raise failure("interrupted preparation")

    monkeypatch.setattr(preparation, "_write_catalog", interrupt_catalog)
    with pytest.raises(failure, match="interrupted preparation"):
        preparation.prepare_corpus_artifacts(root, output, story_order_file=_order_file(tmp_path))

    assert not output.exists()
    assert list(tmp_path.glob(".prepared-*")) == []
    assert source.read_text(encoding="utf-8") == "A local scene"


def test_prepare_corpus_revision_is_deterministic_and_recorded_in_all_artifacts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "corpus"
    source = root / "103" / "episode" / "1.md"
    source.parent.mkdir(parents=True)
    source.write_text("A local scene", encoding="utf-8")
    order = _order_file(tmp_path)
    glossary = tmp_path / "glossary.json"
    glossary.write_text(json.dumps({"characters": {"Kaho": "Kaho"}}), encoding="utf-8")

    from linkura_story_indexer.indexer.preparation import prepare_corpus_artifacts

    first = prepare_corpus_artifacts(root, tmp_path / "prepared-1", story_order_file=order, glossary_file=glossary)
    second = prepare_corpus_artifacts(root, tmp_path / "prepared-2", story_order_file=order, glossary_file=glossary)
    first_catalog = json.loads((first / "catalog.json").read_text(encoding="utf-8"))
    second_catalog = json.loads((second / "catalog.json").read_text(encoding="utf-8"))
    assert first_catalog["revision"] == second_catalog["revision"]
    source_metadata = SourceRecordStore(first / "source_records.db").get_metadata()
    lexical_metadata = LexicalIndex(first / "lexical_index.db").get_metadata()
    assert source_metadata["revision"] == lexical_metadata["revision"] == first_catalog["revision"]

    source.write_text("A changed local scene", encoding="utf-8")
    changed = prepare_corpus_artifacts(root, tmp_path / "prepared-3", story_order_file=order, glossary_file=glossary)
    changed_catalog = json.loads((changed / "catalog.json").read_text(encoding="utf-8"))
    assert changed_catalog["revision"] != first_catalog["revision"]

    glossary.write_text(json.dumps({"characters": {"Kaho": "Changed"}}), encoding="utf-8")
    glossary_changed = prepare_corpus_artifacts(
        root, tmp_path / "prepared-4", story_order_file=order, glossary_file=glossary
    )
    assert json.loads((glossary_changed / "catalog.json").read_text(encoding="utf-8"))["revision"] != changed_catalog["revision"]


def test_prepare_corpus_revision_tracks_source_additions_and_deletions(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    source = root / "103" / "episode" / "1.md"
    source.parent.mkdir(parents=True)
    source.write_text("A local scene", encoding="utf-8")
    order = _order_file(tmp_path)

    from linkura_story_indexer.indexer.preparation import prepare_corpus_artifacts

    first = prepare_corpus_artifacts(root, tmp_path / "prepared-1", story_order_file=order)
    first_revision = json.loads((first / "catalog.json").read_text(encoding="utf-8"))["revision"]
    added = source.parent / "2.md"
    added.write_text("Another local scene", encoding="utf-8")
    second = prepare_corpus_artifacts(root, tmp_path / "prepared-2", story_order_file=order)
    second_revision = json.loads((second / "catalog.json").read_text(encoding="utf-8"))["revision"]
    assert second_revision != first_revision
    source.unlink()
    empty = prepare_corpus_artifacts(root, tmp_path / "prepared-3", story_order_file=order)
    assert json.loads((empty / "catalog.json").read_text(encoding="utf-8"))["revision"] != second_revision


@pytest.mark.parametrize("changed_input", ["source", "addition", "deletion", "order", "glossary", "cache"])
def test_prepare_corpus_detects_mutation_before_publish(tmp_path: Path, monkeypatch, changed_input) -> None:
    root = tmp_path / "corpus"
    source = root / "103" / "episode" / "1.md"
    source.parent.mkdir(parents=True)
    source.write_text("A local scene", encoding="utf-8")
    output = tmp_path / "prepared"
    order = _order_file(tmp_path)
    glossary = tmp_path / "glossary.json"
    cache = tmp_path / "summaries.json"
    glossary.write_text("{}", encoding="utf-8")
    cache.write_text("{}", encoding="utf-8")

    original_write_catalog = preparation._write_catalog

    def mutate_after_build(*args, **kwargs):
        original_write_catalog(*args, **kwargs)
        if changed_input == "source":
            source.write_text("Changed while preparing", encoding="utf-8")
        elif changed_input == "addition":
            (source.parent / "2.md").write_text("Added", encoding="utf-8")
        elif changed_input == "deletion":
            source.unlink()
        elif changed_input == "order":
            order.write_text(order.read_text(encoding="utf-8") + "\n# changed", encoding="utf-8")
        else:
            target = glossary if changed_input == "glossary" else cache
            target.write_text('{"changed": {}}', encoding="utf-8")

    monkeypatch.setattr(preparation, "_write_catalog", mutate_after_build)
    with pytest.raises(ValueError, match="changed during preparation"):
        preparation.prepare_corpus_artifacts(
            root, output, story_order_file=order, glossary_file=glossary, summary_cache_file=cache
        )

    assert not output.exists()
    assert list(tmp_path.glob(".prepared-*")) == []


def test_prepare_corpus_detects_order_change_during_parsing(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "corpus"
    source = root / "103" / "episode" / "1.md"
    source.parent.mkdir(parents=True)
    source.write_text("A scene", encoding="utf-8")
    order = _order_file(tmp_path)
    parse = preparation.prepare_corpus

    def changing_parse(*args, **kwargs):
        result = parse(*args, **kwargs)
        order.write_text(order.read_text(encoding="utf-8") + "\n# changed", encoding="utf-8")
        return result

    monkeypatch.setattr(preparation, "prepare_corpus", changing_parse)
    with pytest.raises(ValueError, match="changed during preparation"):
        preparation.prepare_corpus_artifacts(root, tmp_path / "prepared", story_order_file=order)
    assert not (tmp_path / "prepared").exists()
    assert list(tmp_path.glob(".prepared-*")) == []


def test_record_mutations_invalidate_revision(tmp_path: Path) -> None:
    source = SourceRecordStore(tmp_path / "source.db")
    source.set_metadata({"revision": "old"})
    source.replace_all([], [])
    assert source.get_metadata() == {}
    lexical = LexicalIndex(tmp_path / "lexical.db")
    lexical.set_metadata({"revision": "old"})
    lexical.upsert_records(ids=["a"], documents=["scene"], metadatas=[{}])
    assert lexical.get_metadata() == {}
    lexical.set_metadata({"revision": "new"})
    lexical.delete_records(["a"])
    assert lexical.get_metadata() == {}
