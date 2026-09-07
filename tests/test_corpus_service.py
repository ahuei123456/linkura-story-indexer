import json
from pathlib import Path

import pytest

from linkura_story_indexer.corpus import SceneReference, StoryCorpus
from linkura_story_indexer.indexer.chunker import build_retrieval_chunks
from linkura_story_indexer.indexer.processor import StoryProcessor
from linkura_story_indexer.indexer.source_store import SourceRecordStore
from linkura_story_indexer.lexical import LexicalIndex


def _order_file(path: Path) -> Path:
    result = path / "order.yaml"
    result.write_text(
        """
chronological_order:
  - story_type: Main
    arcs: [\"103\"]
summary_order:
  - story_type: Main
    arcs: [\"103\"]
""",
        encoding="utf-8",
    )
    return result


def test_story_corpus_reads_source_metadata_and_cached_summary_without_query_engine(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_file = tmp_path / "summaries.json"
    cache_file.write_text(
        json.dumps(
            {
                "103|Main|Episode A|1": {
                    "schema_version": "3",
                    "fingerprint": "source-fingerprint",
                    "summary": "Overview:\nA short summary.",
                    "inputs": {"level": "part", "source_file_hashes": {"1.md": "abc"}},
                }
            }
        ),
        encoding="utf-8",
    )

    class QueryEngineMustNotConstruct:
        def __init__(self, *args, **kwargs):
            raise AssertionError("StoryCorpus must not construct a query engine")

    monkeypatch.setattr("linkura_story_indexer.query.engine.StoryQueryEngine", QueryEngineMustNotConstruct)
    monkeypatch.setattr("linkura_story_indexer.indexer.summarizer.HierarchicalSummarizer", QueryEngineMustNotConstruct)
    monkeypatch.setattr("linkura_story_indexer.database.create_generation_model", QueryEngineMustNotConstruct)
    monkeypatch.setattr("linkura_story_indexer.database.create_generation_text_agent", QueryEngineMustNotConstruct)
    monkeypatch.setattr("linkura_story_indexer.database.genai.Client", QueryEngineMustNotConstruct)
    monkeypatch.setattr("linkura_story_indexer.database.chromadb.PersistentClient", QueryEngineMustNotConstruct)
    corpus = StoryCorpus.from_paths(
        source_db=tmp_path / "source.db",
        story_order_file=_order_file(tmp_path),
        summary_cache_file=cache_file,
    )

    assert corpus.iter_scenes() == []
    summary = corpus.get_summary("103|Main|Episode A|1")
    assert summary is not None
    assert summary.metadata == {
        "tier": "Part",
        "arc_id": "103",
        "story_type": "Main",
        "episode_name": "Episode A",
        "part_name": "1",
    }
    assert summary.inputs["source_file_hashes"] == {"1.md": "abc"}
    assert corpus.get_summary("missing") is None


def test_story_corpus_glossary_aliases_and_optional_files(tmp_path: Path) -> None:
    glossary_file = tmp_path / "glossary.json"
    glossary_file.write_text(
        json.dumps({"characters": {"日野下花帆": "Kaho Hinoshita"}}),
        encoding="utf-8",
    )
    corpus = StoryCorpus.from_paths(
        source_db=tmp_path / "source.db",
        story_order_file=_order_file(tmp_path),
        glossary_file=glossary_file,
    )

    assert corpus.lookup_glossary("Kaho") == {
        "category": "characters",
        "term": "日野下花帆",
        "translation": "Kaho Hinoshita",
    }
    assert corpus.summary_cache == {}
    assert corpus.lookup_glossary("unknown") is None


def test_story_corpus_can_wrap_existing_source_store(tmp_path: Path) -> None:
    store = SourceRecordStore(tmp_path / "source.db")
    corpus = StoryCorpus(store, summaries={"YEAR|103": {"summary": "year"}})

    assert corpus.source_store is store
    summary = corpus.get_summary("YEAR|103")
    assert summary is not None
    assert summary.metadata == {"tier": "Year", "arc_id": "103"}


def test_story_corpus_keeps_raw_access_when_summary_cache_is_missing(tmp_path: Path) -> None:
    story_file = tmp_path / "story" / "103" / "第1話『Episode』" / "1.md"
    story_file.parent.mkdir(parents=True)
    story_file.write_text("花帆: raw", encoding="utf-8")
    raw_nodes = StoryProcessor.process_file(story_file)
    raw_nodes[0].metadata.story_order = 1
    raw_nodes[0].metadata.canonical_story_order = 1
    store = SourceRecordStore(tmp_path / "source.db")
    store.replace_all(raw_nodes, build_retrieval_chunks(raw_nodes, min_chars=1))
    corpus = StoryCorpus.from_paths(
        source_db=store.path,
        summary_cache_file=tmp_path / "missing-summaries.json",
    )

    assert len(corpus.iter_scenes()) == 1
    scene = corpus.get_scene(raw_nodes[0].metadata.file_path, 0)
    assert scene is not None
    assert scene["text"] == "花帆: raw"
    assert corpus.get_summary("missing") is None
    assert corpus.max_story_order(arc_id="103") == 1


def test_story_corpus_scene_references_reject_other_or_stale_revisions(tmp_path: Path) -> None:
    story_file = tmp_path / "story" / "103" / "Episode" / "1.md"
    story_file.parent.mkdir(parents=True)
    story_file.write_text("first scene", encoding="utf-8")
    raw_nodes = StoryProcessor.process_file(story_file)
    raw_nodes[0].metadata.story_order = 1
    raw_nodes[0].metadata.canonical_story_order = 1
    store = SourceRecordStore(tmp_path / "source.db")
    store.replace_all(raw_nodes, build_retrieval_chunks(raw_nodes, min_chars=1))
    corpus = StoryCorpus(store, revision="rev-1")

    reference = corpus.scene_reference(raw_nodes[0].metadata.file_path, 0)
    assert reference is not None
    assert corpus.get_scene_by_reference(reference)["text"] == "first scene"
    with pytest.raises(ValueError, match="different corpus revision"):
        corpus.get_scene_by_reference(
            SceneReference("rev-2", reference.file_path, 0, reference.scene_id)
        )

    story_file.write_text("replacement scene", encoding="utf-8")
    replacement_nodes = StoryProcessor.process_file(story_file)
    replacement_nodes[0].metadata.story_order = 1
    replacement_nodes[0].metadata.canonical_story_order = 1
    store.replace_all(replacement_nodes, build_retrieval_chunks(replacement_nodes, min_chars=1))
    replacement = StoryCorpus(store, revision="rev-2")
    with pytest.raises(ValueError, match="different corpus revision"):
        replacement.get_scene_by_reference(reference)


def test_story_corpus_validates_prepared_artifacts_before_reading(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    revision_inputs = {"source_files": {"1.md": "abc"}}
    story_order = {
        "file": {"present": False},
        "positions": {
            "chronological": [[ ["Main", "103"], [4, 2] ]],
            "summary": [[ ["Main", "103"], [3, 1] ]],
            "parts": [[ ["103", "Main", "Episode A"], [["2", 0], ["1", 1]] ]],
        },
    }
    glossary = {"characters": {"花帆": "Kaho Hinoshita"}}
    (prepared / "catalog.json").write_text(
        json.dumps(
            {
                "revision": "rev-1",
                "revision_inputs": revision_inputs,
                "story_order": story_order,
                "glossary": glossary,
                "summaries": [{"cache_key": "YEAR|103", "summary": "year"}],
            }
        ),
        encoding="utf-8",
    )
    source = SourceRecordStore(prepared / "source_records.db")
    source.set_metadata({"revision": "rev-1", "revision_inputs": revision_inputs})
    lexical = LexicalIndex(prepared / "lexical_index.db")
    lexical.set_metadata({"revision": "rev-1", "revision_inputs": revision_inputs})

    corpus = StoryCorpus.from_prepared_dir(prepared)

    assert corpus.revision == "rev-1"
    assert corpus.get_summary("YEAR|103") is not None
    assert corpus.story_order.chronological_positions[("Main", "103")] == (4, 2)
    assert corpus.story_order.part_key("103", "Main", "Episode A", "2") == (0, 0)
    assert corpus.lookup_glossary("Kaho") == {
        "category": "characters",
        "term": "花帆",
        "translation": "Kaho Hinoshita",
    }


def test_story_corpus_rejects_mixed_prepared_revisions(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "catalog.json").write_text(
        json.dumps({"revision": "rev-1", "revision_inputs": {}}), encoding="utf-8"
    )
    source = SourceRecordStore(prepared / "source_records.db")
    source.set_metadata({"revision": "rev-1", "revision_inputs": {}})
    lexical = LexicalIndex(prepared / "lexical_index.db")
    lexical.set_metadata({"revision": "rev-2", "revision_inputs": {}})

    with pytest.raises(ValueError, match="share the catalog revision"):
        StoryCorpus.from_prepared_dir(prepared)


def test_new_prepared_schema_requires_embedded_metadata(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    revision_inputs = {}
    (prepared / "catalog.json").write_text(
        json.dumps(
            {"schema_version": 5, "revision": "rev-1", "revision_inputs": revision_inputs}
        ),
        encoding="utf-8",
    )
    source = SourceRecordStore(prepared / "source_records.db")
    source.set_metadata({"revision": "rev-1", "revision_inputs": revision_inputs})
    lexical = LexicalIndex(prepared / "lexical_index.db")
    lexical.set_metadata({"revision": "rev-1", "revision_inputs": revision_inputs})

    with pytest.raises(ValueError, match="story-order metadata is missing"):
        StoryCorpus.from_prepared_dir(prepared)


def test_prepared_reader_validation_is_read_only_for_missing_or_malformed_databases(
    tmp_path: Path,
) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "catalog.json").write_text(
        json.dumps({"revision": "rev-1", "revision_inputs": {}}), encoding="utf-8"
    )
    missing_source = prepared / "source_records.db"
    missing_lexical = prepared / "lexical_index.db"
    with pytest.raises(FileNotFoundError):
        StoryCorpus.from_prepared_dir(prepared)
    assert not missing_source.exists()
    assert not missing_lexical.exists()

    missing_source.write_bytes(b"not sqlite")
    missing_lexical.write_bytes(b"also not sqlite")
    before = {path: path.read_bytes() for path in (missing_source, missing_lexical)}
    with pytest.raises(ValueError, match="metadata is unreadable"):
        StoryCorpus.from_prepared_dir(prepared)
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize(
    "pointer",
    [
        {"path": "../outside", "revision": "rev-1"},
        {"path": ".revisions/generation", "revision": "wrong"},
        {"path": ".revisions/generation"},
    ],
)
def test_publication_reader_rejects_invalid_pointers(tmp_path: Path, pointer: dict[str, str]) -> None:
    (tmp_path / "current.json").write_text(json.dumps(pointer), encoding="utf-8")
    with pytest.raises(ValueError):
        StoryCorpus.from_publication(tmp_path)
