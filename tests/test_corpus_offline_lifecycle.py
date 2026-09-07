"""Exercise publication recovery with every external service unavailable."""

from pathlib import Path

import pytest

from linkura_story_indexer import cli, database
from linkura_story_indexer.corpus import StoryCorpus
from linkura_story_indexer.indexer import preparation
from linkura_story_indexer.lexical import LexicalIndex


def test_offline_publication_restarts_after_failed_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("Corpus lifecycle attempted to use an external service")

    for name in (
        "initialize_generation_settings", "initialize_ingest_settings",
        "initialize_query_settings", "HierarchicalSummarizer", "StoryQueryEngine",
        "embed_texts",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    for name in (
        "initialize_generation_settings", "initialize_ingest_settings",
        "initialize_query_settings", "create_generation_model",
        "create_generation_text_agent", "embed_texts", "get_chroma_collection",
    ):
        monkeypatch.setattr(database, name, forbidden)
    monkeypatch.setattr(database.genai, "Client", forbidden)
    monkeypatch.setattr(database.chromadb, "PersistentClient", forbidden)

    root = tmp_path / "corpus"
    first = root / "103" / "episode" / "1.md"
    second = first.with_name("2.md")
    first.parent.mkdir(parents=True)
    first.write_text("original retained scene", encoding="utf-8")
    second.write_text("removeduniquetoken", encoding="utf-8")
    order = tmp_path / "order.yaml"
    order.write_text(
        'chronological_order:\n  - story_type: Other\n    arcs: ["103"]\n'
        'summary_order:\n  - story_type: Other\n    arcs: ["103"]\n',
        encoding="utf-8",
    )
    output = tmp_path / "prepared"

    def publish(*, replace: bool = False) -> None:
        preparation.prepare_corpus_artifacts(
            root, output, story_order_file=order,
            summary_cache_file=tmp_path / "absent-cache.json", replace=replace,
        )

    publish()
    publish(replace=True)
    pinned = StoryCorpus.from_publication(output)
    original_scenes = pinned.iter_scenes()
    pinned_reference = pinned.scene_reference(original_scenes[0]["file_path"], 0)
    assert pinned_reference is not None
    pointer_before = (output / "current.json").read_bytes()
    first.write_text("updateduniquetoken", encoding="utf-8")
    second.unlink()

    write_catalog = preparation._write_catalog

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("injected interrupted rebuild")

    monkeypatch.setattr(preparation, "_write_catalog", interrupted)
    with pytest.raises(KeyboardInterrupt, match="injected interrupted rebuild"):
        publish(replace=True)
    assert (output / "current.json").read_bytes() == pointer_before
    assert list(tmp_path.glob(".prepared-*")) == []
    assert StoryCorpus.from_publication(output).iter_scenes() == original_scenes

    monkeypatch.setattr(preparation, "_write_catalog", write_catalog)
    publish(replace=True)
    current = StoryCorpus.from_publication(output)
    assert current.revision != pinned.revision
    assert pinned.iter_scenes() == original_scenes
    assert pinned.get_scene_by_reference(pinned_reference) == original_scenes[0]
    with pytest.raises(ValueError, match="different corpus revision"):
        current.get_scene_by_reference(pinned_reference)
    assert [scene["text"] for scene in current.iter_scenes()] == ["updateduniquetoken"]
    assert current.iter_summaries() == []
    generation, _ = preparation.resolve_publication_pointer(output)
    lexical = LexicalIndex(generation / "lexical_index.db")
    assert lexical.search("removeduniquetoken", n_results=5) == []
    assert lexical.search("updateduniquetoken", n_results=5)
