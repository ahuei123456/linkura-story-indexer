from pathlib import Path
from typing import Any

from linkura_story_indexer import cli
from linkura_story_indexer.lexical import LexicalIndex, expand_query_with_glossary
from linkura_story_indexer.models.story import StoryMetadata, StoryNode


def _metadata(**overrides: Any) -> dict[str, Any]:
    metadata = {
        "arc_id": "103",
        "story_type": "Main",
        "episode_name": "第1話『花咲きたい！』",
        "part_name": "1",
        "file_path": "story/103/第1話『花咲きたい！』/1.md",
        "scene_index": 0,
        "scene_start": 0,
        "scene_end": 1,
        "source_scene_count": 2,
        "canonical_story_order": 1,
        "parent_year_id": "103",
        "parent_episode_id": "103|Main|第1話『花咲きたい！』",
        "parent_part_id": "103|Main|第1話『花咲きたい！』|1",
        "summary_level": 4,
    }
    metadata.update(overrides)
    return metadata


def test_glossary_expansion_adds_full_and_short_aliases() -> None:
    glossary = {"characters": {"日野下花帆": "Kaho Hinoshita"}}

    expanded = expand_query_with_glossary("What does Kaho say?", glossary)

    assert "What does Kaho say?" in expanded
    assert "Kaho Hinoshita" in expanded
    assert "日野下花帆" in expanded
    assert "花帆" in expanded


def test_lexical_index_finds_short_japanese_name_and_preserves_span(tmp_path: Path) -> None:
    index = LexicalIndex(tmp_path / "lexical.db")
    index.upsert_records(
        ids=["chunk:one:0-1", "chunk:two:0-0"],
        documents=["花帆: 走ろう", "さやか: 待って"],
        metadatas=[
            _metadata(scene_start=0, scene_end=1),
            _metadata(scene_start=0, scene_end=0, parent_part_id="other"),
        ],
    )

    results = index.search("花帆", n_results=5)

    assert results == [("花帆: 走ろう", _metadata(scene_start=0, scene_end=1))]


def test_lexical_index_replaces_records_cleanly(tmp_path: Path) -> None:
    index = LexicalIndex(tmp_path / "lexical.db")
    index.upsert_records(
        ids=["chunk:one:0-0"],
        documents=["花帆: 古い"],
        metadatas=[_metadata()],
    )
    index.upsert_records(
        ids=["chunk:one:0-0"],
        documents=["さやか: 新しい"],
        metadatas=[_metadata()],
    )

    assert index.search("花帆", n_results=5) == []
    assert index.search("さやか", n_results=5) == [("さやか: 新しい", _metadata())]


def test_lexical_search_honors_chroma_style_where_filter(tmp_path: Path) -> None:
    index = LexicalIndex(tmp_path / "lexical.db")
    index.upsert_records(
        ids=["chunk:one:0-0", "chunk:two:0-0"],
        documents=["花帆: one", "花帆: two"],
        metadatas=[
            _metadata(parent_part_id="part-one"),
            _metadata(parent_part_id="part-two"),
        ],
    )

    results = index.search(
        "花帆",
        n_results=5,
        where={"$and": [{"summary_level": 4}, {"parent_part_id": "part-two"}]},
    )

    assert results == [("花帆: two", _metadata(parent_part_id="part-two"))]


def test_upsert_story_nodes_writes_matching_lexical_records(tmp_path: Path, monkeypatch) -> None:
    collection_records: dict[str, dict[str, Any]] = {}

    class FakeCollection:
        def upsert(
            self,
            *,
            ids: list[str],
            documents: list[str],
            metadatas: list[dict[str, Any]],
            embeddings: list[list[float]],
        ) -> None:
            for record_id, document, metadata in zip(ids, documents, metadatas, strict=True):
                collection_records[record_id] = {"document": document, "metadata": metadata}

    node = StoryNode(
        text="花帆: こんにちは",
        metadata=StoryMetadata(
            arc_id="103",
            story_type="Main",
            episode_name="第1話『花咲きたい！』",
            part_name="1",
            file_path="story/103/第1話『花咲きたい！』/1.md",
            detected_speakers=["花帆"],
            parent_year_id="103",
            parent_episode_id="103|Main|第1話『花咲きたい！』",
            parent_part_id="103|Main|第1話『花咲きたい！』|1",
        ),
        summary_level=4,
    )
    lexical_index = LexicalIndex(tmp_path / "lexical.db")

    monkeypatch.setattr(cli, "get_chroma_collection", lambda: FakeCollection())
    monkeypatch.setattr(cli, "embed_texts", lambda texts, *, task_type: [[1.0] for _ in texts])

    cli._upsert_story_nodes(
        [node],
        progress_label="Embedding test node",
        glossary={"characters": {"日野下花帆": "Kaho Hinoshita"}},
        lexical_index=lexical_index,
    )

    record_id = cli._node_id(node)
    assert record_id in collection_records
    assert lexical_index.search("Kaho", n_results=5) == [
        ("花帆: こんにちは", collection_records[record_id]["metadata"])
    ]
