from pathlib import Path
from typing import Any

from linkura_story_indexer import cli
from linkura_story_indexer.database import RETRIEVAL_DOCUMENT
from linkura_story_indexer.indexer.processor import StoryProcessor


class FakeCollection:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}

    def upsert(
        self,
        *,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        for record_id, document, metadata, embedding in zip(
            ids,
            documents,
            metadatas,
            embeddings,
            strict=True,
        ):
            self.records[record_id] = {
                "document": document,
                "metadata": metadata,
                "embedding": embedding,
            }


def _write_story_file(root: Path, relative_path: str, content: str) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_raw_scene_upsert_indexes_every_scene_with_required_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    story_root = tmp_path / "story"
    script_path = _write_story_file(
        story_root,
        "103/第1話『花咲きたい！』/1.md",
        "花帆: こんにちは\nさやか: どうしたの？\n---\n花帆: 行こう\nさやか: うん",
    )
    prose_path = _write_story_file(
        story_root,
        "103/第1話『花咲きたい！』/2.md",
        "花帆は廊下を歩いた。\n朝の光が差していた。",
    )
    raw_nodes = [
        *StoryProcessor.process_file(script_path),
        *StoryProcessor.process_file(prose_path),
    ]
    cli._assign_canonical_story_order(raw_nodes)

    collection = FakeCollection()
    embedding_calls: list[dict[str, Any]] = []

    def fake_embed_texts(texts: list[str], *, task_type: str) -> list[list[float]]:
        embedding_calls.append({"texts": texts, "task_type": task_type})
        return [[float(index)] for index, _ in enumerate(texts)]

    monkeypatch.setattr(cli, "get_chroma_collection", lambda: collection)
    monkeypatch.setattr(cli, "embed_texts", fake_embed_texts)

    cli._upsert_story_nodes(
        raw_nodes,
        progress_label="Embedding test scenes",
        glossary={"characters": {"花帆": "Kaho Hinoshita", "さやか": "Sayaka Murano"}},
    )

    assert len(collection.records) == 3
    assert embedding_calls[0]["task_type"] == RETRIEVAL_DOCUMENT
    assert "Aliases: Kaho Hinoshita, Sayaka Murano" in embedding_calls[0]["texts"][0]

    required_keys = {
        "arc_id",
        "story_type",
        "episode_name",
        "part_name",
        "scene_index",
        "canonical_story_order",
        "parent_year_id",
        "parent_episode_id",
        "parent_part_id",
        "file_path",
        "detected_speakers",
        "is_prose",
        "summary_level",
    }
    for record in collection.records.values():
        metadata = record["metadata"]
        assert required_keys <= metadata.keys()
        assert isinstance(metadata["scene_index"], int)
        assert isinstance(metadata["canonical_story_order"], int)
        assert isinstance(metadata["detected_speakers"], str)
        assert isinstance(metadata["is_prose"], bool)
        assert metadata["summary_level"] == 4

    records = list(collection.records.values())
    assert records[0]["metadata"]["detected_speakers"] == "花帆|さやか"
    assert records[0]["metadata"]["is_prose"] is False
    assert records[-1]["metadata"]["detected_speakers"] == ""
    assert records[-1]["metadata"]["is_prose"] is True


def test_node_ids_are_unique_for_summaries_and_scenes(tmp_path: Path) -> None:
    story_root = tmp_path / "story"
    part_one = _write_story_file(
        story_root,
        "103/第1話『花咲きたい！』/1.md",
        "花帆: one\nさやか: two\n---\n花帆: three\nさやか: four",
    )
    part_two = _write_story_file(
        story_root,
        "103/第1話『花咲きたい！』/2.md",
        "花帆: five\nさやか: six",
    )
    raw_nodes = [
        *StoryProcessor.process_file(part_one),
        *StoryProcessor.process_file(part_two),
    ]

    part_summary_one = raw_nodes[0].model_copy(deep=True)
    part_summary_one.summary_level = 3
    part_summary_one.metadata.scene_index = -1
    part_summary_two = raw_nodes[-1].model_copy(deep=True)
    part_summary_two.summary_level = 3
    part_summary_two.metadata.scene_index = -1
    episode_summary = raw_nodes[0].model_copy(deep=True)
    episode_summary.summary_level = 2
    episode_summary.metadata.scene_index = -1
    year_summary = raw_nodes[0].model_copy(deep=True)
    year_summary.summary_level = 1
    year_summary.metadata.scene_index = -1

    nodes = [
        *raw_nodes,
        part_summary_one,
        part_summary_two,
        episode_summary,
        year_summary,
    ]

    ids = [cli._node_id(node) for node in nodes]

    assert len(ids) == len(set(ids))
    assert cli._node_id(raw_nodes[0]) != cli._node_id(raw_nodes[1])
    assert cli._node_id(part_summary_one) != cli._node_id(part_summary_two)


def test_upsert_raw_scenes_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    story_root = tmp_path / "story"
    path = _write_story_file(
        story_root,
        "103/第1話『花咲きたい！』/1.md",
        "花帆: こんにちは\nさやか: どうしたの？\n---\n花帆: 行こう\nさやか: うん",
    )
    raw_nodes = StoryProcessor.process_file(path)
    cli._assign_canonical_story_order(raw_nodes)

    collection = FakeCollection()

    monkeypatch.setattr(cli, "get_chroma_collection", lambda: collection)
    monkeypatch.setattr(
        cli,
        "embed_texts",
        lambda texts, *, task_type: [[1.0] for _ in texts],
    )

    cli._upsert_story_nodes(raw_nodes, progress_label="Embedding test scenes")
    first_count = len(collection.records)
    cli._upsert_story_nodes(raw_nodes, progress_label="Embedding test scenes")

    assert first_count == 2
    assert len(collection.records) == first_count
