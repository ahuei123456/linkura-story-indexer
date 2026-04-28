from pathlib import Path

from linkura_story_indexer.indexer.chunker import build_retrieval_chunks
from linkura_story_indexer.indexer.processor import StoryProcessor


def _write_story_file(root: Path, relative_path: str, content: str) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_tiny_adjacent_scenes_are_coalesced_into_span_chunks(tmp_path: Path) -> None:
    story_root = tmp_path / "story"
    path = _write_story_file(
        story_root,
        "103/第1話『花咲きたい！』/1.md",
        "\n---\n".join(
            [
                "花帆: aaaaaaaaaa",
                "さやか: bbbbbbbbbb",
                "花帆: cccccccccc",
                "さやか: dddddddddd",
                "花帆: eeeeeeeeee",
            ]
        ),
    )
    raw_nodes = StoryProcessor.process_file(path)

    chunks = build_retrieval_chunks(raw_nodes, min_chars=35, target_chars=55, max_chars=80)

    assert len(chunks) < len(raw_nodes)
    assert [(chunk.metadata.scene_start, chunk.metadata.scene_end) for chunk in chunks] == [
        (0, 2),
        (3, 4),
    ]
    assert chunks[0].metadata.source_scene_count == 3
    assert chunks[0].metadata.detected_speakers == ["花帆", "さやか"]
    assert chunks[0].metadata.speakers == ["花帆", "さやか"]
    assert chunks[0].metadata.source_scene_ids == [
        "scene:103|Main|第1話『花咲きたい！』|1:0",
        "scene:103|Main|第1話『花咲きたい！』|1:1",
        "scene:103|Main|第1話『花咲きたい！』|1:2",
    ]
    assert chunks[0].metadata.source_turn_ids == [
        "turn:scene:103|Main|第1話『花咲きたい！』|1:0:0",
        "turn:scene:103|Main|第1話『花咲きたい！』|1:1:0",
        "turn:scene:103|Main|第1話『花咲きたい！』|1:2:0",
    ]


def test_chunks_do_not_cross_file_or_part_boundaries(tmp_path: Path) -> None:
    story_root = tmp_path / "story"
    part_one = _write_story_file(
        story_root,
        "103/第1話『花咲きたい！』/1.md",
        "花帆: one\n---\n花帆: two",
    )
    part_two = _write_story_file(
        story_root,
        "103/第1話『花咲きたい！』/2.md",
        "さやか: three\n---\nさやか: four",
    )
    raw_nodes = [
        *StoryProcessor.process_file(part_one),
        *StoryProcessor.process_file(part_two),
    ]

    chunks = build_retrieval_chunks(raw_nodes, min_chars=50, target_chars=80, max_chars=120)

    assert len(chunks) == 2
    assert {chunk.metadata.file_path for chunk in chunks} == {str(part_one), str(part_two)}
    assert all(chunk.metadata.scene_start == 0 for chunk in chunks)
    assert all(chunk.metadata.scene_end == 1 for chunk in chunks)


def test_abyss_regression_produces_fewer_retrieval_chunks() -> None:
    abyss_path = next(Path("story/105").glob("第5話*/ABYSS.md"))
    raw_nodes = StoryProcessor.process_file(abyss_path)

    chunks = build_retrieval_chunks(raw_nodes)

    assert len(raw_nodes) == 32
    assert len(chunks) < 16
    assert chunks[0].metadata.scene_start == 0
    assert chunks[-1].metadata.scene_end == 31
