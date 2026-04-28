from pathlib import Path

from linkura_story_indexer.indexer.chunker import build_retrieval_chunks
from linkura_story_indexer.indexer.processor import StoryProcessor
from linkura_story_indexer.indexer.source_store import SourceRecordStore


def _write_story_file(root: Path, relative_path: str, content: str) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_source_store_persists_turns_beats_and_chunk_speaker_mapping(tmp_path: Path) -> None:
    story_root = tmp_path / "story"
    script_path = _write_story_file(
        story_root,
        "103/第1話『花咲きたい！』/1.md",
        "花帆: こんにちは\n---\nさやか: どうしたの？",
    )
    prose_path = _write_story_file(
        story_root,
        "103/第1話『花咲きたい！』/2.md",
        "泉は笑った。「行こう」それから走った。",
    )
    raw_nodes = [
        *StoryProcessor.process_file(script_path),
        *StoryProcessor.process_file(prose_path),
    ]
    chunks = build_retrieval_chunks(raw_nodes, min_chars=1, target_chars=500, max_chars=500)
    store = SourceRecordStore(tmp_path / "source.db")

    store.replace_all(raw_nodes, chunks)
    first_kaho_chunks = store.chunk_ids_for_speaker("花帆")
    store.replace_all(raw_nodes, chunks)

    assert store.chunk_ids_for_speaker("花帆") == first_kaho_chunks
    assert store.chunk_ids_for_speaker("さやか") == [
        "chunk:103|Main|第1話『花咲きたい！』|1:0-1"
    ]
    assert store.turns_matching_text("行こう")[0]["speaker"] == "UNKNOWN"
    assert store.count_turns("花帆") == 1
