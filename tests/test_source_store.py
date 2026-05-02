import sqlite3
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


def test_source_store_queries_composite_speaker_tokens_without_dup_turns(tmp_path: Path) -> None:
    story_root = tmp_path / "story"
    script_path = _write_story_file(
        story_root,
        "103/第1話『花咲きたい！』/1.md",
        "\n---\n".join(
            [
                "花帆: こんにちは",
                "梢＆慈: おお……。",
                "瑠璃乃&amp;姫芽: で、バズ曲ってなんだ？？？？？？",
                "全員: おー！！",
            ]
        ),
    )
    raw_nodes = StoryProcessor.process_file(script_path)
    chunks = build_retrieval_chunks(raw_nodes, min_chars=1, target_chars=500, max_chars=500)
    db_path = tmp_path / "source.db"
    store = SourceRecordStore(db_path)

    store.replace_all(raw_nodes, chunks)

    assert raw_nodes[1].dialogue_turns[0].speaker == "梢＆慈"
    assert raw_nodes[1].dialogue_turns[0].speaker_tokens == ["梢", "慈"]
    assert raw_nodes[2].dialogue_turns[0].speaker_tokens == ["瑠璃乃", "姫芽"]
    assert raw_nodes[3].dialogue_turns[0].speaker_tokens == ["全員"]
    assert raw_nodes[3].dialogue_turns[0].speaker_kind == "collective"

    expected_chunk_ids = ["chunk:103|Main|第1話『花咲きたい！』|1:0-3"]
    assert store.chunk_ids_for_speaker("梢") == expected_chunk_ids
    assert store.chunk_ids_for_speaker("慈") == expected_chunk_ids
    assert store.chunk_ids_for_speaker("瑠璃乃") == expected_chunk_ids
    assert store.chunk_ids_for_speaker("姫芽") == expected_chunk_ids
    assert store.chunk_ids_for_speaker("全員") == expected_chunk_ids
    assert store.count_turns("梢") == 1
    assert store.count_turns("慈") == 1
    assert store.count_turns("全員") == 1
    assert store.count_turns("花帆") == 1

    with sqlite3.connect(db_path) as connection:
        dialogue_turn_count = connection.execute(
            "SELECT COUNT(*) FROM dialogue_turns WHERE speaker = ?",
            ("梢＆慈",),
        ).fetchone()[0]
        mapped_speakers = connection.execute(
            """
            SELECT speaker
            FROM dialogue_turn_speakers
            WHERE turn_id = ?
            ORDER BY speaker
            """,
            (raw_nodes[1].dialogue_turns[0].turn_id,),
        ).fetchall()

    assert dialogue_turn_count == 1
    assert [row[0] for row in mapped_speakers] == ["慈", "梢"]


def test_source_store_backfills_speaker_mapping_for_existing_turns(tmp_path: Path) -> None:
    db_path = tmp_path / "source.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE source_scenes (
                scene_id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                parent_part_id TEXT NOT NULL,
                scene_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE dialogue_turns (
                turn_id TEXT PRIMARY KEY,
                scene_id TEXT NOT NULL,
                turn_index INTEGER NOT NULL,
                speaker TEXT NOT NULL,
                text TEXT NOT NULL,
                line_start INTEGER NOT NULL,
                line_end INTEGER NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE retrieval_chunk_sources (
                chunk_id TEXT NOT NULL,
                scene_id TEXT NOT NULL,
                scene_index INTEGER NOT NULL,
                PRIMARY KEY (chunk_id, scene_id)
            )
            """
        )
        connection.execute(
            """
            INSERT INTO source_scenes (
                scene_id, file_path, parent_part_id, scene_index, text, metadata_json
            )
            VALUES ('scene:legacy:0', 'story.md', 'part:legacy', 0, '梢＆慈: おお……。', '{}')
            """
        )
        connection.execute(
            """
            INSERT INTO dialogue_turns (
                turn_id, scene_id, turn_index, speaker, text, line_start, line_end
            )
            VALUES ('turn:legacy:0', 'scene:legacy:0', 0, '梢＆慈', 'おお……。', 0, 0)
            """
        )
        connection.execute(
            """
            INSERT INTO retrieval_chunk_sources (chunk_id, scene_id, scene_index)
            VALUES ('chunk:legacy:0-0', 'scene:legacy:0', 0)
            """
        )

    store = SourceRecordStore(db_path)

    assert store.chunk_ids_for_speaker("梢") == ["chunk:legacy:0-0"]
    assert store.chunk_ids_for_speaker("慈") == ["chunk:legacy:0-0"]
    assert store.count_turns("梢") == 1
