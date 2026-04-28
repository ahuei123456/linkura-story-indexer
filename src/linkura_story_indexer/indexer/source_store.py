import json
import os
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..models.story import DialogueTurn, NarrativeBeat, StoryNode

DEFAULT_SOURCE_DB_PATH = "./source_records.db"


def get_source_db_path() -> str:
    return os.getenv("LINKURA_SOURCE_DB_PATH", DEFAULT_SOURCE_DB_PATH)


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _chunk_id(node: StoryNode) -> str:
    meta = node.metadata
    if meta.chunk_id:
        return meta.chunk_id
    return f"chunk:{meta.parent_part_id}:{meta.scene_start}-{meta.scene_end}"


class SourceRecordStore:
    """Persists atomic source scenes, dialogue turns, narrative beats, and chunk provenance."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or get_source_db_path())
        if self.path.parent != Path("."):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS source_scenes (
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
                CREATE TABLE IF NOT EXISTS dialogue_turns (
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
                CREATE TABLE IF NOT EXISTS narrative_beats (
                    beat_id TEXT PRIMARY KEY,
                    scene_id TEXT NOT NULL,
                    beat_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    line_start INTEGER NOT NULL,
                    line_end INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS retrieval_chunk_sources (
                    chunk_id TEXT NOT NULL,
                    scene_id TEXT NOT NULL,
                    scene_index INTEGER NOT NULL,
                    PRIMARY KEY (chunk_id, scene_id)
                )
                """
            )

    def replace_all(self, raw_nodes: list[StoryNode], retrieval_chunks: list[StoryNode]) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM source_scenes")
            connection.execute("DELETE FROM dialogue_turns")
            connection.execute("DELETE FROM narrative_beats")
            connection.execute("DELETE FROM retrieval_chunk_sources")
            self._upsert_raw_nodes(connection, raw_nodes)
            self._upsert_chunk_mappings(connection, retrieval_chunks)

    def _upsert_raw_nodes(
        self,
        connection: sqlite3.Connection,
        raw_nodes: list[StoryNode],
    ) -> None:
        for node in raw_nodes:
            scene_ids = node.metadata.source_scene_ids
            if not scene_ids:
                continue
            scene_id = scene_ids[0]
            connection.execute(
                """
                INSERT INTO source_scenes (
                    scene_id, file_path, parent_part_id, scene_index, text, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(scene_id) DO UPDATE SET
                    file_path = excluded.file_path,
                    parent_part_id = excluded.parent_part_id,
                    scene_index = excluded.scene_index,
                    text = excluded.text,
                    metadata_json = excluded.metadata_json
                """,
                (
                    scene_id,
                    node.metadata.file_path,
                    node.metadata.parent_part_id,
                    node.metadata.scene_index,
                    node.text,
                    _stable_json(node.metadata.model_dump()),
                ),
            )
            self._upsert_turns(connection, node.dialogue_turns)
            self._upsert_beats(connection, node.narrative_beats)

    def _upsert_turns(
        self,
        connection: sqlite3.Connection,
        turns: Iterable[DialogueTurn],
    ) -> None:
        for turn in turns:
            connection.execute(
                """
                INSERT INTO dialogue_turns (
                    turn_id, scene_id, turn_index, speaker, text, line_start, line_end
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(turn_id) DO UPDATE SET
                    scene_id = excluded.scene_id,
                    turn_index = excluded.turn_index,
                    speaker = excluded.speaker,
                    text = excluded.text,
                    line_start = excluded.line_start,
                    line_end = excluded.line_end
                """,
                (
                    turn.turn_id,
                    turn.scene_id,
                    turn.turn_index,
                    turn.speaker,
                    turn.text,
                    turn.line_start,
                    turn.line_end,
                ),
            )

    def _upsert_beats(
        self,
        connection: sqlite3.Connection,
        beats: Iterable[NarrativeBeat],
    ) -> None:
        for beat in beats:
            connection.execute(
                """
                INSERT INTO narrative_beats (
                    beat_id, scene_id, beat_index, text, line_start, line_end
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(beat_id) DO UPDATE SET
                    scene_id = excluded.scene_id,
                    beat_index = excluded.beat_index,
                    text = excluded.text,
                    line_start = excluded.line_start,
                    line_end = excluded.line_end
                """,
                (
                    beat.beat_id,
                    beat.scene_id,
                    beat.beat_index,
                    beat.text,
                    beat.line_start,
                    beat.line_end,
                ),
            )

    def _upsert_chunk_mappings(
        self,
        connection: sqlite3.Connection,
        retrieval_chunks: list[StoryNode],
    ) -> None:
        for chunk in retrieval_chunks:
            chunk_id = _chunk_id(chunk)
            for offset, scene_id in enumerate(chunk.metadata.source_scene_ids):
                connection.execute(
                    """
                    INSERT INTO retrieval_chunk_sources (chunk_id, scene_id, scene_index)
                    VALUES (?, ?, ?)
                    ON CONFLICT(chunk_id, scene_id) DO UPDATE SET
                        scene_index = excluded.scene_index
                    """,
                    (chunk_id, scene_id, chunk.metadata.scene_start + offset),
                )

    def chunk_ids_for_speaker(self, speaker: str) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT r.chunk_id
                FROM retrieval_chunk_sources AS r
                JOIN dialogue_turns AS t ON t.scene_id = r.scene_id
                WHERE t.speaker = ?
                ORDER BY r.chunk_id
                """,
                (speaker,),
            ).fetchall()
        return [str(row["chunk_id"]) for row in rows]

    def turns_matching_text(self, text: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    t.turn_id,
                    t.scene_id,
                    t.turn_index,
                    t.speaker,
                    t.text,
                    s.file_path,
                    s.parent_part_id,
                    s.scene_index
                FROM dialogue_turns AS t
                JOIN source_scenes AS s ON s.scene_id = t.scene_id
                WHERE t.text LIKE ?
                ORDER BY s.file_path, s.scene_index, t.turn_index
                """,
                (f"%{text}%",),
            ).fetchall()
        return [dict(row) for row in rows]

    def count_turns(
        self,
        speaker: str,
        *,
        parent_part_id: str | None = None,
    ) -> int:
        filters = ["t.speaker = ?"]
        params: list[Any] = [speaker]
        if parent_part_id is not None:
            filters.append("s.parent_part_id = ?")
            params.append(parent_part_id)
        where = " AND ".join(filters)
        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT COUNT(*) AS count
                FROM dialogue_turns AS t
                JOIN source_scenes AS s ON s.scene_id = t.scene_id
                WHERE {where}
                """,
                params,
            ).fetchone()
        return int(row["count"]) if row is not None else 0
