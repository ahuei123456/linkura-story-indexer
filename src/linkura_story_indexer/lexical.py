import json
import os
import re
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

DEFAULT_LEXICAL_DB_PATH = "./lexical_index.db"

_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff々〆〤ー]+")
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]*")


def get_lexical_db_path() -> str:
    return os.getenv("LINKURA_LEXICAL_DB_PATH", DEFAULT_LEXICAL_DB_PATH)


def _ordered_unique(values: Iterable[str]) -> list[str]:
    unique = []
    seen = set()
    for value in values:
        cleaned = value.strip()
        if not cleaned or cleaned in seen:
            continue
        unique.append(cleaned)
        seen.add(cleaned)
    return unique


def _english_alias_parts(value: str) -> list[str]:
    return [part for part in _WORD_RE.findall(value) if len(part) >= 2]


def _japanese_short_aliases(value: str) -> list[str]:
    if not _CJK_RE.fullmatch(value):
        return []
    aliases = []
    if len(value) >= 4:
        aliases.append(value[-2:])
    if len(value) >= 5:
        aliases.append(value[-3:])
    return aliases


def glossary_alias_groups(glossary: dict[str, dict[str, str]] | None) -> list[list[str]]:
    if not glossary:
        return []

    groups = []
    for terms in glossary.values():
        if not isinstance(terms, dict):
            continue
        for japanese, english in terms.items():
            aliases = [japanese, english]
            aliases.extend(_english_alias_parts(english))
            aliases.extend(_japanese_short_aliases(japanese))
            groups.append(_ordered_unique(aliases))
    return groups


def expand_query_with_glossary(
    question: str,
    glossary: dict[str, dict[str, str]] | None,
) -> str:
    additions = []
    question_lower = question.casefold()

    for aliases in glossary_alias_groups(glossary):
        if any(alias in question or alias.casefold() in question_lower for alias in aliases):
            additions.extend(aliases)

    expanded_aliases = _ordered_unique(additions)
    if not expanded_aliases:
        return question
    return f"{question} {' / '.join(expanded_aliases)}"


def _fts_terms(query: str) -> list[str]:
    terms = []
    terms.extend(term for term in _CJK_RE.findall(query) if len(term) >= 3)
    terms.extend(term for term in _WORD_RE.findall(query) if len(term) >= 3)
    return _ordered_unique(terms)


def _like_terms(query: str) -> list[str]:
    terms = []
    terms.extend(term for term in _CJK_RE.findall(query) if len(term) >= 2)
    terms.extend(term for term in _WORD_RE.findall(query) if len(term) >= 3)
    return _ordered_unique(terms)


def _quote_fts_term(term: str) -> str:
    escaped = term.replace('"', '""')
    return f'"{escaped}"'


def _metadata_matches_where(metadata: dict[str, Any], where: dict[str, Any] | None) -> bool:
    if not where:
        return True

    and_filters = where.get("$and")
    if isinstance(and_filters, list):
        return all(
            isinstance(item, dict) and _metadata_matches_where(metadata, item)
            for item in and_filters
        )

    for key, expected in where.items():
        if key == "$and":
            continue
        actual = metadata.get(key)
        if key == "story_order" and actual is None:
            actual = metadata.get("canonical_story_order")
        if isinstance(expected, dict):
            if not _metadata_matches_operator(actual, expected):
                return False
            continue
        if actual != expected:
            return False
    return True


def _metadata_matches_operator(actual: Any, expected: dict[str, Any]) -> bool:
    for operator, value in expected.items():
        if operator == "$eq":
            if actual != value:
                return False
            continue
        if operator == "$in":
            if not isinstance(value, list) or actual not in value:
                return False
            continue

        if not isinstance(actual, (int, float)) or not isinstance(value, (int, float)):
            return False
        if operator == "$lt" and not actual < value:
            return False
        if operator == "$lte" and not actual <= value:
            return False
        if operator == "$gt" and not actual > value:
            return False
        if operator == "$gte" and not actual >= value:
            return False
        if operator not in {"$lt", "$lte", "$gt", "$gte"}:
            return False
    return True


class LexicalIndex:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or get_lexical_db_path())
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
                CREATE TABLE IF NOT EXISTS lexical_records (
                    id TEXT PRIMARY KEY,
                    document TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    search_text TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS lexical_records_fts
                USING fts5(id UNINDEXED, search_text, tokenize='trigram')
                """
            )

    def upsert_records(
        self,
        *,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        search_texts: list[str] | None = None,
    ) -> None:
        if search_texts is None:
            search_texts = documents
        if not (len(ids) == len(documents) == len(metadatas) == len(search_texts)):
            raise ValueError("ids, documents, metadatas, and search_texts must have matching lengths")

        with self._connect() as connection:
            for record_id, document, metadata, search_text in zip(
                ids,
                documents,
                metadatas,
                search_texts,
                strict=True,
            ):
                connection.execute("DELETE FROM lexical_records WHERE id = ?", (record_id,))
                connection.execute("DELETE FROM lexical_records_fts WHERE id = ?", (record_id,))
                metadata_json = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
                connection.execute(
                    """
                    INSERT INTO lexical_records (id, document, metadata_json, search_text)
                    VALUES (?, ?, ?, ?)
                    """,
                    (record_id, document, metadata_json, search_text),
                )
                connection.execute(
                    """
                    INSERT INTO lexical_records_fts (id, search_text)
                    VALUES (?, ?)
                    """,
                    (record_id, search_text),
                )

    def list_ids(self) -> set[str]:
        with self._connect() as connection:
            rows = connection.execute("SELECT id FROM lexical_records").fetchall()
        return {str(row["id"]) for row in rows}

    def delete_records(self, ids: Iterable[str]) -> None:
        sorted_ids = sorted(set(ids))
        if not sorted_ids:
            return

        with self._connect() as connection:
            for record_id in sorted_ids:
                connection.execute("DELETE FROM lexical_records WHERE id = ?", (record_id,))
                connection.execute("DELETE FROM lexical_records_fts WHERE id = ?", (record_id,))

    def search(
        self,
        query: str,
        *,
        n_results: int,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[str, dict[str, Any]]]:
        if n_results < 1:
            return []

        candidate_limit = max(n_results * 10, 50) if where else n_results
        rows: list[sqlite3.Row] = []
        seen_ids: set[str] = set()

        fts_query = " OR ".join(_quote_fts_term(term) for term in _fts_terms(query))
        with self._connect() as connection:
            if fts_query:
                try:
                    rows.extend(
                        connection.execute(
                            """
                            SELECT r.id, r.document, r.metadata_json
                            FROM lexical_records_fts AS f
                            JOIN lexical_records AS r ON r.id = f.id
                            WHERE f.search_text MATCH ?
                            ORDER BY bm25(f)
                            LIMIT ?
                            """,
                            (fts_query, candidate_limit),
                        ).fetchall()
                    )
                except sqlite3.OperationalError:
                    rows = []

            for term in _like_terms(query):
                for row in connection.execute(
                    """
                    SELECT id, document, metadata_json
                    FROM lexical_records
                    WHERE search_text LIKE ?
                    LIMIT ?
                    """,
                    (f"%{term}%", candidate_limit),
                ).fetchall():
                    if row["id"] not in seen_ids:
                        rows.append(row)
                        seen_ids.add(row["id"])

        results = []
        emitted_ids = set()
        for row in rows:
            record_id = str(row["id"])
            if record_id in emitted_ids:
                continue
            emitted_ids.add(record_id)
            metadata = json.loads(str(row["metadata_json"]))
            if not isinstance(metadata, dict):
                continue
            if not _metadata_matches_where(metadata, where):
                continue
            results.append((str(row["document"]), metadata))
            if len(results) >= n_results:
                break

        return results
