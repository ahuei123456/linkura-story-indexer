from typing import Any

from linkura_story_indexer import database
from linkura_story_indexer.query import engine as query_engine
from linkura_story_indexer.query.engine import StoryQueryEngine


def make_engine() -> StoryQueryEngine:
    engine = StoryQueryEngine.__new__(StoryQueryEngine)
    engine.state_ledger = {
        "103": {"characters": [{"name": "Kaho Hinoshita"}]},
        "104": {"characters": [{"name": "Sayaka Murano"}]},
    }
    engine.glossary = None
    return engine


def test_system_prompt_softens_raw_source_claim_and_compacts_ledger():
    engine = make_engine()

    prompt = engine._build_system_prompt({"103"})

    assert "based strictly on the provided raw source text" not in prompt
    assert "Some retrieved context may be generated summaries" in prompt
    assert '{"characters":[{"name":"Kaho Hinoshita"}]}' in prompt
    assert '\n  "characters"' not in prompt
    assert "YEAR 104 FACTS" not in prompt


def test_state_ledger_arc_ids_prefers_explicit_question_arc():
    engine = make_engine()

    arc_ids = engine._state_ledger_arc_ids("What happened in 103?", {"104"})

    assert arc_ids == {"103"}


def test_state_ledger_arc_ids_falls_back_to_retrieved_arcs():
    engine = make_engine()

    arc_ids = engine._state_ledger_arc_ids("What happened to Kaho?", {"103", "104"})

    assert arc_ids == {"103", "104"}


def test_retrieve_uses_query_embedding_task_type(monkeypatch):
    engine = StoryQueryEngine.__new__(StoryQueryEngine)
    calls: list[dict[str, Any]] = []

    class FakeCollection:
        def query(self, **kwargs: Any) -> dict[str, list[list[Any]]]:
            calls.append(kwargs)
            return {
                "documents": [["summary"]],
                "metadatas": [[{"arc_id": "103", "summary_level": 3}]],
            }

    def fake_embed_texts(texts: list[str], *, task_type: str) -> list[list[float]]:
        calls.append({"texts": texts, "task_type": task_type})
        return [[0.1, 0.2]]

    engine.collection = FakeCollection()
    monkeypatch.setattr(query_engine, "embed_texts", fake_embed_texts)

    retrieved = engine._retrieve("question")

    assert retrieved == [("summary", {"arc_id": "103", "summary_level": 3})]
    assert calls[0] == {"texts": ["question"], "task_type": database.RETRIEVAL_QUERY}
    assert calls[1]["query_embeddings"] == [[0.1, 0.2]]
