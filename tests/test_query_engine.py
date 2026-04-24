from typing import Any

from linkura_story_indexer import database
from linkura_story_indexer.query import engine as query_engine
from linkura_story_indexer.query.engine import INSUFFICIENT_SOURCE_CONTEXT, StoryQueryEngine


def make_engine() -> StoryQueryEngine:
    engine = StoryQueryEngine.__new__(StoryQueryEngine)
    engine.state_ledger = {
        "103": {"characters": [{"name": "Kaho Hinoshita"}]},
        "104": {"characters": [{"name": "Sayaka Murano"}]},
    }
    engine.glossary = None
    return engine


def test_system_prompt_restores_raw_source_claim_and_compacts_ledger():
    engine = make_engine()

    prompt = engine._build_system_prompt({"103"})

    assert "based strictly on the provided raw source text" in prompt
    assert "Some retrieved context may be generated summaries" not in prompt
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


def test_query_expands_summary_hits_to_raw_scenes(monkeypatch):
    engine = make_engine()
    query_calls: list[dict[str, Any]] = []
    agent_prompts: list[str] = []

    class FakeCollection:
        def query(self, **kwargs: Any) -> dict[str, list[list[Any]]]:
            query_calls.append(kwargs)
            if len(query_calls) == 1:
                return {
                    "documents": [["part summary"]],
                    "metadatas": [
                        [
                            {
                                "arc_id": "103",
                                "story_type": "Main",
                                "episode_name": "第3話『テスト』",
                                "part_name": "2",
                                "summary_level": 3,
                                "parent_part_id": "103|Main|第3話『テスト』|2",
                            }
                        ]
                    ],
                }
            return {
                "documents": [["花帆: raw scene"]],
                "metadatas": [
                    [
                        {
                            "arc_id": "103",
                            "story_type": "Main",
                            "episode_name": "第3話『テスト』",
                            "part_name": "2",
                            "summary_level": 4,
                            "file_path": "missing.md",
                            "scene_index": 4,
                            "canonical_story_order": 30,
                            "parent_part_id": "103|Main|第3話『テスト』|2",
                        }
                    ]
                ],
            }

    class FakeAgent:
        def run_sync(self, prompt: str) -> Any:
            agent_prompts.append(prompt)

            class Result:
                output = "answered from raw scene"

            return Result()

    monkeypatch.setattr(query_engine, "embed_texts", lambda texts, *, task_type: [[0.1]])
    monkeypatch.setattr(query_engine, "create_text_agent", lambda system_prompt: FakeAgent())
    engine.collection = FakeCollection()

    answer = engine.query("What happened?")

    assert answer == "answered from raw scene"
    assert len(query_calls) == 2
    assert query_calls[1]["where"] == {
        "$and": [
            {"summary_level": 4},
            {"parent_part_id": "103|Main|第3話『テスト』|2"},
        ]
    }
    assert "SUMMARY:" not in agent_prompts[0]
    assert "花帆: raw scene" in agent_prompts[0]
    assert "103 · Episode 3 · Part 2 · Scene 5" in agent_prompts[0]


def test_query_reports_insufficient_source_context_without_raw_evidence(monkeypatch):
    engine = make_engine()
    query_calls: list[dict[str, Any]] = []
    agent_called = False

    class FakeCollection:
        def query(self, **kwargs: Any) -> dict[str, list[list[Any]]]:
            query_calls.append(kwargs)
            return {
                "documents": [["part summary"]],
                "metadatas": [
                    [
                        {
                            "arc_id": "103",
                            "summary_level": 3,
                            "parent_part_id": "103|Main|第3話『テスト』|2",
                        }
                    ]
                ],
            }

    def fake_create_text_agent(system_prompt: str) -> Any:
        nonlocal agent_called
        agent_called = True
        return object()

    monkeypatch.setattr(query_engine, "embed_texts", lambda texts, *, task_type: [[0.1]])
    monkeypatch.setattr(query_engine, "create_text_agent", fake_create_text_agent)
    engine.collection = FakeCollection()

    answer = engine.query("What happened?")

    assert answer == INSUFFICIENT_SOURCE_CONTEXT
    assert len(query_calls) == 2
    assert agent_called is False


def test_fetch_raw_text_returns_only_requested_scene(tmp_path):
    engine = make_engine()
    story_file = tmp_path / "part.md"
    story_file.write_text("scene zero\n---\nscene one\n---\nscene two", encoding="utf-8")

    raw_text = engine._fetch_raw_text({"file_path": str(story_file), "scene_index": 1})

    assert raw_text == "scene one"


def test_citation_label_and_metadata_are_split():
    engine = make_engine()
    metadata = {
        "arc_id": "103",
        "story_type": "Main",
        "episode_name": "第3話『テスト』",
        "part_name": "2",
        "file_path": "story/103/第3話『テスト』/2.md",
        "scene_index": 4,
        "canonical_story_order": 30,
    }

    label = engine._citation_label(metadata)
    citation_metadata = engine._citation_metadata(metadata)

    assert label == "103 · Episode 3 · Part 2 · Scene 5"
    assert "story/" not in label
    assert citation_metadata == {
        "file_path": "story/103/第3話『テスト』/2.md",
        "scene_index": 4,
        "canonical_story_order": 30,
    }
