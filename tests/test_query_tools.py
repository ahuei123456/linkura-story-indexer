from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from linkura_story_indexer.indexer.chunker import build_retrieval_chunks
from linkura_story_indexer.indexer.processor import StoryProcessor
from linkura_story_indexer.indexer.source_store import SourceRecordStore
from linkura_story_indexer.query import engine as query_engine
from linkura_story_indexer.query.engine import RetrievalConfig, StoryQueryEngine
from linkura_story_indexer.query.tools import (
    GetSceneInput,
    LookupGlossaryInput,
    SearchRawInput,
    SearchSummariesInput,
    build_query_toolset,
    get_scene,
    lookup_glossary,
    search_raw,
    search_summaries,
)


def make_engine() -> StoryQueryEngine:
    engine = StoryQueryEngine.__new__(StoryQueryEngine)
    engine.retrieval_config = RetrievalConfig(neighbor_scene_window=0)
    engine.glossary = {"characters": {"日野下花帆": "Kaho Hinoshita"}}
    engine.state_ledger = {}
    return engine


def raw_node(
    text: str = "花帆: raw scene",
    *,
    scene_start: int = 0,
    arc_id: str = "103",
) -> tuple[str, dict[str, Any]]:
    return (
        text,
        {
            "arc_id": arc_id,
            "story_type": "Main",
            "episode_name": "第1話『花咲きたい！』",
            "episode_number": 1,
            "part_name": "1",
            "summary_level": 4,
            "file_path": "story/103/第1話『花咲きたい！』/1.md",
            "scene_index": scene_start,
            "scene_start": scene_start,
            "scene_end": scene_start,
            "source_scene_count": 1,
            "canonical_story_order": scene_start,
            "parent_part_id": "103|Main|第1話『花咲きたい！』|1",
            "chunk_id": f"chunk:103:1:{scene_start}",
            "detected_speakers": "花帆",
        },
    )


def summary_node(
    text: str = "episode summary",
    *,
    summary_level: int = 2,
    arc_id: str = "103",
) -> tuple[str, dict[str, Any]]:
    return (
        text,
        {
            "arc_id": arc_id,
            "story_type": "Main",
            "episode_name": "第1話『花咲きたい！』",
            "episode_number": 1,
            "part_name": "1",
            "summary_level": summary_level,
            "parent_episode_id": "103|Main|第1話『花咲きたい！』",
        },
    )


@pytest.mark.parametrize(
    "model,args",
    [
        (SearchRawInput, {"query": "x", "top_k": 0}),
        (SearchRawInput, {"query": "x", "top_k": 1, "scene_start": -1}),
        (SearchRawInput, {"query": "x", "top_k": 1, "scene_start": 3, "scene_end": 2}),
        (SearchSummariesInput, {"query": "x", "top_k": 0}),
        (SearchSummariesInput, {"query": "x", "top_k": 1, "summary_level": 4}),
        (GetSceneInput, {"file_path": "story.md", "scene_index": -1}),
    ],
)
def test_query_tool_inputs_reject_invalid_arguments(
    model: type[Any],
    args: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(args)


def test_search_raw_returns_ranked_candidates_and_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = make_engine()
    seed = raw_node("weak seed", scene_start=0)
    exact = raw_node("花帆 talks about practice", scene_start=1)
    captured: list[dict[str, Any]] = []

    class FakeSourceStore:
        def chunk_ids_for_speaker(self, speaker: str) -> list[str]:
            assert speaker == "花帆"
            return ["chunk:103:1:0", "chunk:103:1:1"]

    engine.source_store = FakeSourceStore()

    def fake_hybrid_retrieve_trace(
        question: str,
        *,
        n_results: int,
        where: dict[str, Any] | None,
        query_embedding: list[float] | None,
        dense_unavailable_reason: str | None = None,
    ) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, query_engine.StageTrace]]:
        captured.append({"question": question, "n_results": n_results, "where": where})
        assert query_embedding == [0.1]
        assert dense_unavailable_reason is None
        return (
            [seed, exact],
            {
                "dense_raw": engine._trace_stage("dense_raw", []),
                "lexical_raw": engine._trace_stage("lexical_raw", []),
                "rrf_fusion": engine._trace_stage("rrf_fusion", []),
            },
        )

    monkeypatch.setattr(engine, "_query_embedding", lambda question: [0.1])
    monkeypatch.setattr(engine, "_hybrid_retrieve_trace", fake_hybrid_retrieve_trace)
    monkeypatch.setattr(engine, "_fetch_raw_text", lambda metadata: "")

    result = search_raw(
        engine,
            SearchRawInput(
                query="What practice does 花帆 mention?",
            top_k=1,
            arc_id="103",
            episode=1,
            part="1",
            scene_start=1,
            speakers=["花帆"],
        ),
    )

    assert result.candidates[0].text == "花帆 talks about practice"
    assert result.candidates[0].citation_label == "103 · Episode 1 · Part 1 · Scene 2"
    assert result.trace_stages["final_top_k"].candidates is not None
    assert result.trace_stages["final_top_k"].candidates[0].rank == 1
    assert captured[0]["where"] == {
        "$and": [
            {"summary_level": 4},
            {"arc_id": "103"},
            {"episode_number": 1},
            {"part_name": "1"},
            {"scene_end": {"$gte": 1}},
            {"scene_start": {"$lte": 1}},
            {"chunk_id": {"$in": ["chunk:103:1:0", "chunk:103:1:1"]}},
        ]
    }


def test_search_summaries_filters_level_and_arc(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = make_engine()
    summary = summary_node("Kaho starts school.", summary_level=2, arc_id="103")
    raw = raw_node("raw should not leak", arc_id="103")
    captured: list[dict[str, Any]] = []

    def fake_hybrid_retrieve_trace(
        question: str,
        *,
        n_results: int,
        where: dict[str, Any] | None,
        query_embedding: list[float] | None,
        dense_unavailable_reason: str | None = None,
    ) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, query_engine.StageTrace]]:
        captured.append({"n_results": n_results, "where": where})
        return (
            [summary, raw],
            {
                "dense_raw": engine._trace_stage("dense_raw", []),
                "lexical_raw": engine._trace_stage("lexical_raw", []),
                "rrf_fusion": engine._trace_stage("rrf_fusion", []),
            },
        )

    monkeypatch.setattr(engine, "_query_embedding", lambda question: [0.1])
    monkeypatch.setattr(engine, "_hybrid_retrieve_trace", fake_hybrid_retrieve_trace)

    result = search_summaries(
        engine,
        SearchSummariesInput(query="Kaho", top_k=3, summary_level=2, arc_id="103"),
    )

    assert [candidate.text for candidate in result.candidates] == ["Kaho starts school."]
    assert captured == [
        {
            "n_results": 3,
            "where": {"$and": [{"summary_level": 2}, {"arc_id": "103"}]},
        }
    ]


def test_get_scene_returns_exact_source_text_and_structured_errors(tmp_path: Path) -> None:
    story_file = tmp_path / "story" / "103" / "第1話『花咲きたい！』" / "1.md"
    story_file.parent.mkdir(parents=True, exist_ok=True)
    story_file.write_text("花帆: scene zero\n---\nさやか: scene one", encoding="utf-8")
    raw_nodes = StoryProcessor.process_file(story_file)
    chunks = build_retrieval_chunks(raw_nodes, min_chars=1, target_chars=500, max_chars=500)
    store = SourceRecordStore(tmp_path / "source.db")
    store.replace_all(raw_nodes, chunks)

    engine = make_engine()
    engine.source_store = store

    found = get_scene(engine, GetSceneInput(file_path=str(story_file), scene_index=1))
    missing = get_scene(engine, GetSceneInput(file_path=str(story_file), scene_index=5))

    assert found.candidates[0].text == "さやか: scene one"
    assert found.candidates[0].metadata["scene_index"] == 1
    assert found.candidates[0].source_identity is not None
    assert missing.candidates == []
    assert missing.errors == ["scene not found"]


def test_lookup_glossary_resolves_terms_translations_and_aliases() -> None:
    engine = make_engine()

    canonical = lookup_glossary(engine, LookupGlossaryInput(term="日野下花帆"))
    translation = lookup_glossary(engine, LookupGlossaryInput(term="Kaho Hinoshita"))
    alias = lookup_glossary(engine, LookupGlossaryInput(term="花帆"))
    miss = lookup_glossary(engine, LookupGlossaryInput(term="unknown"))

    assert canonical.match_type == "canonical"
    assert translation.match_type == "translation"
    assert alias.match_type == "alias"
    assert alias.canonical_term == "日野下花帆"
    assert alias.translation == "Kaho Hinoshita"
    assert "花帆" in alias.aliases
    assert miss.match_type == "miss"
    assert miss.errors == ["glossary term not found: unknown"]


def test_build_query_toolset_registers_pydantic_schema_tools() -> None:
    engine = make_engine()
    toolset = build_query_toolset(engine)
    test_model = TestModel()
    agent = Agent(test_model)

    agent.run_sync("What tools are available?", toolsets=[toolset])

    request_parameters = test_model.last_model_request_parameters
    assert request_parameters is not None
    tools = request_parameters.function_tools
    by_name = {tool.name: tool for tool in tools}
    assert set(by_name) == {"search_raw", "search_summaries", "get_scene", "lookup_glossary"}
    search_raw_schema = by_name["search_raw"].parameters_json_schema
    assert {"query", "top_k", "scene_start", "scene_end"}.issubset(
        search_raw_schema["properties"]
    )
    assert search_raw_schema["properties"]["top_k"]["minimum"] == 1
