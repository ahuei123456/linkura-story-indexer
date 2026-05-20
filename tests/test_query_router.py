from __future__ import annotations

from typing import Any

import pytest

from linkura_story_indexer.cli import _router_debug_lines
from linkura_story_indexer.query.engine import (
    RetrievalConfig,
    RetrievalTraceResult,
    StoryQueryEngine,
)
from linkura_story_indexer.query.router import (
    FixtureQueryRouter,
    RouterOutput,
    _compressed_numbers,
    _router_instructions,
    validate_router_output,
)


def make_engine() -> StoryQueryEngine:
    engine = StoryQueryEngine.__new__(StoryQueryEngine)
    engine.retrieval_config = RetrievalConfig(neighbor_scene_window=0, final_top_k=5)
    engine.glossary = {"characters": {"日野下花帆": "Kaho Hinoshita"}}
    engine.state_ledger = {}
    return engine


def raw_node(text: str = "raw scene") -> tuple[str, dict[str, Any]]:
    return (
        text,
        {
            "arc_id": "103",
            "story_type": "Main",
            "episode_name": "第1話『花咲きたい！』",
            "episode_number": 1,
            "part_name": "1",
            "summary_level": 4,
            "file_path": "story/103/第1話『花咲きたい！』/1.md",
            "scene_index": 0,
            "scene_start": 0,
            "scene_end": 0,
            "source_scene_count": 1,
            "canonical_story_order": 1,
            "chunk_id": "chunk:103:1:0",
        },
    )


def test_router_output_accepts_structured_tool_selection() -> None:
    output = RouterOutput.model_validate(
        {"tool_name": "search_raw", "args": {"query": "Kaho", "top_k": 3}}
    )

    assert output.tool_name == "search_raw"
    assert output.args == {"query": "Kaho", "top_k": 3}


def test_router_prompt_guides_story_locations_to_filtered_raw_search() -> None:
    instructions = _router_instructions()

    assert "'104th term', '104 term', or 'Year 104' map to arc_id='104'" in instructions
    assert "'episode 1', 'ep 1', or '第1話' map to episode=1" in instructions
    assert "use search_raw with arc_id and episode filters" in instructions
    assert "how did Kosuzu join the school idol club" in instructions


def test_router_debug_lines_include_selection_and_fallback_metadata() -> None:
    lines = _router_debug_lines(
        {
            "router_model": "fixture-router",
            "chosen_tool": "search_raw",
            "validated_args": {"query": "Kaho", "top_k": 5},
            "fallback_used": True,
            "fallback_reason": "invalid tool arguments",
            "validation_errors": ["bad args"],
            "raw_structured_model_output": {"tool_name": "search_summaries", "args": {}},
        }
    )

    assert lines[0] == "[bold cyan]Router:[/bold cyan]"
    assert "  model: fixture-router" in lines
    assert "  chosen_tool: search_raw" in lines
    assert '  args: {"query": "Kaho", "top_k": 5}' in lines
    assert "  fallback_used: True" in lines
    assert "  fallback_reason: invalid tool arguments" in lines
    assert '  validation_errors: ["bad args"]' in lines
    assert '  raw_output: {"args": {}, "tool_name": "search_summaries"}' in lines


def test_router_prompt_includes_available_episode_catalog() -> None:
    engine = make_engine()

    class FakeSourceStore:
        def iter_scenes(self) -> list[dict[str, Any]]:
            return [
                {"metadata": {"arc_id": "104", "story_type": "Main", "episode_number": 1}},
                {"metadata": {"arc_id": "104", "story_type": "Main", "episode_number": 2}},
                {"metadata": {"arc_id": "104", "story_type": "Main", "episode_number": 4}},
                {"metadata": {"arc_id": "105", "story_type": "Main", "episode_number": 12}},
            ]

    engine.source_store = FakeSourceStore()

    instructions = _router_instructions(engine)

    assert "Available numbered episodes by arc/story type" in instructions
    assert "arc_id=104, story_type=Main: episodes 1-2, 4" in instructions
    assert "arc_id=105, story_type=Main: episodes 12" in instructions
    assert "only pass an episode filter if the requested episode appears" in instructions


def test_compressed_numbers_formats_ranges_and_gaps() -> None:
    assert _compressed_numbers({1, 2, 3, 5, 8, 9}) == "1-3, 5, 8-9"


def test_router_rejects_unknown_tool_with_raw_search_fallback() -> None:
    decision = validate_router_output(
        RouterOutput(tool_name="unknown", args={"query": "ignored"}),
        question="original question",
        final_top_k=5,
        router_model="test-router",
    )

    assert decision.fallback_used is True
    assert decision.tool_name == "search_raw"
    assert decision.validated_args.model_dump(include={"query", "top_k"}) == {
        "query": "original question",
        "top_k": 5,
    }
    assert decision.validation_errors == ["unknown tool name: unknown"]


def test_router_validates_selected_tool_arguments() -> None:
    valid = validate_router_output(
        RouterOutput(tool_name="search_summaries", args={"query": "Kaho", "summary_level": 2}),
        question="original",
        final_top_k=5,
        router_model="test-router",
    )
    invalid = validate_router_output(
        RouterOutput(tool_name="search_summaries", args={"query": "Kaho", "summary_level": 4}),
        question="original",
        final_top_k=5,
        router_model="test-router",
    )

    assert valid.fallback_used is False
    assert valid.validated_args.model_dump()["summary_level"] == 2
    assert invalid.fallback_used is True
    assert invalid.tool_name == "search_raw"
    assert "summary_level" in invalid.validation_errors[0]


def test_fixture_router_dispatches_selected_tool_without_model_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = make_engine()
    node = raw_node("花帆: routed raw scene")
    captured: list[dict[str, Any]] = []

    def fake_retrieve_raw_nodes_with_trace(
        question: str,
        *,
        where: dict[str, Any] | None = None,
        top_k: int | None = None,
        n_results: int | None = None,
        analysis: Any = None,
    ) -> RetrievalTraceResult:
        captured.append(
            {"question": question, "where": where, "top_k": top_k, "n_results": n_results}
        )
        return RetrievalTraceResult(
            nodes=[node],
            stages={"final_top_k": engine._trace_stage("final_top_k", [])},
        )

    monkeypatch.setattr(engine, "retrieve_raw_nodes_with_trace", fake_retrieve_raw_nodes_with_trace)
    monkeypatch.setattr(engine, "_fetch_raw_text", lambda metadata: "")

    router = FixtureQueryRouter("search_raw", {"query": "routed query", "top_k": 1})
    result = router.route_and_dispatch(engine, "original question", final_top_k=5)

    assert result.decision.fallback_used is False
    assert result.decision.tool_name == "search_raw"
    assert result.tool_result.candidates[0].text == "花帆: routed raw scene"
    assert captured[0]["question"] == "routed query"
    assert captured[0]["top_k"] == 1


def test_llm_router_trace_records_router_metadata_and_final_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = make_engine()
    engine.retrieval_config = RetrievalConfig(routing_mode="llm_router", final_top_k=5)
    engine.query_router = FixtureQueryRouter("search_raw", {"query": "routed query", "top_k": 1})

    def fake_retrieve_raw_nodes_with_trace(
        question: str,
        *,
        where: dict[str, Any] | None = None,
        top_k: int | None = None,
        n_results: int | None = None,
        analysis: Any = None,
    ) -> RetrievalTraceResult:
        return RetrievalTraceResult(nodes=[raw_node("routed raw scene")], stages={})

    monkeypatch.setattr(engine, "retrieve_raw_nodes_with_trace", fake_retrieve_raw_nodes_with_trace)
    monkeypatch.setattr(engine, "_fetch_raw_text", lambda metadata: "")

    trace = engine.retrieve_with_trace("original", query_id="q-router")

    assert trace.mode == "llm_router"
    assert trace.stages["router"].metadata["chosen_tool"] == "search_raw"
    assert trace.stages["router"].metadata["fallback_used"] is False
    assert trace.stages["final_top_k"].candidates is not None
    assert trace.stages["final_top_k"].candidates[0].text == "routed raw scene"


def test_llm_router_returns_direct_glossary_answer() -> None:
    engine = make_engine()
    engine.retrieval_config = RetrievalConfig(routing_mode="llm_router", final_top_k=5)
    engine.query_router = FixtureQueryRouter("lookup_glossary", {"term": "日野下花帆"})

    answer = engine.query("translate 日野下花帆")

    assert answer.startswith("日野下花帆 translates to Kaho Hinoshita.")
    assert "花帆" in answer

    trace = engine.retrieve_with_trace("translate 日野下花帆", answer_mode=True)

    assert trace.answer_text == answer
    assert trace.stages["router"].metadata["chosen_tool"] == "lookup_glossary"


def test_llm_router_surfaces_tool_errors_in_query_path() -> None:
    engine = make_engine()
    engine.retrieval_config = RetrievalConfig(routing_mode="llm_router", final_top_k=5)
    engine.query_router = FixtureQueryRouter(
        "get_scene",
        {"file_path": "missing.md", "scene_index": 0},
    )

    class FakeSourceStore:
        def get_scene(self, file_path: str, scene_index: int) -> None:
            return None

    engine.source_store = FakeSourceStore()

    answer = engine.query("show missing scene")

    assert "Insufficient source context" in answer
    assert "Tool error: scene not found" in answer


def test_fallback_dispatch_failure_returns_tool_result_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = make_engine()
    engine.query_router = FixtureQueryRouter(error=RuntimeError("model unavailable"))

    def fail_retrieve_raw_nodes_with_trace(
        question: str,
        *,
        where: dict[str, Any] | None = None,
        top_k: int | None = None,
        n_results: int | None = None,
        analysis: Any = None,
    ) -> RetrievalTraceResult:
        raise RuntimeError("retrieval unavailable")

    monkeypatch.setattr(engine, "retrieve_raw_nodes_with_trace", fail_retrieve_raw_nodes_with_trace)

    result = engine.query_router.route_and_dispatch(engine, "original", final_top_k=5)

    assert result.decision.fallback_used is True
    assert result.tool_result.candidates == []
    assert "fallback tool dispatch failed" in result.tool_result.errors[0]
    assert "retrieval unavailable" in result.tool_result.errors[0]


def test_fixture_router_falls_back_on_invalid_args_and_model_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = make_engine()
    captured: list[str] = []

    def fake_retrieve_raw_nodes_with_trace(
        question: str,
        *,
        where: dict[str, Any] | None = None,
        top_k: int | None = None,
        n_results: int | None = None,
        analysis: Any = None,
    ) -> RetrievalTraceResult:
        captured.append(f"{question}:{top_k}")
        return RetrievalTraceResult(nodes=[raw_node()], stages={})

    monkeypatch.setattr(engine, "retrieve_raw_nodes_with_trace", fake_retrieve_raw_nodes_with_trace)
    monkeypatch.setattr(engine, "_fetch_raw_text", lambda metadata: "")

    invalid_args = FixtureQueryRouter("search_raw", {"query": "", "top_k": 1})
    failed_model = FixtureQueryRouter(error=RuntimeError("model unavailable"))

    invalid_result = invalid_args.route_and_dispatch(engine, "original", final_top_k=5)
    failure_result = failed_model.route_and_dispatch(engine, "original", final_top_k=5)

    assert invalid_result.decision.fallback_used is True
    assert invalid_result.decision.fallback_reason == "invalid tool arguments"
    assert failure_result.decision.fallback_used is True
    assert failure_result.decision.fallback_reason == "router model failure"
    assert captured == ["original:5", "original:5"]
