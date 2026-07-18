from __future__ import annotations

from typing import Any

from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from linkura_story_indexer.eval.io import stable_json
from linkura_story_indexer.query import engine as query_engine
from linkura_story_indexer.query.agent import (
    AgentAnswer,
    FixtureQueryAgent,
    QueryAgent,
    build_agent_toolset,
)
from linkura_story_indexer.query.engine import (
    RetrievalConfig,
    RetrievalTraceResult,
    StoryQueryEngine,
)


def make_engine() -> StoryQueryEngine:
    engine = StoryQueryEngine.__new__(StoryQueryEngine)
    engine.retrieval_config = RetrievalConfig(routing_mode="agentic", final_top_k=5)
    engine.glossary = {"characters": {"日野下花帆": "Kaho Hinoshita"}}
    engine.state_ledger = {}
    return engine


def raw_node(text: str = "raw evidence", scene_index: int = 0) -> tuple[str, dict[str, Any]]:
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
            "scene_index": scene_index,
            "scene_start": scene_index,
            "scene_end": scene_index,
            "canonical_story_order": scene_index + 1,
            "chunk_id": f"chunk-103-{scene_index}",
        },
    )


def test_agent_toolset_registers_the_shared_typed_schemas() -> None:
    engine = make_engine()
    recorder = []
    toolset = build_agent_toolset(engine, recorder)
    test_model = TestModel()

    from pydantic_ai import Agent

    Agent(test_model).run_sync("list tools", toolsets=[toolset])

    request_parameters = test_model.last_model_request_parameters
    assert request_parameters is not None
    by_name = {tool.name: tool for tool in request_parameters.function_tools}
    assert set(by_name) == {
        "search_raw",
        "search_summaries",
        "get_scene",
        "lookup_glossary",
        "get_state",
        "count_dialogue",
    }
    assert "speaker" in by_name["count_dialogue"].parameters_json_schema["properties"]


def test_function_model_agent_records_multi_step_calls_and_reranks_candidates(
    monkeypatch: Any,
) -> None:
    engine = make_engine()

    def fake_retrieve(
        question: str,
        *,
        where: dict[str, Any] | None = None,
        top_k: int | None = None,
        n_results: int | None = None,
        analysis: Any = None,
    ) -> RetrievalTraceResult:
        return RetrievalTraceResult(nodes=[raw_node("search evidence")], stages={})

    monkeypatch.setattr(engine, "retrieve_raw_nodes_with_trace", fake_retrieve)
    monkeypatch.setattr(engine, "_fetch_raw_text", lambda metadata: "")

    def scripted_model(messages: list[Any], info: Any) -> ModelResponse:
        tool_returns = sum(
            1
            for message in messages
            for part in getattr(message, "parts", [])
            if getattr(part, "part_kind", None) == "tool-return"
        )
        if tool_returns == 0:
            return ModelResponse(
                parts=[ToolCallPart(tool_name="lookup_glossary", args={"term": "花帆"})]
            )
        if tool_returns == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="search_raw",
                        args={"query": "Kaho", "top_k": 1},
                    )
                ]
            )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args={
                        "answer": "The evidence supports this answer.",
                        "cited_labels": [
                            "103 · Episode 1 · Part 1 · Scene 1",
                        ],
                    },
                )
            ]
        )

    engine.query_agent = QueryAgent(
        model=FunctionModel(scripted_model),
        model_name="function-agent",
    )

    trace = engine.retrieve_with_trace("What happened?", answer_mode=True)

    assert trace.answer_text == "The evidence supports this answer."
    assert trace.stages["agent"].metadata["stop_reason"] == "final_answer"
    assert [
        call["tool_name"] for call in trace.stages["agent"].metadata["tool_calls"]
    ] == ["lookup_glossary", "search_raw"]
    assert trace.stages["final_top_k"].candidates is not None
    assert trace.stages["final_top_k"].candidates[0].rank == 1
    assert trace.final_citation_labels == ["103 · Episode 1 · Part 1 · Scene 1"]


def test_agent_counting_uses_the_sql_tool_result() -> None:
    engine = make_engine()

    class Store:
        def count_turns(
            self,
            speaker: str,
            *,
            arc_id: str | None = None,
            episode: int | None = None,
            part: str | None = None,
        ) -> int:
            return 24

    engine.source_store = Store()
    engine.query_agent = FixtureQueryAgent(
        calls=[("count_dialogue", {"speaker": "花帆", "arc_id": "103"})],
        answer=AgentAnswer(answer="花帆 has 24 dialogue turns."),
    )

    trace = engine.retrieve_with_trace("How many turns?", answer_mode=True)

    assert "24" in (trace.answer_text or "")
    assert trace.stages["agent"].metadata["tool_calls"][0]["tool_name"] == "count_dialogue"


def test_usage_limit_falls_back_to_one_raw_search(monkeypatch: Any) -> None:
    engine = make_engine()
    engine.source_store = object()

    monkeypatch.setattr(
        query_engine,
        "INSUFFICIENT_SOURCE_CONTEXT",
        "fallback context unavailable",
    )

    class FailingAgent(QueryAgent):
        def _run_model(
            self,
            question: str,
            *,
            engine: Any,
            final_top_k: int,
            recorder: list[Any],
        ) -> AgentAnswer:
            from pydantic_ai.exceptions import UsageLimitExceeded

            raise UsageLimitExceeded("request cap")

    engine.query_agent = FailingAgent(model_name="limited")
    result = engine.query_agent.run(engine, "question", final_top_k=5)

    assert result.stop_reason == "usage_limit"
    assert result.tool_calls[0].tool_name == "search_raw"
    assert result.fallback_reason == "request cap"


def test_fixture_agent_traces_are_stable_across_runs(monkeypatch: Any) -> None:
    engine = make_engine()

    def fake_retrieve(*args: Any, **kwargs: Any) -> RetrievalTraceResult:
        return RetrievalTraceResult(nodes=[raw_node()], stages={})

    monkeypatch.setattr(engine, "retrieve_raw_nodes_with_trace", fake_retrieve)
    monkeypatch.setattr(engine, "_fetch_raw_text", lambda metadata: "")
    engine.query_agent = FixtureQueryAgent(
        calls=[("search_raw", {"query": "question", "top_k": 1})],
        answer="fixture answer",
    )

    first = engine.retrieve_with_trace("question", query_id="q", answer_mode=True)
    second = engine.retrieve_with_trace("question", query_id="q", answer_mode=True)

    assert stable_json(first) == stable_json(second)
