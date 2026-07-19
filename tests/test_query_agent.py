from __future__ import annotations

import json
from typing import Any

from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from linkura_story_indexer.eval.io import stable_json
from linkura_story_indexer.query import engine as query_engine
from linkura_story_indexer.query.agent import (
    AgentAnswer,
    AgentAnswerDraft,
    AgentToolCall,
    EvidenceRegistry,
    FixtureQueryAgent,
    QueryAgent,
    _agent_tool_payload,
    build_agent_toolset,
)
from linkura_story_indexer.query.engine import (
    RetrievalConfig,
    RetrievalTraceResult,
    StoryQueryEngine,
)
from linkura_story_indexer.query.tools import ToolCandidate, ToolResult


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


def summary_candidate(summary_level: int = 2, part_name: str = "1") -> ToolCandidate:
    metadata = {
        "arc_id": "103",
        "story_type": "Main",
        "episode_name": "第1話『花咲きたい！』",
        "episode_number": 1,
        "part_name": part_name,
        "summary_level": summary_level,
    }
    return ToolCandidate(
        text="summary evidence",
        citation_label="103 · Main · Episode 1 · Part ALL_PARTS · summary_level 2",
        metadata=metadata,
        rank=1,
    )


def test_evidence_registry_reuses_raw_spans_and_summary_tiers() -> None:
    engine = make_engine()
    registry = EvidenceRegistry(engine)
    raw_metadata = raw_node()[1]
    chunk_candidate = ToolCandidate(
        text="chunk evidence",
        citation_label="raw",
        metadata=raw_metadata,
        rank=1,
    )
    scene_candidate = ToolCandidate(
        text="scene evidence",
        citation_label="raw",
        metadata={key: value for key, value in raw_metadata.items() if key != "chunk_id"},
        rank=1,
    )

    assert registry.register(chunk_candidate) == "e1"
    assert registry.register(scene_candidate) == "e1"
    assert registry.register(summary_candidate()) == "e2"
    assert registry.register(summary_candidate()) == "e2"
    assert registry.register(summary_candidate(summary_level=3, part_name="1")) == "e3"
    assert registry.resolve("missing") is None


def test_evidence_registry_reuses_summary_search_and_location_results() -> None:
    engine = make_engine()
    registry = EvidenceRegistry(engine)
    vector_candidate = summary_candidate()
    fetched_candidate = ToolCandidate(
        text="fetched summary evidence",
        citation_label="same summary tier",
        metadata={
            "arc_id": "103",
            "story_type": "Main",
            "summary_level": 2,
            "episode_number": 1,
        },
        rank=1,
    )

    assert registry.register(vector_candidate) == "e1"
    assert registry.register(fetched_candidate) == "e1"


def test_agent_payload_uses_ids_and_compact_source_blocks() -> None:
    engine = make_engine()
    registry = EvidenceRegistry(engine)
    raw = ToolCandidate(
        text="raw evidence",
        citation_label="must not be exposed",
        metadata=raw_node()[1],
        rank=1,
    )
    summary = summary_candidate()

    payload = _agent_tool_payload(ToolResult(candidates=[raw, summary]), registry)

    assert payload["candidates"] == [
        {
            "id": "e1",
            "text": "raw evidence",
            "source": {
                "file_path": "story/103/第1話『花咲きたい！』/1.md",
                "scene_start": 0,
                "scene_end": 0,
                "scene_index": 0,
            },
        },
        {
            "id": "e2",
            "text": "summary evidence",
            "source": {
                "arc_id": "103",
                "story_type": "Main",
                "summary_level": 2,
                "episode": "1",
                "part": "ALL_PARTS",
            },
        },
    ]
    assert all("citation_label" not in candidate for candidate in payload["candidates"])


def test_agent_run_resolves_cited_ids_and_drops_unknown_ids(monkeypatch: Any) -> None:
    engine = make_engine()

    def fake_retrieve(*args: Any, **kwargs: Any) -> RetrievalTraceResult:
        return RetrievalTraceResult(nodes=[raw_node()], stages={})

    monkeypatch.setattr(engine, "retrieve_raw_nodes_with_trace", fake_retrieve)
    monkeypatch.setattr(engine, "_fetch_raw_text", lambda metadata: "")
    agent = FixtureQueryAgent(
        calls=[("vector_search_raw", {"query": "question", "top_k": 1})],
        answer=AgentAnswerDraft(answer="answered", cited_ids=["e1", "e404"]),
    )

    result = agent.run(engine, "question")

    assert result.answer.cited_labels == ["103 · Episode 1 · Part 1 · Scene 1"]
    assert result.cited_ids == ["e1"]
    assert result.unresolved_cited_ids == ["e404"]


def test_agent_trace_selects_only_model_cited_candidates(monkeypatch: Any) -> None:
    engine = make_engine()

    def fake_retrieve(*args: Any, **kwargs: Any) -> RetrievalTraceResult:
        return RetrievalTraceResult(nodes=[raw_node("first"), raw_node("second", 1)], stages={})

    monkeypatch.setattr(engine, "retrieve_raw_nodes_with_trace", fake_retrieve)
    monkeypatch.setattr(engine, "_fetch_raw_text", lambda metadata: "")
    engine.query_agent = FixtureQueryAgent(
        calls=[("vector_search_raw", {"query": "question", "top_k": 2})],
        answer=AgentAnswerDraft(answer="answered", cited_ids=["e2"]),
    )

    trace = engine.retrieve_with_trace("question", answer_mode=True)

    assert [candidate.text for candidate in trace.stages["final_top_k"].candidates or []] == [
        "second"
    ]
    assert trace.final_citation_labels == ["103 · Episode 1 · Part 1 · Scene 2"]
    tool_payload = trace.stages["agent"].metadata["tool_calls"][0]
    assert [candidate["id"] for candidate in tool_payload["candidates"]] == ["e1", "e2"]
    assert trace.stages["agent"].metadata["model_cited_ids"] == ["e2"]


def test_successful_uncited_answer_keeps_final_evidence_empty(monkeypatch: Any) -> None:
    engine = make_engine()

    def fake_retrieve(*args: Any, **kwargs: Any) -> RetrievalTraceResult:
        return RetrievalTraceResult(nodes=[raw_node()], stages={})

    monkeypatch.setattr(engine, "retrieve_raw_nodes_with_trace", fake_retrieve)
    monkeypatch.setattr(engine, "_fetch_raw_text", lambda metadata: "")
    engine.query_agent = FixtureQueryAgent(
        calls=[("vector_search_raw", {"query": "question", "top_k": 1})],
        answer=AgentAnswerDraft(answer="insufficient context", cited_ids=[]),
    )

    trace = engine.retrieve_with_trace("question", answer_mode=True)

    assert trace.answer_text == "insufficient context"
    assert trace.stages["final_top_k"].candidates == []
    assert trace.final_citation_labels == []


def test_function_model_can_follow_raw_payload_into_get_scene_and_cite_id(
    monkeypatch: Any,
) -> None:
    engine = make_engine()
    node = raw_node("search scene")

    def fake_retrieve(*args: Any, **kwargs: Any) -> RetrievalTraceResult:
        return RetrievalTraceResult(nodes=[node], stages={})

    class Store:
        def get_scene(self, file_path: str, scene_index: int) -> dict[str, Any]:
            assert file_path == node[1]["file_path"]
            assert scene_index == 0
            return {
                "text": "focused scene",
                "file_path": file_path,
                "scene_index": scene_index,
                "metadata": {key: value for key, value in node[1].items() if key != "chunk_id"},
            }

    monkeypatch.setattr(engine, "retrieve_raw_nodes_with_trace", fake_retrieve)
    monkeypatch.setattr(engine, "_fetch_raw_text", lambda metadata: "")
    engine.source_store = Store()

    def tool_return_payload(messages: list[Any]) -> dict[str, Any]:
        for message in reversed(messages):
            for part in getattr(message, "parts", []):
                if getattr(part, "part_kind", None) != "tool-return":
                    continue
                content = part.content
                return json.loads(content) if isinstance(content, str) else content
        raise AssertionError("expected a tool return")

    def scripted_model(messages: list[Any], info: Any) -> ModelResponse:
        tool_returns = sum(
            1
            for message in messages
            for part in getattr(message, "parts", [])
            if getattr(part, "part_kind", None) == "tool-return"
        )
        if tool_returns == 0:
            return ModelResponse(
                parts=[ToolCallPart(tool_name="vector_search_raw", args={"query": "scene"})]
            )
        if tool_returns == 1:
            payload = tool_return_payload(messages)
            source = payload["candidates"][0]["source"]
            assert "citation_label" not in payload["candidates"][0]
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="get_scene",
                        args={
                            "file_path": source["file_path"],
                            "scene_index": source["scene_index"],
                        },
                    )
                ]
            )
        payload = tool_return_payload(messages)
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args={"answer": "focused answer", "cited_ids": [payload["candidates"][0]["id"]]},
                )
            ]
        )

    engine.query_agent = QueryAgent(
        model=FunctionModel(scripted_model),
        model_name="function-agent",
    )

    trace = engine.retrieve_with_trace("What happened?", answer_mode=True)

    assert trace.answer_text == "focused answer"
    assert trace.final_citation_labels == ["103 · Episode 1 · Part 1 · Scene 1"]
    calls = trace.stages["agent"].metadata["tool_calls"]
    assert calls[0]["candidates"][0]["id"] == "e1"
    assert calls[1]["candidates"][0]["id"] == "e1"


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
        "vector_search_raw",
        "vector_search_summaries",
        "get_summaries",
        "get_scene",
        "lookup_glossary",
        "get_state",
        "count_dialogue",
    }
    assert "speaker" in by_name["count_dialogue"].parameters_json_schema["properties"]


def test_query_agent_builds_model_with_agentic_factory(monkeypatch: Any) -> None:
    engine = make_engine()
    test_model = TestModel(
        call_tools=[],
        custom_output_args={"answer": "agent answer", "cited_labels": []},
    )
    calls: list[str | None] = []

    def fake_create_agentic_generation_model(model_name: str | None = None) -> TestModel:
        calls.append(model_name)
        return test_model

    monkeypatch.setattr(
        "linkura_story_indexer.query.agent.create_agentic_generation_model",
        fake_create_agentic_generation_model,
    )

    result = QueryAgent(model_name="gpt-5.6-luna").run(engine, "question")

    assert calls == ["gpt-5.6-luna"]
    assert result.stop_reason == "final_answer"
    assert result.answer.answer == "agent answer"


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
                        tool_name="vector_search_raw",
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
    ] == ["lookup_glossary", "vector_search_raw"]
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
            recorder: list[AgentToolCall],
            registry: EvidenceRegistry,
        ) -> AgentAnswer:
            from pydantic_ai.exceptions import UsageLimitExceeded

            raise UsageLimitExceeded("request cap")

    engine.query_agent = FailingAgent(model_name="limited")
    result = engine.query_agent.run(engine, "question", final_top_k=5)

    assert result.stop_reason == "usage_limit"
    assert result.tool_calls[0].tool_name == "vector_search_raw"
    assert result.fallback_reason == "request cap"


def test_fixture_agent_traces_are_stable_across_runs(monkeypatch: Any) -> None:
    engine = make_engine()

    def fake_retrieve(*args: Any, **kwargs: Any) -> RetrievalTraceResult:
        return RetrievalTraceResult(nodes=[raw_node()], stages={})

    monkeypatch.setattr(engine, "retrieve_raw_nodes_with_trace", fake_retrieve)
    monkeypatch.setattr(engine, "_fetch_raw_text", lambda metadata: "")
    engine.query_agent = FixtureQueryAgent(
        calls=[("vector_search_raw", {"query": "question", "top_k": 1})],
        answer="fixture answer",
    )

    first = engine.retrieve_with_trace("question", query_id="q", answer_mode=True)
    second = engine.retrieve_with_trace("question", query_id="q", answer_mode=True)

    assert stable_json(first) == stable_json(second)
