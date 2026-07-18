from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from inspect import Parameter, Signature
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent, FunctionToolset, UsageLimits
from pydantic_ai.exceptions import UsageLimitExceeded

from linkura_story_indexer.database import (
    create_agentic_generation_model,
    get_generation_model_name,
)
from linkura_story_indexer.query.router import _story_location_catalog
from linkura_story_indexer.query.tools import (
    QUERY_TOOL_REGISTRY,
    QueryToolSpec,
    SearchRawInput,
    ToolResult,
)

AGENT_REQUEST_LIMIT = 8
AgentStopReason = Literal["final_answer", "usage_limit", "model_error"]


class AgentAnswer(BaseModel):
    answer: str = Field(..., min_length=1)
    cited_labels: list[str] = Field(default_factory=list)


@dataclass
class AgentToolCall:
    step: int
    tool_name: str
    args: dict[str, Any]
    result: ToolResult


@dataclass
class AgentRunResult:
    answer: AgentAnswer
    tool_calls: list[AgentToolCall]
    stop_reason: AgentStopReason
    model_name: str
    request_count: int = 0
    fallback_reason: str | None = None

    @property
    def recorder(self) -> list[AgentToolCall]:
        """Compatibility alias for callers that refer to the per-run recorder."""
        return self.tool_calls


def _agent_tool_payload(result: ToolResult) -> dict[str, Any]:
    """Return the small, model-facing subset of a full tool result."""
    payload: dict[str, Any] = {}
    if result.candidates:
        payload["candidates"] = [
            {"citation_label": candidate.citation_label, "text": candidate.text}
            for candidate in result.candidates
        ]

    tool_output = result.metadata.get("tool_output")
    if tool_output is not None:
        payload["tool_output"] = tool_output
    direct_answer = result.metadata.get("direct_answer")
    if isinstance(direct_answer, str):
        payload["direct_answer"] = direct_answer
    if result.warnings:
        payload["warnings"] = list(result.warnings)
    if result.errors:
        payload["errors"] = list(result.errors)
    return payload


def _agent_tool_wrapper(
    engine: Any,
    recorder: list[AgentToolCall],
    spec: QueryToolSpec,
) -> Any:
    def wrapper(args: BaseModel) -> dict[str, Any]:
        step = len(recorder) + 1
        validated_args = spec.input_model.model_validate(args)
        args_dict = validated_args.model_dump(mode="json")
        try:
            result = spec.dispatcher(engine, validated_args)
        except Exception as exc:  # Tool failures should be visible to the agent and trace.
            result = ToolResult(errors=[f"{spec.name} dispatch failed: {exc}"])
        recorder.append(
            AgentToolCall(
                step=step,
                tool_name=spec.name,
                args=args_dict,
                result=result,
            )
        )
        return _agent_tool_payload(result)

    wrapper.__name__ = spec.name
    wrapper.__qualname__ = spec.name
    wrapper.__doc__ = spec.description
    wrapper.__annotations__ = {"args": spec.input_model, "return": dict[str, Any]}
    setattr(
        wrapper,
        "__signature__",
        Signature(
            [
                Parameter(
                    "args",
                    kind=Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=spec.input_model,
                )
            ],
            return_annotation=dict[str, Any],
        ),
    )
    return wrapper


def build_agent_toolset(
    engine: Any,
    recorder: list[AgentToolCall],
) -> FunctionToolset:
    """Build an instrumented toolset from the shared typed query-tool registry."""
    toolset = FunctionToolset()
    for spec in QUERY_TOOL_REGISTRY.values():
        toolset.add_function(
            _agent_tool_wrapper(engine, recorder, spec),
            name=spec.name,
            description=spec.description,
        )
    return toolset


def _agent_instructions(engine: Any | None = None) -> str:
    try:
        story_location_catalog = _story_location_catalog(engine)
    except Exception:
        story_location_catalog = "Available numbered episodes by arc are unavailable."

    return (
        "You are the multi-step retrieval agent for a source-grounded story index. "
        "Use the registered query tools to gather evidence before answering. Never invent a "
        "source, name, translation, state fact, honorific, or number. Your final structured "
        "answer must contain a concise answer and every citation label that directly supports "
        "it in cited_labels. Stop once the evidence is sufficient.\n\n"
        "Tool strategy:\n"
        "- For Japanese character, unit, location, or story-specific terms, call "
        "lookup_glossary first when the term may be ambiguous; use its canonical term or "
        "translation in the next search.\n"
        "- Use vector_search_raw for exact dialogue and scene evidence. After vector_search_raw "
        "identifies a precise file_path and scene_index, use get_scene for a focused follow-up when the "
        "answer depends on that exact scene.\n"
        "- For the summary or recap of a specific year, episode, or part, MUST call get_summaries "
        "with location filters (for example, episode 3 means arc_id plus episode=3). Do not call "
        "vector_search_summaries for a known location; use it only to locate summaries by topic.\n"
        "- Use get_state for point-in-time roles, relationships, aliases, status, locations, "
        "or honorific facts; provide as_of_episode when the question specifies a time.\n"
        "- Quantitative rule: counting questions MUST call count_dialogue. Its SQL result is "
        "the only authoritative number; never estimate or infer a count from prose, and state "
        "the returned number verbatim in the answer.\n\n"
        "Do not cite a label that was not returned by a tool. If a tool returns no evidence, "
        "explain that the source context is insufficient rather than guessing.\n\n"
        f"{story_location_catalog}\n\n"
        "The user prompt is JSON containing question and default_top_k."
    )


def _configured_request_limit() -> int:
    raw_limit = os.getenv("LINKURA_AGENT_REQUEST_LIMIT")
    if raw_limit is None:
        return AGENT_REQUEST_LIMIT
    try:
        limit = int(raw_limit)
    except ValueError:
        return AGENT_REQUEST_LIMIT
    return limit if limit > 0 else AGENT_REQUEST_LIMIT


def _answer_from_tool_calls(tool_calls: Sequence[AgentToolCall]) -> str:
    for call in reversed(tool_calls):
        direct_answer = call.result.metadata.get("direct_answer")
        if isinstance(direct_answer, str) and direct_answer.strip():
            return direct_answer.strip()

    evidence_lines = []
    for call in tool_calls:
        for candidate in call.result.candidates:
            evidence_lines.append(f"[{candidate.citation_label}] {candidate.text}")
    if evidence_lines:
        return "Evidence returned by the query tools:\n" + "\n".join(evidence_lines)

    for call in reversed(tool_calls):
        if call.result.errors:
            return (
                "Insufficient source context: the query tool failed. "
                + call.result.errors[0]
            )
        if call.result.warnings:
            return (
                "Insufficient source context: the query tool returned a warning. "
                + call.result.warnings[0]
            )
    return "Insufficient source context: no query-tool evidence was returned."


class QueryAgent:
    def __init__(self, model_name: str | None = None, *, model: Any | None = None) -> None:
        self.model_name = model_name or get_generation_model_name()
        self.model = model
        self._last_request_count = 0

    def _instructions(self, engine: Any | None) -> str:
        return _agent_instructions(engine)

    def _run_model(
        self,
        question: str,
        *,
        engine: Any,
        final_top_k: int,
        recorder: list[AgentToolCall],
    ) -> AgentAnswer:
        agent: Agent[None, AgentAnswer] = Agent(
            self.model or create_agentic_generation_model(self.model_name),
            output_type=AgentAnswer,
            instructions=self._instructions(engine),
        )
        result = agent.run_sync(
            json.dumps(
                {"question": question, "default_top_k": final_top_k},
                ensure_ascii=False,
            ),
            usage_limits=UsageLimits(request_limit=_configured_request_limit()),
            toolsets=[build_agent_toolset(engine, recorder)],
        )
        self._last_request_count = result.usage().requests
        return result.output

    def _fallback_dispatch(
        self,
        engine: Any,
        question: str,
        final_top_k: int,
        recorder: list[AgentToolCall],
    ) -> None:
        spec = QUERY_TOOL_REGISTRY["vector_search_raw"]
        args = SearchRawInput(query=question, top_k=final_top_k)
        step = len(recorder) + 1
        try:
            result = spec.dispatcher(engine, args)
        except Exception as exc:
            result = ToolResult(errors=[f"fallback tool dispatch failed: {exc}"])
        recorder.append(
            AgentToolCall(
                step=step,
                tool_name=spec.name,
                args=args.model_dump(mode="json"),
                result=result,
            )
        )

    def run(
        self,
        engine: Any,
        question: str,
        *,
        final_top_k: int = 8,
    ) -> AgentRunResult:
        recorder: list[AgentToolCall] = []
        self._last_request_count = 0
        try:
            answer = self._run_model(
                question,
                engine=engine,
                final_top_k=final_top_k,
                recorder=recorder,
            )
        except UsageLimitExceeded as exc:
            reason = str(exc)
            if not recorder:
                self._fallback_dispatch(engine, question, final_top_k, recorder)
            answer = AgentAnswer(answer=_answer_from_tool_calls(recorder))
            return AgentRunResult(
                answer=answer,
                tool_calls=recorder,
                stop_reason="usage_limit",
                model_name=self.model_name,
                request_count=self._last_request_count,
                fallback_reason=reason,
            )
        except Exception as exc:
            reason = str(exc)
            if not recorder:
                self._fallback_dispatch(engine, question, final_top_k, recorder)
            answer = AgentAnswer(answer=_answer_from_tool_calls(recorder))
            return AgentRunResult(
                answer=answer,
                tool_calls=recorder,
                stop_reason="model_error",
                model_name=self.model_name,
                request_count=self._last_request_count,
                fallback_reason=reason,
            )

        return AgentRunResult(
            answer=answer,
            tool_calls=recorder,
            stop_reason="final_answer",
            model_name=self.model_name,
            request_count=self._last_request_count,
        )


class FixtureQueryAgent(QueryAgent):
    """Deterministic scripted agent that still exercises real query dispatchers."""

    def __init__(
        self,
        calls: Sequence[tuple[str, dict[str, Any]]] | None = None,
        answer: str | AgentAnswer = "fixture answer",
        *,
        script: Sequence[tuple[str, dict[str, Any]]] | None = None,
        scripted_calls: Sequence[tuple[str, dict[str, Any]]] | None = None,
        tool_calls: Sequence[tuple[str, dict[str, Any]]] | None = None,
        final_answer: str | AgentAnswer | None = None,
        canned_answer: str | AgentAnswer | None = None,
        cited_labels: Sequence[str] | None = None,
        model_name: str = "fixture-agent",
        error: Exception | None = None,
    ) -> None:
        super().__init__(model_name=model_name)
        selected_calls = calls or script or scripted_calls or tool_calls or []
        self.calls = list(selected_calls)
        if final_answer is not None:
            answer = final_answer
        if canned_answer is not None:
            answer = canned_answer
        self._answer = (
            answer
            if isinstance(answer, AgentAnswer)
            else AgentAnswer(answer=answer, cited_labels=list(cited_labels or []))
        )
        self._error = error

    def _run_model(
        self,
        question: str,
        *,
        engine: Any,
        final_top_k: int,
        recorder: list[AgentToolCall],
    ) -> AgentAnswer:
        if self._error is not None:
            raise self._error
        for tool_name, raw_args in self.calls:
            spec = QUERY_TOOL_REGISTRY.get(tool_name)
            if spec is None:
                raise ValueError(f"unknown fixture tool: {tool_name}")
            args = spec.input_model.model_validate(raw_args)
            step = len(recorder) + 1
            try:
                result = spec.dispatcher(engine, args)
            except Exception as exc:
                result = ToolResult(errors=[f"{tool_name} dispatch failed: {exc}"])
            recorder.append(
                AgentToolCall(
                    step=step,
                    tool_name=tool_name,
                    args=args.model_dump(mode="json"),
                    result=result,
                )
            )
        return self._answer
