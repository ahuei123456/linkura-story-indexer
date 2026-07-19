from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from inspect import Parameter, Signature
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
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
    ToolCandidate,
    ToolResult,
)

AGENT_REQUEST_LIMIT = 8
AgentStopReason = Literal["final_answer", "usage_limit", "model_error"]


class AgentAnswer(BaseModel):
    answer: str = Field(..., min_length=1)
    cited_labels: list[str] = Field(default_factory=list)


class AgentAnswerDraft(BaseModel):
    """The model-facing answer shape, whose citations use run-local evidence IDs."""

    answer: str = Field(..., min_length=1)
    cited_ids: list[str] = Field(default_factory=list)

    # Keeping extras out of the generated schema lets older fixture models that still return
    # cited_labels be read during the transition.  QueryAgent resolves those labels to IDs before
    # exposing the public answer shape.
    model_config = ConfigDict(extra="allow")


class EvidenceRegistry:
    """Assign stable, run-local IDs to the evidence candidates shown to the model."""

    def __init__(self, engine: Any | None = None) -> None:
        self.engine = engine
        self._candidates: dict[str, ToolCandidate] = {}
        self._ids_by_key: dict[tuple[Any, ...], str] = {}
        self._next_id = 1

    @staticmethod
    def _raw_span(metadata: dict[str, Any]) -> tuple[str, int, int] | None:
        file_path = metadata.get("file_path")
        scene_start = metadata.get("scene_start")
        scene_end = metadata.get("scene_end")
        if scene_start is None and isinstance(metadata.get("scene_index"), int):
            scene_start = metadata["scene_index"]
        if scene_end is None and isinstance(metadata.get("scene_index"), int):
            scene_end = metadata["scene_index"]
        if (
            not isinstance(file_path, str)
            or not file_path
            or not isinstance(scene_start, int)
            or not isinstance(scene_end, int)
            or scene_start < 0
            or scene_end < scene_start
        ):
            return None
        return file_path, scene_start, scene_end

    def _summary_episode_value(self, metadata: dict[str, Any]) -> str:
        helper = getattr(self.engine, "_summary_episode_value", None)
        if callable(helper):
            return str(helper(metadata))
        if metadata.get("summary_level") == 1:
            return "ALL_EPISODES"
        episode_number = metadata.get("episode_number")
        if isinstance(episode_number, int) and episode_number > 0:
            return str(episode_number)
        episode_name = metadata.get("episode_name")
        if isinstance(episode_name, str):
            match = re.search(r"第(\d+)話", episode_name)
            if match:
                return match.group(1)
        return str(metadata.get("episode", "unknown"))

    def _summary_part_label(self, metadata: dict[str, Any]) -> str:
        helper = getattr(self.engine, "_summary_part_label", None)
        if callable(helper):
            return str(helper(metadata))
        if metadata.get("summary_level") in {1, 2}:
            return "ALL_PARTS"
        part = metadata.get("part_name")
        return part if isinstance(part, str) and part else "unknown"

    def _identity_keys(self, candidate: ToolCandidate) -> list[tuple[Any, ...]]:
        metadata = candidate.metadata
        summary_level = metadata.get("summary_level")
        if summary_level in {1, 2, 3}:
            return [
                (
                    "summary",
                    metadata.get("arc_id"),
                    metadata.get("story_type"),
                    summary_level,
                    self._summary_episode_value(metadata),
                    self._summary_part_label(metadata),
                )
            ]

        keys: list[tuple[Any, ...]] = []
        chunk_id = metadata.get("chunk_id")
        if isinstance(chunk_id, str) and chunk_id:
            # The chunk ID is the preferred raw identity when it is available.
            keys.append(("raw", chunk_id))
        span = self._raw_span(metadata)
        if span is not None:
            # This alias joins a single-scene chunk to a get_scene result, which has no chunk ID.
            keys.append(("raw", *span))
        if not keys:
            # All normal evidence has one of the identities above.  Keep a same-object fallback
            # for malformed/tool-test candidates without accidentally merging unrelated results.
            keys.append(("object", id(candidate)))
        return keys

    def register(self, candidate: ToolCandidate) -> str:
        """Return the candidate's ID, reusing it for the same underlying evidence."""
        keys = self._identity_keys(candidate)
        evidence_id = next(
            (self._ids_by_key[key] for key in keys if key in self._ids_by_key),
            None,
        )
        if evidence_id is None:
            evidence_id = f"e{self._next_id}"
            self._next_id += 1
            self._candidates[evidence_id] = candidate
        elif (
            candidate.metadata.get("chunk_id")
            and not self._candidates[evidence_id].metadata.get("chunk_id")
        ):
            # If a get_scene span was registered first, retain the preferred chunk-backed
            # representation once the chunk result arrives.
            self._candidates[evidence_id] = candidate

        for key in keys:
            self._ids_by_key[key] = evidence_id
        return evidence_id

    def resolve(self, evidence_id: str) -> ToolCandidate | None:
        """Resolve a model-provided ID, returning None for unknown IDs."""
        return self._candidates.get(evidence_id)

    def items(self):
        return self._candidates.items()

    def source_block(self, candidate: ToolCandidate) -> dict[str, Any]:
        metadata = candidate.metadata
        if metadata.get("summary_level") in {1, 2, 3}:
            return {
                "arc_id": metadata.get("arc_id"),
                "story_type": metadata.get("story_type"),
                "summary_level": metadata.get("summary_level"),
                "episode": self._summary_episode_value(metadata),
                "part": self._summary_part_label(metadata),
            }

        span = self._raw_span(metadata)
        if span is None:
            file_path = metadata.get("file_path")
            scene_start = metadata.get("scene_start")
            scene_end = metadata.get("scene_end")
        else:
            file_path, scene_start, scene_end = span
        source: dict[str, Any] = {
            "file_path": file_path,
            "scene_start": scene_start,
            "scene_end": scene_end,
        }
        if isinstance(scene_start, int) and scene_start == scene_end:
            source["scene_index"] = scene_start
        return source


@dataclass
class AgentToolCall:
    step: int
    tool_name: str
    args: dict[str, Any]
    result: ToolResult
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentRunResult:
    answer: AgentAnswer
    tool_calls: list[AgentToolCall]
    stop_reason: AgentStopReason
    model_name: str
    registry: EvidenceRegistry = field(default_factory=EvidenceRegistry)
    request_count: int = 0
    fallback_reason: str | None = None
    model_cited_ids: list[str] = field(default_factory=list)
    cited_ids: list[str] = field(default_factory=list)
    unresolved_cited_ids: list[str] = field(default_factory=list)

    @property
    def recorder(self) -> list[AgentToolCall]:
        """Compatibility alias for callers that refer to the per-run recorder."""
        return self.tool_calls

    @property
    def evidence_registry(self) -> EvidenceRegistry:
        """Descriptive alias for callers that prefer the registry's full name."""
        return self.registry


def _agent_tool_payload(
    result: ToolResult,
    registry: EvidenceRegistry | None = None,
) -> dict[str, Any]:
    """Return the small, model-facing subset of a full tool result."""
    registry = registry or EvidenceRegistry()
    payload: dict[str, Any] = {}
    if result.candidates:
        payload["candidates"] = [
            {
                "id": registry.register(candidate),
                "text": candidate.text,
                "source": registry.source_block(candidate),
            }
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
    registry: EvidenceRegistry | None = None,
) -> Any:
    registry = registry or EvidenceRegistry(engine)

    def wrapper(args: BaseModel) -> dict[str, Any]:
        step = len(recorder) + 1
        validated_args = spec.input_model.model_validate(args)
        args_dict = validated_args.model_dump(mode="json")
        try:
            result = spec.dispatcher(engine, validated_args)
        except Exception as exc:  # Tool failures should be visible to the agent and trace.
            result = ToolResult(errors=[f"{spec.name} dispatch failed: {exc}"])
        payload = _agent_tool_payload(result, registry)
        recorder.append(
            AgentToolCall(
                step=step,
                tool_name=spec.name,
                args=args_dict,
                result=result,
                payload=payload,
            )
        )
        return payload

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
    registry: EvidenceRegistry | None = None,
) -> FunctionToolset:
    """Build an instrumented toolset from the shared typed query-tool registry."""
    registry = registry or EvidenceRegistry(engine)
    toolset = FunctionToolset()
    for spec in QUERY_TOOL_REGISTRY.values():
        toolset.add_function(
            _agent_tool_wrapper(engine, recorder, spec, registry),
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
        "answer must contain a concise answer and every short evidence ID that directly supports "
        "it in cited_ids. Stop once the evidence is sufficient.\n\n"
        "Tool strategy:\n"
        "- For Japanese character, unit, location, or story-specific terms, call "
        "lookup_glossary first when the term may be ambiguous; use its canonical term or "
        "translation in the next search.\n"
        "- Use vector_search_raw for exact dialogue and scene evidence. After vector_search_raw "
        "identifies a precise source, use get_scene for a focused follow-up when the answer depends "
        "on that exact scene; pass the returned file_path and a zero-based scene_index within the "
        "returned [scene_start, scene_end] span.\n"
        "- For the summary or recap of a specific year, episode, or part, MUST call get_summaries "
        "with location filters (for example, episode 3 means arc_id plus episode=3). Do not call "
        "vector_search_summaries for a known location; use it only to locate summaries by topic.\n"
        "- Use get_state for point-in-time roles, relationships, aliases, status, locations, "
        "or honorific facts; provide as_of_episode when the question specifies a time.\n"
        "- Quantitative rule: counting questions MUST call count_dialogue. Its SQL result is "
        "the only authoritative number; never estimate or infer a count from prose, and state "
        "the returned number verbatim in the answer.\n\n"
        "Tool results expose short IDs such as e1 and e2. Cite those IDs in cited_ids; never invent "
        "an ID and never cite a label that was not returned by a tool. If a tool returns no evidence, "
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
        self._active_registry: EvidenceRegistry | None = None

    def _instructions(self, engine: Any | None) -> str:
        return _agent_instructions(engine)

    def _run_model(
        self,
        question: str,
        *,
        engine: Any,
        final_top_k: int,
        recorder: list[Any],
    ) -> AgentAnswerDraft | AgentAnswer:
        registry = self._active_registry or EvidenceRegistry(engine)
        agent: Agent[None, AgentAnswerDraft] = Agent(
            self.model or create_agentic_generation_model(self.model_name),
            output_type=AgentAnswerDraft,
            instructions=self._instructions(engine),
        )
        result = agent.run_sync(
            json.dumps(
                {"question": question, "default_top_k": final_top_k},
                ensure_ascii=False,
            ),
            usage_limits=UsageLimits(request_limit=_configured_request_limit()),
            toolsets=[build_agent_toolset(engine, recorder, registry)],
        )
        self._last_request_count = result.usage().requests
        return result.output

    def _fallback_dispatch(
        self,
        engine: Any,
        question: str,
        final_top_k: int,
        recorder: list[AgentToolCall],
        registry: EvidenceRegistry | None = None,
    ) -> None:
        registry = registry or self._active_registry or EvidenceRegistry(engine)
        spec = QUERY_TOOL_REGISTRY["vector_search_raw"]
        args = SearchRawInput(query=question, top_k=final_top_k)
        step = len(recorder) + 1
        try:
            result = spec.dispatcher(engine, args)
        except Exception as exc:
            result = ToolResult(errors=[f"fallback tool dispatch failed: {exc}"])
        payload = _agent_tool_payload(result, registry)
        recorder.append(
            AgentToolCall(
                step=step,
                tool_name=spec.name,
                args=args.model_dump(mode="json"),
                result=result,
                payload=payload,
            )
        )

    @staticmethod
    def _ids_for_labels(registry: EvidenceRegistry, labels: Sequence[str]) -> list[str]:
        ids: list[str] = []
        for evidence_id, candidate in registry.items():
            if candidate.citation_label in labels and evidence_id not in ids:
                ids.append(evidence_id)
        return ids

    def _coerce_draft(
        self,
        answer: AgentAnswerDraft | AgentAnswer,
        registry: EvidenceRegistry,
    ) -> AgentAnswerDraft:
        if isinstance(answer, AgentAnswerDraft):
            draft = answer
            legacy_labels = getattr(draft, "model_extra", None) or {}
            cited_labels = legacy_labels.get("cited_labels")
            if isinstance(cited_labels, list):
                cited_ids = list(draft.cited_ids)
                for evidence_id in self._ids_for_labels(registry, cited_labels):
                    if evidence_id not in cited_ids:
                        cited_ids.append(evidence_id)
                if cited_ids != draft.cited_ids:
                    return AgentAnswerDraft(answer=draft.answer, cited_ids=cited_ids)
            return draft
        return AgentAnswerDraft(
            answer=answer.answer,
            cited_ids=self._ids_for_labels(registry, answer.cited_labels),
        )

    def _resolve_draft(
        self,
        draft: AgentAnswerDraft,
        registry: EvidenceRegistry,
    ) -> tuple[AgentAnswer, list[str], list[str], list[str]]:
        model_cited_ids = list(draft.cited_ids)
        resolved_ids: list[str] = []
        cited_labels: list[str] = []
        unresolved_ids: list[str] = []
        for evidence_id in model_cited_ids:
            candidate = registry.resolve(evidence_id)
            if candidate is None:
                unresolved_ids.append(evidence_id)
                continue
            if evidence_id in resolved_ids:
                continue
            resolved_ids.append(evidence_id)
            cited_labels.append(candidate.citation_label)
        return (
            AgentAnswer(answer=draft.answer, cited_labels=cited_labels),
            model_cited_ids,
            resolved_ids,
            unresolved_ids,
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
        registry = EvidenceRegistry(engine)
        self._active_registry = registry
        raw_answer: AgentAnswerDraft | AgentAnswer
        stop_reason: AgentStopReason
        fallback_reason: str | None = None
        try:
            raw_answer = self._run_model(
                question,
                engine=engine,
                final_top_k=final_top_k,
                recorder=recorder,
            )
            stop_reason = "final_answer"
        except UsageLimitExceeded as exc:
            fallback_reason = str(exc)
            if not recorder:
                self._fallback_dispatch(engine, question, final_top_k, recorder)
            raw_answer = AgentAnswerDraft(answer=_answer_from_tool_calls(recorder))
            stop_reason = "usage_limit"
        except Exception as exc:
            fallback_reason = str(exc)
            if not recorder:
                self._fallback_dispatch(engine, question, final_top_k, recorder)
            raw_answer = AgentAnswerDraft(answer=_answer_from_tool_calls(recorder))
            stop_reason = "model_error"
        finally:
            self._active_registry = None

        draft = self._coerce_draft(raw_answer, registry)
        answer, model_cited_ids, cited_ids, unresolved_ids = self._resolve_draft(draft, registry)
        return AgentRunResult(
            answer=answer,
            tool_calls=recorder,
            stop_reason=stop_reason,
            model_name=self.model_name,
            registry=registry,
            request_count=self._last_request_count,
            fallback_reason=fallback_reason,
            model_cited_ids=model_cited_ids,
            cited_ids=cited_ids,
            unresolved_cited_ids=unresolved_ids,
        )


class FixtureQueryAgent(QueryAgent):
    """Deterministic scripted agent that still exercises real query dispatchers."""

    def __init__(
        self,
        calls: Sequence[tuple[str, dict[str, Any]]] | None = None,
        answer: str | AgentAnswer | AgentAnswerDraft = "fixture answer",
        *,
        script: Sequence[tuple[str, dict[str, Any]]] | None = None,
        scripted_calls: Sequence[tuple[str, dict[str, Any]]] | None = None,
        tool_calls: Sequence[tuple[str, dict[str, Any]]] | None = None,
        final_answer: str | AgentAnswer | AgentAnswerDraft | None = None,
        canned_answer: str | AgentAnswer | AgentAnswerDraft | None = None,
        cited_labels: Sequence[str] | None = None,
        cited_ids: Sequence[str] | None = None,
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
        self._draft = answer if isinstance(answer, AgentAnswerDraft) else None
        if isinstance(answer, AgentAnswerDraft):
            self._answer = AgentAnswer(answer=answer.answer)
        elif isinstance(answer, AgentAnswer):
            self._answer = answer
        else:
            self._answer = AgentAnswer(answer=answer, cited_labels=list(cited_labels or []))
        self._cited_ids = list(cited_ids or [])
        self._error = error

    def _run_model(
        self,
        question: str,
        *,
        engine: Any,
        final_top_k: int,
        recorder: list[Any],
    ) -> AgentAnswerDraft | AgentAnswer:
        if self._error is not None:
            raise self._error
        registry = self._active_registry or EvidenceRegistry(engine)
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
            payload = _agent_tool_payload(result, registry)
            recorder.append(
                AgentToolCall(
                    step=step,
                    tool_name=tool_name,
                    args=args.model_dump(mode="json"),
                    result=result,
                    payload=payload,
                )
            )
        if self._draft is not None:
            return self._draft
        if self._cited_ids:
            return AgentAnswerDraft(answer=self._answer.answer, cited_ids=self._cited_ids)
        return AgentAnswerDraft(
            answer=self._answer.answer,
            cited_ids=QueryAgent._ids_for_labels(
                registry,
                self._answer.cited_labels,
            ),
        )
