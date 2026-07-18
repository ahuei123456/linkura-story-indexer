from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, Field, ValidationError
from pydantic_ai import Agent

from linkura_story_indexer.database import create_generation_model, get_router_model_name
from linkura_story_indexer.query.tools import (
    QUERY_TOOL_REGISTRY,
    SearchRawInput,
    ToolResult,
)


class RouterOutput(BaseModel):
    tool_name: str = Field(..., min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class RouterDecision:
    tool_name: str
    validated_args: BaseModel
    raw_output: dict[str, Any] | None
    fallback_used: bool
    fallback_reason: str | None
    router_model: str
    validation_errors: list[str]

    def metadata(self) -> dict[str, Any]:
        return {
            "chosen_tool": self.tool_name,
            "validated_args": self.validated_args.model_dump(mode="json"),
            "raw_structured_model_output": self.raw_output,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "router_model": self.router_model,
            "validation_errors": self.validation_errors,
        }


@dataclass(frozen=True)
class RouterDispatchResult:
    decision: RouterDecision
    tool_result: ToolResult


def _engine_cache_key(engine: Any | None) -> int | None:
    return id(engine) if engine is not None else None


def _fallback_decision(
    question: str,
    *,
    final_top_k: int,
    router_model: str,
    raw_output: dict[str, Any] | None,
    reason: str,
    validation_errors: list[str],
) -> RouterDecision:
    return RouterDecision(
        tool_name="vector_search_raw",
        validated_args=SearchRawInput(query=question, top_k=final_top_k),
        raw_output=raw_output,
        fallback_used=True,
        fallback_reason=reason,
        router_model=router_model,
        validation_errors=validation_errors,
    )


def _raw_output(output: RouterOutput | None) -> dict[str, Any] | None:
    if output is None:
        return None
    return output.model_dump(mode="json")


def validate_router_output(
    output: RouterOutput,
    *,
    question: str,
    final_top_k: int,
    router_model: str | None = None,
) -> RouterDecision:
    model_name = router_model or get_router_model_name()
    raw_output = _raw_output(output)
    spec = QUERY_TOOL_REGISTRY.get(output.tool_name)
    if spec is None:
        return _fallback_decision(
            question,
            final_top_k=final_top_k,
            router_model=model_name,
            raw_output=raw_output,
            reason="unknown tool name",
            validation_errors=[f"unknown tool name: {output.tool_name}"],
        )

    try:
        validated_args = spec.input_model.model_validate(output.args)
    except ValidationError as exc:
        return _fallback_decision(
            question,
            final_top_k=final_top_k,
            router_model=model_name,
            raw_output=raw_output,
            reason="invalid tool arguments",
            validation_errors=[str(exc)],
        )

    return RouterDecision(
        tool_name=spec.name,
        validated_args=validated_args,
        raw_output=raw_output,
        fallback_used=False,
        fallback_reason=None,
        router_model=model_name,
        validation_errors=[],
    )


@lru_cache(maxsize=1)
def _tool_catalog() -> str:
    entries = []
    for spec in QUERY_TOOL_REGISTRY.values():
        entries.append(
            {
                "name": spec.name,
                "description": spec.description,
                "args_schema": spec.input_model.model_json_schema(),
            }
        )
    return json.dumps(entries, ensure_ascii=False, indent=2)


def _compressed_numbers(values: set[int]) -> str:
    numbers = sorted(values)
    if not numbers:
        return "none"

    ranges = []
    start = numbers[0]
    previous = numbers[0]
    for number in numbers[1:]:
        if number == previous + 1:
            previous = number
            continue
        ranges.append(f"{start}" if start == previous else f"{start}-{previous}")
        start = number
        previous = number
    ranges.append(f"{start}" if start == previous else f"{start}-{previous}")
    return ", ".join(ranges)


def _story_location_catalog(engine: Any | None) -> str:
    if engine is None:
        return "Available numbered episodes by arc are unavailable."

    cached = getattr(engine, "_router_story_location_catalog", None)
    if isinstance(cached, str):
        return cached

    source_store = engine.source_store
    episodes_by_arc: dict[tuple[str, str], set[int]] = {}
    for scene in source_store.iter_scenes():
        metadata = scene.get("metadata") if isinstance(scene, dict) else None
        if not isinstance(metadata, dict):
            continue
        arc_id = metadata.get("arc_id")
        story_type = metadata.get("story_type")
        episode = metadata.get("episode_number")
        if (
            not isinstance(arc_id, str)
            or not isinstance(story_type, str)
            or not isinstance(episode, int)
        ):
            continue
        episodes_by_arc.setdefault((arc_id, story_type), set()).add(episode)

    if not episodes_by_arc:
        return "Available numbered episodes by arc are unavailable."

    lines = [
        "Available numbered episodes by arc/story type. Treat chapter and episode as synonyms "
        "for numbered main story entries:"
    ]
    for (arc_id, story_type), episodes in sorted(episodes_by_arc.items()):
        lines.append(
            f"- arc_id={arc_id}, story_type={story_type}: episodes {_compressed_numbers(episodes)}"
        )
    lines.append(
        "When the arc is known, only pass an episode filter if the requested episode appears "
        "in this catalog. If the user asks for an unavailable episode, keep the original "
        "episode wording in query and use broader vector_search_raw filters such as arc_id only."
    )
    return "\n".join(lines)


def _router_instructions_from_catalog(story_location_catalog: str) -> str:
    return (
        "Select exactly one query tool for the user's question. "
        "Return only the tool name and arguments matching that tool's schema. "
        "Do not invent tools.\n\n"
        "Default to vector_search_raw for normal story questions, especially questions that ask "
        "what happened, how something happened, who said, did, knew, or felt something, "
        "where evidence comes from, or anything requiring citations. Use vector_search_raw when "
        "exact evidence, dialogue, speaker-specific facts, or scene-level details are "
        "needed.\n\n"
        "Story location hints: any 3-digit cardinal or ordinal term, year, or arc phrase "
        "maps to that exact arc_id, such as '103rd term' -> arc_id='103', "
        "'104 term' -> arc_id='104', and 'Year 105' -> arc_id='105'. "
        "Phrases like 'episode 1', 'ep 1', or '第1話' map to episode=1. "
        "If the question gives an arc or term and episode but asks what happened or needs "
        "exact evidence, use vector_search_raw with arc_id and episode filters. For a recap "
        "of a known year, episode, or part, MUST use get_summaries with location filters and no "
        "vector search (for example, episode 3 means arc_id plus episode=3). Use "
        "vector_search_summaries only to locate summaries by topic when the story location is "
        "not known. Use get_scene only when both file_path and "
        "zero-based scene_index are explicitly known from the user or a previous retrieved "
        "result. Use lookup_glossary only for direct glossary, translation, or term "
        "resolution questions, not when a glossary term appears inside a broader story "
        "question.\n\n"
        "Use get_state for questions about what is true about a character or the world at a "
        "point in the story, such as roles, aliases, locations, relationships, or status, "
        "optionally as of a specific episode. Use count_dialogue for quantitative questions "
        "such as how many times a speaker talks in a year, episode, or part; its SQL count is "
        "exact, so never answer counting questions with vector_search_raw.\n\n"
        "Examples:\n"
        "Question: what happened in episode 13 of the 103rd term\n"
        "Output: tool_name='vector_search_raw', args={'query':'what happened','arc_id':'103',"
        "'episode':13,'top_k':8}\n"
        "Question: how did Kosuzu join the school idol club at episode 1 of the 104th term\n"
        "Output: tool_name='vector_search_raw', args={'query':'how did Kosuzu join the school "
        "idol club','arc_id':'104','episode':1,'top_k':8}\n"
        "Question: summarize the 104th term\n"
        "Output: tool_name='get_summaries', args={'arc_id':'104','summary_level':1}\n"
        "Question: resolve the glossary term 日野下花帆\n"
        "Output: tool_name='lookup_glossary', args={'term':'日野下花帆'}\n\n"
        f"{story_location_catalog}\n\n"
        f"Registered tools:\n{_tool_catalog()}"
    )


def _router_instructions(engine: Any | None = None) -> str:
    return _router_instructions_from_catalog(_story_location_catalog(engine))


class QueryRouter:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or get_router_model_name()
        self._location_catalogs_by_engine_id: dict[int | None, str] = {}

    def _story_location_catalog(self, engine: Any | None) -> str:
        cache_key = _engine_cache_key(engine)
        if cache_key not in self._location_catalogs_by_engine_id:
            self._location_catalogs_by_engine_id[cache_key] = _story_location_catalog(engine)
        return self._location_catalogs_by_engine_id[cache_key]

    def _instructions(self, engine: Any | None) -> str:
        return _router_instructions_from_catalog(self._story_location_catalog(engine))

    def _run_model(
        self,
        question: str,
        *,
        final_top_k: int,
        engine: Any | None = None,
    ) -> RouterOutput:
        agent: Agent[None, RouterOutput] = Agent(
            create_generation_model(self.model_name),
            output_type=RouterOutput,
            instructions=self._instructions(engine),
        )
        result = agent.run_sync(
            json.dumps(
                {"question": question, "default_top_k": final_top_k},
                ensure_ascii=False,
            )
        )
        return result.output

    def route(
        self,
        question: str,
        *,
        final_top_k: int,
        engine: Any | None = None,
    ) -> RouterDecision:
        try:
            output = self._run_model(question, final_top_k=final_top_k, engine=engine)
        except Exception as exc:
            return _fallback_decision(
                question,
                final_top_k=final_top_k,
                router_model=self.model_name,
                raw_output=None,
                reason="router model failure",
                validation_errors=[str(exc)],
            )
        return validate_router_output(
            output,
            question=question,
            final_top_k=final_top_k,
            router_model=self.model_name,
        )

    def route_and_dispatch(
        self,
        engine: Any,
        question: str,
        *,
        final_top_k: int,
    ) -> RouterDispatchResult:
        decision = self.route(question, final_top_k=final_top_k, engine=engine)
        spec = QUERY_TOOL_REGISTRY[decision.tool_name]
        try:
            result = spec.dispatcher(engine, decision.validated_args)
        except Exception as exc:
            first_error = str(exc)
            decision = _fallback_decision(
                question,
                final_top_k=final_top_k,
                router_model=self.model_name,
                raw_output=decision.raw_output,
                reason="tool dispatch failure",
                validation_errors=[*decision.validation_errors, first_error],
            )
            spec = QUERY_TOOL_REGISTRY[decision.tool_name]
            try:
                result = spec.dispatcher(engine, decision.validated_args)
            except Exception as fallback_exc:
                return RouterDispatchResult(
                    decision=decision,
                    tool_result=ToolResult(
                        errors=[
                            f"fallback tool dispatch failed after {decision.fallback_reason}: "
                            f"{fallback_exc}"
                        ],
                        metadata={"initial_dispatch_error": first_error},
                    ),
                )

        return RouterDispatchResult(decision=decision, tool_result=result)


class FixtureQueryRouter(QueryRouter):
    def __init__(
        self,
        tool_name: str = "vector_search_raw",
        args: dict[str, Any] | None = None,
        *,
        model_name: str = "fixture-router",
        error: Exception | None = None,
    ) -> None:
        self.model_name = model_name
        self._output = RouterOutput(tool_name=tool_name, args=args or {})
        self._error = error

    def _run_model(
        self,
        question: str,
        *,
        final_top_k: int,
        engine: Any | None = None,
    ) -> RouterOutput:
        if self._error is not None:
            raise self._error
        return self._output
