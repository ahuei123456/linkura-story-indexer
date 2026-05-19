from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, cast

from pydantic import BaseModel, Field, model_validator
from pydantic_ai import FunctionToolset

from linkura_story_indexer.eval.models import CandidateScores, SourceIdentity, StageTrace
from linkura_story_indexer.lexical import glossary_alias_groups
from linkura_story_indexer.query.engine import Node, StoryQueryEngine


class SearchRawInput(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(8, ge=1)
    arc_id: str | None = None
    episode: int | None = None
    part: str | None = None
    scene_start: int | None = Field(default=None, ge=0)
    scene_end: int | None = Field(default=None, ge=0)
    speakers: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_scene_range(self) -> SearchRawInput:
        if (
            self.scene_start is not None
            and self.scene_end is not None
            and self.scene_end < self.scene_start
        ):
            raise ValueError("scene_end must be greater than or equal to scene_start")
        return self


class SearchSummariesInput(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(8, ge=1)
    summary_level: Literal[1, 2, 3] | None = None
    arc_id: str | None = None


class GetSceneInput(BaseModel):
    file_path: str = Field(..., min_length=1)
    scene_index: int = Field(..., ge=0)


class LookupGlossaryInput(BaseModel):
    term: str = Field(..., min_length=1)


class ToolCandidate(BaseModel):
    text: str
    citation_label: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_identity: SourceIdentity | None = None
    rank: int = Field(..., ge=1)


class ToolResult(BaseModel):
    candidates: list[ToolCandidate] = Field(default_factory=list)
    trace_stages: dict[str, StageTrace] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GlossaryLookupResult(BaseModel):
    matched_category: str | None = None
    canonical_term: str | None = None
    translation: str | None = None
    aliases: list[str] = Field(default_factory=list)
    match_type: Literal["canonical", "translation", "alias", "miss"] = "miss"
    errors: list[str] = Field(default_factory=list)


def _combine_filters(engine: StoryQueryEngine, filters: list[dict[str, Any]]) -> dict[str, Any] | None:
    return engine._and_where(filters)


def _source_identity(engine: StoryQueryEngine, metadata: dict[str, Any]) -> SourceIdentity | None:
    return engine._source_identity(metadata)


def _candidate_from_node(
    engine: StoryQueryEngine,
    node: Node,
    *,
    rank: int,
    fetch_raw_text: bool = False,
) -> ToolCandidate:
    document, metadata = node
    text = engine._fetch_raw_text(metadata) if fetch_raw_text else ""
    return ToolCandidate(
        text=text or document,
        citation_label=engine._citation_label(metadata),
        metadata=dict(metadata),
        source_identity=_source_identity(engine, metadata),
        rank=rank,
    )


def _speaker_chunk_filter(
    engine: StoryQueryEngine,
    speakers: list[str],
) -> tuple[dict[str, Any] | None, list[str], bool]:
    if not speakers:
        return None, [], False

    source_store = getattr(engine, "source_store", None)
    chunk_ids_for_speaker = getattr(source_store, "chunk_ids_for_speaker", None)
    if not callable(chunk_ids_for_speaker):
        return None, ["speaker filtering unavailable: source store has no speaker index"], False
    typed_chunk_ids_for_speaker = cast(Callable[[str], list[str]], chunk_ids_for_speaker)

    chunk_ids = []
    seen = set()
    for speaker in speakers:
        for chunk_id in typed_chunk_ids_for_speaker(speaker):
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            chunk_ids.append(chunk_id)

    if not chunk_ids:
        return None, [f"no source chunks matched speakers: {', '.join(speakers)}"], True
    return {"chunk_id": {"$in": chunk_ids}}, [], False


def _raw_where(
    engine: StoryQueryEngine,
    args: SearchRawInput,
) -> tuple[dict[str, Any] | None, list[str], bool]:
    filters: list[dict[str, Any]] = [{"summary_level": 4}]
    if args.arc_id is not None:
        filters.append({"arc_id": args.arc_id})
    if args.episode is not None:
        filters.append({"episode_number": args.episode})
    if args.part is not None:
        filters.append({"part_name": args.part})

    scene_start = args.scene_start
    scene_end = args.scene_end if args.scene_end is not None else args.scene_start
    if scene_start is not None:
        filters.append({"scene_end": {"$gte": scene_start}})
    if scene_end is not None:
        filters.append({"scene_start": {"$lte": scene_end}})

    speaker_filter, warnings, empty = _speaker_chunk_filter(engine, args.speakers)
    if speaker_filter is not None:
        filters.append(speaker_filter)

    return _combine_filters(engine, filters), warnings, empty


def _summary_where(engine: StoryQueryEngine, args: SearchSummariesInput) -> dict[str, Any] | None:
    filters: list[dict[str, Any]] = []
    if args.summary_level is None:
        filters.append({"summary_level": {"$in": [1, 2, 3]}})
    else:
        filters.append({"summary_level": args.summary_level})
    if args.arc_id is not None:
        filters.append({"arc_id": args.arc_id})
    return _combine_filters(engine, filters)


def _public_trace_stages(stages: dict[Any, StageTrace]) -> dict[str, StageTrace]:
    return {str(name): stage for name, stage in stages.items()}


def search_raw(engine: StoryQueryEngine, args: SearchRawInput) -> ToolResult:
    where, warnings, empty = _raw_where(engine, args)
    if empty:
        return ToolResult(warnings=warnings, metadata={"where": where})

    expanded_query = engine._expanded_question(args.query)
    query_embedding = None
    dense_unavailable_reason = None
    try:
        query_embedding = engine._query_embedding(expanded_query)
    except Exception as exc:
        dense_unavailable_reason = f"query embedding unavailable: {exc}"

    retrieved_nodes, stages = engine._hybrid_retrieve_trace(
        expanded_query,
        n_results=max(args.top_k, engine._config().raw_candidate_count),
        where=where,
        query_embedding=query_embedding,
        dense_unavailable_reason=dense_unavailable_reason,
    )
    seed_nodes = engine._raw_evidence_nodes(retrieved_nodes)
    stages["raw_seed_filter"] = engine._trace_stage(
        "raw_seed_filter",
        [engine._trace_candidate(node, rank=rank) for rank, node in enumerate(seed_nodes, start=1)],
    )

    expanded_nodes = engine._expand_raw_neighbors(
        expanded_query,
        seed_nodes,
        query_embedding=query_embedding,
    )
    stages["neighbor_expansion"] = engine._trace_stage(
        "neighbor_expansion",
        [
            engine._trace_candidate(
                node,
                rank=rank,
                provenance=provenance,
                provenance_node_id=provenance_node_id,
            )
            for rank, node in enumerate(expanded_nodes, start=1)
            for provenance, provenance_node_id in [
                engine._neighbor_trace_provenance(node, seed_nodes)
            ]
        ],
    )

    ranked_with_scores = engine._score_raw_candidates(
        args.query,
        expanded_query,
        expanded_nodes,
        seed_nodes,
    )
    stages["deterministic_ranking"] = engine._trace_stage(
        "deterministic_ranking",
        [
            engine._trace_candidate(
                node,
                rank=rank,
                scores=CandidateScores(deterministic_score=float(score)),
                signal_breakdown=signal_breakdown,
            )
            for rank, (node, signal_breakdown, score) in enumerate(
                ranked_with_scores,
                start=1,
            )
        ],
    )

    final_nodes = [node for node, _, _ in ranked_with_scores[: args.top_k]]
    stages["final_top_k"] = engine._trace_stage(
        "final_top_k",
        [engine._trace_candidate(node, rank=rank) for rank, node in enumerate(final_nodes, start=1)],
    )

    return ToolResult(
        candidates=[
            _candidate_from_node(engine, node, rank=rank, fetch_raw_text=True)
            for rank, node in enumerate(final_nodes, start=1)
        ],
        trace_stages=_public_trace_stages(stages),
        warnings=warnings,
        metadata={"where": where},
    )


def search_summaries(engine: StoryQueryEngine, args: SearchSummariesInput) -> ToolResult:
    where = _summary_where(engine, args)
    expanded_query = engine._expanded_question(args.query)
    query_embedding = None
    dense_unavailable_reason = None
    try:
        query_embedding = engine._query_embedding(expanded_query)
    except Exception as exc:
        dense_unavailable_reason = f"query embedding unavailable: {exc}"

    summary_nodes, stages = engine._hybrid_retrieve_trace(
        expanded_query,
        n_results=args.top_k,
        where=where,
        query_embedding=query_embedding,
        dense_unavailable_reason=dense_unavailable_reason,
    )
    summary_nodes = [
        node for node in summary_nodes if node[1].get("summary_level") in {1, 2, 3}
    ][: args.top_k]

    return ToolResult(
        candidates=[
            _candidate_from_node(engine, node, rank=rank)
            for rank, node in enumerate(summary_nodes, start=1)
        ],
        trace_stages=_public_trace_stages(stages),
        metadata={"where": where},
    )


def get_scene(engine: StoryQueryEngine, args: GetSceneInput) -> ToolResult:
    source_store = getattr(engine, "source_store", None)
    get_scene_func = getattr(source_store, "get_scene", None)
    if not callable(get_scene_func):
        return ToolResult(errors=["scene lookup unavailable: source store has no get_scene method"])
    typed_get_scene = cast(Callable[[str, int], dict[str, Any] | None], get_scene_func)

    scene = typed_get_scene(args.file_path, args.scene_index)
    if scene is None:
        return ToolResult(
            errors=["scene not found"],
            metadata={"file_path": args.file_path, "scene_index": args.scene_index},
        )

    metadata = dict(scene.get("metadata") or {})
    metadata.setdefault("file_path", scene.get("file_path"))
    metadata.setdefault("scene_index", scene.get("scene_index"))
    metadata.setdefault("scene_start", scene.get("scene_index"))
    metadata.setdefault("scene_end", scene.get("scene_index"))
    candidate = ToolCandidate(
        text=str(scene.get("text", "")),
        citation_label=engine._citation_label(metadata),
        metadata=metadata,
        source_identity=_source_identity(engine, metadata),
        rank=1,
    )
    return ToolResult(candidates=[candidate])


def lookup_glossary(engine: StoryQueryEngine, args: LookupGlossaryInput) -> GlossaryLookupResult:
    glossary = getattr(engine, "glossary", None)
    if not glossary:
        return GlossaryLookupResult(errors=["glossary unavailable"])

    normalized_term = args.term.casefold()
    for category, terms in glossary.items():
        if not isinstance(terms, dict):
            continue
        category_groups = glossary_alias_groups({category: terms})
        for (canonical_term, translation), aliases in zip(
            terms.items(),
            category_groups,
            strict=False,
        ):
            if args.term == canonical_term:
                match_type: Literal["canonical", "translation", "alias"] = "canonical"
            elif normalized_term == str(translation).casefold():
                match_type = "translation"
            elif any(normalized_term == alias.casefold() for alias in aliases):
                match_type = "alias"
            else:
                continue

            return GlossaryLookupResult(
                matched_category=str(category),
                canonical_term=str(canonical_term),
                translation=str(translation),
                aliases=aliases,
                match_type=match_type,
            )

    return GlossaryLookupResult(
        match_type="miss",
        errors=[f"glossary term not found: {args.term}"],
    )


def build_query_toolset(engine: StoryQueryEngine) -> FunctionToolset:
    toolset = FunctionToolset()

    def search_raw_tool(args: SearchRawInput) -> ToolResult:
        """Search raw source scenes and nearby context."""
        return search_raw(engine, args)

    def search_summaries_tool(args: SearchSummariesInput) -> ToolResult:
        """Search indexed year, episode, or part summaries."""
        return search_summaries(engine, args)

    def get_scene_tool(args: GetSceneInput) -> ToolResult:
        """Fetch one exact raw source scene by file path and scene index."""
        return get_scene(engine, args)

    def lookup_glossary_tool(args: LookupGlossaryInput) -> GlossaryLookupResult:
        """Resolve a glossary term, translation, or generated alias."""
        return lookup_glossary(engine, args)

    toolset.add_function(search_raw_tool, name="search_raw")
    toolset.add_function(search_summaries_tool, name="search_summaries")
    toolset.add_function(get_scene_tool, name="get_scene")
    toolset.add_function(lookup_glossary_tool, name="lookup_glossary")
    return toolset
