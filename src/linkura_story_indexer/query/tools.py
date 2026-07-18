from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from pydantic_ai import FunctionToolset

from linkura_story_indexer.eval.models import SourceIdentity, StageName, StageTrace
from linkura_story_indexer.indexer.parser import StoryParser
from linkura_story_indexer.lexical import glossary_aliases_for
from linkura_story_indexer.models.state import StateFact, StatePredicate
from linkura_story_indexer.query.engine import (
    Node,
    StoryQueryEngine,
    filter_state_facts_as_of,
)


class SearchRawInput(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(8, ge=1)
    arc_id: str | None = None
    episode: int | None = None
    part: str | None = None
    scene_start: int | None = Field(default=None, ge=0)
    scene_end: int | None = Field(default=None, ge=0)
    speakers: list[str] = Field(
        default_factory=list,
        description=(
            "Optional speaker names to filter raw chunks. Multiple speakers use OR semantics: "
            "a result may contain any listed speaker, not necessarily all of them."
        ),
    )

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


class GetSummariesInput(BaseModel):
    summary_level: Literal[1, 2, 3] | None = None
    arc_id: str | None = None
    story_type: str | None = None
    episode: int | None = Field(default=None, ge=1)
    part: str | None = None
    limit: int = Field(20, ge=1, le=100)

    @field_validator("arc_id", "story_type", "part")
    @classmethod
    def _strip_scope_filter(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("scope filters must not be blank")
        return stripped

    @model_validator(mode="after")
    def validate_location_filters(self) -> GetSummariesInput:
        if self.episode is not None and self.summary_level == 1:
            raise ValueError("episode cannot be combined with summary_level 1")
        if self.part is not None and self.summary_level in {1, 2}:
            raise ValueError("part cannot be combined with summary_level 1 or 2")
        return self


class GetSceneInput(BaseModel):
    file_path: str = Field(..., min_length=1)
    scene_index: int = Field(..., ge=0)


class LookupGlossaryInput(BaseModel):
    term: str = Field(..., min_length=1)


class GetStateInput(BaseModel):
    arc_id: str = Field(..., min_length=1)
    as_of_episode: int | None = Field(
        default=None,
        ge=1,
        description=(
            "When set, only facts active at the end of this episode are returned under the "
            "ledger's half-open [valid_from, valid_to) validity interval."
        ),
    )
    subject: str | None = None
    predicate: StatePredicate | None = None
    target: str | None = None

    @field_validator("arc_id", "subject", "target")
    @classmethod
    def _reject_blank_identifiers(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("identifier filters must not be blank")
        return stripped


class GetStateResult(BaseModel):
    arc_id: str
    as_of_episode: int | None = None
    as_of_story_order: int | None = None
    facts: list[StateFact] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class CountDialogueInput(BaseModel):
    speaker: str = Field(..., min_length=1)
    arc_id: str | None = None
    episode: int | None = Field(default=None, ge=1)
    part: str | None = None

    @field_validator("speaker")
    @classmethod
    def _normalize_single_speaker(cls, value: str) -> str:
        tokens, _ = StoryParser.parse_speaker_label(value)
        if not tokens:
            raise ValueError("speaker must not be blank")
        if len(tokens) > 1:
            raise ValueError(
                "speaker must be a single name; composite labels such as 梢＆慈 are stored "
                "per named speaker, so query each speaker separately"
            )
        return tokens[0]

    @field_validator("arc_id", "part")
    @classmethod
    def _reject_blank_scope(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("scope filters must not be blank")
        return stripped


DIALOGUE_COUNTING_UNIT = (
    "distinct structured dialogue turns; a composite turn such as 梢＆慈 counts once for each "
    "named speaker it lists, and a turn attributed only to a collective label such as 全員 "
    "counts only for that collective label, never once per known character"
)


class CountDialogueResult(BaseModel):
    speaker: str
    arc_id: str | None = None
    episode: int | None = None
    part: str | None = None
    count: int = Field(0, ge=0)
    counting_unit: str = DIALOGUE_COUNTING_UNIT
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class ToolCandidate(BaseModel):
    text: str
    citation_label: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_identity: SourceIdentity | None = None
    rank: int = Field(..., ge=1)


class ToolResult(BaseModel):
    candidates: list[ToolCandidate] = Field(default_factory=list)
    trace_stages: dict[StageName, StageTrace] = Field(default_factory=dict)
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


QueryToolDispatcher = Callable[[StoryQueryEngine, BaseModel], ToolResult]


@dataclass(frozen=True)
class QueryToolSpec:
    name: str
    input_model: type[BaseModel]
    dispatcher: QueryToolDispatcher
    description: str


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
        source_identity=engine._source_identity(metadata),
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

    return engine._and_where(filters), warnings, empty


def _summary_where(engine: StoryQueryEngine, args: SearchSummariesInput) -> dict[str, Any] | None:
    filters: list[dict[str, Any]] = []
    if args.summary_level is None:
        filters.append({"summary_level": {"$in": [1, 2, 3]}})
    else:
        filters.append({"summary_level": args.summary_level})
    if args.arc_id is not None:
        filters.append({"arc_id": args.arc_id})
    return engine._and_where(filters)


def vector_search_raw(engine: StoryQueryEngine, args: SearchRawInput) -> ToolResult:
    """Run hybrid vector + lexical search over raw source scenes."""
    where, warnings, empty = _raw_where(engine, args)
    if empty:
        return ToolResult(warnings=warnings, metadata={"where": where})

    retrieval = engine.retrieve_raw_nodes_with_trace(
        args.query,
        where=where,
        top_k=args.top_k,
        n_results=max(args.top_k, engine._config().raw_candidate_count),
    )

    return ToolResult(
        candidates=[
            _candidate_from_node(engine, node, rank=rank, fetch_raw_text=True)
            for rank, node in enumerate(retrieval.nodes, start=1)
        ],
        trace_stages=retrieval.stages,
        warnings=warnings,
        metadata={"where": where},
    )


def vector_search_summaries(engine: StoryQueryEngine, args: SearchSummariesInput) -> ToolResult:
    """Run hybrid vector + lexical search over generated summaries."""
    where = _summary_where(engine, args)
    retrieval = engine.retrieve_summary_nodes_with_trace(
        args.query,
        where=where,
        top_k=args.top_k,
    )
    # Summary tier bounds are enforced by the dense and lexical retrieval where clause.
    summary_nodes = retrieval.nodes

    return ToolResult(
        candidates=[
            _candidate_from_node(engine, node, rank=rank)
            for rank, node in enumerate(summary_nodes, start=1)
        ],
        trace_stages=retrieval.stages,
        metadata={"where": where},
    )


def _summary_get_where(
    engine: StoryQueryEngine,
    args: GetSummariesInput,
) -> dict[str, Any] | None:
    if args.summary_level is not None:
        summary_level_filter: Any = args.summary_level
    elif args.part is not None:
        summary_level_filter = 3
    elif args.episode is not None:
        summary_level_filter = {"$in": [2, 3]}
    else:
        summary_level_filter = {"$in": [1, 2, 3]}

    filters: list[dict[str, Any]] = [{"summary_level": summary_level_filter}]
    if args.arc_id is not None:
        filters.append({"arc_id": args.arc_id})
    if args.story_type is not None:
        filters.append({"story_type": args.story_type})
    if args.episode is not None:
        filters.append({"episode_number": args.episode})
    if args.part is not None:
        filters.append({"part_name": args.part})
    return engine._and_where(filters)


def get_summaries(engine: StoryQueryEngine, args: GetSummariesInput) -> ToolResult:
    """Fetch stored year, episode, or part summaries by known story location."""
    where = _summary_get_where(engine, args)
    collection = getattr(engine, "collection", None)
    collection_get = getattr(collection, "get", None)
    if not callable(collection_get):
        return ToolResult(
            errors=["summary lookup unavailable: collection has no get method"],
            metadata={"where": where, "total_matched": 0, "truncated": False},
        )

    typed_collection_get = cast(Callable[..., dict[str, Any]], collection_get)
    results = typed_collection_get(where=where, include=["documents", "metadatas"])
    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []
    nodes: list[Node] = [
        (document, dict(metadata or {}))
        for document, metadata in zip(documents, metadatas, strict=False)
    ]

    def sort_key(node: Node) -> tuple[int, int]:
        metadata = node[1]
        summary_level = metadata.get("summary_level")
        story_order = metadata.get("story_order", metadata.get("canonical_story_order"))
        return (
            summary_level if isinstance(summary_level, int) else 0,
            story_order if isinstance(story_order, int) else 0,
        )

    nodes.sort(key=sort_key)
    total_matched = len(nodes)
    truncated = total_matched > args.limit
    warnings: list[str] = []
    if truncated:
        warnings.append(f"summaries truncated to limit {args.limit} from {total_matched} matches")
    if not nodes:
        warnings.append("no summaries matched the given filters")

    return ToolResult(
        candidates=[
            _candidate_from_node(engine, node, rank=rank)
            for rank, node in enumerate(nodes[: args.limit], start=1)
        ],
        warnings=warnings,
        metadata={
            "where": where,
            "total_matched": total_matched,
            "truncated": truncated,
        },
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
        source_identity=engine._source_identity(metadata),
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
        for canonical_term, translation in terms.items():
            canonical = str(canonical_term)
            translated = str(translation)
            aliases = glossary_aliases_for(canonical, translated)
            if args.term == canonical:
                match_type: Literal["canonical", "translation", "alias"] = "canonical"
            elif normalized_term == translated.casefold():
                match_type = "translation"
            elif any(normalized_term == alias.casefold() for alias in aliases):
                match_type = "alias"
            else:
                continue

            return GlossaryLookupResult(
                matched_category=str(category),
                canonical_term=canonical,
                translation=translated,
                aliases=aliases,
                match_type=match_type,
            )

    return GlossaryLookupResult(
        match_type="miss",
        errors=[f"glossary term not found: {args.term}"],
    )


def get_state(engine: StoryQueryEngine, args: GetStateInput) -> GetStateResult:
    warnings: list[str] = []
    ledger_facts = engine._state_ledger_facts()
    if not ledger_facts:
        warnings.append("state ledger is empty or unavailable")
    facts = [fact for fact in ledger_facts if fact.get("arc") == args.arc_id]

    as_of_story_order: int | None = None
    if args.as_of_episode is not None:
        source_store = getattr(engine, "source_store", None)
        max_story_order = getattr(source_store, "max_story_order", None)
        if not callable(max_story_order):
            return GetStateResult(
                arc_id=args.arc_id,
                as_of_episode=args.as_of_episode,
                warnings=warnings,
                errors=["state lookup unavailable: source store has no max_story_order method"],
            )
        typed_max_story_order = cast(Callable[..., int | None], max_story_order)
        as_of_story_order = typed_max_story_order(arc_id=args.arc_id, episode=args.as_of_episode)
        if as_of_story_order is None:
            return GetStateResult(
                arc_id=args.arc_id,
                as_of_episode=args.as_of_episode,
                warnings=warnings,
                errors=[
                    f"no source scenes found for arc {args.arc_id} episode {args.as_of_episode}"
                ],
            )
        facts = filter_state_facts_as_of(facts, as_of_story_order)

    if args.subject is not None:
        facts = [fact for fact in facts if fact.get("subject") == args.subject]
    if args.predicate is not None:
        facts = [fact for fact in facts if fact.get("predicate") == args.predicate]
    if args.target is not None:
        facts = [fact for fact in facts if fact.get("target") == args.target]

    validated_facts: list[StateFact] = []
    for fact in facts:
        try:
            validated_facts.append(StateFact.model_validate(fact))
        except ValidationError:
            warnings.append(
                "skipped ledger fact without full source provenance: "
                f"subject={fact.get('subject')!r} predicate={fact.get('predicate')!r}"
            )

    validated_facts.sort(
        key=lambda fact: (
            fact.valid_from,
            fact.episode,
            fact.part,
            fact.scene_index,
            fact.subject,
            fact.predicate,
            fact.target or "",
            fact.object,
        )
    )
    return GetStateResult(
        arc_id=args.arc_id,
        as_of_episode=args.as_of_episode,
        as_of_story_order=as_of_story_order,
        facts=validated_facts,
        warnings=warnings,
    )


def count_dialogue(engine: StoryQueryEngine, args: CountDialogueInput) -> CountDialogueResult:
    source_store = getattr(engine, "source_store", None)
    count_turns = getattr(source_store, "count_turns", None)
    if not callable(count_turns):
        return CountDialogueResult(
            speaker=args.speaker,
            arc_id=args.arc_id,
            episode=args.episode,
            part=args.part,
            errors=["dialogue counting unavailable: source store has no count_turns method"],
        )
    typed_count_turns = cast(Callable[..., int], count_turns)
    count = typed_count_turns(
        args.speaker,
        arc_id=args.arc_id,
        episode=args.episode,
        part=args.part,
    )
    return CountDialogueResult(
        speaker=args.speaker,
        arc_id=args.arc_id,
        episode=args.episode,
        part=args.part,
        count=count,
    )


def build_query_toolset(engine: StoryQueryEngine) -> FunctionToolset:
    toolset = FunctionToolset()

    def vector_search_raw_tool(args: SearchRawInput) -> ToolResult:
        """Run hybrid vector + lexical search over raw source scenes.

        Prefer this over summaries when the user asks about specific lines, who said something,
        what happened in a scene, or needs citations to raw story text. Use speaker filters as
        an OR-union when narrowing to scenes involving any listed speaker.
        """
        return vector_search_raw(engine, args)

    def vector_search_summaries_tool(args: SearchSummariesInput) -> ToolResult:
        """Run hybrid vector + lexical search over indexed summaries for topical context.

        Prefer this when the user wants to locate summaries by topic rather than by a known story
        location. The summary-level and arc filters are enforced by the retrieval query.
        """
        return vector_search_summaries(engine, args)

    def get_summaries_tool(args: GetSummariesInput) -> ToolResult:
        """Directly fetch stored summaries by known year, episode, or part location.

        Use this for recap or summarize questions when the arc, episode number, or part name is
        known. It performs no vector search; use vector_search_summaries for topical discovery.
        Pass story_type when Main and Side episodes share the same episode number.
        """
        return get_summaries(engine, args)

    def get_scene_tool(args: GetSceneInput) -> ToolResult:
        """Fetch one exact raw source scene by file path and scene index.

        Use this after a search result has already identified a file_path and scene_index, or
        when the caller already knows that exact source location. Do not use it for discovery
        because it does not search.
        """
        return get_scene(engine, args)

    def lookup_glossary_tool(args: LookupGlossaryInput) -> GlossaryLookupResult:
        """Resolve a Japanese glossary term, English translation, or generated alias.

        Use this first when a query contains character, unit, location, or story-specific terms
        that may need Japanese-English alias expansion before searching. The result gives the
        canonical Japanese term, official translation, aliases, and match type.
        """
        return lookup_glossary(engine, args)

    def get_state_tool(args: GetStateInput) -> GetStateResult:
        """Look up deterministic source-backed world-state facts for one arc.

        Prefer this over text search when the question asks what is true about a character or
        the world at a point in the story: roles, aliases, locations, relationships, group
        membership, goals, status, possessions, or honorifics. Facts come from the temporal
        State Ledger with exact source provenance and no model inference. Set as_of_episode to
        scope results to facts still active at the end of that episode.
        """
        return get_state(engine, args)

    def count_dialogue_tool(args: CountDialogueInput) -> CountDialogueResult:
        """Count exactly how many structured dialogue turns one speaker has.

        Prefer this over searching text and estimating whenever the question is quantitative,
        such as how many times a character speaks in a year, episode, or part. The count is
        computed in SQL over the structured dialogue table, so it is exact and deterministic;
        never answer counting questions from retrieved prose when this tool applies.
        """
        return count_dialogue(engine, args)

    toolset.add_function(vector_search_raw_tool, name="vector_search_raw")
    toolset.add_function(vector_search_summaries_tool, name="vector_search_summaries")
    toolset.add_function(get_summaries_tool, name="get_summaries")
    toolset.add_function(get_scene_tool, name="get_scene")
    toolset.add_function(lookup_glossary_tool, name="lookup_glossary")
    toolset.add_function(get_state_tool, name="get_state")
    toolset.add_function(count_dialogue_tool, name="count_dialogue")
    return toolset


def _dispatch_vector_search_raw(engine: StoryQueryEngine, args: BaseModel) -> ToolResult:
    return vector_search_raw(engine, cast(SearchRawInput, args))


def _dispatch_vector_search_summaries(engine: StoryQueryEngine, args: BaseModel) -> ToolResult:
    return vector_search_summaries(engine, cast(SearchSummariesInput, args))


def _dispatch_get_summaries(engine: StoryQueryEngine, args: BaseModel) -> ToolResult:
    return get_summaries(engine, cast(GetSummariesInput, args))


def _dispatch_get_scene(engine: StoryQueryEngine, args: BaseModel) -> ToolResult:
    return get_scene(engine, cast(GetSceneInput, args))


def _glossary_direct_answer(result: GlossaryLookupResult, term: str) -> str:
    if result.errors:
        return result.errors[0]
    if result.canonical_term is None or result.translation is None:
        return f"Glossary term not found: {term}"
    aliases = ", ".join(result.aliases)
    suffix = f" Aliases: {aliases}." if aliases else ""
    return f"{result.canonical_term} translates to {result.translation}.{suffix}"


def _dispatch_lookup_glossary(engine: StoryQueryEngine, args: BaseModel) -> ToolResult:
    typed_args = cast(LookupGlossaryInput, args)
    result = lookup_glossary(engine, typed_args)
    return ToolResult(
        errors=result.errors,
        metadata={
            "tool_output": result.model_dump(mode="json"),
            "direct_answer": _glossary_direct_answer(result, typed_args.term),
        },
    )


def _state_fact_line(fact: StateFact) -> str:
    target = f" -> {fact.target}" if fact.target else ""
    validity = (
        f"valid from order {fact.valid_from}, open"
        if fact.valid_to is None
        else f"valid from order {fact.valid_from} until order {fact.valid_to}"
    )
    return (
        f"- {fact.subject} {fact.predicate}{target}: {fact.object} "
        f"({fact.arc} · {fact.episode} · Part {fact.part} · Scene {fact.scene + 1}; "
        f"{validity}; quote: {fact.extracted_quote})"
    )


def _state_direct_answer(result: GetStateResult) -> str:
    if result.errors:
        return result.errors[0]
    if not result.facts:
        scope = f"arc {result.arc_id}"
        if result.as_of_episode is not None:
            scope += f" as of episode {result.as_of_episode}"
        return f"No state facts matched {scope}."
    return "\n".join(_state_fact_line(fact) for fact in result.facts)


def _dispatch_get_state(engine: StoryQueryEngine, args: BaseModel) -> ToolResult:
    result = get_state(engine, cast(GetStateInput, args))
    return ToolResult(
        warnings=result.warnings,
        errors=result.errors,
        metadata={
            "tool_output": result.model_dump(mode="json"),
            "direct_answer": _state_direct_answer(result),
        },
    )


def _count_direct_answer(result: CountDialogueResult) -> str:
    if result.errors:
        return result.errors[0]
    scope_parts = []
    if result.arc_id is not None:
        scope_parts.append(f"arc {result.arc_id}")
    if result.episode is not None:
        scope_parts.append(f"episode {result.episode}")
    if result.part is not None:
        scope_parts.append(f"part {result.part}")
    scope = f" in {', '.join(scope_parts)}" if scope_parts else " across all indexed story content"
    unit = "dialogue turn" if result.count == 1 else "dialogue turns"
    return f"{result.speaker} has {result.count} {unit}{scope}."


def _dispatch_count_dialogue(engine: StoryQueryEngine, args: BaseModel) -> ToolResult:
    result = count_dialogue(engine, cast(CountDialogueInput, args))
    return ToolResult(
        warnings=result.warnings,
        errors=result.errors,
        metadata={
            "tool_output": result.model_dump(mode="json"),
            "direct_answer": _count_direct_answer(result),
        },
    )


QUERY_TOOL_REGISTRY: dict[str, QueryToolSpec] = {
    "vector_search_raw": QueryToolSpec(
        name="vector_search_raw",
        input_model=SearchRawInput,
        dispatcher=_dispatch_vector_search_raw,
        description=(
            "Run hybrid vector + lexical search over raw source scenes for exact evidence, "
            "dialogue, scene-level details, and citation-backed answers."
        ),
    ),
    "vector_search_summaries": QueryToolSpec(
        name="vector_search_summaries",
        input_model=SearchSummariesInput,
        dispatcher=_dispatch_vector_search_summaries,
        description=(
            "Run hybrid vector + lexical search over year, episode, or part summaries to locate "
            "topical narrative context rather than exact source evidence."
        ),
    ),
    "get_summaries": QueryToolSpec(
        name="get_summaries",
        input_model=GetSummariesInput,
        dispatcher=_dispatch_get_summaries,
        description=(
            "Directly fetch stored year, episode, or part summaries by arc, episode number, or "
            "part name with no vector search; use for summarize or recap questions where the "
            "location is known. Pass story_type to disambiguate Main and Side episode numbers."
        ),
    ),
    "get_scene": QueryToolSpec(
        name="get_scene",
        input_model=GetSceneInput,
        dispatcher=_dispatch_get_scene,
        description="Fetch one exact raw source scene by file path and zero-based scene index.",
    ),
    "lookup_glossary": QueryToolSpec(
        name="lookup_glossary",
        input_model=LookupGlossaryInput,
        dispatcher=_dispatch_lookup_glossary,
        description=(
            "Resolve a Japanese glossary term, English translation, or generated alias "
            "without performing source retrieval."
        ),
    ),
    "get_state": QueryToolSpec(
        name="get_state",
        input_model=GetStateInput,
        dispatcher=_dispatch_get_state,
        description=(
            "Look up deterministic source-backed world-state facts (roles, aliases, locations, "
            "relationships, status) for one arc, optionally as of a specific episode, instead "
            "of searching prose for what is true at a point in the story."
        ),
    ),
    "count_dialogue": QueryToolSpec(
        name="count_dialogue",
        input_model=CountDialogueInput,
        dispatcher=_dispatch_count_dialogue,
        description=(
            "Count exactly how many structured dialogue turns one speaker has, optionally "
            "scoped by arc, episode, or part, using deterministic SQL counting rather than "
            "text search for quantitative questions."
        ),
    ),
}
