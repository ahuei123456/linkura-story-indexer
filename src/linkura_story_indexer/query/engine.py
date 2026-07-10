import json
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ..console import safe_print
from ..database import (
    RETRIEVAL_QUERY,
    create_text_agent,
    embed_texts,
    get_chroma_collection,
    get_generation_model_name,
)
from ..eval.models import (
    ROUTING_MODES,
    CandidateScores,
    CandidateTrace,
    EvalMode,
    QueryTrace,
    SourceIdentity,
    StageName,
    StageTrace,
)
from ..indexer.parser import StoryParser
from ..indexer.source_store import SourceRecordStore
from ..lexical import LexicalIndex, expand_query_with_glossary
from .analysis import (
    CHRONOLOGY_INTENT,
    EXACT_EVIDENCE_INTENT,
    QUANTITATIVE_INTENT,
    SUMMARY_INTENT,
    QueryAnalysis,
    analyze_query,
)

if TYPE_CHECKING:
    from .router import QueryRouter

ROUTING_CANDIDATE_COUNT = 20
RAW_CANDIDATE_COUNT = 40
SUMMARY_CHILD_CANDIDATE_COUNT = 30
NEIGHBOR_SCENE_WINDOW = 1
MAX_RANKED_CANDIDATES = 40
FINAL_TOP_K = 8
MIN_FINAL_TOP_K = 5
MAX_FINAL_TOP_K = 12
RRF_K = 60
INSUFFICIENT_SOURCE_CONTEXT = (
    "Insufficient source context: no raw source scenes were found for this question."
)

Node = tuple[str, dict[str, Any]]
ScoredRankedNode = tuple[Node, dict[str, Any], int]


@dataclass(frozen=True)
class RetrievalConfig:
    routing_candidate_count: int = ROUTING_CANDIDATE_COUNT
    raw_candidate_count: int = RAW_CANDIDATE_COUNT
    summary_child_candidate_count: int = SUMMARY_CHILD_CANDIDATE_COUNT
    neighbor_scene_window: int = NEIGHBOR_SCENE_WINDOW
    max_ranked_candidates: int = MAX_RANKED_CANDIDATES
    final_top_k: int = FINAL_TOP_K
    rrf_k: int = RRF_K
    routing_mode: Literal["off", "heuristic", "llm_router"] = "off"

    def __post_init__(self) -> None:
        if self.routing_candidate_count < 1:
            raise ValueError("routing_candidate_count must be at least 1")
        if self.raw_candidate_count < 1:
            raise ValueError("raw_candidate_count must be at least 1")
        if self.summary_child_candidate_count < 1:
            raise ValueError("summary_child_candidate_count must be at least 1")
        if self.neighbor_scene_window < 0:
            raise ValueError("neighbor_scene_window must be non-negative")
        if self.max_ranked_candidates < 1:
            raise ValueError("max_ranked_candidates must be at least 1")
        if not MIN_FINAL_TOP_K <= self.final_top_k <= MAX_FINAL_TOP_K:
            raise ValueError(f"final_top_k must be between {MIN_FINAL_TOP_K} and {MAX_FINAL_TOP_K}")
        if self.rrf_k < 1:
            raise ValueError("rrf_k must be at least 1")
        if self.routing_mode not in ROUTING_MODES:
            raise ValueError(f"routing_mode must be one of: {', '.join(ROUTING_MODES)}")


DEFAULT_RETRIEVAL_CONFIG = RetrievalConfig()


@dataclass(frozen=True)
class RetrievalTraceResult:
    nodes: list[Node]
    stages: dict[StageName, StageTrace]


@dataclass(frozen=True)
class RoutedTraceResult:
    nodes: list[Node]
    stages: dict[StageName, StageTrace]
    direct_answer: str | None = None


@dataclass(frozen=True)
class StreamingQueryResult:
    """A prepared query whose answer can be consumed exactly once as text deltas."""

    answer_deltas: Iterator[str]
    router_metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class _PreparedQuery:
    immediate_answer: str | None = None
    system_prompt: str | None = None
    user_prompt: str | None = None
    router_metadata: dict[str, Any] | None = None
    final_citation_labels: tuple[str, ...] = ()


class StoryQueryEngine:
    def __init__(
        self,
        state_file: str = "world_state.json",
        glossary_file: str = "glossary.json",
        retrieval_config: RetrievalConfig | None = None,
        query_router: "QueryRouter | None" = None,
    ):
        self.collection = get_chroma_collection()
        self.lexical_index = LexicalIndex()
        self.source_store: Any = SourceRecordStore()
        self.retrieval_config = retrieval_config or DEFAULT_RETRIEVAL_CONFIG
        self.query_router = query_router

        self.state_ledger: dict[str, Any] = {}
        if os.path.exists(state_file):
            with open(state_file, encoding="utf-8") as f:
                self.state_ledger = json.load(f)

        self.glossary: dict[str, dict[str, str]] | None = None
        if os.path.exists(glossary_file):
            with open(glossary_file, encoding="utf-8") as f:
                self.glossary = json.load(f)

    def _expanded_question(self, question: str) -> str:
        return expand_query_with_glossary(question, self.glossary)

    def _config(self) -> RetrievalConfig:
        return getattr(self, "retrieval_config", DEFAULT_RETRIEVAL_CONFIG)

    def _question_arc_ids(self, question: str) -> set[str]:
        """Find explicit story arc IDs mentioned in the user's question."""
        if not self.state_ledger:
            return set()
        ledger_arc_ids = self._state_ledger_available_arc_ids()
        return {
            match.group("arc")
            for match in re.finditer(
                r"\b(?P<arc>\d{3})(?:st|nd|rd|th)?\b",
                question,
                re.IGNORECASE,
            )
            if match.group("arc") in ledger_arc_ids
        }

    def _state_ledger_arc_ids(self, question: str, retrieved_arc_ids: set[str]) -> set[str]:
        explicit_arc_ids = self._question_arc_ids(question)
        if explicit_arc_ids:
            return explicit_arc_ids
        return retrieved_arc_ids

    def _state_ledger_available_arc_ids(self) -> set[str]:
        if not isinstance(self.state_ledger, dict):
            return set()
        facts = self.state_ledger.get("facts")
        if isinstance(facts, list):
            return {
                fact["arc"]
                for fact in facts
                if isinstance(fact, dict) and isinstance(fact.get("arc"), str)
            }
        return {arc_id for arc_id in self.state_ledger if re.fullmatch(r"\d{3}", str(arc_id))}

    def _state_ledger_facts(self) -> list[dict[str, Any]]:
        if not isinstance(self.state_ledger, dict):
            return []
        facts = self.state_ledger.get("facts")
        if isinstance(facts, list):
            return [dict(fact) for fact in facts if isinstance(fact, dict)]
        legacy_facts = []
        for arc_id, value in self.state_ledger.items():
            if re.fullmatch(r"\d{3}", str(arc_id)):
                legacy_facts.append(
                    {
                        "arc": arc_id,
                        "subject": "legacy_world_state",
                        "predicate": "summary",
                        "object": value,
                        "valid_from": 0,
                        "valid_to": None,
                    }
                )
        return legacy_facts

    def _state_ledger_slice(
        self,
        arc_ids: set[str],
        analysis: QueryAnalysis | None = None,
    ) -> list[dict[str, Any]]:
        facts = [fact for fact in self._state_ledger_facts() if fact.get("arc") in arc_ids]
        if analysis is None or analysis.temporal_constraint is None:
            return facts

        story_order = self._resolve_temporal_story_order(analysis)
        if story_order is None:
            return facts

        operator = analysis.temporal_constraint.operator
        if operator == "as_of":
            active_facts = []
            for fact in facts:
                valid_to = self._fact_valid_to(fact)
                if self._fact_valid_from(fact) <= story_order and (
                    valid_to is None or story_order < valid_to
                ):
                    active_facts.append(fact)
            return active_facts
        if operator == "before":
            return [fact for fact in facts if self._fact_valid_from(fact) < story_order]
        if operator == "after":
            later_facts = []
            for fact in facts:
                valid_to = self._fact_valid_to(fact)
                if self._fact_valid_from(fact) > story_order or (
                    valid_to is not None and valid_to > story_order
                ):
                    later_facts.append(fact)
            return later_facts
        return facts

    def _fact_valid_from(self, fact: dict[str, Any]) -> int:
        value = fact.get("valid_from")
        return int(value) if isinstance(value, int) else 0

    def _fact_valid_to(self, fact: dict[str, Any]) -> int | None:
        value = fact.get("valid_to")
        return int(value) if isinstance(value, int) else None

    def _build_system_prompt(
        self,
        arc_ids: set[str],
        analysis: QueryAnalysis | None = None,
        *,
        context_kind: Literal["raw", "summary"] = "raw",
    ) -> str:
        """Builds the system prompt with invariants and state ledger."""
        context_description = (
            "provided raw source text"
            if context_kind == "raw"
            else "provided retrieved context, which may be generated summaries rather than raw source text"
        )
        citation_source = (
            "retrieved raw evidence labels"
            if context_kind == "raw"
            else "retrieved context labels"
        )
        prompt = (
            "You are an expert lore-keeper and archivist for a Japanese narrative story.\n"
            f"Answer based strictly on the {context_description}.\n"
            "Do NOT use outside knowledge. If the provided context does not contain the answer, "
            "say so.\n"
            "Cite sources using only the CITATION labels provided in retrieved context. "
            "Do not cite raw Japanese episode titles. "
            "Do not convert the Year/Arc ID to a real-world year like 2024.\n"
        )

        if self.glossary:
            prompt += "\n--- OFFICIAL GLOSSARY (MANDATORY TRANSLATIONS) ---\n"
            for cat, terms in self.glossary.items():
                prompt += f"\n{cat.replace('_', ' ').upper()}:\n"
                for jp, en in terms.items():
                    prompt += f" - {jp} -> {en}\n"

        ledger_facts = self._state_ledger_slice(arc_ids, analysis)
        if ledger_facts:
            prompt += "\n--- STATE LEDGER (FACTS) ---\n"
            prompt += (
                "Use these source-backed facts only for routing and consistency. "
                f"Final answer citations must still come from {citation_source}.\n"
            )
            for arc_id in sorted(arc_ids):
                arc_facts = [fact for fact in ledger_facts if fact.get("arc") == arc_id]
                if not arc_facts:
                    continue
                prompt += f"\nYEAR {arc_id} FACTS:\n"
                prompt += json.dumps(arc_facts, ensure_ascii=False, separators=(",", ":")) + "\n"

        return prompt

    def _citation_label(self, metadata: dict[str, Any]) -> str:
        summary_level = metadata.get("summary_level")
        if summary_level in {1, 2, 3}:
            return self._summary_citation_label(metadata)

        arc_id = metadata.get("arc_id", "unknown")
        episode = self._episode_label(metadata)

        part = metadata.get("part_name", "unknown")
        scene_label = self._scene_label(metadata)
        if scene_label:
            return f"{arc_id} · {episode} · Part {part} · {scene_label}"
        return f"{arc_id} · {episode} · Part {part}"

    def _summary_citation_label(self, metadata: dict[str, Any]) -> str:
        summary_level = metadata.get("summary_level")
        arc_id = metadata.get("arc_id", "unknown")
        story_type = metadata.get("story_type", "unknown")
        episode = self._summary_episode_label(metadata)
        part = self._summary_part_label(metadata)
        return (
            f"{arc_id} · {story_type} · {episode} · Part {part} · "
            f"summary_level {summary_level}"
        )

    def _summary_episode_label(self, metadata: dict[str, Any]) -> str:
        if metadata.get("summary_level") == 1:
            return "Episode ALL_EPISODES"

        episode_number = metadata.get("episode_number")
        if isinstance(episode_number, int) and episode_number > 0:
            return f"Episode {episode_number}"

        return self._episode_label(metadata)

    def _summary_episode_value(self, metadata: dict[str, Any]) -> str:
        episode = self._summary_episode_label(metadata)
        if episode.startswith("Episode "):
            return episode.removeprefix("Episode ")
        if episode.startswith("Side Story "):
            return episode.removeprefix("Side Story ")
        return episode

    def _summary_part_label(self, metadata: dict[str, Any]) -> str:
        if metadata.get("summary_level") in {1, 2}:
            return "ALL_PARTS"
        part = metadata.get("part_name")
        if isinstance(part, str) and part:
            return part
        return "unknown"

    def _scene_label(self, metadata: dict[str, Any]) -> str:
        scene_start = metadata.get("scene_start")
        scene_end = metadata.get("scene_end")
        if (
            isinstance(scene_start, int)
            and scene_start >= 0
            and isinstance(scene_end, int)
            and scene_end >= scene_start
        ):
            if scene_start == scene_end:
                return f"Scene {scene_start + 1}"
            return f"Scene {scene_start + 1}-{scene_end + 1}"

        scene_index = metadata.get("scene_index")
        if isinstance(scene_index, int) and scene_index >= 0:
            return f"Scene {scene_index + 1}"
        return ""

    def _citation_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        if metadata.get("summary_level") in {1, 2, 3}:
            return {
                "arc_id": metadata.get("arc_id"),
                "story_type": metadata.get("story_type"),
                "episode": self._summary_episode_value(metadata),
                "part": self._summary_part_label(metadata),
                "summary_level": metadata.get("summary_level"),
                "parent_year_id": metadata.get("parent_year_id"),
                "parent_episode_id": metadata.get("parent_episode_id"),
                "parent_part_id": metadata.get("parent_part_id"),
                "canonical_story_order": metadata.get("canonical_story_order"),
            }
        return {
            "file_path": metadata.get("file_path"),
            "scene_index": metadata.get("scene_index"),
            "scene_start": metadata.get("scene_start"),
            "scene_end": metadata.get("scene_end"),
            "source_scene_count": metadata.get("source_scene_count"),
            "canonical_story_order": metadata.get("canonical_story_order"),
        }

    def _episode_label(self, metadata: dict[str, Any]) -> str:
        story_type = metadata.get("story_type")
        episode_name = str(metadata.get("episode_name", "unknown"))
        match = re.search(r"第(\d+)話", episode_name)
        if match:
            return f"Episode {match.group(1)}"
        if story_type == "Side":
            return f"Side Story {episode_name}"
        return f"Episode {episode_name}"

    def _fetch_raw_text(self, metadata: dict[str, Any]) -> str:
        """Fetches a raw scene span from disk based on file path and scene metadata."""
        file_path = metadata.get("file_path", "")
        scene_start = metadata.get("scene_start")
        scene_end = metadata.get("scene_end")

        if not isinstance(scene_start, int) or not isinstance(scene_end, int):
            scene_index = metadata.get("scene_index")
            scene_start = scene_index
            scene_end = scene_index

        if (
            not file_path
            or not isinstance(scene_start, int)
            or not isinstance(scene_end, int)
            or scene_start < 0
            or scene_end < scene_start
        ):
            return ""
        if not os.path.exists(file_path):
            return ""

        path = Path(file_path)
        with open(path, encoding="utf-8") as f:
            scenes = StoryParser.split_into_scenes(f.read())

        if scene_start >= len(scenes) or scene_end >= len(scenes):
            return ""

        return "\n\n---\n\n".join(scenes[scene_start : scene_end + 1])

    def _query_embedding(self, question: str) -> list[float]:
        return embed_texts([question], task_type=RETRIEVAL_QUERY)[0]

    def _retrieve(
        self,
        question: str,
        *,
        n_results: int = ROUTING_CANDIDATE_COUNT,
        where: dict[str, Any] | None = None,
        query_embedding: list[float] | None = None,
    ) -> list[Node]:
        if query_embedding is None:
            query_embedding = self._query_embedding(question)
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas"],
        }
        if where:
            query_kwargs["where"] = where

        results = self.collection.query(**query_kwargs)

        return self._results_to_nodes(results)

    def _and_where(self, filters: list[dict[str, Any]]) -> dict[str, Any] | None:
        cleaned = [item for item in filters if item]
        if not cleaned:
            return None
        if len(cleaned) == 1:
            return cleaned[0]
        return {"$and": cleaned}

    def _where_for_analysis(
        self,
        analysis: QueryAnalysis | None,
        *,
        summary_level: int,
        include_scene_constraint: bool = False,
    ) -> dict[str, Any]:
        filters: list[dict[str, Any]] = [{"summary_level": summary_level}]
        if analysis is None:
            return filters[0]

        if len(analysis.arc_ids) == 1:
            filters.append({"arc_id": analysis.arc_ids[0]})
        elif len(analysis.arc_ids) > 1:
            filters.append({"arc_id": {"$in": list(analysis.arc_ids)}})

        if summary_level != 1 and analysis.story_type:
            filters.append({"story_type": analysis.story_type})
        if summary_level in {2, 3, 4} and analysis.episode_number is not None:
            filters.append({"episode_number": analysis.episode_number})
        if summary_level in {3, 4} and analysis.part_name:
            filters.append({"part_name": analysis.part_name})

        temporal_filter = self._temporal_story_order_filter(analysis)
        if temporal_filter is not None:
            filters.append(temporal_filter)

        if include_scene_constraint and summary_level == 4 and analysis.scene_constraint is not None:
            scene = analysis.scene_constraint
            filters.append({"scene_start": {"$lte": scene.end}})
            filters.append({"scene_end": {"$gte": scene.start}})

        if summary_level == 4:
            speaker_filter = self._speaker_filter_for_analysis(analysis)
            if speaker_filter is not None:
                filters.append(speaker_filter)

        return self._and_where(filters) or {"summary_level": summary_level}

    def _speaker_filter_for_analysis(self, analysis: QueryAnalysis | None) -> dict[str, Any] | None:
        if analysis is None or not analysis.character_names:
            return None

        source_store = getattr(self, "source_store", None)
        if source_store is None:
            return None

        chunk_ids = []
        seen = set()
        for speaker in analysis.character_names:
            for chunk_id in source_store.chunk_ids_for_speaker(speaker):
                if c…10667 tokens truncated…dary_order = self._story_order(boundary_metadata)
        if boundary_order is None:
            return raw_nodes

        if re.search(r"\bafter\b", question, re.IGNORECASE):
            return [
                node
                for node in raw_nodes
                if (order := self._story_order(node[1])) is not None and order > boundary_order
            ]
        return [
            node
            for node in raw_nodes
            if (order := self._story_order(node[1])) is not None and order < boundary_order
        ]

    def retrieve_raw_nodes_with_trace(
        self,
        question: str,
        *,
        where: dict[str, Any] | None = None,
        top_k: int | None = None,
        n_results: int | None = None,
        analysis: QueryAnalysis | None = None,
    ) -> RetrievalTraceResult:
        """Executes the deterministic raw retrieval pipeline and returns final nodes plus stages."""
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be at least 1")
        if n_results is not None and n_results < 1:
            raise ValueError("n_results must be at least 1")

        expanded_question = self._expanded_question(question)
        query_embedding = None
        dense_unavailable_reason = None
        try:
            query_embedding = self._query_embedding(expanded_question)
        except Exception as exc:
            dense_unavailable_reason = f"query embedding unavailable: {exc}"

        raw_where = where
        if raw_where is None:
            raw_where = self._where_for_analysis(
                analysis,
                summary_level=4,
                include_scene_constraint=True,
            )
        retrieved_nodes, stages = self._hybrid_retrieve_trace(
            expanded_question,
            n_results=(
                n_results if n_results is not None else self._config().raw_candidate_count
            ),
            where=raw_where,
            query_embedding=query_embedding,
            dense_unavailable_reason=dense_unavailable_reason,
        )

        raw_nodes = self._raw_evidence_nodes(retrieved_nodes)
        stages["raw_seed_filter"] = self._trace_stage(
            "raw_seed_filter",
            [
                self._trace_candidate(node, rank=rank)
                for rank, node in enumerate(raw_nodes, start=1)
            ],
        )

        analysis_filtered_nodes = raw_nodes
        if analysis is not None:
            analysis_filtered_nodes = self._filter_raw_nodes_by_analysis(raw_nodes, analysis)
            stages["analysis_filter"] = self._trace_stage(
                "analysis_filter",
                [
                    self._trace_candidate(node, rank=rank)
                    for rank, node in enumerate(analysis_filtered_nodes, start=1)
                ],
            )
        else:
            stages["analysis_filter"] = self._trace_stage(
                "analysis_filter",
                None,
                "query analysis disabled",
            )

        if analysis_filtered_nodes:
            expanded_raw_nodes = self._expand_raw_neighbors(
                expanded_question,
                analysis_filtered_nodes,
                query_embedding=query_embedding,
            )
        else:
            expanded_raw_nodes = []
        stages["neighbor_expansion"] = self._trace_stage(
            "neighbor_expansion",
            [
                self._trace_candidate(
                    node,
                    rank=rank,
                    provenance=provenance,
                    provenance_node_id=provenance_node_id,
                )
                for rank, node in enumerate(expanded_raw_nodes, start=1)
                for provenance, provenance_node_id in [
                    self._neighbor_trace_provenance(node, analysis_filtered_nodes)
                ]
            ],
        )

        semantic_nodes = expanded_raw_nodes
        semantic_reason = "query analysis disabled"
        if analysis is not None:
            semantic_reason = "semantic boundary not detected"
            semantic_nodes = self._filter_raw_nodes_by_analysis(semantic_nodes, analysis)
            if analysis.semantic_boundary is not None:
                semantic_nodes = self._filter_before_semantic_boundary(
                    question,
                    expanded_question,
                    semantic_nodes,
                    analysis_filtered_nodes,
                    analysis,
                )
                semantic_reason = ""
        stages["semantic_boundary_filter"] = self._trace_stage(
            "semantic_boundary_filter",
            [
                self._trace_candidate(node, rank=rank)
                for rank, node in enumerate(semantic_nodes, start=1)
            ]
            if semantic_reason == ""
            else None,
            semantic_reason or None,
        )

        ranked_with_scores = self._score_raw_candidates(
            question,
            expanded_question,
            semantic_nodes,
            analysis_filtered_nodes,
        )
        deterministic_nodes = [node for node, _, _ in ranked_with_scores]
        stages["deterministic_ranking"] = self._trace_stage(
            "deterministic_ranking",
            [
                self._trace_candidate(
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

        final_top_k = top_k if top_k is not None else self._config().final_top_k
        if analysis is not None:
            final_raw_nodes = self._filter_raw_nodes_by_analysis(deterministic_nodes, analysis)[
                :final_top_k
            ]
        else:
            final_raw_nodes = deterministic_nodes[:final_top_k]
        stages["final_top_k"] = self._trace_stage(
            "final_top_k",
            [
                self._trace_candidate(node, rank=rank)
                for rank, node in enumerate(final_raw_nodes, start=1)
            ],
        )
        stages["reranker"] = self._trace_stage(
            "reranker",
            None,
            "reranker stage unavailable; #23 has not landed",
        )

        return RetrievalTraceResult(nodes=final_raw_nodes, stages=stages)

    def retrieve_summary_nodes_with_trace(
        self,
        question: str,
        *,
        where: dict[str, Any] | None,
        top_k: int,
    ) -> RetrievalTraceResult:
        """Executes hybrid retrieval for summary tiers and returns nodes plus trace stages."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        expanded_question = self._expanded_question(question)
        query_embedding = None
        dense_unavailable_reason = None
        try:
            query_embedding = self._query_embedding(expanded_question)
        except Exception as exc:
            dense_unavailable_reason = f"query embedding unavailable: {exc}"

        summary_nodes, stages = self._hybrid_retrieve_trace(
            expanded_question,
            n_results=top_k,
            where=where,
            query_embedding=query_embedding,
            dense_unavailable_reason=dense_unavailable_reason,
        )
        return RetrievalTraceResult(nodes=summary_nodes, stages=stages)

    def _get_query_router(self) -> "QueryRouter":
        if self.query_router is None:
            from .router import QueryRouter

            self.query_router = QueryRouter()
        return self.query_router

    def _trace_candidate_from_tool_candidate(
        self,
        candidate: Any,
    ) -> CandidateTrace:
        metadata = dict(candidate.metadata)
        text = str(candidate.text)
        return CandidateTrace(
            node_id=self._trace_node_id(text, metadata) if metadata else f"tool:{candidate.rank}",
            rank=int(candidate.rank),
            candidate_kind="raw_span" if metadata.get("summary_level") == 4 else "summary",
            text=text,
            metadata=metadata,
            source_span=candidate.source_identity,
        )

    def _router_dispatch_with_trace(self, question: str) -> RoutedTraceResult:
        dispatch = self._get_query_router().route_and_dispatch(
            self,
            question,
            final_top_k=self._config().final_top_k,
        )
        tool_result = dispatch.tool_result
        stages: dict[StageName, StageTrace] = {
            "router": self._trace_stage(
                "router",
                None,
                metadata={
                    **dispatch.decision.metadata(),
                    "tool_warnings": tool_result.warnings,
                    "tool_errors": tool_result.errors,
                    "tool_metadata": tool_result.metadata,
                },
            )
        }
        stages.update(tool_result.trace_stages)

        final_candidates = [
            self._trace_candidate_from_tool_candidate(candidate)
            for candidate in tool_result.candidates
        ]
        if "final_top_k" not in stages:
            stages["final_top_k"] = self._trace_stage("final_top_k", final_candidates)

        nodes = [
            (candidate.text, dict(candidate.metadata))
            for candidate in tool_result.candidates
            if candidate.metadata
        ]
        direct_answer = tool_result.metadata.get("direct_answer")
        return RoutedTraceResult(
            nodes=nodes,
            stages=stages,
            direct_answer=direct_answer if isinstance(direct_answer, str) else None,
        )

    def _router_unavailable_message(self, stages: dict[StageName, StageTrace]) -> str:
        router_stage = stages.get("router")
        metadata = router_stage.metadata if router_stage is not None else {}
        tool_errors = metadata.get("tool_errors")
        if isinstance(tool_errors, list) and tool_errors:
            return f"{INSUFFICIENT_SOURCE_CONTEXT} Tool error: {tool_errors[0]}"
        tool_warnings = metadata.get("tool_warnings")
        if isinstance(tool_warnings, list) and tool_warnings:
            return f"{INSUFFICIENT_SOURCE_CONTEXT} Tool warning: {tool_warnings[0]}"
        return INSUFFICIENT_SOURCE_CONTEXT

    def _routed_chosen_tool(self, stages: dict[StageName, StageTrace]) -> str | None:
        router_stage = stages.get("router")
        if router_stage is None:
            return None
        chosen_tool = router_stage.metadata.get("chosen_tool")
        return chosen_tool if isinstance(chosen_tool, str) else None

    def _answer_from_routed_evidence(self, question: str, routed: RoutedTraceResult) -> str:
        chosen_tool = self._routed_chosen_tool(routed.stages)
        if chosen_tool == "search_summaries":
            return self._answer_from_summary_evidence(question, routed.nodes)
        return self._answer_from_raw_evidence(question, routed.nodes, None)

    def retrieve_with_trace(
        self,
        question: str,
        *,
        query_id: str = "ad-hoc",
        mode: EvalMode | None = None,
        answer_mode: bool = False,
    ) -> QueryTrace:
        """Executes the raw-first retrieval flow and returns deterministic stage traces."""
        effective_mode: EvalMode = mode if mode is not None else self._config().routing_mode
        if effective_mode == "llm_router":
            routed = self._router_dispatch_with_trace(question)
            answer_text = None
            if answer_mode:
                if routed.direct_answer is not None:
                    answer_text = routed.direct_answer
                elif routed.nodes:
                    answer_text = self._answer_from_routed_evidence(question, routed)
                else:
                    answer_text = self._router_unavailable_message(routed.stages)
            return QueryTrace(
                query_id=query_id,
                question=question,
                mode=effective_mode,
                config=self._config().__dict__,
                stages=routed.stages,
                final_citation_labels=[
                    self._citation_label(metadata) for _, metadata in routed.nodes
                ],
                answer_text=answer_text,
            )

        analysis = None
        if effective_mode == "heuristic":
            analysis = analyze_query(question, self.glossary)
        retrieval = self.retrieve_raw_nodes_with_trace(question, analysis=analysis)
        final_raw_nodes = retrieval.nodes

        answer_text = None
        if answer_mode and final_raw_nodes:
            answer_text = self._answer_from_raw_evidence(question, final_raw_nodes, analysis)

        return QueryTrace(
            query_id=query_id,
            question=question,
            mode=effective_mode,
            config=self._config().__dict__,
            stages=retrieval.stages,
            final_citation_labels=[
                self._citation_label(metadata) for _, metadata in final_raw_nodes
            ],
            answer_text=answer_text,
        )

    def query(self, question: str) -> str:
        """Executes the raw-first RAG query flow."""
        prepared = self._prepare_query(question)
        if prepared.immediate_answer is not None:
            return prepared.immediate_answer
        if prepared.system_prompt is None or prepared.user_prompt is None:
            return "No answer generated."
        return self._answer_from_prompts(prepared.system_prompt, prepared.user_prompt)

    def stream_query(self, question: str) -> StreamingQueryResult:
        """Prepares a query once and returns a one-use iterator of answer deltas."""
        prepared = self._prepare_query(question)
        return StreamingQueryResult(
            answer_deltas=self._stream_prepared_answer(prepared),
            router_metadata=prepared.router_metadata,
        )

    def _prepare_query(self, question: str) -> _PreparedQuery:
        if self._config().routing_mode == "llm_router":
            safe_print("Routing query to a typed retrieval tool...")
            routed = self._router_dispatch_with_trace(question)
            router_stage = routed.stages.get("router")
            router_metadata = dict(router_stage.metadata) if router_stage is not None else None
            if routed.direct_answer is not None:
                return _PreparedQuery(
                    immediate_answer=routed.direct_answer,
                    router_metadata=router_metadata,
                )
            if not routed.nodes:
                return _PreparedQuery(
                    immediate_answer=self._router_unavailable_message(routed.stages),
                    router_metadata=router_metadata,
                )
            if self._routed_chosen_tool(routed.stages) == "search_summaries":
                safe_print("Building answer context from routed summary evidence...")
                system_prompt, user_prompt = self._summary_answer_prompts(question, routed.nodes)
            else:
                safe_print("Building answer context from routed source evidence...")
                system_prompt, user_prompt = self._raw_answer_prompts(question, routed.nodes, None)
            return _PreparedQuery(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                router_metadata=router_metadata,
                final_citation_labels=tuple(
                    self._citation_label(metadata) for _, metadata in routed.nodes
                ),
            )

        safe_print("Searching raw source evidence...")
        analysis = None
        if self._config().routing_mode == "heuristic":
            analysis = analyze_query(question, self.glossary)
            structured_answer = self._structured_answer(question, analysis)
            if structured_answer is not None:
                return _PreparedQuery(immediate_answer=structured_answer)
        expanded_question = self._expanded_question(question)
        query_embedding = self._query_embedding(expanded_question)
        retrieved_nodes = self._raw_only_retrieve(
            expanded_question,
            query_embedding=query_embedding,
            analysis=analysis,
        )

        if not retrieved_nodes:
            safe_print("Raw evidence retrieval returned no hits.")

        raw_nodes = self._raw_evidence_nodes(retrieved_nodes)
        if analysis is not None:
            raw_nodes = self._filter_raw_nodes_by_analysis(raw_nodes, analysis)

        if not raw_nodes:
            return _PreparedQuery(immediate_answer=INSUFFICIENT_SOURCE_CONTEXT)

        safe_print("Expanding neighboring raw evidence...")
        expanded_raw_nodes = self._expand_raw_neighbors(
            expanded_question,
            raw_nodes,
            query_embedding=query_embedding,
        )
        if analysis is not None:
            expanded_raw_nodes = self._filter_raw_nodes_by_analysis(expanded_raw_nodes, analysis)
            expanded_raw_nodes = self._filter_before_semantic_boundary(
                question,
                expanded_question,
                expanded_raw_nodes,
                raw_nodes,
                analysis,
            )
        ranked_raw_nodes = self._rank_raw_candidates(
            question,
            expanded_question,
            expanded_raw_nodes,
            raw_nodes,
        )
        if analysis is not None:
            final_raw_nodes = self._filter_raw_nodes_by_analysis(ranked_raw_nodes, analysis)[
                : self._config().final_top_k
            ]
        else:
            final_raw_nodes = ranked_raw_nodes[: self._config().final_top_k]

        if not final_raw_nodes:
            return _PreparedQuery(immediate_answer=INSUFFICIENT_SOURCE_CONTEXT)

        safe_print("Building answer context from raw source scenes...")
        system_prompt, user_prompt = self._raw_answer_prompts(question, final_raw_nodes, analysis)
        return _PreparedQuery(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            final_citation_labels=tuple(
                self._citation_label(metadata) for _, metadata in final_raw_nodes
            ),
        )

    def _stream_prepared_answer(self, prepared: _PreparedQuery) -> Iterator[str]:
        if prepared.immediate_answer is not None:
            yield prepared.immediate_answer
            return
        if prepared.system_prompt is None or prepared.user_prompt is None:
            yield "No answer generated."
            return

        safe_print(f"Synthesizing final answer with {get_generation_model_name()}...")
        result = create_text_agent(prepared.system_prompt).run_stream_sync(prepared.user_prompt)
        source_deltas = result.stream_text(delta=True, debounce_by=0.1)
        pending_whitespace = ""
        emitted_text = False
        try:
            for delta in source_deltas:
                if not delta:
                    continue
                combined = pending_whitespace + delta
                text = combined.rstrip()
                if not text:
                    pending_whitespace = combined
                    continue
                pending_whitespace = combined[len(text) :]
                emitted_text = True
                yield text
        finally:
            close = getattr(source_deltas, "close", None)
            if close is not None:
                close()

        if not emitted_text:
            yield "No answer generated."

