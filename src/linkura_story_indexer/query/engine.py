import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ..console import safe_print
from ..database import RETRIEVAL_QUERY, create_text_agent, embed_texts, get_chroma_collection
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
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                chunk_ids.append(chunk_id)

        if not chunk_ids:
            return None
        return {"chunk_id": {"$in": chunk_ids}}

    def _combine_where(self, *filters: dict[str, Any] | None) -> dict[str, Any] | None:
        flattened: list[dict[str, Any]] = []
        for item in filters:
            if not item:
                continue
            if set(item.keys()) == {"$and"} and isinstance(item.get("$and"), list):
                flattened.extend(
                    sub_item for sub_item in item["$and"] if isinstance(sub_item, dict)
                )
            else:
                flattened.append(item)
        return self._and_where(flattened)

    def _metadata_value(self, metadata: dict[str, Any], key: str) -> Any:
        if key == "story_order":
            return metadata.get("story_order", metadata.get("canonical_story_order"))
        return metadata.get(key)

    def _metadata_matches_filter(self, metadata: dict[str, Any], where: dict[str, Any]) -> bool:
        and_filters = where.get("$and")
        if isinstance(and_filters, list):
            return all(
                isinstance(item, dict) and self._metadata_matches_filter(metadata, item)
                for item in and_filters
            )

        for key, expected in where.items():
            if key == "$and":
                continue
            actual = self._metadata_value(metadata, key)
            if isinstance(expected, dict):
                if not self._metadata_matches_operator(actual, expected):
                    return False
                continue
            if actual != expected:
                return False
        return True

    def _metadata_matches_operator(self, actual: Any, expected: dict[str, Any]) -> bool:
        for operator, value in expected.items():
            if operator == "$eq":
                if actual != value:
                    return False
                continue
            if operator == "$in":
                if not isinstance(value, list) or actual not in value:
                    return False
                continue
            if not isinstance(actual, (int, float)) or not isinstance(value, (int, float)):
                return False
            if operator == "$lt" and not actual < value:
                return False
            if operator == "$lte" and not actual <= value:
                return False
            if operator == "$gt" and not actual > value:
                return False
            if operator == "$gte" and not actual >= value:
                return False
            if operator not in {"$lt", "$lte", "$gt", "$gte"}:
                return False
        return True

    def _temporal_story_order_filter(
        self,
        analysis: QueryAnalysis,
    ) -> dict[str, Any] | None:
        constraint = analysis.temporal_constraint
        if constraint is None:
            return None

        story_order = self._resolve_temporal_story_order(analysis)
        if story_order is None:
            return None

        if constraint.operator == "before":
            return {"story_order": {"$lt": story_order}}
        if constraint.operator == "after":
            return {"story_order": {"$gt": story_order}}
        return {"story_order": {"$lte": story_order}}

    def _resolve_temporal_story_order(self, analysis: QueryAnalysis) -> int | None:
        constraint = analysis.temporal_constraint
        if constraint is None:
            return None

        filters: list[dict[str, Any]] = [{"summary_level": 4}]
        if constraint.episode_number is not None:
            filters.append({"episode_number": constraint.episode_number})
        if constraint.arc_id is not None:
            filters.append({"arc_id": constraint.arc_id})
        if len(analysis.arc_ids) == 1 and constraint.arc_id is None:
            filters.append({"arc_id": analysis.arc_ids[0]})
        if analysis.story_type:
            filters.append({"story_type": analysis.story_type})

        where = self._and_where(filters)
        if where is None or where == {"summary_level": 4}:
            return None

        collection_get = getattr(getattr(self, "collection", None), "get", None)
        if not callable(collection_get):
            return None

        try:
            results = collection_get(where=where, include=["metadatas"])
        except (TypeError, ValueError):
            return None

        if not isinstance(results, dict):
            return None
        orders = [
            int(order)
            for metadata in results.get("metadatas", [])
            if isinstance(metadata, dict)
            for order in [metadata.get("story_order", metadata.get("canonical_story_order"))]
            if isinstance(order, int)
        ]
        if not orders:
            return None
        if constraint.operator == "before":
            return min(orders)
        return max(orders)

    def _results_to_nodes(self, results: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        documents = results.get("documents") or [[]]
        metadatas = results.get("metadatas") or [[]]
        return [
            (document, dict(metadata or {}))
            for document, metadata in zip(documents[0], metadatas[0], strict=False)
        ]

    def _results_to_ranked_nodes(
        self,
        results: dict[str, Any],
    ) -> list[tuple[Node, float | None]]:
        documents = results.get("documents") or [[]]
        metadatas = results.get("metadatas") or [[]]
        distances = results.get("distances") or [[]]
        output = []
        for index, (document, metadata) in enumerate(
            zip(documents[0], metadatas[0], strict=False)
        ):
            distance = distances[0][index] if distances and distances[0] and index < len(distances[0]) else None
            output.append(((document, dict(metadata or {})), distance))
        return output

    def _flat_results_to_nodes(self, results: dict[str, Any]) -> list[Node]:
        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []
        return [
            (document, dict(metadata or {}))
            for document, metadata in zip(documents, metadatas, strict=False)
        ]

    def _lexical_retrieve(
        self,
        question: str,
        *,
        n_results: int = ROUTING_CANDIDATE_COUNT,
        where: dict[str, Any] | None = None,
    ) -> list[Node]:
        lexical_index = getattr(self, "lexical_index", None)
        if lexical_index is None:
            return []
        return lexical_index.search(question, n_results=n_results, where=where)

    def _retrieve_with_dense_scores(
        self,
        question: str,
        *,
        n_results: int,
        where: dict[str, Any] | None,
        query_embedding: list[float],
    ) -> list[tuple[Node, float | None]]:
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        try:
            results = self.collection.query(**query_kwargs)
        except ValueError:
            query_kwargs["include"] = ["documents", "metadatas"]
            results = self.collection.query(**query_kwargs)
        return self._results_to_ranked_nodes(results)

    def _node_key(self, document: str, metadata: dict[str, Any]) -> tuple[Any, ...]:
        scene_start = metadata.get("scene_start")
        if not isinstance(scene_start, int):
            scene_start = metadata.get("scene_index")
        scene_end = metadata.get("scene_end")
        if not isinstance(scene_end, int):
            scene_end = scene_start
        key = (
            metadata.get("summary_level"),
            metadata.get("parent_year_id"),
            metadata.get("parent_episode_id"),
            metadata.get("parent_part_id"),
            metadata.get("file_path"),
            scene_start,
            scene_end,
        )
        if any(part not in (None, "") for part in key):
            return key
        return (document,)

    def _dedupe_nodes(
        self,
        nodes: list[Node],
    ) -> list[Node]:
        deduped = []
        seen = set()
        for document, metadata in nodes:
            key = self._node_key(document, metadata)
            if key in seen:
                continue
            seen.add(key)
            deduped.append((document, metadata))
        return deduped

    def _rrf_fuse(
        self,
        ranked_lists: list[list[Node]],
        *,
        k: int | None = None,
    ) -> list[Node]:
        rrf_k = self._config().rrf_k if k is None else k
        if rrf_k < 1:
            raise ValueError("rrf k must be at least 1")

        scores: dict[tuple[Any, ...], float] = {}
        nodes_by_key: dict[tuple[Any, ...], Node] = {}
        first_seen: dict[tuple[Any, ...], int] = {}
        seen_order = 0

        for ranked_list in ranked_lists:
            seen_in_list: set[tuple[Any, ...]] = set()
            for rank, (document, metadata) in enumerate(ranked_list, start=1):
                key = self._node_key(document, metadata)
                if key in seen_in_list:
                    continue
                seen_in_list.add(key)

                if key not in nodes_by_key:
                    nodes_by_key[key] = (document, metadata)
                    first_seen[key] = seen_order
                    seen_order += 1

                scores[key] = scores.get(key, 0.0) + (1.0 / (rrf_k + rank))

        ranked_keys = sorted(
            scores,
            key=lambda key: (-scores[key], first_seen[key]),
        )
        return [nodes_by_key[key] for key in ranked_keys]

    def _rrf_fuse_with_scores(
        self,
        ranked_lists: list[list[Node]],
        *,
        k: int | None = None,
    ) -> list[tuple[Node, float]]:
        rrf_k = self._config().rrf_k if k is None else k
        if rrf_k < 1:
            raise ValueError("rrf k must be at least 1")

        scores: dict[tuple[Any, ...], float] = {}
        nodes_by_key: dict[tuple[Any, ...], Node] = {}
        first_seen: dict[tuple[Any, ...], int] = {}
        seen_order = 0

        for ranked_list in ranked_lists:
            seen_in_list: set[tuple[Any, ...]] = set()
            for rank, (document, metadata) in enumerate(ranked_list, start=1):
                key = self._node_key(document, metadata)
                if key in seen_in_list:
                    continue
                seen_in_list.add(key)

                if key not in nodes_by_key:
                    nodes_by_key[key] = (document, metadata)
                    first_seen[key] = seen_order
                    seen_order += 1

                scores[key] = scores.get(key, 0.0) + (1.0 / (rrf_k + rank))

        ranked_keys = sorted(scores, key=lambda key: (-scores[key], first_seen[key]))
        return [(nodes_by_key[key], scores[key]) for key in ranked_keys]

    def _hybrid_retrieve(
        self,
        question: str,
        *,
        n_results: int = ROUTING_CANDIDATE_COUNT,
        where: dict[str, Any] | None = None,
        query_embedding: list[float] | None = None,
    ) -> list[Node]:
        dense_nodes = self._retrieve(
            question,
            n_results=n_results,
            where=where,
            query_embedding=query_embedding,
        )
        lexical_nodes = self._lexical_retrieve(question, n_results=n_results, where=where)
        return self._rrf_fuse([dense_nodes, lexical_nodes])[:n_results]

    def _tiered_retrieve(
        self,
        question: str,
        *,
        query_embedding: list[float] | None = None,
        analysis: QueryAnalysis | None = None,
    ) -> list[Node]:
        if query_embedding is None:
            query_embedding = self._query_embedding(question)
        config = self._config()
        summary_levels = self._summary_levels_for_analysis(analysis)
        ranked_lists = []
        for summary_level in summary_levels:
            ranked_lists.append(
                self._hybrid_retrieve(
                    question,
                    n_results=config.routing_candidate_count,
                    where=self._where_for_analysis(analysis, summary_level=summary_level),
                    query_embedding=query_embedding,
                )
            )
        ranked_lists.append(
            self._hybrid_retrieve(
                question,
                n_results=config.raw_candidate_count,
                where=self._where_for_analysis(
                    analysis,
                    summary_level=4,
                    include_scene_constraint=True,
                ),
                query_embedding=query_embedding,
            )
        )
        return self._rrf_fuse(ranked_lists)

    def _summary_levels_for_analysis(self, analysis: QueryAnalysis | None) -> tuple[int, ...]:
        if analysis is None:
            return (1, 2, 3)
        if analysis.scene_constraint is not None or analysis.intent_bucket in {
            EXACT_EVIDENCE_INTENT,
            QUANTITATIVE_INTENT,
        }:
            return (3,)
        if analysis.intent_bucket == SUMMARY_INTENT:
            return (1, 2, 3)
        if analysis.intent_bucket == CHRONOLOGY_INTENT:
            return (2, 3)
        return (1, 2, 3)

    def _raw_scene_filter_for_summary(
        self,
        metadata: dict[str, Any],
        analysis: QueryAnalysis | None = None,
    ) -> dict[str, Any] | None:
        level = metadata.get("summary_level")
        analysis_filter = self._where_for_analysis(
            analysis,
            summary_level=4,
            include_scene_constraint=True,
        )
        if level == 1:
            parent_year_id = metadata.get("parent_year_id") or metadata.get("arc_id")
            if isinstance(parent_year_id, str) and parent_year_id:
                return self._combine_where(analysis_filter, {"parent_year_id": parent_year_id})
        if level == 2:
            parent_episode_id = metadata.get("parent_episode_id")
            if isinstance(parent_episode_id, str) and parent_episode_id:
                return self._combine_where(analysis_filter, {"parent_episode_id": parent_episode_id})
        if level == 3:
            parent_part_id = metadata.get("parent_part_id")
            if isinstance(parent_part_id, str) and parent_part_id:
                return self._combine_where(analysis_filter, {"parent_part_id": parent_part_id})
        return None

    def _expand_summaries_to_raw_scenes(
        self,
        question: str,
        summaries: list[Node],
        *,
        query_embedding: list[float] | None = None,
        analysis: QueryAnalysis | None = None,
    ) -> list[Node]:
        if query_embedding is None:
            query_embedding = self._query_embedding(question)
        child_ranked_lists: list[list[Node]] = []

        for _, metadata in summaries:
            raw_filter = self._raw_scene_filter_for_summary(metadata, analysis)
            if raw_filter is None:
                continue

            child_nodes = self._raw_evidence_nodes(
                self._hybrid_retrieve(
                    question,
                    n_results=self._config().summary_child_candidate_count,
                    where=raw_filter,
                    query_embedding=query_embedding,
                )
            )
            if child_nodes:
                child_ranked_lists.append(child_nodes)

        return self._rrf_fuse(child_ranked_lists)

    def _raw_evidence_nodes(
        self,
        nodes: list[Node],
    ) -> list[Node]:
        return [(document, metadata) for document, metadata in nodes if metadata.get("summary_level") == 4]

    def _scene_span(self, metadata: dict[str, Any]) -> tuple[int, int] | None:
        scene_start = metadata.get("scene_start")
        scene_end = metadata.get("scene_end")
        if not isinstance(scene_start, int) or not isinstance(scene_end, int):
            scene_index = metadata.get("scene_index")
            scene_start = scene_index
            scene_end = scene_index

        if (
            not isinstance(scene_start, int)
            or not isinstance(scene_end, int)
            or scene_start < 0
            or scene_end < scene_start
        ):
            return None
        return scene_start, scene_end

    def _raw_part_filter(self, metadata: dict[str, Any]) -> dict[str, Any] | None:
        parent_part_id = metadata.get("parent_part_id")
        if isinstance(parent_part_id, str) and parent_part_id:
            return {
                "$and": [
                    {"summary_level": 4},
                    {"parent_part_id": parent_part_id},
                ]
            }

        file_path = metadata.get("file_path")
        if isinstance(file_path, str) and file_path:
            return {
                "$and": [
                    {"summary_level": 4},
                    {"file_path": file_path},
                ]
            }
        return None

    def _raw_nodes_for_part(
        self,
        question: str,
        metadata: dict[str, Any],
        *,
        query_embedding: list[float] | None = None,
    ) -> list[Node]:
        raw_filter = self._raw_part_filter(metadata)
        if raw_filter is None:
            return []

        collection_get = getattr(getattr(self, "collection", None), "get", None)
        if callable(collection_get):
            try:
                results = collection_get(
                    where=raw_filter,
                    include=["documents", "metadatas"],
                )
                if isinstance(results, dict):
                    return self._flat_results_to_nodes(results)
            except (TypeError, ValueError):
                pass

        return self._hybrid_retrieve(
            question,
            n_results=max(self._config().raw_candidate_count, self._config().max_ranked_candidates),
            where=raw_filter,
            query_embedding=query_embedding,
        )

    def _sort_raw_nodes(self, nodes: list[Node]) -> list[Node]:
        def sort_key(node: Node) -> tuple[Any, ...]:
            _, metadata = node
            span = self._scene_span(metadata) or (-1, -1)
            return (
                metadata.get("canonical_story_order", 0),
                metadata.get("parent_part_id", ""),
                metadata.get("file_path", ""),
                span[0],
                span[1],
            )

        return sorted(nodes, key=sort_key)

    def _expand_raw_neighbors(
        self,
        question: str,
        raw_nodes: list[Node],
        *,
        query_embedding: list[float] | None = None,
    ) -> list[Node]:
        window = self._config().neighbor_scene_window
        if window < 1:
            return self._dedupe_nodes(raw_nodes)

        expanded_nodes: list[Node] = list(raw_nodes)
        part_cache: dict[tuple[Any, ...], list[Node]] = {}

        for _, metadata in raw_nodes:
            span = self._scene_span(metadata)
            if span is None:
                continue
            part_filter = self._raw_part_filter(metadata)
            if part_filter is None:
                continue

            part_cache_key = tuple(
                sorted(
                    (str(key), json.dumps(value, sort_keys=True))
                    for key, value in part_filter.items()
                )
            )
            if part_cache_key not in part_cache:
                if query_embedding is not None:
                    part_nodes = self._raw_nodes_for_part(
                        question,
                        metadata,
                        query_embedding=query_embedding,
                    )
                else:
                    part_nodes = self._raw_nodes_for_part(question, metadata)
                part_cache[part_cache_key] = self._sort_raw_nodes(
                    part_nodes
                )

            window_start = span[0] - window
            window_end = span[1] + window
            for candidate in part_cache[part_cache_key]:
                _, candidate_metadata = candidate
                candidate_span = self._scene_span(candidate_metadata)
                if candidate_span is None:
                    continue
                if candidate_span[0] <= window_end and candidate_span[1] >= window_start:
                    expanded_nodes.append(candidate)

        return self._dedupe_nodes(expanded_nodes)

    def _normalized_speakers(self, metadata: dict[str, Any]) -> list[str]:
        speakers = metadata.get("detected_speakers")
        if isinstance(speakers, list):
            return [str(speaker) for speaker in speakers if str(speaker)]
        if isinstance(speakers, str):
            return [speaker for speaker in speakers.split("|") if speaker]
        return []

    def _question_quote(self, question: str) -> str | None:
        match = re.search(r"[\"“「『](.+?)[\"”」』]", question)
        if match:
            return match.group(1)

        match = re.search(r"\bwho said\s+(.+?)[?.!]*$", question, re.IGNORECASE)
        if match:
            return match.group(1).strip(" '\"“”「」『』")
        return None

    def _structured_answer(self, question: str, analysis: QueryAnalysis) -> str | None:
        source_store = getattr(self, "source_store", None)
        if source_store is None:
            return None

        question_lower = question.casefold()
        if "who said" in question_lower:
            quote = self._question_quote(question)
            if not quote:
                return None
            matches = source_store.turns_matching_text(quote)
            if not matches:
                return None
            speakers = []
            seen = set()
            for match in matches:
                speaker = str(match.get("speaker", ""))
                if speaker and speaker not in seen:
                    speakers.append(speaker)
                    seen.add(speaker)
            if not speakers:
                return None
            return f"{', '.join(speakers)} said it."

        if analysis.intent_bucket == QUANTITATIVE_INTENT and analysis.character_names:
            speaker = analysis.character_names[0]
            count = source_store.count_turns(speaker)
            return f"{speaker} has {count} dialogue turns in the indexed source records."
        return None

    def _query_terms(self, question: str) -> list[str]:
        terms = []
        terms.extend(term for term in re.findall(r"[\u3040-\u30ff\u3400-\u9fff々〆〤ー]+", question) if len(term) >= 2)
        terms.extend(term for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]*", question) if len(term) >= 2)

        unique_terms = []
        seen = set()
        for term in terms:
            normalized = term.casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            unique_terms.append(term)
        return unique_terms

    def _metadata_match_score(self, question: str, metadata: dict[str, Any]) -> int:
        score = 0
        question_lower = question.casefold()

        arc_id = metadata.get("arc_id")
        if isinstance(arc_id, str) and arc_id and arc_id in question:
            score += 1

        part_name = metadata.get("part_name")
        if isinstance(part_name, str) and part_name and part_name.casefold() in question_lower:
            score += 1

        episode_name = str(metadata.get("episode_name", ""))
        episode_match = re.search(r"第(\d+)話", episode_name)
        if episode_match:
            episode_number = episode_match.group(1)
            if (
                f"episode {episode_number}" in question_lower
                or f"ep {episode_number}" in question_lower
                or f"第{episode_number}話" in question
            ):
                score += 1

        return score

    def _near_seed_score(
        self,
        metadata: dict[str, Any],
        seed_nodes: list[Node],
    ) -> int:
        candidate_span = self._scene_span(metadata)
        if candidate_span is None:
            return 0

        score = 0
        window = self._config().neighbor_scene_window
        for _, seed_metadata in seed_nodes:
            if self._raw_part_filter(metadata) != self._raw_part_filter(seed_metadata):
                continue
            seed_span = self._scene_span(seed_metadata)
            if seed_span is None:
                continue
            if candidate_span[0] <= seed_span[1] + window and candidate_span[1] >= seed_span[0] - window:
                score += 1
        return score

    def _score_raw_candidates(
        self,
        question: str,
        expanded_question: str,
        raw_nodes: list[Node],
        seed_nodes: list[Node],
    ) -> list[ScoredRankedNode]:
        terms = self._query_terms(expanded_question)
        seed_rank = {
            self._node_key(document, metadata): index
            for index, (document, metadata) in enumerate(seed_nodes)
        }

        scored_nodes: list[tuple[int, int, int, Node, dict[str, Any]]] = []
        for index, (document, metadata) in enumerate(raw_nodes):
            searchable = " ".join(
                [
                    document,
                    str(metadata.get("arc_id", "")),
                    str(metadata.get("episode_name", "")),
                    str(metadata.get("part_name", "")),
                    " ".join(self._normalized_speakers(metadata)),
                ]
            )
            searchable_lower = searchable.casefold()
            matched_terms = sum(1 for term in terms if term.casefold() in searchable_lower)
            speaker_matches = sum(
                1
                for speaker in self._normalized_speakers(metadata)
                if speaker and speaker in expanded_question
            )
            key = self._node_key(document, metadata)
            is_seed = key in seed_rank
            metadata_match_score = self._metadata_match_score(question, metadata)
            near_seed_score = self._near_seed_score(metadata, seed_nodes)
            score = (
                matched_terms * 25
                + speaker_matches * 30
                + metadata_match_score * 40
                + near_seed_score * 10
                + (20 if is_seed else 0)
            )
            signal_breakdown = {
                "matched_terms": matched_terms,
                "speaker_matches": speaker_matches,
                "metadata_match_score": metadata_match_score,
                "near_seed_score": near_seed_score,
                "is_seed": is_seed,
            }
            scored_nodes.append(
                (
                    -score,
                    seed_rank.get(key, len(seed_rank) + index),
                    index,
                    (document, metadata),
                    signal_breakdown,
                )
            )

        scored_nodes.sort()
        return [
            (node, signal_breakdown, -negative_score)
            for negative_score, _, _, node, signal_breakdown in scored_nodes
        ][: self._config().max_ranked_candidates]

    def _rank_raw_candidates(
        self,
        question: str,
        expanded_question: str,
        raw_nodes: list[Node],
        seed_nodes: list[Node],
    ) -> list[Node]:
        return [
            node
            for node, _, _ in self._score_raw_candidates(
                question,
                expanded_question,
                raw_nodes,
                seed_nodes,
            )
        ]

    def _build_context_chunks(self, raw_nodes: list[Node]) -> list[str]:
        context_chunks = []
        for idx, (document, meta) in enumerate(raw_nodes):
            arc_id = meta.get("arc_id")
            safe_print(
                f"  Evidence {idx + 1}: Year {arc_id}, Ep: {meta.get('episode_name')}, "
                f"Part: {meta.get('part_name')}, {self._scene_label(meta) or 'Scene unknown'}"
            )

            raw_text = self._fetch_raw_text(meta) or document
            citation = self._citation_label(meta)
            citation_metadata = self._citation_metadata(meta)
            context_chunk = (
                f"--- RAW EVIDENCE {idx + 1} "
                f"(CITATION: {citation}; "
                f"METADATA: {json.dumps(citation_metadata, ensure_ascii=False)}) ---\n"
            )
            context_chunk += f"RAW SOURCE TEXT:\n{raw_text}\n"
            context_chunks.append(context_chunk)

        return context_chunks

    def _build_summary_context_chunks(self, summary_nodes: list[Node]) -> list[str]:
        context_chunks = []
        for idx, (document, meta) in enumerate(summary_nodes):
            citation = self._summary_citation_label(meta)
            citation_metadata = self._citation_metadata(meta)
            safe_print(
                f"  Summary evidence {idx + 1}: Year {meta.get('arc_id')}, "
                f"Story type: {meta.get('story_type')}, "
                f"Episode: {self._summary_episode_value(meta)}, "
                f"Part: {self._summary_part_label(meta)}, "
                f"summary_level: {meta.get('summary_level')}"
            )
            context_chunk = (
                f"--- SUMMARY EVIDENCE {idx + 1} "
                f"(CITATION: {citation}; "
                f"METADATA: {json.dumps(citation_metadata, ensure_ascii=False)}) ---\n"
            )
            context_chunk += (
                "SUMMARY METADATA:\n"
                f"Year: {meta.get('arc_id')}\n"
                f"Story type: {meta.get('story_type')}\n"
                f"Episode: {self._summary_episode_value(meta)}\n"
                f"Part: {self._summary_part_label(meta)}\n"
                f"Summary level: {meta.get('summary_level')}\n\n"
                f"GENERATED SUMMARY TEXT:\n{document}\n"
            )
            context_chunks.append(context_chunk)

        return context_chunks

    def _raw_arc_ids(self, raw_nodes: list[tuple[str, dict[str, Any]]]) -> set[str]:
        arc_ids = set()
        for _, metadata in raw_nodes:
            arc_id = metadata.get("arc_id")
            if isinstance(arc_id, str):
                arc_ids.add(arc_id)
        return arc_ids

    def _answer_from_raw_evidence(
        self,
        question: str,
        raw_nodes: list[Node],
        analysis: QueryAnalysis | None = None,
    ) -> str:
        state_ledger_arc_ids = self._state_ledger_arc_ids(question, self._raw_arc_ids(raw_nodes))
        system_prompt = self._build_system_prompt(state_ledger_arc_ids, analysis)
        combined_context = "\n".join(self._build_context_chunks(raw_nodes))

        user_prompt = (
            "Please answer the following question based ONLY on the raw source text provided below.\n\n"
            "Every factual claim should cite one or more provided CITATION labels exactly as written. "
            "Use episode numbers in citations, never Japanese episode titles.\n\n"
            f"QUESTION: {question}\n\n"
            f"CONTEXT:\n{combined_context}"
        )

        safe_print("Synthesizing final answer with Gemini...")
        result = create_text_agent(system_prompt).run_sync(user_prompt)
        return result.output.strip() or "No answer generated."

    def _answer_from_summary_evidence(
        self,
        question: str,
        summary_nodes: list[Node],
    ) -> str:
        state_ledger_arc_ids = self._state_ledger_arc_ids(question, self._raw_arc_ids(summary_nodes))
        system_prompt = self._build_system_prompt(
            state_ledger_arc_ids,
            None,
            context_kind="summary",
        )
        combined_context = "\n".join(self._build_summary_context_chunks(summary_nodes))

        user_prompt = (
            "Please answer the following question based ONLY on the retrieved context provided "
            "below. The context may contain generated summaries rather than raw source text.\n\n"
            "Every factual claim should cite one or more provided CITATION labels exactly as "
            "written. Use the summary citation labels exactly as provided, including "
            "summary_level. Do not invent scene labels for summary evidence.\n\n"
            f"QUESTION: {question}\n\n"
            f"CONTEXT:\n{combined_context}"
        )

        safe_print("Synthesizing final answer with Gemini...")
        result = create_text_agent(system_prompt).run_sync(user_prompt)
        return result.output.strip() or "No answer generated."

    def _trace_node_id(self, document: str, metadata: dict[str, Any]) -> str:
        chunk_id = metadata.get("chunk_id")
        if isinstance(chunk_id, str) and chunk_id:
            return chunk_id
        file_path = metadata.get("file_path")
        span = self._scene_span(metadata)
        if isinstance(file_path, str) and span is not None:
            return f"{file_path}:{span[0]}-{span[1]}"
        key = self._node_key(document, metadata)
        return json.dumps(key, ensure_ascii=False, sort_keys=True, default=str)

    def _source_identity(self, metadata: dict[str, Any]) -> SourceIdentity | None:
        span = self._scene_span(metadata)
        if span is None:
            return None
        return SourceIdentity(
            arc_id=str(metadata.get("arc_id", "")),
            story_type=str(metadata.get("story_type", "")),
            episode_name=str(metadata.get("episode_name", "")),
            part_name=str(metadata.get("part_name", "")),
            file_path=str(metadata.get("file_path", "")),
            scene_start=span[0],
            scene_end=span[1],
        )

    def _trace_candidate(
        self,
        node: Node,
        *,
        rank: int,
        scores: CandidateScores | None = None,
        signal_breakdown: dict[str, Any] | None = None,
        provenance: str | None = None,
        provenance_node_id: str | None = None,
    ) -> CandidateTrace:
        document, metadata = node
        return CandidateTrace(
            node_id=self._trace_node_id(document, metadata),
            rank=rank,
            candidate_kind="raw_span" if metadata.get("summary_level") == 4 else "summary",
            text=document,
            scores=scores or CandidateScores(),
            metadata=dict(metadata),
            source_span=self._source_identity(metadata),
            signal_breakdown=signal_breakdown or {},
            provenance=provenance,  # type: ignore[arg-type]
            provenance_node_id=provenance_node_id,
        )

    def _trace_stage(
        self,
        name: StageName,
        candidates: list[CandidateTrace] | None,
        unavailable_reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StageTrace:
        return StageTrace(
            name=name,
            candidates=candidates,
            unavailable_reason=unavailable_reason,
            metadata=metadata or {},
        )

    def _neighbor_trace_provenance(
        self,
        node: Node,
        seed_nodes: list[Node],
    ) -> tuple[str | None, str | None]:
        document, metadata = node
        key = self._node_key(document, metadata)
        seed_keys = {
            self._node_key(seed_document, seed_metadata)
            for seed_document, seed_metadata in seed_nodes
        }
        if key in seed_keys:
            return "direct_hit", None

        candidate_span = self._scene_span(metadata)
        if candidate_span is None:
            return None, None
        window = self._config().neighbor_scene_window
        for seed_document, seed_metadata in seed_nodes:
            if self._raw_part_filter(metadata) != self._raw_part_filter(seed_metadata):
                continue
            seed_span = self._scene_span(seed_metadata)
            if seed_span is None:
                continue
            if candidate_span[0] <= seed_span[1] + window and candidate_span[1] >= seed_span[0] - window:
                return "neighbor_of", self._trace_node_id(seed_document, seed_metadata)
        return None, None

    def _hybrid_retrieve_trace(
        self,
        question: str,
        *,
        n_results: int,
        where: dict[str, Any] | None,
        query_embedding: list[float] | None,
        dense_unavailable_reason: str | None = None,
    ) -> tuple[list[Node], dict[StageName, StageTrace]]:
        dense_nodes: list[Node] = []
        dense_distances: dict[tuple[Any, ...], float | None] = {}
        if query_embedding is None:
            dense_unavailable_reason = dense_unavailable_reason or "query embedding unavailable"
        else:
            try:
                dense_ranked_nodes = self._retrieve_with_dense_scores(
                    question,
                    n_results=n_results,
                    where=where,
                    query_embedding=query_embedding,
                )
                for node, distance in dense_ranked_nodes:
                    dense_nodes.append(node)
                    dense_distances[self._node_key(*node)] = distance
            except Exception as exc:
                dense_unavailable_reason = f"dense retrieval unavailable: {exc}"

        lexical_nodes = self._lexical_retrieve(question, n_results=n_results, where=where)
        fused_with_scores = self._rrf_fuse_with_scores([dense_nodes, lexical_nodes])[:n_results]
        fused_nodes = [node for node, _ in fused_with_scores]

        dense_candidates = [
            self._trace_candidate(
                node,
                rank=rank,
                scores=CandidateScores(
                    dense_rank=rank,
                    dense_distance=dense_distances.get(self._node_key(*node)),
                ),
            )
            for rank, node in enumerate(dense_nodes, start=1)
        ]
        lexical_candidates = [
            self._trace_candidate(
                node,
                rank=rank,
                scores=CandidateScores(lexical_rank=rank),
            )
            for rank, node in enumerate(lexical_nodes, start=1)
        ]
        fused_candidates = [
            self._trace_candidate(
                node,
                rank=rank,
                scores=CandidateScores(rrf_score=score),
            )
            for rank, (node, score) in enumerate(fused_with_scores, start=1)
        ]
        return fused_nodes, {
            "dense_raw": self._trace_stage(
                "dense_raw",
                None if dense_unavailable_reason else dense_candidates,
                dense_unavailable_reason,
            ),
            "lexical_raw": self._trace_stage("lexical_raw", lexical_candidates),
            "rrf_fusion": self._trace_stage("rrf_fusion", fused_candidates),
        }

    def _raw_only_retrieve(
        self,
        question: str,
        *,
        query_embedding: list[float] | None = None,
        analysis: QueryAnalysis | None = None,
    ) -> list[Node]:
        return self._hybrid_retrieve(
            question,
            n_results=self._config().raw_candidate_count,
            where=self._where_for_analysis(
                analysis,
                summary_level=4,
                include_scene_constraint=True,
            ),
            query_embedding=query_embedding,
        )

    def _filter_raw_nodes_by_analysis(
        self,
        nodes: list[Node],
        analysis: QueryAnalysis,
    ) -> list[Node]:
        raw_filter = self._where_for_analysis(
            analysis,
            summary_level=4,
            include_scene_constraint=True,
        )
        filtered = [
            (document, metadata)
            for document, metadata in nodes
            if self._metadata_matches_filter(metadata, raw_filter)
        ]
        if analysis.scene_constraint is None:
            return filtered

        scene = analysis.scene_constraint
        return [
            (document, metadata)
            for document, metadata in filtered
            if (span := self._scene_span(metadata)) is not None
            and span[0] <= scene.end
            and span[1] >= scene.start
        ]

    def _story_order(self, metadata: dict[str, Any]) -> int | None:
        order = metadata.get("story_order", metadata.get("canonical_story_order"))
        return order if isinstance(order, int) else None

    def _filter_before_semantic_boundary(
        self,
        question: str,
        expanded_question: str,
        raw_nodes: list[Node],
        seed_nodes: list[Node],
        analysis: QueryAnalysis,
    ) -> list[Node]:
        if analysis.semantic_boundary is None:
            return raw_nodes

        boundary_ranked_nodes = self._rank_raw_candidates(
            analysis.semantic_boundary,
            expanded_question,
            raw_nodes,
            seed_nodes,
        )
        if not boundary_ranked_nodes:
            return []

        _, boundary_metadata = boundary_ranked_nodes[0]
        boundary_order = self._story_order(boundary_metadata)
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
        if self._config().routing_mode == "llm_router":
            safe_print("Routing query to a typed retrieval tool...")
            routed = self._router_dispatch_with_trace(question)
            if routed.direct_answer is not None:
                return routed.direct_answer
            if not routed.nodes:
                return self._router_unavailable_message(routed.stages)
            if self._routed_chosen_tool(routed.stages) == "search_summaries":
                safe_print("Building answer context from routed summary evidence...")
            else:
                safe_print("Building answer context from routed source evidence...")
            return self._answer_from_routed_evidence(question, routed)

        safe_print("Searching raw source evidence...")
        analysis = None
        if self._config().routing_mode == "heuristic":
            analysis = analyze_query(question, self.glossary)
            structured_answer = self._structured_answer(question, analysis)
            if structured_answer is not None:
                return structured_answer
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
            return INSUFFICIENT_SOURCE_CONTEXT

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
            return INSUFFICIENT_SOURCE_CONTEXT

        safe_print("Building answer context from raw source scenes...")
        return self._answer_from_raw_evidence(question, final_raw_nodes, analysis)
