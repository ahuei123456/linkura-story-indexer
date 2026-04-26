import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..console import safe_print
from ..database import RETRIEVAL_QUERY, create_text_agent, embed_texts, get_chroma_collection
from ..indexer.parser import StoryParser
from ..lexical import LexicalIndex, expand_query_with_glossary
from .analysis import (
    CHRONOLOGY_INTENT,
    EXACT_EVIDENCE_INTENT,
    QUANTITATIVE_INTENT,
    SUMMARY_INTENT,
    QueryAnalysis,
    analyze_query,
)

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


@dataclass(frozen=True)
class RetrievalConfig:
    routing_candidate_count: int = ROUTING_CANDIDATE_COUNT
    raw_candidate_count: int = RAW_CANDIDATE_COUNT
    summary_child_candidate_count: int = SUMMARY_CHILD_CANDIDATE_COUNT
    neighbor_scene_window: int = NEIGHBOR_SCENE_WINDOW
    max_ranked_candidates: int = MAX_RANKED_CANDIDATES
    final_top_k: int = FINAL_TOP_K
    rrf_k: int = RRF_K

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


DEFAULT_RETRIEVAL_CONFIG = RetrievalConfig()


class StoryQueryEngine:
    def __init__(
        self,
        state_file: str = "world_state.json",
        glossary_file: str = "glossary.json",
        retrieval_config: RetrievalConfig | None = None,
    ):
        self.collection = get_chroma_collection()
        self.lexical_index = LexicalIndex()
        self.retrieval_config = retrieval_config or DEFAULT_RETRIEVAL_CONFIG

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
        return {arc_id for arc_id in re.findall(r"\b\d{3}\b", question) if arc_id in self.state_ledger}

    def _state_ledger_arc_ids(self, question: str, retrieved_arc_ids: set[str]) -> set[str]:
        explicit_arc_ids = self._question_arc_ids(question)
        if explicit_arc_ids:
            return explicit_arc_ids
        return retrieved_arc_ids

    def _build_system_prompt(self, arc_ids: set[str]) -> str:
        """Builds the system prompt with invariants and state ledger."""
        prompt = (
            "You are an expert lore-keeper and archivist for a Japanese narrative story.\n"
            "Answer based strictly on the provided raw source text in retrieved context.\n"
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

        if self.state_ledger and arc_ids:
            prompt += "\n--- STATE LEDGER (FACTS) ---\n"
            for arc_id in arc_ids:
                if arc_id in self.state_ledger:
                    prompt += f"\nYEAR {arc_id} FACTS:\n"
                    prompt += (
                        json.dumps(
                            self.state_ledger[arc_id],
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )

        return prompt

    def _citation_label(self, metadata: dict[str, Any]) -> str:
        arc_id = metadata.get("arc_id", "unknown")
        episode = self._episode_label(metadata)

        part = metadata.get("part_name", "unknown")
        scene_label = self._scene_label(metadata)
        if scene_label:
            return f"{arc_id} · {episode} · Part {part} · {scene_label}"
        return f"{arc_id} · {episode} · Part {part}"

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

        return self._and_where(filters) or {"summary_level": summary_level}

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

    def _rank_raw_candidates(
        self,
        question: str,
        expanded_question: str,
        raw_nodes: list[Node],
        seed_nodes: list[Node],
    ) -> list[Node]:
        terms = self._query_terms(expanded_question)
        seed_rank = {
            self._node_key(document, metadata): index
            for index, (document, metadata) in enumerate(seed_nodes)
        }

        scored_nodes = []
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
            score = (
                matched_terms * 25
                + speaker_matches * 30
                + self._metadata_match_score(question, metadata) * 40
                + self._near_seed_score(metadata, seed_nodes) * 10
                + (20 if is_seed else 0)
            )
            scored_nodes.append(
                (
                    -score,
                    seed_rank.get(key, len(seed_rank) + index),
                    index,
                    document,
                    metadata,
                )
            )

        scored_nodes.sort()
        ranked_nodes = [(document, metadata) for _, _, _, document, metadata in scored_nodes]
        return ranked_nodes[: self._config().max_ranked_candidates]

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
    ) -> str:
        state_ledger_arc_ids = self._state_ledger_arc_ids(question, self._raw_arc_ids(raw_nodes))
        system_prompt = self._build_system_prompt(state_ledger_arc_ids)
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

    def query(self, question: str) -> str:
        """Executes the Hierarchical RAG query flow."""
        safe_print("Searching vector index by summary tiers and raw evidence...")
        analysis = analyze_query(question, self.glossary)
        expanded_question = self._expanded_question(question)
        query_embedding = self._query_embedding(expanded_question)
        retrieved_nodes = self._tiered_retrieve(
            expanded_question,
            query_embedding=query_embedding,
            analysis=analysis,
        )

        if not retrieved_nodes:
            safe_print("Tiered retrieval returned no hits.")

        direct_raw_nodes = self._raw_evidence_nodes(retrieved_nodes)
        raw_ranked_lists = [direct_raw_nodes] if direct_raw_nodes else []
        if retrieved_nodes:
            safe_print("Expanding summary hits to child raw scenes...")
            child_raw_nodes = self._expand_summaries_to_raw_scenes(
                expanded_question,
                retrieved_nodes,
                query_embedding=query_embedding,
                analysis=analysis,
            )
            if child_raw_nodes:
                raw_ranked_lists.append(child_raw_nodes)

        raw_nodes = self._filter_raw_nodes_by_analysis(self._rrf_fuse(raw_ranked_lists), analysis)

        if not raw_nodes:
            return INSUFFICIENT_SOURCE_CONTEXT

        safe_print("Expanding neighboring raw evidence...")
        expanded_raw_nodes = self._expand_raw_neighbors(
            expanded_question,
            raw_nodes,
            query_embedding=query_embedding,
        )
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
        final_raw_nodes = self._filter_raw_nodes_by_analysis(ranked_raw_nodes, analysis)[
            : self._config().final_top_k
        ]

        if not final_raw_nodes:
            return INSUFFICIENT_SOURCE_CONTEXT

        safe_print("Building answer context from raw source scenes...")
        return self._answer_from_raw_evidence(question, final_raw_nodes)
