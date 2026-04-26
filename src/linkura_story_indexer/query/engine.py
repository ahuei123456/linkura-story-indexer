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

ROUTING_CANDIDATE_COUNT = 20
RAW_CANDIDATE_COUNT = 40
SUMMARY_CHILD_CANDIDATE_COUNT = 30
NEIGHBOR_SCENE_WINDOW = 1
MAX_RANKED_CANDIDATES = 40
FINAL_TOP_K = 8
MIN_FINAL_TOP_K = 5
MAX_FINAL_TOP_K = 12
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

    def _retrieve(
        self,
        question: str,
        *,
        n_results: int = ROUTING_CANDIDATE_COUNT,
        where: dict[str, Any] | None = None,
    ) -> list[Node]:
        query_embedding = embed_texts([question], task_type=RETRIEVAL_QUERY)[0]
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas"],
        }
        if where:
            query_kwargs["where"] = where

        results = self.collection.query(**query_kwargs)

        return self._results_to_nodes(results)

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

    def _hybrid_retrieve(
        self,
        question: str,
        *,
        n_results: int = ROUTING_CANDIDATE_COUNT,
        where: dict[str, Any] | None = None,
    ) -> list[Node]:
        dense_nodes = self._retrieve(question, n_results=n_results, where=where)
        lexical_nodes = self._lexical_retrieve(question, n_results=n_results, where=where)
        return self._dedupe_nodes([*dense_nodes, *lexical_nodes])

    def _raw_scene_filter_for_summary(self, metadata: dict[str, Any]) -> dict[str, Any] | None:
        level = metadata.get("summary_level")
        if level == 1:
            parent_year_id = metadata.get("parent_year_id") or metadata.get("arc_id")
            if isinstance(parent_year_id, str) and parent_year_id:
                return {
                    "$and": [
                        {"summary_level": 4},
                        {"parent_year_id": parent_year_id},
                    ]
                }
        if level == 2:
            parent_episode_id = metadata.get("parent_episode_id")
            if isinstance(parent_episode_id, str) and parent_episode_id:
                return {
                    "$and": [
                        {"summary_level": 4},
                        {"parent_episode_id": parent_episode_id},
                    ]
                }
        if level == 3:
            parent_part_id = metadata.get("parent_part_id")
            if isinstance(parent_part_id, str) and parent_part_id:
                return {
                    "$and": [
                        {"summary_level": 4},
                        {"parent_part_id": parent_part_id},
                    ]
                }
        return None

    def _expand_summaries_to_raw_scenes(
        self,
        question: str,
        summaries: list[Node],
    ) -> list[Node]:
        expanded_nodes: list[Node] = []
        seen: set[tuple[str, int, int]] = set()

        for _, metadata in summaries:
            raw_filter = self._raw_scene_filter_for_summary(metadata)
            if raw_filter is None:
                continue

            for document, raw_metadata in self._hybrid_retrieve(
                question,
                n_results=self._config().summary_child_candidate_count,
                where=raw_filter,
            ):
                if raw_metadata.get("summary_level") != 4:
                    continue
                scene_start_value = raw_metadata.get("scene_start")
                if not isinstance(scene_start_value, int):
                    scene_start_value = raw_metadata.get("scene_index", -1)
                scene_start = scene_start_value if isinstance(scene_start_value, int) else -1

                scene_end_value = raw_metadata.get("scene_end")
                scene_end = scene_end_value if isinstance(scene_end_value, int) else scene_start
                scene_key = (
                    str(raw_metadata.get("file_path", "")),
                    scene_start,
                    scene_end,
                )
                if scene_key in seen:
                    continue
                seen.add(scene_key)
                expanded_nodes.append((document, raw_metadata))

        return expanded_nodes

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

    def _raw_nodes_for_part(self, question: str, metadata: dict[str, Any]) -> list[Node]:
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

    def _expand_raw_neighbors(self, question: str, raw_nodes: list[Node]) -> list[Node]:
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
                part_cache[part_cache_key] = self._sort_raw_nodes(
                    self._raw_nodes_for_part(question, metadata)
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

    def _raw_only_retrieve(self, question: str) -> list[Node]:
        return self._hybrid_retrieve(
            question,
            n_results=self._config().raw_candidate_count,
            where={"summary_level": 4},
        )

    def query(self, question: str) -> str:
        """Executes the Hierarchical RAG query flow."""
        safe_print("Searching vector index for relevant context...")
        expanded_question = self._expanded_question(question)
        retrieved_nodes = self._hybrid_retrieve(
            expanded_question,
            n_results=self._config().routing_candidate_count,
        )

        if not retrieved_nodes:
            safe_print("Initial retrieval returned no hits; trying raw-scene retrieval...")
        raw_nodes = self._raw_evidence_nodes(retrieved_nodes)
        raw_nodes.extend(self._raw_evidence_nodes(self._raw_only_retrieve(expanded_question)))

        if retrieved_nodes:
            safe_print("Expanding summary hits to child raw scenes...")
            raw_nodes.extend(self._expand_summaries_to_raw_scenes(expanded_question, retrieved_nodes))

        raw_nodes = self._dedupe_nodes(raw_nodes)

        if not raw_nodes:
            return INSUFFICIENT_SOURCE_CONTEXT

        safe_print("Expanding neighboring raw evidence...")
        expanded_raw_nodes = self._expand_raw_neighbors(expanded_question, raw_nodes)
        ranked_raw_nodes = self._rank_raw_candidates(
            question,
            expanded_question,
            expanded_raw_nodes,
            raw_nodes,
        )
        final_raw_nodes = ranked_raw_nodes[: self._config().final_top_k]

        if not final_raw_nodes:
            return INSUFFICIENT_SOURCE_CONTEXT

        safe_print("Building answer context from raw source scenes...")
        return self._answer_from_raw_evidence(question, final_raw_nodes)
