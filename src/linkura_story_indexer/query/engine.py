import json
import os
import re
from pathlib import Path
from typing import Any

from ..console import safe_print
from ..database import RETRIEVAL_QUERY, create_text_agent, embed_texts, get_chroma_collection
from ..indexer.parser import StoryParser

INITIAL_RESULT_COUNT = 3
RAW_FALLBACK_RESULT_COUNT = 5
INSUFFICIENT_SOURCE_CONTEXT = (
    "Insufficient source context: no raw source scenes were found for this question."
)


class StoryQueryEngine:
    def __init__(self, state_file: str = "world_state.json", glossary_file: str = "glossary.json"):
        self.collection = get_chroma_collection()

        self.state_ledger: dict[str, Any] = {}
        if os.path.exists(state_file):
            with open(state_file, encoding="utf-8") as f:
                self.state_ledger = json.load(f)

        self.glossary: dict[str, dict[str, str]] | None = None
        if os.path.exists(glossary_file):
            with open(glossary_file, encoding="utf-8") as f:
                self.glossary = json.load(f)

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
        n_results: int = INITIAL_RESULT_COUNT,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[str, dict[str, Any]]]:
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
        summaries: list[tuple[str, dict[str, Any]]],
    ) -> list[tuple[str, dict[str, Any]]]:
        expanded_nodes: list[tuple[str, dict[str, Any]]] = []
        seen: set[tuple[str, int, int]] = set()

        for _, metadata in summaries:
            raw_filter = self._raw_scene_filter_for_summary(metadata)
            if raw_filter is None:
                continue

            for document, raw_metadata in self._retrieve(
                question,
                n_results=RAW_FALLBACK_RESULT_COUNT,
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
        nodes: list[tuple[str, dict[str, Any]]],
    ) -> list[tuple[str, dict[str, Any]]]:
        return [(document, metadata) for document, metadata in nodes if metadata.get("summary_level") == 4]

    def _build_context_chunks(self, raw_nodes: list[tuple[str, dict[str, Any]]]) -> list[str]:
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
        raw_nodes: list[tuple[str, dict[str, Any]]],
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

    def _raw_only_retrieve(self, question: str) -> list[tuple[str, dict[str, Any]]]:
        results = self.collection.query(
            query_embeddings=[embed_texts([question], task_type=RETRIEVAL_QUERY)[0]],
            n_results=RAW_FALLBACK_RESULT_COUNT,
            where={"summary_level": 4},
            include=["documents", "metadatas"],
        )
        return self._results_to_nodes(results)

    def query(self, question: str) -> str:
        """Executes the Hierarchical RAG query flow."""
        safe_print("Searching vector index for relevant context...")
        retrieved_nodes = self._retrieve(question)

        if not retrieved_nodes:
            safe_print("Initial retrieval returned no hits; trying raw-scene retrieval...")
            raw_nodes = self._raw_evidence_nodes(self._raw_only_retrieve(question))
        else:
            raw_nodes = self._raw_evidence_nodes(retrieved_nodes)

        if not raw_nodes and retrieved_nodes:
            safe_print("Initial retrieval found only summaries; expanding to child raw scenes...")
            raw_nodes = self._expand_summaries_to_raw_scenes(question, retrieved_nodes)

        if not raw_nodes:
            return INSUFFICIENT_SOURCE_CONTEXT

        safe_print("Building answer context from raw source scenes...")
        return self._answer_from_raw_evidence(question, raw_nodes)
