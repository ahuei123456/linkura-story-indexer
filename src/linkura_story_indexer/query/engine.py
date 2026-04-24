import json
import os
import re
from pathlib import Path
from typing import Any

from ..console import safe_print
from ..database import create_text_agent, embed_texts, get_chroma_collection


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

    def _build_system_prompt(self, arc_ids: set[str]) -> str:
        """Builds the system prompt with invariants and state ledger."""
        prompt = (
            "You are an expert lore-keeper and archivist for a Japanese narrative story.\n"
            "You are answering a user's question based strictly on the provided raw source text.\n"
            "Do NOT use outside knowledge. If the provided text does not contain the answer, say so.\n"
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
                    prompt += json.dumps(self.state_ledger[arc_id], ensure_ascii=False, indent=2) + "\n"

        return prompt

    def _citation_label(self, metadata: dict[str, Any]) -> str:
        arc_id = metadata.get("arc_id", "unknown")
        level = metadata.get("summary_level", 4)
        if level == 1:
            return f"[{arc_id}]"

        episode = self._episode_label(metadata)
        if level == 2:
            return f"[{arc_id}, {episode}]"

        part = metadata.get("part_name", "unknown")
        return f"[{arc_id}, {episode}, Part {part}]"

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
        """Fetches the raw text from disk based on the retrieved summary metadata."""
        level = metadata.get("summary_level", 4)
        file_path = metadata.get("file_path", "")

        if not file_path or not os.path.exists(file_path):
            return "Raw text not found."

        path = Path(file_path)

        if level == 3:
            with open(path, encoding="utf-8") as f:
                return f"Source: {metadata.get('episode_name')} - {metadata.get('part_name')}\n" + f.read()

        return (
            "Broad Summary retrieved; raw text omitted to preserve context window. "
            "Relying on the SUMMARY text provided above."
        )

    def _retrieve(self, question: str) -> list[tuple[str, dict[str, Any]]]:
        query_embedding = embed_texts([question])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=["documents", "metadatas"],
        )

        documents = results.get("documents") or [[]]
        metadatas = results.get("metadatas") or [[]]
        return [
            (document, dict(metadata or {}))
            for document, metadata in zip(documents[0], metadatas[0], strict=False)
        ]

    def query(self, question: str) -> str:
        """Executes the Hierarchical RAG query flow."""
        safe_print("Searching vector index for relevant summaries...")
        retrieved_nodes = self._retrieve(question)

        if not retrieved_nodes:
            return "No relevant information found in the index."

        arc_ids = set()
        raw_contexts = []

        safe_print("Fetching raw text from disk for top matches...")
        for idx, (summary, meta) in enumerate(retrieved_nodes):
            arc_id = meta.get("arc_id")
            if isinstance(arc_id, str):
                arc_ids.add(arc_id)

            safe_print(
                f"  Match {idx + 1}: Tier {meta.get('summary_level')} - Year {arc_id}, "
                f"Ep: {meta.get('episode_name')}, Part: {meta.get('part_name')}"
            )

            raw_text = self._fetch_raw_text(meta)
            citation = self._citation_label(meta)
            context_chunk = (
                f"--- RETRIEVED CONTEXT {idx + 1} "
                f"(Citation: {citation}; Metadata: {json.dumps(meta, ensure_ascii=False)}) ---\n"
            )
            context_chunk += f"SUMMARY:\n{summary}\n\nRAW TEXT:\n{raw_text}\n"
            raw_contexts.append(context_chunk)

        system_prompt = self._build_system_prompt(arc_ids)
        combined_context = "\n".join(raw_contexts)

        user_prompt = (
            "Please answer the following question based ONLY on the context provided below.\n\n"
            "Every factual claim should cite one or more provided CITATION labels exactly as written. "
            "Use episode numbers in citations, never Japanese episode titles.\n\n"
            f"QUESTION: {question}\n\n"
            f"CONTEXT:\n{combined_context}"
        )

        safe_print("Synthesizing final answer with Gemini...")
        result = create_text_agent(system_prompt).run_sync(user_prompt)
        return result.output.strip() or "No answer generated."
