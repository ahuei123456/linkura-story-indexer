import json
import os
from pathlib import Path

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.llms import ChatMessage, MessageRole

from ..database import get_vector_store


class StoryQueryEngine:
    def __init__(self, state_file: str = "world_state.json", glossary_file: str = "glossary.json"):
        self.llm = Settings.llm
        self.vector_store, self.storage_context = get_vector_store()
        
        # Load the index from the vector store
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            storage_context=self.storage_context
        )
        self.retriever = self.index.as_retriever(similarity_top_k=3)
        
        # Load invariants
        self.state_ledger = {}
        if os.path.exists(state_file):
            with open(state_file, encoding="utf-8") as f:
                self.state_ledger = json.load(f)
                
        self.glossary = None
        if os.path.exists(glossary_file):
            with open(glossary_file, encoding="utf-8") as f:
                self.glossary = json.load(f)

    def _build_system_prompt(self, arc_ids: set[str]) -> str:
        """Builds the system prompt with invariants and state ledger."""
        prompt = (
            "You are an expert lore-keeper and archivist for a Japanese narrative story.\n"
            "You are answering a user's question based strictly on the provided raw source text.\n"
            "Do NOT use outside knowledge. If the provided text does not contain the answer, say so.\n"
            "Cite your sources using the exact format [Year/Arc ID (e.g. 103, 104), Episode, Part]. Do not convert the Year/Arc ID to a real-world year like 2024.\n"
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

    def _fetch_raw_text(self, metadata: dict) -> str:
        """Fetches the raw text from disk based on the retrieved summary metadata."""
        level = metadata.get("summary_level", 4)
        file_path = metadata.get("file_path", "")
        
        if not file_path or not os.path.exists(file_path):
            return "Raw text not found."

        path = Path(file_path)
        
        if level == 3:
            # Tier 3 (Part): Read just this specific file
            with open(path, encoding="utf-8") as f:
                return f"Source: {metadata.get('episode_name')} - {metadata.get('part_name')}\n" + f.read()
        else:
            # For Tier 1 (Year) and Tier 2 (Episode), the raw text is too large.
            # We rely entirely on the node's summary text instead.
            return "Broad Summary retrieved; raw text omitted to preserve context window. Relying on the SUMMARY text provided above."

    def query(self, question: str) -> str:
        """Executes the Hierarchical RAG query flow."""
        print("Searching vector index for relevant summaries...")
        retrieved_nodes = self.retriever.retrieve(question)
        
        if not retrieved_nodes:
            return "No relevant information found in the index."

        arc_ids = set()
        raw_contexts = []
        
        print("Fetching raw text from disk for top matches...")
        for idx, node in enumerate(retrieved_nodes):
            meta = node.metadata
            arc_id = meta.get("arc_id")
            if arc_id:
                arc_ids.add(arc_id)
                
            print(f"  Match {idx+1}: Tier {meta.get('summary_level')} - Year {arc_id}, Ep: {meta.get('episode_name')}, Part: {meta.get('part_name')}")
            
            raw_text = self._fetch_raw_text(meta)
            # If it's a Tier 1 summary, the raw text fetcher might return the fallback message. 
            # We should include the node's own summary text as context in all cases just to be safe.
            context_chunk = f"--- RETRIEVED CONTEXT {idx+1} (Metadata: {json.dumps(meta, ensure_ascii=False)}) ---\n"
            context_chunk += f"SUMMARY:\n{node.text}\n\nRAW TEXT:\n{raw_text}\n"
            raw_contexts.append(context_chunk)

        system_prompt = self._build_system_prompt(arc_ids)
        combined_context = "\n".join(raw_contexts)
        
        user_prompt = (
            f"Please answer the following question based ONLY on the context provided below.\n\n"
            f"QUESTION: {question}\n\n"
            f"CONTEXT:\n{combined_context}"
        )

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=MessageRole.USER, content=user_prompt)
        ]

        print("Synthesizing final answer with Gemini...")
        response = self.llm.chat(messages)
        content = response.message.content
        return content.strip() if content else "No answer generated."
