import json
import os
from typing import Any

from llama_index.core import Settings
from llama_index.core.prompts import PromptTemplate

from ..models.state import WorldState


class StateExtractor:
    """Extracts world facts from summaries using Map-Reduce to build the State Ledger."""

    def __init__(self, cache_file: str = "summaries_cache.json"):
        self.llm = Settings.llm
        self.cache_file = cache_file

    def extract_from_cache(self, output_file: str = "world_state.json"):
        """Reads Episode summaries from the cache, extracts facts, and merges them by Year."""
        if not os.path.exists(self.cache_file):
            print(f"Error: {self.cache_file} not found. Please run the ingest command first.")
            return

        with open(self.cache_file, encoding="utf-8") as f:
            cache = json.load(f)

        # Map phase: Extract facts per Episode
        # We process Episodes (Tier 2) because they offer a good balance of detail and context
        episode_summaries = {k: v for k, v in cache.items() if k.startswith("EPISODE|")}
        
        if not episode_summaries:
            print("No Episode summaries found in cache. Did ingestion complete Tier 2?")
            return

        print(f"Found {len(episode_summaries)} Episode summaries. Starting Map-Reduce extraction...")

        arc_states: dict[str, dict[str, Any]] = {}

        for key, summary in episode_summaries.items():
            # Key format: EPISODE|{arc_id}|{story_type}|{episode_name}
            parts = key.split("|")
            arc_id = parts[1]
            episode_name = parts[3]

            print(f"Extracting facts from Year {arc_id}, Episode: {episode_name}...")
            
            extracted_state = self._extract_facts_from_summary(arc_id, summary)
            
            # Reduce phase: Merge into the global arc state
            if arc_id not in arc_states:
                arc_states[arc_id] = {
                    "characters": {},
                    "locations": set(),
                    "important_groups": set()
                }

            self._merge_states(arc_states[arc_id], extracted_state)

        # Format final output
        final_output = {}
        for arc_id, state in arc_states.items():
            chars_list = []
            for char_dict in state["characters"].values():
                char_copy = char_dict.copy()
                char_copy["nicknames"] = list(char_copy["nicknames"])
                chars_list.append(char_copy)

            final_output[arc_id] = {
                "characters": chars_list,
                "locations": list(state["locations"]),
                "important_groups": list(state["important_groups"])
            }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)
        
        print(f"\nState Ledger successfully written to {output_file}")

    def _extract_facts_from_summary(self, arc_id: str, summary: str) -> WorldState:
        """Uses the LLM to extract a structured WorldState from a summary text."""
        prompt = PromptTemplate(
            "You are a strict archivist extracting factual world-building data from a story summary.\n"
            "Extract any characters (with their roles, nicknames, and honorifics used for others), "
            "locations, and important groups mentioned in the text.\n"
            "Output this information strictly matching the requested JSON schema.\n"
            "The story year (arc_id) is: {arc_id}\n\n"
            "Story Summary:\n{summary}\n"
        )
        
        # We use structured_predict to force the LLM to return data matching the WorldState Pydantic model
        response = self.llm.structured_predict(WorldState, prompt, arc_id=arc_id, summary=summary)
        return response

    def _merge_states(self, global_state: dict[str, Any], new_state: WorldState):
        """Merges a newly extracted WorldState into the global arc state dictionary."""
        # Merge locations
        for loc in new_state.locations:
            global_state["locations"].add(loc)

        # Merge important groups
        for group in new_state.important_groups:
            global_state["important_groups"].add(group)

        # Merge characters
        for char in new_state.characters:
            name = char.name
            if name not in global_state["characters"]:
                global_state["characters"][name] = {
                    "name": name,
                    "role": char.role,
                    "nicknames": set(char.nicknames),
                    "honorifics_used": {h.target_character: h.honorific for h in char.honorifics_used},
                    "is_active": char.is_active
                }
            else:
                existing_char = global_state["characters"][name]
                # Update role if the new one is more descriptive (simple heuristic: longer is better, or just append)
                if len(char.role) > len(existing_char["role"]):
                    existing_char["role"] = char.role
                
                # Merge nicknames
                for nick in char.nicknames:
                    existing_char["nicknames"].add(nick)
                
                # Merge honorifics
                for h_fact in char.honorifics_used:
                    existing_char["honorifics_used"][h_fact.target_character] = h_fact.honorific

                # If ever marked active, keep active
                existing_char["is_active"] = existing_char["is_active"] or char.is_active

        # Before saving the dict back to JSON later, the sets will need to be converted to lists, 
        # which is handled in the format final output step.
