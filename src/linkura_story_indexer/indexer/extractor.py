import json
import os
from typing import Any

from pydantic_ai import Agent

from ..console import safe_print
from ..database import create_generation_model
from ..models.state import WorldState


class StateExtractor:
    """Extracts world facts from summaries using Map-Reduce to build the State Ledger."""

    def __init__(self, cache_file: str = "summaries_cache.json"):
        self.cache_file = cache_file
        self.agent = Agent(
            create_generation_model(),
            instructions=(
                "You are a strict archivist extracting factual world-building data from a "
                "story summary. Extract any characters with their roles, nicknames, and "
                "honorifics used for others, plus locations and important groups. Return "
                "data that strictly matches the requested schema."
            ),
            output_type=WorldState,
        )

    def extract_from_cache(self, output_file: str = "world_state.json"):
        """Reads Episode summaries from the cache, extracts facts, and merges them by Year."""
        if not os.path.exists(self.cache_file):
            safe_print(f"Error: {self.cache_file} not found. Please run the ingest command first.")
            return

        with open(self.cache_file, encoding="utf-8") as f:
            cache = json.load(f)

        # Map phase: Extract facts per Episode
        # We process Episodes (Tier 2) because they offer a good balance of detail and context
        episode_summaries = {}
        for key, value in cache.items():
            if not key.startswith("EPISODE|"):
                continue
            if isinstance(value, dict) and isinstance(value.get("summary"), str):
                episode_summaries[key] = value["summary"]
            elif isinstance(value, str):
                episode_summaries[key] = value
        
        if not episode_summaries:
            safe_print("No Episode summaries found in cache. Did ingestion complete Tier 2?")
            return

        safe_print(f"Found {len(episode_summaries)} Episode summaries. Starting Map-Reduce extraction...")

        arc_states: dict[str, dict[str, Any]] = {}

        for key, summary in episode_summaries.items():
            # Key format: EPISODE|{arc_id}|{story_type}|{episode_name}
            parts = key.split("|")
            arc_id = parts[1]
            episode_name = parts[3]

            safe_print(f"Extracting facts from Year {arc_id}, Episode: {episode_name}...")
            
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
        
        safe_print(f"\nState Ledger successfully written to {output_file}")

    def _extract_facts_from_summary(self, arc_id: str, summary: str) -> WorldState:
        """Uses the LLM to extract a structured WorldState from a summary text."""
        result = self.agent.run_sync(
            f"The story year (arc_id) is: {arc_id}\n\nStory Summary:\n{summary}\n"
        )
        return result.output

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
