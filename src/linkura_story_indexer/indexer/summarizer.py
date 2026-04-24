import json
import os
import re
from collections import defaultdict

from llama_index.core import Settings
from llama_index.core.llms import ChatMessage, MessageRole

from ..models.story import StoryNode


def natural_sort_key(s: str) -> list:
    """Sorts strings naturally, so 'Part 2' comes before 'Part 10'."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def episode_sort_key(ep_key_tuple: tuple) -> tuple:
    """
    Sorts episodes chronologically according to the user's specified narrative flow:
    1. 103 Main
    2. 104 Main
    3. Side stories (102/103)
    4. 105 Main
    """
    arc_id, story_type, episode_name = ep_key_tuple
    
    if story_type == "Main":
        if arc_id == "103":
            group = 1
        elif arc_id == "104":
            group = 2
        elif arc_id == "105":
            group = 4
        else:
            group = 5
    else:
        group = 3
        
    return (group, int(arc_id) if arc_id.isdigit() else 999, natural_sort_key(episode_name))

class HierarchicalSummarizer:
    """Generates rolling summaries for stories to build the RAG hierarchy."""

    def __init__(self, glossary: dict | None = None):
        # We rely on Settings.llm which is configured in database.py
        self.llm = Settings.llm
        self.glossary = glossary

    def _generate_rolling_summary(self, current_text: str, prev_summary: str | None = None, level_name: str = "Part") -> str:
        """Calls the LLM to generate a summary using previous context to prevent drift."""
        system_content = (
            f"You are an expert archivist and translator indexing a Japanese narrative story. "
            f"You must write all summaries in clear, concise ENGLISH.\n"
            f"Summarize the following {level_name}. Focus on plot progression, character actions, and locations."
        )

        if self.glossary:
            system_content += "\n\n--- OFFICIAL GLOSSARY (MANDATORY TRANSLATIONS) ---\n"
            system_content += "When translating or referencing names and terms, you MUST use the following English equivalents:\n"
            for category, terms in self.glossary.items():
                system_content += f"\n{category.replace('_', ' ').upper()}:\n"
                for jp, en in terms.items():
                    system_content += f" - {jp} -> {en}\n"

        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=system_content
            )
        ]
        
        prompt = ""
        if prev_summary:
            prompt += f"--- PREVIOUS CONTEXT (For Continuity) ---\n{prev_summary}\n\n"
            prompt += f"--- CURRENT {level_name.upper()} TEXT (IN JAPANESE) ---\n{current_text}\n\n"
            prompt += f"Write a comprehensive English summary of the CURRENT {level_name.upper()} TEXT. Use the PREVIOUS CONTEXT ONLY to resolve pronouns (e.g., 'she', 'he') and understand the ongoing situation. Do NOT summarize the previous context again."
        else:
            prompt += f"--- CURRENT {level_name.upper()} TEXT (IN JAPANESE) ---\n{current_text}\n\n"
            prompt += f"Write a comprehensive English summary of the CURRENT {level_name.upper()} TEXT."

        messages.append(ChatMessage(role=MessageRole.USER, content=prompt))
        
        response = self.llm.chat(messages)
        content = response.message.content
        return content.strip() if content else ""

    def summarize_hierarchy(self, raw_nodes: list[StoryNode], cache_file: str = "summaries_cache.json") -> list[StoryNode]:
        """
        Builds the full Tier 1-3 hierarchy.
        Returns a flat list of all generated Summary Nodes (Part, Episode, and Year levels)
        so they can all be embedded into the Vector DB.
        """
        all_summaries = []

        # 1. Generate Tier 3 (Part) Summaries
        print("\n--- Generating Tier 3 (Part) Summaries ---")
        part_summaries = self.summarize_parts(raw_nodes, cache_file)
        all_summaries.extend(part_summaries)

        # 2. Generate Tier 2 (Episode) Summaries
        print("\n--- Generating Tier 2 (Episode) Summaries ---")
        episode_summaries = self.summarize_episodes(part_summaries, cache_file)
        all_summaries.extend(episode_summaries)

        # 3. Generate Tier 1 (Year) Summaries
        print("\n--- Generating Tier 1 (Year) Summaries ---")
        year_summaries = self.summarize_years(episode_summaries, cache_file)
        all_summaries.extend(year_summaries)

        return all_summaries

    def summarize_parts(self, raw_nodes: list[StoryNode], cache_file: str = "summaries_cache.json") -> list[StoryNode]:
        """
        Groups raw scenes by Episode -> Part, concatenates them, and generates
        Tier 3 (Part) summaries using a rolling context window.
        Uses a local cache file to resume if processing fails halfway.
        """
        cache: dict[str, str] = {}
        if os.path.exists(cache_file):
            with open(cache_file, encoding="utf-8") as f:
                cache = json.load(f)

        # Group by (arc_id, story_type, episode_name)
        episodes: dict[tuple, dict[str, list[StoryNode]]] = defaultdict(lambda: defaultdict(list))
        
        for node in raw_nodes:
            meta = node.metadata
            ep_key = (meta.arc_id, meta.story_type, meta.episode_name)
            episodes[ep_key][meta.part_name].append(node)

        summary_nodes = []

        # Process each episode sequentially, globally sorted
        sorted_ep_keys = sorted(episodes.keys(), key=episode_sort_key)
        
        prev_summary = None
        
        for ep_key in sorted_ep_keys:
            arc_id, story_type, episode_name = ep_key
            parts = episodes[ep_key]
            
            # Sort parts naturally
            sorted_part_names = sorted(parts.keys(), key=natural_sort_key)
            
            for part_name in sorted_part_names:
                cache_key = f"{arc_id}|{story_type}|{episode_name}|{part_name}"
                
                # Sort scenes within the part by their parsed index
                scenes = sorted(parts[part_name], key=lambda n: n.metadata.scene_index)
                
                # We use the metadata of the first scene as a base, but mark it as a summary
                base_meta = scenes[0].metadata.model_copy(deep=True)
                base_meta.scene_index = -1 # Indicates it covers the whole part

                if cache_key in cache:
                    print(f"Loading cached summary for {cache_key}...")
                    current_summary = cache[cache_key]
                else:
                    # Gemini 3 has a massive context window, so we can concatenate the whole part
                    part_text = "\n\n---\n\n".join([n.text for n in scenes])
                    
                    print(f"Summarizing {cache_key}...")
                    
                    # Generate summary with rolling context
                    current_summary = self._generate_rolling_summary(
                        current_text=part_text, 
                        prev_summary=prev_summary, 
                        level_name="Part"
                    )
                    
                    # Save to cache
                    cache[cache_key] = current_summary
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(cache, f, ensure_ascii=False, indent=2)

                summary_node = StoryNode(
                    text=current_summary,
                    metadata=base_meta,
                    summary_level=3
                )
                summary_nodes.append(summary_node)
                
                # Chain the context for the next part
                prev_summary = current_summary

        return summary_nodes

    def summarize_episodes(self, part_nodes: list[StoryNode], cache_file: str = "summaries_cache.json") -> list[StoryNode]:
        """Aggregates Tier 3 Part Summaries into Tier 2 Episode Summaries."""
        cache: dict[str, str] = {}
        if os.path.exists(cache_file):
            with open(cache_file, encoding="utf-8") as f:
                cache = json.load(f)

        # Group by (arc_id, story_type, episode_name)
        episodes: dict[tuple, list[StoryNode]] = defaultdict(list)
        for node in part_nodes:
            meta = node.metadata
            ep_key = (meta.arc_id, meta.story_type, meta.episode_name)
            episodes[ep_key].append(node)

        summary_nodes = []
        
        sorted_ep_keys = sorted(episodes.keys(), key=episode_sort_key)
        
        prev_summary = None
        for ep_key in sorted_ep_keys:
            arc_id, story_type, episode_name = ep_key
            cache_key = f"EPISODE|{arc_id}|{story_type}|{episode_name}"
            parts = episodes[ep_key]

            # Sort parts to maintain narrative order
            parts = sorted(parts, key=lambda n: natural_sort_key(n.metadata.part_name))
            
            base_meta = parts[0].metadata.model_copy(deep=True)
            base_meta.part_name = "ALL_PARTS" # Represents the whole episode

            if cache_key in cache:
                print(f"Loading cached episode summary for {cache_key}...")
                current_summary = cache[cache_key]
            else:
                combined_text = "\n\n---\n\n".join([f"Part: {n.metadata.part_name}\n{n.text}" for n in parts])
                print(f"Summarizing Episode: {cache_key}...")
                
                current_summary = self._generate_rolling_summary(
                    current_text=combined_text, 
                    prev_summary=prev_summary, 
                    level_name="Episode"
                )
                
                cache[cache_key] = current_summary
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(cache, f, ensure_ascii=False, indent=2)

            summary_nodes.append(StoryNode(
                text=current_summary,
                metadata=base_meta,
                summary_level=2
            ))
            
            prev_summary = current_summary

        return summary_nodes

    def summarize_years(self, episode_nodes: list[StoryNode], cache_file: str = "summaries_cache.json") -> list[StoryNode]:
        """Aggregates Tier 2 Episode Summaries into Tier 1 Year Summaries."""
        cache: dict[str, str] = {}
        if os.path.exists(cache_file):
            with open(cache_file, encoding="utf-8") as f:
                cache = json.load(f)

        # Group by arc_id (Year)
        years: dict[str, list[StoryNode]] = defaultdict(list)
        for node in episode_nodes:
            years[node.metadata.arc_id].append(node)

        summary_nodes = []
        
        # Sort years numerically
        sorted_years = sorted(years.keys(), key=lambda x: int(x) if x.isdigit() else 999)
        
        prev_summary = None
        for arc_id in sorted_years:
            cache_key = f"YEAR|{arc_id}"
            episodes = years[arc_id]
            
            # Sort episodes inside the year
            episodes = sorted(episodes, key=lambda n: episode_sort_key((n.metadata.arc_id, n.metadata.story_type, n.metadata.episode_name)))
            
            base_meta = episodes[0].metadata.model_copy(deep=True)
            base_meta.episode_name = "ALL_EPISODES"
            base_meta.part_name = "ALL_PARTS"

            if cache_key in cache:
                print(f"Loading cached year summary for {cache_key}...")
                current_summary = cache[cache_key]
            else:
                combined_text = "\n\n---\n\n".join([f"Episode: {n.metadata.episode_name}\n{n.text}" for n in episodes])
                print(f"Summarizing Year: {cache_key}...")
                
                current_summary = self._generate_rolling_summary(
                    current_text=combined_text, 
                    prev_summary=prev_summary, 
                    level_name="Year"
                )
                
                cache[cache_key] = current_summary
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(cache, f, ensure_ascii=False, indent=2)

            summary_nodes.append(StoryNode(
                text=current_summary,
                metadata=base_meta,
                summary_level=1
            ))
            
            prev_summary = current_summary

        return summary_nodes