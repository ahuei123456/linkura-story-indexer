import json
import os
from collections import defaultdict
from typing import Any

from ..console import safe_print
from ..database import create_text_agent
from ..models.story import StoryNode
from ..story_order import StoryOrder, default_story_order
from .manifest import (
    SUMMARY_CACHE_SCHEMA_VERSION,
    SummaryCacheContext,
    hash_text,
    stable_hash,
)

SUMMARIZATION_PROMPT_VERSION = "1"


def episode_sort_key(ep_key_tuple: tuple) -> tuple:
    arc_id, story_type, episode_name = ep_key_tuple
    return default_story_order().chronological_episode_key(arc_id, story_type, episode_name)


def _load_cache(cache_file: str) -> dict[str, Any]:
    if not os.path.exists(cache_file):
        return {}
    with open(cache_file, encoding="utf-8") as file:
        loaded = json.load(file)
    if not isinstance(loaded, dict):
        return {}
    return loaded


def _save_cache(cache_file: str, cache: dict[str, Any]) -> None:
    with open(cache_file, "w", encoding="utf-8") as file:
        json.dump(cache, file, ensure_ascii=False, indent=2)


def _cached_summary(cache: dict[str, Any], cache_key: str, fingerprint: str) -> str | None:
    entry = cache.get(cache_key)
    if not isinstance(entry, dict):
        return None
    if entry.get("fingerprint") != fingerprint:
        return None
    summary = entry.get("summary")
    if not isinstance(summary, str):
        return None
    return summary


def _store_cached_summary(
    cache: dict[str, Any],
    cache_key: str,
    *,
    summary: str,
    fingerprint: str,
    inputs: dict[str, Any],
) -> None:
    cache[cache_key] = {
        "schema_version": SUMMARY_CACHE_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "summary": summary,
        "inputs": inputs,
    }


class HierarchicalSummarizer:
    """Generates rolling summaries for stories to build the RAG hierarchy."""

    def __init__(
        self,
        glossary: dict | None = None,
        story_order: StoryOrder | None = None,
        cache_context: SummaryCacheContext | None = None,
    ):
        self.glossary = glossary
        self.story_order = story_order or default_story_order()
        self.cache_context = cache_context

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

        prompt = ""
        if prev_summary:
            prompt += f"--- PREVIOUS CONTEXT (For Continuity) ---\n{prev_summary}\n\n"
            prompt += f"--- CURRENT {level_name.upper()} TEXT (IN JAPANESE) ---\n{current_text}\n\n"
            prompt += f"Write a comprehensive English summary of the CURRENT {level_name.upper()} TEXT. Use the PREVIOUS CONTEXT ONLY to resolve pronouns (e.g., 'she', 'he') and understand the ongoing situation. Do NOT summarize the previous context again."
        else:
            prompt += f"--- CURRENT {level_name.upper()} TEXT (IN JAPANESE) ---\n{current_text}\n\n"
            prompt += f"Write a comprehensive English summary of the CURRENT {level_name.upper()} TEXT."

        result = create_text_agent(system_content).run_sync(prompt)
        return result.output.strip()

    def _base_fingerprint_inputs(self, level: str) -> dict[str, Any]:
        if self.cache_context is None:
            return {
                "level": level,
                "summary_cache_schema_version": SUMMARY_CACHE_SCHEMA_VERSION,
                "summarization_prompt_version": SUMMARIZATION_PROMPT_VERSION,
                "glossary_hash": stable_hash(self.glossary),
                "chat_model": "unconfigured",
                "embedding_model": "unconfigured",
                "parser_version": "unconfigured",
            }
        return {
            "level": level,
            "summary_cache_schema_version": self.cache_context.summary_cache_schema_version,
            "summarization_prompt_version": self.cache_context.summarization_prompt_version,
            "glossary_hash": self.cache_context.glossary_hash,
            "chat_model": self.cache_context.chat_model,
            "embedding_model": self.cache_context.embedding_model,
            "parser_version": self.cache_context.parser_version,
        }

    def _source_file_hashes_for_nodes(self, nodes: list[StoryNode]) -> dict[str, str]:
        grouped_text: dict[str, list[str]] = defaultdict(list)
        for node in nodes:
            grouped_text[node.metadata.file_path].append(node.text)

        hashes = {}
        for file_path, texts in sorted(grouped_text.items()):
            if (
                self.cache_context is not None
                and file_path in self.cache_context.source_file_hashes
            ):
                hashes[file_path] = self.cache_context.source_file_hashes[file_path]
            else:
                hashes[file_path] = hash_text("\n\n---\n\n".join(texts))
        return hashes

    def _part_cache_inputs(
        self,
        *,
        scenes: list[StoryNode],
        part_text: str,
        prev_summary: str | None,
    ) -> dict[str, Any]:
        return {
            **self._base_fingerprint_inputs("part"),
            "source_file_hashes": self._source_file_hashes_for_nodes(scenes),
            "source_text_hash": hash_text(part_text),
            "previous_summary_hash": hash_text(prev_summary) if prev_summary else "",
        }

    def _aggregate_cache_inputs(
        self,
        *,
        level: str,
        child_nodes: list[StoryNode],
        combined_text: str,
        prev_summary: str | None,
    ) -> dict[str, Any]:
        return {
            **self._base_fingerprint_inputs(level),
            "child_summary_hashes": [hash_text(node.text) for node in child_nodes],
            "combined_text_hash": hash_text(combined_text),
            "previous_summary_hash": hash_text(prev_summary) if prev_summary else "",
        }

    def summarize_hierarchy(self, raw_nodes: list[StoryNode], cache_file: str = "summaries_cache.json") -> list[StoryNode]:
        """
        Builds the full Tier 1-3 hierarchy.
        Returns a flat list of all generated Summary Nodes (Part, Episode, and Year levels)
        so they can all be embedded into the Vector DB.
        """
        all_summaries = []

        # 1. Generate Tier 3 (Part) Summaries
        safe_print("\n--- Generating Tier 3 (Part) Summaries ---")
        part_summaries = self.summarize_parts(raw_nodes, cache_file)
        all_summaries.extend(part_summaries)

        # 2. Generate Tier 2 (Episode) Summaries
        safe_print("\n--- Generating Tier 2 (Episode) Summaries ---")
        episode_summaries = self.summarize_episodes(part_summaries, cache_file)
        all_summaries.extend(episode_summaries)

        # 3. Generate Tier 1 (Year) Summaries
        safe_print("\n--- Generating Tier 1 (Year) Summaries ---")
        year_summaries = self.summarize_years(episode_summaries, cache_file)
        all_summaries.extend(year_summaries)

        return all_summaries

    def summarize_parts(self, raw_nodes: list[StoryNode], cache_file: str = "summaries_cache.json") -> list[StoryNode]:
        """
        Groups raw scenes by Episode -> Part, concatenates them, and generates
        Tier 3 (Part) summaries using a rolling context window.
        Uses a local cache file to resume if processing fails halfway.
        """
        cache = _load_cache(cache_file)

        # Group by (arc_id, story_type, episode_name)
        episodes: dict[tuple, dict[str, list[StoryNode]]] = defaultdict(lambda: defaultdict(list))
        
        for node in raw_nodes:
            meta = node.metadata
            ep_key = (meta.arc_id, meta.story_type, meta.episode_name)
            episodes[ep_key][meta.part_name].append(node)

        summary_nodes = []

        # Process each episode sequentially, globally sorted
        sorted_ep_keys = sorted(episodes.keys(), key=lambda ep_key: self.story_order.summary_episode_key(*ep_key))
        
        prev_summary = None
        
        for ep_key in sorted_ep_keys:
            arc_id, story_type, episode_name = ep_key
            parts = episodes[ep_key]
            
            # Sort parts naturally
            sorted_part_names = sorted(
                parts.keys(),
                key=lambda part_name: self.story_order.part_key(
                    arc_id,
                    story_type,
                    episode_name,
                    part_name,
                ),
            )
            
            for part_name in sorted_part_names:
                cache_key = f"{arc_id}|{story_type}|{episode_name}|{part_name}"
                
                # Sort scenes within the part by their parsed index
                scenes = sorted(parts[part_name], key=lambda n: n.metadata.scene_index)
                
                # We use the metadata of the first scene as a base, but mark it as a summary
                base_meta = scenes[0].metadata.model_copy(deep=True)
                base_meta.scene_index = -1 # Indicates it covers the whole part

                # Gemini 3 has a massive context window, so we can concatenate the whole part
                part_text = "\n\n---\n\n".join([n.text for n in scenes])
                cache_inputs = self._part_cache_inputs(
                    scenes=scenes,
                    part_text=part_text,
                    prev_summary=prev_summary,
                )
                fingerprint = stable_hash(cache_inputs)
                cached = _cached_summary(cache, cache_key, fingerprint)

                if cached is not None:
                    safe_print(f"Loading cached summary for {cache_key}...")
                    current_summary = cached
                else:
                    safe_print(f"Summarizing {cache_key}...")
                    
                    # Generate summary with rolling context
                    current_summary = self._generate_rolling_summary(
                        current_text=part_text, 
                        prev_summary=prev_summary, 
                        level_name="Part"
                    )
                    
                    # Save to cache
                    _store_cached_summary(
                        cache,
                        cache_key,
                        summary=current_summary,
                        fingerprint=fingerprint,
                        inputs=cache_inputs,
                    )
                    _save_cache(cache_file, cache)

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
        cache = _load_cache(cache_file)

        # Group by (arc_id, story_type, episode_name)
        episodes: dict[tuple, list[StoryNode]] = defaultdict(list)
        for node in part_nodes:
            meta = node.metadata
            ep_key = (meta.arc_id, meta.story_type, meta.episode_name)
            episodes[ep_key].append(node)

        summary_nodes = []
        
        sorted_ep_keys = sorted(episodes.keys(), key=lambda ep_key: self.story_order.summary_episode_key(*ep_key))
        
        prev_summary = None
        for ep_key in sorted_ep_keys:
            arc_id, story_type, episode_name = ep_key
            cache_key = f"EPISODE|{arc_id}|{story_type}|{episode_name}"
            parts = episodes[ep_key]

            # Sort parts to maintain narrative order
            parts = sorted(
                parts,
                key=lambda n: self.story_order.part_key(
                    n.metadata.arc_id,
                    n.metadata.story_type,
                    n.metadata.episode_name,
                    n.metadata.part_name,
                ),
            )
            
            base_meta = parts[0].metadata.model_copy(deep=True)
            base_meta.part_name = "ALL_PARTS" # Represents the whole episode

            combined_text = "\n\n---\n\n".join([f"Part: {n.metadata.part_name}\n{n.text}" for n in parts])
            cache_inputs = self._aggregate_cache_inputs(
                level="episode",
                child_nodes=parts,
                combined_text=combined_text,
                prev_summary=prev_summary,
            )
            fingerprint = stable_hash(cache_inputs)
            cached = _cached_summary(cache, cache_key, fingerprint)

            if cached is not None:
                safe_print(f"Loading cached episode summary for {cache_key}...")
                current_summary = cached
            else:
                safe_print(f"Summarizing Episode: {cache_key}...")
                
                current_summary = self._generate_rolling_summary(
                    current_text=combined_text, 
                    prev_summary=prev_summary, 
                    level_name="Episode"
                )
                
                _store_cached_summary(
                    cache,
                    cache_key,
                    summary=current_summary,
                    fingerprint=fingerprint,
                    inputs=cache_inputs,
                )
                _save_cache(cache_file, cache)

            summary_nodes.append(StoryNode(
                text=current_summary,
                metadata=base_meta,
                summary_level=2
            ))
            
            prev_summary = current_summary

        return summary_nodes

    def summarize_years(self, episode_nodes: list[StoryNode], cache_file: str = "summaries_cache.json") -> list[StoryNode]:
        """Aggregates Tier 2 Episode Summaries into Tier 1 Year Summaries."""
        cache = _load_cache(cache_file)

        # Group by arc_id (Year)
        years: dict[str, list[StoryNode]] = defaultdict(list)
        for node in episode_nodes:
            years[node.metadata.arc_id].append(node)

        summary_nodes = []
        
        sorted_years = sorted(
            years.keys(),
            key=lambda arc_id: min(
                self.story_order.summary_episode_key(
                    node.metadata.arc_id,
                    node.metadata.story_type,
                    node.metadata.episode_name,
                )
                for node in years[arc_id]
            ),
        )
        
        prev_summary = None
        for arc_id in sorted_years:
            cache_key = f"YEAR|{arc_id}"
            episodes = years[arc_id]
            
            # Sort episodes inside the year
            episodes = sorted(
                episodes,
                key=lambda n: self.story_order.summary_episode_key(
                    n.metadata.arc_id,
                    n.metadata.story_type,
                    n.metadata.episode_name,
                ),
            )
            
            base_meta = episodes[0].metadata.model_copy(deep=True)
            base_meta.episode_name = "ALL_EPISODES"
            base_meta.part_name = "ALL_PARTS"

            combined_text = "\n\n---\n\n".join([f"Episode: {n.metadata.episode_name}\n{n.text}" for n in episodes])
            cache_inputs = self._aggregate_cache_inputs(
                level="year",
                child_nodes=episodes,
                combined_text=combined_text,
                prev_summary=prev_summary,
            )
            fingerprint = stable_hash(cache_inputs)
            cached = _cached_summary(cache, cache_key, fingerprint)

            if cached is not None:
                safe_print(f"Loading cached year summary for {cache_key}...")
                current_summary = cached
            else:
                safe_print(f"Summarizing Year: {cache_key}...")
                
                current_summary = self._generate_rolling_summary(
                    current_text=combined_text, 
                    prev_summary=prev_summary, 
                    level_name="Year"
                )
                
                _store_cached_summary(
                    cache,
                    cache_key,
                    summary=current_summary,
                    fingerprint=fingerprint,
                    inputs=cache_inputs,
                )
                _save_cache(cache_file, cache)

            summary_nodes.append(StoryNode(
                text=current_summary,
                metadata=base_meta,
                summary_level=1
            ))
            
            prev_summary = current_summary

        return summary_nodes
