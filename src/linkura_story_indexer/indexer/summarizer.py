import json
import os
from collections import defaultdict
from typing import Any

from ..console import safe_print
from ..database import create_generation_text_agent
from ..models.story import StoryNode
from ..story_order import StoryOrder, default_story_order
from .manifest import (
    SUMMARY_CACHE_SCHEMA_VERSION,
    SummaryCacheContext,
    hash_text,
    stable_hash,
)

SUMMARIZATION_PROMPT_VERSION = "2"

PART_SUMMARY_SECTIONS = (
    "Overview",
    "Key Events",
    "Character Developments",
    "Continuity Facts",
    "Important Terms",
)
EPISODE_SUMMARY_SECTIONS = (
    "Overview",
    "Episode Arc",
    "Character Developments",
    "Relationship / Unit Developments",
    "Continuity Facts",
    "Important Terms",
)
YEAR_SUMMARY_SECTIONS = (
    "Overview",
    "Episode Index",
    "Character Trajectories",
    "Unit / Club State",
    "Continuity Facts",
    "Important Terms",
)
SUMMARY_SECTIONS_BY_LEVEL = {
    "Part": PART_SUMMARY_SECTIONS,
    "Episode": EPISODE_SUMMARY_SECTIONS,
    "Year": YEAR_SUMMARY_SECTIONS,
}
KNOWN_SUMMARY_SECTIONS = frozenset(
    PART_SUMMARY_SECTIONS + EPISODE_SUMMARY_SECTIONS + YEAR_SUMMARY_SECTIONS
)


def extract_summary_sections(summary: str) -> dict[str, str]:
    """Extract known fixed-label summary sections from generated Markdown."""
    sections: dict[str, list[str]] = {}
    current_label: str | None = None

    for line in summary.splitlines():
        stripped = line.strip()
        if stripped.endswith(":") and stripped[:-1] in KNOWN_SUMMARY_SECTIONS:
            current_label = stripped[:-1]
            sections.setdefault(current_label, [])
            continue

        if current_label is not None:
            sections[current_label].append(line)

    return {label: "\n".join(lines).strip() for label, lines in sections.items()}


def trim_previous_summary_context(prev_summary: str | None) -> str | None:
    """Keep only previous sections useful for resolving current references."""
    if not prev_summary:
        return None

    sections = extract_summary_sections(prev_summary)
    trimmed_parts = []
    for label in ("Overview", "Continuity Facts"):
        content = sections.get(label)
        if content:
            trimmed_parts.append(f"{label}:\n{content}")

    if not trimmed_parts:
        return None

    return "\n\n".join(trimmed_parts)


def _summary_input_label(level_name: str) -> str:
    if level_name == "Part":
        return "CURRENT PART TEXT (RAW PARSED STORY TEXT)"
    if level_name == "Episode":
        return "CURRENT EPISODE INPUT (STRUCTURED PART SUMMARIES)"
    if level_name == "Year":
        return "CURRENT YEAR INPUT (STRUCTURED EPISODE SUMMARIES)"
    return f"CURRENT {level_name.upper()} INPUT"


def _summary_input_instructions(level_name: str) -> str:
    if level_name == "Part":
        return (
            "The current Part input is raw parsed story text. Summarize only the current "
            "part's source text, using previous context only when needed to understand references."
        )
    if level_name == "Episode":
        return (
            "The current Episode input is multiple structured Part summaries. Synthesize across "
            "the child Part summaries into one episode-level summary. Do not concatenate, copy, "
            "or preserve child section structures verbatim."
        )
    if level_name == "Year":
        return (
            "The current Year input is multiple structured Episode summaries. Synthesize across "
            "the child Episode summaries into a year-level episode routing index and status "
            "summary. Do not concatenate, copy, or preserve child section structures verbatim."
        )
    return f"The current input is a {level_name}."


def _global_summary_format_rules() -> str:
    return """Global formatting rules:
- Write all summaries in clear, concise English.
- Use official glossary translations whenever available.
- Use exactly the required section labels for the tier.
- Always emit every required section.
- If a bullet-list section has no applicable entries, write exactly `- None`.
- Do not use Markdown headings, bold text, numbered lists, tables, or extra sections.
- Use bullet lists only under list sections.
- Important Terms should include glossary-mapped terms when relevant, plus salient new entities encountered in the text. It should not dump the full glossary."""


def _summary_format_instructions(level_name: str) -> str:
    if level_name == "Part":
        return """Required Part summary format:

Overview:
[Current-style detailed prose summary, usually 4-8 paragraphs for substantive parts; shorter only if the source is very short. Focus on concrete scene progression, character actions, locations, and immediate outcomes. Do not compress for brevity.]

Key Events:
- Chronological event.
- Chronological event.
- Chronological event.

Character Developments:
- Character Name: concrete emotional, relational, or goal-state change.
- Character Name: concrete emotional, relational, or goal-state change.

Continuity Facts:
- Stable fact, promise, conflict, reveal, relationship change, location, event result, or setup for later.
- Stable fact, promise, conflict, reveal, relationship change, location, event result, or setup for later.

Important Terms:
- Characters, unit names, locations, events, songs, apps, competitions, notable Japanese/English aliases."""

    if level_name == "Episode":
        return """Required Episode summary format:

Overview:
[A detailed prose recap of the whole episode. Preserve the current useful length. Focus on the episode's central conflict, progression, turning points, and resolution. Do not compress into a short abstract.]

Episode Arc:
- Setup: ...
- Escalation: ...
- Turning Point: ...
- Resolution: ...
- Aftermath / Setup: ...

Character Developments:
- Character Name: what changes for them across the episode.
- Character Name: what changes for them across the episode.

Relationship / Unit Developments:
- Pair, group, or unit: how the relationship or dynamic changes.

Continuity Facts:
- Durable facts established by this episode.
- Promises, conflicts, decisions, outcomes, new goals, event results.

Important Terms:
- Characters, units, songs, events, locations, apps, competitions, aliases central to this episode."""

    if level_name == "Year":
        return """Required Year summary format:

Overview:
[A detailed prose summary of the year's overall narrative movement, club status changes, competitions, graduations/transitions, and recurring themes. Keep episode boundaries clear; do not force unrelated episodes into larger arcs.]

Episode Index:
- Episode Name: central conflict and outcome in one line, about 20-30 words.
- Episode Name: central conflict and outcome in one line, about 20-30 words.

Character Trajectories:
- Character Name: year-long growth, setbacks, role changes, relationships, goals.
- Character Name: year-long growth, setbacks, role changes, relationships, goals.

Unit / Club State:
- Unit or club: membership, creative direction, conflicts, achievements, public status.

Continuity Facts:
- Year-level or cross-episode facts only: final states, competition results, graduations/transitions, promises, unresolved threads, and rare genuine multi-episode arcs.

Important Terms:
- Up to 15 major recurring or year-routing terms."""

    raise ValueError(f"Unsupported summary level: {level_name}")


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

    def _build_summary_prompt(
        self,
        current_text: str,
        prev_summary: str | None = None,
        level_name: str = "Part",
    ) -> tuple[str, str]:
        """Build the system instructions and prompt for a tier-specific summary."""
        system_content = (
            "You are an expert archivist and translator indexing a Japanese narrative story. "
            "You must write all summaries in clear, concise ENGLISH.\n"
            f"Summarize the following {level_name}. Focus on plot progression, character actions, locations, continuity, and retrieval-useful named entities."
        )

        if self.glossary:
            system_content += "\n\n--- OFFICIAL GLOSSARY (MANDATORY TRANSLATIONS) ---\n"
            system_content += "When translating or referencing names and terms, you MUST use the following English equivalents:\n"
            for category, terms in self.glossary.items():
                system_content += f"\n{category.replace('_', ' ').upper()}:\n"
                for jp, en in terms.items():
                    system_content += f" - {jp} -> {en}\n"

        prev_context = trim_previous_summary_context(prev_summary)
        prompt_parts = []
        if prev_context:
            prompt_parts.append(f"--- PREVIOUS CONTEXT (For Continuity) ---\n{prev_context}")

        prompt_parts.extend(
            [
                f"--- {_summary_input_label(level_name)} ---\n{current_text}",
                _summary_input_instructions(level_name),
            ]
        )

        if prev_context:
            prompt_parts.append(
                "Use previous context only to resolve references, pronouns, chronology, and ongoing situations needed to understand the current input. Do not summarize previous events again, do not copy previous sections, and do not include prior events unless the current input directly depends on them."
            )

        prompt_parts.extend(
            [
                _global_summary_format_rules(),
                _summary_format_instructions(level_name),
                f"Write the {level_name} summary now using exactly the required format.",
            ]
        )

        return system_content, "\n\n".join(prompt_parts)

    def _generate_rolling_summary(
        self,
        current_text: str,
        prev_summary: str | None = None,
        level_name: str = "Part",
    ) -> str:
        """Calls the LLM to generate a summary using previous context to prevent drift."""
        system_content, prompt = self._build_summary_prompt(current_text, prev_summary, level_name)

        result = create_generation_text_agent(system_content).run_sync(prompt)
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
        inputs = {
            "level": level,
            "summary_cache_schema_version": self.cache_context.summary_cache_schema_version,
            "summarization_prompt_version": self.cache_context.summarization_prompt_version,
            "glossary_hash": self.cache_context.glossary_hash,
            "chat_model": self.cache_context.chat_model,
            "embedding_model": self.cache_context.embedding_model,
            "parser_version": self.cache_context.parser_version,
        }
        if (
            self.cache_context.generation_provider != "google"
            or self.cache_context.generation_model != self.cache_context.chat_model
        ):
            inputs["generation_provider"] = self.cache_context.generation_provider
            inputs["generation_model"] = self.cache_context.generation_model
        return inputs

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
                prev_context = trim_previous_summary_context(prev_summary)
                cache_inputs = self._part_cache_inputs(
                    scenes=scenes,
                    part_text=part_text,
                    prev_summary=prev_context,
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
                        prev_summary=prev_context,
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
            prev_context = trim_previous_summary_context(prev_summary)
            cache_inputs = self._aggregate_cache_inputs(
                level="episode",
                child_nodes=parts,
                combined_text=combined_text,
                prev_summary=prev_context,
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
                    prev_summary=prev_context,
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
            prev_context = trim_previous_summary_context(prev_summary)
            cache_inputs = self._aggregate_cache_inputs(
                level="year",
                child_nodes=episodes,
                combined_text=combined_text,
                prev_summary=prev_context,
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
                    prev_summary=prev_context,
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
