import json
import os
import time
from collections import defaultdict
from typing import Any

from pydantic_ai.exceptions import (
    ModelConnectionError,
    ModelRateLimitError,
    ModelRequestTimeout,
    ModelServerError,
    ModelUnavailableError,
)

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

SUMMARIZATION_PROMPT_VERSION = "4"

# Seconds to wait before each retry of a transient model error (about 8 minutes in total).
RETRY_DELAYS = (15, 30, 60, 120, 240)
TRANSIENT_MODEL_ERRORS = (
    ModelRateLimitError,
    ModelUnavailableError,
    ModelServerError,
    ModelConnectionError,
    ModelRequestTimeout,
)
_sleep = time.sleep


def run_with_retries(instructions: str, prompt: str) -> str:
    """Run one generation call, retrying transient provider errors with backoff."""
    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            return create_generation_text_agent(instructions).run_sync(prompt).output.strip()
        except TRANSIENT_MODEL_ERRORS as exc:
            # Exhausted credits can arrive as a rate-limit error with HTTP 402; waiting won't help.
            if getattr(exc, "status_code", None) == 402 or attempt == len(RETRY_DELAYS):
                raise
            delay = max(RETRY_DELAYS[attempt], getattr(exc, "retry_after", None) or 0)
            safe_print(
                f"Transient model error ({type(exc).__name__}); retrying in {delay:.0f}s "
                f"(attempt {attempt + 2} of {len(RETRY_DELAYS) + 1})..."
            )
            _sleep(delay)
    raise AssertionError("unreachable")

PART_SUMMARY_SECTIONS = (
    "Overview",
    "Key Events",
    "Character Developments",
    "Continuity Facts",
    "Important Terms",
)
EPISODE_SUMMARY_SECTIONS = (
    "Overview",
    "Part Index",
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


def _section_label(line: str) -> str | None:
    """Return the section label a line declares, tolerating common formatting slips.

    Models occasionally emit a label without its colon (`Overview`) or wrapped in Markdown
    (`### Overview:`, `**Overview:**`); treat a line that is otherwise exactly a known label
    the same as the canonical `Overview:` form.
    """
    label = line.strip().lstrip("#").replace("**", "").strip().removesuffix(":").rstrip()
    return label if label in KNOWN_SUMMARY_SECTIONS else None


def normalize_summary_labels(summary: str) -> str:
    """Rewrite section label lines to the canonical `Label:` form."""
    lines = []
    for line in summary.splitlines():
        label = _section_label(line)
        lines.append(f"{label}:" if label is not None else line)
    return "\n".join(lines)


def extract_summary_sections(summary: str) -> dict[str, str]:
    """Extract known fixed-label summary sections from generated Markdown."""
    sections: dict[str, list[str]] = {}
    current_label: str | None = None

    for line in summary.splitlines():
        label = _section_label(line)
        if label is not None:
            current_label = label
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
    return """Faithfulness rules:
- State only what the input says. Every claim must be supported by a specific line or exchange in the input.
- When a speaker mentions two things in sequence (for example, a place and then an event, or one memory and then another), report them as separate facts. Do not connect them by location, time, or cause unless the input explicitly connects them.
- When characters recall past events, describe each memory as the speaker states it. Do not fill in where, when, or how it happened from general knowledge or from nearby lines.
- Before writing the final summary, check each Key Event and Continuity Fact against the input. If no line directly states it, rewrite it to match what the input says or remove it.

Global formatting rules:
- Write all summaries in clear, concise English.
- Use official glossary translations whenever available, following the glossary matching rules.
- Use exactly the required section labels for the tier.
- Always emit every required section.
- If a bullet-list section has no applicable entries, write exactly `- None`.
- Do not use Markdown headings, bold text, numbered lists, tables, or extra sections.
- Use bullet lists only under list sections.
- Important Terms should include glossary-mapped terms when relevant, plus salient new entities encountered in the text. It should not dump the full glossary."""


def build_verification_prompt(
    source_text: str,
    draft_summary: str,
    glossary: dict | None = None,
) -> tuple[str, str]:
    """Build the instructions and prompt for fact-checking a draft Part summary."""
    system_content = (
        "You are a meticulous fact-checker for English summaries of a Japanese story. "
        "You compare a draft summary against its Japanese source text and correct claims "
        "the source does not support."
    )
    characters = (glossary or {}).get("characters") or {}
    if characters:
        system_content += (
            "\n\nOfficial character names (Japanese -> English). The source often uses only a "
            "given name or surname; the full English name is correct for that character and "
            "must not be removed:\n"
            + "".join(f" - {jp} -> {en}\n" for jp, en in characters.items())
        )
    prompt = f"""--- SOURCE TEXT (JAPANESE) ---
{source_text}

--- DRAFT SUMMARY ---
{draft_summary}

Check every sentence and bullet of the draft against the source text. Look for these problems:
- Two separate statements merged into one claim, for example linking an event to a place, time, or cause that the source mentions only nearby.
- Details the source does not state, such as locations, timing, motives, or outcomes filled in from general knowledge.
- A person identified as someone the source does not name, or a nickname attached to the wrong character. A character's full official name is not an error when the source uses part of it.

Then output the complete corrected summary. Fix or remove only the unsupported claims; keep every supported sentence, all names, the section labels, and the format exactly as they are. Output only the summary, with no notes about the changes."""
    return system_content, prompt


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

Part Index:
- Part 1: central part event, conflict, or outcome in one line.
- Part 2: central part event, conflict, or outcome in one line.
- Interlude: brief recap of the interlude beat in one line.

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
- Characters, units, songs, events, locations, apps, competitions, aliases central to this episode.

Episode Part Index label rules:
- Use `Part N:` for numbered parts.
- Use stable English labels for non-numbered parts or interludes, such as `Interlude:` or `Ending:`.
- Do not use raw Japanese part titles as Part Index bullet labels when a generic label is available."""

    if level_name == "Year":
        return """Required Year summary format:

Overview:
[A detailed prose summary of the year's overall narrative movement, club status changes, competitions, graduations/transitions, and recurring themes. Keep episode boundaries clear; do not force unrelated episodes into larger arcs.]

Episode Index:
- Episode 1: central conflict and outcome in one line, about 20-30 words.
- Episode 2: central conflict and outcome in one line, about 20-30 words.

Character Trajectories:
- Character Name: year-long growth, setbacks, role changes, relationships, goals.
- Character Name: year-long growth, setbacks, role changes, relationships, goals.

Unit / Club State:
- Unit or club: membership, creative direction, conflicts, achievements, public status.

Continuity Facts:
- Year-level or cross-episode facts only: final states, competition results, graduations/transitions, promises, unresolved threads, and rare genuine multi-episode arcs.

Important Terms:
- Up to 15 major recurring or year-routing terms.

Year Episode Index label rules:
- Use `Episode N:` for numbered main episodes.
- Use stable English labels for non-numbered special entries, such as `Interlude:` or `Special:`.
- Do not use raw Japanese episode titles as Episode Index bullet labels.
- Japanese or official episode titles and aliases may still be preserved in prose or Important Terms when retrieval-useful."""

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
        verify_parts: bool = True,
    ):
        self.glossary = glossary
        self.story_order = story_order or default_story_order()
        self.cache_context = cache_context
        self.verify_parts = verify_parts

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
            system_content += (
                "Apply an entry only when the input uses that exact term, or an unambiguous "
                "reading of it, for the same referent. A character's given name or surname on "
                "its own (as in speaker labels, e.g. the given name of a full-name entry) "
                "matches that character's entry. Do not map a name to an entry because "
                "it looks or sounds similar (for example, a nickname or given name that shares "
                "a kanji or kana with a glossary name). If a name in the input has no matching "
                "entry, romanize it as written and do not identify it with a glossary character.\n"
                "Write each matched character's full glossary name at least once in every "
                "summary (for example, at first mention); shorter forms are fine afterward.\n"
            )
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

        summary = run_with_retries(system_content, prompt)
        if level_name == "Part" and self.verify_parts:
            summary = self._verify_part_summary(current_text, summary)
        return normalize_summary_labels(summary)

    def _verify_part_summary(self, source_text: str, draft_summary: str) -> str:
        """Run a fact-check pass that corrects claims the source text does not support."""
        system_content, prompt = build_verification_prompt(
            source_text, draft_summary, self.glossary
        )
        return run_with_retries(system_content, prompt) or draft_summary

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
            "part_verification": self.verify_parts,
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
