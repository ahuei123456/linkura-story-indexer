import json
from pathlib import Path
from typing import Any

from pydantic_ai import Agent

from ..console import safe_print
from ..database import create_generation_model
from ..models.state import (
    SINGLE_CURRENT_PREDICATES,
    STATE_PREDICATES,
    TARGET_REQUIRED_PREDICATES,
    TARGET_UNUSED_PREDICATES,
    SceneStateExtraction,
    StateFact,
    StateLedger,
)
from .source_store import DEFAULT_SOURCE_DB_PATH, SourceRecordStore


def _story_order(metadata: dict[str, Any]) -> int:
    order = metadata.get("story_order", metadata.get("canonical_story_order"))
    return int(order) if isinstance(order, int) else 0


class StateExtractor:
    """Extracts source-backed state facts from persisted raw scenes."""

    def __init__(
        self,
        source_db_path: str | Path | None = None,
        *,
        cache_file: str | None = None,
    ) -> None:
        self.source_db_path = source_db_path or DEFAULT_SOURCE_DB_PATH
        self.cache_file = cache_file
        self.source_store = SourceRecordStore(self.source_db_path)
        predicate_list = ", ".join(STATE_PREDICATES)
        target_required = ", ".join(sorted(TARGET_REQUIRED_PREDICATES))
        target_unused = ", ".join(sorted(TARGET_UNUSED_PREDICATES))
        self.agent = Agent(
            create_generation_model(),
            instructions=(
                "You are a strict archivist extracting atomic world-state facts from raw "
                "story scenes. Extract only facts directly supported by the provided scene. "
                f"Use only these predicates: {predicate_list}. "
                "Use status only for transient or semi-durable conditions, not roles or "
                "life stages. Use emotional_stance_toward only for durable emotional "
                "orientations, not one-scene feelings. Use attribute sparingly for durable "
                "traits or skills stated as facts. "
                f"Predicates requiring target: {target_required}. "
                f"Predicates that must not set target: {target_unused}. "
                "For optional target predicates, set target only when the fact is directed "
                "at a specific person, group, object, place, event, memory, or abstract "
                "concept. Every fact must include an exact extracted_quote copied from the "
                "scene text. Return only facts that match the requested schema."
            ),
            output_type=SceneStateExtraction,
        )

    def extract_from_sources(self, output_file: str = "world_state.json") -> StateLedger:
        scenes = self.source_store.iter_scenes()
        if not scenes:
            safe_print(
                f"No raw scenes found in {self.source_db_path}. Run ingest before extracting state."
            )
            ledger = StateLedger()
            self._write_ledger(ledger, output_file)
            return ledger

        safe_print(f"Found {len(scenes)} raw scenes. Starting state fact extraction...")
        facts: list[StateFact] = []
        for scene in scenes:
            metadata = scene["metadata"]
            scene_facts = self._extract_facts_from_scene(str(scene["text"]), metadata)
            for fact in scene_facts.facts:
                if fact.extracted_quote not in scene["text"]:
                    safe_print(
                        "Skipping state fact with quote not found in source scene: "
                        f"{fact.subject} / {fact.predicate}"
                    )
                    continue
                facts.append(
                    StateFact(
                        subject=fact.subject,
                        predicate=fact.predicate,
                        target=fact.target,
                        object=fact.object,
                        confidence=fact.confidence,
                        extracted_quote=fact.extracted_quote,
                        arc=str(metadata.get("arc_id", "")),
                        episode=str(metadata.get("episode_name", "")),
                        part=str(metadata.get("part_name", "")),
                        scene=int(scene["scene_index"]),
                        valid_from=_story_order(metadata),
                        valid_to=None,
                        file_path=str(scene["file_path"]),
                        scene_index=int(scene["scene_index"]),
                    )
                )

        ledger = StateLedger(facts=self._with_valid_to(facts))
        self._write_ledger(ledger, output_file)
        safe_print(f"\nState Ledger successfully written to {output_file}")
        return ledger

    def extract_from_cache(self, output_file: str = "world_state.json") -> StateLedger:
        """Compatibility wrapper. State is now extracted from raw source scenes."""
        if self.cache_file:
            safe_print(
                "Ignoring summaries cache for state extraction; using persisted raw scenes instead."
            )
        return self.extract_from_sources(output_file=output_file)

    def _extract_facts_from_scene(self, scene_text: str, metadata: dict[str, Any]) -> SceneStateExtraction:
        result = self.agent.run_sync(
            "Extract source-backed state facts from this raw scene.\n\n"
            f"Year/arc: {metadata.get('arc_id', '')}\n"
            f"Episode: {metadata.get('episode_name', '')}\n"
            f"Part: {metadata.get('part_name', '')}\n"
            f"Scene index: {metadata.get('scene_index', '')}\n\n"
            f"RAW SCENE:\n{scene_text}\n"
        )
        return result.output

    def _with_valid_to(self, facts: list[StateFact]) -> list[StateFact]:
        sorted_facts = sorted(
            facts,
            key=lambda fact: (
                fact.valid_from,
                fact.arc,
                fact.part,
                fact.scene,
                fact.subject,
                fact.predicate,
                fact.target or "",
                fact.object,
                fact.extracted_quote,
            ),
        )
        open_facts: dict[tuple[str, ...], list[int]] = {}
        for index, fact in enumerate(sorted_facts):
            key = self._supersession_key(fact)
            previous_indices = open_facts.get(key, [])
            if previous_indices:
                previous_object = sorted_facts[previous_indices[-1]].object
            else:
                previous_object = None
            if previous_object is not None and previous_object != fact.object:
                for previous_index in previous_indices:
                    previous = sorted_facts[previous_index]
                    if previous.valid_to is None:
                        sorted_facts[previous_index] = previous.model_copy(
                            update={"valid_to": fact.valid_from}
                        )
                open_facts[key] = [index]
            else:
                open_facts.setdefault(key, []).append(index)
        return sorted_facts

    def _supersession_key(self, fact: StateFact) -> tuple[str, ...]:
        key = (fact.arc, fact.subject, fact.predicate, fact.target or "")
        if fact.predicate in SINGLE_CURRENT_PREDICATES:
            return key
        return (*key, fact.object)

    def _write_ledger(self, ledger: StateLedger, output_file: str) -> None:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(ledger.model_dump(), f, ensure_ascii=False, indent=2)
