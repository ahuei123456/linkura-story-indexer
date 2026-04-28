from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from .indexer.processor import StoryProcessor
from .models.story import StoryNode

ALLOWED_STORY_TYPES = {"Main", "Side", "Other"}
DEFAULT_STORY_ORDER_PATH = Path("story_order.yaml")


class StoryOrderConfigError(ValueError):
    """Raised when the story order manifest is malformed or incomplete."""


def natural_sort_key(value: str) -> list[int | str]:
    """Sort strings naturally, so 'Part 2' comes before 'Part 10'."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", value)]


@dataclass(frozen=True)
class StoryOrder:
    chronological_positions: dict[tuple[str, str], tuple[int, int]]
    summary_positions: dict[tuple[str, str], tuple[int, int]]
    part_positions: dict[tuple[str, str, str], dict[str, int]]

    @classmethod
    def from_file(
        cls,
        config_path: str | Path = DEFAULT_STORY_ORDER_PATH,
        *,
        story_root: str | Path | None = None,
    ) -> StoryOrder:
        path = Path(config_path)
        if not path.exists():
            raise StoryOrderConfigError(f"Story order config not found: {path}")

        with open(path, encoding="utf-8") as file:
            data = yaml.safe_load(file)

        if not isinstance(data, dict):
            raise StoryOrderConfigError("Story order config must be a YAML mapping.")

        story_order = cls(
            chronological_positions=_parse_order(data, "chronological_order"),
            summary_positions=_parse_order(data, "summary_order"),
            part_positions=_parse_part_order_overrides(data),
        )
        if story_root is not None:
            story_order.validate_story_root(Path(story_root))
        return story_order

    def chronological_episode_key(
        self,
        arc_id: str,
        story_type: str,
        episode_name: str,
    ) -> tuple[Any, ...]:
        return self._episode_key(
            self.chronological_positions,
            "chronological_order",
            arc_id,
            story_type,
            episode_name,
        )

    def summary_episode_key(
        self,
        arc_id: str,
        story_type: str,
        episode_name: str,
    ) -> tuple[Any, ...]:
        return self._episode_key(
            self.summary_positions,
            "summary_order",
            arc_id,
            story_type,
            episode_name,
        )

    def chronological_node_key(self, node: StoryNode) -> tuple[Any, ...]:
        meta = node.metadata
        return (
            self.chronological_episode_key(meta.arc_id, meta.story_type, meta.episode_name),
            self.part_key(meta.arc_id, meta.story_type, meta.episode_name, meta.part_name),
            meta.scene_index,
            meta.file_path,
        )

    def summary_node_key(self, node: StoryNode) -> tuple[Any, ...]:
        meta = node.metadata
        return (
            self.summary_episode_key(meta.arc_id, meta.story_type, meta.episode_name),
            self.part_key(meta.arc_id, meta.story_type, meta.episode_name, meta.part_name),
            meta.scene_index,
            meta.file_path,
        )

    def part_key(
        self,
        arc_id: str,
        story_type: str,
        episode_name: str,
        part_name: str,
    ) -> tuple[Any, ...]:
        override = self.part_positions.get((arc_id, story_type, episode_name))
        if override is None:
            return (1, natural_sort_key(part_name))
        if part_name not in override:
            raise StoryOrderConfigError(
                "part_order_overrides for "
                f"{arc_id}|{story_type}|{episode_name} does not include part {part_name!r}."
            )
        return (0, override[part_name])

    def validate_story_root(self, story_root: Path) -> None:
        if not story_root.exists():
            raise StoryOrderConfigError(f"Story root not found: {story_root}")

        known_pairs: set[tuple[str, str]] = set()
        known_parts: dict[tuple[str, str, str], set[str]] = defaultdict(set)

        for story_file in story_root.rglob("*.md"):
            metadata = StoryProcessor.extract_hierarchy(story_file)
            pair = (metadata.story_type, metadata.arc_id)
            episode_key = (metadata.arc_id, metadata.story_type, metadata.episode_name)
            known_pairs.add(pair)
            known_parts[episode_key].add(metadata.part_name)

        self._validate_order_covers_story_pairs("chronological_order", known_pairs)
        self._validate_order_covers_story_pairs("summary_order", known_pairs)
        self._validate_order_references_known_story_pairs("chronological_order", known_pairs)
        self._validate_order_references_known_story_pairs("summary_order", known_pairs)
        self._validate_part_overrides(known_parts)

    def _episode_key(
        self,
        positions: dict[tuple[str, str], tuple[int, int]],
        order_name: str,
        arc_id: str,
        story_type: str,
        episode_name: str,
    ) -> tuple[Any, ...]:
        try:
            group_index, arc_index = positions[(story_type, arc_id)]
        except KeyError as exc:
            raise StoryOrderConfigError(
                f"{order_name} does not cover story_type={story_type!r}, arc_id={arc_id!r}."
            ) from exc
        return (group_index, arc_index, natural_sort_key(episode_name))

    def _validate_order_covers_story_pairs(
        self,
        order_name: str,
        known_pairs: set[tuple[str, str]],
    ) -> None:
        positions = (
            self.chronological_positions
            if order_name == "chronological_order"
            else self.summary_positions
        )
        missing_pairs = sorted(known_pairs - positions.keys())
        if missing_pairs:
            missing = ", ".join(
                f"story_type={story_type!r}, arc_id={arc_id!r}"
                for story_type, arc_id in missing_pairs
            )
            raise StoryOrderConfigError(f"{order_name} does not cover {missing}.")

    def _validate_order_references_known_story_pairs(
        self,
        order_name: str,
        known_pairs: set[tuple[str, str]],
    ) -> None:
        positions = (
            self.chronological_positions
            if order_name == "chronological_order"
            else self.summary_positions
        )
        unknown_pairs = sorted(positions.keys() - known_pairs)
        if unknown_pairs:
            unknown = ", ".join(
                f"story_type={story_type!r}, arc_id={arc_id!r}"
                for story_type, arc_id in unknown_pairs
            )
            raise StoryOrderConfigError(f"{order_name} references unknown {unknown}.")

    def _validate_part_overrides(
        self,
        known_parts: dict[tuple[str, str, str], set[str]],
    ) -> None:
        for episode_key, configured_positions in self.part_positions.items():
            if episode_key not in known_parts:
                arc_id, story_type, episode_name = episode_key
                raise StoryOrderConfigError(
                    "part_order_overrides references unknown episode "
                    f"{arc_id}|{story_type}|{episode_name}."
                )

            configured_parts = set(configured_positions)
            actual_parts = known_parts[episode_key]
            missing_parts = sorted(actual_parts - configured_parts, key=natural_sort_key)
            unknown_parts = sorted(configured_parts - actual_parts, key=natural_sort_key)
            if missing_parts or unknown_parts:
                arc_id, story_type, episode_name = episode_key
                details = []
                if missing_parts:
                    details.append(f"missing parts: {', '.join(missing_parts)}")
                if unknown_parts:
                    details.append(f"unknown parts: {', '.join(unknown_parts)}")
                raise StoryOrderConfigError(
                    "part_order_overrides for "
                    f"{arc_id}|{story_type}|{episode_name} is invalid: "
                    + "; ".join(details)
                    + "."
                )


def load_story_order(
    config_path: str | Path = DEFAULT_STORY_ORDER_PATH,
    *,
    story_root: str | Path | None = None,
) -> StoryOrder:
    return StoryOrder.from_file(config_path, story_root=story_root)


@cache
def default_story_order() -> StoryOrder:
    return load_story_order()


def _parse_order(data: dict[str, Any], field_name: str) -> dict[tuple[str, str], tuple[int, int]]:
    raw_order = data.get(field_name)
    if not isinstance(raw_order, list) or not raw_order:
        raise StoryOrderConfigError(f"{field_name} must be a non-empty list.")

    positions: dict[tuple[str, str], tuple[int, int]] = {}
    for group_index, entry in enumerate(raw_order):
        if not isinstance(entry, dict):
            raise StoryOrderConfigError(f"{field_name}[{group_index}] must be a mapping.")

        story_type = _story_type(entry, f"{field_name}[{group_index}].story_type")
        arcs = entry.get("arcs")
        if not isinstance(arcs, list) or not arcs:
            raise StoryOrderConfigError(f"{field_name}[{group_index}].arcs must be a non-empty list.")

        for arc_index, raw_arc_id in enumerate(arcs):
            arc_id = _arc_id(raw_arc_id, f"{field_name}[{group_index}].arcs[{arc_index}]")
            key = (story_type, arc_id)
            if key in positions:
                raise StoryOrderConfigError(
                    f"{field_name} contains duplicate story_type={story_type!r}, arc_id={arc_id!r}."
                )
            positions[key] = (group_index, arc_index)

    return positions


def _parse_part_order_overrides(data: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, int]]:
    raw_overrides = data.get("part_order_overrides", [])
    if raw_overrides is None:
        return {}
    if not isinstance(raw_overrides, list):
        raise StoryOrderConfigError("part_order_overrides must be a list.")

    overrides: dict[tuple[str, str, str], dict[str, int]] = {}
    for override_index, entry in enumerate(raw_overrides):
        if not isinstance(entry, dict):
            raise StoryOrderConfigError(f"part_order_overrides[{override_index}] must be a mapping.")

        prefix = f"part_order_overrides[{override_index}]"
        arc_id = _arc_id(entry.get("arc_id"), f"{prefix}.arc_id")
        story_type = _story_type(entry, f"{prefix}.story_type")
        episode_name = _required_str(entry.get("episode_name"), f"{prefix}.episode_name")
        parts = entry.get("parts")
        if not isinstance(parts, list) or not parts:
            raise StoryOrderConfigError(f"{prefix}.parts must be a non-empty list.")

        part_positions: dict[str, int] = {}
        for part_index, raw_part_name in enumerate(parts):
            part_name = _required_str(raw_part_name, f"{prefix}.parts[{part_index}]")
            if part_name in part_positions:
                raise StoryOrderConfigError(
                    f"{prefix}.parts contains duplicate part {part_name!r}."
                )
            part_positions[part_name] = part_index

        key = (arc_id, story_type, episode_name)
        if key in overrides:
            raise StoryOrderConfigError(
                "part_order_overrides contains duplicate override for "
                f"{arc_id}|{story_type}|{episode_name}."
            )
        overrides[key] = part_positions

    return overrides


def _story_type(entry: dict[str, Any], field_name: str) -> str:
    story_type = _required_str(entry.get("story_type"), field_name)
    if story_type not in ALLOWED_STORY_TYPES:
        allowed = ", ".join(sorted(ALLOWED_STORY_TYPES))
        raise StoryOrderConfigError(
            f"{field_name} has unknown story_type {story_type!r}; expected one of {allowed}."
        )
    return story_type


def _arc_id(value: Any, field_name: str) -> str:
    if isinstance(value, int):
        return str(value)
    return _required_str(value, field_name)


def _required_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise StoryOrderConfigError(f"{field_name} must be a non-empty string.")
    return value
