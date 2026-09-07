"""Prepare parsed story records for local indexing services."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..models.story import StoryNode
from ..story_order import StoryOrder, default_story_order
from .processor import StoryProcessor


@dataclass
class PreparedCorpus:
    """Parsed source nodes and the inputs used to order them."""

    story_root: Path
    files: tuple[Path, ...]
    raw_nodes: list[StoryNode]
    story_order: StoryOrder


def assign_canonical_story_order(
    nodes: list[StoryNode],
    story_order: StoryOrder | None = None,
) -> None:
    """Set the stable chronological order on parsed source nodes."""
    order_config = story_order or default_story_order()
    for order, node in enumerate(sorted(nodes, key=order_config.chronological_node_key), start=1):
        node.metadata.canonical_story_order = order
        node.metadata.story_order = order


def prepare_corpus(
    story_root: str | Path,
    *,
    story_order: StoryOrder | None = None,
) -> PreparedCorpus:
    """Parse every Markdown source below ``story_root`` and order its nodes."""
    root = Path(story_root)
    if not root.exists():
        raise FileNotFoundError(f"Story root not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Story root is not a directory: {root}")

    files = tuple(sorted(root.rglob("*.md"), key=lambda path: str(path)))
    order_config = story_order or default_story_order()
    raw_nodes: list[StoryNode] = []
    for file_path in files:
        raw_nodes.extend(StoryProcessor.process_file(file_path, story_root=root))
    assign_canonical_story_order(raw_nodes, story_order=order_config)
    return PreparedCorpus(root, files, raw_nodes, order_config)
