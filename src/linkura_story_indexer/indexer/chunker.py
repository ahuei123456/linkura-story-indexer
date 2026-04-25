from itertools import groupby

from ..models.story import StoryNode

MIN_USEFUL_CHARS = 500
TARGET_CHUNK_CHARS = 1200
MAX_CHUNK_CHARS = 1800
SCENE_SEPARATOR = "\n\n---\n\n"


def _part_key(node: StoryNode) -> tuple[str, str]:
    meta = node.metadata
    return meta.parent_part_id, meta.file_path


def _chunk_chars(nodes: list[StoryNode]) -> int:
    if not nodes:
        return 0
    return sum(len(node.text) for node in nodes) + len(SCENE_SEPARATOR) * (len(nodes) - 1)


def _union_speakers(nodes: list[StoryNode]) -> list[str]:
    speakers = []
    seen = set()
    for node in nodes:
        for speaker in node.metadata.detected_speakers:
            if speaker not in seen:
                speakers.append(speaker)
                seen.add(speaker)
    return speakers


def _make_chunk(nodes: list[StoryNode]) -> StoryNode:
    first = nodes[0]
    last = nodes[-1]
    meta = first.metadata.model_copy(deep=True)
    meta.scene_index = first.metadata.scene_index
    meta.scene_start = first.metadata.scene_index
    meta.scene_end = last.metadata.scene_index
    meta.source_scene_count = len(nodes)
    meta.detected_speakers = _union_speakers(nodes)
    meta.is_prose = all(node.metadata.is_prose for node in nodes)
    return StoryNode(
        text=SCENE_SEPARATOR.join(node.text for node in nodes),
        metadata=meta,
        summary_level=4,
    )


def _split_part_nodes(
    nodes: list[StoryNode],
    *,
    min_chars: int,
    target_chars: int,
    max_chars: int,
) -> list[list[StoryNode]]:
    chunks: list[list[StoryNode]] = []
    current: list[StoryNode] = []

    for node in nodes:
        if not current:
            current = [node]
            continue

        current_chars = _chunk_chars(current)
        candidate_chars = current_chars + len(SCENE_SEPARATOR) + len(node.text)
        should_close = current_chars >= min_chars and (
            current_chars >= target_chars or candidate_chars > max_chars
        )

        if should_close:
            chunks.append(current)
            current = [node]
        else:
            current.append(node)

    if current:
        chunks.append(current)

    if len(chunks) >= 2 and _chunk_chars(chunks[-1]) < min_chars:
        combined = [*chunks[-2], *chunks[-1]]
        if _chunk_chars(combined) <= max_chars:
            chunks[-2:] = [combined]

    return chunks


def build_retrieval_chunks(
    raw_nodes: list[StoryNode],
    *,
    min_chars: int = MIN_USEFUL_CHARS,
    target_chars: int = TARGET_CHUNK_CHARS,
    max_chars: int = MAX_CHUNK_CHARS,
) -> list[StoryNode]:
    """Coalesce adjacent raw scenes into retrieval chunks within each source part."""
    sorted_nodes = sorted(
        raw_nodes,
        key=lambda node: (
            node.metadata.parent_part_id,
            node.metadata.file_path,
            node.metadata.scene_index,
        ),
    )

    chunks: list[StoryNode] = []
    for _, part_nodes_iter in groupby(sorted_nodes, key=_part_key):
        part_nodes = list(part_nodes_iter)
        for chunk_nodes in _split_part_nodes(
            part_nodes,
            min_chars=min_chars,
            target_chars=target_chars,
            max_chars=max_chars,
        ):
            chunks.append(_make_chunk(chunk_nodes))

    return chunks
