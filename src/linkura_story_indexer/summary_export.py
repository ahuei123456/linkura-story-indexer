from __future__ import annotations

import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from linkura_story_indexer.indexer.summarizer import (
    SUMMARY_SECTIONS_BY_LEVEL,
    extract_summary_sections,
)
from linkura_story_indexer.story_order import StoryOrder, StoryOrderConfigError, natural_sort_key

SUMMARY_READER_DATA_SCHEMA_VERSION = "2"
DEFAULT_READER_TEMPLATE = Path("web/summary-reader/index.html")
DEFAULT_PRODUCTION_READER_SOURCE = Path("web/summary-reader-production")


def export_summary_reader(
    *,
    cache_file: str | Path = "summaries_cache.json",
    output_dir: str | Path = "site/summary-reader",
    story_order: StoryOrder,
    template_file: str | Path = DEFAULT_READER_TEMPLATE,
) -> Path:
    """Export cached structured summaries as a static summary reader site."""
    cache_path = Path(cache_file)
    if not cache_path.exists():
        raise FileNotFoundError(f"Summary cache not found: {cache_path}")

    template_path = Path(template_file)
    if not template_path.exists():
        raise FileNotFoundError(f"Summary reader template not found: {template_path}")

    with open(cache_path, encoding="utf-8") as file:
        cache = json.load(file)
    if not isinstance(cache, dict):
        raise ValueError("Summary cache must be a JSON object.")

    reader_data = build_summary_reader_data(cache, story_order=story_order)

    destination = Path(output_dir)
    data_dir = destination / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(template_path, destination / "index.html")
    with open(data_dir / "summaries.json", "w", encoding="utf-8") as file:
        json.dump(reader_data, file, ensure_ascii=False, indent=2)
        file.write("\n")

    return destination


def export_production_summary_reader(
    *,
    cache_file: str | Path = "summaries_cache.json",
    output_dir: str | Path = "site/summary-reader-production",
    story_order: StoryOrder,
    source_dir: str | Path = DEFAULT_PRODUCTION_READER_SOURCE,
) -> Path:
    """Export the zip production summary reader with normalized reader data."""
    cache_path = Path(cache_file)
    if not cache_path.exists():
        raise FileNotFoundError(f"Summary cache not found: {cache_path}")

    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Production reader source not found: {source_path}")

    with open(cache_path, encoding="utf-8") as file:
        cache = json.load(file)
    if not isinstance(cache, dict):
        raise ValueError("Summary cache must be a JSON object.")

    destination = Path(output_dir)
    shutil.copytree(
        source_path,
        destination,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("data"),
    )

    data_dir = destination / "data"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / "summaries.json", "w", encoding="utf-8") as file:
        json.dump(
            build_summary_reader_data(cache, story_order=story_order),
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")

    return destination


def build_summary_reader_data(
    cache: dict[str, Any],
    *,
    story_order: StoryOrder,
) -> dict[str, Any]:
    entries = [
        _summary_entry(cache_key, raw_entry)
        for cache_key, raw_entry in cache.items()
        if isinstance(cache_key, str) and isinstance(raw_entry, dict)
    ]
    entries.sort(key=lambda entry: _entry_sort_key(entry, story_order))

    nodes, roots, summary_node_ids = _summary_nodes(entries, story_order)
    summaries = {
        entry["cacheKey"]: _frontend_summary(entry, summary_node_ids.get(entry["cacheKey"], ""))
        for entry in entries
    }
    return {
        "schemaVersion": SUMMARY_READER_DATA_SCHEMA_VERSION,
        "generatedAt": datetime.now(UTC).isoformat(),
        "counts": _counts(entries),
        "roots": roots,
        "nodes": nodes,
        "summaries": summaries,
        "search": [_search_record(summary) for summary in summaries.values()],
    }


def _summary_entry(cache_key: str, raw_entry: dict[str, Any]) -> dict[str, Any]:
    key_metadata = _parse_cache_key(cache_key)
    summary = raw_entry.get("summary")
    if not isinstance(summary, str):
        summary = ""

    sections_by_label = extract_summary_sections(summary)
    level_name = str(key_metadata["tier"])
    required_sections = SUMMARY_SECTIONS_BY_LEVEL.get(level_name, ())
    sections = [
        {"label": label, "content": sections_by_label.get(label, "")}
        for label in required_sections
        if label in sections_by_label
    ]
    if not sections and summary:
        sections = [{"label": "Summary", "content": summary.strip()}]

    inputs = raw_entry.get("inputs")
    if not isinstance(inputs, dict):
        inputs = {}

    important_terms = _important_terms(sections_by_label.get("Important Terms", ""))
    return {
        "id": cache_key,
        "cacheKey": cache_key,
        "tier": level_name,
        "summaryLevel": key_metadata["summaryLevel"],
        "arcId": key_metadata["arcId"],
        "storyType": key_metadata["storyType"],
        "episodeName": key_metadata["episodeName"],
        "partName": key_metadata["partName"],
        "title": _entry_title(key_metadata),
        "sections": sections,
        "importantTerms": important_terms,
        "fingerprint": raw_entry.get("fingerprint", ""),
        "summaryCacheSchemaVersion": raw_entry.get("schema_version", ""),
        "inputLevel": inputs.get("level", ""),
        "models": {
            "chat": inputs.get("chat_model", ""),
            "generationProvider": inputs.get("generation_provider", ""),
            "generation": inputs.get("generation_model", ""),
            "embedding": inputs.get("embedding_model", ""),
        },
    }


def _frontend_summary(entry: dict[str, Any], node_id: str) -> dict[str, Any]:
    section_order = [section["label"] for section in entry["sections"]]
    sections = {section["label"]: section["content"] for section in entry["sections"]}
    return {
        "id": entry["cacheKey"],
        "nodeId": node_id,
        "tier": entry["tier"],
        "title": entry["title"],
        "meta": {
            "arcId": entry["arcId"],
            "storyType": entry["storyType"],
            "episodeName": entry["episodeName"],
            "partName": entry["partName"],
            "summaryLevel": entry["summaryLevel"],
            "cacheKey": entry["cacheKey"],
            "fingerprint": entry["fingerprint"],
            "summaryCacheSchemaVersion": entry["summaryCacheSchemaVersion"],
            "inputLevel": entry["inputLevel"],
            "models": entry["models"],
        },
        "sectionOrder": section_order,
        "sections": sections,
        "importantTerms": entry["importantTerms"],
    }


def _search_record(summary: dict[str, Any]) -> dict[str, Any]:
    sections = summary["sections"]
    searchable_text = "\n".join(
        [
            summary["title"],
            summary["tier"],
            summary["meta"]["arcId"],
            summary["meta"]["storyType"],
            summary["meta"]["episodeName"],
            summary["meta"]["partName"],
            summary["meta"]["cacheKey"],
            *summary["importantTerms"],
            *[f"{label}\n{sections[label]}" for label in summary["sectionOrder"]],
        ]
    )
    return {
        "summaryId": summary["id"],
        "nodeId": summary["nodeId"],
        "tier": summary["tier"],
        "title": summary["title"],
        "text": searchable_text.lower(),
    }


def _episode_title(episode_name: str) -> str:
    match = re.search(r"『(.+)』", episode_name)
    if match:
        return match.group(1)
    return episode_name


def _parse_cache_key(cache_key: str) -> dict[str, str | int]:
    if cache_key.startswith("YEAR|"):
        _, arc_id = cache_key.split("|", 1)
        return {
            "tier": "Year",
            "summaryLevel": 1,
            "arcId": arc_id,
            "storyType": "All",
            "episodeName": "ALL_EPISODES",
            "partName": "ALL_PARTS",
        }

    if cache_key.startswith("EPISODE|"):
        parts = cache_key.split("|", 3)
        if len(parts) != 4:
            raise ValueError(f"Malformed episode summary cache key: {cache_key}")
        _, arc_id, story_type, episode_name = parts
        return {
            "tier": "Episode",
            "summaryLevel": 2,
            "arcId": arc_id,
            "storyType": story_type,
            "episodeName": episode_name,
            "partName": "ALL_PARTS",
        }

    parts = cache_key.split("|", 3)
    if len(parts) != 4:
        raise ValueError(f"Malformed part summary cache key: {cache_key}")
    arc_id, story_type, episode_name, part_name = parts
    return {
        "tier": "Part",
        "summaryLevel": 3,
        "arcId": arc_id,
        "storyType": story_type,
        "episodeName": episode_name,
        "partName": part_name,
    }


def _entry_title(metadata: dict[str, str | int]) -> str:
    tier = metadata["tier"]
    if tier == "Year":
        return f"Arc {metadata['arcId']}"
    if tier == "Episode":
        return str(metadata["episodeName"])
    return f"{metadata['episodeName']} / Part {metadata['partName']}"


def _important_terms(section: str) -> list[str]:
    terms = []
    for line in section.splitlines():
        term = re.sub(r"^[-*]\s*", "", line.strip()).strip()
        if term and term.lower() != "none":
            terms.append(term)
    return terms


def _counts(entries: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"years": 0, "episodes": 0, "parts": 0}
    for entry in entries:
        if entry["tier"] == "Year":
            counts["years"] += 1
        elif entry["tier"] == "Episode":
            counts["episodes"] += 1
        elif entry["tier"] == "Part":
            counts["parts"] += 1
    return counts


def _summary_nodes(
    entries: list[dict[str, Any]],
    story_order: StoryOrder,
) -> tuple[dict[str, dict[str, Any]], list[str], dict[str, str]]:
    by_year: dict[str, dict[str, Any]] = {}
    episodes_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    summary_node_ids: dict[str, str] = {}

    for entry in entries:
        arc_id = entry["arcId"]
        node_id = f"year:{arc_id}"
        year = by_year.setdefault(
            arc_id,
            {
                "id": node_id,
                "kind": "year",
                "label": f"Arc {arc_id}",
                "title": f"Arc {arc_id}",
                "summaryId": "",
                "children": [],
            },
        )
        if entry["tier"] == "Year":
            year["summaryId"] = entry["id"]
            summary_node_ids[entry["id"]] = node_id

    for entry in entries:
        if entry["tier"] != "Episode":
            continue
        episode_key = (entry["arcId"], entry["storyType"], entry["episodeName"])
        node_id = f"episode:{entry['arcId']}:{entry['storyType']}:{entry['episodeName']}"
        episode = {
            "id": node_id,
            "kind": "episode",
            "label": entry["episodeName"],
            "title": _episode_title(entry["episodeName"]),
            "summaryId": entry["id"],
            "children": [],
            "arcId": entry["arcId"],
            "storyType": entry["storyType"],
            "episodeName": entry["episodeName"],
        }
        episodes_by_key[episode_key] = episode
        by_year[entry["arcId"]]["children"].append(episode)
        summary_node_ids[entry["id"]] = node_id

    for entry in entries:
        if entry["tier"] != "Part":
            continue
        episode_key = (entry["arcId"], entry["storyType"], entry["episodeName"])
        episode = episodes_by_key.get(episode_key)
        if episode is None:
            node_id = f"episode:{entry['arcId']}:{entry['storyType']}:{entry['episodeName']}"
            episode = {
                "id": node_id,
                "kind": "episode",
                "label": entry["episodeName"],
                "title": _episode_title(entry["episodeName"]),
                "summaryId": "",
                "children": [],
                "arcId": entry["arcId"],
                "storyType": entry["storyType"],
                "episodeName": entry["episodeName"],
            }
            episodes_by_key[episode_key] = episode
            by_year[entry["arcId"]]["children"].append(episode)
        children = cast(list[dict[str, Any]], episode["children"])
        node_id = f"part:{entry['id']}"
        children.append(
            {
                "id": node_id,
                "kind": "part",
                "label": f"Part {entry['partName']}",
                "title": f"Part {entry['partName']}",
                "summaryId": entry["id"],
                "partName": entry["partName"],
            }
        )
        summary_node_ids[entry["id"]] = node_id

    for year in by_year.values():
        year["children"].sort(key=lambda episode: _episode_sort_key(episode, story_order))
        for episode in year["children"]:
            episode["children"].sort(key=lambda part: _part_sort_key(episode, part, story_order))

    sorted_years = sorted(by_year.values(), key=lambda year: _year_sort_key(year, story_order))
    nodes: dict[str, dict[str, Any]] = {}
    for year in sorted_years:
        _collect_nodes(year, nodes)
    return nodes, [year["id"] for year in sorted_years], summary_node_ids


def _collect_nodes(node: dict[str, Any], nodes: dict[str, dict[str, Any]]) -> None:
    children = node.get("children", [])
    nodes[node["id"]] = {**node, "children": [child["id"] for child in children]}
    for child in children:
        _collect_nodes(child, nodes)


def _entry_sort_key(entry: dict[str, Any], story_order: StoryOrder) -> tuple[Any, ...]:
    tier_position = {"Year": 0, "Episode": 1, "Part": 2}.get(entry["tier"], 9)
    if entry["tier"] == "Year":
        return (_year_sort_key({"label": "", "id": "", "summaryId": "", "children": [], **entry}, story_order), tier_position)
    if entry["tier"] == "Episode":
        return (
            _safe_episode_key(
                story_order,
                entry["arcId"],
                entry["storyType"],
                entry["episodeName"],
            ),
            tier_position,
        )
    return (
        _safe_episode_key(
            story_order,
            entry["arcId"],
            entry["storyType"],
            entry["episodeName"],
        ),
        tier_position,
        _safe_part_key(
            story_order,
            entry["arcId"],
            entry["storyType"],
            entry["episodeName"],
            entry["partName"],
        ),
    )


def _year_sort_key(year: dict[str, Any], story_order: StoryOrder) -> tuple[Any, ...]:
    arc_id = year["arcId"] if "arcId" in year else str(year["id"]).removeprefix("year:")
    positions = [
        position
        for (story_type, configured_arc_id), position in story_order.summary_positions.items()
        if configured_arc_id == arc_id
    ]
    if positions:
        return (min(positions), natural_sort_key(arc_id))
    return ((9999, 9999), natural_sort_key(arc_id))


def _episode_sort_key(episode: dict[str, Any], story_order: StoryOrder) -> tuple[Any, ...]:
    return _safe_episode_key(
        story_order,
        episode["arcId"],
        episode["storyType"],
        episode["episodeName"],
    )


def _part_sort_key(
    episode: dict[str, Any],
    part: dict[str, Any],
    story_order: StoryOrder,
) -> tuple[Any, ...]:
    return _safe_part_key(
        story_order,
        episode["arcId"],
        episode["storyType"],
        episode["episodeName"],
        part["partName"],
    )


def _safe_episode_key(
    story_order: StoryOrder,
    arc_id: str,
    story_type: str,
    episode_name: str,
) -> tuple[Any, ...]:
    try:
        group_index, arc_index, episode_key = story_order.summary_episode_key(
            arc_id,
            story_type,
            episode_name,
        )
        return ((group_index, arc_index), episode_key)
    except StoryOrderConfigError:
        return ((9999, 9999), natural_sort_key(arc_id), story_type, natural_sort_key(episode_name))


def _safe_part_key(
    story_order: StoryOrder,
    arc_id: str,
    story_type: str,
    episode_name: str,
    part_name: str,
) -> tuple[Any, ...]:
    try:
        return story_order.part_key(arc_id, story_type, episode_name, part_name)
    except StoryOrderConfigError:
        return (1, natural_sort_key(part_name))
