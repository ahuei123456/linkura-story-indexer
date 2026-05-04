import json
from pathlib import Path

from linkura_story_indexer.story_order import load_story_order
from linkura_story_indexer.summary_export import (
    build_summary_reader_data,
    export_production_summary_reader,
    export_summary_reader,
)


def _summary(*, label: str = "Part") -> str:
    list_label = "Key Events" if label == "Part" else "Episode Arc"
    if label == "Year":
        list_label = "Episode Index"
    return f"""Overview:
{label} overview.

{list_label}:
- First item.

Continuity Facts:
- Durable fact.

Important Terms:
- Kaho Hinoshita
- Hasunosora
"""


def _story_order(path: Path) -> Path:
    config = path / "story_order.yaml"
    config.write_text(
        """
chronological_order:
  - story_type: Main
    arcs: ["103", "104"]
summary_order:
  - story_type: Main
    arcs: ["104", "103"]
part_order_overrides:
  - arc_id: "103"
    story_type: Main
    episode_name: "Episode A"
    parts:
      - "B"
      - "A"
""",
        encoding="utf-8",
    )
    return config


def test_build_summary_reader_data_parses_sections_and_orders_tree(tmp_path: Path) -> None:
    story_order = load_story_order(_story_order(tmp_path))
    cache = {
        "103|Main|Episode A|A": {
            "schema_version": "1",
            "fingerprint": "part-a",
            "summary": _summary(),
            "inputs": {"level": "part", "chat_model": "chat", "embedding_model": "embed"},
        },
        "103|Main|Episode A|B": {
            "schema_version": "1",
            "fingerprint": "part-b",
            "summary": _summary(),
            "inputs": {"level": "part"},
        },
        "EPISODE|103|Main|Episode A": {
            "schema_version": "1",
            "fingerprint": "episode",
            "summary": _summary(label="Episode"),
            "inputs": {"level": "episode"},
        },
        "YEAR|103": {
            "schema_version": "1",
            "fingerprint": "year-103",
            "summary": _summary(label="Year"),
            "inputs": {"level": "year"},
        },
        "YEAR|104": {
            "schema_version": "1",
            "fingerprint": "year-104",
            "summary": _summary(label="Year"),
            "inputs": {"level": "year"},
        },
    }

    data = build_summary_reader_data(cache, story_order=story_order)

    assert data["schemaVersion"] == "2"
    assert data["counts"] == {"years": 2, "episodes": 1, "parts": 2}
    assert [data["nodes"][node_id]["label"] for node_id in data["roots"]] == ["Arc 104", "Arc 103"]
    episode_id = data["nodes"]["year:103"]["children"][0]
    episode = data["nodes"][episode_id]
    assert [data["nodes"][part_id]["label"] for part_id in episode["children"]] == [
        "Part B",
        "Part A",
    ]

    part_summary = data["summaries"]["103|Main|Episode A|A"]
    assert part_summary["sectionOrder"][0] == "Overview"
    assert part_summary["sections"]["Overview"] == "Part overview."
    assert part_summary["importantTerms"] == ["Kaho Hinoshita", "Hasunosora"]
    assert part_summary["meta"]["models"]["chat"] == "chat"
    assert any(
        record["summaryId"] == "103|Main|Episode A|A" and "kaho hinoshita" in record["text"]
        for record in data["search"]
    )


def test_export_summary_reader_writes_static_site(tmp_path: Path) -> None:
    story_order = load_story_order(_story_order(tmp_path))
    cache_file = tmp_path / "summaries_cache.json"
    cache_file.write_text(
        json.dumps(
            {
                "YEAR|103": {
                    "schema_version": "1",
                    "fingerprint": "year",
                    "summary": _summary(label="Year"),
                    "inputs": {"level": "year"},
                }
            }
        ),
        encoding="utf-8",
    )
    template = tmp_path / "index.html"
    template.write_text("<!doctype html><title>Reader</title>", encoding="utf-8")

    output_dir = export_summary_reader(
        cache_file=cache_file,
        output_dir=tmp_path / "site",
        story_order=story_order,
        template_file=template,
    )

    assert (output_dir / "index.html").read_text(encoding="utf-8").startswith("<!doctype html>")
    data = json.loads((output_dir / "data" / "summaries.json").read_text(encoding="utf-8"))
    assert data["counts"]["years"] == 1
    assert data["roots"] == ["year:103"]
    assert "YEAR|103" in data["summaries"]


def test_export_production_summary_reader_writes_zip_page_site(tmp_path: Path) -> None:
    story_order = load_story_order(_story_order(tmp_path))
    cache_file = tmp_path / "summaries_cache.json"
    cache_file.write_text(
        json.dumps(
            {
                "YEAR|103": {
                    "schema_version": "1",
                    "fingerprint": "year",
                    "summary": _summary(label="Year"),
                    "inputs": {"level": "year"},
                }
            }
        ),
        encoding="utf-8",
    )
    source_dir = tmp_path / "production"
    source_dir.mkdir()
    (source_dir / "index.html").write_text("<!doctype html><title>Production</title>", encoding="utf-8")
    (source_dir / "tweaks-panel.jsx").write_text("// tweaks", encoding="utf-8")

    output_dir = export_production_summary_reader(
        cache_file=cache_file,
        output_dir=tmp_path / "site",
        story_order=story_order,
        source_dir=source_dir,
    )

    assert (output_dir / "index.html").exists()
    assert (output_dir / "tweaks-panel.jsx").exists()
    assert not (output_dir / "data" / "index.json").exists()
    assert not (output_dir / "data" / "summaries_cache.json").exists()
    data = json.loads((output_dir / "data" / "summaries.json").read_text(encoding="utf-8"))
    assert data["schemaVersion"] == "2"
    assert "YEAR|103" in data["summaries"]
