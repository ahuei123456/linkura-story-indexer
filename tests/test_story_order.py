from pathlib import Path

import pytest

from linkura_story_indexer import cli
from linkura_story_indexer.indexer.processor import StoryProcessor
from linkura_story_indexer.indexer.summarizer import HierarchicalSummarizer
from linkura_story_indexer.story_order import StoryOrderConfigError, load_story_order


def _write_story_file(root: Path, relative_path: str, content: str = "text") -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _write_config(root: Path, content: str) -> Path:
    path = root / "story_order.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_valid_fixture_supports_distinct_chronological_and_summary_order(
    tmp_path: Path,
) -> None:
    story_root = tmp_path / "story"
    _write_story_file(story_root, "102/～Shades of Stars～/第1話.md")
    _write_story_file(story_root, "103/～Shades of Stars～/第1話.md")
    _write_story_file(story_root, "103/第1話『花咲きたい！』/1.md")
    _write_story_file(story_root, "104/第1話『未来への歌』/1.md")
    _write_story_file(story_root, "105/第1話『Brand New Stories!!』/1.md")
    config_path = _write_config(
        tmp_path,
        """
chronological_order:
  - story_type: Side
    arcs: ["102", "103"]
  - story_type: Main
    arcs: ["103"]
  - story_type: Main
    arcs: ["104"]
  - story_type: Main
    arcs: ["105"]
summary_order:
  - story_type: Main
    arcs: ["103"]
  - story_type: Main
    arcs: ["104"]
  - story_type: Side
    arcs: ["102", "103"]
  - story_type: Main
    arcs: ["105"]
""",
    )

    story_order = load_story_order(config_path, story_root=story_root)

    assert story_order.chronological_episode_key("103", "Side", "～Shades of Stars～") < (
        story_order.chronological_episode_key("103", "Main", "第1話『花咲きたい！』")
    )
    assert story_order.summary_episode_key("104", "Main", "第1話『未来への歌』") < (
        story_order.summary_episode_key("103", "Side", "～Shades of Stars～")
    )


def test_adding_new_arc_updates_story_order_by_config_only(tmp_path: Path) -> None:
    story_root = tmp_path / "story"
    main_103_path = _write_story_file(story_root, "103/第1話『花咲きたい！』/1.md")
    main_106_path = _write_story_file(story_root, "106/第1話『新しい物語』/1.md")
    config_path = _write_config(
        tmp_path,
        """
chronological_order:
  - story_type: Main
    arcs: ["103", "106"]
summary_order:
  - story_type: Main
    arcs: ["103", "106"]
""",
    )
    story_order = load_story_order(config_path, story_root=story_root)
    raw_nodes = [
        *StoryProcessor.process_file(main_106_path),
        *StoryProcessor.process_file(main_103_path),
    ]

    cli._assign_canonical_story_order(raw_nodes, story_order=story_order)

    order_by_arc = {node.metadata.arc_id: node.metadata.story_order for node in raw_nodes}
    assert order_by_arc["103"] < order_by_arc["106"]


def test_malformed_config_missing_required_field_raises_clear_error(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
chronological_order:
  - story_type: Main
summary_order:
  - story_type: Main
    arcs: ["103"]
""",
    )

    with pytest.raises(StoryOrderConfigError, match="arcs"):
        load_story_order(config_path)


def test_unknown_story_type_raises_clear_error(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
chronological_order:
  - story_type: Bonus
    arcs: ["103"]
summary_order:
  - story_type: Main
    arcs: ["103"]
""",
    )

    with pytest.raises(StoryOrderConfigError, match="unknown story_type"):
        load_story_order(config_path)


def test_story_root_with_uncovered_story_pair_raises_clear_error(tmp_path: Path) -> None:
    story_root = tmp_path / "story"
    _write_story_file(story_root, "103/第1話『花咲きたい！』/1.md")
    config_path = _write_config(
        tmp_path,
        """
chronological_order:
  - story_type: Side
    arcs: ["103"]
summary_order:
  - story_type: Side
    arcs: ["103"]
""",
    )

    with pytest.raises(StoryOrderConfigError, match="chronological_order does not cover"):
        load_story_order(config_path, story_root=story_root)


def test_story_root_with_unknown_configured_arc_raises_clear_error(tmp_path: Path) -> None:
    story_root = tmp_path / "story"
    _write_story_file(story_root, "103/第1話『花咲きたい！』/1.md")
    config_path = _write_config(
        tmp_path,
        """
chronological_order:
  - story_type: Main
    arcs: ["103", "999"]
summary_order:
  - story_type: Main
    arcs: ["103"]
""",
    )

    with pytest.raises(StoryOrderConfigError, match="references unknown"):
        load_story_order(config_path, story_root=story_root)


def test_part_order_override_places_106_marker_last(tmp_path: Path) -> None:
    story_root = tmp_path / "story"
    episode_name = "第12話『ずっと花咲く僕らの桜』"
    _write_story_file(story_root, f"105/{episode_name}/1.md")
    _write_story_file(story_root, f"105/{episode_name}/PERIOD.md")
    marker_path = _write_story_file(story_root, f"105/{episode_name}/___106.md")
    config_path = _write_config(
        tmp_path,
        f"""
chronological_order:
  - story_type: Main
    arcs: ["105"]
summary_order:
  - story_type: Main
    arcs: ["105"]
part_order_overrides:
  - arc_id: "105"
    story_type: Main
    episode_name: "{episode_name}"
    parts:
      - "1"
      - "PERIOD"
      - "___106"
""",
    )
    story_order = load_story_order(config_path, story_root=story_root)
    raw_nodes = [
        *StoryProcessor.process_file(marker_path),
        *StoryProcessor.process_file(story_root / "105" / episode_name / "PERIOD.md"),
        *StoryProcessor.process_file(story_root / "105" / episode_name / "1.md"),
    ]

    cli._assign_canonical_story_order(raw_nodes, story_order=story_order)

    ordered_parts = [
        node.metadata.part_name
        for node in sorted(raw_nodes, key=lambda node: node.metadata.story_order)
    ]
    assert ordered_parts == ["1", "PERIOD", "___106"]


def test_part_order_override_must_list_every_part(tmp_path: Path) -> None:
    story_root = tmp_path / "story"
    episode_name = "第12話『ずっと花咲く僕らの桜』"
    _write_story_file(story_root, f"105/{episode_name}/1.md")
    _write_story_file(story_root, f"105/{episode_name}/___106.md")
    config_path = _write_config(
        tmp_path,
        f"""
chronological_order:
  - story_type: Main
    arcs: ["105"]
summary_order:
  - story_type: Main
    arcs: ["105"]
part_order_overrides:
  - arc_id: "105"
    story_type: Main
    episode_name: "{episode_name}"
    parts:
      - "1"
""",
    )

    with pytest.raises(StoryOrderConfigError, match="missing parts"):
        load_story_order(config_path, story_root=story_root)


def test_year_summary_sort_ignores_part_override_for_episode_summary_nodes(tmp_path: Path) -> None:
    story_root = tmp_path / "story"
    episode_name = "第3話『雨と、風と、太陽と』"
    part_one_path = _write_story_file(story_root, f"103/{episode_name}/1.md")
    _write_story_file(story_root, f"103/{episode_name}/幕間.md")
    config_path = _write_config(
        tmp_path,
        f"""
chronological_order:
  - story_type: Main
    arcs: ["103"]
summary_order:
  - story_type: Main
    arcs: ["103"]
part_order_overrides:
  - arc_id: "103"
    story_type: Main
    episode_name: "{episode_name}"
    parts:
      - "1"
      - "幕間"
""",
    )
    story_order = load_story_order(config_path, story_root=story_root)
    episode_summary = StoryProcessor.process_file(part_one_path)[0]
    episode_summary.summary_level = 2
    episode_summary.metadata.part_name = "ALL_PARTS"

    summarizer = HierarchicalSummarizer(story_order=story_order)

    year_summaries = summarizer.summarize_years([episode_summary], cache_file=str(tmp_path / "cache.json"))

    assert len(year_summaries) == 1
    assert year_summaries[0].metadata.episode_name == "ALL_EPISODES"
