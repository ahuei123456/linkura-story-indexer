from pathlib import Path

import pytest

from linkura_story_indexer.indexer.processor import StoryProcessor


def test_configured_arbitrary_root_supports_main_and_side_stories(tmp_path: Path) -> None:
    root = tmp_path / "export"
    main_episode = "\u7b2c1\u8a71"
    side_episode = "\uff5eShades of Stars\uff5e"
    main_path = root / "103" / main_episode / "1.md"
    side_path = root / "103" / side_episode / "\u7b2c1\u8a71.md"
    main_path.parent.mkdir(parents=True)
    side_path.parent.mkdir(parents=True)
    main_path.write_text("Kaho: Hello", encoding="utf-8")
    side_path.write_text("Sayaka: Hello", encoding="utf-8")

    main_meta = StoryProcessor.extract_hierarchy(main_path, story_root=root)
    side_meta = StoryProcessor.extract_hierarchy(side_path, story_root=root)
    side_node = StoryProcessor.process_file(side_path, story_root=root)[0]

    assert (main_meta.arc_id, main_meta.story_type, main_meta.part_name) == ("103", "Main", "1")
    assert (side_meta.arc_id, side_meta.story_type, side_meta.part_name) == (
        "103",
        "Side",
        "\u7b2c1\u8a71",
    )
    assert side_node.metadata.story_type == "Side"
    assert side_node.metadata.episode_name == side_episode


def test_configured_root_rejects_file_outside_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside configured story root"):
        StoryProcessor.extract_hierarchy(
            tmp_path / "other" / "103" / "episode" / "1.md",
            story_root=tmp_path / "export",
        )

    with pytest.raises(ValueError, match="outside configured story root"):
        StoryProcessor.process_file(
            tmp_path / "other" / "103" / "episode" / "missing.md",
            story_root=tmp_path / "export",
        )
