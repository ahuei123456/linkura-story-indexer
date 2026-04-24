from pathlib import Path

from linkura_story_indexer.indexer.parser import StoryParser
from linkura_story_indexer.indexer.processor import StoryProcessor


def test_parser_splits_scenes():
    content = "Scene 1\n---\nScene 2"
    scenes = StoryParser.split_into_scenes(content)
    assert len(scenes) == 2
    assert scenes[0] == "Scene 1"

def test_script_detection():
    script_content = "Kaho: Hello!\nSayaka: Hi."
    prose_content = "Kaho walked down the street. It was a sunny day."
    assert StoryParser.is_script_format(script_content) is True
    assert StoryParser.is_script_format(prose_content) is False

def test_hierarchy_extraction():
    # Main story test
    path_main = Path("story/103/第1話『花咲きたい！』/1.md")
    meta_main = StoryProcessor.extract_hierarchy(path_main)
    assert meta_main.arc_id == "103"
    assert meta_main.story_type == "Main"
    assert meta_main.episode_name == "第1話『花咲きたい！』"
    assert meta_main.part_name == "1"

    # Side story test
    path_side = Path("story/103/～Shades of Stars～/第1話.md")
    meta_side = StoryProcessor.extract_hierarchy(path_side)
    assert meta_side.arc_id == "103"
    assert meta_side.story_type == "Side"
    assert meta_side.episode_name == "～Shades of Stars～"
    assert meta_side.part_name == "第1話"
