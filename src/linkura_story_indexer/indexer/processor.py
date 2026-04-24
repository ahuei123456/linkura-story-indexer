from pathlib import Path

from ..models.story import StoryMetadata, StoryNode
from .parser import StoryParser


def _parent_ids(arc_id: str, story_type: str, episode_name: str, part_name: str) -> tuple[str, str, str]:
    year_id = arc_id
    episode_id = f"{arc_id}|{story_type}|{episode_name}"
    part_id = f"{episode_id}|{part_name}"
    return year_id, episode_id, part_id


class StoryProcessor:
    """Processes story directories into StoryNodes."""

    @staticmethod
    def extract_hierarchy(file_path: Path) -> StoryMetadata:
        """
        Extracts year, story type, episode, and part info from the directory structure.
        Main Story Path: story/103/第1話『...』/1.md
        Side Story Path: story/103/～Shades of Stars～/第1話.md
        """
        parts = file_path.parts
        try:
            story_idx = parts.index('story')
            arc_id = parts[story_idx + 1]
            # Pattern: story/103/FolderName/FileName.md
            folder_name = parts[story_idx + 2]
            
            if "～" in folder_name:
                # Side story: folder is an Episode (e.g. ～Shades of Stars～)
                # and file name is the Part (e.g. 第1話.md)
                story_type = "Side"
                ep_name = folder_name
                part_name = file_path.stem # '第1話'
            elif "第" in folder_name and "話" in folder_name:
                # Main story: folder is an Episode (e.g. 第1話『花咲きたい！』)
                # and file name is a Part (e.g. 1.md)
                story_type = "Main"
                ep_name = folder_name
                part_name = file_path.stem
            else:
                # Catch-all
                story_type = "Other"
                ep_name = folder_name
                part_name = file_path.stem

            parent_year_id, parent_episode_id, parent_part_id = _parent_ids(
                arc_id,
                story_type,
                ep_name,
                part_name,
            )
            return StoryMetadata(
                arc_id=arc_id,
                story_type=story_type,
                episode_name=ep_name,
                part_name=part_name,
                file_path=str(file_path),
                parent_year_id=parent_year_id,
                parent_episode_id=parent_episode_id,
                parent_part_id=parent_part_id,
            )
        except (ValueError, IndexError):
            part_name = file_path.parent.name
            parent_year_id, parent_episode_id, parent_part_id = _parent_ids(
                "unknown",
                "unknown",
                "unknown",
                part_name,
            )
            return StoryMetadata(
                arc_id="unknown",
                story_type="unknown",
                episode_name="unknown",
                part_name=part_name,
                file_path=str(file_path),
                parent_year_id=parent_year_id,
                parent_episode_id=parent_episode_id,
                parent_part_id=parent_part_id,
            )

    @classmethod
    def process_file(cls, file_path: Path) -> list[StoryNode]:
        """Reads a file, splits it into scenes, and returns StoryNodes."""
        with open(file_path, encoding='utf-8') as f:
            content = f.read()
        
        metadata_base = cls.extract_hierarchy(file_path)
        is_script = StoryParser.is_script_format(content)
        scenes = StoryParser.split_into_scenes(content)
        
        nodes = []
        for i, scene_text in enumerate(scenes):
            meta = metadata_base.model_copy(deep=True)
            meta.scene_index = i
            meta.is_prose = not is_script
            meta.detected_speakers = StoryParser.detect_speakers(scene_text)
            nodes.append(StoryNode(text=scene_text, metadata=meta))
            
        return nodes
