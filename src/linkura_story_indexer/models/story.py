from pydantic import BaseModel, Field


class StoryMetadata(BaseModel):
    arc_id: str = Field(..., description="Year ID, e.g., '102', '103'")
    story_type: str = Field(..., description="'Main' or 'Side'")
    episode_name: str = Field(..., description="Episode or sub-series name")
    part_name: str = Field(..., description="Part or file name")
    file_path: str = Field(..., description="Path to the original markdown file")
    scene_index: int = Field(0, description="Index of the scene within the file (split by ---)")
    is_prose: bool = Field(False, description="True if the content is prose/narrative, False if script")

class StoryNode(BaseModel):
    text: str = Field(..., description="The actual text content of the scene or summary")
    metadata: StoryMetadata
    summary_level: int = Field(4, description="1: Year, 2: Episode, 3: Part, 4: Scene (Raw)")
