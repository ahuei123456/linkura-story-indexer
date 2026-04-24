from pydantic import BaseModel, Field


class StoryMetadata(BaseModel):
    arc_id: str = Field(..., description="Year ID, e.g., '102', '103'")
    story_type: str = Field(..., description="'Main' or 'Side'")
    episode_name: str = Field(..., description="Episode or sub-series name")
    part_name: str = Field(..., description="Part or file name")
    file_path: str = Field(..., description="Path to the original markdown file")
    scene_index: int = Field(0, description="Index of the scene within the file (split by ---)")
    is_prose: bool = Field(False, description="True if the content is prose/narrative, False if script")
    canonical_story_order: int = Field(0, description="Global chronological order for this story node")
    parent_year_id: str = Field("", description="Stable parent year identifier")
    parent_episode_id: str = Field("", description="Stable parent episode identifier")
    parent_part_id: str = Field("", description="Stable parent part identifier")
    detected_speakers: list[str] = Field(
        default_factory=list,
        description="Speakers detected in this scene",
    )


class StoryNode(BaseModel):
    text: str = Field(..., description="The actual text content of the scene or summary")
    metadata: StoryMetadata
    summary_level: int = Field(4, description="1: Year, 2: Episode, 3: Part, 4: Scene (Raw)")
