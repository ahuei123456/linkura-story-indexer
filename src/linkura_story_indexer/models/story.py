from pydantic import BaseModel, Field


class DialogueTurn(BaseModel):
    turn_id: str = Field("", description="Stable identifier for this dialogue turn")
    scene_id: str = Field("", description="Stable source scene identifier")
    turn_index: int = Field(0, description="Ordered index within the source scene")
    speaker: str = Field(..., description="Raw speaker label, or UNKNOWN when not explicit")
    speaker_tokens: list[str] = Field(
        default_factory=list,
        description="Queryable speaker tokens parsed from the raw speaker label",
    )
    speaker_kind: str = Field(
        "",
        description="Speaker label kind, such as named, collective, or unknown",
    )
    text: str = Field(..., description="Dialogue text")
    line_start: int = Field(0, description="Zero-based source line where this turn starts")
    line_end: int = Field(0, description="Zero-based source line where this turn ends")


class NarrativeBeat(BaseModel):
    beat_id: str = Field("", description="Stable identifier for this narrative beat")
    scene_id: str = Field("", description="Stable source scene identifier")
    beat_index: int = Field(0, description="Ordered index within the source scene")
    text: str = Field(..., description="Narrative text")
    line_start: int = Field(0, description="Zero-based source line where this beat starts")
    line_end: int = Field(0, description="Zero-based source line where this beat ends")


class StoryMetadata(BaseModel):
    arc_id: str = Field(..., description="Year ID, e.g., '102', '103'")
    story_type: str = Field(..., description="'Main' or 'Side'")
    episode_name: str = Field(..., description="Episode or sub-series name")
    part_name: str = Field(..., description="Part or file name")
    file_path: str = Field(..., description="Path to the original markdown file")
    scene_index: int = Field(0, description="Index of the scene within the file (split by ---)")
    scene_start: int = Field(0, description="First source scene index covered by this node")
    scene_end: int = Field(0, description="Last source scene index covered by this node")
    source_scene_count: int = Field(1, description="Number of source scenes covered by this node")
    is_prose: bool = Field(False, description="True if the content is prose/narrative, False if script")
    canonical_story_order: int = Field(0, description="Global chronological order for this story node")
    story_order: int = Field(0, description="Alias for canonical_story_order used by query filters")
    episode_number: int = Field(0, description="Numeric episode/part number when one is available")
    parent_year_id: str = Field("", description="Stable parent year identifier")
    parent_episode_id: str = Field("", description="Stable parent episode identifier")
    parent_part_id: str = Field("", description="Stable parent part identifier")
    detected_speakers: list[str] = Field(
        default_factory=list,
        description="Speakers detected in this scene",
    )
    speakers: list[str] = Field(
        default_factory=list,
        description="Unique speakers present in this source scene or retrieval chunk",
    )
    source_scene_ids: list[str] = Field(
        default_factory=list,
        description="Stable source scene IDs covered by this node",
    )
    source_turn_ids: list[str] = Field(
        default_factory=list,
        description="Stable dialogue turn IDs covered by this node",
    )
    source_beat_ids: list[str] = Field(
        default_factory=list,
        description="Stable narrative beat IDs covered by this node",
    )
    chunk_id: str = Field("", description="Stable retrieval chunk identifier when indexed")


class StoryNode(BaseModel):
    text: str = Field(..., description="The actual text content of the scene or summary")
    metadata: StoryMetadata
    summary_level: int = Field(4, description="1: Year, 2: Episode, 3: Part, 4: Scene (Raw)")
    dialogue_turns: list[DialogueTurn] = Field(default_factory=list)
    narrative_beats: list[NarrativeBeat] = Field(default_factory=list)
