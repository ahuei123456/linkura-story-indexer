from pydantic import BaseModel, Field


class HonorificFact(BaseModel):
    target_character: str = Field(..., description="The character being spoken to or about")
    honorific: str = Field(..., description="The honorific or nickname used (e.g., '-chan')")

class CharacterFact(BaseModel):
    name: str
    nicknames: list[str] = Field(default_factory=list)
    role: str = Field(..., description="e.g., '1st Year Student', 'Club President'")
    honorifics_used: list[HonorificFact] = Field(
        default_factory=list, 
        description="List of honorifics or nicknames this character uses for others"
    )
    is_active: bool = True

class WorldState(BaseModel):
    arc_id: str
    characters: list[CharacterFact]
    locations: list[str] = Field(default_factory=list, description="Known areas in the story")
    important_groups: list[str] = Field(default_factory=list, description="e.g., 'Cerise Bouquet'")
