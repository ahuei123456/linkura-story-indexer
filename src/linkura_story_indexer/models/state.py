from pydantic import BaseModel, Field, field_validator


class ExtractedStateFact(BaseModel):
    """Atomic fact as emitted by the scene-level extraction model."""

    subject: str = Field(..., description="Entity the fact is about")
    predicate: str = Field(..., description="Relationship or state name")
    object: str = Field(..., description="Value or target of the fact")
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    extracted_quote: str = Field(
        ...,
        description="Exact substring copied from the source scene that supports this fact",
    )

    @field_validator("subject", "predicate", "object", "extracted_quote")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("fact fields must not be blank")
        return stripped


class SceneStateExtraction(BaseModel):
    """Facts extracted from one raw source scene."""

    facts: list[ExtractedStateFact] = Field(default_factory=list)


class StateFact(ExtractedStateFact):
    """Source-backed temporal ledger fact."""

    arc: str
    episode: str
    part: str
    scene: int
    valid_from: int
    valid_to: int | None = None
    file_path: str
    scene_index: int


class StateLedger(BaseModel):
    """World-state ledger stored as source-backed fact records."""

    schema_version: int = 2
    facts: list[StateFact] = Field(default_factory=list)
