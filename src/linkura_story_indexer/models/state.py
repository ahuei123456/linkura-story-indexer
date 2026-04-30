from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

StatePredicate = Literal[
    "role",
    "alias",
    "honorific_used_for",
    "location",
    "group_membership",
    "relationship",
    "goal",
    "status",
    "attribute",
    "possession",
    "commitment",
    "emotional_stance_toward",
    "life_stage",
]

STATE_PREDICATES: tuple[str, ...] = (
    "role",
    "alias",
    "honorific_used_for",
    "location",
    "group_membership",
    "relationship",
    "goal",
    "status",
    "attribute",
    "possession",
    "commitment",
    "emotional_stance_toward",
    "life_stage",
)

TARGET_REQUIRED_PREDICATES = {
    "honorific_used_for",
    "relationship",
    "emotional_stance_toward",
}

TARGET_UNUSED_PREDICATES = {
    "alias",
    "attribute",
}

SINGLE_CURRENT_PREDICATES = {
    "role",
    "honorific_used_for",
    "location",
    "status",
    "emotional_stance_toward",
    "life_stage",
}

STATE_LEDGER_SCHEMA_VERSION = 3


class ExtractedStateFact(BaseModel):
    """Atomic fact as emitted by the scene-level extraction model."""

    subject: str = Field(..., description="Entity the fact is about")
    predicate: StatePredicate = Field(..., description="Controlled state fact predicate")
    target: str | None = Field(
        default=None,
        description="Directed target for predicates such as honorifics or relationships",
    )
    object: str = Field(..., description="Value of the fact")
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    extracted_quote: str = Field(
        ...,
        description="Exact substring copied from the source scene that supports this fact",
    )

    @field_validator("subject", "object", "extracted_quote")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("fact fields must not be blank")
        return stripped

    @field_validator("target")
    @classmethod
    def _strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @model_validator(mode="after")
    def _validate_target_for_predicate(self) -> "ExtractedStateFact":
        if self.predicate in TARGET_REQUIRED_PREDICATES and self.target is None:
            raise ValueError(f"{self.predicate} facts require target")
        if self.predicate in TARGET_UNUSED_PREDICATES and self.target is not None:
            raise ValueError(f"{self.predicate} facts must not set target")
        return self


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

    schema_version: int = STATE_LEDGER_SCHEMA_VERSION
    facts: list[StateFact] = Field(default_factory=list)
