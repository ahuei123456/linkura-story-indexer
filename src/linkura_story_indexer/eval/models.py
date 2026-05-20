from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

RoutingMode = Literal["off", "heuristic", "llm_router"]
EvalMode = RoutingMode
ROUTING_MODES = ("off", "heuristic", "llm_router")
CandidateKind = Literal["raw_span", "summary", "summary_section", "reranker"]
StageName = Literal[
    "dense_raw",
    "lexical_raw",
    "rrf_fusion",
    "raw_seed_filter",
    "neighbor_expansion",
    "analysis_filter",
    "semantic_boundary_filter",
    "deterministic_ranking",
    "final_top_k",
    "reranker",
    "router",
]
NeighborProvenance = Literal["direct_hit", "neighbor_of"]


class SourceIdentity(BaseModel):
    arc_id: str
    story_type: str
    episode_name: str
    part_name: str
    file_path: str
    scene_start: int
    scene_end: int

    @model_validator(mode="after")
    def validate_span(self) -> "SourceIdentity":
        if self.scene_start < 0:
            raise ValueError("scene_start must be non-negative")
        if self.scene_end < self.scene_start:
            raise ValueError("scene_end must be greater than or equal to scene_start")
        return self


class TemporalConstraint(BaseModel):
    max_story_order: int | None = None
    min_story_order: int | None = None


class GlossaryExpectation(BaseModel):
    required_terms: list[str] = Field(default_factory=list)
    forbidden_terms: list[str] = Field(default_factory=list)


class GoldenQuestion(BaseModel):
    id: str
    question: str
    gold_sources: list[SourceIdentity]
    required_citation_labels: list[str] = Field(default_factory=list)
    glossary_expectations: GlossaryExpectation | None = None
    temporal_constraint: TemporalConstraint | None = None

    @model_validator(mode="after")
    def validate_gold_sources(self) -> "GoldenQuestion":
        if not self.gold_sources:
            raise ValueError("gold_sources must contain at least one source")
        return self


class GoldenSet(BaseModel):
    schema_version: str = "1"
    questions: list[GoldenQuestion]

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "GoldenSet":
        ids = [question.id for question in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("golden question ids must be unique")
        return self


class RunConfig(BaseModel):
    mode: EvalMode = "off"
    golden_set: str
    top_k: int = 8
    answer_mode: bool = False
    reranker_enabled: bool = False


class CandidateScores(BaseModel):
    dense_rank: int | None = None
    dense_distance: float | None = None
    lexical_rank: int | None = None
    lexical_score: float | None = None
    rrf_score: float | None = None
    deterministic_score: float | None = None


class CandidateTrace(BaseModel):
    node_id: str
    rank: int
    candidate_kind: CandidateKind = "raw_span"
    text: str | None = None
    scores: CandidateScores = Field(default_factory=CandidateScores)
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_span: SourceIdentity | None = None
    signal_breakdown: dict[str, Any] = Field(default_factory=dict)
    provenance: NeighborProvenance | None = None
    provenance_node_id: str | None = None


class StageTrace(BaseModel):
    name: StageName
    candidates: list[CandidateTrace] | None = None
    unavailable_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryTrace(BaseModel):
    query_id: str
    question: str
    mode: EvalMode = "off"
    config: dict[str, Any] = Field(default_factory=dict)
    stages: dict[StageName, StageTrace]
    final_citation_labels: list[str] = Field(default_factory=list)
    answer_text: str | None = None


class QueryMetrics(BaseModel):
    query_id: str
    gold_source_ranks: dict[str, int | None] = Field(default_factory=dict)
    recall_at_k: dict[str, bool] = Field(default_factory=dict)
    deterministic_ranking_hit: bool = False
    citation_correct: bool | None = None
    temporal_leakage: bool | None = None
    glossary_consistent: bool | None = None
    reranker_hit: bool | None = None


class AggregateMetrics(BaseModel):
    query_count: int
    recall_at_k: dict[str, float] = Field(default_factory=dict)
    deterministic_ranking_hit_rate: float
    citation_correctness: float | None = None
    temporal_leakage_rate: float | None = None
    glossary_consistency: float | None = None
    reranker_hit_rate: float | None = None
    unavailable_reasons: dict[str, str] = Field(default_factory=dict)


class EvalRun(BaseModel):
    schema_version: str = "1"
    config: RunConfig
    aggregate_metrics: AggregateMetrics
    query_metrics: list[QueryMetrics]
    traces: list[QueryTrace]


class GoldSourceRankDelta(BaseModel):
    query_id: str
    source_key: str
    before_rank: int | None = None
    after_rank: int | None = None
    delta: int | None = None


class EvalDiff(BaseModel):
    aggregate_deltas: dict[str, float | None] = Field(default_factory=dict)
    rank_deltas: list[GoldSourceRankDelta] = Field(default_factory=list)
