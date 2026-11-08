"""Schema for the answer-accuracy benchmark (issue #81).

The retrieval golden set (``eval/golden_questions.json``) scores whether one gold scene
reaches the top-k. This suite scores whether the final *answer* is correct, complete,
supported, and appropriately cautious, while keeping a deterministic evidence check
separate from the model-judged answer quality.
"""

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from linkura_story_indexer.eval.models import EvalMode

AnswerCategory = Literal[
    "fact",
    "causal",
    "relationship_change",
    "quote",
    "alias",
    "chronology",
    "interleave",
    "count",
    "unanswerable",
]
CaseOrigin = Literal["observed_failure", "curated"]
ReviewStatus = Literal["draft", "reviewed"]
ClaimVerdict = Literal["present", "partial", "missing", "contradicted"]


class EvidenceRef(BaseModel):
    """One acceptable source passage for a claim (scene indexes are zero-based, inclusive)."""

    file_path: str
    scene_start: int
    scene_end: int
    quote: str | None = Field(
        default=None,
        description="Short source excerpt so reviewers can verify the claim without opening the file.",
    )

    @model_validator(mode="after")
    def validate_span(self) -> "EvidenceRef":
        if self.scene_start < 0:
            raise ValueError("scene_start must be non-negative")
        if self.scene_end < self.scene_start:
            raise ValueError("scene_end must be greater than or equal to scene_start")
        return self


class ExpectedClaim(BaseModel):
    """A claim a correct answer must make. ``evidence`` lists alternate acceptable sources.

    Claims established by a deterministic query instead of a passage (exact dialogue counts)
    record that query in ``verified_by`` and may omit evidence; the evidence scorer skips them.
    """

    id: str
    claim: str
    evidence: list[EvidenceRef] = Field(default_factory=list)
    verified_by: str | None = None

    @model_validator(mode="after")
    def validate_evidence(self) -> "ExpectedClaim":
        if not self.evidence and not self.verified_by:
            raise ValueError(
                f"expected claim {self.id!r} needs at least one evidence passage or verified_by"
            )
        return self


class ForbiddenClaim(BaseModel):
    """A plausible but wrong claim; asserting it fails the case."""

    id: str
    claim: str
    note: str | None = None


class CaseScope(BaseModel):
    """Spoiler boundary: evidence after ``through_file_path`` (inclusive Part) is leakage."""

    through_file_path: str | None = None
    description: str | None = None


class AnswerCase(BaseModel):
    id: str
    question: str
    categories: list[AnswerCategory]
    origin: CaseOrigin
    origin_note: str
    review_status: ReviewStatus = "draft"
    review_note: str | None = None
    expected_claims: list[ExpectedClaim] = Field(default_factory=list)
    forbidden_claims: list[ForbiddenClaim] = Field(default_factory=list)
    expect_abstention: bool | None = False
    """True: the answer must decline. False: it must answer. None: not checked, for questions
    whose correct answer is that the story does not say (the claims carry the scoring)."""
    scope: CaseScope | None = None
    rubric: str

    @model_validator(mode="after")
    def validate_case(self) -> "AnswerCase":
        if not self.categories:
            raise ValueError(f"case {self.id!r} needs at least one category")
        if self.expect_abstention is not True and not self.expected_claims:
            raise ValueError(f"answerable case {self.id!r} needs at least one expected claim")
        claim_ids = [claim.id for claim in self.expected_claims] + [
            claim.id for claim in self.forbidden_claims
        ]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError(f"claim ids must be unique within case {self.id!r}")
        return self


class AnswerCaseSet(BaseModel):
    schema_version: str = "1"
    suite_version: str
    cases: list[AnswerCase]

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "AnswerCaseSet":
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("answer case ids must be unique")
        return self

    def to_file_json(self) -> str:
        """Serializes the suite for its checked-in JSON file.

        Unset optional fields are omitted to keep the file readable, except
        ``expect_abstention``, where ``null`` means "not checked" rather than "unset".
        """
        data = self.model_dump(mode="json", exclude_none=True)
        for case, raw in zip(self.cases, data["cases"], strict=True):
            raw["expect_abstention"] = case.expect_abstention
        return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


class ModelCallUsage(BaseModel):
    """Usage for one model request, as reported by the provider."""

    model_name: str
    provider: str | None = None
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0


class UsageSummary(BaseModel):
    requests: int = 0
    input_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float | None = None
    cost_unavailable_reason: str | None = None
    by_model: dict[str, dict[str, int]] = Field(default_factory=dict)


class ClaimEvidenceResult(BaseModel):
    claim_id: str
    seen: bool
    """An acceptable passage appeared anywhere the run looked (any tool call or stage)."""
    in_context: bool
    """An acceptable passage reached the final evidence given to the answer step."""


class EvidenceScore(BaseModel):
    """Deterministic provenance checks; no model involved."""

    claims: list[ClaimEvidenceResult] = Field(default_factory=list)
    all_required_seen: bool | None = None
    all_required_in_context: bool | None = None
    citations_found: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    citation_valid: bool | None = None
    temporal_leakage: bool | None = None
    leaked_sources: list[str] = Field(default_factory=list)


class ClaimJudgment(BaseModel):
    claim_id: str
    verdict: ClaimVerdict
    rationale: str


class ForbiddenJudgment(BaseModel):
    claim_id: str
    asserted: bool
    rationale: str


class JudgeVerdict(BaseModel):
    """Structured output of the answer judge."""

    claims: list[ClaimJudgment]
    forbidden: list[ForbiddenJudgment] = Field(default_factory=list)
    unsupported_statements: list[str] = Field(
        default_factory=list,
        description="Story-fact statements in the answer that neither the reference passages "
        "nor the system's own evidence support.",
    )
    abstained: bool
    notes: str = ""


class AnswerScore(BaseModel):
    """Answer-quality scores derived from the judge verdict."""

    judged: bool = False
    unavailable_reason: str | None = None
    failure_reason: str | None = None
    """Set when the case failed without judging, e.g. the agent never produced an answer."""
    completeness: float | None = None
    correct: bool | None = None
    """No expected claim contradicted and no forbidden claim asserted."""
    supported: bool | None = None
    abstention_appropriate: bool | None = None
    passed: bool | None = None
    verdict: JudgeVerdict | None = None


class CaseResult(BaseModel):
    case_id: str
    categories: list[AnswerCategory]
    review_status: ReviewStatus
    mode: EvalMode
    answer_text: str | None = None
    error: str | None = None
    latency_seconds: float | None = None
    usage: UsageSummary = Field(default_factory=UsageSummary)
    judge_usage: UsageSummary = Field(default_factory=UsageSummary)
    evidence: EvidenceScore = Field(default_factory=EvidenceScore)
    answer: AnswerScore = Field(default_factory=AnswerScore)
    trace: dict[str, Any] | None = None


class AnswerRunConfig(BaseModel):
    mode: EvalMode
    cases_path: str
    suite_version: str
    generation_provider: str
    generation_model: str
    router_model: str | None = None
    embedding_model: str
    answer_prompt_version: str
    summary_prompt_version: str
    judge_model: str | None = None
    agent_request_limit: int | None = None
    final_top_k: int
    corpus_revision: str
    code_revision: str
    started_at: str
    reviewed_only: bool = False


class AnswerAggregate(BaseModel):
    case_count: int
    judged_count: int
    pass_rate: float | None = None
    mean_completeness: float | None = None
    correct_rate: float | None = None
    supported_rate: float | None = None
    abstention_accuracy: float | None = None
    evidence_seen_rate: float | None = None
    evidence_in_context_rate: float | None = None
    citation_valid_rate: float | None = None
    temporal_leakage_rate: float | None = None
    evidence_vs_answer: dict[str, int] = Field(default_factory=dict)
    pass_rate_by_category: dict[str, float] = Field(default_factory=dict)
    error_count: int = 0
    no_final_answer_count: int = 0
    mean_latency_seconds: float | None = None
    usage: UsageSummary = Field(default_factory=UsageSummary)
    judge_usage: UsageSummary = Field(default_factory=UsageSummary)


class AnswerEvalRun(BaseModel):
    schema_version: str = "1"
    config: AnswerRunConfig
    aggregate: AnswerAggregate
    results: list[CaseResult]
