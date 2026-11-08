"""Deterministic evidence scoring and model-judged answer scoring for answer cases."""

import json
import re
from collections.abc import Callable, Iterable, Sequence
from typing import Any
from unicodedata import normalize

from linkura_story_indexer.eval.answer_models import (
    AnswerAggregate,
    AnswerCase,
    AnswerScore,
    CaseResult,
    ClaimEvidenceResult,
    EvidenceRef,
    EvidenceScore,
    JudgeVerdict,
)
from linkura_story_indexer.eval.models import QueryTrace
from linkura_story_indexer.eval.usage import merge_usage

Span = tuple[str, int, int]
# Resolves a Part file path to the highest story_order among its scenes.
StoryOrderResolver = Callable[[str], int | None]

_CITATION_PREFIX = re.compile(r"^CITATION\s*:\s*", re.IGNORECASE)
_CITATION_PATTERN = re.compile(r"[\[(]([^\[\]()]*?\d{3} · [^\[\]()]*?)[\])]")


def normalize_path(path: str) -> str:
    cleaned = normalize("NFC", path.replace("\\", "/"))
    return cleaned[2:] if cleaned.startswith("./") else cleaned


def _span_from_mapping(source: Any) -> Span | None:
    if not isinstance(source, dict):
        return None
    file_path = source.get("file_path")
    start = source.get("scene_start", source.get("scene_index"))
    end = source.get("scene_end", start)
    if not isinstance(file_path, str) or not isinstance(start, int) or not isinstance(end, int):
        return None
    return normalize_path(file_path), start, end


def _final_spans(trace: QueryTrace) -> list[Span]:
    stage = trace.stages.get("final_top_k")
    if stage is None or not stage.candidates:
        return []
    spans = []
    for candidate in stage.candidates:
        if candidate.source_span is not None:
            source = candidate.source_span
            spans.append((normalize_path(source.file_path), source.scene_start, source.scene_end))
    return spans


def _agent_tool_spans(trace: QueryTrace) -> list[Span]:
    stage = trace.stages.get("agent")
    if stage is None:
        return []
    spans = []
    for call in stage.metadata.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        for candidate in call.get("candidates") or []:
            if isinstance(candidate, dict) and (span := _span_from_mapping(candidate.get("source"))):
                spans.append(span)
    return spans


def seen_spans(trace: QueryTrace) -> list[Span]:
    """Every raw passage the run surfaced at any stage or tool call."""
    spans: list[Span] = []
    for stage in trace.stages.values():
        for candidate in stage.candidates or []:
            if candidate.source_span is not None:
                source = candidate.source_span
                spans.append((normalize_path(source.file_path), source.scene_start, source.scene_end))
    spans.extend(_agent_tool_spans(trace))
    return spans


def context_spans(trace: QueryTrace) -> list[Span]:
    """Raw passages available to the step that wrote the answer.

    Fixed-pipeline modes answer from the final top-k context. The agent writes its answer
    after reading every tool result, so everything it surfaced counts as context.
    """
    if trace.mode == "agentic":
        return [*_final_spans(trace), *_agent_tool_spans(trace)]
    return _final_spans(trace)


def _overlaps(span: Span, ref: EvidenceRef) -> bool:
    path, start, end = span
    return (
        path == normalize_path(ref.file_path)
        and start <= ref.scene_end
        and end >= ref.scene_start
    )


def _any_overlap(spans: Iterable[Span], refs: Sequence[EvidenceRef]) -> bool:
    span_list = list(spans)
    return any(_overlaps(span, ref) for ref in refs for span in span_list)


def extract_citations(answer_text: str) -> list[str]:
    citations: list[str] = []
    for match in _CITATION_PATTERN.finditer(answer_text):
        for label in match.group(1).split(";"):
            label = _CITATION_PREFIX.sub("", label.strip()).strip()
            if label and label not in citations:
                citations.append(label)
    return citations


def _story_order(metadata: dict[str, Any]) -> int | None:
    order = metadata.get("story_order", metadata.get("canonical_story_order"))
    return order if isinstance(order, int) and order > 0 else None


def score_evidence(
    case: AnswerCase,
    trace: QueryTrace,
    *,
    resolve_story_order: StoryOrderResolver | None = None,
    allowed_extra_citations: Iterable[str] = (),
) -> EvidenceScore:
    seen = seen_spans(trace)
    context = context_spans(trace)
    claims = [
        ClaimEvidenceResult(
            claim_id=claim.id,
            seen=_any_overlap(seen, claim.evidence),
            in_context=_any_overlap(context, claim.evidence),
        )
        for claim in case.expected_claims
        if claim.evidence
    ]
    score = EvidenceScore(claims=claims)
    if claims:
        score.all_required_seen = all(claim.seen for claim in claims)
        score.all_required_in_context = all(claim.in_context for claim in claims)

    if trace.answer_text:
        allowed = {*trace.final_citation_labels, *allowed_extra_citations}
        citations = extract_citations(trace.answer_text)
        score.citations_found = citations
        score.invalid_citations = [label for label in citations if label not in allowed]
    agent_stage = trace.stages.get("agent")
    if agent_stage is not None:
        unresolved = agent_stage.metadata.get("unresolved_cited_ids") or []
        score.invalid_citations.extend(f"unresolved id {item}" for item in unresolved)
        if not score.citations_found:
            score.citations_found = list(trace.final_citation_labels)
    if score.citations_found or score.invalid_citations:
        score.citation_valid = not score.invalid_citations

    boundary_path = case.scope.through_file_path if case.scope else None
    if boundary_path and resolve_story_order is not None:
        boundary = resolve_story_order(boundary_path)
        if boundary is not None:
            final_stage = trace.stages.get("final_top_k")
            leaked = []
            for candidate in (final_stage.candidates or []) if final_stage else []:
                order = _story_order(candidate.metadata)
                if order is not None and order > boundary:
                    leaked.append(candidate.node_id)
            score.temporal_leakage = bool(leaked)
            score.leaked_sources = leaked
    return score


JUDGE_INSTRUCTIONS = """You grade answers to questions about the Japanese story of Link! Like! LoveLive! \
(Hasunosora Girls' High School Idol Club). You receive the question, a grading rubric, reference \
claims with the original Japanese source passages that establish them, plausible-but-wrong \
claims, the candidate answer, and the passages the answering system itself retrieved.

Grade strictly against the reference passages, not your own knowledge of the franchise.

For each expected claim, choose one verdict:
- present: the answer states the claim, in any wording or language, with no material error.
- partial: the answer gets part of it right but omits or blurs a detail the rubric treats as essential.
- missing: the answer does not address the claim.
- contradicted: the answer states something incompatible with the claim.

For each forbidden claim, set asserted=true only if the answer states or clearly implies it. \
Mentioning it in order to reject it is not asserting it.

List in unsupported_statements every specific story-fact statement in the answer that neither the \
reference passages nor the system's retrieved passages support (invented events, wrong speakers, \
wrong episode placement, invented quotes). Do not list hedges, generic framing, or statements that \
are supported by the system's retrieved passages even when they go beyond the rubric.

The candidate answer must be an answer written for the question. If it is instead a raw dump of \
retrieved passages, tool output, or search results, mark every expected claim missing even when \
the right facts appear somewhere inside it, and say so in notes.

Set abstained=true when the answer declines to answer or says the sources are insufficient for \
the main question, even if it adds partial information.

Write rationales in one short English sentence each. Use notes for anything the rubric does not \
cover."""


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n…[truncated]"


def build_judge_prompt(
    case: AnswerCase,
    answer_text: str,
    *,
    reference_passages: dict[str, str],
    system_passages: Sequence[str],
    passage_char_limit: int = 6000,
) -> str:
    claims = []
    for claim in case.expected_claims:
        refs = []
        for ref in claim.evidence:
            key = f"{normalize_path(ref.file_path)}#{ref.scene_start}-{ref.scene_end}"
            refs.append(
                {
                    "source": key,
                    "quote": ref.quote,
                    "passage": _clip(reference_passages.get(key, "(unavailable)"), passage_char_limit),
                }
            )
        entry: dict[str, Any] = {"id": claim.id, "claim": claim.claim, "acceptable_sources": refs}
        if claim.verified_by:
            entry["verified_by"] = claim.verified_by
        claims.append(entry)
    payload = {
        "question": case.question,
        "rubric": case.rubric,
        "expect_abstention": case.expect_abstention,
        "expected_claims": claims,
        "forbidden_claims": [
            {"id": claim.id, "claim": claim.claim, "note": claim.note}
            for claim in case.forbidden_claims
        ],
        "candidate_answer": answer_text,
        "system_retrieved_passages": [
            _clip(passage, passage_char_limit) for passage in system_passages
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def no_final_answer_reason(agent_metadata: dict[str, Any] | None) -> str | None:
    """Why an agentic run has no real answer, or None when it produced one.

    Runs that stopped early and returned raw tool output are failures regardless of what the
    output happens to contain. Older traces without ``answer_source`` count any early stop.
    """
    if not agent_metadata:
        return None
    stop_reason = agent_metadata.get("stop_reason")
    answer_source = agent_metadata.get("answer_source")
    if answer_source == "tool_fallback" or (
        answer_source is None and stop_reason not in {None, "final_answer"}
    ):
        return f"no final answer: agent stopped ({stop_reason}) and returned raw tool output"
    return None


def no_final_answer_score(reason: str) -> AnswerScore:
    return AnswerScore(judged=True, failure_reason=reason, completeness=0.0, passed=False)


def score_answer(case: AnswerCase, verdict: JudgeVerdict) -> AnswerScore:
    judged = {judgment.claim_id: judgment for judgment in verdict.claims}
    weights = {"present": 1.0, "partial": 0.5, "missing": 0.0, "contradicted": 0.0}
    claim_ids = [claim.id for claim in case.expected_claims]
    verdicts = [judged[claim_id].verdict if claim_id in judged else "missing" for claim_id in claim_ids]
    completeness = (
        sum(weights[item] for item in verdicts) / len(verdicts) if verdicts else None
    )
    forbidden_ids = {claim.id for claim in case.forbidden_claims}
    forbidden_asserted = any(
        judgment.asserted for judgment in verdict.forbidden if judgment.claim_id in forbidden_ids
    )
    correct = "contradicted" not in verdicts and not forbidden_asserted
    supported = not verdict.unsupported_statements
    abstention_appropriate = (
        None if case.expect_abstention is None else verdict.abstained == case.expect_abstention
    )
    all_present = all(item == "present" for item in verdicts)
    return AnswerScore(
        judged=True,
        completeness=completeness,
        correct=correct,
        supported=supported,
        abstention_appropriate=abstention_appropriate,
        passed=correct and all_present and abstention_appropriate is not False,
        verdict=verdict,
    )


def _rate(values: Iterable[bool | None]) -> float | None:
    concrete = [value for value in values if value is not None]
    if not concrete:
        return None
    return sum(1 for value in concrete if value) / len(concrete)


def _mean(values: Iterable[float | None]) -> float | None:
    concrete = [value for value in values if value is not None]
    return sum(concrete) / len(concrete) if concrete else None


def aggregate_results(results: Sequence[CaseResult]) -> AnswerAggregate:
    judged = [result for result in results if result.answer.judged]
    crosstab: dict[str, int] = {}
    for result in judged:
        evidence = result.evidence.all_required_in_context
        if evidence is None:
            continue
        key = f"evidence_{'yes' if evidence else 'no'}__answer_{'pass' if result.answer.passed else 'fail'}"
        crosstab[key] = crosstab.get(key, 0) + 1
    by_category: dict[str, list[bool]] = {}
    for result in judged:
        for category in result.categories:
            by_category.setdefault(category, []).append(bool(result.answer.passed))
    return AnswerAggregate(
        case_count=len(results),
        judged_count=len(judged),
        pass_rate=_rate(result.answer.passed for result in judged),
        mean_completeness=_mean(result.answer.completeness for result in judged),
        correct_rate=_rate(result.answer.correct for result in judged),
        supported_rate=_rate(result.answer.supported for result in judged),
        abstention_accuracy=_rate(result.answer.abstention_appropriate for result in judged),
        evidence_seen_rate=_rate(result.evidence.all_required_seen for result in results),
        evidence_in_context_rate=_rate(result.evidence.all_required_in_context for result in results),
        citation_valid_rate=_rate(result.evidence.citation_valid for result in results),
        temporal_leakage_rate=_rate(result.evidence.temporal_leakage for result in results),
        evidence_vs_answer=dict(sorted(crosstab.items())),
        pass_rate_by_category={
            category: sum(values) / len(values) for category, values in sorted(by_category.items())
        },
        error_count=sum(1 for result in results if result.error),
        no_final_answer_count=sum(1 for result in results if result.answer.failure_reason),
        mean_latency_seconds=_mean(result.latency_seconds for result in results),
        usage=merge_usage([result.usage for result in results]),
        judge_usage=merge_usage([result.judge_usage for result in results]),
    )
