from collections.abc import Iterable
from unicodedata import normalize

from linkura_story_indexer.eval.models import (
    AggregateMetrics,
    EvalDiff,
    EvalRun,
    GoldenQuestion,
    GoldSourceRankDelta,
    QueryMetrics,
    QueryTrace,
    SourceIdentity,
    StageName,
)

DEFAULT_RECALL_KS = (1, 3, 5, 8)


def source_key(source: SourceIdentity) -> str:
    return "|".join(
        [
            source.arc_id,
            source.story_type,
            source.episode_name,
            source.part_name,
            source.file_path,
            str(source.scene_start),
            str(source.scene_end),
        ]
    )


def source_overlaps(candidate: SourceIdentity | None, gold: SourceIdentity) -> bool:
    if candidate is None:
        return False
    candidate_path = normalize("NFC", candidate.file_path.replace("\\", "/"))
    gold_path = normalize("NFC", gold.file_path.replace("\\", "/"))
    return (
        normalize("NFC", candidate.arc_id) == normalize("NFC", gold.arc_id)
        and normalize("NFC", candidate.story_type) == normalize("NFC", gold.story_type)
        and normalize("NFC", candidate.episode_name) == normalize("NFC", gold.episode_name)
        and normalize("NFC", candidate.part_name) == normalize("NFC", gold.part_name)
        and candidate_path == gold_path
        and candidate.scene_start <= gold.scene_end
        and candidate.scene_end >= gold.scene_start
    )


def _stage_sources(trace: QueryTrace, stage_name: StageName) -> list[SourceIdentity | None]:
    stage = trace.stages.get(stage_name)
    if stage is None or stage.candidates is None:
        return []
    return [candidate.source_span for candidate in stage.candidates]


def _gold_ranks(trace: QueryTrace, question: GoldenQuestion) -> dict[str, int | None]:
    final_stage = trace.stages.get("final_top_k")
    candidates = final_stage.candidates if final_stage and final_stage.candidates else []
    ranks: dict[str, int | None] = {}
    for gold in question.gold_sources:
        rank = None
        for candidate in candidates:
            if source_overlaps(candidate.source_span, gold):
                rank = candidate.rank
                break
        ranks[source_key(gold)] = rank
    return ranks


def _any_gold_in_stage(trace: QueryTrace, question: GoldenQuestion, stage_name: StageName) -> bool:
    return any(
        source_overlaps(candidate_source, gold)
        for candidate_source in _stage_sources(trace, stage_name)
        for gold in question.gold_sources
    )


def _citation_correct(trace: QueryTrace, question: GoldenQuestion) -> bool | None:
    if not question.required_citation_labels:
        return None
    text = "\n".join([*trace.final_citation_labels, trace.answer_text or ""])
    return all(label in text for label in question.required_citation_labels)


def _temporal_leakage(trace: QueryTrace, question: GoldenQuestion) -> bool | None:
    constraint = question.temporal_constraint
    if constraint is None:
        return None
    final_stage = trace.stages.get("final_top_k")
    candidates = final_stage.candidates if final_stage and final_stage.candidates else []
    for candidate in candidates:
        order = candidate.metadata.get("story_order", candidate.metadata.get("canonical_story_order"))
        if not isinstance(order, int):
            continue
        if constraint.max_story_order is not None and order > constraint.max_story_order:
            return True
        if constraint.min_story_order is not None and order < constraint.min_story_order:
            return True
    return False


def _glossary_consistent(trace: QueryTrace, question: GoldenQuestion) -> bool | None:
    expectation = question.glossary_expectations
    if expectation is None:
        return None
    text = trace.answer_text or "\n".join(trace.final_citation_labels)
    return all(term in text for term in expectation.required_terms) and not any(
        term in text for term in expectation.forbidden_terms
    )


def _audit_metrics(trace: QueryTrace) -> tuple[bool | None, int | None]:
    stage = trace.stages.get("audit")
    if stage is None:
        return None, None
    metadata = stage.metadata
    report = metadata.get("report", metadata.get("audit_report"))
    if isinstance(report, dict):
        flags = report.get("flags", [])
        errors = report.get("errors", [])
    elif hasattr(report, "model_dump"):
        report_data = report.model_dump(mode="json")
        flags = report_data.get("flags", [])
        errors = report_data.get("errors", [])
    else:
        flags = metadata.get("flags", [])
        errors = metadata.get("errors", [])
    if not isinstance(flags, list) or not isinstance(errors, list):
        return None, None
    return not flags and not errors, len(flags)


def _rate(values: Iterable[bool | None], *, invert: bool = False) -> float | None:
    concrete = [value for value in values if value is not None]
    if not concrete:
        return None
    hits = sum(1 for value in concrete if (not value if invert else value))
    return hits / len(concrete)


def evaluate_query(trace: QueryTrace, question: GoldenQuestion) -> QueryMetrics:
    ranks = _gold_ranks(trace, question)
    recall = {
        f"recall@{k}": any(rank is not None and rank <= k for rank in ranks.values())
        for k in DEFAULT_RECALL_KS
    }
    reranker_stage = trace.stages.get("reranker")
    reranker_hit = None
    if reranker_stage is not None and reranker_stage.candidates is not None:
        reranker_hit = _any_gold_in_stage(trace, question, "reranker")
    audit_clean, audit_flag_count = _audit_metrics(trace)
    return QueryMetrics(
        query_id=question.id,
        gold_source_ranks=ranks,
        recall_at_k=recall,
        deterministic_ranking_hit=_any_gold_in_stage(trace, question, "deterministic_ranking"),
        citation_correct=_citation_correct(trace, question),
        temporal_leakage=_temporal_leakage(trace, question),
        glossary_consistent=_glossary_consistent(trace, question),
        reranker_hit=reranker_hit,
        audit_clean=audit_clean,
        audit_flag_count=audit_flag_count,
    )


def aggregate_metrics(query_metrics: list[QueryMetrics], traces: list[QueryTrace]) -> AggregateMetrics:
    query_count = len(query_metrics)
    recall_keys = sorted({key for metric in query_metrics for key in metric.recall_at_k})
    recall_at_k = {
        key: sum(1 for metric in query_metrics if metric.recall_at_k.get(key)) / query_count
        if query_count
        else 0.0
        for key in recall_keys
    }
    unavailable_reasons = {}
    if traces and all(
        trace.stages.get("reranker") and trace.stages["reranker"].candidates is None
        for trace in traces
    ):
        unavailable_reasons["reranker_hit_rate"] = "reranker stage unavailable; #23 has not landed"
    reranker_hit_rate = _rate(metric.reranker_hit for metric in query_metrics)
    audit_clean_rate = _rate(metric.audit_clean for metric in query_metrics)
    return AggregateMetrics(
        query_count=query_count,
        recall_at_k=recall_at_k,
        deterministic_ranking_hit_rate=(
            sum(1 for metric in query_metrics if metric.deterministic_ranking_hit) / query_count
            if query_count
            else 0.0
        ),
        citation_correctness=_rate(metric.citation_correct for metric in query_metrics),
        temporal_leakage_rate=_rate(metric.temporal_leakage for metric in query_metrics),
        glossary_consistency=_rate(metric.glossary_consistent for metric in query_metrics),
        reranker_hit_rate=reranker_hit_rate,
        audit_clean_rate=audit_clean_rate,
        unavailable_reasons=unavailable_reasons,
    )


def diff_runs(before: EvalRun, after: EvalRun) -> EvalDiff:
    deltas: dict[str, float | None] = {}
    before_metrics = before.aggregate_metrics
    after_metrics = after.aggregate_metrics
    for key in sorted(set(before_metrics.recall_at_k) | set(after_metrics.recall_at_k)):
        deltas[key] = after_metrics.recall_at_k.get(key, 0.0) - before_metrics.recall_at_k.get(key, 0.0)
    for field in (
        "deterministic_ranking_hit_rate",
        "citation_correctness",
        "temporal_leakage_rate",
        "glossary_consistency",
        "reranker_hit_rate",
        "audit_clean_rate",
    ):
        before_value = getattr(before_metrics, field)
        after_value = getattr(after_metrics, field)
        deltas[field] = None if before_value is None or after_value is None else after_value - before_value

    after_by_id = {metric.query_id: metric for metric in after.query_metrics}
    rank_deltas = []
    for before_query in before.query_metrics:
        after_query = after_by_id.get(before_query.query_id)
        if after_query is None:
            continue
        for key, before_rank in before_query.gold_source_ranks.items():
            after_rank = after_query.gold_source_ranks.get(key)
            delta = None if before_rank is None or after_rank is None else after_rank - before_rank
            rank_deltas.append(
                GoldSourceRankDelta(
                    query_id=before_query.query_id,
                    source_key=key,
                    before_rank=before_rank,
                    after_rank=after_rank,
                    delta=delta,
                )
            )
    return EvalDiff(aggregate_deltas=deltas, rank_deltas=rank_deltas)
