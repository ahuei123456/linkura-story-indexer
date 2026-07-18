import json
from pathlib import Path
from typing import Any

from linkura_story_indexer.eval.io import load_golden_set, stable_json
from linkura_story_indexer.eval.metrics import aggregate_metrics, diff_runs, evaluate_query
from linkura_story_indexer.eval.models import (
    AggregateMetrics,
    CandidateTrace,
    EvalRun,
    GoldenQuestion,
    QueryMetrics,
    QueryTrace,
    RunConfig,
    SourceIdentity,
    StageTrace,
    TemporalConstraint,
)


def source(
    *,
    scene_start: int = 0,
    scene_end: int = 0,
    file_path: str = "story/105/第5話『眠れる海のお姫様！』/ABYSS.md",
) -> SourceIdentity:
    return SourceIdentity(
        arc_id="105",
        story_type="Main",
        episode_name="第5話『眠れる海のお姫様！』",
        part_name="ABYSS",
        file_path=file_path,
        scene_start=scene_start,
        scene_end=scene_end,
    )


def candidate(
    rank: int,
    candidate_source: SourceIdentity,
    *,
    story_order: int = 10,
) -> CandidateTrace:
    return CandidateTrace(
        node_id=f"node-{rank}",
        rank=rank,
        metadata={"canonical_story_order": story_order},
        source_span=candidate_source,
    )


def trace(
    *,
    final_candidates: list[CandidateTrace],
    deterministic_candidates: list[CandidateTrace] | None = None,
    citations: list[str] | None = None,
    answer_text: str | None = None,
    reranker_candidates: list[CandidateTrace] | None = None,
) -> QueryTrace:
    deterministic_candidates = deterministic_candidates or final_candidates
    return QueryTrace(
        query_id="q1",
        question="question",
        stages={
            "dense_raw": StageTrace(name="dense_raw", candidates=[]),
            "lexical_raw": StageTrace(name="lexical_raw", candidates=[]),
            "rrf_fusion": StageTrace(name="rrf_fusion", candidates=[]),
            "raw_seed_filter": StageTrace(name="raw_seed_filter", candidates=[]),
            "neighbor_expansion": StageTrace(name="neighbor_expansion", candidates=[]),
            "analysis_filter": StageTrace(name="analysis_filter", candidates=None),
            "semantic_boundary_filter": StageTrace(name="semantic_boundary_filter", candidates=None),
            "deterministic_ranking": StageTrace(
                name="deterministic_ranking",
                candidates=deterministic_candidates,
            ),
            "final_top_k": StageTrace(name="final_top_k", candidates=final_candidates),
            "reranker": StageTrace(
                name="reranker",
                candidates=reranker_candidates,
                unavailable_reason=None
                if reranker_candidates is not None
                else "reranker stage unavailable; #23 has not landed",
            ),
        },
        final_citation_labels=citations or [],
        answer_text=answer_text,
    )


def question(**overrides: Any) -> GoldenQuestion:
    values = {
        "id": "q1",
        "question": "question",
        "gold_sources": [source(scene_start=1, scene_end=2)],
    }
    values.update(overrides)
    return GoldenQuestion(**values)


def test_checked_in_golden_set_validates_scene_span_regression() -> None:
    golden = load_golden_set("eval/golden_questions.json")

    regression = next(item for item in golden.questions if item.id == "q001_izumi_acting_past")

    assert len(golden.questions) == 30
    assert regression.gold_sources == [source(scene_start=0, scene_end=0)]
    assert "Izumi's acting background" in regression.question


def test_metrics_count_partial_span_overlap_and_recall() -> None:
    metric = evaluate_query(
        trace(final_candidates=[candidate(1, source(scene_start=0, scene_end=3))]),
        question(),
    )

    assert metric.gold_source_ranks == {
        "105|Main|第5話『眠れる海のお姫様！』|ABYSS|story/105/第5話『眠れる海のお姫様！』/ABYSS.md|1|2": 1
    }
    assert metric.recall_at_k["recall@1"] is True


def test_metrics_report_miss_citation_temporal_and_glossary_results() -> None:
    metric = evaluate_query(
        trace(
            final_candidates=[candidate(4, source(scene_start=8, scene_end=8), story_order=20)],
            citations=["wrong citation"],
            answer_text="Uses Izumi but not the required translated unit name.",
        ),
        question(
            required_citation_labels=["105 · Episode 5 · Part ABYSS · Scene 2"],
            temporal_constraint=TemporalConstraint(max_story_order=15),
            glossary_expectations={"required_terms": ["Hasunosora"], "forbidden_terms": ["bad"]},
        ),
    )

    assert metric.gold_source_ranks[
        "105|Main|第5話『眠れる海のお姫様！』|ABYSS|story/105/第5話『眠れる海のお姫様！』/ABYSS.md|1|2"
    ] is None
    assert metric.citation_correct is False
    assert metric.temporal_leakage is True
    assert metric.glossary_consistent is False
    assert metric.reranker_hit is None


def test_aggregate_marks_unavailable_reranker_metrics() -> None:
    traces = [trace(final_candidates=[candidate(1, source(scene_start=1, scene_end=1))])]
    metrics = [evaluate_query(traces[0], question())]

    aggregate = aggregate_metrics(metrics, traces)

    assert aggregate.reranker_hit_rate is None
    assert aggregate.unavailable_reasons == {
        "reranker_hit_rate": "reranker stage unavailable; #23 has not landed"
    }


def test_stable_json_is_byte_identical_for_equivalent_trace_dict_order() -> None:
    first = trace(final_candidates=[candidate(1, source(scene_start=1, scene_end=1))])
    second_data = json.loads(stable_json(first))
    second = QueryTrace.model_validate(second_data)

    assert stable_json(first) == stable_json(second)


def test_diff_reports_aggregate_and_rank_changes() -> None:
    before_metric = QueryMetrics(
        query_id="q1",
        gold_source_ranks={"gold": 5},
        recall_at_k={"recall@5": True},
    )
    after_metric = QueryMetrics(
        query_id="q1",
        gold_source_ranks={"gold": 2},
        recall_at_k={"recall@5": True},
    )
    before = EvalRun(
        config=RunConfig(golden_set="golden.json"),
        aggregate_metrics=AggregateMetrics(
            query_count=1,
            recall_at_k={"recall@5": 1.0},
            deterministic_ranking_hit_rate=0.0,
        ),
        query_metrics=[before_metric],
        traces=[],
    )
    after = EvalRun(
        config=RunConfig(golden_set="golden.json"),
        aggregate_metrics=AggregateMetrics(
            query_count=1,
            recall_at_k={"recall@5": 1.0},
            deterministic_ranking_hit_rate=1.0,
        ),
        query_metrics=[after_metric],
        traces=[],
    )

    diff = diff_runs(before, after)

    assert diff.aggregate_deltas["deterministic_ranking_hit_rate"] == 1.0
    assert diff.rank_deltas[0].before_rank == 5
    assert diff.rank_deltas[0].after_rank == 2
    assert diff.rank_deltas[0].delta == -3


def test_audit_metrics_aggregate_and_diff() -> None:
    clean_trace = trace(final_candidates=[candidate(1, source(scene_start=1, scene_end=1))])
    clean_trace.stages["audit"] = StageTrace(
        name="audit",
        candidates=None,
        metadata={"report": {"flags": [], "errors": []}},
    )
    flagged_trace = trace(final_candidates=[candidate(1, source(scene_start=1, scene_end=1))])
    flagged_trace.stages["audit"] = StageTrace(
        name="audit",
        candidates=None,
        metadata={
            "report": {
                "flags": [
                    {
                        "flag_type": "retcon",
                        "excerpt": "later",
                        "rationale": "interval",
                        "evidence_ref": "ledger",
                    }
                ],
                "errors": [],
            }
        },
    )

    clean_metric = evaluate_query(clean_trace, question())
    flagged_metric = evaluate_query(flagged_trace, question())
    aggregate = aggregate_metrics(
        [clean_metric, flagged_metric],
        [clean_trace, flagged_trace],
    )

    assert clean_metric.audit_clean is True
    assert clean_metric.audit_flag_count == 0
    assert flagged_metric.audit_clean is False
    assert flagged_metric.audit_flag_count == 1
    assert aggregate.audit_clean_rate == 0.5

    before = EvalRun(
        config=RunConfig(golden_set="golden.json", audit_enabled=True),
        aggregate_metrics=aggregate,
        query_metrics=[clean_metric, flagged_metric],
        traces=[clean_trace, flagged_trace],
    )
    after = EvalRun(
        config=RunConfig(golden_set="golden.json", audit_enabled=True),
        aggregate_metrics=aggregate.model_copy(update={"audit_clean_rate": 1.0}),
        query_metrics=[clean_metric, flagged_metric],
        traces=[clean_trace, flagged_trace],
    )

    assert diff_runs(before, after).aggregate_deltas["audit_clean_rate"] == 0.5


def test_golden_set_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "golden.json"
    payload = {
        "questions": [
            {"id": "duplicate", "question": "one", "gold_sources": [source().model_dump()]},
            {"id": "duplicate", "question": "two", "gold_sources": [source().model_dump()]},
        ]
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    try:
        load_golden_set(path)
    except ValueError as exc:
        assert "golden question ids must be unique" in str(exc)
    else:
        raise AssertionError("duplicate ids should be rejected")
