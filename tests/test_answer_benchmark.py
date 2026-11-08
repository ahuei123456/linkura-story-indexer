import json
import unicodedata
from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from linkura_story_indexer.eval.answer_models import (
    AnswerCase,
    AnswerCaseSet,
    ClaimJudgment,
    ClaimVerdict,
    ForbiddenJudgment,
    JudgeVerdict,
    ModelCallUsage,
)
from linkura_story_indexer.eval.answer_runner import (
    CorpusIndex,
    load_answer_cases,
    rejudge_run,
    render_answer_report,
    rescore_run,
    run_answer_eval,
    run_labels,
    validate_case_sources,
)
from linkura_story_indexer.eval.answer_scoring import (
    build_judge_prompt,
    extract_citations,
    no_final_answer_reason,
    score_answer,
    score_evidence,
)
from linkura_story_indexer.eval.models import (
    CandidateTrace,
    EvalMode,
    QueryTrace,
    SourceIdentity,
    StageName,
    StageTrace,
)
from linkura_story_indexer.eval.usage import (
    UsageRecordingModel,
    merge_usage,
    record_usage,
    summarize_usage,
)

EP8 = "story/104/第8話『Not a marionette』/2.md"
EP8_PT3 = "story/104/第8話『Not a marionette』/3.md"
EP9 = "story/104/第9話『みらくる・ざ・ゆにばーす！』/1.md"


def _case(**overrides) -> AnswerCase:
    data = {
        "id": "c1",
        "question": "Who is Reika-san?",
        "categories": ["alias"],
        "origin": "observed_failure",
        "origin_note": "fixture",
        "expected_claims": [
            {
                "id": "a",
                "claim": "Reika-san is a different person from Aoi Reisawa.",
                "evidence": [
                    {"file_path": EP8, "scene_start": 2, "scene_end": 2},
                    {"file_path": EP8_PT3, "scene_start": 0, "scene_end": 1},
                ],
            },
            {
                "id": "b",
                "claim": "She appears in Episode 8.",
                "evidence": [{"file_path": EP8, "scene_start": 4, "scene_end": 4}],
            },
        ],
        "forbidden_claims": [{"id": "x", "claim": "Reika-san is Aoi Reisawa."}],
        "rubric": "Must not merge the two characters.",
    }
    data.update(overrides)
    return AnswerCase.model_validate(data)


def _candidate(
    path: str,
    start: int,
    end: int | None = None,
    *,
    rank: int = 1,
    order: int | None = None,
    text: str = "passage",
) -> CandidateTrace:
    end = start if end is None else end
    metadata = {"story_order": order} if order is not None else {}
    return CandidateTrace(
        node_id=f"{path}:{start}-{end}",
        rank=rank,
        text=text,
        metadata=metadata,
        source_span=SourceIdentity(
            arc_id="104",
            story_type="Main",
            episode_name="ep",
            part_name="part",
            file_path=path,
            scene_start=start,
            scene_end=end,
        ),
    )


def _trace(
    final: list[CandidateTrace],
    *,
    mode: EvalMode = "off",
    dense: list[CandidateTrace] | None = None,
    answer: str | None = "answer",
    labels: list[str] | None = None,
    agent_calls: list[dict] | None = None,
) -> QueryTrace:
    stages: dict[StageName, StageTrace] = {"final_top_k": StageTrace(name="final_top_k", candidates=final)}
    if dense is not None:
        stages["dense_raw"] = StageTrace(name="dense_raw", candidates=dense)
    if agent_calls is not None:
        stages["agent"] = StageTrace(name="agent", metadata={"tool_calls": agent_calls})
    return QueryTrace(
        query_id="c1",
        question="q",
        mode=mode,
        stages=stages,
        final_citation_labels=labels or [],
        answer_text=answer,
    )


def _verdict(claims: dict[str, ClaimVerdict], *, forbidden=False, unsupported=(), abstained=False):
    return JudgeVerdict(
        claims=[ClaimJudgment(claim_id=key, verdict=value, rationale="r") for key, value in claims.items()],
        forbidden=[ForbiddenJudgment(claim_id="x", asserted=forbidden, rationale="r")],
        unsupported_statements=list(unsupported),
        abstained=abstained,
    )


def test_schema_rejects_answerable_case_without_claims():
    with pytest.raises(ValidationError, match="needs at least one expected claim"):
        _case(expected_claims=[])


def test_schema_allows_unanswerable_case_without_claims():
    case = _case(expected_claims=[], expect_abstention=True, categories=["unanswerable"])
    assert case.expect_abstention


def test_schema_rejects_claim_without_evidence_and_duplicate_ids():
    with pytest.raises(ValidationError, match="at least one evidence"):
        _case(expected_claims=[{"id": "a", "claim": "c", "evidence": []}])
    counted = _case(expected_claims=[{"id": "a", "claim": "10 turns", "verified_by": "count_turns"}])
    assert counted.expected_claims[0].evidence == []
    with pytest.raises(ValidationError, match="unique within case"):
        _case(forbidden_claims=[{"id": "a", "claim": "dup"}])
    with pytest.raises(ValidationError, match="ids must be unique"):
        AnswerCaseSet(suite_version="1", cases=[_case(), _case()])


def test_evidence_requires_every_claim_source():
    case = _case()
    only_first = score_evidence(case, _trace([_candidate(EP8, 2)]))
    assert [claim.in_context for claim in only_first.claims] == [True, False]
    assert only_first.all_required_in_context is False

    both = score_evidence(case, _trace([_candidate(EP8, 2), _candidate(EP8, 3, 5, rank=2)]))
    assert both.all_required_in_context is True


def test_evidence_accepts_alternate_source_and_normalizes_paths():
    case = _case()
    trace = _trace([_candidate(EP8_PT3.replace("/", "\\"), 1), _candidate(EP8, 4, rank=2)])
    assert score_evidence(case, trace).all_required_in_context is True

    nfd_case = _case(
        expected_claims=[
            {"id": "a", "claim": "c", "evidence": [{"file_path": EP9, "scene_start": 0, "scene_end": 0}]}
        ]
    )
    windows_nfd = unicodedata.normalize("NFD", EP9).replace("/", "\\")
    assert windows_nfd != EP9.replace("/", "\\")
    assert score_evidence(nfd_case, _trace([_candidate(windows_nfd, 0)])).all_required_in_context


def test_evidence_skips_claims_verified_by_query():
    case = _case(
        expected_claims=[
            {"id": "n", "claim": "10 turns", "verified_by": "count_turns"},
            {"id": "a", "claim": "c", "evidence": [{"file_path": EP8, "scene_start": 2, "scene_end": 2}]},
        ]
    )
    score = score_evidence(case, _trace([_candidate(EP8, 2)]))
    assert [claim.claim_id for claim in score.claims] == ["a"]
    assert score.all_required_in_context is True
    only_count = _case(expected_claims=[{"id": "n", "claim": "10", "verified_by": "q"}])
    assert score_evidence(only_count, _trace([])).all_required_in_context is None


def test_evidence_separates_seen_from_context():
    case = _case()
    trace = _trace([_candidate(EP8, 2)], dense=[_candidate(EP8, 4)])
    score = score_evidence(case, trace)
    assert score.all_required_seen is True
    assert score.all_required_in_context is False


def test_agentic_context_includes_tool_results():
    case = _case()
    calls = [
        {
            "tool_name": "search",
            "candidates": [
                {"id": "e1", "source": {"file_path": EP8, "scene_start": 2, "scene_end": 2}},
                {"id": "e2", "source": {"file_path": EP8, "scene_index": 4}},
            ],
        }
    ]
    score = score_evidence(case, _trace([], mode="agentic", agent_calls=calls))
    assert score.all_required_in_context is True


def test_citation_validity_and_temporal_leakage():
    case = _case(scope={"through_file_path": EP8})
    label = "104 · Episode 8 · Part 2 · Scene 3"
    trace = _trace(
        [_candidate(EP8, 2, order=10), _candidate(EP9, 0, rank=2, order=20)],
        answer=f"Reika-san appears [{label}] and later ({label}; 104 · Episode 9 · Part 1 · Scene 1).",
        labels=[label],
    )
    score = score_evidence(case, trace, resolve_story_order=lambda path: 12)
    assert score.citations_found == [label, "104 · Episode 9 · Part 1 · Scene 1"]
    assert score.invalid_citations == ["104 · Episode 9 · Part 1 · Scene 1"]
    assert score.citation_valid is False
    assert score.temporal_leakage is True
    assert score.leaked_sources == [f"{EP9}:0-0"]


def test_extract_citations_ignores_plain_brackets_and_strips_prefix():
    assert extract_citations("See [note] and (aside).") == []
    assert extract_citations(
        "Met on the bus (CITATION: 103 · Episode 1 · Part 1 · Scene 1)."
    ) == ["103 · Episode 1 · Part 1 · Scene 1"]


def test_score_answer_rubric_rules():
    case = _case()
    passed = score_answer(case, _verdict({"a": "present", "b": "present"}))
    assert passed.passed and passed.completeness == 1.0 and passed.correct

    partial = score_answer(case, _verdict({"a": "present", "b": "partial"}))
    assert not partial.passed and partial.correct and partial.completeness == 0.75

    contradicted = score_answer(case, _verdict({"a": "contradicted", "b": "present"}))
    assert contradicted.correct is False and not contradicted.passed

    forbidden = score_answer(case, _verdict({"a": "present", "b": "present"}, forbidden=True))
    assert forbidden.correct is False and not forbidden.passed

    missing_judgment = score_answer(case, _verdict({"a": "present"}))
    assert missing_judgment.completeness == 0.5

    unsupported = score_answer(case, _verdict({"a": "present", "b": "present"}, unsupported=["x"]))
    assert unsupported.passed and unsupported.supported is False


def test_score_answer_abstention():
    unanswerable = _case(expected_claims=[], expect_abstention=True, categories=["unanswerable"])
    assert score_answer(unanswerable, _verdict({}, abstained=True)).passed
    assert not score_answer(unanswerable, _verdict({}, abstained=False)).passed
    assert not score_answer(_case(), _verdict({"a": "present", "b": "present"}, abstained=True)).passed


def test_judge_prompt_includes_reference_passages():
    case = _case()
    prompt = json.loads(
        build_judge_prompt(
            case,
            "answer text",
            reference_passages={f"{EP8}#2-2": "れいかさん"},
            system_passages=["retrieved"],
        )
    )
    first_sources = prompt["expected_claims"][0]["acceptable_sources"]
    assert first_sources[0]["passage"] == "れいかさん"
    assert first_sources[1]["passage"] == "(unavailable)"
    assert prompt["forbidden_claims"][0]["id"] == "x"
    assert prompt["system_retrieved_passages"] == ["retrieved"]


def test_usage_summary_costs_and_unpriced_models():
    calls = [
        ModelCallUsage(model_name="gemini-3.8-flash", input_tokens=1_000_000, output_tokens=100_000),
        ModelCallUsage(
            model_name="gpt-6.1-terra-2026-10-01",
            input_tokens=1_000_000,
            cache_read_tokens=500_000,
            output_tokens=0,
        ),
    ]
    summary = summarize_usage(calls)
    assert summary.requests == 2
    # Flash: 0.75 + 0.375; Terra: 0.5 uncached input + 0.025 cached.
    assert summary.estimated_cost_usd == pytest.approx(1.65)

    unpriced = summarize_usage([ModelCallUsage(model_name="mystery", input_tokens=10)])
    assert unpriced.estimated_cost_usd is None
    assert "mystery" in (unpriced.cost_unavailable_reason or "")

    merged = merge_usage([summary, unpriced])
    assert merged.requests == 3 and merged.estimated_cost_usd is None
    assert summarize_usage([]).cost_unavailable_reason is None


def test_usage_recording_model_captures_requests():
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[TextPart("ok")],
            usage=RequestUsage(input_tokens=11, output_tokens=3),
            model_name="gemini-3.8-flash",
        )

    agent = Agent(UsageRecordingModel(FunctionModel(respond)))
    with record_usage() as calls:
        agent.run_sync("hi")
    # FunctionModel reports its own name; real providers report the served model.
    assert [(call.model_name, call.input_tokens, call.output_tokens) for call in calls] == [
        ("function:respond:", 11, 3)
    ]
    agent.run_sync("outside recorder")  # no active recorder: must not raise


class _FakeEngine:
    def __init__(self, traces: dict[str, QueryTrace]):
        self.traces = traces

    def retrieve_with_trace(self, question, *, query_id, mode, answer_mode=False):
        if query_id == "boom":
            raise RuntimeError("provider down")
        return self.traces[query_id]


class _FakeJudge:
    model_label = "fake:judge"

    def __init__(self, verdicts: dict[str, JudgeVerdict]):
        self.verdicts = verdicts

    def judge(self, prompt):
        question = json.loads(prompt)["question"]
        return self.verdicts[question], [
            ModelCallUsage(model_name="gpt-6.1-sol", input_tokens=1000, output_tokens=100)
        ]


def _corpus() -> CorpusIndex:
    return CorpusIndex(
        [
            {"file_path": EP8.replace("/", "\\"), "scene_index": index, "text": f"s{index}",
             "metadata": {"story_order": 10}}
            for index in range(6)
        ]
        + [{"file_path": EP8_PT3, "scene_index": index, "text": "p3", "metadata": {"story_order": 11}}
           for index in range(2)]
    )


def _config(**overrides):
    from linkura_story_indexer.eval.answer_models import AnswerRunConfig

    data = {
        "mode": "off",
        "cases_path": "cases.json",
        "suite_version": "1",
        "generation_provider": "google",
        "generation_model": "gemini-3.8-flash",
        "embedding_model": "gemini-embedding-2",
        "answer_prompt_version": "answer-v3",
        "summary_prompt_version": "4",
        "final_top_k": 8,
        "corpus_revision": "test",
        "code_revision": "test",
        "started_at": "2026-11-07T00:00:00+00:00",
    }
    data.update(overrides)
    return AnswerRunConfig.model_validate(data)


def test_run_answer_eval_end_to_end_with_fixtures():
    good = _case(id="good", question="good?")
    weak = _case(id="weak", question="weak?", categories=["chronology"], review_status="reviewed")
    broken = _case(id="boom", question="boom?")
    cases = AnswerCaseSet(suite_version="1", cases=[good, weak, broken])
    engine = _FakeEngine(
        {
            "good": _trace([_candidate(EP8, 2), _candidate(EP8, 4, rank=2)]),
            "weak": _trace([_candidate(EP8, 4)]),
        }
    )
    judge = _FakeJudge(
        {
            "good?": _verdict({"a": "present", "b": "present"}),
            "weak?": _verdict({"a": "contradicted", "b": "present"}),
        }
    )
    run = run_answer_eval(cases, config=_config(), engine=engine, corpus=_corpus(), judge=judge)
    aggregate = run.aggregate
    assert aggregate.case_count == 3 and aggregate.judged_count == 2 and aggregate.error_count == 1
    assert aggregate.pass_rate == 0.5
    assert aggregate.evidence_vs_answer == {
        "evidence_no__answer_fail": 1,
        "evidence_yes__answer_pass": 1,
    }
    assert aggregate.pass_rate_by_category == {"alias": 1.0, "chronology": 0.0}
    assert aggregate.judge_usage.estimated_cost_usd == pytest.approx(0.006)
    assert run.results[0].trace is not None
    assert "text" not in run.results[0].trace["stages"]["final_top_k"]["candidates"][0]

    reviewed = run_answer_eval(
        cases,
        config=_config(reviewed_only=True),
        engine=engine,
        corpus=_corpus(),
        judge=judge,
    )
    assert [result.case_id for result in reviewed.results] == ["weak"]

    report = render_answer_report([run, run.model_copy(update={"config": _config(mode="agentic")})])
    assert "| Pass rate | 50% | 50% |" in report
    assert "| good | PASS / ev ✓ | PASS / ev ✓ |" in report
    assert "| boom | ERROR | ERROR |" in report
    assert "a: contradicted" in report


def test_unjudged_run_marks_answers_unavailable():
    cases = AnswerCaseSet(suite_version="1", cases=[_case()])
    engine = _FakeEngine({"c1": _trace([_candidate(EP8, 2)])})
    run = run_answer_eval(cases, config=_config(), engine=engine, corpus=_corpus(), judge=None)
    assert run.results[0].answer.unavailable_reason == "judge disabled"
    assert run.aggregate.pass_rate is None


def test_validate_case_sources_reports_missing_scenes():
    cases = AnswerCaseSet(
        suite_version="1",
        cases=[_case(scope={"through_file_path": "story/999/x.md"})],
    )
    problems = validate_case_sources(cases, _corpus())
    assert problems == ["c1: scope path not found: story/999/x.md"]
    bad = AnswerCaseSet(
        suite_version="1",
        cases=[_case(expected_claims=[{"id": "a", "claim": "c", "evidence": [
            {"file_path": EP8, "scene_start": 5, "scene_end": 7}]}])],
    )
    assert validate_case_sources(bad, _corpus()) == [
        f"c1/a: no scene 6 in {EP8}",
        f"c1/a: no scene 7 in {EP8}",
    ]


def test_checked_in_suite_is_valid():
    path = Path("eval/answer_cases.json")
    if not path.exists():
        pytest.skip("answer suite not checked in yet")
    cases = load_answer_cases(path)
    assert len(cases.cases) >= 20


def test_no_final_answer_rules():
    assert no_final_answer_reason(None) is None
    assert no_final_answer_reason({"stop_reason": "final_answer", "answer_source": "agent"}) is None
    assert no_final_answer_reason({"stop_reason": "usage_limit", "answer_source": "evidence_synthesis"}) is None
    assert "usage_limit" in (
        no_final_answer_reason({"stop_reason": "usage_limit", "answer_source": "tool_fallback"}) or ""
    )
    # Traces from before answer_source existed: any early stop had no real answer.
    assert no_final_answer_reason({"stop_reason": "model_error"}) is not None


def test_agent_without_answer_fails_without_judging():
    cases = AnswerCaseSet(suite_version="1", cases=[_case()])
    dump = _trace(
        [_candidate(EP8, 2), _candidate(EP8, 4, rank=2)],
        mode="agentic",
        answer="Evidence returned by the query tools: ...",
        agent_calls=[],
    )
    dump.stages["agent"].metadata.update({"stop_reason": "usage_limit", "answer_source": "tool_fallback"})

    class ExplodingJudge:
        model_label = "never"

        def judge(self, prompt):
            raise AssertionError("judge must not be called")

    run = run_answer_eval(
        cases,
        config=_config(mode="agentic"),
        engine=_FakeEngine({"c1": dump}),
        corpus=_corpus(),
        judge=ExplodingJudge(),
    )
    answer = run.results[0].answer
    assert answer.passed is False and answer.judged and answer.completeness == 0.0
    assert answer.failure_reason and "usage_limit" in answer.failure_reason
    assert run.aggregate.no_final_answer_count == 1 and run.aggregate.pass_rate == 0.0
    report = render_answer_report([run])
    assert "| c1 | NO ANSWER |" in report
    assert "| Agent stopped without answering | 1 |" in report


def test_rescore_fails_stored_fallback_answers(monkeypatch: pytest.MonkeyPatch):
    from linkura_story_indexer.eval import answer_runner

    cases = AnswerCaseSet(suite_version="1", cases=[_case()])
    old_trace = _trace([_candidate(EP8, 2), _candidate(EP8, 4, rank=2)], mode="agentic", agent_calls=[])
    old_trace.stages["agent"].metadata["stop_reason"] = "usage_limit"  # pre-answer_source trace
    judge = _FakeJudge({"Who is Reika-san?": _verdict({"a": "present", "b": "present"})})

    # Simulate a run scored before the no-final-answer rule existed.
    with monkeypatch.context() as patched:
        patched.setattr(answer_runner, "no_final_answer_reason", lambda agent_metadata: None)
        run = run_answer_eval(
            cases,
            config=_config(mode="agentic"),
            engine=_FakeEngine({"c1": old_trace}),
            corpus=_corpus(),
            judge=judge,
        )
    assert run.aggregate.pass_rate == 1.0

    rescored, changed, _ = rescore_run(run)
    assert changed == ["c1"]
    assert rescored.aggregate.pass_rate == 0.0
    assert rescored.aggregate.no_final_answer_count == 1
    again, changed_again, _ = rescore_run(rescored)
    assert changed_again == [] and again.aggregate == rescored.aggregate


def test_luna_pricing_cache_writes_and_long_context():
    from linkura_story_indexer.eval.usage import MODEL_PRICING, request_cost

    luna = MODEL_PRICING["gpt-6-luna"]
    normal = ModelCallUsage(
        model_name="gpt-6-luna",
        input_tokens=100_000,
        cache_read_tokens=60_000,
        cache_write_tokens=20_000,
        output_tokens=10_000,
    )
    # 20k uncached * 0.10 + 60k cached * 0.01 + 20k written * 0.125 + 10k out * 0.50 (per 1M)
    assert request_cost(normal, luna) == pytest.approx((2_000 + 600 + 2_500 + 5_000) / 1_000_000)
    long_prompt = ModelCallUsage(model_name="gpt-6-luna", input_tokens=300_000, output_tokens=10_000)
    assert request_cost(long_prompt, luna) == pytest.approx((300_000 * 0.10 * 2 + 10_000 * 0.50 * 1.5) / 1_000_000)
    # Models without a cache-write rate bill written tokens as normal input.
    flash = MODEL_PRICING["gemini-3.8-flash"]
    written = ModelCallUsage(model_name="gemini-3.8-flash", input_tokens=1_000, cache_write_tokens=500)
    assert request_cost(written, flash) == pytest.approx(1_000 * 0.75 / 1_000_000)
    summary = summarize_usage([normal])
    assert summary.by_model["gpt-6-luna"]["cache_write_tokens"] == 20_000


def test_report_labels_show_model_and_judge_only_when_they_differ():
    cases = AnswerCaseSet(suite_version="1", cases=[_case()])
    engine = _FakeEngine({"c1": _trace([_candidate(EP8, 2), _candidate(EP8, 4, rank=2)])})
    judge = _FakeJudge({"Who is Reika-san?": _verdict({"a": "present", "b": "present"})})
    flash = run_answer_eval(cases, config=_config(mode="agentic"), engine=engine, corpus=_corpus(), judge=judge)
    luna = flash.model_copy(update={"config": _config(mode="agentic", generation_model="gpt-6-luna")})
    assert run_labels([flash, flash]) == ["agentic", "agentic"]
    assert run_labels([flash, luna]) == ["agentic · gemini-3.8-flash", "agentic · gpt-6-luna"]
    argon = luna.model_copy(update={"config": _config(mode="agentic", generation_model="gpt-6-luna", judge_model="google:gemini-4-argon")})
    assert run_labels([luna, argon])[1] == "agentic · judge google:gemini-4-argon"
    report = render_answer_report([flash, luna])
    assert "| Metric | agentic · gemini-3.8-flash | agentic · gpt-6-luna |" in report
    assert "varies by column" in report


def test_rejudge_uses_stored_answers_and_corpus_passages():
    cases = AnswerCaseSet(suite_version="1", cases=[_case(), _case(id="boom", question="boom?")])
    engine = _FakeEngine({"c1": _trace([_candidate(EP8, 2), _candidate(EP8, 4, rank=2)], answer="stored answer")})
    first = _FakeJudge({"Who is Reika-san?": _verdict({"a": "present", "b": "present"})})
    run = run_answer_eval(cases, config=_config(), engine=engine, corpus=_corpus(), judge=first)
    assert run.aggregate.pass_rate == 1.0

    prompts: list[dict] = []

    class StrictJudge:
        model_label = "google:gemini-4-argon"

        def judge(self, prompt):
            prompts.append(json.loads(prompt))
            return _verdict({"a": "present", "b": "missing"}), [
                ModelCallUsage(model_name="gemini-4-argon", input_tokens=1_000, output_tokens=100)
            ]

    rejudged = rejudge_run(run, cases, _corpus(), StrictJudge())
    assert len(prompts) == 1  # the errored case is not re-judged
    assert prompts[0]["candidate_answer"] == "stored answer"
    # System passages are rebuilt from the stored spans (texts were stripped from the trace).
    assert prompts[0]["system_retrieved_passages"] == ["s2", "s4"]
    assert rejudged.config.judge_model == "google:gemini-4-argon"
    assert rejudged.aggregate.pass_rate == 0.0
    assert rejudged.results[1].error == run.results[1].error
    assert rejudged.results[0].usage == run.results[0].usage


def test_abstention_not_checked_when_expectation_is_none():
    case = _case(
        categories=["unanswerable"],
        expect_abstention=None,
        expected_claims=[
            {"id": "a", "claim": "The story never names her.", "evidence": [{"file_path": EP8, "scene_start": 2, "scene_end": 2}]}
        ],
    )
    for abstained in (True, False):
        score = score_answer(case, _verdict({"a": "present"}, abstained=abstained))
        assert score.abstention_appropriate is None and score.passed
    assert not score_answer(case, _verdict({"a": "missing"})).passed
    with pytest.raises(ValidationError, match="needs at least one expected claim"):
        _case(expected_claims=[], expect_abstention=None)


def test_rescore_with_cases_and_targeted_rejudge():
    cases = AnswerCaseSet(suite_version="1", cases=[_case(expect_abstention=True), _case(id="c2", question="two?")])
    engine = _FakeEngine(
        {
            "c1": _trace([_candidate(EP8, 2), _candidate(EP8, 4, rank=2)]),
            "c2": _trace([_candidate(EP8, 2), _candidate(EP8, 4, rank=2)]),
        }
    )
    judge = _FakeJudge(
        {
            "Who is Reika-san?": _verdict({"a": "present", "b": "present"}),
            "two?": _verdict({"a": "present", "b": "present"}),
        }
    )
    run = run_answer_eval(
        cases, config=_config(judge_model="fake:judge"), engine=engine, corpus=_corpus(), judge=judge
    )
    assert [result.answer.passed for result in run.results] == [False, True]  # c1 did not abstain

    # The reviewer relaxes c1's abstention rule and adds a claim to c2.
    c2 = _case(id="c2", question="two?", review_status="reviewed")
    c2.expected_claims.append(c2.expected_claims[0].model_copy(update={"id": "new"}))
    updated = AnswerCaseSet(suite_version="1", cases=[_case(expect_abstention=None), c2])
    rescored, changed, needs_rejudge = rescore_run(run, updated)
    assert changed == ["c1", "c2"]
    assert rescored.results[0].answer.passed is True
    assert needs_rejudge == ["c2"]
    assert rescored.results[1].review_status == "reviewed"

    calls: list[str] = []

    class CountingJudge:
        model_label = "fake:judge"

        def judge(self, prompt):
            calls.append(json.loads(prompt)["question"])
            return _verdict({"a": "present", "b": "present", "new": "present"}), []

    rejudged = rejudge_run(rescored, updated, _corpus(), CountingJudge(), case_ids=["c2"])
    assert calls == ["two?"]
    assert rejudged.results[0].answer == rescored.results[0].answer
    assert rejudged.results[1].answer.passed is True
    assert rejudged.config.judge_model == "fake:judge"


def test_report_accepts_explicit_labels():
    cases = AnswerCaseSet(suite_version="1", cases=[_case()])
    engine = _FakeEngine({"c1": _trace([_candidate(EP8, 2), _candidate(EP8, 4, rank=2)])})
    run = run_answer_eval(cases, config=_config(), engine=engine, corpus=_corpus(), judge=None)
    report = render_answer_report([run, run], ["before fix", "after fix"])
    assert "| Metric | before fix | after fix |" in report
    with pytest.raises(ValueError, match="1 labels for 2 runs"):
        render_answer_report([run, run], ["only one"])
