"""Runs answer cases against the query engine, judges them, and renders reports."""

import hashlib
import json
import re
import subprocess
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic_ai import Agent

from linkura_story_indexer.eval.answer_models import (
    AnswerCase,
    AnswerCaseSet,
    AnswerEvalRun,
    AnswerRunConfig,
    AnswerScore,
    CaseResult,
    JudgeVerdict,
    ModelCallUsage,
)
from linkura_story_indexer.eval.answer_scoring import (
    JUDGE_INSTRUCTIONS,
    StoryOrderResolver,
    aggregate_results,
    build_judge_prompt,
    no_final_answer_reason,
    no_final_answer_score,
    normalize_path,
    score_answer,
    score_evidence,
)
from linkura_story_indexer.eval.io import stable_json
from linkura_story_indexer.eval.models import EvalMode, QueryTrace
from linkura_story_indexer.eval.usage import UsageRecordingModel, record_usage, summarize_usage

DEFAULT_JUDGE_MODEL = "openai:gpt-6.1-sol"
YEAR_ARCS = ("102", "103", "104", "105")


class AnswerEngine(Protocol):
    def retrieve_with_trace(
        self,
        question: str,
        *,
        query_id: str,
        mode: EvalMode,
        answer_mode: bool = False,
    ) -> QueryTrace: ...


class Judge(Protocol):
    model_label: str

    def judge(self, prompt: str) -> tuple[JudgeVerdict, list[ModelCallUsage]]: ...


def load_answer_cases(path: str | Path) -> AnswerCaseSet:
    return AnswerCaseSet.model_validate_json(Path(path).read_text(encoding="utf-8"))


def load_answer_run(path: str | Path) -> AnswerEvalRun:
    return AnswerEvalRun.model_validate_json(Path(path).read_text(encoding="utf-8"))


class CorpusIndex:
    """Scene text and story order keyed by normalized file path."""

    def __init__(self, scenes: Sequence[dict[str, Any]]):
        self._text: dict[tuple[str, int], str] = {}
        self._max_order: dict[str, int] = {}
        for scene in scenes:
            path = normalize_path(str(scene["file_path"]))
            index = int(scene["scene_index"])
            self._text[(path, index)] = str(scene["text"])
            metadata = scene.get("metadata") or {}
            order = metadata.get("story_order", metadata.get("canonical_story_order"))
            if isinstance(order, int):
                self._max_order[path] = max(order, self._max_order.get(path, 0))

    @classmethod
    def from_source_store(cls) -> "CorpusIndex":
        from linkura_story_indexer.indexer.source_store import SourceRecordStore

        return cls(SourceRecordStore().iter_scenes())

    def passage(self, file_path: str, scene_start: int, scene_end: int) -> str | None:
        path = normalize_path(file_path)
        parts = [
            self._text[(path, index)]
            for index in range(scene_start, scene_end + 1)
            if (path, index) in self._text
        ]
        return "\n\n".join(parts) if parts else None

    def files(self) -> list[dict[str, Any]]:
        """Every Part file with its scene count, in story order."""
        counts: dict[str, int] = {}
        for path, index in self._text:
            counts[path] = max(counts.get(path, 0), index + 1)
        return [
            {"file_path": path, "scene_count": counts[path]}
            for path in sorted(counts, key=lambda item: (self._max_order.get(item, 0), item))
        ]

    def search(
        self, pattern: str, *, path_filter: str = "", limit: int = 200
    ) -> list[dict[str, Any]]:
        """Regex search over scene lines; returns file, scene index and the matching line."""
        regex = re.compile(pattern)
        path_filter = normalize_path(path_filter) if path_filter else ""
        hits: list[dict[str, Any]] = []
        for (path, index), text in sorted(
            self._text.items(), key=lambda item: (self._max_order.get(item[0][0], 0), item[0])
        ):
            if path_filter and path_filter not in path:
                continue
            for line in text.splitlines():
                if regex.search(line):
                    hits.append({"file_path": path, "scene_index": index, "line": line.strip()})
                    if len(hits) >= limit:
                        return hits
        return hits

    def has_scene(self, file_path: str, scene_index: int) -> bool:
        return (normalize_path(file_path), scene_index) in self._text

    def max_story_order(self, file_path: str) -> int | None:
        return self._max_order.get(normalize_path(file_path))


def validate_case_sources(cases: AnswerCaseSet, corpus: CorpusIndex) -> list[str]:
    """Returns problems with evidence references that do not resolve to corpus scenes."""
    problems = []
    for case in cases.cases:
        for claim in case.expected_claims:
            for ref in claim.evidence:
                for index in range(ref.scene_start, ref.scene_end + 1):
                    if not corpus.has_scene(ref.file_path, index):
                        problems.append(
                            f"{case.id}/{claim.id}: no scene {index} in {ref.file_path}"
                        )
        if case.scope and case.scope.through_file_path:
            if corpus.max_story_order(case.scope.through_file_path) is None:
                problems.append(f"{case.id}: scope path not found: {case.scope.through_file_path}")
    return problems


class ModelJudge:
    """LLM judge using structured output; defaults to a different provider than generation."""

    def __init__(self, model_spec: str = DEFAULT_JUDGE_MODEL, *, model: Any | None = None):
        self.model_label = model_spec
        self._model = model
        self._model_spec = model_spec

    def _resolve_model(self) -> Any:
        if self._model is not None:
            return self._model
        provider, _, model_name = self._model_spec.partition(":")
        if not model_name:
            raise ValueError(f"Judge model must look like provider:model, got {self._model_spec!r}")
        from linkura_story_indexer import database

        if provider == "openai":
            self._model = database.create_openai_responses_model(model_name)
        elif provider == "google":
            self._model = database.create_google_model(model_name)
        else:
            raise ValueError(f"Unsupported judge provider {provider!r}")
        return self._model

    def judge(self, prompt: str) -> tuple[JudgeVerdict, list[ModelCallUsage]]:
        agent: Agent[None, JudgeVerdict] = Agent(
            UsageRecordingModel(self._resolve_model()),
            instructions=JUDGE_INSTRUCTIONS,
            output_type=JudgeVerdict,
        )
        with record_usage() as calls:
            result = agent.run_sync(prompt)
        return result.output, list(calls)


def _system_passages(trace: QueryTrace, limit: int = 16) -> list[str]:
    passages: list[str] = []
    final_stage = trace.stages.get("final_top_k")
    for candidate in (final_stage.candidates or []) if final_stage else []:
        if candidate.text:
            passages.append(candidate.text)
    agent_stage = trace.stages.get("agent")
    if agent_stage is not None:
        for call in agent_stage.metadata.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            for candidate in call.get("candidates") or []:
                text = candidate.get("text") if isinstance(candidate, dict) else None
                if isinstance(text, str) and text not in passages:
                    passages.append(text)
            output = call.get("tool_output")
            if output is not None:
                passages.append(json.dumps(output, ensure_ascii=False, default=str))
    return passages[:limit]


def _reference_passages(case: AnswerCase, corpus: CorpusIndex) -> dict[str, str]:
    passages = {}
    for claim in case.expected_claims:
        for ref in claim.evidence:
            key = f"{normalize_path(ref.file_path)}#{ref.scene_start}-{ref.scene_end}"
            text = corpus.passage(ref.file_path, ref.scene_start, ref.scene_end)
            if text is not None:
                passages[key] = text
    return passages


def _compact_trace(trace: QueryTrace) -> dict[str, Any]:
    """Trace without candidate texts, which the run file does not need to keep."""
    data = trace.model_dump(mode="json")
    for stage in data.get("stages", {}).values():
        for candidate in stage.get("candidates") or []:
            candidate.pop("text", None)
        for call in stage.get("metadata", {}).get("tool_calls") or []:
            for candidate in call.get("candidates") or []:
                if isinstance(candidate, dict):
                    candidate.pop("text", None)
    return data


def year_summary_labels() -> list[str]:
    return [
        f"{arc} · Main · Episode ALL_EPISODES · Part ALL_PARTS · summary_level 1"
        for arc in YEAR_ARCS
    ]


def run_case(
    case: AnswerCase,
    *,
    mode: EvalMode,
    engine: AnswerEngine,
    corpus: CorpusIndex,
    judge: Judge | None,
    resolve_story_order: StoryOrderResolver | None = None,
) -> CaseResult:
    result = CaseResult(
        case_id=case.id,
        categories=case.categories,
        review_status=case.review_status,
        mode=mode,
    )
    started = time.perf_counter()
    try:
        with record_usage() as calls:
            trace = engine.retrieve_with_trace(
                case.question, query_id=case.id, mode=mode, answer_mode=True
            )
    except Exception as exc:  # noqa: BLE001 - a failed case is recorded, not fatal to the run
        result.latency_seconds = round(time.perf_counter() - started, 3)
        result.error = f"{type(exc).__name__}: {exc}"
        result.answer = AnswerScore(unavailable_reason="system run failed")
        return result
    result.latency_seconds = round(time.perf_counter() - started, 3)
    result.usage = summarize_usage(calls)
    result.answer_text = trace.answer_text
    result.trace = _compact_trace(trace)
    result.evidence = score_evidence(
        case,
        trace,
        resolve_story_order=resolve_story_order or corpus.max_story_order,
        allowed_extra_citations=year_summary_labels(),
    )

    agent_stage = trace.stages.get("agent")
    if reason := no_final_answer_reason(agent_stage.metadata if agent_stage else None):
        result.answer = no_final_answer_score(reason)
        return result
    if judge is None:
        result.answer = AnswerScore(unavailable_reason="judge disabled")
        return result
    if not trace.answer_text:
        result.answer = AnswerScore(unavailable_reason="no answer text produced")
        return result
    prompt = build_judge_prompt(
        case,
        trace.answer_text,
        reference_passages=_reference_passages(case, corpus),
        system_passages=_system_passages(trace),
    )
    try:
        verdict, judge_calls = judge.judge(prompt)
    except Exception as exc:  # noqa: BLE001
        result.answer = AnswerScore(unavailable_reason=f"judge failed: {type(exc).__name__}: {exc}")
        return result
    result.judge_usage = summarize_usage(judge_calls)
    result.answer = score_answer(case, verdict)
    return result


def _git_revision() -> str:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no", "--", "src"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return f"{revision}{'-dirty' if dirty else ''}"


def corpus_revision(
    story_root: str | Path = "story",
    summary_cache: str | Path = "summaries_cache.json",
) -> str:
    """Content hash of the story files plus the summary cache the engine loads."""
    digest = hashlib.sha256()
    root = Path(story_root)
    if not root.exists():
        return "unavailable"
    for path in sorted(root.rglob("*.md"), key=lambda item: item.as_posix()):
        digest.update(normalize_path(path.as_posix()).encode("utf-8"))
        digest.update(path.read_bytes())
    story_hash = digest.hexdigest()[:12]
    cache = Path(summary_cache)
    cache_hash = (
        hashlib.sha256(cache.read_bytes()).hexdigest()[:12] if cache.exists() else "none"
    )
    return f"story:{story_hash} summaries:{cache_hash}"


def build_run_config(
    *,
    mode: EvalMode,
    cases_path: str,
    cases: AnswerCaseSet,
    judge: Judge | None,
    final_top_k: int,
    reviewed_only: bool,
) -> AnswerRunConfig:
    from linkura_story_indexer import database
    from linkura_story_indexer.indexer.summarizer import SUMMARIZATION_PROMPT_VERSION
    from linkura_story_indexer.prompts import PROMPT_VERSION
    from linkura_story_indexer.query.agent import _configured_request_limit

    return AnswerRunConfig(
        mode=mode,
        cases_path=cases_path,
        suite_version=cases.suite_version,
        generation_provider=database.get_generation_provider_name(),
        generation_model=database.get_generation_model_name(),
        router_model=database.get_router_model_name() if mode == "llm_router" else None,
        embedding_model=database.get_embedding_model_name(),
        answer_prompt_version=PROMPT_VERSION,
        summary_prompt_version=SUMMARIZATION_PROMPT_VERSION,
        judge_model=judge.model_label if judge else None,
        agent_request_limit=_configured_request_limit() if mode == "agentic" else None,
        final_top_k=final_top_k,
        corpus_revision=corpus_revision(),
        code_revision=_git_revision(),
        started_at=datetime.now(UTC).isoformat(timespec="seconds"),
        reviewed_only=reviewed_only,
    )


def run_answer_eval(
    cases: AnswerCaseSet,
    *,
    config: AnswerRunConfig,
    engine: AnswerEngine,
    corpus: CorpusIndex,
    judge: Judge | None,
    case_ids: Sequence[str] | None = None,
    on_result: Callable[[CaseResult], None] | None = None,
) -> AnswerEvalRun:
    selected = [
        case
        for case in cases.cases
        if (case_ids is None or case.id in case_ids)
        and (not config.reviewed_only or case.review_status == "reviewed")
    ]
    results = []
    for case in selected:
        result = run_case(case, mode=config.mode, engine=engine, corpus=corpus, judge=judge)
        results.append(result)
        if on_result is not None:
            on_result(result)
    return AnswerEvalRun(config=config, aggregate=aggregate_results(results), results=results)


def _passages_from_stored_trace(trace: dict[str, Any] | None, corpus: CorpusIndex) -> list[str]:
    """Rebuilds the system's raw evidence from a compact trace's source spans.

    Stored traces drop candidate text, so passages are read back from the corpus. Summary
    candidates have no span and are omitted.
    """
    spans: list[tuple[str, int, int]] = []
    stages = (trace or {}).get("stages") or {}
    for candidate in (stages.get("final_top_k") or {}).get("candidates") or []:
        source = candidate.get("source_span") or {}
        if source.get("file_path") is not None:
            spans.append((source["file_path"], source["scene_start"], source["scene_end"]))
    for call in ((stages.get("agent") or {}).get("metadata") or {}).get("tool_calls") or []:
        for candidate in call.get("candidates") or []:
            source = candidate.get("source") or {}
            start = source.get("scene_start", source.get("scene_index"))
            if source.get("file_path") is not None and isinstance(start, int):
                spans.append((source["file_path"], start, source.get("scene_end", start)))
    passages: list[str] = []
    seen: set[tuple[str, int, int]] = set()
    for path, start, end in spans:
        key = (normalize_path(path), start, end)
        if key in seen:
            continue
        seen.add(key)
        text = corpus.passage(path, start, end)
        if text:
            passages.append(text)
    return passages[:16]


def rejudge_run(
    run: AnswerEvalRun,
    cases: AnswerCaseSet,
    corpus: CorpusIndex,
    judge: Judge,
    *,
    case_ids: Sequence[str] | None = None,
    on_result: Callable[[CaseResult], None] | None = None,
) -> AnswerEvalRun:
    """Judges stored answers again; system results are left untouched.

    With ``case_ids``, only those cases are re-judged and the rest keep their scores.
    """
    by_id = {case.id: case for case in cases.cases}
    results = []
    for result in run.results:
        updated = result.model_copy(deep=True)
        case = by_id.get(result.case_id)
        if case_ids is not None and result.case_id not in case_ids:
            results.append(updated)
            continue
        if case is None:
            updated.answer = AnswerScore(unavailable_reason="case no longer in the suite")
        elif result.error or result.answer.failure_reason or not result.answer_text:
            pass  # deterministic outcomes do not depend on the judge
        else:
            prompt = build_judge_prompt(
                case,
                result.answer_text,
                reference_passages=_reference_passages(case, corpus),
                system_passages=_passages_from_stored_trace(result.trace, corpus),
            )
            try:
                verdict, calls = judge.judge(prompt)
                updated.judge_usage = summarize_usage(calls)
                updated.answer = score_answer(case, verdict)
            except Exception as exc:  # noqa: BLE001
                updated.answer = AnswerScore(
                    unavailable_reason=f"judge failed: {type(exc).__name__}: {exc}"
                )
        results.append(updated)
        if on_result is not None:
            on_result(updated)
    config = run.config
    if case_ids is None or judge.model_label != run.config.judge_model:
        config = run.config.model_copy(update={"judge_model": judge.model_label})
    return AnswerEvalRun(config=config, aggregate=aggregate_results(results), results=results)


def rescore_run(
    run: AnswerEvalRun,
    cases: AnswerCaseSet | None = None,
) -> tuple[AnswerEvalRun, list[str], list[str]]:
    """Reapplies scoring rules to stored results without new model calls.

    With ``cases``, judged answers are re-scored from their stored verdicts against the current
    case definitions (rubric-level changes such as abstention expectations). Cases whose
    expected claims the stored verdict never judged are returned as needing a re-judge.

    Returns the updated run, the ids whose score changed, and the ids that need a re-judge.
    """
    by_id = {case.id: case for case in cases.cases} if cases else {}
    changed: list[str] = []
    needs_rejudge: list[str] = []
    results = []
    for result in run.results:
        updated = result.model_copy(deep=True)
        stages = (result.trace or {}).get("stages") or {}
        agent_metadata = (stages.get("agent") or {}).get("metadata")
        reason = no_final_answer_reason(agent_metadata) if not result.error else None
        case = by_id.get(result.case_id)
        verdict = result.answer.verdict
        if reason:
            if updated.answer.failure_reason != reason:
                updated.answer = no_final_answer_score(reason)
        elif case is not None and verdict is not None:
            judged_ids = {judgment.claim_id for judgment in verdict.claims}
            if any(claim.id not in judged_ids for claim in case.expected_claims):
                needs_rejudge.append(result.case_id)
            updated.answer = score_answer(case, verdict)
        updated.categories = case.categories if case is not None else updated.categories
        updated.review_status = case.review_status if case is not None else updated.review_status
        if updated.answer != result.answer:
            changed.append(result.case_id)
        results.append(updated)
    return (
        run.model_copy(update={"results": results, "aggregate": aggregate_results(results)}),
        changed,
        needs_rejudge,
    )


def write_answer_run(run: AnswerEvalRun, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(stable_json(run), encoding="utf-8")


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


def _cost(summary: Any) -> str:
    if summary.estimated_cost_usd is not None:
        return f"${summary.estimated_cost_usd:.4f}"
    if summary.requests == 0:
        return "n/a"
    return "unavailable"


def _case_cell(result: CaseResult) -> str:
    if result.error:
        return "ERROR"
    answer = result.answer
    if answer.failure_reason:
        return "NO ANSWER"
    if not answer.judged:
        status = "unjudged"
    elif answer.passed:
        status = "PASS"
    elif answer.correct is False:
        status = "WRONG"
    else:
        status = f"partial {_pct(answer.completeness)}"
    evidence = result.evidence.all_required_in_context
    marker = "" if evidence is None else (" / ev ✓" if evidence else " / ev ✗")
    return status + marker


def _varying(runs: Sequence[AnswerEvalRun], value: Callable[[AnswerRunConfig], str]) -> str:
    values = list(dict.fromkeys(value(run.config) for run in runs))
    return values[0] if len(values) == 1 else "varies by column (" + ", ".join(values) + ")"


def run_labels(runs: Sequence[AnswerEvalRun]) -> list[str]:
    """Column labels: the mode, plus model and judge when those differ between runs."""
    models = {run.config.generation_model for run in runs}
    judges = {run.config.judge_model for run in runs}
    labels: list[str] = []
    for run in runs:
        parts: list[str] = [run.config.mode]
        if len(models) > 1:
            parts.append(run.config.generation_model)
        if len(judges) > 1:
            parts.append(f"judge {run.config.judge_model or 'off'}")
        labels.append(" · ".join(parts))
    return labels


def render_answer_report(
    runs: Sequence[AnswerEvalRun],
    labels: Sequence[str] | None = None,
) -> str:
    if not runs:
        return "# Answer benchmark\n\nNo runs.\n"
    if labels is not None and len(labels) != len(runs):
        raise ValueError(f"got {len(labels)} labels for {len(runs)} runs")
    first = runs[0].config
    labels = list(labels) if labels is not None else run_labels(runs)
    lines = [
        "# Answer benchmark report",
        "",
        f"- Suite: `{first.cases_path}` version {first.suite_version}",
        f"- Generation: {_varying(runs, lambda c: f'{c.generation_provider}:{c.generation_model}')}; "
        f"answer prompt {first.answer_prompt_version}; summary prompt {first.summary_prompt_version}",
        f"- Judge: {_varying(runs, lambda c: c.judge_model or 'disabled')}",
        f"- Corpus: {first.corpus_revision}; code {first.code_revision}",
        "- Costs are estimates from configured list prices, not provider invoices. "
        "Embedding calls are not metered.",
        "",
        "## Summary by mode",
        "",
        "| Metric | " + " | ".join(labels) + " |",
        "|---|" + "---|" * len(runs),
    ]
    rows: list[tuple[str, Callable[[AnswerEvalRun], str]]] = [
        ("Cases (judged)", lambda run: f"{run.aggregate.case_count} ({run.aggregate.judged_count})"),
        ("Pass rate", lambda run: _pct(run.aggregate.pass_rate)),
        ("Mean completeness", lambda run: _pct(run.aggregate.mean_completeness)),
        ("No contradictions/forbidden claims", lambda run: _pct(run.aggregate.correct_rate)),
        ("No unsupported statements", lambda run: _pct(run.aggregate.supported_rate)),
        ("Abstention appropriate", lambda run: _pct(run.aggregate.abstention_accuracy)),
        ("Required evidence seen", lambda run: _pct(run.aggregate.evidence_seen_rate)),
        ("Required evidence in answer context", lambda run: _pct(run.aggregate.evidence_in_context_rate)),
        ("Citations valid", lambda run: _pct(run.aggregate.citation_valid_rate)),
        ("Temporal leakage", lambda run: _pct(run.aggregate.temporal_leakage_rate)),
        ("Errors", lambda run: str(run.aggregate.error_count)),
        ("Agent stopped without answering", lambda run: str(run.aggregate.no_final_answer_count)),
        (
            "Mean latency",
            lambda run: "n/a"
            if run.aggregate.mean_latency_seconds is None
            else f"{run.aggregate.mean_latency_seconds:.1f}s",
        ),
        ("Model requests", lambda run: str(run.aggregate.usage.requests)),
        (
            "Tokens in/out",
            lambda run: f"{run.aggregate.usage.input_tokens:,}/{run.aggregate.usage.output_tokens:,}",
        ),
        ("Est. system cost", lambda run: _cost(run.aggregate.usage)),
        ("Est. judge cost", lambda run: _cost(run.aggregate.judge_usage)),
    ]
    for label, render in rows:
        lines.append(f"| {label} | " + " | ".join(render(run) for run in runs) + " |")

    lines += [
        "",
        "## Evidence vs answer",
        "",
        "Whether every required passage reached the answer context, against whether the answer passed.",
        "",
        "| Mode | evidence ✓ pass | evidence ✓ fail | evidence ✗ pass | evidence ✗ fail |",
        "|---|---|---|---|---|",
    ]
    for label, run in zip(labels, runs, strict=True):
        counts = run.aggregate.evidence_vs_answer
        lines.append(
            f"| {label} | {counts.get('evidence_yes__answer_pass', 0)} | "
            f"{counts.get('evidence_yes__answer_fail', 0)} | "
            f"{counts.get('evidence_no__answer_pass', 0)} | "
            f"{counts.get('evidence_no__answer_fail', 0)} |"
        )

    categories = sorted({cat for run in runs for cat in run.aggregate.pass_rate_by_category})
    if categories:
        lines += [
            "",
            "## Pass rate by category",
            "",
            "| Category | " + " | ".join(labels) + " |",
            "|---|" + "---|" * len(runs),
        ]
        for category in categories:
            lines.append(
                f"| {category} | "
                + " | ".join(_pct(run.aggregate.pass_rate_by_category.get(category)) for run in runs)
                + " |"
            )

    case_ids: list[str] = []
    for run in runs:
        for result in run.results:
            if result.case_id not in case_ids:
                case_ids.append(result.case_id)
    by_mode = [{result.case_id: result for result in run.results} for run in runs]
    lines += [
        "",
        "## Cases",
        "",
        "| Case | " + " | ".join(labels) + " |",
        "|---|" + "---|" * len(runs),
    ]
    for case_id in case_ids:
        cells = [_case_cell(results[case_id]) if case_id in results else "—" for results in by_mode]
        lines.append(f"| {case_id} | " + " | ".join(cells) + " |")

    failures = [
        (label, result)
        for label, run in zip(labels, runs, strict=True)
        for result in run.results
        if result.error or (result.answer.judged and not result.answer.passed)
    ]
    if failures:
        lines += ["", "## Failure notes", ""]
        for mode, result in failures:
            lines.append(f"### {result.case_id} ({mode})")
            if result.error:
                lines.append(f"- Error: {result.error}")
            if result.answer.failure_reason:
                lines.append(f"- {result.answer.failure_reason}")
            verdict = result.answer.verdict
            if verdict is not None:
                for judgment in verdict.claims:
                    if judgment.verdict != "present":
                        lines.append(
                            f"- {judgment.claim_id}: {judgment.verdict} — {judgment.rationale}"
                        )
                for judgment in verdict.forbidden:
                    if judgment.asserted:
                        lines.append(f"- forbidden {judgment.claim_id} asserted — {judgment.rationale}")
                for statement in verdict.unsupported_statements:
                    lines.append(f"- unsupported: {statement}")
                if not result.answer.abstention_appropriate:
                    lines.append(f"- abstention mismatch (abstained={verdict.abstained})")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"
