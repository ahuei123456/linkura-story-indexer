from pathlib import Path
from typing import Protocol

from linkura_story_indexer.eval.io import load_golden_set, write_eval_run, write_query_trace
from linkura_story_indexer.eval.metrics import aggregate_metrics, evaluate_query
from linkura_story_indexer.eval.models import EvalMode, EvalRun, GoldenSet, QueryTrace, RunConfig
from linkura_story_indexer.query.engine import RetrievalConfig, StoryQueryEngine


class TraceEngine(Protocol):
    def retrieve_with_trace(
        self,
        question: str,
        *,
        query_id: str,
        mode: EvalMode,
        answer_mode: bool = False,
    ) -> QueryTrace: ...


def run_eval(
    golden_set: GoldenSet,
    *,
    golden_set_path: str,
    mode: EvalMode = "raw",
    engine: TraceEngine | None = None,
    answer_mode: bool = False,
) -> EvalRun:
    if engine is None:
        engine = StoryQueryEngine(
            retrieval_config=RetrievalConfig(enable_query_analysis=mode == "raw-analyze")
        )

    traces = [
        engine.retrieve_with_trace(
            question.question,
            query_id=question.id,
            mode=mode,
            answer_mode=answer_mode,
        )
        for question in golden_set.questions
    ]
    query_metrics = [
        evaluate_query(trace, question)
        for trace, question in zip(traces, golden_set.questions, strict=True)
    ]
    return EvalRun(
        config=RunConfig(
            mode=mode,
            golden_set=golden_set_path,
            answer_mode=answer_mode,
            reranker_enabled=mode == "raw-rerank",
        ),
        aggregate_metrics=aggregate_metrics(query_metrics, traces),
        query_metrics=query_metrics,
        traces=traces,
    )


def run_eval_from_file(
    golden_set_path: str | Path,
    *,
    mode: EvalMode = "raw",
    output_path: str | Path | None = None,
    inspect_query_id: str | None = None,
    dump_traces_dir: str | Path | None = None,
    answer_mode: bool = False,
) -> EvalRun:
    golden_set = load_golden_set(golden_set_path)
    run = run_eval(
        golden_set,
        golden_set_path=str(golden_set_path),
        mode=mode,
        answer_mode=answer_mode,
    )
    if output_path is not None:
        write_eval_run(run, output_path)
    if dump_traces_dir is not None:
        directory = Path(dump_traces_dir)
        for trace in run.traces:
            write_query_trace(trace, directory / f"{trace.query_id}.json")
    if inspect_query_id is not None and not any(
        trace.query_id == inspect_query_id for trace in run.traces
    ):
        raise ValueError(f"Unknown query id: {inspect_query_id}")
    return run
