import json
import os
from pathlib import Path
from typing import Any, cast

import typer
from rich.console import Console
from rich.progress import Progress

from .database import (
    RETRIEVAL_DOCUMENT,
    EmbeddingDocument,
    embed_texts,
    get_chat_model_name,
    get_chroma_collection,
    get_embedding_model_name,
    get_generation_model_name,
    get_generation_provider_name,
    initialize_generation_settings,
    initialize_ingest_settings,
    initialize_query_settings,
)
from .eval.io import load_eval_run, stable_json
from .eval.metrics import diff_runs
from .eval.models import ROUTING_MODES, EvalMode
from .eval.runner import run_eval_from_file
from .glossary_candidates import (
    DEFAULT_GLOSSARY_CANDIDATE_FILE,
    category_counts,
    extract_glossary_candidates,
)
from .indexer.chunker import (
    CHUNKER_VERSION,
    MAX_CHUNK_CHARS,
    MIN_USEFUL_CHARS,
    TARGET_CHUNK_CHARS,
    build_retrieval_chunks,
)
from .indexer.extractor import StateExtractor
from .indexer.manifest import (
    RAW_EVIDENCE_SCHEMA_VERSION,
    SUMMARY_CACHE_SCHEMA_VERSION,
    ChunkerConfig,
    IngestionManifest,
    SummaryCacheContext,
    VectorIds,
    hash_files,
    hash_json_file,
    utc_timestamp,
    write_manifest,
)
from .indexer.parser import PARSER_VERSION
from .indexer.processor import StoryProcessor, episode_number_from_names
from .indexer.source_store import SourceRecordStore
from .indexer.summarizer import SUMMARIZATION_PROMPT_VERSION, HierarchicalSummarizer
from .lexical import LexicalIndex, get_lexical_db_path, glossary_alias_groups
from .models.story import StoryNode
from .query.engine import INSUFFICIENT_SOURCE_CONTEXT, RetrievalConfig, StoryQueryEngine
from .story_order import StoryOrder, default_story_order, load_story_order
from .summary_export import (
    DEFAULT_PRODUCTION_READER_SOURCE,
    DEFAULT_READER_TEMPLATE,
    export_production_summary_reader,
    export_summary_reader,
)

app = typer.Typer()
eval_app = typer.Typer(help="Run and compare retrieval evaluation harness outputs.")
app.add_typer(eval_app, name="eval")
console = Console()


def _router_debug_lines(metadata: dict[str, Any]) -> list[str]:
    fields = [
        ("model", metadata.get("router_model")),
        ("chosen_tool", metadata.get("chosen_tool")),
        (
            "args",
            json.dumps(metadata.get("validated_args", {}), ensure_ascii=False, sort_keys=True),
        ),
        ("fallback_used", metadata.get("fallback_used")),
        ("fallback_reason", metadata.get("fallback_reason")),
    ]
    lines = ["[bold cyan]Router:[/bold cyan]"]
    lines.extend(f"  {name}: {value}" for name, value in fields)
    validation_errors = metadata.get("validation_errors")
    if validation_errors:
        lines.append(
            "  validation_errors: "
            + json.dumps(validation_errors, ensure_ascii=False, sort_keys=True)
        )
    raw_output = metadata.get("raw_structured_model_output")
    if raw_output is not None:
        lines.append(
            "  raw_output: " + json.dumps(raw_output, ensure_ascii=False, sort_keys=True)
        )
    return lines


def _print_router_debug(engine: StoryQueryEngine, question: str) -> str:
    trace = engine.retrieve_with_trace(question, answer_mode=True)
    router_stage = trace.stages.get("router")
    if router_stage is not None:
        _print_router_metadata(router_stage.metadata)
    else:
        agent_stage = trace.stages.get("agent")
        if agent_stage is not None:
            _print_agent_metadata(agent_stage.metadata)
    return trace.answer_text or INSUFFICIENT_SOURCE_CONTEXT


def _print_router_metadata(metadata: dict[str, Any]) -> None:
    for line in _router_debug_lines(metadata):
        console.print(line)


def _agent_debug_lines(metadata: dict[str, Any]) -> list[str]:
    lines = ["[bold cyan]Agent:[/bold cyan]"]
    for name in ("model", "request_count", "stop_reason", "fallback_reason"):
        lines.append(f"  {name}: {metadata.get(name)}")
    cited_labels = metadata.get("model_cited_labels")
    if cited_labels:
        lines.append(
            "  model_cited_labels: "
            + json.dumps(cited_labels, ensure_ascii=False, sort_keys=True)
        )
    tool_calls = metadata.get("tool_calls")
    if tool_calls:
        lines.append(
            "  tool_calls: " + json.dumps(tool_calls, ensure_ascii=False, sort_keys=True)
        )
    return lines


def _print_agent_metadata(metadata: dict[str, Any]) -> None:
    for line in _agent_debug_lines(metadata):
        console.print(line)


def _node_id(node: StoryNode) -> str:
    meta = node.metadata
    if node.summary_level == 4:
        if meta.chunk_id:
            return meta.chunk_id
        return f"chunk:{meta.parent_part_id}:{meta.scene_start}-{meta.scene_end}"
    if node.summary_level == 3:
        return f"summary:part:{meta.parent_part_id}"
    if node.summary_level == 2:
        return f"summary:episode:{meta.parent_episode_id}"
    if node.summary_level == 1:
        return f"summary:year:{meta.parent_year_id}"
    return f"level:{node.summary_level}:{meta.parent_part_id}:{meta.scene_index}"


def _story_order_key(node: StoryNode, story_order: StoryOrder | None = None) -> tuple:
    order_config = story_order or default_story_order()
    return order_config.chronological_node_key(node)


def _assign_canonical_story_order(
    nodes: list[StoryNode],
    story_order: StoryOrder | None = None,
) -> None:
    order_config = story_order or default_story_order()
    for order, node in enumerate(sorted(nodes, key=order_config.chronological_node_key), start=1):
        node.metadata.canonical_story_order = order
        node.metadata.story_order = order


def _translation_aliases(node: StoryNode, glossary: dict | None) -> list[str]:
    if not glossary:
        return []

    aliases = []
    seen = set()
    searchable_text = "\n".join([node.text, *node.metadata.detected_speakers])
    for group in glossary_alias_groups(glossary):
        if len(group) < 2:
            continue
        english = group[1]
        if any(alias in searchable_text for alias in group) and english not in seen:
            aliases.append(english)
            seen.add(english)
    return aliases


def _human_scene_span(node: StoryNode) -> str:
    meta = node.metadata
    if meta.scene_start == meta.scene_end:
        return str(meta.scene_start + 1)
    return f"{meta.scene_start + 1}-{meta.scene_end + 1}"


def _embedding_document_title(node: StoryNode) -> str:
    meta = node.metadata
    location = [
        meta.arc_id,
        meta.story_type,
        meta.episode_name,
        f"Part {meta.part_name}",
    ]
    if node.summary_level == 4:
        location.append(f"Scene {_human_scene_span(node)}")
    else:
        location.append(f"Level {node.summary_level} summary")
    return " | ".join(str(part) for part in location if part)


def _summary_tier(node: StoryNode) -> str:
    if node.summary_level == 1:
        return "Year"
    if node.summary_level == 2:
        return "Episode"
    if node.summary_level == 3:
        return "Part"
    return f"Level {node.summary_level}"


def _summary_location_header(node: StoryNode) -> str:
    meta = node.metadata
    episode_name = meta.episode_name if node.summary_level in {2, 3} else "ALL_EPISODES"
    part_name = meta.part_name if node.summary_level == 3 else "ALL_PARTS"
    return "\n".join(
        [
            f"Year: {meta.arc_id}",
            f"Story type: {meta.story_type}",
            f"Episode: {episode_name}",
            f"Part: {part_name}",
            f"Summary level: {node.summary_level}",
            f"Summary tier: {_summary_tier(node)}",
            "",
        ]
    )


def _embedding_document(node: StoryNode, glossary: dict | None = None) -> EmbeddingDocument:
    if node.summary_level != 4:
        return EmbeddingDocument(
            text=f"{_summary_location_header(node)}{node.text}",
            title=_embedding_document_title(node),
        )

    meta = node.metadata
    speakers = ", ".join(meta.detected_speakers) if meta.detected_speakers else "none"
    aliases = ", ".join(_translation_aliases(node, glossary)) or "none"
    header = "\n".join(
        [
            f"Year: {meta.arc_id}",
            f"Story type: {meta.story_type}",
            f"Episode: {meta.episode_name}",
            f"Part: {meta.part_name}",
            f"Scene span: {_human_scene_span(node)}",
            f"Source scene index span: {meta.scene_start}-{meta.scene_end}",
            f"Source scene count: {meta.source_scene_count}",
            f"Canonical story order: {meta.canonical_story_order}",
            f"Speakers: {speakers}",
            f"Aliases: {aliases}",
            "",
        ]
    )
    return EmbeddingDocument(text=f"{header}{node.text}", title=_embedding_document_title(node))


def _lexical_document(node: StoryNode, glossary: dict | None = None) -> str:
    embedding_document = _embedding_document(node, glossary)
    if node.summary_level != 4:
        return embedding_document.text
    return embedding_document.text


def _metadata_for_node(node: StoryNode) -> dict:
    metadata = node.metadata.model_dump()
    if not metadata.get("story_order"):
        metadata["story_order"] = metadata.get("canonical_story_order", 0)
    if not metadata.get("episode_number"):
        metadata["episode_number"] = episode_number_from_names(
            node.metadata.episode_name,
            node.metadata.part_name,
        )
    metadata["detected_speakers"] = "|".join(node.metadata.detected_speakers)
    metadata["speakers"] = "|".join(node.metadata.speakers)
    metadata["source_scene_ids"] = "|".join(node.metadata.source_scene_ids)
    metadata["source_turn_ids"] = "|".join(node.metadata.source_turn_ids)
    metadata["source_beat_ids"] = "|".join(node.metadata.source_beat_ids)
    if node.summary_level == 4 and not metadata.get("chunk_id"):
        metadata["chunk_id"] = _node_id(node)
    metadata["summary_level"] = node.summary_level
    return metadata


def _collection_ids(collection: Any) -> set[str]:
    try:
        records = collection.get(include=[])
    except TypeError:
        records = collection.get()
    ids = records.get("ids", []) if isinstance(records, dict) else []
    return {str(record_id) for record_id in ids}


def _delete_collection_ids(collection: Any, ids: set[str]) -> None:
    if not ids:
        return
    sorted_ids = sorted(ids)
    batch_size = 500
    for start in range(0, len(sorted_ids), batch_size):
        collection.delete(ids=sorted_ids[start : start + batch_size])


def _prune_stale_records(
    *,
    emitted_ids: set[str],
    lexical_index: LexicalIndex | None,
) -> int:
    collection = get_chroma_collection()
    stale_ids = _collection_ids(collection) - emitted_ids
    _delete_collection_ids(collection, stale_ids)
    if lexical_index is not None:
        lexical_stale_ids = lexical_index.list_ids() - emitted_ids
        lexical_index.delete_records(lexical_stale_ids)
        stale_ids |= lexical_stale_ids
    return len(stale_ids)


def _upsert_story_nodes(
    nodes: list[StoryNode],
    *,
    progress_label: str,
    glossary: dict | None = None,
    lexical_index: LexicalIndex | None = None,
) -> list[str]:
    collection = get_chroma_collection()
    batch_size = 32
    emitted_ids = []

    with Progress() as progress:
        task = progress.add_task(progress_label, total=len(nodes))
        for start in range(0, len(nodes), batch_size):
            batch = nodes[start : start + batch_size]
            documents = [node.text for node in batch]
            embedding_documents = [_embedding_document(node, glossary) for node in batch]
            lexical_documents = [_lexical_document(node, glossary) for node in batch]
            metadatas = [_metadata_for_node(node) for node in batch]
            ids = [_node_id(node) for node in batch]
            emitted_ids.extend(ids)

            collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embed_texts(embedding_documents, task_type=RETRIEVAL_DOCUMENT),
            )
            if lexical_index is not None:
                lexical_index.upsert_records(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    search_texts=lexical_documents,
                )
            progress.update(task, advance=len(batch))
    return emitted_ids

@app.command()
def hello():
    """Test command."""
    console.print("Hello!")

@app.command()
def query(
    question: str,
    routing_mode: str = typer.Option(
        "off",
        "--routing-mode",
        help="Routing mode: off, heuristic, llm_router, or agentic.",
    ),
    show_router: bool = typer.Option(
        False,
        "--show-router/--hide-router",
        help="Print router or agent structured decision and execution metadata.",
    ),
    audit: bool = typer.Option(
        False,
        "--audit/--no-audit",
        help="Run the secondary answer audit pass.",
    ),
):
    """Answers a question based on the RAG index and State Ledger."""
    mode = _routing_mode(routing_mode)
    initialize_query_settings()
    engine = StoryQueryEngine(
        retrieval_config=RetrievalConfig(routing_mode=mode, audit_enabled=audit)
    )
    console.print(f"\n[bold blue]Question:[/bold blue] {question}")
    if show_router and mode in {"llm_router", "agentic"}:
        answer = _print_router_debug(engine, question)
    else:
        if show_router:
            console.print(
                "[yellow]--show-router only applies with --routing-mode llm_router or agentic."
                "[/yellow]"
            )
        answer = engine.query(question)
    console.print(f"\n[bold green]Answer:[/bold green]\n{answer}\n")

@app.command()
def chat(
    routing_mode: str = typer.Option(
        "off",
        "--routing-mode",
        help="Routing mode: off, heuristic, llm_router, or agentic.",
    ),
    show_router: bool = typer.Option(
        False,
        "--show-router/--hide-router",
        help="Print router or agent structured execution metadata for each turn.",
    ),
    audit: bool = typer.Option(
        False,
        "--audit/--no-audit",
        help="Run the secondary answer audit pass for each turn.",
    ),
):
    """Starts an interactive chat session with the RAG index.

    With --audit, each answer is prepared and emitted as one completed block so the
    secondary audit can run before the response is displayed.
    """
    mode = _routing_mode(routing_mode)
    initialize_query_settings()
    engine = StoryQueryEngine(
        retrieval_config=RetrievalConfig(routing_mode=mode, audit_enabled=audit)
    )
    console.print("[bold green]Interactive Chat Started! Type 'exit' or 'quit' to end.[/bold green]")

    while True:
        try:
            question = typer.prompt("Question")
        except (KeyboardInterrupt, EOFError):
            break

        if question.strip().lower() in ["exit", "quit"]:
            break
        if not question.strip():
            continue

        answer_deltas = None
        try:
            console.print("\n[dim]Thinking...[/dim]")
            result = engine.stream_query(question)
            answer_deltas = result.answer_deltas
            if show_router and mode == "llm_router" and result.router_metadata is not None:
                _print_router_metadata(result.router_metadata)
            elif show_router and mode == "agentic" and result.router_metadata is not None:
                _print_agent_metadata(result.router_metadata)
            elif show_router:
                console.print(
                    "[yellow]--show-router only applies with --routing-mode llm_router or agentic."
                    "[/yellow]"
                )
            console.print("\n[bold green]Answer:[/bold green]")
            for delta in answer_deltas:
                console.print(delta, end="", markup=False, highlight=True)
            console.print()
        except KeyboardInterrupt:
            close = getattr(answer_deltas, "close", None)
            if close is not None:
                close()
            console.print("\n[yellow]Answer interrupted. Ready for another question.[/yellow]\n")

    console.print("[bold blue]Chat session ended. Goodbye![/bold blue]")

@app.command()
def extract_state(
    source_db: str = typer.Option("source_records.db", help="Path to the raw source records DB"),
    output_file: str = typer.Option("world_state.json", help="Path to output the world state JSON"),
):
    """Extracts source-backed facts from raw scenes to build the State Ledger."""
    initialize_generation_settings()
    console.print(f"Starting state extraction from raw scenes in {source_db}...")
    
    extractor = StateExtractor(source_db_path=source_db)
    extractor.extract_from_sources(output_file=output_file)

@app.command("extract-glossary-candidates")
def extract_glossary_candidates_command(
    story_dir: str = typer.Option("story", help="Directory containing story files"),
    glossary_file: str = typer.Option("glossary.json", help="Existing curated glossary file"),
    output_file: str = typer.Option(
        DEFAULT_GLOSSARY_CANDIDATE_FILE,
        help="Path to write candidate glossary terms",
    ),
    min_count: int = typer.Option(2, help="Minimum frequency required for a candidate"),
    max_examples: int = typer.Option(3, help="Maximum example lines to keep per candidate"),
    max_files: int = typer.Option(10, help="Maximum file paths to keep per candidate"),
    include_katakana_terms: bool = typer.Option(
        False,
        "--include-katakana-terms/--proper-nouns-only",
        help="Also include common katakana terms that are not tagged as proper nouns",
    ),
    include_existing: bool = typer.Option(
        False,
        "--include-existing/--skip-existing",
        help="Include terms already present in the curated glossary",
    ),
):
    """Extracts Japanese proper-noun candidates for manual glossary curation."""
    story_path = Path(story_dir)
    if not story_path.exists():
        console.print(f"[red]Error: Directory {story_dir} not found.[/red]")
        raise typer.Exit(1)

    candidates = extract_glossary_candidates(
        story_dir=story_path,
        glossary_file=Path(glossary_file),
        output_file=Path(output_file),
        min_count=min_count,
        max_examples=max_examples,
        max_files=max_files,
        include_katakana_terms=include_katakana_terms,
        include_existing=include_existing,
    )
    counts = category_counts(candidates)
    count_summary = ", ".join(
        f"{category}: {count}" for category, count in sorted(counts.items())
    )
    console.print(f"Wrote {len(candidates)} glossary candidates to {output_file}.")
    if count_summary:
        console.print(f"By category: {count_summary}")


def _routing_mode(value: str) -> EvalMode:
    if value not in ROUTING_MODES:
        raise typer.BadParameter(f"routing mode must be one of: {', '.join(ROUTING_MODES)}")
    return cast(EvalMode, value)


@eval_app.command("run")
def eval_run_command(
    golden_set: str = typer.Option(
        "eval/golden_questions.json",
        "--golden-set",
        help="Path to the golden question set JSON.",
    ),
    routing_mode: str = typer.Option(
        "off",
        "--routing-mode",
        help="Routing mode: off, heuristic, llm_router, or agentic.",
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        help="Path to write the complete eval run JSON.",
    ),
    inspect_query_id: str | None = typer.Option(
        None,
        "--inspect",
        help="Print the full deterministic trace for one query id.",
    ),
    dump_traces_dir: str | None = typer.Option(
        None,
        "--dump-traces",
        help="Directory to write one deterministic JSON trace per query.",
    ),
    answer_mode: bool = typer.Option(
        False,
        "--answer-mode/--retrieval-only",
        help="Also synthesize answers and evaluate answer text when credentials are configured.",
    ),
    audit: bool = typer.Option(
        False,
        "--audit/--no-audit",
        help="Run the secondary answer audit pass; requires --answer-mode.",
    ),
):
    """Runs the retrieval evaluation harness."""
    if audit and not answer_mode:
        console.print("[red]Error: --audit requires --answer-mode.[/red]")
        raise typer.Exit(1)
    try:
        run = run_eval_from_file(
            golden_set,
            mode=_routing_mode(routing_mode),
            output_path=output,
            inspect_query_id=inspect_query_id,
            dump_traces_dir=dump_traces_dir,
            answer_mode=answer_mode,
            audit_enabled=audit,
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc

    if inspect_query_id is not None:
        trace = next(trace for trace in run.traces if trace.query_id == inspect_query_id)
        console.print(stable_json(trace))
        return

    console.print(stable_json(run.aggregate_metrics))
    if output is not None:
        console.print(f"[bold green]Wrote eval run to {output}[/bold green]")
    if dump_traces_dir is not None:
        console.print(f"[bold green]Wrote query traces to {dump_traces_dir}[/bold green]")


@eval_app.command("diff")
def eval_diff_command(
    run_a: str = typer.Argument(..., help="Baseline eval run JSON."),
    run_b: str = typer.Argument(..., help="Comparison eval run JSON."),
):
    """Diffs two retrieval evaluation run files."""
    try:
        diff = diff_runs(load_eval_run(run_a), load_eval_run(run_b))
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(stable_json(diff))

answers_app = typer.Typer(help="Run and report the answer-accuracy benchmark.")
eval_app.add_typer(answers_app, name="answers")


def _load_cases_and_corpus(cases_path: str) -> tuple[Any, Any]:
    from .eval.answer_runner import CorpusIndex, load_answer_cases, validate_case_sources

    cases = load_answer_cases(cases_path)
    corpus = CorpusIndex.from_source_store()
    problems = validate_case_sources(cases, corpus)
    if problems:
        for problem in problems:
            console.print(f"[red]{problem}[/red]")
        raise typer.Exit(1)
    return cases, corpus


@answers_app.command("validate")
def answers_validate_command(
    cases_path: str = typer.Option("eval/answer_cases.json", "--cases", help="Answer case suite JSON."),
):
    """Validates the answer case schema and that every evidence passage exists."""
    try:
        cases, _ = _load_cases_and_corpus(cases_path)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc
    reviewed = sum(1 for case in cases.cases if case.review_status == "reviewed")
    console.print(
        f"[bold green]{len(cases.cases)} cases valid ({reviewed} reviewed).[/bold green]"
    )


@answers_app.command("run")
def answers_run_command(
    cases_path: str = typer.Option("eval/answer_cases.json", "--cases", help="Answer case suite JSON."),
    modes: list[str] = typer.Option(
        ["off"],
        "--mode",
        help="Routing mode to run; repeat for several, or pass 'all'.",
    ),
    output_dir: str = typer.Option("runs/answers", "--output-dir", help="Directory for run JSON files."),
    judge_model: str = typer.Option(
        "openai:gpt-6.1-sol",
        "--judge",
        help="provider:model for the answer judge.",
    ),
    use_judge: bool = typer.Option(True, "--judge-answers/--no-judge", help="Score answers with the judge."),
    case_ids: list[str] | None = typer.Option(None, "--case", help="Run only these case ids."),
    reviewed_only: bool = typer.Option(
        False, "--reviewed-only", help="Skip cases whose review_status is still draft."
    ),
    provider: str | None = typer.Option(
        None,
        "--provider",
        help="Generation provider for this run (google or openai); overrides LINKURA_INGEST_PROVIDER.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Generation model for this run; overrides LINKURA_INGEST_MODEL without editing .env.",
    ),
):
    """Runs answer cases through the query engine and judges the answers."""
    from .eval.answer_runner import (
        ModelJudge,
        build_run_config,
        render_answer_report,
        run_answer_eval,
        write_answer_run,
    )

    selected_modes = list(ROUTING_MODES) if "all" in modes else [_routing_mode(mode) for mode in modes]
    # Process-local overrides: the model factories read these on every call.
    if provider is not None:
        os.environ["LINKURA_INGEST_PROVIDER"] = provider
    if model is not None:
        os.environ["LINKURA_INGEST_MODEL"] = model
    try:
        cases, corpus = _load_cases_and_corpus(cases_path)
        initialize_query_settings()
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc
    judge = ModelJudge(judge_model) if use_judge else None

    runs = []
    for mode in selected_modes:
        engine = StoryQueryEngine(retrieval_config=RetrievalConfig(routing_mode=mode))
        config = build_run_config(
            mode=mode,
            cases_path=cases_path,
            cases=cases,
            judge=judge,
            final_top_k=engine.retrieval_config.final_top_k,
            reviewed_only=reviewed_only,
        )
        console.print(f"[bold blue]Running {mode}...[/bold blue]")

        def report_progress(result: Any, mode: str = mode) -> None:
            status = "error" if result.error else (
                "pass" if result.answer.passed else ("unjudged" if not result.answer.judged else "fail")
            )
            console.print(f"  [{mode}] {result.case_id}: {status}", markup=False)

        run = run_answer_eval(
            cases,
            config=config,
            engine=engine,
            corpus=corpus,
            judge=judge,
            case_ids=case_ids,
            on_result=report_progress,
        )
        destination = Path(output_dir) / f"{mode}.json"
        write_answer_run(run, destination)
        runs.append(run)
        console.print(f"[bold green]Wrote {destination}[/bold green]")
    report_path = Path(output_dir) / "report.md"
    report_path.write_text(render_answer_report(runs), encoding="utf-8")
    console.print(f"[bold green]Wrote {report_path}[/bold green]")


@answers_app.command("review")
def answers_review_command(
    cases_path: str = typer.Option("eval/answer_cases.json", "--cases", help="Answer case suite JSON."),
    run_dirs: list[str] = typer.Option(
        ["runs/answers"],
        "--runs",
        help="Directory scanned recursively for answer run JSON files; repeat for several.",
    ),
    port: int = typer.Option(8765, "--port", help="Local port to serve on."),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open the page in a browser."),
):
    """Serves a local page for reviewing, editing, and approving answer cases."""
    import webbrowser

    from .eval.answer_runner import CorpusIndex
    from .eval.review_server import ReviewStore, serve

    store = ReviewStore(cases_path, CorpusIndex.from_source_store(), run_dirs)
    try:
        store.load()
    except ValueError as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc
    server = serve(store, port=port)
    url = f"http://127.0.0.1:{server.server_port}/"
    console.print(f"[bold green]Reviewing {cases_path} at {url}[/bold green] (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("Stopped.")
    finally:
        server.server_close()


@answers_app.command("rescore")
def answers_rescore_command(
    run_files: list[str] = typer.Argument(..., help="Answer run JSON files to rescore in place."),
    cases_path: str = typer.Option("eval/answer_cases.json", "--cases", help="Answer case suite JSON."),
):
    """Re-scores saved runs against the current cases and rules, without new model calls."""
    from .eval.answer_runner import (
        load_answer_cases,
        load_answer_run,
        rescore_run,
        write_answer_run,
    )

    try:
        cases = load_answer_cases(cases_path)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc
    for path in run_files:
        try:
            run, changed, needs_rejudge = rescore_run(load_answer_run(path), cases)
        except (FileNotFoundError, ValueError) as exc:
            console.print(f"[red]Error: {exc}[/red]")
            raise typer.Exit(1) from exc
        write_answer_run(run, path)
        summary = ", ".join(changed) if changed else "no changes"
        console.print(f"{path}: {summary}", markup=False)
        if needs_rejudge:
            console.print(
                f"  needs rejudge (claims changed): {', '.join(needs_rejudge)}", markup=False
            )


@answers_app.command("rejudge")
def answers_rejudge_command(
    run_file: str = typer.Argument(..., help="Answer run JSON whose saved answers to re-judge."),
    judge_model: str | None = typer.Option(
        None, "--judge", help="provider:model for the judge (default: the run's own judge)."
    ),
    output: str | None = typer.Option(
        None, "--output", help="Where to write the re-judged run (default: overwrite the input)."
    ),
    cases_path: str = typer.Option("eval/answer_cases.json", "--cases", help="Answer case suite JSON."),
    case_ids: list[str] | None = typer.Option(None, "--case", help="Re-judge only these case ids."),
):
    """Judges a saved run's answers again with a different judge, without rerunning the system."""
    from .eval.answer_runner import ModelJudge, load_answer_run, rejudge_run, write_answer_run

    try:
        cases, corpus = _load_cases_and_corpus(cases_path)
        run = load_answer_run(run_file)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc

    def report_progress(result: Any) -> None:
        status = "pass" if result.answer.passed else ("unjudged" if not result.answer.judged else "fail")
        console.print(f"  {result.case_id}: {status}", markup=False)

    judge_spec = judge_model or run.config.judge_model
    if not judge_spec:
        console.print("[red]Error: the run has no judge; pass --judge.[/red]")
        raise typer.Exit(1)
    rejudged = rejudge_run(
        run, cases, corpus, ModelJudge(judge_spec), case_ids=case_ids, on_result=report_progress
    )
    destination = output or run_file
    write_answer_run(rejudged, destination)
    console.print(f"[bold green]Wrote {destination}[/bold green]")


@answers_app.command("report")
def answers_report_command(
    run_files: list[str] = typer.Argument(..., help="Answer run JSON files to compare."),
    output: str | None = typer.Option(None, "--output", help="Write the Markdown report here."),
    labels: list[str] | None = typer.Option(
        None, "--label", help="Column label for each run, in order; repeat once per run."
    ),
):
    """Renders a Markdown comparison of answer benchmark runs."""
    from .eval.answer_runner import load_answer_run, render_answer_report

    try:
        report = render_answer_report([load_answer_run(path) for path in run_files], labels)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc
    if output is None:
        console.print(report, markup=False)
        return
    Path(output).write_text(report, encoding="utf-8")
    console.print(f"[bold green]Wrote {output}[/bold green]")


@app.command("export-summary-reader")
def export_summary_reader_command(
    cache_file: str = typer.Option("summaries_cache.json", help="Path to the summaries cache file"),
    output_dir: str = typer.Option(
        "site/summary-reader",
        help="Directory to write the static reader site",
    ),
    story_order_file: str = typer.Option("story_order.yaml", help="Path to story order YAML"),
    template_file: str = typer.Option(
        str(DEFAULT_READER_TEMPLATE),
        help="Path to the summary reader HTML template",
    ),
):
    """Exports cached structured summaries as a static browser reader."""
    try:
        story_order = load_story_order(story_order_file)
        destination = export_summary_reader(
            cache_file=cache_file,
            output_dir=output_dir,
            story_order=story_order,
            template_file=template_file,
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc

    console.print(f"[bold green]Summary reader exported to {destination}[/bold green]")

@app.command("export-production-summary-reader")
def export_production_summary_reader_command(
    cache_file: str = typer.Option("summaries_cache.json", help="Path to the summaries cache file"),
    output_dir: str = typer.Option(
        "site/summary-reader-production",
        help="Directory to write the static production reader site",
    ),
    story_order_file: str = typer.Option("story_order.yaml", help="Path to story order YAML"),
    source_dir: str = typer.Option(
        str(DEFAULT_PRODUCTION_READER_SOURCE),
        help="Path to the production reader source directory",
    ),
):
    """Exports the zip production summary reader as a static site."""
    try:
        story_order = load_story_order(story_order_file)
        destination = export_production_summary_reader(
            cache_file=cache_file,
            output_dir=output_dir,
            story_order=story_order,
            source_dir=source_dir,
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc

    console.print(f"[bold green]Production summary reader exported to {destination}[/bold green]")

@app.command()
def ingest(
    story_dir: str = typer.Option("story", help="Directory containing story files"),
    cache_file: str = typer.Option("summaries_cache.json", help="Path to the summaries cache file"),
    manifest_file: str = typer.Option("ingestion_manifest.json", help="Path to the ingestion manifest"),
    prune: bool = typer.Option(True, "--prune/--no-prune", help="Delete indexed records not emitted by this run"),
    verify_summaries: bool = typer.Option(
        True,
        "--verify-summaries/--no-verify-summaries",
        help="Fact-check each Part summary against its source text with a second model call",
    ),
):
    """Walks the story directory, generates hierarchical summaries, and indexes them into ChromaDB."""
    initialize_ingest_settings()
    
    story_path = Path(story_dir)
    if not story_path.exists():
        console.print(f"[red]Error: Directory {story_dir} not found.[/red]")
        raise typer.Exit(1)
        
    md_files = sorted(story_path.rglob("*.md"), key=lambda path: str(path))
    source_file_hashes = hash_files(md_files)
    console.print(f"Found {len(md_files)} markdown files. Parsing scenes...")
    story_order = load_story_order(story_root=story_path)

    raw_nodes = []
    with Progress() as progress:
        task = progress.add_task("[green]Processing files...", total=len(md_files))
        for file in md_files:
            nodes = StoryProcessor.process_file(file)
            raw_nodes.extend(nodes)
            progress.update(task, advance=1)

    _assign_canonical_story_order(raw_nodes, story_order=story_order)

    retrieval_chunks = build_retrieval_chunks(raw_nodes)
    SourceRecordStore().replace_all(raw_nodes, retrieval_chunks)

    console.print(
        f"Parsed {len(raw_nodes)} raw scenes and built {len(retrieval_chunks)} retrieval chunks. "
        "Starting Hierarchical Summarization..."
    )
    
    glossary = None
    glossary_path = Path("glossary.json")
    glossary_hash = hash_json_file(glossary_path)
    if glossary_path.exists():
        with open(glossary_path, encoding="utf-8") as f:
            glossary = json.load(f)
            console.print("Loaded glossary for translation invariants.")

    # Generate Tier 1-3 hierarchical summaries
    generation_provider = get_generation_provider_name()
    generation_model = get_generation_model_name()
    chat_model = get_chat_model_name()
    embedding_model = get_embedding_model_name()
    cache_context = SummaryCacheContext(
        source_file_hashes=source_file_hashes,
        parser_version=PARSER_VERSION,
        summarization_prompt_version=SUMMARIZATION_PROMPT_VERSION,
        glossary_hash=glossary_hash,
        chat_model=chat_model,
        generation_provider=generation_provider,
        generation_model=generation_model,
        embedding_model=embedding_model,
        summary_cache_schema_version=SUMMARY_CACHE_SCHEMA_VERSION,
    )
    summarizer = HierarchicalSummarizer(
        glossary=glossary,
        story_order=story_order,
        cache_context=cache_context,
        verify_parts=verify_summaries,
    )
    summary_nodes = summarizer.summarize_hierarchy(raw_nodes, cache_file=cache_file)
    
    console.print(f"Generated {len(summary_nodes)} hierarchical summaries. Upserting to Vector DB...")
    lexical_index = LexicalIndex(get_lexical_db_path())
    
    raw_ids = _upsert_story_nodes(
        retrieval_chunks,
        progress_label="[green]Embedding raw retrieval chunks...",
        glossary=glossary,
        lexical_index=lexical_index,
    )
    summary_ids = _upsert_story_nodes(
        summary_nodes,
        progress_label="[green]Embedding summaries...",
        glossary=glossary,
        lexical_index=lexical_index,
    )

    emitted_ids = {*raw_ids, *summary_ids}
    if prune:
        pruned_count = _prune_stale_records(emitted_ids=emitted_ids, lexical_index=lexical_index)
        console.print(f"Pruned {pruned_count} stale vector/lexical records.")

    manifest = IngestionManifest(
        timestamp=utc_timestamp(),
        source_file_hashes=source_file_hashes,
        parser_version=PARSER_VERSION,
        chunker_version=CHUNKER_VERSION,
        chunker_config=ChunkerConfig(
            min_chars=MIN_USEFUL_CHARS,
            target_chars=TARGET_CHUNK_CHARS,
            max_chars=MAX_CHUNK_CHARS,
        ),
        summarization_prompt_version=SUMMARIZATION_PROMPT_VERSION,
        glossary_hash=glossary_hash,
        chat_model=chat_model,
        generation_provider=generation_provider,
        generation_model=generation_model,
        embedding_model=embedding_model,
        raw_evidence_schema_version=RAW_EVIDENCE_SCHEMA_VERSION,
        summary_cache_schema_version=SUMMARY_CACHE_SCHEMA_VERSION,
        vector_ids=VectorIds(raw=sorted(raw_ids), summaries=sorted(summary_ids)),
    )
    write_manifest(manifest_file, manifest)
    
    console.print("[bold green]Hierarchical Ingestion complete![/bold green]")

def main():
    # The CLI owns its terminal output; suppress pydantic-ai's first-run Logfire banner.
    os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
    app()

if __name__ == "__main__":
    main()
