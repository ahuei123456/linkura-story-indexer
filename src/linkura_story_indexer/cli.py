import json
import re
from pathlib import Path
from typing import Any

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
    initialize_settings,
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
from .indexer.processor import StoryProcessor
from .indexer.summarizer import SUMMARIZATION_PROMPT_VERSION, HierarchicalSummarizer
from .lexical import LexicalIndex, get_lexical_db_path, glossary_alias_groups
from .models.story import StoryNode
from .query.engine import StoryQueryEngine
from .story_order import StoryOrder, default_story_order, load_story_order

app = typer.Typer()
console = Console()


def _node_id(node: StoryNode) -> str:
    meta = node.metadata
    if node.summary_level == 4:
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


def _episode_number(node: StoryNode) -> int:
    for value in (node.metadata.episode_name, node.metadata.part_name):
        match = re.search(r"第(\d+)話", value)
        if match:
            return int(match.group(1))
    return 0


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


def _embedding_document(node: StoryNode, glossary: dict | None = None) -> EmbeddingDocument:
    if node.summary_level != 4:
        return EmbeddingDocument(text=node.text, title=_embedding_document_title(node))

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
        meta = node.metadata
        header = "\n".join(
            [
                f"Year: {meta.arc_id}",
                f"Story type: {meta.story_type}",
                f"Episode: {meta.episode_name}",
                f"Part: {meta.part_name}",
                f"Summary level: {node.summary_level}",
                "",
            ]
        )
        return f"{header}{node.text}"
    return embedding_document.text


def _metadata_for_node(node: StoryNode) -> dict:
    metadata = node.metadata.model_dump()
    if not metadata.get("story_order"):
        metadata["story_order"] = metadata.get("canonical_story_order", 0)
    if not metadata.get("episode_number"):
        metadata["episode_number"] = _episode_number(node)
    metadata["detected_speakers"] = "|".join(node.metadata.detected_speakers)
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
def query(question: str):
    """Answers a question based on the RAG index and State Ledger."""
    initialize_settings()
    engine = StoryQueryEngine()
    console.print(f"\n[bold blue]Question:[/bold blue] {question}")
    answer = engine.query(question)
    console.print(f"\n[bold green]Answer:[/bold green]\n{answer}\n")

@app.command()
def chat():
    """Starts an interactive chat session with the RAG index."""
    initialize_settings()
    engine = StoryQueryEngine()
    console.print("[bold green]Interactive Chat Started! Type 'exit' or 'quit' to end.[/bold green]")
    
    while True:
        try:
            question = typer.prompt("Question")
            if question.strip().lower() in ["exit", "quit"]:
                break
            if not question.strip():
                continue
                
            console.print("\n[dim]Thinking...[/dim]")
            answer = engine.query(question)
            console.print(f"\n[bold green]Answer:[/bold green]\n{answer}\n")
        except (KeyboardInterrupt, EOFError):
            break
            
    console.print("[bold blue]Chat session ended. Goodbye![/bold blue]")

@app.command()
def extract_state(cache_file: str = typer.Option("summaries_cache.json", help="Path to the summaries cache file"), output_file: str = typer.Option("world_state.json", help="Path to output the world state JSON")):
    """Extracts facts from cached Episode summaries to build the State Ledger."""
    initialize_settings()
    console.print(f"Starting state extraction from {cache_file}...")
    
    extractor = StateExtractor(cache_file=cache_file)
    extractor.extract_from_cache(output_file=output_file)

@app.command()
def ingest(
    story_dir: str = typer.Option("story", help="Directory containing story files"),
    cache_file: str = typer.Option("summaries_cache.json", help="Path to the summaries cache file"),
    manifest_file: str = typer.Option("ingestion_manifest.json", help="Path to the ingestion manifest"),
    prune: bool = typer.Option(True, "--prune/--no-prune", help="Delete indexed records not emitted by this run"),
):
    """Walks the story directory, generates hierarchical summaries, and indexes them into ChromaDB."""
    initialize_settings()
    
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
    chat_model = get_chat_model_name()
    embedding_model = get_embedding_model_name()
    cache_context = SummaryCacheContext(
        source_file_hashes=source_file_hashes,
        parser_version=PARSER_VERSION,
        summarization_prompt_version=SUMMARIZATION_PROMPT_VERSION,
        glossary_hash=glossary_hash,
        chat_model=chat_model,
        embedding_model=embedding_model,
        summary_cache_schema_version=SUMMARY_CACHE_SCHEMA_VERSION,
    )
    summarizer = HierarchicalSummarizer(
        glossary=glossary,
        story_order=story_order,
        cache_context=cache_context,
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
        embedding_model=embedding_model,
        raw_evidence_schema_version=RAW_EVIDENCE_SCHEMA_VERSION,
        summary_cache_schema_version=SUMMARY_CACHE_SCHEMA_VERSION,
        vector_ids=VectorIds(raw=sorted(raw_ids), summaries=sorted(summary_ids)),
    )
    write_manifest(manifest_file, manifest)
    
    console.print("[bold green]Hierarchical Ingestion complete![/bold green]")

def main():
    app()

if __name__ == "__main__":
    main()
