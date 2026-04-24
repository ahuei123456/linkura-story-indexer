import json
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress

from .database import embed_texts, get_chroma_collection, initialize_settings
from .indexer.extractor import StateExtractor
from .indexer.processor import StoryProcessor
from .indexer.summarizer import HierarchicalSummarizer
from .models.story import StoryNode
from .query.engine import StoryQueryEngine

app = typer.Typer()
console = Console()


def _node_id(node: StoryNode) -> str:
    meta = node.metadata
    return (
        f"{meta.arc_id}|{meta.story_type}|{meta.episode_name}|{meta.part_name}|"
        f"level:{node.summary_level}|scene:{meta.scene_index}"
    )


def _upsert_summary_nodes(nodes: list[StoryNode]) -> None:
    collection = get_chroma_collection()
    batch_size = 32

    with Progress() as progress:
        task = progress.add_task("[green]Embedding summaries...", total=len(nodes))
        for start in range(0, len(nodes), batch_size):
            batch = nodes[start : start + batch_size]
            documents = [node.text for node in batch]
            metadatas = []
            for node in batch:
                metadata = node.metadata.model_dump()
                metadata["summary_level"] = node.summary_level
                metadatas.append(metadata)

            collection.upsert(
                ids=[_node_id(node) for node in batch],
                documents=documents,
                metadatas=metadatas,
                embeddings=embed_texts(documents),
            )
            progress.update(task, advance=len(batch))

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
def ingest(story_dir: str = typer.Option("story", help="Directory containing story files")):
    """Walks the story directory, generates hierarchical summaries, and indexes them into ChromaDB."""
    initialize_settings()
    
    story_path = Path(story_dir)
    if not story_path.exists():
        console.print(f"[red]Error: Directory {story_dir} not found.[/red]")
        raise typer.Exit(1)
        
    md_files = list(story_path.rglob("*.md"))
    console.print(f"Found {len(md_files)} markdown files. Parsing scenes...")

    raw_nodes = []
    with Progress() as progress:
        task = progress.add_task("[green]Processing files...", total=len(md_files))
        for file in md_files:
            nodes = StoryProcessor.process_file(file)
            raw_nodes.extend(nodes)
            progress.update(task, advance=1)

    console.print(f"Parsed {len(raw_nodes)} raw scenes. Starting Hierarchical Summarization...")
    
    glossary = None
    glossary_path = Path("glossary.json")
    if glossary_path.exists():
        with open(glossary_path, encoding="utf-8") as f:
            glossary = json.load(f)
            console.print("Loaded glossary for translation invariants.")

    # Generate Tier 1-3 hierarchical summaries
    summarizer = HierarchicalSummarizer(glossary=glossary)
    summary_nodes = summarizer.summarize_hierarchy(raw_nodes)
    
    console.print(f"Generated {len(summary_nodes)} hierarchical summaries. Upserting to Vector DB...")
    
    _upsert_summary_nodes(summary_nodes)
    
    console.print("[bold green]Hierarchical Ingestion complete![/bold green]")

def main():
    app()

if __name__ == "__main__":
    main()
