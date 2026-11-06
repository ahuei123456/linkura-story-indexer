"""Compare Part-summary output across generation models on a sample of parts.

Re-generates selected Part summaries with the current summarization prompt for
each candidate model and writes a side-by-side Markdown report next to the
summary already in the cache. Nothing is written to `summaries_cache.json`.

The previous-part context for each sample comes from the existing cache, so
every candidate sees the same input that a real ingest would give it.

Usage:
    uv run python scripts/compare_summary_models.py \
        --model google:gemini-3.8-flash --model google:gemini-pro-latest \
        --part "104|Main|第8話『Not a marionette』|2" --random 6 --seed 7
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings, ThinkingEffort

from linkura_story_indexer.database import create_google_model, create_openai_model
from linkura_story_indexer.indexer.processor import StoryProcessor
from linkura_story_indexer.indexer.summarizer import (
    SUMMARIZATION_PROMPT_VERSION,
    HierarchicalSummarizer,
    build_verification_prompt,
    trim_previous_summary_context,
)
from linkura_story_indexer.models.story import StoryNode
from linkura_story_indexer.story_order import load_story_order


def _create_model(spec: str) -> Any:
    provider, _, model_name = spec.partition(":")
    if not model_name:
        raise SystemExit(f"Model spec {spec!r} must look like provider:model")
    if provider == "google":
        return create_google_model(model_name)
    if provider == "openai":
        return create_openai_model(model_name)
    raise SystemExit(f"Unsupported provider {provider!r} in {spec!r}")


def _ordered_parts(
    story_dir: Path,
) -> tuple[list[str], dict[str, tuple[str, list[StoryNode]]]]:
    """Return part cache keys in summary order, plus each part's text and scenes."""
    story_order = load_story_order(story_root=story_dir)
    episodes: dict[tuple, dict[str, list[StoryNode]]] = defaultdict(lambda: defaultdict(list))
    for file in sorted(story_dir.rglob("*.md")):
        for node in StoryProcessor.process_file(file):
            meta = node.metadata
            episodes[(meta.arc_id, meta.story_type, meta.episode_name)][meta.part_name].append(node)

    keys: list[str] = []
    parts: dict[str, tuple[str, list[StoryNode]]] = {}
    for ep_key in sorted(episodes, key=lambda key: story_order.summary_episode_key(*key)):
        arc_id, story_type, episode_name = ep_key
        part_names = sorted(
            episodes[ep_key],
            key=lambda name: story_order.part_key(arc_id, story_type, episode_name, name),
        )
        for part_name in part_names:
            scenes = sorted(episodes[ep_key][part_name], key=lambda n: n.metadata.scene_index)
            key = f"{arc_id}|{story_type}|{episode_name}|{part_name}"
            keys.append(key)
            parts[key] = ("\n\n---\n\n".join(n.text for n in scenes), scenes)
    return keys, parts


async def _summarize(
    semaphore: asyncio.Semaphore,
    spec: str,
    system: str,
    prompt: str,
    thinking: ThinkingEffort | None,
    verify_source: str | None = None,
    glossary: dict | None = None,
) -> tuple[str, float]:
    async with semaphore:
        started = time.perf_counter()
        try:
            settings: ModelSettings = {"thinking": thinking} if thinking else {}
            model = _create_model(spec)
            result = await Agent(model, instructions=system).run(prompt, model_settings=settings)
            text = result.output.strip()
            if verify_source is not None:
                verify_system, verify_prompt = build_verification_prompt(
                    verify_source, text, glossary
                )
                result = await Agent(model, instructions=verify_system).run(
                    verify_prompt, model_settings=settings
                )
                text = result.output.strip()
        except Exception as exc:  # report failures inline instead of aborting the run
            text = f"ERROR: {type(exc).__name__}: {exc}"
        return text, time.perf_counter() - started


async def _run(args: argparse.Namespace) -> None:
    cache = json.loads(Path(args.cache).read_text(encoding="utf-8"))
    glossary_path = Path(args.glossary)
    glossary = (
        json.loads(glossary_path.read_text(encoding="utf-8")) if glossary_path.exists() else None
    )
    keys, parts = _ordered_parts(Path(args.story_dir))

    # Story paths may use decomposed kana (e.g. from macOS), so match keys after NFC.
    by_nfc = {unicodedata.normalize("NFC", key): key for key in keys}
    missing = [key for key in args.part if unicodedata.normalize("NFC", key) not in by_nfc]
    if missing:
        raise SystemExit(f"Unknown part keys: {missing}")
    selected = [by_nfc[unicodedata.normalize("NFC", key)] for key in args.part]
    pool = [key for key in keys if key not in selected]
    selected += random.Random(args.seed).sample(pool, args.random)

    summarizer = HierarchicalSummarizer(glossary=glossary)
    runs = [
        (spec if args.repeat == 1 else f"{spec} #{run + 1}", spec)
        for spec in args.model
        for run in range(args.repeat)
    ]
    semaphore = asyncio.Semaphore(args.concurrency)
    jobs = []
    for key in selected:
        index = keys.index(key)
        prev_key = keys[index - 1] if index > 0 else None
        prev_summary = cache.get(prev_key, {}).get("summary") if prev_key else None
        system, prompt = summarizer._build_summary_prompt(
            parts[key][0],
            trim_previous_summary_context(prev_summary),
            "Part",
        )
        for label, spec in runs:
            jobs.append(
                (
                    key,
                    label,
                    _summarize(
                        semaphore,
                        spec,
                        system,
                        prompt,
                        args.thinking,
                        parts[key][0] if args.verify else None,
                        glossary,
                    ),
                )
            )

    print(
        f"Running {len(jobs)} summaries "
        f"({len(selected)} parts x {len(args.model)} models x {args.repeat} runs)..."
    )
    results = await asyncio.gather(*(job[2] for job in jobs))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    by_part: dict[str, dict[str, Any]] = defaultdict(dict)
    for (key, label, _), (text, seconds) in zip(jobs, results, strict=True):
        by_part[key][label] = {"summary": text, "seconds": round(seconds, 1)}

    lines = [
        "# Part summary model comparison",
        "",
        f"Prompt version: {SUMMARIZATION_PROMPT_VERSION}. Baseline is the existing cache entry.",
        "",
    ]
    for key in selected:
        source_file = parts[key][1][0].metadata.file_path
        baseline = cache.get(key, {})
        baseline_model = baseline.get("inputs", {}).get("chat_model", "unknown")
        lines += [f"## {key}", "", f"Source: `{source_file}`", ""]
        lines += [
            f"### Baseline (cache: {baseline_model})",
            "",
            baseline.get("summary", "(none)"),
            "",
        ]
        for label, _ in runs:
            entry = by_part[key][label]
            lines += [f"### {label} ({entry['seconds']}s)", "", entry["summary"], ""]

    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "results.json").write_text(
        json.dumps(
            {"prompt_version": SUMMARIZATION_PROMPT_VERSION, "parts": selected, "results": by_part},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    errors = sum(1 for text, _ in results if text.startswith("ERROR:"))
    print(f"Wrote {output_dir / 'report.md'} ({errors} errors)")


def main() -> None:
    os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", action="append", required=True, help="provider:model")
    parser.add_argument("--part", action="append", default=[], help="Part cache key to include")
    parser.add_argument("--random", type=int, default=0, help="Extra random parts to include")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--thinking",
        choices=["low", "medium", "high"],
        help="Thinking effort override (default: the model's own default)",
    )
    parser.add_argument("--verify", action="store_true", help="Run a fact-check pass on each draft")
    parser.add_argument("--repeat", type=int, default=1, help="Runs per model and part")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--story-dir", default="story")
    parser.add_argument("--cache", default="summaries_cache.json")
    parser.add_argument("--glossary", default="glossary.json")
    parser.add_argument("--output", default=".tmp/summary-comparison")
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
