"""Load and render the packaged answer-generation prompts."""

import json
from importlib.resources import files
from pathlib import Path
from typing import Literal

PROMPT_VERSION = "answer-v2"


class PromptResourceError(RuntimeError):
    """Raised when a required packaged prompt cannot be loaded."""


def load_prompt(name: str) -> str:
    """Load a UTF-8 Markdown prompt from this installed package."""
    resource = files(__package__).joinpath(name)
    try:
        return resource.read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, OSError) as exc:
        raise PromptResourceError(
            f"Required prompt resource {name!r} is missing from package {__package__!r}. "
            "Reinstall or rebuild linkura-story-indexer with its Markdown package resources."
        ) from exc


def _render(name: str, values: dict[str, str]) -> str:
    rendered = load_prompt(name)
    for key, value in values.items():
        token = f"[[{key}]]"
        if token not in rendered:
            raise PromptResourceError(f"Prompt resource {name!r} is missing required token {token}.")
        rendered = rendered.replace(token, value)
    return rendered.strip()


def render_system_prompt(
    *,
    context_kind: Literal["raw", "summary"],
    glossary: str,
    state_ledger: str,
    year_summaries: str,
) -> str:
    """Render the system prompt with dynamic consistency context."""
    if context_kind == "raw":
        context_policy = (
            "Answer based strictly on the provided raw source text. Ground every final factual "
            "claim in that raw evidence."
        )
        citation_policy = "Final citations must come from retrieved raw-evidence labels."
    else:
        context_policy = (
            "The retrieved context may be generated summaries rather than raw source text. "
            "Treat it as summary evidence, disclose that limitation when relevant, and never "
            "present it as a quotation from or direct inspection of the raw source."
        )
        citation_policy = "Final citations must come from retrieved summary-context labels."
    return _render(
        "answer_system.md",
        {
            "PROMPT_VERSION": PROMPT_VERSION,
            "CONTEXT_POLICY": context_policy,
            "CITATION_POLICY": citation_policy,
            "GLOSSARY": glossary,
            "STATE_LEDGER": state_ledger,
            "YEAR_SUMMARIES": year_summaries,
        },
    )


def render_user_prompt(
    *, context_kind: Literal["raw", "summary"], question: str, context: str
) -> str:
    """Render a mode-specific answer request without interpreting dynamic text."""
    return _render(
        f"answer_{context_kind}_user.md",
        {"QUESTION": question, "CONTEXT": context},
    )


def load_year_summaries(cache_file: str | Path) -> dict[str, str]:
    """Load valid Year-level summaries from a hierarchy summary cache."""
    path = Path(cache_file)
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PromptResourceError(
            f"Could not load Year summaries from {path}: {exc}. "
            "Regenerate or repair the summary cache."
        ) from exc
    if not isinstance(loaded, dict):
        raise PromptResourceError(
            f"Could not load Year summaries from {path}: the cache root must be an object."
        )

    summaries: dict[str, str] = {}
    for key, entry in loaded.items():
        if not isinstance(key, str) or not key.startswith("YEAR|") or not isinstance(entry, dict):
            continue
        arc_id = key.removeprefix("YEAR|")
        summary = entry.get("summary")
        level = entry.get("inputs", {}).get("level")
        if arc_id and isinstance(summary, str) and summary.strip() and level == "year":
            summaries[arc_id] = summary.strip()
    return summaries
