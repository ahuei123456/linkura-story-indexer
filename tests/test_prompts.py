from importlib.resources import files

import pytest

from linkura_story_indexer import prompts


def test_packaged_prompt_resource_loads_after_working_directory_change(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    prompt = prompts.load_prompt("answer_system.md")

    assert "# Answer policy" in prompt
    assert files("linkura_story_indexer.prompts").joinpath("answer_raw_user.md").is_file()


def test_raw_and_summary_system_policies_are_distinct() -> None:
    raw = prompts.render_system_prompt(
        context_kind="raw", glossary="", state_ledger="", year_summaries=""
    )
    summary = prompts.render_system_prompt(
        context_kind="summary", glossary="", state_ledger="", year_summaries=""
    )

    assert "raw source text" in raw
    assert "generated summaries" not in raw
    assert "generated summaries" in summary
    assert "never present it as a quotation" in summary
    assert prompts.PROMPT_VERSION in raw


def test_dynamic_braces_and_markdown_are_not_interpreted() -> None:
    question = "What does {character} mean by [[not-a-token]] and **this**?"
    context = 'CITATION: label\n```json\n{"value": "{literal}"}\n```'

    rendered = prompts.render_user_prompt(
        context_kind="raw", question=question, context=context
    )

    assert question in rendered
    assert context in rendered


def test_missing_resource_has_actionable_error() -> None:
    with pytest.raises(prompts.PromptResourceError, match="Reinstall or rebuild"):
        prompts.load_prompt("missing.md")


def test_load_year_summaries_selects_only_valid_year_entries(tmp_path) -> None:
    cache = tmp_path / "summaries.json"
    cache.write_text(
        """{
          "YEAR|104": {"summary": "Year {104} **summary**", "inputs": {"level": "year"}},
          "YEAR|103": {"summary": "Year 103", "inputs": {"level": "year"}},
          "EPISODE|104|Main|1": {"summary": "Episode", "inputs": {"level": "episode"}},
          "YEAR|bad": {"summary": "Wrong level", "inputs": {"level": "episode"}}
        }""",
        encoding="utf-8",
    )

    assert prompts.load_year_summaries(cache) == {
        "104": "Year {104} **summary**",
        "103": "Year 103",
    }


def test_load_year_summaries_allows_missing_cache(tmp_path) -> None:
    assert prompts.load_year_summaries(tmp_path / "missing.json") == {}


def test_load_year_summaries_rejects_malformed_cache(tmp_path) -> None:
    cache = tmp_path / "broken.json"
    cache.write_text("{broken", encoding="utf-8")

    with pytest.raises(prompts.PromptResourceError, match="Regenerate or repair"):
        prompts.load_year_summaries(cache)
