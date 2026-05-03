import json
from pathlib import Path
from typing import Any

import pytest

from linkura_story_indexer import cli
from linkura_story_indexer.indexer.chunker import (
    CHUNKER_VERSION,
    MAX_CHUNK_CHARS,
    MIN_USEFUL_CHARS,
    TARGET_CHUNK_CHARS,
    build_retrieval_chunks,
)
from linkura_story_indexer.indexer.manifest import (
    RAW_EVIDENCE_SCHEMA_VERSION,
    SUMMARY_CACHE_SCHEMA_VERSION,
    ChunkerConfig,
    IngestionManifest,
    SummaryCacheContext,
    VectorIds,
    hash_text,
    stable_hash,
)
from linkura_story_indexer.indexer.parser import PARSER_VERSION
from linkura_story_indexer.indexer.processor import StoryProcessor
from linkura_story_indexer.indexer.summarizer import (
    EPISODE_SUMMARY_SECTIONS,
    PART_SUMMARY_SECTIONS,
    SUMMARIZATION_PROMPT_VERSION,
    YEAR_SUMMARY_SECTIONS,
    HierarchicalSummarizer,
    extract_summary_sections,
    trim_previous_summary_context,
)
from linkura_story_indexer.lexical import LexicalIndex


def _write_story_file(root: Path, relative_path: str, content: str) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _cache_context(path: Path, **overrides: Any) -> SummaryCacheContext:
    values = {
        "source_file_hashes": {str(path): "source-hash-1"},
        "parser_version": PARSER_VERSION,
        "summarization_prompt_version": SUMMARIZATION_PROMPT_VERSION,
        "glossary_hash": "glossary-hash-1",
        "chat_model": "chat-model-1",
        "generation_provider": "google",
        "generation_model": "chat-model-1",
        "embedding_model": "embedding-model-1",
        "summary_cache_schema_version": SUMMARY_CACHE_SCHEMA_VERSION,
    }
    values.update(overrides)
    return SummaryCacheContext(**values)


def test_manifest_serializes_required_fields_with_expected_types() -> None:
    manifest = IngestionManifest(
        timestamp="2026-04-27T12:00:00+00:00",
        source_file_hashes={"story/103/part.md": "abc123"},
        parser_version=PARSER_VERSION,
        chunker_version=CHUNKER_VERSION,
        chunker_config=ChunkerConfig(
            min_chars=MIN_USEFUL_CHARS,
            target_chars=TARGET_CHUNK_CHARS,
            max_chars=MAX_CHUNK_CHARS,
        ),
        summarization_prompt_version=SUMMARIZATION_PROMPT_VERSION,
        glossary_hash="glossary-hash",
        chat_model="chat-model",
        generation_provider="google",
        generation_model="chat-model",
        embedding_model="embedding-model",
        raw_evidence_schema_version=RAW_EVIDENCE_SCHEMA_VERSION,
        summary_cache_schema_version=SUMMARY_CACHE_SCHEMA_VERSION,
        vector_ids=VectorIds(raw=["chunk:part:0-1"], summaries=["summary:part:part"]),
    )

    data = json.loads(manifest.model_dump_json())

    assert data["schema_version"] == "1"
    assert isinstance(data["timestamp"], str)
    assert isinstance(data["source_file_hashes"], dict)
    assert isinstance(data["parser_version"], str)
    assert isinstance(data["chunker_version"], str)
    assert data["chunker_config"] == {
        "min_chars": MIN_USEFUL_CHARS,
        "target_chars": TARGET_CHUNK_CHARS,
        "max_chars": MAX_CHUNK_CHARS,
    }
    assert isinstance(data["summarization_prompt_version"], str)
    assert isinstance(data["glossary_hash"], str)
    assert isinstance(data["chat_model"], str)
    assert data["generation_provider"] == "google"
    assert data["generation_model"] == "chat-model"
    assert isinstance(data["embedding_model"], str)
    assert isinstance(data["raw_evidence_schema_version"], str)
    assert isinstance(data["summary_cache_schema_version"], str)
    assert data["vector_ids"]["raw"] == ["chunk:part:0-1"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_file_hashes", None),
        ("parser_version", "parser-version-2"),
        ("summarization_prompt_version", "prompt-version-2"),
        ("glossary_hash", "glossary-hash-2"),
        ("chat_model", "chat-model-2"),
        ("generation_provider", "openai"),
        ("generation_model", "gpt-5-mini"),
        ("embedding_model", "embedding-model-2"),
        ("summary_cache_schema_version", "summary-schema-2"),
    ],
)
def test_summary_cache_invalidates_when_tracked_input_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str | None,
) -> None:
    story_root = tmp_path / "story"
    path = _write_story_file(
        story_root,
        "103/第1話『花咲きたい！』/1.md",
        "花帆: こんにちは\n---\nさやか: どうしたの？",
    )
    raw_nodes = StoryProcessor.process_file(path)
    cache_file = tmp_path / "summaries_cache.json"
    calls: list[str] = []

    def fake_generate(
        self: HierarchicalSummarizer,
        current_text: str,
        prev_summary: str | None = None,
        level_name: str = "Part",
    ) -> str:
        calls.append(current_text)
        return f"{level_name} summary {len(calls)}"

    monkeypatch.setattr(HierarchicalSummarizer, "_generate_rolling_summary", fake_generate)

    context = _cache_context(path)
    HierarchicalSummarizer(cache_context=context).summarize_parts(
        raw_nodes,
        cache_file=str(cache_file),
    )
    HierarchicalSummarizer(cache_context=context).summarize_parts(
        raw_nodes,
        cache_file=str(cache_file),
    )

    assert len(calls) == 1

    if field == "source_file_hashes":
        changed_context = context.model_copy(
            update={"source_file_hashes": {str(path): "source-hash-2"}}
        )
    else:
        changed_context = context.model_copy(update={field: value})

    HierarchicalSummarizer(cache_context=changed_context).summarize_parts(
        raw_nodes,
        cache_file=str(cache_file),
    )

    assert len(calls) == 2


def test_summarizer_uses_configured_generation_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, str]] = []

    class FakeAgent:
        def __init__(self, instructions: str) -> None:
            self.instructions = instructions

        def run_sync(self, prompt: str) -> Any:
            calls.append({"instructions": self.instructions, "prompt": prompt})

            class Result:
                output = " generated summary "

            return Result()

    def fake_create_generation_text_agent(instructions: str) -> FakeAgent:
        return FakeAgent(instructions)

    monkeypatch.setattr(
        "linkura_story_indexer.indexer.summarizer.create_generation_text_agent",
        fake_create_generation_text_agent,
    )

    summary = HierarchicalSummarizer()._generate_rolling_summary(
        "花帆: こんにちは",
        level_name="Part",
    )

    assert summary == "generated summary"
    assert len(calls) == 1
    assert "expert archivist" in calls[0]["instructions"]
    assert "花帆: こんにちは" in calls[0]["prompt"]


@pytest.mark.parametrize(
    ("level_name", "required_sections"),
    [
        ("Part", PART_SUMMARY_SECTIONS),
        ("Episode", EPISODE_SUMMARY_SECTIONS),
        ("Year", YEAR_SUMMARY_SECTIONS),
    ],
)
def test_summary_prompt_includes_required_tier_sections(
    level_name: str,
    required_sections: tuple[str, ...],
) -> None:
    _, prompt = HierarchicalSummarizer()._build_summary_prompt(
        "source text",
        level_name=level_name,
    )

    for section in required_sections:
        assert f"{section}:" in prompt

    assert "Always emit every required section." in prompt
    assert "If a bullet-list section has no applicable entries, write exactly `- None`." in prompt
    assert "Do not use Markdown headings, bold text, numbered lists, tables, or extra sections." in prompt


@pytest.mark.parametrize(
    ("level_name", "input_phrase"),
    [
        ("Episode", "The current Episode input is multiple structured Part summaries."),
        ("Year", "The current Year input is multiple structured Episode summaries."),
    ],
)
def test_aggregate_summary_prompts_describe_structured_child_inputs(
    level_name: str,
    input_phrase: str,
) -> None:
    _, prompt = HierarchicalSummarizer()._build_summary_prompt(
        "child summaries",
        level_name=level_name,
    )

    assert input_phrase in prompt
    assert "Do not concatenate, copy, or preserve child section structures verbatim." in prompt
    assert f"CURRENT {level_name.upper()} INPUT (STRUCTURED" in prompt
    assert f"CURRENT {level_name.upper()} TEXT (IN JAPANESE)" not in prompt


def test_extract_summary_sections_parses_llm_style_output() -> None:
    output = """Overview:
Kaho starts the scene uncertain.

She regains her footing after talking with Sayaka.

Key Events:
- Kaho arrives at the club room.
- Sayaka asks what changed.

Character Developments:
- Kaho: becomes more willing to ask for help.

Continuity Facts:
- The club room remains the meeting point.

Important Terms:
- Kaho Hinoshita
- Sayaka Murano
"""

    sections = extract_summary_sections(output)

    assert sections["Overview"].startswith("Kaho starts")
    assert "She regains her footing" in sections["Overview"]
    assert sections["Key Events"].splitlines() == [
        "- Kaho arrives at the club room.",
        "- Sayaka asks what changed.",
    ]
    assert sections["Continuity Facts"] == "- The club room remains the meeting point."


def test_trim_previous_summary_context_keeps_only_overview_and_continuity() -> None:
    previous = """Overview:
Earlier overview.

Key Events:
- Should be excluded.

Character Developments:
- Should also be excluded.

Continuity Facts:
- Durable setup.

Important Terms:
- Kaho
"""

    trimmed = trim_previous_summary_context(previous)

    assert trimmed == "Overview:\nEarlier overview.\n\nContinuity Facts:\n- Durable setup."


def test_part_cache_fingerprint_uses_trimmed_previous_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    story_root = tmp_path / "story"
    first_path = _write_story_file(
        story_root,
        "103/第1話『花咲きたい！』/1.md",
        "花帆: こんにちは",
    )
    second_path = _write_story_file(
        story_root,
        "103/第1話『花咲きたい！』/2.md",
        "さやか: どうしたの？",
    )
    raw_nodes = [
        *StoryProcessor.process_file(first_path),
        *StoryProcessor.process_file(second_path),
    ]
    cache_file = tmp_path / "summaries_cache.json"

    def fake_generate(
        self: HierarchicalSummarizer,
        current_text: str,
        prev_summary: str | None = None,
        level_name: str = "Part",
    ) -> str:
        return f"""Overview:
{level_name} overview for {current_text}

Key Events:
- Excluded from previous context.

Character Developments:
- None

Continuity Facts:
- Durable context for {level_name}.

Important Terms:
- Kaho
"""

    monkeypatch.setattr(HierarchicalSummarizer, "_generate_rolling_summary", fake_generate)

    HierarchicalSummarizer(cache_context=_cache_context(first_path)).summarize_parts(
        raw_nodes,
        cache_file=str(cache_file),
    )

    cache = json.loads(cache_file.read_text(encoding="utf-8"))
    second_entry = cache["103|Main|第1話『花咲きたい！』|2"]
    expected_previous_context = """Overview:
Part overview for 花帆: こんにちは

Continuity Facts:
- Durable context for Part."""

    assert second_entry["inputs"]["previous_summary_hash"] == hash_text(
        expected_previous_context
    )


def test_default_generation_context_preserves_legacy_summary_fingerprint_shape(
    tmp_path: Path,
) -> None:
    path = _write_story_file(
        tmp_path / "story",
        "103/第1話『花咲きたい！』/1.md",
        "花帆: こんにちは",
    )
    context = _cache_context(path)

    inputs = HierarchicalSummarizer(cache_context=context)._base_fingerprint_inputs("part")

    assert "generation_provider" not in inputs
    assert "generation_model" not in inputs


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("parser_version", "parser-version-2"),
        ("chunker_version", "chunker-version-2"),
        ("chunker_config", {"min_chars": 1, "target_chars": 2, "max_chars": 3}),
        ("embedding_model", "embedding-model-2"),
        ("raw_evidence_schema_version", "raw-schema-2"),
    ],
)
def test_raw_evidence_fingerprint_changes_for_tracked_inputs(field: str, value: Any) -> None:
    baseline = {
        "parser_version": PARSER_VERSION,
        "chunker_version": CHUNKER_VERSION,
        "chunker_config": {
            "min_chars": MIN_USEFUL_CHARS,
            "target_chars": TARGET_CHUNK_CHARS,
            "max_chars": MAX_CHUNK_CHARS,
        },
        "embedding_model": "embedding-model-1",
        "raw_evidence_schema_version": RAW_EVIDENCE_SCHEMA_VERSION,
    }
    changed = {**baseline, field: value}

    assert stable_hash(changed) != stable_hash(baseline)


class FakePrunableCollection:
    def __init__(self) -> None:
        self.ids = {"chunk:stale:0-0", "chunk:current:0-0"}
        self.deleted: list[str] = []

    def get(self, include: list[str]) -> dict[str, list[str]]:
        assert include == []
        return {"ids": sorted(self.ids)}

    def delete(self, *, ids: list[str]) -> None:
        self.deleted.extend(ids)
        self.ids -= set(ids)


class FakeIngestCollection:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}

    def upsert(
        self,
        *,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        for record_id, document, metadata, embedding in zip(
            ids,
            documents,
            metadatas,
            embeddings,
            strict=True,
        ):
            self.records[record_id] = {
                "document": document,
                "metadata": metadata,
                "embedding": embedding,
            }

    def get(self, include: list[str]) -> dict[str, list[str]]:
        assert include == []
        return {"ids": sorted(self.records)}

    def delete(self, *, ids: list[str]) -> None:
        for record_id in ids:
            self.records.pop(record_id, None)


def test_pruning_deletes_records_not_emitted_by_current_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection = FakePrunableCollection()
    lexical_index = LexicalIndex(tmp_path / "lexical.db")
    metadata = {
        "arc_id": "103",
        "story_type": "Main",
        "episode_name": "第1話『花咲きたい！』",
        "part_name": "1",
        "file_path": "story/103/第1話『花咲きたい！』/1.md",
        "summary_level": 4,
    }
    lexical_index.upsert_records(
        ids=["chunk:stale:0-0", "chunk:current:0-0", "chunk:lexical-only:0-0"],
        documents=["old", "current", "lexical old"],
        metadatas=[metadata, metadata, metadata],
    )

    monkeypatch.setattr(cli, "get_chroma_collection", lambda: collection)

    pruned_count = cli._prune_stale_records(
        emitted_ids={"chunk:current:0-0"},
        lexical_index=lexical_index,
    )

    assert pruned_count == 2
    assert collection.deleted == ["chunk:stale:0-0"]
    assert collection.ids == {"chunk:current:0-0"}
    assert lexical_index.list_ids() == {"chunk:current:0-0"}


def test_reingest_after_rename_prunes_old_file_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    story_root = tmp_path / "story"
    old_path = _write_story_file(
        story_root,
        "103/第1話『花咲きたい！』/1.md",
        "花帆: こんにちは\n---\nさやか: どうしたの？",
    )
    collection = FakeIngestCollection()
    lexical_index = LexicalIndex(tmp_path / "lexical.db")

    monkeypatch.setattr(cli, "get_chroma_collection", lambda: collection)
    monkeypatch.setattr(cli, "embed_texts", lambda texts, *, task_type: [[1.0] for _ in texts])

    first_nodes = build_retrieval_chunks(StoryProcessor.process_file(old_path))
    first_ids = set(
        cli._upsert_story_nodes(
            first_nodes,
            progress_label="Embedding first run",
            lexical_index=lexical_index,
        )
    )
    cli._prune_stale_records(emitted_ids=first_ids, lexical_index=lexical_index)

    old_content = old_path.read_text(encoding="utf-8")
    old_path.unlink()
    new_path = _write_story_file(
        story_root,
        "103/第1話『花咲きたい！』/2.md",
        old_content,
    )

    second_nodes = build_retrieval_chunks(StoryProcessor.process_file(new_path))
    second_ids = set(
        cli._upsert_story_nodes(
            second_nodes,
            progress_label="Embedding second run",
            lexical_index=lexical_index,
        )
    )
    cli._prune_stale_records(emitted_ids=second_ids, lexical_index=lexical_index)

    assert set(collection.records) == second_ids
    assert all(
        record["metadata"]["file_path"] == str(new_path)
        for record in collection.records.values()
    )
    assert all(
        record["metadata"]["file_path"] != str(old_path)
        for record in collection.records.values()
    )


def test_reingest_after_rechunk_prunes_old_chunk_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    story_root = tmp_path / "story"
    path = _write_story_file(
        story_root,
        "103/第1話『花咲きたい！』/1.md",
        "\n---\n".join(
            [
                "花帆: aaaaaaaaaa",
                "さやか: bbbbbbbbbb",
                "花帆: cccccccccc",
                "さやか: dddddddddd",
                "花帆: eeeeeeeeee",
            ]
        ),
    )
    collection = FakeIngestCollection()
    lexical_index = LexicalIndex(tmp_path / "lexical.db")

    monkeypatch.setattr(cli, "get_chroma_collection", lambda: collection)
    monkeypatch.setattr(cli, "embed_texts", lambda texts, *, task_type: [[1.0] for _ in texts])

    raw_nodes = StoryProcessor.process_file(path)
    first_chunks = build_retrieval_chunks(raw_nodes, min_chars=35, target_chars=55, max_chars=80)
    first_ids = set(
        cli._upsert_story_nodes(
            first_chunks,
            progress_label="Embedding first chunks",
            lexical_index=lexical_index,
        )
    )
    cli._prune_stale_records(emitted_ids=first_ids, lexical_index=lexical_index)

    second_chunks = build_retrieval_chunks(raw_nodes, min_chars=1, target_chars=500, max_chars=500)
    second_ids = set(
        cli._upsert_story_nodes(
            second_chunks,
            progress_label="Embedding second chunks",
            lexical_index=lexical_index,
        )
    )
    cli._prune_stale_records(emitted_ids=second_ids, lexical_index=lexical_index)

    assert first_ids != second_ids
    assert set(collection.records) == second_ids
    assert first_ids.isdisjoint(collection.records)
