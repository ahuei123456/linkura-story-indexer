from pathlib import Path
from typing import Any

import pytest

from linkura_story_indexer.indexer import extractor
from linkura_story_indexer.indexer.extractor import StateExtractor
from linkura_story_indexer.indexer.processor import StoryProcessor
from linkura_story_indexer.indexer.source_store import SourceRecordStore
from linkura_story_indexer.models.state import (
    ExtractedStateFact,
    SceneStateExtraction,
)


class FakeRunResult:
    def __init__(self, output: SceneStateExtraction) -> None:
        self.output = output


class FakeAgent:
    def run_sync(self, prompt: str) -> FakeRunResult:
        if "さやかちゃん" in prompt:
            return FakeRunResult(
                SceneStateExtraction(
                    facts=[
                        ExtractedStateFact(
                            subject="花帆",
                            predicate="honorific_used_for:さやか",
                            object="ちゃん",
                            confidence=0.9,
                            extracted_quote="さやかちゃん",
                        )
                    ]
                )
            )
        if "さやかさん" in prompt:
            return FakeRunResult(
                SceneStateExtraction(
                    facts=[
                        ExtractedStateFact(
                            subject="花帆",
                            predicate="honorific_used_for:さやか",
                            object="さん",
                            confidence=0.9,
                            extracted_quote="さやかさん",
                        )
                    ]
                )
            )
        return FakeRunResult(SceneStateExtraction())


class FakeAgentWithBadQuote:
    def run_sync(self, prompt: str) -> FakeRunResult:
        return FakeRunResult(
            SceneStateExtraction(
                facts=[
                    ExtractedStateFact(
                        subject="花帆",
                        predicate="status",
                        object="present",
                        confidence=1.0,
                        extracted_quote="花帆",
                    ),
                    ExtractedStateFact(
                        subject="梢",
                        predicate="status",
                        object="present",
                        confidence=1.0,
                        extracted_quote="not in the scene",
                    ),
                ]
            )
        )


def test_state_extractor_uses_configured_generation_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []

    class InitOnlyFakeAgent:
        def __init__(
            self,
            model: Any,
            *,
            instructions: str,
            output_type: type[SceneStateExtraction],
        ) -> None:
            calls.append(
                {
                    "model": model,
                    "instructions": instructions,
                    "output_type": output_type,
                }
            )

    monkeypatch.setattr(extractor, "create_generation_model", lambda: "generation-model")
    monkeypatch.setattr(extractor, "Agent", InitOnlyFakeAgent)

    StateExtractor(source_db_path=":memory:")

    assert len(calls) == 1
    assert calls[0]["model"] == "generation-model"
    assert calls[0]["output_type"] is SceneStateExtraction
    assert "strict archivist" in calls[0]["instructions"]


def _write_story_file(root: Path, relative_path: str, content: str) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _source_store(tmp_path: Path, content: str) -> SourceRecordStore:
    story_root = tmp_path / "story"
    story_file = _write_story_file(
        story_root,
        "103/第1話『花咲きたい！』/1.md",
        content,
    )
    raw_nodes = StoryProcessor.process_file(story_file)
    for order, node in enumerate(raw_nodes, start=1):
        node.metadata.canonical_story_order = order
        node.metadata.story_order = order
        node.metadata.episode_number = 1
    store = SourceRecordStore(tmp_path / "source.db")
    store.replace_all(raw_nodes, [])
    return store


def _extractor(store: SourceRecordStore, agent: Any) -> StateExtractor:
    state_extractor = StateExtractor.__new__(StateExtractor)
    state_extractor.source_db_path = str(store.path)
    state_extractor.source_store = store
    state_extractor.agent = agent
    return state_extractor


def test_state_ledger_preserves_changed_honorifics_as_temporal_facts(tmp_path: Path) -> None:
    store = _source_store(tmp_path, "花帆: さやかちゃん、行こう\n---\n花帆: さやかさん、お願いします")
    state_extractor = _extractor(store, FakeAgent())

    ledger = state_extractor.extract_from_sources(str(tmp_path / "world_state.json"))

    assert [(fact.object, fact.valid_from, fact.valid_to) for fact in ledger.facts] == [
        ("ちゃん", 1, 2),
        ("さん", 2, None),
    ]


def test_state_ledger_drops_facts_without_source_quote_match(tmp_path: Path) -> None:
    store = _source_store(tmp_path, "花帆: ここにいるよ")
    state_extractor = _extractor(store, FakeAgentWithBadQuote())

    ledger = state_extractor.extract_from_sources(str(tmp_path / "world_state.json"))

    assert [fact.subject for fact in ledger.facts] == ["花帆"]
    assert ledger.facts[0].extracted_quote in store.iter_scenes()[0]["text"]


def test_state_extraction_is_deterministic_for_fixed_inputs(tmp_path: Path) -> None:
    store = _source_store(tmp_path, "花帆: さやかちゃん、行こう\n---\n花帆: さやかさん、お願いします")

    first = _extractor(store, FakeAgent()).extract_from_sources(str(tmp_path / "first.json"))
    second = _extractor(store, FakeAgent()).extract_from_sources(str(tmp_path / "second.json"))

    assert first.model_dump() == second.model_dump()
