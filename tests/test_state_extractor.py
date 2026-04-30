import json
from pathlib import Path
from typing import Any

import pytest

from linkura_story_indexer.indexer import extractor
from linkura_story_indexer.indexer.extractor import StateExtractor
from linkura_story_indexer.models.state import WorldState


def test_state_extractor_uses_configured_generation_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []

    class FakeAgent:
        def __init__(self, model: Any, *, instructions: str, output_type: type[WorldState]):
            calls.append(
                {
                    "model": model,
                    "instructions": instructions,
                    "output_type": output_type,
                }
            )

    monkeypatch.setattr(extractor, "create_generation_model", lambda: "generation-model")
    monkeypatch.setattr(extractor, "Agent", FakeAgent)

    StateExtractor()

    assert len(calls) == 1
    assert calls[0]["model"] == "generation-model"
    assert calls[0]["output_type"] is WorldState
    assert "strict archivist" in calls[0]["instructions"]


def test_state_extractor_reads_structured_summary_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_file = tmp_path / "summaries_cache.json"
    cache_file.write_text(
        json.dumps(
            {
                "EPISODE|103|Main|第1話": {
                    "summary": "Kaho visits the club.",
                    "fingerprint": "abc",
                }
            }
        ),
        encoding="utf-8",
    )
    seen: list[tuple[str, str]] = []

    class FakeAgent:
        def __init__(self, model: Any, *, instructions: str, output_type: type[WorldState]):
            pass

    def fake_extract(self: StateExtractor, arc_id: str, summary: str) -> WorldState:
        seen.append((arc_id, summary))
        return WorldState(arc_id=arc_id, characters=[], locations=[], important_groups=[])

    monkeypatch.setattr(extractor, "create_generation_model", lambda: "generation-model")
    monkeypatch.setattr(extractor, "Agent", FakeAgent)
    monkeypatch.setattr(StateExtractor, "_extract_facts_from_summary", fake_extract)

    StateExtractor(cache_file=str(cache_file)).extract_from_cache(
        output_file=str(tmp_path / "world_state.json")
    )

    assert seen == [("103", "Kaho visits the club.")]
