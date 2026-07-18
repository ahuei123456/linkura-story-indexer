from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from linkura_story_indexer.indexer.chunker import build_retrieval_chunks
from linkura_story_indexer.indexer.processor import StoryProcessor
from linkura_story_indexer.indexer.source_store import SourceRecordStore
from linkura_story_indexer.query import engine as query_engine
from linkura_story_indexer.query.engine import RetrievalConfig, StoryQueryEngine
from linkura_story_indexer.query.tools import (
    DIALOGUE_COUNTING_UNIT,
    QUERY_TOOL_REGISTRY,
    CountDialogueInput,
    GetSceneInput,
    GetStateInput,
    GetSummariesInput,
    LookupGlossaryInput,
    SearchRawInput,
    SearchSummariesInput,
    build_query_toolset,
    count_dialogue,
    get_scene,
    get_state,
    get_summaries,
    lookup_glossary,
    vector_search_raw,
    vector_search_summaries,
)


def make_engine() -> StoryQueryEngine:
    # Bypass __init__ so these unit tests can exercise tool formatting without opening Chroma,
    # SQLite, or model-provider resources. Tests only rely on retrieval_config, glossary,
    # state_ledger, source_store, and monkeypatched retrieval methods.
    engine = StoryQueryEngine.__new__(StoryQueryEngine)
    engine.retrieval_config = RetrievalConfig(neighbor_scene_window=0)
    engine.glossary = {"characters": {"日野下花帆": "Kaho Hinoshita"}}
    engine.state_ledger = {}
    return engine


def raw_node(
    text: str = "花帆: raw scene",
    *,
    scene_start: int = 0,
    arc_id: str = "103",
) -> tuple[str, dict[str, Any]]:
    return (
        text,
        {
            "arc_id": arc_id,
            "story_type": "Main",
            "episode_name": "第1話『花咲きたい！』",
            "episode_number": 1,
            "part_name": "1",
            "summary_level": 4,
            "file_path": "story/103/第1話『花咲きたい！』/1.md",
            "scene_index": scene_start,
            "scene_start": scene_start,
            "scene_end": scene_start,
            "source_scene_count": 1,
            "canonical_story_order": scene_start,
            "parent_part_id": "103|Main|第1話『花咲きたい！』|1",
            "chunk_id": f"chunk:103:1:{scene_start}",
            "detected_speakers": "花帆",
        },
    )


def summary_node(
    text: str = "episode summary",
    *,
    summary_level: int = 2,
    arc_id: str = "103",
) -> tuple[str, dict[str, Any]]:
    return (
        text,
        {
            "arc_id": arc_id,
            "story_type": "Main",
            "episode_name": "第1話『花咲きたい！』",
            "episode_number": 1,
            "part_name": "1",
            "summary_level": summary_level,
            "parent_episode_id": "103|Main|第1話『花咲きたい！』",
        },
    )


@pytest.mark.parametrize(
    "model,args",
    [
        (SearchRawInput, {"query": "x", "top_k": 0}),
        (SearchRawInput, {"query": "x", "top_k": 1, "scene_start": -1}),
        (SearchRawInput, {"query": "x", "top_k": 1, "scene_start": 3, "scene_end": 2}),
        (SearchSummariesInput, {"query": "x", "top_k": 0}),
        (SearchSummariesInput, {"query": "x", "top_k": 1, "summary_level": 4}),
        (GetSummariesInput, {"episode": 1, "summary_level": 1}),
        (GetSummariesInput, {"part": "1", "summary_level": 2}),
        (GetSummariesInput, {"arc_id": " "}),
        (GetSummariesInput, {"story_type": " "}),
        (GetSummariesInput, {"part": " "}),
        (GetSummariesInput, {"episode": 0}),
        (GetSummariesInput, {"limit": 0}),
        (GetSummariesInput, {"limit": 101}),
        (GetSceneInput, {"file_path": "story.md", "scene_index": -1}),
        (GetStateInput, {"arc_id": ""}),
        (GetStateInput, {"arc_id": "   "}),
        (GetStateInput, {"arc_id": "103", "as_of_episode": 0}),
        (GetStateInput, {"arc_id": "103", "predicate": "not_a_predicate"}),
        (GetStateInput, {"arc_id": "103", "subject": " "}),
        (CountDialogueInput, {"speaker": ""}),
        (CountDialogueInput, {"speaker": "   "}),
        (CountDialogueInput, {"speaker": "梢＆慈"}),
        (CountDialogueInput, {"speaker": "花帆", "episode": 0}),
        (CountDialogueInput, {"speaker": "花帆", "arc_id": " "}),
    ],
)
def test_query_tool_inputs_reject_invalid_arguments(
    model: type[Any],
    args: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(args)


def test_vector_search_raw_returns_ranked_candidates_and_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = make_engine()
    seed = raw_node("weak seed", scene_start=0)
    exact = raw_node("花帆 talks about practice", scene_start=1)
    captured: list[dict[str, Any]] = []

    class FakeSourceStore:
        def chunk_ids_for_speaker(self, speaker: str) -> list[str]:
            assert speaker == "花帆"
            return ["chunk:103:1:0", "chunk:103:1:1"]

    engine.source_store = FakeSourceStore()

    def fake_hybrid_retrieve_trace(
        question: str,
        *,
        n_results: int,
        where: dict[str, Any] | None,
        query_embedding: list[float] | None,
        dense_unavailable_reason: str | None = None,
    ) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, query_engine.StageTrace]]:
        captured.append({"question": question, "n_results": n_results, "where": where})
        assert query_embedding == [0.1]
        assert dense_unavailable_reason is None
        return (
            [seed, exact],
            {
                "dense_raw": engine._trace_stage("dense_raw", []),
                "lexical_raw": engine._trace_stage("lexical_raw", []),
                "rrf_fusion": engine._trace_stage("rrf_fusion", []),
            },
        )

    monkeypatch.setattr(engine, "_query_embedding", lambda question: [0.1])
    monkeypatch.setattr(engine, "_hybrid_retrieve_trace", fake_hybrid_retrieve_trace)
    monkeypatch.setattr(engine, "_fetch_raw_text", lambda metadata: "")

    result = vector_search_raw(
        engine,
        SearchRawInput(
            query="What practice does 花帆 mention?",
            top_k=1,
            arc_id="103",
            episode=1,
            part="1",
            scene_start=1,
            speakers=["花帆"],
        ),
    )

    assert result.candidates[0].text == "花帆 talks about practice"
    assert result.candidates[0].citation_label == "103 · Episode 1 · Part 1 · Scene 2"
    assert result.trace_stages["final_top_k"].candidates is not None
    assert result.trace_stages["final_top_k"].candidates[0].rank == 1
    assert captured[0]["where"] == {
        "$and": [
            {"summary_level": 4},
            {"arc_id": "103"},
            {"episode_number": 1},
            {"part_name": "1"},
            {"scene_end": {"$gte": 1}},
            {"scene_start": {"$lte": 1}},
            {"chunk_id": {"$in": ["chunk:103:1:0", "chunk:103:1:1"]}},
        ]
    }


def test_vector_search_summaries_filters_level_and_arc(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = make_engine()
    summary = summary_node("Kaho starts school.", summary_level=2, arc_id="103")
    captured: list[dict[str, Any]] = []

    def fake_hybrid_retrieve_trace(
        question: str,
        *,
        n_results: int,
        where: dict[str, Any] | None,
        query_embedding: list[float] | None,
        dense_unavailable_reason: str | None = None,
    ) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, query_engine.StageTrace]]:
        captured.append({"n_results": n_results, "where": where})
        return (
            [summary],
            {
                "dense_raw": engine._trace_stage("dense_raw", []),
                "lexical_raw": engine._trace_stage("lexical_raw", []),
                "rrf_fusion": engine._trace_stage("rrf_fusion", []),
            },
        )

    monkeypatch.setattr(engine, "_query_embedding", lambda question: [0.1])
    monkeypatch.setattr(engine, "_hybrid_retrieve_trace", fake_hybrid_retrieve_trace)

    result = vector_search_summaries(
        engine,
        SearchSummariesInput(query="Kaho", top_k=3, summary_level=2, arc_id="103"),
    )

    assert [candidate.text for candidate in result.candidates] == ["Kaho starts school."]
    assert (
        result.candidates[0].citation_label
        == "103 · Main · Episode 1 · Part ALL_PARTS · summary_level 2"
    )
    assert captured == [
        {
            "n_results": 3,
            "where": {"$and": [{"summary_level": 2}, {"arc_id": "103"}]},
        }
    ]


def test_get_summaries_maps_filters_and_orders_before_limiting() -> None:
    engine = make_engine()
    calls: list[dict[str, Any]] = []

    class FakeCollection:
        def get(self, **kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs)
            return {
                "documents": ["part later", "episode", "part earlier"],
                "metadatas": [
                    {
                        "summary_level": 3,
                        "story_order": 30,
                        "arc_id": "103",
                        "story_type": "Main",
                        "episode_number": 2,
                        "part_name": "2",
                    },
                    {
                        "summary_level": 2,
                        "story_order": 20,
                        "arc_id": "103",
                        "story_type": "Main",
                        "episode_number": 2,
                    },
                    {
                        "summary_level": 3,
                        "story_order": 10,
                        "arc_id": "103",
                        "story_type": "Main",
                        "episode_number": 1,
                        "part_name": "1",
                    },
                ],
            }

    engine.collection = FakeCollection()

    result = get_summaries(
        engine,
        GetSummariesInput(
            arc_id=" 103 ",
            story_type=" Main ",
            episode=2,
            limit=2,
        ),
    )

    assert calls == [
        {
            "where": {
                "$and": [
                    {"summary_level": {"$in": [2, 3]}},
                    {"arc_id": "103"},
                    {"story_type": "Main"},
                    {"episode_number": 2},
                ]
            },
            "include": ["documents", "metadatas"],
        }
    ]
    assert [candidate.text for candidate in result.candidates] == ["episode", "part earlier"]
    assert [candidate.rank for candidate in result.candidates] == [1, 2]
    assert result.metadata == {
        "where": {
            "$and": [
                {"summary_level": {"$in": [2, 3]}},
                {"arc_id": "103"},
                {"story_type": "Main"},
                {"episode_number": 2},
            ]
        },
        "total_matched": 3,
        "truncated": True,
    }
    assert result.warnings == ["summaries truncated to limit 2 from 3 matches"]
    assert result.trace_stages == {}


@pytest.mark.parametrize(
    ("args", "expected_level_filter"),
    [
        ({}, {"$in": [1, 2, 3]}),
        ({"episode": 3}, {"$in": [2, 3]}),
        ({"part": "1"}, 3),
        ({"summary_level": 2}, 2),
    ],
)
def test_get_summaries_implicitly_narrows_summary_level(
    args: dict[str, Any], expected_level_filter: Any
) -> None:
    engine = make_engine()
    captured: list[dict[str, Any]] = []

    class FakeCollection:
        def get(self, **kwargs: Any) -> dict[str, Any]:
            captured.append(kwargs)
            return {"documents": [], "metadatas": []}

    engine.collection = FakeCollection()
    get_summaries(engine, GetSummariesInput.model_validate(args))

    where = captured[0]["where"]
    if "episode" in args or "part" in args:
        assert where["$and"][0] == {"summary_level": expected_level_filter}
    else:
        assert where == {"summary_level": expected_level_filter}


def test_get_summaries_warns_when_no_documents_match() -> None:
    engine = make_engine()

    class FakeCollection:
        def get(self, **kwargs: Any) -> dict[str, Any]:
            return {"documents": [], "metadatas": []}

    engine.collection = FakeCollection()
    result = get_summaries(engine, GetSummariesInput(arc_id="999"))

    assert result.candidates == []
    assert result.warnings == ["no summaries matched the given filters"]
    assert result.metadata["total_matched"] == 0
    assert result.metadata["truncated"] is False


def test_get_scene_returns_exact_source_text_and_structured_errors(tmp_path: Path) -> None:
    story_file = tmp_path / "story" / "103" / "第1話『花咲きたい！』" / "1.md"
    story_file.parent.mkdir(parents=True, exist_ok=True)
    story_file.write_text("花帆: scene zero\n---\nさやか: scene one", encoding="utf-8")
    raw_nodes = StoryProcessor.process_file(story_file)
    chunks = build_retrieval_chunks(raw_nodes, min_chars=1, target_chars=500, max_chars=500)
    store = SourceRecordStore(tmp_path / "source.db")
    store.replace_all(raw_nodes, chunks)

    engine = make_engine()
    engine.source_store = store

    found = get_scene(engine, GetSceneInput(file_path=str(story_file), scene_index=1))
    missing = get_scene(engine, GetSceneInput(file_path=str(story_file), scene_index=5))

    assert found.candidates[0].text == "さやか: scene one"
    assert found.candidates[0].metadata["scene_index"] == 1
    assert found.candidates[0].source_identity is not None
    assert missing.candidates == []
    assert missing.errors == ["scene not found"]


def test_lookup_glossary_resolves_terms_translations_and_aliases() -> None:
    engine = make_engine()

    canonical = lookup_glossary(engine, LookupGlossaryInput(term="日野下花帆"))
    translation = lookup_glossary(engine, LookupGlossaryInput(term="Kaho Hinoshita"))
    alias = lookup_glossary(engine, LookupGlossaryInput(term="花帆"))
    miss = lookup_glossary(engine, LookupGlossaryInput(term="unknown"))

    assert canonical.match_type == "canonical"
    assert translation.match_type == "translation"
    assert alias.match_type == "alias"
    assert alias.canonical_term == "日野下花帆"
    assert alias.translation == "Kaho Hinoshita"
    assert "花帆" in alias.aliases
    assert miss.match_type == "miss"
    assert miss.errors == ["glossary term not found: unknown"]


def test_build_query_toolset_registers_pydantic_schema_tools() -> None:
    engine = make_engine()
    toolset = build_query_toolset(engine)
    test_model = TestModel()
    agent = Agent(test_model)

    agent.run_sync("What tools are available?", toolsets=[toolset])

    request_parameters = test_model.last_model_request_parameters
    assert request_parameters is not None
    tools = request_parameters.function_tools
    by_name = {tool.name: tool for tool in tools}
    assert set(by_name) == {
        "vector_search_raw",
        "vector_search_summaries",
        "get_summaries",
        "get_scene",
        "lookup_glossary",
        "get_state",
        "count_dialogue",
    }
    vector_search_raw_schema = by_name["vector_search_raw"].parameters_json_schema
    assert {"query", "top_k", "scene_start", "scene_end"}.issubset(
        vector_search_raw_schema["properties"]
    )
    assert vector_search_raw_schema["properties"]["top_k"]["minimum"] == 1
    assert "OR semantics" in vector_search_raw_schema["properties"]["speakers"]["description"]
    get_state_schema = by_name["get_state"].parameters_json_schema
    assert {"arc_id", "as_of_episode", "subject", "predicate", "target"}.issubset(
        get_state_schema["properties"]
    )
    count_dialogue_schema = by_name["count_dialogue"].parameters_json_schema
    assert {"speaker", "arc_id", "episode", "part"}.issubset(count_dialogue_schema["properties"])
    assert set(QUERY_TOOL_REGISTRY) == set(by_name)
    assert QUERY_TOOL_REGISTRY["vector_search_raw"].input_model is SearchRawInput
    assert QUERY_TOOL_REGISTRY["vector_search_summaries"].input_model is SearchSummariesInput
    assert QUERY_TOOL_REGISTRY["get_summaries"].input_model is GetSummariesInput
    assert QUERY_TOOL_REGISTRY["get_state"].input_model is GetStateInput
    assert QUERY_TOOL_REGISTRY["count_dialogue"].input_model is CountDialogueInput


def ledger_fact(
    *,
    subject: str = "花帆",
    predicate: str = "role",
    object_: str = "school idol",
    target: str | None = None,
    arc: str = "103",
    episode: str = "第1話『花咲きたい！』",
    part: str = "1",
    scene: int = 0,
    valid_from: int = 0,
    valid_to: int | None = None,
) -> dict[str, Any]:
    return {
        "subject": subject,
        "predicate": predicate,
        "target": target,
        "object": object_,
        "confidence": 0.9,
        "extracted_quote": f"{subject}: {object_}",
        "arc": arc,
        "episode": episode,
        "part": part,
        "scene": scene,
        "valid_from": valid_from,
        "valid_to": valid_to,
        "file_path": f"story/{arc}/{episode}/{part}.md",
        "scene_index": scene,
    }


class FakeStoryOrderStore:
    def __init__(self, orders: dict[tuple[str, int | None], int]) -> None:
        self.orders = orders
        self.calls: list[tuple[str, int | None]] = []

    def max_story_order(self, *, arc_id: str, episode: int | None = None) -> int | None:
        self.calls.append((arc_id, episode))
        return self.orders.get((arc_id, episode))


def test_get_state_returns_full_history_in_deterministic_order() -> None:
    engine = make_engine()
    newer = ledger_fact(object_="club president", valid_from=7, scene=2)
    older = ledger_fact(object_="student", valid_from=2, valid_to=7)
    other_arc = ledger_fact(arc="104", object_="idol", valid_from=1)
    engine.state_ledger = {"facts": [newer, other_arc, older]}

    result = get_state(engine, GetStateInput(arc_id="103"))

    assert [fact.object for fact in result.facts] == ["student", "club president"]
    assert result.as_of_story_order is None
    assert result.errors == []
    first = result.facts[0]
    assert first.extracted_quote == "花帆: student"
    assert first.file_path == "story/103/第1話『花咲きたい！』/1.md"
    assert first.arc == "103"
    assert first.part == "1"
    assert first.scene_index == 0
    assert first.confidence == 0.9
    assert first.valid_to == 7


def test_get_state_applies_half_open_validity_interval() -> None:
    engine = make_engine()
    at_boundary = ledger_fact(object_="starts at boundary", valid_from=10)
    closed_at_boundary = ledger_fact(object_="closed at boundary", valid_from=1, valid_to=10)
    open_ended = ledger_fact(object_="still open", valid_from=3)
    later = ledger_fact(object_="not yet true", valid_from=11)
    engine.state_ledger = {"facts": [at_boundary, closed_at_boundary, open_ended, later]}
    store = FakeStoryOrderStore({("103", 2): 10})
    engine.source_store = store

    result = get_state(engine, GetStateInput(arc_id="103", as_of_episode=2))

    assert result.as_of_story_order == 10
    assert [fact.object for fact in result.facts] == ["still open", "starts at boundary"]
    assert store.calls == [("103", 2)]


def test_get_state_filters_subject_predicate_and_target() -> None:
    engine = make_engine()
    match = ledger_fact(
        subject="さやか",
        predicate="relationship",
        target="花帆",
        object_="close friend",
    )
    wrong_subject = ledger_fact(
        subject="花帆",
        predicate="relationship",
        target="さやか",
        object_="close friend",
    )
    wrong_predicate = ledger_fact(subject="さやか", predicate="status", object_="tired")
    engine.state_ledger = {"facts": [match, wrong_subject, wrong_predicate]}

    result = get_state(
        engine,
        GetStateInput(arc_id="103", subject="さやか", predicate="relationship", target="花帆"),
    )

    assert len(result.facts) == 1
    assert result.facts[0].subject == "さやか"
    assert result.facts[0].target == "花帆"


def test_get_state_reports_unknown_scope_and_skips_invalid_facts() -> None:
    engine = make_engine()
    engine.state_ledger = {"facts": [ledger_fact()]}
    engine.source_store = FakeStoryOrderStore({})

    missing_scope = get_state(engine, GetStateInput(arc_id="103", as_of_episode=9))
    assert missing_scope.errors == ["no source scenes found for arc 103 episode 9"]
    assert missing_scope.facts == []

    engine.state_ledger = {
        "facts": [
            {
                "arc": "103",
                "subject": "legacy_world_state",
                "predicate": "summary",
                "object": "old blob",
                "valid_from": 0,
                "valid_to": None,
            }
        ]
    }
    legacy = get_state(engine, GetStateInput(arc_id="103"))
    assert legacy.facts == []
    assert any("skipped ledger fact" in warning for warning in legacy.warnings)

    engine.state_ledger = {}
    empty = get_state(engine, GetStateInput(arc_id="103"))
    assert empty.facts == []
    assert "state ledger is empty or unavailable" in empty.warnings


def test_count_dialogue_counts_exact_turns_per_scope(tmp_path: Path) -> None:
    story_root = tmp_path / "story"
    fixture_files = [
        (
            "103/第1話『花咲きたい！』/1.md",
            1,
            "花帆: A\nさやか: a\n---\n梢＆慈: B\n花帆: b\n---\n全員: C\nさやか: c",
        ),
        ("103/第2話『つづき』/1.md", 2, "花帆: E\nさやか: e"),
        ("104/第1話『新学期』/1.md", 1, "花帆: F\nさやか: f"),
    ]
    raw_nodes = []
    for relative_path, episode_number, content in fixture_files:
        file_path = story_root / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        for node in StoryProcessor.process_file(file_path):
            node.metadata.episode_number = episode_number
            raw_nodes.append(node)
    chunks = build_retrieval_chunks(raw_nodes, min_chars=1, target_chars=500, max_chars=500)
    store = SourceRecordStore(tmp_path / "source.db")
    store.replace_all(raw_nodes, chunks)

    engine = make_engine()
    engine.source_store = store

    unscoped = count_dialogue(engine, CountDialogueInput(speaker="花帆"))
    assert unscoped.count == 4
    assert unscoped.counting_unit == DIALOGUE_COUNTING_UNIT
    assert unscoped.errors == []

    arc_scoped = count_dialogue(engine, CountDialogueInput(speaker="花帆", arc_id="103"))
    assert arc_scoped.count == 3

    episode_scoped = count_dialogue(
        engine, CountDialogueInput(speaker="花帆", arc_id="103", episode=1)
    )
    assert episode_scoped.count == 2

    part_scoped = count_dialogue(
        engine, CountDialogueInput(speaker="花帆", arc_id="103", episode=1, part="1")
    )
    assert part_scoped.count == 2

    composite = count_dialogue(engine, CountDialogueInput(speaker="梢", arc_id="103"))
    assert composite.count == 1

    collective = count_dialogue(engine, CountDialogueInput(speaker="全員", arc_id="103"))
    assert collective.count == 1

    empty_scope = count_dialogue(engine, CountDialogueInput(speaker="花帆", arc_id="999"))
    assert empty_scope.count == 0
    assert empty_scope.errors == []


def test_count_dialogue_normalizes_speaker_and_requires_source_store() -> None:
    normalized = CountDialogueInput(speaker="  花帆  ")
    assert normalized.speaker == "花帆"

    engine = make_engine()
    result = count_dialogue(engine, CountDialogueInput(speaker="花帆"))
    assert result.count == 0
    assert result.errors == [
        "dialogue counting unavailable: source store has no count_turns method"
    ]


def test_registry_dispatches_state_and_count_tools_with_direct_answers() -> None:
    engine = make_engine()
    engine.state_ledger = {"facts": [ledger_fact(valid_from=4)]}

    state_result = QUERY_TOOL_REGISTRY["get_state"].dispatcher(engine, GetStateInput(arc_id="103"))
    tool_output = state_result.metadata["tool_output"]
    assert tool_output["facts"][0]["subject"] == "花帆"
    assert tool_output["facts"][0]["extracted_quote"] == "花帆: school idol"
    assert "花帆 role: school idol" in state_result.metadata["direct_answer"]
    assert "quote: 花帆: school idol" in state_result.metadata["direct_answer"]
    assert state_result.errors == []

    empty_state = QUERY_TOOL_REGISTRY["get_state"].dispatcher(engine, GetStateInput(arc_id="104"))
    assert empty_state.metadata["direct_answer"] == "No state facts matched arc 104."

    class FakeCountStore:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any]]] = []

        def count_turns(self, speaker: str, **kwargs: Any) -> int:
            self.calls.append((speaker, kwargs))
            return 2

    store = FakeCountStore()
    engine.source_store = store
    count_result = QUERY_TOOL_REGISTRY["count_dialogue"].dispatcher(
        engine, CountDialogueInput(speaker="花帆", arc_id="103", episode=1)
    )
    assert store.calls == [("花帆", {"arc_id": "103", "episode": 1, "part": None})]
    assert count_result.metadata["tool_output"]["count"] == 2
    assert (
        count_result.metadata["direct_answer"] == "花帆 has 2 dialogue turns in arc 103, episode 1."
    )
