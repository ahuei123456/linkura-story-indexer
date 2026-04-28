from typing import Any

from linkura_story_indexer import database
from linkura_story_indexer.query import engine as query_engine
from linkura_story_indexer.query.analysis import analyze_query
from linkura_story_indexer.query.engine import (
    INSUFFICIENT_SOURCE_CONTEXT,
    RetrievalConfig,
    StoryQueryEngine,
)


def make_engine() -> StoryQueryEngine:
    engine = StoryQueryEngine.__new__(StoryQueryEngine)
    engine.retrieval_config = RetrievalConfig(neighbor_scene_window=0)
    engine.state_ledger = {
        "103": {"characters": [{"name": "Kaho Hinoshita"}]},
        "104": {"characters": [{"name": "Sayaka Murano"}]},
    }
    engine.glossary = None
    return engine


def test_system_prompt_restores_raw_source_claim_and_compacts_ledger():
    engine = make_engine()

    prompt = engine._build_system_prompt({"103"})

    assert "based strictly on the provided raw source text" in prompt
    assert "Some retrieved context may be generated summaries" not in prompt
    assert '{"characters":[{"name":"Kaho Hinoshita"}]}' in prompt
    assert '\n  "characters"' not in prompt
    assert "YEAR 104 FACTS" not in prompt


def test_state_ledger_arc_ids_prefers_explicit_question_arc():
    engine = make_engine()

    arc_ids = engine._state_ledger_arc_ids("What happened in 103?", {"104"})

    assert arc_ids == {"103"}


def test_state_ledger_arc_ids_falls_back_to_retrieved_arcs():
    engine = make_engine()

    arc_ids = engine._state_ledger_arc_ids("What happened to Kaho?", {"103", "104"})

    assert arc_ids == {"103", "104"}


def test_retrieve_uses_query_embedding_task_type(monkeypatch):
    engine = StoryQueryEngine.__new__(StoryQueryEngine)
    calls: list[dict[str, Any]] = []

    class FakeCollection:
        def query(self, **kwargs: Any) -> dict[str, list[list[Any]]]:
            calls.append(kwargs)
            return {
                "documents": [["summary"]],
                "metadatas": [[{"arc_id": "103", "summary_level": 3}]],
            }

    def fake_embed_texts(texts: list[str], *, task_type: str) -> list[list[float]]:
        calls.append({"texts": texts, "task_type": task_type})
        return [[0.1, 0.2]]

    engine.collection = FakeCollection()
    monkeypatch.setattr(query_engine, "embed_texts", fake_embed_texts)

    retrieved = engine._retrieve("question")

    assert retrieved == [("summary", {"arc_id": "103", "summary_level": 3})]
    assert calls[0] == {"texts": ["question"], "task_type": database.RETRIEVAL_QUERY}
    assert calls[1]["query_embeddings"] == [[0.1, 0.2]]


def test_retrieval_config_rejects_invalid_rrf_k():
    try:
        RetrievalConfig(rrf_k=0)
    except ValueError as exc:
        assert "rrf_k must be at least 1" in str(exc)
    else:
        raise AssertionError("RetrievalConfig accepted an invalid RRF k")


def test_rrf_fusion_combines_fixed_ranked_lists():
    engine = make_engine()
    a = raw_node("a", scene_start=0)
    b = raw_node("b", scene_start=1)
    c = raw_node("c", scene_start=2)

    fused = engine._rrf_fuse([[a, b], [b, c]], k=1)

    assert fused == [b, a, c]


def test_hybrid_retrieve_rrf_fuses_and_dedupes_dense_and_lexical(monkeypatch):
    engine = make_engine()
    dense_node = (
        "dense raw",
        {
            "summary_level": 4,
            "file_path": "story/part.md",
            "scene_start": 0,
            "scene_end": 1,
        },
    )
    lexical_duplicate = (
        "lexical raw",
        {
            "summary_level": 4,
            "file_path": "story/part.md",
            "scene_start": 0,
            "scene_end": 1,
        },
    )
    lexical_new = (
        "lexical second",
        {
            "summary_level": 4,
            "file_path": "story/part.md",
            "scene_start": 2,
            "scene_end": 2,
        },
    )

    monkeypatch.setattr(engine, "_retrieve", lambda question, **kwargs: [dense_node])
    monkeypatch.setattr(
        engine,
        "_lexical_retrieve",
        lambda question, **kwargs: [lexical_duplicate, lexical_new],
    )

    assert engine._hybrid_retrieve("expanded question") == [dense_node, lexical_new]


def raw_node(
    document: str,
    *,
    scene_start: int,
    scene_end: int | None = None,
    parent_part_id: str = "103|Main|第3話『テスト』|2",
    file_path: str = "story/part.md",
    detected_speakers: str = "",
) -> tuple[str, dict[str, Any]]:
    return (
        document,
        {
            "arc_id": "103",
            "story_type": "Main",
            "episode_name": "第3話『テスト』",
            "part_name": "2",
            "summary_level": 4,
            "file_path": file_path,
            "scene_index": scene_start,
            "scene_start": scene_start,
            "scene_end": scene_start if scene_end is None else scene_end,
            "source_scene_count": 1,
            "canonical_story_order": 30,
            "parent_part_id": parent_part_id,
            "detected_speakers": detected_speakers,
        },
    )


def test_query_uses_configured_candidate_counts(monkeypatch):
    engine = make_engine()
    engine.retrieval_config = RetrievalConfig(
        routing_candidate_count=21,
        raw_candidate_count=41,
        summary_child_candidate_count=31,
        neighbor_scene_window=0,
        final_top_k=5,
    )
    calls: list[dict[str, Any]] = []

    summary_node = (
        "part summary",
        {
            "arc_id": "103",
            "summary_level": 3,
            "parent_part_id": "103|Main|第3話『テスト』|2",
        },
    )
    raw_child = raw_node("花帆: raw scene", scene_start=4)

    def fake_hybrid_retrieve(
        question: str,
        *,
        n_results: int,
        where: dict[str, Any] | None = None,
        query_embedding: list[float] | None = None,
    ) -> list[tuple[str, dict[str, Any]]]:
        calls.append({"n_results": n_results, "where": where})
        assert query_embedding == [0.1]
        if where == {"summary_level": 3}:
            return [summary_node]
        if where == {"summary_level": 4}:
            return []
        if where == {
            "$and": [
                {"summary_level": 4},
                {"parent_part_id": "103|Main|第3話『テスト』|2"},
            ]
        }:
            return [raw_child]
        return []

    monkeypatch.setattr(engine, "_hybrid_retrieve", fake_hybrid_retrieve)
    monkeypatch.setattr(engine, "_query_embedding", lambda question: [0.1])
    monkeypatch.setattr(engine, "_answer_from_raw_evidence", lambda question, nodes: "answered")

    assert engine.query("What happened?") == "answered"
    assert calls == [
        {"n_results": 21, "where": {"summary_level": 1}},
        {"n_results": 21, "where": {"summary_level": 2}},
        {"n_results": 21, "where": {"summary_level": 3}},
        {"n_results": 41, "where": {"summary_level": 4}},
        {
            "n_results": 31,
            "where": {
                "$and": [
                    {"summary_level": 4},
                    {"parent_part_id": "103|Main|第3話『テスト』|2"},
                ]
            },
        },
    ]


def test_tiered_retrieve_dispatches_each_summary_tier_and_raw(monkeypatch):
    engine = make_engine()
    calls: list[dict[str, Any]] = []

    def fake_hybrid_retrieve(
        question: str,
        *,
        n_results: int,
        where: dict[str, Any] | None = None,
        query_embedding: list[float] | None = None,
    ) -> list[tuple[str, dict[str, Any]]]:
        calls.append({"n_results": n_results, "where": where})
        assert query_embedding == [0.1]
        return []

    monkeypatch.setattr(engine, "_hybrid_retrieve", fake_hybrid_retrieve)
    monkeypatch.setattr(engine, "_query_embedding", lambda question: [0.1])

    assert engine._tiered_retrieve("question") == []
    assert calls == [
        {"n_results": 20, "where": {"summary_level": 1}},
        {"n_results": 20, "where": {"summary_level": 2}},
        {"n_results": 20, "where": {"summary_level": 3}},
        {"n_results": 40, "where": {"summary_level": 4}},
    ]


def test_analysis_where_applies_side_story_filters_to_specific_tiers(monkeypatch):
    engine = make_engine()
    calls: list[dict[str, Any]] = []
    analysis = analyze_query("What happens in the side stories?")

    def fake_hybrid_retrieve(
        question: str,
        *,
        n_results: int,
        where: dict[str, Any] | None = None,
        query_embedding: list[float] | None = None,
    ) -> list[tuple[str, dict[str, Any]]]:
        calls.append({"n_results": n_results, "where": where})
        assert query_embedding == [0.1]
        return []

    monkeypatch.setattr(engine, "_hybrid_retrieve", fake_hybrid_retrieve)

    assert engine._tiered_retrieve("question", query_embedding=[0.1], analysis=analysis) == []
    assert calls == [
        {"n_results": 20, "where": {"summary_level": 1}},
        {
            "n_results": 20,
            "where": {"$and": [{"summary_level": 2}, {"story_type": "Side"}]},
        },
        {
            "n_results": 20,
            "where": {"$and": [{"summary_level": 3}, {"story_type": "Side"}]},
        },
        {
            "n_results": 40,
            "where": {"$and": [{"summary_level": 4}, {"story_type": "Side"}]},
        },
    ]


def test_explicit_scene_where_uses_zero_based_span_overlap() -> None:
    engine = make_engine()
    analysis = analyze_query("ABYSS scene 2")

    where = engine._where_for_analysis(
        analysis,
        summary_level=4,
        include_scene_constraint=True,
    )

    assert where == {
        "$and": [
            {"summary_level": 4},
            {"part_name": "ABYSS"},
            {"scene_start": {"$lte": 1}},
            {"scene_end": {"$gte": 1}},
        ]
    }


def test_scene_point_constraint_matches_containing_coalesced_chunk() -> None:
    engine = make_engine()
    analysis = analyze_query("scene 2")
    nodes = [
        raw_node("scene 1", scene_start=0, scene_end=0),
        raw_node("scenes 2-4", scene_start=1, scene_end=3),
        raw_node("scene 5", scene_start=4, scene_end=4),
    ]

    filtered = engine._filter_raw_nodes_by_analysis(nodes, analysis)

    assert filtered == [nodes[1]]


def test_scene_range_constraint_matches_overlapping_coalesced_chunks() -> None:
    engine = make_engine()
    analysis = analyze_query("scenes 3-7")
    nodes = [
        raw_node("scene 1", scene_start=0, scene_end=0),
        raw_node("scenes 2-4", scene_start=1, scene_end=3),
        raw_node("scenes 5-6", scene_start=4, scene_end=5),
    ]

    filtered = engine._filter_raw_nodes_by_analysis(nodes, analysis)

    assert filtered == [nodes[1], nodes[2]]


def test_temporal_filter_resolves_to_numeric_story_order() -> None:
    engine = make_engine()

    class FakeCollection:
        def get(self, **kwargs: Any) -> dict[str, list[dict[str, Any]]]:
            assert kwargs["where"] == {"$and": [{"summary_level": 4}, {"episode_number": 12}]}
            return {"metadatas": [{"story_order": 120}, {"story_order": 125}]}

    engine.collection = FakeCollection()
    analysis = analyze_query("What did Kaho know before episode 12?")

    where = engine._where_for_analysis(analysis, summary_level=4)

    assert where == {
        "$and": [
            {"summary_level": 4},
            {"story_order": {"$lt": 120}},
        ]
    }


def test_semantic_boundary_keeps_prior_story_order(monkeypatch) -> None:
    engine = make_engine()
    analysis = analyze_query("scenes before Ruri falls asleep")
    before = raw_node("before", scene_start=0)
    boundary = raw_node("boundary", scene_start=1)
    after = raw_node("after", scene_start=2)
    before[1]["story_order"] = 10
    boundary[1]["story_order"] = 20
    after[1]["story_order"] = 30

    monkeypatch.setattr(
        engine,
        "_rank_raw_candidates",
        lambda question, expanded_question, raw_nodes, seed_nodes: [boundary],
    )

    filtered = engine._filter_before_semantic_boundary(
        "scenes before Ruri falls asleep",
        "expanded",
        [before, boundary, after],
        [boundary],
        analysis,
    )

    assert filtered == [before]


def test_tier_two_fanout_retrieves_child_raw_evidence(monkeypatch):
    engine = make_engine()
    calls: list[dict[str, Any]] = []
    tier_two_summary = (
        "episode summary",
        {
            "summary_level": 2,
            "parent_episode_id": "103|Main|第3話『テスト』",
        },
    )
    child = raw_node(
        "child scene",
        scene_start=2,
        parent_part_id="103|Main|第3話『テスト』|2",
    )

    def fake_hybrid_retrieve(
        question: str,
        *,
        n_results: int,
        where: dict[str, Any] | None = None,
        query_embedding: list[float] | None = None,
    ) -> list[tuple[str, dict[str, Any]]]:
        calls.append({"n_results": n_results, "where": where})
        assert query_embedding == [0.1]
        return [child]

    monkeypatch.setattr(engine, "_hybrid_retrieve", fake_hybrid_retrieve)
    monkeypatch.setattr(engine, "_query_embedding", lambda question: [0.1])

    expanded = engine._expand_summaries_to_raw_scenes("question", [tier_two_summary])

    assert expanded == [child]
    assert calls == [
        {
            "n_results": 30,
            "where": {
                "$and": [
                    {"summary_level": 4},
                    {"parent_episode_id": "103|Main|第3話『テスト』"},
                ]
            },
        }
    ]


def test_summary_fanout_preserves_coalesced_child_spans(monkeypatch):
    engine = make_engine()
    tier_one_summary = (
        "year summary",
        {
            "summary_level": 1,
            "parent_year_id": "103",
        },
    )
    coalesced_child = raw_node(
        "coalesced child scene span",
        scene_start=4,
        scene_end=7,
        parent_part_id="103|Main|第3話『テスト』|2",
    )

    monkeypatch.setattr(
        engine,
        "_hybrid_retrieve",
        lambda question, **kwargs: [coalesced_child],
    )
    monkeypatch.setattr(engine, "_query_embedding", lambda question: [0.1])

    expanded = engine._expand_summaries_to_raw_scenes("question", [tier_one_summary])

    assert expanded == [coalesced_child]
    assert engine._scene_span(expanded[0][1]) == (4, 7)


def test_query_expands_summary_hits_to_raw_scenes(monkeypatch):
    engine = make_engine()
    query_calls: list[dict[str, Any]] = []
    embedding_calls: list[list[str]] = []
    agent_prompts: list[str] = []

    class FakeCollection:
        def query(self, **kwargs: Any) -> dict[str, list[list[Any]]]:
            query_calls.append(kwargs)
            if kwargs.get("where") == {"summary_level": 3}:
                return {
                    "documents": [["part summary"]],
                    "metadatas": [
                        [
                            {
                                "arc_id": "103",
                                "story_type": "Main",
                                "episode_name": "第3話『テスト』",
                                "part_name": "2",
                                "summary_level": 3,
                                "parent_part_id": "103|Main|第3話『テスト』|2",
                            }
                        ]
                    ],
                }
            if kwargs.get("where") == {
                "$and": [
                    {"summary_level": 4},
                    {"parent_part_id": "103|Main|第3話『テスト』|2"},
                ]
            }:
                return {
                    "documents": [["花帆: raw scene"]],
                    "metadatas": [
                        [
                            {
                                "arc_id": "103",
                                "story_type": "Main",
                                "episode_name": "第3話『テスト』",
                                "part_name": "2",
                                "summary_level": 4,
                                "file_path": "missing.md",
                                "scene_index": 4,
                                "canonical_story_order": 30,
                                "parent_part_id": "103|Main|第3話『テスト』|2",
                            }
                        ]
                    ],
                }
            return {
                "documents": [[]],
                "metadatas": [[]],
            }

    class FakeAgent:
        def run_sync(self, prompt: str) -> Any:
            agent_prompts.append(prompt)

            class Result:
                output = "answered from raw scene"

            return Result()

    def fake_embed_texts(texts: list[str], *, task_type: str) -> list[list[float]]:
        embedding_calls.append(texts)
        return [[0.1]]

    monkeypatch.setattr(query_engine, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(query_engine, "create_text_agent", lambda system_prompt: FakeAgent())
    engine.collection = FakeCollection()

    answer = engine.query("What happened?")

    assert answer == "answered from raw scene"
    assert embedding_calls == [["What happened?"]]
    assert len(query_calls) == 5
    assert query_calls[4]["where"] == {
        "$and": [
            {"summary_level": 4},
            {"parent_part_id": "103|Main|第3話『テスト』|2"},
        ]
    }
    assert "SUMMARY:" not in agent_prompts[0]
    assert "花帆: raw scene" in agent_prompts[0]
    assert "103 · Episode 3 · Part 2 · Scene 5" in agent_prompts[0]


def test_query_reports_insufficient_source_context_without_raw_evidence(monkeypatch):
    engine = make_engine()
    query_calls: list[dict[str, Any]] = []
    embedding_calls: list[list[str]] = []
    agent_called = False

    class FakeCollection:
        def query(self, **kwargs: Any) -> dict[str, list[list[Any]]]:
            query_calls.append(kwargs)
            if kwargs.get("where") == {"summary_level": 3}:
                return {
                    "documents": [["part summary"]],
                    "metadatas": [
                        [
                            {
                                "arc_id": "103",
                                "summary_level": 3,
                                "parent_part_id": "103|Main|第3話『テスト』|2",
                            }
                        ]
                    ],
                }
            return {
                "documents": [[]],
                "metadatas": [[]],
            }

    def fake_create_text_agent(system_prompt: str) -> Any:
        nonlocal agent_called
        agent_called = True
        return object()

    def fake_embed_texts(texts: list[str], *, task_type: str) -> list[list[float]]:
        embedding_calls.append(texts)
        return [[0.1]]

    monkeypatch.setattr(query_engine, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(query_engine, "create_text_agent", fake_create_text_agent)
    engine.collection = FakeCollection()

    answer = engine.query("What happened?")

    assert answer == INSUFFICIENT_SOURCE_CONTEXT
    assert embedding_calls == [["What happened?"]]
    assert len(query_calls) == 5
    assert agent_called is False


def test_neighbor_expansion_pulls_bounded_scene_window(monkeypatch):
    engine = make_engine()
    engine.retrieval_config = RetrievalConfig(neighbor_scene_window=2)
    hit = raw_node("scene 5", scene_start=5)
    part_nodes = [raw_node(f"scene {index}", scene_start=index) for index in range(10)]

    monkeypatch.setattr(engine, "_raw_nodes_for_part", lambda question, metadata: part_nodes)

    expanded = engine._expand_raw_neighbors("question", [hit])
    expanded_spans = {engine._scene_span(metadata) for _, metadata in expanded}

    assert expanded_spans == {(3, 3), (4, 4), (5, 5), (6, 6), (7, 7)}


def test_neighbor_expansion_dedupes_overlapping_windows(monkeypatch):
    engine = make_engine()
    engine.retrieval_config = RetrievalConfig(neighbor_scene_window=1)
    hits = [raw_node("scene 5", scene_start=5), raw_node("scene 6", scene_start=6)]
    part_nodes = [raw_node(f"scene {index}", scene_start=index) for index in range(4, 8)]

    monkeypatch.setattr(engine, "_raw_nodes_for_part", lambda question, metadata: part_nodes)

    expanded = engine._expand_raw_neighbors("question", hits)
    keys = [engine._node_key(document, metadata) for document, metadata in expanded]

    assert len(keys) == len(set(keys))
    assert {engine._scene_span(metadata) for _, metadata in expanded} == {
        (4, 4),
        (5, 5),
        (6, 6),
        (7, 7),
    }


def test_rank_raw_candidates_prefers_exact_and_speaker_matches():
    engine = make_engine()
    seed = raw_node("unrelated direct candidate", scene_start=0)
    exact_match = raw_node("花帆 talks about practice", scene_start=1, detected_speakers="花帆")

    ranked = engine._rank_raw_candidates(
        "What does 花帆 say?",
        "What does 花帆 say? Kaho Hinoshita / 花帆",
        [seed, exact_match],
        [seed],
    )

    assert ranked[0] == exact_match


def test_analysis_where_can_filter_raw_chunks_by_source_store_speaker() -> None:
    engine = make_engine()

    class FakeSourceStore:
        def chunk_ids_for_speaker(self, speaker: str) -> list[str]:
            if speaker == "花帆":
                return ["chunk:part:0-1"]
            return []

    engine.source_store = FakeSourceStore()
    engine.glossary = {"characters": {"花帆": "Kaho Hinoshita"}}
    analysis = analyze_query("What does Kaho say?", engine.glossary)

    where = engine._where_for_analysis(analysis, summary_level=4)

    assert where == {"$and": [{"summary_level": 4}, {"chunk_id": {"$in": ["chunk:part:0-1"]}}]}


def test_structured_who_said_query_uses_source_turn_records() -> None:
    engine = make_engine()

    class FakeSourceStore:
        def turns_matching_text(self, text: str) -> list[dict[str, Any]]:
            assert text == "行こう"
            return [{"speaker": "泉", "text": "「行こう」"}]

        def count_turns(self, speaker: str) -> int:
            raise AssertionError("count_turns should not be called")

    engine.source_store = FakeSourceStore()
    analysis = analyze_query("Who said 「行こう」?")

    assert engine._structured_answer("Who said 「行こう」?", analysis) == "泉 said it."


def test_structured_quantitative_query_counts_turns_by_speaker() -> None:
    engine = make_engine()

    class FakeSourceStore:
        def count_turns(self, speaker: str) -> int:
            assert speaker == "花帆"
            return 3

    engine.source_store = FakeSourceStore()
    engine.glossary = {"characters": {"花帆": "Kaho Hinoshita"}}
    analysis = analyze_query("How many turns does Kaho have?", engine.glossary)

    assert (
        engine._structured_answer("How many turns does Kaho have?", analysis)
        == "花帆 has 3 dialogue turns in the indexed source records."
    )


def test_query_caps_final_raw_evidence_to_configured_top_k(monkeypatch):
    engine = make_engine()
    engine.retrieval_config = RetrievalConfig(neighbor_scene_window=0, final_top_k=5)
    raw_nodes = [raw_node(f"scene {index}", scene_start=index) for index in range(10)]
    captured_counts = []

    monkeypatch.setattr(engine, "_hybrid_retrieve", lambda question, **kwargs: raw_nodes)
    monkeypatch.setattr(engine, "_query_embedding", lambda question: [0.1])

    def fake_answer(question: str, nodes: list[tuple[str, dict[str, Any]]]) -> str:
        captured_counts.append(len(nodes))
        return "answered"

    monkeypatch.setattr(engine, "_answer_from_raw_evidence", fake_answer)

    assert engine.query("What happened?") == "answered"
    assert captured_counts == [5]


def test_fetch_raw_text_returns_only_requested_scene(tmp_path):
    engine = make_engine()
    story_file = tmp_path / "part.md"
    story_file.write_text("scene zero\n---\nscene one\n---\nscene two", encoding="utf-8")

    raw_text = engine._fetch_raw_text({"file_path": str(story_file), "scene_index": 1})

    assert raw_text == "scene one"


def test_fetch_raw_text_returns_requested_scene_span(tmp_path):
    engine = make_engine()
    story_file = tmp_path / "part.md"
    story_file.write_text("scene zero\n---\nscene one\n---\nscene two", encoding="utf-8")

    raw_text = engine._fetch_raw_text(
        {"file_path": str(story_file), "scene_start": 0, "scene_end": 1}
    )

    assert raw_text == "scene zero\n\n---\n\nscene one"


def test_citation_label_and_metadata_are_split():
    engine = make_engine()
    metadata = {
        "arc_id": "103",
        "story_type": "Main",
        "episode_name": "第3話『テスト』",
        "part_name": "2",
        "file_path": "story/103/第3話『テスト』/2.md",
        "scene_index": 4,
        "canonical_story_order": 30,
    }

    label = engine._citation_label(metadata)
    citation_metadata = engine._citation_metadata(metadata)

    assert label == "103 · Episode 3 · Part 2 · Scene 5"
    assert "story/" not in label
    assert citation_metadata == {
        "file_path": "story/103/第3話『テスト』/2.md",
        "scene_index": 4,
        "scene_start": None,
        "scene_end": None,
        "source_scene_count": None,
        "canonical_story_order": 30,
    }


def test_citation_label_and_metadata_handle_scene_spans():
    engine = make_engine()
    metadata = {
        "arc_id": "103",
        "story_type": "Main",
        "episode_name": "第3話『テスト』",
        "part_name": "2",
        "file_path": "story/103/第3話『テスト』/2.md",
        "scene_index": 0,
        "scene_start": 0,
        "scene_end": 6,
        "source_scene_count": 7,
        "canonical_story_order": 30,
    }

    label = engine._citation_label(metadata)
    citation_metadata = engine._citation_metadata(metadata)

    assert label == "103 · Episode 3 · Part 2 · Scene 1-7"
    assert citation_metadata == {
        "file_path": "story/103/第3話『テスト』/2.md",
        "scene_index": 0,
        "scene_start": 0,
        "scene_end": 6,
        "source_scene_count": 7,
        "canonical_story_order": 30,
    }
