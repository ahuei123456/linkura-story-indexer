from linkura_story_indexer.query.analysis import (
    CHRONOLOGY_INTENT,
    EXACT_EVIDENCE_INTENT,
    QUANTITATIVE_INTENT,
    SUMMARY_INTENT,
    analyze_query,
)


def test_query_analysis_extracts_temporal_episode_constraint() -> None:
    analysis = analyze_query("What did Kaho know before episode 12?")

    assert analysis.temporal_constraint is not None
    assert analysis.temporal_constraint.operator == "before"
    assert analysis.temporal_constraint.episode_number == 12
    assert analysis.episode_number is None
    assert analysis.intent_bucket == CHRONOLOGY_INTENT


def test_query_analysis_extracts_side_story_constraint() -> None:
    analysis = analyze_query("What happens in the side stories?")

    assert analysis.story_type == "Side"
    assert analysis.intent_bucket == SUMMARY_INTENT


def test_query_analysis_extracts_arc_constraint() -> None:
    analysis = analyze_query("What happens in 103?")

    assert analysis.arc_ids == ("103",)
    assert analysis.intent_bucket == SUMMARY_INTENT


def test_query_analysis_extracts_part_scene_constraint_as_zero_based() -> None:
    analysis = analyze_query("What happens in ABYSS scene 2?")

    assert analysis.part_name == "ABYSS"
    assert analysis.scene_constraint is not None
    assert analysis.scene_constraint.start == 1
    assert analysis.scene_constraint.end == 1
    assert analysis.intent_bucket == EXACT_EVIDENCE_INTENT


def test_query_analysis_extracts_scene_range_as_zero_based() -> None:
    analysis = analyze_query("Summarize scenes 3-7.")

    assert analysis.scene_constraint is not None
    assert analysis.scene_constraint.start == 2
    assert analysis.scene_constraint.end == 6


def test_query_analysis_extracts_character_aliases_from_glossary() -> None:
    analysis = analyze_query(
        "What does Kaho do?",
        {"characters": {"日野下花帆": "Kaho Hinoshita"}},
    )

    assert "Kaho Hinoshita" in analysis.character_names
    assert "日野下花帆" in analysis.character_names


def test_query_analysis_classifies_quantitative_intent() -> None:
    analysis = analyze_query("How many times does Kaho speak in 103?")

    assert analysis.intent_bucket == QUANTITATIVE_INTENT
