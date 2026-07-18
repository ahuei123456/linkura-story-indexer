from __future__ import annotations

import json

from linkura_story_indexer.query.agent import FixtureQueryAgent
from linkura_story_indexer.query.audit import (
    AuditFlag,
    AuditReport,
    FixtureAnswerAuditor,
    build_audit_payload,
    render_audit_flags,
)
from linkura_story_indexer.query.engine import RetrievalConfig, StoryQueryEngine


def make_engine(auditor: FixtureAnswerAuditor) -> StoryQueryEngine:
    engine = StoryQueryEngine.__new__(StoryQueryEngine)
    engine.retrieval_config = RetrievalConfig(routing_mode="agentic", audit_enabled=True)
    engine.glossary = {"characters": {"日野下花帆": "Kaho Hinoshita"}}
    engine.state_ledger = {
        "facts": [
            {
                "subject": "花帆",
                "predicate": "honorific_used_for",
                "target": "さやか",
                "object": "ちゃん",
                "arc": "103",
                "episode": "第1話",
                "part": "1",
                "scene": 0,
                "valid_from": 1,
                "valid_to": None,
                "confidence": 1.0,
                "extracted_quote": "さやかちゃん",
                "file_path": "story/103/第1話/1.md",
                "scene_index": 0,
            },
            {
                "subject": "other",
                "predicate": "status",
                "target": None,
                "object": "unscoped",
                "arc": "104",
                "episode": "第1話",
                "part": "1",
                "scene": 0,
                "valid_from": 1,
                "valid_to": None,
                "confidence": 1.0,
                "extracted_quote": "other",
                "file_path": "story/104/第1話/1.md",
                "scene_index": 0,
            },
        ]
    }
    engine.source_store = type("Store", (), {"iter_scenes": lambda self: []})()
    engine.query_agent = FixtureQueryAgent(answer="The draft answer.")
    engine.answer_auditor = auditor
    return engine


def test_audit_payload_contains_evidence_scoped_ledger_and_glossary() -> None:
    payload = build_audit_payload(
        "question",
        "draft answer",
        [{"citation_label": "103 · Scene 1", "text": "evidence"}],
        [{"arc": "103", "object": "active"}],
        {"characters": {"日野下花帆": "Kaho Hinoshita"}},
    )

    data = json.loads(payload)
    assert data["evidence"][0]["text"] == "evidence"
    assert data["state_facts"] == [{"arc": "103", "object": "active"}]
    assert data["glossary"]["characters"]["日野下花帆"] == "Kaho Hinoshita"


def test_fixture_auditor_flags_land_in_trace_and_render() -> None:
    report = AuditReport(
        flags=[
            AuditFlag(
                flag_type="retcon",
                excerpt="later state",
                rationale="outside the validity interval",
                evidence_ref="ledger:103",
            ),
            AuditFlag(
                flag_type="wrong_honorific",
                excerpt="さやかさん",
                rationale="ledger says ちゃん",
                evidence_ref="ledger:103",
            ),
            AuditFlag(
                flag_type="hallucinated_name",
                excerpt="Unknown",
                rationale="not in supplied evidence",
                evidence_ref="absent",
            ),
        ]
    )
    engine = make_engine(FixtureAnswerAuditor(report))

    trace = engine.retrieve_with_trace("What happened in 103?", answer_mode=True)

    assert trace.answer_text == "The draft answer."
    audit_stage = trace.stages["audit"]
    assert audit_stage.metadata["flag_count"] == 3
    assert {flag["flag_type"] for flag in audit_stage.metadata["flags"]} == {
        "retcon",
        "wrong_honorific",
        "hallucinated_name",
    }
    rendered = render_audit_flags(report)
    assert "Audit flags:" in rendered
    assert "wrong_honorific" in rendered
    assert "ledger:103" in rendered


def test_auditor_failure_is_recorded_without_changing_answer() -> None:
    engine = make_engine(
        FixtureAnswerAuditor(AuditReport(), error=RuntimeError("audit unavailable"))
    )

    trace = engine.retrieve_with_trace("What happened in 103?", answer_mode=True)

    assert trace.answer_text == "The draft answer."
    assert trace.stages["audit"].metadata["errors"] == ["audit unavailable"]


def test_query_appends_rendered_audit_for_opt_in_interactive_mode() -> None:
    report = AuditReport(
        flags=[
            AuditFlag(
                flag_type="hallucinated_name",
                excerpt="Unknown",
                rationale="absent from evidence",
                evidence_ref="absent",
            )
        ]
    )
    engine = make_engine(FixtureAnswerAuditor(report))

    answer = engine.query("What happened in 103?")

    assert answer.startswith("The draft answer.")
    assert "Audit flags:" in answer
    assert "hallucinated_name" in answer
