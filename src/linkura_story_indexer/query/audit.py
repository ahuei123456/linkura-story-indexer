from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from linkura_story_indexer.database import create_generation_model, get_generation_model_name

AuditFlagType = Literal["retcon", "wrong_honorific", "hallucinated_name"]


class AuditFlag(BaseModel):
    flag_type: AuditFlagType
    excerpt: str
    rationale: str
    evidence_ref: str


class AuditReport(BaseModel):
    flags: list[AuditFlag] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


def _audit_instructions() -> str:
    return (
        "Audit a draft answer against the supplied source evidence, State Ledger, and official "
        "Glossary. Return a structured AuditReport. Only flag concrete contradictions or "
        "unsupported proper names; do not penalize a cautious answer for omitting details.\n\n"
        "Flag classes:\n"
        "- retcon: the answer contradicts a State Ledger fact's temporal validity interval, "
        "for example treating a later state as true before valid_from or after valid_to.\n"
        "- wrong_honorific: the answer assigns or claims an honorific that contradicts the "
        "State Ledger or the official Glossary.\n"
        "- hallucinated_name: the answer introduces a proper name absent from the supplied "
        "evidence, official Glossary, and State Ledger. Common grammar is not a proper name.\n\n"
        "For each flag include the shortest relevant answer excerpt, a concise rationale, and "
        "an evidence_ref pointing to the supporting source or saying that the required source "
        "is absent. If there are no concrete issues, return an empty flags list."
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_jsonable(item) for item in value]
    return value


def build_audit_payload(
    question: str,
    answer: str,
    evidence: Sequence[Any],
    state_facts: Sequence[Any],
    glossary: Mapping[str, Any] | None,
) -> str:
    """Build a stable, complete audit input without depending on model message history."""
    payload = {
        "question": question,
        "answer": answer,
        "evidence": _jsonable(evidence),
        "state_facts": _jsonable(state_facts),
        "glossary": _jsonable(glossary or {}),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


class AnswerAuditor:
    def __init__(self, model_name: str | None = None, *, model: Any | None = None) -> None:
        self.model_name = model_name or os.getenv("LINKURA_AUDIT_MODEL") or get_generation_model_name()
        self.model = model

    def _run_model(self, payload: str) -> AuditReport:
        agent: Agent[None, AuditReport] = Agent(
            self.model or create_generation_model(self.model_name),
            output_type=AuditReport,
            instructions=_audit_instructions(),
        )
        return agent.run_sync(payload).output

    def audit(
        self,
        question: str,
        answer: str,
        evidence: Sequence[Any],
        state_facts: Sequence[Any],
        glossary: Mapping[str, Any] | None,
    ) -> AuditReport:
        return self._run_model(
            build_audit_payload(question, answer, evidence, state_facts, glossary)
        )

    def run(
        self,
        question: str,
        answer: str,
        evidence: Sequence[Any],
        state_facts: Sequence[Any],
        glossary: Mapping[str, Any] | None,
    ) -> AuditReport:
        return self.audit(
            question,
            answer,
            evidence=evidence,
            state_facts=state_facts,
            glossary=glossary,
        )


class FixtureAnswerAuditor(AnswerAuditor):
    def __init__(
        self,
        report: AuditReport | Mapping[str, Any],
        *,
        model_name: str = "fixture-auditor",
        error: Exception | None = None,
    ) -> None:
        super().__init__(model_name=model_name)
        self.report = report if isinstance(report, AuditReport) else AuditReport.model_validate(report)
        self.error = error

    def _run_model(self, payload: str) -> AuditReport:
        if self.error is not None:
            raise self.error
        return self.report


def render_audit_flags(report: AuditReport) -> str:
    lines: list[str] = []
    if report.flags:
        lines.append("Audit flags:")
        for flag in report.flags:
            lines.append(
                f"- {flag.flag_type}: {flag.excerpt} — {flag.rationale} "
                f"(evidence: {flag.evidence_ref})"
            )
    if report.errors:
        lines.append("Audit errors:")
        lines.extend(f"- {error}" for error in report.errors)
    if not lines:
        return "Audit: clean."
    return "\n".join(lines)
