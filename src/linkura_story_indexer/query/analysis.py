import re
from dataclasses import dataclass
from typing import Any

from ..lexical import glossary_alias_groups

SUMMARY_INTENT = "summary"
EXACT_EVIDENCE_INTENT = "exact_evidence"
COMPARISON_INTENT = "comparison"
CHRONOLOGY_INTENT = "chronology"
QUANTITATIVE_INTENT = "quantitative"


@dataclass(frozen=True)
class SceneConstraint:
    start: int
    end: int
    part_name: str | None = None


@dataclass(frozen=True)
class TemporalConstraint:
    operator: str
    episode_number: int | None = None
    arc_id: str | None = None
    phrase: str = ""


@dataclass(frozen=True)
class QueryAnalysis:
    arc_ids: tuple[str, ...] = ()
    story_type: str | None = None
    episode_number: int | None = None
    part_name: str | None = None
    scene_constraint: SceneConstraint | None = None
    semantic_boundary: str | None = None
    temporal_constraint: TemporalConstraint | None = None
    character_names: tuple[str, ...] = ()
    intent_bucket: str = EXACT_EVIDENCE_INTENT

    @property
    def has_scope(self) -> bool:
        return bool(
            self.arc_ids
            or self.story_type
            or self.episode_number is not None
            or self.part_name
            or self.temporal_constraint
        )


_ARC_RE = re.compile(r"\b\d{3}\b")
_EPISODE_RE = re.compile(r"(?:episode|ep\.?|第)\s*(\d+)", re.IGNORECASE)
_TEMPORAL_EPISODE_RE = re.compile(
    r"\b(?P<operator>before|prior to|as of|until|through|after|since)\s+"
    r"(?:episode|ep\.?|第)\s*(?P<episode>\d+)",
    re.IGNORECASE,
)
_TEMPORAL_ARC_RE = re.compile(
    r"\b(?P<operator>before|prior to|as of|until|through|after|since)\s+"
    r"(?:year|arc)?\s*(?P<arc>\d{3})\b",
    re.IGNORECASE,
)
_SCENE_RE = re.compile(
    r"\b(?:(?P<part>[A-Za-z][A-Za-z0-9_.-]*)\s+)?scenes?\s+"
    r"(?P<start>\d+)(?:\s*[-–]\s*(?P<end>\d+))?",
    re.IGNORECASE,
)
_SEMANTIC_BOUNDARY_RE = re.compile(
    r"\b(?:scenes?\s+)?(?P<operator>before|after)\s+(?P<boundary>.+)",
    re.IGNORECASE,
)


def analyze_query(
    question: str,
    glossary: dict[str, dict[str, str]] | None = None,
) -> QueryAnalysis:
    question_lower = question.casefold()
    temporal_constraint = _extract_temporal_constraint(question)
    scene_constraint = _extract_scene_constraint(question)
    semantic_boundary = _extract_semantic_boundary(question, scene_constraint, temporal_constraint)

    part_name = scene_constraint.part_name if scene_constraint else None
    story_type = _extract_story_type(question_lower)
    episode_number = _extract_episode_number(question, temporal_constraint)
    character_names = _extract_character_names(question, glossary)

    return QueryAnalysis(
        arc_ids=tuple(_ordered_unique(_ARC_RE.findall(question))),
        story_type=story_type,
        episode_number=episode_number,
        part_name=part_name,
        scene_constraint=scene_constraint,
        semantic_boundary=semantic_boundary,
        temporal_constraint=temporal_constraint,
        character_names=tuple(character_names),
        intent_bucket=_intent_bucket(question_lower, scene_constraint, semantic_boundary),
    )


def _ordered_unique(values: list[str]) -> list[str]:
    unique = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        unique.append(value)
        seen.add(value)
    return unique


def _normalized_temporal_operator(operator: str) -> str:
    normalized = operator.casefold()
    if normalized in {"before", "prior to"}:
        return "before"
    if normalized in {"as of", "until", "through"}:
        return "as_of"
    return "after"


def _extract_temporal_constraint(question: str) -> TemporalConstraint | None:
    match = _TEMPORAL_EPISODE_RE.search(question)
    if match:
        return TemporalConstraint(
            operator=_normalized_temporal_operator(match.group("operator")),
            episode_number=int(match.group("episode")),
            phrase=match.group(0),
        )

    match = _TEMPORAL_ARC_RE.search(question)
    if match:
        return TemporalConstraint(
            operator=_normalized_temporal_operator(match.group("operator")),
            arc_id=match.group("arc"),
            phrase=match.group(0),
        )

    if re.search(r"\bas of (?:this|that) point\b", question, re.IGNORECASE):
        return TemporalConstraint(operator="as_of", phrase="as of this point")

    return None


def _extract_scene_constraint(question: str) -> SceneConstraint | None:
    match = _SCENE_RE.search(question)
    if not match:
        return None

    start = max(int(match.group("start")) - 1, 0)
    end = max(int(match.group("end") or match.group("start")) - 1, start)
    return SceneConstraint(
        start=start,
        end=end,
        part_name=match.group("part"),
    )


def _extract_semantic_boundary(
    question: str,
    scene_constraint: SceneConstraint | None,
    temporal_constraint: TemporalConstraint | None,
) -> str | None:
    if scene_constraint is not None or temporal_constraint is not None:
        return None

    match = _SEMANTIC_BOUNDARY_RE.search(question)
    if not match:
        return None

    boundary = match.group("boundary").strip(" ?.")
    if not boundary:
        return None
    return boundary


def _extract_story_type(question_lower: str) -> str | None:
    if "side story" in question_lower or "side stories" in question_lower:
        return "Side"
    if "main story" in question_lower or "main stories" in question_lower:
        return "Main"
    return None


def _extract_episode_number(
    question: str,
    temporal_constraint: TemporalConstraint | None,
) -> int | None:
    match = _EPISODE_RE.search(question)
    if not match:
        return None
    episode_number = int(match.group(1))
    if (
        temporal_constraint is not None
        and temporal_constraint.episode_number == episode_number
        and temporal_constraint.phrase
    ):
        return None
    return episode_number


def _extract_character_names(
    question: str,
    glossary: dict[str, dict[str, str]] | None,
) -> list[str]:
    question_lower = question.casefold()
    names = []
    for aliases in glossary_alias_groups(glossary):
        if any(alias in question or alias.casefold() in question_lower for alias in aliases):
            names.extend(aliases)
    return _ordered_unique(names)


def _intent_bucket(
    question_lower: str,
    scene_constraint: SceneConstraint | None,
    semantic_boundary: str | None,
) -> str:
    if re.search(r"\b(how many|count|number of|total)\b", question_lower):
        return QUANTITATIVE_INTENT
    if re.search(r"\b(compare|contrast|difference|versus| vs )\b", question_lower):
        return COMPARISON_INTENT
    if scene_constraint is not None or re.search(r"\b(quote|exact|evidence|scene)\b", question_lower):
        return EXACT_EVIDENCE_INTENT
    if semantic_boundary is not None or re.search(r"\b(when|first|before|after|chronology)\b", question_lower):
        return CHRONOLOGY_INTENT
    if re.search(r"\b(summary|summarize|overview|recap|what happen(?:s|ed)?)\b", question_lower):
        return SUMMARY_INTENT
    return EXACT_EVIDENCE_INTENT


def analysis_debug_dict(analysis: QueryAnalysis) -> dict[str, Any]:
    return {
        "arc_ids": list(analysis.arc_ids),
        "story_type": analysis.story_type,
        "episode_number": analysis.episode_number,
        "part_name": analysis.part_name,
        "scene_constraint": analysis.scene_constraint,
        "semantic_boundary": analysis.semantic_boundary,
        "temporal_constraint": analysis.temporal_constraint,
        "character_names": list(analysis.character_names),
        "intent_bucket": analysis.intent_bucket,
    }
