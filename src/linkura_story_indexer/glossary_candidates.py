import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sudachipy import Dictionary, SplitMode

from .indexer.parser import StoryParser
from .lexical import glossary_alias_groups

DEFAULT_GLOSSARY_CANDIDATE_FILE = "glossary_candidates.json"
GLOSSARY_CANDIDATE_SCHEMA_VERSION = 1

_JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff々〆〤ー]")
_KATAKANA_RE = re.compile(r"^[\u30a0-\u30ffー・!！A-Za-z0-9]+$")
_NOISE_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)


@dataclass(frozen=True)
class CandidateExample:
    file_path: str
    line_number: int
    line: str


@dataclass
class CandidateAccumulator:
    term: str
    suggested_category: str
    frequency: int = 0
    files: set[str] = field(default_factory=set)
    examples: list[CandidateExample] = field(default_factory=list)


def load_glossary_terms(path: Path) -> set[str]:
    if not path.exists():
        return set()

    with path.open(encoding="utf-8") as file:
        glossary = json.load(file)

    terms = set()
    if not isinstance(glossary, dict):
        return terms

    for aliases in glossary_alias_groups(glossary):
        terms.update(aliases)

    for category_terms in glossary.values():
        if not isinstance(category_terms, dict):
            continue
        for source_term, english_term in category_terms.items():
            terms.add(str(source_term))
            terms.add(str(english_term))
    return terms


def extract_glossary_candidates(
    *,
    story_dir: Path,
    glossary_file: Path,
    output_file: Path,
    min_count: int = 2,
    max_examples: int = 3,
    max_files: int = 10,
    include_katakana_terms: bool = False,
    include_existing: bool = False,
) -> list[dict[str, Any]]:
    if min_count < 1:
        raise ValueError("min_count must be at least 1")
    if max_examples < 1:
        raise ValueError("max_examples must be at least 1")
    if max_files < 1:
        raise ValueError("max_files must be at least 1")

    existing_terms = set() if include_existing else load_glossary_terms(glossary_file)
    accumulators: dict[str, CandidateAccumulator] = {}
    tokenizer = Dictionary().create()

    for path in sorted(story_dir.rglob("*.md"), key=lambda item: str(item)):
        text = path.read_text(encoding="utf-8")
        relative_path = path.as_posix()

        for line_number, line in enumerate(text.splitlines(), start=1):
            line_terms = _candidate_terms_for_line(line, tokenizer, include_katakana_terms)
            for term, category in _dedupe_line_terms(line_terms):
                _record_candidate(
                    accumulators,
                    term=term,
                    suggested_category=category,
                    file_path=relative_path,
                    line_number=line_number,
                    line=line,
                    max_examples=max_examples,
                )

    candidates = [
        _candidate_to_json(accumulator, max_files=max_files)
        for accumulator in accumulators.values()
        if accumulator.frequency >= min_count and accumulator.term not in existing_terms
    ]
    candidates.sort(
        key=lambda candidate: (
            str(candidate["suggested_category"]),
            -int(candidate["frequency"]),
            str(candidate["term"]),
        )
    )

    output = {
        "schema_version": GLOSSARY_CANDIDATE_SCHEMA_VERSION,
        "story_dir": story_dir.as_posix(),
        "glossary_file": glossary_file.as_posix(),
        "min_count": min_count,
        "max_files": max_files,
        "include_katakana_terms": include_katakana_terms,
        "include_existing": include_existing,
        "candidates": candidates,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return candidates


def _candidate_terms_for_line(
    line: str,
    tokenizer: Any,
    include_katakana_terms: bool,
) -> list[tuple[str, str]]:
    terms = []
    speaker, _ = StoryParser.parse_script_line(line)
    if speaker:
        terms.append((speaker, "characters"))
    terms.extend(_terms_from_line(line, tokenizer, include_katakana_terms))
    return terms


def _dedupe_line_terms(terms: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    deduped = []
    seen = set()
    categories_by_priority = {"characters": 0, "locations_and_terms": 1, "units": 2}
    sorted_terms = sorted(
        terms,
        key=lambda item: categories_by_priority.get(item[1], 99),
    )
    for term, category in sorted_terms:
        cleaned = _clean_term(term)
        if cleaned in seen:
            continue
        deduped.append((cleaned, category))
        seen.add(cleaned)
    return deduped


def _terms_from_line(
    line: str,
    tokenizer: Any,
    include_katakana_terms: bool,
) -> list[tuple[str, str]]:
    terms = []
    tokens = list(tokenizer.tokenize(line, SplitMode.C))

    noun_run: list[Any] = []
    for token in tokens:
        if _is_candidate_token(token, include_katakana_terms):
            noun_run.append(token)
            continue
        terms.extend(_terms_from_token_run(noun_run))
        noun_run = []
    terms.extend(_terms_from_token_run(noun_run))

    unique_terms = []
    seen = set()
    for term, category in terms:
        cleaned = _clean_term(term)
        if not _valid_term(cleaned) or cleaned in seen:
            continue
        unique_terms.append((cleaned, category))
        seen.add(cleaned)
    return unique_terms


def _terms_from_token_run(tokens: list[Any]) -> list[tuple[str, str]]:
    if not tokens:
        return []

    terms = []
    if len(tokens) > 1:
        combined = "".join(str(token.surface()) for token in tokens)
        terms.append((combined, _category_for_tokens(tokens)))

    for token in tokens:
        surface = str(token.surface())
        terms.append((surface, _category_for_tokens([token])))
    return terms


def _is_candidate_token(token: Any, include_katakana_terms: bool) -> bool:
    surface = _clean_term(str(token.surface()))
    if not _valid_term(surface):
        return False

    pos = tuple(str(part) for part in token.part_of_speech())
    if len(pos) >= 2 and pos[0] == "名詞" and pos[1] == "固有名詞":
        return True
    if include_katakana_terms and _KATAKANA_RE.fullmatch(surface) and len(surface) >= 3:
        return True
    return False


def _category_for_tokens(tokens: Iterable[Any]) -> str:
    pos_values = [tuple(str(part) for part in token.part_of_speech()) for token in tokens]
    if any(len(pos) >= 4 and pos[2] == "人名" for pos in pos_values):
        return "characters"
    if any(len(pos) >= 4 and pos[2] in {"地名", "組織"} for pos in pos_values):
        return "locations_and_terms"

    return "locations_and_terms"


def _record_candidate(
    accumulators: dict[str, CandidateAccumulator],
    *,
    term: str,
    suggested_category: str,
    file_path: str,
    line_number: int,
    line: str,
    max_examples: int,
) -> None:
    cleaned = _clean_term(term)
    if not _valid_term(cleaned):
        return

    accumulator = accumulators.setdefault(
        cleaned,
        CandidateAccumulator(term=cleaned, suggested_category=suggested_category),
    )
    accumulator.frequency += 1
    accumulator.files.add(file_path)
    if len(accumulator.examples) < max_examples:
        accumulator.examples.append(
            CandidateExample(
                file_path=file_path,
                line_number=line_number,
                line=line.strip(),
            )
        )


def _clean_term(term: str) -> str:
    return term.strip(" \t\r\n。、，,.「」『』（）()[]【】")


def _valid_term(term: str) -> bool:
    if len(term) < 2:
        return False
    if _NOISE_ONLY_RE.fullmatch(term):
        return False
    return bool(_JAPANESE_RE.search(term))


def _candidate_to_json(accumulator: CandidateAccumulator, max_files: int = 10) -> dict[str, Any]:
    files = sorted(accumulator.files)
    return {
        "term": accumulator.term,
        "english": "",
        "suggested_category": accumulator.suggested_category,
        "frequency": accumulator.frequency,
        "file_count": len(files),
        "files": files[:max_files],
        "examples": [
            {
                "file_path": example.file_path,
                "line_number": example.line_number,
                "line": example.line,
            }
            for example in accumulator.examples
        ],
    }


def category_counts(candidates: Iterable[dict[str, Any]]) -> Counter[str]:
    return Counter(str(candidate["suggested_category"]) for candidate in candidates)
