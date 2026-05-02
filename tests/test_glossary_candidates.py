import json
from pathlib import Path

from linkura_story_indexer.glossary_candidates import extract_glossary_candidates


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_extract_glossary_candidates_filters_existing_terms(tmp_path: Path) -> None:
    story_dir = tmp_path / "story"
    glossary_file = tmp_path / "glossary.json"
    output_file = tmp_path / "glossary_candidates.json"
    _write(
        story_dir / "103" / "第1話" / "1.md",
        "藤島慈: こんにちは\n桂城泉: よろしく\n---\n藤島慈: またね\n桂城泉: はい",
    )
    glossary_file.write_text(
        json.dumps({"characters": {"藤島慈": "Megumi Fujishima"}}, ensure_ascii=False),
        encoding="utf-8",
    )

    candidates = extract_glossary_candidates(
        story_dir=story_dir,
        glossary_file=glossary_file,
        output_file=output_file,
        min_count=2,
    )

    terms = {candidate["term"] for candidate in candidates}
    assert "藤島慈" not in terms
    assert "桂城泉" in terms

    data = json.loads(output_file.read_text(encoding="utf-8"))
    izumi = next(candidate for candidate in data["candidates"] if candidate["term"] == "桂城泉")
    assert izumi["english"] == ""
    assert izumi["suggested_category"] == "characters"
    assert izumi["frequency"] >= 2
    assert izumi["examples"][0]["line_number"] == 2


def test_extract_glossary_candidates_can_include_existing_terms(tmp_path: Path) -> None:
    story_dir = tmp_path / "story"
    glossary_file = tmp_path / "glossary.json"
    output_file = tmp_path / "glossary_candidates.json"
    _write(
        story_dir / "103" / "第1話" / "1.md",
        "藤島慈: こんにちは\n藤島慈: またね",
    )
    glossary_file.write_text(
        json.dumps({"characters": {"藤島慈": "Megumi Fujishima"}}, ensure_ascii=False),
        encoding="utf-8",
    )

    candidates = extract_glossary_candidates(
        story_dir=story_dir,
        glossary_file=glossary_file,
        output_file=output_file,
        min_count=2,
        include_existing=True,
    )

    assert "藤島慈" in {candidate["term"] for candidate in candidates}


def test_extract_glossary_candidates_filters_generated_aliases(tmp_path: Path) -> None:
    story_dir = tmp_path / "story"
    glossary_file = tmp_path / "glossary.json"
    output_file = tmp_path / "glossary_candidates.json"
    _write(
        story_dir / "103" / "第1話" / "1.md",
        "花帆: こんにちは\n花帆: またね",
    )
    glossary_file.write_text(
        json.dumps({"characters": {"日野下花帆": "Kaho Hinoshita"}}, ensure_ascii=False),
        encoding="utf-8",
    )

    candidates = extract_glossary_candidates(
        story_dir=story_dir,
        glossary_file=glossary_file,
        output_file=output_file,
        min_count=2,
    )

    assert "花帆" not in {candidate["term"] for candidate in candidates}
