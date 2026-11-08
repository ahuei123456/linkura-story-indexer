import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from linkura_story_indexer.eval.answer_models import AnswerCaseSet
from linkura_story_indexer.eval.answer_runner import CorpusIndex
from linkura_story_indexer.eval.review_server import ReviewError, ReviewStore, serve

PART = "story/104/第8話『Not a marionette』/4.md"


def _corpus() -> CorpusIndex:
    return CorpusIndex(
        [
            {
                "file_path": PART.replace("/", "\\"),
                "scene_index": index,
                "text": f"れいかさん: line {index}\n綴理: reply {index}",
                "metadata": {"story_order": 100 + index},
            }
            for index in range(3)
        ]
    )


def _case(case_id: str = "c1", **overrides):
    data = {
        "id": case_id,
        "question": "Who is Reika-san?",
        "categories": ["alias"],
        "origin": "curated",
        "origin_note": "fixture",
        "expected_claims": [
            {
                "id": "a",
                "claim": "She works at the market.",
                "evidence": [{"file_path": PART, "scene_start": 0, "scene_end": 1, "quote": "line 0"}],
            }
        ],
        "rubric": "r",
    }
    data.update(overrides)
    return data


@pytest.fixture
def store(tmp_path: Path) -> ReviewStore:
    cases_path = tmp_path / "cases.json"
    suite = AnswerCaseSet.model_validate({"suite_version": "1", "cases": [_case()]})
    cases_path.write_text(suite.model_dump_json(), encoding="utf-8")
    return ReviewStore(cases_path, _corpus(), [tmp_path / "runs"])


def test_save_approve_and_edit_after_approval(store: ReviewStore):
    edited = _case(question="Who is れいかさん?")
    case, notice = store.save_case(edited, original_id="c1", approve=False)
    assert case.review_status == "draft" and notice is None

    case, _ = store.save_case(edited, original_id="c1", approve=True)
    assert store.load().cases[0].review_status == "reviewed"

    # Re-saving identical content keeps the approval.
    case, notice = store.save_case(edited, original_id="c1", approve=False)
    assert case.review_status == "reviewed" and notice is None

    changed = _case(question="Changed?")
    case, notice = store.save_case(changed, original_id="c1", approve=False)
    assert case.review_status == "draft"
    assert notice and "back to draft" in notice


def test_save_writes_tidy_json(store: ReviewStore):
    store.save_case(_case(review_note="checked"), original_id="c1", approve=True)
    data = json.loads(store.cases_path.read_text(encoding="utf-8"))
    case = data["cases"][0]
    assert case["review_note"] == "checked"
    assert "verified_by" not in case["expected_claims"][0]  # None fields are omitted
    assert not store.cases_path.with_suffix(".json.tmp").exists()


def test_save_rejects_invalid_cases(store: ReviewStore):
    with pytest.raises(ReviewError, match="needs at least one expected claim"):
        store.save_case(_case(expected_claims=[]), original_id="c1", approve=False)
    bad_scene = _case(
        expected_claims=[{"id": "a", "claim": "x", "evidence": [{"file_path": PART, "scene_start": 5, "scene_end": 5}]}]
    )
    with pytest.raises(ReviewError, match="no scene 5"):
        store.save_case(bad_scene, original_id="c1", approve=False)
    store.save_case(_case("c2"), original_id=None, approve=False)
    with pytest.raises(ReviewError, match="already uses id"):
        store.save_case(_case("c2"), original_id="c1", approve=False)
    with pytest.raises(ReviewError, match="no longer exists"):
        store.save_case(_case("c3"), original_id="gone", approve=False)


def test_rename_new_case_status_and_delete(store: ReviewStore):
    store.save_case(_case("renamed"), original_id="c1", approve=False)
    assert [case.id for case in store.load().cases] == ["renamed"]
    created, _ = store.save_case(_case("new", review_status="reviewed"), original_id=None, approve=False)
    assert created.review_status == "draft"
    assert store.set_status("new", "reviewed").review_status == "reviewed"
    store.delete_case("new")
    assert [case.id for case in store.load().cases] == ["renamed"]
    with pytest.raises(ReviewError):
        store.delete_case("new")


def test_passage_search_and_files(store: ReviewStore):
    passage = store.passage(PART, 0, 1)
    assert "line 0" in passage["text"] and "line 1" in passage["text"]
    assert store.passage(PART, 2, 3)["missing_scenes"] == [3]
    hits = store.search("line [12]", "104/第8話")
    assert [(hit["scene_index"], hit["line"]) for hit in hits] == [(1, "れいかさん: line 1"), (2, "れいかさん: line 2")]
    with pytest.raises(ReviewError, match="Invalid regular expression"):
        store.search("(")
    assert store.corpus.files() == [{"file_path": PART, "scene_count": 3}]


def test_runs_are_listed_without_traces(store: ReviewStore, tmp_path: Path):
    from linkura_story_indexer.eval.answer_models import (
        AnswerAggregate,
        AnswerEvalRun,
        AnswerRunConfig,
        CaseResult,
    )

    config = AnswerRunConfig(
        mode="agentic",
        cases_path="cases.json",
        suite_version="1",
        generation_provider="google",
        generation_model="gemini-3.8-flash",
        embedding_model="gemini-embedding-2",
        answer_prompt_version="answer-v3",
        summary_prompt_version="4",
        final_top_k=8,
        corpus_revision="test",
        code_revision="test",
        started_at="2026-11-07T00:00:00+00:00",
    )
    run = AnswerEvalRun(
        config=config,
        aggregate=AnswerAggregate(case_count=1, judged_count=0),
        results=[CaseResult(case_id="c1", categories=["alias"], review_status="draft", mode="agentic", trace={"big": 1})],
    )
    run_dir = tmp_path / "runs" / "baseline"
    run_dir.mkdir(parents=True)
    (run_dir / "agentic.json").write_text(run.model_dump_json(), encoding="utf-8")
    (run_dir / "notes.json").write_text("{}", encoding="utf-8")  # not a run; ignored
    runs = store.runs()
    assert len(runs) == 1
    assert runs[0]["config"]["mode"] == "agentic"
    assert "trace" not in runs[0]["results"]["c1"]


def test_http_round_trip(store: ReviewStore):
    server = serve(store, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(f"{base}/") as response:
            assert b"Answer benchmark review" in response.read()
        with urllib.request.urlopen(f"{base}/api/state") as response:
            state = json.loads(response.read())
        assert state["suite"]["cases"][0]["id"] == "c1"
        assert "unanswerable" in state["categories"]

        body = json.dumps({"case": _case(), "original_id": "c1", "approve": True}).encode()
        request = urllib.request.Request(f"{base}/api/case", data=body, method="PUT")
        request.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(request) as response:
            assert json.loads(response.read())["case"]["review_status"] == "reviewed"

        cross = urllib.request.Request(f"{base}/api/case", data=body, method="PUT")
        cross.add_header("Origin", "http://evil.example")
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(cross)
        assert error.value.code == 403

        bad = json.dumps({"case": _case(expected_claims=[]), "original_id": "c1"}).encode()
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(urllib.request.Request(f"{base}/api/case", data=bad, method="PUT"))
        assert error.value.code == 400
        assert "expected claim" in json.loads(error.value.read())["error"]
    finally:
        server.shutdown()
        server.server_close()


def test_save_keeps_not_checked_abstention(store: ReviewStore):
    store.save_case(_case(expect_abstention=None), original_id="c1", approve=True)
    raw = json.loads(store.cases_path.read_text(encoding="utf-8"))["cases"][0]
    assert raw["expect_abstention"] is None
    assert store.load().cases[0].expect_abstention is None
