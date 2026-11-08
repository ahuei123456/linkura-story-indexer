"""Local web app for reviewing answer cases and browsing benchmark runs.

Serves one page plus a small JSON API on 127.0.0.1. Edits are validated with the same
pydantic models the runner uses and written back to the case file atomically.
"""

import json
import os
import re
import threading
from collections.abc import Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any, get_args
from urllib.parse import parse_qs, unquote, urlparse

from pydantic import ValidationError

from linkura_story_indexer.eval.answer_models import (
    AnswerCase,
    AnswerCaseSet,
    AnswerCategory,
    AnswerEvalRun,
    CaseOrigin,
)
from linkura_story_indexer.eval.answer_runner import CorpusIndex, validate_case_sources

CASE_CONTENT_EXCLUDE = {"review_status"}


class ReviewError(ValueError):
    """A user-facing validation or lookup failure."""


def _validation_messages(exc: ValidationError) -> list[str]:
    messages = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        messages.append(f"{location}: {error['msg']}" if location else error["msg"])
    return messages


def _case_content(case: AnswerCase) -> dict[str, Any]:
    return case.model_dump(mode="json", exclude=CASE_CONTENT_EXCLUDE)


class ReviewStore:
    def __init__(
        self,
        cases_path: str | Path,
        corpus: CorpusIndex,
        run_dirs: Sequence[str | Path] | None = None,
    ):
        self.cases_path = Path(cases_path)
        self.corpus = corpus
        self.run_dirs = [Path(path) for path in (run_dirs or [])]
        self._lock = threading.Lock()

    # Cases -----------------------------------------------------------------

    def load(self) -> AnswerCaseSet:
        if not self.cases_path.exists():
            return AnswerCaseSet(suite_version="1", cases=[])
        return AnswerCaseSet.model_validate_json(self.cases_path.read_text(encoding="utf-8"))

    def _write(self, suite: AnswerCaseSet) -> None:
        text = suite.to_file_json()
        self.cases_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cases_path.with_suffix(self.cases_path.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, self.cases_path)

    def save_case(
        self,
        payload: dict[str, Any],
        *,
        original_id: str | None,
        approve: bool,
    ) -> tuple[AnswerCase, str | None]:
        """Validates and stores one case; returns it with an optional notice."""
        try:
            case = AnswerCase.model_validate(payload)
        except ValidationError as exc:
            raise ReviewError("\n".join(_validation_messages(exc))) from exc

        problems = validate_case_sources(AnswerCaseSet(suite_version="1", cases=[case]), self.corpus)
        if problems:
            raise ReviewError("\n".join(problems))

        notice = None
        with self._lock:
            suite = self.load()
            cases = list(suite.cases)
            index = next(
                (position for position, item in enumerate(cases) if item.id == original_id),
                None,
            )
            if original_id is not None and index is None:
                raise ReviewError(f"Case {original_id!r} no longer exists; reload the page.")
            if any(item.id == case.id for position, item in enumerate(cases) if position != index):
                raise ReviewError(f"Another case already uses id {case.id!r}.")

            if approve:
                case.review_status = "reviewed"
            elif index is not None:
                previous = cases[index]
                changed = _case_content(previous) != _case_content(case)
                if previous.review_status == "reviewed" and changed:
                    case.review_status = "draft"
                    notice = "Content changed after approval, so the case is back to draft."
                elif not changed:
                    case.review_status = previous.review_status
            else:
                case.review_status = "draft"

            if index is None:
                cases.append(case)
            else:
                cases[index] = case
            self._write(AnswerCaseSet(suite_version=suite.suite_version, cases=cases))
        return case, notice

    def set_status(self, case_id: str, status: str) -> AnswerCase:
        if status not in {"draft", "reviewed"}:
            raise ReviewError(f"Unknown status {status!r}")
        with self._lock:
            suite = self.load()
            for case in suite.cases:
                if case.id == case_id:
                    case.review_status = "reviewed" if status == "reviewed" else "draft"
                    self._write(suite)
                    return case
        raise ReviewError(f"Unknown case {case_id!r}")

    def delete_case(self, case_id: str) -> None:
        with self._lock:
            suite = self.load()
            remaining = [case for case in suite.cases if case.id != case_id]
            if len(remaining) == len(suite.cases):
                raise ReviewError(f"Unknown case {case_id!r}")
            self._write(AnswerCaseSet(suite_version=suite.suite_version, cases=remaining))

    # Runs ------------------------------------------------------------------

    def runs(self) -> list[dict[str, Any]]:
        """Every answer run JSON under the run directories, newest first, without traces."""
        found: list[tuple[float, dict[str, Any]]] = []
        for directory in self.run_dirs:
            if not directory.exists():
                continue
            for path in directory.rglob("*.json"):
                try:
                    run = AnswerEvalRun.model_validate_json(path.read_text(encoding="utf-8"))
                except (ValidationError, OSError, UnicodeDecodeError):
                    continue
                results = {}
                for result in run.results:
                    data = result.model_dump(mode="json", exclude={"trace"})
                    results[result.case_id] = data
                found.append(
                    (
                        path.stat().st_mtime,
                        {
                            "path": path.as_posix(),
                            "config": run.config.model_dump(mode="json"),
                            "aggregate": run.aggregate.model_dump(mode="json"),
                            "results": results,
                        },
                    )
                )
        found.sort(key=lambda item: item[0], reverse=True)
        return [run for _, run in found]

    # State -----------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        suite = self.load()
        return {
            "cases_path": self.cases_path.as_posix(),
            "suite": suite.model_dump(mode="json"),
            "categories": list(get_args(AnswerCategory)),
            "origins": list(get_args(CaseOrigin)),
            "files": self.corpus.files(),
            "run_dirs": [path.as_posix() for path in self.run_dirs],
        }

    def passage(self, file_path: str, start: int, end: int) -> dict[str, Any]:
        if end < start or end - start > 50:
            raise ReviewError("Scene range must be ascending and at most 50 scenes.")
        text = self.corpus.passage(file_path, start, end)
        missing = [
            index for index in range(start, end + 1) if not self.corpus.has_scene(file_path, index)
        ]
        return {"text": text, "missing_scenes": missing}

    def search(self, pattern: str, path_filter: str = "") -> list[dict[str, Any]]:
        if not pattern.strip():
            return []
        try:
            return self.corpus.search(pattern, path_filter=path_filter)
        except re.error as exc:
            raise ReviewError(f"Invalid regular expression: {exc}") from exc


def _page() -> bytes:
    return files(__package__).joinpath("review_page.html").read_bytes()


def make_handler(store: ReviewStore) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
            return

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data: Any, status: int = HTTPStatus.OK) -> None:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8")

        def _error(self, message: str, status: int = HTTPStatus.BAD_REQUEST) -> None:
            self._json({"error": message}, status)

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            data = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(data, dict):
                raise ReviewError("Request body must be a JSON object.")
            return data

        def _same_origin(self) -> bool:
            # Mutating requests must come from this page, not another site in the browser.
            origin = self.headers.get("Origin")
            host = self.headers.get("Host", "")
            return origin is None or urlparse(origin).netloc == host

        def do_GET(self) -> None:  # noqa: N802 - stdlib name
            url = urlparse(self.path)
            query = {key: values[0] for key, values in parse_qs(url.query).items()}
            try:
                if url.path in {"/", "/index.html"}:
                    self._send(HTTPStatus.OK, _page(), "text/html; charset=utf-8")
                elif url.path == "/api/state":
                    self._json(store.state())
                elif url.path == "/api/runs":
                    self._json(store.runs())
                elif url.path == "/api/passage":
                    self._json(
                        store.passage(
                            query.get("file_path", ""),
                            int(query.get("start", "0")),
                            int(query.get("end", query.get("start", "0"))),
                        )
                    )
                elif url.path == "/api/search":
                    self._json(store.search(query.get("q", ""), query.get("path", "")))
                else:
                    self._error("Not found", HTTPStatus.NOT_FOUND)
            except (ReviewError, ValueError) as exc:
                self._error(str(exc))

        def do_PUT(self) -> None:  # noqa: N802
            if not self._same_origin():
                self._error("Cross-origin request rejected", HTTPStatus.FORBIDDEN)
                return
            url = urlparse(self.path)
            try:
                body = self._body()
                if url.path == "/api/case":
                    case, notice = store.save_case(
                        body.get("case") or {},
                        original_id=body.get("original_id"),
                        approve=bool(body.get("approve")),
                    )
                    self._json({"case": case.model_dump(mode="json"), "notice": notice})
                elif url.path.startswith("/api/case/") and url.path.endswith("/status"):
                    case_id = unquote(url.path[len("/api/case/") : -len("/status")])
                    case = store.set_status(case_id, str(body.get("status")))
                    self._json({"case": case.model_dump(mode="json")})
                else:
                    self._error("Not found", HTTPStatus.NOT_FOUND)
            except (ReviewError, ValueError) as exc:
                self._error(str(exc))

        def do_DELETE(self) -> None:  # noqa: N802
            if not self._same_origin():
                self._error("Cross-origin request rejected", HTTPStatus.FORBIDDEN)
                return
            url = urlparse(self.path)
            if not url.path.startswith("/api/case/"):
                self._error("Not found", HTTPStatus.NOT_FOUND)
                return
            try:
                store.delete_case(unquote(url.path[len("/api/case/") :]))
                self._json({"ok": True})
            except ReviewError as exc:
                self._error(str(exc))

    return Handler


def serve(store: ReviewStore, *, port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(store))
