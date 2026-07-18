from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from linkura_story_indexer import cli
from linkura_story_indexer.query.engine import StreamingQueryResult


class RecordingConsole:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def print(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append((args, kwargs))

    def text(self) -> str:
        return "\n".join(str(args[0]) for args, _ in self.calls if args)


def _prompt_from(values: list[str]) -> Any:
    questions = iter(values)
    return lambda prompt: next(questions)


def test_chat_streams_incremental_verbatim_output_and_reuses_router_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording_console = RecordingConsole()

    class FakeEngine:
        def __init__(self, **kwargs: Any):
            self.questions: list[str] = []

        def stream_query(self, question: str) -> StreamingQueryResult:
            self.questions.append(question)
            return StreamingQueryResult(
                answer_deltas=iter(["[first]", " second"]),
                router_metadata={
                    "router_model": "fixture-router",
                    "chosen_tool": "vector_search_raw",
                    "validated_args": {"query": question},
                    "fallback_used": False,
                    "fallback_reason": None,
                },
            )

    engine = FakeEngine()
    monkeypatch.setattr(cli, "console", recording_console)
    monkeypatch.setattr(cli, "initialize_query_settings", lambda: None)
    monkeypatch.setattr(cli, "StoryQueryEngine", lambda **kwargs: engine)
    monkeypatch.setattr(cli.typer, "prompt", _prompt_from(["question", "exit"]))

    cli.chat(routing_mode="llm_router", show_router=True)

    assert engine.questions == ["question"]
    assert "  chosen_tool: vector_search_raw" in recording_console.text()
    chunk_calls = [call for call in recording_console.calls if call[0] in {("[first]",), (" second",)}]
    assert [call[0][0] for call in chunk_calls] == ["[first]", " second"]
    assert all(call[1]["markup"] is False for call in chunk_calls)
    assert all(call[1]["highlight"] is True for call in chunk_calls)
    assert all(call[1]["end"] == "" for call in chunk_calls)


def test_chat_ctrl_c_closes_active_stream_and_continues(monkeypatch: pytest.MonkeyPatch) -> None:
    recording_console = RecordingConsole()
    closed = False

    def interrupted_stream() -> Iterator[str]:
        nonlocal closed
        try:
            yield "partial"
            raise KeyboardInterrupt
        finally:
            closed = True

    class FakeEngine:
        def __init__(self, **kwargs: Any):
            self.questions: list[str] = []

        def stream_query(self, question: str) -> StreamingQueryResult:
            self.questions.append(question)
            if question == "first":
                return StreamingQueryResult(answer_deltas=interrupted_stream())
            return StreamingQueryResult(answer_deltas=iter(["complete"]))

    engine = FakeEngine()
    monkeypatch.setattr(cli, "console", recording_console)
    monkeypatch.setattr(cli, "initialize_query_settings", lambda: None)
    monkeypatch.setattr(cli, "StoryQueryEngine", lambda **kwargs: engine)
    monkeypatch.setattr(cli.typer, "prompt", _prompt_from(["first", "second", "exit"]))

    cli.chat(routing_mode="off", show_router=False)

    assert closed is True
    assert engine.questions == ["first", "second"]
    assert "Answer interrupted. Ready for another question." in recording_console.text()
    assert "complete" in recording_console.text()
    assert "Chat session ended. Goodbye!" in recording_console.text()


@pytest.mark.parametrize("prompt_error", [KeyboardInterrupt, EOFError])
def test_chat_prompt_interrupt_or_eof_exits_normally(
    monkeypatch: pytest.MonkeyPatch,
    prompt_error: type[BaseException],
) -> None:
    recording_console = RecordingConsole()

    class FakeEngine:
        def __init__(self, **kwargs: Any):
            self.stream_called = False

        def stream_query(self, question: str) -> StreamingQueryResult:
            self.stream_called = True
            raise AssertionError("stream_query should not be called")

    engine = FakeEngine()
    monkeypatch.setattr(cli, "console", recording_console)
    monkeypatch.setattr(cli, "initialize_query_settings", lambda: None)
    monkeypatch.setattr(cli, "StoryQueryEngine", lambda **kwargs: engine)
    monkeypatch.setattr(cli.typer, "prompt", lambda prompt: (_ for _ in ()).throw(prompt_error()))

    cli.chat(routing_mode="off", show_router=False)

    assert engine.stream_called is False
    assert "Chat session ended. Goodbye!" in recording_console.text()


def test_query_command_keeps_non_streaming_engine_path(monkeypatch: pytest.MonkeyPatch) -> None:
    recording_console = RecordingConsole()

    class FakeEngine:
        def __init__(self, **kwargs: Any):
            self.query_calls: list[str] = []

        def query(self, question: str) -> str:
            self.query_calls.append(question)
            return "synchronous answer"

        def stream_query(self, question: str) -> StreamingQueryResult:
            raise AssertionError("query command must not use stream_query")

    engine = FakeEngine()
    monkeypatch.setattr(cli, "console", recording_console)
    monkeypatch.setattr(cli, "initialize_query_settings", lambda: None)
    monkeypatch.setattr(cli, "StoryQueryEngine", lambda **kwargs: engine)

    cli.query("question", routing_mode="off", show_router=False)

    assert engine.query_calls == ["question"]
    assert "synchronous answer" in recording_console.text()
