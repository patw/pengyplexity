"""Tests for :mod:`pengyplexity.core.streaming`.

These verify:
* ``StreamEvent.to_sse()`` produces correct SSE wire format.
* Helper functions (``sse_token``, ``sse_activity``, ``sse_done``, etc.)
  produce the right event type + data.
* ``agent_stream`` yields activity → token → done events.
* ``agent_stream`` yields an error event if the agent raises.
* ``collect_events`` correctly parses a generator into a list of dicts.
* ``make_sse_response`` returns a Flask response with the right content type.

All offline — no network, no live model.
"""

from __future__ import annotations

import json
import pytest

from pengyplexity.core.streaming import (
    StreamEvent,
    agent_stream,
    collect_events,
    make_sse_response,
    sse_activity,
    sse_done,
    sse_error,
    sse_event,
    sse_tool_call,
    sse_tool_result,
    sse_token,
)
from pengyplexity.core.agent import AgentResult, AgentStreamEvent


# ---------------------------------------------------------------------------
# StreamEvent
# ---------------------------------------------------------------------------


class TestStreamEvent:
    def test_to_sse_format(self):
        e = StreamEvent(event="token", data={"content": "hello"})
        sse = e.to_sse()
        assert sse.startswith("event: token\n")
        assert 'data: {"content": "hello"}' in sse
        assert sse.endswith("\n\n")

    def test_to_sse_multiline_data(self):
        e = StreamEvent(event="done", data={"answer": "line1\nline2", "sources": []})
        sse = e.to_sse()
        # The data line should be valid JSON (newlines are escaped).
        lines = sse.strip().split("\n")
        data_line = [l for l in lines if l.startswith("data: ")][0]
        payload = json.loads(data_line[6:])
        assert payload["answer"] == "line1\nline2"


# ---------------------------------------------------------------------------
# SSE helper functions
# ---------------------------------------------------------------------------


class TestSSEHelpers:
    def test_sse_token(self):
        result = sse_token("hello world")
        assert "event: token" in result
        assert '"content": "hello world"' in result

    def test_sse_activity(self):
        result = sse_activity("Searching the web...", type="searching")
        assert "event: activity" in result
        assert "Searching the web..." in result
        assert '"type": "searching"' in result

    def test_sse_tool_call(self):
        result = sse_tool_call("web_search", {"query": "cats"})
        assert "event: tool_call" in result
        assert '"name": "web_search"' in result
        assert '"query": "cats"' in result

    def test_sse_tool_result(self):
        result = sse_tool_result("web_search", "found 5 results")
        assert "event: tool_result" in result
        assert '"name": "web_search"' in result

    def test_sse_done(self):
        sources = [{"title": "A", "url": "https://a.com"}]
        result = sse_done("The answer.", sources)
        assert "event: done" in result
        assert '"answer": "The answer."' in result
        assert "https://a.com" in result

    def test_sse_done_no_sources(self):
        result = sse_done("just text")
        assert '"sources": []' in result

    def test_sse_error(self):
        result = sse_error("something broke")
        assert "event: error" in result
        assert '"message": "something broke"' in result

    def test_sse_event_generic(self):
        result = sse_event("custom", {"x": 1})
        assert "event: custom" in result
        assert '"x": 1' in result


# ---------------------------------------------------------------------------
# agent_stream
# ---------------------------------------------------------------------------


class FakeAgent:
    """Minimal agent fake for streaming tests."""

    def __init__(self, answer="Hello from agent.", sources=None):
        self.answer = answer
        self.sources = sources or []

    def run(self, user_message, history=None) -> AgentResult:
        return AgentResult(
            answer=self.answer,
            sources=self.sources,
            iterations=1,
            messages=[],
        )


class FailingAgent:
    def run(self, user_message, history=None):
        raise RuntimeError("model exploded")


class StreamingAgent:
    """A fake agent exposing run_stream (yields AgentStreamEvents)."""

    def __init__(self, answer="Streamed answer.", search=True):
        self.answer = answer
        self.search = search
        self._last_sources = [{"title": "S", "url": "https://s.example"}]

    def run(self, user_message, history=None):
        return AgentResult(answer=self.answer, sources=self._last_sources)

    def run_stream(self, user_message, history=None):
        if self.search:
            yield AgentStreamEvent("activity", {"label": "Searching the web…", "tool": "web_search"})
        yield AgentStreamEvent("token", {"content": "Streamed "})
        yield AgentStreamEvent("token", {"content": "answer."})


class TestAgentStream:
    def test_yields_activity_token_done(self):
        agent = FakeAgent(answer="The answer is 42.")
        events = collect_events(agent_stream(agent, "What is the answer?"))

        # Should be: activity, token, done
        assert len(events) == 3
        assert events[0]["event"] == "activity"
        assert events[0]["data"]["label"] == "Thinking..."
        assert events[1]["event"] == "token"
        assert events[1]["data"]["content"] == "The answer is 42."
        assert events[2]["event"] == "done"
        assert events[2]["data"]["answer"] == "The answer is 42."

    def test_done_includes_sources(self):
        agent = FakeAgent(
            answer="Based on research.",
            sources=[{"title": "Source A", "url": "https://a.example"}],
        )
        events = collect_events(agent_stream(agent, "research"))
        done_event = [e for e in events if e["event"] == "done"][0]
        assert done_event["data"]["sources"] == [
            {"title": "Source A", "url": "https://a.example"}
        ]

    def test_error_event_on_exception(self):
        agent = FailingAgent()
        events = collect_events(agent_stream(agent, "q"))

        # Should be: activity, error (no token, no done)
        assert len(events) == 2
        assert events[0]["event"] == "activity"
        assert events[1]["event"] == "error"
        assert "model exploded" in events[1]["data"]["message"]

    def test_empty_answer_no_token_event(self):
        agent = FakeAgent(answer="")
        events = collect_events(agent_stream(agent, "q"))
        # activity + done (no token since answer is empty)
        event_types = [e["event"] for e in events]
        assert "token" not in event_types
        assert "done" in event_types

    def test_history_passed_through(self):
        agent = FakeAgent(answer="ok")
        history = [{"role": "user", "content": "prior"}]
        list(agent_stream(agent, "new question", history=history))
        # No error — history was accepted.

    def test_streaming_agent_emits_activity_events(self):
        """A run_stream agent yields an SSE activity for tool work."""
        agent = StreamingAgent(answer="Streamed answer.")
        events = collect_events(agent_stream(agent, "q"))

        event_types = [e["event"] for e in events]
        assert "activity" in event_types
        assert "token" in event_types
        assert "done" in event_types

        # The generic tool activity label is surfaced (not the raw tool name).
        act = next(e for e in events if e["event"] == "activity" and "Search" in e["data"]["label"])

        # Tokens are streamed individually.
        token_contents = [e["data"]["content"] for e in events if e["event"] == "token"]
        assert "".join(token_contents) == "Streamed answer."

        done = next(e for e in events if e["event"] == "done")
        assert done["data"]["answer"] == "Streamed answer."


# ---------------------------------------------------------------------------
# collect_events
# ---------------------------------------------------------------------------


class TestCollectEvents:
    def test_parses_multiple_events(self):
        def gen():
            yield sse_token("a")
            yield sse_token("b")
            yield sse_done("done")

        events = collect_events(gen())
        assert len(events) == 3
        assert events[0]["data"]["content"] == "a"
        assert events[1]["data"]["content"] == "b"
        assert events[2]["event"] == "done"

    def test_empty_generator(self):
        def gen():
            return
            yield  # pragma: no cover

        events = collect_events(gen())
        assert events == []


# ---------------------------------------------------------------------------
# make_sse_response (Flask integration)
# ---------------------------------------------------------------------------


class TestMakeSSEResponse:
    def test_returns_flask_response(self, app):
        """Inside a Flask app context, make_sse_response returns a Response."""
        def gen():
            yield sse_token("hi")
            yield sse_done("hi")

        with app.test_request_context():
            resp = make_sse_response(gen())
            assert resp.content_type.startswith("text/event-stream")
            assert resp.headers["Cache-Control"] == "no-cache"

    def test_response_body_is_sse(self, app):
        def gen():
            yield sse_token("hello")

        with app.test_request_context():
            resp = make_sse_response(gen())
            body = b"".join(resp.response)
            assert b"event: token" in body
            assert b'"hello"' in body


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
