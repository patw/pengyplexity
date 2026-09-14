"""SSE (Server-Sent Events) streaming helpers for Pengyplexity.

This module provides:

* :func:`sse_event` — format a single SSE event line.
* :func:`stream_response` — a Flask-friendly generator that yields SSE
  formatted chunks.
* :class:`StreamEvent` — a typed event (token, activity, tool_call, done,
  error).
* :func:`agent_stream` — wraps the agent loop to yield SSE events as the
  model produces tokens and tool activity.

The chat endpoint (``/chat/<id>/ask``) can optionally stream via SSE when
the client requests it (``Accept: text/event-stream`` or an ``?stream=1``
query param). For the offline test suite, we test the **generator** directly
— no live HTTP connection needed.

SSE wire format (RFC-compliant):
```
event: token
data: {"content": "Hello"}

event: activity
data: {"type": "searching", "label": "Searching the web..."}

event: done
data: {"answer": "...", "sources": [...]}
```
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------


@dataclass
class StreamEvent:
    """A single SSE event.

    Attributes
    ----------
    event:
        The event type: "token", "activity", "tool_call", "tool_result",
        "done", "error".
    data:
        The payload dict (serialized to JSON in the `data:` line).
    """

    event: str
    data: Dict[str, Any] = field(default_factory=dict)

    def to_sse(self) -> str:
        """Render this event as SSE wire format."""
        payload = json.dumps(self.data, ensure_ascii=False)
        return f"event: {self.event}\ndata: {payload}\n\n"


# ---------------------------------------------------------------------------
# SSE formatting helpers
# ---------------------------------------------------------------------------


def sse_event(event: str, data: Dict[str, Any]) -> str:
    """Format a single SSE event (convenience wrapper)."""
    return StreamEvent(event=event, data=data).to_sse()


def sse_token(content: str) -> str:
    """Format a token (text chunk) event."""
    return sse_event("token", {"content": content})


def sse_activity(label: str, type: str = "activity") -> str:
    """Format an activity indicator event (e.g. "Searching the web...")."""
    return sse_event("activity", {"type": type, "label": label})


def sse_tool_call(name: str, args: Dict[str, Any]) -> str:
    """Format a tool_call event (model is calling a tool)."""
    return sse_event("tool_call", {"name": name, "args": args})


def sse_tool_result(name: str, content: str) -> str:
    """Format a tool_result event."""
    return sse_event("tool_result", {"name": name, "content": content[:500]})


def sse_done(answer: str, sources: List[Dict[str, str]] = None) -> str:
    """Format the final 'done' event with the full answer + sources."""
    return sse_event("done", {"answer": answer, "sources": sources or []})


def sse_error(message: str) -> str:
    """Format an error event."""
    return sse_event("error", {"message": message})


def sse_title(title: str) -> str:
    """Format a title event (the thread was just auto-named from the first
    question — the client updates the sidebar entry without a page reload).
    """
    return sse_event("title", {"title": title})


def sse_artifact(artifact: Dict[str, Any]) -> str:
    """Format an artifact-ready event (a chart/image the agent just produced).

    ``artifact`` is expected to carry ``artifact_id``, ``filename``, ``kind``,
    ``mime``, ``url`` (servable link) and ``download_url``.
    """
    return sse_event("artifact", artifact)


# ---------------------------------------------------------------------------
# Flask response helper
# ---------------------------------------------------------------------------


def make_sse_response(generator: Generator[str, None, None]):
    """Wrap a generator of SSE strings into a Flask Response.

    Usage in a Flask route::

        return make_sse_response(stream_events())

    The response has ``Content-Type: text/event-stream`` and the
    appropriate headers for SSE.
    """
    from flask import Response

    def _encode():
        for chunk in generator:
            yield chunk.encode("utf-8")

    return Response(
        _encode(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# Agent streaming wrapper
# ---------------------------------------------------------------------------


def agent_stream(
    agent,
    user_message: str,
    history: Optional[List[Dict[str, Any]]] = None,
) -> Generator[str, None, None]:
    """Run the agent and yield SSE-formatted events.

    If the agent exposes a streaming answer path (``run_stream``), the
    generator emits an ``activity`` event, then ``token`` events as the model
    produces the final answer, then a ``done`` event with the full answer +
    sources. If the agent does not stream, it falls back to running
    ``agent.run`` and emitting the whole answer as a single ``token`` event.
    """
    # Signal that work has started.
    yield sse_activity("Thinking...")

    try:
        # Prefer the streaming path so tokens reach the browser live.
        if hasattr(agent, "run_stream"):
            collected: List[str] = []
            for ev in agent.run_stream(user_message, history=history):
                if getattr(ev, "kind", None) == "activity":
                    yield sse_activity(
                        ev.data.get("label", "Working…"),
                        type="tool",
                    )
                else:
                    content = ev.data.get("content") if isinstance(ev.data, dict) else ev
                    if isinstance(content, str):
                        collected.append(content)
                        yield sse_token(content)
            answer = "".join(collected)
        else:
            result = agent.run(user_message, history=history)
            answer = result.answer
            sources = result.sources
            if answer:
                yield sse_token(answer)
            yield sse_done(answer, sources)
            return
    except Exception as e:
        yield sse_error(str(e))
        return

    # Final event with full data.
    sources = getattr(agent, "_last_sources", None) or []
    yield sse_done(answer, sources)


# ---------------------------------------------------------------------------
# Test helper: collect all events from a generator
# ---------------------------------------------------------------------------


def collect_events(gen: Generator[str, None, None]) -> List[Dict[str, Any]]:
    """Consume an SSE generator and return a list of parsed events.

    Each event is a dict: ``{"event": <type>, "data": <parsed json>}``.
    Useful in tests to assert the streaming contract.
    """
    events: List[Dict[str, Any]] = []
    buffer = ""
    for chunk in gen:
        buffer += chunk
        # SSE events are separated by double newlines.
        while "\n\n" in buffer:
            raw, buffer = buffer.split("\n\n", 1)
            event_type = ""
            data_str = ""
            for line in raw.split("\n"):
                if line.startswith("event: "):
                    event_type = line[7:]
                elif line.startswith("data: "):
                    data_str = line[6:]
            data = json.loads(data_str) if data_str else {}
            events.append({"event": event_type, "data": data})
    return events
