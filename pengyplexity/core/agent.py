"""One-turn agent loop for Pengyplexity.

This module provides:

* :data:`SYSTEM_PROMPT` — the dedicated Perplexity-style system prompt that
  presents the model as a research assistant with web search as the star,
  warns against sandbox escape, and points to capabilities (not host tools).
* :class:`AgentResult` — the result of a single agent turn (final answer,
  sources, iteration count).
* :class:`Agent` — the tool-call loop that:
  1. builds the system prompt + user message,
  2. calls the model with SAFE_TOOLS schema,
  3. executes any tool calls the model requests (via an injectable
     ``tool_executor`` callback),
  4. loops until the model produces a final answer (no tool_calls) or hits
     the iteration cap,
  5. extracts sources from web_search / fetch_url results.

Design notes:
* The agent does NOT own the model client or the store — it takes them as
  constructor arguments so tests can inject fakes.
* Tool execution is delegated to a ``tool_executor`` callable
  ``(name: str, args: dict, workspace: Path) -> str``. In production this
  wraps ``confine.resolve`` + ``Runner``; in tests it's a fake.
* The iteration cap (``max_iterations``, default 12) prevents runaway loops.
* Sources are extracted from ``web_search`` results (title + URL pairs) and
  from ``fetch_url`` calls (the fetched URL).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

from .modelclient import ChatResponse, ModelClient, ToolCall

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are Pengyplexity, a research assistant that answers questions using \
web search as your primary tool. You have access to a set of safe, \
confined tools that operate ONLY within this thread's workspace.

## Your capabilities

- **Web search**: Use `web_search` to find current, relevant information. \
This is your primary tool — ALWAYS search before answering factual questions. \
**Budget your searches**: 1 search is enough for most questions, 2-3 for a \
question with a few distinct angles (e.g. ratings AND reviews). Combine \
related aspects into one well-chosen query rather than firing off a separate \
search per angle. You have a hard cap of {max_iterations} tool round-trips \
for this entire turn (searches, fetches, everything) — stop searching once \
you have enough to answer, and never plan on using anywhere near the cap.
- **Fetch**: Use `fetch_url` to read a specific web page for deeper context.
- **Scripts**: You can write and run Python or shell scripts in your \
workspace to perform calculations or data processing. The sandbox has \
**matplotlib, numpy and pandas** and the Python standard library, and has \
**no network** — fetch data with `web_search`/`fetch_url`/`download_file` \
first, then process it in the script.
- **Charts**: Use `make_chart` to generate a chart image — do not just \
describe one in your answer. The script must call `savefig` with exactly the \
filename you passed, into the current directory. If it fails you get the \
interpreter's real output back; read it and fix the script.
- **Looking at images**: Use `read_image` to actually see an image in your \
workspace — a chart you just made, a file the user downloaded. You have \
vision, so look before describing. Charts and generated images are attached \
automatically as soon as you create them.
- **Reports/PDFs**: Use `create_report` to generate a downloadable report \
(HTML + PDF) from Markdown whenever the user asks for a document, \
write-up, or PDF. You MUST actually call this tool — never claim a report \
or PDF was created without calling it; there is no other way to produce one.
- **Memory**: Use `save_memory` to remember durable facts about the user or \
their ongoing work (preferences, recurring projects, context worth keeping \
across conversations) — not one-off details only relevant to this turn. Use \
`search_memory` to recall relevant memories before answering, especially at \
the start of a new thread. When something you remembered has changed or was \
wrong, use `edit_memory` on the id `search_memory` gives you rather than \
saving a second, contradictory memory; use `delete_memory` when the user \
asks you to forget something — it is permanent, so do not use it to tidy up \
on your own initiative.

## Rules (NON-NEGOTIABLE)

1. **Never touch files outside your workspace.** All file paths are \
relative to your workspace. Attempting to access `../`, absolute paths, \
or system paths will fail.
2. **Never use sudo or attempt privilege escalation.** This is \
structurally impossible in your sandbox.
3. **Never modify Pengy, Pengyplexity, or any skill files.** You do not \
have write access to them.
4. **Never attempt to escape the sandbox.** Your code runs in an \
isolated container: no network, no host filesystem, no host processes, \
no credentials in the environment. There is nothing there to find.
5. **Cite your sources.** When answering with information from web \
searches, include a "Sources:" section at the end listing the titles \
and URLs you used.
6. **Be concise and direct.** Answer the question, cite sources, done.

## Format

- Use markdown for formatting.
- Cite sources as: `[n] Title — URL`
- If you produce a chart or file, mention its filename in the workspace.
- Keep answers focused and well-structured.
"""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class AgentResult:
    """Result of one agent turn.

    Attributes
    ----------
    answer:
        The final text answer (markdown).
    sources:
        Extracted sources: list of ``{"title": ..., "url": ...}`` dicts.
    iterations:
        Number of model round-trips (tool-call iterations + final).
    messages:
        The full message history (for persistence / debugging).
    """

    answer: str
    sources: List[Dict[str, str]] = field(default_factory=list)
    iterations: int = 0
    messages: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Streaming event type
# ---------------------------------------------------------------------------


@dataclass
class AgentStreamEvent:
    """A single event yielded by :meth:`Agent.run_stream`.

    ``kind`` is either ``"token"`` (a content chunk of the final answer) or
    ``"activity"`` (a generic, user-facing label describing what the agent is
    doing — e.g. "Searching the web…"). The activity payload carries the label
    (and the underlying tool name) so the app can surface friendly progress
    without exposing raw tool internals.

    Attributes
    ----------
    kind:
        ``"token"`` or ``"activity"``.
    data:
        For tokens: ``{"content": str}``. For activity: ``{"label": str,
        "tool": str}``.
    """

    kind: str
    data: Dict[str, Any] = field(default_factory=dict)



# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


# Type alias for the tool executor callback.
ToolExecutor = Callable[[str, Dict[str, Any], Path], str]


class Agent:
    """One-turn agent loop.

    Parameters
    ----------
    model:
        A :class:`ModelClient` (real or fake).
    tool_executor:
        A callable ``(name, args, workspace) -> str`` that executes a tool
        and returns its string result. In production this wraps confine +
        Runner; in tests it's a fake.
    workspace:
        The per-thread workspace root (a :class:`Path`). All file
        operations are confined here.
    max_iterations:
        Hard cap on model round-trips (default 12). Prevents runaway
        tool-call loops.
    tools:
        Optional list of tool schemas (OpenAI function-calling format).
        If None, the agent will use an empty list (the caller is
        responsible for providing SAFE_TOOLS schemas in production).
    """

    def __init__(
        self,
        model: ModelClient,
        tool_executor: ToolExecutor,
        workspace: Path,
        max_iterations: int = 12,
        tools: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        cancel=None,
    ) -> None:
        self.model = model
        self.tool_executor = tool_executor
        self.workspace = workspace
        self.max_iterations = max_iterations
        self.tools = tools or []
        # Admin-configurable override of SYSTEM_PROMPT (see core/settings.py);
        # None/empty means "use the default". Settable per-turn, mirroring how
        # ``workspace`` is already repointed per request in web.py.
        self.system_prompt = system_prompt
        # Sources gathered by the most recent turn. Initialised here so a
        # caller reading it before (or during) a run gets an empty list rather
        # than an AttributeError, and reset at the start of every turn so one
        # turn never cites the previous turn's pages.
        self._last_sources: List[Dict[str, str]] = []
        # This turn's cancel token (see core/cancel.py), set per request by
        # web.py. Checked only at safe points — between streamed chunks,
        # between tool calls, between loop iterations — so an interrupted
        # turn unwinds through ordinary control flow and the text produced so
        # far is kept rather than lost mid-write.
        self.cancel = cancel

    @property
    def _effective_system_prompt(self) -> str:
        if self.system_prompt:
            # An admin-set override — its own placeholders (date/username/
            # etc.) are already resolved before it lands here (see
            # web.py: _apply_effective_settings), so it's used as-is.
            return self.system_prompt
        return SYSTEM_PROMPT.format(max_iterations=self.max_iterations)

    def run(self, user_message: str, history: Optional[List[Dict[str, Any]]] = None) -> AgentResult:
        """Run one agent turn.

        Parameters
        ----------
        user_message:
            The user's question (new message for this turn).
        history:
            Optional prior message history (from the thread). The system
            prompt is always prepended.

        Returns
        -------
        AgentResult
        """
        # Build the initial message list.
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._effective_system_prompt},
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        sources: List[Dict[str, str]] = []
        self._last_sources = sources
        # A turn that makes a tool call can *also* carry real answer text in
        # the same response (e.g. the model writes its answer and calls
        # `save_memory` in one turn). That text must not be discarded just
        # because the turn wasn't the final one — accumulate every non-empty
        # `response.content` we see, in order, across the whole loop.
        answer_parts: List[str] = []
        iterations = 0

        for i in range(self.max_iterations):
            if self._cancelled:
                return AgentResult(
                    answer="\n\n".join(answer_parts),
                    sources=sources,
                    iterations=iterations,
                    messages=messages,
                )
            iterations += 1
            # On the last allowed round-trip, don't offer tools at all: the
            # model has nowhere left to spend another tool call anyway (the
            # loop is about to end), so withholding tools forces it to
            # synthesize a real answer from whatever it already gathered
            # instead of firing off one more search that would just get cut
            # off, leaving the user with the bare "[Stopped...]" placeholder.
            is_last_iteration = i == self.max_iterations - 1
            call_tools = None if is_last_iteration else (self.tools if self.tools else None)
            response = self.model.chat(messages, tools=call_tools)

            if response.content:
                answer_parts.append(response.content)

            if not response.has_tool_calls:
                # Final answer — no more tool calls.
                return AgentResult(
                    answer="\n\n".join(answer_parts),
                    sources=sources,
                    iterations=iterations,
                    messages=messages,
                )

            # Append the assistant's tool-call message.
            messages.append(_assistant_tool_call_message(response))

            # Execute each tool call.
            for tc in response.tool_calls:
                result_str = self._execute_tool(tc)
                # Track sources from web_search / fetch_url.
                self._extract_sources(tc, result_str, sources)
                # Append the tool result.
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })

            image_message = self._pending_image_message()
            if image_message is not None:
                messages.append(image_message)

        # Hit the iteration cap — return whatever the model said so far.
        answer_parts.append("[Stopped: reached maximum iterations]")
        return AgentResult(
            answer="\n\n".join(answer_parts),
            sources=sources,
            iterations=iterations,
            messages=messages,
        )

    def run_stream(
        self,
        user_message: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Iterator[AgentStreamEvent]:
        """Run the agent and yield :class:`AgentStreamEvent` items live.

        This mirrors :meth:`run` (same tool-call loop, same sources) but emits
        structured streaming events: an ``activity`` event (generic label)
        before each tool executes, and ``token`` events carrying answer text
        **as the model writes it**.

        Each round-trip goes through :meth:`ModelClient.stream_chat`, which
        yields content deltas and returns the assembled
        :class:`~pengyplexity.core.modelclient.ChatResponse` (tool calls
        included). So the same single request both streams the prose and tells
        the loop whether the model wants a tool — no second, non-streaming
        request, and no replaying an already-finished answer in fake chunks
        after the user has watched a spinner for the whole turn.

        Yields
        ------
        AgentStreamEvent
            ``kind="activity"`` with ``data={"label", "tool"}`` for tool work,
            then ``kind="token"`` with ``data={"content"}`` for answer text.
        """
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._effective_system_prompt},
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        sources: List[Dict[str, str]] = []
        self._last_sources = sources
        # See `run()` — a tool-calling turn can carry real answer text
        # alongside the tool call (e.g. the model writes its answer and
        # calls `save_memory` in the same turn), so every non-empty chunk is
        # streamed, not just the final turn's.
        any_content = False

        for i in range(self.max_iterations):
            # See `run()` — withhold tools on the last allowed round-trip so
            # the model is forced to synthesize a real answer from whatever
            # it has instead of firing off one more tool call that would
            # just get cut off.
            is_last_iteration = i == self.max_iterations - 1
            call_tools = None if is_last_iteration else (self.tools if self.tools else None)

            if self._cancelled:
                self._last_sources = sources
                return

            stream = self.model.stream_chat(messages, tools=call_tools)
            response: Optional[ChatResponse] = None
            streamed: List[str] = []
            while True:
                if self._cancelled:
                    # Close the HTTP stream rather than draining it: the
                    # user is no longer waiting for these tokens.
                    stream.close()
                    self._last_sources = sources
                    return
                try:
                    chunk = next(stream)
                except StopIteration as stop:
                    response = stop.value
                    break
                if not chunk:
                    continue
                if any_content and not streamed:
                    # Text from an earlier round-trip is already on screen —
                    # separate this round-trip's text from it.
                    yield AgentStreamEvent("token", {"content": "\n\n"})
                streamed.append(chunk)
                any_content = True
                yield AgentStreamEvent("token", {"content": chunk})

            if response is None:
                # A client whose stream_chat is a plain iterator with no
                # return value — treat what it streamed as the whole answer.
                response = ChatResponse(content="".join(streamed))

            if not response.has_tool_calls:
                self._last_sources = sources
                return

            # Tool-calling turn: emit an activity signal then execute.
            messages.append(_assistant_tool_call_message(response))
            for tc in response.tool_calls:
                if self._cancelled:
                    self._last_sources = sources
                    return
                from ..sandbox.toolpolicy import tool_activity_label
                yield AgentStreamEvent("activity", {
                    "label": tool_activity_label(tc.name),
                    "tool": tc.name,
                })
                # Checked again after the yield: that is where control
                # returns to the consumer, so it is where a Stop pressed
                # while this label was on screen actually arrives.
                if self._cancelled:
                    self._last_sources = sources
                    return
                result_str = self._execute_tool(tc)
                self._extract_sources(tc, result_str, sources)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })

            image_message = self._pending_image_message()
            if image_message is not None:
                messages.append(image_message)

        # Iteration cap reached.
        self._last_sources = sources
        if any_content:
            yield AgentStreamEvent("token", {"content": "\n\n"})
        yield AgentStreamEvent("token", {"content": "[Stopped: reached maximum iterations]"})

    @property
    def _cancelled(self) -> bool:
        """True once the user has pressed Stop for this turn."""
        return self.cancel is not None and self.cancel.cancelled

    def _pending_image_message(self) -> Optional[Dict[str, Any]]:
        """Drain images the turn's tools produced into a follow-up user message.

        A ``role: "tool"`` message carries string content only, so read_image /
        make_chart / generate_image park the encoded picture on the tool
        context instead (Pengy's mechanism — see core/vision.py). Attaching it
        here, *after* every tool result for the round-trip is in place, keeps
        each assistant tool-call message immediately followed by its matching
        tool messages, which some backends require.
        """
        from .vision import build_image_message

        executor = self.tool_executor
        context = getattr(executor, "context", None)
        pending = getattr(context, "pending_images", None) if context else None
        if pending is None:
            return None
        return build_image_message(pending.take())

    def _execute_tool(self, tc: ToolCall) -> str:
        """Execute a tool call via the tool_executor callback."""
        try:
            return self.tool_executor(tc.name, tc.arguments, self.workspace)
        except Exception as e:
            return f"Tool error: {e}"

    def _extract_sources(
        self,
        tc: ToolCall,
        result: str,
        sources: List[Dict[str, str]],
    ) -> None:
        """Extract source info from web_search / fetch_url results.

        For ``web_search``: parse the result for title/URL pairs.
        For ``fetch_url``: record the fetched URL.
        """
        if tc.name == "web_search":
            # The result format from ddgs is a list of dicts with
            # "title" and "href" keys, serialized as JSON or a string.
            _add_search_sources(result, sources)
        elif tc.name == "fetch_url":
            url = tc.arguments.get("url", "")
            if url:
                _add_source(sources, title=url, url=url)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk_text(text: str):
    """Split *text* into small word-ish substrings for a simulated token
    stream (used to replay an already-fetched final answer as it is
    persisted, instead of re-querying the model)."""
    words = text.split(" ")
    out: List[str] = []
    for w in words:
        if out and len(out[-1]) < 20:
            out[-1] += " " + w
        else:
            # Every chunk after the first must carry the space that
            # separated it from the previous word — the caller concatenates
            # chunks directly, so a bare `w` here would glue the last word
            # of the previous chunk to this one with no space between them.
            out.append((" " if out else "") + w)
    return iter(out)


def _assistant_tool_call_message(response: ChatResponse) -> Dict[str, Any]:
    """Build the OpenAI-format assistant message with tool_calls."""
    tool_calls = [
        {
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.name,
                # JSON, not repr(dict): the wire format is JSON, and a Python
                # repr sends single quotes, True/False/None — which a server
                # re-parsing the history rejects or silently mangles.
                "arguments": json.dumps(tc.arguments, ensure_ascii=False),
            },
        }
        for tc in response.tool_calls
    ]
    msg: Dict[str, Any] = {
        "role": "assistant",
        "content": response.content,
        "tool_calls": tool_calls,
    }
    return msg


def _add_source(sources: List[Dict[str, str]], title: str, url: str) -> None:
    """Add a source if not already present (dedup by URL)."""
    if any(s["url"] == url for s in sources):
        return
    sources.append({"title": title, "url": url})


def _add_search_sources(result: str, sources: List[Dict[str, str]]) -> None:
    """Parse a web_search result and extract title/URL pairs.

    Some tool executors (and tests) return a JSON list of
    ``{title, href/url}``; production's ``web_search`` (see
    ``core/search.py``) returns numbered text (``"1. Title\\n   URL:
    https://...\\n   snippet\\n\\n..."``). Try JSON first, then fall back to
    :func:`pengyplexity.core.search.extract_sources`, which already knows how
    to parse that numbered text format correctly.
    """
    import json

    from .search import extract_sources

    # Try JSON list of {title, href/url}
    try:
        data = json.loads(result)
        if isinstance(data, list):
            for item in data:
                title = item.get("title", "")
                url = item.get("href") or item.get("url", "")
                if url:
                    _add_source(sources, title=title, url=url)
            return
    except (json.JSONDecodeError, TypeError):
        pass

    for src in extract_sources(result):
        _add_source(sources, title=src["title"], url=src["url"])
