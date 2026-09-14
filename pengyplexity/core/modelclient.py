"""OpenAI-compatible chat client wrapper for Pengyplexity.

This module provides:

* :class:`ChatResponse` — the result of a single chat round-trip.
* :class:`ModelClient` — the **abstract interface** that the agent loop
  (subtask 8) calls to send messages and receive a response. Tests inject a
  :class:`FakeModelClient` instead of a real HTTP client.
* :class:`OpenAIModelClient` — the real implementation using the
  ``openai`` Python package against any OpenAI-compatible endpoint
  (``PENGYPLEXITY_MODEL_BASE``). Reuses the same wire shape as Pengy's
  ``pengy.core.llm_client.LLMClient`` but is intentionally *simpler*:
  one request → one response, no built-in tool execution (the agent loop
  handles tool calls).
* :class:`FakeModelClient` — records calls, returns canned responses.
  The offline test suite uses this so ``pytest -q`` is green with no
  network.

Design notes:
* The model client does NOT execute tools. It sends the tool *schema* to
  the model and returns the model's response (which may include
  ``tool_calls``). The agent loop (``core/agent.py``) interprets and
  dispatches those calls.
* Streaming is handled separately in ``core/streaming.py`` (subtask 11).
  This module is the non-streaming path.
"""

from __future__ import annotations

import ast
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Iterator


# ---------------------------------------------------------------------------
# Response type
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """A single tool call requested by the model."""

    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ChatResponse:
    """Result of a single chat round-trip.

    Attributes
    ----------
    content:
        The assistant's text content (empty string if it made tool calls
        without text).
    tool_calls:
        List of tool calls the model wants to make (empty if it gave a
        final answer).
    usage:
        Token usage dict: ``{prompt_tokens, completion_tokens, total_tokens}``.
    raw:
        The raw response object (for debugging / streaming extraction).
        May be None for fakes.
    """

    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)
    raw: Any = None

    @property
    def has_tool_calls(self) -> bool:
        """True if the model requested tool execution (not a final answer)."""
        return len(self.tool_calls) > 0


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class ModelClient(ABC):
    """Abstract model client. The agent loop calls :meth:`chat` and receives
    a :class:`ChatResponse`. Tests inject a :class:`FakeModelClient`."""

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
    ) -> ChatResponse:
        """Send *messages* to the model and return the response.

        Parameters
        ----------
        messages:
            OpenAI-format message list (``[{"role": ..., "content": ...}, ...]``).
            May include assistant messages with ``tool_calls`` and role ``"tool"``
            messages for tool results.
        tools:
            Optional list of tool schemas (OpenAI function-calling format).
        temperature:
            Sampling temperature override. None = use the client's default.

        Returns
        -------
        ChatResponse
            The model's response: either text content (final answer) or
            tool_calls (wants to execute tools).
        """
        ...

    def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
    ) -> Iterator[str]:
        """Stream one chat round-trip: yield content chunks, return the response.

        This is the streaming analogue of :meth:`chat` and handles **the same
        turns**, tool-calling ones included: content deltas are yielded as
        they arrive and tool-call deltas are accumulated, so the generator's
        ``return`` value is a complete :class:`ChatResponse` — the agent loop
        needs no second, non-streaming request to discover whether the model
        asked for a tool.

        The default implementation streams nothing and returns the result of
        :meth:`chat`, so a client that cannot stream still works.
        """
        response = self.chat(messages, tools=tools, temperature=temperature)
        if response.content:
            yield response.content
        return response

    def generate_title(self, text: str) -> str:
        """Return a short, human-friendly title for a question.

        Overridden by the real client to make a tiny LLM call. The default
        implementation is a deterministic heuristic (truncate the first line)
        so functional flows work even without a model.
        """
        first = text.splitlines()[0].strip() if text.strip() else "New Thread"
        return first[:60] or "New Thread"


# ---------------------------------------------------------------------------
# Recovery for tool calls leaked as plain text
# ---------------------------------------------------------------------------

# Matches a tag like `<invoke name="...">...body...</invoke>`, tolerant of
# whatever junk a misconfigured backend prefixes onto the tag name (observed
# in the wild as e.g. `<｜｜DSML｜｜ invoke name="run_bash">` — special
# tokens the server's tool-call parser didn't recognize, decoded literally).
_LEAKED_INVOKE_RE = re.compile(
    r'<[^>]*\binvoke\b[^>]*\bname="(?P<name>[^"]*)"[^>]*>'
    r'(?P<body>.*?)'
    r'</[^>]*\binvoke\b[^>]*>',
    re.DOTALL,
)
_LEAKED_ARGS_RE = re.compile(
    r'<[^>]*\bparameter\b[^>]*\bname="arguments"[^>]*>'
    r'(?P<args>.*?)'
    r'</[^>]*\bparameter\b[^>]*>',
    re.DOTALL,
)
# The wrapper tag around one or more <invoke> blocks (e.g. `<...calls>` /
# `</...calls>`) — stripped from the displayed text once its contents have
# been recovered as real tool calls.
_LEAKED_WRAPPER_TAG_RE = re.compile(r'</?[^>]*\bcalls\b[^>]*>')


def _parse_leaked_tool_calls(content: str) -> "tuple[str, List[ToolCall]]":
    """Recover tool calls a misconfigured backend leaked as literal text.

    Returns ``(cleaned_content, tool_calls)``. If nothing is recovered,
    ``cleaned_content`` is *content* unchanged and ``tool_calls`` is empty —
    callers should treat that as "not a leak, just ordinary text".
    """
    tool_calls: List[ToolCall] = []
    if "invoke" not in content or "parameter" not in content:
        return content, tool_calls

    cleaned = content
    for i, m in enumerate(_LEAKED_INVOKE_RE.finditer(content)):
        name = m.group("name")
        args_m = _LEAKED_ARGS_RE.search(m.group("body"))
        if not name or not args_m:
            continue
        raw_args = args_m.group("args").strip()
        try:
            args = json.loads(raw_args)
        except (json.JSONDecodeError, ValueError):
            try:
                # The leaked template sometimes renders the arguments as a
                # Python dict literal (single-quoted) rather than JSON.
                # literal_eval is safe here — it only parses a fixed set of
                # Python literals (dict/list/str/num/...), never executes
                # code.
                args = ast.literal_eval(raw_args)
            except (ValueError, SyntaxError):
                continue
        if not isinstance(args, dict):
            continue
        tool_calls.append(ToolCall(id=f"leaked-{i}", name=name, arguments=args))
        cleaned = cleaned.replace(m.group(0), "")

    if not tool_calls:
        return content, tool_calls
    cleaned = _LEAKED_WRAPPER_TAG_RE.sub("", cleaned).strip()
    return cleaned, tool_calls


# ---------------------------------------------------------------------------
# Real OpenAI-compatible client
# ---------------------------------------------------------------------------


class OpenAIModelClient(ModelClient):
    """Real model client using the ``openai`` package.

    Parameters
    ----------
    base_url:
        OpenAI-compatible endpoint (e.g. ``http://10.0.23.2:8086/v1``).
    api_key:
        API key (can be a dummy for local proxies that don't check).
    model:
        Model name to send in the request (e.g. ``gpt-4o-mini``).
    temperature:
        Default sampling temperature.
    timeout:
        HTTP timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        timeout: float = 300.0,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self._client = None

    def configure(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> None:
        """Live-update connection settings (e.g. from the admin settings page).

        ``model``/``temperature`` are read fresh on every :meth:`chat` call, so
        updating them here takes effect on the very next turn. ``base_url``/
        ``api_key``/``timeout`` are baked into the lazily-built OpenAI SDK
        client (:attr:`client`), so changing any of them invalidates the
        cached client — it is rebuilt (still lazily) on next use.
        """
        rebuild = False
        if base_url is not None and base_url != self.base_url:
            self.base_url = base_url
            rebuild = True
        if api_key is not None and api_key != self.api_key:
            self.api_key = api_key
            rebuild = True
        if timeout is not None and timeout != self.timeout:
            self.timeout = timeout
            rebuild = True
        if model is not None:
            self.model = model
        if temperature is not None:
            self.temperature = temperature
        if rebuild:
            self._client = None

    @property
    def client(self):
        """Lazy-init the OpenAI client (avoids import cost if not used)."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key or "not-needed",
                timeout=self.timeout,
                max_retries=2,
            )
        return self._client

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
    ) -> ChatResponse:
        """Send a chat request and return the (non-streaming) response."""
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0].message

        # Extract tool calls
        tool_calls: List[ToolCall] = []
        if choice.tool_calls:
            for tc in choice.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                    )
                )

        content = choice.content or ""
        if not tool_calls:
            # Some OpenAI-compatible backends, when the model's own
            # tool-calling chat template isn't wired up on the server side,
            # leak the model's raw tool-call markup into plain `content`
            # instead of the structured `tool_calls` field — recover it
            # rather than showing unreadable tag soup as the final answer
            # (and silently dropping the tool call the model intended).
            content, tool_calls = _parse_leaked_tool_calls(content)

        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return ChatResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            raw=response,
        )

    def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
    ) -> Iterator[str]:
        """Stream one round-trip: yield content deltas, return a ChatResponse.

        Uses the OpenAI-compatible ``stream=True`` path. Content deltas are
        yielded the moment they arrive, which is the whole point: the answer
        appears in the browser as the model writes it rather than in one
        burst at the end.

        Tool-call deltas arrive in fragments — the first carries the id and
        function name, the rest append JSON argument text — so they are
        accumulated per ``index`` and assembled into :class:`ToolCall` objects
        for the returned :class:`ChatResponse`.
        """
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            stream = self.client.chat.completions.create(**kwargs)
        except Exception:  # noqa: BLE001
            # Some OpenAI-compatible servers reject `stream_options`. Retry
            # once without it rather than failing the turn.
            kwargs.pop("stream_options", None)
            stream = self.client.chat.completions.create(**kwargs)

        content_parts: List[str] = []
        # index -> {"id", "name", "arguments"}
        partial: Dict[int, Dict[str, str]] = {}
        usage: Dict[str, int] = {}

        for chunk in stream:
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage:
                usage = {
                    "prompt_tokens": chunk_usage.prompt_tokens or 0,
                    "completion_tokens": chunk_usage.completion_tokens or 0,
                    "total_tokens": chunk_usage.total_tokens or 0,
                }
            if not chunk.choices:
                continue
            delta = getattr(chunk.choices[0], "delta", None)
            if delta is None:
                continue
            content = getattr(delta, "content", None)
            if content:
                content_parts.append(content)
                yield content
            for tc in getattr(delta, "tool_calls", None) or []:
                slot = partial.setdefault(
                    tc.index if tc.index is not None else 0,
                    {"id": "", "name": "", "arguments": ""},
                )
                if tc.id:
                    slot["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if fn.name:
                        slot["name"] = fn.name
                    if fn.arguments:
                        slot["arguments"] += fn.arguments

        tool_calls: List[ToolCall] = []
        for index in sorted(partial):
            slot = partial[index]
            if not slot["name"]:
                continue
            try:
                args = json.loads(slot["arguments"] or "{}")
            except (json.JSONDecodeError, TypeError):
                args = {}
            if not isinstance(args, dict):
                args = {}
            tool_calls.append(
                ToolCall(id=slot["id"] or f"call-{index}", name=slot["name"], arguments=args)
            )

        text = "".join(content_parts)
        if not tool_calls:
            # Same recovery as the non-streaming path: a backend whose
            # tool-call template isn't wired up leaks the markup into content.
            cleaned, tool_calls = _parse_leaked_tool_calls(text)
            if tool_calls:
                text = cleaned
        return ChatResponse(content=text, tool_calls=tool_calls, usage=usage)

    def generate_title(self, text: str) -> str:
        """Make a tiny LLM call to produce a short thread title.

        A separate non-streaming request with a minimal system prompt. If the
        model fails or returns nothing, fall back to the heuristic (truncated
        first line) so the thread always gets a usable title.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": (
                        "You generate a concise, human-friendly title (max 6 words) "
                        "for a chat thread based on the user's question. Reply with "
                        "just the title, no punctuation at the end, no quotes, no "
                        "explanation."
                    )},
                    {"role": "user", "content": text},
                ],
                temperature=0.2,
                # Generous headroom: reasoning-heavy models emit reasoning
                # tokens before any content, so a tiny cap returns an empty
                # title and we silently fall back to the heuristic.
                max_tokens=512,
            )
            content = (response.choices[0].message.content or "").strip()
            title = content.replace("\n", " ").strip().strip("\"'.,") 
            if title and len(title) <= 60:
                return title
        except Exception:  # noqa: BLE001
            pass
        # Fallback heuristic.
        first = text.splitlines()[0].strip() if text.strip() else "New Thread"
        return first[:60] or "New Thread"


# ---------------------------------------------------------------------------
# Fake client for offline tests
# ---------------------------------------------------------------------------


@dataclass
class FakeModelClient(ModelClient):
    """Offline test double for :class:`ModelClient`.

    Returns a pre-configured sequence of responses (one per call). When the
    sequence is exhausted, the last response is returned repeatedly.

    Parameters
    ----------
    responses:
        A list of :class:`ChatResponse` to return in order. If empty,
        returns a default "Hello!" text response.
    """

    responses: List[ChatResponse] = field(default_factory=list)
    calls: List[Dict[str, Any]] = field(default_factory=list)
    fake_titles: List[str] = field(default_factory=list)
    _call_index: int = field(default=0, repr=False)

    def generate_title(self, text: str) -> str:
        """Return a canned title, or a heuristic fallback."""
        if self.fake_titles:
            return self.fake_titles[0]
        self.calls.append({"op": "generate_title", "text": text})
        return super().generate_title(text)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
    ) -> ChatResponse:
        """Record the call and return the next canned response."""
        self.calls.append({
            "messages": messages,
            "tools": tools,
            "temperature": temperature,
        })
        if not self.responses:
            return ChatResponse(content="Hello!", tool_calls=[], usage={})
        idx = min(self._call_index, len(self.responses) - 1)
        self._call_index += 1
        return self.responses[idx]

    @property
    def last_call(self) -> Optional[Dict[str, Any]]:
        """The most recent recorded call."""
        return self.calls[-1] if self.calls else None

    @property
    def call_count(self) -> int:
        """Number of times chat() has been called."""
        return len(self.calls)

    def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
    ) -> Iterator[str]:
        """Replay the next canned response as simulated tokens.

        Yields the content in small chunks (word-ish boundaries) so tests can
        exercise the streaming path offline, then returns that
        :class:`ChatResponse` — tool calls included — so the fake honours the
        same contract as the real client.
        """
        self.calls.append({
            "messages": messages,
            "tools": tools,
            "temperature": temperature,
            "stream": True,
        })
        if not self.responses:
            return ChatResponse(content="", tool_calls=[], usage={})
        idx = min(self._call_index, len(self.responses) - 1)
        self._call_index += 1
        response = self.responses[idx]
        for chunk in _chunk_text(response.content):
            yield chunk
        return response


def _chunk_text(text: str) -> Iterator[str]:
    """Split *text* into small substrings suitable for token-stream faking."""
    words = text.split(" ")
    out = []
    for w in words:
        if out and len(out[-1]) < 20:
            out[-1] += " " + w
        else:
            # Every chunk after the first must carry the space that
            # separated it from the previous word — callers concatenate
            # chunks directly, so a bare `w` here would glue words together.
            out.append((" " if out else "") + w)
    return iter(out)


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def create_model_client(
    base_url: str,
    api_key: str = "",
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
    timeout: float = 300.0,
) -> ModelClient:
    """Create a real :class:`OpenAIModelClient` from config values.

    Tests should NOT call this — they use :class:`FakeModelClient` directly.
    """
    return OpenAIModelClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
        timeout=timeout,
    )
