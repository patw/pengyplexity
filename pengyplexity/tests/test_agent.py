"""Tests for :mod:`pengyplexity.core.agent`.

These test the agent loop with a **fake model** and **fake tool executor** —
fully offline, no network, no real tools. The tests verify:

* The agent runs the tool-call loop correctly (tool → result → next call).
* It stops when the model returns no tool_calls (final answer).
* It caps iterations on a runaway model (max_iterations).
* It extracts sources from web_search / fetch_url results.
* It builds the correct message history (system prompt + user + tool msgs).
* The system prompt contains the safety warnings.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pengyplexity.core.agent import (
    SYSTEM_PROMPT,
    Agent,
    AgentResult,
)
from pengyplexity.core.modelclient import (
    ChatResponse,
    FakeModelClient,
    ModelClient,
    ToolCall,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _fake_tool_executor(results: dict | None = None):
    """Create a fake tool executor that records calls and returns canned results.

    *results* maps tool name → string result (default: "ok").
    Returns (executor, calls_list).
    """
    calls: list[tuple[str, dict, Path]] = []
    defaults = results or {}

    def executor(name: str, args: dict, workspace: Path) -> str:
        calls.append((name, args, workspace))
        return defaults.get(name, "ok")

    return executor, calls


def _search_result_json(urls: list[tuple[str, str]]) -> str:
    """Format a web_search result as the JSON list the executor would return."""
    items = [{"title": t, "href": u} for t, u in urls]
    return json.dumps(items)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    def test_mentions_web_search(self):
        assert "web_search" in SYSTEM_PROMPT

    def test_mentions_workspace(self):
        assert "workspace" in SYSTEM_PROMPT

    def test_warns_against_sudo(self):
        assert "sudo" in SYSTEM_PROMPT.lower() or "privilege" in SYSTEM_PROMPT.lower()

    def test_warns_against_escape(self):
        assert "escape" in SYSTEM_PROMPT.lower()

    def test_says_cite_sources(self):
        assert "source" in SYSTEM_PROMPT.lower()

    def test_forbids_self_modification(self):
        assert "never modify" in SYSTEM_PROMPT.lower() or "never" in SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------------------
# Basic agent: no tools, immediate answer
# ---------------------------------------------------------------------------


class TestNoToolCalls:
    def test_immediate_answer(self):
        model = FakeModelClient(
            responses=[ChatResponse(content="The answer is 42.")]
        )
        executor, calls = _fake_tool_executor()
        agent = Agent(model=model, tool_executor=executor, workspace=Path("/ws"))

        result = agent.run("What is the meaning of life?")

        assert result.answer == "The answer is 42."
        assert result.iterations == 1
        assert result.sources == []
        assert len(calls) == 0  # no tools executed

    def test_system_prompt_is_first_message(self):
        model = FakeModelClient(
            responses=[ChatResponse(content="Hi")]
        )
        executor, _ = _fake_tool_executor()
        agent = Agent(model=model, tool_executor=executor, workspace=Path("/ws"))
        agent.run("Hello")

        # The model received the system prompt as the first message.
        first_msg = model.last_call["messages"][0]
        assert first_msg["role"] == "system"
        assert "Pengyplexity" in first_msg["content"]

    def test_user_message_is_included(self):
        model = FakeModelClient(
            responses=[ChatResponse(content="ok")]
        )
        executor, _ = _fake_tool_executor()
        agent = Agent(model=model, tool_executor=executor, workspace=Path("/ws"))
        agent.run("My question")

        msgs = model.last_call["messages"]
        assert any(m["content"] == "My question" for m in msgs)


# ---------------------------------------------------------------------------
# Agent loop: one tool call then final answer
# ---------------------------------------------------------------------------


class TestOneToolCall:
    def test_search_then_answer(self):
        search_result = _search_result_json([
            ("Cat Facts", "https://cats.example.com"),
            ("Kittens", "https://kittens.example.org"),
        ])
        model = FakeModelClient(
            responses=[
                # Round 1: model wants to search.
                ChatResponse(
                    content="",
                    tool_calls=[ToolCall(id="c1", name="web_search",
                                         arguments={"query": "cats"})],
                ),
                # Round 2: model gives final answer.
                ChatResponse(content="Cats are great. [1]"),
            ]
        )
        executor, calls = _fake_tool_executor(
            results={"web_search": search_result}
        )
        agent = Agent(model=model, tool_executor=executor, workspace=Path("/ws"))

        result = agent.run("Tell me about cats")

        assert result.answer == "Cats are great. [1]"
        assert result.iterations == 2
        # Tool was called once.
        assert len(calls) == 1
        assert calls[0][0] == "web_search"
        assert calls[0][1] == {"query": "cats"}
        # Sources extracted from the search result.
        assert len(result.sources) == 2
        assert result.sources[0] == {"title": "Cat Facts", "url": "https://cats.example.com"}
        assert result.sources[1] == {"title": "Kittens", "url": "https://kittens.example.org"}

    def test_tool_result_is_in_messages(self):
        model = FakeModelClient(
            responses=[
                ChatResponse(
                    tool_calls=[ToolCall(id="c1", name="web_search",
                                         arguments={"query": "x"})],
                ),
                ChatResponse(content="done"),
            ]
        )
        executor, _ = _fake_tool_executor(results={"web_search": "tool output here"})
        agent = Agent(model=model, tool_executor=executor, workspace=Path("/ws"))
        agent.run("q")

        # The second model call should include the tool result.
        msgs = model.calls[1]["messages"]
        tool_msgs = [m for m in msgs if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["content"] == "tool output here"
        assert tool_msgs[0]["tool_call_id"] == "c1"


# ---------------------------------------------------------------------------
# Agent loop: multiple tool calls
# ---------------------------------------------------------------------------


class TestMultipleToolCalls:
    def test_search_and_fetch(self):
        search_result = _search_result_json([
            ("Article", "https://example.com/article"),
        ])
        model = FakeModelClient(
            responses=[
                ChatResponse(
                    tool_calls=[
                        ToolCall(id="c1", name="web_search", arguments={"query": "test"}),
                        ToolCall(id="c2", name="fetch_url", arguments={"url": "https://example.com"}),
                    ],
                ),
                ChatResponse(content="Final answer with sources."),
            ]
        )
        executor, calls = _fake_tool_executor(
            results={"web_search": search_result, "fetch_url": "<html>body</html>"}
        )
        agent = Agent(model=model, tool_executor=executor, workspace=Path("/ws"))
        result = agent.run("Research topic")

        assert result.answer == "Final answer with sources."
        assert result.iterations == 2
        assert len(calls) == 2
        # Sources from both search and fetch.
        urls = {s["url"] for s in result.sources}
        assert "https://example.com/article" in urls
        assert "https://example.com" in urls


# ---------------------------------------------------------------------------
# Iteration cap (runaway model)
# ---------------------------------------------------------------------------


class TestIterationCap:
    def test_runaway_model_hits_cap(self):
        """A model that always returns tool_calls is stopped at max_iterations."""
        # Always wants to search — never gives a final answer.
        model = FakeModelClient(
            responses=[
                ChatResponse(
                    tool_calls=[ToolCall(id=f"c{i}", name="web_search",
                                         arguments={"query": f"q{i}"})],
                )
                for i in range(20)  # more than the cap
            ]
        )
        executor, calls = _fake_tool_executor()
        agent = Agent(
            model=model,
            tool_executor=executor,
            workspace=Path("/ws"),
            max_iterations=3,
        )

        result = agent.run("Loop forever")

        # Stopped at the cap.
        assert result.iterations == 3
        assert "Stopped" in result.answer or "maximum" in result.answer.lower()
        # Exactly 3 tool executions (one per iteration).
        assert len(calls) == 3

    def test_custom_max_iterations(self):
        model = FakeModelClient(
            responses=[
                ChatResponse(
                    tool_calls=[ToolCall(id="c", name="glob", arguments={"pattern": "*"})],
                )
            ]  # repeats last forever
        )
        executor, calls = _fake_tool_executor()
        agent = Agent(
            model=model,
            tool_executor=executor,
            workspace=Path("/ws"),
            max_iterations=1,
        )
        result = agent.run("cap at 1")
        assert result.iterations == 1
        assert len(calls) == 1

    def test_last_iteration_withholds_tools_to_force_synthesis(self):
        # A more realistic model: it only avoids returning tool_calls when
        # no tools were offered — matches how a real tool-calling API
        # behaves. This proves the agent forces a genuine synthesized
        # answer out of whatever was gathered, instead of ending on the
        # bare "[Stopped: reached maximum iterations]" placeholder.
        class ToolsAwareModel(ModelClient):
            def chat(self, messages, tools=None, temperature=None):
                if tools is None:
                    return ChatResponse(content="Here's what I found so far: it's well-liked.")
                return ChatResponse(
                    content="",
                    tool_calls=[ToolCall(id="c", name="web_search", arguments={"query": "q"})],
                )

        model = ToolsAwareModel()
        executor, calls = _fake_tool_executor()
        agent = Agent(
            model=model,
            tool_executor=executor,
            workspace=Path("/ws"),
            max_iterations=3,
            tools=[{"type": "function", "function": {"name": "web_search"}}],
        )

        result = agent.run("keep searching forever")

        assert result.iterations == 3
        assert "Here's what I found so far: it's well-liked." in result.answer
        assert "Stopped" not in result.answer
        # Only 2 tool executions — the 3rd (last) iteration got no tools.
        assert len(calls) == 2


# ---------------------------------------------------------------------------
# Source extraction
# ---------------------------------------------------------------------------


class TestSourceExtraction:
    def test_dedup_by_url(self):
        search_result = _search_result_json([
            ("Page A", "https://same.com"),
            ("Page B", "https://same.com"),  # duplicate URL
        ])
        model = FakeModelClient(
            responses=[
                ChatResponse(
                    tool_calls=[ToolCall(id="c1", name="web_search", arguments={"query": "x"})],
                ),
                ChatResponse(content="done"),
            ]
        )
        executor, _ = _fake_tool_executor(results={"web_search": search_result})
        agent = Agent(model=model, tool_executor=executor, workspace=Path("/ws"))
        result = agent.run("q")

        # Only one source for the duplicate URL.
        assert len(result.sources) == 1

    def test_parses_production_text_format(self):
        # This is the format core.search.SearchService.search() actually
        # returns in production (DDGSSearchService / FakeSearchService),
        # NOT JSON — regression test for the title showing up as the
        # literal word "URL" instead of the real result title.
        search_result = (
            "1. Real Title A\n"
            "   URL: https://a.example\n"
            "   some snippet text\n"
            "\n"
            "2. Real Title B\n"
            "   URL: https://b.example\n"
        )
        model = FakeModelClient(
            responses=[
                ChatResponse(
                    tool_calls=[ToolCall(id="c1", name="web_search", arguments={"query": "x"})],
                ),
                ChatResponse(content="done"),
            ]
        )
        executor, _ = _fake_tool_executor(results={"web_search": search_result})
        agent = Agent(model=model, tool_executor=executor, workspace=Path("/ws"))
        result = agent.run("q")

        assert result.sources == [
            {"title": "Real Title A", "url": "https://a.example"},
            {"title": "Real Title B", "url": "https://b.example"},
        ]

    def test_content_alongside_tool_call_is_not_dropped(self):
        # Regression test: some models write the real answer AND call a
        # tool (e.g. save_memory) in the same turn. The turn's `content`
        # must not be discarded just because it also had tool_calls —
        # otherwise the user only ever sees the short wrap-up from the
        # NEXT turn ("I've noted that for later.") and the actual answer
        # vanishes.
        tc = ToolCall(id="c1", name="save_memory", arguments={"title": "x", "summary": "y"})
        model = FakeModelClient(responses=[
            ChatResponse(content="Here is the full answer to your question.", tool_calls=[tc]),
            ChatResponse(content="I've noted that for later."),
        ])
        executor, _ = _fake_tool_executor()
        agent = Agent(model=model, tool_executor=executor, workspace=Path("/ws"))

        result = agent.run("question")

        assert "Here is the full answer to your question." in result.answer
        assert "I've noted that for later." in result.answer

    def test_no_sources_for_non_search_tools(self):
        model = FakeModelClient(
            responses=[
                ChatResponse(
                    tool_calls=[ToolCall(id="c1", name="write_file",
                                         arguments={"path": "f.txt", "content": "hi"})],
                ),
                ChatResponse(content="wrote file"),
            ]
        )
        executor, _ = _fake_tool_executor()
        agent = Agent(model=model, tool_executor=executor, workspace=Path("/ws"))
        result = agent.run("write something")

        assert result.sources == []


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


class TestHistory:
    def test_history_messages_included(self):
        model = FakeModelClient(
            responses=[ChatResponse(content="continuing...")]
        )
        executor, _ = _fake_tool_executor()
        agent = Agent(model=model, tool_executor=executor, workspace=Path("/ws"))

        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]
        agent.run("Follow up", history=history)

        msgs = model.last_call["messages"]
        # system + history + new user message
        assert msgs[0]["role"] == "system"
        assert any(m["content"] == "Previous question" for m in msgs)
        assert any(m["content"] == "Previous answer" for m in msgs)
        assert any(m["content"] == "Follow up" for m in msgs)


# ---------------------------------------------------------------------------
# Tools schema passed to model
# ---------------------------------------------------------------------------


class TestToolsSchema:
    def test_tools_passed_to_model(self):
        tools = [{"type": "function", "function": {"name": "web_search"}}]
        model = FakeModelClient(
            responses=[ChatResponse(content="ok")]
        )
        executor, _ = _fake_tool_executor()
        agent = Agent(
            model=model,
            tool_executor=executor,
            workspace=Path("/ws"),
            tools=tools,
        )
        agent.run("q")

        assert model.last_call["tools"] == tools

    def test_no_tools_means_none(self):
        model = FakeModelClient(
            responses=[ChatResponse(content="ok")]
        )
        executor, _ = _fake_tool_executor()
        agent = Agent(model=model, tool_executor=executor, workspace=Path("/ws"))
        agent.run("q")

        assert model.last_call["tools"] is None


# ---------------------------------------------------------------------------
# Tool error handling
# ---------------------------------------------------------------------------


class TestToolErrorHandling:
    def test_tool_exception_returns_error_string(self):
        model = FakeModelClient(
            responses=[
                ChatResponse(
                    tool_calls=[ToolCall(id="c1", name="run_python",
                                         arguments={"code": "bad"})],
                ),
                ChatResponse(content="It failed."),
            ]
        )

        def failing_executor(name, args, ws):
            raise RuntimeError("sandbox crashed")

        agent = Agent(model=model, tool_executor=failing_executor, workspace=Path("/ws"))
        result = agent.run("run code")

        # Agent should NOT crash; it reports the error in messages.
        assert result.answer == "It failed."
        # The tool error message was sent back to the model.
        msgs = model.calls[1]["messages"]
        tool_msgs = [m for m in msgs if m.get("role") == "tool"]
        assert "Tool error" in tool_msgs[0]["content"]


# ---------------------------------------------------------------------------
# AgentResult shape
# ---------------------------------------------------------------------------


class TestAgentResult:
    def test_result_is_dataclass(self):
        model = FakeModelClient(responses=[ChatResponse(content="hi")])
        executor, _ = _fake_tool_executor()
        agent = Agent(model=model, tool_executor=executor, workspace=Path("/ws"))
        result = agent.run("q")

        assert isinstance(result, AgentResult)
        assert isinstance(result.answer, str)
        assert isinstance(result.sources, list)
        assert isinstance(result.iterations, int)
        assert isinstance(result.messages, list)


# ---------------------------------------------------------------------------
# Streaming answer (run_stream)
# ---------------------------------------------------------------------------


class TestRunStream:
    def test_streams_immediate_answer(self):
        model = FakeModelClient(responses=[ChatResponse(content="The answer is 42.")])
        executor, calls = _fake_tool_executor()
        agent = Agent(model=model, tool_executor=executor, workspace=Path("/ws"))

        events = list(agent.run_stream("What is the answer?"))
        # Only token events (no tool call, no activity).
        assert all(e.kind == "token" for e in events)
        assert "".join(e.data["content"] for e in events) == "The answer is 42."
        assert len(calls) == 0  # no tool call

    def test_streams_after_tool_call_emits_activity(self):
        # First response requests a tool call; second is the final answer.
        search_args = {"query": "meaning of life"}
        tc = ToolCall(id="call_1", name="web_search", arguments=search_args)
        model = FakeModelClient(responses=[
            ChatResponse(content="", tool_calls=[tc]),
            ChatResponse(content="42 (from search)."),
        ])
        executor, calls = _fake_tool_executor({
            "web_search": _search_result_json([("A Source", "https://a.example")]),
        })
        agent = Agent(model=model, tool_executor=executor, workspace=Path("/ws"))

        events = list(agent.run_stream("What is the answer?"))
        # An activity event was emitted for the tool, then token events.
        kinds = [e.kind for e in events]
        assert "activity" in kinds
        act = next(e for e in events if e.kind == "activity")
        assert act.data["tool"] == "web_search"
        assert "Search" in act.data["label"]
        assert "".join(e.data["content"] for e in events if e.kind == "token") == "42 (from search)."
        # The tool call was executed before streaming resumed.
        assert len(calls) == 1
        # Sources were recorded from the search result.
        assert agent._last_sources == [{"title": "A Source", "url": "https://a.example"}]

    def test_content_alongside_tool_call_is_streamed(self):
        # Streaming counterpart of TestSourceExtraction's
        # test_content_alongside_tool_call_is_not_dropped.
        tc = ToolCall(id="c1", name="save_memory", arguments={"title": "x", "summary": "y"})
        model = FakeModelClient(responses=[
            ChatResponse(content="Here is the full answer.", tool_calls=[tc]),
            ChatResponse(content="I've noted that for later."),
        ])
        executor, _ = _fake_tool_executor()
        agent = Agent(model=model, tool_executor=executor, workspace=Path("/ws"))

        events = list(agent.run_stream("question"))
        full_text = "".join(e.data["content"] for e in events if e.kind == "token")
        assert "Here is the full answer." in full_text
        assert "I've noted that for later." in full_text

    def test_long_answer_preserves_spaces_across_chunks(self):
        # Regression test: chunk boundaries (every ~20 chars) must not eat
        # the space between the last word of one chunk and the first word
        # of the next — the browser reassembles the answer by concatenating
        # streamed chunks directly.
        long_answer = (
            "The honest take: the buzz is mixed but curious, and critics "
            "note that twenty two years of chemistry isn't something you "
            "can simply cast for a reboot."
        )
        model = FakeModelClient(responses=[ChatResponse(content=long_answer)])
        executor, _ = _fake_tool_executor()
        agent = Agent(model=model, tool_executor=executor, workspace=Path("/ws"))

        events = list(agent.run_stream("Tell me about the reboot"))
        assert "".join(e.data["content"] for e in events) == long_answer

    def test_final_turn_makes_exactly_one_streamed_request(self):
        model = FakeModelClient(responses=[ChatResponse(content="hello world")])
        executor, _ = _fake_tool_executor()
        agent = Agent(model=model, tool_executor=executor, workspace=Path("/ws"))

        events = list(agent.run_stream("hi"))

        # One request per round-trip, and it is the streaming one: the same
        # call both streams the prose and reports whether the model wanted a
        # tool, so there is never a second request that could re-sample a
        # different answer than the one already shown to the user.
        assert model.call_count == 1
        assert all(c.get("stream") for c in model.calls)
        tokens = "".join(e.data["content"] for e in events if e.kind == "token")
        assert tokens == "hello world"

    def test_streamed_tool_turn_uses_one_request_per_round_trip(self):
        model = FakeModelClient(responses=[
            ChatResponse(tool_calls=[ToolCall(id="c1", name="web_search",
                                              arguments={"query": "q"})]),
            ChatResponse(content="final answer"),
        ])
        executor, calls = _fake_tool_executor()
        agent = Agent(model=model, tool_executor=executor, workspace=Path("/ws"))

        events = list(agent.run_stream("hi"))

        assert model.call_count == 2
        assert all(c.get("stream") for c in model.calls)
        assert [c[0] for c in calls] == ["web_search"]
        tokens = "".join(e.data["content"] for e in events if e.kind == "token")
        assert tokens == "final answer"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
