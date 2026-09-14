"""Tests for :mod:`pengyplexity.core.modelclient`.

These verify:
* ``ChatResponse`` / ``ToolCall`` dataclasses behave correctly.
* ``FakeModelClient`` records calls and returns canned responses in sequence.
* ``OpenAIModelClient`` is constructable from config (no network call at
  construction).
* The abstract ``ModelClient`` interface is satisfied by both real and fake.
* ``create_model_client`` factory returns the right type.

All offline — no network, no real model endpoint.
"""

from __future__ import annotations

import pytest

from pengyplexity.core.modelclient import (
    ChatResponse,
    FakeModelClient,
    ModelClient,
    OpenAIModelClient,
    ToolCall,
    _parse_leaked_tool_calls,
    create_model_client,
)


# ---------------------------------------------------------------------------
# ChatResponse / ToolCall
# ---------------------------------------------------------------------------


class TestChatResponse:
    def test_default_empty(self):
        r = ChatResponse()
        assert r.content == ""
        assert r.tool_calls == []
        assert r.usage == {}
        assert r.has_tool_calls is False

    def test_text_response(self):
        r = ChatResponse(content="Hello world", usage={"total_tokens": 10})
        assert r.content == "Hello world"
        assert r.has_tool_calls is False

    def test_tool_call_response(self):
        tc = ToolCall(id="call_1", name="web_search", arguments={"query": "cats"})
        r = ChatResponse(content="", tool_calls=[tc])
        assert r.has_tool_calls is True
        assert r.tool_calls[0].name == "web_search"
        assert r.tool_calls[0].arguments == {"query": "cats"}

    def test_multiple_tool_calls(self):
        tcs = [
            ToolCall(id="c1", name="web_search", arguments={"query": "a"}),
            ToolCall(id="c2", name="fetch_url", arguments={"url": "http://x"}),
        ]
        r = ChatResponse(tool_calls=tcs)
        assert len(r.tool_calls) == 2
        assert r.has_tool_calls is True


# ---------------------------------------------------------------------------
# ModelClient interface
# ---------------------------------------------------------------------------


class TestModelClientInterface:
    def test_fake_is_model_client(self):
        assert isinstance(FakeModelClient(), ModelClient)

    def test_openai_client_is_model_client(self):
        c = OpenAIModelClient(base_url="http://localhost:9999/v1", model="test")
        assert isinstance(c, ModelClient)


# ---------------------------------------------------------------------------
# FakeModelClient
# ---------------------------------------------------------------------------


class TestFakeModelClient:
    def test_no_responses_returns_hello(self):
        fc = FakeModelClient()
        r = fc.chat([{"role": "user", "content": "hi"}])
        assert r.content == "Hello!"
        assert r.has_tool_calls is False

    def test_generate_title_returns_canned_title(self):
        fc = FakeModelClient(fake_titles=["Capital of France"])
        assert fc.generate_title("What is the capital of France?") == "Capital of France"

    def test_generate_title_falls_back_to_heuristic(self):
        fc = FakeModelClient()
        title = fc.generate_title("What is the capital of France?")
        assert title == "What is the capital of France?"

    def test_returns_responses_in_order(self):
        r1 = ChatResponse(content="first")
        r2 = ChatResponse(content="second")
        fc = FakeModelClient(responses=[r1, r2])

        resp1 = fc.chat([{"role": "user", "content": "1"}])
        resp2 = fc.chat([{"role": "user", "content": "2"}])
        assert resp1.content == "first"
        assert resp2.content == "second"

    def test_exhausted_sequence_repeats_last(self):
        r1 = ChatResponse(content="only")
        fc = FakeModelClient(responses=[r1])
        r1 = fc.chat([{"role": "user", "content": "a"}])
        r2 = fc.chat([{"role": "user", "content": "b"}])
        r3 = fc.chat([{"role": "user", "content": "c"}])
        assert r1.content == "only"
        assert r2.content == "only"  # repeats last
        assert r3.content == "only"

    def test_records_calls(self):
        fc = FakeModelClient()
        fc.chat([{"role": "user", "content": "hi"}], tools=[{"name": "web_search"}])
        assert fc.call_count == 1
        call = fc.last_call
        assert call is not None
        assert call["messages"] == [{"role": "user", "content": "hi"}]
        assert call["tools"] == [{"name": "web_search"}]

    def test_records_multiple_calls(self):
        fc = FakeModelClient()
        fc.chat([{"role": "user", "content": "1"}])
        fc.chat([{"role": "user", "content": "2"}])
        assert fc.call_count == 2
        assert len(fc.calls) == 2
        assert fc.calls[0]["messages"][0]["content"] == "1"
        assert fc.calls[1]["messages"][0]["content"] == "2"

    def test_last_call_none_before_any_call(self):
        fc = FakeModelClient()
        assert fc.last_call is None

    def test_tool_call_response(self):
        tc = ToolCall(id="call_42", name="web_search", arguments={"query": "test"})
        fc = FakeModelClient(responses=[ChatResponse(tool_calls=[tc])])
        r = fc.chat([{"role": "user", "content": "search for test"}])
        assert r.has_tool_calls is True
        assert r.tool_calls[0].id == "call_42"
        assert r.tool_calls[0].name == "web_search"
        assert r.tool_calls[0].arguments == {"query": "test"}

    def test_temperature_recorded(self):
        fc = FakeModelClient()
        fc.chat([{"role": "user", "content": "hi"}], temperature=0.7)
        assert fc.last_call["temperature"] == 0.7

    def test_independent_instances(self):
        """Two FakeModelClients don't share state."""
        fc1 = FakeModelClient(responses=[ChatResponse(content="A")])
        fc2 = FakeModelClient(responses=[ChatResponse(content="B")])
        assert fc1.chat([{"role": "user", "content": "x"}]).content == "A"
        assert fc2.chat([{"role": "user", "content": "x"}]).content == "B"
        assert fc1.call_count == 1
        assert fc2.call_count == 1


# ---------------------------------------------------------------------------
# OpenAIModelClient (no network — just construction + config)
# ---------------------------------------------------------------------------


class TestOpenAIModelClient:
    def test_construction(self):
        c = OpenAIModelClient(
            base_url="http://127.0.0.1:9999/v1",
            api_key="test-key",
            model="gpt-4o-mini",
            temperature=0.5,
        )
        assert c.base_url == "http://127.0.0.1:9999/v1"
        assert c.api_key == "test-key"
        assert c.model == "gpt-4o-mini"
        assert c.temperature == 0.5

    def test_no_network_at_construction(self):
        """Creating the client doesn't hit the network."""
        # Would fail if it tried to connect.
        c = OpenAIModelClient(base_url="http://127.0.0.1:0/v1", model="x")
        assert c.model == "x"

    def test_client_property_lazy(self):
        """The underlying OpenAI client is only created on first access."""
        c = OpenAIModelClient(base_url="http://x", model="y")
        assert c._client is None
        # Don't actually access .client (would try to import openai or connect).
        # Just verify the attribute exists and is None initially.

    def test_is_model_client_subclass(self):
        c = OpenAIModelClient(base_url="http://x", model="y")
        assert isinstance(c, ModelClient)


# ---------------------------------------------------------------------------
# Recovering tool calls a misconfigured backend leaked as plain text
# ---------------------------------------------------------------------------


class TestParseLeakedToolCalls:
    def test_real_world_leaked_run_bash_call(self):
        # Exact shape observed from a real backend whose tool-call chat
        # template wasn't wired up: special tokens the server didn't
        # recognize decoded literally as junk (`｜｜DSML｜｜`) prefixed onto
        # otherwise-ordinary <invoke>/<parameter> tags, and the arguments
        # rendered as a Python dict literal (single-quoted) rather than JSON.
        content = (
            "I've got what I need. Let me put together a quick visual of the score trend.\n\n"
            '<｜｜DSML｜｜ calls>\n'
            '<｜｜DSML｜｜ invoke name="run_bash">\n'
            '<｜｜DSML｜｜ parameter name="arguments" string="false">'
            "{'command': 'pip install matplotlib --quiet'}"
            "</｜｜DSML｜｜ parameter>\n"
            "</｜｜DSML｜｜ invoke>\n"
            "</｜｜DSML｜｜ calls>"
        )

        cleaned, calls = _parse_leaked_tool_calls(content)

        assert cleaned == "I've got what I need. Let me put together a quick visual of the score trend."
        assert len(calls) == 1
        assert calls[0].name == "run_bash"
        assert calls[0].arguments == {"command": "pip install matplotlib --quiet"}

    def test_json_style_arguments(self):
        content = (
            '<invoke name="web_search">'
            '<parameter name="arguments">{"query": "cats"}</parameter>'
            "</invoke>"
        )
        cleaned, calls = _parse_leaked_tool_calls(content)
        assert cleaned == ""
        assert calls == [ToolCall(id="leaked-0", name="web_search", arguments={"query": "cats"})]

    def test_ordinary_text_is_untouched(self):
        content = "Just a normal answer with no tool calls in it."
        cleaned, calls = _parse_leaked_tool_calls(content)
        assert cleaned == content
        assert calls == []

    def test_unparsable_arguments_are_skipped(self):
        content = (
            '<invoke name="run_bash">'
            "<parameter name=\"arguments\">not a dict at all</parameter>"
            "</invoke>"
        )
        cleaned, calls = _parse_leaked_tool_calls(content)
        assert calls == []
        # Nothing recovered — the original text is left alone rather than
        # silently eaten.
        assert cleaned == content


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFactory:
    def test_create_model_client_returns_openai_client(self):
        c = create_model_client(
            base_url="http://10.0.23.2:8086/v1",
            api_key="k",
            model="gpt-4o-mini",
            temperature=0.3,
        )
        assert isinstance(c, OpenAIModelClient)
        assert isinstance(c, ModelClient)
        assert c.base_url == "http://10.0.23.2:8086/v1"
        assert c.model == "gpt-4o-mini"
        assert c.temperature == 0.3


# ---------------------------------------------------------------------------
# Integration: fake client simulating a tool loop
# ---------------------------------------------------------------------------


class TestSimulatedToolLoop:
    """Simulate the pattern the agent loop will use:
    model returns tool_calls → agent executes → sends tool result → model
    returns final answer."""

    def test_two_round_trip(self):
        tool_call_resp = ChatResponse(
            content="",
            tool_calls=[ToolCall(id="c1", name="web_search", arguments={"query": "cats"})],
        )
        final_resp = ChatResponse(
            content="Cats are great pets.",
            usage={"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105},
        )
        fc = FakeModelClient(responses=[tool_call_resp, final_resp])

        # Round 1: user asks question
        r1 = fc.chat([{"role": "user", "content": "Tell me about cats"}])
        assert r1.has_tool_calls is True

        # Agent would execute the tool, then send the result back.
        messages = [
            {"role": "user", "content": "Tell me about cats"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {
                    "name": "web_search", "arguments": '{"query": "cats"}'
                }}
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "Cat facts here..."},
        ]
        r2 = fc.chat(messages)
        assert r2.has_tool_calls is False
        assert r2.content == "Cats are great pets."
        assert r2.usage["total_tokens"] == 105

        # The fake recorded both calls.
        assert fc.call_count == 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
