"""Tests for interrupting a turn (the chat UI's Stop button).

Covers the three halves of a stop: the token and registry themselves, the
agent honouring the flag at its safe points, and the ``/stop`` route plus the
partial answer that has to survive the interruption.

All offline — no model, no network, no bwrap.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from pengyplexity.core.agent import Agent
from pengyplexity.core.cancel import CancelRegistry, CancelToken
from pengyplexity.core.modelclient import ChatResponse, FakeModelClient, ToolCall


class _FakeProc:
    """Stands in for a sandbox subprocess."""

    def __init__(self, alive: bool = True) -> None:
        self.alive = alive
        self.terminated = False
        self.killed = False

    def poll(self):
        return None if self.alive else 0

    def terminate(self):
        self.terminated = True
        self.alive = False

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.killed = True
        self.alive = False


# ---------------------------------------------------------------------------
# CancelToken
# ---------------------------------------------------------------------------


class TestCancelToken:
    def test_starts_uncancelled(self):
        token = CancelToken()
        assert token.cancelled is False
        token.raise_if_cancelled()  # must not raise

    def test_cancel_sets_the_flag(self):
        token = CancelToken()
        token.cancel()
        assert token.cancelled is True

    def test_cancel_kills_registered_processes(self):
        token = CancelToken()
        proc = _FakeProc()
        token.register_process(proc)

        token.cancel()

        # Without this, a Stop during a chart script left the user waiting
        # out the whole execution timeout.
        assert proc.terminated is True

    def test_finished_process_is_not_killed(self):
        token = CancelToken()
        proc = _FakeProc(alive=False)
        token.register_process(proc)

        token.cancel()

        assert proc.terminated is False
        assert proc.killed is False

    def test_unregistered_process_is_left_alone(self):
        token = CancelToken()
        proc = _FakeProc()
        token.register_process(proc)
        token.unregister_process(proc)

        token.cancel()

        assert proc.terminated is False

    def test_process_started_after_cancel_is_killed_at_once(self):
        token = CancelToken()
        token.cancel()
        proc = _FakeProc()

        token.register_process(proc)

        # The turn is already over; letting this one run would do work
        # nobody is waiting for.
        assert proc.terminated is True

    def test_cancel_is_safe_from_another_thread(self):
        token = CancelToken()
        procs = [_FakeProc() for _ in range(20)]
        for p in procs:
            token.register_process(p)

        threads = [threading.Thread(target=token.cancel) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert token.cancelled is True
        assert all(p.terminated for p in procs)


# ---------------------------------------------------------------------------
# CancelRegistry
# ---------------------------------------------------------------------------


class TestCancelRegistry:
    def test_cancel_finds_the_running_turn(self):
        registry = CancelRegistry()
        token = registry.start("alice", "t1")

        assert registry.cancel("alice", "t1") is True
        assert token.cancelled is True

    def test_cancelling_an_idle_thread_is_not_an_error(self):
        registry = CancelRegistry()
        assert registry.cancel("alice", "t1") is False

    def test_one_user_cannot_cancel_anothers_turn(self):
        registry = CancelRegistry()
        alice = registry.start("alice", "t1")

        # Same thread id, different owner.
        assert registry.cancel("bob", "t1") is False
        assert alice.cancelled is False

    def test_turns_in_different_threads_are_independent(self):
        registry = CancelRegistry()
        first = registry.start("alice", "t1")
        second = registry.start("alice", "t2")

        registry.cancel("alice", "t1")

        assert first.cancelled is True
        assert second.cancelled is False

    def test_a_new_turn_cancels_the_previous_one_in_that_thread(self):
        registry = CancelRegistry()
        first = registry.start("alice", "t1")

        second = registry.start("alice", "t1")

        # Two live turns appending to the same message list would interleave.
        assert first.cancelled is True
        assert second.cancelled is False

    def test_finish_deregisters(self):
        registry = CancelRegistry()
        token = registry.start("alice", "t1")
        registry.finish("alice", "t1", token)

        assert len(registry) == 0
        assert registry.cancel("alice", "t1") is False

    def test_a_late_finish_does_not_deregister_the_newer_turn(self):
        registry = CancelRegistry()
        old = registry.start("alice", "t1")
        new = registry.start("alice", "t1")

        registry.finish("alice", "t1", old)

        assert registry.active("alice", "t1") is new


# ---------------------------------------------------------------------------
# The agent honours the token
# ---------------------------------------------------------------------------


def _executor(calls):
    def execute(name, args, workspace):
        calls.append(name)
        return "tool result"

    return execute


class TestAgentCancellation:
    def test_stream_stops_between_round_trips(self):
        token = CancelToken()
        calls = []
        model = FakeModelClient(responses=[
            ChatResponse(content="first ",
                         tool_calls=[ToolCall(id="c1", name="web_search",
                                              arguments={"query": "q"})]),
            ChatResponse(content="second"),
        ])
        agent = Agent(model=model, tool_executor=_executor(calls),
                      workspace=Path("/ws"), cancel=token)

        events = []
        for event in agent.run_stream("hi"):
            events.append(event)
            if event.kind == "activity":
                token.cancel()  # the user presses Stop mid-turn

        tokens = "".join(e.data["content"] for e in events if e.kind == "token")
        # What the model already wrote is kept; the next round-trip never runs.
        assert tokens == "first "
        assert model.call_count == 1

    def test_remaining_tool_calls_are_skipped(self):
        token = CancelToken()
        calls = []
        model = FakeModelClient(responses=[
            ChatResponse(tool_calls=[
                ToolCall(id="c1", name="web_search", arguments={"query": "a"}),
                ToolCall(id="c2", name="web_search", arguments={"query": "b"}),
            ]),
            ChatResponse(content="done"),
        ])
        agent = Agent(model=model, tool_executor=_executor(calls),
                      workspace=Path("/ws"), cancel=token)

        for event in agent.run_stream("hi"):
            if event.kind == "activity":
                token.cancel()

        assert calls == []

    def test_already_cancelled_turn_does_nothing(self):
        token = CancelToken()
        token.cancel()
        calls = []
        model = FakeModelClient(responses=[ChatResponse(content="hello")])
        agent = Agent(model=model, tool_executor=_executor(calls),
                      workspace=Path("/ws"), cancel=token)

        assert list(agent.run_stream("hi")) == []
        assert model.call_count == 0

    def test_no_token_means_no_change_in_behaviour(self):
        calls = []
        model = FakeModelClient(responses=[ChatResponse(content="hello world")])
        agent = Agent(model=model, tool_executor=_executor(calls),
                      workspace=Path("/ws"))

        events = list(agent.run_stream("hi"))

        assert "".join(e.data["content"] for e in events if e.kind == "token") == "hello world"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
