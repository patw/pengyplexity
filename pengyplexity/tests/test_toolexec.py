"""Tests for the production tool executor wiring.

These stay offline: the runner is a :class:`FakeRunner` (records the command,
never launches bwrap) and the search service is a :class:`FakeSearchService`.
They exercise the executor's tool dispatch, path confinement, and the schema
build independently of Pengy.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pengyplexity.core.memory import MemoryStore
from pengyplexity.core.search import FakeSearchService
from pengyplexity.core.toolexec import (
    _resolve,
    build_tool_executor,
    get_tool_schemas,
)
from pengyplexity.sandbox.confine import OutsideWorkspaceError
from pengyplexity.sandbox.executors import FakeRunner, RunResult


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    d = tmp_path / "ws"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def executor(ws: Path):
    runner = FakeRunner(result=RunResult(stdout="hello", stderr="", returncode=0))
    search = FakeSearchService(
        results=[("Result One", "https://example.com", "snippet")],
    )
    return build_tool_executor(runner, search)


def test_get_tool_schemas_is_an_allowlist_subset():
    schemas = get_tool_schemas()
    names = {s["function"]["name"] for s in schemas}
    from pengyplexity.sandbox.toolpolicy import SAFE_TOOLS
    assert names == SAFE_TOOLS
    # no elevated param on run_bash after scrubbing
    for s in schemas:
        if s["function"]["name"] == "run_bash":
            assert "elevated" not in s["function"]["parameters"]["properties"]


def test_executor_reads_confined_file(ws, executor):
    (ws / "note.txt").write_text("hello world")
    out = executor("read_file", {"path": "note.txt"}, ws)
    assert "hello world" in out


def test_executor_rejects_escape(ws, executor):
    out = executor("read_file", {"path": "../../etc/passwd"}, ws)
    assert "Blocked" in out


def test_executor_writes_then_writes_overwrite(ws, executor):
    out = executor("write_file", {"path": "a.txt", "content": "x"}, ws)
    assert "Wrote" in out
    out = executor("write_file", {"path": "a.txt", "content": "y"}, ws)
    assert "exists" in out
    out = executor("write_file", {"path": "a.txt", "content": "y", "overwrite": True}, ws)
    assert "Wrote" in out


def test_executor_runs_python_via_runner(ws, executor):
    out = executor("run_python", {"code": "print(1)"}, ws)
    assert "hello" in out


def test_executor_search_and_fetch(ws, executor):
    out = executor("web_search", {"query": "cats", "max_results": 2}, ws)
    assert "Result One" in out
    out = executor("fetch_url", {"url": "https://example.com"}, ws)
    # FakeSearchService.fetch returns the URL as the body when not mapped
    assert out  # no error


def test_executor_caps_web_search_max_results(ws):
    # A model can pass an arbitrarily large max_results; the executor must
    # clamp it server-side rather than passing it straight through.
    search = FakeSearchService(
        results=[("Result One", "https://example.com", "snippet")],
    )
    runner = FakeRunner(result=RunResult(stdout="hello", stderr="", returncode=0))
    executor = build_tool_executor(runner, search)

    executor("web_search", {"query": "cats", "max_results": 50}, ws)

    assert search.search_calls[-1]["max_results"] == 5


def test_executor_unknown_tool(ws, executor):
    out = executor("nonesuch", {}, ws)
    assert "Unknown tool" in out


def test_resolve_rejects_absolute_outside(ws):
    with pytest.raises(OutsideWorkspaceError):
        _resolve(ws, "/etc/passwd")


# ---------------------------------------------------------------------------
# Memory tools (save_memory / search_memory)
# ---------------------------------------------------------------------------


@pytest.fixture
def memory_store(tmp_path):
    m = MemoryStore(tmp_path / "memories.bson")
    yield m
    m.close()


@pytest.fixture
def executor_with_memory(memory_store):
    runner = FakeRunner(result=RunResult(stdout="", stderr="", returncode=0))
    search = FakeSearchService(results=[])
    ex = build_tool_executor(runner, search, memory_store=memory_store)
    ex.context.owner = "alice"
    return ex


class TestMemoryTools:
    def test_save_memory_requires_configured_service(self, ws):
        runner = FakeRunner()
        ex = build_tool_executor(runner, FakeSearchService(results=[]))
        ex.context.owner = "alice"
        out = ex("save_memory", {"title": "T", "summary": "S"}, ws)
        assert "not configured" in out

    def test_save_memory_requires_owner(self, ws, memory_store):
        runner = FakeRunner()
        ex = build_tool_executor(runner, FakeSearchService(results=[]), memory_store=memory_store)
        out = ex("save_memory", {"title": "T", "summary": "S"}, ws)
        assert "logged-in user" in out

    def test_save_memory_requires_title_and_summary(self, ws, executor_with_memory):
        out = executor_with_memory("save_memory", {"title": "", "summary": ""}, ws)
        assert "required" in out

    def test_save_and_search_memory_round_trip(self, ws, executor_with_memory, memory_store):
        out = executor_with_memory(
            "save_memory",
            {"title": "Deadline", "summary": "Alice has a project deadline on Friday.", "tags": ["work"]},
            ws,
        )
        assert "Saved memory 'Deadline'" in out
        assert len(memory_store.list("alice")) == 1

        out = executor_with_memory("search_memory", {"query": "deadline"}, ws)
        assert "Deadline" in out

    def test_search_memory_no_results(self, ws, executor_with_memory):
        out = executor_with_memory("search_memory", {"query": "nothing here"}, ws)
        # The empty result explains itself with denominators, so the model
        # can report "nothing is saved about this" instead of reading an
        # empty list as a failed lookup and guessing.
        assert "surfaced 0" in out
        assert "nothing saved on this topic" in out

    def test_memory_scoped_per_owner(self, ws, memory_store):
        runner = FakeRunner()
        search = FakeSearchService(results=[])
        ex = build_tool_executor(runner, search, memory_store=memory_store)
        ex.context.owner = "alice"
        ex("save_memory", {"title": "Alice secret", "summary": "Only Alice should see this."}, ws)
        ex.context.owner = "bob"
        out = ex("search_memory", {"query": "secret"}, ws)
        # The empty result explains itself with denominators, so the model
        # can report "nothing is saved about this" instead of reading an
        # empty list as a failed lookup and guessing.
        assert "surfaced 0" in out
        assert "nothing saved on this topic" in out


# ---------------------------------------------------------------------------
# Admin-configurable limits (tool_output_max_chars / download_max_mb / etc.)
# ---------------------------------------------------------------------------


class TestToolOutputTruncation:
    def test_unlimited_by_default(self, ws, executor):
        (ws / "big.txt").write_text("x" * 10_000)
        out = executor("read_file", {"path": "big.txt"}, ws)
        assert len(out) == 10_000

    def test_snipped_when_limit_set(self, ws, executor):
        (ws / "big.txt").write_text("x" * 10_000)
        executor.context.tool_output_max_chars = 100
        out = executor("read_file", {"path": "big.txt"}, ws)
        assert len(out) < 10_000
        assert "snipped" in out

    def test_short_output_untouched_by_limit(self, ws, executor):
        (ws / "small.txt").write_text("hello")
        executor.context.tool_output_max_chars = 100
        out = executor("read_file", {"path": "small.txt"}, ws)
        assert out == "hello"


class TestDownloadLimits:
    def _fake_response(self, data: bytes):
        class _Resp:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *a):
                return False

            def read(self_inner, n=None):
                return data if n is None else data[:n]

        return _Resp()

    def test_download_within_limit_succeeds(self, ws, executor, monkeypatch):
        import urllib.request

        monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: self._fake_response(b"x" * 100))
        executor.context.download_max_mb = 1
        out = executor("download_file", {"url": "https://example.com/f.bin"}, ws)
        assert "Downloaded" in out
        assert (ws / "f.bin").exists()

    def test_download_over_limit_rejected(self, ws, executor, monkeypatch):
        import urllib.request

        monkeypatch.setattr(
            urllib.request, "urlopen",
            lambda req, timeout=None: self._fake_response(b"x" * (2 * 1024 * 1024)),
        )
        executor.context.download_max_mb = 1
        out = executor("download_file", {"url": "https://example.com/big.bin"}, ws)
        assert "exceeds" in out
        assert not (ws / "big.bin").exists()

    def test_download_uses_configured_user_agent(self, ws, executor, monkeypatch):
        import urllib.request

        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["ua"] = req.get_header("User-agent")
            captured["timeout"] = timeout
            return self._fake_response(b"data")

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        executor.context.user_agent = "CustomUA/2.0"
        executor.context.tool_network_timeout = 42
        executor("download_file", {"url": "https://example.com/f.bin"}, ws)
        assert captured["ua"] == "CustomUA/2.0"
        assert captured["timeout"] == 42
