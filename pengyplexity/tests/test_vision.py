"""Tests for the vision path: images actually reaching the model.

A ``role: "tool"`` message carries string content only, so a tool that loads a
picture cannot return it. Images are parked on the tool context and attached
afterwards as a follow-up ``role: "user"`` message with ``image_url`` parts —
Pengy's mechanism (``pengy/core/llm_client.py``), ported in ``core/vision.py``.

Before this existed, ``read_image`` returned the string "(Text-only client;
see path.)": the model was blind even though the app runs on a vision model,
and could not look at the chart it had just produced to check it came out
right.

Everything here is offline — a tiny PNG written by PIL, no model, no network.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from pengyplexity.core.agent import Agent
from pengyplexity.core.modelclient import ChatResponse, FakeModelClient, ToolCall
from pengyplexity.core.search import FakeSearchService
from pengyplexity.core.toolexec import build_tool_executor
from pengyplexity.core.vision import (
    PendingImages,
    build_image_message,
    queue_image,
)
from pengyplexity.sandbox.executors import FakeRunner

PIL = pytest.importorskip("PIL", reason="Pillow is a runtime dependency")


def _write_png(path: Path, size=(40, 30)) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (10, 120, 200)).save(path, format="PNG")
    return path


# ---------------------------------------------------------------------------
# PendingImages
# ---------------------------------------------------------------------------


class TestPendingImages:
    def test_take_drains_the_queue(self):
        pending = PendingImages()
        pending.add("one", "image/png", "AAA")
        pending.add("two", "image/png", "BBB")

        assert len(pending) == 2
        taken = pending.take()
        assert [p.label for p in taken] == ["one", "two"]
        assert len(pending) == 0
        assert pending.take() == []

    def test_build_image_message_shape(self):
        pending = PendingImages()
        pending.add("a chart", "image/png", "QUJD")

        message = build_image_message(pending.take())

        assert message["role"] == "user"
        text, image = message["content"]
        assert text == {"type": "text", "text": "a chart"}
        assert image["type"] == "image_url"
        assert image["image_url"]["url"] == "data:image/png;base64,QUJD"

    def test_build_image_message_is_none_when_empty(self):
        assert build_image_message([]) is None


# ---------------------------------------------------------------------------
# queue_image
# ---------------------------------------------------------------------------


class TestQueueImage:
    def test_encodes_and_queues(self, tmp_path):
        png = _write_png(tmp_path / "chart.png")
        pending = PendingImages()

        summary = queue_image(pending, png, "chart.png")

        assert "40×30" in summary
        assert "attached below" in summary
        assert len(pending) == 1
        # Whatever the re-encoding chose, it must decode to real image bytes.
        image = pending.take()[0]
        assert image.mime.startswith("image/")
        assert len(base64.b64decode(image.b64)) > 0

    def test_reports_the_workspace_path_not_the_host_path(self, tmp_path):
        png = _write_png(tmp_path / "sub" / "shot.png")
        pending = PendingImages()

        summary = queue_image(pending, png, "sub/shot.png")

        assert "sub/shot.png" in summary
        assert str(tmp_path) not in summary

    def test_rejects_a_non_image_extension(self, tmp_path):
        notes = tmp_path / "notes.txt"
        notes.write_text("hello")
        pending = PendingImages()

        result = queue_image(pending, notes, "notes.txt")

        assert "not a recognized image" in result
        assert len(pending) == 0

    def test_undecodable_file_reports_rather_than_raising(self, tmp_path):
        broken = tmp_path / "broken.png"
        broken.write_bytes(b"not actually a png")
        pending = PendingImages()

        result = queue_image(pending, broken, "broken.png")

        assert "Could not decode" in result
        assert len(pending) == 0


# ---------------------------------------------------------------------------
# The tool executor wires images onto the context
# ---------------------------------------------------------------------------


class TestToolExecutorAttachesImages:
    def _executor(self):
        return build_tool_executor(FakeRunner(), FakeSearchService())

    def test_read_image_queues_the_picture(self, tmp_path):
        _write_png(tmp_path / "photo.png")
        execute = self._executor()

        result = execute("read_image", {"path": "photo.png"}, tmp_path)

        assert "attached below" in result
        assert len(execute.context.pending_images) == 1

    def test_read_image_outside_the_workspace_is_blocked(self, tmp_path):
        execute = self._executor()

        result = execute("read_image", {"path": "../escape.png"}, tmp_path)

        assert result.startswith("Blocked:")
        assert len(execute.context.pending_images) == 0

    def test_missing_image_reports_the_relative_path(self, tmp_path):
        execute = self._executor()

        result = execute("read_image", {"path": "nope.png"}, tmp_path)

        assert result == "Image not found: nope.png"


# ---------------------------------------------------------------------------
# The agent attaches queued images to the conversation
# ---------------------------------------------------------------------------


class TestAgentAttachesImages:
    def test_queued_image_becomes_a_follow_up_user_message(self, tmp_path):
        _write_png(tmp_path / "photo.png")
        execute = build_tool_executor(FakeRunner(), FakeSearchService())
        model = FakeModelClient(responses=[
            ChatResponse(tool_calls=[
                ToolCall(id="c1", name="read_image", arguments={"path": "photo.png"})
            ]),
            ChatResponse(content="I can see a blue rectangle."),
        ])
        agent = Agent(model=model, tool_executor=execute, workspace=tmp_path)

        result = agent.run("what is in photo.png?")

        assert result.answer == "I can see a blue rectangle."
        # Second request: system, user, assistant(tool_calls), tool, user(image).
        sent = model.calls[1]["messages"]
        assert sent[-2]["role"] == "tool"
        image_message = sent[-1]
        assert image_message["role"] == "user"
        kinds = [part["type"] for part in image_message["content"]]
        assert kinds == ["text", "image_url"]
        assert image_message["content"][1]["image_url"]["url"].startswith("data:image/")

    def test_no_image_means_no_extra_message(self, tmp_path):
        execute = build_tool_executor(FakeRunner(), FakeSearchService())
        model = FakeModelClient(responses=[
            ChatResponse(tool_calls=[
                ToolCall(id="c1", name="web_search", arguments={"query": "x"})
            ]),
            ChatResponse(content="done"),
        ])
        agent = Agent(model=model, tool_executor=execute, workspace=tmp_path)

        agent.run("search please")

        assert model.calls[1]["messages"][-1]["role"] == "tool"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
