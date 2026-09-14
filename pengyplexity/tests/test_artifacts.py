"""Tests for :mod:`pengyplexity.core.artifacts`.

These verify:
* ``run_chart_script`` writes the script to the workspace and calls the
  Runner (FakeRunner) — no live bwrap.
* ``markdown_to_html`` converts markdown to HTML correctly.
* ``html_to_pdf`` produces valid PDF bytes (skipped if reportlab missing).
* ``create_report`` produces HTML (and PDF if available) files in the
  workspace.
* ``ArtifactRecord`` and ``ArtifactService`` behave correctly.

All offline — no network, no live sandbox.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pengyplexity.core.artifacts import (
    ArtifactRecord,
    ArtifactService,
    create_report,
    html_to_pdf,
    markdown_to_html,
    run_chart_script,
)
from pengyplexity.core.toolexec import SCRATCH_DIR
from pengyplexity.sandbox.executors import FakeRunner, RunResult

# Check if reportlab is available (it may not be on the system python).
try:
    import reportlab  # noqa: F401
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


# ---------------------------------------------------------------------------
# ArtifactRecord
# ---------------------------------------------------------------------------


class TestArtifactRecord:
    def test_fields(self):
        r = ArtifactRecord(
            filename="chart.png",
            path=Path("/ws/chart.png"),
            kind="chart",
            mime="image/png",
            thread_id="t1",
            message_index=2,
            size_bytes=1234,
        )
        assert r.filename == "chart.png"
        assert r.kind == "chart"
        assert r.mime == "image/png"
        assert r.thread_id == "t1"
        assert r.message_index == 2
        assert r.size_bytes == 1234
        assert r.created is not None

    def test_to_dict(self):
        r = ArtifactRecord(
            filename="report.html",
            path=Path("/ws/report.html"),
            kind="report",
            mime="text/html",
        )
        d = r.to_dict()
        assert d["filename"] == "report.html"
        assert d["path"] == "/ws/report.html"
        assert d["kind"] == "report"
        assert d["mime"] == "text/html"


# ---------------------------------------------------------------------------
# run_chart_script
# ---------------------------------------------------------------------------


class TestRunChartScript:
    def test_writes_script_to_workspace(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        runner = FakeRunner(result=RunResult(stdout="", stderr="", returncode=0))
        script = "import matplotlib.pyplot as plt\nplt.savefig('chart.png')"

        record = run_chart_script(
            script=script,
            workspace=ws,
            runner=runner,
            output_filename="chart.png",
            thread_id="t1",
        )

        # The script ran from the workspace's private scratch dir and was
        # cleaned up: writing `chart_script.py` into the workspace root
        # overwrote any file of that name and then showed up in the user's
        # own artifact listing.
        assert not (ws / "chart_script.py").exists()
        assert list((ws / SCRATCH_DIR).glob("chart-*.py")) == []

        # The runner was called with the right command.
        assert len(runner.calls) == 1
        assert runner.calls[0].script.startswith(f"python3 {SCRATCH_DIR}/chart-")
        assert runner.calls[0].script.endswith(".py")

        # The artifact record is correct.
        assert record.filename == "chart.png"
        assert record.kind == "chart"
        assert record.mime == "image/png"
        assert record.thread_id == "t1"
        assert record.path == ws / "chart.png"

    def test_runner_error_still_returns_record(self, tmp_path):
        """Even if the script fails, a record is returned (size=0)."""
        ws = tmp_path / "ws"
        ws.mkdir()
        runner = FakeRunner(
            result=RunResult(stdout="", stderr="error", returncode=1)
        )
        record = run_chart_script(
            script="bad code",
            workspace=ws,
            runner=runner,
        )
        assert record.size_bytes == 0  # file doesn't exist

    def test_existing_file_size_reported(self, tmp_path):
        """If the output file exists, its size is reported."""
        ws = tmp_path / "ws"
        ws.mkdir()
        # Simulate the chart file being created.
        (ws / "chart.png").write_bytes(b"fake-png-data")
        runner = FakeRunner()
        record = run_chart_script(
            script="ok",
            workspace=ws,
            runner=runner,
            output_filename="chart.png",
        )
        assert record.size_bytes == len(b"fake-png-data")


# ---------------------------------------------------------------------------
# markdown_to_html
# ---------------------------------------------------------------------------


class TestMarkdownToHTML:
    def test_basic_heading(self):
        html = markdown_to_html("# Title")
        assert "<h1" in html
        assert "Title" in html

    def test_paragraph(self):
        html = markdown_to_html("Hello world")
        assert "<p>" in html
        assert "Hello world" in html

    def test_bold_and_italic(self):
        html = markdown_to_html("**bold** and *italic*")
        assert "<strong>bold</strong>" in html
        assert "<em>italic</em>" in html

    def test_list(self):
        md = "- item 1\n- item 2"
        html = markdown_to_html(md)
        assert "<li>" in html
        assert "item 1" in html

    def test_table(self):
        md = "| A | B |\n|---|---|\n| 1 | 2 |"
        html = markdown_to_html(md)
        assert "<table>" in html or "<tr>" in html

    def test_code_block(self):
        md = "```\nprint('hi')\n```"
        html = markdown_to_html(md)
        assert "<code>" in html or "<pre>" in html

    def test_link(self):
        html = markdown_to_html("[Google](https://google.com)")
        assert '<a href="https://google.com"' in html


# ---------------------------------------------------------------------------
# html_to_pdf (skipped if reportlab unavailable)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_REPORTLAB, reason="reportlab not installed")
class TestHtmlToPdf:
    def test_produces_pdf_bytes(self):
        pdf = html_to_pdf("# Hello\n\nThis is a test.")
        assert isinstance(pdf, bytes)
        assert len(pdf) > 100
        # PDF files start with %PDF
        assert pdf.startswith(b"%PDF")

    def test_multiline_content(self):
        pdf = html_to_pdf("# Title\n\nPara 1\n\n## Sub\n\nPara 2")
        assert pdf.startswith(b"%PDF")
        assert len(pdf) > 200


# ---------------------------------------------------------------------------
# create_report
# ---------------------------------------------------------------------------


class TestCreateReport:
    def test_creates_html_file(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        artifacts = create_report(
            markdown_text="# My Report\n\nSome content.",
            workspace=ws,
            title="My Report",
            thread_id="t1",
        )

        # HTML artifact always present.
        assert "html" in artifacts
        html_art = artifacts["html"]
        assert html_art.kind == "report"
        assert html_art.mime == "text/html"
        assert html_art.path.exists()
        assert html_art.size_bytes > 0
        # File content is valid HTML.
        content = html_art.path.read_text()
        assert "<h1" in content
        assert "My Report" in content

    def test_creates_pdf_if_available(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        artifacts = create_report(
            markdown_text="# Test\n\nBody.",
            workspace=ws,
            title="Test Report",
        )
        if HAS_REPORTLAB:
            assert "pdf" in artifacts
            pdf_art = artifacts["pdf"]
            assert pdf_art.mime == "application/pdf"
            assert pdf_art.path.exists()
            assert pdf_art.path.read_bytes().startswith(b"%PDF")
        else:
            # Without reportlab, only HTML is produced.
            assert "pdf" not in artifacts
            assert "html" in artifacts

    def test_thread_id_and_message_index(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        artifacts = create_report(
            markdown_text="x",
            workspace=ws,
            thread_id="t42",
            message_index=3,
        )
        assert artifacts["html"].thread_id == "t42"
        assert artifacts["html"].message_index == 3


# ---------------------------------------------------------------------------
# ArtifactService
# ---------------------------------------------------------------------------


class FakeStore:
    """Minimal store fake with create_artifact."""

    def __init__(self):
        self.artifacts = []

    def create_artifact(self, record: dict):
        self.artifacts.append(record)
        return record


class TestArtifactService:
    def test_create_chart(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        store = FakeStore()
        runner = FakeRunner()
        service = ArtifactService(store=store, runner=runner)

        record = service.create_chart(
            workspace=ws,
            script="plt.savefig('x.png')",
            thread_id="t1",
            message_index=0,
        )
        assert record.kind == "chart"
        # Stored in the fake store.
        assert len(store.artifacts) == 1
        assert store.artifacts[0]["kind"] == "chart"

    def test_create_report(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        store = FakeStore()
        runner = FakeRunner()
        service = ArtifactService(store=store, runner=runner)

        artifacts = service.create_report(
            workspace=ws,
            markdown_text="# Hello\n\nWorld.",
            thread_id="t1",
            title="Test",
        )
        assert "html" in artifacts
        # At least one artifact stored.
        assert len(store.artifacts) >= 1


# ---------------------------------------------------------------------------
# Integration: chart → runner → artifact record → store
# ---------------------------------------------------------------------------


class TestChartPipeline:
    def test_full_chart_flow(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        runner = FakeRunner(result=RunResult(stdout="saved", stderr="", returncode=0))
        store = FakeStore()
        service = ArtifactService(store=store, runner=runner)

        script = (
            "import matplotlib.pyplot as plt\n"
            "fig, ax = plt.subplots()\n"
            "ax.plot([1,2,3], [4,5,6])\n"
            "plt.savefig('chart.png')\n"
        )

        record = service.create_chart(
            workspace=ws,
            script=script,
            thread_id="thread-1",
            message_index=1,
            output_filename="my_chart.png",
        )

        # Runner was called.
        assert len(runner.calls) == 1
        assert "python3" in runner.calls[0].script

        # Record is correct.
        assert record.filename == "my_chart.png"
        assert record.kind == "chart"
        assert record.thread_id == "thread-1"
        assert record.message_index == 1

        # Stored.
        assert len(store.artifacts) == 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
