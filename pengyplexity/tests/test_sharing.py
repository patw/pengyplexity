"""Tests for :mod:`pengyplexity.core.sharing`.

These verify:
* ``FakeSharingService`` returns canned URLs and records calls.
* ``ShareResult`` dataclass behaves correctly.
* ``share_message`` calls the right method based on kind.
* ``share_message`` stores the share in the store.
* ``HTTPSharingService`` is constructable (no network at construction).
* The ``SharingService`` ABC is satisfied by both real and fake.

All offline — no network, no real tclip/pengyshare calls.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pengyplexity.core.sharing import (
    FakeSharingService,
    HTTPSharingService,
    ShareResult,
    SharingService,
    share_message,
)
from pengyplexity.core.store import Store


# ---------------------------------------------------------------------------
# ShareResult
# ---------------------------------------------------------------------------


class TestShareResult:
    def test_fields(self):
        r = ShareResult(url="https://tclip.ca/x", kind="text", share_id="abc")
        assert r.url == "https://tclip.ca/x"
        assert r.kind == "text"
        assert r.share_id == "abc"

    def test_to_dict(self):
        r = ShareResult(url="https://x.com", kind="image")
        d = r.to_dict()
        assert d == {"url": "https://x.com", "kind": "image", "share_id": ""}


# ---------------------------------------------------------------------------
# SharingService interface
# ---------------------------------------------------------------------------


class TestSharingServiceInterface:
    def test_fake_is_sharing_service(self):
        assert isinstance(FakeSharingService(), SharingService)

    def test_http_is_sharing_service(self):
        assert isinstance(HTTPSharingService(), SharingService)


# ---------------------------------------------------------------------------
# FakeSharingService
# ---------------------------------------------------------------------------


class TestFakeSharing:
    def test_share_text_returns_canned_url(self):
        fs = FakeSharingService(text_url="https://tclip.ca/abc")
        result = fs.share_text("Hello world", title="Test")
        assert result.url == "https://tclip.ca/abc"
        assert result.kind == "text"
        assert result.share_id == "fake-id-1"

    def test_share_html_returns_canned_url(self):
        fs = FakeSharingService(text_url="https://tclip.ca/xyz")
        result = fs.share_html("<h1>Hi</h1>", title="Report")
        assert result.url == "https://tclip.ca/xyz"
        assert result.kind == "html"

    def test_share_image_returns_canned_url(self):
        fs = FakeSharingService(image_url="https://img.catbee.ca/pic.png")
        result = fs.share_image(Path("/fake/chart.png"))
        assert result.url == "https://img.catbee.ca/pic.png"
        assert result.kind == "image"

    def test_records_calls(self):
        fs = FakeSharingService()
        fs.share_text("hello")
        fs.share_html("<b>bold</b>")
        fs.share_image(Path("/x.png"))
        assert len(fs.calls) == 3
        assert fs.calls[0]["op"] == "share_text"
        assert fs.calls[1]["op"] == "share_html"
        assert fs.calls[2]["op"] == "share_image"

    def test_text_calls_property(self):
        fs = FakeSharingService()
        fs.share_text("a")
        fs.share_html("<i>b</i>")
        fs.share_image(Path("/c.png"))
        assert len(fs.text_calls) == 2
        assert len(fs.image_calls) == 1

    def test_content_recorded(self):
        fs = FakeSharingService()
        fs.share_text("my content", title="My Title")
        assert fs.calls[0]["content"] == "my content"
        assert fs.calls[0]["title"] == "My Title"

    def test_independent_instances(self):
        fs1 = FakeSharingService(text_url="https://a.com")
        fs2 = FakeSharingService(text_url="https://b.com")
        assert fs1.share_text("x").url == "https://a.com"
        assert fs2.share_text("x").url == "https://b.com"


# ---------------------------------------------------------------------------
# HTTPSharingService (construction only, no network)
# ---------------------------------------------------------------------------


class TestHTTPSharingConstruction:
    def test_constructable(self):
        s = HTTPSharingService(
            tclip_url="https://tclip.example.com/api",
            pengyshare_url="https://img.example.com/api",
            timeout=15,
        )
        assert s.tclip_url == "https://tclip.example.com/api"
        assert s.pengyshare_url == "https://img.example.com/api"
        assert s.timeout == 15

    def test_defaults(self):
        s = HTTPSharingService()
        assert "tclip" in s.tclip_url
        assert "img.catbee" in s.pengyshare_url or "pengyshare" in s.pengyshare_url


# ---------------------------------------------------------------------------
# share_message
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "store")
    yield s
    s.close()


class TestShareMessage:
    def test_share_text(self, store):
        fs = FakeSharingService(text_url="https://tclip.ca/xyz")
        result = share_message(
            sharing=fs,
            store=store,
            thread_id="t123",
            message_index=0,
            content="Hello world",
            kind="text",
        )
        assert result.url == "https://tclip.ca/xyz"
        assert result.kind == "text"
        # Stored in the store.
        shares = store.get_shares_for_thread("t123")
        assert len(shares) == 1
        assert shares[0]["url"] == "https://tclip.ca/xyz"
        assert shares[0]["kind"] == "text"

    def test_share_html(self, store):
        fs = FakeSharingService(text_url="https://tclip.ca/rep")
        result = share_message(
            sharing=fs,
            store=store,
            thread_id="t456",
            message_index=1,
            content="<h1>Report</h1>",
            kind="html",
        )
        assert result.kind == "html"
        shares = store.get_shares_for_thread("t456")
        assert len(shares) == 1
        assert shares[0]["message_index"] == 1

    def test_share_report_uses_html_method(self, store):
        fs = FakeSharingService(text_url="https://tclip.ca/rpt")
        result = share_message(
            sharing=fs,
            store=store,
            thread_id="t789",
            message_index=2,
            content="Report markdown",
            kind="report",
        )
        # "report" kind uses share_html internally.
        assert fs.calls[0]["op"] == "share_html"
        assert result.kind == "html"

    def test_share_image(self, store, tmp_path):
        # Create a fake image file.
        img = tmp_path / "chart.png"
        img.write_bytes(b"\x89PNG fake data")
        fs = FakeSharingService(image_url="https://img.catbee.ca/chart1.png")
        result = share_message(
            sharing=fs,
            store=store,
            thread_id="t1",
            message_index=0,
            content="",
            kind="image",
            image_path=img,
        )
        assert result.url == "https://img.catbee.ca/chart1.png"
        assert result.kind == "image"
        assert fs.calls[0]["op"] == "share_image"
        assert fs.calls[0]["path"] == str(img)

    def test_multiple_shares_on_same_thread(self, store):
        fs = FakeSharingService()
        share_message(fs, store, "t1", 0, "first", kind="text")
        share_message(fs, store, "t1", 1, "second", kind="text")
        shares = store.get_shares_for_thread("t1")
        assert len(shares) == 2

    def test_different_threads_separate_shares(self, store):
        fs = FakeSharingService()
        share_message(fs, store, "t1", 0, "a", kind="text")
        share_message(fs, store, "t2", 0, "b", kind="text")
        assert len(store.get_shares_for_thread("t1")) == 1
        assert len(store.get_shares_for_thread("t2")) == 1


# ---------------------------------------------------------------------------
# Integration: share → store → retrieve
# ---------------------------------------------------------------------------


class TestShareIntegration:
    def test_full_share_flow(self, store, tmp_path):
        """Simulate: share a message → URL stored → retrieved later."""
        fs = FakeSharingService(
            text_url="https://tclip.ca/abc123",
            image_url="https://img.catbee.ca/chart1.png",
        )

        # Share text.
        r1 = share_message(fs, store, "thread1", 0, "My answer text", kind="text")
        assert r1.url == "https://tclip.ca/abc123"

        # Share an image.
        img = tmp_path / "output.png"
        img.write_bytes(b"fake-png")
        r2 = share_message(
            fs, store, "thread1", 1, "", kind="image", image_path=img
        )
        assert r2.url == "https://img.catbee.ca/chart1.png"

        # Retrieve all shares for the thread.
        shares = store.get_shares_for_thread("thread1")
        assert len(shares) == 2
        urls = {s["url"] for s in shares}
        assert "https://tclip.ca/abc123" in urls
        assert "https://img.catbee.ca/chart1.png" in urls
        kinds = {s["kind"] for s in shares}
        assert "text" in kinds
        assert "image" in kinds


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
