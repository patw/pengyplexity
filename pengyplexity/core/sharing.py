"""Sharing: tclip (text/HTML) + pengyshare (image) upload helpers.

This module provides:

* :class:`ShareResult` — the result of a share operation (URL, kind).
* :class:`SharingService` — the abstract interface for uploading content.
* :class:`HTTPSharingService` — real implementation that POSTs to
  tclip.catbee.ca (text/HTML) and img.catbee.ca (images).
* :class:`FakeSharingService` — offline test double. Records calls, returns
  canned URLs.

Design notes:
* Both uploaders are **injectable** — tests use :class:`FakeSharingService`
  so the suite runs with no network.
* tclip accepts plain text or HTML and returns a short URL.
* pengyshare accepts image bytes + MIME type and returns a public image URL.
* Share results are stored on the message (via the store) so they can be
  copied/shared later.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class ShareResult:
    """Result of a share upload.

    Attributes
    ----------
    url:
        The public URL to the shared content.
    kind:
        One of: ``"text"``, ``"html"``, ``"image"``, ``"report"``.
    share_id:
        Optional identifier returned by the service (for tclip).
    """

    url: str
    kind: str
    share_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "kind": self.kind,
            "share_id": self.share_id,
        }


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class SharingService(ABC):
    """Abstract sharing interface.

    Implementations:
    * :class:`HTTPSharingService` — real HTTP uploads.
    * :class:`FakeSharingService` — offline test double.
    """

    @abstractmethod
    def share_text(self, content: str, title: str = "") -> ShareResult:
        """Share plain text content (via tclip). Returns a short URL."""
        ...

    @abstractmethod
    def share_html(self, html: str, title: str = "") -> ShareResult:
        """Share HTML content (via tclip). Returns a short URL."""
        ...

    @abstractmethod
    def share_image(self, image_path: Path, mime: str = "image/png") -> ShareResult:
        """Upload an image file (via pengyshare). Returns a public URL."""
        ...


# ---------------------------------------------------------------------------
# Real HTTP implementation
# ---------------------------------------------------------------------------


class HTTPSharingService(SharingService):
    """Real sharing service using tclip.catbee.ca + img.catbee.ca.

    Parameters
    ----------
    tclip_url:
        tclip endpoint (default: ``https://tclip.catbee.ca/api/clip``).
    pengyshare_url:
        pengyshare endpoint (default: ``https://img.catbee.ca/api/upload``).
    timeout:
        HTTP timeout in seconds.
    """

    def __init__(
        self,
        tclip_url: str = "https://tclip.catbee.ca/api/clip",
        pengyshare_url: str = "https://img.catbee.ca/api/upload",
        timeout: int = 30,
    ) -> None:
        self.tclip_url = tclip_url
        self.pengyshare_url = pengyshare_url
        self.timeout = timeout

    def share_text(self, content: str, title: str = "") -> ShareResult:
        """POST text to tclip."""
        import json
        import urllib.request

        payload = json.dumps({"content": content, "title": title}).encode()
        req = urllib.request.Request(
            self.tclip_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode())
        return ShareResult(
            url=data.get("url", ""),
            kind="text",
            share_id=data.get("id", ""),
        )

    def share_html(self, html: str, title: str = "") -> ShareResult:
        """POST HTML to tclip."""
        import json
        import urllib.request

        payload = json.dumps(
            {"content": html, "title": title, "format": "html"}
        ).encode()
        req = urllib.request.Request(
            self.tclip_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode())
        return ShareResult(
            url=data.get("url", ""),
            kind="html",
            share_id=data.get("id", ""),
        )

    def share_image(self, image_path: Path, mime: str = "image/png") -> ShareResult:
        """Upload an image to pengyshare."""
        import urllib.request
        import mimetypes
        import uuid

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Build a multipart form POST.
        boundary = uuid.uuid4().hex
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{image_path.name}"\r\n'
            f"Content-Type: {mime}\r\n\r\n"
        ).encode() + image_path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()

        req = urllib.request.Request(
            self.pengyshare_url,
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            import json
            data = json.loads(resp.read().decode())
        return ShareResult(
            url=data.get("url", ""),
            kind="image",
            share_id=data.get("id", ""),
        )


# ---------------------------------------------------------------------------
# Fake sharing for offline tests
# ---------------------------------------------------------------------------


@dataclass
class FakeSharingService(SharingService):
    """Offline test double for :class:`SharingService`.

    Returns canned URLs and records all calls.

    Parameters
    ----------
    text_url:
        URL to return for text/html shares (default: a fake tclip URL).
    image_url:
        URL to return for image shares (default: a fake pengyshare URL).
    """

    text_url: str = "https://tclip.catbee.ca/fake123"
    image_url: str = "https://img.catbee.ca/fake456.png"
    calls: List[Dict[str, Any]] = field(default_factory=list)

    def share_text(self, content: str, title: str = "") -> ShareResult:
        self.calls.append({
            "op": "share_text",
            "content": content,
            "title": title,
        })
        return ShareResult(url=self.text_url, kind="text", share_id="fake-id-1")

    def share_html(self, html: str, title: str = "") -> ShareResult:
        self.calls.append({
            "op": "share_html",
            "html": html,
            "title": title,
        })
        return ShareResult(url=self.text_url, kind="html", share_id="fake-id-2")

    def share_image(self, image_path: Path, mime: str = "image/png") -> ShareResult:
        self.calls.append({
            "op": "share_image",
            "path": str(image_path),
            "mime": mime,
        })
        return ShareResult(url=self.image_url, kind="image", share_id="fake-img-1")

    @property
    def text_calls(self) -> List[Dict[str, Any]]:
        return [c for c in self.calls if c["op"] in ("share_text", "share_html")]

    @property
    def image_calls(self) -> List[Dict[str, Any]]:
        return [c for c in self.calls if c["op"] == "share_image"]


# ---------------------------------------------------------------------------
# High-level: share a message
# ---------------------------------------------------------------------------


def share_message(
    sharing: SharingService,
    store,
    thread_id: str,
    message_index: int,
    content: str,
    kind: str = "text",
    image_path: Optional[Path] = None,
) -> ShareResult:
    """Share a message and store the share URL on the store.

    Parameters
    ----------
    sharing:
        A :class:`SharingService` instance.
    store:
        A :class:`Store` with ``create_share`` method.
    thread_id:
        The thread containing the message.
    message_index:
        Index of the message being shared.
    content:
        The text/HTML content to share.
    kind:
        ``"text"``, ``"html"``, ``"report"``, or ``"image"``.
    image_path:
        If kind is ``"image"``, the path to the image file.

    Returns
    -------
    ShareResult
    """
    if kind == "image" and image_path is not None:
        result = sharing.share_image(image_path)
    elif kind in ("html", "report"):
        result = sharing.share_html(content, title=f"Share: {thread_id[:8]}")
    else:
        result = sharing.share_text(content, title=f"Share: {thread_id[:8]}")

    # Store the share in the store.
    store.create_share(
        thread_id=thread_id,
        message_index=message_index,
        kind=result.kind,
        url=result.url,
    )

    return result
