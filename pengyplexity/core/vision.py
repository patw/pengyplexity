"""Handing images to a vision model, ported from Pengy.

An OpenAI-compatible ``role: "tool"`` message only accepts **string** content,
so a tool that loads a picture cannot return the picture. Pengy solves this by
parking the encoded image on the tool context and attaching it, after the
turn's tool results are in place, as a follow-up ``role: "user"`` message whose
content is a list of ``text`` / ``image_url`` parts (see
``pengy/core/llm_client.py``). This module is that mechanism, scoped to
Pengyplexity's confined tools.

Without it the model is blind: ``read_image`` could only report a filename and
a byte count, so a vision model could not read a screenshot the user
downloaded, and — more importantly — could not look at the chart or image it
had just produced to check that it came out right.

:func:`encode_for_model` reuses Pengy's ``pengy.core.image_utils.preprocess``
when Pengy is importable and otherwise falls back to an equivalent local
implementation, so the app keeps working as a standalone deployment.
"""

from __future__ import annotations

import base64
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Conservative limits that work across the major vision APIs, matching Pengy's
# defaults (pengy/core/image_utils.py).
MAX_DIMENSION = 4096
MAX_MB = 4.5
JPEG_QUALITY = 85

# Extensions the image tools will attempt to decode.
IMAGE_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".tif"}
)


@dataclass
class PendingImage:
    """One image waiting to be attached to the conversation."""

    label: str
    mime: str
    b64: str

    def to_parts(self) -> List[Dict[str, Any]]:
        """The two content parts this image contributes to a user message."""
        return [
            {"type": "text", "text": self.label},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{self.mime};base64,{self.b64}"},
            },
        ]


class PendingImages:
    """A thread-safe queue of images produced during one agent turn.

    The tool executor adds to it; the agent loop drains it once every tool
    result for the round-trip has been appended, so each ``tool_calls``
    assistant message keeps its matching ``tool`` messages immediately behind
    it (an ordering some backends enforce).
    """

    def __init__(self) -> None:
        self._items: List[PendingImage] = []
        self._lock = threading.Lock()

    def add(self, label: str, mime: str, b64: str) -> None:
        with self._lock:
            self._items.append(PendingImage(label=label, mime=mime, b64=b64))

    def take(self) -> List[PendingImage]:
        """Return everything queued and clear the queue."""
        with self._lock:
            items = self._items
            self._items = []
        return items

    def clear(self) -> None:
        self.take()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


def build_image_message(images: List[PendingImage]) -> Optional[Dict[str, Any]]:
    """Build the follow-up user message carrying *images*, or None if empty."""
    if not images:
        return None
    parts: List[Dict[str, Any]] = []
    for image in images:
        parts.extend(image.to_parts())
    return {"role": "user", "content": parts}


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def encode_for_model(
    path: Path,
    max_dimension: int = MAX_DIMENSION,
    max_mb: float = MAX_MB,
    quality: int = JPEG_QUALITY,
) -> "tuple[bytes, str, tuple[int, int]]":
    """Return ``(bytes, mime, (width, height))`` ready for base64 encoding.

    Prefers Pengy's ``image_utils.preprocess`` so both apps resize and
    re-encode identically; falls back to the local equivalent below when Pengy
    is not installed alongside.
    """
    from PIL import Image

    with Image.open(path) as probe:
        size = probe.size

    try:
        from pengy.core.image_utils import preprocess  # type: ignore
    except ImportError:
        data, mime = _preprocess(path, max_dimension, max_mb, quality)
    else:
        data, mime = preprocess(
            path, max_dimension=max_dimension, max_mb=max_mb, quality=quality
        )
    return data, mime, size


def _preprocess(
    path: Path, max_dimension: int, max_mb: float, quality: int
) -> "tuple[bytes, str]":
    """Standalone equivalent of Pengy's ``image_utils.preprocess``.

    Resize past *max_dimension*, flatten to JPEG when the source is a lossless
    format with no useful alpha, then step the quality down until the encoded
    image fits *max_mb*.
    """
    import io

    from PIL import Image

    max_bytes = int(max_mb * 1024 * 1024)
    img = Image.open(path)
    source_format = (img.format or "").upper()

    if max(img.size) > max_dimension:
        img.thumbnail((max_dimension, max_dimension), Image.LANCZOS)

    if source_format in {"PNG", "GIF", "BMP", "TIFF"}:
        if img.mode == "RGBA":
            flattened = Image.new("RGB", img.size, (255, 255, 255))
            flattened.paste(img, mask=img.split()[3])
            img = flattened
        elif img.mode != "RGB":
            img = img.convert("RGB")
    elif img.mode not in {"RGB", "L"}:
        img = img.convert("RGB")

    for attempt_quality in (quality, 75, 60, 45, 30):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=attempt_quality, optimize=True)
        data = buf.getvalue()
        if len(data) <= max_bytes:
            return data, "image/jpeg"

    # Still too large at the lowest quality: halve the dimensions and retry
    # once. A 4096px cap at quality 30 that is still over 4.5 MB is unusual.
    img.thumbnail((max_dimension // 2, max_dimension // 2), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=30, optimize=True)
    return buf.getvalue(), "image/jpeg"


def queue_image(
    pending: Optional[PendingImages],
    path: Path,
    label_path: str,
    note: str = "",
) -> str:
    """Encode *path* and queue it on *pending*; return the tool-result text.

    *label_path* is the workspace-relative name shown to the model (never the
    host path). The returned string is what the ``role: "tool"`` message
    carries — the picture itself rides along in the follow-up user message the
    agent builds from the queue.
    """
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        return (
            f"{label_path} is not a recognized image file. Supported: "
            f"{', '.join(sorted(IMAGE_SUFFIXES))}. Use read_file for text."
        )
    original_size = path.stat().st_size
    try:
        data, mime, (width, height) = encode_for_model(path)
    except Exception as exc:  # noqa: BLE001
        return f"Could not decode {label_path} as an image: {exc}"

    if pending is None:
        # No vision channel wired up (e.g. a bare unit test) — still report
        # what the file is rather than pretending the image was attached.
        return f"{label_path} — {width}×{height}, {_fmt_size(original_size)}."

    label = note or f"Image loaded by read_image: {label_path}"
    pending.add(label, mime, base64.b64encode(data).decode())

    summary = f"Loaded {label_path} — {width}×{height}, {_fmt_size(original_size)}"
    if len(data) != original_size:
        summary += f" → {mime}, {_fmt_size(len(data))} after preprocessing"
    return summary + ". The image is attached below; look at it directly."


def _fmt_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB"):
        if num_bytes < 1024 or unit == "MB":
            return f"{num_bytes:.0f} {unit}" if unit == "B" else f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} MB"
