"""Image generation / editing via the ``image_gen`` / ``image_edit`` skills.

This module is the app's integration point for the *image* capability listed in
the spec. Following the offline-test rule, the actual image-producing backend is
**injectable**:

* :class:`ImageBackend` — the abstract interface (``generate`` + ``edit``).
* :class:`SkillImageBackend` — the real backend. It shells out to Pengy's
  ``image_gen`` / ``image_edit`` skill scripts (``gen_image.py`` /
  ``edit_image.py``), always with ``--no-upload`` and with ``-o`` pointed into
  the **per-thread workspace** so the produced PNG lands inside the sandbox and
  never touches ``~/Pictures`` or the host network for a share link. The
  shareable URL, when wanted, is added later via :mod:`pengyplexity.core.sharing`
  (pengyshare), not here.
* :class:`FakeImageBackend` — an offline test double that writes a tiny valid
  PNG into the workspace and returns an :class:`ImageResult`, recording every
  call so the suite asserts without any Gemini call or network.

The command construction in :class:`SkillImageBackend` is split out into pure
``build_*`` methods (analogous to :meth:`BwrapRunner.build_argv`) so the
argv — including the confinement-critical ``-o <workspace>/...`` and
``--no-upload`` flags — can be asserted in the test suite with **no skill
script and no network present**.

High-level helpers (:func:`generate_image`, :func:`edit_image`, and
:class:`ImageService`) turn a backend result into an
:class:`~pengyplexity.core.artifacts.ArtifactRecord` of ``kind="image"`` so
the image is stored with its thread and can be rendered inline in the UI.
"""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from pengyplexity.core.artifacts import ArtifactRecord

# ---------------------------------------------------------------------------
# A tiny, dependency-free, valid 1x1 transparent PNG. The fake backend writes
# this so tests get a real on-disk file with a real size and correct magic
# bytes, with no PIL / network requirement.
# ---------------------------------------------------------------------------
_MINI_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9Q"
    "DwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

# Default locations of the real skill scripts on a host that has Pengy. These
# are only *referenced* (to build a command); they are never read at import
# time and never required by the offline suite.
DEFAULT_GEN_SCRIPT = "~/Personal/skills/image_gen/gen_image.py"
DEFAULT_EDIT_SCRIPT = "~/Personal/skills/image_edit/edit_image.py"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class ImageResult:
    """The outcome of a generate/edit operation.

    Attributes
    ----------
    path:
        Workspace path to the produced PNG (always inside the sandbox).
    mime:
        MIME type (default ``"image/png"``).
    prompt:
        The text prompt (or edit instruction).
    kind:
        ``"generate"`` or ``"edit"``.
    alt:
        Descriptive alt text for inline rendering.
    input_images:
        For edits: the input image paths. Empty for generation.
    created:
        Timestamp (UTC).
    size_bytes:
        File size on disk (0 if unknown / not written).
    """

    path: Path
    mime: str = "image/png"
    prompt: str = ""
    kind: str = "generate"
    alt: str = ""
    input_images: List[Path] = field(default_factory=list)
    created: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    size_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "mime": self.mime,
            "prompt": self.prompt,
            "kind": self.kind,
            "alt": self.alt,
            "input_images": [str(p) for p in self.input_images],
            "created": self.created,
            "size_bytes": self.size_bytes,
        }


class ImageBackendError(RuntimeError):
    """Raised when an image backend fails, carrying the reason it gave."""


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------


class ImageBackend(ABC):
    """Abstract image backend.

    Every method returns an :class:`ImageResult` whose ``path`` is *inside*
    ``workspace`` — backends must never write outside the per-thread sandbox.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        workspace: Path,
        *,
        filename: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        resolution: Optional[str] = None,
        alt: str = "",
    ) -> ImageResult:
        """Generate an image from ``prompt`` into ``workspace``."""
        ...

    @abstractmethod
    def edit(
        self,
        prompt: str,
        input_images: Sequence[Union[str, Path]],
        workspace: Path,
        *,
        filename: Optional[str] = None,
        alt: str = "",
    ) -> ImageResult:
        """Edit ``input_images`` per ``prompt`` into ``workspace``."""
        ...


# ---------------------------------------------------------------------------
# Real backend: shell out to the image skills, confined to the workspace
# ---------------------------------------------------------------------------


class SkillImageBackend(ImageBackend):
    """Runs Pengy's ``image_gen`` / ``image_edit`` skill scripts.

    The commands are built so the output is written into the per-thread
    workspace (``-o <workspace>/<filename>``) and the automatic PengyShare
    upload is disabled (``--no-upload``), which keeps every produced file inside
    the sandbox. Sharing (a public URL) is a separate, explicit step done by
    :mod:`pengyplexity.core.sharing`.

    Parameters
    ----------
    gen_script / edit_script:
        Paths to the skill scripts. Defaults to the host locations; inject
        alternatives in tests.
    runner:
        Optional callable ``(cmd: Sequence[str]) -> None`` used to execute the
        command (defaults to :func:`subprocess.run`). Injectable so a real
        backend can be exercised without a live Gemini call.
    uv:
        The ``uv`` executable to use (the skills declare inline PEP 723 deps).
    """

    def __init__(
        self,
        gen_script: Optional[Union[str, Path]] = None,
        edit_script: Optional[Union[str, Path]] = None,
        runner: Optional[Any] = None,
        uv: str = "uv",
        timeout: float = 300.0,
    ) -> None:
        self.gen_script = Path(gen_script or DEFAULT_GEN_SCRIPT).expanduser()
        self.edit_script = Path(edit_script or DEFAULT_EDIT_SCRIPT).expanduser()
        self._runner = runner
        self.uv = uv
        # Wall-clock cap on one skill invocation. Without it a wedged image
        # call holds the Flask worker (and the user's SSE stream) forever.
        self.timeout = timeout

    # -- pure command builders (asserted offline) --------------------------
    def build_generate_command(
        self,
        prompt: str,
        workspace: Path,
        *,
        filename: str,
        aspect_ratio: Optional[str] = None,
        resolution: Optional[str] = None,
    ) -> List[str]:
        """Build the ``uv run gen_image.py ...`` argv.

        The critical safety properties, both asserted in tests:
        * the output is written with ``-o <workspace>/<filename>`` (inside the
          sandbox — never ``~/Pictures``);
        * ``--no-upload`` is always present (no host PengyShare upload here).
        """
        workspace = Path(workspace)
        out = workspace / filename
        cmd: List[str] = [self.uv, "run", str(self.gen_script), prompt, "-o", str(out)]
        if aspect_ratio:
            cmd += ["--aspect-ratio", aspect_ratio]
        if resolution:
            cmd += ["--resolution", resolution]
        cmd += ["--no-upload"]
        return cmd

    def build_edit_command(
        self,
        prompt: str,
        input_images: Sequence[Union[str, Path]],
        workspace: Path,
        *,
        filename: str,
    ) -> List[str]:
        """Build the ``uv run edit_image.py ...`` argv.

        Same safety contract: ``-o <workspace>/<filename>`` + ``--no-upload``,
        and every input image is passed via a repeated ``-i``.
        """
        workspace = Path(workspace)
        out = workspace / filename
        cmd: List[str] = [self.uv, "run", str(self.edit_script), prompt]
        for img in input_images:
            cmd += ["-i", str(Path(img).expanduser())]
        cmd += ["-o", str(out), "--no-upload"]
        return cmd

    # -- execution ---------------------------------------------------------
    def _execute(self, cmd: List[str]) -> None:
        if self._runner is not None:
            self._runner(cmd)
            return
        import subprocess

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                # Never let the skill inherit the web process's stdin: with no
                # prompt argument these scripts read it, which would hang the
                # worker for good.
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            raise ImageBackendError(
                f"the image skill did not finish within {self.timeout}s."
            ) from None
        except OSError as exc:
            raise ImageBackendError(f"could not run the image skill: {exc}") from None
        if proc.returncode != 0:
            # `check=True` would raise CalledProcessError, whose message is
            # just "returned non-zero exit status 1" — the actual reason
            # (missing GOOGLE_API_KEY, an invalid aspect ratio, a refused
            # prompt) is on stderr and has to be carried through, or the model
            # is told only that it failed and retries the same call.
            detail = (proc.stderr or proc.stdout or "").strip().splitlines()
            raise ImageBackendError(
                " ".join(detail[-3:]) if detail else f"exit status {proc.returncode}"
            )

    @staticmethod
    def _size(path: Path) -> int:
        return path.stat().st_size if path.exists() else 0

    def generate(
        self,
        prompt: str,
        workspace: Path,
        *,
        filename: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        resolution: Optional[str] = None,
        alt: str = "",
    ) -> ImageResult:
        filename = filename or f"image_gen_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}.png"
        cmd = self.build_generate_command(
            prompt, workspace, filename=filename,
            aspect_ratio=aspect_ratio, resolution=resolution,
        )
        self._execute(cmd)
        out = Path(workspace) / filename
        return ImageResult(
            path=out, mime="image/png", prompt=prompt, kind="generate",
            alt=alt or prompt, size_bytes=self._size(out),
        )

    def edit(
        self,
        prompt: str,
        input_images: Sequence[Union[str, Path]],
        workspace: Path,
        *,
        filename: Optional[str] = None,
        alt: str = "",
    ) -> ImageResult:
        filename = filename or f"image_edit_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}.png"
        cmd = self.build_edit_command(prompt, input_images, workspace, filename=filename)
        self._execute(cmd)
        out = Path(workspace) / filename
        return ImageResult(
            path=out, mime="image/png", prompt=prompt, kind="edit",
            alt=alt or prompt,
            input_images=[Path(p).expanduser() for p in input_images],
            size_bytes=self._size(out),
        )


# ---------------------------------------------------------------------------
# Fake backend for offline tests
# ---------------------------------------------------------------------------


@dataclass
class FakeImageBackend(ImageBackend):
    """Offline test double for :class:`ImageBackend`.

    Writes a small valid PNG into the workspace (so the artifact is a real
    file) and records every call. No network, no Gemini, no skill scripts.

    Parameters
    ----------
    prompt:
        Optional canned alt text (defaults to the supplied prompt).
    """

    prompt: str = ""
    calls: List[Dict[str, Any]] = field(default_factory=list)

    def generate(
        self,
        prompt: str,
        workspace: Path,
        *,
        filename: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        resolution: Optional[str] = None,
        alt: str = "",
    ) -> ImageResult:
        workspace = Path(workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        filename = filename or "fake_generated.png"
        out = workspace / filename
        out.write_bytes(_MINI_PNG)
        self.calls.append({
            "op": "generate",
            "prompt": prompt,
            "workspace": str(workspace),
            "filename": filename,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "alt": alt,
        })
        return ImageResult(
            path=out, mime="image/png", prompt=prompt, kind="generate",
            alt=alt or self.prompt or prompt, size_bytes=out.stat().st_size,
        )

    def edit(
        self,
        prompt: str,
        input_images: Sequence[Union[str, Path]],
        workspace: Path,
        *,
        filename: Optional[str] = None,
        alt: str = "",
    ) -> ImageResult:
        workspace = Path(workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        filename = filename or "fake_edited.png"
        out = workspace / filename
        out.write_bytes(_MINI_PNG)
        self.calls.append({
            "op": "edit",
            "prompt": prompt,
            "workspace": str(workspace),
            "filename": filename,
            "inputs": [str(Path(p).expanduser()) for p in input_images],
            "alt": alt,
        })
        return ImageResult(
            path=out, mime="image/png", prompt=prompt, kind="edit",
            alt=alt or self.prompt or prompt,
            input_images=[Path(p).expanduser() for p in input_images],
            size_bytes=out.stat().st_size,
        )

    @property
    def generate_calls(self) -> List[Dict[str, Any]]:
        return [c for c in self.calls if c["op"] == "generate"]

    @property
    def edit_calls(self) -> List[Dict[str, Any]]:
        return [c for c in self.calls if c["op"] == "edit"]


# ---------------------------------------------------------------------------
# High-level helpers: backend result -> ArtifactRecord(kind="image")
# ---------------------------------------------------------------------------


def _to_record(result: ImageResult, thread_id: str, message_index: int) -> ArtifactRecord:
    """Wrap an :class:`ImageResult` as an ``image`` artifact record."""
    return ArtifactRecord(
        filename=result.path.name,
        path=result.path,
        kind="image",
        mime=result.mime,
        thread_id=thread_id,
        message_index=message_index,
        created=result.created,
        size_bytes=result.size_bytes,
    )


def generate_image(
    backend: ImageBackend,
    workspace: Path,
    prompt: str,
    thread_id: str = "",
    message_index: int = 0,
    *,
    filename: Optional[str] = None,
    aspect_ratio: Optional[str] = None,
    resolution: Optional[str] = None,
    alt: str = "",
) -> ArtifactRecord:
    """Generate an image into ``workspace`` and return an ``image`` artifact."""
    result = backend.generate(
        prompt, workspace, filename=filename,
        aspect_ratio=aspect_ratio, resolution=resolution, alt=alt,
    )
    return _to_record(result, thread_id, message_index)


def edit_image(
    backend: ImageBackend,
    workspace: Path,
    prompt: str,
    input_images: Sequence[Union[str, Path]],
    thread_id: str = "",
    message_index: int = 0,
    *,
    filename: Optional[str] = None,
    alt: str = "",
) -> ArtifactRecord:
    """Edit images into ``workspace`` and return an ``image`` artifact."""
    result = backend.edit(prompt, input_images, workspace, filename=filename, alt=alt)
    return _to_record(result, thread_id, message_index)


# ---------------------------------------------------------------------------
# ImageService — store integration
# ---------------------------------------------------------------------------


class ImageService:
    """High-level service for creating image artifacts and storing them.

    Parameters
    ----------
    store:
        A :class:`Store` (or compatible object) with an optional
        ``create_artifact`` method (duck-typed so test doubles work).
    backend:
        An :class:`ImageBackend` (SkillImageBackend in production,
        FakeImageBackend in tests).
    """

    def __init__(self, store, backend: ImageBackend) -> None:
        self.store = store
        self.backend = backend

    def _store(self, record: ArtifactRecord) -> ArtifactRecord:
        if hasattr(self.store, "create_artifact"):
            stored = self.store.create_artifact(record.to_dict())
            if stored:
                record._id = stored.get("_id")
        return record

    def create_generate(
        self,
        workspace: Path,
        prompt: str,
        thread_id: str,
        message_index: int = 0,
        *,
        filename: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        resolution: Optional[str] = None,
        alt: str = "",
    ) -> ArtifactRecord:
        """Generate an image and store its artifact record."""
        record = generate_image(
            self.backend, workspace, prompt, thread_id, message_index,
            filename=filename, aspect_ratio=aspect_ratio, resolution=resolution, alt=alt,
        )
        return self._store(record)

    def create_edit(
        self,
        workspace: Path,
        prompt: str,
        input_images: Sequence[Union[str, Path]],
        thread_id: str,
        message_index: int = 0,
        *,
        filename: Optional[str] = None,
        alt: str = "",
    ) -> ArtifactRecord:
        """Edit images and store its artifact record."""
        record = edit_image(
            self.backend, workspace, prompt, input_images, thread_id, message_index,
            filename=filename, alt=alt,
        )
        return self._store(record)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_image_backend(cfg=None, **kwargs) -> ImageBackend:
    """Build the production :class:`SkillImageBackend`.

    The ``cfg``/``kwargs`` hooks exist so callers can override script paths or
    the runner without reaching into the class directly.
    """
    return SkillImageBackend(**kwargs)
