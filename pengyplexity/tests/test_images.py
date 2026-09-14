"""Tests for :mod:`pengyplexity.core.images`.

These verify, all offline (no Gemini, no network, no skill scripts executed):

* :class:`ImageResult` dataclass fields and ``to_dict``.
* :class:`FakeImageBackend` writes a real valid PNG into the workspace and
  records calls for both ``generate`` and ``edit``.
* :class:`SkillImageBackend` builds a *confined* command: the output is always
  ``-o <workspace>/<filename>`` (never ``~/Pictures``) and ``--no-upload`` is
  always present — the image-stays-in-the-sandbox contract.
* The high-level helpers turn a backend result into an ``image``
  :class:`~pengyplexity.core.artifacts.ArtifactRecord`.
* :class:`ImageService` stores the artifact record when the store supports it.

The real skill scripts are never invoked — command construction is pure and
execution is injected (or not exercised) so the suite stays green offline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pengyplexity.core.artifacts import ArtifactRecord
from pengyplexity.core.images import (
    FakeImageBackend,
    ImageBackend,
    ImageResult,
    ImageService,
    SkillImageBackend,
    edit_image,
    generate_image,
)
from pengyplexity.core.store import Store


# A 1x1 PNG must start with this magic prefix.
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# ImageResult
# ---------------------------------------------------------------------------


class TestImageResult:
    def test_fields(self):
        r = ImageResult(
            path=Path("/ws/cat.png"),
            mime="image/png",
            prompt="a corgi",
            kind="generate",
            alt="a corgi",
            size_bytes=68,
        )
        assert r.path == Path("/ws/cat.png")
        assert r.mime == "image/png"
        assert r.kind == "generate"
        assert r.size_bytes == 68
        assert r.created is not None

    def test_to_dict(self):
        r = ImageResult(
            path=Path("/ws/x.png"),
            kind="edit",
            prompt="bw",
            input_images=[Path("/ws/in.png")],
        )
        d = r.to_dict()
        assert d["path"] == "/ws/x.png"
        assert d["kind"] == "edit"
        assert d["input_images"] == ["/ws/in.png"]
        assert d["mime"] == "image/png"

    def test_defaults(self):
        r = ImageResult(path=Path("/a.png"))
        assert r.mime == "image/png"
        assert r.kind == "generate"
        assert r.input_images == []


# ---------------------------------------------------------------------------
# FakeImageBackend
# ---------------------------------------------------------------------------


class TestFakeBackend:
    def test_is_image_backend(self):
        assert isinstance(FakeImageBackend(), ImageBackend)

    def test_generate_writes_valid_png_in_workspace(self, tmp_path):
        fake = FakeImageBackend()
        result = fake.generate("a corgi", tmp_path, filename="corgi.png")
        assert result.kind == "generate"
        assert result.path == tmp_path / "corgi.png"
        assert result.path.exists()
        assert result.path.read_bytes().startswith(PNG_MAGIC)
        assert result.size_bytes == result.path.stat().st_size
        assert result.size_bytes > 0

    def test_generate_records_call(self, tmp_path):
        fake = FakeImageBackend()
        fake.generate("corgi", tmp_path, filename="c.png", aspect_ratio="16:9")
        assert len(fake.generate_calls) == 1
        call = fake.calls[0]
        assert call["op"] == "generate"
        assert call["prompt"] == "corgi"
        assert call["aspect_ratio"] == "16:9"
        assert call["workspace"] == str(tmp_path)

    def test_edit_writes_png_and_records_inputs(self, tmp_path):
        fake = FakeImageBackend()
        inp = tmp_path / "in.png"
        inp.write_bytes(b"input")
        result = fake.edit("make bw", [inp], tmp_path, filename="out.png")
        assert result.kind == "edit"
        assert result.path == tmp_path / "out.png"
        assert result.path.read_bytes().startswith(PNG_MAGIC)
        assert result.input_images == [inp]
        call = fake.edit_calls[0]
        assert call["op"] == "edit"
        assert call["inputs"] == [str(inp)]

    def test_default_filename(self, tmp_path):
        fake = FakeImageBackend()
        g = fake.generate("x", tmp_path)
        e = fake.edit("x", [], tmp_path)
        assert g.path.name == "fake_generated.png"
        assert e.path.name == "fake_edited.png"

    def test_multiple_calls_recorded(self, tmp_path):
        fake = FakeImageBackend()
        fake.generate("a", tmp_path, filename="1.png")
        fake.generate("b", tmp_path, filename="2.png")
        fake.edit("c", [], tmp_path, filename="3.png")
        assert len(fake.calls) == 3
        assert len(fake.generate_calls) == 2
        assert len(fake.edit_calls) == 1

    def test_alt_prefers_explicit(self, tmp_path):
        fake = FakeImageBackend(prompt="canned")
        r1 = fake.generate("p", tmp_path, filename="a.png", alt="explicit")
        r2 = fake.generate("p", tmp_path, filename="b.png")
        assert r1.alt == "explicit"
        assert r2.alt == "canned"  # falls back to backend.prompt


# ---------------------------------------------------------------------------
# SkillImageBackend — the confinement contract (pure command building)
# ---------------------------------------------------------------------------


class TestSkillCommandBuilding:
    @pytest.fixture
    def backend(self):
        return SkillImageBackend(
            gen_script="/skills/gen_image.py",
            edit_script="/skills/edit_image.py",
            uv="uv",
        )

    def test_is_image_backend(self, backend):
        assert isinstance(backend, ImageBackend)

    def test_generate_command_output_is_in_workspace(self, backend):
        cmd = backend.build_generate_command(
            "a corgi", Path("/ws/thread1"), filename="out.png"
        )
        # output flag: -o must be immediately followed by a workspace path
        assert "-o" in cmd
        assert cmd[cmd.index("-o") + 1] == str(Path("/ws/thread1") / "out.png")
        assert "/ws/thread1" in cmd[cmd.index("-o") + 1]

    def test_generate_command_never_targets_pictures(self, backend):
        cmd = backend.build_generate_command("x", Path("/ws/t"), filename="o.png")
        joined = " ".join(cmd)
        assert "Pictures" not in joined

    def test_generate_command_always_no_upload(self, backend):
        cmd = backend.build_generate_command("x", Path("/ws/t"), filename="o.png")
        assert "--no-upload" in cmd

    def test_generate_command_includes_prompt_and_script(self, backend):
        cmd = backend.build_generate_command("a corgi", Path("/ws/t"), filename="o.png")
        assert "/skills/gen_image.py" in cmd
        assert "a corgi" in cmd
        assert cmd[0] == "uv"
        assert cmd[1] == "run"

    def test_generate_command_optional_flags(self, backend):
        cmd = backend.build_generate_command(
            "x", Path("/ws/t"), filename="o.png", aspect_ratio="16:9", resolution="2K"
        )
        assert "--aspect-ratio" in cmd and "16:9" in cmd
        assert "--resolution" in cmd and "2K" in cmd
        # when omitted, they are absent
        plain = backend.build_generate_command("x", Path("/ws/t"), filename="o.png")
        assert "--aspect-ratio" not in plain
        assert "--resolution" not in plain

    def test_edit_command_input_images_via_i(self, backend):
        inputs = [Path("/ws/t/a.png"), Path("/ws/t/b.png")]
        cmd = backend.build_edit_command("group", inputs, Path("/ws/t"), filename="g.png")
        assert "-o" in cmd
        assert cmd[cmd.index("-o") + 1] == str(Path("/ws/t") / "g.png")
        # both inputs present via repeated -i
        assert cmd.count("-i") == 2
        assert "/ws/t/a.png" in cmd
        assert "/ws/t/b.png" in cmd
        assert "/skills/edit_image.py" in cmd

    def test_edit_command_never_targets_pictures_and_no_upload(self, backend):
        cmd = backend.build_edit_command(
            "x", [Path("/ws/t/a.png")], Path("/ws/t"), filename="o.png"
        )
        joined = " ".join(cmd)
        assert "Pictures" not in joined
        assert "--no-upload" in cmd


class TestSkillExecutionInjection:
    def test_generate_runs_injected_runner_and_reports_size(self, tmp_path):
        # The fake runner writes the file so the result reflects a real artifact.
        def fake_runner(cmd):
            # find the -o target and write a PNG
            out = Path(cmd[cmd.index("-o") + 1])
            out.write_bytes(PNG_MAGIC + b"fake")
        backend = SkillImageBackend(
            gen_script="/skills/gen_image.py", runner=fake_runner
        )
        result = backend.generate("x", tmp_path, filename="out.png")
        assert result.path == tmp_path / "out.png"
        assert result.path.exists()
        assert result.size_bytes == result.path.stat().st_size

    def test_runner_receives_command(self, tmp_path):
        seen = {}

        def rec(cmd):
            seen["cmd"] = cmd
        backend = SkillImageBackend(
            gen_script="/skills/gen_image.py", runner=rec
        )
        backend.edit("x", [tmp_path / "in.png"], tmp_path, filename="o.png")
        assert seen["cmd"][0] == "uv"
        assert "--no-upload" in seen["cmd"]


# ---------------------------------------------------------------------------
# High-level helpers -> ArtifactRecord
# ---------------------------------------------------------------------------


class TestHighLevel:
    def test_generate_image_returns_image_record(self, tmp_path):
        fake = FakeImageBackend()
        record = generate_image(fake, tmp_path, "a corgi", thread_id="t1", message_index=3)
        assert isinstance(record, ArtifactRecord)
        assert record.kind == "image"
        assert record.mime == "image/png"
        assert record.thread_id == "t1"
        assert record.message_index == 3
        assert record.path == tmp_path / "fake_generated.png"
        assert record.path.exists()

    def test_edit_image_returns_image_record(self, tmp_path):
        fake = FakeImageBackend()
        inp = tmp_path / "in.png"
        inp.write_bytes(b"x")
        record = edit_image(fake, tmp_path, "bw", [inp], thread_id="t2", message_index=0)
        assert record.kind == "image"
        assert record.path == tmp_path / "fake_edited.png"
        assert record.path.exists()

    def test_record_to_dict_has_image_kind(self, tmp_path):
        fake = FakeImageBackend()
        record = generate_image(fake, tmp_path, "corgi")
        d = record.to_dict()
        assert d["kind"] == "image"
        assert d["mime"] == "image/png"


# ---------------------------------------------------------------------------
# ImageService — store integration
# ---------------------------------------------------------------------------


class _RecordingStore:
    def __init__(self):
        self.artifacts = []

    def create_artifact(self, record):
        self.artifacts.append(record)


class TestImageService:
    def test_service_generate_stores_record(self, tmp_path):
        store = _RecordingStore()
        service = ImageService(store, FakeImageBackend())
        record = service.create_generate(tmp_path, "corgi", "thread9", message_index=1)
        assert record.kind == "image"
        assert len(store.artifacts) == 1
        assert store.artifacts[0]["kind"] == "image"
        assert store.artifacts[0]["thread_id"] == "thread9"

    def test_service_edit_stores_record(self, tmp_path):
        store = _RecordingStore()
        service = ImageService(store, FakeImageBackend())
        inp = tmp_path / "in.png"
        inp.write_bytes(b"x")
        record = service.create_edit(tmp_path, "bw", [inp], "thread9", message_index=2)
        assert record.kind == "image"
        assert len(store.artifacts) == 1
        assert store.artifacts[0]["message_index"] == 2

    def test_service_works_with_real_store(self, tmp_path):
        """The real moofile Store has no create_artifact; the service must not
        crash (duck-typed guard)."""
        store = Store(tmp_path / "store")
        try:
            service = ImageService(store, FakeImageBackend())
            record = service.create_generate(tmp_path, "corgi", "t1")
            assert record.kind == "image"
            assert record.path.exists()
        finally:
            store.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
