"""Tests for the sandbox's Python environment.

The bwrap sandbox mounts only a read-only ``/usr``, so the interpreter inside
it is the host's bare system Python — typically with no matplotlib. Every
``make_chart`` call therefore failed in production while the whole test suite
stayed green, because the tests use ``FakeRunner`` and never look at what an
interpreter inside the sandbox can import.

These tests cover the build contract without building anything: no ``uv``
invocation, no network, no wheels.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pengyplexity.sandbox.pythonenv import (
    DEFAULT_PACKAGES,
    SandboxEnvError,
    SandboxPythonEnv,
)


class _RecordingEnv(SandboxPythonEnv):
    """A :class:`SandboxPythonEnv` whose subprocess calls are recorded, not run."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.commands: list[list[str]] = []

    def _run(self, argv, what):  # noqa: D102 - see base class
        self.commands.append(list(argv))
        # Simulate the venv layout the real build produces.
        self.python.parent.mkdir(parents=True, exist_ok=True)
        self.python.touch()

    def _verify_visible(self) -> None:
        return None


class TestReadiness:
    def test_absent_environment_is_not_ready(self, tmp_path):
        env = SandboxPythonEnv(tmp_path / "sandbox-venv")
        assert env.ready is False

    def test_marker_without_interpreter_is_not_ready(self, tmp_path):
        path = tmp_path / "sandbox-venv"
        path.mkdir()
        (path / ".pengyplexity-sandbox-ready").write_text("matplotlib\n")

        # A build interrupted after the marker but before the interpreter must
        # not be mistaken for a finished one.
        assert SandboxPythonEnv(path).ready is False

    def test_complete_environment_is_ready(self, tmp_path):
        path = tmp_path / "sandbox-venv"
        (path / "bin").mkdir(parents=True)
        (path / "bin" / "python3").touch()
        (path / ".pengyplexity-sandbox-ready").write_text("matplotlib\n")

        assert SandboxPythonEnv(path).ready is True


class TestBuild:
    def test_pins_the_interpreter_the_sandbox_can_see(self, tmp_path):
        env = _RecordingEnv(tmp_path / "venv", base_interpreter="/usr/bin/python3")

        env.build()

        create = env.commands[0]
        # Left to itself `uv venv` picks a uv-managed CPython from the user's
        # home, which is not mounted inside the sandbox: the interpreter
        # symlink dangles and matplotlib is installed but unimportable.
        assert create[:2] == ["uv", "venv"]
        assert "--python" in create
        assert create[create.index("--python") + 1] == "/usr/bin/python3"

    def test_installs_the_declared_packages(self, tmp_path):
        env = _RecordingEnv(tmp_path / "venv", base_interpreter="/usr/bin/python3")

        env.build()

        install = env.commands[1]
        assert install[:4] == ["uv", "pip", "install", "--python"]
        assert set(DEFAULT_PACKAGES).issubset(install)

    def test_writes_the_ready_marker(self, tmp_path):
        env = _RecordingEnv(tmp_path / "venv", base_interpreter="/usr/bin/python3")

        env.build()

        assert env.ready is True

    def test_rejects_an_interpreter_the_sandbox_cannot_see(self, tmp_path):
        class _Unverified(_RecordingEnv):
            def _verify_visible(self):
                return SandboxPythonEnv._verify_visible(self)

        env = _Unverified(tmp_path / "venv", base_interpreter="/usr/bin/python3")
        with pytest.raises(SandboxEnvError, match="not.*visible inside the sandbox"):
            env.build()

    def test_missing_uv_is_an_actionable_error(self, tmp_path):
        env = SandboxPythonEnv(tmp_path / "venv", uv="definitely-not-a-real-binary")

        with pytest.raises(SandboxEnvError, match="not found on PATH"):
            env.build()


class TestEnsure:
    def test_ready_environment_is_returned_without_building(self, tmp_path):
        env = _RecordingEnv(tmp_path / "venv", base_interpreter="/usr/bin/python3")
        env.build()
        env.commands.clear()

        assert env.ensure() == env.path
        assert env.commands == []

    def test_autobuild_disabled_returns_none(self, tmp_path):
        env = _RecordingEnv(
            tmp_path / "venv", base_interpreter="/usr/bin/python3", auto_build=False
        )

        assert env.ensure() is None
        assert env.commands == []

    def test_a_failed_build_is_not_retried_every_call(self, tmp_path):
        env = SandboxPythonEnv(tmp_path / "venv", uv="definitely-not-a-real-binary")

        assert env.ensure() is None
        assert env.last_error is not None
        # A broken host costs one slow attempt per process, not one per chart.
        assert env.ensure() is None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
