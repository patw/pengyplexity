"""Tests for :mod:`pengyplexity.sandbox.executors`.

These assert the **escape-proof contract** of the bwrap sandbox: the exact
argv that :class:`BwrapRunner` builds contains every safety-critical flag in
the correct form, so that a script run inside it **cannot** escape to the
host filesystem, network, or process tree. No live ``bwrap`` is ever
launched — the tests are pure argv inspection + ``FakeRunner`` behaviour.

Key contracts asserted here (mirroring spec.md):

* ``--unshare-all`` is present (unshare every namespace).
* ``--die-with-parent`` is present (killed when Flask exits).
* ``--new-session`` is present (no controlling TTY).
* ``--tmpfs /`` is present with a curated, read-only ``/usr`` (and its usual
  top-level shims) — NOT ``--ro-bind / /``. The real host root (``/home``,
  ``/root``, etc.) is never bound in at all, not even read-only.
* ``--dev /dev``, ``--proc /proc``, and ``--tmpfs /tmp`` are present
  (minimal kernel interfaces + a private scratch tmp, never the host's).
* ``--bind <workspace> <workspace>`` is present (only the thread workspace is RW).
* ``--chdir <workspace>`` is present (CWD is always the sandbox work dir).
* ``--unshare-net`` / ``--unshare-pid`` / ``--unshare-uts`` /
  ``--unshare-cgroup`` are present (explicit per-namespace drops).
* ``--cap-drop ALL`` is present (zero capabilities).
* ``--uid 65534 --gid 65534`` is present (nobody).
* ``--rlimit-as <bytes>`` and ``--rlimit-cpu <seconds>`` are present
  (resource caps).
* The command after ``--`` is ``/bin/sh -c <script>``.
* No host path is bound other than the curated system dirs (RO) and the
  workspace (RW) — in particular, never ``/``, ``/home``, or ``/root``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pengyplexity.sandbox.executors import (
    PYENV_MOUNT,
    BwrapRunner,
    FakeRunner,
    RecordedCall,
    RunResult,
    Runner,
    _minimal_root_argv,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runner(**kw) -> BwrapRunner:
    """A BwrapRunner with a temp-like workspace for argv inspection."""
    return BwrapRunner(**kw)


def _ws() -> Path:
    """A fake workspace path (never created on disk — argv is pure)."""
    return Path("/tmp/pengyplexity-test-ws/thread1")


# ---------------------------------------------------------------------------
# BwrapRunner.build_argv — the escape-proof contract
# ---------------------------------------------------------------------------


class TestBuildArgv:
    """Assert the exact bwrap argv (no live bwrap)."""

    def test_first_element_is_bwrap(self):
        runner = _make_runner()
        argv = runner.build_argv("echo hi", _ws())
        assert argv[0] == "bwrap"

    def test_custom_bwrap_path(self):
        runner = BwrapRunner(bwrap_path="/usr/bin/bwrap")
        argv = runner.build_argv("true", _ws())
        assert argv[0] == "/usr/bin/bwrap"

    # --- namespace isolation ------------------------------------------------

    def test_unshare_all_present(self):
        runner = _make_runner()
        argv = runner.build_argv("true", _ws())
        assert "--unshare-all" in argv

    def test_unshare_net_present(self):
        runner = _make_runner()
        argv = runner.build_argv("true", _ws())
        assert "--unshare-net" in argv

    def test_unshare_pid_present(self):
        runner = _make_runner()
        argv = runner.build_argv("true", _ws())
        assert "--unshare-pid" in argv

    def test_unshare_uts_present(self):
        runner = _make_runner()
        argv = runner.build_argv("true", _ws())
        assert "--unshare-uts" in argv

    def test_unshare_cgroup_present(self):
        runner = _make_runner()
        argv = runner.build_argv("true", _ws())
        assert "--unshare-cgroup" in argv

    # --- parent / session ---------------------------------------------------

    def test_die_with_parent(self):
        runner = _make_runner()
        argv = runner.build_argv("true", _ws())
        assert "--die-with-parent" in argv

    def test_new_session(self):
        runner = _make_runner()
        argv = runner.build_argv("true", _ws())
        assert "--new-session" in argv

    # --- filesystem confinement ---------------------------------------------

    def test_tmpfs_root_not_full_bind(self):
        """The sandbox root is an empty tmpfs, never the real host root."""
        runner = _make_runner()
        argv = runner.build_argv("true", _ws())
        i = argv.index("--tmpfs")
        assert argv[i + 1] == "/"
        # The old, unsafe contract must never reappear: no ("--ro-bind", "/", "/").
        j = 0
        while True:
            try:
                j = argv.index("--ro-bind", j)
            except ValueError:
                break
            assert argv[j + 1] != "/", "must never --ro-bind the real host root"
            j += 1

    def test_ro_bind_usr(self):
        """/usr (where Python/bash/coreutils live) is bound read-only."""
        runner = _make_runner()
        argv = runner.build_argv("true", _ws())
        found = False
        j = 0
        while True:
            try:
                j = argv.index("--ro-bind", j)
            except ValueError:
                break
            if argv[j + 1] == "/usr" and argv[j + 2] == "/usr":
                found = True
            j += 1
        assert found, "/usr must be bound read-only"

    def test_no_home_or_root_or_etc_exposure(self):
        """/home, /root, and /etc are never bound, shimmed, or symlinked in —
        they simply don't exist inside the sandbox."""
        runner = _make_runner()
        argv = runner.build_argv("true", _ws())
        forbidden = {"/home", "/root", "/etc"}
        i = 0
        while i < len(argv):
            if argv[i] in ("--ro-bind", "--bind", "--symlink") and i + 2 < len(argv):
                assert argv[i + 1] not in forbidden
                assert argv[i + 2] not in forbidden
                i += 3
            else:
                i += 1

    def test_tmp_is_private_tmpfs(self):
        """/tmp inside the sandbox is a private, empty tmpfs — never the host's."""
        runner = _make_runner()
        argv = runner.build_argv("true", _ws())
        i = 0
        found = False
        while True:
            try:
                i = argv.index("--tmpfs", i)
            except ValueError:
                break
            if argv[i + 1] == "/tmp":
                found = True
            i += 1
        assert found

    def test_dev_bind(self):
        """Minimal /dev: --dev /dev"""
        runner = _make_runner()
        argv = runner.build_argv("true", _ws())
        i = argv.index("--dev")
        assert argv[i + 1] == "/dev"

    def test_proc_bind(self):
        """Sandboxed /proc: --proc /proc"""
        runner = _make_runner()
        argv = runner.build_argv("true", _ws())
        i = argv.index("--proc")
        assert argv[i + 1] == "/proc"

    def test_workspace_bind_to_work(self):
        """The thread's workspace is the ONLY writable bind: --bind <ws> /work

        It binds onto the FIXED virtual path /work, never the workspace's own
        real host path — binding at the real path would force bwrap to
        synthesize that path's entire parent directory chain as mount points
        inside the sandbox (e.g. /home/<realuser>/.pengyplexity/workspaces/...),
        leaking the real host username and the app's internal data layout.
        """
        ws = Path("/tmp/pengyplexity-test-ws/thread1")
        runner = _make_runner()
        argv = runner.build_argv("ls", ws)
        # There must be exactly one --bind (the workspace), not --ro-bind.
        i = argv.index("--bind")
        assert argv[i + 1] == str(ws.resolve())
        assert argv[i + 2] == "/work"

    def test_chdir_work(self):
        """CWD inside the sandbox is the fixed virtual path: --chdir /work"""
        runner = _make_runner()
        argv = runner.build_argv("pwd", _ws())
        i = argv.index("--chdir")
        assert argv[i + 1] == "/work"

    def test_only_one_writable_bind(self):
        """Only --bind (RW) is the workspace; all other binds are --ro-bind."""
        runner = _make_runner()
        argv = runner.build_argv("true", _ws())
        # Count --bind occurrences (not --ro-bind).
        binds = [i for i, a in enumerate(argv) if a == "--bind"]
        assert len(binds) == 1
        # The single --bind must be the workspace's real path -> /work.
        assert argv[binds[0] + 1] == str(_ws().resolve())
        assert argv[binds[0] + 2] == "/work"

    # --- capability / identity ----------------------------------------------

    def test_cap_drop_all(self):
        """All capabilities dropped: --cap-drop ALL"""
        runner = _make_runner()
        argv = runner.build_argv("true", _ws())
        i = argv.index("--cap-drop")
        assert argv[i + 1] == "ALL"

    def test_uid_65534(self):
        """Runs as nobody: --uid 65534"""
        runner = _make_runner()
        argv = runner.build_argv("true", _ws())
        i = argv.index("--uid")
        assert argv[i + 1] == "65534"

    def test_gid_65534(self):
        """Runs as nogroup: --gid 65534"""
        runner = _make_runner()
        argv = runner.build_argv("true", _ws())
        i = argv.index("--gid")
        assert argv[i + 1] == "65534"

    # --- resource caps (only when the binary supports them) --------------

    def test_rlimit_as_default(self):
        """Default memory cap is 512 MiB (when the binary supports rlimit)."""
        runner = _make_runner()
        argv = runner.build_argv("true", _ws())
        if runner.rlimit_supported:
            i = argv.index("--rlimit-as")
            assert argv[i + 1] == str(512 * 1024 * 1024)
        else:
            assert "--rlimit-as" not in argv

    def test_rlimit_as_custom(self):
        """Custom memory cap is reflected in argv (when supported)."""
        runner = BwrapRunner(mem_bytes=256 * 1024 * 1024)
        argv = runner.build_argv("true", _ws())
        if runner.rlimit_supported:
            i = argv.index("--rlimit-as")
            assert argv[i + 1] == str(256 * 1024 * 1024)
        else:
            assert "--rlimit-as" not in argv

    def test_rlimit_cpu_default(self):
        """Default CPU cap is 30 seconds (when supported)."""
        runner = _make_runner()
        argv = runner.build_argv("true", _ws())
        if runner.rlimit_supported:
            i = argv.index("--rlimit-cpu")
            assert argv[i + 1] == "30"
        else:
            assert "--rlimit-cpu" not in argv

    def test_rlimit_cpu_custom(self):
        """Custom CPU cap is reflected in argv (when supported)."""
        runner = BwrapRunner(cpu_seconds=10)
        argv = runner.build_argv("true", _ws())
        if runner.rlimit_supported:
            i = argv.index("--rlimit-cpu")
            assert argv[i + 1] == "10"
        else:
            assert "--rlimit-cpu" not in argv

    def test_rlimit_flags_reflect_detection(self):
        """The argv must never contain rlimit flags the binary rejects."""
        runner = _make_runner()
        argv = runner.build_argv("true", _ws())
        assert ("--rlimit-as" in argv) == runner.rlimit_supported
        assert ("--rlimit-cpu" in argv) == runner.rlimit_supported

    # --- command tail -------------------------------------------------------

    def test_separator_dashdash(self):
        """The -- separator is present (bwrap options end, command begins)."""
        runner = _make_runner()
        argv = runner.build_argv("echo hi", _ws())
        assert "--" in argv

    def test_command_is_sh_minus_c_script(self):
        """After --: /bin/sh -c <script>"""
        script = "echo hello world"
        runner = _make_runner()
        argv = runner.build_argv(script, _ws())
        i = argv.index("--")
        assert argv[i + 1] == "/bin/sh"
        assert argv[i + 2] == "-c"
        assert argv[i + 3] == script

    def test_script_appears_only_after_separator(self):
        """The script string does not appear in the bwrap options section."""
        script = "MY_UNIQUE_SCRIPT_MARKER_12345"
        runner = _make_runner()
        argv = runner.build_argv(script, _ws())
        i = argv.index("--")
        # Before --: only bwrap flags.
        assert script not in argv[:i]
        # After --: it's the last element.
        assert argv[-1] == script

    # --- no host paths leaked ------------------------------------------------

    def test_no_host_path_bind_other_than_workspace(self):
        """The only --bind is the workspace; no other host path is mounted RW."""
        ws = Path("/tmp/specific-ws/thread42")
        runner = _make_runner()
        argv = runner.build_argv("true", ws)
        # Collect all paths that are arguments to --bind.
        bind_targets = []
        for idx, a in enumerate(argv):
            if a == "--bind" and idx + 1 < len(argv):
                bind_targets.append(argv[idx + 1])
        assert bind_targets == [str(ws.resolve())]

    def test_workspace_in_argv_is_resolved(self):
        """The workspace path in argv is resolved (absolute, no ..)."""
        ws = Path("/tmp/ws/./thread/../thread1")
        runner = _make_runner()
        argv = runner.build_argv("true", ws)
        i = argv.index("--bind")
        ws_arg = argv[i + 1]
        # Must be absolute and fully resolved.
        assert Path(ws_arg).is_absolute()
        assert ".." not in Path(ws_arg).parts


# ---------------------------------------------------------------------------
# Runner interface
# ---------------------------------------------------------------------------


class TestRunnerInterface:
    """BwrapRunner and FakeRunner satisfy the Runner ABC."""

    def test_bwrap_runner_is_runner(self):
        r = BwrapRunner()
        assert isinstance(r, Runner)

    def test_fake_runner_is_runner(self):
        r = FakeRunner()
        assert isinstance(r, Runner)


# ---------------------------------------------------------------------------
# RunResult
# ---------------------------------------------------------------------------


class TestRunResult:
    """Basic RunResult dataclass behaviour."""

    def test_ok_true_when_rc_zero(self):
        assert RunResult(stdout="ok", stderr="", returncode=0).ok is True

    def test_ok_false_when_rc_nonzero(self):
        assert RunResult(stdout="", stderr="err", returncode=1).ok is False

    def test_fields(self):
        r = RunResult(stdout="out", stderr="err", returncode=2)
        assert r.stdout == "out"
        assert r.stderr == "err"
        assert r.returncode == 2


# ---------------------------------------------------------------------------
# FakeRunner — records and returns canned result
# ---------------------------------------------------------------------------


class TestFakeRunner:
    """FakeRunner records calls and returns a deterministic result."""

    def test_returns_default_canned_result(self):
        fr = FakeRunner()
        result = fr.run("echo hi", Path("/ws"))
        assert result == RunResult(stdout="", stderr="", returncode=0)
        assert result.ok

    def test_returns_custom_canned_result(self):
        canned = RunResult(stdout="42", stderr="warning\n", returncode=0)
        fr = FakeRunner(result=canned)
        result = fr.run("cat answer.txt", Path("/ws"))
        assert result == canned
        assert result.stdout == "42"
        assert result.stderr == "warning\n"

    def test_records_calls_in_order(self):
        fr = FakeRunner()
        fr.run("cmd1", Path("/ws1"))
        fr.run("cmd2", Path("/ws2"))
        assert len(fr.calls) == 2
        assert fr.calls[0].script == "cmd1"
        assert fr.calls[0].workspace == Path("/ws1")
        assert fr.calls[1].script == "cmd2"
        assert fr.calls[1].workspace == Path("/ws2")

    def test_last_call(self):
        fr = FakeRunner()
        assert fr.last_call is None
        fr.run("ls", Path("/a"))
        fr.run("pwd", Path("/b"))
        assert fr.last_call is not None
        assert fr.last_call.script == "pwd"
        assert fr.last_call.workspace == Path("/b")

    def test_returns_same_result_every_call(self):
        canned = RunResult(stdout="same", stderr="", returncode=0)
        fr = FakeRunner(result=canned)
        r1 = fr.run("a", Path("/1"))
        r2 = fr.run("b", Path("/2"))
        assert r1 == r2 == canned

    def test_call_record_is_frozen(self):
        fr = FakeRunner()
        fr.run("x", Path("/ws"))
        call = fr.calls[0]
        assert isinstance(call, RecordedCall)
        assert call.script == "x"
        assert call.workspace == Path("/ws")

    def test_multiple_runners_independent(self):
        """Each FakeRunner has its own call list."""
        fr1 = FakeRunner()
        fr2 = FakeRunner()
        fr1.run("a", Path("/1"))
        assert len(fr1.calls) == 1
        assert len(fr2.calls) == 0


# ---------------------------------------------------------------------------
# BwrapRunner timeout config
# ---------------------------------------------------------------------------


class TestBwrapRunnerConfig:
    """Timeout and rlimit config is stored and accessible."""

    def test_default_timeout(self):
        assert BwrapRunner().timeout == 30

    def test_custom_timeout(self):
        assert BwrapRunner(timeout=5).timeout == 5

    def test_default_mem_bytes(self):
        assert BwrapRunner().mem_bytes == 512 * 1024 * 1024

    def test_default_cpu_seconds(self):
        assert BwrapRunner().cpu_seconds == 30


# ---------------------------------------------------------------------------
# Integration: the full argv shape (ordered prefix)
# ---------------------------------------------------------------------------


class TestFullArgvShape:
    """Assert the argv has the documented structure end-to-end."""

    def test_argv_structure(self):
        """The argv follows the documented pattern exactly.

        bwrap [flags...] -- /bin/sh -c <script>
        """
        ws = Path("/tmp/pengyplexity-test-ws/thread1")
        runner = BwrapRunner(timeout=30, mem_bytes=512 * 1024 * 1024, cpu_seconds=30)
        argv = runner.build_argv("python3 -c 'print(1+1)'", ws)

        # Structure: bwrap <flags> -- /bin/sh -c <script>
        assert argv[0] == "bwrap"
        sep = argv.index("--")
        # After --: exactly /bin/sh -c <script>
        assert argv[sep + 1:] == ["/bin/sh", "-c", "python3 -c 'print(1+1)'"]
        # Before --: all are bwrap options (none should be a path we didn't expect)
        flags_section = argv[1:sep]
        # The only paths in the flags section should be the curated minimal
        # root (see _minimal_root_argv: tmpfs "/", "/usr", and whichever of
        # the usual shims exist on this host), "/dev", "/proc", the private
        # "/tmp", and the workspace — no other host user paths.
        expected_paths = {
            tok for tok in _minimal_root_argv() if not tok.startswith("--")
        } | {"/dev", "/proc", "/tmp", "/work", str(ws.resolve())}
        actual_paths = set()
        j = 0
        while j < len(flags_section):
            tok = flags_section[j]
            if tok == "--setenv":
                # --setenv NAME VALUE: neither is a host path, and the values
                # are asserted on their own in TestEnvironmentIsolation.
                j += 3
                continue
            if tok.startswith("--"):
                j += 1
                continue
            if tok == "ALL" or tok.isdigit():
                j += 1
                continue
            # This is a positional arg to a flag; it should be a known path.
            actual_paths.add(tok)
            j += 1
        # All positional args in the flags section must be from the expected set.
        assert actual_paths.issubset(expected_paths), (
            f"Unexpected paths in bwrap argv: {actual_paths - expected_paths}"
        )

    def test_full_argv_with_all_flags(self):
        """Spot-check: every documented flag appears in the argv."""
        ws = Path("/tmp/ws/t1")
        runner = BwrapRunner()
        argv = runner.build_argv("true", ws)
        required_flags = [
            "--unshare-all",
            "--die-with-parent",
            "--new-session",
            "--ro-bind",
            "--dev",
            "--proc",
            "--bind",
            "--chdir",
            "--unshare-net",
            "--unshare-pid",
            "--unshare-uts",
            "--unshare-cgroup",
            "--cap-drop",
            "--uid",
            "--gid",
        ]
        if runner.rlimit_supported:
            required_flags += ["--rlimit-as", "--rlimit-cpu"]
        for flag in required_flags:
            assert flag in argv, f"Missing required flag: {flag}"

    def test_argv_is_deterministic(self):
        """Same inputs → same argv (no randomness, no env dependence)."""
        ws = Path("/tmp/ws/t1")
        runner = BwrapRunner()
        argv1 = runner.build_argv("echo test", ws)
        argv2 = runner.build_argv("echo test", ws)
        assert argv1 == argv2

    def test_different_scripts_give_different_argv(self):
        ws = Path("/tmp/ws/t1")
        runner = BwrapRunner()
        argv1 = runner.build_argv("cmd_a", ws)
        argv2 = runner.build_argv("cmd_b", ws)
        # Only the last element (script) differs.
        assert argv1[:-1] == argv2[:-1]
        assert argv1[-1] == "cmd_a"
        assert argv2[-1] == "cmd_b"


if __name__ == "__main__":  # pragma: no cover - manual smoke
    raise SystemExit(pytest.main([__file__, "-q"]))


# ---------------------------------------------------------------------------
# Environment isolation
# ---------------------------------------------------------------------------


class TestEnvironmentIsolation:
    """The sandbox must not inherit the app's environment.

    bwrap namespaces isolate the filesystem, the network and the process tree,
    but the environment is passed straight through unless it is cleared. The
    app's environment carries its LLM API key, the image-generation key, the
    real host username and home directory, and session socket paths — all of
    which a single `env` from run_bash would have printed back to the model.
    """

    def test_clearenv_precedes_every_setenv(self):
        argv = BwrapRunner().build_argv("env", Path("/tmp/ws"))
        assert "--clearenv" in argv
        assert argv.index("--clearenv") < argv.index("--setenv")

    def test_environment_is_exactly_the_curated_set(self):
        argv = BwrapRunner().build_argv("env", Path("/tmp/ws"))
        env = {}
        for i, tok in enumerate(argv):
            if tok == "--setenv":
                env[argv[i + 1]] = argv[i + 2]

        assert set(env) == {
            "PATH", "HOME", "TMPDIR", "MPLCONFIGDIR", "XDG_CACHE_HOME",
            "LANG", "LC_ALL", "PYTHONUNBUFFERED", "PYTHONDONTWRITEBYTECODE",
            "TERM",
        }
        # Nothing points back at a host home directory.
        assert env["HOME"] == "/tmp"
        assert "/home/" not in " ".join(env.values())

    def test_python_env_is_bound_and_first_on_path(self):
        argv = BwrapRunner(python_env="/data/sandbox-venv").build_argv(
            "python3 x.py", Path("/tmp/ws")
        )
        at = argv.index("/data/sandbox-venv")
        assert argv[at - 1 : at + 2] == [
            "--ro-bind", "/data/sandbox-venv", PYENV_MOUNT
        ]
        assert argv[argv.index("PATH") + 1].startswith(f"{PYENV_MOUNT}/bin:")

    def test_prepare_resolves_the_env_from_its_provider(self):
        runner = BwrapRunner(python_env_provider=lambda: "/data/sandbox-venv")
        assert runner.python_env is None
        runner.prepare()
        assert str(runner.python_env) == "/data/sandbox-venv"
        assert PYENV_MOUNT in runner.build_argv("python3 x.py", Path("/tmp/ws"))

    def test_no_python_env_means_no_pyenv_bind(self):
        argv = BwrapRunner().build_argv("python3 x.py", Path("/tmp/ws"))
        assert PYENV_MOUNT not in argv
        assert PYENV_MOUNT not in argv[argv.index("PATH") + 1]
