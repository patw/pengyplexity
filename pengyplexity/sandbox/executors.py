"""Sandboxed code execution via bubblewrap (``bwrap``).

This is the *execution* half of the escape-proof contract. ``run_python`` and
``run_bash`` are **never** run on the host — they are confined to a per-thread
workspace via a ``bwrap`` sandbox that:

* mounts a **minimal, curated root filesystem** — an empty ``tmpfs`` at ``/``
  with only ``/usr`` (and its usual top-level shims: ``/bin``, ``/sbin``,
  ``/lib``, ``/lib64``, ``/lib32``) bound in read-only, NOT the real host
  root. ``/home``, ``/root``, and every other host path are simply absent
  inside the sandbox — there is nothing to read, not even read-only. (An
  earlier version of this sandbox used ``--ro-bind / /``, which — despite
  being read-only — exposed the *entire* host filesystem, including
  ``/home``, to any ``run_bash``/``run_python`` call. bwrap does not give you
  an isolated filesystem for free the way a container image does: whatever
  you bind is what the sandboxed process sees, so binding the whole host root
  made "read-only" the only protection, not filesystem isolation.)
* binds **only** the thread's workspace as ``/work`` (read-write),
* drops **all** capabilities (``--cap-drop ALL``),
* unshares **all** namespaces (PID, net, UTS, cgroup, IPC, user),
* runs as ``nobody`` (uid 65534, gid 65534),
* enforces resource caps (``--rlimit-as`` / ``--rlimit-cpu``),
* is killed when the parent dies (``--die-with-parent``),
* has a hard wall-clock timeout.

The test suite asserts the **constructed argv** (the escape-proof contract)
without ever invoking a live ``bwrap`` — see
:mod:`pengyplexity.tests.test_executors`.

Design notes:

* ``BwrapRunner.build_argv`` is a pure function: it takes a script string and
  a workspace ``Path`` and returns the full argument vector. No subprocess, no
  filesystem access, no network — fully testable offline.
* ``BwrapRunner.run`` actually launches the process via ``subprocess.run`` with
  a timeout. It is **not** called in the test suite.
* ``FakeRunner`` records each call (script, workspace) and returns a canned
  ``RunResult``. It is the default for tests and for any code that needs a
  deterministic, offline runner.
* The *script* is always a shell command string (``/bin/sh -c <script>``). For
  ``run_python`` the caller wraps the Python code into a shell command (e.g.
  write it to a temp file first, or use ``python3 -c '...'``).
"""

from __future__ import annotations

import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


def _suggests_rlimit_support(bwrap_path: str = "bwrap") -> bool:
    """Best-effort check whether the installed ``bwrap`` supports rlimit caps.

    bwrap >= 0.11 removed ``--rlimit-as`` / ``--rlimit-cpu`` (the memory / CPU
    resource-cap flags). We inspect ``bwrap --help`` once per binary so the
    derived argv only emits those flags when the binary actually accepts them —
    otherwise a valid sandbox command fails with ``Unknown option``.

    Returns True conservatively when the binary cannot be inspected (so the
    pre-0.11 flags are still emitted on hosts that support them).
    """
    try:
        which = shutil.which(bwrap_path)
        if not which:
            return True
        proc = subprocess.run(
            [which, "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode != 0:
            return True
        return "--rlimit-as" in proc.stdout and "--rlimit-cpu" in proc.stdout
    except Exception:  # noqa: BLE001
        # If we cannot inspect, assume support (conservative; pre-0.11 flags).
        return True


# Only these top-level dirs are ever bound into the sandbox root — everything
# else (/home, /root, /etc, /var, /mnt, ...) simply does not exist inside it.
# /usr is required (it's where Python/bash/coreutils actually live on a
# modern usrmerge distro); the rest are the traditional top-level dirs that
# are USUALLY symlinks into /usr — bound only if present on this host.
_REQUIRED_SYSTEM_DIRS = ("/usr",)
_OPTIONAL_SYSTEM_SHIMS = ("/bin", "/sbin", "/lib", "/lib64", "/lib32")

# Read-only host config that is public, carries no secrets, and that common
# libraries look for. Bound with --ro-bind-try so a host missing them is fine.
_OPTIONAL_RO_CONFIG = ("/etc/fonts",)

# The mountpoint the optional sandbox Python environment is bound at. Like
# /work, it is a fixed virtual path that says nothing about the host layout.
PYENV_MOUNT = "/pyenv"

# The sandbox runs with --clearenv, so this is the ENTIRE environment the
# model's code sees. The host environment is never inherited: it carries the
# app's own LLM/image API keys, the real HOME and username, SSH_CONNECTION,
# and IPC socket paths, all of which would otherwise be readable by a single
# `env` call from run_bash.
_BASE_ENV = {
    "HOME": "/tmp",
    "TMPDIR": "/tmp",
    # matplotlib writes a font cache on first use; point it at the private
    # tmpfs so it never litters the user-visible workspace.
    "MPLCONFIGDIR": "/tmp/.mplconfig",
    "XDG_CACHE_HOME": "/tmp/.cache",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PYTHONUNBUFFERED": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "TERM": "dumb",
}

_BASE_PATH = "/usr/local/bin:/usr/bin:/bin"


def _minimal_root_argv() -> List[str]:
    """Bwrap args building a minimal root filesystem: an empty ``tmpfs`` at
    ``/`` plus just enough of the host's ``/usr`` tree for Python/bash/
    coreutils to run.

    This is what replaces ``--ro-bind / /``. That flag is read-only, but
    still mounts the **entire real host filesystem** inside the sandbox —
    bwrap namespaces don't give you a separate filesystem the way a container
    image does; whatever you bind is what the sandboxed process sees. Binding
    the whole root meant a sandboxed script could `ls ~` and see the real
    host home directory, `/etc/passwd`, and anything else world-readable.

    ``/bin``, ``/sbin``, ``/lib``, ``/lib64``, ``/lib32`` are typically
    symlinks into ``/usr`` (usrmerge) — recreated here as virtual symlinks
    (no extra mount). On a host where one is instead a real directory, it's
    bound read-only directly. Either way, no other host path is ever visible.
    """
    argv: List[str] = ["--tmpfs", "/"]
    for d in _REQUIRED_SYSTEM_DIRS:
        argv += ["--ro-bind", d, d]
    for d in _OPTIONAL_SYSTEM_SHIMS:
        p = Path(d)
        if p.is_symlink():
            argv += ["--symlink", str(p.resolve()), d]
        elif p.is_dir():
            argv += ["--ro-bind", d, d]
        # else: doesn't exist on this host — nothing to bind.
    for d in _OPTIONAL_RO_CONFIG:
        # --ro-bind-try: bind if present, silently skip if not, so the argv is
        # identical on every host and a missing /etc/fonts is not a hard error.
        argv += ["--ro-bind-try", d, d]
    return argv


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunResult:
    """Immutable result of a sandboxed script execution."""

    stdout: str
    stderr: str
    returncode: int

    @property
    def ok(self) -> bool:
        """True if the script exited with status 0."""
        return self.returncode == 0


# ---------------------------------------------------------------------------
# Runner interface
# ---------------------------------------------------------------------------


class Runner(ABC):
    """Abstract interface for sandboxed script execution.

    Implementations must confine the script to the given *workspace*: the
    script's CWD is ``/work`` (the workspace root, bind-mounted), and the host
    filesystem is read-only.
    """

    @abstractmethod
    def run(self, script: str, workspace: Path, cancel=None) -> RunResult:
        """Execute *script* (a shell command) with CWD = *workspace*.

        Returns a :class:`RunResult` with stdout, stderr, and returncode. A
        script that exceeds the configured wall-clock timeout comes back as a
        non-zero result explaining that, not as an exception.

        *cancel* is this turn's optional
        :class:`~pengyplexity.core.cancel.CancelToken`. The running process is
        registered on it so pressing Stop kills it straight away rather than
        leaving the user waiting out the execution timeout.
        """
        ...

    def prepare(self) -> None:
        """Make the runner ready to execute (default: nothing to do).

        Called before a batch of runs. :class:`BwrapRunner` uses it to resolve
        (and, the first time, build) the sandbox Python environment, which
        needs host network access and so cannot happen inside the sandbox.
        """
        return None


# ---------------------------------------------------------------------------
# BwrapRunner — the real sandbox backend
# ---------------------------------------------------------------------------


class BwrapRunner(Runner):
    """Executes scripts inside a ``bwrap`` (bubblewrap) sandbox.

    The argv is built by :meth:`build_argv`, which is a **pure function** and
    the primary target of the test suite (it asserts the escape-proof flags
    without launching anything).

    Parameters
    ----------
    bwrap_path:
        Path to the ``bwrap`` binary (default: ``"bwrap"``, found on PATH).
    timeout:
        Hard wall-clock timeout in seconds. The child is killed if it
        exceeds this.
    mem_bytes:
        Address-space memory cap in bytes (``--rlimit-as``).
    cpu_seconds:
        CPU-time cap in seconds (``--rlimit-cpu``).
    """

    def __init__(
        self,
        bwrap_path: str = "bwrap",
        timeout: int = 30,
        mem_bytes: int = 512 * 1024 * 1024,  # 512 MiB
        cpu_seconds: int = 30,
        python_env: "Path | str | None" = None,
        python_env_provider=None,
    ) -> None:
        self.bwrap_path = bwrap_path
        self.timeout = timeout
        self.mem_bytes = mem_bytes
        self.cpu_seconds = cpu_seconds
        # Host directory holding the sandbox's Python environment (a venv with
        # matplotlib/numpy/pandas). Bound read-only at PYENV_MOUNT and put
        # first on PATH, so `python3` inside the sandbox is that interpreter.
        # None (or a path that does not exist) means the sandbox has only the
        # host's bare system Python — see core/sandboxenv.py.
        self.python_env = Path(python_env) if python_env else None
        # Optional zero-argument callable returning that directory, resolved
        # by :meth:`prepare`. Keeps ``build_argv`` a pure function of the
        # already-resolved attribute while letting the (slow, network-bound)
        # first build happen lazily instead of at app start.
        self.python_env_provider = python_env_provider
        # bwrap >= 0.11 removed --rlimit-as/--rlimit-cpu (the "resource caps"
        # flags). Detect once per binary so we emit them only when supported.
        self.rlimit_supported = _suggests_rlimit_support(self.bwrap_path)

    def prepare(self) -> None:
        """Resolve the sandbox Python environment, building it if needed."""
        if self.python_env_provider is None:
            return
        resolved = self.python_env_provider()
        self.python_env = Path(resolved) if resolved else None

    def _env_argv(self) -> List[str]:
        """``--clearenv`` plus the curated environment the script actually gets.

        Without ``--clearenv``, bwrap passes the *parent's* entire environment
        through: a sandboxed ``env`` would print the app's LLM API key, the
        image-generation key, the real host username and home directory, and
        the session's IPC socket paths. Namespaces isolate the filesystem and
        the network, never the environment — that has to be dropped
        explicitly.
        """
        path = _BASE_PATH
        if self.python_env is not None:
            path = f"{PYENV_MOUNT}/bin:{path}"
        argv: List[str] = ["--clearenv", "--setenv", "PATH", path]
        for key, value in _BASE_ENV.items():
            argv += ["--setenv", key, value]
        return argv

    def build_argv(self, script: str, workspace: Path) -> List[str]:
        """Build the exact ``bwrap`` argv for the escape-proof sandbox.

        This is the **contract**: the test suite asserts that this list
        contains the right flags in the right order. No subprocess is launched;
        no filesystem is accessed beyond reading the *workspace* path string.

        The resulting command is::

            bwrap \\
              --unshare-all \\
              --die-with-parent \\
              --new-session \\
              --tmpfs / \\
              --ro-bind /usr /usr \\
              [--symlink|--ro-bind for /bin /sbin /lib /lib64 /lib32, if present] \\
              --dev /dev \\
              --proc /proc \\
              --tmpfs /tmp \\
              --bind <workspace> /work \\
              --chdir /work \\
              --unshare-net \\
              --unshare-pid \\
              --unshare-uts \\
              --unshare-cgroup \\
              --cap-drop ALL \\
              --uid 65534 \\
              --gid 65534 \\
              [--rlimit-as <mem_bytes> --rlimit-cpu <cpu_seconds>] \\
              -- /bin/sh -c <script>

        Key safety properties (asserted in tests):
        * ``--unshare-all`` + explicit ``--unshare-net`` → no host network.
        * ``--unshare-pid`` → no host process tree access.
        * ``--cap-drop ALL`` → no capabilities (can't ``sudo``, can't ``mknod``).
        * ``--tmpfs /`` + a curated ``/usr`` (read-only) → a minimal root with
          only what Python/bash/coreutils need. ``/home``, ``/root``, and
          every other host path simply do not exist inside the sandbox — not
          "read-only", genuinely absent. See :func:`_minimal_root_argv`.
        * ``--tmpfs /tmp`` → a private, empty scratch ``/tmp`` (never the
          host's).
        * ``--bind <workspace> /work`` → the ONLY writable path is the
          thread's workspace, mounted at the fixed virtual path ``/work`` —
          NOT its real host path. Binding at the real path (an earlier
          version of this sandbox did) forces bwrap to synthesize the real
          path's entire parent directory chain as mount points inside the
          sandbox (e.g. ``/home/<realuser>/.pengyplexity/workspaces/...``),
          which leaks the real host username and the app's internal data
          layout even though those synthesized directories are empty. A
          fixed ``/work`` mountpoint carries no information about the host.
        * ``--chdir /work`` → the script's CWD is always the workspace.
        * ``--uid 65534 --gid 65534`` → runs as ``nobody``.
        * ``--die-with-parent`` → killed when the parent (Flask) exits.
        * ``--rlimit-as`` / ``--rlimit-cpu`` → resource caps (only when the
          binary supports them; bwrap >= 0.11 removed these flags).
        """
        ws = str(Path(workspace).resolve())
        argv: List[str] = [
            self.bwrap_path,
            # Namespace isolation: unshare everything (user, pid, ipc, uts,
            # cgroup, net). The explicit --unshare-* flags are listed for
            # documentation clarity and to make the test assertions precise.
            "--unshare-all",
            # Kill the child when the parent dies (prevents orphaned sandboxes).
            "--die-with-parent",
            # No controlling terminal (no TTY escape).
            "--new-session",
        ]
        # Minimal root: empty tmpfs at / plus only the system dirs needed to
        # run Python/bash — NOT the real host root (see _minimal_root_argv).
        argv += _minimal_root_argv()
        argv += [
            # Minimal /dev (no host device access beyond what bwrap provides).
            "--dev", "/dev",
            # /proc inside the sandbox (no host process info).
            "--proc", "/proc",
            # Private scratch /tmp — never the host's.
            "--tmpfs", "/tmp",
            # The ONLY writable bind: the thread's workspace, mounted at the
            # fixed virtual path /work — never its real host path (which
            # would leak the real username/data layout; see build_argv's
            # docstring). Our root is a writable-for-mounting tmpfs now, so
            # bwrap can create this mountpoint freely.
            "--bind", ws, "/work",
            # CWD is the workspace — the model's files live here.
            "--chdir", "/work",
            # Explicit namespace drops (redundant with --unshare-all but make
            # the contract visible and testable per-namespace).
            "--unshare-net",
            "--unshare-pid",
            "--unshare-uts",
            "--unshare-cgroup",
            # Drop ALL capabilities: no CAP_SYS_ADMIN, no CAP_NET_ADMIN, etc.
            "--cap-drop", "ALL",
            # Run as nobody (uid 65534, gid 65534).
            "--uid", "65534",
            "--gid", "65534",
        ]
        # The sandbox Python environment (matplotlib/numpy/pandas), read-only
        # at a fixed virtual path. Without it the only interpreter in the
        # sandbox is the host's bare system Python, so every chart script dies
        # on `import matplotlib` — see core/sandboxenv.py.
        if self.python_env is not None:
            argv += ["--ro-bind", str(self.python_env), PYENV_MOUNT]
        # Drop the inherited host environment and set only what the script
        # needs (see _env_argv).
        argv += self._env_argv()
        # Resource caps (memory / CPU-time). bwrap >= 0.11 removed these flags;
        # only emit them when the installed binary supports them. The parent's
        # subprocess wall-clock timeout is the fallback runaway guard.
        if self.rlimit_supported:
            argv += ["--rlimit-as", str(self.mem_bytes)]
            argv += ["--rlimit-cpu", str(self.cpu_seconds)]
        # End of bwrap options; the command to run follows.
        argv += ["--", "/bin/sh", "-c", script]
        return argv

    def run(self, script: str, workspace: Path, cancel=None) -> RunResult:
        """Launch the script in a ``bwrap`` sandbox and wait for it.

        This is the **live** execution path. It is NOT called in the offline
        test suite — tests assert :meth:`build_argv` instead.

        Uses ``Popen`` rather than ``subprocess.run`` so the child can be
        registered on *cancel* and killed the moment the user presses Stop.
        ``--die-with-parent`` only covers the parent exiting, which is not
        what happens when one turn is interrupted.
        """
        argv = self.build_argv(script, workspace)
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # The script must never be able to read the app's stdin.
            stdin=subprocess.DEVNULL,
            # cwd is set inside the sandbox via --chdir; this is the host-side
            # fallback (shouldn't matter since --chdir overrides it).
            cwd=str(workspace),
            # Its own process group, so killing it takes the whole sandbox
            # down rather than just the bwrap parent.
            start_new_session=True,
        )
        if cancel is not None:
            cancel.register_process(proc)
        try:
            try:
                stdout, stderr = proc.communicate(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
                # Reported as a normal non-zero result, not an exception: the
                # exception's message embeds the whole argv, which includes
                # the real host workspace path, and that would be handed
                # straight back to the model as the tool result.
                return RunResult(
                    stdout=_as_text(stdout),
                    stderr=f"Killed: exceeded the {self.timeout}s execution time limit.",
                    returncode=124,
                )
        finally:
            if cancel is not None:
                cancel.unregister_process(proc)

        if cancel is not None and cancel.cancelled:
            return RunResult(
                stdout=_as_text(stdout),
                stderr="Stopped at the user's request.",
                returncode=130,
            )
        return RunResult(
            stdout=_as_text(stdout),
            stderr=_as_text(stderr),
            returncode=proc.returncode,
        )


def _as_text(raw) -> str:
    """Decode partial output captured by a TimeoutExpired (bytes or str)."""
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


# ---------------------------------------------------------------------------
# FakeRunner — offline test double
# ---------------------------------------------------------------------------


@dataclass
class RecordedCall:
    """A single recorded invocation of a :class:`FakeRunner`."""

    script: str
    workspace: Path
    # The turn's CancelToken, when the caller passed one through.
    cancel: object = None


class FakeRunner(Runner):
    """A :class:`Runner` that records calls and returns a canned result.

    The whole test suite uses this instead of a live ``bwrap``. It:

    * records every call (script + workspace) in :attr:`calls`,
    * returns the pre-configured :attr:`result` (or a configurable per-call
      override).

    Parameters
    ----------
    result:
        The canned :class:`RunResult` returned for every call. Defaults to an
        empty successful result.
    """

    def __init__(self, result: RunResult | None = None) -> None:
        self.result = result or RunResult(stdout="", stderr="", returncode=0)
        self.calls: List[RecordedCall] = []

    def run(self, script: str, workspace: Path, cancel=None) -> RunResult:
        self.calls.append(RecordedCall(script=script, workspace=Path(workspace), cancel=cancel))
        return self.result

    @property
    def last_call(self) -> RecordedCall | None:
        """The most recent recorded call, or None if no calls yet."""
        return self.calls[-1] if self.calls else None
