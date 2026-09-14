"""The Python environment the sandbox runs the model's code in.

``run_python`` / ``run_bash`` / ``make_chart`` execute inside a bwrap sandbox
whose only filesystem is an empty tmpfs plus a read-only ``/usr`` (see
:mod:`pengyplexity.sandbox.executors`). That means the interpreter reachable
inside the sandbox is the host's **bare system Python** — which on a typical
host has no matplotlib, no numpy, no pandas. Every chart script the model
writes therefore died on ``import matplotlib`` with the real cause hidden
behind a generic "the script did not produce the file" message.

This module fixes that by building a **separate, throwaway virtualenv on the
host**, once, and binding it into the sandbox read-only at
:data:`~pengyplexity.sandbox.executors.PYENV_MOUNT` (``/pyenv``) with
``/pyenv/bin`` first on ``PATH``. A venv is relocatable in the only way that
matters here: the interpreter derives ``sys.prefix`` from the path it was
invoked by, so a venv built at ``<data_dir>/sandbox-venv`` works unchanged
when it appears as ``/pyenv`` inside the sandbox.

The environment is **not** the app's own venv. It is built from a fixed,
declared package list, is mounted read-only, contains no application code and
no credentials, and holds nothing the model could not have installed itself
given a network — it only exists because the sandbox deliberately has none.

Building needs the network (it downloads wheels), so it happens on the host,
outside the sandbox, and is cached: :meth:`SandboxPythonEnv.ensure` builds at
most once per process and is safe to call from several request threads. Run
``pengyplexity-admin build-sandbox`` to pre-build it at deploy time and keep
the first chart of the day fast.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from pathlib import Path
from typing import List, Optional, Sequence

# What the model's code can import inside the sandbox. Deliberately small and
# chart/data oriented — this is a research assistant's scratchpad, not a
# general-purpose build environment.
DEFAULT_PACKAGES: tuple[str, ...] = (
    "matplotlib",
    "numpy",
    "pandas",
)

# Marker written after a successful build. Its presence means "this directory
# is a complete sandbox env", so an interrupted build is never mistaken for a
# finished one.
_READY_MARKER = ".pengyplexity-sandbox-ready"

# A build downloads and compiles wheels; give it room but never block forever.
BUILD_TIMEOUT = 900

# The venv MUST be built against an interpreter that also exists *inside* the
# sandbox, i.e. one under a directory the sandbox bind-mounts (/usr). Left to
# itself, ``uv venv`` happily picks a uv-managed CPython from
# ~/.local/share/uv/python/... — which is not bound in, so ``/pyenv/bin/python3``
# becomes a broken symlink, PATH falls through to the system interpreter, and
# that one cannot see the venv's site-packages: matplotlib is installed and
# still not importable. Candidates are tried in order.
_INTERPRETER_CANDIDATES = (
    "/usr/bin/python3",
    "/usr/local/bin/python3",
)

# Prefixes an interpreter may live under and still be visible in the sandbox —
# these mirror the read-only binds in sandbox/executors.py.
_SANDBOX_VISIBLE_PREFIXES = ("/usr/", "/bin/", "/sbin/", "/lib/", "/lib64/")


class SandboxEnvError(RuntimeError):
    """Raised when the sandbox Python environment cannot be built."""


class SandboxPythonEnv:
    """Builds (once) and locates the sandbox's Python environment.

    Parameters
    ----------
    path:
        Host directory for the venv (``<data_dir>/sandbox-venv`` by default,
        set from :class:`~pengyplexity.config.Config`).
    packages:
        Distributions to install into it. Defaults to :data:`DEFAULT_PACKAGES`.
    uv:
        The ``uv`` executable used to create the venv and install into it.
    auto_build:
        When True (the default) :meth:`ensure` builds the environment on first
        use if it is missing. Set False to require an explicit
        ``build-sandbox`` run and never pay the build cost inside a request.
    """

    def __init__(
        self,
        path: Path | str,
        packages: Sequence[str] = DEFAULT_PACKAGES,
        uv: str = "uv",
        auto_build: bool = True,
        base_interpreter: Optional[str] = None,
    ) -> None:
        self.path = Path(path)
        self.packages = tuple(packages)
        self.uv = uv
        self.auto_build = auto_build
        self.base_interpreter = base_interpreter or _find_base_interpreter()
        self._lock = threading.Lock()
        # Set once a build in this process has failed, so a broken host (no
        # uv, no network) costs one slow attempt per process rather than one
        # per chart.
        self._failure: Optional[str] = None

    # -- state ------------------------------------------------------------

    @property
    def ready(self) -> bool:
        """True if a completed environment is already on disk."""
        return (self.path / _READY_MARKER).exists() and self.python.exists()

    @property
    def python(self) -> Path:
        """Host path of the environment's interpreter."""
        return self.path / "bin" / "python3"

    # -- build ------------------------------------------------------------

    def ensure(self) -> Optional[Path]:
        """Return the environment path, building it first if necessary.

        Returns ``None`` when the environment is absent and cannot be built —
        the caller falls back to the bare system Python rather than failing
        the whole turn. :attr:`last_error` then explains why.
        """
        if self.ready:
            return self.path
        if not self.auto_build:
            return None
        with self._lock:
            # Another thread may have finished the build while we waited.
            if self.ready:
                return self.path
            if self._failure is not None:
                return None
            try:
                self.build()
            except SandboxEnvError as exc:
                self._failure = str(exc)
                return None
            return self.path

    @property
    def last_error(self) -> Optional[str]:
        """Why the most recent build attempt failed, or None."""
        return self._failure

    def build(self, force: bool = False) -> Path:
        """Create the environment and install :attr:`packages` into it.

        Raises :class:`SandboxEnvError` with an actionable message if ``uv`` is
        missing or a step fails. Safe to re-run: ``force`` rebuilds from
        scratch, otherwise a ready environment is returned untouched.
        """
        if self.ready and not force:
            return self.path
        if shutil.which(self.uv) is None:
            raise SandboxEnvError(
                f"{self.uv!r} was not found on PATH; it is needed to build the "
                "sandbox Python environment."
            )
        # A partial directory from an interrupted build must not be reused.
        if self.path.exists():
            shutil.rmtree(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if self.base_interpreter is None:
            raise SandboxEnvError(
                "no system Python was found under "
                f"{' or '.join(_INTERPRETER_CANDIDATES)}; the sandbox can only "
                "use an interpreter that exists inside it."
            )
        self._run(
            [self.uv, "venv", "--python", self.base_interpreter, str(self.path)],
            "create the sandbox venv",
        )
        self._run(
            [self.uv, "pip", "install", "--python", str(self.python), *self.packages],
            "install the sandbox packages",
        )
        if not self.python.exists():
            raise SandboxEnvError(
                f"the sandbox venv was created but has no interpreter at {self.python}."
            )
        self._verify_visible()
        (self.path / _READY_MARKER).write_text(
            "\n".join(self.packages) + "\n", encoding="utf-8"
        )
        self._failure = None
        return self.path

    def _verify_visible(self) -> None:
        """Fail the build if the venv points at an interpreter the sandbox lacks.

        Catching this here turns a silent, confusing runtime failure (every
        chart dies on ``import matplotlib`` even though it is installed) into
        a clear build-time error.
        """
        try:
            target = self.python.resolve()
        except OSError as exc:
            raise SandboxEnvError(f"could not resolve {self.python}: {exc}") from None
        if not str(target).startswith(_SANDBOX_VISIBLE_PREFIXES):
            raise SandboxEnvError(
                f"the sandbox venv was built against {target}, which is not "
                "visible inside the sandbox (only /usr and its shims are "
                "mounted). Rebuild with --python /usr/bin/python3."
            )

    def _run(self, argv: List[str], what: str) -> None:
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=BUILD_TIMEOUT,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            raise SandboxEnvError(
                f"timed out after {BUILD_TIMEOUT}s trying to {what}."
            ) from None
        except OSError as exc:
            raise SandboxEnvError(f"could not {what}: {exc}") from None
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip().splitlines()
            tail = " ".join(detail[-3:]) if detail else "no output"
            raise SandboxEnvError(f"failed to {what}: {tail}")


def _find_base_interpreter() -> Optional[str]:
    """The first system interpreter that also exists inside the sandbox."""
    for candidate in _INTERPRETER_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None
