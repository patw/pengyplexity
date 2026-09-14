"""Confinement: ``resolve()`` pins every file path to a per-thread workspace.

This is the *structural* half of the escape-proof contract — the policy half
lives in :mod:`pengyplexity.sandbox.toolpolicy` (which tools the model is ever
handed) and the execution half in
:mod:`pengyplexity.sandbox.executors` (how ``run_python`` / ``run_bash`` are
confined to ``bwrap``). Together they make it impossible, **by construction
rather than by prompt**, for the model to read or write anything outside its
thread's sandbox.

The contract, checked in :mod:`pengyplexity.tests.test_confine`:

* a *relative* path is joined onto the workspace root and returned;
* an *absolute* path is allowed **only** if it already lies inside the root —
  any absolute path outside the workspace is rejected;
* ``..`` / traversal that would land outside the root is rejected;
* a *symlink* whose target points outside the root is rejected (best-effort:
  ``resolve()`` follows symlinks, so a link that points out is caught, and a
  link whose target is missing is caught lexically in stage 1 anyway);
* an empty path means the workspace root itself and is allowed.

The guard is **lexical first** (so it rejects ``../`` and absolute escapes even
for a file that does not yet exist — e.g. the destination of a ``write_file``),
then **realpath-based** (so it also rejects symlinks that point out). Both
stages are best-effort but layered, so the common escape vectors are each
blocked on their own.

A *clear "outside workspace" error* (:class:`OutsideWorkspaceError`) is raised
instead of silently widening the search — the caller decides how to surface it.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath


class OutsideWorkspaceError(ValueError):
    """Raised when a path would resolve outside the per-thread workspace.

    Carries the offending raw path and the workspace root it was confined to so
    the caller can produce a clear, actionable message (never a silent widen).
    """

    def __init__(self, raw_path: str, root: Path | str, reason: str = ""):
        self.raw_path = str(raw_path)
        self.root = Path(root)
        self.reason = reason
        super().__init__(
            f"path escapes workspace {self.root}: {self.raw_path!r}"
            + (f" ({self.reason})" if self.reason else "")
        )


def _reject(raw_path: str, root: Path, reason: str) -> None:
    raise OutsideWorkspaceError(raw_path, root, reason)


def _lexical_within(root: PurePosixPath, candidate: PurePosixPath) -> bool:
    """Return True if *candidate* (already ``os.path.normpath``-ed) is *root* or
    lies beneath it — purely string-based, so it does not touch the filesystem
    and therefore catches ``..`` for files that do not yet exist."""
    if candidate == root:
        return True
    try:
        rel = candidate.relative_to(root)
    except ValueError:
        return False
    # PurePosixPath.relative_to raises on a sibling that merely *shares a name
    # prefix* (e.g. /root-x), so a successful relative_to() already guarantees
    # the candidate is exactly /root or /root/<...>. ``..`` segments were
    # removed by normpath before we got here.
    return rel.parts != () or candidate == root


def resolve(root, raw_path) -> Path:
    """Confine *raw_path* to *root* and return the confined absolute :class:`Path`.

    Parameters
    ----------
    root:
        The per-thread workspace root (a :class:`Path` or ``str``). It is
        normalised but need not exist yet.
    raw_path:
        The model-supplied path — relative to *root* by convention, but an
        absolute path is tolerated **only** if it resolves inside *root*.

    Raises
    ------
    OutsideWorkspaceError
        If the path resolves outside *root* (absolute escape, ``..`` traversal,
        or a symlink pointing out).
    """
    root_p = Path(root)
    raw = "" if raw_path is None else str(raw_path)

    # --- Stage 1: lexical containment (no filesystem access) --------------
    # Path(root) / raw keeps `raw` when it is absolute, otherwise joins onto
    # root. normpath removes "." and collapses ".." *textually* so we can
    # decide containment for a file that does not yet exist (write destinations
    # are exactly this case).
    candidate = (root_p / raw)
    norm = PurePosixPath(os.path.normpath(str(candidate)))
    root_norm = PurePosixPath(os.path.normpath(str(root_p)))
    if not _lexical_within(root_norm, norm):
        _reject(raw, root_p, "resolves outside the workspace")

    # --- Stage 2: resolve symlinks and re-check containment ---------------
    # resolve(strict=False) follows every symlink it can and reports a
    # nonexistent tail as-is, so a link pointing out of the workspace is
    # caught here even when the lexical path (stage 1) looked fine (e.g.
    # root/link -> /etc/passwd, or a nested root/sub/link -> /secret).
    resolved = candidate.resolve(strict=False)
    root_resolved = root_p.resolve(strict=False)
    try:
        resolved.relative_to(root_resolved)
    except ValueError:
        _reject(raw, root_p, "symlink resolves outside the workspace")

    return resolved


def workspace(user, thread, base=None) -> Path:
    """Return the per-thread workspace root: ``<base>/<user>/<thread>``.

    The spec layout is ``~/.pengyplexity/workspaces/<user>/<thread_id>/``; *base*
    defaults to that (``<data_dir>/workspaces``) and is injectable so tests —
    and the app, which passes its configured data dir — never write to ``$HOME``
    by accident.

    The *user* and *thread* components are sanitised to a single path element
    (no separators, no ``.``/``..``), so a hostile id like ``../../etc`` cannot
    walk out of the workspaces tree.
    """
    if base is None:
        base = Path.home() / ".pengyplexity" / "workspaces"
    base = Path(base)
    return (base / _component("user", user) / _component("thread", thread)).resolve(
        strict=False
    )


def _component(kind: str, value) -> str:
    """Reduce *value* to a single safe path component (reject separators/`.`)."""
    name = "" if value is None else str(value).strip()
    # Collapse any separators to a single token and strip dot segments so the
    # result is a leaf filename, never a traversal.
    name = name.replace("/", "_").replace(os.sep, "_") if name else name
    parts = [p for p in PurePosixPath(name).parts if p not in {".", ".."}]
    name = "_".join(parts).strip("._")
    if not name:
        raise OutsideWorkspaceError(value, f"<{kind}>", "empty or traversal-only id")
    return name
