"""Tests for :mod:`pengyplexity.sandbox.confine`.

These assert the *structural* half of the escape-proof contract: every path is
pinned to the per-thread workspace, absolute escapes / ``..`` traversal /
symlink escapes are each rejected with a clear "outside workspace" error, and the
per-thread workspace root helper builds a sane, traversal-proof layout. Fully
offline — everything happens inside ``tmp_path`` (or a pure string check), no
network, no live sandbox, and nothing ever points at ``$HOME``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pengyplexity.sandbox.confine import (
    OutsideWorkspaceError,
    resolve,
    workspace,
)


# ---------------------------------------------------------------------------
# Relative paths resolve inside the root (the allowed, normal case)
# ---------------------------------------------------------------------------


def test_relative_path_resolves_inside_root(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    out = resolve(root, "notes.txt")
    assert out == (root / "notes.txt").resolve()
    assert out.is_relative_to(root)


def test_relative_nested_path_resolves_inside_root(tmp_path):
    root = tmp_path / "ws"
    (root / "sub").mkdir(parents=True)
    out = resolve(root, "sub/deep/file.md")
    assert out == (root / "sub" / "deep" / "file.md").resolve()
    assert out.is_relative_to(root)


def test_dot_segments_collapse_inside_root(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    out = resolve(root, "a/./b/../c.txt")
    assert out == (root / "a" / "c.txt").resolve()
    assert out.is_relative_to(root)


def test_empty_path_means_the_workspace_root(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    out = resolve(root, "")
    assert out == root.resolve()
    # The root itself is legitimately "inside" its own workspace.
    assert out == root.resolve()


# ---------------------------------------------------------------------------
# Absolute escapes are rejected
# ---------------------------------------------------------------------------


def test_absolute_path_outside_root_is_rejected(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    with pytest.raises(OutsideWorkspaceError):
        resolve(root, "/etc/passwd")


def test_absolute_home_escape_is_rejected(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    with pytest.raises(OutsideWorkspaceError):
        resolve(root, str(Path.home() / "Personal"))


def test_absolute_path_inside_root_is_allowed(tmp_path):
    root = tmp_path / "ws"
    (root / "data").mkdir(parents=True)
    inner = root / "data" / "file.txt"
    # An absolute path that is *already* inside the workspace is fine.
    out = resolve(root, str(inner.resolve()))
    assert out == inner.resolve()
    assert out.is_relative_to(root)


def test_sibling_dir_sharing_a_name_prefix_is_rejected(tmp_path):
    # /root and /root-evil share a string prefix but are different trees.
    root = tmp_path / "root"
    sibling = tmp_path / "root-evil"
    root.mkdir()
    sibling.mkdir()
    target = sibling / "leak.txt"
    target.write_text("secret", encoding="utf-8")
    with pytest.raises(OutsideWorkspaceError):
        resolve(root, str(target))


# ---------------------------------------------------------------------------
# ``..`` traversal is rejected — even for files that do not exist yet
# ---------------------------------------------------------------------------


def test_dotdot_to_parent_is_rejected(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    with pytest.raises(OutsideWorkspaceError):
        resolve(root, "../sibling.txt")


def test_dotdot_deep_escape_is_rejected(tmp_path):
    root = tmp_path / "ws" / "nested"
    root.mkdir(parents=True)
    with pytest.raises(OutsideWorkspaceError):
        resolve(root, "../../escape.txt")


def test_dotdot_is_rejected_even_when_target_does_not_exist(tmp_path):
    # write_file destinations don't exist yet — the guard must be lexical and
    # not silently widen just because the file is missing.
    root = tmp_path / "ws"
    root.mkdir()
    # Enough ".." to climb above the workspace root, through dirs that don't
    # exist, so this is a pure lexical (no-filesystem) rejection.
    with pytest.raises(OutsideWorkspaceError):
        resolve(root, "missing/dir/../../../../escape.txt")


def test_internal_dotdot_staying_inside_is_allowed(tmp_path):
    root = tmp_path / "ws"
    (root / "a").mkdir(parents=True)
    # a/../b stays inside the workspace -> legal.
    out = resolve(root, "a/../b.txt")
    assert out == (root / "b.txt").resolve()


# ---------------------------------------------------------------------------
# Symlink escapes are rejected (best-effort, layered on the lexical check)
# ---------------------------------------------------------------------------


def test_symlink_pointing_out_is_rejected(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("top secret", encoding="utf-8")
    link = root / "link.txt"
    os.symlink(outside, link)
    with pytest.raises(OutsideWorkspaceError):
        resolve(root, "link.txt")


def test_nested_symlink_dir_pointing_out_is_rejected(tmp_path):
    root = tmp_path / "ws"
    (root / "sub").mkdir(parents=True)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "data.txt").write_text("hidden", encoding="utf-8")
    link = root / "sub" / "escape"
    os.symlink(outside, link)
    with pytest.raises(OutsideWorkspaceError):
        resolve(root, "sub/escape/data.txt")


def test_symlink_staying_inside_is_allowed(tmp_path):
    root = tmp_path / "ws"
    (root / "real").mkdir(parents=True)
    (root / "real" / "doc.txt").write_text("ok", encoding="utf-8")
    link = root / "alias.txt"
    os.symlink(root / "real" / "doc.txt", link)
    out = resolve(root, "alias.txt")
    assert out == (root / "real" / "doc.txt").resolve()
    assert out.is_relative_to(root)


# ---------------------------------------------------------------------------
# The error is clear and carries context
# ---------------------------------------------------------------------------


def test_error_message_names_the_workspace_and_path(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    with pytest.raises(OutsideWorkspaceError) as excinfo:
        resolve(root, "../nope.txt")
    err = excinfo.value
    assert str(root) in str(err)
    assert "../nope.txt" in str(err)
    assert err.root == root
    assert err.raw_path == "../nope.txt"
    # It is a ValueError so callers can catch the broad type too.
    assert isinstance(err, ValueError)


# ---------------------------------------------------------------------------
# The per-thread workspace root helper
# ---------------------------------------------------------------------------


def test_workspace_layout_matches_spec(tmp_path):
    base = tmp_path / "data" / "workspaces"
    ws = workspace("alice", "t123", base=base)
    assert ws == (base / "alice" / "t123").resolve()
    assert ws.is_relative_to(base)


def test_workspace_is_unique_per_user_and_thread(tmp_path):
    base = tmp_path / "workspaces"
    a = workspace("alice", "t1", base=base)
    b = workspace("alice", "t2", base=base)
    c = workspace("bob", "t1", base=base)
    assert a != b
    assert a != c
    assert b.is_relative_to(base)
    assert c.is_relative_to(base)


def test_workspace_sanitises_separators_in_ids(tmp_path):
    base = tmp_path / "workspaces"
    ws = workspace("a/b", "t/../x", base=base)
    # Traversal collapses into a single leaf; it cannot walk out of `base`.
    assert ws.is_relative_to(base)
    assert ".." not in ws.parts
    # The returned root, when used as a confinement root, stays inside base.
    inner = resolve(ws, "f.txt")
    assert inner.is_relative_to(base)


def test_workspace_rejects_empty_or_pure_traversal_ids(tmp_path):
    base = tmp_path / "workspaces"
    with pytest.raises(OutsideWorkspaceError):
        workspace("", "t1", base=base)
    with pytest.raises(OutsideWorkspaceError):
        workspace("alice", "../..", base=base)


def test_workspace_with_injected_base_never_touches_home(tmp_path, monkeypatch):
    # Even if HOME points somewhere weird, an injected base wins.
    monkeypatch.setenv("HOME", str(tmp_path / "fakehome"))
    base = tmp_path / "data" / "workspaces"
    ws = workspace("alice", "t1", base=base)
    assert ws.is_relative_to(base)
    # And it must NOT be under the (fake) HOME.
    assert not ws.is_relative_to(tmp_path / "fakehome")


if __name__ == "__main__":  # pragma: no cover - manual smoke
    raise SystemExit(pytest.main([__file__, "-q"]))
