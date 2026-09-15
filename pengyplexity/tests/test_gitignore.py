"""Guards .gitignore coverage for moofile's store files.

The stores are the sensitive part of the data directory: ``users.bson`` holds
password hashes and ``settings.bson`` can hold the model API key (the admin
settings panel overrides the model connection at runtime). None of them may ever
be committable, whatever a run happens to leave behind.

This asks ``git check-ignore`` about each name in each place a run can leave it
rather than re-implementing gitignore matching, so it exercises the real rules.
It skips when git is unavailable or the checkout is not a git repository (an
unpacked sdist, say) — there is nothing to ignore in that case.

Known residual gap, documented in .gitignore and asserted here as a gap rather
than quietly ignored: a store renamed via ``PENGYPLEXITY_STORE`` to a non-``.bson``
extension (e.g. ``store.db``) is not matched by any pattern.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent

# Every store moofile creates for this app, in both cases (gitignore matching is
# case-sensitive, and a store can arrive from a case-insensitive filesystem).
STORE_NAMES = [
    "users.bson",
    "memories.bson",
    "threads.bson",
    "settings.bson",
    "artifacts.bson",
    "shares.bson",
    "api_keys.bson",
    "pengyplexity.bson",
    "USERS.BSON",
    "Users.Bson",
]

# moofile writes these beside each store.
SIDECAR_SUFFIXES = ["", ".cache", ".lock", ".meta"]

# Where a run can leave them: the repo root, a nested directory, and an in-repo
# data directory (PENGYPLEXITY_DATA_DIR can be pointed anywhere).
LOCATIONS = ["", "data/", ".pengyplexity/", "deep/nested/"]


def _git_repo() -> bool:
    if shutil.which("git") is None:
        return False
    return (
        subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--git-dir"],
            capture_output=True,
        ).returncode
        == 0
    )


pytestmark = pytest.mark.skipif(
    not _git_repo(), reason="needs git and a git checkout to test .gitignore"
)


def _ignored(paths: list[str]) -> set[str]:
    """The subset of *paths* that .gitignore matches (one git call)."""
    proc = subprocess.run(
        ["git", "-C", str(REPO), "check-ignore", *paths],
        capture_output=True,
        text=True,
    )
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def _all_store_paths() -> list[str]:
    return [
        f"{loc}{name}{suffix}"
        for loc in LOCATIONS
        for name in STORE_NAMES
        for suffix in SIDECAR_SUFFIXES
    ]


def test_moofile_stores_are_ignored_everywhere():
    """No store file may be committable, at any depth."""
    paths = _all_store_paths()
    ignored = _ignored(paths)
    leaked = sorted(set(paths) - ignored)
    assert not leaked, (
        "these moofile store files are NOT ignored by .gitignore, so a store "
        "holding password hashes or the model API key could be committed:\n  "
        + "\n  ".join(leaked)
    )


def test_bson_sidecar_and_case_variants_are_ignored():
    """The awkward shapes: chained sidecars and upper/mixed case names."""
    for path in (
        "threads.bson.bson.meta",  # moofile's doubled-suffix meta file
        "artifacts.bson.analytics",
        "backup.bson.bak",
        "deep/USERS.BSON.lock",
    ):
        assert path in _ignored([path]), f"{path} is committable"


def test_env_is_secret_but_env_example_is_tracked():
    """.env is a secret; .env.example is documentation and must stay in the repo."""
    assert ".env" in _ignored([".env"])
    assert "data/.env" in _ignored(["data/.env"])
    assert ".env.example" not in _ignored([".env.example"]), (
        ".env.example must NOT be ignored — it is the documented config template"
    )


def test_pengyplexity_data_dir_is_ignored():
    """The whole in-repo data directory, whatever is in it."""
    assert ".pengyplexity/" in _ignored([".pengyplexity/"])
    assert ".pengyplexity/workspaces/1/2/output.csv" in _ignored(
        [".pengyplexity/workspaces/1/2/output.csv"]
    )
    assert "sandbox-venv/bin/python3" in _ignored(["sandbox-venv/bin/python3"])
