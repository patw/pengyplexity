"""Guard for the top-level README: it exists and documents the safety model.

This is the objective-function check for the "README + full pytest green"
subtask: the deliverable is a self-contained, documented repo, so the docs
must actually be present and cover the safety model (the point of the app),
config, bootstrap, and how to run the offline suite.
"""

from __future__ import annotations

from pathlib import Path

README = Path(__file__).resolve().parent.parent.parent / "README.md"


def _text() -> str:
    assert README.is_file(), f"README.md missing at {README}"
    return README.read_text(encoding="utf-8")


def test_readme_exists():
    assert README.is_file()


def test_readme_covers_safety_model():
    t = _text()
    # The three structural layers of the safety boundary must be named.
    assert "SAFE_TOOLS" in t
    assert "confine" in t.lower() or "confinement" in t.lower()
    assert "bwrap" in t.lower()
    assert "self-modif" in t.lower()  # no self-modification
    # The bwrap argv contract is the escape-proof bit — call it out.
    assert "--cap-drop ALL" in t
    assert "--unshare-net" in t


def test_readme_documents_config_bootstrap_and_run():
    t = _text()
    # Config
    assert "PENGYPLEXITY_MODEL_BASE" in t
    assert "PENGYPLEXITY_DATA_DIR" in t
    # Bootstrap (no self-sign-up — the CLI creates the first admin)
    assert "create-admin" in t
    # How to run the app
    assert "flask --app" in t
    # How to run the offline suite
    assert "pytest" in t
    assert "offline" in t.lower()


def test_readme_states_no_self_signup_and_hard_non_goals():
    t = _text().lower()
    assert "no self sign-up" in t or "no self-sign-up" in t
    assert "docker" in t  # listed as a non-goal
