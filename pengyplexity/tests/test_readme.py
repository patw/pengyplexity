"""Guards for the documentation set.

Three documents, three jobs:

    README.md      — the front door: the pitch, a screenshot, safety at a glance,
                     a quick start, and links out.
    SPEC.md        — the engineering detail: architecture, the full safety model,
                     features, routes, the data model, every setting, non-goals.
    INSTALLING.md  — installing and operating: requirements, first run, the systemd
                     service, nginx + Let's Encrypt, day-to-day operation, and
                     troubleshooting.
    API.md         — the JSON API reference for programs (bots, scripts).

The split is deliberate — a 600-line README buries the pitch, which is the one
thing a front page has to do. So these tests assert the *contract* of that split
rather than any particular wording: the front door stays short and still names
the safety model, and the detail genuinely lives in the two linked documents
instead of having been deleted in the move.

If a future rewrite fails one of these, the fix is to restore the content (or
move it), not to relax the assertion.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
README = ROOT / "README.md"
SPEC = ROOT / "SPEC.md"
INSTALLING = ROOT / "INSTALLING.md"
API = ROOT / "API.md"

# A front-door README that has grown back into a manual has failed at its job.
README_MAX_LINES = 200


def _text(path: Path) -> str:
    assert path.is_file(), f"{path.name} missing at {path}"
    return path.read_text(encoding="utf-8")


# ── The front door ─────────────────────────────────────────────────────────


def test_readme_exists():
    assert README.is_file()


def test_readme_stays_short():
    """The README is the front door, not the manual."""
    lines = _text(README).splitlines()
    assert len(lines) <= README_MAX_LINES, (
        f"README.md is {len(lines)} lines (max {README_MAX_LINES}); detail belongs "
        "in SPEC.md / INSTALLING.md, with the README linking to it"
    )


def test_readme_covers_safety_model():
    """The pitch has to say what actually makes it safe."""
    t = _text(README)
    assert "SAFE_TOOLS" in t
    assert "confine" in t.lower() or "confinement" in t.lower()
    assert "bwrap" in t.lower()
    assert "self-modif" in t.lower()
    assert "--cap-drop ALL" in t
    assert "--unshare-net" in t


def test_readme_has_quick_start_and_links_out():
    """Enough to actually start it, plus a route to the detail."""
    t = _text(README)
    # Config + bootstrap + run
    assert "PENGYPLEXITY_MODEL_BASE" in t
    assert "PENGYPLEXITY_DATA_DIR" in t
    assert "create-admin" in t
    assert "flask --app" in t
    # How to run the offline suite
    assert "pytest" in t
    assert "offline" in t.lower()
    # ...and the two documents the detail moved into
    assert "SPEC.md" in t
    assert "INSTALLING.md" in t


def test_readme_states_no_self_signup_and_no_containers():
    t = _text(README).lower()
    assert "no self sign-up" in t or "no self-sign-up" in t
    assert "docker" in t  # listed as a non-goal


# ── The spec ───────────────────────────────────────────────────────────────


def test_spec_and_installing_exist():
    assert SPEC.is_file(), "SPEC.md missing — the detail has to live somewhere"
    assert INSTALLING.is_file(), "INSTALLING.md missing"


def test_spec_documents_the_safety_model_in_depth():
    t = _text(SPEC)
    assert "SAFE_TOOLS" in t
    assert "bwrap --unshare-all" in t  # the actual sandbox argv
    assert "--clearenv" in t  # environment isolation (keys would leak otherwise)
    assert "OutsideWorkspaceError" in t  # confinement violation type
    assert "FORBIDDEN_TOOL_NAMES" in t  # raw skills never reach the model


def test_spec_documents_configuration_and_data_model():
    t = _text(SPEC)
    for key in (
        "PENGYPLEXITY_DATA_DIR",
        "PENGYPLEXITY_EXEC_TIMEOUT",
        "PENGYPLEXITY_MEMORY_SEMANTIC",
        "PENGYPLEXITY_SANDBOX_VENV",
    ):
        assert key in t, f"SPEC.md no longer documents {key}"
    assert "moofile" in t  # the data model section
    assert "## Non-goals (hard)" in t


# ── Installing & operating ─────────────────────────────────────────────────


def test_installing_covers_install_and_operate():
    t = _text(INSTALLING)
    assert "bubblewrap" in t.lower()
    assert "uv sync" in t
    assert "create-admin" in t
    assert "systemd" in t.lower()
    assert "loginctl enable-linger" in t  # else the unit dies at reboot
    assert "nginx" in t
    assert "proxy_buffering off" in t  # else SSE stops streaming
    assert "certbot" in t
    assert "Troubleshooting" in t


# ── The API reference ──────────────────────────────────────────────────────


def test_api_doc_covers_the_client_contract():
    """What a bot author needs: how to authenticate, what to call, what comes back."""
    t = _text(API)
    assert "Authorization: Bearer" in t
    assert "/api/v1" in t
    assert "POST /api/v1/threads/<id>/messages" in t
    for event in ("token", "done", "artifact", "message"):
        assert f"`{event}`" in t, f"API.md no longer documents the {event} event"
    for code in ("turn_in_progress", "rate_limited", "too_many_concurrent_turns"):
        assert code in t, f"API.md no longer documents {code}"
    assert "PENGYPLEXITY_API_RATE_LIMIT" in t


def test_api_doc_is_linked_both_ways():
    readme, spec, api = _text(README), _text(SPEC), _text(API)
    assert "API.md" in readme and "API.md" in spec
    assert "README.md" in api and "SPEC.md" in api


def test_docs_cross_link():
    """Every doc routes the reader to the other two."""
    readme, spec, installing = _text(README), _text(SPEC), _text(INSTALLING)
    assert "SPEC.md" in readme and "INSTALLING.md" in readme
    assert "README.md" in spec and "INSTALLING.md" in spec
    assert "README.md" in installing and "SPEC.md" in installing
