"""The curated SAFE_TOOLS allowlist for Pengyplexity, plus the full audit.

This module is the *policy half* of the safety boundary (confinement lives in
:mod:`pengyplexity.sandbox.confine`, sandboxed execution in
:mod:`pengyplexity.sandbox.executors`). Its job is to make it **structurally
impossible** for the chat model to reach a dangerous capability: the frontend
hands the model *only* the tools in :data:`SAFE_TOOLS`, with the ``elevated``/
sudo parameter stripped from ``run_bash`` and the file-tool descriptions edited
to reflect that every path is confined to the per-thread workspace.

Design note (offline rule): this module is **pure data** — it does *not* import
``pengy`` at import time. That keeps the test suite green with no Pengy
installation and no network. ``build_safe_tool_schemas`` accepts an explicit
``pengy_tools`` list (the tests pass a stub) and only falls back to lazily
importing ``pengy.core.tools.TOOLS`` when called with no argument.
"""

from __future__ import annotations

from typing import Iterable

# ---------------------------------------------------------------------------
# The allowlist
# ---------------------------------------------------------------------------
#
# Pengy's ``pengy.core.tools.TOOLS`` holds exactly these 16 tools. Pengyplexity
# does NOT expose a "safer subset of 16": it exposes all 16 *names*, but each is
# **re-scoped** before reaching the model so that none of them can escape:
#   - every *file* tool has its path resolved through confine.resolve() to the
#     per-thread workspace (see confine.py), so "read the whole host" is gone;
#   - ``run_python`` / ``run_bash`` are executed ONLY inside the bwrap sandbox
#     (see executors.py), so "run code on the host" is gone;
#   - ``run_bash``'s ``elevated``/sudo parameter is STRIPPED (see
#     ``scrub_tool_schema``), so "escalate to root" is gone.
#
# What is deliberately NOT in this set are the *skills* the chat could otherwise
# reach — those are capabilities the app provides through dedicated,
# injectable, sandboxed code paths (artifacts / images / sharing), never as raw
# host-executing tools. See :data:`FORBIDDEN_TOOL_NAMES` and ``EXCLUDED``.
SAFE_TOOLS = frozenset(
    {
        # -- read-only (no host side effects once confined) ---------------
        "read_file",
        "read_multiple_files",
        "read_image",
        "directory_tree",
        "search_content",
        "glob",
        # -- the app's job: web as the star ------------------------------
        "web_search",
        "fetch_url",
        "download_file",
        # -- workspace-scoped writers (confined to the thread workspace) --
        "write_file",
        "replace_in_file",
        "apply_changes",
        # -- sandboxed runners (bwrap: host read-only, network drop) ------
        "run_python",
        "run_bash",
        # -- harness / no host access -------------------------------------
        "todowrite",
        "ask_user_question",
        # -- Pengyplexity's own capability wrappers (not raw Pengy skills) --
        # These are NOT the forbidden host-shelling skill names below (that
        # would be "plot"/"image_gen"/"image_edit"). They are the app's own,
        # differently-named tools that call the injectable, workspace-confined
        # core/artifacts + core/images services, so a chart, image, or report
        # the model produces is actually registered as an artifact and can be
        # served/rendered inline instead of being an unreachable claim about a
        # file "in your workspace".
        "make_chart",
        "generate_image",
        "edit_image",
        "create_report",
        # -- private, per-user memory (never the shared bottalk board) -----
        # These read/write ONLY the calling user's own rows in
        # core/memory.MemoryStore — the same rows that user can already edit
        # and delete by hand on the /memories page. This is a different
        # capability from the forbidden raw "bottalk" skill, which posts to a
        # shared, cross-bot, network-visible message bus — nothing here ever
        # leaves the app's local database or crosses between users.
        "save_memory",
        "search_memory",
        "edit_memory",
        "delete_memory",
    }
)

# Tools that are SAFE_TOOLS but whose *description* must be rewritten so the
# model understands that paths are relative to its workspace, not the host.
_CONFINED_FILE_TOOLS = frozenset(
    {
        "read_file",
        "read_multiple_files",
        "read_image",
        "directory_tree",
        "search_content",
        "glob",
        "write_file",
        "replace_in_file",
        "apply_changes",
        "download_file",
    }
)

_WORKSPACE_NOTE = (
    " All file paths are relative to this thread's workspace; absolute paths or "
    "traversal outside the workspace are rejected."
)

# ---------------------------------------------------------------------------
# Per-tool rationale (one line each, required for every allowed tool)
# ---------------------------------------------------------------------------
RATIONALES: dict[str, str] = {
    "read_file": "Read-only; path is confined to the thread workspace via confine.resolve.",
    "read_multiple_files": "Read-only batch read; every path is confined to the workspace.",
    "read_image": "Read-only image view; confined path, reuses Pengy preprocess; no writes.",
    "directory_tree": "Read-only listing rooted in the workspace.",
    "search_content": "Read-only text search rooted in the workspace.",
    "glob": "Read-only file listing rooted in the workspace.",
    "web_search": "The app's job (DuckDuckGo); the only tool with network, no host FS access.",
    "fetch_url": "The app's job; reads a web-page body — network only, no host FS.",
    "download_file": "The app's job; destination dir is resolved under the workspace.",
    "write_file": "Workspace-scoped writer; every path is confined so it can only write inside the sandbox.",
    "replace_in_file": "Workspace-scoped editor; confined path, transactional single-occurrence edit.",
    "apply_changes": "Workspace-scoped bounded editor; all change paths are confined to the workspace.",
    "run_python": "Runs ONLY inside the bwrap sandbox (host read-only, network drop) via BwrapRunner.",
    "run_bash": "Runs ONLY inside the bwrap sandbox; the 'elevated'/sudo parameter is stripped so it can never request root.",
    "todowrite": "Side-effect-free task list; no filesystem or network access.",
    "ask_user_question": "Harness-handled clarification; no host access (never reaches execute_tool).",
    "make_chart": "Runs a chart script ONLY inside the bwrap sandbox via ArtifactService/run_chart_script; output is confined to the workspace and registered as an artifact.",
    "generate_image": "Calls the injectable ImageService (never a raw host tool); output is written only inside the workspace and registered as an artifact.",
    "edit_image": "Calls the injectable ImageService; every input image path is confined via confine.resolve before use, output is written only inside the workspace and registered as an artifact.",
    "create_report": "Calls the injectable ArtifactService (core/artifacts.markdown_to_html/html_to_pdf); output is written only inside the workspace and registered as an artifact — no host access or network.",
    "save_memory": "Writes ONLY to the calling user's own rows in the private MemoryStore; never a network call, never shared with other users.",
    "search_memory": "Reads ONLY the calling user's own rows in the private MemoryStore; never a network call, never shared with other users.",
    "edit_memory": "Edits ONLY the calling user's own rows in the private MemoryStore; the prior value of every changed field is kept in the memory's update_history, so an edit is auditable rather than a silent overwrite.",
    "delete_memory": "Deletes ONLY the calling user's own rows in the private MemoryStore — the same delete the user can already perform on the /memories page; never a network call, never touches another user's rows.",
}

# ---------------------------------------------------------------------------
# Names that must NEVER appear as a tool handed to the model. These are the
# skill / host-capability names the chat could otherwise reach. Asserting the
# allowlist is disjoint from this set is the "no skill-write / no elevated
# tool present" contract checked in CI.
# ---------------------------------------------------------------------------
FORBIDDEN_TOOL_NAMES = frozenset(
    {
        # skills that shell out / hit the host or the network:
        "plot",
        "image_gen",
        "image_edit",
        "clip",
        "pengyshare",
        "screenshot",
        "tts",
        "email",
        "scheduler",
        "sonarr",
        "thewatcher",
        "bottalk",
        # anything that is explicitly a privilege-escalation surface:
        "elevated",
        "sudo",
    }
)

# The audit: a human-readable record of every skill/capability that was
# considered and why it is provided (or not) as a raw tool.
EXCLUDED: dict[str, str] = {
    "plot": "The raw Pengy skill name is excluded — never a raw host tool. Pengyplexity exposes its own 'make_chart' tool instead, which runs matplotlib inside the bwrap sandbox via core/artifacts.",
    "image_gen": "The raw Pengy skill name is excluded — never a raw host tool. Pengyplexity exposes its own 'generate_image' tool instead, which calls an injectable ImageService (core/images) confined to the workspace.",
    "image_edit": "The raw Pengy skill name is excluded — never a raw host tool. Pengyplexity exposes its own 'edit_image' tool instead, which calls an injectable ImageService (core/images) with every input path confined to the workspace.",
    "clip": "tclip (HTML/text) upload — provided through core/sharing behind an injectable uploader.",
    "pengyshare": "Image upload to img.catbee.ca — provided through core/sharing behind an injectable uploader.",
    "screenshot": "Shells out to Playwright/Chromium on the host — excluded; no host process-tree access.",
    "tts": "Shells out to a TTS engine and writes host files — excluded.",
    "email": "Sends email via Gmail SMTP with a stored key — excluded; host network + secret access.",
    "scheduler": "Manages host cron — excluded; host control.",
    "sonarr": "Drives a host media server over its HTTP API — excluded; host network + control.",
    "thewatcher": "Deploys/queries host services over SSH — excluded; host network + control.",
    "bottalk": "Posts to the shared, cross-bot agent memory bus — excluded; outbound network + shared side effects. Pengyplexity's own 'save_memory'/'search_memory' tools are a different, unrelated capability: a private, per-user, local-only memory store (core/memory.py) that never leaves the app or crosses between users.",
    "run_bash.elevated": "The sudo/elevation parameter of run_bash — STRIPPED from the schema so no tool can request root.",
}


def is_safe_tool(name: str) -> bool:
    """Return True if *name* is on the SAFE_TOOLS allowlist."""
    return name in SAFE_TOOLS


# ---------------------------------------------------------------------------
# Generic, user-facing activity labels (the app surfaces these instead of the
# raw tool name, since Pengyplexity is aimed at non-technical users).
# ---------------------------------------------------------------------------
TOOL_ACTIVITY_LABELS: dict[str, str] = {
    # Research / the app's job
    "web_search": "Searching the web…",
    "fetch_url": "Finding more sources…",
    "download_file": "Downloading a document…",
    # File introspection
    "read_file": "Reading a document…",
    "read_multiple_files": "Reading files…",
    "read_image": "Viewing an image…",
    "directory_tree": "Exploring files…",
    "search_content": "Searching through files…",
    "glob": "Finding files…",
    # Workspace-scoped writers
    "write_file": "Creating a document…",
    "replace_in_file": "Updating a document…",
    "apply_changes": "Updating files…",
    # Sandboxed code execution
    "run_python": "Running Python…",
    "run_bash": "Running Bash…",
    # Harness / no host access
    "todowrite": "Organising tasks…",
    "ask_user_question": "Asking a clarifying question…",
    # Pengyplexity's own capability wrappers
    "make_chart": "Creating a chart…",
    "generate_image": "Generating an image…",
    "edit_image": "Editing an image…",
    "create_report": "Writing a report…",
    "save_memory": "Remembering that…",
    "search_memory": "Recalling memories…",
    "edit_memory": "Updating a memory…",
    "delete_memory": "Forgetting that…",
}


def tool_activity_label(name: str) -> str:
    """Return a generic, user-facing label for a tool call.

    Maps a SAFE_TOOLS name to a short, human "what's happening" phrase the
    frontend can show while the agent runs. Unknown tools get a neutral
    fallback so the value is always safe to display.
    """
    return TOOL_ACTIVITY_LABELS.get(name, "Working on it…")



def scrub_tool_schema(schema: dict) -> dict:
    """Return a copy of *schema* safe to hand to the model.

    - Drops the ``elevated`` parameter from ``run_bash`` (and removes any
      sudo/elevation wording from its description).
    - Appends the workspace-confinement note to confined file tools.

    The input is never mutated.
    """
    fn = schema.get("function", {})
    name = fn.get("name")
    if name not in SAFE_TOOLS:
        # Not a tool we ever hand out; return unchanged so callers can detect
        # it is not on the allowlist via build_safe_tool_schemas.
        return schema

    parameters = dict(fn.get("parameters") or {})
    props = dict(parameters.get("properties") or {})

    if name == "run_bash":
        props.pop("elevated", None)
        req = [r for r in (parameters.get("required") or []) if r != "elevated"]
        if "required" in parameters:
            parameters["required"] = req

    if name in _CONFINED_FILE_TOOLS:
        params = dict(props.get("path") or {})
        if "path" in props:
            params["description"] = (params.get("description", "") + _WORKSPACE_NOTE).strip()
            props["path"] = params
        for multi in ("paths",):
            if multi in props:
                params = dict(props[multi])
                params["description"] = (params.get("description", "") + _WORKSPACE_NOTE).strip()
                props[multi] = params

    parameters["properties"] = props

    description = fn.get("description", "")
    if name == "run_bash":
        # Rewrite run_bash's description to remove sudo/elevation wording and
        # state the confinement explicitly.
        description = (
            "Run a command with bash inside this thread's sandbox. The command is "
            "non-interactive: stdin is closed, so anything that prompts or waits for "
            "input (a password prompt, an editor, `read`) will fail — pass "
            "non-interactive flags instead. There is no privilege escalation and no "
            "host network; the working directory is the thread workspace and any "
            "command that tries to reach outside it will be refused. Commands are "
            "killed once the timeout elapses."
        )
    elif name in _CONFINED_FILE_TOOLS and _WORKSPACE_NOTE.strip() not in description:
        description = (description + _WORKSPACE_NOTE).strip()

    import copy

    out = copy.deepcopy(schema)
    out["function"]["description"] = description
    out["function"]["parameters"] = parameters
    return out


def build_safe_tool_schemas(pengy_tools: Iterable[dict] | None = None) -> list[dict]:
    """Return the list of tool schemas to hand to the chat model.

    Filters *pengy_tools* (defaults to lazily importing ``pengy.core.tools.TOOLS``)
    down to :data:`SAFE_TOOLS` and scrubs each schema. Any tool not on the
    allowlist is dropped outright.
    """
    if pengy_tools is None:
        from pengy.core.tools import TOOLS as pengy_tools  # type: ignore  # noqa: WPS433

    out: list[dict] = []
    for schema in pengy_tools:
        name = (schema.get("function") or {}).get("name")
        if name not in SAFE_TOOLS:
            continue
        out.append(scrub_tool_schema(schema))
    return out
