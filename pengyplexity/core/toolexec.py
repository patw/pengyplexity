"""Production tool executor for the Pengyplexity agent.

This ties together the three halves of the safety boundary — the curated
``SAFE_TOOLS`` allowlist (:mod:`pengyplexity.sandbox.toolpolicy`), path
confinement (:mod:`pengyplexity.sandbox.confine`), and sandboxed execution
(:mod:`pengyplexity.sandbox.executors`) — plus the app's own web-search /
fetch service (:mod:`pengyplexity.core.search`).

It exposes :func:`build_tool_executor`, which returns a callable with the
signature the agent expects::

    execute(name: str, args: dict, workspace: Path) -> str

Every file path is resolved through ``confine.resolve(root, raw_path)`` so a
malicious path (``../``, absolute, symlink-escape) is rejected *before* any
filesystem access. Every ``run_python`` / ``run_bash`` is dispatched to a
:class:`Runner` (the real ``BwrapRunner`` in production) so the code runs in a
host-read-only, network-dropped, non-root, capability-dropped sandbox.

The executor is **self-contained**: it does not import ``pengy``. It carries its
own OpenAPI schemas for the 16 ``SAFE_TOOLS`` (so the app does not depend on
Pengy's tool inventory at runtime), building them through
``toolpolicy.build_safe_tool_schemas`` so the scrubbing (``elevated`` stripping,
workspace note) still applies.
"""

from __future__ import annotations

import glob as _glob
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..sandbox import confine
from ..sandbox.executors import Runner
from ..sandbox.toolpolicy import SAFE_TOOLS, build_safe_tool_schemas
from .artifacts import ArtifactService
from .images import ImageService
from .memory import MemoryStore
from .search import SearchService
from .vision import PendingImages, queue_image

# Signature the Agent's tool_executor callback must have.
ToolExecutor = Callable[[str, Dict, Path], str]

# Hard ceiling on `web_search`'s `max_results`, enforced server-side
# regardless of what the model requests (the tool schema hints at this too,
# but hints aren't enforced by every backend) — more than a handful of
# results per search is noise, not signal, for a single-turn answer.
WEB_SEARCH_MAX_RESULTS = 5

# Never listed by directory_tree — bookkeeping the app itself puts in the
# workspace, not something the model wrote or needs to see.
_SKIP_DIRS = frozenset({".pengyplexity", "__pycache__", ".cache"})

# Tool names that are Pengyplexity's own capability wrappers, not part of
# Pengy's real tool inventory — always included in the schemas handed to the
# model regardless of whether ``pengy`` happens to be importable.
_APP_ONLY_TOOL_NAMES = frozenset(
    {"make_chart", "generate_image", "edit_image", "create_report", "save_memory", "search_memory"}
)


@dataclass
class ToolContext:
    """Per-turn state the artifact- and memory-producing tools read/write.

    The production executor is built once and shared by the app's single
    ``Agent`` instance (mirroring how ``agent.workspace`` is already
    repointed per request in ``web.py``). Before each turn, the caller sets
    ``thread_id``/``message_index``/``owner`` so ``make_chart``/
    ``generate_image``/``edit_image`` record artifacts against the right
    thread and message, and ``save_memory``/``search_memory`` read/write only
    the calling user's own rows; after the turn, ``artifacts`` holds every
    :class:`~pengyplexity.core.artifacts.ArtifactRecord` produced so the
    caller (e.g. the SSE stream) can surface them immediately instead of
    waiting for a full page reload.
    """

    thread_id: str = ""
    message_index: int = 0
    owner: str = ""
    artifacts: List[Any] = field(default_factory=list)
    # Images produced this turn (by read_image, make_chart, generate_image,
    # edit_image), waiting to be attached to the conversation as a follow-up
    # user message so a vision model can actually look at them. A tool result
    # is a plain string and cannot carry a picture — see core/vision.py.
    pending_images: PendingImages = field(default_factory=PendingImages)
    # This turn's cancel token (see core/cancel.py), set per request. Passed
    # to the sandbox runner so Stop kills a running script immediately, and
    # checked before each tool so an interrupted turn does not keep working
    # through the tool calls the model already asked for.
    cancel: Any = None
    # Admin-configurable limits (see core/settings.py), refreshed each turn by
    # web.py from the effective settings (store override, else Config default):
    tool_output_max_chars: int = 0
    download_max_mb: float = 0
    tool_network_timeout: int = 15
    user_agent: str = "Mozilla/5.0 (Pengyplexity)"


# ---------------------------------------------------------------------------
# Self-contained SAFE_TOOLS schemas (OpenAPI function-calling shape).
# Used only as the fallback when ``pengy`` is not importable; ``build_safe_tool_
# schemas`` will scrub them (drop ``elevated``, append the workspace note).
# ---------------------------------------------------------------------------


def _param(name: str, typ: str, desc: str, extra: Optional[dict] = None) -> dict:
    p = {"type": typ, "description": desc}
    if extra:
        p.update(extra)
    return {name: p}


def _schema(name: str, description: str, properties: dict, required: List[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def _builtin_tool_schemas() -> List[dict]:
    """Return the OpenAPI schemas for the 16 SAFE_TOOLS, self-contained."""
    return [
        _schema(
            "read_file",
            "Read text from a file relative to the thread workspace.",
            {**_param("path", "string", "path to read (relative to workspace)"),
             **_param("limit", "integer", "max lines", {"default": None}),
             **_param("offset", "integer", "line offset", {"default": None})},
            ["path"],
        ),
        _schema(
            "read_multiple_files",
            "Read several files at once.",
            {**_param("paths", "array", "list of paths (relative to workspace)")},
            ["paths"],
        ),
        _schema(
            "read_image",
            "Look at an image file. The picture is attached to the conversation "
            "and you can see it directly — use this before describing any image.",
            {**_param("path", "string", "image path (relative to workspace)")},
            ["path"],
        ),
        _schema(
            "directory_tree",
            "List files in the workspace as a tree.",
            {**_param("path", "string", "root dir (default workspace)", {"default": ""}),
             **_param("max_depth", "integer", "max depth", {"default": 3})},
            [],
        ),
        _schema(
            "search_content",
            "Search text in files under the workspace.",
            {**_param("pattern", "string", "text to match"),
             **_param("path", "string", "search root (default workspace)", {"default": ""}),
             **_param("file_glob", "string", "glob to filter files", {"default": ""}),
             **_param("context_lines", "integer", "context lines", {"default": 0})},
            ["pattern"],
        ),
        _schema(
            "glob",
            "Find files matching a glob under the workspace.",
            {**_param("pattern", "string", "glob pattern (relative to workspace)")},
            ["pattern"],
        ),
        _schema(
            "write_file",
            "Write text to a file (path confined to the workspace).",
            {**_param("path", "string", "destination path (relative)"),
             **_param("content", "string", "file content"),
             **_param("overwrite", "boolean", "overwrite if exists", {"default": False})},
            ["path", "content"],
        ),
        _schema(
            "replace_in_file",
            "Replace one exact occurrence of text in a file.",
            {**_param("path", "string", "file path (relative)"),
             **_param("old_str", "string", "exact text to find"),
             **_param("new_str", "string", "replacement text")},
            ["path", "old_str", "new_str"],
        ),
        _schema(
            "apply_changes",
            "Apply a set of bounded edits to files under the workspace.",
            {**_param("changes", "array", "list of {path, old, new} edit dicts")},
            ["changes"],
        ),
        _schema(
            "web_search",
            "Search the web and return results.",
            {**_param("query", "string", "search query"),
             **_param("max_results", "integer", "max results (capped at 5)",
                       {"default": 5, "maximum": 5})},
            ["query"],
        ),
        _schema(
            "fetch_url",
            "Fetch a web page and return its text.",
            {**_param("url", "string", "URL to fetch"),
             **_param("max_chars", "integer", "max chars", {"default": None})},
            ["url"],
        ),
        _schema(
            "download_file",
            "Download a URL into the workspace.",
            {**_param("url", "string", "URL to download"),
             **_param("dir", "string", "destination dir (relative)", {"default": ""}),
             **_param("filename", "string", "output filename", {"default": None})},
            ["url"],
        ),
        _schema(
            "run_python",
            "Run Python code inside the sandboxed workspace. matplotlib, numpy, pandas and the "
            "standard library are available; there is no network.",
            {**_param("code", "string", "python source to execute")},
            ["code"],
        ),
        _schema(
            "run_bash",
            "Run a shell command inside the sandboxed workspace (no sudo/network).",
            {**_param("command", "string", "command to run")},
            ["command"],
        ),
        _schema(
            "todowrite",
            "Maintain a task list (no filesystem/network side effects).",
            {**_param("todos", "array", "list of task items")},
            ["todos"],
        ),
        _schema(
            "ask_user_question",
            "Ask the user a clarifying question.",
            {**_param("question", "string", "the question")},
            ["question"],
        ),
        _schema(
            "make_chart",
            "Generate a chart image by running a Python script inside the sandbox, which has "
            "matplotlib, numpy and pandas but no network. The script must save the image to the "
            "given filename in the current directory (e.g. plt.savefig('chart.png')). The finished "
            "chart is shown to the user inline and attached so you can check it yourself.",
            {**_param("script", "string", "Python source that produces and saves the chart image"),
             **_param("filename", "string", "output PNG filename (relative to workspace)", {"default": "chart.png"})},
            ["script"],
        ),
        _schema(
            "generate_image",
            "Generate a new image from a text prompt. The result is saved as a PNG and shown to the user inline.",
            {**_param("prompt", "string", "description of the image to generate"),
             **_param("filename", "string", "output PNG filename (relative to workspace)", {"default": None}),
             **_param("aspect_ratio", "string", "optional aspect ratio, e.g. '16:9'", {"default": None}),
             **_param("resolution", "string", "optional resolution, e.g. '1024x1024'", {"default": None})},
            ["prompt"],
        ),
        _schema(
            "edit_image",
            "Edit one or more existing images already in the workspace per a text instruction. "
            "The result is saved as a PNG and shown to the user inline.",
            {**_param("prompt", "string", "editing instruction"),
             **_param("input_images", "array", "workspace-relative paths to the input image(s)"),
             **_param("filename", "string", "output PNG filename (relative to workspace)", {"default": None})},
            ["prompt", "input_images"],
        ),
        _schema(
            "create_report",
            "Generate a downloadable report from Markdown text: an HTML file plus a PDF. "
            "Use this when the user asks for a document, write-up, or PDF to download — "
            "not for the normal chat answer, which is already rendered as Markdown inline.",
            {**_param("markdown", "string", "the report body, as Markdown"),
             **_param("title", "string", "report title (used for the filename and heading)", {"default": "Report"})},
            ["markdown"],
        ),
        _schema(
            "save_memory",
            "Save a durable memory about the user or the conversation, so it can be recalled in "
            "future conversations. Use this for facts worth remembering long-term (preferences, "
            "ongoing projects, recurring context) — not for one-off details only relevant to this turn.",
            {**_param("title", "string", "short title for the memory"),
             **_param("summary", "string", "1-2 sentence summary (this is what search matches on)"),
             **_param("body", "string", "optional longer detail", {"default": None}),
             **_param("tags", "array", "optional list of short tags to group related memories", {"default": None})},
            ["title", "summary"],
        ),
        _schema(
            "search_memory",
            "Search the user's saved memories (title/summary/body, and semantically) to recall "
            "relevant facts from past conversations before answering. Only memories that are "
            "actually relevant are returned; an empty result means nothing is saved on the "
            "topic, which is a real answer — say so rather than guessing.",
            {**_param("query", "string", "what to search for"),
             **_param("limit", "integer", "max results", {"default": 5}),
             **_param("min_signal", "number",
                      "relevance floor from 0 to 1 (default 0.45). Pass 0 to see near "
                      "misses when you want to confirm whether anything related exists at all.",
                      {"default": None})},
            ["query"],
        ),
    ]


def get_tool_schemas() -> List[dict]:
    """Return the scrubbed SAFE_TOOLS schemas.

    If ``pengy`` is importable, derive from ``pengy.core.tools.TOOLS`` (matching
    the original design); otherwise fall back to the self-contained set above.
    In both cases the schemas are scrubbed via ``toolpolicy.build_safe_tool_schemas``.
    """
    try:
        from pengy.core.tools import TOOLS as pengy_tools  # type: ignore  # noqa: WPS433
    except ImportError:
        pengy_tools = _builtin_tool_schemas()
    else:
        # Pengy's own inventory has no knowledge of Pengyplexity's capability
        # wrappers (they aren't raw Pengy tools) — append them so they're
        # always offered to the model regardless of whether Pengy is present.
        pengy_tools = list(pengy_tools) + [
            s for s in _builtin_tool_schemas()
            if s["function"]["name"] in _APP_ONLY_TOOL_NAMES
        ]
    return build_safe_tool_schemas(pengy_tools)


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


def _resolve(root: Path, raw: str) -> Path:
    """Resolve *raw* under *root* via confine (raises OutsideWorkspaceError)."""
    return confine.resolve(root, raw)


def _rel(root: Path, path: Path) -> str:
    """Render *path* as workspace-relative, the way the model addresses it.

    Tool output must never contain the real host path of the workspace: it
    embeds the OS user the app runs as and the app's internal data layout,
    and the model then echoes it into answers and into later tool arguments.
    """
    try:
        rel = Path(path).resolve().relative_to(Path(root).resolve())
    except (ValueError, OSError):
        return Path(path).name
    return rel.as_posix() or "."


def _read_file(root: Path, args: Dict) -> str:
    p = _resolve(root, str(args.get("path", "")))
    if not p.exists() or not p.is_file():
        return f"File not found: {args.get('path')}"
    data = p.read_text(errors="replace")
    offset = args.get("offset")
    limit = args.get("limit")
    if offset or limit:
        lines = data.splitlines()
        start = int(offset) if offset else 0
        end = (start + int(limit)) if limit else len(lines)
        data = "\n".join(lines[start:end])
    return data


def _read_multiple(root: Path, args: Dict) -> str:
    out = []
    for raw in args.get("paths") or []:
        p = _resolve(root, str(raw))
        if p.exists() and p.is_file():
            out.append(f"### {raw}\n{p.read_text(errors='replace')}")
        else:
            out.append(f"### {raw}\n[missing]")
    return "\n\n".join(out)


def _read_image(root: Path, context: "ToolContext", args: Dict) -> str:
    """Load an image and attach it to the conversation for a vision model.

    The picture cannot ride in the return value (a ``role: "tool"`` message
    takes string content only), so it is queued on the context and the agent
    loop attaches it as a follow-up user message — Pengy's mechanism, see
    core/vision.py.
    """
    raw = str(args.get("path", ""))
    p = _resolve(root, raw)
    if not p.exists() or not p.is_file():
        return f"Image not found: {raw}"
    return queue_image(context.pending_images, p, _rel(root, p))


def _dir_tree(root: Path, args: Dict) -> str:
    base = _resolve(root, str(args.get("path", "") or "."))
    max_depth = int(args.get("max_depth", 3) or 3)
    lines = []
    if not base.exists():
        return f"Directory not found: {args.get('path')}"
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        rel = Path(dirpath).relative_to(base)
        depth = 0 if str(rel) == "." else len(rel.parts)
        if depth > max_depth:
            dirnames[:] = []
            continue
        indent = "  " * depth
        lines.append(f"{indent}{rel if str(rel) != '.' else '.'}/")
        for fn in filenames:
            lines.append(f"{indent}  {fn}")
    return "\n".join(lines)


def _search_content(root: Path, args: Dict) -> str:
    pattern = str(args.get("pattern", ""))
    search_root = _resolve(root, str(args.get("path", "") or "."))
    file_glob = str(args.get("file_glob", "")) or "*"
    context = int(args.get("context_lines", 0) or 0)
    if not search_root.exists():
        return f"Path not found: {args.get('path')}"
    out = []
    for fp in search_root.rglob(file_glob):
        if not fp.is_file():
            continue
        try:
            text = fp.read_text(errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines()):
            if pattern in line:
                # Workspace-relative, never the real host path (see _rel).
                out.append(f"{_rel(root, fp)}:{i+1}: {line}")
    return "\n".join(out) or "No matches."


def _glob(root: Path, args: Dict) -> str:
    pattern = str(args.get("pattern", ""))
    full = os.path.join(str(root), pattern)
    matches = sorted(_glob.glob(full, recursive=True))
    # only keep matches that resolve inside the workspace (best-effort)
    kept = []
    for m in matches:
        try:
            confine.resolve(root, Path(m).relative_to(root).as_posix())
            kept.append(_rel(root, Path(m)))
        except Exception:
            continue
    return "\n".join(kept) or "No matches."


def _write_file(root: Path, args: Dict) -> str:
    p = _resolve(root, str(args.get("path", "")))
    content = str(args.get("content", ""))
    overwrite = bool(args.get("overwrite", False))
    if p.exists() and not overwrite:
        return f"File exists (use overwrite=true): {args.get('path')}"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"Wrote {_rel(root, p)} ({len(content)} chars)."


def _replace_file(root: Path, args: Dict) -> str:
    p = _resolve(root, str(args.get("path", "")))
    if not p.exists() or not p.is_file():
        return f"File not found: {args.get('path')}"
    old = str(args.get("old_str", ""))
    new = str(args.get("new_str", ""))
    text = p.read_text(errors="replace")
    if text.count(old) != 1:
        return f"Expected exactly 1 match, found {text.count(old)}."
    p.write_text(text.replace(old, new))
    return f"Replaced in {_rel(root, p)}."


def _apply_changes(root: Path, args: Dict) -> str:
    changes = args.get("changes") or []
    msgs = []
    for c in changes:
        if not isinstance(c, dict):
            continue
        p = _resolve(root, str(c.get("path", "")))
        if not p.exists() or not p.is_file():
            msgs.append(f"[skip] {c.get('path')}: missing")
            continue
        text = p.read_text(errors="replace")
        old = str(c.get("old", c.get("old_str", "")))
        new = str(c.get("new", c.get("new_str", "")))
        if text.count(old) != 1:
            msgs.append(f"[skip] {c.get('path')}: {text.count(old)} matches")
            continue
        p.write_text(text.replace(old, new))
        msgs.append(f"[ok] {c.get('path')}")
    return "\n".join(msgs)


# Scratch space inside the workspace for files the app writes on the model's
# behalf (the body of run_python, a chart script). Kept in its own dot-dir so
# it never collides with, or hides, the model's own files, and skipped by
# directory_tree.
SCRATCH_DIR = ".pengyplexity"


def scratch_path(root: Path, name: str) -> Path:
    """Return a unique path under the workspace's scratch dir, creating it."""
    import uuid

    scratch = Path(root) / SCRATCH_DIR
    scratch.mkdir(parents=True, exist_ok=True)
    stem, _, suffix = name.rpartition(".")
    return scratch / f"{stem or name}-{uuid.uuid4().hex[:8]}.{suffix or 'tmp'}"


def _decode(text: str) -> str:
    """Decode bytes from a download into text (best-effort)."""
    return text.decode("utf-8", errors="replace")


def _run_python(root: Path, runner: Runner, args: Dict, cancel=None) -> str:
    code = str(args.get("code", ""))
    # A private scratch dir, not `script.py` in the workspace root: that name
    # clobbered any file the user or the model already had there, and the
    # unlink afterwards deleted it.
    script = scratch_path(root, "run.py")
    script.write_text(code, encoding="utf-8")
    try:
        runner.prepare()
        res = runner.run(f"python3 {SCRATCH_DIR}/{script.name}", root, cancel=cancel)
    finally:
        script.unlink(missing_ok=True)
    return _format_run(res.stdout, res.stderr, res.returncode)


def _run_bash(root: Path, runner: Runner, args: Dict, cancel=None) -> str:
    command = str(args.get("command", ""))
    runner.prepare()
    res = runner.run(command, root, cancel=cancel)
    return _format_run(res.stdout, res.stderr, res.returncode)


def _format_run(stdout: str, stderr: str, rc: int) -> str:
    parts = []
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(f"[stderr]\n{stderr}")
    parts.append(f"[exit {rc}]")
    return "\n".join(parts)


def _make_chart(
    root: Path,
    artifact_service: Optional[ArtifactService],
    context: ToolContext,
    args: Dict,
) -> str:
    if artifact_service is None:
        return "Chart capability is not configured."
    script = str(args.get("script", ""))
    filename = str(args.get("filename") or "chart.png")
    record = artifact_service.create_chart(
        workspace=root,
        script=script,
        thread_id=context.thread_id,
        message_index=context.message_index,
        output_filename=filename,
        cancel=context.cancel,
    )
    if record.size_bytes <= 0:
        # Hand back what the interpreter actually said. The previous message
        # ("check the script for errors") hid the real cause — most often a
        # missing library in the sandbox — so the model rewrote a correct
        # script over and over, or gave up and described a chart it never
        # made.
        detail = _run_detail(record)
        return (
            f"The script ran but did not produce '{filename}'.\n{detail}\n"
            "Fix the script and call make_chart again. Note: the sandbox has "
            "matplotlib, numpy and pandas, no network, and the file must be "
            "saved to the current directory under exactly this filename."
        )
    context.artifacts.append(record)
    # Attach the rendered chart so a vision model can check it came out right
    # (axes labelled, nothing clipped, the data plotted as intended) instead
    # of reporting success on a file it has never seen.
    _attach_artifact_image(context, root, record, "Chart just produced")
    return f"Created chart '{record.filename}' ({record.size_bytes} bytes)."


def _run_detail(record) -> str:
    """Format the sandbox run's output for a failed artifact, or a fallback."""
    result = getattr(record, "run_result", None)
    if result is None:
        return "The sandbox produced no output."
    parts = []
    if result.stdout.strip():
        parts.append(f"[stdout]\n{result.stdout.strip()}")
    if result.stderr.strip():
        parts.append(f"[stderr]\n{result.stderr.strip()}")
    parts.append(f"[exit {result.returncode}]")
    return "\n".join(parts)


def _attach_artifact_image(context: "ToolContext", root: Path, record, note: str) -> str:
    """Queue a just-produced image artifact for the vision model to look at."""
    path = Path(record.path)
    if not path.exists() or not path.is_file():
        return ""
    return queue_image(
        context.pending_images, path, _rel(root, path),
        note=f"{note}: {record.filename}",
    )


def _create_report(
    root: Path,
    artifact_service: Optional[ArtifactService],
    context: ToolContext,
    args: Dict,
) -> str:
    if artifact_service is None:
        return "Report capability is not configured."
    markdown_text = str(args.get("markdown", ""))
    if not markdown_text.strip():
        return "The 'markdown' argument is empty — nothing to report."
    title = str(args.get("title") or "Report")
    records = artifact_service.create_report(
        workspace=root,
        markdown_text=markdown_text,
        thread_id=context.thread_id,
        title=title,
        message_index=context.message_index,
    )
    context.artifacts.extend(records.values())
    names = ", ".join(f"'{r.filename}'" for r in records.values())
    return f"Created report: {names}."


def _generate_image(
    root: Path,
    image_service: Optional[ImageService],
    context: ToolContext,
    args: Dict,
) -> str:
    if image_service is None:
        return "Image generation is not configured."
    prompt = str(args.get("prompt", ""))
    try:
        record = image_service.create_generate(
            workspace=root,
            prompt=prompt,
            thread_id=context.thread_id,
            message_index=context.message_index,
            filename=args.get("filename") or None,
            aspect_ratio=args.get("aspect_ratio") or None,
            resolution=args.get("resolution") or None,
        )
    except Exception as e:  # noqa: BLE001
        return f"Image generation failed: {e}"
    if record.size_bytes <= 0:
        return (
            f"The image backend returned no data for '{record.filename}'. "
            "It may be misconfigured — report this rather than retrying."
        )
    context.artifacts.append(record)
    _attach_artifact_image(context, root, record, "Image just generated")
    return f"Generated image '{record.filename}' ({record.size_bytes} bytes)."


def _edit_image(
    root: Path,
    image_service: Optional[ImageService],
    context: ToolContext,
    args: Dict,
) -> str:
    if image_service is None:
        return "Image editing is not configured."
    prompt = str(args.get("prompt", ""))
    raw_inputs = args.get("input_images") or []
    resolved = []
    for raw in raw_inputs:
        try:
            p = _resolve(root, str(raw))
        except confine.OutsideWorkspaceError as e:
            return f"Blocked: {e}"
        if not p.exists() or not p.is_file():
            return f"Input image not found: {raw}"
        resolved.append(p)
    if not resolved:
        return "No input images given."
    try:
        record = image_service.create_edit(
            workspace=root,
            prompt=prompt,
            input_images=resolved,
            thread_id=context.thread_id,
            message_index=context.message_index,
            filename=args.get("filename") or None,
        )
    except Exception as e:  # noqa: BLE001
        return f"Image edit failed: {e}"
    if record.size_bytes <= 0:
        return (
            f"The image backend returned no data for '{record.filename}'. "
            "It may be misconfigured — report this rather than retrying."
        )
    context.artifacts.append(record)
    _attach_artifact_image(context, root, record, "Edited image")
    return f"Edited image saved as '{record.filename}' ({record.size_bytes} bytes)."


def _save_memory(
    memory_store: Optional[MemoryStore],
    context: ToolContext,
    args: Dict,
) -> str:
    if memory_store is None:
        return "Memory capability is not configured."
    if not context.owner:
        return "Memory capability requires a logged-in user context."
    title = str(args.get("title", "")).strip()
    summary = str(args.get("summary", "")).strip()
    if not title or not summary:
        return "Both 'title' and 'summary' are required to save a memory."
    body = str(args.get("body") or "")
    tags = args.get("tags") or []
    doc = memory_store.create(owner=context.owner, title=title, summary=summary, tags=tags, body=body)
    return f"Saved memory '{doc['title']}' (id={doc['_id']})."


def _search_memory(
    memory_store: Optional[MemoryStore],
    context: ToolContext,
    args: Dict,
) -> str:
    if memory_store is None:
        return "Memory capability is not configured."
    if not context.owner:
        return "Memory capability requires a logged-in user context."
    query = str(args.get("query", "")).strip()
    if not query:
        return "A 'query' is required."
    try:
        limit = int(args.get("limit") or 5)
    except (TypeError, ValueError):
        limit = 5
    try:
        min_signal = args.get("min_signal")
        min_signal = float(min_signal) if min_signal is not None else None
    except (TypeError, ValueError):
        min_signal = None

    found = memory_store.search(
        context.owner, query, limit=limit, statuses=("active",), min_signal=min_signal
    )
    if not found.results:
        # The advisory carries the denominators, so the model can say "nothing
        # is saved about this" instead of treating an empty result as a
        # failed lookup and guessing.
        return found.advisory or "No matching memories found."

    lines = []
    for hit in found.results:
        doc = hit.doc
        tags = ", ".join(doc.get("tags") or [])
        line = f"- [{doc['_id']}] {doc['title']}: {doc['summary']}"
        if tags:
            line += f" (tags: {tags})"
        if hit.signal is not None:
            line += f" [match {hit.signal:.2f}, {hit.confidence}]"
        lines.append(line)
    out = "\n".join(lines)
    if found.filtered:
        out += (
            f"\n\n({found.surfaced} of {found.examined} candidates cleared the "
            f"relevance floor, from a {found.corpus}-memory corpus.)"
        )
    if found.advisory:
        out += f"\n\n{found.advisory}"
    return out


# ---------------------------------------------------------------------------
# The executor factory
# ---------------------------------------------------------------------------


def build_tool_executor(
    runner: Runner,
    search: SearchService,
    artifact_service: Optional[ArtifactService] = None,
    image_service: Optional[ImageService] = None,
    memory_store: Optional[MemoryStore] = None,
) -> ToolExecutor:
    """Return the production tool-executor callable.

    Parameters
    ----------
    runner:
        The sandboxed code runner (``BwrapRunner`` in production).
    search:
        The web search + fetch service (``DDGSSearchService`` in production).
    artifact_service:
        Optional :class:`ArtifactService` backing ``make_chart``. If omitted,
        that tool returns an explanatory error instead of failing.
    image_service:
        Optional :class:`ImageService` backing ``generate_image``/
        ``edit_image``. If omitted, those tools return an explanatory error.
    memory_store:
        Optional :class:`~pengyplexity.core.memory.MemoryStore` backing
        ``save_memory``/``search_memory``. If omitted, those tools return an
        explanatory error.

    Returns
    -------
    Callable[[str, Dict, Path], str]
        ``(name, args, workspace) -> str``. Each tool is dispatched to the
        appropriate confined handler. The returned callable also carries a
        ``.context`` attribute (a :class:`ToolContext`) the caller mutates
        per turn (``thread_id``/``message_index``) and reads afterwards
        (``artifacts``) — see :class:`ToolContext`.
    """
    context = ToolContext()

    def execute(name: str, args: Dict, workspace: Path) -> str:
        ws = Path(workspace)
        # The model may have asked for several tools in one turn. Once the
        # user has pressed Stop there is no point running the rest of them —
        # nobody is waiting for the result, and a chart written now would be
        # filed against a turn that is already over.
        if context.cancel is not None and context.cancel.cancelled:
            return "Stopped at the user's request."
        try:
            if name == "read_file":
                result = _read_file(ws, args)
            elif name == "read_multiple_files":
                result = _read_multiple(ws, args)
            elif name == "read_image":
                result = _read_image(ws, context, args)
            elif name == "directory_tree":
                result = _dir_tree(ws, args)
            elif name == "search_content":
                result = _search_content(ws, args)
            elif name == "glob":
                result = _glob(ws, args)
            elif name == "write_file":
                result = _write_file(ws, args)
            elif name == "replace_in_file":
                result = _replace_file(ws, args)
            elif name == "apply_changes":
                result = _apply_changes(ws, args)
            elif name == "web_search":
                requested = int(args.get("max_results", 5) or 5)
                result = search.search(str(args.get("query", "")), min(requested, WEB_SEARCH_MAX_RESULTS))
            elif name == "fetch_url":
                result = search.fetch(str(args.get("url", "")), args.get("max_chars"))
            elif name == "download_file":
                result = _download(ws, args, context)
            elif name == "run_python":
                result = _run_python(ws, runner, args, context.cancel)
            elif name == "run_bash":
                result = _run_bash(ws, runner, args, context.cancel)
            elif name == "todowrite":
                result = f"[todowrite] recorded {len(args.get('todos') or [])} item(s)."
            elif name == "ask_user_question":
                result = "[ask_user_question] Not available in this automated context."
            elif name == "make_chart":
                runner.prepare()
                result = _make_chart(ws, artifact_service, context, args)
            elif name == "create_report":
                result = _create_report(ws, artifact_service, context, args)
            elif name == "generate_image":
                result = _generate_image(ws, image_service, context, args)
            elif name == "edit_image":
                result = _edit_image(ws, image_service, context, args)
            elif name == "save_memory":
                result = _save_memory(memory_store, context, args)
            elif name == "search_memory":
                result = _search_memory(memory_store, context, args)
            else:
                result = f"Unknown tool: {name}"
        except confine.OutsideWorkspaceError as e:
            return f"Blocked: {e}"
        except Exception as e:  # noqa: BLE001
            return f"Tool error: {e}"
        return _snip(result, context.tool_output_max_chars)

    execute.context = context
    return execute


def _snip(text: str, max_chars: int) -> str:
    """Snip *text* to *max_chars* (head+tail), matching Pengy's tool-output
    truncation so one huge tool result can't blow up the context window.
    ``max_chars <= 0`` means unlimited.
    """
    if not text or max_chars <= 0 or len(text) <= max_chars:
        return text
    half = max_chars // 2
    omitted = len(text) - max_chars
    return f"{text[:half]}\n...[snipped {omitted} characters]...\n{text[-half:]}"


def _download(root: Path, args: Dict, context: "ToolContext") -> str:
    import urllib.request

    url = str(args.get("url", ""))
    dest_raw = str(args.get("dir", "") or "")
    dest = _resolve(root, dest_raw) if dest_raw else root
    dest.mkdir(parents=True, exist_ok=True)
    filename = args.get("filename")
    if not filename:
        filename = url.split("/")[-1] or "download"
    out = dest / str(filename).replace("/", "_").replace(os.sep, "_")
    max_bytes = int(context.download_max_mb * 1024 * 1024) if context.download_max_mb else 0
    try:
        req = urllib.request.Request(url, headers={"User-Agent": context.user_agent})
        with urllib.request.urlopen(req, timeout=context.tool_network_timeout) as resp:
            data = resp.read(max_bytes + 1) if max_bytes else resp.read()
        if max_bytes and len(data) > max_bytes:
            return f"Download error: exceeds the {context.download_max_mb:g} MB limit."
        out.write_bytes(data)
        return f"Downloaded {_rel(root, out)} ({len(data)} bytes)."
    except Exception as e:  # noqa: BLE001
        return f"Download error: {e}"
