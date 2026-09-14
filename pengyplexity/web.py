"""Flask blueprint for the web UI: login, chat, thread history.

This is the main user-facing interface. Routes:

* ``GET/POST /login`` — the login page.
* ``POST /logout`` — end the session.
* ``GET /chat`` — redirect to the most recent thread (or new thread).
* ``GET /chat/new`` — start a new thread.
* ``GET /chat/<thread_id>`` — view/reopen a thread.
* ``POST /chat/<thread_id>/ask`` — ask a question in a thread.

The blueprint requires the Flask app to have an :class:`AppState` with
``store`` and ``auth`` populated. Tests inject a fake agent via
``state.agent``.

Login-required: every route except ``/login`` redirects to ``/login`` if
the user is not authenticated (``session['user_id']`` is absent).
"""

from __future__ import annotations

import mimetypes

from flask import (
    Blueprint,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)

from .app import get_state
from .core.auth import (
    AuthService,
    InvalidCredentialsError,
    UserDisabledError,
)
from .sandbox import confine

_ARTIFACT_INLINE_KINDS = {"chart", "image"}

web_bp = Blueprint("web", __name__, url_prefix="")


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------


def _require_login():
    """Return the current user doc, or None (caller should redirect)."""
    user_id = session.get("user_id")
    if not user_id:
        return None
    state = get_state(current_app)
    user = state.store.get_user_by_id(user_id)
    if user is None or not user.get("enabled", True):
        session.clear()
        return None
    return user


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@web_bp.route("/login", methods=["GET", "POST"])
def login():
    """Login page (GET) and login handler (POST)."""
    state = get_state(current_app)

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        try:
            user = state.auth.authenticate(username, password)
        except (InvalidCredentialsError, UserDisabledError) as e:
            return render_template(
                "login.html", error=str(e), username=username
            )

        # Success: store user_id in session.
        session["user_id"] = user["_id"]
        session["username"] = user["username"]
        return redirect(url_for("web.chat_new"))

    return render_template("login.html")


# Explicit name for the POST action in the template
login_post = login


@web_bp.route("/logout", methods=["POST"])
def logout():
    """End the session and redirect to /login."""
    session.clear()
    return redirect(url_for("web.login"))


@web_bp.route("/chat")
def chat_index():
    """Redirect to the most recent thread or a new one."""
    user = _require_login()
    if user is None:
        return redirect(url_for("web.login"))

    threads = state_store(current_app).list_threads_for_user(user["username"])
    if threads:
        return redirect(url_for("web.chat_view", thread_id=threads[0]["_id"]))
    return redirect(url_for("web.chat_new"))


@web_bp.route("/chat/new")
def chat_new():
    """Start a new thread.

    To avoid piling up blank "New Thread" entries, if the user already has an
    empty thread (no messages) we reuse it instead of creating another.
    """
    user = _require_login()
    if user is None:
        return redirect(url_for("web.login"))

    store = state_store(current_app)
    blank = _find_blank_thread(store, user["username"])
    if blank is not None:
        return redirect(url_for("web.chat_view", thread_id=blank["_id"]))

    thread = store.create_thread(user["username"])
    return redirect(url_for("web.chat_view", thread_id=thread["_id"]))


def _find_blank_thread(store, owner: str):
    """Return the owner's most recent thread with no messages, or None."""
    for t in store.list_threads_for_user(owner):
        if not t.get("messages"):
            return t
    return None


@web_bp.route("/chat/<thread_id>")
def chat_view(thread_id: str):
    """View/reopen a thread."""
    user = _require_login()
    if user is None:
        return redirect(url_for("web.login"))

    store = state_store(current_app)
    thread = store.get_thread(thread_id)

    # Ownership check: user can only see their own threads.
    if thread is None or thread.get("owner") != user["username"]:
        return render_template("login.html", error="Thread not found"), 404

    threads = store.list_threads_for_user(user["username"])
    return render_template(
        "chat.html",
        thread=thread,
        threads=threads,
        current_thread_id=thread_id,
        current_user=user,
        artifacts=_thread_artifacts(store, thread_id),
    )


@web_bp.route("/chat/<thread_id>/ask", methods=["POST"])
def ask(thread_id: str):
    """Ask a question in a thread (POST).

    Two response modes:
    * **SSE streaming** — when the client sends ``Accept: text/event-stream``
      (or ``?stream=1``), the agent's final answer is streamed as it is
      produced (``token``/``activity``/``done`` events). The assistant message
      is persisted to the store after the turn completes.
    * **Re-render** — otherwise the standard behaviour: run the agent, persist
      the answer, and re-render the thread page.

    The agent is resolved from ``AppState.agent``. In production this is a
    fully-wired agent (real model, real tools); in tests it's a fake.
    """
    user = _require_login()
    if user is None:
        return redirect(url_for("web.login"))

    store = state_store(current_app)
    thread = store.get_thread(thread_id)
    if thread is None or thread.get("owner") != user["username"]:
        return "Thread not found", 404

    question = request.form.get("question", "").strip()
    if not question:
        # Re-render with the same thread (validation error).
        threads = store.list_threads_for_user(user["username"])
        return render_template(
            "chat.html",
            thread=thread,
            threads=threads,
            current_thread_id=thread_id,
            current_user=user,
        )

    # Store the user's question.
    store.append_message(thread_id, "user", question)

    # Name the thread from the first question, if it's still an untitled/blank one.
    new_title = _maybe_name_thread(store, thread_id, question)

    wants_stream = (
        request.headers.get("Accept", "").startswith("text/event-stream")
        or request.args.get("stream") == "1"
    )
    if wants_stream:
        return _ask_stream(thread_id, user, store, question, new_title=new_title)

    answer, sources = _run_agent(thread_id, user, question)

    # Store the assistant's answer.
    store.append_message(
        thread_id, "assistant", answer, sources=sources
    )

    # Re-render the thread.
    thread = store.get_thread(thread_id)
    threads = store.list_threads_for_user(user["username"])
    return render_template(
        "chat.html",
        thread=thread,
        threads=threads,
        current_thread_id=thread_id,
        current_user=user,
        artifacts=_thread_artifacts(store, thread_id),
    )


def _run_agent(thread_id: str, user, question: str):
    """Pin the agent to *thread_id*'s workspace, run it on *question*, and
    return ``(answer, sources)``. Used by the non-streaming ask path."""
    state = get_state(current_app)
    agent = state.new_agent()
    if agent is None:
        return "[Agent not configured]", []

    agent.workspace = confine.workspace(
        user["username"], thread_id, base=state.config.workspace_root()
    )
    # A brand-new thread has no workspace directory on disk yet (it's created
    # lazily by write_file). Without this, the FIRST tool call in a thread —
    # if it's run_bash/run_python/make_chart rather than write_file — hits a
    # missing `cwd`/bind source and raises/exits with a raw host path (e.g.
    # "/home/<realuser>/.pengyplexity/workspaces/...") in the tool result,
    # leaking the real host username straight past the sandbox.
    agent.workspace.mkdir(parents=True, exist_ok=True)
    _prime_tool_context(agent, thread_id, state)
    _apply_effective_settings(state, agent, username=user["username"])
    history = _thread_history(state_store(current_app), thread_id)
    result = agent.run(question, history=history)
    return result.answer, result.sources


def _prime_tool_context(agent, thread_id: str, state) -> None:
    """Point the shared tool executor's artifact context at *thread_id*.

    ``make_chart``/``generate_image``/``edit_image`` (see ``core/toolexec.py``)
    record the artifacts they create against ``context.thread_id`` /
    ``context.message_index``. The assistant's answer for this turn hasn't
    been appended yet, so its eventual index is simply the thread's current
    message count (the user's question was already appended before the agent
    runs). Also resets ``context.artifacts`` so a caller can inspect exactly
    what this turn produced.
    """
    executor = getattr(agent, "tool_executor", None)
    context = getattr(executor, "context", None) if executor is not None else None
    if context is None:
        return
    thread = state.store.get_thread(thread_id)
    context.thread_id = thread_id
    context.message_index = len(thread.get("messages", [])) if thread else 0
    context.owner = thread.get("owner", "") if thread else ""
    context.artifacts = []


def _apply_effective_settings(state, agent, username: str | None = None) -> None:
    """Refresh the shared agent/runner/model/tool-context from the effective
    settings (admin override in the store, else the env-loaded Config) —
    see ``core/settings.py``. Mutates existing objects in place (the same
    "shared singleton repointed per turn" pattern as ``agent.workspace``)
    rather than reconstructing them, so a setting change takes effect on the
    very next turn with no app restart.

    *agent* is **this request's** agent (see ``AppState.new_agent``), not a
    shared one. The runner/model/search it configures are shared, but every
    value written there is a global admin setting, identical for every
    request, so concurrent writes of the same value are harmless. The
    per-turn values (system prompt, iteration cap, tool-output limits) land
    on the request's own agent and tool context.

    *username* is the logged-in app user (``current_user["username"]``), used
    to fill ``{username}`` in the system message template — NOT the OS user
    the process runs as, since Pengyplexity is multi-user.

    A no-op for fakes used in tests (``FakeAgent``/``FakeModelClient`` etc.
    simply lack the attributes this touches, all accessed via ``getattr``).
    """
    from .core.settings import effective_settings, render_system_message

    if agent is None:
        return
    settings = effective_settings(state.store, state.config)

    if hasattr(agent, "system_prompt"):
        raw_message = settings["system_message"]
        if raw_message:
            try:
                raw_message = render_system_message(raw_message, username=username)
            except (KeyError, IndexError, ValueError):
                # Bad admin-entered placeholder (e.g. {typo}) — fall back to
                # the raw template rather than breaking every turn.
                pass
        agent.system_prompt = raw_message or None
    if hasattr(agent, "max_iterations"):
        agent.max_iterations = int(settings["max_agent_iterations"])

    model = getattr(agent, "model", None)
    if model is not None and hasattr(model, "configure"):
        model.configure(
            base_url=settings["model_base"],
            api_key=settings["model_key"],
            model=settings["model_name"],
            temperature=float(settings["model_temperature"]),
            timeout=float(settings["llm_timeout"]),
        )

    runner = state.runner
    if runner is not None:
        if hasattr(runner, "timeout"):
            runner.timeout = int(settings["exec_timeout"])
        if hasattr(runner, "mem_bytes"):
            runner.mem_bytes = int(settings["exec_mem_mb"]) * 1024 * 1024
        if hasattr(runner, "cpu_seconds"):
            runner.cpu_seconds = int(settings["exec_cpu_seconds"])

    memory = state.memory
    if memory is not None:
        # The right relevance cut point depends on the corpus and the
        # embedding model, and the shipped defaults come from a small probe —
        # so it is tunable live rather than needing a redeploy. See the
        # calibration note in core/memory.py.
        if hasattr(memory, "signal_floor"):
            memory.signal_floor = float(settings["memory_signal_floor"])
        if hasattr(memory, "signal_confident"):
            memory.signal_confident = float(settings["memory_signal_confident"])

    search = state.search
    if search is not None:
        if hasattr(search, "timeout"):
            search.timeout = int(settings["tool_network_timeout"])
        if hasattr(search, "user_agent"):
            search.user_agent = settings["user_agent"]

    executor = getattr(agent, "tool_executor", None)
    context = getattr(executor, "context", None) if executor is not None else None
    if context is not None:
        context.tool_output_max_chars = int(settings["tool_output_max_chars"])
        context.download_max_mb = float(settings["download_max_mb"])
        context.tool_network_timeout = int(settings["tool_network_timeout"])
        context.user_agent = settings["user_agent"]


def _thread_history(store, thread_id: str) -> list:
    """Build the prior message history for *thread_id* (excluding the just-added
    user message — the agent adds it itself)."""
    thread = store.get_thread(thread_id)
    history = []
    for msg in thread.get("messages", []):
        if msg["role"] in ("user", "assistant"):
            history.append({"role": msg["role"], "content": msg["content"]})
    if history and history[-1]["role"] == "user":
        history = history[:-1]
    return history


def _ask_stream(thread_id: str, user, store, question: str, new_title: str | None = None):
    """Return an SSE response streaming the agent's answer for *thread_id*.

    Persists the assistant message (with any sources) to the store once the
    turn completes, so the re-render path and the stream agree on history.

    *new_title*, when set, is the thread's freshly auto-generated title (see
    ``_maybe_name_thread``) — the sidebar entry stays "New Thread" until the
    client is told, since the stream never re-renders the sidebar itself.
    """
    from flask import stream_with_context

    from .core.streaming import agent_stream, make_sse_response, sse_artifact, sse_error, sse_title

    state = get_state(current_app)
    agent = state.new_agent()
    if agent is None:
        return make_sse_response(iter([sse_error("[Agent not configured]")]))

    agent.workspace = confine.workspace(
        user["username"], thread_id, base=state.config.workspace_root()
    )
    # A brand-new thread has no workspace directory on disk yet (it's created
    # lazily by write_file). Without this, the FIRST tool call in a thread —
    # if it's run_bash/run_python/make_chart rather than write_file — hits a
    # missing `cwd`/bind source and raises/exits with a raw host path (e.g.
    # "/home/<realuser>/.pengyplexity/workspaces/...") in the tool result,
    # leaking the real host username straight past the sandbox.
    agent.workspace.mkdir(parents=True, exist_ok=True)
    _prime_tool_context(agent, thread_id, state)
    _apply_effective_settings(state, agent, username=user["username"])
    history = _thread_history(store, thread_id)

    # Register this turn so POST /chat/<id>/stop can interrupt it. The token
    # also reaches the sandbox runner (via the tool context) so Stop kills a
    # running script instead of waiting out the execution timeout.
    owner = user["username"]
    cancel = state.cancels.start(owner, thread_id)
    agent.cancel = cancel
    executor = getattr(agent, "tool_executor", None)
    tool_context = getattr(executor, "context", None) if executor is not None else None
    if tool_context is not None:
        tool_context.cancel = cancel

    collected = []

    def generate():
        persisted = False
        try:
            if new_title:
                yield sse_title(new_title)
            for event in agent_stream(agent, question, history=history):
                # Record tokens so we can persist the final answer.
                if event.startswith("event: token"):
                    import json as _json
                    data = event.split("\ndata: ", 1)[-1].rsplit("\n", 1)[0]
                    try:
                        collected.append(_json.loads(data)["content"])
                    except Exception:
                        pass
                yield event
            _persist_answer(store, thread_id, agent, collected, cancel)
            persisted = True

            # Any chart/image the model produced this turn (via make_chart /
            # generate_image / edit_image) is already persisted as an
            # artifact — surface it to the client now so it renders inline
            # without needing a full page reload. Requires an active request
            # context for url_for, hence stream_with_context wrapping below.
            executor = getattr(agent, "tool_executor", None)
            context = getattr(executor, "context", None) if executor is not None else None
            for record in (context.artifacts if context else []):
                if not record._id:
                    continue
                inline = record.kind in _ARTIFACT_INLINE_KINDS
                yield sse_artifact({
                    "artifact_id": record._id,
                    "filename": record.filename,
                    "kind": record.kind,
                    "mime": record.mime,
                    "url": url_for(
                        "web.download_artifact",
                        thread_id=thread_id,
                        artifact_id=record._id,
                        inline=1 if inline else None,
                    ),
                    "download_url": url_for(
                        "web.download_artifact",
                        thread_id=thread_id,
                        artifact_id=record._id,
                    ),
                })
        except Exception as e:  # noqa: BLE001
            yield sse_error(str(e))
        finally:
            # Reached on the normal path, on an error, and — the case that
            # matters — on GeneratorExit when the browser hangs up because
            # the user pressed Stop. Whatever the model had written by then
            # is the user's; persisting it here is what keeps a stopped turn
            # from vanishing out of the thread on the next page load.
            if not persisted:
                _persist_answer(
                    store, thread_id, agent, collected, cancel, interrupted=True
                )
            state.cancels.finish(owner, thread_id, cancel)

    return make_sse_response(stream_with_context(generate()))


def _persist_answer(
    store, thread_id: str, agent, collected, cancel, interrupted: bool = False
) -> None:
    """Store this turn's answer, marking it when the turn did not run to completion.

    The one place a turn's assistant message is written, so a stopped turn is
    saved exactly like a finished one — the text the model had already
    produced belongs to the user and has to survive a reload.

    *interrupted* means the generator is unwinding because the browser hung
    up (``GeneratorExit``), rather than finishing normally.
    """
    answer = "".join(collected).strip()
    stopped = cancel is not None and cancel.cancelled

    if stopped:
        note = "_[Stopped]_"
    elif interrupted:
        note = "_[Interrupted]_"
    else:
        note = ""

    if note:
        if not answer and interrupted and not stopped:
            # Nothing was produced and nobody asked to stop: an early failure
            # already surfaced as an `error` event, and a blank assistant
            # bubble in the thread would only be confusing.
            return
        answer = f"{answer}\n\n{note}" if answer else note

    try:
        store.append_message(
            thread_id, "assistant", answer,
            sources=getattr(agent, "_last_sources", None) or [],
        )
    except Exception:  # noqa: BLE001
        # Never let bookkeeping raise out of a generator that is already
        # unwinding — it would replace the real reason in the log.
        if not interrupted:
            raise


@web_bp.route("/chat/<thread_id>/stop", methods=["POST"])
def stop_turn(thread_id: str):
    """Interrupt the turn currently running in *thread_id*.

    The browser also aborts its own request, which is what makes the UI feel
    instant. This route is the server half: it flips the turn's cancel flag
    and kills whatever the sandbox is running, so a stop during a long chart
    script takes effect immediately rather than after the execution timeout.

    Keyed by owner as well as thread, so one user can never stop another's
    turn. Stopping a thread with nothing running is not an error — the turn
    may have finished between the click and the request.
    """
    user = _require_login()
    if user is None:
        return {"error": "not authenticated"}, 401

    store = state_store(current_app)
    thread = store.get_thread(thread_id)
    if thread is None or thread.get("owner") != user["username"]:
        return {"error": "thread not found"}, 404

    state = get_state(current_app)
    stopped = state.cancels.cancel(user["username"], thread_id)
    return {"status": "ok", "stopped": stopped}


@web_bp.route("/chat/<thread_id>/artifact/<artifact_id>")
def download_artifact(thread_id: str, artifact_id: str):
    """Serve an artifact file for *thread_id*.

    The file is served **only** if:

    1. the request is authenticated and the user owns *thread_id*;
    2. the artifact record belongs to *thread_id*;
    3. the recorded path resolves **inside the thread's workspace**
       (``confine.resolve``) — a record pointing at a sibling dir, a
       ``..`` path, or an absolute path outside the workspace is rejected
       with 403. This is the structural escape-proof guard: the route can
       never be turned into a path-traversal read.

    ``?inline=1`` (or a chart/image artifact) serves the file inline so the
    chat pane can embed it in an ``<img>``; otherwise it is a download.
    """
    user = _require_login()
    if user is None:
        return redirect(url_for("web.login"))

    store = state_store(current_app)
    thread = store.get_thread(thread_id)
    if thread is None or thread.get("owner") != user["username"]:
        return "Thread not found", 404

    if not hasattr(store, "get_artifact"):
        return "Artifact storage unavailable", 404

    artifact = store.get_artifact(artifact_id)
    if artifact is None or artifact.get("thread_id") != thread_id:
        return "Artifact not found", 404

    # Confinement: re-derive the thread's workspace root and require the
    # recorded path to resolve inside it. ``resolve`` accepts an absolute path
    # only if it already lies inside the root, so a hostile record cannot
    # widen out of the sandbox.
    state = get_state(current_app)
    workspace_root = confine.workspace(
        user["username"], thread_id, base=state.config.workspace_root()
    )
    try:
        confined = confine.resolve(workspace_root, artifact.get("path"))
    except confine.OutsideWorkspaceError:
        return "Artifact path is outside the workspace", 403

    if not confined.exists() or not confined.is_file():
        return "Artifact file is missing", 404

    mimetype = artifact.get("mime") or mimetypes.guess_type(
        confined.name
    )[0] or "application/octet-stream"

    inline = request.args.get("inline") == "1" or artifact.get("kind") in {
        "chart",
        "image",
    }
    return send_file(
        confined,
        mimetype=mimetype,
        as_attachment=not inline,
        download_name=confined.name,
    )


@web_bp.route("/account/password", methods=["GET", "POST"])
def change_password():
    """Self-service password change for the logged-in user.

    Distinct from the admin-only ``/admin/users/<id>/reset-password``: this
    requires the caller to prove they know their *current* password rather
    than an admin overriding it.
    """
    user = _require_login()
    if user is None:
        return redirect(url_for("web.login"))

    state = get_state(current_app)
    error = None
    success = None

    if request.method == "POST":
        current_password = request.form.get("current_password", "")
        new_password = request.form.get("new_password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not new_password:
            error = "New password is required."
        elif new_password != confirm_password:
            error = "New password and confirmation do not match."
        else:
            try:
                state.auth.change_password(user, current_password, new_password)
                success = "Password updated."
            except InvalidCredentialsError as e:
                error = str(e)

    return render_template(
        "account.html",
        current_user=user,
        error=error,
        success=success or request.args.get("success"),
    )


@web_bp.route("/account/theme", methods=["POST"])
def update_theme():
    """Save the logged-in user's theme preference (mode + accent).

    Ported from Pengy's Qt theme system (see ``core/theme.py`` /
    ``static/css/theme.css``); ``base.html``'s context processor reads this
    back on every page via ``data-theme``/``data-accent`` attributes.
    """
    from .core.theme import normalize_theme_accent, normalize_theme_mode

    user = _require_login()
    if user is None:
        return redirect(url_for("web.login"))

    state = get_state(current_app)
    mode = normalize_theme_mode(request.form.get("theme_mode"))
    accent = normalize_theme_accent(request.form.get("theme_accent"))
    state.store.update_user(user["_id"], theme_mode=mode, theme_accent=accent)
    return redirect(url_for("web.change_password", success="Appearance updated"))


@web_bp.route("/memories", methods=["GET", "POST"])
def memories():
    """List/search the current user's memories, and create new ones.

    ``GET ?q=...`` runs a hybrid (lexical + semantic) search over the user's
    memories; with no query, lists everything (newest first). ``POST``
    creates a new memory from the form.
    """
    user = _require_login()
    if user is None:
        return redirect(url_for("web.login"))

    state = get_state(current_app)
    memory = state.memory
    error = None
    success = request.args.get("success")

    if request.method == "POST":
        title = request.form.get("title", "").strip()
        summary = request.form.get("summary", "").strip()
        body = request.form.get("body", "").strip()
        tags = _parse_tags(request.form.get("tags", ""))
        if memory is None:
            error = "Memory capability is not configured."
        elif not title or not summary:
            error = "Title and summary are required."
        else:
            memory.create(owner=user["username"], title=title, summary=summary, tags=tags, body=body)
            success = "Memory saved."

    query = request.args.get("q", "").strip()
    signals = {}
    # ``?all=1`` drops the relevance floor, for checking whether a near miss
    # exists at all. The page links to it from the empty state.
    show_all = request.args.get("all") == "1"
    results = []
    found = None
    if memory is not None:
        if query:
            found = memory.search(
                user["username"], query, limit=25,
                min_signal=0 if show_all else None,
            )
            results = found.docs
            # Keyed by id so the template can badge each row with how well it
            # actually matched, rather than presenting every hit as equal.
            signals = {
                hit.doc["_id"]: hit
                for hit in found.results
                if hit.signal is not None
            }
        else:
            results = memory.list(user["username"], statuses=("active", "superseded", "deprecated"))

    return render_template(
        "memories.html",
        current_user=user,
        memories=results,
        query=query,
        found=found,
        signals=signals,
        show_all=show_all,
        error=error,
        success=success,
    )


@web_bp.route("/memories/<memory_id>/edit", methods=["GET", "POST"])
def edit_memory(memory_id: str):
    """View/edit a single memory the current user owns."""
    user = _require_login()
    if user is None:
        return redirect(url_for("web.login"))

    state = get_state(current_app)
    memory = state.memory
    if memory is None:
        return "Memory capability is not configured", 404

    doc = memory.get(memory_id, owner=user["username"])
    if doc is None:
        return "Memory not found", 404

    error = None
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        summary = request.form.get("summary", "").strip()
        body = request.form.get("body", "").strip()
        status = request.form.get("status", "active")
        tags = _parse_tags(request.form.get("tags", ""))
        if not title or not summary:
            error = "Title and summary are required."
        else:
            doc = memory.update(
                memory_id,
                owner=user["username"],
                editor=user["username"],
                title=title,
                summary=summary,
                body=body,
                status=status,
                tags=tags,
            )
            return redirect(url_for("web.memories", success="Memory updated"))

    return render_template("memory_edit.html", current_user=user, memory=doc, error=error)


@web_bp.route("/memories/<memory_id>/delete", methods=["POST"])
def delete_memory(memory_id: str):
    """Delete a memory the current user owns."""
    user = _require_login()
    if user is None:
        return redirect(url_for("web.login"))

    state = get_state(current_app)
    memory = state.memory
    if memory is not None:
        memory.delete(memory_id, owner=user["username"])
    return redirect(url_for("web.memories"))


def _parse_tags(raw: str) -> list:
    return [t.strip() for t in raw.split(",") if t.strip()]


@web_bp.route("/workspace")
def workspace():
    """A gallery of every artifact (chart/image/report) across all of the
    current user's threads, with a per-thread or all-at-once ZIP download."""
    user = _require_login()
    if user is None:
        return redirect(url_for("web.login"))

    store = state_store(current_app)
    threads = store.list_threads_for_user(user["username"])
    threads_by_id = {t["_id"]: t for t in threads}

    artifacts = _owner_artifacts(store, user["username"])
    groups = {}
    for a in artifacts:
        groups.setdefault(a.get("thread_id", ""), []).append(a)

    # Order groups by the owning thread's `updated` time, newest first.
    ordered_group_ids = sorted(
        groups.keys(),
        key=lambda tid: (threads_by_id.get(tid) or {}).get("updated") or (threads_by_id.get(tid) or {}).get("created"),
        reverse=True,
    )

    return render_template(
        "workspace.html",
        current_user=user,
        threads_by_id=threads_by_id,
        group_ids=ordered_group_ids,
        groups=groups,
        total_count=len(artifacts),
    )


@web_bp.route("/workspace/zip")
def workspace_zip():
    """Stream a ZIP of the user's artifacts.

    With ``?thread_id=<id>`` zips just that thread's artifacts (ownership
    re-checked); otherwise zips everything the user owns. Every artifact path
    is re-confined to its own thread's workspace before being read — the same
    structural guard as ``download_artifact`` — so a hostile/stale record
    can never be used to read outside the sandbox.
    """
    import io
    import zipfile

    user = _require_login()
    if user is None:
        return redirect(url_for("web.login"))

    store = state_store(current_app)
    state = get_state(current_app)
    thread_id = request.args.get("thread_id")

    if thread_id:
        thread = store.get_thread(thread_id)
        if thread is None or thread.get("owner") != user["username"]:
            return "Thread not found", 404
        artifacts = _thread_artifacts(store, thread_id)
        zip_name = f"{_safe_zip_stem(thread.get('title') or thread_id)}.zip"
    else:
        artifacts = _owner_artifacts(store, user["username"])
        zip_name = f"{_safe_zip_stem(user['username'])}-workspace.zip"

    buf = io.BytesIO()
    used_names = set()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for a in artifacts:
            tid = a.get("thread_id", "")
            thread = store.get_thread(tid)
            if thread is None or thread.get("owner") != user["username"]:
                continue
            workspace_root = confine.workspace(
                user["username"], tid, base=state.config.workspace_root()
            )
            try:
                confined = confine.resolve(workspace_root, a.get("path"))
            except confine.OutsideWorkspaceError:
                continue
            if not confined.exists() or not confined.is_file():
                continue
            arcname = f"{tid}/{confined.name}"
            n = 1
            while arcname in used_names:
                arcname = f"{tid}/{n}_{confined.name}"
                n += 1
            used_names.add(arcname)
            zf.write(confined, arcname=arcname)
    buf.seek(0)

    return send_file(
        buf,
        mimetype="application/zip",
        as_attachment=True,
        download_name=zip_name,
    )


def _owner_artifacts(store, owner: str) -> list:
    """All artifacts across every thread *owner* owns, newest first."""
    if hasattr(store, "get_artifacts_for_owner"):
        return store.get_artifacts_for_owner(owner)
    if not hasattr(store, "get_artifacts_for_thread"):
        return []
    out = []
    for t in store.list_threads_for_user(owner):
        out.extend(store.get_artifacts_for_thread(t["_id"]))
    out.sort(key=lambda a: a.get("created") or "", reverse=True)
    return out


def _safe_zip_stem(name: str) -> str:
    stem = "".join(c if (c.isalnum() or c in ("-", "_")) else "-" for c in name.strip().lower())
    stem = stem.strip("-") or "download"
    return stem[:60]


@web_bp.route("/chat/<thread_id>/delete", methods=["POST"])
def delete_thread(thread_id: str):
    """Delete a thread the current user owns, then redirect to the next one."""
    user = _require_login()
    if user is None:
        return redirect(url_for("web.login"))

    store = state_store(current_app)
    thread = store.get_thread(thread_id)
    if thread is None or thread.get("owner") != user["username"]:
        return "Thread not found", 404

    store.delete_thread(thread_id)

    # Redirect to the user's most recent remaining thread, or to /chat/new
    # (which will create one, or reuse an existing blank).
    remaining = store.list_threads_for_user(user["username"])
    if remaining:
        return redirect(url_for("web.chat_view", thread_id=remaining[0]["_id"]))
    return redirect(url_for("web.chat_new", _external=False))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def state_store(app):
    """Shorthand: get the store from app state."""
    return get_state(app).store


def _default_thread_title() -> str:
    return "New Thread"


def _maybe_name_thread(store, thread_id: str, question: str) -> Optional[str]:
    """Set a human-friendly title on a thread from its first question.

    Only names a thread whose title is still the default "New Thread". Uses a
    small LLM call via the app's model client; on any failure it falls back to
    a truncated first line of the question so the title is always set.

    Returns the new title, or ``None`` if the thread already had a real title
    (nothing changed) — callers use this to tell the client about the rename.
    """
    thread = store.get_thread(thread_id)
    if thread is None:
        return None
    title = (thread.get("title") or "").strip()
    if title and title != _default_thread_title():
        return None
    model = _title_model()
    new_title = model.generate_title(question) if model is not None else None
    if not new_title:
        first = question.splitlines()[0].strip() if question.strip() else "New Thread"
        new_title = first[:60] or "New Thread"
    store.update_thread_title(thread_id, new_title)
    return new_title


def _title_model():
    """Return the app's model client (for title generation), or None.

    Prefers the shared client on the app state. Building a whole request
    agent just to reach its model would be wasteful, and a test that injects
    a fake agent still needs its fake model honoured — hence the fallback.
    """
    try:
        state = get_state(current_app)
    except RuntimeError:
        return None
    injected = getattr(state, "agent", None)
    if injected is not None:
        return getattr(injected, "model", None)
    return getattr(state, "model", None)


def _thread_artifacts(store, thread_id: str) -> list:
    """Return this thread's artifact records for the template (empty if the
    store predates the artifacts collection, so fakes keep working)."""
    if not hasattr(store, "get_artifacts_for_thread"):
        return []
    return store.get_artifacts_for_thread(thread_id)
