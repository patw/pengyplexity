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
    make_response,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)

from .app import get_state
from .core.apikeys import TooManyKeysError
from .core.auth import (
    AuthService,
    InvalidCredentialsError,
    UserDisabledError,
)
from .sandbox import confine
from .turns import (
    DEFAULT_THREAD_TITLE,
    AgentUnavailable,
    ThreadGone,
    apply_effective_settings,
    begin_turn,
    maybe_name_thread,
    persist_answer,
    prime_tool_context,
    run_turn,
    thread_history,
    title_model,
)

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

    wants_stream = (
        request.headers.get("Accept", "").startswith("text/event-stream")
        or request.args.get("stream") == "1"
    )
    if wants_stream:
        return _ask_stream(thread_id, user, question)

    # Store the user's question.
    store.append_message(thread_id, "user", question)

    # Name the thread from the first question, if it's still an untitled/blank one.
    _maybe_name_thread(store, thread_id, question)

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
    # Created up front so the first tool call can't hit a missing directory
    # and leak the host path — see turns.begin_turn.
    agent.workspace.mkdir(parents=True, exist_ok=True)
    prime_tool_context(agent, thread_id, state)
    settings = apply_effective_settings(state, agent, username=user["username"])
    history = thread_history(
        state_store(current_app), thread_id, max_messages=int(settings["thread_history_messages"])
    )
    result = agent.run(question, history=history)
    return result.answer, result.sources


# The turn machinery moved to turns.py so the JSON API shares it; these names
# are kept for existing callers.
_prime_tool_context = prime_tool_context
_apply_effective_settings = apply_effective_settings
_thread_history = thread_history
_persist_answer = persist_answer


def _ask_stream(thread_id: str, user, question: str):
    """Return an SSE response streaming the agent's answer for *thread_id*.

    The turn itself — storing the question, naming the thread, registering it
    for Stop, persisting the (possibly partial) answer — is
    :func:`pengyplexity.turns.run_turn`, shared with the JSON API.
    """
    from flask import stream_with_context

    from .core.streaming import make_sse_response, sse_error

    state = get_state(current_app)
    try:
        turn = begin_turn(state, user, thread_id, question)
    except AgentUnavailable as e:
        return make_sse_response(iter([sse_error(f"[{e}]")]))
    except ThreadGone:
        return "Thread not found", 404

    def artifact_links(record):
        inline = record.kind in _ARTIFACT_INLINE_KINDS
        return {
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
        }

    def generate():
        events = run_turn(turn, artifact_links)
        try:
            for event in events:
                yield event.to_sse()
        finally:
            events.close()

    return make_sse_response(stream_with_context(generate()))


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

    return _render_account(
        user, error=error, success=success or request.args.get("success")
    )


def _render_account(user, **context):
    """Render the Account page (appearance, password, API keys)."""
    state = get_state(current_app)
    keys = state.api_keys.list_for_user(user) if state.api_keys is not None else []
    context.setdefault("api_key_success", request.args.get("api_key_success"))
    context.setdefault("api_key_error", request.args.get("api_key_error"))
    api_base = url_for("api.me", _external=True)[: -len("/me")]
    resp = make_response(render_template(
        "account.html",
        current_user=user,
        api_keys=keys,
        api_base=api_base,
        **context,
    ))
    if context.get("new_api_key"):
        # The plaintext key is on this page and nowhere else; keep it out of
        # every cache.
        resp.headers["Cache-Control"] = "no-store"
    return resp


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


@web_bp.route("/account/api-keys", methods=["POST"])
def create_api_key():
    """Mint an API key and show it — once — on the Account page.

    Rendered directly rather than redirected: the plaintext must never pass
    through a URL or the (signed but readable) session cookie.
    """
    user = _require_login()
    if user is None:
        return redirect(url_for("web.login"))

    state = get_state(current_app)
    try:
        key, record = state.api_keys.create(user, request.form.get("name", ""))
    except TooManyKeysError as e:
        return _render_account(user, api_key_error=str(e))
    return _render_account(
        user,
        new_api_key=key,
        api_key_success=f"API key '{record['name']}' created.",
    )


@web_bp.route("/account/api-keys/<key_id>/revoke", methods=["POST"])
def revoke_api_key(key_id: str):
    """Revoke one of the logged-in user's API keys."""
    user = _require_login()
    if user is None:
        return redirect(url_for("web.login"))

    state = get_state(current_app)
    if state.api_keys.revoke(user, key_id):
        return redirect(url_for(
            "web.change_password", api_key_success="API key revoked.", _anchor="api-keys"
        ))
    return redirect(url_for(
        "web.change_password", api_key_error="API key not found.", _anchor="api-keys"
    ))


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
    return DEFAULT_THREAD_TITLE


def _maybe_name_thread(store, thread_id: str, question: str) -> Optional[str]:
    """Name an untitled thread from its first question — see
    :func:`pengyplexity.turns.maybe_name_thread`."""
    return maybe_name_thread(get_state(current_app), thread_id, question)


def _title_model():
    """The app's model client for title generation, or None."""
    try:
        state = get_state(current_app)
    except RuntimeError:
        return None
    return title_model(state)


def _thread_artifacts(store, thread_id: str) -> list:
    """Return this thread's artifact records for the template (empty if the
    store predates the artifacts collection, so fakes keep working)."""
    if not hasattr(store, "get_artifacts_for_thread"):
        return []
    return store.get_artifacts_for_thread(thread_id)
