"""JSON API (``/api/v1``) for programs: the Discord bot, scripts, anything
that is not a browser.

A user creates a personal API key on their Account page and sends it as
``Authorization: Bearer <key>``. The API then acts as that user, with the same
agent the web UI uses — every turn goes through :mod:`pengyplexity.turns`, so
tools, sandbox, admin settings, persistence and cancellation are identical.

Conventions:

* **Bearer keys only.** The session cookie is ignored here, so a browser
  visiting a hostile page cannot be steered into API calls (no CSRF surface).
* **JSON in, JSON out.** Every error, including 404/405 for unknown routes
  under ``/api/``, is ``{"error": {"code": "...", "message": "..."}}`` with a
  stable machine-readable ``code``.
* **Refusals are clean.** A rate-limited, busy, or invalid request is refused
  before anything is written, so a client can retry it as-is.
* **Owner-scoped.** Anything not owned by the key's user is a 404, the same as
  something that does not exist.

The route table and event reference live in SPEC.md ("HTTP API").
"""

from __future__ import annotations

import mimetypes
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from flask import (
    Blueprint,
    current_app,
    g,
    jsonify,
    request,
    send_file,
    stream_with_context,
    url_for,
)
from werkzeug.exceptions import HTTPException

from .app import get_state
from .core.cancel import TooManyTurns, TurnInProgress
from .core.memory import VALID_STATUSES
from .core.streaming import StreamEvent, make_sse_response
from .sandbox import confine
from .turns import (
    DEFAULT_THREAD_TITLE,
    AgentUnavailable,
    ThreadGone,
    Turn,
    begin_turn,
    run_turn,
)

api_bp = Blueprint("api", __name__, url_prefix="/api/v1")

API_PREFIX = "/api/"
MAX_BODY_BYTES = 1024 * 1024
MAX_CONTENT_CHARS = 100_000
MAX_TITLE_CHARS = 200
MAX_SUMMARY_CHARS = 2_000
MAX_TAGS = 20
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 200
_INLINE_ARTIFACT_KINDS = {"chart", "image"}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ApiError(Exception):
    """An error with an HTTP status and a stable machine-readable code."""

    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.headers = headers or {}


def _error_response(status: int, code: str, message: str, headers=None):
    resp = jsonify({"error": {"code": code, "message": message}})
    resp.status_code = status
    for name, value in (headers or {}).items():
        resp.headers[name] = value
    return resp


@api_bp.errorhandler(ApiError)
def _handle_api_error(e: ApiError):
    return _error_response(e.status, e.code, e.message, e.headers)


@api_bp.app_errorhandler(HTTPException)
def _handle_http_exception(e: HTTPException):
    """JSON for HTTP errors under ``/api/`` (including unmatched routes, which
    never reach a blueprint handler); everything else keeps Flask's default."""
    if not request.path.startswith(API_PREFIX):
        return e
    code = (e.name or "error").lower().replace(" ", "_")
    headers = {}
    valid_methods = getattr(e, "valid_methods", None)
    if valid_methods:
        headers["Allow"] = ", ".join(valid_methods)
    return _error_response(e.code or 500, code, e.description or e.name, headers)


@api_bp.errorhandler(Exception)
def _handle_unexpected(e: Exception):
    if isinstance(e, HTTPException):
        return _handle_http_exception(e)
    current_app.logger.exception("Unhandled error in API request %s", request.path)
    return _error_response(500, "internal_error", "Internal server error")


# ---------------------------------------------------------------------------
# Authentication + response hardening
# ---------------------------------------------------------------------------

_WWW_AUTHENTICATE = {"WWW-Authenticate": 'Bearer realm="pengyplexity"'}


@api_bp.before_request
def _authenticate():
    state = get_state(current_app)
    scheme, _, token = request.headers.get("Authorization", "").partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not token:
        raise ApiError(
            401, "unauthorized",
            "Send an API key as 'Authorization: Bearer <key>'. "
            "Create one on your Account page.",
            _WWW_AUTHENTICATE,
        )
    found = state.api_keys.authenticate(token)
    if found is None:
        raise ApiError(
            401, "invalid_api_key",
            "API key is invalid, revoked, or belongs to a disabled account.",
            _WWW_AUTHENTICATE,
        )
    g.api_user, g.api_key = found

    # After auth, so an anonymous caller learns nothing from the size check.
    if request.content_length is not None and request.content_length > MAX_BODY_BYTES:
        raise ApiError(413, "payload_too_large", f"Request body exceeds {MAX_BODY_BYTES} bytes.")


@api_bp.after_request
def _harden(resp):
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    if resp.mimetype == "application/json":
        resp.headers.setdefault("Cache-Control", "no-store")
    return resp


# ---------------------------------------------------------------------------
# Request parsing
# ---------------------------------------------------------------------------


def _json_body(required: bool = True) -> Dict[str, Any]:
    if not request.get_data(cache=True):
        if required:
            raise ApiError(400, "invalid_request", "Request body must be a JSON object.")
        return {}
    data = request.get_json(silent=True, force=True)
    if not isinstance(data, dict):
        raise ApiError(400, "invalid_json", "Request body must be a JSON object.")
    return data


def _string_field(
    data: Dict[str, Any],
    name: str,
    required: bool = False,
    max_chars: int = 0,
    allow_empty: bool = False,
) -> Optional[str]:
    value = data.get(name)
    if value is None:
        if required:
            raise ApiError(400, "invalid_request", f"'{name}' is required.")
        return None
    if not isinstance(value, str):
        raise ApiError(400, "invalid_request", f"'{name}' must be a string.")
    value = value.strip()
    if not value and not allow_empty:
        raise ApiError(400, "invalid_request", f"'{name}' must not be empty.")
    if max_chars and len(value) > max_chars:
        raise ApiError(
            400, "invalid_request", f"'{name}' is longer than {max_chars} characters."
        )
    return value


def _bool_field(data: Dict[str, Any], name: str, default: bool) -> bool:
    value = data.get(name)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ApiError(400, "invalid_request", f"'{name}' must be true or false.")
    return value


def _tags_field(data: Dict[str, Any]) -> Optional[List[str]]:
    value = data.get("tags")
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(t, str) for t in value):
        raise ApiError(400, "invalid_request", "'tags' must be a list of strings.")
    if len(value) > MAX_TAGS:
        raise ApiError(400, "invalid_request", f"'tags' may hold at most {MAX_TAGS} tags.")
    return value


def _status_field(data: Dict[str, Any]) -> Optional[str]:
    value = _string_field(data, "status")
    if value is not None and value not in VALID_STATUSES:
        allowed = ", ".join(sorted(VALID_STATUSES))
        raise ApiError(400, "invalid_request", f"'status' must be one of: {allowed}.")
    return value


def _int_arg(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = request.args.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ApiError(400, "invalid_request", f"'{name}' must be an integer.") from None
    if not minimum <= value <= maximum:
        raise ApiError(
            400, "invalid_request", f"'{name}' must be between {minimum} and {maximum}."
        )
    return value


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _iso(value: Any) -> Any:
    if not isinstance(value, datetime):
        return value
    if value.tzinfo is None:  # moofile hands datetimes back naive (UTC)
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return _iso(value)


def _key_json(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": doc["_id"],
        "name": doc.get("name", ""),
        "display_prefix": doc.get("display_prefix", ""),
        "created": _iso(doc.get("created")),
        "last_used": _iso(doc.get("last_used")),
    }


def _artifact_urls(thread_id: str, artifact_id: str) -> Dict[str, str]:
    url = url_for("api.get_artifact", thread_id=thread_id, artifact_id=artifact_id)
    return {"url": url, "download_url": f"{url}?download=1"}


def _artifact_json(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": doc["_id"],
        "filename": doc.get("filename", ""),
        "kind": doc.get("kind", ""),
        "mime": doc.get("mime", ""),
        "size_bytes": doc.get("size_bytes", 0),
        "message_index": doc.get("message_index"),
        "created": _iso(doc.get("created")),
        **_artifact_urls(doc.get("thread_id", ""), doc["_id"]),
    }


def _message_json(index: int, msg: Dict[str, Any], artifacts: List[Dict[str, Any]]):
    return {
        "index": index,
        "role": msg.get("role"),
        "type": msg.get("type", "text"),
        "content": msg.get("content", ""),
        "sources": msg.get("sources") or [],
        "created": _iso(msg.get("created")),
        "artifacts": [
            _artifact_json(a) for a in artifacts if a.get("message_index") == index
        ],
    }


def _thread_artifacts(thread_id: str) -> List[Dict[str, Any]]:
    store = get_state(current_app).store
    if not hasattr(store, "get_artifacts_for_thread"):
        return []
    return store.get_artifacts_for_thread(thread_id)


def _thread_summary(thread: Dict[str, Any]) -> Dict[str, Any]:
    state = get_state(current_app)
    return {
        "id": thread["_id"],
        "title": thread.get("title") or "",
        "created": _iso(thread.get("created")),
        "updated": _iso(thread.get("updated")),
        "message_count": len(thread.get("messages") or []),
        "busy": state.cancels.active(thread.get("owner", ""), thread["_id"]) is not None,
    }


def _thread_detail(thread: Dict[str, Any]) -> Dict[str, Any]:
    artifacts = _thread_artifacts(thread["_id"])
    out = _thread_summary(thread)
    out["messages"] = [
        _message_json(i, m, artifacts) for i, m in enumerate(thread.get("messages") or [])
    ]
    return out


def _memory_json(doc: Dict[str, Any], hit: Any = None) -> Dict[str, Any]:
    out = {
        "id": doc["_id"],
        "title": doc.get("title", ""),
        "summary": doc.get("summary", ""),
        "body": doc.get("body", ""),
        "tags": doc.get("tags") or [],
        "status": doc.get("status", "active"),
        "created": _iso(doc.get("created")),
        "updated": _iso(doc.get("updated")),
        "update_history": _jsonable(doc.get("update_history") or []),
    }
    if hit is not None:
        out["match"] = {
            "signal": hit.signal,
            "signal_kind": hit.signal_kind,
            "confidence": hit.confidence,
        }
    return out


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------


def _username() -> str:
    return g.api_user["username"]


def _owned_thread(thread_id: str) -> Dict[str, Any]:
    thread = get_state(current_app).store.get_thread(thread_id)
    if thread is None or thread.get("owner") != _username():
        raise ApiError(404, "thread_not_found", "Thread not found.")
    return thread


def _memory_service():
    memory = get_state(current_app).memory
    if memory is None:
        raise ApiError(503, "memory_unavailable", "Memory is not configured on this server.")
    return memory


# ---------------------------------------------------------------------------
# Account
# ---------------------------------------------------------------------------


@api_bp.get("/me")
def me():
    state = get_state(current_app)
    user = g.api_user
    limiter = state.api_rate_limiter
    return {
        "user": {
            "id": user["_id"],
            "username": user["username"],
            "is_admin": bool(user.get("is_admin")),
        },
        "api_key": _key_json(g.api_key),
        "limits": {
            "turns_per_minute": getattr(limiter, "limit", 0) if limiter else 0,
            "max_concurrent_turns": state.config.api_max_concurrent_turns,
            "max_content_chars": MAX_CONTENT_CHARS,
        },
    }


@api_bp.get("/keys")
def list_keys():
    state = get_state(current_app)
    current_id = g.api_key["_id"]
    keys = []
    for doc in state.api_keys.list_for_user(g.api_user):
        keys.append({**_key_json(doc), "current": doc["_id"] == current_id})
    return {"keys": keys}


@api_bp.delete("/keys/<key_id>")
def revoke_key(key_id: str):
    """Revoke one of the caller's keys — including the one making the call,
    which is the quickest response to a leaked key. Keys cannot be *created*
    over the API; that needs the web session."""
    if not get_state(current_app).api_keys.revoke(g.api_user, key_id):
        raise ApiError(404, "api_key_not_found", "API key not found.")
    return "", 204


# ---------------------------------------------------------------------------
# Threads
# ---------------------------------------------------------------------------


@api_bp.get("/threads")
def list_threads():
    limit = _int_arg("limit", DEFAULT_PAGE_SIZE, 1, MAX_PAGE_SIZE)
    offset = _int_arg("offset", 0, 0, 10**9)
    threads = get_state(current_app).store.list_threads_for_user(_username())
    return {
        "threads": [_thread_summary(t) for t in threads[offset:offset + limit]],
        "total": len(threads),
        "limit": limit,
        "offset": offset,
    }


@api_bp.post("/threads")
def create_thread():
    """Create a thread. Unlike the web UI's /chat/new, an existing blank
    thread is never reused: a program asking for a thread wants its own."""
    data = _json_body(required=False)
    title = _string_field(data, "title", max_chars=MAX_TITLE_CHARS)
    thread = get_state(current_app).store.create_thread(
        _username(), title or DEFAULT_THREAD_TITLE
    )
    return _thread_detail(thread), 201


@api_bp.get("/threads/<thread_id>")
def get_thread(thread_id: str):
    return _thread_detail(_owned_thread(thread_id))


@api_bp.patch("/threads/<thread_id>")
def rename_thread(thread_id: str):
    _owned_thread(thread_id)
    title = _string_field(_json_body(), "title", required=True, max_chars=MAX_TITLE_CHARS)
    store = get_state(current_app).store
    store.update_thread_title(thread_id, title)
    return _thread_summary(_owned_thread(thread_id))


@api_bp.delete("/threads/<thread_id>")
def delete_thread(thread_id: str):
    _owned_thread(thread_id)
    state = get_state(current_app)
    # Don't leave a turn running against a thread that no longer exists.
    state.cancels.cancel(_username(), thread_id)
    state.store.delete_thread(thread_id)
    return "", 204


@api_bp.post("/threads/<thread_id>/stop")
def stop_turn(thread_id: str):
    _owned_thread(thread_id)
    stopped = get_state(current_app).cancels.cancel(_username(), thread_id)
    return {"stopped": stopped}


@api_bp.post("/threads/<thread_id>/messages")
def post_message(thread_id: str):
    """Ask a question in a thread — the API's equivalent of the web UI's Ask.

    ``{"content": "...", "stream": false}`` waits for the whole turn and
    returns the persisted answer as JSON. ``"stream": true`` (or
    ``Accept: text/event-stream``) streams the same events the browser gets,
    followed by a final ``message`` event carrying the persisted answer.
    """
    state = get_state(current_app)
    user = g.api_user
    _owned_thread(thread_id)

    data = _json_body()
    content = _string_field(data, "content", required=True, max_chars=MAX_CONTENT_CHARS)
    wants_sse = "text/event-stream" in request.headers.get("Accept", "")
    stream = _bool_field(data, "stream", default=wants_sse)

    limiter = state.api_rate_limiter
    if limiter is not None:
        allowed, retry_after = limiter.hit(user["_id"])
        if not allowed:
            raise ApiError(
                429, "rate_limited",
                f"Too many questions; retry in {retry_after}s.",
                {"Retry-After": str(retry_after)},
            )

    max_active = state.config.api_max_concurrent_turns
    try:
        turn = begin_turn(
            state, user, thread_id, content, exclusive=True, max_active=max_active
        )
    except TurnInProgress:
        raise ApiError(
            409, "turn_in_progress",
            "This thread is already answering a question. Wait for it to "
            "finish, or POST to its /stop endpoint.",
        ) from None
    except TooManyTurns:
        raise ApiError(
            429, "too_many_concurrent_turns",
            f"You already have {max_active} questions running; wait for one to finish.",
        ) from None
    except AgentUnavailable:
        raise ApiError(503, "agent_unavailable", "The agent is not configured on this server.") from None
    except ThreadGone:
        raise ApiError(404, "thread_not_found", "Thread not found.") from None

    def artifact_links(record) -> Dict[str, str]:
        return _artifact_urls(thread_id, record._id)

    if stream:
        return _stream_turn(turn, artifact_links)
    return _complete_turn(turn, artifact_links)


def _turn_result(turn: Turn) -> Optional[Dict[str, Any]]:
    """The persisted outcome of a finished turn, or None if the thread is gone."""
    thread = get_state(current_app).store.get_thread(turn.thread_id)
    if thread is None:
        return None
    messages = thread.get("messages") or []
    message = None
    if turn.answer_index < len(messages):
        candidate = messages[turn.answer_index]
        if candidate.get("role") == "assistant":
            message = _message_json(
                turn.answer_index, candidate, _thread_artifacts(turn.thread_id)
            )
    return {
        "thread": _thread_summary(thread),
        "message": message,
        "stopped": bool(turn.cancel is not None and turn.cancel.cancelled),
    }


def _complete_turn(turn: Turn, artifact_links):
    error = None
    for event in run_turn(turn, artifact_links):
        if event.event == "error":
            error = event.data.get("message") or "The agent failed."
    result = _turn_result(turn)
    if result is None:
        raise ApiError(404, "thread_not_found", "Thread was deleted during the turn.")
    if error is not None:
        body = {"error": {"code": "agent_error", "message": error}, **result}
        return jsonify(body), 502
    return result


def _stream_turn(turn: Turn, artifact_links):
    def generate():
        events = run_turn(turn, artifact_links)
        try:
            for event in events:
                yield event.to_sse()
        finally:
            # Deterministically unwind the turn (persist + deregister) when
            # the client hangs up, rather than whenever it is collected.
            events.close()
        result = _turn_result(turn)
        if result is not None:
            yield StreamEvent("message", result).to_sse()

    return make_sse_response(stream_with_context(generate()))


@api_bp.get("/threads/<thread_id>/artifacts/<artifact_id>")
def get_artifact(thread_id: str, artifact_id: str):
    """Serve an artifact file, confined to the thread's workspace exactly as
    the web UI's download route is (a record pointing outside it is a 403)."""
    _owned_thread(thread_id)
    state = get_state(current_app)
    store = state.store
    artifact = store.get_artifact(artifact_id) if hasattr(store, "get_artifact") else None
    if artifact is None or artifact.get("thread_id") != thread_id:
        raise ApiError(404, "artifact_not_found", "Artifact not found.")

    workspace_root = confine.workspace(
        _username(), thread_id, base=state.config.workspace_root()
    )
    try:
        confined = confine.resolve(workspace_root, artifact.get("path"))
    except confine.OutsideWorkspaceError:
        raise ApiError(403, "artifact_outside_workspace", "Artifact path is outside the workspace.") from None
    if not confined.is_file():
        raise ApiError(404, "artifact_missing", "Artifact file is missing.")

    mimetype = (
        artifact.get("mime")
        or mimetypes.guess_type(confined.name)[0]
        or "application/octet-stream"
    )
    inline = (
        request.args.get("download") != "1"
        and artifact.get("kind") in _INLINE_ARTIFACT_KINDS
    )
    return send_file(
        confined, mimetype=mimetype, as_attachment=not inline, download_name=confined.name
    )


# ---------------------------------------------------------------------------
# Memories
# ---------------------------------------------------------------------------


@api_bp.get("/memories")
def list_memories():
    """List the caller's memories, or search them with ``?q=``.

    ``?all=1`` drops the relevance floor on a search (the web page's "show
    near misses").
    """
    memory = _memory_service()
    limit = _int_arg("limit", 25, 1, MAX_PAGE_SIZE)
    query = request.args.get("q", "").strip()
    if query:
        found = memory.search(
            _username(), query, limit=limit,
            min_signal=0 if request.args.get("all") == "1" else None,
        )
        return {
            "query": query,
            "memories": [_memory_json(hit.doc, hit) for hit in found.results],
            "confident": bool(found.confident),
            "advisory": found.advisory,
        }
    offset = _int_arg("offset", 0, 0, 10**9)
    docs = memory.list(_username(), statuses=tuple(VALID_STATUSES))
    return {
        "memories": [_memory_json(d) for d in docs[offset:offset + limit]],
        "total": len(docs),
        "limit": limit,
        "offset": offset,
    }


@api_bp.post("/memories")
def create_memory():
    memory = _memory_service()
    data = _json_body()
    doc = memory.create(
        owner=_username(),
        title=_string_field(data, "title", required=True, max_chars=MAX_TITLE_CHARS),
        summary=_string_field(data, "summary", required=True, max_chars=MAX_SUMMARY_CHARS),
        body=_string_field(data, "body", max_chars=MAX_CONTENT_CHARS, allow_empty=True) or "",
        tags=_tags_field(data) or [],
        status=_status_field(data) or "active",
    )
    return _memory_json(doc), 201


@api_bp.get("/memories/<memory_id>")
def get_memory(memory_id: str):
    doc = _memory_service().get(memory_id, owner=_username())
    if doc is None:
        raise ApiError(404, "memory_not_found", "Memory not found.")
    return _memory_json(doc)


@api_bp.patch("/memories/<memory_id>")
def update_memory(memory_id: str):
    memory = _memory_service()
    data = _json_body()
    fields = {
        "title": _string_field(data, "title", max_chars=MAX_TITLE_CHARS),
        "summary": _string_field(data, "summary", max_chars=MAX_SUMMARY_CHARS),
        "body": _string_field(data, "body", max_chars=MAX_CONTENT_CHARS, allow_empty=True),
        "tags": _tags_field(data),
        "status": _status_field(data),
    }
    doc = memory.update(memory_id, owner=_username(), editor=_username(), **fields)
    if doc is None:
        raise ApiError(404, "memory_not_found", "Memory not found.")
    return _memory_json(doc)


@api_bp.delete("/memories/<memory_id>")
def delete_memory(memory_id: str):
    if not _memory_service().delete(memory_id, owner=_username()):
        raise ApiError(404, "memory_not_found", "Memory not found.")
    return "", 204
