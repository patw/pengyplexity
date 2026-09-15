"""Tests for the JSON API (``/api/v1``) and the Account page's API key UI.

Offline, like the rest of the suite: a temp moofile store, fake agents, and
the Flask test client.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from pengyplexity.app import create_app, get_state
from pengyplexity.config import Config
from pengyplexity.core.agent import AgentResult, AgentStreamEvent
from pengyplexity.core.auth import create_admin
from pengyplexity.core.ratelimit import RateLimiter
from pengyplexity.core.streaming import collect_events
from pengyplexity.sandbox import confine

API = "/api/v1"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class StreamingAgent:
    """Streams a canned answer and cites a source, like the real Agent."""

    def __init__(self, chunks=("Cats ", "are ", "mammals."), sources=None):
        self.chunks = chunks
        self._last_sources = sources if sources is not None else [
            {"title": "Cat Wiki", "url": "https://cats.example.com"}
        ]
        self.calls = []
        self.cancel = None
        self.workspace = None

    def run_stream(self, user_message, history=None):
        self.calls.append({"message": user_message, "history": history})
        yield AgentStreamEvent("activity", {"label": "Searching the web…", "tool": "web_search"})
        for chunk in self.chunks:
            yield AgentStreamEvent("token", {"content": chunk})


class FailingAgent:
    cancel = None
    workspace = None

    def run(self, user_message, history=None):
        raise RuntimeError("model exploded")


class StoppedMidStreamAgent:
    """Presses Stop on itself after the first chunk."""

    cancel = None
    workspace = None
    _last_sources = []

    def run_stream(self, user_message, history=None):
        yield AgentStreamEvent("token", {"content": "partial "})
        self.cancel.cancel()
        if self.cancel.cancelled:
            return
        yield AgentStreamEvent("token", {"content": "never sent"})


class ChartAgent:
    """Produces a chart artifact through the tool context, as make_chart does."""

    def __init__(self, store):
        self.store = store
        self.tool_executor = SimpleNamespace(context=SimpleNamespace(artifacts=[]))
        self.cancel = None
        self.workspace = None
        self._last_sources = []

    def run_stream(self, user_message, history=None):
        ctx = self.tool_executor.context
        path = self.workspace / "chart.png"
        path.write_bytes(b"PNGDATA")
        doc = self.store.create_artifact({
            "filename": "chart.png",
            "path": str(path),
            "kind": "chart",
            "mime": "image/png",
            "thread_id": ctx.thread_id,
            "message_index": ctx.message_index,
            "created": datetime.now(timezone.utc),
            "size_bytes": 7,
        })
        ctx.artifacts.append(SimpleNamespace(
            _id=doc["_id"], filename="chart.png", kind="chart", mime="image/png"
        ))
        yield AgentStreamEvent("token", {"content": "Here is your chart."})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app(tmp_path):
    cfg = Config(
        data_dir=tmp_path / "data",
        store_path=tmp_path / "data" / "store.bson",
        model_base="http://127.0.0.1:0/v1",
        model_key="test",
        secret_key="test-secret",
    )
    app = create_app(cfg=cfg)
    state = get_state(app)
    admin = create_admin(state.store, "admin", "admin-pass")
    state.auth.create_user(admin, "alice", "alice-pass")
    state.agent = StreamingAgent()
    return app


@pytest.fixture
def state(app):
    return get_state(app)


@pytest.fixture
def client(app):
    with app.test_client() as c:
        yield c


def _key_for(state, username, name="test"):
    user = state.store.get_user_by_username(username)
    key, _ = state.api_keys.create(user, name)
    return key


@pytest.fixture
def alice_key(state):
    return _key_for(state, "alice")


@pytest.fixture
def auth(alice_key):
    return {"Authorization": f"Bearer {alice_key}"}


@pytest.fixture
def thread(state):
    return state.store.create_thread("alice")


def _error_code(resp):
    return resp.get_json()["error"]["code"]


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


class TestAuthentication:
    def test_missing_header_is_401_json(self, client):
        resp = client.get(f"{API}/me")
        assert resp.status_code == 401
        assert _error_code(resp) == "unauthorized"
        assert resp.headers["WWW-Authenticate"].startswith("Bearer")

    @pytest.mark.parametrize("header", ["Basic abc", "Bearer", "Bearer   ", "pgy_nope"])
    def test_malformed_header_is_401(self, client, header):
        resp = client.get(f"{API}/me", headers={"Authorization": header})
        assert resp.status_code == 401

    def test_unknown_key_is_401(self, client):
        from pengyplexity.core.apikeys import generate_key

        resp = client.get(f"{API}/me", headers={"Authorization": f"Bearer {generate_key()}"})
        assert resp.status_code == 401
        assert _error_code(resp) == "invalid_api_key"

    def test_valid_key(self, client, auth):
        resp = client.get(f"{API}/me", headers=auth)
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["user"]["username"] == "alice"
        assert body["user"]["is_admin"] is False
        assert body["api_key"]["name"] == "test"
        assert "key_hash" not in body["api_key"]
        assert body["limits"]["turns_per_minute"] == 30

    def test_scheme_is_case_insensitive(self, client, alice_key):
        resp = client.get(f"{API}/me", headers={"Authorization": f"bearer {alice_key}"})
        assert resp.status_code == 200

    def test_web_session_does_not_authenticate_the_api(self, client):
        client.post("/login", data={"username": "alice", "password": "alice-pass"})
        assert client.get("/chat", follow_redirects=False).status_code == 302  # logged in
        resp = client.get(f"{API}/me")
        assert resp.status_code == 401

    def test_disabled_user_is_locked_out(self, client, auth, state):
        alice = state.store.get_user_by_username("alice")
        state.store.set_user_enabled(alice["_id"], False)
        assert client.get(f"{API}/me", headers=auth).status_code == 401

    def test_deleted_user_is_locked_out(self, client, auth, state):
        admin = state.store.get_user_by_username("admin")
        alice = state.store.get_user_by_username("alice")
        state.auth.delete_user(admin, alice["_id"])
        assert client.get(f"{API}/me", headers=auth).status_code == 401

    def test_responses_are_not_cached_or_sniffed(self, client, auth):
        resp = client.get(f"{API}/me", headers=auth)
        assert resp.headers["Cache-Control"] == "no-store"
        assert resp.headers["X-Content-Type-Options"] == "nosniff"


class TestErrorShape:
    def test_unknown_api_route_is_json_404(self, client, auth):
        resp = client.get(f"{API}/nope", headers=auth)
        assert resp.status_code == 404
        assert _error_code(resp) == "not_found"

    def test_wrong_method_is_json_405(self, client, auth):
        resp = client.put(f"{API}/threads", headers=auth)
        assert resp.status_code == 405
        assert _error_code(resp) == "method_not_allowed"
        assert "GET" in resp.headers["Allow"]

    def test_non_api_404_is_untouched(self, client):
        resp = client.get("/definitely-not-a-page")
        assert resp.status_code == 404
        assert not resp.is_json

    def test_oversized_body_is_413(self, client, auth, thread):
        resp = client.post(
            f"{API}/threads/{thread['_id']}/messages",
            data="x" * (1024 * 1024 + 1),
            headers={**auth, "Content-Type": "application/json"},
        )
        assert resp.status_code == 413

    def test_unexpected_exception_is_json_500(self, client, auth, state, monkeypatch):
        def boom(owner):
            raise RuntimeError("store on fire")

        monkeypatch.setattr(state.store, "list_threads_for_user", boom)
        resp = client.get(f"{API}/threads", headers=auth)
        assert resp.status_code == 500
        assert _error_code(resp) == "internal_error"
        assert "store on fire" not in resp.get_data(as_text=True)


# ---------------------------------------------------------------------------
# Threads
# ---------------------------------------------------------------------------


class TestThreads:
    def test_create_list_get_rename_delete(self, client, auth, state):
        resp = client.post(f"{API}/threads", headers=auth)
        assert resp.status_code == 201
        created = resp.get_json()
        assert created["title"] == "New Thread"
        assert created["messages"] == []
        tid = created["id"]

        resp = client.post(f"{API}/threads", json={"title": "Research"}, headers=auth)
        assert resp.get_json()["title"] == "Research"

        listing = client.get(f"{API}/threads", headers=auth).get_json()
        assert listing["total"] == 2
        assert {t["title"] for t in listing["threads"]} == {"New Thread", "Research"}

        resp = client.patch(f"{API}/threads/{tid}", json={"title": "Renamed"}, headers=auth)
        assert resp.status_code == 200
        assert resp.get_json()["title"] == "Renamed"
        assert client.get(f"{API}/threads/{tid}", headers=auth).get_json()["title"] == "Renamed"

        assert client.delete(f"{API}/threads/{tid}", headers=auth).status_code == 204
        assert state.store.get_thread(tid) is None
        assert client.get(f"{API}/threads/{tid}", headers=auth).status_code == 404

    def test_create_never_reuses_a_blank_thread(self, client, auth):
        a = client.post(f"{API}/threads", headers=auth).get_json()["id"]
        b = client.post(f"{API}/threads", headers=auth).get_json()["id"]
        assert a != b

    def test_pagination(self, client, auth, state):
        for i in range(5):
            state.store.create_thread("alice", f"T{i}")
        page = client.get(f"{API}/threads?limit=2&offset=1", headers=auth).get_json()
        assert page["total"] == 5
        assert len(page["threads"]) == 2
        assert (page["limit"], page["offset"]) == (2, 1)

    @pytest.mark.parametrize("query", ["limit=0", "limit=9999", "limit=abc", "offset=-1"])
    def test_bad_pagination_is_400(self, client, auth, query):
        assert client.get(f"{API}/threads?{query}", headers=auth).status_code == 400

    @pytest.mark.parametrize("body", [{"title": ""}, {"title": 5}, {}, {"title": "x" * 201}])
    def test_rename_validation(self, client, auth, thread, body):
        resp = client.patch(f"{API}/threads/{thread['_id']}", json=body, headers=auth)
        assert resp.status_code == 400

    def test_other_users_threads_are_invisible(self, client, auth, state):
        theirs = state.store.create_thread("admin", "Admin's secret")
        tid = theirs["_id"]
        listing = client.get(f"{API}/threads", headers=auth).get_json()
        assert listing["total"] == 0
        for method, url, kwargs in [
            ("get", f"{API}/threads/{tid}", {}),
            ("patch", f"{API}/threads/{tid}", {"json": {"title": "pwned"}}),
            ("delete", f"{API}/threads/{tid}", {}),
            ("post", f"{API}/threads/{tid}/messages", {"json": {"content": "hi"}}),
            ("post", f"{API}/threads/{tid}/stop", {}),
            ("get", f"{API}/threads/{tid}/artifacts/whatever", {}),
        ]:
            resp = getattr(client, method)(url, headers=auth, **kwargs)
            assert resp.status_code == 404, (method, url)
            assert _error_code(resp) == "thread_not_found"
        after = state.store.get_thread(tid)
        assert after["title"] == "Admin's secret"
        assert after["messages"] == []


# ---------------------------------------------------------------------------
# Asking questions
# ---------------------------------------------------------------------------


class TestAsk:
    def test_non_streaming_turn(self, client, auth, state, thread):
        tid = thread["_id"]
        resp = client.post(
            f"{API}/threads/{tid}/messages", json={"content": "What is a cat?"}, headers=auth
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["stopped"] is False
        assert body["message"]["index"] == 1
        assert body["message"]["role"] == "assistant"
        assert body["message"]["content"] == "Cats are mammals."
        assert body["message"]["sources"] == [
            {"title": "Cat Wiki", "url": "https://cats.example.com"}
        ]
        # Named from the first question, like the web UI.
        assert body["thread"]["title"] == "What is a cat?"
        assert body["thread"]["busy"] is False

        stored = state.store.get_thread(tid)["messages"]
        assert [(m["role"], m["content"]) for m in stored] == [
            ("user", "What is a cat?"),
            ("assistant", "Cats are mammals."),
        ]
        assert state.cancels.active("alice", tid) is None

    def test_history_is_passed_on_follow_ups(self, client, auth, state, thread):
        url = f"{API}/threads/{thread['_id']}/messages"
        client.post(url, json={"content": "first"}, headers=auth)
        client.post(url, json={"content": "second"}, headers=auth)
        history = state.agent.calls[-1]["history"]
        assert [m["content"] for m in history] == ["first", "Cats are mammals."]

    def test_streaming_turn(self, client, auth, state, thread):
        tid = thread["_id"]
        resp = client.post(
            f"{API}/threads/{tid}/messages",
            json={"content": "What is a cat?", "stream": True},
            headers=auth,
        )
        assert resp.status_code == 200
        assert resp.content_type.startswith("text/event-stream")
        events = collect_events([resp.get_data(as_text=True)])
        kinds = [e["event"] for e in events]
        assert kinds[0] == "title"
        assert "activity" in kinds and "token" in kinds and "done" in kinds
        assert kinds[-1] == "message"
        tokens = "".join(e["data"]["content"] for e in events if e["event"] == "token")
        assert tokens == "Cats are mammals."
        final = events[-1]["data"]
        assert final["message"]["content"] == "Cats are mammals."
        assert final["stopped"] is False

        assert len(state.store.get_thread(tid)["messages"]) == 2
        assert state.cancels.active("alice", tid) is None

    def test_accept_header_selects_streaming(self, client, auth, thread):
        resp = client.post(
            f"{API}/threads/{thread['_id']}/messages",
            json={"content": "hi"},
            headers={**auth, "Accept": "text/event-stream"},
        )
        assert resp.content_type.startswith("text/event-stream")

    def test_explicit_stream_false_beats_accept_header(self, client, auth, thread):
        resp = client.post(
            f"{API}/threads/{thread['_id']}/messages",
            json={"content": "hi", "stream": False},
            headers={**auth, "Accept": "text/event-stream"},
        )
        assert resp.is_json

    @pytest.mark.parametrize("kwargs", [
        {"json": {}},
        {"json": {"content": ""}},
        {"json": {"content": "   "}},
        {"json": {"content": 42}},
        {"json": {"content": "hi", "stream": "yes"}},
        {"json": {"content": "x" * 100_001}},
        {"json": ["content"]},
        {"data": "{not json", "content_type": "application/json"},
        {},
    ])
    def test_validation_refuses_without_writing(self, client, auth, state, thread, kwargs):
        resp = client.post(f"{API}/threads/{thread['_id']}/messages", headers=auth, **kwargs)
        assert resp.status_code == 400
        assert _error_code(resp) in {"invalid_request", "invalid_json"}
        assert state.store.get_thread(thread["_id"])["messages"] == []

    def test_agent_error_is_502_and_leaves_no_blank_answer(self, client, auth, state, thread):
        state.agent = FailingAgent()
        resp = client.post(
            f"{API}/threads/{thread['_id']}/messages", json={"content": "q"}, headers=auth
        )
        assert resp.status_code == 502
        body = resp.get_json()
        assert body["error"]["code"] == "agent_error"
        assert "model exploded" in body["error"]["message"]
        assert body["message"] is None
        roles = [m["role"] for m in state.store.get_thread(thread["_id"])["messages"]]
        assert roles == ["user"]
        assert state.cancels.active("alice", thread["_id"]) is None

    def test_agent_unavailable_is_503_and_stores_nothing(self, client, auth, state, thread):
        state.agent = None
        state.agent_factory = None
        resp = client.post(
            f"{API}/threads/{thread['_id']}/messages", json={"content": "q"}, headers=auth
        )
        assert resp.status_code == 503
        assert _error_code(resp) == "agent_unavailable"
        assert state.store.get_thread(thread["_id"])["messages"] == []
        assert state.cancels.active("alice", thread["_id"]) is None

    def test_stopped_turn_keeps_partial_answer(self, client, auth, state, thread):
        state.agent = StoppedMidStreamAgent()
        resp = client.post(
            f"{API}/threads/{thread['_id']}/messages", json={"content": "q"}, headers=auth
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["stopped"] is True
        assert "partial" in body["message"]["content"]
        assert "[Stopped]" in body["message"]["content"]
        assert "never sent" not in body["message"]["content"]

    def test_artifacts_are_reported_and_downloadable(self, client, auth, state, thread):
        state.agent = ChartAgent(state.store)
        tid = thread["_id"]
        body = client.post(
            f"{API}/threads/{tid}/messages", json={"content": "chart please"}, headers=auth
        ).get_json()
        artifacts = body["message"]["artifacts"]
        assert len(artifacts) == 1
        art = artifacts[0]
        assert art["filename"] == "chart.png" and art["kind"] == "chart"

        resp = client.get(art["url"], headers=auth)
        assert resp.status_code == 200
        assert resp.data == b"PNGDATA"
        assert "attachment" not in resp.headers.get("Content-Disposition", "")

        resp = client.get(art["download_url"], headers=auth)
        assert "attachment" in resp.headers["Content-Disposition"]

        # The thread view attaches it to the same message.
        detail = client.get(f"{API}/threads/{tid}", headers=auth).get_json()
        assert detail["messages"][1]["artifacts"][0]["id"] == art["id"]

    def test_streaming_emits_artifact_events(self, client, auth, state, thread):
        state.agent = ChartAgent(state.store)
        resp = client.post(
            f"{API}/threads/{thread['_id']}/messages",
            json={"content": "chart", "stream": True},
            headers=auth,
        )
        events = collect_events([resp.get_data(as_text=True)])
        art = next(e["data"] for e in events if e["event"] == "artifact")
        assert art["url"].startswith(f"{API}/threads/{thread['_id']}/artifacts/")


class TestAdmission:
    def test_rate_limit_refuses_before_writing(self, client, auth, state, thread):
        state.api_rate_limiter = RateLimiter(1)
        url = f"{API}/threads/{thread['_id']}/messages"
        assert client.post(url, json={"content": "one"}, headers=auth).status_code == 200
        resp = client.post(url, json={"content": "two"}, headers=auth)
        assert resp.status_code == 429
        assert _error_code(resp) == "rate_limited"
        assert int(resp.headers["Retry-After"]) >= 1
        assert len(state.store.get_thread(thread["_id"])["messages"]) == 2

    def test_rate_limit_is_per_user(self, client, auth, state, thread):
        state.api_rate_limiter = RateLimiter(1)
        client.post(f"{API}/threads/{thread['_id']}/messages", json={"content": "a"}, headers=auth)
        admin_auth = {"Authorization": f"Bearer {_key_for(state, 'admin')}"}
        theirs = state.store.create_thread("admin")
        resp = client.post(
            f"{API}/threads/{theirs['_id']}/messages", json={"content": "b"}, headers=admin_auth
        )
        assert resp.status_code == 200

    def test_busy_thread_is_409_and_the_running_turn_survives(self, client, auth, state, thread):
        running = state.cancels.start("alice", thread["_id"])
        resp = client.post(
            f"{API}/threads/{thread['_id']}/messages", json={"content": "q"}, headers=auth
        )
        assert resp.status_code == 409
        assert _error_code(resp) == "turn_in_progress"
        assert running.cancelled is False
        assert state.store.get_thread(thread["_id"])["messages"] == []

    def test_concurrent_turn_cap(self, client, auth, state, thread):
        state.config.api_max_concurrent_turns = 1
        other = state.store.create_thread("alice")
        state.cancels.start("alice", other["_id"])
        resp = client.post(
            f"{API}/threads/{thread['_id']}/messages", json={"content": "q"}, headers=auth
        )
        assert resp.status_code == 429
        assert _error_code(resp) == "too_many_concurrent_turns"
        assert state.store.get_thread(thread["_id"])["messages"] == []

    def test_busy_flag_and_stop(self, client, auth, state, thread):
        tid = thread["_id"]
        assert client.post(f"{API}/threads/{tid}/stop", headers=auth).get_json() == {"stopped": False}

        token = state.cancels.start("alice", tid)
        assert client.get(f"{API}/threads/{tid}", headers=auth).get_json()["busy"] is True
        assert client.post(f"{API}/threads/{tid}/stop", headers=auth).get_json() == {"stopped": True}
        assert token.cancelled

    def test_deleting_a_thread_stops_its_turn(self, client, auth, state, thread):
        token = state.cancels.start("alice", thread["_id"])
        assert client.delete(f"{API}/threads/{thread['_id']}", headers=auth).status_code == 204
        assert token.cancelled


class TestArtifactConfinement:
    def _record(self, state, thread, path):
        return state.store.create_artifact({
            "filename": "x.png",
            "path": str(path),
            "kind": "image",
            "mime": "image/png",
            "thread_id": thread["_id"],
            "message_index": 1,
            "created": datetime.now(timezone.utc),
            "size_bytes": 1,
        })

    def test_path_outside_workspace_is_403(self, client, auth, state, thread, tmp_path):
        secret = tmp_path / "secret.png"
        secret.write_bytes(b"TOP-SECRET")
        art = self._record(state, thread, secret)
        resp = client.get(f"{API}/threads/{thread['_id']}/artifacts/{art['_id']}", headers=auth)
        assert resp.status_code == 403
        assert b"TOP-SECRET" not in resp.data

    def test_dotdot_path_is_403(self, client, auth, state, thread):
        art = self._record(state, thread, "../../../../../etc/hostname")
        resp = client.get(f"{API}/threads/{thread['_id']}/artifacts/{art['_id']}", headers=auth)
        assert resp.status_code == 403

    def test_artifact_from_another_thread_is_404(self, client, auth, state, thread):
        other = state.store.create_thread("alice")
        ws = confine.workspace("alice", other["_id"], base=state.config.workspace_root())
        ws.mkdir(parents=True)
        (ws / "x.png").write_bytes(b"X")
        art = self._record(state, other, ws / "x.png")
        resp = client.get(f"{API}/threads/{thread['_id']}/artifacts/{art['_id']}", headers=auth)
        assert resp.status_code == 404

    def test_missing_file_is_404(self, client, auth, state, thread):
        ws = confine.workspace("alice", thread["_id"], base=state.config.workspace_root())
        art = self._record(state, thread, ws / "ghost.png")
        resp = client.get(f"{API}/threads/{thread['_id']}/artifacts/{art['_id']}", headers=auth)
        assert resp.status_code == 404
        assert _error_code(resp) == "artifact_missing"


# ---------------------------------------------------------------------------
# Memories
# ---------------------------------------------------------------------------


class TestMemories:
    def test_crud(self, client, auth):
        resp = client.post(
            f"{API}/memories",
            json={"title": "Likes cats", "summary": "Alice loves cats.", "tags": ["Pets", "cats"]},
            headers=auth,
        )
        assert resp.status_code == 201
        mem = resp.get_json()
        assert mem["tags"] == ["pets", "cats"]
        assert mem["status"] == "active"
        mid = mem["id"]

        listing = client.get(f"{API}/memories", headers=auth).get_json()
        assert listing["total"] == 1

        resp = client.patch(
            f"{API}/memories/{mid}", json={"summary": "Alice adores cats.", "status": "superseded"},
            headers=auth,
        )
        assert resp.status_code == 200
        updated = resp.get_json()
        assert updated["summary"] == "Alice adores cats."
        assert updated["status"] == "superseded"
        assert updated["update_history"][0]["editor"] == "alice"
        assert updated["update_history"][0]["timestamp"].endswith("Z")

        assert client.get(f"{API}/memories/{mid}", headers=auth).status_code == 200
        assert client.delete(f"{API}/memories/{mid}", headers=auth).status_code == 204
        assert client.get(f"{API}/memories/{mid}", headers=auth).status_code == 404

    def test_search(self, client, auth):
        client.post(
            f"{API}/memories",
            json={"title": "Deadline", "summary": "Project deadline is Friday."},
            headers=auth,
        )
        body = client.get(f"{API}/memories?q=deadline", headers=auth).get_json()
        assert body["query"] == "deadline"
        assert any(m["title"] == "Deadline" for m in body["memories"])
        assert all("match" in m for m in body["memories"])

    @pytest.mark.parametrize("body", [
        {"summary": "no title"},
        {"title": "no summary"},
        {"title": "t", "summary": "s", "tags": "not-a-list"},
        {"title": "t", "summary": "s", "tags": [1, 2]},
        {"title": "t", "summary": "s", "status": "bogus"},
    ])
    def test_create_validation(self, client, auth, body):
        assert client.post(f"{API}/memories", json=body, headers=auth).status_code == 400

    def test_scoped_per_user(self, client, auth, state):
        doc = state.memory.create("admin", "Admin secret", "Only admin's.")
        mid = doc["_id"]
        assert client.get(f"{API}/memories", headers=auth).get_json()["total"] == 0
        assert client.get(f"{API}/memories/{mid}", headers=auth).status_code == 404
        assert client.patch(f"{API}/memories/{mid}", json={"title": "x"}, headers=auth).status_code == 404
        assert client.delete(f"{API}/memories/{mid}", headers=auth).status_code == 404
        assert state.memory.get(mid)["title"] == "Admin secret"


# ---------------------------------------------------------------------------
# Keys over the API
# ---------------------------------------------------------------------------


class TestKeysEndpoint:
    def test_list_marks_the_current_key(self, client, auth, state):
        _key_for(state, "alice", "second")
        keys = client.get(f"{API}/keys", headers=auth).get_json()["keys"]
        assert len(keys) == 2
        assert [k["name"] for k in keys if k["current"]] == ["test"]
        assert all("key_hash" not in k for k in keys)

    def test_a_key_can_revoke_itself(self, client, auth):
        me = client.get(f"{API}/me", headers=auth).get_json()
        resp = client.delete(f"{API}/keys/{me['api_key']['id']}", headers=auth)
        assert resp.status_code == 204
        assert client.get(f"{API}/me", headers=auth).status_code == 401

    def test_cannot_revoke_another_users_key(self, client, auth, state):
        admin = state.store.get_user_by_username("admin")
        _, record = state.api_keys.create(admin)
        resp = client.delete(f"{API}/keys/{record['_id']}", headers=auth)
        assert resp.status_code == 404
        assert state.store.get_api_key(record["_id"]) is not None

    def test_keys_cannot_be_created_over_the_api(self, client, auth):
        assert client.post(f"{API}/keys", json={"name": "x"}, headers=auth).status_code == 405


# ---------------------------------------------------------------------------
# Account page UI
# ---------------------------------------------------------------------------


class TestAccountApiKeys:
    @pytest.fixture
    def web(self, client):
        client.post("/login", data={"username": "alice", "password": "alice-pass"})
        return client

    def test_requires_login(self, client):
        resp = client.post("/account/api-keys", data={"name": "bot"})
        assert resp.status_code == 302
        assert "/login" in resp.headers["Location"]

    def test_create_shows_the_key_once_and_it_works(self, web, state):
        resp = web.post("/account/api-keys", data={"name": "discord-bot"})
        assert resp.status_code == 200
        assert resp.headers["Cache-Control"] == "no-store"
        page = resp.get_data(as_text=True)
        assert "discord-bot" in page
        import re

        key = re.search(r'id="new-api-key">(pgy_[A-Za-z0-9_-]+)<', page).group(1)

        # Not shown again.
        again = web.get("/account/password").get_data(as_text=True)
        assert key not in again
        assert key[:12] in again  # the display prefix is

        resp = web.get(f"{API}/me", headers={"Authorization": f"Bearer {key}"})
        assert resp.status_code == 200

    def test_revoke_from_the_account_page(self, web, state):
        key = _key_for(state, "alice", "old")
        alice = state.store.get_user_by_username("alice")
        key_id = state.api_keys.list_for_user(alice)[0]["_id"]
        resp = web.post(f"/account/api-keys/{key_id}/revoke")
        assert resp.status_code == 302
        assert web.get(f"{API}/me", headers={"Authorization": f"Bearer {key}"}).status_code == 401

    def test_cannot_revoke_someone_elses_key_from_the_page(self, web, state):
        key = _key_for(state, "admin")
        admin = state.store.get_user_by_username("admin")
        key_id = state.api_keys.list_for_user(admin)[0]["_id"]
        resp = web.post(f"/account/api-keys/{key_id}/revoke")
        assert "api_key_error" in resp.headers["Location"]
        assert state.api_keys.authenticate(key) is not None

    def test_account_page_lists_keys_without_secrets(self, web, state):
        key = _key_for(state, "alice", "listed-key")
        page = web.get("/account/password").get_data(as_text=True)
        assert "listed-key" in page
        assert key not in page
        assert "/api/v1" in page


class TestWebSharesTheTurnEngine:
    def test_web_stream_error_leaves_no_blank_answer(self, client, state):
        state.agent = FailingAgent()
        thread = state.store.create_thread("alice")
        client.post("/login", data={"username": "alice", "password": "alice-pass"})
        resp = client.post(f"/chat/{thread['_id']}/ask?stream=1", data={"question": "q"})
        assert "model exploded" in resp.get_data(as_text=True)
        roles = [m["role"] for m in state.store.get_thread(thread["_id"])["messages"]]
        assert roles == ["user"]
