"""Tests for the web UI: login, chat, thread history.

These use the Flask test client to verify:
* Login page renders, correct login succeeds, wrong password fails.
* Unauthenticated requests redirect to /login.
* Authenticated chat: ask a question (fake agent), see the answer.
* Thread history sidebar lists prior threads.
* New thread creation.

All offline: the agent is a fake (``state.agent``), the store is a temp
moofile store, and no network is touched.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from pengyplexity.app import create_app, get_state
from pengyplexity.config import Config
from pengyplexity.core.auth import create_admin
from pengyplexity.core.agent import AgentResult
from pengyplexity.core.modelclient import FakeModelClient, ChatResponse


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg(tmp_path):
    return Config(
        data_dir=tmp_path / "data",
        store_path=tmp_path / "data" / "store.bson",
        model_base="http://127.0.0.1:0/v1",
        model_key="test",
        secret_key="test-secret",
    )


class FakeAgent:
    """A minimal agent fake that returns a canned answer."""

    def __init__(self, answer="This is a test answer.", sources=None):
        self.answer = answer
        self.sources = sources or []
        self.calls: list[dict] = []

    def run(self, user_message: str, history=None) -> AgentResult:
        self.calls.append({"message": user_message, "history": history})
        return AgentResult(
            answer=self.answer,
            sources=self.sources,
            iterations=1,
            messages=[{"role": "user", "content": user_message}],
        )


@pytest.fixture
def app_with_agent(cfg):
    """An app with a temp store, a bootstrap admin, and a fake agent."""
    app = create_app(cfg=cfg)
    state = get_state(app)

    # Bootstrap an admin user.
    create_admin(state.store, "admin", "admin-pass")

    # Inject a fake agent.
    state.agent = FakeAgent(answer="Cats are mammals.")

    return app


@pytest.fixture
def client(app_with_agent):
    with app_with_agent.test_client() as c:
        yield c


@pytest.fixture
def logged_in_client(app_with_agent):
    """A test client that is already logged in as admin."""
    with app_with_agent.test_client() as c:
        c.post("/login", data={"username": "admin", "password": "admin-pass"})
        yield c


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------


class TestLogin:
    def test_login_page_renders(self, client):
        resp = client.get("/login")
        assert resp.status_code == 200
        assert b"Pengyplexity" in resp.data
        assert b"username" in resp.data

    def test_login_success_redirects(self, client):
        resp = client.post(
            "/login",
            data={"username": "admin", "password": "admin-pass"},
            follow_redirects=False,
        )
        assert resp.status_code == 302
        assert "/chat" in resp.headers["Location"]

    def test_login_wrong_password(self, client):
        resp = client.post(
            "/login",
            data={"username": "admin", "password": "wrong"},
        )
        assert resp.status_code == 200
        assert b"Invalid" in resp.data  # error message shown

    def test_login_unknown_user(self, client):
        resp = client.post(
            "/login",
            data={"username": "nobody", "password": "whatever"},
        )
        assert resp.status_code == 200
        assert b"Invalid" in resp.data

    def test_logout(self, logged_in_client):
        resp = logged_in_client.post("/logout")
        assert resp.status_code == 302
        # After logout, /chat redirects to /login.
        resp = logged_in_client.get("/chat")
        assert resp.status_code == 302
        assert "/login" in resp.headers["Location"]


# ---------------------------------------------------------------------------
# Login required
# ---------------------------------------------------------------------------


class TestLoginRequired:
    def test_chat_redirects_to_login_when_unauthenticated(self, client):
        resp = client.get("/chat", follow_redirects=False)
        assert resp.status_code == 302
        assert "/login" in resp.headers["Location"]

    def test_chat_new_redirects_to_login(self, client):
        resp = client.get("/chat/new", follow_redirects=False)
        assert resp.status_code == 302
        assert "/login" in resp.headers["Location"]

    def test_ask_redirects_to_login(self, client):
        resp = client.post("/chat/some-id/ask", data={"question": "hi"})
        assert resp.status_code == 302
        assert "/login" in resp.headers["Location"]


# ---------------------------------------------------------------------------
# Chat (authenticated)
# ---------------------------------------------------------------------------


class TestSidebarAdminLink:
    """The Workspace/Memories nav row shows an Admin link only for admins."""

    def test_admin_sees_admin_link(self, logged_in_client, app_with_agent):
        resp = logged_in_client.get("/chat/new", follow_redirects=True)
        assert resp.status_code == 200
        assert b"Admin" in resp.data

    def test_nonadmin_does_not_see_admin_link(self, app_with_agent):
        state = get_state(app_with_agent)
        admin_doc = state.store.get_user_by_username("admin")
        state.auth.create_user(admin_doc, "pleb", "pleb-pass", is_admin=False)
        with app_with_agent.test_client() as c:
            c.post("/login", data={"username": "pleb", "password": "pleb-pass"})
            resp = c.get("/chat/new", follow_redirects=True)
            assert resp.status_code == 200
            assert b"\xf0\x9f\x9b\xa0 Admin" not in resp.data


class TestAccountBackLink:
    def test_account_page_has_a_back_to_chat_link(self, logged_in_client):
        resp = logged_in_client.get("/account/password")
        assert resp.status_code == 200
        assert b"account-topnav-back" in resp.data
        assert b"Back to chat" in resp.data


class TestChat:
    def test_new_thread_created(self, logged_in_client, app_with_agent):
        resp = logged_in_client.get("/chat/new", follow_redirects=True)
        assert resp.status_code == 200
        assert b"chat" in resp.data.lower() or b"Ask" in resp.data

    def test_ask_question(self, logged_in_client, app_with_agent):
        # Create a thread.
        state = get_state(app_with_agent)
        thread = state.store.create_thread("admin", "Test thread")
        tid = thread["_id"]

        # Ask a question.
        resp = logged_in_client.post(
            f"/chat/{tid}/ask",
            data={"question": "What is a cat?"},
        )
        assert resp.status_code == 200

        # The response should contain the answer from the fake agent.
        assert b"Cats are mammals." in resp.data

        # The thread should have both messages.
        updated = state.store.get_thread(tid)
        assert len(updated["messages"]) == 2
        assert updated["messages"][0]["role"] == "user"
        assert updated["messages"][0]["content"] == "What is a cat?"
        assert updated["messages"][1]["role"] == "assistant"
        assert updated["messages"][1]["content"] == "Cats are mammals."

    def test_ask_with_sources(self, app_with_agent, cfg):
        """Agent returns sources; they appear in the rendered page."""
        state = get_state(app_with_agent)
        state.agent = FakeAgent(
            answer="Based on research.",
            sources=[{"title": "Cat Wiki", "url": "https://cats.example.com"}],
        )
        thread = state.store.create_thread("admin")
        tid = thread["_id"]

        with app_with_agent.test_client() as c:
            c.post("/login", data={"username": "admin", "password": "admin-pass"})
            resp = c.post(f"/chat/{tid}/ask", data={"question": "research cats"})
            assert resp.status_code == 200
            assert b"Cat Wiki" in resp.data
            assert b"https://cats.example.com" in resp.data

    def test_ask_empty_question_no_error(self, logged_in_client, app_with_agent):
        """Posting an empty question doesn't crash."""
        state = get_state(app_with_agent)
        thread = state.store.create_thread("admin")
        tid = thread["_id"]

        resp = logged_in_client.post(
            f"/chat/{tid}/ask",
            data={"question": ""},
        )
        assert resp.status_code == 200  # re-renders, no crash

    def test_ask_nonexistent_thread(self, logged_in_client):
        resp = logged_in_client.post(
            "/chat/no-such-id/ask",
            data={"question": "hello"},
        )
        assert resp.status_code == 404

    def test_cannot_access_other_users_thread(self, app_with_agent, cfg):
        """User A cannot view User B's thread."""
        state = get_state(app_with_agent)
        # Create a second user.
        create_admin(state.store, "bob", "bob-pass")
        # Bob creates a thread.
        bob_thread = state.store.create_thread("bob", "Bob's secret")
        bob_tid = bob_thread["_id"]

        # Alice (admin) tries to view Bob's thread.
        with app_with_agent.test_client() as c:
            c.post("/login", data={"username": "admin", "password": "admin-pass"})
            resp = c.get(f"/chat/{bob_tid}")
            assert resp.status_code == 404

    def test_ask_stream_returns_sse(self, app_with_agent, cfg):
        """Asking with Accept: text/event-stream returns an SSE stream and
        persists the answer."""
        state = get_state(app_with_agent)
        thread = state.store.create_thread("admin")
        tid = thread["_id"]

        with app_with_agent.test_client() as c:
            c.post("/login", data={"username": "admin", "password": "admin-pass"})
            resp = c.post(
                f"/chat/{tid}/ask",
                data={"question": "stream me"},
                headers={"Accept": "text/event-stream"},
            )
            assert resp.status_code == 200
            assert resp.content_type.startswith("text/event-stream")
            body = resp.get_data(as_text=True)
            assert "event: activity" in body
            assert "event: token" in body
            assert "event: done" in body

        # The assistant answer was persisted from the streamed tokens.
        updated = state.store.get_thread(tid)
        assert len(updated["messages"]) == 2
        assert updated["messages"][1]["role"] == "assistant"
        assert updated["messages"][1]["content"] == "Cats are mammals."

    def test_ask_stream_via_query_param(self, app_with_agent, cfg):
        """?stream=1 also selects the SSE path."""
        state = get_state(app_with_agent)
        thread = state.store.create_thread("admin")
        tid = thread["_id"]

        with app_with_agent.test_client() as c:
            c.post("/login", data={"username": "admin", "password": "admin-pass"})
            resp = c.post(
                f"/chat/{tid}/ask?stream=1",
                data={"question": "stream me"},
            )
            assert resp.status_code == 200
            assert resp.content_type.startswith("text/event-stream")

    def test_ask_stream_persists_sources(self, app_with_agent, cfg):
        """Sources collected during streaming are persisted on the answer."""
        state = get_state(app_with_agent)
        src = [{"title": "Cat Wiki", "url": "https://cats.example.com"}]
        state.agent = FakeAgent(answer="Cats are mammals.", sources=src)
        thread = state.store.create_thread("admin")
        tid = thread["_id"]

        with app_with_agent.test_client() as c:
            c.post("/login", data={"username": "admin", "password": "admin-pass"})
            resp = c.post(
                f"/chat/{tid}/ask?stream=1",
                data={"question": "stream me"},
            )
            assert resp.status_code == 200
            body = resp.get_data(as_text=True)
            # The done event carries the sources.
            assert "https://cats.example.com" in body


# ---------------------------------------------------------------------------
# Thread history sidebar
# ---------------------------------------------------------------------------


class TestThreadHistory:
    def test_sidebar_lists_threads(self, logged_in_client, app_with_agent):
        state = get_state(app_with_agent)
        state.store.create_thread("admin", "Thread One")
        state.store.create_thread("admin", "Thread Two")

        # View any thread — the sidebar should list all.
        threads = state.store.list_threads_for_user("admin")
        resp = logged_in_client.get(f"/chat/{threads[0]['_id']}")
        assert resp.status_code == 200
        assert b"Thread One" in resp.data
        assert b"Thread Two" in resp.data

    def test_chat_index_redirects_to_most_recent(self, logged_in_client, app_with_agent):
        state = get_state(app_with_agent)
        t1 = state.store.create_thread("admin", "First")
        t2 = state.store.create_thread("admin", "Second")

        resp = logged_in_client.get("/chat", follow_redirects=False)
        assert resp.status_code == 302
        # Should redirect to one of the threads.
        assert "/chat/" in resp.headers["Location"]


class TestThreadDelete:
    def test_delete_thread_removes_it(self, logged_in_client, app_with_agent):
        state = get_state(app_with_agent)
        thread = state.store.create_thread("admin", "To Delete")
        tid = thread["_id"]

        resp = logged_in_client.post(f"/chat/{tid}/delete")
        assert resp.status_code == 302
        assert state.store.get_thread(tid) is None
        assert state.store.list_threads_for_user("admin") == []

    def test_delete_redirects_to_remaining_thread(self, logged_in_client, app_with_agent):
        state = get_state(app_with_agent)
        keep = state.store.create_thread("admin", "Keep")
        drop = state.store.create_thread("admin", "Drop")

        resp = logged_in_client.post(f"/chat/{drop['_id']}/delete")
        assert resp.status_code == 302
        assert keep["_id"] in resp.headers["Location"]

    def test_cannot_delete_other_users_thread(self, app_with_agent, cfg):
        state = get_state(app_with_agent)
        create_admin(state.store, "bob", "bob-pass")
        bob_thread = state.store.create_thread("bob", "Bob's")
        with app_with_agent.test_client() as c:
            c.post("/login", data={"username": "admin", "password": "admin-pass"})
            resp = c.post(f"/chat/{bob_thread['_id']}/delete")
            assert resp.status_code == 404
            assert state.store.get_thread(bob_thread["_id"]) is not None

    def test_delete_button_present_on_chat_page(self, logged_in_client, app_with_agent):
        state = get_state(app_with_agent)
        thread = state.store.create_thread("admin", "Named")
        state.store.append_message(thread["_id"], "user", "hello")
        resp = logged_in_client.get(f"/chat/{thread['_id']}")
        assert b"delete" in resp.data.lower()


class TestBlankThreadGuard:
    def test_new_reuses_existing_blank_thread(self, logged_in_client, app_with_agent):
        state = get_state(app_with_agent)
        blank = state.store.create_thread("admin")  # no messages
        resp = logged_in_client.get("/chat/new", follow_redirects=False)
        assert resp.status_code == 302
        assert blank["_id"] in resp.headers["Location"]
        # No extra thread was created.
        assert len(state.store.list_threads_for_user("admin")) == 1

    def test_new_creates_when_no_blank(self, logged_in_client, app_with_agent):
        state = get_state(app_with_agent)
        non_blank = state.store.create_thread("admin", "Used")
        state.store.append_message(non_blank["_id"], "user", "hi")
        before = len(state.store.list_threads_for_user("admin"))
        resp = logged_in_client.get("/chat/new", follow_redirects=False)
        assert resp.status_code == 302
        # A new thread was created (there was no blank one).
        assert len(state.store.list_threads_for_user("admin")) == before + 1


class TestThreadNaming:
    def test_question_names_thread(self, logged_in_client, app_with_agent):
        state = get_state(app_with_agent)
        thread = state.store.create_thread("admin")  # default title

        logged_in_client.post(
            f"/chat/{thread['_id']}/ask",
            data={"question": "What is the capital of France?"},
        )
        updated = state.store.get_thread(thread["_id"])
        # Title should no longer be the default "New Thread".
        assert updated["title"] != "New Thread"
        assert "capital" in updated["title"].lower() or "France" in updated["title"].lower()

    def test_already_named_thread_not_renamed(self, logged_in_client, app_with_agent):
        state = get_state(app_with_agent)
        thread = state.store.create_thread("admin", "My Custom Title")
        logged_in_client.post(
            f"/chat/{thread['_id']}/ask",
            data={"question": "Totally different question"},
        )
        updated = state.store.get_thread(thread["_id"])
        assert updated["title"] == "My Custom Title"



# ---------------------------------------------------------------------------
# Page structure
# ---------------------------------------------------------------------------


class TestPageStructure:
    def test_chat_page_has_input_box(self, logged_in_client, app_with_agent):
        state = get_state(app_with_agent)
        thread = state.store.create_thread("admin")
        resp = logged_in_client.get(f"/chat/{thread['_id']}")
        assert b"textarea" in resp.data or b"Ask" in resp.data

    def test_chat_page_has_sidebar(self, logged_in_client, app_with_agent):
        state = get_state(app_with_agent)
        thread = state.store.create_thread("admin")
        resp = logged_in_client.get(f"/chat/{thread['_id']}")
        assert b"sidebar" in resp.data

    def test_login_shows_error_on_failure(self, client):
        resp = client.post("/login", data={"username": "x", "password": "y"})
        assert b"alert" in resp.data or b"Invalid" in resp.data

# ---------------------------------------------------------------------------
# Admin UI
# ---------------------------------------------------------------------------


class TestAdminRequiresLogin:
    def test_admin_page_redirects_when_unauthenticated(self, client):
        resp = client.get("/admin", follow_redirects=False)
        assert resp.status_code == 302
        assert "/login" in resp.headers["Location"]

    def test_admin_create_redirects_when_unauthenticated(self, client):
        resp = client.post("/admin/users", data={"username": "x", "password": "y"})
        assert resp.status_code == 302
        assert "/login" in resp.headers["Location"]


class TestAdminBlockedForNonAdmin:
    """A non-admin user is logged in but should get 403 on admin routes."""

    @pytest.fixture
    def app_with_nonadmin(self, cfg):
        app = create_app(cfg=cfg)
        state = get_state(app)
        create_admin(state.store, "admin", "admin-pass")
        # Create a non-admin user via the auth service (proper password hash).
        admin_doc = state.store.get_user_by_username("admin")
        state.auth.create_user(admin_doc, "pleb", "pleb-pass", is_admin=False)
        state.agent = FakeAgent()
        return app

    @pytest.fixture
    def nonadmin_client(self, app_with_nonadmin):
        with app_with_nonadmin.test_client() as c:
            # Login as the non-admin user.
            c.post("/login", data={"username": "pleb", "password": "pleb-pass"})
            yield c

    def test_nonadmin_gets_403_on_list(self, nonadmin_client):
        resp = nonadmin_client.get("/admin")
        assert resp.status_code == 403

    def test_nonadmin_gets_403_on_create(self, nonadmin_client):
        resp = nonadmin_client.post(
            "/admin/users", data={"username": "hacker", "password": "x"}
        )
        assert resp.status_code == 403


class TestAdminListUsers:
    def test_admin_lists_users(self, logged_in_client, app_with_agent):
        state = get_state(app_with_agent)
        # Add a second user.
        state.auth.create_user(
            state.store.get_user_by_username("admin"), "alice", "alice-pass"
        )
        resp = logged_in_client.get("/admin")
        assert resp.status_code == 200
        assert b"admin" in resp.data
        assert b"alice" in resp.data

    def test_admin_page_shows_roles(self, logged_in_client, app_with_agent):
        state = get_state(app_with_agent)
        state.auth.create_user(
            state.store.get_user_by_username("admin"), "bob", "bob-pass", is_admin=True
        )
        resp = logged_in_client.get("/admin")
        assert b"Admin" in resp.data
        assert b"User" in resp.data


class TestAdminCreateUser:
    def test_create_user_success(self, logged_in_client, app_with_agent):
        resp = logged_in_client.post(
            "/admin/users",
            data={"username": "newuser", "password": "newpass"},
            follow_redirects=False,
        )
        assert resp.status_code == 302
        assert "success" in resp.headers["Location"]

        # Verify the user was created.
        state = get_state(app_with_agent)
        user = state.store.get_user_by_username("newuser")
        assert user is not None
        assert user["username"] == "newuser"
        assert user["is_admin"] is False

    def test_create_admin_user(self, logged_in_client, app_with_agent):
        resp = logged_in_client.post(
            "/admin/users",
            data={"username": "newadmin", "password": "pass", "is_admin": "on"},
        )
        assert resp.status_code == 302

        state = get_state(app_with_agent)
        user = state.store.get_user_by_username("newadmin")
        assert user["is_admin"] is True

    def test_create_duplicate_user_shows_error(self, logged_in_client, app_with_agent):
        resp = logged_in_client.post(
            "/admin/users",
            data={"username": "admin", "password": "x"},
            follow_redirects=True,
        )
        assert resp.status_code == 200
        assert b"already exists" in resp.data

    def test_create_empty_username_shows_error(self, logged_in_client):
        resp = logged_in_client.post(
            "/admin/users",
            data={"username": "", "password": "x"},
            follow_redirects=True,
        )
        assert resp.status_code == 200
        assert b"required" in resp.data


class TestAdminEnableDisable:
    @pytest.fixture
    def app_with_users(self, cfg):
        app = create_app(cfg=cfg)
        state = get_state(app)
        create_admin(state.store, "admin", "admin-pass")
        state.auth.create_user(
            state.store.get_user_by_username("admin"), "target", "target-pass"
        )
        state.agent = FakeAgent()
        return app

    def test_disable_user(self, app_with_users):
        state = get_state(app_with_users)
        target = state.store.get_user_by_username("target")

        with app_with_users.test_client() as c:
            c.post("/login", data={"username": "admin", "password": "admin-pass"})
            resp = c.post(f"/admin/users/{target['_id']}/disable", follow_redirects=False)
            assert resp.status_code == 302

            # Verify user is now disabled.
            updated = state.store.get_user_by_id(target["_id"])
            assert updated["enabled"] is False

    def test_enable_user(self, app_with_users):
        state = get_state(app_with_users)
        target = state.store.get_user_by_username("target")
        # First disable, then enable.
        state.auth.set_enabled(
            state.store.get_user_by_username("admin"), target["_id"], False
        )
        assert state.store.get_user_by_id(target["_id"])["enabled"] is False

        with app_with_users.test_client() as c:
            c.post("/login", data={"username": "admin", "password": "admin-pass"})
            resp = c.post(f"/admin/users/{target['_id']}/enable", follow_redirects=False)
            assert resp.status_code == 302

            updated = state.store.get_user_by_id(target["_id"])
            assert updated["enabled"] is True

    def test_disable_nonexistent_user(self, logged_in_client):
        resp = logged_in_client.post(
            "/admin/users/nonexistent-id/disable", follow_redirects=True
        )
        assert resp.status_code == 200
        assert b"not found" in resp.data


class TestAdminResetPassword:
    def test_reset_password_success(self, logged_in_client, app_with_agent):
        state = get_state(app_with_agent)
        user_doc = state.auth.create_user(
            state.store.get_user_by_username("admin"), "victim", "old-pass"
        )

        resp = logged_in_client.post(
            f"/admin/users/{user_doc['_id']}/reset-password",
            data={"new_password": "brand-new-pass"},
        )
        assert resp.status_code == 302

        # Verify the user can now log in with the new password.
        with app_with_agent.test_client() as c:
            resp2 = c.post(
                "/login",
                data={"username": "victim", "password": "brand-new-pass"},
                follow_redirects=False,
            )
            assert resp2.status_code == 302
            assert "/chat" in resp2.headers["Location"]

    def test_reset_password_empty_shows_error(self, logged_in_client, app_with_agent):
        state = get_state(app_with_agent)
        user_doc = state.auth.create_user(
            state.store.get_user_by_username("admin"), "victim", "pass"
        )
        resp = logged_in_client.post(
            f"/admin/users/{user_doc['_id']}/reset-password",
            data={"new_password": ""},
            follow_redirects=True,
        )
        assert resp.status_code == 200
        assert b"required" in resp.data


class TestAdminDeleteUser:
    def test_delete_user_success(self, logged_in_client, app_with_agent):
        state = get_state(app_with_agent)
        user_doc = state.auth.create_user(
            state.store.get_user_by_username("admin"), "doomed", "pass"
        )
        uid = user_doc["_id"]

        resp = logged_in_client.post(f"/admin/users/{uid}/delete", follow_redirects=False)
        assert resp.status_code == 302

        # Verify user is gone.
        assert state.store.get_user_by_username("doomed") is None

    def test_cannot_delete_self(self, logged_in_client, app_with_agent):
        state = get_state(app_with_agent)
        admin_doc = state.store.get_user_by_username("admin")

        resp = logged_in_client.post(
            f"/admin/users/{admin_doc['_id']}/delete",
            follow_redirects=True,
        )
        assert resp.status_code == 200
        assert b"Cannot delete" in resp.data
        # Admin still exists.
        assert state.store.get_user_by_username("admin") is not None

    def test_delete_nonexistent_user(self, logged_in_client):
        resp = logged_in_client.post(
            "/admin/users/fake-id/delete", follow_redirects=True
        )
        assert resp.status_code == 200
        assert b"not found" in resp.data


# ---------------------------------------------------------------------------
# Admin settings: system message, model connection, agent/tool/sandbox limits
# ---------------------------------------------------------------------------


class TestAdminSettings:
    def test_requires_admin(self, client, app_with_agent):
        resp = client.get("/admin/settings", follow_redirects=False)
        assert resp.status_code == 302
        assert "/login" in resp.headers["Location"]

    def test_nonadmin_gets_403(self, app_with_agent):
        state = get_state(app_with_agent)
        admin_doc = state.store.get_user_by_username("admin")
        state.auth.create_user(admin_doc, "pleb", "pleb-pass", is_admin=False)
        with app_with_agent.test_client() as c:
            c.post("/login", data={"username": "pleb", "password": "pleb-pass"})
            resp = c.get("/admin/settings")
            assert resp.status_code == 403

    def test_get_shows_config_defaults_as_placeholders(self, logged_in_client, app_with_agent, cfg):
        resp = logged_in_client.get("/admin/settings")
        assert resp.status_code == 200
        assert cfg.model_base.encode() in resp.data

    def test_save_and_reload_shows_saved_values(self, logged_in_client):
        resp = logged_in_client.post(
            "/admin/settings",
            data={
                "system_message": "You are TestBot.",
                "max_agent_iterations": "3",
                "tool_output_max_chars": "500",
            },
            follow_redirects=True,
        )
        assert resp.status_code == 200
        assert b"Settings saved" in resp.data
        assert b"You are TestBot." in resp.data

    def test_save_applies_to_shared_agent_on_next_turn(self, logged_in_client, app_with_agent):
        """Saved settings are read by _apply_effective_settings before a turn —
        exercised directly here since FakeAgent stands in for the real Agent
        in most fixtures (only lacks attrs _apply_effective_settings checks
        for via getattr/hasattr, so nothing breaks either way)."""
        from pengyplexity.core.agent import Agent
        from pengyplexity.core.modelclient import FakeModelClient
        from pengyplexity.web import _apply_effective_settings

        state = get_state(app_with_agent)
        real_agent = Agent(
            model=FakeModelClient(responses=[]),
            tool_executor=lambda name, args, ws: "",
            workspace=state.config.workspace_root(),
        )
        state.agent = real_agent

        logged_in_client.post(
            "/admin/settings", data={"system_message": "Custom prompt", "max_agent_iterations": "4"}
        )
        # Applied to the agent handling *this* request — agents are built per
        # request now, so the settings go to the one passed in rather than to
        # a process-wide singleton.
        _apply_effective_settings(state, state.new_agent())
        assert real_agent.system_prompt == "Custom prompt"
        assert real_agent.max_iterations == 4

    def test_each_request_gets_its_own_agent_and_tool_context(self, app_with_agent):
        """Two concurrent turns must not share per-turn state.

        The server is threaded. A single shared agent meant the second
        request repointed ``agent.workspace`` and the tool context's thread
        id / owner while the first turn was still running, so one user's
        remaining tool calls wrote into the other's workspace and their
        charts and memories were filed under the wrong thread and username.
        """
        state = get_state(app_with_agent)
        state.agent = None  # use the production factory, not the fixture fake

        first = state.new_agent()
        second = state.new_agent()

        assert first is not second
        assert first.tool_executor is not second.tool_executor
        assert first.tool_executor.context is not second.tool_executor.context

        first.workspace = Path("/ws/alice")
        first.tool_executor.context.owner = "alice"
        second.workspace = Path("/ws/bob")
        second.tool_executor.context.owner = "bob"

        assert first.workspace == Path("/ws/alice")
        assert first.tool_executor.context.owner == "alice"

    def test_injected_agent_still_overrides_the_factory(self, app_with_agent):
        state = get_state(app_with_agent)
        fake = FakeAgent(answer="hi")
        state.agent = fake

        assert state.new_agent() is fake
        assert state.new_agent() is fake

    def test_reset_clears_overrides(self, logged_in_client, app_with_agent):
        logged_in_client.post("/admin/settings", data={"system_message": "Temp override"})
        resp = logged_in_client.post("/admin/settings/reset", follow_redirects=True)
        assert resp.status_code == 200
        assert b"reset to defaults" in resp.data

        state = get_state(app_with_agent)
        assert state.store.get_settings()["values"] == {}

    def test_blank_field_falls_back_to_default(self, logged_in_client, app_with_agent, cfg):
        logged_in_client.post("/admin/settings", data={"model_base": "http://override.example/v1"})
        logged_in_client.post("/admin/settings", data={"model_base": ""})

        from pengyplexity.core.settings import effective_settings

        state = get_state(app_with_agent)
        eff = effective_settings(state.store, state.config)
        assert eff["model_base"] == cfg.model_base


# ---------------------------------------------------------------------------
# Artifacts: inline rendering + confined download route
# ---------------------------------------------------------------------------


def _artifact_block(page: bytes) -> bytes:
    """The rendered message area, with the page's inline <script> removed.

    That script builds artifact markup client-side for streamed turns, so it
    contains the literal text "<img" — an assertion over the whole page body
    can never tell a rendered artifact from the code that renders one.
    """
    import re

    return re.sub(rb"<script\b.*?</script>", b"", page, flags=re.DOTALL)


class TestArtifacts:
    """Artifacts render inline in the chat pane and are served through a
    download route that is *confined* to the thread's workspace.

    Everything is offline: the workspace is a temp dir under ``tmp_path``,
    the store is the real moofile store pointed there, and the files are
    plain bytes written by the test.
    """

    def _workspace_for(self, app, owner, thread_id) -> Path:
        state = get_state(app)
        from pengyplexity.sandbox import confine

        return confine.workspace(owner, thread_id, base=state.config.workspace_root())

    def _add_thread_with_messages(self, app, owner="admin"):
        state = get_state(app)
        thread = state.store.create_thread(owner)
        state.store.append_message(thread["_id"], "user", "Make me a chart")
        state.store.append_message(thread["_id"], "assistant", "Here is your chart")
        return thread

    def _record_artifact(self, app, thread, filename, kind, mime,
                         path, message_index=1):
        state = get_state(app)
        return state.store.create_artifact({
            "filename": filename,
            "path": str(path),
            "kind": kind,
            "mime": mime,
            "thread_id": thread["_id"],
            "message_index": message_index,
            "created": datetime.now(timezone.utc),
            "size_bytes": path.stat().st_size if path.exists() else 0,
        })

    def _write_workspace_file(self, app, owner, thread_id, filename, data=b"\x89PNG"):
        ws = self._workspace_for(app, owner, thread_id)
        ws.mkdir(parents=True, exist_ok=True)
        (ws / filename).write_bytes(data)
        return ws / filename

    def test_chart_rendered_inline_in_chat(self, logged_in_client, app_with_agent):
        thread = self._add_thread_with_messages(app_with_agent)
        path = self._write_workspace_file(
            app_with_agent, "admin", thread["_id"], "chart.png", b"PNGDATA"
        )
        self._record_artifact(
            app_with_agent, thread, "chart.png", "chart", "image/png", path
        )

        resp = logged_in_client.get(f"/chat/{thread['_id']}")
        assert resp.status_code == 200
        # An <img> pointing at the download route, plus the filename link.
        # Scoped to the artifact block: the page's own <script> mentions "<img"
        # as a string (the client-side Markdown renderer builds one), so a
        # whole-page substring check would pass on every page.
        assert b'class="artifact artifact-inline"' in resp.data
        assert _artifact_block(resp.data).count(b"<img") == 1
        assert b"chart.png" in resp.data
        assert f"/chat/{thread['_id']}/artifact/".encode() in resp.data

    def test_report_rendered_as_doc_link_not_img(self, logged_in_client, app_with_agent):
        thread = self._add_thread_with_messages(app_with_agent)
        path = self._write_workspace_file(
            app_with_agent, "admin", thread["_id"], "report.html", b"<html></html>"
        )
        self._record_artifact(
            app_with_agent, thread, "report.html", "report", "text/html", path
        )

        resp = logged_in_client.get(f"/chat/{thread['_id']}")
        assert resp.status_code == 200
        # Reports are not inlined as <img>; they're a doc link.
        assert b'class="artifact artifact-doc"' in resp.data
        assert b"<img" not in _artifact_block(resp.data)
        assert b"report.html" in resp.data
        assert b"report" in resp.data

    def test_no_artifacts_no_img(self, logged_in_client, app_with_agent):
        thread = self._add_thread_with_messages(app_with_agent)
        resp = logged_in_client.get(f"/chat/{thread['_id']}")
        assert resp.status_code == 200
        assert b'class="artifact' not in resp.data
        assert b"<img" not in _artifact_block(resp.data)

    def test_download_artifact_returns_bytes(self, logged_in_client, app_with_agent):
        thread = self._add_thread_with_messages(app_with_agent)
        path = self._write_workspace_file(
            app_with_agent, "admin", thread["_id"], "chart.png", b"PNGDATA"
        )
        art = self._record_artifact(
            app_with_agent, thread, "chart.png", "chart", "image/png", path
        )

        resp = logged_in_client.get(f"/chat/{thread['_id']}/artifact/{art['_id']}")
        assert resp.status_code == 200
        assert resp.data == b"PNGDATA"
        assert resp.headers["Content-Type"].startswith("image/png")

    def test_download_inline_image_not_attachment(self, logged_in_client, app_with_agent):
        thread = self._add_thread_with_messages(app_with_agent)
        path = self._write_workspace_file(
            app_with_agent, "admin", thread["_id"], "pic.png", b"IMG"
        )
        art = self._record_artifact(
            app_with_agent, thread, "pic.png", "image", "image/png", path
        )
        resp = logged_in_client.get(f"/chat/{thread['_id']}/artifact/{art['_id']}")
        assert resp.status_code == 200
        # Inline (chart/image) artifacts must NOT be a download attachment.
        cd = resp.headers.get("Content-Disposition", "")
        assert "attachment" not in cd

    def test_download_outside_workspace_rejected(self, logged_in_client, app_with_agent, tmp_path):
        """An artifact record whose path points OUTSIDE the thread workspace is
        rejected (403), even though it belongs to the thread. This is the
        escape-proof guard on the download route."""
        thread = self._add_thread_with_messages(app_with_agent)
        # A secret file outside any workspace.
        secret = tmp_path / "secret.png"
        secret.write_bytes(b"TOP-SECRET")
        art = self._record_artifact(
            app_with_agent, thread, "secret.png", "image", "image/png", secret
        )

        resp = logged_in_client.get(f"/chat/{thread['_id']}/artifact/{art['_id']}")
        assert resp.status_code == 403

    def test_download_dotdot_traversal_rejected(self, logged_in_client, app_with_agent):
        thread = self._add_thread_with_messages(app_with_agent)
        state = get_state(app_with_agent)
        ws = self._workspace_for(app_with_agent, "admin", thread["_id"])
        art = state.store.create_artifact({
            "filename": "esc.png",
            "path": "../../../../../etc/hostname",
            "kind": "image",
            "mime": "image/png",
            "thread_id": thread["_id"],
            "message_index": 1,
            "created": datetime.now(timezone.utc),
            "size_bytes": 0,
        })
        resp = logged_in_client.get(f"/chat/{thread['_id']}/artifact/{art['_id']}")
        assert resp.status_code == 403

    def test_download_other_users_thread_rejected(self, app_with_agent):
        state = get_state(app_with_agent)
        create_admin(state.store, "bob", "bob-pass")
        bob_thread = state.store.create_thread("bob")
        path = self._write_workspace_file(
            app_with_agent, "bob", bob_thread["_id"], "chart.png", b"BOB"
        )
        art = self._record_artifact(
            app_with_agent, bob_thread, "chart.png", "chart", "image/png", path
        )
        # Admin (owner of a different thread) can't read Bob's artifact.
        with app_with_agent.test_client() as c:
            c.post("/login", data={"username": "admin", "password": "admin-pass"})
            resp = c.get(f"/chat/{bob_thread['_id']}/artifact/{art['_id']}")
            assert resp.status_code == 404

    def test_download_nonexistent_artifact_404(self, logged_in_client, app_with_agent):
        thread = self._add_thread_with_messages(app_with_agent)
        resp = logged_in_client.get(f"/chat/{thread['_id']}/artifact/no-such-id")
        assert resp.status_code == 404

    def test_download_missing_file_404(self, logged_in_client, app_with_agent):
        thread = self._add_thread_with_messages(app_with_agent)
        # Record points inside the workspace, but the file was never written.
        ws = self._workspace_for(app_with_agent, "admin", thread["_id"])
        art = self._record_artifact(
            app_with_agent, thread, "ghost.png", "image", "image/png", ws / "ghost.png"
        )
        resp = logged_in_client.get(f"/chat/{thread['_id']}/artifact/{art['_id']}")
        assert resp.status_code == 404

    def test_download_requires_login(self, client, app_with_agent):
        thread = self._add_thread_with_messages(app_with_agent)
        art = self._record_artifact(
            app_with_agent, thread, "chart.png", "chart", "image/png",
            self._write_workspace_file(app_with_agent, "admin", thread["_id"], "chart.png"),
        )
        resp = client.get(
            f"/chat/{thread['_id']}/artifact/{art['_id']}", follow_redirects=False
        )
        assert resp.status_code == 302
        assert "/login" in resp.headers["Location"]


class TestMemoriesPages:
    """The memory notebook UI: list/search, create, edit, delete.

    ``state.memory`` is a real :class:`MemoryStore` even in ``app_with_agent``
    (only ``state.agent`` is swapped for a fake) since it's wired
    unconditionally in ``_wire_production_agent``.
    """

    def test_requires_login(self, client):
        resp = client.get("/memories", follow_redirects=False)
        assert resp.status_code == 302
        assert "/login" in resp.headers["Location"]

    def test_create_and_list(self, logged_in_client):
        resp = logged_in_client.post(
            "/memories",
            data={"title": "Likes cats", "summary": "Admin loves cats.", "tags": "pets, cats"},
        )
        assert resp.status_code == 200
        assert b"Likes cats" in resp.data
        assert b"Memory saved" in resp.data

    def test_create_requires_title_and_summary(self, logged_in_client):
        resp = logged_in_client.post("/memories", data={"title": "", "summary": ""})
        assert resp.status_code == 200
        assert b"required" in resp.data

    def test_search_finds_created_memory(self, logged_in_client):
        logged_in_client.post(
            "/memories", data={"title": "Deadline", "summary": "Project deadline is Friday.", "tags": ""}
        )
        resp = logged_in_client.get("/memories?q=deadline")
        assert resp.status_code == 200
        assert b"Deadline" in resp.data

    def test_memories_scoped_per_user(self, app_with_agent, cfg):
        from pengyplexity.core.auth import create_admin

        state = get_state(app_with_agent)
        create_admin(state.store, "bob", "bob-pass")

        with app_with_agent.test_client() as c:
            c.post("/login", data={"username": "admin", "password": "admin-pass"})
            c.post("/memories", data={"title": "Admin secret", "summary": "Only admin should see this."})

        with app_with_agent.test_client() as c:
            c.post("/login", data={"username": "bob", "password": "bob-pass"})
            resp = c.get("/memories")
            assert b"Admin secret" not in resp.data

    def test_edit_memory(self, logged_in_client, app_with_agent):
        state = get_state(app_with_agent)
        logged_in_client.post("/memories", data={"title": "Old", "summary": "Old summary."})
        mid = state.memory.list("admin")[0]["_id"]

        resp = logged_in_client.get(f"/memories/{mid}/edit")
        assert resp.status_code == 200
        assert b"Old" in resp.data

        resp = logged_in_client.post(
            f"/memories/{mid}/edit",
            data={"title": "New", "summary": "New summary.", "tags": "", "body": "", "status": "active"},
            follow_redirects=True,
        )
        assert resp.status_code == 200
        assert b"New" in resp.data

    def test_cannot_edit_other_users_memory(self, app_with_agent):
        from pengyplexity.core.auth import create_admin

        state = get_state(app_with_agent)
        create_admin(state.store, "bob", "bob-pass")
        doc = state.memory.create("bob", "Bob secret", "Only bob's.")

        with app_with_agent.test_client() as c:
            c.post("/login", data={"username": "admin", "password": "admin-pass"})
            resp = c.get(f"/memories/{doc['_id']}/edit")
            assert resp.status_code == 404

    def test_delete_memory(self, logged_in_client, app_with_agent):
        state = get_state(app_with_agent)
        logged_in_client.post("/memories", data={"title": "Temp", "summary": "Temporary."})
        mid = state.memory.list("admin")[0]["_id"]

        resp = logged_in_client.post(f"/memories/{mid}/delete", follow_redirects=True)
        assert resp.status_code == 200
        assert state.memory.get(mid) is None


class TestWorkspacePage:
    """The cross-thread artifact gallery + ZIP download."""

    def _add_thread_with_artifact(self, app, owner="admin", filename="chart.png"):
        state = get_state(app)
        from pengyplexity.sandbox import confine

        thread = state.store.create_thread(owner)
        ws = confine.workspace(owner, thread["_id"], base=state.config.workspace_root())
        ws.mkdir(parents=True, exist_ok=True)
        path = ws / filename
        path.write_bytes(b"\x89PNG-data")
        state.store.create_artifact({
            "filename": filename,
            "path": str(path),
            "kind": "chart",
            "mime": "image/png",
            "thread_id": thread["_id"],
            "message_index": 0,
            "created": datetime.now(timezone.utc),
            "size_bytes": path.stat().st_size,
        })
        return thread

    def test_requires_login(self, client):
        resp = client.get("/workspace", follow_redirects=False)
        assert resp.status_code == 302
        assert "/login" in resp.headers["Location"]

    def test_empty_workspace(self, logged_in_client):
        resp = logged_in_client.get("/workspace")
        assert resp.status_code == 200
        assert b"No charts" in resp.data

    def test_lists_artifact_across_threads(self, logged_in_client, app_with_agent):
        self._add_thread_with_artifact(app_with_agent)
        resp = logged_in_client.get("/workspace")
        assert resp.status_code == 200
        assert b"chart.png" in resp.data
        assert b"Download all" in resp.data

    def test_zip_all_contains_the_file(self, logged_in_client, app_with_agent):
        import io
        import zipfile

        self._add_thread_with_artifact(app_with_agent)
        resp = logged_in_client.get("/workspace/zip")
        assert resp.status_code == 200
        assert resp.headers["Content-Type"] == "application/zip"
        zf = zipfile.ZipFile(io.BytesIO(resp.data))
        names = zf.namelist()
        assert any(n.endswith("chart.png") for n in names)

    def test_zip_scoped_to_thread(self, logged_in_client, app_with_agent):
        import io
        import zipfile

        t1 = self._add_thread_with_artifact(app_with_agent, filename="a.png")
        self._add_thread_with_artifact(app_with_agent, filename="b.png")
        resp = logged_in_client.get(f"/workspace/zip?thread_id={t1['_id']}")
        assert resp.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(resp.data))
        names = zf.namelist()
        assert any(n.endswith("a.png") for n in names)
        assert not any(n.endswith("b.png") for n in names)

    def test_other_users_artifacts_not_included(self, app_with_agent):
        from pengyplexity.core.auth import create_admin

        state = get_state(app_with_agent)
        create_admin(state.store, "bob", "bob-pass")
        self._add_thread_with_artifact(app_with_agent, owner="admin", filename="admin.png")

        with app_with_agent.test_client() as c:
            c.post("/login", data={"username": "bob", "password": "bob-pass"})
            resp = c.get("/workspace")
            assert b"admin.png" not in resp.data


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))


# ---------------------------------------------------------------------------
# Stopping a turn
# ---------------------------------------------------------------------------


class _StoppedMidStreamAgent:
    """An agent that is stopped partway through writing its answer.

    Cancels its own token after the first chunk, which is exactly what a Stop
    press looks like from inside the turn: the flag flips while the generator
    is suspended on a yield.
    """

    def __init__(self, chunks=("partial ", "answer")):
        self.chunks = chunks
        self.cancel = None
        self.tool_executor = None
        self.workspace = None
        self._last_sources = [{"title": "T", "url": "https://example.com"}]

    def run_stream(self, user_message, history=None):
        from pengyplexity.core.agent import AgentStreamEvent

        yield AgentStreamEvent("token", {"content": self.chunks[0]})
        self.cancel.cancel()
        if self.cancel.cancelled:
            return
        yield AgentStreamEvent("token", {"content": self.chunks[1]})


class TestStopTurn:
    def _thread(self, app, owner="admin"):
        return get_state(app).store.create_thread(owner)

    def test_stop_requires_login(self, client, app_with_agent):
        thread = self._thread(app_with_agent)
        resp = client.post(f"/chat/{thread['_id']}/stop")
        assert resp.status_code == 401

    def test_stop_on_someone_elses_thread_is_404(self, logged_in_client, app_with_agent):
        thread = self._thread(app_with_agent, owner="someone-else")
        resp = logged_in_client.post(f"/chat/{thread['_id']}/stop")
        assert resp.status_code == 404

    def test_stop_with_nothing_running_is_not_an_error(self, logged_in_client, app_with_agent):
        thread = self._thread(app_with_agent)
        resp = logged_in_client.post(f"/chat/{thread['_id']}/stop")
        # The turn may have finished between the click and this request.
        assert resp.status_code == 200
        assert resp.get_json()["stopped"] is False

    def test_stop_cancels_the_running_turn(self, logged_in_client, app_with_agent):
        thread = self._thread(app_with_agent)
        state = get_state(app_with_agent)
        token = state.cancels.start("admin", thread["_id"])

        resp = logged_in_client.post(f"/chat/{thread['_id']}/stop")

        assert resp.get_json() == {"status": "ok", "stopped": True}
        assert token.cancelled is True

    def test_partial_answer_is_persisted_when_stopped(self, logged_in_client, app_with_agent):
        thread = self._thread(app_with_agent)
        thread_id = thread["_id"]
        state = get_state(app_with_agent)
        state.agent = _StoppedMidStreamAgent()

        resp = logged_in_client.post(
            f"/chat/{thread_id}/ask?stream=1",
            data={"question": "hello"},
            headers={"Accept": "text/event-stream"},
        )
        body = resp.get_data(as_text=True)
        assert "partial" in body

        messages = state.store.get_thread(thread_id)["messages"]
        assert [m["role"] for m in messages] == ["user", "assistant"]
        answer = messages[-1]["content"]
        # The text the model had already written is the user's — it must
        # survive the interruption rather than vanishing on reload.
        assert "partial" in answer
        assert "answer" not in answer
        assert "[Stopped]" in answer

    def test_turn_is_deregistered_when_it_finishes(self, logged_in_client, app_with_agent):
        thread = self._thread(app_with_agent)
        state = get_state(app_with_agent)

        resp = logged_in_client.post(
            f"/chat/{thread['_id']}/ask?stream=1",
            data={"question": "hello"},
            headers={"Accept": "text/event-stream"},
        )
        resp.get_data()

        assert state.cancels.active("admin", thread["_id"]) is None
