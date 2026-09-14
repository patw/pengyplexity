"""Tests for :mod:`pengyplexity.core.store`.

These round-trip the moofile-backed store: create/read/update/delete on users,
threads (with embedded messages), shares, and settings. All paths point into
``tmp_path`` — no ``$HOME``, no network, no live services.
"""

from __future__ import annotations

import pytest

from pengyplexity.core.store import Store


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    """A Store backed by a temp directory."""
    s = Store(tmp_path / "store")
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Users CRUD
# ---------------------------------------------------------------------------


class TestUsers:
    def test_create_user_returns_doc_with_id(self, store):
        doc = store.create_user("alice", "hashed-pw-123")
        assert doc["username"] == "alice"
        assert doc["password_hash"] == "hashed-pw-123"
        assert doc["is_admin"] is False
        assert doc["enabled"] is True
        assert doc["created"] is not None
        assert doc["last_login"] is None
        assert "_id" in doc

    def test_create_admin_user(self, store):
        doc = store.create_user("admin", "pw", is_admin=True)
        assert doc["is_admin"] is True

    def test_get_user_by_username(self, store):
        store.create_user("bob", "pw")
        found = store.get_user_by_username("bob")
        assert found is not None
        assert found["username"] == "bob"

    def test_get_user_by_username_not_found(self, store):
        assert store.get_user_by_username("nobody") is None

    def test_get_user_by_id(self, store):
        doc = store.create_user("carol", "pw")
        found = store.get_user_by_id(doc["_id"])
        assert found is not None
        assert found["username"] == "carol"

    def test_list_users(self, store):
        store.create_user("u1", "pw")
        store.create_user("u2", "pw")
        users = store.list_users()
        assert len(users) == 2
        usernames = {u["username"] for u in users}
        assert usernames == {"u1", "u2"}

    def test_user_count(self, store):
        assert store.user_count() == 0
        store.create_user("a", "pw")
        store.create_user("b", "pw")
        assert store.user_count() == 2

    def test_update_user_fields(self, store):
        doc = store.create_user("dave", "pw")
        store.update_user(doc["_id"], last_login="some-timestamp")
        updated = store.get_user_by_id(doc["_id"])
        assert updated["last_login"] == "some-timestamp"

    def test_set_user_enabled(self, store):
        doc = store.create_user("erin", "pw")
        store.set_user_enabled(doc["_id"], False)
        updated = store.get_user_by_id(doc["_id"])
        assert updated["enabled"] is False
        store.set_user_enabled(doc["_id"], True)
        updated = store.get_user_by_id(doc["_id"])
        assert updated["enabled"] is True

    def test_set_user_password(self, store):
        doc = store.create_user("frank", "old-hash")
        store.set_user_password(doc["_id"], "new-hash")
        updated = store.get_user_by_id(doc["_id"])
        assert updated["password_hash"] == "new-hash"

    def test_touch_last_login(self, store):
        doc = store.create_user("gina", "pw")
        assert doc["last_login"] is None
        store.touch_last_login(doc["_id"])
        updated = store.get_user_by_id(doc["_id"])
        assert updated["last_login"] is not None

    def test_delete_user(self, store):
        doc = store.create_user("hank", "pw")
        assert store.delete_user(doc["_id"]) is True
        assert store.get_user_by_id(doc["_id"]) is None
        assert store.user_count() == 0

    def test_delete_nonexistent_user(self, store):
        assert store.delete_user("nonexistent-id") is False


# ---------------------------------------------------------------------------
# Threads (with embedded messages)
# ---------------------------------------------------------------------------


class TestThreads:
    def test_create_thread(self, store):
        doc = store.create_thread("alice", "My first thread")
        assert doc["owner"] == "alice"
        assert doc["title"] == "My first thread"
        assert doc["messages"] == []
        assert doc["created"] is not None
        assert doc["updated"] is not None
        assert "_id" in doc

    def test_create_thread_default_title(self, store):
        doc = store.create_thread("bob")
        assert doc["title"] == "New Thread"

    def test_get_thread(self, store):
        doc = store.create_thread("alice", "Find me")
        found = store.get_thread(doc["_id"])
        assert found is not None
        assert found["title"] == "Find me"

    def test_get_thread_not_found(self, store):
        assert store.get_thread("no-such-id") is None

    def test_list_threads_for_user(self, store):
        store.create_thread("alice", "A1")
        store.create_thread("alice", "A2")
        store.create_thread("bob", "B1")
        alice_threads = store.list_threads_for_user("alice")
        assert len(alice_threads) == 2
        bob_threads = store.list_threads_for_user("bob")
        assert len(bob_threads) == 1
        assert bob_threads[0]["title"] == "B1"

    def test_append_message(self, store):
        doc = store.create_thread("alice")
        tid = doc["_id"]
        updated = store.append_message(tid, "user", "Hello world")
        assert updated is not None
        assert len(updated["messages"]) == 1
        msg = updated["messages"][0]
        assert msg["role"] == "user"
        assert msg["type"] == "text"
        assert msg["content"] == "Hello world"
        assert msg["sources"] == []
        assert msg["created"] is not None

    def test_append_multiple_messages(self, store):
        doc = store.create_thread("alice")
        tid = doc["_id"]
        store.append_message(tid, "user", "Q1")
        store.append_message(tid, "assistant", "A1")
        store.append_message(tid, "user", "Q2")
        thread = store.get_thread(tid)
        assert len(thread["messages"]) == 3
        assert thread["messages"][0]["content"] == "Q1"
        assert thread["messages"][1]["content"] == "A1"
        assert thread["messages"][2]["content"] == "Q2"

    def test_append_message_with_sources(self, store):
        doc = store.create_thread("alice")
        tid = doc["_id"]
        sources = [{"title": "Example", "url": "https://example.com"}]
        updated = store.append_message(tid, "assistant", "Answer", sources=sources)
        msg = updated["messages"][0]
        assert msg["sources"] == sources

    def test_append_message_with_type(self, store):
        doc = store.create_thread("alice")
        tid = doc["_id"]
        store.append_message(tid, "assistant", "chart.png", msg_type="image")
        thread = store.get_thread(tid)
        assert thread["messages"][0]["type"] == "image"

    def test_append_message_to_nonexistent_thread(self, store):
        assert store.append_message("no-thread", "user", "hi") is None

    def test_delete_thread(self, store):
        doc = store.create_thread("alice")
        assert store.delete_thread(doc["_id"]) is True
        assert store.get_thread(doc["_id"]) is None

    def test_update_thread_title(self, store):
        doc = store.create_thread("alice")
        tid = doc["_id"]
        assert store.update_thread_title(tid, "Capital of France") is True
        assert store.get_thread(tid)["title"] == "Capital of France"

    def test_update_thread_title_nonexistent(self, store):
        assert store.update_thread_title("nope", "X") is False

    def test_thread_count(self, store):
        assert store.thread_count() == 0
        store.create_thread("a")
        store.create_thread("b")
        assert store.thread_count() == 2

    def test_messages_embedded_not_separate_collection(self, store):
        """The data model keeps messages embedded in the thread doc."""
        doc = store.create_thread("alice")
        tid = doc["_id"]
        store.append_message(tid, "user", "test")
        thread = store.get_thread(tid)
        # Messages are a list inside the thread document itself.
        assert isinstance(thread["messages"], list)


# ---------------------------------------------------------------------------
# Shares
# ---------------------------------------------------------------------------


class TestShares:
    def test_create_share(self, store):
        share = store.create_share("t123", 0, "text", "https://tclip.ca/abc")
        assert share["thread_id"] == "t123"
        assert share["message_index"] == 0
        assert share["kind"] == "text"
        assert share["url"] == "https://tclip.ca/abc"
        assert share["created"] is not None
        assert "_id" in share

    def test_get_shares_for_thread(self, store):
        store.create_share("t1", 0, "text", "https://tclip.ca/1")
        store.create_share("t1", 1, "image", "https://img.catbee.ca/2")
        store.create_share("t2", 0, "text", "https://tclip.ca/3")
        shares = store.get_shares_for_thread("t1")
        assert len(shares) == 2
        assert {s["kind"] for s in shares} == {"text", "image"}

    def test_get_share_by_id(self, store):
        share = store.create_share("t1", 0, "text", "https://tclip.ca/x")
        found = store.get_share_by_id(share["_id"])
        assert found is not None
        assert found["url"] == "https://tclip.ca/x"

    def test_delete_share(self, store):
        share = store.create_share("t1", 0, "text", "https://tclip.ca/d")
        assert store.delete_share(share["_id"]) is True
        assert store.get_share_by_id(share["_id"]) is None

    def test_share_count(self, store):
        assert store.share_count() == 0
        store.create_share("t1", 0, "text", "url1")
        assert store.share_count() == 1


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class TestSettings:
    def test_get_settings_returns_default(self, store):
        doc = store.get_settings()
        assert doc["_id"] == "global"
        assert doc.get("values", {}) == {}

    def test_set_and_get_setting(self, store):
        store.set_setting("research_budget", 10)
        assert store.get_setting("research_budget") == 10

    def test_set_setting_overwrites(self, store):
        store.set_setting("key", "v1")
        store.set_setting("key", "v2")
        assert store.get_setting("key") == "v2"

    def test_get_setting_default(self, store):
        assert store.get_setting("nonexistent", "fallback") == "fallback"

    def test_multiple_settings_independent(self, store):
        store.set_setting("a", 1)
        store.set_setting("b", "two")
        assert store.get_setting("a") == 1
        assert store.get_setting("b") == "two"

    def test_settings_persist_across_gets(self, store):
        store.set_setting("x", 42)
        doc = store.get_settings()
        assert doc["values"]["x"] == 42


# ---------------------------------------------------------------------------
# Persistence (round-trip: close and reopen)
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_users_persist(self, tmp_path):
        path = tmp_path / "store"
        s1 = Store(path)
        doc = s1.create_user("persist", "pw")
        s1.close()
        s2 = Store(path)
        found = s2.get_user_by_username("persist")
        assert found is not None
        assert found["_id"] == doc["_id"]
        s2.close()

    def test_threads_persist(self, tmp_path):
        path = tmp_path / "store"
        s1 = Store(path)
        doc = s1.create_thread("alice", "Persist me")
        tid = doc["_id"]
        s1.append_message(tid, "user", "hello")
        s1.close()
        s2 = Store(path)
        found = s2.get_thread(tid)
        assert found is not None
        assert found["title"] == "Persist me"
        assert len(found["messages"]) == 1
        assert found["messages"][0]["content"] == "hello"
        s2.close()

    def test_shares_persist(self, tmp_path):
        path = tmp_path / "store"
        s1 = Store(path)
        share = s1.create_share("t1", 0, "text", "https://tclip.ca/z")
        s1.close()
        s2 = Store(path)
        assert s2.share_count() == 1
        assert s2.get_share_by_id(share["_id"]) is not None
        s2.close()

    def test_context_manager(self, tmp_path):
        path = tmp_path / "store"
        with Store(path) as s:
            s.create_user("ctx", "pw")
        # After context exit, data is persisted.
        s2 = Store(path)
        assert s2.get_user_by_username("ctx") is not None
        s2.close()


# ---------------------------------------------------------------------------
# Multiple threads per user (data model sanity)
# ---------------------------------------------------------------------------


class TestDataModel:
    def test_full_flow(self, store):
        """Simulate: create user → create thread → ask → answer → share."""
        user = store.create_user("alice", "hashed")
        thread = store.create_thread(user["username"], "How does X work?")
        tid = thread["_id"]

        store.append_message(tid, "user", "How does X work?")
        store.append_message(
            tid, "assistant",
            "X works by doing Y.",
            sources=[{"title": "Source A", "url": "https://a.example"}],
        )

        share = store.create_share(tid, 1, "text", "https://tclip.ca/xyz")

        # Verify the thread has both messages.
        t = store.get_thread(tid)
        assert len(t["messages"]) == 2
        assert t["messages"][0]["role"] == "user"
        assert t["messages"][1]["role"] == "assistant"
        assert t["messages"][1]["sources"][0]["url"] == "https://a.example"

        # Verify the share links to the thread.
        shares = store.get_shares_for_thread(tid)
        assert len(shares) == 1
        assert shares[0]["url"] == "https://tclip.ca/xyz"
        assert shares[0]["message_index"] == 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
