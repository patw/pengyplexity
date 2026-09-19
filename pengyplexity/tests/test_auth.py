"""Tests for :mod:`pengyplexity.core.auth`.

These cover:
* Password hashing (werkzeug scrypt) — never plaintext, correct verify.
* ``AuthService.authenticate`` — correct credentials accepted, wrong
  password rejected, disabled user rejected.
* Admin-gated CRUD — admin can create/list/enable/disable/reset/delete
  users; non-admin is blocked with ``NotAdminError``.
* ``create_admin`` bootstrap — creates the first admin user.

All tests use a temp moofile store — no network, no Flask request context.
"""

from __future__ import annotations

import pytest

from pengyplexity.core.auth import (
    AuthError,
    AuthService,
    InvalidCredentialsError,
    NotAdminError,
    UserDisabledError,
    UserExistsError,
    UserNotFoundError,
    create_admin,
    hash_password,
    verify_password,
)
from pengyplexity.core.store import Store


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "store")
    yield s
    s.close()


@pytest.fixture
def auth(store):
    return AuthService(store=store)


@pytest.fixture
def admin_user(store):
    """A bootstrap admin user (bypasses AuthService since there's no admin yet)."""
    return create_admin(store, "admin", "admin-pass-123")


@pytest.fixture
def regular_user(store):
    """A non-admin user, created via the bootstrap admin."""
    return create_admin(store, "admin", "admin-pass-123") or store.create_user(
        "bob", hash_password("bob-pass-456"), is_admin=False
    )


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------


class TestPasswordHashing:
    def test_hash_is_not_plaintext(self):
        h = hash_password("secret123")
        assert "secret123" not in h
        assert len(h) > 20  # scrypt hashes are long

    def test_verify_correct_password(self):
        h = hash_password("mypassword")
        assert verify_password("mypassword", h) is True

    def test_verify_wrong_password(self):
        h = hash_password("mypassword")
        assert verify_password("wrongpassword", h) is False

    def test_two_hashes_differ_for_same_password(self):
        """Salt is random, so two hashes of the same password differ."""
        h1 = hash_password("same")
        h2 = hash_password("same")
        assert h1 != h2  # random salt
        assert verify_password("same", h1)
        assert verify_password("same", h2)

    def test_empty_password(self):
        h = hash_password("")
        assert verify_password("", h) is True
        assert verify_password("x", h) is False


# ---------------------------------------------------------------------------
# Authentication (AuthService.authenticate)
# ---------------------------------------------------------------------------


class TestAuthenticate:
    def test_correct_credentials(self, store, auth):
        user = create_admin(store, "alice", "pass123")
        result = auth.authenticate("alice", "pass123")
        assert result["username"] == "alice"
        assert result["_id"] == user["_id"]

    def test_wrong_password(self, store, auth):
        create_admin(store, "alice", "pass123")
        with pytest.raises(InvalidCredentialsError):
            auth.authenticate("alice", "wrongpass")

    def test_unknown_user(self, store, auth):
        with pytest.raises(InvalidCredentialsError):
            auth.authenticate("nobody", "whatever")

    def test_disabled_user_rejected(self, store, auth):
        user = create_admin(store, "alice", "pass123")
        store.set_user_enabled(user["_id"], False)
        with pytest.raises(UserDisabledError):
            auth.authenticate("alice", "pass123")

    def test_last_login_updated(self, store, auth):
        user = create_admin(store, "alice", "pass123")
        assert user["last_login"] is None
        auth.authenticate("alice", "pass123")
        updated = store.get_user_by_id(user["_id"])
        assert updated["last_login"] is not None

    def test_is_a_subclass_of_auth_error(self):
        assert issubclass(InvalidCredentialsError, AuthError)
        assert issubclass(UserDisabledError, AuthError)
        assert issubclass(NotAdminError, AuthError)


# ---------------------------------------------------------------------------
# Admin-gated user CRUD
# ---------------------------------------------------------------------------


class TestAdminCRUD:
    def test_admin_can_create_user(self, store, auth, admin_user):
        new_user = auth.create_user(admin_user, "carol", "carol-pass")
        assert new_user["username"] == "carol"
        assert new_user["is_admin"] is False
        found = store.get_user_by_username("carol")
        assert found is not None
        assert verify_password("carol-pass", found["password_hash"])

    def test_admin_can_create_admin_user(self, store, auth, admin_user):
        new_admin = auth.create_user(admin_user, "dave", "dave-pass", is_admin=True)
        assert new_admin["is_admin"] is True

    def test_duplicate_username_rejected(self, store, auth, admin_user):
        auth.create_user(admin_user, "erin", "pass1")
        with pytest.raises(UserExistsError):
            auth.create_user(admin_user, "erin", "pass2")

    def test_admin_can_list_users(self, store, auth, admin_user):
        auth.create_user(admin_user, "u1", "p1")
        auth.create_user(admin_user, "u2", "p2")
        users = auth.list_users(admin_user)
        usernames = {u["username"] for u in users}
        assert "admin" in usernames
        assert "u1" in usernames
        assert "u2" in usernames

    def test_admin_can_get_user(self, store, auth, admin_user):
        target = auth.create_user(admin_user, "frank", "fpass")
        found = auth.get_user(admin_user, target["_id"])
        assert found is not None
        assert found["username"] == "frank"

    def test_admin_can_disable_user(self, store, auth, admin_user):
        target = auth.create_user(admin_user, "gina", "gpass")
        auth.set_enabled(admin_user, target["_id"], False)
        disabled = store.get_user_by_id(target["_id"])
        assert disabled["enabled"] is False

    def test_admin_can_enable_user(self, store, auth, admin_user):
        target = auth.create_user(admin_user, "hank", "hpass")
        auth.set_enabled(admin_user, target["_id"], False)
        auth.set_enabled(admin_user, target["_id"], True)
        enabled = store.get_user_by_id(target["_id"])
        assert enabled["enabled"] is True

    def test_admin_can_reset_password(self, store, auth, admin_user):
        target = auth.create_user(admin_user, "iris", "old-pass")
        auth.reset_password(admin_user, target["_id"], "new-pass")
        updated = store.get_user_by_id(target["_id"])
        assert verify_password("new-pass", updated["password_hash"])
        assert not verify_password("old-pass", updated["password_hash"])

    def test_admin_can_delete_user(self, store, auth, admin_user):
        target = auth.create_user(admin_user, "jack", "jpass")
        auth.delete_user(admin_user, target["_id"])
        assert store.get_user_by_id(target["_id"]) is None

    def test_admin_cannot_delete_self(self, store, auth, admin_user):
        with pytest.raises(ValueError, match="your own"):
            auth.delete_user(admin_user, admin_user["_id"])

    def test_set_enabled_nonexistent_user(self, store, auth, admin_user):
        with pytest.raises(UserNotFoundError):
            auth.set_enabled(admin_user, "no-such-id", True)

    def test_reset_password_nonexistent_user(self, store, auth, admin_user):
        with pytest.raises(UserNotFoundError):
            auth.reset_password(admin_user, "no-such-id", "newpass")

    def test_set_system_message_stores_and_clears(self, store, auth, admin_user):
        target = auth.create_user(admin_user, "bot", "bot-pass")
        auth.set_system_message(admin_user, target["_id"], "  Keep it short.  ")
        assert store.get_user_by_id(target["_id"])["system_message"] == "Keep it short."
        auth.set_system_message(admin_user, target["_id"], "")
        assert store.get_user_by_id(target["_id"])["system_message"] == ""

    def test_set_system_message_nonexistent_user(self, store, auth, admin_user):
        with pytest.raises(UserNotFoundError):
            auth.set_system_message(admin_user, "no-such-id", "hi")

    def test_delete_nonexistent_user(self, store, auth, admin_user):
        with pytest.raises(UserNotFoundError):
            auth.delete_user(admin_user, "no-such-id")


# ---------------------------------------------------------------------------
# Non-admin is blocked
# ---------------------------------------------------------------------------


class TestNonAdminBlocked:
    def test_non_admin_cannot_create_user(self, store, auth, admin_user):
        bob = store.create_user("bob", hash_password("bob-pass"), is_admin=False)
        with pytest.raises(NotAdminError):
            auth.create_user(bob, "eve", "eve-pass")

    def test_non_admin_cannot_list_users(self, store, auth, admin_user):
        bob = store.create_user("bob", hash_password("bob-pass"), is_admin=False)
        with pytest.raises(NotAdminError):
            auth.list_users(bob)

    def test_non_admin_cannot_set_a_system_message(self, store, auth, admin_user):
        bob = store.create_user("bob", hash_password("bob-pass"), is_admin=False)
        with pytest.raises(NotAdminError):
            auth.set_system_message(bob, bob["_id"], "answer however I like")

    def test_non_admin_cannot_get_user(self, store, auth, admin_user):
        bob = store.create_user("bob", hash_password("bob-pass"), is_admin=False)
        with pytest.raises(NotAdminError):
            auth.get_user(bob, admin_user["_id"])

    def test_non_admin_cannot_set_enabled(self, store, auth, admin_user):
        bob = store.create_user("bob", hash_password("bob-pass"), is_admin=False)
        with pytest.raises(NotAdminError):
            auth.set_enabled(bob, admin_user["_id"], False)

    def test_non_admin_cannot_reset_password(self, store, auth, admin_user):
        bob = store.create_user("bob", hash_password("bob-pass"), is_admin=False)
        with pytest.raises(NotAdminError):
            auth.reset_password(bob, admin_user["_id"], "newpass")

    def test_non_admin_cannot_delete_user(self, store, auth, admin_user):
        bob = store.create_user("bob", hash_password("bob-pass"), is_admin=False)
        with pytest.raises(NotAdminError):
            auth.delete_user(bob, admin_user["_id"])


# ---------------------------------------------------------------------------
# Bootstrap: create_admin
# ---------------------------------------------------------------------------


class TestCreateAdmin:
    def test_creates_admin(self, store):
        user = create_admin(store, "root", "root-pass")
        assert user["username"] == "root"
        assert user["is_admin"] is True
        assert user["enabled"] is True
        assert verify_password("root-pass", user["password_hash"])

    def test_duplicate_rejected(self, store):
        create_admin(store, "root", "pass1")
        with pytest.raises(UserExistsError):
            create_admin(store, "root", "pass2")

    def test_can_authenticate_after_bootstrap(self, store, auth):
        create_admin(store, "root", "admin123")
        result = auth.authenticate("root", "admin123")
        assert result["username"] == "root"
        assert result["is_admin"] is True


# ---------------------------------------------------------------------------
# Integration: full admin lifecycle
# ---------------------------------------------------------------------------


class TestFullLifecycle:
    def test_bootstrap_then_manage_users(self, store, auth):
        """Simulate the real flow: bootstrap admin, create users, manage them."""
        admin = create_admin(store, "admin", "admin-secure")
        
        # Admin can login.
        logged_in = auth.authenticate("admin", "admin-secure")
        assert logged_in["is_admin"] is True

        # Admin creates a user.
        carol = auth.create_user(logged_in, "carol", "carol-secure")
        assert carol["is_admin"] is False

        # Carol can login.
        carol_login = auth.authenticate("carol", "carol-secure")
        assert carol_login["username"] == "carol"

        # Carol CANNOT create users.
        with pytest.raises(NotAdminError):
            auth.create_user(carol_login, "dave", "dave-pass")

        # Admin disables Carol.
        auth.set_enabled(logged_in, carol["_id"], False)

        # Carol can no longer login.
        with pytest.raises(UserDisabledError):
            auth.authenticate("carol", "carol-secure")

        # Admin resets Carol's password and re-enables.
        auth.reset_password(logged_in, carol["_id"], "new-carol-pass")
        auth.set_enabled(logged_in, carol["_id"], True)

        # Carol can login with new password.
        carol2 = auth.authenticate("carol", "new-carol-pass")
        assert carol2["username"] == "carol"

        # Admin deletes Carol.
        auth.delete_user(logged_in, carol["_id"])
        with pytest.raises(InvalidCredentialsError):
            auth.authenticate("carol", "new-carol-pass")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
