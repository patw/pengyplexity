"""Tests for personal API keys (core/apikeys.py), the rate limiter
(core/ratelimit.py), and the cancel registry's API admission rules.

All offline: a real moofile store under ``tmp_path``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from pengyplexity.core.apikeys import (
    DEFAULT_NAME,
    KEY_PREFIX,
    MAX_KEYS_PER_USER,
    MAX_NAME_LENGTH,
    ApiKeyService,
    TooManyKeysError,
    generate_key,
    hash_key,
    looks_like_key,
)
from pengyplexity.core.auth import AuthService, create_admin
from pengyplexity.core.cancel import CancelRegistry, TooManyTurns, TurnInProgress
from pengyplexity.core.ratelimit import RateLimiter
from pengyplexity.core.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "store")
    yield s
    s.close()


@pytest.fixture
def admin(store):
    return create_admin(store, "admin", "admin-pass")


@pytest.fixture
def alice(store, admin):
    return AuthService(store).create_user(admin, "alice", "alice-pass")


@pytest.fixture
def keys(store):
    return ApiKeyService(store)


class TestKeyFormat:
    def test_generated_keys_are_prefixed_and_unique(self):
        a, b = generate_key(), generate_key()
        assert a.startswith(KEY_PREFIX) and b.startswith(KEY_PREFIX)
        assert a != b
        assert looks_like_key(a)

    def test_malformed_keys_are_rejected_by_shape(self):
        for bad in ("", "pgy_", "pgy_short", "sk-" + "x" * 43, KEY_PREFIX + "x" * 500, None, 42):
            assert not looks_like_key(bad)

    def test_hash_is_stable_and_not_the_key(self):
        key = generate_key()
        assert hash_key(key) == hash_key(key)
        assert key not in hash_key(key)


class TestCreate:
    def test_plaintext_is_returned_but_never_stored(self, keys, store, alice):
        key, record = keys.create(alice, "bot")
        assert "key_hash" not in record
        stored = store.get_api_key(record["_id"])
        assert stored["key_hash"] == hash_key(key)
        assert key not in repr(stored)
        assert record["display_prefix"] == key[: len(record["display_prefix"])]
        assert record["name"] == "bot"

    def test_blank_name_gets_a_default_and_long_names_are_trimmed(self, keys, alice):
        _, blank = keys.create(alice, "   ")
        _, long = keys.create(alice, "x" * 500)
        assert blank["name"] == DEFAULT_NAME
        assert len(long["name"]) == MAX_NAME_LENGTH

    def test_per_user_cap(self, keys, alice, admin):
        for _ in range(MAX_KEYS_PER_USER):
            keys.create(alice)
        with pytest.raises(TooManyKeysError):
            keys.create(alice)
        # Someone else's allowance is unaffected.
        keys.create(admin)

    def test_list_hides_hashes_and_is_per_user(self, keys, alice, admin):
        keys.create(alice, "a")
        keys.create(admin, "b")
        listed = keys.list_for_user(alice)
        assert [k["name"] for k in listed] == ["a"]
        assert all("key_hash" not in k for k in listed)


class TestAuthenticate:
    def test_valid_key_resolves_to_its_user(self, keys, alice):
        key, record = keys.create(alice)
        user, key_doc = keys.authenticate(key)
        assert user["_id"] == alice["_id"]
        assert key_doc["_id"] == record["_id"]
        assert "key_hash" not in key_doc

    def test_unknown_and_malformed_keys_fail(self, keys, alice):
        keys.create(alice)
        assert keys.authenticate(generate_key()) is None
        assert keys.authenticate("not-a-key") is None

    def test_revoked_key_fails(self, keys, alice):
        key, record = keys.create(alice)
        assert keys.revoke(alice, record["_id"]) is True
        assert keys.authenticate(key) is None

    def test_cannot_revoke_someone_elses_key(self, keys, alice, admin):
        key, record = keys.create(alice)
        assert keys.revoke(admin, record["_id"]) is False
        assert keys.authenticate(key) is not None

    def test_disabled_owner_fails_immediately(self, keys, store, alice):
        key, _ = keys.create(alice)
        store.set_user_enabled(alice["_id"], False)
        assert keys.authenticate(key) is None
        store.set_user_enabled(alice["_id"], True)
        assert keys.authenticate(key) is not None

    def test_deleting_a_user_deletes_their_keys(self, keys, store, alice, admin):
        key, _ = keys.create(alice)
        AuthService(store).delete_user(admin, alice["_id"])
        assert store.list_api_keys_for_user(alice["_id"]) == []
        assert keys.authenticate(key) is None

    def test_last_used_is_recorded_but_throttled(self, keys, store, alice):
        key, record = keys.create(alice)
        assert store.get_api_key(record["_id"])["last_used"] is None

        keys.authenticate(key)
        first = store.get_api_key(record["_id"])["last_used"]
        assert first is not None

        keys.authenticate(key)
        assert store.get_api_key(record["_id"])["last_used"] == first

        stale = datetime.now(timezone.utc) - timedelta(hours=1)
        store.touch_api_key(record["_id"], stale)
        keys.authenticate(key)
        refreshed = store.get_api_key(record["_id"])["last_used"].replace(tzinfo=timezone.utc)
        assert refreshed > stale + timedelta(minutes=30)


class TestRateLimiter:
    def test_allows_up_to_the_limit_then_refuses(self):
        now = [0.0]
        limiter = RateLimiter(2, window=60, clock=lambda: now[0])
        assert limiter.hit("u") == (True, 0)
        assert limiter.hit("u") == (True, 0)
        allowed, retry_after = limiter.hit("u")
        assert allowed is False
        assert retry_after == 60

    def test_window_slides(self):
        now = [0.0]
        limiter = RateLimiter(1, window=60, clock=lambda: now[0])
        limiter.hit("u")
        now[0] = 45.0
        assert limiter.hit("u") == (False, 15)
        now[0] = 60.0
        assert limiter.hit("u")[0] is True

    def test_refusals_do_not_extend_the_wait(self):
        now = [0.0]
        limiter = RateLimiter(1, window=60, clock=lambda: now[0])
        limiter.hit("u")
        for t in (10.0, 20.0, 30.0):
            now[0] = t
            assert limiter.hit("u")[0] is False
        now[0] = 60.0
        assert limiter.hit("u")[0] is True

    def test_keys_are_independent_and_zero_disables(self):
        limiter = RateLimiter(1)
        assert limiter.hit("a")[0] and limiter.hit("b")[0]
        assert limiter.hit("a")[0] is False
        unlimited = RateLimiter(0)
        assert all(unlimited.hit("a")[0] for _ in range(100))


class TestRegistryAdmission:
    def test_default_start_replaces_the_running_turn(self):
        reg = CancelRegistry()
        old = reg.start("alice", "t1")
        new = reg.start("alice", "t1")
        assert old.cancelled and not new.cancelled
        assert reg.active("alice", "t1") is new

    def test_exclusive_start_refuses_a_busy_thread_without_cancelling_it(self):
        reg = CancelRegistry()
        running = reg.start("alice", "t1")
        with pytest.raises(TurnInProgress):
            reg.start("alice", "t1", exclusive=True)
        assert not running.cancelled
        assert reg.active("alice", "t1") is running

    def test_max_active_caps_threads_per_owner(self):
        reg = CancelRegistry()
        reg.start("alice", "t1", max_active=2)
        reg.start("alice", "t2", max_active=2)
        with pytest.raises(TooManyTurns):
            reg.start("alice", "t3", max_active=2)
        # Other owners are unaffected, and replacing a running thread is not new.
        reg.start("bob", "t9", max_active=2)
        reg.start("alice", "t1", max_active=2)

    def test_finishing_frees_a_slot(self):
        reg = CancelRegistry()
        token = reg.start("alice", "t1", max_active=1)
        reg.finish("alice", "t1", token)
        reg.start("alice", "t2", max_active=1)
