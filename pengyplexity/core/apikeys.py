"""Personal API keys for the JSON API (``/api/v1``).

A logged-in user mints keys from their Account page; a program (the Discord
bot, a script) then sends one as ``Authorization: Bearer <key>`` and acts as
that user — same threads, same memories, same workspace, same limits.

Design:

* **Only a hash is stored.** A key is 256 bits from :mod:`secrets`, so a plain
  SHA-256 digest is enough (a slow password hash buys nothing against a
  full-entropy secret) and lets a request look its key up by digest rather
  than scanning every row. The plaintext is shown to the user exactly once, at
  creation, and cannot be recovered afterwards.
* **Keys carry no privileges of their own.** Every request re-reads the owning
  user, so disabling or deleting the account shuts its keys off immediately;
  there is no cached grant to outlive the account.
* **Keys cannot mint keys.** Creation lives behind the web session (the
  Account page), so a leaked key can be revoked without it having spread.
* **Revoking deletes the row.** There is no "revoked" flag that some code path
  could forget to check.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

KEY_PREFIX = "pgy_"
# token_urlsafe(32) is 43 characters; anything far outside this is not a key
# and is rejected without touching the store.
_MIN_KEY_LEN = len(KEY_PREFIX) + 32
_MAX_KEY_LEN = len(KEY_PREFIX) + 64
# How much of the key is kept in the clear so the user can tell keys apart.
_DISPLAY_CHARS = len(KEY_PREFIX) + 8

MAX_KEYS_PER_USER = 20
MAX_NAME_LENGTH = 64
DEFAULT_NAME = "API key"
# ``last_used`` is a convenience, not an audit log: writing it on every call
# would turn each read-only API request into a store write.
_LAST_USED_RESOLUTION = timedelta(minutes=1)


class ApiKeyError(Exception):
    """Base error for API key management."""


class TooManyKeysError(ApiKeyError):
    """Raised when a user already holds :data:`MAX_KEYS_PER_USER` keys."""


def generate_key() -> str:
    """Return a new plaintext API key."""
    return KEY_PREFIX + secrets.token_urlsafe(32)


def hash_key(key: str) -> str:
    """The digest stored in place of *key*."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def looks_like_key(key: str) -> bool:
    """Cheap shape check, so garbage never reaches a store lookup."""
    return (
        isinstance(key, str)
        and key.startswith(KEY_PREFIX)
        and _MIN_KEY_LEN <= len(key) <= _MAX_KEY_LEN
    )


def public_view(doc: Dict[str, Any]) -> Dict[str, Any]:
    """A key record without its hash — safe for templates and API responses."""
    return {k: v for k, v in doc.items() if k != "key_hash"}


@dataclass
class ApiKeyService:
    """Create, list, revoke and verify API keys over a :class:`Store`."""

    store: Any

    def create(self, user: Dict[str, Any], name: str = "") -> Tuple[str, Dict[str, Any]]:
        """Mint a key for *user*. Returns ``(plaintext, public record)``.

        The plaintext is not stored anywhere; the caller must show it now.
        """
        if len(self.store.list_api_keys_for_user(user["_id"])) >= MAX_KEYS_PER_USER:
            raise TooManyKeysError(
                f"You already have {MAX_KEYS_PER_USER} API keys; revoke one first."
            )
        name = (name or "").strip()[:MAX_NAME_LENGTH] or DEFAULT_NAME
        key = generate_key()
        doc = self.store.create_api_key({
            "user_id": user["_id"],
            "name": name,
            "display_prefix": key[:_DISPLAY_CHARS],
            "key_hash": hash_key(key),
            "created": datetime.now(timezone.utc),
            "last_used": None,
        })
        return key, public_view(doc)

    def list_for_user(self, user: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [public_view(d) for d in self.store.list_api_keys_for_user(user["_id"])]

    def revoke(self, user: Dict[str, Any], key_id: str) -> bool:
        """Delete one of *user*'s keys. False if it isn't theirs (or doesn't exist)."""
        doc = self.store.get_api_key(key_id)
        if doc is None or doc.get("user_id") != user["_id"]:
            return False
        return bool(self.store.delete_api_key(key_id))

    def authenticate(self, key: str) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Resolve a presented key to ``(user, key record)``, or None.

        None covers every failure — malformed, unknown, revoked, owner deleted,
        owner disabled — so a caller cannot leak which one it was.
        """
        if not looks_like_key(key):
            return None
        doc = self.store.get_api_key_by_hash(hash_key(key))
        if doc is None:
            return None
        user = self.store.get_user_by_id(doc.get("user_id"))
        if user is None or not user.get("enabled", True):
            return None
        self._touch(doc)
        return user, public_view(doc)

    def _touch(self, doc: Dict[str, Any]) -> None:
        now = datetime.now(timezone.utc)
        last = doc.get("last_used")
        if last is not None:
            if last.tzinfo is None:  # moofile hands datetimes back naive (UTC)
                last = last.replace(tzinfo=timezone.utc)
            if now - last < _LAST_USED_RESOLUTION:
                return
        try:
            self.store.touch_api_key(doc["_id"], now)
        except Exception:  # noqa: BLE001
            # The key was revoked between lookup and touch. The request was
            # already authorised by then; bookkeeping must not fail it.
            pass
