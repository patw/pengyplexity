"""moofile-backed data store for Pengyplexity.

Provides the collections defined in spec.md:

* **users** — ``{ _id, username, password_hash, is_admin, enabled, created, last_login }``
* **threads** — ``{ _id, owner, title, created, updated, messages: [...] }``
* **shares** — ``{ _id, thread_id, message_index, kind, url, created }``
* **artifacts** — ``{ _id, filename, path, kind, mime, thread_id, message_index,
  created, size_bytes }`` (a file produced in a thread's workspace: chart /
  report / image / code)
* **settings** — a single document for admin-configurable values

The store is a thin wrapper around :class:`moofile.Collection` that:

* opens one collection per logical domain (all in the same BSON file for
  simplicity — moofile handles concurrent access within one process),
* provides convenience methods (get-by-username, list-threads-for-user,
  append-message, etc.),
* is fully injectable for tests (pass a temp path; the suite never touches
  ``$HOME``).

All timestamps are timezone-aware UTC (``datetime.now(timezone.utc)``).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import moofile


def _utcnow() -> datetime:
    """Timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class Store:
    """Pengyplexity data store backed by moofile.

    Parameters
    ----------
    path:
        Path to the ``.bson`` file. Parent directories are created if missing.
        In tests, pass a ``tmp_path``-relative path.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._users = moofile.Collection(str(self._path / "users.bson"))
        self._threads = moofile.Collection(str(self._path / "threads.bson"))
        self._shares = moofile.Collection(str(self._path / "shares.bson"))
        self._artifacts = moofile.Collection(str(self._path / "artifacts.bson"))
        self._settings = moofile.Collection(str(self._path / "settings.bson"))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush and close all collections."""
        for col in (
            self._users,
            self._threads,
            self._shares,
            self._artifacts,
            self._settings,
        ):
            col.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------

    def create_user(
        self,
        username: str,
        password_hash: str,
        is_admin: bool = False,
    ) -> Dict[str, Any]:
        """Insert a new user document and return it."""
        doc = {
            "username": username,
            "password_hash": password_hash,
            "is_admin": is_admin,
            "enabled": True,
            "created": _utcnow(),
            "last_login": None,
        }
        inserted = self._users.insert(doc)
        return inserted

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Return the user doc with this username, or None."""
        return self._users.find_one({"username": username})

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Return the user doc with this _id, or None."""
        return self._users.find_one({"_id": user_id})

    def list_users(self) -> List[Dict[str, Any]]:
        """Return all user documents."""
        return self._users.find({}).to_list()

    def update_user(self, user_id: str, **fields: Any) -> bool:
        """Update fields on a user doc by _id. Returns True if a doc was matched."""
        return self._users.update_one({"_id": user_id}, set=fields)

    def set_user_enabled(self, user_id: str, enabled: bool) -> bool:
        """Enable or disable a user."""
        return self._users.update_one({"_id": user_id}, set={"enabled": enabled})

    def set_user_password(self, user_id: str, password_hash: str) -> bool:
        """Reset a user's password hash."""
        return self._users.update_one({"_id": user_id}, set={"password_hash": password_hash})

    def touch_last_login(self, user_id: str) -> bool:
        """Update ``last_login`` to now."""
        return self._users.update_one({"_id": user_id}, set={"last_login": _utcnow()})

    def delete_user(self, user_id: str) -> bool:
        """Delete a user by _id."""
        return self._users.delete_one({"_id": user_id})

    def user_count(self) -> int:
        return self._users.count({})

    # ------------------------------------------------------------------
    # Threads
    # ------------------------------------------------------------------

    def create_thread(
        self,
        owner: str,
        title: str = "New Thread",
    ) -> Dict[str, Any]:
        """Create a new thread with an empty messages list."""
        now = _utcnow()
        doc = {
            "owner": owner,
            "title": title,
            "created": now,
            "updated": now,
            "messages": [],
        }
        inserted = self._threads.insert(doc)
        return inserted

    def get_thread(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Return the thread doc with this _id, or None."""
        return self._threads.find_one({"_id": thread_id})

    def list_threads_for_user(self, owner: str) -> List[Dict[str, Any]]:
        """All threads owned by *owner*, most recently updated first."""
        threads = self._threads.find({"owner": owner}).to_list()
        threads.sort(key=lambda t: t.get("updated", datetime.min.replace(tzinfo=timezone.utc)), reverse=True)
        return threads

    def append_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        msg_type: str = "text",
        sources: Optional[List[Dict[str, str]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Append a message to a thread and update the thread's ``updated`` timestamp.

        Returns the updated thread doc, or None if the thread doesn't exist.
        """
        thread = self._threads.find_one({"_id": thread_id})
        if thread is None:
            return None
        now = _utcnow()
        message = {
            "role": role,
            "type": msg_type,
            "content": content,
            "sources": sources or [],
            "created": now,
        }
        messages = thread.get("messages", [])
        messages.append(message)
        self._threads.replace_one(
            {"_id": thread_id},
            {**thread, "messages": messages, "updated": now},
        )
        # Return the updated thread
        return self._threads.find_one({"_id": thread_id})

    def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread by _id (and orphaned shares will remain but are harmless)."""
        return self._threads.delete_one({"_id": thread_id})

    def update_thread_title(self, thread_id: str, title: str) -> bool:
        """Set a thread's title. Returns True if a thread was matched."""
        try:
            self._threads.update_one({"_id": thread_id}, set={"title": title})
            return True
        except Exception:  # noqa: BLE001  (moofile raises DocumentNotFoundError when nothing matches)
            return False

    def thread_count(self) -> int:
        return self._threads.count({})

    # ------------------------------------------------------------------
    # Shares
    # ------------------------------------------------------------------

    def create_share(
        self,
        thread_id: str,
        message_index: int,
        kind: str,
        url: str,
    ) -> Dict[str, Any]:
        """Record a share (tclip/pengyshare URL) for a message in a thread."""
        doc = {
            "thread_id": thread_id,
            "message_index": message_index,
            "kind": kind,
            "url": url,
            "created": _utcnow(),
        }
        return self._shares.insert(doc)

    def get_shares_for_thread(self, thread_id: str) -> List[Dict[str, Any]]:
        """All shares belonging to a thread."""
        return self._shares.find({"thread_id": thread_id}).to_list()

    def get_share_by_id(self, share_id: str) -> Optional[Dict[str, Any]]:
        return self._shares.find_one({"_id": share_id})

    def delete_share(self, share_id: str) -> bool:
        return self._shares.delete_one({"_id": share_id})

    def share_count(self) -> int:
        return self._shares.count({})

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------

    def create_artifact(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Insert an artifact record and return the stored doc (with ``_id``).

        *doc* is the output of :meth:`ArtifactRecord.to_dict` (or an equivalent
        mapping with ``filename``, ``path``, ``kind``, ``mime``, ``thread_id``,
        ``message_index``, ``created``, ``size_bytes``).
        """
        return self._artifacts.insert(doc)

    def get_artifact(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        """Return the artifact doc with this ``_id``, or None."""
        return self._artifacts.find_one({"_id": artifact_id})

    def get_artifacts_for_thread(self, thread_id: str) -> List[Dict[str, Any]]:
        """All artifacts belonging to *thread_id*, in creation order.

        Sorted by ``message_index`` then ``created`` so the chat template can
        attach each artifact to the message that produced it.
        """
        docs = self._artifacts.find({"thread_id": thread_id}).to_list()
        docs.sort(
            key=lambda a: (
                a.get("message_index", 0),
                a.get("created") or datetime.min.replace(tzinfo=timezone.utc),
            )
        )
        return docs

    def get_artifacts_for_owner(self, owner: str) -> List[Dict[str, Any]]:
        """All artifacts across every thread *owner* owns, newest first.

        Artifacts don't carry an ``owner`` field directly (only
        ``thread_id``), so this joins through the owner's threads — fine at
        this app's scale (one workspace gallery per user).
        """
        thread_ids = [t["_id"] for t in self.list_threads_for_user(owner)]
        if not thread_ids:
            return []
        docs = self._artifacts.find({"thread_id": {"$in": thread_ids}}).to_list()
        docs.sort(key=lambda a: a.get("created") or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        return docs

    def delete_artifact(self, artifact_id: str) -> bool:
        """Delete an artifact record by ``_id``."""
        return self._artifacts.delete_one({"_id": artifact_id})

    def artifact_count(self) -> int:
        return self._artifacts.count({})

    # ------------------------------------------------------------------
    # Settings (single document)
    # ------------------------------------------------------------------

    _SETTINGS_ID = "global"

    def get_settings(self) -> Dict[str, Any]:
        """Return the global settings doc, creating a default one if missing."""
        doc = self._settings.find_one({"_id": self._SETTINGS_ID})
        if doc is None:
            doc = self._settings.insert({"_id": self._SETTINGS_ID, "values": {}})
        return doc

    def set_setting(self, key: str, value: Any) -> None:
        """Set a single key in the global settings values dict."""
        doc = self.get_settings()
        values = dict(doc.get("values", {}))
        values[key] = value
        # moofile has no partial-document update for a nested dict field, so
        # replace the whole doc: delete + reinsert (single-doc collection).
        self._settings.delete_one({"_id": self._SETTINGS_ID})
        self._settings.insert({"_id": self._SETTINGS_ID, "values": values})

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a single key from the global settings."""
        doc = self.get_settings()
        return doc.get("values", {}).get(key, default)

    def reset_settings(self) -> None:
        """Clear every admin override, reverting to Config/env defaults."""
        self._settings.delete_one({"_id": self._SETTINGS_ID})
