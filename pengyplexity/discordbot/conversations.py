"""Which Pengyplexity thread holds each Discord conversation's history.

The agent sees a thread's earlier messages on every turn, so follow-ups only
work if the bot asks in the same thread again. This map is what makes that
survive a restart. Keys name a Discord conversation:

* ``channel:<channel id>`` — an ordinary channel the bot talks in. One rolling
  conversation per channel, so the bot keeps its memory of the room between
  questions instead of starting over at every mention.
* ``thread:<channel id>`` — a Discord thread the bot started for a question;
  every later message in it continues the conversation.
* ``dm:<channel id>`` — a direct-message conversation.
* ``msg:<message id>`` — a message in an ordinary channel. Each answer the bot
  posts is recorded this way, so replying to it continues that thread.

Each key also carries a ``last_seen`` watermark: the id of the last Discord
message whose text was handed to the model. The bot injects only the channel
messages newer than it, so a rolling conversation never sees the same message
twice (the Pengyplexity thread already holds everything before it).

Stored with moofile like the rest of Pengyplexity, one small document per key.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import moofile


def channel_key(channel_id: int) -> str:
    return f"channel:{channel_id}"


def thread_key(channel_id: int) -> str:
    return f"thread:{channel_id}"


def dm_key(channel_id: int) -> str:
    return f"dm:{channel_id}"


def message_key(message_id: int) -> str:
    return f"msg:{message_id}"


class ConversationMap:
    def __init__(self, path: Path | str) -> None:
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._col = moofile.Collection(str(path), indexes=["key"])

    def thread_for(self, key: str) -> Optional[str]:
        doc = self._col.find_one({"key": key})
        return doc.get("thread_id") if doc else None

    def _set(self, key: str, fields: dict) -> None:
        now = datetime.now(timezone.utc)
        if self._col.find_one({"key": key}):
            self._col.update_one({"key": key}, set={**fields, "updated": now})
        else:
            self._col.insert({"key": key, **fields, "created": now, "updated": now})

    def link(self, key: str, thread_id: str) -> None:
        self._set(key, {"thread_id": thread_id})

    def last_seen(self, key: str) -> Optional[int]:
        """The last Discord message id already shown to the model, if any."""
        doc = self._col.find_one({"key": key})
        value = doc.get("last_seen") if doc else None
        return int(value) if value else None

    def mark_seen(self, key: str, message_id: int) -> None:
        """Record that everything up to *message_id* has reached the model."""
        self._set(key, {"last_seen": int(message_id)})

    def forget(self, key: str) -> bool:
        return self._col.delete_many({"key": key}) > 0

    def close(self) -> None:
        self._col.close()
