"""Which Pengyplexity thread holds each Discord conversation's history.

The agent sees a thread's earlier messages on every turn, so follow-ups only
work if the bot asks in the same thread again. This map is what makes that
survive a restart. Keys name a Discord conversation:

* ``thread:<channel id>`` — a Discord thread the bot started for a question;
  every later message in it continues the conversation.
* ``dm:<channel id>`` — a direct-message conversation.
* ``msg:<message id>`` — a message in an ordinary channel. Each answer the bot
  posts is recorded this way, so replying to it continues that thread.

Stored with moofile like the rest of Pengyplexity, one small document per key.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import moofile


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

    def link(self, key: str, thread_id: str) -> None:
        now = datetime.now(timezone.utc)
        if self._col.find_one({"key": key}):
            self._col.update_one({"key": key}, set={"thread_id": thread_id, "updated": now})
        else:
            self._col.insert({"key": key, "thread_id": thread_id, "created": now, "updated": now})

    def forget(self, key: str) -> bool:
        return self._col.delete_many({"key": key}) > 0

    def close(self) -> None:
        self._col.close()
