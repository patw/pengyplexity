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

A rolling conversation also *ends* eventually, via :meth:`ConversationMap.
roll_over`. A thread's whole history is replayed to the model on every turn,
so a channel bound to one thread forever makes each question dearer than the
last — a room that has been chatting for a month can be resending tens of
thousands of tokens to be asked the time. Rolling over is cheap here because
nothing durable is lost: the bot hands the model the room's recent messages
as context anyway, and anything worth keeping should already be a memory.

Stored with moofile like the rest of Pengyplexity, one small document per key.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import moofile

# Conversations that roll over when they get old or long. A ``msg:`` key is
# someone replying to one specific answer — that is an explicit request to
# continue *that* conversation, so it is never rolled over out from under them.
ROLLING_PREFIXES = ("channel:", "thread:", "dm:")


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
        self._set(key, {"thread_id": thread_id, "turns": 0})

    def record_turn(self, key: str) -> int:
        """Count a question asked in this conversation; returns the new total."""
        doc = self._col.find_one({"key": key})
        turns = int((doc or {}).get("turns") or 0) + 1
        self._set(key, {"turns": turns})
        return turns

    def stale(self, key: str, max_idle: timedelta, max_turns: int) -> Optional[str]:
        """Why this conversation should start over, or None to carry on.

        Idle first: a room that went quiet for hours has moved on, and the
        next question is usually a new subject. The turn cap is the backstop
        for a channel that never goes quiet.
        """
        if not key.startswith(ROLLING_PREFIXES):
            return None
        doc = self._col.find_one({"key": key})
        if not doc or not doc.get("thread_id"):
            return None
        updated = doc.get("updated")
        if max_idle and isinstance(updated, datetime):
            # moofile can hand back a naive datetime; treat it as the UTC it
            # was written as rather than crashing on the comparison.
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=timezone.utc)
            idle = datetime.now(timezone.utc) - updated
            if idle >= max_idle:
                return f"idle for {idle.total_seconds() / 3600:.1f}h"
        turns = int(doc.get("turns") or 0)
        if max_turns and turns >= max_turns:
            return f"{turns} turns"
        return None

    def roll_over(self, key: str, max_idle: timedelta, max_turns: int) -> Optional[str]:
        """Start this conversation over if it is stale; returns the reason it
        was rolled over, or None if it was left alone.

        Forgetting the key drops the thread link, the turn count *and* the
        ``last_seen`` watermark together — so the next question opens a new
        thread and re-sends the room's recent messages to it, which a thread
        with no history of its own needs.
        """
        reason = self.stale(key, max_idle, max_turns)
        if reason:
            self.forget(key)
        return reason

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
