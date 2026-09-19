"""Text shaping for Discord.

Pure functions with no discord.py import, so all of it is unit-testable:
turning a Discord message into a question, and an API answer into messages
that fit Discord's limits.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

MESSAGE_LIMIT = 2000
THREAD_NAME_LIMIT = 100
MAX_QUOTE_CHARS = 4000
# Status lines shown while a turn runs. The server sends a precise label
# ("Searching the web…"), and the bot throws it away: nobody in a channel is
# debugging the agent, they are waiting for an answer. A penguin doing
# something penguin-ish carries the only information that matters — still
# working — and reads like a bot with a pulse rather than a progress bar.
PENGUIN_ACTIVITIES = (
    "Dreaming of fish…",
    "Running from orcas…",
    "Waddling to the library…",
    "Sliding downhill on my belly…",
    "Consulting the colony…",
    "Diving for answers…",
    "Shaking off the snow…",
    "Huddling for warmth…",
    "Counting pebbles…",
    "Negotiating with a seal…",
    "Preening…",
    "Swimming laps…",
    "Checking the ice…",
    "Following the fish…",
    "Arguing with a skua…",
    "Balancing an egg…",
    "Tobogganing across the shelf…",
    "Squinting at the horizon…",
    "Hopping between floes…",
    "Warming my feet…",
    "Stealing a very nice pebble…",
    "Surfacing for air…",
    "Marching single file…",
    "Braving the blizzard…",
    "Listening for the colony…",
    "Shuffling towards the water…",
)
DEFAULT_ACTIVITY = "Thinking about fish…"
# Budget for the recent-channel-messages block. The API accepts 100k characters,
# but context the model has to wade through costs tokens and attention on every
# turn, so the room's chatter gets a small, fixed allowance.
MAX_HISTORY_CHARS = 8000
MAX_HISTORY_LINE = 500
# Image attachments described to the model in one question.
MAX_ATTACHMENTS = 4
# People listed in the roster that precedes a question.
MAX_ROSTER = 20

_USER_MENTION = re.compile(r"<@!?(\d+)>")
_ROLE_MENTION = re.compile(r"<@&(\d+)>")
_CHANNEL_MENTION = re.compile(r"<#(\d+)>")
_CUSTOM_EMOJI = re.compile(r"<a?(:\w+:)\d+>")


# ── Discord → question ─────────────────────────────────────────────────────


def clean_question(
    content: str,
    bot_id: int,
    users: Optional[Mapping[int, str]] = None,
    roles: Optional[Mapping[int, str]] = None,
    channels: Optional[Mapping[int, str]] = None,
) -> str:
    """Drop the bot's own mention and turn Discord's ``<@id>`` / ``<#id>``
    syntax into readable names, so the model sees what the user saw."""
    users, roles, channels = users or {}, roles or {}, channels or {}
    text = re.sub(rf"<@!?{int(bot_id)}>", "", content)
    text = _USER_MENTION.sub(lambda m: "@" + users.get(int(m.group(1)), "user"), text)
    text = _ROLE_MENTION.sub(lambda m: "@" + roles.get(int(m.group(1)), "role"), text)
    text = _CHANNEL_MENTION.sub(lambda m: "#" + channels.get(int(m.group(1)), "channel"), text)
    text = _CUSTOM_EMOJI.sub(r"\1", text)
    return text.strip()


def quote_context(author: str, quoted: str, question: str) -> str:
    """A question asked in reply to someone's message, with that message
    quoted — "@bot is this true?" means nothing without it."""
    quoted = quoted.strip()
    if len(quoted) > MAX_QUOTE_CHARS:
        quoted = quoted[:MAX_QUOTE_CHARS].rstrip() + "…"
    block = "\n".join(f"> {line}" for line in quoted.splitlines())
    return f"{author} wrote:\n{block}\n\n{question}"


@dataclass(frozen=True)
class Speaker:
    """Who said something on Discord.

    Three names matter and no one of them is enough. The *alias* is what people
    read, but it differs per server and two people can share one. The
    *username* is the globally unique handle, but it can be changed. The *id*
    never changes but means nothing to a reader. The agent needs all three: it
    talks with many people in one channel, so a memory saved about "the user"
    is worse than no memory at all.
    """

    display_name: str = ""
    username: str = ""
    user_id: int = 0

    @property
    def label(self) -> str:
        """Short form, for a line of chat: ``Alice (@alice_dev)``."""
        alias = (self.display_name or "").strip()
        handle = (self.username or "").strip()
        if not handle:
            return alias or "someone"
        if not alias or alias.lower() == handle.lower():
            return f"@{handle}"
        return f"{alias} (@{handle})"

    @property
    def full(self) -> str:
        """Long form, carrying the id that outlives any rename."""
        return f"{self.label}, id {self.user_id}" if self.user_id else self.label


def format_history(
    entries: Iterable[Tuple[Speaker, str]], max_chars: int = MAX_HISTORY_CHARS
) -> str:
    """Recent channel messages as ``who: text`` lines, oldest first.

    *entries* arrive oldest first. When they don't all fit, the oldest are
    dropped rather than the newest — the messages just before the question are
    the ones that make it make sense.
    """
    kept: List[str] = []
    total = 0
    for speaker, text in reversed(list(entries)):
        body = " ".join(text.split())
        if not body:
            continue
        if len(body) > MAX_HISTORY_LINE:
            body = body[: MAX_HISTORY_LINE - 1].rstrip() + "…"
        line = f"{speaker.label}: {body}"
        if total + len(line) + 1 > max_chars:
            break
        kept.append(line)
        total += len(line) + 1
    kept.reverse()
    return "\n".join(kept)


def format_roster(speakers: Iterable[Speaker], max_people: int = MAX_ROSTER) -> str:
    """Everyone in the conversation, once each, with their ids.

    The ids live here rather than on every chat line: one line per person is
    cheap, where repeating an 18-digit id per message is not.
    """
    seen = set()
    lines: List[str] = []
    for speaker in speakers:
        marker = speaker.user_id or speaker.username or speaker.display_name
        if not marker or marker in seen:
            continue
        seen.add(marker)
        lines.append(f"- {speaker.full}")
        if len(lines) >= max_people:
            break
    return "\n".join(lines)


def build_question(
    question: str,
    *,
    asker: Optional[Speaker] = None,
    history: str = "",
    roster: str = "",
    channel: Optional[str] = None,
) -> str:
    """Assemble what the agent is actually asked.

    The question stays last and is labelled, so the model — and the server's
    thread auto-titling, which reads the start of the first question — can
    tell it apart from the context in front of it.
    """
    if asker is None and not history.strip():
        return question
    parts: List[str] = []
    if asker is not None:
        where = f"#{channel}" if channel else "a direct message"
        parts.append(
            f"[Discord {where}. Several people talk here, so refer to each by "
            'name or @handle — never "the user" — and say who a memory is '
            "about when you save one.]"
        )
    if roster.strip():
        parts.append("[People in this conversation]\n" + roster.strip())
    if history.strip():
        parts.append(
            "[Recent messages, for context — the question follows]\n" + history.strip()
        )
    head = f"[The question, from {asker.full}]" if asker is not None else "[The question]"
    parts.append(f"{head}\n{question}")
    return "\n\n".join(parts)


def describe_attachments(
    items: Sequence[Tuple[str, str]], max_files: int = MAX_ATTACHMENTS
) -> str:
    """Tell the model about image attachments, and how to actually look at them.

    An image cannot ride in the question — the API takes a string. What does
    work is the tool pair the agent already has: ``download_file`` pulls the
    Discord CDN URL into the thread workspace and ``read_image`` hands it to
    the vision model. Naming both is what makes it reliable; given a bare URL
    the agent tends to reach for ``fetch_url``, which would only return HTML.
    """
    lines = [f"- {name}: {url}" for name, url in items[:max_files]]
    if not lines:
        return ""
    dropped = len(items) - len(lines)
    if dropped > 0:
        lines.append(f"- (and {dropped} more, not listed)")
    return (
        "[Images attached to this message. Use download_file to save each one "
        "into the workspace, then read_image to look at it.]\n" + "\n".join(lines)
    )


def thread_name(question: str) -> str:
    """A Discord thread name (≤100 chars) from a question's first line."""
    first = next((line for line in question.splitlines() if line.strip()), "")
    name = " ".join(first.split())
    if len(name) > THREAD_NAME_LIMIT:
        name = name[: THREAD_NAME_LIMIT - 1].rstrip() + "…"
    return name or "Question"


# ── Answer → Discord ───────────────────────────────────────────────────────


def split_message(message: str, limit: int = MESSAGE_LIMIT) -> List[str]:
    """Split into ≤limit chunks, preferring newline boundaries and keeping
    ``` fences balanced so Discord doesn't render the rest as code."""
    if len(message) <= limit:
        return [message] if message.strip() else []

    # Headroom for the fence markers the balancer may add to a chunk.
    effective = limit - 8
    pieces = []
    while len(message) > effective:
        cut = message.rfind("\n", 0, effective)
        if cut < effective // 2:
            cut = effective
        pieces.append(message[:cut])
        message = message[cut:].lstrip("\n")
    if message:
        pieces.append(message)

    out = []
    in_fence = False
    for piece in pieces:
        opened = in_fence
        if piece.count("```") % 2 == 1:
            in_fence = not in_fence
        if opened:
            piece = "```\n" + piece
        if in_fence:
            piece = piece + "\n```"
        if piece.strip():
            out.append(piece)
    return out


def penguin_activity(exclude: Optional[str] = None) -> str:
    """A random penguin status line, never the one already on screen."""
    choices = [line for line in PENGUIN_ACTIVITIES if line != exclude]
    return random.choice(choices or list(PENGUIN_ACTIVITIES))


def progress_text(label: str, partial: str, limit: int = MESSAGE_LIMIT) -> str:
    """The live progress message: a penguin status line until the answer
    starts, then the tail of the answer so far with the line beneath it."""
    footer = f"-# 🐧 {label or DEFAULT_ACTIVITY}"
    if not partial.strip():
        return footer
    footer = "\n" + footer
    room = limit - len(footer) - 4  # 4 = a closing "\n```"
    body = partial if len(partial) <= room else "…" + partial[-(room - 1):]
    if body.count("```") % 2 == 1:
        body += "\n```"
    return body + footer


def render_answer(
    message: Optional[Dict[str, Any]],
    error: Optional[str] = None,
    notes: Iterable[str] = (),
) -> str:
    """The final text for a turn: the saved answer and any notes.

    The turn's sources are deliberately *not* appended. A numbered "Sources"
    footer under every answer is the most chatbot-looking thing in a channel;
    the bot's user is instead told (via its per-user system message) to work a
    link into the sentence when one is genuinely worth having.
    """
    parts = []
    content = str((message or {}).get("content") or "").strip()
    if content:
        parts.append(content)
    if error and not content:
        # With partial content the saved answer already ends in an _[Error]_ note.
        parts.append(f"⚠️ {error}")
    parts.extend(notes)
    return "\n\n".join(parts) if parts else "⚠️ No answer came back."


def describe_api_error(err: Any) -> str:
    """A user-facing sentence for an ``ApiError``."""
    code = getattr(err, "code", "")
    if code == "turn_in_progress":
        return ("⏳ This conversation is still answering an earlier question "
                "(possibly from the web UI). Try again once it finishes.")
    if code == "rate_limited":
        wait = getattr(err, "retry_after", None) or 60
        return f"🐢 Too many questions right now. Try again in {wait}s."
    if code == "too_many_concurrent_turns":
        return "🐢 I'm already answering several questions. Try again when one finishes."
    if code in {"unauthorized", "invalid_api_key"}:
        return "🔒 Pengyplexity rejected my API key. An admin needs to give me a new one."
    if code == "agent_unavailable":
        return "🛠️ Pengyplexity has no model configured right now."
    if getattr(err, "status", None) == 0:
        return "📡 I can't reach the Pengyplexity server right now."
    return f"⚠️ Pengyplexity error: {getattr(err, 'message', err)}"


def megabytes(size: int) -> str:
    return f"{size / (1024 * 1024):.1f} MB"
