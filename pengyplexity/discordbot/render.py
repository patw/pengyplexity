"""Text shaping for Discord.

Pure functions with no discord.py import, so all of it is unit-testable:
turning a Discord message into a question, and an API answer into messages
that fit Discord's limits.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional

MESSAGE_LIMIT = 2000
THREAD_NAME_LIMIT = 100
MAX_SOURCES = 10
MAX_QUOTE_CHARS = 4000
STOP_EMOJI = "⏹️"

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


def progress_text(label: str, partial: str, limit: int = MESSAGE_LIMIT) -> str:
    """The live progress message: the tool activity until the answer starts,
    then the tail of the answer so far."""
    hint = f" · {STOP_EMOJI} to stop"
    if not partial.strip():
        return f"-# 🔎 {label or 'Thinking…'}{hint}"
    footer = f"\n-# ✍️ Writing…{hint}"
    room = limit - len(footer) - 4  # 4 = a closing "\n```"
    body = partial if len(partial) <= room else "…" + partial[-(room - 1):]
    if body.count("```") % 2 == 1:
        body += "\n```"
    return body + footer


def _link_text(text: str) -> str:
    return " ".join(re.sub(r"[\[\]]", " ", text).split()) or "link"


def format_sources(sources: Iterable[Mapping[str, Any]]) -> str:
    """A numbered Sources list. ``(<url>)`` stops Discord unfurling every link."""
    lines, seen = [], set()
    for source in sources:
        url = str(source.get("url") or "").strip()
        if not url.startswith(("http://", "https://")) or url in seen or ">" in url:
            continue
        seen.add(url)
        title = _link_text(str(source.get("title") or url))
        lines.append(f"{len(lines) + 1}. [{title}](<{url}>)")
        if len(lines) >= MAX_SOURCES:
            break
    return "**Sources**\n" + "\n".join(lines) if lines else ""


def render_answer(
    message: Optional[Dict[str, Any]],
    error: Optional[str] = None,
    notes: Iterable[str] = (),
) -> str:
    """The final text for a turn: the saved answer, its sources, and notes."""
    parts = []
    content = str((message or {}).get("content") or "").strip()
    if content:
        parts.append(content)
    sources = format_sources((message or {}).get("sources") or [])
    if sources:
        parts.append(sources)
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
