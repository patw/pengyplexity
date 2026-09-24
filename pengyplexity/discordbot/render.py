"""Text shaping for Discord.

Pure functions with no discord.py import, so all of it is unit-testable:
turning a Discord message into a question, and an API answer into messages
that fit Discord's limits.
"""

from __future__ import annotations

import posixpath
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlsplit

MESSAGE_LIMIT = 2000
THREAD_NAME_LIMIT = 100
MAX_QUOTE_CHARS = 4000
# What of a long answer stays in the channel once the rest moves to a thread.
TEASER_CHARS = 300
MOVED_TO_THREAD = "-# 🧵 Long one — writing it up in the thread."
MORE_IN_THREAD = "-# 🧵 More in the thread."
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
# Pictures described to the model in one question. Each one the agent looks
# at costs a download and a vision call, so the whole conversation — the
# question, the message it replies to, and the room's recent messages — shares
# this allowance rather than getting one each.
MAX_IMAGES = 6
# What a URL has to end in before the bot calls it a picture. Discord's own
# CDN links carry a query string, which is why only the path is examined.
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".avif", ".heic")
# People listed in the roster that precedes a question.
MAX_ROSTER = 20

_USER_MENTION = re.compile(r"<@!?(\d+)>")
_ROLE_MENTION = re.compile(r"<@&(\d+)>")
_CHANNEL_MENTION = re.compile(r"<#(\d+)>")
_CUSTOM_EMOJI = re.compile(r"<a?(:\w+:)\d+>")
# Discord wraps a link in <> to suppress its embed, and people end sentences
# with links, so the trailing punctuation comes off separately.
_URL_IN_TEXT = re.compile(r"https?://[^\s<>\"']+")
_URL_TAIL = ".,;:!?)]}'\"…"


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
    images: str = "",
    channel: Optional[str] = None,
) -> str:
    """Assemble what the agent is actually asked.

    The question stays last and is labelled, so the model — and the server's
    thread auto-titling, which reads the start of the first question — can
    tell it apart from the context in front of it. The pictures go just above
    it, nearest the thing they are probably about.
    """
    if asker is None and not history.strip() and not images.strip():
        return question
    parts: List[str] = []
    if asker is not None:
        where = f"#{channel}" if channel else "a direct message"
        parts.append(
            f"[Discord {where}. Several people talk here, so refer to each by "
            'name or @handle — never "the user" — and say who a memory is '
            f"about when you save one. Before you answer, search_memory for "
            f"{asker.label}, the person asking — a conversation here starts "
            "over often, so what you already know about them lives in your "
            "memories rather than in this thread.]"
        )
    if roster.strip():
        parts.append("[People in this conversation]\n" + roster.strip())
    if history.strip():
        parts.append("[Recent messages, for context]\n" + history.strip())
    if images.strip():
        parts.append(images.strip())
    head = f"[The question, from {asker.full}]" if asker is not None else "[The question]"
    parts.append(f"{head}\n{question}")
    return "\n\n".join(parts)


@dataclass(frozen=True)
class Image:
    """A picture somewhere in the conversation, and where it came from.

    *source* is what the model is told about its provenance — "attached to the
    question", "posted by Alice earlier in the channel". Which picture someone
    means is usually a matter of who posted it and when, so a list of bare
    URLs is not enough.
    """

    filename: str
    url: str
    source: str = ""

    @property
    def label(self) -> str:
        return f"{self.filename} ({self.source})" if self.source else self.filename


def looks_like_image_url(url: str) -> bool:
    """Whether a URL (or a filename) points at a picture.

    Only the path is examined: a Discord CDN link ends in
    ``cat.png?ex=…&is=…&hm=…``, and every one of those would be missed by a
    plain ``endswith``.
    """
    path = urlsplit(str(url or "")).path or ""
    return path.lower().endswith(IMAGE_SUFFIXES)


def image_filename(url: str, fallback: str = "image") -> str:
    """A name for a picture that arrived as a bare URL."""
    name = posixpath.basename(urlsplit(str(url or "")).path or "")
    return name or fallback


def image_links(text: str) -> List[Tuple[str, str]]:
    """Image URLs pasted into a message's text, as ``(filename, url)``.

    Someone dropping a link to a picture is asking about a picture just as
    much as someone attaching one, but Discord reports it as ordinary text.
    Only URLs that end in an image extension count — guessing at the rest
    would send the agent off to download web pages.
    """
    found: List[Tuple[str, str]] = []
    seen = set()
    for match in _URL_IN_TEXT.finditer(text or ""):
        url = match.group(0).rstrip(_URL_TAIL)
        if url in seen or not looks_like_image_url(url):
            continue
        seen.add(url)
        found.append((image_filename(url), url))
    return found


def describe_images(images: Sequence[Image], max_files: int = MAX_IMAGES) -> str:
    """Tell the model which pictures are in play, and how to actually look.

    An image cannot ride in the question — the API takes a string. What does
    work is the tool pair the agent already has: ``download_file`` pulls the
    URL into the thread workspace and ``read_image`` hands it to the vision
    model. Naming both is what makes it reliable; given a bare URL the agent
    tends to reach for ``fetch_url``, which would only return HTML.

    *images* arrive most-relevant first — the question's own attachments, then
    the room's, newest first — because that is the order the cap keeps.
    """
    kept: List[Image] = []
    seen = set()
    for image in images:
        if not image.url or image.url in seen:
            continue
        seen.add(image.url)
        if len(kept) < max_files:
            kept.append(image)
    if not kept:
        return ""
    lines = [f"- {image.label}: {image.url}" for image in kept]
    dropped = len(seen) - len(kept)
    if dropped > 0:
        lines.append(f"- (and {dropped} more, not listed)")
    return (
        "[Pictures in this conversation. To look at one, use download_file to "
        "save its URL into the workspace, then read_image — that is the only "
        "way to see it, and fetch_url cannot. Anything the question is about "
        "is worth opening, including a picture someone posted earlier.]\n"
        + "\n".join(lines)
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


def answer_teaser(answer: str, limit: int = TEASER_CHARS) -> str:
    """The opening of a long answer, for the channel message it leaves behind
    when the rest goes into a thread.

    The first paragraph of prose: a heading alone says nothing, and a code
    block or table cut short renders as rubble. The bot's user is told to lead
    with the answer, so that paragraph is usually all a passer-by needs. Cut at
    a sentence end when one is near the limit, so it doesn't stop mid-word.
    """
    for para in re.split(r"\n\s*\n", answer or ""):
        para = para.strip()
        if not para or para.startswith(("#", "```", "|", ">")):
            continue
        # Prose that runs into a code block keeps only the prose.
        para = para.split("```", 1)[0].strip()
        if not para:
            continue
        if len(para) <= limit:
            return para
        head = para[:limit]
        stop = max(head.rfind(". "), head.rfind("! "), head.rfind("? "))
        if stop >= limit // 2:
            return head[: stop + 1]
        space = head.rfind(" ")
        return (head[:space] if space > 0 else head).rstrip(" ,;:-") + "…"
    return ""


def moved_answer_text(answer: str) -> str:
    """What stays in the channel once an answer has moved to a thread."""
    teaser = answer_teaser(answer)
    return f"{teaser}\n{MORE_IN_THREAD}" if teaser else MORE_IN_THREAD


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
