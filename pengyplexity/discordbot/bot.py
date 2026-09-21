"""The discord.py side of the bot.

How it answers:

* **@mention in a channel** is answered in that channel, and the channel is
  one rolling conversation: the bot keeps what it learned there rather than
  starting over at every mention. With ``PENGYPLEXITY_DISCORD_THREADS=1`` it
  opens a Discord thread from the question and answers there instead.
* **Inside a thread the bot started**, every message is a follow-up — no
  mention needed.
* **Replying to one of the bot's answers** in a channel continues that
  answer's conversation.
* **DMs**, when enabled, are one running conversation per person; ``!new``
  starts a fresh one.

It only speaks when spoken to, but it does not listen only to the person who
asked: each question carries the channel messages posted since the bot last
answered there, so "what do you all think?" has something to refer to.

Pictures are passed as URLs for the agent to fetch and view itself — the API
takes a string, so the bytes cannot ride along with the question. Everything
a message can carry a picture as counts: a file attachment, a link Discord
turned into an embed, and a bare image URL in the text. They are collected
from the room's recent messages too, not just the one that mentioned the bot,
because "what is that thing in the screenshot above?" is an ordinary question.

While a turn runs, the bot edits one progress message live: a penguin-flavoured
status line, then the answer as it streams. There is no stop button — that is a
web-UI affordance, and a reaction on a busy channel message is not one.
"""

from __future__ import annotations

import asyncio
import io
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import discord

from ..core.ratelimit import RateLimiter
from .apiclient import PengyplexityClient
from .config import BotConfig
from .conversations import ConversationMap, channel_key, dm_key, message_key, thread_key
from .render import (
    Image,
    Speaker,
    build_question,
    clean_question,
    describe_images,
    format_history,
    format_roster,
    image_filename,
    image_links,
    looks_like_image_url,
    penguin_activity,
    progress_text,
    quote_context,
    split_message,
    thread_name,
)
from .session import Upload, answer_question

log = logging.getLogger("pengyplexity.discord")

QUEUED_EMOJI = "⏳"
NEW_CONVERSATION_COMMANDS = {"!new", "!reset"}
# Discord auto-archives an idle thread after this many minutes (1 day).
THREAD_ARCHIVE_MINUTES = 1440


@dataclass
class Route:
    """Where a message's answer goes, and which conversation it belongs to."""

    key: Optional[str]
    channel: discord.abc.Messageable
    reply_to: Optional[discord.Message] = None
    # A Discord thread the bot may rename once Pengyplexity names the question.
    thread: Optional[discord.Thread] = None
    new: bool = False
    # Name for the Pengyplexity thread when this conversation creates one.
    title: Optional[str] = None


async def verify_api(api: PengyplexityClient) -> dict:
    """Check the API key before going online, and warn about an admin key."""
    me = await api.me()
    user = me.get("user") or {}
    log.info("Pengyplexity at %s accepts the key for user %r.", api.server_root, user.get("username"))
    if user.get("is_admin"):
        log.warning(
            "The bot's API key belongs to an ADMIN account. Give the bot a dedicated "
            "non-admin user: everyone on Discord shares that account's threads and memories."
        )
    return me


class PengyplexityBot(discord.Client):
    def __init__(
        self,
        config: BotConfig,
        api: Optional[PengyplexityClient] = None,
        conversations: Optional[ConversationMap] = None,
    ) -> None:
        intents = discord.Intents.default()
        # Needed to read follow-ups in the bot's threads, which don't mention it.
        intents.message_content = True
        super().__init__(
            intents=intents,
            # An answer can quote "@everyone" or a user; it must never ping them.
            allowed_mentions=discord.AllowedMentions.none(),
        )
        self.config = config
        self.api = api or PengyplexityClient(config.api_url, config.api_key)
        self.conversations = conversations or ConversationMap(config.state_path)
        self.user_limiter = RateLimiter(config.user_rate_limit)
        # Per-conversation lock + number of messages holding or awaiting it.
        self._locks: Dict[str, List] = {}

    async def setup_hook(self) -> None:
        # Fail before connecting to Discord if the key is wrong.
        await verify_api(self.api)

    async def close(self) -> None:
        await super().close()
        await self.api.close()
        self.conversations.close()

    async def on_ready(self) -> None:
        log.info("Connected to Discord as %s (id %s).", self.user, self.user.id)
        if not self.config.allow_dms:
            log.info("DMs are off (PENGYPLEXITY_DISCORD_ALLOW_DMS=0).")

    # -- Routing --------------------------------------------------------------

    def _channel_allowed(self, channel) -> bool:
        if not self.config.channel_ids:
            return True
        ids = {channel.id, getattr(channel, "parent_id", None)}
        return bool(ids & self.config.channel_ids)

    def _route(self, message: discord.Message) -> Optional[Route]:
        channel = message.channel
        if isinstance(channel, discord.DMChannel):
            if not self.config.allow_dms:
                return None
            return Route(dm_key(channel.id), channel)
        if not self._channel_allowed(channel):
            return None
        if isinstance(channel, discord.Thread) and self.conversations.thread_for(thread_key(channel.id)):
            return Route(thread_key(channel.id), channel, thread=channel)
        ref = message.reference
        if ref is not None and ref.message_id and self.conversations.thread_for(message_key(ref.message_id)):
            return Route(message_key(ref.message_id), channel, reply_to=message)
        # raw_mentions, not mentioned_in(): @everyone must not summon the bot.
        if self.user.id in message.raw_mentions:
            return Route(None, channel, reply_to=message, new=True)
        return None

    def _clean(self, msg: discord.Message) -> str:
        return clean_question(
            msg.content,
            self.user.id,
            users={u.id: u.display_name for u in msg.mentions},
            roles={r.id: r.name for r in msg.role_mentions},
            channels={c.id: getattr(c, "name", "channel") for c in msg.channel_mentions},
        )

    def _question(self, message: discord.Message) -> Tuple[str, List[Image]]:
        """What was asked, and the pictures that came with it."""
        question = self._clean(message)
        ref = message.reference
        resolved = ref.resolved if ref is not None else None
        replied = resolved if isinstance(resolved, discord.Message) else None
        if question and replied is not None and replied.author != self.user and replied.content:
            question = quote_context(_speaker(replied.author).label, self._clean(replied), question)

        images: List[Image] = []
        if self.config.send_images:
            images = _images_in(message, "with the question")
            if replied is not None:
                who = _speaker(replied.author).label
                images += _images_in(replied, f"in the message this replies to, from {who}")
        if not question and images:
            # A mention carrying nothing but a picture is still a question.
            question = "Look at the attached image and tell me about it."
        return question, images

    async def _channel_context(
        self, message: discord.Message, key: Optional[str]
    ) -> Tuple[str, List[Speaker], List[Image]]:
        """The messages posted here since the model last heard from this channel:
        their text, who wrote them, and any pictures in them.

        Only what is new: everything older is already in the Pengyplexity
        thread, so re-sending it would pay for the same tokens every turn.

        A picture is collected even when its message has no text at all — a
        screenshot posted with no caption is exactly the one the next question
        is about. The pictures come back newest first, which is the order the
        cap in :func:`describe_images` keeps.
        """
        limit = self.config.history_lines
        if limit <= 0 or key is None or isinstance(message.channel, discord.DMChannel):
            return "", [], []
        seen = self.conversations.last_seen(key)
        entries: List[Tuple[Speaker, str]] = []
        images: List[Image] = []
        try:
            async for past in message.channel.history(limit=limit, before=message):
                # Newest first, so the first already-seen message ends the walk.
                if seen is not None and past.id <= seen:
                    break
                # Its own answers are already the thread's assistant turns.
                if past.author.id == self.user.id:
                    continue
                speaker = _speaker(past.author)
                if self.config.send_images:
                    images += _images_in(past, f"posted earlier here by {speaker.label}")
                text = self._clean(past)
                names = [a.filename for a in past.attachments]
                if names:
                    text = " ".join([text, *(f"[attached {n}]" for n in names)]).strip()
                if text:
                    entries.append((speaker, text))
        except discord.HTTPException as e:
            # Usually a missing 'Read Message History'; answer without context.
            log.debug("Could not read history in %s: %s", message.channel, e)
            return "", [], []
        entries.reverse()
        return format_history(entries), [speaker for speaker, _ in entries], images

    async def _open_conversation(self, message: discord.Message, question: str) -> Route:
        channel = message.channel
        if self.config.use_threads and isinstance(channel, discord.TextChannel):
            try:
                thread = await message.create_thread(
                    name=thread_name(question), auto_archive_duration=THREAD_ARCHIVE_MINUTES
                )
            except discord.HTTPException as e:
                log.warning(
                    "Could not start a thread in #%s (%s); replying in the channel. "
                    "Grant 'Create Public Threads' to answer in threads.",
                    channel, e,
                )
            else:
                return Route(thread_key(thread.id), thread, thread=thread)
        if isinstance(channel, (discord.TextChannel, discord.Thread)):
            # One rolling conversation per channel. It is named here because the
            # question it would be auto-named from starts with the room's
            # recent messages, which makes a poor title.
            name = getattr(channel, "name", None)
            return Route(
                channel_key(channel.id),
                channel,
                reply_to=message,
                title=f"Discord #{name}" if name else None,
            )
        return Route(message_key(message.id), channel, reply_to=message)

    # -- Events ---------------------------------------------------------------

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot or self.user is None:
            return
        route = self._route(message)
        if route is None:
            return

        question, images = self._question(message)
        if not question:
            if route.new:
                await _quietly(message.reply(
                    "Ask me a question after the mention, e.g. `@me what changed in Python 3.13?`",
                    mention_author=False,
                ))
            return

        if isinstance(message.channel, discord.DMChannel) and question.lower() in NEW_CONVERSATION_COMMANDS:
            self.conversations.forget(route.key)
            await _quietly(message.channel.send("🆕 Started a fresh conversation."))
            return

        allowed, retry_after = self.user_limiter.hit(str(message.author.id))
        if not allowed:
            await _quietly(message.reply(
                f"🐢 Slow down a little — try again in {retry_after}s.", mention_author=False
            ))
            return

        if route.new:
            route = await self._open_conversation(message, question)
        # Before the context is built, so a rolled-over conversation re-sends
        # the room's recent messages to the fresh thread instead of the one or
        # two lines since the last watermark.
        reason = self.conversations.roll_over(
            route.key, self.config.max_idle, self.config.max_turns
        )
        if reason:
            log.info("Starting a fresh conversation for %s (%s).", route.key, reason)
        history, speakers, past_images = await self._channel_context(message, route.key)
        asker = _speaker(message.author)
        question = build_question(
            question,
            asker=asker,
            history=history,
            # One line each, asker first; pointless when nobody else spoke.
            roster=format_roster([asker, *speakers]) if speakers else "",
            # The question's own pictures first: the cap keeps what comes
            # first, and those are the ones it is most likely about.
            images=describe_images(images + past_images),
            channel=getattr(message.channel, "name", None),
        )
        await self._answer(message, route, question)

    async def _answer(self, message: discord.Message, route: Route, question: str) -> None:
        # One question at a time per conversation: the API refuses a second
        # turn in a busy thread, so later questions queue here instead.
        entry = self._locks.setdefault(route.key, [asyncio.Lock(), 0])
        entry[1] += 1
        try:
            lock = entry[0]
            queued = lock.locked()
            if queued:
                await _quietly(message.add_reaction(QUEUED_EMOJI))
            async with lock:
                if queued:
                    await _quietly(message.remove_reaction(QUEUED_EMOJI, self.user))
                surface = DiscordSurface(self, route)
                if not await surface.open():
                    return
                try:
                    await answer_question(
                        self.api, self.conversations, route.key, question, surface,
                        max_upload_bytes=self.config.max_upload_bytes,
                        title=route.title,
                    )
                except Exception:
                    log.exception("Failed to answer a question in %s", route.key)
                    await surface.deliver("⚠️ Something went wrong on my side.", [])
                finally:
                    # The question reached the thread even if answering it
                    # failed, so the next turn must not replay it as context.
                    self.conversations.mark_seen(route.key, message.id)
                    await surface.close()
        finally:
            entry[1] -= 1
            if entry[1] == 0:
                self._locks.pop(route.key, None)


class DiscordSurface:
    """A question's progress message, edited live, then replaced by the answer."""

    def __init__(self, bot: PengyplexityBot, route: Route) -> None:
        self.bot = bot
        self.route = route
        self.placeholder: Optional[discord.Message] = None
        self._wanted: Optional[str] = None
        self._shown: Optional[str] = None
        self._ticker: Optional[asyncio.Task] = None

    async def open(self) -> bool:
        text = progress_text(penguin_activity(), "")
        try:
            if self.route.reply_to is not None:
                self.placeholder = await self.route.reply_to.reply(text, mention_author=False)
            else:
                self.placeholder = await self.route.channel.send(text)
        except discord.HTTPException as e:
            log.warning("Cannot post in %s: %s", self.route.channel, e)
            return False
        self._shown = text
        self._ticker = asyncio.create_task(self._tick())
        return True

    async def progress(self, text: str) -> None:
        self._wanted = text

    async def _tick(self) -> None:
        # Edit at a steady pace instead of per token: Discord rate-limits
        # edits, and the latest text is all that matters.
        while True:
            await asyncio.sleep(self.bot.config.edit_interval)
            wanted = self._wanted
            if wanted is None or wanted == self._shown:
                continue
            try:
                await self.placeholder.edit(content=wanted, suppress=True)
                self._shown = wanted
            except discord.HTTPException as e:
                log.debug("Progress edit failed: %s", e)

    async def _stop_ticker(self) -> None:
        if self._ticker is not None:
            self._ticker.cancel()
            try:
                await self._ticker
            except (asyncio.CancelledError, Exception):
                pass
            self._ticker = None

    async def rename(self, title: str) -> None:
        thread = self.route.thread
        if thread is None or not title or thread.owner_id != self.bot.user.id:
            return
        await _quietly(thread.edit(name=thread_name(title)))

    async def deliver(self, text: str, uploads: List[Upload]) -> List[int]:
        await self._stop_ticker()
        chunks = split_message(text) or ["…"]
        sent: List[int] = []
        try:
            await self.placeholder.edit(content=chunks[0], suppress=True)
            sent.append(self.placeholder.id)
        except discord.HTTPException as e:
            log.warning("Could not edit the answer into place: %s", e)
        for chunk in chunks[1:]:
            try:
                msg = await self.route.channel.send(chunk, suppress_embeds=True)
                sent.append(msg.id)
            except discord.HTTPException as e:
                log.warning("Could not send a %d-character answer chunk: %s", len(chunk), e)
        if uploads:
            # Files go in their own message, so an upload Discord refuses
            # (too big for this server) never costs the text of the answer.
            files = [discord.File(io.BytesIO(u.data), filename=u.filename) for u in uploads]
            try:
                msg = await self.route.channel.send(files=files)
                sent.append(msg.id)
            except discord.HTTPException as e:
                log.warning("Could not upload %d file(s): %s", len(files), e)
                await _quietly(self.route.channel.send("-# Couldn't attach the files for this answer."))
        return sent

    async def close(self) -> None:
        await self._stop_ticker()


def _speaker(user) -> Speaker:
    """A Discord user's three names. ``display_name`` is already the nickname
    here or the global one, so it is the alias a reader would recognise."""
    handle = getattr(user, "name", "") or ""
    alias = getattr(user, "display_name", None) or getattr(user, "global_name", None) or handle
    return Speaker(display_name=alias, username=handle, user_id=getattr(user, "id", 0) or 0)


def _is_image_attachment(attachment: discord.Attachment) -> bool:
    """Discord usually reports a content type; when it doesn't, the filename
    is the only evidence there is, and it has to actually look like a picture
    — guessing would send the agent off to download a random file."""
    content_type = (attachment.content_type or "").strip()
    if content_type:
        return content_type.startswith("image/")
    return looks_like_image_url(attachment.filename)


def _embed_image_url(embed: discord.Embed) -> Optional[str]:
    """The picture behind a link Discord turned into an embed, if there is one.

    This is how a pasted imgur or Tenor link becomes something the agent can
    look at: the link itself serves a web page, but ``image``/``thumbnail``
    point at the media. ``image`` is the full-size one, so it wins; a Tenor
    GIF only has the thumbnail. ``embed.video`` is deliberately ignored —
    ``read_image`` cannot open an mp4.
    """
    for candidate in (embed.image, embed.thumbnail):
        url = getattr(candidate, "url", None)
        if url:
            return str(url)
    # Only when the link *is* the picture; otherwise this is the page's URL.
    url = getattr(embed, "url", None)
    return str(url) if url and looks_like_image_url(url) else None


def _images_in(message: discord.Message, source: str) -> List[Image]:
    """Every picture one Discord message carries.

    Three ways a picture gets into a channel, and only the first was ever
    handled: attached as a file, pasted as a bare image URL, or linked and
    embedded by Discord. The embeds are checked last because a pasted link
    also produces one, and the text match names the file better.

    An embed arrives on a *later* gateway event than the message it belongs
    to, so a link pasted in the very message that mentions the bot may not
    have one yet — that case is caught by the URL in the text instead.
    """
    found: List[Image] = []
    seen = set()

    def add(filename: str, url: Optional[str]) -> None:
        if url and url not in seen:
            seen.add(url)
            found.append(Image(filename, url, source))

    for attachment in message.attachments:
        if _is_image_attachment(attachment):
            add(attachment.filename, attachment.url)
    for filename, url in image_links(message.content or ""):
        add(filename, url)
    for embed in message.embeds:
        url = _embed_image_url(embed)
        add(image_filename(url or ""), url)
    return found


async def _quietly(coro) -> None:
    """Await a best-effort Discord call (a reaction, a rename, a notice)."""
    try:
        await coro
    except discord.HTTPException as e:
        log.debug("Discord call failed: %s", e)
