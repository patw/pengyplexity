"""The discord.py side of the bot.

How it answers:

* **@mention in a channel** starts a conversation. The bot opens a Discord
  thread from the question (or replies inline if it may not create threads,
  or ``PENGYPLEXITY_DISCORD_THREADS=0``) and answers there.
* **Inside a thread the bot started**, every message is a follow-up — no
  mention needed.
* **Replying to one of the bot's answers** in a channel continues that
  answer's conversation.
* **DMs**, when enabled, are one running conversation per person; ``!new``
  starts a fresh one.

While a turn runs, the bot edits one progress message live (tool activity,
then the answer as it streams), and the asker can react ⏹️ to stop it.
"""

from __future__ import annotations

import asyncio
import io
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import discord

from ..core.ratelimit import RateLimiter
from .apiclient import ApiError, PengyplexityClient
from .config import BotConfig
from .conversations import ConversationMap, dm_key, message_key, thread_key
from .render import (
    STOP_EMOJI,
    clean_question,
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
        # Progress message id → (thread id, asker id), for the Stop reaction.
        self.active: Dict[int, Tuple[str, int]] = {}

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

    def _question(self, message: discord.Message) -> str:
        def clean(msg: discord.Message) -> str:
            return clean_question(
                msg.content,
                self.user.id,
                users={u.id: u.display_name for u in msg.mentions},
                roles={r.id: r.name for r in msg.role_mentions},
                channels={c.id: getattr(c, "name", "channel") for c in msg.channel_mentions},
            )

        question = clean(message)
        ref = message.reference
        resolved = ref.resolved if ref is not None else None
        if (
            question
            and isinstance(resolved, discord.Message)
            and resolved.author != self.user
            and resolved.content
        ):
            question = quote_context(resolved.author.display_name, clean(resolved), question)
        return question

    async def _open_conversation(self, message: discord.Message, question: str) -> Route:
        if self.config.use_threads and isinstance(message.channel, discord.TextChannel):
            try:
                thread = await message.create_thread(
                    name=thread_name(question), auto_archive_duration=THREAD_ARCHIVE_MINUTES
                )
            except discord.HTTPException as e:
                log.warning(
                    "Could not start a thread in #%s (%s); replying in the channel. "
                    "Grant 'Create Public Threads' to answer in threads.",
                    message.channel, e,
                )
            else:
                return Route(thread_key(thread.id), thread, thread=thread)
        return Route(message_key(message.id), message.channel, reply_to=message)

    # -- Events ---------------------------------------------------------------

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot or self.user is None:
            return
        route = self._route(message)
        if route is None:
            return

        question = self._question(message)
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
                surface = DiscordSurface(self, route, asker_id=message.author.id)
                if not await surface.open():
                    return
                try:
                    await answer_question(
                        self.api, self.conversations, route.key, question, surface,
                        max_upload_bytes=self.config.max_upload_bytes,
                    )
                except Exception:
                    log.exception("Failed to answer a question in %s", route.key)
                    await surface.deliver("⚠️ Something went wrong on my side.", [])
                finally:
                    await surface.close()
        finally:
            entry[1] -= 1
            if entry[1] == 0:
                self._locks.pop(route.key, None)

    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent) -> None:
        if self.user is None or payload.user_id == self.user.id or str(payload.emoji) != STOP_EMOJI:
            return
        active = self.active.get(payload.message_id)
        if active is None:
            return
        thread_id, asker_id = active
        if payload.user_id != asker_id:
            return
        try:
            if await self.api.stop(thread_id):
                log.info("Stopped the turn in thread %s on request.", thread_id)
        except ApiError as e:
            log.warning("Stop failed for thread %s: %s", thread_id, e)


class DiscordSurface:
    """A question's progress message, edited live, then replaced by the answer."""

    def __init__(self, bot: PengyplexityBot, route: Route, asker_id: int) -> None:
        self.bot = bot
        self.route = route
        self.asker_id = asker_id
        self.placeholder: Optional[discord.Message] = None
        self._wanted: Optional[str] = None
        self._shown: Optional[str] = None
        self._ticker: Optional[asyncio.Task] = None

    async def open(self) -> bool:
        text = progress_text("Thinking…", "")
        try:
            if self.route.reply_to is not None:
                self.placeholder = await self.route.reply_to.reply(text, mention_author=False)
            else:
                self.placeholder = await self.route.channel.send(text)
        except discord.HTTPException as e:
            log.warning("Cannot post in %s: %s", self.route.channel, e)
            return False
        self._shown = text
        await _quietly(self.placeholder.add_reaction(STOP_EMOJI))
        self._ticker = asyncio.create_task(self._tick())
        return True

    async def started(self, thread_id: str) -> None:
        self.bot.active[self.placeholder.id] = (thread_id, self.asker_id)

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
        if self.placeholder is not None:
            self.bot.active.pop(self.placeholder.id, None)
            await _quietly(self.placeholder.remove_reaction(STOP_EMOJI, self.bot.user))


async def _quietly(coro) -> None:
    """Await a best-effort Discord call (a reaction, a rename, a notice)."""
    try:
        await coro
    except discord.HTTPException as e:
        log.debug("Discord call failed: %s", e)
