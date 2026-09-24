"""Tests for how the Discord surface places an answer: in the channel, or —
when it runs long — in a thread started from the bot's reply.

Discord is replaced by small fakes that record what was posted where.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

discord = pytest.importorskip("discord")

from pengyplexity.discordbot.bot import DiscordSurface, Route  # noqa: E402
from pengyplexity.discordbot.conversations import (  # noqa: E402
    ConversationMap,
    channel_key,
    thread_key,
)
from pengyplexity.discordbot.render import MORE_IN_THREAD, MOVED_TO_THREAD  # noqa: E402

LIMIT = 200


class FakeMessage:
    _next_id = 5000

    def __init__(self, room, content=""):
        FakeMessage._next_id += 1
        self.id = FakeMessage._next_id
        self.room = room
        self.content = content
        self.thread = None

    async def edit(self, content=None, suppress=False):
        self.content = content

    async def reply(self, content, mention_author=False):
        return await self.room.send(content)

    async def create_thread(self, name, auto_archive_duration):
        if self.room.refuse_threads:
            raise discord.HTTPException(SimpleNamespace(status=403, reason="Forbidden"), "Missing Permissions")
        self.thread = FakeRoom(name=name)
        return self.thread


class FakeRoom:
    """A channel or a thread: somewhere messages can be sent."""

    _next_id = 900

    def __init__(self, name="general", refuse_threads=False):
        FakeRoom._next_id += 1
        self.id = FakeRoom._next_id
        self.name = name
        self.refuse_threads = refuse_threads
        self.messages = []

    async def send(self, content=None, files=None, suppress_embeds=False):
        msg = FakeMessage(self, content or "")
        msg.files = files
        self.messages.append(msg)
        return msg


@pytest.fixture
def conversations(tmp_path):
    convo = ConversationMap(tmp_path / "discord.bson")
    yield convo
    convo.close()


@pytest.fixture
def bot(conversations, monkeypatch):
    # The surface only moves answers out of ordinary text channels.
    monkeypatch.setattr(discord, "TextChannel", FakeRoom)
    config = SimpleNamespace(thread_over=LIMIT, edit_interval=0.01)
    return SimpleNamespace(config=config, conversations=conversations, user=SimpleNamespace(id=1))


def channel_route(conversations, room):
    key = channel_key(room.id)
    conversations.link(key, "pgy-thread-1")
    question = FakeMessage(room, "@bot explain everything")
    return Route(key, room, reply_to=question, topic="explain everything")


LONG = "The short version: yes.\n\n" + "Here is a great deal of detail. " * 20


def run(coro):
    return asyncio.run(coro)


def test_a_short_answer_stays_in_the_channel(bot, conversations):
    room = FakeRoom()
    route = channel_route(conversations, room)

    async def go():
        surface = DiscordSurface(bot, route)
        assert await surface.open()
        await surface.progress("x", "Yes.")
        await asyncio.sleep(0.05)
        ids = await surface.deliver("Yes.", [])
        await surface.close()
        return surface, ids

    surface, ids = run(go())
    assert surface.placeholder.content == "Yes."
    assert surface.placeholder.thread is None
    assert ids == [surface.placeholder.id]


def test_a_long_answer_moves_to_a_thread_as_it_streams(bot, conversations):
    room = FakeRoom()
    route = channel_route(conversations, room)

    async def go():
        surface = DiscordSurface(bot, route)
        await surface.open()
        await surface.progress("x", LONG[: LIMIT + 1])
        await asyncio.sleep(0.05)
        # Moved before the answer is finished, so the channel never shows it.
        assert surface.placeholder.content == MOVED_TO_THREAD
        assert surface.placeholder.thread is not None
        ids = await surface.deliver(LONG, [])
        await surface.close()
        return surface, ids

    surface, ids = run(go())
    thread = surface.placeholder.thread
    assert thread.name == "explain everything"
    assert surface.placeholder.content == f"The short version: yes.\n{MORE_IN_THREAD}"
    assert thread.messages[0].content == LONG
    # Nothing but the reply was posted in the channel.
    assert room.messages == [surface.placeholder]
    # Replying to the teaser or talking in the thread continues the conversation.
    assert ids == [surface.placeholder.id, thread.messages[0].id]
    assert conversations.thread_for(thread_key(thread.id)) == "pgy-thread-1"


def test_a_long_answer_that_arrives_at_once_still_moves(bot, conversations):
    room = FakeRoom()
    route = channel_route(conversations, room)

    async def go():
        surface = DiscordSurface(bot, route)
        await surface.open()
        await surface.deliver(LONG, [])
        await surface.close()
        return surface

    surface = run(go())
    assert surface.placeholder.thread.messages[-1].content == LONG
    assert surface.placeholder.content.endswith(MORE_IN_THREAD)


def test_uploads_follow_the_answer_into_the_thread(bot, conversations):
    from pengyplexity.discordbot.session import Upload

    room = FakeRoom()
    route = channel_route(conversations, room)

    async def go():
        surface = DiscordSurface(bot, route)
        await surface.open()
        await surface.deliver(LONG, [Upload("chart.png", b"png")])
        await surface.close()
        return surface

    surface = run(go())
    assert surface.placeholder.thread.messages[-1].files
    assert room.messages == [surface.placeholder]


def test_falls_back_to_the_channel_without_thread_permission(bot, conversations):
    room = FakeRoom(refuse_threads=True)
    route = channel_route(conversations, room)

    async def go():
        surface = DiscordSurface(bot, route)
        await surface.open()
        await surface.progress("x", LONG)
        await asyncio.sleep(0.05)
        await surface.deliver(LONG, [])
        await surface.close()
        return surface

    surface = run(go())
    assert surface.placeholder.content == LONG


def test_never_moves_out_of_a_thread(bot, conversations):
    thread = FakeRoom(name="a thread")
    key = thread_key(thread.id)
    conversations.link(key, "pgy-thread-2")
    route = Route(key, thread, thread=thread, topic="q")

    async def go():
        surface = DiscordSurface(bot, route)
        await surface.open()
        await surface.deliver(LONG, [])
        return surface

    surface = run(go())
    assert surface.placeholder.content == LONG
    assert surface.placeholder.thread is None


def test_can_be_turned_off(bot, conversations):
    bot.config.thread_over = 0
    room = FakeRoom()
    route = channel_route(conversations, room)

    async def go():
        surface = DiscordSurface(bot, route)
        await surface.open()
        await surface.deliver(LONG, [])
        return surface

    surface = run(go())
    assert surface.placeholder.content == LONG
