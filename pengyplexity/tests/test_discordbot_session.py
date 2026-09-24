"""Tests for the Discord bot's API client, conversation map and question flow.

The client talks real HTTP to a real Pengyplexity app served on 127.0.0.1
(with the fake agents from ``test_api.py``), so the bot is tested against the
actual API contract rather than a mock of it — still fully offline. Discord
itself is replaced by a fake :class:`~pengyplexity.discordbot.session.Surface`.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("aiohttp")

from werkzeug.serving import make_server  # noqa: E402

from pengyplexity.app import create_app, get_state  # noqa: E402
from pengyplexity.config import Config  # noqa: E402
from pengyplexity.core.auth import create_admin  # noqa: E402
from pengyplexity.discordbot.apiclient import (  # noqa: E402
    ApiError,
    PengyplexityClient,
    SSEParser,
    api_root,
)
from pengyplexity.discordbot.conversations import (  # noqa: E402
    ConversationMap,
    channel_key,
    message_key,
    thread_key,
)
from datetime import datetime, timedelta, timezone  # noqa: E402
from pengyplexity.discordbot.render import progress_text  # noqa: E402
from pengyplexity.discordbot.session import answer_question  # noqa: E402
from pengyplexity.tests.test_api import ChartAgent, StreamingAgent  # noqa: E402

BIG = 10 * 1024 * 1024


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app(tmp_path):
    cfg = Config(
        data_dir=tmp_path / "data",
        store_path=tmp_path / "data" / "store.bson",
        model_base="http://127.0.0.1:0/v1",
        model_key="test",
        secret_key="test-secret",
    )
    app = create_app(cfg=cfg)
    state = get_state(app)
    admin = create_admin(state.store, "admin", "admin-pass")
    state.auth.create_user(admin, "discordbot", "bot-pass")
    state.agent = StreamingAgent()
    return app


@pytest.fixture
def state(app):
    return get_state(app)


@pytest.fixture
def server(app):
    srv = make_server("127.0.0.1", 0, app, threaded=True)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()
    thread.join(timeout=5)


@pytest.fixture
def key(state):
    user = state.store.get_user_by_username("discordbot")
    plaintext, _ = state.api_keys.create(user, "discord")
    return plaintext


@pytest.fixture
def conversations(tmp_path):
    convo = ConversationMap(tmp_path / "bot" / "discord.bson")
    yield convo
    convo.close()


class FakeSurface:
    def __init__(self):
        self.progress_updates = []
        self.titles = []
        self.delivered = []
        self._next_id = 1000

    async def progress(self, label, partial):
        self.progress_updates.append(progress_text(label, partial))

    async def rename(self, title):
        self.titles.append(title)

    async def deliver(self, text, uploads):
        self.delivered.append((text, uploads))
        self._next_id += 1
        return [self._next_id]


_surface = FakeSurface


def run(coro):
    return asyncio.run(coro)


async def _ask(server, key, conversations, convo_key, question, surface, max_bytes=BIG, title=None):
    async with PengyplexityClient(server, key) as api:
        return await answer_question(
            api, conversations, convo_key, question, surface,
            max_upload_bytes=max_bytes, title=title,
        )


# ---------------------------------------------------------------------------
# Pure pieces
# ---------------------------------------------------------------------------


def test_api_root_normalisation():
    assert api_root("https://p.example.com/") == "https://p.example.com/api/v1"
    assert api_root("https://p.example.com/api/v1/") == "https://p.example.com/api/v1"
    assert api_root("https://p.example.com/pengy") == "https://p.example.com/pengy/api/v1"


def test_sse_parser_handles_comments_multiline_data_and_bad_json():
    parser = SSEParser()
    got = []
    for line in [": keep-alive", "event: token", 'data: {"content":', 'data: "hi"}', "",
                 "event: odd", "data: not json", "\r", "", ""]:
        event = parser.feed(line)
        if event:
            got.append(event)
    assert got == [("token", {"content": "hi"}), ("odd", {"raw": "not json"})]


def test_artifact_urls_never_carry_the_key_off_the_server():
    api = PengyplexityClient("https://p.example.com/pengy", "pgy_x")
    assert api.artifact_url("/api/v1/threads/t/artifacts/a") == "https://p.example.com/api/v1/threads/t/artifacts/a"
    assert api.artifact_url("https://p.example.com/x") == "https://p.example.com/x"
    for evil in ("https://evil.example.com/steal", "//evil.example.com/steal", "http://p.example.com/x"):
        with pytest.raises(ApiError) as e:
            api.artifact_url(evil)
        assert e.value.code == "foreign_url"


def test_conversation_map_persists_across_restarts(tmp_path):
    path = tmp_path / "discord.bson"
    convo = ConversationMap(path)
    convo.link(thread_key(1), "t1")
    convo.link(message_key(2), "t2")
    convo.link(thread_key(1), "t3")  # relinking replaces
    convo.close()

    convo = ConversationMap(path)
    try:
        assert convo.thread_for(thread_key(1)) == "t3"
        assert convo.thread_for(message_key(2)) == "t2"
        assert convo.forget(message_key(2)) is True
        assert convo.thread_for(message_key(2)) is None
        assert convo.forget(message_key(2)) is False
    finally:
        convo.close()


def _attachment(filename, content_type="image/png"):
    return SimpleNamespace(
        filename=filename, url=f"https://cdn.example/{filename}", content_type=content_type
    )


def _embed(image=None, thumbnail=None, url=None):
    media = lambda u: SimpleNamespace(url=u)  # noqa: E731 - discord.py's proxy shape
    return SimpleNamespace(image=media(image), thumbnail=media(thumbnail), url=url)


def _message(content="", attachments=(), embeds=()):
    return SimpleNamespace(content=content, attachments=list(attachments), embeds=list(embeds))


def test_only_image_attachments_are_offered_to_the_agent():
    pytest.importorskip("discord")
    from pengyplexity.discordbot.bot import _images_in

    message = _message(attachments=[
        _attachment("cat.png", "image/png"),
        _attachment("notes.pdf", "application/pdf"),
        # Discord does not always report a content type; guessing "image" from
        # the name would send the agent off to download a random file.
        _attachment("mystery.bin", None),
        _attachment("dog.jpg", None),
    ])
    found = _images_in(message, "with the question")
    assert [(i.filename, i.url) for i in found] == [
        ("cat.png", "https://cdn.example/cat.png"),
        ("dog.jpg", "https://cdn.example/dog.jpg"),
    ]
    assert all(i.source == "with the question" for i in found)
    assert _images_in(_message(), "x") == []


def test_pasted_and_embedded_pictures_count_too():
    pytest.importorskip("discord")
    from pengyplexity.discordbot.bot import _images_in

    # A bare link in the text, and a link Discord resolved into an embed.
    message = _message(
        content="is this https://i.example/cat.png the same as that?",
        embeds=[_embed(image="https://media.example/dog.jpg", url="https://example.com/dog")],
    )
    assert [i.url for i in _images_in(message, "here")] == [
        "https://i.example/cat.png",
        "https://media.example/dog.jpg",
    ]


def test_an_embedded_link_is_not_listed_twice_or_as_a_web_page():
    pytest.importorskip("discord")
    from pengyplexity.discordbot.bot import _images_in

    # Discord embeds a pasted image link, so the same URL arrives both ways.
    pasted = _message(
        content="https://i.example/cat.png",
        embeds=[_embed(image="https://i.example/cat.png", url="https://i.example/cat.png")],
    )
    assert [i.url for i in _images_in(pasted, "here")] == ["https://i.example/cat.png"]

    # A Tenor GIF has only a thumbnail; its own URL serves a web page.
    tenor = _message(embeds=[_embed(thumbnail="https://media.tenor.com/x.gif",
                                    url="https://tenor.com/view/x-gif-1")])
    assert [i.url for i in _images_in(tenor, "here")] == ["https://media.tenor.com/x.gif"]

    # An ordinary link embed carries no picture at all.
    article = _message(embeds=[_embed(url="https://example.com/article")])
    assert _images_in(article, "here") == []


def test_channel_conversations_remember_what_was_already_seen(tmp_path):
    path = tmp_path / "discord.bson"
    convo = ConversationMap(path)
    try:
        assert convo.last_seen(channel_key(5)) is None
        convo.link(channel_key(5), "t1")
        convo.mark_seen(channel_key(5), 100)
        convo.mark_seen(channel_key(5), 200)
        # A watermark set before the thread exists survives being linked.
        convo.mark_seen(channel_key(6), 7)
        convo.link(channel_key(6), "t2")
        assert convo.thread_for(channel_key(6)) == "t2"
        assert convo.last_seen(channel_key(6)) == 7
    finally:
        convo.close()

    convo = ConversationMap(path)
    try:
        assert convo.thread_for(channel_key(5)) == "t1"
        assert convo.last_seen(channel_key(5)) == 200
    finally:
        convo.close()


def test_a_conversation_rolls_over_once_it_is_old_or_long(tmp_path):
    convo = ConversationMap(tmp_path / "discord.bson")
    hour, never = timedelta(hours=1), timedelta(0)
    try:
        key = channel_key(9)
        convo.link(key, "t1")
        convo.mark_seen(key, 100)

        # Fresh and short: left alone.
        assert convo.roll_over(key, hour, 20) is None
        assert convo.thread_for(key) == "t1"

        # Long: the turn cap catches a channel that never goes quiet.
        for _ in range(3):
            convo.record_turn(key)
        assert convo.roll_over(key, hour, 20) is None
        assert "3 turns" in convo.roll_over(key, hour, 3)
        # Everything goes together: the next question opens a new thread AND
        # re-sends the room's recent messages to it.
        assert convo.thread_for(key) is None
        assert convo.last_seen(key) is None

        # Idle: the room moved on hours ago.
        convo.link(key, "t2")
        convo.record_turn(key)
        assert convo.roll_over(key, never, 0) is None   # both halves disabled
        five_hours_ago = datetime.now(timezone.utc) - timedelta(hours=5)
        convo._col.update_one({"key": key}, set={"updated": five_hours_ago})
        assert convo.roll_over(key, hour, 0).startswith("idle for 5.0h")
        assert convo.thread_for(key) is None
    finally:
        convo.close()


def test_replying_to_an_answer_never_rolls_over(tmp_path):
    """A 'msg:' key is someone replying to one specific answer — an explicit
    request to continue that conversation, however old it is."""
    convo = ConversationMap(tmp_path / "discord.bson")
    try:
        convo.link(message_key(11), "t1")
        for _ in range(50):
            convo.record_turn(message_key(11))
        assert convo.roll_over(message_key(11), timedelta(0.0), 1) is None
        assert convo.thread_for(message_key(11)) == "t1"
    finally:
        convo.close()


# ---------------------------------------------------------------------------
# Against a live app
# ---------------------------------------------------------------------------


def test_me_identifies_the_bot_user(server, key):
    async def go():
        async with PengyplexityClient(server + "/api/v1/", key) as api:
            return await api.me()

    assert run(go())["user"]["username"] == "discordbot"


def test_bad_key_is_an_api_error_with_the_stable_code(server):
    async def go():
        async with PengyplexityClient(server, "pgy_" + "n" * 43) as api:
            await api.me()

    with pytest.raises(ApiError) as e:
        run(go())
    assert (e.value.status, e.value.code) == (401, "invalid_api_key")


def test_first_question_creates_a_thread_and_follow_ups_reuse_it(server, key, conversations, state):
    surface = _surface()
    outcome = run(_ask(server, key, conversations, thread_key(42), "Are cats mammals?", surface))

    thread_id = conversations.thread_for(thread_key(42))
    assert thread_id and outcome.thread_id == thread_id
    assert surface.titles == ["Are cats mammals?"]
    # The agent's real label ("Searching the web…") is replaced by a penguin.
    assert any("🐧" in p for p in surface.progress_updates)
    assert not any("Searching the web" in p for p in surface.progress_updates)
    assert any("Cats are" in p for p in surface.progress_updates)

    (text, uploads), = surface.delivered
    assert text.startswith("Cats are mammals.")
    # No sources footer: a link goes in the sentence, not a numbered list.
    assert "**Sources**" not in text
    assert uploads == []
    # The answer's message is remembered, so a reply to it continues the thread.
    assert conversations.thread_for(message_key(outcome.message_ids[0])) == thread_id

    follow_up = _surface()
    run(_ask(server, key, conversations, thread_key(42), "And dogs?", follow_up))
    assert follow_up.titles == []  # already named
    thread = state.store.get_thread(thread_id)
    assert [m["role"] for m in thread["messages"]] == ["user", "assistant", "user", "assistant"]
    assert thread["owner"] == "discordbot"


def test_a_rolled_over_channel_starts_a_new_thread(server, key, conversations, state):
    run(_ask(server, key, conversations, channel_key(77), "Are cats mammals?", _surface()))
    first = conversations.thread_for(channel_key(77))
    # Asking is what counts a turn, so the cap measures the conversation's
    # real cost rather than how long the bot has been running.
    assert conversations.record_turn(channel_key(77)) == 2

    assert conversations.roll_over(channel_key(77), timedelta(hours=3), 2) is not None
    run(_ask(server, key, conversations, channel_key(77), "And dogs?", _surface()))
    second = conversations.thread_for(channel_key(77))

    assert second and second != first
    # The old thread keeps its history; the new one starts clean, which is the
    # whole point — the next question pays for one exchange, not the room's month.
    assert len(state.store.get_thread(first)["messages"]) == 2
    assert len(state.store.get_thread(second)["messages"]) == 2


def test_a_channel_thread_is_named_rather_than_titled_from_the_question(
    server, key, conversations, state
):
    # A channel question starts with the room's recent messages, so letting the
    # server auto-title from it would name the thread after someone's chatter.
    surface = _surface()
    run(_ask(
        server, key, conversations, channel_key(3),
        "[Recent messages in #general]\nbob: lunch?\n\n[The question]\nAre cats mammals?",
        surface, title="Discord #general",
    ))
    thread_id = conversations.thread_for(channel_key(3))
    assert state.store.get_thread(thread_id)["title"] == "Discord #general"
    assert surface.titles == []  # nothing for the bot to rename

    # The second question reuses that thread and leaves the name alone.
    run(_ask(server, key, conversations, channel_key(3), "And dogs?", _surface(),
             title="Discord #general"))
    assert conversations.thread_for(channel_key(3)) == thread_id
    assert state.store.get_thread(thread_id)["title"] == "Discord #general"


def test_artifacts_are_downloaded_for_upload(server, key, conversations, state):
    state.agent = ChartAgent(state.store)
    surface = _surface()
    run(_ask(server, key, conversations, thread_key(7), "Chart it", surface))
    (text, uploads), = surface.delivered
    assert "Here is your chart." in text
    assert [(u.filename, u.data) for u in uploads] == [("chart.png", b"PNGDATA")]


def test_oversized_artifacts_become_a_note(server, key, conversations, state):
    state.agent = ChartAgent(state.store)
    surface = _surface()
    run(_ask(server, key, conversations, thread_key(7), "Chart it", surface, max_bytes=3))
    (text, uploads), = surface.delivered
    assert uploads == []
    assert "chart.png is too large to attach" in text


def test_a_thread_deleted_in_the_web_ui_is_replaced(server, key, conversations, state):
    run(_ask(server, key, conversations, thread_key(9), "first", _surface()))
    old = conversations.thread_for(thread_key(9))
    state.store.delete_thread(old)

    surface = _surface()
    outcome = run(_ask(server, key, conversations, thread_key(9), "second", surface))
    new = conversations.thread_for(thread_key(9))
    assert new and new != old and outcome.thread_id == new
    assert surface.delivered[0][0].startswith("Cats are mammals.")


def test_busy_thread_is_explained_not_raised(server, key, conversations, state):
    run(_ask(server, key, conversations, thread_key(5), "first", _surface()))
    thread_id = conversations.thread_for(thread_key(5))
    token = state.cancels.start("discordbot", thread_id, exclusive=True)
    try:
        surface = _surface()
        outcome = run(_ask(server, key, conversations, thread_key(5), "second", surface))
    finally:
        state.cancels.finish("discordbot", thread_id, token)
    assert outcome.error == "turn_in_progress"
    assert "still answering" in surface.delivered[0][0]
    # Refused before anything was written, and the mapping is kept.
    assert len(state.store.get_thread(thread_id)["messages"]) == 2
    assert conversations.thread_for(thread_key(5)) == thread_id


def test_revoked_key_is_explained(server, key, conversations, state):
    user = state.store.get_user_by_username("discordbot")
    for doc in state.api_keys.list_for_user(user):
        state.api_keys.revoke(user, doc["_id"])
    surface = _surface()
    outcome = run(_ask(server, key, conversations, thread_key(1), "hi", surface))
    assert outcome.error == "invalid_api_key"
    assert "API key" in surface.delivered[0][0]


def test_unreachable_server_is_explained(key, conversations):
    surface = _surface()
    outcome = run(_ask("http://127.0.0.1:9", key, conversations, thread_key(1), "hi", surface))
    assert outcome.error == "unreachable"
    assert "can't reach" in surface.delivered[0][0]


def test_cli_check_reads_the_env_file(server, key, tmp_path, monkeypatch, capsys):
    pytest.importorskip("discord")
    pytest.importorskip("dotenv")
    from pengyplexity.discordbot.cli import main

    for name in ("DISCORD_BOT_TOKEN", "PENGYPLEXITY_API_URL", "PENGYPLEXITY_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    env_file = tmp_path / "bot.env"
    env_file.write_text(
        f"DISCORD_BOT_TOKEN=unused\nPENGYPLEXITY_API_URL={server}\nPENGYPLEXITY_API_KEY={key}\n"
    )
    assert main(["--env-file", str(env_file), "--check"]) == 0
    assert "OK" in capsys.readouterr().out

    monkeypatch.setenv("PENGYPLEXITY_API_KEY", "pgy_" + "n" * 43)  # the environment wins
    assert main(["--env-file", str(env_file), "--check"]) == 1
    assert "API key" in capsys.readouterr().err
