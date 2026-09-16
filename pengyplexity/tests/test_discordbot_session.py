"""Tests for the Discord bot's API client, conversation map and question flow.

The client talks real HTTP to a real Pengyplexity app served on 127.0.0.1
(with the fake agents from ``test_api.py``), so the bot is tested against the
actual API contract rather than a mock of it — still fully offline. Discord
itself is replaced by a fake :class:`~pengyplexity.discordbot.session.Surface`.
"""

from __future__ import annotations

import asyncio
import threading

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
    message_key,
    thread_key,
)
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
        self.thread_ids = []
        self.progress_updates = []
        self.titles = []
        self.delivered = []
        self._next_id = 1000

    async def started(self, thread_id):
        self.thread_ids.append(thread_id)

    async def progress(self, text):
        self.progress_updates.append(text)

    async def rename(self, title):
        self.titles.append(title)

    async def deliver(self, text, uploads):
        self.delivered.append((text, uploads))
        self._next_id += 1
        return [self._next_id]


_surface = FakeSurface


def run(coro):
    return asyncio.run(coro)


async def _ask(server, key, conversations, convo_key, question, surface, max_bytes=BIG):
    async with PengyplexityClient(server, key) as api:
        return await answer_question(
            api, conversations, convo_key, question, surface, max_upload_bytes=max_bytes
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
    assert surface.thread_ids == [thread_id]
    assert surface.titles == ["Are cats mammals?"]
    assert any("Searching the web" in p for p in surface.progress_updates)
    assert any("Cats are" in p for p in surface.progress_updates)

    (text, uploads), = surface.delivered
    assert text.startswith("Cats are mammals.")
    assert "**Sources**\n1. [Cat Wiki](<https://cats.example.com>)" in text
    assert uploads == []
    # The answer's message is remembered, so a reply to it continues the thread.
    assert conversations.thread_for(message_key(outcome.message_ids[0])) == thread_id

    follow_up = _surface()
    run(_ask(server, key, conversations, thread_key(42), "And dogs?", follow_up))
    assert follow_up.thread_ids == [thread_id]
    assert follow_up.titles == []  # already named
    thread = state.store.get_thread(thread_id)
    assert [m["role"] for m in thread["messages"]] == ["user", "assistant", "user", "assistant"]
    assert thread["owner"] == "discordbot"


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


def test_stop_on_an_idle_thread(server, key, state):
    thread = state.store.create_thread("discordbot")

    async def go():
        async with PengyplexityClient(server, key) as api:
            return await api.stop(thread["_id"])

    assert run(go()) is False


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
