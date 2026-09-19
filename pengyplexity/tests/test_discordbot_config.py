"""Tests for the Discord bot's settings (``discordbot/config.py``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pengyplexity.discordbot.config import ConfigError, load_bot_config

ALL_VARS = (
    "DISCORD_BOT_TOKEN",
    "PENGYPLEXITY_API_URL",
    "PENGYPLEXITY_API_KEY",
    "PENGYPLEXITY_DATA_DIR",
    "PENGYPLEXITY_DISCORD_STATE",
    "PENGYPLEXITY_DISCORD_CHANNELS",
    "PENGYPLEXITY_DISCORD_ALLOW_DMS",
    "PENGYPLEXITY_DISCORD_THREADS",
    "PENGYPLEXITY_DISCORD_HISTORY",
    "PENGYPLEXITY_DISCORD_IMAGES",
    "PENGYPLEXITY_DISCORD_USER_RATE_LIMIT",
    "PENGYPLEXITY_DISCORD_MAX_UPLOAD_MB",
)


@pytest.fixture
def env(monkeypatch, tmp_path):
    for name in ALL_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "discord-token")
    monkeypatch.setenv("PENGYPLEXITY_API_URL", "http://127.0.0.1:5080")
    monkeypatch.setenv("PENGYPLEXITY_API_KEY", "pgy_" + "k" * 43)
    return monkeypatch


def test_defaults(env, tmp_path):
    cfg = load_bot_config()
    assert cfg.api_url == "http://127.0.0.1:5080"
    assert cfg.state_path == tmp_path / ".pengyplexity" / "discord.bson"
    assert cfg.channel_ids == frozenset()
    assert cfg.allow_dms is False
    # Answers land in the channel, not a thread spun off from every question.
    assert cfg.use_threads is False
    assert cfg.history_lines == 20
    assert cfg.send_images is True
    assert cfg.user_rate_limit == 6
    assert cfg.max_upload_bytes == 10 * 1024 * 1024


def test_secrets_stay_out_of_repr(env):
    text = repr(load_bot_config())
    assert "discord-token" not in text and "pgy_" not in text


@pytest.mark.parametrize("missing", ["DISCORD_BOT_TOKEN", "PENGYPLEXITY_API_URL", "PENGYPLEXITY_API_KEY"])
def test_required_settings_are_named(env, missing):
    env.setenv(missing, "  ")
    with pytest.raises(ConfigError, match=missing):
        load_bot_config()


def test_a_discord_token_in_the_api_key_slot_is_caught(env):
    env.setenv("PENGYPLEXITY_API_KEY", "MTIz.discord.token")
    with pytest.raises(ConfigError, match="pgy_"):
        load_bot_config()


def test_api_url_needs_a_scheme(env):
    env.setenv("PENGYPLEXITY_API_URL", "pengy.example.com")
    with pytest.raises(ConfigError, match="http"):
        load_bot_config()


def test_overrides(env, tmp_path):
    env.setenv("PENGYPLEXITY_DISCORD_CHANNELS", "111, 222\n333")
    env.setenv("PENGYPLEXITY_DISCORD_ALLOW_DMS", "1")
    env.setenv("PENGYPLEXITY_DISCORD_THREADS", "1")
    env.setenv("PENGYPLEXITY_DISCORD_HISTORY", "5")
    env.setenv("PENGYPLEXITY_DISCORD_IMAGES", "0")
    env.setenv("PENGYPLEXITY_DISCORD_USER_RATE_LIMIT", "0")
    env.setenv("PENGYPLEXITY_DISCORD_MAX_UPLOAD_MB", "25")
    env.setenv("PENGYPLEXITY_DISCORD_STATE", "~/bot/state.bson")
    cfg = load_bot_config()
    assert cfg.channel_ids == frozenset({111, 222, 333})
    assert cfg.allow_dms is True
    assert cfg.use_threads is True
    assert cfg.history_lines == 5
    assert cfg.send_images is False
    assert cfg.user_rate_limit == 0
    assert cfg.max_upload_bytes == 25 * 1024 * 1024
    # A .env value is not shell-expanded; the bot expands it.
    assert cfg.state_path == Path(tmp_path) / "bot" / "state.bson"


def test_state_follows_the_data_dir(env, tmp_path):
    env.setenv("PENGYPLEXITY_DATA_DIR", str(tmp_path / "pdata"))
    assert load_bot_config().state_path == tmp_path / "pdata" / "discord.bson"


@pytest.mark.parametrize("raw,expected", [("0", 0), ("-5", 0), ("not a number", 20)])
def test_history_lines_is_never_negative(env, raw, expected):
    # A negative limit would reach discord.py's history() as a limit.
    env.setenv("PENGYPLEXITY_DISCORD_HISTORY", raw)
    assert load_bot_config().history_lines == expected


def test_bad_channel_id(env):
    env.setenv("PENGYPLEXITY_DISCORD_CHANNELS", "general")
    with pytest.raises(ConfigError, match="general"):
        load_bot_config()
