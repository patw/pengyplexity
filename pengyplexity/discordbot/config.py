"""Settings for the Discord bot.

Read from the environment; the CLI loads a ``.env`` file into it first
(without overriding anything already set). Standard library only, so the
offline test suite can import it without the ``discord`` extra.

Three settings are required — the Discord bot token, where Pengyplexity is,
and the API key of the Pengyplexity user the bot acts as. Everything else has
a default. See DISCORD.md.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import FrozenSet

from ..config import _env_float, _env_int, _env_str, _home

REQUIRED = ("DISCORD_BOT_TOKEN", "PENGYPLEXITY_API_URL", "PENGYPLEXITY_API_KEY")

DEFAULT_USER_RATE_LIMIT = 6
# A rolling conversation starts over once the room has been quiet this long.
DEFAULT_IDLE_HOURS = 3.0
# ...or after this many questions, for a channel that never goes quiet.
DEFAULT_MAX_TURNS = 20
# Discord's upload cap for a server without boosts.
DEFAULT_MAX_UPLOAD_MB = 10.0
# Channel messages of context handed to the model with a question.
DEFAULT_HISTORY_LINES = 20
# An in-channel answer longer than this many characters moves to a thread.
DEFAULT_THREAD_OVER = 1000


class ConfigError(ValueError):
    """A setting is missing or malformed; the message says which and how to fix it."""


@dataclass
class BotConfig:
    """Resolved bot settings. Secrets are kept out of ``repr`` so a logged
    config cannot leak them."""

    discord_token: str = field(repr=False)
    api_url: str
    api_key: str = field(repr=False)
    # The moofile store mapping Discord conversations to Pengyplexity threads.
    state_path: Path
    # Channels the bot answers in (threads count as their parent); empty = all.
    channel_ids: FrozenSet[int] = frozenset()
    allow_dms: bool = False
    # Answer a new question in a Discord thread started from it (else reply inline
    # and keep one rolling conversation per channel).
    use_threads: bool = False
    # An answer in a channel that runs past this many characters is moved into
    # a Discord thread started from the bot's reply, leaving only its opening
    # in the channel; 0 = never. Short answers stay in the channel.
    thread_over: int = DEFAULT_THREAD_OVER
    # Preceding channel messages to hand the model as context; 0 = none.
    history_lines: int = DEFAULT_HISTORY_LINES
    # A channel/DM conversation starts a fresh Pengyplexity thread once the
    # room has been quiet this many hours, or after this many questions —
    # whichever comes first. The whole thread is replayed to the model every
    # turn, so without this each question in a long-lived room costs more
    # than the last. 0 disables that half of the rule.
    idle_hours: float = DEFAULT_IDLE_HOURS
    max_turns: int = DEFAULT_MAX_TURNS
    # Pass image attachments to the agent (it fetches and views them itself).
    send_images: bool = True
    # Questions one Discord user may ask per minute; 0 = no limit. The server's
    # own API limit is shared by everyone talking to the bot, so this keeps one
    # person from spending all of it.
    user_rate_limit: int = DEFAULT_USER_RATE_LIMIT
    max_upload_mb: float = DEFAULT_MAX_UPLOAD_MB
    # Seconds between live edits of the progress message (Discord rate-limits edits).
    edit_interval: float = 1.5

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @property
    def max_idle(self) -> timedelta:
        return timedelta(hours=self.idle_hours)


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return raw not in {"0", "false", "False", "no"}


def _channel_ids(raw: str) -> FrozenSet[int]:
    ids = set()
    for part in re.split(r"[,\s]+", raw.strip()):
        if not part:
            continue
        if not part.isdigit():
            raise ConfigError(
                f"PENGYPLEXITY_DISCORD_CHANNELS: {part!r} is not a channel ID "
                "(right-click a channel with Developer Mode on → Copy Channel ID)."
            )
        ids.add(int(part))
    return frozenset(ids)


def load_bot_config() -> BotConfig:
    """Build a :class:`BotConfig` from the environment, or raise :class:`ConfigError`."""
    missing = [name for name in REQUIRED if not os.environ.get(name, "").strip()]
    if missing:
        raise ConfigError(
            f"Missing required setting(s): {', '.join(missing)}. "
            "Copy .env.example to .env and fill in the Discord section."
        )

    api_url = os.environ["PENGYPLEXITY_API_URL"].strip()
    if not api_url.startswith(("http://", "https://")):
        raise ConfigError(
            f"PENGYPLEXITY_API_URL must start with http:// or https:// (got {api_url!r})."
        )
    api_key = os.environ["PENGYPLEXITY_API_KEY"].strip()
    if not api_key.startswith("pgy_"):
        raise ConfigError(
            "PENGYPLEXITY_API_KEY does not look like a Pengyplexity API key (they "
            "start with pgy_). Create one on the bot user's Account page."
        )

    data_dir = Path(_env_str("PENGYPLEXITY_DATA_DIR", str(_home() / ".pengyplexity")))
    state_path = Path(_env_str("PENGYPLEXITY_DISCORD_STATE", str(data_dir / "discord.bson")))

    return BotConfig(
        discord_token=os.environ["DISCORD_BOT_TOKEN"].strip(),
        api_url=api_url,
        api_key=api_key,
        # A .env value is not shell-expanded, so "~/.pengyplexity" arrives literally.
        state_path=state_path.expanduser(),
        channel_ids=_channel_ids(os.environ.get("PENGYPLEXITY_DISCORD_CHANNELS", "")),
        allow_dms=_flag("PENGYPLEXITY_DISCORD_ALLOW_DMS", False),
        use_threads=_flag("PENGYPLEXITY_DISCORD_THREADS", False),
        thread_over=max(0, _env_int("PENGYPLEXITY_DISCORD_THREAD_OVER", DEFAULT_THREAD_OVER)),
        # A negative count is meaningless and would reach discord.py as a limit.
        history_lines=max(0, _env_int("PENGYPLEXITY_DISCORD_HISTORY", DEFAULT_HISTORY_LINES)),
        idle_hours=max(0.0, _env_float("PENGYPLEXITY_DISCORD_IDLE_HOURS", DEFAULT_IDLE_HOURS)),
        max_turns=max(0, _env_int("PENGYPLEXITY_DISCORD_MAX_TURNS", DEFAULT_MAX_TURNS)),
        send_images=_flag("PENGYPLEXITY_DISCORD_IMAGES", True),
        user_rate_limit=_env_int("PENGYPLEXITY_DISCORD_USER_RATE_LIMIT", DEFAULT_USER_RATE_LIMIT),
        max_upload_mb=_env_float("PENGYPLEXITY_DISCORD_MAX_UPLOAD_MB", DEFAULT_MAX_UPLOAD_MB),
    )
