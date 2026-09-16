# Pengyplexity — Discord bot

Put Pengyplexity in a Discord server: @mention the bot with a question and
the answer streams into a Discord thread, with its sources and any charts or
images attached. It uses the same agent, sandbox and settings as the web UI.

> For the overview, see the [README](README.md). The bot is an ordinary
> client of the [JSON API](API.md), and deploying the app itself is in
> [INSTALLING.md](INSTALLING.md).

**Contents:** [How it works](#how-it-works) · [Setup](#setup) ·
[Using it](#using-it) · [Settings](#settings) ·
[Running as a service](#running-as-a-service-systemd) ·
[Troubleshooting](#troubleshooting)

---

## How it works

`pengyplexity-discord` is its own long-running process. It holds **one API
key for one Pengyplexity user** and talks to `/api/v1` like any other
program. It never opens the stores or the sandbox, so it can run on the app's
host or on any machine that can reach the app.

- Every Discord conversation is a Pengyplexity thread in the bot user's
  account. You can read them all in the web UI by logging in as that user.
- The bot keeps a small map from Discord conversations to thread ids in
  `~/.pengyplexity/discord.bson`, so conversations survive restarts. If a
  thread is deleted in the web UI, the next message there starts a new one.
- **Everyone on Discord shares that one account.** They share its threads,
  its memories and its API rate limit. That is why the bot gets a dedicated
  user and not your own account. It also warns at startup if the key belongs
  to an admin.

## Setup

**1. Create the bot's Pengyplexity user.** As an admin, open `/admin`, create
a normal (non-admin) user such as `discord`, log in as it, open **Account**,
and create an API key. Copy it; it is shown once.

**2. Create the Discord application.** In the
[Developer Portal](https://discord.com/developers/applications):

- **New Application** → **Bot** → **Reset Token**, and copy the token.
- Under **Privileged Gateway Intents**, turn on **Message Content Intent**.
  The bot needs it to read follow-ups in its threads, which don't mention it.
- **OAuth2 → URL Generator**: scope `bot`, with the permissions *View
  Channels*, *Send Messages*, *Send Messages in Threads*, *Create Public
  Threads*, *Read Message History*, *Attach Files*, *Add Reactions* and
  *Embed Links*. Open the generated URL to invite the bot to your server.

**3. Install and configure.**

```bash
uv sync --extra discord
cp .env.example .env      # or add to your existing .env
```

Fill in the Discord section of `.env`:

```bash
DISCORD_BOT_TOKEN=MTIz…            # step 2
PENGYPLEXITY_API_URL=https://pengyplexity.example.com   # or http://127.0.0.1:5080
PENGYPLEXITY_API_KEY=pgy_…         # step 1
```

The bot loads `.env` itself, and anything already set in the environment
wins. (The web app does not read `.env`; it needs the variables exported.)

**4. Check, then run.**

```bash
uv run pengyplexity-discord --check   # verifies the settings and the API key
uv run pengyplexity-discord
```

## Using it

| Where | What happens |
| --- | --- |
| **@mention in a channel** | Starts a conversation. The bot opens a thread named after the question and answers there. |
| **In a thread the bot started** | Every message is a follow-up; no mention needed. |
| **Reply to one of its answers** | Continues that answer's conversation (for inline replies without threads). |
| **@mention in reply to someone's message** | That message is quoted into the question, so "@bot is this true?" works. |
| **DM** (when enabled) | One running conversation per person; send `!new` to start fresh. |

While it works, the bot edits a single progress message: first the tool
activity ("Searching the web…"), then the answer as it is written. React
**⏹️** on that message to stop it; only the person who asked can do this. The
final answer replaces the progress message and is split across messages if it
is over 2,000 characters. Sources are listed without link previews. Charts
and images are attached in a message of their own.

Questions sent while the conversation is still answering get a ⏳ reaction
and are answered in order. Answers never ping anyone: `@everyone` or a user
mention in an answer is shown as text only.

## Settings

| Variable | Default | Meaning |
| --- | --- | --- |
| `DISCORD_BOT_TOKEN` | *(required)* | The bot token from the Developer Portal |
| `PENGYPLEXITY_API_URL` | *(required)* | Where the app is; with or without `/api/v1` |
| `PENGYPLEXITY_API_KEY` | *(required)* | The bot user's API key (`pgy_…`) |
| `PENGYPLEXITY_DISCORD_CHANNELS` | *(all)* | Comma-separated channel IDs to answer in; threads count as their parent channel |
| `PENGYPLEXITY_DISCORD_ALLOW_DMS` | `0` | Answer direct messages |
| `PENGYPLEXITY_DISCORD_THREADS` | `1` | Open a thread per question; `0` replies inline in the channel |
| `PENGYPLEXITY_DISCORD_USER_RATE_LIMIT` | `6` | Questions per Discord user per minute; `0` = no limit |
| `PENGYPLEXITY_DISCORD_MAX_UPLOAD_MB` | `10` | Largest artifact to attach (Discord's limit depends on the server's boosts) |
| `PENGYPLEXITY_DISCORD_STATE` | `<data dir>/discord.bson` | The conversation map |

The bot is also subject to the server's per-user API limits,
`PENGYPLEXITY_API_RATE_LIMIT` and `PENGYPLEXITY_API_MAX_CONCURRENT_TURNS`.
Every Discord user counts against the one bot account, so raise those on the
server if a busy server hits them.

## Running as a service (systemd)

Like the app, a **user** unit is enough:

```ini
# ~/.config/systemd/user/pengyplexity-discord.service
[Unit]
Description=Pengyplexity Discord bot
After=network-online.target pengyplexity.service
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=%h/pengyplexity
ExecStart=%h/.local/bin/uv run --extra discord pengyplexity-discord
Restart=always
RestartSec=10
Environment=PATH=%h/.local/bin:/usr/local/bin:/usr/bin:/bin

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now pengyplexity-discord.service
loginctl enable-linger "$USER"     # keep running without a login session
journalctl --user -u pengyplexity-discord -f
```

discord.py reconnects to Discord on its own. `Restart=always` covers the
rest, such as the app being down when the bot starts. The bot reads `.env`
from `WorkingDirectory`.

## Troubleshooting

**`turn on the Message Content Intent`** — enable it under Bot → Privileged
Gateway Intents, then restart the bot.

**`Discord rejected DISCORD_BOT_TOKEN`** — the token was reset or mistyped.
Reset it in the portal and update `.env`.

**`Pengyplexity rejected my API key`** — the key was revoked, or its user was
disabled or deleted. Make a new key on that user's Account page. Run
`--check` to confirm.

**The bot is online but never answers** — check that it can see the channel,
that the channel is in `PENGYPLEXITY_DISCORD_CHANNELS` if you set it, and
that you mentioned the bot itself rather than a role with the same name.

**It replies inline instead of opening threads** — it lacks *Create Public
Threads* in that channel. The log says so.

**Answers arrive in one lump** — the proxy is buffering the SSE response. Set
`proxy_buffering off` as in [INSTALLING.md](INSTALLING.md).
