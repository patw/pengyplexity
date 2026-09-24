# Pengyplexity — Discord bot

Put Pengyplexity in a Discord server: @mention the bot with a question and the
answer streams into the channel, with any charts or images attached. It reads
the recent messages around it, so it can follow what the room is talking
about, and it can look at images people post. It uses the same agent, sandbox
and settings as the web UI.

> For the overview, see the [README](README.md). The bot is an ordinary
> client of the [JSON API](API.md), and deploying the app itself is in
> [INSTALLING.md](INSTALLING.md).

**Contents:** [How it works](#how-it-works) · [Setup](#setup) ·
[Using it](#using-it) · [Giving it a voice](#giving-it-a-voice) ·
[Settings](#settings) ·
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
  A channel is **one rolling conversation** (named `Discord #channel`), so the
  bot keeps what it learned there instead of starting fresh at every mention.
- That conversation **starts over** once the room has been quiet for
  `PENGYPLEXITY_DISCORD_IDLE_HOURS`, or after `PENGYPLEXITY_DISCORD_MAX_TURNS`
  questions. A thread is replayed to the model in full on every turn, so a
  channel bound to one thread for a month ends up resending tens of thousands
  of tokens to be asked the time. Little is lost when it rolls over: the next
  question still carries the room's recent messages, and anything worth
  keeping should be a memory by then. A reply to one of the bot's answers
  always continues *that* conversation, however old it is.
- Each question carries the channel messages posted **since the bot last
  answered there**, up to `PENGYPLEXITY_DISCORD_HISTORY`. Only the new ones:
  everything older is already in the Pengyplexity thread, so the same text is
  never paid for twice.
- **Everyone is named.** Each message is attributed to its author's alias and
  unique `@handle`, the people in the conversation are listed once with their
  permanent Discord ids, and the question says who asked. Without this the
  agent hears a single anonymous voice and saves memories about "the user",
  which is worthless in a room with dozens of people in it.
- The bot still only speaks when spoken to — an @mention, a reply to one of
  its answers, or a message in a thread it started. It never chimes in
  uninvited.
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
and create an API key. Copy it; it is shown once. While you are in the admin
pages, give that user its own system message so it answers like a person in a
channel rather than filing a report — see
[Giving it a voice](#giving-it-a-voice).

**2. Create the Discord application.** In the
[Developer Portal](https://discord.com/developers/applications):

- **New Application** → **Bot** → **Reset Token**, and copy the token.
- Under **Privileged Gateway Intents**, turn on **Message Content Intent**.
  The bot needs it to read follow-ups in its threads, which don't mention it,
  and to read the surrounding channel messages it uses as context.
- **OAuth2 → URL Generator**: scope `bot`, with the permissions *View
  Channels*, *Send Messages*, *Send Messages in Threads*, *Create Public
  Threads*, *Read Message History*, *Attach Files*, *Add Reactions* and
  *Embed Links*. Open the generated URL to invite the bot to your server.
  *Read Message History* is what lets it see the conversation around a
  question; without it, it still answers, but only from the question itself.

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
| **@mention in a channel** | Answered in the channel, with the recent messages there as context. The channel is one continuing conversation. A long answer moves into a thread (see below). |
| **@mention with an image** | The image is passed to the agent, which fetches it and looks at it. A picture with no text still counts as a question. |
| **Reply to one of its answers** | Continues that answer's conversation. |
| **@mention in reply to someone's message** | That message is quoted into the question, so "@bot is this true?" works. |
| **In a thread the bot started** | Every message is a follow-up; no mention needed. That includes a thread a long answer moved into. |
| **DM** (when enabled) | One running conversation per person; send `!new` to start fresh. |

While it works, the bot edits a single progress message: a penguin doing
something penguin-ish ("🐧 Dreaming of fish…"), then the answer as it is
written. That line is flavour, not a report — a channel is no place to watch
tool calls, so the agent's real activity labels are dropped and a random one
is shown instead. The final answer replaces the progress message and is split
across messages if it is over 2,000 characters. Charts and images are attached
in a message of their own.

**Long answers go into a thread.** Quick answers stay in the channel, but
once an answer runs past `PENGYPLEXITY_DISCORD_THREAD_OVER` characters (1,000
by default) the bot starts a thread from its reply and carries on writing in
there. The channel keeps just the answer's opening paragraph and a
"🧵 More in the thread" line, so a research write-up doesn't bury the room.
The switch happens while the answer streams, so the channel never shows the
whole thing, and charts go into the thread with it. Anything said in that
thread is a follow-up — no mention needed — and it continues the same
conversation. The opening paragraph works best when the bot leads with the
answer, which the system message in [Giving it a voice](#giving-it-a-voice)
asks it to. Set the variable to `0` to keep everything in the channel.

**There is no stop button.** A turn runs to completion; stopping one part-way
is a web-UI affordance and stays there.

Images work the other way round from everything else: the API takes a string,
so a picture cannot ride along with the question. Instead the bot passes the
picture's URL, and the agent downloads it into the thread workspace and views
it with its own image tools. That means an image the bot can see is one the
web will serve — it does not re-upload the bytes itself.

All three ways a picture reaches a channel count: a file attachment, a bare
image link in the text, and a link Discord resolved into an embed (an imgur
page, a Tenor GIF). So do pictures in the room's **recent messages**, not just
in the message that mentioned the bot — "what's in the screenshot above?" is
an ordinary question. Up to six pictures are offered per question, the ones on
the question itself first, then the channel's newest.

Two things it still cannot see. A picture posted before the bot last answered
in that channel is not re-offered — its URL already went to the Pengyplexity
thread on an earlier turn, and re-sending it every turn would pay for it
every turn. And a link pasted in the *same* message that mentions the bot is
only seen if the URL itself ends in `.png`/`.jpg`/etc., because Discord
attaches the embed a moment after the message arrives.

Questions sent while the conversation is still answering get a ⏳ reaction
and are answered in order. Answers never ping anyone: `@everyone` or a user
mention in an answer is shown as text only.

## Giving it a voice

Out of the box the agent answers like a research tool: structured markdown
ending in a numbered list of sources. That reads well in the web UI and badly
in a chat channel, where it comes across as a bot filing a report. A channel
also asks two things of it that a single-user web session does not: work out
which of several people it is talking to, and look at the pictures they post.

Fix it by giving the bot's Pengyplexity user its own instructions, under
**Admin → User Management → the bot's row → System message**. A user's
instructions are *appended* to the system prompt rather than replacing it, so
the agent keeps everything it knows about charts, images, memory and its
sandbox — you are adding to it, not replacing it. Other users, including
whoever uses the web UI, are unaffected.

A starting point — the first bullets are about tone, the rest about the two
habits a channel needs:

```text
You are talking in a Discord channel, not writing a report. These
instructions override anything above about format and citations.

- Keep it to a few sentences. No headings, no bullet lists, no bold labels,
  unless someone actually asks for a list.
- Lead with the answer. No preamble and no restating the question.
- Still search the web before answering anything factual, but never add a
  "Sources:" section and never use [n] citation markers. If one link is
  genuinely worth having, put it in a sentence as <https://example.com>.
- Talk like a person: contractions, plain words, dry humour when it fits.
- If you don't know, say so in one sentence rather than hedging at length.
- Before you answer, search_memory for whoever is asking — by display name
  and by @handle, both of which the question tells you. You are talking to
  many different people in the same account, each conversation here starts
  over often, and what you already know about this one is in your memories
  rather than in front of you. Do it even when the question looks
  self-contained: knowing which of them asked is usually the difference
  between a useful answer and a generic one.
- Search memory for the *subject* too when a question refers to something
  the room has been working on ("how's the deploy going?").
- Save what is worth keeping about a person, and say in the memory who it is
  about, with their @handle. A memory that says "the user" is worthless here.
- When a question is about a picture, look at it. The question lists the URL
  of every picture in play: download_file it, then read_image. Never answer
  about an image from its filename or from what people said about it.
```

The bot never appends a sources list of its own, so the only links that
appear are the ones the model chose to write into the answer.

## Settings

| Variable | Default | Meaning |
| --- | --- | --- |
| `DISCORD_BOT_TOKEN` | *(required)* | The bot token from the Developer Portal |
| `PENGYPLEXITY_API_URL` | *(required)* | Where the app is; with or without `/api/v1` |
| `PENGYPLEXITY_API_KEY` | *(required)* | The bot user's API key (`pgy_…`) |
| `PENGYPLEXITY_DISCORD_CHANNELS` | *(all)* | Comma-separated channel IDs to answer in; threads count as their parent channel |
| `PENGYPLEXITY_DISCORD_ALLOW_DMS` | `0` | Answer direct messages |
| `PENGYPLEXITY_DISCORD_THREADS` | `0` | `0` answers in the channel, one rolling conversation per channel; `1` opens a Discord thread per question |
| `PENGYPLEXITY_DISCORD_THREAD_OVER` | `1000` | An in-channel answer longer than this many characters moves into a thread, leaving its opening in the channel; `0` = never |
| `PENGYPLEXITY_DISCORD_HISTORY` | `20` | Channel messages of context sent with a question; `0` = none |
| `PENGYPLEXITY_DISCORD_IDLE_HOURS` | `3` | Start a fresh conversation once the room has been quiet this long; `0` = never on idle |
| `PENGYPLEXITY_DISCORD_MAX_TURNS` | `20` | ...or after this many questions; `0` = no cap |
| `PENGYPLEXITY_DISCORD_IMAGES` | `1` | Pass image attachments to the agent to fetch and look at |
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

**It answers in the channel instead of opening threads** — that is the
default. Set `PENGYPLEXITY_DISCORD_THREADS=1` for a thread per question. If it
is already `1` and you still get channel replies, the bot lacks *Create Public
Threads* there; the log says so.

**Long answers still land in the channel** — the bot needs *Create Public
Threads* to move them; without it, it answers in the channel and logs a
warning. Check `PENGYPLEXITY_DISCORD_THREAD_OVER` isn't `0`.

**It doesn't seem to know what the channel was talking about** — it needs
*Read Message History* in that channel, and `PENGYPLEXITY_DISCORD_HISTORY`
above `0`. Note it only ever sees messages posted *after* its last answer
there, so the very first question in a channel has little to go on.

**It says it can't see an image** — the agent fetches the attachment from
Discord itself, so the app's host needs outbound network access to
`cdn.discordapp.com`. Check `PENGYPLEXITY_DISCORD_IMAGES=1` too.

**Answers arrive in one lump** — the proxy is buffering the SSE response. Set
`proxy_buffering off` as in [INSTALLING.md](INSTALLING.md).
