"""A persistent Discord bot that answers with Pengyplexity.

The bot is a separate process and an ordinary API client: it holds one API
key for one dedicated Pengyplexity user and talks to ``/api/v1`` like any
other program (see API.md). It never touches the stores or the sandbox
directly, so it can run on the same host as the app or anywhere that can
reach it.

Layout:

* :mod:`.config` — settings from the environment / ``.env`` (stdlib only).
* :mod:`.apiclient` — the async HTTP client (threads, streamed answers, stop,
  artifact download).
* :mod:`.conversations` — the persistent map from a Discord conversation to
  the Pengyplexity thread holding its history.
* :mod:`.render` — pure text shaping for Discord (message limits, sources,
  progress previews).
* :mod:`.session` — one question from start to delivered answer, against a
  small ``Surface`` interface so it is testable without Discord.
* :mod:`.bot` — the discord.py glue: routing, threads, live edits, Stop.
* :mod:`.cli` — the ``pengyplexity-discord`` entry point.

Only :mod:`.bot` and :mod:`.cli` import discord.py (the ``discord`` extra).
"""
