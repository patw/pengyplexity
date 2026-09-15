# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **JSON API at `/api/v1`** — the groundwork for a Discord bot. A program can
  do everything a user does in the chat UI: create, list, rename and delete
  threads; ask questions with a plain JSON reply or an SSE stream; stop a
  running turn; download artifacts; and manage memories. Errors are always
  JSON with a stable `code`. Full reference in the new **API.md**, linked
  from the README.
- **Personal API keys** — created and revoked on the Account page. A key is
  shown once and only its SHA-256 hash is stored. A key acts as its user and
  stops working the moment that user is disabled or deleted. Keys can revoke
  keys (including themselves) but cannot create them.
- **API admission limits** — per-user questions per minute
  (`PENGYPLEXITY_API_RATE_LIMIT`, default 30) and threads answering at once
  (`PENGYPLEXITY_API_MAX_CONCURRENT_TURNS`, default 4). Asking in a thread
  that is already answering returns `409` instead of silently cancelling the
  running turn. Every refusal happens before anything is written.

### Changed

- The chat turn lifecycle moved from `web.py` into `turns.py`, and the browser
  and the API now share it, so the two cannot drift apart.
- A streamed turn that fails before producing any text no longer saves an
  empty assistant message. If it fails partway, the partial answer is saved
  with an `_[Error]_` note.
- If no agent is configured, a streamed question is no longer written to the
  thread.

## [0.1.0] - 2026-09-14

Initial public release. Pengyplexity is a Perplexity-style, sandboxed AI Q&A web
app: you ask a question, an LLM answers, and web search is treated as a
first-class tool — all inside a deliberately locked-down sandbox.

### Added

- **App + service layer** — Flask app factory with `AppState` dependency
  injection, so the model client, search, sandbox runner, sharing and image
  backends are all swappable (and fakeable in tests).
- **Safety model (the point of the app):**
  - `SAFE_TOOLS` — a curated, audited allowlist of tools with a per-tool
    rationale (`sandbox/toolpolicy.py`); there is no elevated/sudo tool.
  - Path confinement (`sandbox/confine.py`) — every file path is resolved
    inside the caller's per-thread workspace.
  - Sandboxed execution (`sandbox/executors.py`) — `run_python` / `run_bash`
    run under bubblewrap with an empty tmpfs root, a read-only `/usr`,
    `--unshare-net`, `--cap-drop ALL`, and memory/CPU/wall-clock rlimits.
  - A dedicated sandbox Python environment (matplotlib/numpy/pandas) that is
    built on the host once and bind-mounted read-only at `/pyenv`.
  - No self-modification: the app cannot edit its own skills, its own code, or
    the host.
- **Answers** — SSE streaming (token / activity / done / error events), a
  non-streaming fallback, in-turn cancellation, and source citation extraction.
- **Research** — web search + `fetch_url` tool loop, and a bounded
  deep-research loop that produces a markdown report rendered to HTML and
  downloadable as PDF.
- **Artifacts** — sandboxed chart generation, `generate_image` / `edit_image`
  integration, a per-user workspace gallery, and workspace ZIP download.
- **Sharing** — text/HTML sharing to tclip and image sharing to Pengyshare.
- **Auth** — session login with `werkzeug.security` password hashing,
  admin-gated user CRUD, and **no self-sign-up** (the first admin is created
  with `pengyplexity-admin create-admin`).
- **Memory** — a moofile-backed memory store with semantic + lexical search and
  an active/superseded/deprecated lifecycle.
- **Admin settings** — most tunables are overridable at runtime from
  `/admin/settings` without a restart; env vars are the startup default.
- **Tests** — an offline pytest suite (584 passing, 1 skipped when `reportlab`
  is absent) with fakes for the model, search, sandbox runner, sharing and
  image backends. No network and no live `bwrap` required. Includes
  `test_readme.py`, which asserts the documentation contract (the README stays a
  front door, and the detail lives in `SPEC.md` / `INSTALLING.md`), and
  `test_gitignore.py`, which asserts that no moofile store — including its
  `.cache` / `.lock` / `.meta` sidecars and case variants — is ever committable.
- **Docs & repo hygiene** — three documents with three jobs: a short,
  marketing-oriented **README** (pitch, screenshot, safety at a glance, quick
  start, links out), **SPEC.md** for the engineering detail (architecture, the
  safety model in full, features, routes, data model, every setting,
  non-goals), and **INSTALLING.md** for installing and operating (requirements,
  first run, systemd service, nginx + Let's Encrypt, day-to-day operation,
  troubleshooting). Plus an `.env.example` template, this changelog, and the
  MIT license.

### Security

- The sandbox is the security boundary: no host filesystem, no network, no
  capabilities, no elevated tools, and no way for the model's code to reach the
  host process tree.

### Notes

- Requires Linux with `bubblewrap` installed on the host, plus an
  OpenAI-compatible model endpoint.
- **Docker is a non-goal.** A default container cannot create the namespaces
  `bwrap` needs, so containerising the app would silently disable the sandbox
  rather than contain it.
