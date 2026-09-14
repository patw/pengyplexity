# Pengyplexity

A **Perplexity-style, sandboxed AI Q&A web app** in Python/Flask. A user asks a
question and an LLM answers it — with **web search as the star** — using a model
client and tool engine whose design is carried over from the author's Pengy agent
codebase (the relevant parts live here; Pengy itself is not a dependency).

The defining difference from a full agent harness: **Pengyplexity is a
deliberately locked-down subset.** It exposes only a curated set of *safe* tools,
confines every file path and every `run_python`/`run_bash` to a per-thread
workspace (bubblewrap), and **can never self-modify its own skills or the host.**

**Nothing here reaches the system filesystem, a host network socket, or the host
process tree.** The sandbox is not a feature of this app — it is the point of it.

---

## Architecture at a glance

```
pengyplexity/
  app.py                # Flask app factory + AppState (service injection point)
  config.py             # env-derived Config (stdlib only, safe to import offline)
  web.py                # web_bp: login, chat, thread history, ask (SSE)
  admin.py              # admin_bp: list/add/enable/disable/reset/delete users
  cli.py                # `create-admin` bootstrap (no self-sign-up)
  core/
    modelclient.py      # OpenAI-compatible chat client (reuses pengy.core.llm_client shape)
    agent.py            # one-turn tool loop: system prompt + SAFE_TOOLS -> answer + sources
    streaming.py        # SSE helpers (token / activity / done / error events)
    search.py           # web_search + fetch_url orchestration, citation extraction
    deepresearch.py     # bounded multi-query research loop -> report markdown
    artifacts.py        # run-chart-in-sandbox; report markdown -> HTML -> PDF
    store.py            # moofile collections: users, threads (embedded msgs), shares, settings
    auth.py             # session login, pbkdf2/scrypt hashing, admin-gated user CRUD
    sharing.py          # tclip (text/HTML) + pengyshare (image) upload helpers
    images.py           # image_gen / image_edit skill integration
  sandbox/
    confine.py          # resolve(root, raw_path) — the path-confinement guard
    executors.py        # Runner interface + BwrapRunner (real) + FakeRunner (tests)
    toolpolicy.py       # SAFE_TOOLS allowlist + per-tool rationale + the audit
  templates/            # Jinja: base, login, chat, admin users
  static/               # Bootstrap-flavored CSS/JS (dependency-free)
  tests/                # offline pytest suite (no network, no live bwrap)
```

### Request flow

1. User posts a question (`POST /chat/<thread_id>/ask`).
2. The **agent** sends the system prompt + the `SAFE_TOOLS` schema to the model
   client and runs the tool-call loop (capped at
   `PENGYPLEXITY_MAX_AGENT_ITERATIONS`).
3. Every file tool call is resolved through `confine.resolve` so the path is
   forced inside the per-thread workspace. Every `run_python`/`run_bash` is
   dispatched to the **bwrap runner** (see below).
4. When the model stops emitting tool calls, the final answer + extracted
   **sources** (title + URL) are returned and streamed to the browser over **SSE**.

### Services are injectable

`create_app` attaches an `AppState` dataclass to
`app.extensions["pengyplexity"]` holding `config`, `store`, `auth`, `agent`,
`runner`, and `search`. Every external boundary (model, search, sandbox executor,
clip/pengyshare upload, image gen/edit) is an interface behind an injectable
implementation — which is exactly what lets the **entire test suite run offline**.

---

## Safety model (the point of the app)

Pengyplexity's safety is **structural**, not a matter of prompt politeness. Three
layers each block escape on their own, and the test suite *asserts* the contract
rather than merely relying on it.

### 1. Curated tool allowlist (`sandbox/toolpolicy.py`)

`SAFE_TOOLS` is an explicit `frozenset` re-scoping the 16 tool **names** in
`pengy.core.tools.TOOLS`, plus 3 of Pengyplexity's own capability wrappers
(`make_chart`, `generate_image`, `edit_image`, `save_memory`, `search_memory` —
see below), so 21 total:

| Category | Tools | How it's made safe |
| --- | --- | --- |
| Read-only | `read_file`, `read_multiple_files`, `read_image`, `directory_tree`, `search_content`, `glob` | Paths confined to the workspace |
| The app's job (web as star) | `web_search`, `fetch_url`, `download_file` | Network only via these tools |
| Workspace-scoped writers | `write_file`, `replace_in_file`, `apply_changes` | Paths confined to the workspace |
| Sandboxed runners | `run_python`, `run_bash` | Executed only inside bwrap |
| Harness (no host access) | `todowrite`, `ask_user_question` | No filesystem / no host effect |
| App capability wrappers | `make_chart`, `generate_image`, `edit_image` | Call `core/artifacts` / `core/images` (never the raw host-shelling skill); output confined to the workspace and registered as a servable artifact |
| App capability wrappers | `save_memory`, `search_memory` | Call `core/memory.MemoryStore`, scoped to the calling user's own rows only; never a network call, never shared with other users |

Key points:

- **`run_bash`'s `elevated`/sudo parameter is stripped** by
  `toolpolicy.scrub_tool_schema`, so privilege escalation is not even expressible.
- **Raw skills are not tools.** The raw Pengy skill names — `plot`, `image_gen`,
  `image_edit`, `clip`, `pengyshare` — never reach the model directly; they live in
  `FORBIDDEN_TOOL_NAMES`. Instead the app exposes its own, differently-named
  wrappers (`make_chart`, `generate_image`, `edit_image`) that call the same
  dedicated, injectable, sandboxed code paths (`core/artifacts`/`core/images`) and
  register real, servable artifacts. Host-shelling skills (`screenshot`, `tts`,
  `email`, `scheduler`, …) are excluded entirely with no wrapper. `test_toolpolicy`
  asserts `SAFE_TOOLS.isdisjoint(FORBIDDEN_TOOL_NAMES)`.
- **The frontend hands the model only `SAFE_TOOLS`** (with descriptions edited to say
  "paths are relative to your workspace"), never the full `TOOLS` inventory.

### 2. Path confinement (`sandbox/confine.py`)

Each thread owns a sandbox root: `~/.pengyplexity/workspaces/<user>/<thread_id>/`.
`resolve(root, raw_path)` applies a **two-stage guard**:

1. **Lexical** — `os.path.normpath(Path(root) / raw)` with no filesystem access, so
   `..` traversal and absolute-path escapes are rejected even for a file that does
   not yet exist (the `write_file` destination case).
2. **Realpath** — `.resolve(strict=False)` re-checked for containment, so a symlink
   pointing out of the root is caught.

The contract: a **relative** path is joined onto the root; an **absolute** path is
accepted *only* if it already resolves inside the root; an **empty** path means the
root itself. `workspace(user, thread, base=None)` returns the per-thread root with
the user/thread ids sanitised to a single path element (a hostile id like
`../../etc` cannot traverse out of the workspaces tree). A violation raises
`OutsideWorkspaceError` instead of silently widening.

### 3. Isolated code execution (`sandbox/executors.py`)

`run_python` and `run_bash` are **never run on the host**. `BwrapRunner` builds the
escape-proof `bwrap` argv:

```
bwrap --unshare-all --die-with-parent --new-session \
      --tmpfs / --ro-bind /usr /usr \
      [--symlink|--ro-bind for /bin /sbin /lib /lib64 /lib32, if present] \
      --ro-bind-try /etc/fonts /etc/fonts \
      --dev /dev --proc /proc --tmpfs /tmp \
      --bind <workspace> /work --chdir /work \
      --unshare-net --unshare-pid --unshare-uts --unshare-cgroup \
      --cap-drop ALL --uid 65534 --gid 65534 \
      --ro-bind <sandbox-venv> /pyenv \
      --clearenv --setenv PATH /pyenv/bin:/usr/local/bin:/usr/bin:/bin \
      --setenv HOME /tmp --setenv TMPDIR /tmp ... \
      [--rlimit-as <mem_bytes> --rlimit-cpu <cpu_seconds>, pre-bwrap-0.11 only] \
      -- /bin/sh -c <script>
```

A throwaway, **network-dropped**, **non-root**, **capability-dropped** sandbox
with a hard wall-clock timeout (runaway code is killed by the parent via
`--die-with-parent`). The sandbox root is an **empty `tmpfs`**, not the real
host root: only `/usr` (read-only, plus its usual top-level shims) is bound
in — just enough for Python/bash/coreutils to run. `/home`, `/root`, `/etc`,
and every other host path simply do not exist inside the sandbox, so
`curl` / socket / sudo escape *and* reading arbitrary host files are both
structurally impossible. (An earlier version used `--ro-bind / /`, which is
read-only but still mounts the *entire* host filesystem — bwrap namespaces
don't give you an isolated filesystem the way a container image does; a
sandboxed `ls ~` could see the real host home directory. `--tmpfs /` +
a curated `/usr` closes that.) The model's cwd is always the thread's
workspace, the only writable bind. `test_executors.py` asserts this exact
argv **without ever invoking a live bwrap** — that is the escape-proof
contract, checked in CI.

**The environment is cleared too (`--clearenv`).** Namespaces isolate the
filesystem, the network and the process tree; they do nothing about the
environment, which bwrap otherwise passes straight through from the parent.
That parent is the Flask process, so a single `env` from `run_bash` would
have printed the app's LLM API key, the image-generation key, the real host
username and home directory, and the session's IPC socket paths. The sandbox
now gets exactly ten variables, all fixed constants, listed in
`_BASE_ENV`. `test_executors.py::TestEnvironmentIsolation` pins that set.

**The sandbox Python environment (`sandbox/pythonenv.py`).** Because the only
filesystem the sandbox has is a read-only `/usr`, the interpreter inside it is
the host's bare system Python — which normally has no matplotlib, no numpy, no
pandas, so *every* chart script failed on `import matplotlib`. A separate venv
is therefore built on the host (once) and bind-mounted read-only at `/pyenv`
with `/pyenv/bin` first on `PATH`. It is built against `/usr/bin/python3`
specifically: `uv venv` left to itself picks a uv-managed CPython under the
user's home, which is not mounted inside the sandbox, so the interpreter
symlink dangles and the packages stay unimportable. Build it ahead of time
with `pengyplexity-admin build-sandbox`; otherwise it is built on first use.
The venv holds no application code and no credentials, is mounted read-only,
and changes nothing about the confinement contract above.

### No self-modification

The model has no write access outside its workspace, and there is no tool that
reaches the Pengy / Pengyplexity / skills trees. This is guaranteed structurally by
confinement — and reinforced by the system prompt.

---

## Features

- **Ask & answer** — an input box; the model answers with `web_search` first,
  citing **Sources** (title + URL), optionally fetching a URL for deeper reading.
- **Deep research** — decomposes a question into multiple web searches (bounded by
  `PENGYPLEXITY_RESEARCH_QUERY_BUDGET`), synthesises a **report** (markdown), and
  renders it to HTML in the browser and **downloads it as PDF** (reportlab).
- **Artifacts** — the model calls `make_chart` to run a script in its workspace and
  produce a **chart** (matplotlib → PNG), registered as an artifact and rendered
  inline in the chat (both on full page render and, via an SSE `artifact` event,
  during a streamed answer). Reports (markdown → HTML/PDF) work the same way.
- **Chat history** — a left sidebar of previous threads; click to reopen.
- **Multi-user + login** — a login page. **No self sign-up.** Users are created and
  managed only by an admin (see *Bootstrap* and *Admin UI* below).
- **Account** — any logged-in user can change their own password at
  `/account/password` (distinct from the admin-only password reset).
- **Share** — pushes a message/thread to `tclip` (text/HTML) or `pengyshare`
  (images), storing the returned URL on the message.
- **Images** — the model calls `generate_image` / `edit_image` (wrapping the
  `image_gen` / `image_edit` skills behind `core/images.ImageService`); the result
  is registered as an artifact and rendered inline, the same as a chart.
- **Workspace** (`/workspace`) — a gallery of every chart/image/report artifact
  across all of a user's threads, grouped by thread, with a "download as ZIP"
  button per-thread or for everything at once. Every path in the ZIP is
  re-confined to its owning thread's workspace before being read (same guard as
  `download_artifact`), so a stale/hostile artifact record can't be used to read
  outside the sandbox.
- **Memory** (`/memories`, `core/memory.py`) — a per-user, private notebook the
  model can write to (`save_memory`) and search (`search_memory`) to recall
  facts across conversations, modeled on `~/Personal/BotTalk`'s search/storage
  design: a moofile collection with BM25 lexical indexes plus a semantic vector
  index (moofile auto-embed), queried via lexical, semantic, or an RRF-fused
  hybrid search. Every edit appends to an `update_history` (with the prior
  value) rather than silently overwriting, and memories carry the same
  `active`/`superseded`/`deprecated` lifecycle as BotTalk posts. Unlike
  BotTalk's shared, cross-bot board, this store is private per user and never
  leaves the app — a completely different capability from the forbidden raw
  `bottalk` skill. The user can view, search, edit, and delete their own
  memories directly at `/memories`.
- **Admin settings** (`/admin/settings`, `core/settings.py`) — Pengy's
  settings dialog reimagined as one global, admin-only doc (Pengyplexity is
  multi-user; this isn't a per-user desktop preference). Live-editable, no
  restart needed:
  - **System message** — override the agent's system prompt entirely.
  - **Model connection** — base URL, API key, model name, temperature, LLM
    request timeout; changing base URL/API key/timeout rebuilds the
    underlying OpenAI SDK client on next use (`OpenAIModelClient.configure`).
  - **Agent & tool limits** — max agent iterations, tool output truncation
    (head+tail snip, like Pengy's `tool_output_max_chars`), download size cap,
    web tool timeout/User-Agent (`web_search`/`fetch_url`/`download_file`).
  - **Sandbox execution** — `run_python`/`run_bash`'s wall-clock timeout,
    memory limit, and CPU-time limit (the bwrap argv shape itself — no host
    network, no host filesystem, non-root, capability-dropped — stays
    structural and is not a setting).

  Every field falls back to the env-loaded `Config` default when unset. The
  shared agent/runner/model-client/tool-context are mutated in place before
  each turn (the same pattern `agent.workspace` already used), so a saved
  setting takes effect on the very next chat turn.
- **Theming** (`/account/password`, `core/theme.py`, `static/css/theme.css`) —
  a direct port of Pengy's Qt theme system: 3 modes (system/light/dark) × 8
  accents (default/blue/teal/green/orange/red/pink/purple), each accent
  carrying its own light *and* dark surface tint (not just a highlight
  color) — the exact `BASE_THEMES`/`ACCENTS`/`ACCENT_SURFACES`/`SEMANTIC`
  tables from `~/Personal/Pengy/pengy/ui/theme.py`, transcribed to CSS custom
  properties instead of a Qt stylesheet. User-selectable on the Account page
  (radio buttons + accent swatches) with an instant JS preview before
  saving; the choice is stored per-user (`users.theme_mode`/`theme_accent`)
  and applied on every page via a `data-theme`/`data-accent` attribute pair
  on `<html>`, injected by a Flask context processor so no route has to pass
  it explicitly. "System" mode omits `data-theme` and follows the browser's
  `prefers-color-scheme`; light/dark are explicit overrides. **Unlike Pengy
  (whose default is "system"/blue), a new Pengyplexity user — and every
  unauthenticated page — defaults to light + the orange accent**, per its own
  explicit CSS `:root` base plus a stamped `data-accent="orange"`, regardless
  of the visitor's OS preference. The accent *named* "default" is still blue
  and still selectable; it is simply no longer the starting point.

### Routes

| Route | Purpose |
| --- | --- |
| `GET/POST /login`, `POST /logout` | Session auth (no sign-up) |
| `GET/POST /account/password` | Self-service password change + theme picker (any logged-in user) |
| `POST /account/theme` | Save theme mode + accent preference |
| `GET /chat`, `/chat/new`, `/chat/<thread_id>` | Chat UI + thread history sidebar |
| `POST /chat/<thread_id>/ask` | Ask a question (SSE-streamed answer) |
| `GET /workspace` | Cross-thread artifact gallery |
| `GET /workspace/zip[?thread_id=]` | Download all (or one thread's) artifacts as a ZIP |
| `GET/POST /memories` | List/search memories (`?q=`), create a new one |
| `GET/POST /memories/<id>/edit` | View/edit one memory |
| `POST /memories/<id>/delete` | Delete a memory |
| `GET /admin` | List users (admin only) |
| `POST /admin/users` | Create a user (admin only) |
| `POST /admin/users/<id>/enable|disable|reset-password|delete` | User lifecycle (admin only) |
| `GET/POST /admin/settings` | View/edit global settings — system message, model connection, agent/tool limits, sandbox execution limits (admin only) |
| `POST /admin/settings/reset` | Clear all setting overrides back to `Config`/env defaults (admin only) |
| `GET /healthz` | Liveness probe |

Non-admins get a 403 on `/admin`; unauthenticated requests redirect to `/login`.

---

## Data model (moofile)

Store file: `~/.pengyplexity/pengyplexity.bson` (override with
`PENGYPLEXITY_STORE`). Collections:

- **users** — `{ _id, username, password_hash, is_admin, enabled, created, last_login }`
  (passwords hashed via `werkzeug.security`, never plaintext).
- **threads** — `{ _id, owner, title, created, updated, messages: [...] }`, with
  messages embedded: `{ role, type (text|artifact|image|share), content,
  sources: [{title,url}], created }`.
- **shares** — `{ _id, thread_id, message_index, kind (text|image|report), url, created }`.
- **settings** — a single admin-configurable doc (e.g. the research query budget).

A separate store file, `~/.pengyplexity/memories.bson` (override with the
`memory_store_file` config), holds the **memories** collection: `{ _id, owner,
title, summary, tags, body, status (active|superseded|deprecated), created,
updated, update_history: [{editor, timestamp, changed, prior}] }`, plus the
internal `search_text`/`search_embedding` fields moofile's auto-embed
maintains (never returned to templates or tool results — see
`memory.strip_internal`).

---

## Configuration

All config is env-driven with sane defaults (`PENGYPLEXITY_*`). Secrets come from
the environment and are never committed — see `.env.example` for a documented
template of every setting below.

| Variable | Default | Meaning |
| --- | --- | --- |
| `PENGYPLEXITY_DATA_DIR` | `~/.pengyplexity` | Root for the store + per-thread workspaces |
| `PENGYPLEXITY_STORE` | `<data_dir>/pengyplexity.bson` | Explicit store file path |
| `PENGYPLEXITY_MODEL_BASE` | `http://10.0.23.2:8086/v1` | OpenAI-compatible endpoint |
| `PENGYPLEXITY_MODEL_KEY` | *(empty)* | API key |
| `PENGYPLEXITY_MODEL_NAME` | `gpt-4o-mini` | Model name |
| `PENGYPLEXITY_MODEL_TEMPERATURE` | `0.3` | Sampling temperature |
| `PENGYPLEXITY_LLM_TIMEOUT` | `300` | HTTP timeout (s) per LLM API request |
| `PENGYPLEXITY_MAX_AGENT_ITERATIONS` | `12` | Hard tool-loop cap (runaway guard) |
| `PENGYPLEXITY_RESEARCH_QUERY_BUDGET` | `6` | Max web queries per deep-research run |
| `PENGYPLEXITY_TOOL_OUTPUT_MAX_CHARS` | `250000` | Snip (head+tail) tool output longer than this; `0` = no limit |
| `PENGYPLEXITY_DOWNLOAD_MAX_MB` | `100` | Max size for `download_file`; `0` = unlimited |
| `PENGYPLEXITY_TOOL_NETWORK_TIMEOUT` | `15` | Timeout (s) for `web_search`/`fetch_url`/`download_file` |
| `PENGYPLEXITY_USER_AGENT` | `Mozilla/5.0 (Pengyplexity)` | User-Agent sent by `web_search`/`fetch_url`/`download_file` |
| `PENGYPLEXITY_SECRET_KEY` | dev placeholder | Flask session key — **set in prod** |
| `PENGYPLEXITY_DEBUG` | `0` | Flask debug mode |
| `PENGYPLEXITY_EXEC_TIMEOUT` | `30` | Wall-clock timeout (s) per `run_*` |
| `PENGYPLEXITY_EXEC_MEM_BYTES` | `512 MiB` | bwrap `--rlimit-as` |
| `PENGYPLEXITY_EXEC_CPU_SECONDS` | `30` | bwrap `--rlimit-cpu` |
| `PENGYPLEXITY_SANDBOX_VENV` | `<data_dir>/sandbox-venv` | Python environment bind-mounted into the sandbox at `/pyenv` |
| `PENGYPLEXITY_SANDBOX_AUTOBUILD` | `1` | Build that environment on first use; `0` requires `build-sandbox` |
| `PENGYPLEXITY_MEMORY_SIGNAL_FLOOR` | `0.33` | Minimum semantic similarity for a memory to be returned at all |
| `PENGYPLEXITY_MEMORY_SIGNAL_CONFIDENT` | `0.50` | Similarity at or above which a memory is a strong match, not a lead |
| `PENGYPLEXITY_MEMORY_SEMANTIC` | `1` | Enable the memory store's semantic search leg (moofile auto-embed). `0` = lexical (BM25) only, no embedding model load |

All of the above (except storage paths, `SECRET_KEY`, and `DEBUG`) can also be
overridden live from `/admin/settings` without a restart — see *Admin
settings* above; the env var is just the startup default.

---

## Setup & running

Requires **Python 3.11+**, `uv`, and Linux with `bwrap` (bubblewrap) installed
on the host for live sandboxed execution. The test suite does **not** need
`bwrap`.

```bash
# 1. Install the project (uv-managed, resolves flask/moofile/ddgs/markdown/reportlab/pillow)
uv sync

# 2. Bootstrap the first admin (no self-sign-up exists)
uv run python -m pengyplexity.cli create-admin admin --password 'changeme'
#    ...or: PENGYPLEXITY_ADMIN_PASSWORD='changeme' uv run pengyplexity-admin create-admin admin

# 3. Build the sandbox's Python environment (matplotlib/numpy/pandas).
#    Optional — the app builds it on first use — but doing it here keeps the
#    first chart fast and surfaces a host without uv or network right away.
uv run pengyplexity-admin build-sandbox

# 4. Run the app
uv run flask --app pengyplexity.app:create_app run
```

Point the app at your model with the `PENGYPLEXITY_MODEL_*` env vars before step 3.

---

## Running as a service (systemd)

There is no server framework to configure — the app is a single process — so a
systemd **user** unit is enough, with no root and no system-wide install:

```ini
# ~/.config/systemd/user/pengyplexity.service
[Unit]
Description=Pengyplexity — sandboxed Perplexity-style AI Q&A web app
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=%h/pengyplexity
ExecStart=%h/.local/bin/uv run flask --app pengyplexity.app:create_app run --host 127.0.0.1 --port 5080
Restart=on-failure
RestartSec=5
Environment=PATH=%h/.local/bin:/usr/local/bin:/usr/bin:/bin
Environment=PENGYPLEXITY_MODEL_BASE=http://127.0.0.1:8086/v1
Environment=PENGYPLEXITY_MODEL_NAME=gpt-4o-mini
Environment=PENGYPLEXITY_MODEL_KEY=replace-me
Environment=PENGYPLEXITY_SECRET_KEY=replace-with-a-long-random-string
Environment=PENGYPLEXITY_DATA_DIR=%h/.pengyplexity

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now pengyplexity.service
loginctl enable-linger "$USER"     # start at boot, with no login session
```

Two details worth getting right:

- **Bind to `127.0.0.1`, not `0.0.0.0`.** A public bind skips TLS and the reverse
  proxy and hands the app to anything that can reach the port.
- **`loginctl enable-linger`** — without it a *user* unit only runs while that
  user has a session, so the app silently does not come back after a reboot.

`flask run` is Flask's development server. That is acceptable for a personal
instance behind a proxy on a trusted host; if you expect real traffic, put a
production WSGI server (waitress, gunicorn) behind the same unit instead. See
**Non-goals** for why "just put it in a container" is not the answer here.

---

## Deploying behind a reverse proxy (nginx + Let's Encrypt)

Order matters: point DNS at the host first, then stand up the HTTP vhost (so the
ACME challenge can be answered), issue the certificate, and only then switch the
vhost to HTTPS.

```nginx
# /etc/nginx/sites-available/pengyplexity.example.com
server {
  listen 443 ssl;
  server_name pengyplexity.example.com;

  ssl_certificate     /etc/letsencrypt/live/pengyplexity.example.com/fullchain.pem;
  ssl_certificate_key /etc/letsencrypt/live/pengyplexity.example.com/privkey.pem;

  client_max_body_size 1m;          # chat requests are small

  location / {
    proxy_pass http://127.0.0.1:5080;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;

    # Answers stream token-by-token as text/event-stream (SSE). With buffering
    # ON the answer instead arrives in one lump when the turn finishes — this
    # is the single most commonly missed line in an SSE deployment.
    proxy_http_version 1.1;
    proxy_set_header Connection "";
    proxy_buffering off;
    proxy_cache off;
    proxy_read_timeout 3600s;
    proxy_send_timeout 3600s;
  }
}

# Port 80: ACME challenges (KEEP — renewals use this) + redirect to HTTPS
server {
  listen 80;
  server_name pengyplexity.example.com;

  location /.well-known/acme-challenge/ { root /var/www/certbot; }
  location / { return 301 https://$host$request_uri; }
}
```

```bash
nginx -t && systemctl reload nginx
certbot certonly --webroot -w /var/www/certbot -d pengyplexity.example.com
nginx -t && systemctl reload nginx
```

Leave the `/.well-known/acme-challenge/` location in the port-80 block
permanently — certificate renewals keep using it after you switch to HTTPS.

---

## Operating

**Where state lives.** Everything is under `PENGYPLEXITY_DATA_DIR`
(`~/.pengyplexity` by default): the stores (`pengyplexity.bson`,
`memories.bson`), the sandbox environment (`sandbox-venv/`), and per-user
workspaces (`workspaces/`). Deleting that directory resets the instance — which
also makes it the thing to back up.

| Task | Command |
| --- | --- |
| Back up | `tar czf pengyplexity-$(date +%F).tgz -C "$HOME" .pengyplexity` |
| Upgrade | `git pull && uv sync && systemctl --user restart pengyplexity` |
| Restart | `systemctl --user restart pengyplexity` |
| Logs | `journalctl --user -u pengyplexity -f` |
| Rebuild sandbox env | `uv run pengyplexity-admin build-sandbox --force` |
| Add another admin | `uv run pengyplexity-admin create-admin <name>` |

- **Backups.** Stop the service first, or copy while it is idle — moofile is a
  single-file store, so a copy taken mid-write is not guaranteed consistent.
- **Users.** Manage them at `/admin/users` as an admin: create, enable/disable,
  reset password, delete. There is no self-sign-up, and `create-admin` refuses a
  username that already exists.
- **Live tuning.** `/admin/settings` overrides most values at runtime with no
  restart. The environment variable is only the startup default, so an override
  made there lasts until the next restart — set the env var too if you want it
  permanent.
- **Secrets.** `PENGYPLEXITY_SECRET_KEY` and the model key belong in the service
  environment (or an `EnvironmentFile=`), never in the repo. Sessions are signed
  with the secret, so changing it logs every user out.

---

## Running the tests (offline)

```bash
python -m pytest -q
```

The suite is **the objective function** and is green with **no network and no live
bwrap**. Everything external sits behind an injectable interface and is faked in
`pengyplexity/tests/conftest.py`:

- **Model** — `FakeModelClient` (canned chat/tool-call sequence).
- **Search** — `FakeSearchService` (canned results + fetch bodies).
- **Sandbox** — `FakeRunner` (records the command, returns canned output); the real
  `BwrapRunner.build_argv` is asserted *purely*, never executed.
- **Sharing** — `FakeSharingService` (canned URLs).
- **Images** — `FakeImageBackend` (writes a real 1×1 PNG, no PIL / no network).
- **Store** — points at a `tmp_path` (never `$HOME`).

Coverage highlights: `test_confine.py` (path-escape rejection), `test_executors.py`
(the exact bwrap argv contract), `test_toolpolicy.py` (allowlist exactness + no
forbidden tool), `test_agent.py` (tool loop, iteration cap, source citation),
`test_web.py` (login-required, ask, history, admin CRUD), and the feature suites for
search / deepresearch / artifacts / streaming / sharing / images / auth / store.

Two tests are skipped when `reportlab` is unavailable in the interpreter (the
PDF-rendering path); everything else runs fully offline.

---

## Troubleshooting

**`bwrap: Creating new namespace failed: Operation not permitted`.**
The host cannot create the namespaces the sandbox needs — typically
unprivileged user namespaces are disabled, or the app is running inside a
container. To confirm the sandbox itself works:

```bash
bwrap --ro-bind / / --unshare-net --cap-drop ALL echo ok
```

If that fails, `run_python` / `run_bash` will fail too, and the safety model is
not actually in force — fix the host before exposing the app. See **Non-goals**
for why containers are not the workaround.

**Every chart fails, or `import matplotlib` fails inside the sandbox.**
The sandbox mounts only a read-only `/usr`, so the interpreter it reaches is the
host's bare system Python. Build the environment explicitly and make sure
`PENGYPLEXITY_SANDBOX_VENV` (if set) points at a venv built against a *system*
interpreter:

```bash
uv run pengyplexity-admin build-sandbox --force
```

A venv created from a uv-managed interpreter under `~/.local/share/uv` is not
visible inside the sandbox; the build now detects that and refuses it rather
than leaving you with a silent `import matplotlib` failure.

**Answers appear all at once instead of streaming.**
Something is buffering the SSE response — set `proxy_buffering off;` on the
proxying location (see the nginx section above).

**Everyone is logged out after a restart.**
`PENGYPLEXITY_SECRET_KEY` is still the development placeholder, or it changes
between restarts. Set a fixed, random value in the service environment.

**Model errors (`connection refused`, `401`, `404`).**
Check that `PENGYPLEXITY_MODEL_BASE` points at an OpenAI-compatible endpoint
reachable *from the host*, that the path ends in `/v1`, and that
`PENGYPLEXITY_MODEL_KEY` matches it. `PENGYPLEXITY_MODEL_NAME` must be a model
the endpoint actually serves.

---

## Non-goals (hard)

- **No self-sign-up** — admins only, via the CLI or the admin UI.
- **No host control** — no `elevated`/sudo, no tool that reads/writes outside a
  thread's workspace, no tool that edits Pengy/Pengyplexity/skills, no host process
  tree or arbitrary host network. The app cannot "escape".
- **No skill self-modification** — guaranteed by confinement, reinforced by the
  system prompt.
- **No Docker/podman** — `bwrap` is the sandbox, and a container cannot run it
  for you. A default Docker container forbids the namespace syscalls `bwrap`
  needs (`bwrap: Creating new namespace failed: Operation not permitted`), so a
  containerised build would either fail at runtime or — worse — ship with the
  sandbox switched off while still advertising it. Run Pengyplexity on the host
  (or in a VM), not in a hardened container.

---

## License

MIT — see [LICENSE](LICENSE). © 2026 Pat Wendorf.
