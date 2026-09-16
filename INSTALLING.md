# Installing & operating Pengyplexity

Everything needed to get Pengyplexity running and to keep it running: the
requirements, a first run, running it as a service, putting it behind nginx with
HTTPS, day-to-day operation, and what to do when something breaks.

> New here? Read the [README](README.md) first. For internals — the sandbox
> argv, the data model, every setting — see [SPEC.md](SPEC.md).

## Requirements

| | |
| --- | --- |
| **OS** | Linux. The sandbox uses `bwrap` (bubblewrap), so macOS/Windows are out unless you have a Linux VM. |
| **Python** | 3.11 or newer. |
| **`uv`** | Used to install and run the project (<https://docs.astral.sh/uv/>). |
| **`bubblewrap`** | Required for live sandboxed execution (`apt install bubblewrap`). Not needed for the test suite. |
| **A model** | Any OpenAI-compatible `/chat/completions` endpoint. No model is bundled. |

Check the host can actually sandbox before going further — everything else is
moot if this fails:

```bash
bwrap --ro-bind / / --unshare-net --cap-drop ALL echo ok   # expect: ok
```

## Configuration you must set

Pengyplexity reads its configuration from the environment. Start from the
template — `.env.example` documents every setting, and [SPEC.md has the full
table](SPEC.md#configuration). The four that matter on a first install:

| Variable | Why |
| --- | --- |
| `PENGYPLEXITY_MODEL_BASE` | Where your OpenAI-compatible endpoint lives, e.g. `http://127.0.0.1:8086/v1` |
| `PENGYPLEXITY_MODEL_NAME` | The model to ask |
| `PENGYPLEXITY_SECRET_KEY` | Flask session key. **Set a real random value in production** — sessions are signed with it. |
| `PENGYPLEXITY_DATA_DIR` | Where all state lives (default `~/.pengyplexity`) |

The app reads the *environment*, not the file, so export it (or use
`EnvironmentFile=` in the unit below):

```bash
cp .env.example .env
set -a; . ./.env; set +a
```

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
| Discord bot | Setup and its own systemd unit: [DISCORD.md](DISCORD.md) |

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

## License

MIT — see [LICENSE](LICENSE). © 2026 Pat Wendorf.
