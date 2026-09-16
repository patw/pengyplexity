# Pengyplexity — HTTP API

A JSON API at `/api/v1` for programs: the [Discord bot](DISCORD.md), scripts,
anything that is not a browser. It gives a program the same agent the chat UI uses — same
tools, same sandbox, same admin settings, same threads and memories — acting as
one user.

> For the overview, see the [README](README.md). For how a turn runs and the
> safety model around it, see [SPEC.md](SPEC.md). To deploy it, see
> [INSTALLING.md](INSTALLING.md).

**Contents:** [Authentication](#authentication) ·
[Conventions](#conventions) · [Endpoints](#endpoints) ·
[Asking a question](#asking-a-question) · [Limits](#limits) ·
[Example](#example) · [Writing a client](#writing-a-client)

---

## Authentication

1. Log in, open **Account**, and create an API key. It is shown **once**; only
   a SHA-256 hash is stored (`core/apikeys.py`), with an 8-character prefix
   kept for telling keys apart.
2. Send it on every request: `Authorization: Bearer pgy_…`.

Rules:

- The key acts as its user: same threads, memories, workspace and admin
  settings. Every request re-reads the user, so **disabling or deleting the
  account shuts its keys off immediately**; deleting it also deletes the keys.
- **Keys cannot create keys.** Creation needs the web session, so a leaked key
  can be revoked (`DELETE /api/v1/keys/<id>`, or from the Account page) without
  having spread.
- The session cookie is **ignored** under `/api/`, so there is no CSRF surface.
- At most 20 keys per user.

## Conventions

- JSON request bodies, JSON responses. Timestamps are ISO-8601 UTC (`…Z`).
- Every error, including 404/405 for unknown routes, is
  `{"error": {"code": "<stable_code>", "message": "<human text>"}}`.
- Anything the caller does not own is a `404`, exactly like something that
  does not exist.
- A refused request (validation, rate limit, busy thread) is refused **before
  anything is written**, so it is always safe to retry as-is.
- Request bodies are capped at 1 MiB; a question at 100,000 characters.

## Endpoints

| Method & path | Purpose |
| --- | --- |
| `GET /api/v1/me` | The key's user, the key itself, and the limits in force |
| `GET /api/v1/keys` | Your keys (`current: true` marks the one calling) |
| `DELETE /api/v1/keys/<id>` | Revoke a key — including the calling one |
| `GET /api/v1/threads?limit=&offset=` | Your threads, most recently updated first (`limit` 1–200, default 50) |
| `POST /api/v1/threads` | Create a thread; body `{"title"?}`. Never reuses a blank one → `201` |
| `GET /api/v1/threads/<id>` | A thread with its messages, each with `sources` and `artifacts` |
| `PATCH /api/v1/threads/<id>` | Rename: `{"title"}` |
| `DELETE /api/v1/threads/<id>` | Delete (stops a running turn first) → `204` |
| `POST /api/v1/threads/<id>/messages` | **Ask a question**: `{"content", "stream"?}` |
| `POST /api/v1/threads/<id>/stop` | Stop the running turn → `{"stopped": bool}` |
| `GET /api/v1/threads/<id>/artifacts/<artifact_id>[?download=1]` | An artifact's file (confined to the thread's workspace, same guard as the web route) |
| `GET /api/v1/memories?q=&limit=&offset=&all=1` | List your memories, or search them with `q` (`all=1` drops the relevance floor) |
| `POST /api/v1/memories` | Create: `{"title", "summary", "body"?, "tags"?, "status"?}` → `201` |
| `GET/PATCH/DELETE /api/v1/memories/<id>` | Read / edit (any subset of those fields) / delete |

Thread objects carry `busy: true` while a turn is running in them.

## Asking a question

`POST /api/v1/threads/<id>/messages` with `{"content": "…"}`.

**Non-streaming** (default) waits for the whole turn:

```json
{
  "thread":  {"id": "…", "title": "…", "message_count": 2, "busy": false, "…": "…"},
  "message": {"index": 1, "role": "assistant", "content": "…",
              "sources": [{"title": "…", "url": "…"}],
              "artifacts": [{"id": "…", "filename": "chart.png", "kind": "chart",
                             "mime": "image/png", "url": "/api/v1/threads/…/artifacts/…",
                             "download_url": "…?download=1"}]},
  "stopped": false
}
```

A turn can run for minutes (web searches, sandboxed scripts), so an
interactive client should stream instead. If the agent fails the response is
`502` with `error.code = "agent_error"` plus the same fields (`message` holds
any partial answer that was saved, or `null`).

**Streaming** — `"stream": true` or `Accept: text/event-stream` — returns SSE
with the same events the browser gets, then one final `message` event:

| Event | Data |
| --- | --- |
| `title` | `{"title"}` — the thread was just named from this question |
| `activity` | `{"type", "label"}` — e.g. "Searching the web…" |
| `token` | `{"content"}` — the next piece of the answer |
| `done` | `{"answer", "sources"}` — the agent finished |
| `artifact` | `{"artifact_id", "filename", "kind", "mime", "url", "download_url"}` |
| `error` | `{"message"}` |
| `message` | The non-streaming response body above: the answer **as saved** (including a `_[Stopped]_` note), plus `stopped` |

Hanging up mid-stream is safe: whatever was written so far is saved with an
`_[Interrupted]_` note and the thread stops being busy.

## Limits

| Status | `error.code` | When |
| --- | --- | --- |
| `409` | `turn_in_progress` | The thread is already answering. Wait, or `POST …/stop`. (The browser instead replaces the running turn; an API client may still be reading it.) |
| `429` | `rate_limited` | More than `PENGYPLEXITY_API_RATE_LIMIT` questions this minute. Honour `Retry-After`. |
| `429` | `too_many_concurrent_turns` | `PENGYPLEXITY_API_MAX_CONCURRENT_TURNS` threads already answering for this user. |
| `503` | `agent_unavailable` | The server has no agent configured. |

Both limits are per user and in-process, which matches how Pengyplexity is
deployed: one host process, never replicated.

## Example

```bash
KEY=pgy_…   # from Account → API keys
BASE=https://pengyplexity.example.com/api/v1

TID=$(curl -s -X POST "$BASE/threads" -H "Authorization: Bearer $KEY" | jq -r .id)
curl -N "$BASE/threads/$TID/messages" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"content": "What changed in Python 3.13?", "stream": true}'
```

Behind nginx, `/api/` needs the same `proxy_buffering off` as the chat UI
(the `location /` block in [INSTALLING.md](INSTALLING.md) already covers it).

---

## Writing a client

A minimal streaming client in Python (`requests`), the shape a bot needs:

```python
import json
import requests

BASE = "https://pengyplexity.example.com/api/v1"
HEADERS = {"Authorization": "Bearer pgy_…"}


def ask(thread_id: str, question: str):
    """Yield (event, data) pairs for one question."""
    with requests.post(
        f"{BASE}/threads/{thread_id}/messages",
        headers=HEADERS,
        json={"content": question, "stream": True},
        stream=True,
        timeout=(10, None),  # a turn can run for minutes
    ) as resp:
        if resp.status_code != 200:
            error = resp.json()["error"]
            raise RuntimeError(f"{resp.status_code} {error['code']}: {error['message']}")
        event = None
        for line in resp.iter_lines(decode_unicode=True):
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                yield event, json.loads(line[6:])


thread = requests.post(f"{BASE}/threads", headers=HEADERS, timeout=10).json()
for event, data in ask(thread["id"], "What changed in Python 3.13?"):
    if event == "token":
        print(data["content"], end="", flush=True)
    elif event == "message":
        final = data["message"]  # the answer as saved, with sources + artifacts
```

Things worth handling:

- **Keep one thread per conversation** (e.g. per Discord channel or reply
  chain) and reuse its id; the agent sees the thread's history on every turn.
- **`409 turn_in_progress`** — the thread is still answering. Queue the
  question, or `POST …/stop` first if the user wants to interrupt.
- **`429`** — back off. `rate_limited` carries `Retry-After`;
  `too_many_concurrent_turns` clears when one of the user's other threads
  finishes.
- **Artifacts** — `url` values are paths relative to the server; fetch them
  with the same `Authorization` header and re-upload the bytes (a chat
  platform cannot fetch them itself).
- **Treat the final `message` event as the source of truth** — it is what was
  saved, including any `_[Stopped]_` / `_[Interrupted]_` / `_[Error]_` note.
- **One key = one Pengyplexity user.** A bot holding a single key puts every
  one of its users' threads and memories into that one account.
