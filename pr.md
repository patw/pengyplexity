# PR: Pengyplexity — initial public release (v0.1.0)

> Draft pull-request body. Branch: `main` → `main` (initial import).
> Delete this file once the PR is open, or keep it as the PR record.

## What

Publishes **Pengyplexity**: a Perplexity-style, sandboxed AI Q&A web app
(Python/Flask) where web search is a first-class tool and every piece of code
the model writes runs inside a bubblewrap sandbox it cannot escape.

This PR is the **initial import** — the application itself, its offline test
suite, and the repo-hygiene files needed to make it a proper public repo
(license, changelog, `.gitignore`, `.env.example`, install/operate docs).

## Why

The app is finished enough to be worth showing: the safety model is the whole
point of it, and it is now enforced by structure rather than by prompt wording —
a curated tool allowlist, path confinement, and namespace-isolated execution.
It has been running as a deployed service, and the suite is green offline, so
another person can clone it, point it at any OpenAI-compatible endpoint, and
get a working instance without network access to anything but their model.

## Changes

**Application (new):**

- `pengyplexity/app.py` — Flask factory + `AppState` (injectable services).
- `pengyplexity/web.py`, `admin.py`, `cli.py` — chat/SSE, auth + admin CRUD,
  and the `create-admin` / `build-sandbox` CLI.
- `pengyplexity/core/` — model client, agent loop, streaming, search,
  deep research, artifacts, images, sharing, auth, store, memory, settings.
- `pengyplexity/sandbox/` — `toolpolicy.py` (SAFE_TOOLS allowlist),
  `confine.py` (path confinement), `executors.py` (bwrap runner),
  `pythonenv.py` (the sandbox Python env).
- `pengyplexity/templates/`, `static/` — Bootstrap-flavoured, dependency-free UI.
- `pengyplexity/tests/` — the offline suite.

**Repo hygiene (new in this pass):**

- `LICENSE` — MIT, © 2026 Pat Wendorf.
- `CHANGELOG.md` — Keep a Changelog format, seeded with the 0.1.0 entry.
- `.gitignore` — Python/venv/secrets/runtime-state/caches, with `.env.example`
  explicitly *not* ignored.
- `.env.example` — documented template for all `PENGYPLEXITY_*` settings.
- `README.md` — extended with install, systemd service, nginx+HTTPS deployment,
  operating, and troubleshooting sections.
- `pyproject.toml` — added authors, license, keywords and classifiers.

## How to test

```bash
# Offline test suite — no network, no live bwrap needed
uv sync
uv run pytest                 # expect: 574 passed, 1 skipped

# End-to-end, minimal
uv run python -m pengyplexity.cli create-admin admin --password 'changeme'
PENGYPLEXITY_MODEL_BASE=http://127.0.0.1:8086/v1 uv run flask --app pengyplexity.app:create_app run
# then browse http://127.0.0.1:5000/ , log in, ask a question
```

Worth exercising by hand: a question that triggers `web_search` (sources appear),
a chart request (proves the sandbox venv + `make_chart` path), and a
deliberately hostile prompt trying to read a host file such as `/etc/passwd`
(should be refused/confined — this is the security claim, so it deserves a
manual check, not just unit tests).

## Review notes / decisions

- **No Dockerfile — deliberately.** The README lists Docker as a hard non-goal,
  and that is not a stylistic preference. I verified the constraint on the
  deployment host:

  ```
  $ docker run --rm alpine sh -c "apk add --no-cache bubblewrap; \
      bwrap --ro-bind / / --unshare-net --cap-drop ALL echo INSIDE_OK"
  bwrap: Creating new namespace failed: Operation not permitted
  ```

  A default container cannot create the namespaces `bwrap` needs, so a
  containerised build would either fail at runtime or — worse — ship the safety
  model switched off while still advertising it. Documented in the README
  non-goals and troubleshooting instead, with the actual error string so the
  next person finds it by grepping.
- **MIT with the email in the copyright line** — `Copyright (c) 2026 Pat
  Wendorf <dungeons@gmail.com>`, per request.
- **`uv.lock` is committed on purpose.** This is an application, not a library:
  a reproducible resolved dependency set is worth more than a floating range.
- **The README is under test** (`tests/test_readme.py` asserts it documents the
  safety model, the config keys, bootstrap, `flask --app`, and the offline
  suite). Any doc rewrite has to keep those strings — they are treated as part
  of the deliverable, not as incidental prose.
- **Version stays `0.1.0`** in `pyproject.toml` and the changelog's first entry.
  Classified `Development Status :: 4 - Beta` (deployed and green, but no
  external users yet) — say the word if you'd rather this be `3 - Alpha`.

## Known limitations / follow-ups

- No CI workflow yet (`.github/workflows/`) — the suite is offline, so it is a
  cheap add and would be the natural next PR.
- No `[project.urls]` in `pyproject.toml` — intentionally left out rather than
  guessing a repo URL before the remote exists.
- The README's opening paragraph referenced the internal "ralph workspace", and
  framed the app as *reusing* the Pengy codebase; for a public reader that reads
  as a dependency on a private repo. Reworded to describe provenance and design
  lineage instead. Flagging it because it is your framing, not mine — easy to
  revert.
- The app runs behind nginx with `proxy_buffering off` so SSE streams properly;
  that detail is documented in the README deployment section because it is the
  single most likely thing to be got wrong.

## Checklist

- [x] Offline suite green (`574 passed, 1 skipped`)
- [x] No secrets committed (`.env` ignored; only `.env.example` is tracked)
- [x] License present and declared in `pyproject.toml`
- [x] Changelog seeded
- [x] Install + operate instructions documented
- [ ] Remote created and pushed (left to the maintainer)
