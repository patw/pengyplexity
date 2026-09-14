"""Pengyplexity — a Perplexity-style, sandboxed AI Q&A web app.

Pengyplexity reuses the Pengy codebase (``pengy.core.llm_client`` for the model
client and ``pengy.core.tools`` for the tool engine) but wraps it in a
multi-user, sandboxed web shell. The defining constraint is the *safety
boundary*: only a curated set of safe tools is exposed, every file path and
every code-execution call is confined to a per-thread workspace, and the app
can never self-modify its own skills or the host.

See ``spec.md`` for the full design and ``AGENTS.md`` for build notes.
"""

__version__ = "0.1.0"

__all__ = ["__version__", "create_app"]


def create_app(config_overrides=None, cfg=None):
    """Application factory (thin re-export of :func:`pengyplexity.app.create_app`).

    Imported here so callers can ``from pengyplexity import create_app``.
    Importing the package does NOT pull in heavy dependencies (Flask, moofile,
    ...) — those are imported lazily inside :mod:`pengyplexity.app`.
    """
    from .app import create_app as _create_app

    return _create_app(config_overrides=config_overrides, cfg=cfg)
