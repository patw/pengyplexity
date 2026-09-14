"""Admin-editable global settings, layered over env-derived :class:`Config`.

Modeled on Pengy's ``settings.json`` (``~/Personal/Pengy/pengy/core/config.py``):
a flat dict of overridable values, editable from a settings UI. Pengyplexity
is multi-user, so — unlike Pengy's single-user file — the override lives in
the store's existing (previously unused) ``settings`` collection
(:meth:`Store.get_settings`/:meth:`Store.set_setting`) as ONE global,
admin-only document, per the app's "one admin manages everyone" model.

An override is "unset" when it's missing or an empty string — in that case
:func:`effective_settings` falls back to the ``Config`` value that was
resolved from the environment at startup. This is why every field below is
stored/edited as a plain string/number rather than requiring a sentinel.

This module is pure (no Flask import) so it can be unit-tested and reused by
both ``admin.py`` (the settings form) and ``web.py`` (applying the effective
settings to the shared agent/runner/model/tool-context before each turn).
"""

from __future__ import annotations

import getpass
import platform
import socket
from datetime import date
from typing import Any, Dict

# Each entry: (key, cast). ``cast`` turns a form string into the stored type;
# an empty string always means "unset" (falls back to the Config default)
# regardless of cast, so it's applied by the caller before casting.
NUMERIC_FIELDS = {
    "model_temperature": float,
    "llm_timeout": float,
    "max_agent_iterations": int,
    "tool_output_max_chars": int,
    "download_max_mb": float,
    "tool_network_timeout": int,
    "exec_timeout": int,
    "exec_mem_mb": int,
    "exec_cpu_seconds": int,
    "memory_signal_floor": float,
    "memory_signal_confident": float,
}

STRING_FIELDS = ("system_message", "model_base", "model_key", "model_name", "user_agent")

ALL_FIELDS = tuple(STRING_FIELDS) + tuple(NUMERIC_FIELDS)

# Fields whose value is a secret and should be masked in any UI that lists
# current overrides (the edit form itself still needs the real value).
SENSITIVE_FIELDS = frozenset({"model_key"})


def defaults_from_config(config: Any) -> Dict[str, Any]:
    """The as-if-no-override values, derived from the env-loaded ``Config``."""
    return {
        "system_message": "",
        "model_base": config.model_base,
        "model_key": config.model_key,
        "model_name": config.model_name,
        "model_temperature": config.model_temperature,
        "llm_timeout": config.llm_timeout,
        "max_agent_iterations": config.max_agent_iterations,
        "tool_output_max_chars": config.tool_output_max_chars,
        "download_max_mb": config.download_max_mb,
        "tool_network_timeout": config.tool_network_timeout,
        "user_agent": config.user_agent,
        "exec_timeout": config.exec_timeout,
        "exec_mem_mb": config.exec_mem_bytes // (1024 * 1024),
        "exec_cpu_seconds": config.exec_cpu_seconds,
        "memory_signal_floor": config.memory_signal_floor,
        "memory_signal_confident": config.memory_signal_confident,
    }


def raw_overrides(store: Any) -> Dict[str, Any]:
    """The admin-set override values as stored (no fallback applied)."""
    if not hasattr(store, "get_settings"):
        return {}
    doc = store.get_settings() or {}
    return dict(doc.get("values", {}) or {})


def effective_settings(store: Any, config: Any) -> Dict[str, Any]:
    """Merge admin overrides over the ``Config`` defaults.

    A missing key, or an override explicitly set to ``""``, falls back to the
    ``Config`` value. This is what the agent/tool wiring reads before every
    turn (see ``web.py: _prime_agent_for_turn``).
    """
    out = defaults_from_config(config)
    overrides = raw_overrides(store)
    for key in out:
        val = overrides.get(key)
        if val not in (None, ""):
            out[key] = val
    return out


def render_system_message(template: str, username: str | None = None) -> str:
    """Fill ``{date}``/``{username}``/``{hostname}``/``{osinfo}`` placeholders
    in an admin-set system message template, matching Pengy's
    ``core/config.py: render_system_message``. Unknown ``{...}`` placeholders
    raise ``KeyError`` from ``str.format`` just as they do in Pengy, so a typo
    surfaces immediately rather than being silently left in the prompt.

    Unlike Pengy (single-user, so ``{username}`` is the OS login), Pengyplexity
    is multi-user: *username* should be the logged-in app user, not the host
    process's OS account — pass the caller's ``current_user["username"]``.
    Falls back to the OS user only when none is given (e.g. no request
    context), matching the previous behavior.
    """
    return template.format(
        date=date.today().strftime("%B %d, %Y"),
        username=username or getpass.getuser(),
        hostname=socket.gethostname(),
        osinfo=f"{platform.system()} {platform.release()}",
    )


def save_settings_from_form(store: Any, form: Any) -> None:
    """Apply a settings-form submission (a mapping-like of strings) to the
    store. Blank fields clear the override (reverting to the Config default);
    non-blank numeric fields are cast, with an invalid number silently
    ignored (left as whatever was previously stored) rather than raising —
    this is an admin convenience form, not an API.
    """
    for key in STRING_FIELDS:
        store.set_setting(key, form.get(key, "").strip())
    for key, cast in NUMERIC_FIELDS.items():
        raw = form.get(key, "").strip()
        if raw == "":
            store.set_setting(key, "")
            continue
        try:
            store.set_setting(key, cast(raw))
        except (TypeError, ValueError):
            continue
