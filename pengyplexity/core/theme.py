"""User-selectable theming, ported from Pengy's Qt theme system
(``~/Personal/Pengy/pengy/ui/theme.py``).

The actual palette (light/dark base + 8 accents, each with its own light/dark
surface tint) lives as CSS custom properties in
``static/css/theme.css`` — a direct transcription of Pengy's
``BASE_THEMES``/``ACCENTS``/``ACCENT_SURFACES``/``SEMANTIC`` dicts. This
module only holds the small bit of server-side logic: the valid mode/accent
names (for validating a saved preference) and how a mode resolves to the
``data-theme`` attribute the templates stamp on ``<html>``.

Unlike Pengy (a single local user, default mode "system"), Pengyplexity is
multi-user and explicitly defaults new users to **light + the orange
accent** — the CSS's ``:root`` is the light palette, so an unthemed page is
light even before a preference is known; the orange surface tint is applied
by the ``data-accent`` attribute the templates stamp alongside it. Note that
the *accent named* ``"default"`` is still blue (Pengy's own default accent)
— it stays selectable, it is simply no longer what a new user starts on.
"""

from __future__ import annotations

from typing import Optional

THEME_MODES = ("system", "light", "dark")
ACCENT_NAMES = ("default", "blue", "teal", "green", "orange", "red", "pink", "purple")

DEFAULT_THEME_MODE = "light"
DEFAULT_THEME_ACCENT = "orange"


def normalize_theme_mode(mode: Optional[str]) -> str:
    """Validate a stored/submitted mode, falling back to the app default."""
    return mode if mode in THEME_MODES else DEFAULT_THEME_MODE


def normalize_theme_accent(accent: Optional[str]) -> str:
    """Validate a stored/submitted accent, falling back to the app default."""
    return accent if accent in ACCENT_NAMES else DEFAULT_THEME_ACCENT


def html_theme_attr(mode: Optional[str]) -> Optional[str]:
    """The ``data-theme`` attribute value for a resolved mode.

    ``"system"`` means "no explicit attribute" — the page follows the
    browser's ``prefers-color-scheme`` instead (see theme.css). ``None``
    signals the template to omit the attribute entirely.
    """
    mode = normalize_theme_mode(mode)
    return mode if mode != "system" else None
