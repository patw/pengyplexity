"""Tests for :mod:`pengyplexity.core.theme` and the theme picker web routes.

Ported from Pengy's Qt theme system: light/dark/system mode + 8 accents,
user-selectable, defaulting to light + the orange accent (see core/theme.py's
module docstring for why Pengyplexity's default differs from Pengy's
"system"/blue default).
"""

from __future__ import annotations

import re

import pytest

from pengyplexity.core.theme import (
    ACCENT_NAMES,
    DEFAULT_THEME_ACCENT,
    DEFAULT_THEME_MODE,
    THEME_MODES,
    html_theme_attr,
    normalize_theme_accent,
    normalize_theme_mode,
)


class TestNormalization:
    def test_default_mode_is_light(self):
        assert DEFAULT_THEME_MODE == "light"

    def test_valid_modes_pass_through(self):
        for mode in THEME_MODES:
            assert normalize_theme_mode(mode) == mode

    def test_invalid_or_missing_mode_falls_back_to_light(self):
        assert normalize_theme_mode(None) == "light"
        assert normalize_theme_mode("") == "light"
        assert normalize_theme_mode("bogus") == "light"

    def test_valid_accents_pass_through(self):
        for accent in ACCENT_NAMES:
            assert normalize_theme_accent(accent) == accent

    def test_default_accent_is_orange(self):
        assert DEFAULT_THEME_ACCENT == "orange"

    def test_invalid_or_missing_accent_falls_back_to_default(self):
        assert normalize_theme_accent(None) == DEFAULT_THEME_ACCENT
        assert normalize_theme_accent("nonexistent") == DEFAULT_THEME_ACCENT

    def test_the_named_default_accent_is_still_selectable(self):
        # "default" (blue) remains a valid explicit choice — it is just no
        # longer what a new user starts on.
        assert normalize_theme_accent("default") == "default"


class TestHtmlThemeAttr:
    def test_light_and_dark_pass_through(self):
        assert html_theme_attr("light") == "light"
        assert html_theme_attr("dark") == "dark"

    def test_system_omits_the_attribute(self):
        assert html_theme_attr("system") is None

    def test_invalid_falls_back_to_light_default(self):
        assert html_theme_attr("nonsense") == "light"


# ---------------------------------------------------------------------------
# Web: the account page's theme picker + the base.html context processor
# ---------------------------------------------------------------------------


def _html_tag(data: bytes) -> str:
    m = re.search(rb"<html[^>]*>", data)
    assert m is not None
    return m.group(0).decode()


@pytest.fixture
def app_with_agent(cfg):
    """A real app (no fake agent needed — these tests never call /ask)."""
    from pengyplexity.app import create_app, get_state
    from pengyplexity.core.auth import create_admin

    app = create_app(cfg=cfg)
    create_admin(get_state(app).store, "admin", "admin-pass")
    return app


@pytest.fixture
def logged_in_client(app_with_agent):
    with app_with_agent.test_client() as c:
        c.post("/login", data={"username": "admin", "password": "admin-pass"})
        yield c


class TestThemeWebRoutes:
    def test_requires_login(self, client):
        resp = client.post("/account/theme", data={"theme_mode": "dark"}, follow_redirects=False)
        assert resp.status_code == 302
        assert "/login" in resp.headers["Location"]

    def test_unauthenticated_page_defaults_to_light_orange(self, client):
        resp = client.get("/login")
        tag = _html_tag(resp.data)
        assert 'data-theme="light"' in tag
        assert 'data-accent="orange"' in tag

    def test_new_user_defaults_to_light_orange_accent(self, logged_in_client):
        resp = logged_in_client.get("/account/password")
        tag = _html_tag(resp.data)
        assert 'data-theme="light"' in tag
        assert 'data-accent="orange"' in tag

    def test_save_persists_and_applies_on_next_page(self, logged_in_client):
        resp = logged_in_client.post(
            "/account/theme",
            data={"theme_mode": "dark", "theme_accent": "teal"},
            follow_redirects=True,
        )
        assert resp.status_code == 200
        assert b"Appearance updated" in resp.data
        tag = _html_tag(resp.data)
        assert 'data-theme="dark"' in tag
        assert 'data-accent="teal"' in tag

    def test_system_mode_omits_data_theme_attribute(self, logged_in_client):
        logged_in_client.post("/account/theme", data={"theme_mode": "system", "theme_accent": "purple"})
        resp = logged_in_client.get("/account/password")
        tag = _html_tag(resp.data)
        assert "data-theme=" not in tag
        assert 'data-accent="purple"' in tag

    def test_invalid_values_fall_back_to_defaults(self, logged_in_client):
        logged_in_client.post("/account/theme", data={"theme_mode": "bogus", "theme_accent": "bogus"})
        resp = logged_in_client.get("/account/password")
        tag = _html_tag(resp.data)
        assert 'data-theme="light"' in tag
        assert 'data-accent="orange"' in tag

    def test_preference_is_per_user(self, app_with_agent):
        from pengyplexity.app import get_state
        from pengyplexity.core.auth import create_admin

        state = get_state(app_with_agent)
        create_admin(state.store, "bob", "bob-pass")

        with app_with_agent.test_client() as c:
            c.post("/login", data={"username": "admin", "password": "admin-pass"})
            c.post("/account/theme", data={"theme_mode": "dark", "theme_accent": "red"})

        with app_with_agent.test_client() as c:
            c.post("/login", data={"username": "bob", "password": "bob-pass"})
            resp = c.get("/account/password")
            tag = _html_tag(resp.data)
            assert 'data-theme="light"' in tag
            assert 'data-accent="orange"' in tag

    def test_account_page_shows_theme_form(self, logged_in_client):
        resp = logged_in_client.get("/account/password")
        assert resp.status_code == 200
        assert b"Appearance" in resp.data
        assert b'name="theme_mode"' in resp.data
        assert b'name="theme_accent"' in resp.data


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
