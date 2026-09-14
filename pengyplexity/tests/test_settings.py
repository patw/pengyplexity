"""Tests for :mod:`pengyplexity.core.settings`.

Pure logic over a real :class:`Store` (temp dir) — no Flask, no network.
"""

from __future__ import annotations

import pytest

from pengyplexity.config import Config
from pengyplexity.core.settings import (
    defaults_from_config,
    effective_settings,
    raw_overrides,
    save_settings_from_form,
)
from pengyplexity.core.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "store")
    yield s
    s.close()


@pytest.fixture
def cfg(tmp_path):
    return Config(data_dir=tmp_path, store_path=tmp_path / "s.bson", model_base="http://cfg.example/v1")


class TestDefaultsFromConfig:
    def test_covers_every_field(self, cfg):
        defaults = defaults_from_config(cfg)
        assert defaults["model_base"] == "http://cfg.example/v1"
        assert defaults["system_message"] == ""
        assert defaults["exec_mem_mb"] == cfg.exec_mem_bytes // (1024 * 1024)


class TestEffectiveSettings:
    def test_no_overrides_returns_config_defaults(self, store, cfg):
        eff = effective_settings(store, cfg)
        assert eff["model_base"] == cfg.model_base
        assert eff["max_agent_iterations"] == cfg.max_agent_iterations

    def test_override_takes_precedence(self, store, cfg):
        store.set_setting("model_base", "http://override.example/v1")
        eff = effective_settings(store, cfg)
        assert eff["model_base"] == "http://override.example/v1"

    def test_empty_string_override_falls_back(self, store, cfg):
        store.set_setting("model_base", "http://override.example/v1")
        store.set_setting("model_base", "")
        eff = effective_settings(store, cfg)
        assert eff["model_base"] == cfg.model_base

    def test_numeric_override(self, store, cfg):
        store.set_setting("max_agent_iterations", 3)
        eff = effective_settings(store, cfg)
        assert eff["max_agent_iterations"] == 3


class TestSaveSettingsFromForm:
    def test_saves_string_and_numeric_fields(self, store, cfg):
        form = {
            "system_message": "Custom prompt",
            "model_base": "http://x.example/v1",
            "model_key": "secret",
            "model_name": "gpt-x",
            "user_agent": "UA/1",
            "model_temperature": "0.7",
            "llm_timeout": "60",
            "max_agent_iterations": "5",
            "tool_output_max_chars": "1000",
            "download_max_mb": "10",
            "tool_network_timeout": "20",
            "exec_timeout": "15",
            "exec_mem_mb": "128",
            "exec_cpu_seconds": "15",
        }
        save_settings_from_form(store, form)
        eff = effective_settings(store, cfg)
        assert eff["system_message"] == "Custom prompt"
        assert eff["model_temperature"] == 0.7
        assert eff["max_agent_iterations"] == 5
        assert eff["exec_mem_mb"] == 128

    def test_blank_field_clears_override(self, store, cfg):
        store.set_setting("model_base", "http://override.example/v1")
        save_settings_from_form(store, {"model_base": ""})
        eff = effective_settings(store, cfg)
        assert eff["model_base"] == cfg.model_base

    def test_invalid_number_ignored_not_raised(self, store, cfg):
        save_settings_from_form(store, {"max_agent_iterations": "not-a-number"})
        # Should not raise; the field is simply left unset.
        eff = effective_settings(store, cfg)
        assert eff["max_agent_iterations"] == cfg.max_agent_iterations

    def test_missing_fields_treated_as_blank(self, store, cfg):
        save_settings_from_form(store, {})
        eff = effective_settings(store, cfg)
        assert eff == defaults_from_config(cfg)


class TestRawOverrides:
    def test_empty_when_nothing_set(self, store):
        assert raw_overrides(store) == {}

    def test_reflects_stored_values(self, store):
        store.set_setting("model_name", "custom-model")
        assert raw_overrides(store)["model_name"] == "custom-model"
