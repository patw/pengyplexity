"""Scaffold smoke tests: the Flask app factory boots and serves a blank base.

These run fully offline (no network, no bwrap, no writes outside tmp_path).
"""

from __future__ import annotations

from pengyplexity import __version__
from pengyplexity.app import create_app, get_state
from pengyplexity.config import load_config


def test_version():
    assert isinstance(__version__, str) and __version__


def test_index_renders_base_template(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"Pengyplexity" in r.data
    assert b"<!DOCTYPE html>" in r.data


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.get_json() == {"status": "ok", "app": "pengyplexity"}


def test_create_app_exposes_state(app, cfg):
    state = get_state(app)
    assert state.config is cfg
    # Store path is confined to the temp data dir, never $HOME.
    assert state.config.store_file.is_relative_to(cfg.data_dir)
    assert state.config.store_file.name == "pengyplexity.bson"
    # Data dir was created on disk (inside tmp_path).
    assert state.config.data_dir.is_dir()


def test_create_app_is_reentrant(cfg, tmp_data):
    # Two apps from the same config should both work independently.
    a1 = create_app(cfg=cfg)
    a2 = create_app(cfg=cfg)
    with a1.test_client() as c1, a2.test_client() as c2:
        assert c1.get("/healthz").status_code == 200
        assert c2.get("/").status_code == 200


def test_load_config_reads_env(tmp_path, monkeypatch):
    monkeypatch.setenv("PENGYPLEXITY_DATA_DIR", str(tmp_path / "envdata"))
    monkeypatch.setenv("PENGYPLEXITY_STORE", str(tmp_path / "envdata" / "custom.bson"))
    monkeypatch.setenv("PENGYPLEXITY_MODEL_BASE", "http://example.invalid/v1")
    monkeypatch.setenv("PENGYPLEXITY_MAX_AGENT_ITERATIONS", "7")
    c = load_config()
    assert c.data_dir == tmp_path / "envdata"
    assert c.store_file == tmp_path / "envdata" / "custom.bson"
    assert c.model_base == "http://example.invalid/v1"
    assert c.max_agent_iterations == 7
