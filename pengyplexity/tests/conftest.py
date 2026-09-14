"""Shared fixtures for the Pengyplexity test suite.

The offline test rule: every fixture that would touch the network or the host
filesystem is redirected into ``tmp_path`` (per-test temp dirs) and the
heavy services (model, search, executor, uploaders) are faked here as they are
introduced by later subtasks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pengyplexity.app import create_app
from pengyplexity.config import Config


@pytest.fixture
def tmp_data(tmp_path) -> Path:
    """A temp data dir standing in for ``~/.pengyplexity`` (never writes to $HOME)."""
    d = tmp_path / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def cfg(tmp_data: Path) -> Config:
    """A Config fully pointed at temp storage + an offline model endpoint."""
    return Config(
        data_dir=tmp_data,
        store_path=tmp_data / "pengyplexity.bson",
        model_base="http://127.0.0.1:0/v1",  # unreachable: proves no network in tests
        model_key="test-key",
        secret_key="test-secret",
    )


@pytest.fixture
def app(cfg: Config):
    """A Pengyplexity Flask app wired to temp storage."""
    return create_app(cfg=cfg)


@pytest.fixture
def client(app):
    with app.test_client() as test_client:
        yield test_client
