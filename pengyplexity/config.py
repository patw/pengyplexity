"""Configuration for Pengyplexity.

All configuration comes from environment variables with sane defaults, so the
app runs out-of-the-box for local development and is fully configurable in a
deployment. Secrets (the LLM key, Flask's secret key) are read from the environment
only, and are never committed — see ``.env.example``.

The module deliberately imports only the standard library so it is safe to
import during the offline test suite (no Flask / network dependencies here).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Default local model endpoint (an OpenAI-compatible server on the LAN).
# Override with PENGYPLEXITY_MODEL_BASE / _KEY / _NAME.
DEFAULT_MODEL_BASE = "http://10.0.23.2:8086/v1"
DEFAULT_MODEL_NAME = "gpt-4o-mini"


def _env_str(name: str, default: str) -> str:
    val = os.environ.get(name)
    return val if val is not None and val != "" else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _home() -> Path:
    return Path(os.environ.get("HOME", str(Path.home())))


@dataclass
class Config:
    """Resolved, plain-data configuration.

    Everything is a concrete value (no callables) so it can be trivially
    inspected, copied, and faked in tests.
    """

    # --- Storage ----------------------------------------------------------
    # Root for all Pengyplexity state (store file + per-thread workspaces).
    data_dir: Path = field(default_factory=lambda: _home() / ".pengyplexity")
    # Override the moofile store path directly (else <data_dir>/pengyplexity.bson).
    store_path: Path | None = None
    # Whether the memory store's semantic (vector) search leg is enabled. When
    # False, memory search is lexical (BM25) only — no embedding model load.
    memory_semantic: bool = True
    # Memory search relevance thresholds (see core/memory.py's calibration
    # note). Corpus- and model-specific, so tunable per deployment: a result
    # below the floor is not returned at all, one at or above `confident` is
    # reported as a strong match rather than a lead.
    memory_signal_floor: float = 0.33
    memory_signal_confident: float = 0.50

    # --- Model client -----------------------------------------------------
    model_base: str = DEFAULT_MODEL_BASE
    model_key: str = ""
    model_name: str = DEFAULT_MODEL_NAME
    model_temperature: float = 0.3
    # HTTP timeout (seconds) for each LLM API request.
    llm_timeout: float = 300.0

    # --- Agent behaviour --------------------------------------------------
    # Hard cap on tool-call iterations per turn (runaway-loop guard).
    max_agent_iterations: int = 12
    # Bounded number of web queries the deep-research loop will perform.
    research_query_budget: int = 6
    # Tool output longer than this is snipped (head+tail); 0 = no limit.
    tool_output_max_chars: int = 250_000
    # Default max download size (MB) for download_file; 0 = unlimited.
    download_max_mb: float = 100.0
    # Timeout (seconds) for web_search / fetch_url / download_file.
    tool_network_timeout: int = 15
    # User-Agent header sent by web_search/fetch_url/download_file.
    user_agent: str = "Mozilla/5.0 (Pengyplexity)"

    # --- Flask / app ------------------------------------------------------
    secret_key: str = "insecure-dev-key-change-me"
    debug: bool = False

    # --- Sandboxed execution ---------------------------------------------
    # Wall-clock timeout (seconds) for a single run_python / run_bash.
    exec_timeout: int = 30
    # Memory (bytes) and CPU-time (seconds) caps passed to bwrap rlimits.
    exec_mem_bytes: int = 512 * 1024 * 1024  # 512 MiB
    exec_cpu_seconds: int = 30
    # Host directory holding the sandbox's Python environment (matplotlib /
    # numpy / pandas), bind-mounted read-only into the sandbox. Without it the
    # sandbox has only the host's bare system Python and every chart script
    # fails on `import matplotlib` — see sandbox/pythonenv.py.
    sandbox_venv: Path | None = None
    # Build that environment on first use when it is missing. Set False to
    # require an explicit `pengyplexity-admin build-sandbox`.
    sandbox_autobuild: bool = True

    @property
    def store_file(self) -> Path:
        if self.store_path is not None:
            return Path(self.store_path)
        return self.data_dir / "pengyplexity.bson"

    @property
    def memory_store_file(self) -> Path:
        return self.data_dir / "memories.bson"

    @property
    def sandbox_venv_path(self) -> Path:
        """Where the sandbox Python environment lives (default under data_dir)."""
        if self.sandbox_venv is not None:
            return Path(self.sandbox_venv)
        return self.data_dir / "sandbox-venv"

    def workspace_root(self) -> Path:
        """Per-user workspaces live under ``<data_dir>/workspaces/<user>/``."""
        return self.data_dir / "workspaces"

    def to_flask_config(self) -> dict:
        """Flask's ``app.config`` mapping (keys upper-cased)."""
        return {
            "SECRET_KEY": self.secret_key,
            "DEBUG": self.debug,
            "PENGYPLEXITY_DATA_DIR": str(self.data_dir),
            "PENGYPLEXITY_STORE": str(self.store_file),
            "PENGYPLEXITY_MODEL_BASE": self.model_base,
            "PENGYPLEXITY_MODEL_KEY": self.model_key,
            "PENGYPLEXITY_MODEL_NAME": self.model_name,
            "PENGYPLEXITY_MODEL_TEMPERATURE": self.model_temperature,
            "PENGYPLEXITY_LLM_TIMEOUT": self.llm_timeout,
            "PENGYPLEXITY_MAX_AGENT_ITERATIONS": self.max_agent_iterations,
            "PENGYPLEXITY_RESEARCH_QUERY_BUDGET": self.research_query_budget,
            "PENGYPLEXITY_TOOL_OUTPUT_MAX_CHARS": self.tool_output_max_chars,
            "PENGYPLEXITY_DOWNLOAD_MAX_MB": self.download_max_mb,
            "PENGYPLEXITY_TOOL_NETWORK_TIMEOUT": self.tool_network_timeout,
            "PENGYPLEXITY_USER_AGENT": self.user_agent,
            "PENGYPLEXITY_EXEC_TIMEOUT": self.exec_timeout,
            "PENGYPLEXITY_EXEC_MEM_BYTES": self.exec_mem_bytes,
            "PENGYPLEXITY_EXEC_CPU_SECONDS": self.exec_cpu_seconds,
            "PENGYPLEXITY_SANDBOX_VENV": str(self.sandbox_venv_path),
            "PENGYPLEXITY_SANDBOX_AUTOBUILD": self.sandbox_autobuild,
            "PENGYPLEXITY_MEMORY_SEMANTIC": self.memory_semantic,
            "PENGYPLEXITY_MEMORY_SIGNAL_FLOOR": self.memory_signal_floor,
            "PENGYPLEXITY_MEMORY_SIGNAL_CONFIDENT": self.memory_signal_confident,
        }


def load_config() -> Config:
    """Build a :class:`Config` from the environment with sane defaults."""
    data_dir = Path(_env_str("PENGYPLEXITY_DATA_DIR", str(_home() / ".pengyplexity")))
    store_raw = os.environ.get("PENGYPLEXITY_STORE")
    store_path = Path(store_raw) if store_raw else None
    return Config(
        data_dir=data_dir,
        store_path=store_path,
        model_base=_env_str("PENGYPLEXITY_MODEL_BASE", DEFAULT_MODEL_BASE),
        model_key=_env_str("PENGYPLEXITY_MODEL_KEY", ""),
        model_name=_env_str("PENGYPLEXITY_MODEL_NAME", DEFAULT_MODEL_NAME),
        model_temperature=_env_float("PENGYPLEXITY_MODEL_TEMPERATURE", 0.3),
        llm_timeout=_env_float("PENGYPLEXITY_LLM_TIMEOUT", 300.0),
        max_agent_iterations=_env_int("PENGYPLEXITY_MAX_AGENT_ITERATIONS", 12),
        research_query_budget=_env_int("PENGYPLEXITY_RESEARCH_QUERY_BUDGET", 6),
        tool_output_max_chars=_env_int("PENGYPLEXITY_TOOL_OUTPUT_MAX_CHARS", 250_000),
        download_max_mb=_env_float("PENGYPLEXITY_DOWNLOAD_MAX_MB", 100.0),
        tool_network_timeout=_env_int("PENGYPLEXITY_TOOL_NETWORK_TIMEOUT", 15),
        user_agent=_env_str("PENGYPLEXITY_USER_AGENT", "Mozilla/5.0 (Pengyplexity)"),
        secret_key=_env_str("PENGYPLEXITY_SECRET_KEY", "insecure-dev-key-change-me"),
        debug=_env_str("PENGYPLEXITY_DEBUG", "0") in {"1", "true", "True"},
        exec_timeout=_env_int("PENGYPLEXITY_EXEC_TIMEOUT", 30),
        exec_mem_bytes=_env_int("PENGYPLEXITY_EXEC_MEM_BYTES", 512 * 1024 * 1024),
        exec_cpu_seconds=_env_int("PENGYPLEXITY_EXEC_CPU_SECONDS", 30),
        sandbox_venv=(
            Path(os.environ["PENGYPLEXITY_SANDBOX_VENV"])
            if os.environ.get("PENGYPLEXITY_SANDBOX_VENV")
            else None
        ),
        sandbox_autobuild=_env_str("PENGYPLEXITY_SANDBOX_AUTOBUILD", "1")
        not in {"0", "false", "False"},
        memory_semantic=_env_str("PENGYPLEXITY_MEMORY_SEMANTIC", "1") not in {"0", "false", "False"},
        memory_signal_floor=_env_float("PENGYPLEXITY_MEMORY_SIGNAL_FLOOR", 0.33),
        memory_signal_confident=_env_float("PENGYPLEXITY_MEMORY_SIGNAL_CONFIDENT", 0.50),
    )
