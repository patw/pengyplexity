"""Flask application factory for Pengyplexity.

This module owns :func:`create_app`, the single entry point used by ``flask --app``,
the ``python -m pengyplexity.cli`` bootstrap, and the offline test suite. It is kept
deliberately thin at this stage: it builds a Flask app, loads configuration, exposes a
blank base template at ``/``, and a ``/healthz`` probe. Subsequent subtasks layer on
auth, chat, threads, admin, sharing, and artifacts — each registers its own blueprint
inside :func:`create_app` so tests can construct a fully-faked app.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from flask import Flask, render_template  # noqa: F401  (re-exported for convenience)

from .config import Config, load_config
from .core.cancel import CancelRegistry


@dataclass
class AppState:
    """Container for resolved services, attached to the app under
    ``app.extensions["pengyplexity"]``.

    Populated incrementally as subtasks land. Keeping services on one object (rather
    than module globals) makes the app trivially fakes-injectable for the offline
    test suite: a test can swap ``state.store`` / ``state.agent`` / etc.
    """

    config: Config
    # Filled in by later subtasks:
    store: object | None = None
    auth: object | None = None
    # An explicitly injected agent, used for *every* request when set. In
    # production this stays None and :attr:`agent_factory` is used instead;
    # tests assign a fake here and it wins.
    agent: object | None = None
    # Builds a fresh agent per request. See :meth:`new_agent`.
    agent_factory: object | None = None
    model: object | None = None
    runner: object | None = None
    search: object | None = None
    memory: object | None = None
    sandbox_env: object | None = None
    # Tracks the in-flight turn per (owner, thread) so the /stop route — a
    # separate request on a separate thread — can interrupt it. See
    # core/cancel.py.
    cancels: CancelRegistry = field(default_factory=CancelRegistry)
    # Personal API keys for /api/v1 (core/apikeys.py) and the per-user turn
    # rate limiter that guards it (core/ratelimit.py).
    api_keys: object | None = None
    api_rate_limiter: object | None = None

    def __post_init__(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @property
    def data_dir(self) -> Path:
        return self.config.data_dir

    def new_agent(self):
        """Return the agent to use for one request.

        Production builds a **fresh** agent per request rather than sharing
        one. The agent and its tool executor carry per-turn state — the
        thread's workspace path, and the tool context's thread id, owner,
        artifact list and pending images — which the request handler used to
        repoint on a single shared instance. The server is threaded, so two
        people asking at once would overwrite each other's: the second
        request's workspace would capture the first user's remaining tool
        calls, and their charts and memories would be filed under the other
        person's thread and username.

        The expensive, genuinely shared pieces (model client, sandbox runner,
        search service, stores) are built once and reused; only the thin
        per-turn wrapper is new.
        """
        if self.agent is not None:
            return self.agent
        if self.agent_factory is not None:
            return self.agent_factory()
        return None


def create_app(config_overrides: dict | None = None, cfg: Config | None = None) -> Flask:
    """Build and return the Pengyplexity Flask application.

    Parameters
    ----------
    config_overrides:
        Optional mapping of Flask ``app.config`` keys applied *after* the env-derived
        defaults. Tests use this to point the app at a temp data dir / temp store.
    cfg:
        Optional pre-built :class:`~pengyplexity.config.Config`. When omitted, config
        is loaded from the environment. Useful for the CLI and for tests that want a
        fully explicit configuration.
    """
    config = cfg if cfg is not None else load_config()
    if config_overrides:
        # Apply overrides to the Config dataclass so both the app and services
        # observe the same values.
        for key, value in config_overrides.items():
            # Map uppercase Flask-style keys to lowercase dataclass fields.
            field_name = key.lower()
            if hasattr(config, field_name):
                setattr(config, field_name, value)

    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )
    app.config.from_mapping(config.to_flask_config())

    state = AppState(config=config)
    app.extensions["pengyplexity"] = state

    # Wire up the store (real moofile-backed) and auth service.
    # Tests can swap state.store / state.auth / state.agent for fakes.
    from .core.store import Store
    from .core.apikeys import ApiKeyService
    from .core.auth import AuthService
    from .core.ratelimit import RateLimiter

    state.store = Store(config.store_file.parent)
    state.auth = AuthService(store=state.store)
    state.api_keys = ApiKeyService(store=state.store)
    state.api_rate_limiter = RateLimiter(config.api_rate_limit)

    # Build the production tool loop (model + search + sandbox) so the chat
    # endpoint answers real questions out of the box. Tests that inject a fake
    # ``state.agent`` afterwards simply replace this default.
    _wire_production_agent(state)

    register_core_routes(app)
    register_blueprints(app)
    return app


def _wire_production_agent(state: "AppState") -> None:
    """Attach a real agent (model + search + runner + tool executor) to *state*.

    This is the production glue the app needs to actually answer questions: it
    builds the OpenAI-compatible model client, the DuckDuckGo search service,
    the bwrap sandbox runner, and the confined tool executor, then constructs an
    :class:`~pengyplexity.core.agent.Agent`. Tests skip this by substituting a
    fake ``state.agent`` after ``create_app``.
    """
    cfg = state.config

    from .core.agent import Agent
    from .core.artifacts import ArtifactService
    from .core.images import ImageService, create_image_backend
    from .core.memory import MemoryStore
    from .core.modelclient import create_model_client
    from .core.search import DDGSSearchService
    from .core.toolexec import build_tool_executor, get_tool_schemas
    from .sandbox import confine
    from .sandbox.executors import BwrapRunner
    from .sandbox.pythonenv import SandboxPythonEnv

    # The sandbox mounts only a read-only /usr, so the interpreter inside it
    # is the host's bare system Python — no matplotlib, no numpy, no pandas.
    # This venv is bound in read-only so chart scripts actually run; it is
    # built on first use (or ahead of time via `pengyplexity-admin
    # build-sandbox`) because building needs network access the sandbox does
    # not have.
    sandbox_env = SandboxPythonEnv(
        cfg.sandbox_venv_path, auto_build=cfg.sandbox_autobuild
    )
    runner = BwrapRunner(
        timeout=cfg.exec_timeout,
        mem_bytes=cfg.exec_mem_bytes,
        cpu_seconds=cfg.exec_cpu_seconds,
        python_env=sandbox_env.path if sandbox_env.ready else None,
        python_env_provider=sandbox_env.ensure,
    )
    search = DDGSSearchService(timeout=cfg.tool_network_timeout, user_agent=cfg.user_agent)
    model = create_model_client(
        base_url=cfg.model_base,
        api_key=cfg.model_key,
        model=cfg.model_name,
        temperature=cfg.model_temperature,
        timeout=cfg.llm_timeout,
    )
    workspace = cfg.workspace_root()  # per-thread roots are created under this
    workspace.mkdir(parents=True, exist_ok=True)

    # ``make_chart``/``generate_image``/``edit_image`` register their output as
    # real artifacts (see core/artifacts.py, core/images.py) so they can
    # actually be served/rendered inline instead of just being a claim about a
    # file "in your workspace" that nothing else can reach.
    artifact_service = ArtifactService(store=state.store, runner=runner)
    image_service = ImageService(store=state.store, backend=create_image_backend())
    memory_store = MemoryStore(
        cfg.memory_store_file,
        enable_semantic=cfg.memory_semantic,
        signal_floor=cfg.memory_signal_floor,
        signal_confident=cfg.memory_signal_confident,
    )
    # Built once: the schema list is static, and deriving it walks the whole
    # tool inventory.
    tool_schemas = get_tool_schemas()

    def make_agent():
        """Build one request's agent over the shared services.

        The tool executor is per-agent because it owns the turn's mutable
        ``ToolContext`` (thread id, owner, artifacts, pending images) — see
        :meth:`AppState.new_agent` for why sharing that across concurrent
        requests crossed users' work.
        """
        return Agent(
            model=model,
            tool_executor=build_tool_executor(
                runner, search, artifact_service, image_service, memory_store
            ),
            workspace=workspace,
            max_iterations=cfg.max_agent_iterations,
            tools=tool_schemas,
        )

    # Left as None on purpose: production goes through the factory, and a test
    # that assigns ``state.agent`` overrides it (see AppState.new_agent).
    state.agent = None
    state.agent_factory = make_agent
    state.model = model
    state.runner = runner
    state.search = search
    state.memory = memory_store
    state.sandbox_env = sandbox_env


def get_state(app: Flask) -> AppState:
    """Return the :class:`AppState` attached by :func:`create_app`."""
    return app.extensions["pengyplexity"]


def register_core_routes(app: Flask) -> None:
    """Register the always-present routes (home + health probe).

    Later subtasks register their own blueprints separately so this core set
    stays stable and trivially testable.
    """

    @app.route("/")
    def index():
        # Already logged in? Jump straight to the chat UI.
        from flask import redirect, session
        if session.get("user_id"):
            from .web import chat_index
            return chat_index()
        return render_template("index.html")

    @app.route("/healthz")
    def healthz():
        return {"status": "ok", "app": "pengyplexity"}

    @app.context_processor
    def inject_theme():
        """Make the logged-in user's theme preference available to every
        template (login/index included) without every route having to pass
        it explicitly. Unauthenticated pages and users with no saved
        preference get the app default (light / default accent — see
        core/theme.py)."""
        from flask import session

        from .core.theme import normalize_theme_accent, normalize_theme_mode

        user = None
        user_id = session.get("user_id")
        if user_id:
            state = get_state(app)
            user = state.store.get_user_by_id(user_id)
        return {
            "theme_mode": normalize_theme_mode((user or {}).get("theme_mode")),
            "theme_accent": normalize_theme_accent((user or {}).get("theme_accent")),
        }


def register_blueprints(app: Flask) -> None:
    """Register feature blueprints (web UI, admin, etc.).

    Called from :func:`create_app` after core routes. Each blueprint
    is imported lazily so the app factory stays fast and testable.
    """
    from .web import web_bp
    from .admin import admin_bp
    from .api import api_bp
    app.register_blueprint(web_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(api_bp)
