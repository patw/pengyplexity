"""One chat turn's lifecycle, shared by the web UI and the JSON API.

A turn is: store the question, name the thread if it is new, build a fresh
agent pinned to the thread's workspace, apply the admin's live settings,
register the turn so it can be stopped, run the agent, persist the answer
(even a partial one), and report any artifacts it produced.

The browser's SSE route (``web.py``) and the API (``api.py``) both go through
:func:`begin_turn` + :func:`run_turn`, so an API client gets exactly the agent
the web UI gets — same tools, same sandbox, same settings, same persistence
and cancellation rules — rather than a second implementation that drifts.

Everything here takes the :class:`~pengyplexity.app.AppState` explicitly; the
only Flask-specific piece a caller supplies is how to build artifact URLs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional

from .core.streaming import StreamEvent, agent_events
from .sandbox import confine

DEFAULT_THREAD_TITLE = "New Thread"


class AgentUnavailable(Exception):
    """No agent is configured, so there is nothing to answer with."""


class ThreadGone(Exception):
    """The thread was deleted between the caller's ownership check and the turn."""


@dataclass
class Turn:
    """A started turn, ready for :func:`run_turn`."""

    state: Any
    owner: str
    thread_id: str
    question: str
    agent: Any
    cancel: Any
    history: List[Dict[str, Any]]
    new_title: Optional[str] = None
    # Where this turn's assistant message will land in the thread.
    answer_index: int = 0

    @property
    def tool_context(self):
        executor = getattr(self.agent, "tool_executor", None)
        return getattr(executor, "context", None) if executor is not None else None


def begin_turn(
    state,
    user: Dict[str, Any],
    thread_id: str,
    question: str,
    exclusive: bool = False,
    max_active: int = 0,
) -> Turn:
    """Start a turn in *thread_id* (which the caller has checked *user* owns).

    Raises :class:`AgentUnavailable`, or — from the cancel registry, before
    anything is written — :class:`~pengyplexity.core.cancel.TurnInProgress`
    (``exclusive``) / :class:`~pengyplexity.core.cancel.TooManyTurns`
    (``max_active``). A refused turn leaves the thread untouched, so a client
    can simply retry.
    """
    store = state.store
    owner = user["username"]

    agent = state.new_agent()
    if agent is None:
        raise AgentUnavailable("Agent not configured")

    # Registered first: this is the step that can refuse the turn, and it
    # must do so before the question lands in the thread.
    cancel = state.cancels.start(owner, thread_id, exclusive=exclusive, max_active=max_active)
    try:
        updated = store.append_message(thread_id, "user", question)
        if updated is None:
            raise ThreadGone(thread_id)
        new_title = maybe_name_thread(state, thread_id, question)

        agent.workspace = confine.workspace(
            owner, thread_id, base=state.config.workspace_root()
        )
        # A brand-new thread has no workspace directory on disk yet (it's
        # created lazily by write_file). Without this, the FIRST tool call in
        # a thread — if it's run_bash/run_python/make_chart rather than
        # write_file — hits a missing `cwd`/bind source and raises/exits with
        # a raw host path (e.g. "/home/<realuser>/.pengyplexity/workspaces/...")
        # in the tool result, leaking the real host username straight past the
        # sandbox.
        agent.workspace.mkdir(parents=True, exist_ok=True)
        prime_tool_context(agent, thread_id, state)
        settings = apply_effective_settings(state, agent, username=owner)
        history = thread_history(
            store, thread_id, max_messages=int(settings["thread_history_messages"])
        )

        # The token also reaches the sandbox runner (via the tool context) so
        # Stop kills a running script instead of waiting out the timeout.
        agent.cancel = cancel
        turn = Turn(
            state=state,
            owner=owner,
            thread_id=thread_id,
            question=question,
            agent=agent,
            cancel=cancel,
            history=history,
            new_title=new_title,
            answer_index=len(updated.get("messages") or []),
        )
        if turn.tool_context is not None:
            turn.tool_context.cancel = cancel
        return turn
    except BaseException:
        state.cancels.finish(owner, thread_id, cancel)
        raise


def run_turn(
    turn: Turn,
    artifact_links: Callable[[Any], Dict[str, str]],
) -> Iterator[StreamEvent]:
    """Run *turn*, yielding ``title`` / ``activity`` / ``token`` / ``done`` /
    ``artifact`` / ``error`` events.

    *artifact_links(record)* returns the URL fields for one artifact event
    (``url``, ``download_url``); it runs while the generator is being
    consumed, so a Flask caller wraps the generator in
    ``stream_with_context``.

    The answer is persisted exactly once, including when the consumer stops
    iterating early (``GeneratorExit`` — a browser or API client hanging up),
    and the turn is always deregistered.
    """
    state = turn.state
    collected: List[str] = []
    failed = False
    persisted = False
    try:
        if turn.new_title:
            yield StreamEvent("title", {"title": turn.new_title})
        for event in agent_events(turn.agent, turn.question, history=turn.history):
            if event.event == "token":
                collected.append(event.data.get("content", ""))
            elif event.event == "error":
                failed = True
            yield event
        persist_answer(
            state.store, turn.thread_id, turn.agent, collected, turn.cancel, failed=failed
        )
        persisted = True

        # Any chart/image the model produced this turn (via make_chart /
        # generate_image / edit_image) is already persisted as an artifact —
        # surface it now so the client can show it without a reload.
        context = turn.tool_context
        for record in (context.artifacts if context else []):
            if not record._id:
                continue
            yield StreamEvent("artifact", {
                "artifact_id": record._id,
                "filename": record.filename,
                "kind": record.kind,
                "mime": record.mime,
                **artifact_links(record),
            })
    except Exception as e:  # noqa: BLE001
        failed = True
        yield StreamEvent("error", {"message": str(e)})
    finally:
        # Reached on the normal path, on an error, and — the case that
        # matters — on GeneratorExit when the client hangs up because the user
        # pressed Stop. Whatever the model had written by then is the user's;
        # persisting it here is what keeps a stopped turn from vanishing out
        # of the thread on the next page load.
        if not persisted:
            persist_answer(
                state.store, turn.thread_id, turn.agent, collected, turn.cancel,
                interrupted=True, failed=failed,
            )
        state.cancels.finish(turn.owner, turn.thread_id, turn.cancel)


def persist_answer(
    store, thread_id: str, agent, collected, cancel,
    interrupted: bool = False, failed: bool = False,
) -> None:
    """Store this turn's answer, marking it when the turn did not run to completion.

    The one place a turn's assistant message is written, so a stopped turn is
    saved exactly like a finished one — the text the model had already
    produced belongs to the user and has to survive a reload.

    *interrupted* means the generator is unwinding because the client hung up
    (``GeneratorExit``) rather than finishing normally. *failed* means the
    turn ended in an ``error`` event.
    """
    answer = "".join(collected).strip()
    stopped = cancel is not None and cancel.cancelled

    if stopped:
        note = "_[Stopped]_"
    elif interrupted:
        note = "_[Interrupted]_"
    elif failed:
        note = "_[Error]_"
    else:
        note = ""

    if note:
        if not answer and not stopped:
            # Nothing was produced and nobody asked to stop: the failure was
            # already surfaced as an `error` event, and a blank assistant
            # bubble in the thread would only be confusing.
            return
        answer = f"{answer}\n\n{note}" if answer else note

    try:
        store.append_message(
            thread_id, "assistant", answer,
            sources=getattr(agent, "_last_sources", None) or [],
        )
    except Exception:  # noqa: BLE001
        # Never let bookkeeping raise out of a generator that is already
        # unwinding — it would replace the real reason in the log.
        if not interrupted:
            raise


def prime_tool_context(agent, thread_id: str, state) -> None:
    """Point the agent's tool executor's artifact context at *thread_id*.

    ``make_chart``/``generate_image``/``edit_image`` (see ``core/toolexec.py``)
    record the artifacts they create against ``context.thread_id`` /
    ``context.message_index``. The assistant's answer for this turn hasn't
    been appended yet, so its eventual index is simply the thread's current
    message count (the user's question was already appended before the agent
    runs). Also resets ``context.artifacts`` so a caller can inspect exactly
    what this turn produced.
    """
    executor = getattr(agent, "tool_executor", None)
    context = getattr(executor, "context", None) if executor is not None else None
    if context is None:
        return
    thread = state.store.get_thread(thread_id)
    context.thread_id = thread_id
    context.message_index = len(thread.get("messages", [])) if thread else 0
    context.owner = thread.get("owner", "") if thread else ""
    context.artifacts = []


def _compose_system_prompt(state, settings, username: str | None) -> str | None:
    """The system prompt for one turn: the admin's global template (or the
    built-in default), with that user's own instructions appended.

    A per-user message *adds to* the prompt instead of replacing it. Replacing
    is what the global override does, and it is the right thing there; doing it
    per user would mean a note about tone ("you are in a Discord channel, keep
    it short") silently costing the agent everything the default prompt tells
    it about charts, images, memory and the sandbox.
    """
    from .core.agent import SYSTEM_PROMPT
    from .core.settings import render_system_message, user_system_message

    def render(text: str) -> str:
        try:
            return render_system_message(text, username=username)
        except (KeyError, IndexError, ValueError):
            # Bad admin-entered placeholder (e.g. {typo}) — fall back to the
            # raw template rather than breaking every turn.
            return text

    base = settings["system_message"]
    base = render(base) if base else ""
    extra = user_system_message(state.store, username)
    if not extra:
        return base or None
    if not base:
        base = SYSTEM_PROMPT.format(max_iterations=int(settings["max_agent_iterations"]))
    return f"{base}\n\n{render(extra)}"


def apply_effective_settings(state, agent, username: str | None = None) -> Dict[str, Any]:
    """Refresh the shared agent/runner/model/tool-context from the effective
    settings (admin override in the store, else the env-loaded Config), and
    return those settings so a caller needing one of them (the history cap)
    does not re-read the store —
    see ``core/settings.py``. Mutates existing objects in place (the same
    "shared singleton repointed per turn" pattern as ``agent.workspace``)
    rather than reconstructing them, so a setting change takes effect on the
    very next turn with no app restart.

    *agent* is **this request's** agent (see ``AppState.new_agent``), not a
    shared one. The runner/model/search it configures are shared, but every
    value written there is a global admin setting, identical for every
    request, so concurrent writes of the same value are harmless. The
    per-turn values (system prompt, iteration cap, tool-output limits) land
    on the request's own agent and tool context.

    *username* is the app user the turn runs as, used to fill ``{username}``
    in the system message template — NOT the OS user the process runs as,
    since Pengyplexity is multi-user.

    A no-op for fakes used in tests (``FakeAgent``/``FakeModelClient`` etc.
    simply lack the attributes this touches, all accessed via ``getattr``).
    """
    from .core.settings import effective_settings

    settings = effective_settings(state.store, state.config)
    if agent is None:
        return settings

    if hasattr(agent, "system_prompt"):
        agent.system_prompt = _compose_system_prompt(state, settings, username)
    if hasattr(agent, "max_iterations"):
        agent.max_iterations = int(settings["max_agent_iterations"])

    model = getattr(agent, "model", None)
    if model is not None and hasattr(model, "configure"):
        model.configure(
            base_url=settings["model_base"],
            api_key=settings["model_key"],
            model=settings["model_name"],
            temperature=float(settings["model_temperature"]),
            timeout=float(settings["llm_timeout"]),
        )

    runner = state.runner
    if runner is not None:
        if hasattr(runner, "timeout"):
            runner.timeout = int(settings["exec_timeout"])
        if hasattr(runner, "mem_bytes"):
            runner.mem_bytes = int(settings["exec_mem_mb"]) * 1024 * 1024
        if hasattr(runner, "cpu_seconds"):
            runner.cpu_seconds = int(settings["exec_cpu_seconds"])

    memory = state.memory
    if memory is not None:
        # The right relevance cut point depends on the corpus and the
        # embedding model, and the shipped defaults come from a small probe —
        # so it is tunable live rather than needing a redeploy. See the
        # calibration note in core/memory.py.
        if hasattr(memory, "signal_floor"):
            memory.signal_floor = float(settings["memory_signal_floor"])
        if hasattr(memory, "signal_confident"):
            memory.signal_confident = float(settings["memory_signal_confident"])

    search = state.search
    if search is not None:
        if hasattr(search, "timeout"):
            search.timeout = int(settings["tool_network_timeout"])
        if hasattr(search, "user_agent"):
            search.user_agent = settings["user_agent"]

    executor = getattr(agent, "tool_executor", None)
    context = getattr(executor, "context", None) if executor is not None else None
    if context is not None:
        context.tool_output_max_chars = int(settings["tool_output_max_chars"])
        context.download_max_mb = float(settings["download_max_mb"])
        context.tool_network_timeout = int(settings["tool_network_timeout"])
        context.user_agent = settings["user_agent"]
    return settings


def thread_history(store, thread_id: str, max_messages: int = 0) -> list:
    """Build the prior message history for *thread_id* (excluding the just-added
    user message — the agent adds it itself).

    *max_messages* keeps only the most recent N messages; 0 means all of them.
    The whole history is re-sent to the model on every turn, so an unbounded
    one makes each question in a long-lived thread cost more than the last —
    a Discord channel bound to a single thread for weeks is the case that
    bites. Trimmed history still starts on a user message, so the turns stay
    paired the way a chat completion expects.
    """
    thread = store.get_thread(thread_id)
    history = []
    for msg in thread.get("messages", []):
        if msg["role"] in ("user", "assistant"):
            history.append({"role": msg["role"], "content": msg["content"]})
    if history and history[-1]["role"] == "user":
        history = history[:-1]
    if max_messages and len(history) > max_messages:
        history = history[-max_messages:]
        # Slicing can land on an assistant turn whose question is now gone,
        # which reads as the model answering nothing.
        if history and history[0]["role"] == "assistant":
            history = history[1:]
    return history


def maybe_name_thread(state, thread_id: str, question: str) -> Optional[str]:
    """Set a human-friendly title on a thread from its first question.

    Only names a thread whose title is still the default "New Thread". Uses a
    small LLM call via the app's model client; on any failure it falls back to
    a truncated first line of the question so the title is always set.

    Returns the new title, or ``None`` if the thread already had a real title
    (nothing changed) — callers use this to tell the client about the rename.
    """
    store = state.store
    thread = store.get_thread(thread_id)
    if thread is None:
        return None
    title = (thread.get("title") or "").strip()
    if title and title != DEFAULT_THREAD_TITLE:
        return None
    model = title_model(state)
    new_title = model.generate_title(question) if model is not None else None
    if not new_title:
        first = question.splitlines()[0].strip() if question.strip() else DEFAULT_THREAD_TITLE
        new_title = first[:60] or DEFAULT_THREAD_TITLE
    store.update_thread_title(thread_id, new_title)
    return new_title


def title_model(state):
    """Return the app's model client (for title generation), or None.

    Prefers the shared client on the app state. Building a whole request
    agent just to reach its model would be wasteful, and a test that injects
    a fake agent still needs its fake model honoured — hence the fallback.
    """
    injected = getattr(state, "agent", None)
    if injected is not None:
        return getattr(injected, "model", None)
    return getattr(state, "model", None)
