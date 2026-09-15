"""Interrupting a turn in progress.

The chat UI's Ask button turns into Stop while the agent is working, and
pressing it has to do two separate things:

1. **Free the browser immediately** — the client aborts its own ``fetch`` and
   finalizes whatever text has arrived. That part needs no server help.
2. **Wind the server down** — stop the agent loop, kill whatever the sandbox
   is currently running, and persist the partial answer. That is what this
   module provides.

The shape is Pengy's (``pengy/web/app.py``: a registry of active workers, a
``/stop`` route, and a ``cancel()`` that flips a flag *and* kills the run's
subprocesses via ``ToolContext.kill_all``), adapted to Pengyplexity, where the
SSE request itself drives the agent rather than a background worker thread.

Two pieces:

* :class:`CancelToken` — one turn's flag plus the set of sandbox processes it
  currently has running. Cancelling sets the flag and kills them, so a stop
  during a 30-second chart script takes effect at once instead of after the
  execution timeout.
* :class:`CancelRegistry` — maps a live turn to its token so the ``/stop``
  route, which arrives on a *different* request and a different thread, can
  find it.

The flag is only ever *read* by the agent at safe points (between streamed
chunks, between tool calls, between loop iterations), so a cancelled turn
always unwinds through normal control flow — the partial answer is kept and
persisted, never lost to a mid-write kill.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional, Set, Tuple


class TurnCancelled(Exception):
    """Raised inside a turn once its :class:`CancelToken` has been cancelled."""


class TurnInProgress(Exception):
    """Raised by an exclusive :meth:`CancelRegistry.start` when the thread is busy."""


class TooManyTurns(Exception):
    """Raised by :meth:`CancelRegistry.start` when the owner is at their cap."""


class CancelToken:
    """The cancellation flag for a single turn, plus its running processes."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._procs: Set[Any] = set()
        self._lock = threading.Lock()

    # -- state -------------------------------------------------------------

    @property
    def cancelled(self) -> bool:
        """True once :meth:`cancel` has been called."""
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        """Raise :class:`TurnCancelled` if this turn has been stopped."""
        if self._event.is_set():
            raise TurnCancelled()

    # -- control -----------------------------------------------------------

    def cancel(self) -> None:
        """Stop the turn: set the flag and kill anything it is running now.

        Called from the ``/stop`` request's thread, not the turn's own, so
        every mutation is under the lock.
        """
        self._event.set()
        with self._lock:
            procs = list(self._procs)
            self._procs.clear()
        for proc in procs:
            _kill(proc)

    def register_process(self, proc: Any) -> None:
        """Track a sandbox subprocess so :meth:`cancel` can kill it.

        A process started after cancellation is killed immediately rather than
        being allowed to run: the turn is already over.
        """
        with self._lock:
            self._procs.add(proc)
        if self._event.is_set():
            self.unregister_process(proc)
            _kill(proc)

    def unregister_process(self, proc: Any) -> None:
        """Stop tracking a subprocess that has finished on its own."""
        with self._lock:
            self._procs.discard(proc)


def _kill(proc: Any) -> None:
    """Terminate *proc*, then kill it if it does not go quietly."""
    try:
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:  # noqa: BLE001 - TimeoutExpired and anything odder
            proc.kill()
    except Exception:  # noqa: BLE001
        # The process may already be reaped; stopping is best-effort and must
        # never fail the /stop request.
        pass


class CancelRegistry:
    """Tracks the in-flight turn for each ``(owner, thread)`` pair.

    The ``/stop`` route arrives as its own request on its own thread and has
    only the thread id to go on, so a running turn has to be findable by that.
    Keyed by owner as well so one user can never cancel another's turn even if
    they guess a thread id.
    """

    def __init__(self) -> None:
        self._tokens: Dict[Tuple[str, str], CancelToken] = {}
        self._lock = threading.Lock()

    def start(
        self,
        owner: str,
        thread_id: str,
        exclusive: bool = False,
        max_active: int = 0,
    ) -> CancelToken:
        """Register a new turn and return its token.

        By default a turn already running for the same thread is cancelled
        first: the user has asked a new question in a thread that was still
        working, and two concurrent turns would interleave into the same
        message list. That suits the browser, where the old turn's page is
        gone. An API client may still be reading the old turn, so it passes
        ``exclusive=True`` and gets :class:`TurnInProgress` instead.

        ``max_active > 0`` caps how many threads *owner* may have running at
        once (:class:`TooManyTurns`). Replacing a turn in the same thread does
        not count as a new one. Both checks happen under the registry lock, so
        two requests racing for the last slot cannot both win.
        """
        token = CancelToken()
        key = (owner, thread_id)
        with self._lock:
            previous = self._tokens.get(key)
            if previous is not None and exclusive:
                raise TurnInProgress(thread_id)
            if previous is None and max_active > 0:
                running = sum(1 for (o, _t) in self._tokens if o == owner)
                if running >= max_active:
                    raise TooManyTurns(owner)
            self._tokens[key] = token
        if previous is not None:
            previous.cancel()
        return token

    def cancel(self, owner: str, thread_id: str) -> bool:
        """Cancel the turn running for this thread. True if there was one."""
        with self._lock:
            token = self._tokens.get((owner, thread_id))
        if token is None:
            return False
        token.cancel()
        return True

    def finish(self, owner: str, thread_id: str, token: CancelToken) -> None:
        """Deregister *token* once its turn is over.

        Compares identity before removing so a turn finishing late never
        deregisters the newer turn that replaced it.
        """
        with self._lock:
            if self._tokens.get((owner, thread_id)) is token:
                del self._tokens[(owner, thread_id)]

    def active(self, owner: str, thread_id: str) -> Optional[CancelToken]:
        """The token for this thread's running turn, or None."""
        with self._lock:
            return self._tokens.get((owner, thread_id))

    def __len__(self) -> int:
        with self._lock:
            return len(self._tokens)
