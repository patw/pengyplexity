"""A small in-process sliding-window rate limiter for the JSON API.

Every API turn runs a model call and possibly the sandbox, so an unthrottled
client (a Discord bot relaying a busy channel, a script stuck in a retry loop)
can pin the host. This caps turns per user per minute.

In-process on purpose: Pengyplexity runs as a single host process (bwrap rules
out containers, and the stores are moofile files), so there is no shared
backend to coordinate with.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, Deque, Dict, Tuple


class RateLimiter:
    """Allow at most *limit* events per *window* seconds per key.

    ``limit <= 0`` disables limiting. *clock* is injectable for tests.
    """

    def __init__(
        self,
        limit: int,
        window: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.limit = limit
        self.window = window
        self._clock = clock
        self._hits: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()

    def hit(self, key: str) -> Tuple[bool, int]:
        """Record an attempt for *key*.

        Returns ``(allowed, retry_after_seconds)``. A refused attempt is not
        counted, so a client that backs off for ``retry_after`` gets through.
        """
        if self.limit <= 0:
            return True, 0
        now = self._clock()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                retry_after = max(1, int(self.window - (now - hits[0]) + 0.999))
                return False, retry_after
            hits.append(now)
            return True, 0
