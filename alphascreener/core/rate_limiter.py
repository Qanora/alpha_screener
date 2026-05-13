"""Async rate limiter for LLM API call throttling (issue #15)."""

import asyncio
import logging
import time
from collections import deque

logger = logging.getLogger(__name__)


class RateLimiter:
    """Sliding-window rate limiter for async API calls.

    Limits the number of calls per second. RPS=0 blocks indefinitely.
    Uses deque for O(1) expiry of old timestamps.
    """

    def __init__(self, rps: int) -> None:
        if rps < 0:
            raise ValueError(f"rps must be >= 0, got {rps}")
        self._rps = rps
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a permit. May block if the rate limit is exceeded."""
        if self._rps == 0:
            logger.debug("RateLimiter RPS=0, blocking indefinitely")
            await asyncio.Event().wait()
            return

        async with self._lock:
            now = time.monotonic()
            window_start = now - 1.0
            while self._timestamps and self._timestamps[0] < window_start:
                self._timestamps.popleft()

            if len(self._timestamps) < self._rps:
                self._timestamps.append(now)
                return

            # Capture earliest timestamp while holding lock
            wait_time = self._timestamps[0] + 1.0 - now

        if wait_time > 0:
            logger.debug(
                "RateLimiter throttling for %.3fs (%d/%d calls in window)",
                wait_time,
                len(self._timestamps),
                self._rps,
            )
            await asyncio.sleep(wait_time)

        async with self._lock:
            self._timestamps.append(time.monotonic())
