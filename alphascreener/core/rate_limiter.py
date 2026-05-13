"""Async rate limiter for LLM API call throttling (issue #15)."""

import asyncio
import time


class RateLimiter:
    """Token-bucket-like rate limiter for async API calls.

    Limits the number of calls per second using a sliding window approach.
    Thread-safe for use with asyncio.
    """

    def __init__(self, rps: int) -> None:
        if rps < 0:
            raise ValueError(f"rps must be >= 0, got {rps}")
        self._rps = rps
        self._timestamps: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a permit. May block if the rate limit is exceeded."""
        if self._rps == 0:
            # RPS=0 means fully blocked — wait indefinitely
            await asyncio.Event().wait()
            return

        async with self._lock:
            now = time.monotonic()
            # Remove timestamps older than 1 second
            window_start = now - 1.0
            self._timestamps = [t for t in self._timestamps if t >= window_start]

            if len(self._timestamps) >= self._rps:
                # Need to wait until the oldest timestamp expires
                wait_time = self._timestamps[0] - window_start
                if wait_time > 0:
                    # Release lock while waiting
                    pass

            # If we would exceed, wait
            while len(self._timestamps) >= self._rps:
                # Release lock temporarily to let others proceed
                self._lock.release()
                try:
                    wait_until = self._timestamps[0] + 1.0
                    sleep_time = wait_until - time.monotonic()
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
                finally:
                    await self._lock.acquire()
                    now = time.monotonic()
                    window_start = now - 1.0
                    self._timestamps = [t for t in self._timestamps if t >= window_start]

            self._timestamps.append(time.monotonic())
