"""Per-application token buckets for unauthenticated cluster requests."""

import threading
import time


class ProbeRateLimiter:
    """Token-bucket rate limiter for the unauthenticated probe endpoint."""

    def __init__(self, rate_per_second: float = 5.0, burst: int = 10) -> None:
        self._rate = float(rate_per_second)
        self._burst = int(burst)
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            tokens, updated = self._buckets.get(key, (float(self._burst), now))
            tokens = min(float(self._burst), tokens + (now - updated) * self._rate)
            if tokens < 1.0:
                self._buckets[key] = (tokens, now)
                return False
            self._buckets[key] = (tokens - 1.0, now)
            # Bound the map: drop idle buckets once it grows past 4096 keys.
            if len(self._buckets) > 4096:
                self._buckets = {
                    k: v for k, v in self._buckets.items() if now - v[1] < 600.0
                }
            return True
