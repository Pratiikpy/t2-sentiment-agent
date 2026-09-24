"""Time, injected. Nothing in this project reads the wall clock except :class:`SystemClock`.

Every schedule, cooldown, weekend rule and daily limit depends on "now", and a test that cannot set
"now" cannot test a Friday 19:45 UTC pre-flatten or a UTC-midnight kill-switch reset. So modules
take a :class:`sentiment_agent.types.Clock` and tests pass :class:`ManualClock`.
"""

import json
import threading
import time
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

BITGET_TIME_URL = "https://api.bitget.com/api/v2/public/time"
RESYNC_EVERY = 600.0
MAX_OFFSET = 300.0


class SystemClock:
    """The real clock, always UTC."""

    def now(self) -> datetime:
        return datetime.now(UTC)


def bitget_offset(fetch: Callable[[], float] | None = None) -> float:  # pragma: no cover - network
    """Seconds to add to this machine's clock to read Bitget's server time, from one round trip
    (the midpoint of the request is taken as the instant the server stamped)."""
    if fetch is not None:
        return fetch()
    before = time.time()
    with urllib.request.urlopen(BITGET_TIME_URL, timeout=10) as resp:
        server = int(json.load(resp)["data"]["serverTime"]) / 1000
    after = time.time()
    return server - (before + after) / 2


class VenueClock:
    """The real clock, corrected to the venue's own time.

    **Why.** Bitget stamps every order and fill on its server clock. This machine measured 3.27 s
    behind it on 2026-09-24, and the first Demo plumbing test proved what that costs: a
    reconciliation one second after a fill read up to "now" on the local clock, the fill was stamped
    later than that, and the book never learned of the position it had just opened. Every "now" in
    the agent — windows, cooldowns, marks, the ledger — is taken on the venue's clock instead.

    The offset is re-measured every :data:`RESYNC_EVERY` seconds; a failed measurement keeps the
    last good one. An offset larger than :data:`MAX_OFFSET` is refused as a fault, not applied.
    Time never runs backwards: a re-measure that would move "now" earlier holds it still instead.
    """

    def __init__(self, offset: Callable[[], float] = bitget_offset) -> None:
        self._measure = offset
        self._lock = threading.Lock()
        self._offset = 0.0
        self._measured_at = float("-inf")
        self._last = datetime.min.replace(tzinfo=UTC)
        self._resync(force=True)

    @property
    def offset_seconds(self) -> float:
        return self._offset

    def _resync(self, *, force: bool = False) -> None:
        mono = time.monotonic()
        if not force and mono - self._measured_at < RESYNC_EVERY:
            return
        try:
            value = float(self._measure())
        except Exception:  # keep the last good offset; the next tick tries again
            self._measured_at = mono
            return
        if abs(value) <= MAX_OFFSET:
            self._offset = value
        self._measured_at = mono

    def now(self) -> datetime:
        with self._lock:
            self._resync()
            current = datetime.now(UTC) + timedelta(seconds=self._offset)
            if current < self._last:
                current = self._last
            self._last = current
            return current


class ManualClock:
    """A clock a test moves by hand. Refuses naive or non-UTC times."""

    def __init__(self, start: datetime) -> None:
        if start.tzinfo is None or start.utcoffset() != timedelta(0):
            raise ValueError("ManualClock needs a timezone-aware UTC datetime")
        self._now = start.astimezone(UTC)

    def now(self) -> datetime:
        return self._now

    def set(self, when: datetime) -> None:
        if when.tzinfo is None or when.utcoffset() != timedelta(0):
            raise ValueError("ManualClock needs a timezone-aware UTC datetime")
        if when < self._now:
            raise ValueError("ManualClock does not run backwards")
        self._now = when.astimezone(UTC)

    def advance(self, delta: timedelta) -> datetime:
        if delta < timedelta(0):
            raise ValueError("ManualClock does not run backwards")
        self._now = self._now + delta
        return self._now


__all__ = ["ManualClock", "SystemClock", "VenueClock", "bitget_offset"]
