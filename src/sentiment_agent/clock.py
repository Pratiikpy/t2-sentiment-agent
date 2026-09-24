"""Time, injected. Nothing in this project reads the wall clock except :class:`SystemClock`.

Every schedule, cooldown, weekend rule and daily limit depends on "now", and a test that cannot set
"now" cannot test a Friday 19:45 UTC pre-flatten or a UTC-midnight kill-switch reset. So modules
take a :class:`sentiment_agent.types.Clock` and tests pass :class:`ManualClock`.
"""

from datetime import UTC, datetime, timedelta


class SystemClock:
    """The real clock, always UTC."""

    def now(self) -> datetime:
        return datetime.now(UTC)


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


__all__ = ["ManualClock", "SystemClock"]
