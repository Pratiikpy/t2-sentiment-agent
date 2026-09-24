"""The venue clock: the agent's "now" follows Bitget's server time, never runs backwards, and a
failed or absurd measurement never moves it."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from sentiment_agent import clock as clock_module
from sentiment_agent.clock import VenueClock


def test_now_is_shifted_by_the_measured_offset() -> None:
    venue = VenueClock(offset=lambda: 3.27)
    assert venue.offset_seconds == pytest.approx(3.27)
    gap = (venue.now() - datetime.now(UTC)).total_seconds()
    assert 3.0 < gap < 3.6


def test_a_failed_or_absurd_measurement_keeps_the_last_good_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answers = iter([2.0, 10_000.0])
    venue = VenueClock(offset=lambda: next(answers))
    assert venue.offset_seconds == 2.0
    monkeypatch.setattr(clock_module, "RESYNC_EVERY", 0.0)
    venue.now()  # re-measures: 10,000 s is refused
    assert venue.offset_seconds == 2.0

    def boom() -> float:
        raise OSError("offline")

    broken = VenueClock(offset=boom)
    assert broken.offset_seconds == 0.0


def test_time_never_runs_backwards(monkeypatch: pytest.MonkeyPatch) -> None:
    answers = iter([5.0, -5.0])
    venue = VenueClock(offset=lambda: next(answers))
    first = venue.now()
    monkeypatch.setattr(clock_module, "RESYNC_EVERY", 0.0)
    second = venue.now()  # offset drops by 10 s; now holds instead of jumping back
    assert second >= first
    assert second - first < timedelta(seconds=1)
