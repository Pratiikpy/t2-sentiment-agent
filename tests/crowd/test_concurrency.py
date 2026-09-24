"""Crowd collection with searches in flight at once, as the production wiring runs it.

The concurrent paths must record exactly what the serial ones record (same calls, same order, same
items), keep the serial path's stop-on-a-repeating-failure behaviour, and genuinely overlap. No test
here runs ``twitter`` or ``rdt``: the runners are scripted, as in ``test_adapters.py``.
"""

import threading
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from sentiment_agent.clock import ManualClock
from sentiment_agent.crowd.adapters import (
    MAX_CONCURRENCY,
    CompositeCrowd,
    RedditCollector,
    XCollector,
)
from sentiment_agent.hashing import canonical_json
from sentiment_agent.types import SourceCall, SourceHealth, TextItem

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "crowd"
NOW = datetime(2026, 9, 23, 13, 0, tzinfo=UTC)
SINCE = NOW - timedelta(hours=24)
SYMBOLS = ["NVDAUSDT", "TSLAUSDT", "AAPLUSDT", "METAUSDT", "COINUSDT", "BTCUSDT"]


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


class ByQuery:
    """Answers by the query in the argv, so the answer does not depend on call order."""

    def __init__(self, default: str, special: dict[str, tuple[int, str]] | None = None) -> None:
        self._default = default
        self._special = special or {}
        self._lock = threading.Lock()
        self.queries: list[str] = []

    def __call__(self, argv: Sequence[str], timeout: float) -> tuple[int, str, str]:
        query = argv[2]
        with self._lock:
            self.queries.append(query)
        for needle, (status, name) in self._special.items():
            if needle in query:
                return status, _fixture(name), ""
        return 0, _fixture(self._default), ""


class Overlapping(ByQuery):
    """Every search waits until ``width`` searches are in flight together; a serial collector
    would wait alone and the barrier would break, which the collector records as an ERROR."""

    def __init__(self, default: str, width: int) -> None:
        super().__init__(default)
        self._barrier = threading.Barrier(width, timeout=10)
        self.broken = False

    def __call__(self, argv: Sequence[str], timeout: float) -> tuple[int, str, str]:
        try:
            self._barrier.wait()
        except threading.BrokenBarrierError:
            self.broken = True
            raise
        return super().__call__(argv, timeout)


def _x(runner: ByQuery, concurrency: int = 1) -> XCollector:
    return XCollector(runner=runner, clock=ManualClock(NOW), concurrency=concurrency)


def _reddit(runner: ByQuery, concurrency: int = 1) -> RedditCollector:
    return RedditCollector(runner=runner, clock=ManualClock(NOW), concurrency=concurrency)


@pytest.mark.parametrize("width", [2, 3, MAX_CONCURRENCY])
def test_concurrent_collection_records_exactly_what_the_serial_one_does(width: int) -> None:
    serial = _x(ByQuery("x_search_ok.json")).collect(SYMBOLS, since=SINCE)
    concurrent = _x(ByQuery("x_search_ok.json"), width).collect(SYMBOLS, since=SINCE)
    assert canonical_json(concurrent) == canonical_json(serial)
    assert [c.source for c in concurrent[1]] == [f"twitter-cli:search:{s}" for s in SYMBOLS]
    assert concurrent[0], "the fixture must yield items, or this comparison proves little"


def test_searches_really_overlap() -> None:
    # The first search runs alone (to catch a repeating failure cheaply); the other five then run
    # as one batch, which can only pass the barrier if all five are in flight together.
    runner = Overlapping("x_search_ok.json", width=len(SYMBOLS) - 1)
    first_alone = ByQuery("x_search_ok.json")

    class Split:
        def __init__(self) -> None:
            self.n = 0

        def __call__(self, argv: Sequence[str], timeout: float) -> tuple[int, str, str]:
            self.n += 1
            return (first_alone if self.n == 1 else runner)(argv, timeout)

    _, calls = _x(ByQuery("x_search_ok.json")).collect(SYMBOLS, since=SINCE)
    collector = XCollector(runner=Split(), clock=ManualClock(NOW), concurrency=len(SYMBOLS) - 1)
    _, got = collector.collect(SYMBOLS, since=SINCE)
    assert not runner.broken
    assert [c.health for c in got] == [c.health for c in calls]


def test_a_failure_on_the_first_search_skips_the_rest_without_calling_them() -> None:
    runner = ByQuery("x_search_ok.json", special={"$NVDA": (1, "x_error_not_authenticated.json")})
    items, calls = _x(runner, concurrency=3).collect(SYMBOLS, since=SINCE)
    assert items == ()
    assert runner.queries == ["$NVDA"]
    assert [c.health for c in calls] == [SourceHealth.ERROR] + [SourceHealth.DISABLED] * 5
    assert all((c.error or "").startswith("skipped: not_authenticated") for c in calls[1:])


def test_a_repeating_failure_in_flight_skips_every_search_not_yet_started() -> None:
    runner = ByQuery("x_search_ok.json", special={"$TSLA": (1, "x_error_rate_limited.json")})
    _, calls = _x(runner, concurrency=2).collect(SYMBOLS, since=SINCE)
    assert [c.source for c in calls] == [f"twitter-cli:search:{s}" for s in SYMBOLS]
    tsla = calls[1]
    assert tsla.health is SourceHealth.ERROR
    assert (tsla.error or "").startswith("rate_limited")
    skipped = [c for c in calls if c.health is SourceHealth.DISABLED]
    assert all((c.error or "").startswith("skipped: rate_limited") for c in skipped)
    # A search is either skipped or really made, never both. How many were already in flight when
    # the limit was reported depends on thread timing, so only the invariant is asserted.
    assert len(runner.queries) + len(skipped) == len(SYMBOLS)
    assert all(c.health is not SourceHealth.DISABLED for c in calls[:2])


@pytest.mark.parametrize("width", [0, MAX_CONCURRENCY + 1])
def test_an_impossible_width_is_refused(width: int) -> None:
    with pytest.raises(ValueError, match="concurrency"):
        _x(ByQuery("x_search_ok.json"), width)


def test_parallel_collectors_merge_exactly_as_serial_ones_do() -> None:
    def crowd(parallel: bool) -> CompositeCrowd:
        return CompositeCrowd(
            [_x(ByQuery("x_search_ok.json")), _reddit(ByQuery("reddit_search_ok.json"))],
            parallel=parallel,
        )

    serial = crowd(False).collect(SYMBOLS, since=SINCE)
    parallel = crowd(True).collect(SYMBOLS, since=SINCE)
    assert canonical_json(parallel) == canonical_json(serial)
    assert [c.source.split(":")[0] for c in parallel[1]] == ["twitter-cli"] * len(SYMBOLS) + [
        "rdt-cli"
    ] * len(SYMBOLS)


def test_parallel_collectors_really_overlap() -> None:
    barrier = threading.Barrier(2, timeout=10)
    met: list[bool] = []

    class Meeting(ByQuery):
        def __call__(self, argv: Sequence[str], timeout: float) -> tuple[int, str, str]:
            if not met:
                barrier.wait()  # X's and Reddit's first searches must be in flight together
                met.append(True)
            return super().__call__(argv, timeout)

    items, calls = CompositeCrowd(
        [_x(Meeting("x_search_ok.json")), _reddit(Meeting("reddit_search_ok.json"))],
        parallel=True,
    ).collect(["NVDAUSDT"], since=SINCE)
    assert met
    assert all(c.health is not SourceHealth.ERROR for c in calls)
    assert items


def test_a_collector_that_raises_is_recorded_and_the_other_still_answers() -> None:
    class Broken(XCollector):
        def collect(
            self, symbols: Sequence[str], *, since: datetime
        ) -> tuple[tuple[TextItem, ...], tuple[SourceCall, ...]]:
            raise RuntimeError("a defect")

    _, calls = CompositeCrowd(
        [
            Broken(runner=ByQuery("x_search_ok.json"), clock=ManualClock(NOW)),
            _reddit(ByQuery("reddit_search_ok.json")),
        ],
        parallel=True,
    ).collect(["NVDAUSDT"], since=SINCE)
    assert calls[0].health is SourceHealth.ERROR
    assert "RuntimeError: a defect" in (calls[0].error or "")
    assert calls[1].source == "rdt-cli:search:NVDAUSDT"
