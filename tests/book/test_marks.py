"""The hourly mark: stamped on its hour or refused, Demo-marked equity, the live mirror, the
venue's equity passed through, and the point fed back into day-open and peak equity."""

from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from book.factories import START, T0, H, d, make_fill
from helpers import make_quote
from sentiment_agent.book.book import BookBuilder, BookError, utc_midnight
from sentiment_agent.book.marks import (
    MARK_MAX_LATENESS,
    MARK_QUOTE_WINDOW,
    MarkError,
    hour_floor,
    mark_point,
)
from sentiment_agent.policy import POLICY_V1
from sentiment_agent.types import Activation, OrderPurpose, PriceSource, Quote, Side

NVDA = "NVDAUSDT"
BTC = "BTCUSDT"
DEMO = PriceSource.DEMO
LIVE = PriceSource.LIVE


def quote(symbol: str, mark: str, *, source: PriceSource = DEMO, at: datetime = T0) -> Quote:
    return make_quote(
        symbol,
        source=source,
        last=mark,
        mark=mark,
        index=mark,
        bid=mark,
        ask=mark,
        at=at,
    )


def held_book() -> BookBuilder:
    """Long 2 NVDA @ 100 and short 0.01 BTC @ 60,000, fees 0.12 and 0.36."""
    b = BookBuilder(starting_equity=START, policy=POLICY_V1)
    b.apply_fill(
        make_fill(Side.BUY, "2", "100", at=T0 - H), decision_id="d1", purpose=OrderPurpose.OPEN
    )
    b.apply_fill(
        make_fill(Side.SELL, "0.01", "60000", symbol=BTC, at=T0 - H),
        decision_id="d1",
        purpose=OrderPurpose.OPEN,
    )
    return b


def test_hour_floor_drops_minutes_seconds_and_microseconds() -> None:
    assert hour_floor(datetime(2026, 9, 23, 13, 59, 59, 999_999, tzinfo=UTC)) == T0
    assert hour_floor(T0) == T0


def test_hour_floor_refuses_naive_and_non_utc_times() -> None:
    with pytest.raises(BookError, match="UTC"):
        hour_floor(datetime(2026, 9, 23, 13, 30))  # noqa: DTZ001 - the point of the test
    with pytest.raises(BookError, match="UTC"):
        hour_floor(datetime(2026, 9, 23, 15, 30, tzinfo=timezone(timedelta(hours=2))))


def test_a_mark_is_stamped_on_its_hour_with_demo_and_live_equity() -> None:
    b = held_book()
    at = T0 + timedelta(seconds=40)
    demo = {NVDA: quote(NVDA, "105", at=at), BTC: quote(BTC, "59000", at=at)}
    live = {
        NVDA: quote(NVDA, "106", source=LIVE, at=at),
        BTC: quote(BTC, "59500", source=LIVE, at=at),
    }
    point = mark_point(b, at=at, demo=demo, live=live, venue_equity=d("10015.5"))
    assert point.at == T0
    # realized equity 10000 - 0.12 - 0.36; Demo: +2 x 5 and -0.01 x -1000 = +10 + 10
    assert point.equity_book == d("10019.52")
    # live: +2 x 6 and -0.01 x -500 = +12 + 5
    assert point.equity_live_mirror == d("10016.52")
    assert point.equity_venue == d("10015.5")
    by_symbol = {p.symbol: p for p in point.positions}
    assert by_symbol[NVDA].qty == d(2)
    assert by_symbol[NVDA].demo_mark == d(105)
    assert by_symbol[NVDA].live_mark == d(106)
    assert by_symbol[NVDA].demo_index == d(105)
    assert by_symbol[NVDA].unrealized_demo == d(10)
    assert by_symbol[NVDA].unrealized_live == d(12)
    assert by_symbol[BTC].unrealized_demo == d(10)
    assert by_symbol[BTC].unrealized_live == d(5)
    assert point.gross_weight == pytest.approx((210 + 590) / 10019.52)
    assert point.net_weight == pytest.approx((210 - 590) / 10019.52)
    # the same number the book reports at the same marks
    marks = {NVDA: d(105), BTC: d(59000)}
    same = b.state(at=at, marks=marks, mark_source=DEMO, activation=Activation.ACTIVE)
    assert same.equity == point.equity_book
    assert same.gross_weight == pytest.approx(point.gross_weight)


def test_a_mark_later_than_the_window_after_its_hour_is_refused() -> None:
    b = held_book()
    ok = T0 + MARK_MAX_LATENESS
    demo = {NVDA: quote(NVDA, "100", at=ok), BTC: quote(BTC, "60000", at=ok)}
    assert mark_point(b, at=ok, demo=demo, live={}, venue_equity=None).at == T0
    late = ok + timedelta(seconds=1)
    demo = {NVDA: quote(NVDA, "100", at=late), BTC: quote(BTC, "60000", at=late)}
    with pytest.raises(MarkError, match="not at all"):
        mark_point(b, at=late, demo=demo, live={}, venue_equity=None)


def test_an_open_position_without_a_fresh_demo_mark_cannot_be_marked() -> None:
    b = held_book()
    with pytest.raises(MarkError, match=BTC):
        mark_point(b, at=T0, demo={NVDA: quote(NVDA, "100")}, live={}, venue_equity=None)
    stale = T0 - MARK_QUOTE_WINDOW - timedelta(seconds=1)
    demo = {NVDA: quote(NVDA, "100"), BTC: quote(BTC, "60000", at=stale)}
    with pytest.raises(MarkError, match=BTC):
        mark_point(b, at=T0, demo=demo, live={}, venue_equity=None)
    demo = {NVDA: quote(NVDA, "0"), BTC: quote(BTC, "60000")}
    with pytest.raises(MarkError, match=NVDA):
        mark_point(b, at=T0, demo=demo, live={}, venue_equity=None)


def test_a_mirror_with_a_hole_is_no_mirror() -> None:
    b = held_book()
    demo = {NVDA: quote(NVDA, "105"), BTC: quote(BTC, "59000")}
    stale = T0 - MARK_QUOTE_WINDOW - timedelta(seconds=1)
    for live in (
        {NVDA: quote(NVDA, "106", source=LIVE)},
        {NVDA: quote(NVDA, "106", source=LIVE), BTC: quote(BTC, "59500", source=LIVE, at=stale)},
    ):
        point = mark_point(b, at=T0, demo=demo, live=live, venue_equity=None)
        assert point.equity_live_mirror is None
        by_symbol = {p.symbol: p for p in point.positions}
        assert by_symbol[NVDA].live_mark == d(106)
        assert by_symbol[BTC].live_mark is None
        assert by_symbol[BTC].unrealized_live is None
        assert point.equity_book == d("10019.52")


def test_quotes_are_filed_under_their_own_symbol_and_environment() -> None:
    b = held_book()
    demo = {NVDA: quote(NVDA, "105"), BTC: quote(BTC, "59000")}
    wrong = {**demo, NVDA: quote(NVDA, "1", source=LIVE)}
    with pytest.raises(MarkError, match="live quote"):
        mark_point(b, at=T0, demo=wrong, live={}, venue_equity=None)
    with pytest.raises(MarkError, match="demo quote"):
        mark_point(b, at=T0, demo=demo, live={NVDA: quote(NVDA, "1")}, venue_equity=None)
    with pytest.raises(MarkError, match="holds a quote for"):
        mark_point(b, at=T0, demo={**demo, NVDA: quote(BTC, "1")}, live={}, venue_equity=None)


def test_a_flat_book_is_marked_at_its_realized_equity() -> None:
    b = BookBuilder(starting_equity=START, policy=POLICY_V1)
    b.apply_fill(
        make_fill(Side.BUY, "1", "100", at=T0 - H, fee="0.06"),
        decision_id="d1",
        purpose=OrderPurpose.OPEN,
    )
    b.apply_fill(
        make_fill(Side.SELL, "1", "103", at=T0 - H / 2, fee="0.0618"),
        decision_id="d2",
        purpose=OrderPurpose.CLOSE,
    )
    point = mark_point(b, at=T0, demo={}, live={}, venue_equity=None)
    assert point.equity_book == START + 3 - d("0.1218")
    assert point.equity_live_mirror == point.equity_book
    assert point.positions == ()
    assert (point.gross_weight, point.net_weight) == (0.0, 0.0)


def test_a_book_with_no_equity_left_carries_zero_weights() -> None:
    b = BookBuilder(starting_equity=d(100), policy=POLICY_V1)
    b.apply_fill(
        make_fill(Side.BUY, "10", "100", at=T0 - H, fee="0"),
        decision_id="d1",
        purpose=OrderPurpose.OPEN,
    )
    point = mark_point(b, at=T0, demo={NVDA: quote(NVDA, "80")}, live={}, venue_equity=None)
    assert point.equity_book == d(-100)
    assert (point.gross_weight, point.net_weight) == (0.0, 0.0)


def test_a_non_finite_venue_equity_is_refused() -> None:
    b = held_book()
    demo = {NVDA: quote(NVDA, "105"), BTC: quote(BTC, "59000")}
    with pytest.raises(MarkError, match="finite"):
        mark_point(b, at=T0, demo=demo, live={}, venue_equity=Decimal("NaN"))


def test_marking_does_not_record_the_point() -> None:
    b = held_book()
    demo = {NVDA: quote(NVDA, "105"), BTC: quote(BTC, "59000")}
    mark_point(b, at=T0, demo=demo, live={}, venue_equity=None)
    assert b.marks() == ()


def test_a_mark_as_of_a_time_before_the_books_fills_is_refused() -> None:
    b = held_book()
    b.apply_fill(
        make_fill(Side.BUY, "1", "100", at=T0 + 2 * H),
        decision_id="d2",
        purpose=OrderPurpose.INCREASE,
    )
    demo = {NVDA: quote(NVDA, "105"), BTC: quote(BTC, "59000")}
    with pytest.raises(BookError, match="after"):
        mark_point(b, at=T0, demo=demo, live={}, venue_equity=None)


def test_the_midnight_mark_becomes_the_next_days_open_and_counts_toward_the_peak() -> None:
    b = held_book()
    midnight = utc_midnight(T0) + timedelta(days=1)
    demo = {NVDA: quote(NVDA, "110", at=midnight), BTC: quote(BTC, "58000", at=midnight)}
    point = mark_point(b, at=midnight, demo=demo, live={}, venue_equity=None)
    b.record_mark(point)
    later = midnight + 3 * H
    s = b.state(
        at=later,
        marks={NVDA: d(100), BTC: d(60000)},
        mark_source=DEMO,
        activation=Activation.ACTIVE,
    )
    # at midnight: realized equity 9999.52, +2 x 10, -0.01 x -2000 = +40
    assert point.equity_book == d("10039.52")
    assert s.day_open_equity == point.equity_book
    assert s.peak_equity == point.equity_book
    assert s.equity == d("9999.52")
    assert s.day_return == pytest.approx(9999.52 / 10039.52 - 1)
