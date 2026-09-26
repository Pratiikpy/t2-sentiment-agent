"""The book from fills: positions, realized and unrealized P&L, fees, closed trades, exit reasons,
day-open and peak equity, rebalance counters, stops, and independence from arrival order.

Every expected number below is worked by hand in the test's comment, and a seeded sweep checks the
book against an oracle that does not use average cost at all (cash flows plus marked positions)."""

import random
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal, localcontext

import pytest

from book.factories import FEE_RATE, START, T0, H, d, make_fill, make_mark
from sentiment_agent.book.book import (
    EXIT_FLIP,
    EXIT_LIQUIDATION,
    EXIT_MODEL_CLOSE,
    EXIT_PROTECTIVE,
    EXIT_STOP,
    EXIT_TAKE_PROFIT,
    EXIT_VENUE_INITIATED,
    FILL_CLOCK_TOLERANCE,
    BookBuilder,
    BookError,
    Cause,
    MissingMarkError,
    exit_reason,
    is_model_initiated,
    utc_midnight,
)
from sentiment_agent.policy import POLICY_V1
from sentiment_agent.types import (
    Activation,
    BookState,
    ClosedTrade,
    Fill,
    FillVenue,
    GuardId,
    OrderPurpose,
    PriceSource,
    ProtectiveReason,
    Side,
    StopSync,
)

NVDA = "NVDAUSDT"
BTC = "BTCUSDT"
BUY = Side.BUY
SELL = Side.SELL
OPEN = OrderPurpose.OPEN
INCREASE = OrderPurpose.INCREASE
REDUCE = OrderPurpose.REDUCE
CLOSE = OrderPurpose.CLOSE
PROTECTIVE = OrderPurpose.PROTECTIVE_EXIT
EPS = Decimal("1e-18")


def builder(start: Decimal = START) -> BookBuilder:
    return BookBuilder(starting_equity=start, policy=POLICY_V1)


def state(
    b: BookBuilder,
    at: datetime,
    marks: dict[str, Decimal] | None = None,
    source: PriceSource = PriceSource.DEMO,
) -> BookState:
    return b.state(at=at, marks=marks or {}, mark_source=source, activation=Activation.ACTIVE)


def model(
    b: BookBuilder, fill: Fill, purpose: OrderPurpose, decision: str = "d1"
) -> list[ClosedTrade]:
    """Apply a fill of a model-initiated order."""
    return b.apply_fill(fill, decision_id=decision, purpose=purpose)


# ------------------------------------------------------------------------------------------------
# Positions
# ------------------------------------------------------------------------------------------------


def test_a_long_open_books_quantity_entry_and_fee() -> None:
    b = builder()
    closed = model(b, make_fill(BUY, "2", "100", at=T0), OPEN)
    assert closed == []
    pos = b.positions()[NVDA]
    assert (pos.qty, pos.avg_entry, pos.fees_paid) == (d(2), d(100), d("0.12"))
    assert pos.realized_pnl == 0
    assert pos.opened_at == pos.last_increase_at == T0
    assert pos.last_decision_id == "d1"
    s = state(b, T0 + H, {NVDA: d(105)})
    # 10000 - 0.12 fee + 2 x (105 - 100) unrealized
    assert s.equity == d("10009.88")
    assert s.realized_total == 0
    assert s.fees_total == d("0.12")
    assert s.weight(NVDA) == pytest.approx(210 / 10009.88)


def test_a_short_open_is_negative_and_profits_when_the_mark_falls() -> None:
    b = builder()
    model(b, make_fill(SELL, "3", "50", at=T0), OPEN)
    pos = b.positions()[NVDA]
    assert pos.qty == d(-3)
    assert pos.avg_entry == d(50)
    s = state(b, T0 + H, {NVDA: d(48)})
    # 10000 - 0.09 fee + (-3) x (48 - 50) = +6
    assert s.equity == d("10005.91")
    assert s.weight(NVDA) < 0


def test_an_add_moves_the_average_entry_and_the_last_increase() -> None:
    b = builder()
    model(b, make_fill(BUY, "1", "100", at=T0), OPEN)
    model(b, make_fill(BUY, "3", "108", at=T0 + H), INCREASE, "d2")
    pos = b.positions()[NVDA]
    # (1 x 100 + 3 x 108) / 4 = 106
    assert pos.avg_entry == d(106)
    assert pos.qty == d(4)
    assert pos.opened_at == T0
    assert pos.last_increase_at == T0 + H
    assert pos.last_decision_id == "d2"


def test_a_partial_reduce_realizes_at_the_average_entry_and_keeps_it() -> None:
    b = builder()
    model(b, make_fill(BUY, "1", "100", at=T0), OPEN)
    model(b, make_fill(BUY, "3", "108", at=T0 + H), INCREASE)
    closed = model(b, make_fill(SELL, "1", "110", at=T0 + 2 * H), REDUCE)
    assert closed == []
    pos = b.positions()[NVDA]
    assert pos.qty == d(3)
    assert pos.avg_entry == d(106)
    assert pos.realized_pnl == d(4)  # (110 - 106) x 1
    assert pos.last_increase_at == T0 + H  # a reduction is not an increase
    # fees: 0.06 + 0.1944 + 0.066
    assert pos.fees_paid == d("0.3204")
    s = state(b, T0 + 2 * H, {NVDA: d(107)})
    # realized 4, fees 0.3204, unrealized 3 x (107 - 106) = 3
    assert s.equity == START + 4 - d("0.3204") + 3
    assert s.realized_total == d(4)


def test_a_close_ends_the_trade_with_both_legs_fees() -> None:
    b = builder()
    model(b, make_fill(BUY, "1", "100", at=T0), OPEN, "d1")
    model(b, make_fill(BUY, "3", "108", at=T0 + H), INCREASE, "d2")
    model(b, make_fill(SELL, "1", "110", at=T0 + 2 * H), REDUCE, "d3")
    closed = model(b, make_fill(SELL, "3", "104", at=T0 + 3 * H), CLOSE, "d3")
    assert len(closed) == 1
    trade = closed[0]
    assert b.positions() == {}
    assert trade.symbol == NVDA
    assert trade.direction == 1
    assert trade.opened_at == T0
    assert trade.closed_at == T0 + 3 * H
    assert trade.entry_avg == d(106)  # (100 + 324) / 4
    assert trade.exit_avg == d("105.5")  # (110 + 312) / 4
    assert trade.max_abs_qty == d(4)
    assert trade.gross_pnl == d(-2)  # (110 - 106) x 1 + (104 - 106) x 3
    # 0.06 + 0.1944 + 0.066 + 0.1872: every fill of the trade, entry and exit
    assert trade.fees == d("0.5076")
    assert trade.net_pnl == d("-2.5076")
    assert trade.decision_ids == ("d1", "d2", "d3")
    assert trade.exit_reason == EXIT_MODEL_CLOSE
    assert b.closed_trades() == (trade,)
    s = state(b, T0 + 3 * H)
    assert s.equity == START + trade.net_pnl
    assert s.consecutive_losses == 1


def test_a_short_round_trip_realizes_the_price_fall() -> None:
    b = builder()
    model(b, make_fill(SELL, "1", "100", at=T0, fee="0"), OPEN)
    (trade,) = model(b, make_fill(BUY, "1", "90", at=T0 + H, fee="0"), CLOSE)
    assert trade.direction == -1
    assert trade.gross_pnl == d(10)
    assert trade.net_pnl == d(10)


def test_one_fill_that_crosses_zero_closes_one_trade_and_opens_the_other_side() -> None:
    b = builder()
    model(b, make_fill(BUY, "2", "100", at=T0), OPEN, "d1")
    # sell 5 @ 110, fee 0.33: 2 close the long, 3 open a short (Nautilus flip_position)
    (trade,) = model(b, make_fill(SELL, "5", "110", at=T0 + H), OPEN, "d2")
    assert trade.exit_reason == EXIT_FLIP
    assert trade.gross_pnl == d(20)
    assert trade.fees == d("0.12") + d("0.132")  # entry fee + 2/5 of the flip fee
    assert trade.net_pnl == d("19.748")
    assert trade.decision_ids == ("d1", "d2")
    pos = b.positions()[NVDA]
    assert pos.qty == d(-3)
    assert pos.avg_entry == d(110)
    assert pos.fees_paid == d("0.198")  # the other 3/5
    assert pos.realized_pnl == 0
    assert pos.opened_at == pos.last_increase_at == T0 + H
    assert pos.last_decision_id == "d2"
    s = state(b, T0 + H, {NVDA: d(108)})
    assert s.fees_total == d("0.45")  # nothing lost or double counted in the split
    assert s.equity == START + 20 - d("0.45") + 6


def test_a_flip_sent_as_close_then_open_is_a_model_close_and_a_new_trade() -> None:
    b = builder()
    model(b, make_fill(BUY, "2", "100", at=T0), OPEN)
    at = T0 + H
    (trade,) = model(b, make_fill(SELL, "2", "110", at=at, trade_side="close"), CLOSE, "d2")
    assert trade.exit_reason == EXIT_MODEL_CLOSE
    assert model(b, make_fill(SELL, "3", "110", at=at, trade_side="open"), OPEN, "d2") == []
    assert b.positions()[NVDA].qty == d(-3)


def test_gross_pnl_equals_exit_value_minus_entry_value_after_adds_and_reduces() -> None:
    b = builder()
    model(b, make_fill(BUY, "1", "100", at=T0, fee="0"), OPEN)
    model(b, make_fill(BUY, "1", "110", at=T0 + H, fee="0"), INCREASE)  # avg 105
    model(b, make_fill(SELL, "1", "120", at=T0 + 2 * H, fee="0"), REDUCE)  # +15
    model(b, make_fill(BUY, "1", "90", at=T0 + 3 * H, fee="0"), INCREASE)  # avg 97.5
    (trade,) = model(b, make_fill(SELL, "2", "100", at=T0 + 4 * H, fee="0"), CLOSE)  # +5
    assert trade.gross_pnl == d(20)
    assert trade.entry_avg == d(100)
    assert abs(trade.exit_avg * 3 - d(320)) < EPS
    assert abs((trade.exit_avg - trade.entry_avg) * 3 - trade.gross_pnl) < EPS


def test_a_realized_loss_and_an_unrealized_gain_are_kept_apart() -> None:
    b = builder()
    model(b, make_fill(BUY, "4", "100", at=T0, fee="0.24"), OPEN)
    model(b, make_fill(SELL, "1", "95", at=T0 + H, fee="0.057"), REDUCE)
    s = state(b, T0 + H, {NVDA: d(103)})
    assert s.realized_total == d(-5)  # (95 - 100) x 1
    unrealized = s.equity - (START + s.realized_total - s.fees_total)
    assert unrealized == d(9)  # 3 x (103 - 100)
    assert b.unrealized({NVDA: d(103)}) == {NVDA: d(9)}
    assert b.realized_equity() == START - 5 - d("0.297")


# ------------------------------------------------------------------------------------------------
# Exit reasons
# ------------------------------------------------------------------------------------------------

_REASONS: list[tuple[str | None, OrderPurpose | None, str | None, Cause | None, str]] = [
    ("d1", CLOSE, None, None, EXIT_MODEL_CLOSE),
    ("d1", REDUCE, None, None, EXIT_MODEL_CLOSE),
    ("d1", CLOSE, "market", None, EXIT_MODEL_CLOSE),
    (None, None, "position_stop_loss_market", None, EXIT_STOP),
    (None, None, "stop_loss_market", None, EXIT_STOP),
    (None, None, "move_stop_market", None, EXIT_STOP),
    (None, None, "POSITION_STOP_LOSS_MARKET", None, EXIT_STOP),
    (None, None, None, ProtectiveReason.STOP_FILLED, EXIT_STOP),
    (None, None, "position_stop_profit_market", None, EXIT_TAKE_PROFIT),
    (None, None, "liquidation", None, EXIT_LIQUIDATION),
    (None, None, "plan_market", None, "venue_order:plan_market"),
    (None, None, None, None, EXIT_VENUE_INITIATED),
    (None, CLOSE, None, ProtectiveReason.DAILY_KILL, "protective_exit:daily_kill"),
    (None, CLOSE, None, None, EXIT_PROTECTIVE),
    (None, PROTECTIVE, None, ProtectiveReason.LLM_OUTAGE, "protective_exit:llm_outage"),
    ("d1", PROTECTIVE, None, GuardId.G2_WEEKEND_FREEZE, "protective_exit:G2_weekend_freeze"),
]


@pytest.mark.parametrize(("decision", "purpose", "delegate", "cause", "expected"), _REASONS)
def test_exit_reason_names_what_closed_the_trade(
    decision: str | None,
    purpose: OrderPurpose | None,
    delegate: str | None,
    cause: Cause | None,
    expected: str,
) -> None:
    b = builder()
    model(b, make_fill(BUY, "1", "100", at=T0), OPEN)
    (trade,) = b.apply_fill(
        make_fill(SELL, "1", "96", at=T0 + H),
        decision_id=decision,
        purpose=purpose,
        delegate_type=delegate,
        cause=cause,
    )
    assert trade.exit_reason == expected


def test_a_stop_is_a_stop_even_when_it_would_otherwise_read_as_a_flip() -> None:
    reason = exit_reason(
        decision_id=None,
        purpose=None,
        delegate_type="position_stop_loss_market",
        cause=None,
        flipped=True,
    )
    assert reason == EXIT_STOP
    assert (
        exit_reason(decision_id="d", purpose=OPEN, delegate_type=None, cause=None, flipped=True)
        == EXIT_FLIP
    )


def test_model_initiated_means_a_decision_and_no_forced_exit() -> None:
    assert is_model_initiated("d1", OPEN)
    assert is_model_initiated("d1", CLOSE)
    assert not is_model_initiated("d1", PROTECTIVE)
    assert not is_model_initiated(None, CLOSE)
    assert not is_model_initiated("d1", None)


# ------------------------------------------------------------------------------------------------
# Day-open equity, peak, fees today, rebalances, losing streak
# ------------------------------------------------------------------------------------------------


def test_day_open_before_any_midnight_is_the_starting_equity() -> None:
    b = builder()
    model(b, make_fill(BUY, "1", "100", at=T0), OPEN)
    assert state(b, T0 + H, {NVDA: d(90)}).day_open_equity == START


def test_day_open_after_a_flat_midnight_is_the_exact_realized_equity() -> None:
    b = builder()
    model(b, make_fill(BUY, "1", "100", at=T0, fee="0.06"), OPEN)
    model(b, make_fill(SELL, "1", "97", at=T0 + H, fee="0.0582"), CLOSE)
    next_day = utc_midnight(T0) + timedelta(days=1, hours=2)
    model(b, make_fill(BUY, "1", "98", at=next_day - H, fee="0"), OPEN)  # opened after midnight
    s = state(b, next_day, {NVDA: d(80)})
    assert s.day_open_equity == START - 3 - d("0.1182")
    assert s.day_return < 0


def test_day_open_is_the_midnight_mark_when_one_was_logged() -> None:
    b = builder()
    model(b, make_fill(BUY, "1", "100", at=T0), OPEN)
    midnight = utc_midnight(T0) + timedelta(days=1)
    b.record_mark(make_mark(midnight, "10012.5", mirror="10020"))
    b.record_mark(make_mark(midnight + H, "10030"))
    at = midnight + 2 * H
    assert state(b, at, {NVDA: d(101)}).day_open_equity == d("10012.5")
    assert state(b, at, {NVDA: d(101)}, PriceSource.LIVE).day_open_equity == d(10020)


def test_day_open_falls_back_to_the_first_mark_of_the_day_then_to_now() -> None:
    b = builder()
    model(b, make_fill(BUY, "1", "100", at=T0), OPEN)  # held across midnight
    midnight = utc_midnight(T0) + timedelta(days=1)
    now = midnight + 30 * timedelta(minutes=1)
    s = state(b, now, {NVDA: d(95)})
    assert s.day_open_equity == s.equity  # nothing better is known yet
    b.record_mark(make_mark(midnight + H, "9990"))
    assert state(b, now, {NVDA: d(95)}).day_open_equity == s.equity  # a later mark is not used
    assert state(b, midnight + 2 * H, {NVDA: d(95)}).day_open_equity == d(9990)
    # a midnight mark without a live mirror falls through for the LIVE book
    b.record_mark(make_mark(midnight, "10001"))
    assert state(b, midnight + 2 * H, {NVDA: d(95)}).day_open_equity == d(10001)
    live = state(b, midnight + 2 * H, {NVDA: d(95)}, PriceSource.LIVE)
    assert live.day_open_equity == live.equity


def test_the_first_mark_logged_for_an_hour_is_kept() -> None:
    b = builder()
    midnight = utc_midnight(T0) + timedelta(days=1)
    b.record_mark(make_mark(midnight, "10001"))
    b.record_mark(make_mark(midnight, "12345"))
    assert b.marks() == (make_mark(midnight, "10001"),)
    assert state(b, midnight + H).day_open_equity == d(10001)


def test_peak_is_the_highest_logged_mark_up_to_now_or_the_equity_now() -> None:
    b = builder()
    model(b, make_fill(BUY, "10", "100", at=T0, fee="0"), OPEN)
    b.record_mark(make_mark(T0 + H, "10050"))
    b.record_mark(make_mark(T0 + 2 * H, "10020"))
    b.record_mark(make_mark(T0 + 4 * H, "10100"))  # after the state asked for below
    s = state(b, T0 + 2 * H + H / 2, {NVDA: d(101)})
    assert s.equity == d(10010)
    assert s.peak_equity == d(10050)
    assert s.drawdown == pytest.approx(10010 / 10050 - 1)
    high = state(b, T0 + 2 * H + H / 2, {NVDA: d(107)})
    assert high.peak_equity == high.equity == d(10070)
    assert state(b, T0 + 5 * H, {NVDA: d(101)}).peak_equity == d(10100)


def test_peak_never_falls_below_the_starting_equity() -> None:
    b = builder()
    model(b, make_fill(BUY, "10", "100", at=T0, fee="0"), OPEN)
    s = state(b, T0 + H, {NVDA: d(90)})
    assert s.peak_equity == START
    assert s.drawdown == pytest.approx(-0.01)


def test_fees_today_reset_at_midnight_and_fees_total_do_not() -> None:
    b = builder()
    model(b, make_fill(BUY, "1", "100", at=T0, fee="0.06"), OPEN)
    midnight = utc_midnight(T0) + timedelta(days=1)
    marks = {NVDA: d(100)}
    assert state(b, midnight - timedelta(seconds=1), marks).fees_today == d("0.06")
    assert state(b, midnight, marks).fees_today == 0
    model(b, make_fill(SELL, "1", "100", at=midnight + H, fee="0.06"), CLOSE)
    day_two = state(b, midnight + 2 * H)
    assert day_two.fees_today == d("0.06")
    assert day_two.fees_total == d("0.12")
    assert state(b, midnight + timedelta(days=1)).fees_today == 0


def test_rebalances_count_model_orders_per_symbol_and_reset_at_midnight() -> None:
    b = builder()
    # one model order filled in two parts counts once
    model(b, make_fill(BUY, "1", "100", at=T0, client_oid="A"), OPEN)
    model(b, make_fill(BUY, "1", "100", at=T0, client_oid="A"), OPEN)
    model(b, make_fill(SELL, "1", "101", at=T0 + H, client_oid="B"), REDUCE, "d2")
    # kernel exits, guard-forced exits and venue stops are not the model's rebalances
    b.apply_fill(
        make_fill(SELL, "0.5", "99", at=T0 + 2 * H, client_oid="C"),
        decision_id=None,
        purpose=CLOSE,
        cause=ProtectiveReason.DAILY_KILL,
    )
    b.apply_fill(
        make_fill(BUY, "1", "99", at=T0 + 2 * H, client_oid="E"),
        decision_id="d3",
        purpose=PROTECTIVE,
    )
    b.apply_fill(make_fill(SELL, "1", "98", at=T0 + 3 * H), decision_id=None, purpose=None)
    # another symbol, and a model order with no clientOid counted by its venue order id
    model(b, make_fill(BUY, "0.01", "60000", symbol=BTC, at=T0, venue_order_id="v1"), OPEN)
    model(b, make_fill(BUY, "0.01", "60000", symbol=BTC, at=T0, venue_order_id="v1"), OPEN)
    marks = {NVDA: d(100), BTC: d(60000)}
    assert state(b, T0 + 3 * H, marks).rebalances_today == {BTC: 1, NVDA: 2}
    midnight = utc_midnight(T0) + timedelta(days=1)
    assert state(b, midnight, marks).rebalances_today == {}
    model(b, make_fill(SELL, "0.01", "60100", symbol=BTC, at=midnight + H), CLOSE, "d4")
    assert state(b, midnight + H, marks).rebalances_today == {BTC: 1}


def test_losing_streak_counts_trailing_losses_across_symbols() -> None:
    b = builder()

    def round_trip(symbol: str, entry: str, exit_: str, hour: int) -> None:
        model(b, make_fill(BUY, "1", entry, symbol=symbol, at=T0 + hour * H), OPEN)
        model(b, make_fill(SELL, "1", exit_, symbol=symbol, at=T0 + hour * H + H / 2), CLOSE)

    round_trip(NVDA, "100", "99", 0)
    round_trip(BTC, "100", "98", 1)
    assert state(b, T0 + 2 * H).consecutive_losses == 2
    round_trip(NVDA, "100", "105", 2)
    assert state(b, T0 + 3 * H).consecutive_losses == 0
    round_trip(NVDA, "100", "90", 3)
    assert state(b, T0 + 4 * H).consecutive_losses == 1
    model(b, make_fill(BUY, "1", "100", at=T0 + 5 * H, fee="0"), OPEN)
    model(b, make_fill(SELL, "1", "100", at=T0 + 5 * H + H / 2, fee="0"), CLOSE)
    assert state(b, T0 + 6 * H).consecutive_losses == 0  # a scratch trade is not a loss


def test_the_last_loss_is_when_the_latest_losing_trade_closed() -> None:
    """Policy v2's losing-streak cool-off (run2-a5) is measured from here."""
    b = builder()

    def round_trip(entry: str, exit_: str, hour: int) -> None:
        model(b, make_fill(BUY, "1", entry, at=T0 + hour * H), OPEN)
        model(b, make_fill(SELL, "1", exit_, at=T0 + hour * H + H / 2), CLOSE)

    assert state(b, T0).last_loss_at is None
    round_trip("100", "99", 0)
    assert state(b, T0 + H).last_loss_at == T0 + H / 2
    round_trip("100", "98", 1)
    round_trip("100", "105", 2)  # a winner ends the streak but not the record of the last loss
    assert state(b, T0 + 3 * H).last_loss_at == T0 + H + H / 2


# ------------------------------------------------------------------------------------------------
# Arrival order, duplicates, refusals
# ------------------------------------------------------------------------------------------------


def test_a_late_stop_fill_is_booked_where_the_venue_executed_it() -> None:
    open_ = make_fill(BUY, "1", "100", at=T0)
    ours = make_fill(BUY, "1", "101", at=T0 + H / 6)  # the agent's order at 13:10
    stop = make_fill(SELL, "1", "96", at=T0 + H / 12)  # the stop fired at 13:05
    late = builder()
    model(late, open_, OPEN, "d1")
    model(late, ours, INCREASE, "d2")
    closed = late.apply_fill(
        stop, decision_id=None, purpose=None, delegate_type="position_stop_loss_market"
    )
    assert [t.exit_reason for t in closed] == [EXIT_STOP]
    in_order = builder()
    model(in_order, open_, OPEN, "d1")
    in_order.apply_fill(
        stop, decision_id=None, purpose=None, delegate_type="position_stop_loss_market"
    )
    model(in_order, ours, INCREASE, "d2")
    assert late.closed_trades() == in_order.closed_trades()
    at, marks = T0 + H, {NVDA: d(102)}
    assert state(late, at, marks) == state(in_order, at, marks)
    pos = late.positions()[NVDA]
    assert (pos.qty, pos.avg_entry, pos.opened_at) == (d(1), d(101), T0 + H / 6)


def test_the_book_does_not_depend_on_arrival_order() -> None:
    rng = random.Random(20260924)  # noqa: S311 - a reproducible shuffle, not a secret
    fills = [
        (make_fill(BUY, "2", "100", at=T0), OPEN, "d1"),
        (make_fill(BUY, "1", "104", at=T0 + H), INCREASE, "d2"),
        (make_fill(SELL, "2", "103", at=T0 + 2 * H), REDUCE, "d3"),
        (make_fill(SELL, "3", "99", at=T0 + 3 * H), OPEN, "d4"),
        (make_fill(BUY, "2", "97", at=T0 + 4 * H), CLOSE, "d5"),
        (make_fill(BUY, "0.01", "60000", symbol=BTC, at=T0 + H), OPEN, "d2"),
        (make_fill(SELL, "0.01", "60300", symbol=BTC, at=T0 + 3 * H), CLOSE, "d4"),
    ]
    reference = builder()
    for fill, purpose, decision in fills:
        model(reference, fill, purpose, decision)
    at, marks = T0 + 5 * H, {NVDA: d(98)}
    for _ in range(25):
        shuffled = fills[:]
        rng.shuffle(shuffled)
        b = builder()
        for fill, purpose, decision in shuffled:
            model(b, fill, purpose, decision)
        assert b.closed_trades() == reference.closed_trades()
        assert state(b, at, marks) == state(reference, at, marks)


def test_an_identical_duplicate_is_ignored_and_a_conflicting_one_refused() -> None:
    b = builder()
    fill = make_fill(BUY, "1", "100", at=T0)
    model(b, fill, OPEN)
    assert model(b, fill, OPEN) == []
    assert b.fill_count == 1
    assert b.positions()[NVDA].qty == d(1)
    conflicting = fill.model_copy(update={"exec_price": d(101), "exec_value": d(101)})
    with pytest.raises(BookError, match="different content"):
        model(b, conflicting, OPEN)


def test_fills_from_two_venues_never_mix() -> None:
    b = builder()
    model(b, make_fill(BUY, "1", "100", at=T0, venue=FillVenue.SIMULATED), OPEN)
    with pytest.raises(BookError, match="never mix"):
        model(b, make_fill(BUY, "1", "100", at=T0, venue=FillVenue.BITGET_DEMO), OPEN)
    assert b.venue is FillVenue.SIMULATED


def test_a_fee_in_another_coin_is_refused() -> None:
    with pytest.raises(BookError, match="fee"):
        model(builder(), make_fill(BUY, "1", "100", at=T0, fee_coin="BGB"), OPEN)
    assert model(builder(), make_fill(BUY, "1", "100", at=T0, fee_coin="usdt"), OPEN) == []


@pytest.mark.parametrize("start", [Decimal(0), Decimal(-1), Decimal("NaN"), Decimal("Infinity")])
def test_the_starting_equity_must_be_positive_and_finite(start: Decimal) -> None:
    with pytest.raises(BookError, match="starting equity"):
        builder(start)


def test_an_open_position_without_a_mark_has_no_equity() -> None:
    b = builder()
    model(b, make_fill(BUY, "1", "100", at=T0), OPEN)
    with pytest.raises(MissingMarkError, match=NVDA):
        state(b, T0 + H, {BTC: d(60000)})


@pytest.mark.parametrize("bad", [Decimal(0), Decimal(-5), Decimal("NaN"), Decimal("sNaN")])
def test_a_mark_must_be_a_positive_finite_price(bad: Decimal) -> None:
    b = builder()
    with pytest.raises(BookError, match="positive"):
        state(b, T0 + H, {BTC: bad})


def test_every_supplied_mark_is_carried_for_the_planner() -> None:
    b = builder()
    model(b, make_fill(BUY, "1", "100", at=T0), OPEN)
    s = state(b, T0 + H, {NVDA: d(101), BTC: d(60000)})
    assert s.marks == {NVDA: d(101), BTC: d(60000)}
    assert s.weight(BTC) == 0.0


def test_a_book_as_of_a_time_before_its_fills_is_refused() -> None:
    b = builder()
    model(b, make_fill(BUY, "1", "100", at=T0 + 2 * H), OPEN)
    with pytest.raises(BookError, match="after"):
        state(b, T0, {NVDA: d(100)})
    # a venue clock a little ahead of ours is ordinary
    s = state(b, T0 + 2 * H - FILL_CLOCK_TOLERANCE, {NVDA: d(100)})
    assert s.as_of == T0 + 2 * H - FILL_CLOCK_TOLERANCE


def test_naive_or_non_utc_times_are_refused() -> None:
    b = builder()
    with pytest.raises(BookError, match="UTC"):
        state(b, datetime(2026, 9, 23, 13, 0))  # noqa: DTZ001 - the point of the test
    with pytest.raises(BookError, match="UTC"):
        state(b, datetime(2026, 9, 23, 14, 0, tzinfo=timezone(timedelta(hours=1))))


def test_the_decimal_context_of_the_caller_does_not_change_the_book() -> None:
    fills = [
        make_fill(BUY, "3", "100.01", at=T0),
        make_fill(BUY, "7", "99.97", at=T0 + H),
        make_fill(SELL, "4", "101.13", at=T0 + 2 * H),
    ]
    reference = builder()
    for fill in fills:
        model(reference, fill, OPEN)
    with localcontext() as ctx:
        ctx.prec = 4
        b = builder()
        for fill in fills:
            model(b, fill, OPEN)
        coarse = state(b, T0 + 3 * H, {NVDA: d("100.5")})
    assert coarse == state(reference, T0 + 3 * H, {NVDA: d("100.5")})


# ------------------------------------------------------------------------------------------------
# Stops
# ------------------------------------------------------------------------------------------------


def sync(action: str, price: str | None, venue_id: str | None, symbol: str = NVDA) -> StopSync:
    return StopSync.model_validate(
        {
            "symbol": symbol,
            "action": action,
            "stop_price": None if price is None else Decimal(price),
            "venue_id": venue_id,
            "at": T0 + H,
        }
    )


def test_the_stop_follows_preset_sync_replace_and_cancel() -> None:
    b = builder()
    model(b, make_fill(BUY, "1", "100", at=T0), OPEN)
    b.apply_stop_sync(sync("preset", "96", None))
    assert (b.positions()[NVDA].stop_price, b.positions()[NVDA].stop_venue_id) == (d(96), None)
    b.apply_stop_sync(sync("verified", None, "s1"))  # the venue lists it; price unchanged
    assert (b.positions()[NVDA].stop_price, b.positions()[NVDA].stop_venue_id) == (d(96), "s1")
    b.apply_stop_sync(sync("replaced", "96.5", "s2"))
    b.apply_stop_sync(sync("cancelled", None, "s1"))  # the old stop, cancelled after
    assert (b.positions()[NVDA].stop_price, b.positions()[NVDA].stop_venue_id) == (
        d("96.5"),
        "s2",
    )
    b.apply_stop_sync(sync("cancelled", None, "s2"))
    assert b.positions()[NVDA].stop_price is None
    b.apply_stop_sync(sync("placed", "95", "s3"))
    assert b.positions()[NVDA].stop_venue_id == "s3"
    b.apply_stop_sync(sync("missing", "96", None))  # the manager could not place one
    assert (b.positions()[NVDA].stop_price, b.positions()[NVDA].stop_venue_id) == (None, None)


def test_a_stop_belongs_to_its_trade_and_never_carries_over() -> None:
    b = builder()
    model(b, make_fill(BUY, "1", "100", at=T0), OPEN)
    b.apply_stop_sync(sync("placed", "96", "s1"))
    model(b, make_fill(SELL, "3", "101", at=T0 + H), OPEN)  # flips to short 2
    pos = b.positions()[NVDA]
    assert pos.qty == d(-2)
    assert pos.stop_price is None
    b.apply_stop_sync(sync("placed", "105", "s2"))
    model(b, make_fill(BUY, "2", "100", at=T0 + 2 * H), CLOSE)
    assert b.positions() == {}
    model(b, make_fill(BUY, "1", "100", at=T0 + 3 * H), OPEN)
    assert b.positions()[NVDA].stop_price is None


def test_a_stop_sync_for_a_flat_symbol_changes_nothing() -> None:
    b = builder()
    b.apply_stop_sync(sync("placed", "96", "s1"))
    model(b, make_fill(BUY, "1", "100", at=T0), OPEN)
    assert b.positions()[NVDA].stop_price is None


def test_a_stop_at_a_non_positive_price_is_refused() -> None:
    b = builder()
    model(b, make_fill(BUY, "1", "100", at=T0), OPEN)
    with pytest.raises(BookError, match="stop price"):
        b.apply_stop_sync(sync("placed", "0", "s1"))


def test_a_venue_fill_keeps_the_last_model_decision_on_the_position() -> None:
    b = builder()
    model(b, make_fill(BUY, "2", "100", at=T0), OPEN, "d1")
    model(b, make_fill(BUY, "1", "100", at=T0 + H), INCREASE, "d2")
    b.apply_fill(make_fill(SELL, "1", "98", at=T0 + 2 * H), decision_id=None, purpose=None)
    assert b.positions()[NVDA].last_decision_id == "d2"


# ------------------------------------------------------------------------------------------------
# Oracle sweep: average cost against cash flows
# ------------------------------------------------------------------------------------------------


def test_the_book_agrees_with_a_cash_flow_oracle_on_random_fill_sequences() -> None:
    """Whatever the sequence, equity = start - fees + sum(-signed qty x price) + sum(qty x mark):
    an identity that knows nothing about average cost, flips or trade boundaries."""
    rng = random.Random(7)  # noqa: S311 - a reproducible sweep, not a secret
    symbols = (NVDA, BTC)
    for case in range(300):
        b = builder()
        cash = Decimal(0)
        fees = Decimal(0)
        held: dict[str, Decimal] = dict.fromkeys(symbols, Decimal(0))
        at = T0
        for _ in range(rng.randint(1, 25)):
            symbol = rng.choice(symbols)
            side = rng.choice((BUY, SELL))
            qty = Decimal(rng.randint(1, 500)) / 100
            price = Decimal(rng.randint(9000, 11000)) / 100
            fee = (qty * price * FEE_RATE).quantize(Decimal("0.00000001"))
            at += timedelta(minutes=rng.randint(0, 90))
            purpose = rng.choice((OPEN, INCREASE, REDUCE, CLOSE, PROTECTIVE, None))
            b.apply_fill(
                make_fill(side, qty, price, symbol=symbol, at=at, fee=fee),
                decision_id=rng.choice(("d1", "d2", None)),
                purpose=purpose,
            )
            signed = qty if side is BUY else -qty
            cash -= signed * price
            fees += fee
            held[symbol] += signed
        marks = {s: Decimal(rng.randint(9000, 11000)) / 100 for s in symbols}
        s = state(b, at, marks)
        oracle = START - fees + cash + sum(held[x] * marks[x] for x in symbols)
        assert abs(s.equity - oracle) < Decimal("1e-15"), case
        for symbol in symbols:
            qty = s.positions[symbol].qty if symbol in s.positions else Decimal(0)
            assert qty == held[symbol], case
        trades = b.closed_trades()
        open_realized = sum((p.realized_pnl for p in s.positions.values()), Decimal(0))
        open_fees = sum((p.fees_paid for p in s.positions.values()), Decimal(0))
        assert abs(sum(t.gross_pnl for t in trades) + open_realized - s.realized_total) < EPS
        assert abs(sum(t.fees for t in trades) + open_fees - s.fees_total) < EPS
        assert s.fees_total == fees
        for trade in trades:
            assert trade.net_pnl == trade.gross_pnl - trade.fees
            assert trade.opened_at <= trade.closed_at
            assert trade.max_abs_qty > 0
        assert [t.closed_at for t in trades] == sorted(t.closed_at for t in trades)


def test_the_utc_midnight_helper() -> None:
    assert utc_midnight(datetime(2026, 9, 23, 23, 59, 59, tzinfo=UTC)) == datetime(
        2026, 9, 23, tzinfo=UTC
    )
    with pytest.raises(BookError):
        utc_midnight(datetime(2026, 9, 23))  # noqa: DTZ001
