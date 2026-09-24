"""The simulated venue: Demo ask/bid fills, the 6 bps fee, netting, and stops on the Demo mark."""

from datetime import timedelta
from decimal import Decimal

import pytest

from execution.fakes import FakeMarket
from helpers import make_intent, mint_for_test
from sentiment_agent.clock import ManualClock
from sentiment_agent.execution.bgc import (
    HOLD_ONE_WAY,
    TransportRefusedError,
    VenueReadError,
    build_place_args,
    would_send,
)
from sentiment_agent.execution.simulated import DEMO_TAKER_FEE, SimulatedVenue
from sentiment_agent.types import (
    Fill,
    FillVenue,
    OrderPurpose,
    Side,
    VenueAck,
    VenueOrder,
    VenueOrderStatus,
    VenueRejection,
    VenueUnknown,
)


@pytest.fixture
def market(clock: ManualClock) -> FakeMarket:
    m = FakeMarket()
    m.set("NVDAUSDT", bid="222.82", ask="222.88", mark="222.84", at=clock.now())
    m.set("BTCUSDT", bid="112340.0", ask="112350.0", mark="112345.0", at=clock.now())
    return m


@pytest.fixture
def venue(market: FakeMarket, clock: ManualClock) -> SimulatedVenue:
    return SimulatedVenue(market=market, clock=clock, starting_equity=Decimal("10000"))


def all_fills(venue: SimulatedVenue, clock: ManualClock) -> list[Fill]:
    return venue.fills(since=clock.now() - timedelta(days=1), until=clock.now())


def all_orders(venue: SimulatedVenue, clock: ManualClock) -> list[VenueOrder]:
    return venue.history(since=clock.now() - timedelta(days=1), until=clock.now())


def test_the_default_fee_is_the_demo_taker_fee() -> None:
    assert Decimal("0.0006") == DEMO_TAKER_FEE


def test_a_buy_fills_at_the_demo_ask_and_pays_6_bps(
    venue: SimulatedVenue, clock: ManualClock
) -> None:
    ack = venue.place(mint_for_test(make_intent(qty="2.00")))
    assert isinstance(ack, VenueAck)
    fills = all_fills(venue, clock)
    assert len(fills) == 1
    fill = fills[0]
    assert fill.exec_price == Decimal("222.88")
    assert fill.exec_qty == Decimal("2.00")
    assert fill.exec_value == Decimal("445.76")
    assert fill.fee_paid == Decimal("445.76") * Decimal("0.0006")
    assert fill.trade_scope == "taker"
    assert fill.trade_side == "open"
    assert fill.venue is FillVenue.SIMULATED
    assert fill.venue_order_id == ack.venue_order_id
    order = venue.order(client_oid=make_intent(qty="2.00").client_oid)
    assert order is not None
    assert order.status is VenueOrderStatus.FILLED
    assert order.avg_price == Decimal("222.88")
    assert [p.qty for p in venue.positions()] == [Decimal("2.00")]


def test_a_sell_fills_at_the_demo_bid(venue: SimulatedVenue, clock: ManualClock) -> None:
    intent = make_intent(symbol="BTCUSDT", side=Side.SELL, qty="0.01", price="112345.0")
    venue.place(mint_for_test(intent))
    fill = all_fills(venue, clock)[0]
    assert fill.exec_price == Decimal("112340.0")
    assert [p.qty for p in venue.positions()] == [Decimal("-0.01")]


def test_the_preset_stop_is_registered_with_the_position(venue: SimulatedVenue) -> None:
    intent = make_intent()
    venue.place(mint_for_test(intent))
    stops = venue.stop_orders()
    assert [(s.symbol, s.stop_price) for s in stops] == [("NVDAUSDT", intent.stop_loss_price)]


def test_a_long_stop_triggers_on_the_demo_mark_and_exits_at_the_bid(
    venue: SimulatedVenue, market: FakeMarket, clock: ManualClock
) -> None:
    intent = make_intent()  # stop at 222.84 * 0.96 = 213.9264
    venue.place(mint_for_test(intent))
    clock.advance(timedelta(minutes=5))
    market.set("NVDAUSDT", bid="213.00", ask="213.10", mark="213.93", at=clock.now())
    assert venue.poll() == []  # mark 213.93 is above the stop
    market.set("NVDAUSDT", bid="213.80", ask="213.90", mark="213.9264", at=clock.now())
    fills = venue.poll()
    assert len(fills) == 1
    exit_fill = fills[0]
    assert exit_fill.side is Side.SELL
    assert exit_fill.exec_price == Decimal("213.80")
    assert exit_fill.client_oid is None
    assert exit_fill.trade_side == "close"
    assert exit_fill.exec_pnl == (Decimal("213.80") - Decimal("222.88")) * Decimal("1.00")
    assert venue.positions() == []
    assert venue.stop_orders() == []
    order = venue.order(client_oid=intent.client_oid)
    assert order is not None
    stop_orders = [o for o in all_orders(venue, clock) if o.client_oid is None]
    assert stop_orders[0].delegate_type == "position_stop_loss_market"


def test_a_short_stop_triggers_when_the_mark_rises_to_it(
    venue: SimulatedVenue, market: FakeMarket, clock: ManualClock
) -> None:
    intent = make_intent(symbol="BTCUSDT", side=Side.SELL, qty="0.01", price="112345.0")
    venue.place(mint_for_test(intent))
    assert intent.stop_loss_price is not None
    below = intent.stop_loss_price - 1
    market.set("BTCUSDT", bid=str(below - 5), ask=str(below + 5), mark=str(below), at=clock.now())
    assert venue.poll() == []
    stop = intent.stop_loss_price
    market.set("BTCUSDT", bid=str(stop), ask=str(stop + 10), mark=str(stop), at=clock.now())
    fills = venue.poll()
    assert fills[0].side is Side.BUY
    assert fills[0].exec_price == stop + 10


def test_reduce_only_without_a_position_is_rejected(venue: SimulatedVenue) -> None:
    close = make_intent(side=Side.SELL, purpose=OrderPurpose.CLOSE)
    outcome = venue.place(mint_for_test(close))
    assert isinstance(outcome, VenueRejection)
    assert "no position" in outcome.message
    assert venue.positions() == []


def test_an_oversized_reduce_only_order_closes_the_position_and_no_more(
    venue: SimulatedVenue,
) -> None:
    venue.place(mint_for_test(make_intent(qty="1.00")))
    close = make_intent(side=Side.SELL, qty="3.00", purpose=OrderPurpose.CLOSE)
    venue.place(mint_for_test(close))
    assert venue.positions() == []
    order = venue.order(client_oid=close.client_oid)
    assert order is not None
    assert order.cum_exec_qty == Decimal("1.00")


def test_an_order_across_zero_closes_then_opens(venue: SimulatedVenue, clock: ManualClock) -> None:
    venue.place(mint_for_test(make_intent(qty="1.00")))
    flip = make_intent(side=Side.SELL, qty="3.00", purpose=OrderPurpose.OPEN, price="222.84")
    venue.place(mint_for_test(flip))
    fills = [f for f in all_fills(venue, clock) if f.client_oid == flip.client_oid]
    assert [(f.trade_side, f.exec_qty) for f in fills] == [
        ("close", Decimal("1.00")),
        ("open", Decimal("2.00")),
    ]
    assert fills[0].exec_pnl == (Decimal("222.82") - Decimal("222.88")) * Decimal("1.00")
    assert [p.qty for p in venue.positions()] == [Decimal("-2.00")]


def test_a_known_client_oid_is_refused_as_the_venue_would(
    venue: SimulatedVenue, clock: ManualClock
) -> None:
    intent = make_intent()
    venue.place(mint_for_test(intent))
    again = venue.place(mint_for_test(intent))
    assert isinstance(again, VenueUnknown)
    assert "already knows" in again.reason
    assert len(all_fills(venue, clock)) == 1


def test_no_quote_is_a_retryable_rejection(venue: SimulatedVenue) -> None:
    outcome = venue.place(mint_for_test(make_intent(symbol="TSLAUSDT")))
    assert isinstance(outcome, VenueRejection)
    assert outcome.retryable


def test_equity_is_realised_minus_fees_plus_unrealised_at_the_demo_mark(
    venue: SimulatedVenue, market: FakeMarket, clock: ManualClock
) -> None:
    venue.place(mint_for_test(make_intent(qty="1.00")))
    fee = Decimal("222.88") * Decimal("0.0006")
    market.set("NVDAUSDT", bid="229.90", ask="230.10", mark="230.00", at=clock.now())
    account = venue.account()
    assert account.equity_usdt == Decimal("10000") - fee + (Decimal("230.00") - Decimal("222.88"))
    assert venue.fees_paid == fee


def test_equity_without_a_mark_is_an_error_not_a_guess(
    venue: SimulatedVenue, market: FakeMarket
) -> None:
    venue.place(mint_for_test(make_intent()))
    del market.demo["NVDAUSDT"]
    with pytest.raises(VenueReadError):
        venue.account()


def test_place_stop_replaces_the_symbols_stop_and_cancel_removes_it(
    venue: SimulatedVenue,
) -> None:
    intent = make_intent()
    venue.place(mint_for_test(intent))
    old = venue.stop_orders()[0]
    placed = venue.place_stop(
        symbol="NVDAUSDT",
        pos_side="long",
        qty=Decimal("1.00"),
        stop_price=Decimal("214.00"),
        client_oid=intent.client_oid,
    )
    assert placed.action == "placed"
    stops = venue.stop_orders()
    assert [(s.venue_id, s.stop_price) for s in stops] == [(placed.venue_id, Decimal("214.00"))]
    assert placed.venue_id != old.venue_id
    with pytest.raises(TransportRefusedError):
        venue.cancel_stop(symbol="NVDAUSDT", venue_id=str(old.venue_id))
    assert placed.venue_id is not None
    cancelled = venue.cancel_stop(symbol="NVDAUSDT", venue_id=placed.venue_id)
    assert cancelled.action == "cancelled"
    assert venue.stop_orders() == []
    with pytest.raises(TransportRefusedError, match="short"):
        venue.place_stop(
            symbol="NVDAUSDT",
            pos_side="short",
            qty=Decimal("1.00"),
            stop_price=Decimal("230"),
            client_oid=intent.client_oid,
        )


def test_the_preview_is_what_agent_hub_would_send(venue: SimulatedVenue) -> None:
    intent = make_intent()
    preview = venue.preview(intent)
    assert preview.would_send == would_send(intent, hold_mode=HOLD_ONE_WAY)
    assert preview.argv == tuple(build_place_args(intent, hold_mode=HOLD_ONE_WAY, dry_run=True))
    assert venue.positions() == []  # a preview sends nothing


def test_a_tampered_approval_is_refused(venue: SimulatedVenue) -> None:
    approved = mint_for_test(make_intent())
    object.__setattr__(approved, "_intent", make_intent(qty="40.00"))
    with pytest.raises(TransportRefusedError):
        venue.place(approved)
    assert venue.positions() == []


def test_construction_is_checked(market: FakeMarket, clock: ManualClock) -> None:
    with pytest.raises(ValueError, match="positive"):
        SimulatedVenue(market=market, clock=clock, starting_equity=Decimal(0))
    with pytest.raises(ValueError, match="negative"):
        SimulatedVenue(
            market=market, clock=clock, starting_equity=Decimal(1), fee_rate=Decimal("-0.1")
        )
