"""StopManager: exactly one full-position stop per open position, 4% from the average entry."""

import re
from datetime import datetime
from decimal import Decimal

import pytest

from execution.fakes import FakeMarket
from helpers import T0, empty_book, make_intent, mint_for_test
from sentiment_agent.clock import ManualClock
from sentiment_agent.execution.bgc import VenueWriteError
from sentiment_agent.execution.simulated import SimulatedVenue
from sentiment_agent.execution.stops import StopManager, desired_stop_price, stop_client_oid
from sentiment_agent.policy import POLICY_V1
from sentiment_agent.types import (
    BookState,
    Category,
    InstrumentSpec,
    Position,
    PriceSource,
    StopSync,
    VenuePosition,
    VenueStopOrder,
)

OID = re.compile(r"^sa[0-9a-f]{30}$")


def spec(symbol: str = "NVDAUSDT", step: str = "0.01") -> InstrumentSpec:
    return InstrumentSpec(
        symbol=symbol,
        category=Category.USDT_FUTURES,
        source=PriceSource.DEMO,
        base_coin=symbol.removesuffix("USDT"),
        quote_coin="USDT",
        status="online",
        min_order_qty=Decimal("0.01"),
        qty_step=Decimal("0.01"),
        price_step=Decimal(step),
        min_order_amount=Decimal("5"),
        max_market_order_qty=Decimal("60"),
        max_order_qty=None,
        taker_fee_rate=Decimal("0.0006"),
        maker_fee_rate=Decimal("0.0002"),
        max_leverage=25,
        fund_interval_hours=8,
        fetched_at=T0,
    )


def position(symbol: str = "NVDAUSDT", qty: str = "1.00", avg: str = "222.88") -> Position:
    return Position(
        symbol=symbol,
        qty=Decimal(qty),
        avg_entry=Decimal(avg),
        opened_at=T0,
        last_increase_at=T0,
        realized_pnl=Decimal(0),
        fees_paid=Decimal(0),
        stop_price=None,
        stop_venue_id=None,
        last_decision_id="decision-test",
    )


def book(*positions: Position) -> BookState:
    return empty_book().model_copy(
        update={
            "positions": {p.symbol: p for p in positions},
            "marks": {p.symbol: p.avg_entry for p in positions},
        }
    )


class StopBook(SimulatedVenue):
    """A venue whose stop orders the test controls, including several per symbol."""

    def __init__(
        self,
        clock: ManualClock,
        *,
        refuse_second: bool = False,
        fail_place: bool = False,
        timeout_place: bool = False,
        fail_cancel: bool = False,
    ) -> None:
        super().__init__(market=FakeMarket(), clock=clock, starting_equity=Decimal(1))
        self.now = clock.now
        self.refuse_second = refuse_second
        self.fail_place = fail_place
        self.timeout_place = timeout_place
        self.fail_cancel = fail_cancel
        self.book_stops: dict[str, VenueStopOrder] = {}
        self.calls: list[tuple[str, ...]] = []
        self._next = 0

    def seed(self, symbol: str, price: str | None, venue_id: str | None = None) -> VenueStopOrder:
        self._next += 1
        vid = venue_id or f"seed-{self._next}"
        stop = VenueStopOrder(
            symbol=symbol,
            venue_id=vid,
            stop_price=Decimal(price) if price is not None else None,
            blob=None,
        )
        self.book_stops[vid] = stop
        return stop

    def listing(self) -> list[VenueStopOrder]:
        return list(self.book_stops.values())

    def place_stop(
        self, *, symbol: str, pos_side: str, qty: Decimal, stop_price: Decimal, client_oid: str
    ) -> StopSync:
        self.calls.append(("place", symbol, pos_side, str(qty), str(stop_price), client_oid))
        if self.timeout_place:
            raise VenueWriteError("stop placement timed out", outcome_unknown=True)
        exists = any(s.symbol == symbol for s in self.book_stops.values())
        if self.fail_place or (self.refuse_second and exists):
            raise VenueWriteError("stop placement refused")
        self._next += 1
        vid = f"stop-{self._next}"
        self.book_stops[vid] = VenueStopOrder(
            symbol=symbol, venue_id=vid, stop_price=stop_price, blob=None
        )
        return StopSync(
            symbol=symbol, action="placed", stop_price=stop_price, venue_id=vid, at=self.now()
        )

    def cancel_stop(self, *, symbol: str, venue_id: str) -> StopSync:
        self.calls.append(("cancel", symbol, venue_id))
        if self.fail_cancel:
            raise VenueWriteError("cancel refused")
        del self.book_stops[venue_id]
        return StopSync(
            symbol=symbol, action="cancelled", stop_price=None, venue_id=venue_id, at=self.now()
        )


def held(b: BookState) -> list[VenuePosition]:
    """The venue's read of the same positions the ledger holds (the two agree)."""
    return [
        VenuePosition(symbol=p.symbol, qty=p.qty, avg_price=p.avg_entry, blob=None)
        for p in b.positions.values()
        if not p.is_flat
    ]


def manager(venue: SimulatedVenue, clock: ManualClock, **specs: InstrumentSpec) -> StopManager:
    return StopManager(
        transport=venue, policy=POLICY_V1, clock=clock, specs=specs or {"NVDAUSDT": spec()}
    )


def actions(results: list[StopSync]) -> list[str]:
    return [r.action for r in results]


# --- the price ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("qty", "avg", "expected"),
    [
        ("1", "222.88", "213.97"),  # 213.9648 rounded up, towards the entry
        ("1", "100", "96.00"),
        ("-1", "100", "104.00"),
        ("-1", "100.003", "104.00"),  # 104.00312 rounded down, towards the entry
        ("-2", "222.88", "231.79"),  # 231.7952 rounded down
    ],
)
def test_desired_stop_is_4_percent_rounded_towards_the_entry(
    qty: str, avg: str, expected: str
) -> None:
    pos = position(qty=qty, avg=avg)
    price = desired_stop_price(pos, stop_loss_pct=POLICY_V1.stop_loss_pct, spec=spec())
    assert price == Decimal(expected)
    loss = abs(price / pos.avg_entry - 1)
    assert loss <= Decimal("0.04")


def test_without_a_spec_the_price_is_exact() -> None:
    price = desired_stop_price(position(avg="222.88"), stop_loss_pct=0.04, spec=None)
    assert price == Decimal("222.88") * Decimal("0.96")


def test_stop_client_oid_is_deterministic_and_well_formed() -> None:
    pos = position()
    a = stop_client_oid("NVDAUSDT", pos, Decimal("213.97"), attempt="place")
    assert OID.fullmatch(a)
    assert a == stop_client_oid("NVDAUSDT", pos, Decimal("213.97"), attempt="place")
    assert a != stop_client_oid("NVDAUSDT", pos, Decimal("213.97"), attempt="replace")
    assert a != stop_client_oid("NVDAUSDT", pos, Decimal("213.98"), attempt="place")


# --- sync --------------------------------------------------------------------------------------


def test_a_position_without_a_stop_gets_one(clock: ManualClock) -> None:
    venue = StopBook(clock)
    results = manager(venue, clock).sync(
        book(position()), venue.listing(), venue_positions=held(book(position()))
    )
    assert actions(results) == ["placed"]
    kind, symbol, pos_side, qty, price, oid = venue.calls[0]
    assert (kind, symbol, pos_side, qty, price) == ("place", "NVDAUSDT", "long", "1.00", "213.97")
    assert OID.fullmatch(oid)
    assert [s.stop_price for s in venue.listing()] == [Decimal("213.97")]


def test_a_short_gets_its_stop_above(clock: ManualClock) -> None:
    venue = StopBook(clock)
    manager(venue, clock).sync(
        book(position(qty="-2.00")),
        venue.listing(),
        venue_positions=held(book(position(qty="-2.00"))),
    )
    assert venue.calls[0][2] == "short"
    assert venue.calls[0][3] == "2.00"


def test_a_stop_at_the_right_price_is_verified(clock: ManualClock) -> None:
    venue = StopBook(clock)
    venue.seed("NVDAUSDT", "213.97")
    results = manager(venue, clock).sync(
        book(position()), venue.listing(), venue_positions=held(book(position()))
    )
    assert actions(results) == ["verified"]
    assert venue.calls == []


def test_a_stop_at_the_wrong_price_is_replaced_new_first(clock: ManualClock) -> None:
    venue = StopBook(clock)
    old = venue.seed("NVDAUSDT", "200.00")
    results = manager(venue, clock).sync(
        book(position()), venue.listing(), venue_positions=held(book(position()))
    )
    assert actions(results) == ["replaced", "cancelled"]
    assert [c[0] for c in venue.calls] == ["place", "cancel"]
    assert venue.calls[1][2] == old.venue_id
    assert [(s.symbol, s.stop_price) for s in venue.listing()] == [("NVDAUSDT", Decimal("213.97"))]


def test_when_the_venue_allows_one_stop_the_old_goes_first(clock: ManualClock) -> None:
    venue = StopBook(clock, refuse_second=True)
    venue.seed("NVDAUSDT", "200.00")
    stops = manager(venue, clock)
    results = stops.sync(book(position()), venue.listing(), venue_positions=held(book(position())))
    assert [c[0] for c in venue.calls] == ["place", "cancel", "place"]
    assert actions(results) == ["cancelled", "replaced"]
    assert [s.stop_price for s in venue.listing()] == [Decimal("213.97")]
    assert len(stops.errors()) == 1  # the refused first placement, recorded


def test_a_replacement_that_timed_out_keeps_the_old_stop(clock: ManualClock) -> None:
    venue = StopBook(clock, timeout_place=True)
    venue.seed("NVDAUSDT", "200.00")
    stops = manager(venue, clock)
    results = stops.sync(book(position()), venue.listing(), venue_positions=held(book(position())))
    assert results == []
    assert [c[0] for c in venue.calls] == ["place"]
    assert len(venue.listing()) == 1
    assert stops.errors()[0].outcome_unknown


def test_a_position_that_cannot_be_protected_is_reported_missing(clock: ManualClock) -> None:
    venue = StopBook(clock, fail_place=True)
    stops = manager(venue, clock)
    results = stops.sync(book(position()), venue.listing(), venue_positions=held(book(position())))
    assert actions(results) == ["missing"]
    assert results[0].stop_price == Decimal("213.97")
    assert stops.errors()[0].action == "place"


def test_duplicates_are_cancelled_and_one_is_kept(clock: ManualClock) -> None:
    venue = StopBook(clock)
    venue.seed("NVDAUSDT", "213.97")
    venue.seed("NVDAUSDT", "213.97")
    venue.seed("NVDAUSDT", "190.00")
    results = manager(venue, clock).sync(
        book(position()), venue.listing(), venue_positions=held(book(position()))
    )
    assert actions(results) == ["verified", "cancelled", "cancelled"]
    assert len(venue.listing()) == 1


def test_orphans_and_stops_of_flat_positions_are_cancelled(clock: ManualClock) -> None:
    venue = StopBook(clock)
    venue.seed("BTCUSDT", "100000")
    venue.seed("NVDAUSDT", "213.97")
    flat = position(qty="0")
    results = manager(venue, clock).sync(
        book(flat), venue.listing(), venue_positions=held(book(flat))
    )
    assert actions(results) == ["cancelled", "cancelled"]
    assert venue.listing() == []


def test_failed_and_impossible_cancellations_are_errors_not_silence(clock: ManualClock) -> None:
    venue = StopBook(clock, fail_cancel=True)
    venue.seed("BTCUSDT", "100000")
    no_id = VenueStopOrder(symbol="TSLAUSDT", venue_id=None, stop_price=Decimal(1), blob=None)
    stops = manager(venue, clock)
    results = stops.sync(book(), [*venue.listing(), no_id], venue_positions=held(book()))
    assert results == []
    assert {(e.symbol, e.action) for e in stops.errors()} == {
        ("BTCUSDT", "cancel"),
        ("TSLAUSDT", "cancel"),
    }


def test_a_stop_listing_without_a_price_does_not_count_as_protection(clock: ManualClock) -> None:
    venue = StopBook(clock)
    venue.seed("NVDAUSDT", None, venue_id="tp-only")
    results = manager(venue, clock).sync(
        book(position()), venue.listing(), venue_positions=held(book(position()))
    )
    assert actions(results) == ["replaced", "cancelled"]


# --- against the simulated venue ----------------------------------------------------------------


def _book_from(venue: SimulatedVenue, at: datetime) -> BookState:
    positions = []
    for p in venue.positions():
        assert p.avg_price is not None
        positions.append(position(symbol=p.symbol, qty=str(p.qty), avg=str(p.avg_price)))
    return book(*positions).model_copy(update={"as_of": at})


def test_after_an_increase_there_is_exactly_one_stop_at_the_new_average(
    clock: ManualClock,
) -> None:
    market = FakeMarket()
    market.set("NVDAUSDT", bid="222.82", ask="222.88", mark="222.84", at=clock.now())
    venue = SimulatedVenue(market=market, clock=clock, starting_equity=Decimal("10000"))
    venue.place(mint_for_test(make_intent(qty="1.00")))
    market.set("NVDAUSDT", bid="230.00", ask="230.10", mark="230.05", at=clock.now())
    venue.place(mint_for_test(make_intent(qty="1.00", ruling_id="ruling-2", price="230.05")))
    current = _book_from(venue, clock.now())
    avg = current.positions["NVDAUSDT"].avg_entry
    assert avg == (Decimal("222.88") + Decimal("230.10")) / 2
    stops = manager(venue, clock)
    results = stops.sync(current, venue.stop_orders(), venue_positions=venue.positions())
    assert actions(results) == ["replaced"]  # the simulator holds one full-position stop
    listing = venue.stop_orders()
    assert len(listing) == 1
    assert listing[0].stop_price == desired_stop_price(
        current.positions["NVDAUSDT"], stop_loss_pct=0.04, spec=spec()
    )
    assert actions(stops.sync(current, venue.stop_orders(), venue_positions=venue.positions())) == [
        "verified"
    ]


# --- the venue's positions decide ---------------------------------------------------------------


def test_a_stop_is_never_cancelled_while_the_venue_holds_the_position(clock: ManualClock) -> None:
    """The ledger missed a fill (it shows BTCUSDT flat); the venue holds it. Its stop stays."""
    venue = StopBook(clock)
    venue.seed("BTCUSDT", "96000", venue_id="123")
    on_venue = [
        VenuePosition(symbol="BTCUSDT", qty=Decimal("1"), avg_price=Decimal(100000), blob=None)
    ]
    stops = StopManager(
        transport=venue,
        policy=POLICY_V1,
        clock=clock,
        specs={"BTCUSDT": spec("BTCUSDT", step="0.1")},
    )
    results = stops.sync(book(), venue.listing(), venue_positions=on_venue)
    assert actions(results) == ["verified"]
    assert not any(c[0] == "cancel" for c in venue.calls)
    assert [s.venue_id for s in venue.listing()] == ["123"]


def test_a_venue_position_the_ledger_lacks_gets_a_stop_at_the_venue_price(
    clock: ManualClock,
) -> None:
    venue = StopBook(clock)
    on_venue = [
        VenuePosition(
            symbol="NVDAUSDT", qty=Decimal("2.00"), avg_price=Decimal("222.88"), blob=None
        )
    ]
    results = manager(venue, clock).sync(book(), venue.listing(), venue_positions=on_venue)
    assert actions(results) == ["placed"]
    assert venue.calls[0][:5] == ("place", "NVDAUSDT", "long", "2.00", "213.97")


def test_a_quantity_disagreement_protects_the_venue_quantity(clock: ManualClock) -> None:
    """The ledger holds 1, the venue 3 (two fills not yet folded in): the stop covers 3."""
    venue = StopBook(clock)
    on_venue = [VenuePosition(symbol="NVDAUSDT", qty=Decimal("3.00"), avg_price=None, blob=None)]
    results = manager(venue, clock).sync(
        book(position()), venue.listing(), venue_positions=on_venue
    )
    assert actions(results) == ["placed"]
    assert venue.calls[0][3] == "3.00"  # sized to the venue, priced at the ledger's same-side entry
    assert venue.calls[0][4] == "213.97"


def test_the_ledger_position_the_venue_does_not_hold_gets_no_stop_and_its_stop_goes(
    clock: ManualClock,
) -> None:
    venue = StopBook(clock)
    venue.seed("NVDAUSDT", "213.97", venue_id="stale")
    results = manager(venue, clock).sync(book(position()), venue.listing(), venue_positions=[])
    assert actions(results) == ["cancelled"]
    assert venue.listing() == []


def test_an_unpriceable_venue_position_is_reported_missing_and_keeps_its_stop(
    clock: ManualClock,
) -> None:
    venue = StopBook(clock)
    venue.seed("TSLAUSDT", "300", venue_id="keep")
    on_venue = [
        VenuePosition(symbol="TSLAUSDT", qty=Decimal("1"), avg_price=None, blob=None),
        VenuePosition(symbol="AAPLUSDT", qty=Decimal("-1"), avg_price=None, blob=None),
    ]
    results = manager(venue, clock).sync(book(), venue.listing(), venue_positions=on_venue)
    assert [(r.symbol, r.action) for r in results] == [("AAPLUSDT", "missing")]
    assert results[0].stop_price is None
    assert [s.venue_id for s in venue.listing()] == ["keep"]
    assert venue.calls == []


def test_an_unread_venue_cancels_nothing_but_still_places(clock: ManualClock) -> None:
    venue = StopBook(clock)
    venue.seed("BTCUSDT", "96000", venue_id="orphan-or-not")
    venue.seed("NVDAUSDT", "213.97", venue_id="dup-1")
    venue.seed("NVDAUSDT", "213.97", venue_id="dup-2")
    venue.seed("TSLAUSDT", "1", venue_id="wrong-price")
    tsla = position(symbol="TSLAUSDT", qty="1.00", avg="300")
    stops = StopManager(
        transport=venue,
        policy=POLICY_V1,
        clock=clock,
        specs={"NVDAUSDT": spec(), "TSLAUSDT": spec("TSLAUSDT"), "AMZNUSDT": spec("AMZNUSDT")},
    )
    amzn = position(symbol="AMZNUSDT", qty="1.00", avg="200")
    results = stops.sync(book(position(), tsla, amzn), venue.listing(), venue_positions=None)
    assert not any(c[0] == "cancel" for c in venue.calls)
    assert sorted((r.symbol, r.action) for r in results) == [
        ("AMZNUSDT", "placed"),
        ("NVDAUSDT", "verified"),
        ("TSLAUSDT", "replaced"),
    ]
    ids = {s.venue_id for s in venue.listing()}
    assert {"orphan-or-not", "dup-1", "dup-2", "wrong-price"} <= ids
