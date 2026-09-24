"""Reconciliation: UNKNOWN resolved by reading back, never by resending; differences reported."""

import json
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from execution.fakes import (
    DEMO_KEY,
    DEMO_PASSPHRASE,
    DEMO_SECRET,
    MemoryBlobStore,
    MemoryLedger,
    ScriptedRunner,
    fixture,
    fixture_result,
    has,
    ok_result,
    verb,
)
from helpers import T0, empty_book, make_intent, mint_for_test
from sentiment_agent.clock import ManualClock
from sentiment_agent.execution.bgc import (
    BgcResult,
    BgcTimeoutError,
    BgcTransport,
    VenueReadError,
    utc_to_ms,
)
from sentiment_agent.execution.environment import DRY_RUN_FLAG, DemoCredentials, EnvironmentRefused
from sentiment_agent.execution.executor import Executor
from sentiment_agent.execution.orders import OrderTracker
from sentiment_agent.execution.reconcile import Reconciler
from sentiment_agent.types import (
    AccountSnapshot,
    ApprovedOrder,
    BookState,
    DryRunPreview,
    EnvironmentProof,
    EventKind,
    Fill,
    FillVenue,
    OrderIntent,
    OrderState,
    Position,
    ReconciliationReport,
    RunMode,
    Side,
    VenueAck,
    VenueOrder,
    VenueOrderStatus,
    VenuePosition,
    VenueRejection,
    VenueStopOrder,
    VenueUnknown,
)

CREDS = DemoCredentials(api_key=DEMO_KEY, secret_key=DEMO_SECRET, passphrase=DEMO_PASSPHRASE)
INTENT = make_intent()
VENUE_ID = "121211212122"


# --- a venue the test controls ------------------------------------------------------------------


class FakeVenue:
    """Read side of a venue. Reconciliation never sends, so the write side fails the test."""

    def __init__(self, clock: ManualClock) -> None:
        self.clock = clock
        self.orders: dict[str, VenueOrder] = {}
        self.fill_rows: list[Fill] = []
        self.position_rows: list[VenuePosition] = []
        self.stop_rows: list[VenueStopOrder] = []
        self.equity: Decimal | None = Decimal("10000")
        self.history_rows: list[VenueOrder] = []
        self.failing: set[str] = set()
        self.reads: list[str] = []

    @property
    def venue(self) -> FillVenue:
        return FillVenue.BITGET_DEMO

    def _read(self, what: str) -> None:
        self.reads.append(what)
        if what in self.failing:
            raise VenueReadError(f"{what} failed")
        if "environment" in self.failing:
            raise EnvironmentRefused("40099")

    def preview(self, intent: OrderIntent) -> DryRunPreview:
        raise AssertionError("reconciliation never previews")

    def place(self, order: ApprovedOrder) -> VenueAck | VenueRejection | VenueUnknown:
        raise AssertionError("reconciliation never sends")

    def order(self, *, client_oid: str) -> VenueOrder | None:
        self._read("order")
        return self.orders.get(client_oid)

    def fills(self, *, since: datetime, until: datetime) -> list[Fill]:
        self._read("fills")
        return [f for f in self.fill_rows if since <= f.executed_at <= until]

    def history(self, *, since: datetime, until: datetime) -> list[VenueOrder]:
        self._read("history")
        return list(self.history_rows)

    def positions(self) -> list[VenuePosition]:
        self._read("positions")
        return list(self.position_rows)

    def stop_orders(self) -> list[VenueStopOrder]:
        self._read("stops")
        return list(self.stop_rows)

    def account(self) -> AccountSnapshot:
        self._read("account")
        return AccountSnapshot(
            at=self.clock.now(), equity_usdt=self.equity, available_usdt=None, blob=None
        )


def venue_order(
    intent: OrderIntent, status: VenueOrderStatus, *, executed: str = "1.00", at: datetime = T0
) -> VenueOrder:
    return VenueOrder(
        venue_order_id=VENUE_ID,
        client_oid=intent.client_oid,
        symbol=intent.symbol,
        side=intent.side,
        order_type="market",
        qty=intent.qty,
        cum_exec_qty=Decimal(executed),
        cum_exec_value=Decimal(executed) * Decimal("222.88"),
        avg_price=Decimal("222.88") if Decimal(executed) else None,
        status=status,
        reduce_only=False,
        delegate_type="market",
        cancel_reason=None,
        fees=(),
        created_at=at,
        updated_at=at,
        blob=None,
    )


def fill(
    exec_id: str,
    *,
    side: Side = Side.BUY,
    qty: str = "1.00",
    client_oid: str | None = INTENT.client_oid,
    symbol: str = "NVDAUSDT",
    at: datetime = T0,
) -> Fill:
    return Fill(
        exec_id=exec_id,
        venue_order_id=VENUE_ID,
        client_oid=client_oid,
        symbol=symbol,
        side=side,
        exec_price=Decimal("222.88"),
        exec_qty=Decimal(qty),
        exec_value=Decimal(qty) * Decimal("222.88"),
        fee_paid=Decimal("0.13"),
        fee_coin="USDT",
        trade_scope="taker",
        trade_side="open",
        exec_pnl=None,
        executed_at=at,
        venue=FillVenue.BITGET_DEMO,
    )


def holding(qty: str, symbol: str = "NVDAUSDT") -> BookState:
    position = Position(
        symbol=symbol,
        qty=Decimal(qty),
        avg_entry=Decimal("222.88"),
        opened_at=T0,
        last_increase_at=T0,
        realized_pnl=Decimal(0),
        fees_paid=Decimal(0),
        stop_price=None,
        stop_venue_id=None,
        last_decision_id=None,
    )
    return empty_book().model_copy(
        update={"positions": {symbol: position}, "marks": {symbol: Decimal("222.88")}}
    )


def reconciler(
    venue: Any, clock: ManualClock, tracker: OrderTracker | None = None, **kw: Any
) -> tuple[Reconciler, MemoryLedger, OrderTracker]:
    ledger = MemoryLedger(RunMode.PAPER, clock)
    tracker = tracker or OrderTracker()
    return (
        Reconciler(transport=venue, ledger=ledger, tracker=tracker, clock=clock, **kw),
        ledger,
        tracker,
    )


def kinds(report: ReconciliationReport) -> list[str]:
    return [d.kind for d in report.discrepancies]


def run(rec: Reconciler, book: BookState, clock: ManualClock, **kw: Any) -> ReconciliationReport:
    return rec.run(
        book=book, since=clock.now() - timedelta(hours=1), known_fill_ids=frozenset(), **kw
    )


# --- the whole path: a timed-out send, resolved by reading back ----------------------------------


def _proof() -> EnvironmentProof:
    return EnvironmentProof(
        checked_at=T0,
        mode=RunMode.PAPER,
        credentials_file=".secrets/demo.env",
        key_declared_demo=True,
        bgc_package="@bitget-ai/bitget-agent-cli@3.0.0",
        paptrading_header_confirmed=True,
        demo_read_ok=True,
        demo_read_code="00000",
        live_read_rejected=True,
        live_read_code="400",
        hold_mode="one_way_mode",
        account=AccountSnapshot(
            at=T0, equity_usdt=Decimal("10000"), available_usdt=None, blob=None
        ),
        passed=True,
        reasons=(),
        detail={"key_fingerprint": CREDS.fingerprint},
    )


def _detail_row(status: str) -> dict[str, Any]:
    row: dict[str, Any] = dict(
        fixture("loopback_history_doc")["result"]["stdout"]["data"]["list"][0]
    )
    row.update(
        orderId=VENUE_ID,
        clientOid=INTENT.client_oid,
        symbol="NVDAUSDT",
        side="buy",
        orderType="market",
        qty="1",
        cumExecQty="1",
        cumExecValue="222.88",
        avgPrice="222.88",
        orderStatus=status,
        createdTime=utc_to_ms(T0),
        updatedTime=utc_to_ms(T0),
    )
    return row


def _fill_row() -> dict[str, Any]:
    row: dict[str, Any] = dict(fixture("loopback_fills_doc")["result"]["stdout"]["data"]["list"][0])
    row.update(
        execId="exec-1",
        orderId=VENUE_ID,
        clientOid=INTENT.client_oid,
        symbol="NVDAUSDT",
        side="buy",
        execPrice="222.88",
        execQty="1",
        execValue="222.88",
        createdTime=utc_to_ms(T0),
    )
    return row


def _account_assets(equity: str) -> BgcResult:
    ok = fixture_result("loopback_account_assets_doc")
    assert ok.stdout is not None
    stdout: dict[str, Any] = json_copy(ok.stdout)
    stdout["data"]["usdtEquity"] = equity
    for row in stdout["data"].get("assets") or []:
        if row.get("coin") == "USDT":
            row["equity"] = equity  # the book reconciles against the USDT row
    return BgcResult(0, stdout, None, 1)


def json_copy(value: dict[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = json.loads(json.dumps(value))
    return copied


def test_a_timed_out_send_is_resolved_by_reading_it_back(clock: ManualClock) -> None:
    runner = ScriptedRunner()
    runner.on(has(DRY_RUN_FLAG), fixture_result("dryrun_place_open_buy_one_way"))
    runner.on(verb("order", "--action", "place"), BgcTimeoutError("send timed out"))
    runner.on(verb("order", "--action", "detail"), ok_result(_detail_row("filled")))
    runner.on(verb("order", "--action", "fills"), ok_result({"list": [_fill_row()], "cursor": ""}))
    position_row = dict(fixture("loopback_positions_doc")["result"]["stdout"]["data"]["list"][0])
    position_row.update(symbol="NVDAUSDT", posSide="long", total="1", avgPrice="222.88")
    runner.on(verb("position"), ok_result({"list": [position_row]}))
    stop_row = dict(fixture("loopback_stop_orders_doc")["result"]["stdout"]["data"][0])
    stop_row.update(symbol="NVDAUSDT", stopLoss="213.97", takeProfit="")
    runner.on(verb("strategy_order"), ok_result([stop_row]))
    runner.on(verb("raw"), _account_assets("10000"))

    ledger = MemoryLedger(RunMode.PAPER, clock)
    ledger.append(EventKind.ENVIRONMENT_PROOF, _proof())
    transport = BgcTransport(
        runner=runner,
        clock=clock,
        blobs=MemoryBlobStore(),
        credentials=CREDS,
        proof=_proof(),
        dry_run_only=False,
    )
    tracker = OrderTracker()
    Executor(transport=transport, ledger=ledger, tracker=tracker, clock=clock).execute(
        [mint_for_test(INTENT)]
    )
    assert tracker.state(INTENT.client_oid) is OrderState.UNKNOWN

    clock.advance(timedelta(minutes=1))
    report = Reconciler(transport=transport, ledger=ledger, tracker=tracker, clock=clock).run(
        book=empty_book(), since=T0 - timedelta(hours=1), known_fill_ids=frozenset()
    )
    assert tracker.state(INTENT.client_oid) is OrderState.FILLED
    assert report.resolved_unknown == (INTENT.client_oid,)
    assert report.new_fill_ids == ("exec-1",)
    assert report.clean, report.discrepancies
    assert report.account is not None
    assert report.account.equity_usdt == Decimal("10000")
    sends = [
        a
        for a in runner.argvs()
        if a[:3] == ("order", "--action", "place") and DRY_RUN_FLAG not in a
    ]
    assert len(sends) == 1, "an UNKNOWN order is read back, never sent again"
    assert ledger.kinds()[-1] is EventKind.RECONCILIATION
    assert EventKind.FILL in ledger.kinds()


# --- orders --------------------------------------------------------------------------------------


def _unknown_tracker(clock: ManualClock) -> OrderTracker:
    tracker = OrderTracker()
    tracker.transition(INTENT.client_oid, OrderState.SUBMITTED, at=clock.now(), reason="sent")
    tracker.transition(INTENT.client_oid, OrderState.UNKNOWN, at=clock.now(), reason="timeout")
    return tracker


def test_an_order_the_venue_cannot_find_stays_live_until_the_grace_ends(
    clock: ManualClock,
) -> None:
    venue = FakeVenue(clock)
    rec, ledger, tracker = reconciler(venue, clock, _unknown_tracker(clock), unknown_grace_s=300)
    clock.advance(timedelta(seconds=100))
    report = run(rec, empty_book(), clock)
    assert tracker.state(INTENT.client_oid) is OrderState.UNKNOWN
    assert kinds(report) == ["unknown_order"]
    assert report.resolved_unknown == ()
    clock.advance(timedelta(seconds=300))
    report = run(rec, empty_book(), clock)
    assert tracker.state(INTENT.client_oid) is OrderState.REJECTED
    assert report.resolved_unknown == (INTENT.client_oid,)
    change = tracker.last_change(INTENT.client_oid)
    assert change is not None
    assert "never reached the venue" in change.reason
    assert "not sent again" in change.reason
    assert EventKind.ORDER_STATE in ledger.kinds()


def test_an_unreadable_order_is_reported_and_its_state_kept(clock: ManualClock) -> None:
    venue = FakeVenue(clock)
    venue.failing.add("order")
    rec, _ledger, tracker = reconciler(venue, clock, _unknown_tracker(clock))
    report = run(rec, empty_book(), clock)
    assert kinds(report) == ["unknown_order"]
    assert "unreadable" in report.discrepancies[0].detail
    assert tracker.state(INTENT.client_oid) is OrderState.UNKNOWN


def test_a_working_order_moves_to_the_venues_state(clock: ManualClock) -> None:
    venue = FakeVenue(clock)
    tracker = OrderTracker()
    tracker.transition(INTENT.client_oid, OrderState.SUBMITTED, at=T0, reason="sent")
    tracker.transition(INTENT.client_oid, OrderState.ACCEPTED, at=T0, reason="ack")
    venue.orders[INTENT.client_oid] = venue_order(INTENT, VenueOrderStatus.CANCELLED, executed="0")
    rec, _ledger, tracker = reconciler(venue, clock, tracker)
    report = run(rec, empty_book(), clock)
    assert tracker.state(INTENT.client_oid) is OrderState.CANCELLED
    assert report.orders_checked == 1
    assert report.clean


def test_an_illegal_venue_state_is_a_discrepancy(clock: ManualClock) -> None:
    venue = FakeVenue(clock)
    tracker = OrderTracker()
    tracker.transition(INTENT.client_oid, OrderState.SUBMITTED, at=T0, reason="sent")
    tracker.transition(INTENT.client_oid, OrderState.PARTIALLY_FILLED, at=T0, reason="part")
    venue.orders[INTENT.client_oid] = venue_order(INTENT, VenueOrderStatus.LIVE, executed="0")
    rec, _ledger, tracker = reconciler(venue, clock, tracker)
    report = run(rec, empty_book(), clock)
    assert kinds(report) == ["unknown_order"]
    assert tracker.state(INTENT.client_oid) is OrderState.PARTIALLY_FILLED


# --- fills ---------------------------------------------------------------------------------------


def test_new_fills_are_written_once_and_known_ones_skipped(clock: ManualClock) -> None:
    venue = FakeVenue(clock)
    venue.fill_rows = [fill("f-1"), fill("f-1"), fill("f-2", qty="0.50")]
    venue.position_rows = [
        VenuePosition(symbol="NVDAUSDT", qty=Decimal("1.50"), avg_price=None, blob=None)
    ]
    venue.stop_rows = [
        VenueStopOrder(symbol="NVDAUSDT", venue_id="s", stop_price=Decimal(1), blob=None)
    ]
    tracker = OrderTracker()
    tracker.transition(INTENT.client_oid, OrderState.SUBMITTED, at=T0, reason="sent")
    tracker.transition(INTENT.client_oid, OrderState.FILLED, at=T0, reason="filled")
    rec, ledger, tracker = reconciler(venue, clock, tracker)
    report = rec.run(
        book=empty_book(), since=T0 - timedelta(hours=1), known_fill_ids=frozenset({"f-2"})
    )
    assert report.new_fill_ids == ("f-1",)
    assert [k for k in ledger.kinds() if k is EventKind.FILL] == [EventKind.FILL]
    assert tracker.has_fill("f-1")
    assert kinds(report) == ["position_mismatch"]  # f-2 is known to the caller, not to the book


def test_a_foreign_fill_that_adds_exposure_is_an_orphan_and_still_booked(
    clock: ManualClock,
) -> None:
    venue = FakeVenue(clock)
    venue.fill_rows = [fill("manual-1", client_oid="web-ui-order")]
    venue.position_rows = [
        VenuePosition(symbol="NVDAUSDT", qty=Decimal("1.00"), avg_price=None, blob=None)
    ]
    venue.stop_rows = [
        VenueStopOrder(symbol="NVDAUSDT", venue_id="s", stop_price=Decimal(1), blob=None)
    ]
    rec, ledger, _tracker = reconciler(venue, clock)
    report = run(rec, empty_book(), clock)
    assert kinds(report) == ["orphan_fill"]
    assert report.new_fill_ids == ("manual-1",)
    assert EventKind.FILL in ledger.kinds()


def test_a_venue_stop_fill_that_reduces_the_book_is_expected(clock: ManualClock) -> None:
    venue = FakeVenue(clock)
    venue.fill_rows = [fill("stop-1", side=Side.SELL, client_oid=None)]
    rec, _ledger, _tracker = reconciler(venue, clock)
    report = run(rec, holding("1.00"), clock)
    assert report.clean, report.discrepancies
    assert report.new_fill_ids == ("stop-1",)


def test_an_executed_order_without_its_fills_is_a_missing_fill(clock: ManualClock) -> None:
    venue = FakeVenue(clock)
    tracker = OrderTracker()
    tracker.transition(INTENT.client_oid, OrderState.SUBMITTED, at=T0, reason="sent")
    tracker.transition(INTENT.client_oid, OrderState.ACCEPTED, at=T0, reason="ack")
    venue.orders[INTENT.client_oid] = venue_order(INTENT, VenueOrderStatus.FILLED)
    venue.position_rows = [
        VenuePosition(symbol="NVDAUSDT", qty=Decimal("1.00"), avg_price=None, blob=None)
    ]
    venue.stop_rows = [
        VenueStopOrder(symbol="NVDAUSDT", venue_id="s", stop_price=Decimal(1), blob=None)
    ]
    rec, _ledger, _tracker = reconciler(venue, clock, tracker)
    report = run(rec, holding("1.00"), clock)
    assert kinds(report) == ["missing_fill"]


def test_unreadable_fills_are_reported(clock: ManualClock) -> None:
    venue = FakeVenue(clock)
    venue.failing.add("fills")
    rec, _ledger, _tracker = reconciler(venue, clock)
    report = run(rec, empty_book(), clock)
    assert kinds(report) == ["missing_fill"]
    assert not report.fills_read, "a failed read must not move the fills window on"
    assert report.unreconciled
    assert report.unreconciled[0].startswith("missing_fill")


def test_a_clean_sweep_reads_fills_and_leaves_nothing_unreconciled(clock: ManualClock) -> None:
    venue = FakeVenue(clock)
    rec, _ledger, _tracker = reconciler(venue, clock)
    report = run(rec, empty_book(), clock)
    assert report.fills_read
    assert report.unreconciled == ()


# --- positions, stops, account -------------------------------------------------------------------


def test_positions_are_compared_exactly(clock: ManualClock) -> None:
    venue = FakeVenue(clock)
    venue.position_rows = [
        VenuePosition(symbol="NVDAUSDT", qty=Decimal("2.00"), avg_price=None, blob=None),
        VenuePosition(symbol="BTCUSDT", qty=Decimal("-0.01"), avg_price=None, blob=None),
    ]
    venue.stop_rows = [
        VenueStopOrder(symbol="NVDAUSDT", venue_id="s", stop_price=Decimal(1), blob=None)
    ]
    rec, _ledger, _tracker = reconciler(venue, clock)
    report = run(rec, holding("1.00"), clock)
    mismatches = [d for d in report.discrepancies if d.kind == "position_mismatch"]
    assert sorted(d.symbol or "" for d in mismatches) == ["BTCUSDT", "NVDAUSDT"]


def test_a_position_without_a_stop_and_stops_without_positions(clock: ManualClock) -> None:
    venue = FakeVenue(clock)
    venue.position_rows = [
        VenuePosition(symbol="NVDAUSDT", qty=Decimal("1.00"), avg_price=None, blob=None)
    ]
    venue.stop_rows = [
        VenueStopOrder(symbol="BTCUSDT", venue_id="o1", stop_price=Decimal(1), blob=None),
        VenueStopOrder(symbol="NVDAUSDT", venue_id="tp", stop_price=None, blob=None),
    ]
    rec, _ledger, _tracker = reconciler(venue, clock)
    report = run(rec, holding("1.00"), clock)
    assert sorted(kinds(report)) == ["missing_stop", "orphan_stop"]


def test_a_second_stop_on_one_position_is_an_orphan(clock: ManualClock) -> None:
    venue = FakeVenue(clock)
    venue.position_rows = [
        VenuePosition(symbol="NVDAUSDT", qty=Decimal("1.00"), avg_price=None, blob=None)
    ]
    venue.stop_rows = [
        VenueStopOrder(symbol="NVDAUSDT", venue_id="a", stop_price=Decimal(213), blob=None),
        VenueStopOrder(symbol="NVDAUSDT", venue_id="b", stop_price=Decimal(214), blob=None),
    ]
    rec, _ledger, _tracker = reconciler(venue, clock)
    assert kinds(run(rec, holding("1.00"), clock)) == ["orphan_stop"]


@pytest.mark.parametrize(("equity", "gap"), [("10000", False), ("10049", False), ("10051", True)])
def test_equity_gap_beyond_the_tolerance(clock: ManualClock, equity: str, gap: bool) -> None:
    venue = FakeVenue(clock)
    venue.equity = Decimal(equity)
    rec, _ledger, _tracker = reconciler(venue, clock)
    report = run(rec, empty_book(), clock)
    assert ("equity_gap" in kinds(report)) is gap


def test_unreadable_positions_stops_and_account_are_reported(clock: ManualClock) -> None:
    venue = FakeVenue(clock)
    venue.failing.update({"positions", "stops", "account"})
    rec, _ledger, _tracker = reconciler(venue, clock)
    report = run(rec, empty_book(), clock)
    assert sorted(kinds(report)) == ["equity_gap", "missing_stop", "position_mismatch"]
    assert report.account is None
    assert [u.split(":")[0] for u in report.unreconciled] == ["position_mismatch"]


# --- history, modes, refusals --------------------------------------------------------------------


def test_the_daily_sweep_finds_orders_this_ledger_never_made(clock: ManualClock) -> None:
    venue = FakeVenue(clock)
    foreign = venue_order(make_intent(ruling_id="elsewhere"), VenueOrderStatus.FILLED)
    stop_exit = venue_order(make_intent(ruling_id="stop"), VenueOrderStatus.FILLED).model_copy(
        update={"client_oid": None, "delegate_type": "position_stop_loss_market"}
    )
    manual = foreign.model_copy(update={"client_oid": None, "delegate_type": "normal"})
    venue.history_rows = [foreign, stop_exit, manual]
    rec, _ledger, _tracker = reconciler(venue, clock)
    report = run(rec, empty_book(), clock, full_history=True)
    assert kinds(report) == ["unknown_order", "unknown_order"]
    assert report.orders_checked == 3
    assert "history" in venue.reads
    plain = run(reconciler(FakeVenue(clock), clock)[0], empty_book(), clock)
    assert plain.clean


def test_dryrun_reconciles_nothing(clock: ManualClock) -> None:
    venue = FakeVenue(clock)
    ledger = MemoryLedger(RunMode.DRYRUN, clock)
    rec = Reconciler(transport=venue, ledger=ledger, tracker=OrderTracker(), clock=clock)
    report = rec.run(book=empty_book(), since=T0, known_fill_ids=frozenset())
    assert report.clean
    assert report.orders_checked == 0
    assert venue.reads == []
    assert ledger.kinds() == [EventKind.RECONCILIATION]


def test_40099_during_reconciliation_is_not_swallowed(clock: ManualClock) -> None:
    venue = FakeVenue(clock)
    venue.failing.add("environment")
    rec, _ledger, _tracker = reconciler(venue, clock, _unknown_tracker(clock))
    with pytest.raises(EnvironmentRefused):
        run(rec, empty_book(), clock)


def test_a_clean_sweep_is_logged(clock: ManualClock) -> None:
    rec, ledger, _tracker = reconciler(FakeVenue(clock), clock)
    report = run(rec, empty_book(), clock)
    assert report.clean
    logged = ReconciliationReport.model_validate(ledger.rows[-1].payload)
    assert logged == report
