"""The executor: preview then send, durable before the network, never twice, every answer logged."""

from datetime import timedelta
from decimal import Decimal
from typing import Any

import pytest

from execution.fakes import (
    DEMO_KEY,
    DEMO_PASSPHRASE,
    DEMO_SECRET,
    FakeMarket,
    MemoryBlobStore,
    MemoryLedger,
    ScriptedRunner,
    WriterOnlyLedger,
    fixture,
    fixture_result,
    has,
    ok_result,
    verb,
)
from helpers import T0, make_intent, mint_for_test
from sentiment_agent.clock import ManualClock
from sentiment_agent.execution.bgc import (
    BgcResult,
    BgcTimeoutError,
    BgcTransport,
    utc_to_ms,
)
from sentiment_agent.execution.environment import (
    DRY_RUN_FLAG,
    DemoCredentials,
    EnvironmentRefused,
)
from sentiment_agent.execution.executor import ApprovalTamperedError, Executor
from sentiment_agent.execution.orders import OrderTracker
from sentiment_agent.execution.simulated import SimulatedVenue
from sentiment_agent.types import (
    AccountSnapshot,
    ApprovedOrder,
    EnvironmentProof,
    EventKind,
    OrderIntent,
    OrderPurpose,
    OrderState,
    OrderStateChange,
    RunMode,
    Side,
    VenueRejection,
)

CREDS = DemoCredentials(api_key=DEMO_KEY, secret_key=DEMO_SECRET, passphrase=DEMO_PASSPHRASE)
INTENT = make_intent()  # NVDAUSDT buy 1.00: the intent the dry-run fixture was captured for
VENUE_ID = "121211212122"  # the orderId of the documented place-order answer


def _proof(*, passed: bool = True) -> EnvironmentProof:
    return EnvironmentProof(
        checked_at=T0,
        mode=RunMode.PAPER,
        credentials_file=".secrets/demo.env",
        key_declared_demo=True,
        bgc_package="@bitget-ai/bitget-agent-cli@3.0.0",
        paptrading_header_confirmed=True,
        demo_read_ok=passed,
        demo_read_code="00000",
        live_read_rejected=True,
        live_read_code="400",
        hold_mode="one_way_mode",
        account=AccountSnapshot(
            at=T0, equity_usdt=Decimal("10000"), available_usdt=Decimal("10000"), blob=None
        ),
        passed=passed,
        reasons=(),
        detail={"key_fingerprint": CREDS.fingerprint},
    )


def _advance(clock: ManualClock) -> Any:
    return lambda seconds: clock.advance(timedelta(seconds=seconds))


def _states(ledger: MemoryLedger, oid: str) -> list[tuple[OrderState, OrderState]]:
    changes = [
        OrderStateChange.model_validate(e.payload)
        for e in ledger.events(frozenset({EventKind.ORDER_STATE}))
    ]
    return [(c.from_state, c.to_state) for c in changes if c.client_oid == oid]


# --- simulated ---------------------------------------------------------------------------------


@pytest.fixture
def sim(clock: ManualClock) -> tuple[SimulatedVenue, MemoryLedger, OrderTracker, Executor]:
    market = FakeMarket()
    market.set("NVDAUSDT", bid="222.82", ask="222.88", mark="222.84", at=clock.now())
    venue = SimulatedVenue(market=market, clock=clock, starting_equity=Decimal("10000"))
    ledger = MemoryLedger(RunMode.SIMULATED, clock)
    tracker = OrderTracker()
    executor = Executor(
        transport=venue, ledger=ledger, tracker=tracker, clock=clock, sleep=_advance(clock)
    )
    return venue, ledger, tracker, executor


def test_preview_then_send_then_the_answer_then_the_fill(sim: Any) -> None:
    venue, ledger, _tracker, executor = sim
    cards = executor.execute([mint_for_test(INTENT)])
    assert ledger.kinds() == [
        EventKind.ORDER_PREVIEW,
        EventKind.ORDER_SUBMITTED,
        EventKind.ORDER_STATE,
        EventKind.ORDER_ACK,
        EventKind.ORDER_STATE,
        EventKind.STOP_SYNC,
        EventKind.ORDER_STATE,
        EventKind.FILL,
    ]
    assert _states(ledger, INTENT.client_oid) == [
        (OrderState.INITIALISED, OrderState.SUBMITTED),
        (OrderState.SUBMITTED, OrderState.ACCEPTED),
        (OrderState.ACCEPTED, OrderState.FILLED),
    ]
    (card,) = cards
    assert card.state is OrderState.FILLED
    assert card.venue_order_id is not None
    assert card.venue_order_id.startswith("sim-order-")
    assert len(card.fills) == 1
    assert card.fills[0].exec_price == Decimal("222.88")
    submitted = ledger.rows[1].payload
    assert submitted["argv"][-1] == "--paper-trading"
    assert DRY_RUN_FLAG not in submitted["argv"]
    assert ledger.rows[5].payload["action"] == "preset"
    assert [p.qty for p in venue.positions()] == [Decimal("1.00")]


def test_the_same_batch_twice_sends_once(sim: Any) -> None:
    venue, ledger, _tracker, executor = sim
    batch = [mint_for_test(INTENT)]
    executor.execute(batch)
    before = len(ledger.rows)
    cards = executor.execute(batch)
    assert len(ledger.rows) == before
    assert cards[0].state is OrderState.FILLED
    assert [p.qty for p in venue.positions()] == [Decimal("1.00")]


def test_after_a_restart_a_known_client_oid_is_still_never_resent(
    sim: Any, clock: ManualClock
) -> None:
    venue, ledger, _tracker, executor = sim
    executor.execute([mint_for_test(INTENT)])
    rebuilt = OrderTracker()
    rebuilt.restore(ledger.events())
    again = Executor(transport=venue, ledger=ledger, tracker=rebuilt, clock=clock)
    cards = again.execute([mint_for_test(INTENT)])
    assert cards[0].state is OrderState.FILLED
    assert len([k for k in ledger.kinds() if k is EventKind.ORDER_SUBMITTED]) == 1
    assert [p.qty for p in venue.positions()] == [Decimal("1.00")]


def test_a_batch_naming_one_order_twice_sends_it_once(sim: Any) -> None:
    venue, ledger, _tracker, executor = sim
    cards = executor.execute([mint_for_test(INTENT), mint_for_test(INTENT)])
    assert len(cards) == 1
    assert [p.qty for p in venue.positions()] == [Decimal("1.00")]
    assert len([k for k in ledger.kinds() if k is EventKind.ORDER_SUBMITTED]) == 1


def test_a_tampered_approval_stops_the_whole_batch(sim: Any) -> None:
    venue, ledger, _tracker, executor = sim
    good = mint_for_test(make_intent(symbol="NVDAUSDT", qty="2.00"))
    bad = mint_for_test(INTENT)
    object.__setattr__(bad, "_intent", make_intent(qty="40.00"))
    with pytest.raises(ApprovalTamperedError):
        executor.execute([good, bad])
    assert ledger.kinds() == [EventKind.NOTE]
    assert venue.positions() == []


def test_a_simulated_venue_can_never_write_to_a_paper_ledger(clock: ManualClock) -> None:
    venue = SimulatedVenue(market=FakeMarket(), clock=clock, starting_equity=Decimal(1))
    for mode in (RunMode.PAPER, RunMode.DRYRUN):
        with pytest.raises(ValueError, match="simulated fill must never"):
            Executor(
                transport=venue,
                ledger=MemoryLedger(mode, clock),
                tracker=OrderTracker(),
                clock=clock,
            )
    demo = BgcTransport(
        runner=ScriptedRunner(),
        clock=clock,
        blobs=MemoryBlobStore(),
        credentials=None,
        proof=None,
        dry_run_only=True,
    )
    with pytest.raises(ValueError, match="simulated fill must never"):
        Executor(
            transport=demo,
            ledger=MemoryLedger(RunMode.SIMULATED, clock),
            tracker=OrderTracker(),
            clock=clock,
        )


def test_an_order_the_venue_refuses_is_rejected(sim: Any) -> None:
    _venue, ledger, _tracker, executor = sim
    close = make_intent(side=Side.SELL, purpose=OrderPurpose.CLOSE)
    cards = executor.execute([mint_for_test(close)])
    assert cards[0].state is OrderState.REJECTED
    assert EventKind.ORDER_REJECTED in ledger.kinds()
    rejected = VenueRejection.model_validate(
        next(ledger.events(frozenset({EventKind.ORDER_REJECTED}))).payload
    )
    assert "no position" in rejected.message


# --- paper, through bgc ------------------------------------------------------------------------


def _detail(intent: OrderIntent, status: str) -> BgcResult:
    row: dict[str, Any] = dict(
        fixture("loopback_history_doc")["result"]["stdout"]["data"]["list"][0]
    )
    row.update(
        orderId=VENUE_ID,
        clientOid=intent.client_oid,
        symbol=intent.symbol,
        side=intent.side.value,
        orderType="market",
        qty="1",
        cumExecQty="1" if status == "filled" else "0",
        cumExecValue="222.88" if status == "filled" else "0",
        avgPrice="222.88" if status == "filled" else "",
        orderStatus=status,
        posSide="",
        holdMode="one_way_mode",
        createdTime=utc_to_ms(T0),
        updatedTime=utc_to_ms(T0),
    )
    return ok_result(row)


def _fills_for(intent: OrderIntent) -> BgcResult:
    row: dict[str, Any] = dict(fixture("loopback_fills_doc")["result"]["stdout"]["data"]["list"][0])
    row.update(
        execId="exec-1",
        orderId=VENUE_ID,
        clientOid=intent.client_oid,
        symbol=intent.symbol,
        side=intent.side.value,
        execPrice="222.88",
        execQty="1",
        execValue="222.88",
        feeDetail=[{"feeCoin": "USDT", "fee": "0.133728"}],
        createdTime=utc_to_ms(T0),
    )
    return ok_result({"list": [row], "cursor": ""})


def _paper_runner(place: BgcResult | BaseException | None = None) -> ScriptedRunner:
    runner = ScriptedRunner()
    runner.on(has(DRY_RUN_FLAG), fixture_result("dryrun_place_open_buy_one_way"))
    runner.on(
        verb("order", "--action", "place"),
        place if place is not None else fixture_result("loopback_place_ack"),
    )
    runner.on(verb("order", "--action", "detail"), _detail(INTENT, "live"), once=True)
    runner.on(verb("order", "--action", "detail"), _detail(INTENT, "filled"))
    runner.on(verb("order", "--action", "fills"), _fills_for(INTENT))
    return runner


def _paper(
    clock: ManualClock, runner: ScriptedRunner, *, proof: EnvironmentProof | None = None
) -> tuple[MemoryLedger, OrderTracker, Executor]:
    ledger = MemoryLedger(RunMode.PAPER, clock)
    ledger.append(EventKind.ENVIRONMENT_PROOF, proof or _proof())
    transport = BgcTransport(
        runner=runner,
        clock=clock,
        blobs=MemoryBlobStore(),
        credentials=CREDS,
        proof=_proof(),
        dry_run_only=False,
    )
    tracker = OrderTracker()
    executor = Executor(
        transport=transport,
        ledger=ledger,
        tracker=tracker,
        clock=clock,
        poll_interval_s=2.0,
        sleep=_advance(clock),
    )
    return ledger, tracker, executor


def test_paper_flow_through_agent_hub(clock: ManualClock) -> None:
    runner = _paper_runner()
    ledger, _tracker, executor = _paper(clock, runner)
    (card,) = executor.execute([mint_for_test(INTENT)])
    assert card.state is OrderState.FILLED
    assert card.venue_order_id == VENUE_ID
    assert [f.exec_id for f in card.fills] == ["exec-1"]
    assert card.fills[0].fee_paid == Decimal("0.133728")
    calls = runner.calls
    assert DRY_RUN_FLAG in calls[0][0]
    assert "BITGET_API_KEY" not in calls[0][1]
    assert calls[1][0][:3] == ("order", "--action", "place")
    assert DRY_RUN_FLAG not in calls[1][0]
    assert calls[1][1]["BITGET_API_KEY"] == DEMO_KEY
    assert calls[1][0] == tuple(ledger.rows[2].payload["argv"])  # the submitted argv was sent
    preview_seq = next(e.seq for e in ledger.rows if e.kind is EventKind.ORDER_PREVIEW)
    submit_seq = next(e.seq for e in ledger.rows if e.kind is EventKind.ORDER_SUBMITTED)
    ack_seq = next(e.seq for e in ledger.rows if e.kind is EventKind.ORDER_ACK)
    assert preview_seq < submit_seq < ack_seq
    preview_event = ledger.rows[preview_seq]
    assert preview_event.blobs, "the dry-run answer is kept as evidence"
    assert _states(ledger, INTENT.client_oid)[-1] == (OrderState.ACCEPTED, OrderState.FILLED)


@pytest.mark.parametrize("latest", ["missing", "failed"])
def test_paper_needs_a_passed_latest_proof_in_the_ledger(clock: ManualClock, latest: str) -> None:
    runner = _paper_runner()
    ledger, tracker, _ = _paper(clock, runner)
    if latest == "missing":
        ledger = MemoryLedger(RunMode.PAPER, clock)
    else:
        ledger.append(EventKind.ENVIRONMENT_PROOF, _proof(passed=False))
    transport = BgcTransport(
        runner=runner,
        clock=clock,
        blobs=MemoryBlobStore(),
        credentials=CREDS,
        proof=_proof(),
        dry_run_only=False,
    )
    executor = Executor(transport=transport, ledger=ledger, tracker=tracker, clock=clock)
    with pytest.raises(EnvironmentRefused):
        executor.execute([mint_for_test(INTENT)])
    assert runner.calls == []


def test_paper_refuses_a_ledger_it_cannot_read(clock: ManualClock) -> None:
    runner = _paper_runner()
    ledger, tracker, _ = _paper(clock, runner)
    transport = BgcTransport(
        runner=runner,
        clock=clock,
        blobs=MemoryBlobStore(),
        credentials=CREDS,
        proof=_proof(),
        dry_run_only=False,
    )
    executor = Executor(
        transport=transport, ledger=WriterOnlyLedger(ledger), tracker=tracker, clock=clock
    )
    with pytest.raises(EnvironmentRefused, match="read"):
        executor.execute([mint_for_test(INTENT)])


def test_a_timed_out_send_is_unknown_and_left_for_reconciliation(clock: ManualClock) -> None:
    runner = _paper_runner(place=BgcTimeoutError("bgc order --action place did not finish"))
    ledger, tracker, executor = _paper(clock, runner)
    (card,) = executor.execute([mint_for_test(INTENT)])
    assert card.state is OrderState.UNKNOWN
    assert tracker.unknown() == [INTENT.client_oid]
    assert EventKind.ORDER_UNKNOWN in ledger.kinds()
    assert not [a for a in runner.argvs() if a[:3] == ("order", "--action", "detail")]
    # Executing the batch again does not resend: the clientOid is in the ledger.
    executor.execute([mint_for_test(INTENT)])
    sends = [
        a
        for a in runner.argvs()
        if a[:3] == ("order", "--action", "place") and DRY_RUN_FLAG not in a
    ]
    assert len(sends) == 1


def test_a_venue_rejection_is_logged_and_final(clock: ManualClock) -> None:
    runner = _paper_runner(place=fixture_result("loopback_place_rejected_balance"))
    ledger, _tracker, executor = _paper(clock, runner)
    (card,) = executor.execute([mint_for_test(INTENT)])
    assert card.state is OrderState.REJECTED
    assert EventKind.ORDER_REJECTED in ledger.kinds()
    assert EventKind.ORDER_ACK not in ledger.kinds()


def test_40099_on_a_send_is_logged_then_raised(clock: ManualClock) -> None:
    runner = _paper_runner(place=fixture_result("loopback_place_40099"))
    ledger, tracker, executor = _paper(clock, runner)
    with pytest.raises(EnvironmentRefused, match="40099"):
        executor.execute([mint_for_test(INTENT)])
    rejected = VenueRejection.model_validate(
        next(ledger.events(frozenset({EventKind.ORDER_REJECTED}))).payload
    )
    assert rejected.code == "40099"
    assert rejected.blob is not None, "the 40099 answer is kept as evidence"
    assert tracker.state(INTENT.client_oid) is OrderState.REJECTED


def test_an_unexpected_failure_while_sending_is_unknown_and_raised(clock: ManualClock) -> None:
    runner = _paper_runner(place=RuntimeError("the runner broke"))
    ledger, tracker, executor = _paper(clock, runner)
    with pytest.raises(RuntimeError, match="runner broke"):
        executor.execute([mint_for_test(INTENT)])
    assert tracker.state(INTENT.client_oid) is OrderState.UNKNOWN
    assert EventKind.ORDER_UNKNOWN in ledger.kinds()


def test_a_failed_preview_denies_the_order_and_nothing_is_sent(clock: ManualClock) -> None:
    runner = ScriptedRunner().on(has(DRY_RUN_FLAG), fixture_result("local_paper_and_read_only"))
    ledger, tracker, executor = _paper(clock, runner)
    (card,) = executor.execute([mint_for_test(INTENT)])
    assert card.state is OrderState.DENIED
    assert tracker.state(INTENT.client_oid) is OrderState.DENIED
    assert EventKind.ORDER_SUBMITTED not in ledger.kinds()
    assert len(runner.calls) == 1


def test_a_preview_that_disagrees_with_the_approval_denies_it(clock: ManualClock) -> None:
    other = fixture_result("dryrun_place_close_one_way")
    runner = ScriptedRunner().on(has(DRY_RUN_FLAG), other)
    _ledger, tracker, executor = _paper(clock, runner)
    executor.execute([mint_for_test(INTENT)])
    assert tracker.state(INTENT.client_oid) is OrderState.DENIED
    change = tracker.last_change(INTENT.client_oid)
    assert change is not None
    assert "preview" in change.reason


def test_polling_stops_at_the_timeout_and_leaves_a_working_order_live(clock: ManualClock) -> None:
    runner = ScriptedRunner()
    runner.on(has(DRY_RUN_FLAG), fixture_result("dryrun_place_open_buy_one_way"))
    runner.on(verb("order", "--action", "place"), fixture_result("loopback_place_ack"))
    runner.on(verb("order", "--action", "detail"), _detail(INTENT, "live"))
    ledger, tracker, executor = _paper(clock, runner)
    start = clock.now()
    (card,) = executor.execute([mint_for_test(INTENT)])
    assert card.state is OrderState.ACCEPTED
    assert tracker.live() == [INTENT.client_oid]
    assert clock.now() - start <= timedelta(seconds=62)
    details = [a for a in runner.argvs() if a[:3] == ("order", "--action", "detail")]
    assert 20 <= len(details) <= 32
    assert EventKind.FILL not in ledger.kinds()


def test_a_read_failure_while_polling_is_not_fatal(clock: ManualClock) -> None:
    runner = ScriptedRunner()
    runner.on(has(DRY_RUN_FLAG), fixture_result("dryrun_place_open_buy_one_way"))
    runner.on(verb("order", "--action", "place"), fixture_result("loopback_place_ack"))
    runner.on(
        verb("order", "--action", "detail"),
        fixture_result("loopback_place_network_error"),
        once=True,
    )
    runner.on(verb("order", "--action", "detail"), _detail(INTENT, "filled"))
    runner.on(verb("order", "--action", "fills"), _fills_for(INTENT))
    _ledger, _tracker, executor = _paper(clock, runner)
    (card,) = executor.execute([mint_for_test(INTENT)])
    assert card.state is OrderState.FILLED


def test_a_venue_state_that_is_not_a_legal_move_is_noted_not_applied(clock: ManualClock) -> None:
    runner = ScriptedRunner()
    runner.on(has(DRY_RUN_FLAG), fixture_result("dryrun_place_open_buy_one_way"))
    runner.on(verb("order", "--action", "place"), fixture_result("loopback_place_ack"))
    for status in ("partially_filled", "live"):
        runner.on(verb("order", "--action", "detail"), _detail(INTENT, status), once=True)
    runner.on(verb("order", "--action", "detail"), _detail(INTENT, "filled"))
    runner.on(verb("order", "--action", "fills"), _fills_for(INTENT))
    ledger, tracker, executor = _paper(clock, runner)
    executor.execute([mint_for_test(INTENT)])
    assert _states(ledger, INTENT.client_oid)[1:] == [
        (OrderState.SUBMITTED, OrderState.ACCEPTED),
        (OrderState.ACCEPTED, OrderState.PARTIALLY_FILLED),
        (OrderState.PARTIALLY_FILLED, OrderState.FILLED),
    ]
    notes = [e.payload["text"] for e in ledger.events(frozenset({EventKind.NOTE}))]
    assert len(notes) == 1
    assert "left for reconciliation" in notes[0]
    assert tracker.state(INTENT.client_oid) is OrderState.FILLED


# --- dry run -----------------------------------------------------------------------------------


def test_dryrun_previews_and_stops(clock: ManualClock) -> None:
    runner = ScriptedRunner().on(has(DRY_RUN_FLAG), fixture_result("dryrun_place_open_buy_one_way"))
    transport = BgcTransport(
        runner=runner,
        clock=clock,
        blobs=MemoryBlobStore(),
        credentials=None,
        proof=None,
        dry_run_only=True,
    )
    ledger = MemoryLedger(RunMode.DRYRUN, clock)
    tracker = OrderTracker()
    executor = Executor(transport=transport, ledger=ledger, tracker=tracker, clock=clock)
    (card,) = executor.execute([mint_for_test(INTENT)])
    assert ledger.kinds() == [EventKind.ORDER_PREVIEW]
    assert card.state is OrderState.INITIALISED
    assert tracker.state(INTENT.client_oid) is None
    assert all(DRY_RUN_FLAG in a for a in runner.argvs())
    assert all("BITGET_API_KEY" not in env for _a, env, _t in runner.calls)


def test_an_empty_batch_does_nothing(clock: ManualClock) -> None:
    ledger = MemoryLedger(RunMode.PAPER, clock)
    transport = BgcTransport(
        runner=ScriptedRunner(),
        clock=clock,
        blobs=MemoryBlobStore(),
        credentials=CREDS,
        proof=_proof(),
        dry_run_only=False,
    )
    executor = Executor(transport=transport, ledger=ledger, tracker=OrderTracker(), clock=clock)
    assert executor.execute([]) == []
    assert ledger.rows == []


def test_only_the_kernel_can_mint_what_the_executor_accepts() -> None:
    with pytest.raises(PermissionError):
        ApprovedOrder(INTENT, INTENT.ruling_id, _token=object())
