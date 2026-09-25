"""The projection: every event kind read back typed, and the book rebuilt from the log alone.

The central property: a projection built by replaying a ledger equals one fed the same events one
at a time (as the runtime feeds it), at every prefix, and both equal a book driven by hand with the
attribution the log implies. The rest pins what the log may and may not do."""

from collections.abc import Sequence
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from book.factories import (
    T0,
    H,
    MemoryLedger,
    d,
    make_fill,
    make_genesis,
    make_intent,
    make_mark,
    make_proof,
)
from helpers import empty_book
from sentiment_agent.book.book import (
    EXIT_STOP,
    EXIT_VENUE_INITIATED,
    BookBuilder,
)
from sentiment_agent.book.projection import ORDER_EVENT_KINDS, Projection, ProjectionError
from sentiment_agent.clock import ManualClock
from sentiment_agent.ledger.chain import HashChainLedger
from sentiment_agent.policy import POLICY_V1
from sentiment_agent.types import (
    AccountSnapshot,
    Activation,
    Amendment,
    AnchorRecord,
    BreakerTransition,
    BudgetState,
    CrowdReport,
    DecisionEvent,
    DecisionRecord,
    DryRunPreview,
    EnvironmentProof,
    EventKind,
    FeedAlarm,
    FeedHealthReport,
    Fill,
    FillVenue,
    Genesis,
    GuardId,
    GuardRuling,
    GuardStatus,
    HealthBeat,
    InstrumentRuling,
    KernelRuling,
    LedgerEvent,
    LlmCallRecord,
    LlmDecision,
    LlmOutcome,
    LlmUsage,
    MarketMood,
    MarkPoint,
    Model,
    Note,
    OrderIntent,
    OrderPlan,
    OrderPurpose,
    OrderState,
    OrderStateChange,
    OrderSubmitted,
    PerceptionSnapshot,
    PriceSource,
    ProtectiveAction,
    ProtectiveReason,
    ReconciliationReport,
    RunMode,
    Side,
    SnapshotEvent,
    SourceHealth,
    Stance,
    StopSync,
    TargetProposal,
    Thinking,
    ToolkitProbe,
    ToolkitSurface,
    Trigger,
    TriggerKind,
    VenueAck,
    VenueRejection,
    VenueUnknown,
    parse_payload,
)

NVDA = "NVDAUSDT"
BTC = "BTCUSDT"
DEMO = FillVenue.BITGET_DEMO
AT = T0 + 6 * H
MARKS = {NVDA: d(101), BTC: d(60100)}


# ------------------------------------------------------------------------------------------------
# Builders
# ------------------------------------------------------------------------------------------------


def decided(decision_id: str, symbol: str = NVDA) -> DecisionEvent:
    call = LlmCallRecord(
        model="qwen3.8-max",
        thinking=Thinking.LOW,
        prompt_version="system_v1",
        prompt_hash="0" * 64,
        request_blob=None,
        response_blobs=(),
        attempts=1,
        usage=LlmUsage(
            prompt_tokens=10,
            completion_tokens=5,
            reasoning_tokens=0,
            total_tokens=15,
            reported=True,
        ),
        latency_ms=10,
        outcome=LlmOutcome.DECIDED,
    )
    decision = LlmDecision(
        stance=Stance.ACT,
        targets=(
            TargetProposal(
                symbol=symbol,
                target=1.0,
                thesis="crowd is short",
                invalidation="funding flips",
                horizon_hours=24,
                crowd_belief="down",
                our_view="up",
                confidence=0.55,
            ),
        ),
        rejected_alternatives=(),
        mandate_response="deployed 5%",
        summary="long",
    )
    record = DecisionRecord(
        decision_id=decision_id,
        decided_at=T0,
        trigger_ids=("t1",),
        snapshot_id="snap-1",
        book_before=empty_book(at=T0),
        mandate=POLICY_V1.mandate,
        policy_version=POLICY_V1.version,
        call=call,
        outcome=LlmOutcome.DECIDED,
        decision=decision,
        grounding={},
        proposed_weights={symbol: 0.05},
    )
    return DecisionEvent(record=record)


def model_ruling(ruling_id: str, decision_id: str, symbol: str, weight: float) -> KernelRuling:
    return KernelRuling(
        ruling_id=ruling_id,
        at=T0,
        decision_id=decision_id,
        protective_reason=None,
        activation_before=Activation.ACTIVE,
        activation_after=Activation.ACTIVE,
        book_rulings=(),
        instruments=(
            InstrumentRuling(
                symbol=symbol,
                current_weight=0.0,
                proposed_weight=weight,
                approved_weight=weight,
                binding_guard=None,
                rulings=(),
            ),
        ),
        guards_applied=(),
    )


def daily_kill_ruling(ruling_id: str, symbol: str, current: float) -> KernelRuling:
    return KernelRuling(
        ruling_id=ruling_id,
        at=T0 + 4 * H,
        decision_id=None,
        protective_reason=ProtectiveReason.DAILY_KILL,
        activation_before=Activation.ACTIVE,
        activation_after=Activation.HALTED,
        book_rulings=(
            GuardRuling(
                guard=GuardId.G5_DAILY_KILL,
                symbol=None,
                status=GuardStatus.FIRED,
                forces_exit=True,
                reason="book down 1.6% on the day",
                basis="envelope_clean.py",
            ),
        ),
        instruments=(
            InstrumentRuling(
                symbol=symbol,
                current_weight=current,
                proposed_weight=None,
                approved_weight=0.0,
                binding_guard=GuardId.G5_DAILY_KILL,
                rulings=(),
            ),
        ),
        guards_applied=(GuardId.G5_DAILY_KILL,),
    )


def weekend_ruling(ruling_id: str, decision_id: str, symbol: str) -> KernelRuling:
    return KernelRuling(
        ruling_id=ruling_id,
        at=T0,
        decision_id=decision_id,
        protective_reason=None,
        activation_before=Activation.ACTIVE,
        activation_after=Activation.ACTIVE,
        book_rulings=(),
        instruments=(
            InstrumentRuling(
                symbol=symbol,
                current_weight=0.05,
                proposed_weight=0.05,
                approved_weight=0.0,
                binding_guard=GuardId.G2_WEEKEND_FREEZE,
                rulings=(
                    GuardRuling(
                        guard=GuardId.G2_WEEKEND_FREEZE,
                        symbol=symbol,
                        status=GuardStatus.FIRED,
                        forces_exit=True,
                        reason="weekend freeze",
                        basis="weekend_vol.json",
                    ),
                ),
            ),
        ),
        guards_applied=(GuardId.G2_WEEKEND_FREEZE,),
    )


def plan(ruling_id: str, *intents: OrderIntent) -> OrderPlan:
    return OrderPlan(
        plan_id=f"plan-{ruling_id}", ruling_id=ruling_id, created_at=T0, intents=intents, skipped=()
    )


def submitted(intent: OrderIntent, at: datetime = T0) -> OrderSubmitted:
    return OrderSubmitted(
        client_oid=intent.client_oid,
        intent=intent,
        approval_hash="1" * 64,
        submitted_at=at,
        argv=("order", "--action", "place", "--paper-trading"),
    )


def stop_sync(action: str, price: str | None, venue_id: str | None, at: datetime) -> StopSync:
    return StopSync.model_validate(
        {
            "symbol": NVDA,
            "action": action,
            "stop_price": None if price is None else Decimal(price),
            "venue_id": venue_id,
            "at": at,
        }
    )


def snapshot() -> SnapshotEvent:
    return SnapshotEvent(
        snapshot=PerceptionSnapshot(
            snapshot_id="",
            taken_at=T0,
            mode=RunMode.PAPER,
            policy_version=POLICY_V1.version,
            universe=POLICY_V1.symbols,
            demo_quotes={},
            live_quotes={},
            features={},
            mood=MarketMood(),
            crowd=CrowdReport(
                items=0,
                withheld=0,
                distinct_stories=0,
                duplication_ratio=0.0,
                clusters=(),
                mentions={},
            ),
            text=(),
            calendar=(),
            source_calls=(),
            facts={},
        )
    )


def account(equity: str, at: datetime) -> AccountSnapshot:
    return AccountSnapshot(at=at, equity_usdt=Decimal(equity), available_usdt=None, blob=None)


def reconciliation(equity: str | None, at: datetime) -> ReconciliationReport:
    return ReconciliationReport(
        at=at,
        orders_checked=1,
        new_fill_ids=(),
        resolved_unknown=(),
        discrepancies=(),
        account=None if equity is None else account(equity, at),
    )


def transition(frm: Activation, to: Activation, at: datetime) -> BreakerTransition:
    return BreakerTransition(at=at, from_state=frm, to_state=to, trips=("test",))


# ------------------------------------------------------------------------------------------------
# The scenario: a PAPER ledger with every event kind
# ------------------------------------------------------------------------------------------------

INTENT_A = make_intent(symbol=NVDA, side=Side.BUY, qty="2", price="100", stop="96")
INTENT_B = make_intent(
    symbol=BTC, side=Side.SELL, qty="0.01", price="60000", ruling_id="ruling-2", decision_id="d2"
)
INTENT_C = make_intent(
    symbol=BTC,
    side=Side.BUY,
    qty="0.01",
    price="60000",
    purpose=OrderPurpose.CLOSE,
    ruling_id="ruling-3",
    decision_id=None,
)
FILL_A1 = make_fill(
    Side.BUY,
    "1",
    "100.1",
    at=T0 + timedelta(seconds=1),
    client_oid=INTENT_A.client_oid,
    venue_order_id="o-A",
    venue=DEMO,
)
FILL_A2 = make_fill(
    Side.BUY,
    "1",
    "100.2",
    at=T0 + timedelta(seconds=1),
    client_oid=INTENT_A.client_oid,
    venue_order_id="o-A",
    venue=DEMO,
)
SYNC_REPLACED = stop_sync("replaced", "96.15", "stop-1", T0 + timedelta(seconds=30))
FILL_STOP = make_fill(
    Side.SELL, "2", "96.1", at=T0 + 2 * H, client_oid=None, venue_order_id="stop-1", venue=DEMO
)
FILL_B = make_fill(
    Side.SELL,
    "0.01",
    "60000",
    symbol=BTC,
    at=T0 + 3 * H,
    client_oid=INTENT_B.client_oid,
    venue=DEMO,
)
FILL_C = make_fill(
    Side.BUY,
    "0.01",
    "60900",
    symbol=BTC,
    at=T0 + 4 * H,
    client_oid=INTENT_C.client_oid,
    venue=DEMO,
)
MARK_1 = make_mark(T0 + H, "10001.57982")


def paper_ledger() -> MemoryLedger:
    ledger = MemoryLedger(RunMode.PAPER)
    at = T0 - H
    ledger.append(EventKind.GENESIS, make_genesis())
    ledger.append(EventKind.ENVIRONMENT_PROOF, make_proof("10000", at=at))
    ledger.append(EventKind.TOOLKIT_PROBE, ToolkitProbe(at=at, rows=()))
    ledger.append(EventKind.SNAPSHOT, snapshot())
    ledger.append(
        EventKind.TRIGGER,
        Trigger(
            trigger_id="t1",
            kind=TriggerKind.HEARTBEAT_US_OPEN,
            fired_at=T0,
            symbols=(),
            detail="09:30 New York",
            source="schedule",
        ),
    )
    ledger.append(EventKind.DECISION, decided("d1"))
    ledger.append(EventKind.KERNEL_RULING, model_ruling("ruling-1", "d1", NVDA, 0.05))
    ledger.append(EventKind.ORDER_PLAN, plan("ruling-1", INTENT_A))
    ledger.append(
        EventKind.ORDER_PREVIEW,
        DryRunPreview(
            client_oid=INTENT_A.client_oid,
            operation_id="placeOrder",
            method="POST",
            path="/api/v3/trade/place-order",
            would_send={"symbol": NVDA},
            argv=("order", "--dry-run", "--paper-trading"),
            captured_at=T0,
            blob=None,
        ),
    )
    ledger.append(EventKind.ORDER_SUBMITTED, submitted(INTENT_A))
    ledger.append(
        EventKind.ORDER_ACK,
        VenueAck(client_oid=INTENT_A.client_oid, venue_order_id="o-A", acked_at=T0, blob=None),
    )
    ledger.append(EventKind.FILL, FILL_A1)
    ledger.append(EventKind.STOP_SYNC, SYNC_REPLACED)
    ledger.append(EventKind.FILL, FILL_A2)  # a later fill of the same order: no second preset
    ledger.append(
        EventKind.ORDER_STATE,
        OrderStateChange(
            client_oid=INTENT_A.client_oid,
            venue_order_id="o-A",
            from_state=OrderState.ACCEPTED,
            to_state=OrderState.FILLED,
            at=T0,
            reason="filled",
        ),
    )
    ledger.append(EventKind.MARK, MARK_1)
    ledger.append(
        EventKind.BREAKER_TRANSITION,
        transition(Activation.ACTIVE, Activation.REDUCE_ONLY, T0 + H),
    )
    ledger.append(
        EventKind.BUDGET_STATE,
        BudgetState(
            day="2026-09-23", cap_tokens=150_000, spent_tokens=15, calls=1, unreported_calls=0
        ),
    )
    ledger.append(EventKind.NOTE, Note(at=T0 + H, author="system", text="started"))
    ledger.append(
        EventKind.HEALTH,
        HealthBeat(
            at=T0 + H,
            iteration=1,
            activation=Activation.REDUCE_ONLY,
            open_positions=1,
            last_decision_at=T0,
            budget=None,
        ),
    )
    ledger.append(
        EventKind.FEED_HEALTH,
        FeedHealthReport(
            at=T0 + H,
            snapshot_id="snap-feeds",
            light=True,
            alarms=(
                FeedAlarm(
                    feed="sentiment_index.current",
                    surface=ToolkitSurface.SIGNAL_MCP,
                    health=SourceHealth.HOLLOW,
                    since=T0,
                    snapshots=2,
                ),
            ),
        ),
    )
    ledger.append(EventKind.FILL, FILL_STOP)  # the venue stop, Bitget reusing the stop's id
    ledger.append(EventKind.RECONCILIATION, reconciliation("9991.5", T0 + 2 * H))
    ledger.append(
        EventKind.PROTECTIVE_ACTION,
        ProtectiveAction(
            at=T0 + 2 * H,
            reason=ProtectiveReason.STOP_FILLED,
            symbols=(NVDA,),
            ruling_id="none",
            detail="venue stop filled",
        ),
    )
    ledger.append(EventKind.DECISION, decided("d2", BTC))
    ledger.append(EventKind.KERNEL_RULING, model_ruling("ruling-2", "d2", BTC, -0.05))
    ledger.append(EventKind.ORDER_PLAN, plan("ruling-2", INTENT_B))
    ledger.append(EventKind.ORDER_SUBMITTED, submitted(INTENT_B, T0 + 3 * H))
    ledger.append(EventKind.FILL, FILL_B)
    ledger.append(EventKind.KERNEL_RULING, daily_kill_ruling("ruling-3", BTC, -0.06))
    ledger.append(EventKind.ORDER_PLAN, plan("ruling-3", INTENT_C))
    ledger.append(
        EventKind.ORDER_REJECTED,
        VenueRejection(
            client_oid=INTENT_C.client_oid,
            code="40762",
            message="insufficient",
            category="param",
            retryable=True,
            at=T0 + 4 * H,
            blob=None,
        ),
    )
    ledger.append(
        EventKind.ORDER_UNKNOWN,
        VenueUnknown(client_oid=INTENT_C.client_oid, at=T0 + 4 * H, reason="timeout"),
    )
    ledger.append(EventKind.FILL, FILL_C)
    ledger.append(
        EventKind.ANCHOR,
        AnchorRecord(
            target_seq=0,
            target_hash="a" * 64,
            submitted_at=T0 + 4 * H,
            status="submitted",
            ots_blob=None,
            detail="",
        ),
    )
    v2 = POLICY_V1.model_copy(update={"version": "policy-v2"})
    ledger.append(
        EventKind.AMENDMENT,
        Amendment(
            amendment_id="a1",
            at=T0 + 5 * H,
            reason="test",
            previous_policy_hash=POLICY_V1.content_hash(),
            new_policy_hash=v2.content_hash(),
            new_policy=v2,
            owner_confirmed=True,
        ),
    )
    ledger.append(
        EventKind.BREAKER_TRANSITION,
        transition(Activation.REDUCE_ONLY, Activation.ACTIVE, T0 + 5 * H),
    )
    return ledger


def by_hand() -> BookBuilder:
    """The book the scenario's log implies, driven directly."""
    b = BookBuilder(starting_equity=d(10000), policy=POLICY_V1)
    b.apply_fill(FILL_A1, decision_id="d1", purpose=OrderPurpose.OPEN)
    b.apply_stop_sync(stop_sync("preset", "96", None, FILL_A1.executed_at))
    b.apply_stop_sync(SYNC_REPLACED)
    b.apply_fill(FILL_A2, decision_id="d1", purpose=OrderPurpose.OPEN)
    b.record_mark(MARK_1)
    b.apply_fill(FILL_STOP, decision_id=None, purpose=None, cause=ProtectiveReason.STOP_FILLED)
    b.apply_fill(FILL_B, decision_id="d2", purpose=OrderPurpose.OPEN)
    b.apply_stop_sync(
        StopSync(
            symbol=BTC,
            action="preset",
            stop_price=INTENT_B.stop_loss_price,
            venue_id=None,
            at=FILL_B.executed_at,
        )
    )
    b.apply_fill(
        FILL_C, decision_id=None, purpose=OrderPurpose.CLOSE, cause=ProtectiveReason.DAILY_KILL
    )
    return b


# ------------------------------------------------------------------------------------------------
# Replay equals incremental equals by hand
# ------------------------------------------------------------------------------------------------


def test_replay_equals_incremental_equals_by_hand() -> None:
    ledger = paper_ledger()
    replayed = Projection.from_ledger(ledger, POLICY_V1)
    incremental = Projection(POLICY_V1)
    for event in ledger.events():
        incremental.apply(event)
        if incremental.starting_equity is not None:
            incremental.book(at=AT, marks=MARKS, mark_source=PriceSource.DEMO)  # fed as it goes
    book = replayed.book(at=AT, marks=MARKS, mark_source=PriceSource.DEMO)
    assert incremental.book(at=AT, marks=MARKS, mark_source=PriceSource.DEMO) == book
    assert incremental.closed_trades == replayed.closed_trades
    hand = by_hand().state(
        at=AT, marks=MARKS, mark_source=PriceSource.DEMO, activation=Activation.ACTIVE
    )
    assert book == hand
    assert replayed.closed_trades == by_hand().closed_trades()


def test_every_prefix_of_the_log_replays_to_the_incremental_state() -> None:
    ledger = paper_ledger()
    incremental = Projection(POLICY_V1)
    for n, event in enumerate(ledger.events(), start=1):
        incremental.apply(event)
        replayed = Projection.from_ledger(ledger.prefix(n), POLICY_V1)
        if replayed.starting_equity is None:
            with pytest.raises(ProjectionError, match="starting equity"):
                replayed.book(at=AT, marks=MARKS, mark_source=PriceSource.DEMO)
            continue
        for source in (PriceSource.DEMO, PriceSource.LIVE):
            assert incremental.book(at=AT, marks=MARKS, mark_source=source) == replayed.book(
                at=AT, marks=MARKS, mark_source=source
            ), n
        assert incremental.closed_trades == replayed.closed_trades


def test_the_scenario_book_is_what_the_log_says() -> None:
    projection = Projection.from_ledger(paper_ledger(), POLICY_V1)
    stop_trade, kill_trade = projection.closed_trades
    # NVDA: bought 1 @ 100.1 and 1 @ 100.2, stopped out 2 @ 96.1 by the venue stop
    assert stop_trade.symbol == NVDA
    assert stop_trade.exit_reason == EXIT_STOP
    assert stop_trade.gross_pnl == d("-8.1")  # (96.1 - 100.15) x 2
    assert stop_trade.decision_ids == ("d1",)
    # BTC: short 0.01 @ 60,000, closed by the daily kill at 60,900
    assert kill_trade.symbol == BTC
    assert kill_trade.direction == -1
    assert kill_trade.exit_reason == "protective_exit:daily_kill"
    assert kill_trade.gross_pnl == d(-9)
    book = projection.book(at=AT, marks=MARKS, mark_source=PriceSource.DEMO)
    assert book.positions == {}
    assert book.starting_equity == d(10000)
    assert book.equity == d(10000) + stop_trade.net_pnl + kill_trade.net_pnl
    assert book.peak_equity == d("10001.57982")  # the 14:00 mark
    assert book.rebalances_today == {BTC: 1, NVDA: 1}  # the kill's close is not a rebalance
    assert book.consecutive_losses == 2
    assert book.activation is Activation.ACTIVE
    assert projection.starting_equity == d(10000)  # the reconciliation after a fill is not used


def test_a_book_as_of_an_earlier_time_uses_only_what_had_happened() -> None:
    projection = Projection.from_ledger(paper_ledger(), POLICY_V1)
    at = T0 + H + H / 2
    book = projection.book(at=at, marks={NVDA: d(99)}, mark_source=PriceSource.DEMO)
    position = book.positions[NVDA]
    assert position.qty == d(2)
    assert position.avg_entry == d("100.15")
    # the preset (96) was replaced by the synced stop, and the order's second fill did not
    # bring the preset back
    assert (position.stop_price, position.stop_venue_id) == (d("96.15"), "stop-1")
    assert book.activation is Activation.REDUCE_ONLY
    assert projection.activation(at) is Activation.REDUCE_ONLY
    assert projection.activation() is Activation.ACTIVE
    # the main book is untouched by the look back
    assert projection.book(at=AT, marks=MARKS, mark_source=PriceSource.DEMO).positions == {}


# ------------------------------------------------------------------------------------------------
# Typed views
# ------------------------------------------------------------------------------------------------


def test_every_event_kind_is_readable_as_its_typed_payload() -> None:
    ledger = paper_ledger()
    projection = Projection.from_ledger(ledger, POLICY_V1)
    assert len(projection) == len(ledger)
    assert [e.seq for e in projection] == list(range(len(ledger)))
    assert projection.head_seq == len(ledger) - 1
    assert projection.mode is RunMode.PAPER
    assert isinstance(projection.genesis, Genesis)
    views: Sequence[tuple[tuple[object, ...], type[object], int]] = [
        (projection.amendments, Amendment, 1),
        (projection.environment_proofs, EnvironmentProof, 1),
        (projection.toolkit_probes, ToolkitProbe, 1),
        (projection.snapshots, PerceptionSnapshot, 1),
        (projection.triggers, Trigger, 1),
        (projection.decisions, DecisionRecord, 2),
        (projection.rulings, KernelRuling, 3),
        (projection.plans, OrderPlan, 3),
        (projection.previews, DryRunPreview, 1),
        (projection.submissions, OrderSubmitted, 2),
        (projection.acks, VenueAck, 1),
        (projection.rejections, VenueRejection, 1),
        (projection.unknowns, VenueUnknown, 1),
        (projection.order_states, OrderStateChange, 1),
        (projection.fills, Fill, 5),
        (projection.stop_syncs, StopSync, 1),
        (projection.protective_actions, ProtectiveAction, 1),
        (projection.breaker_transitions, BreakerTransition, 2),
        (projection.reconciliations, ReconciliationReport, 1),
        (projection.marks, MarkPoint, 1),
        (projection.budget_states, BudgetState, 1),
        (projection.anchors, AnchorRecord, 1),
        (projection.health_beats, HealthBeat, 1),
        (projection.feed_health, FeedHealthReport, 1),
        (projection.notes, Note, 1),
    ]
    for items, model, count in views:
        assert len(items) == count, model
        assert all(isinstance(item, model) for item in items), model
    assert projection.fills == (FILL_A1, FILL_A2, FILL_STOP, FILL_B, FILL_C)
    kinds_seen = {e.kind for e in projection}
    assert kinds_seen == set(EventKind)


def test_order_events_come_back_in_ledger_order_with_their_raw_events() -> None:
    projection = Projection.from_ledger(paper_ledger(), POLICY_V1)
    assert [type(e) for e in projection.order_events] == [
        DryRunPreview,
        OrderSubmitted,
        VenueAck,
        OrderStateChange,
        OrderSubmitted,
        VenueRejection,
        VenueUnknown,
    ]
    raw = projection.events(ORDER_EVENT_KINDS)
    assert [e.kind for e in raw] == [
        EventKind.ORDER_PREVIEW,
        EventKind.ORDER_SUBMITTED,
        EventKind.ORDER_ACK,
        EventKind.ORDER_STATE,
        EventKind.ORDER_SUBMITTED,
        EventKind.ORDER_REJECTED,
        EventKind.ORDER_UNKNOWN,
    ]
    assert all(isinstance(e, LedgerEvent) for e in raw)
    assert len(projection.events()) == len(projection)


def test_intents_rulings_and_decisions_can_be_looked_up() -> None:
    projection = Projection.from_ledger(paper_ledger(), POLICY_V1)
    assert projection.intent(INTENT_A.client_oid) == INTENT_A
    assert projection.intent("sa" + "f" * 30) is None
    ruling = projection.ruling("ruling-3")
    assert ruling is not None
    assert ruling.protective_reason is ProtectiveReason.DAILY_KILL
    decision = projection.decision("d2")
    assert decision is not None
    assert decision.proposed_weights == {BTC: 0.05}
    assert projection.decision("nope") is None


# ------------------------------------------------------------------------------------------------
# Starting equity and scope
# ------------------------------------------------------------------------------------------------


def one_fill_ledger(mode: RunMode, *before: tuple[EventKind, Model]) -> MemoryLedger:
    ledger = MemoryLedger(mode)
    for kind, payload in before:
        ledger.append(kind, payload)
    venue = DEMO if mode is RunMode.PAPER else FillVenue.SIMULATED
    ledger.append(EventKind.FILL, make_fill(Side.BUY, "1", "100", at=T0, venue=venue))
    return ledger


def test_a_failed_proofs_account_is_never_the_starting_equity() -> None:
    failed = make_proof("99999", passed=False)
    assert not failed.passed
    ledger = one_fill_ledger(
        RunMode.PAPER,
        (EventKind.ENVIRONMENT_PROOF, failed),
        (EventKind.ENVIRONMENT_PROOF, make_proof("10000")),
    )
    assert Projection.from_ledger(ledger, POLICY_V1).starting_equity == d(10000)


def test_a_paper_ledger_without_an_account_read_before_its_first_fill_has_no_book() -> None:
    ledger = one_fill_ledger(RunMode.PAPER)
    ledger.append(EventKind.ENVIRONMENT_PROOF, make_proof("10000", at=T0 + H))
    projection = Projection.from_ledger(ledger, POLICY_V1, starting_equity=d(5000))
    assert projection.starting_equity is None  # the fallback never applies to PAPER
    with pytest.raises(ProjectionError, match="PAPER ledger records no account equity"):
        projection.book(at=AT, marks=MARKS, mark_source=PriceSource.DEMO)


def test_the_latest_account_read_before_the_first_fill_is_the_starting_equity() -> None:
    # 2026-09-24: an early Demo read valued gifted coins at 3.78M; the read just before trading
    # was 50,000 USDT. A superseded pre-trade read is stale, so the latest one starts the book.
    ledger = one_fill_ledger(
        RunMode.PAPER,
        (EventKind.ENVIRONMENT_PROOF, make_proof("3782464.95")),
        (EventKind.ENVIRONMENT_PROOF, make_proof("50000")),
    )
    assert Projection.from_ledger(ledger, POLICY_V1).starting_equity == d(50000)


def test_an_account_read_after_the_first_fill_never_moves_the_start() -> None:
    ledger = one_fill_ledger(RunMode.PAPER, (EventKind.ENVIRONMENT_PROOF, make_proof("50000")))
    ledger.append(EventKind.ENVIRONMENT_PROOF, make_proof("61000", at=T0 + H))
    assert Projection.from_ledger(ledger, POLICY_V1).starting_equity == d(50000)


def test_a_non_positive_account_read_is_skipped() -> None:
    ledger = one_fill_ledger(
        RunMode.PAPER,
        (EventKind.ENVIRONMENT_PROOF, make_proof("0")),
        (EventKind.RECONCILIATION, reconciliation("10000", T0 - H)),
    )
    assert Projection.from_ledger(ledger, POLICY_V1).starting_equity == d(10000)


def test_a_simulated_ledger_reads_a_reconciliation_or_falls_back() -> None:
    recorded = one_fill_ledger(
        RunMode.SIMULATED, (EventKind.RECONCILIATION, reconciliation("25000", T0 - H))
    )
    projection = Projection.from_ledger(recorded, POLICY_V1, starting_equity=d(1))
    assert projection.starting_equity == d(25000)  # the log wins over the fallback
    bare = one_fill_ledger(RunMode.SIMULATED, (EventKind.RECONCILIATION, reconciliation(None, T0)))
    assert Projection.from_ledger(bare, POLICY_V1, starting_equity=d(7)).starting_equity == d(7)
    with pytest.raises(ProjectionError, match="no fallback"):
        Projection.from_ledger(bare, POLICY_V1).book(
            at=AT, marks=MARKS, mark_source=PriceSource.DEMO
        )


def test_nothing_before_the_genesis_enters_the_book() -> None:
    ledger = MemoryLedger(RunMode.PAPER)
    ledger.append(EventKind.ENVIRONMENT_PROOF, make_proof("10000"))
    plumbing = make_fill(Side.BUY, "0.001", "60000", symbol=BTC, at=T0 - 2 * H, venue=DEMO)
    ledger.append(EventKind.FILL, plumbing)
    ledger.append(EventKind.GENESIS, make_genesis())
    ledger.append(EventKind.ENVIRONMENT_PROOF, make_proof("9999.9"))
    ledger.append(EventKind.FILL, make_fill(Side.BUY, "1", "100", at=T0, venue=DEMO))
    projection = Projection.from_ledger(ledger, POLICY_V1)
    assert projection.starting_equity == d("9999.9")
    book = projection.book(at=AT, marks=MARKS, mark_source=PriceSource.DEMO)
    assert set(book.positions) == {NVDA}
    assert projection.fills[0] == plumbing  # still visible in the log's view


# ------------------------------------------------------------------------------------------------
# Attribution
# ------------------------------------------------------------------------------------------------


def paper_with(*events: tuple[EventKind, Model]) -> Projection:
    ledger = MemoryLedger(RunMode.PAPER)
    ledger.append(EventKind.GENESIS, make_genesis())
    ledger.append(EventKind.ENVIRONMENT_PROOF, make_proof("10000"))
    for kind, payload in events:
        ledger.append(kind, payload)
    return Projection.from_ledger(ledger, POLICY_V1)


def test_a_guard_forced_exit_inside_a_decision_names_its_guard_and_is_not_a_rebalance() -> None:
    open_ = make_intent(seed="open")
    forced = make_intent(
        side=Side.SELL,
        purpose=OrderPurpose.PROTECTIVE_EXIT,
        ruling_id="ruling-w",
        decision_id="d9",
        seed="forced",
    )
    projection = paper_with(
        (EventKind.ORDER_PLAN, plan("ruling-1", open_)),
        (EventKind.FILL, make_fill(Side.BUY, "2", "100", client_oid=open_.client_oid, venue=DEMO)),
        (EventKind.KERNEL_RULING, weekend_ruling("ruling-w", "d9", NVDA)),
        (EventKind.ORDER_PLAN, plan("ruling-w", forced)),
        (
            EventKind.FILL,
            make_fill(Side.SELL, "2", "99", at=T0 + H, client_oid=forced.client_oid, venue=DEMO),
        ),
    )
    (trade,) = projection.closed_trades
    assert trade.exit_reason == "protective_exit:G2_weekend_freeze"
    assert trade.decision_ids == ("d1", "d9")
    book = projection.book(at=AT, marks=MARKS, mark_source=PriceSource.DEMO)
    assert book.rebalances_today == {NVDA: 1}


def test_a_venue_fill_that_is_not_a_known_stop_is_venue_initiated() -> None:
    open_ = make_intent(seed="open")
    projection = paper_with(
        (EventKind.ORDER_SUBMITTED, submitted(open_)),
        (EventKind.FILL, make_fill(Side.BUY, "2", "100", client_oid=open_.client_oid, venue=DEMO)),
        (EventKind.STOP_SYNC, stop_sync("placed", "96", "stop-9", T0 + timedelta(seconds=5))),
        (
            EventKind.FILL,
            make_fill(Side.SELL, "2", "95", at=T0 + H, venue_order_id="manual-1", venue=DEMO),
        ),
    )
    (trade,) = projection.closed_trades
    assert trade.exit_reason == EXIT_VENUE_INITIATED  # never passed off as the stop


def test_the_preset_stop_of_an_opening_order_is_set_on_the_position() -> None:
    open_ = make_intent(seed="open", stop="96.5")
    projection = paper_with(
        (EventKind.ORDER_PLAN, plan("ruling-1", open_)),
        (EventKind.FILL, make_fill(Side.BUY, "2", "100", client_oid=open_.client_oid, venue=DEMO)),
    )
    position = projection.book(at=AT, marks=MARKS, mark_source=PriceSource.DEMO).positions[NVDA]
    assert (position.stop_price, position.stop_venue_id) == (d("96.5"), None)
    assert position.last_decision_id == "d1"


def test_a_fill_that_contradicts_its_intent_is_refused() -> None:
    open_ = make_intent(seed="open")
    with pytest.raises(ProjectionError, match="was booked under"):
        paper_with(
            (EventKind.ORDER_PLAN, plan("ruling-1", open_)),
            (
                EventKind.FILL,
                make_fill(Side.SELL, "2", "100", client_oid=open_.client_oid, venue=DEMO),
            ),
        )
    with pytest.raises(ProjectionError, match="was booked under"):
        paper_with(
            (EventKind.ORDER_PLAN, plan("ruling-1", open_)),
            (
                EventKind.FILL,
                make_fill(
                    Side.BUY, "1", "60000", symbol=BTC, client_oid=open_.client_oid, venue=DEMO
                ),
            ),
        )


def test_one_client_oid_cannot_name_two_intents() -> None:
    first = make_intent(seed="same")
    second = first.model_copy(update={"qty": d(3), "notional": d(300)})
    assert second.client_oid == first.client_oid
    with pytest.raises(ProjectionError, match="two different intents"):
        paper_with(
            (EventKind.ORDER_PLAN, plan("ruling-1", first)),
            (EventKind.ORDER_SUBMITTED, submitted(second)),
        )


# ------------------------------------------------------------------------------------------------
# Logs the projection refuses
# ------------------------------------------------------------------------------------------------


def test_events_must_arrive_in_seq_order() -> None:
    ledger = paper_ledger()
    first, second = list(ledger.events())[:2]
    projection = Projection(POLICY_V1)
    projection.apply(first)
    projection.apply(second)
    for event in (second, first):
        with pytest.raises(ProjectionError, match="in order"):
            projection.apply(event)


def test_one_ledger_holds_one_mode() -> None:
    ledger = MemoryLedger(RunMode.PAPER)
    ledger.append(EventKind.GENESIS, make_genesis())
    ledger.append(EventKind.NOTE, Note(at=T0, author="system", text="x"), mode=RunMode.SIMULATED)
    with pytest.raises(ProjectionError, match="modes never share"):
        Projection.from_ledger(ledger, POLICY_V1)


@pytest.mark.parametrize(
    ("mode", "venue"),
    [
        (RunMode.PAPER, FillVenue.SIMULATED),
        (RunMode.SIMULATED, FillVenue.BITGET_DEMO),
        (RunMode.DRYRUN, FillVenue.SIMULATED),
        (RunMode.DRYRUN, FillVenue.BITGET_DEMO),
    ],
)
def test_a_fill_from_the_wrong_venue_for_the_mode_is_refused(
    mode: RunMode, venue: FillVenue
) -> None:
    ledger = MemoryLedger(mode)
    ledger.append(EventKind.FILL, make_fill(Side.BUY, "1", "100", venue=venue))
    with pytest.raises(ProjectionError, match="fill"):
        Projection.from_ledger(ledger, POLICY_V1)


def test_an_inconsistent_breaker_log_reads_as_halted() -> None:
    projection = paper_with(
        (EventKind.BREAKER_TRANSITION, transition(Activation.ACTIVE, Activation.HALTED, T0)),
        (EventKind.BREAKER_TRANSITION, transition(Activation.ACTIVE, Activation.ACTIVE, T0 + H)),
    )
    assert projection.activation() is Activation.HALTED
    backwards = paper_with(
        (EventKind.BREAKER_TRANSITION, transition(Activation.ACTIVE, Activation.HALTED, T0 + H)),
        (EventKind.BREAKER_TRANSITION, transition(Activation.HALTED, Activation.REDUCE_ONLY, T0)),
    )
    assert backwards.activation() is Activation.HALTED
    assert paper_with().activation() is Activation.ACTIVE


def test_an_empty_ledger_projects_nothing() -> None:
    projection = Projection.from_ledger(MemoryLedger(), POLICY_V1)
    assert len(projection) == 0
    assert projection.head_seq is None
    assert projection.mode is None
    assert projection.genesis is None
    assert projection.starting_equity is None
    assert projection.fills == ()
    assert projection.activation() is Activation.ACTIVE


# ------------------------------------------------------------------------------------------------
# The real ledger
# ------------------------------------------------------------------------------------------------


def test_the_hash_chained_ledger_on_disk_replays_to_the_same_book(
    tmp_path: Path, clock: ManualClock
) -> None:
    """Written through ``ledger/chain.py`` and read back from disk, as the runtime does after a
    restart; the runtime's own incremental projection (fed each appended event) agrees."""
    chain = HashChainLedger(tmp_path / "paper.jsonl", mode=RunMode.PAPER, clock=clock)
    live = Projection(POLICY_V1)
    for event in paper_ledger().events():
        live.apply(chain.append(event.kind, parse_payload(event)))
    restarted = Projection.from_ledger(
        HashChainLedger(tmp_path / "paper.jsonl", mode=RunMode.PAPER, clock=clock), POLICY_V1
    )
    expected = Projection.from_ledger(paper_ledger(), POLICY_V1)
    for source in (PriceSource.DEMO, PriceSource.LIVE):
        book = expected.book(at=AT, marks=MARKS, mark_source=source)
        assert restarted.book(at=AT, marks=MARKS, mark_source=source) == book
        assert live.book(at=AT, marks=MARKS, mark_source=source) == book
    assert restarted.closed_trades == expected.closed_trades
    assert restarted.fills == expected.fills


def test_the_midnight_mark_in_the_log_is_the_next_days_open() -> None:
    open_ = make_intent(seed="open")
    midnight = T0.replace(hour=0) + timedelta(days=1)
    projection = paper_with(
        (EventKind.ORDER_PLAN, plan("ruling-1", open_)),
        (EventKind.FILL, make_fill(Side.BUY, "2", "100", client_oid=open_.client_oid, venue=DEMO)),
        (EventKind.MARK, make_mark(midnight - H, "10020")),
        (EventKind.MARK, make_mark(midnight, "10011.88", mirror="10013.88")),
        (EventKind.MARK, make_mark(midnight + H, "10030")),
    )
    at = midnight + 2 * H
    demo = projection.book(at=at, marks=MARKS, mark_source=PriceSource.DEMO)
    assert demo.day_open_equity == d("10011.88")
    assert demo.peak_equity == d(10030)
    live = projection.book(at=at, marks=MARKS, mark_source=PriceSource.LIVE)
    assert live.day_open_equity == d("10013.88")
    assert live.peak_equity == d("10013.88")  # only the midnight mark carries a mirror


def test_a_book_built_on_the_fallback_is_rebuilt_when_the_log_records_the_account() -> None:
    ledger = MemoryLedger(RunMode.SIMULATED)
    live = Projection(POLICY_V1, starting_equity=d(5000))
    assert live.book(at=AT, marks=MARKS, mark_source=PriceSource.DEMO).equity == d(5000)
    for kind, payload in (
        (EventKind.RECONCILIATION, reconciliation("25000", T0 - H)),
        (EventKind.FILL, make_fill(Side.BUY, "1", "100", at=T0, fee="0")),
    ):
        live.apply(ledger.append(kind, payload))
    replayed = Projection.from_ledger(ledger, POLICY_V1, starting_equity=d(5000))
    book = replayed.book(at=AT, marks=MARKS, mark_source=PriceSource.DEMO)
    assert book.starting_equity == d(25000)
    assert live.book(at=AT, marks=MARKS, mark_source=PriceSource.DEMO) == book
