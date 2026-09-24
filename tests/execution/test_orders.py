"""Order state machine: the transition table, the tracker, and its rebuild from the ledger."""

from datetime import timedelta

import pytest

from execution.fakes import MemoryLedger
from helpers import T0, client_oid_for, make_intent
from sentiment_agent.clock import ManualClock
from sentiment_agent.execution.orders import (
    LEGAL_TRANSITIONS,
    DuplicateOrder,
    IllegalTransition,
    OrderTracker,
    map_venue_status,
)
from sentiment_agent.hashing import content_hash
from sentiment_agent.types import (
    EventKind,
    Fill,
    FillVenue,
    OrderState,
    OrderStateChange,
    OrderSubmitted,
    RunMode,
    Side,
    VenueOrderStatus,
)

A = client_oid_for("order-a")
B = client_oid_for("order-b")
C = client_oid_for("order-c")


def test_every_state_has_a_row_and_terminal_states_are_closed() -> None:
    assert set(LEGAL_TRANSITIONS) == set(OrderState)
    for state in OrderState:
        if state.is_terminal and state is not OrderState.FILLED:
            assert LEGAL_TRANSITIONS[state] == frozenset(), state
    # A venue can bust a fill after the fact: FILLED keeps exactly one exit.
    assert LEGAL_TRANSITIONS[OrderState.FILLED] == frozenset({OrderState.VOIDED})


def test_unknown_is_reachable_only_from_in_flight_or_working_states() -> None:
    sources = {s for s, targets in LEGAL_TRANSITIONS.items() if OrderState.UNKNOWN in targets}
    assert OrderState.INITIALISED not in sources
    assert not any(s.is_terminal for s in sources)
    assert OrderState.SUBMITTED in sources
    # ...and it is left only for a venue answer, never back to SUBMITTED (that would be a resend).
    assert OrderState.SUBMITTED not in LEGAL_TRANSITIONS[OrderState.UNKNOWN]
    assert OrderState.DENIED not in LEGAL_TRANSITIONS[OrderState.UNKNOWN]


def test_denied_is_ours_and_rejected_is_the_venues() -> None:
    assert OrderState.DENIED in LEGAL_TRANSITIONS[OrderState.INITIALISED]
    assert OrderState.REJECTED not in LEGAL_TRANSITIONS[OrderState.INITIALISED]
    assert OrderState.DENIED not in LEGAL_TRANSITIONS[OrderState.SUBMITTED]
    assert OrderState.REJECTED in LEGAL_TRANSITIONS[OrderState.SUBMITTED]


@pytest.mark.parametrize(
    ("venue", "ours"),
    [
        (VenueOrderStatus.LIVE, OrderState.ACCEPTED),
        (VenueOrderStatus.NEW, OrderState.ACCEPTED),
        (VenueOrderStatus.PARTIALLY_FILLED, OrderState.PARTIALLY_FILLED),
        (VenueOrderStatus.FILLED, OrderState.FILLED),
        (VenueOrderStatus.CANCELLED, OrderState.CANCELLED),
    ],
)
def test_map_venue_status(venue: VenueOrderStatus, ours: OrderState) -> None:
    assert map_venue_status(venue) is ours


def test_a_new_order_starts_from_initialised() -> None:
    tracker = OrderTracker()
    assert tracker.state(A) is None
    change = tracker.transition(A, OrderState.SUBMITTED, at=T0, reason="sent")
    assert change.from_state is OrderState.INITIALISED
    assert change.to_state is OrderState.SUBMITTED
    assert tracker.state(A) is OrderState.SUBMITTED


def test_submitting_a_known_order_is_a_duplicate() -> None:
    tracker = OrderTracker()
    tracker.transition(A, OrderState.SUBMITTED, at=T0, reason="sent")
    with pytest.raises(DuplicateOrder):
        tracker.transition(A, OrderState.SUBMITTED, at=T0, reason="again")
    tracker.transition(B, OrderState.DENIED, at=T0, reason="kernel")
    with pytest.raises(DuplicateOrder):
        tracker.transition(B, OrderState.SUBMITTED, at=T0, reason="after a denial")


def test_an_illegal_move_raises_and_changes_nothing() -> None:
    tracker = OrderTracker()
    tracker.transition(A, OrderState.SUBMITTED, at=T0, reason="sent")
    with pytest.raises(IllegalTransition):
        tracker.transition(A, OrderState.VOIDED, at=T0, reason="bad")
    assert tracker.state(A) is OrderState.SUBMITTED
    assert len(tracker.history(A)) == 1
    with pytest.raises(IllegalTransition):
        tracker.transition(C, OrderState.FILLED, at=T0, reason="never sent")
    assert tracker.state(C) is None


def test_live_includes_submitted_and_unknown_and_excludes_terminal() -> None:
    tracker = OrderTracker()
    tracker.transition(A, OrderState.SUBMITTED, at=T0, reason="sent")
    tracker.transition(B, OrderState.SUBMITTED, at=T0, reason="sent")
    tracker.transition(B, OrderState.UNKNOWN, at=T0, reason="timeout")
    tracker.transition(C, OrderState.SUBMITTED, at=T0, reason="sent")
    tracker.transition(C, OrderState.FILLED, at=T0, reason="filled")
    d = client_oid_for("order-d")
    tracker.transition(d, OrderState.DENIED, at=T0, reason="kernel")
    assert tracker.live() == [A, B]
    assert tracker.unknown() == [B]
    assert tracker.known() == [A, B, C, d]


def test_the_venue_order_id_is_carried_forward() -> None:
    tracker = OrderTracker()
    tracker.transition(A, OrderState.SUBMITTED, at=T0, reason="sent")
    tracker.transition(A, OrderState.ACCEPTED, at=T0, reason="ack", venue_order_id="123")
    change = tracker.transition(A, OrderState.FILLED, at=T0, reason="filled")
    assert change.venue_order_id == "123"
    assert tracker.venue_order_id(A) == "123"


def _fill(exec_id: str, oid: str) -> Fill:
    return Fill(
        exec_id=exec_id,
        venue_order_id="123",
        client_oid=oid,
        symbol="NVDAUSDT",
        side=Side.BUY,
        exec_price=make_intent().reference_price,
        exec_qty=make_intent().qty,
        exec_value=make_intent().notional,
        fee_paid=make_intent().expected_fee,
        fee_coin="USDT",
        trade_scope="taker",
        trade_side="open",
        exec_pnl=None,
        executed_at=T0,
        venue=FillVenue.SIMULATED,
    )


def test_restore_rebuilds_the_same_states_fills_and_submissions(clock: ManualClock) -> None:
    ledger = MemoryLedger(RunMode.SIMULATED, clock)
    live = OrderTracker()
    intent = make_intent()
    oid = intent.client_oid
    ledger.append(
        EventKind.ORDER_SUBMITTED,
        OrderSubmitted(
            client_oid=oid,
            intent=intent,
            approval_hash=content_hash("x"),
            submitted_at=T0,
            argv=("order",),
        ),
    )
    for to, reason in (
        (OrderState.SUBMITTED, "sent"),
        (OrderState.UNKNOWN, "timeout"),
        (OrderState.FILLED, "reconciled"),
    ):
        clock.advance(timedelta(seconds=1))
        ledger.append(
            EventKind.ORDER_STATE, live.transition(oid, to, at=clock.now(), reason=reason)
        )
    ledger.append(EventKind.FILL, _fill("f-1", oid))
    other = client_oid_for("in-flight")
    ledger.append(
        EventKind.ORDER_STATE, live.transition(other, OrderState.SUBMITTED, at=T0, reason="sent")
    )

    rebuilt = OrderTracker()
    rebuilt.restore(ledger.events())
    assert rebuilt.state(oid) is OrderState.FILLED
    assert rebuilt.state(other) is OrderState.SUBMITTED
    assert rebuilt.live() == [other]
    assert rebuilt.has_fill("f-1")
    submitted = rebuilt.submitted(oid)
    assert submitted is not None
    assert submitted.intent == intent
    assert [c.to_state for c in rebuilt.history(oid)] == [
        OrderState.SUBMITTED,
        OrderState.UNKNOWN,
        OrderState.FILLED,
    ]


def test_restore_refuses_a_ledger_that_contradicts_the_state_machine(clock: ManualClock) -> None:
    ledger = MemoryLedger(RunMode.SIMULATED, clock)
    ledger.append(
        EventKind.ORDER_STATE,
        OrderStateChange(
            client_oid=A,
            venue_order_id=None,
            from_state=OrderState.INITIALISED,
            to_state=OrderState.SUBMITTED,
            at=T0,
            reason="sent",
        ),
    )
    ledger.append(
        EventKind.ORDER_STATE,
        OrderStateChange(
            client_oid=A,
            venue_order_id=None,
            from_state=OrderState.SUBMITTED,
            to_state=OrderState.VOIDED,
            at=T0,
            reason="corrupt",
        ),
    )
    with pytest.raises(IllegalTransition):
        OrderTracker().restore(ledger.events())


def test_restore_refuses_a_change_that_starts_from_the_wrong_state(clock: ManualClock) -> None:
    ledger = MemoryLedger(RunMode.SIMULATED, clock)
    ledger.append(
        EventKind.ORDER_STATE,
        OrderStateChange(
            client_oid=A,
            venue_order_id=None,
            from_state=OrderState.ACCEPTED,
            to_state=OrderState.FILLED,
            at=T0,
            reason="skipped a step",
        ),
    )
    with pytest.raises(IllegalTransition):
        OrderTracker().restore(ledger.events())
