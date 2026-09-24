"""The breaker: transitions, latches, stepwise recovery and restore-after-crash."""

from datetime import UTC, datetime, timedelta

import pytest

from helpers import T0
from kernel.kbuild import book, breaker_state, inputs
from sentiment_agent.clock import ManualClock
from sentiment_agent.kernel.breaker import (
    AWAITING_CLEAN_DECISION,
    DAILY_KILL,
    DRAWDOWN_HALT,
    DRAWDOWN_REDUCE_ONLY,
    LLM_OUTAGE,
    LOSING_STREAK,
    STALE_SNAPSHOT,
    UNREADABLE_STATE,
    Breaker,
    book_conditions,
    most_severe,
    next_utc_midnight,
)
from sentiment_agent.policy import POLICY_V1
from sentiment_agent.types import Activation, BookState, BreakerTransition, KernelInputs

ACTIVE, REDUCE_ONLY, HALTED = Activation.ACTIVE, Activation.REDUCE_ONLY, Activation.HALTED
FRESH = inputs()


def _breaker(at: datetime = T0) -> tuple[Breaker, ManualClock]:
    clock = ManualClock(at)
    return Breaker(POLICY_V1, clock), clock


def _fresh(clock: ManualClock) -> KernelInputs:
    return inputs(snapshot_at=clock.now(), at=clock.now())


def test_most_severe_and_midnight() -> None:
    assert most_severe(ACTIVE, REDUCE_ONLY) is REDUCE_ONLY
    assert most_severe(REDUCE_ONLY, HALTED, ACTIVE) is HALTED
    assert next_utc_midnight(T0) == datetime(2026, 9, 24, tzinfo=UTC)


@pytest.mark.parametrize(
    ("b", "activation", "trips"),
    [
        (book(), ACTIVE, ()),
        (book(equity="9751", peak="10000"), ACTIVE, ()),
        (book(equity="9750", peak="10000"), REDUCE_ONLY, (DRAWDOWN_REDUCE_ONLY,)),
        (book(equity="9600", peak="10000"), HALTED, (DRAWDOWN_HALT,)),
        (book(losses=4), REDUCE_ONLY, (LOSING_STREAK,)),
        (book(losses=3), ACTIVE, ()),
        (book(equity="9850", day_open="10000", peak="10000"), HALTED, (DAILY_KILL,)),
    ],
)
def test_book_conditions(b: BookState, activation: Activation, trips: tuple[str, ...]) -> None:
    assert book_conditions(b, llm_outage=False, policy=POLICY_V1) == (activation, trips)


def test_book_conditions_evaluate_every_rule() -> None:
    b = book(equity="9600", peak="10000", day_open="9800", losses=5)
    activation, trips = book_conditions(b, llm_outage=True, policy=POLICY_V1)
    assert activation is HALTED
    assert set(trips) == {DRAWDOWN_HALT, DAILY_KILL, LLM_OUTAGE, LOSING_STREAK}
    _, without_kill = book_conditions(
        b, llm_outage=False, policy=POLICY_V1, include_daily_kill=False
    )
    assert DAILY_KILL not in without_kill


def test_no_transition_when_nothing_changes() -> None:
    brk, _ = _breaker()
    state, transition = brk.assess(book(), inputs=FRESH, llm_outage=False)
    assert transition is None
    assert state.activation is ACTIVE


def test_drawdown_ladder_escalates_immediately_and_recovers_only_after_a_clean_decision() -> None:
    brk, clock = _breaker()
    state, t = brk.assess(book(equity="9750", peak="10000"), inputs=FRESH, llm_outage=False)
    assert state.activation is REDUCE_ONLY
    assert t is not None
    assert (t.from_state, t.to_state, t.trips) == (ACTIVE, REDUCE_ONLY, (DRAWDOWN_REDUCE_ONLY,))
    state, t = brk.assess(book(equity="9600", peak="10000"), inputs=FRESH, llm_outage=False)
    assert state.activation is HALTED
    # Equity recovers to above the reduce-only line: HALTED steps down to REDUCE_ONLY only.
    clock.advance(timedelta(minutes=1))
    state, t = brk.assess(book(), inputs=_fresh(clock), llm_outage=False)
    assert state.activation is REDUCE_ONLY
    assert state.trips == (AWAITING_CLEAN_DECISION,)
    # The protective loop, with no decision, cannot re-arm the book.
    clock.advance(timedelta(minutes=1))
    state, t = brk.assess(book(), inputs=_fresh(clock), llm_outage=False)
    assert state.activation is REDUCE_ONLY
    assert t is None
    # One clean decision does.
    state, t = brk.assess(book(), inputs=_fresh(clock), llm_outage=False, decision_id="d7")
    assert state.activation is ACTIVE
    assert t is not None
    assert t.to_state is ACTIVE
    assert state.since == clock.now()


def test_losing_streak_holds_reduce_only_until_it_clears_and_a_decision_follows() -> None:
    brk, _ = _breaker()
    state, _ = brk.assess(book(losses=4), inputs=FRESH, llm_outage=False, decision_id="d1")
    assert state.activation is REDUCE_ONLY
    state, _ = brk.assess(book(losses=4), inputs=FRESH, llm_outage=False, decision_id="d2")
    assert state.activation is REDUCE_ONLY
    state, _ = brk.assess(book(losses=0), inputs=FRESH, llm_outage=False)
    assert (state.activation, state.trips) == (REDUCE_ONLY, (AWAITING_CLEAN_DECISION,))
    state, _ = brk.assess(book(losses=0), inputs=FRESH, llm_outage=False, decision_id="d3")
    assert state.activation is ACTIVE


def test_a_stale_snapshot_is_transient() -> None:
    brk, clock = _breaker()
    stale = inputs(snapshot_at=T0 - timedelta(minutes=16))
    state, t = brk.assess(book(), inputs=stale, llm_outage=False)
    assert (state.activation, state.trips) == (REDUCE_ONLY, (STALE_SNAPSHOT,))
    no_snapshot = inputs(snapshot_at=None)
    state, t = brk.assess(book(), inputs=no_snapshot, llm_outage=False)
    assert state.activation is REDUCE_ONLY
    assert t is None
    state, t = brk.assess(book(), inputs=_fresh(clock), llm_outage=False)
    assert state.activation is ACTIVE
    assert t is not None


def test_a_stale_snapshot_keeps_an_owed_clean_decision() -> None:
    brk, clock = _breaker()
    brk.assess(book(losses=4), inputs=FRESH, llm_outage=False)
    state, _ = brk.assess(book(), inputs=inputs(snapshot_at=None), llm_outage=False)
    assert state.trips == (STALE_SNAPSHOT, AWAITING_CLEAN_DECISION)
    state, _ = brk.assess(book(), inputs=_fresh(clock), llm_outage=False)
    assert (state.activation, state.trips) == (REDUCE_ONLY, (AWAITING_CLEAN_DECISION,))


def test_daily_kill_halts_until_the_next_utc_day_then_steps_down() -> None:
    brk, clock = _breaker(datetime(2026, 9, 23, 18, 0, tzinfo=UTC))
    killed = book(equity="9850", day_open="10000", peak="10000")
    state, t = brk.assess(killed, inputs=_fresh(clock), llm_outage=False)
    assert state.activation is HALTED
    assert state.trips == (DAILY_KILL,)
    assert state.halted_until == datetime(2026, 9, 24, tzinfo=UTC)
    # Equity bounces above the kill line the same day: still halted.
    clock.advance(timedelta(hours=2))
    bounced = book(equity="9900", day_open="10000", peak="10000")
    state, t = brk.assess(bounced, inputs=_fresh(clock), llm_outage=False, decision_id="d1")
    assert state.activation is HALTED
    assert state.trips == (DAILY_KILL,)
    assert t is None
    # The next UTC day: the day-open equity resets, and the halt steps down to REDUCE_ONLY.
    clock.set(datetime(2026, 9, 24, 0, 0, 30, tzinfo=UTC))
    new_day = book(equity="9900", day_open="9900", peak="10000")
    state, t = brk.assess(new_day, inputs=_fresh(clock), llm_outage=False, decision_id="d2")
    assert state.activation is REDUCE_ONLY
    assert state.halted_until is None
    state, t = brk.assess(new_day, inputs=_fresh(clock), llm_outage=False, decision_id="d3")
    assert state.activation is ACTIVE


def test_outage_halt_lifts_straight_to_active_at_the_next_valid_decision() -> None:
    brk, clock = _breaker()
    state, t = brk.assess(book(), inputs=FRESH, llm_outage=True)
    assert (state.activation, state.trips) == (HALTED, (LLM_OUTAGE,))
    # The protective loop, with no valid decision, cannot lift it.
    clock.advance(timedelta(minutes=1))
    state, t = brk.assess(book(), inputs=_fresh(clock), llm_outage=False)
    assert state.activation is HALTED
    assert t is None
    # Another failed cycle keeps it.
    state, _ = brk.assess(book(), inputs=_fresh(clock), llm_outage=True, decision_id="bad")
    assert state.activation is HALTED
    state, t = brk.assess(book(), inputs=_fresh(clock), llm_outage=False, decision_id="d9")
    assert state.activation is ACTIVE
    assert t is not None
    assert (t.from_state, t.to_state) == (HALTED, ACTIVE)


def test_outage_combined_with_another_halt_recovers_stepwise() -> None:
    brk, clock = _breaker(datetime(2026, 9, 23, 22, 0, tzinfo=UTC))
    killed = book(equity="9850", day_open="10000", peak="10000")
    state, _ = brk.assess(killed, inputs=_fresh(clock), llm_outage=True)
    assert set(state.trips) == {DAILY_KILL, LLM_OUTAGE}
    clock.set(datetime(2026, 9, 24, 1, 0, tzinfo=UTC))
    new_day = book(equity="9850", day_open="9850", peak="10000")
    state, _ = brk.assess(new_day, inputs=_fresh(clock), llm_outage=False, decision_id="d1")
    assert state.activation is REDUCE_ONLY
    state, _ = brk.assess(new_day, inputs=_fresh(clock), llm_outage=False, decision_id="d2")
    assert state.activation is ACTIVE


def test_state_is_what_assess_returned() -> None:
    brk, _ = _breaker()
    state, _ = brk.assess(book(losses=4), inputs=FRESH, llm_outage=False)
    assert brk.state() == state


def _run(brk: Breaker, clock: ManualClock) -> list[BreakerTransition]:
    """A realistic day: a drawdown, a kill, an outage, recovery. Returns the logged transitions."""
    log: list[BreakerTransition] = []
    steps: list[tuple[timedelta, BookState, bool, str | None]] = [
        (timedelta(0), book(equity="9750", peak="10000", day_open="9800"), False, None),
        (timedelta(hours=1), book(equity="9640", peak="10000", day_open="9800"), False, None),
        (timedelta(hours=1), book(equity="9650", peak="10000", day_open="9800"), True, None),
        (timedelta(hours=1), book(equity="9650", peak="10000", day_open="9800"), False, "d1"),
        (timedelta(hours=9), book(equity="9800", peak="10000", day_open="9800"), False, "d2"),
    ]
    for delta, b, outage, decision_id in steps:
        clock.advance(delta)
        _, t = brk.assess(b, inputs=_fresh(clock), llm_outage=outage, decision_id=decision_id)
        if t is not None:
            log.append(t)
    return log


def test_restore_rebuilds_the_same_state_from_the_logged_transitions() -> None:
    live, clock = _breaker(datetime(2026, 9, 23, 12, 0, tzinfo=UTC))
    log = _run(live, clock)
    assert len(log) >= 3
    # Rebuild at every prefix of the log, as a restart at any point would.
    for n in range(len(log) + 1):
        restored, _ = _breaker(datetime(2026, 9, 23, 12, 0, tzinfo=UTC))
        restored.restore(log[:n])
        assert restored.state().activation is (log[n - 1].to_state if n else ACTIVE)
    assert restored.state() == live.state()


def test_restore_rebuilds_a_daily_kill_latch() -> None:
    live, clock = _breaker(datetime(2026, 9, 23, 18, 0, tzinfo=UTC))
    live.assess(book(equity="9850", day_open="10000"), inputs=_fresh(clock), llm_outage=False)
    log = [
        BreakerTransition(
            at=datetime(2026, 9, 23, 18, 0, tzinfo=UTC),
            from_state=ACTIVE,
            to_state=HALTED,
            trips=(DAILY_KILL,),
        )
    ]
    restored, _ = _breaker(datetime(2026, 9, 23, 19, 0, tzinfo=UTC))
    restored.restore(log)
    assert restored.state() == live.state()
    assert restored.state().halted_until == datetime(2026, 9, 24, tzinfo=UTC)


def _t(
    at: datetime, a: Activation, b: Activation, trips: tuple[str, ...] = ()
) -> BreakerTransition:
    return BreakerTransition(at=at, from_state=a, to_state=b, trips=trips)


@pytest.mark.parametrize(
    "log",
    [
        # does not start where the state is
        [_t(T0, REDUCE_ONLY, ACTIVE)],
        # a gap in the chain
        [_t(T0, ACTIVE, HALTED, (DRAWDOWN_HALT,)), _t(T0, REDUCE_ONLY, ACTIVE)],
        # time runs backwards
        [
            _t(T0, ACTIVE, REDUCE_ONLY, (LOSING_STREAK,)),
            _t(T0 - timedelta(hours=1), REDUCE_ONLY, ACTIVE),
        ],
        # HALTED -> ACTIVE out of a halt that was not outage-only
        [_t(T0, ACTIVE, HALTED, (DRAWDOWN_HALT,)), _t(T0, HALTED, ACTIVE)],
    ],
)
def test_an_unreadable_log_resolves_to_halted(log: list[BreakerTransition]) -> None:
    brk, _ = _breaker()
    brk.restore(log)
    state = brk.state()
    assert state.activation is HALTED
    assert state.trips == (UNREADABLE_STATE,)


def test_an_unreadable_halt_recovers_stepwise() -> None:
    brk, clock = _breaker()
    brk.restore([_t(T0, REDUCE_ONLY, ACTIVE)])
    state, _ = brk.assess(book(), inputs=_fresh(clock), llm_outage=False, decision_id="d1")
    assert state.activation is REDUCE_ONLY
    state, _ = brk.assess(book(), inputs=_fresh(clock), llm_outage=False, decision_id="d2")
    assert state.activation is ACTIVE


def test_restore_accepts_the_legal_outage_recovery_and_an_empty_log() -> None:
    brk, _ = _breaker()
    brk.restore([])
    assert brk.state() == breaker_state(at=T0)
    brk.restore(
        [
            _t(T0, ACTIVE, HALTED, (LLM_OUTAGE,)),
            _t(T0 + timedelta(hours=1), HALTED, ACTIVE),
        ]
    )
    assert brk.state().activation is ACTIVE
    assert brk.state().since == T0 + timedelta(hours=1)


def test_a_breaker_can_start_from_a_given_state() -> None:
    halted = breaker_state(HALTED, (DRAWDOWN_HALT,))
    brk = Breaker(POLICY_V1, ManualClock(T0), halted)
    assert brk.state() is halted
