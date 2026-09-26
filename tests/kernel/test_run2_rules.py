"""Policy v2's scoring window (run2-a4) and the lapsing losing-streak trip (run2-a5), in the kernel
and the breaker.

Under policy v1 neither exists, and these tests also hold v1 to its old behaviour: a record without
a window, and a losing-streak trip that only a winning trade clears."""

from datetime import UTC, datetime, timedelta

import pytest

from kernel.kbuild import (
    BTC,
    NVDA,
    book,
    breaker_state,
    decision,
    inputs,
    position,
    qty_for_weight,
)
from sentiment_agent.clock import ManualClock
from sentiment_agent.kernel.breaker import (
    AWAITING_CLEAN_DECISION,
    LOSING_STREAK,
    Breaker,
    book_conditions,
)
from sentiment_agent.kernel.kernel import RiskKernel
from sentiment_agent.policy import POLICY_V1, POLICY_V2
from sentiment_agent.types import (
    Activation,
    BookState,
    GuardId,
    KernelRuling,
    Policy,
    ProtectiveReason,
)

WINDOW = POLICY_V2.scoring_window
assert WINDOW is not None
END = WINDOW.end
"""Thursday 2026-10-01 00:00 UTC: after the US close, before the Friday freeze."""


def _held(at: datetime) -> BookState:
    return book(
        positions=[
            position(NVDA, qty_for_weight("0.03", NVDA)),
            position(BTC, qty_for_weight("0.02", BTC)),
        ],
        at=at,
    )


def _protective(policy: Policy, at: datetime) -> KernelRuling | None:
    kernel = RiskKernel(policy, ManualClock(at))
    return kernel.protective(
        book=_held(at), inputs=inputs(at=at, snapshot_at=at), breaker=breaker_state(at=at)
    )


def _weights(ruling: KernelRuling) -> dict[str, float]:
    return {i.symbol: i.approved_weight for i in ruling.instruments}


# --- run2-a4: the scoring window ------------------------------------------------------------------


def test_the_window_end_closes_every_leg_of_every_class() -> None:
    ruling = _protective(POLICY_V2, END)
    assert ruling is not None
    assert ruling.protective_reason is ProtectiveReason.WINDOW_END
    # BTC is crypto, which the weekend freeze never touches; the window closes it too.
    assert _weights(ruling) == {NVDA: 0.0, BTC: 0.0}
    for instrument in ruling.instruments:
        assert instrument.binding_guard is GuardId.G2_WEEKEND_FREEZE
        (g2,) = [g for g in instrument.rulings if g.guard is GuardId.G2_WEEKEND_FREEZE]
        assert g2.inputs["scoring_window_end"] == END.isoformat()
        assert "scoring window ended at 2026-10-01 00:00 UTC" in g2.reason


def test_nothing_is_closed_before_the_window_ends() -> None:
    assert _protective(POLICY_V2, END - timedelta(minutes=1)) is None


def test_policy_v1_has_no_window() -> None:
    assert POLICY_V1.scoring_window is None
    assert "scoring_window" not in POLICY_V1.model_dump()
    assert _protective(POLICY_V1, END) is None


def test_no_exposure_opens_after_the_window() -> None:
    at = END + timedelta(hours=13, minutes=30)  # Thursday's US open
    kernel = RiskKernel(POLICY_V2, ManualClock(at))
    ruling = kernel.rule(
        proposed={NVDA: 0.03, BTC: -0.02},
        book=book(at=at),
        inputs=inputs(at=at, snapshot_at=at),
        context=decision([NVDA, BTC]),
        breaker=breaker_state(at=at),
    )
    assert _weights(ruling) == {NVDA: 0.0, BTC: 0.0}
    assert ruling.changed_by_kernel


# --- run2-a5: the losing-streak trip lapses -------------------------------------------------------

T = datetime(2026, 9, 29, 12, 0, tzinfo=UTC)


def _streak(last_loss_hours_ago: float | None, losses: int = 4) -> BookState:
    last = None if last_loss_hours_ago is None else T - timedelta(hours=last_loss_hours_ago)
    return book(losses=losses, at=T).model_copy(update={"last_loss_at": last})


@pytest.mark.parametrize(
    ("hours_ago", "activation", "trips"),
    [
        (0.0, Activation.REDUCE_ONLY, (LOSING_STREAK,)),
        (23.99, Activation.REDUCE_ONLY, (LOSING_STREAK,)),
        (24.0, Activation.ACTIVE, ()),
        (72.0, Activation.ACTIVE, ()),
        # A streak with no recorded loss time cannot be shown to have lapsed.
        (None, Activation.REDUCE_ONLY, (LOSING_STREAK,)),
    ],
)
def test_policy_v2_lets_the_streak_lapse_24_hours_after_the_last_loss(
    hours_ago: float | None, activation: Activation, trips: tuple[str, ...]
) -> None:
    got = book_conditions(_streak(hours_ago), llm_outage=False, policy=POLICY_V2)
    assert got == (activation, trips)


def test_policy_v1_keeps_the_streak_until_a_winner_closes() -> None:
    got = book_conditions(_streak(72.0), llm_outage=False, policy=POLICY_V1)
    assert got == (Activation.REDUCE_ONLY, (LOSING_STREAK,))


def test_a_lapsed_streak_returns_to_active_through_a_clean_decision() -> None:
    clock = ManualClock(T)
    breaker = Breaker(POLICY_V2, clock)
    fresh = inputs(at=T, snapshot_at=T)
    state, _ = breaker.assess(_streak(1.0), inputs=fresh, llm_outage=False, decision_id="d1")
    assert (state.activation, state.trips) == (Activation.REDUCE_ONLY, (LOSING_STREAK,))
    # 24 hours on, the book still flat and the count still four: the trip has lapsed, and the
    # breaker steps back as it does for any cleared condition, through a clean decision.
    later = T + timedelta(hours=23)
    clock.set(later)
    lapsed = book(losses=4, at=later).model_copy(update={"last_loss_at": T - timedelta(hours=1)})
    fresh = inputs(at=later, snapshot_at=later)
    state, _ = breaker.assess(lapsed, inputs=fresh, llm_outage=False)
    assert (state.activation, state.trips) == (Activation.REDUCE_ONLY, (AWAITING_CLEAN_DECISION,))
    state, _ = breaker.assess(lapsed, inputs=fresh, llm_outage=False, decision_id="d2")
    assert state.activation is Activation.ACTIVE


def test_a_book_state_without_a_loss_serialises_as_before() -> None:
    assert "last_loss_at" not in book(at=T).model_dump(mode="json")
    assert _streak(2.0).model_dump(mode="json")["last_loss_at"].startswith("2026-09-29T10:00")
