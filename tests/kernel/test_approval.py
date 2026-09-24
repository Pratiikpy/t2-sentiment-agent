"""The approval minter refuses anything that is not exactly what the kernel ruled and the planner
derived, and mints nothing from a plan with any fault in it."""

from decimal import Decimal

import pytest

from helpers import T0, unvalidated_copy
from kernel.kbuild import (
    BTC,
    NVDA,
    UNIVERSE,
    book,
    breaker_state,
    decision,
    flat_quote,
    inputs,
    kernel,
    live_quote,
    position,
)
from sentiment_agent.kernel.approval import ApprovalError, approve
from sentiment_agent.kernel.planner import build_intent, plan_id_for, plan_orders
from sentiment_agent.policy import POLICY_V1
from sentiment_agent.types import (
    ALL_GUARDS,
    ApprovedOrder,
    BookState,
    KernelInputs,
    KernelRuling,
    OrderIntent,
    OrderPlan,
    OrderPurpose,
    Side,
)

P = POLICY_V1


def _inputs() -> KernelInputs:
    return inputs(
        (NVDA, BTC),
        demo={
            NVDA: flat_quote(NVDA, "100", spread="0.02"),
            BTC: flat_quote(BTC, "80000", spread="2"),
        },
        live={NVDA: live_quote(NVDA, last="100"), BTC: live_quote(BTC, last="80000")},
    )


def _book() -> BookState:
    return book(positions=[position(BTC, "0.0050")], marks={BTC: "80000"})


def _plan(
    proposed: dict[str, float] | None = None, *, b: BookState | None = None
) -> tuple[KernelRuling, OrderPlan, BookState, KernelInputs]:
    b = b if b is not None else _book()
    inp = _inputs()
    k, _ = kernel()
    ruling = k.rule(
        proposed={NVDA: 0.03, BTC: 0.0} if proposed is None else proposed,
        book=b,
        inputs=inp,
        context=decision(UNIVERSE),
        breaker=breaker_state(),
        guards=ALL_GUARDS,
    )
    return ruling, plan_orders(ruling, b, inp, P, now=T0), b, inp


def _replan(plan: OrderPlan, intents: tuple[OrderIntent, ...]) -> OrderPlan:
    """The plan with ``intents`` swapped in and its id recomputed, as a forger would."""
    return plan.model_copy(
        update={
            "intents": intents,
            "plan_id": plan_id_for(plan.ruling_id, intents, plan.skipped),
        }
    )


def _refused(plan: OrderPlan, ruling: KernelRuling, b: BookState, inp: KernelInputs) -> str:
    with pytest.raises(ApprovalError) as info:
        approve(plan, ruling, b, inp)
    return str(info.value)


def test_a_kernel_plan_is_minted_in_order() -> None:
    ruling, plan, b, inp = _plan()
    assert [(i.symbol, i.purpose) for i in plan.intents] == [
        (BTC, OrderPurpose.CLOSE),
        (NVDA, OrderPurpose.OPEN),
    ]
    orders = approve(plan, ruling, b, inp)
    assert [o.intent for o in orders] == list(plan.intents)
    assert all(isinstance(o, ApprovedOrder) and o.verify() for o in orders)
    assert all(o.ruling_id == ruling.ruling_id for o in orders)


def test_an_empty_plan_mints_nothing() -> None:
    ruling, plan, b, inp = _plan({NVDA: 0.0, BTC: 0.04})
    assert plan.intents == ()
    assert approve(plan, ruling, b, inp) == ()


def test_a_tampered_intent_is_refused() -> None:
    ruling, plan, b, inp = _plan()
    close, open_ = plan.intents
    bigger = open_.model_copy(update={"qty": Decimal("30.00"), "notional": Decimal("3000.00")})
    text = _refused(_replan(plan, (close, bigger)), ruling, b, inp)
    assert "does not hash to its intent_id" in text


def test_nothing_is_minted_when_one_intent_is_bad() -> None:
    ruling, plan, b, inp = _plan()
    close, open_ = plan.intents
    turned = open_.model_copy(update={"side": Side.SELL, "stop_loss_price": Decimal("104")})
    with pytest.raises(ApprovalError):
        approve(_replan(plan, (close, turned)), ruling, b, inp)


def test_a_plan_for_another_ruling_is_refused() -> None:
    ruling, plan, b, inp = _plan()
    other, _, _, _ = _plan({NVDA: 0.02, BTC: 0.0})
    assert other.ruling_id != ruling.ruling_id
    text = _refused(plan, other, b, inp)
    assert "answers ruling" in text
    assert "planned under another ruling" in text


def test_a_ruling_changed_after_issue_is_refused() -> None:
    ruling, plan, b, inp = _plan()
    nvda = ruling.instruments[1]
    # Still a valid ruling on its own terms (the request itself is raised), but not the kernel's.
    raised = nvda.model_copy(update={"proposed_weight": 0.05, "approved_weight": 0.05})
    edited = ruling.model_copy(update={"instruments": (ruling.instruments[0], raised)})
    text = _refused(plan, edited, b, inp)
    assert "does not hash to its id" in text


def test_a_ruling_that_fails_its_own_checks_is_refused() -> None:
    ruling, plan, b, inp = _plan()
    nvda = ruling.instruments[1]
    widened = unvalidated_copy(nvda, approved_weight=0.2)
    forged = unvalidated_copy(ruling, instruments=(ruling.instruments[0], widened))
    text = _refused(plan, forged, b, inp)
    assert "fails its own checks" in text


def test_an_intent_that_would_exceed_the_approved_weight_is_refused() -> None:
    ruling, plan, b, inp = _plan()
    close, open_ = plan.intents
    # A correctly derived identity for a larger order: only the weight check can catch it.
    larger = build_intent(
        ruling=ruling,
        symbol=NVDA,
        side=Side.BUY,
        qty=Decimal("3.50"),
        purpose=OrderPurpose.OPEN,
        price=Decimal(100),
        spec=inp.specs[NVDA],
        stop=open_.stop_loss_price,
        split_index=0,
        split_count=1,
    )
    text = _refused(_replan(plan, (close, larger)), ruling, b, inp)
    assert "exceeds it" in text


def test_a_second_leg_that_pushes_past_the_approval_is_refused() -> None:
    ruling, plan, b, inp = _plan()
    close, open_ = plan.intents
    again = build_intent(
        ruling=ruling,
        symbol=NVDA,
        side=Side.BUY,
        qty=Decimal("1.00"),
        purpose=OrderPurpose.INCREASE,
        price=Decimal(100),
        spec=inp.specs[NVDA],
        stop=open_.stop_loss_price,
        split_index=1,
        split_count=2,
    )
    text = _refused(_replan(plan, (close, open_, again)), ruling, b, inp)
    assert "exceeds it" in text


def test_exposure_cannot_be_added_under_a_protective_ruling() -> None:
    held = book(
        positions=[position(BTC, "0.0050"), position(NVDA, "3.00")],
        marks={BTC: "80000", NVDA: "100"},
        equity="9800",
        day_open="10000",
    )
    inp = _inputs()
    k, _ = kernel()
    ruling = k.protective(book=held, inputs=inp, breaker=breaker_state())
    assert ruling is not None
    plan = plan_orders(ruling, held, inp, P, now=T0)
    assert len(approve(plan, ruling, held, inp)) == 2
    opener = build_intent(
        ruling=ruling,
        symbol=NVDA,
        side=Side.SELL,
        qty=Decimal("1.00"),
        purpose=OrderPurpose.OPEN,
        price=Decimal(100),
        spec=inp.specs[NVDA],
        stop=Decimal("104.00"),
        split_index=0,
        split_count=1,
    )
    text = _refused(_replan(plan, (*plan.intents, opener)), ruling, held, inp)
    assert "without a model decision" in text


def test_a_reduction_larger_than_the_holding_is_refused() -> None:
    ruling, plan, b, inp = _plan()
    _, open_ = plan.intents
    overclose = build_intent(
        ruling=ruling,
        symbol=BTC,
        side=Side.SELL,
        qty=Decimal("0.0080"),
        purpose=OrderPurpose.CLOSE,
        price=Decimal(80000),
        spec=inp.specs[BTC],
        stop=None,
        split_index=0,
        split_count=1,
    )
    text = _refused(_replan(plan, (overclose, open_)), ruling, b, inp)
    assert "more than the" in text


def test_a_reduction_of_an_instrument_the_ruling_holds_is_refused() -> None:
    ruling, plan, b, inp = _plan({NVDA: 0.03, BTC: 0.04})
    trim = build_intent(
        ruling=ruling,
        symbol=BTC,
        side=Side.SELL,
        qty=Decimal("0.0010"),
        purpose=OrderPurpose.REDUCE,
        price=Decimal(80000),
        spec=inp.specs[BTC],
        stop=None,
        split_index=0,
        split_count=1,
    )
    text = _refused(_replan(plan, (trim, *plan.intents)), ruling, b, inp)
    assert "does not reduce this instrument" in text


def test_a_changed_reference_price_is_refused() -> None:
    ruling, plan, b, inp = _plan()
    close, open_ = plan.intents
    cheap = build_intent(
        ruling=ruling,
        symbol=NVDA,
        side=Side.BUY,
        qty=open_.qty,
        purpose=OrderPurpose.OPEN,
        price=Decimal(90),
        spec=inp.specs[NVDA],
        stop=Decimal("86.00"),
        split_index=0,
        split_count=1,
    )
    text = _refused(_replan(plan, (close, cheap)), ruling, b, inp)
    assert "is not the planning price" in text


def test_a_stop_farther_than_the_ruled_percentage_is_refused() -> None:
    ruling, plan, b, inp = _plan()
    close, open_ = plan.intents
    loose = build_intent(
        ruling=ruling,
        symbol=NVDA,
        side=Side.BUY,
        qty=open_.qty,
        purpose=OrderPurpose.OPEN,
        price=Decimal(100),
        spec=inp.specs[NVDA],
        stop=Decimal("90.00"),
        split_index=0,
        split_count=1,
    )
    text = _refused(_replan(plan, (close, loose)), ruling, b, inp)
    assert "farther than 4.00%" in text


def test_the_same_plan_against_a_moved_book_is_checked_against_that_book() -> None:
    ruling, plan, _, inp = _plan()
    # The BTC close was already filled: selling again would open a short under a reduce-only leg.
    flat = book(positions=[], marks={})
    text = _refused(plan, ruling, flat, inp)
    assert "reducing sell leg against a position of 0" in text


def test_an_approval_cannot_be_forged_afterwards() -> None:
    ruling, plan, b, inp = _plan()
    (first, _) = approve(plan, ruling, b, inp)
    with pytest.raises(AttributeError):
        first._intent = plan.intents[1]
    assert first.verify()
    assert first.intent == plan.intents[0]
