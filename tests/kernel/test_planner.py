"""The planner: grid rounding, stops, identity, splits, and every shape of leg a ruling asks for."""

from collections.abc import Mapping
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from helpers import T0
from kernel.kbuild import (
    BTC,
    NVDA,
    SPX,
    UNIVERSE,
    book,
    breaker_state,
    decision,
    demo_quote,
    flat_quote,
    inputs,
    kernel,
    live_quote,
    position,
    spec,
)
from sentiment_agent.hashing import content_hash
from sentiment_agent.kernel.planner import (
    client_oid,
    current_weight,
    intent_core,
    order_price,
    plan_orders,
    planning_price,
    round_qty,
    split_quantity,
    stop_price,
    target_quantity,
    weight_of,
)
from sentiment_agent.policy import POLICY_V1
from sentiment_agent.types import (
    ALL_GUARDS,
    BookState,
    GuardId,
    KernelInputs,
    KernelRuling,
    OrderIntent,
    OrderPlan,
    OrderPurpose,
    Position,
    ProtectiveReason,
    Quote,
    Side,
    client_oid_of,
)

P = POLICY_V1


# --- grid arithmetic ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("qty", "step", "expected"),
    [
        ("0.0579", "0.01", "0.05"),
        ("0.05", "0.01", "0.05"),
        ("0.0099", "0.01", "0.00"),
        ("-0.0579", "0.01", "-0.05"),
        ("-0.0001", "0.01", "0.00"),
        ("1.23456789", "0.0001", "1.2345"),
        ("7", "0.25", "7.00"),
        ("7.3", "0.25", "7.25"),
    ],
)
def test_round_qty_rounds_toward_zero_onto_the_grid(qty: str, step: str, expected: str) -> None:
    result = round_qty(Decimal(qty), spec(NVDA, step=step))
    assert str(result) == expected


def test_round_qty_refuses_a_non_positive_step() -> None:
    with pytest.raises(ValueError, match="positive"):
        round_qty(Decimal(1), spec(NVDA, step="0"))


@pytest.mark.parametrize(
    ("entry", "side", "step", "expected"),
    [
        # long: 4% below, rounded up (toward the entry)
        ("100.01", Side.BUY, "0.01", "96.01"),
        ("100", Side.BUY, "0.01", "96.00"),
        ("83169.7", Side.BUY, "0.1", "79843.0"),
        # short: 4% above, rounded down (toward the entry)
        ("99.99", Side.SELL, "0.01", "103.98"),
        ("100", Side.SELL, "0.01", "104.00"),
        ("83168.5", Side.SELL, "0.1", "86495.2"),
    ],
)
def test_stop_price_rounds_toward_the_entry(
    entry: str, side: Side, step: str, expected: str
) -> None:
    stop = stop_price(Decimal(entry), side, P, spec(NVDA, price_step=step))
    assert str(stop) == expected
    loss = abs(Decimal(entry) - stop) / Decimal(entry)
    assert loss <= Decimal("0.04")


def test_stop_price_refuses_what_it_cannot_place() -> None:
    with pytest.raises(ValueError, match="positive"):
        stop_price(Decimal(0), Side.BUY, P, spec(NVDA))
    with pytest.raises(ValueError, match="no long stop"):
        stop_price(Decimal(12), Side.BUY, P, spec(NVDA, price_step="10"))
    with pytest.raises(ValueError, match="no short stop"):
        stop_price(Decimal(12), Side.SELL, P, spec(NVDA, price_step="10"))


@pytest.mark.parametrize(
    ("qty", "cap", "legs"),
    [
        ("59.99", "60", ["59.99"]),
        ("60", "60", ["60"]),
        ("60.01", "60", ["30.01", "30.00"]),
        ("125.00", "60", ["41.67", "41.67", "41.66"]),
        ("0.0120", None, ["0.0120"]),
    ],
)
def test_split_quantity_near_equal_legs_under_the_cap(
    qty: str, cap: str | None, legs: list[str]
) -> None:
    result = split_quantity(Decimal(qty), spec(NVDA, max_market=cap))
    assert [str(x) for x in result] == legs
    assert sum(result) == Decimal(qty)


def test_split_quantity_puts_an_off_grid_remainder_on_the_last_leg() -> None:
    result = split_quantity(Decimal("120.005"), spec(NVDA, max_market="60"))
    assert sum(result) == Decimal("120.005")
    assert all(x <= 60 for x in result)
    assert result[-1] % Decimal("0.01") != 0


def test_weights_and_quantities_share_one_price() -> None:
    held = position(NVDA, "2")
    b = book(positions=[held], marks={NVDA: "200"})
    inp = inputs((NVDA,), demo={NVDA: flat_quote(NVDA, "250")})
    assert planning_price(NVDA, b, inp) == Decimal(250)
    assert current_weight(NVDA, b, inp) == pytest.approx(0.05)
    no_quote = inputs((NVDA,), demo={})
    assert planning_price(NVDA, b, no_quote) == Decimal(200)
    bare = book(positions=[position(NVDA, "2", entry="150")], marks={})
    assert planning_price(NVDA, bare, no_quote) is None
    assert order_price(NVDA, bare, no_quote) == Decimal(150)
    assert weight_of(Decimal(-1), Decimal(100), Decimal(0)) == -1.0
    assert target_quantity(0.05, Decimal(10_000), Decimal(250), spec(NVDA)) == Decimal("2.00")
    assert target_quantity(0.0, Decimal(10_000), Decimal(250), spec(NVDA)) == Decimal("0.00")


# --- identity -----------------------------------------------------------------------------------


def _core(**change: object) -> dict[str, object]:
    core = intent_core(
        ruling_id="r1",
        symbol=NVDA,
        side=Side.BUY,
        qty=Decimal("1.00"),
        purpose=OrderPurpose.OPEN,
        split_index=0,
    )
    core.update(change)
    return core


def test_client_oid_is_deterministic_and_derived_from_the_intent_hash() -> None:
    oid = client_oid(_core())
    assert oid == client_oid(_core())
    assert oid == client_oid_of(content_hash(_core()))
    assert len(oid) == 32
    assert oid.startswith("sa")
    int(oid[2:], 16)


@pytest.mark.parametrize(
    "change",
    [
        {"ruling_id": "r2"},
        {"symbol": BTC},
        {"side": "sell"},
        {"qty": "1.01"},
        {"qty": "1.0"},
        {"purpose": "increase"},
        {"split_index": 1},
    ],
)
def test_client_oid_changes_with_every_identity_field(change: Mapping[str, object]) -> None:
    assert client_oid(_core(**change)) != client_oid(_core())


# --- plans --------------------------------------------------------------------------------------


def _ruled(
    proposed: dict[str, float] | None,
    b: BookState,
    inp: KernelInputs,
    *,
    at: datetime = T0,
) -> tuple[KernelRuling, OrderPlan]:
    k, _ = kernel(at)
    ruling = k.rule(
        proposed=proposed,
        book=b,
        inputs=inp,
        context=decision(UNIVERSE),
        breaker=breaker_state(at=at),
        guards=ALL_GUARDS,
    )
    return ruling, plan_orders(ruling, b, inp, P, now=at)


def _only(plan: OrderPlan) -> OrderIntent:
    assert len(plan.intents) == 1, plan.intents
    return plan.intents[0]


def _post_weight(intents: tuple[OrderIntent, ...], held: Decimal, price: Decimal) -> float:
    for i in intents:
        held += i.qty if i.side is Side.BUY else -i.qty
    return float(held * price / Decimal(10_000))


FLAT = {s: flat_quote(s, "100", spread="0.02") for s in (NVDA,)}


def _flat_inputs(demo: Mapping[str, Quote] | None = None) -> KernelInputs:
    return inputs((NVDA,), demo=demo or FLAT, live={NVDA: live_quote(NVDA, last="100")})


def test_open_from_flat() -> None:
    ruling, plan = _ruled({NVDA: 0.03}, book(), _flat_inputs())
    intent = _only(plan)
    assert (intent.side, intent.qty, intent.purpose) == (
        Side.BUY,
        Decimal("3.00"),
        OrderPurpose.OPEN,
    )
    assert not intent.reduce_only
    assert intent.reference_price == Decimal(100)
    assert intent.notional == Decimal("300.00")
    assert intent.expected_fee == Decimal("300.00") * Decimal("0.0006")
    assert intent.stop_loss_price == Decimal("96.01")  # 4% below the ask 100.01, rounded up
    assert intent.decision_id == "d1"
    assert intent.ruling_id == ruling.ruling_id == plan.ruling_id
    assert intent.client_oid == client_oid_of(intent.intent_id)
    assert plan.skipped == ()


def test_open_short_from_flat() -> None:
    _, plan = _ruled({NVDA: -0.03}, book(), _flat_inputs())
    intent = _only(plan)
    assert (intent.side, intent.qty) == (Side.SELL, Decimal("3.00"))
    assert intent.stop_loss_price == Decimal("103.98")  # 4% above the bid 99.99, rounded down


def test_increase_adds_only_the_difference() -> None:
    b = book(positions=[position(NVDA, "2.00", entry="95")], marks={NVDA: "100"})
    _, plan = _ruled({NVDA: 0.035}, b, _flat_inputs())
    intent = _only(plan)
    assert (intent.side, intent.qty, intent.purpose) == (
        Side.BUY,
        Decimal("1.50"),
        OrderPurpose.INCREASE,
    )


def test_increase_rounding_never_exceeds_the_approved_weight() -> None:
    b = book(positions=[position(NVDA, "2.00")], marks={NVDA: "100"})
    ruling, plan = _ruled({NVDA: 0.03337}, b, _flat_inputs())
    intent = _only(plan)
    assert intent.qty == Decimal("1.33")
    approved = ruling.instruments[0].approved_weight
    assert _post_weight(plan.intents, Decimal("2.00"), Decimal(100)) <= approved


def test_partial_reduction() -> None:
    b = book(positions=[position(NVDA, "4.00")], marks={NVDA: "100"})
    _, plan = _ruled({NVDA: 0.0133}, b, _flat_inputs())
    intent = _only(plan)
    # Target 1.33 rounds toward zero, so the reduction rounds up: never short of what was approved.
    assert (intent.side, intent.qty, intent.purpose) == (
        Side.SELL,
        Decimal("2.67"),
        OrderPurpose.REDUCE,
    )
    assert intent.reduce_only
    assert intent.stop_loss_price is None


def test_close_to_flat() -> None:
    b = book(positions=[position(NVDA, "-4.00")], marks={NVDA: "100"})
    _, plan = _ruled({NVDA: 0.0}, b, _flat_inputs())
    intent = _only(plan)
    assert (intent.side, intent.qty, intent.purpose) == (
        Side.BUY,
        Decimal("4.00"),
        OrderPurpose.CLOSE,
    )
    assert intent.reduce_only


def test_a_reduction_that_would_leave_dust_closes_instead() -> None:
    # 0.04 NVDA at 100 is 4 USDT, under the 5 USDT minimum: the residue could never be closed.
    b = book(positions=[position(NVDA, "3.00")], marks={NVDA: "100"})
    _, plan = _ruled({NVDA: 0.0004}, b, _flat_inputs())
    intent = _only(plan)
    assert (intent.qty, intent.purpose) == (Decimal("3.00"), OrderPurpose.CLOSE)


def test_a_reduction_below_the_venue_minimum_is_skipped_and_logged() -> None:
    b = book(positions=[position(NVDA, "3.00")], marks={NVDA: "100"})
    _, plan = _ruled({NVDA: 0.0297}, b, _flat_inputs())  # 0.03 NVDA = 3 USDT
    assert plan.intents == ()
    (skip,) = plan.skipped
    assert skip.symbol == NVDA
    assert "minOrderAmount" in skip.reason
    assert skip.wanted_delta_weight == pytest.approx(-0.0003)


def test_an_increase_below_the_venue_minimum_never_reaches_the_plan() -> None:
    # G11 refuses it in the kernel, so the ruling keeps the holding and the planner sends nothing.
    b = book(positions=[position(NVDA, "3.00")], marks={NVDA: "100"})
    ruling, plan = _ruled({NVDA: 0.0303}, b, _flat_inputs())
    assert ruling.instruments[0].binding_guard is GuardId.G11_ELIGIBILITY
    assert plan.intents == ()
    assert plan.skipped == ()


def test_planner_skips_an_increase_below_the_minimum_when_the_kernel_did_not_check() -> None:
    b = book(positions=[position(NVDA, "3.00")], marks={NVDA: "100"})
    k, _ = kernel()
    inp = _flat_inputs()
    ruling = k.rule(
        proposed={NVDA: 0.0303},
        book=b,
        inputs=inp,
        context=decision(UNIVERSE),
        breaker=breaker_state(),
        guards=frozenset(),
    )
    plan = plan_orders(ruling, b, inp, P, now=T0)
    assert plan.intents == ()
    assert "increase skipped" in plan.skipped[0].reason


def test_a_flip_is_a_close_then_an_open_with_reducing_legs_first() -> None:
    held = position(NVDA, "3.00", last_increase_at=T0 - timedelta(hours=30))
    b = book(positions=[held, position(BTC, "0.0030")], marks={NVDA: "100", BTC: "83176.5"})
    inp = inputs(
        (NVDA, BTC),
        demo={NVDA: FLAT[NVDA], BTC: demo_quote(BTC)},
        live={NVDA: _flat_inputs().live_quotes[NVDA], BTC: live_quote(BTC)},
    )
    _, plan = _ruled({NVDA: -0.02, BTC: 0.05}, b, inp)
    purposes = [(i.symbol, i.side, i.purpose) for i in plan.intents]
    assert purposes == [
        (NVDA, Side.SELL, OrderPurpose.CLOSE),
        (BTC, Side.BUY, OrderPurpose.INCREASE),
        (NVDA, Side.SELL, OrderPurpose.OPEN),
    ]
    close, _, open_ = plan.intents
    assert close.qty == Decimal("3.00")
    assert open_.qty == Decimal("2.00")
    assert open_.stop_loss_price == Decimal("103.98")


def test_a_flip_inside_the_minimum_hold_keeps_only_the_close() -> None:
    held = position(NVDA, "3.00", last_increase_at=T0 - timedelta(hours=2))
    b = book(positions=[held], marks={NVDA: "100"})
    ruling, plan = _ruled({NVDA: -0.02}, b, _flat_inputs())
    assert ruling.instruments[0].binding_guard is GuardId.G6_TURNOVER
    intent = _only(plan)
    assert (intent.side, intent.purpose) == (Side.SELL, OrderPurpose.CLOSE)


def test_orders_above_the_market_order_cap_are_split() -> None:
    # 5% of 1,000,000 at 7659.5 is 6.5278 SP500USDT; its Demo market-order cap is 5.
    b = book(equity="1000000")
    inp = inputs((SPX,), live={SPX: live_quote(SPX)})
    _, plan = _ruled({SPX: 0.05}, b, inp)
    assert [i.qty for i in plan.intents] == [Decimal("3.2639"), Decimal("3.2639")]
    assert [(i.split_index, i.split_count) for i in plan.intents] == [(0, 2), (1, 2)]
    assert len({i.client_oid for i in plan.intents}) == 2
    assert all(i.stop_loss_price == plan.intents[0].stop_loss_price for i in plan.intents)


def test_a_hold_sends_nothing() -> None:
    b = book(positions=[position(NVDA, "3.00")], marks={NVDA: "100"})
    ruling, plan = _ruled(None, b, _flat_inputs())
    assert ruling.instruments[0].approved_weight == ruling.instruments[0].current_weight
    assert plan.intents == ()
    assert plan.skipped == ()


def test_protective_rulings_plan_protective_exits() -> None:
    held = [position(NVDA, "3.00"), position(BTC, "-0.0030")]
    b = book(positions=held, marks={NVDA: "100", BTC: "83176.5"}, equity="9800", day_open="10000")
    inp = inputs((NVDA, BTC), demo={NVDA: FLAT[NVDA], BTC: demo_quote(BTC)})
    k, _ = kernel()
    ruling = k.protective(book=b, inputs=inp, breaker=breaker_state())
    assert ruling is not None
    assert ruling.protective_reason is ProtectiveReason.DAILY_KILL
    plan = plan_orders(ruling, b, inp, P, now=T0)
    assert [(i.symbol, i.side, i.qty, i.purpose) for i in plan.intents] == [
        (BTC, Side.BUY, Decimal("0.0030"), OrderPurpose.PROTECTIVE_EXIT),
        (NVDA, Side.SELL, Decimal("3.00"), OrderPurpose.PROTECTIVE_EXIT),
    ]
    assert all(i.decision_id is None and i.reduce_only for i in plan.intents)


def test_a_guard_forced_exit_inside_a_decision_is_protective() -> None:
    held = position(NVDA, "3.00")
    b = book(positions=[held], marks={NVDA: "100"})
    broken = {
        NVDA: demo_quote(NVDA, mark="110", index="100", bid="99.99", ask="100.01", last="100")
    }
    _, plan = _ruled({NVDA: 0.03}, b, _flat_inputs(demo=broken))
    intent = _only(plan)
    assert intent.purpose is OrderPurpose.PROTECTIVE_EXIT
    # The model asking for the close itself is a model close.
    _, plan = _ruled({NVDA: 0.0}, b, _flat_inputs(demo=broken))
    assert _only(plan).purpose is OrderPurpose.CLOSE


def test_a_close_needs_no_quote() -> None:
    held = position(NVDA, "3.00", entry="97")
    b = book(positions=[held], marks={}, equity="9800", day_open="10000")
    inp = inputs((NVDA,), demo={}, live={}, specs={}, index_moves={})
    k, _ = kernel()
    ruling = k.protective(book=b, inputs=inp, breaker=breaker_state())
    assert ruling is not None
    plan = plan_orders(ruling, b, inp, P, now=T0)
    intent = _only(plan)
    assert (intent.qty, intent.reference_price) == (Decimal("3.00"), Decimal(97))
    assert intent.expected_fee == Decimal("291.00") * Decimal("0.0006")


def test_increases_without_a_spec_or_price_are_skipped_when_unguarded() -> None:
    k, _ = kernel()
    for inp in (
        inputs((NVDA,), specs={}),
        inputs((NVDA,), demo={}),
    ):
        ruling = k.rule(
            proposed={NVDA: 0.03},
            book=book(),
            inputs=inp,
            context=decision(UNIVERSE),
            breaker=breaker_state(),
            guards=frozenset(),
        )
        plan = plan_orders(ruling, book(), inp, P, now=T0)
        assert plan.intents == ()
        assert len(plan.skipped) == 1


def test_the_plan_is_deterministic() -> None:
    b = book(positions=[position(NVDA, "2.00")], marks={NVDA: "100"})
    first = _ruled({NVDA: 0.04, BTC: 0.01}, b, inputs((NVDA, BTC)))
    second = _ruled({NVDA: 0.04, BTC: 0.01}, b, inputs((NVDA, BTC)))
    assert first == second
    later = plan_orders(first[0], b, inputs((NVDA, BTC)), P, now=T0 + timedelta(hours=1))
    assert later.plan_id == first[1].plan_id
    assert [i.client_oid for i in later.intents] == [i.client_oid for i in first[1].intents]


def test_positions_are_read_from_the_book_not_the_ruling() -> None:
    held: list[Position] = [position(NVDA, "2.00")]
    b = book(positions=held, marks={NVDA: "100"})
    ruling, _ = _ruled({NVDA: 0.03}, b, _flat_inputs())
    later = book(positions=[position(NVDA, "3.00")], marks={NVDA: "100"})
    plan = plan_orders(ruling, later, _flat_inputs(), P, now=T0)
    # The book already reached the approved weight: nothing more to add.
    assert plan.intents == ()
    assert "rounds to nothing" in plan.skipped[0].reason
