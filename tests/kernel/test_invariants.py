"""A seeded sweep of random books, proposals and market inputs through rule/protective, the planner
and the minter. Over every case:

* only-reduce holds for every instrument: the approved weight never exceeds the reference in size
  and never turns its side;
* the per-name and gross caps hold whenever G3 applied, and every forced exit is flat;
* G6 and G7 never block a reduction;
* a ruling with no model decision never approves an increase, and no exposure-adding order is ever
  minted except under a DECIDED decision;
* every kernel ruling is approvable, and every minted adding order leaves its instrument within the
  approved weight.

The generator deliberately produces the states the guards exist for: broken Demo marks, stale and
missing quotes, one-sided books, offline instruments, symbols outside the universe, weekend
boundaries, drawdowns, kills, outages, losing streaks, spent fee budgets and off-grid holdings.
"""

import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from kernel.kbuild import LIMITS, PRICES, UNIVERSE, book, breaker_state, position, spec
from sentiment_agent.clock import ManualClock
from sentiment_agent.kernel.approval import approve
from sentiment_agent.kernel.guards import Leg
from sentiment_agent.kernel.kernel import RiskKernel
from sentiment_agent.kernel.planner import plan_orders
from sentiment_agent.policy import POLICY_V1
from sentiment_agent.types import (
    ALL_GUARDS,
    VENUE_GUARDS,
    WEIGHT_EPS,
    Activation,
    BookState,
    GroundingFigure,
    GroundingReport,
    GuardId,
    GuardStatus,
    KernelInputs,
    KernelRuling,
    OrderPlan,
    PriceSource,
    Quote,
    RulingContext,
    Side,
)

SEED = 20260924
CASES = 10_000
P = POLICY_V1

FOREIGN = {"ETHUSDT": "2000", "XRPUSDT": "0.5", "PEPEUSDT": "0.00001"}
POOL = (*UNIVERSE, *FOREIGN)
WEEK = datetime(2026, 9, 21, tzinfo=UTC)  # a Monday
BOUNDARIES = (
    WEEK + timedelta(days=4, hours=18),
    WEEK + timedelta(days=4, hours=19, minutes=45),
    WEEK + timedelta(days=4, hours=19, minutes=59, seconds=59),
    WEEK + timedelta(days=4, hours=20),
    WEEK + timedelta(days=7),
    WEEK + timedelta(days=3, hours=23, minutes=59, seconds=59),
)
UNRESOLVED = GroundingReport(
    figures=(
        GroundingFigure(
            raw="12%",
            value=12.0,
            unit="%",
            context="up 12%",
            resolved=False,
            source=None,
            known_value=None,
        ),
    )
)


@dataclass(frozen=True)
class Case:
    kind: str
    """``decision``, ``no_proposal``, ``protective`` or ``outage``."""
    at: datetime
    book: BookState
    inputs: KernelInputs
    breaker_activation: Activation
    proposed: dict[str, float] | None
    context: RulingContext | None
    guards: frozenset[GuardId]


def _price(symbol: str) -> Decimal:
    return Decimal(PRICES[symbol][0]) if symbol in PRICES else Decimal(FOREIGN[symbol])


def _jitter(rng: random.Random, value: Decimal, spread: float) -> Decimal:
    return (value * Decimal(repr(1 + rng.uniform(-spread, spread)))).quantize(Decimal("0.0001"))


def _demo_quote(rng: random.Random, symbol: str, at: datetime) -> Quote:
    base = _jitter(rng, _price(symbol), 0.01)
    roll = rng.random()
    gap = rng.uniform(0.031, 0.08) if roll < 0.05 else rng.uniform(-0.004, 0.004)
    index = (base * Decimal(repr(1 + gap))).quantize(Decimal("0.0001"))
    spread_bps = rng.choice((0.5, 3.0, 8.0, 19.9, 20.0, 25.0, 60.0))
    half = base * Decimal(repr(spread_bps / 20_000))
    bid, ask = base - half, base + half
    shape = rng.random()
    if shape < 0.02:
        bid = Decimal(0)
    elif shape < 0.04:
        bid, ask = ask, bid
    age = timedelta(seconds=rng.choice((0, 30, 119, 120, 121, 400)))
    return Quote(
        symbol=symbol,
        source=PriceSource.DEMO,
        ts=at - age,
        fetched_at=at - age / 2,
        last=_jitter(rng, base, 0.0005),
        mark=base,
        index=index,
        bid=bid,
        ask=ask,
        funding_rate=None,
        open_interest=None,
        turnover_24h=None,
        price_change_24h=None,
    )


def _live_quote(rng: random.Random, demo: Quote, at: datetime) -> Quote:
    far = rng.random() < 0.05
    last = _jitter(rng, demo.last, 0.05 if far else 0.0005)
    return Quote(
        symbol=demo.symbol,
        source=PriceSource.LIVE,
        ts=at,
        fetched_at=at,
        last=last,
        mark=last,
        index=last,
        bid=last,
        ask=last,
        funding_rate=None,
        open_interest=None,
        turnover_24h=None,
        price_change_24h=None,
    )


def _case(rng: random.Random) -> Case:
    at = (
        rng.choice(BOUNDARIES)
        if rng.random() < 0.15
        else WEEK + timedelta(minutes=rng.randrange(7 * 24 * 60))
    )
    equity = Decimal(rng.choice(("10000", "2500", "100000", "1000000")))
    # Positions.
    held_symbols = rng.sample(POOL, rng.randrange(0, 7))
    positions = []
    for symbol in held_symbols:
        if symbol in FOREIGN and rng.random() < 0.7:
            continue
        weight = Decimal(repr(rng.uniform(-0.08, 0.08)))
        qty = weight * equity / _price(symbol)
        step = Decimal(LIMITS[symbol][1]) if symbol in LIMITS else Decimal("0.01")
        if rng.random() < 0.9:
            qty = (qty / step).to_integral_value() * step
        if qty == 0:
            continue
        increased = at - timedelta(minutes=rng.randrange(0, 48 * 60))
        positions.append(
            position(symbol, str(qty), entry=str(_price(symbol)), last_increase_at=increased)
        )
    marks = {
        p.symbol: str(_jitter(rng, _price(p.symbol), 0.01)) for p in positions if rng.random() < 0.9
    }
    peak = equity * Decimal(repr(1 + rng.choice((0.0, 0.01, 0.025, 0.03, 0.04, 0.05))))
    day_open = equity * Decimal(repr(1 + rng.choice((0.0, -0.005, 0.01, 0.015, 0.02))))
    if rng.random() < 0.02:
        day_open = Decimal(0)
    b = book(
        equity=str(equity),
        positions=positions,
        marks=marks,
        peak=str(peak),
        day_open=str(day_open),
        starting=str(peak),
        fees_today=str(equity * Decimal(repr(rng.choice((0.0, 0.0003, 0.0007, 0.001))))),
        fees_total=str(equity * Decimal(repr(rng.choice((0.0, 0.001, 0.002, 0.003))))),
        rebalances={s: rng.randrange(0, 4) for s in rng.sample(POOL, 3)},
        losses=rng.choice((0, 0, 1, 3, 4, 6)),
        at=at,
    )
    # Market inputs.
    demo = {s: _demo_quote(rng, s, at) for s in UNIVERSE if rng.random() < 0.9}
    live = {s: _live_quote(rng, q, at) for s, q in demo.items() if rng.random() < 0.9}
    specs = {
        s: spec(s, status="online" if rng.random() < 0.95 else "halt", at=at)
        for s in UNIVERSE
        if rng.random() < 0.92
    }
    moves = {s: (None if rng.random() < 0.05 else rng.uniform(0.0, 60.0)) for s in UNIVERSE}
    snapshot_at = (
        None if rng.random() < 0.05 else at - timedelta(minutes=rng.choice((0, 5, 15, 16, 30)))
    )
    inputs = KernelInputs(
        at=at,
        demo_quotes=demo,
        live_quotes=live,
        specs=specs,
        demo_index_move_bps_3h=moves,
        snapshot_id="snap",
        snapshot_taken_at=snapshot_at,
    )
    activation = rng.choices(
        (Activation.ACTIVE, Activation.REDUCE_ONLY, Activation.HALTED), weights=(8, 1, 1)
    )[0]
    roll = rng.random()
    if roll < 0.6:
        names = rng.sample(POOL, rng.randrange(0, 9))
        proposed = {}
        for s in names:
            style = rng.random()
            current = next((p for p in positions if p.symbol == s), None)
            if style < 0.15:
                proposed[s] = 0.0
            elif style < 0.25 and current is not None:
                proposed[s] = float(current.qty * _price(s) / equity)
            else:
                proposed[s] = rng.uniform(-0.12, 0.12)
        grounding = {}
        for s in proposed:
            g = rng.random()
            if g < 0.8:
                grounding[s] = GroundingReport(figures=())
            elif g < 0.9:
                grounding[s] = UNRESOLVED
        context = RulingContext(
            decision_id=f"decided-{rng.randrange(10**9)}",
            protective_reason=None,
            grounding=grounding,
            invalidation_fired={s: rng.random() < 0.2 for s in proposed},
        )
        guard_roll = rng.random()
        guards = (
            ALL_GUARDS
            if guard_roll < 0.85
            else VENUE_GUARDS
            if guard_roll < 0.93
            else frozenset(g for g in GuardId if rng.random() < 0.5)
        )
        return Case("decision", at, b, inputs, activation, proposed, context, guards)
    if roll < 0.7:
        context = RulingContext(
            decision_id=f"decided-{rng.randrange(10**9)}", protective_reason=None
        )
        return Case("no_proposal", at, b, inputs, activation, None, context, ALL_GUARDS)
    kind = "outage" if roll < 0.8 else "protective"
    return Case(kind, at, b, inputs, activation, None, None, ALL_GUARDS)


def _rule(case: Case) -> KernelRuling | None:
    kernel = RiskKernel(P, ManualClock(case.at))
    breaker = breaker_state(case.breaker_activation, ("sweep",), at=case.at)
    if case.context is not None:
        return kernel.rule(
            proposed=case.proposed,
            book=case.book,
            inputs=case.inputs,
            context=case.context,
            breaker=breaker,
            guards=case.guards,
        )
    return kernel.protective(
        book=case.book, inputs=case.inputs, breaker=breaker, llm_outage=case.kind == "outage"
    )


def _check_ruling(case: Case, ruling: KernelRuling) -> None:
    decided = case.kind == "decision"
    for ir in ruling.instruments:
        reference = ir.current_weight if ir.proposed_weight is None else ir.proposed_weight
        approved = ir.approved_weight
        # Only reduce: never larger than the reference, never on the other side.
        assert abs(approved) <= abs(reference) + WEIGHT_EPS, ir
        assert approved == 0 or (approved > 0) == (reference > 0), ir
        if GuardId.G3_SIZE in ruling.guards_applied:
            assert abs(approved) <= P.per_name_max + WEIGHT_EPS, ir
        if any(g.forces_exit for g in (*ir.rulings, *ruling.book_rulings)):
            assert approved == 0, ir
        leg = Leg(symbol=ir.symbol, current=ir.current_weight, proposed=ir.proposed_weight)
        if not leg.adds_exposure:
            # Reductions, closes and holds are never blocked by the turnover or fee guards.
            for g in ir.rulings:
                if g.guard in (GuardId.G6_TURNOVER, GuardId.G7_FEE_BUDGET):
                    assert g.status is not GuardStatus.FIRED, g
                    assert g.ceiling_abs_weight is None or (
                        g.ceiling_abs_weight >= abs(reference) - WEIGHT_EPS
                    ), g
        if not decided:
            # Without a model decision nothing may grow: at most what is held, on its side.
            assert approved == 0 or (
                approved * ir.current_weight > 0
                and abs(approved) <= abs(ir.current_weight) + WEIGHT_EPS
            ), ir
    if GuardId.G3_SIZE in ruling.guards_applied:
        gross = sum(abs(ir.approved_weight) for ir in ruling.instruments)
        assert gross <= P.gross_max + 1e-9, gross


def _check_orders(case: Case, ruling: KernelRuling, plan: OrderPlan) -> int:
    orders = approve(plan, ruling, case.book, case.inputs)
    assert [o.intent for o in orders] == list(plan.intents)
    held = {p.symbol: p.qty for p in case.book.positions.values()}
    adds = 0
    for order in orders:
        intent = order.intent
        before = held.get(intent.symbol, Decimal(0))
        after = before + (intent.qty if intent.side is Side.BUY else -intent.qty)
        held[intent.symbol] = after
        if not intent.purpose.adds_exposure:
            assert intent.reduce_only
            assert abs(after) < abs(before)
            assert after == 0 or (after > 0) == (before > 0)
            continue
        adds += 1
        # An exposure-adding order exists only under a DECIDED model decision.
        assert case.kind == "decision"
        assert ruling.decision_id is not None
        assert case.context is not None
        assert ruling.decision_id == case.context.decision_id
        assert intent.stop_loss_price is not None
        ir = ruling.instrument(intent.symbol)
        assert ir is not None
        post = float(abs(after) * intent.reference_price / case.book.equity)
        assert post <= abs(ir.approved_weight) + WEIGHT_EPS
        assert (after > 0) == (ir.approved_weight > 0)
    return adds


def test_the_seeded_sweep_never_breaks_an_invariant() -> None:
    rng = random.Random(SEED)  # noqa: S311 - a seeded, reproducible sweep, not a secret
    seen = {"decision": 0, "no_proposal": 0, "protective": 0, "outage": 0}
    changed = exits = adding_orders = reducing_orders = protective_rulings = 0
    for index in range(CASES):
        case = _case(rng)
        seen[case.kind] += 1
        ruling = _rule(case)
        if ruling is None:
            assert case.context is None
            continue
        if case.context is None:
            protective_rulings += 1
        _check_ruling(case, ruling)
        plan = plan_orders(ruling, case.book, case.inputs, P, now=case.at)
        adds = _check_orders(case, ruling, plan)
        adding_orders += adds
        reducing_orders += len(plan.intents) - adds
        changed += ruling.changed_by_kernel
        exits += sum(
            1 for ir in ruling.instruments if ir.approved_weight == 0 and ir.current_weight != 0
        )
        if index % 997 == 0:
            again = _rule(case)
            assert again is not None
            assert again.ruling_id == ruling.ruling_id
    # The sweep must actually exercise what it claims to: every kind of case, the kernel changing
    # proposals, forced exits, protective rulings, and orders of both kinds reaching the minter.
    assert all(n >= CASES // 20 for n in seen.values()), seen
    assert changed > CASES // 4
    assert exits > CASES // 10
    assert protective_rulings > CASES // 50
    assert adding_orders > CASES // 20
    assert reducing_orders > CASES // 10
