"""The planner: an approved ruling becomes the market orders the venue will accept.

The kernel speaks in weights (signed fractions of equity). The venue speaks in base-coin quantities
on a grid, with a minimum quantity, a minimum notional and a maximum market-order size
(``GET /api/v3/market/instruments``: ``quantityMultiplier``, ``minOrderQty``, ``minOrderAmount``,
``maxMarketOrderQty``; Demo values in ``validation/demo_venue/universe_probe.json``). This module is
the one place that translates between them, and it is shared by the kernel (G4, G11), the planner
itself and the approval minter so the three can never disagree about a quantity or a price.

Rules, each of which the approval minter re-checks:

* **One price.** A position's weight, a target quantity and an order's notional are all computed at
  :func:`planning_price` (the Demo mark from the kernel inputs, then the book's mark) against
  ``book.equity``. A weight the kernel ruled on and a quantity the planner sends therefore use the
  same number.
* **Round the target position toward zero, not the order.** Rounding the *target* quantity toward
  zero to ``quantityMultiplier`` makes the post-trade weight at most the approved weight in both
  directions: an increase comes out slightly smaller, a reduction slightly larger. Rounding the
  order instead would leave a reduction short of what was approved.
* **Reductions are never blocked by size, and never leave dust.** A close to flat always goes out
  for the full position, whatever its size. A partial reduction that would leave a residue below the
  venue's minimum quantity or notional closes the position instead (a further reduction, which the
  only-reduce rule always permits), so no position is ever stranded below the size that can close
  it. A partial reduction that is itself below the minimums is skipped and logged.
* **Increases below the venue minimums are skipped and logged**, never rounded up: rounding up
  would add exposure the kernel did not approve.
* **Split above ``maxMarketOrderQty``** into near-equal legs on the grid, none above the cap.
* **A flip is a close, then an open.** Every reducing leg is ``reduceOnly``; every adding leg
  carries its preset venue stop 4% from its expected entry, triggered on mark (G4). Reducing legs
  are ordered before adding legs in the plan.
* **``clientOid`` is derived, never random**: ``"sa"`` + the first 30 hex characters of the SHA-256
  of the intent core (ruling id, symbol, side, quantity, purpose, split index), so a retried intent
  reaches the venue under the same id and Bitget's own de-duplication applies. The pattern is ARGUS
  ``execution/orders.py`` ``deterministic_client_order_id`` (MIT, same author), narrowed to the
  fields DESIGN.md §10.7 names.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import ROUND_CEILING, ROUND_DOWN, ROUND_FLOOR, Decimal
from typing import Final

from sentiment_agent.hashing import content_hash
from sentiment_agent.types import (
    BookState,
    InstrumentRuling,
    InstrumentSpec,
    KernelInputs,
    KernelRuling,
    OrderIntent,
    OrderPlan,
    OrderPurpose,
    Policy,
    Quote,
    Side,
    SkippedLeg,
    client_oid_of,
)

MEASURED_DEMO_TAKER_FEE: Final = Decimal("0.0006")
"""Demo ``takerFeeRate`` on all 14 universe instruments (``universe_probe.json``, 2026-09-24). Used
for an order's expected fee only when its instrument spec is absent, which happens only for a close;
never used to size anything."""

_ZERO: Final = Decimal(0)


# ------------------------------------------------------------------------------------------------
# Grid arithmetic
# ------------------------------------------------------------------------------------------------


def _to_grid(value: Decimal, step: Decimal, rounding: str) -> Decimal:
    if step <= 0:
        raise ValueError(f"grid step must be positive, got {step}")
    units = (value / step).to_integral_value(rounding=rounding)
    result = units * step
    if result == 0:
        result = abs(result)  # never emit "-0.00"
    exponent = step.as_tuple().exponent
    if isinstance(exponent, int) and exponent <= 0:
        result = result.quantize(step)
    return result


def round_qty(qty: Decimal, spec: InstrumentSpec) -> Decimal:
    """``qty`` rounded toward zero onto the instrument's ``quantityMultiplier`` grid."""
    return _to_grid(qty, spec.qty_step, ROUND_DOWN)


def stop_price(entry: Decimal, side: Side, policy: Policy, spec: InstrumentSpec) -> Decimal:
    """The preset venue stop for an order entering at ``entry`` on ``side`` (G4).

    A buy opens or grows a long, so its stop sits ``stop_loss_pct`` below the entry; a sell opens
    or grows a short, so its stop sits above. The price is rounded onto ``priceMultiplier``
    **toward the entry**: up for a long, down for a short. The stop can therefore only be tighter
    than 4%, never looser, so the loss it allows never exceeds the policy. Raises ``ValueError``
    when the grid is too coarse to place any stop strictly on the losing side of the entry.
    """
    if entry <= 0:
        raise ValueError(f"entry price must be positive, got {entry}")
    pct = Decimal(repr(policy.stop_loss_pct))
    if side is Side.BUY:
        stop = _to_grid(entry * (1 - pct), spec.price_step, ROUND_CEILING)
        if not _ZERO < stop < entry:
            raise ValueError(
                f"no long stop fits between 0 and the entry {entry} on a {spec.price_step} grid"
            )
    else:
        stop = _to_grid(entry * (1 + pct), spec.price_step, ROUND_FLOOR)
        if stop <= entry:
            raise ValueError(
                f"no short stop fits above the entry {entry} on a {spec.price_step} grid"
            )
    return stop


def split_quantity(qty: Decimal, spec: InstrumentSpec) -> tuple[Decimal, ...]:
    """``qty`` split into near-equal legs, none above ``maxMarketOrderQty`` (G11).

    Legs stay on the quantity grid; a total that is itself off the grid (only ever a close of a
    position the venue reported that way) puts the off-grid remainder on the last leg.
    """
    cap_raw = spec.max_market_order_qty
    if cap_raw is None or cap_raw <= 0 or qty <= cap_raw:
        return (qty,)
    step = spec.qty_step
    cap = _to_grid(cap_raw, step, ROUND_DOWN)
    if cap <= 0:
        raise ValueError(f"maxMarketOrderQty {cap_raw} is below one {step} step")
    count = int((qty / cap).to_integral_value(rounding=ROUND_CEILING))
    base = _to_grid(qty / count, step, ROUND_DOWN)
    legs = [base] * count
    remainder = qty - base * count
    whole_steps = int((remainder / step).to_integral_value(rounding=ROUND_DOWN))
    for i in range(whole_steps):
        legs[i] += step
    legs[-1] += remainder - step * whole_steps
    if any(leg <= 0 or leg > cap for leg in legs) or sum(legs) != qty:  # pragma: no cover
        raise AssertionError(f"split of {qty} under cap {cap} produced {legs}")
    return tuple(legs)


# ------------------------------------------------------------------------------------------------
# Prices and weights: the one definition the kernel, planner and minter share
# ------------------------------------------------------------------------------------------------


def planning_price(symbol: str, book: BookState, inputs: KernelInputs) -> Decimal | None:
    """The price every weight and quantity here is computed at: the Demo mark from the kernel
    inputs, else the book's mark. ``None`` when neither is known."""
    quote = inputs.demo_quotes.get(symbol)
    if quote is not None and quote.mark > 0:
        return quote.mark
    mark = book.marks.get(symbol)
    if mark is not None and mark > 0:
        return mark
    return None


def order_price(symbol: str, book: BookState, inputs: KernelInputs) -> Decimal | None:
    """:func:`planning_price`, falling back to the position's average entry.

    The fallback exists so a close can always be expressed: a protective exit must not depend on a
    quote being available. It is never used to size an increase (the kernel's G4, G10 and G11 refuse
    any increase without a Demo quote)."""
    price = planning_price(symbol, book, inputs)
    if price is not None:
        return price
    position = book.positions.get(symbol)
    if position is not None and position.avg_entry > 0:
        return position.avg_entry
    return None


def weight_of(qty: Decimal, price: Decimal, equity: Decimal) -> float:
    """Signed weight of ``qty`` at ``price``. With no positive equity, any position is the whole
    book or more, reported as ``±1.0`` (the breaker halts on that book anyway)."""
    if qty == 0:
        return 0.0
    if equity <= 0:
        return 1.0 if qty > 0 else -1.0
    return float(qty * price / equity)


def current_weight(symbol: str, book: BookState, inputs: KernelInputs) -> float:
    """The weight the kernel rules on: the book's position at :func:`order_price`."""
    position = book.positions.get(symbol)
    if position is None or position.is_flat:
        return 0.0
    price = order_price(symbol, book, inputs)
    if price is None:
        # A position with no price and no positive entry is corrupt; call it the whole book so
        # nothing is ever added to it and the breaker's drawdown rules treat it as maximal.
        return 1.0 if position.qty > 0 else -1.0
    return weight_of(position.qty, price, book.equity)


def target_quantity(
    abs_weight: float, equity: Decimal, price: Decimal, spec: InstrumentSpec
) -> Decimal:
    """Unsigned base quantity worth ``abs_weight`` of ``equity`` at ``price``, rounded toward zero.

    ``abs_weight`` enters as its shortest decimal representation (``repr``), so the Decimal product
    is exact and ``float(target * price / equity) <= abs_weight`` always holds."""
    if abs_weight <= 0 or equity <= 0 or price <= 0:
        return round_qty(_ZERO, spec)
    return round_qty(Decimal(repr(abs_weight)) * equity / price, spec)


def entry_price(side: Side, quote: Quote | None, fallback: Decimal) -> Decimal:
    """Expected fill price of a market order: the Demo ask for a buy, the bid for a sell (G8 makes
    every order a taker), else ``fallback``. The preset stop is placed from this price."""
    if quote is not None:
        touch = quote.ask if side is Side.BUY else quote.bid
        if touch > 0:
            return touch
    return fallback


# ------------------------------------------------------------------------------------------------
# Intent identity
# ------------------------------------------------------------------------------------------------


def intent_core(
    *,
    ruling_id: str,
    symbol: str,
    side: Side,
    qty: Decimal,
    purpose: OrderPurpose,
    split_index: int,
) -> dict[str, object]:
    """The fields an intent's identity is derived from (DESIGN.md §10.7)."""
    return {
        "ruling_id": ruling_id,
        "symbol": symbol,
        "side": side.value,
        "qty": str(qty),
        "purpose": purpose.value,
        "split_index": split_index,
    }


def client_oid(core: Mapping[str, object]) -> str:
    """``"sa"`` + the first 30 hex characters of the intent hash: 32 characters, deterministic.
    Identical to ``types.client_oid_of(content_hash(core))``, which ``OrderIntent`` enforces."""
    return client_oid_of(content_hash(core))


def fee_rate(spec: InstrumentSpec | None) -> Decimal:
    return spec.taker_fee_rate if spec is not None else MEASURED_DEMO_TAKER_FEE


def build_intent(
    *,
    ruling: KernelRuling,
    symbol: str,
    side: Side,
    qty: Decimal,
    purpose: OrderPurpose,
    price: Decimal,
    spec: InstrumentSpec | None,
    stop: Decimal | None,
    split_index: int,
    split_count: int,
) -> OrderIntent:
    """One validated :class:`OrderIntent` with its derived identity."""
    core = intent_core(
        ruling_id=ruling.ruling_id,
        symbol=symbol,
        side=side,
        qty=qty,
        purpose=purpose,
        split_index=split_index,
    )
    notional = qty * price
    return OrderIntent(
        intent_id=content_hash(core),
        ruling_id=ruling.ruling_id,
        decision_id=ruling.decision_id,
        symbol=symbol,
        side=side,
        qty=qty,
        reduce_only=not purpose.adds_exposure,
        purpose=purpose,
        reference_price=price,
        notional=notional,
        expected_fee=notional * fee_rate(spec),
        stop_loss_price=stop,
        client_oid=client_oid(core),
        split_index=split_index,
        split_count=split_count,
    )


# ------------------------------------------------------------------------------------------------
# The plan
# ------------------------------------------------------------------------------------------------


def forced_by_guard(ruling: InstrumentRuling) -> bool:
    """True when the kernel moved this instrument because a guard forced an exit, as opposed to the
    model asking for the reduction. Such legs are ``PROTECTIVE_EXIT`` and do not count toward the
    model's daily rebalance allowance (``BookState.rebalances_today``)."""
    if ruling.binding_guard is None or not ruling.changed_by_kernel:
        return False
    return any(g.guard is ruling.binding_guard and g.forces_exit for g in ruling.rulings)


def _meets_minimums(qty: Decimal, price: Decimal, spec: InstrumentSpec) -> str | None:
    if qty < spec.min_order_qty:
        return f"{qty} is below minOrderQty {spec.min_order_qty}"
    if qty * price < spec.min_order_amount:
        return f"notional {qty * price} is below minOrderAmount {spec.min_order_amount}"
    return None


class _Planner:
    """Accumulates legs for one plan; reducing legs are emitted before adding legs."""

    def __init__(self, ruling: KernelRuling, book: BookState, inputs: KernelInputs, policy: Policy):
        self.ruling = ruling
        self.book = book
        self.inputs = inputs
        self.policy = policy
        self.reducing: list[OrderIntent] = []
        self.adding: list[OrderIntent] = []
        self.skipped: list[SkippedLeg] = []

    def skip(self, ir: InstrumentRuling, reason: str) -> None:
        self.skipped.append(
            SkippedLeg(
                symbol=ir.symbol,
                wanted_delta_weight=ir.approved_weight - ir.current_weight,
                reason=reason,
            )
        )

    def emit(
        self,
        ir: InstrumentRuling,
        *,
        side: Side,
        qty: Decimal,
        purpose: OrderPurpose,
        price: Decimal,
        spec: InstrumentSpec | None,
        stop: Decimal | None,
    ) -> None:
        legs = split_quantity(qty, spec) if spec is not None else (qty,)
        target = self.adding if purpose.adds_exposure else self.reducing
        for index, leg_qty in enumerate(legs):
            target.append(
                build_intent(
                    ruling=self.ruling,
                    symbol=ir.symbol,
                    side=side,
                    qty=leg_qty,
                    purpose=purpose,
                    price=price,
                    spec=spec,
                    stop=stop,
                    split_index=index,
                    split_count=len(legs),
                )
            )

    def close(self, ir: InstrumentRuling, held: Decimal, *, forced: bool) -> None:
        price = order_price(ir.symbol, self.book, self.inputs)
        if price is None:
            self.skip(ir, "close not expressible: no Demo mark, no book mark, no positive entry")
            return
        self.emit(
            ir,
            side=Side.SELL if held > 0 else Side.BUY,
            qty=abs(held),
            purpose=OrderPurpose.PROTECTIVE_EXIT if forced else OrderPurpose.CLOSE,
            price=price,
            spec=self.inputs.specs.get(ir.symbol),
            stop=None,
        )

    def add(self, ir: InstrumentRuling, held: Decimal) -> None:
        """Open (from flat, or after a flip's close) or grow a position toward the approved
        weight."""
        symbol = ir.symbol
        spec = self.inputs.specs.get(symbol)
        price = planning_price(symbol, self.book, self.inputs)
        if spec is None:
            self.skip(ir, "no Demo instrument spec: an increase cannot be sized")
            return
        if price is None or self.book.equity <= 0:
            self.skip(ir, "no Demo mark or no positive equity: an increase cannot be sized")
            return
        side = Side.BUY if ir.approved_weight > 0 else Side.SELL
        target = target_quantity(abs(ir.approved_weight), self.book.equity, price, spec)
        same_side = held != 0 and (held > 0) == (ir.approved_weight > 0)
        # A holding the venue reported off the grid would put the increase off it too; rounding
        # the increase toward zero keeps it on the grid and within the approved weight.
        delta = round_qty(target - abs(held), spec) if same_side else target
        if delta <= 0:
            self.skip(ir, f"the increase rounds to nothing on the {spec.qty_step} grid")
            return
        short = _meets_minimums(delta, price, spec)
        if short is not None:
            self.skip(ir, f"increase skipped: {short}")
            return
        entry = entry_price(side, self.inputs.demo_quotes.get(symbol), price)
        try:
            stop = stop_price(entry, side, self.policy, spec)
        except ValueError as exc:
            self.skip(ir, f"increase skipped: {exc}")
            return
        if not (stop < price if side is Side.BUY else stop > price):
            # G4 refuses this in the kernel; a ruling made without G4 still never plans a stop
            # that would trigger on the fill (OrderIntent refuses one on the wrong side).
            self.skip(ir, f"increase skipped: the stop {stop} is not on the losing side of {price}")
            return
        self.emit(
            ir,
            side=side,
            qty=delta,
            purpose=OrderPurpose.INCREASE if same_side else OrderPurpose.OPEN,
            price=price,
            spec=spec,
            stop=stop,
        )

    def reduce(self, ir: InstrumentRuling, held: Decimal, *, forced: bool) -> None:
        """Shrink a position on its own side toward the approved weight."""
        symbol = ir.symbol
        spec = self.inputs.specs.get(symbol)
        price = planning_price(symbol, self.book, self.inputs)
        if spec is None or price is None or self.book.equity <= 0:
            self.skip(ir, "partial reduction not sizeable: no instrument spec, mark or equity")
            return
        target = target_quantity(abs(ir.approved_weight), self.book.equity, price, spec)
        if target >= abs(held):
            return  # rounding leaves the held quantity already within the approved weight
        if target == 0 or _meets_minimums(target, price, spec) is not None:
            # A target that rounds to nothing is a close; a residue below the venue minimum would
            # be stranded where no order could close it, so it is closed now.
            self.close(ir, held, forced=forced)
            return
        delta = abs(held) - target
        short = _meets_minimums(delta, price, spec)
        if short is not None:
            self.skip(ir, f"reduction skipped: {short}")
            return
        self.emit(
            ir,
            side=Side.SELL if held > 0 else Side.BUY,
            qty=delta,
            purpose=OrderPurpose.PROTECTIVE_EXIT if forced else OrderPurpose.REDUCE,
            price=price,
            spec=spec,
            stop=None,
        )

    def plan(self, ir: InstrumentRuling) -> None:
        position = self.book.positions.get(ir.symbol)
        held = position.qty if position is not None else _ZERO
        approved = ir.approved_weight
        if approved == ir.current_weight:
            return  # the kernel kept what is held: nothing to send
        forced = self.ruling.protective_reason is not None or forced_by_guard(ir)
        if held == 0:
            if approved != 0:
                self.add(ir, held)
            return
        if approved == 0 or (approved > 0) != (held > 0):
            self.close(ir, held, forced=forced)
            if approved != 0:
                self.add(ir, _ZERO)  # the open half of a flip
            return
        if abs(approved) > abs(ir.current_weight):
            self.add(ir, held)
        else:
            self.reduce(ir, held, forced=forced)


def plan_id_for(
    ruling_id: str, intents: Sequence[OrderIntent], skipped: Sequence[SkippedLeg]
) -> str:
    """Deterministic plan identity: the ruling, the intents' ids and the skipped legs."""
    return content_hash(
        {"ruling_id": ruling_id, "intents": [i.intent_id for i in intents], "skipped": skipped}
    )


def plan_orders(
    ruling: KernelRuling,
    book: BookState,
    inputs: KernelInputs,
    policy: Policy,
    *,
    now: datetime,
) -> OrderPlan:
    """Turn the approved weights of ``ruling`` into orders. Reducing legs first, then adding legs;
    each group in instrument order, split legs in index order."""
    planner = _Planner(ruling, book, inputs, policy)
    for ir in ruling.instruments:
        planner.plan(ir)
    intents = (*planner.reducing, *planner.adding)
    skipped = tuple(planner.skipped)
    return OrderPlan(
        plan_id=plan_id_for(ruling.ruling_id, intents, skipped),
        ruling_id=ruling.ruling_id,
        created_at=now,
        intents=intents,
        skipped=skipped,
    )


__all__ = [
    "MEASURED_DEMO_TAKER_FEE",
    "build_intent",
    "client_oid",
    "current_weight",
    "entry_price",
    "fee_rate",
    "forced_by_guard",
    "intent_core",
    "order_price",
    "plan_id_for",
    "plan_orders",
    "planning_price",
    "round_qty",
    "split_quantity",
    "stop_price",
    "target_quantity",
    "weight_of",
]
