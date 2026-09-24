"""The planner: approved weights become the legs the venue will be asked to fill.

A port of the primary's ``sentiment_agent.kernel.planner`` rules onto what ``getagent.trade``
documents (``references/sdk/trade/contract.md``, ``patterns.md``):

* **One price.** Weights, target quantities and notionals use the planning price (the data layer's
  mark), against the book's equity, as the kernel did.
* **Round the target toward zero, not the order**, so an increase comes out slightly smaller and a
  reduction slightly larger than approved; never the other way.
* **Increases below the venue minimums are skipped and recorded**, never rounded up.
* **A flip is a close, then an open.** Reducing legs go first; every opening leg carries its preset
  stop ``STOP_LOSS_PCT`` from the expected entry, rounded toward the entry.
* **Reductions are never blocked by size.** A reduction that would leave a residue below the venue
  minimum closes the position instead.

What differs, and why: the trade SDK has documented calls to close a whole position
(``close_position``) and to open one (``open_long_market``/``open_short_market`` with a
preset stop), but the literals ``place_order`` needs for a reduce-only partial order
(``trade_side``, ``pos_side``) are not documented. The replica does not guess them. A partial
reduction is a close of the whole position followed by an open of the remainder with a fresh
stop; it pays the taker fee on the remainder twice, and the book keeps it one trade
(``Book.apply(continues_trade=...)``). In follow-trade mode the venue sizes each leg from its
notional through ``compute_qty``, as the SDK requires; the planned quantity here drives the
replica's own paper fills in signal-only mode.

Every leg's intent is emitted as canonical JSON (:mod:`.canonical`) and hashed outside the sandbox.
"""

from dataclasses import dataclass, field
from decimal import Decimal

from . import policy_v1 as policy
from .canonical import canonical
from .kernel import (
    Inputs,
    InstrumentRuling,
    KernelRuling,
    Limits,
    entry_price,
    planning_price,
    round_qty,
    stop_price,
    target_quantity,
)

ZERO = Decimal(0)

CLOSE = "close"
OPEN = "open"
INCREASE = "increase"
REDUCE_CLOSE = "reduce_close"
REDUCE_REOPEN = "reduce_reopen"


@dataclass(frozen=True)
class Leg:
    symbol: str
    action: str
    side: str
    qty: Decimal
    notional: Decimal
    reference_price: Decimal
    stop: Decimal | None
    purpose: str
    """``model`` or ``protective``."""

    @property
    def adds_exposure(self) -> bool:
        return self.action in (OPEN, INCREASE, REDUCE_REOPEN)

    def intent(self, *, run_id: str, seq: int, decision_ref: str | None) -> dict[str, object]:
        return {
            "record": "t2sa-replica-intent",
            "policy_hash": policy.POLICY_HASH,
            "run_id": run_id,
            "seq": seq,
            "decision_ref": decision_ref,
            "symbol": self.symbol,
            "action": self.action,
            "side": self.side,
            "qty_planned": str(self.qty),
            "notional": str(self.notional),
            "reference_price": str(self.reference_price),
            "stop_loss_price": None if self.stop is None else str(self.stop),
            "reduce_only": not self.adds_exposure,
            "order_type": "market",
            "purpose": self.purpose,
        }


@dataclass
class SymbolPlan:
    symbol: str
    ruling: InstrumentRuling
    legs: list[Leg] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def reduces(self) -> bool:
        return any(not leg.adds_exposure for leg in self.legs) or abs(self.ruling.approved) < abs(
            self.ruling.current
        )

    @property
    def signal_action(self) -> str:
        approved = self.ruling.approved
        return "long" if approved > 0 else "short" if approved < 0 else "close"

    def to_json(
        self, *, run_id: str, first_seq: int, decision_ref: str | None
    ) -> dict[str, object]:
        intents = [
            leg.intent(run_id=run_id, seq=first_seq + i, decision_ref=decision_ref)
            for i, leg in enumerate(self.legs)
        ]
        return {
            "symbol": self.symbol,
            "current_weight": self.ruling.current,
            "approved_weight": self.ruling.approved,
            "binding_guard": self.ruling.binding_guard,
            "signal_action": self.signal_action,
            "intents": [{"canonical": canonical(i), "intent": i} for i in intents],
            "skipped": list(self.skipped),
        }


def _meets_minimums(qty: Decimal, price: Decimal, limits: Limits) -> str | None:
    if qty < limits.min_qty:
        return f"{qty} is below the minimum quantity {limits.min_qty}"
    if qty * price < limits.min_amount:
        return f"notional {qty * price} is below the minimum {limits.min_amount}"
    return None


def _open_leg(
    plan: SymbolPlan,
    *,
    action: str,
    side: str,
    qty: Decimal,
    price: Decimal,
    inputs: Inputs,
    limits: Limits,
    purpose: str,
) -> None:
    quote = inputs.quotes.get(plan.symbol)
    entry = entry_price(side, quote, price)
    try:
        stop = stop_price(entry, side, limits.price_step)
    except ValueError as exc:
        plan.skipped.append(f"{action} skipped: {exc}")
        return
    if not (stop < price if side == "buy" else stop > price):
        plan.skipped.append(
            f"{action} skipped: the stop {stop} is not on the losing side of {price}"
        )
        return
    plan.legs.append(Leg(plan.symbol, action, side, qty, qty * price, price, stop, purpose))


def plan_symbol(
    ruling: InstrumentRuling,
    *,
    held: Decimal,
    equity: Decimal,
    inputs: Inputs,
    limits: Limits | None,
    book_price: Decimal | None,
    protective: bool,
) -> SymbolPlan:
    """The legs that move ``held`` toward ``ruling.approved``."""
    plan = SymbolPlan(symbol=ruling.symbol, ruling=ruling)
    approved = ruling.approved
    if approved == ruling.current:
        return plan
    purpose = "protective" if protective or ruling.forced_exit else "model"
    symbol = ruling.symbol
    price = planning_price(symbol, inputs)
    close_price = price if price is not None else book_price
    if held != 0 and (approved == 0 or (approved > 0) != (held > 0)):
        if close_price is None or close_price <= 0:
            plan.skipped.append("close not expressible: no mark and no positive entry")
            return plan
        side = "sell" if held > 0 else "buy"
        plan.legs.append(
            Leg(symbol, CLOSE, side, abs(held), abs(held) * close_price, close_price, None, purpose)
        )
        if approved == 0:
            return plan
        held = ZERO
    if limits is None:
        plan.skipped.append(
            "no instrument limits: an increase or partial reduction cannot be sized"
        )
        return plan
    if price is None or equity <= 0:
        plan.skipped.append("no mark or no positive equity: the leg cannot be sized")
        return plan
    side = "buy" if approved > 0 else "sell"
    target = target_quantity(abs(approved), equity, price, limits)
    if held == 0:
        short = (
            "the target rounds to nothing"
            if target <= 0
            else _meets_minimums(target, price, limits)
        )
        if short is not None:
            plan.skipped.append(f"open skipped: {short}")
            return plan
        _open_leg(
            plan,
            action=OPEN,
            side=side,
            qty=target,
            price=price,
            inputs=inputs,
            limits=limits,
            purpose=purpose,
        )
        return plan
    if abs(approved) > abs(ruling.current):
        delta = round_qty(target - abs(held), limits)
        short = (
            "the increase rounds to nothing"
            if delta <= 0
            else _meets_minimums(delta, price, limits)
        )
        if short is not None:
            plan.skipped.append(f"increase skipped: {short}")
            return plan
        _open_leg(
            plan,
            action=INCREASE,
            side=side,
            qty=delta,
            price=price,
            inputs=inputs,
            limits=limits,
            purpose=purpose,
        )
        return plan
    if target >= abs(held):
        return plan
    close_side = "sell" if held > 0 else "buy"
    if target == 0 or _meets_minimums(target, price, limits) is not None:
        plan.legs.append(
            Leg(symbol, CLOSE, close_side, abs(held), abs(held) * price, price, None, purpose)
        )
        return plan
    reduction = abs(held) - target
    short = _meets_minimums(reduction, price, limits)
    if short is not None:
        plan.skipped.append(f"reduction skipped: {short}")
        return plan
    plan.legs.append(
        Leg(symbol, REDUCE_CLOSE, close_side, abs(held), abs(held) * price, price, None, purpose)
    )
    _open_leg(
        plan,
        action=REDUCE_REOPEN,
        side=side,
        qty=target,
        price=price,
        inputs=inputs,
        limits=limits,
        purpose=purpose,
    )
    if plan.legs[-1].action != REDUCE_REOPEN:
        plan.skipped.append(
            "the remainder could not be re-opened with a valid stop: the position closes"
        )
    return plan


def plan_ruling(
    ruling: KernelRuling,
    *,
    held: dict[str, Decimal],
    entries: dict[str, Decimal],
    equity: Decimal,
    inputs: Inputs,
    limits: dict[str, Limits],
) -> list[SymbolPlan]:
    """One plan per instrument the ruling moves: reducing plans first, then adding plans, each
    group in symbol order."""
    plans: list[SymbolPlan] = []
    for instrument in ruling.instruments:
        plan = plan_symbol(
            instrument,
            held=held.get(instrument.symbol, ZERO),
            equity=equity,
            inputs=inputs,
            limits=limits.get(instrument.symbol),
            book_price=entries.get(instrument.symbol),
            protective=ruling.protective_reason is not None,
        )
        if plan.legs or plan.skipped:
            plans.append(plan)
    return sorted(plans, key=lambda p: (0 if p.reduces else 1, p.symbol))


__all__ = [
    "CLOSE",
    "INCREASE",
    "OPEN",
    "REDUCE_CLOSE",
    "REDUCE_REOPEN",
    "Leg",
    "SymbolPlan",
    "plan_ruling",
    "plan_symbol",
]
