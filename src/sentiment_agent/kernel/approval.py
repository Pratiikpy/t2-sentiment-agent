"""The approval minter: the only production module that can create an :class:`ApprovedOrder`.

The executor accepts nothing else, so this is the last gate between the model and the venue. It
trusts nothing it is handed. Before minting, it re-derives every fact an order rests on from the
ruling, the book and the market inputs, and refuses the **whole** plan if anything disagrees: a
plan that contains one tampered intent is a plan nobody should send any part of.

Checked, in order:

1. **The ruling is the kernel's.** It re-validates (every only-reduce and self-consistency check in
   ``types.py`` runs again) and its id is the hash of its content (``kernel.ruling_hash``), so a
   ruling edited after the kernel issued it no longer matches.
2. **The plan is this ruling's.** Same ruling id, a plan id that is the hash of its intents and
   skipped legs, and no intent from another ruling or decision.
3. **Each intent is what the planner derives.** Its ``intent_id`` is the hash of its core (ruling,
   symbol, side, quantity, purpose, split index) and its ``clientOid`` follows from that, so a
   quantity or side changed after planning is caught; its reference price is the planning price
   recomputed from the book and inputs; its notional and expected fee follow from those.
4. **Each intent moves the book the way the ruling says**, simulated leg by leg from the book's
   positions:

   * a reducing leg is reduce-only, trades against the held side, never exceeds what is held, and
     belongs to an instrument the ruling reduces, closes or flips;
   * an exposure-adding leg exists only under a model decision (never under a protective ruling),
     trades in the direction of the approved weight, leaves the position within the approved
     weight at the planning price, clears the venue's grid and minimums, stays within the
     market-order cap, and carries a stop on its grid no farther from the expected entry than the
     stop percentage G4 recorded for this ruling.

What this gate cannot know: whether the decision id belongs to a ``DECIDED`` record (the kernel
refuses to rule a proposal without a decision id, and ``DecisionRecord`` carries no proposal
without a decision), and whether the ruling's ceilings were the right policy (the policy is the
kernel's input, not the minter's). Both are enforced upstream and covered by the kernel's tests.
"""

from collections.abc import Sequence
from decimal import Decimal

from pydantic import ValidationError

from sentiment_agent.hashing import content_hash
from sentiment_agent.kernel.kernel import ruling_hash
from sentiment_agent.kernel.planner import (
    entry_price,
    fee_rate,
    intent_core,
    order_price,
    plan_id_for,
    planning_price,
)
from sentiment_agent.types import (
    _MINT_TOKEN,
    ApprovedOrder,
    BookState,
    GuardId,
    InstrumentRuling,
    KernelInputs,
    KernelRuling,
    OrderIntent,
    OrderPlan,
    OrderPurpose,
    Side,
)


class ApprovalError(RuntimeError):
    """The plan does not match its ruling. Nothing in it is minted."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = tuple(problems)
        super().__init__("approval refused: " + "; ".join(self.problems))


def _signed(side: Side, qty: Decimal) -> Decimal:
    return qty if side is Side.BUY else -qty


def _stop_pct(ruling: InstrumentRuling) -> float | None:
    for g in ruling.rulings:
        if g.guard is GuardId.G4_STOP:
            value = g.inputs.get("stop_loss_pct")
            return value if isinstance(value, float) else None
    return None


def _check_ruling(ruling: KernelRuling, plan: OrderPlan, problems: list[str]) -> None:
    # Re-validated from its plain form, so a ruling built without validation is caught, and with
    # its id blanked, which is the form its id is the hash of.
    dump = ruling.model_dump()
    dump["ruling_id"] = ""
    try:
        blank = KernelRuling.model_validate(dump)
    except ValidationError as exc:
        problems.append(f"the ruling fails its own checks: {exc.errors()[0]['msg']}")
    else:
        if ruling_hash(blank) != ruling.ruling_id:
            problems.append(
                "the ruling's content does not hash to its id: it was changed after issue"
            )
    try:
        OrderPlan.model_validate(plan.model_dump())
    except ValidationError as exc:
        problems.append(f"the plan fails its own checks: {exc.errors()[0]['msg']}")
    if plan.ruling_id != ruling.ruling_id:
        problems.append(f"the plan answers ruling {plan.ruling_id}, not {ruling.ruling_id}")
    if plan.plan_id != plan_id_for(plan.ruling_id, plan.intents, plan.skipped):
        problems.append("the plan's content does not hash to its id")


class _Minter:
    def __init__(self, ruling: KernelRuling, book: BookState, inputs: KernelInputs) -> None:
        self.ruling = ruling
        self.book = book
        self.inputs = inputs
        self.problems: list[str] = []
        self.held: dict[str, Decimal] = {s: p.qty for s, p in book.positions.items()}

    def refuse(self, intent: OrderIntent, text: str) -> None:
        self.problems.append(f"{intent.client_oid} ({intent.symbol}): {text}")

    def check(self, intent: OrderIntent) -> None:
        ruling = self.ruling
        if intent.ruling_id != ruling.ruling_id:
            self.refuse(intent, "planned under another ruling")
            return
        if intent.decision_id != ruling.decision_id:
            self.refuse(intent, "carries another decision id than its ruling")
        core = intent_core(
            ruling_id=intent.ruling_id,
            symbol=intent.symbol,
            side=intent.side,
            qty=intent.qty,
            purpose=intent.purpose,
            split_index=intent.split_index,
        )
        if intent.intent_id != content_hash(core):
            self.refuse(intent, "its content does not hash to its intent_id: it was changed")
            return
        ir = ruling.instrument(intent.symbol)
        if ir is None:
            self.refuse(intent, "the ruling has no ruling for this instrument")
            return
        price = (
            planning_price(intent.symbol, self.book, self.inputs)
            if intent.purpose.adds_exposure
            else order_price(intent.symbol, self.book, self.inputs)
        )
        if price is None or price != intent.reference_price:
            self.refuse(
                intent,
                f"reference price {intent.reference_price} is not the planning price {price}",
            )
            return
        spec = self.inputs.specs.get(intent.symbol)
        if intent.notional != intent.qty * price:
            self.refuse(intent, "notional is not quantity times the reference price")
        if intent.expected_fee != intent.notional * fee_rate(spec):
            self.refuse(intent, "expected fee does not follow from the notional and the fee rate")
        before = self.held.get(intent.symbol, Decimal(0))
        after = before + _signed(intent.side, intent.qty)
        if intent.purpose.adds_exposure:
            self.check_adding(intent, ir, before, after, price)
        else:
            self.check_reducing(intent, ir, before, after)
        self.held[intent.symbol] = after

    def check_reducing(
        self, intent: OrderIntent, ir: InstrumentRuling, before: Decimal, after: Decimal
    ) -> None:
        current, approved = ir.current_weight, ir.approved_weight
        reduces = current != 0 and (
            approved == 0 or (approved > 0) != (current > 0) or abs(approved) < abs(current)
        )
        if not reduces:
            self.refuse(intent, "the ruling does not reduce this instrument")
        if before == 0 or (intent.side is Side.SELL) != (before > 0):
            self.refuse(intent, f"a reducing {intent.side} leg against a position of {before}")
            return
        if intent.qty > abs(before):
            self.refuse(intent, f"reduces {intent.qty}, more than the {abs(before)} held")
        last_leg = intent.split_index == intent.split_count - 1
        if intent.purpose is OrderPurpose.CLOSE and last_leg and after != 0:
            self.refuse(intent, "a close that leaves a position")
        if not last_leg and after == 0:
            self.refuse(intent, "an early leg of a split closes the whole position")
        if intent.purpose is OrderPurpose.REDUCE and after == 0:
            self.refuse(intent, "a partial reduction that closes the position")
        if intent.purpose is not OrderPurpose.PROTECTIVE_EXIT and self.ruling.decision_id is None:
            self.refuse(intent, "a protective ruling's legs are protective exits")
        spec = self.inputs.specs.get(intent.symbol)
        cap = spec.max_market_order_qty if spec is not None else None
        if cap is not None and cap > 0 and intent.qty > cap:
            self.refuse(intent, f"{intent.qty} is above maxMarketOrderQty {cap}")

    def check_adding(
        self,
        intent: OrderIntent,
        ir: InstrumentRuling,
        before: Decimal,
        after: Decimal,
        price: Decimal,
    ) -> None:
        ruling = self.ruling
        approved, current = ir.approved_weight, ir.current_weight
        if ruling.decision_id is None or ruling.protective_reason is not None:
            self.refuse(intent, "adds exposure without a model decision")
            return
        grows = approved != 0 and (
            current == 0 or (approved > 0) != (current > 0) or abs(approved) > abs(current)
        )
        if not grows:
            self.refuse(intent, "the ruling approves no increase for this instrument")
            return
        if (intent.side is Side.BUY) != (approved > 0):
            self.refuse(intent, f"a {intent.side} leg against an approved weight of {approved}")
            return
        # An open starts from flat; the later legs of a split open, like every increase, continue
        # a position already on the approved side.
        opens_from_flat = intent.purpose is OrderPurpose.OPEN and intent.split_index == 0
        if opens_from_flat and before != 0:
            self.refuse(intent, f"an open against a position of {before}")
        if not opens_from_flat and (before == 0 or (before > 0) != (approved > 0)):
            self.refuse(intent, f"a continuing {intent.purpose} leg against a position of {before}")
        equity = self.book.equity
        if equity <= 0:
            self.refuse(intent, "no positive equity to measure the approved weight against")
            return
        post = abs(after) * price / equity
        if post > Decimal(repr(abs(approved))):
            self.refuse(
                intent, f"leaves |weight| {post} above the approved {abs(approved)}: exceeds it"
            )
        spec = self.inputs.specs.get(intent.symbol)
        if spec is None:
            self.refuse(intent, "no Demo instrument spec to check the order against")
            return
        if intent.qty % spec.qty_step != 0:
            self.refuse(intent, f"{intent.qty} is off the {spec.qty_step} quantity grid")
        if intent.qty < spec.min_order_qty:
            self.refuse(intent, f"{intent.qty} is below minOrderQty {spec.min_order_qty}")
        if intent.qty * price < spec.min_order_amount:
            self.refuse(intent, f"notional is below minOrderAmount {spec.min_order_amount}")
        cap = spec.max_market_order_qty
        if cap is not None and cap > 0 and intent.qty > cap:
            self.refuse(intent, f"{intent.qty} is above maxMarketOrderQty {cap}")
        self.check_stop(intent, ir, price, spec.price_step)

    def check_stop(
        self, intent: OrderIntent, ir: InstrumentRuling, price: Decimal, price_step: Decimal
    ) -> None:
        stop = intent.stop_loss_price
        if stop is None:  # pragma: no cover - OrderIntent refuses an adding leg without a stop
            self.refuse(intent, "an exposure-adding leg without a stop")
            return
        if stop % price_step != 0:
            self.refuse(intent, f"stop {stop} is off the {price_step} price grid")
        entry = entry_price(intent.side, self.inputs.demo_quotes.get(intent.symbol), price)
        losing = stop < entry if intent.side is Side.BUY else stop > entry
        if not losing:
            self.refuse(intent, f"stop {stop} is not on the losing side of the entry {entry}")
            return
        if GuardId.G4_STOP in self.ruling.guards_applied:
            pct = _stop_pct(ir)
            if pct is None:
                self.refuse(intent, "G4 recorded no stop percentage for this instrument")
            elif abs(entry - stop) / entry > Decimal(repr(pct)):
                self.refuse(
                    intent,
                    f"stop {stop} is farther than {pct:.2%} from the expected entry {entry}",
                )


def approve(
    plan: OrderPlan, ruling: KernelRuling, book: BookState, inputs: KernelInputs
) -> tuple[ApprovedOrder, ...]:
    """Mint one :class:`ApprovedOrder` per intent in ``plan``, in plan order, or raise
    :class:`ApprovalError` naming every problem found and mint nothing."""
    problems: list[str] = []
    _check_ruling(ruling, plan, problems)
    minter = _Minter(ruling, book, inputs)
    for intent in plan.intents:
        minter.check(intent)
    problems.extend(minter.problems)
    if problems:
        raise ApprovalError(problems)
    return tuple(ApprovedOrder(i, ruling.ruling_id, _token=_MINT_TOKEN) for i in plan.intents)


__all__ = ["ApprovalError", "approve"]
