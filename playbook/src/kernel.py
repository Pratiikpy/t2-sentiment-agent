"""The risk kernel, ported: the model proposes, the kernel can only reduce.

This is the primary agent's kernel (``sentiment_agent.kernel.guards`` and ``kernel.kernel``, MIT,
same author) rewritten for the sandbox, which has no pydantic in its author contract. Every number
it enforces is read from ``policy_v1`` (generated from the primary's ``POLICY_V1``), and the rules
are the same functions guard by guard:

* **The invariant.** For every instrument, with ``reference`` the proposed weight (or the current
  weight when nothing was proposed), the approved weight has the sign of the reference or is zero,
  and its magnitude is at most the reference's. :class:`InstrumentRuling` refuses construction
  otherwise, exactly as the primary's contract type does.
* **Evaluate all, bind the minimum.** Every applied guard contributes a ceiling on ``|weight|`` (a
  forced exit is a ceiling of zero); the approved weight is the reference clipped to the smallest;
  the binding guard's reason names every other guard that was also below the request. Three passes:
  every guard but G3 and G11's sizing; G3 per name and gross; G11's venue minimums on what is left.
* **Only an increase can be refused.** A refusal is a ceiling equal to what is already held on the
  proposal's side, so a reduction or a close is always within it. A missing input is a refusal
  (fail-closed) reported as ``not_evaluated``.

Where the replica's inputs differ from the primary's, the guard reads the replica's nearest
equivalent, and says so in its ``inputs``:

* G1 compares the data layer's mark with its index (the primary: Demo mark and Demo index), and the
  execution venue's last price with the data layer's (the primary: Demo last and live last) when
  the venue is the trade proxy; with no trade proxy (signal-only) the venue *is* the data layer and
  the gap is not applicable. The stale-venue refusal reads the mean 1-hour move of the data
  layer's 1H closes over three hours (the primary: the Demo index's).
* G11 reads this package's instrument limits (``policy_v1.INSTRUMENT_LIMITS``, the live Bitget rows
  of the primary's ``universe_probe.json``) and the subscriber's configured symbols; the trade
  proxy's own ``compute_qty`` has the last word when an order is sized.
* ``BookView.history_known`` is the replica's addition: when ``.state/`` was lost while positions
  are open, the day-open equity, peak, fees and hold times are unknown, and G5, G6, G7 and G10 then
  refuse every increase as not evaluated.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import ROUND_CEILING, ROUND_DOWN, ROUND_FLOOR, Decimal

from . import policy_v1 as policy
from .breaker import ACTIVE, HALTED, REDUCE_ONLY, BreakerState, book_conditions, most_severe
from .sessions import as_utc, session_open, weekend_phase

EPS = 1e-12

G1 = "G1_venue_integrity"
G2 = "G2_weekend_freeze"
G3 = "G3_size"
G4 = "G4_stop"
G5 = "G5_daily_kill"
G6 = "G6_turnover"
G7 = "G7_fee_budget"
G8 = "G8_taker_only"
G9 = "G9_grounding"
G10 = "G10_breaker"
G11 = "G11_eligibility"

GUARD_ORDER: tuple[str, ...] = tuple(policy.GUARD_IDS)
ALL_GUARDS: frozenset[str] = frozenset(GUARD_ORDER)
PROTECTIVE_GUARDS: frozenset[str] = frozenset({G1, G2, G5, G10})

PASSED = "passed"
FIRED = "fired"
NOT_EVALUATED = "not_evaluated"
NOT_APPLICABLE = "not_applicable"

PROTECTIVE_PRIORITY = ("daily_kill", "llm_outage", "breaker", "venue_integrity", "weekend_freeze")

_ASSET_CLASS: dict[str, str] = {symbol: asset for symbol, asset, _ in policy.UNIVERSE}
_GAP_P99: dict[str, float] = {symbol: gap for symbol, _, gap in policy.UNIVERSE}


class KernelError(ValueError):
    """The kernel was asked to rule on something it must not rule on."""


# ------------------------------------------------------------------------------------------------
# Inputs
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Quote:
    """One instrument's prices from the data layer (``getagent.data``, exchange ``bitget``)."""

    symbol: str
    mark: Decimal | None
    index: Decimal | None
    last: Decimal | None
    bid: Decimal | None
    ask: Decimal | None
    ts: datetime | None
    fetched_at: datetime

    @property
    def spread_bps(self) -> float | None:
        if self.bid is None or self.ask is None:
            return None
        mid = (self.bid + self.ask) / 2
        return float((self.ask - self.bid) / mid * 10_000) if mid > 0 else float("inf")

    @property
    def mark_index_gap(self) -> float | None:
        if self.mark is None or self.index is None:
            return None
        return float(abs(self.mark / self.index - 1)) if self.index > 0 else float("inf")

    @property
    def observed_at(self) -> datetime:
        """The older of the feed's own timestamp and the fetch, as the primary ages a quote."""
        return self.fetched_at if self.ts is None else min(self.ts, self.fetched_at)


@dataclass(frozen=True)
class Limits:
    """An instrument's order limits: quantity grid and minimums, and the price grid when known."""

    min_qty: Decimal
    qty_step: Decimal
    min_amount: Decimal
    price_step: Decimal | None = None


@dataclass(frozen=True)
class PositionView:
    qty: Decimal
    """Signed base quantity; positive is long."""
    avg_entry: Decimal
    last_increase_at: datetime | None

    @property
    def is_flat(self) -> bool:
        return self.qty == 0


@dataclass(frozen=True)
class BookView:
    equity: Decimal
    starting_equity: Decimal
    peak_equity: Decimal
    day_open_equity: Decimal
    fees_today: Decimal
    fees_total: Decimal
    consecutive_losses: int
    positions: Mapping[str, PositionView]
    rebalances_today: Mapping[str, int]
    history_known: bool = True
    """False when the persisted history was lost while positions are open (see module doc)."""

    @property
    def drawdown(self) -> float:
        return float(self.equity / self.peak_equity - 1) if self.peak_equity > 0 else 0.0

    @property
    def day_return(self) -> float | None:
        if not self.history_known or self.day_open_equity <= 0:
            return None
        return float(self.equity / self.day_open_equity - 1)

    def held(self) -> list[str]:
        return sorted(s for s, p in self.positions.items() if not p.is_flat)


@dataclass(frozen=True)
class Inputs:
    at: datetime
    quotes: Mapping[str, Quote]
    move_bps_3h: Mapping[str, float | None]
    limits: Mapping[str, Limits]
    snapshot_taken_at: datetime | None
    configured: tuple[str, ...]
    """Symbols this subscription may trade: the policy universe narrowed by the subscriber."""
    venue_last: Mapping[str, Decimal] = field(default_factory=dict)
    """The trade proxy's own last price, when trading through it (follow-trade)."""
    venue_is_data_layer: bool = True


# ------------------------------------------------------------------------------------------------
# Rulings
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GuardRuling:
    guard: str
    symbol: str | None
    status: str
    reason: str
    ceiling: float | None = None
    forces_exit: bool = False
    inputs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.forces_exit and self.status != FIRED:
            raise KernelError(f"{self.guard}: a guard that forces an exit has fired")
        if self.ceiling is not None and not (math.isfinite(self.ceiling) and self.ceiling >= 0):
            raise KernelError(f"{self.guard}: a ceiling must be a finite non-negative weight")

    @property
    def effective_ceiling(self) -> float | None:
        return 0.0 if self.forces_exit else self.ceiling

    def to_json(self) -> dict[str, object]:
        return {
            "guard": self.guard,
            "symbol": self.symbol,
            "status": self.status,
            "ceiling": self.ceiling,
            "forces_exit": self.forces_exit,
            "reason": self.reason,
            "inputs": dict(self.inputs),
        }


@dataclass(frozen=True)
class InstrumentRuling:
    symbol: str
    current: float
    proposed: float | None
    approved: float
    binding_guard: str | None
    rulings: tuple[GuardRuling, ...]

    def __post_init__(self) -> None:
        for value in (self.current, self.approved, self.proposed):
            if value is not None and not math.isfinite(value):
                raise KernelError(f"{self.symbol}: weights must be finite")
        reference = self.reference
        if abs(self.approved) > abs(reference) + EPS:
            raise KernelError(
                f"{self.symbol}: approved {self.approved} exceeds reference {reference}; "
                "the kernel may only reduce"
            )
        if self.approved != 0 and (self.approved > 0) != (reference > 0):
            raise KernelError(f"{self.symbol}: the kernel may never turn a side")
        for ruling in self.rulings:
            if ruling.symbol != self.symbol:
                raise KernelError(f"{self.symbol}: carries a ruling for {ruling.symbol!r}")
            ceiling = ruling.effective_ceiling
            if ceiling is not None and abs(self.approved) > ceiling + EPS:
                raise KernelError(f"{self.symbol}: approved breaches {ruling.guard}")
        if self.changed_by_kernel and self.binding_guard is None:
            raise KernelError(
                f"{self.symbol}: the kernel changed the weight without naming a guard"
            )

    @property
    def reference(self) -> float:
        return self.current if self.proposed is None else self.proposed

    @property
    def changed_by_kernel(self) -> bool:
        return abs(self.approved - self.reference) > EPS

    @property
    def forced_exit(self) -> bool:
        """The kernel moved this instrument because a guard forced an exit."""
        if self.binding_guard is None or not self.changed_by_kernel:
            return False
        return any(g.guard == self.binding_guard and g.forces_exit for g in self.rulings)

    def to_json(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "current_weight": self.current,
            "proposed_weight": self.proposed,
            "approved_weight": self.approved,
            "binding_guard": self.binding_guard,
            "rulings": [r.to_json() for r in self.rulings],
        }


@dataclass(frozen=True)
class KernelRuling:
    at: datetime
    decision_ref: str | None
    protective_reason: str | None
    activation_before: str
    activation_after: str
    book_rulings: tuple[GuardRuling, ...]
    instruments: tuple[InstrumentRuling, ...]
    guards_applied: tuple[str, ...]

    def __post_init__(self) -> None:
        if (self.decision_ref is None) == (self.protective_reason is None):
            raise KernelError("a ruling answers either a model decision or a protective reason")
        if self.protective_reason is not None and any(
            i.proposed is not None for i in self.instruments
        ):
            raise KernelError("a protective ruling has no model proposal to rule on")
        for inst in self.instruments:
            for ruling in self.book_rulings:
                ceiling = ruling.effective_ceiling
                if ceiling is not None and abs(inst.approved) > ceiling + EPS:
                    raise KernelError(f"{inst.symbol}: approved breaches book-level {ruling.guard}")

    def approved(self) -> dict[str, float]:
        return {i.symbol: i.approved for i in self.instruments}

    def to_json(self) -> dict[str, object]:
        return {
            "at": self.at.isoformat(),
            "decision_ref": self.decision_ref,
            "protective_reason": self.protective_reason,
            "activation_before": self.activation_before,
            "activation_after": self.activation_after,
            "guards_applied": list(self.guards_applied),
            "book_rulings": [r.to_json() for r in self.book_rulings],
            "instruments": [i.to_json() for i in self.instruments],
        }


# ------------------------------------------------------------------------------------------------
# Legs
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Leg:
    symbol: str
    current: float
    proposed: float | None = None

    @property
    def reference(self) -> float:
        return self.current if self.proposed is None else self.proposed

    @property
    def hold_ceiling(self) -> float:
        return abs(self.current) if self.current * self.reference > 0 else 0.0

    @property
    def adds_exposure(self) -> bool:
        return self.reference != 0 and abs(self.reference) > self.hold_ceiling + EPS

    @property
    def inert(self) -> bool:
        return self.current == 0 and self.reference == 0


# ------------------------------------------------------------------------------------------------
# Prices, weights and the price grid (the primary's planner helpers)
# ------------------------------------------------------------------------------------------------


def _to_grid(value: Decimal, step: Decimal, rounding: str) -> Decimal:
    if step <= 0:
        raise ValueError(f"grid step must be positive, got {step}")
    result = (value / step).to_integral_value(rounding=rounding) * step
    if result == 0:
        result = abs(result)
    exponent = step.as_tuple().exponent
    if isinstance(exponent, int) and exponent <= 0:
        result = result.quantize(step)
    return result


def round_qty(qty: Decimal, limits: Limits) -> Decimal:
    return _to_grid(qty, limits.qty_step, ROUND_DOWN)


def stop_price(entry: Decimal, side: str, price_step: Decimal | None) -> Decimal:
    """The preset stop ``STOP_LOSS_PCT`` from ``entry``, rounded toward the entry on the price grid
    (never looser than the policy). Raises ``ValueError`` when no stop fits on the losing side."""
    if entry <= 0:
        raise ValueError(f"entry price must be positive, got {entry}")
    pct = Decimal(repr(policy.STOP_LOSS_PCT))
    if side == "buy":
        raw = entry * (1 - pct)
        stop = raw if price_step is None else _to_grid(raw, price_step, ROUND_CEILING)
        if not Decimal(0) < stop < entry:
            raise ValueError(f"no long stop fits between 0 and the entry {entry}")
    else:
        raw = entry * (1 + pct)
        stop = raw if price_step is None else _to_grid(raw, price_step, ROUND_FLOOR)
        if stop <= entry:
            raise ValueError(f"no short stop fits above the entry {entry}")
    return stop


def entry_price(side: str, quote: Quote | None, fallback: Decimal) -> Decimal:
    """Expected fill of a market order: the ask for a buy, the bid for a sell, else ``fallback``."""
    if quote is not None:
        touch = quote.ask if side == "buy" else quote.bid
        if touch is not None and touch > 0:
            return touch
    return fallback


def planning_price(symbol: str, inputs: Inputs) -> Decimal | None:
    quote = inputs.quotes.get(symbol)
    if quote is not None and quote.mark is not None and quote.mark > 0:
        return quote.mark
    return None


def order_price(symbol: str, book: BookView, inputs: Inputs) -> Decimal | None:
    price = planning_price(symbol, inputs)
    if price is not None:
        return price
    position = book.positions.get(symbol)
    if position is not None and position.avg_entry > 0:
        return position.avg_entry
    return None


def weight_of(qty: Decimal, price: Decimal, equity: Decimal) -> float:
    if qty == 0:
        return 0.0
    if equity <= 0:
        return 1.0 if qty > 0 else -1.0
    return float(qty * price / equity)


def current_weight(symbol: str, book: BookView, inputs: Inputs) -> float:
    position = book.positions.get(symbol)
    if position is None or position.is_flat:
        return 0.0
    price = order_price(symbol, book, inputs)
    if price is None:
        return 1.0 if position.qty > 0 else -1.0
    return weight_of(position.qty, price, book.equity)


def target_quantity(abs_weight: float, equity: Decimal, price: Decimal, limits: Limits) -> Decimal:
    if abs_weight <= 0 or equity <= 0 or price <= 0:
        return round_qty(Decimal(0), limits)
    return round_qty(Decimal(repr(abs_weight)) * equity / price, limits)


# ------------------------------------------------------------------------------------------------
# Shared construction
# ------------------------------------------------------------------------------------------------


def _num(value: float | Decimal | None) -> object:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else str(number)


def _pct(value: float) -> str:
    return f"{value:.2%}" if math.isfinite(value) else str(value)


def _conclude(
    guard: str,
    leg: Leg,
    *,
    inputs: Mapping[str, object],
    ok: str,
    exits: Sequence[str] = (),
    refusals: Sequence[str] = (),
    missing: Sequence[str] = (),
) -> GuardRuling:
    """Precedence: exit > refusal of an actual increase > missing input > idle refusal > pass."""
    if exits:
        text = "; ".join(exits)
        if leg.inert:
            return GuardRuling(
                guard,
                leg.symbol,
                PASSED,
                f"{text}; nothing held or proposed, so nothing to exit",
                ceiling=0.0,
                inputs=inputs,
            )
        return GuardRuling(
            guard, leg.symbol, FIRED, f"exit: {text}", ceiling=0.0, forces_exit=True, inputs=inputs
        )
    missing_note = f"; not evaluated: missing {', '.join(missing)}" if missing else ""
    if refusals and leg.adds_exposure:
        return GuardRuling(
            guard,
            leg.symbol,
            FIRED,
            f"no increase: {'; '.join(refusals)}{missing_note}",
            ceiling=leg.hold_ceiling,
            inputs=inputs,
        )
    if missing:
        refusal_note = f"; also {'; '.join(refusals)}" if refusals else ""
        return GuardRuling(
            guard,
            leg.symbol,
            NOT_EVALUATED,
            f"not evaluated, so no increase is permitted: missing {', '.join(missing)}"
            f"{refusal_note}",
            ceiling=leg.hold_ceiling,
            inputs=inputs,
        )
    if refusals:
        return GuardRuling(
            guard,
            leg.symbol,
            PASSED,
            f"{'; '.join(refusals)}; this would refuse an increase, and none was requested",
            ceiling=leg.hold_ceiling,
            inputs=inputs,
        )
    return GuardRuling(guard, leg.symbol, PASSED, ok, inputs=inputs)


def _conclude_book(
    guard: str,
    *,
    inputs: Mapping[str, object],
    ok: str,
    exits: Sequence[str] = (),
    refusals: Sequence[str] = (),
    missing: Sequence[str] = (),
) -> GuardRuling:
    if exits:
        return GuardRuling(
            guard,
            None,
            FIRED,
            f"exit everything: {'; '.join(exits)}",
            ceiling=0.0,
            forces_exit=True,
            inputs=inputs,
        )
    if refusals:
        note = f"; not evaluated: missing {', '.join(missing)}" if missing else ""
        return GuardRuling(
            guard,
            None,
            FIRED,
            f"no new exposure anywhere: {'; '.join(refusals)}{note}",
            inputs=inputs,
        )
    if missing:
        return GuardRuling(
            guard,
            None,
            NOT_EVALUATED,
            f"not evaluated, so no increase is permitted: missing {', '.join(missing)}",
            inputs=inputs,
        )
    return GuardRuling(guard, None, PASSED, ok, inputs=inputs)


# ------------------------------------------------------------------------------------------------
# G1-G11
# ------------------------------------------------------------------------------------------------


def g1_venue_integrity(leg: Leg, *, inputs: Inputs) -> GuardRuling:
    exits: list[str] = []
    refusals: list[str] = []
    missing: list[str] = []
    limit = policy.MARK_INDEX_MAX_GAP
    floor = policy.STALE_INDEX_MIN_MOVE_BPS_3H
    symbol = leg.symbol
    quote = inputs.quotes.get(symbol)
    values: dict[str, object] = {
        "mark_index_limit": limit,
        "stale_index_min_move_bps_3h": floor,
        "venue": "data_layer" if inputs.venue_is_data_layer else "trade_proxy",
    }
    gap = None if quote is None else quote.mark_index_gap
    if gap is None:
        missing.append("the mark and index")
    else:
        values.update(
            mark=None if quote is None else str(quote.mark),
            index=None if quote is None else str(quote.index),
            mark_index_gap=_num(gap),
        )
        if gap > limit:
            exits.append(f"mark is {_pct(gap)} from its index, beyond the {_pct(limit)} limit")
    p99 = _GAP_P99.get(symbol)
    if p99 is None:
        missing.append("a measured venue gap p99 (symbol outside the universe)")
    elif not inputs.venue_is_data_layer:
        values["venue_gap_p99_bps"] = p99
        venue = inputs.venue_last.get(symbol)
        reference = None if quote is None else quote.last
        if venue is None or reference is None:
            missing.append("the venue's and the data layer's last prices")
        elif venue <= 0 or reference <= 0:
            missing.append("a positive last price on both the venue and the data layer")
        else:
            gap_bps = float(abs(venue / reference - 1) * 10_000)
            values.update(venue_last=str(venue), data_last=str(reference), venue_gap_bps=gap_bps)
            if gap_bps > p99:
                exits.append(
                    f"venue last {venue} is {gap_bps:.1f} bps from the data layer's {reference}, "
                    f"beyond the measured p99 of {p99} bps"
                )
    else:
        values["venue_gap"] = "not applicable: the venue is the data layer"
    asset = _ASSET_CLASS.get(symbol)
    is_open = session_open(asset, inputs.at)
    values["session_open"] = is_open
    if is_open:
        move = inputs.move_bps_3h.get(symbol)
        if move is None or not math.isfinite(move):
            missing.append("the 3h price move")
        else:
            values["move_bps_3h"] = move
            if move < floor:
                refusals.append(
                    f"price moved {move:.2f} bps/h over 3h, below {floor} bps/h while its session "
                    "should be open (stale venue)"
                )
    return _conclude(
        G1,
        leg,
        inputs=values,
        exits=exits,
        refusals=refusals,
        missing=missing,
        ok="mark within the index limit, venue gap within p99, price moving",
    )


def g2_weekend_freeze(leg: Leg, *, at: datetime) -> GuardRuling:
    asset = _ASSET_CLASS.get(leg.symbol)
    if asset is None:
        return _conclude(
            G2, leg, inputs={}, missing=("the asset class (symbol outside the universe)",), ok=""
        )
    if asset not in policy.US_SESSION_ASSET_CLASSES:
        return GuardRuling(
            G2,
            leg.symbol,
            NOT_APPLICABLE,
            f"{asset} trades through the weekend",
            inputs={"asset_class": asset},
        )
    phase = weekend_phase(at)
    values: dict[str, object] = {"asset_class": asset, "phase": phase}
    exits: list[str] = []
    refusals: list[str] = []
    if phase == "frozen":
        exits.append("inside the weekend freeze, when the venue does not price US legs")
    elif phase == "preflatten":
        exits.append(
            f"pre-flatten: within {policy.WEEKEND_PREFLATTEN_MINUTES} minutes of the weekend freeze"
        )
    elif phase == "no_open_buffer":
        refusals.append(f"within {policy.WEEKEND_NO_OPEN_BUFFER_HOURS:g}h of the weekend freeze")
    return _conclude(
        G2,
        leg,
        inputs=values,
        exits=exits,
        refusals=refusals,
        ok="outside the weekend freeze and its buffers",
    )


@dataclass(frozen=True)
class GrossAllocation:
    cap: float
    requested: float
    held: float
    new: float
    ceilings: dict[str, float]

    @property
    def binds(self) -> bool:
        return bool(self.ceilings)

    def note(self) -> str:
        if not self.binds:
            return f"gross {self.requested:.4%} within the {self.cap:.0%} cap"
        if self.held >= self.cap:
            return (
                f"gross {self.requested:.4%} over the {self.cap:.0%} cap even without new exposure "
                f"(held {self.held:.4%}): held weights scaled by {self.cap / self.held:.4f}, "
                "nothing new admitted"
            )
        return (
            f"gross {self.requested:.4%} over the {self.cap:.0%} cap: held {self.held:.4%} kept, "
            f"new exposure {self.new:.4%} scaled by {(self.cap - self.held) / self.new:.4f}"
        )


def allocate_gross(candidates: Mapping[str, tuple[float, float]], cap: float) -> GrossAllocation:
    held = {s: min(c, h) for s, (c, h) in candidates.items()}
    new = {s: c - held[s] for s, (c, _) in candidates.items()}
    total_held = sum(held.values())
    total_new = sum(new.values())
    requested = total_held + total_new
    ceilings: dict[str, float] = {}
    if requested > cap + EPS:
        if total_held >= cap:
            factor = cap / total_held
            ceilings = {s: held[s] * factor for s in candidates}
        else:
            factor = (cap - total_held) / total_new
            ceilings = {s: held[s] + new[s] * factor for s in candidates}
    return GrossAllocation(
        cap=cap, requested=requested, held=total_held, new=total_new, ceilings=ceilings
    )


def g3_size(leg: Leg, *, gross_ceiling: float | None, gross_note: str = "") -> GuardRuling:
    per_name = policy.PER_NAME_MAX
    ceiling, which = per_name, f"the {per_name:.0%} per-name cap"
    if gross_ceiling is not None and gross_ceiling < per_name:
        ceiling, which = (
            max(0.0, gross_ceiling),
            f"its share of the {policy.GROSS_MAX:.0%} gross cap",
        )
    requested = abs(leg.reference)
    values: dict[str, object] = {
        "per_name_max": per_name,
        "gross_max": policy.GROSS_MAX,
        "gross_ceiling": gross_ceiling,
        "requested_abs_weight": requested,
    }
    note = f" ({gross_note})" if gross_note else ""
    if ceiling < requested - EPS:
        return GuardRuling(
            G3,
            leg.symbol,
            FIRED,
            f"{requested:.4%} requested; {which} allows {ceiling:.4%}{note}",
            ceiling=ceiling,
            inputs=values,
        )
    return GuardRuling(
        G3,
        leg.symbol,
        PASSED,
        f"{requested:.4%} is within {which} ({ceiling:.4%}){note}",
        ceiling=ceiling,
        inputs=values,
    )


def g4_stop(leg: Leg, *, inputs: Inputs) -> GuardRuling:
    values: dict[str, object] = {
        "stop_loss_pct": policy.STOP_LOSS_PCT,
        "trigger": policy.STOP_TRIGGER,
    }
    if not leg.adds_exposure:
        return GuardRuling(
            G4,
            leg.symbol,
            NOT_APPLICABLE,
            "no exposure-adding order, so no preset stop is required",
            inputs=values,
        )
    quote = inputs.quotes.get(leg.symbol)
    limits = inputs.limits.get(leg.symbol)
    missing: list[str] = []
    if quote is None or quote.mark is None or quote.mark <= 0:
        missing.append("the quote (expected entry and mark)")
    if limits is None:
        missing.append("the instrument limits")
    if missing or quote is None or quote.mark is None:
        return _conclude(G4, leg, inputs=values, missing=missing, ok="")
    side = "buy" if leg.reference > 0 else "sell"
    entry = entry_price(side, quote, quote.mark)
    price_step = None if limits is None else limits.price_step
    values.update(
        side=side,
        entry=str(entry),
        price_step="unknown: no venue grid" if price_step is None else str(price_step),
    )
    try:
        stop = stop_price(entry, side, price_step)
    except ValueError as exc:
        return _conclude(G4, leg, inputs=values, refusals=(f"no valid stop: {exc}",), ok="")
    distance = float(abs(entry - stop) / entry)
    values.update(stop_price=str(stop), stop_distance=distance, mark=str(quote.mark))
    losing_side = stop < quote.mark if side == "buy" else stop > quote.mark
    if not losing_side:
        return _conclude(
            G4,
            leg,
            inputs=values,
            refusals=(
                f"the stop {stop} ({distance:.2%} from the expected entry {entry}) is not on the "
                f"losing side of the mark {quote.mark}: it would trigger on the fill",
            ),
            ok="",
        )
    return _conclude(
        G4,
        leg,
        inputs=values,
        ok=f"preset stop at {stop}, {distance:.2%} from the expected entry {entry}",
    )


def g5_daily_kill(leg: Leg | None, *, book: BookView) -> GuardRuling:
    limit = policy.DAILY_KILL_PCT
    values: dict[str, object] = {
        "day_open_equity": str(book.day_open_equity),
        "equity": str(book.equity),
        "kill_at": -limit,
    }
    exits: list[str] = []
    missing: list[str] = []
    day_return = book.day_return
    if not book.history_known:
        missing.append("the 00:00 UTC equity (persisted history lost)")
    elif day_return is None:
        missing.append("a positive 00:00 UTC equity")
    else:
        values["day_return"] = day_return
        if day_return <= -limit:
            exits.append(
                f"the book is {day_return:.3%} since 00:00 UTC, at or beyond the -{limit:.1%} "
                "daily kill"
            )
    ok = f"day return within the -{limit:.1%} kill"
    if leg is None:
        return _conclude_book(G5, inputs=values, exits=exits, missing=missing, ok=ok)
    return _conclude(G5, leg, inputs=values, exits=exits, missing=missing, ok=ok)


def g6_turnover(
    leg: Leg, *, book: BookView, invalidation_declared: bool, at: datetime
) -> GuardRuling:
    limit = policy.MAX_REBALANCES_PER_NAME_PER_DAY
    hold = timedelta(hours=policy.MIN_HOLD_HOURS)
    rebalances = book.rebalances_today.get(leg.symbol, 0)
    values: dict[str, object] = {
        "rebalances_today": rebalances,
        "daily_limit": limit,
        "min_hold_hours": policy.MIN_HOLD_HOURS,
        "invalidation_declared": invalidation_declared,
    }
    refusals: list[str] = []
    missing: list[str] = []
    notes: list[str] = []
    if not book.history_known:
        missing.append("today's order count and the last increase time (persisted history lost)")
    if rebalances >= limit:
        refusals.append(f"{rebalances} model orders on {leg.symbol} today, limit {limit}")
    position = book.positions.get(leg.symbol)
    if position is not None and not position.is_flat and book.history_known:
        if position.last_increase_at is None:
            missing.append("the time of the last increase")
        else:
            since = as_utc(at) - as_utc(position.last_increase_at)
            values["hours_since_last_increase"] = since.total_seconds() / 3600
            if since < hold:
                if invalidation_declared:
                    notes.append("minimum hold lifted: the model declared its invalidation fired")
                else:
                    refusals.append(
                        f"last increase {since.total_seconds() / 3600:.2f}h ago, inside the "
                        f"{policy.MIN_HOLD_HOURS}h minimum hold, with no invalidation declared"
                    )
    ok = "; ".join(notes) or "within the daily order limit and the minimum hold"
    return _conclude(G6, leg, inputs=values, refusals=refusals, missing=missing, ok=ok)


def g7_fee_budget(leg: Leg | None, *, book: BookView) -> GuardRuling:
    values: dict[str, object] = {
        "daily_budget_bps": policy.FEE_BUDGET_DAILY_BPS,
        "window_budget_bps": policy.FEE_BUDGET_WINDOW_BPS,
        "fees_today": str(book.fees_today),
        "fees_total": str(book.fees_total),
    }
    refusals: list[str] = []
    missing: list[str] = []
    if not book.history_known:
        missing.append("the fees paid (persisted history lost)")
    elif book.day_open_equity <= 0 or book.starting_equity <= 0:
        missing.append("a positive day-open and starting equity")
    else:
        daily = float(book.fees_today / book.day_open_equity * 10_000)
        window = float(book.fees_total / book.starting_equity * 10_000)
        values.update(fees_today_bps=daily, fees_window_bps=window)
        if daily >= policy.FEE_BUDGET_DAILY_BPS:
            refusals.append(
                f"fees today {daily:.2f} bps reached the {policy.FEE_BUDGET_DAILY_BPS} bps budget"
            )
        if window >= policy.FEE_BUDGET_WINDOW_BPS:
            refusals.append(
                f"fees this window {window:.2f} bps reached the "
                f"{policy.FEE_BUDGET_WINDOW_BPS} bps budget"
            )
    ok = "fees within the daily and window budgets"
    if leg is None:
        return _conclude_book(G7, inputs=values, refusals=refusals, missing=missing, ok=ok)
    return _conclude(G7, leg, inputs=values, refusals=refusals, missing=missing, ok=ok)


def g8_taker_only(leg: Leg, *, inputs: Inputs) -> GuardRuling:
    limit = policy.MAX_OPEN_SPREAD_BPS
    values: dict[str, object] = {
        "order_type": "market",
        "taker_only": policy.TAKER_ONLY,
        "max_open_spread_bps": limit,
    }
    refusals: list[str] = []
    missing: list[str] = []
    quote = inputs.quotes.get(leg.symbol)
    if quote is None or quote.bid is None or quote.ask is None:
        missing.append("the bid and ask")
    elif quote.bid <= 0 or quote.ask <= 0 or quote.ask < quote.bid:
        values.update(bid=str(quote.bid), ask=str(quote.ask))
        refusals.append(f"the book is not two-sided (bid {quote.bid}, ask {quote.ask})")
    else:
        spread = quote.spread_bps
        values.update(bid=str(quote.bid), ask=str(quote.ask), spread_bps=_num(spread))
        if spread is not None and spread > limit:
            refusals.append(f"spread {spread:.2f} bps is wider than {limit} bps")
    return _conclude(
        G8,
        leg,
        inputs=values,
        refusals=refusals,
        missing=missing,
        ok=f"market order; spread within {limit} bps",
    )


def g9_grounding(leg: Leg, *, report: object | None) -> GuardRuling:
    """``report`` is a :class:`src.grounding.Report` (duck-typed so the kernel stays importable
    without it): ``figures`` and ``unresolved`` sequences of figures with a ``raw`` string."""
    values: dict[str, object] = {"tolerance": policy.GROUNDING_TOLERANCE}
    if not leg.adds_exposure:
        return GuardRuling(
            G9,
            leg.symbol,
            NOT_APPLICABLE,
            "adds no exposure, so grounding does not bind",
            inputs=values,
        )
    if report is None:
        return _conclude(
            G9, leg, inputs=values, missing=("a grounding report for this target",), ok=""
        )
    figures = list(getattr(report, "figures", ()))
    unresolved = [f for f in figures if not getattr(f, "resolved", False)]
    values.update(
        figures=len(figures),
        unresolved=len(unresolved),
        coverage=1.0 if not figures else 1 - len(unresolved) / len(figures),
    )
    refusals: list[str] = []
    if unresolved:
        shown = ", ".join(repr(getattr(f, "raw", "")) for f in unresolved[:5])
        refusals.append(
            f"{len(unresolved)} of {len(figures)} figures do not resolve to a snapshot fact "
            f"within {policy.GROUNDING_TOLERANCE:.0%}: {shown}"
        )
    return _conclude(
        G9,
        leg,
        inputs=values,
        refusals=refusals,
        ok=f"all {len(figures)} figures resolve to snapshot facts",
    )


def g10_breaker(
    leg: Leg | None,
    *,
    breaker: BreakerState,
    book: BookView,
    llm_outage: bool,
    inputs: Inputs,
) -> GuardRuling:
    demanded, trips = book_conditions(
        equity=float(book.equity),
        drawdown=book.drawdown,
        day_return=book.day_return,
        consecutive_losses=book.consecutive_losses,
        llm_outage=llm_outage,
        include_daily_kill=False,
    )
    effective = most_severe(breaker.activation, demanded)
    reasons = ", ".join(dict.fromkeys((*breaker.trips, *trips))) or "no trip recorded"
    values: dict[str, object] = {
        "breaker_state": breaker.activation,
        "breaker_trips": ",".join(breaker.trips),
        "book_demands": demanded,
        "book_trips": ",".join(trips),
        "effective": effective,
        "drawdown": book.drawdown,
        "consecutive_losses": book.consecutive_losses,
        "llm_outage": llm_outage,
    }
    exits: list[str] = []
    refusals: list[str] = []
    missing: list[str] = []
    if effective == HALTED:
        exits.append(f"breaker halted ({reasons})")
    elif effective == REDUCE_ONLY:
        refusals.append(f"breaker reduce-only ({reasons})")
    if not book.history_known:
        missing.append("the peak equity (persisted history lost)")
    ok = "breaker active; snapshot and quote fresh"
    if leg is None:
        return _conclude_book(
            G10, inputs=values, exits=exits, refusals=refusals, missing=missing, ok=ok
        )
    snapshot_limit = timedelta(minutes=policy.BREAKER_SNAPSHOT_MAX_AGE_MINUTES)
    quote_limit = timedelta(seconds=policy.BREAKER_QUOTE_MAX_AGE_SECONDS)
    at = as_utc(inputs.at)
    if inputs.snapshot_taken_at is None:
        missing.append("the perception snapshot")
    else:
        age = at - as_utc(inputs.snapshot_taken_at)
        values["snapshot_age_seconds"] = age.total_seconds()
        if abs(age) > snapshot_limit:
            refusals.append(
                f"snapshot is {age.total_seconds() / 60:.1f} min old, limit "
                f"{policy.BREAKER_SNAPSHOT_MAX_AGE_MINUTES} min"
            )
    quote = inputs.quotes.get(leg.symbol)
    if quote is None:
        missing.append("the quote")
    else:
        age = at - as_utc(quote.observed_at)
        values["quote_age_seconds"] = age.total_seconds()
        if abs(age) > quote_limit:
            refusals.append(
                f"quote is {age.total_seconds():.0f} s old, "
                f"limit {policy.BREAKER_QUOTE_MAX_AGE_SECONDS} s"
            )
    return _conclude(
        G10, leg, inputs=values, exits=exits, refusals=refusals, missing=missing, ok=ok
    )


def g11_eligibility(
    leg: Leg,
    *,
    book: BookView,
    inputs: Inputs,
    candidate_abs_weight: float | None = None,
) -> GuardRuling:
    symbol = leg.symbol
    if symbol in policy.EXCLUDED:
        return _conclude(
            G11,
            leg,
            inputs={"in_universe": False, "excluded": True},
            exits=(f"{symbol} is excluded for measured venue failures",),
            ok="",
        )
    if symbol not in _ASSET_CLASS:
        return _conclude(
            G11,
            leg,
            inputs={"in_universe": False, "excluded": False},
            exits=(f"{symbol} is not in the {len(policy.UNIVERSE)}-symbol universe",),
            ok="",
        )
    values: dict[str, object] = {"in_universe": True, "configured": symbol in inputs.configured}
    refusals: list[str] = []
    missing: list[str] = []
    if symbol not in inputs.configured:
        return _conclude(
            G11,
            leg,
            inputs=values,
            exits=(f"{symbol} is not among this subscription's configured symbols",),
            ok="",
        )
    candidate = abs(leg.reference) if candidate_abs_weight is None else candidate_abs_weight
    values["candidate_abs_weight"] = candidate
    limits = inputs.limits.get(symbol)
    if limits is None:
        missing.append("the instrument limits")
    else:
        values.update(
            min_order_qty=str(limits.min_qty),
            min_order_amount=str(limits.min_amount),
            qty_step=str(limits.qty_step),
        )
        if leg.adds_exposure and candidate > leg.hold_ceiling + EPS:
            price = planning_price(symbol, inputs)
            if price is None or book.equity <= 0:
                missing.append("a mark and positive equity to size the order")
            else:
                target = target_quantity(candidate, book.equity, price, limits)
                position = book.positions.get(symbol)
                held = Decimal(0) if position is None else position.qty
                same_side = held != 0 and (held > 0) == (leg.reference > 0)
                order = round_qty(target - abs(held), limits) if same_side else target
                values.update(order_qty=str(order), order_notional=str(order * price))
                if order <= 0:
                    refusals.append(f"the increase rounds to nothing on the {limits.qty_step} grid")
                elif order < limits.min_qty:
                    refusals.append(f"order {order} is below the minimum quantity {limits.min_qty}")
                elif order * price < limits.min_amount:
                    refusals.append(
                        f"order notional {order * price} is below the minimum {limits.min_amount}"
                    )
    return _conclude(
        G11,
        leg,
        inputs=values,
        refusals=refusals,
        missing=missing,
        ok="universe symbol, configured, order clears the venue minimums",
    )


# ------------------------------------------------------------------------------------------------
# The kernel
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Scene:
    book: BookView
    inputs: Inputs
    grounding: Mapping[str, object]
    invalidation_fired: Mapping[str, bool]
    breaker: BreakerState
    applied: frozenset[str]
    llm_outage: bool


def rule(
    *,
    proposed: Mapping[str, float],
    decision_ref: str,
    book: BookView,
    inputs: Inputs,
    breaker: BreakerState,
    grounding: Mapping[str, object],
    invalidation_fired: Mapping[str, bool],
    guards: frozenset[str] = ALL_GUARDS,
) -> KernelRuling:
    """Rule on a model decision's proposed weights. Every proposed and every held symbol is ruled;
    a held symbol the proposal does not name is ruled as a hold."""
    if not decision_ref:
        raise KernelError("a proposal can only be ruled under a model decision")
    bad = sorted(s for s, w in proposed.items() if not math.isfinite(w))
    if bad:
        raise KernelError(f"non-finite proposed weight for {', '.join(bad)}")
    scene = _Scene(book, inputs, grounding, invalidation_fired, breaker, guards, False)
    symbols = sorted(set(proposed) | set(book.held()))
    legs = [
        Leg(symbol=s, current=current_weight(s, book, inputs), proposed=proposed.get(s))
        for s in symbols
    ]
    book_rulings, instruments = _evaluate(legs, scene)
    return _seal(
        scene,
        decision_ref=decision_ref,
        protective_reason=None,
        book_rulings=book_rulings,
        instruments=instruments,
    )


def protective(
    *, book: BookView, inputs: Inputs, breaker: BreakerState, llm_outage: bool = False
) -> KernelRuling | None:
    """The between-decisions check: daily kill, weekend pre-flatten and freeze, venue integrity
    exits, breaker halts and model outage. Held positions only, as holds; ``None`` when nothing
    needs to change."""
    held = book.held()
    if not held:
        return None
    scene = _Scene(book, inputs, {}, {}, breaker, PROTECTIVE_GUARDS, llm_outage)
    legs = [Leg(symbol=s, current=current_weight(s, book, inputs)) for s in held]
    book_rulings, instruments = _evaluate(legs, scene)
    acting = [i for i in instruments if i.changed_by_kernel]
    if not acting:
        return None
    return _seal(
        scene,
        decision_ref=None,
        protective_reason=_protective_reason(acting, llm_outage=llm_outage),
        book_rulings=book_rulings,
        instruments=instruments,
    )


def _evaluate(
    legs: Sequence[Leg], scene: _Scene
) -> tuple[tuple[GuardRuling, ...], tuple[InstrumentRuling, ...]]:
    applied = scene.applied
    first = {leg.symbol: _first_pass(leg, scene) for leg in legs}
    candidates: dict[str, float] = {}
    for leg in legs:
        ceilings = [c for g in first[leg.symbol].values() if (c := g.effective_ceiling) is not None]
        if G3 in applied:
            ceilings.append(policy.PER_NAME_MAX)
        candidates[leg.symbol] = min([abs(leg.reference), *ceilings])
    allocation: GrossAllocation | None = None
    if G3 in applied:
        allocation = allocate_gross(
            {leg.symbol: (candidates[leg.symbol], leg.hold_ceiling) for leg in legs},
            policy.GROSS_MAX,
        )
    instruments: list[InstrumentRuling] = []
    for leg in legs:
        rulings = dict(first[leg.symbol])
        candidate = candidates[leg.symbol]
        if allocation is not None:
            gross_ceiling = allocation.ceilings.get(leg.symbol)
            rulings[G3] = g3_size(
                leg,
                gross_ceiling=gross_ceiling,
                gross_note=allocation.note() if allocation.binds else "",
            )
            if gross_ceiling is not None:
                candidate = min(candidate, gross_ceiling)
        if G11 in applied:
            rulings[G11] = g11_eligibility(
                leg, book=scene.book, inputs=scene.inputs, candidate_abs_weight=candidate
            )
        instruments.append(_bind(leg, [rulings[g] for g in GUARD_ORDER if g in rulings]))
    return _book_rulings(scene, allocation), tuple(instruments)


def _first_pass(leg: Leg, scene: _Scene) -> dict[str, GuardRuling]:
    applied = scene.applied
    book, inputs = scene.book, scene.inputs
    out: dict[str, GuardRuling] = {}
    if G1 in applied:
        out[G1] = g1_venue_integrity(leg, inputs=inputs)
    if G2 in applied:
        out[G2] = g2_weekend_freeze(leg, at=inputs.at)
    if G4 in applied:
        out[G4] = g4_stop(leg, inputs=inputs)
    if G5 in applied:
        out[G5] = g5_daily_kill(leg, book=book)
    if G6 in applied:
        out[G6] = g6_turnover(
            leg,
            book=book,
            invalidation_declared=scene.invalidation_fired.get(leg.symbol, False),
            at=inputs.at,
        )
    if G7 in applied:
        out[G7] = g7_fee_budget(leg, book=book)
    if G8 in applied:
        out[G8] = g8_taker_only(leg, inputs=inputs)
    if G9 in applied:
        out[G9] = g9_grounding(leg, report=scene.grounding.get(leg.symbol))
    if G10 in applied:
        out[G10] = g10_breaker(
            leg, breaker=scene.breaker, book=book, llm_outage=scene.llm_outage, inputs=inputs
        )
    if G11 in applied:
        out[G11] = g11_eligibility(leg, book=book, inputs=inputs, candidate_abs_weight=0.0)
    return out


def _book_rulings(scene: _Scene, allocation: GrossAllocation | None) -> tuple[GuardRuling, ...]:
    book, applied = scene.book, scene.applied
    out: list[GuardRuling] = []
    if allocation is not None:
        out.append(
            GuardRuling(
                G3,
                None,
                FIRED if allocation.binds else PASSED,
                allocation.note(),
                ceiling=policy.GROSS_MAX,
                inputs={
                    "gross_max": allocation.cap,
                    "gross_requested": allocation.requested,
                    "gross_held": allocation.held,
                    "gross_new": allocation.new,
                },
            )
        )
    if G5 in applied:
        out.append(g5_daily_kill(None, book=book))
    if G7 in applied:
        out.append(g7_fee_budget(None, book=book))
    if G10 in applied:
        out.append(
            g10_breaker(
                None,
                breaker=scene.breaker,
                book=book,
                llm_outage=scene.llm_outage,
                inputs=scene.inputs,
            )
        )
    return tuple(out)


def _seal(
    scene: _Scene,
    *,
    decision_ref: str | None,
    protective_reason: str | None,
    book_rulings: tuple[GuardRuling, ...],
    instruments: tuple[InstrumentRuling, ...],
) -> KernelRuling:
    after = scene.breaker.activation
    if G10 in scene.applied:
        demanded, _ = book_conditions(
            equity=float(scene.book.equity),
            drawdown=scene.book.drawdown,
            day_return=scene.book.day_return,
            consecutive_losses=scene.book.consecutive_losses,
            llm_outage=scene.llm_outage,
            include_daily_kill=False,
        )
        after = most_severe(after, demanded)
    if any(g.guard == G5 and g.forces_exit for g in book_rulings):
        after = HALTED
    return KernelRuling(
        at=scene.inputs.at,
        decision_ref=decision_ref,
        protective_reason=protective_reason,
        activation_before=scene.breaker.activation,
        activation_after=after,
        book_rulings=book_rulings,
        instruments=instruments,
        guards_applied=tuple(g for g in GUARD_ORDER if g in scene.applied),
    )


def _bind(leg: Leg, rulings: Sequence[GuardRuling]) -> InstrumentRuling:
    reference = leg.reference
    requested = abs(reference)
    below: list[tuple[tuple[float, bool, int], float, GuardRuling]] = []
    for g in rulings:
        ceiling = g.effective_ceiling
        if ceiling is not None and ceiling < requested - EPS:
            below.append(((ceiling, not g.forces_exit, GUARD_ORDER.index(g.guard)), ceiling, g))
    below.sort(key=lambda item: item[0])
    approved = reference
    binding: str | None = None
    final = tuple(rulings)
    if below:
        _, ceiling, winner = below[0]
        approved = math.copysign(ceiling, reference) if ceiling > 0 else 0.0
        binding = winner.guard
        if len(below) > 1:
            trail = "; ".join(
                f"{g.guard} {'exit' if g.forces_exit else f'at {c:.4%}'}" for _, c, g in below[1:]
            )
            winner = GuardRuling(
                winner.guard,
                winner.symbol,
                winner.status,
                f"{winner.reason} | also below: {trail}",
                ceiling=winner.ceiling,
                forces_exit=winner.forces_exit,
                inputs=winner.inputs,
            )
        final = tuple(winner if g.guard == winner.guard else g for g in rulings)
    return InstrumentRuling(
        symbol=leg.symbol,
        current=leg.current,
        proposed=leg.proposed,
        approved=approved,
        binding_guard=binding,
        rulings=final,
    )


def _protective_reason(acting: Sequence[InstrumentRuling], *, llm_outage: bool) -> str:
    causes: set[str] = set()
    for ir in acting:
        for g in ir.rulings:
            if not (g.forces_exit and g.status == FIRED):
                continue
            if g.guard == G5:
                causes.add("daily_kill")
            elif g.guard == G10:
                causes.add("llm_outage" if llm_outage else "breaker")
            elif g.guard == G1:
                causes.add("venue_integrity")
            elif g.guard == G2:
                causes.add("weekend_freeze")
    for reason in PROTECTIVE_PRIORITY:
        if reason in causes:
            return reason
    raise KernelError("a protective ruling changed an instrument without a forced exit")


def activation_allows_increase(activation: str) -> bool:
    return activation == ACTIVE
