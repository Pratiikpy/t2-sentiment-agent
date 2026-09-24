"""The eleven guards: one pure function each, each returning a :class:`GuardRuling`.

A guard never fetches, never reads the clock and never raises for bad market data: it is handed
the facts and says what they permit. :mod:`sentiment_agent.kernel.kernel` evaluates every applicable
guard on every ruling and binds the minimum (DESIGN.md §10.2).

**What a guard can say about one instrument** (the ``Leg``: current weight, proposed weight):

* **Exit** (``forces_exit``, ceiling 0): the instrument must be flat. Used only for measured venue
  failures (G1), the weekend freeze (G2), the daily kill (G5), a halted breaker (G10) and symbols
  outside the universe (G11).
* **Refuse the increase** (ceiling = the *hold ceiling*): the approved weight may keep what is held
  on the same side but may not add to it or open the other side. A reduction or a close is always
  within this ceiling, which is how "reductions are never blocked" (DESIGN.md §10.4) holds by
  construction for G6, G7 and every other refusal.
* **Cap** (a plain ceiling on ``|weight|``): only G3's per-name and gross limits.
* **Not evaluated**: an input was missing. Treated exactly like a refusal of the increase
  (fail-closed), and reported as such.
* **Not applicable**: the rule has nothing to say (the weekend on a crypto leg, a stop on a leg
  that adds nothing).

**Status is about effect**, as ``GuardStatus`` defines it: FIRED means the guard's ceiling is below
the request or it forced an exit of something held. A refusal that had nothing to refuse (the model
was reducing anyway) is PASSED, with the would-be ceiling and the reason recorded, so a reader can
still see the condition. NOT_EVALUATED is reported whenever an input was missing, whatever the
request, because "could not check" must never read as "checked and passed".

Book-level guards (G5, G7, G10) are called with ``leg=None`` for the book's own ruling
(``symbol=None``) and with a leg for their effect on one instrument.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Final, Literal

from sentiment_agent.kernel.breaker import book_conditions, most_severe
from sentiment_agent.kernel.planner import entry_price, round_qty, stop_price, target_quantity
from sentiment_agent.types import (
    Activation,
    AssetClass,
    BookState,
    BreakerState,
    GroundingReport,
    GuardId,
    GuardRuling,
    GuardStatus,
    InstrumentSpec,
    Policy,
    Position,
    Quote,
    Side,
    UniverseEntry,
    WeekendRule,
)

EPS: Final = 1e-12
"""Weight tolerance, the same one ``InstrumentRuling`` uses for its only-reduce check."""

InputValue = float | str | bool | None
WeekendPhase = Literal["open", "no_open_buffer", "preflatten", "frozen"]


@dataclass(frozen=True, slots=True)
class Leg:
    """One instrument as the kernel sees it: what is held and what was asked for."""

    symbol: str
    current: float
    """Signed weight held now."""
    proposed: float | None = None
    """Signed weight the model asked for; ``None`` for a hold (protective rulings)."""

    @property
    def reference(self) -> float:
        """What the only-reduce invariant is measured against."""
        return self.current if self.proposed is None else self.proposed

    @property
    def hold_ceiling(self) -> float:
        """The largest ``|weight|`` in the reference's direction that adds nothing: what is already
        held on that side, or zero when nothing is (an open, or the far side of a flip)."""
        return abs(self.current) if self.current * self.reference > 0 else 0.0

    @property
    def adds_exposure(self) -> bool:
        return self.reference != 0 and abs(self.reference) > self.hold_ceiling + EPS

    @property
    def inert(self) -> bool:
        return self.current == 0 and self.reference == 0


# ------------------------------------------------------------------------------------------------
# Shared construction
# ------------------------------------------------------------------------------------------------

_BASIS_CACHE: dict[int, tuple[Policy, dict[GuardId, str]]] = {}


def _basis(policy: Policy, guard: GuardId) -> str:
    cached = _BASIS_CACHE.get(id(policy))
    if cached is None or cached[0] is not policy:
        cached = (policy, {b.guard: b.basis for b in policy.guard_bases})
        _BASIS_CACHE[id(policy)] = cached
    return cached[1][guard]


def _num(value: float | Decimal | None) -> InputValue:
    """A ruling input: a finite number as a float, a non-finite one spelled out (hashing refuses
    NaN and infinity)."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else str(number)


def _pct(value: float) -> str:
    return f"{value:.2%}" if math.isfinite(value) else str(value)


def _ruling(
    guard: GuardId,
    symbol: str | None,
    status: GuardStatus,
    *,
    policy: Policy,
    reason: str,
    inputs: Mapping[str, InputValue],
    ceiling: float | None = None,
    forces_exit: bool = False,
) -> GuardRuling:
    return GuardRuling(
        guard=guard,
        symbol=symbol,
        status=status,
        ceiling_abs_weight=ceiling,
        forces_exit=forces_exit,
        reason=reason,
        basis=_basis(policy, guard),
        inputs=dict(inputs),
    )


def _not_applicable(
    guard: GuardId, leg: Leg, policy: Policy, reason: str, inputs: Mapping[str, InputValue]
) -> GuardRuling:
    return _ruling(
        guard,
        leg.symbol,
        GuardStatus.NOT_APPLICABLE,
        policy=policy,
        reason=reason,
        inputs=inputs,
    )


def _conclude(
    guard: GuardId,
    leg: Leg,
    policy: Policy,
    *,
    inputs: Mapping[str, InputValue],
    ok: str,
    exits: Sequence[str] = (),
    refusals: Sequence[str] = (),
    missing: Sequence[str] = (),
) -> GuardRuling:
    """The per-instrument verdict from a guard's findings, with precedence exit > refusal of an
    actual increase > missing input > refusal with nothing to refuse > pass."""
    if exits:
        text = "; ".join(exits)
        if leg.inert:
            return _ruling(
                guard,
                leg.symbol,
                GuardStatus.PASSED,
                policy=policy,
                reason=f"{text}; nothing held or proposed, so nothing to exit",
                inputs=inputs,
                ceiling=0.0,
            )
        return _ruling(
            guard,
            leg.symbol,
            GuardStatus.FIRED,
            policy=policy,
            reason=f"exit: {text}",
            inputs=inputs,
            ceiling=0.0,
            forces_exit=True,
        )
    missing_note = f"; not evaluated: missing {', '.join(missing)}" if missing else ""
    if refusals and leg.adds_exposure:
        return _ruling(
            guard,
            leg.symbol,
            GuardStatus.FIRED,
            policy=policy,
            reason=f"no increase: {'; '.join(refusals)}{missing_note}",
            inputs=inputs,
            ceiling=leg.hold_ceiling,
        )
    if missing:
        refusal_note = f"; also {'; '.join(refusals)}" if refusals else ""
        return _ruling(
            guard,
            leg.symbol,
            GuardStatus.NOT_EVALUATED,
            policy=policy,
            reason=f"not evaluated, so no increase is permitted: missing "
            f"{', '.join(missing)}{refusal_note}",
            inputs=inputs,
            ceiling=leg.hold_ceiling,
        )
    if refusals:
        return _ruling(
            guard,
            leg.symbol,
            GuardStatus.PASSED,
            policy=policy,
            reason=f"{'; '.join(refusals)}; this would refuse an increase, and none was requested",
            inputs=inputs,
            ceiling=leg.hold_ceiling,
        )
    return _ruling(guard, leg.symbol, GuardStatus.PASSED, policy=policy, reason=ok, inputs=inputs)


def _conclude_book(
    guard: GuardId,
    policy: Policy,
    *,
    inputs: Mapping[str, InputValue],
    ok: str,
    exits: Sequence[str] = (),
    refusals: Sequence[str] = (),
    missing: Sequence[str] = (),
) -> GuardRuling:
    """The book-level verdict (``symbol=None``): FIRED when the condition is on, since it applies to
    every instrument, including ones not yet asked about."""
    if exits:
        return _ruling(
            guard,
            None,
            GuardStatus.FIRED,
            policy=policy,
            reason=f"exit everything: {'; '.join(exits)}",
            inputs=inputs,
            ceiling=0.0,
            forces_exit=True,
        )
    if refusals:
        note = f"; not evaluated: missing {', '.join(missing)}" if missing else ""
        return _ruling(
            guard,
            None,
            GuardStatus.FIRED,
            policy=policy,
            reason=f"no new exposure anywhere: {'; '.join(refusals)}{note}",
            inputs=inputs,
        )
    if missing:
        return _ruling(
            guard,
            None,
            GuardStatus.NOT_EVALUATED,
            policy=policy,
            reason=f"not evaluated, so no increase is permitted: missing {', '.join(missing)}",
            inputs=inputs,
        )
    return _ruling(guard, None, GuardStatus.PASSED, policy=policy, reason=ok, inputs=inputs)


# ------------------------------------------------------------------------------------------------
# Calendar
# ------------------------------------------------------------------------------------------------


def weekend_phase(at: datetime, rule: WeekendRule) -> WeekendPhase:
    """Where ``at`` falls against the weekly freeze (policy v1: Friday 20:00 -> Monday 00:00 UTC).

    ``frozen`` inside the freeze; ``preflatten`` in the ``preflatten_minutes`` before it (19:45 on);
    ``no_open_buffer`` in the ``no_open_buffer_hours`` before that (18:00 on); ``open`` otherwise.
    Each window includes its start and excludes its end.
    """
    week_start = (at - timedelta(days=at.weekday())).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    freeze_offset = timedelta(days=rule.freeze_weekday, hours=rule.freeze_hour)
    length_hours = (
        rule.reopen_weekday * 24 + rule.reopen_hour - rule.freeze_weekday * 24 - rule.freeze_hour
    ) % (7 * 24)
    length = timedelta(hours=length_hours)
    starts = [week_start + freeze_offset + timedelta(weeks=k) for k in (-1, 0, 1)]
    if length > timedelta(0) and any(s <= at < s + length for s in starts):
        return "frozen"
    until = min(s for s in starts if s > at) - at
    if until <= timedelta(minutes=rule.preflatten_minutes):
        return "preflatten"
    if until <= timedelta(hours=rule.no_open_buffer_hours):
        return "no_open_buffer"
    return "open"


def session_open(asset_class: AssetClass | None, at: datetime, rule: WeekendRule) -> bool:
    """Whether the instrument's Demo index should be moving: always for crypto, outside the freeze
    for US legs. An unknown asset class is treated as always open (the strictest reading)."""
    if asset_class is None or not asset_class.follows_us_session:
        return True
    return weekend_phase(at, rule) != "frozen"


# ------------------------------------------------------------------------------------------------
# G1-G11
# ------------------------------------------------------------------------------------------------


def g1_venue_integrity(
    leg: Leg,
    *,
    entry: UniverseEntry | None,
    demo: Quote | None,
    live: Quote | None,
    index_move_bps_3h: float | None,
    at: datetime,
    policy: Policy,
) -> GuardRuling:
    """Exit when the Demo mark departs more than 3% from the Demo index, or the Demo-live gap
    (last prices, the measured basis) exceeds the instrument's p99; refuse increases when the Demo
    index has stopped moving while its session should be open."""
    exits: list[str] = []
    refusals: list[str] = []
    missing: list[str] = []
    limit = policy.mark_index_max_gap
    stale_floor = policy.stale_index_min_move_bps_3h
    inputs: dict[str, InputValue] = {
        "mark_index_limit": limit,
        "stale_index_min_move_bps_3h": stale_floor,
    }
    if demo is None:
        missing.append("the Demo quote")
    else:
        gap = demo.mark_index_gap
        inputs.update(
            demo_mark=str(demo.mark), demo_index=str(demo.index), mark_index_gap=_num(gap)
        )
        if gap > limit:
            exits.append(
                f"Demo mark {demo.mark} is {_pct(gap)} from its index {demo.index}, "
                f"beyond the {_pct(limit)} limit"
            )
    if entry is None:
        missing.append("a measured Demo-live p99 (symbol outside the universe)")
    else:
        inputs["demo_live_p99_bps"] = entry.demo_live_gap_p99_bps
        if live is None:
            missing.append("the live quote")
        elif demo is not None:
            if live.last <= 0 or demo.last <= 0:
                missing.append("a positive last price on both Demo and live")
            else:
                gap_bps = float(abs(demo.last / live.last - 1) * 10_000)
                inputs.update(
                    demo_last=str(demo.last), live_last=str(live.last), demo_live_gap_bps=gap_bps
                )
                if gap_bps > entry.demo_live_gap_p99_bps:
                    exits.append(
                        f"Demo last {demo.last} is {gap_bps:.1f} bps from live {live.last}, beyond "
                        f"the measured p99 of {entry.demo_live_gap_p99_bps} bps"
                    )
    is_open = session_open(entry.asset_class if entry else None, at, policy.weekend)
    inputs["session_open"] = is_open
    if is_open:
        if index_move_bps_3h is None or not math.isfinite(index_move_bps_3h):
            missing.append("the Demo index's 3h move")
        else:
            inputs["demo_index_move_bps_3h"] = index_move_bps_3h
            if index_move_bps_3h < stale_floor:
                refusals.append(
                    f"the Demo index moved {index_move_bps_3h:.2f} bps/h over 3h, below "
                    f"{stale_floor} bps/h while its session should be open (stale venue)"
                )
    return _conclude(
        GuardId.G1_VENUE_INTEGRITY,
        leg,
        policy,
        inputs=inputs,
        exits=exits,
        refusals=refusals,
        missing=missing,
        ok="Demo mark within the index limit, Demo-live gap within p99, Demo index moving",
    )


def g2_weekend_freeze(
    leg: Leg, *, asset_class: AssetClass | None, at: datetime, policy: Policy
) -> GuardRuling:
    """US-equity and US-index legs: flat from Friday 20:00 to Monday 00:00 UTC, pre-flattened from
    19:45, and no new exposure from 18:00. Crypto is not affected."""
    rule = policy.weekend
    if asset_class is None:
        return _conclude(
            GuardId.G2_WEEKEND_FREEZE,
            leg,
            policy,
            inputs={},
            missing=("the asset class (symbol outside the universe)",),
            ok="",
        )
    if not asset_class.follows_us_session:
        return _not_applicable(
            GuardId.G2_WEEKEND_FREEZE,
            leg,
            policy,
            f"{asset_class.value} trades through the weekend on Demo",
            {"asset_class": asset_class.value},
        )
    phase = weekend_phase(at, rule)
    inputs: dict[str, InputValue] = {"asset_class": asset_class.value, "phase": phase}
    exits: list[str] = []
    refusals: list[str] = []
    if phase == "frozen":
        exits.append("inside the weekend freeze, when Demo does not price US legs")
    elif phase == "preflatten":
        exits.append(f"pre-flatten: within {rule.preflatten_minutes} minutes of the weekend freeze")
    elif phase == "no_open_buffer":
        refusals.append(f"within {rule.no_open_buffer_hours:g}h of the weekend freeze")
    return _conclude(
        GuardId.G2_WEEKEND_FREEZE,
        leg,
        policy,
        inputs=inputs,
        exits=exits,
        refusals=refusals,
        ok="outside the weekend freeze and its buffers",
    )


@dataclass(frozen=True, slots=True)
class GrossAllocation:
    """How G3's gross cap was shared out. ``ceilings`` is empty when the cap does not bind."""

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
    """Share the gross cap across instruments.

    ``candidates`` maps each symbol to (its ``|weight|`` after every other guard, its hold ceiling).
    What is already held is kept first and new exposure is scaled pro rata into whatever headroom
    remains, so the cap never churns existing positions to make room for new ones. If the held part
    alone exceeds the cap (price drift), the held parts are scaled down pro rata and nothing new is
    admitted."""
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


def g3_size(
    leg: Leg,
    *,
    gross_ceiling: float | None,
    policy: Policy,
    gross_note: str = "",
) -> GuardRuling:
    """At most ``per_name_max`` per name, and this name's share of the ``gross_max`` headroom
    (``gross_ceiling``, from :func:`allocate_gross`; ``None`` when the gross cap does not bind)."""
    per_name = policy.per_name_max
    ceiling, which = per_name, f"the {per_name:.0%} per-name cap"
    if gross_ceiling is not None and gross_ceiling < per_name:
        ceiling, which = (
            max(0.0, gross_ceiling),
            f"its share of the {policy.gross_max:.0%} gross cap",
        )
    requested = abs(leg.reference)
    inputs: dict[str, InputValue] = {
        "per_name_max": per_name,
        "gross_max": policy.gross_max,
        "gross_ceiling": gross_ceiling,
        "requested_abs_weight": requested,
    }
    note = f" ({gross_note})" if gross_note else ""
    if ceiling < requested - EPS:
        return _ruling(
            GuardId.G3_SIZE,
            leg.symbol,
            GuardStatus.FIRED,
            policy=policy,
            reason=f"{requested:.4%} requested; {which} allows {ceiling:.4%}{note}",
            inputs=inputs,
            ceiling=ceiling,
        )
    return _ruling(
        GuardId.G3_SIZE,
        leg.symbol,
        GuardStatus.PASSED,
        policy=policy,
        reason=f"{requested:.4%} is within {which} ({ceiling:.4%}){note}",
        inputs=inputs,
        ceiling=ceiling,
    )


def g4_stop(
    leg: Leg, *, spec: InstrumentSpec | None, demo: Quote | None, policy: Policy
) -> GuardRuling:
    """An exposure-adding order must be able to carry its venue stop ``stop_loss_pct`` from the
    expected entry, on the price grid, triggered on mark. No placeable stop, no increase."""
    inputs: dict[str, InputValue] = {
        "stop_loss_pct": policy.stop_loss_pct,
        "trigger": policy.stop_trigger,
    }
    if not leg.adds_exposure:
        return _not_applicable(
            GuardId.G4_STOP,
            leg,
            policy,
            "no exposure-adding order, so no preset stop is required",
            inputs,
        )
    missing: list[str] = []
    if spec is None:
        missing.append("the instrument spec (price grid)")
    if demo is None:
        missing.append("the Demo quote (expected entry)")
    if spec is None or demo is None:
        return _conclude(GuardId.G4_STOP, leg, policy, inputs=inputs, missing=missing, ok="")
    side = Side.BUY if leg.reference > 0 else Side.SELL
    entry = entry_price(side, demo, demo.mark)
    inputs.update(side=side.value, entry=str(entry))
    try:
        stop = stop_price(entry, side, policy, spec)
    except ValueError as exc:
        return _conclude(
            GuardId.G4_STOP, leg, policy, inputs=inputs, refusals=(f"no valid stop: {exc}",), ok=""
        )
    distance = float(abs(entry - stop) / entry)
    inputs.update(stop_price=str(stop), stop_distance=distance, mark=str(demo.mark))
    losing_side = stop < demo.mark if side is Side.BUY else stop > demo.mark
    if not losing_side:
        return _conclude(
            GuardId.G4_STOP,
            leg,
            policy,
            inputs=inputs,
            refusals=(
                f"the stop {stop} ({distance:.2%} from the expected entry {entry}) is not on the "
                f"losing side of the Demo mark {demo.mark}: it would trigger on the fill",
            ),
            ok="",
        )
    return _conclude(
        GuardId.G4_STOP,
        leg,
        policy,
        inputs=inputs,
        ok=f"preset stop at {stop}, {distance:.2%} from the expected entry {entry}, "
        f"triggered on {policy.stop_trigger}",
    )


def g5_daily_kill(leg: Leg | None, *, book: BookState, policy: Policy) -> GuardRuling:
    """Equity down ``daily_kill_pct`` or more from the 00:00 UTC equity flattens the book. The
    breaker keeps it halted until the next UTC day (:mod:`sentiment_agent.kernel.breaker`)."""
    limit = policy.daily_kill_pct
    inputs: dict[str, InputValue] = {
        "day_open_equity": str(book.day_open_equity),
        "equity": str(book.equity),
        "kill_at": -limit,
    }
    exits: list[str] = []
    missing: list[str] = []
    if book.day_open_equity <= 0:
        missing.append("a positive 00:00 UTC equity")
    else:
        day_return = book.day_return
        inputs["day_return"] = day_return
        if day_return <= -limit:
            exits.append(
                f"the book is {day_return:.3%} since 00:00 UTC, at or beyond the -{limit:.1%} "
                "daily kill"
            )
    ok = f"day return within the -{limit:.1%} kill"
    if leg is None:
        return _conclude_book(
            GuardId.G5_DAILY_KILL, policy, inputs=inputs, exits=exits, missing=missing, ok=ok
        )
    return _conclude(
        GuardId.G5_DAILY_KILL, leg, policy, inputs=inputs, exits=exits, missing=missing, ok=ok
    )


def g6_turnover(
    leg: Leg,
    *,
    position: Position | None,
    rebalances_today: int,
    invalidation_declared: bool,
    at: datetime,
    policy: Policy,
) -> GuardRuling:
    """At most ``max_rebalances_per_name_per_day`` model orders per name per UTC day, and no
    increase or flip within ``min_hold_hours`` of the last increase unless the model declares the
    stated invalidation fired. Only increases are refused: a reduction is never blocked."""
    limit = policy.max_rebalances_per_name_per_day
    hold = timedelta(hours=policy.min_hold_hours)
    inputs: dict[str, InputValue] = {
        "rebalances_today": rebalances_today,
        "daily_limit": limit,
        "min_hold_hours": policy.min_hold_hours,
        "invalidation_declared": invalidation_declared,
    }
    refusals: list[str] = []
    notes: list[str] = []
    if rebalances_today >= limit:
        refusals.append(f"{rebalances_today} model orders on {leg.symbol} today, limit {limit}")
    if position is not None and not position.is_flat:
        since = at - position.last_increase_at
        inputs["hours_since_last_increase"] = since.total_seconds() / 3600
        if since < hold:
            if invalidation_declared:
                notes.append("minimum hold lifted: the model declared its invalidation fired")
            else:
                refusals.append(
                    f"last increase {since.total_seconds() / 3600:.2f}h ago, inside the "
                    f"{policy.min_hold_hours}h minimum hold, with no invalidation declared"
                )
    ok = "; ".join(notes) or "within the daily order limit and the minimum hold"
    return _conclude(GuardId.G6_TURNOVER, leg, policy, inputs=inputs, refusals=refusals, ok=ok)


def g7_fee_budget(leg: Leg | None, *, book: BookState, policy: Policy) -> GuardRuling:
    """No new exposure once fees today reach ``fee_budget_daily_bps`` of the 00:00 UTC equity, or
    fees since genesis reach ``fee_budget_window_bps`` of the starting equity. Increases only."""
    inputs: dict[str, InputValue] = {
        "daily_budget_bps": policy.fee_budget_daily_bps,
        "window_budget_bps": policy.fee_budget_window_bps,
        "fees_today": str(book.fees_today),
        "fees_total": str(book.fees_total),
    }
    refusals: list[str] = []
    missing: list[str] = []
    if book.day_open_equity <= 0 or book.starting_equity <= 0:
        missing.append("a positive day-open and starting equity")
    else:
        daily = float(book.fees_today / book.day_open_equity * 10_000)
        window = float(book.fees_total / book.starting_equity * 10_000)
        inputs.update(fees_today_bps=daily, fees_window_bps=window)
        if daily >= policy.fee_budget_daily_bps:
            refusals.append(
                f"fees today {daily:.2f} bps reached the {policy.fee_budget_daily_bps} bps budget"
            )
        if window >= policy.fee_budget_window_bps:
            refusals.append(
                f"fees this window {window:.2f} bps reached the "
                f"{policy.fee_budget_window_bps} bps budget"
            )
    ok = "fees within the daily and window budgets"
    if leg is None:
        return _conclude_book(
            GuardId.G7_FEE_BUDGET, policy, inputs=inputs, refusals=refusals, missing=missing, ok=ok
        )
    return _conclude(
        GuardId.G7_FEE_BUDGET, leg, policy, inputs=inputs, refusals=refusals, missing=missing, ok=ok
    )


def g8_taker_only(leg: Leg, *, demo: Quote | None, policy: Policy) -> GuardRuling:
    """Market (taker) orders only, which the order type enforces structurally; and no opening or
    increase while the Demo spread is wider than ``max_open_spread_bps`` or the book is
    one-sided."""
    limit = policy.max_open_spread_bps
    inputs: dict[str, InputValue] = {
        "order_type": "market",
        "taker_only": policy.taker_only,
        "max_open_spread_bps": limit,
    }
    refusals: list[str] = []
    missing: list[str] = []
    if demo is None:
        missing.append("the Demo quote")
    elif demo.bid <= 0 or demo.ask <= 0 or demo.ask < demo.bid:
        inputs.update(bid=str(demo.bid), ask=str(demo.ask))
        refusals.append(f"the Demo book is not two-sided (bid {demo.bid}, ask {demo.ask})")
    else:
        spread = demo.spread_bps
        inputs.update(bid=str(demo.bid), ask=str(demo.ask), spread_bps=_num(spread))
        if spread > limit:
            refusals.append(f"Demo spread {spread:.2f} bps is wider than {limit} bps")
    return _conclude(
        GuardId.G8_TAKER_ONLY,
        leg,
        policy,
        inputs=inputs,
        refusals=refusals,
        missing=missing,
        ok=f"market order; Demo spread within {limit} bps",
    )


def g9_grounding(leg: Leg, *, report: GroundingReport | None, policy: Policy) -> GuardRuling:
    """A target whose thesis, invalidation or view states a figure that does not resolve to a
    snapshot fact (within ``grounding_tolerance``, checked upstream) may not add exposure."""
    inputs: dict[str, InputValue] = {"tolerance": policy.grounding_tolerance}
    if not leg.adds_exposure:
        return _not_applicable(
            GuardId.G9_GROUNDING,
            leg,
            policy,
            "adds no exposure, so grounding does not bind",
            inputs,
        )
    if report is None:
        return _conclude(
            GuardId.G9_GROUNDING,
            leg,
            policy,
            inputs=inputs,
            missing=("a grounding report for this target",),
            ok="",
        )
    unresolved = report.unresolved
    inputs.update(figures=len(report.figures), unresolved=len(unresolved), coverage=report.coverage)
    refusals: list[str] = []
    if unresolved:
        shown = ", ".join(repr(f.raw) for f in unresolved[:5])
        refusals.append(
            f"{len(unresolved)} of {len(report.figures)} figures do not resolve to a snapshot fact "
            f"within {policy.grounding_tolerance:.0%}: {shown}"
        )
    return _conclude(
        GuardId.G9_GROUNDING,
        leg,
        policy,
        inputs=inputs,
        refusals=refusals,
        ok=f"all {len(report.figures)} figures resolve to snapshot facts",
    )


def g10_breaker(
    leg: Leg | None,
    *,
    breaker: BreakerState,
    book: BookState,
    llm_outage: bool,
    demo: Quote | None,
    snapshot_taken_at: datetime | None,
    at: datetime,
    policy: Policy,
    venue_unreconciled: Sequence[str] = (),
) -> GuardRuling:
    """The circuit breaker, and the freshness of what an increase would rest on.

    The effective activation is the more severe of the breaker's state and what the book demands
    right now (defence in depth: an un-assessed breaker still cannot let a 4% drawdown trade).
    HALTED or a model outage flattens the book; REDUCE_ONLY refuses increases. Per instrument, a
    perception snapshot older than ``snapshot_max_age_minutes`` or a Demo quote older than
    ``quote_max_age_seconds`` cannot open anything.

    ``venue_unreconciled`` (``KernelInputs.venue_unreconciled``) refuses every increase, book-wide:
    when the latest reconciliation could not read or find a fill, or the venue holds a different
    position, the book this ruling sizes from may not be the venue's, and a weight computed on it
    could breach the per-name cap on the venue. Reductions stay allowed; it lifts at the first
    sweep without such a discrepancy."""
    rule = policy.breaker
    demanded, trips = book_conditions(
        book, llm_outage=llm_outage, policy=policy, include_daily_kill=False
    )
    effective = most_severe(breaker.activation, demanded)
    reasons = ", ".join(dict.fromkeys((*breaker.trips, *trips))) or "no trip recorded"
    inputs: dict[str, InputValue] = {
        "breaker_state": breaker.activation.value,
        "breaker_trips": ",".join(breaker.trips),
        "book_demands": demanded.value,
        "book_trips": ",".join(trips),
        "effective": effective.value,
        "drawdown": book.drawdown,
        "consecutive_losses": book.consecutive_losses,
        "llm_outage": llm_outage,
    }
    exits: list[str] = []
    refusals: list[str] = []
    missing: list[str] = []
    if effective is Activation.HALTED:
        exits.append(f"breaker halted ({reasons})")
    elif effective is Activation.REDUCE_ONLY:
        refusals.append(f"breaker reduce-only ({reasons})")
    if venue_unreconciled:
        inputs["venue_unreconciled"] = float(len(venue_unreconciled))
        refusals.append(
            "the book is not reconciled to the venue ("
            + "; ".join(venue_unreconciled[:3])
            + (f"; +{len(venue_unreconciled) - 3} more" if len(venue_unreconciled) > 3 else "")
            + ")"
        )
    ok = "breaker active; snapshot and quote fresh"
    if leg is None:
        return _conclude_book(
            GuardId.G10_BREAKER, policy, inputs=inputs, exits=exits, refusals=refusals, ok=ok
        )
    snapshot_limit = timedelta(minutes=rule.snapshot_max_age_minutes)
    quote_limit = timedelta(seconds=rule.quote_max_age_seconds)
    if snapshot_taken_at is None:
        missing.append("the perception snapshot")
    else:
        age = at - snapshot_taken_at
        inputs["snapshot_age_seconds"] = age.total_seconds()
        if abs(age) > snapshot_limit:
            refusals.append(
                f"snapshot is {age.total_seconds() / 60:.1f} min old, limit "
                f"{rule.snapshot_max_age_minutes} min"
            )
    if demo is None:
        missing.append("the Demo quote")
    else:
        age = at - min(demo.ts, demo.fetched_at)
        inputs["quote_age_seconds"] = age.total_seconds()
        if abs(age) > quote_limit:
            refusals.append(
                f"Demo quote is {age.total_seconds():.0f} s old, "
                f"limit {rule.quote_max_age_seconds} s"
            )
    return _conclude(
        GuardId.G10_BREAKER,
        leg,
        policy,
        inputs=inputs,
        exits=exits,
        refusals=refusals,
        missing=missing,
        ok=ok,
    )


def g11_eligibility(
    leg: Leg,
    *,
    policy: Policy,
    spec: InstrumentSpec | None,
    position_qty: Decimal,
    price: Decimal | None,
    equity: Decimal,
    candidate_abs_weight: float | None = None,
) -> GuardRuling:
    """Universe symbols only (anything else is exited); increases only while Demo lists the
    instrument online, and only when the order they imply clears ``minOrderQty`` and
    ``minOrderAmount`` after rounding to ``quantityMultiplier``. Orders above
    ``maxMarketOrderQty`` are split by the planner, not refused.

    ``candidate_abs_weight`` is the ``|weight|`` every other guard left (the kernel passes it);
    the request's own size when ``None``."""
    symbol = leg.symbol
    if symbol in policy.excluded:
        return _conclude(
            GuardId.G11_ELIGIBILITY,
            leg,
            policy,
            inputs={"in_universe": False, "excluded": True},
            exits=(f"{symbol} is excluded for measured Demo venue failures",),
            ok="",
        )
    if policy.entry(symbol) is None:
        return _conclude(
            GuardId.G11_ELIGIBILITY,
            leg,
            policy,
            inputs={"in_universe": False, "excluded": False},
            exits=(f"{symbol} is not in the {len(policy.universe)}-symbol universe",),
            ok="",
        )
    inputs: dict[str, InputValue] = {"in_universe": True}
    refusals: list[str] = []
    missing: list[str] = []
    candidate = abs(leg.reference) if candidate_abs_weight is None else candidate_abs_weight
    inputs["candidate_abs_weight"] = candidate
    if spec is None:
        missing.append("the Demo instrument spec")
    else:
        inputs.update(
            status=spec.status,
            min_order_qty=str(spec.min_order_qty),
            min_order_amount=str(spec.min_order_amount),
            qty_step=str(spec.qty_step),
            max_market_order_qty=None
            if spec.max_market_order_qty is None
            else str(spec.max_market_order_qty),
        )
        if spec.status != "online":
            refusals.append(f"Demo lists {symbol} as {spec.status!r}, not online")
        elif leg.adds_exposure and candidate > leg.hold_ceiling + EPS:
            if price is None or equity <= 0:
                missing.append("a Demo mark and positive equity to size the order")
            else:
                target = target_quantity(candidate, equity, price, spec)
                same_side = position_qty != 0 and (position_qty > 0) == (leg.reference > 0)
                order = round_qty(target - abs(position_qty), spec) if same_side else target
                inputs.update(order_qty=str(order), order_notional=str(order * price))
                if order <= 0:
                    refusals.append(f"the increase rounds to nothing on the {spec.qty_step} grid")
                elif order < spec.min_order_qty:
                    refusals.append(f"order {order} is below minOrderQty {spec.min_order_qty}")
                elif order * price < spec.min_order_amount:
                    refusals.append(
                        f"order notional {order * price} is below minOrderAmount "
                        f"{spec.min_order_amount}"
                    )
                elif spec.max_market_order_qty is not None and spec.max_market_order_qty > 0:
                    legs = math.ceil(order / spec.max_market_order_qty)
                    inputs["split_legs"] = legs
    return _conclude(
        GuardId.G11_ELIGIBILITY,
        leg,
        policy,
        inputs=inputs,
        refusals=refusals,
        missing=missing,
        ok="universe symbol, online on Demo, order clears the venue minimums",
    )


__all__ = [
    "EPS",
    "GrossAllocation",
    "InputValue",
    "Leg",
    "WeekendPhase",
    "allocate_gross",
    "g1_venue_integrity",
    "g2_weekend_freeze",
    "g3_size",
    "g4_stop",
    "g5_daily_kill",
    "g6_turnover",
    "g7_fee_budget",
    "g8_taker_only",
    "g9_grounding",
    "g10_breaker",
    "g11_eligibility",
    "session_open",
    "weekend_phase",
]
