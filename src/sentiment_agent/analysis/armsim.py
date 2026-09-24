"""The arm simulator: any target-weight schedule, ruled, sized, filled and marked as the book is.

Every comparison in the published record (baselines, the ungoverned twin, rival agents, the red
team) needs to answer one question: *if a different decision-maker had asked for these weights at
these instants, what would the paper book have done?* An answer is only fair if everything except
the decision-maker is identical, so this module does not approximate the production path, it runs
it:

* **The kernel rules** every schedule entry with the production :class:`RiskKernel`, restricted to
  the arm's ``spec.guards`` (``VENUE_GUARDS`` for baselines and rivals, every guard for the
  governed replica, none for the ungoverned twin). The kernel is re-bound to a simulation clock
  (``type(kernel)(kernel.policy, clock)``): a ruling is a pure function of its arguments and the
  kernel's instant (``kernel/kernel.py`` module docstring), so this is the same kernel at the
  simulated time.
* **The planner sizes** every approved weight with the production :func:`plan_orders`: quantities on
  the Demo grid, below-minimum legs skipped, splits above ``maxMarketOrderQty``, flips as a close
  then an open.
* **The book books** every fill with the production :class:`BookBuilder`: average-cost positions,
  gross realized P&L, fees, closed trades flat to flat, flips with fees split by quantity, and the
  book state (peak, day-open equity, the model's rebalance count, losing streak) the kernel reads.
* **The breaker** is the production :class:`Breaker` when G10 is among the arm's guards.

**The cost model**, the same for every arm and every fill: taker only (G8), filled at the reference
Demo mark moved by half the Demo spread against the order (``spreads_bps``: the snapshot Demo
spread per symbol, one number per symbol so a stop fill between snapshots is costed like any other),
plus the Demo taker fee (``takerFeeRate`` from the instrument spec, 6 bps on all 14 names,
``validation/demo_venue/universe_probe.json``). The envelope costs a side at 6 bps + 2 bps
(``envelope_clean.py:33``); here the half-spread is the measured one.

**Marks** are the Demo **mark** price, 1H candles (``demo_marks``), the price the book is marked at
and the venue's stops trigger on. The equity is marked at every UTC hour from ``start`` to
``until`` at the close of the hour's candle; a decision between hours sizes and fills at the Demo
quote in its kernel inputs.

**Between decisions** (only for the guards the arm carries):

* **Stops (G4).** Every open position carries the venue stop ``stop_loss_pct`` from its average
  entry, on the price grid, as ``execution/stops.py`` keeps it. A stop is checked against each
  hour's mark candle: a candle that opens beyond the stop fills at the open (a gap), one whose
  range reaches it fills at the stop. In the hour a position was opened or changed, only the close
  is used, because an hourly candle cannot say whether its low came before or after the fill. A
  stop fill is stamped at the candle's open for a gap and at the candle's end otherwise.
* **Protective checks (G1, G2, G5, G10)**, at every hour, through the kernel exactly as the
  60-second loop runs them (a ruling on the held book with no proposal, DESIGN.md §10.5), with the
  hour's mark as the only price. The kernel is consulted whenever one of its exit conditions can
  hold (a US leg held in the weekend pre-flatten or freeze, the day's loss at the kill, a halted
  breaker); G1 cannot exit without a Demo quote and there is none between snapshots. Pass
  ``audit_protective=True`` to consult the kernel at every hour regardless: the result is
  identical (tested), only slower.

What hourly data cannot show is stated rather than guessed: an intra-hour dip that recovers by the
close does not trip the daily kill here, while the real loop, checking every minute, could.

**Units.** ``starting_equity`` is in the book's quote currency (USDT for the real book) because
the venue minimums are: G11 and the planner refuse an order below ``minOrderQty`` or
``minOrderAmount`` (5 USDT) exactly as the executor would, so an arm simulated at the default
``1.0`` places no order on the real Demo grid. Pass the real book's starting equity to compare with
it.
"""

import bisect
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Final

from sentiment_agent.analysis.bootstrap import DEFAULT_RESAMPLES
from sentiment_agent.analysis.metrics import metric_set
from sentiment_agent.book.book import BookBuilder
from sentiment_agent.clock import ManualClock
from sentiment_agent.kernel.breaker import Breaker, book_conditions, most_severe
from sentiment_agent.kernel.guards import weekend_phase
from sentiment_agent.kernel.kernel import PROTECTIVE_GUARDS, RiskKernel
from sentiment_agent.kernel.planner import MEASURED_DEMO_TAKER_FEE, plan_orders, stop_price
from sentiment_agent.types import (
    Activation,
    ArmMark,
    ArmResult,
    ArmSpec,
    BookState,
    BreakerState,
    Candle,
    Fill,
    FillVenue,
    GuardId,
    InstrumentRuling,
    InstrumentSpec,
    KernelInputs,
    KernelRuling,
    MarkPoint,
    OrderPlan,
    OrderPurpose,
    PerceptionSnapshot,
    Policy,
    Position,
    PositionMark,
    PriceSource,
    ProtectiveReason,
    RulingContext,
    Side,
)

HOUR: Final = timedelta(hours=1)
HOURLY_INTERVALS: Final[frozenset[str]] = frozenset({"1H", "1h"})
QUOTE_COIN: Final = "USDT"

ScheduleEntry = tuple[datetime, Mapping[str, float], KernelInputs]
"""``(instant, target weights, the kernel inputs at that instant)``."""

GUARD_CAUSE: Final[Mapping[GuardId, ProtectiveReason]] = {
    GuardId.G1_VENUE_INTEGRITY: ProtectiveReason.VENUE_INTEGRITY,
    GuardId.G2_WEEKEND_FREEZE: ProtectiveReason.WEEKEND_FREEZE,
    GuardId.G5_DAILY_KILL: ProtectiveReason.DAILY_KILL,
    GuardId.G10_BREAKER: ProtectiveReason.BREAKER,
}
"""The protective reason behind an exit a protective-loop guard forced (``kernel.py``
``_protective_reason`` files rulings the same way)."""

_ZERO: Final = Decimal(0)
_ONE: Final = Decimal(1)


def hour_floor(at: datetime) -> datetime:
    return at.replace(minute=0, second=0, microsecond=0)


def _require_utc(at: datetime, what: str) -> None:
    if at.tzinfo is None or at.utcoffset() != timedelta(0):
        raise ValueError(f"{what} must be timezone-aware UTC")


def _on_hour(at: datetime) -> bool:
    return (at.minute, at.second, at.microsecond) == (0, 0, 0)


def _decimal(value: float, what: str) -> Decimal:
    if not math.isfinite(value):
        raise ValueError(f"{what} must be finite, got {value}")
    return Decimal(repr(float(value)))


@dataclass(frozen=True, slots=True)
class _Series:
    """One symbol's Demo mark candles, indexed by open time and by close time."""

    by_open: dict[datetime, Candle]
    closes_at: list[datetime]
    closes: list[Decimal]


def _index_candles(symbol: str, candles: Sequence[Candle]) -> _Series:
    by_open: dict[datetime, Candle] = {}
    for c in candles:
        if c.symbol != symbol:
            raise ValueError(f"demo_marks[{symbol!r}] holds a candle for {c.symbol!r}")
        if c.source is not PriceSource.DEMO:
            raise ValueError(f"demo_marks[{symbol!r}] holds a {c.source.value} candle")
        if c.kind != "mark":
            raise ValueError(
                f"demo_marks[{symbol!r}] holds a {c.kind!r} candle; the book is marked, and its "
                "stops trigger, on the Demo mark price"
            )
        if c.interval not in HOURLY_INTERVALS:
            raise ValueError(f"demo_marks[{symbol!r}] holds a {c.interval!r} candle, not 1H")
        if not _on_hour(c.open_time):
            raise ValueError(f"{symbol} candle at {c.open_time.isoformat()} is not on the hour")
        if min(c.open, c.high, c.low, c.close) <= 0 or c.low > c.high:
            raise ValueError(f"{symbol} candle at {c.open_time.isoformat()} is not a valid candle")
        if c.open_time in by_open and by_open[c.open_time] != c:
            raise ValueError(f"{symbol} has two different candles at {c.open_time.isoformat()}")
        by_open[c.open_time] = c
    opens = sorted(by_open)
    return _Series(
        by_open=by_open,
        closes_at=[t + HOUR for t in opens],
        closes=[by_open[t].close for t in opens],
    )


class ArmSimulator:
    """Runs target-weight schedules through the production kernel, planner and book (module
    docstring). One simulator can run any number of arms; runs share nothing but the price path."""

    def __init__(
        self,
        *,
        kernel: RiskKernel,
        policy: Policy,
        demo_marks: Mapping[str, Sequence[Candle]],
        spreads_bps: Mapping[str, float],
        starting_equity: float = 1.0,
        specs: Mapping[str, InstrumentSpec] | None = None,
        audit_protective: bool = False,
    ) -> None:
        if kernel.policy != policy:
            raise ValueError("the kernel rules under a different policy from the simulator's")
        if not math.isfinite(starting_equity) or starting_equity <= 0:
            raise ValueError(f"starting equity must be positive and finite, got {starting_equity}")
        for symbol, spread in spreads_bps.items():
            if not math.isfinite(spread) or spread < 0:
                raise ValueError(f"spreads_bps[{symbol!r}] must be finite and non-negative")
            if spread >= 20_000:
                raise ValueError(f"spreads_bps[{symbol!r}] = {spread} is not a spread")
        checked_specs: dict[str, InstrumentSpec] = {}
        for symbol, spec in (specs or {}).items():
            if spec.symbol != symbol:
                raise ValueError(f"specs[{symbol!r}] holds the spec of {spec.symbol!r}")
            if spec.source is not PriceSource.DEMO:
                raise ValueError(f"specs[{symbol!r}] must be the Demo venue's instrument limits")
            checked_specs[symbol] = spec
        self._kernel = kernel
        self._policy = policy
        self._series = {s: _index_candles(s, c) for s, c in demo_marks.items()}
        self._marks = {s: tuple(c) for s, c in demo_marks.items()}
        self._spreads = dict(spreads_bps)
        self._start = starting_equity
        self._specs = checked_specs
        self._audit = audit_protective

    # --------------------------------------------------------------------------------------------
    # What the simulator knows
    # --------------------------------------------------------------------------------------------

    @property
    def policy(self) -> Policy:
        return self._policy

    @property
    def starting_equity(self) -> float:
        return self._start

    @property
    def demo_marks(self) -> Mapping[str, tuple[Candle, ...]]:
        return self._marks

    @property
    def spreads_bps(self) -> Mapping[str, float]:
        return self._spreads

    @property
    def specs(self) -> Mapping[str, InstrumentSpec]:
        return self._specs

    def candle(self, symbol: str, open_time: datetime) -> Candle | None:
        series = self._series.get(symbol)
        return None if series is None else series.by_open.get(open_time)

    def price_at(self, symbol: str, at: datetime) -> Decimal | None:
        """The Demo mark known at ``at``: the close of the latest candle that closed at or before
        it. ``None`` before the first candle closes."""
        series = self._series.get(symbol)
        if series is None:
            return None
        i = bisect.bisect_right(series.closes_at, at)
        return series.closes[i - 1] if i > 0 else None

    def price_after(self, symbol: str, at: datetime) -> Decimal | None:
        """The first Demo mark observed at or after ``at``: the close of the first candle closing at
        or after it. ``None`` when no candle closes that late."""
        series = self._series.get(symbol)
        if series is None:
            return None
        i = bisect.bisect_left(series.closes_at, at)
        return series.closes[i] if i < len(series.closes) else None

    def last_close(self, symbol: str) -> tuple[datetime, Decimal] | None:
        series = self._series.get(symbol)
        if series is None or not series.closes:
            return None
        return series.closes_at[-1], series.closes[-1]

    def default_until(self) -> datetime | None:
        """The last hour every supplied mark series reaches (``None`` without candles)."""
        ends = [s.closes_at[-1] for s in self._series.values() if s.closes_at]
        return min(ends) if ends else None

    def half_spread(self, symbol: str) -> Decimal:
        """Half the Demo spread of ``symbol`` as a fraction of price."""
        spread = self._spreads.get(symbol)
        if spread is None:
            raise ValueError(f"no Demo spread for {symbol}: its fills cannot be costed")
        return _decimal(spread, "spread") / Decimal(20_000)

    def fee_rate(self, symbol: str, spec: InstrumentSpec | None = None) -> Decimal:
        """The Demo taker fee of ``symbol``: from ``spec``, else the simulator's specs, else the
        measured Demo rate (0.0006 on all 14 names)."""
        chosen = spec if spec is not None else self._specs.get(symbol)
        return chosen.taker_fee_rate if chosen is not None else MEASURED_DEMO_TAKER_FEE

    def cost_rate(self, symbol: str) -> float:
        """One side's cost as a fraction of notional: taker fee plus half the Demo spread."""
        return float(self.fee_rate(symbol) + self.half_spread(symbol))

    def kernel_inputs(self, snapshot: PerceptionSnapshot) -> KernelInputs:
        """The kernel inputs a decision on ``snapshot`` was ruled with: its Demo and live quotes,
        the simulator's Demo instrument specs, the Demo index staleness from its features."""
        if not self._specs:
            raise ValueError(
                "the simulator holds no Demo instrument specs; pass specs= the limits the real "
                "book was ruled with, or G11 refuses every increase for want of them"
            )
        return KernelInputs(
            at=snapshot.taken_at,
            demo_quotes=dict(snapshot.demo_quotes),
            live_quotes=dict(snapshot.live_quotes),
            specs=dict(self._specs),
            demo_index_move_bps_3h={
                s: f.demo_index_move_bps_3h for s, f in snapshot.features.items()
            },
            snapshot_id=snapshot.snapshot_id,
            snapshot_taken_at=snapshot.taken_at,
        )

    # --------------------------------------------------------------------------------------------
    # Running an arm
    # --------------------------------------------------------------------------------------------

    def run(
        self,
        spec: ArmSpec,
        schedule: Sequence[tuple[datetime, Mapping[str, float], KernelInputs]],
        *,
        contexts: Sequence[RulingContext] | None = None,
        start: datetime | None = None,
        until: datetime | None = None,
        ci_resamples: int = DEFAULT_RESAMPLES,
    ) -> ArmResult:
        """Rule, size, fill and mark ``schedule`` under ``spec.guards``.

        Each entry is ruled at its instant as a model decision. ``contexts`` (one per entry) carries
        what a real decision gives the kernel beyond weights: its id, grounding reports and declared
        invalidations. Without it each entry is ruled under the id ``"<arm_id>#<index>"``, with no
        grounding (so G9, if the arm carries it, fails closed on every increase).

        Marks run hourly from ``start`` (default: the hour of the first entry) to ``until``
        (default: :meth:`default_until`), both on the hour. Entries at or after ``until`` are not
        simulated, since no mark could show them; an entry before ``start`` is an error.
        """
        entries = list(schedule)
        if contexts is not None and len(contexts) != len(entries):
            raise ValueError("contexts must hold one ruling context per schedule entry")
        previous: datetime | None = None
        for t, _, _ in entries:
            _require_utc(t, "a schedule instant")
            if previous is not None and t < previous:
                raise ValueError("schedule entries must be in time order")
            previous = t
        if start is None:
            if not entries:
                raise ValueError("an empty schedule needs an explicit start")
            start = hour_floor(entries[0][0])
        if until is None:
            until = self.default_until()
            if until is None:
                raise ValueError("no Demo marks to run against")
        for name, value in (("start", start), ("until", until)):
            _require_utc(value, name)
            if not _on_hour(value):
                raise ValueError(f"{name} must be on the hour, got {value.isoformat()}")
        if until < start:
            raise ValueError("until is before start")
        if entries and entries[0][0] < start:
            raise ValueError("a schedule entry precedes the start of the marks")
        run = _ArmRun(self, spec, start=start, until=until, audit=self._audit)
        for index, (t, targets, inputs) in enumerate(entries):
            if t >= until:
                break
            context = (
                contexts[index]
                if contexts is not None
                else RulingContext(decision_id=f"{spec.arm_id}#{index}", protective_reason=None)
            )
            if context.decision_id is None or context.protective_reason is not None:
                raise ValueError("a schedule entry is ruled as a model decision, with its id")
            run.decide(t, targets, inputs, context)
        run.finish()
        marks = tuple(run.marks)
        trades = run.book.closed_trades()
        return ArmResult(
            spec=spec,
            marks=marks,
            trades=trades,
            metrics=metric_set(
                spec.arm_id,
                marks,
                trades,
                traded_notional=float(run.notional),
                fees=float(run.fees),
                ci_resamples=ci_resamples,
            ),
        )


class _ArmRun:
    """The state of one arm while it runs."""

    def __init__(
        self,
        sim: ArmSimulator,
        spec: ArmSpec,
        *,
        start: datetime,
        until: datetime,
        audit: bool,
    ) -> None:
        self.sim = sim
        self.spec = spec
        self.policy = sim.policy
        self.guards = frozenset(spec.guards)
        self.protective_guards = self.guards & PROTECTIVE_GUARDS
        self.audit = audit
        self.clock = ManualClock(start)
        self.kernel = type(sim._kernel)(self.policy, self.clock)
        self.breaker = (
            Breaker(self.policy, self.clock) if GuardId.G10_BREAKER in self.guards else None
        )
        self.idle_breaker = BreakerState(activation=Activation.ACTIVE, since=start, trips=())
        self.book = BookBuilder(
            starting_equity=_decimal(sim.starting_equity, "equity"), policy=self.policy
        )
        self.specs: dict[str, InstrumentSpec] = dict(sim.specs)
        self.last_price: dict[str, Decimal] = {}
        self.stops: dict[str, Decimal] = {}
        self.changed_at: dict[str, datetime] = {}
        self.marks: list[ArmMark] = []
        self.notional = _ZERO
        self.fees = _ZERO
        self.fills = 0
        self.until = until
        self._hour(start)
        self.next_hour = start + HOUR

    # --- time ------------------------------------------------------------------------------

    def _advance(self, t: datetime) -> None:
        while self.next_hour <= t and self.next_hour <= self.until:
            self._hour(self.next_hour)
            self.next_hour += HOUR

    def finish(self) -> None:
        self._advance(self.until)

    def _hour(self, h: datetime) -> None:
        """The candle that ends at ``h``: stops inside it, then protective checks, then the mark."""
        held = self.book.positions()
        start = h - HOUR
        for symbol, position in held.items():
            candle = self.sim.candle(symbol, start)
            if candle is None:
                raise ValueError(
                    f"no Demo mark candle for {symbol} at {start.isoformat()}; a held position "
                    "cannot be marked or have its stop checked without one"
                )
            if GuardId.G4_STOP in self.guards:
                self._check_stop(symbol, position, candle, h)
            self.last_price[symbol] = candle.close
        self.clock.set(h)
        self._protective(h)
        self._mark(h)

    # --- book ------------------------------------------------------------------------------

    def _state(self, at: datetime, prices: Mapping[str, Decimal]) -> BookState:
        activation = (
            self.breaker.state().activation if self.breaker is not None else Activation.ACTIVE
        )
        return self.book.state(
            at=at, marks=prices, mark_source=PriceSource.DEMO, activation=activation
        )

    def _assess(
        self, book: BookState, inputs: KernelInputs, decision_id: str | None
    ) -> BreakerState:
        if self.breaker is None:
            return self.idle_breaker
        state, _ = self.breaker.assess(
            book, inputs=inputs, llm_outage=False, decision_id=decision_id
        )
        return state

    def _held_prices(self, held: Mapping[str, Position]) -> dict[str, Decimal]:
        return {s: self.last_price[s] for s in held}

    def _mark(self, h: datetime) -> None:
        held = self.book.positions()
        prices = self._held_prices(held)
        equity = self.book.equity(prices)
        gross = net = 0.0
        if equity > 0:
            gross = float(sum((abs(p.qty) * prices[s] for s, p in held.items()), _ZERO) / equity)
            net = float(sum((p.qty * prices[s] for s, p in held.items()), _ZERO) / equity)
        self.marks.append(ArmMark(at=h, equity=float(equity), gross_weight=gross, net_weight=net))
        self.book.record_mark(
            MarkPoint(
                at=h,
                equity_book=equity,
                equity_venue=None,
                equity_live_mirror=None,
                gross_weight=gross,
                net_weight=net,
                positions=tuple(
                    PositionMark(
                        symbol=s,
                        qty=p.qty,
                        demo_mark=prices[s],
                        live_mark=None,
                        demo_index=None,
                        unrealized_demo=p.qty * (prices[s] - p.avg_entry),
                        unrealized_live=None,
                    )
                    for s, p in held.items()
                ),
            )
        )

    # --- fills -----------------------------------------------------------------------------

    def _fill(
        self,
        *,
        symbol: str,
        side: Side,
        qty: Decimal,
        price: Decimal,
        at: datetime,
        client_oid: str | None,
        decision_id: str | None,
        purpose: OrderPurpose | None,
        cause: ProtectiveReason | GuardId | None,
    ) -> None:
        held = self.book.positions().get(symbol)
        reducing = held is not None and (held.qty > 0) != (side is Side.BUY)
        if not reducing and symbol not in self.sim.demo_marks:
            raise ValueError(f"no Demo marks for {symbol}: a position in it could not be marked")
        fee = qty * price * self.sim.fee_rate(symbol, self.specs.get(symbol))
        self.fills += 1
        exec_id = f"{self.spec.arm_id}:{self.fills}"
        fill = Fill(
            exec_id=exec_id,
            venue_order_id=exec_id,
            client_oid=client_oid,
            symbol=symbol,
            side=side,
            exec_price=price,
            exec_qty=qty,
            exec_value=price * qty,
            fee_paid=fee,
            fee_coin=QUOTE_COIN,
            trade_scope="taker",
            trade_side="close" if reducing else "open",
            exec_pnl=None,
            executed_at=at,
            venue=FillVenue.SIMULATED,
        )
        self.book.apply_fill(fill, decision_id=decision_id, purpose=purpose, cause=cause)
        self.notional += price * qty
        self.fees += fee
        self.changed_at[symbol] = at
        position = self.book.positions().get(symbol)
        if position is None:
            self.stops.pop(symbol, None)
        elif GuardId.G4_STOP in self.guards:
            self.stops[symbol] = self._stop_for(symbol, position)

    def _stop_for(self, symbol: str, position: Position) -> Decimal:
        """The venue stop ``execution/stops.py`` keeps: ``stop_loss_pct`` from the average entry, on
        the price grid toward the entry (tighter, never looser)."""
        side = Side.BUY if position.qty > 0 else Side.SELL
        spec = self.specs.get(symbol)
        if spec is not None:
            try:
                return stop_price(position.avg_entry, side, self.policy, spec)
            except ValueError:
                pass
        pct = Decimal(repr(self.policy.stop_loss_pct))
        return position.avg_entry * (_ONE - pct if side is Side.BUY else _ONE + pct)

    def _taker_price(self, symbol: str, reference: Decimal, side: Side) -> Decimal:
        half = self.sim.half_spread(symbol)
        return reference * (_ONE + half) if side is Side.BUY else reference * (_ONE - half)

    def _check_stop(self, symbol: str, position: Position, candle: Candle, h: datetime) -> None:
        stop = self.stops.get(symbol)
        if stop is None:
            return
        long = position.qty > 0
        candle_start = candle.open_time
        changed = self.changed_at.get(symbol, candle_start)
        # A gap fill is stamped at the candle's open, unless the position was filled at that very
        # instant: the book folds a closing fill before an opening one at the same timestamp, so
        # the stop is then stamped at the candle's end to stay after the fill it closes.
        gap_at = candle_start if changed < candle_start else h
        trigger: Decimal | None = None
        at = h
        if changed > candle_start:
            # Changed inside this hour: only the close is known to come after the fill.
            if (long and candle.close <= stop) or (not long and candle.close >= stop):
                trigger = candle.close
        elif long:
            if candle.open <= stop:
                trigger, at = candle.open, gap_at
            elif candle.low <= stop:
                trigger = stop
        elif candle.open >= stop:
            trigger, at = candle.open, gap_at
        elif candle.high >= stop:
            trigger = stop
        if trigger is None:
            return
        side = Side.SELL if long else Side.BUY
        self._fill(
            symbol=symbol,
            side=side,
            qty=abs(position.qty),
            price=self._taker_price(symbol, trigger, side),
            at=at,
            client_oid=None,
            decision_id=None,
            purpose=None,
            cause=ProtectiveReason.STOP_FILLED,
        )

    def _execute(self, plan: OrderPlan, ruling: KernelRuling, at: datetime) -> None:
        by_symbol = {inst.symbol: inst for inst in ruling.instruments}
        for intent in plan.intents:
            cause: ProtectiveReason | GuardId | None = None
            if intent.purpose is OrderPurpose.PROTECTIVE_EXIT:
                cause = _exit_cause(ruling, by_symbol.get(intent.symbol))
            self._fill(
                symbol=intent.symbol,
                side=intent.side,
                qty=intent.qty,
                price=self._taker_price(intent.symbol, intent.reference_price, intent.side),
                at=at,
                client_oid=intent.client_oid,
                decision_id=ruling.decision_id,
                purpose=intent.purpose,
                cause=cause,
            )
            self.last_price[intent.symbol] = intent.reference_price

    # --- decisions and protective checks ------------------------------------------------

    def decide(
        self,
        t: datetime,
        targets: Mapping[str, float],
        inputs: KernelInputs,
        context: RulingContext,
    ) -> None:
        self._advance(t)
        self.clock.set(t)
        self.specs.update(inputs.specs)
        held = self.book.positions()
        prices: dict[str, Decimal] = {}
        for symbol in held:
            quote = inputs.demo_quotes.get(symbol)
            prices[symbol] = (
                quote.mark if quote is not None and quote.mark > 0 else self.last_price[symbol]
            )
        book = self._state(t, prices)
        breaker = self._assess(book, inputs, context.decision_id)
        ruling = self.kernel.rule(
            proposed=dict(targets),
            book=book,
            inputs=inputs,
            context=context,
            breaker=breaker,
            guards=self.guards,
        )
        plan = plan_orders(ruling, book, inputs, self.policy, now=t)
        self._execute(plan, ruling, t)

    def _protective(self, h: datetime) -> None:
        if not self.protective_guards:
            return
        held = self.book.positions()
        if not held:
            return
        book = self._state(h, self._held_prices(held))
        inputs = KernelInputs(
            at=h,
            demo_quotes={},
            live_quotes={},
            specs=dict(self.specs),
            demo_index_move_bps_3h={},
            snapshot_id=None,
            snapshot_taken_at=None,
        )
        breaker = self._assess(book, inputs, None)
        cause = self._exit_condition(book, breaker, h, held)
        if cause is None and not self.audit:
            return
        ruling = self.kernel.rule(
            proposed=None,
            book=book,
            inputs=inputs,
            context=RulingContext(
                decision_id=None, protective_reason=cause or ProtectiveReason.BREAKER
            ),
            breaker=breaker,
            guards=self.protective_guards,
        )
        if not ruling.changed_by_kernel:
            return
        plan = plan_orders(ruling, book, inputs, self.policy, now=h)
        self._execute(plan, ruling, h)

    def _exit_condition(
        self, book: BookState, breaker: BreakerState, h: datetime, held: Mapping[str, Position]
    ) -> ProtectiveReason | None:
        """Whether any protective guard the arm carries could force an exit now, using the guards'
        own arithmetic (``BookState.day_return``, ``book_conditions``, ``weekend_phase``). The
        kernel then rules; this only decides whether asking it can change anything."""
        guards = self.protective_guards
        if (
            GuardId.G5_DAILY_KILL in guards
            and book.day_open_equity > 0
            and book.day_return <= -self.policy.daily_kill_pct
        ):
            return ProtectiveReason.DAILY_KILL
        if GuardId.G10_BREAKER in guards:
            demanded, _ = book_conditions(
                book, llm_outage=False, policy=self.policy, include_daily_kill=False
            )
            if most_severe(breaker.activation, demanded) is Activation.HALTED:
                return ProtectiveReason.BREAKER
        if GuardId.G2_WEEKEND_FREEZE in guards and weekend_phase(h, self.policy.weekend) in (
            "preflatten",
            "frozen",
        ):
            for symbol in held:
                entry = self.policy.entry(symbol)
                if entry is not None and entry.asset_class.follows_us_session:
                    return ProtectiveReason.WEEKEND_FREEZE
        return None


def _exit_cause(
    ruling: KernelRuling, inst: InstrumentRuling | None
) -> ProtectiveReason | GuardId | None:
    """Why a forced exit happened: the protective reason of the guard that bound, for a ruling of
    the protective loop; the guard itself, for an exit forced inside a model decision (the
    ``cause`` ``book/book.py`` records either way)."""
    guard = inst.binding_guard if inst is not None else None
    if ruling.protective_reason is not None:
        if guard is not None and guard in GUARD_CAUSE:
            return GUARD_CAUSE[guard]
        return ruling.protective_reason
    return guard


__all__ = [
    "GUARD_CAUSE",
    "HOUR",
    "HOURLY_INTERVALS",
    "ArmSimulator",
    "ScheduleEntry",
    "hour_floor",
]
