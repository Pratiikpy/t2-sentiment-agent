"""The live-marked mirror, and the weekend the Demo venue could not trade.

Two questions a careful judge asks of a paper record on a sandbox venue, answered from the same log
(win plan Move 5d, DESIGN.md §14.4):

**Is the Demo result a sandbox artefact?** (:func:`live_mirror`.) Every fill the book really took on
Demo is replayed on Bitget's **live** prices: same instant, same side, same quantity, priced at the
live 1H close known at that instant (no later price), charged the same fee rate on the live
notional, and the resulting book marked every hour at live closes. If Demo priced the universe
honestly, the two equity curves agree to within the Demo-live gap; where they part, the gap is on
the page. This is deliberately stricter than the hourly ``equity_live_mirror`` in each ``MARK``
(``book/marks.py``), which re-marks only the *open* positions at live and keeps every realized
Demo P&L, and so converges back to the Demo book whenever the book is flat. Here the realized P&L
is live too.

**What did the weekend freeze cost or save?** (:func:`weekend_counterfactual`.) UTA Demo stops
pricing the US-equity and US-index perps from Friday 20:00 to Monday 00:00 UTC
(``validation/demo_venue/weekend_vol.json``), so G2 flattens those legs and refuses new ones
(``kernel/guards.py`` ``g2_weekend_freeze``). Live Bitget keeps trading them. This arm holds, at
live prices, exactly the exposure G2 took away:

* an *episode* starts wherever G2 **FIRED** on a US-session leg: in a decision's ruling (the model
  asked for a position the freeze or its no-open buffer refused) or in a protective ruling (the
  pre-flatten or freeze closed a position the model held);
* its weight is what the kernel would have approved with G2 lifted: the model's request (the held
  weight, for a protective ruling) cut to every other guard's limit recorded in the same ruling.
  G3's gross share and G11's size check are recomputed in the ruling with G2 already in force, so
  the per-name cap stands in for G3 and G11 is not re-derived;
* it runs from the first live hourly close at or after its start to the first at or after the
  earliest of: the weekend reopen after it, the next decided draft that addresses the symbol, and
  the next episode on the same symbol;
* its quantity is its weight of the starting equity (``1.0``: every figure is a fraction of equity),
  at the live price where it starts, held without costs. It answers what those legs did on live, not
  what trading them would have netted.
"""

import bisect
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Final

from sentiment_agent.analysis.armsim import HOUR, hour_floor
from sentiment_agent.analysis.bootstrap import DEFAULT_RESAMPLES
from sentiment_agent.analysis.metrics import metric_set
from sentiment_agent.book.book import BookBuilder
from sentiment_agent.policy import POLICY_V1
from sentiment_agent.types import (
    ArmKind,
    ArmMark,
    ArmResult,
    ArmSpec,
    Candle,
    DecisionRecord,
    Fill,
    FillVenue,
    GuardId,
    GuardRuling,
    GuardStatus,
    InstrumentRuling,
    KernelRuling,
    LlmOutcome,
    Policy,
    PriceSource,
    Side,
    WeekendRule,
)

MAX_PRICE_AGE: Final = timedelta(hours=2)
"""The oldest live close the mirror will use for an instant. A live perp prints every minute, so
anything older is a hole in the data, reported rather than bridged."""

QUOTE_COIN: Final = "USDT"

_INTERVALS: Final[Mapping[str, timedelta]] = {
    "1m": timedelta(minutes=1),
    "3m": timedelta(minutes=3),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1H": timedelta(hours=1),
    "1h": timedelta(hours=1),
    "4H": timedelta(hours=4),
    "6H": timedelta(hours=6),
    "12H": timedelta(hours=12),
    "1D": timedelta(days=1),
}
"""Bitget candle granularities (``/api/v3/market/history-candles`` ``interval``)."""

MIRROR_SPEC: Final = ArmSpec(
    arm_id="mirror_live",
    kind=ArmKind.MIRROR_LIVE,
    title="The same fills on live prices",
    description="Every Demo fill replayed at Bitget's live price at the same instant, same fee "
    "rate, and marked hourly at live closes: the book without the sandbox.",
    provenance="src/sentiment_agent/analysis/mirror.py (this repository, MIT)",
    uses_llm=True,
    guards=(),
)

WEEKEND_SPEC: Final = ArmSpec(
    arm_id="weekend_counterfactual",
    kind=ArmKind.WEEKEND_COUNTERFACTUAL,
    title="The legs the weekend freeze took away, on live",
    description="The exposure G2 refused or closed on US legs, held at live prices until the "
    "weekend reopen or the model's next decision on the symbol. Before costs.",
    provenance="src/sentiment_agent/analysis/mirror.py (this repository, MIT)",
    uses_llm=True,
    guards=(),
)


# ------------------------------------------------------------------------------------------------
# Live prices
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _LiveSeries:
    closes_at: list[datetime]
    closes: list[Decimal]


class LivePrices:
    """Live closes by symbol, answering "the last live price known at this instant"."""

    def __init__(self, candles: Mapping[str, Sequence[Candle]]) -> None:
        self._series: dict[str, _LiveSeries] = {}
        self._first_open: datetime | None = None
        self._last_close: datetime | None = None
        for symbol, rows in candles.items():
            points: dict[datetime, Decimal] = {}
            for c in rows:
                if c.symbol != symbol:
                    raise ValueError(f"live_marks[{symbol!r}] holds a candle for {c.symbol!r}")
                if c.source is not PriceSource.LIVE:
                    raise ValueError(f"live_marks[{symbol!r}] holds a {c.source.value} candle")
                step = _INTERVALS.get(c.interval)
                if step is None:
                    raise ValueError(f"unknown candle interval {c.interval!r}")
                if c.close <= 0:
                    raise ValueError(
                        f"{symbol} live close at {c.open_time.isoformat()} is not positive"
                    )
                close_at = c.open_time + step
                if close_at in points and points[close_at] != c.close:
                    raise ValueError(f"{symbol} has two live closes at {close_at.isoformat()}")
                points[close_at] = c.close
                if self._first_open is None or c.open_time < self._first_open:
                    self._first_open = c.open_time
                if self._last_close is None or close_at > self._last_close:
                    self._last_close = close_at
            ordered = sorted(points)
            self._series[symbol] = _LiveSeries(ordered, [points[t] for t in ordered])

    @property
    def first_open(self) -> datetime | None:
        return self._first_open

    @property
    def last_close(self) -> datetime | None:
        return self._last_close

    def at(self, symbol: str, when: datetime) -> Decimal | None:
        """The last live close at or before ``when``, no older than :data:`MAX_PRICE_AGE`."""
        series = self._series.get(symbol)
        if series is None:
            return None
        i = bisect.bisect_right(series.closes_at, when)
        if i == 0 or when - series.closes_at[i - 1] > MAX_PRICE_AGE:
            return None
        return series.closes[i - 1]

    def after(self, symbol: str, when: datetime) -> tuple[datetime, Decimal] | None:
        """The first live close at or after ``when``."""
        series = self._series.get(symbol)
        if series is None:
            return None
        i = bisect.bisect_left(series.closes_at, when)
        if i == len(series.closes_at):
            return None
        return series.closes_at[i], series.closes[i]

    def require_at(self, symbol: str, when: datetime) -> Decimal:
        price = self.at(symbol, when)
        if price is None:
            raise ValueError(
                f"no live {symbol} close within {MAX_PRICE_AGE} before {when.isoformat()}"
            )
        return price


def _grid(start: datetime, until: datetime) -> list[datetime]:
    if (start.minute, start.second, start.microsecond) != (0, 0, 0) or (
        until.minute,
        until.second,
        until.microsecond,
    ) != (0, 0, 0):
        raise ValueError("the mark grid must start and end on the hour")
    if until < start:
        raise ValueError("until is before start")
    hours = int((until - start) / HOUR)
    return [start + HOUR * i for i in range(hours + 1)]


def _hour_ceil(at: datetime) -> datetime:
    floor = hour_floor(at)
    return floor if floor == at else floor + HOUR


def _weights(
    builder: BookBuilder, prices: Mapping[str, Decimal], equity: Decimal
) -> tuple[float, float]:
    if equity <= 0:
        return 0.0, 0.0
    held = builder.positions()
    gross = sum((abs(p.qty) * prices[s] for s, p in held.items()), Decimal(0))
    net = sum((p.qty * prices[s] for s, p in held.items()), Decimal(0))
    return float(gross / equity), float(net / equity)


# ------------------------------------------------------------------------------------------------
# The live mirror
# ------------------------------------------------------------------------------------------------


def live_mirror(
    fills: Sequence[Fill],
    live_marks: Mapping[str, Sequence[Candle]],
    starting_equity: Decimal,
    *,
    start: datetime | None = None,
    until: datetime | None = None,
    ci_resamples: int = DEFAULT_RESAMPLES,
) -> ArmResult:
    """The book's own fills on live prices (module docstring).

    The grid runs hourly from ``start`` to ``until`` (defaults: the hour of the earliest live candle
    and of the latest live close); pass the governed book's first and last mark to put the two on
    one grid. A fill after ``until`` is outside the record and is not replayed; one before ``start``
    is an error. Fills are replayed in venue time order (the book's own order, ``book/book.py``).
    For fill prices closer than an hour, pass live candles finer than 1H; the hourly marks use the
    same "last close at or before" rule.
    """
    if (
        not isinstance(starting_equity, Decimal)
        or not starting_equity.is_finite()
        or starting_equity <= 0
    ):
        raise ValueError(
            f"starting equity must be a positive, finite Decimal, got {starting_equity!r}"
        )
    live = LivePrices(live_marks)
    ordered = sorted(
        enumerate(fills),
        key=lambda item: (item[1].executed_at, 0 if item[1].trade_side == "close" else 1, item[0]),
    )
    if start is None:
        candidates = [hour_floor(f.executed_at) for f in fills]
        if live.first_open is not None:
            candidates.append(hour_floor(live.first_open))
        if not candidates:
            raise ValueError("no fills and no live candles: nothing to mirror")
        start = min(candidates)
    if until is None:
        if live.last_close is None:
            raise ValueError("no live candles to mark the mirror at")
        until = hour_floor(live.last_close)
    grid = _grid(start, until)
    builder = BookBuilder(starting_equity=starting_equity, policy=POLICY_V1)
    notional = Decimal(0)
    fees = Decimal(0)
    marks: list[ArmMark] = []
    queue = [f for _, f in ordered]
    if queue and queue[0].executed_at < start:
        raise ValueError(f"fill {queue[0].exec_id} precedes the start of the mirror's grid")
    position = 0
    for hour in grid:
        while position < len(queue) and queue[position].executed_at <= hour:
            fill = queue[position]
            position += 1
            if fill.fee_coin.strip().upper() != QUOTE_COIN:
                raise ValueError(f"fill {fill.exec_id} paid its fee in {fill.fee_coin!r}")
            price = live.require_at(fill.symbol, fill.executed_at)
            fee = fill.fee_paid * price / fill.exec_price
            replay = fill.model_copy(
                update={
                    "exec_price": price,
                    "exec_value": price * fill.exec_qty,
                    "fee_paid": fee,
                    "blob": None,
                }
            )
            builder.apply_fill(replay, decision_id=None, purpose=None)
            notional += price * fill.exec_qty
            fees += fee
        held = builder.positions()
        prices = {s: live.require_at(s, hour) for s in held}
        equity = builder.equity(prices)
        gross, net = _weights(builder, prices, equity)
        marks.append(ArmMark(at=hour, equity=float(equity), gross_weight=gross, net_weight=net))
    trades = tuple(t.model_copy(update={"exit_reason": "mirror"}) for t in builder.closed_trades())
    return ArmResult(
        spec=MIRROR_SPEC,
        marks=tuple(marks),
        trades=trades,
        metrics=metric_set(
            MIRROR_SPEC.arm_id,
            marks,
            trades,
            traded_notional=float(notional),
            fees=float(fees),
            ci_resamples=ci_resamples,
        ),
    )


# ------------------------------------------------------------------------------------------------
# The weekend counterfactual
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WeekendEpisode:
    """One leg G2 took away: held at live from ``start`` to ``end`` at ``weight`` of equity."""

    symbol: str
    weight: float
    start: datetime
    end: datetime
    decision_id: str | None
    ruling_id: str


def reopen_after(at: datetime, rule: WeekendRule) -> datetime:
    """The end of the first weekend freeze ending after ``at`` (Monday 00:00 UTC in policy v1)."""
    week_start = (at - timedelta(days=at.weekday())).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    offset = timedelta(days=rule.freeze_weekday, hours=rule.freeze_hour)
    length = timedelta(
        hours=(
            rule.reopen_weekday * 24
            + rule.reopen_hour
            - rule.freeze_weekday * 24
            - rule.freeze_hour
        )
        % (7 * 24)
    )
    for k in (-1, 0, 1, 2):
        end = week_start + offset + timedelta(weeks=k) + length
        if end > at:
            return end
    raise AssertionError("unreachable: a freeze ends within two weeks of any instant")


def _ceiling(ruling: GuardRuling) -> float | None:
    return 0.0 if ruling.forces_exit else ruling.ceiling_abs_weight


def weight_without_g2(
    inst: InstrumentRuling, book_rulings: Sequence[GuardRuling], policy: Policy
) -> float:
    """What the kernel would have approved for ``inst`` with G2 lifted: the request cut to every
    other recorded limit, with the per-name cap standing in for G3 (module docstring)."""
    requested = inst.reference
    size = min(abs(requested), policy.per_name_max)
    for ruling in (*inst.rulings, *book_rulings):
        if ruling.guard in (GuardId.G2_WEEKEND_FREEZE, GuardId.G3_SIZE, GuardId.G11_ELIGIBILITY):
            continue
        ceiling = _ceiling(ruling)
        if ceiling is not None:
            size = min(size, ceiling)
    return math.copysign(size, requested) if size > 0 else 0.0


def weekend_episodes(
    decisions: Sequence[DecisionRecord],
    rulings: Sequence[KernelRuling],
    *,
    policy: Policy = POLICY_V1,
) -> list[WeekendEpisode]:
    """Every leg G2 took away, with its end (module docstring)."""
    decided = {d.decision_id for d in decisions if d.outcome is LlmOutcome.DECIDED}
    ordered = sorted(rulings, key=lambda r: r.at)
    addressed: dict[str, list[datetime]] = {}
    for ruling in ordered:
        if ruling.decision_id in decided:
            for inst in ruling.instruments:
                if inst.proposed_weight is not None:
                    addressed.setdefault(inst.symbol, []).append(ruling.at)
    starts: list[tuple[datetime, str, float, str | None, str]] = []
    for ruling in ordered:
        if ruling.decision_id is not None and ruling.decision_id not in decided:
            continue
        for inst in ruling.instruments:
            entry = policy.entry(inst.symbol)
            if entry is None or not entry.asset_class.follows_us_session:
                continue
            g2 = next((g for g in inst.rulings if g.guard is GuardId.G2_WEEKEND_FREEZE), None)
            if g2 is None or g2.status is not GuardStatus.FIRED:
                continue
            weight = weight_without_g2(inst, ruling.book_rulings, policy)
            if weight == 0.0 or abs(weight) <= abs(inst.approved_weight) + 1e-12:
                continue
            starts.append((ruling.at, inst.symbol, weight, ruling.decision_id, ruling.ruling_id))
    episodes: list[WeekendEpisode] = []
    for index, (at, symbol, weight, decision_id, ruling_id) in enumerate(starts):
        candidates = [reopen_after(at, policy.weekend)]
        later_decisions = [t for t in addressed.get(symbol, []) if t > at]
        if later_decisions:
            candidates.append(later_decisions[0])
        later_episodes = [s[0] for s in starts[index + 1 :] if s[1] == symbol and s[0] > at]
        if later_episodes:
            candidates.append(later_episodes[0])
        episodes.append(
            WeekendEpisode(
                symbol=symbol,
                weight=weight,
                start=at,
                end=min(candidates),
                decision_id=decision_id,
                ruling_id=ruling_id,
            )
        )
    return episodes


def weekend_counterfactual(
    decisions: Sequence[DecisionRecord],
    rulings: Sequence[KernelRuling],
    live_marks: Mapping[str, Sequence[Candle]],
    *,
    policy: Policy = POLICY_V1,
    ci_resamples: int = DEFAULT_RESAMPLES,
) -> ArmResult:
    """The legs the weekend freeze took away, held at live prices (module docstring). Marks run
    hourly across the episodes; with no episode the arm has no marks and every metric reads as
    nothing happened."""
    live = LivePrices(live_marks)
    episodes = weekend_episodes(decisions, rulings, policy=policy)
    unit = Decimal(1)
    builder = BookBuilder(starting_equity=unit, policy=policy)
    fills: list[tuple[datetime, int, Fill, str | None]] = []
    notional = Decimal(0)
    starts: list[datetime] = []
    ends: list[datetime] = []
    for n, episode in enumerate(episodes):
        opened = live.after(episode.symbol, episode.start)
        if opened is None:
            continue  # the record ends before a live close shows the episode at all
        open_at, open_price = opened
        closed = live.after(episode.symbol, episode.end)
        if closed is not None and closed[0] <= open_at:
            continue  # starts and ends inside one hour: nothing held across a close
        qty = Decimal(repr(abs(episode.weight))) * unit / open_price
        side = Side.BUY if episode.weight > 0 else Side.SELL
        fills.append(
            (
                open_at,
                1,
                _episode_fill(episode, n, "open", side, qty, open_price, open_at),
                episode.decision_id,
            )
        )
        notional += qty * open_price
        starts.append(open_at)
        if closed is not None:
            close_at, close_price = closed
            back = Side.SELL if side is Side.BUY else Side.BUY
            fills.append(
                (
                    close_at,
                    0,
                    _episode_fill(episode, n, "close", back, qty, close_price, close_at),
                    episode.decision_id,
                )
            )
            notional += qty * close_price
            ends.append(close_at)
    if not starts:
        return _empty_weekend(ci_resamples)
    last = _hour_ceil(max([*starts, *ends]))
    if len(ends) < len(starts) and live.last_close is not None:
        last = max(last, hour_floor(live.last_close))  # an episode still open: mark it to the end
    grid = _grid(hour_floor(min(starts)), last)
    fills.sort(key=lambda item: (item[0], item[1]))
    marks: list[ArmMark] = []
    position = 0
    for hour in grid:
        while position < len(fills) and fills[position][0] <= hour:
            _, _, fill, decision_id = fills[position]
            position += 1
            builder.apply_fill(fill, decision_id=decision_id, purpose=None)
        held = builder.positions()
        prices = {s: live.require_at(s, hour) for s in held}
        equity = builder.equity(prices)
        gross, net = _weights(builder, prices, equity)
        marks.append(ArmMark(at=hour, equity=float(equity), gross_weight=gross, net_weight=net))
    trades = tuple(
        t.model_copy(update={"exit_reason": "weekend_counterfactual"})
        for t in builder.closed_trades()
    )
    return ArmResult(
        spec=WEEKEND_SPEC,
        marks=tuple(marks),
        trades=trades,
        metrics=metric_set(
            WEEKEND_SPEC.arm_id,
            marks,
            trades,
            traded_notional=float(notional),
            fees=0.0,
            ci_resamples=ci_resamples,
        ),
    )


def _episode_fill(
    episode: WeekendEpisode,
    n: int,
    leg: str,
    side: Side,
    qty: Decimal,
    price: Decimal,
    at: datetime,
) -> Fill:
    exec_id = f"weekend:{n}:{leg}"
    return Fill(
        exec_id=exec_id,
        venue_order_id=exec_id,
        client_oid=None,
        symbol=episode.symbol,
        side=side,
        exec_price=price,
        exec_qty=qty,
        exec_value=price * qty,
        fee_paid=Decimal(0),
        fee_coin=QUOTE_COIN,
        trade_scope=None,
        trade_side="open" if leg == "open" else "close",
        exec_pnl=None,
        executed_at=at,
        venue=FillVenue.SIMULATED,
    )


def _empty_weekend(ci_resamples: int) -> ArmResult:
    return ArmResult(
        spec=WEEKEND_SPEC,
        marks=(),
        trades=(),
        metrics=metric_set(
            WEEKEND_SPEC.arm_id, (), (), traded_notional=0.0, fees=0.0, ci_resamples=ci_resamples
        ),
    )


__all__ = [
    "MAX_PRICE_AGE",
    "MIRROR_SPEC",
    "WEEKEND_SPEC",
    "LivePrices",
    "WeekendEpisode",
    "live_mirror",
    "reopen_after",
    "weekend_counterfactual",
    "weekend_episodes",
    "weight_without_g2",
]
