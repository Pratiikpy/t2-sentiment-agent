"""Event triggers and their admission (DESIGN.md §8).

The agent does not poll the model. It wakes Qwen at the heartbeats (:mod:`.schedule`) and when the
world does something a sentiment agent should react to. This module turns perception snapshots into
those events and decides which of them may start a decision.

Events (thresholds from ``policy.triggers``, frozen at genesis):

``fear_greed_extreme``
    The crypto or the US equity-market Fear & Greed index changes band: into extreme fear
    (``<= 25``), into extreme greed (``>= 76``), back out of either, or straight across. Bands are
    Bitget's own (``bitget-signal/skills/sentiment-analyst/SKILL.md``). Staying inside a band never
    re-fires. The first reading on record fires only if it is already extreme.
``funding_zscore``
    Live funding z-score of a crypto leg (BTCUSDT) strictly beyond ``±2`` against the last 90
    settlements (``PositioningFeatures.funding_z_live``, computed by perception).
``open_interest_jump``
    Absolute 1-hour open-interest change strictly above the instrument's frozen threshold, the
    99th percentile of the trailing 30 days of 1-hour changes (:func:`oi_jump_thresholds`, run once
    at genesis). Both are in percent: ``100 * (OI_t / OI_{t-1h} - 1)``, the unit of
    ``PositioningFeatures.oi_change_1h_pct``. An instrument with no threshold is not evaluated.
``coordinated_cluster``
    A crowd story cluster flagged coordinated (3+ distinct sources inside 2 hours, decided by the
    crowd module) names a universe instrument. One trigger per instrument per cluster.
``earnings_event``
    A universe US equity, held or not, reports earnings within 24 hours of the snapshot. One
    trigger per instrument per earnings date.
``filing_event``
    A Form 4 or 8-K row, dated on or after the day the position was opened, for a *held* US
    equity. One trigger per filing.

Admission (:meth:`TriggerEngine.admit`) applies, in order:

1. **Duplicates** of an already processed ``trigger_id`` are refused.
2. **Heartbeats and owner triggers are always admitted**; they are not events.
3. **Weekend freeze.** An event whose instruments all follow the US session is refused while G2
   holds those legs flat (Friday 20:00 to Monday 00:00 UTC): the model could not act on it, and the
   Monday 00:00 UTC funding heartbeat reads the full book anyway. This rule is an addition to
   DESIGN.md §8, made to keep the Qwen budget for decisions that can change the book.
4. **Cooldown.** One admission per (kind, instrument) per 240 minutes; the Fear & Greed key is the
   index series rather than an instrument. At exactly 240 minutes the key is free again, as in
   freqtrade's pair locks, which hold while ``lock_end_time > now``
   (``freqtrade/persistence/pairlock_middleware.py:151``, ``protections/iprotection.py:142``).
5. **Daily cap.** At most 8 *event decisions* per UTC day. One ``admit`` call is one decision, so
   several events admitted together count once, and events admitted alongside a heartbeat or an
   owner trigger count nothing, because that decision happens anyway.

Every trigger ``admit`` receives comes back, admitted or refused with a reason, so the runtime can
log all of them and a reader can see what the agent chose not to wake up for.

**How the runtime drives it, and why restore is exact.** The engine's state is a fold over the
triggers ``admit`` has processed, and nothing else: :meth:`evaluate` and :meth:`due_heartbeats` only
read it. ``admit`` re-stamps every trigger of one call with a single admission instant, strictly
later than the previous call's, so on restart the TRIGGER events in the ledger regroup into the
exact batches that were admitted, and :meth:`restore` replays them through the same code path.
The contract with the runtime is therefore::

    engine = TriggerEngine(policy, clock, oi_thresholds=thresholds_frozen_at_genesis)
    engine.restore(every TRIGGER payload in the ledger, admitted or refused)
    each tick:
        admitted, refused = engine.admit(engine.due_heartbeats(last_tick) + engine.evaluate(...))
        append a TRIGGER event for every trigger in admitted and in refused, as returned

A condition that persists (funding still extreme, an open-interest surge still inside its hour) is
one event per cooldown window, not one per snapshot. A *different* event on the same key inside the
window (the z-score flips sign, a second coordinated cluster, a second filing, the index crossing
back) is emitted and refused by the cooldown, and so appears in the log.

Trigger ``detail`` strings contain only numbers, times and identifiers this project generated. No
crowd text, filing title or label from an outside source is copied into them, because triggers are
rendered into the prompt and are not spotlighted (red-team, win plan Move 15).
"""

import math
from bisect import bisect_left
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Final, Literal

from sentiment_agent.events.schedule import (
    as_utc,
    heartbeats_between,
    iso_z,
    validate_schedule,
    weekend_phase,
)
from sentiment_agent.hashing import content_hash
from sentiment_agent.types import (
    AssetClass,
    BookState,
    Clock,
    PerceptionSnapshot,
    Policy,
    Trigger,
    TriggerKind,
)

Band = Literal["extreme_fear", "neutral", "extreme_greed"]

UNCONDITIONAL_KINDS: Final[frozenset[TriggerKind]] = frozenset(
    {TriggerKind.HEARTBEAT_US_OPEN, TriggerKind.HEARTBEAT_FUNDING, TriggerKind.OWNER_MANUAL}
)
"""Always admitted, never cooled down, never counted against the event cap."""

FEAR_GREED_SERIES: Final[tuple[str, ...]] = ("crypto_fear_greed", "market_fear_greed")
"""The two indices, named as their ``MarketMood`` fields; ``Trigger.source`` is
``mood.<series>``."""

REFUSED_DUPLICATE: Final = "duplicate"
REFUSED_WEEKEND: Final = "weekend_freeze"
REFUSED_COOLDOWN: Final = "cooldown"
REFUSED_DAILY_CAP: Final = "daily_cap"
REFUSED_BUDGET: Final = "budget"
"""Refused by the runtime after this engine admitted it (``runtime/loop.py``): the day's token
budget could not also carry the heartbeats still to come. The engine counted it as admitted, so its
cooldown and daily slot are spent in the live run and in :meth:`TriggerEngine.restore` alike."""
"""Every refusal reason starts with one of these codes and a colon."""

HOUR: Final = timedelta(hours=1)
PAIR_TOLERANCE: Final = timedelta(minutes=5)
"""A 1-hour open-interest change pairs a reading with the one nearest to an hour before it, and
only if that one is within 5 minutes of the hour: a gap in the series is not a 1-hour change."""

_ID_HEX: Final = 16
_KIND_ORDER: Final[dict[TriggerKind, int]] = {kind: i for i, kind in enumerate(TriggerKind)}
_SAFE_LABEL: Final = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.:/@"
)


# ------------------------------------------------------------------------------------------------
# Open-interest thresholds (computed once, at genesis)
# ------------------------------------------------------------------------------------------------


def hourly_oi_changes_pct(series: Sequence[tuple[datetime, float]]) -> list[float]:
    """Every 1-hour open-interest change in ``series``, in percent, in time order.

    Each reading is paired with the reading nearest to one hour earlier, if that one lies within
    :data:`PAIR_TOLERANCE` of the hour; on a tie the earlier one. Non-finite and non-positive
    readings are dropped and a repeated timestamp keeps its last valid reading. This is the
    definition perception uses for the live ``oi_change_1h_pct``
    (``perception/features.py``, ``oi_change_pct`` with ``hours=1``), so the threshold and the
    value it is compared with measure the same thing.
    """
    points: dict[datetime, float] = {}
    for at, value in series:
        number = float(value)
        if math.isfinite(number) and number > 0:
            points[as_utc(at, name="open-interest timestamp")] = number
    times = sorted(points)
    changes: list[float] = []
    for i, at in enumerate(times):
        target = at - HOUR
        j = bisect_left(times, target, 0, i)
        nearest = min(
            (k for k in (j - 1, j) if 0 <= k < i),
            key=lambda k: abs(times[k] - target),
            default=None,
        )
        if nearest is None or abs(times[nearest] - target) > PAIR_TOLERANCE:
            continue
        changes.append(100.0 * (points[at] / points[times[nearest]] - 1.0))
    return changes


def quantile_type7(sorted_values: Sequence[float], q: float) -> float:
    """Hyndman-Fan definition 7 (linear between order statistics): numpy's and R's default."""
    if not sorted_values:
        raise ValueError("no values")
    h = (len(sorted_values) - 1) * q
    lo = math.floor(h)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (h - lo) * (sorted_values[hi] - sorted_values[lo])


def minimum_samples(quantile: float) -> int:
    """Fewest 1-hour changes that support the quantile: ``ceil(1 / (1 - q))``, 100 at q = 0.99.

    Below that, the estimate is the sample maximum under another name.
    """
    return math.ceil(round(1.0 / (1.0 - quantile), 9))


def oi_jump_thresholds(
    history: Mapping[str, Sequence[tuple[datetime, float]]], *, quantile: float
) -> dict[str, float]:
    """Per instrument, the ``quantile`` of the absolute 1-hour open-interest change, in percent.

    ``history`` is the trailing window the policy names (30 days, ``oi_jump_lookback_days``); the
    caller trims it. An instrument is left out, and so never fires, when its series yields fewer
    than :func:`minimum_samples` 1-hour changes (a daily series yields none) or when the threshold
    is zero (a series that never moves has nothing to jump against).
    """
    if not (math.isfinite(quantile) and 0.0 < quantile < 1.0):
        raise ValueError(f"quantile must lie strictly between 0 and 1, got {quantile}")
    need = minimum_samples(quantile)
    out: dict[str, float] = {}
    for symbol in sorted(history):
        magnitudes = sorted(abs(c) for c in hourly_oi_changes_pct(history[symbol]))
        if len(magnitudes) < need:
            continue
        threshold = quantile_type7(magnitudes, quantile)
        if math.isfinite(threshold) and threshold > 0:
            out[symbol] = threshold
    return out


# ------------------------------------------------------------------------------------------------
# Small pure helpers
# ------------------------------------------------------------------------------------------------


def band_of(value: float, policy: Policy) -> Band:
    """The Fear & Greed band: ``<= low`` extreme fear, ``>= high`` extreme greed, else neutral."""
    if value <= policy.triggers.fear_greed_low:
        return "extreme_fear"
    if value >= policy.triggers.fear_greed_high:
        return "extreme_greed"
    return "neutral"


def cooldown_key(trigger: Trigger) -> str | None:
    """The (kind, instrument) key the cooldown is kept on; ``None`` for heartbeats and owner.

    Fear & Greed keys on its series (``Trigger.source``), since one index moves many instruments.
    """
    if trigger.kind in UNCONDITIONAL_KINDS:
        return None
    if trigger.kind is TriggerKind.FEAR_GREED_EXTREME or not trigger.symbols:
        return f"{trigger.kind.value}|{trigger.source}"
    return f"{trigger.kind.value}|{trigger.symbols[0]}"


def _short_hash(*parts: object) -> str:
    return content_hash(list(parts))[:_ID_HEX]


def _label(text: str, limit: int = 64) -> str:
    """An identifier made safe to render: allowed characters only, bounded length."""
    kept = "".join(ch for ch in text if ch in _SAFE_LABEL)[:limit]
    return kept or "unlabelled"


def _sign(value: float) -> int:
    return 1 if value > 0 else -1


# ------------------------------------------------------------------------------------------------
# The engine
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EngineState:
    """Everything the engine remembers, in canonical order. Two engines that processed the same
    triggers have equal states, which is what restore-after-crash is tested against."""

    last_stamp: datetime | None
    bands: tuple[tuple[str, Band], ...]
    last_admitted: tuple[tuple[str, datetime], ...]
    emissions: tuple[tuple[str, datetime, int], ...]
    event_decisions: tuple[tuple[date, int], ...]
    processed: frozenset[str]


class TriggerEngine:
    """Evaluates snapshots into events, lists due heartbeats, and admits what may wake the model."""

    def __init__(self, policy: Policy, clock: Clock, *, oi_thresholds: Mapping[str, float]) -> None:
        validate_schedule(policy)
        rule = policy.triggers
        if rule.fear_greed_low >= rule.fear_greed_high:
            raise ValueError("the fear band must sit below the greed band")
        thresholds: dict[str, float] = {}
        for symbol, value in sorted(oi_thresholds.items()):
            if policy.entry(symbol) is None:
                raise ValueError(f"open-interest threshold for {symbol!r}, outside the universe")
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"open-interest threshold for {symbol} must be positive: {value}")
            thresholds[symbol] = float(value)
        self._policy = policy
        self._clock = clock
        self._oi_thresholds = thresholds
        self._cooldown = timedelta(minutes=rule.cooldown_minutes)
        universe = policy.universe
        self._crypto = tuple(u.symbol for u in universe if u.asset_class is AssetClass.CRYPTO)
        self._us_session = tuple(u.symbol for u in universe if u.asset_class.follows_us_session)
        self._equities = frozenset(
            u.symbol for u in universe if u.asset_class is AssetClass.US_EQUITY
        )
        self._reset()

    # --- state ----------------------------------------------------------------------------------

    def _reset(self) -> None:
        self._last_stamp: datetime | None = None
        self._bands: dict[str, Band] = {}
        self._last_admitted: dict[str, datetime] = {}
        self._emissions: dict[str, tuple[datetime, int]] = {}
        self._event_decisions: dict[date, int] = {}
        self._processed: set[str] = set()

    def state(self) -> EngineState:
        return EngineState(
            last_stamp=self._last_stamp,
            bands=tuple(sorted(self._bands.items())),
            last_admitted=tuple(sorted(self._last_admitted.items())),
            emissions=tuple(sorted((k, at, s) for k, (at, s) in self._emissions.items())),
            event_decisions=tuple(sorted(self._event_decisions.items())),
            processed=frozenset(self._processed),
        )

    def event_decisions_on(self, day: date) -> int:
        """Event decisions admitted on UTC ``day`` (for status lines and cards)."""
        return self._event_decisions.get(day, 0)

    @property
    def oi_thresholds(self) -> dict[str, float]:
        return dict(self._oi_thresholds)

    def restore(self, fired: Iterable[Trigger]) -> None:
        """Rebuild the state from the ledger: every TRIGGER payload, admitted or refused.

        Replaces whatever the engine held. Triggers regroup into their admission batches by
        ``fired_at`` (one instant per ``admit`` call) and are replayed in time order through the
        same code path ``admit`` uses; order within the input does not matter.
        """
        self._reset()
        batches: dict[datetime, list[Trigger]] = {}
        for trigger in fired:
            batches.setdefault(trigger.fired_at, []).append(trigger)
        for stamp in sorted(batches):
            self._apply(batches[stamp], stamp)

    # --- clock ----------------------------------------------------------------------------------

    def _now(self) -> datetime:
        return as_utc(self._clock.now(), name="clock.now()")

    def _next_stamp(self) -> datetime:
        stamp = self._now()
        if self._last_stamp is not None and stamp <= self._last_stamp:
            stamp = self._last_stamp + timedelta(microseconds=1)
        return stamp

    # --- heartbeats -----------------------------------------------------------------------------

    def due_heartbeats(self, since: datetime) -> list[Trigger]:
        """Heartbeats scheduled in ``(since, now]`` that have not been processed yet.

        Pass the previous tick, or the genesis time on the first start. Heartbeats missed while
        the process was down are all returned, so one catch-up decision sees every one of them,
        and a ``since`` earlier than needed is harmless: processed heartbeats are never offered
        again. Nothing is marked until ``admit`` processes them.
        """
        now = self._now()
        start = as_utc(since, name="since")
        if start >= now:
            return []
        return [
            t
            for t in heartbeats_between(start, now, self._policy)
            if t.trigger_id not in self._processed
        ]

    # --- evaluation -----------------------------------------------------------------------------

    def evaluate(self, snapshot: PerceptionSnapshot, book: BookState) -> list[Trigger]:
        """The events ``snapshot`` shows, as candidates for :meth:`admit`. Reads state, never
        writes it. ``book`` says which equities are held (filings)."""
        if snapshot.policy_version != self._policy.version:
            raise ValueError(
                f"snapshot was taken under {snapshot.policy_version!r}, "
                f"the engine runs {self._policy.version!r}"
            )
        now = self._now()
        out: list[Trigger] = []
        seen: set[str] = set()

        def add(trigger: Trigger) -> None:
            if trigger.trigger_id not in seen:
                seen.add(trigger.trigger_id)
                out.append(trigger)

        for trigger in self._fear_greed(snapshot, now):
            add(trigger)
        for trigger in self._level_triggers(snapshot, now):
            add(trigger)
        for trigger in self._clusters(snapshot, now):
            add(trigger)
        for trigger in self._earnings(snapshot, now):
            add(trigger)
        for trigger in self._filings(snapshot, book, now):
            add(trigger)
        return out

    def _fear_greed(self, snapshot: PerceptionSnapshot, now: datetime) -> list[Trigger]:
        rule = self._policy.triggers
        mood = snapshot.mood
        words: dict[Band | None, str] = {
            "extreme_fear": "extreme fear",
            "neutral": "neutral",
            "extreme_greed": "extreme greed",
            None: "none on record",
        }
        out: list[Trigger] = []
        for series, value, symbols, name in (
            ("crypto_fear_greed", mood.crypto_fear_greed, self._crypto, "crypto"),
            ("market_fear_greed", mood.market_fear_greed, self._us_session, "US equity-market"),
        ):
            if value is None:
                continue
            band = band_of(value, self._policy)
            previous = self._bands.get(series)
            if band == previous or (previous is None and band == "neutral"):
                continue
            if band == "extreme_fear":
                threshold = rule.fear_greed_low
                verb = f"entered extreme fear (<= {threshold})"
            elif band == "extreme_greed":
                threshold = rule.fear_greed_high
                verb = f"entered extreme greed (>= {threshold})"
            else:
                crossed_fear = previous == "extreme_fear"
                threshold = rule.fear_greed_low if crossed_fear else rule.fear_greed_high
                verb = f"left {words[previous]} for neutral (crossed {threshold})"
            before = words[previous]
            agree = ""
            if series == "crypto_fear_greed" and mood.crypto_sources_agree is not None:
                agree = "; the two crypto sources " + (
                    "agree on the band" if mood.crypto_sources_agree else "disagree on the band"
                )
            out.append(
                Trigger(
                    trigger_id=f"{TriggerKind.FEAR_GREED_EXTREME.value}:{series}:"
                    + _short_hash(series, value, snapshot.snapshot_id, iso_z(now)),
                    kind=TriggerKind.FEAR_GREED_EXTREME,
                    fired_at=now,
                    symbols=symbols,
                    detail=f"{name} Fear & Greed {value} {verb}; previous band: {before}{agree}",
                    observed=float(value),
                    threshold=float(threshold),
                    source=f"mood.{series}",
                    snapshot_id=snapshot.snapshot_id,
                )
            )
        return out

    def _suppressed(self, key: str, sign: int, now: datetime) -> bool:
        """The same condition, same direction, was already processed inside the cooldown."""
        last = self._emissions.get(key)
        return last is not None and last[1] == sign and now - last[0] < self._cooldown

    def _level_triggers(self, snapshot: PerceptionSnapshot, now: datetime) -> list[Trigger]:
        rule = self._policy.triggers
        out: list[Trigger] = []
        for symbol in self._crypto:
            features = snapshot.features.get(symbol)
            z = None if features is None else features.funding_z_live
            if z is None or not math.isfinite(z) or abs(z) <= rule.funding_z_threshold:
                continue
            kind = TriggerKind.FUNDING_ZSCORE
            if self._suppressed(f"{kind.value}|{symbol}", _sign(z), now):
                continue
            rate = None if features is None else features.funding_rate_live
            rate_text = (
                "" if rate is None else f"; live funding rate {rate * 100:.4f}% per interval"
            )
            out.append(
                Trigger(
                    trigger_id=f"{kind.value}:{symbol}:"
                    + _short_hash(symbol, z, snapshot.snapshot_id, iso_z(now)),
                    kind=kind,
                    fired_at=now,
                    symbols=(symbol,),
                    detail=f"live {symbol} funding z-score {z:+.2f} over the last "
                    f"{rule.funding_z_lookback_settlements} settlements, beyond "
                    f"±{rule.funding_z_threshold:g}{rate_text}",
                    observed=z,
                    threshold=math.copysign(rule.funding_z_threshold, z),
                    source=f"features.{symbol}.funding_z_live",
                    snapshot_id=snapshot.snapshot_id,
                )
            )
        for symbol, threshold in self._oi_thresholds.items():
            features = snapshot.features.get(symbol)
            change = None if features is None else features.oi_change_1h_pct
            if change is None or not math.isfinite(change) or abs(change) <= threshold:
                continue
            kind = TriggerKind.OPEN_INTEREST_JUMP
            if self._suppressed(f"{kind.value}|{symbol}", _sign(change), now):
                continue
            out.append(
                Trigger(
                    trigger_id=f"{kind.value}:{symbol}:"
                    + _short_hash(symbol, change, snapshot.snapshot_id, iso_z(now)),
                    kind=kind,
                    fired_at=now,
                    symbols=(symbol,),
                    detail=f"{symbol} open interest changed {change:+.2f}% in 1h; frozen "
                    f"p{self._policy.triggers.oi_jump_quantile * 100:g} threshold "
                    f"±{threshold:.2f}%",
                    observed=change,
                    threshold=threshold,
                    source=f"features.{symbol}.oi_change_1h_pct",
                    snapshot_id=snapshot.snapshot_id,
                )
            )
        return out

    def _universe_symbol(self, raw: str) -> str | None:
        """A universe symbol from an instrument name or an equity's underlying ticker."""
        name = raw.strip().lstrip("$").upper()
        if self._policy.entry(name) is not None:
            return name
        perp = f"{name}USDT"
        return perp if perp in self._equities else None

    def _clusters(self, snapshot: PerceptionSnapshot, now: datetime) -> list[Trigger]:
        kind = TriggerKind.COORDINATED_CLUSTER
        out: list[Trigger] = []
        for cluster in sorted(snapshot.crowd.clusters, key=lambda c: c.cluster_id):
            if not cluster.coordinated:
                continue
            named = sorted({s for raw in cluster.symbols if (s := self._universe_symbol(raw))})
            for symbol in named:
                trigger_id = f"{kind.value}:{symbol}:" + _short_hash(symbol, cluster.cluster_id)
                if trigger_id in self._processed:
                    continue
                velocity = (
                    ""
                    if cluster.velocity_per_hour is None
                    else f", {cluster.velocity_per_hour:.1f} items/h"
                )
                out.append(
                    Trigger(
                        trigger_id=trigger_id,
                        kind=kind,
                        fired_at=now,
                        symbols=(symbol,),
                        detail=f"coordinated cluster {_label(cluster.cluster_id)} names {symbol}: "
                        f"{cluster.distinct_sources} distinct sources, {len(cluster.item_ids)} "
                        f"items, first seen {iso_z(cluster.first_seen)}, last seen "
                        f"{iso_z(cluster.last_seen)}{velocity}",
                        observed=float(cluster.distinct_sources),
                        threshold=float(self._policy.triggers.coordinated_min_sources),
                        source="crowd.clusters",
                        snapshot_id=snapshot.snapshot_id,
                    )
                )
        return out

    def _earnings(self, snapshot: PerceptionSnapshot, now: datetime) -> list[Trigger]:
        kind = TriggerKind.EARNINGS_EVENT
        lookahead = self._policy.triggers.earnings_lookahead_hours
        taken = snapshot.taken_at
        dates: dict[tuple[str, datetime], str] = {}
        for item in snapshot.calendar:
            if item.kind != "earnings" or item.symbol is None or item.at is None:
                continue
            symbol = self._universe_symbol(item.symbol)
            if symbol is not None and symbol in self._equities:
                dates.setdefault((symbol, item.at), f"calendar ({_label(item.source)})")
        for symbol in sorted(self._equities):
            features = snapshot.features.get(symbol)
            if features is not None and features.next_earnings_at is not None:
                dates.setdefault(
                    (symbol, features.next_earnings_at), f"features.{symbol}.next_earnings_at"
                )
        # Measured from the snapshot; a report already out by the time a late snapshot is
        # evaluated is not an upcoming event.
        earliest = max(taken, now)
        out: list[Trigger] = []
        for (symbol, at), origin in sorted(dates.items()):
            if not earliest <= at <= taken + timedelta(hours=lookahead):
                continue
            trigger_id = f"{kind.value}:{symbol}:{iso_z(at)}"
            if trigger_id in self._processed:
                continue
            hours = (at - taken).total_seconds() / 3600.0
            out.append(
                Trigger(
                    trigger_id=trigger_id,
                    kind=kind,
                    fired_at=now,
                    symbols=(symbol,),
                    detail=f"{symbol} reports earnings at {iso_z(at)}, {hours:.1f}h after the "
                    f"snapshot (lookahead {lookahead}h; from {origin})",
                    observed=hours,
                    threshold=float(lookahead),
                    source="calendar" if origin.startswith("calendar") else origin,
                    snapshot_id=snapshot.snapshot_id,
                )
            )
        return out

    def _filings(
        self, snapshot: PerceptionSnapshot, book: BookState, now: datetime
    ) -> list[Trigger]:
        kind = TriggerKind.FILING_EVENT
        out: list[Trigger] = []
        items = sorted(
            (i for i in snapshot.calendar if i.kind in ("form4", "filing_8k")),
            key=lambda i: (i.symbol or "", i.at or now, i.kind, i.title, i.url or "", i.source),
        )
        for item in items:
            if item.symbol is None or item.at is None:
                continue
            symbol = self._universe_symbol(item.symbol)
            if symbol is None or symbol not in self._equities:
                continue
            position = book.positions.get(symbol)
            if position is None or position.is_flat:
                continue
            if item.at.date() < position.opened_at.date():
                continue
            trigger_id = f"{kind.value}:{symbol}:" + _short_hash(
                symbol, item.kind, iso_z(item.at), item.title, item.url, item.source
            )
            if trigger_id in self._processed:
                continue
            form = "Form 4" if item.kind == "form4" else "8-K"
            out.append(
                Trigger(
                    trigger_id=trigger_id,
                    kind=kind,
                    fired_at=now,
                    symbols=(symbol,),
                    detail=f"new {form} row for held {symbol} dated {item.at.date().isoformat()} "
                    f"({_label(item.source)}); position open since {iso_z(position.opened_at)}",
                    source="calendar",
                    snapshot_id=snapshot.snapshot_id,
                )
            )
        return out

    # --- admission ------------------------------------------------------------------------------

    def admit(self, triggers: Sequence[Trigger]) -> tuple[list[Trigger], list[tuple[Trigger, str]]]:
        """Decide which triggers may start one decision. See the module docstring for the rules.

        Every trigger is returned exactly once, re-stamped with this call's admission instant:
        admitted, or refused with a reason. Log all of them; :meth:`restore` needs both.
        """
        if not triggers:
            return [], []
        stamp = self._next_stamp()
        batch = [t.model_copy(update={"fired_at": stamp}) for t in triggers]
        return self._apply(batch, stamp)

    def _canonical(self, trigger: Trigger) -> tuple[int, int, str, str]:
        return (
            0 if trigger.kind in UNCONDITIONAL_KINDS else 1,
            _KIND_ORDER[trigger.kind],
            cooldown_key(trigger) or "",
            trigger.trigger_id,
        )

    def _frozen_out(self, trigger: Trigger, stamp: datetime) -> bool:
        if not trigger.symbols or weekend_phase(stamp, self._policy) != "frozen":
            return False
        entries = [self._policy.entry(s) for s in trigger.symbols]
        return all(e is not None and e.asset_class.follows_us_session for e in entries)

    def _remember(self, trigger: Trigger) -> None:
        """The part of the state a processed trigger changes, admitted or not."""
        self._processed.add(trigger.trigger_id)
        observed = trigger.observed
        if observed is None or not math.isfinite(observed):
            return
        if trigger.kind is TriggerKind.FEAR_GREED_EXTREME:
            self._bands[trigger.source.removeprefix("mood.")] = band_of(observed, self._policy)
        elif trigger.kind in (TriggerKind.FUNDING_ZSCORE, TriggerKind.OPEN_INTEREST_JUMP):
            key = cooldown_key(trigger)
            if key is not None:
                self._emissions[key] = (trigger.fired_at, _sign(observed))

    def _apply(
        self, batch: Sequence[Trigger], stamp: datetime
    ) -> tuple[list[Trigger], list[tuple[Trigger, str]]]:
        """Process one admission batch. Shared by :meth:`admit` and :meth:`restore`."""
        rule = self._policy.triggers
        admitted: list[Trigger] = []
        refused: list[tuple[Trigger, str]] = []
        fresh: list[Trigger] = []
        in_batch: set[str] = set()
        for trigger in sorted(batch, key=self._canonical):
            if trigger.trigger_id in self._processed or trigger.trigger_id in in_batch:
                refused.append(
                    (trigger, f"{REFUSED_DUPLICATE}: {trigger.trigger_id} was already processed")
                )
            else:
                in_batch.add(trigger.trigger_id)
                fresh.append(trigger)
        carried = any(t.kind in UNCONDITIONAL_KINDS for t in fresh)
        day = stamp.date()
        used = self._event_decisions.get(day, 0)
        room = carried or used < rule.max_event_decisions_per_day
        took_event = False
        for trigger in fresh:
            self._remember(trigger)
            if trigger.kind in UNCONDITIONAL_KINDS:
                admitted.append(trigger)
                continue
            key = cooldown_key(trigger) or trigger.trigger_id
            last = self._last_admitted.get(key)
            if self._frozen_out(trigger, stamp):
                refused.append(
                    (
                        trigger,
                        f"{REFUSED_WEEKEND}: every instrument named follows the US session and "
                        "the weekend freeze holds those legs flat",
                    )
                )
            elif last is not None and stamp - last < self._cooldown:
                refused.append(
                    (
                        trigger,
                        f"{REFUSED_COOLDOWN}: {key} was admitted at {iso_z(last)}; free again at "
                        f"{iso_z(last + self._cooldown)} ({rule.cooldown_minutes} min per key)",
                    )
                )
            elif not room:
                refused.append(
                    (
                        trigger,
                        f"{REFUSED_DAILY_CAP}: {used} of {rule.max_event_decisions_per_day} event "
                        f"decisions already taken on {day.isoformat()} UTC",
                    )
                )
            else:
                admitted.append(trigger)
                self._last_admitted[key] = stamp
                took_event = True
        if took_event and not carried:
            self._event_decisions[day] = used + 1
        if self._last_stamp is None or stamp > self._last_stamp:
            self._last_stamp = stamp
        return admitted, refused


__all__ = [
    "FEAR_GREED_SERIES",
    "HOUR",
    "PAIR_TOLERANCE",
    "REFUSED_BUDGET",
    "REFUSED_COOLDOWN",
    "REFUSED_DAILY_CAP",
    "REFUSED_DUPLICATE",
    "REFUSED_WEEKEND",
    "UNCONDITIONAL_KINDS",
    "Band",
    "EngineState",
    "TriggerEngine",
    "band_of",
    "cooldown_key",
    "hourly_oi_changes_pct",
    "minimum_samples",
    "oi_jump_thresholds",
    "quantile_type7",
]
