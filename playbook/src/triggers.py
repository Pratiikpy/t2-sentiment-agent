"""What wakes the model: two heartbeats and six kinds of event, with the primary's admission rules.

A port of ``sentiment_agent.events.schedule`` and ``events.triggers`` (DESIGN.md §8). The model is
not polled: the Playbook runs every ``CRON_MINUTES`` for the protective checks, and a model decision
is made only when a trigger is admitted.

* **Heartbeats**: 00:00, 08:00 and 16:00 UTC (live BTCUSDT funding settlements) and the US cash
  open, 09:30 America/New_York on weekdays. Always admitted. A run admits every heartbeat scheduled
  since the previous run's check (at most ``HEARTBEAT_CATCH_UP`` back), so a skipped run is caught
  up once and a heartbeat is never taken twice.
* **Events**: a Fear & Greed band change (crypto or US equity market; extreme bands from the
  policy; staying inside a band never re-fires; the first reading fires only if already extreme),
  the funding z-score beyond the threshold for every instrument whose asset class is in
  ``TRIGGER_FUNDING_Z_ASSET_CLASSES`` (BTCUSDT alone under policy v1), a 1-hour open-interest
  change beyond the
  trailing 99th percentile, a coordinated story cluster naming an instrument, an earnings report
  inside the lookahead, and a new insider filing for a held equity.
* **Admission**, in order: duplicates refused; heartbeats admitted; an event whose instruments all
  follow the US session refused during the weekend freeze; one admission per key per cooldown; at
  most ``TRIGGER_MAX_EVENT_DECISIONS_PER_DAY`` event decisions per UTC day, counting nothing when a
  heartbeat carries the decision anyway.

The replica's addition: **without persisted state** (``.state/`` not hydrated) the cooldown and
the daily cap cannot be enforced across runs, so every event is refused with that reason and only
heartbeats (bounded by the calendar itself) can wake the model. Trigger details carry only numbers,
times and identifiers this package generated, never third-party text.
"""

import math
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from . import policy_v1 as policy
from .perception import Snapshot
from .sessions import (
    HEARTBEAT_FUNDING,
    HEARTBEAT_US_OPEN,
    follows_us_session,
    heartbeat_id,
    heartbeats_between,
    iso_z,
    parse_iso,
    utc_day,
    weekend_phase,
)

FEAR_GREED = "fear_greed_extreme"
FUNDING_Z = "funding_zscore"
OI_JUMP = "open_interest_jump"
COORDINATED = "coordinated_cluster"
EARNINGS = "earnings_event"
FILING = "filing_event"

HEARTBEATS = frozenset({HEARTBEAT_US_OPEN, HEARTBEAT_FUNDING})
KIND_ORDER = (
    HEARTBEAT_US_OPEN,
    HEARTBEAT_FUNDING,
    FEAR_GREED,
    FUNDING_Z,
    OI_JUMP,
    COORDINATED,
    EARNINGS,
    FILING,
)

HEARTBEAT_CATCH_UP = timedelta(hours=1)
PROCESSED_KEPT = 400
"""Processed trigger ids kept in the state; older ones cannot recur (heartbeats and dated events
only move forward)."""

_ASSET_CLASS = {symbol: asset for symbol, asset, _ in policy.UNIVERSE}
_CRYPTO = tuple(s for s, a, _ in policy.UNIVERSE if a == "crypto")
_FUNDING_SCOPE = tuple(
    s for s, a, _ in policy.UNIVERSE if a in policy.TRIGGER_FUNDING_Z_ASSET_CLASSES
)
_US_SESSION = tuple(s for s, a, _ in policy.UNIVERSE if a in policy.US_SESSION_ASSET_CLASSES)
_EQUITIES = tuple(s for s, a, _ in policy.UNIVERSE if a == "us_equity")


@dataclass(frozen=True)
class Trigger:
    trigger_id: str
    kind: str
    fired_at: datetime
    symbols: tuple[str, ...]
    detail: str
    source: str
    observed: float | None = None
    threshold: float | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "trigger_id": self.trigger_id,
            "kind": self.kind,
            "fired_at": iso_z(self.fired_at),
            "symbols": list(self.symbols),
            "detail": self.detail,
            "source": self.source,
            "observed": self.observed,
            "threshold": self.threshold,
        }


def band_of(value: float) -> str:
    if value <= policy.TRIGGER_FEAR_GREED_LOW:
        return "extreme_fear"
    if value >= policy.TRIGGER_FEAR_GREED_HIGH:
        return "extreme_greed"
    return "neutral"


def cooldown_key(trigger: Trigger) -> str | None:
    if trigger.kind in HEARTBEATS:
        return None
    if trigger.kind == FEAR_GREED or not trigger.symbols:
        return f"{trigger.kind}|{trigger.source}"
    return f"{trigger.kind}|{trigger.symbols[0]}"


def _sign(value: float) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


class TriggerBook:
    """The admission state, kept in ``.state/`` under ``triggers``. ``stateless`` is true when the
    run found no persisted state: the dict is still written back, so the next run is stateful if
    the runner hydrates ``.state/``."""

    def __init__(self, state: MutableMapping[str, object], now: datetime, *, stateless: bool):
        self.stateless = stateless
        self.now = now
        self.state = state
        self.state.setdefault("bands", {})
        self.state.setdefault("emissions", {})
        self.state.setdefault("last_admitted", {})
        self.state.setdefault("event_decisions", {})
        self.state.setdefault("processed", [])

    # --- typed views on the persisted dict -------------------------------------------------------

    def _dict(self, name: str) -> dict[str, object]:
        value = self.state.get(name)
        if not isinstance(value, dict):
            value = {}
            self.state[name] = value
        return value

    def _processed(self) -> list[str]:
        value = self.state.get("processed")
        if not isinstance(value, list):
            value = []
            self.state["processed"] = value
        return value

    def processed(self, trigger_id: str) -> bool:
        return trigger_id in self._processed()

    def _remember_id(self, trigger_id: str) -> None:
        ids = self._processed()
        if trigger_id not in ids:
            ids.append(trigger_id)
        del ids[: max(0, len(ids) - PROCESSED_KEPT)]

    # --- evaluation ------------------------------------------------------------------------------

    def due_heartbeats(self) -> list[Trigger]:
        checked = parse_iso(self.state.get("heartbeat_checked_at"))
        floor = self.now - HEARTBEAT_CATCH_UP
        window = self.now - timedelta(minutes=policy.CRON_MINUTES)
        start = window if checked is None else max(checked, floor)
        out: list[Trigger] = []
        for kind, scheduled in heartbeats_between(start, self.now):
            trigger_id = heartbeat_id(kind, scheduled)
            if self.processed(trigger_id):
                continue
            symbols = _US_SESSION if kind == HEARTBEAT_US_OPEN else _CRYPTO
            detail = (
                f"US cash open scheduled {iso_z(scheduled)}"
                if kind == HEARTBEAT_US_OPEN
                else f"live funding settlement at {scheduled:%H:%M} UTC, "
                f"scheduled {iso_z(scheduled)}"
            )
            out.append(Trigger(trigger_id, kind, self.now, symbols, detail, "schedule"))
        self.state["heartbeat_checked_at"] = iso_z(self.now)
        return out

    def events(self, snapshot: Snapshot, held_since: Mapping[str, datetime]) -> list[Trigger]:
        out: list[Trigger] = []
        out.extend(self._fear_greed(snapshot))
        out.extend(self._levels(snapshot))
        out.extend(self._clusters(snapshot))
        out.extend(self._earnings(snapshot))
        out.extend(self._filings(snapshot, held_since))
        return out

    def _fear_greed(self, snapshot: Snapshot) -> list[Trigger]:
        bands = self._dict("bands")
        out: list[Trigger] = []
        for series, symbols, name in (
            ("crypto_fear_greed", _CRYPTO, "crypto"),
            ("market_fear_greed", _US_SESSION, "US equity-market"),
        ):
            value = snapshot.mood.get(series)
            if value is None:
                continue
            band = band_of(value)
            previous = bands.get(series)
            if band == previous or (previous is None and band == "neutral"):
                continue
            if band == "extreme_fear":
                threshold = float(policy.TRIGGER_FEAR_GREED_LOW)
                verb = f"entered extreme fear (<= {policy.TRIGGER_FEAR_GREED_LOW})"
            elif band == "extreme_greed":
                threshold = float(policy.TRIGGER_FEAR_GREED_HIGH)
                verb = f"entered extreme greed (>= {policy.TRIGGER_FEAR_GREED_HIGH})"
            else:
                crossed_fear = previous == "extreme_fear"
                threshold = float(
                    policy.TRIGGER_FEAR_GREED_LOW
                    if crossed_fear
                    else policy.TRIGGER_FEAR_GREED_HIGH
                )
                verb = f"left {previous} for neutral (crossed {threshold:g})"
            out.append(
                Trigger(
                    trigger_id=f"{FEAR_GREED}:{series}:{value:g}:{iso_z(self.now)}",
                    kind=FEAR_GREED,
                    fired_at=self.now,
                    symbols=symbols,
                    detail=f"{name} Fear & Greed {value:g} {verb}; "
                    f"previous band: {previous or 'none on record'}",
                    source=f"mood.{series}",
                    observed=value,
                    threshold=threshold,
                )
            )
        return out

    def _suppressed(self, key: str, sign: int) -> bool:
        last = self._dict("emissions").get(key)
        if not isinstance(last, list) or len(last) != 2:
            return False
        at = parse_iso(last[0])
        return (
            at is not None
            and last[1] == sign
            and self.now - at < timedelta(minutes=policy.TRIGGER_COOLDOWN_MINUTES)
        )

    def _levels(self, snapshot: Snapshot) -> list[Trigger]:
        out: list[Trigger] = []
        for symbol in _FUNDING_SCOPE:
            z = snapshot.feature(symbol, "funding_z")
            if z is None or not math.isfinite(z) or abs(z) <= policy.TRIGGER_FUNDING_Z_THRESHOLD:
                continue
            if self._suppressed(f"{FUNDING_Z}|{symbol}", _sign(z)):
                continue
            out.append(
                Trigger(
                    trigger_id=f"{FUNDING_Z}:{symbol}:{z:+.4f}:{iso_z(self.now)}",
                    kind=FUNDING_Z,
                    fired_at=self.now,
                    symbols=(symbol,),
                    detail=f"live {symbol} funding z-score {z:+.2f} over the last "
                    f"{policy.TRIGGER_FUNDING_Z_LOOKBACK_SETTLEMENTS} settlements, beyond "
                    f"+/-{policy.TRIGGER_FUNDING_Z_THRESHOLD:g}",
                    source=f"features.{symbol}.funding_z",
                    observed=z,
                    threshold=math.copysign(policy.TRIGGER_FUNDING_Z_THRESHOLD, z),
                )
            )
        threshold = snapshot.oi_jump_threshold_pct
        change = snapshot.feature("BTCUSDT", "oi_change_1h_pct")
        if (
            threshold is not None
            and change is not None
            and math.isfinite(change)
            and abs(change) > threshold
            and not self._suppressed(f"{OI_JUMP}|BTCUSDT", _sign(change))
        ):
            out.append(
                Trigger(
                    trigger_id=f"{OI_JUMP}:BTCUSDT:{change:+.4f}:{iso_z(self.now)}",
                    kind=OI_JUMP,
                    fired_at=self.now,
                    symbols=("BTCUSDT",),
                    detail=f"BTCUSDT open interest changed {change:+.2f}% in 1h; trailing "
                    f"p{policy.TRIGGER_OI_JUMP_QUANTILE * 100:g} threshold +/-{threshold:.2f}%",
                    source="features.BTCUSDT.oi_change_1h_pct",
                    observed=change,
                    threshold=threshold,
                )
            )
        return out

    def _clusters(self, snapshot: Snapshot) -> list[Trigger]:
        if snapshot.crowd is None:
            return []
        out: list[Trigger] = []
        for cluster in snapshot.crowd.clusters:
            if not cluster.coordinated:
                continue
            for symbol in cluster.symbols:
                trigger_id = f"{COORDINATED}:{symbol}:{cluster.cluster_id}"
                if self.processed(trigger_id):
                    continue
                out.append(
                    Trigger(
                        trigger_id=trigger_id,
                        kind=COORDINATED,
                        fired_at=self.now,
                        symbols=(symbol,),
                        detail=f"a coordinated story ({cluster.size} copies from {cluster.sources} "
                        f"sources, first seen "
                        f"{'unknown' if cluster.first_seen is None else iso_z(cluster.first_seen)}"
                        ") "
                        f"names {symbol}",
                        source="crowd.clusters",
                        observed=float(cluster.sources),
                        threshold=float(policy.TRIGGER_COORDINATED_MIN_SOURCES),
                    )
                )
        return out

    def _earnings(self, snapshot: Snapshot) -> list[Trigger]:
        out: list[Trigger] = []
        horizon = snapshot.taken_at + timedelta(hours=policy.TRIGGER_EARNINGS_LOOKAHEAD_HOURS)
        earliest = max(snapshot.taken_at, self.now)
        for entry in sorted(snapshot.earnings, key=lambda e: (e.symbol, e.at)):
            if entry.symbol not in _EQUITIES or not earliest <= entry.at <= horizon:
                continue
            trigger_id = f"{EARNINGS}:{entry.symbol}:{iso_z(entry.at)}"
            if self.processed(trigger_id):
                continue
            hours = (entry.at - snapshot.taken_at).total_seconds() / 3600.0
            out.append(
                Trigger(
                    trigger_id=trigger_id,
                    kind=EARNINGS,
                    fired_at=self.now,
                    symbols=(entry.symbol,),
                    detail=f"{entry.symbol} reports earnings on {entry.report_date}, about "
                    f"{hours:.1f}h after the snapshot "
                    f"(lookahead {policy.TRIGGER_EARNINGS_LOOKAHEAD_HOURS}h)",
                    source="equity.calendar.earnings",
                    observed=hours,
                    threshold=float(policy.TRIGGER_EARNINGS_LOOKAHEAD_HOURS),
                )
            )
        return out

    def _filings(self, snapshot: Snapshot, held_since: Mapping[str, datetime]) -> list[Trigger]:
        out: list[Trigger] = []
        for entry in sorted(snapshot.filings, key=lambda f: (f.symbol, f.filed, f.key)):
            if entry.symbol not in held_since:
                continue
            trigger_id = f"{FILING}:{entry.symbol}:{entry.filed}:{hash_text(entry.key):08x}"
            if self.processed(trigger_id):
                continue
            out.append(
                Trigger(
                    trigger_id=trigger_id,
                    kind=FILING,
                    fired_at=self.now,
                    symbols=(entry.symbol,),
                    detail=f"new insider filing for held {entry.symbol} dated {entry.filed}; "
                    f"position open since {iso_z(held_since[entry.symbol])}",
                    source="equity.ownership.insider_trading",
                )
            )
        return out

    # --- deferral --------------------------------------------------------------------------------

    def defer(self, triggers: Sequence[Trigger]) -> None:
        """Keep admitted triggers whose decision this run could not make (out of time) for the next
        run. They were admitted once; the next run decides on them without admitting them again."""
        self.state["pending"] = [t.to_json() for t in triggers]

    def take_pending(self) -> list[Trigger]:
        """Deferred triggers, at most ``HEARTBEAT_CATCH_UP`` old; the list is cleared."""
        raw = self.state.get("pending")
        self.state["pending"] = []
        out: list[Trigger] = []
        if not isinstance(raw, list):
            return out
        for item in raw:
            if not isinstance(item, dict):
                continue
            fired = parse_iso(item.get("fired_at"))
            symbols = item.get("symbols")
            if (
                fired is None
                or self.now - fired > HEARTBEAT_CATCH_UP
                or not isinstance(symbols, list)
            ):
                continue
            observed = item.get("observed")
            threshold = item.get("threshold")
            out.append(
                Trigger(
                    trigger_id=str(item.get("trigger_id")),
                    kind=str(item.get("kind")),
                    fired_at=fired,
                    symbols=tuple(str(s) for s in symbols),
                    detail=str(item.get("detail")),
                    source=str(item.get("source")),
                    observed=float(observed) if isinstance(observed, (int, float)) else None,
                    threshold=float(threshold) if isinstance(threshold, (int, float)) else None,
                )
            )
        return out

    # --- admission -------------------------------------------------------------------------------

    def admit(self, triggers: Sequence[Trigger]) -> tuple[list[Trigger], list[tuple[Trigger, str]]]:
        admitted: list[Trigger] = []
        refused: list[tuple[Trigger, str]] = []
        fresh: list[Trigger] = []
        seen: set[str] = set()
        order = {kind: i for i, kind in enumerate(KIND_ORDER)}
        for trigger in sorted(
            triggers,
            key=lambda t: (
                0 if t.kind in HEARTBEATS else 1,
                order.get(t.kind, 99),
                cooldown_key(t) or "",
                t.trigger_id,
            ),
        ):
            if self.processed(trigger.trigger_id) or trigger.trigger_id in seen:
                refused.append((trigger, f"duplicate: {trigger.trigger_id} was already processed"))
            else:
                seen.add(trigger.trigger_id)
                fresh.append(trigger)
        carried = any(t.kind in HEARTBEATS for t in fresh)
        day = utc_day(self.now)
        counts = self._dict("event_decisions")
        used_raw = counts.get(day, 0)
        used = used_raw if isinstance(used_raw, int) else policy.TRIGGER_MAX_EVENT_DECISIONS_PER_DAY
        room = carried or used < policy.TRIGGER_MAX_EVENT_DECISIONS_PER_DAY
        last_admitted = self._dict("last_admitted")
        cooldown = timedelta(minutes=policy.TRIGGER_COOLDOWN_MINUTES)
        took_event = False
        for trigger in fresh:
            self._remember(trigger)
            if trigger.kind in HEARTBEATS:
                admitted.append(trigger)
                continue
            key = cooldown_key(trigger) or trigger.trigger_id
            last = parse_iso(last_admitted.get(key))
            if self.stateless:
                refused.append(
                    (
                        trigger,
                        "stateless: no persisted state, so the cooldown and the daily event cap "
                        "cannot be enforced; only heartbeats may wake the model",
                    )
                )
            elif self._frozen_out(trigger):
                refused.append(
                    (
                        trigger,
                        "weekend: every instrument named follows the US session and the weekend "
                        "freeze holds those legs flat",
                    )
                )
            elif last is not None and self.now - last < cooldown:
                refused.append(
                    (
                        trigger,
                        f"cooldown: {key} was admitted at {iso_z(last)}; free again at "
                        f"{iso_z(last + cooldown)} ({policy.TRIGGER_COOLDOWN_MINUTES} min per key)",
                    )
                )
            elif not room:
                refused.append(
                    (
                        trigger,
                        f"daily cap: {used} of {policy.TRIGGER_MAX_EVENT_DECISIONS_PER_DAY} event "
                        f"decisions already taken on {day} UTC",
                    )
                )
            else:
                admitted.append(trigger)
                last_admitted[key] = iso_z(self.now)
                took_event = True
        if took_event and not carried:
            counts[day] = used + 1
        for stale_day in [d for d in counts if d != day]:
            del counts[stale_day]
        return admitted, refused

    def _frozen_out(self, trigger: Trigger) -> bool:
        if not trigger.symbols or weekend_phase(self.now) != "frozen":
            return False
        return all(follows_us_session(_ASSET_CLASS.get(s)) for s in trigger.symbols)

    def _remember(self, trigger: Trigger) -> None:
        self._remember_id(trigger.trigger_id)
        observed = trigger.observed
        if observed is None or not math.isfinite(observed):
            return
        if trigger.kind == FEAR_GREED:
            self._dict("bands")[trigger.source.removeprefix("mood.")] = band_of(observed)
        elif trigger.kind in (FUNDING_Z, OI_JUMP):
            key = cooldown_key(trigger)
            if key is not None:
                self._dict("emissions")[key] = [iso_z(trigger.fired_at), _sign(observed)]


def hash_text(text: str) -> int:
    """A stable, non-cryptographic 32-bit FNV-1a of ``text``, for short trigger ids. Python's
    ``hash`` is salted per process, so it cannot name a trigger across runs, and ``hashlib`` is not
    allowed in the sandbox. A collision would merge two filings of one name on one day into one
    trigger, which wakes the model once instead of twice; no decision depends on the id."""
    value = 0x811C9DC5
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 0x01000193) & 0xFFFFFFFF
    return value
