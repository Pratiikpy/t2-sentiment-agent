"""The circuit breaker: the state the kernel carries between runs.

A port of the primary's ``sentiment_agent.kernel.breaker`` (itself ported from ARGUS
``risk/circuit.py``, MIT, same author), with the same states, trips, latches and stepwise recovery:

* ``active`` trades normally; ``reduce_only`` may shrink or close but never add; ``halted`` is flat
  and may not open. An unreadable state resolves to ``halted`` (the safe end).
* Trips, every one evaluated: ``drawdown_halt`` (equity 4% below peak), ``daily_kill`` (1.5% below
  the 00:00 UTC equity, latched until the next UTC day), ``llm_outage`` (latched until a valid
  decision), ``equity_nonpositive``, ``drawdown_reduce_only`` (2.5% below peak), ``losing_streak``
  (4 losing closed trades in a row) and the transient ``stale_snapshot``.
* Recovery: out of ``halted`` to ``reduce_only`` (straight to ``active`` only out of an
  outage-only halt at a valid decision); out of ``reduce_only`` to ``active`` after one clean
  decision, unless only the transient trip put it there.

What differs: the state is a plain dict persisted in ``.state/`` by :mod:`.book`, not a ledger of
typed transitions, because the sandbox keeps no ledger. Thresholds are the generated policy's.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from . import policy_v1 as policy
from .sessions import as_utc, iso_z, next_utc_midnight, parse_iso

ACTIVE = "active"
REDUCE_ONLY = "reduce_only"
HALTED = "halted"

DRAWDOWN_HALT = "drawdown_halt"
DAILY_KILL = "daily_kill"
LLM_OUTAGE = "llm_outage"
EQUITY_NONPOSITIVE = "equity_nonpositive"
DRAWDOWN_REDUCE_ONLY = "drawdown_reduce_only"
LOSING_STREAK = "losing_streak"
STALE_SNAPSHOT = "stale_snapshot"
AWAITING_CLEAN_DECISION = "awaiting_clean_decision"
UNREADABLE_STATE = "unreadable_state"
HISTORY_LOST = "history_lost"

TRANSIENT_TRIPS = frozenset({STALE_SNAPSHOT})
_SEVERITY = {ACTIVE: 0, REDUCE_ONLY: 1, HALTED: 2}


@dataclass(frozen=True)
class BreakerState:
    activation: str
    since: datetime
    trips: tuple[str, ...] = ()
    halted_until: datetime | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "activation": self.activation,
            "since": iso_z(self.since),
            "trips": list(self.trips),
            "halted_until": None if self.halted_until is None else iso_z(self.halted_until),
        }


@dataclass(frozen=True)
class Transition:
    at: datetime
    from_state: str
    to_state: str
    trips: tuple[str, ...] = field(default_factory=tuple)

    def to_json(self) -> dict[str, object]:
        return {
            "at": iso_z(self.at),
            "from": self.from_state,
            "to": self.to_state,
            "trips": list(self.trips),
        }


def breaker_from_json(raw: object, now: datetime) -> BreakerState:
    """A state read back from ``.state/``. Anything unreadable resolves to ``halted``."""
    if raw is None:
        return BreakerState(activation=ACTIVE, since=now)
    if not isinstance(raw, dict):
        return BreakerState(activation=HALTED, since=now, trips=(UNREADABLE_STATE,))
    activation = raw.get("activation")
    since = parse_iso(raw.get("since"))
    trips = raw.get("trips")
    halted_raw = raw.get("halted_until")
    halted_until = parse_iso(halted_raw) if halted_raw is not None else None
    if (
        activation not in _SEVERITY
        or since is None
        or not isinstance(trips, list)
        or not all(isinstance(t, str) for t in trips)
        or (halted_raw is not None and halted_until is None)
    ):
        return BreakerState(activation=HALTED, since=now, trips=(UNREADABLE_STATE,))
    return BreakerState(
        activation=str(activation), since=since, trips=tuple(trips), halted_until=halted_until
    )


def most_severe(*states: str) -> str:
    return max(states, key=lambda s: _SEVERITY[s])


def book_conditions(
    *,
    equity: float,
    drawdown: float,
    day_return: float | None,
    consecutive_losses: int,
    llm_outage: bool,
    include_daily_kill: bool = True,
) -> tuple[str, tuple[str, ...]]:
    """The activation the book's own condition demands now, and the trips behind it.

    ``day_return`` is ``None`` when the 00:00 UTC equity is not known (the daily kill cannot then be
    judged here; the kernel's G5 reports it as not evaluated)."""
    halts: list[str] = []
    reduces: list[str] = []
    if equity <= 0:
        halts.append(EQUITY_NONPOSITIVE)
    if drawdown <= -policy.BREAKER_HALT_DRAWDOWN:
        halts.append(DRAWDOWN_HALT)
    elif drawdown <= -policy.BREAKER_REDUCE_ONLY_DRAWDOWN:
        reduces.append(DRAWDOWN_REDUCE_ONLY)
    if include_daily_kill and day_return is not None and day_return <= -policy.DAILY_KILL_PCT:
        halts.append(DAILY_KILL)
    if llm_outage:
        halts.append(LLM_OUTAGE)
    if consecutive_losses >= policy.BREAKER_LOSING_STREAK_REDUCE_ONLY:
        reduces.append(LOSING_STREAK)
    if halts:
        return HALTED, (*halts, *reduces)
    if reduces:
        return REDUCE_ONLY, tuple(reduces)
    return ACTIVE, ()


def snapshot_stale(snapshot_taken_at: datetime | None, now: datetime) -> bool:
    if snapshot_taken_at is None:
        return True
    limit = timedelta(minutes=policy.BREAKER_SNAPSHOT_MAX_AGE_MINUTES)
    return abs(as_utc(now) - as_utc(snapshot_taken_at)) > limit


def _needs_clean_decision(trips: tuple[str, ...] | list[str]) -> bool:
    return any(t not in TRANSIENT_TRIPS for t in trips)


def _ordered(trips: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(trips))


def assess(
    before: BreakerState,
    *,
    now: datetime,
    equity: float,
    drawdown: float,
    day_return: float | None,
    consecutive_losses: int,
    llm_outage: bool,
    snapshot_taken_at: datetime | None,
    valid_decision: bool,
    history_lost: bool = False,
) -> tuple[BreakerState, Transition | None]:
    """Judge the book and move the state legally (``Breaker.assess`` in the primary).

    ``valid_decision`` is true only after a model decision that passed the contract this run; it is
    what lifts an outage halt and what counts as the clean decision out of ``reduce_only``.
    ``history_lost`` (the replica's addition) holds the book at ``reduce_only`` while the persisted
    history that the drawdown and daily-kill rules need is missing.
    """
    demanded, trips = book_conditions(
        equity=equity,
        drawdown=drawdown,
        day_return=day_return,
        consecutive_losses=consecutive_losses,
        llm_outage=llm_outage,
    )
    if snapshot_stale(snapshot_taken_at, now):
        trips = (*trips, STALE_SNAPSHOT)
        demanded = most_severe(demanded, REDUCE_ONLY)
    if history_lost:
        trips = (*trips, HISTORY_LOST)
        demanded = most_severe(demanded, REDUCE_ONLY)
    valid = valid_decision and not llm_outage

    halted_until = before.halted_until if DAILY_KILL in before.trips else None
    if DAILY_KILL in trips:
        fresh = next_utc_midnight(now)
        halted_until = fresh if halted_until is None else max(halted_until, fresh)

    if before.activation == HALTED:
        latched: list[str] = []
        if DAILY_KILL in before.trips and halted_until is not None and now < halted_until:
            latched.append(DAILY_KILL)
        if LLM_OUTAGE in before.trips and not valid:
            latched.append(LLM_OUTAGE)
        if latched:
            demanded = HALTED
            trips = _ordered((*latched, *trips))
    if DAILY_KILL not in trips:
        halted_until = None

    target, trips = _target(before, demanded, trips, valid_decision=valid)
    after = BreakerState(
        activation=target,
        since=before.since if target == before.activation else now,
        trips=trips,
        halted_until=halted_until if target == HALTED else None,
    )
    if after.activation == before.activation and after.trips == before.trips:
        return after, None
    return after, Transition(
        at=now, from_state=before.activation, to_state=after.activation, trips=after.trips
    )


def _target(
    before: BreakerState, demanded: str, trips: tuple[str, ...], *, valid_decision: bool
) -> tuple[str, tuple[str, ...]]:
    current = before.activation
    if _SEVERITY[demanded] >= _SEVERITY[current]:
        if (
            demanded == REDUCE_ONLY
            and not _needs_clean_decision(trips)
            and _needs_clean_decision(before.trips)
        ):
            return demanded, _ordered((*trips, AWAITING_CLEAN_DECISION))
        return demanded, tuple(trips)
    if current == HALTED:
        outage_only = set(before.trips) <= {LLM_OUTAGE}
        if outage_only and valid_decision:
            return demanded, tuple(trips)
        return REDUCE_ONLY, _ordered((*trips, AWAITING_CLEAN_DECISION))
    if _needs_clean_decision(before.trips) and not valid_decision:
        return REDUCE_ONLY, (AWAITING_CLEAN_DECISION,)
    return ACTIVE, ()
