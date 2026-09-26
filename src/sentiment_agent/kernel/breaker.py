# Ported from ARGUS ``src/argus/risk/circuit.py`` (MIT, same author) at commit
# 3dec6baf9dfa7be37c7b452e26a9b139df252f75, sha256
# 365b7ad90197a313227761d462e7a854170e7350d72957a806ee62cc8fac4d85. The ARGUS repository is
# private, so its licence and every pin are recorded in third_party/argus/PROVENANCE.md.
# Taken: the three-state activation with HALTED as the safe end and unreadable state resolving to
# it, evaluate-every-rule (no short-circuit) assessment, the losing-streak and stale-evidence trips,
# and stepwise recovery out of a halt. Changed: thresholds come from ``policy.breaker`` instead of
# module constants; the drawdown ladder's risk multiplier is dropped (the kernel's ceilings do that
# job); the daily-kill and model-outage halts carry the recovery exceptions DESIGN.md §10.6 states;
# state is persisted as typed ``BreakerTransition`` ledger events and rebuilt from them.
"""The circuit breaker: the state the kernel carries between decisions.

A guard judges one ruling; the breaker judges the book over time. It is what lets a daily kill stay
in force after equity bounces, and what stops an outage from being forgotten by the next 60-second
protective check.

**Three states.** ``ACTIVE`` trades normally; ``REDUCE_ONLY`` may shrink or close but never add;
``HALTED`` is flat and may not open (the kernel's G10 forces every position out). Pattern:
``serenity-guardrails`` ``etoro_trading/guards.py:26-60`` (Apache-2.0), via ARGUS.

**Trips** (every rule is evaluated; none short-circuits), thresholds from ``policy.breaker`` and
``policy.daily_kill_pct``:

* ``drawdown_halt``: equity at least 4% below its peak -> HALTED.
* ``daily_kill``: equity at least 1.5% below the 00:00 UTC equity -> HALTED until the next UTC day.
* ``llm_outage``: the model failed to produce a valid decision -> HALTED until a valid decision.
* ``equity_nonpositive``: no equity to trade -> HALTED.
* ``drawdown_reduce_only``: at least 2.5% below peak -> REDUCE_ONLY.
* ``losing_streak``: 4 losing closed trades in a row -> REDUCE_ONLY.
* ``stale_snapshot``: no perception snapshot, or one older than 15 minutes -> REDUCE_ONLY. The only
  *transient* trip: it lifts as soon as a fresh snapshot arrives, because its cause is then known to
  be gone.

**Recovery** is stepwise (DESIGN.md §10.6). Escalation is immediate. From HALTED the breaker steps
to REDUCE_ONLY, never straight to ACTIVE, with one exception: a halt whose only cause was a model
outage lifts straight to ACTIVE at the next valid decision. From REDUCE_ONLY it returns to ACTIVE
only after one *clean decision* (a valid model decision assessed with no trip standing), unless
the only thing that put it there was the transient ``stale_snapshot`` trip. A clean decision is
signalled by passing ``decision_id`` to :meth:`Breaker.assess`; the 60-second protective loop
passes none, so it can never lift a halt or re-arm the book on its own.
"""

from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta
from typing import Final

from sentiment_agent.types import (
    Activation,
    BookState,
    BreakerState,
    BreakerTransition,
    Clock,
    KernelInputs,
    Policy,
)

DRAWDOWN_HALT: Final = "drawdown_halt"
DAILY_KILL: Final = "daily_kill"
LLM_OUTAGE: Final = "llm_outage"
EQUITY_NONPOSITIVE: Final = "equity_nonpositive"
DRAWDOWN_REDUCE_ONLY: Final = "drawdown_reduce_only"
LOSING_STREAK: Final = "losing_streak"
STALE_SNAPSHOT: Final = "stale_snapshot"
AWAITING_CLEAN_DECISION: Final = "awaiting_clean_decision"
UNREADABLE_STATE: Final = "unreadable_state"

TRANSIENT_TRIPS: Final[frozenset[str]] = frozenset({STALE_SNAPSHOT})
"""Trips that lift without a clean decision once their condition clears."""

_SEVERITY: Final[dict[Activation, int]] = {
    Activation.ACTIVE: 0,
    Activation.REDUCE_ONLY: 1,
    Activation.HALTED: 2,
}

_LEGAL: Final[dict[Activation, frozenset[Activation]]] = {
    Activation.ACTIVE: frozenset({Activation.ACTIVE, Activation.REDUCE_ONLY, Activation.HALTED}),
    Activation.REDUCE_ONLY: frozenset(
        {Activation.ACTIVE, Activation.REDUCE_ONLY, Activation.HALTED}
    ),
    Activation.HALTED: frozenset({Activation.HALTED, Activation.REDUCE_ONLY, Activation.ACTIVE}),
}
"""Every move is legal except HALTED -> ACTIVE, which is legal only out of an outage-only halt
(checked separately, because it depends on the trips of the halt)."""


def most_severe(*states: Activation) -> Activation:
    """The most restrictive of ``states`` (HALTED > REDUCE_ONLY > ACTIVE)."""
    return max(states, key=lambda s: _SEVERITY[s])


def next_utc_midnight(at: datetime) -> datetime:
    return at.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)


def book_conditions(
    book: BookState, *, llm_outage: bool, policy: Policy, include_daily_kill: bool = True
) -> tuple[Activation, tuple[str, ...]]:
    """The activation the book's own condition demands right now, and the trips behind it.

    Pure, so the kernel's G10 can recompute it on every ruling as defence in depth: a breaker the
    runtime forgot to assess still cannot let a book in a 4% drawdown open anything."""
    rule = policy.breaker
    halts: list[str] = []
    reduces: list[str] = []
    if book.equity <= 0:
        halts.append(EQUITY_NONPOSITIVE)
    drawdown = book.drawdown
    if drawdown <= -rule.halt_drawdown:
        halts.append(DRAWDOWN_HALT)
    elif drawdown <= -rule.reduce_only_drawdown:
        reduces.append(DRAWDOWN_REDUCE_ONLY)
    if (
        include_daily_kill
        and book.day_open_equity > 0
        and book.day_return <= -policy.daily_kill_pct
    ):
        halts.append(DAILY_KILL)
    if llm_outage:
        halts.append(LLM_OUTAGE)
    if book.consecutive_losses >= rule.losing_streak_reduce_only and not _streak_lapsed(
        book, rule.losing_streak_cooloff_hours
    ):
        reduces.append(LOSING_STREAK)
    if halts:
        return Activation.HALTED, (*halts, *reduces)
    if reduces:
        return Activation.REDUCE_ONLY, tuple(reduces)
    return Activation.ACTIVE, ()


def _streak_lapsed(book: BookState, cooloff_hours: int | None) -> bool:
    """Whether a losing streak's trip has lapsed (run2-a5): the policy sets a cool-off and the last
    losing close is older than it. Measured on the book's own clock, ``book.as_of``."""
    if cooloff_hours is None or book.last_loss_at is None:
        return False
    return book.as_of - book.last_loss_at >= timedelta(hours=cooloff_hours)


def snapshot_stale(inputs: KernelInputs, *, now: datetime, policy: Policy) -> bool:
    """No perception snapshot, or one older than ``snapshot_max_age_minutes`` (either direction)."""
    taken = inputs.snapshot_taken_at
    if taken is None:
        return True
    limit = timedelta(minutes=policy.breaker.snapshot_max_age_minutes)
    return abs(now - taken) > limit


def _needs_clean_decision(trips: Iterable[str]) -> bool:
    return any(t not in TRANSIENT_TRIPS for t in trips)


def _ordered(trips: Iterable[str]) -> tuple[str, ...]:
    """Trips without duplicates, in first-seen order."""
    return tuple(dict.fromkeys(trips))


class Breaker:
    """Carries activation between rulings. Every change is returned as a ``BreakerTransition`` for
    the ledger, and :meth:`restore` rebuilds the same state from those events after a restart."""

    def __init__(self, policy: Policy, clock: Clock, state: BreakerState | None = None) -> None:
        self._policy = policy
        self._clock = clock
        self._state = state or BreakerState(
            activation=Activation.ACTIVE, since=clock.now(), trips=()
        )

    def state(self) -> BreakerState:
        return self._state

    def assess(
        self,
        book: BookState,
        *,
        inputs: KernelInputs,
        llm_outage: bool,
        decision_id: str | None = None,
    ) -> tuple[BreakerState, BreakerTransition | None]:
        """Judge the book and move the state legally.

        ``decision_id`` names a valid (``DECIDED``) model decision this assessment follows; pass it
        from the decision cycle only. It is what lifts an outage halt and what counts as the clean
        decision that returns REDUCE_ONLY to ACTIVE. The protective loop passes ``None``.

        Returns the new state and the transition to log, or ``None`` when neither the activation
        nor the standing trips changed.
        """
        now = self._clock.now()
        before = self._state
        demanded, trips = book_conditions(book, llm_outage=llm_outage, policy=self._policy)
        if snapshot_stale(inputs, now=now, policy=self._policy):
            trips = (*trips, STALE_SNAPSHOT)
            demanded = most_severe(demanded, Activation.REDUCE_ONLY)
        valid_decision = decision_id is not None and not llm_outage

        halted_until = before.halted_until if DAILY_KILL in before.trips else None
        if DAILY_KILL in trips:
            fresh = next_utc_midnight(now)
            halted_until = fresh if halted_until is None else max(halted_until, fresh)

        # Latches: a halt cause that must outlive its condition.
        if before.activation is Activation.HALTED:
            latched: list[str] = []
            if DAILY_KILL in before.trips and halted_until is not None and now < halted_until:
                latched.append(DAILY_KILL)
            if LLM_OUTAGE in before.trips and not valid_decision:
                latched.append(LLM_OUTAGE)
            if latched:
                demanded = Activation.HALTED
                trips = _ordered((*latched, *trips))
        if DAILY_KILL not in trips:
            halted_until = None

        target, trips = self._target(before, demanded, trips, valid_decision=valid_decision)
        after = BreakerState(
            activation=target,
            since=before.since if target is before.activation else now,
            trips=trips,
            halted_until=halted_until if target is Activation.HALTED else None,
        )
        self._state = after
        if after.activation is before.activation and after.trips == before.trips:
            return after, None
        return after, BreakerTransition(
            at=now, from_state=before.activation, to_state=after.activation, trips=after.trips
        )

    @staticmethod
    def _target(
        before: BreakerState,
        demanded: Activation,
        trips: Sequence[str],
        *,
        valid_decision: bool,
    ) -> tuple[Activation, tuple[str, ...]]:
        current = before.activation
        if _SEVERITY[demanded] >= _SEVERITY[current]:
            # Reduce-only for a transient reason only: keep any clean decision already owed.
            if (
                demanded is Activation.REDUCE_ONLY
                and not _needs_clean_decision(trips)
                and _needs_clean_decision(before.trips)
            ):
                return demanded, _ordered((*trips, AWAITING_CLEAN_DECISION))
            return demanded, tuple(trips)
        # Recovery: the book demands less than the current state.
        if current is Activation.HALTED:
            outage_only = set(before.trips) <= {LLM_OUTAGE}
            if outage_only and valid_decision:
                return demanded, tuple(trips)
            return Activation.REDUCE_ONLY, _ordered((*trips, AWAITING_CLEAN_DECISION))
        # current is REDUCE_ONLY and demanded is ACTIVE
        if _needs_clean_decision(before.trips) and not valid_decision:
            return Activation.REDUCE_ONLY, (AWAITING_CLEAN_DECISION,)
        return Activation.ACTIVE, ()

    def restore(self, transitions: Iterable[BreakerTransition]) -> None:
        """Rebuild the state from logged transitions, in order.

        Any inconsistency (a transition that does not start where the previous one ended, time
        running backwards, or an illegal move) means the state cannot be trusted, and it resolves to
        HALTED with the ``unreadable_state`` trip: the safe end, as ARGUS ``risk/circuit.py`` and
        ``serenity-guardrails`` both specify."""
        state = self._state
        last_at: datetime | None = None
        for t in transitions:
            problem = self._replay_problem(state, t, last_at)
            if problem is not None:
                self._state = BreakerState(
                    activation=Activation.HALTED,
                    since=t.at if last_at is None else max(t.at, last_at),
                    trips=(UNREADABLE_STATE,),
                )
                return
            last_at = t.at
            halted_until: datetime | None = None
            if t.to_state is Activation.HALTED and DAILY_KILL in t.trips:
                carried = state.activation is Activation.HALTED and DAILY_KILL in state.trips
                halted_until = (
                    state.halted_until
                    if carried and state.halted_until is not None
                    else next_utc_midnight(t.at)
                )
            state = BreakerState(
                activation=t.to_state,
                since=min(state.since, t.at) if t.to_state is state.activation else t.at,
                trips=t.trips,
                halted_until=halted_until,
            )
        self._state = state

    @staticmethod
    def _replay_problem(
        state: BreakerState, t: BreakerTransition, last_at: datetime | None
    ) -> str | None:
        if t.from_state is not state.activation:
            return f"transition starts at {t.from_state}, state is {state.activation}"
        if last_at is not None and t.at < last_at:
            return "transitions out of time order"
        if t.to_state not in _LEGAL[t.from_state]:
            return f"{t.from_state} -> {t.to_state} is not a legal move"
        if (
            t.from_state is Activation.HALTED
            and t.to_state is Activation.ACTIVE
            and not set(state.trips) <= {LLM_OUTAGE}
        ):
            return "HALTED -> ACTIVE is legal only out of an outage-only halt"
        return None


__all__ = [
    "AWAITING_CLEAN_DECISION",
    "DAILY_KILL",
    "DRAWDOWN_HALT",
    "DRAWDOWN_REDUCE_ONLY",
    "EQUITY_NONPOSITIVE",
    "LLM_OUTAGE",
    "LOSING_STREAK",
    "STALE_SNAPSHOT",
    "TRANSIENT_TRIPS",
    "UNREADABLE_STATE",
    "Breaker",
    "book_conditions",
    "most_severe",
    "next_utc_midnight",
    "snapshot_stale",
]
