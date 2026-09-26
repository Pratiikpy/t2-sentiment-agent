"""Trigger replay: what a trigger policy would have woken the model for on a recorded run.

Run 2 changes which snapshot each trigger kind is evaluated on (``run2-d1``) and extends the funding
z-score trigger to the whole universe (``run2-a1``). Both are declared in run 2's genesis with the
number of event decisions they would have admitted on run 1's own snapshots, and that number comes
from here, not from an estimate: the logged snapshots are fed, in the order they were taken, through
a fresh :class:`~sentiment_agent.events.triggers.TriggerEngine` exactly as the runtime drives it
(``runtime/loop.py``), with the heartbeats that fell due between them.

How the replay follows the runtime, and where it cannot:

* A light snapshot is evaluated for :data:`~sentiment_agent.events.triggers.LIGHT_SNAPSHOT_KINDS`,
  a full one for :data:`~sentiment_agent.events.triggers.FULL_SNAPSHOT_KINDS`, each together with
  the heartbeats due since the previous snapshot, as one admission batch. A light snapshot followed
  within :data:`SAME_TICK` by a full one was one tick of the loop, and is one batch here, as the
  runtime now admits it (the full snapshot's events join that tick's decision). Full snapshots
  exist in a recorded run only where it decided, so a run-1 replay sees full-snapshot kinds only at
  run 1's own decisions; that is a property of the input.
* Every trigger is re-stamped at its snapshot's time (a :class:`~sentiment_agent.clock.ManualClock`
  set to ``taken_at``), so cooldowns and the daily cap are measured on the recorded clock.
* The token budget is not the engine's rule but the runtime's (``RunLoop._budget_short``), and it
  depends on what each decision actually spent, which a replay does not know. The result therefore
  gives the engine's count and, beside it, the count that also passes the budget rule when every
  decision spends its worst-case bound: the second is a floor, the first a ceiling.
* The anchor-mark refusal (no decision before the first MARK) is not modelled: a recorded run
  already has its anchor mark before its first snapshot.
"""

import dataclasses
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Final

from sentiment_agent.clock import ManualClock
from sentiment_agent.events.schedule import heartbeats_between, iso_z
from sentiment_agent.events.triggers import (
    FULL_SNAPSHOT_KINDS,
    LIGHT_SNAPSHOT_KINDS,
    UNCONDITIONAL_KINDS,
    TriggerEngine,
)
from sentiment_agent.kernel.breaker import next_utc_midnight
from sentiment_agent.perception.snapshot import is_light
from sentiment_agent.types import BookState, PerceptionSnapshot, Policy, Trigger, TriggerKind

SAME_TICK: Final = timedelta(minutes=2)
"""A full snapshot this soon after a light one was taken in the same tick of the loop."""


@dataclass(frozen=True)
class ReplayBatch:
    """One admission batch of the replay: a snapshot, and the heartbeats due before it."""

    at: datetime
    snapshot_id: str
    light: bool
    admitted: tuple[Trigger, ...]
    refused: tuple[tuple[Trigger, str], ...]
    budget_refused: bool = False
    members: int = 1
    """Snapshots in this tick (2 for a light snapshot and the full one taken in the same tick)."""

    @property
    def decision(self) -> bool:
        return bool(self.admitted)

    @property
    def event_decision(self) -> bool:
        """A decision woken by events alone (the ones the daily cap and the budget count)."""
        return self.decision and not any(t.kind in UNCONDITIONAL_KINDS for t in self.admitted)


@dataclass
class ReplayResult:
    """Every batch of one replay, and the counts the genesis declares."""

    policy_version: str
    batches: list[ReplayBatch] = field(default_factory=list)

    @property
    def snapshots(self) -> int:
        return sum(b.members for b in self.batches)

    def event_decisions(self, *, with_budget: bool = False) -> int:
        return sum(
            1 for b in self.batches if b.event_decision and not (with_budget and b.budget_refused)
        )

    def heartbeat_decisions(self) -> int:
        return sum(1 for b in self.batches if b.decision and not b.event_decision)

    def admitted_by_kind(self) -> dict[str, int]:
        counts = Counter(t.kind.value for b in self.batches for t in b.admitted)
        return dict(sorted(counts.items()))

    def refused_by_reason(self) -> dict[str, int]:
        counts = Counter(
            f"{t.kind.value}:{why.split(':', 1)[0]}" for b in self.batches for t, why in b.refused
        )
        return dict(sorted(counts.items()))

    def event_decisions_by_day(self) -> dict[str, int]:
        counts = Counter(b.at.date().isoformat() for b in self.batches if b.event_decision)
        return dict(sorted(counts.items()))

    def summary(self) -> dict[str, Any]:
        """The figures a genesis or a validation file records, JSON-ready."""
        first = self.batches[0].at if self.batches else None
        last = self.batches[-1].at if self.batches else None
        return {
            "policy_version": self.policy_version,
            "snapshots": self.snapshots,
            "ticks": len(self.batches),
            "full_snapshots": sum(1 for b in self.batches if not b.light),
            "first_snapshot": iso_z(first) if first else None,
            "last_snapshot": iso_z(last) if last else None,
            "hours": round((last - first).total_seconds() / 3600.0, 2) if first and last else 0,
            "event_decisions": self.event_decisions(),
            "event_decisions_within_worst_case_budget": self.event_decisions(with_budget=True),
            "event_decisions_by_day": self.event_decisions_by_day(),
            "heartbeat_decisions": self.heartbeat_decisions(),
            "admitted_by_kind": self.admitted_by_kind(),
            "refused_by_kind_and_reason": self.refused_by_reason(),
            "event_decision_times": [
                {
                    "at": iso_z(b.at),
                    "triggers": [f"{t.kind.value}:{','.join(t.symbols)}" for t in b.admitted],
                }
                for b in self.batches
                if b.event_decision
            ],
        }


def replay_admissions(
    snapshots: Sequence[PerceptionSnapshot],
    *,
    policy: Policy,
    oi_thresholds: Mapping[str, float],
    start: datetime,
    book_at: Callable[[PerceptionSnapshot], BookState],
    decision_bound_tokens: int | None = None,
) -> ReplayResult:
    """Replay ``snapshots`` (in time order) through a fresh engine under ``policy``.

    ``start`` is when the run began (its genesis): heartbeats are due from then on. ``book_at``
    gives the book the runtime held at a snapshot (it decides which equities a filing concerns).
    ``decision_bound_tokens`` is the worst-case spend of one decision
    (:func:`~sentiment_agent.llm.budget.decision_bound`); with it, each event decision is also
    checked against ``policy.decision.daily_token_cap`` as the runtime does, every earlier decision
    that day counted at that bound.
    """
    ordered = sorted(snapshots, key=lambda s: s.taken_at)
    clock = ManualClock(ordered[0].taken_at if ordered else start)
    engine = TriggerEngine(policy, clock, oi_thresholds=oi_thresholds)
    result = ReplayResult(policy_version=policy.version)
    since = start
    spent: Counter[str] = Counter()
    for tick in _ticks(ordered):
        snapshot = tick[-1]
        clock.set(snapshot.taken_at)
        light = is_light(snapshot)
        candidates = list(engine.due_heartbeats(since))
        for member in tick:
            kinds = LIGHT_SNAPSHOT_KINDS if is_light(member) else FULL_SNAPSHOT_KINDS
            if member.policy_version != policy.version:
                member = member.model_copy(update={"policy_version": policy.version})
            known = {t.trigger_id for t in candidates}
            candidates.extend(
                t
                for t in engine.evaluate(member, book_at(member), kinds=kinds)
                if t.trigger_id not in known
            )
        since = snapshot.taken_at
        if not candidates:
            result.batches.append(
                ReplayBatch(
                    at=snapshot.taken_at,
                    snapshot_id=snapshot.snapshot_id,
                    light=light,
                    admitted=(),
                    refused=(),
                    members=len(tick),
                )
            )
            continue
        admitted, refused = engine.admit(candidates)
        batch = ReplayBatch(
            at=snapshot.taken_at,
            snapshot_id=snapshot.snapshot_id,
            light=light,
            admitted=tuple(admitted),
            refused=tuple(refused),
            members=len(tick),
        )
        if decision_bound_tokens is not None and batch.decision:
            day = snapshot.taken_at.date().isoformat()
            if batch.event_decision:
                midnight = next_utc_midnight(snapshot.taken_at)
                ahead = [
                    t
                    for t in heartbeats_between(snapshot.taken_at, midnight, policy)
                    if t.fired_at < midnight
                ]
                remaining = policy.decision.daily_token_cap - spent[day]
                if remaining < (1 + len(ahead)) * decision_bound_tokens:
                    batch = dataclasses.replace(batch, budget_refused=True)
            if not batch.budget_refused:
                spent[day] += decision_bound_tokens
        result.batches.append(batch)
    return result


def _ticks(ordered: Sequence[PerceptionSnapshot]) -> list[list[PerceptionSnapshot]]:
    """Snapshots grouped into loop ticks: a light snapshot and a full one taken within
    :data:`SAME_TICK` after it are one tick; every other snapshot is a tick of its own."""
    ticks: list[list[PerceptionSnapshot]] = []
    for snapshot in ordered:
        last = ticks[-1] if ticks else None
        if (
            last is not None
            and len(last) == 1
            and is_light(last[0])
            and not is_light(snapshot)
            and snapshot.taken_at - last[0].taken_at <= SAME_TICK
        ):
            last.append(snapshot)
        else:
            ticks.append([snapshot])
    return ticks


def funding_extremes(
    snapshots: Sequence[PerceptionSnapshot], *, threshold: float, min_abs_rate: float = 0.0
) -> dict[str, int]:
    """Per instrument, the snapshots whose live funding z-score lies strictly beyond the
    threshold (and, with ``min_abs_rate``, whose live rate is at least that level): the raw
    condition before any cooldown, cap or weekend rule."""
    counts: Counter[str] = Counter()
    for snapshot in snapshots:
        for symbol, features in snapshot.features.items():
            z = features.funding_z_live
            if z is None or abs(z) <= threshold:
                continue
            rate = features.funding_rate_live
            if min_abs_rate and (rate is None or abs(rate) < min_abs_rate):
                continue
            counts[symbol] += 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def funding_level_profile(
    snapshots: Sequence[PerceptionSnapshot], *, threshold: float, floors: Sequence[float]
) -> dict[str, Any]:
    """How often the live funding z-score leaves ``threshold`` across every instrument-snapshot
    that carries one, the distribution of the absolute live rate there, and how often the z-score
    and a level of at least each floor hold together. The basis of policy v2's level floor
    (``run2-a2``): a z-score computed on a series that sits at exactly zero scores a single tick as
    an extreme."""
    rates: list[float] = []
    beyond = 0
    both: Counter[float] = Counter()
    for snapshot in snapshots:
        for features in snapshot.features.values():
            z, rate = features.funding_z_live, features.funding_rate_live
            if z is None or rate is None:
                continue
            rates.append(abs(rate))
            if abs(z) <= threshold:
                continue
            beyond += 1
            for floor in floors:
                if abs(rate) >= floor:
                    both[floor] += 1
    if not rates:
        raise ValueError("no instrument-snapshot carries a live funding z-score and rate")
    ordered = sorted(rates)

    def quantile(q: float) -> float:
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    n = len(ordered)
    return {
        "instrument_snapshots": n,
        "beyond_threshold": beyond,
        "beyond_threshold_share": round(beyond / n, 4),
        "abs_rate_median": quantile(0.5),
        "abs_rate_p95": quantile(0.95),
        "abs_rate_zero_share": round(sum(1 for r in ordered if r == 0) / n, 4),
        "beyond_with_level_share_by_floor": {
            f"{floor:g}": round(both[floor] / n, 4) for floor in floors
        },
    }


def kinds_of(batches: Sequence[ReplayBatch]) -> frozenset[TriggerKind]:
    """Every kind admitted anywhere in ``batches``."""
    return frozenset(t.kind for b in batches for t in b.admitted)


__all__ = [
    "SAME_TICK",
    "ReplayBatch",
    "ReplayResult",
    "funding_extremes",
    "funding_level_profile",
    "kinds_of",
    "replay_admissions",
]
