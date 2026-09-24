"""Running the rivals: every arm on the same snapshots, shown its own book, marked by one simulator.

:func:`run_rivals` asks each arm for its targets at every recorded snapshot, in time order, and runs
the resulting schedule through the :class:`~sentiment_agent.analysis.armsim.ArmSimulator`: the
production kernel under the arm's guards (the venue guards for every rival), the production planner
and book, the same Demo mark path, spreads, fees and stops as every other arm. Every arm is marked
on the same hourly grid, from the hour of the first snapshot to the last hour the marks reach, so
the published results line up hour for hour with the agent's, the baselines' and each other's.

What an arm is shown
--------------------
A rival must decide on its own book, not ours. The harness keeps each arm's previous targets and
rebuilds from them the :class:`~sentiment_agent.types.BookState` the arm sees at the next snapshot:
its positions (quantity from its target weight at the snapshot's Demo mark, average entry from the
marks at which it opened and added), its rebalances since 00:00 UTC, and the account scale (equity)
from ``books[i]``. Nothing else of ``books[i]`` reaches the arm. The shadow book follows what the
arm *asked for*; the simulator may have filled less (a guard refused, a venue minimum skipped, a
stop fired between snapshots). Those differences are the kernel's and the venue's, and they are
visible in the arm's result, not hidden from it.

The proposal each schedule entry carries names every symbol the arm has ever held, with an explicit
zero for one it now wants flat, because the kernel rules an unnamed held symbol as a hold
(``kernel/kernel.py`` ``RiskKernel.rule``) and an arm's returned map means "everything else flat".

Spend
-----
:func:`estimate_qwen_tokens` is the upper bound the owner approves before an LLM arm is run: each
LLM arm's own bound per snapshot (``max_tokens_per_snapshot``, computed the way the daily budget
bounds a call) times the number of snapshots. A run that exceeds its approved budget stops with the
budget's refusal rather than publishing a rival that silently stopped deciding.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from itertools import pairwise
from typing import Final, Protocol, runtime_checkable

from sentiment_agent.analysis.armsim import ArmSimulator, hour_floor
from sentiment_agent.rivals.registry import RivalArm, cap_weights
from sentiment_agent.types import (
    Activation,
    ArmResult,
    BookState,
    KernelInputs,
    PerceptionSnapshot,
    Policy,
    Position,
    PriceSource,
)

WEIGHT_CHANGE_EPS: Final = 1e-12


@runtime_checkable
class EstimatesTokens(Protocol):
    """An LLM arm that can bound its spend before it runs."""

    def max_tokens_per_snapshot(self) -> int: ...


def estimate_qwen_tokens(arms: Sequence[RivalArm], n_snapshots: int) -> int:
    """The most Qwen tokens running ``arms`` over ``n_snapshots`` snapshots can spend: the sum over
    arms whose spec uses an LLM of their per-snapshot bound times ``n_snapshots``. Arms without an
    LLM cost nothing.

    Raises ``TypeError`` for an LLM arm that cannot bound its spend, since an unbounded arm cannot
    be approved, and ``ValueError`` for a negative snapshot count.
    """
    if isinstance(n_snapshots, bool) or not isinstance(n_snapshots, int) or n_snapshots < 0:
        raise ValueError("n_snapshots must be a non-negative integer")
    total = 0
    for arm in arms:
        if not arm.spec.uses_llm:
            continue
        if not isinstance(arm, EstimatesTokens):
            raise TypeError(f"{arm.spec.arm_id} uses an LLM but cannot bound its token spend")
        per_snapshot = arm.max_tokens_per_snapshot()
        if isinstance(per_snapshot, bool) or not isinstance(per_snapshot, int) or per_snapshot < 0:
            raise ValueError(f"{arm.spec.arm_id} reported an invalid per-snapshot bound")
        total += per_snapshot * n_snapshots
    return total


# ================================================================================================
# The book an arm is shown
# ================================================================================================


@dataclass(slots=True)
class _Holding:
    weight: float
    entry: Decimal | None
    opened_at: datetime
    last_increase_at: datetime


@dataclass(slots=True)
class ShadowBook:
    """One arm's own positions, rebuilt from the targets it asked for (module docstring)."""

    holdings: dict[str, _Holding] = field(default_factory=dict)
    ever_held: set[str] = field(default_factory=set)
    rebalances: dict[tuple[date, str], int] = field(default_factory=dict)
    last_price: dict[str, Decimal] = field(default_factory=dict)

    def _price(
        self, symbol: str, snapshot: PerceptionSnapshot, context: BookState
    ) -> Decimal | None:
        quote = snapshot.demo_quotes.get(symbol)
        if quote is not None and quote.mark > 0:
            return quote.mark
        mark = context.marks.get(symbol)
        if mark is not None and mark > 0:
            return mark
        return self.last_price.get(symbol)

    def view(self, snapshot: PerceptionSnapshot, context: BookState) -> BookState:
        """The book the arm sees at ``snapshot``, on the account scale of ``context``."""
        equity = context.equity
        if equity <= 0:
            raise ValueError(
                f"the account equity at {snapshot.taken_at.isoformat()} is not positive"
            )
        positions: dict[str, Position] = {}
        marks: dict[str, Decimal] = {}
        for symbol, holding in self.holdings.items():
            price = self._price(symbol, snapshot, context)
            if price is None or holding.entry is None:
                continue
            self.last_price[symbol] = price
            marks[symbol] = price
            positions[symbol] = Position(
                symbol=symbol,
                qty=Decimal(repr(holding.weight)) * equity / price,
                avg_entry=holding.entry,
                opened_at=holding.opened_at,
                last_increase_at=holding.last_increase_at,
                realized_pnl=Decimal(0),
                fees_paid=Decimal(0),
                stop_price=None,
                stop_venue_id=None,
                last_decision_id=None,
            )
        day = snapshot.taken_at.date()
        return BookState(
            as_of=snapshot.taken_at,
            mark_source=PriceSource.DEMO,
            starting_equity=equity,
            equity=equity,
            peak_equity=equity,
            day_open_equity=equity,
            positions=positions,
            marks=marks,
            fees_today=Decimal(0),
            fees_total=Decimal(0),
            realized_total=Decimal(0),
            rebalances_today={s: n for (d, s), n in self.rebalances.items() if d == day},
            consecutive_losses=0,
            activation=Activation.ACTIVE,
        )

    def proposal(self, targets: dict[str, float]) -> dict[str, float]:
        """``targets``, plus an explicit zero for every symbol the arm ever held and now omits."""
        return {**dict.fromkeys(sorted(self.ever_held), 0.0), **targets}

    def update(
        self, snapshot: PerceptionSnapshot, context: BookState, targets: dict[str, float]
    ) -> None:
        """Record what the arm asked for at ``snapshot``."""
        at = snapshot.taken_at
        for symbol in sorted(set(self.holdings) | set(targets)):
            old = self.holdings.get(symbol)
            before = 0.0 if old is None else old.weight
            after = targets.get(symbol, 0.0)
            if abs(after - before) <= WEIGHT_CHANGE_EPS:
                continue
            key = (at.date(), symbol)
            self.rebalances[key] = self.rebalances.get(key, 0) + 1
            if after == 0.0:
                self.holdings.pop(symbol, None)
                continue
            price = self._price(symbol, snapshot, context)
            if price is not None:
                self.last_price[symbol] = price
            self.ever_held.add(symbol)
            if old is None or (before > 0) != (after > 0):
                self.holdings[symbol] = _Holding(after, price, at, at)
            elif abs(after) > abs(before):
                entry = old.entry
                if entry is not None and price is not None:
                    added = Decimal(repr(abs(after) - abs(before)))
                    kept = Decimal(repr(abs(before)))
                    entry = (kept * entry + added * price) / (kept + added)
                else:
                    entry = price if entry is None else entry
                self.holdings[symbol] = _Holding(after, entry, old.opened_at, at)
            else:
                self.holdings[symbol] = _Holding(
                    after, old.entry, old.opened_at, old.last_increase_at
                )


# ================================================================================================
# The run
# ================================================================================================


def _check_inputs(
    arms: Sequence[RivalArm], snapshots: Sequence[PerceptionSnapshot], books: Sequence[BookState]
) -> None:
    if not arms:
        raise ValueError("no rival arms to run")
    ids = [arm.spec.arm_id for arm in arms]
    if len(ids) != len(set(ids)):
        raise ValueError("two arms share an arm_id; their results could not be told apart")
    if not snapshots:
        raise ValueError("no snapshots: a rival can only be compared on recorded snapshots")
    if len(books) != len(snapshots):
        raise ValueError(
            f"{len(snapshots)} snapshots but {len(books)} books; one book per snapshot"
        )
    for earlier, later in pairwise(snapshots):
        if later.taken_at <= earlier.taken_at:
            raise ValueError(
                "snapshots must be in strictly increasing time order "
                f"({later.snapshot_id} at {later.taken_at.isoformat()})"
            )


def arm_schedule(
    arm: RivalArm,
    snapshots: Sequence[PerceptionSnapshot],
    books: Sequence[BookState],
    inputs: Sequence[KernelInputs],
    policy: Policy,
) -> list[tuple[datetime, dict[str, float], KernelInputs]]:
    """One arm's schedule: its capped targets at every snapshot, decided on its own book."""
    shadow = ShadowBook()
    schedule: list[tuple[datetime, dict[str, float], KernelInputs]] = []
    for snapshot, context, kernel_inputs in zip(snapshots, books, inputs, strict=True):
        shown = shadow.view(snapshot, context)
        raw = arm.targets(snapshot, shown)
        for symbol, weight in raw.items():
            if not math.isfinite(weight):
                raise ValueError(f"{arm.spec.arm_id}: non-finite weight for {symbol}")
        targets = cap_weights(raw, policy)
        schedule.append((snapshot.taken_at, shadow.proposal(targets), kernel_inputs))
        shadow.update(snapshot, context, targets)
    return schedule


def run_rivals(
    arms: Sequence[RivalArm],
    snapshots: Sequence[PerceptionSnapshot],
    books: Sequence[BookState],
    sim: ArmSimulator,
) -> tuple[ArmResult, ...]:
    """One :class:`~sentiment_agent.types.ArmResult` per arm, in ``arms`` order, all marked on the
    same hourly grid: from the hour of the first snapshot to the last hour the simulator's Demo
    marks reach (:meth:`~sentiment_agent.analysis.armsim.ArmSimulator.default_until`).

    ``books[i]`` is the account at ``snapshots[i]``; only its equity scale reaches an arm (module
    docstring). Snapshots must be in strictly increasing time order. A snapshot at or after the last
    mark is not shown to any arm, since no mark could show what it decided (and an LLM arm would
    spend on it for nothing). A budget refusal from an LLM arm propagates and ends the run.
    """
    _check_inputs(arms, snapshots, books)
    policy = sim.policy
    until = sim.default_until()
    if until is None:
        raise ValueError("the simulator holds no Demo marks to run against")
    kept = [(s, b) for s, b in zip(snapshots, books, strict=True) if s.taken_at < until]
    if not kept:
        raise ValueError(f"no snapshot is before the last Demo mark ({until.isoformat()})")
    shown = [s for s, _ in kept]
    accounts = [b for _, b in kept]
    start = hour_floor(shown[0].taken_at)
    inputs = [sim.kernel_inputs(s) for s in shown]
    results = tuple(
        sim.run(
            arm.spec,
            arm_schedule(arm, shown, accounts, inputs, policy),
            start=start,
            until=until,
        )
        for arm in arms
    )
    grids = {tuple(mark.at for mark in result.marks) for result in results}
    if len(grids) != 1:
        raise RuntimeError(
            "the simulator marked the arms on different hours; results would not align"
        )
    return results


__all__ = [
    "EstimatesTokens",
    "ShadowBook",
    "arm_schedule",
    "estimate_qwen_tokens",
    "run_rivals",
]
