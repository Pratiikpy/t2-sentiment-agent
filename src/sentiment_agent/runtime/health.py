"""Health: the heartbeat file a watcher reads, and the status lines a person reads.

The loop rewrites ``var/health/<mode>.json`` on every tick (atomically, so a reader never sees half
a file) and logs the same :class:`~sentiment_agent.types.HealthBeat` to the ledger once an hour. A
watcher that finds the file older than a few ticks knows the loop has stopped, without reading the
ledger; the ledger keeps the hourly record for the published log (win plan Move 8).

:func:`status_lines` is what ``t2sa status`` prints: everything read from the ledger (the book, the
breaker, the last decision, the budget, reconciliation, marks, anchors) and from the health file,
never from the venue, so it answers while the loop runs and costs nothing. The same lines come from
a running :class:`~sentiment_agent.runtime.wiring.App` or from a ledger alone
(:func:`status_from_ledger`), through one :class:`StatusView`.
"""

from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Final

from pydantic import ValidationError

from sentiment_agent.book.projection import Projection, ProjectionError
from sentiment_agent.ledger.chain import HashChainLedger, LedgerError, ledger_path
from sentiment_agent.llm.budget import utc_day
from sentiment_agent.types import (
    Activation,
    BookState,
    BudgetState,
    Clock,
    EventKind,
    Genesis,
    HealthBeat,
    LedgerEvent,
    Policy,
    PriceSource,
    RunMode,
)

if TYPE_CHECKING:
    from sentiment_agent.runtime.wiring import App

STALE_AFTER: Final = timedelta(minutes=3)
"""A health file older than this means the loop is not ticking (it ticks every 30 s)."""


def write_health(path: Path, beat: HealthBeat) -> None:
    """Write ``beat`` to ``path`` atomically: a temporary file, flushed, then renamed over it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    data = beat.model_dump_json(indent=2).encode("utf-8") + b"\n"
    with temp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    for attempt in range(20):
        try:
            temp.replace(path)
            return
        except PermissionError:  # Windows: a reader holds the old file open for an instant
            if attempt == 19:
                with contextlib.suppress(OSError):
                    temp.unlink()
                raise
            time.sleep(0.01)


def read_health(path: Path) -> HealthBeat | None:
    """The last beat written, or ``None`` when there is none or it does not parse."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    try:
        return HealthBeat.model_validate_json(raw)
    except ValidationError:
        return None


def beat_for(
    app: App, *, iteration: int, last_decision_at: datetime | None, detail: str
) -> HealthBeat:
    """The heartbeat of a running app, from its ledger state."""
    budget: BudgetState | None
    try:
        budget = app.budget.state()
    except ValueError:
        budget = None
    return HealthBeat(
        at=app.clock.now(),
        iteration=iteration,
        activation=app.breaker.state().activation,
        open_positions=len(app.held_symbols()),
        last_decision_at=last_decision_at,
        budget=budget,
        detail=detail,
    )


# ================================================================================================
# Status
# ================================================================================================


@dataclass(frozen=True)
class StatusView:
    """What the status lines are made of. Built from an app or from a ledger alone."""

    mode: RunMode
    policy: Policy
    projection: Projection
    head: LedgerEvent | None
    genesis_event: LedgerEvent | None
    health: HealthBeat | None
    now: datetime
    activation: Activation
    breaker_trips: tuple[str, ...]
    budget: BudgetState | None
    running: bool | None
    """True when this process runs the loop, None when read from outside."""


def status_lines(app: App) -> list[str]:
    """The status of a running app, one fact per line."""
    state = app.breaker.state()
    view = StatusView(
        mode=app.mode,
        policy=app.policy,
        projection=app.projection,
        head=app.chain.head(),
        genesis_event=app.genesis_event,
        health=read_health(app.paths.health),
        now=app.clock.now(),
        activation=state.activation,
        breaker_trips=state.trips,
        budget=app.budget.state(),
        running=True,
    )
    return lines_for(view)


def status_from_ledger(
    root: Path, mode: RunMode, *, clock: Clock, policy: Policy, starting_equity: Decimal | None
) -> list[str]:
    """The status of a mode read from its ledger and health file, without building an app."""
    path = ledger_path(root, mode)
    if not path.exists():
        return [f"{mode.value}: no ledger yet ({path.relative_to(root).as_posix()})"]
    chain = HashChainLedger(path, mode=mode, clock=clock)
    try:
        projection = Projection.from_ledger(
            chain, policy, starting_equity=None if mode is RunMode.PAPER else starting_equity
        )
    except (LedgerError, ProjectionError) as exc:
        return [f"{mode.value}: the ledger cannot be read honestly: {exc}. Run `t2sa verify`"]
    genesis_event = next(iter(chain.events(frozenset({EventKind.GENESIS}))), None)
    transitions = projection.breaker_transitions
    trips = transitions[-1].trips if transitions else ()
    states = projection.budget_states
    today = utc_day(clock)
    budget = next((s for s in reversed(states) if s.day == today), None)
    view = StatusView(
        mode=mode,
        policy=policy,
        projection=projection,
        head=chain.head(),
        genesis_event=genesis_event,
        health=read_health(root / "var" / "health" / f"{mode.value}.json"),
        now=clock.now(),
        activation=projection.activation(),
        breaker_trips=trips,
        budget=budget,
        running=None,
    )
    return lines_for(view)


def lines_for(view: StatusView) -> list[str]:
    p = view.projection
    lines: list[str] = []
    head = view.head
    lines.append(
        f"mode {view.mode.value}: "
        + (
            "empty ledger"
            if head is None
            else f"{head.seq + 1} events, head seq {head.seq} {head.hash[:16]} at {_t(head.ts)}"
        )
    )
    lines.append(_genesis_line(view))
    lines.append(
        f"breaker {view.activation.value}"
        + (f" ({', '.join(view.breaker_trips)})" if view.breaker_trips else "")
    )
    lines.extend(_book_lines(view))
    decisions = p.decisions
    if decisions:
        last = decisions[-1]
        stance = last.decision.stance.value if last.decision is not None else "no decision"
        lines.append(
            f"last decision {last.decision_id} at {_t(last.decided_at)}: {last.outcome.value}, "
            f"{stance}; {len(decisions)} decision(s) in the log, "
            f"{sum(1 for d in decisions if d.outcome.is_outage)} outage(s)"
        )
    else:
        lines.append("no decision yet")
    owner = sum(1 for t in p.triggers if t.kind.value == "owner_manual")
    lines.append(
        f"triggers logged {len(p.triggers)} (owner interventions {owner}); protective actions "
        f"{len(p.protective_actions)}; orders submitted {len(p.submissions)}, fills "
        f"{len(p.fills)}"
    )
    budget = view.budget
    if budget is not None:
        lines.append(
            f"Qwen budget {budget.day}: {budget.spent_tokens} of {budget.cap_tokens} tokens, "
            f"{budget.calls} call(s), {budget.unreported_calls} unreported"
        )
    recs = p.reconciliations
    if recs:
        last_rec = recs[-1]
        kinds = sorted({d.kind for d in last_rec.discrepancies})
        lines.append(
            f"last reconciliation {_t(last_rec.at)}: "
            + (
                "clean"
                if last_rec.clean
                else f"{len(last_rec.discrepancies)} discrepancy(ies) ({', '.join(kinds)})"
            )
        )
    marks = p.marks
    if marks:
        m = marks[-1]
        lines.append(
            f"last hourly mark {_t(m.at)}: equity {m.equity_book} (venue "
            f"{m.equity_venue if m.equity_venue is not None else 'not read'}), gross "
            f"{m.gross_weight:.2%}; {len(marks)} mark(s)"
        )
    anchors = p.anchors
    if anchors:
        a = anchors[-1]
        lines.append(f"last anchor: seq {a.target_seq} {a.status} at {_t(a.submitted_at)}")
    lines.append(_health_line(view))
    return lines


def _genesis_line(view: StatusView) -> str:
    event = view.genesis_event
    if event is None:
        if view.mode is RunMode.PAPER:
            return "genesis: none. No paper order may be sent until `t2sa genesis` writes it"
        return "genesis: none (a rehearsal without pre-registration)"
    genesis = Genesis.model_validate(event.payload)
    active = genesis.policy_hash
    for amendment in view.projection.amendments:
        active = amendment.new_policy_hash
    loaded = view.policy.content_hash()
    verdict = (
        "matches the loaded policy"
        if loaded == active
        else (f"DOES NOT match the loaded policy {loaded[:16]}: no order is sent")
    )
    return (
        f"genesis {event.hash[:16]} at {_t(genesis.created_at)}; policy in force {active[:16]} "
        f"({len(view.projection.amendments)} amendment(s)) {verdict}"
    )


def _book_lines(view: StatusView) -> list[str]:
    p = view.projection
    if p.starting_equity is None:
        return ["book: no starting equity recorded yet"]
    snapshots = p.snapshots
    marks: dict[str, Decimal] = {}
    if snapshots:
        marks = {s: q.mark for s, q in snapshots[-1].demo_quotes.items() if q.mark > 0}
    try:
        builder = p.builder()
        for symbol, position in builder.positions().items():
            marks.setdefault(symbol, position.avg_entry)
        at = max(view.now, builder.last_fill_at or view.now)
        book: BookState = p.book(at=at, marks=marks, mark_source=PriceSource.DEMO)
    except (ProjectionError, ValueError) as exc:
        return [f"book: unavailable ({exc})"]
    lines = [
        f"equity {book.equity:.2f} USDT (start {book.starting_equity}), day "
        f"{book.day_return:+.3%}, drawdown {book.drawdown:+.3%}, gross {book.gross_weight:.2%}, "
        f"fees today {book.fees_today:.4f}, {len(p.closed_trades)} closed trade(s)"
    ]
    for symbol, pos in sorted(book.positions.items()):
        stop = f"stop {pos.stop_price}" if pos.stop_price is not None else "NO STOP"
        lines.append(
            f"  {symbol} qty {pos.qty} @ {pos.avg_entry}, weight {book.weight(symbol):+.3%}, "
            f"{stop}, opened {_t(pos.opened_at)}"
        )
    return lines


def _health_line(view: StatusView) -> str:
    beat = view.health
    if beat is None:
        return "health: no heartbeat file (the loop has not run)"
    age = view.now - beat.at
    state = "ticking" if age <= STALE_AFTER else f"STALE ({_age(age)} since the last tick)"
    extra = f"; {beat.detail}" if beat.detail else ""
    return (
        f"health: {state}, iteration {beat.iteration}, {beat.activation.value}, "
        f"{beat.open_positions} open position(s), last beat {_t(beat.at)}{extra}"
    )


def _t(at: datetime) -> str:
    return at.strftime("%Y-%m-%dT%H:%M:%SZ")


def _age(delta: timedelta) -> str:
    seconds = int(delta.total_seconds())
    if seconds < 120:
        return f"{seconds} s"
    if seconds < 7200:
        return f"{seconds // 60} min"
    return f"{seconds // 3600} h"


__all__ = [
    "STALE_AFTER",
    "StatusView",
    "beat_for",
    "lines_for",
    "read_health",
    "status_from_ledger",
    "status_lines",
    "write_health",
]
