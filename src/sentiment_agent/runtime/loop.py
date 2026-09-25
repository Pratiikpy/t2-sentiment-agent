"""The run loop: one tick at a time, every cadence the design names, state always from the ledger.

A tick is short and never sleeps. :meth:`RunLoop.run_forever` calls it every ``interval_s`` (30 s by
default) and each tick does only what is due (DESIGN.md §4-§11):

========================  ===================================================================
Every tick                Fold in events another process appended; poll the simulated venue so its
                          stops can fire (SIMULATED); take any owner request from the inbox.
Every 5 min               A light snapshot, logged, and the light-snapshot trigger kinds
                          evaluated on it (first, so the protective check below rules on a fresh
                          snapshot).
Every 60 s                The protective check: fresh keyless quotes, the breaker assessed, and
                          :meth:`RiskKernel.protective` (daily kill, weekend pre-flatten and freeze,
                          venue integrity, breaker halts). It can only reduce.
Every tick                Due heartbeats (the US open, each funding settlement) and owner requests
                          join the candidates; the trigger engine admits or refuses every one, and
                          every one is logged either way.
Before admission          When the candidates will start a decision
                          (:meth:`~sentiment_agent.events.triggers.TriggerEngine.preview`), the
                          full snapshot is taken first and the full-snapshot trigger kinds
                          (coordinated clusters, earnings, filings) are evaluated on it and admitted
                          in the same batch, so they join that decision (run 2, ``run2-d1``).
On an admitted set        :meth:`RunLoop.decision_cycle`: full snapshot -> DecisionAgent -> kernel
                          -> planner -> approval -> Executor -> reconciliation -> stop sync.
Every snapshot            Feed health (:mod:`sentiment_agent.perception.feeds`): a ``feed_health``
                          event on every full snapshot, and on a light one whenever a source
                          starts or stops failing or the instruments Fear & Greed, funding or open
                          interest cannot be evaluated for change (run 2, ``run2-d2``).
Every 15 min, 00:05 UTC   Reconciliation (read-only), and the full order history once a day; the
                          stop manager keeps one venue stop per open position.
First tick of a record    The anchor MARK: the starting equity on the current hour, flat, before
                          any decision can be admitted (:meth:`RunLoop._anchor_mark`).
Every UTC hour            The hourly MARK, then the public export and a HEALTH event.
Daily after 00:10 UTC     The ledger head is stamped with OpenTimestamps; pending stamps upgraded.
Every tick                The health file (``var/health/<mode>.json``).
========================  ===================================================================

**Nothing deterministic adds exposure.** The only path to an exposure-adding order is
:meth:`RunLoop.decision_cycle` with a ``DECIDED`` model decision, ruled by the kernel and minted by
the approval module. A model outage is answered by the kernel's protective flatten
(``llm_outage``), never by a rule placing a trade in the model's place (handbook:224).

**Quotes the kernel rules on are fresh.** A heartbeat decision streams several thousand reasoning
tokens, so the snapshot's quotes can be minutes old when the answer arrives. The kernel therefore
rules on quotes read after the decision (G10 refuses anything older than 120 s), against the book
valued at those quotes, while citing the snapshot the model read (G10 also refuses a snapshot older
than 15 minutes).

**Everything the kernel read is kept.** Each ``KERNEL_RULING`` event carries a blob with the book,
market inputs, breaker state and context it ruled on, so ``t2sa replay`` can rule again from the log
alone and show the same ruling id and the same intent ids.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Final, Literal

from sentiment_agent.book.marks import MARK_MAX_LATENESS, MarkError, hour_floor, mark_point
from sentiment_agent.events.schedule import heartbeats_between
from sentiment_agent.events.triggers import (
    FULL_SNAPSHOT_KINDS,
    LIGHT_SNAPSHOT_KINDS,
    REFUSED_BUDGET,
    UNCONDITIONAL_KINDS,
)
from sentiment_agent.execution.environment import EnvironmentRefused
from sentiment_agent.hashing import content_hash
from sentiment_agent.kernel.approval import ApprovalError
from sentiment_agent.kernel.breaker import next_utc_midnight
from sentiment_agent.ledger.anchor import stamp, upgrade
from sentiment_agent.ledger.chain import GenesisError, LedgerError, referenced_blobs
from sentiment_agent.ledger.genesis import require_genesis
from sentiment_agent.llm.budget import MEASURED_PROMPT_BYTES, decision_bound
from sentiment_agent.perception.feeds import changed, feed_report, summary_line
from sentiment_agent.runtime.health import beat_for, write_health
from sentiment_agent.runtime.wiring import App, ruling_context_blob
from sentiment_agent.sources.toolkit import ToolkitFacade, probe_all
from sentiment_agent.types import (
    AnchorRecord,
    BlobRef,
    BookState,
    BreakerState,
    CardOrder,
    DecisionCard,
    DecisionEvent,
    DecisionRecord,
    EventKind,
    FeedHealthReport,
    KernelInputs,
    KernelRuling,
    LedgerEvent,
    LlmOutcome,
    MarkPoint,
    Model,
    OrderState,
    PerceptionSnapshot,
    ProtectiveAction,
    Quote,
    ReconciliationReport,
    RulingContext,
    RunMode,
    SnapshotEvent,
    ToolkitProbe,
    Trigger,
    TriggerKind,
    UtcDatetime,
)

PROTECTIVE_EVERY: Final = timedelta(seconds=60)
LIGHT_SNAPSHOT_EVERY: Final = timedelta(minutes=5)
RECONCILE_EVERY: Final = timedelta(minutes=15)
DAILY_RECONCILE_AFTER: Final = time(0, 5)
DAILY_ANCHOR_AFTER: Final = time(0, 10)
ANCHOR_UPGRADE_EVERY: Final = timedelta(hours=6)
HEALTH_EVENT_EVERY: Final = timedelta(hours=1)
SPECS_EVERY: Final = timedelta(hours=24)
TOOLKIT_PROBE_EVERY: Final = timedelta(hours=24)
FILL_OVERLAP: Final = timedelta(minutes=5)
"""Each sweep re-reads fills from a little before the last one: venue timestamps may lead ours, and
fills are de-duplicated by ``execId``, so overlap costs nothing and a gap would lose a fill."""
FULL_HISTORY_LOOKBACK: Final = timedelta(days=1)
GENESIS_FILL_GAP: Final = timedelta(milliseconds=1)
"""The fills window opens this long after the genesis: the venue stamps fills to the millisecond,
and a pre-genesis plumbing fill may carry the genesis's own instant."""
MAX_CONSECUTIVE_FAILURES: Final = 5
OWNER_SOURCE: Final = "owner:t2sa decide"
OWNER_REASON_CHARS: Final = 200
_OWNER_SAFE: Final = re.compile(r"[^A-Za-z0-9 .,:;%()+\-/'_]")


class TickReport(Model):
    """What one tick did, in order. ``did`` names each action (``"protective_check"``, ...)."""

    at: UtcDatetime
    did: tuple[str, ...]


def owner_trigger(
    reason: str, *, at: datetime, symbols: Sequence[str] = (), request_id: str | None = None
) -> Trigger:
    """An ``owner_manual`` trigger: logged and published as a human intervention (DESIGN.md §8).

    The reason is the owner's own words, reduced to plain characters and a bounded length because
    triggers are rendered into the prompt unspotlighted."""
    clean = " ".join(_OWNER_SAFE.sub(" ", reason).split())[:OWNER_REASON_CHARS] or "no reason given"
    ident = request_id or content_hash({"at": at, "reason": clean, "symbols": list(symbols)})[:16]
    return Trigger(
        trigger_id=f"{TriggerKind.OWNER_MANUAL.value}@{ident}",
        kind=TriggerKind.OWNER_MANUAL,
        fired_at=at,
        symbols=tuple(symbols),
        detail=f"owner requested a decision: {clean}",
        source=OWNER_SOURCE,
    )


def request_decision(inbox: Path, reason: str, *, at: datetime, symbols: Sequence[str]) -> Path:
    """Leave an owner request for the running loop to pick up at its next tick."""
    inbox.mkdir(parents=True, exist_ok=True)
    stamp_text = at.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    ident = content_hash({"at": at, "reason": reason, "symbols": list(symbols)})[:12]
    path = inbox / f"decide-{stamp_text}-{ident}.json"
    temp = path.with_suffix(".tmp")
    record = {"reason": reason, "at": at.astimezone(UTC).isoformat(), "symbols": list(symbols)}
    temp.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    temp.replace(path)
    return path


@dataclass
class _Acted:
    ruling: KernelRuling
    orders: list[CardOrder]
    sent: bool


@dataclass(frozen=True)
class _Prepared:
    """A full snapshot taken before admission, and the full-snapshot events it shows."""

    snapshot: PerceptionSnapshot
    book: BookState
    riders: tuple[Trigger, ...]


class RunLoop:
    """Drives one :class:`App`. Holds only schedule markers; every fact is read from the ledger."""

    def __init__(self, app: App, *, probe_toolkit: bool = True) -> None:
        """``probe_toolkit=False`` is for one-shot commands (``t2sa once``, ``t2sa decide``): the
        daily probe runs on a worker thread and is logged by a *later* tick, which a single tick
        never has, so starting it there would spend minutes of calls and discard the result.
        ``t2sa probe-toolkit`` measures on demand."""
        self._app = app
        self._probe_enabled = probe_toolkit
        projection = app.projection
        self._iteration = 0
        self._started = False
        self._since = app.resumed_from_ts or app.started_at
        self._last_protective: datetime | None = None
        snapshots = projection.snapshots
        self._last_light: datetime | None = snapshots[-1].taken_at if snapshots else None
        reconciliations = projection.reconciliations
        self._last_reconcile: datetime | None = reconciliations[-1].at if reconciliations else None
        read = [r for r in reconciliations if r.fills_read]
        self._fills_cursor: datetime | None = read[-1].at if read else None
        """When the venue's fills were last read successfully: the next sweep's window starts here
        (less the overlap), never at a failed attempt, so a fill cannot fall behind the window."""
        self._last_full_day: str | None = None
        marks = projection.marks
        self._last_mark_hour: datetime | None = marks[-1].at if marks else None
        self._mark_problem_hour: datetime | None = None
        anchors = projection.anchors
        self._last_anchor_day: str | None = (
            max(a.submitted_at for a in anchors).date().isoformat() if anchors else None
        )
        self._last_upgrade: datetime | None = None
        beats = projection.health_beats
        self._last_health_event: datetime | None = beats[-1].at if beats else None
        self._last_specs = app.started_at
        probes = projection.toolkit_probes
        self._last_probe: datetime | None = probes[-1].at if probes else None
        decisions = projection.decisions
        self._last_decision_at: datetime | None = decisions[-1].decided_at if decisions else None
        self._failures = 0
        self._last_error = ""
        self._pending_owner: list[Trigger] = []
        self._feeds: FeedHealthReport | None = self._replay_feeds()
        """The feed health as of the newest snapshot, rebuilt from every logged snapshot so a
        restart continues the same streaks (:mod:`sentiment_agent.perception.feeds` is pure)."""
        self._probe_worker: threading.Thread | None = None
        self._probe_result: ToolkitProbe | str | None = None
        self.last_card: DecisionCard | None = None
        """The card of the most recent decision cycle this loop ran."""
        self.exporter: Callable[[App], Sequence[str]] | None = None
        """Called after every hourly mark to publish the record (``cli`` sets the public export).
        It must return promptly (the CLI's runs on a worker thread) and returns notes to log."""

    @property
    def app(self) -> App:
        return self._app

    @property
    def iteration(self) -> int:
        return self._iteration

    def request_owner_decision(self, reason: str, *, symbols: Sequence[str] = ()) -> Trigger:
        """Queue an owner trigger for the next tick (in-process; ``t2sa decide`` uses the inbox)."""
        trigger = owner_trigger(reason, at=self._app.clock.now(), symbols=self._universe(symbols))
        self._pending_owner.append(trigger)
        return trigger

    # ============================================================================================
    # The tick
    # ============================================================================================

    def tick(self) -> TickReport:
        app = self._app
        app.sync()
        now = app.clock.now()
        did: list[str] = []
        if not self._started:
            self._startup(now, did)
            self._started = True
        if now - self._last_specs >= SPECS_EVERY:
            app.refresh_specs()
            app.rebuild_stops()
            self._last_specs = now
            did.append("specs_refreshed")

        self._anchor_mark(now, did)

        if app.simulated_venue is not None:
            fired = app.simulated_venue.poll()
            if fired:
                did.append("venue_stop_fired")
                self._reconcile(now, full=False, did=did)

        candidates: list[Trigger] = []
        if _due(self._last_light, now, LIGHT_SNAPSHOT_EVERY):
            snapshot, book = self._snapshot(light=True)
            self._last_light = snapshot.taken_at
            did.append("light_snapshot")
            candidates.extend(app.triggers.evaluate(snapshot, book, kinds=LIGHT_SNAPSHOT_KINDS))

        if _due(self._last_protective, now, PROTECTIVE_EVERY):
            self._protective(now, did)
            self._last_protective = now
        candidates.extend(app.triggers.due_heartbeats(self._since))
        candidates.extend(self._inbox(did))
        if self._pending_owner:
            candidates.extend(self._pending_owner)
            self._pending_owner = []
        self._since = now
        if candidates:
            prepared = self._prepare(candidates, did)
            if prepared is not None:
                candidates.extend(prepared.riders)
            admitted = self._admit(candidates, did)
            if admitted:
                self.decision_cycle(admitted, prepared=prepared)
                did.append("decision_cycle")

        if app.mode is not RunMode.DRYRUN:
            day = now.date().isoformat()
            if now.time() >= DAILY_RECONCILE_AFTER and self._last_full_day != day:
                self._reconcile(now, full=True, did=did)
                self._last_full_day = day
            elif _due(self._last_reconcile, now, RECONCILE_EVERY):
                self._reconcile(now, full=False, did=did)

        self._mark(now, did)
        self._anchor(now, did)
        self._probe(now, did)
        self._health(now, did)
        self._iteration += 1
        return TickReport(at=now, did=tuple(did))

    def run_forever(self, *, stop: threading.Event, interval_s: float = 30.0) -> None:
        """Tick until ``stop`` is set. A failed tick is logged and the loop carries on; five in a
        row, a broken ledger, a refused environment or a pre-registration mismatch stop it."""
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        while not stop.is_set():
            try:
                self.tick()
                self._failures = 0
            except (EnvironmentRefused, LedgerError, GenesisError):
                raise
            except Exception as exc:
                self._failures += 1
                self._last_error = f"{type(exc).__name__}: {exc}"[:500]
                self._record_failure()
                if self._failures >= MAX_CONSECUTIVE_FAILURES:
                    raise
            stop.wait(interval_s)

    def _record_failure(self) -> None:
        app = self._app
        text = f"tick {self._iteration} failed ({self._failures} in a row): {self._last_error}"
        try:
            app.note(text)
        except LedgerError:
            raise
        except Exception as exc:  # the note is best effort; the health file below still says it
            text = f"{text} (and the note could not be logged: {type(exc).__name__})"
        write_health(
            app.paths.health,
            beat_for(
                app, iteration=self._iteration, last_decision_at=self._last_decision_at, detail=text
            ),
        )

    # ============================================================================================
    # Startup
    # ============================================================================================

    def _startup(self, now: datetime, did: list[str]) -> None:
        app = self._app
        resumed = (
            "a fresh ledger"
            if app.resumed_from_seq is None
            else f"seq {app.resumed_from_seq} ({app.resumed_from_ts:%Y-%m-%dT%H:%M:%SZ})"
        )
        lines = [
            f"runtime started: mode {app.mode.value}, model {app.chat_model.model_name} "
            f"({app.llm}), policy {app.policy.version} {app.policy.content_hash()[:16]}, resumed "
            f"from {resumed}; {len(app.tracker.live())} live order(s) to reconcile, "
            f"{len(app.held_symbols())} open position(s), breaker "
            f"{app.breaker.state().activation.value}",
            *app.startup_notes,
        ]
        app.note(" | ".join(lines))
        did.append("startup")
        if app.mode is RunMode.DRYRUN:
            if app.resumed_from_seq is None:
                app.note(
                    f"dry-run book: notional starting equity {app.projection.starting_equity} "
                    "USDT; nothing is sent in this mode, so the book stays flat"
                )
            return
        self._reconcile(now, full=True, did=did)
        self._last_full_day = now.date().isoformat()

    # ============================================================================================
    # Protective check
    # ============================================================================================

    def _protective(self, now: datetime, did: list[str]) -> None:
        app = self._app
        demo, live = app.quotes()
        book = app.book(at=now, demo=demo)
        snapshot = app.latest_snapshot()
        inputs = app.kernel_inputs(at=now, demo=demo, live=live, snapshot=snapshot)
        state = self._assess(book, inputs, llm_outage=False, decision_id=None)
        did.append("protective_check")
        ruling = app.kernel.protective(book=book, inputs=inputs, breaker=state)
        if ruling is None:
            return
        self._act(
            ruling, book, inputs, breaker=state, context=None, proposed=None, llm_outage=False
        )
        did.append(f"protective_{ruling.protective_reason}")

    def _assess(
        self,
        book: BookState,
        inputs: KernelInputs,
        *,
        llm_outage: bool,
        decision_id: str | None,
    ) -> BreakerState:
        app = self._app
        state, transition = app.breaker.assess(
            book, inputs=inputs, llm_outage=llm_outage, decision_id=decision_id
        )
        if transition is not None:
            app.log(EventKind.BREAKER_TRANSITION, transition)
        return state

    # ============================================================================================
    # Snapshots and triggers
    # ============================================================================================

    def _snapshot(self, *, light: bool) -> tuple[PerceptionSnapshot, BookState]:
        """Build and log a snapshot. The book passed in is valued at the newest quotes the ledger
        holds; the returned book is valued at the snapshot's own Demo quotes."""
        app = self._app
        previous = app.latest_snapshot()
        now = app.clock.now()
        before = app.book(at=now, demo=previous.demo_quotes if previous is not None else {})
        snapshot = app.snapshots.build(book=before, light=light)
        app.log(EventKind.SNAPSHOT, SnapshotEvent(snapshot=snapshot))
        self._log_feeds(snapshot, light=light)
        book = app.book(at=app.clock.now(), demo=snapshot.demo_quotes)
        return snapshot, book

    def _feed_report(
        self, snapshot: PerceptionSnapshot, previous: FeedHealthReport | None
    ) -> FeedHealthReport:
        app = self._app
        return feed_report(
            snapshot,
            policy=app.policy,
            previous=previous,
            oi_thresholds=app.oi_thresholds,
            funding_scope=app.triggers.funding_scope,
        )

    def _replay_feeds(self) -> FeedHealthReport | None:
        report: FeedHealthReport | None = None
        for snapshot in self._app.projection.snapshots:
            report = self._feed_report(snapshot, report)
        return report

    def _log_feeds(self, snapshot: PerceptionSnapshot, *, light: bool) -> FeedHealthReport:
        """Feed health after ``snapshot``: logged on every full snapshot (its decision card
        carries it) and on a light one when :func:`~sentiment_agent.perception.feeds.changed`."""
        previous = self._feeds
        report = self._feed_report(snapshot, previous)
        self._feeds = report
        if not light or changed(report, previous):
            self._app.log(EventKind.FEED_HEALTH, report)
        return report

    @property
    def feed_health(self) -> FeedHealthReport | None:
        """The feed health as of the newest snapshot this loop knows."""
        return self._feeds

    def _prepare(self, candidates: Sequence[Trigger], did: list[str]) -> _Prepared | None:
        """The full snapshot for a decision the candidates will start, taken before admission.

        Returns ``None`` (nothing taken) unless the engine would admit at least one candidate and
        :meth:`_admit` would not then refuse the batch for budget or a missing anchor mark. The
        snapshot's full-snapshot trigger kinds become riders in the same admission batch, so a
        coordinated cluster, an earnings date or a filing seen on it joins the decision rather than
        waking another one; heartbeat batches still count nothing against the daily cap."""
        app = self._app
        would = app.triggers.preview(candidates)
        if not would or self._budget_short(would) is not None or not app.projection.marks:
            return None
        snapshot, book = self._snapshot(light=False)
        self._last_light = snapshot.taken_at
        known = {t.trigger_id for t in candidates}
        riders = tuple(
            t
            for t in app.triggers.evaluate(snapshot, book, kinds=FULL_SNAPSHOT_KINDS)
            if t.trigger_id not in known
        )
        did.append(f"full_snapshot_riders_{len(riders)}")
        return _Prepared(snapshot=snapshot, book=book, riders=riders)

    def _inbox(self, did: list[str]) -> list[Trigger]:
        app = self._app
        inbox = app.paths.inbox
        if not inbox.is_dir():
            return []
        out: list[Trigger] = []
        for path in sorted(inbox.glob("decide-*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                reason = str(record.get("reason", ""))
                symbols = self._universe([str(s) for s in record.get("symbols", [])])
            except (OSError, ValueError, AttributeError) as exc:
                app.note(f"owner request {path.name} unreadable and skipped: {exc}")
                path.unlink(missing_ok=True)
                continue
            trigger = owner_trigger(
                reason, at=app.clock.now(), symbols=symbols, request_id=path.stem[-12:]
            )
            app.note(f"owner request {path.name}: {trigger.detail}", author="owner")
            out.append(trigger)
            path.unlink(missing_ok=True)
        if out:
            did.append("owner_request")
        return out

    def _universe(self, symbols: Sequence[str]) -> tuple[str, ...]:
        universe = set(self._app.symbols)
        return tuple(s for s in dict.fromkeys(symbols) if s in universe)

    def _admit(self, candidates: Sequence[Trigger], did: list[str]) -> list[Trigger]:
        app = self._app
        admitted, refused = app.triggers.admit(candidates)
        short = self._budget_short(admitted)
        if short is None and admitted and not app.projection.marks:
            short = (
                "no_anchor_mark: the record has no MARK yet, so a fill now would fall before the "
                "first point of the return series and its costs would be dropped from every metric"
            )
        if short is not None:
            refused = [*refused, *((t, short) for t in admitted)]
            admitted = []
        for trigger in admitted:
            app.log(EventKind.TRIGGER, trigger)
        for trigger, _ in refused:
            app.log(EventKind.TRIGGER, trigger)
        if refused:
            app.note(
                "triggers refused at admission: "
                + "; ".join(f"{t.trigger_id} ({t.kind.value}): {why}" for t, why in refused)
            )
        did.append(f"triggers_admitted_{len(admitted)}_refused_{len(refused)}")
        return admitted

    def _prompt_bytes_estimate(self) -> int:
        """The larger of the recorded prompt and the newest logged request body (whose JSON
        framing only makes it larger than its messages)."""
        decisions = self._app.projection.decisions
        ref = decisions[-1].call.request_blob if decisions else None
        return max(MEASURED_PROMPT_BYTES, ref.size if ref is not None else 0)

    def _budget_short(self, admitted: Sequence[Trigger]) -> str | None:
        """Why an event-only admission must be refused for budget, or ``None``.

        A heartbeat or an owner request is never refused here. An event decision is admitted only
        while the day's remaining tokens carry it *and* every heartbeat still scheduled before
        00:00 UTC, each at its worst case (:func:`~sentiment_agent.llm.budget.decision_bound`):
        a cap reached mid-day would refuse a heartbeat's call, and a refused call is answered by
        the kernel's outage flatten."""
        if not admitted or any(t.kind in UNCONDITIONAL_KINDS for t in admitted):
            return None
        app = self._app
        now = app.clock.now()
        midnight = next_utc_midnight(now)
        ahead = [t for t in heartbeats_between(now, midnight, app.policy) if t.fired_at < midnight]
        per = decision_bound(self._prompt_bytes_estimate(), app.policy)
        need = (1 + len(ahead)) * per
        remaining = app.budget.state().remaining
        if remaining >= need:
            return None
        return (
            f"{REFUSED_BUDGET}: {remaining} tokens remain today; this decision and the "
            f"{len(ahead)} heartbeat(s) still due before 00:00 UTC need up to {need} "
            f"({per} each at worst)"
        )

    # ============================================================================================
    # The decision cycle
    # ============================================================================================

    def decision_cycle(
        self, triggers: Sequence[Trigger], *, prepared: _Prepared | None = None
    ) -> DecisionCard | None:
        """One decision over the whole book for an admitted trigger set, end to end.

        ``prepared`` is the full snapshot :meth:`tick` took before admission; without it the cycle
        takes its own. Returns the cycle's decision card, built from the events this cycle logged
        (from the prepared snapshot's own events on), or ``None`` when there is nothing to decide
        on (no trigger)."""
        if not triggers:
            return None
        app = self._app
        if prepared is not None:
            first = _first_seq_index(app.ledger.appended, prepared.snapshot.snapshot_id)
            snapshot, book = prepared.snapshot, prepared.book
        else:
            first = len(app.ledger.appended)
            snapshot, book = self._snapshot(light=False)
            self._last_light = snapshot.taken_at
        record = app.agent.decide(snapshot, book, triggers)
        app.log(EventKind.DECISION, DecisionEvent(record=record))
        app.log(EventKind.BUDGET_STATE, app.budget.state())
        self._last_decision_at = record.decided_at

        demo, live = app.quotes()
        now = app.clock.now()
        fresh_book = app.book(at=now, demo=demo)
        inputs = app.kernel_inputs(at=now, demo=demo, live=live, snapshot=snapshot)
        acted: _Acted | None
        if record.outcome is LlmOutcome.DECIDED and record.decision is not None:
            state = self._assess(
                fresh_book, inputs, llm_outage=False, decision_id=record.decision_id
            )
            context = RulingContext(
                decision_id=record.decision_id,
                protective_reason=None,
                grounding=dict(record.grounding),
                invalidation_fired={
                    t.symbol: t.invalidation_triggered for t in record.decision.targets
                },
            )
            ruling = app.kernel.rule(
                proposed=record.proposed_weights,
                book=fresh_book,
                inputs=inputs,
                context=context,
                breaker=state,
            )
            acted = self._act(
                ruling,
                fresh_book,
                inputs,
                breaker=state,
                context=context,
                proposed=record.proposed_weights,
                llm_outage=False,
            )
        else:
            state = self._assess(fresh_book, inputs, llm_outage=True, decision_id=None)
            outage = app.kernel.protective(
                book=fresh_book, inputs=inputs, breaker=state, llm_outage=True
            )
            if outage is None:
                app.note(
                    f"model outage ({record.outcome.value}) on decision {record.decision_id}: the "
                    "book is flat, so there is nothing to flatten; no rule trades in the model's "
                    "place"
                )
                acted = None
            else:
                acted = self._act(
                    outage,
                    fresh_book,
                    inputs,
                    breaker=state,
                    context=None,
                    proposed=None,
                    llm_outage=True,
                )
        feeds = self._feeds
        if feeds is not None and feeds.snapshot_id != snapshot.snapshot_id:
            feeds = None  # pragma: no cover - every snapshot is reported as it is logged
        card = self._card(record, snapshot, triggers, acted, first, feeds)
        self.last_card = card
        return card

    # ============================================================================================
    # Acting on a ruling
    # ============================================================================================

    def _act(
        self,
        ruling: KernelRuling,
        book: BookState,
        inputs: KernelInputs,
        *,
        breaker: BreakerState,
        context: RulingContext | None,
        proposed: dict[str, float] | None,
        llm_outage: bool,
    ) -> _Acted:
        """Log the ruling with what it read, plan, approve, execute, then reconcile and sync stops.

        A PAPER send additionally requires, at this moment, that the pre-registered policy is still
        the one loaded: an amendment logged by another process stops sends until a restart."""
        app = self._app
        kind: Literal["rule", "protective"] = "rule" if context is not None else "protective"
        blob = ruling_context_blob(
            app.blobs,
            kind=kind,
            book=book,
            inputs=inputs,
            breaker=breaker,
            context=context,
            proposed=proposed,
            guards=[g.value for g in ruling.guards_applied],
            llm_outage=llm_outage,
        )
        app.log(EventKind.KERNEL_RULING, ruling, blobs=[blob])
        if ruling.protective_reason is not None:
            changed = tuple(i.symbol for i in ruling.instruments if i.changed_by_kernel)
            app.log(
                EventKind.PROTECTIVE_ACTION,
                ProtectiveAction(
                    at=ruling.at,
                    reason=ruling.protective_reason,
                    symbols=changed,
                    ruling_id=ruling.ruling_id,
                    detail=_protective_detail(ruling),
                ),
            )
        plan = app.planner.plan(ruling, book, inputs, now=app.clock.now())
        app.log(EventKind.ORDER_PLAN, plan)
        if not plan.intents:
            return _Acted(ruling=ruling, orders=[], sent=False)
        if app.mode is RunMode.PAPER:
            require_genesis(app.chain, app.policy)
        try:
            approved = app.planner.approve(plan, ruling, book, inputs)
        except ApprovalError as refused:
            now = app.clock.now()
            for intent in plan.intents:
                if app.tracker.state(intent.client_oid) is None:
                    change = app.tracker.transition(
                        intent.client_oid,
                        OrderState.DENIED,
                        at=now,
                        reason=f"approval refused the plan: {refused}"[:500],
                    )
                    app.log(EventKind.ORDER_STATE, change)
            app.note(f"approval refused plan {plan.plan_id[:16]}; nothing was sent: {refused}")
            return _Acted(ruling=ruling, orders=[], sent=False)
        orders = app.executor.execute(approved)
        sent = app.mode is not RunMode.DRYRUN and any(
            o.state not in (OrderState.DENIED, OrderState.INITIALISED) for o in orders
        )
        if sent:
            self._reconcile(app.clock.now(), full=False, did=None)
        return _Acted(ruling=ruling, orders=orders, sent=sent)

    # ============================================================================================
    # Reconciliation and stops
    # ============================================================================================

    def reconcile(self, *, full: bool) -> ReconciliationReport | None:
        """One reconciliation sweep now (``t2sa reconcile``), then a stop sync. ``None`` in
        DRYRUN, where nothing was sent and there is nothing to read back."""
        app = self._app
        app.sync()
        if app.mode is RunMode.DRYRUN:
            return None
        self._reconcile(app.clock.now(), full=full, did=None)
        reports = app.projection.reconciliations
        return reports[-1] if reports else None

    def _reconcile(self, now: datetime, *, full: bool, did: list[str] | None) -> None:
        app = self._app
        if app.mode is RunMode.DRYRUN:
            return
        demo, _ = app.quotes()
        book = app.book(at=now, demo=demo)
        report = app.reconciler.run(
            book=book,
            since=self._fills_since(now, full=full),
            known_fill_ids=app.tracker.fill_ids(),
            full_history=full,
        )
        self._last_reconcile = report.at
        if report.fills_read:
            self._fills_cursor = report.at
        if did is not None:
            did.append("reconcile_full" if full else "reconcile")
        self._sync_stops(demo)

    def _fills_since(self, now: datetime, *, full: bool) -> datetime:
        """The start of the fills window.

        A regular sweep starts at the last sweep whose fills read *succeeded* (less the overlap),
        so fills unreadable for any length of time are still inside the next window. The daily
        full sweep reads the whole run: from the genesis when there is one, else one day back.

        **Never before the genesis.** Fills executed before it (the owner-approved plumbing test,
        DESIGN.md section 16) belong to the pre-genesis ledger and are disclosed there, never
        scored; a window reaching past the genesis would import them into the scored log."""
        cursor = None if self._fills_cursor is None else self._fills_cursor - FILL_OVERLAP
        genesis = self._app.genesis
        # Strictly after the genesis: a plumbing fill can carry the genesis's own timestamp.
        floor = genesis.created_at + GENESIS_FILL_GAP if genesis is not None else None
        if full:
            since = floor if floor is not None else now - FULL_HISTORY_LOOKBACK
            if floor is None and cursor is not None:
                since = min(since, cursor)
        else:
            since = cursor if cursor is not None else now - FULL_HISTORY_LOOKBACK
        return max(since, floor) if floor is not None else since

    def _sync_stops(self, demo: dict[str, Quote]) -> None:
        app = self._app
        stops = app.stops
        if stops is None:
            return
        book = app.book(at=app.clock.now(), demo=demo)
        try:
            venue_stops = app.transport.stop_orders()
        except EnvironmentRefused:
            raise
        except Exception as exc:  # a failed read is logged; reconciliation reports the gap
            app.note(
                f"stop sync skipped: venue stop orders unreadable: {type(exc).__name__}: {exc}"
            )
            return
        try:
            venue_positions = app.transport.positions()
        except EnvironmentRefused:
            raise
        except Exception as exc:  # stops are still placed; none is cancelled on an unread venue
            app.note(
                "stop sync: venue positions unreadable, so no stop is cancelled this sync: "
                f"{type(exc).__name__}: {exc}"
            )
            venue_positions = None
        if not venue_stops and not app.held_symbols() and not venue_positions:
            return
        positions = book.positions
        for sync in stops.sync(book, venue_stops, venue_positions=venue_positions):
            held = positions.get(sync.symbol)
            if (
                sync.action == "verified"
                and held is not None
                and held.stop_venue_id == sync.venue_id
                and held.stop_price == sync.stop_price
            ):
                continue  # the ledger already records exactly this stop on this position
            app.log(EventKind.STOP_SYNC, sync, blobs=[sync.blob] if sync.blob else [])
        for error in stops.errors():
            app.note(
                f"stop {error.action} for {error.symbol} failed"
                + (" (outcome unknown)" if error.outcome_unknown else "")
                + f": {error.message}"
            )

    # ============================================================================================
    # Marks, export, anchors, probe, health
    # ============================================================================================

    def _anchor_mark(self, now: datetime, did: list[str]) -> None:
        """The first MARK of a record: the starting equity on the current hour, before any fill.

        Every metric is computed between consecutive MARKs (``analysis/metrics.py``), so a fill
        before the first MARK would leave its fee and its P&L to that mark outside total return,
        Sharpe, drawdown and every interval: usually the largest cost of a run, the initial book
        build. This point anchors the series at the equity the book starts from, while the book is
        still flat, and :meth:`_admit` refuses every decision until a MARK exists.
        ``scripts/recompute.py`` requires the first MARK to be exactly this.

        The point is stamped on the hour the first tick falls in. The book has no position and no
        fill, so its equity on that hour is the starting equity, whatever the minute. A ledger
        that already holds fills but no MARK (a record from before this rule) gets no anchor: its
        first regular hourly mark stands, and nothing here pretends otherwise."""
        app = self._app
        projection = app.projection
        if self._last_mark_hour is not None or projection.marks:
            return
        start = projection.starting_equity
        if start is None or projection.fills:
            return
        hour = hour_floor(now)
        point = MarkPoint(
            at=hour,
            equity_book=start,
            equity_venue=None,
            equity_live_mirror=start,
            gross_weight=0.0,
            net_weight=0.0,
            positions=(),
        )
        app.log(EventKind.MARK, point)
        self._last_mark_hour = hour
        did.append("anchor_mark")
        self._export(did)

    def _mark(self, now: datetime, did: list[str]) -> None:
        app = self._app
        hour = hour_floor(now)
        if self._last_mark_hour is not None and hour <= self._last_mark_hour:
            return
        if now - hour > MARK_MAX_LATENESS:
            return
        if app.projection.starting_equity is None:
            return
        demo, live = app.quotes()
        venue_equity = None
        if app.mode is RunMode.PAPER:
            try:
                venue_equity = app.transport.account().equity_usdt
            except EnvironmentRefused:
                raise
            except Exception as exc:  # the venue equity is published beside the book, not needed
                app.note(f"venue equity unreadable for the {hour:%H:%M} mark: {exc}")
        try:
            point = mark_point(
                app.projection.builder(), at=now, demo=demo, live=live, venue_equity=venue_equity
            )
        except MarkError as exc:
            if self._mark_problem_hour != hour:
                app.note(f"hourly mark for {hour:%Y-%m-%dT%H:%M}Z not taken yet: {exc}")
                self._mark_problem_hour = hour
            return
        app.log(EventKind.MARK, point)
        self._last_mark_hour = point.at
        did.append("mark")
        self._export(did)

    def _export(self, did: list[str]) -> None:
        """Publish the record after a MARK (the exporter the CLI sets); a failure is a note."""
        app = self._app
        if self.exporter is not None:
            try:
                for text in self.exporter(app):
                    app.note(text[:2000])
                did.append("export")
            except (EnvironmentRefused, LedgerError):
                raise
            except Exception as exc:  # a failed export never stops trading; the next hour retries
                app.note(f"hourly export failed: {type(exc).__name__}: {exc}"[:600])

    def _anchor(self, now: datetime, did: list[str]) -> None:
        app = self._app
        day = now.date().isoformat()
        if now.time() >= DAILY_ANCHOR_AFTER and self._last_anchor_day != day:
            head = app.chain.head()
            if head is not None:
                record = stamp(
                    head,
                    runner=app.ots_runner,
                    blobs=app.blobs,
                    workdir=app.paths.anchors,
                    clock=app.clock,
                )
                app.log(EventKind.ANCHOR, record)
                did.append(f"anchor_{record.status}")
            self._last_anchor_day = day
        if _due(self._last_upgrade, now, ANCHOR_UPGRADE_EVERY):
            self._last_upgrade = now
            for record in _pending_anchors(app.projection.anchors):
                upgraded = upgrade(
                    record,
                    runner=app.ots_runner,
                    blobs=app.blobs,
                    workdir=app.paths.anchors,
                    clock=app.clock,
                )
                if upgraded.status != record.status or upgraded.ots_blob != record.ots_blob:
                    app.log(EventKind.ANCHOR, upgraded)
                    did.append(f"anchor_{upgraded.status}")

    def _probe(self, now: datetime, did: list[str]) -> None:
        """The daily measurement of every Bitget toolkit surface (DESIGN.md §14.8).

        It calls every catalog entry of both services, which takes up to a minute, so it runs on a
        worker thread and its result is logged by the first tick after it finishes: the 60-second
        protective check never waits on it. The worker only reads; the ledger is written here."""
        app = self._app
        worker = self._probe_worker
        if worker is not None and not worker.is_alive():
            self._probe_worker = None
            result = self._probe_result
            self._probe_result = None
            if isinstance(result, ToolkitProbe):
                app.log(EventKind.TOOLKIT_PROBE, result)
                did.append("toolkit_probe")
            elif result is not None:
                app.note(f"toolkit probe failed: {result}")
        toolkit = app.toolkit
        if (
            not self._probe_enabled
            or not isinstance(toolkit, ToolkitFacade)
            or self._probe_worker is not None
        ):
            return
        if not _due(self._last_probe, now, TOOLKIT_PROBE_EVERY):
            return
        self._last_probe = now

        def measure() -> None:
            try:
                self._probe_result = probe_all(toolkit, universe=app.symbols, clock=app.clock)
            except Exception as exc:  # reported as a note by the next tick
                self._probe_result = f"{type(exc).__name__}: {exc}"

        self._probe_worker = threading.Thread(target=measure, name="t2sa-probe", daemon=True)
        self._probe_worker.start()
        did.append("toolkit_probe_started")

    def wait_for_probe(self, timeout_s: float = 300.0) -> None:
        """Let a running toolkit probe finish (at shutdown, and in tests)."""
        if self._probe_worker is not None:
            self._probe_worker.join(timeout_s)

    def _health(self, now: datetime, did: list[str]) -> None:
        app = self._app
        beat = beat_for(
            app,
            iteration=self._iteration,
            last_decision_at=self._last_decision_at,
            detail=summary_line(self._feeds),
        )
        write_health(app.paths.health, beat)
        if _due(self._last_health_event, now, HEALTH_EVENT_EVERY):
            app.log(EventKind.HEALTH, beat)
            self._last_health_event = now
            did.append("health_event")

    # ============================================================================================
    # The card
    # ============================================================================================

    def _card(
        self,
        record: DecisionRecord,
        snapshot: PerceptionSnapshot,
        triggers: Sequence[Trigger],
        acted: _Acted | None,
        first: int,
        feeds: FeedHealthReport | None,
    ) -> DecisionCard:
        app = self._app
        events = app.ledger.appended[first:]
        refs: dict[str, BlobRef] = {}
        for event in events:
            for ref in referenced_blobs(event):
                refs.setdefault(ref.sha256, ref)
        previews = tuple(
            p
            for p in app.projection.previews
            if acted is not None and any(o.client_oid == p.client_oid for o in acted.orders)
        )
        return DecisionCard(
            card_id="card-" + record.decision_id,
            at=record.decided_at,
            decision_id=record.decision_id,
            ruling_id=None if acted is None else acted.ruling.ruling_id,
            triggers=tuple(triggers),
            coverage=snapshot.coverage(),
            shown_text=snapshot.text,
            outcome=record.outcome,
            decision=record.decision,
            grounding=dict(record.grounding),
            kernel=None if acted is None else acted.ruling,
            previews=previews,
            orders=() if acted is None else tuple(acted.orders),
            ledger_seqs=tuple(e.seq for e in events),
            blobs=tuple(refs.values()),
            feed_health=feeds,
        )


def _first_seq_index(appended: Sequence[LedgerEvent], snapshot_id: str) -> int:
    """Where the SNAPSHOT event of ``snapshot_id`` sits in ``appended`` (the card's first
    event); the end of the list when it is not there."""
    for i in range(len(appended) - 1, -1, -1):
        event = appended[i]
        if event.kind is EventKind.SNAPSHOT and (
            event.payload.get("snapshot", {}).get("snapshot_id") == snapshot_id
        ):
            return i
    return len(appended)


def _due(last: datetime | None, now: datetime, every: timedelta) -> bool:
    return last is None or now - last >= every


def _pending_anchors(anchors: Sequence[AnchorRecord]) -> list[AnchorRecord]:
    """The newest record per anchored event, when that record is still pending."""
    newest: dict[int, AnchorRecord] = {}
    for record in anchors:
        newest[record.target_seq] = record
    return [r for r in newest.values() if r.status == "submitted" and r.ots_blob is not None]


def _protective_detail(ruling: KernelRuling) -> str:
    parts: list[str] = []
    for inst in ruling.instruments:
        if not inst.changed_by_kernel:
            continue
        guard = inst.binding_guard.value if inst.binding_guard is not None else "none"
        parts.append(
            f"{inst.symbol} {inst.current_weight:+.4%} -> {inst.approved_weight:+.4%} ({guard})"
        )
    return "; ".join(parts) or "no instrument changed"


__all__ = [
    "DAILY_ANCHOR_AFTER",
    "DAILY_RECONCILE_AFTER",
    "FILL_OVERLAP",
    "LIGHT_SNAPSHOT_EVERY",
    "MAX_CONSECUTIVE_FAILURES",
    "PROTECTIVE_EVERY",
    "RECONCILE_EVERY",
    "RunLoop",
    "TickReport",
    "owner_trigger",
    "request_decision",
]
