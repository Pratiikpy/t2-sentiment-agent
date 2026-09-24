"""The ledger, read back: every event kind as typed records, and the book rebuilt from them.

Nothing the agent needs after a restart lives only in memory (DESIGN.md §4). The runtime builds a
:class:`Projection` from the ledger at start, then feeds it each event it appends
(:meth:`Projection.apply`), so the running state and a replay of the log are one code path and
cannot drift. Typed views of every event kind serve the modules that restore their own state
(breaker transitions, token budget, trigger history, order states) and the ones that publish
(cards, analysis).

**The book is rebuilt from the log alone.**

* **Scope.** When the ledger has a genesis, nothing logged before it enters the book (a pre-genesis
  plumbing order is disclosed, never scored, DESIGN.md §16). The typed views still show the whole
  log.
* **Starting equity** is the venue's account equity as first recorded in scope before any fill:
  the account read of a *passed* environment proof, or of a reconciliation (DESIGN.md §13: "the
  venue's account equity at genesis"). A failed proof's account read is never used, because a
  proof fails exactly when the key may not be a Demo key. In a PAPER ledger nothing else is
  accepted; SIMULATED and DRYRUN runs, whose venue may never report an account, may pass a
  ``starting_equity`` for the case where their ledger records none.
* **Attribution.** A fill is matched to the order intent it executed by ``clientOid`` (from
  ``ORDER_PLAN`` and ``ORDER_SUBMITTED``), which gives the book its decision id and purpose; the
  intent's ruling gives a protective exit its cause (the protective reason, or the guard that forced
  the exit inside a decision). A fill whose order the agent did not plan is venue-originated; when
  its order id is a stop the ledger recorded (``STOP_SYNC``), it is booked as the stop firing,
  otherwise as ``venue_initiated``. Nothing is inferred from prices.
* **Stops.** A filled exposure-adding intent sets its preset stop on the position (once per order,
  so a later fill of the same order cannot overwrite a newer stop); every ``STOP_SYNC`` after it
  updates the stop. **Marks** are fed back from the ``MARK`` events for day-open and peak equity.
* **Activation** on the returned book is the breaker's last logged transition as of the requested
  time; a transition that does not start where the previous one ended makes it ``HALTED``, the safe
  end. The breaker restores its own full state (``kernel/breaker.py``).

The projection refuses a log it cannot read honestly: events out of ``seq`` order, more than one run
mode in one file, a simulated fill in a PAPER ledger (or a Demo fill in a simulated one, or any fill
in a DRYRUN ledger), a fill booked under one of the agent's ``clientOid``s with a symbol or side
that contradicts the intent.

**A contract gap, stated here so it is not lost.** Bitget reports the ``delegateType`` of the order
behind a fill only on the order (``VenueOrder.delegate_type``), and the ledger has no event that
carries a venue-originated order. So a stop fill is recognised from the ledger only by the venue
id match above. :meth:`BookBuilder.apply_fill` already classifies by ``delegateType``; the day a
fill carries it, the projection passes it through.
"""

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Final, TypeVar, cast

from sentiment_agent.book.book import (
    FILL_CLOCK_TOLERANCE,
    BookBuilder,
    BookError,
    Cause,
    require_utc,
)
from sentiment_agent.types import (
    Activation,
    Amendment,
    AnchorRecord,
    BookState,
    BreakerTransition,
    BudgetState,
    ClosedTrade,
    DecisionEvent,
    DecisionRecord,
    DryRunPreview,
    EnvironmentProof,
    EventKind,
    Fill,
    FillVenue,
    Genesis,
    HealthBeat,
    KernelRuling,
    LedgerEvent,
    LedgerReader,
    MarkPoint,
    Model,
    Note,
    OrderIntent,
    OrderPlan,
    OrderPurpose,
    OrderStateChange,
    OrderSubmitted,
    PerceptionSnapshot,
    Policy,
    PriceSource,
    ProtectiveAction,
    ProtectiveReason,
    ReconciliationReport,
    RunMode,
    SnapshotEvent,
    StopSync,
    ToolkitProbe,
    Trigger,
    VenueAck,
    VenueRejection,
    VenueUnknown,
    parse_payload,
)

M = TypeVar("M", bound=Model)

OrderEvent = (
    DryRunPreview | OrderSubmitted | VenueAck | VenueRejection | VenueUnknown | OrderStateChange
)
"""The payload of every order-lifecycle event, in ledger order."""

ORDER_EVENT_KINDS: Final[frozenset[EventKind]] = frozenset(
    {
        EventKind.ORDER_PREVIEW,
        EventKind.ORDER_SUBMITTED,
        EventKind.ORDER_ACK,
        EventKind.ORDER_REJECTED,
        EventKind.ORDER_UNKNOWN,
        EventKind.ORDER_STATE,
    }
)

_FILL_VENUE: Final[Mapping[RunMode, FillVenue]] = {
    RunMode.PAPER: FillVenue.BITGET_DEMO,
    RunMode.SIMULATED: FillVenue.SIMULATED,
}
"""The only venue whose fills each mode's ledger may hold. DRYRUN sends nothing, so it has none."""

_STOP_IDS_FROM: Final = frozenset({"preset", "placed", "replaced", "verified"})


class ProjectionError(BookError):
    """A ledger the projection cannot read honestly."""


@dataclass(frozen=True, slots=True)
class _FillInput:
    fill: Fill
    decision_id: str | None
    purpose: OrderPurpose | None
    cause: Cause | None
    preset_stop: StopSync | None


_BookInput = _FillInput | StopSync | MarkPoint


def _feed(builder: BookBuilder, item: _BookInput) -> None:
    if isinstance(item, _FillInput):
        builder.apply_fill(
            item.fill, decision_id=item.decision_id, purpose=item.purpose, cause=item.cause
        )
        if item.preset_stop is not None:
            builder.apply_stop_sync(item.preset_stop)
    elif isinstance(item, StopSync):
        builder.apply_stop_sync(item)
    else:
        builder.record_mark(item)


def _input_time(item: _BookInput) -> datetime:
    if isinstance(item, _FillInput):
        return item.fill.executed_at
    return item.at


class Projection:
    """Typed views of a ledger and the book rebuilt from it. Build with :meth:`from_ledger`; keep
    current with :meth:`apply`. One reader per process, like the ledger has one writer: it is not
    safe to apply events from two threads at once."""

    def __init__(self, policy: Policy, *, starting_equity: Decimal | None = None) -> None:
        self._policy = policy
        self._fallback_equity = starting_equity
        self._events: list[LedgerEvent] = []
        self._payloads: dict[EventKind, list[Model]] = {kind: [] for kind in EventKind}
        self._order_events: list[OrderEvent] = []
        self._mode: RunMode | None = None
        self._head: int | None = None
        self._genesis: Genesis | None = None
        self._intents: dict[str, OrderIntent] = {}
        self._rulings: dict[str, KernelRuling] = {}
        self._decisions: dict[str, DecisionRecord] = {}
        self._reset_book_scope()

    def _reset_book_scope(self) -> None:
        self._inputs: list[_BookInput] = []
        self._recorded_equity: Decimal | None = None
        self._fill_seen = False
        self._stop_venue_ids: set[str] = set()
        self._presets_applied: set[str] = set()
        self._builder: BookBuilder | None = None
        self._fed = 0

    # --- construction -----------------------------------------------------------------------

    @classmethod
    def from_ledger(
        cls,
        reader: LedgerReader,
        policy: Policy,
        *,
        starting_equity: Decimal | None = None,
    ) -> "Projection":
        """Replay every event of ``reader`` in ``seq`` order.

        ``starting_equity`` is a fallback for SIMULATED and DRYRUN ledgers that record no account
        read before their first fill; a recorded account read always wins, and a PAPER ledger never
        uses the fallback (module docstring)."""
        projection = cls(policy, starting_equity=starting_equity)
        for event in reader.events():
            projection.apply(event)
        return projection

    def apply(self, event: LedgerEvent) -> None:
        """Add the next ledger event. Events must arrive in increasing ``seq`` order."""
        if self._head is not None and event.seq <= self._head:
            raise ProjectionError(
                f"event seq {event.seq} arrived after seq {self._head}; the ledger is read in order"
            )
        if self._mode is None:
            self._mode = event.mode
        elif event.mode is not self._mode:
            raise ProjectionError(
                f"event seq {event.seq} is {event.mode.value} in a {self._mode.value} ledger; "
                "modes never share a ledger (DESIGN.md §5)"
            )
        payload = parse_payload(event)
        self._dispatch(event, payload)
        self._events.append(event)
        self._payloads[event.kind].append(payload)
        if event.kind in ORDER_EVENT_KINDS:
            self._order_events.append(cast(OrderEvent, payload))
        self._head = event.seq

    # --- handlers ---------------------------------------------------------------------------

    def _dispatch(self, event: LedgerEvent, payload: Model) -> None:
        kind = event.kind
        if kind is EventKind.GENESIS:
            self._on_genesis(cast(Genesis, payload))
        elif kind is EventKind.ENVIRONMENT_PROOF:
            self._on_environment_proof(cast(EnvironmentProof, payload))
        elif kind is EventKind.RECONCILIATION:
            self._on_reconciliation(cast(ReconciliationReport, payload))
        elif kind is EventKind.DECISION:
            record = cast(DecisionEvent, payload).record
            self._decisions.setdefault(record.decision_id, record)
        elif kind is EventKind.KERNEL_RULING:
            ruling = cast(KernelRuling, payload)
            self._rulings.setdefault(ruling.ruling_id, ruling)
        elif kind is EventKind.ORDER_PLAN:
            for intent in cast(OrderPlan, payload).intents:
                self._note_intent(intent)
        elif kind is EventKind.ORDER_SUBMITTED:
            self._note_intent(cast(OrderSubmitted, payload).intent)
        elif kind is EventKind.STOP_SYNC:
            self._on_stop_sync(cast(StopSync, payload))
        elif kind is EventKind.MARK:
            self._add_input(cast(MarkPoint, payload))
        elif kind is EventKind.FILL:
            self._on_fill(event.mode, cast(Fill, payload))

    def _on_genesis(self, genesis: Genesis) -> None:
        if self._genesis is None:
            self._genesis = genesis
            self._reset_book_scope()

    def _note_account(self, equity: Decimal | None) -> None:
        if self._fill_seen or self._recorded_equity is not None:
            return
        if equity is not None and equity.is_finite() and equity > 0:
            self._recorded_equity = equity
            if self._builder is not None and self._builder.starting_equity != equity:
                # Built from the fallback before the log said otherwise; no fill has been booked,
                # so rebuilding from the recorded equity loses nothing and matches a replay.
                self._builder = None
                self._fed = 0

    def _on_environment_proof(self, proof: EnvironmentProof) -> None:
        if proof.passed and proof.account is not None:
            self._note_account(proof.account.equity_usdt)

    def _on_reconciliation(self, report: ReconciliationReport) -> None:
        if report.account is not None:
            self._note_account(report.account.equity_usdt)

    def _note_intent(self, intent: OrderIntent) -> None:
        known = self._intents.get(intent.client_oid)
        if known is not None and known != intent:
            raise ProjectionError(
                f"clientOid {intent.client_oid} names two different intents in the ledger"
            )
        self._intents[intent.client_oid] = intent

    def _on_stop_sync(self, sync: StopSync) -> None:
        if sync.venue_id is not None and sync.action in _STOP_IDS_FROM:
            self._stop_venue_ids.add(sync.venue_id)
        self._add_input(sync)

    def _on_fill(self, mode: RunMode, fill: Fill) -> None:
        expected = _FILL_VENUE.get(mode)
        if expected is None:
            raise ProjectionError(
                f"fill {fill.exec_id} in a {mode.value} ledger, which sends nothing"
            )
        if fill.venue is not expected:
            raise ProjectionError(
                f"fill {fill.exec_id} is from {fill.venue.value}; a {mode.value} ledger "
                f"holds only {expected.value} fills"
            )
        intent = self._intents.get(fill.client_oid) if fill.client_oid else None
        decision_id: str | None = None
        purpose: OrderPurpose | None = None
        cause: Cause | None = None
        preset: StopSync | None = None
        if intent is not None:
            if intent.symbol != fill.symbol or intent.side is not fill.side:
                raise ProjectionError(
                    f"fill {fill.exec_id} ({fill.symbol} {fill.side.value}) was booked under "
                    f"clientOid {intent.client_oid}, whose intent is {intent.symbol} "
                    f"{intent.side.value}"
                )
            decision_id = intent.decision_id
            purpose = intent.purpose
            cause = self._cause_of(intent)
            if (
                intent.purpose.adds_exposure
                and intent.stop_loss_price is not None
                and intent.client_oid not in self._presets_applied
            ):
                self._presets_applied.add(intent.client_oid)
                preset = StopSync(
                    symbol=fill.symbol,
                    action="preset",
                    stop_price=intent.stop_loss_price,
                    venue_id=None,
                    at=fill.executed_at,
                )
        elif fill.venue_order_id in self._stop_venue_ids:
            cause = ProtectiveReason.STOP_FILLED
        self._fill_seen = True
        self._add_input(
            _FillInput(
                fill=fill, decision_id=decision_id, purpose=purpose, cause=cause, preset_stop=preset
            )
        )

    def _cause_of(self, intent: OrderIntent) -> Cause | None:
        ruling = self._rulings.get(intent.ruling_id)
        if ruling is None:
            return None
        if ruling.protective_reason is not None:
            return ruling.protective_reason
        if intent.purpose is OrderPurpose.PROTECTIVE_EXIT:
            instrument = ruling.instrument(intent.symbol)
            if instrument is not None:
                return instrument.binding_guard
        return None

    # --- the book ---------------------------------------------------------------------------

    def _add_input(self, item: _BookInput) -> None:
        self._inputs.append(item)
        if self._builder is not None:
            _feed(self._builder, item)
            self._fed = len(self._inputs)

    @property
    def starting_equity(self) -> Decimal | None:
        """The equity the book starts from, or ``None`` while the log cannot say."""
        if self._recorded_equity is not None:
            return self._recorded_equity
        if self._mode is RunMode.PAPER:
            return None
        return self._fallback_equity

    def _require_starting_equity(self) -> Decimal:
        equity = self.starting_equity
        if equity is not None:
            return equity
        if self._mode is RunMode.PAPER:
            raise ProjectionError(
                "the PAPER ledger records no account equity (a passed environment proof or a "
                "reconciliation) before its first fill, so the book has no starting equity"
            )
        raise ProjectionError(
            "the ledger records no account equity before its first fill and no fallback "
            "starting equity was given"
        )

    def builder(self) -> BookBuilder:
        """The book over every fill in scope, fed as events arrive."""
        if self._builder is None:
            self._builder = BookBuilder(
                starting_equity=self._require_starting_equity(), policy=self._policy
            )
            self._fed = 0
        for item in self._inputs[self._fed :]:
            _feed(self._builder, item)
        self._fed = len(self._inputs)
        return self._builder

    def _builder_as_of(self, at: datetime) -> BookBuilder:
        main = self.builder()
        last = main.last_fill_at
        if last is None or last <= at + FILL_CLOCK_TOLERANCE:
            return main
        past = BookBuilder(starting_equity=main.starting_equity, policy=self._policy)
        for item in self._inputs:
            if _input_time(item) <= at:
                _feed(past, item)
        return past

    def book(
        self, *, at: datetime, marks: Mapping[str, Decimal], mark_source: PriceSource
    ) -> BookState:
        """The book at ``at`` from the fills executed by then, valued at ``marks``."""
        at = require_utc(at, what="at")
        return self._builder_as_of(at).state(
            at=at, marks=marks, mark_source=mark_source, activation=self.activation(at)
        )

    @property
    def closed_trades(self) -> tuple[ClosedTrade, ...]:
        return self.builder().closed_trades()

    def activation(self, at: datetime | None = None) -> Activation:
        """The breaker's activation as of ``at`` (all transitions when ``None``), from the log."""
        state = Activation.ACTIVE
        last_at: datetime | None = None
        for transition in self.breaker_transitions:
            if at is not None and transition.at > at:
                break
            if transition.from_state is not state or (
                last_at is not None and transition.at < last_at
            ):
                return Activation.HALTED
            state = transition.to_state
            last_at = transition.at
        return state

    # --- typed views ------------------------------------------------------------------------

    def _typed(self, kind: EventKind, model: type[M]) -> tuple[M, ...]:
        items = self._payloads[kind]
        for item in items:
            if not isinstance(item, model):  # pragma: no cover - parse_payload guarantees it
                raise ProjectionError(f"{kind.value} payload is not a {model.__name__}")
        return tuple(cast(list[M], items))

    def events(self, kinds: Iterable[EventKind] | None = None) -> tuple[LedgerEvent, ...]:
        """The raw events, in ``seq`` order (for modules that restore from events, such as
        ``OrderTracker.restore``)."""
        if kinds is None:
            return tuple(self._events)
        wanted = frozenset(kinds)
        return tuple(e for e in self._events if e.kind in wanted)

    def __iter__(self) -> Iterator[LedgerEvent]:
        return iter(self._events)

    def __len__(self) -> int:
        return len(self._events)

    @property
    def mode(self) -> RunMode | None:
        return self._mode

    @property
    def head_seq(self) -> int | None:
        return self._head

    @property
    def policy(self) -> Policy:
        return self._policy

    @property
    def genesis(self) -> Genesis | None:
        return self._genesis

    @property
    def amendments(self) -> tuple[Amendment, ...]:
        return self._typed(EventKind.AMENDMENT, Amendment)

    @property
    def environment_proofs(self) -> tuple[EnvironmentProof, ...]:
        return self._typed(EventKind.ENVIRONMENT_PROOF, EnvironmentProof)

    @property
    def toolkit_probes(self) -> tuple[ToolkitProbe, ...]:
        return self._typed(EventKind.TOOLKIT_PROBE, ToolkitProbe)

    @property
    def snapshots(self) -> tuple[PerceptionSnapshot, ...]:
        return tuple(s.snapshot for s in self._typed(EventKind.SNAPSHOT, SnapshotEvent))

    @property
    def triggers(self) -> tuple[Trigger, ...]:
        return self._typed(EventKind.TRIGGER, Trigger)

    @property
    def decisions(self) -> tuple[DecisionRecord, ...]:
        return tuple(d.record for d in self._typed(EventKind.DECISION, DecisionEvent))

    @property
    def rulings(self) -> tuple[KernelRuling, ...]:
        return self._typed(EventKind.KERNEL_RULING, KernelRuling)

    @property
    def plans(self) -> tuple[OrderPlan, ...]:
        return self._typed(EventKind.ORDER_PLAN, OrderPlan)

    @property
    def order_events(self) -> tuple[OrderEvent, ...]:
        """Previews, submissions, acknowledgements, rejections, unknowns and state changes."""
        return tuple(self._order_events)

    @property
    def previews(self) -> tuple[DryRunPreview, ...]:
        return self._typed(EventKind.ORDER_PREVIEW, DryRunPreview)

    @property
    def submissions(self) -> tuple[OrderSubmitted, ...]:
        return self._typed(EventKind.ORDER_SUBMITTED, OrderSubmitted)

    @property
    def acks(self) -> tuple[VenueAck, ...]:
        return self._typed(EventKind.ORDER_ACK, VenueAck)

    @property
    def rejections(self) -> tuple[VenueRejection, ...]:
        return self._typed(EventKind.ORDER_REJECTED, VenueRejection)

    @property
    def unknowns(self) -> tuple[VenueUnknown, ...]:
        return self._typed(EventKind.ORDER_UNKNOWN, VenueUnknown)

    @property
    def order_states(self) -> tuple[OrderStateChange, ...]:
        return self._typed(EventKind.ORDER_STATE, OrderStateChange)

    @property
    def fills(self) -> tuple[Fill, ...]:
        return self._typed(EventKind.FILL, Fill)

    @property
    def stop_syncs(self) -> tuple[StopSync, ...]:
        return self._typed(EventKind.STOP_SYNC, StopSync)

    @property
    def protective_actions(self) -> tuple[ProtectiveAction, ...]:
        return self._typed(EventKind.PROTECTIVE_ACTION, ProtectiveAction)

    @property
    def breaker_transitions(self) -> tuple[BreakerTransition, ...]:
        return self._typed(EventKind.BREAKER_TRANSITION, BreakerTransition)

    @property
    def reconciliations(self) -> tuple[ReconciliationReport, ...]:
        return self._typed(EventKind.RECONCILIATION, ReconciliationReport)

    @property
    def marks(self) -> tuple[MarkPoint, ...]:
        return self._typed(EventKind.MARK, MarkPoint)

    @property
    def budget_states(self) -> tuple[BudgetState, ...]:
        return self._typed(EventKind.BUDGET_STATE, BudgetState)

    @property
    def anchors(self) -> tuple[AnchorRecord, ...]:
        return self._typed(EventKind.ANCHOR, AnchorRecord)

    @property
    def health_beats(self) -> tuple[HealthBeat, ...]:
        return self._typed(EventKind.HEALTH, HealthBeat)

    @property
    def notes(self) -> tuple[Note, ...]:
        return self._typed(EventKind.NOTE, Note)

    # --- lookups ----------------------------------------------------------------------------

    def intent(self, client_oid: str) -> OrderIntent | None:
        return self._intents.get(client_oid)

    def ruling(self, ruling_id: str) -> KernelRuling | None:
        return self._rulings.get(ruling_id)

    def decision(self, decision_id: str) -> DecisionRecord | None:
        return self._decisions.get(decision_id)


__all__ = ["ORDER_EVENT_KINDS", "OrderEvent", "Projection", "ProjectionError"]
