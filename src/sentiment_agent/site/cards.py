"""Decision cards: everything a judge needs to check one decision, computed from the ledger alone.

One :class:`~sentiment_agent.types.DecisionCard` per logged decision (a trade, a hold, a written
abstention, or a model outage) and one per protective ruling (the kernel acting on its own clock:
daily kill, weekend freeze, venue integrity, breaker, model outage). A card is never written beside
the record; it is read out of it:

* **Why the agent woke up**: the ``TRIGGER`` events named by the decision, as logged.
* **What it saw**: the ``SNAPSHOT`` the decision names by id, with the health of every source call
  (``coverage``) and every text item exactly as the model received it, withheld items shown as
  withheld (``shown_text``).
* **What it decided**: the validated model answer (thesis, invalidation, crowd belief against our
  view, rejected alternatives, the mandate response or the written reasons for staying flat) and the
  grounding report for every number in it.
* **What the kernel did**: the ``KERNEL_RULING`` that answers the decision (or the protective
  ruling itself), with every guard's status, ceiling, reason, inputs and measured basis.
* **What was sent and what came back**: the ``ORDER_PLAN`` under that ruling, each intent's dry-run
  preview (the Agent Hub payload and argv), its ``clientOid``, the venue ``orderId`` from the
  acknowledgement, its latest order state, its fills and the net P&L they realised.
* **Proof**: the ledger sequence number of every event the card was read from, and every blob those
  events commit to. Each blob is fetched from the store and re-hashed while the card is built, so a
  card cannot cite raw material that is missing or altered.

A card that cannot be built honestly is not built: a decision that names a trigger or snapshot the
ledger does not hold, or a proof blob the store cannot produce, raises :class:`CardError`. The
export refuses to publish over it rather than publish a card that asserts more than the log shows.

**Net P&L per order** (``CardOrder.net_pnl``) is the price P&L realised on the quantity the order's
fills closed, less every fee those fills paid; an order that only opened or added has realised
nothing yet but its fees. ``None`` when the order has no fill. The fold that measures what a fill
closed is the book's own definition (``book/book.py`` ``_apply``: fills in venue time order, a
closing fill before an opening one at the same instant, then ledger order; average cost; a fill that
crosses zero closes the old side at its price and opens the rest), reimplemented here because the
book reports P&L per trade, not per fill. ``tests/site/test_cards.py`` checks that the two agree:
summed over every order of a book that ends flat, the per-order figures equal the closed trades'
net P&L.

Nothing here reads the network, a clock or a credential.
"""

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from typing import Final

from sentiment_agent.book.book import DECIMAL_CONTEXT
from sentiment_agent.book.projection import Projection
from sentiment_agent.hashing import sha256_hex
from sentiment_agent.ledger.chain import LedgerError, referenced_blobs
from sentiment_agent.types import (
    BlobRef,
    BlobStore,
    CardOrder,
    DecisionCard,
    DecisionEvent,
    DecisionRecord,
    DryRunPreview,
    EventKind,
    FeedHealthReport,
    Fill,
    KernelRuling,
    LedgerEvent,
    Model,
    OrderIntent,
    OrderPlan,
    OrderState,
    OrderStateChange,
    OrderSubmitted,
    PerceptionSnapshot,
    ProtectiveAction,
    ScreenedItem,
    Side,
    SnapshotEvent,
    SourceHealth,
    Trigger,
    VenueAck,
    VenueRejection,
    VenueUnknown,
    parse_payload,
)

CARD_ID: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
"""A card id is also a file name (``cards/<id>.json``, ``cards/<id>.html``): letters, digits,
``-`` and ``_`` only, so no id read from a log can reach outside the ``cards`` folder."""

DECISION_CARD_PREFIX: Final = "dec-"
PROTECTIVE_CARD_PREFIX: Final = "prot-"
_ID_HEX: Final = 32

_ORDER_KINDS: Final = frozenset(
    {
        EventKind.ORDER_PREVIEW,
        EventKind.ORDER_SUBMITTED,
        EventKind.ORDER_ACK,
        EventKind.ORDER_REJECTED,
        EventKind.ORDER_UNKNOWN,
        EventKind.ORDER_STATE,
    }
)


class CardError(RuntimeError):
    """A card that cannot be built from the ledger without asserting more than the ledger holds."""


def decision_card_id(decision_id: str) -> str:
    """The card id of a decision: the decision id itself when it is a safe file name (the agent's
    ids are ``dec-`` + 32 hex), otherwise ``dec-`` + the first 32 hex of its SHA-256."""
    if CARD_ID.fullmatch(decision_id):
        return decision_id
    return DECISION_CARD_PREFIX + sha256_hex(decision_id.encode("utf-8"))[:_ID_HEX]


def protective_card_id(ruling_id: str) -> str:
    """The card id of a protective ruling: ``prot-`` + the first 32 hex of the ruling id (the
    kernel's ids are content hashes), or of the id's own SHA-256 when it is not hex."""
    if re.fullmatch(r"[0-9a-f]{32,}", ruling_id):
        return PROTECTIVE_CARD_PREFIX + ruling_id[:_ID_HEX]
    return PROTECTIVE_CARD_PREFIX + sha256_hex(ruling_id.encode("utf-8"))[:_ID_HEX]


# ================================================================================================
# Per-fill realised P&L
# ================================================================================================


def realized_pnl_by_fill(fills: Sequence[Fill]) -> dict[str, Decimal]:
    """Price P&L each fill realised, by ``exec_id``, fees excluded.

    ``fills`` are in ledger (arrival) order; they are folded per symbol in venue time order, a
    closing fill before an opening one at the same instant, then arrival, exactly as the book folds
    them (``book/book.py``). A fill that only opens or adds realises zero. A fill seen twice (the
    same ``exec_id``) is counted once.
    """
    by_symbol: dict[str, list[tuple[int, Fill]]] = defaultdict(list)
    seen: set[str] = set()
    for arrival, fill in enumerate(fills):
        if fill.exec_id in seen:
            continue
        seen.add(fill.exec_id)
        by_symbol[fill.symbol].append((arrival, fill))
    realised: dict[str, Decimal] = {}
    with localcontext(DECIMAL_CONTEXT):
        for records in by_symbol.values():
            records.sort(
                key=lambda r: (r[1].executed_at, 0 if r[1].trade_side == "close" else 1, r[0])
            )
            qty = Decimal(0)
            avg = Decimal(0)
            for _, fill in records:
                direction = 1 if fill.side is Side.BUY else -1
                amount = fill.exec_qty
                price = fill.exec_price
                pnl = Decimal(0)
                remaining = amount
                if qty != 0 and (qty > 0) != (direction > 0):
                    held_direction = 1 if qty > 0 else -1
                    closing = min(amount, abs(qty))
                    pnl = (price - avg) * closing * held_direction
                    qty -= closing * held_direction
                    remaining = amount - closing
                    if qty == 0:
                        avg = Decimal(0)
                if remaining > 0:
                    if qty == 0:
                        avg = price
                        qty = remaining * direction
                    else:
                        held = abs(qty)
                        avg = (avg * held + price * remaining) / (held + remaining)
                        qty += remaining * direction
                realised[fill.exec_id] = pnl
    return realised


# ================================================================================================
# The index: every event the cards read, by what links it
# ================================================================================================


@dataclass(frozen=True, slots=True)
class _At:
    """A payload and the event it was read from."""

    event: LedgerEvent
    payload: Model

    @property
    def seq(self) -> int:
        return self.event.seq


@dataclass(slots=True)
class _Index:
    events: dict[int, LedgerEvent] = field(default_factory=dict)
    snapshots: dict[str, list[_At]] = field(default_factory=lambda: defaultdict(list))
    triggers: dict[str, list[_At]] = field(default_factory=lambda: defaultdict(list))
    decisions: list[_At] = field(default_factory=list)
    decision_rulings: dict[str, list[_At]] = field(default_factory=lambda: defaultdict(list))
    protective_rulings: list[_At] = field(default_factory=list)
    plans: dict[str, list[_At]] = field(default_factory=lambda: defaultdict(list))
    order_events: dict[str, list[_At]] = field(default_factory=lambda: defaultdict(list))
    fills: list[_At] = field(default_factory=list)
    protective_actions: dict[str, list[_At]] = field(default_factory=lambda: defaultdict(list))
    feed_health: dict[str, list[_At]] = field(default_factory=lambda: defaultdict(list))
    """Feed-health reports by the snapshot they describe (contract 1.1.0)."""


def _oid_of(payload: Model) -> str | None:
    if isinstance(
        payload,
        DryRunPreview
        | OrderSubmitted
        | VenueAck
        | VenueRejection
        | VenueUnknown
        | OrderStateChange,
    ):
        return payload.client_oid
    return None


def _index(events: Iterable[LedgerEvent]) -> _Index:
    index = _Index()
    for event in events:
        index.events[event.seq] = event
        payload = parse_payload(event)
        at = _At(event, payload)
        kind = event.kind
        if isinstance(payload, SnapshotEvent):
            index.snapshots[payload.snapshot.snapshot_id].append(at)
        elif isinstance(payload, Trigger):
            index.triggers[payload.trigger_id].append(at)
        elif isinstance(payload, DecisionEvent):
            index.decisions.append(at)
        elif isinstance(payload, KernelRuling):
            if payload.decision_id is not None:
                index.decision_rulings[payload.decision_id].append(at)
            else:
                index.protective_rulings.append(at)
        elif isinstance(payload, OrderPlan):
            index.plans[payload.ruling_id].append(at)
        elif kind in _ORDER_KINDS:
            oid = _oid_of(payload)
            if oid is not None:
                index.order_events[oid].append(at)
        elif isinstance(payload, Fill):
            index.fills.append(at)
        elif isinstance(payload, ProtectiveAction):
            index.protective_actions[payload.ruling_id].append(at)
        elif isinstance(payload, FeedHealthReport):
            index.feed_health[payload.snapshot_id].append(at)
    return index


def _latest_before(candidates: Sequence[_At], seq: int) -> _At | None:
    """The last candidate logged before ``seq``, else the first logged after it."""
    before = [c for c in candidates if c.seq < seq]
    if before:
        return before[-1]
    return candidates[0] if candidates else None


# ================================================================================================
# Orders
# ================================================================================================


@dataclass(frozen=True, slots=True)
class OrderTrail:
    """One planned order as the ledger records it, beyond what a :class:`CardOrder` carries.

    Built for every intent of every plan; the export publishes these as ``orders.json``.
    """

    card_id: str
    plan_id: str
    intent: OrderIntent
    order: CardOrder
    preview: DryRunPreview | None
    submitted: OrderSubmitted | None
    ack: VenueAck | None
    rejection: VenueRejection | None
    unknown: VenueUnknown | None
    states: tuple[OrderStateChange, ...]
    seqs: tuple[int, ...]


class _Orders:
    """Order trails keyed by ``clientOid``, and the fills matched to them."""

    def __init__(self, index: _Index) -> None:
        self._index = index
        ordered_fills = [_fill_of(at) for at in index.fills]
        self._realised = realized_pnl_by_fill(ordered_fills)
        self._fills_by_oid: dict[str, list[_At]] = defaultdict(list)
        self._fills_by_venue_id: dict[str, list[_At]] = defaultdict(list)
        for at in index.fills:
            fill = _fill_of(at)
            if fill.client_oid:
                self._fills_by_oid[fill.client_oid].append(at)
            self._fills_by_venue_id[fill.venue_order_id].append(at)
        self.claimed_fill_seqs: set[int] = set()

    def trail(self, card_id: str, plan: OrderPlan, intent: OrderIntent) -> OrderTrail:
        oid = intent.client_oid
        records = self._index.order_events.get(oid, [])
        previews = [r.payload for r in records if isinstance(r.payload, DryRunPreview)]
        submitted = [r.payload for r in records if isinstance(r.payload, OrderSubmitted)]
        acks = [r.payload for r in records if isinstance(r.payload, VenueAck)]
        rejections = [r.payload for r in records if isinstance(r.payload, VenueRejection)]
        unknowns = [r.payload for r in records if isinstance(r.payload, VenueUnknown)]
        states = tuple(r.payload for r in records if isinstance(r.payload, OrderStateChange))

        venue_id: str | None = acks[-1].venue_order_id if acks else None
        if venue_id is None:
            for change in reversed(states):
                if change.venue_order_id:
                    venue_id = change.venue_order_id
                    break
        fill_records: dict[str, _At] = {}
        for at in self._fills_by_oid.get(oid, []):
            fill_records.setdefault(_fill_of(at).exec_id, at)
        if venue_id is not None:
            for at in self._fills_by_venue_id.get(venue_id, []):
                fill = _fill_of(at)
                if fill.client_oid in (None, "", oid):
                    fill_records.setdefault(fill.exec_id, at)
        ordered = sorted(fill_records.values(), key=lambda a: a.seq)
        fills = tuple(_fill_of(at) for at in ordered)
        if venue_id is None and fills:
            venue_id = fills[0].venue_order_id
        self.claimed_fill_seqs.update(at.seq for at in ordered)

        net: Decimal | None = None
        if fills:
            with localcontext(DECIMAL_CONTEXT):
                net = sum(
                    (self._realised.get(f.exec_id, Decimal(0)) - f.fee_paid for f in fills),
                    Decimal(0),
                )
        state = states[-1].to_state if states else OrderState.INITIALISED
        order = CardOrder(
            client_oid=oid,
            venue_order_id=venue_id,
            symbol=intent.symbol,
            side=intent.side,
            qty=intent.qty,
            purpose=intent.purpose,
            state=state,
            fills=fills,
            net_pnl=net,
        )
        seqs = tuple(sorted({r.seq for r in records} | {at.seq for at in ordered}))
        return OrderTrail(
            card_id=card_id,
            plan_id=plan.plan_id,
            intent=intent,
            order=order,
            preview=previews[-1] if previews else None,
            submitted=submitted[-1] if submitted else None,
            ack=acks[-1] if acks else None,
            rejection=rejections[-1] if rejections else None,
            unknown=unknowns[-1] if unknowns else None,
            states=states,
            seqs=seqs,
        )

    def previews_of(self, intents: Sequence[OrderIntent]) -> tuple[DryRunPreview, ...]:
        out: list[DryRunPreview] = []
        for intent in intents:
            for record in self._index.order_events.get(intent.client_oid, []):
                if isinstance(record.payload, DryRunPreview):
                    out.append(record.payload)
        return tuple(out)


def _fill_of(at: _At) -> Fill:
    payload = at.payload
    if not isinstance(payload, Fill):  # pragma: no cover - the index files only fills here
        raise CardError(f"seq {at.seq} is not a fill")
    return payload


# ================================================================================================
# Cards
# ================================================================================================


@dataclass(frozen=True, slots=True)
class CardBundle:
    """The cards and the order trails they were built with (the export publishes both)."""

    cards: tuple[DecisionCard, ...]
    trails: tuple[OrderTrail, ...]
    venue_originated_fills: tuple[tuple[int, Fill], ...]
    """Fills whose order the agent did not plan (a venue stop firing, or an order placed outside
    the agent), with their ledger seq. They belong to no card; the book books them by their own
    rules and the export lists them in ``orders.json``."""


def _proof(
    seqs: Iterable[int], index: _Index, blobs: BlobStore, card_id: str
) -> tuple[BlobRef, ...]:
    """Every blob the events at ``seqs`` commit to, each fetched and re-hashed."""
    refs: dict[str, BlobRef] = {}
    for seq in sorted(set(seqs)):
        event = index.events[seq]
        try:
            found = referenced_blobs(event)
        except LedgerError as exc:
            raise CardError(f"card {card_id}: {exc}") from None
        for ref in found:
            refs.setdefault(ref.sha256, ref)
    for sha, ref in refs.items():
        try:
            data = blobs.get(sha)
        except Exception as exc:
            raise CardError(
                f"card {card_id}: blob {sha} ({ref.media_type}) cannot be read from the store: "
                f"{type(exc).__name__}: {exc}"
            ) from None
        if sha256_hex(data) != sha or len(data) != ref.size:
            raise CardError(
                f"card {card_id}: blob {sha} does not hash to its name or has the wrong size; "
                "the store was altered"
            )
    return tuple(refs.values())


def _plan_orders(
    card_id: str, ruling: KernelRuling | None, index: _Index, orders: _Orders
) -> tuple[list[OrderTrail], list[int], tuple[DryRunPreview, ...]]:
    if ruling is None:
        return [], [], ()
    trails: list[OrderTrail] = []
    seqs: list[int] = []
    intents: list[OrderIntent] = []
    for plan_at in index.plans.get(ruling.ruling_id, []):
        plan = plan_at.payload
        if not isinstance(plan, OrderPlan):  # pragma: no cover - indexed by type
            continue
        seqs.append(plan_at.seq)
        for intent in plan.intents:
            trail = orders.trail(card_id, plan, intent)
            trails.append(trail)
            seqs.extend(trail.seqs)
            intents.append(intent)
    return trails, seqs, orders.previews_of(intents)


def _decision_card(
    at: _At, index: _Index, orders: _Orders, blobs: BlobStore
) -> tuple[DecisionCard, list[OrderTrail]]:
    payload = at.payload
    if not isinstance(payload, DecisionEvent):  # pragma: no cover - indexed by type
        raise CardError(f"seq {at.seq} is not a decision")
    record: DecisionRecord = payload.record
    card_id = decision_card_id(record.decision_id)
    seqs: list[int] = [at.seq]

    snapshot_at = _latest_before(index.snapshots.get(record.snapshot_id, []), at.seq)
    if snapshot_at is None:
        raise CardError(
            f"decision {record.decision_id} (seq {at.seq}) names snapshot {record.snapshot_id}, "
            "which the ledger does not hold"
        )
    snapshot_payload = snapshot_at.payload
    if not isinstance(snapshot_payload, SnapshotEvent):  # pragma: no cover - indexed by type
        raise CardError(f"seq {snapshot_at.seq} is not a snapshot")
    snapshot: PerceptionSnapshot = snapshot_payload.snapshot
    seqs.append(snapshot_at.seq)

    triggers: list[Trigger] = []
    for trigger_id in record.trigger_ids:
        trigger_at = _latest_before(index.triggers.get(trigger_id, []), at.seq)
        if trigger_at is None or not isinstance(trigger_at.payload, Trigger):
            raise CardError(
                f"decision {record.decision_id} (seq {at.seq}) names trigger {trigger_id}, which "
                "the ledger does not hold"
            )
        triggers.append(trigger_at.payload)
        seqs.append(trigger_at.seq)

    rulings = index.decision_rulings.get(record.decision_id, [])
    after = [r for r in rulings if r.seq > at.seq]
    ruling_at = after[0] if after else (rulings[-1] if rulings else None)
    ruling: KernelRuling | None = None
    if ruling_at is not None and isinstance(ruling_at.payload, KernelRuling):
        ruling = ruling_at.payload
        seqs.append(ruling_at.seq)
    trails, order_seqs, previews = _plan_orders(card_id, ruling, index, orders)
    seqs.extend(order_seqs)

    coverage: dict[str, SourceHealth] = snapshot.coverage()
    shown: tuple[ScreenedItem, ...] = snapshot.text
    feeds_at = _latest_before(index.feed_health.get(record.snapshot_id, []), at.seq)
    feeds: FeedHealthReport | None = None
    if feeds_at is not None and isinstance(feeds_at.payload, FeedHealthReport):
        feeds = feeds_at.payload
        seqs.append(feeds_at.seq)
    ledger_seqs = tuple(sorted(set(seqs)))
    card = DecisionCard(
        card_id=card_id,
        at=record.decided_at,
        decision_id=record.decision_id,
        ruling_id=ruling.ruling_id if ruling is not None else None,
        triggers=tuple(triggers),
        coverage=coverage,
        shown_text=shown,
        outcome=record.outcome,
        decision=record.decision,
        grounding=dict(record.grounding),
        kernel=ruling,
        previews=previews,
        orders=tuple(t.order for t in trails),
        ledger_seqs=ledger_seqs,
        blobs=_proof(ledger_seqs, index, blobs, card_id),
        feed_health=feeds,
    )
    return card, trails


def _protective_card(
    at: _At, index: _Index, orders: _Orders, blobs: BlobStore
) -> tuple[DecisionCard, list[OrderTrail]]:
    ruling = at.payload
    if not isinstance(ruling, KernelRuling):  # pragma: no cover - indexed by type
        raise CardError(f"seq {at.seq} is not a kernel ruling")
    card_id = protective_card_id(ruling.ruling_id)
    seqs: list[int] = [at.seq]
    trails, order_seqs, previews = _plan_orders(card_id, ruling, index, orders)
    seqs.extend(order_seqs)
    seqs.extend(a.seq for a in index.protective_actions.get(ruling.ruling_id, []))
    ledger_seqs = tuple(sorted(set(seqs)))
    card = DecisionCard(
        card_id=card_id,
        at=ruling.at,
        decision_id=None,
        ruling_id=ruling.ruling_id,
        triggers=(),
        coverage={},
        shown_text=(),
        outcome=None,
        decision=None,
        grounding={},
        kernel=ruling,
        previews=previews,
        orders=tuple(t.order for t in trails),
        ledger_seqs=ledger_seqs,
        blobs=_proof(ledger_seqs, index, blobs, card_id),
    )
    return card, trails


def build_card_bundle(projection: Projection, blobs: BlobStore) -> CardBundle:
    """Every card, every order trail and every venue-originated fill, from one pass over the log.

    Cards are ordered by time, then by ledger position. Raises :class:`CardError` when a card cannot
    be built honestly (module docstring), or when two cards would share an id.
    """
    index = _index(projection.events())
    orders = _Orders(index)
    built: list[tuple[DecisionCard, int, list[OrderTrail]]] = []
    for at in index.decisions:
        card, trails = _decision_card(at, index, orders, blobs)
        built.append((card, at.seq, trails))
    for at in index.protective_rulings:
        card, trails = _protective_card(at, index, orders, blobs)
        built.append((card, at.seq, trails))
    built.sort(key=lambda item: (item[0].at, item[1]))
    ids = [card.card_id for card, _, _ in built]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise CardError(
            f"two cards would share the id(s) {duplicates}; the ledger repeats a record"
        )
    venue_fills = tuple(
        (at.seq, _fill_of(at)) for at in index.fills if at.seq not in orders.claimed_fill_seqs
    )
    return CardBundle(
        cards=tuple(card for card, _, _ in built),
        trails=tuple(t for _, _, trails in built for t in trails),
        venue_originated_fills=venue_fills,
    )


def build_cards(projection: Projection, blobs: BlobStore) -> tuple[DecisionCard, ...]:
    """One card per decision, abstention, outage and protective ruling, in time order."""
    return build_card_bundle(projection, blobs).cards


def card_index(cards: Sequence[DecisionCard]) -> Mapping[str, DecisionCard]:
    """Cards by id."""
    return {card.card_id: card for card in cards}


__all__ = [
    "CARD_ID",
    "DECISION_CARD_PREFIX",
    "PROTECTIVE_CARD_PREFIX",
    "CardBundle",
    "CardError",
    "OrderTrail",
    "build_card_bundle",
    "build_cards",
    "card_index",
    "decision_card_id",
    "protective_card_id",
    "realized_pnl_by_fill",
]
