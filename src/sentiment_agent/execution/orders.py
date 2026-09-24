"""Order state machine and the order tracker rebuilt from the ledger.

Ported from ARGUS ``src/argus/execution/orders.py`` (same author, MIT) at commit
``3dec6baf9dfa7be37c7b452e26a9b139df252f75``, sha256
``bf2e41ee9d6ffbf87ae485ad2766afb43a82ac433de63f62cfca6d8aba7cc160``. The ARGUS repository is
private, so its licence and every pin are recorded in ``third_party/argus/PROVENANCE.md``. Copied,
not imported: this project never imports from the ARGUS tree.

What was taken, and from where ARGUS took it:

* **The transition table** (ARGUS ``_LEGAL``), itself a reimplementation of NautilusTrader's order
  model read in ``crates/model/src/enums.rs:1304-1388`` (LGPL-3.0; studied, not vendored). The
  three corrections that reading made to ARGUS's first draft carry over unchanged: DENIED (our own
  kernel refused) is not REJECTED (the venue refused); PENDING_UPDATE and PENDING_CANCEL are open
  *and* in flight; and SUBMITTED counts as live, because filtering reconciliation on "open" alone
  silently drops an order the venue reports as pending (``enums.rs:1340-1343``).
* **UNKNOWN**, an ARGUS addition that Nautilus does not have: a request whose outcome is not known.
  It is reachable only from an in-flight or working state and left only through reconciliation,
  never by resending.
* **DuplicateOrder**: a client order id that was already submitted is never submitted again.

What differs from ARGUS, and why:

* ARGUS kept each ``Order`` as a mutable object with its history. Here the ledger is the only
  record (DESIGN.md §12): every change is an :class:`~sentiment_agent.types.OrderStateChange`
  written as an ``ORDER_STATE`` event, and :class:`OrderTracker` is a projection of those events,
  rebuilt by :meth:`OrderTracker.restore` after a crash. Nothing about an order lives only in
  memory.
* :class:`~sentiment_agent.types.OrderState` is defined once, in the shared contract, so the state
  predicates (``is_live``, ``is_terminal``) are not repeated here.
* The first transition of an order the tracker has never seen starts from INITIALISED. There is no
  separate "create" step, because an order that was never previewed or submitted has no event and
  therefore no state worth recording.
"""

from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Final

from sentiment_agent.types import (
    EventKind,
    Fill,
    LedgerEvent,
    OrderState,
    OrderStateChange,
    OrderSubmitted,
    VenueOrderStatus,
)

LEGAL_TRANSITIONS: Final[Mapping[OrderState, frozenset[OrderState]]] = {
    OrderState.INITIALISED: frozenset({OrderState.DENIED, OrderState.SUBMITTED}),
    OrderState.SUBMITTED: frozenset(
        {
            OrderState.ACCEPTED,
            OrderState.REJECTED,
            OrderState.FILLED,  # a market order can fill straight through
            OrderState.PARTIALLY_FILLED,
            OrderState.CANCELLED,
            OrderState.UNKNOWN,
        }
    ),
    OrderState.ACCEPTED: frozenset(
        {
            OrderState.TRIGGERED,
            OrderState.PENDING_UPDATE,
            OrderState.PENDING_CANCEL,
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.EXPIRED,
            OrderState.VOIDED,
            OrderState.UNKNOWN,
        }
    ),
    OrderState.TRIGGERED: frozenset(
        {
            OrderState.PENDING_UPDATE,
            OrderState.PENDING_CANCEL,
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.EXPIRED,
            OrderState.UNKNOWN,
        }
    ),
    OrderState.PENDING_UPDATE: frozenset(
        {
            OrderState.ACCEPTED,
            OrderState.TRIGGERED,
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.EXPIRED,
            OrderState.REJECTED,  # the modify itself can be refused
            OrderState.UNKNOWN,
        }
    ),
    OrderState.PENDING_CANCEL: frozenset(
        {
            OrderState.ACCEPTED,  # cancel refused, order still working
            OrderState.CANCELLED,
            OrderState.PARTIALLY_FILLED,  # raced a fill
            OrderState.FILLED,
            OrderState.EXPIRED,
            OrderState.REJECTED,
            OrderState.UNKNOWN,
        }
    ),
    OrderState.PARTIALLY_FILLED: frozenset(
        {
            OrderState.PENDING_UPDATE,
            OrderState.PENDING_CANCEL,
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.EXPIRED,
            OrderState.VOIDED,
            OrderState.UNKNOWN,
        }
    ),
    # Reconciliation is the only exit from UNKNOWN, and it lands wherever the venue says.
    OrderState.UNKNOWN: frozenset(
        {
            OrderState.ACCEPTED,
            OrderState.TRIGGERED,
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.EXPIRED,
            OrderState.VOIDED,
        }
    ),
    # Terminal. VOIDED stays reachable from FILLED because a venue can bust a fill afterwards.
    OrderState.FILLED: frozenset({OrderState.VOIDED}),
    OrderState.DENIED: frozenset(),
    OrderState.CANCELLED: frozenset(),
    OrderState.REJECTED: frozenset(),
    OrderState.EXPIRED: frozenset(),
    OrderState.VOIDED: frozenset(),
}
"""Every legal move between order states. A move not listed here is a defect, never absorbed."""

_VENUE_STATUS: Final[Mapping[VenueOrderStatus, OrderState]] = {
    VenueOrderStatus.LIVE: OrderState.ACCEPTED,
    VenueOrderStatus.NEW: OrderState.ACCEPTED,
    VenueOrderStatus.PARTIALLY_FILLED: OrderState.PARTIALLY_FILLED,
    VenueOrderStatus.FILLED: OrderState.FILLED,
    VenueOrderStatus.CANCELLED: OrderState.CANCELLED,
}


class IllegalTransition(RuntimeError):  # noqa: N818 - name fixed by DESIGN.md §18
    """An order was moved between states the venue protocol does not permit."""


class DuplicateOrder(RuntimeError):  # noqa: N818 - name fixed by DESIGN.md §18
    """A client order id that is already in the ledger was submitted again.

    A replayed approval must never produce a second order. The executor checks
    :meth:`OrderTracker.state` first and skips a known ``clientOid``; this exception is the backstop
    if anything ever tries to submit one anyway.
    """


def map_venue_status(status: VenueOrderStatus) -> OrderState:
    """Bitget UTA ``orderStatus`` to our state (DESIGN.md §11.3).

    ``live`` ("order created") and ``new`` ("order matching") are both working at the venue, so both
    are ACCEPTED (legacy-docs/uta/trade/Get-Order-Details, fetched 2026-09-24).
    """
    return _VENUE_STATUS[status]


class OrderTracker:
    """Order states, rebuilt from ``ORDER_STATE`` events and advanced one legal step at a time.

    :meth:`transition` validates and applies a change and returns it; the caller appends the
    returned :class:`~sentiment_agent.types.OrderStateChange` to the ledger. After a restart,
    :meth:`restore` replays those events and arrives at the same states, so an order that was in
    flight when the process died is still live (SUBMITTED or UNKNOWN) and is resolved by
    reconciliation rather than forgotten or resent.

    The tracker also remembers every fill id it has seen in the ledger, so a fill read back from the
    venue twice is written once.
    """

    def __init__(self) -> None:
        self._states: dict[str, OrderState] = {}
        self._history: dict[str, list[OrderStateChange]] = {}
        self._venue_ids: dict[str, str] = {}
        self._submitted: dict[str, OrderSubmitted] = {}
        self._fill_ids: set[str] = set()
        self._order: list[str] = []

    # --- rebuild ----------------------------------------------------------------------------

    def restore(self, events: Iterable[LedgerEvent]) -> None:
        """Replay order-state, submission and fill events from the ledger.

        Events are applied in ``seq`` order. An illegal transition in the ledger raises
        :class:`IllegalTransition`: the log is our own record, and a log that contradicts the state
        machine is corrupt, not something to smooth over.
        """
        for event in sorted(events, key=lambda e: e.seq):
            if event.kind is EventKind.ORDER_STATE:
                change = OrderStateChange.model_validate(event.payload)
                self._apply(change)
            elif event.kind is EventKind.ORDER_SUBMITTED:
                submitted = OrderSubmitted.model_validate(event.payload)
                self._submitted.setdefault(submitted.client_oid, submitted)
            elif event.kind is EventKind.FILL:
                self._fill_ids.add(Fill.model_validate(event.payload).exec_id)

    # --- queries ----------------------------------------------------------------------------

    def state(self, client_oid: str) -> OrderState | None:
        """The current state, or ``None`` for an order the ledger has never recorded."""
        return self._states.get(client_oid)

    def live(self) -> list[str]:
        """Every order carrying exposure: open, in flight or unknown (``OrderState.is_live``).

        SUBMITTED is included on purpose (the Nautilus trap in the module docstring).
        """
        return [oid for oid in self._order if self._states[oid].is_live]

    def unknown(self) -> list[str]:
        """Orders whose venue outcome is not known. Resolvable only by reconciliation."""
        return [oid for oid in self._order if self._states[oid] is OrderState.UNKNOWN]

    def known(self) -> list[str]:
        """Every client order id the ledger has a state for, in first-seen order."""
        return list(self._order)

    def venue_order_id(self, client_oid: str) -> str | None:
        return self._venue_ids.get(client_oid)

    def submitted(self, client_oid: str) -> OrderSubmitted | None:
        """The ``ORDER_SUBMITTED`` record for an order, when one was logged."""
        return self._submitted.get(client_oid)

    def history(self, client_oid: str) -> tuple[OrderStateChange, ...]:
        return tuple(self._history.get(client_oid, ()))

    def last_change(self, client_oid: str) -> OrderStateChange | None:
        changes = self._history.get(client_oid)
        return changes[-1] if changes else None

    def has_fill(self, exec_id: str) -> bool:
        return exec_id in self._fill_ids

    def fill_ids(self) -> frozenset[str]:
        return frozenset(self._fill_ids)

    # --- changes ----------------------------------------------------------------------------

    def note_submitted(self, submitted: OrderSubmitted) -> None:
        """Record an ``ORDER_SUBMITTED`` payload the caller has just written to the ledger."""
        self._submitted.setdefault(submitted.client_oid, submitted)

    def note_fill(self, exec_id: str) -> None:
        """Record a fill id the caller has just written to the ledger."""
        self._fill_ids.add(exec_id)

    def transition(
        self,
        client_oid: str,
        to: OrderState,
        *,
        at: datetime,
        reason: str,
        venue_order_id: str | None = None,
    ) -> OrderStateChange:
        """Validate and apply one move; return the change for the caller to log.

        An order the tracker has never seen starts from INITIALISED. Submitting an order that is
        already known raises :class:`DuplicateOrder`; any other move not in
        :data:`LEGAL_TRANSITIONS` raises :class:`IllegalTransition`. Neither leaves any trace.
        """
        current = self._states.get(client_oid)
        if current is not None and to is OrderState.SUBMITTED:
            raise DuplicateOrder(
                f"{client_oid} is already in the ledger ({current}); a replayed approval must not "
                "reach the venue twice"
            )
        change = OrderStateChange(
            client_oid=client_oid,
            venue_order_id=venue_order_id or self._venue_ids.get(client_oid),
            from_state=OrderState.INITIALISED if current is None else current,
            to_state=to,
            at=at,
            reason=reason,
        )
        self._apply(change)
        return change

    def _apply(self, change: OrderStateChange) -> None:
        oid = change.client_oid
        current = self._states.get(oid)
        expected_from = OrderState.INITIALISED if current is None else current
        if change.from_state is not expected_from:
            raise IllegalTransition(
                f"{oid}: the change starts from {change.from_state} but the order is in "
                f"{expected_from}"
            )
        if change.to_state not in LEGAL_TRANSITIONS[change.from_state]:
            raise IllegalTransition(
                f"{oid}: {change.from_state} -> {change.to_state} is not a legal order transition"
            )
        if current is None:
            self._order.append(oid)
        self._states[oid] = change.to_state
        self._history.setdefault(oid, []).append(change)
        if change.venue_order_id:
            self._venue_ids[oid] = change.venue_order_id


__all__ = [
    "LEGAL_TRANSITIONS",
    "DuplicateOrder",
    "IllegalTransition",
    "OrderTracker",
    "map_venue_status",
]
