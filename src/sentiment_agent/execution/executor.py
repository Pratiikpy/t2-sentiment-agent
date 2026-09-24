"""The executor: approved orders only, each previewed, logged before it is sent, never sent twice.

For every :class:`~sentiment_agent.types.ApprovedOrder` (DESIGN.md §11.1, §11.3):

1. **All approvals are checked before anything is sent.** An approval whose hash no longer matches
   its intent refuses the whole batch (:class:`ApprovalTamperedError`); nothing in it is sent.
2. **A PAPER ledger must hold a passed environment proof**, and its latest proof must be the passed
   one; otherwise :class:`~sentiment_agent.execution.environment.EnvironmentRefused`. The transport
   enforces the same rule on its own (``BgcTransport`` cannot be built without a passed proof).
3. **A ``clientOid`` already in the ledger is never sent again.** Executing the same approved batch
   twice (a restart after a crash) sends only what was never sent, and reports the rest at their
   recorded state.
4. **Preview, then send.** The dry-run preview is logged (``ORDER_PREVIEW``) before anything else;
   if it fails or disagrees with the intent, the order is DENIED and never sent. In DRYRUN mode the
   preview is where it stops.
5. **The intent to send is durable before the network call:** ``ORDER_SUBMITTED`` and the SUBMITTED
   state are written first, so a crash mid-send leaves a live order for reconciliation to resolve,
   never a forgotten one.
6. **The venue's answer is logged as it came:** ``ORDER_ACK`` (with the venue ``orderId``),
   ``ORDER_REJECTED`` or ``ORDER_UNKNOWN``. A timeout is UNKNOWN, not rejected; UNKNOWN is resolved
   only by reading the order back by its ``clientOid`` (the reconciler), never by resending. 40099
   on a send is logged as a rejection and re-raised.
7. **After an acknowledgement**, the order is read back until it is terminal or ``poll_timeout_s``
   passes, each observed state is logged, and the order's fills are written once each.

The executor never decides anything. It cannot change a quantity, a side or a stop, and the only
orders it can express are the ones the kernel minted.
"""

import math
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from typing import Final

from sentiment_agent.execution.bgc import TransportRefusedError, VenueCallError
from sentiment_agent.execution.environment import (
    DRY_RUN_FLAG,
    ENVIRONMENT_MISMATCH_CODE,
    PAPER_FLAG,
    EnvironmentRefused,
)
from sentiment_agent.execution.orders import (
    LEGAL_TRANSITIONS,
    OrderTracker,
    map_venue_status,
)
from sentiment_agent.types import (
    ApprovedOrder,
    BlobRef,
    CardOrder,
    Clock,
    EnvironmentProof,
    EventKind,
    Fill,
    FillVenue,
    LedgerWriter,
    Model,
    Note,
    OrderIntent,
    OrderState,
    OrderSubmitted,
    RunMode,
    StopSync,
    VenueAck,
    VenueOrder,
    VenueRejection,
    VenueTransport,
    VenueUnknown,
)

FILL_LOOKBACK: Final = timedelta(minutes=5)
"""How far before the send the order's fills are searched for (venue timestamps may lead ours)."""


class ApprovalTamperedError(RuntimeError):
    """An approval no longer matches the intent it was minted for. Nothing in the batch was sent."""


class Executor:
    """Sends approved orders through a :class:`~sentiment_agent.types.VenueTransport`."""

    def __init__(
        self,
        *,
        transport: VenueTransport,
        ledger: LedgerWriter,
        tracker: OrderTracker,
        clock: Clock,
        poll_timeout_s: float = 60.0,
        poll_interval_s: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        expected = (
            FillVenue.SIMULATED if ledger.mode is RunMode.SIMULATED else FillVenue.BITGET_DEMO
        )
        if transport.venue is not expected:
            raise ValueError(
                f"a {ledger.mode} ledger takes a {expected} transport, not {transport.venue}: "
                "a simulated fill must never reach the paper record"
            )
        if poll_timeout_s < 0 or poll_interval_s <= 0:
            raise ValueError("poll timeout must be >= 0 and the interval > 0")
        self._transport = transport
        self._ledger = ledger
        self._tracker = tracker
        self._clock = clock
        self._poll_timeout_s = poll_timeout_s
        self._poll_interval_s = poll_interval_s
        self._sleep = sleep

    # --- the batch ----------------------------------------------------------------------------

    def execute(self, approved: Sequence[ApprovedOrder]) -> list[CardOrder]:
        """Execute a batch in order; return one card line per distinct ``clientOid``."""
        for order in approved:
            if not order.verify():
                self._log(
                    EventKind.NOTE,
                    Note(
                        at=self._clock.now(),
                        author="system",
                        text=f"refused a tampered approval for {order.intent.client_oid}; "
                        "nothing in this batch was sent",
                    ),
                )
                raise ApprovalTamperedError(
                    f"approval for {order.intent.client_oid} does not match its intent"
                )
        if approved and self._ledger.mode is RunMode.PAPER:
            self._require_passed_proof()

        cards: list[CardOrder] = []
        seen: set[str] = set()
        for order in approved:
            oid = order.intent.client_oid
            if oid in seen:
                continue
            seen.add(oid)
            if self._tracker.state(oid) is not None:
                cards.append(self._card(order.intent, ()))
                continue
            cards.append(self._execute_one(order))
        return cards

    def _require_passed_proof(self) -> None:
        events = getattr(self._ledger, "events", None)
        if not callable(events):
            raise EnvironmentRefused(
                "paper execution needs a ledger it can read the environment proof from"
            )
        latest = None
        for event in events(frozenset({EventKind.ENVIRONMENT_PROOF})):
            latest = event
        if latest is None:
            raise EnvironmentRefused("the paper ledger holds no environment proof")
        proof = EnvironmentProof.model_validate(latest.payload)
        if not proof.passed:
            raise EnvironmentRefused("the latest environment proof did not pass", proof=proof)

    # --- one order ------------------------------------------------------------------------------

    def _execute_one(self, order: ApprovedOrder) -> CardOrder:
        intent = order.intent
        oid = intent.client_oid
        try:
            preview = self._transport.preview(intent)
        except EnvironmentRefused:
            raise
        except (TransportRefusedError, VenueCallError, TimeoutError, OSError, ValueError) as exc:
            self._move(
                oid, OrderState.DENIED, f"executor: dry-run preview failed, never sent: {exc}"
            )
            return self._card(intent, ())
        if PAPER_FLAG not in preview.argv:
            self._move(oid, OrderState.DENIED, "executor: preview argv lacks --paper-trading")
            return self._card(intent, ())
        self._log(EventKind.ORDER_PREVIEW, preview, preview.blob)
        if self._ledger.mode is RunMode.DRYRUN:
            return self._card(intent, ())

        submitted_at = self._clock.now()
        submitted = OrderSubmitted(
            client_oid=oid,
            intent=intent,
            approval_hash=order.approval_hash,
            submitted_at=submitted_at,
            argv=tuple(arg for arg in preview.argv if arg != DRY_RUN_FLAG),
        )
        self._log(EventKind.ORDER_SUBMITTED, submitted)
        self._tracker.note_submitted(submitted)
        self._move(oid, OrderState.SUBMITTED, "sent to the venue")

        try:
            outcome = self._transport.place(order)
        except EnvironmentRefused as exc:
            self._reject(
                VenueRejection(
                    client_oid=oid,
                    code=ENVIRONMENT_MISMATCH_CODE,
                    message=str(exc),
                    category="environment",
                    retryable=False,
                    at=self._clock.now(),
                    blob=exc.evidence,
                )
            )
            raise
        except TransportRefusedError as exc:
            self._reject(
                VenueRejection(
                    client_oid=oid,
                    code=None,
                    message=f"refused by the transport before sending: {exc}",
                    category="local",
                    retryable=False,
                    at=self._clock.now(),
                    blob=None,
                )
            )
            return self._card(intent, ())
        except Exception as exc:
            self._unknown(
                VenueUnknown(
                    client_oid=oid,
                    at=self._clock.now(),
                    reason=f"send raised {type(exc).__name__}: {exc}",
                )
            )
            raise

        if isinstance(outcome, VenueRejection):
            self._reject(outcome)
            return self._card(intent, ())
        if isinstance(outcome, VenueUnknown):
            self._unknown(outcome)
            return self._card(intent, ())
        return self._after_ack(intent, outcome, submitted_at)

    def _after_ack(self, intent: OrderIntent, ack: VenueAck, submitted_at: datetime) -> CardOrder:
        oid = intent.client_oid
        self._log(EventKind.ORDER_ACK, ack, ack.blob)
        self._move(oid, OrderState.ACCEPTED, "acknowledged by the venue", ack.venue_order_id)
        if intent.stop_loss_price is not None:
            self._log(
                EventKind.STOP_SYNC,
                StopSync(
                    symbol=intent.symbol,
                    action="preset",
                    stop_price=intent.stop_loss_price,
                    venue_id=None,
                    at=self._clock.now(),
                ),
            )
        started = self._clock.now()
        deadline = started + timedelta(seconds=self._poll_timeout_s)
        polls = max(1, math.ceil(self._poll_timeout_s / self._poll_interval_s)) + 1
        for attempt in range(polls):
            try:
                observed = self._transport.order(client_oid=oid)
            except EnvironmentRefused:
                raise
            except (VenueCallError, TimeoutError, OSError):
                observed = None
            if observed is not None:
                self._observe(observed)
            state = self._tracker.state(oid)
            if (state is not None and state.is_terminal) or self._clock.now() >= deadline:
                break
            if attempt < polls - 1:
                self._sleep(self._poll_interval_s)
        fills = self._collect_fills(intent, ack.venue_order_id, submitted_at - FILL_LOOKBACK)
        return self._card(intent, fills)

    def _observe(self, observed: VenueOrder) -> None:
        oid = observed.client_oid
        if oid is None:
            return
        current = self._tracker.state(oid)
        target = map_venue_status(observed.status)
        if current is None or target is current:
            return
        if target not in LEGAL_TRANSITIONS[current]:
            self._log(
                EventKind.NOTE,
                Note(
                    at=self._clock.now(),
                    author="system",
                    text=f"{oid}: the venue reports {observed.status} while the ledger holds "
                    f"{current}; left for reconciliation",
                ),
            )
            return
        self._move(oid, target, f"venue orderStatus {observed.status}", observed.venue_order_id)

    def _collect_fills(
        self, intent: OrderIntent, venue_order_id: str, since: datetime
    ) -> tuple[Fill, ...]:
        state = self._tracker.state(intent.client_oid)
        if state not in (OrderState.FILLED, OrderState.PARTIALLY_FILLED, OrderState.CANCELLED):
            return ()
        try:
            fills = self._transport.fills(since=since, until=self._clock.now())
        except EnvironmentRefused:
            raise
        except (VenueCallError, TimeoutError, OSError):
            return ()
        mine = [
            fill
            for fill in fills
            if fill.client_oid == intent.client_oid or fill.venue_order_id == venue_order_id
        ]
        for fill in mine:
            if not self._tracker.has_fill(fill.exec_id):
                self._log(EventKind.FILL, fill, fill.blob)
                self._tracker.note_fill(fill.exec_id)
        return tuple(mine)

    # --- logging ------------------------------------------------------------------------------

    def _log(self, kind: EventKind, payload: Model, *blobs: BlobRef | None) -> None:
        self._ledger.append(kind, payload, blobs=[b for b in blobs if b is not None])

    def _move(
        self, oid: str, to: OrderState, reason: str, venue_order_id: str | None = None
    ) -> None:
        change = self._tracker.transition(
            oid, to, at=self._clock.now(), reason=reason, venue_order_id=venue_order_id
        )
        self._log(EventKind.ORDER_STATE, change)

    def _reject(self, rejection: VenueRejection) -> None:
        self._log(EventKind.ORDER_REJECTED, rejection, rejection.blob)
        self._move(
            rejection.client_oid,
            OrderState.REJECTED,
            f"rejected: {rejection.code or rejection.category}: {rejection.message}",
        )

    def _unknown(self, unknown: VenueUnknown) -> None:
        self._log(EventKind.ORDER_UNKNOWN, unknown)
        self._move(unknown.client_oid, OrderState.UNKNOWN, f"outcome unknown: {unknown.reason}")

    def _card(self, intent: OrderIntent, fills: Sequence[Fill]) -> CardOrder:
        oid = intent.client_oid
        return CardOrder(
            client_oid=oid,
            venue_order_id=self._tracker.venue_order_id(oid),
            symbol=intent.symbol,
            side=intent.side,
            qty=intent.qty,
            purpose=intent.purpose,
            state=self._tracker.state(oid) or OrderState.INITIALISED,
            fills=tuple(fills),
            net_pnl=None,
        )


__all__ = ["FILL_LOOKBACK", "ApprovalTamperedError", "Executor"]
