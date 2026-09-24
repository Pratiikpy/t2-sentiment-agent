"""Reconciliation: read the venue back, write what it holds, and report every difference.

Read-only (DESIGN.md §11.5). One :meth:`Reconciler.run`:

1. **Orders.** Every live order (SUBMITTED, working, PARTIALLY_FILLED or UNKNOWN) is read back by
   its ``clientOid``. A new venue state is logged as an ``ORDER_STATE`` change; an UNKNOWN that the
   venue answers is *resolved*. An order the venue has no record of stays live until
   ``unknown_grace_s`` after it was sent, then becomes REJECTED ("never reached the venue"). It is
   never sent again: the next decision cycle plans a new intent under a new ``clientOid``.
2. **Fills.** Fills since ``since`` are read, de-duplicated by ``execId`` against the ledger, and
   the new ones written as ``FILL`` events, whoever caused them. A fill that is not one of our
   orders is still written (the book must hold what the venue holds); it is a venue stop or
   liquidation when it reduces the book, and an ``orphan_fill`` otherwise.
3. **Positions.** The book plus the new fills is compared with the venue's positions, per symbol,
   exactly (``position_mismatch``).
4. **Stops.** Every open position must have a venue stop (``missing_stop``); a stop on a symbol with
   no position, or a second stop on one symbol, is an ``orphan_stop``.
5. **Equity.** The venue's account equity is compared with the book's. The two are read at slightly
   different moments and marks, and whether Demo credits funding is NOT VERIFIED (DESIGN.md §13), so
   a gap above ``equity_tolerance`` of the book's equity is an ``equity_gap``; the hourly ``MARK``
   event publishes the raw gap regardless.

Every difference is a :class:`~sentiment_agent.types.Discrepancy` in the logged
``RECONCILIATION`` report. Nothing is silently corrected. A read that fails is itself reported
(against the order it concerned, as ``unknown_order``), because a reconciliation that could not see
something must not look clean.

In DRYRUN mode nothing was ever sent and no credential exists, so there is nothing to reconcile;
the report says so and no venue call is made.
"""

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Final

from sentiment_agent.execution.bgc import VenueCallError
from sentiment_agent.execution.environment import EnvironmentRefused
from sentiment_agent.execution.orders import LEGAL_TRANSITIONS, OrderTracker, map_venue_status
from sentiment_agent.types import (
    AccountSnapshot,
    BlobRef,
    BookState,
    Clock,
    Discrepancy,
    EventKind,
    Fill,
    LedgerWriter,
    Model,
    OrderState,
    ReconciliationReport,
    RunMode,
    Side,
    VenueOrder,
    VenueStopOrder,
    VenueTransport,
)

UNKNOWN_GRACE_S: Final = 300.0
"""How long an order the venue cannot find stays live before it is declared never received."""

EQUITY_TOLERANCE: Final = Decimal("0.005")


def _signed(fill: Fill) -> Decimal:
    return fill.exec_qty if fill.side is Side.BUY else -fill.exec_qty


VENUE_CLOCK_SLACK = timedelta(seconds=60)
"""How far past the local clock a sweep reads, so a fill stamped on a faster venue clock is not
left outside the window."""


class Reconciler:
    """Compares the ledger's picture with the venue's and logs the report."""

    def __init__(
        self,
        *,
        transport: VenueTransport,
        ledger: LedgerWriter,
        tracker: OrderTracker,
        clock: Clock,
        unknown_grace_s: float = UNKNOWN_GRACE_S,
        equity_tolerance: Decimal = EQUITY_TOLERANCE,
    ) -> None:
        if unknown_grace_s < 0 or equity_tolerance < 0:
            raise ValueError("grace and tolerance cannot be negative")
        self._transport = transport
        self._ledger = ledger
        self._tracker = tracker
        self._clock = clock
        self._grace = timedelta(seconds=unknown_grace_s)
        self._equity_tolerance = equity_tolerance

    def run(
        self,
        *,
        book: BookState,
        since: datetime,
        known_fill_ids: frozenset[str],
        full_history: bool = False,
    ) -> ReconciliationReport:
        """One sweep. ``full_history`` also reads order history (the daily sweep) to find orders
        the venue holds under a ``clientOid`` this ledger has never recorded."""
        now = self._clock.now()
        if self._ledger.mode is RunMode.DRYRUN:
            report = ReconciliationReport(
                at=now,
                orders_checked=0,
                new_fill_ids=(),
                resolved_unknown=(),
                discrepancies=(),
                account=None,
            )
            self._log(EventKind.RECONCILIATION, report)
            return report

        discrepancies: list[Discrepancy] = []
        observed, resolved, checked = self._orders(now, discrepancies)
        # The venue stamps orders and fills on its own clock. This machine measured 3.27 s behind
        # Bitget's server time on 2026-09-24, and a sweep one second after a fill used ``until =
        # now`` and missed it: the fill was stamped "in the future". Reading ahead by a bounded
        # slack costs nothing, because fills are deduplicated by id.
        until = now + VENUE_CLOCK_SLACK
        if full_history:
            checked += self._history(since, until, discrepancies)

        new_fills, fills_read = self._fills(
            book, since, until, known_fill_ids, observed, discrepancies
        )
        expected = self._expected_positions(book, new_fills)
        self._positions(expected, discrepancies)
        self._stops(expected, discrepancies)
        account = self._account(book, discrepancies)

        report = ReconciliationReport(
            at=now,
            orders_checked=checked,
            new_fill_ids=tuple(f.exec_id for f in new_fills),
            resolved_unknown=tuple(resolved),
            discrepancies=tuple(discrepancies),
            account=account,
            fills_read=fills_read,
        )
        self._log(EventKind.RECONCILIATION, report, *(() if account is None else (account.blob,)))
        return report

    # --- orders -------------------------------------------------------------------------------

    def _orders(
        self, now: datetime, discrepancies: list[Discrepancy]
    ) -> tuple[dict[str, VenueOrder], list[str], int]:
        observed: dict[str, VenueOrder] = {}
        resolved: list[str] = []
        checked = 0
        for oid in self._tracker.live():
            checked += 1
            was = self._tracker.state(oid)
            try:
                venue = self._transport.order(client_oid=oid)
            except EnvironmentRefused:
                raise
            except (VenueCallError, TimeoutError, OSError) as exc:
                discrepancies.append(
                    Discrepancy(
                        kind="unknown_order",
                        symbol=self._symbol_of(oid),
                        client_oid=oid,
                        detail=f"order detail unreadable, state kept at {was}: {exc}",
                    )
                )
                continue
            if venue is None:
                self._not_found(oid, was, now, discrepancies, resolved)
                continue
            observed[oid] = venue
            if map_venue_status(venue.status) is not was and not self._apply(oid, venue):
                discrepancies.append(
                    Discrepancy(
                        kind="unknown_order",
                        symbol=venue.symbol,
                        client_oid=oid,
                        detail=f"the venue reports {venue.status} while the ledger holds {was}, "
                        "which is not a legal move; state kept",
                    )
                )
            if was is OrderState.UNKNOWN and self._tracker.state(oid) is not OrderState.UNKNOWN:
                resolved.append(oid)
        return observed, resolved, checked

    def _not_found(
        self,
        oid: str,
        was: OrderState | None,
        now: datetime,
        discrepancies: list[Discrepancy],
        resolved: list[str],
    ) -> None:
        last = self._tracker.last_change(oid)
        sent_at = last.at if last is not None else now
        if (
            was in (OrderState.SUBMITTED, OrderState.UNKNOWN)
            and now - sent_at >= self._grace
            and OrderState.REJECTED in LEGAL_TRANSITIONS[was]
        ):
            change = self._tracker.transition(
                oid,
                OrderState.REJECTED,
                at=now,
                reason=f"the venue has no record of this clientOid {now - sent_at} after it was "
                "sent: it never reached the venue; it is not sent again",
            )
            self._log(EventKind.ORDER_STATE, change)
            if was is OrderState.UNKNOWN:
                resolved.append(oid)
            return
        discrepancies.append(
            Discrepancy(
                kind="unknown_order",
                symbol=self._symbol_of(oid),
                client_oid=oid,
                detail=f"the venue has no record of this order yet (state {was})",
            )
        )

    def _apply(self, oid: str, venue: VenueOrder) -> bool:
        """Move the tracker to the venue's state and log it. False when the move is not legal."""
        current = self._tracker.state(oid)
        target = map_venue_status(venue.status)
        if current is None or target is current:
            return False
        if target not in LEGAL_TRANSITIONS[current]:
            return False
        change = self._tracker.transition(
            oid,
            target,
            at=self._clock.now(),
            reason=f"reconciled: venue orderStatus {venue.status}",
            venue_order_id=venue.venue_order_id,
        )
        self._log(EventKind.ORDER_STATE, change)
        return True

    def _history(self, since: datetime, until: datetime, discrepancies: list[Discrepancy]) -> int:
        history = getattr(self._transport, "history", None)
        if not callable(history):
            return 0
        try:
            orders: Sequence[VenueOrder] = history(since=since, until=until)
        except EnvironmentRefused:
            raise
        except (VenueCallError, TimeoutError, OSError) as exc:
            discrepancies.append(
                Discrepancy(
                    kind="unknown_order",
                    symbol=None,
                    client_oid=None,
                    detail=f"order history unreadable: {exc}",
                )
            )
            return 0
        known = set(self._tracker.known())
        for order in orders:
            if order.client_oid and order.client_oid in known:
                current = self._tracker.state(order.client_oid)
                target = map_venue_status(order.status)
                if (
                    current is not None
                    and target is not current
                    and not self._apply(order.client_oid, order)
                ):
                    discrepancies.append(
                        Discrepancy(
                            kind="unknown_order",
                            symbol=order.symbol,
                            client_oid=order.client_oid,
                            detail=f"venue history says {order.status}, the ledger holds "
                            f"{current}; not a legal move",
                        )
                    )
                continue
            if _venue_initiated(order):
                continue
            discrepancies.append(
                Discrepancy(
                    kind="unknown_order",
                    symbol=order.symbol,
                    client_oid=order.client_oid,
                    detail=f"venue order {order.venue_order_id} ({order.delegate_type}, "
                    f"{order.status}) is not in this ledger",
                )
            )
        return len(orders)

    # --- fills --------------------------------------------------------------------------------

    def _fills(
        self,
        book: BookState,
        since: datetime,
        now: datetime,
        known_fill_ids: frozenset[str],
        observed: Mapping[str, VenueOrder],
        discrepancies: list[Discrepancy],
    ) -> tuple[list[Fill], bool]:
        """The new fills, and whether the venue's fills could be read at all."""
        try:
            fills = self._transport.fills(since=since, until=now)
        except EnvironmentRefused:
            raise
        except (VenueCallError, TimeoutError, OSError) as exc:
            discrepancies.append(
                Discrepancy(
                    kind="missing_fill",
                    symbol=None,
                    client_oid=None,
                    detail=f"fills since {since.isoformat()} unreadable: {exc}",
                )
            )
            return [], False
        ours = set(self._tracker.known())
        running = {s: p.qty for s, p in book.positions.items()}
        new: list[Fill] = []
        written: set[str] = set()
        for fill in sorted(fills, key=lambda f: (f.executed_at, f.exec_id)):
            if (
                fill.exec_id in known_fill_ids
                or fill.exec_id in written
                or self._tracker.has_fill(fill.exec_id)
            ):
                continue
            written.add(fill.exec_id)
            held = running.get(fill.symbol, Decimal(0))
            if fill.client_oid not in ours and not _reduces(held, fill):
                discrepancies.append(
                    Discrepancy(
                        kind="orphan_fill",
                        symbol=fill.symbol,
                        client_oid=fill.client_oid,
                        detail=f"fill {fill.exec_id} ({fill.side} {fill.exec_qty} @ "
                        f"{fill.exec_price}, order {fill.venue_order_id}) is not one of our "
                        "orders and does not reduce the book; written so the book matches the "
                        "venue",
                    )
                )
            running[fill.symbol] = held + _signed(fill)
            self._log(EventKind.FILL, fill, fill.blob)
            self._tracker.note_fill(fill.exec_id)
            new.append(fill)

        seen_by_order: dict[str, Decimal] = {}
        for fill in fills:
            if fill.client_oid:
                seen_by_order[fill.client_oid] = (
                    seen_by_order.get(fill.client_oid, Decimal(0)) + fill.exec_qty
                )
        for oid, venue in observed.items():
            if venue.created_at < since or venue.cum_exec_qty <= 0:
                continue
            got = seen_by_order.get(oid, Decimal(0))
            if got < venue.cum_exec_qty:
                discrepancies.append(
                    Discrepancy(
                        kind="missing_fill",
                        symbol=venue.symbol,
                        client_oid=oid,
                        detail=f"the venue reports {venue.cum_exec_qty} executed, fills show {got}",
                    )
                )
        return new, True

    @staticmethod
    def _expected_positions(book: BookState, new_fills: Iterable[Fill]) -> dict[str, Decimal]:
        expected = {s: p.qty for s, p in book.positions.items() if not p.is_flat}
        for fill in new_fills:
            expected[fill.symbol] = expected.get(fill.symbol, Decimal(0)) + _signed(fill)
        return {s: q for s, q in expected.items() if q != 0}

    # --- positions, stops, account --------------------------------------------------------------

    def _positions(self, expected: Mapping[str, Decimal], discrepancies: list[Discrepancy]) -> None:
        try:
            venue_rows = self._transport.positions()
        except EnvironmentRefused:
            raise
        except (VenueCallError, TimeoutError, OSError) as exc:
            discrepancies.append(
                Discrepancy(
                    kind="position_mismatch",
                    symbol=None,
                    client_oid=None,
                    detail=f"venue positions unreadable: {exc}",
                )
            )
            return
        venue: dict[str, Decimal] = {}
        for row in venue_rows:
            venue[row.symbol] = venue.get(row.symbol, Decimal(0)) + row.qty
        for symbol in sorted(set(expected) | set(venue)):
            ours, theirs = expected.get(symbol, Decimal(0)), venue.get(symbol, Decimal(0))
            if ours != theirs:
                discrepancies.append(
                    Discrepancy(
                        kind="position_mismatch",
                        symbol=symbol,
                        client_oid=None,
                        detail=f"venue holds {theirs}, the book holds {ours}",
                    )
                )

    def _stops(self, expected: Mapping[str, Decimal], discrepancies: list[Discrepancy]) -> None:
        try:
            stops = self._transport.stop_orders()
        except EnvironmentRefused:
            raise
        except (VenueCallError, TimeoutError, OSError) as exc:
            discrepancies.append(
                Discrepancy(
                    kind="missing_stop",
                    symbol=None,
                    client_oid=None,
                    detail=f"venue stop orders unreadable: {exc}",
                )
            )
            return
        by_symbol: dict[str, list[VenueStopOrder]] = {}
        for stop in stops:
            by_symbol.setdefault(stop.symbol, []).append(stop)
        for symbol in sorted(expected):
            protective = [s for s in by_symbol.get(symbol, []) if s.stop_price is not None]
            if not protective:
                discrepancies.append(
                    Discrepancy(
                        kind="missing_stop",
                        symbol=symbol,
                        client_oid=None,
                        detail=f"position {expected[symbol]} has no venue stop-loss",
                    )
                )
            for extra in protective[1:]:
                discrepancies.append(
                    Discrepancy(
                        kind="orphan_stop",
                        symbol=symbol,
                        client_oid=None,
                        detail=f"second stop {extra.venue_id} at {extra.stop_price}",
                    )
                )
        for symbol in sorted(set(by_symbol) - set(expected)):
            for stop in by_symbol[symbol]:
                discrepancies.append(
                    Discrepancy(
                        kind="orphan_stop",
                        symbol=symbol,
                        client_oid=None,
                        detail=f"stop {stop.venue_id} at {stop.stop_price} with no position",
                    )
                )

    def _account(self, book: BookState, discrepancies: list[Discrepancy]) -> AccountSnapshot | None:
        try:
            account = self._transport.account()
        except EnvironmentRefused:
            raise
        except (VenueCallError, TimeoutError, OSError) as exc:
            discrepancies.append(
                Discrepancy(
                    kind="equity_gap",
                    symbol=None,
                    client_oid=None,
                    detail=f"venue account unreadable: {exc}",
                )
            )
            return None
        if account.equity_usdt is not None and book.equity > 0:
            gap = account.equity_usdt - book.equity
            if abs(gap) > book.equity * self._equity_tolerance:
                discrepancies.append(
                    Discrepancy(
                        kind="equity_gap",
                        symbol=None,
                        client_oid=None,
                        detail=f"venue equity {account.equity_usdt} vs book {book.equity} "
                        f"(gap {gap}, tolerance {self._equity_tolerance:%} of book)",
                    )
                )
        return account

    # --- helpers ------------------------------------------------------------------------------

    def _symbol_of(self, oid: str) -> str | None:
        submitted = self._tracker.submitted(oid)
        return submitted.intent.symbol if submitted is not None else None

    def _log(self, kind: EventKind, payload: Model, *blobs: BlobRef | None) -> None:
        self._ledger.append(kind, payload, blobs=[b for b in blobs if b is not None])


def _reduces(held: Decimal, fill: Fill) -> bool:
    signed = _signed(fill)
    return held != 0 and (held > 0) != (signed > 0) and abs(signed) <= abs(held)


_VENUE_INITIATED_MARKERS: Final = ("stop", "tpsl", "liquidation", "delivery", "offset", "reduce")


def _venue_initiated(order: VenueOrder) -> bool:
    """A venue-side order (stop, TP/SL, liquidation, delivery, netting), by its ``delegateType``
    (legacy-docs/uta/trade/Get-Order-Details lists them). Anything else not in the ledger is
    foreign, including an order placed by hand in the Demo web UI."""
    delegate = (order.delegate_type or "").lower()
    return any(marker in delegate for marker in _VENUE_INITIATED_MARKERS)


__all__ = ["EQUITY_TOLERANCE", "UNKNOWN_GRACE_S", "Reconciler"]
