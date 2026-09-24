"""Venue stops: exactly one full-position stop per open position, 4% from its average entry.

Guard G4 puts a preset ``stopLoss`` (triggered on mark) on every order that opens or increases a
position, so protection is atomic with the fill. That preset is set at the *order's* reference
price; once a position has been built from several fills, or partly reduced, the stop that matters
is the one at the *position's* average entry. :class:`StopManager` reconciles the venue to that
(DESIGN.md §11.4):

* a position with no stop gets one (``placed``);
* a position whose stop sits at the wrong price gets a new one first and loses the old one second
  (``replaced`` then ``cancelled``), so the position is never left without protection. If the venue
  refuses a second full-position stop while the first exists (NOT VERIFIED either way, DESIGN.md
  §20), the old one is cancelled and the new one placed straight after. If the new placement timed
  out, its outcome is unknown, so the old stop is kept and the next sync decides;
* a stop at the right price is kept (``verified``) and any duplicates for the same symbol are
  cancelled;
* a stop for a symbol the *venue* holds no position in is cancelled (an orphan: the position was
  closed or flipped).

**The venue's positions decide, not the ledger's.** The manager is handed the venue's own position
read alongside its stops. Where the two agree, the ledger's position (with its average entry) is
protected; where they disagree (a fill the ledger has not folded in yet, a manual or venue-side
position) the venue's position is protected at the venue's average price, because that is the
exposure a stop must cover. A stop is an orphan only when the venue is flat in its symbol, so a
ledger that missed a fill can never cancel the stop on a live venue position. When the venue's
positions could not be read, nothing at all is cancelled (placements still go ahead): a stop can
only reduce risk, and an unread venue is no evidence that it protects nothing.

A position the manager could not protect is reported as ``missing``; it is never silent. A stop
cancellation the venue refused leaves the stop in place (it can only reduce risk) and is kept in
:meth:`StopManager.errors` for the runtime to log; reconciliation reports it as an orphan until it
is gone.

The stop price is ``avg_entry × (1 − 4%)`` for a long and ``× (1 + 4%)`` for a short, rounded to the
instrument's price step *towards* the entry, so the loss at the stop is never more than 4%. When no
instrument specification is supplied the price is not rounded and the venue may refuse it; the
refusal is reported as ``missing``.

Each stop carries a deterministic ``clientOid`` derived from the position and the price, so a
repeated placement after a timeout reaches the venue under the same id and is de-duplicated there
(``placeStrategyOrder``: "the idempotent validity period is six hours").
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import Final

from sentiment_agent.execution.bgc import (
    BgcTransport,
    TransportRefusedError,
    VenueWriteError,
)
from sentiment_agent.execution.simulated import SimulatedVenue
from sentiment_agent.hashing import content_hash
from sentiment_agent.types import (
    CLIENT_OID_HEX,
    CLIENT_OID_PREFIX,
    BookState,
    Clock,
    InstrumentSpec,
    Policy,
    Position,
    StopSync,
    VenuePosition,
    VenueStopOrder,
)

PRICE_MATCH_REL: Final = Decimal("0.000001")
"""Without a price step, two stop prices within one part per million are the same stop."""

VENUE_ONLY_OPENED_AT: Final = datetime(1970, 1, 1, tzinfo=UTC)
"""``opened_at`` for a position only the venue reports: fixed, so its stop's ``clientOid`` stays the
same from one sync to the next and a retried placement is de-duplicated by the venue."""


@dataclass(frozen=True, slots=True)
class StopError:
    """A stop write the venue refused or that timed out."""

    symbol: str
    action: str
    venue_id: str | None
    message: str
    outcome_unknown: bool


def stop_client_oid(symbol: str, position: Position, stop_price: Decimal, *, attempt: str) -> str:
    """Deterministic, 32 characters, the same shape as an order's ``clientOid``."""
    digest = content_hash(
        {
            "kind": "stop",
            "symbol": symbol,
            "qty": position.qty,
            "avg_entry": position.avg_entry,
            "opened_at": position.opened_at,
            "stop_price": stop_price,
            "attempt": attempt,
        }
    )
    return CLIENT_OID_PREFIX + digest[:CLIENT_OID_HEX]


def desired_stop_price(
    position: Position, *, stop_loss_pct: float, spec: InstrumentSpec | None
) -> Decimal:
    """The stop for a position, rounded to the price step towards the entry."""
    pct = Decimal(str(stop_loss_pct))
    long = position.qty > 0
    raw = position.avg_entry * (1 - pct) if long else position.avg_entry * (1 + pct)
    if spec is None or spec.price_step <= 0:
        return raw
    steps = raw / spec.price_step
    rounded = steps.to_integral_value(rounding=ROUND_CEILING if long else ROUND_FLOOR)
    return rounded * spec.price_step


class StopManager:
    """Reconciles venue stops to one full-position stop per open position."""

    def __init__(
        self,
        *,
        transport: BgcTransport | SimulatedVenue,
        policy: Policy,
        clock: Clock,
        specs: Mapping[str, InstrumentSpec] | None = None,
    ) -> None:
        self._transport = transport
        self._policy = policy
        self._clock = clock
        self._specs = dict(specs or {})
        self._errors: list[StopError] = []

    def errors(self) -> tuple[StopError, ...]:
        """Stop writes that failed during the last :meth:`sync`."""
        return tuple(self._errors)

    def _same_price(self, symbol: str, a: Decimal, b: Decimal) -> bool:
        spec = self._specs.get(symbol)
        if spec is not None and spec.price_step > 0:
            return abs(a - b) < spec.price_step / 2
        return abs(a - b) <= abs(b) * PRICE_MATCH_REL

    def _place(
        self, symbol: str, position: Position, price: Decimal, *, attempt: str
    ) -> StopSync | None:
        try:
            return self._transport.place_stop(
                symbol=symbol,
                pos_side="long" if position.qty > 0 else "short",
                qty=abs(position.qty),
                stop_price=price,
                client_oid=stop_client_oid(symbol, position, price, attempt=attempt),
            )
        except (VenueWriteError, TransportRefusedError) as exc:
            self._errors.append(
                StopError(
                    symbol=symbol,
                    action="place",
                    venue_id=None,
                    message=str(exc),
                    outcome_unknown=isinstance(exc, VenueWriteError) and exc.outcome_unknown,
                )
            )
            return None

    def _cancel(self, symbol: str, venue_id: str | None) -> StopSync | None:
        if venue_id is None:
            self._errors.append(
                StopError(
                    symbol=symbol,
                    action="cancel",
                    venue_id=None,
                    message="the venue listed a stop without an order id; it cannot be cancelled",
                    outcome_unknown=False,
                )
            )
            return None
        try:
            return self._transport.cancel_stop(symbol=symbol, venue_id=venue_id)
        except (VenueWriteError, TransportRefusedError) as exc:
            self._errors.append(
                StopError(
                    symbol=symbol,
                    action="cancel",
                    venue_id=venue_id,
                    message=str(exc),
                    outcome_unknown=isinstance(exc, VenueWriteError) and exc.outcome_unknown,
                )
            )
            return None

    def _missing(self, symbol: str, price: Decimal | None) -> StopSync:
        return StopSync(
            symbol=symbol, action="missing", stop_price=price, venue_id=None, at=self._clock.now()
        )

    @staticmethod
    def protected_positions(
        book: BookState, venue_positions: Sequence[VenuePosition] | None
    ) -> dict[str, Position]:
        """The positions stops must cover: the venue's where it was read, the ledger's otherwise.

        A symbol where the venue and the ledger hold the same quantity keeps the ledger's position
        (its average entry is the one the ledger's P&L uses). A disagreement is resolved to the
        venue: its quantity at its average price, or at the ledger's average entry when the venue
        gives none and the ledger holds the same side. A venue position that cannot be priced
        either way is returned with ``avg_entry`` 0 and is reported ``missing`` by :meth:`sync`.
        With no venue read at all, the ledger's open positions are used."""
        ledger = {s: p for s, p in book.positions.items() if not p.is_flat}
        if venue_positions is None:
            return ledger
        venue: dict[str, tuple[Decimal, Decimal | None]] = {}
        for row in venue_positions:
            qty, avg = venue.get(row.symbol, (Decimal(0), None))
            venue[row.symbol] = (qty + row.qty, row.avg_price if row.avg_price is not None else avg)
        out: dict[str, Position] = {}
        for symbol, (qty, venue_avg) in venue.items():
            if qty == 0:
                continue
            ours = ledger.get(symbol)
            if ours is not None and ours.qty == qty:
                out[symbol] = ours
                continue
            avg = venue_avg if venue_avg is not None and venue_avg > 0 else None
            if avg is None and ours is not None and (ours.qty > 0) == (qty > 0):
                avg = ours.avg_entry
            opened = ours.opened_at if ours is not None else VENUE_ONLY_OPENED_AT
            out[symbol] = Position(
                symbol=symbol,
                qty=qty,
                avg_entry=avg if avg is not None else Decimal(0),
                opened_at=opened,
                last_increase_at=ours.last_increase_at if ours is not None else opened,
                realized_pnl=Decimal(0),
                fees_paid=Decimal(0),
                stop_price=None,
                stop_venue_id=None,
                last_decision_id=None,
            )
        return out

    def sync(
        self,
        book: BookState,
        venue_stops: Sequence[VenueStopOrder],
        *,
        venue_positions: Sequence[VenuePosition] | None,
    ) -> list[StopSync]:
        """Bring the venue's stops to one per open venue position; report every action taken.

        ``venue_positions`` is the venue's own position read, or ``None`` when that read failed, in
        which case no stop is cancelled (module docstring)."""
        self._errors = []
        can_cancel = venue_positions is not None
        results: list[StopSync] = []
        by_symbol: dict[str, list[VenueStopOrder]] = {}
        for stop in venue_stops:
            by_symbol.setdefault(stop.symbol, []).append(stop)

        protected = self.protected_positions(book, venue_positions)
        for symbol in sorted(protected):
            position = protected[symbol]
            if position.avg_entry <= 0:
                # A venue position with no price to set a stop from: keep whatever stop it has.
                if not by_symbol.pop(symbol, []):
                    results.append(self._missing(symbol, None))
                continue
            desired = desired_stop_price(
                position,
                stop_loss_pct=self._policy.stop_loss_pct,
                spec=self._specs.get(symbol),
            )
            existing = by_symbol.pop(symbol, [])
            keep = next(
                (
                    s
                    for s in existing
                    if s.stop_price is not None and self._same_price(symbol, s.stop_price, desired)
                ),
                None,
            )
            if keep is not None:
                results.append(
                    StopSync(
                        symbol=symbol,
                        action="verified",
                        stop_price=keep.stop_price,
                        venue_id=keep.venue_id,
                        at=self._clock.now(),
                        blob=keep.blob,
                    )
                )
                if can_cancel:
                    for extra in existing:
                        if extra is not keep:
                            cancelled = self._cancel(symbol, extra.venue_id)
                            if cancelled is not None:
                                results.append(cancelled)
                continue

            if not existing:
                placed = self._place(symbol, position, desired, attempt="place")
                results.append(placed if placed is not None else self._missing(symbol, desired))
                continue

            # Wrong price: the new stop first, then the old ones.
            placed = self._place(symbol, position, desired, attempt="replace")
            if placed is not None:
                results.append(placed.model_copy(update={"action": "replaced"}))
                if can_cancel:
                    for old in existing:
                        cancelled = self._cancel(symbol, old.venue_id)
                        if cancelled is not None:
                            results.append(cancelled)
                continue
            if not can_cancel or (self._errors and self._errors[-1].outcome_unknown):
                # The new stop may exist, or the venue's positions are unread: the old stops still
                # protect, and the next sync decides.
                continue
            # The venue may allow one full-position stop at a time: clear, then place at once.
            for old in existing:
                cancelled = self._cancel(symbol, old.venue_id)
                if cancelled is not None:
                    results.append(cancelled)
            placed = self._place(symbol, position, desired, attempt="replace-after-cancel")
            if placed is not None:
                results.append(placed.model_copy(update={"action": "replaced"}))
            else:
                results.append(self._missing(symbol, desired))

        if not can_cancel:
            return results
        # Every symbol left is one the venue holds no position in: ``protected`` has the rest.
        for symbol in sorted(by_symbol):
            for orphan in by_symbol[symbol]:
                cancelled = self._cancel(symbol, orphan.venue_id)
                if cancelled is not None:
                    results.append(cancelled)
        return results


__all__ = [
    "PRICE_MATCH_REL",
    "VENUE_ONLY_OPENED_AT",
    "StopError",
    "StopManager",
    "desired_stop_price",
    "stop_client_oid",
]
