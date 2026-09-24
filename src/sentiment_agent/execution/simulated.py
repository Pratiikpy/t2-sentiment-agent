"""A simulated venue with the same transport interface as Agent Hub, for the SIMULATED mode.

It lets the whole pipeline run end to end before the Demo key exists (DESIGN.md §5, §11.6), and it
prices everything from Bitget's own Demo venue so a simulated result is read on the same prices a
paper result will be:

* **Fills.** A market order fills in full at the keyless Demo ask (buys) or bid (sells) at the
  moment it is placed, and pays ``fee_rate`` of the fill value. The default, 6 bps, is the Demo
  taker fee on all 14 universe instruments (``validation/demo_venue/universe_probe.json``,
  ``takerFeeRate 0.0006``). Maker fills are never simulated: every order here is a taker (G8).
* **Netting.** One position per symbol, as in one-way mode. An order that crosses zero closes the
  position and opens the remainder on the other side, reported as two fills (``close`` then
  ``open``) so realised P&L and trade boundaries are exact.
* **Reduce-only.** A reduce-only order against a flat book is rejected; one larger than the
  position is filled up to the position. The second rule is this simulator's choice: Bitget's
  handling of an oversized reduce-only market order is NOT VERIFIED, and the planner never asks
  for one (a stop firing between planning and sending is how it would happen).
* **Stops.** One full-position stop per symbol, set by an order's preset ``stopLoss`` or by
  :meth:`place_stop`; a newer one replaces the older. :meth:`poll` triggers a long's stop when the
  Demo *mark* is at or below it and a short's when at or above (G4 triggers on mark), and fills
  the exit at the Demo bid or ask like any market order. A stop disappears with its position.

Nothing here reads a credential or places anything on a real venue. Fills carry
:attr:`~sentiment_agent.types.FillVenue.SIMULATED`, and the executor refuses to pair this venue with
a PAPER ledger, so a simulated fill can never enter the scored record.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Final

from sentiment_agent.execution.bgc import (
    HOLD_ONE_WAY,
    PLACE_ORDER_PATH,
    TransportRefusedError,
    VenueReadError,
    build_place_args,
    would_send,
)
from sentiment_agent.types import (
    AccountSnapshot,
    ApprovedOrder,
    Clock,
    DryRunPreview,
    FeeLine,
    Fill,
    FillVenue,
    MarketData,
    OrderIntent,
    PriceSource,
    Quote,
    Side,
    StopSync,
    VenueAck,
    VenueOrder,
    VenueOrderStatus,
    VenuePosition,
    VenueRejection,
    VenueStopOrder,
    VenueUnknown,
)

DEMO_TAKER_FEE: Final = Decimal("0.0006")


@dataclass(slots=True)
class _Position:
    qty: Decimal = Decimal(0)
    avg: Decimal = Decimal(0)
    realized: Decimal = Decimal(0)


@dataclass(slots=True)
class _Stop:
    venue_id: str
    stop_price: Decimal
    client_oid: str | None


@dataclass(slots=True)
class _Records:
    orders: dict[str, VenueOrder] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)


class SimulatedVenue:
    """:class:`~sentiment_agent.types.VenueTransport` backed by keyless Demo quotes."""

    def __init__(
        self,
        *,
        market: MarketData,
        clock: Clock,
        starting_equity: Decimal,
        fee_rate: Decimal = DEMO_TAKER_FEE,
    ) -> None:
        if starting_equity <= 0:
            raise ValueError("starting equity must be positive")
        if fee_rate < 0:
            raise ValueError("fee rate cannot be negative")
        self._market = market
        self._clock = clock
        self._starting_equity = starting_equity
        self._fee_rate = fee_rate
        self._positions: dict[str, _Position] = {}
        self._stops: dict[str, _Stop] = {}
        self._book = _Records()
        self._fees_paid = Decimal(0)
        self._order_seq = 0
        self._fill_seq = 0
        self._stop_seq = 0

    # --- VenueTransport ---------------------------------------------------------------------

    @property
    def venue(self) -> FillVenue:
        return FillVenue.SIMULATED

    def preview(self, intent: OrderIntent) -> DryRunPreview:
        """What Agent Hub would send for this intent, built by the same function the transport
        checks ``bgc``'s preview against. Nothing is run."""
        return DryRunPreview(
            client_oid=intent.client_oid,
            operation_id="placeOrder",
            method="POST",
            path=PLACE_ORDER_PATH,
            would_send=would_send(intent, hold_mode=HOLD_ONE_WAY),
            argv=tuple(build_place_args(intent, hold_mode=HOLD_ONE_WAY, dry_run=True)),
            captured_at=self._clock.now(),
            blob=None,
        )

    def place(self, order: ApprovedOrder) -> VenueAck | VenueRejection | VenueUnknown:
        if not order.verify():
            raise TransportRefusedError("the approval does not match its intent; nothing was sent")
        intent = order.intent
        now = self._clock.now()
        if any(o.client_oid == intent.client_oid for o in self._book.orders.values()):
            # As BgcTransport reads a duplicate-clientOid refusal: the order exists, so the send's
            # outcome is resolved by reading it back, never by treating it as refused.
            return VenueUnknown(
                client_oid=intent.client_oid,
                at=now,
                reason="simulated venue: the venue already knows this clientOid",
            )
        quote = self._quote(intent.symbol)
        if quote is None:
            return self._reject(intent, None, "no Demo quote for the symbol", "network", True)
        position = self._positions.setdefault(intent.symbol, _Position())
        qty = intent.qty
        if intent.reduce_only:
            closes = (position.qty > 0 and intent.side is Side.SELL) or (
                position.qty < 0 and intent.side is Side.BUY
            )
            if not closes:
                return self._reject(
                    intent, None, "reduce-only order with no position to reduce", "param"
                )
            qty = min(qty, abs(position.qty))
        price = quote.ask if intent.side is Side.BUY else quote.bid
        if price <= 0:
            return self._reject(intent, None, "Demo quote has no price on this side", "network")
        self._order_seq += 1
        venue_order_id = f"sim-order-{self._order_seq:08d}"
        fills = self._execute(
            symbol=intent.symbol,
            side=intent.side,
            qty=qty,
            price=price,
            venue_order_id=venue_order_id,
            client_oid=intent.client_oid,
            at=now,
        )
        if intent.stop_loss_price is not None and self._positions[intent.symbol].qty != 0:
            self._stop_seq += 1
            self._stops[intent.symbol] = _Stop(
                venue_id=f"sim-stop-{self._stop_seq:08d}",
                stop_price=intent.stop_loss_price,
                client_oid=None,
            )
        self._record_order(
            venue_order_id=venue_order_id,
            client_oid=intent.client_oid,
            symbol=intent.symbol,
            side=intent.side,
            qty=intent.qty,
            fills=fills,
            reduce_only=intent.reduce_only,
            delegate_type="market",
            at=now,
        )
        return VenueAck(
            client_oid=intent.client_oid, venue_order_id=venue_order_id, acked_at=now, blob=None
        )

    def order(self, *, client_oid: str) -> VenueOrder | None:
        for order in self._book.orders.values():
            if order.client_oid == client_oid:
                return order
        return None

    def history(self, *, since: datetime, until: datetime) -> list[VenueOrder]:
        return [o for o in self._book.orders.values() if since <= o.created_at <= until]

    def fills(self, *, since: datetime, until: datetime) -> list[Fill]:
        return [f for f in self._book.fills if since <= f.executed_at <= until]

    def positions(self) -> list[VenuePosition]:
        return [
            VenuePosition(symbol=symbol, qty=p.qty, avg_price=p.avg, blob=None)
            for symbol, p in sorted(self._positions.items())
            if p.qty != 0
        ]

    def stop_orders(self) -> list[VenueStopOrder]:
        return [
            VenueStopOrder(symbol=symbol, venue_id=s.venue_id, stop_price=s.stop_price, blob=None)
            for symbol, s in sorted(self._stops.items())
        ]

    def account(self) -> AccountSnapshot:
        """Equity = starting equity + realised P&L − fees + unrealised P&L at the Demo mark."""
        open_symbols = [s for s, p in self._positions.items() if p.qty != 0]
        quotes = self._market.quotes(PriceSource.DEMO, open_symbols) if open_symbols else {}
        unrealized = Decimal(0)
        for symbol in open_symbols:
            quote = quotes.get(symbol)
            if quote is None:
                raise VenueReadError(f"no Demo mark for {symbol}; equity cannot be computed")
            position = self._positions[symbol]
            unrealized += (quote.mark - position.avg) * position.qty
        equity = self.realized_equity + unrealized
        return AccountSnapshot(
            at=self._clock.now(), equity_usdt=equity, available_usdt=None, blob=None
        )

    # --- stops ------------------------------------------------------------------------------

    def place_stop(
        self, *, symbol: str, pos_side: str, qty: Decimal, stop_price: Decimal, client_oid: str
    ) -> StopSync:
        """A full-position stop; replaces any stop the symbol already has, as ``tpslMode full``."""
        position = self._positions.get(symbol)
        if position is None or position.qty == 0:
            raise TransportRefusedError(f"no {symbol} position to protect")
        if (pos_side == "long") != (position.qty > 0):
            raise TransportRefusedError(f"{symbol} position is not {pos_side}")
        if qty <= 0 or stop_price <= 0:
            raise TransportRefusedError("stop quantity and price must be positive")
        self._stop_seq += 1
        stop = _Stop(
            venue_id=f"sim-stop-{self._stop_seq:08d}", stop_price=stop_price, client_oid=client_oid
        )
        self._stops[symbol] = stop
        return StopSync(
            symbol=symbol,
            action="placed",
            stop_price=stop_price,
            venue_id=stop.venue_id,
            at=self._clock.now(),
        )

    def cancel_stop(self, *, symbol: str, venue_id: str) -> StopSync:
        stop = self._stops.get(symbol)
        if stop is None or stop.venue_id != venue_id:
            raise TransportRefusedError(f"no stop {venue_id} on {symbol}")
        del self._stops[symbol]
        return StopSync(
            symbol=symbol,
            action="cancelled",
            stop_price=None,
            venue_id=venue_id,
            at=self._clock.now(),
        )

    def poll(self) -> list[Fill]:
        """Trigger stops on the Demo mark. Returns the fills this poll produced."""
        if not self._stops:
            return []
        quotes = self._market.quotes(PriceSource.DEMO, sorted(self._stops))
        produced: list[Fill] = []
        now = self._clock.now()
        for symbol in sorted(self._stops):
            stop = self._stops[symbol]
            position = self._positions.get(symbol)
            quote = quotes.get(symbol)
            if position is None or position.qty == 0:
                del self._stops[symbol]
                continue
            if quote is None:
                continue
            long = position.qty > 0
            triggered = quote.mark <= stop.stop_price if long else quote.mark >= stop.stop_price
            if not triggered:
                continue
            side = Side.SELL if long else Side.BUY
            price = quote.bid if long else quote.ask
            qty = abs(position.qty)
            self._order_seq += 1
            venue_order_id = f"sim-order-{self._order_seq:08d}"
            fills = self._execute(
                symbol=symbol,
                side=side,
                qty=qty,
                price=price,
                venue_order_id=venue_order_id,
                client_oid=None,
                at=now,
            )
            self._record_order(
                venue_order_id=venue_order_id,
                client_oid=None,
                symbol=symbol,
                side=side,
                qty=qty,
                fills=fills,
                reduce_only=True,
                delegate_type="position_stop_loss_market",
                at=now,
            )
            produced.extend(fills)
        return produced

    # --- extras for tests and the runtime ---------------------------------------------------

    @property
    def realized_equity(self) -> Decimal:
        """Starting equity + realised P&L − fees (no unrealised)."""
        realized = sum((p.realized for p in self._positions.values()), Decimal(0))
        return self._starting_equity + realized - self._fees_paid

    @property
    def fees_paid(self) -> Decimal:
        return self._fees_paid

    # --- internals --------------------------------------------------------------------------

    def _quote(self, symbol: str) -> Quote | None:
        return self._market.quotes(PriceSource.DEMO, [symbol]).get(symbol)

    def _reject(
        self,
        intent: OrderIntent,
        code: str | None,
        message: str,
        category: str,
        retryable: bool = False,
    ) -> VenueRejection:
        return VenueRejection(
            client_oid=intent.client_oid,
            code=code,
            message=f"simulated venue: {message}",
            category=category,
            retryable=retryable,
            at=self._clock.now(),
            blob=None,
        )

    def _fill(
        self,
        *,
        symbol: str,
        side: Side,
        qty: Decimal,
        price: Decimal,
        venue_order_id: str,
        client_oid: str | None,
        trade_side: str,
        pnl: Decimal | None,
        at: datetime,
    ) -> Fill:
        self._fill_seq += 1
        value = qty * price
        fee = value * self._fee_rate
        self._fees_paid += fee
        return Fill(
            exec_id=f"sim-fill-{self._fill_seq:08d}",
            venue_order_id=venue_order_id,
            client_oid=client_oid,
            symbol=symbol,
            side=side,
            exec_price=price,
            exec_qty=qty,
            exec_value=value,
            fee_paid=fee,
            fee_coin="USDT",
            trade_scope="taker",
            trade_side="open" if trade_side == "open" else "close",
            exec_pnl=pnl,
            executed_at=at,
            venue=FillVenue.SIMULATED,
        )

    def _execute(
        self,
        *,
        symbol: str,
        side: Side,
        qty: Decimal,
        price: Decimal,
        venue_order_id: str,
        client_oid: str | None,
        at: datetime,
    ) -> list[Fill]:
        position = self._positions.setdefault(symbol, _Position())
        signed = qty if side is Side.BUY else -qty
        fills: list[Fill] = []
        closing = min(qty, abs(position.qty)) if position.qty * signed < 0 else Decimal(0)
        if closing > 0:
            direction = Decimal(1) if position.qty > 0 else Decimal(-1)
            pnl = (price - position.avg) * closing * direction
            position.realized += pnl
            position.qty -= closing * direction
            if position.qty == 0:
                position.avg = Decimal(0)
                self._stops.pop(symbol, None)
            fills.append(
                self._fill(
                    symbol=symbol,
                    side=side,
                    qty=closing,
                    price=price,
                    venue_order_id=venue_order_id,
                    client_oid=client_oid,
                    trade_side="close",
                    pnl=pnl,
                    at=at,
                )
            )
        opening = qty - closing
        if opening > 0:
            add = opening if side is Side.BUY else -opening
            new_qty = position.qty + add
            position.avg = (position.avg * abs(position.qty) + price * opening) / abs(new_qty)
            position.qty = new_qty
            fills.append(
                self._fill(
                    symbol=symbol,
                    side=side,
                    qty=opening,
                    price=price,
                    venue_order_id=venue_order_id,
                    client_oid=client_oid,
                    trade_side="open",
                    pnl=None,
                    at=at,
                )
            )
        self._book.fills.extend(fills)
        return fills

    def _record_order(
        self,
        *,
        venue_order_id: str,
        client_oid: str | None,
        symbol: str,
        side: Side,
        qty: Decimal,
        fills: Sequence[Fill],
        reduce_only: bool,
        delegate_type: str,
        at: datetime,
    ) -> None:
        executed = sum((f.exec_qty for f in fills), Decimal(0))
        value = sum((f.exec_value for f in fills), Decimal(0))
        fees = sum((f.fee_paid for f in fills), Decimal(0))
        self._book.orders[venue_order_id] = VenueOrder(
            venue_order_id=venue_order_id,
            client_oid=client_oid,
            symbol=symbol,
            side=side,
            order_type="market",
            qty=qty,
            cum_exec_qty=executed,
            cum_exec_value=value,
            avg_price=value / executed if executed > 0 else None,
            status=VenueOrderStatus.FILLED if executed > 0 else VenueOrderStatus.CANCELLED,
            reduce_only=reduce_only,
            delegate_type=delegate_type,
            cancel_reason=None if executed > 0 else "nothing to execute",
            fees=(FeeLine(coin="USDT", raw=fees),) if fees else (),
            created_at=at,
            updated_at=at,
            blob=None,
        )


__all__ = ["DEMO_TAKER_FEE", "SimulatedVenue"]
