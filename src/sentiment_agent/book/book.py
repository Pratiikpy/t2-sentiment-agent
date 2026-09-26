"""Positions, closed trades and the book's state, rebuilt from fills and nothing else.

The book never holds anything a restart could lose: every number here is a fold over the fills,
stop syncs and hourly marks the ledger records (``book/projection.py`` feeds them in). The same
fills applied in any arrival order give the same book.

**Positions** (one-way netting, one position per symbol, as UTA Demo runs them): signed quantity,
average entry at cost, gross realized P&L and fees paid. The method is NautilusTrader's, studied
and reimplemented (LGPL-3.0, nothing copied; commit ``9d89383``):

* an opening or increasing fill moves the average entry to the quantity-weighted mean of the old
  entry and the fill price (``crates/model/src/position.rs:503-617``, ``calculate_avg_px_open_px``);
* a reducing fill realizes ``(price - avg_entry) x qty x direction`` on the reduced quantity and
  leaves the average entry where it was (same lines; ``calculate_pnl_raw`` at ``:1081-1101`` caps
  the quantity at the position's size);
* a fill larger than the position it reduces **flips** it: the position's size closes at the fill
  price, the remainder opens the other side at the same price, and the fill's fee is split between
  the two in proportion to quantity (``crates/execution/src/engine/mod.rs:3896-3925``,
  ``flip_position``: ``commission1 = commission x position_qty / fill_qty``, ``commission2 =
  commission - commission1``).

Departures from Nautilus, on purpose: realized P&L here is **gross** (price only) and fees are kept
apart, because ``equity = starting equity + realized - fees + unrealized`` (DESIGN.md §13) and a
closed trade reports gross, fees and net separately (``types.ClosedTrade``). Nautilus nets the
commission into ``realized_pnl``.

**Time order, not arrival order.** Fills are folded in the order the venue executed them
(``executed_at``; at the same instant a closing fill before an opening one, then the venue's own
trade id, so nothing depends on the order fills were logged in). The ledger does not always receive
them in that order: a venue stop that fired at 13:05 is picked up
by the 15-minute reconciliation sweep, possibly after an order the agent sent at 13:10. Folding in
ledger order would book the 13:10 buy as an increase of a position the venue had already closed,
and the book would disagree with the venue's own position. So each symbol is re-folded from its
time-ordered fills whenever one arrives; the result is independent of arrival order by
construction. A late fill can therefore change trades already reported: :meth:`BookBuilder.
closed_trades` is always authoritative.

**Closed trades** run flat to flat on one symbol; a flip closes one trade and opens the next, the
same definition as the envelope simulation the expected numbers come from
(``validation/demo_venue/envelope_clean.py:36-66``: a position opens, is marked, and closes with its
exit cost; win rate is over those closes). A trade's P&L includes the fees of every fill in it,
entry and exit (``policy.METRICS`` ``win_rate``).

**Exit reasons.** How a trade ended, from what the caller knows about the closing fill:

* ``stop_filled``: the closing order's Bitget ``delegateType`` is a stop-loss type
  (:data:`STOP_LOSS_DELEGATE_TYPES`), or the caller identified it as the venue stop
  (``cause=ProtectiveReason.STOP_FILLED``);
* ``take_profit_filled`` / ``liquidation``: the other venue-originated types this agent can meet;
* ``venue_order:<delegateType>`` for any other venue-originated order, ``venue_initiated`` when the
  closing order is not one this agent planned and its type is unknown (a manual order in the Demo
  UI would land here, and is never passed off as a stop);
* ``flip``: one of the agent's own orders crossed zero;
* ``protective_exit`` (``:<cause>`` when known): the kernel closed it without a model decision, or a
  guard forced the exit inside a decision (planner purpose ``PROTECTIVE_EXIT``);
* ``model_close``: the model's own reduction or close.

The ``delegateType`` values are Bitget's UTA list (``https://www.bitget.com/api-doc/uta/trade/
Get-Order-History``, response field ``delegateType``, read 2026-09-24).

**Book-level state** (:meth:`BookBuilder.state`):

* ``day_open_equity`` is the equity at the most recent 00:00 UTC: the hourly ``MARK`` logged at
  midnight when there is one; otherwise, if every position was flat at midnight, the exact
  realized equity at midnight; otherwise the first mark of the day; otherwise the equity now. The
  envelope simulation measures its daily kill from the same point (``envelope_clean.py:40,59``).
* ``peak_equity`` is the highest of the starting equity, every hourly mark up to ``at`` and the
  equity now. The hourly grid is deliberate: it is what the ledger records, it is the grid the
  published max drawdown uses (``policy.METRICS``), and it is the grid the breaker's thresholds
  were calibrated on (2.5% reduce-only sits past the worst no-edge 72h drawdown, -2.48%, measured
  on hourly marks, ``envelope_clean.py:62``). An intra-hour high was never logged, cannot be
  reproduced from the log, and would trip the breaker more often than it was calibrated to.
* ``rebalances_today`` counts, per symbol, the **model-initiated orders** whose first fill falls on
  the UTC day of ``at``: fills that carry a decision id and a purpose other than
  ``PROTECTIVE_EXIT``. Kernel exits, guard-forced exits and venue stops are not the model's
  rebalances (``types.BookState``; ``kernel/planner.py`` ``forced_by_guard``). An order is counted
  once however many fills it has.
* ``fees_today`` sums the fees of fills executed on the UTC day of ``at``.
* ``consecutive_losses`` counts the most recent closed trades with ``net_pnl < 0``, in closing
  order; a trade that did not lose resets it (the ARGUS rule,
  ``argus/src/argus/eval/risk_layer_comparison.py:284``, MIT, same author).

**Stops.** A position shows the venue stop that protects *this* position: the preset stop of its
opening order and every later :class:`~sentiment_agent.types.StopSync`. A stop belongs to the trade
it was set on; when the trade closes or flips, the next trade starts without one until a sync says
otherwise. A cancellation clears the stop only when it names the stop currently in force (the
stop manager cancels the old stop *after* placing its replacement, ``execution/stops.py``).

The book's arithmetic does not depend on the policy it is built under, deliberately: an amendment
must never rewrite the P&L of fills that already happened.
"""

import bisect
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext
from typing import Final, Literal

from sentiment_agent.types import (
    Activation,
    BookState,
    ClosedTrade,
    Fill,
    FillVenue,
    GuardId,
    MarkPoint,
    OrderPurpose,
    Policy,
    Position,
    PriceSource,
    ProtectiveReason,
    Side,
    StopSync,
)

QUOTE_COIN: Final = "USDT"
"""Every universe instrument is a USDT-margined perpetual; a fee in any other coin cannot be booked
without a conversion price, so it is refused rather than guessed."""

FILL_CLOCK_TOLERANCE: Final = timedelta(minutes=5)
"""How far a fill's venue timestamp may run ahead of the ``at`` a state is asked for.

Venue and local clocks drift; a fill stamped a few seconds after our ``now`` is ordinary. A fill
minutes in the future of the requested instant means a book is being asked for a time before
fills it already holds, which is a caller's error, and :meth:`BookBuilder.state` refuses it."""

EXIT_MODEL_CLOSE: Final = "model_close"
EXIT_FLIP: Final = "flip"
EXIT_PROTECTIVE: Final = "protective_exit"
EXIT_STOP: Final = ProtectiveReason.STOP_FILLED.value
EXIT_TAKE_PROFIT: Final = "take_profit_filled"
EXIT_LIQUIDATION: Final = "liquidation"
EXIT_VENUE_INITIATED: Final = "venue_initiated"
EXIT_VENUE_ORDER_PREFIX: Final = "venue_order:"

STOP_LOSS_DELEGATE_TYPES: Final[frozenset[str]] = frozenset(
    {
        "stop_loss_market",
        "stop_loss_limit",
        "stop_loss_chase",
        "position_stop_loss_market",
        "position_stop_loss_limit",
        "trader_stop_loss",
        "move_stop_market",
        "move_stop_limit",
    }
)
"""Bitget UTA ``delegateType`` values that are a stop loss (trailing stops included)."""

TAKE_PROFIT_DELEGATE_TYPES: Final[frozenset[str]] = frozenset(
    {
        "stop_profit_market",
        "stop_profit_limit",
        "stop_profit_chase",
        "position_stop_profit_market",
        "position_stop_profit_limit",
        "trader_stop_profit",
    }
)
LIQUIDATION_DELEGATE_TYPES: Final[frozenset[str]] = frozenset({"liquidation"})

DECIMAL_CONTEXT: Final = Context(prec=28, rounding=ROUND_HALF_EVEN)
"""The decimal context every book computation runs in, fixed so a caller that changes the thread's
context cannot change the book (and a replay in another process computes the same digits)."""

_ZERO: Final = Decimal(0)

Cause = ProtectiveReason | GuardId
"""Why a fill happened when the model did not ask for it: a protective reason, or the guard that
forced the exit inside a decision."""


class BookError(ValueError):
    """A fill, mark or request the book cannot account for honestly."""


class MissingMarkError(BookError):
    """An open position has no mark, so equity cannot be computed."""


def require_utc(at: datetime, *, what: str = "timestamp") -> datetime:
    if at.tzinfo is None or at.utcoffset() != timedelta(0):
        raise BookError(f"{what} must be timezone-aware UTC")
    return at.astimezone(UTC)


def utc_midnight(at: datetime) -> datetime:
    """00:00 UTC of ``at``'s UTC day."""
    return require_utc(at).replace(hour=0, minute=0, second=0, microsecond=0)


def is_model_initiated(decision_id: str | None, purpose: OrderPurpose | None) -> bool:
    """An order the model asked for: it answers a decision and no guard forced it."""
    return (
        decision_id is not None
        and purpose is not None
        and purpose is not OrderPurpose.PROTECTIVE_EXIT
    )


def exit_reason(
    *,
    decision_id: str | None,
    purpose: OrderPurpose | None,
    delegate_type: str | None,
    cause: Cause | None,
    flipped: bool,
) -> str:
    """How a trade ended, from what is known about its closing fill (module docstring)."""
    kind = (delegate_type or "").strip().lower() or None
    if kind in STOP_LOSS_DELEGATE_TYPES or cause is ProtectiveReason.STOP_FILLED:
        return EXIT_STOP
    if kind in TAKE_PROFIT_DELEGATE_TYPES:
        return EXIT_TAKE_PROFIT
    if kind in LIQUIDATION_DELEGATE_TYPES:
        return EXIT_LIQUIDATION
    if purpose is None:
        return EXIT_VENUE_ORDER_PREFIX + kind if kind is not None else EXIT_VENUE_INITIATED
    if flipped:
        return EXIT_FLIP
    if purpose is OrderPurpose.PROTECTIVE_EXIT or decision_id is None:
        return EXIT_PROTECTIVE if cause is None else f"{EXIT_PROTECTIVE}:{cause.value}"
    return EXIT_MODEL_CLOSE


# ------------------------------------------------------------------------------------------------
# The fold
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Record:
    """One applied fill with what the caller knows about it."""

    fill: Fill
    decision_id: str | None
    purpose: OrderPurpose | None
    delegate_type: str | None
    cause: Cause | None

    @property
    def key(self) -> tuple[datetime, int, int, str]:
        """Venue time; at one instant, closing fills first, then the venue's trade id (compared
        as a number when it is one: shorter numeric ids sort first)."""
        exec_id = self.fill.exec_id
        rank = 0 if self.fill.trade_side == "close" else 1
        return (self.fill.executed_at, rank, len(exec_id), exec_id)

    @property
    def order_key(self) -> str:
        oid = self.fill.client_oid
        return oid if oid else f"venue-order:{self.fill.venue_order_id}"


@dataclass(slots=True)
class _Trade:
    """The trade in progress on one symbol."""

    key: str
    direction: Literal[1, -1]
    qty: Decimal
    avg: Decimal
    opened_at: datetime
    last_increase_at: datetime
    realized: Decimal
    fees: Decimal
    entry_qty: Decimal
    entry_value: Decimal
    exit_qty: Decimal = _ZERO
    exit_value: Decimal = _ZERO
    max_abs_qty: Decimal = _ZERO
    decision_ids: list[str] = field(default_factory=list)
    last_decision_id: str | None = None

    def note(self, decision_id: str | None) -> None:
        if decision_id is None:
            return
        if decision_id not in self.decision_ids:
            self.decision_ids.append(decision_id)
        self.last_decision_id = decision_id


@dataclass(frozen=True, slots=True)
class _Closed:
    trade: ClosedTrade
    exec_id: str
    order: tuple[datetime, int, int, str]


@dataclass(slots=True)
class _SymbolBook:
    trade: _Trade | None = None
    closed: list[_Closed] = field(default_factory=list)
    realized: Decimal = _ZERO
    fees: Decimal = _ZERO


def _close(symbol: str, trade: _Trade, record: _Record, *, flipped: bool) -> _Closed:
    fill = record.fill
    return _Closed(
        trade=ClosedTrade(
            symbol=symbol,
            opened_at=trade.opened_at,
            closed_at=fill.executed_at,
            direction=trade.direction,
            entry_avg=trade.entry_value / trade.entry_qty,
            exit_avg=trade.exit_value / trade.exit_qty,
            max_abs_qty=trade.max_abs_qty,
            gross_pnl=trade.realized,
            fees=trade.fees,
            net_pnl=trade.realized - trade.fees,
            decision_ids=tuple(trade.decision_ids),
            exit_reason=exit_reason(
                decision_id=record.decision_id,
                purpose=record.purpose,
                delegate_type=record.delegate_type,
                cause=record.cause,
                flipped=flipped,
            ),
        ),
        exec_id=fill.exec_id,
        order=record.key,
    )


def _apply(book: _SymbolBook, symbol: str, record: _Record) -> None:
    fill = record.fill
    direction: Literal[1, -1] = 1 if fill.side is Side.BUY else -1
    qty = fill.exec_qty
    price = fill.exec_price
    fee = fill.fee_paid
    book.fees += fee
    remaining = qty
    fee_left = fee
    trade = book.trade

    if trade is not None and trade.direction != direction:
        closing = min(qty, abs(trade.qty))
        close_fee = fee if closing == qty else fee * closing / qty
        pnl = (price - trade.avg) * closing * trade.direction
        trade.realized += pnl
        book.realized += pnl
        trade.fees += close_fee
        trade.exit_qty += closing
        trade.exit_value += closing * price
        trade.qty -= closing * trade.direction
        trade.note(record.decision_id)
        remaining = qty - closing
        fee_left = fee - close_fee
        if trade.qty == 0:
            book.closed.append(_close(symbol, trade, record, flipped=remaining > 0))
            book.trade = None
            trade = None

    if remaining <= 0:
        return
    if trade is None:
        opened = _Trade(
            key=f"{symbol}:{fill.exec_id}:{'reopen' if remaining != qty else 'open'}",
            direction=direction,
            qty=remaining * direction,
            avg=price,
            opened_at=fill.executed_at,
            last_increase_at=fill.executed_at,
            realized=_ZERO,
            fees=fee_left,
            entry_qty=remaining,
            entry_value=remaining * price,
            max_abs_qty=remaining,
        )
        opened.note(record.decision_id)
        book.trade = opened
        return
    held = abs(trade.qty)
    trade.avg = (trade.avg * held + price * remaining) / (held + remaining)
    trade.qty += remaining * direction
    trade.last_increase_at = fill.executed_at
    trade.fees += fee_left
    trade.entry_qty += remaining
    trade.entry_value += remaining * price
    trade.max_abs_qty = max(trade.max_abs_qty, abs(trade.qty))
    trade.note(record.decision_id)


def _fold(symbol: str, records: Iterable[_Record]) -> _SymbolBook:
    book = _SymbolBook()
    with localcontext(DECIMAL_CONTEXT):
        for record in records:
            _apply(book, symbol, record)
    return book


# ------------------------------------------------------------------------------------------------
# The builder
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Stop:
    trade_key: str
    price: Decimal | None
    venue_id: str | None


_SETS_STOP: Final = frozenset({"preset", "placed", "replaced", "verified"})


def _validated_marks(marks: Mapping[str, Decimal], *, what: str) -> dict[str, Decimal]:
    out: dict[str, Decimal] = {}
    for symbol, mark in marks.items():
        if not isinstance(mark, Decimal) or not mark.is_finite() or mark <= 0:
            raise BookError(f"{what}[{symbol!r}] must be a positive, finite Decimal, got {mark!r}")
        out[symbol] = mark
    return out


class BookBuilder:
    """Positions, closed trades and book state from fills, stop syncs and hourly marks.

    Feed it in any order: fills are folded in venue time order (module docstring). Nothing here
    fetches or reads a clock; ``at`` is always the caller's.
    """

    def __init__(self, *, starting_equity: Decimal, policy: Policy) -> None:
        if (
            not isinstance(starting_equity, Decimal)
            or not starting_equity.is_finite()
            or starting_equity <= 0
        ):
            raise BookError(
                f"starting equity must be a positive, finite Decimal, got {starting_equity!r}"
            )
        self._start = starting_equity
        self._policy = policy
        self._records: dict[str, list[_Record]] = {}
        self._books: dict[str, _SymbolBook] = {}
        self._by_exec: dict[str, _Record] = {}
        self._stops: dict[str, _Stop] = {}
        self._marks: dict[datetime, MarkPoint] = {}
        self._venue: FillVenue | None = None
        self._last_fill_at: datetime | None = None

    # --- inputs -----------------------------------------------------------------------------

    def apply_fill(
        self,
        fill: Fill,
        *,
        decision_id: str | None,
        purpose: OrderPurpose | None,
        delegate_type: str | None = None,
        cause: Cause | None = None,
    ) -> list[ClosedTrade]:
        """Book one fill. Returns the trades this fill closed (after re-folding its symbol).

        ``decision_id`` and ``purpose`` come from the order intent the fill executed (``None`` for
        an order this agent did not plan, such as a venue stop). ``delegate_type`` is the closing
        order's Bitget ``delegateType`` when known; ``cause`` the protective reason or forcing
        guard. A fill already applied (same ``exec_id``) is ignored when identical and refused when
        it differs.
        """
        existing = self._by_exec.get(fill.exec_id)
        if existing is not None:
            if existing.fill != fill:
                raise BookError(f"fill {fill.exec_id} was applied before with different content")
            return []
        if fill.fee_coin.strip().upper() != QUOTE_COIN:
            raise BookError(
                f"fill {fill.exec_id} paid its fee in {fill.fee_coin!r}; only {QUOTE_COIN} fees "
                "can be booked"
            )
        if not fill.symbol:
            raise BookError(f"fill {fill.exec_id} has no symbol")
        if self._venue is not None and fill.venue is not self._venue:
            raise BookError(
                f"fill {fill.exec_id} is from {fill.venue.value}; this book holds "
                f"{self._venue.value} fills and the two never mix"
            )
        self._venue = fill.venue
        record = _Record(
            fill=fill,
            decision_id=decision_id,
            purpose=purpose,
            delegate_type=delegate_type,
            cause=cause,
        )
        records = self._records.setdefault(fill.symbol, [])
        bisect.insort(records, record, key=lambda r: r.key)
        self._by_exec[fill.exec_id] = record
        book = _fold(fill.symbol, records)
        self._books[fill.symbol] = book
        if self._last_fill_at is None or fill.executed_at > self._last_fill_at:
            self._last_fill_at = fill.executed_at
        return [c.trade for c in book.closed if c.exec_id == fill.exec_id]

    def apply_stop_sync(self, sync: StopSync) -> None:
        """Record what the venue's stop for ``sync.symbol`` now is (module docstring, Stops).

        A sync for a symbol with no open position changes nothing: there is no position for the
        stop to protect (the stop manager cancels such orphans)."""
        book = self._books.get(sync.symbol)
        trade = book.trade if book is not None else None
        if trade is None:
            return
        current = self._stops.get(sync.symbol)
        if current is not None and current.trade_key != trade.key:
            current = None
        if sync.action in _SETS_STOP:
            price = sync.stop_price
            if price is not None and (not price.is_finite() or price <= 0):
                raise BookError(f"stop price for {sync.symbol} must be positive, got {price}")
            if price is None and current is not None:
                price = current.price
            self._stops[sync.symbol] = _Stop(trade.key, price, sync.venue_id)
        elif sync.action == "cancelled":
            if current is not None and current.venue_id == sync.venue_id:
                del self._stops[sync.symbol]
        else:  # "missing": the venue holds no stop for the position
            self._stops.pop(sync.symbol, None)

    def record_mark(self, point: MarkPoint) -> None:
        """Remember an hourly mark (the ``MARK`` event) for day-open and peak equity. The first
        mark logged for an hour is the one kept."""
        self._marks.setdefault(point.at, point)

    # --- reads ------------------------------------------------------------------------------

    @property
    def starting_equity(self) -> Decimal:
        return self._start

    @property
    def policy(self) -> Policy:
        return self._policy

    @property
    def venue(self) -> FillVenue | None:
        """The venue of the fills booked so far; ``None`` before the first fill."""
        return self._venue

    @property
    def last_fill_at(self) -> datetime | None:
        return self._last_fill_at

    @property
    def fill_count(self) -> int:
        return len(self._by_exec)

    def require_as_of(self, at: datetime) -> datetime:
        """``at`` as UTC, refused when the book already holds fills too far after it."""
        at = require_utc(at, what="at")
        last = self._last_fill_at
        if last is not None and last > at + FILL_CLOCK_TOLERANCE:
            raise BookError(
                f"the book holds a fill executed at {last.isoformat()}, after {at.isoformat()}; "
                "a book as of an earlier time must be rebuilt from the fills up to it"
            )
        return at

    def positions(self) -> dict[str, Position]:
        """Every open position, with the stop that protects it."""
        out: dict[str, Position] = {}
        for symbol in sorted(self._books):
            trade = self._books[symbol].trade
            if trade is None:
                continue
            stop = self._stops.get(symbol)
            if stop is not None and stop.trade_key != trade.key:
                stop = None
            out[symbol] = Position(
                symbol=symbol,
                qty=trade.qty,
                avg_entry=trade.avg,
                opened_at=trade.opened_at,
                last_increase_at=trade.last_increase_at,
                realized_pnl=trade.realized,
                fees_paid=trade.fees,
                stop_price=stop.price if stop is not None else None,
                stop_venue_id=stop.venue_id if stop is not None else None,
                last_decision_id=trade.last_decision_id,
            )
        return out

    @property
    def realized_total(self) -> Decimal:
        with localcontext(DECIMAL_CONTEXT):
            return sum((b.realized for b in self._books.values()), _ZERO)

    @property
    def fees_total(self) -> Decimal:
        with localcontext(DECIMAL_CONTEXT):
            return sum((b.fees for b in self._books.values()), _ZERO)

    def realized_equity(self) -> Decimal:
        """Starting equity + realized - fees: the equity with every position valued at entry."""
        with localcontext(DECIMAL_CONTEXT):
            return self._start + self.realized_total - self.fees_total

    def unrealized(self, marks: Mapping[str, Decimal]) -> dict[str, Decimal]:
        """Unrealized P&L of each open position at ``marks``. Every open position needs a mark."""
        checked = _validated_marks(marks, what="marks")
        open_trades: dict[str, _Trade] = {}
        for symbol, book in sorted(self._books.items()):
            if book.trade is not None:
                open_trades[symbol] = book.trade
        missing = [s for s in open_trades if s not in checked]
        if missing:
            raise MissingMarkError(f"no mark for open positions {missing}")
        with localcontext(DECIMAL_CONTEXT):
            return {s: t.qty * (checked[s] - t.avg) for s, t in open_trades.items()}

    def equity(self, marks: Mapping[str, Decimal]) -> Decimal:
        """Starting equity + realized - fees + unrealized at ``marks`` (DESIGN.md §13)."""
        unrealized = self.unrealized(marks)
        with localcontext(DECIMAL_CONTEXT):
            return self.realized_equity() + sum(unrealized.values(), _ZERO)

    def closed_trades(self) -> tuple[ClosedTrade, ...]:
        """Every closed trade, in the order they closed on the venue."""
        closed = [c for b in self._books.values() for c in b.closed]
        closed.sort(key=lambda c: c.order)
        return tuple(c.trade for c in closed)

    def marks(self) -> tuple[MarkPoint, ...]:
        return tuple(self._marks[at] for at in sorted(self._marks))

    def state(
        self,
        *,
        at: datetime,
        marks: Mapping[str, Decimal],
        mark_source: PriceSource,
        activation: Activation,
    ) -> BookState:
        """The book at ``at``, valued at ``marks`` (every open position needs one)."""
        at = self.require_as_of(at)
        checked = _validated_marks(marks, what="marks")
        equity = self.equity(checked)
        day_start = utc_midnight(at)
        day_end = day_start + timedelta(days=1)
        with localcontext(DECIMAL_CONTEXT):
            fees_today = sum(
                (
                    r.fill.fee_paid
                    for records in self._records.values()
                    for r in records
                    if day_start <= r.fill.executed_at < day_end
                ),
                _ZERO,
            )
        return BookState(
            as_of=at,
            mark_source=mark_source,
            starting_equity=self._start,
            equity=equity,
            peak_equity=self._peak(at, mark_source, equity),
            day_open_equity=self._day_open(day_start, at, mark_source, equity),
            positions=self.positions(),
            marks=checked,
            fees_today=fees_today,
            fees_total=self.fees_total,
            realized_total=self.realized_total,
            rebalances_today=self._rebalances(day_start, day_end),
            consecutive_losses=self._consecutive_losses(),
            activation=activation,
            last_loss_at=self._last_loss_at(at),
        )

    # --- book-level folds -------------------------------------------------------------------

    @staticmethod
    def _mark_equity(point: MarkPoint, source: PriceSource) -> Decimal | None:
        return point.equity_book if source is PriceSource.DEMO else point.equity_live_mirror

    def _peak(self, at: datetime, source: PriceSource, equity: Decimal) -> Decimal:
        peak = max(self._start, equity)
        for point in self._marks.values():
            if point.at > at:
                continue
            value = self._mark_equity(point, source)
            if value is not None and value > peak:
                peak = value
        return peak

    def _day_open(
        self, day_start: datetime, at: datetime, source: PriceSource, equity: Decimal
    ) -> Decimal:
        midnight = self._marks.get(day_start)
        if midnight is not None:
            value = self._mark_equity(midnight, source)
            if value is not None:
                return value
        flat, realized_equity = self._realized_equity_before(day_start)
        if flat:
            return realized_equity
        for when in sorted(self._marks):
            if day_start < when <= at:
                value = self._mark_equity(self._marks[when], source)
                if value is not None:
                    return value
        return equity

    def _realized_equity_before(self, cutoff: datetime) -> tuple[bool, Decimal]:
        """Whether every position was flat just before ``cutoff``, and the realized equity then."""
        flat = True
        realized = _ZERO
        fees = _ZERO
        with localcontext(DECIMAL_CONTEXT):
            for symbol, records in self._records.items():
                prefix = [r for r in records if r.fill.executed_at < cutoff]
                if not prefix:
                    continue
                book = _fold(symbol, prefix)
                realized += book.realized
                fees += book.fees
                if book.trade is not None:
                    flat = False
            return flat, self._start + realized - fees

    def _rebalances(self, day_start: datetime, day_end: datetime) -> dict[str, int]:
        first: dict[str, _Record] = {}
        for records in self._records.values():
            for record in records:  # time order, so the first seen is the order's first fill
                first.setdefault(record.order_key, record)
        counts: dict[str, int] = {}
        for record in first.values():
            if not is_model_initiated(record.decision_id, record.purpose):
                continue
            if day_start <= record.fill.executed_at < day_end:
                counts[record.fill.symbol] = counts.get(record.fill.symbol, 0) + 1
        return dict(sorted(counts.items()))

    def _last_loss_at(self, at: datetime) -> datetime | None:
        losses = [t.closed_at for t in self.closed_trades() if t.net_pnl < 0 and t.closed_at <= at]
        return max(losses) if losses else None

    def _consecutive_losses(self) -> int:
        streak = 0
        for trade in self.closed_trades():
            streak = streak + 1 if trade.net_pnl < 0 else 0
        return streak


__all__ = [
    "DECIMAL_CONTEXT",
    "EXIT_FLIP",
    "EXIT_LIQUIDATION",
    "EXIT_MODEL_CLOSE",
    "EXIT_PROTECTIVE",
    "EXIT_STOP",
    "EXIT_TAKE_PROFIT",
    "EXIT_VENUE_INITIATED",
    "EXIT_VENUE_ORDER_PREFIX",
    "FILL_CLOCK_TOLERANCE",
    "LIQUIDATION_DELEGATE_TYPES",
    "QUOTE_COIN",
    "STOP_LOSS_DELEGATE_TYPES",
    "TAKE_PROFIT_DELEGATE_TYPES",
    "BookBuilder",
    "BookError",
    "Cause",
    "MissingMarkError",
    "exit_reason",
    "is_model_initiated",
    "require_utc",
    "utc_midnight",
]
