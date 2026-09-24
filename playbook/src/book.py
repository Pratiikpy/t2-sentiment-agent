"""The replica's book and its persisted state.

The primary rebuilds its book from a hash-chained ledger of fills (``sentiment_agent.book``). The
Playbook has no ledger; the runner may hydrate and sync ``.state/`` between runs, and nothing else
survives (``references/sandbox-runtime.md``, State and Persistence). So the book lives in one JSON
file there, written after every fill so a run killed at the 180-second limit loses nothing it did.

Accounting follows the primary's definitions (``policy.METRICS``, ``book/book.py``):

* equity = starting equity + realized P&L - fees + unrealized P&L at the marks;
* the daily kill is measured from the equity at the first run of the UTC day, the drawdown from the
  highest equity seen;
* a closed trade is one symbol from flat to flat (a flip closes one and opens one); its net P&L
  includes both fees; the losing streak counts the most recent closed trades with net P&L < 0;
* ``rebalances_today`` counts the orders the model caused per name since 00:00 UTC (protective
  exits excluded).

Starting equity is the subscription's ``margin_budget``, the same denominator the platform uses for
the Playbook's return (``package-schema.md``). Fills are exact in signal-only mode (the replica is
its own paper venue) and estimated in follow-trade mode from the venue's position change and the
price read before the order (the trade proxy's fill fields are not documented); every estimate is
labelled as one, and the platform's own paper metrics are the authoritative figures there.

State status, per run: ``fresh`` (no file: a new book, or a runner that does not hydrate
``.state/``), ``loaded``, ``corrupt`` (unreadable, another version or another policy: the breaker
halts, as the primary's does on an unreadable state), and ``lost`` (follow-trade only: no file but
the venue holds positions, so the day-open equity, peak, fees and hold times are unknown; every
increase is refused as not evaluated until the next UTC day rebuilds the baselines).
"""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from . import policy_v1 as policy
from .kernel import BookView, PositionView
from .sessions import as_utc, iso_z, next_utc_midnight, parse_iso, utc_day

STATE_DIR = Path(".state")
STATE_FILE = "t2sa_replica_state.json"
STATE_VERSION = 1
ZERO = Decimal(0)
TAKER_FEE = Decimal(policy.MEASURED_TAKER_FEE)


def _dec(value: object, default: Decimal = ZERO) -> Decimal:
    if isinstance(value, bool) or value is None:
        return default
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return default
    return result if result.is_finite() else default


@dataclass
class Position:
    qty: Decimal
    avg_entry: Decimal
    opened_at: datetime
    last_increase_at: datetime | None
    stop: Decimal | None = None
    realized: Decimal = ZERO
    fees: Decimal = ZERO
    trade_net: Decimal = ZERO
    """Net P&L of the trade in progress (flat to flat), fees included."""

    def to_json(self) -> dict[str, object]:
        return {
            "qty": str(self.qty),
            "avg_entry": str(self.avg_entry),
            "opened_at": iso_z(self.opened_at),
            "last_increase_at": None
            if self.last_increase_at is None
            else iso_z(self.last_increase_at),
            "stop": None if self.stop is None else str(self.stop),
            "realized": str(self.realized),
            "fees": str(self.fees),
            "trade_net": str(self.trade_net),
        }

    @staticmethod
    def from_json(raw: Mapping[str, object]) -> "Position":
        opened = parse_iso(raw.get("opened_at"))
        if opened is None:
            raise ValueError("position without an opening time")
        stop = raw.get("stop")
        return Position(
            qty=_dec(raw.get("qty")),
            avg_entry=_dec(raw.get("avg_entry")),
            opened_at=opened,
            last_increase_at=parse_iso(raw.get("last_increase_at")),
            stop=None if stop is None else _dec(stop),
            realized=_dec(raw.get("realized")),
            fees=_dec(raw.get("fees")),
            trade_net=_dec(raw.get("trade_net")),
        )


@dataclass
class Fill:
    """One fill as the book records it."""

    symbol: str
    side: str
    qty: Decimal
    price: Decimal
    fee: Decimal
    at: datetime
    purpose: str
    estimated: bool
    realized: Decimal = ZERO
    closed_trade_net: Decimal | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "qty": str(self.qty),
            "price": str(self.price),
            "fee": str(self.fee),
            "at": iso_z(self.at),
            "purpose": self.purpose,
            "estimated": self.estimated,
            "realized": str(self.realized),
            "closed_trade_net": None
            if self.closed_trade_net is None
            else str(self.closed_trade_net),
        }


@dataclass
class Book:
    mode: str
    starting_equity: Decimal
    created_at: datetime
    positions: dict[str, Position] = field(default_factory=dict)
    realized_total: Decimal = ZERO
    fees_total: Decimal = ZERO
    fees_today: Decimal = ZERO
    day: str = ""
    day_open_equity: Decimal = ZERO
    peak_equity: Decimal = ZERO
    consecutive_losses: int = 0
    closed_trades: int = 0
    winning_trades: int = 0
    rebalances_today: dict[str, int] = field(default_factory=dict)
    history_lost_until: datetime | None = None
    window_started_at: datetime | None = None
    status: str = "fresh"
    marks: dict[str, Decimal] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    # --- persistence -----------------------------------------------------------------------------

    def to_json(self) -> dict[str, object]:
        return {
            "version": STATE_VERSION,
            "policy_hash": policy.POLICY_HASH,
            "mode": self.mode,
            "starting_equity": str(self.starting_equity),
            "created_at": iso_z(self.created_at),
            "positions": {s: p.to_json() for s, p in sorted(self.positions.items())},
            "realized_total": str(self.realized_total),
            "fees_total": str(self.fees_total),
            "fees_today": str(self.fees_today),
            "day": self.day,
            "day_open_equity": str(self.day_open_equity),
            "peak_equity": str(self.peak_equity),
            "consecutive_losses": self.consecutive_losses,
            "closed_trades": self.closed_trades,
            "winning_trades": self.winning_trades,
            "rebalances_today": dict(sorted(self.rebalances_today.items())),
            "history_lost_until": None
            if self.history_lost_until is None
            else iso_z(self.history_lost_until),
            "window_started_at": None
            if self.window_started_at is None
            else iso_z(self.window_started_at),
        }

    @staticmethod
    def from_json(raw: Mapping[str, object], *, mode: str) -> "Book":
        created = parse_iso(raw.get("created_at"))
        positions_raw = raw.get("positions")
        rebalances_raw = raw.get("rebalances_today")
        if (
            created is None
            or not isinstance(positions_raw, dict)
            or not isinstance(rebalances_raw, dict)
        ):
            raise ValueError("state is missing required fields")
        positions = {
            str(s): Position.from_json(p) for s, p in positions_raw.items() if isinstance(p, dict)
        }
        rebalances = {
            str(s): int(n)
            for s, n in rebalances_raw.items()
            if isinstance(n, int) and not isinstance(n, bool)
        }
        losses = raw.get("consecutive_losses")
        closed = raw.get("closed_trades")
        wins = raw.get("winning_trades")
        book = Book(
            mode=mode,
            starting_equity=_dec(raw.get("starting_equity")),
            created_at=created,
            positions=positions,
            realized_total=_dec(raw.get("realized_total")),
            fees_total=_dec(raw.get("fees_total")),
            fees_today=_dec(raw.get("fees_today")),
            day=str(raw.get("day") or ""),
            day_open_equity=_dec(raw.get("day_open_equity")),
            peak_equity=_dec(raw.get("peak_equity")),
            consecutive_losses=losses
            if isinstance(losses, int) and not isinstance(losses, bool)
            else 0,
            closed_trades=closed if isinstance(closed, int) and not isinstance(closed, bool) else 0,
            winning_trades=wins if isinstance(wins, int) and not isinstance(wins, bool) else 0,
            rebalances_today=rebalances,
            history_lost_until=parse_iso(raw.get("history_lost_until")),
            window_started_at=parse_iso(raw.get("window_started_at")),
            status="loaded",
        )
        # A flat position record is the first half of a partial reduction whose re-open never ran
        # (the run ended between the two): the trade it carried is closed now.
        for symbol in [s for s, p in book.positions.items() if p.qty == 0]:
            book.finish_trade(symbol)
        return book

    # --- valuation -------------------------------------------------------------------------------

    @property
    def history_known(self) -> bool:
        return self.history_lost_until is None

    def mark_price(self, symbol: str) -> Decimal | None:
        mark = self.marks.get(symbol)
        if mark is not None and mark > 0:
            return mark
        position = self.positions.get(symbol)
        if position is not None and position.avg_entry > 0:
            return position.avg_entry
        return None

    def unrealized(self, symbol: str) -> Decimal:
        position = self.positions.get(symbol)
        price = self.mark_price(symbol)
        if position is None or price is None:
            return ZERO
        return (price - position.avg_entry) * position.qty

    def equity(self) -> Decimal:
        unrealized = sum((self.unrealized(s) for s in self.positions), ZERO)
        return self.starting_equity + self.realized_total - self.fees_total + unrealized

    def weight(self, symbol: str) -> float | None:
        position = self.positions.get(symbol)
        price = self.mark_price(symbol)
        equity = self.equity()
        if position is None or price is None or equity <= 0:
            return None
        return float(position.qty * price / equity)

    def held(self) -> dict[str, int]:
        return {
            s: (1 if p.qty > 0 else -1) for s, p in sorted(self.positions.items()) if p.qty != 0
        }

    # --- the run's clock -------------------------------------------------------------------------

    def open_run(self, now: datetime, marks: Mapping[str, Decimal]) -> None:
        """Mark to the run's prices, roll the UTC day, and move the peak."""
        self.marks = {s: m for s, m in marks.items() if m > 0}
        today = utc_day(now)
        equity = self.equity()
        if self.window_started_at is None:
            self.window_started_at = as_utc(now)
        if self.day != today:
            self.day = today
            self.fees_today = ZERO
            self.rebalances_today = {}
            self.day_open_equity = equity
            if self.history_lost_until is not None and as_utc(now) >= self.history_lost_until:
                self.history_lost_until = None
                self.peak_equity = equity
                self.notes.append(
                    f"history rebuilt at {iso_z(now)}: day-open and peak restart from this equity"
                )
        if self.peak_equity <= 0 or equity > self.peak_equity:
            self.peak_equity = equity

    def lose_history(self, now: datetime, reason: str) -> None:
        """The persisted history is gone while positions are open (``lost``/``corrupt``): the
        baselines are unknown until the next UTC day, and the fee window restarts now."""
        self.history_lost_until = next_utc_midnight(now)
        self.window_started_at = as_utc(now)
        self.day = utc_day(now)
        self.notes.append(f"{reason}; history unknown until {iso_z(self.history_lost_until)}")

    # --- fills -----------------------------------------------------------------------------------

    def apply(
        self,
        *,
        symbol: str,
        side: str,
        qty: Decimal,
        price: Decimal,
        fee: Decimal,
        at: datetime,
        purpose: str,
        estimated: bool,
        stop: Decimal | None = None,
        counts_as_increase: bool = True,
        continues_trade: bool = False,
    ) -> Fill:
        """Apply one fill. ``continues_trade`` keeps the trade record open across a close that is
        the first half of a partial reduction (the replica reduces by closing and re-opening the
        remainder); ``counts_as_increase`` is false for that re-open, which adds nothing to the
        position the model held."""
        if qty <= 0 or price <= 0:
            raise ValueError(f"a fill needs a positive quantity and price, got {qty} at {price}")
        signed = qty if side == "buy" else -qty
        record = Fill(symbol, side, qty, price, fee, at, purpose, estimated)
        self.fees_total += fee
        self.fees_today += fee
        position = self.positions.get(symbol)
        if position is None or position.qty == 0:
            carried = position.trade_net if position is not None and continues_trade else ZERO
            self.positions[symbol] = Position(
                qty=signed,
                avg_entry=price,
                opened_at=position.opened_at if position is not None and continues_trade else at,
                last_increase_at=at
                if counts_as_increase
                else (position.last_increase_at if position else at),
                stop=stop,
                fees=fee,
                trade_net=carried - fee,
            )
            return record
        if (position.qty > 0) == (signed > 0):
            total = abs(position.qty) + qty
            position.avg_entry = (abs(position.qty) * position.avg_entry + qty * price) / total
            position.qty += signed
            position.fees += fee
            position.trade_net -= fee
            if counts_as_increase:
                position.last_increase_at = at
            if stop is not None:
                position.stop = stop
            return record
        closing = min(abs(position.qty), qty)
        direction = Decimal(1) if position.qty > 0 else Decimal(-1)
        pnl = (price - position.avg_entry) * closing * direction
        close_fee = fee if closing == qty else fee * closing / qty
        self.realized_total += pnl
        record.realized = pnl
        position.realized += pnl
        position.fees += close_fee
        position.trade_net += pnl - close_fee
        position.qty += signed if closing == qty else -position.qty
        remainder = qty - closing
        if position.qty == 0:
            if continues_trade:
                self.positions[symbol] = position
            else:
                record.closed_trade_net = position.trade_net
                self._close_trade(position.trade_net)
                del self.positions[symbol]
        if remainder > 0:
            open_fee = fee - close_fee
            self.positions[symbol] = Position(
                qty=remainder if side == "buy" else -remainder,
                avg_entry=price,
                opened_at=at,
                last_increase_at=at,
                stop=stop,
                fees=open_fee,
                trade_net=-open_fee,
            )
        return record

    def finish_trade(self, symbol: str) -> Decimal | None:
        """Close the trade record of a position left flat by a partial reduction whose re-open did
        not happen (skipped or refused): it is then a closed trade like any other."""
        position = self.positions.get(symbol)
        if position is None or position.qty != 0:
            return None
        del self.positions[symbol]
        self._close_trade(position.trade_net)
        return position.trade_net

    def _close_trade(self, net: Decimal) -> None:
        self.closed_trades += 1
        if net > 0:
            self.winning_trades += 1
        self.consecutive_losses = self.consecutive_losses + 1 if net < 0 else 0

    def count_model_order(self, symbol: str) -> None:
        self.rebalances_today[symbol] = self.rebalances_today.get(symbol, 0) + 1

    # --- the venue's view (follow-trade) ---------------------------------------------------------

    def reconcile(
        self,
        venue: Mapping[str, tuple[Decimal, Decimal | None]],
        *,
        now: datetime,
        symbols: tuple[str, ...],
    ) -> list[dict[str, object]]:
        """Adopt the venue's positions where they differ from the book's.

        ``venue`` maps symbol -> (signed quantity, entry price if the venue reported one). A
        position the book held that the venue closed, with a recorded stop, is booked as a stop
        fill at that stop price (the preset stop lives on the venue and fires without us); any
        other difference is adopted at the current mark and recorded as external."""
        changes: list[dict[str, object]] = []
        for symbol in symbols:
            venue_qty, venue_entry = venue.get(symbol, (ZERO, None))
            position = self.positions.get(symbol)
            book_qty = ZERO if position is None else position.qty
            if venue_qty == book_qty:
                if position is not None and venue_entry is not None and venue_entry > 0:
                    position.avg_entry = venue_entry
                continue
            delta = venue_qty - book_qty
            side = "buy" if delta > 0 else "sell"
            stop_fill = (
                position is not None
                and venue_qty == 0
                and position.stop is not None
                and position.stop > 0
            )
            price = position.stop if stop_fill and position is not None else self.mark_price(symbol)
            if price is None or price <= 0:
                price = venue_entry if venue_entry is not None and venue_entry > 0 else None
            if price is None:
                changes.append(
                    {
                        "symbol": symbol,
                        "kind": "unpriced",
                        "book_qty": str(book_qty),
                        "venue_qty": str(venue_qty),
                    }
                )
                continue
            fee = abs(delta) * price * TAKER_FEE
            fill = self.apply(
                symbol=symbol,
                side=side,
                qty=abs(delta),
                price=price,
                fee=fee,
                at=now,
                purpose="stop_filled" if stop_fill else "external",
                estimated=True,
            )
            adopted = self.positions.get(symbol)
            if adopted is not None and venue_entry is not None and venue_entry > 0:
                adopted.avg_entry = venue_entry
            changes.append(
                {
                    "symbol": symbol,
                    "kind": "stop_filled" if stop_fill else "external",
                    "book_qty": str(book_qty),
                    "venue_qty": str(venue_qty),
                    "fill": fill.to_json(),
                }
            )
        return changes

    # --- the kernel's view -----------------------------------------------------------------------

    def view(self) -> BookView:
        equity = self.equity()
        return BookView(
            equity=equity,
            starting_equity=self.starting_equity,
            peak_equity=self.peak_equity,
            day_open_equity=self.day_open_equity if self.history_known else ZERO,
            fees_today=self.fees_today,
            fees_total=self.fees_total,
            consecutive_losses=self.consecutive_losses,
            positions={
                s: PositionView(
                    qty=p.qty, avg_entry=p.avg_entry, last_increase_at=p.last_increase_at
                )
                for s, p in self.positions.items()
                if p.qty != 0
            },
            rebalances_today=dict(self.rebalances_today),
            history_known=self.history_known,
        )

    def summary(self) -> dict[str, object]:
        equity = self.equity()
        view = self.view()
        return {
            "mode": self.mode,
            "status": self.status,
            "equity": str(equity),
            "starting_equity": str(self.starting_equity),
            "return_pct": float((equity / self.starting_equity - 1) * 100)
            if self.starting_equity > 0
            else None,
            "peak_equity": str(self.peak_equity),
            "drawdown": view.drawdown,
            "day_open_equity": str(self.day_open_equity),
            "day_return": view.day_return,
            "realized_total": str(self.realized_total),
            "fees_total": str(self.fees_total),
            "fees_today": str(self.fees_today),
            "closed_trades": self.closed_trades,
            "winning_trades": self.winning_trades,
            "consecutive_losses": self.consecutive_losses,
            "history_known": self.history_known,
            "window_started_at": None
            if self.window_started_at is None
            else iso_z(self.window_started_at),
            "positions": {
                s: {
                    **p.to_json(),
                    "mark": None if self.mark_price(s) is None else str(self.mark_price(s)),
                    "weight": self.weight(s),
                }
                for s, p in sorted(self.positions.items())
            },
            "notes": list(self.notes),
        }


@dataclass
class LoadedState:
    book: Book
    raw: dict[str, object]
    existed: bool
    problem: str | None = None


def load_state(*, directory: Path, mode: str, margin_budget: Decimal, now: datetime) -> LoadedState:
    """Read ``.state/`` into a book. Never raises: an unreadable file is a ``corrupt`` state."""
    path = directory / STATE_FILE
    fresh = Book(mode=mode, starting_equity=margin_budget, created_at=as_utc(now), status="fresh")
    if not path.exists():
        return LoadedState(book=fresh, raw={}, existed=False)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        fresh.status = "corrupt"
        return LoadedState(book=fresh, raw={}, existed=True, problem=f"unreadable state: {exc}")
    if not isinstance(raw, dict):
        fresh.status = "corrupt"
        return LoadedState(book=fresh, raw={}, existed=True, problem="state is not a JSON object")
    if raw.get("version") != STATE_VERSION:
        fresh.status = "corrupt"
        return LoadedState(
            book=fresh, raw={}, existed=True, problem=f"state version {raw.get('version')!r}"
        )
    if raw.get("policy_hash") != policy.POLICY_HASH:
        fresh.status = "corrupt"
        return LoadedState(
            book=fresh, raw={}, existed=True, problem="state written under another policy"
        )
    book_raw = raw.get("book")
    try:
        book = Book.from_json(book_raw if isinstance(book_raw, dict) else {}, mode=mode)
    except (ValueError, TypeError, KeyError) as exc:
        fresh.status = "corrupt"
        return LoadedState(
            book=fresh, raw={}, existed=True, problem=f"state book unreadable: {exc}"
        )
    if book.starting_equity <= 0:
        fresh.status = "corrupt"
        return LoadedState(book=fresh, raw={}, existed=True, problem="state has no starting equity")
    if str(raw.get("mode")) != mode:
        fresh.notes.append(
            f"mode changed from {raw.get('mode')!r} to {mode!r}: "
            "the previous book is not this venue's"
        )
        fresh.status = "mode_changed"
        return LoadedState(book=fresh, raw=raw, existed=True, problem="mode changed")
    return LoadedState(book=book, raw=raw, existed=True)


def save_state(directory: Path, *, book: Book, extra: Mapping[str, object]) -> None:
    """Write the state atomically (a temporary file, then a rename)."""
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": STATE_VERSION,
        "policy_hash": policy.POLICY_HASH,
        "mode": book.mode,
        "book": book.to_json(),
    }
    payload.update({k: v for k, v in extra.items() if k not in payload})
    text = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    temporary = directory / (STATE_FILE + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(directory / STATE_FILE)


def finite(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None
