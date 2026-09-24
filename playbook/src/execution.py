"""Execution: the replica's own paper fills. Follow-trade is refused.

**Signal-only.** The runner places no orders (``runtime.is_signal_only()``), and the replica is its
own paper venue, as the primary's ``SimulatedVenue`` is: market orders fill in full at the data
layer's ask (buy) or bid (sell), the mark when the book is one-sided, with the measured taker fee.

**Follow-trade is refused.** ``getagent.trade`` routes orders to a subscriber's bound subaccount,
and whether that subaccount is paper is NOT VERIFIED: the package schema does not say so, and this
path has no Demo check, no 40099 check and no environment proof. It would also size against a
subscriber-set ``margin_budget`` rather than the subaccount's equity. The manifest declares
``follow_trade_supported: false``, :meth:`cycle.Run.run` refuses a follow-trade run before it reads
or writes anything, and :func:`execute_follow` raises, so no single change re-enables routing. The
read-only proxy below (:class:`ProxyVenue`) is what a future, proven follow-trade path would
reconcile against; nothing in this package can reach it while the refusal stands.

What the refused path used to do, for the record: each leg PRE-CHECK -> EXECUTE -> POST-CHECK
through ``trade.contract`` (``references/sdk/trade/patterns.md``, Pattern 4), with the preset stop
aligned by ``trade.helpers.resolve_contract_tpsl`` and the venue's stop moved after an increase.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from .book import TAKER_FEE, Book
from .kernel import Quote
from .planner import REDUCE_CLOSE, REDUCE_REOPEN, Leg, SymbolPlan

ZERO = Decimal(0)
MAX_ENVELOPE_CHARS = 1500


def _dec(value: object) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return result if result.is_finite() else None


def jsonable(value: object, depth: int = 0) -> object:
    """A JSON-safe rendering of an SDK envelope, bounded in depth and size."""
    if depth > 6:
        return repr(value)[:200]
    if value is None or isinstance(value, (bool, int, str)):
        return value if not isinstance(value, str) else value[:500]
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): jsonable(v, depth + 1) for k, v in list(value.items())[:50]}
    if isinstance(value, (list, tuple)):
        return [jsonable(v, depth + 1) for v in list(value)[:50]]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return jsonable(dump(), depth + 1)
        except Exception:  # an envelope that cannot dump itself is recorded by repr
            return repr(value)[:MAX_ENVELOPE_CHARS]
    fields = getattr(value, "raw", None)
    if isinstance(fields, (Mapping, list)):
        return {"raw": jsonable(fields, depth + 1), "repr": repr(value)[:300]}
    return repr(value)[:MAX_ENVELOPE_CHARS]


@dataclass
class LegRecord:
    seq: int
    leg: Leg
    status: str = "planned"
    """``filled``, ``partial``, ``skipped``, ``rejected``, ``failed`` or ``unchanged``."""
    venue: str = "virtual"
    qty_sent: str | None = None
    order_id: str | None = None
    stop_sent: str | None = None
    before_qty: str | None = None
    after_qty: str | None = None
    price_estimate: str | None = None
    fill: dict[str, object] | None = None
    envelopes: list[object] = field(default_factory=list)
    detail: str = ""

    def to_json(self) -> dict[str, object]:
        return {
            "seq": self.seq,
            "symbol": self.leg.symbol,
            "action": self.leg.action,
            "side": self.leg.side,
            "purpose": self.leg.purpose,
            "status": self.status,
            "venue": self.venue,
            "qty_planned": str(self.leg.qty),
            "qty_sent": self.qty_sent,
            "order_id": self.order_id,
            "stop_planned": None if self.leg.stop is None else str(self.leg.stop),
            "stop_sent": self.stop_sent,
            "before_qty": self.before_qty,
            "after_qty": self.after_qty,
            "price_estimate": self.price_estimate,
            "fill": self.fill,
            "envelopes": self.envelopes,
            "detail": self.detail,
        }


def _book_leg(
    book: Book,
    leg: Leg,
    *,
    qty: Decimal,
    price: Decimal,
    at: datetime,
    estimated: bool,
    stop: Decimal | None,
) -> dict[str, object]:
    fill = book.apply(
        symbol=leg.symbol,
        side=leg.side,
        qty=qty,
        price=price,
        fee=qty * price * TAKER_FEE,
        at=at,
        purpose=f"{leg.purpose}:{leg.action}",
        estimated=estimated,
        stop=stop,
        counts_as_increase=leg.action != REDUCE_REOPEN,
        continues_trade=leg.action == REDUCE_CLOSE,
    )
    return fill.to_json()


def _count(book: Book, plan: SymbolPlan, records: list[LegRecord]) -> None:
    """Model orders per name today: one per logical leg the model caused (a flip is two; a partial
    reduction, sent as a close and a re-open, is one)."""
    for record in records:
        if record.leg.purpose != "model" or record.status not in ("filled", "partial"):
            continue
        if record.leg.action == REDUCE_REOPEN:
            continue
        book.count_model_order(plan.symbol)


def _finish(book: Book, plan: SymbolPlan) -> None:
    """A partial reduction whose re-open did not happen leaves a flat record: close its trade."""
    book.finish_trade(plan.symbol)


# ------------------------------------------------------------------------------------------------
# Signal-only: the replica's own paper venue
# ------------------------------------------------------------------------------------------------


def execute_virtual(
    plan: SymbolPlan, *, book: Book, quote: Quote | None, at: datetime, first_seq: int
) -> list[LegRecord]:
    records: list[LegRecord] = []
    for index, leg in enumerate(plan.legs):
        record = LegRecord(seq=first_seq + index, leg=leg)
        touch = None
        if quote is not None:
            touch = quote.ask if leg.side == "buy" else quote.bid
        price = touch if touch is not None and touch > 0 else leg.reference_price
        if leg.action == REDUCE_REOPEN and book.positions.get(leg.symbol) is None:
            record.status = "skipped"
            record.detail = "the close before it did not fill"
            records.append(record)
            continue
        record.price_estimate = str(price)
        record.qty_sent = str(leg.qty)
        record.fill = _book_leg(
            book, leg, qty=leg.qty, price=price, at=at, estimated=False, stop=leg.stop
        )
        record.status = "filled"
        records.append(record)
    _count(book, plan, records)
    _finish(book, plan)
    return records


# ------------------------------------------------------------------------------------------------
# Follow-trade: refused (module docstring)
# ------------------------------------------------------------------------------------------------


class ProxyVenue:
    """``getagent.trade``'s reads behind a small surface, so the rest of the package never touches
    it and tests can pass a fake. It has no mutating method."""

    def __init__(self, trade: Any):
        self.trade = trade

    def all_positions(self, symbols: tuple[str, ...]) -> dict[str, tuple[Decimal, Decimal | None]]:
        """Every symbol's signed venue quantity and reported entry, from one position read."""
        result = self.trade.contract.current_position()
        out: dict[str, tuple[Decimal, Decimal | None]] = {}
        for symbol in symbols:
            qty, entry, _ = self._select(result, symbol)
            out[symbol] = (qty, entry)
        return out

    def positions(self, symbol: str) -> tuple[Decimal, Decimal | None, object]:
        """Signed venue quantity, the entry price if reported, and the raw selections."""
        result = self.trade.contract.current_position(symbol=symbol)
        return self._select(result, symbol)

    def _select(self, result: object, symbol: str) -> tuple[Decimal, Decimal | None, object]:
        trade = self.trade
        qty = ZERO
        entry: Decimal | None = None
        raws: dict[str, object] = {}
        for hold_side, sign in (("long", Decimal(1)), ("short", Decimal(-1))):
            selection = trade.helpers.find_contract_position(result, symbol, hold_side=hold_side)
            if selection is None:
                continue
            size = _dec(getattr(selection, "size", None)) or ZERO
            qty += sign * abs(size)
            raw = getattr(selection, "raw", None)
            raws[hold_side] = jsonable(raw)
            if isinstance(raw, Mapping):
                for key in (
                    "openPriceAvg",
                    "averageOpenPrice",
                    "openAvgPrice",
                    "avgOpenPrice",
                    "entryPrice",
                ):
                    found = _dec(raw.get(key))
                    if found is not None and found > 0:
                        entry = found
                        break
        return qty, entry, raws

    def price(self, symbol: str) -> Decimal | None:
        value = self.trade.helpers.contract_price(symbol)
        return _dec(value)

    def price_step(self, symbol: str) -> Decimal | None:
        rules = self.trade.helpers.contract_rules(symbol)
        step = _dec(getattr(rules, "price_step", None))
        return step if step is not None and step > 0 else None


class FollowTradeRefused(RuntimeError):
    """Follow-trade routing is refused in this package; see :data:`FOLLOW_TRADE_REFUSAL`."""


FOLLOW_TRADE_REFUSAL = (
    "follow-trade is refused: getagent.trade sends orders to a subscriber's bound subaccount, and "
    "nothing proves that subaccount is paper (no Demo check, no 40099 check, no environment "
    "proof), nor ties margin_budget to its read equity; orders from this agent go to Bitget's "
    "Demo environment only"
)


def execute_follow(*_: object, **__: object) -> list[LegRecord]:
    """Refused, always. Kept as the one named entry point so that a manifest change or a runtime
    that invokes the follow-trade callback anyway fails loudly instead of trading: routing orders
    through ``getagent.trade`` would need its own environment proof first, and a cap tying
    ``margin_budget`` to the subaccount's read equity (primary repository, DESIGN.md)."""
    raise FollowTradeRefused(FOLLOW_TRADE_REFUSAL)
