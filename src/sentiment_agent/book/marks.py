"""The hourly mark: the book on the UTC hour, the series every published Sharpe and drawdown uses.

One :class:`~sentiment_agent.types.MarkPoint` per UTC hour carries three equities (DESIGN.md §13):

* ``equity_book``: starting equity + realized - fees + unrealized at the **Demo mark**
  (``Quote.mark`` from keyless Demo tickers). The primary series: the venue the orders go to,
  marked the way the venue triggers our stops (G4 triggers on mark).
* ``equity_venue``: the venue's own account equity, passed in when it was read (PAPER mode). Not
  recomputed here; the gap to ``equity_book`` is a reconciliation finding, never corrected.
* ``equity_live_mirror``: the same positions and the same realized P&L, with the open positions
  marked at **live** prices (§14.4), so a reader can see the Demo result is not a sandbox artefact.
  ``None`` when any open position lacks a usable live quote: a mirror with a hole in it is not a
  mirror.

**On the hour, or not at all.** A point is stamped with the hour it belongs to (:func:`hour_floor`)
and refused when it is computed more than :data:`MARK_MAX_LATENESS` after that hour, or from a quote
fetched more than :data:`MARK_QUOTE_WINDOW` from it. A runtime that was down at 14:00 and wakes at
14:40 leaves the 14:00 point missing, which the metrics see as a gap; it never writes 14:40 prices
under a 14:00 label. The envelope the expected numbers come from marks on the same hourly grid
(``validation/demo_venue/envelope_clean.py:48-62``).

Weights are computed at the Demo mark against ``equity_book``, as ``BookState.weight`` does.
"""

from collections.abc import Mapping
from datetime import datetime, timedelta
from decimal import Decimal, localcontext
from typing import Final

from sentiment_agent.book.book import DECIMAL_CONTEXT, BookBuilder, BookError, require_utc
from sentiment_agent.types import MarkPoint, PositionMark, PriceSource, Quote

MARK_MAX_LATENESS: Final = timedelta(minutes=5)
"""Latest a point may be computed after its hour. The runtime ticks every 30 s, so a healthy run
marks within a minute of the hour; five minutes absorbs a slow tick without letting a later price
pass as the hour's."""

MARK_QUOTE_WINDOW: Final = timedelta(minutes=5)
"""A quote used for a point must have been fetched within this distance of the point's hour."""


class MarkError(BookError):
    """A mark that cannot be stamped honestly on its hour."""


def hour_floor(at: datetime) -> datetime:
    """The UTC hour ``at`` falls in (``13:59:59.9`` -> ``13:00:00``). Refuses naive or non-UTC."""
    return require_utc(at).replace(minute=0, second=0, microsecond=0)


def _checked_quotes(
    quotes: Mapping[str, Quote], source: PriceSource, what: str
) -> Mapping[str, Quote]:
    for symbol, quote in quotes.items():
        if quote.symbol != symbol:
            raise MarkError(f"{what}[{symbol!r}] holds a quote for {quote.symbol!r}")
        if quote.source is not source:
            raise MarkError(f"{what}[{symbol!r}] is a {quote.source.value} quote")
    return quotes


def _usable(quote: Quote | None, hour: datetime) -> Quote | None:
    """The quote, when it was fetched near the hour and carries a positive mark."""
    if quote is None or abs(quote.fetched_at - hour) > MARK_QUOTE_WINDOW:
        return None
    if not quote.mark.is_finite() or quote.mark <= 0:
        return None
    return quote


def mark_point(
    builder: BookBuilder,
    *,
    at: datetime,
    demo: Mapping[str, Quote],
    live: Mapping[str, Quote],
    venue_equity: Decimal | None,
) -> MarkPoint:
    """The book on the hour ``at`` falls in. Raises :class:`MarkError` when it cannot be stamped
    honestly: too late after the hour, or an open position without a fresh Demo mark.

    Pure: the point is returned, not recorded. The caller logs it as a ``MARK`` event, and the
    projection feeds the logged point back with :meth:`BookBuilder.record_mark`, so the live run and
    a replay of the log see the same marks.
    """
    at = builder.require_as_of(at)
    hour = hour_floor(at)
    if at - hour > MARK_MAX_LATENESS:
        raise MarkError(
            f"{at.isoformat()} is {at - hour} after {hour.isoformat()}; a mark is taken within "
            f"{MARK_MAX_LATENESS} of its hour or not at all"
        )
    demo = _checked_quotes(demo, PriceSource.DEMO, "demo")
    live = _checked_quotes(live, PriceSource.LIVE, "live")
    if venue_equity is not None and not venue_equity.is_finite():
        raise MarkError("venue equity must be finite")

    positions = builder.positions()
    demo_used: dict[str, Quote] = {}
    missing: list[str] = []
    for symbol in positions:
        quote = _usable(demo.get(symbol), hour)
        if quote is None:
            missing.append(symbol)
        else:
            demo_used[symbol] = quote
    if missing:
        raise MarkError(
            f"no Demo mark fetched within {MARK_QUOTE_WINDOW} of {hour.isoformat()} for open "
            f"positions {missing}"
        )
    live_used = {s: q for s in positions if (q := _usable(live.get(s), hour)) is not None}

    realized_equity = builder.realized_equity()
    with localcontext(DECIMAL_CONTEXT):
        marks: list[PositionMark] = []
        unrealized_demo_total = Decimal(0)
        unrealized_live_total = Decimal(0)
        gross = Decimal(0)
        net = Decimal(0)
        for symbol, position in positions.items():
            demo_quote = demo_used[symbol]
            live_quote = live_used.get(symbol)
            unrealized_demo = position.qty * (demo_quote.mark - position.avg_entry)
            unrealized_live = (
                position.qty * (live_quote.mark - position.avg_entry)
                if live_quote is not None
                else None
            )
            unrealized_demo_total += unrealized_demo
            if unrealized_live is not None:
                unrealized_live_total += unrealized_live
            exposure = position.qty * demo_quote.mark
            gross += abs(exposure)
            net += exposure
            marks.append(
                PositionMark(
                    symbol=symbol,
                    qty=position.qty,
                    demo_mark=demo_quote.mark,
                    live_mark=live_quote.mark if live_quote is not None else None,
                    demo_index=demo_quote.index,
                    unrealized_demo=unrealized_demo,
                    unrealized_live=unrealized_live,
                )
            )
        equity_book = realized_equity + unrealized_demo_total
        mirror = (
            realized_equity + unrealized_live_total if len(live_used) == len(positions) else None
        )
        gross_weight = float(gross / equity_book) if equity_book > 0 else 0.0
        net_weight = float(net / equity_book) if equity_book > 0 else 0.0
    return MarkPoint(
        at=hour,
        equity_book=equity_book,
        equity_venue=venue_equity,
        equity_live_mirror=mirror,
        gross_weight=gross_weight,
        net_weight=net_weight,
        positions=tuple(marks),
    )


__all__ = ["MARK_MAX_LATENESS", "MARK_QUOTE_WINDOW", "MarkError", "hour_floor", "mark_point"]
