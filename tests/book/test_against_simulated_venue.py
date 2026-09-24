"""The book against an independent implementation of the same accounting: the simulated venue.

``execution/simulated.py`` keeps its own positions, average prices, realized P&L and fees, written
separately from ``book/book.py``. Driving it with random orders (adds, partial reduces, closes,
single-order flips and stop triggers on the Demo mark) and feeding its fills to the book must give
the same positions, the same realized equity and the same marked equity as the venue reports. Two
implementations agreeing on hundreds of random paths is evidence neither can give alone."""

import random
from collections.abc import Sequence
from datetime import datetime, timedelta
from decimal import Decimal

from book.factories import START, T0
from helpers import make_intent, make_quote, mint_for_test
from sentiment_agent.book.book import EXIT_STOP, BookBuilder
from sentiment_agent.clock import ManualClock
from sentiment_agent.execution.simulated import SimulatedVenue
from sentiment_agent.policy import POLICY_V1
from sentiment_agent.types import (
    Activation,
    Candle,
    CandleKind,
    FundingPoint,
    InstrumentSpec,
    OrderIntent,
    OrderPurpose,
    PriceSource,
    Quote,
    Side,
    VenueAck,
)

SYMBOLS = ("NVDAUSDT", "BTCUSDT")


class _Market:
    """Keyless Demo tickers, set by the test."""

    def __init__(self) -> None:
        self.demo: dict[str, Quote] = {}

    def set(self, symbol: str, mid: Decimal, at: datetime) -> None:
        half = (mid * Decimal("0.0001")).quantize(Decimal("0.01"))
        self.demo[symbol] = make_quote(
            symbol,
            last=str(mid),
            mark=str(mid),
            index=str(mid),
            bid=str(mid - half),
            ask=str(mid + half),
            at=at,
        )

    def instruments(self, source: PriceSource, symbols: Sequence[str]) -> dict[str, InstrumentSpec]:
        return {}

    def quotes(self, source: PriceSource, symbols: Sequence[str]) -> dict[str, Quote]:
        assert source is PriceSource.DEMO
        return {s: self.demo[s] for s in symbols if s in self.demo}

    def candles(
        self,
        source: PriceSource,
        symbol: str,
        *,
        kind: CandleKind,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> list[Candle]:
        return []

    def funding_history(self, symbol: str, *, limit: int) -> list[FundingPoint]:
        return []


def _intent(
    n: int, symbol: str, side: Side, qty: Decimal, price: Decimal, reduces: bool
) -> OrderIntent:
    purpose = OrderPurpose.REDUCE if reduces else OrderPurpose.OPEN
    return make_intent(
        symbol=symbol,
        side=side,
        qty=str(qty),
        purpose=purpose,
        ruling_id=f"ruling-{n}",
        price=str(price),
    )


def test_the_book_matches_the_simulated_venue_on_random_paths() -> None:
    rng = random.Random(20260924)  # noqa: S311 - a reproducible sweep, not a secret
    stops_seen = 0
    for path in range(60):
        clock = ManualClock(T0)
        market = _Market()
        prices = {"NVDAUSDT": Decimal("180"), "BTCUSDT": Decimal("60000")}
        for symbol, mid in prices.items():
            market.set(symbol, mid, clock.now())
        venue = SimulatedVenue(market=market, clock=clock, starting_equity=START)
        book = BookBuilder(starting_equity=START, policy=POLICY_V1)
        attribution: dict[str, OrderIntent] = {}
        seen: set[str] = set()
        held: dict[str, Decimal] = dict.fromkeys(SYMBOLS, Decimal(0))
        for step in range(rng.randint(5, 30)):
            clock.advance(timedelta(minutes=rng.randint(1, 45)))
            symbol = rng.choice(SYMBOLS)
            drift = Decimal(rng.randint(-600, 600)) / 10_000  # up to 6% a step: stops do fire
            prices[symbol] = (prices[symbol] * (1 + drift)).quantize(Decimal("0.01"))
            market.set(symbol, prices[symbol], clock.now())
            for fill in venue.poll():
                assert fill.client_oid is None  # a stop: the venue's own order
            open_now = {p.symbol: p.qty for p in venue.positions()}
            held[symbol] = open_now.get(symbol, Decimal(0))
            step_qty = Decimal("0.01") if symbol == "BTCUSDT" else Decimal("0.5")
            qty = step_qty * rng.randint(1, 6)
            side = rng.choice((Side.BUY, Side.SELL))
            signed = qty if side is Side.BUY else -qty
            reduces = held[symbol] != 0 and (held[symbol] > 0) != (signed > 0)
            reduce_only = reduces and qty <= abs(held[symbol]) and rng.random() < 0.7
            sent = _intent(path * 100 + step, symbol, side, qty, prices[symbol], reduce_only)
            outcome = venue.place(mint_for_test(sent))
            assert isinstance(outcome, VenueAck)
            attribution[sent.client_oid] = sent
        for fill in venue.fills(since=T0, until=clock.now()):
            assert fill.exec_id not in seen
            seen.add(fill.exec_id)
            intent = attribution.get(fill.client_oid) if fill.client_oid else None
            book.apply_fill(
                fill,
                decision_id="d" if intent is not None else None,
                purpose=intent.purpose if intent is not None else None,
                delegate_type=None if intent is not None else "position_stop_loss_market",
            )
        venue_positions = {p.symbol: p for p in venue.positions()}
        mine = book.positions()
        assert set(mine) == set(venue_positions), path
        for symbol, ours in mine.items():
            theirs = venue_positions[symbol]
            assert ours.qty == theirs.qty, (path, symbol)
            assert theirs.avg_price is not None
            assert abs(ours.avg_entry - theirs.avg_price) < Decimal("1e-18"), (path, symbol)
        assert abs(book.realized_equity() - venue.realized_equity) < Decimal("1e-15"), path
        assert book.fees_total == venue.fees_paid, path
        marks = {s: market.demo[s].mark for s in SYMBOLS}
        state = book.state(
            at=clock.now(), marks=marks, mark_source=PriceSource.DEMO, activation=Activation.ACTIVE
        )
        equity = venue.account().equity_usdt
        assert equity is not None
        assert abs(state.equity - equity) < Decimal("1e-15"), path
        stops_seen += sum(1 for t in book.closed_trades() if t.exit_reason == EXIT_STOP)
    assert stops_seen > 0  # the paths exercised venue stops, not only the agent's own orders
