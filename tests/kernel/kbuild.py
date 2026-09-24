"""Builders for the kernel's tests.

Instrument limits and default quotes are the Demo values measured by keyless reads on 2026-09-24
10:31 UTC (``validation/demo_venue/universe_probe.json``): minimum quantity, quantity and price
precision, minimum notional, maximum market-order quantity and taker fee per symbol, and each
symbol's Demo mark, index, bid, ask and last with its live last beside it.
"""

from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta
from decimal import Decimal

from helpers import T0
from sentiment_agent.clock import ManualClock
from sentiment_agent.kernel.kernel import RiskKernel
from sentiment_agent.policy import POLICY_V1
from sentiment_agent.types import (
    Activation,
    BookState,
    BreakerState,
    Category,
    GroundingFigure,
    GroundingReport,
    InstrumentSpec,
    KernelInputs,
    Position,
    PriceSource,
    ProtectiveReason,
    Quote,
    RulingContext,
)

NVDA = "NVDAUSDT"
BTC = "BTCUSDT"
SPX = "SP500USDT"
MSTR = "MSTRUSDT"
TSLA = "TSLAUSDT"

# symbol -> (minQty, quantity step, price step, minAmount, maxMarketOrderQty or None)
LIMITS: dict[str, tuple[str, str, str, str, str | None]] = {
    "BTCUSDT": ("0.0001", "0.0001", "0.1", "5", None),
    "SP500USDT": ("0.0001", "0.0001", "0.1", "5", "5"),
    "NDX100USDT": ("0.0001", "0.0001", "0.1", "5", "5"),
    "MSTRUSDT": ("0.01", "0.01", "0.01", "5", "200"),
    "HOODUSDT": ("0.01", "0.01", "0.01", "5", "105"),
    "CRCLUSDT": ("0.01", "0.01", "0.01", "5", "50"),
    "SNDKUSDT": ("0.01", "0.01", "0.01", "5", "30"),
    "COINUSDT": ("0.01", "0.01", "0.01", "5", "25"),
    "TSLAUSDT": ("0.01", "0.01", "0.01", "5", "35"),
    "GOOGLUSDT": ("0.01", "0.01", "0.01", "5", "300"),
    "METAUSDT": ("0.01", "0.01", "0.01", "5", "200"),
    "NVDAUSDT": ("0.01", "0.01", "0.01", "5", "60"),
    "AMZNUSDT": ("0.01", "0.01", "0.01", "5", "200"),
    "AAPLUSDT": ("0.01", "0.01", "0.01", "5", "50"),
}

# symbol -> (Demo mark, Demo index, Demo bid, Demo ask, Demo last, live last)
PRICES: dict[str, tuple[str, str, str, str, str, str]] = {
    "BTCUSDT": ("83176.5", "83218.1489377636867091", "83168.5", "83169.7", "83176.5", "83176.9"),
    "SP500USDT": ("7659.5", "7659.4259478346672704", "7658.4", "7660.5", "7658.4", "7660.7"),
    "NDX100USDT": ("30157.4", "30156.731914514077554", "30151", "30161", "30165", "30159"),
    "MSTRUSDT": ("158.86", "158.8567615163751899", "158.83", "158.89", "158.89", "158.63"),
    "HOODUSDT": ("119.87", "119.8701760949054126", "119.84", "119.9", "119.86", "119.95"),
    "CRCLUSDT": ("90.75", "90.7482248864844345", "90.72", "90.78", "90.78", "90.92"),
    "SNDKUSDT": ("1765.71", "1765.7098283376509806", "1765.57", "1766.08", "1765.57", "1766.3"),
    "COINUSDT": ("194.8", "194.8010066958829788", "194.77", "194.82", "194.99", "194.91"),
    "TSLAUSDT": ("375.84", "375.8606881127750031", "375.73", "375.9", "375.73", "375.9"),
    "GOOGLUSDT": ("336.08", "336.0759329261913302", "335.99", "336.13", "336.13", "337.37"),
    "METAUSDT": ("725.83", "725.8402350359911272", "725.63", "725.86", "725.68", "726.3"),
    "NVDAUSDT": ("222.84", "222.8429058752641667", "222.78", "222.86", "222.78", "223"),
    "AMZNUSDT": ("247.59", "247.5931084450400511", "247.56", "247.64", "247.56", "247.71"),
    "AAPLUSDT": ("336.34", "336.3412595011005205", "336.31", "336.38", "336.22", "336.38"),
}

UNIVERSE: tuple[str, ...] = tuple(PRICES)
"""All 14 policy symbols, each with its measured Demo limits and prices."""


def spec(
    symbol: str,
    *,
    status: str = "online",
    min_qty: str | None = None,
    step: str | None = None,
    price_step: str | None = None,
    min_amount: str | None = None,
    max_market: str | None = "default",
    fee: str = "0.0006",
    at: datetime = T0,
) -> InstrumentSpec:
    d_min, d_step, d_px, d_amt, d_max = LIMITS.get(symbol, ("0.01", "0.01", "0.01", "5", None))
    maximum = d_max if max_market == "default" else max_market
    return InstrumentSpec(
        symbol=symbol,
        category=Category.USDT_FUTURES,
        source=PriceSource.DEMO,
        base_coin=symbol.removesuffix("USDT"),
        quote_coin="USDT",
        status=status,
        min_order_qty=Decimal(min_qty or d_min),
        qty_step=Decimal(step or d_step),
        price_step=Decimal(price_step or d_px),
        min_order_amount=Decimal(min_amount or d_amt),
        max_market_order_qty=None if maximum is None else Decimal(maximum),
        max_order_qty=None,
        taker_fee_rate=Decimal(fee),
        maker_fee_rate=Decimal("0.0002"),
        max_leverage=None,
        fund_interval_hours=None,
        fetched_at=at,
    )


def demo_quote(
    symbol: str,
    *,
    mark: str | None = None,
    index: str | None = None,
    bid: str | None = None,
    ask: str | None = None,
    last: str | None = None,
    at: datetime = T0,
    fetched_at: datetime | None = None,
) -> Quote:
    d_mark, d_index, d_bid, d_ask, d_last, _ = PRICES[symbol]
    return Quote(
        symbol=symbol,
        source=PriceSource.DEMO,
        ts=at,
        fetched_at=at if fetched_at is None else fetched_at,
        last=Decimal(last or d_last),
        mark=Decimal(mark or d_mark),
        index=Decimal(index or d_index),
        bid=Decimal(bid or d_bid),
        ask=Decimal(ask or d_ask),
        funding_rate=None,
        open_interest=None,
        turnover_24h=None,
        price_change_24h=None,
    )


def flat_quote(symbol: str, price: str, *, spread: str = "0", at: datetime = T0) -> Quote:
    """A Demo quote at one round price (mark = index = last), for exact arithmetic."""
    p, half = Decimal(price), Decimal(spread) / 2
    return Quote(
        symbol=symbol,
        source=PriceSource.DEMO,
        ts=at,
        fetched_at=at,
        last=p,
        mark=p,
        index=p,
        bid=p - half,
        ask=p + half,
        funding_rate=None,
        open_interest=None,
        turnover_24h=None,
        price_change_24h=None,
    )


def live_quote(symbol: str, *, last: str | None = None, at: datetime = T0) -> Quote:
    live_last = Decimal(last or PRICES[symbol][5])
    return Quote(
        symbol=symbol,
        source=PriceSource.LIVE,
        ts=at,
        fetched_at=at,
        last=live_last,
        mark=live_last,
        index=live_last,
        bid=live_last,
        ask=live_last,
        funding_rate=None,
        open_interest=None,
        turnover_24h=None,
        price_change_24h=None,
    )


def position(
    symbol: str,
    qty: str,
    *,
    entry: str | None = None,
    last_increase_at: datetime | None = None,
) -> Position:
    increased = T0 - timedelta(hours=30) if last_increase_at is None else last_increase_at
    return Position(
        symbol=symbol,
        qty=Decimal(qty),
        avg_entry=Decimal(entry or (PRICES[symbol][0] if symbol in PRICES else "100")),
        opened_at=min(increased, T0 - timedelta(hours=30)),
        last_increase_at=increased,
        realized_pnl=Decimal(0),
        fees_paid=Decimal(0),
        stop_price=None,
        stop_venue_id=None,
        last_decision_id="d0",
    )


def book(
    *,
    equity: str = "10000",
    positions: Iterable[Position] = (),
    marks: Mapping[str, str] | None = None,
    peak: str | None = None,
    day_open: str | None = None,
    starting: str | None = None,
    fees_today: str = "0",
    fees_total: str = "0",
    rebalances: Mapping[str, int] | None = None,
    losses: int = 0,
    at: datetime = T0,
) -> BookState:
    held = {p.symbol: p for p in positions}
    mark_map = (
        {s: Decimal(PRICES[s][0]) for s in held if s in PRICES}
        if marks is None
        else {s: Decimal(v) for s, v in marks.items()}
    )
    e = Decimal(equity)
    return BookState(
        as_of=at,
        mark_source=PriceSource.DEMO,
        starting_equity=Decimal(starting or equity),
        equity=e,
        peak_equity=Decimal(peak or equity),
        day_open_equity=Decimal(day_open or equity),
        positions=held,
        marks=mark_map,
        fees_today=Decimal(fees_today),
        fees_total=Decimal(fees_total),
        realized_total=Decimal(0),
        rebalances_today=dict(rebalances or {}),
        consecutive_losses=losses,
        activation=Activation.ACTIVE,
    )


def inputs(
    symbols: Iterable[str] = (BTC, NVDA, SPX, MSTR, TSLA),
    *,
    demo: Mapping[str, Quote] | None = None,
    live: Mapping[str, Quote] | None = None,
    specs: Mapping[str, InstrumentSpec] | None = None,
    index_moves: Mapping[str, float | None] | None = None,
    snapshot_at: datetime | None = T0,
    at: datetime = T0,
) -> KernelInputs:
    names = tuple(symbols)
    return KernelInputs(
        at=at,
        demo_quotes=dict(demo) if demo is not None else {s: demo_quote(s, at=at) for s in names},
        live_quotes=dict(live) if live is not None else {s: live_quote(s, at=at) for s in names},
        specs=dict(specs) if specs is not None else {s: spec(s, at=at) for s in names},
        demo_index_move_bps_3h=dict(index_moves)
        if index_moves is not None
        else dict.fromkeys(names, 30.0),
        snapshot_id="snap-1",
        snapshot_taken_at=snapshot_at,
    )


def breaker_state(
    activation: Activation = Activation.ACTIVE,
    trips: tuple[str, ...] = (),
    *,
    at: datetime = T0,
    halted_until: datetime | None = None,
) -> BreakerState:
    return BreakerState(activation=activation, since=at, trips=trips, halted_until=halted_until)


GROUNDED = GroundingReport(figures=())
UNGROUNDED = GroundingReport(
    figures=(
        GroundingFigure(
            raw="7.3%",
            value=7.3,
            unit="%",
            context="funding is 7.3% annualised",
            resolved=False,
            source=None,
            known_value=None,
        ),
    )
)


def decision(
    symbols: Iterable[str] = (BTC, NVDA, SPX, MSTR, TSLA),
    *,
    decision_id: str = "d1",
    grounding: Mapping[str, GroundingReport] | None = None,
    invalidation: Mapping[str, bool] | None = None,
) -> RulingContext:
    return RulingContext(
        decision_id=decision_id,
        protective_reason=None,
        grounding=dict(grounding) if grounding is not None else dict.fromkeys(symbols, GROUNDED),
        invalidation_fired=dict(invalidation or {}),
    )


def protective_context(reason: ProtectiveReason = ProtectiveReason.BREAKER) -> RulingContext:
    return RulingContext(decision_id=None, protective_reason=reason)


def kernel(at: datetime = T0) -> tuple[RiskKernel, ManualClock]:
    clock = ManualClock(at)
    return RiskKernel(POLICY_V1, clock), clock


def qty_for_weight(weight: str, symbol: str, equity: str = "10000") -> str:
    """The quantity whose weight at the default Demo mark is ``weight`` (not rounded)."""
    return str(Decimal(weight) * Decimal(equity) / Decimal(PRICES[symbol][0]))
