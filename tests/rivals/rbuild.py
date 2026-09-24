"""Builders for the rival tests.

Instrument limits are the Demo rows of ``validation/demo_venue/universe_probe.json`` (keyless reads
of 2026-09-24 10:31 UTC), read from the file, so the simulator sizes on the venue's real grid.
Prices and texts are chosen so expected results can be worked out by hand.
"""

import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from functools import cache
from pathlib import Path

from sentiment_agent.analysis.armsim import ArmSimulator
from sentiment_agent.clock import ManualClock
from sentiment_agent.kernel.kernel import RiskKernel
from sentiment_agent.policy import POLICY_V1
from sentiment_agent.types import (
    Activation,
    BookState,
    Candle,
    Category,
    Completion,
    CrowdReport,
    InstrumentSpec,
    LlmUsage,
    MarketMood,
    PerceptionSnapshot,
    Position,
    PositioningFeatures,
    PriceSource,
    Quote,
    RunMode,
    ScreenedItem,
    TextChannel,
    TextItem,
)

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "validation" / "demo_venue" / "universe_probe.json"

T0 = datetime(2026, 9, 23, 13, 0, tzinfo=UTC)
"""Wednesday 13:00 UTC: a weekday, every session open."""
HOUR = timedelta(hours=1)

BTC = "BTCUSDT"
NVDA = "NVDAUSDT"
TSLA = "TSLAUSDT"
META = "METAUSDT"
AAPL = "AAPLUSDT"
COIN = "COINUSDT"

PRICES: dict[str, str] = {BTC: "83000", NVDA: "200", TSLA: "400", META: "700", AAPL: "250"}


@cache
def _probe() -> dict[str, dict[str, str]]:
    data = json.loads(PROBE.read_text(encoding="utf-8"))
    return {symbol: rows["demo"] for symbol, rows in data["instruments"].items()}


def spec(symbol: str, *, at: datetime = T0) -> InstrumentSpec:
    row = _probe()[symbol]
    return InstrumentSpec(
        symbol=symbol,
        category=Category.USDT_FUTURES,
        source=PriceSource.DEMO,
        base_coin=symbol.removesuffix("USDT"),
        quote_coin="USDT",
        status=row["status"],
        min_order_qty=Decimal(row["minQty"]),
        qty_step=Decimal(1).scaleb(-int(row["qtyPrec"])),
        price_step=Decimal(1).scaleb(-int(row["pxPrec"])),
        min_order_amount=Decimal(row["minAmt"]),
        max_market_order_qty=Decimal(row["maxMkt"]) if row["maxMkt"] else None,
        max_order_qty=Decimal(row["maxOrd"]) if row["maxOrd"] else None,
        taker_fee_rate=Decimal(row["taker"]),
        maker_fee_rate=Decimal(row["maker"]),
        max_leverage=int(row["maxLev"]) if row["maxLev"] else None,
        fund_interval_hours=None,
        fetched_at=at,
    )


def quote(
    symbol: str, price: str | Decimal, *, at: datetime, source: PriceSource = PriceSource.DEMO
) -> Quote:
    p = Decimal(price)
    half = p * Decimal(2) / Decimal(20_000)
    return Quote(
        symbol=symbol,
        source=source,
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


def features(
    symbol: str,
    *,
    long_short: float | None = None,
    taker: float | None = None,
    change_24h: float | None = None,
    live_last: float | None = None,
) -> PositioningFeatures:
    entry = POLICY_V1.entry(symbol)
    assert entry is not None
    return PositioningFeatures(
        symbol=symbol,
        asset_class=entry.asset_class,
        demo_last=None,
        live_last=live_last,
        funding_rate_live=None,
        funding_z_live=None,
        open_interest_live=None,
        oi_change_1h_pct=None,
        oi_change_24h_pct=None,
        retail_long_short_ratio=long_short,
        top_trader_long_short_ratio=None,
        taker_buy_sell_ratio=taker,
        price_change_24h_pct=change_24h,
        ma20_distance_atr=None,
        social_mentions_24h=None,
        social_velocity_per_hour=None,
        coordinated_cluster=False,
        demo_mark_index_gap_bps=None,
        demo_live_gap_bps=None,
        demo_spread_bps=None,
        demo_index_move_bps_3h=10.0,
    )


_counter = iter(range(1_000_000))


def item(
    text: str,
    *,
    channel: TextChannel = "x",
    source: str = "acct",
    at: datetime = T0 - timedelta(minutes=30),
    symbols: Sequence[str] = (),
    item_id: str | None = None,
) -> TextItem:
    return TextItem(
        item_id=item_id or f"{channel}-{next(_counter)}",
        channel=channel,
        source=source,
        url=None,
        published_at=at,
        fetched_at=at,
        text=text,
        symbols=tuple(symbols),
    )


def screened(text_item: TextItem, *, withheld: bool = False) -> ScreenedItem:
    return ScreenedItem(
        item=text_item,
        detections=(),
        withheld=withheld,
        prompt_text="[withheld]" if withheld else text_item.text,
    )


def snapshot(
    texts: Iterable[TextItem | ScreenedItem] = (),
    *,
    at: datetime = T0,
    prices: Mapping[str, str] | None = None,
    fear_greed: int | None = None,
    feats: Mapping[str, PositioningFeatures] | None = None,
    name: str | None = None,
) -> PerceptionSnapshot:
    chosen = dict(prices) if prices is not None else dict(PRICES)
    feature_map = {s: features(s) for s in chosen}
    feature_map.update(feats or {})
    return PerceptionSnapshot(
        snapshot_id=name or f"snap-{at.isoformat()}",
        taken_at=at,
        mode=RunMode.SIMULATED,
        policy_version=POLICY_V1.version,
        universe=POLICY_V1.symbols,
        demo_quotes={s: quote(s, p, at=at) for s, p in chosen.items()},
        live_quotes={s: quote(s, p, at=at, source=PriceSource.LIVE) for s, p in chosen.items()},
        features=feature_map,
        mood=MarketMood(crypto_fear_greed=fear_greed),
        crowd=CrowdReport(
            items=0, withheld=0, distinct_stories=0, duplication_ratio=0.0, clusters=(), mentions={}
        ),
        text=tuple(t if isinstance(t, ScreenedItem) else screened(t) for t in texts),
        calendar=(),
        source_calls=(),
        facts={},
    )


def book(
    equity: str = "100000",
    *,
    at: datetime = T0,
    weights: Mapping[str, float] | None = None,
    prices: Mapping[str, str] | None = None,
    entries: Mapping[str, str] | None = None,
    rebalances: Mapping[str, int] | None = None,
) -> BookState:
    """A book holding ``weights`` of equity at ``prices`` (default :data:`PRICES`)."""
    e = Decimal(equity)
    marks = {s: Decimal((prices or PRICES)[s]) for s in (weights or {})}
    positions = {
        s: Position(
            symbol=s,
            qty=Decimal(repr(w)) * e / marks[s],
            avg_entry=Decimal((entries or {}).get(s, str(marks[s]))),
            opened_at=at - HOUR,
            last_increase_at=at - HOUR,
            realized_pnl=Decimal(0),
            fees_paid=Decimal(0),
            stop_price=None,
            stop_venue_id=None,
            last_decision_id=None,
        )
        for s, w in (weights or {}).items()
    }
    return BookState(
        as_of=at,
        mark_source=PriceSource.DEMO,
        starting_equity=e,
        equity=e,
        peak_equity=e,
        day_open_equity=e,
        positions=positions,
        marks=marks,
        fees_today=Decimal(0),
        fees_total=Decimal(0),
        realized_total=Decimal(0),
        rebalances_today=dict(rebalances or {}),
        consecutive_losses=0,
        activation=Activation.ACTIVE,
    )


def text_completion(content: str, *, finish_reason: str = "stop") -> Completion:
    """A finished free-text completion, one token per UTF-8 byte (synthetic usage)."""
    size = len(content.encode("utf-8"))
    return Completion(
        content=content,
        reasoning="",
        usage=LlmUsage(
            prompt_tokens=0,
            completion_tokens=size,
            reasoning_tokens=0,
            total_tokens=size,
            reported=True,
        ),
        finish_reason=finish_reason,
        raw_id=f"fake-text-{size}",
        latency_ms=0,
    )


def candles(symbol: str, start: datetime, closes: Sequence[str]) -> list[Candle]:
    """Hourly Demo mark candles from ``start``, each opening at the previous close."""
    out: list[Candle] = []
    previous = Decimal(closes[0])
    for i, text in enumerate(closes):
        close = Decimal(text)
        out.append(
            Candle(
                symbol=symbol,
                source=PriceSource.DEMO,
                kind="mark",
                interval="1H",
                open_time=start + HOUR * i,
                open=previous,
                high=max(previous, close),
                low=min(previous, close),
                close=close,
                volume=None,
            )
        )
        previous = close
    return out


def simulator(marks: Mapping[str, Sequence[Candle]], *, equity: float = 100_000.0) -> ArmSimulator:
    return ArmSimulator(
        kernel=RiskKernel(POLICY_V1, ManualClock(T0)),
        policy=POLICY_V1,
        demo_marks=marks,
        spreads_bps=dict.fromkeys(marks, 2.0),
        starting_equity=equity,
        specs={s: spec(s) for s in marks},
    )
