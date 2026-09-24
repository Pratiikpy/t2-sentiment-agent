"""Builders for the analysis tests.

Instrument limits are the Demo rows of ``validation/demo_venue/universe_probe.json`` (keyless reads
of 2026-09-24 10:31 UTC), read from the file rather than retyped, so a test runs on the venue's
real grid, minimums and fee. Prices are round numbers chosen so expected results can be worked out
by hand.
"""

import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from functools import cache
from pathlib import Path

from sentiment_agent.analysis.armsim import ArmSimulator
from sentiment_agent.clock import ManualClock
from sentiment_agent.hashing import content_hash
from sentiment_agent.kernel.kernel import RiskKernel
from sentiment_agent.policy import POLICY_V1
from sentiment_agent.types import (
    Activation,
    BookState,
    Candle,
    CandleKind,
    Category,
    CrowdReport,
    DecisionRecord,
    GroundingFigure,
    GroundingReport,
    InstrumentSpec,
    KernelInputs,
    LlmCallRecord,
    LlmDecision,
    LlmOutcome,
    LlmUsage,
    MarketMood,
    PerceptionSnapshot,
    PositioningFeatures,
    PriceSource,
    Quote,
    RunMode,
    Stance,
    TargetProposal,
    Thinking,
)

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "validation" / "demo_venue" / "universe_probe.json"

WED = datetime(2026, 9, 23, 13, 0, tzinfo=UTC)
"""Wednesday 13:00 UTC: a weekday, every session open."""
FRI = datetime(2026, 9, 25, 13, 0, tzinfo=UTC)
"""Friday 13:00 UTC; the weekend freeze starts at 20:00 and ends Monday 00:00."""
HOUR = timedelta(hours=1)
EQUITY = 100_000.0

BTC = "BTCUSDT"
NVDA = "NVDAUSDT"
TSLA = "TSLAUSDT"
META = "METAUSDT"
AAPL = "AAPLUSDT"
AMZN = "AMZNUSDT"


@cache
def _probe() -> dict[str, dict[str, str]]:
    data = json.loads(PROBE.read_text(encoding="utf-8"))
    return {symbol: rows["demo"] for symbol, rows in data["instruments"].items()}


def spec(symbol: str, *, at: datetime = WED) -> InstrumentSpec:
    """The Demo instrument spec, from the recorded probe."""
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


def specs(symbols: Iterable[str]) -> dict[str, InstrumentSpec]:
    return {s: spec(s) for s in symbols}


def quote(
    symbol: str,
    price: str | Decimal,
    *,
    at: datetime,
    source: PriceSource = PriceSource.DEMO,
    spread_bps: str = "2",
) -> Quote:
    """A quote at one price (mark = index = last), with bid and ask ``spread_bps`` apart."""
    p = Decimal(price)
    half = p * Decimal(spread_bps) / Decimal(20_000)
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


def inputs(prices: Mapping[str, str | Decimal], *, at: datetime) -> KernelInputs:
    """Kernel inputs where every guard can pass: Demo and live agree, the Demo index moves."""
    return KernelInputs(
        at=at,
        demo_quotes={s: quote(s, p, at=at) for s, p in prices.items()},
        live_quotes={s: quote(s, p, at=at, source=PriceSource.LIVE) for s, p in prices.items()},
        specs=specs(prices),
        demo_index_move_bps_3h=dict.fromkeys(prices, 10.0),
        snapshot_id=None,
        snapshot_taken_at=at,
    )


def candles(
    symbol: str,
    start: datetime,
    closes: Sequence[str | Decimal],
    *,
    source: PriceSource = PriceSource.DEMO,
    kind: CandleKind = "mark",
    lows: Mapping[int, str] | None = None,
    highs: Mapping[int, str] | None = None,
    opens: Mapping[int, str] | None = None,
    first_open: str | Decimal | None = None,
    interval: str = "1H",
) -> list[Candle]:
    """Hourly candles from ``start``: candle ``i`` opens at the previous close (or ``first_open``,
    else its own close), closes at ``closes[i]``; lows, highs and opens can be overridden by
    index."""
    out: list[Candle] = []
    previous = Decimal(first_open) if first_open is not None else Decimal(closes[0])
    for i, close_text in enumerate(closes):
        close = Decimal(close_text)
        open_ = Decimal((opens or {}).get(i, previous))
        low = Decimal((lows or {}).get(i, min(open_, close)))
        high = Decimal((highs or {}).get(i, max(open_, close)))
        out.append(
            Candle(
                symbol=symbol,
                source=source,
                kind=kind,
                interval=interval,
                open_time=start + HOUR * i,
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=None,
            )
        )
        previous = close
    return out


def flat_candles(symbol: str, start: datetime, hours: int, price: str) -> list[Candle]:
    return candles(symbol, start, [price] * hours)


def kernel() -> RiskKernel:
    return RiskKernel(POLICY_V1, ManualClock(WED))


def simulator(
    marks: Mapping[str, Sequence[Candle]],
    *,
    spreads: Mapping[str, float] | None = None,
    equity: float = EQUITY,
    with_specs: bool = True,
    audit: bool = False,
) -> ArmSimulator:
    return ArmSimulator(
        kernel=kernel(),
        policy=POLICY_V1,
        demo_marks=marks,
        spreads_bps=dict(spreads) if spreads is not None else dict.fromkeys(marks, 2.0),
        starting_equity=equity,
        specs=specs(marks) if with_specs else None,
        audit_protective=audit,
    )


def features(
    symbol: str,
    *,
    funding_z: float | None = None,
    long_short: float | None = None,
    index_move: float | None = 10.0,
) -> PositioningFeatures:
    entry = POLICY_V1.entry(symbol)
    assert entry is not None
    return PositioningFeatures(
        symbol=symbol,
        asset_class=entry.asset_class,
        demo_last=None,
        live_last=None,
        funding_rate_live=None,
        funding_z_live=funding_z,
        open_interest_live=None,
        oi_change_1h_pct=None,
        oi_change_24h_pct=None,
        retail_long_short_ratio=long_short,
        top_trader_long_short_ratio=None,
        taker_buy_sell_ratio=None,
        price_change_24h_pct=None,
        ma20_distance_atr=None,
        social_mentions_24h=None,
        social_velocity_per_hour=None,
        coordinated_cluster=False,
        demo_mark_index_gap_bps=None,
        demo_live_gap_bps=None,
        demo_spread_bps=None,
        demo_index_move_bps_3h=index_move,
    )


def snapshot(
    prices: Mapping[str, str | Decimal],
    *,
    at: datetime,
    feats: Mapping[str, PositioningFeatures] | None = None,
    name: str | None = None,
) -> PerceptionSnapshot:
    feature_map = dict(feats) if feats is not None else {s: features(s) for s in prices}
    return PerceptionSnapshot(
        snapshot_id=name or f"snap-{at.isoformat()}",
        taken_at=at,
        mode=RunMode.SIMULATED,
        policy_version=POLICY_V1.version,
        universe=POLICY_V1.symbols,
        demo_quotes={s: quote(s, p, at=at) for s, p in prices.items()},
        live_quotes={s: quote(s, p, at=at, source=PriceSource.LIVE) for s, p in prices.items()},
        features=feature_map,
        mood=MarketMood(),
        crowd=CrowdReport(
            items=0, withheld=0, distinct_stories=0, duplication_ratio=0.0, clusters=(), mentions={}
        ),
        text=(),
        calendar=(),
        source_calls=(),
        facts={},
    )


def book(equity: str = "100000", *, at: datetime = WED) -> BookState:
    e = Decimal(equity)
    return BookState(
        as_of=at,
        mark_source=PriceSource.DEMO,
        starting_equity=e,
        equity=e,
        peak_equity=e,
        day_open_equity=e,
        positions={},
        marks={},
        fees_today=Decimal(0),
        fees_total=Decimal(0),
        realized_total=Decimal(0),
        rebalances_today={},
        consecutive_losses=0,
        activation=Activation.ACTIVE,
    )


def grounded() -> GroundingReport:
    return GroundingReport(
        figures=(
            GroundingFigure(
                raw="2%",
                value=2.0,
                unit="%",
                context="test",
                resolved=True,
                source="fact",
                known_value=2.0,
            ),
        )
    )


def ungrounded() -> GroundingReport:
    return GroundingReport(
        figures=(
            GroundingFigure(
                raw="37%",
                value=37.0,
                unit="%",
                context="test",
                resolved=False,
                source=None,
                known_value=None,
            ),
        )
    )


def decision(
    decision_id: str,
    targets: Mapping[str, float],
    *,
    at: datetime,
    snapshot_id: str,
    grounding: Mapping[str, GroundingReport] | None = None,
    outcome: LlmOutcome = LlmOutcome.DECIDED,
) -> DecisionRecord:
    """A decision record whose proposed weights are ``targets`` (already weights, not targets in
    [-1, 1]: the tests reason in weights)."""
    usage = LlmUsage(
        prompt_tokens=10, completion_tokens=10, reasoning_tokens=0, total_tokens=20, reported=True
    )
    call = LlmCallRecord(
        model="fake",
        thinking=Thinking.LOW,
        prompt_version="prompt-v1",
        prompt_hash=content_hash(decision_id),
        request_blob=None,
        response_blobs=(),
        attempts=1 if outcome is LlmOutcome.DECIDED else 3,
        usage=usage,
        latency_ms=1,
        outcome=outcome,
    )
    if outcome is not LlmOutcome.DECIDED:
        return DecisionRecord(
            decision_id=decision_id,
            decided_at=at,
            trigger_ids=(),
            snapshot_id=snapshot_id,
            book_before=book(at=at),
            mandate=POLICY_V1.mandate,
            policy_version=POLICY_V1.version,
            call=call,
            outcome=outcome,
            decision=None,
            grounding={},
            proposed_weights={},
        )
    proposals = tuple(
        TargetProposal(
            symbol=s,
            target=w / POLICY_V1.per_name_max,
            thesis="t",
            invalidation="i",
            horizon_hours=24,
            crowd_belief="c",
            our_view="v",
            confidence=0.5,
        )
        for s, w in targets.items()
    )
    stance = Stance.ACT if proposals else Stance.FLAT_WITH_REASONS
    return DecisionRecord(
        decision_id=decision_id,
        decided_at=at,
        trigger_ids=(),
        snapshot_id=snapshot_id,
        book_before=book(at=at),
        mandate=POLICY_V1.mandate,
        policy_version=POLICY_V1.version,
        call=call,
        outcome=outcome,
        decision=LlmDecision(
            stance=stance,
            targets=proposals,
            rejected_alternatives=(),
            mandate_response="m",
            flat_reasons=() if proposals else ("nothing to do",),
            summary="s",
        ),
        grounding=dict(grounding) if grounding is not None else {s: grounded() for s in targets},
        proposed_weights=dict(targets),
    )
