"""Feature math against hand-computed series, the live/Demo provenance rule, mood, and facts."""

import math
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from helpers import T0, empty_book, make_quote
from sentiment_agent.perception.features import (
    FUNDING_SD_FLOOR,
    OI_MATCH_TOLERANCE,
    OI_MAX_AGE,
    build_features,
    coverage_facts,
    crowd_facts,
    facts_from,
    fear_greed_band,
    funding_z,
    index_move_bps_3h,
    ma_distance_atr,
    mood_from,
    oi_change_pct,
    social_signals,
)
from sentiment_agent.policy import POLICY_V1
from sentiment_agent.types import (
    AssetClass,
    BookState,
    CalendarItem,
    Candle,
    CandleKind,
    CrowdReport,
    DerivativesReading,
    FundingPoint,
    MarketMood,
    MoodReading,
    Position,
    PositioningFeatures,
    PriceSource,
    Quote,
    SourceCall,
    SourceHealth,
    StoryCluster,
    ToolkitSurface,
)

HOUR = timedelta(hours=1)


def candles(
    closes: Sequence[float],
    *,
    symbol: str = "NVDAUSDT",
    source: PriceSource = PriceSource.LIVE,
    kind: CandleKind = "market",
    start: datetime = T0 - timedelta(hours=40),
    highs: Sequence[float] | None = None,
    lows: Sequence[float] | None = None,
    step: timedelta = HOUR,
) -> list[Candle]:
    out: list[Candle] = []
    for i, close in enumerate(closes):
        high = close if highs is None else highs[i]
        low = close if lows is None else lows[i]
        out.append(
            Candle(
                symbol=symbol,
                source=source,
                kind=kind,
                interval="1H",
                open_time=start + i * step,
                open=Decimal(str(close)),
                high=Decimal(str(high)),
                low=Decimal(str(low)),
                close=Decimal(str(close)),
                volume=None,
            )
        )
    return out


def funding_points(rates: Sequence[float], *, symbol: str = "BTCUSDT") -> list[FundingPoint]:
    start = T0 - timedelta(hours=8 * len(rates))
    return [
        FundingPoint(
            symbol=symbol,
            source=PriceSource.LIVE,
            ts=start + i * timedelta(hours=8),
            rate=Decimal(str(r)),
        )
        for i, r in enumerate(rates)
    ]


UNMEASURED = CrowdReport(
    items=0, withheld=0, distinct_stories=0, duplication_ratio=0.0, clusters=(), mentions={}
)


# --- funding z ---------------------------------------------------------------------------------


def test_funding_z_matches_a_hand_computed_value() -> None:
    history = funding_points([0.0001, 0.0002, 0.0003, 0.0004, 0.0005])
    # mean 0.0003, population sd sqrt(2) * 0.0001; (0.0006 - 0.0003) / (sqrt(2) * 0.0001)
    z = funding_z(history, 0.0006, 5)
    assert z == pytest.approx(3 / math.sqrt(2), rel=1e-12)


def test_funding_z_uses_only_the_last_lookback_settlements_in_time_order() -> None:
    history = funding_points([0.05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005])
    shuffled = [history[i] for i in (3, 0, 5, 1, 4, 2)]
    # The 0.05 outlier is the oldest point and falls outside a lookback of 5.
    assert funding_z(shuffled, 0.0006, 5) == pytest.approx(3 / math.sqrt(2), rel=1e-12)


def test_funding_z_needs_the_full_lookback() -> None:
    assert funding_z(funding_points([0.0001, 0.0002, 0.0003]), 0.0004, 5) is None


def test_funding_z_refuses_a_spread_below_one_quote_tick() -> None:
    flat = funding_points([0.0] * 90)
    assert funding_z(flat, 0.000064, 90) is None
    nearly_flat = funding_points([0.0] * 89 + [0.000001])
    sd = math.sqrt((1e-6) ** 2 / 90 * (1 - 1 / 90))
    assert sd < FUNDING_SD_FLOOR
    assert funding_z(nearly_flat, 0.000064, 90) is None


def test_funding_z_non_finite_current_is_unmeasured() -> None:
    history = funding_points([0.0001, 0.0002, 0.0003, 0.0004, 0.0005])
    assert funding_z(history, math.nan, 5) is None
    assert funding_z(history, math.inf, 5) is None


def test_funding_z_refuses_a_mixed_series_and_a_tiny_lookback() -> None:
    mixed = funding_points([0.0001, 0.0002]) + funding_points([0.0003], symbol="NVDAUSDT")
    with pytest.raises(ValueError, match="mixes"):
        funding_z(mixed, 0.0001, 2)
    with pytest.raises(ValueError, match="lookback"):
        funding_z(funding_points([0.0001]), 0.0001, 1)


# --- MA / ATR distance -------------------------------------------------------------------------


def test_ma_distance_atr_on_a_steady_trend() -> None:
    closes = [101.0 + i for i in range(20)]  # 101 .. 120
    bars = candles(closes, highs=[c + 1 for c in closes], lows=[c - 1 for c in closes])
    # SMA20 = 110.5; every TR = max(2, |c+1-(c-1)|, |c-1-(c-1)|) = 2; (120 - 110.5) / 2 = 4.75
    assert ma_distance_atr(bars) == pytest.approx(4.75, rel=1e-12)


def test_ma_distance_atr_counts_the_gap_in_the_true_range() -> None:
    closes = [100.0] * 19 + [109.0]
    highs = [100.0] * 19 + [110.0]
    lows = [100.0] * 19 + [108.0]
    bars = candles(closes, highs=highs, lows=lows)
    # 13 flat TRs of 0, then max(110-108, |110-100|, |108-100|) = 10: ATR = 10 / 14.
    # SMA20 = (19 * 100 + 109) / 20 = 100.45; (109 - 100.45) / (10 / 14) = 11.97
    assert ma_distance_atr(bars) == pytest.approx((109 - 100.45) / (10 / 14), rel=1e-12)


def test_ma_distance_atr_is_independent_of_how_much_history_was_fetched() -> None:
    closes = [100.0 + 3 * math.sin(i / 3) for i in range(60)]
    bars = candles(closes, highs=[c + 0.5 for c in closes], lows=[c - 0.7 for c in closes])
    assert ma_distance_atr(bars) == pytest.approx(ma_distance_atr(bars[-20:]), rel=1e-12)


def test_ma_distance_atr_unmeasured_cases() -> None:
    assert ma_distance_atr(candles([100.0] * 19)) is None  # too few bars
    assert ma_distance_atr(candles([100.0] * 20)) is None  # zero ATR
    assert ma_distance_atr(candles([100.0] * 19 + [0.0])) is None  # non-positive close


def test_ma_distance_atr_sorts_and_refuses_mixed_series() -> None:
    closes = [101.0 + i for i in range(20)]
    bars = candles(closes, highs=[c + 1 for c in closes], lows=[c - 1 for c in closes])
    assert ma_distance_atr(list(reversed(bars))) == pytest.approx(4.75, rel=1e-12)
    mixed = bars[:-1] + candles([120.0], source=PriceSource.DEMO, start=bars[-1].open_time)
    with pytest.raises(ValueError, match="mixes"):
        ma_distance_atr(mixed)


# --- stale-index detector ----------------------------------------------------------------------


def index(closes: Sequence[float], *, start: datetime = T0 - 4 * HOUR) -> list[Candle]:
    return candles(closes, source=PriceSource.DEMO, kind="index", start=start)


def test_stale_index_detector_on_a_moving_index() -> None:
    value = index_move_bps_3h(index([100.0, 100.5, 100.2, 100.8]))
    expected = (0.5 / 100 + abs(100.2 / 100.5 - 1) + abs(100.8 / 100.2 - 1)) / 3 * 10_000
    assert value == pytest.approx(expected, rel=1e-12)
    assert value is not None
    assert value > POLICY_V1.stale_index_min_move_bps_3h


def test_stale_index_detector_on_a_frozen_index() -> None:
    value = index_move_bps_3h(index([7659.43, 7659.43, 7659.44, 7659.43]))
    assert value is not None
    assert value < POLICY_V1.stale_index_min_move_bps_3h
    assert index_move_bps_3h(index([100.0] * 4)) == 0.0


def test_stale_index_detector_reads_only_the_newest_four_bars() -> None:
    bars = index([50.0, 200.0, 100.0, 100.0, 100.0, 100.0], start=T0 - 6 * HOUR)
    assert index_move_bps_3h(bars) == 0.0


def test_stale_index_detector_refuses_a_gap_and_a_short_series() -> None:
    gapped = index([100.0, 100.5]) + index([100.2, 100.8], start=T0 - HOUR)
    assert index_move_bps_3h(gapped) is None
    assert index_move_bps_3h(index([100.0, 100.5, 100.2])) is None


def test_stale_index_detector_reads_index_candles_only() -> None:
    with pytest.raises(ValueError, match="index candles"):
        index_move_bps_3h(candles([100.0] * 4, source=PriceSource.DEMO, kind="mark"))


# --- open interest -----------------------------------------------------------------------------


def oi_series(
    values: Sequence[float], *, end: datetime = T0 - HOUR
) -> list[tuple[datetime, float]]:
    return [(end - (len(values) - 1 - i) * HOUR, v) for i, v in enumerate(values)]


def test_oi_change_one_hour_and_one_day() -> None:
    series = oi_series([1000.0 + 10 * i for i in range(25)])  # 1000 .. 1240
    assert oi_change_pct(series, hours=1, at=T0) == pytest.approx((1240 / 1230 - 1) * 100)
    assert oi_change_pct(series, hours=24, at=T0) == pytest.approx(24.0)


def test_oi_change_ignores_points_after_the_observation_time() -> None:
    series = [*oi_series([1000.0, 1010.0]), (T0 + HOUR, 5000.0)]
    assert oi_change_pct(series, hours=1, at=T0) == pytest.approx(1.0)


def test_oi_change_refuses_a_stale_series() -> None:
    series = oi_series([1000.0, 1010.0], end=T0 - OI_MAX_AGE - timedelta(minutes=1))
    assert oi_change_pct(series, hours=1, at=T0) is None


def test_oi_change_needs_a_reference_within_the_pairing_tolerance() -> None:
    newest = T0 - HOUR
    inside = [(newest - HOUR - OI_MATCH_TOLERANCE, 1000.0), (newest, 1010.0)]
    outside = [
        (newest - HOUR - OI_MATCH_TOLERANCE - timedelta(minutes=1), 1000.0),
        (newest, 1010.0),
    ]
    assert oi_change_pct(inside, hours=1, at=T0) == pytest.approx(1.0)
    assert oi_change_pct(outside, hours=1, at=T0) is None
    assert oi_change_pct(oi_series([1000.0] * 10), hours=24, at=T0) is None


def test_oi_change_drops_unusable_points() -> None:
    series = [*oi_series([0.0, 1010.0]), (T0 - timedelta(minutes=30), math.nan)]
    assert oi_change_pct(series, hours=1, at=T0) is None
    with pytest.raises(ValueError, match="UTC"):
        oi_change_pct(series, hours=1, at=datetime(2026, 9, 23, 13, 0))  # noqa: DTZ001


def test_oi_change_agrees_with_the_trigger_thresholds_statistic() -> None:
    triggers = pytest.importorskip("sentiment_agent.events.triggers", exc_type=ImportError)
    series = oi_series([1000.0, 1013.0, 990.0, 1050.5, 1049.0])
    ours = oi_change_pct(series, hours=1, at=T0)
    theirs = triggers.hourly_oi_changes_pct(series)
    assert ours == pytest.approx(theirs[-1], rel=1e-12)
    assert triggers.PAIR_TOLERANCE == OI_MATCH_TOLERANCE


# --- mood --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "band"),
    [
        (0, "extreme_fear"),
        (25, "extreme_fear"),
        (26, "fear"),
        (45, "fear"),
        (46, "neutral"),
        (55, "neutral"),
        (56, "greed"),
        (75, "greed"),
        (76, "extreme_greed"),
        (100, "extreme_greed"),
    ],
)
def test_fear_greed_bands_are_bitgets(value: int, band: str) -> None:
    assert fear_greed_band(value, POLICY_V1) == band


def test_mood_keeps_source_labels_and_checks_agreement() -> None:
    mood = mood_from(
        MoodReading(
            crypto_fear_greed=24,
            crypto_fear_greed_label="Extreme Fear",
            crypto_fear_greed_alt=30,
            market_fear_greed=60,
        ),
        POLICY_V1,
    )
    assert mood == MarketMood(
        crypto_fear_greed=24,
        crypto_fear_greed_label="Extreme Fear",
        market_fear_greed=60,
        market_fear_greed_label="Greed",
        crypto_sources_agree=False,
    )
    agree = mood_from(MoodReading(crypto_fear_greed=40, crypto_fear_greed_alt=30), POLICY_V1)
    assert agree.crypto_sources_agree is True
    assert agree.crypto_fear_greed_label == "Fear"


def test_mood_falls_back_to_the_second_crypto_source() -> None:
    mood = mood_from(MoodReading(crypto_fear_greed_alt=80), POLICY_V1)
    assert mood.crypto_fear_greed == 80
    assert mood.crypto_fear_greed_label == "Extreme Greed"
    assert mood.crypto_sources_agree is None


def test_mood_with_nothing_measured_is_all_none() -> None:
    assert mood_from(MoodReading(), POLICY_V1) == MarketMood()


# --- build_features: provenance, units, social, calendar ---------------------------------------


def social_report(symbol: str = "NVDAUSDT", *, coordinated: bool = True) -> CrowdReport:
    first = T0 - timedelta(hours=3)
    clusters = (
        StoryCluster(
            cluster_id="story-a",
            representative="<<UNTRUSTED NVDA squeeze UNTRUSTED>>",
            item_ids=("x:1", "x:2", "x:3"),
            sources=("a", "b", "c"),
            symbols=(symbol,),
            first_seen=first,
            last_seen=first + timedelta(minutes=30),
            distinct_sources=3,
            coordinated=coordinated,
            velocity_per_hour=6.0,
        ),
        StoryCluster(
            cluster_id="story-b",
            representative="<<UNTRUSTED earnings next week UNTRUSTED>>",
            item_ids=("reddit:9",),
            sources=("r/stocks",),
            symbols=(symbol,),
            first_seen=first + timedelta(hours=2),
            last_seen=first + timedelta(hours=2),
            distinct_sources=1,
            coordinated=False,
            velocity_per_hour=None,
        ),
    )
    mentions = dict.fromkeys(POLICY_V1.symbols, 0)
    mentions[symbol] = 2
    return CrowdReport(
        items=4,
        withheld=0,
        distinct_stories=2,
        duplication_ratio=2.0,
        clusters=clusters,
        mentions=mentions,
    )


def nvda_inputs() -> tuple[Quote, Quote]:
    demo = make_quote("NVDAUSDT", source=PriceSource.DEMO).model_copy(
        update={"funding_rate": Decimal("-0.001"), "open_interest": Decimal("999")}
    )
    live = make_quote(
        "NVDAUSDT",
        source=PriceSource.LIVE,
        last="223.00",
        mark="223.01",
        index="223.02",
        bid="222.99",
        ask="223.01",
    ).model_copy(
        update={
            "funding_rate": Decimal("0.000064"),
            "open_interest": Decimal("51234.5"),
            "price_change_24h": Decimal("0.0213"),
        }
    )
    return demo, live


def nvda_features(**overrides: object) -> PositioningFeatures:
    demo, live = nvda_inputs()
    kwargs: dict[str, object] = {
        "symbol": "NVDAUSDT",
        "asset_class": AssetClass.US_EQUITY,
        "demo": demo,
        "live": live,
        "derivatives": None,
        "funding": [],
        "live_1h": [],
        "demo_index_1h": [],
        "crowd": UNMEASURED,
        "calendar": [],
        "policy": POLICY_V1,
    }
    kwargs.update(overrides)
    return build_features(**kwargs)  # type: ignore[arg-type]


def test_crowd_positioning_is_read_from_live_and_the_venue_from_demo() -> None:
    demo, live = nvda_inputs()
    f = nvda_features()
    # Live: the crowd.
    assert f.live_last == 223.0
    assert f.funding_rate_live == pytest.approx(0.000064)
    assert f.open_interest_live == pytest.approx(51234.5)
    assert f.price_change_24h_pct == pytest.approx(2.13)
    # Demo: the venue. Demo's sandbox funding (-0.1%) and open interest never reach a feature.
    assert f.demo_last == pytest.approx(222.81)
    assert f.demo_mark_index_gap_bps == pytest.approx(abs(222.84 / 222.8431 - 1) * 10_000)
    assert f.demo_spread_bps == pytest.approx(demo.spread_bps)
    assert f.demo_live_gap_bps == pytest.approx(abs(222.81 / 223.0 - 1) * 10_000)
    values = {v for _, v in f if isinstance(v, float)}
    assert -0.001 not in values
    assert 999.0 not in values
    # Swapping the arguments is refused rather than silently mixed.
    with pytest.raises(ValueError, match="demo quote"):
        nvda_features(demo=live, live=demo)
    with pytest.raises(ValueError, match="passed for"):
        nvda_features(live=make_quote("TSLAUSDT", source=PriceSource.LIVE))


def test_unmeasured_inputs_give_none_never_zero() -> None:
    f = nvda_features(demo=None, live=None)
    for name, value in f:
        if name in ("symbol", "asset_class", "coordinated_cluster"):
            continue
        assert value is None, name
    assert f.coordinated_cluster is False


def test_derivatives_positioning_for_btc() -> None:
    history = oi_series([1000.0 + 10 * i for i in range(25)])
    reading = DerivativesReading(
        symbol="BTCUSDT",
        retail_long_short_ratio=1.8,
        top_trader_account_ratio=1.2,
        top_trader_position_ratio=0.9,
        taker_buy_sell_ratio=1.05,
        open_interest_history=tuple(history),
        funding_rate=0.0001,
    )
    live = make_quote(
        "BTCUSDT",
        source=PriceSource.LIVE,
        last="83176.9",
        mark="83176.9",
        index="83216.158",
        bid="83176.9",
        ask="83177",
    )
    rates = [0.00001 * ((i % 7) - 3) for i in range(90)]
    live = live.model_copy(update={"funding_rate": Decimal("0.0003")})
    f = build_features(
        symbol="BTCUSDT",
        asset_class=AssetClass.CRYPTO,
        demo=None,
        live=live,
        derivatives=reading,
        funding=funding_points(rates),
        live_1h=[],
        demo_index_1h=[],
        crowd=UNMEASURED,
        calendar=[],
        policy=POLICY_V1,
    )
    assert f.retail_long_short_ratio == 1.8
    assert f.top_trader_long_short_ratio == 0.9  # position-weighted wins
    assert f.taker_buy_sell_ratio == 1.05
    assert f.oi_change_1h_pct == pytest.approx((1240 / 1230 - 1) * 100)
    assert f.oi_change_24h_pct == pytest.approx(24.0)
    assert f.funding_rate_live == 0.0003  # from the live ticker, not the toolkit reading
    assert f.funding_z_live == pytest.approx(funding_z(funding_points(rates), 0.0003, 90))
    account_only = reading.model_copy(update={"top_trader_position_ratio": None})
    g = build_features(
        symbol="BTCUSDT",
        asset_class=AssetClass.CRYPTO,
        demo=None,
        live=live,
        derivatives=account_only,
        funding=[],
        live_1h=[],
        demo_index_1h=[],
        crowd=UNMEASURED,
        calendar=[],
        policy=POLICY_V1,
    )
    assert g.top_trader_long_short_ratio == 1.2
    assert g.funding_z_live is None  # no history, no z


def test_open_interest_change_needs_a_quote_to_place_it_in_time() -> None:
    reading = DerivativesReading(
        symbol="NVDAUSDT",
        retail_long_short_ratio=None,
        top_trader_account_ratio=None,
        top_trader_position_ratio=None,
        taker_buy_sell_ratio=None,
        open_interest_history=tuple(oi_series([1000.0, 1010.0])),
    )
    assert nvda_features(derivatives=reading).oi_change_1h_pct == pytest.approx(1.0)
    assert nvda_features(derivatives=reading, demo=None, live=None).oi_change_1h_pct is None


def test_social_signals_from_the_social_report() -> None:
    f = nvda_features(crowd=social_report())
    assert f.social_mentions_24h == 2
    # Two distinct stories, first seen T0-3h and last seen T0-1h: one story an hour. The three
    # copies of the first story are not counted again as pace.
    assert f.social_velocity_per_hour == pytest.approx(1.0)
    assert f.coordinated_cluster is True
    quiet = nvda_features(crowd=social_report(coordinated=False))
    assert quiet.coordinated_cluster is False
    other = social_signals(social_report(), "TSLAUSDT")
    assert other == (0, None, False)  # measured zero, not unmeasured
    assert social_signals(UNMEASURED, "NVDAUSDT") == (None, None, False)


def test_one_story_has_no_pace_and_the_rate_floor_is_one_hour() -> None:
    report = social_report()
    single = report.model_copy(update={"clusters": report.clusters[:1]})
    assert social_signals(single, "NVDAUSDT")[1] is None
    close = report.clusters[1].model_copy(
        update={
            "first_seen": report.clusters[0].first_seen + timedelta(minutes=5),
            "last_seen": report.clusters[0].first_seen + timedelta(minutes=10),
        }
    )
    tight = report.model_copy(update={"clusters": (report.clusters[0], close)})
    # Two stories inside 30 minutes read as two per hour, not four.
    assert social_signals(tight, "NVDAUSDT")[1] == pytest.approx(2.0)


def test_next_earnings_matches_perp_or_underlying_and_ignores_other_rows() -> None:
    soon = T0 + timedelta(days=1)
    later = T0 + timedelta(days=30)
    calendar = [
        CalendarItem(
            symbol="NVDA", kind="earnings", at=later, title="Q3", source="equity_calendar"
        ),
        CalendarItem(
            symbol="NVDAUSDT", kind="earnings", at=soon, title="Q3", source="equity_calendar"
        ),
        CalendarItem(symbol="NVDA", kind="form4", at=T0, title="Form 4", source="insider"),
        CalendarItem(symbol="TSLA", kind="earnings", at=T0, title="Q3", source="equity_calendar"),
        CalendarItem(symbol=None, kind="macro", at=T0, title="CPI", source="macro"),
    ]
    assert nvda_features(calendar=calendar).next_earnings_at == soon
    btc = make_quote("BTCUSDT", source=PriceSource.LIVE)
    crypto = build_features(
        symbol="BTCUSDT",
        asset_class=AssetClass.CRYPTO,
        demo=None,
        live=btc,
        derivatives=None,
        funding=[],
        live_1h=[],
        demo_index_1h=[],
        crowd=UNMEASURED,
        calendar=[c.model_copy(update={"symbol": "BTC"}) for c in calendar],
        policy=POLICY_V1,
    )
    assert crypto.next_earnings_at is None


def test_non_finite_venue_values_never_reach_a_feature() -> None:
    demo, _ = nvda_inputs()
    broken = demo.model_copy(
        update={"index": Decimal("0"), "bid": Decimal("0"), "ask": Decimal("0")}
    )
    f = nvda_features(demo=broken)
    assert f.demo_mark_index_gap_bps is None
    assert f.demo_spread_bps is None


# --- facts -------------------------------------------------------------------------------------


def book_with_position() -> BookState:
    book = empty_book("10000")
    position = Position(
        symbol="NVDAUSDT",
        qty=Decimal("-2.25"),
        avg_entry=Decimal("222.84"),
        opened_at=T0 - timedelta(hours=30),
        last_increase_at=T0 - timedelta(hours=6),
        realized_pnl=Decimal("0"),
        fees_paid=Decimal("0.30"),
        stop_price=Decimal("231.75"),
        stop_venue_id="stop-1",
        last_decision_id="decision-1",
    )
    return book.model_copy(
        update={
            "equity": Decimal("10012.5"),
            "peak_equity": Decimal("10020"),
            "day_open_equity": Decimal("10005"),
            "positions": {"NVDAUSDT": position},
            "marks": {"NVDAUSDT": Decimal("218.40")},
            "fees_today": Decimal("0.30"),
            "fees_total": Decimal("0.30"),
            "rebalances_today": {"NVDAUSDT": 1},
        }
    )


def test_facts_hold_every_numeric_feature_field() -> None:
    f = nvda_features(crowd=social_report())
    mood = MarketMood(crypto_fear_greed=24, market_fear_greed=60)
    facts = facts_from({"NVDAUSDT": f}, mood, None)
    for name, value in f:
        key = f"NVDAUSDT.{name}"
        if isinstance(value, bool) or not isinstance(value, int | float):
            assert key not in facts
        else:
            assert facts[key] == value
    assert facts["mood.crypto_fear_greed"] == 24
    assert facts["mood.market_fear_greed"] == 60
    assert not any(k.startswith("book.") for k in facts)
    assert all(math.isfinite(v) for v in facts.values())


def test_facts_hold_the_books_recorded_numbers_under_the_prompts_keys() -> None:
    book = book_with_position()
    facts = facts_from({}, MarketMood(), book)
    assert facts == {
        "book.equity": 10012.5,
        "book.starting_equity": 10000.0,
        "book.peak_equity": 10020.0,
        "book.day_open_equity": 10005.0,
        "book.realized_total": 0.0,
        "book.fees_today": 0.30,
        "book.fees_total": 0.30,
        "book.consecutive_losses": 0.0,
        "book.open_positions": 1.0,
        "NVDAUSDT.model_orders_today": 1.0,
        "NVDAUSDT.position_qty": -2.25,
        "NVDAUSDT.position_avg_entry": 222.84,
        "NVDAUSDT.position_realized_pnl": 0.0,
        "NVDAUSDT.position_fees_paid": 0.30,
        "NVDAUSDT.position_mark": 218.40,
        "NVDAUSDT.position_stop_price": 231.75,
    }


def test_a_flat_position_and_a_missing_mark_contribute_only_what_is_known() -> None:
    book = book_with_position()
    facts = facts_from({}, MarketMood(), book.model_copy(update={"marks": {}}))
    assert "NVDAUSDT.position_mark" not in facts
    assert facts["NVDAUSDT.position_stop_price"] == 231.75
    flat = book.positions["NVDAUSDT"].model_copy(update={"qty": Decimal("0")})
    flat_book = book.model_copy(update={"positions": {"NVDAUSDT": flat}})
    flat_facts = facts_from({}, MarketMood(), flat_book)
    assert not [k for k in flat_facts if ".position_" in k]
    assert flat_facts["book.open_positions"] == 0


def test_every_book_fact_key_is_one_the_prompt_renders_in_its_section() -> None:
    """The prompt lists the book, position and instrument facts it shows in fixed sections; a key
    emitted here under any other name would fall into its catch-all section instead."""
    prompt = pytest.importorskip("sentiment_agent.decision.prompt", exc_type=ImportError)
    facts = facts_from({}, MarketMood(), book_with_position())
    for key in facts:
        head, _, name = key.partition(".")
        if head == "book":
            assert key in prompt.BOOK_FACTS, key
        else:
            assert head == "NVDAUSDT", key
            assert name in (*prompt.POSITION_FACTS, *prompt.INSTRUMENT_FACTS), key
    numeric = [
        name
        for name, field in PositioningFeatures.model_fields.items()
        if name not in ("symbol", "asset_class", "coordinated_cluster", "next_earnings_at")
        and field.annotation is not None
    ]
    assert set(prompt.FEATURE_FACTS) == set(numeric)


def test_crowd_and_coverage_facts() -> None:
    assert crowd_facts(social_report()) == {
        "crowd.items": 4.0,
        "crowd.withheld": 0.0,
        "crowd.distinct_stories": 2.0,
        "crowd.duplication_ratio": 2.0,
        "crowd.coordinated_clusters": 1.0,
    }
    at = datetime(2026, 9, 23, 13, 0, tzinfo=UTC)
    calls = [
        SourceCall(
            call_id=f"c{i}",
            surface=ToolkitSurface.SIGNAL_MCP,
            source=f"s{i}",
            health=health,
            started_at=at,
            latency_ms=0,
            rows=0,
            blob=None,
        )
        for i, health in enumerate(
            [SourceHealth.OK, SourceHealth.EMPTY, SourceHealth.HOLLOW, SourceHealth.DISABLED]
        )
    ]
    assert coverage_facts(calls) == {
        "coverage.calls_total": 4.0,
        "coverage.calls_answered": 2.0,
        "coverage.calls_failed": 1.0,
        "coverage.calls_disabled": 1.0,
    }
