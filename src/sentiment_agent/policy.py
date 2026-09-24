"""Policy v1: every number the agent obeys, each with the measurement it answers.

Frozen at genesis: its hash is in the genesis record, and a running agent refuses to trade when
the policy it loaded does not hash to the one the genesis pre-registered. Changing any value after
genesis is an :class:`~sentiment_agent.types.Amendment` event, logged and published, never an edit.
Thresholds are never tuned inside the paper window.

Sources cited below:

* ``validation/demo_venue/*`` — the Demo-venue measurements that chose this design, copied here
  with their scripts so they can be re-run.
* ``validation/demo_venue/universe_probe.json`` — keyless ``instruments`` and ``tickers`` reads for
  the 14 instruments on Demo and live, 2026-09-24 10:31 UTC.
"""

from typing import Final

from sentiment_agent.types import (
    AssetClass,
    BreakerRule,
    DecisionRule,
    GuardBasis,
    GuardId,
    Mandate,
    MetricDefinition,
    Policy,
    Thinking,
    TriggerRule,
    UniverseEntry,
    WeekendRule,
)

_GAP_BASIS = (
    "p99 of |Demo close / live close - 1| over the Demo history, 1H candles "
    "(validation/demo_venue/{file})"
)

UNIVERSE: Final[tuple[UniverseEntry, ...]] = (
    UniverseEntry(
        symbol="BTCUSDT",
        asset_class=AssetClass.CRYPTO,
        demo_live_gap_p99_bps=8.3,
        basis=_GAP_BASIS.format(file="demo_integrity.json")
        + "; 0 mark-vs-index excursions over 3% in the last 30 days",
    ),
    UniverseEntry(
        symbol="SP500USDT",
        asset_class=AssetClass.US_INDEX,
        demo_live_gap_p99_bps=266.5,
        basis=_GAP_BASIS.format(file="demo_integrity.json") + "; listed on Demo 2026-09-03",
    ),
    UniverseEntry(
        symbol="NDX100USDT",
        asset_class=AssetClass.US_INDEX,
        demo_live_gap_p99_bps=63.2,
        basis=_GAP_BASIS.format(file="demo_integrity.json") + "; listed on Demo 2026-09-03",
    ),
    *(
        UniverseEntry(
            symbol=symbol,
            asset_class=AssetClass.US_EQUITY,
            demo_live_gap_p99_bps=gap,
            basis=_GAP_BASIS.format(file="fade_demo.json (tracking)")
            + "; 0 mark-vs-index excursions over 3% since Demo listing 2026-08-25",
        )
        for symbol, gap in (
            ("MSTRUSDT", 316.2),
            ("HOODUSDT", 383.0),
            ("CRCLUSDT", 308.7),
            ("SNDKUSDT", 348.2),
            ("COINUSDT", 165.0),
            ("TSLAUSDT", 113.7),
            ("GOOGLUSDT", 104.7),
            ("METAUSDT", 67.2),
            ("NVDAUSDT", 129.9),
            ("AMZNUSDT", 92.0),
            ("AAPLUSDT", 82.3),
        )
    ),
)

EXCLUDED: Final[tuple[str, ...]] = ("ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT")
"""Refused outright: Demo mark-vs-index excursions over 3% in the last 30 days (ETH 4, SOL 8,
DOGE 11, XRP 252) and Demo-live close gaps with p99 up to 19,096 bps
(validation/demo_venue/demo_integrity.json). Everything else outside UNIVERSE is refused by G11."""

MANDATE_TEXT: Final = (
    "You manage a paper book on Bitget's Demo venue. You hold a risk budget of up to 25% of "
    "equity gross, at most 5% of equity in any one name, long or short. At every decision you "
    "either deploy some of that budget, with a written thesis and invalidation per position, or "
    "decline it in writing. Flat with reasons is a valid, counted answer. A risk kernel you do "
    "not control may shrink or refuse what you ask for; it can never add to it."
)

METRICS: Final[tuple[MetricDefinition, ...]] = (
    MetricDefinition(
        name="hourly_return",
        formula="r_t = E_t / E_{t-1} - 1 on the UTC hour grid, E = equity_book",
        unit="fraction",
        notes="Same grid and definition as validation/demo_venue/envelope_clean.py",
    ),
    MetricDefinition(
        name="sharpe_ann",
        formula="mean(r) / pstdev(r) * sqrt(8760)",
        unit="annualised ratio",
        notes="Population stdev, as envelope_clean.py. Undefined (None) when pstdev is 0.",
    ),
    MetricDefinition(
        name="sharpe_se_ann",
        formula="sqrt((1 + 0.5 * S_h^2) / n) * sqrt(8760), S_h = mean(r) / pstdev(r)",
        unit="annualised ratio",
        notes="Lo (2002) iid standard error; about 11 at n = 72 with S_h near 0",
    ),
    MetricDefinition(
        name="sortino_ann",
        formula="mean(r) / sqrt(mean(min(r, 0)^2)) * sqrt(8760)",
        unit="annualised ratio",
    ),
    MetricDefinition(
        name="max_drawdown",
        formula="min_t (E_t / max_{s<=t} E_s - 1)",
        unit="fraction (non-positive)",
    ),
    MetricDefinition(
        name="win_rate",
        formula="count(net_pnl > 0) / count(closed trades)",
        unit="fraction",
        notes="A closed trade is one symbol from flat to flat; a flip closes one and opens one. "
        "net_pnl includes both fees. None when no trade has closed.",
    ),
    MetricDefinition(
        name="turnover",
        formula="sum(|fill notional|) / mean(E)",
        unit="multiple of equity",
    ),
    MetricDefinition(
        name="ci90",
        formula="moving-block bootstrap of r, block = ceil(n^(1/3)) hours, 10,000 resamples, "
        "seed 20260924; 5th and 95th percentiles over the resamples where the statistic is "
        "defined, the share where it is not published beside the band; the Sharpe ratio of an "
        "all-zero resample is 0",
        unit="per metric",
        notes="Descriptive, not inferential: 72 hourly marks cannot separate skill from luck.",
    ),
)

EXPECTED_ENVELOPE: Final[dict[str, str]] = {
    "source": "validation/demo_venue/envelope_clean.json (no-edge coin-flip book, "
    "17,400 windows of 72h, 25% gross)",
    "return_on_equity_bps": "median -8.6, p05 -88.1, p95 +80.6",
    "max_drawdown_pct": "median -0.42, p95 -1.10, worst -2.48",
    "sharpe_ann": "median -2.3, p05 -20.2, p95 +16.8",
    "win_rate": "median 0.45, p05 0.18, p95 0.71",
    "closed_trades": "median 11",
}

GUARD_BASES: Final[tuple[GuardBasis, ...]] = (
    GuardBasis(
        guard=GuardId.G1_VENUE_INTEGRITY,
        rule="Refuse or exit an instrument whose Demo mark departs more than 3% from its Demo "
        "index, whose Demo-live price gap exceeds its measured p99, or whose Demo index has not "
        "moved (mean |1H move| over 3h below 2 bps) while its session should be open.",
        basis="Demo flash prints (BTCUSDT Demo mark 65.6% from its index, 2026-06-29 00:00 UTC); "
        "excursion counts and p99 gaps in demo_integrity.json and fade_demo.json; weekend Demo "
        "index moves 0.1-1.5 bps/h vs 18-66 bps/h on weekdays (weekend_vol.json)",
    ),
    GuardBasis(
        guard=GuardId.G2_WEEKEND_FREEZE,
        rule="US-equity and US-index legs flat from Friday 20:00 UTC to Monday 00:00 UTC; "
        "pre-flatten from 19:45; no new equity exposure in the 2h before the freeze.",
        basis="UTA Demo does not track these perps on weekends. Mostly it freezes: weekend mean "
        "|1H move| of the Demo index 0.1-1.5 bps against 5.6-29.1 bps live (weekend_vol.json); "
        "MSTR +204 bps live vs +5 bps Demo from Saturday 00:00 to 12:00 UTC on 2026-09-19. "
        "Sometimes it jumps on its own: AAPL +80 bps Demo vs +17 bps live on 2026-09-12 "
        "(fade_demo2.txt)",
    ),
    GuardBasis(
        guard=GuardId.G3_SIZE,
        rule="At most 5% of equity per name and 25% gross.",
        basis="Envelope at 25% gross: worst 72h drawdown -2.48% under no edge "
        "(envelope_clean.json); the drawdown is the one quantitative number design controls",
    ),
    GuardBasis(
        guard=GuardId.G4_STOP,
        rule="Every exposure-adding order carries a venue stop 4% from entry, triggered on mark.",
        basis="Envelope simulation uses -4% per position (envelope_clean.py); a stop that lives on "
        "the venue survives a crashed agent, one that lives in our loop does not",
    ),
    GuardBasis(
        guard=GuardId.G5_DAILY_KILL,
        rule="Book down 1.5% from the 00:00 UTC equity flattens everything; halted until the "
        "next UTC day.",
        basis="Envelope simulation uses a -1.5% daily kill (envelope_clean.py)",
    ),
    GuardBasis(
        guard=GuardId.G6_TURNOVER,
        rule="At most 2 model-initiated orders per name per UTC day; no increase or side flip "
        "within 24h of the last increase unless the model declares the stated invalidation "
        "fired. Reductions are never blocked (see DESIGN.md §10.4).",
        basis="Round-trip cost 12-16 bps against a measured intraday edge of ~0.00% on these "
        "perps; 360+ intraday variants were negative in the author's earlier study (not in "
        "this repository)",
    ),
    GuardBasis(
        guard=GuardId.G7_FEE_BUDGET,
        rule="No new exposure once fees paid today reach 7 bps of equity, or 20 bps over the "
        "window.",
        basis="Demo taker fee 6 bps (instruments takerFeeRate 0.0006 on all 14 names, "
        "universe_probe.json); envelope expects ~11 round trips at 5% = 6.6 bps of equity per "
        "72h; caps are about 3x that",
    ),
    GuardBasis(
        guard=GuardId.G8_TAKER_ONLY,
        rule="Market (taker) orders only; refuse an opening when the Demo spread exceeds 20 bps.",
        basis="Maker fills are adversely selected: a +90% limit-fill simulation replayed at "
        "-0.45% in the author's earlier fill-model replay (not in this repository). Demo "
        "spreads measured 0.1-6.6 bps (universe_probe.json)",
    ),
    GuardBasis(
        guard=GuardId.G9_GROUNDING,
        rule="A target whose thesis, invalidation or view states a number that does not resolve "
        "to a snapshot fact within 2% may not add exposure.",
        basis="An unattributable figure is where an unsupported fact enters a trade. The check is "
        "ported from ARGUS agents/grounding.py (MIT); its fabricated-figure test set is re-run in "
        "this repository (tests/decision/test_grounding.py) rather than cited",
    ),
    GuardBasis(
        guard=GuardId.G10_BREAKER,
        rule="Drawdown from peak 2.5% -> reduce-only; 4% -> halted; 4 losing trades in a row -> "
        "reduce-only; a snapshot older than 15 min or a quote older than 120 s cannot open; "
        "a model outage flattens the book.",
        basis="2.5% is past the worst no-edge 72h drawdown (-2.48%, envelope_clean.json); "
        "losing-streak and staleness rules ported from ARGUS risk/circuit.py (MIT); a model "
        "outage never leads to a deterministic trade (DESIGN.md §9.5)",
    ),
    GuardBasis(
        guard=GuardId.G11_ELIGIBILITY,
        rule="Only the 14 universe symbols, only when Demo lists them online, only at or above "
        "minOrderQty and minOrderAmount after rounding to quantityMultiplier; orders above "
        "maxMarketOrderQty are split.",
        basis="Demo instrument limits (e.g. NVDAUSDT maxMarketOrderQty 60, minOrderQty 0.01) in "
        "universe_probe.json",
    ),
)

POLICY_V1: Final[Policy] = Policy(
    version="policy-v1",
    universe=UNIVERSE,
    excluded=EXCLUDED,
    per_name_max=0.05,
    gross_max=0.25,
    stop_loss_pct=0.04,
    stop_trigger="mark",
    daily_kill_pct=0.015,
    max_rebalances_per_name_per_day=2,
    min_hold_hours=24,
    fee_budget_daily_bps=7.0,
    fee_budget_window_bps=20.0,
    taker_only=True,
    max_open_spread_bps=20.0,
    mark_index_max_gap=0.03,
    stale_index_min_move_bps_3h=2.0,
    grounding_tolerance=0.02,
    weekend=WeekendRule(
        freeze_weekday=4,
        freeze_hour=20,
        reopen_weekday=0,
        reopen_hour=0,
        preflatten_minutes=15,
        no_open_buffer_hours=2.0,
        basis="Same window as the envelope simulation's tradable() (envelope_clean.py); "
        "the Demo freeze itself is in weekend_vol.json",
    ),
    breaker=BreakerRule(
        reduce_only_drawdown=0.025,
        halt_drawdown=0.04,
        losing_streak_reduce_only=4,
        snapshot_max_age_minutes=15,
        quote_max_age_seconds=120,
        basis="See GuardBasis G10",
    ),
    triggers=TriggerRule(
        us_open_local="09:30",
        funding_heartbeat_hours_utc=(0, 8, 16),
        fear_greed_low=25,
        fear_greed_high=76,
        funding_z_threshold=2.0,
        funding_z_lookback_settlements=90,
        oi_jump_quantile=0.99,
        oi_jump_lookback_days=30,
        coordinated_min_sources=3,
        coordinated_window_minutes=120,
        earnings_lookahead_hours=24,
        cooldown_minutes=240,
        max_event_decisions_per_day=8,
        basis="Heartbeat and event list: DESIGN.md §8. Fear & Greed bands are Bitget's own "
        "(bitget-signal skills/sentiment-analyst/SKILL.md: 0-25 extreme fear, 76-100 extreme "
        "greed). Coordination constants from ARGUS agents/novelty.py (3 sources inside 2h). "
        "Live BTCUSDT funding settles every 8h (instruments fundInterval, re-read at genesis).",
    ),
    decision=DecisionRule(
        model="qwen3.8-max",
        thinking_heartbeat=Thinking.FULL,
        thinking_event=Thinking.LOW,
        daily_token_cap=2_400_000,
        max_attempts=3,
        max_completion_tokens=8192,
        call_timeout_seconds=600,
        min_horizon_hours=24,
        temperature=0.0,
        basis="FULL reasoning on the <=4 daily heartbeats, LOW on <=8 event decisions: measured "
        "FULL ~4.3k and LOW ~0.8k completion tokens per call (ARGUS llm/qwen.py). Streaming is "
        "required for FULL: the gateway closes unstreamed requests at ~120 s. The cap is sized "
        "to the budget's own projection (llm/budget.py: one token per prompt byte plus the "
        "completion cap), prompt included, for the busiest day the trigger rules allow: recorded "
        "dry-run prompt 40,784 B (system 11,898 + user 28,886), x1.25 headroom = 50,980 B; one "
        "attempt = 50,980 + 2x16 + 128 + 8,192 = 59,332; a retry adds 8,192 + 32 = 67,556; three "
        "attempts = 194,444 per decision; (4 heartbeats + 8 events) x 194,444 = 2,333,328, "
        "rounded up to 2,400,000. Real spend is lower (a byte bounds a token from above); the "
        "prompt_tokens of the first live calls re-check it before genesis. Owner requests are "
        "outside this sum; the heartbeats left in the day are reserved before an event decision "
        "is admitted. The cap is the owner's to approve.",
    ),
    mandate=Mandate(
        risk_budget_gross=0.25,
        per_name_max=0.05,
        min_horizon_hours=24,
        text=MANDATE_TEXT,
    ),
    guard_bases=GUARD_BASES,
    metrics=METRICS,
    expected_envelope=EXPECTED_ENVELOPE,
)

__all__ = ["EXCLUDED", "GUARD_BASES", "MANDATE_TEXT", "METRICS", "POLICY_V1", "UNIVERSE"]
