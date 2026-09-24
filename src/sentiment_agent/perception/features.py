"""Per-instrument positioning features, the market mood, and the flat ``facts`` map.

Everything here is a pure function of values the snapshot builder already fetched, so every feature
can be recomputed from the logged snapshot and its source blobs, and every one is tested against a
hand-computed series.

**Live measures the crowd; Demo measures the venue (DESIGN.md §6.1).** Funding, open interest, the
long/short and taker ratios, the 24h change and the overheating distance come from live data only.
Demo supplies the venue-integrity fields (mark-index gap, Demo-live gap, spread, the stale-index
detector) and ``demo_last``. Demo's funding and open interest are sandbox settings, not a crowd, and
no Demo funding or open-interest value is ever read into a feature: :func:`build_features` takes
them from ``live`` and nowhere else, and a test asserts it.

**Units.** ``*_pct`` fields are percent (``2.5`` is +2.5%), ``*_bps`` fields are basis points,
``funding_rate_live`` is the venue's own fraction per settlement (``0.000048``), ratios are
dimensionless. A value that was not measured is ``None``, never ``0``; a value that cannot be
computed honestly (a zero standard deviation, too little history, a gap in a series that must be
contiguous) is ``None`` too, because a number built on a degenerate input is a number the model
would believe.

**Facts.** :func:`facts_from` flattens the features, the mood and the book's recorded numbers into
``name -> value``; :func:`crowd_facts` and :func:`coverage_facts` add the crowd summary and the
source-coverage counts. Their union is the snapshot's ``facts``: the perception half of the
grounding reference (G9). The decision prompt (``decision.prompt.decision_facts``) calls the same
three functions, adds what only a decision cycle knows (the kernel's rules, the session clock, the
triggers, per-story figures in display order, the book as percentages and budgets left), and lets
the snapshot's logged value win a shared key. One key scheme, stable and cited by the model in
``evidence``: ``<SYMBOL>.<feature field>``, ``<SYMBOL>.position_<field>``,
``<SYMBOL>.model_orders_today``, ``book.<field>``, ``mood.<field>``, ``crowd.<field>``,
``coverage.<field>``. Numbers carried by third-party text are deliberately *not* facts: a figure
that reaches a thesis only through a post somebody else wrote is the unsupported fact G9 exists to
stop.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from decimal import Decimal
from itertools import pairwise
from typing import Final

from sentiment_agent.types import (
    AssetClass,
    BookState,
    CalendarItem,
    Candle,
    CrowdReport,
    DerivativesReading,
    FundingPoint,
    MarketMood,
    MoodReading,
    Policy,
    PositioningFeatures,
    Quote,
    SourceCall,
    SourceHealth,
)

MA_BARS: Final = 20
"""Bars in the moving average of the overheating feature (DESIGN.md §6.3: 20-bar 1H mean)."""

ATR_BARS: Final = 14
"""True ranges in the ATR of the overheating feature (DESIGN.md §6.3: 14-bar ATR)."""

HOUR: Final = timedelta(hours=1)

STALE_INDEX_MOVES: Final = 3
"""Hourly moves averaged by the stale-index detector (policy: mean |1H move| over 3h)."""

FUNDING_SD_FLOOR: Final = 1e-6
"""Smallest funding standard deviation a z-score is computed against.

The venue quotes funding to six decimals (``universe_probe.json``: ``0.000048``, ``-0.000009``),
so a series whose spread is below one tick is constant at the resolution it is measured in, and a
z-score against it measures rounding. Several equity perps settle at exactly ``0`` for long
stretches; their z-score is ``None``, not a large number."""

OI_MAX_AGE: Final = timedelta(hours=3)
"""The newest open-interest point must be at most this old for a change to be reported.

An hourly series stamped at the period start, plus a publication lag, can be close to two hours old
just before the hour; three hours tolerates that and still refuses a day-old series presented as the
last hour's move."""

OI_MATCH_TOLERANCE: Final = timedelta(minutes=5)
"""How far the reference point may sit from ``newest - hours`` and still define an ``hours`` change.

The same pairing rule as ``events.triggers.hourly_oi_changes_pct`` (``PAIR_TOLERANCE``, 5 minutes),
which computes the frozen open-interest-jump threshold at genesis: the observed change and the
threshold it is compared with must be the same statistic."""

SOCIAL_RATE_FLOOR: Final = HOUR
"""Minimum observation span for the social arrival rate, so ten stories inside five minutes read as
ten per hour rather than a hundred and twenty."""

# Bitget's own Fear & Greed bands (bitget-signal skills/sentiment-analyst/references/signal-guide.md
# lines 7-11): 0-25 Extreme Fear, 26-45 Fear, 46-55 Neutral, 56-75 Greed, 76-100 Extreme Greed. The
# two extremes come from the policy (they drive the fear_greed_extreme trigger and are frozen at
# genesis); the inner edges are the guide's.
_FEAR_UPPER: Final = 45
_NEUTRAL_UPPER: Final = 55
FEAR_GREED_LABELS: Final[Mapping[str, str]] = {
    "extreme_fear": "Extreme Fear",
    "fear": "Fear",
    "neutral": "Neutral",
    "greed": "Greed",
    "extreme_greed": "Extreme Greed",
}

SOCIAL_CHANNELS: Final = frozenset({"x", "reddit"})
"""Crowd text that counts as *social* for the social features. News and filings are shown to the
model as text and clustered in the crowd report, but they are not the crowd talking."""


# ================================================================================================
# Helpers
# ================================================================================================


def _finite(value: float | Decimal | int | None) -> float | None:
    """``float(value)`` when it is a finite number, else ``None``. Nothing non-finite is logged:
    canonical JSON refuses NaN and infinity, and a snapshot must always hash."""
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _uniform(values: Iterable[str], what: str) -> None:
    distinct = set(values)
    if len(distinct) > 1:
        raise ValueError(f"a {what} series mixes {sorted(distinct)}; series are never mixed")


def _ordered_candles(candles: Sequence[Candle]) -> list[Candle]:
    """One candle per ``open_time`` (the later occurrence wins), ascending, from one series."""
    _uniform((c.symbol for c in candles), "candle symbol")
    _uniform((c.source for c in candles), "candle source")
    _uniform((c.kind for c in candles), "candle kind")
    _uniform((c.interval for c in candles), "candle interval")
    by_time = {c.open_time: c for c in candles}
    return [by_time[t] for t in sorted(by_time)]


def _pct(ratio: float | None) -> float | None:
    return None if ratio is None else _finite(ratio * 100.0)


# ================================================================================================
# Feature math
# ================================================================================================


def funding_z(history: Sequence[FundingPoint], current: float, lookback: int) -> float | None:
    """Z-score of ``current`` against the last ``lookback`` settled funding rates.

    Point-in-time by construction: ``current`` is the venue's rate for the *next* settlement and the
    baseline is settlements that already happened, so the point being scored is never inside its
    own baseline (the self-inclusion bias ARGUS ``eval/cointegration_comparison.py`` measured in
    FinceptTerminal's formula). Population standard deviation over exactly ``lookback`` points, as
    ARGUS ``research/cointegration.py:498-520``: with fewer settlements than ``lookback`` the
    statistic would be a different one, so it is ``None`` rather than computed on what exists.
    ``None`` as well when the spread is below :data:`FUNDING_SD_FLOOR`.
    """
    if lookback < 2:
        raise ValueError("a funding z-score needs a lookback of at least 2 settlements")
    _uniform((p.symbol for p in history), "funding symbol")
    _uniform((p.source for p in history), "funding source")
    if not math.isfinite(current):
        return None
    by_time = {p.ts: p for p in history}
    ordered = [by_time[t] for t in sorted(by_time)]
    if len(ordered) < lookback:
        return None
    window = [float(p.rate) for p in ordered[-lookback:]]
    if not all(math.isfinite(x) for x in window):
        return None
    mean = math.fsum(window) / lookback
    sd = math.sqrt(math.fsum((x - mean) ** 2 for x in window) / lookback)
    if not sd >= FUNDING_SD_FLOOR:
        return None
    return _finite((current - mean) / sd)


def ma_distance_atr(
    candles: Sequence[Candle], *, ma: int = MA_BARS, atr: int = ATR_BARS
) -> float | None:
    """``(close - SMA(close, ma)) / ATR(atr)`` on the newest bar: how stretched price is.

    Positive is above its mean (a crowded, overheating long is the handbook's "reduce before
    overheating", handbook:233), negative below, in units of recent hourly range.

    * Computed on the bars given. The snapshot passes completed 1H bars (``venue.public_api``
      returns closed candles only), so the value is as of the last completed hour: a bar still
      forming has a partial range that would understate its true range.
    * ATR is the arithmetic mean of the last ``atr`` true ranges, ``TR = max(H - L,
      |H - C_prev|, |L - C_prev|)``. That is the seed value of Wilder's ATR (``bukosabino/ta``
      ``volatility.py:50``, MIT); the recursive Wilder form, and Bitget's own ``EMA(TR, n)``
      (``bitget-signal/skills/technical-analysis/src/kline_indicators.py`` ATR), depend on how
      many bars were fetched before the window, and a feature that changes with the fetch length
      cannot be recomputed from the log. Deliberate departure, stated here.
    * ``None`` with fewer than ``max(ma, atr + 1)`` bars, a non-positive close, or a zero ATR.
    """
    if ma < 1 or atr < 1:
        raise ValueError("ma and atr must be at least 1")
    bars = _ordered_candles(candles)
    if len(bars) < max(ma, atr + 1):
        return None
    closes = [float(b.close) for b in bars[-max(ma, atr + 1) :]]
    if any(not math.isfinite(c) or c <= 0 for c in closes):
        return None
    sma = math.fsum(closes[-ma:]) / ma
    tail = bars[-(atr + 1) :]
    ranges: list[float] = []
    for previous, current in pairwise(tail):
        prev_close = float(previous.close)
        high, low = float(current.high), float(current.low)
        ranges.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
    average_range = math.fsum(ranges) / atr
    if not average_range > 0:
        return None
    return _finite((closes[-1] - sma) / average_range)


def index_move_bps_3h(index_candles: Sequence[Candle]) -> float | None:
    """Mean ``|close_t / close_{t-1} - 1|`` in bps over the three newest hourly moves of an index.

    The stale-index detector of G1 (policy ``stale_index_min_move_bps_3h``). Same statistic as the
    measurement it answers (``validation/demo_venue/weekend_path.py``: consecutive closes exactly
    one hour apart, absolute return in bps, mean): the Demo index moved 0.1-1.5 bps/h on weekends
    against 18-66 bps/h on weekdays (``weekend_vol.json``).

    Pass **completed** candles only: a bar still forming has moved for only part of its hour and
    would read as stale early in every hour. The four newest bars must be consecutive hours; a gap
    makes the value ``None`` (unmeasured, so G1 fails closed for increases) rather than a mean over
    a longer span that would pass for three hours.
    """
    bars = _ordered_candles(index_candles)
    if any(b.kind != "index" for b in bars):
        raise ValueError("the stale-index detector reads index candles only")
    if len(bars) < STALE_INDEX_MOVES + 1:
        return None
    window = bars[-(STALE_INDEX_MOVES + 1) :]
    moves: list[float] = []
    for previous, current in pairwise(window):
        if current.open_time - previous.open_time != HOUR or previous.close <= 0:
            return None
        moves.append(abs(float(current.close / previous.close) - 1.0) * 10_000.0)
    return _finite(math.fsum(moves) / len(moves))


def oi_change_pct(
    history: Sequence[tuple[datetime, float]], *, hours: int, at: datetime
) -> float | None:
    """Percent change in open interest over ``hours``, ending at the newest point at or before
    ``at``.

    The reference is the point nearest ``newest - hours``, and must lie within
    :data:`OI_MATCH_TOLERANCE` of it, so a "1h change" is a one-hour change. The newest point must
    be at most :data:`OI_MAX_AGE` before ``at``. Points after ``at`` are ignored (point-in-time),
    and non-finite or non-positive values are dropped. ``None`` when any of that cannot be met.
    """
    if hours < 1:
        raise ValueError("hours must be at least 1")
    if at.tzinfo is None or at.utcoffset() != timedelta(0):
        raise ValueError("at must be timezone-aware UTC")
    by_time: dict[datetime, float] = {}
    for ts, value in history:
        if ts <= at and math.isfinite(value) and value > 0:
            by_time[ts] = value
    if len(by_time) < 2:
        return None
    stamps = sorted(by_time)
    newest = stamps[-1]
    if at - newest > OI_MAX_AGE:
        return None
    target = newest - timedelta(hours=hours)
    reference = min(stamps[:-1], key=lambda t: (abs(t - target), t))
    if abs(reference - target) > OI_MATCH_TOLERANCE:
        return None
    return _finite((by_time[newest] / by_time[reference] - 1.0) * 100.0)


def fear_greed_band(value: int, policy: Policy) -> str:
    """Bitget's band for a 0-100 Fear & Greed reading, extremes taken from the policy."""
    if value <= policy.triggers.fear_greed_low:
        return "extreme_fear"
    if value <= _FEAR_UPPER:
        return "fear"
    if value <= _NEUTRAL_UPPER:
        return "neutral"
    if value < policy.triggers.fear_greed_high:
        return "greed"
    return "extreme_greed"


def mood_from(reading: MoodReading, policy: Policy) -> MarketMood:
    """The mood the model sees, from every Fear & Greed source that answered.

    The primary crypto reading is used when present; otherwise the second crypto source stands in,
    labelled with Bitget's band name because that source carries no label of its own. A source's
    own label is kept as it said it. ``crypto_sources_agree`` is ``True``/``False`` only when both
    crypto sources answered, and says whether they fall in the same Bitget band; ``None``
    otherwise.
    """
    crypto = reading.crypto_fear_greed
    crypto_label = reading.crypto_fear_greed_label
    if crypto is None and reading.crypto_fear_greed_alt is not None:
        crypto = reading.crypto_fear_greed_alt
        crypto_label = None
    if crypto is not None and not crypto_label:
        crypto_label = FEAR_GREED_LABELS[fear_greed_band(crypto, policy)]

    market_label = reading.market_fear_greed_label
    if reading.market_fear_greed is not None and not market_label:
        market_label = FEAR_GREED_LABELS[fear_greed_band(reading.market_fear_greed, policy)]

    agree: bool | None = None
    if reading.crypto_fear_greed is not None and reading.crypto_fear_greed_alt is not None:
        agree = fear_greed_band(reading.crypto_fear_greed, policy) == fear_greed_band(
            reading.crypto_fear_greed_alt, policy
        )
    return MarketMood(
        crypto_fear_greed=crypto,
        crypto_fear_greed_label=crypto_label if crypto is not None else None,
        market_fear_greed=reading.market_fear_greed,
        market_fear_greed_label=market_label if reading.market_fear_greed is not None else None,
        crypto_sources_agree=agree,
    )


def _calendar_names(item_symbol: str | None, symbol: str) -> bool:
    """A calendar row names ``symbol`` whether it carries the perp (``NVDAUSDT``) or the underlying
    ticker (``NVDA``); ARGUS ``market/bitget_mcp.py`` ``underlying_of`` strips the suffix the same
    way."""
    if item_symbol is None:
        return False
    wanted = item_symbol.strip().upper()
    return wanted in (symbol.upper(), symbol.upper().removesuffix("USDT"))


def social_signals(crowd: CrowdReport, symbol: str) -> tuple[int | None, float | None, bool]:
    """``(social_mentions_24h, social_velocity_per_hour, coordinated_cluster)`` for one symbol.

    ``crowd.mentions`` carries an entry for every universe symbol when the text was measured (zero
    included, ``crowd.novelty.build_report``) and none when it was not, which is how unmeasured
    stays ``None``. See :func:`build_features` for the definitions.
    """
    naming = [c for c in crowd.clusters if symbol in c.symbols]
    coordinated = any(c.coordinated for c in naming)
    mentions = crowd.mentions.get(symbol)
    if mentions is None:
        return None, None, coordinated
    if len(naming) < 2:
        return mentions, None, coordinated
    span = max(c.last_seen for c in naming) - min(c.first_seen for c in naming)
    hours = max(span, SOCIAL_RATE_FLOOR).total_seconds() / 3600.0
    return mentions, _finite(len(naming) / hours), coordinated


def build_features(
    *,
    symbol: str,
    asset_class: AssetClass,
    demo: Quote | None,
    live: Quote | None,
    derivatives: DerivativesReading | None,
    funding: Sequence[FundingPoint],
    live_1h: Sequence[Candle],
    demo_index_1h: Sequence[Candle],
    crowd: CrowdReport,
    calendar: Sequence[CalendarItem],
    policy: Policy,
) -> PositioningFeatures:
    """Everything the model and the fixed-rule baselines read about one instrument.

    Inputs are taken as given; the caller (``perception.snapshot``) is responsible for passing the
    right series: ``funding`` settled live rates at or before the snapshot, ``live_1h`` live
    ``market`` candles, ``demo_index_1h`` **completed** Demo ``index`` candles, ``crowd`` the report
    over *social* text only (X and Reddit, :data:`SOCIAL_CHANNELS`), and ``calendar`` rows dated
    today (New York) or later, because this function has no clock. Every quote must be for
    ``symbol`` and from the environment its parameter names; anything else raises, because a Demo
    number in a live field is exactly the confusion DESIGN.md §6.1 forbids.

    * Social: ``social_mentions_24h`` is distinct X and Reddit stories naming the symbol (copies
      counted once, withheld items never counted); ``social_velocity_per_hour`` is those stories
      per hour of the span they arrived over (at least :data:`SOCIAL_RATE_FLOOR`), ``None`` for
      fewer than two stories, which have no rate; ``coordinated_cluster`` is whether any of them is
      coordinated (§6.4). Copies are deliberately not counted in the rate: a burst of copies is the
      coordination flag's evidence, and counting it again as velocity would let one promoted post
      read as a crowd.
    * Top-trader ratio: the position-weighted ratio when present, else the account ratio. Position
      weight is money, which is what a crowded trade is made of.
    * Open-interest changes are judged against the newest quote ``fetched_at`` (the snapshot's
      observation time); with neither quote there is no time to judge the series' age against, and
      they are ``None``.
    * ``next_earnings_at``: the earliest earnings row naming the symbol, equities only.
    """
    for quote, expected in ((demo, "demo"), (live, "live")):
        if quote is None:
            continue
        if quote.symbol != symbol:
            raise ValueError(f"{expected} quote for {quote.symbol} passed for {symbol}")
        if quote.source.value != expected:
            raise ValueError(f"a {quote.source} quote passed as the {expected} quote for {symbol}")
    if derivatives is not None and derivatives.symbol != symbol:
        raise ValueError(f"derivatives for {derivatives.symbol} passed for {symbol}")

    funding_rate_live = _finite(live.funding_rate) if live is not None else None
    funding_z_live = (
        funding_z(funding, funding_rate_live, policy.triggers.funding_z_lookback_settlements)
        if funding_rate_live is not None
        else None
    )

    # Open-interest changes are placed in time by the moment the quotes were fetched (the snapshot's
    # observation time); with no quote at all there is nothing to judge the series' age against, and
    # a change that cannot be shown to be current is not reported.
    oi_1h = oi_24h = None
    observed = [q.fetched_at for q in (demo, live) if q is not None]
    if derivatives is not None and derivatives.open_interest_history and observed:
        observed_at = max(observed)
        oi_1h = oi_change_pct(derivatives.open_interest_history, hours=1, at=observed_at)
        oi_24h = oi_change_pct(derivatives.open_interest_history, hours=24, at=observed_at)

    top_ratio = None
    if derivatives is not None:
        top_ratio = (
            derivatives.top_trader_position_ratio
            if derivatives.top_trader_position_ratio is not None
            else derivatives.top_trader_account_ratio
        )

    demo_live_gap = None
    if demo is not None and live is not None and live.last > 0:
        demo_live_gap = _finite(abs(demo.last / live.last - 1) * 10_000)

    mentions, velocity, coordinated = social_signals(crowd, symbol)

    next_earnings = None
    if asset_class is AssetClass.US_EQUITY:
        dates = [
            item.at
            for item in calendar
            if item.kind == "earnings"
            and item.at is not None
            and _calendar_names(item.symbol, symbol)
        ]
        next_earnings = min(dates) if dates else None

    return PositioningFeatures(
        symbol=symbol,
        asset_class=asset_class,
        demo_last=_finite(demo.last) if demo is not None else None,
        live_last=_finite(live.last) if live is not None else None,
        funding_rate_live=funding_rate_live,
        funding_z_live=funding_z_live,
        open_interest_live=_finite(live.open_interest) if live is not None else None,
        oi_change_1h_pct=oi_1h,
        oi_change_24h_pct=oi_24h,
        retail_long_short_ratio=(
            _finite(derivatives.retail_long_short_ratio) if derivatives is not None else None
        ),
        top_trader_long_short_ratio=_finite(top_ratio),
        taker_buy_sell_ratio=(
            _finite(derivatives.taker_buy_sell_ratio) if derivatives is not None else None
        ),
        price_change_24h_pct=(_pct(_finite(live.price_change_24h)) if live is not None else None),
        ma20_distance_atr=ma_distance_atr(live_1h) if live_1h else None,
        social_mentions_24h=mentions,
        social_velocity_per_hour=velocity,
        coordinated_cluster=coordinated,
        demo_mark_index_gap_bps=(
            _finite(demo.mark_index_gap * 10_000) if demo is not None else None
        ),
        demo_live_gap_bps=demo_live_gap,
        demo_spread_bps=_finite(demo.spread_bps) if demo is not None else None,
        demo_index_move_bps_3h=index_move_bps_3h(demo_index_1h) if demo_index_1h else None,
        next_earnings_at=next_earnings,
    )


# ================================================================================================
# Facts: the perception half of the grounding reference
# ================================================================================================


def _put(facts: dict[str, float], key: str, value: float | Decimal | int | None) -> None:
    number = _finite(value)
    if number is not None:
        facts[key] = number


def facts_from(
    features: Mapping[str, PositioningFeatures], mood: MarketMood, book: BookState | None
) -> dict[str, float]:
    """Flatten features, mood and the book's recorded numbers into ``name -> value``.

    Every numeric feature field is included generically, so a field added to
    :class:`PositioningFeatures` is a fact without an edit here; booleans and timestamps are not
    numbers the model is asked to quote and are left out. Values are exact, not rounded: the prompt
    prints them to six significant digits and grounding compares within a relative tolerance.

    The book contributes only the numbers it holds as recorded (equity, realized P&L, fees, entry,
    mark, stop, order counts), under the keys the prompt renders them with. Percentages, weights,
    budgets left and hours held are derived by the prompt at decision time
    (``decision.prompt._book_facts``, ``_position_facts``), because they depend on the decision
    clock; a second derivation here could only disagree with it, and the snapshot's value would win.
    """
    facts: dict[str, float] = {}
    for symbol, feature in features.items():
        for name, value in feature:
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            _put(facts, f"{symbol}.{name}", value)

    _put(facts, "mood.crypto_fear_greed", mood.crypto_fear_greed)
    _put(facts, "mood.market_fear_greed", mood.market_fear_greed)

    if book is not None:
        _book_facts(facts, book)
    return facts


def _book_facts(facts: dict[str, float], book: BookState) -> None:
    _put(facts, "book.equity", book.equity)
    _put(facts, "book.starting_equity", book.starting_equity)
    _put(facts, "book.peak_equity", book.peak_equity)
    _put(facts, "book.day_open_equity", book.day_open_equity)
    _put(facts, "book.realized_total", book.realized_total)
    _put(facts, "book.fees_today", book.fees_today)
    _put(facts, "book.fees_total", book.fees_total)
    _put(facts, "book.consecutive_losses", book.consecutive_losses)
    _put(facts, "book.open_positions", sum(1 for p in book.positions.values() if not p.is_flat))
    for symbol, count in book.rebalances_today.items():
        _put(facts, f"{symbol}.model_orders_today", count)
    for symbol, position in book.positions.items():
        if position.is_flat:
            continue
        _put(facts, f"{symbol}.position_qty", position.qty)
        _put(facts, f"{symbol}.position_avg_entry", position.avg_entry)
        _put(facts, f"{symbol}.position_realized_pnl", position.realized_pnl)
        _put(facts, f"{symbol}.position_fees_paid", position.fees_paid)
        mark = book.marks.get(symbol)
        if mark is not None and mark > 0:
            _put(facts, f"{symbol}.position_mark", mark)
        _put(facts, f"{symbol}.position_stop_price", position.stop_price)


def crowd_facts(report: CrowdReport) -> dict[str, float]:
    """The crowd summary the prompt's crowd section opens with, and how many stories were
    coordinated. Per-story figures are numbered by the prompt in its own display order."""
    facts: dict[str, float] = {}
    _put(facts, "crowd.items", report.items)
    _put(facts, "crowd.withheld", report.withheld)
    _put(facts, "crowd.distinct_stories", report.distinct_stories)
    _put(facts, "crowd.duplication_ratio", report.duplication_ratio)
    _put(facts, "crowd.coordinated_clusters", sum(1 for c in report.clusters if c.coordinated))
    return facts


_ANSWERED: Final = frozenset({SourceHealth.OK, SourceHealth.EMPTY})
_FAILED: Final = frozenset({SourceHealth.ERROR, SourceHealth.TIMEOUT, SourceHealth.HOLLOW})


def coverage_facts(calls: Sequence[SourceCall]) -> dict[str, float]:
    """How many source calls answered, failed or were not made, beside the coverage list."""
    facts: dict[str, float] = {}
    _put(facts, "coverage.calls_total", len(calls))
    _put(facts, "coverage.calls_answered", sum(1 for c in calls if c.health in _ANSWERED))
    _put(facts, "coverage.calls_failed", sum(1 for c in calls if c.health in _FAILED))
    _put(
        facts,
        "coverage.calls_disabled",
        sum(1 for c in calls if c.health is SourceHealth.DISABLED),
    )
    return facts


__all__ = [
    "ATR_BARS",
    "FEAR_GREED_LABELS",
    "FUNDING_SD_FLOOR",
    "MA_BARS",
    "OI_MATCH_TOLERANCE",
    "OI_MAX_AGE",
    "SOCIAL_CHANNELS",
    "SOCIAL_RATE_FLOOR",
    "STALE_INDEX_MOVES",
    "build_features",
    "coverage_facts",
    "crowd_facts",
    "facts_from",
    "fear_greed_band",
    "funding_z",
    "index_move_bps_3h",
    "ma_distance_atr",
    "mood_from",
    "oi_change_pct",
    "social_signals",
]
