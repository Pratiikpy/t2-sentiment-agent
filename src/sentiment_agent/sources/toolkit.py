"""The two Bitget services as one :class:`~sentiment_agent.types.ToolkitReader`, and the probe that
measures every surface of both for the toolkit coverage matrix.

**Which service answers each reading, and why.** The design listed bitget-signal as the source of
crowd positioning and bitget-mcp-server as its cross-check. Measured on 2026-09-24 the order has to
be the other way round:

* every bitget-signal positioning call answered ``{"error": ""}`` after 15 seconds, as it has on
  every day it has been measured since 2026-09-13, and its success shape has never been observed;
* bitget-mcp-server answered the same series (Binance futures, the upstream bitget-signal itself
  names) in under a second, with pinned field names and timestamps.

So :meth:`ToolkitFacade.derivatives` reads bitget-mcp-server first and asks bitget-signal only for
a field bitget-mcp-server left empty, at the same hourly period. Each field comes from one source
per snapshot and the returned calls say which; the open-interest *series* is taken whole from one
source, never spliced from two.

**Fear & Greed** is read from both services on every call, because the snapshot reports whether
the two crypto readings agree (``perception.features.mood_from``). The primary crypto reading is
bitget-mcp-server's, so a trigger band cannot flip merely because the slower service happened to
answer this time; bitget-signal's reading is kept beside it as ``crypto_fear_greed_alt`` when both
answered, and stands in as the primary only when bitget-mcp-server did not. Both services wrap the
same upstream (alternative.me), so agreement checks the plumbing, not a second opinion. Values are
never averaged.

Calls run concurrently; every method returns its calls in a fixed order and never raises.
"""

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import partial
from typing import Any, Final, TypeVar

from sentiment_agent.sources.bitget_data import (
    ENTRY_CRYPTO_FEAR_GREED,
    ENTRY_EARNINGS,
    ENTRY_FUNDING,
    ENTRY_INSIDER,
    ENTRY_LONG_SHORT,
    ENTRY_MARKET_FEAR_GREED,
    ENTRY_OPEN_INTEREST,
    ENTRY_TAKER,
    ENTRY_TOP_ACCOUNT,
    ENTRY_TOP_POSITION,
    NEW_YORK,
    UNDERLYING,
    USED_ENTRIES,
    BitgetDataService,
    source_label,
)
from sentiment_agent.sources.mcp_http import Invocation, source_call
from sentiment_agent.sources.signal_skills import REDDIT_FILTERS, SignalSkills
from sentiment_agent.types import (
    CalendarItem,
    Clock,
    DerivativesReading,
    MoodReading,
    SourceCall,
    SourceHealth,
    TextItem,
    ToolkitProbe,
    ToolkitSurface,
    ToolkitUse,
)

T = TypeVar("T")

SIGNAL_FALLBACK_PERIOD: Final = "1h"
"""bitget-signal is asked for the hourly bucket bitget-mcp-server is read at, so a fallback value
describes the same bucket length as the primary."""

SIGNAL_FEAR_GREED_SOURCE: Final = f"{ToolkitSurface.SIGNAL_MCP.value}:sentiment_index.current"

_DATA = ToolkitSurface.DATA_MCP
_SIGNAL = ToolkitSurface.SIGNAL_MCP


def _failed_call(surface: ToolkitSurface, source: str, clock: Clock, reason: str) -> SourceCall:
    """The call record for a reader that raised (the readers do not; this is the last line)."""
    invocation = Invocation(clock.now(), 0, None, SourceHealth.ERROR, reason)
    return source_call(
        surface=surface,
        source=source,
        params={},
        invocation=invocation,
        health=SourceHealth.ERROR,
        error=f"reader failed: {reason}",
    )


def _guarded(fn: Callable[[], T], fallback: Callable[[str], T]) -> Callable[[], T]:
    """``fn`` with any exception turned into ``fallback(reason)``."""

    def run() -> T:
        try:
            return fn()
        except Exception as exc:
            return fallback(f"{type(exc).__name__}: {exc}")

    return run


FearGreed = tuple[int | None, str | None, SourceCall]
Ratio = tuple[float | None, SourceCall]
Series = tuple[tuple[tuple[datetime, float], ...], SourceCall]
Texts = tuple[tuple[TextItem, ...], SourceCall]
Calendar = tuple[tuple[CalendarItem, ...], SourceCall]


class ToolkitFacade:
    """bitget-signal and bitget-mcp-server behind the
    :class:`~sentiment_agent.types.ToolkitReader` protocol."""

    def __init__(
        self, signal: SignalSkills, data: BitgetDataService, *, max_workers: int = 8
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        self._signal = signal
        self._data = data
        self._clock = data.clock
        self._workers = max_workers

    @property
    def signal(self) -> SignalSkills:
        return self._signal

    @property
    def data(self) -> BitgetDataService:
        return self._data

    def _fear_greed_fallback(
        self, surface: ToolkitSurface, source: str
    ) -> Callable[[str], FearGreed]:
        clock = self._clock
        return lambda reason: (None, None, _failed_call(surface, source, clock, reason))

    def _ratio_fallback(self, source: str) -> Callable[[str], Ratio]:
        clock = self._clock
        return lambda reason: (None, _failed_call(_SIGNAL, source, clock, reason))

    def _texts_fallback(self, source: str) -> Callable[[str], Texts]:
        clock = self._clock
        return lambda reason: ((), _failed_call(_SIGNAL, source, clock, reason))

    def _calendar_fallback(self, source: str) -> Callable[[str], Calendar]:
        clock = self._clock
        return lambda reason: ((), _failed_call(_DATA, source, clock, reason))

    # --- ToolkitReader --------------------------------------------------------------------------

    def mood(self) -> tuple[MoodReading, tuple[SourceCall, ...]]:
        """Fear & Greed from both services. Calls: bitget-mcp-server crypto, bitget-signal crypto,
        bitget-mcp-server equity market."""
        data_crypto = _guarded(
            self._data.crypto_fear_greed,
            self._fear_greed_fallback(_DATA, f"do_query:{ENTRY_CRYPTO_FEAR_GREED}"),
        )
        signal_crypto = _guarded(
            self._signal.fear_greed,
            self._fear_greed_fallback(_SIGNAL, "sentiment_index.current"),
        )
        market = _guarded(
            self._data.market_fear_greed,
            self._fear_greed_fallback(_DATA, f"do_query:{ENTRY_MARKET_FEAR_GREED}"),
        )
        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            f_data, f_signal, f_market = (
                pool.submit(fn) for fn in (data_crypto, signal_crypto, market)
            )
            data_value, data_label, data_call = f_data.result()
            signal_value, signal_label, signal_call = f_signal.result()
            market_value, market_label, market_call = f_market.result()

        primary_value: int | None = None
        primary_label: str | None = None
        primary_source: str | None = None
        alt_value: int | None = None
        alt_source: str | None = None
        if data_value is not None:
            primary_value, primary_label = data_value, data_label
            primary_source = source_label(ENTRY_CRYPTO_FEAR_GREED)
            if signal_value is not None:
                alt_value, alt_source = signal_value, SIGNAL_FEAR_GREED_SOURCE
        elif signal_value is not None:
            primary_value, primary_label = signal_value, signal_label
            primary_source = SIGNAL_FEAR_GREED_SOURCE
        reading = MoodReading(
            crypto_fear_greed=primary_value,
            crypto_fear_greed_label=primary_label,
            crypto_fear_greed_source=primary_source,
            crypto_fear_greed_alt=alt_value,
            crypto_fear_greed_alt_source=alt_source,
            market_fear_greed=market_value,
            market_fear_greed_label=market_label if market_value is not None else None,
            market_fear_greed_source=(
                source_label(ENTRY_MARKET_FEAR_GREED) if market_value is not None else None
            ),
        )
        return reading, (data_call, signal_call, market_call)

    def derivatives(self, symbol: str) -> tuple[DerivativesReading, tuple[SourceCall, ...]]:
        """Crowd positioning. Calls: the six bitget-mcp-server entries, then a bitget-signal call
        for each field those left empty (long/short, top long/short, taker, open interest)."""
        try:
            primary, calls = self._data.derivatives(symbol)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            primary = DerivativesReading(
                symbol=symbol,
                retail_long_short_ratio=None,
                top_trader_account_ratio=None,
                top_trader_position_ratio=None,
                taker_buy_sell_ratio=None,
            )
            calls = (_failed_call(_DATA, "do_query:derivatives", self._clock, reason),)

        period = SIGNAL_FALLBACK_PERIOD
        ratios: dict[str, Callable[[], Ratio]] = {}
        if primary.retail_long_short_ratio is None:
            ratios["derivatives_sentiment.long_short"] = _guarded(
                lambda: self._signal.long_short(symbol, period),
                self._ratio_fallback("derivatives_sentiment.long_short"),
            )
        if primary.top_trader_account_ratio is None:
            ratios["derivatives_sentiment.top_ls"] = _guarded(
                lambda: self._signal.top_long_short(symbol, period),
                self._ratio_fallback("derivatives_sentiment.top_ls"),
            )
        if primary.taker_buy_sell_ratio is None:
            ratios["derivatives_sentiment.taker_ratio"] = _guarded(
                lambda: self._signal.taker_ratio(symbol, period),
                self._ratio_fallback("derivatives_sentiment.taker_ratio"),
            )
        series_fn: Callable[[], Series] | None = None
        if not primary.open_interest_history:
            clock = self._clock
            series_fn = _guarded(
                lambda: self._signal.open_interest(symbol, period),
                lambda reason: (
                    (),
                    _failed_call(_SIGNAL, "derivatives_sentiment.open_interest", clock, reason),
                ),
            )
        if not ratios and series_fn is None:
            return primary, calls

        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            ratio_futures: dict[str, Future[Ratio]] = {
                source: pool.submit(fn) for source, fn in ratios.items()
            }
            series_future = pool.submit(series_fn) if series_fn is not None else None
            ratio_results = {source: f.result() for source, f in ratio_futures.items()}
            series_result = series_future.result() if series_future is not None else None

        def pick(current: float | None, source: str) -> float | None:
            return ratio_results[source][0] if source in ratio_results else current

        reading = DerivativesReading(
            symbol=primary.symbol,
            retail_long_short_ratio=pick(
                primary.retail_long_short_ratio, "derivatives_sentiment.long_short"
            ),
            top_trader_account_ratio=pick(
                primary.top_trader_account_ratio, "derivatives_sentiment.top_ls"
            ),
            top_trader_position_ratio=primary.top_trader_position_ratio,
            taker_buy_sell_ratio=pick(
                primary.taker_buy_sell_ratio, "derivatives_sentiment.taker_ratio"
            ),
            open_interest_history=(
                series_result[0] if series_result is not None else primary.open_interest_history
            ),
            funding_rate=primary.funding_rate,
        )
        extra = [ratio_results[s][1] for s in ratios]
        if series_result is not None:
            extra.append(series_result[1])
        return reading, (*calls, *extra)

    def news(self, limit: int) -> tuple[tuple[TextItem, ...], tuple[SourceCall, ...]]:
        """Headlines from bitget-signal's ``news_feed``. bitget-mcp-server's one news entry needs a
        label vocabulary its catalog does not publish, so it is not read."""
        items, call = _guarded(
            lambda: self._signal.news(limit), self._texts_fallback("news_feed.latest")
        )()
        return items, (call,)

    def reddit_trending(self, limit: int) -> tuple[tuple[TextItem, ...], tuple[SourceCall, ...]]:
        """Up to ``limit`` trending tickers from each of
        :data:`~sentiment_agent.sources.signal_skills.REDDIT_FILTERS`, one call per filter."""
        source = "derivatives_sentiment.reddit_trending"
        jobs = [
            _guarded(
                partial(self._signal.reddit_trending, limit, subreddit_filter=name),
                self._texts_fallback(source),
            )
            for name in REDDIT_FILTERS
        ]
        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            outcomes = [f.result() for f in [pool.submit(job) for job in jobs]]
        items: dict[str, TextItem] = {}
        for found, _ in outcomes:
            for item in found:
                items.setdefault(item.item_id, item)
        return tuple(items.values()), tuple(call for _, call in outcomes)

    def calendar(
        self, symbols: Sequence[str], *, since: datetime
    ) -> tuple[tuple[CalendarItem, ...], tuple[SourceCall, ...]]:
        """Earnings dated on or after ``since``'s New York date, and insider filings filed on or
        after it, for every equity perp in ``symbols``; other symbols have neither and cost no
        call. Calls: every earnings call in symbol order, then every filings call."""
        if since.tzinfo is None:
            raise ValueError("since must be timezone-aware")
        equities = [s for s in dict.fromkeys(symbols) if s in UNDERLYING]
        if not equities:
            return (), ()
        earnings_jobs = [
            _guarded(
                partial(self._data.earnings, s),
                self._calendar_fallback(f"do_query:{ENTRY_EARNINGS}"),
            )
            for s in equities
        ]
        filing_jobs = [
            _guarded(
                partial(self._data.insider_filings, s, since=since),
                self._calendar_fallback(f"do_query:{ENTRY_INSIDER}"),
            )
            for s in equities
        ]
        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            earnings = [f.result() for f in [pool.submit(job) for job in earnings_jobs]]
            filings = [f.result() for f in [pool.submit(job) for job in filing_jobs]]
        first_day = since.astimezone(NEW_YORK).date()
        items: list[CalendarItem] = []
        for found, _ in earnings:
            items.extend(
                i
                for i in found
                if i.at is not None and i.at.astimezone(NEW_YORK).date() >= first_day
            )
        for found, _ in filings:
            items.extend(found)
        ordered = sorted(items, key=lambda i: (i.at or since, i.symbol or "", i.kind, i.title))
        calls = tuple(call for _, call in earnings) + tuple(call for _, call in filings)
        return tuple(ordered), calls


# ================================================================================================
# Probe: the measured column of the toolkit coverage matrix
# ================================================================================================


@dataclass(frozen=True, slots=True)
class Declared:
    """What a used source is for, where it is read, and which judged line it serves."""

    purpose: str
    judged_line: str
    used_in: tuple[str, ...]
    visible_at: str


PERCEPTION_LINE: Final = (
    "Agent architecture quality: the Track 2 perception layer the handbook names "
    "(bitget-signal + bitget-mcp-server)"
)
EXPLAIN_LINE: Final = (
    "decision explainability: the value is shown to the model and on the decision card"
)
TRIGGER_LINE: Final = "decision explainability: an event trigger fires on it and says why"
MATRIX: Final = "toolkit coverage matrix"
CARD_POSITIONING: Final = "decision card: positioning table"
CARD_TEXT: Final = "decision card: text shown to the model"
CARD_CALENDAR: Final = "decision card: calendar"
CARD_MOOD: Final = "decision card: mood and source agreement"

_FEATURES = ("perception.features.build_features",)
_FALLBACK = ("sources.toolkit.ToolkitFacade.derivatives",)
_TEXT = ("crowd.quarantine", "crowd.novelty", "perception.snapshot")
_MOOD = ("perception.features.mood_from", "events.triggers")

USED: Final[Mapping[tuple[ToolkitSurface, str], Declared]] = {
    (_DATA, f"do_query:{ENTRY_CRYPTO_FEAR_GREED}"): Declared(
        "crypto Fear & Greed (primary)", TRIGGER_LINE, _MOOD, CARD_MOOD
    ),
    (_DATA, f"do_query:{ENTRY_MARKET_FEAR_GREED}"): Declared(
        "equity-market Fear & Greed", TRIGGER_LINE, _MOOD, CARD_MOOD
    ),
    (_SIGNAL, "sentiment_index.current"): Declared(
        "crypto Fear & Greed from the second service: agreement check, stand-in for the primary",
        EXPLAIN_LINE,
        ("perception.features.mood_from",),
        CARD_MOOD,
    ),
    (_DATA, f"do_query:{ENTRY_LONG_SHORT}"): Declared(
        "retail long/short account ratio (Binance futures crowd)",
        EXPLAIN_LINE,
        _FEATURES,
        CARD_POSITIONING,
    ),
    (_DATA, f"do_query:{ENTRY_TOP_ACCOUNT}"): Declared(
        "top-trader long/short account ratio", EXPLAIN_LINE, _FEATURES, CARD_POSITIONING
    ),
    (_DATA, f"do_query:{ENTRY_TOP_POSITION}"): Declared(
        "top-trader long/short position ratio", EXPLAIN_LINE, _FEATURES, CARD_POSITIONING
    ),
    (_DATA, f"do_query:{ENTRY_TAKER}"): Declared(
        "taker buy/sell volume ratio", EXPLAIN_LINE, _FEATURES, CARD_POSITIONING
    ),
    (_DATA, f"do_query:{ENTRY_OPEN_INTEREST}"): Declared(
        "hourly open-interest history: 1h and 24h change, the jump trigger",
        TRIGGER_LINE,
        ("perception.features.oi_change_pct", "events.triggers"),
        CARD_POSITIONING,
    ),
    (_DATA, f"do_query:{ENTRY_FUNDING}"): Declared(
        "Binance funding rate (reported in percent, converted to a fraction)",
        EXPLAIN_LINE,
        _FEATURES,
        CARD_POSITIONING,
    ),
    (_SIGNAL, "derivatives_sentiment.long_short"): Declared(
        "retail long/short when bitget-mcp-server has none", PERCEPTION_LINE, _FALLBACK, MATRIX
    ),
    (_SIGNAL, "derivatives_sentiment.top_ls"): Declared(
        "top-trader long/short when bitget-mcp-server has none", PERCEPTION_LINE, _FALLBACK, MATRIX
    ),
    (_SIGNAL, "derivatives_sentiment.taker_ratio"): Declared(
        "taker ratio when bitget-mcp-server has none", PERCEPTION_LINE, _FALLBACK, MATRIX
    ),
    (_SIGNAL, "derivatives_sentiment.open_interest"): Declared(
        "open-interest series when bitget-mcp-server has none (taken whole, never spliced)",
        PERCEPTION_LINE,
        _FALLBACK,
        MATRIX,
    ),
    (_SIGNAL, "derivatives_sentiment.reddit_trending"): Declared(
        "Reddit attention ranking (crypto and stocks), screened as crowd text",
        EXPLAIN_LINE,
        _TEXT,
        CARD_TEXT,
    ),
    (_SIGNAL, "news_feed.latest"): Declared(
        "news headlines, screened as crowd text", EXPLAIN_LINE, _TEXT, CARD_TEXT
    ),
    (_DATA, f"do_query:{ENTRY_EARNINGS}"): Declared(
        "earnings dates of the eleven underlying stocks (vendor date offset corrected)",
        TRIGGER_LINE,
        ("perception.features.build_features", "events.triggers"),
        CARD_CALENDAR,
    ),
    (_DATA, f"do_query:{ENTRY_INSIDER}"): Declared(
        "insider (Form 4) filings of the eleven underlying stocks",
        TRIGGER_LINE,
        ("events.triggers",),
        CARD_CALENDAR,
    ),
}
"""Every source the agent reads, keyed ``(surface, SourceCall.source)``."""

_OWN_PRICES: Final = "not used: prices come from Bitget's own public v3 API, the venue orders go to"
_NOT_REGISTERED: Final = "not used: not a pre-registered input of policy v1"

SIGNAL_TOOL_PROBES: Final[tuple[tuple[str, Mapping[str, Any], str], ...]] = (
    (
        "technical_analysis",
        {"action": "rsi", "symbol": "BTC/USDT"},
        "not used: indicators are computed from Bitget's own candles (ARGUS measured this tool's "
        "MACD signal and histogram fields swapped)",
    ),
    ("crypto_market", {"action": "global"}, _OWN_PRICES),
    ("crypto_price", {"action": "price", "symbol": "BTC"}, _OWN_PRICES),
    ("crypto_derivatives", {"action": "ticker_24h", "symbol": "BTC/USDT"}, _OWN_PRICES),
    ("global_assets", {"action": "price", "symbol": "NVDA"}, _OWN_PRICES),
    ("macro_indicators", {"action": "latest_release", "indicator": "cpi"}, _NOT_REGISTERED),
    ("rates_yields", {"action": "yield_curve"}, _NOT_REGISTERED),
    ("cross_asset", {"action": "correlation"}, _NOT_REGISTERED),
    (
        "tradfi_news",
        {"action": "earnings"},
        "not used: its description says it needs a Finnhub key; earnings come from "
        "bitget-mcp-server",
    ),
    (
        "social_trending",
        {"action": "trending", "platform": "xueqiu", "limit": 5},
        "not used: Chinese hot lists name no universe instrument reliably",
    ),
    ("defi_analytics", {"action": "tvl_rank", "limit": 5}, "not used: no DeFi instrument traded"),
    ("dex_market", {"action": "trending"}, "not used: no DEX instrument traded"),
    ("network_status", {"action": "btc_fees"}, _NOT_REGISTERED),
    ("global_data", {"action": "forex", "symbols": "EUR"}, _NOT_REGISTERED),
    (
        "cn_market",
        {"action": "index", "symbol": "sh000001"},
        "not used: no China-listed instrument",
    ),
    ("backtest", {"action": "chart"}, "not used: a live paper run does not backtest"),
    (
        "sentiment_index",
        {"action": "history", "days": 7},
        "not used: the current value is read each snapshot and logged, which is its history",
    ),
    (
        "derivatives_sentiment",
        {"action": "top_position", "symbol": "BTCUSDT", "period": "1h"},
        "not used: bitget-mcp-server's top-position entry is read instead",
    ),
)
"""One cheap call per bitget-signal tool or action the agent does not use, so the matrix says how
each behaves, not only that it exists. Arguments follow each tool's ``tools/list`` schema."""

PROBE_ARGS_BY_CATEGORY: Final[Mapping[str, Mapping[str, str]]] = {
    "crypto": {"symbol": "BTCUSDT"},
    "equity": {"symbol": "NVDA"},
    "etf": {"symbol": "QQQ"},
}
"""Arguments for measuring an unused catalog entry, per category (as ARGUS
``data/data_coverage.json`` did). Entries whose required parameters these do not cover answer with
the service's own "missing param" error, which is recorded as measured."""

_UNUSED_REASONS: Final[tuple[tuple[str, str], ...]] = (
    ("crypto_futures_open_interest", "not used: the hourly history entry carries the same value"),
    ("crypto_futures_liquidations", _NOT_REGISTERED),
    ("crypto_futures_big_trades", _NOT_REGISTERED),
    ("crypto_spot_", _OWN_PRICES),
    ("crypto_futures_", _OWN_PRICES),
    ("crypto_market", _OWN_PRICES),
    ("crypto_coin_info", "not used: reference data, not sentiment"),
    ("crypto_", _NOT_REGISTERED),
    ("equity_", "not used: fundamentals, estimates and holdings; the agent's horizon is hours"),
    ("etf_", "not used: no ETF is traded"),
    ("news_label_search", "not used: needs an integer label the catalog does not define"),
)

PROBE_TEXT_LIMIT: Final = 10
PROBE_CALENDAR_LOOKBACK: Final = timedelta(days=30)

_SEVERITY: Final[Mapping[SourceHealth, int]] = {
    SourceHealth.TIMEOUT: 5,
    SourceHealth.ERROR: 4,
    SourceHealth.HOLLOW: 3,
    SourceHealth.EMPTY: 2,
    SourceHealth.DISABLED: 1,
    SourceHealth.OK: 0,
}


def unused_reason(entry_id: str) -> str:
    for prefix, reason in _UNUSED_REASONS:
        if entry_id.startswith(prefix):
            return reason
    return "not used by policy v1"


def aggregate_health(calls: Sequence[SourceCall]) -> tuple[SourceHealth, str]:
    """One health for several calls to one source (one per symbol, say): ``OK`` when any call
    answered with substance and none failed in transport; otherwise the most severe outcome seen,
    so a partial failure is never shown as green."""
    if not calls:
        return SourceHealth.DISABLED, "not called"
    counts = Counter(call.health for call in calls)
    failures = counts[SourceHealth.ERROR] + counts[SourceHealth.TIMEOUT]
    if counts[SourceHealth.OK] and not failures:
        health = SourceHealth.OK
    else:
        health = max(counts, key=lambda h: _SEVERITY[h])
    ordered = sorted(counts.items(), key=lambda kv: (-_SEVERITY[kv[0]], kv[0].value))
    detail = f"{len(calls)} call(s): " + ", ".join(f"{h.value} {n}" for h, n in ordered)
    not_ok = [c for c in calls if c.health is not SourceHealth.OK]
    if not_ok:
        labels = [
            f"{c.params.get('symbol') or c.params.get('filter') or '-'}={c.health.value}"
            for c in not_ok
        ]
        detail += "; not ok: " + ", ".join(labels[:16])
        example = next((c.error for c in not_ok if c.error), None)
        if example:
            detail += f"; e.g. {example[:160]}"
    return health, detail


def _unused_row(call: SourceCall, reason: str, at: datetime) -> ToolkitUse:
    measured = (
        f"measured {call.health.value}: {call.error[:200]}"
        if call.error
        else f"measured {call.health.value} ({call.rows} rows)"
    )
    return ToolkitUse(
        surface=call.surface,
        entry=call.source,
        purpose="measured, not used",
        judged_line="",
        used_in=(),
        last_health=call.health,
        last_checked_at=at,
        visible_at=MATRIX,
        notes=f"{reason}; {measured}",
    )


def _catalog_row(health: SourceHealth, at: datetime, notes: str) -> ToolkitUse:
    return ToolkitUse(
        surface=_DATA,
        entry="guide",
        purpose="the catalog of every bitget-mcp-server entry",
        judged_line=PERCEPTION_LINE,
        used_in=("sources.toolkit.probe_all",),
        last_health=health,
        last_checked_at=at,
        visible_at=MATRIX,
        notes=notes[:500],
    )


def probe_all(facade: ToolkitFacade, *, universe: Sequence[str], clock: Clock) -> ToolkitProbe:
    """Measure every bitget-signal tool and every bitget-mcp-server catalog entry, once.

    Used sources are measured through the same typed readers the agent runs, so a parser defect
    shows here exactly as it would in a snapshot: mood; positioning for every universe symbol from
    both services (bitget-signal directly too, since the facade only asks it for missing fields);
    news; Reddit; and the calendar for the universe's equity perps. Every other tool and catalog
    entry is called once with arguments from its schema and listed as measured-not-used with the
    reason. Never raises: a failed catalog becomes a row that says so.
    """
    at = clock.now()
    symbols = list(dict.fromkeys(universe))
    grouped: dict[tuple[ToolkitSurface, str], list[SourceCall]] = defaultdict(list)

    def keep(calls: Sequence[SourceCall]) -> None:
        for call in calls:
            grouped[(call.surface, call.source)].append(call)

    signal = facade.signal
    data = facade.data
    period = SIGNAL_FALLBACK_PERIOD
    ratio_methods = (signal.long_short, signal.top_long_short, signal.taker_ratio)
    with ThreadPoolExecutor(max_workers=8) as pool:
        mood = pool.submit(facade.mood)
        positioning = [pool.submit(data.derivatives, s) for s in symbols]
        signal_ratios = [pool.submit(m, s, period) for s in symbols for m in ratio_methods]
        signal_series = [pool.submit(signal.open_interest, s, period) for s in symbols]
        news = pool.submit(facade.news, PROBE_TEXT_LIMIT)
        reddit = pool.submit(facade.reddit_trending, PROBE_TEXT_LIMIT)
        calendar = pool.submit(facade.calendar, symbols, since=at - PROBE_CALENDAR_LOOKBACK)
        unused_signal = [
            (pool.submit(signal.probe, tool, args), reason)
            for tool, args, reason in SIGNAL_TOOL_PROBES
        ]
        keep(mood.result()[1])
        for f_positioning in positioning:
            keep(f_positioning.result()[1])
        keep([f.result()[1] for f in signal_ratios])
        keep([f.result()[1] for f in signal_series])
        keep(news.result()[1])
        keep(reddit.result()[1])
        keep(calendar.result()[1])
        unused_signal_calls = [(f.result(), reason) for f, reason in unused_signal]

    rows: list[ToolkitUse] = []
    for (surface, source), declared in USED.items():
        health, detail = aggregate_health(grouped.get((surface, source), []))
        rows.append(
            ToolkitUse(
                surface=surface,
                entry=source,
                purpose=declared.purpose,
                judged_line=declared.judged_line,
                used_in=declared.used_in,
                last_health=health,
                last_checked_at=at,
                visible_at=declared.visible_at,
                notes=detail,
            )
        )
    rows.extend(_unused_row(call, reason, at) for call, reason in unused_signal_calls)

    try:
        catalog = data.catalog()
    except Exception as exc:
        reason = f"catalog failed; unused entries not measured: {type(exc).__name__}: {exc}"
        rows.append(_catalog_row(SourceHealth.ERROR, at, reason))
        return ToolkitProbe(at=at, rows=tuple(rows))

    categories = Counter(category for _, category in catalog)
    summary = ", ".join(f"{name} {count}" for name, count in sorted(categories.items()))
    rows.append(_catalog_row(SourceHealth.OK, at, f"{len(catalog)} entries: {summary}"))
    listed = {entry for entry, _ in catalog}
    missing = [entry for entry in USED_ENTRIES if entry not in listed]
    if missing:
        rows.append(
            ToolkitUse(
                surface=_DATA,
                entry="guide:missing-used-entries",
                purpose="entries the agent reads that the live catalog no longer lists",
                judged_line=PERCEPTION_LINE,
                used_in=("sources.bitget_data",),
                last_health=SourceHealth.ERROR,
                last_checked_at=at,
                visible_at=MATRIX,
                notes="missing from the catalog: " + ", ".join(missing),
            )
        )
    used = set(USED_ENTRIES)
    unused = [(entry, category) for entry, category in catalog if entry not in used]
    with ThreadPoolExecutor(max_workers=8) as pool:
        measured = [
            (pool.submit(data.query, entry, **PROBE_ARGS_BY_CATEGORY.get(category, {})), entry)
            for entry, category in unused
        ]
        for future, entry in measured:
            rows.append(_unused_row(future.result()[1], unused_reason(entry), at))
    return ToolkitProbe(at=at, rows=tuple(rows))


__all__ = [
    "EXPLAIN_LINE",
    "PERCEPTION_LINE",
    "PROBE_ARGS_BY_CATEGORY",
    "SIGNAL_FALLBACK_PERIOD",
    "SIGNAL_TOOL_PROBES",
    "TRIGGER_LINE",
    "USED",
    "Declared",
    "ToolkitFacade",
    "aggregate_health",
    "probe_all",
    "unused_reason",
]
