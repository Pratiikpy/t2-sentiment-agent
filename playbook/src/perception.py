"""Perception through ``getagent.data``: prices, positioning, mood, crowd counts and the calendar.

Every endpoint, parameter and response field used here is the one documented in the getagent
skill's DataSDK reference (``references/sdk/data/crypto.md``, ``sentiment.md``, ``equity.md``);
``provider=`` is never passed (the upload validator refuses it). Each call is timed, its health
recorded, and a failure degrades the snapshot instead of crashing the run: a missing input is a
``None`` fact, which the kernel treats as not evaluated, so it can only ever refuse an increase.

The live/Demo rule of the primary (DESIGN.md §6.1) holds here too: positioning (funding, open
interest, long/short ratios) is read from live Bitget through the data layer, never from a paper
venue's configuration. The feature formulas are the primary's (``perception/features.py``), with
their constants generated into ``policy_v1``: the 20-bar moving average in 14-bar ATR units, the
mean absolute 1-hour move over three hours, the funding z-score against exactly the policy's number
of settlements (population standard deviation, a spread floor below one quoted tick), and open
interest changes paired to the reading nearest one hour (or a day) earlier.

What the replica cannot see is stated, not approximated: the Demo index (G1 reads the data layer's
price series instead), X and Reddit post text (forum mention counts only), and bitget-signal and
bitget-mcp-server, which are not reachable from the sandbox.
"""

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from functools import partial
from itertools import pairwise

from getagent import data

from . import policy_v1 as policy
from .crowd import CrowdReading, NewsItem, read_crowd
from .kernel import Quote
from .sessions import HOUR, as_utc, iso_z, parse_iso, us_open_utc

KLINE_BARS = (
    max(
        policy.FEATURE_MA_BARS,
        policy.FEATURE_ATR_BARS + 1,
        25,
        policy.FEATURE_STALE_INDEX_MOVES + 1,
    )
    + 5
)
"""1H bars fetched per instrument: enough for the 24h change, the MA and ATR windows and the stale
check, with a few to spare for a gap."""

NEWS_PER_SOURCE = 5
"""Articles per source from ``sentiment.news(category="all")`` (44 sources)."""

BTC = "BTCUSDT"


# ------------------------------------------------------------------------------------------------
# Time budget
# ------------------------------------------------------------------------------------------------


@dataclass
class Deadline:
    """The run's wall clock. The runner kills a run at 180 s (``sandbox-runtime.md``)."""

    started: datetime
    clock: Callable[[], datetime]
    budget_seconds: float = 180.0

    def elapsed(self) -> float:
        return (as_utc(self.clock()) - as_utc(self.started)).total_seconds()

    def before(self, seconds: float) -> bool:
        """Whether the run is still before ``seconds`` of its budget."""
        return self.elapsed() < seconds


# ------------------------------------------------------------------------------------------------
# Parsing helpers
# ------------------------------------------------------------------------------------------------


def number(value: object) -> float | None:
    """A finite float from a number or a numeric string; ``None`` for anything else."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float, Decimal)):
        result = float(value)
    elif isinstance(value, str):
        try:
            result = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    return result if math.isfinite(result) else None


def decimal(value: object) -> Decimal | None:
    """A finite positive-or-zero-or-negative Decimal from a number or numeric string."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None
    return result if result.is_finite() else None


def timestamp(value: object) -> datetime | None:
    """A UTC instant from epoch seconds or milliseconds, an ISO string or a datetime.

    A naive datetime or ISO string from the data layer is read as UTC (the SDK documents UTC).
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    if isinstance(value, (int, float)):
        seconds = float(value)
        if not math.isfinite(seconds) or seconds <= 0:
            return None
        if seconds > 1e14:
            seconds /= 1e6
        elif seconds > 1e11:
            seconds /= 1e3
        return datetime.fromtimestamp(seconds, tz=UTC)
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            return timestamp(int(text))
        parsed = parse_iso(text)
        if parsed is not None:
            return parsed
        try:
            naive = datetime.fromisoformat(text)
        except ValueError:
            return None
        return naive.replace(tzinfo=UTC) if naive.tzinfo is None else naive.astimezone(UTC)
    to_py = getattr(value, "to_pydatetime", None)
    if callable(to_py):
        converted = to_py()
        return timestamp(converted) if isinstance(converted, datetime) else None
    return None


def row_time(row: Mapping[str, object], *keys: str) -> datetime | None:
    for key in keys:
        found = timestamp(row.get(key))
        if found is not None:
            return found
    return None


# ------------------------------------------------------------------------------------------------
# The snapshot
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceCall:
    name: str
    health: str
    rows: int
    seconds: float
    detail: str = ""

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "health": self.health,
            "rows": self.rows,
            "seconds": round(self.seconds, 3),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class Candle:
    open_time: datetime
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class CalendarEntry:
    symbol: str
    at: datetime
    report_date: str
    timing: str


@dataclass(frozen=True)
class FilingEntry:
    symbol: str
    filed: str
    key: str


@dataclass
class Snapshot:
    taken_at: datetime
    quotes: dict[str, Quote] = field(default_factory=dict)
    funding_rate: dict[str, float] = field(default_factory=dict)
    candles: dict[str, list[Candle]] = field(default_factory=dict)
    features: dict[str, dict[str, float]] = field(default_factory=dict)
    mood: dict[str, float] = field(default_factory=dict)
    funding_history: list[tuple[datetime, float]] = field(default_factory=list)
    oi_series: list[tuple[datetime, float]] = field(default_factory=list)
    oi_jump_threshold_pct: float | None = None
    crowd: CrowdReading | None = None
    earnings: list[CalendarEntry] = field(default_factory=list)
    filings: list[FilingEntry] = field(default_factory=list)
    sources: list[SourceCall] = field(default_factory=list)

    def feature(self, symbol: str, name: str) -> float | None:
        return self.features.get(symbol, {}).get(name)

    def move_bps_3h(self) -> dict[str, float | None]:
        return {s: f.get("move_bps_3h") for s, f in self.features.items()}

    def coverage(self) -> list[dict[str, object]]:
        return [s.to_json() for s in self.sources]


# ------------------------------------------------------------------------------------------------
# Feature math (the primary's formulas)
# ------------------------------------------------------------------------------------------------


def closed_candles(candles: Iterable[Candle], now: datetime) -> list[Candle]:
    """Completed 1H bars in time order, one per open time."""
    by_time: dict[datetime, Candle] = {}
    for candle in candles:
        if candle.open_time + HOUR <= now:
            by_time[candle.open_time] = candle
    return [by_time[t] for t in sorted(by_time)]


def fresh(candles: Sequence[Candle], now: datetime) -> bool:
    """The newest closed bar is recent: refuse to act on a stalled feed (the skill's rule,
    ``now - (last_open + interval) > 2 x interval``)."""
    if not candles:
        return False
    return now - (candles[-1].open_time + HOUR) <= 2 * HOUR


def ma_distance_atr(candles: Sequence[Candle]) -> float | None:
    """``(close - SMA(close, 20)) / ATR(14)`` on the newest bar (``features.ma_distance_atr``)."""
    ma, atr = policy.FEATURE_MA_BARS, policy.FEATURE_ATR_BARS
    if len(candles) < max(ma, atr + 1):
        return None
    closes = [c.close for c in candles[-max(ma, atr + 1) :]]
    if any(not math.isfinite(c) or c <= 0 for c in closes):
        return None
    sma = math.fsum(closes[-ma:]) / ma
    tail = candles[-(atr + 1) :]
    ranges = [
        max(cur.high - cur.low, abs(cur.high - prev.close), abs(cur.low - prev.close))
        for prev, cur in pairwise(tail)
    ]
    average = math.fsum(ranges) / atr
    if not average > 0:
        return None
    value = (closes[-1] - sma) / average
    return value if math.isfinite(value) else None


def move_bps_3h(candles: Sequence[Candle]) -> float | None:
    """Mean ``|close_t / close_{t-1} - 1|`` in bps over the three newest hourly moves, the four
    newest bars exactly one hour apart (``features.index_move_bps_3h``)."""
    count = policy.FEATURE_STALE_INDEX_MOVES
    if len(candles) < count + 1:
        return None
    window = candles[-(count + 1) :]
    moves: list[float] = []
    for prev, cur in pairwise(window):
        if cur.open_time - prev.open_time != HOUR or prev.close <= 0:
            return None
        moves.append(abs(cur.close / prev.close - 1.0) * 10_000.0)
    value = math.fsum(moves) / len(moves)
    return value if math.isfinite(value) else None


def change_24h_pct(candles: Sequence[Candle]) -> float | None:
    if not candles:
        return None
    newest = candles[-1]
    target = newest.open_time - timedelta(hours=24)
    for candle in candles:
        if candle.open_time == target and candle.close > 0:
            return (newest.close / candle.close - 1.0) * 100.0
    return None


def funding_z(history: Sequence[tuple[datetime, float]], current: float | None) -> float | None:
    """Z-score of ``current`` against exactly the last ``lookback`` settlements
    (``features.funding_z``): ``None`` with fewer, or with a spread under the floor."""
    lookback = policy.TRIGGER_FUNDING_Z_LOOKBACK_SETTLEMENTS
    if current is None or not math.isfinite(current):
        return None
    by_time = dict(history)
    ordered = [by_time[t] for t in sorted(by_time)]
    if len(ordered) < lookback:
        return None
    window = ordered[-lookback:]
    if not all(math.isfinite(x) for x in window):
        return None
    mean = math.fsum(window) / lookback
    sd = math.sqrt(math.fsum((x - mean) ** 2 for x in window) / lookback)
    if not sd >= policy.FEATURE_FUNDING_SD_FLOOR:
        return None
    value = (current - mean) / sd
    return value if math.isfinite(value) else None


def oi_change_pct(
    series: Sequence[tuple[datetime, float]], *, hours: int, at: datetime
) -> float | None:
    """Percent change in open interest over ``hours`` (``features.oi_change_pct``)."""
    by_time = {t: v for t, v in series if t <= at and math.isfinite(v) and v > 0}
    if len(by_time) < 2:
        return None
    stamps = sorted(by_time)
    newest = stamps[-1]
    if at - newest > timedelta(minutes=policy.FEATURE_OI_MAX_AGE_MINUTES):
        return None
    target = newest - timedelta(hours=hours)
    reference = min(stamps[:-1], key=lambda t: (abs(t - target), t))
    if abs(reference - target) > timedelta(minutes=policy.FEATURE_OI_MATCH_TOLERANCE_MINUTES):
        return None
    value = (by_time[newest] / by_time[reference] - 1.0) * 100.0
    return value if math.isfinite(value) else None


def hourly_oi_changes_pct(series: Sequence[tuple[datetime, float]]) -> list[float]:
    """Every 1-hour change, paired to the reading nearest one hour earlier within the tolerance
    (``events.triggers.hourly_oi_changes_pct``)."""
    points = {t: v for t, v in series if math.isfinite(v) and v > 0}
    times = sorted(points)
    tolerance = timedelta(minutes=policy.FEATURE_OI_MATCH_TOLERANCE_MINUTES)
    changes: list[float] = []
    for i, at in enumerate(times):
        target = at - HOUR
        earlier = [k for k in range(i) if abs(times[k] - target) <= tolerance]
        if not earlier:
            continue
        nearest = min(earlier, key=lambda k: (abs(times[k] - target), times[k]))
        changes.append(100.0 * (points[at] / points[times[nearest]] - 1.0))
    return changes


def quantile_type7(sorted_values: Sequence[float], q: float) -> float:
    h = (len(sorted_values) - 1) * q
    lo = math.floor(h)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (h - lo) * (sorted_values[hi] - sorted_values[lo])


def oi_jump_threshold(series: Sequence[tuple[datetime, float]]) -> float | None:
    """The policy quantile of |1-hour change| over the trailing window, *excluding* the newest
    change (so a jump is never inside its own baseline). The primary freezes this threshold at
    genesis; the replica, which has no genesis step, recomputes it from the same trailing window
    each run. ``None`` below the minimum sample the quantile needs (100 at the 99th)."""
    q = policy.TRIGGER_OI_JUMP_QUANTILE
    changes = hourly_oi_changes_pct(series)[:-1]
    need = math.ceil(round(1.0 / (1.0 - q), 9))
    magnitudes = sorted(abs(c) for c in changes)
    if len(magnitudes) < need:
        return None
    threshold = quantile_type7(magnitudes, q)
    return threshold if math.isfinite(threshold) and threshold > 0 else None


def fear_greed_band(value: float) -> str:
    """Bitget's bands, extremes from the policy (``features.fear_greed_band``)."""
    if value <= policy.TRIGGER_FEAR_GREED_LOW:
        return "extreme_fear"
    if value <= policy.FEATURE_FEAR_UPPER:
        return "fear"
    if value <= policy.FEATURE_NEUTRAL_UPPER:
        return "neutral"
    if value < policy.TRIGGER_FEAR_GREED_HIGH:
        return "greed"
    return "extreme_greed"


def earnings_instant(report_date: str, timing: str) -> datetime | None:
    """When an earnings report lands, from a calendar date and its timing: after the close for
    ``amc``-style timings, at the US open otherwise (before the open, or unstated)."""
    try:
        day = date.fromisoformat(report_date[:10])
    except ValueError:
        return None
    opens = us_open_utc(day)
    if opens is None:
        return datetime(day.year, day.month, day.day, 12, tzinfo=UTC)
    lowered = timing.lower()
    if "amc" in lowered or "after" in lowered or "close" in lowered:
        return opens + timedelta(hours=6, minutes=30)
    return opens


# ------------------------------------------------------------------------------------------------
# Reading the data layer
# ------------------------------------------------------------------------------------------------


class Perception:
    """One run's reads. ``clock`` is injected so tests can fix time."""

    def __init__(
        self, *, clock: Callable[[], datetime], deadline: Deadline, symbols: Sequence[str]
    ):
        self.clock = clock
        self.deadline = deadline
        self.symbols = tuple(symbols)
        self.snapshot = Snapshot(taken_at=as_utc(clock()))
        self.news_items: list[NewsItem] = []

    # --- plumbing -------------------------------------------------------------------------------

    def _records(
        self, name: str, fetch: Callable[[], object], *, until: float
    ) -> list[dict[str, object]]:
        if not self.deadline.before(until):
            self.snapshot.sources.append(SourceCall(name, "skipped", 0, 0.0, "time budget"))
            return []
        started = as_utc(self.clock())
        try:
            result = fetch()
            rows = data.to_records(result)
        except Exception as exc:  # the data layer's failures are data, not crashes
            seconds = (as_utc(self.clock()) - started).total_seconds()
            detail = f"{type(exc).__name__}: {str(exc)[:200]}"
            self.snapshot.sources.append(SourceCall(name, "error", 0, seconds, detail))
            return []
        seconds = (as_utc(self.clock()) - started).total_seconds()
        records: list[dict[str, object]] = []
        if isinstance(rows, Mapping):
            rows = [rows]
        for row in rows or []:
            if isinstance(row, Mapping):
                records.append({str(k): v for k, v in row.items()})
        health = "ok" if records else "empty"
        self.snapshot.sources.append(SourceCall(name, health, len(records), seconds))
        return records

    # --- prices ---------------------------------------------------------------------------------

    def marks(self, *, until: float) -> None:
        """Mark, index and funding for every instrument, one call (``crypto.futures.mark_price``,
        no symbol, exchange ``bitget``); a per-symbol call for any instrument it did not carry."""
        rows = self._records(
            "crypto.futures.mark_price[bitget]",
            lambda: data.crypto.futures.mark_price(exchange="bitget"),
            until=until,
        )
        wanted = set(self.symbols)
        seen = self._take_marks(rows, wanted)
        for symbol in sorted(wanted - seen):
            more = self._records(
                f"crypto.futures.mark_price[bitget,{symbol}]",
                partial(data.crypto.futures.mark_price, symbol=symbol, exchange="bitget"),
                until=until,
            )
            self._take_marks(more, {symbol})

    def _take_marks(self, rows: list[dict[str, object]], wanted: set[str]) -> set[str]:
        fetched = as_utc(self.clock())
        seen: set[str] = set()
        for row in rows:
            symbol = str(row.get("symbol") or "").upper().replace("_", "").replace("/", "")
            if symbol not in wanted:
                continue
            mark = decimal(row.get("mark_price"))
            index = decimal(row.get("index_price"))
            if mark is None and index is None:
                continue
            seen.add(symbol)
            previous = self.snapshot.quotes.get(symbol)
            self.snapshot.quotes[symbol] = Quote(
                symbol=symbol,
                mark=mark,
                index=index,
                last=None if previous is None else previous.last,
                bid=None if previous is None else previous.bid,
                ask=None if previous is None else previous.ask,
                ts=row_time(row, "time", "timestamp"),
                fetched_at=fetched,
            )
            rate = number(row.get("last_funding_rate"))
            if rate is not None:
                self.snapshot.funding_rate[symbol] = rate
        return seen

    def tickers(self, symbols: Iterable[str], *, until: float) -> None:
        """Last, bid and ask (``crypto.futures.ticker``, exchange ``bitget``), merged into the
        quotes: G8's spread, G4's expected entry and G1's venue-gap reference."""
        for symbol in symbols:
            rows = self._records(
                f"crypto.futures.ticker[bitget,{symbol}]",
                partial(data.crypto.futures.ticker, symbol=symbol, exchange="bitget"),
                until=until,
            )
            if not rows:
                continue
            row = rows[-1]
            fetched = as_utc(self.clock())
            previous = self.snapshot.quotes.get(symbol)
            last = decimal(row.get("last"))
            bid = decimal(row.get("bid"))
            ask = decimal(row.get("ask"))
            ts = row_time(row, "timestamp", "time")
            self.snapshot.quotes[symbol] = Quote(
                symbol=symbol,
                mark=None if previous is None else previous.mark,
                index=None if previous is None else previous.index,
                last=last,
                bid=bid,
                ask=ask,
                ts=ts
                if previous is None or previous.ts is None or ts is None
                else min(ts, previous.ts),
                fetched_at=fetched if previous is None else min(fetched, previous.fetched_at),
            )
            change = number(row.get("change_percent"))
            if change is not None:
                self.snapshot.features.setdefault(symbol, {})["ticker_change_24h_pct"] = change

    def klines(self, symbols: Iterable[str], *, until: float) -> None:
        """Completed 1H bars (``crypto.futures.kline``, exchange ``bitget``, closed bars only) and
        the features computed from them."""
        now = self.snapshot.taken_at
        for symbol in symbols:
            rows = self._records(
                f"crypto.futures.kline[bitget,{symbol},1h]",
                partial(
                    data.crypto.futures.kline,
                    symbol=symbol,
                    interval="1h",
                    exchange="bitget",
                    limit=KLINE_BARS,
                    closed_only=True,
                ),
                until=until,
            )
            candles: list[Candle] = []
            for row in rows:
                opened = row_time(row, "time", "date", "timestamp")
                values = [number(row.get(k)) for k in ("open", "high", "low", "close")]
                if opened is None or any(v is None for v in values):
                    continue
                o, h, low, c = (float(v) for v in values if v is not None)
                candles.append(Candle(opened, o, h, low, c))
            bars = closed_candles(candles, now)
            self.snapshot.candles[symbol] = bars
            feats = self.snapshot.features.setdefault(symbol, {})
            if not fresh(bars, now):
                feats.pop("move_bps_3h", None)
                continue
            for name, value in (
                ("move_bps_3h", move_bps_3h(bars)),
                ("ma20_distance_atr", ma_distance_atr(bars)),
                ("price_change_24h_pct", change_24h_pct(bars)),
            ):
                if value is not None:
                    feats[name] = value
            feats["last_close_1h"] = bars[-1].close

    # --- positioning ------------------------------------------------------------------------------

    def btc_positioning(self, *, until: float) -> None:
        """BTCUSDT funding history and open interest, read every run (the triggers use them)."""
        now = self.snapshot.taken_at
        feats = self.snapshot.features.setdefault(BTC, {})
        history = self._funding_history(until=until)
        self.snapshot.funding_history = history
        current = self.snapshot.funding_rate.get(BTC)
        if current is not None:
            feats["funding_rate"] = current
        z = funding_z(history, current)
        if z is not None:
            feats["funding_z"] = z
        rows = self._records(
            "crypto.futures.open_interest[bitget,BTCUSDT,1h]",
            lambda: data.crypto.futures.open_interest(
                symbol=BTC,
                interval="1h",
                exchange="bitget",
                days=policy.TRIGGER_OI_JUMP_LOOKBACK_DAYS,
                limit=1000,
            ),
            until=until,
        )
        series: list[tuple[datetime, float]] = []
        for row in rows:
            at = row_time(row, "time", "date", "timestamp")
            value = number(row.get("open_interest"))
            if value is None:
                value = number(row.get("oi_close"))
            if at is not None and value is not None:
                series.append((at, value))
        series.sort()
        self.snapshot.oi_series = series
        if series:
            feats["open_interest"] = series[-1][1]
        for hours, name in ((1, "oi_change_1h_pct"), (24, "oi_change_24h_pct")):
            value = oi_change_pct(series, hours=hours, at=now)
            if value is not None:
                feats[name] = value
        self.snapshot.oi_jump_threshold_pct = oi_jump_threshold(series)

    def btc_ratios(self, *, until: float) -> None:
        """The BTCUSDT retail and top-trader long/short ratios, read on decision runs."""
        feats = self.snapshot.features.setdefault(BTC, {})
        for endpoint, name in (
            ("long_short_ratio", "retail_long_short_ratio"),
            ("long_short_top_position_ratio", "top_trader_long_short_ratio"),
        ):
            fetch = getattr(data.crypto.futures, endpoint)
            rows = self._records(
                f"crypto.futures.{endpoint}[bitget,BTCUSDT,1h]",
                partial(fetch, symbol=BTC, interval="1h", exchange="bitget", limit=2),
                until=until,
            )
            newest = max(
                ((row_time(r, "timestamp", "time", "date"), r) for r in rows),
                key=lambda item: item[0] or datetime.min.replace(tzinfo=UTC),
                default=None,
            )
            if newest is not None:
                ratio = number(newest[1].get("long_short_ratio"))
                if ratio is not None:
                    feats[name] = ratio

    def _funding_history(self, *, until: float) -> list[tuple[datetime, float]]:
        """Settled funding rates, one per funding period, tried with the pair symbol and then the
        base asset (the SDK notes that one upstream's funding endpoints take ``BTC``).

        Rows are grouped by ``timestamp``, which the SDK documents as the funding period; within a
        period the latest row stands. The newest period is still open and is dropped, so the
        baseline holds settlements only, as the primary's does. If the periods turn out to be
        closer together than four hours, the series is not settlement-level (the field meant
        something else) and no history is returned: the z-score is then not computed at all,
        rather than computed on a different statistic under the same name."""
        now = self.snapshot.taken_at
        for symbol in (BTC, "BTC"):
            rows = self._records(
                f"crypto.futures.funding_rate[bitget,{symbol}]",
                partial(
                    data.crypto.futures.funding_rate,
                    symbol=symbol,
                    exchange="bitget",
                    interval="1h",
                    days=30,
                    limit=1000,
                ),
                until=until,
            )
            ordered = sorted(
                (
                    (row_time(row, "date", "time") or row_time(row, "timestamp"), row)
                    for row in rows
                ),
                key=lambda item: item[0] or datetime.min.replace(tzinfo=UTC),
            )
            by_period: dict[datetime, float] = {}
            for _, row in ordered:
                period = row_time(row, "timestamp")
                rate = number(row.get("funding_rate"))
                if period is not None and rate is not None and period <= now:
                    by_period[period] = rate
            if not by_period:
                continue
            periods = sorted(by_period)[:-1]
            gaps = sorted(b - a for a, b in pairwise(periods))
            if not gaps or gaps[len(gaps) // 2] < timedelta(hours=4):
                return []
            return [(t, by_period[t]) for t in periods]
        return []

    # --- mood, crowd, calendar ------------------------------------------------------------------

    def mood(self, *, until: float) -> None:
        rows = self._records(
            "crypto.sentiment.crypto_fear_greed",
            lambda: data.crypto.sentiment.crypto_fear_greed(limit=2),
            until=until,
        )
        dated = sorted(
            ((row_time(r, "date", "time", "timestamp"), number(r.get("value"))) for r in rows),
            key=lambda item: item[0] or datetime.min.replace(tzinfo=UTC),
        )
        values = [v for _, v in dated if v is not None]
        if values:
            self.snapshot.mood["crypto_fear_greed"] = values[-1]
        if len(values) >= 2:
            self.snapshot.mood["crypto_fear_greed_prior"] = values[-2]
        rows = self._records(
            "sentiment.market_fear_greed", lambda: data.sentiment.market_fear_greed(), until=until
        )
        if rows:
            score = number(rows[-1].get("score"))
            prior = number(rows[-1].get("previous_close"))
            if score is not None:
                self.snapshot.mood["market_fear_greed"] = score
            if prior is not None:
                self.snapshot.mood["market_fear_greed_prior"] = prior

    def crowd(self, *, until: float) -> None:
        """News headlines, clustered and counted (every run: the coordination trigger)."""
        rows = self._records(
            "sentiment.news[all]",
            lambda: data.sentiment.news(category="all", limit=NEWS_PER_SOURCE),
            until=until,
        )
        items = [
            NewsItem(
                source=str(r.get("source") or "unknown"),
                published=row_time(r, "published", "date", "time"),
                title=str(r.get("title") or ""),
                summary=str(r.get("summary") or "")[:600],
            )
            for r in rows
            if str(r.get("title") or "").strip()
        ]
        self.news_items = items
        self.snapshot.crowd = read_crowd(items, universe=self.symbols, now=self.snapshot.taken_at)

    def forums(self, *, until: float) -> None:
        """Forum mention counts (``sentiment.trending``), added on decision runs to the crowd
        reading this run already made from the news."""
        forum_rows: list[dict[str, object]] = []
        for scope in ("all-stocks", "all-crypto"):
            forum_rows.extend(
                self._records(
                    f"sentiment.trending[{scope}]",
                    partial(data.sentiment.trending, filter=scope),
                    until=until,
                )
            )
        self.snapshot.crowd = read_crowd(
            self.news_items,
            universe=self.symbols,
            now=self.snapshot.taken_at,
            forum_rows=forum_rows,
        )

    def earnings(self, equities: Sequence[str], *, until: float) -> None:
        """Earnings reports of universe US equities inside the policy's lookahead
        (``equity.calendar.earnings``)."""
        if not equities:
            return
        now = self.snapshot.taken_at
        horizon = now + timedelta(hours=policy.TRIGGER_EARNINGS_LOOKAHEAD_HOURS)
        rows = self._records(
            "equity.calendar.earnings[us]",
            lambda: data.equity.calendar.earnings(
                start_time=int((now - timedelta(days=1)).timestamp() * 1000),
                end_time=int((horizon + timedelta(days=1)).timestamp() * 1000),
                country="us",
            ),
            until=until,
        )
        tickers = {_underlying(s): s for s in equities}
        for row in rows:
            symbol = tickers.get(str(row.get("symbol") or "").upper())
            report = str(row.get("report_date") or "")
            if symbol is None or not report:
                continue
            at = earnings_instant(report, str(row.get("reporting_time") or ""))
            if at is None:
                continue
            self.snapshot.earnings.append(
                CalendarEntry(symbol, at, report[:10], str(row.get("reporting_time") or ""))
            )

    def filings(self, held_since: Mapping[str, datetime], *, until: float) -> None:
        """Insider filings (``equity.ownership.insider_trading``) for held US equities, filed on or
        after the day the position was opened. Only dates and a key are kept; no name or title."""
        for symbol, opened in sorted(held_since.items()):
            ticker = _underlying(symbol)
            rows = self._records(
                f"equity.ownership.insider_trading[{ticker}]",
                partial(
                    data.equity.ownership.insider_trading,
                    symbol=ticker,
                    limit=20,
                    start_time=int(opened.timestamp() * 1000),
                ),
                until=until,
            )
            for row in rows:
                filed_at = row_time(row, "filing_date")
                if filed_at is None or filed_at.date() < opened.date():
                    continue
                key = "|".join(
                    str(row.get(k) or "")
                    for k in ("filing_date", "transaction_date", "owner_cik", "form", "filing_url")
                )
                self.snapshot.filings.append(FilingEntry(symbol, filed_at.date().isoformat(), key))


def _underlying(symbol: str) -> str:
    """The US ticker of a universe equity perpetual (``NVDAUSDT`` -> ``NVDA``)."""
    return symbol[: -len("USDT")] if symbol.endswith("USDT") else symbol


def describe_quote(quote: Quote) -> dict[str, object]:
    return {
        "mark": None if quote.mark is None else str(quote.mark),
        "index": None if quote.index is None else str(quote.index),
        "last": None if quote.last is None else str(quote.last),
        "bid": None if quote.bid is None else str(quote.bid),
        "ask": None if quote.ask is None else str(quote.ask),
        "ts": None if quote.ts is None else iso_z(quote.ts),
        "fetched_at": iso_z(quote.fetched_at),
    }
