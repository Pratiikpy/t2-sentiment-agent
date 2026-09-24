"""Keyless Bitget public market data, live and UTA Demo (module M1).

Four public v3 endpoints, read without any credential: ``tickers``, ``instruments``,
``history-candles`` (types market, mark and index) and ``history-fund-rate``. The same endpoints
answer with UTA Demo data when the request carries the header ``paptrading: 1``; without it they
answer with live data. Every request is recorded as a :class:`~sentiment_agent.types.SourceCall`
(with the raw body in the blob store when one is given), so a snapshot can say exactly what was
asked, what came back and what failed.

Where this departs from Bitget's own SDK, and why
    ``agent-sdk/src/client/rest-client.ts:274-280`` sends ``paptrading: 1`` only on private
    (account) calls, because several public endpoints answer 404 under it. This client does send it
    on public calls, but only on :data:`DEMO_ENDPOINTS`, the four paths measured to answer with
    Demo data under it (``validation/demo_venue/universe_probe.json``; re-measured for this module
    on 2026-09-24 and pinned by ``tests/fixtures/venue/cassette.json``). A Demo read of any other
    path is refused rather than sent without the header, because that would return live data
    labelled as Demo.

Facts this module relies on, each measured with keyless GETs on 2026-09-24 and pinned by fixture
    * An unknown or unlisted pair answers HTTP 400 with code ``25100`` on all four endpoints, live
      and Demo. A pair listed live but not on Demo (e.g. AAOIUSDT) answers ``25100`` under the Demo
      header, which is also the cleanest evidence that the header really routes to Demo.
    * ``history-candles`` returns only closed bars, oldest first, and ``endTime`` filters on the
      bar's *close*: a bar is returned when ``open + interval <= endTime``. The next page therefore
      ends at the oldest open time of the current page. (``validation/demo_venue/fetch.py:111``
      walks back with ``oldest - 1``, which skips one bar at every page boundary.) ``limit`` above
      100 answers ``40020``; interval names are case-sensitive (``1h`` answers ``40020``); daily
      bars open at 16:00 UTC.
    * ``history-fund-rate`` returns ``data.resultList`` newest first; ``cursor`` is a page number
      from 1 to 100 (101 answers ``40808``); the SDK catalog documents ``limit`` up to 200
      (``agent-sdk/src/generated/catalog.ts:107``) but the live API answers ``40020`` above 100.
    * Tickers and instruments carry every number as a JSON string. Mark and index candles carry
      ``"0"`` in the volume columns, which is a placeholder, not a measured zero, so it parses to
      ``None``.

Transport policy
    * Requests are paced: at least ``min_interval_s`` between request starts across the whole
      client, all threads included. The default 0.06 s keeps below the 20 requests per second the
      SDK budgets for public endpoints (``agent-sdk/src/tools/common.ts:4-5``).
    * Transient failures are retried as the SDK does (``agent-sdk/src/utils/retry.ts:23-28``): at
      most two retries, on HTTP 429/500/502/503/504 and on connection errors. Two deliberate
      differences: the backoff has no jitter (one client, no herd to spread), and a timeout is not
      retried, because a timeout has already cost ``timeout_s`` and the caller's next cycle is the
      retry. Application errors (a non-``00000`` code) are deterministic and never retried.
    * ``quotes`` and ``instruments`` send one request per symbol, concurrently with paced starts.
      One request per symbol keeps each raw blob small (the whole-category live ticker list is about
      390 KB); concurrency keeps the cross-section tight, so a Demo quote and a live quote of the
      same symbol are about a second apart rather than the ~6 s a sequential walk of 14 symbols
      takes. Each :class:`Quote` carries the venue's own ``ts`` for anyone who needs the exact skew.
    * Never an ``ACCESS-*`` header, never a credential, never a POST. The ``User-Agent`` is set
      explicitly. Redirects are refused.

Failure semantics
    * A single-series method (``candles``, ``funding_history``) raises :class:`PublicApiError` when
      it cannot produce its answer.
    * A multi-symbol method (``quotes``, ``instruments``) returns the symbols that answered. A
      symbol the venue does not list is absent (its call is recorded ``EMPTY`` with code 25100); a
      symbol whose call failed is absent (recorded ``ERROR`` or ``TIMEOUT``). Only when no symbol
      answered and at least one call failed for a reason other than "not listed" does it raise, so
      an outage is never mistaken for an empty venue.
"""

import http.client
import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, Final, TypeVar

from sentiment_agent import __version__
from sentiment_agent.types import (
    PROJECT_SLUG,
    BlobRef,
    BlobStore,
    Candle,
    CandleKind,
    Category,
    Clock,
    FundingPoint,
    InstrumentSpec,
    PriceSource,
    Quote,
    SourceCall,
    SourceHealth,
    ToolkitSurface,
)

BASE_URL: Final = "https://api.bitget.com"

PATH_TICKERS: Final = "/api/v3/market/tickers"
PATH_INSTRUMENTS: Final = "/api/v3/market/instruments"
PATH_HISTORY_CANDLES: Final = "/api/v3/market/history-candles"
PATH_HISTORY_FUND_RATE: Final = "/api/v3/market/history-fund-rate"

DEMO_ENDPOINTS: Final[frozenset[str]] = frozenset(
    {PATH_TICKERS, PATH_INSTRUMENTS, PATH_HISTORY_CANDLES, PATH_HISTORY_FUND_RATE}
)
"""Paths measured to answer with UTA Demo data under ``paptrading: 1`` (history-candles for the
types market, mark and index). The Demo header is sent on these and nowhere else."""

DEMO_HEADER_NAME: Final = "paptrading"
DEMO_HEADER_VALUE: Final = "1"
USER_AGENT: Final = f"{PROJECT_SLUG}/{__version__} (keyless public market data)"
CATEGORY: Final = Category.USDT_FUTURES.value

OK_CODE: Final = "00000"
NOT_LISTED_CODE: Final = "25100"
"""``Trading pair X does not exist``: the venue does not list the symbol (HTTP 400)."""

PAGE_LIMIT: Final = 100
"""Rows per page for history-candles and history-fund-rate; 101 answers ``40020`` on both."""
MAX_FUNDING_CURSOR: Final = 100
MAX_FUNDING_POINTS: Final = PAGE_LIMIT * MAX_FUNDING_CURSOR

INTERVALS: Final[Mapping[str, timedelta]] = {
    "1m": timedelta(minutes=1),
    "3m": timedelta(minutes=3),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1H": timedelta(hours=1),
    "4H": timedelta(hours=4),
    "6H": timedelta(hours=6),
    "12H": timedelta(hours=12),
    "1D": timedelta(days=1),
}
"""The interval enum of ``agent-sdk/src/generated/catalog.ts:111``, case-sensitive."""
CANDLE_KINDS: Final[frozenset[str]] = frozenset({"market", "mark", "index"})

RETRY_STATUSES: Final[frozenset[int]] = frozenset({429, 500, 502, 503, 504})
MAX_RETRIES: Final = 2
RETRY_BACKOFF_S: Final[tuple[float, ...]] = (0.25, 0.5)
MAX_FANOUT: Final = 16
MAX_BODY_BYTES: Final = 16 * 1024 * 1024

_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)
_ONE_MS: Final = timedelta(milliseconds=1)
_SYMBOL: Final = re.compile(r"^[A-Z0-9]{2,40}$")

# Timing seams. Pacing, backoff and latency use a monotonic timer, never the wall clock; tests
# replace these two names to observe and control them.
_monotonic: Callable[[], float] = time.monotonic
_sleep: Callable[[float], None] = time.sleep

HttpGet = Callable[[str, Mapping[str, str], float], tuple[int, bytes]]
"""``(url, headers, timeout_s) -> (status, body)``. Returns error statuses with their body; raises
``OSError`` (timeouts included) or ``http.client.HTTPException`` when no response arrived."""

_T = TypeVar("_T")


# ================================================================================================
# Errors
# ================================================================================================


class PublicApiError(RuntimeError):
    """A public market-data request that produced no usable answer.

    ``code`` is the venue's own code (e.g. ``"25100"``, ``"40020"``) when the venue answered with
    one, and ``None`` for a transport failure, a non-JSON body, or a response whose shape could not
    be parsed.
    """

    def __init__(
        self, message: str, *, code: str | None, path: str, http_status: int | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.path = path
        self.http_status = http_status


class ResponseTooLargeError(OSError):
    """The response body exceeded :data:`MAX_BODY_BYTES`."""


class _EmptyFieldError(ValueError):
    """A required value was present but empty: the venue answered with nothing usable."""


class _TransportError(Exception):
    def __init__(
        self,
        message: str,
        *,
        started_at: datetime,
        latency_ms: int,
        attempts: int,
        timed_out: bool,
    ) -> None:
        super().__init__(message)
        self.started_at = started_at
        self.latency_ms = latency_ms
        self.attempts = attempts
        self.timed_out = timed_out


# ================================================================================================
# Transport
# ================================================================================================


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """The public API never redirects; a redirect surfaces as an HTTP error instead of being
    followed with our headers to wherever it points."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        return None


_OPENER: Final = urllib.request.build_opener(_RefuseRedirects)


def urllib_get(url: str, headers: Mapping[str, str], timeout: float) -> tuple[int, bytes]:
    """GET ``url`` over HTTPS with exactly ``headers``. Error statuses come back with their body."""
    if urllib.parse.urlsplit(url).scheme != "https":
        raise ValueError(f"refusing a non-HTTPS URL: {url!r}")
    # The scheme is checked above; S310 guards against file:// and custom schemes.
    request = urllib.request.Request(url, headers=dict(headers), method="GET")  # noqa: S310
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            return int(response.status), _read_capped(response)
    except urllib.error.HTTPError as exc:
        body = _read_capped(exc) if exc.fp is not None else b""
        return int(exc.code), body


def _read_capped(stream: Any) -> bytes:
    data = stream.read(MAX_BODY_BYTES + 1)
    if not isinstance(data, bytes):
        raise TypeError("response body is not bytes")
    if len(data) > MAX_BODY_BYTES:
        raise ResponseTooLargeError(f"response body exceeds {MAX_BODY_BYTES} bytes")
    return data


def _is_timeout(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    return isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, TimeoutError)


def _request_headers(source: PriceSource, path: str) -> dict[str, str]:
    """The only place request headers are built. Demo header on Demo endpoints only; never auth."""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json", "locale": "en-US"}
    if source is PriceSource.DEMO:
        if path not in DEMO_ENDPOINTS:
            raise ValueError(
                f"{path} is not measured to answer under {DEMO_HEADER_NAME}: {DEMO_HEADER_VALUE}; "
                "refusing a Demo read that could return live data"
            )
        headers[DEMO_HEADER_NAME] = DEMO_HEADER_VALUE
    _assert_keyless(headers)
    return headers


def _assert_keyless(headers: Mapping[str, str]) -> None:
    for name in headers:
        if name.lower().startswith("access-"):
            raise RuntimeError(
                f"refusing to send {name}: the public market-data client carries no credential"
            )


class _Pacer:
    """At least ``interval`` seconds between request starts, across every thread of one client."""

    def __init__(self, interval: float) -> None:
        self._interval = interval
        self._lock = threading.Lock()
        self._next: float | None = None

    def wait(self) -> None:
        with self._lock:
            now = _monotonic()
            if self._next is not None and now < self._next:
                _sleep(self._next - now)
                now = max(self._next, _monotonic())
            self._next = now + self._interval


@dataclass(frozen=True)
class _Exchange:
    started_at: datetime
    latency_ms: int
    attempts: int
    status: int
    body: bytes


# ================================================================================================
# Value parsing
# ================================================================================================


def _field(row: Mapping[str, Any], key: str) -> Any:
    if key not in row:
        raise ValueError(f"missing field {key!r}")
    return row[key]


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _to_decimal(value: Any, what: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, Decimal)):
        raise ValueError(f"{what}: expected a decimal string, got {type(value).__name__}")
    try:
        number = Decimal(value.strip()) if isinstance(value, str) else Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{what}: {value!r} is not a decimal") from exc
    if not number.is_finite():
        raise ValueError(f"{what}: {value!r} is not finite")
    return number


def _required_decimal(row: Mapping[str, Any], key: str) -> Decimal:
    value = _field(row, key)
    if _is_blank(value):
        raise _EmptyFieldError(f"{key!r} is empty")
    return _to_decimal(value, key)


def _optional_decimal(row: Mapping[str, Any], key: str) -> Decimal | None:
    value = row.get(key)
    return None if _is_blank(value) else _to_decimal(value, key)


def _optional_int(row: Mapping[str, Any], key: str) -> int | None:
    number = _optional_decimal(row, key)
    if number is None:
        return None
    if number != number.to_integral_value():
        raise ValueError(f"{key}: {number} is not an integer")
    return int(number)


def _required_text(row: Mapping[str, Any], key: str) -> str:
    value = _field(row, key)
    if not isinstance(value, str):
        raise ValueError(f"{key}: expected a string, got {type(value).__name__}")
    if value.strip() == "":
        raise _EmptyFieldError(f"{key!r} is empty")
    return value


def _ms_to_utc(value: Any, what: str) -> datetime:
    if _is_blank(value):
        raise _EmptyFieldError(f"{what} is empty")
    number = _to_decimal(value, what)
    if number != number.to_integral_value() or number < 0:
        raise ValueError(f"{what}: {value!r} is not a millisecond timestamp")
    return _EPOCH + int(number) * _ONE_MS


def _utc_ms(value: datetime, what: str) -> int:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{what} must be a timezone-aware UTC datetime")
    return (value.astimezone(UTC) - _EPOCH) // _ONE_MS


def _check_symbol(symbol: str) -> str:
    if not isinstance(symbol, str) or not _SYMBOL.fullmatch(symbol):
        raise ValueError(f"not a Bitget symbol: {symbol!r}")
    return symbol


def _unique_symbols(symbols: Sequence[str]) -> list[str]:
    if isinstance(symbols, str):
        raise TypeError("symbols must be a sequence of symbols, not one string")
    seen: dict[str, None] = {}
    for symbol in symbols:
        seen[_check_symbol(symbol)] = None
    return list(seen)


def _interval(interval: str) -> timedelta:
    step = INTERVALS.get(interval)
    if step is None:
        raise ValueError(f"interval {interval!r} is not one of {sorted(INTERVALS)}")
    return step


def _check_kind(kind: str) -> None:
    if kind not in CANDLE_KINDS:
        raise ValueError(f"candle kind {kind!r} is not one of {sorted(CANDLE_KINDS)}")


def parse_instrument(
    row: Mapping[str, Any], source: PriceSource, fetched_at: datetime
) -> InstrumentSpec:
    """One row of ``GET /api/v3/market/instruments``. Raises ``ValueError`` on a malformed row."""
    qty_step = _required_decimal(row, "quantityMultiplier")
    price_step = _required_decimal(row, "priceMultiplier")
    if qty_step <= 0 or price_step <= 0:
        raise ValueError("quantityMultiplier and priceMultiplier must be positive")
    return InstrumentSpec(
        symbol=_required_text(row, "symbol"),
        category=Category(_required_text(row, "category")),
        source=source,
        base_coin=_required_text(row, "baseCoin"),
        quote_coin=_required_text(row, "quoteCoin"),
        status=_required_text(row, "status"),
        min_order_qty=_required_decimal(row, "minOrderQty"),
        qty_step=qty_step,
        price_step=price_step,
        min_order_amount=_required_decimal(row, "minOrderAmount"),
        max_market_order_qty=_optional_decimal(row, "maxMarketOrderQty"),
        max_order_qty=_optional_decimal(row, "maxOrderQty"),
        taker_fee_rate=_required_decimal(row, "takerFeeRate"),
        maker_fee_rate=_required_decimal(row, "makerFeeRate"),
        max_leverage=_optional_int(row, "maxLeverage"),
        fund_interval_hours=_optional_int(row, "fundInterval"),
        fetched_at=fetched_at,
    )


def parse_ticker(row: Mapping[str, Any], source: PriceSource, fetched_at: datetime) -> Quote:
    """One row of ``GET /api/v3/market/tickers``. ``price24hPcnt`` is already a fraction."""
    return Quote(
        symbol=_required_text(row, "symbol"),
        source=source,
        ts=_ms_to_utc(_field(row, "ts"), "ts"),
        fetched_at=fetched_at,
        last=_required_decimal(row, "lastPrice"),
        mark=_required_decimal(row, "markPrice"),
        index=_required_decimal(row, "indexPrice"),
        bid=_required_decimal(row, "bid1Price"),
        ask=_required_decimal(row, "ask1Price"),
        funding_rate=_optional_decimal(row, "fundingRate"),
        open_interest=_optional_decimal(row, "openInterest"),
        turnover_24h=_optional_decimal(row, "turnover24h"),
        price_change_24h=_optional_decimal(row, "price24hPcnt"),
    )


def parse_candle(
    row: Sequence[str], *, symbol: str, source: PriceSource, kind: CandleKind, interval: str
) -> Candle:
    """One row of ``GET /api/v3/market/history-candles``:
    ``[openTime ms, open, high, low, close, baseVolume, quoteVolume]``.

    Volume is read for ``market`` candles only; mark and index candles carry a ``"0"`` placeholder.
    """
    _check_kind(kind)
    _interval(interval)
    if isinstance(row, (str, bytes)) or not isinstance(row, Sequence) or len(row) < 5:
        raise ValueError(f"a candle row needs at least 5 columns, got {row!r}")

    def cell(i: int, name: str) -> Decimal:
        if _is_blank(row[i]):
            raise _EmptyFieldError(f"candle {name} is empty")
        return _to_decimal(row[i], f"candle {name}")

    volume: Decimal | None = None
    if kind == "market" and len(row) > 5 and not _is_blank(row[5]):
        volume = _to_decimal(row[5], "candle volume")
    return Candle(
        symbol=symbol,
        source=source,
        kind=kind,
        interval=interval,
        open_time=_ms_to_utc(row[0], "candle open time"),
        open=cell(1, "open"),
        high=cell(2, "high"),
        low=cell(3, "low"),
        close=cell(4, "close"),
        volume=volume,
    )


def _parse_funding(row: Mapping[str, Any], symbol: str) -> FundingPoint:
    if _required_text(row, "symbol") != symbol:
        raise ValueError(f"funding row for {row.get('symbol')!r} in a {symbol} response")
    return FundingPoint(
        symbol=symbol,
        source=PriceSource.LIVE,
        ts=_ms_to_utc(_field(row, "fundingRateTimestamp"), "fundingRateTimestamp"),
        rate=_required_decimal(row, "fundingRate"),
    )


def _rows_for(data: Any, symbol: str) -> list[Mapping[str, Any]]:
    """The rows of a list-shaped ``data`` that describe ``symbol`` (at most one)."""
    if not isinstance(data, list):
        raise ValueError(f"expected a list, got {type(data).__name__}")
    rows = [r for r in data if isinstance(r, Mapping) and r.get("symbol") == symbol]
    if len(rows) > 1:
        raise ValueError(f"{len(rows)} rows for {symbol}")
    return rows


# ================================================================================================
# The client
# ================================================================================================


_Parse = Callable[[Any, datetime], tuple[_T, int]]


class BitgetPublicApi:
    """Keyless Bitget public market data, live and UTA Demo. Satisfies ``types.MarketData``.

    Thread-safe: requests from any thread share one pacer, and :meth:`drain_calls` hands back every
    recorded :class:`SourceCall` exactly once, in request order per method call.
    """

    def __init__(
        self,
        *,
        clock: Clock,
        http: HttpGet = urllib_get,
        blobs: BlobStore | None = None,
        timeout_s: float = 15.0,
        min_interval_s: float = 0.06,
    ) -> None:
        if not timeout_s > 0:
            raise ValueError("timeout_s must be positive")
        if not min_interval_s >= 0:
            raise ValueError("min_interval_s must be non-negative")
        self._clock = clock
        self._http = http
        self._blobs = blobs
        self._timeout_s = timeout_s
        self._pacer = _Pacer(min_interval_s)
        self._calls: list[SourceCall] = []
        self._calls_lock = threading.Lock()

    # -- MarketData --------------------------------------------------------------------------

    def instruments(self, source: PriceSource, symbols: Sequence[str]) -> dict[str, InstrumentSpec]:
        """Instrument specifications for ``symbols`` on ``source``; unlisted symbols are absent."""
        return self._fan_out(
            _unique_symbols(symbols), lambda symbol, sink: self._instrument(source, symbol, sink)
        )

    def quotes(self, source: PriceSource, symbols: Sequence[str]) -> dict[str, Quote]:
        """Current tickers for ``symbols`` on ``source``; unlisted or failed symbols are absent."""
        return self._fan_out(
            _unique_symbols(symbols), lambda symbol, sink: self._quote(source, symbol, sink)
        )

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
        """Every closed candle that opened at or after ``start`` and closed at or before ``end``,
        oldest first, walking back from ``end`` 100 rows a page."""
        _check_symbol(symbol)
        _check_kind(kind)
        step = _interval(interval)
        start_ms = _utc_ms(start, "start")
        end_ms = _utc_ms(end, "end")
        if start_ms >= end_ms:
            raise ValueError("start must be before end")
        step_ms = step // _ONE_MS
        max_pages = -(-(end_ms - start_ms) // (step_ms * PAGE_LIMIT)) + 1
        base_label = f"history-candles.{source.value}:{symbol}:{kind}:{interval}"

        def parse(data: Any, _fetched_at: datetime) -> tuple[list[Candle], int]:
            if not isinstance(data, list):
                raise ValueError(f"expected a list of candle rows, got {type(data).__name__}")
            page = [
                parse_candle(row, symbol=symbol, source=source, kind=kind, interval=interval)
                for row in data
            ]
            if len({c.open_time for c in page}) != len(page):
                raise ValueError("duplicate open times within one page")
            return page, len(page)

        collected: dict[datetime, Candle] = {}
        sink: list[SourceCall] = []
        try:
            cursor_ms = end_ms
            page_no = 1
            while True:
                params = {
                    "category": CATEGORY,
                    "symbol": symbol,
                    "interval": interval,
                    "type": kind,
                    "limit": str(PAGE_LIMIT),
                    "endTime": str(cursor_ms),
                }
                label = base_label if page_no == 1 else f"{base_label}#p{page_no}"
                page = self._call(
                    sink=sink,
                    source=source,
                    path=PATH_HISTORY_CANDLES,
                    params=params,
                    label=label,
                    parse=parse,
                )
                if not page:
                    break
                oldest_ms = min(_utc_ms(c.open_time, "open_time") for c in page)
                if oldest_ms >= cursor_ms:
                    raise PublicApiError(
                        f"{base_label}: page {page_no} made no progress past endTime={cursor_ms}",
                        code=None,
                        path=PATH_HISTORY_CANDLES,
                    )
                for c in page:
                    if start <= c.open_time and c.open_time + step <= end:
                        collected[c.open_time] = c
                if len(page) < PAGE_LIMIT or oldest_ms <= start_ms:
                    break
                if page_no >= max_pages:
                    raise PublicApiError(
                        f"{base_label}: start not reached after {page_no} pages",
                        code=None,
                        path=PATH_HISTORY_CANDLES,
                    )
                cursor_ms = oldest_ms
                page_no += 1
        finally:
            self._commit(sink)
        return [collected[t] for t in sorted(collected)]

    def funding_history(self, symbol: str, *, limit: int) -> list[FundingPoint]:
        """The newest ``limit`` live funding settlements, oldest first.

        Live only, by design (DESIGN.md §6.1): Demo funding is a sandbox setting, not a crowd.
        """
        _check_symbol(symbol)
        if isinstance(limit, bool) or not 1 <= limit <= MAX_FUNDING_POINTS:
            raise ValueError(f"limit must be between 1 and {MAX_FUNDING_POINTS}")
        base_label = f"history-fund-rate.{PriceSource.LIVE.value}:{symbol}"

        def parse(data: Any, _fetched_at: datetime) -> tuple[list[FundingPoint], int]:
            if not isinstance(data, Mapping):
                raise ValueError(f"expected an object with resultList, got {type(data).__name__}")
            rows = _field(data, "resultList")
            if not isinstance(rows, list):
                raise ValueError("resultList is not a list")
            points = []
            for row in rows:
                if not isinstance(row, Mapping):
                    raise ValueError("a funding row is not an object")
                points.append(_parse_funding(row, symbol))
            return points, len(points)

        points: dict[datetime, FundingPoint] = {}
        sink: list[SourceCall] = []
        try:
            for cursor in range(1, MAX_FUNDING_CURSOR + 1):
                params = {
                    "category": CATEGORY,
                    "symbol": symbol,
                    "limit": str(PAGE_LIMIT),
                    "cursor": str(cursor),
                }
                label = base_label if cursor == 1 else f"{base_label}#p{cursor}"
                page = self._call(
                    sink=sink,
                    source=PriceSource.LIVE,
                    path=PATH_HISTORY_FUND_RATE,
                    params=params,
                    label=label,
                    parse=parse,
                )
                for point in page:
                    points[point.ts] = point
                if len(page) < PAGE_LIMIT or len(points) >= limit:
                    break
        finally:
            self._commit(sink)
        ordered = [points[t] for t in sorted(points)]
        return ordered[-limit:]

    def drain_calls(self) -> tuple[SourceCall, ...]:
        """Every call recorded since the last drain, then forget them."""
        with self._calls_lock:
            calls = tuple(self._calls)
            self._calls.clear()
        return calls

    # -- one symbol ------------------------------------------------------------------------------

    def _quote(self, source: PriceSource, symbol: str, sink: list[SourceCall]) -> Quote | None:
        def parse(data: Any, fetched_at: datetime) -> tuple[Quote | None, int]:
            rows = _rows_for(data, symbol)
            if not rows:
                return None, 0
            return parse_ticker(rows[0], source, fetched_at), 1

        return self._call(
            sink=sink,
            source=source,
            path=PATH_TICKERS,
            params={"category": CATEGORY, "symbol": symbol},
            label=f"tickers.{source.value}:{symbol}",
            parse=parse,
        )

    def _instrument(
        self, source: PriceSource, symbol: str, sink: list[SourceCall]
    ) -> InstrumentSpec | None:
        def parse(data: Any, fetched_at: datetime) -> tuple[InstrumentSpec | None, int]:
            rows = _rows_for(data, symbol)
            if not rows:
                return None, 0
            return parse_instrument(rows[0], source, fetched_at), 1

        return self._call(
            sink=sink,
            source=source,
            path=PATH_INSTRUMENTS,
            params={"category": CATEGORY, "symbol": symbol},
            label=f"instruments.{source.value}:{symbol}",
            parse=parse,
        )

    # -- machinery -------------------------------------------------------------------------------

    def _fan_out(
        self,
        symbols: Sequence[str],
        fetch: Callable[[str, list[SourceCall]], _T | None],
    ) -> dict[str, _T]:
        if not symbols:
            return {}
        sinks: dict[str, list[SourceCall]] = {symbol: [] for symbol in symbols}
        results: dict[str, _T] = {}
        failures: list[PublicApiError] = []
        futures: list[tuple[str, Future[_T | None]]] = []
        try:
            with ThreadPoolExecutor(
                max_workers=min(MAX_FANOUT, len(symbols)), thread_name_prefix="bitget-public"
            ) as pool:
                futures = [(s, pool.submit(fetch, s, sinks[s])) for s in symbols]
            for symbol, future in futures:
                try:
                    value = future.result()
                except PublicApiError as exc:
                    if exc.code != NOT_LISTED_CODE:
                        failures.append(exc)
                    continue
                if value is not None:
                    results[symbol] = value
        finally:
            for symbol in symbols:
                self._commit(sinks[symbol])
        if not results and failures:
            raise failures[0]
        return results

    def _call(
        self,
        *,
        sink: list[SourceCall],
        source: PriceSource,
        path: str,
        params: Mapping[str, str],
        label: str,
        parse: "_Parse[_T]",
    ) -> _T:
        headers = _request_headers(source, path)
        url = f"{BASE_URL}{path}?{urllib.parse.urlencode(params)}"

        def record(
            health: SourceHealth,
            *,
            started_at: datetime,
            latency_ms: int,
            rows: int = 0,
            blob: BlobRef | None = None,
            error: str | None = None,
        ) -> None:
            sink.append(
                SourceCall(
                    call_id=f"pub-{uuid.uuid4().hex}",
                    surface=ToolkitSurface.PUBLIC_MARKET_API,
                    source=label,
                    params=dict(params),
                    health=health,
                    started_at=started_at,
                    latency_ms=latency_ms,
                    rows=rows,
                    blob=blob,
                    error=error,
                )
            )

        try:
            exchange = self._fetch(url, headers)
        except _TransportError as failure:
            record(
                SourceHealth.TIMEOUT if failure.timed_out else SourceHealth.ERROR,
                started_at=failure.started_at,
                latency_ms=failure.latency_ms,
                error=str(failure),
            )
            raise PublicApiError(f"{label}: {failure}", code=None, path=path) from failure

        started_at, latency_ms = exchange.started_at, exchange.latency_ms
        retried = f" after {exchange.attempts} attempts" if exchange.attempts > 1 else ""
        envelope = _decode_envelope(exchange.body)
        blob = self._store(exchange.body, envelope is not None)
        if envelope is None:
            message = f"HTTP {exchange.status}{retried}: the body is not a JSON object"
            record(
                SourceHealth.ERROR,
                blob=blob,
                error=message,
                started_at=started_at,
                latency_ms=latency_ms,
            )
            raise PublicApiError(
                f"{label}: {message}", code=None, path=path, http_status=exchange.status
            )

        raw_code = envelope.get("code")
        code = None if raw_code is None else str(raw_code)
        if exchange.status != 200 or code != OK_CODE:
            message = f"{code}: {envelope.get('msg')} (HTTP {exchange.status}{retried})"
            health = SourceHealth.EMPTY if code == NOT_LISTED_CODE else SourceHealth.ERROR
            record(health, blob=blob, error=message, started_at=started_at, latency_ms=latency_ms)
            raise PublicApiError(
                f"{label}: {message}", code=code, path=path, http_status=exchange.status
            )

        fetched_at = self._clock.now()
        try:
            result, rows = parse(envelope.get("data"), fetched_at)
        except _EmptyFieldError as exc:
            message = f"hollow response: {exc}"
            record(
                SourceHealth.HOLLOW,
                blob=blob,
                error=message,
                started_at=started_at,
                latency_ms=latency_ms,
            )
            raise PublicApiError(f"{label}: {message}", code=None, path=path) from exc
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            message = f"unexpected response shape: {exc}"
            record(
                SourceHealth.ERROR,
                blob=blob,
                error=message,
                started_at=started_at,
                latency_ms=latency_ms,
            )
            raise PublicApiError(f"{label}: {message}", code=None, path=path) from exc
        record(
            SourceHealth.OK if rows else SourceHealth.EMPTY,
            rows=rows,
            blob=blob,
            started_at=started_at,
            latency_ms=latency_ms,
        )
        return result

    def _fetch(self, url: str, headers: Mapping[str, str]) -> _Exchange:
        attempts = 0
        started_at: datetime | None = None
        t0 = 0.0
        while True:
            self._pacer.wait()
            if started_at is None:
                started_at = self._clock.now()
                t0 = _monotonic()
            attempts += 1
            try:
                status, body = self._http(url, headers, self._timeout_s)
            except (OSError, http.client.HTTPException) as exc:
                timed_out = _is_timeout(exc)
                if not timed_out and attempts <= MAX_RETRIES:
                    _sleep(RETRY_BACKOFF_S[attempts - 1])
                    continue
                what = f"timeout after {self._timeout_s:g} s" if timed_out else repr(exc)
                tries = f" (attempt {attempts})" if attempts > 1 else ""
                raise _TransportError(
                    f"{what}{tries}",
                    started_at=started_at,
                    latency_ms=_elapsed_ms(t0),
                    attempts=attempts,
                    timed_out=timed_out,
                ) from exc
            if status in RETRY_STATUSES and attempts <= MAX_RETRIES:
                _sleep(RETRY_BACKOFF_S[attempts - 1])
                continue
            return _Exchange(started_at, _elapsed_ms(t0), attempts, status, body)

    def _store(self, body: bytes, is_json: bool) -> BlobRef | None:
        if self._blobs is None or not body:
            return None
        return self._blobs.put(body, "application/json" if is_json else "application/octet-stream")

    def _commit(self, calls: list[SourceCall]) -> None:
        if calls:
            with self._calls_lock:
                self._calls.extend(calls)


def _elapsed_ms(t0: float) -> int:
    return max(0, round((_monotonic() - t0) * 1000))


def _decode_envelope(body: bytes) -> dict[str, Any] | None:
    try:
        decoded = json.loads(body.decode("utf-8"), parse_float=Decimal)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return decoded if isinstance(decoded, dict) else None


__all__ = [
    "BASE_URL",
    "CANDLE_KINDS",
    "CATEGORY",
    "DEMO_ENDPOINTS",
    "DEMO_HEADER_NAME",
    "DEMO_HEADER_VALUE",
    "INTERVALS",
    "MAX_FUNDING_POINTS",
    "NOT_LISTED_CODE",
    "PAGE_LIMIT",
    "PATH_HISTORY_CANDLES",
    "PATH_HISTORY_FUND_RATE",
    "PATH_INSTRUMENTS",
    "PATH_TICKERS",
    "USER_AGENT",
    "BitgetPublicApi",
    "HttpGet",
    "PublicApiError",
    "ResponseTooLargeError",
    "parse_candle",
    "parse_instrument",
    "parse_ticker",
    "urllib_get",
]
