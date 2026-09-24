"""bitget-mcp-server, read as typed readings: mood, crowd positioning, earnings and insider filings.

Bitget's data service exposes two tools, ``guide`` (the catalog: five categories, 67 entries on
2026-09-24) and ``do_query``, which executes one entry. **``do_query`` takes ``entry_id``**; the
catalog lists the same field as ``id``, and a call with ``id`` is refused with a pydantic
validation error (recorded in ``tests/fixtures/sources/data_session.json``). Every reply is the
envelope ``{"success", "status_code", "data": {"results": [...], "provider", ...}, "error"}``;
``status_code: 204`` with ``data: ""`` means the entry had nothing for these arguments.

Entries used, with what was measured about each on 2026-09-24 (fixtures in
``tests/fixtures/sources/``):

* ``crypto_sentiment_crypto_fear_greed``: daily rows, newest first, ``value`` an integer 0-100 and
  ``classification`` a label; ``provider: feargreed``. Its values equal alternative.me's published
  index on the same days, and alternative.me is also what bitget-signal's ``sentiment_index``
  wraps (its error key is ``alt_me_error``). **So the two crypto Fear & Greed sources share one
  upstream**: their agreement checks the two Bitget services' plumbing, not two opinions.
* ``sentiment_market_fear_greed``: one row, ``score`` a float 0-100, ``rating`` a lower-case label,
  ``timestamp`` intraday; the equity-market index (``provider: feargreed``).
* ``crypto_futures_long_short_ratio``, ``..._top_account_ratio``, ``..._top_position_ratio``,
  ``crypto_futures_open_interest_history``, ``crypto_futures_taker_volume``: Binance futures
  positioning (``exchange: binance`` on every row), hourly with ``interval=1h``. **They answer for
  the US-equity perps too**: Binance lists ``NVDAUSDT`` and the entries returned its long/short,
  top-trader and open-interest series. ``exchange=bitget`` is accepted as well, but its newest
  BTCUSDT row was 8 hours old and its NVDAUSDT row 20 days old, so Binance's crowd is read.
  Row order differs by entry (oldest first for ratios, newest first for taker volume), so the
  newest row is chosen by its ``time`` (epoch ms), never by position.
* ``crypto_futures_funding_rate``: 4-hour rows whose ``funding_rate`` is **in percent**. Pinned
  against Binance's own ``/fapi/v1/fundingRate``: the row for 2026-06-26T12:00Z carries
  ``0.007308`` with ``funding_timestamp`` 1782489600000, and Binance settled ``0.00007308`` at that
  time. :data:`FUNDING_PERCENT` converts to the fraction Bitget's own tickers use.
* ``crypto_futures_taker_volume`` refuses a call without ``exchange`` (HTTP 400 "Parameter
  [exchange] is required"), so it is always sent.
* ``equity_calendar``: quarterly rows with ``perf_brief_dsclsr_date`` (confirmed) and
  ``perf_briefing_fore_dsclsr_date`` (forecast) and a session tag ``is_trading_time`` (``盘后``,
  after the close). **These dates run one day early.** Checked against the actual announcement
  dates of five of the eleven names: NVDA (vendor 2025-11-18, reported 2025-11-19; also 2025-08-26
  / 08-27, 2025-05-27 / 05-28, 2025-02-25 / 02-26, 2024-11-19 / 11-20), AAPL (2025-10-29 /
  10-30), TSLA (2025-10-21 / 10-22), META (2025-10-28 / 10-29) and COIN (2025-10-29 / 10-30). Every
  pair differs by exactly one day; NVDA's vendor dates all fall on Tuesdays although NVDA reports
  on Wednesdays. :data:`VENDOR_DATE_LAG` adds the day back, and every item says it did. The
  actual dates are the companies' own announcement dates as known to the author (NOT VERIFIED
  against a second vendor in this build); the day-of-week pattern is the independent check.
* ``equity_ownership_insider_trading``: SEC insider filings, ``filing_date`` (EDGAR, US Eastern),
  ``transaction_type`` (the SEC's Form 4 transaction code, or null for a holdings-only row),
  ``filing_url`` (the EDGAR index page). The form type is not given per row; the entry's name says
  insider trading, and the rows are read as Form 4 filings on that basis (NOT VERIFIED per row).

Provenance
----------
The session handling, ``entry_id`` and the envelope reading are taken from ARGUS
``argus/src/argus/market/bitget_mcp.py`` (MIT, Copyright (c) 2026 Pratiikpy, the same author) at
commit ``3dec6baf9dfa7be37c7b452e26a9b139df252f75``, sha256
``bc8f550f718cb6a40339439ed2233505d9845317f1a94d13a702ba5ae8905721`` (licence and pins:
``third_party/argus/PROVENANCE.md``), which also found ``equity_calendar`` (not
``equity_calendar_earnings``) by listing the catalog and the confirmed/forecast date fields.
Copied, never imported. New here: freshness limits, the funding unit, the vendor date offset, the
session times, and the derivatives entries.
"""

import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, time, timedelta
from types import MappingProxyType
from typing import Any, Final
from zoneinfo import ZoneInfo

from sentiment_agent.sources.mcp_http import (
    Invocation,
    McpError,
    disabled_call,
    finite,
    first_present,
    freshness_problem,
    hollow,
    invoke,
    newest,
    row_time,
    source_call,
    tool_payload,
)
from sentiment_agent.types import (
    CalendarItem,
    Clock,
    DerivativesReading,
    McpCaller,
    SourceCall,
    SourceHealth,
    ToolkitSurface,
)

SURFACE: Final = ToolkitSurface.DATA_MCP
NEW_YORK: Final = ZoneInfo("America/New_York")

UNDERLYING: Final[Mapping[str, str]] = MappingProxyType(
    {
        "MSTRUSDT": "MSTR",
        "HOODUSDT": "HOOD",
        "CRCLUSDT": "CRCL",
        "SNDKUSDT": "SNDK",
        "COINUSDT": "COIN",
        "TSLAUSDT": "TSLA",
        "GOOGLUSDT": "GOOGL",
        "METAUSDT": "META",
        "NVDAUSDT": "NVDA",
        "AMZNUSDT": "AMZN",
        "AAPLUSDT": "AAPL",
    }
)
"""The eleven US-equity perps of the policy universe and the stock each tracks. The venue's naming
is the mapping (ticker + ``USDT``); all eleven tickers answered ``equity_calendar`` on
2026-09-24."""

ENTRY_CRYPTO_FEAR_GREED: Final = "crypto_sentiment_crypto_fear_greed"
ENTRY_MARKET_FEAR_GREED: Final = "sentiment_market_fear_greed"
ENTRY_LONG_SHORT: Final = "crypto_futures_long_short_ratio"
ENTRY_TOP_ACCOUNT: Final = "crypto_futures_long_short_top_account_ratio"
ENTRY_TOP_POSITION: Final = "crypto_futures_long_short_top_position_ratio"
ENTRY_TAKER: Final = "crypto_futures_taker_volume"
ENTRY_OPEN_INTEREST: Final = "crypto_futures_open_interest_history"
ENTRY_FUNDING: Final = "crypto_futures_funding_rate"
ENTRY_EARNINGS: Final = "equity_calendar"
ENTRY_INSIDER: Final = "equity_ownership_insider_trading"

USED_ENTRIES: Final = (
    ENTRY_CRYPTO_FEAR_GREED,
    ENTRY_MARKET_FEAR_GREED,
    ENTRY_LONG_SHORT,
    ENTRY_TOP_ACCOUNT,
    ENTRY_TOP_POSITION,
    ENTRY_TAKER,
    ENTRY_OPEN_INTEREST,
    ENTRY_FUNDING,
    ENTRY_EARNINGS,
    ENTRY_INSIDER,
)

EXCHANGE: Final = "binance"
"""Whose crowd is read. The service's default, sent explicitly so a default change is visible."""

INTERVAL: Final = "1h"
RATIO_ROWS: Final = "3"
OPEN_INTEREST_ROWS: Final = "48"
"""Two days of hourly points: enough for a 24-hour change with a missing hour to spare."""
FUNDING_ROWS: Final = "6"
FEAR_GREED_ROWS: Final = "3"
INSIDER_ROWS: Final = "20"
EARNINGS_LOOKBACK_DAYS: Final = 120
"""``start_date`` for ``equity_calendar``: filters on the fiscal period end, and companies report
three to seven weeks after it, so 120 days keeps the current quarter's upcoming row in the reply."""

HOURLY_MAX_AGE: Final = timedelta(hours=3)
FUNDING_MAX_AGE: Final = timedelta(hours=12)
"""Funding rows are 4-hourly; three periods."""
CRYPTO_FEAR_GREED_MAX_AGE: Final = timedelta(hours=48)
MARKET_FEAR_GREED_MAX_AGE: Final = timedelta(hours=96)
"""The equity-market index does not move while US markets are shut; a Friday reading is still the
latest on Monday morning, and a holiday adds a day."""

FUNDING_PERCENT: Final = 100.0
"""``crypto_futures_funding_rate`` reports percent; divided by this to give a fraction."""

VENDOR_DATE_LAG: Final = timedelta(days=1)
"""The measured offset of ``equity_calendar``'s announcement dates (module docstring)."""

SESSION_AFTER_CLOSE: Final = "盘后"
SESSION_BEFORE_OPEN: Final = "盘前"
SESSION_INTRADAY: Final = "盘中"
_SESSIONS: Final[Mapping[str, tuple[time, str]]] = {
    SESSION_AFTER_CLOSE: (time(16, 0), "after the close"),
    SESSION_BEFORE_OPEN: (time(0, 0), "before the open"),
    SESSION_INTRADAY: (time(9, 30), "during the session"),
}
"""``at`` is the earliest moment the announcement can happen given the session tag, in New York:
after the close is 16:00, during the session 09:30, before the open (or untagged) the start of the
day. A risk decision should see the event no later than it can occur."""

_REPORT_TYPES: Final[Mapping[str, str]] = {
    "一季报": "Q1",
    "二季报": "Q2",
    "三季报": "Q3",
    "四季报": "Q4",
    "年报": "annual",
}

FORM4_CODES: Final[Mapping[str, str]] = {
    "P": "open-market purchase",
    "S": "open-market sale",
    "A": "grant or award",
    "D": "disposition to the issuer",
    "F": "shares withheld for exercise price or tax",
    "G": "gift",
    "M": "option exercise or conversion",
    "C": "conversion of a derivative",
    "X": "exercise of an in- or at-the-money derivative",
    "J": "other acquisition or disposition",
}
"""SEC Form 4 transaction codes (Form 4 General Instructions, Table of Transaction Codes). An
unlisted code is shown as the code itself, never guessed at."""


class _Reading:
    __slots__ = ("error", "health", "rows")

    def __init__(self, health: SourceHealth, rows: int = 0, error: str | None = None) -> None:
        self.health = health
        self.rows = rows
        self.error = error


def _source(entry_id: str) -> str:
    return f"do_query:{entry_id}"


def source_label(entry_id: str) -> str:
    """How a reading names where it came from, e.g. ``bitget_mcp_server:do_query:...``."""
    return f"{SURFACE.value}:{_source(entry_id)}"


def _day(value: Any) -> date | None:
    if not isinstance(value, str) or len(value.strip()) < 10:
        return None
    try:
        return date.fromisoformat(value.strip()[:10])
    except ValueError:
        return None


def _new_york_midnight(day: date) -> datetime:
    return datetime.combine(day, time(0, 0), tzinfo=NEW_YORK).astimezone(UTC)


class BitgetDataService:
    """Typed readings from bitget-mcp-server. No method raises for anything the server does,
    except :meth:`catalog`, which raises :class:`McpError` so a failed catalog is never read as an
    empty one."""

    def __init__(self, mcp: McpCaller, clock: Clock, *, max_workers: int = 6) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        self._mcp = mcp
        self._clock = clock
        self._workers = max_workers
        self._catalog_lock = threading.Lock()

    @property
    def server(self) -> str:
        return self._mcp.server

    @property
    def clock(self) -> Clock:
        return self._clock

    # --- catalog --------------------------------------------------------------------------------

    def catalog(self) -> list[tuple[str, str]]:
        """Every catalog entry as ``(entry_id, category)``, in the service's order."""
        with self._catalog_lock:
            top = self._guide({})
            categories = top.get("categories")
            if not isinstance(categories, list) or not categories:
                raise McpError(f"{self.server}: guide returned no categories")
            out: list[tuple[str, str]] = []
            for category in categories:
                key = category.get("key") if isinstance(category, dict) else None
                if not isinstance(key, str) or not key:
                    continue
                listing = self._guide({"category": key})
                entries = listing.get("entries")
                if not isinstance(entries, list):
                    raise McpError(f"{self.server}: guide({key}) returned no entry list")
                for entry in entries:
                    entry_id = entry.get("id") if isinstance(entry, dict) else None
                    if isinstance(entry_id, str) and entry_id:
                        out.append((entry_id, key))
            return out

    def _guide(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        result = self._mcp.call_tool("guide", arguments)
        if result.is_error:
            raise McpError(f"{self.server}: guide failed: {result.text[:200]}")
        try:
            payload = tool_payload(result)
        except ValueError as exc:
            raise McpError(f"{self.server}: guide reply is not JSON") from exc
        if not isinstance(payload, dict):
            raise McpError(f"{self.server}: guide reply is not an object")
        return payload

    # --- generic query --------------------------------------------------------------------------

    def _query(
        self, entry_id: str, params: Mapping[str, str]
    ) -> tuple[list[dict[str, Any]], Invocation, _Reading | None]:
        invocation = invoke(
            self._mcp, self._clock, "do_query", {"entry_id": entry_id, "params": dict(params)}
        )
        if invocation.failure is not None:
            return [], invocation, _Reading(invocation.failure, error=invocation.error)
        result = invocation.result
        if result is None:
            return [], invocation, _Reading(SourceHealth.ERROR, error="no result")
        if result.is_error:
            return (
                [],
                invocation,
                _Reading(SourceHealth.ERROR, error=f"tool reported an error: {result.text[:300]}"),
            )
        try:
            payload = tool_payload(result)
        except ValueError:
            return (
                [],
                invocation,
                _Reading(
                    SourceHealth.ERROR, error=f"reply text is not JSON: {result.text[:200]!r}"
                ),
            )
        rows, failed = _envelope(payload)
        return rows, invocation, failed

    def _record(
        self,
        entry_id: str,
        params: Mapping[str, str],
        invocation: Invocation,
        reading: _Reading,
    ) -> SourceCall:
        return source_call(
            surface=SURFACE,
            source=_source(entry_id),
            params=params,
            invocation=invocation,
            health=reading.health,
            rows=reading.rows,
            error=reading.error,
        )

    def query(self, entry_id: str, **params: str) -> tuple[list[dict[str, Any]], SourceCall]:
        """Execute one catalog entry. Rows are returned as the service sent them; the health says
        whether they carry anything (``HOLLOW`` when every value is empty, zero or an error)."""
        if not entry_id.strip():
            raise ValueError("entry_id must name a catalog entry")
        rows, invocation, failed = self._query(entry_id, params)
        if failed is not None:
            return [], self._record(entry_id, params, invocation, failed)
        if hollow(rows):
            reading = _Reading(SourceHealth.HOLLOW, len(rows), "every row is empty, zero or error")
            return rows, self._record(entry_id, params, invocation, reading)
        return rows, self._record(
            entry_id, params, invocation, _Reading(SourceHealth.OK, len(rows))
        )

    # --- Fear & Greed ---------------------------------------------------------------------------

    def _fear_greed(
        self,
        entry_id: str,
        params: Mapping[str, str],
        value_key: str,
        label_key: str,
        max_age: timedelta,
    ) -> tuple[int | None, str | None, SourceCall]:
        rows, invocation, failed = self._query(entry_id, params)
        if failed is not None:
            return None, None, self._record(entry_id, params, invocation, failed)
        valued = [row for row in rows if _scale_0_100(row.get(value_key)) is not None]
        latest = newest(valued)
        if latest is None:
            keys = sorted({str(k) for row in rows for k in row})[:12]
            reading = _Reading(
                SourceHealth.HOLLOW,
                len(rows),
                f"no timestamped 0-100 '{value_key}' in the rows (pinned scale); keys: {keys}",
            )
            return None, None, self._record(entry_id, params, invocation, reading)
        row, moment = latest
        stale = freshness_problem(moment, self._clock.now(), max_age)
        if stale is not None:
            reading = _Reading(SourceHealth.HOLLOW, len(rows), stale)
            return None, None, self._record(entry_id, params, invocation, reading)
        label = row.get(label_key)
        text = str(label).strip() if label is not None and str(label).strip() else None
        call = self._record(entry_id, params, invocation, _Reading(SourceHealth.OK, len(rows)))
        return _scale_0_100(row.get(value_key)), text, call

    def crypto_fear_greed(self) -> tuple[int | None, str | None, SourceCall]:
        """The crypto Fear & Greed index (daily), 0-100, and its label."""
        return self._fear_greed(
            ENTRY_CRYPTO_FEAR_GREED,
            {"limit": FEAR_GREED_ROWS},
            "value",
            "classification",
            CRYPTO_FEAR_GREED_MAX_AGE,
        )

    def market_fear_greed(self) -> tuple[int | None, str | None, SourceCall]:
        """The equity-market Fear & Greed index, 0-100 (rounded half up), and its rating."""
        return self._fear_greed(
            ENTRY_MARKET_FEAR_GREED, {}, "score", "rating", MARKET_FEAR_GREED_MAX_AGE
        )

    # --- crowd positioning ----------------------------------------------------------------------

    def _latest_value(
        self,
        entry_id: str,
        params: Mapping[str, str],
        value_of: Callable[[Mapping[str, Any]], float | None],
        max_age: timedelta,
    ) -> tuple[float | None, SourceCall]:
        rows, invocation, failed = self._query(entry_id, params)
        if failed is not None:
            return None, self._record(entry_id, params, invocation, failed)
        valued = [row for row in rows if value_of(row) is not None]
        latest = newest(valued)
        if latest is None:
            keys = sorted({str(k) for row in rows for k in row})[:12]
            reading = _Reading(
                SourceHealth.HOLLOW, len(rows), f"no timestamped value in the rows; keys: {keys}"
            )
            return None, self._record(entry_id, params, invocation, reading)
        row, moment = latest
        stale = freshness_problem(moment, self._clock.now(), max_age)
        if stale is not None:
            return None, self._record(
                entry_id, params, invocation, _Reading(SourceHealth.HOLLOW, len(rows), stale)
            )
        call = self._record(entry_id, params, invocation, _Reading(SourceHealth.OK, len(rows)))
        return value_of(row), call

    def _open_interest(
        self, params: Mapping[str, str]
    ) -> tuple[tuple[tuple[datetime, float], ...], SourceCall]:
        rows, invocation, failed = self._query(ENTRY_OPEN_INTEREST, params)
        if failed is not None:
            return (), self._record(ENTRY_OPEN_INTEREST, params, invocation, failed)
        points: dict[datetime, float] = {}
        for row in rows:
            moment = row_time(row)
            value = finite(row.get("open_interest"))
            if moment is not None and value is not None and value > 0:
                points[moment] = value
        series = tuple(sorted(points.items()))
        if not series:
            reading = _Reading(SourceHealth.HOLLOW, len(rows), "no timestamped open_interest")
            return (), self._record(ENTRY_OPEN_INTEREST, params, invocation, reading)
        stale = freshness_problem(series[-1][0], self._clock.now(), HOURLY_MAX_AGE)
        if stale is not None:
            reading = _Reading(SourceHealth.HOLLOW, len(rows), stale)
            return (), self._record(ENTRY_OPEN_INTEREST, params, invocation, reading)
        call = self._record(
            ENTRY_OPEN_INTEREST, params, invocation, _Reading(SourceHealth.OK, len(rows))
        )
        return series, call

    def derivatives(self, symbol: str) -> tuple[DerivativesReading, tuple[SourceCall, ...]]:
        """Binance futures crowd positioning for one symbol, every entry called concurrently.

        Returns the calls in a fixed order (long/short, top account, top position, taker, open
        interest, funding) whatever order they finished in.
        """
        if not symbol.strip():
            raise ValueError("symbol must name an instrument")
        ratio = {"symbol": symbol, "interval": INTERVAL, "limit": RATIO_ROWS, "exchange": EXCHANGE}
        taker = {"symbol": symbol, "interval": INTERVAL, "limit": RATIO_ROWS, "exchange": EXCHANGE}
        oi = {
            "symbol": symbol,
            "interval": INTERVAL,
            "limit": OPEN_INTEREST_ROWS,
            "exchange": EXCHANGE,
        }
        funding = {"symbol": symbol, "limit": FUNDING_ROWS, "exchange": EXCHANGE}
        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            f_ls = pool.submit(
                self._latest_value, ENTRY_LONG_SHORT, ratio, _ratio_value, HOURLY_MAX_AGE
            )
            f_ta = pool.submit(
                self._latest_value, ENTRY_TOP_ACCOUNT, ratio, _ratio_value, HOURLY_MAX_AGE
            )
            f_tp = pool.submit(
                self._latest_value, ENTRY_TOP_POSITION, ratio, _ratio_value, HOURLY_MAX_AGE
            )
            f_tk = pool.submit(self._latest_value, ENTRY_TAKER, taker, _taker_value, HOURLY_MAX_AGE)
            f_oi = pool.submit(self._open_interest, oi)
            f_fr = pool.submit(
                self._latest_value, ENTRY_FUNDING, funding, _funding_fraction, FUNDING_MAX_AGE
            )
            long_short, c_ls = f_ls.result()
            top_account, c_ta = f_ta.result()
            top_position, c_tp = f_tp.result()
            taker_ratio, c_tk = f_tk.result()
            series, c_oi = f_oi.result()
            funding_rate, c_fr = f_fr.result()
        reading = DerivativesReading(
            symbol=symbol,
            retail_long_short_ratio=long_short,
            top_trader_account_ratio=top_account,
            top_trader_position_ratio=top_position,
            taker_buy_sell_ratio=taker_ratio,
            open_interest_history=series,
            funding_rate=funding_rate,
        )
        return reading, (c_ls, c_ta, c_tp, c_tk, c_oi, c_fr)

    # --- calendar -------------------------------------------------------------------------------

    def _ticker(self, symbol: str) -> str | None:
        if symbol in UNDERLYING:
            return UNDERLYING[symbol]
        return None

    def earnings(self, symbol: str) -> tuple[tuple[CalendarItem, ...], SourceCall]:
        """Earnings announcements of the stock behind an equity perp, past and scheduled, one item
        per announcement day, oldest first. A symbol with no underlying stock is not called
        (``DISABLED``)."""
        ticker = self._ticker(symbol)
        if ticker is None:
            return (), disabled_call(
                surface=SURFACE,
                source=_source(ENTRY_EARNINGS),
                params={"symbol": symbol},
                clock=self._clock,
                reason=f"{symbol} is not one of the eleven equity perps; it has no earnings",
            )
        today = self._clock.now().astimezone(NEW_YORK).date()
        start = (today - timedelta(days=EARNINGS_LOOKBACK_DAYS)).isoformat()
        params = {"symbol": ticker, "start_date": start}
        rows, invocation, failed = self._query(ENTRY_EARNINGS, params)
        if failed is not None:
            return (), self._record(ENTRY_EARNINGS, params, invocation, failed)
        items = _earnings_items(symbol, ticker, rows)
        if not items:
            reading = _Reading(
                SourceHealth.HOLLOW, len(rows), "no row carries an announcement date"
            )
            return (), self._record(ENTRY_EARNINGS, params, invocation, reading)
        call = self._record(
            ENTRY_EARNINGS, params, invocation, _Reading(SourceHealth.OK, len(rows))
        )
        return items, call

    def insider_filings(
        self, symbol: str, *, since: datetime
    ) -> tuple[tuple[CalendarItem, ...], SourceCall]:
        """Insider (Form 4) filings of the stock behind an equity perp, filed on or after the New
        York date of ``since`` (EDGAR dates carry no time), oldest first."""
        if since.tzinfo is None:
            raise ValueError("since must be timezone-aware")
        ticker = self._ticker(symbol)
        if ticker is None:
            return (), disabled_call(
                surface=SURFACE,
                source=_source(ENTRY_INSIDER),
                params={"symbol": symbol},
                clock=self._clock,
                reason=f"{symbol} is not one of the eleven equity perps; it has no insider filings",
            )
        params = {"symbol": ticker, "limit": INSIDER_ROWS}
        rows, invocation, failed = self._query(ENTRY_INSIDER, params)
        if failed is not None:
            return (), self._record(ENTRY_INSIDER, params, invocation, failed)
        first_day = since.astimezone(NEW_YORK).date()
        items: dict[tuple[str, str], CalendarItem] = {}
        undated = 0
        for row in rows:
            filed = _day(row.get("filing_date"))
            if filed is None:
                undated += 1
                continue
            if filed < first_day:
                continue
            item = _insider_item(symbol, ticker, filed, row)
            items.setdefault((item.title, item.url or ""), item)
        ordered = tuple(sorted(items.values(), key=lambda i: (i.at or since, i.title)))
        note = f"{undated} row(s) without a filing date skipped" if undated else None
        if undated == len(rows):
            reading = _Reading(SourceHealth.HOLLOW, len(rows), note)
            return (), self._record(ENTRY_INSIDER, params, invocation, reading)
        call = self._record(
            ENTRY_INSIDER, params, invocation, _Reading(SourceHealth.OK, len(rows), note)
        )
        return ordered, call


# ================================================================================================
# Row interpretation
# ================================================================================================


def _envelope(payload: Any) -> tuple[list[dict[str, Any]], _Reading | None]:
    """The rows of a ``do_query`` reply, or the reading that says why there are none."""
    if not isinstance(payload, dict):
        return [], _Reading(SourceHealth.ERROR, error="reply is not the {success, data} envelope")
    status = payload.get("status_code")
    if payload.get("success") is not True:
        data = payload.get("data")
        detail = payload.get("error") or (data.get("detail") if isinstance(data, dict) else None)
        return [], _Reading(
            SourceHealth.ERROR, error=f"service reported failure (status {status}): {detail}"
        )
    data = payload.get("data")
    if status == 204 or data in (None, "", {}):
        return [], _Reading(SourceHealth.EMPTY, error=f"answered with no data (status {status})")
    if not isinstance(data, dict):
        return [], _Reading(SourceHealth.ERROR, error=f"unexpected data type {type(data).__name__}")
    results = data.get("results")
    if isinstance(results, dict):
        # Snapshot entries (tickers, order books) answer one object, not a list (live probe,
        # 2026-09-24: crypto_spot_ticker, crypto_futures_order_book). That is one row of data.
        results = [results]
    if not isinstance(results, list):
        return [], _Reading(SourceHealth.ERROR, error="the data object carries no results")
    rows = [dict(row) for row in results if isinstance(row, dict)]
    if not rows:
        return [], _Reading(SourceHealth.EMPTY, error="answered with no rows")
    return rows, None


def _scale_0_100(value: Any) -> int | None:
    number = finite(value)
    if number is None or not 0 <= number <= 100:
        return None
    return int(number + 0.5)


def _ratio_value(row: Mapping[str, Any]) -> float | None:
    ratio = finite(row.get("long_short_ratio"))
    if ratio is None:
        longs = finite(row.get("long_account"))
        shorts = finite(row.get("short_account"))
        if longs is not None and shorts is not None and shorts > 0 and longs >= 0:
            ratio = longs / shorts
    return ratio if ratio is not None and ratio > 0 else None


def _taker_value(row: Mapping[str, Any]) -> float | None:
    ratio = finite(row.get("buy_sell_ratio"))
    if ratio is None:
        buys = finite(row.get("buy_vol"))
        sells = finite(row.get("sell_vol"))
        if buys is not None and sells is not None and sells > 0 and buys >= 0:
            ratio = buys / sells
    return ratio if ratio is not None and ratio > 0 else None


def _funding_fraction(row: Mapping[str, Any]) -> float | None:
    percent = finite(row.get("funding_rate"))
    return None if percent is None else percent / FUNDING_PERCENT


def _session(tag: Any) -> tuple[time, str]:
    if isinstance(tag, str) and tag in _SESSIONS:
        return _SESSIONS[tag]
    return time(0, 0), "session not given"


def _earnings_items(
    symbol: str, ticker: str, rows: Sequence[Mapping[str, Any]]
) -> tuple[CalendarItem, ...]:
    """One item per announcement day. A day confirmed by any row is confirmed; the quarterly label
    wins over the annual one when both rows describe the same announcement.

    A forecast-only day is dropped when the same fiscal period has a confirmed day elsewhere: a
    rescheduled announcement leaves its old forecast row behind, and read literally that row is a
    phantom event that would wake the agent on the wrong day."""
    days: dict[date, dict[str, Any]] = {}
    for row in rows:
        confirmed_raw = row.get("perf_brief_dsclsr_date")
        forecast_raw = row.get("perf_briefing_fore_dsclsr_date")
        raw = confirmed_raw if _day(confirmed_raw) is not None else forecast_raw
        vendor_day = _day(raw)
        if vendor_day is None:
            continue
        actual = vendor_day + VENDOR_DATE_LAG
        entry = days.setdefault(
            actual,
            {
                "confirmed": False,
                "labels": set(),
                "fiscal": None,
                "period": None,
                "vendor": raw,
                "session": None,
                "periods": set(),
            },
        )
        entry["confirmed"] = entry["confirmed"] or _day(confirmed_raw) is not None
        label = _REPORT_TYPES.get(str(row.get("report_type_name") or ""))
        if label:
            entry["labels"].add(label)
            # The annual report and the fourth quarter are one announcement.
            entry["periods"].add(
                (str(row.get("fiscal_year")), "Q4" if label == "annual" else label)
            )
        entry["fiscal"] = entry["fiscal"] or row.get("fiscal_year")
        if _day(confirmed_raw) is not None or entry["period"] is None:
            entry["period"] = row.get("period_ending") or entry["period"]
        entry["session"] = entry["session"] or row.get("is_trading_time")
    confirmed_periods: set[tuple[str, str]] = set()
    for entry in days.values():
        if entry["confirmed"]:
            confirmed_periods |= entry["periods"]
    items: list[CalendarItem] = []
    for actual, entry in sorted(days.items()):
        periods: set[tuple[str, str]] = entry["periods"]
        if not entry["confirmed"] and periods and periods <= confirmed_periods:
            continue
        clock_time, session_text = _session(entry["session"])
        at = datetime.combine(actual, clock_time, tzinfo=NEW_YORK).astimezone(UTC)
        labels: set[str] = entry["labels"]
        quarter = next((q for q in ("Q1", "Q2", "Q3", "Q4") if q in labels), None)
        period_label = quarter or ("annual" if "annual" in labels else "report")
        fiscal = f"fiscal {entry['fiscal']} " if entry["fiscal"] else ""
        ending = f", period ending {entry['period']}" if entry["period"] else ""
        status = "confirmed" if entry["confirmed"] else "expected"
        title = (
            f"{ticker} earnings ({fiscal}{period_label}{ending}): {status} for {actual.isoformat()}"
            f" {session_text} (New York); vendor date {entry['vendor']} read one day later"
        )
        items.append(
            CalendarItem(
                symbol=symbol,
                kind="earnings",
                at=at,
                title=title,
                source=source_label(ENTRY_EARNINGS),
            )
        )
    return tuple(items)


def _number_text(value: float) -> str:
    return f"{value:,.0f}" if value == int(value) else f"{value:,.4f}".rstrip("0").rstrip(".")


def _insider_item(symbol: str, ticker: str, filed: date, row: Mapping[str, Any]) -> CalendarItem:
    owner = str(row.get("owner_name") or "unnamed insider").strip()
    role = str(first_present(row, ("owner_title", "ownership_type")) or "insider").strip()
    code = row.get("transaction_type")
    parts = [f"{ticker} insider filing {filed.isoformat()}: {owner} ({role})"]
    if isinstance(code, str) and code.strip():
        code = code.strip().upper()
        what = FORM4_CODES.get(code, f"transaction code {code}")
        shares = finite(row.get("securities_transacted"))
        price = finite(row.get("transaction_price"))
        text = f"{what} (code {code})"
        if shares is not None:
            text += f", {_number_text(shares)} shares"
        if price is not None and price > 0:
            text += f" at {price:,.2f}"
        traded = _day(row.get("transaction_date"))
        if traded is not None:
            text += f" on {traded.isoformat()}"
        parts.append(text)
    else:
        parts.append("no transaction reported (holdings row)")
    held = finite(row.get("securities_owned"))
    if held is not None:
        parts.append(f"{_number_text(held)} shares held after")
    link = row.get("filing_url")
    url = link.strip() if isinstance(link, str) and link.strip().startswith("http") else None
    return CalendarItem(
        symbol=symbol,
        kind="form4",
        at=_new_york_midnight(filed),
        title="; ".join(parts),
        source=source_label(ENTRY_INSIDER),
        url=url,
    )


__all__ = [
    "CRYPTO_FEAR_GREED_MAX_AGE",
    "EARNINGS_LOOKBACK_DAYS",
    "ENTRY_CRYPTO_FEAR_GREED",
    "ENTRY_EARNINGS",
    "ENTRY_FUNDING",
    "ENTRY_INSIDER",
    "ENTRY_LONG_SHORT",
    "ENTRY_MARKET_FEAR_GREED",
    "ENTRY_OPEN_INTEREST",
    "ENTRY_TAKER",
    "ENTRY_TOP_ACCOUNT",
    "ENTRY_TOP_POSITION",
    "EXCHANGE",
    "FORM4_CODES",
    "FUNDING_MAX_AGE",
    "FUNDING_PERCENT",
    "HOURLY_MAX_AGE",
    "MARKET_FEAR_GREED_MAX_AGE",
    "NEW_YORK",
    "SURFACE",
    "UNDERLYING",
    "USED_ENTRIES",
    "VENDOR_DATE_LAG",
    "BitgetDataService",
    "source_label",
]
