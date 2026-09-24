"""bitget-mcp-server readings, replayed from the recorded session: the catalog, entry_id, Fear &
Greed scales, crowd positioning (crypto and equity perps), the funding unit, earnings dates with
the vendor's measured one-day offset, insider filings, freshness, and every refusal."""

from collections.abc import Mapping
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

import pytest

from sentiment_agent.clock import ManualClock, SystemClock
from sentiment_agent.sources.bitget_data import (
    ENTRY_EARNINGS,
    ENTRY_INSIDER,
    EXCHANGE,
    FUNDING_PERCENT,
    NEW_YORK,
    UNDERLYING,
    USED_ENTRIES,
    VENDOR_DATE_LAG,
    BitgetDataService,
    _earnings_items,
)
from sentiment_agent.sources.mcp_http import DATA_MCP_URL, StreamableHttpMcp, parse_instant
from sentiment_agent.types import SourceHealth, ToolkitSurface
from sources.replay import RecordedHttp, Session, call_key, load_json, load_session

DATA = load_session("data_session.json")
UPSTREAM = load_json("upstream_shapes.json")


def service(
    session: Session = DATA, *, at: datetime | None = None, http: RecordedHttp | None = None
) -> tuple[BitgetDataService, RecordedHttp]:
    replay = http if http is not None else RecordedHttp(session)
    clock = ManualClock(at if at is not None else session.captured_at)
    mcp = StreamableHttpMcp(session.url, server_label=session.server, clock=clock, http=replay)
    return BitgetDataService(mcp, clock), replay


def recorded_rows(entry_id: str, **params: str) -> list[dict[str, Any]]:
    payload = DATA.payload(call_key("do_query", {"entry_id": entry_id, "params": params}))
    rows: list[dict[str, Any]] = payload["data"]["results"]
    return rows


def newest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda r: r["time"])


def ratio_params(symbol: str) -> dict[str, str]:
    return {"symbol": symbol, "interval": "1h", "limit": "3", "exchange": EXCHANGE}


# ================================================================================================
# Catalog and the do_query contract
# ================================================================================================


def test_catalog_lists_every_entry_and_every_used_entry_exists() -> None:
    data, _ = service()
    catalog = data.catalog()
    guide = DATA.payload(call_key("guide", {}))
    expected = {c["key"]: c["entry_count"] for c in guide["categories"]}
    counts: dict[str, int] = {}
    for _, category in catalog:
        counts[category] = counts.get(category, 0) + 1
    assert counts == expected
    assert len(catalog) == sum(expected.values()) == 67
    assert set(USED_ENTRIES) <= {entry for entry, _ in catalog}


def test_do_query_is_sent_with_entry_id_never_id() -> None:
    data, http = service()
    data.market_fear_greed()
    calls = [m for _, m, _ in http.requests if m.get("method") == "tools/call"]
    arguments = calls[0]["params"]["arguments"]
    assert calls[0]["params"]["name"] == "do_query"
    assert arguments["entry_id"] == "sentiment_market_fear_greed"
    assert "id" not in arguments
    # and the recorded refusal shows why: the catalog's `id` key is not do_query's argument
    refused = DATA.exchanges[call_key("do_query", {"id": "crypto_sentiment_crypto_fear_greed"})]
    assert '"isError":true' in refused.body.replace(" ", "")
    assert "Missing required argument" in refused.body


# ================================================================================================
# Fear & Greed: fields and 0-100 scales pinned by the recording
# ================================================================================================


def test_crypto_fear_greed_pinned_fields_and_scale() -> None:
    rows = recorded_rows("crypto_sentiment_crypto_fear_greed", limit="3")
    today = newest(rows)
    assert set(today) >= {"value", "classification", "date", "time"}
    assert isinstance(today["value"], int)
    assert 0 <= today["value"] <= 100
    data, _ = service()
    value, label, call = data.crypto_fear_greed()
    assert value == today["value"]
    assert label == today["classification"]
    assert call.health is SourceHealth.OK
    assert call.source == "do_query:crypto_sentiment_crypto_fear_greed"
    assert call.surface is ToolkitSurface.DATA_MCP
    assert call.params == {"limit": "3"}


def test_market_fear_greed_pinned_fields_and_scale() -> None:
    row = newest(recorded_rows("sentiment_market_fear_greed"))
    assert set(row) >= {"score", "rating", "timestamp"}
    assert isinstance(row["score"], float)
    assert 0 <= row["score"] <= 100
    data, _ = service()
    value, label, call = data.market_fear_greed()
    assert value == int(row["score"] + 0.5)
    assert label == row["rating"]
    assert call.health is SourceHealth.OK


def test_fear_greed_staleness() -> None:
    today = parse_instant(
        newest(recorded_rows("crypto_sentiment_crypto_fear_greed", limit="3"))["time"]
    )
    assert today is not None
    data, _ = service(at=today + timedelta(hours=47))
    assert data.crypto_fear_greed()[2].health is SourceHealth.OK
    stale, _ = service(at=today + timedelta(hours=49))
    value, _, call = stale.crypto_fear_greed()
    assert value is None
    assert call.health is SourceHealth.HOLLOW
    assert call.error is not None
    assert "stale" in call.error


# ================================================================================================
# Crowd positioning
# ================================================================================================


@pytest.mark.parametrize("symbol", ["BTCUSDT", "NVDAUSDT"])
def test_derivatives_read_the_newest_rows(symbol: str) -> None:
    data, _ = service()
    reading, calls = data.derivatives(symbol)
    assert [c.source for c in calls] == [
        "do_query:crypto_futures_long_short_ratio",
        "do_query:crypto_futures_long_short_top_account_ratio",
        "do_query:crypto_futures_long_short_top_position_ratio",
        "do_query:crypto_futures_taker_volume",
        "do_query:crypto_futures_open_interest_history",
        "do_query:crypto_futures_funding_rate",
    ]
    assert all(c.health is SourceHealth.OK for c in calls)
    params = ratio_params(symbol)
    ls = newest(recorded_rows("crypto_futures_long_short_ratio", **params))
    assert reading.retail_long_short_ratio == pytest.approx(ls["long_short_ratio"])
    top = newest(recorded_rows("crypto_futures_long_short_top_account_ratio", **params))
    assert reading.top_trader_account_ratio == pytest.approx(top["long_short_ratio"])
    pos = newest(recorded_rows("crypto_futures_long_short_top_position_ratio", **params))
    assert reading.top_trader_position_ratio == pytest.approx(pos["long_short_ratio"])
    taker = newest(recorded_rows("crypto_futures_taker_volume", **params))
    assert reading.taker_buy_sell_ratio == pytest.approx(taker["buy_sell_ratio"])
    oi_rows = recorded_rows(
        "crypto_futures_open_interest_history",
        symbol=symbol,
        interval="1h",
        limit="48",
        exchange=EXCHANGE,
    )
    assert len(reading.open_interest_history) == len({r["time"] for r in oi_rows})
    times = [t for t, _ in reading.open_interest_history]
    assert times == sorted(times)
    assert reading.open_interest_history[-1][1] == pytest.approx(newest(oi_rows)["open_interest"])
    funding = newest(
        recorded_rows("crypto_futures_funding_rate", symbol=symbol, limit="6", exchange=EXCHANGE)
    )
    assert reading.funding_rate == pytest.approx(funding["funding_rate"] / FUNDING_PERCENT)
    assert all(r["exchange"] == "binance" for r in oi_rows)


def test_row_order_differs_by_entry_and_is_not_relied_on() -> None:
    params = ratio_params("BTCUSDT")
    ratios = recorded_rows("crypto_futures_long_short_ratio", **params)
    taker = recorded_rows("crypto_futures_taker_volume", **params)
    assert ratios[0]["time"] < ratios[-1]["time"]  # oldest first
    assert taker[0]["time"] > taker[-1]["time"]  # newest first


def test_index_perp_has_no_binance_crowd_and_says_empty() -> None:
    data, _ = service()
    reading, calls = data.derivatives("SP500USDT")
    assert all(c.health is SourceHealth.EMPTY for c in calls)
    assert reading.retail_long_short_ratio is None
    assert reading.open_interest_history == ()
    assert reading.funding_rate is None


def test_funding_is_reported_in_percent() -> None:
    """Pinned against Binance's own settled rates. A 4-hour row whose bar ends at a settlement
    carries the rate predicted at the bar's close, in percent: within 0.0005 percentage points of
    100 times what Binance then settled, and exactly equal on at least one row of the recording.
    Read as a fraction instead, the same rows would be off by two orders of magnitude."""
    binance = {
        int(r["fundingTime"]) // 1000: float(r["fundingRate"])
        for r in UPSTREAM["replies"]["binance_funding_rate"]["payload"]
    }
    rows = recorded_rows(
        "crypto_futures_funding_rate", symbol="BTCUSDT", limit="6", exchange=EXCHANGE
    )
    as_percent: list[float] = []
    as_fraction: list[float] = []
    for row in rows:
        settles = int(row["funding_timestamp"]) // 1000
        if settles == row["time"] // 1000 + 4 * 3600 and settles in binance:
            as_percent.append(abs(row["funding_rate"] / FUNDING_PERCENT - binance[settles]))
            as_fraction.append(abs(row["funding_rate"] - binance[settles]))
    assert len(as_percent) >= 2
    assert max(as_percent) < 5e-6 < min(as_fraction)
    assert min(as_percent) < 1e-12


def test_stale_and_future_rows_are_refused() -> None:
    newest_oi = parse_instant(
        newest(
            recorded_rows(
                "crypto_futures_open_interest_history",
                symbol="BTCUSDT",
                interval="1h",
                limit="48",
                exchange=EXCHANGE,
            )
        )["time"]
    )
    assert newest_oi is not None
    late, _ = service(at=newest_oi + timedelta(hours=4))
    reading, calls = late.derivatives("BTCUSDT")
    by_source = {c.source: c for c in calls}
    assert by_source["do_query:crypto_futures_open_interest_history"].health is SourceHealth.HOLLOW
    assert reading.open_interest_history == ()
    assert by_source["do_query:crypto_futures_funding_rate"].health is SourceHealth.OK  # 12h limit
    early, _ = service(at=newest_oi - timedelta(hours=2))
    early_reading, early_calls = early.derivatives("BTCUSDT")
    assert early_reading.retail_long_short_ratio is None
    error = {c.source: c for c in early_calls}["do_query:crypto_futures_long_short_ratio"].error
    assert error is not None
    assert "ahead of now" in error


# ================================================================================================
# Earnings: the vendor's dates run one day early
# ================================================================================================

KNOWN_ANNOUNCEMENTS: Mapping[str, tuple[date, ...]] = {
    "NVDA": (
        date(2024, 11, 20),
        date(2025, 2, 26),
        date(2025, 5, 28),
        date(2025, 8, 27),
        date(2025, 11, 19),
    ),
    "AAPL": (date(2025, 1, 30), date(2025, 5, 1), date(2025, 7, 31), date(2025, 10, 30)),
    "TSLA": (date(2025, 1, 29), date(2025, 4, 22), date(2025, 7, 23), date(2025, 10, 22)),
    "META": (date(2025, 1, 29), date(2025, 4, 30), date(2025, 7, 30), date(2025, 10, 29)),
    "COIN": (date(2025, 2, 13), date(2025, 5, 8), date(2025, 7, 31), date(2025, 10, 30)),
}
"""The companies' own announcement dates, as known to the author (NOT VERIFIED against a second
vendor here). NVDA reports on Wednesdays, AAPL and COIN on Thursdays."""


@pytest.mark.parametrize("ticker", sorted(KNOWN_ANNOUNCEMENTS))
def test_vendor_brief_dates_are_one_day_before_the_announcement(ticker: str) -> None:
    rows = recorded_rows("equity_calendar", symbol=ticker, start_date="2024-10-01")
    vendor = {
        date.fromisoformat(r["perf_brief_dsclsr_date"])
        for r in rows
        if r.get("perf_brief_dsclsr_date")
    }
    for announced in KNOWN_ANNOUNCEMENTS[ticker]:
        assert announced - VENDOR_DATE_LAG in vendor, (ticker, announced)
        assert announced not in vendor, (ticker, announced)


@pytest.mark.parametrize("symbol", sorted(UNDERLYING))
def test_earnings_for_every_underlying(symbol: str) -> None:
    data, _ = service()
    items, call = data.earnings(symbol)
    assert call.health is SourceHealth.OK, call.error
    assert call.params["symbol"] == UNDERLYING[symbol]
    assert items
    for item in items:
        assert item.symbol == symbol
        assert item.kind == "earnings"
        assert item.at is not None
        assert "read one day later" in item.title
        assert item.source == "bitget_mcp_server:do_query:equity_calendar"
    days = [i.at for i in items if i.at is not None]
    assert days == sorted(days)
    assert len({d.astimezone(NEW_YORK).date() for d in days}) == len(days)  # one per day


def test_earnings_item_time_follows_the_session_tag() -> None:
    items, _ = service()[0].earnings("NVDAUSDT")
    latest = items[-1]
    rows = recorded_rows(
        ENTRY_EARNINGS, symbol="NVDA", start_date=_earnings_start(DATA.captured_at)
    )
    confirmed = max(
        date.fromisoformat(r["perf_brief_dsclsr_date"]) for r in rows if r["perf_brief_dsclsr_date"]
    )
    assert latest.at is not None
    local = latest.at.astimezone(NEW_YORK)
    assert local.date() == confirmed + VENDOR_DATE_LAG
    assert local.time() == time(16, 0)  # 盘后: after the close
    assert "confirmed" in latest.title


def _earnings_start(at: datetime) -> str:
    return (at.astimezone(NEW_YORK).date() - timedelta(days=120)).isoformat()


def test_session_tags_map_to_the_earliest_possible_moment() -> None:
    def row(tag: str | None) -> dict[str, Any]:
        return {
            "perf_brief_dsclsr_date": None,
            "perf_briefing_fore_dsclsr_date": "2026-10-27",
            "is_trading_time": tag,
            "report_type_name": "三季报",
            "fiscal_year": "2026",
            "period_ending": "2026-09-30",
        }

    expected = {"盘后": time(16, 0), "盘前": time(0, 0), "盘中": time(9, 30), None: time(0, 0)}
    for tag, moment in expected.items():
        (item,) = _earnings_items("XUSDT", "X", [row(tag)])
        assert item.at is not None
        local = item.at.astimezone(NEW_YORK)
        assert (local.date(), local.time()) == (date(2026, 10, 28), moment)
        assert "expected" in item.title
        assert "Q3" in item.title


def test_quarter_and_annual_rows_of_one_announcement_become_one_item() -> None:
    base = {
        "perf_brief_dsclsr_date": "2026-02-24",
        "perf_briefing_fore_dsclsr_date": "2026-02-24",
        "is_trading_time": "盘后",
        "fiscal_year": "2026",
        "period_ending": "2026-01-24",
    }
    items = _earnings_items(
        "NVDAUSDT",
        "NVDA",
        [{**base, "report_type_name": "年报"}, {**base, "report_type_name": "四季报"}],
    )
    assert len(items) == 1
    assert "Q4" in items[0].title


def test_a_superseded_forecast_is_not_a_second_event() -> None:
    def row(brief: str | None, forecast: str, report: str) -> dict[str, Any]:
        return {
            "perf_brief_dsclsr_date": brief,
            "perf_briefing_fore_dsclsr_date": forecast,
            "is_trading_time": "盘后",
            "report_type_name": report,
            "fiscal_year": "2026",
            "period_ending": "2026-09-30",
        }

    rows = [
        row(None, "2026-10-20", "三季报"),  # the old forecast, left behind by a reschedule
        row("2026-10-27", "2026-10-27", "三季报"),  # the confirmed date
        row(None, "2027-01-26", "四季报"),  # next quarter: only a forecast so far, kept
    ]
    items = _earnings_items("XUSDT", "X", rows)
    days = [i.at.astimezone(NEW_YORK).date() for i in items if i.at is not None]
    assert days == [date(2026, 10, 28), date(2027, 1, 27)]
    assert "confirmed" in items[0].title
    assert "expected" in items[1].title


def test_symbols_without_an_underlying_are_not_called() -> None:
    data, http = service()
    items, call = data.earnings("BTCUSDT")
    filings, filing_call = data.insider_filings("BTCUSDT", since=DATA.captured_at)
    assert items == filings == ()
    assert call.health is SourceHealth.DISABLED
    assert filing_call.health is SourceHealth.DISABLED
    assert http.requests == []


# ================================================================================================
# Insider filings
# ================================================================================================


def test_insider_filings_filtered_by_new_york_filing_date() -> None:
    rows = recorded_rows(ENTRY_INSIDER, symbol="NVDA", limit="20")
    since = DATA.captured_at - timedelta(days=30)
    data, _ = service()
    items, call = data.insider_filings("NVDAUSDT", since=since)
    assert call.health is SourceHealth.OK
    assert call.rows == len(rows)
    first_day = since.astimezone(NEW_YORK).date()
    expected_days = sorted(
        {
            date.fromisoformat(r["filing_date"])
            for r in rows
            if date.fromisoformat(r["filing_date"]) >= first_day
        }
    )
    assert expected_days
    assert sorted({i.at.astimezone(NEW_YORK).date() for i in items if i.at}) == expected_days
    for item in items:
        assert item.kind == "form4"
        assert item.symbol == "NVDAUSDT"
        assert item.url is not None
        assert item.url.startswith("https://www.sec.gov")
        assert item.at is not None
        assert item.at.astimezone(NEW_YORK).time() == time(0, 0)
    in_window_sales = [
        r
        for r in rows
        if r.get("transaction_type") == "S" and date.fromisoformat(r["filing_date"]) >= first_day
    ]
    assert in_window_sales  # the recording carries sales inside the window
    sales = [i for i in items if "open-market sale (code S)" in i.title]
    assert len(sales) == len(in_window_sales)
    assert all(" shares at " in i.title for i in sales)


def test_insider_filings_after_everything_is_ok_and_empty() -> None:
    data, _ = service()
    items, call = data.insider_filings("AAPLUSDT", since=DATA.captured_at + timedelta(days=5))
    assert items == ()
    assert call.health is SourceHealth.OK
    with pytest.raises(ValueError, match="timezone-aware"):
        data.insider_filings("AAPLUSDT", since=datetime(2026, 9, 1))  # noqa: DTZ001 - the point


# ================================================================================================
# Refusals and failures
# ================================================================================================


@pytest.mark.parametrize(
    ("entry", "params", "fragment"),
    [
        ("no_such_entry", {}, "Unknown entry_id"),
        ("news_label_search", {}, "Missing required param: 'label'"),
        ("crypto_futures_taker_volume", {"symbol": "BTCUSDT"}, "Parameter [exchange] is required"),
    ],
)
def test_recorded_refusals_are_errors(entry: str, params: dict[str, str], fragment: str) -> None:
    data, _ = service()
    rows, call = data.query(entry, **params)
    assert rows == []
    assert call.health is SourceHealth.ERROR
    assert call.error is not None
    assert fragment in call.error


def test_query_returns_rows_and_marks_hollow_rows() -> None:
    data, _ = service()
    rows, call = data.query("crypto_futures_long_short_ratio", **ratio_params("BTCUSDT"))
    assert rows
    assert call.health is SourceHealth.OK
    hollow_http = RecordedHttp(DATA)
    hollow_http.overrides[call_key("do_query", {"entry_id": "x", "params": {}})] = (
        200,
        {"content-type": "text/event-stream"},
        _sse(
            {
                "success": True,
                "status_code": 200,
                "data": {"results": [{"symbol": "", "error": ""}]},
            }
        ),
    )
    hollow_data, _ = service(http=hollow_http)
    _, hollow_call = hollow_data.query("x")
    assert hollow_call.health is SourceHealth.HOLLOW


def test_single_object_results_are_one_row() -> None:
    """Snapshot entries answer ``results`` as one object (live probe, 2026-09-24)."""
    ticker = {
        "symbol": "BTC/USDT",
        "exchange": "binance",
        "timestamp": "2026-09-24T11:54:54.006000Z",
        "last": 83445.7,
        "bid": 83445.69,
        "ask": 83445.7,
    }
    http = RecordedHttp(DATA)
    http.overrides[
        call_key("do_query", {"entry_id": "crypto_spot_ticker", "params": {"symbol": "BTCUSDT"}})
    ] = (
        200,
        {"content-type": "text/event-stream"},
        _sse({"success": True, "status_code": 200, "data": {"results": ticker}}),
    )
    data, _ = service(http=http)
    rows, call = data.query("crypto_spot_ticker", symbol="BTCUSDT")
    assert rows == [ticker]
    assert call.health is SourceHealth.OK
    assert call.rows == 1


def _sse(structured: dict[str, Any]) -> bytes:
    import json

    from sources.replay import sse_reply

    return sse_reply(
        {
            "content": [{"type": "text", "text": json.dumps(structured)}],
            "isError": False,
            "structuredContent": structured,
        }
    )


def test_transport_failure_and_timeout_become_health() -> None:
    def failing(message: dict[str, Any]) -> tuple[int, dict[str, str], bytes]:
        return 502, {}, b"bad gateway"

    http = RecordedHttp(DATA, default=failing)
    data, _ = service(http=http)
    _, call = data.query("unrecorded_entry")
    assert call.health is SourceHealth.ERROR
    assert call.error is not None
    assert "HTTP 502" in call.error

    def slow(url: str, body: bytes, headers: Mapping[str, str], timeout: float) -> Any:
        raise TimeoutError("slow")

    clock = ManualClock(DATA.captured_at)
    timed_out = BitgetDataService(
        StreamableHttpMcp(DATA_MCP_URL, server_label="x", clock=clock, http=slow), clock
    )
    reading, calls = timed_out.derivatives("BTCUSDT")
    assert {c.health for c in calls} == {SourceHealth.TIMEOUT}
    assert reading.retail_long_short_ratio is None
    assert timed_out.crypto_fear_greed()[2].health is SourceHealth.TIMEOUT
    assert timed_out.earnings("NVDAUSDT")[1].health is SourceHealth.TIMEOUT


def test_catalog_failure_raises_rather_than_reading_as_empty() -> None:
    from sentiment_agent.sources.mcp_http import McpError

    def failing(message: dict[str, Any]) -> tuple[int, dict[str, str], bytes]:
        return 500, {}, b""

    http = RecordedHttp(DATA, default=failing)
    http.overrides[call_key("guide", {})] = (500, {}, b"down")
    data, _ = service(http=http)
    with pytest.raises(McpError, match="HTTP 500"):
        data.catalog()


# ================================================================================================
# Live drift (keyless; run with --run-live-public)
# ================================================================================================


@pytest.fixture
def live() -> BitgetDataService:
    clock = SystemClock()
    return BitgetDataService(
        StreamableHttpMcp(DATA_MCP_URL, server_label="bitget-mcp-server", clock=clock), clock
    )


@pytest.mark.live_public
def test_live_fear_greed_fields_and_scales(live: BitgetDataService) -> None:
    crypto, crypto_label, crypto_call = live.crypto_fear_greed()
    market, market_label, market_call = live.market_fear_greed()
    assert crypto_call.health is SourceHealth.OK, crypto_call.error
    assert market_call.health is SourceHealth.OK, market_call.error
    assert crypto is not None
    assert 0 <= crypto <= 100
    assert crypto_label
    assert market is not None
    assert 0 <= market <= 100
    assert market_label


@pytest.mark.live_public
def test_live_btc_positioning_answers(live: BitgetDataService) -> None:
    reading, calls = live.derivatives("BTCUSDT")
    assert all(c.health is SourceHealth.OK for c in calls), [(c.source, c.error) for c in calls]
    assert reading.retail_long_short_ratio is not None
    assert len(reading.open_interest_history) >= 25
    assert reading.funding_rate is not None
    assert abs(reading.funding_rate) < 0.01  # a fraction; percent would be 100x larger


@pytest.mark.live_public
def test_live_calendar_answers_for_all_eleven(live: BitgetDataService) -> None:
    since = datetime.now(UTC) - timedelta(days=60)
    for symbol in UNDERLYING:
        _, call = live.earnings(symbol)
        assert call.health is SourceHealth.OK, (symbol, call.error)
        _, filings = live.insider_filings(symbol, since=since)
        assert filings.health in (SourceHealth.OK, SourceHealth.EMPTY), (symbol, filings.error)
