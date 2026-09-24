"""The facade (one ToolkitReader over both services) and the coverage probe, replayed from the
recorded sessions: which service answers each reading, the Fear & Greed agreement inputs, the
fallback rules, the calendar window, and that nothing ever raises."""

import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from sentiment_agent.clock import SystemClock
from sentiment_agent.sources.bitget_data import (
    ENTRY_CRYPTO_FEAR_GREED,
    EXCHANGE,
    NEW_YORK,
    USED_ENTRIES,
    BitgetDataService,
)
from sentiment_agent.sources.mcp_http import (
    DATA_MCP_URL,
    SIGNAL_MCP_URL,
    StreamableHttpMcp,
    source_call,
)
from sentiment_agent.sources.signal_skills import SignalSkills
from sentiment_agent.sources.toolkit import (
    SIGNAL_TOOL_PROBES,
    USED,
    ToolkitFacade,
    aggregate_health,
    probe_all,
    unused_reason,
)
from sentiment_agent.types import SourceCall, SourceHealth, ToolkitReader, ToolkitSurface
from sources.replay import (
    RecordedHttp,
    call_key,
    load_json,
    load_session,
    sse_reply,
    tool_text,
)

SIGNAL = load_session("signal_session.json")
DATA = load_session("data_session.json")
UPSTREAM = load_json("upstream_shapes.json")
SSE = {"content-type": "text/event-stream"}


def facade(
    signal_http: RecordedHttp | None = None, data_http: RecordedHttp | None = None
) -> tuple[ToolkitFacade, RecordedHttp, RecordedHttp]:
    s_http = signal_http if signal_http is not None else RecordedHttp(SIGNAL)
    d_http = data_http if data_http is not None else RecordedHttp(DATA)
    s_clock, d_clock = SIGNAL.clock(), DATA.clock()
    signal = SignalSkills(
        StreamableHttpMcp(SIGNAL.url, server_label=SIGNAL.server, clock=s_clock, http=s_http),
        s_clock,
    )
    data = BitgetDataService(
        StreamableHttpMcp(DATA.url, server_label=DATA.server, clock=d_clock, http=d_http), d_clock
    )
    return ToolkitFacade(signal, data), s_http, d_http


def tool_calls(http: RecordedHttp) -> list[dict[str, Any]]:
    return [m for _, m, _ in http.requests if m.get("method") == "tools/call"]


def data_reply(structured: dict[str, Any]) -> tuple[int, dict[str, str], bytes]:
    return (
        200,
        SSE,
        sse_reply(
            {
                "content": [{"type": "text", "text": json.dumps(structured)}],
                "isError": False,
                "structuredContent": structured,
            }
        ),
    )


EMPTY_204 = data_reply({"success": True, "status_code": 204, "data": "", "error": None})
HOLLOW_SIGNAL = (200, SSE, sse_reply(tool_text({"error": ""})))


def signal_reply(payload: Any) -> tuple[int, dict[str, str], bytes]:
    return 200, SSE, sse_reply(tool_text(payload))


def test_facade_satisfies_the_reader_protocol() -> None:
    reader: ToolkitReader = facade()[0]
    assert reader is not None


# ================================================================================================
# Mood
# ================================================================================================


def test_mood_from_the_recording() -> None:
    toolkit, _, _ = facade()
    mood, calls = toolkit.mood()
    rows = DATA.payload(
        call_key("do_query", {"entry_id": ENTRY_CRYPTO_FEAR_GREED, "params": {"limit": "3"}})
    )["data"]["results"]
    today = max(rows, key=lambda r: r["time"])
    assert mood.crypto_fear_greed == today["value"]
    assert mood.crypto_fear_greed_label == today["classification"]
    assert mood.crypto_fear_greed_source == (
        "bitget_mcp_server:do_query:crypto_sentiment_crypto_fear_greed"
    )
    assert mood.crypto_fear_greed_alt is None  # bitget-signal was hollow
    assert mood.market_fear_greed is not None
    assert [c.source for c in calls] == [
        "do_query:crypto_sentiment_crypto_fear_greed",
        "sentiment_index.current",
        "do_query:sentiment_market_fear_greed",
    ]
    assert [c.health for c in calls] == [SourceHealth.OK, SourceHealth.HOLLOW, SourceHealth.OK]


def _signal_fear_greed(value: int, label: str) -> RecordedHttp:
    http = RecordedHttp(SIGNAL)
    http.overrides[call_key("sentiment_index", {"action": "current"})] = signal_reply(
        {"data": [{"value": str(value), "value_classification": label, "timestamp": "1790208000"}]}
    )
    return http


def test_both_crypto_sources_are_kept_apart_never_averaged() -> None:
    toolkit, _, _ = facade(signal_http=_signal_fear_greed(40, "Fear"))
    mood, _ = toolkit.mood()
    assert mood.crypto_fear_greed == 71
    assert mood.crypto_fear_greed_alt == 40
    assert mood.crypto_fear_greed_alt_source == "bitget_signal_mcp:sentiment_index.current"
    assert mood.crypto_fear_greed_label == "Greed"


def test_signal_stands_in_only_when_the_primary_fails() -> None:
    failing = RecordedHttp(DATA)
    failing.overrides[
        call_key("do_query", {"entry_id": ENTRY_CRYPTO_FEAR_GREED, "params": {"limit": "3"}})
    ] = (502, {}, b"bad gateway")
    toolkit, _, _ = facade(signal_http=_signal_fear_greed(40, "Fear"), data_http=failing)
    mood, calls = toolkit.mood()
    assert (mood.crypto_fear_greed, mood.crypto_fear_greed_label) == (40, "Fear")
    assert mood.crypto_fear_greed_source == "bitget_signal_mcp:sentiment_index.current"
    assert mood.crypto_fear_greed_alt is None  # one answer is not an agreement check
    assert calls[0].health is SourceHealth.ERROR


def test_no_crypto_reading_when_neither_answers() -> None:
    failing = RecordedHttp(DATA)
    failing.overrides[
        call_key("do_query", {"entry_id": ENTRY_CRYPTO_FEAR_GREED, "params": {"limit": "3"}})
    ] = EMPTY_204
    failing.overrides[
        call_key("do_query", {"entry_id": "sentiment_market_fear_greed", "params": {}})
    ] = EMPTY_204
    toolkit, _, _ = facade(data_http=failing)
    mood, calls = toolkit.mood()
    assert mood.crypto_fear_greed is None
    assert mood.crypto_fear_greed_source is None
    assert mood.market_fear_greed is None
    assert mood.market_fear_greed_label is None
    assert [c.health for c in calls] == [
        SourceHealth.EMPTY,
        SourceHealth.HOLLOW,
        SourceHealth.EMPTY,
    ]


# ================================================================================================
# Positioning
# ================================================================================================


def test_signal_is_not_asked_when_the_primary_answered() -> None:
    toolkit, s_http, _ = facade()
    reading, calls = toolkit.derivatives("BTCUSDT")
    assert tool_calls(s_http) == []
    assert len(calls) == 6
    assert all(c.surface is ToolkitSurface.DATA_MCP for c in calls)
    assert reading.retail_long_short_ratio is not None


def test_signal_fills_only_what_the_primary_left_empty() -> None:
    toolkit, s_http, _ = facade()
    reading, calls = toolkit.derivatives("SP500USDT")
    assert [c.source for c in calls[6:]] == [
        "derivatives_sentiment.long_short",
        "derivatives_sentiment.top_ls",
        "derivatives_sentiment.taker_ratio",
        "derivatives_sentiment.open_interest",
    ]
    assert all(c.health is SourceHealth.EMPTY for c in calls[:6])
    assert all(c.health is SourceHealth.HOLLOW for c in calls[6:])
    assert {c["params"]["arguments"]["period"] for c in tool_calls(s_http)} == {"1h"}
    assert reading.retail_long_short_ratio is None
    assert reading.top_trader_position_ratio is None  # bitget-signal has no fallback for it


def test_fallback_values_come_from_signal_and_the_series_is_never_spliced() -> None:
    params = {"symbol": "BTCUSDT", "interval": "1h", "limit": "3", "exchange": EXCHANGE}
    data_http = RecordedHttp(DATA)
    data_http.overrides[
        call_key("do_query", {"entry_id": "crypto_futures_long_short_ratio", "params": params})
    ] = EMPTY_204
    oi_params = {**params, "limit": "48"}
    data_http.overrides[
        call_key(
            "do_query",
            {"entry_id": "crypto_futures_open_interest_history", "params": oi_params},
        )
    ] = EMPTY_204
    signal_http = RecordedHttp(SIGNAL)
    ls_rows = UPSTREAM["replies"]["binance_global_long_short"]["payload"]
    oi_rows = UPSTREAM["replies"]["binance_open_interest_hist"]["payload"]
    signal_http.overrides[
        call_key(
            "derivatives_sentiment",
            {"action": "long_short", "symbol": "BTCUSDT", "period": "1h"},
        )
    ] = signal_reply(ls_rows)
    signal_http.overrides[
        call_key(
            "derivatives_sentiment",
            {"action": "open_interest", "symbol": "BTCUSDT", "period": "1h"},
        )
    ] = signal_reply(oi_rows)
    toolkit, s_http, _ = facade(signal_http=signal_http, data_http=data_http)
    reading, calls = toolkit.derivatives("BTCUSDT")
    newest_ls = max(ls_rows, key=lambda r: r["timestamp"])
    assert reading.retail_long_short_ratio == pytest.approx(float(newest_ls["longShortRatio"]))
    assert len(reading.open_interest_history) == len(oi_rows)  # whole series, one source
    assert reading.top_trader_account_ratio is not None  # still bitget-mcp-server's
    asked = sorted(c["params"]["arguments"]["action"] for c in tool_calls(s_http))
    assert asked == ["long_short", "open_interest"]
    assert [c.source for c in calls[6:]] == [
        "derivatives_sentiment.long_short",
        "derivatives_sentiment.open_interest",
    ]
    assert all(c.health is SourceHealth.OK for c in calls[6:])


# ================================================================================================
# Text and calendar
# ================================================================================================


def test_news_and_reddit_from_the_recording_are_hollow_and_counted() -> None:
    toolkit, _, _ = facade()
    news, news_calls = toolkit.news(10)
    reddit, reddit_calls = toolkit.reddit_trending(10)
    assert news == ()
    assert reddit == ()
    assert [c.health for c in news_calls] == [SourceHealth.HOLLOW]
    assert [c.params["filter"] for c in reddit_calls] == ["all-crypto", "all-stocks"]
    assert all(c.health is SourceHealth.HOLLOW for c in reddit_calls)


def test_reddit_merges_both_filters() -> None:
    signal_http = RecordedHttp(SIGNAL)
    for name in ("all-crypto", "all-stocks"):
        payload = UPSTREAM["replies"][f"apewisdom_{name.replace('-', '_')}"]["payload"]
        signal_http.overrides[
            call_key(
                "derivatives_sentiment",
                {"action": "reddit_trending", "limit": 5, "filter": name},
            )
        ] = signal_reply(payload)
    toolkit, _, _ = facade(signal_http=signal_http)
    items, calls = toolkit.reddit_trending(5)
    assert len(items) == 10
    assert {i.source for i in items} == {"reddit/all-crypto", "reddit/all-stocks"}
    assert all(c.health is SourceHealth.OK for c in calls)


def test_calendar_window_symbols_and_order() -> None:
    toolkit, _, _ = facade()
    since = DATA.captured_at - timedelta(days=30)
    symbols = ["NVDAUSDT", "AAPLUSDT", "TSLAUSDT", "BTCUSDT", "NVDAUSDT"]
    items, calls = toolkit.calendar(symbols, since=since)
    assert [c.source for c in calls] == ["do_query:equity_calendar"] * 3 + [
        "do_query:equity_ownership_insider_trading"
    ] * 3
    assert [c.params["symbol"] for c in calls] == ["NVDA", "AAPL", "TSLA"] * 2
    assert all(c.health is SourceHealth.OK for c in calls)
    first_day = since.astimezone(NEW_YORK).date()
    assert items
    assert all(i.at is not None and i.at.astimezone(NEW_YORK).date() >= first_day for i in items)
    assert {i.symbol for i in items} <= {"NVDAUSDT", "AAPLUSDT", "TSLAUSDT"}
    keys = [(i.at or since, i.symbol or "", i.kind, i.title) for i in items]
    assert keys == sorted(keys)
    everything, _ = toolkit.calendar(["NVDAUSDT"], since=datetime(2020, 1, 1, tzinfo=UTC))
    assert any(i.kind == "earnings" for i in everything)


def test_calendar_without_equities_makes_no_call() -> None:
    toolkit, _, d_http = facade()
    assert toolkit.calendar(["BTCUSDT", "SP500USDT"], since=DATA.captured_at) == ((), ())
    assert d_http.requests == []
    with pytest.raises(ValueError, match="timezone-aware"):
        toolkit.calendar(["NVDAUSDT"], since=datetime(2026, 9, 1))  # noqa: DTZ001 - the point


# ================================================================================================
# The facade never raises
# ================================================================================================


class ExplodingData(BitgetDataService):
    def crypto_fear_greed(self) -> Any:
        raise RuntimeError("boom")

    def market_fear_greed(self) -> Any:
        raise RuntimeError("boom")

    def derivatives(self, symbol: str) -> Any:
        raise RuntimeError("boom")

    def earnings(self, symbol: str) -> Any:
        raise RuntimeError("boom")

    def insider_filings(self, symbol: str, *, since: datetime) -> Any:
        raise RuntimeError("boom")


class ExplodingSignal(SignalSkills):
    def fear_greed(self) -> Any:
        raise RuntimeError("boom")

    def long_short(self, symbol: str, period: str = "4h") -> Any:
        raise RuntimeError("boom")

    def top_long_short(self, symbol: str, period: str = "4h") -> Any:
        raise RuntimeError("boom")

    def taker_ratio(self, symbol: str, period: str = "4h") -> Any:
        raise RuntimeError("boom")

    def open_interest(self, symbol: str, period: str = "1h") -> Any:
        raise RuntimeError("boom")

    def news(self, limit: int, keyword: str | None = None) -> Any:
        raise RuntimeError("boom")

    def reddit_trending(self, limit: int, *, subreddit_filter: str = "all-crypto") -> Any:
        raise RuntimeError("boom")


def test_every_reader_failure_becomes_an_error_call() -> None:
    clock = DATA.clock()
    unreachable = StreamableHttpMcp(
        DATA_MCP_URL, server_label="x", clock=clock, http=RecordedHttp(DATA)
    )
    toolkit = ToolkitFacade(ExplodingSignal(unreachable, clock), ExplodingData(unreachable, clock))
    mood, mood_calls = toolkit.mood()
    assert mood.crypto_fear_greed is None
    assert [c.health for c in mood_calls] == [SourceHealth.ERROR] * 3
    reading, calls = toolkit.derivatives("BTCUSDT")
    assert reading.retail_long_short_ratio is None
    assert all(c.health is SourceHealth.ERROR for c in calls)
    assert len(calls) == 5  # the data failure, then four signal fallbacks
    assert toolkit.news(5)[1][0].health is SourceHealth.ERROR
    assert all(c.health is SourceHealth.ERROR for c in toolkit.reddit_trending(5)[1])
    items, calendar_calls = toolkit.calendar(["NVDAUSDT"], since=DATA.captured_at)
    assert items == ()
    assert [c.health for c in calendar_calls] == [SourceHealth.ERROR] * 2
    assert all(c.error is not None and "boom" in c.error for c in calendar_calls)


# ================================================================================================
# Probe
# ================================================================================================


def probe_facade(guide_failure: bool = False, drop_entry: str | None = None) -> ToolkitFacade:
    signal_http = RecordedHttp(SIGNAL, default=lambda message: HOLLOW_SIGNAL)
    data_http = RecordedHttp(DATA, default=lambda message: EMPTY_204)
    if guide_failure:
        data_http.overrides[call_key("guide", {})] = (500, {}, b"down")
    if drop_entry is not None:
        key = call_key("guide", {"category": "sentiment"})
        listing = DATA.payload(key)
        kept = [e for e in listing["entries"] if e["id"] != drop_entry]
        data_http.overrides[key] = data_reply({"entries": kept})
    return facade(signal_http=signal_http, data_http=data_http)[0]


UNIVERSE = ("BTCUSDT", "NVDAUSDT", "SP500USDT", "AAPLUSDT")


def test_probe_measures_every_used_source_and_every_catalog_entry() -> None:
    probe = probe_all(probe_facade(), universe=UNIVERSE, clock=DATA.clock())
    assert probe.at == DATA.captured_at
    rows = {(r.surface, r.entry): r for r in probe.rows}
    assert len(rows) == len(probe.rows)  # one row per surface entry
    for key in USED:
        assert key in rows, key
        assert rows[key].used_in
        assert rows[key].judged_line
    positioning = rows[(ToolkitSurface.DATA_MCP, "do_query:crypto_futures_long_short_ratio")]
    assert positioning.last_health is SourceHealth.OK
    assert "SP500USDT=empty" in positioning.notes
    assert "AAPLUSDT=empty" in positioning.notes
    fallback = rows[(ToolkitSurface.SIGNAL_MCP, "derivatives_sentiment.long_short")]
    assert fallback.last_health is SourceHealth.HOLLOW
    assert "4 call(s)" in fallback.notes
    guide = rows[(ToolkitSurface.DATA_MCP, "guide")]
    assert guide.last_health is SourceHealth.OK
    assert guide.notes.startswith("67 entries")
    unused_data = [
        r
        for r in probe.rows
        if r.surface is ToolkitSurface.DATA_MCP and r.purpose == "measured, not used"
    ]
    assert len(unused_data) == 67 - len(USED_ENTRIES)
    by_entry = {r.entry: r for r in unused_data}
    label_search = by_entry.pop("do_query:news_label_search")  # recorded: it needs a label
    assert label_search.last_health is SourceHealth.ERROR
    assert "Missing required param" in label_search.notes
    assert all(r.last_health is SourceHealth.EMPTY for r in by_entry.values())
    assert all(r.notes.startswith("not used") for r in unused_data)
    unused_signal = [
        r
        for r in probe.rows
        if r.surface is ToolkitSurface.SIGNAL_MCP and r.purpose == "measured, not used"
    ]
    assert len(unused_signal) == len(SIGNAL_TOOL_PROBES)
    assert {r.entry for r in unused_signal} >= {"technical_analysis.rsi", "backtest.chart"}


def test_probe_reports_a_failed_catalog_instead_of_raising() -> None:
    probe = probe_all(probe_facade(guide_failure=True), universe=UNIVERSE, clock=DATA.clock())
    guide = [r for r in probe.rows if r.entry == "guide"]
    assert len(guide) == 1
    assert guide[0].last_health is SourceHealth.ERROR
    assert "HTTP 500" in guide[0].notes
    assert not [
        r
        for r in probe.rows
        if r.surface is ToolkitSurface.DATA_MCP and r.purpose == "measured, not used"
    ]


def test_probe_names_a_used_entry_the_catalog_dropped() -> None:
    probe = probe_all(
        probe_facade(drop_entry="sentiment_market_fear_greed"),
        universe=UNIVERSE,
        clock=DATA.clock(),
    )
    missing = [r for r in probe.rows if r.entry == "guide:missing-used-entries"]
    assert len(missing) == 1
    assert "sentiment_market_fear_greed" in missing[0].notes


def _calls(*healths: SourceHealth) -> Sequence[SourceCall]:
    from sentiment_agent.sources.mcp_http import Invocation

    invocation = Invocation(DATA.captured_at, 1, None, None, None)
    return [
        source_call(
            surface=ToolkitSurface.DATA_MCP,
            source="s",
            params={"symbol": f"S{i}"},
            invocation=invocation,
            health=h,
        )
        for i, h in enumerate(healths)
    ]


def test_aggregate_health_never_shows_a_partial_failure_as_green() -> None:
    assert aggregate_health(_calls(SourceHealth.OK, SourceHealth.EMPTY))[0] is SourceHealth.OK
    assert aggregate_health(_calls(SourceHealth.OK, SourceHealth.ERROR))[0] is SourceHealth.ERROR
    assert aggregate_health(_calls(SourceHealth.OK, SourceHealth.TIMEOUT))[0] is (
        SourceHealth.TIMEOUT
    )
    assert aggregate_health(_calls(SourceHealth.HOLLOW, SourceHealth.EMPTY))[0] is (
        SourceHealth.HOLLOW
    )
    assert aggregate_health([])[0] is SourceHealth.DISABLED
    detail = aggregate_health(_calls(SourceHealth.OK, SourceHealth.EMPTY))[1]
    assert detail.startswith("2 call(s)")
    assert "S1=empty" in detail


def test_unused_reasons_cover_every_catalog_entry() -> None:
    catalog = BitgetDataService(
        StreamableHttpMcp(DATA.url, server_label="d", clock=DATA.clock(), http=RecordedHttp(DATA)),
        DATA.clock(),
    ).catalog()
    for entry, _ in catalog:
        if entry not in USED_ENTRIES:
            assert unused_reason(entry).startswith("not used"), entry


# ================================================================================================
# Live drift (keyless; run with --run-live-public)
# ================================================================================================


@pytest.mark.live_public
def test_live_facade_mood_and_positioning() -> None:
    clock = SystemClock()
    toolkit = ToolkitFacade(
        SignalSkills(
            StreamableHttpMcp(SIGNAL_MCP_URL, server_label="bitget-signal", clock=clock), clock
        ),
        BitgetDataService(
            StreamableHttpMcp(DATA_MCP_URL, server_label="bitget-mcp-server", clock=clock), clock
        ),
    )
    mood, calls = toolkit.mood()
    assert mood.crypto_fear_greed is not None, [(c.source, c.health, c.error) for c in calls]
    assert mood.market_fear_greed is not None
    reading, positioning = toolkit.derivatives("BTCUSDT")
    assert reading.retail_long_short_ratio is not None
    assert all(c.surface is ToolkitSurface.DATA_MCP for c in positioning[:6])
