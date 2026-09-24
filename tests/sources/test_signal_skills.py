"""bitget-signal readings: the recorded (hollow) replies, the upstream success shapes the parsers
accept (NOT VERIFIED as bitget-signal's own), freshness, and every failure mode."""

import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from sentiment_agent.clock import ManualClock, SystemClock
from sentiment_agent.sources.mcp_http import (
    SIGNAL_MCP_URL,
    McpError,
    McpTimeoutError,
    StreamableHttpMcp,
)
from sentiment_agent.sources.signal_skills import (
    FEAR_GREED_MAX_AGE,
    NEWS_FEEDS,
    SignalSkills,
    max_age_for,
)
from sentiment_agent.types import McpToolResult, SourceHealth, ToolkitSurface
from sources.replay import RecordedHttp, load_json, load_session

SIGNAL = load_session("signal_session.json")
UPSTREAM = load_json("upstream_shapes.json")
UPSTREAM_AT = datetime.fromisoformat(UPSTREAM["captured_at"])


def upstream(name: str) -> Any:
    return UPSTREAM["replies"][name]["payload"]


def recorded() -> SignalSkills:
    clock = SIGNAL.clock()
    mcp = StreamableHttpMcp(
        SIGNAL.url, server_label=SIGNAL.server, clock=clock, http=RecordedHttp(SIGNAL)
    )
    return SignalSkills(mcp, clock)


class FakeSignal:
    """An McpCaller answering each tool with a chosen payload (JSON text, as bitget-signal does),
    a chosen raw text, or a raised exception."""

    def __init__(self, answer: Callable[[str, Mapping[str, Any]], Any]) -> None:
        self._answer = answer
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @property
    def server(self) -> str:
        return "bitget-signal"

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> McpToolResult:
        self.calls.append((name, dict(arguments)))
        outcome = self._answer(name, arguments)
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, McpToolResult):
            return outcome
        text = outcome if isinstance(outcome, str) else json.dumps(outcome)
        return McpToolResult(
            server="bitget-signal", tool=name, is_error=False, structured=None, text=text, raw=None
        )


def fake(payload: Any, *, at: datetime = UPSTREAM_AT) -> tuple[SignalSkills, FakeSignal]:
    caller = FakeSignal(lambda name, args: payload)
    return SignalSkills(caller, ManualClock(at)), caller


# ================================================================================================
# The recorded replies: every sentiment call was hollow on 2026-09-24
# ================================================================================================


def test_recorded_fear_greed_is_hollow_not_a_value() -> None:
    value, label, call = recorded().fear_greed()
    assert (value, label) == (None, None)
    assert call.health is SourceHealth.HOLLOW
    assert call.surface is ToolkitSurface.SIGNAL_MCP
    assert call.source == "sentiment_index.current"
    assert call.error is not None
    assert "alt_me_error" in call.error
    assert call.started_at == SIGNAL.captured_at
    assert call.latency_ms >= 0


@pytest.mark.parametrize(
    ("method", "source"),
    [
        ("long_short", "derivatives_sentiment.long_short"),
        ("top_long_short", "derivatives_sentiment.top_ls"),
        ("taker_ratio", "derivatives_sentiment.taker_ratio"),
    ],
)
def test_recorded_ratios_are_hollow(method: str, source: str) -> None:
    reader = recorded()
    for period in ("4h", "1h"):
        value, call = getattr(reader, method)("BTCUSDT", period)
        assert value is None
        assert call.health is SourceHealth.HOLLOW
        assert call.source == source
        assert call.params == {"symbol": "BTCUSDT", "period": period}


def test_recorded_open_interest_is_hollow() -> None:
    series, call = recorded().open_interest("BTCUSDT")
    assert series == ()
    assert call.health is SourceHealth.HOLLOW
    assert call.params == {"symbol": "BTCUSDT", "period": "1h"}


def test_recorded_news_is_hollow_with_every_feed_failed() -> None:
    items, call = recorded().news(10)
    assert items == ()
    assert call.health is SourceHealth.HOLLOW
    assert call.params == {"feeds": NEWS_FEEDS, "limit": "10"}


@pytest.mark.parametrize("name", ["all-crypto", "all-stocks"])
def test_recorded_reddit_is_hollow(name: str) -> None:
    items, call = recorded().reddit_trending(10, subreddit_filter=name)
    assert items == ()
    assert call.health is SourceHealth.HOLLOW
    assert call.params == {"limit": "10", "filter": name}


def test_refused_action_is_an_error_not_a_quiet_market() -> None:
    call = recorded().probe("derivatives_sentiment", {"action": "not_an_action"})
    assert call.health is SourceHealth.ERROR
    assert call.error is not None
    assert "Unknown action" in call.error


def test_unknown_tool_is_an_error() -> None:
    call = recorded().probe("no_such_tool", {})
    assert call.health is SourceHealth.ERROR
    assert call.error is not None
    assert "Unknown tool" in call.error


# ================================================================================================
# Success shapes: the upstreams' own documented fields (NOT VERIFIED as bitget-signal's reply)
# ================================================================================================


def test_long_short_reads_the_newest_binance_row() -> None:
    rows = upstream("binance_global_long_short")
    newest_row = max(rows, key=lambda r: r["timestamp"])
    reader, caller = fake(rows)
    value, call = reader.long_short("BTCUSDT", "1h")
    assert value == pytest.approx(float(newest_row["longShortRatio"]))
    assert call.health is SourceHealth.OK
    assert call.rows == len(rows)
    assert caller.calls == [
        ("derivatives_sentiment", {"action": "long_short", "symbol": "BTCUSDT", "period": "1h"})
    ]


def test_row_order_does_not_decide_which_row_is_current() -> None:
    rows = upstream("binance_top_long_short")
    newest_row = max(rows, key=lambda r: r["timestamp"])
    reader, _ = fake(list(reversed(rows)))
    value, _ = reader.top_long_short("BTCUSDT", "1h")
    assert value == pytest.approx(float(newest_row["longShortRatio"]))


def test_ratio_is_derived_from_accounts_when_no_ratio_field() -> None:
    rows = [
        {k: v for k, v in row.items() if k != "longShortRatio"}
        for row in upstream("binance_global_long_short")
    ]
    newest_row = max(rows, key=lambda r: r["timestamp"])
    reader, _ = fake(rows)
    value, _ = reader.long_short("BTCUSDT", "1h")
    expected = float(newest_row["longAccount"]) / float(newest_row["shortAccount"])
    assert value == pytest.approx(expected)


def test_taker_ratio_and_wrapped_payloads() -> None:
    rows = upstream("binance_taker_ratio")
    newest_row = max(rows, key=lambda r: r["timestamp"])
    for payload in (rows, {"data": rows}, {"symbol": "BTCUSDT", "results": rows}):
        reader, _ = fake(payload)
        value, call = reader.taker_ratio("BTCUSDT", "1h")
        assert value == pytest.approx(float(newest_row["buySellRatio"]))
        assert call.health is SourceHealth.OK


def test_open_interest_series_is_sorted_and_typed() -> None:
    rows = upstream("binance_open_interest_hist")
    reader, _ = fake(list(reversed(rows)))
    series, call = reader.open_interest("BTCUSDT")
    assert call.health is SourceHealth.OK
    assert [t for t, _ in series] == sorted(t for t, _ in series)
    assert len(series) == len(rows)
    assert series[-1][1] == pytest.approx(
        float(max(rows, key=lambda r: r["timestamp"])["sumOpenInterest"])
    )
    assert all(t.tzinfo is not None for t, _ in series)


def test_fear_greed_alternative_me_shape() -> None:
    payload = upstream("alternative_me_fng")
    today = max(payload["data"], key=lambda r: int(r["timestamp"]))
    reader, _ = fake(payload)
    value, label, call = reader.fear_greed()
    assert value == int(today["value"])
    assert label == today["value_classification"]
    assert call.health is SourceHealth.OK
    assert 0 <= value <= 100


def test_fear_greed_scale_is_pinned_to_0_100() -> None:
    reader, _ = fake(
        {"data": [{"value": "150", "value_classification": "?", "timestamp": "1790208000"}]}
    )
    value, _, call = reader.fear_greed()
    assert value is None
    assert call.health is SourceHealth.HOLLOW
    zero, _ = fake(
        {"data": [{"value": 0, "value_classification": "Extreme Fear", "timestamp": "1790208000"}]}
    )
    assert zero.fear_greed()[0] == 0  # zero is a reading, not an absence


def test_reddit_rows_become_text_items_with_symbols() -> None:
    crypto, _ = fake(upstream("apewisdom_all_crypto"))
    items, call = crypto.reddit_trending(5)
    assert call.health is SourceHealth.OK
    assert len(items) == 5
    first = items[0]
    assert first.channel == "reddit"
    assert first.symbols == ("BTCUSDT",)
    assert first.source == "reddit/all-crypto"
    assert "mentions" in first.text
    assert first.published_at == UPSTREAM_AT
    assert first.item_id.startswith("reddit-")

    stocks, _ = fake(upstream("apewisdom_all_stocks"))
    stock_items, _ = stocks.reddit_trending(8, subreddit_filter="all-stocks")
    by_symbol = {i.symbols[0]: i for i in stock_items if i.symbols}
    assert "METAUSDT" in by_symbol
    assert "&amp;" not in " ".join(i.text for i in stock_items)  # HTML entities decoded


def test_reddit_item_ids_are_stable_within_an_hour() -> None:
    payload = upstream("apewisdom_all_crypto")
    first, _ = fake(payload, at=UPSTREAM_AT)
    later, _ = fake(payload, at=UPSTREAM_AT.replace(minute=59))
    next_hour, _ = fake(payload, at=UPSTREAM_AT.replace(minute=0) + timedelta(hours=1))
    ids = [r.reddit_trending(3)[0][0].item_id for r in (first, later, next_hour)]
    assert ids[0] == ids[1] != ids[2]


def test_news_success_shape_is_parsed_dated_and_bounded() -> None:
    feeds = [
        {
            "feed": "coindesk",
            "items": [
                {
                    "title": "Bitcoin <b>holds</b> 84k",
                    "link": "https://www.coindesk.com/a",
                    "published": "Thu, 24 Sep 2026 10:05:00 +0000",
                    "summary": "<p>Spot ETF flows turned positive.</p>",
                },
                {"title": "Undated story", "link": "https://www.coindesk.com/b"},
                {
                    "title": "Naive date",
                    "published": "2026-09-24T09:00:00",
                    "link": "https://www.coindesk.com/c",
                },
            ],
        },
        {
            "feed": "cnbc",
            "items": [
                {
                    "title": "Nvidia shares rise",
                    "link": "https://www.cnbc.com/d",
                    "published": "2026-09-24T11:00:00Z",
                }
            ],
        },
        {"feed": "decrypt", "error": "timeout", "items": []},
    ]
    reader, caller = fake(feeds)
    items, call = reader.news(10, keyword="bitcoin")
    assert call.health is SourceHealth.OK
    assert [i.source for i in items] == ["cnbc", "coindesk"]  # newest first
    assert items[1].text == "Bitcoin holds 84k — Spot ETF flows turned positive."
    assert items[1].published_at == datetime(2026, 9, 24, 10, 5, tzinfo=UTC)
    assert call.error is not None
    assert "2 item(s) without a title or a zoned date dropped" in call.error
    assert "1 of 3 feeds reported an error" in call.error
    assert caller.calls[0][1]["keyword"] == "bitcoin"
    one, _ = reader.news(1)
    assert len(one) == 1


# ================================================================================================
# Shapes that are not recognised, stale or out of range are never read as values
# ================================================================================================


def test_unrecognised_shape_is_hollow_and_names_its_keys() -> None:
    reader, _ = fake({"symbol": "BTCUSDT", "sentiment": "bullish", "score_x": 3})
    value, call = reader.long_short("BTCUSDT", "1h")
    assert value is None
    assert call.health is SourceHealth.HOLLOW
    assert call.error is not None
    assert "NOT VERIFIED" in call.error
    assert "score_x" in call.error


def test_rows_without_timestamps_are_not_current() -> None:
    reader, _ = fake([{"symbol": "BTCUSDT", "longShortRatio": "1.2"}])
    value, call = reader.long_short("BTCUSDT", "1h")
    assert value is None
    assert call.health is SourceHealth.HOLLOW
    assert call.error is not None
    assert "timestamp" in call.error


def test_stale_rows_are_hollow() -> None:
    rows = upstream("binance_global_long_short")
    reader, _ = fake(rows, at=UPSTREAM_AT + max_age_for("1h") + timedelta(hours=1))
    value, call = reader.long_short("BTCUSDT", "1h")
    assert value is None
    assert call.health is SourceHealth.HOLLOW
    assert call.error is not None
    assert "stale" in call.error
    fresh_4h, _ = fake(rows, at=UPSTREAM_AT + timedelta(hours=5))
    assert fresh_4h.long_short("BTCUSDT", "4h")[0] is not None  # 4h series may be 12h old


def test_fear_greed_staleness_limit() -> None:
    payload = upstream("alternative_me_fng")
    newest_day = datetime.fromtimestamp(max(int(r["timestamp"]) for r in payload["data"]), tz=UTC)
    ok, _ = fake(payload, at=newest_day + FEAR_GREED_MAX_AGE - timedelta(minutes=1))
    assert ok.fear_greed()[0] is not None
    stale, _ = fake(payload, at=newest_day + FEAR_GREED_MAX_AGE + timedelta(minutes=1))
    assert stale.fear_greed()[2].health is SourceHealth.HOLLOW


def test_empty_payload_is_empty() -> None:
    reader, _ = fake([])
    assert reader.long_short("BTCUSDT", "1h")[1].health is SourceHealth.EMPTY


# ================================================================================================
# Failure modes: every one becomes a health, none escapes
# ================================================================================================


@pytest.mark.parametrize(
    ("outcome", "health"),
    [
        (McpTimeoutError("slow"), SourceHealth.TIMEOUT),
        (McpError("HTTP 502"), SourceHealth.ERROR),
        (RuntimeError("defect in a stand-in"), SourceHealth.ERROR),
        ("not json at all", SourceHealth.ERROR),
        (
            McpToolResult(
                server="s", tool="t", is_error=True, structured=None, text="boom", raw=None
            ),
            SourceHealth.ERROR,
        ),
    ],
)
def test_failures_become_health_for_every_reading(outcome: Any, health: SourceHealth) -> None:
    reader = SignalSkills(FakeSignal(lambda name, args: outcome), ManualClock(UPSTREAM_AT))
    assert reader.fear_greed()[2].health is health
    assert reader.long_short("BTCUSDT")[1].health is health
    assert reader.top_long_short("BTCUSDT")[1].health is health
    assert reader.taker_ratio("BTCUSDT")[1].health is health
    assert reader.open_interest("BTCUSDT")[1].health is health
    assert reader.reddit_trending(5)[1].health is health
    assert reader.news(5)[1].health is health
    assert reader.probe("x", {"action": "y"}).health is health


def test_argument_validation() -> None:
    reader, _ = fake([])
    with pytest.raises(ValueError, match="period"):
        reader.long_short("BTCUSDT", "3h")
    with pytest.raises(ValueError, match="instrument code"):
        reader.long_short("btc/usdt", "1h")
    with pytest.raises(ValueError, match=r"1\.\.50"):
        reader.reddit_trending(0)
    with pytest.raises(ValueError, match="at least 1"):
        reader.news(0)


# ================================================================================================
# Live drift (keyless; run with --run-live-public)
# ================================================================================================


@pytest.mark.live_public
def test_live_signal_calls_classify_without_error() -> None:
    """Whatever the server's upstreams are doing, our calls are well-formed: a reply is OK or
    HOLLOW, never ERROR (which would mean the server refused our arguments)."""
    clock = SystemClock()
    reader = SignalSkills(
        StreamableHttpMcp(SIGNAL_MCP_URL, server_label="bitget-signal", clock=clock), clock
    )
    value, _, call = reader.fear_greed()
    assert call.health in (SourceHealth.OK, SourceHealth.HOLLOW, SourceHealth.TIMEOUT)
    if value is not None:
        assert 0 <= value <= 100
    ratio, ratio_call = reader.long_short("BTCUSDT", "1h")
    assert ratio_call.health in (SourceHealth.OK, SourceHealth.HOLLOW, SourceHealth.TIMEOUT)
    if ratio is not None:
        assert 0 < ratio < 100
