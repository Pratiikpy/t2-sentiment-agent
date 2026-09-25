"""run2-d3: the upstreams Bitget's data services wrap, read directly when both services fail.

Replayed from the upstream replies recorded on 2026-09-24 (``upstream_shapes.json``), with the
clock set to the moment they were recorded, so every row is as fresh as it was then."""

import json
import urllib.parse
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest

from sentiment_agent.clock import ManualClock
from sentiment_agent.sources.bitget_data import BitgetDataService
from sentiment_agent.sources.signal_skills import SignalSkills
from sentiment_agent.sources.toolkit import ToolkitFacade
from sentiment_agent.sources.upstream import (
    NEWS_FEEDS,
    UpstreamDirect,
    UpstreamError,
    urllib_get,
)
from sentiment_agent.types import (
    DerivativesReading,
    SourceCall,
    SourceHealth,
    ToolkitSurface,
)
from sources.replay import load_json

UPSTREAM = load_json("upstream_shapes.json")
REPLIES: Mapping[str, Any] = UPSTREAM["replies"]
CAPTURED = datetime.fromisoformat(UPSTREAM["captured_at"])

RSS = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>
<item><title>Bitcoin funding turns negative</title><link>https://x.test/a</link>
<pubDate>Thu, 24 Sep 2026 10:00:00 GMT</pubDate></item>
<item><title>Old story</title><link>https://x.test/b</link>
<pubDate>Mon, 21 Sep 2026 10:00:00 GMT</pubDate></item>
<item><title>No date</title><link>https://x.test/c</link></item>
</channel></rss>"""

BY_PATH = {
    "futures/data/globalLongShortAccountRatio": "binance_global_long_short",
    "futures/data/topLongShortAccountRatio": "binance_top_long_short",
    "futures/data/takerlongshortRatio": "binance_taker_ratio",
    "futures/data/openInterestHist": "binance_open_interest_hist",
    "fapi/v1/fundingRate": "binance_funding_rate",
    "fng/": "alternative_me_fng",
}


def recorded(url: str) -> bytes:
    """The recorded reply for a URL, by path: the recording asked for fewer rows than the
    readers do, which a path match ignores."""
    path = urllib.parse.urlsplit(url).path.lstrip("/")
    if path == "futures/data/topLongShortPositionRatio":
        path = "futures/data/topLongShortAccountRatio"  # same shape; not recorded separately
    if path in BY_PATH:
        return json.dumps(REPLIES[BY_PATH[path]]["payload"]).encode()
    if any(url == feed for _, feed in NEWS_FEEDS):
        return RSS
    raise UpstreamError(f"not recorded: {url}")


def reader(now: datetime = CAPTURED, fetch: Any = recorded) -> UpstreamDirect:
    return UpstreamDirect(ManualClock(now), fetch=fetch)


def newest(key: str, field: str) -> float:
    rows = REPLIES[key]["payload"]
    return float(max(rows, key=lambda r: r["timestamp"])[field])


class TestEachReading:
    def test_positioning_is_the_newest_row_from_binance(self) -> None:
        up = reader()
        ls, call = up.long_short("BTCUSDT")
        assert ls == pytest.approx(newest("binance_global_long_short", "longShortRatio"))
        assert call.surface is ToolkitSurface.UPSTREAM_DIRECT
        assert call.source == "upstream:binance.globalLongShortAccountRatio"
        assert call.health is SourceHealth.OK
        assert call.params["symbol"] == "BTCUSDT"
        taker, _ = up.taker("BTCUSDT")
        assert taker == pytest.approx(newest("binance_taker_ratio", "buySellRatio"))

    def test_open_interest_is_the_whole_series_in_contracts(self) -> None:
        series, call = reader().open_interest("BTCUSDT")
        rows = REPLIES["binance_open_interest_hist"]["payload"]
        assert len(series) == len(rows)
        assert call.health is SourceHealth.OK
        assert series[-1][1] == pytest.approx(
            newest("binance_open_interest_hist", "sumOpenInterest")
        )
        assert [p[0] for p in series] == sorted(p[0] for p in series)

    def test_funding_is_already_a_fraction(self) -> None:
        rows = REPLIES["binance_funding_rate"]["payload"]
        latest = max(rows, key=lambda r: r["fundingTime"])
        when = datetime.fromtimestamp(latest["fundingTime"] / 1000, UTC)
        rate, call = reader(now=when + timedelta(hours=1)).funding("BTCUSDT")
        assert rate == pytest.approx(float(latest["fundingRate"]))
        assert call.health is SourceHealth.OK

    def test_fear_and_greed_from_alternative_me(self) -> None:
        value, label, call = reader().fear_greed()
        row = max(REPLIES["alternative_me_fng"]["payload"]["data"], key=lambda r: r["timestamp"])
        assert value == int(row["value"])
        assert label == row["value_classification"]
        assert call.source == "upstream:alternative_me.fng"

    def test_news_keeps_fresh_dated_titles_from_every_feed(self) -> None:
        items, calls = reader(now=datetime(2026, 9, 24, 12, tzinfo=UTC)).news(10)
        assert [c.source for c in calls] == [f"upstream:rss.{name}" for name, _ in NEWS_FEEDS]
        assert {i.text for i in items} == {"Bitcoin funding turns negative"}
        assert all(i.channel == "news" and i.source in dict(NEWS_FEEDS) for i in items)


class TestNothingIsAQuietMarket:
    def test_a_stale_row_is_hollow_not_a_value(self) -> None:
        value, call = reader(now=CAPTURED + timedelta(days=3)).long_short("BTCUSDT")
        assert value is None
        assert call.health is SourceHealth.HOLLOW
        assert "stale" in (call.error or "")

    def test_an_unreachable_upstream_is_an_error_call(self) -> None:
        def down(url: str) -> bytes:
            raise UpstreamError("HTTP 451")

        value, call = reader(fetch=down).taker("BTCUSDT")
        assert value is None
        assert call.health is SourceHealth.ERROR
        assert call.error == "HTTP 451"

    def test_an_error_object_is_hollow(self) -> None:
        value, call = reader(
            fetch=lambda url: b'{"code": -1121, "msg": "Invalid symbol."}'
        ).long_short("NOPEUSDT")
        assert value is None
        assert call.health is SourceHealth.HOLLOW

    def test_only_https_is_fetched(self) -> None:
        with pytest.raises(UpstreamError, match="non-https"):
            urllib_get("file:///etc/passwd")


class _Dead:
    """Both Bitget services, down: every reader raises or answers empty."""

    def __init__(self, clock: ManualClock) -> None:
        self.clock = clock

    def __getattr__(self, name: str) -> Any:
        def fail(*args: object, **kwargs: object) -> Any:
            raise RuntimeError(f"{name}: 503 from upstream")

        return fail

    def derivatives(self, symbol: str) -> tuple[DerivativesReading, tuple[SourceCall, ...]]:
        raise RuntimeError("do_query: 503 from upstream")


def dead_facade(upstream: UpstreamDirect | None) -> ToolkitFacade:
    clock = ManualClock(CAPTURED)
    dead = _Dead(clock)
    return ToolkitFacade(cast(SignalSkills, dead), cast(BitgetDataService, dead), upstream=upstream)


class TestTheFacadeFallsBack:
    def test_every_positioning_field_is_filled_after_both_services_fail(self) -> None:
        reading, calls = dead_facade(reader()).derivatives("BTCUSDT")
        assert reading.retail_long_short_ratio is not None
        assert reading.top_trader_account_ratio is not None
        assert reading.top_trader_position_ratio is not None
        assert reading.taker_buy_sell_ratio is not None
        assert len(reading.open_interest_history) == 3
        upstream = [c for c in calls if c.surface is ToolkitSurface.UPSTREAM_DIRECT]
        bitget = [c for c in calls if c.surface is not ToolkitSurface.UPSTREAM_DIRECT]
        assert bitget  # still recorded
        assert all(c.health is SourceHealth.ERROR for c in bitget)
        assert calls[: len(bitget)] == tuple(bitget)  # the Bitget calls come first
        assert [c.source for c in upstream] == [
            "upstream:binance.globalLongShortAccountRatio",
            "upstream:binance.topLongShortAccountRatio",
            "upstream:binance.topLongShortPositionRatio",
            "upstream:binance.takerlongshortRatio",
            "upstream:binance.openInterestHist",
            "upstream:binance.fundingRate",
        ]

    def test_fear_and_greed_names_where_it_came_from(self) -> None:
        mood, calls = dead_facade(reader()).mood()
        assert mood.crypto_fear_greed is not None
        assert mood.crypto_fear_greed_source == "upstream:alternative_me.fng"
        assert calls[-1].surface is ToolkitSurface.UPSTREAM_DIRECT

    def test_news_comes_from_the_publishers(self) -> None:
        facade = dead_facade(reader(now=datetime(2026, 9, 24, 12, tzinfo=UTC)))
        items, calls = facade.news(5)
        assert items  # asked first
        assert calls[0].surface is ToolkitSurface.SIGNAL_MCP

    def test_without_the_fallback_nothing_changes(self) -> None:
        reading, calls = dead_facade(None).derivatives("BTCUSDT")
        assert reading.retail_long_short_ratio is None
        assert not any(c.surface is ToolkitSurface.UPSTREAM_DIRECT for c in calls)
        mood, _ = dead_facade(None).mood()
        assert mood.crypto_fear_greed is None
