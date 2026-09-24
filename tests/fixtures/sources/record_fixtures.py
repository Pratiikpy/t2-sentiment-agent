"""Record the keyless fixtures the sources tests replay. Run by hand; pytest never runs it.

    PYTHONPATH=src python tests/fixtures/sources/record_fixtures.py

Every MCP exchange is made through the production client and readers
(:class:`~sentiment_agent.sources.mcp_http.StreamableHttpMcp`, :class:`SignalSkills`,
:class:`BitgetDataService`), so the recorded requests are byte-for-byte what the agent sends and
the replay matches them exactly. Only keyless public endpoints are called: bitget-signal,
bitget-mcp-server, and (for ``upstream_shapes.json``) the public upstreams bitget-signal names in
its tool descriptions. No credential is read; no order is placed; no model is called.

Writes, next to this file:

* ``signal_session.json``: bitget-signal handshake, ``tools/list`` and every call the agent makes;
* ``data_session.json``: bitget-mcp-server handshake, ``tools/list``, the catalog, every entry the
  agent reads (positioning for BTCUSDT, NVDAUSDT and SP500USDT; earnings for all eleven underlying
  stocks; insider filings for three), the earnings history used as evidence for the vendor date
  offset, and four refusals;
* ``upstream_shapes.json``: Binance futures, alternative.me and ApeWisdom replies, the documented
  shapes bitget-signal's parsers accept while its own success shape is unobserved.
"""

import json
import sys
import threading
import urllib.request
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any

from sentiment_agent.clock import SystemClock
from sentiment_agent.sources.bitget_data import UNDERLYING, BitgetDataService
from sentiment_agent.sources.mcp_http import (
    DATA_MCP_URL,
    SIGNAL_MCP_URL,
    StreamableHttpMcp,
    urllib_post,
)
from sentiment_agent.sources.signal_skills import REDDIT_FILTERS, SignalSkills

HERE = Path(__file__).resolve().parent
USER_AGENT = "curl/8.0"
KEPT_RESPONSE_HEADERS = ("content-type", "mcp-session-id")
KEPT_REQUEST_HEADERS = ("user-agent", "accept", "content-type", "mcp-protocol-version")

INSIDER_SYMBOLS = ("NVDAUSDT", "AAPLUSDT", "TSLAUSDT")
"""Insider filings are recorded for three names only: each reply is about 40 KB, and the replay
tests need the shape, not all eleven. The live drift test calls all eleven."""

EARNINGS_EVIDENCE = ("NVDA", "AAPL", "TSLA", "META", "COIN")
"""Stocks whose recorded earnings history pins the vendor's one-day date offset."""

UPSTREAMS: Mapping[str, str] = {
    "binance_global_long_short": "https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol=BTCUSDT&period=1h&limit=3",
    "binance_top_long_short": "https://fapi.binance.com/futures/data/topLongShortAccountRatio?symbol=BTCUSDT&period=1h&limit=3",
    "binance_taker_ratio": "https://fapi.binance.com/futures/data/takerlongshortRatio?symbol=BTCUSDT&period=1h&limit=3",
    "binance_open_interest_hist": "https://fapi.binance.com/futures/data/openInterestHist?symbol=BTCUSDT&period=1h&limit=3",
    "binance_funding_rate": "https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&limit=10",
    "alternative_me_fng": "https://api.alternative.me/fng/?limit=2",
    "apewisdom_all_crypto": "https://apewisdom.io/api/v1.0/filter/all-crypto/page/1",
    "apewisdom_all_stocks": "https://apewisdom.io/api/v1.0/filter/all-stocks/page/1",
}


class Recorder:
    """An HttpPost that forwards to the real transport and keeps every exchange."""

    def __init__(self) -> None:
        self.exchanges: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def __call__(
        self, url: str, body: bytes, headers: Mapping[str, str], timeout: float
    ) -> tuple[int, Mapping[str, str], bytes]:
        status, reply_headers, raw = urllib_post(url, body, headers, timeout)
        record = {
            "request": json.loads(body),
            "request_headers": {
                k: v for k, v in headers.items() if k.lower() in KEPT_REQUEST_HEADERS
            },
            "status": status,
            "headers": {k: v for k, v in reply_headers.items() if k in KEPT_RESPONSE_HEADERS},
            "body": raw.decode("utf-8"),
        }
        with self._lock:
            self.exchanges.append(record)
        return status, reply_headers, raw


def _write(name: str, server: str, url: str, started: datetime, recorder: Recorder) -> None:
    document = {
        "server": server,
        "url": url,
        "captured_at": started.isoformat(),
        "captured_with": "tests/fixtures/sources/record_fixtures.py",
        "user_agent": USER_AGENT,
        "note": "Real keyless exchanges, unedited except that volatile response headers "
        "(cookies, Cloudflare ids, dates) are dropped.",
        "exchanges": recorder.exchanges,
    }
    (HERE / name).write_text(json.dumps(document, indent=1, ensure_ascii=False) + "\n", "utf-8")
    print(f"wrote {name}: {len(recorder.exchanges)} exchanges")


def _run_all(jobs: list[Callable[[], Any]]) -> None:
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in [pool.submit(job) for job in jobs]:
            future.result()


def record_signal(started: datetime) -> None:
    clock = SystemClock()
    recorder = Recorder()
    mcp = StreamableHttpMcp(
        SIGNAL_MCP_URL, server_label="bitget-signal", clock=clock, http=recorder
    )
    signal = SignalSkills(mcp, clock)
    mcp.initialize()
    mcp.list_tools()
    jobs: list[Callable[[], Any]] = [
        signal.fear_greed,
        lambda: signal.long_short("BTCUSDT"),
        lambda: signal.top_long_short("BTCUSDT"),
        lambda: signal.taker_ratio("BTCUSDT"),
        lambda: signal.open_interest("BTCUSDT"),
        lambda: signal.news(10),
        lambda: mcp.call_tool("no_such_tool", {}),
        lambda: mcp.call_tool("derivatives_sentiment", {"action": "not_an_action"}),
    ]
    for symbol in ("BTCUSDT", "SP500USDT"):
        jobs += [
            partial(signal.long_short, symbol, "1h"),
            partial(signal.top_long_short, symbol, "1h"),
            partial(signal.taker_ratio, symbol, "1h"),
            partial(signal.open_interest, symbol, "1h"),
        ]
    jobs += [partial(signal.reddit_trending, 10, subreddit_filter=f) for f in REDDIT_FILTERS]
    _run_all(jobs)
    _write("signal_session.json", "bitget-signal", SIGNAL_MCP_URL, started, recorder)


def record_data(started: datetime) -> None:
    clock = SystemClock()
    recorder = Recorder()
    mcp = StreamableHttpMcp(
        DATA_MCP_URL, server_label="bitget-mcp-server", clock=clock, http=recorder
    )
    data = BitgetDataService(mcp, clock)
    mcp.initialize()
    mcp.list_tools()
    data.catalog()
    since = started - timedelta(days=30)
    jobs: list[Callable[[], Any]] = [
        data.crypto_fear_greed,
        data.market_fear_greed,
        lambda: mcp.call_tool("do_query", {"id": "crypto_sentiment_crypto_fear_greed"}),
        lambda: data.query("no_such_entry"),
        lambda: data.query("news_label_search"),
        lambda: data.query("crypto_futures_taker_volume", symbol="BTCUSDT"),
    ]
    jobs += [partial(data.derivatives, s) for s in ("BTCUSDT", "NVDAUSDT", "SP500USDT")]
    jobs += [partial(data.earnings, s) for s in UNDERLYING]
    jobs += [partial(data.insider_filings, s, since=since) for s in INSIDER_SYMBOLS]
    jobs += [
        partial(data.query, "equity_calendar", symbol=t, start_date="2024-10-01")
        for t in EARNINGS_EVIDENCE
    ]
    _run_all(jobs)
    _write("data_session.json", "bitget-mcp-server", DATA_MCP_URL, started, recorder)


def record_upstreams(started: datetime) -> None:
    replies: dict[str, Any] = {}
    for name, url in UPSTREAMS.items():
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - fixed https URLs
            payload = json.loads(response.read())
        if isinstance(payload, dict) and isinstance(payload.get("results"), list):
            payload = {**payload, "results": payload["results"][:8]}  # keep the fixture small
        replies[name] = {"url": url, "payload": payload}
    document = {
        "captured_at": started.isoformat(),
        "captured_with": "tests/fixtures/sources/record_fixtures.py",
        "note": "Public upstream replies, recorded to pin the field names bitget-signal's "
        "parsers accept. bitget-signal's own success shape has never been observed "
        "(NOT VERIFIED); these are what its tool descriptions say it wraps.",
        "replies": replies,
    }
    path = HERE / "upstream_shapes.json"
    path.write_text(json.dumps(document, indent=1, ensure_ascii=False) + "\n", "utf-8")
    print(f"wrote upstream_shapes.json: {len(replies)} replies")


def main(argv: list[str]) -> int:
    started = datetime.now(UTC).replace(microsecond=0)
    wanted = set(argv) or {"signal", "data", "upstream"}
    if "data" in wanted:
        record_data(started)
    if "upstream" in wanted:
        record_upstreams(started)
    if "signal" in wanted:
        record_signal(started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
