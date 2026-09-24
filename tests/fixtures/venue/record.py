"""Record the venue fixtures: real keyless responses from Bitget's public v3 market endpoints.

    python tests/fixtures/venue/record.py

Writes ``tests/fixtures/venue/cassette.json``. Every exchange is captured through
:class:`BitgetPublicApi` itself (so the recorded URLs are exactly what the client sends) or, for the
raw probes that pin venue semantics the client depends on, through :func:`urllib_get` with the
client's own headers. Keyless GETs only: no credential, no account endpoint, no order.

Each exchange stores the URL, the headers sent, the HTTP status, the body verbatim and the capture
time. The tests replay them offline; nothing in the test suite reaches the network.
"""

import json
import sys
import threading
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sentiment_agent.clock import SystemClock
from sentiment_agent.types import PriceSource
from sentiment_agent.venue import public_api
from sentiment_agent.venue.public_api import (
    BASE_URL,
    PATH_HISTORY_CANDLES,
    BitgetPublicApi,
    PublicApiError,
    urllib_get,
)

OUT = Path(__file__).with_name("cassette.json")
SYMBOLS = ["BTCUSDT", "SP500USDT", "NVDAUSDT", "AAOIUSDT", "NOSUCHUSDT"]
"""Crypto, index and equity perps UTA Demo lists; AAOIUSDT is listed live but not on Demo;
NOSUCHUSDT is listed nowhere."""


def _iso(at: datetime) -> str:
    return at.astimezone(UTC).isoformat().replace("+00:00", "Z")


class _Recorder:
    def __init__(self) -> None:
        self.exchanges: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def __call__(self, url: str, headers: Mapping[str, str], timeout: float) -> tuple[int, bytes]:
        status, body = urllib_get(url, headers, timeout)
        with self._lock:
            self.exchanges.append(
                {
                    "captured_at": _iso(datetime.now(UTC)),
                    "url": url,
                    "headers": dict(headers),
                    "status": status,
                    "body": body.decode("utf-8"),
                }
            )
        return status, body


def _scenario(name: str, call: dict[str, Any], run: Any) -> dict[str, Any]:
    recorder = _Recorder()
    api = BitgetPublicApi(clock=SystemClock(), http=recorder)
    error: dict[str, Any] | None = None
    try:
        run(api)
    except PublicApiError as exc:
        error = {"code": exc.code, "http_status": exc.http_status, "message": str(exc)}
    recorder.exchanges.sort(key=lambda e: e["url"])
    print(f"{name}: {len(recorder.exchanges)} exchange(s)" + (f", error {error}" if error else ""))
    return {"name": name, "call": call, "error": error, "exchanges": recorder.exchanges}


def _raw(name: str, note: str, queries: list[dict[str, str]]) -> dict[str, Any]:
    recorder = _Recorder()
    headers = public_api._request_headers(PriceSource.LIVE, PATH_HISTORY_CANDLES)
    for query in queries:
        url = f"{BASE_URL}{PATH_HISTORY_CANDLES}?" + "&".join(f"{k}={v}" for k, v in query.items())
        recorder(url, headers, 15.0)
    print(f"{name}: {len(recorder.exchanges)} exchange(s)")
    return {
        "name": name,
        "note": note,
        "call": None,
        "error": None,
        "exchanges": recorder.exchanges,
    }


def main() -> int:
    captured_at = datetime.now(UTC)
    end = captured_at.replace(minute=0, second=0, microsecond=0)
    start_150h = end - timedelta(hours=150)
    scenarios: list[dict[str, Any]] = []

    for source in (PriceSource.LIVE, PriceSource.DEMO):
        scenarios.append(
            _scenario(
                f"instruments_{source.value}",
                {"method": "instruments", "source": source.value, "symbols": SYMBOLS},
                lambda api, s=source: api.instruments(s, SYMBOLS),
            )
        )
        scenarios.append(
            _scenario(
                f"quotes_{source.value}",
                {"method": "quotes", "source": source.value, "symbols": SYMBOLS},
                lambda api, s=source: api.quotes(s, SYMBOLS),
            )
        )

    candle_calls: list[tuple[str, PriceSource, str, str, str, datetime, datetime]] = [
        ("candles_live_btc_market_1h_2pages", PriceSource.LIVE, "BTCUSDT", "market", "1H",
         start_150h, end),
        ("candles_demo_nvda_market_1h_2pages", PriceSource.DEMO, "NVDAUSDT", "market", "1H",
         start_150h, end),
        ("candles_demo_nvda_mark_1h_2pages", PriceSource.DEMO, "NVDAUSDT", "mark", "1H",
         start_150h, end),
        ("candles_demo_nvda_index_1h_2pages", PriceSource.DEMO, "NVDAUSDT", "index", "1H",
         start_150h, end),
        ("candles_live_nvda_index_1h_1page", PriceSource.LIVE, "NVDAUSDT", "index", "1H",
         end - timedelta(hours=24), end),
        ("candles_demo_nvda_market_history_start", PriceSource.DEMO, "NVDAUSDT", "market", "1H",
         datetime(2026, 8, 20, tzinfo=UTC), datetime(2026, 8, 27, tzinfo=UTC)),
        ("candles_live_btc_market_1d", PriceSource.LIVE, "BTCUSDT", "market", "1D",
         end - timedelta(days=10), end),
        ("candles_live_unknown_symbol", PriceSource.LIVE, "NOSUCHUSDT", "market", "1H",
         end - timedelta(hours=5), end),
    ]  # fmt: skip
    for name, source, symbol, kind, interval, start, stop in candle_calls:
        call = {
            "method": "candles",
            "source": source.value,
            "symbol": symbol,
            "kind": kind,
            "interval": interval,
            "start": _iso(start),
            "end": _iso(stop),
        }
        scenarios.append(
            _scenario(
                name,
                call,
                lambda api, so=source, sy=symbol, k=kind, i=interval, a=start, b=stop: api.candles(
                    so, sy, kind=k, interval=i, start=a, end=b
                ),
            )
        )

    for name, symbol, limit in (
        ("funding_live_btc_150", "BTCUSDT", 150),
        ("funding_live_nvda_5", "NVDAUSDT", 5),
        ("funding_live_unknown_symbol", "NOSUCHUSDT", 5),
    ):
        scenarios.append(
            _scenario(
                name,
                {"method": "funding_history", "symbol": symbol, "limit": limit},
                lambda api, sy=symbol, n=limit: api.funding_history(sy, limit=n),
            )
        )

    # Raw probes that pin the venue semantics the client's pagination depends on.
    base = {"category": "USDT-FUTURES", "symbol": "BTCUSDT", "interval": "1H", "type": "market"}
    first = _Recorder()
    headers = public_api._request_headers(PriceSource.LIVE, PATH_HISTORY_CANDLES)
    first_url = f"{BASE_URL}{PATH_HISTORY_CANDLES}?" + "&".join(
        f"{k}={v}" for k, v in {**base, "limit": "3"}.items()
    )
    _, body = first(first_url, headers, 15.0)
    oldest = min(int(row[0]) for row in json.loads(body)["data"])
    probe = _raw(
        "raw_endtime_semantics",
        "history-candles limit=3: no endTime; endTime = oldest open of the first answer; "
        "endTime = that minus 1 ms. Pins that endTime filters on the bar's close.",
        [
            {**base, "limit": "3", "endTime": str(oldest)},
            {**base, "limit": "3", "endTime": str(oldest - 1)},
        ],
    )
    probe["exchanges"] = first.exchanges + probe["exchanges"]
    scenarios.append(probe)
    scenarios.append(
        _raw(
            "raw_limit_101",
            "history-candles with limit=101: the venue refuses more than 100 rows a page.",
            [{**base, "limit": "101"}],
        )
    )

    cassette = {
        "captured_at": _iso(captured_at),
        "recorder": "tests/fixtures/venue/record.py",
        "base_url": BASE_URL,
        "note": (
            "Real keyless responses from Bitget's public v3 market endpoints, live and UTA "
            "Demo (paptrading: 1). Bodies are verbatim. Replayed offline by "
            "tests/venue/test_public_api.py."
        ),
        "scenarios": scenarios,
    }
    OUT.write_text(json.dumps(cassette, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
