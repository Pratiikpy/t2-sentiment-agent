"""M1 venue: keyless Bitget public market data, live and UTA Demo.

Offline tests replay ``tests/fixtures/venue/cassette.json`` (real keyless responses recorded by
``tests/fixtures/venue/record.py``) or scripted responses. Tests marked ``live_public`` call the
real public endpoints to catch schema drift; they run only with ``--run-live-public``.
"""

import hashlib
import http.server
import inspect
import io
import json
import threading
import urllib.error
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from email.message import Message
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from sentiment_agent import __version__
from sentiment_agent.clock import ManualClock, SystemClock
from sentiment_agent.types import (
    PROJECT_SLUG,
    BlobRef,
    Category,
    MarketData,
    PriceSource,
    SourceCall,
    SourceHealth,
    ToolkitSurface,
)
from sentiment_agent.venue import public_api
from sentiment_agent.venue.public_api import (
    BASE_URL,
    DEMO_ENDPOINTS,
    NOT_LISTED_CODE,
    PAGE_LIMIT,
    PATH_HISTORY_CANDLES,
    PATH_HISTORY_FUND_RATE,
    PATH_INSTRUMENTS,
    PATH_TICKERS,
    USER_AGENT,
    BitgetPublicApi,
    PublicApiError,
    ResponseTooLargeError,
    parse_candle,
    parse_instrument,
    parse_ticker,
    urllib_get,
)

CASSETTE_PATH = Path(__file__).parents[1] / "fixtures" / "venue" / "cassette.json"
T0 = datetime(2026, 9, 23, 13, 0, tzinfo=UTC)

TICKER_KEYS = frozenset(
    {
        "symbol",
        "ts",
        "lastPrice",
        "markPrice",
        "indexPrice",
        "bid1Price",
        "ask1Price",
        "fundingRate",
        "openInterest",
        "turnover24h",
        "price24hPcnt",
    }
)
INSTRUMENT_KEYS = frozenset(
    {
        "symbol",
        "category",
        "baseCoin",
        "quoteCoin",
        "status",
        "minOrderQty",
        "quantityMultiplier",
        "priceMultiplier",
        "minOrderAmount",
        "maxMarketOrderQty",
        "maxOrderQty",
        "takerFeeRate",
        "makerFeeRate",
        "maxLeverage",
        "fundInterval",
    }
)
FUNDING_KEYS = frozenset({"symbol", "fundingRate", "fundingRateTimestamp"})


# ================================================================================================
# Harness
# ================================================================================================


class FakeTime:
    """Stands in for the module's monotonic timer and sleep. Time moves only when slept."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []
        self._lock = threading.RLock()

    def monotonic(self) -> float:
        with self._lock:
            return self.now

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self.sleeps.append(seconds)
            self.now += seconds

    def advance(self, seconds: float) -> None:
        with self._lock:
            self.now += seconds


@pytest.fixture(autouse=True)
def fake_time(monkeypatch: pytest.MonkeyPatch) -> FakeTime:
    """No test in this module ever sleeps for real, and every timing is observable."""
    fake = FakeTime()
    monkeypatch.setattr(public_api, "_monotonic", fake.monotonic)
    monkeypatch.setattr(public_api, "_sleep", fake.sleep)
    return fake


Response = tuple[int, bytes] | BaseException
Request = tuple[str, dict[str, str], float]


class ScriptedHttp:
    """An ``HttpGet`` that answers from a handler and records every request."""

    def __init__(self, handler: Callable[[str, dict[str, str]], Response]) -> None:
        self._handler = handler
        self.requests: list[Request] = []
        self._lock = threading.Lock()

    def __call__(self, url: str, headers: Mapping[str, str], timeout: float) -> tuple[int, bytes]:
        with self._lock:
            self.requests.append((url, dict(headers), timeout))
        answer = self._handler(url, dict(headers))
        if isinstance(answer, BaseException):
            raise answer
        return answer

    @property
    def urls(self) -> list[str]:
        return [r[0] for r in self.requests]


def sequence(*answers: Response) -> Callable[[str, dict[str, str]], Response]:
    """Answers in order, one per request, whatever the URL."""
    queue = list(answers)
    lock = threading.Lock()

    def handler(_url: str, _headers: dict[str, str]) -> Response:
        with lock:
            if not queue:
                raise AssertionError("more requests than scripted answers")
            return queue.pop(0)

    return handler


def envelope(data: Any, *, code: str = "00000", msg: str = "success") -> bytes:
    return json.dumps({"code": code, "msg": msg, "requestTime": 1, "data": data}).encode()


def query_of(url: str) -> dict[str, str]:
    from urllib.parse import parse_qsl, urlsplit

    return dict(parse_qsl(urlsplit(url).query))


def path_of(url: str) -> str:
    from urllib.parse import urlsplit

    return urlsplit(url).path


class FakeBlobs:
    def __init__(self) -> None:
        self.stored: dict[str, tuple[bytes, str]] = {}

    def put(self, data: bytes, media_type: str) -> BlobRef:
        sha = hashlib.sha256(data).hexdigest()
        self.stored[sha] = (data, media_type)
        return BlobRef(sha256=sha, media_type=media_type, size=len(data))

    def get(self, sha256: str) -> bytes:
        return self.stored[sha256][0]


def api_with(
    handler: Callable[[str, dict[str, str]], Response],
    clock: ManualClock,
    **kwargs: Any,
) -> tuple[BitgetPublicApi, ScriptedHttp]:
    http = ScriptedHttp(handler)
    return BitgetPublicApi(clock=clock, http=http, **kwargs), http


def ticker_row(symbol: str = "NVDAUSDT", **overrides: str) -> dict[str, str]:
    row = {
        "category": "USDT-FUTURES",
        "symbol": symbol,
        "ts": "1790248031182",
        "lastPrice": "223.27",
        "ask1Price": "223.29",
        "bid1Price": "223.28",
        "price24hPcnt": "-0.02229",
        "turnover24h": "10788585.9476",
        "indexPrice": "223.0600829147522046",
        "markPrice": "223.21",
        "fundingRate": "0",
        "openInterest": "70718.299999999926",
    }
    row.update(overrides)
    return row


def hour_ms(i: int) -> int:
    """Open time, in ms, of synthetic hourly bar ``i`` (bar 0 opens at 2026-09-01 00:00 UTC)."""
    return int(datetime(2026, 9, 1, tzinfo=UTC).timestamp() * 1000) + i * 3_600_000


def synthetic_candle_venue(
    n_bars: int, *, honour_end_time: bool = True
) -> Callable[[str, dict[str, str]], Response]:
    """A history-candles venue with the measured semantics: closed bars only, ``endTime`` filters on
    the bar's close, the newest ``limit`` bars come back oldest first."""

    def handler(url: str, _headers: dict[str, str]) -> Response:
        q = query_of(url)
        limit = int(q["limit"])
        if limit > PAGE_LIMIT:
            return 400, envelope(None, code="40020", msg="Parameter limit error")
        end = int(q["endTime"]) if honour_end_time and "endTime" in q else hour_ms(n_bars)
        eligible = [i for i in range(n_bars) if hour_ms(i) + 3_600_000 <= end]
        page = eligible[-limit:]
        rows = [[str(hour_ms(i)), "10", "11", "9", "10.5", "1", "10"] for i in page]
        return 200, envelope(rows)

    return handler


# ================================================================================================
# Cassette replay: real recorded responses
# ================================================================================================


def load_cassette() -> dict[str, Any]:
    data = json.loads(CASSETTE_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


CASSETTE = load_cassette()
SCENARIOS: dict[str, dict[str, Any]] = {s["name"]: s for s in CASSETTE["scenarios"]}
CAPTURED_AT = datetime.fromisoformat(CASSETTE["captured_at"])


def replay(name: str) -> ScriptedHttp:
    exchanges = {e["url"]: e for e in SCENARIOS[name]["exchanges"]}

    def handler(url: str, _headers: dict[str, str]) -> Response:
        exchange = exchanges.get(url)
        if exchange is None:
            raise AssertionError(f"request not in the cassette: {url}")
        return int(exchange["status"]), str(exchange["body"]).encode("utf-8")

    return ScriptedHttp(handler)


def run_scenario(name: str, api: BitgetPublicApi) -> Any:
    call = SCENARIOS[name]["call"]
    method = call["method"]
    if method in ("instruments", "quotes"):
        return getattr(api, method)(PriceSource(call["source"]), call["symbols"])
    if method == "candles":
        return api.candles(
            PriceSource(call["source"]),
            call["symbol"],
            kind=call["kind"],
            interval=call["interval"],
            start=datetime.fromisoformat(call["start"]),
            end=datetime.fromisoformat(call["end"]),
        )
    if method == "funding_history":
        return api.funding_history(call["symbol"], limit=call["limit"])
    raise AssertionError(method)


def replayed(name: str) -> tuple[Any, BitgetPublicApi, ScriptedHttp]:
    http = replay(name)
    api = BitgetPublicApi(clock=ManualClock(CAPTURED_AT), http=http)
    return run_scenario(name, api), api, http


def raw_body(name: str, needle: str) -> dict[str, Any]:
    for exchange in SCENARIOS[name]["exchanges"]:
        if needle in exchange["url"]:
            body = json.loads(exchange["body"])
            assert isinstance(body, dict)
            return body
    raise AssertionError(f"{needle} not in {name}")


def test_cassette_is_real_and_stamped() -> None:
    assert CASSETTE["base_url"] == BASE_URL
    assert CAPTURED_AT.tzinfo is not None
    assert CAPTURED_AT.year == 2026
    for scenario in CASSETTE["scenarios"]:
        assert scenario["exchanges"], scenario["name"]
        for exchange in scenario["exchanges"]:
            assert exchange["url"].startswith(BASE_URL + "/api/v3/market/")
            assert datetime.fromisoformat(exchange["captured_at"]) >= CAPTURED_AT
            assert not any(h.lower().startswith("access-") for h in exchange["headers"])


def test_recorded_rows_carry_every_field_the_parsers_read() -> None:
    for name in ("quotes_live", "quotes_demo"):
        body = raw_body(name, "symbol=NVDAUSDT")
        assert set(body["data"][0]) >= TICKER_KEYS, name
    for name in ("instruments_live", "instruments_demo"):
        body = raw_body(name, "symbol=NVDAUSDT")
        assert set(body["data"][0]) >= INSTRUMENT_KEYS, name
    body = raw_body("funding_live_btc_150", "cursor=1")
    assert set(body["data"]["resultList"][0]) >= FUNDING_KEYS
    row = raw_body("candles_live_btc_market_1h_2pages", "type=market")["data"][0]
    assert len(row) == 7


@pytest.mark.parametrize("source", [PriceSource.LIVE, PriceSource.DEMO])
def test_instruments_parse_recorded_rows(source: PriceSource) -> None:
    name = f"instruments_{source.value}"
    specs, api, http = replayed(name)
    assert set(specs) == (
        {"BTCUSDT", "SP500USDT", "NVDAUSDT", "AAOIUSDT"}
        if source is PriceSource.LIVE
        else {"BTCUSDT", "SP500USDT", "NVDAUSDT"}
    )
    assert list(specs) == [s for s in SCENARIOS[name]["call"]["symbols"] if s in specs]
    for symbol, spec in specs.items():
        raw = raw_body(name, f"symbol={symbol}")["data"][0]
        assert spec.symbol == symbol
        assert spec.source is source
        assert spec.category is Category.USDT_FUTURES
        assert spec.qty_step == Decimal(raw["quantityMultiplier"])
        assert spec.price_step == Decimal(raw["priceMultiplier"])
        assert spec.min_order_qty == Decimal(raw["minOrderQty"])
        assert spec.min_order_amount == Decimal(raw["minOrderAmount"])
        assert spec.taker_fee_rate == Decimal("0.0006")
        assert spec.maker_fee_rate == Decimal("0.0002")
        assert spec.fund_interval_hours == int(raw["fundInterval"])
        assert spec.max_leverage == int(raw["maxLeverage"])
        mmq = raw["maxMarketOrderQty"]
        assert spec.max_market_order_qty == (Decimal(mmq) if mmq else None)
        assert spec.fetched_at == CAPTURED_AT
    calls = api.drain_calls()
    assert [c.source for c in calls] == [
        f"instruments.{source.value}:{s}" for s in SCENARIOS[name]["call"]["symbols"]
    ]
    health = {c.params["symbol"]: c.health for c in calls}
    assert health["NOSUCHUSDT"] is SourceHealth.EMPTY
    if source is PriceSource.DEMO:
        assert health["AAOIUSDT"] is SourceHealth.EMPTY
        assert all(h is SourceHealth.OK for s, h in health.items() if s in specs)
    assert len(http.requests) == len(SCENARIOS[name]["call"]["symbols"])


def test_demo_instrument_limits_match_the_design() -> None:
    specs, _, _ = replayed("instruments_demo")
    nvda = specs["NVDAUSDT"]
    assert nvda.min_order_qty == Decimal("0.01")
    assert nvda.qty_step == Decimal("0.01")
    assert nvda.max_market_order_qty == Decimal("60")
    assert nvda.max_leverage == 25
    assert nvda.min_order_amount == Decimal("5")
    # Demo publishes no market-order cap for BTCUSDT: an empty string, parsed as "not measured".
    assert specs["BTCUSDT"].max_market_order_qty is None
    assert specs["BTCUSDT"].max_order_qty is None


@pytest.mark.parametrize("source", [PriceSource.LIVE, PriceSource.DEMO])
def test_quotes_parse_recorded_rows_exactly(source: PriceSource) -> None:
    name = f"quotes_{source.value}"
    quotes, api, _ = replayed(name)
    assert "NOSUCHUSDT" not in quotes
    assert ("AAOIUSDT" in quotes) is (source is PriceSource.LIVE)
    for symbol, quote in quotes.items():
        raw = raw_body(name, f"symbol={symbol}")["data"][0]
        assert quote.symbol == symbol
        assert quote.source is source
        assert quote.ts == datetime.fromtimestamp(int(raw["ts"]) / 1000, UTC)
        assert quote.last == Decimal(raw["lastPrice"])
        assert quote.mark == Decimal(raw["markPrice"])
        assert str(quote.index) == raw["indexPrice"]  # every digit kept
        assert quote.bid == Decimal(raw["bid1Price"])
        assert quote.ask == Decimal(raw["ask1Price"])
        assert quote.price_change_24h == Decimal(raw["price24hPcnt"])
        assert quote.open_interest == Decimal(raw["openInterest"])
        assert quote.fetched_at == CAPTURED_AT
        assert quote.bid <= quote.ask
        assert abs(quote.price_change_24h or 0) < 1  # a fraction, not a percentage
    calls = api.drain_calls()
    assert all(c.surface is ToolkitSurface.PUBLIC_MARKET_API for c in calls)
    missing = [c for c in calls if c.params["symbol"] == "NOSUCHUSDT"]
    assert len(missing) == 1
    assert missing[0].health is SourceHealth.EMPTY
    assert missing[0].error is not None
    assert missing[0].error.startswith(NOT_LISTED_CODE)


def test_demo_and_live_recorded_quotes_are_different_venues() -> None:
    live, _, _ = replayed("quotes_live")
    demo, _, _ = replayed("quotes_demo")
    assert live["NVDAUSDT"].turnover_24h != demo["NVDAUSDT"].turnover_24h
    assert "AAOIUSDT" in live
    assert "AAOIUSDT" not in demo


@pytest.mark.parametrize(
    "name",
    [
        "candles_live_btc_market_1h_2pages",
        "candles_demo_nvda_market_1h_2pages",
        "candles_demo_nvda_mark_1h_2pages",
        "candles_demo_nvda_index_1h_2pages",
    ],
)
def test_recorded_two_page_walk_is_gapless(name: str) -> None:
    candles, api, http = replayed(name)
    call = SCENARIOS[name]["call"]
    start = datetime.fromisoformat(call["start"])
    end = datetime.fromisoformat(call["end"])
    assert len(http.requests) == 2
    first, second = (query_of(u) for u in http.urls)
    assert first["endTime"] == str(int(end.timestamp() * 1000))
    first_page = json.loads(
        next(e for e in SCENARIOS[name]["exchanges"] if e["url"] == http.urls[0])["body"]
    )["data"]
    assert second["endTime"] == min(r[0] for r in first_page)  # next page ends at the oldest open
    assert first["limit"] == second["limit"] == str(PAGE_LIMIT)
    assert len(candles) == 150
    assert all(b.open_time - a.open_time == timedelta(hours=1) for a, b in pairwise(candles))
    assert candles[0].open_time == start
    assert candles[-1].open_time + timedelta(hours=1) == end
    assert all(c.kind == call["kind"] and c.source.value == call["source"] for c in candles)
    if call["kind"] == "market":
        assert all(c.volume is not None for c in candles)
    else:
        assert all(c.volume is None for c in candles)
    calls = api.drain_calls()
    assert [c.source for c in calls] == [
        f"history-candles.{call['source']}:{call['symbol']}:{call['kind']}:1H",
        f"history-candles.{call['source']}:{call['symbol']}:{call['kind']}:1H#p2",
    ]
    assert [c.rows for c in calls] == [PAGE_LIMIT, PAGE_LIMIT]
    assert all(c.health is SourceHealth.OK for c in calls)


def test_recorded_short_page_ends_the_walk_at_the_demo_history_start() -> None:
    candles, _, http = replayed("candles_demo_nvda_market_history_start")
    assert len(http.requests) == 1  # fewer than 100 rows: nothing older exists
    assert candles[0].open_time == datetime(2026, 8, 25, 6, tzinfo=UTC)
    assert candles[-1].open_time == datetime(2026, 8, 26, 23, tzinfo=UTC)
    assert len(candles) == 42


def test_recorded_daily_bars_open_at_1600_utc() -> None:
    candles, _, _ = replayed("candles_live_btc_market_1d")
    assert candles
    assert {c.open_time.hour for c in candles} == {16}
    assert all(b.open_time - a.open_time == timedelta(days=1) for a, b in pairwise(candles))


def test_recorded_live_index_candles_parse() -> None:
    candles, _, _ = replayed("candles_live_nvda_index_1h_1page")
    assert len(candles) == 24
    assert all(c.volume is None and c.source is PriceSource.LIVE for c in candles)


@pytest.mark.parametrize(
    ("name", "path"),
    [
        ("candles_live_unknown_symbol", PATH_HISTORY_CANDLES),
        ("funding_live_unknown_symbol", PATH_HISTORY_FUND_RATE),
    ],
)
def test_recorded_unknown_symbol_raises_with_the_venue_code(name: str, path: str) -> None:
    http = replay(name)
    api = BitgetPublicApi(clock=ManualClock(CAPTURED_AT), http=http)
    with pytest.raises(PublicApiError) as caught:
        run_scenario(name, api)
    assert caught.value.code == NOT_LISTED_CODE
    assert caught.value.path == path
    assert caught.value.http_status == 400
    (call,) = api.drain_calls()
    assert call.health is SourceHealth.EMPTY
    assert call.rows == 0


def test_recorded_funding_history_pages_by_cursor() -> None:
    points, api, http = replayed("funding_live_btc_150")
    assert [query_of(u)["cursor"] for u in http.urls] == ["1", "2"]
    assert all(query_of(u)["limit"] == "100" for u in http.urls)
    assert len(points) == 150
    assert all(a.ts < b.ts for a, b in pairwise(points))
    assert all(p.source is PriceSource.LIVE and p.symbol == "BTCUSDT" for p in points)
    newest = raw_body("funding_live_btc_150", "cursor=1")["data"]["resultList"][0]
    assert points[-1].rate == Decimal(newest["fundingRate"])
    assert points[-1].ts == datetime.fromtimestamp(int(newest["fundingRateTimestamp"]) / 1000, UTC)
    assert all(b.ts - a.ts == timedelta(hours=8) for a, b in pairwise(points))
    assert [c.source for c in api.drain_calls()] == [
        "history-fund-rate.live:BTCUSDT",
        "history-fund-rate.live:BTCUSDT#p2",
    ]


def test_recorded_funding_small_limit_is_one_page() -> None:
    points, _, http = replayed("funding_live_nvda_5")
    assert len(http.requests) == 1
    assert len(points) == 5


def test_recorded_end_time_filters_on_bar_close() -> None:
    """Pins the venue behaviour the walk-back relies on, which validation/fetch.py:111 misses."""
    exchanges = SCENARIOS["raw_endtime_semantics"]["exchanges"]
    bodies = {e["url"]: json.loads(e["body"])["data"] for e in exchanges}
    no_end = next(rows for url, rows in bodies.items() if "endTime" not in url)
    oldest = min(int(r[0]) for r in no_end)
    at_oldest = bodies[next(u for u in bodies if f"endTime={oldest}" in u)]
    minus_one = bodies[next(u for u in bodies if f"endTime={oldest - 1}" in u)]
    hour = 3_600_000
    assert [int(r[0]) for r in at_oldest] == [oldest - 3 * hour, oldest - 2 * hour, oldest - hour]
    assert [int(r[0]) for r in minus_one] == [
        oldest - 4 * hour,
        oldest - 3 * hour,
        oldest - 2 * hour,
    ]


def test_recorded_limit_101_answer_surfaces_its_code(clock: ManualClock) -> None:
    exchange = SCENARIOS["raw_limit_101"]["exchanges"][0]
    api, _ = api_with(lambda _u, _h: (int(exchange["status"]), exchange["body"].encode()), clock)
    with pytest.raises(PublicApiError) as caught:
        api.funding_history("BTCUSDT", limit=5)
    assert caught.value.code == "40020"
    assert caught.value.http_status == 400


# ================================================================================================
# Headers: Demo only where measured, never a credential, explicit User-Agent
# ================================================================================================


def all_replayed_requests() -> list[tuple[PriceSource, Request]]:
    """Every request the client makes when replaying each recorded call, with the price source the
    call asked for (funding history is live by design)."""
    requests: list[tuple[PriceSource, Request]] = []
    for name, scenario in SCENARIOS.items():
        if scenario["call"] is None:
            continue
        source = PriceSource(scenario["call"].get("source", PriceSource.LIVE.value))
        http = replay(name)
        api = BitgetPublicApi(clock=ManualClock(CAPTURED_AT), http=http)
        try:
            run_scenario(name, api)
        except PublicApiError:
            assert scenario["error"] is not None, name
        requests.extend((source, request) for request in http.requests)
    return requests


def test_paptrading_header_only_for_demo_and_only_on_demo_endpoints() -> None:
    requests = all_replayed_requests()
    sources = {source for source, _ in requests}
    assert sources == {PriceSource.LIVE, PriceSource.DEMO}
    for source, (url, headers, _) in requests:
        lower = {k.lower(): v for k, v in headers.items()}
        if source is PriceSource.DEMO:
            assert lower.get("paptrading") == "1", url
            assert path_of(url) in DEMO_ENDPOINTS
        else:
            assert "paptrading" not in lower, url


def test_recorded_exchanges_carried_the_header_they_claim() -> None:
    for scenario in SCENARIOS.values():
        demo = scenario["call"] is not None and scenario["call"].get("source") == "demo"
        for exchange in scenario["exchanges"]:
            assert (exchange["headers"].get("paptrading") == "1") is demo, scenario["name"]


def test_no_access_header_is_ever_sent_and_user_agent_is_explicit() -> None:
    for _, (_, headers, timeout) in all_replayed_requests():
        assert not any(name.lower().startswith("access-") for name in headers)
        assert headers["User-Agent"] == USER_AGENT
        assert timeout == 15.0
    assert USER_AGENT.startswith(f"{PROJECT_SLUG}/{__version__}")
    assert "urllib" not in USER_AGENT.lower()


def test_funding_history_is_live_only(clock: ManualClock) -> None:
    api, http = api_with(sequence((200, envelope({"resultList": []}))), clock)
    assert api.funding_history("BTCUSDT", limit=5) == []
    ((_, headers, _),) = http.requests
    assert "paptrading" not in {k.lower() for k in headers}


def test_demo_read_of_an_unmeasured_path_is_refused() -> None:
    with pytest.raises(ValueError, match="not measured"):
        public_api._request_headers(PriceSource.DEMO, "/api/v3/market/open-interest")
    live = public_api._request_headers(PriceSource.LIVE, "/api/v3/market/open-interest")
    assert "paptrading" not in live


@pytest.mark.parametrize("path", sorted(DEMO_ENDPOINTS))
def test_demo_headers_on_every_demo_endpoint(path: str) -> None:
    headers = public_api._request_headers(PriceSource.DEMO, path)
    assert headers == {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "locale": "en-US",
        "paptrading": "1",
    }


def test_demo_endpoints_are_exactly_the_four_measured_paths() -> None:
    measured = {PATH_TICKERS, PATH_INSTRUMENTS, PATH_HISTORY_CANDLES, PATH_HISTORY_FUND_RATE}
    assert measured == DEMO_ENDPOINTS


@pytest.mark.parametrize("name", ["ACCESS-KEY", "access-sign", "Access-Passphrase"])
def test_an_access_header_is_refused(name: str) -> None:
    with pytest.raises(RuntimeError, match="no credential"):
        public_api._assert_keyless({"User-Agent": USER_AGENT, name: "x"})


def test_module_source_never_names_a_credential() -> None:
    source = Path(public_api.__file__).read_text(encoding="utf-8")
    for needle in ("BITGET_API_KEY", "SECRET", "PASSPHRASE", "environ", ".secrets", "hmac"):
        assert needle not in source, needle


# ================================================================================================
# Errors and SourceCall health
# ================================================================================================


def test_non_00000_code_raises_with_the_code_and_is_not_retried(clock: ManualClock) -> None:
    api, http = api_with(
        sequence((200, envelope(None, code="40034", msg="Parameter does not exist"))), clock
    )
    with pytest.raises(PublicApiError) as caught:
        api.funding_history("BTCUSDT", limit=5)
    assert caught.value.code == "40034"
    assert caught.value.http_status == 200
    assert len(http.requests) == 1
    (call,) = api.drain_calls()
    assert call.health is SourceHealth.ERROR
    assert call.error == "40034: Parameter does not exist (HTTP 200)"


def test_timeout_is_recorded_as_timeout_and_not_retried(
    clock: ManualClock, fake_time: FakeTime
) -> None:
    api, http = api_with(sequence(TimeoutError("read timed out")), clock)
    with pytest.raises(PublicApiError) as caught:
        api.candles(
            PriceSource.LIVE,
            "BTCUSDT",
            kind="market",
            interval="1H",
            start=T0 - timedelta(hours=5),
            end=T0,
        )
    assert caught.value.code is None
    assert len(http.requests) == 1
    assert fake_time.sleeps == []
    (call,) = api.drain_calls()
    assert call.health is SourceHealth.TIMEOUT
    assert call.rows == 0
    assert call.blob is None
    assert call.error == "timeout after 15 s"
    assert call.started_at == T0


def test_urlerror_wrapping_a_timeout_is_a_timeout(clock: ManualClock) -> None:
    api, _ = api_with(sequence(urllib.error.URLError(TimeoutError("connect"))), clock)
    with pytest.raises(PublicApiError):
        api.funding_history("BTCUSDT", limit=1)
    (call,) = api.drain_calls()
    assert call.health is SourceHealth.TIMEOUT


def test_connection_errors_are_retried_with_backoff(
    clock: ManualClock, fake_time: FakeTime
) -> None:
    ok = (200, envelope({"resultList": [{"symbol": "BTCUSDT", "fundingRate": "0.0001",
                                          "fundingRateTimestamp": "1790236800000"}]}))  # fmt: skip
    api, http = api_with(
        sequence(ConnectionResetError("reset"), ConnectionResetError("reset"), ok),
        clock,
        min_interval_s=0,
    )
    points = api.funding_history("BTCUSDT", limit=1)
    assert len(points) == 1
    assert len(http.requests) == 3
    assert fake_time.sleeps == [0.25, 0.5]
    (call,) = api.drain_calls()
    assert call.health is SourceHealth.OK
    assert call.latency_ms == 750


def test_connection_errors_exhaust_retries(clock: ManualClock) -> None:
    api, http = api_with(sequence(*(ConnectionRefusedError("refused") for _ in range(3))), clock)
    with pytest.raises(PublicApiError) as caught:
        api.funding_history("BTCUSDT", limit=1)
    assert caught.value.code is None
    assert len(http.requests) == 3
    (call,) = api.drain_calls()
    assert call.health is SourceHealth.ERROR
    assert call.error is not None
    assert "ConnectionRefusedError" in call.error
    assert "attempt 3" in call.error


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_retryable_statuses_are_retried(status: int, clock: ManualClock) -> None:
    ok = (200, envelope({"resultList": []}))
    api, http = api_with(sequence((status, b"busy"), ok), clock)
    assert api.funding_history("BTCUSDT", limit=1) == []
    assert len(http.requests) == 2
    (call,) = api.drain_calls()
    assert call.health is SourceHealth.EMPTY


def test_rate_limit_that_persists_is_an_error_with_the_venue_code(clock: ManualClock) -> None:
    limited = (429, envelope(None, code="429", msg="Too Many Requests"))
    api, http = api_with(sequence(limited, limited, limited), clock)
    with pytest.raises(PublicApiError) as caught:
        api.funding_history("BTCUSDT", limit=1)
    assert caught.value.code == "429"
    assert caught.value.http_status == 429
    assert len(http.requests) == 3
    (call,) = api.drain_calls()
    assert call.health is SourceHealth.ERROR
    assert call.error == "429: Too Many Requests (HTTP 429 after 3 attempts)"


def test_a_non_json_body_is_an_error(clock: ManualClock) -> None:
    page = (502, b"<html>Bad gateway</html>")
    blobs = FakeBlobs()
    api, _ = api_with(sequence(page, page, page), clock, blobs=blobs)
    with pytest.raises(PublicApiError) as caught:
        api.funding_history("BTCUSDT", limit=1)
    assert caught.value.code is None
    assert caught.value.http_status == 502
    (call,) = api.drain_calls()
    assert call.health is SourceHealth.ERROR
    assert call.blob is not None
    assert blobs.stored[call.blob.sha256] == (
        b"<html>Bad gateway</html>",
        "application/octet-stream",
    )


def test_an_unexpected_shape_is_an_error(clock: ManualClock) -> None:
    api, _ = api_with(sequence((200, envelope({"rows": []}))), clock)
    with pytest.raises(PublicApiError) as caught:
        api.candles(
            PriceSource.DEMO,
            "NVDAUSDT",
            kind="index",
            interval="1H",
            start=T0 - timedelta(hours=3),
            end=T0,
        )
    assert caught.value.code is None
    (call,) = api.drain_calls()
    assert call.health is SourceHealth.ERROR
    assert call.error is not None
    assert call.error.startswith("unexpected response shape")


def test_an_empty_required_price_is_hollow(clock: ManualClock) -> None:
    api, _ = api_with(sequence((200, envelope([ticker_row(lastPrice="")]))), clock)
    with pytest.raises(PublicApiError):
        api.quotes(PriceSource.DEMO, ["NVDAUSDT"])
    (call,) = api.drain_calls()
    assert call.health is SourceHealth.HOLLOW
    assert call.error == "hollow response: 'lastPrice' is empty"


def test_an_answer_with_no_matching_row_is_empty(clock: ManualClock) -> None:
    api, _ = api_with(sequence((200, envelope([]))), clock)
    assert api.quotes(PriceSource.LIVE, ["NVDAUSDT"]) == {}
    (call,) = api.drain_calls()
    assert call.health is SourceHealth.EMPTY
    assert call.error is None


def test_a_row_for_another_symbol_is_ignored(clock: ManualClock) -> None:
    api, _ = api_with(sequence((200, envelope([ticker_row("TSLAUSDT")]))), clock)
    assert api.quotes(PriceSource.LIVE, ["NVDAUSDT"]) == {}


def test_partial_failure_keeps_what_answered(clock: ManualClock) -> None:
    def handler(url: str, _h: dict[str, str]) -> Response:
        symbol = query_of(url)["symbol"]
        if symbol == "AAPLUSDT":
            return TimeoutError("slow")
        if symbol == "AAOIUSDT":
            return 400, envelope(None, code=NOT_LISTED_CODE, msg="Trading pair does not exist")
        return 200, envelope([ticker_row(symbol)])

    api, _ = api_with(handler, clock)
    quotes = api.quotes(PriceSource.DEMO, ["AAPLUSDT", "AAOIUSDT", "NVDAUSDT"])
    assert list(quotes) == ["NVDAUSDT"]
    calls = api.drain_calls()
    assert [(c.params["symbol"], c.health) for c in calls] == [
        ("AAPLUSDT", SourceHealth.TIMEOUT),
        ("AAOIUSDT", SourceHealth.EMPTY),
        ("NVDAUSDT", SourceHealth.OK),
    ]


def test_a_total_outage_raises_rather_than_returning_empty(clock: ManualClock) -> None:
    api, _ = api_with(lambda _u, _h: TimeoutError("down"), clock)
    with pytest.raises(PublicApiError):
        api.quotes(PriceSource.LIVE, ["BTCUSDT", "NVDAUSDT"])
    assert [c.health for c in api.drain_calls()] == [SourceHealth.TIMEOUT] * 2


def test_nothing_listed_is_an_empty_answer_not_an_outage(clock: ManualClock) -> None:
    not_listed = (400, envelope(None, code=NOT_LISTED_CODE, msg="does not exist"))
    api, _ = api_with(lambda _u, _h: not_listed, clock)
    assert api.instruments(PriceSource.DEMO, ["AAOIUSDT", "AXTIUSDT"]) == {}


def test_hollow_symbol_does_not_sink_the_others(clock: ManualClock) -> None:
    def handler(url: str, _h: dict[str, str]) -> Response:
        symbol = query_of(url)["symbol"]
        overrides = {"bid1Price": ""} if symbol == "TSLAUSDT" else {}
        return 200, envelope([ticker_row(symbol, **overrides)])

    api, _ = api_with(handler, clock)
    assert list(api.quotes(PriceSource.DEMO, ["TSLAUSDT", "NVDAUSDT"])) == ["NVDAUSDT"]


def test_a_programming_error_is_not_swallowed(clock: ManualClock) -> None:
    def handler(_u: str, _h: dict[str, str]) -> Response:
        raise KeyError("bug in the transport")

    api, _ = api_with(handler, clock)
    with pytest.raises(KeyError):
        api.quotes(PriceSource.LIVE, ["NVDAUSDT", "BTCUSDT"])


# ================================================================================================
# Pagination
# ================================================================================================


def test_candles_walk_back_by_end_time_until_start(clock: ManualClock) -> None:
    api, http = api_with(synthetic_candle_venue(250), clock)
    start = datetime.fromtimestamp(hour_ms(0) / 1000, UTC)
    end = datetime.fromtimestamp(hour_ms(250) / 1000, UTC)
    candles = api.candles(
        PriceSource.LIVE, "BTCUSDT", kind="market", interval="1H", start=start, end=end
    )
    assert [c.open_time for c in candles] == [
        datetime.fromtimestamp(hour_ms(i) / 1000, UTC) for i in range(250)
    ]
    assert [query_of(u)["endTime"] for u in http.urls] == [
        str(hour_ms(250)),
        str(hour_ms(150)),
        str(hour_ms(50)),
    ]
    assert all(query_of(u)["limit"] == "100" for u in http.urls)
    assert [c.rows for c in api.drain_calls()] == [100, 100, 50]


def test_candles_stop_once_start_is_covered(clock: ManualClock) -> None:
    api, http = api_with(synthetic_candle_venue(500), clock)
    start = datetime.fromtimestamp(hour_ms(420) / 1000, UTC)
    end = datetime.fromtimestamp(hour_ms(500) / 1000, UTC)
    candles = api.candles(
        PriceSource.DEMO, "NVDAUSDT", kind="mark", interval="1H", start=start, end=end
    )
    assert len(http.requests) == 1
    assert len(candles) == 80
    assert candles[0].open_time == start


def test_candles_trim_to_bars_fully_inside_the_window(clock: ManualClock) -> None:
    rows = [[str(hour_ms(i)), "1", "1", "1", "1", "0", "0"] for i in (0, 1, 2, 3)]
    api, _ = api_with(sequence((200, envelope(rows))), clock)
    start = datetime.fromtimestamp(hour_ms(1) / 1000, UTC) - timedelta(minutes=30)
    end = datetime.fromtimestamp(hour_ms(3) / 1000, UTC) + timedelta(minutes=30)
    candles = api.candles(
        PriceSource.LIVE, "BTCUSDT", kind="index", interval="1H", start=start, end=end
    )
    # Bar 0 opened before start; bar 3 had not closed by end (a forming bar is never kept).
    assert [c.open_time for c in candles] == [
        datetime.fromtimestamp(hour_ms(i) / 1000, UTC) for i in (1, 2)
    ]


def test_candles_raise_when_the_venue_ignores_end_time(clock: ManualClock) -> None:
    api, http = api_with(synthetic_candle_venue(300, honour_end_time=False), clock)
    with pytest.raises(PublicApiError, match="no progress"):
        api.candles(
            PriceSource.LIVE,
            "BTCUSDT",
            kind="market",
            interval="1H",
            start=datetime.fromtimestamp(hour_ms(0) / 1000, UTC),
            end=datetime.fromtimestamp(hour_ms(300) / 1000, UTC),
        )
    assert len(http.requests) == 2
    assert len(api.drain_calls()) == 2  # the calls made are still recorded


def test_candles_empty_first_page_is_an_empty_series(clock: ManualClock) -> None:
    api, _ = api_with(sequence((200, envelope([]))), clock)
    assert (
        api.candles(
            PriceSource.DEMO,
            "NVDAUSDT",
            kind="market",
            interval="1H",
            start=T0 - timedelta(hours=2),
            end=T0,
        )
        == []
    )
    (call,) = api.drain_calls()
    assert call.health is SourceHealth.EMPTY


def test_duplicate_open_times_in_one_page_are_refused(clock: ManualClock) -> None:
    row = [str(hour_ms(0)), "1", "1", "1", "1", "0", "0"]
    api, _ = api_with(sequence((200, envelope([row, row]))), clock)
    with pytest.raises(PublicApiError, match="duplicate"):
        api.candles(
            PriceSource.LIVE,
            "BTCUSDT",
            kind="market",
            interval="1H",
            start=datetime.fromtimestamp(hour_ms(0) / 1000, UTC),
            end=datetime.fromtimestamp(hour_ms(2) / 1000, UTC),
        )


def funding_page(first_ts_ms: int, n: int, symbol: str = "BTCUSDT") -> bytes:
    step = 8 * 3_600_000
    rows = [
        {
            "symbol": symbol,
            "fundingRate": "0.0001",
            "fundingRateTimestamp": str(first_ts_ms - i * step),
        }
        for i in range(n)
    ]
    return envelope({"resultList": rows})


def test_funding_pages_until_the_limit_is_met(clock: ManualClock) -> None:
    top = hour_ms(2000)
    step = 8 * 3_600_000
    api, http = api_with(
        sequence(
            (200, funding_page(top, 100)),
            (200, funding_page(top - 100 * step, 100)),
            (200, funding_page(top - 200 * step, 100)),
        ),
        clock,
    )
    points = api.funding_history("BTCUSDT", limit=250)
    assert [query_of(u)["cursor"] for u in http.urls] == ["1", "2", "3"]
    assert len(points) == 250
    assert points[-1].ts == datetime.fromtimestamp(top / 1000, UTC)
    assert all(b.ts - a.ts == timedelta(hours=8) for a, b in pairwise(points))


def test_funding_stops_at_a_short_page(clock: ManualClock) -> None:
    api, http = api_with(sequence((200, funding_page(hour_ms(900), 40))), clock)
    assert len(api.funding_history("BTCUSDT", limit=500)) == 40
    assert len(http.requests) == 1


def test_funding_survives_a_settlement_between_pages(clock: ManualClock) -> None:
    """A new settlement shifts the pages by one row: the duplicate is dropped and one more page
    is read, so the caller still gets ``limit`` distinct points."""
    top = hour_ms(3000)
    step = 8 * 3_600_000
    api, http = api_with(
        sequence(
            (200, funding_page(top, 100)),
            (200, funding_page(top - 99 * step, 100)),  # shifted: repeats the last row of page 1
            (200, funding_page(top - 199 * step, 100)),
        ),
        clock,
    )
    points = api.funding_history("BTCUSDT", limit=200)
    assert len(http.requests) == 3
    assert len(points) == 200
    assert len({p.ts for p in points}) == 200


def test_funding_row_for_another_symbol_is_refused(clock: ManualClock) -> None:
    api, _ = api_with(sequence((200, funding_page(hour_ms(10), 3, symbol="ETHUSDT"))), clock)
    with pytest.raises(PublicApiError, match="unexpected response shape"):
        api.funding_history("BTCUSDT", limit=3)


# ================================================================================================
# Pacing, latency, recording
# ================================================================================================


def test_sequential_requests_are_paced(clock: ManualClock, fake_time: FakeTime) -> None:
    starts: list[float] = []

    def handler(url: str, headers: dict[str, str]) -> Response:
        starts.append(fake_time.monotonic())
        return synthetic_candle_venue(350)(url, headers)

    api, _ = api_with(handler, clock, min_interval_s=0.06)
    api.candles(
        PriceSource.LIVE,
        "BTCUSDT",
        kind="market",
        interval="1H",
        start=datetime.fromtimestamp(hour_ms(0) / 1000, UTC),
        end=datetime.fromtimestamp(hour_ms(350) / 1000, UTC),
    )
    assert len(starts) == 4
    assert all(b - a >= 0.06 - 1e-9 for a, b in pairwise(starts))
    assert fake_time.sleeps == pytest.approx([0.06, 0.06, 0.06])


def test_no_wait_when_requests_are_already_far_apart(
    clock: ManualClock, fake_time: FakeTime
) -> None:
    def handler(url: str, _h: dict[str, str]) -> Response:
        fake_time.advance(0.4)  # the request itself took longer than the pacing interval
        return 200, funding_page(hour_ms(100), 100 if query_of(url)["cursor"] == "1" else 10)

    api, _ = api_with(handler, clock, min_interval_s=0.06)
    api.funding_history("BTCUSDT", limit=150)
    assert fake_time.sleeps == []
    assert [c.latency_ms for c in api.drain_calls()] == [400, 400]


def test_concurrent_requests_are_paced_across_threads(
    clock: ManualClock, fake_time: FakeTime
) -> None:
    starts: list[float] = []
    lock = threading.Lock()

    def handler(url: str, _h: dict[str, str]) -> Response:
        with lock:
            starts.append(fake_time.monotonic())
        return 200, envelope([ticker_row(query_of(url)["symbol"])])

    symbols = [f"S{i}USDT" for i in range(12)]
    api, _ = api_with(handler, clock, min_interval_s=0.06)
    quotes = api.quotes(PriceSource.LIVE, symbols)
    assert list(quotes) == symbols
    ordered = sorted(starts)
    assert len(ordered) == 12
    assert all(b - a >= 0.06 - 1e-9 for a, b in pairwise(ordered))
    assert [c.params["symbol"] for c in api.drain_calls()] == symbols  # request order, not finish


def test_pacing_zero_never_sleeps(clock: ManualClock, fake_time: FakeTime) -> None:
    api, _ = api_with(synthetic_candle_venue(250), clock, min_interval_s=0)
    api.candles(
        PriceSource.LIVE,
        "BTCUSDT",
        kind="market",
        interval="1H",
        start=datetime.fromtimestamp(hour_ms(0) / 1000, UTC),
        end=datetime.fromtimestamp(hour_ms(250) / 1000, UTC),
    )
    assert fake_time.sleeps == []


def test_drain_calls_hands_each_call_back_once(clock: ManualClock) -> None:
    api, _ = api_with(lambda _u, _h: (200, envelope([ticker_row()])), clock)
    api.quotes(PriceSource.DEMO, ["NVDAUSDT"])
    first = api.drain_calls()
    assert len(first) == 1
    assert api.drain_calls() == ()
    api.quotes(PriceSource.DEMO, ["NVDAUSDT"])
    second = api.drain_calls()
    assert first[0].call_id != second[0].call_id


def test_source_call_records_the_request(clock: ManualClock) -> None:
    blobs = FakeBlobs()
    body = envelope([ticker_row()])
    api, _ = api_with(lambda _u, _h: (200, body), clock, blobs=blobs)
    api.quotes(PriceSource.DEMO, ["NVDAUSDT"])
    (call,) = api.drain_calls()
    assert isinstance(call, SourceCall)
    assert call.surface is ToolkitSurface.PUBLIC_MARKET_API
    assert call.source == "tickers.demo:NVDAUSDT"
    assert call.params == {"category": "USDT-FUTURES", "symbol": "NVDAUSDT"}
    assert call.health is SourceHealth.OK
    assert call.rows == 1
    assert call.started_at == T0
    assert call.error is None
    assert call.blob is not None
    assert call.blob.sha256 == hashlib.sha256(body).hexdigest()
    assert blobs.get(call.blob.sha256) == body
    assert call.blob.media_type == "application/json"
    assert call.call_id.startswith("pub-")


def test_error_bodies_are_kept_as_evidence(clock: ManualClock) -> None:
    blobs = FakeBlobs()
    body = envelope(None, code=NOT_LISTED_CODE, msg="Trading pair X does not exist")
    api, _ = api_with(lambda _u, _h: (400, body), clock, blobs=blobs)
    assert api.instruments(PriceSource.DEMO, ["AAOIUSDT"]) == {}
    (call,) = api.drain_calls()
    assert call.blob is not None
    assert blobs.get(call.blob.sha256) == body


def test_duplicate_symbols_are_requested_once(clock: ManualClock) -> None:
    api, http = api_with(
        lambda url, _h: (200, envelope([ticker_row(query_of(url)["symbol"])])), clock
    )
    quotes = api.quotes(PriceSource.LIVE, ["NVDAUSDT", "BTCUSDT", "NVDAUSDT"])
    assert list(quotes) == ["NVDAUSDT", "BTCUSDT"]
    assert len(http.requests) == 2


def test_no_symbols_means_no_request(clock: ManualClock) -> None:
    api, http = api_with(lambda _u, _h: AssertionError("no request expected"), clock)
    assert api.quotes(PriceSource.LIVE, []) == {}
    assert api.instruments(PriceSource.DEMO, []) == {}
    assert http.requests == []


# ================================================================================================
# Argument validation: refused before any request
# ================================================================================================


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"interval": "1h"}, "interval"),
        ({"kind": "premium"}, "kind"),
        ({"start": datetime(2026, 9, 23, 10, 0)}, "UTC"),  # noqa: DTZ001 - the naive case under test
        ({"start": datetime(2026, 9, 23, 10, 0, tzinfo=timezone(timedelta(hours=8)))}, "UTC"),
        ({"start": T0}, "before"),
        ({"symbol": "nvdausdt"}, "symbol"),
        ({"symbol": "NVDA/USDT"}, "symbol"),
    ],
)
def test_candle_arguments_are_validated(
    kwargs: dict[str, Any], match: str, clock: ManualClock
) -> None:
    api, http = api_with(lambda _u, _h: AssertionError("no request expected"), clock)
    args: dict[str, Any] = {
        "symbol": "BTCUSDT",
        "kind": "market",
        "interval": "1H",
        "start": T0 - timedelta(hours=3),
        "end": T0,
    }
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        api.candles(
            PriceSource.LIVE,
            args["symbol"],
            kind=args["kind"],
            interval=args["interval"],
            start=args["start"],
            end=args["end"],
        )
    assert http.requests == []


@pytest.mark.parametrize("limit", [0, -1, 10_001, True])
def test_funding_limit_is_validated(limit: int, clock: ManualClock) -> None:
    api, http = api_with(lambda _u, _h: AssertionError("no request expected"), clock)
    with pytest.raises(ValueError, match="limit"):
        api.funding_history("BTCUSDT", limit=limit)
    assert http.requests == []


def test_symbols_must_be_a_sequence_of_symbols(clock: ManualClock) -> None:
    api, _ = api_with(lambda _u, _h: AssertionError("no request expected"), clock)
    with pytest.raises(TypeError):
        api.quotes(PriceSource.LIVE, "BTCUSDT")
    with pytest.raises(ValueError, match="symbol"):
        api.instruments(PriceSource.LIVE, ["BTCUSDT", "btc usdt"])


def test_constructor_arguments_are_validated(clock: ManualClock) -> None:
    with pytest.raises(ValueError, match="timeout"):
        BitgetPublicApi(clock=clock, timeout_s=0)
    with pytest.raises(ValueError, match="min_interval"):
        BitgetPublicApi(clock=clock, min_interval_s=-0.1)


# ================================================================================================
# Parsers
# ================================================================================================


def instrument_row(**overrides: str) -> dict[str, str]:
    row = dict(raw_body("instruments_demo", "symbol=NVDAUSDT")["data"][0])
    row.update(overrides)
    return row


def test_parse_instrument_optional_fields_are_none_when_blank() -> None:
    spec = parse_instrument(
        instrument_row(maxMarketOrderQty="", maxOrderQty="", maxLeverage="", fundInterval=""),
        PriceSource.DEMO,
        T0,
    )
    assert spec.max_market_order_qty is None
    assert spec.max_order_qty is None
    assert spec.max_leverage is None
    assert spec.fund_interval_hours is None


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"category": "SPOT"}, "not a valid Category"),
        ({"quantityMultiplier": "0"}, "must be positive"),
        ({"priceMultiplier": "-0.01"}, "must be positive"),
        ({"maxLeverage": "25.5"}, "not an integer"),
        ({"takerFeeRate": "NaN"}, "not finite"),
        ({"minOrderQty": "Infinity"}, "not finite"),
        ({"minOrderAmount": "five"}, "not a decimal"),
        ({"status": ""}, "is empty"),
    ],
)
def test_parse_instrument_refuses_malformed_rows(overrides: dict[str, str], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        parse_instrument(instrument_row(**overrides), PriceSource.DEMO, T0)


def test_parse_instrument_missing_required_field() -> None:
    row = instrument_row()
    del row["takerFeeRate"]
    with pytest.raises(ValueError, match="takerFeeRate"):
        parse_instrument(row, PriceSource.LIVE, T0)


def test_parse_ticker_keeps_every_digit_and_reads_ms_timestamps() -> None:
    quote = parse_ticker(ticker_row(), PriceSource.LIVE, T0)
    assert quote.index == Decimal("223.0600829147522046")
    assert quote.ts == datetime(2026, 9, 24, 11, 7, 11, 182000, tzinfo=UTC)
    assert quote.price_change_24h == Decimal("-0.02229")
    assert quote.funding_rate == Decimal("0")  # a measured zero stays zero
    assert quote.fetched_at == T0


def test_parse_ticker_optional_fields_blank_are_none() -> None:
    quote = parse_ticker(
        ticker_row(fundingRate="", openInterest="", turnover24h="", price24hPcnt=""),
        PriceSource.DEMO,
        T0,
    )
    assert quote.funding_rate is None
    assert quote.open_interest is None
    assert quote.turnover_24h is None
    assert quote.price_change_24h is None


def test_parse_ticker_refuses_a_naive_fetch_time() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        parse_ticker(ticker_row(), PriceSource.LIVE, datetime(2026, 9, 24))  # noqa: DTZ001


def test_parse_ticker_accepts_json_numbers_exactly() -> None:
    """If the venue ever sends numbers instead of strings, parse_float=Decimal keeps them exact."""
    body = b'{"symbol":"NVDAUSDT","ts":1790248031182,"lastPrice":223.27,"markPrice":223.21,' \
        b'"indexPrice":223.0600829147522046,"bid1Price":223.28,"ask1Price":223.29}'  # fmt: skip
    row = public_api._decode_envelope(body)
    assert row is not None
    quote = parse_ticker(row, PriceSource.LIVE, T0)
    assert quote.index == Decimal("223.0600829147522046")
    assert quote.funding_rate is None


@pytest.mark.parametrize(
    ("bad_ts", "match"),
    [
        ("", "ts is empty"),
        ("-5", "not a millisecond timestamp"),
        ("12.5", "not a millisecond timestamp"),
        ("abc", "not a decimal"),
    ],
)
def test_parse_ticker_refuses_bad_timestamps(bad_ts: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        parse_ticker(ticker_row(ts=bad_ts), PriceSource.LIVE, T0)


def test_parse_candle_reads_volume_only_for_market_candles() -> None:
    row = ["1790236800000", "224.5", "224.6", "222.5", "222.7", "813.99", "52245798.2"]
    market = parse_candle(
        row, symbol="BTCUSDT", source=PriceSource.LIVE, kind="market", interval="1H"
    )
    assert market.volume == Decimal("813.99")
    assert market.open_time == datetime(2026, 9, 24, 8, tzinfo=UTC)
    for kind in ("mark", "index"):
        placeholder = [*row[:5], "0", "0"]
        candle = parse_candle(
            placeholder, symbol="BTCUSDT", source=PriceSource.DEMO, kind=kind, interval="1H"
        )
        assert candle.volume is None
        assert candle.kind == kind


def test_parse_candle_with_five_columns_has_no_volume() -> None:
    candle = parse_candle(
        ["1790236800000", "1", "2", "0.5", "1.5"],
        symbol="BTCUSDT",
        source=PriceSource.LIVE,
        kind="market",
        interval="4H",
    )
    assert candle.volume is None
    assert candle.close == Decimal("1.5")


@pytest.mark.parametrize(
    ("row", "match"),
    [
        (["1790236800000", "1", "2", "0.5"], "at least 5 columns"),
        (["1790236800000", "1", "2", "0.5", "NaN"], "not finite"),
        (["1790236800000", "1", "", "0.5", "1"], "high is empty"),
        (["not-a-time", "1", "2", "0.5", "1"], "not a decimal"),
        ("1790236800000,1,2,0.5,1", "at least 5 columns"),
    ],
)
def test_parse_candle_refuses_malformed_rows(row: Any, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        parse_candle(row, symbol="BTCUSDT", source=PriceSource.LIVE, kind="market", interval="1H")


def test_parse_candle_refuses_unknown_kind_and_interval() -> None:
    row = ["1790236800000", "1", "2", "0.5", "1"]
    with pytest.raises(ValueError, match="kind"):
        parse_candle(row, symbol="X", source=PriceSource.LIVE, kind="premium", interval="1H")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="interval"):
        parse_candle(row, symbol="X", source=PriceSource.LIVE, kind="market", interval="2H")


# ================================================================================================
# urllib_get and the protocol
# ================================================================================================


class _FakeResponse:
    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self._body = io.BytesIO(body)

    def read(self, n: int = -1) -> bytes:
        return self._body.read(n)

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class _FakeOpener:
    def __init__(self, answer: Callable[[Any], _FakeResponse]) -> None:
        self.answer = answer
        self.seen: list[tuple[Any, float]] = []

    def open(self, request: Any, timeout: float) -> _FakeResponse:
        self.seen.append((request, timeout))
        return self.answer(request)


def test_urllib_get_sends_exactly_the_given_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    opener = _FakeOpener(lambda _r: _FakeResponse(200, b'{"code":"00000"}'))
    monkeypatch.setattr(public_api, "_OPENER", opener)
    headers = public_api._request_headers(PriceSource.DEMO, PATH_TICKERS)
    url = f"{BASE_URL}{PATH_TICKERS}?category=USDT-FUTURES&symbol=NVDAUSDT"
    assert urllib_get(url, headers, 7.5) == (200, b'{"code":"00000"}')
    ((request, timeout),) = opener.seen
    assert timeout == 7.5
    assert request.get_method() == "GET"
    assert request.full_url == url
    sent = {k.lower(): v for k, v in request.header_items()}
    assert sent == {k.lower(): v for k, v in headers.items()}
    assert request.data is None


def test_urllib_get_returns_error_statuses_with_their_body(monkeypatch: pytest.MonkeyPatch) -> None:
    body = b'{"code":"25100","msg":"Trading pair X does not exist"}'

    def answer(request: Any) -> _FakeResponse:
        raise urllib.error.HTTPError(
            request.full_url, 400, "Bad Request", Message(), io.BytesIO(body)
        )

    monkeypatch.setattr(public_api, "_OPENER", _FakeOpener(answer))
    assert urllib_get(f"{BASE_URL}{PATH_TICKERS}", {}, 1.0) == (400, body)


def test_urllib_get_refuses_non_https() -> None:
    for url in ("http://api.bitget.com/api/v3/market/tickers", "file:///etc/passwd"):
        with pytest.raises(ValueError, match="HTTPS"):
            urllib_get(url, {}, 1.0)


def test_urllib_get_caps_the_body(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(public_api, "MAX_BODY_BYTES", 10)
    monkeypatch.setattr(
        public_api, "_OPENER", _FakeOpener(lambda _r: _FakeResponse(200, b"x" * 11))
    )
    with pytest.raises(ResponseTooLargeError):
        urllib_get(f"{BASE_URL}{PATH_TICKERS}", {}, 1.0)


def test_a_too_large_body_is_a_recorded_error(clock: ManualClock) -> None:
    api, http = api_with(lambda _u, _h: ResponseTooLargeError("too big"), clock)
    with pytest.raises(PublicApiError):
        api.funding_history("BTCUSDT", limit=1)
    (call,) = api.drain_calls()
    assert call.health is SourceHealth.ERROR
    assert len(http.requests) == 3  # an OSError that is not a timeout is retried


@pytest.fixture
def redirecting_server() -> Iterator[str]:
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:9/elsewhere")
            self.end_headers()

        def log_message(self, format: str, *args: Any) -> None:
            return None

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/redirect"
    finally:
        server.shutdown()
        server.server_close()


def test_the_opener_refuses_redirects(redirecting_server: str) -> None:
    with pytest.raises(urllib.error.HTTPError) as caught:
        public_api._OPENER.open(redirecting_server, timeout=5)
    assert caught.value.code == 302


def test_client_satisfies_the_market_data_protocol(clock: ManualClock) -> None:
    market: MarketData = BitgetPublicApi(clock=clock)
    for name in ("instruments", "quotes", "candles", "funding_history"):
        ours = inspect.signature(getattr(BitgetPublicApi, name))
        theirs = inspect.signature(getattr(MarketData, name))
        assert [(p.name, p.kind) for p in ours.parameters.values()] == [
            (p.name, p.kind) for p in theirs.parameters.values()
        ], name
    assert callable(market.quotes)


def test_constructor_signature_is_the_specified_one() -> None:
    params = inspect.signature(BitgetPublicApi.__init__).parameters
    assert [(n, p.kind, p.default) for n, p in params.items() if n != "self"] == [
        ("clock", inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.empty),
        ("http", inspect.Parameter.KEYWORD_ONLY, urllib_get),
        ("blobs", inspect.Parameter.KEYWORD_ONLY, None),
        ("timeout_s", inspect.Parameter.KEYWORD_ONLY, 15.0),
        ("min_interval_s", inspect.Parameter.KEYWORD_ONLY, 0.06),
    ]


# ================================================================================================
# Live schema drift (keyless public endpoints only; run with --run-live-public)
# ================================================================================================


def live_raw(path: str, query: dict[str, str], source: PriceSource) -> tuple[int, dict[str, Any]]:
    from urllib.parse import urlencode

    status, body = urllib_get(
        f"{BASE_URL}{path}?{urlencode(query)}", public_api._request_headers(source, path), 15.0
    )
    decoded = json.loads(body)
    assert isinstance(decoded, dict)
    return status, decoded


@pytest.fixture
def real_time(monkeypatch: pytest.MonkeyPatch) -> None:
    import time

    monkeypatch.setattr(public_api, "_monotonic", time.monotonic)
    monkeypatch.setattr(public_api, "_sleep", time.sleep)


@pytest.mark.live_public
@pytest.mark.usefixtures("real_time")
@pytest.mark.parametrize("source", [PriceSource.LIVE, PriceSource.DEMO])
def test_live_schema_still_carries_every_parsed_field(source: PriceSource) -> None:
    base = {"category": "USDT-FUTURES", "symbol": "NVDAUSDT"}
    _, tickers = live_raw(PATH_TICKERS, base, source)
    assert tickers["code"] == "00000"
    assert set(tickers["data"][0]) >= TICKER_KEYS
    _, instruments = live_raw(PATH_INSTRUMENTS, base, source)
    assert set(instruments["data"][0]) >= INSTRUMENT_KEYS
    for kind in ("market", "mark", "index"):
        _, candles = live_raw(
            PATH_HISTORY_CANDLES, {**base, "interval": "1H", "type": kind, "limit": "2"}, source
        )
        assert candles["code"] == "00000"
        assert all(len(row) >= 6 for row in candles["data"])
    _, funding = live_raw(
        PATH_HISTORY_FUND_RATE, {**base, "symbol": "BTCUSDT", "limit": "2", "cursor": "1"}, source
    )
    assert set(funding["data"]["resultList"][0]) >= FUNDING_KEYS


@pytest.mark.live_public
@pytest.mark.usefixtures("real_time")
def test_live_client_end_to_end() -> None:
    api = BitgetPublicApi(clock=SystemClock())
    symbols = ["BTCUSDT", "NVDAUSDT"]
    for source in (PriceSource.LIVE, PriceSource.DEMO):
        assert set(api.instruments(source, symbols)) == set(symbols)
        quotes = api.quotes(source, symbols)
        assert set(quotes) == set(symbols)
        assert all(q.source is source and q.bid <= q.ask for q in quotes.values())
    end = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    candles = api.candles(
        PriceSource.LIVE,
        "BTCUSDT",
        kind="market",
        interval="1H",
        start=end - timedelta(hours=150),
        end=end,
    )
    assert len(candles) == 150
    assert all(b.open_time - a.open_time == timedelta(hours=1) for a, b in pairwise(candles))
    funding = api.funding_history("BTCUSDT", limit=150)
    assert len(funding) == 150
    assert all(c.health in (SourceHealth.OK, SourceHealth.EMPTY) for c in api.drain_calls())


@pytest.mark.live_public
@pytest.mark.usefixtures("real_time")
def test_live_end_time_still_filters_on_bar_close() -> None:
    base = {"category": "USDT-FUTURES", "symbol": "BTCUSDT", "interval": "1H", "type": "market"}
    _, first = live_raw(PATH_HISTORY_CANDLES, {**base, "limit": "3"}, PriceSource.LIVE)
    oldest = min(int(r[0]) for r in first["data"])
    _, page = live_raw(
        PATH_HISTORY_CANDLES, {**base, "limit": "3", "endTime": str(oldest)}, PriceSource.LIVE
    )
    assert max(int(r[0]) for r in page["data"]) == oldest - 3_600_000


@pytest.mark.live_public
@pytest.mark.usefixtures("real_time")
def test_live_demo_header_routes_to_demo() -> None:
    """AAOIUSDT is listed live and not on UTA Demo: the header visibly changes the venue."""
    query = {"category": "USDT-FUTURES", "symbol": "AAOIUSDT"}
    _, live = live_raw(PATH_TICKERS, query, PriceSource.LIVE)
    status, demo = live_raw(PATH_TICKERS, query, PriceSource.DEMO)
    assert live["code"] == "00000"
    assert (status, demo["code"]) == (400, NOT_LISTED_CODE)


@pytest.mark.live_public
@pytest.mark.usefixtures("real_time")
def test_live_limits_and_unknown_pairs_answer_as_recorded() -> None:
    base = {"category": "USDT-FUTURES", "symbol": "BTCUSDT"}
    status, over = live_raw(
        PATH_HISTORY_FUND_RATE, {**base, "limit": "101", "cursor": "1"}, PriceSource.LIVE
    )
    assert (status, over["code"]) == (400, "40020")
    status, unknown = live_raw(PATH_TICKERS, {**base, "symbol": "NOSUCHUSDT"}, PriceSource.LIVE)
    assert (status, unknown["code"]) == (400, NOT_LISTED_CODE)
