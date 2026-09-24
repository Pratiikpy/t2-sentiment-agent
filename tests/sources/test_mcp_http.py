"""The MCP transport: SSE framing, the handshake, session and id handling, failures, and the payload
helpers both readers share. Replays real recorded sessions (``tests/fixtures/sources``)."""

import http.server
import json
import math
import threading
import time
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from sentiment_agent.clock import ManualClock, SystemClock
from sentiment_agent.hashing import sha256_hex
from sentiment_agent.sources.mcp_http import (
    DATA_MCP_URL,
    PROTOCOL_VERSION,
    SIGNAL_MCP_URL,
    McpError,
    McpTimeoutError,
    StreamableHttpMcp,
    error_envelope,
    finite,
    freshness_problem,
    hollow,
    invoke,
    newest,
    parse_instant,
    parse_sse_or_json,
    row_time,
    rows_of,
    source_call,
    urllib_post,
)
from sentiment_agent.types import BlobRef, McpToolResult, SourceHealth, ToolkitSurface
from sources.replay import RecordedHttp, call_key, load_session, sse_reply

SIGNAL = load_session("signal_session.json")
DATA = load_session("data_session.json")

# ================================================================================================
# Framing
# ================================================================================================


def test_recorded_sse_reply_parses() -> None:
    body = DATA.exchanges["initialize"].body
    assert body.startswith("event: message\r\ndata: ")  # CRLF framing, as recorded
    message = parse_sse_or_json(body.encode(), request_id=1)
    assert message["result"]["serverInfo"]["name"] == "bitget-mcp-server"


def test_ping_comments_before_the_reply_are_skipped() -> None:
    key = call_key("sentiment_index", {"action": "current"})
    body = SIGNAL.exchanges[key].body
    assert body.startswith(": ping")  # recorded live: keep-alive comments precede slow replies
    recorded_id = SIGNAL.exchanges[key].request["id"]
    message = parse_sse_or_json(body.encode(), request_id=recorded_id)
    assert json.loads(message["result"]["content"][0]["text"]) == {"alt_me_error": ""}


def test_plain_json_body_parses() -> None:
    raw = json.dumps({"jsonrpc": "2.0", "id": 7, "result": {"ok": True}}).encode()
    assert parse_sse_or_json(raw, request_id=7)["result"] == {"ok": True}
    assert parse_sse_or_json(raw)["result"] == {"ok": True}


def test_multi_line_data_is_joined_and_bom_and_cr_endings_are_accepted() -> None:
    bom = b"\xef\xbb\xbf"
    raw = bom + b'event: message\rdata: {"jsonrpc":"2.0",\rdata: "id":3,"result":{}}\r\r'
    assert parse_sse_or_json(raw, request_id=3)["id"] == 3


def test_frame_is_selected_by_id_and_foreign_frames_are_refused() -> None:
    frames = (
        b'data: {"jsonrpc":"2.0","method":"notifications/progress","params":{}}\n\n'
        b'event: ping\ndata: {"jsonrpc":"2.0","id":2,"result":{"from":"ping"}}\n\n'
        b'data: {"jsonrpc":"2.0","id":1,"result":{"which":"other"}}\n\n'
        b'data: {"jsonrpc":"2.0","id":2,"result":{"which":"mine"}}\n\n'
    )
    assert parse_sse_or_json(frames, request_id=2)["result"] == {"which": "mine"}
    with pytest.raises(McpError, match="no JSON-RPC response with id 9"):
        parse_sse_or_json(frames, request_id=9)
    with pytest.raises(McpError, match="2 responses in one reply"):
        parse_sse_or_json(frames)


def test_duplicate_ids_and_missing_responses_are_refused() -> None:
    frame = b'data: {"jsonrpc":"2.0","id":1,"result":{}}\n\n'
    twice = frame + frame
    with pytest.raises(McpError, match="refusing to choose"):
        parse_sse_or_json(twice, request_id=1)
    with pytest.raises(McpError, match="no JSON-RPC response"):
        parse_sse_or_json(b": ping\n\n")
    with pytest.raises(McpError, match="undecodable frames: 1"):
        parse_sse_or_json(b'data: {"jsonrpc":"2.0","id":1,"res', request_id=1)
    with pytest.raises(McpError, match="neither SSE nor valid JSON"):
        parse_sse_or_json(b'{"jsonrpc":')


# ================================================================================================
# Client against the recorded sessions
# ================================================================================================


class MemoryBlobs:
    def __init__(self) -> None:
        self.items: dict[str, tuple[bytes, str]] = {}

    def put(self, data: bytes, media_type: str) -> BlobRef:
        digest = sha256_hex(data)
        self.items[digest] = (data, media_type)
        return BlobRef(sha256=digest, media_type=media_type, size=len(data))

    def get(self, sha256: str) -> bytes:
        return self.items[sha256][0]


def _client(session: Any, http: RecordedHttp, **kwargs: Any) -> StreamableHttpMcp:
    return StreamableHttpMcp(
        session.url, server_label=session.server, clock=session.clock(), http=http, **kwargs
    )


def test_handshake_order_session_propagation_and_headers() -> None:
    http = RecordedHttp(DATA)
    mcp = _client(DATA, http)
    mcp.call_tool("do_query", {"entry_id": "sentiment_market_fear_greed", "params": {}})
    methods = [message.get("method") for _, message, _ in http.requests]
    assert methods == ["initialize", "notifications/initialized", "tools/call"]
    recorded_session = DATA.exchanges["initialize"].headers["mcp-session-id"]
    assert mcp.session_id == recorded_session
    first = http.requests[0][2]
    assert "Mcp-Session-Id" not in first  # nothing to send before the server assigns one
    for _, _, headers in http.requests[1:]:
        assert headers["Mcp-Session-Id"] == recorded_session
        assert headers["MCP-Protocol-Version"] == PROTOCOL_VERSION
    for _, _, headers in http.requests:
        assert headers["User-Agent"] == "curl/8.0"
        assert headers["Accept"] == "application/json, text/event-stream"
    assert mcp.server_info["name"] == "bitget-mcp-server"
    assert mcp.protocol_version == PROTOCOL_VERSION


def test_ids_are_unique_and_every_reply_matches_its_request() -> None:
    http = RecordedHttp(DATA)
    mcp = _client(DATA, http)
    for _ in range(3):
        mcp.call_tool("guide", {})
    ids = [message["id"] for _, message, _ in http.requests if "id" in message]
    assert ids == sorted(set(ids))


def test_reply_for_another_request_is_refused() -> None:
    def wrong_id(message: dict[str, Any]) -> tuple[int, dict[str, str], bytes]:
        frame = {"jsonrpc": "2.0", "id": message["id"] + 1000, "result": {"content": []}}
        return 200, {"content-type": "text/event-stream"}, f"data: {json.dumps(frame)}\n\n".encode()

    http = RecordedHttp(DATA, default=wrong_id)
    mcp = _client(DATA, http)
    with pytest.raises(McpError, match="refusing to return another request's response"):
        mcp.call_tool("not_recorded", {})


def test_tools_list_pins_the_tools_the_readers_call() -> None:
    signal_tools = {t["name"]: t for t in _client(SIGNAL, RecordedHttp(SIGNAL)).list_tools()}
    assert len(signal_tools) == 19
    schema = signal_tools["derivatives_sentiment"]["inputSchema"]["properties"]
    assert {"long_short", "top_ls", "taker_ratio", "open_interest", "reddit_trending"} <= set(
        schema["action"]["enum"]
    )
    assert "filter" in schema
    assert (
        "current" in signal_tools["sentiment_index"]["inputSchema"]["properties"]["action"]["enum"]
    )
    assert "latest" in signal_tools["news_feed"]["inputSchema"]["properties"]["action"]["enum"]

    data_tools = {t["name"]: t for t in _client(DATA, RecordedHttp(DATA)).list_tools()}
    assert set(data_tools) == {"guide", "do_query"}
    assert data_tools["do_query"]["inputSchema"]["required"] == ["entry_id"]


def test_tools_list_follows_pagination() -> None:
    sse = {"content-type": "text/event-stream"}
    http = RecordedHttp(DATA)
    http.overrides["tools/list"] = (
        200,
        sse,
        sse_reply({"tools": [{"name": "a"}], "nextCursor": "p2"}),
    )
    http.overrides["tools/list p2"] = (200, sse, sse_reply({"tools": [{"name": "b"}]}))
    assert [t["name"] for t in _client(DATA, http).list_tools()] == ["a", "b"]


def test_call_tool_returns_structured_text_and_blob() -> None:
    blobs = MemoryBlobs()
    mcp = _client(DATA, RecordedHttp(DATA), blobs=blobs)
    result = mcp.call_tool("do_query", {"entry_id": "sentiment_market_fear_greed", "params": {}})
    assert result.server == "bitget-mcp-server"
    assert result.tool == "do_query"
    assert not result.is_error
    assert result.structured is not None
    assert result.structured["success"] is True
    assert json.loads(result.text) == result.structured
    assert result.raw is not None
    stored, media = blobs.items[result.raw.sha256]
    assert media == "text/event-stream"
    assert stored.startswith(b"event: message")

    signal = _client(SIGNAL, RecordedHttp(SIGNAL))
    hollow_result = signal.call_tool("sentiment_index", {"action": "current"})
    assert hollow_result.structured is None
    assert json.loads(hollow_result.text) == {"alt_me_error": ""}


def test_tool_error_is_returned_not_raised() -> None:
    result = _client(SIGNAL, RecordedHttp(SIGNAL)).call_tool("no_such_tool", {})
    assert result.is_error
    assert "Unknown tool" in result.text
    refused = _client(DATA, RecordedHttp(DATA)).call_tool(
        "do_query", {"id": "crypto_sentiment_crypto_fear_greed"}
    )
    assert refused.is_error
    assert "entry_id" in refused.text
    assert "Missing required argument" in refused.text


def test_http_403_is_named_and_carries_the_status() -> None:
    def forbidden(message: dict[str, Any]) -> tuple[int, dict[str, str], bytes]:
        return 403, {"content-type": "application/json"}, b'{"error_code":1010}'

    http = RecordedHttp(DATA, default=forbidden)
    http.overrides["initialize"] = (403, {}, b'{"error_code":1010}')
    mcp = _client(DATA, http)
    with pytest.raises(McpError, match="HTTP 403") as caught:
        mcp.call_tool("guide", {})
    assert caught.value.code == 403


def test_expired_session_is_renewed_once() -> None:
    state = {"expired": True}

    def expire_once(message: dict[str, Any]) -> tuple[int, dict[str, str], bytes]:
        if state["expired"]:
            state["expired"] = False
            return 404, {}, b"session not found"
        return (
            200,
            {"content-type": "text/event-stream"},
            sse_reply({"content": [], "isError": False}),
        )

    http = RecordedHttp(DATA, default=expire_once)
    mcp = _client(DATA, http)
    result = mcp.call_tool("unrecorded_tool", {})
    assert not result.is_error
    assert mcp.sessions_started == 2
    methods = [m.get("method") for _, m, _ in http.requests]
    assert methods.count("initialize") == 2


def test_jsonrpc_error_member_raises_with_its_code() -> None:
    def rpc_error(message: dict[str, Any]) -> tuple[int, dict[str, str], bytes]:
        frame = {"jsonrpc": "2.0", "id": message["id"], "error": {"code": -32602, "message": "bad"}}
        return 200, {"content-type": "application/json"}, json.dumps(frame).encode()

    mcp = _client(DATA, RecordedHttp(DATA, default=rpc_error))
    with pytest.raises(McpError, match="JSON-RPC error -32602") as caught:
        mcp.call_tool("unrecorded_tool", {})
    assert caught.value.code == -32602


def test_transport_timeout_and_failure_are_distinct() -> None:
    def times_out(url: str, body: bytes, headers: Mapping[str, str], timeout: float) -> Any:
        raise TimeoutError("slow")

    def breaks(url: str, body: bytes, headers: Mapping[str, str], timeout: float) -> Any:
        raise OSError("connection reset")

    clock = ManualClock(datetime(2026, 9, 24, tzinfo=UTC))
    slow = StreamableHttpMcp(DATA_MCP_URL, server_label="x", clock=clock, http=times_out)
    with pytest.raises(McpTimeoutError):
        slow.call_tool("guide", {})
    broken = StreamableHttpMcp(DATA_MCP_URL, server_label="x", clock=clock, http=breaks)
    with pytest.raises(McpError, match="transport failure") as caught:
        broken.call_tool("guide", {})
    assert not isinstance(caught.value, McpTimeoutError)


def test_constructor_refusals() -> None:
    clock = ManualClock(datetime(2026, 9, 24, tzinfo=UTC))
    with pytest.raises(ValueError, match="403"):
        StreamableHttpMcp(DATA_MCP_URL, server_label="x", clock=clock, user_agent=" ")
    with pytest.raises(ValueError, match="only https"):
        StreamableHttpMcp("http://agent.bitget.com/mcp", server_label="x", clock=clock)
    with pytest.raises(ValueError, match="only https"):
        StreamableHttpMcp("file:///etc/passwd", server_label="x", clock=clock)
    with pytest.raises(ValueError, match="server_label"):
        StreamableHttpMcp(DATA_MCP_URL, server_label="", clock=clock)


def test_concurrent_calls_share_one_session_and_never_reuse_an_id() -> None:
    http = RecordedHttp(DATA)
    mcp = _client(DATA, http)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: mcp.call_tool("guide", {}), range(24)))
    assert all(not r.is_error for r in results)
    assert mcp.sessions_started == 1
    ids = [message["id"] for _, message, _ in http.requests if "id" in message]
    assert len(ids) == len(set(ids)) == 25  # one initialize, 24 calls


def test_requests_in_flight_are_capped_per_server() -> None:
    state = {"now": 0, "peak": 0}
    lock = threading.Lock()
    session = RecordedHttp(DATA)

    def slow(url: str, body: bytes, headers: Mapping[str, str], timeout: float) -> Any:
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(0.02)
        try:
            return session(url, body, headers, timeout)
        finally:
            with lock:
                state["now"] -= 1

    mcp = StreamableHttpMcp(
        DATA.url, server_label="d", clock=DATA.clock(), http=slow, max_in_flight=3
    )
    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(lambda _: mcp.call_tool("guide", {}), range(24)))
    assert state["peak"] == 3
    with pytest.raises(ValueError, match="max_in_flight"):
        StreamableHttpMcp(DATA.url, server_label="d", clock=DATA.clock(), max_in_flight=0)


# ================================================================================================
# The real standard-library transport, against a local server
# ================================================================================================


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        if self.path == "/forbidden":
            self.send_response(403)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error_code":1010}')
            return
        if self.path == "/trickle":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for _ in range(20):
                self.wfile.write(b": ping\n\n")
                self.wfile.flush()
                time.sleep(0.1)
            return
        frame = {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {"ua": self.headers["User-Agent"]},
        }
        body = f"event: message\r\ndata: {json.dumps(frame)}\r\n\r\n".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Mcp-Session-Id", "local-session")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def local_server() -> Iterator[str]:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_urllib_post_reads_sse_and_headers(local_server: str) -> None:
    body = json.dumps({"jsonrpc": "2.0", "id": 5, "method": "x"}).encode()
    status, headers, raw = urllib_post(
        local_server + "/mcp",
        body,
        {"User-Agent": "curl/8.0", "Content-Type": "application/json"},
        5,
    )
    assert status == 200
    assert headers["mcp-session-id"] == "local-session"  # lower-cased keys
    assert parse_sse_or_json(raw, request_id=5)["result"] == {"ua": "curl/8.0"}


def test_urllib_post_returns_error_statuses(local_server: str) -> None:
    status, _, raw = urllib_post(local_server + "/forbidden", b"{}", {}, 5)
    assert status == 403
    assert b"1010" in raw


def test_urllib_post_enforces_a_wall_clock_deadline(local_server: str) -> None:
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="did not finish"):
        urllib_post(local_server + "/trickle", b"{}", {}, 0.5)
    assert time.monotonic() - started < 1.9  # keep-alives must not extend the deadline


def test_client_end_to_end_over_real_http(local_server: str) -> None:
    mcp = StreamableHttpMcp(local_server + "/mcp", server_label="local", clock=SystemClock())
    result = mcp.call_tool("anything", {})
    assert mcp.session_id == "local-session"
    assert result.text == ""  # the local server's result has no content


def test_urllib_post_refuses_other_schemes() -> None:
    with pytest.raises(ValueError, match="only https"):
        urllib_post("ftp://example.com/x", b"", {}, 1)


# ================================================================================================
# invoke / source_call
# ================================================================================================


class _Caller:
    def __init__(self, outcome: Any) -> None:
        self._outcome = outcome

    @property
    def server(self) -> str:
        return "fake"

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> McpToolResult:
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        result: McpToolResult = self._outcome
        return result


@pytest.mark.parametrize(
    ("raised", "health"),
    [
        (McpTimeoutError("slow"), SourceHealth.TIMEOUT),
        (McpError("refused"), SourceHealth.ERROR),
        (KeyError("defect"), SourceHealth.ERROR),
    ],
)
def test_invoke_never_raises(raised: Exception, health: SourceHealth, clock: ManualClock) -> None:
    invocation = invoke(_Caller(raised), clock, "tool", {})
    assert invocation.result is None
    assert invocation.failure is health
    assert invocation.error
    assert invocation.started_at == clock.now()
    assert invocation.latency_ms >= 0


def test_source_call_records_params_as_strings_and_the_blob(clock: ManualClock) -> None:
    blob = BlobRef(sha256="a" * 64, media_type="text/event-stream", size=3)
    result = McpToolResult(
        server="s", tool="t", is_error=False, structured=None, text="{}", raw=blob
    )
    invocation = invoke(_Caller(result), clock, "t", {})
    call = source_call(
        surface=ToolkitSurface.SIGNAL_MCP,
        source="t.a",
        params={"limit": 10, "flag": True, "symbol": "BTCUSDT"},
        invocation=invocation,
        health=SourceHealth.OK,
        rows=4,
    )
    assert call.params == {"limit": "10", "flag": "true", "symbol": "BTCUSDT"}
    assert call.blob == blob
    assert call.rows == 4
    assert call.error is None
    assert call.call_id.startswith("bitget_signal_mcp-")


# ================================================================================================
# Payload helpers
# ================================================================================================


def test_hollow_on_every_recorded_signal_reply() -> None:
    for key in SIGNAL.exchanges:
        if not key.startswith("tools/call") or '"action":"not_an_action"' in key:
            continue
        if "no_such_tool" in key:
            continue
        payload = SIGNAL.payload(key)
        assert hollow(payload), key


def test_hollow_recognises_substance() -> None:
    assert hollow({"alt_me_error": ""})
    assert hollow([{"feed": "coindesk", "error": "", "items": []}])
    assert hollow({"t3m": {"error": ""}, "spread_10y2y": 0.0, "inverted": False})  # ARGUS case
    assert hollow({"symbol": "BTCUSDT", "period": "4h", "value": float("nan")})
    assert hollow({"note": "no data", "source": "x", "value": "  "})
    assert not hollow({"symbol": "BTCUSDT", "longShortRatio": "1.2"})
    assert not hollow({"value": 0, "flag": True})
    assert not hollow([{"error": ""}, {"items": [{"title": "x"}]}])


def test_hollow_treats_backend_naming_keys_as_request_description() -> None:
    # recorded by the live probe of 2026-09-24 from social_trending
    assert hollow({"platform": "xueqiu", "provider": "all_failed", "items": []})
    assert hollow({"symbol": "BTCUSDT", "exchange": "binance", "interval": "1h", "rows": []})
    assert not hollow({"platform": "weibo", "items": [{"title": "a story", "hot": 12}]})


def test_error_envelope_distinguishes_refusal_from_silence() -> None:
    assert error_envelope({"error": ""}) == (
        SourceHealth.HOLLOW,
        'upstream error envelope {"error": ""}',
    )
    refused = error_envelope({"error": "Unknown action: not_an_action"})
    assert refused is not None
    assert refused[0] is SourceHealth.ERROR
    assert error_envelope({"error": "", "value": 1}) is None
    assert error_envelope([]) is None


def test_finite_and_instants() -> None:
    assert finite("1.25") == 1.25
    assert finite(True) is None
    assert finite("nan") is None
    assert finite(math.inf) is None
    assert parse_instant(1790240400) == datetime(2026, 9, 24, 9, tzinfo=UTC)
    assert parse_instant(1790240400000) == datetime(2026, 9, 24, 9, tzinfo=UTC)
    assert parse_instant("1790208000") == datetime(2026, 9, 24, tzinfo=UTC)
    assert parse_instant("2026-09-24T11:00:00Z") == datetime(2026, 9, 24, 11, tzinfo=UTC)
    assert parse_instant("2026-09-24T11:00:00") is None  # naive: zone unknown
    assert parse_instant("2026-09-24") is None
    assert parse_instant(5) is None  # 1970: out of range


def test_row_time_prefers_epoch_ms_time() -> None:
    # bitget-mcp-server ratio rows carry seconds in `timestamp` and milliseconds in `time`
    row = {"timestamp": 1790240400, "date": "2026-09-24T09:00:00Z", "time": 1790240400000}
    assert row_time(row) == datetime(2026, 9, 24, 9, tzinfo=UTC)
    assert row_time({"date": "2026-09-24T09:00:00Z"}) == datetime(2026, 9, 24, 9, tzinfo=UTC)
    assert row_time({"value": 1}) is None


def test_newest_does_not_trust_row_order() -> None:
    rows = [{"time": 3_000_000_000_000, "v": "late"}, {"time": 1_800_000_000_000, "v": "early"}]
    found = newest(rows)
    assert found is not None
    assert found[0]["v"] == "late"
    assert newest([{"v": 1}]) is None


def test_rows_of_unwraps_common_envelopes() -> None:
    assert rows_of([{"a": 1}, 2]) == [{"a": 1}]
    assert rows_of({"data": [{"a": 1}]}) == [{"a": 1}]
    assert rows_of({"success": True, "data": {"results": [{"a": 1}]}}) == [{"a": 1}]
    assert rows_of({"a": 1}) == [{"a": 1}]
    assert rows_of("text") == []


def test_freshness_problem() -> None:
    now = datetime(2026, 9, 24, 12, tzinfo=UTC)
    assert freshness_problem(now - timedelta(hours=1), now, timedelta(hours=3)) is None
    stale = freshness_problem(now - timedelta(hours=5), now, timedelta(hours=3))
    assert stale is not None
    assert "stale" in stale
    ahead = freshness_problem(now + timedelta(hours=2), now, timedelta(hours=3))
    assert ahead is not None
    assert "ahead of now" in ahead
    missing = freshness_problem(None, now, timedelta(hours=3))
    assert missing is not None
    assert "no parseable timestamp" in missing


# ================================================================================================
# Live drift (keyless; run with --run-live-public)
# ================================================================================================


@pytest.mark.live_public
@pytest.mark.parametrize("url", [DATA_MCP_URL, SIGNAL_MCP_URL])
def test_live_user_agent_rule_still_holds(url: str) -> None:
    """Both servers 403 Python's default User-Agent. If this starts failing, the quirk is gone and
    the explicit User-Agent is merely harmless; update the module docstring."""
    body = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "t", "version": "0"},
            },
        }
    ).encode()
    base = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    status_default, _, _ = urllib_post(url, body, {**base, "User-Agent": "Python-urllib/3.11"}, 30)
    status_curl, headers, raw = urllib_post(url, body, {**base, "User-Agent": "curl/8.0"}, 30)
    assert status_default == 403
    assert status_curl == 200
    assert headers.get("mcp-session-id")
    assert parse_sse_or_json(raw, request_id=1)["result"]["protocolVersion"]


@pytest.mark.live_public
def test_live_tool_surfaces_match_the_recording() -> None:
    clock = SystemClock()
    signal = StreamableHttpMcp(SIGNAL_MCP_URL, server_label="bitget-signal", clock=clock)
    data = StreamableHttpMcp(DATA_MCP_URL, server_label="bitget-mcp-server", clock=clock)
    live_signal = {t["name"] for t in signal.list_tools()}
    recorded_signal = {t["name"] for t in _client(SIGNAL, RecordedHttp(SIGNAL)).list_tools()}
    assert recorded_signal <= live_signal
    live_data = {t["name"]: t for t in data.list_tools()}
    assert live_data["do_query"]["inputSchema"]["required"] == ["entry_id"]
