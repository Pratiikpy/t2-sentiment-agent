"""Replay of recorded MCP sessions, for the sources tests.

The fixtures in ``tests/fixtures/sources/`` are real keyless exchanges with bitget-signal and
bitget-mcp-server, recorded by ``tests/fixtures/sources/record_fixtures.py`` through the production
client. :class:`RecordedHttp` is an :data:`~sentiment_agent.sources.mcp_http.HttpPost` that serves
them back: a request is matched to its recording by method, tool name and arguments, and the
recorded JSON-RPC id in the reply is rewritten to the id of the request being answered, so the
client's id correlation is exercised exactly as it is live.
"""

import json
import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from sentiment_agent.clock import ManualClock

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "sources"
PLACEHOLDER_ID = "__ID__"


def request_key(message: Mapping[str, Any]) -> str:
    """How a request is matched to its recording."""
    method = str(message.get("method"))
    if method == "tools/call":
        params = message.get("params") or {}
        body = {"name": params.get("name"), "arguments": params.get("arguments")}
        return "tools/call " + json.dumps(
            body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
    if method == "tools/list":
        cursor = (message.get("params") or {}).get("cursor")
        return f"tools/list {cursor}" if cursor else "tools/list"
    return method


def call_key(tool: str, arguments: Mapping[str, Any]) -> str:
    return request_key(
        {"method": "tools/call", "params": {"name": tool, "arguments": dict(arguments)}}
    )


@dataclass(frozen=True)
class Exchange:
    request: dict[str, Any]
    status: int
    headers: dict[str, str]
    body: str
    request_headers: dict[str, str]


@dataclass(frozen=True)
class Session:
    server: str
    url: str
    captured_at: datetime
    exchanges: dict[str, Exchange]
    meta: dict[str, Any]

    def clock(self) -> ManualClock:
        """A clock standing at the capture time, so freshness is judged as it was live."""
        return ManualClock(self.captured_at)

    def payload(self, key: str) -> Any:
        """The decoded tool payload of one recorded exchange (for assertions)."""
        from sentiment_agent.sources.mcp_http import parse_sse_or_json

        exchange = self.exchanges[key]
        message = parse_sse_or_json(exchange.body.encode("utf-8"))
        result = message["result"]
        if isinstance(result.get("structuredContent"), dict):
            return result["structuredContent"]
        text = "".join(c.get("text", "") for c in result.get("content", []))
        return json.loads(text)


def load_session(name: str) -> Session:
    raw = json.loads((FIXTURES / name).read_text("utf-8"))
    exchanges: dict[str, Exchange] = {}
    for item in raw["exchanges"]:
        key = request_key(item["request"])
        exchanges.setdefault(
            key,
            Exchange(
                request=item["request"],
                status=int(item["status"]),
                headers=dict(item["headers"]),
                body=item["body"],
                request_headers=dict(item.get("request_headers", {})),
            ),
        )
    meta = {k: v for k, v in raw.items() if k != "exchanges"}
    return Session(
        server=raw["server"],
        url=raw["url"],
        captured_at=datetime.fromisoformat(raw["captured_at"]),
        exchanges=exchanges,
        meta=meta,
    )


def rewrite_id(body: str, recorded: Any, wanted: Any) -> str:
    """Put ``wanted`` where the recorded frame carries ``recorded`` as its id. The frame's id is
    the first ``"id":`` member after ``"jsonrpc"``, so only the first occurrence is replaced, and
    only a whole value (``"id":1`` never matches inside ``"id":10``)."""
    pattern = re.compile(r'"id"\s*:\s*' + re.escape(json.dumps(recorded)) + r"(?=\s*[,}])")
    rewritten, count = pattern.subn(f'"id":{json.dumps(wanted)}', body, count=1)
    if count != 1:
        raise AssertionError(f"recorded body carries no id {recorded!r}")
    return rewritten


@dataclass
class RecordedHttp:
    """Serves a recorded session. Unrecorded requests fail the test loudly unless ``overrides``
    supplies them. Thread-safe (the readers call concurrently)."""

    session: Session
    overrides: dict[str, tuple[int, dict[str, str], bytes]] = field(default_factory=dict)
    default: Callable[[dict[str, Any]], tuple[int, dict[str, str], bytes]] | None = None
    """Answers requests the session did not record (``None``: they fail the test)."""
    requests: list[tuple[str, dict[str, Any], dict[str, str]]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __call__(
        self, url: str, body: bytes, headers: Mapping[str, str], timeout: float
    ) -> tuple[int, Mapping[str, str], bytes]:
        message = json.loads(body)
        with self._lock:
            self.requests.append((url, message, dict(headers)))
        if "id" not in message:  # a notification: the servers answer 202 with no body
            return 202, {"content-type": "application/json"}, b""
        key = request_key(message)
        if key in self.overrides:
            status, reply_headers, reply = self.overrides[key]
            text = reply.decode("utf-8")
            if PLACEHOLDER_ID in text:
                text = rewrite_id(text, PLACEHOLDER_ID, message["id"])
            return status, reply_headers, text.encode("utf-8")
        exchange = self.session.exchanges.get(key)
        if exchange is None and self.default is not None:
            status, reply_headers, reply = self.default(message)
            text = reply.decode("utf-8")
            if PLACEHOLDER_ID in text:
                text = rewrite_id(text, PLACEHOLDER_ID, message["id"])
            return status, reply_headers, text.encode("utf-8")
        if exchange is None:
            raise AssertionError(f"unrecorded request: {key}")
        text = exchange.body
        if exchange.status == 200 and text:
            text = rewrite_id(text, exchange.request["id"], message["id"])
        return exchange.status, dict(exchange.headers), text.encode("utf-8")

    def keys(self) -> list[str]:
        return [request_key(message) for _, message, _ in self.requests]


def sse_reply(result: Mapping[str, Any]) -> bytes:
    """An SSE-framed JSON-RPC result whose id :class:`RecordedHttp` fills in."""
    frame = json.dumps(
        {"jsonrpc": "2.0", "id": PLACEHOLDER_ID, "result": dict(result)}, separators=(",", ":")
    )
    return f"event: message\r\ndata: {frame}\r\n\r\n".encode()


def tool_text(payload: Any) -> dict[str, Any]:
    """A ``tools/call`` result carrying ``payload`` as JSON text (how bitget-signal answers)."""
    return {"content": [{"type": "text", "text": json.dumps(payload)}], "isError": False}


def load_json(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text("utf-8"))


__all__ = [
    "FIXTURES",
    "PLACEHOLDER_ID",
    "Exchange",
    "RecordedHttp",
    "Session",
    "call_key",
    "load_json",
    "load_session",
    "request_key",
    "rewrite_id",
    "sse_reply",
    "tool_text",
]
