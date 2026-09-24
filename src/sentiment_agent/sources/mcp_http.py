"""MCP over streamable HTTP, and the plumbing both Bitget readers share.

Two Bitget services are read through the Model Context Protocol, both keyless:

* ``bitget-signal`` at :data:`SIGNAL_MCP_URL`, the server behind Bitget's five research Skills
  (``bitget-signal/scripts/install.js:30``; it reports itself as ``market-data-mcp``);
* ``bitget-mcp-server`` at :data:`DATA_MCP_URL`, Bitget's catalog of 67 datasets behind the two
  tools ``guide`` and ``do_query`` (it reports itself as ``bitget-mcp-server 4.0.5``).

The transport follows the MCP streamable-HTTP specification (revision 2025-06-18): every message
is a ``POST`` accepting ``application/json`` and ``text/event-stream``; the ``initialize`` reply
carries an ``Mcp-Session-Id`` header that every later request repeats, together with
``MCP-Protocol-Version``; ``notifications/initialized`` is sent once before any tool call; a request
answered ``404`` under a session means the session expired, and the client starts a new one.

Measured on 2026-09-24 against both servers, and each one load-bearing:

* **Both return 403 to Python's default User-Agent** (Cloudflare error 1010) and 200 to any explicit
  one, so :class:`StreamableHttpMcp` always sends ``user_agent`` (default ``curl/8.0``). A client
  that omits it fails in a way that looks exactly like a credential problem on a service that needs
  no credential (ARGUS ``market/bitget_mcp.py`` found this first).
* **Replies are Server-Sent Events** with CRLF line ends (``event: message``, ``data: {...}``), not
  a bare JSON body, and a slow tool can be preceded by keep-alive events. :func:`parse_sse_or_json`
  implements the SSE field grammar rather than looking for one ``data:`` prefix.
* **A reply is matched to its request by JSON-RPC id.** ARGUS ``market/evidence.py``
  (``BitgetSkillSource``) once sent one id for every call and took the last frame of the body; a
  live probe then had ``sentiment_index`` return another tool's payload. Ids here are unique per
  client and the frame is selected by id; a reply without the right id is refused.

The typed readers (:mod:`.signal_skills`, :mod:`.bitget_data`) never let a failure out as an
exception: :func:`invoke` turns a transport failure into ``TIMEOUT`` or ``ERROR``, and
:func:`source_call` records every call, answered or not, as a
:class:`~sentiment_agent.types.SourceCall`. The payload helpers used by both readers
(:func:`hollow`, :func:`finite`, :func:`row_time`, :func:`rows_of`) live here too.

Provenance
----------
:func:`hollow` is ported from ARGUS ``argus/src/argus/market/skills.py`` (MIT, Copyright (c) 2026
Pratiikpy, the same author as this project) at commit ``3dec6baf9dfa7be37c7b452e26a9b139df252f75``,
sha256 ``297bde83bb872a054af392f37e8693d074f1a3818b6d385345b3c8d246e8f654``. The ARGUS repository is
private, so its licence and every pin are recorded in ``third_party/argus/PROVENANCE.md``. Copied,
never imported. Changed: a non-finite number and a whitespace-only string are hollow too (neither
carries a reading, and a NaN would stop the snapshot from hashing), and four keys that name the
backend are treated as describing the request (:data:`_REQUEST_KEYS`). The session handshake and the
User-Agent rule are taken from ARGUS ``argus/src/argus/market/bitget_mcp.py`` (same commit, sha256
``bc8f550f718cb6a40339439ed2233505d9845317f1a94d13a702ba5ae8905721``); the SSE parser, session
expiry, protocol-version header, pagination and deadline are new here.
"""

import http.client
import json
import math
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final

from sentiment_agent.types import (
    BlobRef,
    BlobStore,
    Clock,
    McpCaller,
    McpToolResult,
    SourceCall,
    SourceHealth,
    ToolkitSurface,
)

SIGNAL_MCP_URL: Final = "https://datahub.noxiaohao.com/mcp"
"""bitget-signal, as its installer registers it (``bitget-signal/scripts/install.js:30``)."""

DATA_MCP_URL: Final = "https://agent.bitget.com/mcp"
"""bitget-mcp-server, the HTTP transport the S2 handbook publishes."""

PROTOCOL_VERSION: Final = "2025-06-18"
"""The MCP revision requested. Both servers answered with this revision on 2026-09-24."""

CLIENT_INFO: Final[Mapping[str, str]] = {"name": "t2-sentiment-agent", "version": "0.1.0"}

MAX_REPLY_BYTES: Final = 32 * 1024 * 1024
"""A reply larger than this is refused. The largest reply measured was 156 KB
(``crypto_futures_funding_rate`` with no limit); 32 MB is a runaway guard, not a budget."""

ERROR_TEXT_LIMIT: Final = 500
"""How much of an error message a :class:`SourceCall` keeps."""

MAX_LIST_PAGES: Final = 50
"""``tools/list`` pagination stops here even if the server keeps returning a cursor."""

HttpPost = Callable[[str, bytes, Mapping[str, str], float], tuple[int, Mapping[str, str], bytes]]
"""``(url, body, headers, timeout_s) -> (status, headers, body)``.

Must return non-2xx statuses rather than raise them, raise :class:`TimeoutError` when the deadline
passes, and raise :class:`OSError` for any other transport failure."""


class McpError(RuntimeError):
    """The server could not be reached, refused the request, or replied with something unusable.

    Raised by :class:`StreamableHttpMcp`; the typed readers catch it and record the call as
    ``ERROR``. ``code`` is the JSON-RPC error code or the HTTP status when there was one.
    """

    def __init__(self, message: str, *, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code


class McpTimeoutError(McpError):
    """The server did not answer inside the deadline. Recorded as ``TIMEOUT``, not ``ERROR``:
    a timeout means we do not know, which is a different fact from a refusal."""


# ================================================================================================
# HTTP
# ================================================================================================

_LOCAL_HOSTS: Final = frozenset({"localhost", "127.0.0.1", "::1"})


def _check_url(url: str) -> None:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme == "https" and parts.hostname:
        return
    if parts.scheme == "http" and parts.hostname in _LOCAL_HOSTS:
        return
    raise ValueError(f"refusing to call {url!r}: only https (or http to localhost) is allowed")


def _lower_headers(headers: Any) -> dict[str, str]:
    return {str(k).lower(): str(v) for k, v in headers.items()}


def urllib_post(
    url: str, body: bytes, headers: Mapping[str, str], timeout: float
) -> tuple[int, Mapping[str, str], bytes]:
    """The standard-library :data:`HttpPost`.

    Reads the reply in chunks against a wall-clock deadline, because a server that trickles
    keep-alive events would otherwise defeat a per-read socket timeout forever. The worst case is
    therefore about twice ``timeout`` (one blocked read after the deadline check), never unbounded.
    """
    _check_url(url)
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")  # noqa: S310 - scheme checked above
    deadline = time.monotonic() + timeout
    try:
        response = urllib.request.urlopen(request, timeout=timeout)  # noqa: S310 - scheme checked above
    except urllib.error.HTTPError as exc:
        try:
            data = exc.read()
        except (OSError, http.client.HTTPException):
            data = b""
        return exc.code, _lower_headers(exc.headers), data
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise TimeoutError(f"connecting to {url} timed out") from exc
        raise OSError(f"cannot reach {url}: {exc.reason}") from exc
    except http.client.HTTPException as exc:
        raise OSError(f"HTTP protocol failure from {url}: {exc!r}") from exc
    with response:
        chunks: list[bytes] = []
        total = 0
        try:
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"reply from {url} did not finish within {timeout:.0f}s")
                chunk = response.read1(65536)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_REPLY_BYTES:
                    raise OSError(f"reply from {url} exceeded {MAX_REPLY_BYTES} bytes")
                chunks.append(chunk)
        except http.client.HTTPException as exc:
            raise OSError(f"reply from {url} broke off: {exc!r}") from exc
        return int(response.status), _lower_headers(response.headers), b"".join(chunks)


def _header(headers: Mapping[str, str], name: str) -> str | None:
    """Case-insensitive header lookup (servers send ``mcp-session-id`` in lower case)."""
    wanted = name.lower()
    for key, value in headers.items():
        if key.lower() == wanted:
            return value
    return None


# ================================================================================================
# Framing
# ================================================================================================

_LINE_BREAK: Final = re.compile(r"\r\n|\r|\n")


def _sse_messages(text: str) -> tuple[list[Any], int]:
    """Decode every ``message`` event of an SSE body. Returns ``(payloads, undecodable)``.

    The WHATWG event-stream grammar: lines end in CRLF, LF or CR; a blank line dispatches the
    event; ``data`` lines accumulate and are joined with ``\\n``; a line starting with ``:`` is a
    comment (keep-alive); one space after the colon is dropped; an event without an ``event`` field
    is a ``message``. An event still open at the end of the body is dispatched too (the grammar
    would discard it; a truncated frame then fails to decode instead of vanishing).
    """
    payloads: list[Any] = []
    undecodable = 0
    event = ""
    data: list[str] = []

    def dispatch() -> None:
        nonlocal undecodable
        if data and event in ("", "message"):
            try:
                payloads.append(json.loads("\n".join(data)))
            except json.JSONDecodeError:
                undecodable += 1

    for line in [*_LINE_BREAK.split(text), ""]:
        if line == "":
            dispatch()
            event, data = "", []
            continue
        if line.startswith(":"):
            continue
        name, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if name == "data":
            data.append(value)
        elif name == "event":
            event = value
    return payloads, undecodable


def _is_response(message: Any) -> bool:
    return (
        isinstance(message, dict)
        and "method" not in message
        and ("result" in message or "error" in message)
    )


def parse_sse_or_json(raw: bytes, *, request_id: int | str | None = None) -> dict[str, Any]:
    """The JSON-RPC response carried by one HTTP reply, whether SSE-framed or a plain JSON body.

    With ``request_id`` the response whose ``id`` equals it is returned and any other response is
    refused; without it, exactly one response must be present. Server notifications and requests
    inside the stream (``method`` members) are skipped. Raises :class:`McpError` when no acceptable
    response is found. A body that is not UTF-8 is decoded with replacement: the raw bytes are kept
    in the blob store, and numbers are ASCII either way.
    """
    text = raw.decode("utf-8", errors="replace").removeprefix("﻿")
    stripped = text.lstrip()
    undecodable = 0
    if stripped.startswith(("{", "[")):
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise McpError(f"reply is neither SSE nor valid JSON: {text[:200]!r}") from exc
        messages = decoded if isinstance(decoded, list) else [decoded]
    else:
        messages, undecodable = _sse_messages(text)

    responses = [m for m in messages if _is_response(m)]
    if request_id is not None:
        matches = [m for m in responses if m.get("id") == request_id]
        if len(matches) == 1:
            return dict(matches[0])
        if len(matches) > 1:
            raise McpError(f"{len(matches)} responses carry id {request_id!r}; refusing to choose")
        seen = [m.get("id") for m in responses]
        raise McpError(
            f"no JSON-RPC response with id {request_id!r} in the reply (ids seen: {seen}, "
            f"undecodable frames: {undecodable}); refusing to return another request's response"
        )
    if len(responses) == 1:
        return dict(responses[0])
    if not responses:
        raise McpError(
            f"no JSON-RPC response in the reply (undecodable frames: {undecodable}): {text[:200]!r}"
        )
    raise McpError(f"{len(responses)} responses in one reply and no request id to choose by")


# ================================================================================================
# Client
# ================================================================================================


class StreamableHttpMcp:
    """One MCP server over streamable HTTP. Satisfies :class:`~sentiment_agent.types.McpCaller`.

    Thread-safe: ids come from a locked counter and the session is established once under the
    lock, so the readers may call tools concurrently. The session is created on first use;
    :meth:`initialize` may also be called directly (it always starts a new session).
    """

    def __init__(
        self,
        url: str,
        *,
        server_label: str,
        clock: Clock,
        http: HttpPost = urllib_post,
        blobs: BlobStore | None = None,
        timeout_s: float = 45.0,
        user_agent: str = "curl/8.0",
        max_in_flight: int = 6,
    ) -> None:
        """``max_in_flight`` caps concurrent requests to this server whatever the callers' thread
        pools do; the probe alone would otherwise put about fifty on one host at once."""
        _check_url(url)
        if max_in_flight < 1:
            raise ValueError("max_in_flight must be at least 1")
        if not server_label.strip():
            raise ValueError("server_label must name the server")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if not user_agent.strip():
            raise ValueError(
                "user_agent must be explicit: both Bitget MCP servers return 403 to Python's "
                "default User-Agent"
            )
        self._url = url
        self._label = server_label
        self._clock = clock
        self._http = http
        self._blobs = blobs
        self._timeout = timeout_s
        self._user_agent = user_agent
        self._lock = threading.RLock()
        self._in_flight = threading.BoundedSemaphore(max_in_flight)
        self._next_id = 0
        self._session: str | None = None
        self._protocol: str | None = None
        self._server_info: dict[str, Any] = {}
        self._ready = False
        self._sessions_started = 0

    # --- introspection --------------------------------------------------------------------------

    @property
    def server(self) -> str:
        return self._label

    @property
    def url(self) -> str:
        return self._url

    @property
    def session_id(self) -> str | None:
        return self._session

    @property
    def protocol_version(self) -> str | None:
        return self._protocol

    @property
    def server_info(self) -> dict[str, Any]:
        """``serverInfo`` from the last ``initialize`` (e.g. name and version)."""
        return dict(self._server_info)

    @property
    def sessions_started(self) -> int:
        """How many sessions this client has opened (more than one means one expired)."""
        return self._sessions_started

    # --- protocol -------------------------------------------------------------------------------

    def initialize(self) -> dict[str, Any]:
        """Open a new session: ``initialize``, then ``notifications/initialized``."""
        with self._lock:
            self._session = None
            self._protocol = None
            self._ready = False
            params = {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": dict(CLIENT_INFO),
            }
            message, _, headers = self._exchange("initialize", params, retry_expired=False)
            result = message.get("result")
            if not isinstance(result, dict):
                raise McpError(f"{self._label}: initialize returned no result object")
            self._session = _header(headers, "Mcp-Session-Id")
            version = result.get("protocolVersion")
            self._protocol = version if isinstance(version, str) and version else PROTOCOL_VERSION
            info = result.get("serverInfo")
            self._server_info = dict(info) if isinstance(info, dict) else {}
            self._notify("notifications/initialized")
            self._ready = True
            self._sessions_started += 1
            return dict(result)

    def list_tools(self) -> list[dict[str, Any]]:
        """Every tool the server offers, following ``nextCursor`` pagination."""
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        for _ in range(MAX_LIST_PAGES):
            params: dict[str, Any] = {} if cursor is None else {"cursor": cursor}
            message, _, _ = self._rpc("tools/list", params)
            result = message.get("result")
            if not isinstance(result, dict) or not isinstance(result.get("tools"), list):
                raise McpError(f"{self._label}: tools/list returned no tool list")
            tools.extend(dict(t) for t in result["tools"] if isinstance(t, dict))
            following = result.get("nextCursor")
            if not isinstance(following, str) or not following:
                return tools
            cursor = following
        raise McpError(f"{self._label}: tools/list still paginating after {MAX_LIST_PAGES} pages")

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> McpToolResult:
        """Call one tool. A tool that reports failure (``isError``) is returned, not raised: that
        is the server answering. Transport and protocol failures raise :class:`McpError`."""
        message, raw, headers = self._rpc(
            "tools/call", {"name": name, "arguments": dict(arguments)}
        )
        result = message.get("result")
        if not isinstance(result, dict):
            raise McpError(f"{self._label}: tools/call {name} returned no result object")
        content = result.get("content")
        texts: list[str] = []
        if isinstance(content, list):
            texts = [
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
        structured = result.get("structuredContent")
        return McpToolResult(
            server=self._label,
            tool=name,
            is_error=bool(result.get("isError", False)),
            structured=dict(structured) if isinstance(structured, dict) else None,
            text="\n".join(texts),
            raw=self._store(raw, headers),
        )

    # --- internals ------------------------------------------------------------------------------

    def _ensure_session(self) -> None:
        if self._ready:
            return
        with self._lock:
            if not self._ready:
                self.initialize()

    def _new_id(self) -> int:
        with self._lock:
            self._next_id += 1
            return self._next_id

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": self._user_agent,
        }
        if self._session:
            headers["Mcp-Session-Id"] = self._session
        if self._protocol:
            headers["MCP-Protocol-Version"] = self._protocol
        return headers

    def _post(self, payload: Mapping[str, Any]) -> tuple[int, Mapping[str, str], bytes]:
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        try:
            with self._in_flight:
                return self._http(self._url, body, self._headers(), self._timeout)
        except TimeoutError as exc:
            raise McpTimeoutError(f"{self._label}: {exc or 'timed out'}") from exc
        except OSError as exc:
            raise McpError(f"{self._label}: transport failure: {exc}") from exc

    def _rpc(
        self, method: str, params: Mapping[str, Any]
    ) -> tuple[dict[str, Any], bytes, Mapping[str, str]]:
        self._ensure_session()
        return self._exchange(method, params, retry_expired=True)

    def _exchange(
        self, method: str, params: Mapping[str, Any], *, retry_expired: bool
    ) -> tuple[dict[str, Any], bytes, Mapping[str, str]]:
        request_id = self._new_id()
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": dict(params)}
        status, headers, raw = self._post(payload)
        if status == 404 and retry_expired and self._session is not None:
            # The specification's signal for an expired session: start a new one, then retry once.
            self.initialize()
            request_id = self._new_id()
            payload = {**payload, "id": request_id}
            status, headers, raw = self._post(payload)
        if not 200 <= status < 300:
            raise McpError(self._http_failure(method, status, raw), code=status)
        message = parse_sse_or_json(raw, request_id=request_id)
        error = message.get("error")
        if error is not None:
            code = error.get("code") if isinstance(error, dict) else None
            text = error.get("message") if isinstance(error, dict) else error
            raise McpError(
                f"{self._label}: {method} failed: JSON-RPC error {code}: {text}",
                code=code if isinstance(code, int) else None,
            )
        return message, raw, headers

    def _notify(self, method: str) -> None:
        """Send a notification. The server's status is not asserted: a notification has no reply,
        and a server that dislikes it will say so on the next request."""
        self._post({"jsonrpc": "2.0", "method": method})

    def _http_failure(self, method: str, status: int, raw: bytes) -> str:
        snippet = raw[:200].decode("utf-8", errors="replace")
        hint = ""
        if status == 403:
            hint = " (Cloudflare refuses some User-Agents; this client sends an explicit one)"
        return f"{self._label}: {method} answered HTTP {status}{hint}: {snippet!r}"

    def _store(self, raw: bytes, headers: Mapping[str, str]) -> BlobRef | None:
        if self._blobs is None:
            return None
        media = (_header(headers, "Content-Type") or "application/octet-stream").split(";")[0]
        return self._blobs.put(raw, media.strip() or "application/octet-stream")


# ================================================================================================
# Calls, recorded
# ================================================================================================


@dataclass(frozen=True, slots=True)
class Invocation:
    """One tool call as it happened: when, how long, and what came back (or why nothing did)."""

    started_at: datetime
    latency_ms: int
    result: McpToolResult | None
    failure: SourceHealth | None
    """``TIMEOUT`` or ``ERROR`` when the transport failed; ``None`` when the server answered."""
    error: str | None


def _clip(text: str) -> str:
    return text if len(text) <= ERROR_TEXT_LIMIT else text[: ERROR_TEXT_LIMIT - 3] + "..."


def invoke(mcp: McpCaller, clock: Clock, tool: str, arguments: Mapping[str, Any]) -> Invocation:
    """Call a tool and never raise. Latency is measured on a monotonic timer, not on ``clock``."""
    started = clock.now()
    t0 = time.perf_counter()

    def elapsed() -> int:
        return max(0, round((time.perf_counter() - t0) * 1000))

    try:
        result = mcp.call_tool(tool, arguments)
    except McpTimeoutError as exc:
        return Invocation(started, elapsed(), None, SourceHealth.TIMEOUT, _clip(str(exc)))
    except McpError as exc:
        return Invocation(started, elapsed(), None, SourceHealth.ERROR, _clip(str(exc)))
    except Exception as exc:  # a caller that is not ours (or a defect) must not crash a snapshot
        detail = _clip(f"{type(exc).__name__}: {exc}")
        return Invocation(started, elapsed(), None, SourceHealth.ERROR, detail)
    return Invocation(started, elapsed(), result, None, None)


def source_call(
    *,
    surface: ToolkitSurface,
    source: str,
    params: Mapping[str, Any],
    invocation: Invocation,
    health: SourceHealth,
    rows: int = 0,
    error: str | None = None,
) -> SourceCall:
    """The :class:`SourceCall` for one invocation. ``params`` are recorded as strings."""
    blob = invocation.result.raw if invocation.result is not None else None
    detail = error if error is not None else invocation.error
    return SourceCall(
        call_id=f"{surface.value}-{uuid.uuid4().hex}",
        surface=surface,
        source=source,
        params={str(k): _param_text(v) for k, v in params.items()},
        health=health,
        started_at=invocation.started_at,
        latency_ms=invocation.latency_ms,
        rows=max(0, rows),
        blob=blob,
        error=_clip(detail) if detail else None,
    )


def disabled_call(
    *, surface: ToolkitSurface, source: str, params: Mapping[str, Any], clock: Clock, reason: str
) -> SourceCall:
    """A call deliberately not made, recorded so its absence is visible."""
    return SourceCall(
        call_id=f"{surface.value}-{uuid.uuid4().hex}",
        surface=surface,
        source=source,
        params={str(k): _param_text(v) for k, v in params.items()},
        health=SourceHealth.DISABLED,
        started_at=clock.now(),
        latency_ms=0,
        rows=0,
        blob=None,
        error=_clip(reason),
    )


def _param_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"), sort_keys=True, default=str)


def tool_payload(result: McpToolResult) -> Any:
    """The decoded payload of a tool result: ``structuredContent`` when the server sent it,
    otherwise the text content parsed as JSON. Raises :class:`ValueError` when the text is not
    JSON."""
    if result.structured is not None:
        return result.structured
    text = result.text.strip()
    if not text:
        return None
    return json.loads(text)


# ================================================================================================
# Payload helpers shared by the readers
# ================================================================================================

_REQUEST_KEYS: Final = frozenset(
    {
        "url",
        "note",
        "source",
        "feed",
        "symbol",
        "timeframe",
        "period",
        "platform",
        "provider",
        "exchange",
        "interval",
    }
)
"""Keys that describe the request or the backend rather than answer it. A payload made only of
these carries nothing about the market. The first seven are ARGUS ``market/skills.py``
``_HOLLOW_KEYS``; ``platform``, ``provider``, ``exchange`` and ``interval`` were added after the
live probe of 2026-09-24 saw ``social_trending`` answer ``{"platform": "xueqiu", "provider":
"all_failed", "items": []}``, which the original rule passed as substantive."""


def _hollow_scalar(value: Any) -> bool:
    if value is None or value is False:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, bool):
        return False
    if isinstance(value, int | float):
        return value == 0 or not math.isfinite(value)
    return False


def hollow(payload: Any) -> bool:
    """True when a payload has the *shape* of an answer and none of the substance.

    Seen live: ``sentiment_index`` answering ``{"alt_me_error": ""}``, every
    ``derivatives_sentiment`` action answering ``{"error": ""}``, and ``news_feed`` answering five
    feeds each ``{"feed": ..., "error": "", "items": []}`` (2026-09-24; the same on 2026-09-13,
    -20, -21 and -22 in ARGUS and in two other S2 projects). ARGUS also saw ``rates_yields``
    answer every tenor ``{"error": ""}`` beside a derived ``spread_10y2y: 0.0``.

    The rule: a container is hollow when every child container is hollow and every scalar is a
    zero, a false, an empty or blank string, a ``None`` or a non-finite number, or sits under a key
    that only names the request or an error. One real value under a real key makes the payload
    substantive.
    """
    if isinstance(payload, dict):
        for key, value in payload.items():
            name = str(key)
            if name.endswith("error") or name in _REQUEST_KEYS:
                continue
            if isinstance(value, dict | list):
                if not hollow(value):
                    return False
            elif not _hollow_scalar(value):
                return False
        return True
    if isinstance(payload, list):
        return all(hollow(item) for item in payload)
    return _hollow_scalar(payload)


def error_envelope(payload: Any) -> tuple[SourceHealth, str] | None:
    """Classify an error envelope (a dict whose keys all end in ``error``), else ``None``.

    Two kinds were recorded from bitget-signal on 2026-09-24, both with ``isError: false``:
    ``{"error": ""}`` / ``{"alt_me_error": ""}``, an upstream that failed without saying why, which
    is ``HOLLOW``; and ``{"error": "Unknown action: not_an_action"}``, the server refusing the
    request, which is ``ERROR`` because it says our call is wrong.
    """
    if not (isinstance(payload, dict) and payload):
        return None
    if not all(str(key).endswith("error") for key in payload):
        return None
    text = json.dumps(payload, ensure_ascii=False)[:200]
    messages = [v for v in payload.values() if isinstance(v, str) and v.strip()]
    if messages:
        return SourceHealth.ERROR, f"the server refused the call: {text}"
    return SourceHealth.HOLLOW, f"upstream error envelope {text}"


def finite(value: Any) -> float | None:
    """A finite float from a number or a numeric string; ``None`` for anything else (bools
    included: a flag is not a measurement)."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


_EPOCH_MS_FLOOR: Final = 10**11
"""Epoch values at or above this are milliseconds (10**11 seconds is the year 5138)."""
_EARLIEST: Final = datetime(2000, 1, 1, tzinfo=UTC)
_LATEST: Final = datetime(2100, 1, 1, tzinfo=UTC)


def parse_instant(value: Any) -> datetime | None:
    """An aware UTC datetime from epoch seconds or milliseconds (number or digit string) or an
    ISO-8601 string with an offset. Naive and out-of-range values are refused (``None``): a time
    whose zone is unknown cannot be compared with ``now``."""
    if isinstance(value, bool) or value is None:
        return None
    moment: datetime | None = None
    if isinstance(value, int | float) or (isinstance(value, str) and value.strip().isdigit()):
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            return None
        seconds = number / 1000 if number >= _EPOCH_MS_FLOOR else number
        try:
            moment = datetime.fromtimestamp(seconds, tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip())
        except ValueError:
            return None
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return None
        moment = parsed.astimezone(UTC)
    if moment is None or not _EARLIEST <= moment < _LATEST:
        return None
    return moment


_TIME_KEYS: Final = ("time", "timestamp", "date", "ts")


def row_time(row: Mapping[str, Any]) -> datetime | None:
    """The time a row describes: the first of ``time``, ``timestamp``, ``date``, ``ts`` that
    parses. On bitget-mcp-server ``time`` is epoch milliseconds on every row seen, while
    ``timestamp`` is seconds on some entries and ISO text on others, so ``time`` is read first."""
    for key in _TIME_KEYS:
        if key in row:
            moment = parse_instant(row[key])
            if moment is not None:
                return moment
    return None


_ROW_CONTAINERS: Final = ("data", "results", "history", "items", "rows", "list")


def rows_of(payload: Any) -> list[dict[str, Any]]:
    """The row dicts of a payload: the payload itself when it is a list, the first list found under
    a conventional container key (one level of nesting followed), or the payload as one row."""
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in _ROW_CONTAINERS:
            inner = payload.get(key)
            if isinstance(inner, list):
                return [dict(item) for item in inner if isinstance(item, dict)]
            if isinstance(inner, dict):
                nested = rows_of(inner)
                if nested and nested != [dict(inner)]:
                    return nested
        return [dict(payload)]
    return []


def first_present(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    """The value of the first key in ``keys`` that the row carries with a non-``None`` value."""
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return None


FUTURE_TOLERANCE: Final = timedelta(hours=1)
"""A row stamped further ahead of ``now`` than this is refused as a clock or data defect."""


def freshness_problem(moment: datetime | None, now: datetime, max_age: timedelta) -> str | None:
    """Why a row's time disqualifies it as *current*, or ``None`` when it is fresh."""
    if moment is None:
        return "row carries no parseable timestamp, so its freshness cannot be checked"
    if moment > now + FUTURE_TOLERANCE:
        return f"row is stamped {moment.isoformat()}, ahead of now ({now.isoformat()})"
    if now - moment > max_age:
        hours = (now - moment).total_seconds() / 3600
        return (
            f"newest row is stamped {moment.isoformat()}, {hours:.1f}h old "
            f"(limit {max_age.total_seconds() / 3600:.0f}h): stale, not current"
        )
    return None


def newest(rows: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], datetime] | None:
    """The row with the latest :func:`row_time` (rows without one are skipped). The order rows
    arrive in is not relied on: bitget-mcp-server returns ratios oldest first, the taker entry
    newest first, and the crypto Fear & Greed entry newest first."""
    best: tuple[Mapping[str, Any], datetime] | None = None
    for row in rows:
        moment = row_time(row)
        if moment is not None and (best is None or moment > best[1]):
            best = (row, moment)
    return best


def iter_dicts(values: Any) -> Iterator[dict[str, Any]]:
    if isinstance(values, list):
        for item in values:
            if isinstance(item, dict):
                yield item


__all__ = [
    "CLIENT_INFO",
    "DATA_MCP_URL",
    "FUTURE_TOLERANCE",
    "MAX_REPLY_BYTES",
    "PROTOCOL_VERSION",
    "SIGNAL_MCP_URL",
    "HttpPost",
    "Invocation",
    "McpError",
    "McpTimeoutError",
    "StreamableHttpMcp",
    "disabled_call",
    "error_envelope",
    "finite",
    "first_present",
    "freshness_problem",
    "hollow",
    "invoke",
    "iter_dicts",
    "newest",
    "parse_instant",
    "parse_sse_or_json",
    "row_time",
    "rows_of",
    "source_call",
    "tool_payload",
    "urllib_post",
]
