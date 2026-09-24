"""bitget-signal, read as typed sentiment readings.

The five research Skills Bitget ships with Agent Hub (``bitget-signal/skills/*/SKILL.md``) all
call one keyless MCP server. This module makes the calls the ``sentiment-analyst`` and
``news-briefing`` Skills document, with their documented tool names, actions and arguments:

==========================  ======================================================================
reading                     call (``SKILL.md``)
==========================  ======================================================================
crypto Fear & Greed         ``sentiment_index(action="current")``
retail long/short           ``derivatives_sentiment(action="long_short", symbol, period)``
top-trader long/short       ``derivatives_sentiment(action="top_ls", symbol, period)``
taker buy/sell ratio        ``derivatives_sentiment(action="taker_ratio", symbol, period)``
open interest history       ``derivatives_sentiment(action="open_interest", symbol, period)``
Reddit trending             ``derivatives_sentiment(action="reddit_trending", limit, filter)``
news                        ``news_feed(action="latest", feeds, limit, keyword)``
==========================  ======================================================================

**What the server returned when this was built.** Every one of those calls, made on 2026-09-24,
answered after 15-31 seconds with a well-formed envelope and nothing in it: ``{"alt_me_error":
""}`` for the index, ``{"error": ""}`` for every ``derivatives_sentiment`` action, and five feeds of
``{"feed": ..., "error": "", "items": []}`` for news. ARGUS measured the same on 2026-09-13, -20 and
-21, and two other S2 projects on -20 and -22. The server is timing out against its own upstream
providers. Those replies are recorded as fixtures and classified ``HOLLOW``, never as a quiet
market.

**NOT VERIFIED: the shape of a successful reply.** Nobody has been seen to receive one. The tool
descriptions name the upstreams (``derivatives_sentiment``: "Binance futures long/short ratios, open
interest history, taker buy/sell ratio" and "Reddit crypto trending (ApeWisdom)"; the index's error
key ``alt_me_error`` names alternative.me), so the parsers accept exactly the field names those
upstreams publish, recorded from the upstreams themselves on 2026-09-24
(``tests/fixtures/sources/upstream_shapes.json``), plus the snake-case spellings bitget-mcp-server
uses for the same data. A reply in any other shape is ``HOLLOW`` with the keys it carried, so a
changed shape is reported rather than misread. No value is ever guessed from an unknown key.

Every value must also be *current*: rows carry timestamps, and the newest must be younger than a
limit set from the series' own period (:func:`max_age_for`), or the reading is ``HOLLOW``
("stale"). A long/short ratio from last week presented as today's crowd is worse than none.
"""

import html
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any, Final

from sentiment_agent.hashing import content_hash
from sentiment_agent.sources.mcp_http import (
    Invocation,
    error_envelope,
    finite,
    first_present,
    freshness_problem,
    hollow,
    invoke,
    iter_dicts,
    newest,
    parse_instant,
    row_time,
    rows_of,
    source_call,
    tool_payload,
)
from sentiment_agent.types import (
    Clock,
    McpCaller,
    SourceCall,
    SourceHealth,
    TextItem,
    ToolkitSurface,
)

SURFACE: Final = ToolkitSurface.SIGNAL_MCP

PERIODS: Final = frozenset({"5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"})
"""The ``period`` enum of ``derivatives_sentiment`` (its ``tools/list`` schema, 2026-09-24)."""

_PERIOD_HOURS: Final[Mapping[str, float]] = {
    "5m": 5 / 60,
    "15m": 0.25,
    "30m": 0.5,
    "1h": 1,
    "2h": 2,
    "4h": 4,
    "6h": 6,
    "12h": 12,
    "1d": 24,
}

NEWS_FEEDS: Final = "cointelegraph,coindesk,decrypt,blockworks,cnbc"
"""The feed set ``news-briefing/SKILL.md`` uses for a topic query: the four core crypto outlets
plus CNBC for the US-equity side of the universe."""

REDDIT_FILTERS: Final = ("all-crypto", "all-stocks")
"""``reddit_trending`` filters. ``all-crypto`` is the tool's default; ``all-stocks`` covers the
eleven US-equity names. Both are ApeWisdom filter names, checked against its public API
(``https://apewisdom.io/api/v1.0/filter/all-stocks/page/1`` answered on 2026-09-24)."""

FEAR_GREED_MAX_AGE: Final = timedelta(hours=48)
"""The index is published once a day at 00:00 UTC; two days allows one missed publication."""

NEWS_SUMMARY_CHARS: Final = 600

_RATIO_KEYS: Final = ("longShortRatio", "long_short_ratio", "ratio")
_LONG_KEYS: Final = ("longAccount", "long_account")
_SHORT_KEYS: Final = ("shortAccount", "short_account")
_TAKER_KEYS: Final = ("buySellRatio", "buy_sell_ratio", "taker_ratio", "takerRatio")
_BUY_KEYS: Final = ("buyVol", "buy_vol")
_SELL_KEYS: Final = ("sellVol", "sell_vol")
_OI_KEYS: Final = ("sumOpenInterest", "open_interest", "openInterest")
_FNG_VALUE_KEYS: Final = ("value", "score")
_FNG_LABEL_KEYS: Final = ("value_classification", "classification", "rating", "label")
_NEWS_TIME_KEYS: Final = ("published", "pubDate", "published_at", "date", "updated", "time")
_NEWS_LINK_KEYS: Final = ("link", "url")
_NEWS_SUMMARY_KEYS: Final = ("summary", "description", "content")

_TAG: Final = re.compile(r"<[^>]+>")
_SPACE: Final = re.compile(r"\s+")
_SYMBOL: Final = re.compile(r"^[A-Z0-9]{2,30}$")


def max_age_for(period: str) -> timedelta:
    """How old the newest row of a ``period`` series may be: three periods, at least one hour.
    One late bucket is normal; three means the feed has stopped."""
    return timedelta(hours=max(1.0, 3 * _PERIOD_HOURS[period]))


def _check_period(period: str) -> None:
    if period not in PERIODS:
        raise ValueError(f"period {period!r} is not one of {sorted(PERIODS)}")


def _check_symbol(symbol: str) -> None:
    if not _SYMBOL.fullmatch(symbol):
        raise ValueError(f"symbol {symbol!r} is not an upper-case instrument code like BTCUSDT")


def _clean_text(value: Any) -> str:
    text = html.unescape(_TAG.sub(" ", str(value)))
    return _SPACE.sub(" ", text).strip()


def _news_time(value: Any) -> datetime | None:
    """RSS dates are RFC 2822, Atom dates ISO 8601. A date without a zone is refused: guessing it
    would misplace the item in time, and the coordination detector reads times."""
    moment = parse_instant(value)
    if moment is not None:
        return moment
    if isinstance(value, str) and value.strip():
        try:
            parsed = parsedate_to_datetime(value.strip())
        except (TypeError, ValueError, IndexError):
            return None
        if parsed.tzinfo is None:
            return None
        return parse_instant(parsed.isoformat())
    return None


class _Reading:
    """What one call produced, before it becomes a :class:`SourceCall`."""

    __slots__ = ("error", "health", "rows")

    def __init__(self, health: SourceHealth, rows: int = 0, error: str | None = None) -> None:
        self.health = health
        self.rows = rows
        self.error = error


class SignalSkills:
    """Typed readings from bitget-signal. No method raises for anything the server does.

    ``mcp`` is any :class:`~sentiment_agent.types.McpCaller`: the real
    :class:`~sentiment_agent.sources.mcp_http.StreamableHttpMcp` at
    :data:`~sentiment_agent.sources.mcp_http.SIGNAL_MCP_URL`, or a recorded stand-in in tests.
    """

    def __init__(self, mcp: McpCaller, clock: Clock) -> None:
        self._mcp = mcp
        self._clock = clock

    @property
    def server(self) -> str:
        return self._mcp.server

    @property
    def clock(self) -> Clock:
        return self._clock

    # --- calls ----------------------------------------------------------------------------------

    def _call(
        self, tool: str, arguments: Mapping[str, Any]
    ) -> tuple[Invocation, Any, _Reading | None]:
        """Invoke and decode. Returns the invocation, the decoded payload, and a finished reading
        when the call already failed (transport error, tool error, undecodable, empty, hollow)."""
        invocation = invoke(self._mcp, self._clock, tool, arguments)
        if invocation.failure is not None:
            return invocation, None, _Reading(invocation.failure, error=invocation.error)
        result = invocation.result
        if result is None:  # invoke() guarantees a result when there is no failure
            return invocation, None, _Reading(SourceHealth.ERROR, error="no result")
        if result.is_error:
            return (
                invocation,
                None,
                _Reading(SourceHealth.ERROR, error=f"tool reported an error: {result.text[:300]}"),
            )
        try:
            payload = tool_payload(result)
        except ValueError:
            return (
                invocation,
                None,
                _Reading(SourceHealth.ERROR, error=f"tool text is not JSON: {result.text[:200]!r}"),
            )
        if payload in (None, "", [], {}):
            return invocation, payload, _Reading(SourceHealth.EMPTY, error="answered with no data")
        envelope = error_envelope(payload)
        if envelope is not None:
            return invocation, payload, _Reading(envelope[0], error=envelope[1])
        if hollow(payload):
            return (
                invocation,
                payload,
                _Reading(
                    SourceHealth.HOLLOW,
                    rows=len(rows_of(payload)),
                    error="answered with a hollow payload (every value empty, zero or an error)",
                ),
            )
        return invocation, payload, None

    def _record(
        self,
        source: str,
        params: Mapping[str, Any],
        invocation: Invocation,
        reading: _Reading,
    ) -> SourceCall:
        return source_call(
            surface=SURFACE,
            source=source,
            params=params,
            invocation=invocation,
            health=reading.health,
            rows=reading.rows,
            error=reading.error,
        )

    # --- Fear & Greed ---------------------------------------------------------------------------

    def fear_greed(self) -> tuple[int | None, str | None, SourceCall]:
        """The crypto Fear & Greed index, 0-100, and its label."""
        arguments = {"action": "current"}
        invocation, payload, failed = self._call("sentiment_index", arguments)
        params: dict[str, Any] = {}
        if failed is not None:
            return None, None, self._record("sentiment_index.current", params, invocation, failed)
        rows = rows_of(payload)
        candidates = [
            row
            for row in rows
            if _fear_greed_value(first_present(row, _FNG_VALUE_KEYS)) is not None
        ]
        latest = newest(candidates)
        if latest is None:
            reading = _Reading(
                SourceHealth.HOLLOW,
                rows=len(rows),
                error=_unrecognised(rows, "a timestamped 0-100 value", candidates),
            )
            return None, None, self._record("sentiment_index.current", params, invocation, reading)
        row, moment = latest
        stale = freshness_problem(moment, self._clock.now(), FEAR_GREED_MAX_AGE)
        if stale is not None:
            reading = _Reading(SourceHealth.HOLLOW, rows=len(rows), error=stale)
            return None, None, self._record("sentiment_index.current", params, invocation, reading)
        value = _fear_greed_value(first_present(row, _FNG_VALUE_KEYS))
        label = first_present(row, _FNG_LABEL_KEYS)
        text = str(label).strip() if label is not None and str(label).strip() else None
        call = self._record(
            "sentiment_index.current", params, invocation, _Reading(SourceHealth.OK, len(rows))
        )
        return value, text, call

    # --- derivatives positioning ----------------------------------------------------------------

    def _ratio(
        self,
        action: str,
        symbol: str,
        period: str,
        value_of: Callable[[Mapping[str, Any]], float | None],
    ) -> tuple[float | None, SourceCall]:
        _check_symbol(symbol)
        _check_period(period)
        source = f"derivatives_sentiment.{action}"
        params = {"symbol": symbol, "period": period}
        invocation, payload, failed = self._call(
            "derivatives_sentiment", {"action": action, **params}
        )
        if failed is not None:
            return None, self._record(source, params, invocation, failed)
        rows = rows_of(payload)
        valued = [row for row in rows if value_of(row) is not None]
        latest = newest(valued)
        if latest is None:
            reading = _Reading(
                SourceHealth.HOLLOW, rows=len(rows), error=_unrecognised(rows, "a ratio", valued)
            )
            return None, self._record(source, params, invocation, reading)
        row, moment = latest
        stale = freshness_problem(moment, self._clock.now(), max_age_for(period))
        if stale is not None:
            reading = _Reading(SourceHealth.HOLLOW, rows=len(rows), error=stale)
            return None, self._record(source, params, invocation, reading)
        return value_of(row), self._record(
            source, params, invocation, _Reading(SourceHealth.OK, len(rows))
        )

    def long_short(self, symbol: str, period: str = "4h") -> tuple[float | None, SourceCall]:
        """Retail (all-account) long/short ratio: long accounts over short accounts."""
        return self._ratio("long_short", symbol, period, _long_short_ratio)

    def top_long_short(self, symbol: str, period: str = "4h") -> tuple[float | None, SourceCall]:
        """Top-trader long/short account ratio (``top_ls``)."""
        return self._ratio("top_ls", symbol, period, _long_short_ratio)

    def taker_ratio(self, symbol: str, period: str = "4h") -> tuple[float | None, SourceCall]:
        """Taker buy volume over taker sell volume."""
        return self._ratio("taker_ratio", symbol, period, _taker_ratio)

    def open_interest(
        self, symbol: str, period: str = "1h"
    ) -> tuple[tuple[tuple[datetime, float], ...], SourceCall]:
        """Open-interest history, oldest first, one point per timestamp."""
        _check_symbol(symbol)
        _check_period(period)
        source = "derivatives_sentiment.open_interest"
        params = {"symbol": symbol, "period": period}
        invocation, payload, failed = self._call(
            "derivatives_sentiment", {"action": "open_interest", **params}
        )
        if failed is not None:
            return (), self._record(source, params, invocation, failed)
        rows = rows_of(payload)
        points = _series(rows, _OI_KEYS)
        if not points:
            reading = _Reading(
                SourceHealth.HOLLOW,
                rows=len(rows),
                error=_unrecognised(rows, "a timestamped positive open interest", []),
            )
            return (), self._record(source, params, invocation, reading)
        stale = freshness_problem(points[-1][0], self._clock.now(), max_age_for(period))
        if stale is not None:
            reading = _Reading(SourceHealth.HOLLOW, rows=len(rows), error=stale)
            return (), self._record(source, params, invocation, reading)
        call = self._record(source, params, invocation, _Reading(SourceHealth.OK, len(rows)))
        return points, call

    # --- crowd text -----------------------------------------------------------------------------

    def reddit_trending(
        self, limit: int, *, subreddit_filter: str = "all-crypto"
    ) -> tuple[tuple[TextItem, ...], SourceCall]:
        """The most-mentioned tickers on Reddit, one :class:`TextItem` per ticker.

        A ranking is an observation made now, not a post, so ``published_at`` is the fetch time and
        the item id carries the hour: the same ticker ranked in the next hour is a new observation.
        """
        if not 1 <= limit <= 50:
            raise ValueError("reddit_trending limit must be within 1..50 (the tool's range)")
        if not subreddit_filter.strip():
            raise ValueError("subreddit_filter must name an ApeWisdom filter")
        source = "derivatives_sentiment.reddit_trending"
        params = {"limit": limit, "filter": subreddit_filter}
        invocation, payload, failed = self._call(
            "derivatives_sentiment", {"action": "reddit_trending", **params}
        )
        if failed is not None:
            return (), self._record(source, params, invocation, failed)
        rows = rows_of(payload)
        fetched = self._clock.now()
        items: list[TextItem] = []
        for row in rows:
            item = _reddit_item(row, subreddit_filter, fetched)
            if item is not None:
                items.append(item)
        if not items:
            reading = _Reading(
                SourceHealth.HOLLOW,
                rows=len(rows),
                error=_unrecognised(rows, "a ticker with a mention count", []),
            )
            return (), self._record(source, params, invocation, reading)
        items = items[:limit]
        call = self._record(source, params, invocation, _Reading(SourceHealth.OK, len(rows)))
        return tuple(items), call

    def news(
        self, limit: int, keyword: str | None = None
    ) -> tuple[tuple[TextItem, ...], SourceCall]:
        """Latest headlines across :data:`NEWS_FEEDS`, newest first, at most ``limit`` in total.

        The tool's own ``limit`` is per feed (1-10), so ``min(limit, 10)`` is requested from each
        feed and the merged list is cut to ``limit``. Items without a zoned publication time are
        dropped and counted in the call's note: an undated story placed at the fetch time would look
        simultaneous with every other undated story, and simultaneity is what the coordination
        detector looks for.
        """
        if limit < 1:
            raise ValueError("news limit must be at least 1")
        source = "news_feed.latest"
        params: dict[str, Any] = {"feeds": NEWS_FEEDS, "limit": min(limit, 10)}
        if keyword is not None and keyword.strip():
            params["keyword"] = keyword.strip()
        invocation, payload, failed = self._call("news_feed", {"action": "latest", **params})
        if failed is not None:
            return (), self._record(source, params, invocation, failed)
        fetched = self._clock.now()
        feeds = (
            [f for f in iter_dicts(payload) if "items" in f] if isinstance(payload, list) else []
        )
        if not feeds and isinstance(payload, dict):
            feeds = [payload] if "items" in payload else []
        # news-briefing/SKILL.md: 'Failed feeds return {"feed": "...", "error": "..."}'.
        failed_feeds = [
            str(f.get("feed", "?")) for f in feeds if "error" in f and not f.get("items")
        ]
        items: dict[str, TextItem] = {}
        undated = 0
        raw_items = 0
        for feed in feeds:
            outlet = str(feed.get("feed") or feed.get("source") or "news")
            for entry in iter_dicts(feed.get("items")):
                raw_items += 1
                item = _news_item(entry, outlet, fetched)
                if item is None:
                    undated += 1
                    continue
                items.setdefault(item.item_id, item)
        ordered = sorted(items.values(), key=lambda i: (i.published_at, i.item_id), reverse=True)
        notes: list[str] = []
        if failed_feeds:
            notes.append(f"{len(failed_feeds)} of {len(feeds)} feeds reported an error")
        if undated:
            notes.append(f"{undated} item(s) without a title or a zoned date dropped")
        note = "; ".join(notes) or None
        if not ordered:
            what = "no feed carried a dated headline" if feeds else "no feed list in the reply"
            reading = _Reading(SourceHealth.HOLLOW, rows=raw_items, error=note or what)
            return (), self._record(source, params, invocation, reading)
        call = self._record(source, params, invocation, _Reading(SourceHealth.OK, raw_items, note))
        return tuple(ordered[:limit]), call

    # --- probe ----------------------------------------------------------------------------------

    def probe(self, tool: str, arguments: Mapping[str, Any]) -> SourceCall:
        """Call any tool once and classify only whether it answered with substance. Used for the
        toolkit coverage matrix; the result is not interpreted beyond OK/EMPTY/HOLLOW/failure."""
        action = arguments.get("action")
        source = f"{tool}.{action}" if isinstance(action, str) else tool
        params = {k: v for k, v in arguments.items() if k != "action"}
        invocation, payload, failed = self._call(tool, arguments)
        if failed is not None:
            return self._record(source, params, invocation, failed)
        rows = rows_of(payload)
        return self._record(source, params, invocation, _Reading(SourceHealth.OK, len(rows)))


# ================================================================================================
# Row interpretation
# ================================================================================================


def _fear_greed_value(value: Any) -> int | None:
    """A Fear & Greed value on the pinned 0-100 scale, rounded half up; ``None`` outside it."""
    number = finite(value)
    if number is None or not 0 <= number <= 100:
        return None
    return int(number + 0.5)


def _long_short_ratio(row: Mapping[str, Any]) -> float | None:
    ratio = finite(first_present(row, _RATIO_KEYS))
    if ratio is None:
        longs = finite(first_present(row, _LONG_KEYS))
        shorts = finite(first_present(row, _SHORT_KEYS))
        if longs is not None and shorts is not None and shorts > 0 and longs >= 0:
            ratio = longs / shorts
    return ratio if ratio is not None and ratio > 0 else None


def _taker_ratio(row: Mapping[str, Any]) -> float | None:
    ratio = finite(first_present(row, _TAKER_KEYS))
    if ratio is None:
        buys = finite(first_present(row, _BUY_KEYS))
        sells = finite(first_present(row, _SELL_KEYS))
        if buys is not None and sells is not None and sells > 0 and buys >= 0:
            ratio = buys / sells
    return ratio if ratio is not None and ratio > 0 else None


def _series(
    rows: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> tuple[tuple[datetime, float], ...]:
    """``(time, value)`` points, oldest first; the last row wins for a repeated timestamp; rows
    without a time or a positive finite value are skipped."""
    points: dict[datetime, float] = {}
    for row in rows:
        moment = row_time(row)
        value = finite(first_present(row, keys))
        if moment is not None and value is not None and value > 0:
            points[moment] = value
    return tuple(sorted(points.items()))


def _unrecognised(
    rows: Sequence[Mapping[str, Any]], wanted: str, matched: Sequence[Mapping[str, Any]]
) -> str:
    if not rows:
        return f"no rows where {wanted} was expected"
    keys = sorted({str(k) for row in rows for k in row})[:12]
    if matched:
        return f"rows carry {wanted} but none has a parseable timestamp; keys: {keys}"
    return f"reply shape not recognised (NOT VERIFIED success shape): no {wanted}; keys: {keys}"


def _reddit_item(
    row: Mapping[str, Any], subreddit_filter: str, fetched: datetime
) -> TextItem | None:
    ticker_raw = row.get("ticker")
    mentions = finite(row.get("mentions"))
    if not isinstance(ticker_raw, str) or not ticker_raw.strip() or mentions is None:
        return None
    ticker = ticker_raw.strip().upper()
    base = ticker.removesuffix(".X")
    name = _clean_text(row.get("name") or base)
    parts = [f"Reddit ({subreddit_filter}) trending: {base} ({name}), {int(mentions)} mentions"]
    before = finite(row.get("mentions_24h_ago"))
    if before is not None:
        parts.append(f"{int(before)} the day before")
    rank = finite(row.get("rank"))
    if rank is not None:
        rank_text = f"rank {int(rank)}"
        rank_before = finite(row.get("rank_24h_ago"))
        if rank_before is not None and rank_before > 0:
            rank_text += f" (was {int(rank_before)})"
        parts.append(rank_text)
    upvotes = finite(row.get("upvotes"))
    if upvotes is not None:
        parts.append(f"{int(upvotes)} upvotes")
    hour = fetched.replace(minute=0, second=0, microsecond=0).isoformat()
    item_id = "reddit-" + content_hash([subreddit_filter, ticker, hour])[:20]
    return TextItem(
        item_id=item_id,
        channel="reddit",
        source=f"reddit/{subreddit_filter}",
        url=None,
        published_at=fetched,
        fetched_at=fetched,
        text=", ".join(parts) + ".",
        symbols=(f"{base}USDT",) if _SYMBOL.fullmatch(f"{base}USDT") else (),
    )


def _news_item(entry: Mapping[str, Any], outlet: str, fetched: datetime) -> TextItem | None:
    title = _clean_text(entry.get("title") or "")
    published = None
    for key in _NEWS_TIME_KEYS:
        if key in entry:
            published = _news_time(entry[key])
            if published is not None:
                break
    if not title or published is None:
        return None
    link = first_present(entry, _NEWS_LINK_KEYS)
    url = str(link).strip() if isinstance(link, str) and link.startswith("http") else None
    summary = _clean_text(first_present(entry, _NEWS_SUMMARY_KEYS) or "")
    if len(summary) > NEWS_SUMMARY_CHARS:
        summary = summary[: NEWS_SUMMARY_CHARS - 3].rstrip() + "..."
    text = f"{title} — {summary}" if summary and summary != title else title
    item_id = "news-" + content_hash([outlet, url or title, published.isoformat()])[:20]
    return TextItem(
        item_id=item_id,
        channel="news",
        source=outlet,
        url=url,
        published_at=published,
        fetched_at=fetched,
        text=text,
    )


__all__ = [
    "FEAR_GREED_MAX_AGE",
    "NEWS_FEEDS",
    "PERIODS",
    "REDDIT_FILTERS",
    "SURFACE",
    "SignalSkills",
    "max_age_for",
]
