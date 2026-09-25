"""The sources Bitget's two data services wrap, read directly when neither service answers them.

**Why this exists: run 1 went blind.** From bitget-mcp-server's first failing call
(2026-09-25 08:33 UTC) to seq 546 of run 1's ledger, that service answered 144 of 1,356 calls
(every ``do_query`` a 503 from its own upstream after 10:14), and bitget-signal answered none of
the 939 calls it was asked over the whole run. Crypto Fear & Greed and BTCUSDT's crowd positioning
were missing from 146 of those 164 snapshots, so the Fear & Greed trigger could not fire and the
model saw no positioning; the figures are recomputed from run 1's published ledger by
``scripts/feed_outage.py`` (``validation/run2/run1_feed_outage.json``). The upstreams those
services name answered directly from the same machine. Run 2 declares this module as ``run2-d3``
(``sentiment_agent.run2``).

**The same numbers, not substitutes.** Each reading here is taken from the source the Bitget service
itself reads, so a fallback value describes the same thing the primary would have:

* retail long/short: Binance USD-M ``futures/data/globalLongShortAccountRatio``,
  ``longShortRatio``;
* top-trader account and position ratios: ``futures/data/topLongShortAccountRatio`` and
  ``topLongShortPositionRatio``, ``longShortRatio``;
* taker buy/sell: ``futures/data/takerlongshortRatio``, ``buySellRatio``;
* open interest (the series): ``futures/data/openInterestHist``, ``sumOpenInterest``;
* funding rate: ``fapi/v1/fundingRate``, ``fundingRate``;
* crypto Fear & Greed: alternative.me ``fng``, ``value``;
* news: the publishers' own RSS feeds (:data:`NEWS_FEEDS`), the item title.

bitget-mcp-server reads Binance (``bitget_data.EXCHANGE``) and returns open interest in contracts,
which is ``sumOpenInterest`` (not the dollar ``sumOpenInterestValue``); bitget-signal's tool
descriptions name Binance futures and alternative.me. Binance's funding rate is already a fraction,
where the Bitget service reports a percent (``bitget_data.FUNDING_PERCENT``). The news feeds are the
exception: bitget-signal aggregates 44 feeds it does not list, so the four publishers here are a
smaller, named set, and the source of every item says which one it came from.

**Labelled, never merged into the Bitget count.** Every call is a
:class:`~sentiment_agent.types.SourceCall` on :attr:`~sentiment_agent.types.ToolkitSurface.
UPSTREAM_DIRECT`, whose source names the upstream (``upstream:binance.openInterestHist``). The
Bitget services' own calls are still made first and still recorded, so their failure stays a
feed-health alarm and the toolkit coverage matrix still counts only what Bitget answered. A field
is read here only when both Bitget services left it empty, and a series is taken whole from one
source, never spliced.

Every value must be current (the same age limits the Bitget readers apply) and every call returns
its record without raising; a failure is a recorded ``error`` or ``hollow`` call, never a quiet
market. The raw reply of each call is kept as a blob when a store is given.
"""

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any, Final

from sentiment_agent.hashing import canonical_json
from sentiment_agent.sources.mcp_http import finite, newest, parse_instant, row_time
from sentiment_agent.types import (
    BlobRef,
    BlobStore,
    Clock,
    SourceCall,
    SourceHealth,
    TextItem,
    ToolkitSurface,
)

BINANCE: Final = "https://fapi.binance.com/"
FEAR_GREED_URL: Final = "https://api.alternative.me/fng/?limit=2"
NEWS_FEEDS: Final[tuple[tuple[str, str], ...]] = (
    ("coindesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("cointelegraph", "https://cointelegraph.com/rss"),
    ("decrypt", "https://decrypt.co/feed"),
    ("cnbc", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
)
"""The publishers read when bitget-signal's news feed is dark (ARGUS ``market/skill_mirror.py``
reads the same four)."""

PERIOD: Final = "1h"
RATIO_ROWS: Final = 3
OPEN_INTEREST_ROWS: Final = 48
FUNDING_ROWS: Final = 6
HOURLY_MAX_AGE: Final = timedelta(hours=3)
FUNDING_MAX_AGE: Final = timedelta(hours=12)
FEAR_GREED_MAX_AGE: Final = timedelta(hours=48)
"""alternative.me publishes once a day; a reading older than two days is stale."""
NEWS_MAX_AGE: Final = timedelta(hours=24)
TIMEOUT_S: Final = 15.0
USER_AGENT: Final = "Mozilla/5.0 (t2-sentiment-agent)"
MEDIA_TYPE: Final = "application/json"
RSS_MEDIA_TYPE: Final = "application/rss+xml"

_SURFACE: Final = ToolkitSurface.UPSTREAM_DIRECT

Fetch = Callable[[str], bytes]
"""GET a URL and return its body; raise :class:`UpstreamError` on any failure."""


class UpstreamError(RuntimeError):
    """An upstream did not answer; carries its own reason."""


def urllib_get(url: str) -> bytes:
    if not url.startswith("https://"):
        raise UpstreamError(f"refused a non-https URL: {url[:60]}")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310 - https checked above
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:  # noqa: S310 - https checked above
            return bytes(response.read())
    except urllib.error.HTTPError as exc:
        raise UpstreamError(f"HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise UpstreamError(f"{type(exc).__name__}: {exc}") from exc


class UpstreamDirect:
    """The upstreams behind bitget-mcp-server and bitget-signal, read with no key."""

    def __init__(
        self,
        clock: Clock,
        *,
        fetch: Fetch = urllib_get,
        blobs: BlobStore | None = None,
    ) -> None:
        self._clock = clock
        self._fetch = fetch
        self._blobs = blobs

    # ---- plumbing ----------------------------------------------------------------------------

    def _get(
        self, name: str, url: str, params: Mapping[str, Any], media_type: str = MEDIA_TYPE
    ) -> tuple[bytes | None, SourceCall]:
        started = self._clock.now()
        t0 = time.perf_counter()
        try:
            body = self._fetch(url)
        except Exception as exc:
            return None, self._record(name, params, started, t0, SourceHealth.ERROR, None, str(exc))
        blob = self._blobs.put(body, media_type) if self._blobs is not None else None
        return body, self._record(name, params, started, t0, SourceHealth.OK, blob, None)

    def _record(
        self,
        name: str,
        params: Mapping[str, Any],
        started: datetime,
        t0: float,
        health: SourceHealth,
        blob: BlobRef | None,
        error: str | None,
        rows: int = 0,
    ) -> SourceCall:
        return SourceCall(
            call_id=f"{_SURFACE.value}-{uuid.uuid4().hex}",
            surface=_SURFACE,
            source=f"upstream:{name}",
            params={str(k): str(v) for k, v in params.items()},
            health=health,
            started_at=started,
            latency_ms=max(0, int((time.perf_counter() - t0) * 1000)),
            rows=rows,
            blob=blob,
            error=None if error is None else error[:300],
        )

    @staticmethod
    def _settle(
        call: SourceCall, health: SourceHealth, rows: int, error: str | None = None
    ) -> SourceCall:
        return call.model_copy(update={"health": health, "rows": rows, "error": error})

    def _rows(
        self, name: str, url: str, params: Mapping[str, Any]
    ) -> tuple[list[Mapping[str, Any]], SourceCall]:
        body, call = self._get(name, url, params)
        if body is None:
            return [], call
        try:
            loaded = json.loads(body.decode("utf-8"))
        except ValueError:
            return [], self._settle(call, SourceHealth.HOLLOW, 0, "reply is not JSON")
        if not isinstance(loaded, list):
            keys = sorted(loaded)[:6] if isinstance(loaded, dict) else type(loaded).__name__
            return [], self._settle(call, SourceHealth.HOLLOW, 0, f"not a list of rows: {keys}")
        rows = [r for r in loaded if isinstance(r, Mapping)]
        return rows, call

    def _latest(
        self,
        name: str,
        path: str,
        symbol: str,
        value_of: Callable[[Mapping[str, Any]], float | None],
        limit: int,
        max_age: timedelta,
    ) -> tuple[float | None, SourceCall]:
        params = {"symbol": symbol, "period": PERIOD, "limit": limit}
        rows, call = self._rows(name, f"{BINANCE}{path}?{urllib.parse.urlencode(params)}", params)
        if call.health is not SourceHealth.OK:
            return None, call
        valued = [r for r in rows if value_of(r) is not None]
        latest = newest(valued)
        if latest is None:
            health = SourceHealth.EMPTY if not rows else SourceHealth.HOLLOW
            return None, self._settle(call, health, len(rows), "no usable row")
        row, moment = latest
        if self._clock.now() - moment > max_age:
            return None, self._settle(
                call, SourceHealth.HOLLOW, len(rows), f"stale: newest row {moment.isoformat()}"
            )
        return value_of(row), self._settle(call, SourceHealth.OK, len(rows))

    # ---- readings -----------------------------------------------------------------------------

    def long_short(self, symbol: str) -> tuple[float | None, SourceCall]:
        return self._latest(
            "binance.globalLongShortAccountRatio",
            "futures/data/globalLongShortAccountRatio",
            symbol,
            _ratio,
            RATIO_ROWS,
            HOURLY_MAX_AGE,
        )

    def top_account(self, symbol: str) -> tuple[float | None, SourceCall]:
        return self._latest(
            "binance.topLongShortAccountRatio",
            "futures/data/topLongShortAccountRatio",
            symbol,
            _ratio,
            RATIO_ROWS,
            HOURLY_MAX_AGE,
        )

    def top_position(self, symbol: str) -> tuple[float | None, SourceCall]:
        return self._latest(
            "binance.topLongShortPositionRatio",
            "futures/data/topLongShortPositionRatio",
            symbol,
            _ratio,
            RATIO_ROWS,
            HOURLY_MAX_AGE,
        )

    def taker(self, symbol: str) -> tuple[float | None, SourceCall]:
        return self._latest(
            "binance.takerlongshortRatio",
            "futures/data/takerlongshortRatio",
            symbol,
            _taker,
            RATIO_ROWS,
            HOURLY_MAX_AGE,
        )

    def open_interest(self, symbol: str) -> tuple[tuple[tuple[datetime, float], ...], SourceCall]:
        """Hourly open interest in contracts, the whole series from this one source."""
        params = {"symbol": symbol, "period": PERIOD, "limit": OPEN_INTEREST_ROWS}
        url = f"{BINANCE}futures/data/openInterestHist?{urllib.parse.urlencode(params)}"
        rows, call = self._rows("binance.openInterestHist", url, params)
        if call.health is not SourceHealth.OK:
            return (), call
        points: dict[datetime, float] = {}
        for row in rows:
            moment = row_time(row)
            value = finite(row.get("sumOpenInterest"))
            if moment is not None and value is not None and value > 0:
                points[moment] = value
        series = tuple(sorted(points.items()))
        if not series:
            return (), self._settle(call, SourceHealth.HOLLOW, len(rows), "no timestamped rows")
        if self._clock.now() - series[-1][0] > HOURLY_MAX_AGE:
            return (), self._settle(
                call, SourceHealth.HOLLOW, len(rows), f"stale: newest {series[-1][0].isoformat()}"
            )
        return series, self._settle(call, SourceHealth.OK, len(rows))

    def funding(self, symbol: str) -> tuple[float | None, SourceCall]:
        params = {"symbol": symbol, "limit": FUNDING_ROWS}
        url = f"{BINANCE}fapi/v1/fundingRate?{urllib.parse.urlencode(params)}"
        rows, call = self._rows("binance.fundingRate", url, params)
        if call.health is not SourceHealth.OK:
            return None, call
        dated = [({"time": r.get("fundingTime"), **r}) for r in rows]
        latest = newest([r for r in dated if finite(r.get("fundingRate")) is not None])
        if latest is None:
            return None, self._settle(call, SourceHealth.HOLLOW, len(rows), "no usable row")
        row, moment = latest
        if self._clock.now() - moment > FUNDING_MAX_AGE:
            return None, self._settle(
                call, SourceHealth.HOLLOW, len(rows), f"stale: newest row {moment.isoformat()}"
            )
        return finite(row.get("fundingRate")), self._settle(call, SourceHealth.OK, len(rows))

    def fear_greed(self) -> tuple[int | None, str | None, SourceCall]:
        params = {"limit": 2}
        body, call = self._get("alternative_me.fng", FEAR_GREED_URL, params)
        if body is None:
            return None, None, call
        try:
            data = json.loads(body.decode("utf-8")).get("data") or []
        except (ValueError, AttributeError):
            return None, None, self._settle(call, SourceHealth.HOLLOW, 0, "reply is not the index")
        rows = [r for r in data if isinstance(r, Mapping)]
        latest = newest(rows)
        if latest is None:
            return None, None, self._settle(call, SourceHealth.HOLLOW, len(rows), "no dated value")
        row, moment = latest
        value = finite(row.get("value"))
        if value is None or not 0 <= value <= 100:
            return None, None, self._settle(call, SourceHealth.HOLLOW, len(rows), "no value")
        if self._clock.now() - moment > FEAR_GREED_MAX_AGE:
            return (
                None,
                None,
                self._settle(call, SourceHealth.HOLLOW, len(rows), f"stale: {moment.isoformat()}"),
            )
        label = row.get("value_classification")
        return (
            round(value),
            str(label) if label else None,
            self._settle(call, SourceHealth.OK, len(rows)),
        )

    def news(self, limit: int) -> tuple[tuple[TextItem, ...], tuple[SourceCall, ...]]:
        """The newest ``limit`` headlines across :data:`NEWS_FEEDS`, one call per feed."""
        now = self._clock.now()
        items: list[TextItem] = []
        calls: list[SourceCall] = []
        for outlet, url in NEWS_FEEDS:
            body, call = self._get(f"rss.{outlet}", url, {"feed": outlet}, RSS_MEDIA_TYPE)
            if body is None:
                calls.append(call)
                continue
            try:
                found = _rss_items(outlet, body, fetched_at=now)
            except ET.ParseError as exc:
                calls.append(self._settle(call, SourceHealth.HOLLOW, 0, f"not a feed: {exc}"))
                continue
            fresh = [i for i in found if now - i.published_at <= NEWS_MAX_AGE]
            health = SourceHealth.OK if fresh else SourceHealth.EMPTY
            calls.append(self._settle(call, health, len(fresh)))
            items.extend(fresh)
        items.sort(key=lambda i: i.published_at, reverse=True)
        return tuple(items[:limit]), tuple(calls)


def _ratio(row: Mapping[str, Any]) -> float | None:
    ratio = finite(row.get("longShortRatio"))
    if ratio is None:
        longs, shorts = finite(row.get("longAccount")), finite(row.get("shortAccount"))
        if longs is not None and shorts is not None and shorts > 0 and longs >= 0:
            ratio = longs / shorts
    return ratio if ratio is not None and ratio > 0 else None


def _taker(row: Mapping[str, Any]) -> float | None:
    ratio = finite(row.get("buySellRatio"))
    if ratio is None:
        buys, sells = finite(row.get("buyVol")), finite(row.get("sellVol"))
        if buys is not None and sells is not None and sells > 0 and buys >= 0:
            ratio = buys / sells
    return ratio if ratio is not None and ratio > 0 else None


def _when(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        moment = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return parse_instant(text)
    return moment.astimezone(UTC) if moment.tzinfo else moment.replace(tzinfo=UTC)


def _rss_items(outlet: str, body: bytes, *, fetched_at: datetime) -> list[TextItem]:
    """RSS ``<item>`` or Atom ``<entry>`` elements as news :class:`TextItem` s (title only: the
    title is what a headline cluster reads, and summaries carry publisher markup)."""
    # Publisher feeds are untrusted. ElementTree never resolves external entities, and the expat
    # bundled with Python 3.11 caps entity amplification, so the two XML attacks S314 names are
    # closed without a dependency.
    root = ET.fromstring(body)  # noqa: S314 - see above
    out: list[TextItem] = []
    entries = root.iter()
    for node in entries:
        tag = node.tag.rsplit("}", 1)[-1]
        if tag not in ("item", "entry"):
            continue
        fields = {child.tag.rsplit("}", 1)[-1]: child for child in node}
        title = (fields["title"].text or "").strip() if "title" in fields else ""
        link_node = fields.get("link")
        link = ""
        if link_node is not None:
            link = (link_node.text or link_node.get("href") or "").strip()
        stamp = None
        for key in ("pubDate", "published", "updated", "date"):
            if key in fields:
                stamp = _when(fields[key].text)
                if stamp is not None:
                    break
        if not title or stamp is None:
            continue
        digest = hashlib.sha256(canonical_json({"outlet": outlet, "link": link, "title": title}))
        out.append(
            TextItem(
                item_id=f"news-{outlet}-{digest.hexdigest()[:16]}",
                channel="news",
                source=outlet,
                url=link or None,
                published_at=stamp,
                fetched_at=fetched_at,
                text=title,
            )
        )
    return out


__all__ = ["NEWS_FEEDS", "UpstreamDirect", "UpstreamError", "urllib_get"]
