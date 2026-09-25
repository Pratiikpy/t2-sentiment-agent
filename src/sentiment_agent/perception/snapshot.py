"""The perception snapshot: one complete, logged picture of the world at decision time.

:class:`SnapshotBuilder` asks every source once, in a fixed order, and assembles a
:class:`~sentiment_agent.types.PerceptionSnapshot`: Demo and live quotes, per-instrument
:class:`~sentiment_agent.types.PositioningFeatures`, the market mood, screened crowd text and its
coordination report, the calendar, every source call, and the flat ``facts`` map. The snapshot is
logged in full before any decision reads it (DESIGN.md §7), so every baseline, rival arm and
counterfactual can be recomputed from the log without calling anything again.

**Nothing a source does can crash a snapshot.** A market call that raises, a toolkit reader that
raises in breach of its contract, a crowd collector with no login: each becomes a
:class:`~sentiment_agent.types.SourceCall` with health ``ERROR`` or ``TIMEOUT``, and every value it
would have supplied is ``None``. Every source failing at once still yields a valid snapshot whose
coverage says so. Our own code is different: a defect in quarantine or clustering raises, because a
silently degraded defence is worse than a loud one.

**Source calls.** Market data is recorded twice over, on purpose. Each request the builder makes is
one *logical* call (``public_v3.tickers[demo]``,
``public_v3.history_candles[live,market,1H]:NVDAUSDT``) whose health is the outcome the snapshot
actually got, uniform across market implementations. When the market object also keeps a transport
log (``BitgetPublicApi.drain_calls``), those per-HTTP-page calls follow their logical call and carry
the raw response blobs. Calls already sitting in that log when a build starts belong to someone else
(the protective loop, say) and are drained and dropped, so a snapshot's calls are exactly the ones
it made. Toolkit and crowd calls are recorded as their
readers report them.

**Live and Demo are never mixed (DESIGN.md §6.1).** Every returned quote, candle and funding point
is checked against the environment, symbol and kind it was requested for, and anything else is
dropped before a feature can read it.

**Light mode** (every 5 minutes, for trigger evaluation) skips crowd text, news and the calendar.
Those sources appear in the coverage as ``DISABLED`` with the reason, so "not asked" is never
confused with "asked and silent", and the features they feed are ``None``.

``snapshot_id`` is :func:`seal`: the content hash of the snapshot with the id blank, so the id
proves the content and a changed snapshot no longer matches its id.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Final, Protocol, TypeVar, runtime_checkable
from zoneinfo import ZoneInfo

from sentiment_agent.crowd.novelty import build_report
from sentiment_agent.crowd.quarantine import screen
from sentiment_agent.hashing import content_hash
from sentiment_agent.perception.features import (
    SOCIAL_CHANNELS,
    STALE_INDEX_MOVES,
    build_features,
    coverage_facts,
    crowd_facts,
    facts_from,
    mood_from,
    social_signals,
)
from sentiment_agent.types import (
    AssetClass,
    BookState,
    CalendarItem,
    Candle,
    CandleKind,
    Clock,
    CrowdCollector,
    CrowdReport,
    DerivativesReading,
    FundingPoint,
    MarketData,
    MoodReading,
    PerceptionSnapshot,
    Policy,
    PositioningFeatures,
    PriceSource,
    Quote,
    RunMode,
    ScreenedItem,
    SourceCall,
    SourceHealth,
    TextItem,
    ToolkitReader,
    ToolkitSurface,
)

CANDLE_INTERVAL: Final = "1H"

LIVE_CANDLE_HOURS: Final = 30
"""Window of live 1H candles fetched for the overheating feature: 20 bars for the mean, 15 for the
14-bar ATR, plus room for a missing bar or two."""

TEXT_LOOKBACK: Final = timedelta(hours=24)
"""Crowd text older than this is not shown or counted: ``social_mentions_24h`` is a 24-hour count.
Items dropped here remain in the source calls' raw blobs."""

TEXT_CLOCK_SKEW: Final = timedelta(minutes=5)
"""A post stamped up to this far after the snapshot is kept (a source clock running fast); further
ahead than that is a malformed timestamp and the item is dropped."""

CALENDAR_LOOKBACK: Final = timedelta(hours=48)
"""How far back filings are read, so consecutive full snapshots overlap and a new Form 4 or 8-K is
never missed between them (full snapshots run at least every 8h on the funding heartbeat)."""

NEW_YORK: Final = ZoneInfo("America/New_York")

NEWS_LIMIT: Final = 30
REDDIT_LIMIT: Final = 30

_ANSWERED: Final = frozenset({SourceHealth.OK, SourceHealth.EMPTY})

LIGHT_SNAPSHOT_REASON: Final = (
    "light snapshot: crowd text, news and calendar are read on full snapshots"
)
"""The reason a light snapshot records on the sources it does not ask (:func:`is_light`)."""

_UNMEASURED_CROWD: Final = CrowdReport(
    items=0, withheld=0, distinct_stories=0, duplication_ratio=0.0, clusters=(), mentions={}
)
"""A report over text that was not read. ``mentions`` is empty (not zero per symbol), which is what
keeps the social features ``None`` rather than a measured zero."""

T = TypeVar("T")


@runtime_checkable
class _TransportLog(Protocol):
    """Implemented by ``venue.public_api.BitgetPublicApi``: the per-HTTP-call log with raw blobs."""

    def drain_calls(self) -> tuple[SourceCall, ...]: ...


def _timed_out(exc: BaseException) -> bool:
    """Whether ``exc``, or anything it wraps, is a timeout (``urllib`` nests them in ``reason``)."""
    seen: set[int] = set()
    stack: list[BaseException] = [exc]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, TimeoutError):
            return True
        for linked in (current.__cause__, current.__context__, getattr(current, "reason", None)):
            if isinstance(linked, BaseException):
                stack.append(linked)
    return False


def _error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:500]


@dataclass(frozen=True, slots=True)
class _EarlyText:
    """A full snapshot's text sources, started on worker threads when the build begins."""

    news: Future[tuple[tuple[TextItem, ...], tuple[SourceCall, ...]]]
    reddit: Future[tuple[tuple[TextItem, ...], tuple[SourceCall, ...]]]
    crowd: Future[tuple[tuple[TextItem, ...], tuple[SourceCall, ...]]]
    calendar: Future[tuple[tuple[CalendarItem, ...], tuple[SourceCall, ...]]]


class _CallLog:
    """Every source call one build makes, in the order it made them."""

    def __init__(self, clock: Clock, taken_at: datetime) -> None:
        self._clock = clock
        self._taken_at = taken_at
        self.calls: list[SourceCall] = []

    def _call_id(self, source: str, params: Mapping[str, str]) -> str:
        seed = {
            "taken_at": self._taken_at,
            "n": len(self.calls),
            "source": source,
            "params": dict(params),
        }
        return f"pc-{len(self.calls):03d}-{content_hash(seed)[:12]}"

    def record(
        self,
        *,
        surface: ToolkitSurface,
        source: str,
        params: Mapping[str, str],
        health: SourceHealth,
        started_at: datetime,
        rows: int = 0,
        error: str | None = None,
    ) -> None:
        latency = max(0, int((self._clock.now() - started_at).total_seconds() * 1000))
        self.calls.append(
            SourceCall(
                call_id=self._call_id(source, params),
                surface=surface,
                source=source,
                params=dict(params),
                health=health,
                started_at=started_at,
                latency_ms=latency,
                rows=rows,
                blob=None,
                error=error,
            )
        )

    def extend(self, calls: Iterable[SourceCall]) -> None:
        self.calls.extend(c for c in calls if isinstance(c, SourceCall))

    def disabled(self, surface: ToolkitSurface, source: str, reason: str) -> None:
        self.record(
            surface=surface,
            source=source,
            params={"reason": reason},
            health=SourceHealth.DISABLED,
            started_at=self._clock.now(),
        )

    def guarded(
        self,
        fetch: Callable[[], T],
        *,
        surface: ToolkitSurface,
        source: str,
        params: Mapping[str, str],
        rows: Callable[[T], int] | None,
        transport: Callable[[], tuple[SourceCall, ...]] | None = None,
    ) -> T | None:
        """Run ``fetch``. A raised exception becomes an ERROR/TIMEOUT call and ``None``.

        With ``rows`` given (a logical market call) the outcome is recorded here, followed by the
        per-HTTP calls ``transport`` drains: OK with rows; with none, EMPTY when nothing failed on
        the wire and the wire's failure otherwise, so a request whose every page failed is not
        reported as an empty answer. With ``rows`` ``None`` (a toolkit or crowd reader, which
        reports its own calls) only a raised exception is recorded here.
        """
        started = self._clock.now()
        try:
            value = fetch()
        except Exception as exc:  # a source's failure degrades the snapshot, never crashes it
            self.record(
                surface=surface,
                source=source,
                params=params,
                health=SourceHealth.TIMEOUT if _timed_out(exc) else SourceHealth.ERROR,
                started_at=started,
                error=_error_text(exc),
            )
            if transport is not None:
                self.extend(transport())
            return None
        if rows is not None:
            wire = transport() if transport is not None else ()
            count = rows(value)
            health, error = _outcome(count, wire)
            self.record(
                surface=surface,
                source=source,
                params=params,
                health=health,
                started_at=started,
                rows=count,
                error=error,
            )
            self.extend(wire)
        return value


_FAILED: Final = frozenset({SourceHealth.ERROR, SourceHealth.TIMEOUT, SourceHealth.HOLLOW})


def _outcome(count: int, wire: Sequence[SourceCall]) -> tuple[SourceHealth, str | None]:
    """The health of a logical request that returned ``count`` usable rows over ``wire``."""
    if count > 0:
        return SourceHealth.OK, None
    failed = [c for c in wire if c.health in _FAILED]
    if not failed:
        return SourceHealth.EMPTY, None
    kinds = {c.health for c in failed}
    health = kinds.pop() if len(kinds) == 1 else SourceHealth.ERROR
    first = failed[0].error or failed[0].health.value
    return health, f"no rows: {len(failed)} of {len(wire)} transport call(s) failed ({first})"[:500]


def _hour_floor(at: datetime) -> datetime:
    return at.replace(minute=0, second=0, microsecond=0)


def _new_york_date(at: datetime) -> date:
    return at.astimezone(NEW_YORK).date()


def _iso(at: datetime) -> str:
    return at.isoformat().replace("+00:00", "Z")


def assemble_text(items: Iterable[TextItem], *, taken_at: datetime) -> tuple[TextItem, ...]:
    """The crowd text a snapshot keeps: one item per ``item_id`` (first occurrence wins), published
    inside :data:`TEXT_LOOKBACK` before ``taken_at`` (and at most :data:`TEXT_CLOCK_SKEW` after),
    newest first, ties broken by id so the order is reproducible."""
    kept: dict[str, TextItem] = {}
    for item in items:
        if item.item_id in kept:
            continue
        if taken_at - TEXT_LOOKBACK <= item.published_at <= taken_at + TEXT_CLOCK_SKEW:
            kept[item.item_id] = item
    return tuple(sorted(kept.values(), key=lambda i: (-i.published_at.timestamp(), i.item_id)))


def crowd_reports(
    screened: Sequence[ScreenedItem],
    *,
    policy: Policy,
    text_measured: bool,
    social_measured: bool,
) -> tuple[CrowdReport, CrowdReport]:
    """``(all_text_report, social_report)`` over screened text.

    The first is the snapshot's ``crowd`` (every channel: what the model reads). The second is the
    X and Reddit subset, from which the social features are computed, because news syndication is
    not the crowd talking. An unmeasured side is the empty report with no ``mentions``.
    """
    universe = policy.symbols
    full = (
        build_report(screened, universe=universe, policy=policy)
        if text_measured
        else _UNMEASURED_CROWD
    )
    social = (
        build_report(
            [s for s in screened if s.item.channel in SOCIAL_CHANNELS],
            universe=universe,
            policy=policy,
        )
        if social_measured
        else _UNMEASURED_CROWD
    )
    return full, social


def snapshot_hash(snapshot: PerceptionSnapshot) -> str:
    """The content hash of ``snapshot`` with ``snapshot_id`` blank: what its id must equal."""
    return content_hash(snapshot.model_copy(update={"snapshot_id": ""}))


def seal(snapshot: PerceptionSnapshot) -> PerceptionSnapshot:
    """``snapshot`` with ``snapshot_id`` set to its content hash (idempotent)."""
    return snapshot.model_copy(update={"snapshot_id": snapshot_hash(snapshot)})


def is_light(snapshot: PerceptionSnapshot) -> bool:
    """Whether ``snapshot`` was taken in light mode: it records ``crowd.collect`` as not asked,
    for the light-snapshot reason. Read from the snapshot itself, so a logged one answers too."""
    return any(
        c.source == "crowd.collect"
        and c.health is SourceHealth.DISABLED
        and c.params.get("reason") == LIGHT_SNAPSHOT_REASON
        for c in snapshot.source_calls
    )


def is_sealed(snapshot: PerceptionSnapshot) -> bool:
    """Whether ``snapshot_id`` still proves the content."""
    return snapshot.snapshot_id == snapshot_hash(snapshot)


_SOCIAL_FACT_FIELDS: Final = ("social_mentions_24h", "social_velocity_per_hour")


def rebuild_text(
    snapshot: PerceptionSnapshot, items: Sequence[TextItem], *, policy: Policy
) -> PerceptionSnapshot:
    """``snapshot`` with its crowd text replaced by ``items``, screened and clustered again, and
    resealed.

    For the red team and the rival harness, which inject text into recorded snapshots: the attacked
    text goes through the same quarantine and clustering a live snapshot's text does, so the
    defences are exercised rather than bypassed (DESIGN.md §14.5). The social features, the crowd
    report and the crowd facts are recomputed; every other feature, quote, fact and source call is
    kept as logged. The replacement text counts as measured on both the all-text and social side.
    """
    text = assemble_text(items, taken_at=snapshot.taken_at)
    screened = screen(text)
    full, social = crowd_reports(screened, policy=policy, text_measured=True, social_measured=True)
    features: dict[str, PositioningFeatures] = {}
    facts = {
        key: value
        for key, value in snapshot.facts.items()
        if not key.startswith("crowd.") and key.rpartition(".")[2] not in _SOCIAL_FACT_FIELDS
    }
    for symbol, feature in snapshot.features.items():
        mentions, velocity, coordinated = social_signals(social, symbol)
        updated = feature.model_copy(
            update={
                "social_mentions_24h": mentions,
                "social_velocity_per_hour": velocity,
                "coordinated_cluster": coordinated,
            }
        )
        features[symbol] = PositioningFeatures.model_validate(updated.model_dump())
        for key, value in facts_from({symbol: features[symbol]}, snapshot.mood, None).items():
            if key.rpartition(".")[2] in _SOCIAL_FACT_FIELDS:
                facts[key] = value
    facts.update(crowd_facts(full))
    rebuilt = snapshot.model_copy(
        update={
            "snapshot_id": "",
            "features": features,
            "crowd": full,
            "text": screened,
            "facts": facts,
        }
    )
    return seal(PerceptionSnapshot.model_validate(rebuilt.model_dump()))


class SnapshotBuilder:
    """Builds, fills and seals one :class:`PerceptionSnapshot` per call to :meth:`build`.

    Stateless between builds: nothing learned in one snapshot changes the next, so a snapshot can
    always be explained from its own calls. The builder never reads a credential and never calls a
    private endpoint; ``mode`` is recorded, not acted on, because perception is keyless in every
    run mode (DESIGN.md §5).
    """

    def __init__(
        self,
        *,
        market: MarketData,
        toolkit: ToolkitReader,
        crowd: CrowdCollector,
        policy: Policy,
        clock: Clock,
        mode: RunMode,
        concurrent_text: bool = False,
    ) -> None:
        self._market = market
        self._toolkit = toolkit
        self._crowd = crowd
        self._policy = policy
        self._clock = clock
        self._mode = mode
        self._concurrent_text = concurrent_text

    # -- market data ------------------------------------------------------------------------------

    def _drain_transport(self) -> tuple[SourceCall, ...]:
        if isinstance(self._market, _TransportLog):
            return tuple(self._market.drain_calls())
        return ()

    def _quotes(self, log: _CallLog, source: PriceSource) -> dict[str, Quote]:
        symbols = self._policy.symbols

        def fetch() -> dict[str, Quote]:
            got = self._market.quotes(source, symbols)
            return {
                symbol: quote
                for symbol, quote in got.items()
                if symbol in symbols and quote.symbol == symbol and quote.source is source
            }

        quotes = log.guarded(
            fetch,
            surface=ToolkitSurface.PUBLIC_MARKET_API,
            source=f"public_v3.tickers[{source}]",
            params={"symbols": ",".join(symbols)},
            rows=len,
            transport=self._drain_transport,
        )
        return quotes or {}

    def _candles(
        self,
        log: _CallLog,
        source: PriceSource,
        symbol: str,
        *,
        kind: CandleKind,
        start: datetime,
        end: datetime,
    ) -> list[Candle]:
        def fetch() -> list[Candle]:
            got = self._market.candles(
                source, symbol, kind=kind, interval=CANDLE_INTERVAL, start=start, end=end
            )
            return [
                c
                for c in got
                if c.symbol == symbol
                and c.source is source
                and c.kind == kind
                and c.interval == CANDLE_INTERVAL
                and start <= c.open_time <= end
            ]

        candles = log.guarded(
            fetch,
            surface=ToolkitSurface.PUBLIC_MARKET_API,
            source=f"public_v3.history_candles[{source},{kind},{CANDLE_INTERVAL}]:{symbol}",
            params={"symbol": symbol, "start": _iso(start), "end": _iso(end)},
            rows=len,
            transport=self._drain_transport,
        )
        return candles or []

    def _funding(self, log: _CallLog, symbol: str, *, taken_at: datetime) -> list[FundingPoint]:
        limit = self._policy.triggers.funding_z_lookback_settlements

        def fetch() -> list[FundingPoint]:
            got = self._market.funding_history(symbol, limit=limit)
            return [
                p
                for p in got
                if p.symbol == symbol and p.source is PriceSource.LIVE and p.ts <= taken_at
            ]

        points = log.guarded(
            fetch,
            surface=ToolkitSurface.PUBLIC_MARKET_API,
            source=f"public_v3.history_fund_rate[live]:{symbol}",
            params={"symbol": symbol, "limit": str(limit)},
            rows=len,
            transport=self._drain_transport,
        )
        return points or []

    # -- toolkit and crowd ------------------------------------------------------------------------

    def _derivatives(self, log: _CallLog, symbol: str) -> DerivativesReading | None:
        answer = log.guarded(
            lambda: self._toolkit.derivatives(symbol),
            surface=ToolkitSurface.SIGNAL_MCP,
            source=f"toolkit.derivatives:{symbol}",
            params={"symbol": symbol},
            rows=None,
        )
        if answer is None:
            return None
        reading, calls = answer
        log.extend(calls)
        return reading if reading.symbol == symbol else None

    def _mood(self, log: _CallLog) -> MoodReading:
        answer = log.guarded(
            self._toolkit.mood,
            surface=ToolkitSurface.SIGNAL_MCP,
            source="toolkit.mood",
            params={},
            rows=None,
        )
        if answer is None:
            return MoodReading()
        reading, calls = answer
        log.extend(calls)
        return reading

    def _text_source(
        self,
        log: _CallLog,
        fetch: Callable[[], tuple[tuple[TextItem, ...], tuple[SourceCall, ...]]],
        *,
        surface: ToolkitSurface,
        source: str,
        params: Mapping[str, str],
    ) -> tuple[tuple[TextItem, ...], bool]:
        """Items from one text source, and whether it answered (any of its calls OK or EMPTY)."""
        before = len(log.calls)
        answer = log.guarded(fetch, surface=surface, source=source, params=params, rows=None)
        if answer is None:
            return (), False
        items, calls = answer
        log.extend(calls)
        answered = any(c.health in _ANSWERED for c in log.calls[before:])
        return tuple(i for i in items if isinstance(i, TextItem)), answered

    # -- the snapshot -----------------------------------------------------------------------------

    def build(self, *, book: BookState | None = None, light: bool = False) -> PerceptionSnapshot:
        """Ask every source once and return the sealed snapshot.

        Never raises for a source failure; raises only for a defect in our own screening code.

        With ``concurrent_text`` a full snapshot starts its four text sources (news, Reddit
        trending, the crowd CLIs, the calendar) on worker threads before the market reads begin,
        and collects them where the serial build would have called them. The slowest of them
        (the crowd searches, minutes serially) then overlaps the market and toolkit reads instead
        of following them, so the model reads a snapshot minutes younger. The recorded calls, their
        order and every fact are the same as the serial build's: only when each call ran changes.
        """
        policy = self._policy
        taken_at = self._clock.now()
        log = self._run_log(taken_at)
        universe = policy.symbols
        if light or not self._concurrent_text:
            return self._build(book=book, light=light, taken_at=taken_at, log=log, early=None)
        since = taken_at - TEXT_LOOKBACK
        pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="snapshot-text")
        try:
            early = _EarlyText(
                news=pool.submit(self._toolkit.news, NEWS_LIMIT),
                reddit=pool.submit(self._toolkit.reddit_trending, REDDIT_LIMIT),
                crowd=pool.submit(self._crowd.collect, universe, since=since),
                calendar=pool.submit(self._calendar_fetch, taken_at),
            )
            return self._build(book=book, light=light, taken_at=taken_at, log=log, early=early)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

    def _build(
        self,
        *,
        book: BookState | None,
        light: bool,
        taken_at: datetime,
        log: _CallLog,
        early: _EarlyText | None,
    ) -> PerceptionSnapshot:
        policy = self._policy
        universe = policy.symbols

        demo_quotes = self._quotes(log, PriceSource.DEMO)
        live_quotes = self._quotes(log, PriceSource.LIVE)

        live_start = taken_at - timedelta(hours=LIVE_CANDLE_HOURS)
        index_start = _hour_floor(taken_at) - timedelta(hours=STALE_INDEX_MOVES + 1)
        funding: dict[str, list[FundingPoint]] = {}
        live_1h: dict[str, list[Candle]] = {}
        demo_index_1h: dict[str, list[Candle]] = {}
        derivatives: dict[str, DerivativesReading | None] = {}
        for entry in policy.universe:
            symbol = entry.symbol
            funding[symbol] = self._funding(log, symbol, taken_at=taken_at)
            live_1h[symbol] = self._candles(
                log, PriceSource.LIVE, symbol, kind="market", start=live_start, end=taken_at
            )
            index = self._candles(
                log, PriceSource.DEMO, symbol, kind="index", start=index_start, end=taken_at
            )
            # Completed bars only: a bar still forming has moved for part of its hour.
            demo_index_1h[symbol] = [
                c for c in index if c.open_time + timedelta(hours=1) <= taken_at
            ]
            derivatives[symbol] = (
                self._derivatives(log, symbol) if entry.asset_class is AssetClass.CRYPTO else None
            )

        mood = mood_from(self._mood(log), policy)

        text: tuple[TextItem, ...] = ()
        calendar: tuple[CalendarItem, ...] = ()
        text_measured = social_measured = False
        if light:
            reason = LIGHT_SNAPSHOT_REASON
            log.disabled(ToolkitSurface.SIGNAL_MCP, "toolkit.news", reason)
            log.disabled(ToolkitSurface.SIGNAL_MCP, "toolkit.reddit_trending", reason)
            log.disabled(ToolkitSurface.CROWD_X, "crowd.collect", reason)
            log.disabled(ToolkitSurface.DATA_MCP, "toolkit.calendar", reason)
        else:
            since = taken_at - TEXT_LOOKBACK
            news, news_ok = self._text_source(
                log,
                early.news.result if early else lambda: self._toolkit.news(NEWS_LIMIT),
                surface=ToolkitSurface.SIGNAL_MCP,
                source="toolkit.news",
                params={"limit": str(NEWS_LIMIT)},
            )
            reddit, reddit_ok = self._text_source(
                log,
                early.reddit.result
                if early
                else lambda: self._toolkit.reddit_trending(REDDIT_LIMIT),
                surface=ToolkitSurface.SIGNAL_MCP,
                source="toolkit.reddit_trending",
                params={"limit": str(REDDIT_LIMIT)},
            )
            collected, crowd_ok = self._text_source(
                log,
                early.crowd.result if early else lambda: self._crowd.collect(universe, since=since),
                surface=ToolkitSurface.CROWD_X,
                source="crowd.collect",
                params={"symbols": ",".join(universe), "since": _iso(since)},
            )
            text = assemble_text((*news, *reddit, *collected), taken_at=taken_at)
            text_measured = news_ok or reddit_ok or crowd_ok
            social_measured = reddit_ok or crowd_ok
            calendar = self._calendar(
                log, taken_at=taken_at, fetch=early.calendar.result if early else None
            )

        screened = screen(text)
        crowd, social = crowd_reports(
            screened,
            policy=policy,
            text_measured=text_measured,
            social_measured=social_measured,
        )
        # Earnings are dated in New York, often by day alone (midnight New York). A row dated today
        # there may be tonight's release, so it stays upcoming until New York's day ends.
        today = _new_york_date(taken_at)
        upcoming = tuple(c for c in calendar if c.at is not None and _new_york_date(c.at) >= today)

        features = {
            entry.symbol: build_features(
                symbol=entry.symbol,
                asset_class=entry.asset_class,
                demo=demo_quotes.get(entry.symbol),
                live=live_quotes.get(entry.symbol),
                derivatives=derivatives[entry.symbol],
                funding=funding[entry.symbol],
                live_1h=live_1h[entry.symbol],
                demo_index_1h=demo_index_1h[entry.symbol],
                crowd=social,
                calendar=upcoming,
                policy=policy,
            )
            for entry in policy.universe
        }

        facts = facts_from(features, mood, book)
        facts.update(crowd_facts(crowd))
        facts.update(coverage_facts(log.calls))

        return seal(
            PerceptionSnapshot(
                snapshot_id="",
                taken_at=taken_at,
                mode=self._mode,
                policy_version=policy.version,
                universe=universe,
                demo_quotes=demo_quotes,
                live_quotes=live_quotes,
                features=features,
                mood=mood,
                crowd=crowd,
                text=screened,
                calendar=calendar,
                source_calls=tuple(log.calls),
                facts=facts,
            )
        )

    def _run_log(self, taken_at: datetime) -> _CallLog:
        # Transport calls logged before this build started were made by someone else.
        self._drain_transport()
        return _CallLog(self._clock, taken_at)

    def _equities(self) -> list[str]:
        return [e.symbol for e in self._policy.universe if e.asset_class is AssetClass.US_EQUITY]

    def _calendar_fetch(
        self, taken_at: datetime
    ) -> tuple[tuple[CalendarItem, ...], tuple[SourceCall, ...]]:
        """The calendar read itself, for a worker thread (no equities: nothing to ask)."""
        equities = self._equities()
        if not equities:
            return (), ()
        return self._toolkit.calendar(equities, since=taken_at - CALENDAR_LOOKBACK)

    def _calendar(
        self,
        log: _CallLog,
        *,
        taken_at: datetime,
        fetch: Callable[[], tuple[tuple[CalendarItem, ...], tuple[SourceCall, ...]]] | None = None,
    ) -> tuple[CalendarItem, ...]:
        equities = self._equities()
        if not equities:
            return ()
        since = taken_at - CALENDAR_LOOKBACK
        answer = log.guarded(
            fetch or (lambda: self._toolkit.calendar(equities, since=since)),
            surface=ToolkitSurface.DATA_MCP,
            source="toolkit.calendar",
            params={"symbols": ",".join(equities), "since": _iso(since)},
            rows=None,
        )
        if answer is None:
            return ()
        items, calls = answer
        log.extend(calls)
        return tuple(i for i in items if isinstance(i, CalendarItem))


__all__ = [
    "CALENDAR_LOOKBACK",
    "CANDLE_INTERVAL",
    "LIGHT_SNAPSHOT_REASON",
    "LIVE_CANDLE_HOURS",
    "NEWS_LIMIT",
    "REDDIT_LIMIT",
    "TEXT_CLOCK_SKEW",
    "TEXT_LOOKBACK",
    "SnapshotBuilder",
    "assemble_text",
    "crowd_reports",
    "is_light",
    "is_sealed",
    "rebuild_text",
    "seal",
    "snapshot_hash",
]
