"""Feed health: which sources are failing, since when, and which trigger kinds that leaves blind.

Run 1 showed why this exists. ``sentiment_index.current`` came back hollow in 226 of 226
snapshots, bitget-mcp-server's ``do_query`` answered 503 on every call from 2026-09-25 10:14 UTC,
and the calendar never carried an upcoming report date; every one of those was recorded as a
:class:`~sentiment_agent.types.SourceCall` inside a snapshot, and nothing said so anywhere a person
looks. A degraded snapshot is by design not a crash (DESIGN.md §6.2); it must be a visible fact.

:func:`feed_report` turns one snapshot, and the report before it, into a
:class:`~sentiment_agent.types.FeedHealthReport`:

* **Alarms.** A source is failing when any call of that name on the snapshot came back ``error``,
  ``timeout`` or ``hollow`` (:data:`~sentiment_agent.types.FAILING_HEALTH`). An alarm carries the
  start of its streak and how many consecutive snapshots asked the source and got a failure. A
  source the snapshot did not ask (a light snapshot does not read crowd text or the calendar)
  keeps its alarm as it was: not asked is not recovered. A source that answered clears its alarm.
* **Blind trigger kinds.** For every trigger kind, the instruments it could not evaluate on this
  snapshot and the cause: a named failing feed (``feed_failed``), sources that answered with
  nothing usable (``no_data``, e.g. a calendar with no upcoming report date), a snapshot that does
  not read those sources by design (``cadence``), or no open-interest threshold frozen at genesis
  (``no_threshold``). The feeds each kind reads are the sources whose values reach the trigger in
  :mod:`sentiment_agent.events.triggers`, named as the snapshot records them.

The runtime logs the report as a ``feed_health`` event on every full (decision) snapshot and on a
light snapshot whenever :func:`changed` says an alarm was raised or cleared or the blindness of a
kind every snapshot evaluates changed; the decision card carries the report of the
snapshot the model read, and the public page shows the open alarms, their history and the blind
kinds.

Everything here is a pure function of the snapshot, the previous report and the frozen thresholds:
it calls nothing, so the same report can be rebuilt from the log.
"""

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Final

from sentiment_agent.events.triggers import LIGHT_SNAPSHOT_KINDS
from sentiment_agent.perception.snapshot import is_light
from sentiment_agent.types import (
    FAILING_HEALTH,
    AssetClass,
    BlindCause,
    BlindTrigger,
    FeedAlarm,
    FeedHealthReport,
    PerceptionSnapshot,
    Policy,
    SourceCall,
    SourceHealth,
    TriggerKind,
)

ANSWERED: Final = frozenset({SourceHealth.OK, SourceHealth.EMPTY})

CRYPTO_FEAR_GREED_FEEDS: Final = (
    "toolkit.mood",
    "sentiment_index.current",
    "do_query:crypto_sentiment_crypto_fear_greed",
)
MARKET_FEAR_GREED_FEEDS: Final = ("toolkit.mood", "do_query:sentiment_market_fear_greed")
FUNDING_HISTORY_FEEDS: Final = ("public_v3.history_fund_rate[live]:", "history-fund-rate.live:")
"""The live settlement history the z-score is computed over (per instrument); the live rate it
scores comes from the live tickers."""
OPEN_INTEREST_FEEDS: Final = (
    "derivatives_sentiment.open_interest",
    "do_query:crypto_futures_open_interest_history",
)
CROWD_TEXT_FEEDS: Final = (
    "crowd.collect",
    "twitter-cli:",
    "rdt-cli:",
    "toolkit.news",
    "news_feed.",
    "toolkit.reddit_trending",
    "derivatives_sentiment.reddit_trending",
)
"""Every channel whose text reaches the story clusters (``PerceptionSnapshot.crowd``)."""
EARNINGS_FEEDS: Final = ("toolkit.calendar", "do_query:equity_calendar")
FILING_FEEDS: Final = ("toolkit.calendar", "do_query:equity_ownership_insider_trading")

CADENCE_REASON: Final = (
    "not read on a light snapshot: crowd text and the calendar are read on full snapshots, and "
    "this kind is evaluated on every full snapshot (DESIGN.md §7)"
)


def _matches(source: str, feeds: Sequence[str]) -> bool:
    """A feed name ending in ``:`` or ``.`` is a prefix (a family of per-symbol calls)."""
    return any(source.startswith(f) if f[-1] in ":." else source == f for f in feeds)


def _named(feeds: Sequence[str]) -> Callable[[str], bool]:
    def match(source: str) -> bool:
        return _matches(source, feeds)

    return match


def _failing(calls: Sequence[SourceCall], match: Callable[[str], bool]) -> tuple[str, ...]:
    return tuple(
        sorted({c.source for c in calls if c.health in FAILING_HEALTH and match(c.source)})
    )


def _answered(calls: Sequence[SourceCall], match: Callable[[str], bool]) -> bool:
    return any(c.health in ANSWERED and match(c.source) for c in calls)


def _for_symbol(symbol: str, underlying: str | None = None) -> Callable[[SourceCall], bool]:
    """Calls made for one instrument: named ``...:SYMBOL`` or asked with its symbol or ticker."""
    names = {symbol} | ({underlying} if underlying else set())

    def match(call: SourceCall) -> bool:
        return call.source.rpartition(":")[2] in names or call.params.get("symbol") in names

    return match


def _underlying(symbol: str) -> str:
    return symbol.removesuffix("USDT")


# ------------------------------------------------------------------------------------------------
# Alarms
# ------------------------------------------------------------------------------------------------


def _alarms(
    snapshot: PerceptionSnapshot, previous: FeedHealthReport | None
) -> tuple[tuple[FeedAlarm, ...], tuple[str, ...], tuple[str, ...]]:
    by_source: dict[str, list[SourceCall]] = defaultdict(list)
    for call in snapshot.source_calls:
        by_source[call.source].append(call)
    before = {a.feed: a for a in previous.alarms} if previous is not None else {}
    alarms: dict[str, FeedAlarm] = {}
    raised: list[str] = []
    cleared: list[str] = []
    for source, calls in sorted(by_source.items()):
        asked = [c for c in calls if c.health is not SourceHealth.DISABLED]
        if not asked:
            continue
        failed = [c for c in asked if c.health in FAILING_HEALTH]
        prior = before.get(source)
        if not failed:
            if prior is not None:
                cleared.append(source)
            continue
        first = failed[0]
        error = first.error or first.health.value
        if len(asked) > 1:
            error = f"{len(failed)} of {len(asked)} calls failed; first: {error}"
        alarms[source] = FeedAlarm(
            feed=source,
            surface=first.surface,
            health=first.health,
            since=prior.since if prior is not None else snapshot.taken_at,
            snapshots=prior.snapshots + 1 if prior is not None else 1,
            error=error[:500],
        )
        if prior is None:
            raised.append(source)
    for feed, prior in before.items():
        if feed not in alarms and feed not in cleared:
            alarms[feed] = prior  # not asked on this snapshot: the alarm stands as it was
    return tuple(alarms[k] for k in sorted(alarms)), tuple(raised), tuple(cleared)


# ------------------------------------------------------------------------------------------------
# Blind trigger kinds
# ------------------------------------------------------------------------------------------------


class _Blind:
    """Collects blind instruments per (kind, cause, feeds) so a report stays one line per cause."""

    def __init__(self) -> None:
        self._groups: dict[tuple[TriggerKind, BlindCause, tuple[str, ...], str], list[str]] = {}

    def add(
        self,
        kind: TriggerKind,
        symbols: Sequence[str],
        cause: BlindCause,
        reason: str,
        feeds: Sequence[str] = (),
    ) -> None:
        key = (kind, cause, tuple(feeds), reason)
        self._groups.setdefault(key, []).extend(symbols)

    def result(self) -> tuple[BlindTrigger, ...]:
        out = [
            BlindTrigger(
                kind=kind,
                symbols=tuple(dict.fromkeys(symbols)),
                cause=cause,
                reason=reason,
                feeds=feeds,
            )
            for (kind, cause, feeds, reason), symbols in self._groups.items()
        ]
        order = {kind: i for i, kind in enumerate(TriggerKind)}
        return tuple(sorted(out, key=lambda b: (order[b.kind], b.cause, b.symbols, b.feeds)))


def _unread(what: str, feeds: Sequence[str]) -> str:
    return f"{what}: failing source(s) " + ", ".join(feeds)


def blind_triggers(
    snapshot: PerceptionSnapshot,
    *,
    policy: Policy,
    oi_thresholds: Mapping[str, float],
    funding_scope: Sequence[str],
) -> tuple[BlindTrigger, ...]:
    """Every trigger kind that could not be evaluated on ``snapshot``, per instrument, with why."""
    calls = snapshot.source_calls
    blind = _Blind()
    universe = policy.universe
    crypto = [u.symbol for u in universe if u.asset_class is AssetClass.CRYPTO]
    us_session = [u.symbol for u in universe if u.asset_class.follows_us_session]
    equities = [u.symbol for u in universe if u.asset_class is AssetClass.US_EQUITY]

    # Fear & Greed, one index per series.
    mood = snapshot.mood
    for value, symbols, feeds, name in (
        (mood.crypto_fear_greed, crypto, CRYPTO_FEAR_GREED_FEEDS, "crypto Fear & Greed"),
        (mood.market_fear_greed, us_session, MARKET_FEAR_GREED_FEEDS, "equity-market F&G"),
    ):
        if value is not None:
            continue
        failing = _failing(calls, _named(feeds))
        if failing:
            blind.add(
                TriggerKind.FEAR_GREED_EXTREME,
                symbols,
                "feed_failed",
                _unread(f"{name} unread", failing),
                failing,
            )
        else:
            blind.add(
                TriggerKind.FEAR_GREED_EXTREME,
                symbols,
                "no_data",
                f"{name}: its sources answered with no index value",
            )

    # Funding z-score, per instrument in the policy's scope.
    for symbol in funding_scope:
        features = snapshot.features.get(symbol)
        if features is not None and features.funding_z_live is not None:
            continue
        match = _for_symbol(symbol)
        failing = tuple(
            sorted(
                {
                    c.source
                    for c in calls
                    if c.health in FAILING_HEALTH
                    and (
                        (_matches(c.source, FUNDING_HISTORY_FEEDS) and match(c))
                        or c.source in ("public_v3.tickers[live]", f"tickers.live:{symbol}")
                    )
                }
            )
        )
        if failing:
            blind.add(
                TriggerKind.FUNDING_ZSCORE,
                [symbol],
                "feed_failed",
                _unread("live funding z-score not computed", failing),
                failing,
            )
        else:
            blind.add(
                TriggerKind.FUNDING_ZSCORE,
                [symbol],
                "no_data",
                "live funding z-score not computed: fewer than "
                f"{policy.triggers.funding_z_lookback_settlements} settlements, no live rate, or "
                "no spread in the window",
            )

    # Open-interest jumps: instruments with a frozen threshold, and crypto legs without one.
    for symbol in crypto:
        if symbol not in oi_thresholds:
            blind.add(
                TriggerKind.OPEN_INTEREST_JUMP,
                [symbol],
                "no_threshold",
                "no open-interest threshold was frozen at genesis for this instrument",
            )
    for symbol in sorted(oi_thresholds):
        features = snapshot.features.get(symbol)
        if features is not None and features.oi_change_1h_pct is not None:
            continue
        failing = _failing(calls, _named((*OPEN_INTEREST_FEEDS, f"toolkit.derivatives:{symbol}")))
        if failing:
            blind.add(
                TriggerKind.OPEN_INTEREST_JUMP,
                [symbol],
                "feed_failed",
                _unread("1-hour open-interest change not computed", failing),
                failing,
            )
        else:
            blind.add(
                TriggerKind.OPEN_INTEREST_JUMP,
                [symbol],
                "no_data",
                "1-hour open-interest change not computed: no reading an hour apart",
            )

    if is_light(snapshot):
        for kind in (
            TriggerKind.COORDINATED_CLUSTER,
            TriggerKind.EARNINGS_EVENT,
            TriggerKind.FILING_EVENT,
        ):
            blind.add(kind, (), "cadence", CADENCE_REASON)
        return blind.result()

    # Coordinated clusters: blind only when no text channel answered at all.
    if not _answered(calls, _named(CROWD_TEXT_FEEDS)):
        failing = _failing(calls, _named(CROWD_TEXT_FEEDS))
        if failing:
            blind.add(
                TriggerKind.COORDINATED_CLUSTER,
                (),
                "feed_failed",
                _unread("no crowd text read", failing),
                failing,
            )
        else:
            blind.add(
                TriggerKind.COORDINATED_CLUSTER,
                (),
                "no_data",
                "no crowd text channel is configured or answered",
            )

    # Earnings and filings, per equity, from the per-ticker calendar calls.
    for kind, feeds, what in (
        (TriggerKind.EARNINGS_EVENT, EARNINGS_FEEDS, "earnings calendar unread"),
        (TriggerKind.FILING_EVENT, FILING_FEEDS, "insider filings unread"),
    ):
        wrapper = _failing(calls, _named(("toolkit.calendar",)))
        for symbol in equities:
            match = _for_symbol(symbol, _underlying(symbol))
            failing = wrapper or tuple(
                sorted(
                    {
                        c.source
                        for c in calls
                        if c.health in FAILING_HEALTH and _matches(c.source, feeds) and match(c)
                    }
                )
            )
            if failing:
                blind.add(kind, [symbol], "feed_failed", _unread(what, failing), failing)
            elif kind is TriggerKind.EARNINGS_EVENT:
                features = snapshot.features.get(symbol)
                if features is None or features.next_earnings_at is None:
                    blind.add(
                        kind,
                        [symbol],
                        "no_data",
                        "the earnings calendar answered but lists no upcoming report date",
                    )
    return blind.result()


# ------------------------------------------------------------------------------------------------
# The report
# ------------------------------------------------------------------------------------------------


def feed_report(
    snapshot: PerceptionSnapshot,
    *,
    policy: Policy,
    previous: FeedHealthReport | None,
    oi_thresholds: Mapping[str, float],
    funding_scope: Sequence[str],
) -> FeedHealthReport:
    """The feed health as of ``snapshot``, continuing ``previous`` (module docstring)."""
    alarms, raised, cleared = _alarms(snapshot, previous)
    return FeedHealthReport(
        at=snapshot.taken_at,
        snapshot_id=snapshot.snapshot_id,
        light=is_light(snapshot),
        alarms=alarms,
        raised=raised,
        cleared=cleared,
        blind=blind_triggers(
            snapshot, policy=policy, oi_thresholds=oi_thresholds, funding_scope=funding_scope
        ),
    )


def _blind_keys(report: FeedHealthReport) -> frozenset[tuple[TriggerKind, str, tuple[str, ...]]]:
    """Blindness of the kinds every snapshot evaluates. The crowd and calendar kinds are read on
    full snapshots only, whose reports are always logged, so comparing them across a light and a
    full snapshot would only log the cadence."""
    return frozenset(
        (b.kind, b.cause, b.symbols) for b in report.blind if b.kind in LIGHT_SNAPSHOT_KINDS
    )


def changed(report: FeedHealthReport, previous: FeedHealthReport | None) -> bool:
    """Whether ``report`` is worth a ledger event after ``previous``: the first report, an alarm
    raised or cleared, or a change in which instruments Fear & Greed, funding or open interest
    cannot be evaluated for, or why."""
    if previous is None or report.raised or report.cleared:
        return True
    return _blind_keys(report) != _blind_keys(previous)


def summary_line(report: FeedHealthReport | None) -> str:
    """One line for the health file and ``t2sa status``; empty when nothing is failing."""
    if report is None or not (report.alarms or report.failure_blind()):
        return ""
    feeds = ", ".join(f"{a.feed} ({a.health.value} x{a.snapshots})" for a in report.alarms)
    kinds = sorted({b.kind.value for b in report.failure_blind()})
    text = f"{len(report.alarms)} feed(s) failing: {feeds}"
    if kinds:
        text += "; blind trigger kinds: " + ", ".join(kinds)
    return text[:1000]


__all__ = [
    "ANSWERED",
    "CADENCE_REASON",
    "CROWD_TEXT_FEEDS",
    "CRYPTO_FEAR_GREED_FEEDS",
    "EARNINGS_FEEDS",
    "FILING_FEEDS",
    "FUNDING_HISTORY_FEEDS",
    "MARKET_FEAR_GREED_FEEDS",
    "OPEN_INTEREST_FEEDS",
    "blind_triggers",
    "changed",
    "feed_report",
    "summary_line",
]
