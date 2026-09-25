"""Feed health (run 2, change run2-d2): alarms, streaks, and why a trigger kind is blind."""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from sentiment_agent.perception.feeds import CADENCE_REASON, changed, feed_report, summary_line
from sentiment_agent.perception.snapshot import LIGHT_SNAPSHOT_REASON, is_light
from sentiment_agent.policy import POLICY_V2
from sentiment_agent.types import (
    CrowdReport,
    FeedAlarm,
    FeedHealthReport,
    MarketMood,
    PerceptionSnapshot,
    PositioningFeatures,
    RunMode,
    SourceCall,
    SourceHealth,
    ToolkitSurface,
    TriggerKind,
)

T0 = datetime(2026, 9, 28, 1, 0, tzinfo=UTC)
FIVE = timedelta(minutes=5)
OI = {"BTCUSDT": 1.34}
SCOPE = POLICY_V2.symbols
EQUITIES = tuple(u.symbol for u in POLICY_V2.universe if u.asset_class.value == "us_equity")
LIGHT_OFF = ("toolkit.news", "toolkit.reddit_trending", "crowd.collect", "toolkit.calendar")


def call(
    source: str,
    health: SourceHealth = SourceHealth.OK,
    *,
    params: dict[str, str] | None = None,
    error: str | None = None,
    surface: ToolkitSurface = ToolkitSurface.SIGNAL_MCP,
) -> SourceCall:
    return SourceCall(
        call_id=f"pc-{source}-{health}",
        surface=surface,
        source=source,
        params=params or {},
        health=health,
        started_at=T0,
        latency_ms=1,
        rows=1 if health is SourceHealth.OK else 0,
        blob=None,
        error=error,
    )


def feature(symbol: str, **values: Any) -> PositioningFeatures:
    entry = POLICY_V2.entry(symbol)
    assert entry is not None
    data: dict[str, Any] = {
        name: None
        for name in PositioningFeatures.model_fields
        if name not in ("symbol", "asset_class", "coordinated_cluster")
    }
    data.update(symbol=symbol, asset_class=entry.asset_class, coordinated_cluster=False)
    data.update(values)
    return PositioningFeatures.model_validate(data)


def healthy_features() -> dict[str, PositioningFeatures]:
    out = {s: feature(s, funding_z_live=0.5) for s in SCOPE}
    out["BTCUSDT"] = feature("BTCUSDT", funding_z_live=0.5, oi_change_1h_pct=0.2)
    return out


def snap(
    at: datetime,
    calls: Sequence[SourceCall],
    *,
    light: bool = True,
    feats: dict[str, PositioningFeatures] | None = None,
    crypto_fg: int | None = 50,
    market_fg: int | None = 50,
) -> PerceptionSnapshot:
    recorded = list(calls)
    if light:
        recorded += [
            call(s, SourceHealth.DISABLED, params={"reason": LIGHT_SNAPSHOT_REASON})
            for s in LIGHT_OFF
        ]
    return PerceptionSnapshot(
        snapshot_id=f"snap-{at.isoformat()}",
        taken_at=at,
        mode=RunMode.SIMULATED,
        policy_version=POLICY_V2.version,
        universe=POLICY_V2.symbols,
        demo_quotes={},
        live_quotes={},
        features=feats if feats is not None else healthy_features(),
        mood=MarketMood(crypto_fear_greed=crypto_fg, market_fear_greed=market_fg),
        crowd=CrowdReport(
            items=0, withheld=0, distinct_stories=0, duplication_ratio=0.0, clusters=(), mentions={}
        ),
        text=(),
        calendar=(),
        source_calls=tuple(recorded),
        facts={},
    )


def report(
    snapshot: PerceptionSnapshot, previous: FeedHealthReport | None = None
) -> FeedHealthReport:
    return feed_report(
        snapshot, policy=POLICY_V2, previous=previous, oi_thresholds=OI, funding_scope=SCOPE
    )


def blind_of(r: FeedHealthReport, kind: TriggerKind) -> list[tuple[str, tuple[str, ...]]]:
    return [(b.cause, b.symbols) for b in r.blind if b.kind is kind]


# --- alarms --------------------------------------------------------------------------------------


def test_a_hollow_source_raises_an_alarm_and_counts_its_streak() -> None:
    hollow = call("sentiment_index.current", SourceHealth.HOLLOW, error="upstream error envelope")
    first = report(snap(T0, [hollow]))
    assert first.raised == ("sentiment_index.current",)
    (alarm,) = first.alarms
    assert (alarm.health, alarm.since, alarm.snapshots) == (SourceHealth.HOLLOW, T0, 1)
    assert alarm.error == "upstream error envelope"
    second = report(snap(T0 + FIVE, [hollow]), first)
    assert second.raised == ()
    assert second.alarms[0].since == T0
    assert second.alarms[0].snapshots == 2
    assert not changed(second, first)


def test_an_answer_clears_the_alarm_and_not_asking_does_not() -> None:
    error = call("do_query:equity_calendar", SourceHealth.ERROR, error="status 503")
    first = report(snap(T0, [error], light=False))
    assert first.raised == ("do_query:equity_calendar",)
    # A light snapshot does not ask the calendar: the alarm stands, unchanged.
    light = report(snap(T0 + FIVE, []), first)
    assert light.alarms == first.alarms
    assert light.cleared == ()
    # The next full snapshot's answer clears it.
    ok = report(snap(T0 + 2 * FIVE, [call("do_query:equity_calendar")], light=False), light)
    assert ok.alarms == ()
    assert ok.cleared == ("do_query:equity_calendar",)
    assert changed(ok, light)


def test_one_failing_call_of_several_is_an_alarm_that_says_how_many() -> None:
    calls = [
        call("do_query:equity_calendar", params={"symbol": "NVDA"}),
        call(
            "do_query:equity_calendar",
            SourceHealth.TIMEOUT,
            params={"symbol": "TSLA"},
            error="timed out",
        ),
    ]
    (alarm,) = report(snap(T0, calls, light=False)).alarms
    assert alarm.health is SourceHealth.TIMEOUT
    assert alarm.error == "1 of 2 calls failed; first: timed out"


@pytest.mark.parametrize("health", [SourceHealth.EMPTY, SourceHealth.DISABLED, SourceHealth.OK])
def test_empty_disabled_and_ok_are_not_failures(health: SourceHealth) -> None:
    assert report(snap(T0, [call("news_feed.latest", health)])).alarms == ()


def test_an_alarm_for_a_healthy_state_is_refused() -> None:
    with pytest.raises(ValueError, match="failing source"):
        FeedAlarm(
            feed="x",
            surface=ToolkitSurface.SIGNAL_MCP,
            health=SourceHealth.EMPTY,
            since=T0,
            snapshots=1,
        )


# --- blind trigger kinds -------------------------------------------------------------------------


def test_a_healthy_light_snapshot_is_blind_only_by_cadence() -> None:
    r = report(snap(T0, [call("sentiment_index.current")]))
    assert r.light
    assert {b.cause for b in r.blind} == {"cadence"}
    assert {b.kind for b in r.blind} == {
        TriggerKind.COORDINATED_CLUSTER,
        TriggerKind.EARNINGS_EVENT,
        TriggerKind.FILING_EVENT,
    }
    assert all(b.reason == CADENCE_REASON for b in r.blind)
    assert r.failure_blind() == ()


def test_fear_greed_blind_names_the_failing_sources() -> None:
    calls = [
        call("sentiment_index.current", SourceHealth.HOLLOW),
        call(
            "do_query:crypto_sentiment_crypto_fear_greed",
            SourceHealth.ERROR,
            surface=ToolkitSurface.DATA_MCP,
        ),
        call("do_query:sentiment_market_fear_greed", surface=ToolkitSurface.DATA_MCP),
    ]
    r = report(snap(T0, calls, crypto_fg=None, market_fg=40))
    (crypto,) = [b for b in r.blind if b.kind is TriggerKind.FEAR_GREED_EXTREME]
    assert crypto.cause == "feed_failed"
    assert crypto.symbols == ("BTCUSDT",)
    assert crypto.feeds == (
        "do_query:crypto_sentiment_crypto_fear_greed",
        "sentiment_index.current",
    )
    assert "sentiment_index.current" in crypto.reason


def test_funding_blind_per_instrument_with_its_history_feed() -> None:
    feats = healthy_features()
    feats["NVDAUSDT"] = feature("NVDAUSDT", funding_z_live=None)
    feats["TSLAUSDT"] = feature("TSLAUSDT", funding_z_live=None)
    calls = [
        call(
            "public_v3.history_fund_rate[live]:NVDAUSDT",
            SourceHealth.ERROR,
            surface=ToolkitSurface.PUBLIC_MARKET_API,
        ),
        call(
            "public_v3.history_fund_rate[live]:TSLAUSDT",
            surface=ToolkitSurface.PUBLIC_MARKET_API,
        ),
    ]
    r = report(snap(T0, calls, feats=feats))
    assert sorted(blind_of(r, TriggerKind.FUNDING_ZSCORE)) == [
        ("feed_failed", ("NVDAUSDT",)),
        ("no_data", ("TSLAUSDT",)),
    ]


def test_open_interest_blind_and_a_crypto_leg_without_a_threshold() -> None:
    feats = healthy_features()
    feats["BTCUSDT"] = feature("BTCUSDT", funding_z_live=0.1, oi_change_1h_pct=None)
    calls = [call("do_query:crypto_futures_open_interest_history", SourceHealth.ERROR)]
    r = report(snap(T0, calls, feats=feats))
    assert blind_of(r, TriggerKind.OPEN_INTEREST_JUMP) == [("feed_failed", ("BTCUSDT",))]
    no_threshold = feed_report(
        snap(T0, [], feats=feats),
        policy=POLICY_V2,
        previous=None,
        oi_thresholds={},
        funding_scope=SCOPE,
    )
    assert blind_of(no_threshold, TriggerKind.OPEN_INTEREST_JUMP) == [
        ("no_threshold", ("BTCUSDT",))
    ]


def test_a_full_snapshot_with_no_crowd_answer_is_blind_to_clusters() -> None:
    calls = [
        call("crowd.collect", SourceHealth.ERROR, error="twitter-cli not logged in"),
        call("toolkit.news", SourceHealth.HOLLOW),
    ]
    r = report(snap(T0, calls, light=False))
    (clusters,) = [b for b in r.blind if b.kind is TriggerKind.COORDINATED_CLUSTER]
    assert clusters.cause == "feed_failed"
    assert clusters.feeds == ("crowd.collect", "toolkit.news")
    # One channel answering is enough to see clusters.
    answered = report(snap(T0, [*calls, call("rdt-cli:search:NVDAUSDT")], light=False))
    assert blind_of(answered, TriggerKind.COORDINATED_CLUSTER) == []


def test_earnings_blind_by_failure_or_by_an_empty_calendar() -> None:
    feats = healthy_features()
    calls = [
        call(
            "do_query:equity_calendar",
            SourceHealth.ERROR,
            params={"symbol": "NVDA"},
            surface=ToolkitSurface.DATA_MCP,
        ),
        *(
            call(
                "do_query:equity_calendar",
                params={"symbol": s.removesuffix("USDT")},
                surface=ToolkitSurface.DATA_MCP,
            )
            for s in EQUITIES
            if s != "NVDAUSDT"
        ),
    ]
    feats["AAPLUSDT"] = feature(
        "AAPLUSDT", funding_z_live=0.1, next_earnings_at=T0 + timedelta(days=20)
    )
    r = report(snap(T0, calls, light=False, feats=feats))
    earnings = blind_of(r, TriggerKind.EARNINGS_EVENT)
    assert ("feed_failed", ("NVDAUSDT",)) in earnings
    (no_data,) = [symbols for cause, symbols in earnings if cause == "no_data"]
    assert "AAPLUSDT" not in no_data
    assert "NVDAUSDT" not in no_data
    assert len(no_data) == len(EQUITIES) - 2


def test_changed_follows_blindness_of_the_light_kinds_not_the_cadence() -> None:
    light = report(snap(T0, [call("sentiment_index.current")]))
    assert changed(light, None)
    full = report(snap(T0 + FIVE, [call("sentiment_index.current")], light=False), light)
    assert not changed(full, light)  # cadence blindness came and went: nothing to log
    failing = report(snap(T0 + 2 * FIVE, [call("sentiment_index.current")], crypto_fg=None), full)
    assert changed(failing, full)  # crypto F&G blind, no alarm raised


def test_summary_line() -> None:
    assert summary_line(None) == ""
    assert summary_line(report(snap(T0, [call("sentiment_index.current")]))) == ""
    hollow_call = call("sentiment_index.current", SourceHealth.HOLLOW)
    hollow = report(snap(T0, [hollow_call], crypto_fg=None))
    line = summary_line(hollow)
    assert line.startswith("1 feed(s) failing: sentiment_index.current (hollow x1)")
    assert "blind trigger kinds: fear_greed_extreme" in line


def test_is_light_reads_the_snapshot_itself() -> None:
    assert is_light(snap(T0, []))
    assert not is_light(snap(T0, [], light=False))


def test_report_invariants() -> None:
    alarm = FeedAlarm(
        feed="a",
        surface=ToolkitSurface.SIGNAL_MCP,
        health=SourceHealth.ERROR,
        since=T0,
        snapshots=1,
    )
    with pytest.raises(ValueError, match="raised feed"):
        FeedHealthReport(at=T0, snapshot_id="s", light=True, alarms=(), raised=("a",))
    with pytest.raises(ValueError, match="cleared feed"):
        FeedHealthReport(at=T0, snapshot_id="s", light=True, alarms=(alarm,), cleared=("a",))
    with pytest.raises(ValueError, match="twice"):
        FeedHealthReport(at=T0, snapshot_id="s", light=True, alarms=(alarm, alarm))
