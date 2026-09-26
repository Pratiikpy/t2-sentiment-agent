"""Run 2's runtime changes, end to end through the loop: full-snapshot trigger kinds join the
decision they are seen on (run2-d1), and feed health is logged, carried on the card and restored
(run2-d2)."""

import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from runtime.support import FakeOts, FakeToolkit, FakeWorld, flat, make_parts
from sentiment_agent.clock import ManualClock
from sentiment_agent.execution.simulated import SimulatedVenue
from sentiment_agent.ledger.blobs import FileBlobStore
from sentiment_agent.policy import POLICY_V2
from sentiment_agent.redteam.attacks import coordinated_pump
from sentiment_agent.runtime.health import read_health, status_lines
from sentiment_agent.runtime.loop import RunLoop
from sentiment_agent.runtime.wiring import App, build_app
from sentiment_agent.site.cards import build_cards
from sentiment_agent.types import (
    EventKind,
    FeedHealthReport,
    MoodReading,
    RunMode,
    SourceCall,
    SourceHealth,
    ToolkitSurface,
    Trigger,
    TriggerKind,
)

# A weekday inside run 2's scoring window (2026-09-28 to 2026-10-01 00:00 UTC).
WEDNESDAY = datetime(2026, 9, 30, tzinfo=UTC)


def _at(hh: int, mm: int, ss: int = 0) -> datetime:
    return WEDNESDAY.replace(hour=hh, minute=mm, second=ss)


class Rig:
    def __init__(self, root: Path, start: datetime, script: Sequence[Any] = ()) -> None:
        self.clock = ManualClock(start)
        self.world = FakeWorld(self.clock)
        self.toolkit = FakeToolkit(self.clock)
        venue = SimulatedVenue(
            market=self.world, clock=self.clock, starting_equity=Decimal("10000")
        )
        parts = make_parts(
            self.clock,
            world=self.world,
            toolkit=self.toolkit,
            script=script,
            venue=venue,
            ots=FakeOts(),
        )
        self.app: App = build_app(
            root, RunMode.SIMULATED, llm="scripted", clock=self.clock, parts=parts
        )
        self.loop = RunLoop(self.app)

    def payloads(self, kind: EventKind) -> list[dict[str, Any]]:
        return [e.payload for e in self.app.chain.events(frozenset({kind}))]


@pytest.fixture
def rig(workdir: Path) -> Any:
    made = Rig(workdir, _at(13, 29), script=[flat("the US open; nothing to do")])
    yield made
    made.app.close()


def test_the_runtime_loads_policy_v2_by_default(rig: Rig) -> None:
    assert rig.app.policy == POLICY_V2
    assert rig.app.triggers.funding_scope == POLICY_V2.symbols


def test_a_coordinated_cluster_on_the_full_snapshot_joins_the_heartbeat_decision(
    rig: Rig,
) -> None:
    rig.toolkit.news_items = coordinated_pump(
        "NVDAUSDT", direction="long", n_accounts=4, window=timedelta(minutes=40), at=_at(13, 20)
    )
    rig.loop.tick()  # 13:29: a light snapshot, which does not read text
    assert rig.payloads(EventKind.TRIGGER) == []
    rig.clock.set(_at(13, 30))
    report = rig.loop.tick()
    assert "decision_cycle" in report.did
    assert "full_snapshot_riders_1" in report.did
    triggers = [Trigger.model_validate(t) for t in rig.payloads(EventKind.TRIGGER)]
    assert sorted(t.kind.value for t in triggers) == ["coordinated_cluster", "heartbeat_us_open"]
    (cluster,) = [t for t in triggers if t.kind is TriggerKind.COORDINATED_CLUSTER]
    assert cluster.symbols == ("NVDAUSDT",)
    # One admission batch, one decision, and it names both triggers.
    assert len({t.fired_at for t in triggers}) == 1
    (record,) = rig.app.projection.decisions
    assert sorted(record.trigger_ids) == sorted(t.trigger_id for t in triggers)
    # The cluster was read off the very snapshot the model decided on.
    assert cluster.snapshot_id == record.snapshot_id
    # A heartbeat batch takes no event slot.
    assert rig.app.triggers.event_decisions_on(WEDNESDAY.date()) == 0
    # Light snapshots never evaluated the cluster; only the decision's full snapshot did.
    assert [s.snapshot_id for s in rig.app.projection.snapshots if s.crowd.clusters] == [
        record.snapshot_id
    ]


def test_no_full_snapshot_is_taken_for_candidates_that_will_not_decide(workdir: Path) -> None:
    rig = Rig(workdir, _at(12, 0), script=[flat("extreme fear, and nothing to do")])
    try:
        rig.toolkit.crypto_fear_greed = 20
        first = rig.loop.tick()
        assert "full_snapshot_riders_0" in first.did
        assert "decision_cycle" in first.did
        # Back to neutral inside the cooldown: emitted, refused, and no full snapshot is taken.
        rig.toolkit.crypto_fear_greed = 50
        rig.clock.set(_at(12, 5))
        second = rig.loop.tick()
        assert "triggers_admitted_0_refused_1" in second.did
        assert not any(d.startswith("full_snapshot") for d in second.did)
        light = [s.snapshot_id for s in rig.app.projection.snapshots if not s.crowd.items]
        assert len(rig.app.projection.snapshots) == 3  # light, full (the decision), light
        assert len(light) == 3
    finally:
        rig.app.close()


def test_feed_health_is_logged_on_the_first_snapshot_and_on_every_decision(rig: Rig) -> None:
    rig.loop.tick()
    rig.clock.set(_at(13, 30))
    rig.loop.tick()
    reports = [FeedHealthReport.model_validate(p) for p in rig.payloads(EventKind.FEED_HEALTH)]
    snapshots = rig.app.projection.snapshots
    assert [r.snapshot_id for r in reports] == [snapshots[0].snapshot_id, snapshots[-1].snapshot_id]
    assert [r.light for r in reports] == [True, False]
    card = rig.loop.last_card
    assert card is not None
    assert card.feed_health == reports[-1]
    # The card built from the ledger alone carries the same report.
    (ledger_card,) = build_cards(rig.app.projection, FileBlobStore(rig.app.paths.blobs))
    assert ledger_card.feed_health == card.feed_health


def _hollow_mood(toolkit: FakeToolkit) -> None:
    def mood() -> tuple[MoodReading, tuple[SourceCall, ...]]:
        now = toolkit.clock.now()
        call = SourceCall(
            call_id=f"hollow-{now.isoformat()}",
            surface=ToolkitSurface.SIGNAL_MCP,
            source="sentiment_index.current",
            health=SourceHealth.HOLLOW,
            started_at=now,
            latency_ms=3,
            rows=0,
            blob=None,
            error='upstream error envelope {"alt_me_error": ""}',
        )
        return MoodReading(market_fear_greed=50, market_fear_greed_source="fake"), (call,)

    toolkit.mood = mood  # type: ignore[method-assign]


def test_a_failing_source_is_an_alarm_in_the_ledger_the_health_file_and_the_status(
    workdir: Path,
) -> None:
    rig = Rig(workdir, _at(12, 0))
    try:
        rig.loop.tick()
        _hollow_mood(rig.toolkit)
        rig.clock.set(_at(12, 5))
        rig.loop.tick()
        rig.clock.set(_at(12, 10))
        rig.loop.tick()  # still hollow: no new event, the streak grows in memory
        reports = [FeedHealthReport.model_validate(p) for p in rig.payloads(EventKind.FEED_HEALTH)]
        assert len(reports) == 2
        raised = reports[-1]
        assert raised.raised == ("sentiment_index.current",)
        (blind,) = raised.failure_blind()
        assert blind.kind is TriggerKind.FEAR_GREED_EXTREME
        assert blind.feeds == ("sentiment_index.current",)
        loop_now = rig.loop.feed_health
        assert loop_now is not None
        assert loop_now.alarms[0].snapshots == 2
        beat = read_health(rig.app.paths.health)
        assert beat is not None
        assert "sentiment_index.current (hollow x2)" in beat.detail
        assert "fear_greed_extreme" in beat.detail
        lines = status_lines(rig.app)
        assert any(line.startswith("feeds as of") and "hollow" in line for line in lines)
        # A restarted loop rebuilds the same streak from the logged snapshots.
        assert RunLoop(rig.app).feed_health == loop_now
    finally:
        rig.app.close()


def test_the_feed_report_is_published(rig: Rig, tmp_path: Path) -> None:
    from sentiment_agent.site.export import export_public, load_json
    from sentiment_agent.site.render import render_site

    rig.loop.tick()
    rig.clock.set(_at(13, 30))
    rig.loop.tick()
    out = tmp_path / "public"
    export_public(
        ledger=rig.app.chain,
        blobs=FileBlobStore(rig.app.paths.blobs),
        out=out,
        arms=(),
        twin=None,
        redteam=None,
        toolkit=(),
        clock=rig.clock,
    )
    feeds = load_json(out / "feeds.json")
    assert feeds["status"] == "logged"
    assert feeds["counts"]["reports"] == 2
    assert feeds["latest"]["light"] is False
    summary = load_json(out / "summary.json")
    assert summary["counts"]["feed_alarms_open"] == 0
    render_site(out)
    index = (out / "index.html").read_text(encoding="utf-8")
    assert 'id="feeds"' in index
    assert "Trigger kinds that could not fire" in index
    card_pages = sorted((out / "cards").glob("*.html"))
    assert card_pages
    assert "Feed health on this snapshot" in card_pages[0].read_text(encoding="utf-8")
    genesis = json.loads((out / "genesis.json").read_text(encoding="utf-8"))
    assert genesis["present"] is False


def test_no_decision_is_spent_after_the_scoring_window(workdir: Path) -> None:
    """run2-a4: from the window's end G2 refuses every opening, so a trigger due then is recorded
    as seen and no model call is made on it."""
    thursday = datetime(2026, 10, 1, 13, 29, tzinfo=UTC)
    rig = Rig(workdir, thursday, script=[flat("never asked")])
    try:
        rig.loop.tick()
        rig.clock.set(thursday.replace(minute=30))
        report = rig.loop.tick()
        assert "window_closed" in report.did
        assert "decision_cycle" not in report.did
        assert rig.payloads(EventKind.DECISION) == []
    finally:
        rig.app.close()
