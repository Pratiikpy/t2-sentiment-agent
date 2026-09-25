"""The trigger replay behind run 2's declared counts (events/replay.py)."""

from datetime import datetime, timedelta

from events.test_triggers import cluster, features, snapshot, utc
from helpers import empty_book
from sentiment_agent.events.replay import SAME_TICK, funding_extremes, replay_admissions
from sentiment_agent.llm.budget import MEASURED_PROMPT_BYTES, decision_bound
from sentiment_agent.perception.snapshot import LIGHT_SNAPSHOT_REASON
from sentiment_agent.policy import POLICY_V1, POLICY_V2
from sentiment_agent.types import (
    BookState,
    PerceptionSnapshot,
    PositioningFeatures,
    SourceCall,
    SourceHealth,
    ToolkitSurface,
)

FIVE = timedelta(minutes=5)
START = utc(2026, 9, 24, 17, 0)  # a Thursday


def light(at: datetime, *feats: PositioningFeatures) -> PerceptionSnapshot:
    snap = snapshot(at, feats=list(feats))
    marker = SourceCall(
        call_id=f"off-{at.isoformat()}",
        surface=ToolkitSurface.CROWD_X,
        source="crowd.collect",
        params={"reason": LIGHT_SNAPSHOT_REASON},
        health=SourceHealth.DISABLED,
        started_at=at,
        latency_ms=0,
        rows=0,
        blob=None,
    )
    return snap.model_copy(update={"source_calls": (marker,)})


def book_at(snapshot: PerceptionSnapshot) -> BookState:
    return empty_book(at=snapshot.taken_at)


def nvda_extreme_every_five_minutes(hours: int) -> list[PerceptionSnapshot]:
    return [
        light(START + i * FIVE, features("NVDAUSDT", funding_z_live=3.0)) for i in range(hours * 12)
    ]


def test_policy_v1_ignores_an_equity_funding_extreme_and_v2_wakes_once_per_cooldown() -> None:
    snaps = nvda_extreme_every_five_minutes(9)
    v1 = replay_admissions(snaps, policy=POLICY_V1, oi_thresholds={}, start=START, book_at=book_at)
    assert v1.event_decisions() == 0
    v2 = replay_admissions(snaps, policy=POLICY_V2, oi_thresholds={}, start=START, book_at=book_at)
    # 17:00, 21:00, then 01:00 on the next UTC day: 240 minutes apart, the 00:00 heartbeat aside.
    times = [b.at for b in v2.batches if b.event_decision]
    assert times == [START, START + timedelta(hours=4), START + timedelta(hours=8)]
    summary = v2.summary()
    assert summary["event_decisions"] == 3
    assert summary["event_decisions_by_day"] == {"2026-09-24": 2, "2026-09-25": 1}
    assert summary["heartbeat_decisions"] == 1  # the 00:00 funding heartbeat, inside 9 hours
    assert summary["admitted_by_kind"] == {"funding_zscore": 3, "heartbeat_funding": 1}
    assert summary["snapshots"] == 108
    assert funding_extremes(snaps, threshold=2.0) == {"NVDAUSDT": 108}


def test_a_full_snapshot_in_the_same_tick_joins_the_light_one() -> None:
    at = START + timedelta(hours=7)  # 00:00 UTC: the funding heartbeat
    before = light(at - FIVE)
    tick_light = light(at + timedelta(seconds=10))
    full = snapshot(at + timedelta(seconds=40), clusters=[cluster("c1", ["NVDAUSDT"], at=at)])
    later = snapshot(at + 2 * SAME_TICK + timedelta(seconds=40))
    result = replay_admissions(
        [before, tick_light, full, later],
        policy=POLICY_V2,
        oi_thresholds={},
        start=START,
        book_at=book_at,
    )
    batches = result.batches
    assert [b.members for b in batches] == [1, 2, 1]
    joined = batches[1]
    assert sorted(t.kind.value for t in joined.admitted) == [
        "coordinated_cluster",
        "heartbeat_funding",
    ]
    assert not joined.event_decision
    assert result.event_decisions() == 0


def test_the_budget_floor_refuses_what_the_worst_case_cannot_carry() -> None:
    snaps = nvda_extreme_every_five_minutes(9)
    huge = decision_bound(MEASURED_PROMPT_BYTES, POLICY_V2) * 20
    result = replay_admissions(
        snaps,
        policy=POLICY_V2,
        oi_thresholds={},
        start=START,
        book_at=book_at,
        decision_bound_tokens=huge,
    )
    assert result.event_decisions() == 3
    assert result.event_decisions(with_budget=True) == 0
