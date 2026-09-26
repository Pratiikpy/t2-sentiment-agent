"""Health: the heartbeat file, and the status lines read from an app or from a ledger alone."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from runtime.support import FakeWorld, decision, make_parts, target
from sentiment_agent.clock import ManualClock
from sentiment_agent.execution.simulated import SimulatedVenue
from sentiment_agent.policy import POLICY_V1
from sentiment_agent.runtime.health import (
    STALE_AFTER,
    read_health,
    status_from_ledger,
    status_lines,
    write_health,
)
from sentiment_agent.runtime.loop import RunLoop
from sentiment_agent.runtime.wiring import App, build_app
from sentiment_agent.types import Activation, BudgetState, HealthBeat, RunMode

FRIDAY_1329 = datetime(2026, 9, 25, 13, 29, tzinfo=UTC)


def _beat(at: datetime, **fields: object) -> HealthBeat:
    return HealthBeat.model_validate(
        {
            "at": at,
            "iteration": 3,
            "activation": Activation.ACTIVE,
            "open_positions": 1,
            "last_decision_at": None,
            "budget": BudgetState(
                day="2026-09-25", cap_tokens=150_000, spent_tokens=10, calls=1, unreported_calls=0
            ),
            **fields,
        }
    )


def test_a_beat_round_trips_and_leaves_no_temporary_file(tmp_path: Path) -> None:
    path = tmp_path / "var" / "health" / "paper.json"
    beat = _beat(FRIDAY_1329, detail="all well")
    write_health(path, beat)
    assert read_health(path) == beat
    write_health(path, _beat(FRIDAY_1329 + timedelta(seconds=30)))
    assert read_health(path) == _beat(FRIDAY_1329 + timedelta(seconds=30))
    assert [p.name for p in path.parent.iterdir()] == ["paper.json"]


def test_a_missing_or_unreadable_beat_is_none(tmp_path: Path) -> None:
    path = tmp_path / "beat.json"
    assert read_health(path) is None
    path.write_text("{not json", "utf-8")
    assert read_health(path) is None
    path.write_text('{"at": "2026-09-25T13:00:00"}', "utf-8")
    assert read_health(path) is None


def _traded_app(root: Path) -> tuple[ManualClock, App]:
    clock = ManualClock(FRIDAY_1329)
    world = FakeWorld(clock)
    venue = SimulatedVenue(market=world, clock=clock, starting_equity=Decimal("10000"))
    app = build_app(
        root,
        RunMode.SIMULATED,
        llm="scripted",
        clock=clock,
        parts=make_parts(
            clock,
            world=world,
            venue=venue,
            script=[decision("act", [target("NVDAUSDT", 1.0)])],
        ),
    )
    loop = RunLoop(app)
    loop.tick()
    clock.set(FRIDAY_1329 + timedelta(minutes=1))
    loop.tick()
    return clock, app


def test_status_lines_of_a_running_app(workdir: Path) -> None:
    _clock, app = _traded_app(workdir)
    try:
        lines = status_lines(app)
    finally:
        app.close()
    text = "\n".join(lines)
    assert lines[0].startswith("mode simulated:")
    assert "genesis: none (a rehearsal without pre-registration)" in text
    assert "breaker active" in text
    assert "NVDAUSDT qty" in text
    assert "stop " in text
    assert "NO STOP" not in text
    assert "last decision dec-" in text
    assert "decided, act" in text
    assert "Qwen budget 2026-09-25" in text
    assert "last reconciliation" in text
    assert "health: ticking" in text


def test_status_from_the_ledger_alone_and_a_stale_heartbeat(workdir: Path) -> None:
    clock, app = _traded_app(workdir)
    app.close()
    lines = status_from_ledger(
        workdir, RunMode.SIMULATED, clock=clock, policy=POLICY_V1, starting_equity=Decimal("10000")
    )
    text = "\n".join(lines)
    assert "NVDAUSDT qty" in text
    assert "health: ticking" in text
    clock.advance(STALE_AFTER + timedelta(minutes=1))
    stale = "\n".join(
        status_from_ledger(
            workdir,
            RunMode.SIMULATED,
            clock=clock,
            policy=POLICY_V1,
            starting_equity=Decimal("10000"),
        )
    )
    assert "STALE" in stale


def test_status_of_a_mode_without_a_ledger(workdir: Path, clock: ManualClock) -> None:
    lines = status_from_ledger(
        workdir, RunMode.PAPER, clock=clock, policy=POLICY_V1, starting_equity=None
    )
    assert lines == ["paper: no ledger yet (var/ledger/paper.jsonl)"]


def test_the_beat_names_the_disk_and_alarms_when_it_is_low(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil
    from collections import namedtuple

    from sentiment_agent.runtime import health

    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda _p: usage(100 * 1024**3, 0, 40 * 1024**3))
    assert health.disk_note(workdir) == "disk free 40.0 GB"
    monkeypatch.setattr(shutil, "disk_usage", lambda _p: usage(100 * 1024**3, 0, 2 * 1024**3))
    assert "DISK LOW" in health.disk_note(workdir)

    def broken(_p: Path) -> None:
        raise OSError("gone")

    monkeypatch.setattr(shutil, "disk_usage", broken)
    assert health.disk_note(workdir) == "disk free unknown (OSError)"
