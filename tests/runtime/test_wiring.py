"""Wiring: what each mode builds, what it never reads, and the state it rebuilds from the ledger."""

import dataclasses
import io
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from runtime.support import FAKE_COMMIT, FakeBgc, FakeWorld, decision, flat, make_parts, target
from sentiment_agent.clock import ManualClock
from sentiment_agent.execution.bgc import BgcTransport
from sentiment_agent.execution.simulated import SimulatedVenue
from sentiment_agent.ledger.chain import HashChainLedger, ledger_path
from sentiment_agent.policy import POLICY_V1
from sentiment_agent.runtime import wiring
from sentiment_agent.runtime.cli import run_cli
from sentiment_agent.runtime.loop import RunLoop
from sentiment_agent.runtime.wiring import (
    InstanceLock,
    InstanceLocked,
    RefusedToStart,
    archive_ledger,
    build_app,
    frozen_oi_thresholds,
    git_commit,
    lock_hashes,
    scrub_note,
)
from sentiment_agent.types import EventKind, Note, PriceSource, RunMode

FRIDAY_1329 = datetime(2026, 9, 25, 13, 29, tzinfo=UTC)


def _refuse(*_: Any, **__: Any) -> Any:
    raise AssertionError("a credential was read")


def test_simulated_and_dryrun_read_no_credential(
    workdir: Path, clock: ManualClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(wiring, "load_demo_credentials", _refuse)
    monkeypatch.setattr(wiring, "load_qwen_env", _refuse)
    monkeypatch.setattr(wiring, "prove_environment", _refuse)
    simulated = build_app(
        workdir, RunMode.SIMULATED, llm="scripted", clock=clock, parts=make_parts(clock)
    )
    try:
        assert isinstance(simulated.transport, SimulatedVenue)
        assert simulated.proof is None
    finally:
        simulated.close()
    dryrun = build_app(
        workdir,
        RunMode.DRYRUN,
        llm="scripted",
        clock=clock,
        parts=make_parts(clock, bgc_runner=FakeBgc()),
    )
    try:
        assert isinstance(dryrun.transport, BgcTransport)
        assert dryrun.transport.dry_run_only
        assert dryrun.stops is None
        assert dryrun.chain.path == ledger_path(workdir.resolve(), RunMode.DRYRUN)
    finally:
        dryrun.close()


def test_live_qwen_needs_its_key_file(workdir: Path, clock: ManualClock) -> None:
    with pytest.raises(RefusedToStart, match=r"qwen.env"):
        build_app(workdir, RunMode.SIMULATED, llm="live", clock=clock, parts=make_parts(clock))


def test_each_mode_writes_its_own_ledger(workdir: Path, clock: ManualClock) -> None:
    app = build_app(
        workdir, RunMode.SIMULATED, llm="scripted", clock=clock, parts=make_parts(clock)
    )
    try:
        RunLoop(app).tick()
    finally:
        app.close()
    assert ledger_path(workdir, RunMode.SIMULATED).exists()
    assert not ledger_path(workdir, RunMode.PAPER).exists()
    assert {e.mode for e in app.chain.events()} == {RunMode.SIMULATED}


def test_state_is_rebuilt_from_the_ledger_on_restart(workdir: Path) -> None:
    clock = ManualClock(FRIDAY_1329)
    world = FakeWorld(clock)
    venue = SimulatedVenue(market=world, clock=clock, starting_equity=Decimal("10000"))
    script = [decision("act", [target("NVDAUSDT", 1.0), target("BTCUSDT", -1.0)])]
    first = build_app(
        workdir,
        RunMode.SIMULATED,
        llm="scripted",
        clock=clock,
        parts=make_parts(clock, world=world, script=script, venue=venue),
    )
    loop = RunLoop(first)
    loop.tick()
    clock.set(FRIDAY_1329 + timedelta(minutes=1))
    loop.tick()
    demo = world.quotes(PriceSource.DEMO, first.symbols)
    book = first.book(at=clock.now(), demo=demo)
    breaker = first.breaker.state()
    triggers = first.triggers.state()
    budget = first.budget.state()
    orders = {oid: first.tracker.state(oid) for oid in first.tracker.known()}
    first.close()

    clock.advance(timedelta(seconds=30))
    second = build_app(
        workdir,
        RunMode.SIMULATED,
        llm="scripted",
        clock=clock,
        parts=make_parts(clock, world=world, venue=venue),
    )
    try:
        assert second.resumed_from_seq == first.chain.head().seq  # type: ignore[union-attr]
        again = second.book(at=clock.now(), demo=demo)
        assert again.positions == book.positions
        assert again.equity == book.equity
        assert second.breaker.state().activation is breaker.activation
        assert second.triggers.state() == triggers
        assert second.budget.state() == budget
        assert {oid: second.tracker.state(oid) for oid in second.tracker.known()} == orders
        assert second.tracker.fill_ids() == first.tracker.fill_ids()
    finally:
        second.close()


def test_events_appended_by_another_process_are_folded_in(
    workdir: Path, clock: ManualClock
) -> None:
    app = build_app(
        workdir, RunMode.SIMULATED, llm="scripted", clock=clock, parts=make_parts(clock)
    )
    try:
        app.note("first")
        other = HashChainLedger(app.chain.path, mode=RunMode.SIMULATED, clock=clock)
        other.append(EventKind.NOTE, Note(at=clock.now(), author="owner", text="from outside"))
        assert app.sync() == 1
        assert [n.text for n in app.projection.notes] == ["first", "from outside"]
        app.note("after")
        assert [n.text for n in app.projection.notes] == ["first", "from outside", "after"]
        assert app.projection.head_seq == app.chain.head().seq  # type: ignore[union-attr]
    finally:
        app.close()


def test_the_genesis_thresholds_are_used_and_cannot_be_overridden(tmp_path: Path) -> None:
    clock = ManualClock(FRIDAY_1329)
    root = tmp_path / "root"
    root.mkdir()
    parts = make_parts(clock, oi_thresholds={"BTCUSDT": 4.2})
    assert (
        run_cli(
            ["genesis", "--mode", "simulated", "--commit", FAKE_COMMIT],
            root=root,
            clock=clock,
            parts=parts,
            out=io.StringIO(),
        )
        == 0
    )
    with pytest.raises(RefusedToStart, match="frozen at genesis"):
        build_app(root, RunMode.SIMULATED, llm="scripted", clock=clock, parts=parts)
    app = build_app(
        root,
        RunMode.SIMULATED,
        llm="scripted",
        clock=clock,
        parts=dataclasses.replace(parts, oi_thresholds=None),
    )
    try:
        assert app.oi_thresholds == {"BTCUSDT": 4.2}
        assert app.triggers.oi_thresholds == {"BTCUSDT": 4.2}
        assert app.genesis is not None
    finally:
        app.close()


def test_frozen_thresholds_disable_a_series_too_short_to_support_the_quantile() -> None:
    at = FRIDAY_1329
    long = [(at - timedelta(hours=h), 1_000_000.0 * (1 + 0.001 * (h % 7))) for h in range(400)]
    short = [(at - timedelta(hours=h), 1_000_000.0) for h in range(20)]
    record = frozen_oi_thresholds(
        {"BTCUSDT": long, "OTHER": short},
        {"BTCUSDT": "400 points"},
        policy=POLICY_V1,
        at=at,
    )
    assert set(record["thresholds_pct"]) == {"BTCUSDT"}
    assert record["disabled"] == ["OTHER"]
    assert record["points"] == {"BTCUSDT": 400, "OTHER": 20}
    assert record["thresholds_pct"]["BTCUSDT"] > 0


def test_a_simulated_book_is_not_resumed_on_a_fresh_venue_and_can_be_archived(
    workdir: Path,
) -> None:
    clock = ManualClock(FRIDAY_1329)
    world = FakeWorld(clock)
    venue = SimulatedVenue(market=world, clock=clock, starting_equity=Decimal("10000"))
    app = build_app(
        workdir,
        RunMode.SIMULATED,
        llm="scripted",
        clock=clock,
        parts=make_parts(
            clock, world=world, venue=venue, script=[decision("act", [target("NVDAUSDT", 1.0)])]
        ),
    )
    loop = RunLoop(app)
    loop.tick()
    clock.set(FRIDAY_1329 + timedelta(minutes=1))
    loop.tick()
    assert app.held_symbols() == ("NVDAUSDT",)
    app.close()
    with pytest.raises(RefusedToStart, match="does not survive a restart"):
        build_app(
            workdir,
            RunMode.SIMULATED,
            llm="scripted",
            clock=clock,
            parts=make_parts(clock, world=world),
        )
    archived = archive_ledger(workdir, RunMode.SIMULATED, clock)
    assert archived is not None
    assert archived.exists()
    assert archived.with_name(archived.name + ".head").exists()
    fresh = build_app(
        workdir,
        RunMode.SIMULATED,
        llm="scripted",
        clock=clock,
        parts=make_parts(clock, world=world),
    )
    fresh.close()
    with pytest.raises(RefusedToStart, match="never archived"):
        archive_ledger(workdir, RunMode.PAPER, clock)


def test_a_held_symbol_without_a_quote_is_valued_at_the_newest_logged_mark(
    workdir: Path,
) -> None:
    clock = ManualClock(FRIDAY_1329)
    world = FakeWorld(clock)
    venue = SimulatedVenue(market=world, clock=clock, starting_equity=Decimal("10000"))
    app = build_app(
        workdir,
        RunMode.SIMULATED,
        llm="scripted",
        clock=clock,
        parts=make_parts(
            clock,
            world=world,
            venue=venue,
            script=[decision("act", [target("NVDAUSDT", 1.0)]), flat()],
        ),
    )
    try:
        loop = RunLoop(app)
        loop.tick()
        clock.set(FRIDAY_1329 + timedelta(minutes=1))
        loop.tick()
        snapshot_mark = app.latest_snapshot().demo_quotes["NVDAUSDT"].mark  # type: ignore[union-attr]
        book = app.book(at=clock.now(), demo={})
        assert book.marks["NVDAUSDT"] == snapshot_mark
        world.fail_quotes = True
        assert app.quotes() == ({}, {})
    finally:
        app.close()


def test_the_instance_lock_names_its_holder_and_is_released_on_close(
    workdir: Path, clock: ManualClock
) -> None:
    path = workdir / "var" / "run" / "x.lock"
    first = InstanceLock(path, mode=RunMode.PAPER, clock=clock)
    first.acquire()
    assert first.held
    second = InstanceLock(path, mode=RunMode.PAPER, clock=clock)
    with pytest.raises(InstanceLocked) as caught:
        second.acquire()
    assert "pid" in str(caught.value)
    first.release()
    second.acquire()
    second.release()


def test_git_commit_is_read_from_the_checkout_without_running_git(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    git = root / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    assert git_commit(root) is None
    (git / "HEAD").write_text("ref: refs/heads/main\n", "utf-8")
    assert git_commit(root) is None
    (git / "packed-refs").write_text(f"# pack-refs\n{FAKE_COMMIT} refs/heads/main\n", "utf-8")
    assert git_commit(root) == FAKE_COMMIT
    other = "f" * 40
    (git / "refs" / "heads" / "main").write_text(other + "\n", "utf-8")
    assert git_commit(root) == other
    (git / "HEAD").write_text(FAKE_COMMIT.upper() + "\n", "utf-8")
    assert git_commit(root) == FAKE_COMMIT
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    (worktree / ".git").write_text(f"gitdir: {git}\n", "utf-8")
    assert git_commit(worktree) == FAKE_COMMIT


def test_lock_hashes_cover_the_lockfiles_present(tmp_path: Path) -> None:
    (tmp_path / "uv.lock").write_text("lock", "utf-8")
    hashes = lock_hashes(tmp_path)
    assert list(hashes) == ["uv.lock"]
    assert len(hashes["uv.lock"]) == 64


def test_a_note_never_carries_a_local_path_or_a_credential_variable_name(
    workdir: Path, clock: ManualClock
) -> None:
    raw = (
        f"tick failed: FileNotFoundError: [Errno 2] no such file: {workdir / 'var' / 'x.json'}; "
        r"also C:\Users\someone\secret.txt, /home/someone/x, file:///c:/x and "
        "BITGET_API_KEY is not set; https://api.bitget.com/api/v3/market/tickers stays"
    )
    clean = scrub_note(raw, workdir)
    assert str(workdir) not in clean
    assert "someone" not in clean
    assert "file:/" not in clean
    assert "BITGET_" not in clean
    assert "BITGET-API_KEY" in clean
    assert "https://api.bitget.com/api/v3/market/tickers" in clean
    app = build_app(
        workdir, RunMode.SIMULATED, llm="scripted", clock=clock, parts=make_parts(clock)
    )
    try:
        event = app.note(raw)
        assert event.payload["text"] == scrub_note(raw, workdir.resolve())
        assert str(workdir) not in event.payload["text"]
    finally:
        app.close()
