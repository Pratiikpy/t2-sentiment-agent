"""The runtime refuses before it can send, and a restart never sends twice.

Every PAPER precondition is exercised through the real wiring and the real CLI: no genesis, a
policy other than the pre-registered one, no Demo credential file, an environment proof that does
not pass, 40099 on a Demo read and on a send (exit code 3), a second instance of the same mode, and
a crash between the venue accepting an order and the ledger hearing about it.
"""

import dataclasses
import io
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from runtime.support import (
    FAKE_COMMIT,
    FakeBgc,
    FakeToolkit,
    FakeWorld,
    decision,
    make_parts,
    paper_project,
    target,
)
from sentiment_agent.clock import ManualClock
from sentiment_agent.execution.environment import EnvironmentRefused
from sentiment_agent.execution.simulated import SimulatedVenue
from sentiment_agent.ledger.chain import GenesisError, ledger_path
from sentiment_agent.llm.fakes import ScriptedChatModel
from sentiment_agent.policy import POLICY_V1
from sentiment_agent.runtime.cli import (
    EXIT_ENVIRONMENT,
    EXIT_LOCKED,
    EXIT_OK,
    EXIT_PREREGISTRATION,
    run_cli,
)
from sentiment_agent.runtime.loop import RunLoop
from sentiment_agent.runtime.wiring import InstanceLocked, Parts, build_app, open_chain
from sentiment_agent.types import (
    ApprovedOrder,
    EnvironmentProof,
    EventKind,
    OrderState,
    PriceSource,
    RunMode,
    VenueAck,
    VenueRejection,
    VenueUnknown,
)

GENESIS_AT = datetime(2026, 9, 25, 13, 20, tzinfo=UTC)
"""A Friday, ten minutes before the US open heartbeat."""


def _cli(root: Path, clock: ManualClock, parts: Parts, *argv: str) -> tuple[int, str]:
    out = io.StringIO()
    code = run_cli(list(argv), root=root, clock=clock, parts=parts, out=out)
    return code, out.getvalue()


def _paper_genesis(root: Path, clock: ManualClock) -> None:
    """Prove the Demo key and write the paper genesis, as the owner does (DESIGN.md §16)."""
    parts = make_parts(clock, bgc_runner=FakeBgc(), oi_thresholds={"BTCUSDT": 5.0})
    code, text = _cli(root, clock, parts, "prove-env")
    assert code == EXIT_OK, text
    code, text = _cli(root, clock, parts, "genesis", "--mode", "paper", "--commit", FAKE_COMMIT)
    assert code == EXIT_OK, text
    assert "Genesis hash:" in text


def _paper_parts(clock: ManualClock, bgc: FakeBgc, *, script: list[Any] | None = None) -> Parts:
    model = ScriptedChatModel(script or [])
    return make_parts(clock, bgc_runner=bgc, chat_model=model)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return paper_project(tmp_path / "project")


@pytest.fixture
def paper_clock() -> ManualClock:
    return ManualClock(GENESIS_AT)


# ================================================================================================
# PAPER preconditions
# ================================================================================================


def test_paper_without_a_genesis_refuses_and_logs_nothing(
    root: Path, paper_clock: ManualClock
) -> None:
    parts = _paper_parts(paper_clock, FakeBgc())
    with pytest.raises(GenesisError, match="no genesis"):
        build_app(root, RunMode.PAPER, llm="live", clock=paper_clock, parts=parts)
    paper = ledger_path(root, RunMode.PAPER)
    assert not paper.exists() or paper.stat().st_size == 0
    code, text = _cli(root, paper_clock, parts, "once", "--mode", "paper")
    assert code == EXIT_PREREGISTRATION
    assert "no genesis" in text


def test_a_policy_other_than_the_pre_registered_one_refuses_before_anything_is_logged(
    root: Path, paper_clock: ManualClock
) -> None:
    _paper_genesis(root, paper_clock)
    head_before = open_chain(root, RunMode.PAPER, paper_clock).head()
    tuned = POLICY_V1.model_copy(update={"stop_loss_pct": 0.05})
    parts = dataclasses.replace(_paper_parts(paper_clock, FakeBgc()), policy=tuned)
    with pytest.raises(GenesisError, match="policy in force"):
        build_app(root, RunMode.PAPER, llm="live", clock=paper_clock, parts=parts)
    code, text = _cli(root, paper_clock, parts, "once", "--mode", "paper")
    assert code == EXIT_PREREGISTRATION
    assert "no order is sent under an unregistered policy" in text
    assert open_chain(root, RunMode.PAPER, paper_clock).head() == head_before


def test_a_missing_demo_credential_file_refuses_with_exit_3(
    root: Path, paper_clock: ManualClock
) -> None:
    _paper_genesis(root, paper_clock)
    (root / ".secrets" / "demo.env").unlink()
    parts = _paper_parts(paper_clock, FakeBgc())
    with pytest.raises(EnvironmentRefused, match="does not exist"):
        build_app(root, RunMode.PAPER, llm="live", clock=paper_clock, parts=parts)
    code, text = _cli(root, paper_clock, parts, "once", "--mode", "paper")
    assert code == EXIT_ENVIRONMENT
    assert "ENVIRONMENT REFUSED" in text
    assert "No order was sent" in text


def test_a_proof_that_does_not_pass_is_logged_and_refuses(
    root: Path, paper_clock: ManualClock
) -> None:
    _paper_genesis(root, paper_clock)
    inconclusive = FakeBgc(live="loopback_live_negative_param")
    parts = _paper_parts(paper_clock, inconclusive)
    with pytest.raises(EnvironmentRefused, match="did not pass"):
        build_app(root, RunMode.PAPER, llm="live", clock=paper_clock, parts=parts)
    head = open_chain(root, RunMode.PAPER, paper_clock).head()
    assert head is not None
    assert head.kind is EventKind.ENVIRONMENT_PROOF
    proof = EnvironmentProof.model_validate(head.payload)
    assert not proof.passed
    assert any("inconclusive" in r for r in proof.reasons)
    assert inconclusive.sends() == []


def test_40099_on_a_demo_read_exits_3_with_the_failed_proof_logged(
    root: Path, paper_clock: ManualClock
) -> None:
    _paper_genesis(root, paper_clock)
    live_key = FakeBgc(demo_overview="loopback_demo_overview_40099")
    code, text = _cli(
        root, paper_clock, _paper_parts(paper_clock, live_key), "once", "--mode", "paper"
    )
    assert code == EXIT_ENVIRONMENT
    assert "not a Demo key" in text
    head = open_chain(root, RunMode.PAPER, paper_clock).head()
    assert head is not None
    assert head.kind is EventKind.ENVIRONMENT_PROOF
    assert EnvironmentProof.model_validate(head.payload).demo_read_code == "40099"
    assert live_key.sends() == []


def test_40099_on_a_send_exits_3_after_logging_the_rejection(
    root: Path, paper_clock: ManualClock
) -> None:
    _paper_genesis(root, paper_clock)
    bgc = FakeBgc(send="loopback_place_40099")
    script = [decision("act", [target("NVDAUSDT", 1.0), target("AAPLUSDT", 1.0)])]
    paper_clock.set(GENESIS_AT + timedelta(minutes=11))
    parts = dataclasses.replace(
        _paper_parts(paper_clock, bgc, script=script),
        market=FakeWorld(paper_clock),
        toolkit=FakeToolkit(paper_clock),
    )
    code, text = _cli(root, paper_clock, parts, "once", "--mode", "paper")
    assert code == EXIT_ENVIRONMENT, text
    assert "40099" in text or "not a Demo key" in text
    chain = open_chain(root, RunMode.PAPER, paper_clock)
    rejected = list(chain.events(frozenset({EventKind.ORDER_REJECTED})))
    assert len(rejected) == 1
    assert rejected[0].payload["code"] == "40099"
    assert len(bgc.sends()) == 1, "nothing is sent after the first 40099"
    assert not list(chain.events(frozenset({EventKind.FILL})))
    # Every argv the transport built carries --paper-trading; the only one without is the
    # environment proof's read-only live probe.
    for argv in bgc.calls:
        assert ("--paper-trading" in argv) != ("--read-only" in argv)


# ================================================================================================
# One instance per mode
# ================================================================================================


def test_a_second_instance_of_a_mode_is_refused(workdir: Path, clock: ManualClock) -> None:
    parts = make_parts(clock)
    first = build_app(workdir, RunMode.SIMULATED, llm="scripted", clock=clock, parts=parts)
    try:
        with pytest.raises(InstanceLocked, match="another simulated instance"):
            build_app(workdir, RunMode.SIMULATED, llm="scripted", clock=clock, parts=parts)
        code, text = _cli(workdir, clock, parts, "once", "--mode", "simulated", "--llm", "recorded")
        assert code == EXIT_LOCKED
        assert "another simulated instance is running" in text
        # Another mode is a separate lock and a separate ledger.
        dryrun_parts = make_parts(clock, bgc_runner=FakeBgc())
        other = build_app(workdir, RunMode.DRYRUN, llm="scripted", clock=clock, parts=dryrun_parts)
        other.close()
    finally:
        first.close()
    again = build_app(workdir, RunMode.SIMULATED, llm="scripted", clock=clock, parts=parts)
    again.close()


# ================================================================================================
# Crash between the send and the answer
# ================================================================================================


class DropsTheFirstAnswer(SimulatedVenue):
    """A venue that accepts the first order it is sent and then loses the connection before the
    answer arrives: the order exists at the venue, and the process never heard."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.drop_next = True
        self.sent: list[str] = []

    def place(self, order: ApprovedOrder) -> VenueAck | VenueRejection | VenueUnknown:
        self.sent.append(order.intent.client_oid)
        answer = super().place(order)
        if self.drop_next:
            self.drop_next = False
            raise ConnectionResetError("the connection dropped after the order was sent")
        return answer


def test_a_crash_after_the_send_resolves_from_the_ledger_and_never_sends_again(
    workdir: Path,
) -> None:
    clock = ManualClock(datetime(2026, 9, 25, 13, 29, tzinfo=UTC))
    world = FakeWorld(clock)
    venue = DropsTheFirstAnswer(market=world, clock=clock, starting_equity=Decimal("10000"))
    script = [decision("act", [target("NVDAUSDT", 1.0), target("AAPLUSDT", 1.0)])]
    parts = make_parts(clock, world=world, script=script, venue=venue)
    app = build_app(workdir, RunMode.SIMULATED, llm="scripted", clock=clock, parts=parts)
    loop = RunLoop(app)
    loop.tick()
    clock.set(datetime(2026, 9, 25, 13, 30, 5, tzinfo=UTC))
    with pytest.raises(ConnectionResetError):
        loop.tick()
    first_oid = venue.sent[0]
    assert app.tracker.state(first_oid) is OrderState.UNKNOWN
    unknown = list(app.chain.events(frozenset({EventKind.ORDER_UNKNOWN})))
    assert [e.payload["client_oid"] for e in unknown] == [first_oid]
    app.close()

    # The process restarts; the venue kept the order it filled.
    clock.advance(timedelta(minutes=3))
    restarted = build_app(
        workdir,
        RunMode.SIMULATED,
        llm="scripted",
        clock=clock,
        parts=make_parts(clock, world=world, venue=venue),
    )
    assert restarted.tracker.state(first_oid) is OrderState.UNKNOWN
    report = RunLoop(restarted).tick()
    assert "reconcile_full" in report.did
    assert restarted.tracker.state(first_oid) is OrderState.FILLED
    assert venue.sent.count(first_oid) == 1, "an order whose outcome was unknown is never resent"
    reconciliations = restarted.projection.reconciliations
    assert any(first_oid in r.resolved_unknown for r in reconciliations)
    book = restarted.book(at=clock.now(), demo=world.quotes(PriceSource.DEMO, ["NVDAUSDT"]))
    venue_qty = {p.symbol: p.qty for p in venue.positions()}
    assert {s: p.qty for s, p in book.positions.items()} == venue_qty
    restarted.close()
