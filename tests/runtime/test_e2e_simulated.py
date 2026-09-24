"""A full simulated trading day, end to end, through the real runtime.

Friday 2026-09-25 from 12:50 UTC, then a restart on Monday 2026-09-28 after the process was down
over the weekend. Everything is the production code path; only the outside world is fake (keyless
market data, the two Bitget services, the crowd, the chat model's answers, the ``ots`` calendar)
and the venue is :class:`SimulatedVenue`. What happens, and what is asserted:

* 12:40  ``t2sa genesis --mode simulated``: a rehearsal pre-registration, stamped.
* 13:30  the US-open heartbeat wakes the model. It asks for eight longs; the kernel refuses MSTR
         (Demo mark 5% off its index, G1) and GOOGL (a figure it cannot ground, G9), and shares the
         25% gross cap across the other six (G3). Six orders fill, each with its venue stop.
* 14:10  NVDA gaps down 6%: its venue stop fires on the next poll, and reconciliation books it.
* 14:40  the AAPL Demo mark departs 5% from its index: the protective check exits AAPL (G1).
* 15:00  crypto Fear & Greed crosses into extreme fear: an event decision; the model service
         fails; the kernel flattens the book (``llm_outage``), and the breaker halts.
* 16:00  the funding heartbeat: a valid decision lifts the outage halt and opens four legs.
* 19:46  weekend pre-flatten: the US legs are closed before the freeze (G2); BTC stays.
* hourly marks through 21:00, then the process stops.
* Monday 00:01 the restart rebuilds everything from the ledger, catches up the seven heartbeats it
         missed in one decision, and opens five legs.
* 00:30  four equity legs gap 12% through their stops; the day's loss passes 1.5%, and the daily
         kill flattens BTC (G5) and halts the breaker.
* then: export, ``scripts/recompute.py`` on the export, ``t2sa verify``, and ``t2sa replay`` of the
  first decision, which must reproduce its ruling id and every intent id.
"""

import dataclasses
import io
import json
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from runtime.support import (
    FAKE_COMMIT,
    FakeOts,
    FakeToolkit,
    FakeWorld,
    decision,
    make_parts,
    target,
)
from sentiment_agent.clock import ManualClock
from sentiment_agent.execution.simulated import SimulatedVenue
from sentiment_agent.llm.client import QwenTransportError
from sentiment_agent.runtime.cli import read_events, replay_decision, run_cli
from sentiment_agent.runtime.loop import RunLoop
from sentiment_agent.runtime.wiring import App, Parts, build_app
from sentiment_agent.types import (
    Activation,
    DecisionCard,
    EventKind,
    GuardId,
    GuardStatus,
    KernelRuling,
    LlmOutcome,
    MarkPoint,
    OrderPlan,
    OrderSubmitted,
    ProtectiveAction,
    ProtectiveReason,
    RunMode,
    Trigger,
    TriggerKind,
)

RECOMPUTE = Path(__file__).resolve().parents[2] / "scripts" / "recompute.py"
FRIDAY = datetime(2026, 9, 25, tzinfo=UTC)
MONDAY = datetime(2026, 9, 28, tzinfo=UTC)
EQUITY_LEGS = ("NVDAUSDT", "AAPLUSDT", "METAUSDT", "AMZNUSDT", "TSLAUSDT", "SP500USDT")


def _at(day: datetime, hh: int, mm: int) -> datetime:
    return day.replace(hour=hh, minute=mm)


def _run(
    loop: RunLoop,
    clock: ManualClock,
    until: datetime,
    events: dict[datetime, Callable[[], object]],
    cards: list[DecisionCard],
) -> None:
    """Tick every two minutes until ``until``, firing each scheduled world event when its time
    comes; collect every decision card."""
    while clock.now() <= until:
        for when in sorted(events):
            if clock.now() >= when:
                events.pop(when)()
        loop.tick()
        card = loop.last_card
        if card is not None and (not cards or cards[-1] is not card):
            cards.append(card)
        clock.advance(timedelta(minutes=2))


@pytest.fixture(scope="module")
def day(tmp_path_factory: pytest.TempPathFactory) -> "Day":
    return Day.run(tmp_path_factory.mktemp("e2e"))


@dataclasses.dataclass
class Day:
    root: Path
    clock: ManualClock
    app: App
    cards: list[DecisionCard]
    venue: SimulatedVenue
    parts: Parts
    ots: FakeOts
    genesis_output: str

    @classmethod
    def run(cls, root: Path) -> "Day":
        (root / "pyproject.toml").write_text('[project]\nname = "t2-sentiment-agent"\n', "utf-8")
        clock = ManualClock(_at(FRIDAY, 12, 40))
        world = FakeWorld(clock)
        toolkit = FakeToolkit(clock)
        ots = FakeOts()
        venue = SimulatedVenue(market=world, clock=clock, starting_equity=Decimal("10000"))
        world.mark_skew["MSTRUSDT"] = Decimal("0.05")
        friday_script = [
            decision(
                "act",
                [
                    *(target(s, 1.0) for s in EQUITY_LEGS),
                    target("GOOGLUSDT", 1.0, thesis="GOOGLUSDT trades at 999.99 and must rise"),
                    target("MSTRUSDT", 1.0),
                ],
            ),
            QwenTransportError("scripted outage: the gateway is down"),
            decision(
                "act",
                [
                    target("METAUSDT", 1.0),
                    target("AMZNUSDT", 1.0),
                    target("NDX100USDT", 1.0),
                    target("BTCUSDT", 1.0),
                ],
            ),
        ]
        genesis_parts = make_parts(
            clock,
            world=world,
            toolkit=toolkit,
            venue=venue,
            ots=ots,
            oi_thresholds={"BTCUSDT": 5.0},
        )
        out = io.StringIO()
        code = run_cli(
            ["genesis", "--mode", "simulated", "--commit", FAKE_COMMIT],
            root=root,
            clock=clock,
            parts=genesis_parts,
            out=out,
        )
        assert code == 0, out.getvalue()

        cards: list[DecisionCard] = []
        clock.set(_at(FRIDAY, 12, 50))
        parts = make_parts(
            clock, world=world, toolkit=toolkit, script=friday_script, venue=venue, ots=ots
        )
        app = build_app(root, RunMode.SIMULATED, llm="scripted", clock=clock, parts=parts)
        friday: dict[datetime, Callable[[], object]] = {
            _at(FRIDAY, 14, 10): lambda: world.move("NVDAUSDT", "-6"),
            _at(FRIDAY, 14, 40): lambda: world.mark_skew.__setitem__("AAPLUSDT", Decimal("0.05")),
            _at(FRIDAY, 14, 46): lambda: world.mark_skew.pop("AAPLUSDT"),
            # The legs the outage will close are in profit, so the closes do not all lose.
            _at(FRIDAY, 14, 50): lambda: [
                world.move(s, "1.5") for s in ("METAUSDT", "AMZNUSDT", "TSLAUSDT", "SP500USDT")
            ],
            _at(FRIDAY, 15, 0): lambda: setattr(toolkit, "crypto_fear_greed", 20),
        }
        _run(RunLoop(app), clock, _at(FRIDAY, 21, 2), friday, cards)
        app.close()

        # The weekend: the process is down; BTC drifts up 1%.
        world.move("BTCUSDT", "1")
        clock.set(_at(MONDAY, 0, 1))
        monday_script = [
            decision(
                "act",
                [
                    target(s, 1.0)
                    for s in ("BTCUSDT", "NVDAUSDT", "AAPLUSDT", "METAUSDT", "AMZNUSDT")
                ],
            )
        ]
        parts = make_parts(
            clock, world=world, toolkit=toolkit, script=monday_script, venue=venue, ots=ots
        )
        app = build_app(root, RunMode.SIMULATED, llm="scripted", clock=clock, parts=parts)
        monday: dict[datetime, Callable[[], object]] = {
            _at(MONDAY, 0, 30): lambda: [
                world.move(s, "-12") for s in ("NVDAUSDT", "AAPLUSDT", "METAUSDT", "AMZNUSDT")
            ],
        }
        _run(RunLoop(app), clock, _at(MONDAY, 1, 6), monday, cards)
        return cls(root, clock, app, cards, venue, parts, ots, out.getvalue())

    def events(self, kind: EventKind) -> list[dict[str, object]]:
        return [e.payload for e in self.app.chain.events(frozenset({kind}))]


def _protective(day: Day) -> list[ProtectiveAction]:
    return list(day.app.projection.protective_actions)


# ================================================================================================
# The day
# ================================================================================================


def test_genesis_was_written_first_with_its_thresholds_and_stamped(day: Day) -> None:
    events = list(day.app.chain.events())
    assert events[0].kind is EventKind.GENESIS
    assert events[1].kind is EventKind.ANCHOR
    assert day.app.oi_thresholds == {"BTCUSDT": 5.0}
    assert "open-interest thresholds frozen: BTCUSDT 5.0000%" in day.genesis_output
    assert any(call[1] == "stamp" for call in day.ots.calls)


def test_heartbeats_and_events_woke_the_model(day: Day) -> None:
    triggers = [Trigger.model_validate(p) for p in day.events(EventKind.TRIGGER)]
    kinds = [t.kind for t in triggers]
    assert TriggerKind.HEARTBEAT_US_OPEN in kinds
    assert TriggerKind.FEAR_GREED_EXTREME in kinds
    funding = [t for t in triggers if t.kind is TriggerKind.HEARTBEAT_FUNDING]
    # Friday 16:00, then the six missed over the weekend and Monday 00:00, caught up at once.
    assert len(funding) == 1 + 7
    monday = day.cards[-1]
    assert monday.at == _at(MONDAY, 0, 1)
    assert len(monday.triggers) == 7


def test_every_decision_is_logged_with_its_outcome(day: Day) -> None:
    outcomes = [d.outcome for d in day.app.projection.decisions]
    assert outcomes == [
        LlmOutcome.DECIDED,
        LlmOutcome.TRANSPORT_ERROR,
        LlmOutcome.DECIDED,
        LlmOutcome.DECIDED,
    ]
    assert [c.outcome for c in day.cards] == outcomes
    budget = day.app.projection.budget_states
    assert len(budget) == 4


def test_the_kernel_refused_and_approved_on_the_first_decision(day: Day) -> None:
    first = day.cards[0]
    assert first.kernel is not None
    ruling = first.kernel
    mstr = ruling.instrument("MSTRUSDT")
    assert mstr is not None
    assert mstr.approved_weight == 0.0
    assert mstr.binding_guard is GuardId.G1_VENUE_INTEGRITY
    googl = ruling.instrument("GOOGLUSDT")
    assert googl is not None
    assert googl.approved_weight == 0.0
    g9 = next(r for r in googl.rulings if r.guard is GuardId.G9_GROUNDING)
    assert g9.status is GuardStatus.FIRED
    approved = {i.symbol: i.approved_weight for i in ruling.instruments if i.approved_weight}
    assert set(approved) == set(EQUITY_LEGS)
    assert sum(approved.values()) == pytest.approx(0.25)
    assert all(i.binding_guard is GuardId.G3_SIZE for i in ruling.instruments if i.approved_weight)
    assert {o.symbol for o in first.orders} == set(EQUITY_LEGS)
    assert all(o.fills for o in first.orders)
    assert all(p.argv[-1] == "--dry-run" for p in first.previews)


def test_only_a_decided_model_decision_ever_adds_exposure(day: Day) -> None:
    decided = {
        d.decision_id for d in day.app.projection.decisions if d.outcome is LlmOutcome.DECIDED
    }
    submitted = [OrderSubmitted.model_validate(p) for p in day.events(EventKind.ORDER_SUBMITTED)]
    assert submitted
    for order in submitted:
        if order.intent.purpose.adds_exposure:
            assert order.intent.decision_id in decided
            assert order.intent.stop_loss_price is not None
        else:
            assert order.intent.reduce_only


def test_a_venue_stop_fired_between_decisions_and_was_booked(day: Day) -> None:
    fills = day.app.projection.fills
    stop_fills = [f for f in fills if f.symbol == "NVDAUSDT" and f.client_oid is None]
    assert stop_fills, "the NVDA stop fill came from the venue, not from an order we sent"
    assert stop_fills[0].executed_at == _at(FRIDAY, 14, 10)
    trades = [t for t in day.app.projection.closed_trades if t.symbol == "NVDAUSDT"]
    assert trades[0].closed_at == _at(FRIDAY, 14, 10)
    assert trades[0].net_pnl < 0


def test_protective_actions_venue_integrity_outage_weekend_and_kill(day: Day) -> None:
    by_reason = {p.reason: p for p in _protective(day)}
    assert by_reason[ProtectiveReason.VENUE_INTEGRITY].symbols == ("AAPLUSDT",)
    assert by_reason[ProtectiveReason.VENUE_INTEGRITY].at == _at(FRIDAY, 14, 40)
    outage = by_reason[ProtectiveReason.LLM_OUTAGE]
    assert set(outage.symbols) == {"AMZNUSDT", "METAUSDT", "SP500USDT", "TSLAUSDT"}
    weekend = by_reason[ProtectiveReason.WEEKEND_FREEZE]
    assert set(weekend.symbols) == {"AMZNUSDT", "METAUSDT", "NDX100USDT"}
    assert weekend.at == _at(FRIDAY, 19, 46)
    kill = by_reason[ProtectiveReason.DAILY_KILL]
    assert kill.symbols == ("BTCUSDT",)
    assert kill.at.date() == MONDAY.date()
    rulings = {r.ruling_id: r for r in day.app.projection.rulings}
    kill_ruling: KernelRuling = rulings[kill.ruling_id]
    btc = kill_ruling.instrument("BTCUSDT")
    assert btc is not None
    assert btc.binding_guard is GuardId.G5_DAILY_KILL


def test_the_breaker_halted_on_the_outage_recovered_and_halted_on_the_kill(day: Day) -> None:
    moves = [(t.from_state, t.to_state, t.trips) for t in day.app.projection.breaker_transitions]
    assert (Activation.ACTIVE, Activation.HALTED, ("llm_outage",)) in moves
    assert (Activation.HALTED, Activation.ACTIVE, ()) in moves
    assert any(to is Activation.HALTED and "daily_kill" in trips for _, to, trips in moves)
    assert day.app.breaker.state().activation is Activation.HALTED
    assert day.app.held_symbols() == ()


def test_hourly_marks_on_the_grid_and_the_book_equals_the_venue(day: Day) -> None:
    marks = [MarkPoint.model_validate(p).at for p in day.events(EventKind.MARK)]
    friday = [_at(FRIDAY, h, 0) for h in range(12, 22)]  # 12:00 is the anchor, flat
    monday = [_at(MONDAY, 0, 0), _at(MONDAY, 1, 0)]
    assert marks == friday + monday
    reports = day.app.projection.reconciliations
    assert reports[-1].clean
    assert day.venue.positions() == []


def test_every_submitted_order_was_previewed_and_answered(day: Day) -> None:
    previews = {p["client_oid"] for p in day.events(EventKind.ORDER_PREVIEW)}
    acks = {p["client_oid"] for p in day.events(EventKind.ORDER_ACK)}
    submitted = {p["client_oid"] for p in day.events(EventKind.ORDER_SUBMITTED)}
    assert submitted
    assert submitted <= previews
    assert submitted == acks
    plans = [OrderPlan.model_validate(p) for p in day.events(EventKind.ORDER_PLAN)]
    planned = {i.client_oid for plan in plans for i in plan.intents}
    assert submitted <= planned


# ================================================================================================
# Publish, recompute, verify, replay
# ================================================================================================


@pytest.fixture(scope="module")
def exported(day: Day) -> Path:
    day.app.close()
    out = io.StringIO()
    code = run_cli(
        ["export", "--mode", "simulated", "--out", "public-sim", "--coin-flips", "10"],
        root=day.root,
        clock=day.clock,
        parts=day.parts,
        out=out,
    )
    assert code == 0, out.getvalue()
    return day.root / "public-sim"


def test_the_export_is_complete_and_recompute_passes(exported: Path) -> None:
    for name in (
        "ledger.jsonl",
        "metrics.json",
        "equity_hourly.csv",
        "trades.csv",
        "arms.json",
        "twin.json",
        "index.html",
        "summary.json",
    ):
        assert (exported / name).is_file(), name
    done = subprocess.run(  # noqa: S603 - the interpreter running the tests, a fixed script
        [sys.executable, str(RECOMPUTE), str(exported)],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "all checks agree" in done.stdout
    assert "marks from fills" in done.stdout


def test_the_baselines_rank_the_replica_and_the_live_gap_is_published(exported: Path) -> None:
    """Review finding: the coin flips are costed by the simulator, so the agent is ranked through
    its governed replica, and the live book's gap to it is published and shown on the page."""
    summary = json.loads((exported / "arms_summary.json").read_text(encoding="utf-8"))
    assert summary["coin_flip"]["ranked_arm_id"] == "twin_governed_replica"
    gap = summary["replica_vs_live"]
    assert gap["ranked_arm_id"] == "twin_governed_replica"
    assert gap["fills_priced"] >= 1
    assert gap["slippage"]["n"] == gap["fills_priced"]
    assert gap["return_gap"] == pytest.approx(
        gap["live_total_return"] - gap["replica_total_return"]
    )
    page = (exported / "index.html").read_text(encoding="utf-8")
    assert "Simulator check" in page
    assert "costed exactly as the seeds are" in page


def test_verify_reports_the_exported_chain_intact(day: Day, exported: Path) -> None:
    out = io.StringIO()
    code = run_cli(
        ["verify", "--public", str(exported)],
        root=day.root,
        clock=day.clock,
        parts=day.parts,
        out=out,
    )
    assert code == 0, out.getvalue()
    assert "INTACT" in out.getvalue()


def test_replay_reproduces_the_first_cycle_ruling_and_intents(day: Day, exported: Path) -> None:
    first = day.cards[0]
    assert first.decision_id is not None
    out = io.StringIO()
    code = run_cli(
        ["replay", "--decision", first.decision_id, "--public", str(exported)],
        root=day.root,
        clock=day.clock,
        parts=day.parts,
        out=out,
    )
    assert code == 0, out.getvalue()
    assert "REPRODUCED" in out.getvalue()
    from sentiment_agent.ledger.blobs import FileBlobStore
    from sentiment_agent.policy import POLICY_V1

    result = replay_decision(
        read_events(exported / "ledger.jsonl"),
        FileBlobStore(exported / "blobs"),
        first.decision_id,
        policy=POLICY_V1,
    )
    assert result.decision_matches
    assert result.ruling_id == first.ruling_id
    assert result.ruling_matches
    assert result.plan_matches
    logged = {o.client_oid for o in first.orders}
    assert {"sa" + i[:30] for i in result.intent_ids} == logged


def test_replay_of_the_outage_cycle_reproduces_the_protective_flatten(
    day: Day, exported: Path
) -> None:
    outage = day.cards[1]
    assert outage.outcome is LlmOutcome.TRANSPORT_ERROR
    assert outage.decision_id is not None
    out = io.StringIO()
    code = run_cli(
        ["replay", "--decision", outage.decision_id, "--public", str(exported)],
        root=day.root,
        clock=day.clock,
        parts=day.parts,
        out=out,
    )
    assert code == 0, out.getvalue()
    assert "IDENTICAL" in out.getvalue()
