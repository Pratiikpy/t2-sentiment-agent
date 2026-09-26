"""A full simulated trading day, end to end, through the real runtime.

Friday 2026-09-25 from 12:50 UTC, then a restart on Monday 2026-09-28 after the process was down
over the weekend, under the policy the runtime loads (policy v2). Everything is the production code
path; only the outside world is fake (keyless market data, the two Bitget services, the crowd, the
chat model's answers, the ``ots`` calendar) and the venue is :class:`SimulatedVenue`. What happens,
and what is asserted:

* 12:40  ``t2sa genesis --mode simulated``: a rehearsal pre-registration, stamped.
* 13:30  the US-open heartbeat wakes the model. It asks for eight longs; the kernel refuses MSTR
         (Demo mark 5% off its index, G1) and GOOGL (a figure it cannot ground, G9), and cuts the
         other six to the 10% net cap (G3, run2-a3). Six orders fill, each with its venue stop.
* 14:10  NVDA gaps down 6%: its venue stop fires on the next poll, and reconciliation books it.
* 14:40  the AAPL Demo mark departs 5% from its index: the protective check exits AAPL (G1).
* 15:00  crypto Fear & Greed crosses into extreme fear: an event decision; the model service
         fails. The first failure holds the book under its venue stops (run2-a6).
* 16:00  the funding heartbeat: the model fails again, and the book is held again.
* 16:30  an owner-requested decision: the third failure in a row flattens the book
         (``llm_outage``), and the breaker halts.
* 17:00  another owner request: a valid decision lifts the outage halt and opens four legs, cut to
         the net cap.
* 19:46  weekend pre-flatten: the US legs are closed before the freeze (G2); BTC stays.
* hourly marks through 21:00, then the process stops.
* Monday 00:01 the restart rebuilds everything from the ledger, catches up the seven heartbeats it
         missed in one decision, closes BTC and opens a hedged book: four equity longs, two index
         shorts, 25% gross.
* 00:30  the four longs gap 12% through their stops; the day's loss passes 1.5%, and the daily
         kill flattens the two shorts (G5) and halts the breaker.
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
            QwenTransportError("scripted outage: the gateway is still down"),
            QwenTransportError("scripted outage: a third failure in a row"),
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
        loop = RunLoop(app)
        friday: dict[datetime, Callable[[], object]] = {
            _at(FRIDAY, 14, 10): lambda: world.move("NVDAUSDT", "-6"),
            _at(FRIDAY, 14, 40): lambda: world.mark_skew.__setitem__("AAPLUSDT", Decimal("0.05")),
            _at(FRIDAY, 14, 46): lambda: world.mark_skew.pop("AAPLUSDT"),
            # The legs the outage will close are in profit, so the closes do not all lose.
            _at(FRIDAY, 14, 50): lambda: [
                world.move(s, "1.5") for s in ("METAUSDT", "AMZNUSDT", "TSLAUSDT", "SP500USDT")
            ],
            _at(FRIDAY, 15, 0): lambda: setattr(toolkit, "crypto_fear_greed", 20),
            _at(FRIDAY, 16, 30): lambda: loop.request_owner_decision(
                "the model has failed twice; ask again", symbols=[]
            ),
            _at(FRIDAY, 17, 0): lambda: loop.request_owner_decision(
                "the model service is back; decide", symbols=[]
            ),
        }
        _run(loop, clock, _at(FRIDAY, 21, 2), friday, cards)
        app.close()

        # The weekend: the process is down; BTC drifts up 1%.
        world.move("BTCUSDT", "1")
        clock.set(_at(MONDAY, 0, 1))
        monday_script = [
            decision(
                "act",
                [
                    target("BTCUSDT", 0.0),
                    *(target(s, 1.0) for s in ("NVDAUSDT", "AAPLUSDT", "METAUSDT", "AMZNUSDT")),
                    *(target(s, -1.0) for s in ("SP500USDT", "NDX100USDT")),
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
    owner = [t for t in triggers if t.kind is TriggerKind.OWNER_MANUAL]
    assert [t.fired_at for t in owner] == [_at(FRIDAY, 16, 30), _at(FRIDAY, 17, 0)]
    monday = day.cards[-1]
    assert monday.at == _at(MONDAY, 0, 1)
    assert len(monday.triggers) == 7


def test_every_decision_is_logged_with_its_outcome(day: Day) -> None:
    outcomes = [d.outcome for d in day.app.projection.decisions]
    assert outcomes == [
        LlmOutcome.DECIDED,
        LlmOutcome.TRANSPORT_ERROR,
        LlmOutcome.TRANSPORT_ERROR,
        LlmOutcome.TRANSPORT_ERROR,
        LlmOutcome.DECIDED,
        LlmOutcome.DECIDED,
    ]
    assert [c.outcome for c in day.cards] == outcomes
    budget = day.app.projection.budget_states
    assert len(budget) == 6


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
    # Six longs: the 25% gross cap would allow 25%; the 10% net cap (run2-a3) binds first.
    assert sum(approved.values()) == pytest.approx(0.10)
    assert all(i.binding_guard is GuardId.G3_SIZE for i in ruling.instruments if i.approved_weight)
    assert any("net cap" in g.reason for g in ruling.book_rulings if g.guard is GuardId.G3_SIZE)
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
    assert outage.at == _at(FRIDAY, 16, 30)
    weekend = by_reason[ProtectiveReason.WEEKEND_FREEZE]
    assert set(weekend.symbols) == {"AMZNUSDT", "METAUSDT", "NDX100USDT"}
    assert weekend.at == _at(FRIDAY, 19, 46)
    kill = by_reason[ProtectiveReason.DAILY_KILL]
    assert kill.symbols == ("NDX100USDT", "SP500USDT")
    assert kill.at.date() == MONDAY.date()
    rulings = {r.ruling_id: r for r in day.app.projection.rulings}
    kill_ruling: KernelRuling = rulings[kill.ruling_id]
    for symbol in kill.symbols:
        short = kill_ruling.instrument(symbol)
        assert short is not None
        assert short.current_weight < 0
        assert short.binding_guard is GuardId.G5_DAILY_KILL


def test_the_first_two_model_failures_hold_the_book_and_the_third_flattens_it(day: Day) -> None:
    """run2-a6: a failed decision is not a reason to trade until three fail in a row."""
    outages = [c for c in day.cards if c.outcome is LlmOutcome.TRANSPORT_ERROR]
    assert [c.at for c in outages] == [
        _at(FRIDAY, 15, 0),
        _at(FRIDAY, 16, 0),
        _at(FRIDAY, 16, 30),
    ]
    assert [bool(c.orders) for c in outages] == [False, False, True]
    notes = [str(n["text"]) for n in day.events(EventKind.NOTE)]
    assert any("1 of 3 in a row: positions held under their venue stops" in n for n in notes)
    assert any("2 of 3 in a row: positions held under their venue stops" in n for n in notes)
    # The legs the first two failures held were closed by the third, not before.
    held = {"AMZNUSDT", "METAUSDT", "SP500USDT", "TSLAUSDT"}
    closes = {
        t.symbol: t.closed_at
        for t in day.app.projection.closed_trades
        if t.symbol in held and t.opened_at == _at(FRIDAY, 13, 30)
    }
    assert closes == dict.fromkeys(held, _at(FRIDAY, 16, 30))


def test_the_monday_book_is_hedged_and_its_longs_stopped_out(day: Day) -> None:
    monday = day.cards[-1]
    assert monday.kernel is not None
    approved = {i.symbol: i.approved_weight for i in monday.kernel.instruments}
    longs = sum(w for w in approved.values() if w > 0)
    shorts = -sum(w for w in approved.values() if w < 0)
    assert longs + shorts == pytest.approx(0.25)
    assert longs - shorts <= 0.10 + 1e-9
    assert approved["BTCUSDT"] == 0.0
    stopped = {
        t.symbol
        for t in day.app.projection.closed_trades
        if t.closed_at.date() == MONDAY.date() and t.exit_reason == "venue_initiated"
    }
    assert stopped == {"NVDAUSDT", "AAPLUSDT", "METAUSDT", "AMZNUSDT"}


def test_the_breaker_halted_on_the_outage_recovered_and_halted_on_the_kill(day: Day) -> None:
    transitions = day.app.projection.breaker_transitions
    moves = [(t.from_state, t.to_state, t.trips) for t in transitions]
    assert (Activation.ACTIVE, Activation.HALTED, ("llm_outage",)) in moves
    assert (Activation.HALTED, Activation.ACTIVE, ()) in moves
    halted = next(t for t in transitions if t.trips == ("llm_outage",))
    assert halted.at == _at(FRIDAY, 16, 30)
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
    # Policy v2 scores the record over its pre-registered window (run2-a4): Friday is outside it,
    # and both the export and the independent recompute count Monday's hour alone.
    assert "scored over the pre-registered window 2026-09-28 00:00 to 2026-10-01 00:00 UTC" in (
        done.stdout
    )
    metrics = json.loads((exported / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["n_hours"] == 1
    assert metrics["n_closed_trades"] == 7  # BTC's close at 00:01 and the six at 00:31
    trades = (exported / "trades.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(trades) - 1 == 16  # the full record is still published
    page = (exported / "index.html").read_text(encoding="utf-8")
    assert "scored over the pre-registered window 2026-09-28 00:00 to 2026-10-01 00:00 UTC" in page


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


@pytest.mark.parametrize("index", [1, 3], ids=["held", "flattened"])
def test_replay_of_an_outage_cycle_reproduces_it(day: Day, exported: Path, index: int) -> None:
    outage = day.cards[index]
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
