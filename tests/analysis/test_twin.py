"""The governed/ungoverned twin: intervention accounting against the production kernel's rulings."""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from analysis.abuild import (
    AAPL,
    BTC,
    HOUR,
    NVDA,
    WED,
    book,
    candles,
    decision,
    grounded,
    inputs,
    simulator,
    ungrounded,
)
from sentiment_agent.analysis.armsim import ArmSimulator
from sentiment_agent.analysis.twin import (
    GOVERNED_REPLICA_SPEC,
    UNGOVERNED_SPEC,
    governed_replica,
    guard_counts,
    twin_report,
    ungoverned_arm,
    violated_guards,
)
from sentiment_agent.clock import ManualClock
from sentiment_agent.kernel.kernel import RiskKernel
from sentiment_agent.policy import POLICY_V1
from sentiment_agent.types import (
    ALL_GUARDS,
    Activation,
    BookState,
    BreakerState,
    DecisionRecord,
    GuardId,
    GuardStatus,
    KernelInputs,
    KernelRuling,
    LlmOutcome,
    Position,
    RulingContext,
)

PRICES = {BTC: "80000", NVDA: "200", AAPL: "300"}
D1 = WED
D2 = WED + 4 * HOUR
D3 = WED + 8 * HOUR
NVDA_CLOSES = ["200", "198", "196", "194", *["194"] * 12]
"""NVDA falls 3% between the first decision and the second (13:00 -> 17:00 closes)."""


def _sim() -> ArmSimulator:
    return simulator(
        {
            BTC: candles(BTC, WED, ["80000"] * 16),
            NVDA: candles(NVDA, WED, NVDA_CLOSES),
            AAPL: candles(AAPL, WED, ["300"] * 16),
        }
    )


def _inputs(at: datetime, name: str) -> KernelInputs:
    return inputs(PRICES, at=at).model_copy(update={"snapshot_id": name})


def _held_book(at: datetime) -> BookState:
    qty = Decimal("0.0625")
    position = Position(
        symbol=BTC,
        qty=qty,
        avg_entry=Decimal("80008"),
        opened_at=D1,
        last_increase_at=D1,
        realized_pnl=Decimal(0),
        fees_paid=Decimal("3.0003"),
        stop_price=None,
        stop_venue_id=None,
        last_decision_id="d1",
    )
    base = book("99996.4997", at=at)
    return base.model_copy(update={"positions": {BTC: position}, "marks": {BTC: Decimal("80000")}})


def _rule(record: DecisionRecord, given: KernelInputs, state: BookState) -> KernelRuling:
    kernel = RiskKernel(POLICY_V1, ManualClock(given.at))
    assert record.decision is not None
    return kernel.rule(
        proposed=record.proposed_weights,
        book=state,
        inputs=given,
        context=RulingContext(
            decision_id=record.decision_id,
            protective_reason=None,
            grounding=dict(record.grounding),
            invalidation_fired={
                t.symbol: t.invalidation_triggered for t in record.decision.targets
            },
        ),
        breaker=BreakerState(activation=Activation.ACTIVE, since=WED, trips=()),
        guards=ALL_GUARDS,
    )


def _scenario() -> tuple[list[DecisionRecord], list[KernelInputs], list[KernelRuling]]:
    """d1 asks for BTC and NVDA at 5%; NVDA's thesis cites a figure that is not in the snapshot, so
    G9 refuses it. d2 keeps BTC and asks for nothing in NVDA. d3 is a model outage."""
    d1 = decision(
        "d1",
        {BTC: 0.05, NVDA: 0.05},
        at=D1,
        snapshot_id="s1",
        grounding={BTC: grounded(), NVDA: ungrounded()},
    )
    d2 = decision("d2", {BTC: 0.05, NVDA: 0.0}, at=D2, snapshot_id="s2")
    d3 = decision("d3", {}, at=D3, snapshot_id="s3", outcome=LlmOutcome.TIMEOUT)
    given = [_inputs(D1, "s1"), _inputs(D2, "s2"), _inputs(D3, "s3")]
    rulings = [_rule(d1, given[0], book(at=D1)), _rule(d2, given[1], _held_book(D2))]
    return [d1, d2, d3], given, rulings


def test_intervention_accounting() -> None:
    decisions, given, rulings = _scenario()
    sim = _sim()
    governed = governed_replica(sim, decisions, given, ci_resamples=0)
    ungoverned = ungoverned_arm(sim, decisions, given, ci_resamples=0)
    report = twin_report(decisions, rulings, governed, ungoverned, human_takeovers=2, sim=sim)

    assert report.n_decisions == 2  # the outage wrote no draft
    assert report.n_interventions == 1
    assert report.intervention_rate == 0.5
    assert report.risk_violation_rate_ungoverned == 0.25  # 1 of 4 proposed legs
    (item,) = report.interventions
    assert (item.decision_id, item.symbol, item.guard) == ("d1", NVDA, GuardId.G9_GROUNDING)
    assert item.ruling_id == rulings[0].ruling_id
    assert (item.proposed_weight, item.approved_weight) == (0.05, 0.0)
    assert item.violations_prevented == (GuardId.G9_GROUNDING,)
    cost = 0.0006 + 0.0001
    assert item.pnl_ungoverned == pytest.approx(0.05 * (194 / 200 - 1) - cost * 0.05)
    assert item.pnl_governed == 0.0
    assert report.prevented_loss == pytest.approx(-item.pnl_ungoverned)
    assert report.forgone_gain == 0.0
    assert report.human_takeovers == 2
    assert report.max_drawdown_governed == governed.metrics.max_drawdown
    assert report.max_drawdown_ungoverned == ungoverned.metrics.max_drawdown
    assert report.max_drawdown_ungoverned < report.max_drawdown_governed
    assert guard_counts(report) == {GuardId.G9_GROUNDING: 1}


def test_the_two_arms_differ_only_by_the_kernel() -> None:
    decisions, given, _ = _scenario()
    sim = _sim()
    governed = governed_replica(sim, decisions, given, ci_resamples=0)
    ungoverned = ungoverned_arm(sim, decisions, given, ci_resamples=0)
    assert governed.spec == GOVERNED_REPLICA_SPEC
    assert set(governed.spec.guards) == ALL_GUARDS
    assert ungoverned.spec == UNGOVERNED_SPEC
    assert ungoverned.spec.guards == ()
    ungoverned_symbols = {t.symbol for t in ungoverned.trades}
    assert NVDA in ungoverned_symbols  # the refused draft, run anyway, and closed by d2
    nvda = next(t for t in ungoverned.trades if t.symbol == NVDA)
    assert nvda.decision_ids == ("d1", "d2")
    assert nvda.net_pnl < 0
    assert NVDA not in {t.symbol for t in governed.trades}
    # The outage at d3 flattens the governed replica; the ungoverned arm has no kernel to do it.
    btc_closed = [t for t in governed.trades if t.symbol == BTC]
    assert btc_closed
    assert btc_closed[0].closed_at == D3
    assert governed.marks[-1].gross_weight == 0.0
    assert ungoverned.marks[-1].gross_weight > 0.04


def test_a_report_with_interventions_needs_the_simulator() -> None:
    decisions, given, rulings = _scenario()
    sim = _sim()
    governed = governed_replica(sim, decisions, given, ci_resamples=0)
    with pytest.raises(ValueError, match="simulator"):
        twin_report(decisions, rulings, governed, governed, human_takeovers=0)


def test_a_report_without_interventions_does_not() -> None:
    decisions, _, rulings = _scenario()
    only_d2 = [decisions[1]]
    arm = governed_replica(_sim(), only_d2, [_inputs(D2, "s2")], ci_resamples=0)
    report = twin_report(only_d2, rulings[1:], arm, arm, human_takeovers=0)
    assert report.n_decisions == 1
    assert report.n_interventions == 0
    assert report.intervention_rate == 0.0
    assert report.prevented_loss == report.forgone_gain == 0.0


def test_decisions_without_a_ruling_are_not_counted() -> None:
    decisions, _, rulings = _scenario()
    arm = governed_replica(_sim(), decisions[1:2], [_inputs(D2, "s2")], ci_resamples=0)
    report = twin_report(decisions, rulings[1:], arm, arm, human_takeovers=0)
    assert report.n_decisions == 1


def test_bad_inputs_are_refused() -> None:
    decisions, given, rulings = _scenario()
    sim = _sim()
    with pytest.raises(ValueError, match="one set of kernel inputs"):
        ungoverned_arm(sim, decisions, given[:2])
    swapped = [given[1], given[0], given[2]]
    with pytest.raises(ValueError, match="snapshot"):
        ungoverned_arm(sim, decisions, swapped)
    arm = governed_replica(sim, decisions, given, ci_resamples=0)
    with pytest.raises(ValueError, match="two kernel rulings"):
        twin_report(decisions, [rulings[0], rulings[0]], arm, arm, human_takeovers=0, sim=sim)
    with pytest.raises(ValueError, match="negative"):
        twin_report(decisions, rulings, arm, arm, human_takeovers=-1, sim=sim)


def test_a_missing_input_is_an_intervention_but_not_a_violation() -> None:
    """G9 with no grounding report fails closed (NOT_EVALUATED): the kernel still cut the leg, but
    the model broke no rule."""
    record = decision("d9", {AAPL: 0.05}, at=D1, snapshot_id="s9", grounding={})
    ruling = _rule(record, _inputs(D1, "s9"), book(at=D1))
    inst = ruling.instrument(AAPL)
    assert inst is not None
    assert inst.binding_guard is GuardId.G9_GROUNDING
    g9 = next(g for g in inst.rulings if g.guard is GuardId.G9_GROUNDING)
    assert g9.status is GuardStatus.NOT_EVALUATED
    assert violated_guards(inst, ruling.book_rulings) == ()
    sim = _sim()
    arm = governed_replica(sim, [record], [_inputs(D1, "s9")], ci_resamples=0)
    report = twin_report([record], [ruling], arm, arm, human_takeovers=0, sim=sim)
    assert report.n_interventions == 1
    assert report.risk_violation_rate_ungoverned == 0.0


def test_violations_list_every_fired_guard_below_the_request() -> None:
    """In the weekend no-open buffer an ungrounded NVDA open breaks two rules: G2 and G9."""
    friday_1830 = datetime(2026, 9, 25, 18, 30, tzinfo=WED.tzinfo)
    record = decision(
        "d5", {NVDA: 0.05}, at=friday_1830, snapshot_id="s5", grounding={NVDA: ungrounded()}
    )
    given = inputs(PRICES, at=friday_1830).model_copy(update={"snapshot_id": "s5"})
    ruling = _rule(record, given, book(at=friday_1830))
    inst = ruling.instrument(NVDA)
    assert inst is not None
    assert violated_guards(inst, ruling.book_rulings) == (
        GuardId.G2_WEEKEND_FREEZE,
        GuardId.G9_GROUNDING,
    )
    assert inst.binding_guard is GuardId.G2_WEEKEND_FREEZE  # ties resolve in guard order


def test_forgone_gain_when_the_refused_leg_would_have_won() -> None:
    rising = simulator(
        {
            BTC: candles(BTC, WED, ["80000"] * 16),
            NVDA: candles(NVDA, WED, ["200", "204", "206", "210", *["210"] * 12]),
            AAPL: candles(AAPL, WED, ["300"] * 16),
        }
    )
    decisions, given, rulings = _scenario()
    arm = governed_replica(rising, decisions, given, ci_resamples=0)
    report = twin_report(decisions, rulings, arm, arm, human_takeovers=0, sim=rising)
    (item,) = report.interventions
    assert item.pnl_ungoverned == pytest.approx(0.05 * (210 / 200 - 1) - 0.0007 * 0.05)
    assert report.forgone_gain == pytest.approx(item.pnl_ungoverned)
    assert report.prevented_loss == 0.0


def test_the_last_intervention_runs_to_the_end_of_the_record() -> None:
    decisions, given, rulings = _scenario()
    sim = _sim()
    arm = governed_replica(sim, decisions[:1], given[:1], until=WED + 6 * HOUR, ci_resamples=0)
    report = twin_report(decisions[:1], rulings[:1], arm, arm, human_takeovers=0, sim=sim)
    (item,) = report.interventions
    # Window 13:00 -> 19:00 (the arms' last mark): first closes at 14:00 (200) and 19:00 (194).
    assert item.pnl_ungoverned == pytest.approx(0.05 * (194 / 200 - 1) - 0.0007 * 0.05)
    assert arm.marks[-1].at == WED + timedelta(hours=6)
