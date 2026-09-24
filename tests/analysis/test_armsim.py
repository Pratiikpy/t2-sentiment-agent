"""The arm simulator: costs, stops, the daily kill and the weekend freeze, through the production
kernel, planner and book."""

from datetime import timedelta
from decimal import Decimal

import pytest

from analysis.abuild import (
    AAPL,
    BTC,
    EQUITY,
    FRI,
    HOUR,
    META,
    NVDA,
    TSLA,
    WED,
    candles,
    flat_candles,
    inputs,
    kernel,
    simulator,
    snapshot,
)
from sentiment_agent.analysis.armsim import ArmSimulator
from sentiment_agent.clock import ManualClock
from sentiment_agent.kernel.kernel import RiskKernel
from sentiment_agent.policy import POLICY_V1
from sentiment_agent.types import (
    ALL_GUARDS,
    VENUE_GUARDS,
    ArmKind,
    ArmResult,
    ArmSpec,
    GuardId,
    PriceSource,
    ProtectiveReason,
    RulingContext,
)

ONE_BP = Decimal("0.0001")


def arm(guards: frozenset[GuardId] | set[GuardId], arm_id: str = "arm") -> ArmSpec:
    return ArmSpec(
        arm_id=arm_id,
        kind=ArmKind.BASELINE,
        title="t",
        description="d",
        provenance="tests",
        uses_llm=False,
        guards=tuple(g for g in GuardId if g in guards),
    )


VENUE = arm(VENUE_GUARDS)


# ------------------------------------------------------------------------------------------------
# Costs
# ------------------------------------------------------------------------------------------------


def test_round_trip_costs_are_half_spread_and_taker_fee_each_side() -> None:
    """0.0625 BTC at 80,000, 2 bp spread: buy at 80,008, sell at 79,992, 6 bp fee each side.
    Equity falls by 0.0625 x 16 + 3.0003 + 2.9997 = 7.0 exactly."""
    sim = simulator({BTC: flat_candles(BTC, WED, 6, "80000")})
    schedule = [
        (WED, {BTC: 0.05}, inputs({BTC: "80000"}, at=WED)),
        (WED + 2 * HOUR, {BTC: 0.0}, inputs({BTC: "80000"}, at=WED + 2 * HOUR)),
    ]
    result = sim.run(VENUE, schedule, until=WED + 3 * HOUR, ci_resamples=0)
    assert [m.equity for m in result.marks] == [EQUITY, 99_996.4997, 99_996.4997, 99_993.0]
    assert result.marks[1].gross_weight == pytest.approx(5000 / 99_996.4997)
    (trade,) = result.trades
    assert trade.entry_avg == Decimal("80008")
    assert trade.exit_avg == Decimal("79992")
    assert trade.max_abs_qty == Decimal("0.0625")
    assert trade.gross_pnl == Decimal("-1")
    assert trade.fees == Decimal("6")
    assert trade.net_pnl == Decimal("-7")
    assert trade.exit_reason == "model_close"
    assert trade.decision_ids == ("arm#0", "arm#1")
    assert result.metrics.fees_paid == pytest.approx(6.0)
    assert result.metrics.turnover == pytest.approx((5000.5 + 4999.5) / (399_985.9994 / 4))
    assert result.metrics.total_return == pytest.approx(-7e-5)


def test_cost_model_helpers() -> None:
    sim = simulator({BTC: flat_candles(BTC, WED, 3, "80000")}, spreads={BTC: 3.0})
    assert sim.half_spread(BTC) == Decimal("0.00015")
    assert sim.fee_rate(BTC) == Decimal("0.0006")
    assert sim.cost_rate(BTC) == pytest.approx(0.00075)
    with pytest.raises(ValueError, match="no Demo spread"):
        sim.half_spread(NVDA)


def test_price_lookups() -> None:
    sim = simulator({BTC: candles(BTC, WED, ["100", "110", "120"])})
    assert sim.price_at(BTC, WED) is None
    assert sim.price_at(BTC, WED + HOUR) == Decimal("100")
    assert sim.price_at(BTC, WED + HOUR * 1.5) == Decimal("100")
    assert sim.price_after(BTC, WED + HOUR * 1.5) == Decimal("110")
    assert sim.price_after(BTC, WED + 3 * HOUR) == Decimal("120")
    assert sim.price_after(BTC, WED + 4 * HOUR) is None
    assert sim.default_until() == WED + 3 * HOUR
    assert sim.last_close(BTC) == (WED + 3 * HOUR, Decimal("120"))
    assert sim.price_at(NVDA, WED) is None


# ------------------------------------------------------------------------------------------------
# Stops
# ------------------------------------------------------------------------------------------------


def _stop_run(
    closes: list[str], *, lows: dict[int, str] | None = None, opens: dict[int, str] | None = None,
    at_minutes: int = 0, spec: ArmSpec = VENUE,
) -> ArmResult:  # fmt: skip
    sim = simulator({BTC: candles(BTC, WED, closes, lows=lows, opens=opens)})
    at = WED + timedelta(minutes=at_minutes)
    return sim.run(spec, [(at, {BTC: 0.05}, inputs({BTC: "80000"}, at=at))], ci_resamples=0)


STOP = Decimal("76807.7")
"""4% below the 80,008 entry, rounded up onto the 0.1 grid (tighter, never looser)."""


def test_stop_fills_at_the_stop_when_the_candle_reaches_it() -> None:
    result = _stop_run(["80000", "79000"], lows={0: "76000"})
    (trade,) = result.trades
    assert trade.exit_reason == "stop_filled"
    assert trade.closed_at == WED + HOUR
    assert trade.exit_avg == STOP * (1 - ONE_BP)
    assert result.marks[-1].gross_weight == 0.0


def test_stop_gaps_fill_at_the_open() -> None:
    result = _stop_run(["80000", "75500"], opens={1: "75000"})
    (trade,) = result.trades
    assert trade.exit_avg == Decimal("75000") * (1 - ONE_BP)
    assert trade.closed_at == WED + HOUR  # stamped at the open of the candle that gapped


def test_stop_in_the_hour_of_entry_uses_only_the_close() -> None:
    """Entered at 13:30: the 13:00 candle's low of 70,000 may have come before the fill."""
    result = _stop_run(["79000", "77000"], lows={0: "70000", 1: "76000"}, at_minutes=30)
    (trade,) = result.trades
    assert trade.closed_at == WED + 2 * HOUR
    assert trade.exit_avg == STOP * (1 - ONE_BP)


def test_stop_in_the_hour_of_entry_fires_on_the_close() -> None:
    result = _stop_run(["76000", "76000"], at_minutes=30)
    (trade,) = result.trades
    assert trade.closed_at == WED + HOUR
    assert trade.exit_avg == Decimal("76000") * (1 - ONE_BP)


def test_no_stop_without_g4() -> None:
    result = _stop_run(
        ["80000", "60000"], lows={0: "50000"}, spec=arm(VENUE_GUARDS - {GuardId.G4_STOP})
    )
    assert result.trades == ()
    assert result.marks[-1].gross_weight > 0


# ------------------------------------------------------------------------------------------------
# The daily kill
# ------------------------------------------------------------------------------------------------

FIVE = {BTC: "80000", NVDA: "200", TSLA: "400", META: "700", AAPL: "300"}


def _kill_scenario(audit: bool = False) -> ArmResult:
    """Five 5% longs on Wednesday 13:00; every name falls 8% in the 14:00 candle (a 2% loss of
    equity), so the 15:00 check kills the book. A 17:00 decision is refused; Thursday's is not."""
    hours = 15
    marks = {}
    for symbol, price in FIVE.items():
        p = Decimal(price)
        closes = [p, p * Decimal("0.92"), *[p * Decimal("0.92")] * (hours - 2)]
        marks[symbol] = candles(symbol, WED, closes)
    sim = simulator(marks, audit=audit)
    dropped = {s: str(Decimal(p) * Decimal("0.92")) for s, p in FIVE.items()}
    thursday = WED + 12 * HOUR  # 01:00 UTC Thursday
    schedule = [
        (WED, dict.fromkeys(FIVE, 0.05), inputs(FIVE, at=WED)),
        (WED + 4 * HOUR, {BTC: 0.05}, inputs(dropped, at=WED + 4 * HOUR)),
        (thursday, {BTC: 0.05}, inputs(dropped, at=thursday)),
    ]
    return sim.run(arm(VENUE_GUARDS - {GuardId.G4_STOP}), schedule, ci_resamples=0)


def test_daily_kill_flattens_and_holds_until_the_next_day() -> None:
    result = _kill_scenario()
    killed = [t for t in result.trades if t.exit_reason == "protective_exit:daily_kill"]
    assert {t.symbol for t in killed} == set(FIVE)
    assert all(t.closed_at == WED + 2 * HOUR for t in killed)
    by_hour = {m.at: m for m in result.marks}
    assert by_hour[WED + 2 * HOUR].gross_weight == 0.0
    assert by_hour[WED + 5 * HOUR].gross_weight == 0.0  # the 17:00 decision was refused
    assert by_hour[WED + 13 * HOUR].gross_weight > 0.04  # Thursday's decision was not
    assert by_hour[WED + 2 * HOUR].equity == pytest.approx(EQUITY * (1 - 0.25 * 0.08), rel=2e-3)


def test_protective_prefilter_never_changes_a_result() -> None:
    assert _kill_scenario(audit=False) == _kill_scenario(audit=True)
    assert _weekend_scenario(audit=False) == _weekend_scenario(audit=True)


# ------------------------------------------------------------------------------------------------
# The weekend freeze
# ------------------------------------------------------------------------------------------------


def _weekend_scenario(audit: bool = False, spec: ArmSpec = VENUE) -> ArmResult:
    prices = {BTC: "80000", NVDA: "200", AAPL: "300"}
    marks = {s: flat_candles(s, FRI, 12, p) for s, p in prices.items()}
    sim = simulator(marks, audit=audit)
    open_at = FRI + HOUR  # 14:00
    buffer_at = FRI + timedelta(hours=5, minutes=30)  # 18:30, inside the no-open buffer
    schedule = [
        (open_at, {NVDA: 0.05, BTC: 0.05}, inputs(prices, at=open_at)),
        (buffer_at, {NVDA: 0.05, BTC: 0.05, AAPL: 0.05}, inputs(prices, at=buffer_at)),
    ]
    return sim.run(spec, schedule, until=FRI + 9 * HOUR, ci_resamples=0)


def test_weekend_freeze_flattens_us_legs_and_refuses_opens_in_the_buffer() -> None:
    result = _weekend_scenario()
    (trade,) = result.trades
    assert trade.symbol == NVDA
    assert trade.exit_reason == "protective_exit:weekend_freeze"
    assert trade.closed_at == FRI + 7 * HOUR  # Friday 20:00
    last = result.marks[-1]
    assert last.gross_weight == pytest.approx(0.05, rel=5e-3)  # BTC, untouched by G2
    buffer_mark = next(m for m in result.marks if m.at == FRI + 6 * HOUR)
    assert buffer_mark.gross_weight == pytest.approx(0.10, rel=5e-3)  # no AAPL was opened


def test_ungoverned_arm_holds_through_the_weekend() -> None:
    result = _weekend_scenario(spec=arm(set()))
    assert result.trades == ()
    assert result.marks[-1].gross_weight == pytest.approx(0.15, rel=5e-3)


# ------------------------------------------------------------------------------------------------
# Units, grid and refusals
# ------------------------------------------------------------------------------------------------


def test_unit_equity_places_nothing_on_the_real_grid() -> None:
    sim = simulator({BTC: flat_candles(BTC, WED, 4, "80000")}, equity=1.0)
    result = sim.run(VENUE, [(WED, {BTC: 0.05}, inputs({BTC: "80000"}, at=WED))], ci_resamples=0)
    assert result.trades == ()
    assert {m.equity for m in result.marks} == {1.0}
    assert result.metrics.turnover == 0.0


def test_entries_at_or_after_until_are_not_simulated() -> None:
    sim = simulator({BTC: flat_candles(BTC, WED, 4, "80000")})
    late = WED + 2 * HOUR
    result = sim.run(
        VENUE, [(late, {BTC: 0.05}, inputs({BTC: "80000"}, at=late))], start=WED, until=late
    )
    assert result.marks[-1].gross_weight == 0.0
    assert [m.at for m in result.marks] == [WED, WED + HOUR, late]


def test_contexts_carry_the_decision_ids() -> None:
    sim = simulator({BTC: flat_candles(BTC, WED, 4, "80000")})
    schedule = [
        (WED, {BTC: 0.05}, inputs({BTC: "80000"}, at=WED)),
        (WED + HOUR, {BTC: 0.0}, inputs({BTC: "80000"}, at=WED + HOUR)),
    ]
    contexts = [
        RulingContext(decision_id="d-open", protective_reason=None),
        RulingContext(decision_id="d-close", protective_reason=None),
    ]
    result = sim.run(VENUE, schedule, contexts=contexts, ci_resamples=0)
    assert result.trades[0].decision_ids == ("d-open", "d-close")
    with pytest.raises(ValueError, match="one ruling context per"):
        sim.run(VENUE, schedule, contexts=contexts[:1])
    protective = [RulingContext(decision_id=None, protective_reason=ProtectiveReason.BREAKER)] * 2
    with pytest.raises(ValueError, match="model decision"):
        sim.run(VENUE, schedule, contexts=protective)


def test_g9_without_grounding_fails_closed() -> None:
    sim = simulator({BTC: flat_candles(BTC, WED, 4, "80000")})
    result = sim.run(
        arm(ALL_GUARDS), [(WED, {BTC: 0.05}, inputs({BTC: "80000"}, at=WED))], ci_resamples=0
    )
    assert result.trades == ()
    assert result.marks[-1].gross_weight == 0.0


def test_schedule_and_grid_are_checked() -> None:
    sim = simulator({BTC: flat_candles(BTC, WED, 4, "80000")})
    entry = (WED + HOUR, {BTC: 0.05}, inputs({BTC: "80000"}, at=WED + HOUR))
    earlier = (WED, {BTC: 0.0}, inputs({BTC: "80000"}, at=WED))
    with pytest.raises(ValueError, match="time order"):
        sim.run(VENUE, [entry, earlier])
    with pytest.raises(ValueError, match="precedes"):
        sim.run(VENUE, [earlier], start=WED + HOUR)
    with pytest.raises(ValueError, match="on the hour"):
        sim.run(VENUE, [entry], start=WED + timedelta(minutes=5))
    with pytest.raises(ValueError, match="before start"):
        sim.run(VENUE, [], start=WED + 2 * HOUR, until=WED + HOUR)
    with pytest.raises(ValueError, match="explicit start"):
        sim.run(VENUE, [])


def test_a_held_position_needs_its_candles() -> None:
    sim = simulator({BTC: flat_candles(BTC, WED, 1, "80000")})
    with pytest.raises(ValueError, match="no Demo mark candle"):
        sim.run(
            VENUE,
            [(WED, {BTC: 0.05}, inputs({BTC: "80000"}, at=WED))],
            until=WED + 3 * HOUR,
        )


def test_simulator_inputs_are_checked() -> None:
    marks = {BTC: flat_candles(BTC, WED, 2, "80000")}
    other = POLICY_V1.model_copy(update={"version": "policy-other"})
    with pytest.raises(ValueError, match="different policy"):
        ArmSimulator(
            kernel=RiskKernel(other, ManualClock(WED)),
            policy=POLICY_V1,
            demo_marks=marks,
            spreads_bps={BTC: 1.0},
        )
    with pytest.raises(ValueError, match="non-negative"):
        simulator(marks, spreads={BTC: -1.0})
    with pytest.raises(ValueError, match="positive and finite"):
        simulator(marks, equity=0.0)
    with pytest.raises(ValueError, match="mark"):
        simulator({BTC: candles(BTC, WED, ["1"], kind="index")})
    with pytest.raises(ValueError, match="live"):
        simulator({BTC: candles(BTC, WED, ["1"], source=PriceSource.LIVE)})
    with pytest.raises(ValueError, match="not 1H"):
        simulator({BTC: candles(BTC, WED, ["1"], interval="4H")})
    with pytest.raises(ValueError, match="holds a candle for"):
        simulator({NVDA: flat_candles(BTC, WED, 1, "1")})


def test_kernel_inputs_from_a_snapshot() -> None:
    sim = simulator({BTC: flat_candles(BTC, WED, 2, "80000")})
    snap = snapshot({BTC: "80000", NVDA: "200"}, at=WED)
    built = sim.kernel_inputs(snap)
    assert built.snapshot_id == snap.snapshot_id
    assert built.snapshot_taken_at == WED
    assert built.demo_quotes == snap.demo_quotes
    assert built.demo_index_move_bps_3h == {BTC: 10.0, NVDA: 10.0}
    assert set(built.specs) == {BTC}
    bare = simulator({BTC: flat_candles(BTC, WED, 2, "80000")}, with_specs=False)
    with pytest.raises(ValueError, match="no Demo instrument specs"):
        bare.kernel_inputs(snap)


def test_the_kernel_passed_in_is_not_mutated_or_used_for_time() -> None:
    k = kernel()
    sim = ArmSimulator(
        kernel=k,
        policy=POLICY_V1,
        demo_marks={BTC: flat_candles(BTC, FRI, 4, "80000")},
        spreads_bps={BTC: 2.0},
        starting_equity=EQUITY,
    )
    at = FRI
    result = sim.run(VENUE, [(at, {BTC: 0.05}, inputs({BTC: "80000"}, at=at))], ci_resamples=0)
    assert result.marks[-1].gross_weight > 0  # ruled at Friday's instant, not the kernel's clock
