"""The live-marked mirror and the weekend counterfactual."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from analysis.abuild import (
    BTC,
    HOUR,
    NVDA,
    WED,
    candles,
    decision,
    grounded,
    inputs,
    ungrounded,
)
from sentiment_agent.analysis.mirror import (
    MIRROR_SPEC,
    WEEKEND_SPEC,
    live_mirror,
    reopen_after,
    weekend_counterfactual,
    weekend_episodes,
    weight_without_g2,
)
from sentiment_agent.clock import ManualClock
from sentiment_agent.kernel.kernel import RiskKernel
from sentiment_agent.policy import POLICY_V1
from sentiment_agent.types import (
    ALL_GUARDS,
    Activation,
    BookState,
    BreakerState,
    Candle,
    DecisionRecord,
    Fill,
    FillVenue,
    GuardId,
    KernelRuling,
    LlmOutcome,
    Position,
    PriceSource,
    ProtectiveReason,
    RulingContext,
    Side,
)

LIVE = PriceSource.LIVE
SAT = datetime(2026, 9, 26, 10, 0, tzinfo=UTC)
SUN = datetime(2026, 9, 27, 10, 0, tzinfo=UTC)
FRI_20 = datetime(2026, 9, 25, 20, 0, tzinfo=UTC)
MON = datetime(2026, 9, 28, 0, 0, tzinfo=UTC)


def fill(
    exec_id: str, side: Side, qty: str, price: str, fee: str, at: datetime, *, coin: str = "USDT"
) -> Fill:
    return Fill(
        exec_id=exec_id,
        venue_order_id=exec_id,
        client_oid=None,
        symbol=BTC,
        side=side,
        exec_price=Decimal(price),
        exec_qty=Decimal(qty),
        exec_value=Decimal(price) * Decimal(qty),
        fee_paid=Decimal(fee),
        fee_coin=coin,
        trade_scope="taker",
        trade_side="open" if side is Side.BUY else "close",
        exec_pnl=None,
        executed_at=at,
        venue=FillVenue.BITGET_DEMO,
    )


LIVE_BTC = candles(
    BTC, WED - HOUR, ["80000", "80100", "80500", "81000", "81000", "81000"], source=LIVE
)
"""Live closes: 13:00 80,000; 14:00 80,100; 15:00 80,500; 16:00-18:00 81,000."""

BUY = fill("a", Side.BUY, "0.0625", "80008", "3.0003", WED + timedelta(minutes=30))
SELL = fill("b", Side.SELL, "0.0625", "80900", "3.03375", WED + timedelta(hours=2, minutes=10))


# ------------------------------------------------------------------------------------------------
# The live mirror
# ------------------------------------------------------------------------------------------------


def test_live_mirror_replays_every_fill_at_live_prices() -> None:
    """The buy at 13:30 is replayed at the 13:00 live close (80,000), the sell at 15:10 at the 15:00
    close (80,500); each fee is the same rate on the live notional."""
    result = live_mirror([BUY, SELL], {BTC: LIVE_BTC}, Decimal(100_000), ci_resamples=0)
    assert result.spec == MIRROR_SPEC
    equity = {m.at: m.equity for m in result.marks}
    assert equity[WED - HOUR] == 100_000.0
    assert equity[WED] == 100_000.0
    assert equity[WED + HOUR] == pytest.approx(94_997 + 0.0625 * 80_100)
    assert equity[WED + 2 * HOUR] == pytest.approx(94_997 + 0.0625 * 80_500)
    assert equity[WED + 3 * HOUR] == pytest.approx(100_025.23125)
    (trade,) = result.trades
    assert trade.entry_avg == Decimal(80_000)
    assert trade.exit_avg == Decimal(80_500)
    assert trade.gross_pnl == Decimal("31.25")
    assert trade.fees == Decimal("6.01875")
    assert trade.net_pnl == Decimal("25.23125")
    assert trade.exit_reason == "mirror"
    assert result.metrics.fees_paid == pytest.approx(6.01875)
    marks = [m.equity for m in result.marks]
    assert result.metrics.turnover == pytest.approx(10_031.25 / (sum(marks) / len(marks)))


def test_live_mirror_on_an_explicit_grid() -> None:
    result = live_mirror(
        [BUY, SELL],
        {BTC: LIVE_BTC},
        Decimal(100_000),
        start=WED,
        until=WED + 2 * HOUR,
        ci_resamples=0,
    )
    assert [m.at for m in result.marks] == [WED, WED + HOUR, WED + 2 * HOUR]
    assert result.trades == ()  # the sell at 15:10 is after the grid ends
    with pytest.raises(ValueError, match="precedes the start"):
        live_mirror(
            [BUY], {BTC: LIVE_BTC}, Decimal(100_000), start=WED + HOUR, until=WED + 2 * HOUR
        )


def test_live_mirror_with_minute_candles_prices_at_the_minute() -> None:
    minutes = [
        *candles(BTC, WED - HOUR, ["80000"], source=LIVE),
        *(
            c
            for i in range(60)
            for c in candles(
                BTC, WED + timedelta(minutes=i), [str(80_000 + i)], source=LIVE, interval="1m"
            )
        ),
    ]
    result = live_mirror([BUY], {BTC: minutes}, Decimal(100_000), until=WED + HOUR, ci_resamples=0)
    # The 13:29 candle closes at 13:30 at 80,029: the fill's live price. The 14:00 mark is the
    # 13:59 candle's close, 80,059. The fee is 3.0003 scaled to the live notional.
    fee = 3.0003 * 80_029 / 80_008
    expected = 100_000 + 0.0625 * (80_059 - 80_029) - fee
    assert result.marks[-1].equity == pytest.approx(expected, rel=1e-12)
    (only,) = [m for m in result.marks if m.at == WED + HOUR]
    assert only.gross_weight == pytest.approx(0.0625 * 80_059 / expected)


def test_live_mirror_refuses_what_it_cannot_price() -> None:
    late = fill("c", Side.BUY, "0.0625", "80000", "3", WED + timedelta(hours=8))
    with pytest.raises(ValueError, match="no live BTCUSDT close"):
        live_mirror([late], {BTC: LIVE_BTC}, Decimal(100_000), until=WED + 9 * HOUR)
    with pytest.raises(ValueError, match="fee"):
        live_mirror(
            [fill("d", Side.BUY, "1", "1", "1", WED, coin="BGB")], {BTC: LIVE_BTC}, Decimal(1000)
        )
    with pytest.raises(ValueError, match="demo"):
        live_mirror([], {BTC: candles(BTC, WED, ["1"])}, Decimal(1000))
    with pytest.raises(ValueError, match="interval"):
        live_mirror([], {BTC: candles(BTC, WED, ["1"], source=LIVE, interval="2H")}, Decimal(1000))
    with pytest.raises(ValueError, match="positive, finite"):
        live_mirror([], {BTC: LIVE_BTC}, Decimal(0))
    with pytest.raises(ValueError, match="nothing to mirror"):
        live_mirror([], {}, Decimal(1000))


# ------------------------------------------------------------------------------------------------
# The weekend counterfactual
# ------------------------------------------------------------------------------------------------


def _nvda_book(at: datetime) -> BookState:
    position = Position(
        symbol=NVDA,
        qty=Decimal(25),
        avg_entry=Decimal(200),
        opened_at=at - timedelta(hours=6),
        last_increase_at=at - timedelta(hours=6),
        realized_pnl=Decimal(0),
        fees_paid=Decimal(3),
        stop_price=None,
        stop_venue_id=None,
        last_decision_id="d0",
    )
    return BookState(
        as_of=at,
        mark_source=PriceSource.DEMO,
        starting_equity=Decimal(100_000),
        equity=Decimal(100_000),
        peak_equity=Decimal(100_000),
        day_open_equity=Decimal(100_000),
        positions={NVDA: position},
        marks={NVDA: Decimal(200)},
        fees_today=Decimal(0),
        fees_total=Decimal(0),
        realized_total=Decimal(0),
        rebalances_today={},
        consecutive_losses=0,
        activation=Activation.ACTIVE,
    )


BREAKER = BreakerState(activation=Activation.ACTIVE, since=WED, trips=())


def _freeze_ruling() -> KernelRuling:
    """The protective loop at Friday 20:00 closing a held 5% NVDA long (G2)."""
    kernel = RiskKernel(POLICY_V1, ManualClock(FRI_20))
    ruling = kernel.protective(
        book=_nvda_book(FRI_20), inputs=inputs({NVDA: "200"}, at=FRI_20), breaker=BREAKER
    )
    assert ruling is not None
    assert ruling.protective_reason is ProtectiveReason.WEEKEND_FREEZE
    return ruling


def _decision_ruling(record: DecisionRecord, at: datetime) -> KernelRuling:
    kernel = RiskKernel(POLICY_V1, ManualClock(at))
    return kernel.rule(
        proposed=record.proposed_weights,
        book=_nvda_book(at).model_copy(update={"positions": {}, "marks": {}}),
        inputs=inputs({NVDA: "200"}, at=at),
        context=RulingContext(
            decision_id=record.decision_id, protective_reason=None, grounding=dict(record.grounding)
        ),
        breaker=BREAKER,
        guards=ALL_GUARDS,
    )


def _weekend_live(closes_from_fri_19: list[str]) -> dict[str, list[Candle]]:
    return {NVDA: candles(NVDA, FRI_20 - HOUR, closes_from_fri_19, source=LIVE)}


def test_the_freeze_closes_a_held_leg_that_live_kept_trading() -> None:
    hours = int((MON - FRI_20) / HOUR)  # 52
    closes = ["200", *[str(200 + 4 * (i + 1) / hours) for i in range(hours)], "204"]
    result = weekend_counterfactual([], [_freeze_ruling()], _weekend_live(closes), ci_resamples=0)
    assert result.spec == WEEKEND_SPEC
    assert result.marks[0].at == FRI_20
    assert result.marks[-1].at == MON
    assert result.marks[-1].equity == pytest.approx(1.0 + 0.05 * (204 / 200 - 1))
    (trade,) = result.trades
    assert (trade.opened_at, trade.closed_at) == (FRI_20, MON)
    assert trade.net_pnl == pytest.approx(Decimal("0.001"))
    assert trade.exit_reason == "weekend_counterfactual"
    assert result.metrics.fees_paid == 0.0


def test_a_weekend_request_runs_until_the_model_decides_again() -> None:
    saturday = decision("sat", {NVDA: 0.05}, at=SAT, snapshot_id="s-sat")
    sunday = decision("sun", {NVDA: 0.0}, at=SUN, snapshot_id="s-sun")
    rulings = [_decision_ruling(saturday, SAT), _decision_ruling(sunday, SUN)]
    (episode,) = weekend_episodes([saturday, sunday], rulings)
    assert (episode.symbol, episode.weight, episode.start, episode.end) == (NVDA, 0.05, SAT, SUN)
    assert episode.decision_id == "sat"
    hours = int((SUN - FRI_20) / HOUR) + 3
    closes = ["200", *[str(200 + i * 0.1) for i in range(1, hours)]]
    result = weekend_counterfactual(
        [saturday, sunday], rulings, _weekend_live(closes), ci_resamples=0
    )
    (trade,) = result.trades
    assert trade.decision_ids == ("sat",)
    assert (trade.opened_at, trade.closed_at) == (SAT, SUN)


def test_a_leg_another_guard_would_also_refuse_is_not_an_episode() -> None:
    """Ungrounded: G9 refuses it whatever the weekend does, so G2 took nothing away."""
    saturday = decision(
        "sat", {NVDA: 0.05}, at=SAT, snapshot_id="s", grounding={NVDA: ungrounded()}
    )
    ruling = _decision_ruling(saturday, SAT)
    inst = ruling.instrument(NVDA)
    assert inst is not None
    assert weight_without_g2(inst, ruling.book_rulings, POLICY_V1) == 0.0
    assert weekend_episodes([saturday], [ruling]) == []


def test_crypto_and_outages_are_not_episodes() -> None:
    sat_btc = decision("b", {BTC: 0.05}, at=SAT, snapshot_id="s", grounding={BTC: grounded()})
    kernel = RiskKernel(POLICY_V1, ManualClock(SAT))
    ruling = kernel.rule(
        proposed=sat_btc.proposed_weights,
        book=_nvda_book(SAT).model_copy(update={"positions": {}, "marks": {}}),
        inputs=inputs({BTC: "80000"}, at=SAT),
        context=RulingContext(
            decision_id="b", protective_reason=None, grounding=dict(sat_btc.grounding)
        ),
        breaker=BREAKER,
        guards=ALL_GUARDS,
    )
    assert weekend_episodes([sat_btc], [ruling]) == []
    freeze = _freeze_ruling()
    outage = decision("o", {}, at=FRI_20, snapshot_id="s", outcome=LlmOutcome.TIMEOUT)
    assert len(weekend_episodes([outage], [freeze])) == 1  # a protective ruling needs no decision


def test_no_episodes_gives_an_empty_arm() -> None:
    result = weekend_counterfactual([], [], {}, ci_resamples=0)
    assert result.marks == ()
    assert result.trades == ()
    assert result.metrics.n_hours == 0
    assert result.metrics.total_return == 0.0


def test_reopen_after() -> None:
    rule = POLICY_V1.weekend
    assert reopen_after(WED, rule) == MON
    assert reopen_after(FRI_20, rule) == MON
    assert reopen_after(SAT, rule) == MON
    assert reopen_after(MON, rule) == MON + timedelta(weeks=1)
    assert reopen_after(MON - timedelta(seconds=1), rule) == MON


def test_weight_without_g2_on_a_protective_exit() -> None:
    ruling = _freeze_ruling()
    inst = ruling.instrument(NVDA)
    assert inst is not None
    assert inst.binding_guard is GuardId.G2_WEEKEND_FREEZE
    assert weight_without_g2(inst, ruling.book_rulings, POLICY_V1) == pytest.approx(0.05)
