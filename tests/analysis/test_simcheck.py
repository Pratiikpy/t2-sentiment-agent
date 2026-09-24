"""The simulator checked against the venue, and the arm every baseline is ranked against.

Review finding: the coin flips and baselines are filled by the simulator's cost model, the live book
by the venue, so ranking the live book among them reads any systematic venue-vs-model price gap as
skill. The comparator is the governed replica; the live book is published beside it with the gap.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from analysis.test_baselines import ours
from helpers import make_intent
from sentiment_agent.analysis.baselines import (
    REPLICA_ARM_ID,
    coin_flip_summary,
    comparator_arm,
)
from sentiment_agent.analysis.simcheck import model_price, simulator_check, slippage_bps
from sentiment_agent.analysis.twin import GOVERNED_REPLICA_SPEC
from sentiment_agent.types import ArmResult, Fill, FillVenue, OrderIntent, Side

T0 = datetime(2026, 9, 23, 12, tzinfo=UTC)
HALF = Decimal("0.0001")  # 1 bp half spread


def fill_for(intent: OrderIntent, price: str, *, exec_id: str = "e1") -> Fill:
    p = Decimal(price)
    return Fill(
        exec_id=exec_id,
        venue_order_id="o1",
        client_oid=intent.client_oid,
        symbol=intent.symbol,
        side=intent.side,
        exec_price=p,
        exec_qty=intent.qty,
        exec_value=p * intent.qty,
        fee_paid=p * intent.qty * Decimal("0.0006"),
        fee_coin="USDT",
        trade_scope="taker",
        trade_side="open",
        exec_pnl=None,
        executed_at=T0,
        venue=FillVenue.BITGET_DEMO,
    )


def renamed(arm: ArmResult, arm_id: str, equity: list[float]) -> ArmResult:
    times = [m.at for m in arm.marks]
    fresh = ours(times, 0.05, equity)
    return fresh.model_copy(
        update={
            "spec": fresh.spec.model_copy(update={"arm_id": arm_id}),
            "metrics": fresh.metrics.model_copy(update={"arm_id": arm_id}),
        }
    )


def test_the_replica_id_is_the_twin_module_s() -> None:
    assert GOVERNED_REPLICA_SPEC.arm_id == REPLICA_ARM_ID


def test_the_model_price_is_the_simulator_s_taker_price() -> None:
    buy = make_intent(side=Side.BUY, price="100")
    sell = make_intent(side=Side.SELL, price="100", purpose=buy.purpose)
    assert model_price(buy, HALF) == Decimal("100.0100")
    assert model_price(sell, HALF) == Decimal("99.9900")


@pytest.mark.parametrize(
    ("side", "price", "expected"),
    [
        (Side.BUY, "100.02", 1.0),  # paid 1 bp above the model's 100.01: worse
        (Side.BUY, "100.00", -1.0),  # better than the model
        (Side.SELL, "99.98", 1.0),  # sold 1 bp below the model's 99.99: worse
        (Side.SELL, "100.00", -1.0),
    ],
)
def test_slippage_is_positive_when_the_venue_filled_worse(
    side: Side, price: str, expected: float
) -> None:
    intent = make_intent(side=side, price="100")
    assert slippage_bps(fill_for(intent, price), intent, HALF) == pytest.approx(expected, abs=0.01)


def test_the_check_prices_our_fills_and_counts_the_venue_s_apart() -> None:
    times = [T0 + timedelta(hours=h) for h in range(3)]
    live = ours(times, 0.05, [100_000.0, 100_050.0, 100_100.0])
    replica = renamed(live, REPLICA_ARM_ID, [100_000.0, 100_080.0, 100_160.0])
    intent = make_intent(side=Side.BUY, price="100")
    ours_fill = fill_for(intent, "100.03")
    stop_fill = fill_for(intent, "96").model_copy(update={"client_oid": None, "exec_id": "e2"})
    check = simulator_check(
        [ours_fill, stop_fill],
        {intent.client_oid: intent}.get,
        lambda _symbol: HALF,
        live=live,
        replica=replica,
    )
    assert check.ranked_arm_id == REPLICA_ARM_ID
    assert check.fills_priced == 1
    assert check.fills_unpriced == 1
    assert check.slippage.median_bps == pytest.approx(2.0, abs=0.01)
    assert check.return_gap == pytest.approx(0.001 - 0.0016)


def test_without_a_replica_the_live_book_is_ranked_and_says_so() -> None:
    times = [T0 + timedelta(hours=h) for h in range(2)]
    live = ours(times, 0.0)
    check = simulator_check([], lambda _oid: None, lambda _s: HALF, live=live, replica=None)
    assert check.ranked_arm_id == live.spec.arm_id
    assert check.return_gap is None
    assert check.slippage.n == 0


def test_coin_flips_rank_the_replica_not_the_live_book() -> None:
    times = [T0 + timedelta(hours=h) for h in range(3)]
    live = ours(times, 0.05, [100_000.0, 100_500.0, 101_000.0])  # a venue gift of +1%
    replica = renamed(live, REPLICA_ARM_ID, [100_000.0, 99_990.0, 99_980.0])
    flips = [
        renamed(live, f"baseline_coin_flip_s{i:04d}", [100_000.0, 100_000.0 + i, 100_000.0 + i])
        for i in range(10)
    ]
    arms = [replica, *flips]
    chosen = comparator_arm(live, arms)
    assert chosen.spec.arm_id == REPLICA_ARM_ID
    summary = coin_flip_summary(chosen, arms)
    assert summary.ranked_arm_id == REPLICA_ARM_ID
    assert summary.share_below_ours_total_return == 0.0, "the replica lost to every flip"
    naive = coin_flip_summary(live, arms)
    assert naive.share_below_ours_total_return == 1.0, "the live book would have looked best"
    assert comparator_arm(live, flips) is live
