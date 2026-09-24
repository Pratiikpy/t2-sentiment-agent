"""Metrics against hand-computed series and against the envelope's own code on the same input."""

import ast
import math
import random
import statistics
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from sentiment_agent.analysis.bootstrap import percentile
from sentiment_agent.analysis.metrics import (
    ANNUALISE,
    BOOK_ARM_ID,
    CI_METRICS,
    book_arm,
    book_metrics,
    hourly_returns,
    max_drawdown,
    max_drawdown_of,
    mean,
    metric_set,
    pstdev,
    sharpe_ann,
    sharpe_se_ann,
    sortino_ann,
    total_return_of,
    win_rate,
)
from sentiment_agent.types import (
    ArmMark,
    ClosedTrade,
    Fill,
    FillVenue,
    MarkPoint,
    Side,
)

T = datetime(2026, 9, 23, 13, 0, tzinfo=UTC)
ENVELOPE = Path(__file__).resolve().parents[2] / "validation" / "demo_venue" / "envelope_clean.py"


def marks(equity: Sequence[float], start: datetime = T) -> list[ArmMark]:
    return [
        ArmMark(at=start + timedelta(hours=i), equity=e, gross_weight=0.1, net_weight=0.05)
        for i, e in enumerate(equity)
    ]


def trade(net: str, *, i: int = 0) -> ClosedTrade:
    return ClosedTrade(
        symbol="BTCUSDT",
        opened_at=T + timedelta(hours=i),
        closed_at=T + timedelta(hours=i + 1),
        direction=1,
        entry_avg=Decimal(100),
        exit_avg=Decimal(101),
        max_abs_qty=Decimal(1),
        gross_pnl=Decimal(net),
        fees=Decimal(0),
        net_pnl=Decimal(net),
        decision_ids=(),
        exit_reason="model_close",
    )


# ------------------------------------------------------------------------------------------------
# Hand-computed
# ------------------------------------------------------------------------------------------------

EQUITY = [100.0, 110.0, 99.0, 99.0, 108.9]
"""Returns +10%, -10%, 0, +10%: mean 0.025, population variance 0.006875."""


def test_hourly_returns_hand() -> None:
    assert hourly_returns(EQUITY) == pytest.approx([0.1, -0.1, 0.0, 0.1], abs=1e-15)
    assert hourly_returns([]) == []
    assert hourly_returns([5.0]) == []


def test_sharpe_hand() -> None:
    r = hourly_returns(EQUITY)
    expected = 0.025 / math.sqrt(0.006875) * math.sqrt(8760)
    assert sharpe_ann(r) == pytest.approx(expected, rel=1e-12)
    assert math.sqrt(8760) == ANNUALISE


def test_sharpe_se_hand() -> None:
    r = hourly_returns(EQUITY)
    s_h = 0.025 / math.sqrt(0.006875)
    expected = math.sqrt((1 + 0.5 * s_h**2) / 4) * math.sqrt(8760)
    assert sharpe_se_ann(r) == pytest.approx(expected, rel=1e-12)


def test_sharpe_se_is_about_eleven_at_72_hours() -> None:
    rng = random.Random(3)  # noqa: S311 - seeded, reproducible test data
    r = [rng.gauss(0, 0.001) for _ in range(72)]
    se = sharpe_se_ann(r)
    assert se is not None
    assert 10.5 < se < 12.5  # policy.METRICS: "about 11 at n = 72 with S_h near 0"


def test_sortino_hand() -> None:
    r = hourly_returns(EQUITY)
    downside = math.sqrt(0.1**2 / 4)
    assert sortino_ann(r) == pytest.approx(0.025 / downside * math.sqrt(8760), rel=1e-12)


def test_max_drawdown_hand() -> None:
    assert max_drawdown(EQUITY) == pytest.approx(99 / 110 - 1, rel=1e-12)
    assert max_drawdown([100.0, 101.0, 102.0]) == 0.0
    assert max_drawdown([]) == 0.0
    assert max_drawdown([100.0, 50.0, 200.0, 100.0]) == pytest.approx(-0.5)


def test_win_rate_hand() -> None:
    trades = [trade("5"), trade("-2"), trade("0"), trade("0.01")]
    assert win_rate(trades) == 0.5  # a trade that nets exactly zero is not a win
    assert win_rate([]) is None


def test_path_statistics() -> None:
    assert total_return_of([0.1, -0.1]) == pytest.approx(1.1 * 0.9 - 1)
    assert total_return_of([]) == 0.0
    assert max_drawdown_of([0.1, -0.1, 0.0, 0.1]) == pytest.approx(0.99 / 1.1 - 1)


# ------------------------------------------------------------------------------------------------
# Undefined is None, never a number
# ------------------------------------------------------------------------------------------------


def test_undefined_statistics_are_none() -> None:
    assert sharpe_ann([]) is None
    assert sharpe_se_ann([]) is None
    assert sortino_ann([]) is None
    assert sharpe_ann([0.0] * 10) is None
    assert sharpe_ann([0.001]) is None


def test_constant_nonzero_returns_have_no_sharpe() -> None:
    """A constant series must not produce a spurious tiny deviation and an enormous ratio."""
    r = [0.1] * 72  # 0.1 is not exact in binary; a naive two-pass stdev leaves ~1e-17
    assert pstdev(r) == 0.0
    assert sharpe_ann(r) is None
    assert sharpe_se_ann(r) is None


def test_sortino_without_losses_is_none() -> None:
    assert sortino_ann([0.01, 0.02, 0.0]) is None


def test_non_finite_inputs_are_refused() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        sharpe_ann([0.1, math.nan])
    with pytest.raises(ValueError, match="non-finite"):
        hourly_returns([1.0, math.inf])
    with pytest.raises(ValueError, match="not positive"):
        hourly_returns([1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="not positive"):
        max_drawdown([-1.0, -2.0])


def test_primitives_match_statistics() -> None:
    rng = random.Random(11)  # noqa: S311 - seeded, reproducible test data
    for _ in range(50):
        x = [rng.gauss(0.0001, 0.003) for _ in range(rng.randint(2, 200))]
        assert mean(x) == pytest.approx(statistics.fmean(x), rel=1e-13)
        assert pstdev(x) == pytest.approx(statistics.pstdev(x), rel=1e-12)
    assert mean([]) is None
    assert pstdev([]) is None


# ------------------------------------------------------------------------------------------------
# The envelope's own code, on the same input
# ------------------------------------------------------------------------------------------------


class _CapturingStatistics:
    """Stands in for ``statistics`` inside the envelope's ``run`` and records the returns series
    it computes its Sharpe ratio from."""

    def __init__(self) -> None:
        self.returns: list[list[float]] = []

    def pstdev(self, x: Sequence[float]) -> float:
        self.returns.append(list(x))
        return statistics.pstdev(x)

    def mean(self, x: Sequence[float]) -> float:
        return float(statistics.mean(x))


def _envelope_namespace(
    px: dict[str, dict[int, float]],
) -> tuple[dict[str, Any], _CapturingStatistics]:
    """``run``, ``tradable``, ``pct`` and ``SIDE_COST`` taken from ``envelope_clean.py`` itself, by
    parsing it (its top level fetches candles, so it cannot simply be imported)."""
    tree = ast.parse(ENVELOPE.read_text(encoding="utf-8"))
    keep: list[ast.stmt] = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in {"run", "tradable", "pct"})
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "SIDE_COST" for t in node.targets)
        )
    ]
    assert {getattr(n, "name", "SIDE_COST") for n in keep} == {
        "run",
        "tradable",
        "pct",
        "SIDE_COST",
    }
    shim = _CapturingStatistics()
    import datetime as dt

    namespace: dict[str, Any] = {
        "px": px,
        "H": 3600 * 1000,
        "CRYPTO": ["BTCUSDT"],
        "dt": dt,
        "math": math,
        "st": shim,
    }
    code = compile(ast.Module(body=keep, type_ignores=[]), str(ENVELOPE), "exec")
    exec(code, namespace)  # noqa: S102 - the repository's own evidence script, parsed not fetched
    return namespace, shim


def _synthetic_prices(seed: int) -> tuple[dict[str, dict[int, float]], int]:
    rng = random.Random(seed)  # noqa: S311 - seeded, reproducible test data
    h = 3600 * 1000
    t0 = int(datetime(2026, 9, 21, 13, tzinfo=UTC).timestamp() * 1000)  # a Monday
    px: dict[str, dict[int, float]] = {}
    for symbol in ("BTCUSDT", "NVDAUSDT", "AAPLUSDT", "TSLAUSDT", "METAUSDT", "AMZNUSDT"):
        price = 100.0 + rng.random() * 100
        series = {}
        for i in range(-2, 80):
            series[t0 + i * h] = price
            price *= math.exp(rng.gauss(0, 0.012))
        px[symbol] = series
    return px, t0


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6])
def test_metrics_agree_with_envelope_clean(seed: int) -> None:
    px, t0 = _synthetic_prices(seed)
    namespace, shim = _envelope_namespace(px)
    ret, mdd, sharpe, closed = namespace["run"](t0, random.Random(seed))  # noqa: S311 - seeded, reproducible test data
    (rets,) = shim.returns
    equity = [1.0]
    for r in rets:
        equity.append(equity[-1] * (1 + r))
    assert hourly_returns(equity) == pytest.approx(rets, rel=1e-9, abs=1e-15)
    ours = sharpe_ann(rets)
    if math.isnan(sharpe):
        assert ours is None
    else:
        assert ours == pytest.approx(sharpe, rel=1e-12)
    assert max_drawdown(equity) == pytest.approx(mdd, abs=1e-12)
    closed_trades = [trade(repr(c), i=i) for i, c in enumerate(closed)]
    expected = sum(x > 0 for x in closed) / len(closed) if closed else None
    assert win_rate(closed_trades) == expected
    assert total_return_of(rets) == pytest.approx(equity[-1] - 1, rel=1e-12)
    # The envelope charges the exits of positions still open after its last mark, which no mark
    # shows: its return can only be at or below the marked equity's.
    assert ret <= equity[-1] - 1 + 1e-12


def test_percentile_is_the_envelopes_rule() -> None:
    namespace, _ = _envelope_namespace({})
    pct = namespace["pct"]
    rng = random.Random(5)  # noqa: S311 - seeded, reproducible test data
    for _ in range(200):
        values = [rng.gauss(0, 1) for _ in range(rng.randint(1, 60))]
        for q in (0.0, 0.05, 0.5, 0.95, 1.0):
            assert percentile(sorted(values), q) == pct(values, q)


# ------------------------------------------------------------------------------------------------
# metric_set
# ------------------------------------------------------------------------------------------------


def test_metric_set_hand() -> None:
    result = metric_set(
        "arm", marks(EQUITY), [trade("5"), trade("-1")], traded_notional=90.0, fees=0.3
    )
    assert result.arm_id == "arm"
    assert result.n_hours == 4
    assert result.n_closed_trades == 2
    assert result.total_return == pytest.approx(0.089)
    assert result.max_drawdown == pytest.approx(99 / 110 - 1)
    assert result.win_rate == 0.5
    assert result.turnover == pytest.approx(90.0 / statistics.fmean(EQUITY))
    assert result.fees_paid == 0.3
    assert result.label == "descriptive, not inferential"
    assert set(result.ci90) <= set(CI_METRICS)


def test_metric_set_intervals_bracket_a_trending_series() -> None:
    rng = random.Random(8)  # noqa: S311 - seeded, reproducible test data
    equity = [10_000.0]
    for _ in range(72):
        equity.append(equity[-1] * (1 + rng.gauss(0.0002, 0.001)))
    result = metric_set("arm", marks(equity), [], traded_notional=0.0, fees=0.0, ci_resamples=500)
    assert set(result.ci90) == set(CI_METRICS)
    for name, (lo, hi) in result.ci90.items():
        assert lo <= hi, name
    lo, hi = result.ci90["sharpe_ann"]
    assert result.sharpe_ann is not None
    assert lo <= result.sharpe_ann <= hi


def test_metric_set_flat_arm() -> None:
    result = metric_set("flat", marks([1.0] * 10), [], traded_notional=0.0, fees=0.0)
    assert result.total_return == 0.0
    assert result.sharpe_ann is None
    assert result.sortino_ann is None
    assert result.win_rate is None
    assert result.turnover == 0.0
    # A book that never trades earned nothing at no risk on every resample: its Sharpe band is
    # 0..0 (the point Sharpe stays undefined), and Sortino is undefined on every resample.
    assert result.ci90 == {
        "total_return": (0.0, 0.0),
        "sharpe_ann": (0.0, 0.0),
        "max_drawdown": (0.0, 0.0),
    }
    assert result.ci90_undefined_share == {"sortino_ann": 1.0}


def test_a_mostly_flat_book_keeps_its_sharpe_band_and_states_what_it_rests_on() -> None:
    """Review finding: 72 hourly returns, 50 of them flat. The old rule dropped the Sharpe and
    Sortino bands on the first undefined resample and left a point Sharpe of ~9 with no band."""
    rng = random.Random(7)  # noqa: S311 - seeded, reproducible test data
    r = [0.0] * 50 + [abs(rng.gauss(0.0004, 0.0003)) for _ in range(22)]
    rng.shuffle(r)
    equity = [10_000.0]
    for x in r:
        equity.append(equity[-1] * (1 + x))
    result = metric_set("mostly_flat", marks(equity), [], traded_notional=0.0, fees=0.0)
    assert result.sharpe_ann is not None
    assert result.sharpe_ann > 5
    assert "sharpe_ann" in result.ci90, "the band is published"
    lo, hi = result.ci90["sharpe_ann"]
    assert lo <= hi
    assert "sharpe_ann" not in result.ci90_undefined_share, "an all-flat resample has Sharpe 0"
    if "sortino_ann" in result.ci90:
        assert 0.0 < result.ci90_undefined_share.get("sortino_ann", 0.0) < 1.0


def test_metric_set_without_intervals_and_without_marks() -> None:
    assert (
        metric_set("a", marks(EQUITY), [], traded_notional=1.0, fees=0.0, ci_resamples=0).ci90 == {}
    )
    empty = metric_set("a", [], [], traded_notional=0.0, fees=0.0)
    assert (empty.n_hours, empty.total_return, empty.max_drawdown, empty.turnover) == (
        0,
        0.0,
        0.0,
        0.0,
    )
    with pytest.raises(ValueError, match="no marks"):
        metric_set("a", [], [], traded_notional=5.0, fees=0.0)


def test_metric_set_refuses_an_off_grid_or_unordered_series() -> None:
    off = [ArmMark(at=T + timedelta(minutes=30), equity=1.0, gross_weight=0, net_weight=0)]
    with pytest.raises(ValueError, match="on the hour"):
        metric_set("a", off, [], traded_notional=0.0, fees=0.0)
    backwards = list(reversed(marks([1.0, 1.1])))
    with pytest.raises(ValueError, match="strictly increase"):
        metric_set("a", backwards, [], traded_notional=0.0, fees=0.0)
    with pytest.raises(ValueError, match="non-negative"):
        metric_set("a", marks([1.0, 1.1]), [], traded_notional=-1.0, fees=0.0)
    with pytest.raises(ValueError, match="cannot be negative"):
        metric_set("a", marks([1.0, 1.1]), [], traded_notional=0.0, fees=0.0, ci_resamples=-1)


def test_metric_set_is_deterministic() -> None:
    rng = random.Random(4)  # noqa: S311 - seeded, reproducible test data
    equity = [100.0 * (1 + rng.gauss(0, 0.01)) for _ in range(30)]
    first = metric_set("a", marks(equity), [], traded_notional=10.0, fees=0.1, ci_resamples=300)
    second = metric_set("a", marks(equity), [], traded_notional=10.0, fees=0.1, ci_resamples=300)
    assert first == second


# ------------------------------------------------------------------------------------------------
# The book, from its own ledger records
# ------------------------------------------------------------------------------------------------


def _mark_point(at: datetime, equity: str, gross: float) -> MarkPoint:
    return MarkPoint(
        at=at,
        equity_book=Decimal(equity),
        equity_venue=None,
        equity_live_mirror=None,
        gross_weight=gross,
        net_weight=gross,
        positions=(),
    )


def _fill(exec_id: str, side: Side, qty: str, price: str, fee: str, at: datetime) -> Fill:
    return Fill(
        exec_id=exec_id,
        venue_order_id=exec_id,
        client_oid=None,
        symbol="BTCUSDT",
        side=side,
        exec_price=Decimal(price),
        exec_qty=Decimal(qty),
        exec_value=Decimal(price) * Decimal(qty),
        fee_paid=Decimal(fee),
        fee_coin="USDT",
        trade_scope="taker",
        trade_side="open" if side is Side.BUY else "close",
        exec_pnl=None,
        executed_at=at,
        venue=FillVenue.BITGET_DEMO,
    )


def test_book_metrics_use_equity_book_and_every_fill() -> None:
    points = [
        _mark_point(T, "10000", 0.0),
        _mark_point(T + timedelta(hours=1), "10050", 0.05),
        _mark_point(T + timedelta(hours=2), "10020", 0.0),
    ]
    fills = [
        _fill("a", Side.BUY, "0.01", "50000", "0.3", T + timedelta(minutes=30)),
        _fill("b", Side.SELL, "0.01", "50200", "0.3012", T + timedelta(hours=1, minutes=30)),
    ]
    result = book_metrics(points, [trade("1.3988")], fills, ci_resamples=0)
    assert result.arm_id == BOOK_ARM_ID
    assert result.total_return == pytest.approx(0.002)
    assert result.turnover == pytest.approx(1002.0 / statistics.fmean([10000, 10050, 10020]))
    assert result.fees_paid == pytest.approx(0.6012)
    arm = book_arm(points, [trade("1.3988")], fills, ci_resamples=0)
    assert arm.metrics == result
    assert [m.equity for m in arm.marks] == [10000.0, 10050.0, 10020.0]
    assert arm.spec.arm_id == BOOK_ARM_ID
