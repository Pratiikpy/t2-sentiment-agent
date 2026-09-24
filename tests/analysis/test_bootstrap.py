"""The moving-block bootstrap: its protocol, determinism, and coverage."""

import math
import random
import statistics
from collections.abc import Sequence

import pytest

from sentiment_agent.analysis.bootstrap import (
    DEFAULT_LEVEL,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    block_bootstrap_bands,
    block_bootstrap_ci,
    block_bootstrap_cis,
    default_block,
    percentile,
    resample_starts,
)
from sentiment_agent.analysis.metrics import sharpe_ann, sortino_ann


def fmean(x: Sequence[float]) -> float:
    return math.fsum(x) / len(x)


def series(n: int, seed: int = 1) -> list[float]:
    rng = random.Random(seed)  # noqa: S311 - seeded, reproducible test data
    return [rng.gauss(0.0001, 0.002) for _ in range(n)]


def test_policy_defaults() -> None:
    assert (DEFAULT_RESAMPLES, DEFAULT_SEED, DEFAULT_LEVEL) == (10_000, 20260924, 0.90)


@pytest.mark.parametrize(
    ("n", "b"),
    [(1, 1), (2, 2), (8, 2), (9, 3), (27, 3), (28, 4), (64, 4), (65, 5), (72, 5), (125, 5),
     (1000, 10), (1001, 11), (999_999, 100), (1_000_000, 100)],
)  # fmt: skip
def test_default_block_is_the_exact_integer_cube_root_ceiling(n: int, b: int) -> None:
    assert default_block(n) == b
    assert b**3 >= n > (b - 1) ** 3


def test_default_block_refuses_an_empty_series() -> None:
    with pytest.raises(ValueError, match="at least one"):
        default_block(0)


def test_the_protocol_step_by_step() -> None:
    """Replaying the documented protocol by hand gives the function's interval exactly."""
    r = series(20, seed=9)
    b = default_block(len(r))
    rng = random.Random(DEFAULT_SEED)  # noqa: S311 - seeded, reproducible test data
    means = []
    for _ in range(300):
        path: list[float] = []
        for _k in range(math.ceil(len(r) / b)):
            s = rng.randrange(len(r) - b + 1)
            path.extend(r[s : s + b])
        means.append(fmean(path[: len(r)]))
    means.sort()
    q = (1 - 0.9) / 2
    expected = (means[int(q * 299)], means[int((1 - q) * 299)])
    assert block_bootstrap_ci(r, fmean, resamples=300) == expected


def test_resample_starts_stay_in_range() -> None:
    rng = random.Random(0)  # noqa: S311 - seeded, reproducible test data
    for _ in range(200):
        starts = resample_starts(72, 5, rng)
        assert len(starts) == 15
        assert all(0 <= s <= 67 for s in starts)


def test_deterministic_by_seed() -> None:
    r = series(72)
    first = block_bootstrap_ci(r, fmean, resamples=500)
    assert first == block_bootstrap_ci(r, fmean, resamples=500)
    assert first != block_bootstrap_ci(r, fmean, resamples=500, seed=DEFAULT_SEED + 1)


def test_many_statistics_share_resamples() -> None:
    """One pass for several statistics returns exactly what separate calls return."""
    r = series(72, seed=3)
    together = block_bootstrap_cis(
        r, {"mean": fmean, "sharpe": sharpe_ann, "sortino": sortino_ann}, resamples=400
    )
    assert together["mean"] == block_bootstrap_ci(r, fmean, resamples=400)
    assert together["sharpe"] == block_bootstrap_ci(r, sharpe_ann, resamples=400)
    assert together["sortino"] == block_bootstrap_ci(r, sortino_ann, resamples=400)


def test_interval_is_ordered_and_contains_the_estimate() -> None:
    r = series(72, seed=5)
    lo, hi = block_bootstrap_ci(r, fmean, resamples=2000) or (math.nan, math.nan)
    assert lo < fmean(r) < hi


def test_an_interval_rests_on_the_defined_resamples_and_says_how_many_were_not() -> None:
    """Review finding: dropping a band on one undefined resample removed exactly the bands of a
    mostly flat book. The band is kept over the defined resamples, the share beside it."""
    r = [0.0] * 30 + [0.01]
    intervals, shares = block_bootstrap_bands(r, {"sharpe": sharpe_ann}, resamples=2000)
    assert "sharpe" in intervals
    lo, hi = intervals["sharpe"]
    assert lo <= hi
    assert 0.0 < shares["sharpe"] < 1.0  # the constant (all-zero) resamples, counted
    assert block_bootstrap_ci(r, sharpe_ann, resamples=2000) == intervals["sharpe"]


def test_no_interval_only_when_undefined_on_every_resample() -> None:
    always_up = [0.01, 0.02, 0.03] * 5
    intervals, shares = block_bootstrap_bands(always_up, {"sortino": sortino_ann}, resamples=100)
    assert intervals == {}
    assert shares == {"sortino": 1.0}
    assert block_bootstrap_ci(always_up, sortino_ann, resamples=100) is None


def test_a_statistic_defined_everywhere_reports_no_share() -> None:
    _, shares = block_bootstrap_bands(series(72, seed=9), {"mean": fmean}, resamples=200)
    assert shares == {}


def test_no_interval_for_fewer_than_two_observations() -> None:
    assert block_bootstrap_ci([], fmean) is None
    assert block_bootstrap_ci([0.01], fmean) is None


def test_parameters_are_checked() -> None:
    r = series(10)
    with pytest.raises(ValueError, match="level"):
        block_bootstrap_ci(r, fmean, level=1.0)
    with pytest.raises(ValueError, match="resample"):
        block_bootstrap_ci(r, fmean, resamples=0)
    with pytest.raises(ValueError, match="at least 1"):
        block_bootstrap_ci(r, fmean, block=0)
    with pytest.raises(ValueError, match="longer than the series"):
        block_bootstrap_ci(r, fmean, block=11)
    with pytest.raises(ValueError, match="non-finite"):
        block_bootstrap_ci([0.1, math.nan], fmean)


def test_percentile_rule() -> None:
    values = sorted(float(i) for i in range(10_000))
    assert percentile(values, 0.05) == 499.0
    assert percentile(values, 0.95) == 9499.0
    assert percentile([7.0], 0.5) == 7.0
    with pytest.raises(ValueError, match="no values"):
        percentile([], 0.5)
    with pytest.raises(ValueError, match="q must be"):
        percentile([1.0], 1.5)


def test_coverage_of_the_mean_on_iid_data() -> None:
    """On iid returns with a known mean, the 90% interval covers it in roughly 90% of samples. A
    sanity check of the machinery, not a claim about hourly P&L (short samples under-cover)."""
    covered = 0
    trials = 200
    for trial in range(trials):
        rng = random.Random(1000 + trial)  # noqa: S311 - seeded, reproducible test data
        r = [rng.gauss(0.001, 0.01) for _ in range(72)]
        interval = block_bootstrap_ci(r, fmean, resamples=400, seed=trial)
        assert interval is not None
        covered += interval[0] <= 0.001 <= interval[1]
    assert 0.78 <= covered / trials <= 0.97


def test_blocks_widen_the_interval_under_autocorrelation() -> None:
    """With strongly autocorrelated returns an iid bootstrap (block 1) understates the spread; the
    default block does less so. This is why the record resamples blocks."""
    rng = random.Random(21)  # noqa: S311 - seeded, reproducible test data
    r = [0.0]
    for _ in range(199):
        r.append(0.8 * r[-1] + rng.gauss(0, 0.001))
    iid = block_bootstrap_ci(r, fmean, block=1, resamples=2000)
    blocked = block_bootstrap_ci(r, fmean, resamples=2000)
    assert iid is not None
    assert blocked is not None
    assert blocked[1] - blocked[0] > 1.3 * (iid[1] - iid[0])


def test_sample_mean_matches_statistics() -> None:
    r = series(50)
    assert fmean(r) == pytest.approx(statistics.fmean(r))
