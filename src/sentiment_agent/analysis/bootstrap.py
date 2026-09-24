"""Moving-block bootstrap intervals for statistics of an hourly return series.

**What it is for.** Every metric in the published record carries a 90% interval, labelled
descriptive, not inferential (``policy.METRICS`` ``ci90``). Hourly returns are autocorrelated
(positions are held for many hours, so one move lands in several consecutive returns), and an iid
bootstrap would understate the spread. Resampling contiguous blocks keeps that short-range
dependence inside each block (Künsch 1989, "The jackknife and the bootstrap for general stationary
observations", Annals of Statistics 17(3)).

**The resampling protocol**, fixed because ``scripts/recompute.py`` must reproduce every interval
from the published CSV with the standard library alone:

1. ``n = len(r)``; the block length ``b`` defaults to ``ceil(n^(1/3))``, computed exactly in
   integers (the smallest ``b`` with ``b^3 >= n``: 72 hours gives 5), not through a float cube root,
   which puts ``64 ** (1/3)`` at 3.9999999999999996. The ``n^(1/3)`` rate is Hall, Horowitz and Jing
   (1995), "On blocking rules for the bootstrap with dependent data", Biometrika 82(3).
2. ``rng = random.Random(seed)`` (Mersenne Twister, stable across CPython versions for an integer
   seed). For each of ``resamples`` resamples: ``k = ceil(n / b)`` block starts, each
   ``rng.randrange(n - b + 1)``, drawn in order; the blocks ``r[s:s+b]`` are concatenated and cut to
   ``n``. This is the moving-block scheme of ``bashtage/arch`` ``MovingBlockBootstrap.
   update_indices`` (``arch/bootstrap/base.py:1695-1710``, commit ``704bb70``, NCSA licence;
   studied, not copied): the same count of blocks, the same start range, the same truncation.
3. The statistic is evaluated on every resample. A resample on which it is undefined (``None`` or
   non-finite, e.g. a Sortino ratio on a resample with no losing hour) is counted, and the
   interval is taken over the resamples where it is defined, **with the share of undefined
   resamples published beside it** (:func:`block_bootstrap_bands`). An earlier rule dropped the
   whole interval on a single undefined resample; for a mostly flat book (a weekend freeze, a run
   of flat-with-reasons stances) that removed the Sharpe and Sortino bands exactly where their
   uncertainty matters most, leaving a point Sharpe of ~9 with no band. Conditioning on the
   defined resamples is stated, not hidden: the share says how much of the resampling it
   excludes. Where a statistic has a natural value on a degenerate path it is defined there
   instead (``analysis/metrics.py``: a Sharpe ratio of 0 on a path with no return and no risk).
   Only a statistic undefined on *every* resample gets no interval.
4. The interval is the ``alpha = (1 - level) / 2`` and ``1 - alpha`` percentiles by the rule
   ``envelope_clean.py:70`` uses for the pre-registered envelope, ``sorted(v)[int(q * (m - 1))]``
   (a lower nearest rank, no interpolation), so the record's bands and the envelope's are read the
   same way. ``arch`` interpolates (``np.percentile``, ``base.py:789``); at 10,000 resamples the two
   differ by less than one resample's spacing.

Because every statistic of one call is computed on the *same* resamples (one generator, one pass),
:func:`block_bootstrap_cis` returns exactly what separate :func:`block_bootstrap_ci` calls with the
same seed return, in a fraction of the time.
"""

import math
import random
from collections.abc import Callable, Mapping, Sequence
from typing import Final

DEFAULT_RESAMPLES: Final = 10_000
DEFAULT_SEED: Final = 20260924
DEFAULT_LEVEL: Final = 0.90

Statistic = Callable[[Sequence[float]], float | None]


def default_block(n: int) -> int:
    """``ceil(n^(1/3))`` in exact integer arithmetic: the smallest ``b >= 1`` with ``b^3 >= n``."""
    if n < 1:
        raise ValueError("a block length needs at least one observation")
    b = max(1, round(math.pow(n, 1.0 / 3.0)))
    while b**3 < n:
        b += 1
    while b > 1 and (b - 1) ** 3 >= n:
        b -= 1
    return b


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """``sorted_values[int(q * (m - 1))]``: the envelope's lower-nearest-rank rule
    (``envelope_clean.py:70``). ``sorted_values`` must already be sorted."""
    if not sorted_values:
        raise ValueError("no values to take a percentile of")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    return sorted_values[int(q * (len(sorted_values) - 1))]


def resample_starts(n: int, block: int, rng: random.Random) -> list[int]:
    """The block starts of one resample, drawn from ``rng`` in order (protocol step 2)."""
    count = -(-n // block)
    top = n - block + 1
    return [rng.randrange(top) for _ in range(count)]


def _validate(block: int | None, resamples: int, level: float) -> None:
    if block is not None and block < 1:
        raise ValueError(f"block length must be at least 1, got {block}")
    if resamples < 1:
        raise ValueError("at least one resample is needed")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be strictly between 0 and 1, got {level}")


def block_bootstrap_bands(
    r: Sequence[float],
    stats: Mapping[str, Statistic],
    *,
    block: int | None = None,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    level: float = DEFAULT_LEVEL,
) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    """Intervals for several statistics on the same resamples, and for each statistic undefined
    on some resample, the share of resamples it was undefined on (protocol step 3).

    A statistic undefined on every resample has no interval and a share of 1.0. With fewer than
    two observations nothing is computed and both maps are empty."""
    _validate(block, resamples, level)
    values = [float(x) for x in r]
    if any(not math.isfinite(x) for x in values):
        raise ValueError("returns contain a non-finite value")
    n = len(values)
    if n < 2 or not stats:
        return {}, {}
    b = default_block(n) if block is None else block
    if b > n:
        raise ValueError(f"block length {b} is longer than the series (n = {n})")
    rng = random.Random(seed)  # noqa: S311 - a reproducible resampler, not a secret
    draws: dict[str, list[float]] = {name: [] for name in stats}
    undefined: dict[str, int] = dict.fromkeys(stats, 0)
    for _ in range(resamples):
        path: list[float] = []
        for start in resample_starts(n, b, rng):
            path.extend(values[start : start + b])
        del path[n:]
        for name, stat in stats.items():
            value = stat(path)
            if value is None or not math.isfinite(value):
                undefined[name] += 1
                continue
            draws[name].append(value)
    alpha = (1.0 - level) / 2.0
    intervals: dict[str, tuple[float, float]] = {}
    shares: dict[str, float] = {}
    for name in stats:
        if undefined[name]:
            shares[name] = undefined[name] / resamples
        if not draws[name]:
            continue
        ordered = sorted(draws[name])
        intervals[name] = (percentile(ordered, alpha), percentile(ordered, 1.0 - alpha))
    return intervals, shares


def block_bootstrap_cis(
    r: Sequence[float],
    stats: Mapping[str, Statistic],
    *,
    block: int | None = None,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    level: float = DEFAULT_LEVEL,
) -> dict[str, tuple[float, float]]:
    """The intervals of :func:`block_bootstrap_bands` alone. Publish the undefined shares with
    them wherever a band may rest on part of the resamples."""
    intervals, _ = block_bootstrap_bands(
        r, stats, block=block, resamples=resamples, seed=seed, level=level
    )
    return intervals


def block_bootstrap_ci(
    r: Sequence[float],
    stat: Callable[[Sequence[float]], float | None],
    *,
    block: int | None = None,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    level: float = DEFAULT_LEVEL,
) -> tuple[float, float] | None:
    """The ``level`` interval of ``stat`` over moving-block resamples of ``r`` (module docstring).
    ``None`` with fewer than two observations, or when ``stat`` is undefined on every resample."""
    result = block_bootstrap_cis(
        r, {"stat": stat}, block=block, resamples=resamples, seed=seed, level=level
    )
    return result.get("stat")


__all__ = [
    "DEFAULT_LEVEL",
    "DEFAULT_RESAMPLES",
    "DEFAULT_SEED",
    "Statistic",
    "block_bootstrap_bands",
    "block_bootstrap_ci",
    "block_bootstrap_cis",
    "default_block",
    "percentile",
    "resample_starts",
]
