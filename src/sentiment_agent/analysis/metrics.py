"""The quantitative record: every number the handbook's quantitative half reads, defined once.

The definitions are the ones pre-registered in ``POLICY_V1.metrics`` and hashed into the genesis,
and they are the same as ``validation/demo_venue/envelope_clean.py``, which produced the expected
envelope the result is read against (DESIGN.md §15). ``scripts/recompute.py`` reimplements every
function here with the standard library alone; the two must agree on any published record.

Definitions (``E`` is the hourly equity series, ``r`` its hourly returns, ``n = len(r)``):

* ``r_t = E_t / E_{t-1} - 1`` on the UTC hour grid (:func:`hourly_returns`).
* ``sharpe_ann = mean(r) / pstdev(r) * sqrt(8760)``: population standard deviation, as
  ``envelope_clean.py:65-67``. Undefined (``None``) when ``pstdev(r)`` is zero or there is no
  return. This departs on purpose from empyrical's sample standard deviation (``ddof=1``,
  ``stefan-jansen/empyrical-reloaded`` ``src/empyrical/stats.py:650-655``, commit ``efe6066``):
  the envelope that is the reference for this record uses ``pstdev``.
* ``sharpe_se_ann = sqrt((1 + S_h^2 / 2) / n) * sqrt(8760)`` with ``S_h = mean(r) / pstdev(r)``, the
  iid standard error of Lo (2002), "The Statistics of Sharpe Ratios" (Financial Analysts Journal
  58(4)), as pre-registered in ``policy.METRICS``. About 11 at n = 72.
* ``sortino_ann = mean(r) / sqrt(mean(min(r, 0)^2)) * sqrt(8760)``: downside deviation against
  zero over *all* n observations, the same construction as empyrical ``downside_risk``
  (``stats.py:816-828``). Undefined when no return is negative.
* ``max_drawdown = min_t (E_t / max_{s<=t} E_s - 1)``, non-positive; ``0.0`` for a series that never
  falls (``envelope_clean.py:62`` tracks ``peak`` and ``mdd`` the same way).
* ``win_rate = count(net_pnl > 0) / count(closed trades)``, a closed trade being one symbol flat to
  flat with both fees in ``net_pnl`` (``envelope_clean.py:82``). A trade that nets exactly zero is
  not a win. Undefined when no trade has closed.
* ``turnover = sum(|fill notional|) / mean(E)``.
* ``ci90``: moving-block bootstrap of ``r`` (:mod:`sentiment_agent.analysis.bootstrap`) for
  ``total_return``, ``sharpe_ann``, ``sortino_ann`` and ``max_drawdown``, recomputed on each
  resampled path, taken over the resamples where the statistic is defined, with the share of
  resamples where it is not in ``ci90_undefined_share`` (non-zero shares only). On a resample every
  return of which is exactly zero the Sharpe ratio is ``0``. Labelled descriptive, not inferential:
  72 hourly marks cannot separate skill from luck, and the record says so on its face.

**Numerics, fixed so two implementations agree to the last few bits.** ``mean`` is
``math.fsum(x) / n``. ``pstdev`` is exactly zero when every value is identical (tested with ``max ==
min``, so a constant series can never produce a spurious tiny deviation and an enormous Sharpe) and
otherwise ``sqrt(fsum((x - mean)^2) / n)``, two-pass. Against ``statistics.pstdev``, which
``envelope_clean.py`` uses, the difference is a few units in the last place (tested); the fsum form
is kept because the bootstrap evaluates it 10,000 times per metric and ``statistics`` computes in
exact fractions, about 15 times slower.

**Marks** are strictly increasing and on the hour. A missing hour is not invented: the return across
it is one observation. The book module writes a ``MARK`` every UTC hour, so gaps are what a crash
leaves, and they are visible in ``n_hours``.
"""

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal
from typing import Final

from sentiment_agent.analysis.bootstrap import (
    DEFAULT_LEVEL,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    block_bootstrap_bands,
)
from sentiment_agent.types import (
    ArmKind,
    ArmMark,
    ArmResult,
    ArmSpec,
    ClosedTrade,
    Fill,
    MarkPoint,
    MetricSet,
)

HOURS_PER_YEAR: Final = 8760
ANNUALISE: Final = math.sqrt(HOURS_PER_YEAR)
"""``sqrt(8760)``: hourly ratios to annual, as ``envelope_clean.py:67``."""

CI_METRICS: Final[tuple[str, ...]] = ("total_return", "sharpe_ann", "sortino_ann", "max_drawdown")
"""The metrics that get a bootstrap interval, in the order they are reported."""

BOOK_ARM_ID: Final = "ours_governed"
"""The arm id of the real, governed paper book (``metrics.json`` in the published record)."""


# ------------------------------------------------------------------------------------------------
# Primitives
# ------------------------------------------------------------------------------------------------


def _finite(values: Sequence[float], what: str) -> list[float]:
    out: list[float] = []
    for value in values:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{what} contains a non-finite value ({value!r})")
        out.append(number)
    return out


def mean(x: Sequence[float]) -> float | None:
    """``fsum(x) / n``; ``None`` for an empty series."""
    if not x:
        return None
    return math.fsum(x) / len(x)


def pstdev(x: Sequence[float]) -> float | None:
    """Population standard deviation, two-pass over ``fsum``; exactly ``0.0`` for a constant
    series; ``None`` for an empty one."""
    if not x:
        return None
    if max(x) == min(x):
        return 0.0
    m = math.fsum(x) / len(x)
    return math.sqrt(math.fsum((v - m) ** 2 for v in x) / len(x))


# ------------------------------------------------------------------------------------------------
# The metric functions
# ------------------------------------------------------------------------------------------------


def hourly_returns(equity: Sequence[float]) -> list[float]:
    """``E_t / E_{t-1} - 1`` for consecutive marks. Every equity that divides must be positive:
    a return from a non-positive equity is not defined, and the book that produced one is ruined."""
    values = _finite(equity, "equity")
    out: list[float] = []
    for before, after in itertools.pairwise(values):
        if before <= 0:
            raise ValueError(f"equity {before} is not positive; a return from it is undefined")
        out.append(after / before - 1.0)
    return out


def sharpe_ann(r: Sequence[float]) -> float | None:
    """``mean(r) / pstdev(r) * sqrt(8760)``; ``None`` when undefined."""
    values = _finite(r, "returns")
    sd = pstdev(values)
    m = mean(values)
    if sd is None or m is None or sd == 0.0:
        return None
    return m / sd * ANNUALISE


def sharpe_se_ann(r: Sequence[float]) -> float | None:
    """Lo (2002) iid standard error of the annualised Sharpe ratio; ``None`` when the Sharpe ratio
    itself is undefined."""
    values = _finite(r, "returns")
    sd = pstdev(values)
    m = mean(values)
    if sd is None or m is None or sd == 0.0:
        return None
    s_h = m / sd
    return math.sqrt((1.0 + 0.5 * s_h * s_h) / len(values)) * ANNUALISE


def sortino_ann(r: Sequence[float]) -> float | None:
    """``mean(r) / sqrt(mean(min(r, 0)^2)) * sqrt(8760)``; ``None`` when no return is negative."""
    values = _finite(r, "returns")
    if not values:
        return None
    downside = math.sqrt(math.fsum(min(v, 0.0) ** 2 for v in values) / len(values))
    if downside == 0.0:
        return None
    return math.fsum(values) / len(values) / downside * ANNUALISE


def max_drawdown(equity: Sequence[float]) -> float:
    """``min_t (E_t / max_{s<=t} E_s - 1)``: ``0.0`` for an empty or never-falling series."""
    values = _finite(equity, "equity")
    worst = 0.0
    peak: float | None = None
    for value in values:
        peak = value if peak is None else max(peak, value)
        if peak <= 0:
            raise ValueError(f"equity peak {peak} is not positive; a drawdown from it is undefined")
        worst = min(worst, value / peak - 1.0)
    return worst


def win_rate(trades: Sequence[ClosedTrade]) -> float | None:
    """Share of closed trades with ``net_pnl > 0``; ``None`` when none has closed."""
    if not trades:
        return None
    return sum(1 for t in trades if t.net_pnl > 0) / len(trades)


def total_return_of(r: Sequence[float]) -> float:
    """``prod(1 + r) - 1``: the total return of a return path (``0.0`` for an empty one). Used on
    bootstrap resamples, where there is no equity series, only returns."""
    value = 1.0
    for x in r:
        value *= 1.0 + x
    return value - 1.0


def max_drawdown_of(r: Sequence[float]) -> float:
    """The maximum drawdown of the equity path ``1, 1 * (1 + r_1), ...`` (bootstrap resamples)."""
    value = 1.0
    peak = 1.0
    worst = 0.0
    for x in r:
        value *= 1.0 + x
        peak = max(peak, value)
        if peak <= 0:
            raise ValueError("a return path whose equity peak is not positive has no drawdown")
        worst = min(worst, value / peak - 1.0)
    return worst


def sharpe_ann_of_path(r: Sequence[float]) -> float | None:
    """:func:`sharpe_ann` on a bootstrap resample, defined as ``0.0`` on a path with no return and
    no risk (every return exactly zero): a resample of a mostly flat book drawn only from its flat
    hours earned nothing at no risk, and dropping it would condition the band on the book looking
    active. A constant *non-zero* path stays undefined."""
    values = _finite(r, "returns")
    if values and max(values) == 0.0 and min(values) == 0.0:
        return 0.0
    return sharpe_ann(values)


CI_STATS: Final[Mapping[str, Callable[[Sequence[float]], float | None]]] = {
    "total_return": total_return_of,
    "sharpe_ann": sharpe_ann_of_path,
    "sortino_ann": sortino_ann,
    "max_drawdown": max_drawdown_of,
}
"""The statistic each interval is computed with, as a function of a (resampled) return path."""


# ------------------------------------------------------------------------------------------------
# A whole arm
# ------------------------------------------------------------------------------------------------


def check_marks(marks: Sequence[ArmMark]) -> None:
    """Marks are on the UTC hour and strictly increasing (the metric grid, ``policy.METRICS``)."""
    previous = None
    for mark in marks:
        at = mark.at
        if (at.minute, at.second, at.microsecond) != (0, 0, 0):
            raise ValueError(f"mark at {at.isoformat()} is not on the hour")
        if previous is not None and at <= previous:
            raise ValueError(f"marks must strictly increase in time ({at.isoformat()})")
        previous = at


def metric_set(
    arm_id: str,
    marks: Sequence[ArmMark],
    trades: Sequence[ClosedTrade],
    *,
    traded_notional: float,
    fees: float,
    ci_resamples: int = DEFAULT_RESAMPLES,
    ci_seed: int = DEFAULT_SEED,
    ci_level: float = DEFAULT_LEVEL,
) -> MetricSet:
    """Every pre-registered metric of one arm.

    ``traded_notional`` and ``fees`` are in the same currency as the marks' equity (quote coin for a
    book, equity fractions for a unit-equity arm). ``ci_resamples=0`` skips the intervals, for the
    coin-flip seeds, whose uncertainty is the distribution across seeds rather than a resample of
    any one of them.
    """
    check_marks(marks)
    if not math.isfinite(traded_notional) or traded_notional < 0:
        raise ValueError(f"traded notional must be finite and non-negative, got {traded_notional}")
    if not math.isfinite(fees):
        raise ValueError(f"fees must be finite, got {fees}")
    if ci_resamples < 0:
        raise ValueError("ci_resamples cannot be negative")
    equity = [m.equity for m in marks]
    r = hourly_returns(equity)
    if equity:
        average = math.fsum(equity) / len(equity)
        if average <= 0:
            raise ValueError("mean equity is not positive; turnover is undefined")
        turnover = traded_notional / average
        total = equity[-1] / equity[0] - 1.0
    else:
        if traded_notional > 0:
            raise ValueError("notional was traded but there are no marks to divide it by")
        turnover = 0.0
        total = 0.0
    ci90: dict[str, tuple[float, float]] = {}
    undefined: dict[str, float] = {}
    if ci_resamples > 0:
        ci90, undefined = block_bootstrap_bands(
            r,
            {name: CI_STATS[name] for name in CI_METRICS},
            resamples=ci_resamples,
            seed=ci_seed,
            level=ci_level,
        )
    return MetricSet(
        arm_id=arm_id,
        n_hours=len(r),
        n_closed_trades=len(trades),
        total_return=total,
        sharpe_ann=sharpe_ann(r),
        sharpe_se_ann=sharpe_se_ann(r),
        sortino_ann=sortino_ann(r),
        max_drawdown=max_drawdown(equity),
        win_rate=win_rate(trades),
        turnover=turnover,
        fees_paid=fees,
        ci90=ci90,
        ci90_undefined_share=undefined,
    )


# ------------------------------------------------------------------------------------------------
# The real book, exactly as the published record computes it
# ------------------------------------------------------------------------------------------------


def book_marks(marks: Sequence[MarkPoint]) -> tuple[ArmMark, ...]:
    """The book's hourly ``MARK`` events as arm marks (``E = equity_book``, DESIGN.md §15.1)."""
    return tuple(
        ArmMark(
            at=m.at,
            equity=float(m.equity_book),
            gross_weight=m.gross_weight,
            net_weight=m.net_weight,
        )
        for m in marks
    )


def fill_totals(fills: Sequence[Fill]) -> tuple[Decimal, Decimal]:
    """``(sum |exec_value|, sum fee_paid)`` over the fills, exact."""
    notional = sum((abs(f.exec_value) for f in fills), Decimal(0))
    paid = sum((f.fee_paid for f in fills), Decimal(0))
    return notional, paid


def book_metrics(
    marks: Sequence[MarkPoint],
    trades: Sequence[ClosedTrade],
    fills: Sequence[Fill],
    *,
    ci_resamples: int = DEFAULT_RESAMPLES,
) -> MetricSet:
    """The governed book's metrics from its ledger: hourly ``MARK`` events, the book's closed trades
    and every ``FILL``. This is what ``public/metrics.json`` must hold for ``scripts/recompute.py``
    to pass: turnover from ``sum |exec_value|``, fees from ``sum fee_paid``."""
    notional, paid = fill_totals(fills)
    return metric_set(
        BOOK_ARM_ID,
        book_marks(marks),
        trades,
        traded_notional=float(notional),
        fees=float(paid),
        ci_resamples=ci_resamples,
    )


BOOK_SPEC: Final = ArmSpec(
    arm_id=BOOK_ARM_ID,
    kind=ArmKind.OURS_GOVERNED,
    title="The agent, governed",
    description="The paper book itself: Qwen's decisions after the risk kernel, filled on Bitget "
    "UTA Demo through Agent Hub, marked hourly at the Demo mark.",
    provenance="ledger MARK and FILL events; book/book.py closed trades",
    uses_llm=True,
    guards=(),
)
"""``guards`` is empty because the real book is not re-ruled here: every guard already ruled on it,
and its rulings are in the ledger."""


def book_arm(
    marks: Sequence[MarkPoint],
    trades: Sequence[ClosedTrade],
    fills: Sequence[Fill],
    *,
    ci_resamples: int = DEFAULT_RESAMPLES,
) -> ArmResult:
    """The governed book as an :class:`ArmResult`, the reference every other arm is read against."""
    return ArmResult(
        spec=BOOK_SPEC,
        marks=book_marks(marks),
        trades=tuple(trades),
        metrics=book_metrics(marks, trades, fills, ci_resamples=ci_resamples),
    )


__all__ = [
    "ANNUALISE",
    "BOOK_ARM_ID",
    "BOOK_SPEC",
    "CI_METRICS",
    "CI_STATS",
    "HOURS_PER_YEAR",
    "book_arm",
    "book_marks",
    "book_metrics",
    "check_marks",
    "fill_totals",
    "hourly_returns",
    "max_drawdown",
    "max_drawdown_of",
    "mean",
    "metric_set",
    "pstdev",
    "sharpe_ann",
    "sharpe_ann_of_path",
    "sharpe_se_ann",
    "sortino_ann",
    "total_return_of",
    "win_rate",
]
