"""Same-clock baselines: what fixed rules and pure chance did on the agent's own timestamps.

The handbook asks what the agent adds over fixed-rule baselines (handbook:237). Each baseline here
is run by :class:`~sentiment_agent.analysis.armsim.ArmSimulator` on the same perception snapshots
the agent decided on, under the venue's guards only (``types.VENUE_GUARDS``: venue integrity,
weekend freeze, size, stop, daily kill, taker only, eligibility), so a comparison isolates the
decision-maker rather than the rules around it (DESIGN.md §14.3).

* **Flat** (``baseline_flat``): never trades. Its equity is the starting equity; its Sharpe ratio is
  undefined. The floor every other arm is read against.
* **BTC at our gross** (``baseline_btc_at_our_gross``): long BTCUSDT at the governed book's mean
  gross weight over the record (ex post, so its risk matches ours), capped at the policy's gross
  limit, entered at the first snapshot and re-set at the first snapshot of every later UTC day
  (so a kill or a refused entry is recovered the next day, as the agent's own book would be). G3's
  5% per-name cap is **not** applied to this arm: a single-name benchmark at our gross would
  otherwise be impossible whenever our gross is above 5%. Every other venue guard applies.
* **Crowd fade** (``baseline_crowd_fade``): the sub-theme's own mechanism with no language model,
  :func:`crowd_fade_targets` on every snapshot.
**Who the baselines are read against.** Every arm here is filled by the simulator's cost model
(Demo mark ± half the median snapshot spread, plus the taker fee), so the agent is ranked through
its **governed replica** (``analysis/twin.py``, ``twin_governed_replica``), which is simulated the
same way, never through the live book, whose fills are the venue's: a systematic gap between the
venue's prices and the model's would otherwise be counted as skill. :func:`comparator_arm` picks
the replica; the live book is published beside it with the gap measured
(``analysis/simcheck.py``). The BTC arm's weight is read from the same comparator.

* **Coin flip** (``baseline_coin_flip_s0000`` ...): the null of no edge, exactly the envelope's
  (``validation/demo_venue/envelope_clean.py:40-48``): at its first snapshot, and then at the first
  snapshot at least the mandate's minimum horizon (24h) after its previous draw, it closes
  everything and draws ``round(gross_max / per_name_max)`` = 5 names uniformly from those it could
  open (US-session names are left out from the weekend no-open buffer until the reopen, as the
  envelope's ``tradable`` does), each long or short with probability one half, at ``per_name_max``.
  One seed per arm, ``random.Random(20260924 + i)``, 1,000 seeds by default, reported as a
  distribution (:func:`coin_flip_summary`), never as a single arm.

**The crowd-fade rule**, fixed before the window and taken from Bitget's own reading of these
signals (``bitget-signal/skills/sentiment-analyst/references/signal-guide.md``, "Long/Short Ratio"
and "Funding Rate"; ``policy.triggers`` for the funding threshold):

* live funding z-score above ``+2`` (longs paying far above normal) votes short; below ``-2`` votes
  long. Strictly beyond the threshold, as the ``funding_zscore`` trigger fires
  (``events/triggers.py``).
* retail long share ``ratio / (1 + ratio)`` above 0.65 ("longs very crowded") votes short; below
  0.45 ("shorts dominant") votes long; the 0.45-0.65 band is Bitget's "balanced".
* the fade is taken at ``per_name_max`` in the direction of the votes when at least one vote is cast
  and none points the other way; conflicting votes, or none, mean flat. A symbol with no features
  in the snapshot is not addressed (a held position in it is kept, as a hold).
"""

import math
import random
import statistics
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Final, Literal

from sentiment_agent.analysis.armsim import ArmSimulator, hour_floor
from sentiment_agent.analysis.bootstrap import percentile
from sentiment_agent.kernel.guards import weekend_phase
from sentiment_agent.types import (
    VENUE_GUARDS,
    ArmKind,
    ArmResult,
    ArmSpec,
    GuardId,
    KernelInputs,
    Model,
    PerceptionSnapshot,
    Policy,
)

FLAT_ID: Final = "baseline_flat"
BTC_AT_OUR_GROSS_ID: Final = "baseline_btc_at_our_gross"
CROWD_FADE_ID: Final = "baseline_crowd_fade"
COIN_FLIP_PREFIX: Final = "baseline_coin_flip"
COIN_FLIP_SEED_BASE: Final = 20260924
BTC_SYMBOL: Final = "BTCUSDT"

LONG_SHARE_CROWDED: Final = 0.65
"""Retail long share above which Bitget's signal guide reads longs as very crowded."""
LONG_SHARE_SHORTS_CROWDED: Final = 0.45
"""Retail long share below which Bitget's signal guide reads shorts as dominant."""

_VENUE_ORDER: Final[tuple[GuardId, ...]] = tuple(g for g in GuardId if g in VENUE_GUARDS)
_PROVENANCE: Final = "src/sentiment_agent/analysis/baselines.py (this repository, MIT)"


def _spec(arm_id: str, title: str, description: str, guards: Sequence[GuardId]) -> ArmSpec:
    return ArmSpec(
        arm_id=arm_id,
        kind=ArmKind.BASELINE,
        title=title,
        description=description,
        provenance=_PROVENANCE,
        uses_llm=False,
        guards=tuple(guards),
    )


FLAT_SPEC: Final = _spec(FLAT_ID, "Flat", "Never trades: the starting equity, held.", _VENUE_ORDER)
BTC_AT_OUR_GROSS_SPEC: Final = _spec(
    BTC_AT_OUR_GROSS_ID,
    "BTC at our gross",
    "Long BTCUSDT at the governed book's mean gross weight, re-set daily, under the venue guards "
    "except the 5% per-name cap (which would make a single-name benchmark at our gross "
    "impossible).",
    tuple(g for g in _VENUE_ORDER if g is not GuardId.G3_SIZE),
)
CROWD_FADE_SPEC: Final = _spec(
    CROWD_FADE_ID,
    "Crowd fade, fixed rule",
    "Fades live funding z-scores beyond +/-2 and retail long shares outside Bitget's 0.45-0.65 "
    "balanced band, at 5% per name, with no language model.",
    _VENUE_ORDER,
)
COIN_FLIP_SPEC: Final = _spec(
    COIN_FLIP_PREFIX,
    "Coin flip (the null)",
    "Every 24h, five tradable names drawn at random, each long or short on a coin flip, at 5% per "
    "name: the envelope's no-edge book on our timestamps. One arm per seed; read as a "
    "distribution.",
    _VENUE_ORDER,
)


def baseline_specs() -> tuple[ArmSpec, ...]:
    """Flat, BTC at our gross, crowd fade, and the coin-flip template (each seed's arm carries
    :func:`coin_flip_spec`)."""
    return (FLAT_SPEC, BTC_AT_OUR_GROSS_SPEC, CROWD_FADE_SPEC, COIN_FLIP_SPEC)


def coin_flip_spec(seed_index: int) -> ArmSpec:
    if seed_index < 0:
        raise ValueError("seed index must be non-negative")
    return COIN_FLIP_SPEC.model_copy(
        update={
            "arm_id": f"{COIN_FLIP_PREFIX}_s{seed_index:04d}",
            "title": f"Coin flip, seed {COIN_FLIP_SEED_BASE + seed_index}",
        }
    )


# ------------------------------------------------------------------------------------------------
# The fixed rules
# ------------------------------------------------------------------------------------------------


def long_share(ratio: float) -> float:
    """Retail long share from a long/short account ratio (longs over shorts)."""
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError(f"a long/short ratio must be positive and finite, got {ratio}")
    return ratio / (1.0 + ratio)


def crowd_fade_targets(snapshot: PerceptionSnapshot, policy: Policy) -> dict[str, float]:
    """The crowd-fade weights for every universe symbol the snapshot has features for (module
    docstring). Zero where nothing is extreme, so a faded position is closed once the crowd
    unwinds."""
    threshold = policy.triggers.funding_z_threshold
    out: dict[str, float] = {}
    for symbol in policy.symbols:
        features = snapshot.features.get(symbol)
        if features is None:
            continue
        votes: list[int] = []
        z = features.funding_z_live
        if z is not None and math.isfinite(z) and abs(z) > threshold:
            votes.append(-1 if z > 0 else 1)
        ratio = features.retail_long_short_ratio
        if ratio is not None and math.isfinite(ratio) and ratio > 0:
            share = long_share(ratio)
            if share > LONG_SHARE_CROWDED:
                votes.append(-1)
            elif share < LONG_SHARE_SHORTS_CROWDED:
                votes.append(1)
        direction = 0
        if votes and (all(v > 0 for v in votes) or all(v < 0 for v in votes)):
            direction = votes[0]
        out[symbol] = direction * policy.per_name_max
    return out


def tradable_symbols(snapshot: PerceptionSnapshot, policy: Policy) -> list[str]:
    """Universe symbols with a Demo quote that the weekend rule lets the book open now."""
    phase = weekend_phase(snapshot.taken_at, policy.weekend)
    out: list[str] = []
    for symbol in policy.symbols:
        entry = policy.entry(symbol)
        if entry is None or symbol not in snapshot.demo_quotes:
            continue
        if phase != "open" and entry.asset_class.follows_us_session:
            continue
        out.append(symbol)
    return out


def coin_flip_draws(
    snapshots: Sequence[PerceptionSnapshot], policy: Policy, seed_index: int
) -> list[tuple[PerceptionSnapshot, dict[str, float]]]:
    """The coin flip's draws: at the first snapshot and then at the first snapshot at least the
    minimum horizon after the previous draw (module docstring). Each draw names every universe
    symbol with a Demo quote, so what was not drawn is closed."""
    rng = random.Random(COIN_FLIP_SEED_BASE + seed_index)  # noqa: S311 - a seeded null, not a secret
    horizon = timedelta(hours=policy.decision.min_horizon_hours)
    k = round(policy.gross_max / policy.per_name_max)
    draws: list[tuple[PerceptionSnapshot, dict[str, float]]] = []
    last: datetime | None = None
    for snapshot in sorted(snapshots, key=lambda s: s.taken_at):
        if last is not None and snapshot.taken_at < last + horizon:
            continue
        last = snapshot.taken_at
        targets = {s: 0.0 for s in policy.symbols if s in snapshot.demo_quotes}
        eligible = tradable_symbols(snapshot, policy)
        for symbol in rng.sample(eligible, min(k, len(eligible))):
            targets[symbol] = rng.choice((1, -1)) * policy.per_name_max
        draws.append((snapshot, targets))
    return draws


REPLICA_ARM_ID: Final = "twin_governed_replica"
"""``analysis/twin.py``'s ``GOVERNED_REPLICA_SPEC.arm_id``, named here to avoid an import cycle."""


def comparator_arm(book: ArmResult, arms: Sequence[ArmResult]) -> ArmResult:
    """The arm baselines and coin flips are ranked against: the governed replica when ``arms``
    holds one (simulated like them), else ``book`` (a record with no ruled decision has no
    replica, and no fill for a cost model to misprice)."""
    return next((a for a in arms if a.spec.arm_id == REPLICA_ARM_ID), book)


def btc_target_weight(ours: ArmResult, policy: Policy) -> float:
    """The comparator's mean gross weight over its marks, within ``[0, gross_max]``. Pass the
    replica (:func:`comparator_arm`), not the live book."""
    if not ours.marks:
        return 0.0
    gross = math.fsum(m.gross_weight for m in ours.marks) / len(ours.marks)
    return min(max(gross, 0.0), policy.gross_max)


# ------------------------------------------------------------------------------------------------
# Running them
# ------------------------------------------------------------------------------------------------


def _window(
    sim: ArmSimulator, snapshots: Sequence[PerceptionSnapshot], ours: ArmResult
) -> tuple[datetime, datetime]:
    """The governed book's mark grid, extended back to the hour of the first snapshot when the
    agent decided before its first hourly mark."""
    starts: list[datetime] = []
    if snapshots:
        starts.append(hour_floor(min(s.taken_at for s in snapshots)))
    if ours.marks:
        starts.append(ours.marks[0].at)
    if not starts:
        raise ValueError("no snapshots and no governed marks: nothing to put the baselines on")
    until = ours.marks[-1].at if ours.marks else sim.default_until()
    if until is None:
        raise ValueError("no Demo marks to run the baselines against")
    return min(starts), until


def run_baselines(
    sim: ArmSimulator,
    snapshots: Sequence[PerceptionSnapshot],
    ours: ArmResult,
    *,
    coin_flip_seeds: int = 1000,
) -> tuple[ArmResult, ...]:
    """Flat, BTC at our gross, crowd fade and ``coin_flip_seeds`` coin flips, on the agent's own
    snapshots and on the hourly grid of ``ours``, the comparator (:func:`comparator_arm`: the
    governed replica, simulated like these arms). The coin-flip arms carry no bootstrap
    interval each: their spread across seeds is the statement of uncertainty
    (:func:`coin_flip_summary`)."""
    if coin_flip_seeds < 0:
        raise ValueError("coin_flip_seeds cannot be negative")
    policy = sim.policy
    ordered = sorted(snapshots, key=lambda s: s.taken_at)
    start, until = _window(sim, ordered, ours)
    inputs: dict[str, KernelInputs] = {s.snapshot_id: sim.kernel_inputs(s) for s in ordered}

    flat = sim.run(
        FLAT_SPEC,
        [(s.taken_at, {}, inputs[s.snapshot_id]) for s in ordered],
        start=start,
        until=until,
    )

    weight = btc_target_weight(ours, policy)
    btc_schedule = []
    last_day = None
    for s in ordered:
        day = s.taken_at.date()
        if day != last_day:
            btc_schedule.append((s.taken_at, {BTC_SYMBOL: weight}, inputs[s.snapshot_id]))
            last_day = day
    btc = sim.run(BTC_AT_OUR_GROSS_SPEC, btc_schedule, start=start, until=until)

    fade = sim.run(
        CROWD_FADE_SPEC,
        [(s.taken_at, crowd_fade_targets(s, policy), inputs[s.snapshot_id]) for s in ordered],
        start=start,
        until=until,
    )

    flips = [
        sim.run(
            coin_flip_spec(i),
            [
                (s.taken_at, targets, inputs[s.snapshot_id])
                for s, targets in coin_flip_draws(ordered, policy, i)
            ],
            start=start,
            until=until,
            ci_resamples=0,
        )
        for i in range(coin_flip_seeds)
    ]
    return (flat, btc, fade, *flips)


# ------------------------------------------------------------------------------------------------
# The coin-flip distribution
# ------------------------------------------------------------------------------------------------


class DistributionBand(Model):
    """5th percentile, median and 95th percentile across seeds (the envelope's conventions:
    ``percentile`` is ``envelope_clean.py:70``'s rule, the median is ``statistics.median``)."""

    n: int
    p05: float | None
    median: float | None
    p95: float | None


class CoinFlipDistribution(Model):
    """Where the agent sits among the no-edge coin flips on the same clock, read through the arm
    named by ``ranked_arm_id`` (the governed replica, simulated like the seeds)."""

    ranked_arm_id: str
    seeds: int
    total_return: DistributionBand
    sharpe_ann: DistributionBand
    max_drawdown: DistributionBand
    win_rate: DistributionBand
    ours_total_return: float
    ours_sharpe_ann: float | None
    share_below_ours_total_return: float | None
    """Share of coin-flip seeds whose total return is strictly below ours."""
    share_below_ours_sharpe_ann: float | None
    """Share of seeds with a defined Sharpe ratio strictly below ours (``None`` when ours or all
    of theirs is undefined)."""
    label: Literal["descriptive, not inferential"] = "descriptive, not inferential"


def _band(values: Sequence[float]) -> DistributionBand:
    if not values:
        return DistributionBand(n=0, p05=None, median=None, p95=None)
    ordered = sorted(values)
    return DistributionBand(
        n=len(ordered),
        p05=percentile(ordered, 0.05),
        median=float(statistics.median(ordered)),
        p95=percentile(ordered, 0.95),
    )


def coin_flip_arms(arms: Sequence[ArmResult]) -> list[ArmResult]:
    """The coin-flip arms among ``arms`` (as :func:`run_baselines` returns them)."""
    prefix = f"{COIN_FLIP_PREFIX}_s"
    return [a for a in arms if a.spec.arm_id.startswith(prefix)]


def coin_flip_summary(ours: ArmResult, arms: Sequence[ArmResult]) -> CoinFlipDistribution:
    """The coin-flip distribution and ``ours``' place in it. Pass the comparator
    (:func:`comparator_arm`), so ours is costed exactly as the seeds are. ``arms`` may be every
    baseline arm; only the coin flips are read."""
    flips = coin_flip_arms(arms)
    returns = [a.metrics.total_return for a in flips]
    sharpes = [a.metrics.sharpe_ann for a in flips if a.metrics.sharpe_ann is not None]
    ours_return = ours.metrics.total_return
    ours_sharpe = ours.metrics.sharpe_ann
    return CoinFlipDistribution(
        ranked_arm_id=ours.spec.arm_id,
        seeds=len(flips),
        total_return=_band(returns),
        sharpe_ann=_band(sharpes),
        max_drawdown=_band([a.metrics.max_drawdown for a in flips]),
        win_rate=_band([a.metrics.win_rate for a in flips if a.metrics.win_rate is not None]),
        ours_total_return=ours_return,
        ours_sharpe_ann=ours_sharpe,
        share_below_ours_total_return=(
            sum(1 for r in returns if r < ours_return) / len(returns) if returns else None
        ),
        share_below_ours_sharpe_ann=(
            sum(1 for s in sharpes if s < ours_sharpe) / len(sharpes)
            if ours_sharpe is not None and sharpes
            else None
        ),
    )


__all__ = [
    "BTC_AT_OUR_GROSS_ID",
    "BTC_AT_OUR_GROSS_SPEC",
    "BTC_SYMBOL",
    "COIN_FLIP_PREFIX",
    "COIN_FLIP_SEED_BASE",
    "COIN_FLIP_SPEC",
    "CROWD_FADE_ID",
    "CROWD_FADE_SPEC",
    "FLAT_ID",
    "FLAT_SPEC",
    "LONG_SHARE_CROWDED",
    "LONG_SHARE_SHORTS_CROWDED",
    "REPLICA_ARM_ID",
    "CoinFlipDistribution",
    "DistributionBand",
    "baseline_specs",
    "btc_target_weight",
    "coin_flip_arms",
    "coin_flip_draws",
    "coin_flip_spec",
    "coin_flip_summary",
    "comparator_arm",
    "crowd_fade_targets",
    "long_share",
    "run_baselines",
    "tradable_symbols",
]
