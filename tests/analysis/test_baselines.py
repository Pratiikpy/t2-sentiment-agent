"""Same-clock baselines: the crowd-fade rule, the coin-flip null, and the runs themselves."""

import math
from datetime import datetime, timedelta

import pytest

from analysis.abuild import (
    AAPL,
    AMZN,
    BTC,
    FRI,
    HOUR,
    META,
    NVDA,
    TSLA,
    WED,
    candles,
    features,
    simulator,
    snapshot,
)
from sentiment_agent.analysis.baselines import (
    BTC_AT_OUR_GROSS_ID,
    BTC_AT_OUR_GROSS_SPEC,
    COIN_FLIP_PREFIX,
    COIN_FLIP_SPEC,
    CROWD_FADE_ID,
    CROWD_FADE_SPEC,
    FLAT_ID,
    FLAT_SPEC,
    baseline_specs,
    btc_target_weight,
    coin_flip_arms,
    coin_flip_draws,
    coin_flip_spec,
    coin_flip_summary,
    crowd_fade_targets,
    long_share,
    run_baselines,
    tradable_symbols,
)
from sentiment_agent.analysis.metrics import metric_set
from sentiment_agent.policy import POLICY_V1
from sentiment_agent.types import (
    VENUE_GUARDS,
    ArmKind,
    ArmMark,
    ArmResult,
    ArmSpec,
    GuardId,
    PerceptionSnapshot,
)

PRICES = {BTC: "80000", NVDA: "200", AAPL: "300"}


def ours(times: list[datetime], gross: float, equity: list[float] | None = None) -> ArmResult:
    values = equity or [100_000.0] * len(times)
    marks = [
        ArmMark(at=t, equity=e, gross_weight=gross, net_weight=gross)
        for t, e in zip(times, values, strict=True)
    ]
    spec = ArmSpec(
        arm_id="ours_governed",
        kind=ArmKind.OURS_GOVERNED,
        title="ours",
        description="d",
        provenance="tests",
        uses_llm=True,
        guards=(),
    )
    return ArmResult(
        spec=spec,
        marks=tuple(marks),
        trades=(),
        metrics=metric_set(
            "ours_governed", marks, [], traded_notional=0.0, fees=0.0, ci_resamples=0
        ),
    )


# ------------------------------------------------------------------------------------------------
# The crowd-fade rule
# ------------------------------------------------------------------------------------------------


def test_crowd_fade_rule_table() -> None:
    snap = snapshot(
        dict.fromkeys((BTC, NVDA, TSLA, META, AAPL, AMZN), "100"),
        at=WED,
        feats={
            BTC: features(BTC, funding_z=2.5),  # longs paying far above normal: fade them
            NVDA: features(NVDA, funding_z=-2.1),  # shorts paying: fade them
            TSLA: features(TSLA, funding_z=2.0),  # exactly at the threshold: not beyond it
            META: features(META, long_short=2.0),  # long share 0.667 > 0.65: longs crowded
            AAPL: features(AAPL, long_short=0.8),  # long share 0.444 < 0.45: shorts dominant
            AMZN: features(AMZN, funding_z=2.5, long_short=0.5),  # the two disagree
        },
    )
    targets = crowd_fade_targets(snap, POLICY_V1)
    assert targets == {
        BTC: -0.05,
        NVDA: 0.05,
        TSLA: 0.0,
        META: -0.05,
        AAPL: 0.05,
        AMZN: 0.0,
    }


def test_crowd_fade_agreeing_votes_and_balanced_band() -> None:
    snap = snapshot(
        {BTC: "100", NVDA: "100"},
        at=WED,
        feats={
            BTC: features(BTC, funding_z=3.0, long_short=3.0),  # both say longs are crowded
            NVDA: features(NVDA, long_short=1.5),  # long share 0.6: Bitget's balanced band
        },
    )
    assert crowd_fade_targets(snap, POLICY_V1) == {BTC: -0.05, NVDA: 0.0}


def test_crowd_fade_skips_symbols_without_features() -> None:
    snap = snapshot({BTC: "100"}, at=WED, feats={})
    assert crowd_fade_targets(snap, POLICY_V1) == {}


def test_long_share() -> None:
    assert long_share(1.0) == 0.5
    assert long_share(3.0) == 0.75
    for bad in (0.0, -1.0, math.inf):
        with pytest.raises(ValueError, match="positive and finite"):
            long_share(bad)


# ------------------------------------------------------------------------------------------------
# Specs
# ------------------------------------------------------------------------------------------------


def test_baseline_specs() -> None:
    specs = baseline_specs()
    assert [s.arm_id for s in specs] == [
        FLAT_ID,
        BTC_AT_OUR_GROSS_ID,
        CROWD_FADE_ID,
        COIN_FLIP_PREFIX,
    ]
    assert all(s.kind is ArmKind.BASELINE and not s.uses_llm for s in specs)
    venue = set(VENUE_GUARDS)
    assert set(FLAT_SPEC.guards) == venue
    assert set(CROWD_FADE_SPEC.guards) == venue
    assert set(COIN_FLIP_SPEC.guards) == venue
    assert set(BTC_AT_OUR_GROSS_SPEC.guards) == venue - {GuardId.G3_SIZE}
    assert coin_flip_spec(3).arm_id == "baseline_coin_flip_s0003"
    assert coin_flip_spec(3).guards == COIN_FLIP_SPEC.guards
    with pytest.raises(ValueError, match="non-negative"):
        coin_flip_spec(-1)


# ------------------------------------------------------------------------------------------------
# The coin flip
# ------------------------------------------------------------------------------------------------


def every_four_hours(
    start: datetime, count: int, prices: dict[str, str]
) -> list[PerceptionSnapshot]:
    return [snapshot(prices, at=start + 4 * HOUR * i) for i in range(count)]


def test_coin_flip_draws_every_24_hours_and_closes_what_it_did_not_draw() -> None:
    universe = dict.fromkeys(POLICY_V1.symbols, "100")
    snaps = every_four_hours(WED, 19, universe)  # Wednesday 13:00 to Saturday 13:00
    draws = coin_flip_draws(snaps, POLICY_V1, 0)
    assert [s.taken_at for s, _ in draws] == [WED + timedelta(days=d) for d in range(4)]
    for snap, targets in draws:
        assert set(targets) == set(POLICY_V1.symbols)
        live = {s: w for s, w in targets.items() if w != 0}
        assert all(abs(w) == POLICY_V1.per_name_max for w in live.values())
        if snap.taken_at.weekday() == 5:  # Saturday: only crypto can be opened
            assert set(live) == {BTC}
        else:
            assert len(live) == 5


def test_coin_flip_is_reproducible_by_seed() -> None:
    universe = dict.fromkeys(POLICY_V1.symbols, "100")
    snaps = every_four_hours(WED, 13, universe)
    assert coin_flip_draws(snaps, POLICY_V1, 7) == coin_flip_draws(snaps, POLICY_V1, 7)
    seeds = [tuple(sorted(t.items())) for _, t in coin_flip_draws(snaps, POLICY_V1, 8)]
    others = [tuple(sorted(t.items())) for _, t in coin_flip_draws(snaps, POLICY_V1, 9)]
    assert seeds != others


def test_coin_flip_sides_are_balanced_over_many_seeds() -> None:
    universe = dict.fromkeys(POLICY_V1.symbols, "100")
    snaps = [snapshot(universe, at=WED)]
    longs = shorts = 0
    for seed in range(400):
        ((_, targets),) = coin_flip_draws(snaps, POLICY_V1, seed)
        longs += sum(1 for w in targets.values() if w > 0)
        shorts += sum(1 for w in targets.values() if w < 0)
    assert longs + shorts == 400 * 5
    assert 0.45 < longs / (longs + shorts) < 0.55


def test_tradable_symbols_follow_the_weekend_rule() -> None:
    quoted = {BTC: "1", NVDA: "1", AAPL: "1"}
    assert tradable_symbols(snapshot(quoted, at=WED), POLICY_V1) == [BTC, NVDA, AAPL]
    buffer = snapshot(quoted, at=FRI + timedelta(hours=5, minutes=30))  # Friday 18:30
    assert tradable_symbols(buffer, POLICY_V1) == [BTC]


def test_btc_weight_is_our_mean_gross_within_the_cap() -> None:
    times = [WED + HOUR * i for i in range(4)]
    assert btc_target_weight(ours(times, 0.12), POLICY_V1) == pytest.approx(0.12)
    assert btc_target_weight(ours(times, 0.4), POLICY_V1) == POLICY_V1.gross_max
    assert btc_target_weight(ours([], 0.1), POLICY_V1) == 0.0


# ------------------------------------------------------------------------------------------------
# Running them
# ------------------------------------------------------------------------------------------------


def _run(seeds: int) -> tuple[tuple[ArmResult, ...], ArmResult]:
    hours = 30
    marks = {
        s: candles(s, WED, [str(float(p) * (1 + 0.001 * ((i % 5) - 2))) for i in range(hours)])
        for s, p in PRICES.items()
    }
    sim = simulator(marks)
    snaps = [
        snapshot(
            PRICES,
            at=WED + 4 * HOUR * i,
            feats={s: features(s, funding_z=3.0 if s == BTC else None) for s in PRICES},
        )
        for i in range(7)
    ]
    reference = ours([WED + HOUR * i for i in range(hours)], 0.1)
    return run_baselines(sim, snaps, reference, coin_flip_seeds=seeds), reference


def test_run_baselines_returns_every_arm_on_our_grid() -> None:
    arms, reference = _run(seeds=5)
    ids = [a.spec.arm_id for a in arms]
    assert ids[:3] == [FLAT_ID, BTC_AT_OUR_GROSS_ID, CROWD_FADE_ID]
    assert ids[3:] == [f"{COIN_FLIP_PREFIX}_s{i:04d}" for i in range(5)]
    grid = [m.at for m in reference.marks]
    for result in arms:
        assert [m.at for m in result.marks] == grid

    flat, btc, fade = arms[:3]
    assert {m.equity for m in flat.marks} == {100_000.0}
    assert flat.metrics.sharpe_ann is None

    assert btc.marks[1].gross_weight == pytest.approx(0.10, rel=5e-3)  # 10%: G3's 5% cap is off
    assert fade.marks[1].net_weight == pytest.approx(-0.05, rel=5e-3)  # BTC funding z of +3: short
    assert set(fade.metrics.ci90) >= {"total_return", "max_drawdown"}

    for flip in arms[3:]:
        assert flip.metrics.ci90 == {}
        assert flip.marks[1].gross_weight == pytest.approx(0.15, rel=5e-3)  # 3 names quoted


def test_run_baselines_is_reproducible() -> None:
    first, _ = _run(seeds=3)
    second, _ = _run(seeds=3)
    assert first == second


def test_run_baselines_needs_the_demo_specs() -> None:
    sim = simulator({BTC: candles(BTC, WED, ["1", "1"])}, with_specs=False)
    with pytest.raises(ValueError, match="no Demo instrument specs"):
        run_baselines(sim, [snapshot({BTC: "1"}, at=WED)], ours([WED, WED + HOUR], 0.0))
    with pytest.raises(ValueError, match="cannot be negative"):
        run_baselines(sim, [], ours([WED], 0.0), coin_flip_seeds=-1)


# ------------------------------------------------------------------------------------------------
# The distribution
# ------------------------------------------------------------------------------------------------


def _flip(index: int, total: float) -> ArmResult:
    times = [WED, WED + HOUR, WED + 2 * HOUR]
    equity = [1.0, 1.0 + total / 2, 1.0 + total]
    base = ours(times, 0.0, equity)
    return base.model_copy(update={"spec": coin_flip_spec(index)})


def test_coin_flip_summary() -> None:
    flips = [_flip(i, (i - 50) / 1000) for i in range(101)]  # total returns -5% .. +5%
    mine = ours([WED, WED + HOUR, WED + 2 * HOUR], 0.0, [1.0, 1.005, 1.012])
    summary = coin_flip_summary(mine, [*flips, mine])
    assert summary.seeds == 101
    assert summary.total_return.n == 101
    assert summary.total_return.median == pytest.approx(0.0, abs=1e-12)
    assert summary.total_return.p05 == pytest.approx(-0.045, abs=1e-9)
    assert summary.total_return.p95 == pytest.approx(0.045, abs=1e-9)
    assert summary.ours_total_return == pytest.approx(0.012)
    assert summary.share_below_ours_total_return == pytest.approx(62 / 101)
    assert summary.label == "descriptive, not inferential"
    assert len(coin_flip_arms([*flips, mine])) == 101


def test_coin_flip_summary_without_seeds() -> None:
    mine = ours([WED, WED + HOUR], 0.0)
    summary = coin_flip_summary(mine, [])
    assert summary.seeds == 0
    assert summary.total_return.median is None
    assert summary.share_below_ours_total_return is None
    assert summary.share_below_ours_sharpe_ann is None
