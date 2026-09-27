r"""Replay candidate trading designs over every nine-day window of Bitget history, through the
production kernel, planner and book, before run 2's genesis (T2 of the pre-genesis audit).

    python scripts/strategy_replay.py --history <history.json> --marks <history_marks.json> \
        [--window-hours 216] [--seeds 50] [--out validation/run2/strategy_replay.json]

Why this exists. The pre-genesis audit (`research/harvest/quant/T2_STRATEGY_AUDIT.md` in the
ARGUS workspace) measured run 2's funding-fade trigger in a proxy simulator and found it loses. A
proxy can be wrong where the real kernel is not, so every figure here comes from
`analysis/armsim.ArmSimulator`: the production `RiskKernel` rules each target, `plan_orders` sizes
it on the Demo grid, `BookBuilder` books the fills, and the venue stops (G4), weekend freeze (G2),
daily kill (G5) and breaker (G10) act between decisions exactly as in the live run. Only the
decision-maker is replaced: each arm is a fixed rule standing in for the model, so the question is
what the rules and the kernel do to Sharpe, drawdown and win rate, not what Qwen would decide.

Data (keyless public endpoints, fetched by `research/harvest/quant/replay/fetch_*.py`):
* Demo mark 1H candles, where the Demo venue serves them (BTC from 2026-01-30; equities from
  2026-08-25; SP500/NDX100 from 2026-09-03), else the live mark as a proxy. Every window records
  which symbols it marked on live prices (`proxy_symbols`).
* Live funding settlements (270 per symbol, ~90 days), read at settlement, never ahead of it.
* Demo instrument specs as served; half the Demo spread from `validation/demo_venue/
  universe_probe.json`'s bid/ask per symbol (one number per symbol, as armsim documents).

Windows start at 00:00 UTC every day with a full window of marks behind them; each carries the
policy's own scoring window set to itself, so G2 closes every leg at its end as it will in run 2.
Overlapping windows are not independent: the report says how many non-overlapping ones there are.
Nothing here calls a model or a credential.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentiment_agent.analysis.armsim import ArmSimulator  # noqa: E402
from sentiment_agent.clock import ManualClock  # noqa: E402
from sentiment_agent.kernel.kernel import RiskKernel  # noqa: E402
from sentiment_agent.policy import POLICY_V2  # noqa: E402
from sentiment_agent.types import (  # noqa: E402
    ArmKind,
    ArmSpec,
    Candle,
    GuardId,
    InstrumentSpec,
    KernelInputs,
    Policy,
    PriceSource,
    Quote,
    ScoringWindow,
)

HOUR = timedelta(hours=1)
HEARTBEAT_HOURS = (0, 8, 16)
US_OPEN_HOUR = 14
"""09:30 New York is 13:30 UTC in EDT; the first hourly mark after it is 14:00."""
GUARDS = tuple(g for g in GuardId if g is not GuardId.G9_GROUNDING)
"""Every guard but G9: grounding checks a model's cited numbers, and a rule cites none."""
STARTING_EQUITY = 10_000.0
FUNDING_LOOKBACK = 90
EQUITIES = frozenset(e.symbol for e in POLICY_V2.universe if e.asset_class.value == "us_equity")
Z_THRESHOLD = 2.0


# ------------------------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class History:
    marks: dict[str, list[Candle]]
    live: dict[str, dict[datetime, float]]
    funding: dict[str, list[tuple[datetime, float]]]
    specs: dict[str, InstrumentSpec]
    spreads_bps: dict[str, float]
    demo_from: dict[str, datetime]


def _ts(value: int) -> datetime:
    return datetime.fromtimestamp(value, UTC)


def _candle(symbol: str, row: Sequence[float]) -> Candle:
    return Candle(
        symbol=symbol,
        source=PriceSource.DEMO,
        kind="mark",
        interval="1H",
        open_time=_ts(int(row[0])),
        open=Decimal(str(row[1])),
        high=Decimal(str(row[2])),
        low=Decimal(str(row[3])),
        close=Decimal(str(row[4])),
        volume=None,
    )


def load(history_path: Path, marks_path: Path, probe_path: Path) -> History:
    history = json.loads(history_path.read_text(encoding="utf-8"))
    marks = json.loads(marks_path.read_text(encoding="utf-8"))
    probe = json.loads(probe_path.read_text(encoding="utf-8"))["instruments"]
    merged: dict[str, list[Candle]] = {}
    live: dict[str, dict[datetime, float]] = {}
    demo_from: dict[str, datetime] = {}
    for symbol, rows in marks["live_mark"].items():
        demo_rows = {int(r[0]): r for r in marks["demo_mark"].get(symbol, [])}
        if demo_rows:
            demo_from[symbol] = _ts(min(demo_rows))
        by_time = {int(r[0]): r for r in rows}
        by_time.update(demo_rows)
        merged[symbol] = [_candle(symbol, by_time[t]) for t in sorted(by_time)]
        live[symbol] = {_ts(int(r[0])): float(r[4]) for r in rows}
    funding = {
        s: sorted((_ts(int(t)), float(r)) for t, r in rows)
        for s, rows in history["funding"].items()
    }
    specs = {s: InstrumentSpec.model_validate(v) for s, v in marks["demo_specs"].items()}
    spreads: dict[str, float] = {}
    for symbol in merged:
        demo = probe[symbol]["demo"]
        bid, ask = float(demo["bid"]), float(demo["ask"])
        spreads[symbol] = (ask - bid) / ((ask + bid) / 2) * 10_000
    return History(merged, live, funding, specs, spreads, demo_from)


def _quote(
    symbol: str, source: PriceSource, at: datetime, price: float, half_spread: float
) -> Quote:
    p = Decimal(str(price))
    h = Decimal(str(half_spread))
    return Quote(
        symbol=symbol,
        source=source,
        ts=at,
        fetched_at=at,
        last=p,
        mark=p,
        index=p,
        bid=p * (1 - h),
        ask=p * (1 + h),
        funding_rate=None,
        open_interest=None,
        turnover_24h=None,
        price_change_24h=None,
    )


def kernel_inputs(sim: ArmSimulator, hist: History, at: datetime) -> KernelInputs:
    demo: dict[str, Quote] = {}
    live: dict[str, Quote] = {}
    move: dict[str, float | None] = {}
    for symbol in hist.marks:
        price = sim.price_at(symbol, at)
        if price is None:
            continue
        half = hist.spreads_bps[symbol] / 20_000
        demo[symbol] = _quote(symbol, PriceSource.DEMO, at, float(price), half)
        live_price = hist.live[symbol].get(at - HOUR)
        live[symbol] = _quote(
            symbol,
            PriceSource.LIVE,
            at,
            live_price if live_price is not None else float(price),
            half,
        )
        # G1's stale-venue input, the statistic `perception/features.index_move_bps_3h` computes:
        # the mean |hourly return| in bps over the three newest completed bars. The Demo index is
        # not fetched here; its mark, which tracks it, stands in (`proxy` noted in the docstring).
        closes = [sim.price_at(symbol, at - k * HOUR) for k in range(4)]
        if any(c is None or c == 0 for c in closes):
            move[symbol] = None
        else:
            steps = [
                abs(float(closes[k] / closes[k + 1] - 1)) * 10_000  # type: ignore[operator]
                for k in range(3)
            ]
            move[symbol] = sum(steps) / 3
    return KernelInputs(
        at=at,
        demo_quotes=demo,
        live_quotes=live,
        specs=dict(hist.specs),
        demo_index_move_bps_3h=move,
        snapshot_id=None,
        snapshot_taken_at=at,
    )


# ------------------------------------------------------------------------------------------------
# The stand-in decision rules. Each returns {instant: targets}; targets are weights (fraction of
# equity), and a symbol absent from a later entry keeps nothing (targets are the whole book).
# ------------------------------------------------------------------------------------------------

Rule = Callable[
    [History, Policy, datetime, datetime, random.Random], list[tuple[datetime, dict[str, float]]]
]


def _decision_hours(start: datetime, end: datetime) -> list[datetime]:
    out, t = [], start
    while t < end:
        if t.hour in HEARTBEAT_HOURS or t.hour == US_OPEN_HOUR:
            out.append(t)
        t += HOUR
    return out


def _extremes(
    hist: History,
    policy: Policy,
    start: datetime,
    end: datetime,
    floor: float,
    *,
    symbols: frozenset[str] | None = None,
    floor_quantile: float | None = None,
) -> list[tuple[datetime, str, float]]:
    """(the first hour after a settlement, symbol, rate) for settlements whose live rate is at or
    beyond ``floor`` and whose z over the prior 90 settlements is beyond +/-2 — run 2's
    `funding_zscore` condition (policy v2: every class, level floor run2-a2), read at settlement.

    ``symbols`` narrows the scope; ``floor_quantile`` replaces the fixed floor with that quantile
    of |rate| over the same prior 90 settlements (the audit's per-instrument floor, 4.3)."""
    out = []
    scope = {
        e.symbol
        for e in policy.universe
        if e.asset_class in policy.triggers.funding_z_asset_classes
    }
    if symbols is not None:
        scope &= symbols
    for symbol, series in hist.funding.items():
        if symbol not in scope:
            continue
        for i in range(FUNDING_LOOKBACK, len(series)):
            at, rate = series[i]
            if not start <= at < end:
                continue
            prior = [r for _, r in series[i - FUNDING_LOOKBACK : i]]
            level = floor
            if floor_quantile is not None:
                ranked = sorted(abs(r) for r in prior)
                level = ranked[min(len(ranked) - 1, int(floor_quantile * len(ranked)))]
            if abs(rate) < level or rate == 0:
                continue
            sd = statistics.pstdev(prior)
            if sd == 0 or abs((rate - statistics.fmean(prior)) / sd) <= Z_THRESHOLD:
                continue
            out.append((at.replace(minute=0, second=0, microsecond=0) + HOUR, symbol, rate))
    return sorted(out)


def fade_rule(
    floor: float,
    hold_hours: int = 24,
    *,
    symbols: frozenset[str] | None = None,
    floor_quantile: float | None = None,
) -> Rule:
    """Half a target against the crowd on each funding extreme, held ``hold_hours``, then closed
    at the next decision hour: the model's registered behaviour on its only event signal."""

    def rule(
        hist: History, policy: Policy, start: datetime, end: datetime, _rng: random.Random
    ) -> list[tuple[datetime, dict[str, float]]]:
        half = policy.per_name_max / 2
        events = _extremes(
            hist, policy, start, end, floor, symbols=symbols, floor_quantile=floor_quantile
        )
        opens: dict[str, tuple[datetime, float]] = {}
        instants = sorted({t for t, _, _ in events} | set(_decision_hours(start, end)))
        schedule = []
        for t in instants:
            changed = False
            for symbol, (since, _) in list(opens.items()):
                if t - since >= timedelta(hours=hold_hours):
                    del opens[symbol]
                    changed = True
            for at, symbol, rate in events:
                if at == t and symbol not in opens:
                    opens[symbol] = (t, -half if rate > 0 else half)
                    changed = True
            if changed:
                schedule.append((t, {s: w for s, (_, w) in opens.items()}))
        return schedule

    return rule


def coin_flip_rule(
    hist: History, policy: Policy, start: datetime, end: datetime, rng: random.Random
) -> list[tuple[datetime, dict[str, float]]]:
    """Every day at 00:00, five names drawn at random, each long or short at half a target: the
    no-edge book (analysis/baselines.py's coin flip, at run 2's half-target size)."""
    half = policy.per_name_max / 2
    names = sorted(hist.marks)
    schedule, t = [], start
    while t < end:
        picks = rng.sample(names, 5)
        schedule.append((t, {s: half if rng.random() < 0.5 else -half for s in picks}))
        t += timedelta(days=1)
    return schedule


def flat_rule(
    hist: History, policy: Policy, start: datetime, end: datetime, _rng: random.Random
) -> list[tuple[datetime, dict[str, float]]]:
    return [(start, {})]


def btc_hold_rule(
    hist: History, policy: Policy, start: datetime, end: datetime, _rng: random.Random
) -> list[tuple[datetime, dict[str, float]]]:
    return [(start, {"BTCUSDT": policy.per_name_max})]


def cross_section_rule(
    hist: History, policy: Policy, start: datetime, end: datetime, _rng: random.Random
) -> list[tuple[datetime, dict[str, float]]]:
    """Daily at 00:00: short the two equities with the highest summed live funding over the last
    9 settlements, long the two lowest, half a target each (market-neutral by count)."""
    half = policy.per_name_max / 2
    equities = [e.symbol for e in policy.universe if e.asset_class.value == "us_equity"]
    schedule, t = [], start
    while t < end:
        scores = {}
        for s in equities:
            past = [r for at, r in hist.funding.get(s, []) if at < t][-9:]
            if len(past) == 9:
                scores[s] = sum(past)
        if len(scores) >= 4:
            ranked = sorted(scores, key=scores.get)  # type: ignore[arg-type]
            schedule.append(
                (t, {**dict.fromkeys(ranked[:2], half), **dict.fromkeys(ranked[-2:], -half)})
            )
        t += timedelta(days=1)
    return schedule


def equal_long_rule(
    hist: History, policy: Policy, start: datetime, end: datetime, _rng: random.Random
) -> list[tuple[datetime, dict[str, float]]]:
    """Long every US equity at 1% of equity, re-set daily at 00:00 (the audit's market arm)."""
    equities = [e.symbol for e in policy.universe if e.asset_class.value == "us_equity"]
    schedule, t = [], start
    while t < end:
        schedule.append((t, dict.fromkeys(equities, 0.01)))
        t += timedelta(days=1)
    return schedule


ARMS: dict[str, Rule] = {
    "fade_7.5bp_registered": fade_rule(0.00075),
    "fade_5bp": fade_rule(0.0005),
    "fade_equities_7.5bp": fade_rule(0.00075, symbols=EQUITIES),
    "fade_btc_p97": fade_rule(0.0, symbols=frozenset({"BTCUSDT"}), floor_quantile=0.97),
    "cross_section_funding": cross_section_rule,
    "btc_hold_5pct": btc_hold_rule,
    "equal_weight_long_1pct": equal_long_rule,
    "flat": flat_rule,
}


# ------------------------------------------------------------------------------------------------
# Running
# ------------------------------------------------------------------------------------------------


def _spec(arm_id: str) -> ArmSpec:
    return ArmSpec(
        arm_id=arm_id,
        kind=ArmKind.BASELINE,
        title=arm_id,
        description="stand-in rule for the pre-genesis replay",
        provenance="scripts/strategy_replay.py",
        uses_llm=False,
        guards=GUARDS,
    )


def windows(hist: History, hours: int, funding_from: datetime) -> list[tuple[datetime, datetime]]:
    starts_ok = max(max(c[0].open_time for c in hist.marks.values()), funding_from)
    first = (starts_ok + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    last_end = min(c[-1].open_time for c in hist.marks.values())
    out, t = [], first
    while t + timedelta(hours=hours) <= last_end:
        out.append((t, t + timedelta(hours=hours)))
        t += timedelta(days=1)
    return out


def run_window(
    hist: History,
    start: datetime,
    end: datetime,
    seeds: int,
    arms: Mapping[str, Rule],
    *,
    take_profit: float | None = None,
) -> dict[str, Any]:
    policy = Policy.model_validate(
        POLICY_V2.model_copy(
            update={
                "scoring_window": ScoringWindow(start=start, end=end, basis="replay window"),
                "take_profit_pct": take_profit,
            }
        ).model_dump()
    )
    sim = ArmSimulator(
        kernel=RiskKernel(policy, ManualClock(start)),
        policy=policy,
        demo_marks=hist.marks,
        spreads_bps=hist.spreads_bps,
        starting_equity=STARTING_EQUITY,
        specs=hist.specs,
    )
    cache: dict[datetime, KernelInputs] = {}

    def inputs(at: datetime) -> KernelInputs:
        if at not in cache:
            cache[at] = kernel_inputs(sim, hist, at)
        return cache[at]

    def one(arm_id: str, rule: Rule, rng: random.Random) -> dict[str, Any]:
        schedule = [(t, w, inputs(t)) for t, w in rule(hist, policy, start, end, rng) if t < end]
        # One hour past the end, so G2's close at the scoring window's end is simulated and every
        # leg becomes a closed trade, as in run 2 (run2-a4).
        result = sim.run(_spec(arm_id), schedule, start=start, until=end + HOUR, ci_resamples=0)
        m = result.metrics
        return {
            "return": m.total_return,
            "sharpe": m.sharpe_ann,
            "max_dd": m.max_drawdown,
            "win_rate": m.win_rate,
            "closed_trades": m.n_closed_trades,
            "fees": m.fees_paid,
            "turnover": m.turnover,
        }

    out: dict[str, Any] = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "proxy_symbols": sorted(s for s, t in hist.demo_from.items() if t > start)
        + sorted(s for s in hist.marks if s not in hist.demo_from),
        "arms": {},
    }
    for arm_id, rule in arms.items():
        out["arms"][arm_id] = one(arm_id, rule, random.Random(0))  # noqa: S311 - a seeded replay, not a secret
    flips = [
        one(f"coin_flip_{i}", coin_flip_rule, random.Random(1000 + i))  # noqa: S311
        for i in range(seeds)
    ]
    out["arms"]["coin_flip"] = {"seeds": flips}
    return out


def _band(values: Sequence[float | None]) -> dict[str, Any]:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return {"n": 0}

    def q(p: float) -> float:
        return vals[min(len(vals) - 1, max(0, round(p * (len(vals) - 1))))]

    return {
        "n": len(vals),
        "p10": round(q(0.1), 5),
        "median": round(q(0.5), 5),
        "p90": round(q(0.9), 5),
    }


def summarise(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    arms = [a for a in rows[0]["arms"] if a != "coin_flip"] if rows else []
    summary: dict[str, Any] = {}
    for arm in [*arms, "coin_flip"]:
        per: dict[str, list[float | None]] = {
            k: [] for k in ("return", "sharpe", "max_dd", "win_rate", "closed_trades")
        }
        for row in rows:
            results = row["arms"][arm]["seeds"] if arm == "coin_flip" else [row["arms"][arm]]
            for r in results:
                for k in per:
                    per[k].append(r[k])
        summary[arm] = {k: _band(v) for k, v in per.items()}
        summary[arm]["p_return_positive"] = round(
            sum(1 for v in per["return"] if v is not None and v > 0) / max(1, len(per["return"])), 3
        )
    return summary


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--marks", type=Path, required=True)
    parser.add_argument("--window-hours", type=int, default=216)
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0, help="first N windows only (smoke run)")
    parser.add_argument(
        "--take-profit", type=float, default=None, help="Policy.take_profit_pct for every arm (T4)"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "validation/run2/strategy_replay.json")
    args = parser.parse_args(argv)
    hist = load(args.history, args.marks, ROOT / "validation/demo_venue/universe_probe.json")
    funding_from = max(
        series[FUNDING_LOOKBACK][0]
        for series in hist.funding.values()
        if len(series) > FUNDING_LOOKBACK
    )
    spans = windows(hist, args.window_hours, funding_from)
    if args.limit:
        spans = spans[: args.limit]
    rows = []
    for i, (start, end) in enumerate(spans):
        rows.append(run_window(hist, start, end, args.seeds, ARMS, take_profit=args.take_profit))
        print(f"window {i + 1}/{len(spans)} {start:%Y-%m-%d}", flush=True)
    mid = len(rows) // 2
    report = {
        "method": __doc__,
        "window_hours": args.window_hours,
        "take_profit_pct": args.take_profit,
        "windows": len(rows),
        "non_overlapping_windows": len(rows) * 24 // args.window_hours,
        "starting_equity": STARTING_EQUITY,
        "guards": [g.value for g in GUARDS],
        "summary": {
            "all": summarise(rows),
            "first_half": summarise(rows[:mid]),
            "second_half": summarise(rows[mid:]),
            "demo_marked_only": summarise([r for r in rows if not r["proxy_symbols"]]),
        },
        "rows": rows,
    }
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"]["all"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
