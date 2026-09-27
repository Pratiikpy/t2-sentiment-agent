r"""Measure how run 1's funding extremes reverted, before run 2 decides on a reversion exit.

    python scripts/funding_reversion.py <public dir> [--until-seq N] [--out FILE]

    python scripts/funding_reversion.py ../t2-sentiment-agent/public \
        --out validation/run2/funding_reversion.json

Reads the run's published ``ledger.jsonl`` and ``blobs/`` (read only, chain verified first) and
takes every logged snapshot in order. An *episode* opens on the first snapshot where an
instrument's live funding z-score is beyond policy v2's threshold with the live rate at or above
v2's level floor (the condition ``funding_zscore`` fires on under ``run2-a2``) and closes when the
z-score first comes back inside each reversion level measured. For every episode it records:

* the hours from the extreme to the first snapshot with ``|z|`` at or inside each level, or that
  the record ended first;
* the live price's move, in basis points, from the extreme to that snapshot and to 24 hours after
  the extreme (the policy's minimum hold), signed both ways: *fade* (short a crowd paying to be
  long, long a crowd paying to be short) and *follow* (the other side), each less the 12 bp
  round-trip taker fee.

Run 1 placed no order, so none of this is a trading result: it says how long a funding extreme
lasted on this universe and what the price did meanwhile, which is what a reversion exit would have
to be judged against. DESIGN.md §24 records the result and why run 2 carries no such exit.
Nothing here calls the network, a credential or the model.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from replay_triggers import _read  # noqa: E402
from sentiment_agent.policy import POLICY_V2  # noqa: E402
from sentiment_agent.types import EventKind, PerceptionSnapshot, SnapshotEvent  # noqa: E402

LEVELS = (1.0, 0.5, 0.0)
"""``|z|`` levels a reversion is measured at; 0.0 means the z-score changed sign or reached zero."""

ROUND_TRIP_BPS = 12.0
"""Two taker fills at the Demo ``takerFeeRate`` of 0.0006 (``validation/universe_probe.json``)."""

HOLD = timedelta(hours=POLICY_V2.min_hold_hours)


@dataclass
class Episode:
    symbol: str
    opened_at: datetime
    z: float
    rate: float
    price: float
    reverted: dict[float, tuple[datetime, float]] = field(default_factory=dict)
    at_hold: float | None = None


def _inside(z: float, level: float, sign: int) -> bool:
    return z * sign <= 0 if level == 0.0 else abs(z) <= level


def _bps(entry: float, exit_: float) -> float:
    return (exit_ / entry - 1.0) * 10_000


def episodes(snapshots: list[PerceptionSnapshot]) -> tuple[list[Episode], datetime]:
    rule = POLICY_V2.triggers
    open_: dict[str, Episode] = {}
    done: list[Episode] = []
    for snapshot in snapshots:
        now = snapshot.taken_at
        for symbol, features in sorted(snapshot.features.items()):
            z, rate, price = features.funding_z_live, features.funding_rate_live, features.live_last
            if z is None or not math.isfinite(z) or price is None or price <= 0:
                continue
            episode = open_.get(symbol)
            if episode is not None:
                sign = 1 if episode.z > 0 else -1
                if episode.at_hold is None and now - episode.opened_at >= HOLD:
                    episode.at_hold = price
                for level in LEVELS:
                    if level not in episode.reverted and _inside(z, level, sign):
                        episode.reverted[level] = (now, price)
                if len(episode.reverted) == len(LEVELS) and episode.at_hold is not None:
                    done.append(open_.pop(symbol))
                continue
            if abs(z) <= rule.funding_z_threshold:
                continue
            if rule.funding_abs_min and (rate is None or abs(rate) < rule.funding_abs_min):
                continue
            open_[symbol] = Episode(symbol, now, z, float(rate or 0.0), price)
    return done + list(open_.values()), snapshots[-1].taken_at


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(values),
        "median": round(statistics.median(ordered), 2),
        "mean": round(statistics.fmean(ordered), 2),
        "min": round(ordered[0], 2),
        "max": round(ordered[-1], 2),
    }


def report(found: list[Episode], ended: datetime) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for e in sorted(found, key=lambda e: (e.opened_at, e.symbol)):
        fade = -1 if e.z > 0 else 1
        row: dict[str, Any] = {
            "symbol": e.symbol,
            "opened_at": e.opened_at.isoformat(),
            "z": round(e.z, 3),
            "rate_bp": round(e.rate * 10_000, 2),
        }
        for level in LEVELS:
            hit = e.reverted.get(level)
            key = f"z_within_{level:g}"
            if hit is None:
                row[key] = {
                    "reverted": False,
                    "observed_hours": round((ended - e.opened_at) / timedelta(hours=1), 2),
                }
                continue
            at, price = hit
            move = _bps(e.price, price)
            row[key] = {
                "reverted": True,
                "hours": round((at - e.opened_at) / timedelta(hours=1), 2),
                "fade_net_bps": round(fade * move - ROUND_TRIP_BPS, 1),
                "follow_net_bps": round(-fade * move - ROUND_TRIP_BPS, 1),
            }
        if e.at_hold is not None:
            move = _bps(e.price, e.at_hold)
            row["at_min_hold"] = {
                "fade_net_bps": round(fade * move - ROUND_TRIP_BPS, 1),
                "follow_net_bps": round(-fade * move - ROUND_TRIP_BPS, 1),
            }
        rows.append(row)
    by_level: dict[str, Any] = {}
    for level in LEVELS:
        key = f"z_within_{level:g}"
        hits = [r[key] for r in rows if r[key]["reverted"]]
        by_level[key] = {
            "reverted": len(hits),
            "not_reverted_before_record_end": len(rows) - len(hits),
            "hours": _summary([h["hours"] for h in hits]),
            "fade_net_bps": _summary([h["fade_net_bps"] for h in hits]),
            "follow_net_bps": _summary([h["follow_net_bps"] for h in hits]),
        }
    held = [r["at_min_hold"] for r in rows if "at_min_hold" in r]
    return {
        "episodes": len(rows),
        "record_ends": ended.isoformat(),
        "rule": {
            "funding_z_threshold": POLICY_V2.triggers.funding_z_threshold,
            "funding_abs_min": POLICY_V2.triggers.funding_abs_min,
            "min_hold_hours": POLICY_V2.min_hold_hours,
            "round_trip_bps": ROUND_TRIP_BPS,
        },
        "by_level": by_level,
        "at_min_hold": {
            "fade_net_bps": _summary([h["fade_net_bps"] for h in held]),
            "follow_net_bps": _summary([h["follow_net_bps"] for h in held]),
        },
        "rows": rows,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("public", type=Path, help="the run's public/ folder")
    parser.add_argument("--until-seq", type=int, default=None, help="last ledger seq to read")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    events = _read(args.public)
    if args.until_seq is not None:
        events = [e for e in events if e.seq <= args.until_seq]
    snapshots = [
        SnapshotEvent.model_validate(e.payload).snapshot
        for e in events
        if e.kind is EventKind.SNAPSHOT
    ]
    if not snapshots:
        raise SystemExit("the record holds no snapshot")
    found, ended = episodes(snapshots)
    out = {
        "input": {
            "ledger_events": len(events),
            "ledger_head_seq": events[-1].seq,
            "ledger_head_hash": events[-1].hash,
            "snapshots": len(snapshots),
        },
        **report(found, ended),
    }
    text = json.dumps(out, indent=2, ensure_ascii=False) + "\n"
    if args.out is not None:
        args.out.write_text(text, encoding="utf-8")
    print(
        json.dumps({k: out[k] for k in ("input", "episodes", "by_level", "at_min_hold")}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
