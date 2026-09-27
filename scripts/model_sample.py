r"""T3 of the pre-genesis audit: the live model on run 1's recorded decision moments, under system
prompt v2 and the candidate v3, with each proposal's forward P&L on Demo marks. SIMULATED.

    python scripts/model_sample.py <run 1 public dir> --marks <history_marks.json> --dry-run
    python scripts/model_sample.py <run 1 public dir> --marks <history_marks.json> \
        --secrets-root <folder holding .secrets/qwen.env> --prompts v2,v3 --max-tokens N \
        [--kinds events,heartbeats] [--limit K] [--out validation/run2/model_sample.json]

Why. Replaying run 2's rules through the real kernel (`strategy_replay.py`) found no rule-based
signal with an edge, so run 2's numbers will come from the model's own decisions at its heartbeats
and events. This script measures those decisions before any paper week is spent on them: for every
decision moment run 2's triggers admit on run 1's recorded market (`events.replay`, the same
admissions `act_rate.py` replays, heartbeats included here), it calls the real
:class:`~sentiment_agent.decision.agent.DecisionAgent` with the live Qwen model once per prompt,
rules each proposal with the real :class:`~sentiment_agent.kernel.kernel.RiskKernel` (`act_rate.
decide_and_rule`), and scores what the kernel approved against what the Demo mark did next.

Scores per decision (Demo 1H mark closes from `research/harvest/quant/replay/history_marks.json`):
* forward P&L over 24 h of the approved weights, less 12 bps per unit of weight traded (a taker
  round trip), in bps of equity; the same book reversed (TradeRank's test, quant_B.md B5); and the
  coin flip: the same absolute weights with random signs, 1,000 draws, its mean and the share of
  draws the model's book beats;
* the book's side: net weight, and the share of gross that is short.

The prompt under test is chosen by pointing ``decision/prompt.py``'s ``SYSTEM_TEMPLATE`` at the
candidate file for that call; the production default is untouched. ``--dry-run`` calls nothing
and prints how many calls would be made and the runtime's own token bound for them. Every call is
written to ``--blobs``. Nothing here sends an order or touches a ledger.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from act_rate import decide_and_rule, read_record, write  # noqa: E402
from sentiment_agent.clock import ManualClock  # noqa: E402
from sentiment_agent.decision import prompt as prompt_module  # noqa: E402
from sentiment_agent.decision.agent import DecisionAgent  # noqa: E402
from sentiment_agent.events.replay import ReplayBatch, replay_admissions  # noqa: E402
from sentiment_agent.kernel.kernel import RiskKernel  # noqa: E402
from sentiment_agent.ledger.blobs import FileBlobStore  # noqa: E402
from sentiment_agent.llm.budget import DailyTokenBudget, decision_bound  # noqa: E402
from sentiment_agent.llm.client import QwenChatModel, load_qwen_env  # noqa: E402
from sentiment_agent.policy import POLICY_V2  # noqa: E402

PROMPTS = {
    "v2": "decision/prompts/system_v2.md",
    "v3": "decision/prompts/system_v3.md",
}
LABEL = "SIMULATED — the live model on run 1's recorded moments; no order, no ledger"
ROUND_TRIP_BPS = 12.0
HOLD = timedelta(hours=24)
COIN_DRAWS = 1000


def batches(record: Any, kinds: set[str]) -> list[ReplayBatch]:
    replay = replay_admissions(
        record.snapshots,
        policy=POLICY_V2,
        oi_thresholds=record.thresholds,
        start=record.genesis.created_at,
        book_at=record.book_at,
        decision_bound_tokens=decision_bound(record.prompt_bytes, POLICY_V2),
    )
    out = []
    for b in replay.batches:
        if not b.decision or b.budget_refused:
            continue
        kind = "events" if b.event_decision else "heartbeats"
        if kind in kinds:
            out.append(b)
    return out


def load_marks(path: Path) -> dict[str, dict[datetime, float]]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    return {
        s: {datetime.fromtimestamp(int(r[0]), UTC) + timedelta(hours=1): float(r[4]) for r in rows}
        for s, rows in doc["demo_mark"].items()
    }


def _close_at(series: dict[datetime, float], at: datetime) -> float | None:
    """The close of the latest hourly candle that closed at or before ``at``."""
    hour = at.replace(minute=0, second=0, microsecond=0)
    for k in range(3):
        v = series.get(hour - timedelta(hours=k))
        if v is not None:
            return v
    return None


def score(weights: dict[str, float], at: datetime, marks: dict[str, dict[datetime, float]],
          rng: random.Random) -> dict[str, Any] | None:  # fmt: skip
    moves: dict[str, float] = {}
    for s, w in weights.items():
        if w == 0:
            continue
        a, b = _close_at(marks.get(s, {}), at), _close_at(marks.get(s, {}), at + HOLD)
        if a is None or b is None:
            return None
        moves[s] = b / a - 1
    gross = sum(abs(w) for w in weights.values())
    fees = gross * ROUND_TRIP_BPS
    book = sum(weights[s] * m for s, m in moves.items()) * 10_000 - fees
    reversed_ = sum(-weights[s] * m for s, m in moves.items()) * 10_000 - fees
    flips = [
        sum(abs(weights[s]) * m * (1 if rng.random() < 0.5 else -1) for s, m in moves.items())
        * 10_000
        - fees
        for _ in range(COIN_DRAWS)
    ]
    short = sum(-w for w in weights.values() if w < 0)
    return {
        "forward_24h_bps": round(book, 3),
        "reversed_bps": round(reversed_, 3),
        "coin_flip_mean_bps": round(statistics.fmean(flips), 3),
        "beats_coin_flip_share": round(sum(1 for f in flips if book > f) / COIN_DRAWS, 3),
        "net": round(sum(weights.values()), 4),
        "short_share": round(short / gross, 3) if gross else None,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("public", type=Path)
    parser.add_argument("--marks", type=Path, required=True)
    parser.add_argument("--secrets-root", type=Path)
    parser.add_argument("--prompts", default="v2,v3")
    parser.add_argument("--kinds", default="events,heartbeats")
    parser.add_argument("--limit", type=int, default=0, help="at most K moments per kind")
    parser.add_argument("--until-seq", type=int, default=10**9)
    parser.add_argument("--max-tokens", type=int, default=0, help="hard ceiling for the run")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--blobs", type=Path, default=ROOT / "var" / "model_sample" / "blobs")
    parser.add_argument(
        "--out", type=Path, default=ROOT / "validation" / "run2" / "model_sample.json"
    )
    args = parser.parse_args(argv)
    prompts = [p for p in args.prompts.split(",") if p]
    kinds = set(args.kinds.split(","))
    record = read_record(args.public, args.until_seq)
    chosen = batches(record, kinds)
    if args.limit:
        by_kind: dict[bool, list[ReplayBatch]] = {True: [], False: []}
        for b in chosen:
            if len(by_kind[b.event_decision]) < args.limit:
                by_kind[b.event_decision].append(b)
        chosen = sorted(by_kind[True] + by_kind[False], key=lambda b: b.at)
    bound = decision_bound(record.prompt_bytes, POLICY_V2)
    calls = len(chosen) * len(prompts)
    events = sum(1 for b in chosen if b.event_decision)
    print(f"{len(chosen)} moments ({events} events, {len(chosen) - events} heartbeats) x "
          f"{len(prompts)} prompts = {calls} calls; runtime bound {bound} tokens per call, "
          f"{bound * calls} at most", flush=True)  # fmt: skip
    if args.dry_run:
        return 0
    if args.secrets_root is None or args.max_tokens <= 0:
        raise SystemExit("a live run needs --secrets-root and an explicit --max-tokens ceiling")
    marks = load_marks(args.marks)
    clock = ManualClock(chosen[0].at if chosen else record.genesis.created_at)
    blobs = FileBlobStore(args.blobs)
    model = QwenChatModel(
        credentials=load_qwen_env(args.secrets_root),
        budget=DailyTokenBudget(args.max_tokens, clock),
        clock=clock,
        blobs=blobs,
        timeout_s=float(POLICY_V2.decision.call_timeout_seconds),
    )
    kernel = RiskKernel(POLICY_V2, clock)
    rng = random.Random(20260927)  # noqa: S311 - a seeded null, not a secret
    rows: list[dict[str, Any]] = []
    spent = 0
    for batch in chosen:
        for name in prompts:
            prompt_module.SYSTEM_TEMPLATE = PROMPTS[name]  # type: ignore[misc]
            agent = DecisionAgent(model=model, policy=POLICY_V2, blobs=blobs, clock=clock)
            row = decide_and_rule(agent, kernel, clock, record, batch, POLICY_V2)
            spent += int(row.get("tokens") or 0)
            row["prompt"] = name
            row["kind"] = "event" if batch.event_decision else "heartbeat"
            approved = row.get("approved_weights") or {}
            row["score"] = score(dict(approved), batch.at, marks, rng) if approved else None
            rows.append(row)
            print(f"{row['at']} {row['kind']} {name} -> {row['stance']} {approved} "
                  f"{row['score']}", flush=True)  # fmt: skip
    prompt_module.SYSTEM_TEMPLATE = PROMPTS["v2"]  # type: ignore[misc]

    def summary(name: str) -> dict[str, Any]:
        mine = [r for r in rows if r["prompt"] == name]
        scored = [r["score"] for r in mine if r["score"]]
        return {
            "decisions": len(mine),
            "acted": sum(1 for r in mine if r["stance"] == "act"),
            "median_short_share": statistics.median(
                [s["short_share"] for s in scored if s["short_share"] is not None]
            ) if scored else None,
            "mean_forward_24h_bps": round(statistics.fmean(s["forward_24h_bps"] for s in scored), 3)
            if scored else None,
            "mean_reversed_bps": round(statistics.fmean(s["reversed_bps"] for s in scored), 3)
            if scored else None,
            "mean_beats_coin_flip_share": round(
                statistics.fmean(s["beats_coin_flip_share"] for s in scored), 3
            ) if scored else None,
        }  # fmt: skip

    out = {
        "label": LABEL,
        "input": {"public": args.public.name, "policy_hash": POLICY_V2.content_hash(),
                  "prompts": {p: PROMPTS[p] for p in prompts}, "model": model.model_name},
        "tokens_spent": spent,
        "summary": {p: summary(p) for p in prompts},
        "decisions": rows,
    }  # fmt: skip
    write(args.out, out)
    print(json.dumps(out["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
