r"""Run 2's recorded decisions ruled again by another policy's kernel, SIMULATED, no model call.

    python scripts/rerule.py <public dir> [--until-seq N] [--answers FILE] [--answer-blobs DIR]
                             [--policy run2-a1..run2-a6|policy-v2] [--out FILE]

    python scripts/rerule.py ../t2-sentiment-agent/public --until-seq 363 \
        --out validation/run2/rerule_v2.json

``act_rate.py`` put run 2's agent to the live model on the 15 event decisions ``run2-a1`` would
have woken on run 1's market, and ruled each proposal with that step's kernel, which changed none
of them. This script asks whether policy v2's kernel would: it rebuilds the same 15 decisions,
answers each with the completion the model actually returned then (read from the attempt records
``act_rate.py`` stored under ``--answer-blobs``; no model is called and nothing is spent), lets the
real :class:`~sentiment_agent.decision.agent.DecisionAgent` parse and ground that answer exactly as
the runtime does, and rules the proposal with the kernel of ``--policy``.

A check makes the replay honest: every re-parsed proposal must equal the one ``act_rate.py``
recorded for the same moment, or the script stops. What it does not model: each decision is
ruled against the book run 1 actually held, which was flat, as in ``act_rate.py``; in a live run
the earlier decisions' positions would be held and count against the caps. It sends no order and
writes no ledger, and it is labelled SIMULATED wherever it is published.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from act_rate import STEPS, decide_and_rule, event_batches, read_record, write  # noqa: E402
from sentiment_agent.clock import ManualClock  # noqa: E402
from sentiment_agent.decision.agent import DecisionAgent  # noqa: E402
from sentiment_agent.kernel.kernel import RiskKernel  # noqa: E402
from sentiment_agent.ledger.blobs import FileBlobStore  # noqa: E402
from sentiment_agent.llm.fakes import ScriptedChatModel  # noqa: E402
from sentiment_agent.policy import POLICY_V2, RUN2_STEPS  # noqa: E402
from sentiment_agent.types import Completion  # noqa: E402

LABEL = (
    "SIMULATED — run 2's recorded answers on run 1's market, ruled again by another kernel; "
    "no model call, no order, no ledger"
)
CLUSTER = frozenset(s for c in POLICY_V2.cluster_caps for s in c.symbols)


def recorded_answers(blobs: Path) -> dict[str, list[Completion]]:
    """Every completion ``act_rate.py`` recorded, keyed by the decision time its request names
    (``Decision time 2026-09-24T21:31:26Z``), in attempt order."""
    found: dict[str, list[tuple[int, Completion]]] = defaultdict(list)
    for path in sorted(blobs.iterdir()):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(doc, dict) or "attempt" not in doc or "completion" not in doc:
            continue
        text = json.dumps(doc["request"])
        marker = "Decision time "
        at = text[text.index(marker) + len(marker) :].split(".", 1)[0]
        found[at].append((int(doc["attempt"]), Completion.model_validate(doc["completion"])))
    return {at: [c for _, c in sorted(pairs, key=lambda p: p[0])] for at, pairs in found.items()}


def _iso_z(at: str) -> str:
    return at.replace("+00:00", "Z").split(".")[0].removesuffix("Z") + "Z"


def _exposure(weights: dict[str, float]) -> dict[str, float]:
    return {
        "gross": round(sum(abs(w) for w in weights.values()), 6),
        "net": round(sum(weights.values()), 6),
        "crypto_beta_gross": round(sum(abs(w) for s, w in weights.items() if s in CLUSTER), 6),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("public", type=Path, help="the run's public/ folder (read only)")
    parser.add_argument("--until-seq", type=int, default=363)
    parser.add_argument(
        "--answers", type=Path, default=ROOT / "validation" / "run2" / "act_rate.json"
    )
    parser.add_argument("--answer-blobs", type=Path, default=ROOT / "var" / "act_rate" / "blobs")
    parser.add_argument(
        "--policy",
        choices=[*STEPS, "policy-v2"],
        default="policy-v2",
        help="the policy whose kernel rules the recorded proposals",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "validation" / "run2" / "rerule_v2.json")
    args = parser.parse_args(argv)
    kernel_policy = STEPS.get(args.policy, POLICY_V2)

    measured: dict[str, Any] = json.loads(args.answers.read_text(encoding="utf-8"))
    found = [
        (change_id, p)
        for change_id, p in RUN2_STEPS
        if p.content_hash() == measured["input"]["policy_hash"]
    ]
    if not found:
        raise SystemExit("the answers were not recorded under any run-2 policy step")
    step, decided_under = found[0]
    record = read_record(args.public, args.until_seq)
    if record.events[-1].hash != measured["input"]["ledger_head_hash"]:
        raise SystemExit("the answers were recorded on a different cut of the ledger")
    batches = event_batches(record, decided_under)
    answers = recorded_answers(args.answer_blobs)
    by_time = {_iso_z(r["at"]): r for r in measured["decisions"]}

    rows: list[dict[str, Any]] = []
    for batch in batches:
        at = _iso_z(batch.at.isoformat())
        before = by_time.get(at)
        script = answers.get(at)
        if before is None or not script:
            raise SystemExit(f"no recorded answer for the decision at {at}")
        clock = ManualClock(batch.at)
        agent = DecisionAgent(
            model=ScriptedChatModel(script),
            policy=decided_under,
            blobs=FileBlobStore(ROOT / "var" / "rerule" / "blobs"),
            clock=clock,
        )
        row = decide_and_rule(
            agent, RiskKernel(kernel_policy, clock), clock, record, batch, decided_under
        )
        if row["proposed_weights"] != before["proposed_weights"]:
            raise SystemExit(
                f"{at}: the recorded answer re-parses to {row['proposed_weights']}, but "
                f"act_rate.py recorded {before['proposed_weights']}"
            )
        row.pop("tokens", None)
        row["approved_weights_before"] = before.get("approved_weights")
        row["exposure_proposed"] = _exposure(row["proposed_weights"])
        row["exposure_approved"] = _exposure(row.get("approved_weights") or {})
        rows.append(row)
        print(
            f"{at} proposed {row['exposure_proposed']} -> approved {row['exposure_approved']}",
            flush=True,
        )

    acted = [r for r in rows if r["stance"] == "act"]
    cut = [r for r in acted if r.get("kernel_changed")]
    doc = {
        "label": LABEL,
        "input": {
            "public": args.public.name,
            "until_seq": args.until_seq,
            "ledger_head_hash": record.events[-1].hash,
            "answers": args.answers.relative_to(ROOT).as_posix()
            if args.answers.is_relative_to(ROOT)
            else args.answers.name,
            "answers_policy_step": step,
            "answers_policy_hash": decided_under.content_hash(),
            "kernel_policy": kernel_policy.version,
            "kernel_policy_step": args.policy,
            "kernel_policy_hash": kernel_policy.content_hash(),
        },
        "event_decisions": len(rows),
        "acted": len(acted),
        "cut_by_kernel": len(cut),
        "max_net_proposed": min((r["exposure_proposed"]["net"] for r in acted), default=None),
        "max_net_approved": min((r["exposure_approved"]["net"] for r in acted), default=None),
        "max_crypto_beta_proposed": max(
            (r["exposure_proposed"]["crypto_beta_gross"] for r in acted), default=None
        ),
        "max_crypto_beta_approved": max(
            (r["exposure_approved"]["crypto_beta_gross"] for r in acted), default=None
        ),
        "decisions": rows,
    }
    write(args.out, doc)
    print(json.dumps({k: v for k, v in doc.items() if k != "decisions"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
