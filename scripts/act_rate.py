r"""Run 2's act rate, measured on run 1's market, SIMULATED.

The event decisions run 2's trigger rules would have woken on run 1's recorded market, each put to
the live decision agent under policy v2.

    python scripts/act_rate.py <public dir> --secrets-root DIR [--until-seq N] [--out FILE]

    python scripts/act_rate.py ../t2-sentiment-agent/public --until-seq 363 \
        --secrets-root <a folder holding .secrets/qwen.env> \
        --out validation/run2/act_rate.json

``replay_triggers.py`` counted the triggers run 2 would admit (15 event decisions under policy v2
on run 1's first 364 events); it did not ask the agent anything, so "how often would run 2 act"
had no measured answer (readiness backlog L20). This script replays the same admissions and, for
each event decision, calls the real :class:`~sentiment_agent.decision.agent.DecisionAgent` with
the live Qwen model on the snapshot run 1 recorded at that moment, relabelled to policy v2 exactly
as the trigger replay does, the book run 1 held then, and the triggers that woke it. The real
:class:`~sentiment_agent.kernel.kernel.RiskKernel` then rules on each proposal with the
instrument specs run 1 logged.

What it measures and what it does not: the agent's stance and proposed weights, and whether the
kernel would have approved an order, on real recorded inputs. It sends no order, touches no
ledger, and reads run 1's record read-only (the chain is verified first). The market is run 1's,
the moment is past, and the model is called now, so this is a simulation of run 2's decisions,
labelled SIMULATED wherever it is published, never a paper-trading result. Every call is written
to ``--blobs`` so the answer behind each figure can be read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from replay_triggers import OI_MEDIA_TYPE, _Events, _read  # noqa: E402
from sentiment_agent.book.projection import Projection  # noqa: E402
from sentiment_agent.clock import ManualClock  # noqa: E402
from sentiment_agent.decision.agent import DecisionAgent  # noqa: E402
from sentiment_agent.events.replay import replay_admissions  # noqa: E402
from sentiment_agent.kernel.kernel import RiskKernel  # noqa: E402
from sentiment_agent.ledger.blobs import FileBlobStore  # noqa: E402
from sentiment_agent.llm.budget import (  # noqa: E402
    MEASURED_PROMPT_BYTES,
    DailyTokenBudget,
    decision_bound,
)
from sentiment_agent.llm.client import QwenChatModel, load_qwen_env  # noqa: E402
from sentiment_agent.policy import POLICY_V1, POLICY_V2  # noqa: E402
from sentiment_agent.runtime.cli import decisions_with_inputs  # noqa: E402
from sentiment_agent.types import (  # noqa: E402
    BreakerState,
    DecisionEvent,
    EventKind,
    Genesis,
    KernelInputs,
    PerceptionSnapshot,
    PriceSource,
    RulingContext,
    SnapshotEvent,
)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("public", type=Path, help="the run's public/ folder (read only)")
    parser.add_argument(
        "--secrets-root", type=Path, required=True, help="a folder holding .secrets/qwen.env"
    )
    parser.add_argument("--until-seq", type=int, default=363)
    parser.add_argument("--blobs", type=Path, default=ROOT / "var" / "act_rate" / "blobs")
    parser.add_argument("--out", type=Path, default=ROOT / "validation" / "run2" / "act_rate.json")
    args = parser.parse_args(argv)
    public: Path = args.public

    events = [e for e in _read(public) if e.seq <= args.until_seq]
    genesis_event = events[0]
    genesis = Genesis.model_validate(genesis_event.payload)
    if genesis.policy_hash != POLICY_V1.content_hash():
        raise SystemExit("this ledger was not pre-registered under policy v1")
    oi_ref = next(b for b in genesis_event.blobs if b.media_type == OI_MEDIA_TYPE)
    oi = json.loads((public / "blobs" / oi_ref.sha256).read_text(encoding="utf-8"))
    thresholds = {s: float(v) for s, v in oi["thresholds_pct"].items()}

    snapshots: list[PerceptionSnapshot] = [
        SnapshotEvent.model_validate(e.payload).snapshot
        for e in events
        if e.kind is EventKind.SNAPSHOT
    ]
    by_id = {s.snapshot_id: s for s in snapshots}
    decisions = [
        DecisionEvent.model_validate(e.payload).record
        for e in events
        if e.kind is EventKind.DECISION
    ]
    prompt_bytes = max(
        [MEASURED_PROMPT_BYTES]
        + [d.call.request_blob.size for d in decisions if d.call.request_blob is not None]
    )
    projection = Projection.from_ledger(_Events(events), genesis.policy)

    def book_at(snapshot: PerceptionSnapshot) -> Any:
        marks = {s: q.mark for s, q in snapshot.demo_quotes.items() if q.mark is not None}
        return projection.book(at=snapshot.taken_at, marks=marks, mark_source=PriceSource.DEMO)

    specs: dict[str, Any] = {}
    _, logged, _ = decisions_with_inputs(events, FileBlobStore(public / "blobs"))
    for given in logged:
        specs.update(given.specs)

    replay = replay_admissions(
        snapshots,
        policy=POLICY_V2,
        oi_thresholds=thresholds,
        start=genesis.created_at,
        book_at=book_at,
        decision_bound_tokens=decision_bound(prompt_bytes, POLICY_V2),
    )
    batches = [b for b in replay.batches if b.event_decision and not b.budget_refused]
    bound = decision_bound(prompt_bytes, POLICY_V2)
    cap = bound * max(len(batches), 1)
    print(
        f"{len(batches)} event decision(s); Qwen spend capped at {cap} tokens "
        f"({bound} per decision, the runtime's own bound)",
        flush=True,
    )

    clock = ManualClock(batches[0].at if batches else genesis.created_at)
    blobs = FileBlobStore(args.blobs)
    model = QwenChatModel(
        credentials=load_qwen_env(args.secrets_root),
        budget=DailyTokenBudget(cap, clock),
        clock=clock,
        blobs=blobs,
        timeout_s=float(POLICY_V2.decision.call_timeout_seconds),
    )
    agent = DecisionAgent(model=model, policy=POLICY_V2, blobs=blobs, clock=clock)
    kernel = RiskKernel(POLICY_V2, clock)

    rows: list[dict[str, Any]] = []
    for batch in batches:
        seen = by_id[batch.snapshot_id].model_copy(update={"policy_version": POLICY_V2.version})
        clock.set(seen.taken_at)
        book = book_at(seen)
        record = agent.decide(seen, book, list(batch.admitted))
        row: dict[str, Any] = {
            "at": seen.taken_at.isoformat(),
            "triggers": [f"{t.kind.value}:{','.join(t.symbols)}" for t in batch.admitted],
            "outcome": record.outcome.value,
            "stance": record.decision.stance.value if record.decision else None,
            "proposed_weights": record.proposed_weights,
            "tokens": record.call.usage.total_tokens,
            "decision_id": record.decision_id,
        }
        if record.decision is not None and record.proposed_weights:
            ruling = kernel.rule(
                proposed=dict(record.proposed_weights),
                book=book,
                inputs=KernelInputs(
                    at=seen.taken_at,
                    demo_quotes=dict(seen.demo_quotes),
                    live_quotes=dict(seen.live_quotes),
                    specs=specs,
                    demo_index_move_bps_3h={
                        s: f.demo_index_move_bps_3h for s, f in seen.features.items()
                    },
                    snapshot_id=seen.snapshot_id,
                    snapshot_taken_at=seen.taken_at,
                ),
                context=RulingContext(
                    decision_id=record.decision_id,
                    protective_reason=None,
                    grounding=dict(record.grounding),
                    invalidation_fired={},
                ),
                breaker=BreakerState(activation=book.activation, since=book.as_of, trips=()),
            )
            row["approved_weights"] = {r.symbol: r.approved_weight for r in ruling.instruments}
            row["kernel_changed"] = any(
                abs(r.approved_weight - record.proposed_weights.get(r.symbol, r.approved_weight))
                > 1e-12
                for r in ruling.instruments
            )
        rows.append(row)
        print(
            f"{row['at']} {row['triggers']} -> {row['outcome']} {row['stance']} "
            f"{row['proposed_weights']}",
            flush=True,
        )

    acted = [r for r in rows if r["stance"] == "act"]
    approved = [
        r for r in acted if any(abs(w) > 0 for w in (r.get("approved_weights") or {}).values())
    ]
    out = {
        "label": "SIMULATED — run 2's agent on run 1's recorded market; no order, no ledger",
        "input": {
            "public": public.name,
            "until_seq": args.until_seq,
            "ledger_head_hash": events[-1].hash,
            "policy": POLICY_V2.version,
            "policy_hash": POLICY_V2.content_hash(),
            "model": model.model_name,
        },
        "event_decisions": len(rows),
        "decided": sum(1 for r in rows if r["outcome"] == "decided"),
        "stances": {
            s: sum(1 for r in rows if r["stance"] == s)
            for s in sorted({str(r["stance"]) for r in rows})
        },
        "act_rate": round(len(acted) / len(rows), 4) if rows else None,
        "acted_and_approved_by_kernel": len(approved),
        "tokens_spent": sum(int(r["tokens"] or 0) for r in rows),
        "decisions": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in out.items() if k != "decisions"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
