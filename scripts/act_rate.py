r"""Run 2's act rate, measured on run 1's market, SIMULATED.

The event decisions run 2's trigger rules would have woken on run 1's recorded market, each put to
the live decision agent under policy v2, or under v1 amended up to a named step (``--policy``).

    python scripts/act_rate.py <public dir> --secrets-root DIR [--until-seq N] [--out FILE]
                               [--policy run2-a1..run2-a6|policy-v2]

    python scripts/act_rate.py ../t2-sentiment-agent/public --until-seq 363 \
        --secrets-root <a folder holding .secrets/qwen.env> \
        --policy run2-a1 --out validation/run2/act_rate.json

``replay_triggers.py`` counted the triggers run 2 would admit (15 event decisions under run2-a1 on
run 1's first 364 events, 3 under policy v2); it did not ask the agent anything, so "how often
would run 2 act" had no measured answer (readiness backlog L20). This script replays the same
admissions and, for each event decision, calls the real
:class:`~sentiment_agent.decision.agent.DecisionAgent` with the live Qwen model on the snapshot
run 1 recorded at that moment, relabelled to that policy exactly as the trigger replay does, the
book run 1 held then, and the triggers that woke it. The real
:class:`~sentiment_agent.kernel.kernel.RiskKernel` then rules on each proposal with the instrument
specs run 1 logged.

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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from replay_triggers import OI_MEDIA_TYPE, _Events, _read  # noqa: E402
from sentiment_agent.book.projection import Projection  # noqa: E402
from sentiment_agent.clock import ManualClock  # noqa: E402
from sentiment_agent.decision.agent import DecisionAgent  # noqa: E402
from sentiment_agent.events.replay import ReplayBatch, replay_admissions  # noqa: E402
from sentiment_agent.kernel.kernel import RiskKernel  # noqa: E402
from sentiment_agent.ledger.blobs import FileBlobStore  # noqa: E402
from sentiment_agent.llm.budget import (  # noqa: E402
    MEASURED_PROMPT_BYTES,
    DailyTokenBudget,
    decision_bound,
)
from sentiment_agent.llm.client import QwenChatModel, load_qwen_env  # noqa: E402
from sentiment_agent.policy import POLICY_V1, POLICY_V2, RUN2_STEPS  # noqa: E402
from sentiment_agent.runtime.cli import decisions_with_inputs  # noqa: E402
from sentiment_agent.types import (  # noqa: E402
    BookState,
    BreakerState,
    DecisionEvent,
    EventKind,
    Genesis,
    InstrumentSpec,
    KernelInputs,
    PerceptionSnapshot,
    Policy,
    PriceSource,
    RulingContext,
    SnapshotEvent,
)

STEPS = dict(RUN2_STEPS)
"""``--policy`` choices: each run-2 amendment's id and the policy in force after it.
``validation/run2/act_rate.json`` was measured under ``run2-a1``, before the level floor and the
book caps existed."""

LABEL = "SIMULATED — run 2's agent on run 1's recorded market; no order, no ledger"


@dataclass(frozen=True)
class Recorded:
    """Run 1's record up to a sequence number, read once: what every replayed decision is given."""

    events: list[Any]
    genesis: Genesis
    thresholds: dict[str, float]
    snapshots: list[PerceptionSnapshot]
    prompt_bytes: int
    specs: dict[str, InstrumentSpec]
    book_at: Callable[[PerceptionSnapshot], BookState]

    def snapshot(self, snapshot_id: str) -> PerceptionSnapshot:
        return next(s for s in self.snapshots if s.snapshot_id == snapshot_id)


def read_record(public: Path, until_seq: int) -> Recorded:
    events = [e for e in _read(public) if e.seq <= until_seq]
    genesis_event = events[0]
    genesis = Genesis.model_validate(genesis_event.payload)
    if genesis.policy_hash != POLICY_V1.content_hash():
        raise SystemExit("this ledger was not pre-registered under policy v1")
    oi_ref = next(b for b in genesis_event.blobs if b.media_type == OI_MEDIA_TYPE)
    oi = json.loads((public / "blobs" / oi_ref.sha256).read_text(encoding="utf-8"))
    snapshots: list[PerceptionSnapshot] = [
        SnapshotEvent.model_validate(e.payload).snapshot
        for e in events
        if e.kind is EventKind.SNAPSHOT
    ]
    decisions = [
        DecisionEvent.model_validate(e.payload).record
        for e in events
        if e.kind is EventKind.DECISION
    ]
    projection = Projection.from_ledger(_Events(events), genesis.policy)

    def book_at(snapshot: PerceptionSnapshot) -> BookState:
        marks = {s: q.mark for s, q in snapshot.demo_quotes.items() if q.mark is not None}
        return projection.book(at=snapshot.taken_at, marks=marks, mark_source=PriceSource.DEMO)

    specs: dict[str, InstrumentSpec] = {}
    _, logged, _ = decisions_with_inputs(events, FileBlobStore(public / "blobs"))
    for given in logged:
        specs.update(given.specs)
    return Recorded(
        events=events,
        genesis=genesis,
        thresholds={s: float(v) for s, v in oi["thresholds_pct"].items()},
        snapshots=snapshots,
        prompt_bytes=max(
            [MEASURED_PROMPT_BYTES]
            + [d.call.request_blob.size for d in decisions if d.call.request_blob is not None]
        ),
        specs=specs,
        book_at=book_at,
    )


def event_batches(record: Recorded, policy: Policy) -> list[ReplayBatch]:
    """The event decisions ``policy``'s triggers admit on the record, within the token budget."""
    replay = replay_admissions(
        record.snapshots,
        policy=policy,
        oi_thresholds=record.thresholds,
        start=record.genesis.created_at,
        book_at=record.book_at,
        decision_bound_tokens=decision_bound(record.prompt_bytes, policy),
    )
    return [b for b in replay.batches if b.event_decision and not b.budget_refused]


def decide_and_rule(
    agent: DecisionAgent,
    kernel: RiskKernel,
    clock: ManualClock,
    record: Recorded,
    batch: ReplayBatch,
    policy: Policy,
) -> dict[str, Any]:
    """One event decision: the agent on the recorded snapshot and book, then the kernel's ruling
    on what it proposed, with the instrument specs run 1 logged."""
    seen = record.snapshot(batch.snapshot_id).model_copy(update={"policy_version": policy.version})
    clock.set(seen.taken_at)
    book = record.book_at(seen)
    decided = agent.decide(seen, book, list(batch.admitted))
    row: dict[str, Any] = {
        "at": seen.taken_at.isoformat(),
        "triggers": [f"{t.kind.value}:{','.join(t.symbols)}" for t in batch.admitted],
        "outcome": decided.outcome.value,
        "stance": decided.decision.stance.value if decided.decision else None,
        "proposed_weights": decided.proposed_weights,
        "tokens": decided.call.usage.total_tokens,
        "decision_id": decided.decision_id,
    }
    if decided.decision is not None and decided.proposed_weights:
        ruling = kernel.rule(
            proposed=dict(decided.proposed_weights),
            book=book,
            inputs=KernelInputs(
                at=seen.taken_at,
                demo_quotes=dict(seen.demo_quotes),
                live_quotes=dict(seen.live_quotes),
                specs=record.specs,
                demo_index_move_bps_3h={
                    s: f.demo_index_move_bps_3h for s, f in seen.features.items()
                },
                snapshot_id=seen.snapshot_id,
                snapshot_taken_at=seen.taken_at,
            ),
            context=RulingContext(
                decision_id=decided.decision_id,
                protective_reason=None,
                grounding=dict(decided.grounding),
                invalidation_fired={},
            ),
            breaker=BreakerState(activation=book.activation, since=book.as_of, trips=()),
        )
        row["approved_weights"] = {r.symbol: r.approved_weight for r in ruling.instruments}
        row["kernel_changed"] = any(
            abs(r.approved_weight - decided.proposed_weights.get(r.symbol, r.approved_weight))
            > 1e-12
            for r in ruling.instruments
        )
        row["binding_guards"] = sorted(
            {r.binding_guard.value for r in ruling.instruments if r.binding_guard is not None}
        )
        row["book_rulings"] = [
            g.reason for g in ruling.book_rulings if g.status.value == "fired" and g.reason
        ]
    return row


def write(out: Path, doc: dict[str, Any]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
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
    parser.add_argument(
        "--policy",
        choices=[*STEPS, "policy-v2"],
        default="policy-v2",
        help="the run-2 policy to decide under: v2, or v1 amended up to a named step",
    )
    args = parser.parse_args(argv)
    policy = STEPS.get(args.policy, POLICY_V2)
    public: Path = args.public
    record = read_record(public, args.until_seq)
    batches = event_batches(record, policy)
    bound = decision_bound(record.prompt_bytes, policy)
    cap = bound * max(len(batches), 1)
    print(
        f"{len(batches)} event decision(s); Qwen spend capped at {cap} tokens "
        f"({bound} per decision, the runtime's own bound)",
        flush=True,
    )

    clock = ManualClock(batches[0].at if batches else record.genesis.created_at)
    blobs = FileBlobStore(args.blobs)
    model = QwenChatModel(
        credentials=load_qwen_env(args.secrets_root),
        budget=DailyTokenBudget(cap, clock),
        clock=clock,
        blobs=blobs,
        timeout_s=float(policy.decision.call_timeout_seconds),
    )
    agent = DecisionAgent(model=model, policy=policy, blobs=blobs, clock=clock)
    kernel = RiskKernel(policy, clock)

    rows: list[dict[str, Any]] = []
    for batch in batches:
        row = decide_and_rule(agent, kernel, clock, record, batch, policy)
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
        "label": LABEL,
        "input": {
            "public": public.name,
            "until_seq": args.until_seq,
            "ledger_head_hash": record.events[-1].hash,
            "policy": policy.version,
            "policy_step": args.policy,
            "policy_hash": policy.content_hash(),
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
    write(args.out, out)
    print(json.dumps({k: v for k, v in out.items() if k != "decisions"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
