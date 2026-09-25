r"""Replay a recorded paper run's snapshots through the trigger rules of policy v1 and policy v2.

    python scripts/replay_triggers.py <public dir> [--until-seq N] [--out FILE]

    python scripts/replay_triggers.py ../t2-sentiment-agent/public --until-seq 363 \
        --out validation/run2/run1_trigger_replay.json

Reads the run's published ``ledger.jsonl`` and ``blobs/`` (read only; the chain is verified first,
and ``--until-seq`` cuts the log at a sequence number so a figure taken from a growing record can
be reproduced exactly), takes every logged snapshot in order and feeds it through a fresh trigger
engine exactly as the runtime does (``sentiment_agent.events.replay``), once under each policy,
with the open-interest thresholds the run's genesis froze. It prints and writes:

* ``as_run``: the TRIGGER events the run itself logged, admitted and refused, for comparison;
* ``policy_v1``: run 2's trigger wiring (``run2-d1``: full-snapshot kinds evaluated on full
  snapshots) under the policy the run ran; it isolates the wiring fix, and its heartbeat decisions
  must equal the run's own;
* ``policy_v2``: the same wiring under run 2's policy (``funding_zscore`` over the whole universe);
* ``funding_extremes``: per instrument, the snapshots with a live funding z-score beyond the
  threshold, before any cooldown, cap or weekend rule.

Run 2's genesis declares these figures (``sentiment_agent.run2``), and a test holds the declaration
to this file. Nothing here calls the network, a credential or the model.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentiment_agent.book.projection import Projection  # noqa: E402
from sentiment_agent.events.replay import funding_extremes, replay_admissions  # noqa: E402
from sentiment_agent.ledger.chain import verify_file  # noqa: E402
from sentiment_agent.llm.budget import MEASURED_PROMPT_BYTES, decision_bound  # noqa: E402
from sentiment_agent.policy import POLICY_V1, POLICY_V2  # noqa: E402
from sentiment_agent.types import (  # noqa: E402
    DecisionEvent,
    EventKind,
    Genesis,
    LedgerEvent,
    PerceptionSnapshot,
    PriceSource,
    SnapshotEvent,
    Trigger,
)

OI_MEDIA_TYPE = "application/vnd.t2sa.oi-thresholds+json"


class _Events:
    def __init__(self, events: list[LedgerEvent]) -> None:
        self._events = events

    def events(self, kinds: frozenset[EventKind] | None = None) -> Iterator[LedgerEvent]:
        return iter([e for e in self._events if kinds is None or e.kind in kinds])

    def head(self) -> LedgerEvent | None:
        return self._events[-1] if self._events else None


def _read(public: Path) -> list[LedgerEvent]:
    ledger = public / "ledger.jsonl"
    verification = verify_file(ledger, blobs_root=public / "blobs")
    if not verification.intact:
        raise SystemExit(f"{ledger} does not verify: {verification.anchor}")
    with ledger.open(encoding="utf-8") as handle:
        return [LedgerEvent.model_validate_json(line) for line in handle if line.strip()]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("public", type=Path, help="the run's public/ folder")
    parser.add_argument("--until-seq", type=int, default=None, help="last ledger seq to read")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    public: Path = args.public

    events = _read(public)
    if args.until_seq is not None:
        events = [e for e in events if e.seq <= args.until_seq]
    genesis_event = events[0]
    genesis = Genesis.model_validate(genesis_event.payload)
    oi_ref = next(b for b in genesis_event.blobs if b.media_type == OI_MEDIA_TYPE)
    oi_record = json.loads((public / "blobs" / oi_ref.sha256).read_text(encoding="utf-8"))
    thresholds = {s: float(v) for s, v in oi_record["thresholds_pct"].items()}

    snapshots: list[PerceptionSnapshot] = [
        SnapshotEvent.model_validate(e.payload).snapshot
        for e in events
        if e.kind is EventKind.SNAPSHOT
    ]
    as_run = [Trigger.model_validate(e.payload) for e in events if e.kind is EventKind.TRIGGER]
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

    out: dict[str, Any] = {
        "input": {
            "ledger_events": len(events),
            "ledger_head_seq": events[-1].seq,
            "ledger_head_hash": events[-1].hash,
            "genesis_hash": genesis_event.hash,
            "genesis_policy_hash": genesis.policy_hash,
            "code_commit": genesis.code_commit,
            "oi_thresholds_pct": thresholds,
            "decision_bound_tokens": decision_bound(prompt_bytes, POLICY_V2),
        },
        "as_run": {
            "triggers": len(as_run),
            "by_kind": dict(sorted(Counter(t.kind.value for t in as_run).items())),
            "decisions": len(decisions),
        },
    }
    if genesis.policy_hash != POLICY_V1.content_hash():
        raise SystemExit("this ledger was not pre-registered under policy v1")
    for name, policy in (("policy_v1", POLICY_V1), ("policy_v2", POLICY_V2)):
        result = replay_admissions(
            snapshots,
            policy=policy,
            oi_thresholds=thresholds,
            start=genesis.created_at,
            book_at=book_at,
            decision_bound_tokens=decision_bound(prompt_bytes, policy),
        )
        out[name] = {"policy_hash": policy.content_hash(), **result.summary()}
    out["funding_extremes"] = {
        "threshold": POLICY_V2.triggers.funding_z_threshold,
        "snapshots_beyond_by_symbol": funding_extremes(
            snapshots, threshold=POLICY_V2.triggers.funding_z_threshold
        ),
    }
    text = json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
