"""Run 2's switchover checks (RUNBOOK "Run 2", §4), run against the two roots and read, never
written: run 2's genesis is the one this code declares, run 2's first hourly record is actually on
its page, and run 1's record is exactly the one it closed with.

    python scripts/switchover_check.py --run1 <run 1 root> --run2 <run 2 root> [--run1-head HASH]

Each check prints PASS, FAIL or PENDING with the evidence it read; the exit code is 1 if any check
FAILs. PENDING means the thing checked has not happened yet (no genesis, no publish), not that it
went wrong. ``--run1-head`` is the ledger head printed when run 1 was stopped; without it the check
reports the head it finds and whether the chain verifies, and says it could not compare.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sentiment_agent.clock import ManualClock
from sentiment_agent.ledger.chain import HashChainLedger, LedgerError, ledger_path
from sentiment_agent.policy import ACTIVE_POLICY
from sentiment_agent.run2 import DECLARED_CHANGES, PREDECESSOR
from sentiment_agent.types import EventKind, Genesis, RunMode


@dataclass(frozen=True)
class Check:
    name: str
    state: str  # PASS | FAIL | PENDING
    evidence: str

    def line(self) -> str:
        return f"{self.state:8} {self.name}: {self.evidence}"


def _chain(root: Path) -> HashChainLedger | None:
    path = ledger_path(root, RunMode.PAPER)
    if not path.exists():
        return None
    return HashChainLedger(path, mode=RunMode.PAPER, clock=ManualClock(datetime.now(UTC)))


def check_genesis(run1: Path, run2: Path) -> list[Check]:
    chain = _chain(run2)
    first = next(iter(chain.events(frozenset({EventKind.GENESIS})))) if chain else None
    if chain is None or first is None:
        return [Check("run 2 genesis", "PENDING", "run 2 has no paper genesis yet")]
    genesis = Genesis.model_validate(first.payload)
    out = [
        Check(
            "policy in force",
            "PASS" if genesis.policy.version == ACTIVE_POLICY.version else "FAIL",
            f"genesis {genesis.policy.version} {genesis.policy_hash[:16]}, this code "
            f"{ACTIVE_POLICY.version} {ACTIVE_POLICY.content_hash()[:16]}",
        )
    ]
    run1_chain = _chain(run1)
    run1_first = (
        next(iter(run1_chain.events(frozenset({EventKind.GENESIS})))) if run1_chain else None
    )
    named = genesis.predecessor.genesis_hash if genesis.predecessor else None
    actual = run1_first.hash if run1_first else None
    out.append(
        Check(
            "predecessor",
            "PASS" if named and named == PREDECESSOR.genesis_hash == actual else "FAIL",
            f"run 2 names {named}, this code declares {PREDECESSOR.genesis_hash}, run 1's "
            f"ledger opens with {actual}",
        )
    )
    declared = [c.change_id for c in genesis.declared_changes]
    expected = [c.change_id for c in DECLARED_CHANGES]
    out.append(
        Check(
            "declared changes",
            "PASS" if declared == expected else "FAIL",
            f"genesis {declared}, this code {expected}",
        )
    )
    out.append(
        Check(
            "window and mandate (owner's P05/P06)",
            "PASS" if "window" in genesis.statement.lower() else "FAIL",
            f"the statement reads: {genesis.statement[:300]!r}",
        )
    )
    return out


def check_first_publish(run2: Path) -> Check:
    log = run2 / "var" / "logs" / "site.log"
    if not log.exists():
        return Check("first hourly publish", "PENDING", "run 2 has no site.log yet")
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    published = [line for line in lines if " published record generated " in line]
    aliased = [line for line in published if "Aliased" in line]
    if aliased:
        return Check("first hourly publish", "PASS", aliased[0][:200])
    if published or lines:
        return Check("first hourly publish", "FAIL", f"no aliased deploy; last line: {lines[-1]}")
    return Check("first hourly publish", "PENDING", "site.log is empty")


def check_run1_untouched(run1: Path, expected_head: str | None) -> Check:
    chain = _chain(run1)
    if chain is None:
        return Check("run 1 untouched", "FAIL", "run 1's paper ledger is missing")
    try:
        chain.verify()
    except LedgerError as exc:
        return Check("run 1 untouched", "FAIL", f"run 1's chain does not verify: {exc}")
    head = chain.head()
    if head is None:
        return Check("run 1 untouched", "FAIL", "run 1's paper ledger is empty")
    if expected_head is None:
        return Check(
            "run 1 untouched",
            "PENDING",
            f"chain verifies, head seq {head.seq} {head.hash}; pass --run1-head to compare",
        )
    return Check(
        "run 1 untouched",
        "PASS" if head.hash == expected_head else "FAIL",
        f"head seq {head.seq} {head.hash}, closed at {expected_head}",
    )


def run(run1: Path, run2: Path, run1_head: str | None) -> list[Check]:
    return [
        *check_genesis(run1, run2),
        check_first_publish(run2),
        check_run1_untouched(run1, run1_head),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run1", type=Path, required=True)
    parser.add_argument("--run2", type=Path, required=True)
    parser.add_argument("--run1-head", default=None)
    parser.add_argument("--json", action="store_true", help="print the checks as JSON")
    args = parser.parse_args(argv)
    checks = run(args.run1, args.run2, args.run1_head)
    if args.json:
        print(json.dumps([c.__dict__ for c in checks], indent=1))
    else:
        for check in checks:
            print(check.line())
    return 1 if any(c.state == "FAIL" for c in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
