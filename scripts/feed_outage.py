r"""How blind a recorded paper run was: which sources answered, and which readings stayed empty.

    python scripts/feed_outage.py <public dir> [--until-seq N] [--out FILE]

    python scripts/feed_outage.py ../t2-sentiment-agent/public --until-seq 551 \
        --out validation/run2/run1_feed_outage.json

Reads the run's published ``ledger.jsonl`` (read only; the chain is verified first, and
``--until-seq`` cuts the log so a figure taken from a growing record can be reproduced), and for
every logged snapshot counts each source call by surface and health, and each reading the model
and the triggers depend on by whether it arrived: crypto Fear & Greed, and per crypto instrument
the retail long/short ratio, the taker ratio and open interest. It reports the whole run and the
stretch since bitget-mcp-server's first failing call.

Run 2 declares the result as the evidence for ``run2-d3`` (``sentiment_agent.run2``): the
upstreams those services wrap, read directly when both fail. Nothing here calls the network.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentiment_agent.ledger.chain import verify_file  # noqa: E402
from sentiment_agent.types import (  # noqa: E402
    FAILING_HEALTH,
    AssetClass,
    EventKind,
    Genesis,
    LedgerEvent,
    PerceptionSnapshot,
    SnapshotEvent,
    SourceHealth,
    ToolkitSurface,
)

ANSWERED = frozenset({SourceHealth.OK, SourceHealth.EMPTY})


def _read(public: Path) -> tuple[list[LedgerEvent], int]:
    """The chain-verified events, and how many referenced blobs the public folder lacks.

    Only the hash chain is required here: every count below reads event payloads, none reads a
    blob, so a published folder missing some raw payloads can still be counted (the number
    missing is reported, not hidden)."""
    ledger = public / "ledger.jsonl"
    verification = verify_file(ledger, blobs_root=public / "blobs")
    if verification.first_break_at is not None or verification.truncated:
        raise SystemExit(f"{ledger} does not verify: {verification.break_reason}")
    with ledger.open(encoding="utf-8") as handle:
        events = [LedgerEvent.model_validate_json(line) for line in handle if line.strip()]
    return events, len(verification.missing_blobs)


def tally(snapshots: list[PerceptionSnapshot], crypto: list[str]) -> dict[str, Any]:
    """Calls by surface and health, and how many snapshots carried each reading."""
    calls: dict[str, Counter[str]] = {}
    for snap in snapshots:
        for call in snap.source_calls:
            calls.setdefault(call.surface.value, Counter())[call.health.value] += 1
    by_surface = {}
    for surface, health in sorted(calls.items()):
        asked = sum(n for h, n in health.items() if h != SourceHealth.DISABLED.value)
        answered = sum(health.get(h.value, 0) for h in ANSWERED)
        failing = sum(health.get(h.value, 0) for h in FAILING_HEALTH)
        by_surface[surface] = {
            "asked": asked,
            "answered": answered,
            "failing": failing,
            "by_health": dict(sorted(health.items())),
        }

    def carried(present: Any) -> int:
        return sum(1 for snap in snapshots if present(snap))

    readings: dict[str, Any] = {
        "crypto_fear_greed": carried(lambda s: s.mood.crypto_fear_greed is not None),
    }
    for symbol in crypto:
        readings[symbol] = {
            name: carried(
                lambda s, name=name, symbol=symbol: (
                    symbol in s.features and getattr(s.features[symbol], name) is not None
                )
            )
            for name in (
                "retail_long_short_ratio",
                "taker_buy_sell_ratio",
                "open_interest_live",
            )
        }
    return {"snapshots": len(snapshots), "calls": by_surface, "readings_present": readings}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("public", type=Path, help="the run's public/ folder")
    parser.add_argument("--until-seq", type=int, default=None, help="last ledger seq to read")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    events, missing_blobs = _read(args.public)
    if args.until_seq is not None:
        events = [e for e in events if e.seq <= args.until_seq]
    genesis = Genesis.model_validate(events[0].payload)
    crypto = [u.symbol for u in genesis.policy.universe if u.asset_class is AssetClass.CRYPTO]
    snapshots = [
        SnapshotEvent.model_validate(e.payload).snapshot
        for e in events
        if e.kind is EventKind.SNAPSHOT
    ]
    first_failure = next(
        (
            call.started_at
            for snap in snapshots
            for call in snap.source_calls
            if call.surface is ToolkitSurface.DATA_MCP and call.health in FAILING_HEALTH
        ),
        None,
    )
    since = [s for s in snapshots if first_failure is not None and s.taken_at >= first_failure]
    out = {
        "input": {
            "genesis_hash": events[0].hash,
            "ledger_head_seq": events[-1].seq,
            "published_blobs_missing": missing_blobs,
            "first_snapshot": snapshots[0].taken_at.isoformat() if snapshots else None,
            "last_snapshot": snapshots[-1].taken_at.isoformat() if snapshots else None,
        },
        "whole_run": tally(snapshots, crypto),
        "since_first_data_mcp_failure": {
            "from": first_failure.isoformat() if first_failure is not None else None,
            **tally(since, crypto),
        },
    }
    text = json.dumps(out, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
