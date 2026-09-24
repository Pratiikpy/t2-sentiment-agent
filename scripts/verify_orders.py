"""Check that every venue ``orderId`` in ``public/orders.json`` exists on the Bitget Demo account.

    python scripts/verify_orders.py [--public public/orders.json] [--root .] [--json]

For anyone checking the published record, judges included. It needs:

* ``pip install -e .`` in this repository and ``npm ci --ignore-scripts`` in ``tools/agent-hub``
  (Bitget's own CLI, pinned);
* a Bitget **Demo** API key for the account that traded, in ``<root>/.secrets/demo.env`` with
  ``BITGET_KEY_ENVIRONMENT=demo``. The owner may publish a read-only Demo key for this (DESIGN.md
  §21); a key with trade permission works too, because nothing here writes.

**Read-only by construction.** The only command this script runs is
``bgc order --action detail --orderId <id> --view full --paper-trading``: one read per order, on
the Demo environment. It builds no other argv. Before the key is handed to ``bgc``, the installed
CLI is checked against the pinned release (the same vendor contract the agent's environment proof
uses).

For each published order it prints ``FOUND``, ``MISSING`` (the venue has no such order),
``MISMATCH`` (the venue's order disagrees with the published clientOid, symbol or side) or
``ERROR`` (the read failed), and a summary.

Exit codes: 0 every published order was found and matches; 1 an order is missing or disagrees;
2 the check could not run (no key, no CLI, unreadable file, or a read failed); 3 the key is not a
Demo key (40099).

The file may be a JSON list of orders or an object with an ``orders`` list. Each order needs
``venue_order_id`` (an order the venue never acknowledged has ``null`` and is reported as not sent);
``client_oid``, ``symbol`` and ``side`` are compared when present.
"""

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sentiment_agent.execution.bgc import (
    BgcRunner,
    BgcTimeoutError,
    BgcUnavailableError,
    SubprocessBgcRunner,
    VenueParseError,
    detail_by_order_id_args,
    is_order_not_found,
    parse_venue_order,
)
from sentiment_agent.execution.environment import (
    AGENT_HUB_DIR,
    PAPER_FLAG,
    READ_ONLY_FLAG,
    EnvironmentRefused,
    failure_of,
    load_demo_credentials,
    result_mentions_environment_mismatch,
    vendor_contract,
)

EXIT_OK = 0
EXIT_MISMATCH = 1
EXIT_CANNOT_RUN = 2
EXIT_NOT_DEMO = 3
TIMEOUT_S = 90.0


@dataclass(frozen=True, slots=True)
class Check:
    venue_order_id: str | None
    client_oid: str | None
    outcome: str
    """``FOUND``, ``MISSING``, ``MISMATCH``, ``ERROR`` or ``NOT_SENT``."""
    detail: str


def load_orders(path: Path) -> list[Mapping[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("orders") if isinstance(data, Mapping) else data
    if not isinstance(rows, list) or not all(isinstance(r, Mapping) for r in rows):
        raise ValueError(f"{path} is neither a list of orders nor an object with an orders list")
    return rows


def _published(order: Mapping[str, Any], key: str) -> str | None:
    value = order.get(key)
    return None if value in (None, "") else str(value)


def check_order(order: Mapping[str, Any], *, runner: BgcRunner, env: Mapping[str, str]) -> Check:
    """Read one published order back from the Demo venue. Raises EnvironmentRefused on 40099."""
    venue_id = _published(order, "venue_order_id")
    client_oid = _published(order, "client_oid")
    if venue_id is None:
        return Check(None, client_oid, "NOT_SENT", "no venue orderId was published")
    try:
        args = detail_by_order_id_args(venue_id)
    except ValueError as exc:
        return Check(venue_id, client_oid, "ERROR", str(exc))
    if (
        PAPER_FLAG not in args
        or READ_ONLY_FLAG in args
        or args[:3]
        != [
            "order",
            "--action",
            "detail",
        ]
    ):
        raise RuntimeError("refusing an argv that is not a Demo order read")
    try:
        result = runner(args, env=env, timeout_s=TIMEOUT_S)
    except (BgcTimeoutError, OSError) as exc:
        return Check(venue_id, client_oid, "ERROR", f"read did not complete: {exc}")
    if result_mentions_environment_mismatch(result):
        raise EnvironmentRefused("40099: the key in .secrets/demo.env is not a Demo key")
    failure = failure_of(result)
    if failure is not None:
        if is_order_not_found(failure):
            return Check(venue_id, client_oid, "MISSING", failure.message)
        return Check(venue_id, client_oid, "ERROR", f"{failure.type}: {failure.message}")
    data = (result.stdout or {}).get("data")
    if not isinstance(data, Mapping) or not data.get("orderId"):
        return Check(venue_id, client_oid, "MISSING", "the venue returned no order")
    try:
        venue = parse_venue_order(data, blob=None)
    except VenueParseError as exc:
        return Check(venue_id, client_oid, "ERROR", f"unreadable order: {exc}")
    problems: list[str] = []
    if venue.venue_order_id != venue_id:
        problems.append(f"orderId {venue.venue_order_id}")
    if client_oid is not None and venue.client_oid != client_oid:
        problems.append(f"clientOid {venue.client_oid} (published {client_oid})")
    symbol = _published(order, "symbol")
    if symbol is not None and venue.symbol != symbol:
        problems.append(f"symbol {venue.symbol} (published {symbol})")
    side = _published(order, "side")
    if side is not None and venue.side.value != side.lower():
        problems.append(f"side {venue.side} (published {side})")
    summary = (
        f"{venue.symbol} {venue.side} qty {venue.qty} executed {venue.cum_exec_qty} "
        f"status {venue.status}"
    )
    if problems:
        return Check(venue_id, client_oid, "MISMATCH", f"{summary}; differs: {', '.join(problems)}")
    return Check(venue_id, client_oid, "FOUND", summary)


def run(
    orders: Sequence[Mapping[str, Any]], *, runner: BgcRunner, env: Mapping[str, str]
) -> tuple[list[Check], int]:
    checks = [check_order(order, runner=runner, env=env) for order in orders]
    outcomes = {c.outcome for c in checks}
    if "ERROR" in outcomes:
        return checks, EXIT_CANNOT_RUN
    if outcomes & {"MISSING", "MISMATCH"}:
        return checks, EXIT_MISMATCH
    return checks, EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--public", type=Path, default=Path("public/orders.json"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", action="store_true", help="print a JSON report")
    args = parser.parse_args(argv)

    try:
        orders = load_orders(args.public)
    except (OSError, ValueError) as exc:
        print(f"cannot read {args.public}: {exc}", file=sys.stderr)
        return EXIT_CANNOT_RUN
    contract = vendor_contract(args.root / AGENT_HUB_DIR)
    if not contract.ok:
        for finding in contract.findings:
            print(f"Agent Hub check failed: {finding}", file=sys.stderr)
        return EXIT_CANNOT_RUN
    try:
        credentials = load_demo_credentials(args.root)
        runner = SubprocessBgcRunner(args.root / AGENT_HUB_DIR)
    except (EnvironmentRefused, BgcUnavailableError) as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_CANNOT_RUN

    try:
        checks, code = run(orders, runner=runner, env=credentials.child_env())
    except EnvironmentRefused as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_NOT_DEMO

    if args.json:
        print(json.dumps({"exit_code": code, "checks": [asdict(c) for c in checks]}, indent=2))
    else:
        for c in checks:
            print(f"{c.outcome:9} {c.venue_order_id or '-':22} {c.client_oid or '-':34} {c.detail}")
        counts = {
            name: sum(c.outcome == name for c in checks)
            for name in sorted({c.outcome for c in checks})
        }
        print(
            f"{len(checks)} published orders: " + ", ".join(f"{v} {k}" for k, v in counts.items())
        )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
