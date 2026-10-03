"""Verify a paper record's signed heads, with plain Python and nothing installed.

    python scripts/verify_heads.py <record> [<heads file>]

``<record>`` is the published page (``https://...``) or a local export directory; its
``ledger.jsonl`` is the chain checked against. ``<heads file>`` defaults to ``signed_heads.json``
beside the record; the committed run files are ``validation/run<N>/signed_heads.json``. Every
signature is checked against the public key in the heads file **and** against the key committed in
this repository (``keys/ledger_signing.pub``), and each signed hash against the hash the chain has
at that sequence number. Exit code 0 only when every head passes.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentiment_agent.ledger.signature import (  # noqa: E402 - path set just above
    chain_hashes_from_jsonl,
    verify_heads,
)

COMMITTED_KEY = ROOT / "keys" / "ledger_signing.pub"


def _read(where: str, name: str) -> str:
    if where.startswith(("https://", "http://")):
        url = f"{where.rstrip('/')}/{name}"
        with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 - reader's URL
            return str(response.read().decode("utf-8"))
    return (Path(where) / name).read_text(encoding="utf-8")


def main(argv: list[str]) -> int:
    if len(argv) not in (1, 2):
        print(__doc__)
        return 2
    where = argv[0]
    heads_text = (
        Path(argv[1]).read_text(encoding="utf-8")
        if len(argv) == 2
        else _read(where, "signed_heads.json")
    )
    document = json.loads(heads_text)
    chain = chain_hashes_from_jsonl(_read(where, "ledger.jsonl"))
    committed = (
        COMMITTED_KEY.read_text(encoding="utf-8").strip() if COMMITTED_KEY.exists() else None
    )
    key_ok = committed is not None and committed == document["public_key"]
    print(f"public key in the heads file: {document['public_key']}")
    print(
        f"key committed in this repository: {committed or 'MISSING'} -> "
        f"{'same key' if key_ok else 'DIFFERENT OR MISSING'}"
    )
    print(f"chain read: {len(chain)} events, last seq {max(chain) if chain else 'none'}")
    checks = verify_heads(document, chain)
    for check in checks:
        state = (
            "PASS"
            if check.ok
            else "FAIL signature"
            if not check.signature_valid
            else "FAIL hash differs from the chain"
            if check.matches_chain is False
            else "NOT CHECKED: the published chain does not reach this head"
        )
        print(f"seq {check.seq}: {state}")
    passed = sum(1 for c in checks if c.ok)
    print(f"{passed} of {len(checks)} signed heads verify against key and chain")
    return 0 if key_ok and checks and passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
