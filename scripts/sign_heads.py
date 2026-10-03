"""Sign a paper record's heads with the operator's key (the operator's side of verify_heads.py).

    python scripts/sign_heads.py <record> <heads file>
    python scripts/sign_heads.py --new-key

``<record>`` is the published page (``https://...``) or a local export directory; ``genesis.json``
and ``ledger.jsonl`` are read from it and nothing in it is written. The heads go to ``<heads file>``
(committed in this repository, e.g. ``validation/run2/signed_heads.json``), so signing a run in
progress never touches that run's root or its page.

Heads signed: every Bitcoin-anchored sequence number in ``genesis.json`` and the newest event in
``ledger.jsonl``. Heads already in the file are kept as they were signed; only new sequence numbers
are added. The secret lives in ``.secrets/ledger_signing.key`` (gitignored); the public key in
``keys/ledger_signing.pub`` is committed, which is what a reader checks against.
"""

from __future__ import annotations

import json
import secrets
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentiment_agent.ledger.signature import (  # noqa: E402 - path set just above
    SignedHead,
    chain_hashes_from_jsonl,
    public_key,
    sign_head,
    signed_heads_document,
)

SECRET = ROOT / ".secrets" / "ledger_signing.key"
PUBLIC = ROOT / "keys" / "ledger_signing.pub"


def read_record_file(where: str, name: str) -> str:
    """``name`` from a published page or a local export directory."""
    if where.startswith(("https://", "http://")):
        url = f"{where.rstrip('/')}/{name}"
        with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 - operator's URL
            return str(response.read().decode("utf-8"))
    return (Path(where) / name).read_text(encoding="utf-8")


def _new_key() -> int:
    if SECRET.exists():
        print(f"a key already exists at {SECRET}; refusing to replace it")
        return 1
    SECRET.parent.mkdir(parents=True, exist_ok=True)
    secret = secrets.token_bytes(32)
    SECRET.write_text(secret.hex() + "\n", encoding="utf-8")
    PUBLIC.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC.write_text(public_key(secret).hex() + "\n", encoding="utf-8")
    print(f"public key {public_key(secret).hex()} written to {PUBLIC}; commit it")
    return 0


def main(argv: list[str]) -> int:
    if argv == ["--new-key"]:
        return _new_key()
    if len(argv) != 2:
        print(__doc__)
        return 2
    where, target = argv[0], Path(argv[1])
    secret = bytes.fromhex(SECRET.read_text(encoding="utf-8").strip())
    genesis_doc = json.loads(read_record_file(where, "genesis.json"))
    genesis = str(genesis_doc["hash"])
    chain = chain_hashes_from_jsonl(read_record_file(where, "ledger.jsonl"))
    wanted = {int(a["seq"]) for a in genesis_doc.get("anchors") or []}
    if chain:
        wanted.add(max(chain))
    kept: list[SignedHead] = []
    if target.exists():
        old = json.loads(target.read_text(encoding="utf-8"))
        if old["public_key"] != public_key(secret).hex() or old["genesis_hash"] != genesis:
            print(f"{target} was made with another key or for another genesis; refusing")
            return 1
        kept = [
            SignedHead(int(h["seq"]), h["hash"], h["signed_at"], h["signature"])
            for h in old["heads"]
        ]
    have = {h.seq for h in kept}
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    new = [
        sign_head(secret, genesis, seq, chain[seq], now)
        for seq in sorted(wanted - have)
        if seq in chain
    ]
    heads = sorted([*kept, *new], key=lambda h: h.seq)
    document = signed_heads_document(public_key(secret), genesis, heads)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    print(f"{len(new)} new head(s) signed, {len(heads)} in all -> {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
