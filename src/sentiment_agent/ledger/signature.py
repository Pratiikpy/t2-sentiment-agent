"""Signed ledger heads: who wrote the record, checkable with plain Python and nothing installed.

The OpenTimestamps anchors (``ledger/anchor.py``) prove *when* a head hash existed. They do not
prove *who* wrote it: anyone holding the ledger file could have stamped a rewritten chain the
same day. A signature closes that. Every exported head (genesis hash, sequence number, event
hash) is signed with the operator's Ed25519 key; the public key is committed in this repository
(``keys/ledger_signing.pub``), so a key swapped after the fact is visible in git history. A reader
then checks two things against the published ``ledger.jsonl``: each signature verifies under that
key, and each signed hash is the hash the chain actually has at that sequence number.

Following Kaaval's signed, separately published record (an S2 Agentic Trading entry,
``research/s2-field/projects/40_kaaval.md``), with one deliberate difference: Kaaval signs every
event at write time, which changes the event format, so it can only start with a new run. Signing
heads at export leaves the record's bytes untouched, applies to a run already in progress, and still
binds every earlier event, because each head hash chains over everything before it. What it cannot
prove is that a head was signed *before* a given time — that is the anchors' job, and the two are
published side by side.

**The Ed25519 implementation** is the reference code of RFC 8032 section 6 (IETF Trust, Simplified
BSD licence), kept in the RFC's own structure so it can be read against the standard line by line.
It is slow (pure-Python big integers, tens of milliseconds a signature) and not constant-time, which
is acceptable for signing a few hundred heads on the operator's own machine and for verifying them
anywhere; it is not a general-purpose signing library. It is checked against the RFC's test vectors
(``tests/ledger/test_signature.py``). The project keeps its two runtime dependencies (pyproject.toml
line 19), so no cryptography package is added.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

# --- RFC 8032 section 6, reference implementation --------------------------------------------

_P: Final = 2**255 - 19
_Q: Final = 2**252 + 27742317777372353535851937790883648493

Point = tuple[int, int, int, int]


def _sha512(data: bytes) -> bytes:
    return hashlib.sha512(data).digest()


def _modp_inv(x: int) -> int:
    return pow(x, _P - 2, _P)


_D: Final = -121665 * _modp_inv(121666) % _P


def _sha512_modq(data: bytes) -> int:
    return int.from_bytes(_sha512(data), "little") % _Q


def _point_add(p1: Point, p2: Point) -> Point:
    a = (p1[1] - p1[0]) * (p2[1] - p2[0]) % _P
    b = (p1[1] + p1[0]) * (p2[1] + p2[0]) % _P
    c = 2 * p1[3] * p2[3] * _D % _P
    d = 2 * p1[2] * p2[2] % _P
    e, f, g, h = b - a, d - c, d + c, b + a
    return (e * f, g * h, f * g, e * h)


def _point_mul(s: int, point: Point) -> Point:
    acc: Point = (0, 1, 1, 0)
    while s > 0:
        if s & 1:
            acc = _point_add(acc, point)
        point = _point_add(point, point)
        s >>= 1
    return acc


def _point_equal(p1: Point, p2: Point) -> bool:
    if (p1[0] * p2[2] - p2[0] * p1[2]) % _P != 0:
        return False
    return (p1[1] * p2[2] - p2[1] * p1[2]) % _P == 0


_MODP_SQRT_M1: Final = pow(2, (_P - 1) // 4, _P)


def _recover_x(y: int, sign: int) -> int | None:
    if y >= _P:
        return None
    x2 = (y * y - 1) * _modp_inv(_D * y * y + 1)
    if x2 == 0:
        return None if sign else 0
    x = pow(x2, (_P + 3) // 8, _P)
    if (x * x - x2) % _P != 0:
        x = x * _MODP_SQRT_M1 % _P
    if (x * x - x2) % _P != 0:
        return None
    if (x & 1) != sign:
        x = _P - x
    return x


_G_Y: Final = 4 * _modp_inv(5) % _P
_G_X_OR_NONE: Final = _recover_x(_G_Y, 0)
if _G_X_OR_NONE is None:  # pragma: no cover - the base point is fixed by the standard
    raise RuntimeError("the Ed25519 base point did not decode")
_G_X: Final = _G_X_OR_NONE
_G: Final[Point] = (_G_X, _G_Y, 1, _G_X * _G_Y % _P)


def _point_compress(point: Point) -> bytes:
    zinv = _modp_inv(point[2])
    x = point[0] * zinv % _P
    y = point[1] * zinv % _P
    return int.to_bytes(y | ((x & 1) << 255), 32, "little")


def _point_decompress(data: bytes) -> Point | None:
    if len(data) != 32:
        raise ValueError("an Ed25519 point is 32 bytes")
    y = int.from_bytes(data, "little")
    sign = y >> 255
    y &= (1 << 255) - 1
    x = _recover_x(y, sign)
    if x is None:
        return None
    return (x, y, 1, x * y % _P)


def _secret_expand(secret: bytes) -> tuple[int, bytes]:
    if len(secret) != 32:
        raise ValueError("an Ed25519 secret key is 32 bytes")
    h = _sha512(secret)
    a = int.from_bytes(h[:32], "little")
    a &= (1 << 254) - 8
    a |= 1 << 254
    return a, h[32:]


def public_key(secret: bytes) -> bytes:
    """The 32-byte public key of a 32-byte secret (RFC 8032 5.1.5)."""
    a, _prefix = _secret_expand(secret)
    return _point_compress(_point_mul(a, _G))


def sign(secret: bytes, message: bytes) -> bytes:
    """The 64-byte Ed25519 signature of ``message`` (RFC 8032 5.1.6)."""
    a, prefix = _secret_expand(secret)
    key = _point_compress(_point_mul(a, _G))
    r = _sha512_modq(prefix + message)
    big_r = _point_compress(_point_mul(r, _G))
    h = _sha512_modq(big_r + key + message)
    s = (r + h * a) % _Q
    return big_r + int.to_bytes(s, 32, "little")


def verify(key: bytes, message: bytes, signature: bytes) -> bool:
    """Whether ``signature`` is a valid Ed25519 signature of ``message`` under ``key``
    (RFC 8032 5.1.7). Malformed input is a failed verification, never an exception."""
    if len(key) != 32 or len(signature) != 64:
        return False
    point_a = _point_decompress(key)
    if point_a is None:
        return False
    point_r = _point_decompress(signature[:32])
    if point_r is None:
        return False
    s = int.from_bytes(signature[32:], "little")
    if s >= _Q:
        return False
    h = _sha512_modq(signature[:32] + key + message)
    return _point_equal(_point_mul(s, _G), _point_add(point_r, _point_mul(h, point_a)))


# --- signed heads -----------------------------------------------------------------------------

DOMAIN: Final = b"t2sa-ledger-head-v1"
"""Prefixed to every signed message, so a head signature cannot be replayed as anything else."""


def head_message(genesis_hash: str, seq: int, event_hash: str) -> bytes:
    """The exact bytes a head signature covers."""
    if seq < 0:
        raise ValueError("a sequence number cannot be negative")
    return b"\n".join([DOMAIN, genesis_hash.encode(), str(seq).encode(), event_hash.encode()])


@dataclass(frozen=True, slots=True)
class SignedHead:
    seq: int
    hash: str
    signed_at: str
    signature: str

    def as_json(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "hash": self.hash,
            "signed_at": self.signed_at,
            "signature": self.signature,
        }


def sign_head(
    secret: bytes, genesis_hash: str, seq: int, event_hash: str, signed_at: str
) -> SignedHead:
    signature = sign(secret, head_message(genesis_hash, seq, event_hash))
    return SignedHead(seq=seq, hash=event_hash, signed_at=signed_at, signature=signature.hex())


@dataclass(frozen=True, slots=True)
class HeadCheck:
    seq: int
    signature_valid: bool
    matches_chain: bool | None
    """``None`` when the ledger handed to the check does not reach that sequence number."""

    @property
    def ok(self) -> bool:
        return self.signature_valid and self.matches_chain is True


def verify_heads(document: Mapping[str, Any], chain_hashes: Mapping[int, str]) -> list[HeadCheck]:
    """Check every head in a ``signed_heads.json`` document against its own key and against the
    ledger's actual hash at that sequence number (``chain_hashes``: seq -> event hash)."""
    key = bytes.fromhex(str(document["public_key"]))
    genesis = str(document["genesis_hash"])
    out: list[HeadCheck] = []
    for head in document["heads"]:
        seq = int(head["seq"])
        try:
            signature = bytes.fromhex(str(head["signature"]))
        except ValueError:
            signature = b""
        valid = verify(key, head_message(genesis, seq, str(head["hash"])), signature)
        actual = chain_hashes.get(seq)
        out.append(
            HeadCheck(
                seq=seq,
                signature_valid=valid,
                matches_chain=None if actual is None else actual == head["hash"],
            )
        )
    return out


def chain_hashes_from_jsonl(text: str) -> dict[int, str]:
    """seq -> hash from the text of a ``ledger.jsonl``. Lines are split on ``\\n`` only: an event
    can carry U+2028 or U+0085 inside a JSON string (quoted crowd text), which
    :meth:`str.splitlines` would cut an event at. A line that does not parse (a write in progress
    at the end of a live file) is skipped rather than guessed at."""
    out: dict[int, str] = {}
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        out[int(event["seq"])] = str(event["hash"])
    return out


def signed_heads_document(
    public: bytes, genesis_hash: str, heads: Sequence[SignedHead]
) -> dict[str, Any]:
    return {
        "scheme": "ed25519 over t2sa-ledger-head-v1\\n<genesis>\\n<seq>\\n<hash>",
        "public_key": public.hex(),
        "genesis_hash": genesis_hash,
        "heads": [h.as_json() for h in heads],
    }


__all__ = [
    "DOMAIN",
    "HeadCheck",
    "SignedHead",
    "chain_hashes_from_jsonl",
    "head_message",
    "public_key",
    "sign",
    "sign_head",
    "signed_heads_document",
    "verify",
    "verify_heads",
]
