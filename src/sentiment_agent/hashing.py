"""Canonical JSON and SHA-256: the one definition of "the same content" in this project.

Three things hash content and must agree byte for byte: the ledger chain (`ledger/chain.py`), the
genesis pre-registration (`ledger/genesis.py`) and the order idempotency key (`clientOid`,
`kernel/planner.py`). If each serialised on its own, two identical decisions could hash differently
and a retried order would reach the venue under a new id. So they all call this module, and
`scripts/recompute.py` reimplements exactly the rules below with the standard library.

Canonical form, value by value:

* A pydantic model: its ``model_dump(mode="json")``.
* ``Decimal``: ``str(value)``, exactly as written (``Decimal("0.10")`` stays ``"0.10"``, and is a
  different value from ``"0.1"``). Producers quantize before hashing when they mean the same number.
* ``datetime``: timezone-aware only (a naive one is refused), converted to UTC and written as
  pydantic writes it in JSON mode, ``2026-09-23T13:00:00Z`` (microseconds only when non-zero), so a
  bare datetime and the same datetime inside a model hash alike. ``date``: ``YYYY-MM-DD``.
* ``Enum``: its value. Mapping keys: strings, enum values, or other scalars as ``str(key)``; two
  keys that collapse to one string are refused.
* ``list`` and ``tuple``: JSON arrays. ``set`` and ``frozenset`` are refused, having no order.
* ``float``: as ``json.dumps`` writes it (shortest round-trip repr). ``NaN`` and infinity are
  refused: a non-finite float in a hashed record is a defect upstream, not something to encode.
  ``-0.0`` is written ``-0.0`` and is not normalised.

Then ``json.dumps`` with sorted keys, no whitespace, ``ensure_ascii=False``, UTF-8.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel
from pydantic_core import to_jsonable_python

HASH_HEX_LENGTH = 64
ZERO_HASH = "0" * HASH_HEX_LENGTH
"""``prev_hash`` of the genesis event."""


def _key(key: Any) -> str:
    if isinstance(key, Enum):
        return str(key.value)
    return str(key)


def _plain(value: Any) -> Any:
    """Reduce a value to JSON-native types, deterministically."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("naive datetimes cannot be hashed; use timezone-aware UTC")
        return to_jsonable_python(value.astimezone(UTC))
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, timedelta):
        raise TypeError("timedelta has no canonical form here; hash its total seconds instead")
    if isinstance(value, (set, frozenset)):
        raise TypeError("sets have no order and cannot be hashed canonically; sort them first")
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for k, v in value.items():
            key = _key(k)
            if key in out:
                raise ValueError(f"two mapping keys canonicalise to the same string {key!r}")
            out[key] = _plain(v)
        return out
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(v) for v in value]
    return value


def canonical_json(value: Any) -> bytes:
    """The canonical byte form of ``value``. Raises ``ValueError`` on NaN or infinity."""
    return json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    """Full 64-character hex digest. Never truncated for chain links."""
    return hashlib.sha256(data).hexdigest()


def content_hash(value: Any) -> str:
    """``sha256_hex(canonical_json(value))``."""
    return sha256_hex(canonical_json(value))


__all__ = ["HASH_HEX_LENGTH", "ZERO_HASH", "canonical_json", "content_hash", "sha256_hex"]
