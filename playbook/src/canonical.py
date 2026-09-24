"""Canonical JSON, so the replica's records can be hashed outside the sandbox.

The Playbook sandbox has no ``hashlib`` (``references/sandbox-runtime.md``, allowed standard
library). The primary agent derives every intent's identity from ``sha256(canonical_json(core))``
(``sentiment_agent.hashing``, ``kernel/planner.py``). The replica cannot compute that digest, so it
emits the canonical *string* of every intent, decision and ruling in its signal output, and the
digest is taken outside: ``python scripts/build_playbook.py --hash-signals signal.json`` in the
primary repository. For the digest outside to be the primary's own ``content_hash`` of the same
record, the bytes must be the primary's canonical form, which for JSON-native values is exactly
this: sorted keys, no whitespace, ``ensure_ascii=False``, UTF-8, NaN and infinity refused. Values
are reduced first the way ``sentiment_agent.hashing._plain`` reduces them (a ``Decimal`` is its
string, a datetime its ISO form in UTC with ``Z``, a tuple a list, a set refused).
"""

import json
import math
from collections.abc import Mapping
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import Enum


def plain(value: object) -> object:
    """``value`` reduced to JSON-native types, deterministically."""
    if isinstance(value, Enum):
        return plain(value.value)
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("a non-finite Decimal cannot be canonicalised")
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("naive datetimes cannot be canonicalised; use timezone-aware UTC")
        text = value.astimezone(UTC).isoformat()
        return text[: -len("+00:00")] + "Z" if text.endswith("+00:00") else text
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (set, frozenset)):
        raise TypeError("sets have no order and cannot be canonicalised; sort them first")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("NaN and infinity cannot be canonicalised")
    if isinstance(value, Mapping):
        out: dict[str, object] = {}
        for key, item in value.items():
            name = str(key.value) if isinstance(key, Enum) else str(key)
            if name in out:
                raise ValueError(f"two keys canonicalise to the same string {name!r}")
            out[name] = plain(item)
        return out
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def canonical(value: object) -> str:
    """The canonical JSON text of ``value``; ``canonical(v).encode('utf-8')`` is what is hashed."""
    return json.dumps(
        plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def compact(value: object) -> object:
    """``value`` made JSON-safe for signal output: canonical types, non-finite floats as ``None``.

    Used for the human-readable parts of a signal. Anything that will be hashed goes through
    :func:`canonical` instead, which refuses what this function silently cleans.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Decimal) and not value.is_finite():
        return None
    if isinstance(value, datetime) and (value.tzinfo is None or value.utcoffset() is None):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(k.value) if isinstance(k, Enum) else str(k): compact(v) for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [compact(v) for v in value]
    if isinstance(value, (Decimal, datetime, date, Enum)):
        return plain(value)
    return value
