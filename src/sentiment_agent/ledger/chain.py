"""The append-only, hash-chained event log. Everything a judge reads is computed from it.

One JSON line per :class:`~sentiment_agent.types.LedgerEvent`, one file per
:class:`~sentiment_agent.types.RunMode` (``var/ledger/{simulated,dryrun,paper}.jsonl``, DESIGN.md
§5, §12). Each event carries ``hash = sha256(canonical({seq, ts, kind, mode, payload, blobs,
prev_hash}))`` over the full event, with ``prev_hash`` the previous event's hash (the zero hash at
seq 0), full 64-hex digests throughout. ``canonical`` is
:func:`sentiment_agent.hashing.canonical_json`.

Properties, and where each is enforced:

* **Nothing is ever rewritten.** The only write is an append. An outcome (a fill, a mark, an
  anchor upgrade) is a new event, never an edit of an old one, so the hash covers every field of
  every event and there is no unhashed field to protect separately.
* **Every line is canonical.** A line is exactly ``canonical_json(event)``: sorted keys, no
  whitespace, values as the writer emits them. So a verifier with only the standard library can
  recompute each hash from the parsed line (``scripts/recompute.py`` does), and a hand edit that
  respells a value without changing it is still an edit.
* **One writer at a time.** An ``O_CREAT | O_EXCL`` lock file beside the log, and the head is
  re-read from disk *inside* the lock, so two writers can never compute the same ``seq`` from a
  stale head. A lock left by a writer that died is broken after :data:`LOCK_STALE_SECONDS`, under a
  second lock so two waiters cannot both break it, and the breaking is reported.
* **Truncation is evident.** After every append a sidecar anchor (``<name>.head``, atomic replace)
  records the event count and head hash. :meth:`HashChainLedger.verify` flags a log shorter than its
  anchor, or one whose anchored head is no longer there (a replaced tail), and :meth:`append`
  refuses to write over either: extending a truncated log would rewrite the anchor and erase the
  evidence.
* **Typed on write and on read.** :meth:`append` accepts only the payload model registered for the
  kind (:data:`~sentiment_agent.types.EVENT_PAYLOADS`), and every line read back must parse as that
  model and re-serialise to exactly the stored payload.
* **Pre-registration rules.** A ``GENESIS`` event may only be seq 0, so there is at most one. An
  ``AMENDMENT`` needs a genesis before it, the owner's confirmation, and must amend the policy that
  is active at that point. The same rules are applied when the log is read, so a log that breaks
  them does not verify. (Whether the scored PAPER log *starts* with a genesis is checked where it
  matters, before an order: :func:`sentiment_agent.ledger.genesis.require_genesis`. Pre-genesis
  records such as the plumbing test belong in a separate ledger file, DESIGN.md §16.)
* **Time never runs backwards in the log.** An event takes the clock's time, or its predecessor's
  if the clock reads earlier (a stepped system clock); the verifier treats a decreasing ``ts`` as a
  break, since this writer never produces one.
* **Modes never mix.** Every event carries the ledger's mode; a file named after a mode
  (``paper.jsonl``) can only be opened as that mode, and a line of another mode is a break.

Provenance
----------
The lock and anchor patterns are ported from ARGUS ``src/argus/paper/ledger.py`` (MIT, Copyright (c)
2026 Pratiikpy, the same author as this project) at commit
``3dec6baf9dfa7be37c7b452e26a9b139df252f75``, sha256
``6ec8723130b6fd5a54aab1c669985135c168a3f24f217c7bf9bfc387998284be``. The ARGUS repository is
private, so its licence and every pin are recorded in ``third_party/argus/PROVENANCE.md``. Copied
and reworked, never imported. Taken:

* the ``O_CREAT | O_EXCL`` lock with the holder written into it, a bounded wait, and stale-lock
  breaking (``_exclusive``, ledger.py:77-133), added in ARGUS after two cycles both wrote seq 41;
* building the event inside the lock against the head on disk (``_append_built``, :344-385);
* the head anchor sidecar written by atomic replace (``_write_anchor``, :653-682; ARGUS took that
  from ``serenity-guardrails`` ``etoro_trading/journal.py:21-60``, Apache-2.0);
* verification that reports the first break and a removed or replaced tail (``verify``, :692-760).

Changed, each for a defect in the ARGUS original:

* ARGUS hashed a subset of fields to 16 hex characters and had to add a separate settlement seal
  when direct file edits of unhashed outcome fields went undetected. Here the hash covers the whole
  event at full length and outcomes are new events, so there is nothing unhashed.
* ARGUS rewrote the whole file to attach outcomes (``_persist_settlement``). Nothing here rewrites.
* ARGUS judged its anchor only when the log was not longer than it, so a replaced tail with one
  more event appended read as "anchor agrees". Here the anchored head must be present at its seq
  whatever the log's length.
* ARGUS broke a stale lock by deleting it directly, which races a second waiter into deleting the
  fresh lock of a third writer. Here breaking happens under its own ``O_EXCL`` lock and only after
  re-reading that the same holder is still there; a holder releases only a lock that still carries
  its own token, and re-checks that before it writes.
* ARGUS kept appending after its own ``verify`` found a break. Here a writer refuses to extend a
  log whose tail is torn, whose last line changed under it, or whose anchor says it was truncated.
"""

import contextlib
import copy
import json
import logging
import os
import re
import secrets
import socket
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, TypeVar

from pydantic import BaseModel, Field, ValidationError

from sentiment_agent.hashing import ZERO_HASH, canonical_json, sha256_hex
from sentiment_agent.ledger.blobs import FileBlobStore
from sentiment_agent.types import (
    EVENT_PAYLOADS,
    Amendment,
    BlobRef,
    Clock,
    EventKind,
    Genesis,
    LedgerEvent,
    Model,
    RunMode,
)

LOCK_WAIT_SECONDS: float = 30.0
"""How long a writer waits for the lock before :class:`LedgerLocked`. An append holds the lock for
the time it takes to write and fsync one line and the anchor (milliseconds), so a wait this long
means the holder is stuck, not busy. Read at call time, so a test may lower it."""

LOCK_POLL_SECONDS: float = 0.02

LOCK_STALE_SECONDS: float = 120.0
"""Age (file modification time) after which a lock belongs to a writer that died. Two orders of
magnitude above the longest append, so a live writer is never judged dead. Lock age is measured
against the file system's own clock (``time.time()`` against ``st_mtime``); it is the one place this
module reads the wall clock, because a lock file's age is a property of the file system, not of the
trading clock that stamps events."""

ANCHOR_SUFFIX: Final = ".head"
LOCK_SUFFIX: Final = ".lock"
BREAK_SUFFIX: Final = ".lock.break"

_REPLACE_ATTEMPTS: Final = 20
_RETRY_BACKOFF_S: Final = 0.01
_DIGEST: Final = re.compile(r"^[0-9a-f]{64}$")

_log = logging.getLogger(__name__)
_T = TypeVar("_T")


class LedgerError(RuntimeError):
    """The ledger was asked to do something that would make it untrustworthy."""


class LedgerLocked(LedgerError):  # noqa: N818 - the name is the module contract (DESIGN M10)
    """Another writer holds the ledger. Better refused than written beside it."""


class GenesisError(LedgerError):
    """The pre-registration is missing, misplaced, or does not match what is running."""


class _BrokenLineError(Exception):
    """A line that does not verify. Internal: converted to a report or a :class:`LedgerError`."""


# ================================================================================================
# Hashing
# ================================================================================================


def _preimage(
    *,
    seq: int,
    ts: datetime,
    kind: EventKind,
    mode: RunMode,
    payload: Mapping[str, Any],
    blobs: Sequence[BlobRef],
    prev_hash: str,
) -> bytes:
    if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
        raise ValueError(f"seq must be a non-negative integer, not {seq!r}")
    if not isinstance(ts, datetime) or ts.tzinfo is None or ts.utcoffset() is None:
        raise ValueError("an event's ts must be a timezone-aware datetime")
    if not isinstance(prev_hash, str) or not _DIGEST.fullmatch(prev_hash):
        raise ValueError("prev_hash must be a full 64-character lowercase hex digest")
    return canonical_json(
        {
            "seq": seq,
            "ts": ts,
            "kind": EventKind(kind),
            "mode": RunMode(mode),
            "payload": payload,
            "blobs": list(blobs),
            "prev_hash": prev_hash,
        }
    )


def event_hash(
    *,
    seq: int,
    ts: datetime,
    kind: EventKind,
    mode: RunMode,
    payload: Mapping[str, Any],
    blobs: Sequence[BlobRef],
    prev_hash: str,
) -> str:
    """``sha256(canonical_json({seq, ts, kind, mode, payload, blobs, prev_hash}))``, full hex.

    ``ts`` is written as pydantic writes it in JSON mode (``2026-09-23T13:00:00Z``), ``kind`` and
    ``mode`` as their values, ``payload`` as the JSON-mode dump of the payload model, and each blob
    reference as ``{"media_type", "sha256", "size"}``. Because every line of the log is canonical,
    the same digest is reached from a parsed line with the standard library alone:
    ``sha256(json.dumps({k: line[k] for k in those seven}, sort_keys=True, separators=(",", ":"),
    ensure_ascii=False).encode())``.
    """
    return sha256_hex(
        _preimage(
            seq=seq,
            ts=ts,
            kind=kind,
            mode=mode,
            payload=payload,
            blobs=blobs,
            prev_hash=prev_hash,
        )
    )


def event_preimage(event: LedgerEvent) -> bytes:
    """The exact bytes whose SHA-256 is ``event.hash`` (when the event is intact).

    Timestamping this file with OpenTimestamps commits to the event hash itself
    (:mod:`sentiment_agent.ledger.anchor`).
    """
    return _preimage(
        seq=event.seq,
        ts=event.ts,
        kind=event.kind,
        mode=event.mode,
        payload=event.payload,
        blobs=event.blobs,
        prev_hash=event.prev_hash,
    )


def _self_consistent(event: LedgerEvent) -> bool:
    return sha256_hex(event_preimage(event)) == event.hash


# ================================================================================================
# Payloads and blob references
# ================================================================================================


def _first_error(exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return str(exc)
    first = errors[0]
    where = ".".join(str(part) for part in first.get("loc", ())) or "(root)"
    return f"{where}: {first.get('msg', 'invalid')}"


def _typed_payload(kind: EventKind, payload: Mapping[str, Any]) -> Model:
    """The payload as its registered model; :class:`_BrokenLineError` if it is not exactly one."""
    model_type = EVENT_PAYLOADS[kind]
    try:
        model = model_type.model_validate(payload)
    except ValidationError as exc:
        raise _BrokenLineError(
            f"the payload is not a valid {model_type.__name__}: {_first_error(exc)}"
        ) from None
    if model.model_dump(mode="json") != payload:
        raise _BrokenLineError(
            f"the payload is not in the form a {model_type.__name__} is written in"
        )
    return model


def _walk_refs(value: object) -> Iterator[BlobRef]:
    if isinstance(value, BlobRef):
        yield value
    elif isinstance(value, BaseModel):
        for name in type(value).model_fields:
            yield from _walk_refs(getattr(value, name))
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _walk_refs(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_refs(item)


def referenced_blobs(event: LedgerEvent) -> tuple[BlobRef, ...]:
    """Every blob an event commits to: its ``blobs`` and every reference inside its payload.

    A source call's raw response, an LLM call's request and completion, a dry-run preview's raw
    output and an anchor's proof are all :class:`~sentiment_agent.types.BlobRef` fields nested in
    payloads; the chain commits to them through the payload hash. First occurrence of each digest,
    in order. Raises :class:`LedgerError` if the payload does not parse as its registered model.
    """
    seen: dict[str, BlobRef] = {}
    for ref in event.blobs:
        seen.setdefault(ref.sha256, ref)
    try:
        model = _typed_payload(event.kind, event.payload)
    except _BrokenLineError as exc:
        raise LedgerError(f"seq {event.seq}: {exc}") from None
    for ref in _walk_refs(model):
        seen.setdefault(ref.sha256, ref)
    return tuple(seen.values())


# ================================================================================================
# The pre-registration rules (genesis and amendments)
# ================================================================================================


def _policy_step(current: str | None, event: LedgerEvent, payload: Model) -> str | None:
    """The active policy hash after ``event``. Raises :class:`GenesisError` on a broken rule."""
    if event.kind is EventKind.GENESIS:
        if event.seq != 0:
            raise GenesisError(
                f"a genesis may only be seq 0; this one would be seq {event.seq}. The "
                "pre-registration comes first and exactly once"
            )
        if not isinstance(payload, Genesis):  # pragma: no cover - guaranteed by EVENT_PAYLOADS
            raise GenesisError("a genesis event must carry a Genesis")
        if payload.mode is not event.mode:
            raise GenesisError(
                f"the genesis pre-registers a {payload.mode.value} run but sits in a "
                f"{event.mode.value} ledger"
            )
        return payload.policy_hash
    if event.kind is EventKind.AMENDMENT:
        if not isinstance(payload, Amendment):  # pragma: no cover - guaranteed by EVENT_PAYLOADS
            raise GenesisError("an amendment event must carry an Amendment")
        if current is None:
            raise GenesisError(
                f"seq {event.seq}: an amendment needs a genesis before it; nothing to amend"
            )
        if not payload.owner_confirmed:
            raise GenesisError(
                f"seq {event.seq}: a policy change after genesis needs the owner's confirmation"
            )
        if payload.previous_policy_hash != current:
            raise GenesisError(
                f"seq {event.seq}: the amendment replaces policy {payload.previous_policy_hash}, "
                f"but the active policy is {current}"
            )
        return payload.new_policy_hash
    return current


def policy_hash_after(events: Iterable[LedgerEvent]) -> str | None:
    """The policy hash in force after ``events``: the genesis's, then each amendment's in turn.

    ``None`` when there is no genesis. Events of other kinds are ignored, so a filtered stream of
    genesis and amendment events gives the same answer as the whole log. Raises
    :class:`GenesisError` when the history breaks a pre-registration rule.
    """
    current: str | None = None
    for event in events:
        if event.kind not in (EventKind.GENESIS, EventKind.AMENDMENT):
            continue
        try:
            payload = _typed_payload(event.kind, event.payload)
        except _BrokenLineError as exc:
            raise GenesisError(f"seq {event.seq}: {exc}") from None
        current = _policy_step(current, event, payload)
    return current


# ================================================================================================
# Walking the chain (shared by reading, appending and verifying)
# ================================================================================================


def _refuse_constant(token: str) -> float:
    raise ValueError(f"{token} is not a JSON number")


@dataclass
class _Walk:
    """The state of a walk down the chain: what the next line must be to link on."""

    mode: RunMode | None
    """The mode every event must carry; ``None`` takes it from seq 0."""
    seq: int = 0
    prev_hash: str = ZERO_HASH
    prev_ts: datetime | None = None
    policy_hash: str | None = None

    def step(self, line: bytes) -> LedgerEvent:
        """Accept one line (no newline) as the next event, or raise :class:`_BrokenLineError`.

        The state advances only when every check passes.
        """
        index = self.seq
        try:
            raw = json.loads(line, parse_constant=_refuse_constant)
        except (UnicodeDecodeError, ValueError) as exc:
            raise _BrokenLineError(f"line {index} is not JSON ({exc})") from None
        try:
            event = LedgerEvent.model_validate(raw)
        except ValidationError as exc:
            raise _BrokenLineError(
                f"line {index} is not a ledger event ({_first_error(exc)})"
            ) from None
        try:
            canonical = canonical_json(event)
        except (TypeError, ValueError) as exc:
            raise _BrokenLineError(f"line {index} has no canonical form ({exc})") from None
        if canonical != line:
            raise _BrokenLineError(
                f"line {index} is not in canonical form (sorted keys, no whitespace, values as "
                "written); it was edited after it was written"
            )
        if event.seq != index:
            raise _BrokenLineError(f"line {index} carries seq {event.seq}")
        if event.prev_hash != self.prev_hash:
            linked_to = "the zero hash" if index == 0 else f"the hash of seq {index - 1}"
            raise _BrokenLineError(f"seq {index}: prev_hash is not {linked_to}")
        if not _self_consistent(event):
            raise _BrokenLineError(f"seq {index}: the event does not hash to its hash field")
        mode = self.mode if self.mode is not None else event.mode
        if event.mode is not mode:
            raise _BrokenLineError(
                f"seq {index} is a {event.mode.value} event in a {mode.value} ledger"
            )
        if self.prev_ts is not None and event.ts < self.prev_ts:
            raise _BrokenLineError(f"seq {index}: ts runs backwards from seq {index - 1}")
        try:
            payload = _typed_payload(event.kind, event.payload)
        except _BrokenLineError as exc:
            raise _BrokenLineError(f"seq {index}: {exc}") from None
        try:
            policy_hash = _policy_step(self.policy_hash, event, payload)
        except GenesisError as exc:
            raise _BrokenLineError(str(exc)) from None
        self.mode = mode
        self.seq = index + 1
        self.prev_hash = event.hash
        self.prev_ts = event.ts
        self.policy_hash = policy_hash
        return event


def _lenient_parse(line: bytes) -> LedgerEvent | None:
    """A line past a break, parsed only to count it and find the blobs it names."""
    try:
        return LedgerEvent.model_validate(json.loads(line, parse_constant=_refuse_constant))
    except (UnicodeDecodeError, ValueError, ValidationError):
        return None


# ================================================================================================
# Files: the writer lock, the anchor sidecar, atomic writes
# ================================================================================================


def _retrying(action: Callable[..., _T], *args: object) -> _T:
    """Run a file operation, retrying briefly on ``PermissionError``.

    On Windows a rename onto, or deletion of, a file that another process holds open at that
    instant fails instead of waiting; those holders are readers that let go in microseconds.
    """
    attempt = 0
    while True:
        try:
            return action(*args)
        except PermissionError:
            attempt += 1
            if attempt >= _REPLACE_ATTEMPTS:
                raise
            time.sleep(_RETRY_BACKOFF_S)


def _atomic_write(path: Path, data: bytes) -> None:
    temp = path.with_name(f"{path.name}.{os.getpid()}-{secrets.token_hex(4)}.tmp")
    try:
        with temp.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        _retrying(os.replace, temp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temp.unlink()


def _append_line(path: Path, data: bytes) -> None:
    """One ``write`` on an ``O_APPEND`` descriptor, then ``fsync``."""
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags, 0o644)
    try:
        view = memoryview(data)
        written = 0
        while written < len(data):
            written += os.write(fd, view[written:])
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError):
        return None


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return None


class _WriterLock:
    """``O_CREAT | O_EXCL`` on ``<ledger>.lock``, holding a token that names this holder."""

    def __init__(self, ledger: Path) -> None:
        self.path = ledger.with_name(ledger.name + LOCK_SUFFIX)
        self.breaker = ledger.with_name(ledger.name + BREAK_SUFFIX)
        self.token = f"pid={os.getpid()} host={socket.gethostname()} nonce={secrets.token_hex(8)}"
        self.broken: list[str] = []

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + LOCK_WAIT_SECONDS
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
        last_error = ""
        while True:
            try:
                fd = os.open(self.path, flags, 0o644)
            except (FileExistsError, PermissionError) as exc:
                # PermissionError: on Windows a lock being deleted at this instant cannot be
                # recreated until the deletion completes.
                last_error = type(exc).__name__
                if self._break_if_stale():
                    continue
                if time.monotonic() >= deadline:
                    holder = _read_text(self.path) or "an unknown holder"
                    raise LedgerLocked(
                        f"{self.path.name} has been held for over {LOCK_WAIT_SECONDS:g}s by "
                        f"{holder.strip()} ({last_error}); refusing to write beside it"
                    ) from None
                time.sleep(LOCK_POLL_SECONDS)
                continue
            try:
                os.write(fd, self.token.encode("utf-8"))
            except BaseException:
                os.close(fd)
                with contextlib.suppress(OSError):
                    self.path.unlink()
                raise
            os.close(fd)
            return

    def _break_if_stale(self) -> bool:
        """Remove a lock whose holder died. True when the caller should try again at once."""
        holder = _read_text(self.path)
        stamped = _mtime(self.path)
        if stamped is None:
            return True
        if time.time() - stamped < LOCK_STALE_SECONDS:
            return False
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
        try:
            os.close(os.open(self.breaker, flags, 0o644))
        except (FileExistsError, PermissionError):
            # Another waiter is breaking it now, or died doing so. A break lock is held for
            # microseconds, so one older than the stale age is itself left over.
            broke_at = _mtime(self.breaker)
            if broke_at is not None and time.time() - broke_at >= LOCK_STALE_SECONDS:
                with contextlib.suppress(OSError):
                    self.breaker.unlink()
            return False
        try:
            # Re-read under the break lock: the same holder, still stale. Nobody can create a new
            # lock while this one exists, and no other waiter can break it while we hold the
            # break lock, so what is deleted is exactly what was judged dead.
            if _read_text(self.path) != holder or _mtime(self.path) != stamped:
                return True
            try:
                _retrying(self.path.unlink)
            except FileNotFoundError:
                return True
            described = (holder or "an unreadable lock").strip()
            self.broken.append(described)
            _log.warning(
                "broke a stale ledger lock %s held by %s; its writer died. The next read checks "
                "the log's tail before anything is appended",
                self.path.name,
                described,
            )
            return True
        finally:
            with contextlib.suppress(OSError):
                self.breaker.unlink()

    def held(self) -> bool:
        return _read_text(self.path) == self.token

    def release(self) -> None:
        current = _read_text(self.path)
        if current != self.token:
            _log.warning(
                "the ledger lock %s was no longer ours at release (%s); left in place",
                self.path.name,
                "gone" if current is None else f"now {current.strip()}",
            )
            return
        with contextlib.suppress(FileNotFoundError):
            _retrying(self.path.unlink)


@dataclass(frozen=True)
class _AnchorRead:
    anchor: dict[str, Any] | None
    problem: str | None


def _anchor_path_of(ledger: Path) -> Path:
    return ledger.with_name(ledger.name + ANCHOR_SUFFIX)


def _read_anchor(path: Path) -> _AnchorRead:
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return _AnchorRead(None, None)
    try:
        loaded = json.loads(raw, parse_constant=_refuse_constant)
    except (UnicodeDecodeError, ValueError) as exc:
        return _AnchorRead(None, f"not JSON ({exc})")
    if not isinstance(loaded, dict):
        return _AnchorRead(None, "not a JSON object")
    count = loaded.get("events")
    head = loaded.get("head_hash")
    mode = loaded.get("mode")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        return _AnchorRead(None, "its event count is missing or not a positive integer")
    if not isinstance(head, str) or not _DIGEST.fullmatch(head):
        return _AnchorRead(None, "its head hash is missing or malformed")
    if mode not in {m.value for m in RunMode}:
        return _AnchorRead(None, "its mode is missing or unknown")
    return _AnchorRead(loaded, None)


@dataclass(frozen=True)
class _AnchorVerdict:
    truncated: bool
    note: str


def _judge_anchor(
    read: _AnchorRead, hashes: Sequence[str | None], mode: RunMode | None
) -> _AnchorVerdict:
    """Compare the anchor with the hashes present by line (``None`` for an unreadable line)."""
    if read.problem is not None:
        return _AnchorVerdict(True, f"the head anchor is unreadable ({read.problem})")
    anchor = read.anchor
    present = len(hashes)
    if anchor is None:
        if present == 0:
            return _AnchorVerdict(False, "empty ledger, no head anchor")
        return _AnchorVerdict(
            False, "no head anchor beside the log; removal of its tail cannot be ruled out here"
        )
    count = int(anchor["events"])
    head = str(anchor["head_hash"])
    if mode is not None and anchor["mode"] != mode.value:
        return _AnchorVerdict(
            True, f"the head anchor belongs to a {anchor['mode']} ledger, not a {mode.value} one"
        )
    if count > present:
        return _AnchorVerdict(
            True,
            f"the head anchor records {count} events and {present} are present: "
            f"{count - present} removed from the end",
        )
    if hashes[count - 1] != head:
        return _AnchorVerdict(
            True,
            f"the head anchor records seq {count - 1} as {head}, which is not the event at "
            f"seq {count - 1}: the tail was replaced",
        )
    if count < present:
        return _AnchorVerdict(
            False,
            f"the head anchor agrees up to seq {count - 1} and is {present - count} event(s) "
            "behind the log (a writer stopped between writing a line and its anchor)",
        )
    return _AnchorVerdict(False, f"the head anchor agrees: {count} events, head {head}")


def _mode_named_by(path: Path) -> RunMode | None:
    """The mode a file named ``<mode>.jsonl`` is reserved for, else ``None``."""
    if not path.name.endswith(".jsonl"):
        return None
    try:
        return RunMode(path.name[: -len(".jsonl")])
    except ValueError:
        return None


def ledger_path(root: Path, mode: RunMode) -> Path:
    """``<root>/var/ledger/<mode>.jsonl``: the one working ledger for a mode (DESIGN.md §5)."""
    return Path(root) / "var" / "ledger" / f"{RunMode(mode).value}.jsonl"


# ================================================================================================
# Verification
# ================================================================================================


class ChainVerification(Model):
    """What :meth:`HashChainLedger.verify` and :func:`verify_file` found.

    ``intact`` is true only when every line verifies, the head anchor vouches for the tail, and
    (when a blob store was given) every referenced blob is present and unaltered. Without a blob
    store, blobs are not checked and ``missing_blobs`` is empty.
    """

    events: int = Field(ge=0)
    """Lines that parse as ledger events, verified or not."""
    intact: bool
    first_break_at: int | None
    """Seq (equal to the 0-based line number) of the first line that does not verify."""
    truncated: bool
    """The head anchor does not vouch for this tail: events were removed from the end, the tail
    was replaced, or the anchor itself is unreadable or belongs to another ledger."""
    anchor: str
    """The head-anchor comparison, in words."""
    head_hash: str | None
    """The stored hash of the last event that parses."""
    missing_blobs: tuple[str, ...]
    """Referenced digests the store cannot produce: absent, or altered so they no longer hash to
    their name (the committed bytes are then missing too)."""
    break_reason: str | None = None
    """Why ``first_break_at`` does not verify. An addition to the DESIGN.md §18 interface, with a
    default so every consumer of the listed fields is unaffected."""


def _verify_bytes(
    data: bytes | None,
    anchor: _AnchorRead,
    *,
    mode: RunMode | None,
    blobs: FileBlobStore | None,
) -> ChainVerification:
    walk = _Walk(mode=mode)
    by_line: list[LedgerEvent | None] = []
    first_break: int | None = None
    reason: str | None = None
    chunks = data.split(b"\n") if data else [b""]
    complete, tail = chunks[:-1], chunks[-1]
    for index, line in enumerate(complete):
        if first_break is None:
            try:
                by_line.append(walk.step(line))
                continue
            except _BrokenLineError as exc:
                first_break, reason = index, str(exc)
        by_line.append(_lenient_parse(line))
    if tail:
        by_line.append(_lenient_parse(tail))
        if first_break is None:
            first_break = len(complete)
            reason = (
                f"line {first_break} has no line end: a writer stopped mid-line, or the file "
                "was cut"
            )
    events = [event for event in by_line if event is not None]
    judged_mode = walk.mode if walk.mode is not None else (events[0].mode if events else None)
    verdict = _judge_anchor(
        anchor, [event.hash if event else None for event in by_line], judged_mode
    )
    missing: set[str] = set()
    if blobs is not None:
        for event in events:
            try:
                refs = referenced_blobs(event)
            except LedgerError:
                refs = event.blobs
            missing.update(ref.sha256 for ref in refs if not blobs.has(ref.sha256))
    return ChainVerification(
        events=len(events),
        intact=first_break is None and not verdict.truncated and not missing,
        first_break_at=first_break,
        truncated=verdict.truncated,
        anchor=verdict.note,
        head_hash=events[-1].hash if events else None,
        missing_blobs=tuple(sorted(missing)),
        break_reason=reason,
    )


def _read_bytes(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def verify_file(path: Path, *, blobs_root: Path | None = None) -> ChainVerification:
    """Verify a ledger file on its own, e.g. the exported ``public/ledger.jsonl``.

    The mode is taken from seq 0 (or from the file name, for ``<mode>.jsonl``). The head anchor is
    read from ``<name>.head`` beside the file when it is there. Blobs are checked when
    ``blobs_root`` is given; nothing is created or written.
    """
    path = Path(path)
    return _verify_bytes(
        _read_bytes(path),
        _read_anchor(_anchor_path_of(path)),
        mode=_mode_named_by(path),
        blobs=FileBlobStore(blobs_root) if blobs_root is not None else None,
    )


# ================================================================================================
# The ledger
# ================================================================================================


class HashChainLedger:
    """One mode's log. Satisfies :class:`types.LedgerWriter` and :class:`types.LedgerReader`.

    Reads are served from a cache that is brought up to date from disk on every call, reading only
    what was appended since, so events written by another process are always seen. A reader that
    finds the log broken raises :class:`LedgerError` rather than project anything from it;
    :meth:`verify` is the non-raising diagnosis.
    """

    def __init__(self, path: Path, *, mode: RunMode, clock: Clock) -> None:
        self._path = Path(path)
        self._mode = RunMode(mode)
        self._clock = clock
        reserved = _mode_named_by(self._path)
        if reserved is not None and reserved is not self._mode:
            raise LedgerError(
                f"{self._path.name} is the {reserved.value} ledger; it cannot be opened as "
                f"{self._mode.value}. Modes never share a ledger file"
            )
        self._mutex = threading.RLock()
        self._walk = _Walk(mode=self._mode)
        self._events: list[LedgerEvent] = []
        self._offset = 0
        self._last_line = b""
        self._stale_broken: list[str] = []

    @property
    def mode(self) -> RunMode:
        return self._mode

    @property
    def path(self) -> Path:
        return self._path

    @property
    def anchor_path(self) -> Path:
        """The head-anchor sidecar, ``<name>.head``."""
        return _anchor_path_of(self._path)

    @property
    def stale_locks_broken(self) -> tuple[str, ...]:
        """Holders of stale locks this object broke (a writer that died). For the runtime to log."""
        return tuple(self._stale_broken)

    # --- reading ---------------------------------------------------------------------------------

    def events(self, kinds: frozenset[EventKind] | None = None) -> Iterator[LedgerEvent]:
        """Every event on disk, in order, optionally only those of ``kinds``."""
        with self._mutex:
            self._refresh(for_write=False)
            snapshot = list(self._events)
        if kinds is None:
            return iter(snapshot)
        wanted = frozenset(EventKind(kind) for kind in kinds)
        return iter([event for event in snapshot if event.kind in wanted])

    def head(self) -> LedgerEvent | None:
        with self._mutex:
            self._refresh(for_write=False)
            return self._events[-1] if self._events else None

    def _refresh(self, *, for_write: bool) -> None:
        """Read what was appended since the last read, checking every new line as it links on."""
        name = self._path.name
        try:
            size = self._path.stat().st_size
        except FileNotFoundError:
            if self._events:
                raise LedgerError(
                    f"{name} disappeared after {len(self._events)} events were read from it; "
                    "the ledger is append-only"
                ) from None
            return
        if size < self._offset:
            raise LedgerError(
                f"{name} shrank from {self._offset} to {size} bytes after it was read; the "
                "ledger is append-only. Run verify"
            )
        with self._path.open("rb") as handle:
            if self._last_line:
                handle.seek(self._offset - len(self._last_line))
                if handle.read(len(self._last_line)) != self._last_line:
                    raise LedgerError(
                        f"seq {self._events[-1].seq} of {name} changed after it was read; the "
                        "ledger is append-only. Run verify"
                    )
            handle.seek(self._offset)
            fresh = handle.read()
        position = 0
        while True:
            end = fresh.find(b"\n", position)
            if end < 0:
                break
            try:
                event = self._walk.step(fresh[position:end])
            except _BrokenLineError as exc:
                raise LedgerError(
                    f"{name} does not verify: {exc}. Nothing is read from or written to it until "
                    "it is examined (verify locates the break)"
                ) from None
            self._events.append(event)
            self._last_line = fresh[position : end + 1]
            self._offset += end + 1 - position
            position = end + 1
        tail = fresh[position:]
        if tail and for_write:
            after = f"seq {self._events[-1].seq}" if self._events else "the start"
            raise LedgerError(
                f"{name} ends with {len(tail)} bytes and no line end after {after}: a writer "
                "stopped mid-line. The file is left exactly as it is for the owner to examine; "
                "nothing is appended after a torn line"
            )

    # --- writing ---------------------------------------------------------------------------------

    def append(
        self, kind: EventKind, payload: Model, *, blobs: Sequence[BlobRef] = ()
    ) -> LedgerEvent:
        """Write one event and return it.

        ``payload`` must be exactly the model registered for ``kind``. Raises :class:`GenesisError`
        for a genesis anywhere but seq 0, or an amendment that has no genesis, is unconfirmed, or
        does not amend the active policy; :class:`LedgerLocked` when another writer holds the log;
        :class:`LedgerError` when the log on disk is broken, torn or truncated.
        """
        kind = EventKind(kind)
        expected = EVENT_PAYLOADS[kind]
        if type(payload) is not expected:
            raise LedgerError(
                f"a {kind.value} event carries a {expected.__name__}, not a "
                f"{type(payload).__name__}"
            )
        body = payload.model_dump(mode="json")
        try:
            revived = expected.model_validate(body)
        except ValidationError as exc:
            raise LedgerError(
                f"the {expected.__name__} does not survive being written: {_first_error(exc)}"
            ) from None
        if revived != payload:
            raise LedgerError(f"the {expected.__name__} does not survive being written unchanged")
        refs = tuple(blobs)
        for ref in refs:
            if type(ref) is not BlobRef:
                raise LedgerError(f"an attached blob must be a BlobRef, not {type(ref).__name__}")
        if len({ref.sha256 for ref in refs}) != len(refs):
            raise LedgerError("the same blob is attached twice")

        with self._mutex, self._locked() as lock:
            self._refresh(for_write=True)
            self._refuse_if_truncated()
            head = self._events[-1] if self._events else None
            seq = 0 if head is None else head.seq + 1
            prev_hash = ZERO_HASH if head is None else head.hash
            ts = self._now()
            if head is not None and ts < head.ts:
                ts = head.ts
            event = LedgerEvent(
                seq=seq,
                ts=ts,
                kind=kind,
                mode=self._mode,
                payload=body,
                blobs=refs,
                prev_hash=prev_hash,
                hash=event_hash(
                    seq=seq,
                    ts=ts,
                    kind=kind,
                    mode=self._mode,
                    payload=body,
                    blobs=refs,
                    prev_hash=prev_hash,
                ),
            )
            _policy_step(self._walk.policy_hash, event, revived)
            line = canonical_json(event)
            probe = copy.copy(self._walk)
            try:
                written = probe.step(line)
            except _BrokenLineError as exc:  # pragma: no cover - the writer's output is readable
                raise LedgerError(
                    f"refusing to write an event the reader would reject: {exc}"
                ) from None
            if not lock.held():
                raise LedgerLocked(
                    f"{lock.path.name} was taken from this writer as stale before it wrote; "
                    "nothing was written"
                )
            _append_line(self._path, line + b"\n")
            self._walk = probe
            self._events.append(written)
            self._last_line = line + b"\n"
            self._offset += len(self._last_line)
            try:
                self._write_anchor()
            except OSError as exc:
                # The event is durable; reporting failure now would invite a duplicate. An anchor
                # left behind is tolerated by verify and caught up by the next append.
                _log.error(
                    "seq %d is written but %s could not be updated: %s",
                    written.seq,
                    self.anchor_path.name,
                    exc,
                )
        return written

    def _now(self) -> datetime:
        now = self._clock.now()
        if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
            raise LedgerError("the clock returned a naive datetime; events are stamped in UTC")
        return now.astimezone(UTC)

    @contextlib.contextmanager
    def _locked(self) -> Iterator[_WriterLock]:
        lock = _WriterLock(self._path)
        lock.acquire()
        self._stale_broken.extend(lock.broken)
        try:
            yield lock
        finally:
            try:
                lock.release()
            except OSError as exc:
                _log.error("could not release %s: %s", lock.path.name, exc)

    def _refuse_if_truncated(self) -> None:
        read = _read_anchor(self.anchor_path)
        verdict = _judge_anchor(read, [event.hash for event in self._events], self._mode)
        if verdict.truncated:
            raise LedgerError(
                f"refusing to append to {self._path.name}: {verdict.note}. Writing now would "
                "overwrite the anchor that records it"
            )

    def _write_anchor(self) -> None:
        head = self._events[-1]
        record = {
            "events": len(self._events),
            "head_seq": head.seq,
            "head_hash": head.hash,
            "mode": self._mode.value,
        }
        _atomic_write(self.anchor_path, json.dumps(record, sort_keys=True).encode("utf-8") + b"\n")

    # --- verification ----------------------------------------------------------------------------

    def verify(self, blobs: FileBlobStore | None = None) -> ChainVerification:
        """Re-walk the whole file from disk, compare it with its anchor, and check its blobs.

        The file and its anchor are read together under the writer lock, so an append in progress
        is never mistaken for a torn line; the walk itself runs after the lock is released.
        """
        with self._mutex:
            if self._path.parent.is_dir():
                with self._locked():
                    data = _read_bytes(self._path)
                    anchor = _read_anchor(self.anchor_path)
            else:
                data, anchor = None, _AnchorRead(None, None)
        return _verify_bytes(data, anchor, mode=self._mode, blobs=blobs)


__all__ = [
    "ANCHOR_SUFFIX",
    "LOCK_POLL_SECONDS",
    "LOCK_STALE_SECONDS",
    "LOCK_WAIT_SECONDS",
    "ChainVerification",
    "GenesisError",
    "HashChainLedger",
    "LedgerError",
    "LedgerLocked",
    "event_hash",
    "event_preimage",
    "ledger_path",
    "policy_hash_after",
    "referenced_blobs",
    "verify_file",
]
