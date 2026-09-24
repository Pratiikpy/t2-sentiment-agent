"""OpenTimestamps anchoring: a commitment published where this project cannot rewrite it.

A hash chain proves the log was not edited; it proves nothing to someone who distrusts its author,
who could rebuild the whole chain. So the genesis event and each day's head are stamped with
OpenTimestamps (DESIGN.md §12): public calendars aggregate the digest into a Merkle tree whose root
goes into a Bitcoin transaction, and the proof (an ``.ots`` file) is the path from our digest to
that root. Once the block is mined, the proof shows the digest existed before it, and nobody here
can mine or rewrite a Bitcoin block.

**What is stamped is the event hash itself.** :func:`stamp` writes the event's hash preimage
(:func:`~sentiment_agent.ledger.chain.event_preimage`, whose SHA-256 is the event's ``hash``) to a
file and runs ``ots stamp`` on it, so the proof's file digest *is* the ledger hash a judge sees.
Anyone can check it with the reference client alone: ``ots verify -d <event hash> <proof>.ots``.
Before a proof is accepted, its header is read back and must commit to exactly that hash; a proof
for anything else is recorded as a failure and never stored as ours.

**It is asynchronous, and that is the mechanism, not a caveat.** A fresh proof is *pending*: the
calendars have the digest but no block has confirmed their aggregate yet (typically a few hours).
:func:`upgrade` runs ``ots upgrade`` later to collect the Bitcoin attestation. A pending proof is
recorded as ``submitted``, a complete one as ``upgraded``, and a stamp that did not produce a proof
as ``failed``; none is ever described as more than it is. Each result is a new ``ANCHOR`` ledger
event (the caller appends it); an upgrade never edits the record it upgrades.

**What it does not prove.** Existence before a time. Not that the policy was good, that it was
followed, or that no other commitment was made and withheld.

The reference client is invoked, not reimplemented: ``opentimestamps-client`` 0.7.2 (LGPL-3.0),
installed as the ``ots`` command (``pip install -e .[anchor]``). Its behaviour, read in the
installed source: ``ots stamp FILE`` writes ``FILE.ots`` and exits 1 unless at least 2 of its 4
default calendars answered within ``--timeout`` seconds (``otsclient/cmds.py:147-209``,
``:92-128``); ``ots upgrade FILE.ots`` rewrites the file when calendars have news, keeps
``FILE.ots.bak``, and exits 0 only when a Bitcoin attestation is present (``cmds.py:336-382``).
The ``.ots`` layout read here (header magic, version 1, the SHA-256 file-hash op ``0x08``, the
32-byte digest; attestation tags ``83dfe30d2ef90c8e`` pending and ``0588960d73d71901`` Bitcoin,
each after a ``0x00`` marker with a length-prefixed payload) is from ``python-opentimestamps``
0.4.5 ``core/timestamp.py:273-341`` and ``core/notary.py:32-39, 140-255``. Studied, not copied.

**On Windows the stock ``ots`` command cannot start**, and :func:`subprocess_ots_runner` works
around it rather than record every anchor as failed. The client imports ``python-bitcoinlib``
0.12.2 (the latest release), whose ``bitcoin/core/key.py:27-29`` loads OpenSSL with
``ctypes.util.find_library('ssl.35' | 'ssl' | 'libeay32')``. On Windows none of those names
resolves, ``LoadLibrary(None)`` raises ``TypeError: argument of type 'NoneType' is not iterable``,
and ``ots`` exits before reading its arguments (measured here on 2026-09-24; upstream issue
``petertodd/python-bitcoinlib#316``, open). The fix proposed upstream (PR #319, open) is the one
applied: fall back to the ``libcrypto-*.dll`` that CPython ships in ``<base_prefix>/DLLs``, which
exports every function the module binds at import. So when the reference client is installed in
this interpreter, the runner starts it as ``python -c <bootstrap>``: the bootstrap installs that
fallback (on Windows only, and only for a name ``find_library`` could not resolve) and calls
``otsclient.ots.main`` (what the ``ots`` console script calls) with the same arguments. Otherwise
it runs the ``ots`` executable on ``PATH``. Measured with the fallback: ``ots --version`` prints
``v0.7.2`` and exits 0; the end-to-end test runs ``stamp`` and ``upgrade`` through this runner.

The tests never contact a public calendar: they pass a fake :data:`OtsRunner`, or run the reference
client through the production runner against a calendar on the loopback interface. Nothing in the
build stamps a live calendar.
"""

import contextlib
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

from sentiment_agent.hashing import sha256_hex
from sentiment_agent.ledger.chain import event_preimage
from sentiment_agent.types import AnchorRecord, BlobRef, BlobStore, Clock, LedgerEvent

OtsRunner = Callable[[Sequence[str], Path, float], tuple[int, str, str]]
"""``(argv, cwd, timeout_s) -> (exit code, stdout, stderr)``. ``argv[0]`` is always ``"ots"``."""

OTS_COMMAND: Final = "ots"
OTS_MEDIA_TYPE: Final = "application/vnd.opentimestamps.v1"
CALENDAR_TIMEOUT_SECONDS: Final = 20
"""Passed to ``ots stamp --timeout``: how long to wait for the calendars (the client's default is
5 s, tight for a submission made once a day that must not fail for a slow round trip)."""
STAMP_TIMEOUT_SECONDS: Final = 120.0
UPGRADE_TIMEOUT_SECONDS: Final = 120.0
DETAIL_MAX_CHARS: Final = 600

_HEADER_MAGIC: Final = b"\x00OpenTimestamps\x00\x00Proof\x00\xbf\x89\xe2\xe8\x84\xe8\x92\x94"
_MAJOR_VERSION: Final = 1
_OP_SHA256: Final = 0x08
_DIGEST_BYTES: Final = 32
_PENDING_TAG: Final = bytes.fromhex("83dfe30d2ef90c8e")
_BITCOIN_TAG: Final = bytes.fromhex("0588960d73d71901")
_URI_CHARS: Final = frozenset(
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._/:"
)
"""``PendingAttestation.ALLOWED_URI_CHARS`` (``core/notary.py``): what a calendar URI may hold."""

_WINDOWS_PATH: Final = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/][^\s'\"]*")
"""A drive path. The look-behind keeps the ``s://`` of ``https://`` from reading as drive ``s:``."""
_POSIX_PATH: Final = re.compile(r"(?<![\w:/])/(?:home|Users|root|tmp|var|private|mnt)/[^\s'\"]*")


class AnchorError(RuntimeError):
    """Asked to anchor something that cannot honestly be anchored."""


@dataclass(frozen=True)
class ProofSummary:
    """What an ``.ots`` file commits to and which attestations it carries."""

    digest: str | None
    """The SHA-256 the proof is for, hex; ``None`` when the header is not a SHA-256 proof."""
    calendars: tuple[str, ...]
    """URIs of the calendars holding a pending attestation."""
    bitcoin_heights: tuple[int, ...]
    """Heights of the Bitcoin blocks attesting it (not verified here; ``ots verify`` does that)."""
    problem: str | None


def _varuint(data: bytes, position: int) -> tuple[int, int]:
    """An unsigned LEB128 integer at ``position``; returns ``(value, next position)``."""
    value = 0
    shift = 0
    while True:
        if position >= len(data):
            raise ValueError("a length runs past the end of the proof")
        byte = data[position]
        position += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, position
        shift += 7
        if shift > 63:
            raise ValueError("a length is longer than 64 bits")


def _attestation_payloads(data: bytes, start: int, tag: bytes) -> list[bytes]:
    marker = b"\x00" + tag
    payloads: list[bytes] = []
    at = data.find(marker, start)
    while at >= 0:
        try:
            length, begin = _varuint(data, at + len(marker))
        except ValueError:
            break
        payloads.append(data[begin : begin + length])
        at = data.find(marker, at + 1)
    return payloads


def read_proof(data: bytes) -> ProofSummary:
    """Read an ``.ots`` file's header and attestations. Never raises on malformed input."""
    if not data.startswith(_HEADER_MAGIC):
        return ProofSummary(None, (), (), "not an OpenTimestamps proof (bad header)")
    try:
        major, position = _varuint(data, len(_HEADER_MAGIC))
    except ValueError as exc:
        return ProofSummary(None, (), (), f"unreadable version ({exc})")
    if major != _MAJOR_VERSION:
        return ProofSummary(None, (), (), f"unsupported proof version {major}")
    if position >= len(data) or data[position] != _OP_SHA256:
        return ProofSummary(None, (), (), "the file hash is not SHA-256")
    digest_bytes = data[position + 1 : position + 1 + _DIGEST_BYTES]
    if len(digest_bytes) != _DIGEST_BYTES:
        return ProofSummary(None, (), (), "the proof is cut short inside its digest")
    body = position + 1 + _DIGEST_BYTES
    calendars: list[str] = []
    for payload in _attestation_payloads(data, body, _PENDING_TAG):
        try:
            length, begin = _varuint(payload, 0)
        except ValueError:
            continue
        uri = payload[begin : begin + length]
        if uri and len(uri) == length and all(byte in _URI_CHARS for byte in uri):
            calendars.append(uri.decode("ascii"))
    heights: list[int] = []
    for payload in _attestation_payloads(data, body, _BITCOIN_TAG):
        try:
            height, _ = _varuint(payload, 0)
        except ValueError:
            continue
        heights.append(height)
    return ProofSummary(
        digest=digest_bytes.hex(),
        calendars=tuple(dict.fromkeys(calendars)),
        bitcoin_heights=tuple(sorted(set(heights))),
        problem=None,
    )


def _scrub(text: str, *known: Path) -> str:
    """Text fit for the published ledger: no local path, one line, bounded length."""
    for path in known:
        for spelling in {str(path), path.as_posix()}:
            text = text.replace(spelling, "<workdir>")
    text = _WINDOWS_PATH.sub("<path>", text)
    text = _POSIX_PATH.sub("<path>", text)
    text = " ".join(text.split())
    if len(text) > DETAIL_MAX_CHARS:
        text = text[: DETAIL_MAX_CHARS - 1] + "…"
    return text


def _tail(stdout: str, stderr: str, lines: int = 3) -> str:
    kept = [line.strip() for line in (stderr or stdout).splitlines() if line.strip()]
    return " | ".join(kept[-lines:]) if kept else "no output"


_OTS_BOOTSTRAP: Final = """\
import ctypes.util, glob, os, sys
if sys.platform == "win32":
    _find = ctypes.util.find_library
    def _find_library(name, _find=_find):
        found = _find(name)
        if found is None and name in ("ssl", "libeay32", "libcrypto"):
            for base in dict.fromkeys((sys.base_prefix, sys.prefix)):
                bundled = sorted(glob.glob(os.path.join(base, "DLLs", "libcrypto-*.dll")))
                if bundled:
                    return bundled[-1]
        return found
    ctypes.util.find_library = _find_library
from otsclient.ots import main
sys.argv[0] = "ots"
main()
"""
"""Starts the reference client with the Windows OpenSSL fallback described in the module docstring
(the approach of python-bitcoinlib PR #319). Changes nothing on other platforms."""


def _reference_client_here() -> bool:
    """True when ``opentimestamps-client`` is importable by this interpreter. Found, not imported:
    importing it is exactly what fails on Windows without the fallback."""
    try:
        return importlib.util.find_spec("otsclient") is not None
    except (ImportError, ValueError):
        return False


def subprocess_ots_runner(argv: Sequence[str], cwd: Path, timeout_s: float) -> tuple[int, str, str]:
    """The production :data:`OtsRunner`: the reference OpenTimestamps client, no shell.

    Runs the client installed in this interpreter through :data:`_OTS_BOOTSTRAP` when there is one,
    else the ``ots`` executable on ``PATH``. Exit 127 when neither exists, 124 on timeout, 126 when
    it cannot be started. The child gets the parent's environment without any ``BITGET_*``
    variable: it needs no credential.
    """
    if not argv or argv[0] != OTS_COMMAND:
        raise ValueError("this runner only starts the ots command")
    if _reference_client_here():
        command = [sys.executable, "-c", _OTS_BOOTSTRAP, *argv[1:]]
    else:
        executable = shutil.which(OTS_COMMAND)
        if executable is None:
            return (
                127,
                "",
                "the ots command is not installed (pip install -e .[anchor], which installs "
                "opentimestamps-client)",
            )
        command = [executable, *argv[1:]]
    environment = {k: v for k, v in os.environ.items() if not k.upper().startswith("BITGET_")}
    try:
        done = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return 124, "", f"ots did not finish within {timeout_s:g}s"
    except OSError as exc:
        return 126, "", f"ots could not be started ({type(exc).__name__}: {exc})"
    return done.returncode, done.stdout, done.stderr


def _run(runner: OtsRunner, argv: list[str], cwd: Path, timeout_s: float) -> tuple[int, str, str]:
    try:
        return runner(argv, cwd, timeout_s)
    except OSError as exc:
        return 126, "", f"the ots runner failed ({type(exc).__name__}: {exc})"


def _iso(moment: datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


def stamp(
    event: LedgerEvent,
    *,
    runner: OtsRunner,
    blobs: BlobStore,
    workdir: Path,
    clock: Clock,
) -> AnchorRecord:
    """Submit ``event.hash`` to the OpenTimestamps calendars. Never raises for a failed submission.

    Returns ``submitted`` with the pending proof stored as a blob, or ``failed`` with the reason and
    no blob. Raises :class:`AnchorError` only when the event does not hash to its own ``hash``
    field: a hash that does not verify is never timestamped.
    """
    preimage = event_preimage(event)
    if sha256_hex(preimage) != event.hash:
        raise AnchorError(
            f"seq {event.seq} does not hash to its hash field; refusing to timestamp an event "
            "that was altered or built by hand"
        )
    submitted_at = clock.now()
    workdir.mkdir(parents=True, exist_ok=True)
    name = f"seq{event.seq:08d}-{event.hash}.json"
    with tempfile.TemporaryDirectory(
        dir=workdir, prefix="ots-stamp-", ignore_cleanup_errors=True
    ) as scratch:
        folder = Path(scratch)
        (folder / name).write_bytes(preimage)
        argv = [OTS_COMMAND, "stamp", "--timeout", str(CALENDAR_TIMEOUT_SECONDS), name]
        code, out, err = _run(runner, argv, folder, STAMP_TIMEOUT_SECONDS)
        proof_file = folder / f"{name}.ots"
        proof = proof_file.read_bytes() if proof_file.is_file() else None
        detail_paths = (folder, workdir)

    def failed(why: str) -> AnchorRecord:
        return AnchorRecord(
            target_seq=event.seq,
            target_hash=event.hash,
            submitted_at=submitted_at,
            status="failed",
            ots_blob=None,
            detail=_scrub(f"not anchored: {why}", *detail_paths),
        )

    if code != 0:
        return failed(f"ots stamp exited {code}: {_tail(out, err)}")
    if proof is None:
        return failed("ots stamp exited 0 but wrote no proof")
    summary = read_proof(proof)
    if summary.problem is not None:
        return failed(f"the file ots wrote is not a usable proof ({summary.problem})")
    if summary.digest != event.hash:
        return failed(f"the proof commits to {summary.digest}, not to this event's hash")
    if not summary.calendars and not summary.bitcoin_heights:
        return failed("the proof carries no calendar or Bitcoin attestation")
    ref: BlobRef = blobs.put(proof, OTS_MEDIA_TYPE)
    if summary.bitcoin_heights:
        return AnchorRecord(
            target_seq=event.seq,
            target_hash=event.hash,
            submitted_at=submitted_at,
            status="upgraded",
            ots_blob=ref,
            detail=_complete_detail(event.hash, summary, clock.now()),
        )
    return AnchorRecord(
        target_seq=event.seq,
        target_hash=event.hash,
        submitted_at=submitted_at,
        status="submitted",
        ots_blob=ref,
        detail=_scrub(
            f"pending in {len(summary.calendars)} calendar(s): {', '.join(summary.calendars)}. "
            "A Bitcoin attestation follows once a block confirms the calendars' aggregate "
            "(typically hours); ots upgrade collects it. Check with: ots verify -d "
            f"{event.hash} <proof>.ots",
            *detail_paths,
        ),
    )


def _complete_detail(target_hash: str, summary: ProofSummary, at: datetime) -> str:
    heights = ", ".join(str(h) for h in summary.bitcoin_heights)
    return _scrub(
        f"complete as of {_iso(at)}: the proof carries a Bitcoin attestation at block {heights}. "
        f"It is not verified against Bitcoin here; check with: ots verify -d {target_hash} "
        "<proof>.ots"
    )


def upgrade(
    record: AnchorRecord,
    *,
    runner: OtsRunner,
    blobs: BlobStore,
    workdir: Path,
    clock: Clock,
) -> AnchorRecord:
    """Try to complete a pending proof. Returns a new record; ``record`` itself is never changed.

    ``upgraded`` when ``ots upgrade`` exits 0 and the proof now carries a Bitcoin attestation;
    otherwise still ``submitted``, with the newest proof (a partial upgrade is kept) and what
    happened. A record already ``upgraded`` is returned as it is. Raises :class:`AnchorError` for a
    ``failed`` record (there is no proof to upgrade) or a stored proof that does not commit to the
    record's target.
    """
    if record.status == "failed" or record.ots_blob is None:
        raise AnchorError(
            f"the anchor of seq {record.target_seq} failed; there is no proof to upgrade. Stamp it "
            "again"
        )
    if record.status == "upgraded":
        return record
    current = blobs.get(record.ots_blob.sha256)
    if read_proof(current).digest != record.target_hash:
        raise AnchorError(
            f"the stored proof {record.ots_blob.sha256} does not commit to seq "
            f"{record.target_seq}'s hash"
        )
    workdir.mkdir(parents=True, exist_ok=True)
    name = f"seq{record.target_seq:08d}-{record.target_hash}.json.ots"
    with tempfile.TemporaryDirectory(
        dir=workdir, prefix="ots-upgrade-", ignore_cleanup_errors=True
    ) as scratch:
        folder = Path(scratch)
        proof_file = folder / name
        proof_file.write_bytes(current)
        code, out, err = _run(
            runner, [OTS_COMMAND, "upgrade", name], folder, UPGRADE_TIMEOUT_SECONDS
        )
        after: bytes | None = None
        with contextlib.suppress(OSError):
            after = proof_file.read_bytes()
        detail_paths = (folder, workdir)
    now = clock.now()
    summary = read_proof(after) if after is not None else None
    if after is None or summary is None or summary.digest != record.target_hash:
        # ots replaces the file only with a proof for the same digest; anything else is not ours.
        after, summary = current, read_proof(current)
    ref = record.ots_blob if after == current else blobs.put(after, OTS_MEDIA_TYPE)
    if code == 0 and summary.bitcoin_heights:
        return AnchorRecord(
            target_seq=record.target_seq,
            target_hash=record.target_hash,
            submitted_at=record.submitted_at,
            status="upgraded",
            ots_blob=ref,
            detail=_complete_detail(record.target_hash, summary, now),
        )
    if code == 0:
        why = "ots upgrade exited 0 but the proof carries no Bitcoin attestation"
    else:
        why = f"ots upgrade exited {code}: {_tail(out, err)}"
    progress = "; the proof gained attestations" if ref is not record.ots_blob else ""
    return AnchorRecord(
        target_seq=record.target_seq,
        target_hash=record.target_hash,
        submitted_at=record.submitted_at,
        status="submitted",
        ots_blob=ref,
        detail=_scrub(f"still pending as of {_iso(now)} ({why}){progress}", *detail_paths),
    )


__all__ = [
    "CALENDAR_TIMEOUT_SECONDS",
    "OTS_COMMAND",
    "OTS_MEDIA_TYPE",
    "AnchorError",
    "OtsRunner",
    "ProofSummary",
    "read_proof",
    "stamp",
    "subprocess_ots_runner",
    "upgrade",
]
