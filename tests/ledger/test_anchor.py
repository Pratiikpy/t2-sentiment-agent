"""OpenTimestamps anchoring over a fake ``ots``, the proof reader, and the reference client itself.

The fake mimics what ``opentimestamps-client`` 0.7.2 does to files (``otsclient/cmds.py``): ``stamp
FILE`` writes ``FILE.ots`` for the file's SHA-256; ``upgrade FILE.ots`` keeps ``FILE.ots.bak`` and
rewrites the proof, exiting 0 only when it is complete. The last test runs the real client, as a
Python child process, against a calendar served on the loopback interface: nothing leaves the
machine and no public calendar is written to.
"""

import hashlib
import http.server
import importlib.util
import os
import shutil
import sys
import threading
from collections.abc import Iterator, Sequence
from datetime import timedelta
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import pytest

from sentiment_agent.clock import ManualClock
from sentiment_agent.ledger import anchor
from sentiment_agent.ledger.anchor import (
    OTS_MEDIA_TYPE,
    AnchorError,
    read_proof,
    stamp,
    subprocess_ots_runner,
    upgrade,
)
from sentiment_agent.ledger.blobs import FileBlobStore
from sentiment_agent.ledger.chain import HashChainLedger
from sentiment_agent.types import AnchorRecord, EventKind, LedgerEvent, Note, RunMode

MAGIC = b"\x00OpenTimestamps\x00\x00Proof\x00\xbf\x89\xe2\xe8\x84\xe8\x92\x94"
PENDING = bytes.fromhex("83dfe30d2ef90c8e")
BITCOIN = bytes.fromhex("0588960d73d71901")
CALENDAR = "https://a.pool.opentimestamps.org"


# --- building proofs the way the reference client lays them out -----------------------------------


def _varuint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _varbytes(data: bytes) -> bytes:
    return _varuint(len(data)) + data


def _proof(
    digest: bytes, *, calendars: Sequence[str] = (CALENDAR,), bitcoin: int | None = None
) -> bytes:
    """A detached SHA-256 proof: append a nonce, sha256, then the attestations on the result."""
    attestations = [PENDING + _varbytes(_varbytes(uri.encode())) for uri in sorted(calendars)]
    if bitcoin is not None:
        attestations.insert(0, BITCOIN + _varbytes(_varuint(bitcoin)))
    body = b"".join(b"\xff\x00" + a for a in attestations[:-1]) + b"\x00" + attestations[-1]
    return MAGIC + b"\x01" + b"\x08" + digest + b"\xf0" + _varbytes(b"n" * 16) + b"\x08" + body


def _digest_of(proof: bytes) -> bytes:
    start = len(MAGIC) + 2
    return proof[start : start + 32]


Upgrade = Literal["complete", "pending", "error", "partial", "other-digest", "no-bitcoin"]


class FakeOts:
    """Emulates the reference client's effect on files, and records every call."""

    def __init__(
        self,
        *,
        stamp_code: int = 0,
        stamp_writes: Literal["proof", "nothing", "garbage", "other-digest", "bare"] = "proof",
        stamp_stderr: str = f"Submitting to remote calendar {CALENDAR}\n",
        upgrade_as: Upgrade = "complete",
    ) -> None:
        self.stamp_code = stamp_code
        self.stamp_writes = stamp_writes
        self.stamp_stderr = stamp_stderr
        self.upgrade_as = upgrade_as
        self.calls: list[tuple[tuple[str, ...], Path, float]] = []
        self.stamped_digests: list[str] = []

    def __call__(self, argv: Sequence[str], cwd: Path, timeout_s: float) -> tuple[int, str, str]:
        self.calls.append((tuple(argv), cwd, timeout_s))
        assert argv[0] == "ots"
        target = cwd / argv[-1]
        if argv[1] == "stamp":
            digest = hashlib.sha256(target.read_bytes()).digest()
            self.stamped_digests.append(digest.hex())
            out = target.with_name(target.name + ".ots")
            if self.stamp_writes == "proof":
                out.write_bytes(_proof(digest))
            elif self.stamp_writes == "garbage":
                out.write_bytes(b"not a proof at all")
            elif self.stamp_writes == "other-digest":
                out.write_bytes(_proof(hashlib.sha256(b"something else").digest()))
            elif self.stamp_writes == "bare":
                out.write_bytes(MAGIC + b"\x01\x08" + digest)
            return self.stamp_code, "", self.stamp_stderr
        assert argv[1] == "upgrade"
        current = target.read_bytes()
        digest = _digest_of(current)
        if self.upgrade_as == "pending":
            return 1, "", "Failed! Timestamp not complete\n"
        if self.upgrade_as == "error":
            return 1, "", f"Calendar {CALENDAR}: Not Found in C:\\Users\\someone\\ots\\cache\n"
        target.with_name(target.name + ".bak").write_bytes(current)
        if self.upgrade_as == "complete":
            target.write_bytes(_proof(digest, bitcoin=915_123))
            return 0, "", "Got 1 attestation(s) from cache\nSuccess! Timestamp complete\n"
        if self.upgrade_as == "partial":
            target.write_bytes(_proof(digest, calendars=(CALENDAR, "https://b.pool.example")))
            return 1, "", "Failed! Timestamp not complete\n"
        if self.upgrade_as == "other-digest":
            target.write_bytes(_proof(hashlib.sha256(b"other").digest(), bitcoin=915_123))
            return 0, "", "Success! Timestamp complete\n"
        target.write_bytes(_proof(digest))  # "no-bitcoin"
        return 0, "", "Success! Timestamp complete\n"


@pytest.fixture
def event(tmp_path: Path, clock: ManualClock) -> LedgerEvent:
    ledger = HashChainLedger(tmp_path / "simulated.jsonl", mode=RunMode.SIMULATED, clock=clock)
    ledger.append(EventKind.NOTE, Note(at=clock.now(), author="system", text="first"))
    return ledger.append(EventKind.NOTE, Note(at=clock.now(), author="system", text="head"))


@pytest.fixture
def store(tmp_path: Path) -> FileBlobStore:
    return FileBlobStore(tmp_path / "blobs")


# --- stamping -------------------------------------------------------------------------------------


def test_stamp_commits_to_the_event_hash_itself(
    event: LedgerEvent, store: FileBlobStore, tmp_path: Path, clock: ManualClock
) -> None:
    ots = FakeOts()
    workdir = tmp_path / "work"
    record = stamp(event, runner=ots, blobs=store, workdir=workdir, clock=clock)

    assert record.status == "submitted"
    assert (record.target_seq, record.target_hash) == (event.seq, event.hash)
    assert record.submitted_at == clock.now()
    assert record.ots_blob is not None
    assert record.ots_blob.media_type == OTS_MEDIA_TYPE
    assert read_proof(store.get(record.ots_blob.sha256)).digest == event.hash
    assert ots.stamped_digests == [event.hash]  # the file ots hashed is the event's preimage
    [(argv, cwd, timeout)] = ots.calls
    assert argv == ("ots", "stamp", "--timeout", "20", f"seq00000001-{event.hash}.json")
    assert cwd.parent == workdir
    assert timeout == anchor.STAMP_TIMEOUT_SECONDS
    assert list(workdir.iterdir()) == []  # the scratch folder is gone
    assert CALENDAR in record.detail
    assert f"ots verify -d {event.hash}" in record.detail


def test_an_anchor_record_is_itself_a_ledger_event(
    tmp_path: Path, clock: ManualClock, store: FileBlobStore
) -> None:
    ledger = HashChainLedger(tmp_path / "paper.jsonl", mode=RunMode.PAPER, clock=clock)
    head = ledger.append(EventKind.NOTE, Note(at=clock.now(), author="system", text="head"))
    record = stamp(head, runner=FakeOts(), blobs=store, workdir=tmp_path / "work", clock=clock)
    assert record.ots_blob is not None
    ledger.append(EventKind.ANCHOR, record, blobs=(record.ots_blob,))
    clock.advance(timedelta(hours=6))
    upgraded = upgrade(
        record, runner=FakeOts(), blobs=store, workdir=tmp_path / "work", clock=clock
    )
    ledger.append(EventKind.ANCHOR, upgraded)  # a new event; the first record is untouched
    result = ledger.verify(store)
    assert result.intact
    assert result.events == 3


@pytest.mark.parametrize(
    ("ots", "reason"),
    [
        (
            FakeOts(
                stamp_code=1,
                stamp_writes="nothing",
                stamp_stderr="Failed to create timestamp: need at least 2 attestations but "
                "received 0 within timeout\n",
            ),
            "ots stamp exited 1: Failed to create timestamp: need at least 2 attestations",
        ),
        (FakeOts(stamp_writes="nothing"), "exited 0 but wrote no proof"),
        (FakeOts(stamp_writes="garbage"), "not a usable proof (not an OpenTimestamps proof"),
        (FakeOts(stamp_writes="other-digest"), "not to this event's hash"),
        (FakeOts(stamp_writes="bare"), "no calendar or Bitcoin attestation"),
    ],
)
def test_a_failed_stamp_is_recorded_as_failed_not_raised(
    event: LedgerEvent,
    store: FileBlobStore,
    tmp_path: Path,
    clock: ManualClock,
    ots: FakeOts,
    reason: str,
) -> None:
    record = stamp(event, runner=ots, blobs=store, workdir=tmp_path / "work", clock=clock)
    assert (record.status, record.ots_blob) == ("failed", None)
    assert record.detail.startswith("not anchored: ")
    assert reason in record.detail
    assert store.verify_all() == []
    assert not store.root.exists()  # no proof was stored as ours


def test_a_missing_ots_command_or_a_broken_runner_is_a_failed_stamp(
    event: LedgerEvent, store: FileBlobStore, tmp_path: Path, clock: ManualClock
) -> None:
    def not_installed(argv: Sequence[str], cwd: Path, timeout_s: float) -> tuple[int, str, str]:
        return 127, "", "the ots command is not installed"

    def broken(argv: Sequence[str], cwd: Path, timeout_s: float) -> tuple[int, str, str]:
        raise PermissionError("access denied")

    first = stamp(event, runner=not_installed, blobs=store, workdir=tmp_path / "w", clock=clock)
    assert first.status == "failed"
    assert "exited 127" in first.detail
    second = stamp(event, runner=broken, blobs=store, workdir=tmp_path / "w", clock=clock)
    assert second.status == "failed"
    assert "PermissionError: access denied" in second.detail


def test_no_local_path_reaches_the_record(
    event: LedgerEvent, store: FileBlobStore, tmp_path: Path, clock: ManualClock
) -> None:
    workdir = tmp_path / "work"
    leaky = FakeOts(
        stamp_code=1,
        stamp_writes="nothing",
        stamp_stderr=(
            f"Could not read {workdir / 'x.json'}: denied\n"
            "cache at C:\\Users\\someone\\AppData\\Local\\ots and /home/runner/.cache/ots\n"
            "Failed to create timestamp\n"
        ),
    )
    record = stamp(event, runner=leaky, blobs=store, workdir=workdir, clock=clock)
    for leaked in (str(tmp_path), "someone", "AppData", "/home/runner"):
        assert leaked not in record.detail
    assert "<workdir>" in record.detail
    assert "<path>" in record.detail


def test_an_altered_event_is_never_stamped(
    event: LedgerEvent, store: FileBlobStore, tmp_path: Path, clock: ManualClock
) -> None:
    altered = event.model_copy(update={"payload": {**event.payload, "text": "edited"}})
    ots = FakeOts()
    with pytest.raises(AnchorError, match="refusing to timestamp"):
        stamp(altered, runner=ots, blobs=store, workdir=tmp_path / "work", clock=clock)
    assert ots.calls == []


# --- upgrading ------------------------------------------------------------------------------------


def _submitted(
    event: LedgerEvent, store: FileBlobStore, tmp_path: Path, clock: ManualClock
) -> AnchorRecord:
    record = stamp(event, runner=FakeOts(), blobs=store, workdir=tmp_path / "work", clock=clock)
    assert record.status == "submitted"
    clock.advance(timedelta(hours=6))
    return record


def test_upgrade_completes_a_pending_proof(
    event: LedgerEvent, store: FileBlobStore, tmp_path: Path, clock: ManualClock
) -> None:
    pending = _submitted(event, store, tmp_path, clock)
    ots = FakeOts(upgrade_as="complete")
    done = upgrade(pending, runner=ots, blobs=store, workdir=tmp_path / "work", clock=clock)
    assert done.status == "upgraded"
    assert (done.target_seq, done.target_hash) == (pending.target_seq, pending.target_hash)
    assert done.submitted_at == pending.submitted_at  # when it was submitted does not change
    assert done.ots_blob is not None
    assert pending.ots_blob is not None
    assert done.ots_blob.sha256 != pending.ots_blob.sha256
    summary = read_proof(store.get(done.ots_blob.sha256))
    assert (summary.digest, summary.bitcoin_heights) == (event.hash, (915_123,))
    assert "block 915123" in done.detail
    assert "not verified against Bitcoin here" in done.detail
    [(argv, _, _)] = ots.calls
    assert argv == ("ots", "upgrade", f"seq00000001-{event.hash}.json.ots")
    assert pending.status == "submitted"  # the record it upgraded is unchanged


@pytest.mark.parametrize(
    ("mode", "why", "new_blob"),
    [
        ("pending", "ots upgrade exited 1: Failed! Timestamp not complete", False),
        ("error", "ots upgrade exited 1: Calendar", False),
        ("no-bitcoin", "exited 0 but the proof carries no Bitcoin attestation", False),
        ("partial", "the proof gained attestations", True),
        ("other-digest", "no Bitcoin attestation", False),
    ],
)
def test_an_incomplete_upgrade_stays_submitted(
    event: LedgerEvent,
    store: FileBlobStore,
    tmp_path: Path,
    clock: ManualClock,
    mode: Upgrade,
    why: str,
    new_blob: bool,
) -> None:
    pending = _submitted(event, store, tmp_path, clock)
    result = upgrade(
        pending, runner=FakeOts(upgrade_as=mode), blobs=store, workdir=tmp_path / "w", clock=clock
    )
    assert result.status == "submitted"
    assert result.detail.startswith("still pending as of 2026-09-23T19:00:00Z")
    assert why in result.detail
    assert "someone" not in result.detail  # the error's local path was scrubbed
    assert result.ots_blob is not None
    assert pending.ots_blob is not None
    assert (result.ots_blob.sha256 != pending.ots_blob.sha256) is new_blob
    assert read_proof(store.get(result.ots_blob.sha256)).digest == event.hash


def test_upgrade_refuses_what_it_cannot_honestly_upgrade(
    event: LedgerEvent, store: FileBlobStore, tmp_path: Path, clock: ManualClock
) -> None:
    failed = stamp(
        event,
        runner=FakeOts(stamp_code=1, stamp_writes="nothing"),
        blobs=store,
        workdir=tmp_path / "w",
        clock=clock,
    )
    with pytest.raises(AnchorError, match="no proof to upgrade"):
        upgrade(failed, runner=FakeOts(), blobs=store, workdir=tmp_path / "w", clock=clock)

    pending = _submitted(event, store, tmp_path, clock)
    foreign = store.put(_proof(hashlib.sha256(b"not this event").digest()), OTS_MEDIA_TYPE)
    mismatched = pending.model_copy(update={"ots_blob": foreign})
    with pytest.raises(AnchorError, match="does not commit"):
        upgrade(mismatched, runner=FakeOts(), blobs=store, workdir=tmp_path / "w", clock=clock)


def test_a_complete_proof_is_not_upgraded_again(
    event: LedgerEvent, store: FileBlobStore, tmp_path: Path, clock: ManualClock
) -> None:
    pending = _submitted(event, store, tmp_path, clock)
    done = upgrade(pending, runner=FakeOts(), blobs=store, workdir=tmp_path / "w", clock=clock)
    ots = FakeOts()
    assert upgrade(done, runner=ots, blobs=store, workdir=tmp_path / "w", clock=clock) is done
    assert ots.calls == []


# --- reading proofs -------------------------------------------------------------------------------


def test_read_proof_reads_the_header_and_attestations() -> None:
    digest = hashlib.sha256(b"x").digest()
    summary = read_proof(_proof(digest, calendars=(CALENDAR, "https://b.pool.x"), bitcoin=800_000))
    assert summary.problem is None
    assert summary.digest == digest.hex()
    assert set(summary.calendars) == {CALENDAR, "https://b.pool.x"}
    assert summary.bitcoin_heights == (800_000,)


@pytest.mark.parametrize(
    ("data", "problem"),
    [
        (b"", "bad header"),
        (b"%PDF-1.7", "bad header"),
        (MAGIC, "unreadable version"),
        (MAGIC + b"\x02\x08" + b"\x00" * 32, "unsupported proof version 2"),
        (MAGIC + b"\x01\x02" + b"\x00" * 20, "not SHA-256"),
        (MAGIC + b"\x01\x08" + b"\x00" * 31, "cut short"),
    ],
)
def test_read_proof_never_raises_on_malformed_input(data: bytes, problem: str) -> None:
    summary = read_proof(data)
    assert summary.digest is None
    assert summary.problem is not None
    assert problem in summary.problem


def test_a_uri_outside_the_calendar_alphabet_is_ignored() -> None:
    digest = hashlib.sha256(b"x").digest()
    hostile = _proof(digest, calendars=("https://ok.example", "https://evil.example/?q=1"))
    assert read_proof(hostile).calendars == ("https://ok.example",)


def _reference() -> tuple[ModuleType, ModuleType, ModuleType, ModuleType]:
    return (
        pytest.importorskip("opentimestamps.core.timestamp"),
        pytest.importorskip("opentimestamps.core.op"),
        pytest.importorskip("opentimestamps.core.notary"),
        pytest.importorskip("opentimestamps.core.serialize"),
    )


def test_the_reader_agrees_with_the_reference_library() -> None:
    timestamp, op, notary, serialize = _reference()
    digest = hashlib.sha256(b"ledger event preimage").digest()
    detached = timestamp.DetachedTimestampFile(op.OpSHA256(), timestamp.Timestamp(digest))
    tip = detached.timestamp.ops.add(op.OpAppend(b"\x07" * 16)).ops.add(op.OpSHA256())
    tip.attestations.add(notary.PendingAttestation(CALENDAR))
    tip.attestations.add(notary.PendingAttestation("https://finney.calendar.eternitywall.com"))
    tip.attestations.add(notary.BitcoinBlockHeaderAttestation(915_123))
    context = serialize.BytesSerializationContext()
    detached.serialize(context)
    summary = read_proof(context.getbytes())
    assert summary.digest == digest.hex()
    assert set(summary.calendars) == {CALENDAR, "https://finney.calendar.eternitywall.com"}
    assert summary.bitcoin_heights == (915_123,)

    # and the fake's proofs are proofs the reference library reads the same way
    ours = _proof(digest, calendars=(CALENDAR,), bitcoin=700_000)
    parsed = timestamp.DetachedTimestampFile.deserialize(
        serialize.BytesDeserializationContext(ours)
    )
    assert parsed.file_digest == digest
    kinds = {type(att).__name__ for _, att in parsed.timestamp.all_attestations()}
    assert kinds == {"PendingAttestation", "BitcoinBlockHeaderAttestation"}


# --- the production runner ------------------------------------------------------------------------


def test_the_runner_starts_only_ots(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only starts the ots command"):
        subprocess_ots_runner(["python", "-c", "0"], tmp_path, 5.0)


def test_the_runner_reports_a_missing_ots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(anchor, "_reference_client_here", lambda: False)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    code, _, err = subprocess_ots_runner(["ots", "--version"], tmp_path, 5.0)
    assert code == 127
    assert "not installed" in err


def test_the_runner_runs_without_credentials_and_times_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The Python interpreter stands in for an ots executable on PATH: the one program a test may
    # start. The route through an installed client is exercised by the two tests after this one.
    monkeypatch.setattr(anchor, "_reference_client_here", lambda: False)
    monkeypatch.setattr(shutil, "which", lambda name: sys.executable)
    monkeypatch.setenv("BITGET_API_KEY", "must-not-reach-the-child")
    script = "import os; print(os.environ.get('BITGET_API_KEY')); print(os.getcwd())"
    code, out, _ = subprocess_ots_runner(["ots", "-c", script], tmp_path, 60.0)
    assert code == 0
    assert out.splitlines() == ["None", str(tmp_path)]
    code, _, err = subprocess_ots_runner(
        ["ots", "-c", "import time; time.sleep(30)"], tmp_path, 0.5
    )
    assert code == 124
    assert "did not finish" in err


def _require_reference_client() -> None:
    # Located, not imported: importing otsclient in-process is what fails on Windows (anchor.py).
    for name in ("otsclient", "opentimestamps"):
        if importlib.util.find_spec(name) is None:
            pytest.skip(f"{name} is not installed (pip install -e .[anchor])")


def test_the_runner_starts_the_installed_reference_client(tmp_path: Path) -> None:
    _require_reference_client()
    code, out, err = subprocess_ots_runner(["ots", "--version"], tmp_path, 120.0)
    assert code == 0, err
    assert (out + err).strip().startswith("v0.")


# --- end to end with the reference client ---------------------------------------------------------


class _Calendar:
    """An OpenTimestamps calendar on 127.0.0.1, speaking ``opentimestamps/calendar.py``'s protocol:
    ``POST /digest`` answers a pending timestamp; ``GET /timestamp/<hex>`` answers 404 until the
    test "mines" the block, then the Merkle path to a Bitcoin-attested root."""

    def __init__(self) -> None:
        self.timestamp, self.op, self.notary, self.serialize = _reference()
        self.commitments: set[bytes] = set()
        self.mined = False
        calendar = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                digest = self.rfile.read(int(self.headers["Content-Length"]))
                stamp_ = calendar.timestamp.Timestamp(digest)
                tip = stamp_.ops.add(calendar.op.OpAppend(b"calendar-nonce")).ops.add(
                    calendar.op.OpSHA256()
                )
                tip.attestations.add(calendar.notary.PendingAttestation(calendar.url))
                calendar.commitments.add(tip.msg)
                self._answer(200, calendar.encode(stamp_))

            def do_GET(self) -> None:
                commitment = bytes.fromhex(self.path.rsplit("/", 1)[-1])
                if not calendar.mined or commitment not in calendar.commitments:
                    self._answer(404, b"Pending confirmation in Bitcoin blockchain")
                    return
                # A real calendar answers with the Merkle path from the commitment to the root it
                # published, the Bitcoin attestation sitting on that root.
                stamp_ = calendar.timestamp.Timestamp(commitment)
                root = stamp_.ops.add(calendar.op.OpAppend(b"merkle-sibling")).ops.add(
                    calendar.op.OpSHA256()
                )
                root.attestations.add(calendar.notary.BitcoinBlockHeaderAttestation(915_123))
                self._answer(200, calendar.encode(stamp_))

            def _answer(self, status: int, body: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                return

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def encode(self, stamp_: Any) -> bytes:
        context = self.serialize.BytesSerializationContext()
        stamp_.serialize(context)
        return bytes(context.getbytes())

    def __enter__(self) -> "_Calendar":
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.shutdown()
        self.server.server_close()


def _loopback_runner(url: str) -> anchor.OtsRunner:
    """The production runner, with the client's own options pointing it at the loopback calendar
    only: no default calendars, no whitelist but ours, no user cache."""

    def run(argv: Sequence[str], cwd: Path, timeout_s: float) -> tuple[int, str, str]:
        assert argv[0] == "ots"
        command, rest = argv[1], list(argv[2:])
        own = ["-c", url, "-m", "1"] if command == "stamp" else []
        return subprocess_ots_runner(
            [
                "ots",
                "--no-cache",
                "--no-default-whitelist",
                "--whitelist",
                url,
                command,
                *own,
                *rest,
            ],
            cwd,
            timeout_s,
        )

    return run


@pytest.fixture
def calendar(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Calendar]:
    _require_reference_client()
    for name in list(os.environ):
        if "proxy" in name.lower():
            monkeypatch.delenv(name)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    with _Calendar() as running:
        yield running


def test_stamp_and_upgrade_with_the_reference_client(
    calendar: _Calendar,
    event: LedgerEvent,
    store: FileBlobStore,
    tmp_path: Path,
    clock: ManualClock,
) -> None:
    runner = _loopback_runner(calendar.url)
    workdir = tmp_path / "work"
    record = stamp(event, runner=runner, blobs=store, workdir=workdir, clock=clock)
    assert record.status == "submitted", record.detail
    assert record.ots_blob is not None
    assert calendar.url in record.detail

    proof = store.get(record.ots_blob.sha256)
    parsed = calendar.timestamp.DetachedTimestampFile.deserialize(
        calendar.serialize.BytesDeserializationContext(proof)
    )
    assert parsed.file_digest.hex() == event.hash  # the reference client agrees: it is our hash

    clock.advance(timedelta(hours=1))
    early = upgrade(record, runner=runner, blobs=store, workdir=workdir, clock=clock)
    assert early.status == "submitted"
    assert "Timestamp not complete" in early.detail

    calendar.mined = True
    clock.advance(timedelta(hours=5))
    done = upgrade(early, runner=runner, blobs=store, workdir=workdir, clock=clock)
    assert done.status == "upgraded", done.detail
    assert done.ots_blob is not None
    completed = calendar.timestamp.DetachedTimestampFile.deserialize(
        calendar.serialize.BytesDeserializationContext(store.get(done.ots_blob.sha256))
    )
    heights = {
        att.height
        for _, att in completed.timestamp.all_attestations()
        if type(att).__name__ == "BitcoinBlockHeaderAttestation"
    }
    assert heights == {915_123}
    assert list(workdir.iterdir()) == []
