"""The hash chain: append and verify, tamper evidence, truncation, the lock, typed payloads."""

import hashlib
import itertools
import json
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from helpers import T0, client_oid_for
from sentiment_agent.clock import ManualClock
from sentiment_agent.hashing import ZERO_HASH, canonical_json, sha256_hex
from sentiment_agent.ledger import chain
from sentiment_agent.ledger.blobs import FileBlobStore
from sentiment_agent.ledger.chain import (
    HashChainLedger,
    LedgerError,
    LedgerLocked,
    event_hash,
    event_preimage,
    ledger_path,
    policy_hash_after,
    referenced_blobs,
    verify_file,
)
from sentiment_agent.types import (
    AnchorRecord,
    DryRunPreview,
    EventKind,
    LedgerEvent,
    LedgerReader,
    LedgerWriter,
    Note,
    RunMode,
    Trigger,
    TriggerKind,
    parse_payload,
)

ROOT = Path(__file__).resolve().parents[2]
_HASHED = ("seq", "ts", "kind", "mode", "payload", "blobs", "prev_hash")


# --- helpers --------------------------------------------------------------------------------------


def _note(text: str, at: datetime = T0) -> Note:
    return Note(at=at, author="system", text=text)


def _ledger(
    tmp_path: Path, clock: ManualClock, mode: RunMode = RunMode.SIMULATED
) -> HashChainLedger:
    return HashChainLedger(ledger_path(tmp_path, mode), mode=mode, clock=clock)


def _filled(tmp_path: Path, clock: ManualClock, n: int) -> HashChainLedger:
    ledger = _ledger(tmp_path, clock)
    for i in range(n):
        clock.advance(timedelta(minutes=1))
        ledger.append(EventKind.NOTE, _note(f"note {i}"))
    return ledger


def _lines(path: Path) -> list[bytes]:
    data = path.read_bytes()
    assert data.endswith(b"\n")
    return data.split(b"\n")[:-1]


def _write_lines(path: Path, lines: list[bytes]) -> None:
    path.write_bytes(b"".join(line + b"\n" for line in lines))


def _canonical(obj: dict[str, Any]) -> bytes:
    """Canonical JSON with the standard library only, as a judge's verifier would write it."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _rehash(obj: dict[str, Any]) -> dict[str, Any]:
    """What a careful forger does: recompute the edited event's own hash."""
    obj["hash"] = hashlib.sha256(_canonical({k: obj[k] for k in _HASHED})).hexdigest()
    return obj


def _lock_of(ledger: HashChainLedger) -> Path:
    return ledger.path.with_name(ledger.path.name + ".lock")


def _age(path: Path, seconds: float) -> None:
    old = time.time() - seconds
    os.utime(path, (old, old))


# --- appending and reading ------------------------------------------------------------------------


def test_each_event_links_to_the_one_before(tmp_path: Path, clock: ManualClock) -> None:
    ledger = _filled(tmp_path, clock, 4)
    events = list(ledger.events())
    assert [e.seq for e in events] == [0, 1, 2, 3]
    assert events[0].prev_hash == ZERO_HASH
    for before, after in itertools.pairwise(events):
        assert after.prev_hash == before.hash
        assert after.ts > before.ts
    assert all(e.mode is RunMode.SIMULATED for e in events)
    assert ledger.head() == events[-1]
    assert [parse_payload(e) for e in events] == [_note(f"note {i}") for i in range(4)]

    result = ledger.verify()
    assert result.intact
    assert (result.events, result.first_break_at, result.truncated) == (4, None, False)
    assert result.head_hash == events[-1].hash
    assert result.anchor.startswith("the head anchor agrees: 4 events")
    assert result.break_reason is None


def test_the_hash_is_the_documented_formula(tmp_path: Path, clock: ManualClock) -> None:
    ledger = _filled(tmp_path, clock, 2)
    for event in ledger.events():
        fields = {name: getattr(event, name) for name in _HASHED}
        assert event_hash(**fields) == event.hash
        assert sha256_hex(event_preimage(event)) == event.hash
        assert len(event.hash) == 64


def test_every_line_is_canonical_and_recomputes_with_the_standard_library(
    tmp_path: Path, clock: ManualClock
) -> None:
    ledger = _filled(tmp_path, clock, 3)
    for line, event in zip(_lines(ledger.path), ledger.events(), strict=True):
        assert line == canonical_json(event)
        obj = json.loads(line)
        body = {name: obj[name] for name in _HASHED}
        assert hashlib.sha256(_canonical(body)).hexdigest() == obj["hash"] == event.hash


def test_event_hash_refuses_what_it_cannot_commit_to() -> None:
    fields: dict[str, Any] = {
        "seq": 0,
        "ts": T0,
        "kind": EventKind.NOTE,
        "mode": RunMode.SIMULATED,
        "payload": {},
        "blobs": (),
        "prev_hash": ZERO_HASH,
    }
    assert len(event_hash(**fields)) == 64
    with pytest.raises(ValueError, match="aware"):
        event_hash(**{**fields, "ts": datetime(2026, 9, 23, 13, 0)})  # noqa: DTZ001
    with pytest.raises(ValueError, match="64-character"):
        event_hash(**{**fields, "prev_hash": "0" * 16})
    with pytest.raises(ValueError, match="non-negative"):
        event_hash(**{**fields, "seq": -1})


def test_ledgers_satisfy_the_contract_protocols(tmp_path: Path, clock: ManualClock) -> None:
    ledger = _ledger(tmp_path, clock)
    writer: LedgerWriter = ledger
    reader: LedgerReader = ledger
    assert writer.mode is RunMode.SIMULATED
    writer.append(EventKind.NOTE, _note("typed"))
    assert reader.head() is not None


def test_another_writer_s_events_are_seen(tmp_path: Path, clock: ManualClock) -> None:
    first = _ledger(tmp_path, clock)
    second = _ledger(tmp_path, clock)
    first.append(EventKind.NOTE, _note("from the first"))
    assert [e.seq for e in second.events()] == [0]
    appended = second.append(EventKind.NOTE, _note("from the second"))
    assert appended.seq == 1
    head = first.head()
    assert head is not None
    assert head.hash == appended.hash
    third = first.append(EventKind.NOTE, _note("the first again"))
    assert (third.seq, third.prev_hash) == (2, appended.hash)
    assert first.verify().intact


def test_events_can_be_filtered_by_kind(tmp_path: Path, clock: ManualClock) -> None:
    ledger = _ledger(tmp_path, clock)
    ledger.append(EventKind.NOTE, _note("a"))
    trigger = Trigger(
        trigger_id="t1",
        kind=TriggerKind.HEARTBEAT_US_OPEN,
        fired_at=T0,
        symbols=("NVDAUSDT",),
        detail="US open",
        source="schedule",
    )
    ledger.append(EventKind.TRIGGER, trigger)
    ledger.append(EventKind.NOTE, _note("b"))
    assert [e.seq for e in ledger.events(frozenset({EventKind.NOTE}))] == [0, 2]
    [only] = ledger.events(frozenset({EventKind.TRIGGER}))
    assert parse_payload(only) == trigger
    assert list(ledger.events(frozenset({EventKind.FILL}))) == []


def test_an_empty_ledger(tmp_path: Path, clock: ManualClock) -> None:
    ledger = HashChainLedger(
        tmp_path / "nowhere" / "simulated.jsonl", mode=RunMode.SIMULATED, clock=clock
    )
    assert ledger.head() is None
    assert list(ledger.events()) == []
    result = ledger.verify()
    assert (result.events, result.intact, result.head_hash) == (0, True, None)
    assert not (tmp_path / "nowhere").exists()  # verifying creates nothing


def test_ledger_path_is_one_file_per_mode(tmp_path: Path) -> None:
    assert ledger_path(tmp_path, RunMode.PAPER) == tmp_path / "var" / "ledger" / "paper.jsonl"
    assert ledger_path(tmp_path, RunMode.DRYRUN).name == "dryrun.jsonl"


def test_line_separators_inside_text_do_not_split_events(
    tmp_path: Path, clock: ManualClock
) -> None:
    # U+2028, U+2029 and U+0085 end a line for str.splitlines(); JSON leaves them unescaped when
    # ensure_ascii is off. The log is split on b"\n" only, so they stay inside their event.
    separators = [chr(0x2028), chr(0x2029), chr(0x85), chr(0x0D), chr(0x0A), chr(0x0C)]
    text = "one " + " two ".join(separators) + " end"
    ledger = _ledger(tmp_path, clock)
    ledger.append(EventKind.NOTE, _note(text))
    ledger.append(EventKind.NOTE, _note("after"))
    reopened = _ledger(tmp_path, clock)
    assert [parse_payload(e) for e in reopened.events()] == [_note(text), _note("after")]
    assert reopened.verify().intact


# --- time and mode --------------------------------------------------------------------------------


class _ScriptedClock:
    def __init__(self, *times: datetime) -> None:
        self._times = list(times)

    def now(self) -> datetime:
        return self._times.pop(0)


def test_time_never_runs_backwards_in_the_log(tmp_path: Path) -> None:
    later = T0 + timedelta(hours=1)
    ledger = HashChainLedger(
        tmp_path / "simulated.jsonl", mode=RunMode.SIMULATED, clock=_ScriptedClock(later, T0)
    )
    first = ledger.append(EventKind.NOTE, _note("clock at 14:00"))
    second = ledger.append(EventKind.NOTE, _note("clock stepped back to 13:00"))
    assert first.ts == second.ts == later
    assert ledger.verify().intact


def test_a_backdated_event_is_a_break(tmp_path: Path, clock: ManualClock) -> None:
    ledger = _filled(tmp_path, clock, 3)
    lines = _lines(ledger.path)
    obj = json.loads(lines[2])
    obj["ts"] = "2026-09-23T12:00:00Z"
    lines[2] = _canonical(_rehash(obj))
    _write_lines(ledger.path, lines)
    result = verify_file(ledger.path)
    assert result.first_break_at == 2
    assert result.break_reason == "seq 2: ts runs backwards from seq 1"


def test_a_utc_offset_clock_is_stored_as_utc(tmp_path: Path) -> None:
    offset = datetime(2026, 9, 23, 18, 30, tzinfo=UTC).astimezone()
    ledger = HashChainLedger(
        tmp_path / "simulated.jsonl", mode=RunMode.SIMULATED, clock=_ScriptedClock(offset)
    )
    event = ledger.append(EventKind.NOTE, _note("local clock"))
    assert event.ts == datetime(2026, 9, 23, 18, 30, tzinfo=UTC)
    assert event.ts.utcoffset() == timedelta(0)


def test_a_naive_clock_is_refused(tmp_path: Path) -> None:
    naive = _ScriptedClock(datetime(2026, 9, 23, 13, 0))  # noqa: DTZ001
    ledger = HashChainLedger(tmp_path / "simulated.jsonl", mode=RunMode.SIMULATED, clock=naive)
    with pytest.raises(LedgerError, match="naive"):
        ledger.append(EventKind.NOTE, _note("x"))


def test_a_file_named_for_a_mode_opens_only_as_that_mode(
    tmp_path: Path, clock: ManualClock
) -> None:
    with pytest.raises(LedgerError, match="paper ledger"):
        HashChainLedger(tmp_path / "paper.jsonl", mode=RunMode.SIMULATED, clock=clock)
    HashChainLedger(tmp_path / "scratch.jsonl", mode=RunMode.PAPER, clock=clock)


def _forge_next(head: LedgerEvent, *, mode: RunMode, payload: dict[str, Any]) -> bytes:
    """A correctly linked and hashed line, as anyone who knows the formula could write it."""
    fields: dict[str, Any] = {
        "seq": head.seq + 1,
        "ts": head.ts,
        "kind": EventKind.NOTE,
        "mode": mode,
        "payload": payload,
        "blobs": (),
        "prev_hash": head.hash,
    }
    return canonical_json(LedgerEvent(**fields, hash=event_hash(**fields)))


def test_an_event_of_another_mode_is_a_break(tmp_path: Path, clock: ManualClock) -> None:
    ledger = HashChainLedger(tmp_path / "mixed.jsonl", mode=RunMode.SIMULATED, clock=clock)
    ledger.append(EventKind.NOTE, _note("simulated"))
    head = ledger.append(EventKind.NOTE, _note("simulated"))
    with ledger.path.open("ab") as handle:
        body = _note("a paper event").model_dump(mode="json")
        handle.write(_forge_next(head, mode=RunMode.PAPER, payload=body) + b"\n")
    result = verify_file(ledger.path)
    assert result.first_break_at == 2
    assert result.break_reason == "seq 2 is a paper event in a simulated ledger"
    with pytest.raises(LedgerError, match="paper event in a simulated ledger"):
        HashChainLedger(ledger.path, mode=RunMode.SIMULATED, clock=clock).head()


def test_a_file_named_for_a_mode_holding_another_mode_does_not_verify(
    tmp_path: Path, clock: ManualClock
) -> None:
    ledger = HashChainLedger(tmp_path / "scratch.jsonl", mode=RunMode.SIMULATED, clock=clock)
    ledger.append(EventKind.NOTE, _note("simulated"))
    renamed = tmp_path / "paper.jsonl"
    renamed.write_bytes(ledger.path.read_bytes())
    result = verify_file(renamed)
    assert (result.first_break_at, result.intact) == (0, False)


# --- typed payloads and attached blobs ------------------------------------------------------------


class _LoudNote(Note):
    volume: int = 11


def test_a_payload_must_be_the_model_registered_for_its_kind(
    tmp_path: Path, clock: ManualClock
) -> None:
    ledger = _ledger(tmp_path, clock)
    with pytest.raises(LedgerError, match="carries a Trigger, not a Note"):
        ledger.append(EventKind.TRIGGER, _note("wrong kind"))
    with pytest.raises(LedgerError, match="carries a Note, not a _LoudNote"):
        ledger.append(EventKind.NOTE, _LoudNote(at=T0, author="system", text="subclass"))
    assert ledger.head() is None


def test_a_payload_that_would_not_survive_the_log_is_refused(
    tmp_path: Path, clock: ManualClock
) -> None:
    preview = DryRunPreview(
        client_oid=client_oid_for("preview"),
        operation_id="uta_place_order",
        method="POST",
        path="/api/v3/trade/place-order",
        would_send={"legs": ("a", "b")},  # a tuple reads back as a list
        argv=("order", "--dry-run"),
        captured_at=T0,
        blob=None,
    )
    ledger = _ledger(tmp_path, clock)
    with pytest.raises(LedgerError, match="survive"):
        ledger.append(EventKind.ORDER_PREVIEW, preview)
    fine = preview.model_copy(update={"would_send": {"legs": ["a", "b"]}})
    assert parse_payload(ledger.append(EventKind.ORDER_PREVIEW, fine)) == fine


def test_a_hand_written_payload_of_the_wrong_shape_does_not_verify(
    tmp_path: Path, clock: ManualClock
) -> None:
    ledger = _ledger(tmp_path, clock)
    head = ledger.append(EventKind.NOTE, _note("genuine"))
    forged = _forge_next(head, mode=RunMode.SIMULATED, payload={"text": "no author"})
    with ledger.path.open("ab") as handle:
        handle.write(forged + b"\n")
    result = ledger.verify()
    assert result.first_break_at == 1
    assert result.break_reason is not None
    assert "not a valid Note" in result.break_reason


def test_attached_blobs_must_be_references_attached_once(
    tmp_path: Path, clock: ManualClock
) -> None:
    store = FileBlobStore(tmp_path / "blobs")
    ref = store.put(b"raw", "application/json")
    ledger = _ledger(tmp_path, clock)
    with pytest.raises(LedgerError, match="twice"):
        ledger.append(EventKind.NOTE, _note("x"), blobs=(ref, ref))
    with pytest.raises(LedgerError, match="BlobRef"):
        ledger.append(EventKind.NOTE, _note("x"), blobs=(ref.model_dump(),))  # type: ignore[arg-type]
    assert ledger.append(EventKind.NOTE, _note("x"), blobs=(ref,)).blobs == (ref,)


# --- tamper evidence ------------------------------------------------------------------------------


@pytest.mark.parametrize("index", range(5))
def test_editing_any_line_locates_the_first_break(
    tmp_path: Path, clock: ManualClock, index: int
) -> None:
    ledger = _filled(tmp_path, clock, 5)
    lines = _lines(ledger.path)
    obj = json.loads(lines[index])
    obj["payload"]["text"] = "a kinder history"
    lines[index] = _canonical(obj)
    _write_lines(ledger.path, lines)
    for result in (ledger.verify(), verify_file(ledger.path)):
        assert not result.intact
        assert result.first_break_at == index
        assert result.break_reason == f"seq {index}: the event does not hash to its hash field"
        assert result.events == 5


@pytest.mark.parametrize("index", range(4))
def test_a_rehashed_edit_breaks_the_next_link(
    tmp_path: Path, clock: ManualClock, index: int
) -> None:
    ledger = _filled(tmp_path, clock, 5)
    lines = _lines(ledger.path)
    obj = json.loads(lines[index])
    obj["payload"]["text"] = "a kinder history"
    lines[index] = _canonical(_rehash(obj))
    _write_lines(ledger.path, lines)
    result = ledger.verify()
    assert result.first_break_at == index + 1
    assert result.break_reason == f"seq {index + 1}: prev_hash is not the hash of seq {index}"


def test_a_rehashed_edit_of_the_last_line_is_caught_by_the_anchor(
    tmp_path: Path, clock: ManualClock
) -> None:
    ledger = _filled(tmp_path, clock, 5)
    lines = _lines(ledger.path)
    obj = json.loads(lines[-1])
    obj["payload"]["text"] = "a kinder ending"
    lines[-1] = _canonical(_rehash(obj))
    _write_lines(ledger.path, lines)
    result = ledger.verify()
    assert result.first_break_at is None
    assert result.truncated
    assert not result.intact
    assert "the tail was replaced" in result.anchor


def test_respelling_a_value_is_still_an_edit(tmp_path: Path, clock: ManualClock) -> None:
    ledger = _filled(tmp_path, clock, 3)
    lines = _lines(ledger.path)
    lines[1] = lines[1].replace(b'":', b'": ', 1)
    _write_lines(ledger.path, lines)
    result = ledger.verify()
    assert result.first_break_at == 1
    assert result.break_reason is not None
    assert "canonical form" in result.break_reason


def test_removing_a_line_from_the_middle_is_a_break(tmp_path: Path, clock: ManualClock) -> None:
    ledger = _filled(tmp_path, clock, 5)
    lines = _lines(ledger.path)
    del lines[2]
    _write_lines(ledger.path, lines)
    result = ledger.verify()
    assert (result.first_break_at, result.break_reason) == (2, "line 2 carries seq 3")


def test_a_line_that_is_not_an_event_is_a_break(tmp_path: Path, clock: ManualClock) -> None:
    ledger = _filled(tmp_path, clock, 3)
    lines = _lines(ledger.path)
    lines.insert(1, b'{"hello": "world"}')
    _write_lines(ledger.path, lines)
    result = ledger.verify()
    assert result.first_break_at == 1
    assert result.break_reason is not None
    assert "not a ledger event" in result.break_reason
    assert result.events == 3  # the three real events still parse


def test_a_reader_refuses_a_broken_log(tmp_path: Path, clock: ManualClock) -> None:
    ledger = _filled(tmp_path, clock, 3)
    lines = _lines(ledger.path)
    obj = json.loads(lines[1])
    obj["payload"]["text"] = "edited"
    lines[1] = _canonical(obj)
    _write_lines(ledger.path, lines)
    fresh = _ledger(tmp_path, clock)
    with pytest.raises(LedgerError, match="seq 1: the event does not hash"):
        list(fresh.events())
    with pytest.raises(LedgerError, match="does not verify"):
        fresh.append(EventKind.NOTE, _note("on top of a break"))


def test_a_line_changed_under_a_reader_is_refused(tmp_path: Path, clock: ManualClock) -> None:
    ledger = _filled(tmp_path, clock, 3)
    assert ledger.head() is not None
    lines = _lines(ledger.path)
    obj = json.loads(lines[-1])
    obj["payload"]["text"] = "rewritten"
    lines[-1] = _canonical(_rehash(obj))
    _write_lines(ledger.path, lines)
    with pytest.raises(LedgerError, match=r"seq 2 of simulated\.jsonl changed after it was read"):
        ledger.head()


# --- truncation and the head anchor ---------------------------------------------------------------


def test_deleting_the_tail_is_flagged_as_truncation(tmp_path: Path, clock: ManualClock) -> None:
    ledger = _filled(tmp_path, clock, 5)
    _write_lines(ledger.path, _lines(ledger.path)[:3])
    fresh = _ledger(tmp_path, clock)
    for result in (fresh.verify(), verify_file(ledger.path)):
        assert (result.events, result.first_break_at) == (3, None)  # what is left still links
        assert result.truncated
        assert not result.intact
        assert result.anchor == (
            "the head anchor records 5 events and 3 are present: 2 removed from the end"
        )


def test_a_writer_refuses_to_extend_a_truncated_log(tmp_path: Path, clock: ManualClock) -> None:
    ledger = _filled(tmp_path, clock, 5)
    anchor = ledger.anchor_path.read_bytes()
    _write_lines(ledger.path, _lines(ledger.path)[:3])
    kept = ledger.path.read_bytes()
    with pytest.raises(LedgerError, match="shrank"):
        ledger.head()
    fresh = _ledger(tmp_path, clock)
    with pytest.raises(LedgerError, match="2 removed from the end"):
        fresh.append(EventKind.NOTE, _note("papering over it"))
    assert ledger.path.read_bytes() == kept
    assert ledger.anchor_path.read_bytes() == anchor


def test_a_deleted_log_is_not_restarted(tmp_path: Path, clock: ManualClock) -> None:
    ledger = _filled(tmp_path, clock, 3)
    ledger.path.unlink()
    with pytest.raises(LedgerError, match="disappeared"):
        ledger.head()
    fresh = _ledger(tmp_path, clock)
    assert list(fresh.events()) == []
    with pytest.raises(LedgerError, match="3 removed from the end"):
        fresh.append(EventKind.NOTE, _note("a new beginning"))
    result = fresh.verify()
    assert (result.events, result.truncated, result.intact) == (0, True, False)


def test_a_replaced_tail_is_flagged_and_not_extended(tmp_path: Path, clock: ManualClock) -> None:
    ledger = _filled(tmp_path, clock, 3)
    forge_dir = tmp_path / "forge"
    forge_dir.mkdir()
    forged = forge_dir / "scratch.jsonl"
    _write_lines(forged, _lines(ledger.path)[:2])
    forger = HashChainLedger(forged, mode=RunMode.SIMULATED, clock=clock)
    forger.append(EventKind.NOTE, _note("a kinder seq 2"))
    ledger.path.write_bytes(forged.read_bytes())  # the original anchor stays beside it

    result = _ledger(tmp_path, clock).verify()
    assert (result.events, result.first_break_at) == (3, None)
    assert result.truncated
    assert "the tail was replaced" in result.anchor

    forger.append(EventKind.NOTE, _note("and one more, to look longer than the anchor"))
    ledger.path.write_bytes(forged.read_bytes())
    result = _ledger(tmp_path, clock).verify()
    assert (result.events, result.first_break_at) == (4, None)
    assert result.truncated
    assert "the tail was replaced" in result.anchor
    with pytest.raises(LedgerError, match="the tail was replaced"):
        _ledger(tmp_path, clock).append(EventKind.NOTE, _note("on the forgery"))


def test_an_anchor_left_behind_by_a_crash_is_tolerated(tmp_path: Path, clock: ManualClock) -> None:
    ledger = _filled(tmp_path, clock, 2)
    earlier = ledger.anchor_path.read_bytes()
    ledger.append(EventKind.NOTE, _note("written, then the writer died before its anchor"))
    ledger.anchor_path.write_bytes(earlier)
    result = ledger.verify()
    assert result.intact
    assert "1 event(s) behind" in result.anchor
    ledger.append(EventKind.NOTE, _note("the next append catches the anchor up"))
    assert ledger.verify().anchor.startswith("the head anchor agrees: 4 events")


def test_a_missing_anchor_is_reported_honestly(tmp_path: Path, clock: ManualClock) -> None:
    ledger = _filled(tmp_path, clock, 2)
    ledger.anchor_path.unlink()
    result = ledger.verify()
    assert result.intact  # nothing on disk contradicts the log ...
    assert "cannot be ruled out" in result.anchor  # ... and the report says what that is worth


@pytest.mark.parametrize(
    ("content", "note"),
    [
        (b"{not json", "unreadable (not JSON"),
        (b"[1, 2]", "unreadable (not a JSON object)"),
        (b'{"events": 0, "head_hash": null, "mode": "simulated"}', "count is missing"),
        (b'{"events": 2, "head_hash": "abc", "mode": "simulated"}', "head hash is missing"),
    ],
)
def test_an_unreadable_anchor_is_not_trusted(
    tmp_path: Path, clock: ManualClock, content: bytes, note: str
) -> None:
    ledger = _filled(tmp_path, clock, 2)
    ledger.anchor_path.write_bytes(content)
    result = ledger.verify()
    assert result.truncated
    assert not result.intact
    assert note in result.anchor
    with pytest.raises(LedgerError, match="unreadable"):
        ledger.append(EventKind.NOTE, _note("x"))


def test_an_anchor_from_another_ledger_is_not_trusted(tmp_path: Path, clock: ManualClock) -> None:
    ledger = _filled(tmp_path, clock, 2)
    head = ledger.head()
    assert head is not None
    ledger.anchor_path.write_text(
        json.dumps({"events": 2, "head_hash": head.hash, "mode": "paper"}), encoding="utf-8"
    )
    result = ledger.verify()
    assert result.truncated
    assert "belongs to a paper ledger" in result.anchor


def test_an_append_whose_anchor_cannot_be_written_still_reports_its_event(
    tmp_path: Path,
    clock: ManualClock,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    ledger = _filled(tmp_path, clock, 1)

    def disk_full(path: Path, data: bytes) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(chain, "_atomic_write", disk_full)
    with caplog.at_level(logging.ERROR, logger="sentiment_agent.ledger.chain"):
        event = ledger.append(EventKind.NOTE, _note("durable"))
    assert event.seq == 1
    assert "could not be updated" in caplog.text
    assert "1 event(s) behind" in ledger.verify().anchor
    monkeypatch.undo()
    ledger.append(EventKind.NOTE, _note("caught up"))
    assert ledger.verify().anchor.startswith("the head anchor agrees: 3 events")


# --- torn writes ----------------------------------------------------------------------------------


def test_a_torn_last_line_stops_writers_and_fails_verification(
    tmp_path: Path, clock: ManualClock
) -> None:
    ledger = _filled(tmp_path, clock, 3)
    whole = _lines(ledger.path)
    with ledger.path.open("ab") as handle:
        handle.write(whole[0][:40])  # a writer died mid-line
    result = ledger.verify()
    assert result.first_break_at == 3
    assert result.break_reason is not None
    assert "no line end" in result.break_reason
    assert [e.seq for e in _ledger(tmp_path, clock).events()] == [0, 1, 2]  # readers skip it
    before = ledger.path.read_bytes()
    with pytest.raises(LedgerError, match="stopped mid-line"):
        ledger.append(EventKind.NOTE, _note("after the tear"))
    assert ledger.path.read_bytes() == before


# --- blobs ----------------------------------------------------------------------------------------


def test_tampered_and_missing_blobs_are_found(tmp_path: Path, clock: ManualClock) -> None:
    store = FileBlobStore(tmp_path / "blobs")
    raw = store.put(b'{"code":"00000"}', "application/json")
    proof = store.put(b"\x00OpenTimestamps proof", "application/vnd.opentimestamps.v1")
    ledger = _ledger(tmp_path, clock)
    first = ledger.append(EventKind.NOTE, _note("carries a raw response"), blobs=(raw,))
    record = AnchorRecord(
        target_seq=0,
        target_hash=first.hash,
        submitted_at=T0,
        status="submitted",
        ots_blob=proof,
        detail="pending",
    )
    anchored = ledger.append(EventKind.ANCHOR, record)
    assert referenced_blobs(first) == (raw,)
    assert referenced_blobs(anchored) == (proof,)  # nested in the payload, not attached

    assert ledger.verify(store).intact
    store.path_of(raw.sha256).write_bytes(b'{"code":"40099"}')
    result = ledger.verify(store)
    assert (result.intact, result.first_break_at, result.missing_blobs) == (
        False,
        None,
        (raw.sha256,),
    )
    store.path_of(proof.sha256).unlink()
    expected = tuple(sorted((raw.sha256, proof.sha256)))
    assert ledger.verify(store).missing_blobs == expected
    assert verify_file(ledger.path, blobs_root=store.root).missing_blobs == expected
    unchecked = ledger.verify()
    assert unchecked.intact
    assert unchecked.missing_blobs == ()


# --- the writer lock ------------------------------------------------------------------------------


def test_a_second_writer_is_refused_while_the_lock_is_held(
    tmp_path: Path, clock: ManualClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(chain, "LOCK_WAIT_SECONDS", 0.2)
    ledger = _filled(tmp_path, clock, 1)
    lock = _lock_of(ledger)
    lock.write_text("pid=4242 host=elsewhere nonce=live", encoding="utf-8")
    before = ledger.path.read_bytes()
    with pytest.raises(LedgerLocked, match="pid=4242 host=elsewhere"):
        ledger.append(EventKind.NOTE, _note("beside another writer"))
    with pytest.raises(LedgerLocked):
        ledger.verify()
    assert ledger.path.read_bytes() == before
    assert lock.read_text(encoding="utf-8") == "pid=4242 host=elsewhere nonce=live"
    lock.unlink()
    assert ledger.append(EventKind.NOTE, _note("after it let go")).seq == 1
    assert not lock.exists()


def test_a_stale_lock_is_broken_and_reported(
    tmp_path: Path, clock: ManualClock, caplog: pytest.LogCaptureFixture
) -> None:
    ledger = _filled(tmp_path, clock, 1)
    lock = _lock_of(ledger)
    lock.write_text("pid=4242 host=gone nonce=dead", encoding="utf-8")
    _age(lock, 10 * chain.LOCK_STALE_SECONDS)
    with caplog.at_level(logging.WARNING, logger="sentiment_agent.ledger.chain"):
        event = ledger.append(EventKind.NOTE, _note("after the dead writer"))
    assert event.seq == 1
    assert ledger.stale_locks_broken == ("pid=4242 host=gone nonce=dead",)
    assert "broke a stale ledger lock" in caplog.text
    assert not lock.exists()
    assert ledger.verify().intact


def test_a_lock_being_broken_by_another_waiter_is_left_to_it(
    tmp_path: Path, clock: ManualClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(chain, "LOCK_WAIT_SECONDS", 0.2)
    ledger = _filled(tmp_path, clock, 1)
    lock = _lock_of(ledger)
    lock.write_text("pid=4242 host=gone nonce=dead", encoding="utf-8")
    _age(lock, 10 * chain.LOCK_STALE_SECONDS)
    breaker = ledger.path.with_name(ledger.path.name + ".lock.break")
    breaker.write_bytes(b"")  # a fresh break lock: someone else is breaking it right now
    with pytest.raises(LedgerLocked):
        ledger.append(EventKind.NOTE, _note("x"))
    assert lock.exists()
    assert ledger.stale_locks_broken == ()


def test_a_break_lock_left_by_a_dead_waiter_is_cleared(tmp_path: Path, clock: ManualClock) -> None:
    ledger = _filled(tmp_path, clock, 1)
    lock = _lock_of(ledger)
    lock.write_text("pid=4242 host=gone nonce=dead", encoding="utf-8")
    breaker = ledger.path.with_name(ledger.path.name + ".lock.break")
    breaker.write_bytes(b"")
    _age(lock, 10 * chain.LOCK_STALE_SECONDS)
    _age(breaker, 10 * chain.LOCK_STALE_SECONDS)
    assert ledger.append(EventKind.NOTE, _note("x")).seq == 1
    assert not lock.exists()
    assert not breaker.exists()


def test_a_writer_whose_lock_was_taken_writes_nothing(
    tmp_path: Path, clock: ManualClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = _filled(tmp_path, clock, 1)
    before = ledger.path.read_bytes()
    monkeypatch.setattr(chain._WriterLock, "held", lambda self: False)
    with pytest.raises(LedgerLocked, match="nothing was written"):
        ledger.append(EventKind.NOTE, _note("x"))
    assert ledger.path.read_bytes() == before


def test_a_holder_never_releases_a_lock_that_is_no_longer_its_own(tmp_path: Path) -> None:
    lock = chain._WriterLock(tmp_path / "simulated.jsonl")
    lock.acquire()
    assert lock.held()
    lock.path.write_text("pid=7 host=new-holder nonce=fresh", encoding="utf-8")
    assert not lock.held()
    lock.release()
    assert lock.path.read_text(encoding="utf-8") == "pid=7 host=new-holder nonce=fresh"


@pytest.mark.parametrize("shared", [False, True])
def test_writers_in_threads_never_share_a_seq(
    tmp_path: Path, clock: ManualClock, shared: bool
) -> None:
    common = _ledger(tmp_path, clock)
    errors: list[BaseException] = []

    def write(tag: str) -> None:
        ledger = common if shared else _ledger(tmp_path, clock)
        try:
            for i in range(10):
                ledger.append(EventKind.NOTE, _note(f"{tag} {i}"))
        except BaseException as exc:  # surfaced by the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(tag,)) for tag in "ABCD"]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    result = common.verify()
    assert (result.events, result.intact) == (40, True)
    notes = [parse_payload(e) for e in _ledger(tmp_path, clock).events()]
    assert sorted(n.text for n in notes if isinstance(n, Note)) == sorted(
        f"{tag} {i}" for tag in "ABCD" for i in range(10)
    )


_PROCESS_WRITER = """
import sys
from pathlib import Path
from sentiment_agent.clock import SystemClock
from sentiment_agent.ledger.chain import HashChainLedger
from sentiment_agent.types import EventKind, Note, RunMode

clock = SystemClock()
ledger = HashChainLedger(Path(sys.argv[1]), mode=RunMode.SIMULATED, clock=clock)
for i in range(int(sys.argv[3])):
    ledger.append(EventKind.NOTE, Note(at=clock.now(), author="system", text=f"{sys.argv[2]} {i}"))
"""


def test_writers_in_separate_processes_never_share_a_seq(tmp_path: Path) -> None:
    path = tmp_path / "var" / "ledger" / "simulated.jsonl"
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    writers = [
        subprocess.Popen(  # noqa: S603 - a Python interpreter with a fixed script
            [sys.executable, "-c", _PROCESS_WRITER, str(path), tag, "15"],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for tag in ("P", "Q")
    ]
    for writer in writers:
        _, err = writer.communicate(timeout=120)
        assert writer.returncode == 0, err.decode(errors="replace")
    clock = ManualClock(T0)
    result = HashChainLedger(path, mode=RunMode.SIMULATED, clock=clock).verify()
    assert (result.events, result.intact) == (30, True)


# --- the pre-registration fold --------------------------------------------------------------------


def test_policy_hash_after_ignores_everything_but_the_pre_registration(
    tmp_path: Path, clock: ManualClock
) -> None:
    ledger = _filled(tmp_path, clock, 3)
    assert policy_hash_after(ledger.events()) is None
