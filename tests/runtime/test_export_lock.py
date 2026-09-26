"""The export lock the page publisher shares (run 1's export failure, 2026-09-26).

``t2sa export`` holds ``var/run/export-<mode>.lock`` for the whole export;
``scripts/publish_site.ps1`` holds the same byte while it copies ``public/``. These tests check that
an export waits out a copy rather than losing its hour, and, on Windows, that the PowerShell lock
and the Python lock really exclude each other: the whole fix rests on .NET ``FileStream.Lock`` and
``msvcrt.locking`` taking the same byte-range lock, which is a claim about two runtimes and is run
here, not assumed. One
publish pass is run too, with the Vercel CLI shadowed: a half-published copy is never deployed, a
finished record is deployed once, and a running export is waited for rather than copied.
"""

import json
import shutil
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from sentiment_agent.clock import ManualClock
from sentiment_agent.runtime.wiring import InstanceLock, InstanceLocked
from sentiment_agent.types import RunMode

PUBLISHER = Path(__file__).resolve().parents[2] / "scripts" / "publish_site.ps1"
BUSY = "the paper export lock is held by another export or by the page publisher"


def _lock(path: Path) -> InstanceLock:
    clock = ManualClock(datetime(2026, 9, 26, tzinfo=UTC))
    return InstanceLock(path, mode=RunMode.PAPER, clock=clock, busy=BUSY)


def test_a_held_lock_is_waited_for_and_then_taken(tmp_path: Path) -> None:
    path = tmp_path / "var" / "run" / "export-paper.lock"
    holder = _lock(path)
    holder.acquire()
    threading.Timer(0.6, holder.release).start()
    waiter = _lock(path)
    started = time.monotonic()
    waiter.acquire(wait_s=10)
    try:
        assert waiter.held
        assert time.monotonic() - started >= 0.5
    finally:
        waiter.release()


def test_a_lock_held_past_the_wait_is_refused_with_what_holds_it(tmp_path: Path) -> None:
    path = tmp_path / "var" / "run" / "export-paper.lock"
    holder = _lock(path)
    holder.acquire()
    try:
        with pytest.raises(InstanceLocked) as refused:
            _lock(path).acquire(wait_s=0.6)
        assert BUSY in str(refused.value)
        assert "pid" in str(refused.value)
    finally:
        holder.release()


POWERSHELL = shutil.which("powershell")


@pytest.mark.skipif(sys.platform != "win32" or POWERSHELL is None, reason="Windows publisher")
class TestThePublishersLock:
    def _publisher(self, root: Path, body: str) -> subprocess.Popen[str]:
        assert POWERSHELL is not None
        command = f". '{PUBLISHER}' -Root '{root}' -LockWaitMinutes 0; {body}"
        return subprocess.Popen(  # noqa: S603 - this project's own script, allowed by conftest
            [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def test_a_copy_in_progress_holds_off_the_export(self, tmp_path: Path) -> None:
        publisher = self._publisher(
            tmp_path,
            "$s = Enter-ExportLock; if ($null -eq $s) { exit 2 }; 'locked'; "
            "[Console]::Out.Flush(); Start-Sleep -Seconds 3; Exit-ExportLock $s; exit 0",
        )
        assert publisher.stdout is not None
        assert publisher.stdout.readline().strip() == "locked"
        path = tmp_path / "var" / "run" / "export-paper.lock"
        with pytest.raises(InstanceLocked) as refused:
            _lock(path).acquire()
        assert BUSY in str(refused.value)
        export = _lock(path)
        export.acquire(wait_s=30)
        export.release()
        assert publisher.wait(30) == 0

    def test_a_running_export_holds_off_the_copy(self, tmp_path: Path) -> None:
        export = _lock(tmp_path / "var" / "run" / "export-paper.lock")
        export.acquire()
        try:
            publisher = self._publisher(
                tmp_path, "$s = Enter-ExportLock; if ($null -eq $s) { exit 2 }; exit 0"
            )
            assert publisher.wait(60) == 2
        finally:
            export.release()
        publisher = self._publisher(
            tmp_path, "$s = Enter-ExportLock; if ($null -eq $s) { exit 2 }; Exit-ExportLock $s"
        )
        assert publisher.wait(60) == 0


FAKE_VERCEL = "function vercel { 'Aliased     https://fake.example (no deploy)' }; "
"""Shadows the Vercel CLI inside the test's PowerShell (a function wins over a program), so a
publish pass that reaches its deploy step deploys nothing."""


@pytest.mark.skipif(sys.platform != "win32" or POWERSHELL is None, reason="Windows publisher")
class TestOnePublishPass:
    def _record(self, root: Path, *, summary_head: str, ledger_head: str) -> None:
        public = root / "public"
        (public / "blobs").mkdir(parents=True)
        (public / "blobs" / ("c" * 64)).write_bytes(b"evidence")
        summary = {"generated_at": "2026-09-26T01:00:30Z", "ledger": {"head_hash": summary_head}}
        (public / "summary.json").write_text(json.dumps(summary), "utf-8")
        anchor = {"events": 3, "head_hash": ledger_head, "head_seq": 2, "mode": "paper"}
        (public / "ledger.jsonl.head").write_text(json.dumps(anchor), "utf-8")

    def _run(self, root: Path, body: str) -> list[str]:
        assert POWERSHELL is not None
        command = f". '{PUBLISHER}' -Root '{root}' -LockWaitMinutes 0; {FAKE_VERCEL}{body}"
        done = subprocess.run(  # noqa: S603 - this project's own script, allowed by conftest
            [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert done.returncode == 0, done.stderr
        log = root / "var" / "logs" / "site.log"
        # The disk advisory depends on this machine's free space, not on the publish under test.
        return [
            line.split(" ", 1)[1]
            for line in log.read_text("utf-8").splitlines()
            if not line.split(" ", 1)[1].startswith("DISK LOW")
        ]

    def test_a_half_published_copy_is_not_deployed(self, tmp_path: Path) -> None:
        self._record(tmp_path, summary_head="a" * 64, ledger_head="b" * 64)
        lines = self._run(tmp_path, "Invoke-PublishOnce")
        assert lines == [
            f"not published: the copy is not one finished export (summary head {'a' * 64}, "
            f"ledger head {'b' * 64}, record generated 2026-09-26T01:00:30Z)"
        ]
        stage = tmp_path / "var" / "site" / "t2-sentiment-agent-live"
        assert (stage / "blobs" / ("c" * 64)).read_bytes() == b"evidence"

    def test_a_finished_record_is_deployed_once(self, tmp_path: Path) -> None:
        self._record(tmp_path, summary_head="d" * 64, ledger_head="d" * 64)
        lines = self._run(tmp_path, "Invoke-PublishOnce; Invoke-PublishOnce")
        assert lines == [
            "published record generated 2026-09-26T01:00:30Z Aliased     "
            "https://fake.example (no deploy)",
            "not redeployed: no new record since the one generated 2026-09-26T01:00:30Z",
        ]

    def test_an_export_in_progress_is_waited_for_not_copied(self, tmp_path: Path) -> None:
        self._record(tmp_path, summary_head="d" * 64, ledger_head="d" * 64)
        export = _lock(tmp_path / "var" / "run" / "export-paper.lock")
        export.acquire()
        try:
            lines = self._run(tmp_path, "Invoke-PublishOnce")
        finally:
            export.release()
        assert len(lines) == 1
        assert lines[0].startswith("not published: an export held ")
        assert not (tmp_path / "var" / "site" / "t2-sentiment-agent-live" / "summary.json").exists()
