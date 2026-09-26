"""Test-wide guarantees. These hold for every test in every module, and are not optional.

1. **No network.** Any socket connection to a non-loopback address fails the test, unless the test
   is marked ``live_public`` *and* the run was started with ``--run-live-public``. Live-public tests
   may only reach keyless public Bitget market endpoints. Qwen is never reachable from a test.
2. **No child processes but Python.** A child process is a network door the socket guard cannot
   see (``node`` running ``bgc``, ``curl``, ``twitter``, ``rdt``, ``ots``). Every module that runs
   a command takes an injectable runner (``BgcRunner``, ``CommandRunner``, ``OtsRunner``), and tests
   pass fakes. The only program a test may start is a Python interpreter (e.g. to run
   ``scripts/recompute.py`` the way a judge would), never through a shell. One exception: the
   standard library's ``platform`` module asks Windows for its version with the shell command
   ``ver``, which cannot reach a network, so ``platform.platform()`` keeps working in tests.
3. **No credentials.** Every ``BITGET_*`` variable is removed from the environment before each test,
   so no code path under test can pick up a key, live or Demo.
4. **No wall clock in assertions.** Use the ``clock`` fixture (a :class:`ManualClock`).
"""

import os
import re
import socket
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from sentiment_agent.clock import ManualClock

_LOOPBACK = {"127.0.0.1", "::1", "localhost"}
_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex
_real_create_connection = socket.create_connection
_real_popen_init = subprocess.Popen.__init__
_PYTHON_NAME = re.compile(r"python(?:3(?:\.\d+)*)?w?")
_OS_VERSION_QUERIES = frozenset({"ver", "cmd /c ver", "command /c ver"})
"""What ``platform._syscmd_ver`` runs on Windows (CPython 3.11 ``Lib/platform.py``)."""

_OS_PROCESSOR_QUERIES = frozenset({("uname", "-p")})
"""What ``platform._Processor.from_subprocess`` runs on Linux and macOS, without a shell (CPython
3.11 ``Lib/platform.py``). Found by CI: ``platform.platform()`` on ubuntu-latest was refused."""

_PUBLISHER_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "publish_site.ps1"
"""The one foreign program a test may start: Windows PowerShell dot-sourcing this project's own
page publisher, so ``tests/runtime/test_export_lock.py`` can check that its lock and the export's
exclude each other across the two runtimes. Dot-sourced, the script only defines functions; it
reaches no network and no account."""


def _is_publisher_check(args: Any, kwargs: dict[str, Any]) -> bool:
    if kwargs.get("shell") or not isinstance(args, list | tuple) or len(args) < 2:
        return False
    if Path(os.fsdecode(args[0])).name.lower() not in {"powershell", "powershell.exe"}:
        return False
    flags = [os.fsdecode(a) for a in args[1:-1]]
    command = os.fsdecode(args[-1])
    return flags == ["-NoProfile", "-ExecutionPolicy", "Bypass", "-Command"] and (
        command.startswith(f". '{_PUBLISHER_SCRIPT}' ")
    )


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-live-public",
        action="store_true",
        default=False,
        help="Allow tests marked live_public to call keyless public Bitget market endpoints.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-live-public"):
        return
    skip = pytest.mark.skip(reason="live_public: pass --run-live-public to run")
    for item in items:
        if "live_public" in item.keywords:
            item.add_marker(skip)


def _host_of(address: Any) -> str:
    if isinstance(address, tuple) and address:
        return str(address[0])
    return str(address)


@pytest.fixture(autouse=True)
def _no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if "live_public" in request.keywords and request.config.getoption("--run-live-public"):
        return

    def guarded_connect(self: socket.socket, address: Any) -> None:
        if _host_of(address) not in _LOOPBACK:
            raise RuntimeError(f"network access is forbidden in tests (tried {address!r})")
        _real_connect(self, address)

    def guarded_connect_ex(self: socket.socket, address: Any) -> int:
        if _host_of(address) not in _LOOPBACK:
            raise RuntimeError(f"network access is forbidden in tests (tried {address!r})")
        return _real_connect_ex(self, address)

    def guarded_create_connection(address: Any, *args: Any, **kwargs: Any) -> socket.socket:
        if _host_of(address) not in _LOOPBACK:
            raise RuntimeError(f"network access is forbidden in tests (tried {address!r})")
        return _real_create_connection(address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)


def _program_of(args: Any, kwargs: dict[str, Any]) -> str:
    executable = kwargs.get("executable")
    if executable:
        return str(executable)
    if isinstance(args, (str, bytes, os.PathLike)):
        return os.fsdecode(args)
    if isinstance(args, Sequence) and args:
        return os.fsdecode(args[0])
    return ""


def _is_python(program: str) -> bool:
    if not program:
        return False
    path = Path(program)
    if path.is_absolute() and path.exists() and Path(sys.executable).exists():
        try:
            if path.samefile(sys.executable):
                return True
        except OSError:
            return False
    name = path.name.lower()
    stem = name[:-4] if name.endswith(".exe") else name
    return _PYTHON_NAME.fullmatch(stem) is not None


@pytest.fixture(autouse=True)
def _no_foreign_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    def guarded_popen_init(self: subprocess.Popen[Any], args: Any, *a: Any, **kw: Any) -> None:
        program = _program_of(args, kw)
        if kw.get("shell") and isinstance(args, str) and args in _OS_VERSION_QUERIES:
            _real_popen_init(self, args, *a, **kw)
            return
        if (
            not kw.get("shell")
            and isinstance(args, list | tuple)
            and tuple(str(x) for x in args) in _OS_PROCESSOR_QUERIES
        ):
            _real_popen_init(self, args, *a, **kw)
            return
        if _is_publisher_check(args, kw):
            _real_popen_init(self, args, *a, **kw)
            return
        if kw.get("shell") or not _is_python(program):
            raise RuntimeError(
                f"starting {program or args!r} is forbidden in tests; inject a fake runner "
                "(only a Python interpreter, without a shell, may be started)"
            )
        _real_popen_init(self, args, *a, **kw)

    def refuse_system(command: Any) -> int:
        raise RuntimeError(f"os.system({command!r}) is forbidden in tests")

    monkeypatch.setattr(subprocess.Popen, "__init__", guarded_popen_init)
    monkeypatch.setattr(os, "system", refuse_system)
    if hasattr(os, "startfile"):
        monkeypatch.delattr(os, "startfile")


@pytest.fixture(autouse=True)
def _no_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.startswith("BITGET_"):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture
def clock() -> ManualClock:
    """Wednesday 2026-09-23 13:00 UTC: a weekday, half an hour before the US open."""
    return ManualClock(datetime(2026, 9, 23, 13, 0, tzinfo=UTC))


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    """A throwaway project root with the directories the runtime expects."""
    for sub in ("var/ledger", "var/blobs", "public", ".secrets"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    return tmp_path


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES
