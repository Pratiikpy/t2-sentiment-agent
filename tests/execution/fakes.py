"""In-memory stand-ins for the execution tests: blob store, ledger, bgc runner, market data.

The ledger and blob store follow the ``types.LedgerWriter``/``LedgerReader``/``BlobStore``
contracts closely enough that the executor and reconciler cannot tell the difference: the ledger
refuses a payload that is not the model registered for its kind, and chains each event to the
previous one by hash. The runner replays ``bgc`` answers recorded by
``tests/fixtures/execution/capture.py`` and records every call it receives.
"""

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from helpers import make_quote
from sentiment_agent.execution.bgc import BgcResult
from sentiment_agent.hashing import ZERO_HASH, content_hash, sha256_hex
from sentiment_agent.types import (
    EVENT_PAYLOADS,
    BlobRef,
    Candle,
    CandleKind,
    Clock,
    EventKind,
    FundingPoint,
    InstrumentSpec,
    LedgerEvent,
    Model,
    PriceSource,
    Quote,
    RunMode,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "execution"


def fixture(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    return data


def fixture_result(name: str) -> BgcResult:
    """The ``bgc`` answer recorded in a fixture, as the runner would return it."""
    result = fixture(name)["result"]
    return BgcResult(
        exit_code=result["exit_code"],
        stdout=result["stdout"],
        stderr=result["stderr"],
        duration_ms=1,
    )


def ok_result(data: Any) -> BgcResult:
    """A successful ``bgc`` read, shaped as ``callRead`` prints it (index.ts:235-239)."""
    return BgcResult(
        exit_code=0,
        stdout={"endpoint": "GET (test)", "requestTime": "2026-09-23T13:00:00Z", "data": data},
        stderr=None,
        duration_ms=1,
    )


class MemoryBlobStore:
    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}

    def put(self, data: bytes, media_type: str) -> BlobRef:
        digest = sha256_hex(data)
        self.data[digest] = data
        return BlobRef(sha256=digest, media_type=media_type, size=len(data))

    def get(self, sha256: str) -> bytes:
        return self.data[sha256]


class MemoryLedger:
    """A hash-chained, typed, in-memory ledger (writer and reader)."""

    def __init__(self, mode: RunMode, clock: Clock) -> None:
        self._mode = mode
        self._clock = clock
        self.rows: list[LedgerEvent] = []

    @property
    def mode(self) -> RunMode:
        return self._mode

    def append(
        self, kind: EventKind, payload: Model, *, blobs: Sequence[BlobRef] = ()
    ) -> LedgerEvent:
        expected = EVENT_PAYLOADS[kind]
        if type(payload) is not expected:
            raise TypeError(f"{kind} takes {expected.__name__}, not {type(payload).__name__}")
        seq = len(self.rows)
        prev = self.rows[-1].hash if self.rows else ZERO_HASH
        ts = self._clock.now()
        dumped = payload.model_dump(mode="json")
        digest = content_hash(
            {
                "seq": seq,
                "ts": ts,
                "kind": kind,
                "mode": self._mode,
                "payload": dumped,
                "blobs": list(blobs),
                "prev_hash": prev,
            }
        )
        event = LedgerEvent(
            seq=seq,
            ts=ts,
            kind=kind,
            mode=self._mode,
            payload=dumped,
            blobs=tuple(blobs),
            prev_hash=prev,
            hash=digest,
        )
        self.rows.append(event)
        return event

    def events(self, kinds: frozenset[EventKind] | None = None) -> Iterator[LedgerEvent]:
        return iter([e for e in self.rows if kinds is None or e.kind in kinds])

    def head(self) -> LedgerEvent | None:
        return self.rows[-1] if self.rows else None

    def kinds(self) -> list[EventKind]:
        return [e.kind for e in self.rows]


class WriterOnlyLedger:
    """A ledger that can be written and not read back."""

    def __init__(self, inner: MemoryLedger) -> None:
        self._inner = inner

    @property
    def mode(self) -> RunMode:
        return self._inner.mode

    def append(
        self, kind: EventKind, payload: Model, *, blobs: Sequence[BlobRef] = ()
    ) -> LedgerEvent:
        return self._inner.append(kind, payload, blobs=blobs)


Answer = BgcResult | BaseException | Callable[[Sequence[str]], BgcResult]


class ScriptedRunner:
    """Answers ``bgc`` calls from a list of (predicate, answer) rules and records every call.

    A rule's predicate receives the argv. An answer is a result, an exception to raise, or a
    function of the argv. The first matching rule answers; a rule listed with ``once=True`` is used
    up after it answers. An unmatched call fails the test.
    """

    def __init__(self) -> None:
        self.rules: list[tuple[Callable[[Sequence[str]], bool], Answer, bool]] = []
        self.calls: list[tuple[tuple[str, ...], dict[str, str], float]] = []

    def on(
        self, predicate: Callable[[Sequence[str]], bool], answer: Answer, *, once: bool = False
    ) -> "ScriptedRunner":
        self.rules.append((predicate, answer, once))
        return self

    def __call__(
        self, args: Sequence[str], *, env: Mapping[str, str], timeout_s: float
    ) -> BgcResult:
        self.calls.append((tuple(args), dict(env), timeout_s))
        for index, (predicate, answer, once) in enumerate(self.rules):
            if predicate(args):
                if once:
                    del self.rules[index]
                if isinstance(answer, BaseException):
                    raise answer
                if isinstance(answer, BgcResult):
                    return answer
                return answer(args)
        raise AssertionError(f"unexpected bgc call: {list(args)}")

    def argvs(self) -> list[tuple[str, ...]]:
        return [call[0] for call in self.calls]


def verb(*words: str) -> Callable[[Sequence[str]], bool]:
    """Match an argv that starts with ``words`` (e.g. ``verb("order", "--action", "place")``)."""
    return lambda args: tuple(args[: len(words)]) == words


def has(flag: str) -> Callable[[Sequence[str]], bool]:
    return lambda args: flag in args


def both(*predicates: Callable[[Sequence[str]], bool]) -> Callable[[Sequence[str]], bool]:
    return lambda args: all(p(args) for p in predicates)


def lacks(flag: str) -> Callable[[Sequence[str]], bool]:
    return lambda args: flag not in args


class FakeMarket:
    """``types.MarketData`` with quotes the test sets. Only quotes are used by the executor."""

    def __init__(self) -> None:
        self.demo: dict[str, Quote] = {}
        self.calls = 0

    def set(
        self,
        symbol: str,
        *,
        bid: str,
        ask: str,
        mark: str,
        at: datetime,
    ) -> None:
        mid = (Decimal(bid) + Decimal(ask)) / 2
        self.demo[symbol] = make_quote(
            symbol, bid=bid, ask=ask, mark=mark, last=str(mid), index=mark, at=at
        )

    def instruments(self, source: PriceSource, symbols: Sequence[str]) -> dict[str, InstrumentSpec]:
        return {}

    def quotes(self, source: PriceSource, symbols: Sequence[str]) -> dict[str, Quote]:
        self.calls += 1
        assert source is PriceSource.DEMO
        return {s: self.demo[s] for s in symbols if s in self.demo}

    def candles(
        self,
        source: PriceSource,
        symbol: str,
        *,
        kind: CandleKind,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> list[Candle]:
        return []

    def funding_history(self, symbol: str, *, limit: int) -> list[FundingPoint]:
        return []


# --- a project root on disk -------------------------------------------------------------------

DEMO_KEY = "bg_demo_fixture_key_0001"
DEMO_SECRET = "demo-fixture-secret-0002"
DEMO_PASSPHRASE = "demo-fixture-pass-0003"

_SDK_SNIPPET = """\
  buildHeaders(config, endpoint, bodyJson) {
    if (this.config.paperTrading && config.auth === "private") {
      headers.set("paptrading", "1");
    }
  }
  if (cli.paperTrading && cli.readOnly) {
    throw new ConfigError("paperTrading and readOnly are mutually exclusive.");
  }
  if (dryRun) {
    return previewResult(op, riskLevel, rest);
  }
"""
_CLI_SNIPPET = """\
    const toolKey = key === "dry-run" ? "dryRun" : key;
      readOnly: globals["read-only"] === true,
      paperTrading: globals["paper-trading"] === true,
"""


def write_demo_env(
    root: Path,
    *,
    declaration: str | None = "demo",
    key: str | None = DEMO_KEY,
    secret: str | None = DEMO_SECRET,
    passphrase: str | None = DEMO_PASSPHRASE,
    extra: str = "",
) -> Path:
    lines = ["# Bitget Demo key for the test"]
    if declaration is not None:
        lines.append(f"BITGET_KEY_ENVIRONMENT={declaration}")
    for name, value in (
        ("BITGET_API_KEY", key),
        ("BITGET_SECRET_KEY", secret),
        ("BITGET_PASSPHRASE", passphrase),
    ):
        if value is not None:
            lines.append(f"{name}={value}")
    path = root / ".secrets" / "demo.env"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n" + extra, encoding="utf-8")
    return path


def make_agent_hub(
    root: Path,
    *,
    header: bool = True,
    cli_version: str = "3.0.0",
    sdk_version: str = "3.0.0",
    locked_version: str = "3.0.0",
) -> Path:
    """A ``tools/agent-hub`` whose installed files satisfy (or, by argument, break) the contract."""
    hub = root / "tools" / "agent-hub"
    for package, version, text in (
        ("bitget-agent-cli", cli_version, _CLI_SNIPPET),
        (
            "bitget-agent-sdk",
            sdk_version,
            _SDK_SNIPPET if header else _SDK_SNIPPET.replace('"paptrading", "1"', '"x", "1"'),
        ),
    ):
        pkg = hub / "node_modules" / "@bitget-ai" / package
        (pkg / "lib").mkdir(parents=True, exist_ok=True)
        (pkg / "package.json").write_text(
            json.dumps({"name": f"@bitget-ai/{package}", "version": version}), encoding="utf-8"
        )
        (pkg / "lib" / "index.js").write_text(text, encoding="utf-8")
    lock = {
        "packages": {
            "node_modules/@bitget-ai/bitget-agent-cli": {"version": locked_version},
            "node_modules/@bitget-ai/bitget-agent-sdk": {"version": locked_version},
        }
    }
    (hub / "package-lock.json").write_text(json.dumps(lock), encoding="utf-8")
    return hub
