"""scripts/recompute.py: the published record recomputes with the standard library alone, agrees
with the ledger module's hash and this package's metrics, and fails on any edit."""

import csv
import importlib.util
import json
import math
import random
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from analysis.abuild import BTC, NVDA, flat_candles, inputs, simulator
from sentiment_agent.analysis.bootstrap import block_bootstrap_bands
from sentiment_agent.analysis.metrics import CI_STATS, book_arm, book_metrics, metric_set
from sentiment_agent.book.book import BookBuilder
from sentiment_agent.clock import ManualClock
from sentiment_agent.hashing import canonical_json, content_hash
from sentiment_agent.ledger.blobs import FileBlobStore
from sentiment_agent.ledger.chain import HashChainLedger, event_hash
from sentiment_agent.ledger.genesis import build_genesis
from sentiment_agent.policy import POLICY_V1
from sentiment_agent.types import (
    VENUE_GUARDS,
    ArmKind,
    ArmMark,
    ArmSpec,
    ClosedTrade,
    EventKind,
    Fill,
    FillVenue,
    GuardId,
    MarkPoint,
    OrderPurpose,
    PositionMark,
    RunMode,
    Side,
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "recompute.py"
START = datetime(2026, 9, 23, 0, 0, tzinfo=UTC)
HOUR = timedelta(hours=1)
HOURS = 48


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("recompute_under_test", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module


recompute = _load()


# ------------------------------------------------------------------------------------------------
# A generated paper record, written by the real ledger and booked by the real book
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PlannedFill:
    hour: int
    """Appended to the ledger during this hour, before its mark."""
    minute: int
    """Executed at this minute of the hour (a fill may be logged after a later one)."""
    symbol: str
    side: Side
    qty: str
    trade_side: str
    purpose: OrderPurpose | None
    decision_id: str | None


PLAN = (
    PlannedFill(2, 10, BTC, Side.BUY, "0.05", "open", OrderPurpose.OPEN, "d1"),
    PlannedFill(3, 20, NVDA, Side.SELL, "20", "open", OrderPurpose.OPEN, "d2"),
    PlannedFill(6, 5, BTC, Side.BUY, "0.02", "open", OrderPurpose.INCREASE, "d3"),
    PlannedFill(9, 40, BTC, Side.SELL, "0.03", "close", OrderPurpose.REDUCE, "d4"),
    PlannedFill(12, 15, NVDA, Side.BUY, "20", "close", OrderPurpose.CLOSE, "d5"),
    PlannedFill(15, 30, BTC, Side.SELL, "0.08", "close", OrderPurpose.CLOSE, "d6"),  # flips short
    PlannedFill(20, 10, NVDA, Side.BUY, "15", "open", OrderPurpose.OPEN, "d7"),
    PlannedFill(22, 50, BTC, Side.BUY, "0.04", "close", OrderPurpose.CLOSE, "d8"),
    # A venue stop that fired at 26:05, picked up by reconciliation after the 26:10 order below.
    PlannedFill(26, 10, NVDA, Side.BUY, "5", "open", OrderPurpose.INCREASE, "d9"),
    PlannedFill(26, 5, NVDA, Side.SELL, "15", "close", None, None),
    PlannedFill(33, 0, BTC, Side.BUY, "0.03", "open", OrderPurpose.OPEN, "d10"),
    PlannedFill(40, 45, BTC, Side.SELL, "0.03", "close", OrderPurpose.CLOSE, "d11"),
)


def _prices() -> dict[str, list[Decimal]]:
    rng = random.Random(42)  # noqa: S311 - seeded, reproducible test data
    out: dict[str, list[Decimal]] = {}
    for symbol, start in ((BTC, 80_000.0), (NVDA, 200.0)):
        price = start
        series = []
        for _ in range(HOURS + 1):
            series.append(Decimal(repr(round(price, 2))))
            price *= math.exp(rng.gauss(0, 0.004))
        out[symbol] = series
    return out


@dataclass
class Record:
    public: Path
    fills: list[Fill]
    marks: list[MarkPoint]
    trades: tuple[ClosedTrade, ...]
    ledger_path: Path


def _write_record(root: Path) -> Record:
    clock = ManualClock(START)
    ledger_path = root / "var" / "ledger" / "paper.jsonl"
    ledger_path.parent.mkdir(parents=True)
    ledger = HashChainLedger(ledger_path, mode=RunMode.PAPER, clock=clock)
    blobs = FileBlobStore(root / "var" / "blobs")
    genesis = build_genesis(
        policy=POLICY_V1,
        prompt_hashes={"src/sentiment_agent/decision/prompts/system_v1.md": content_hash("p")},
        mode=RunMode.PAPER,
        code_commit="a" * 40,
        lock_hashes={"uv.lock": content_hash("lock")},
        bgc_package="@bitget-ai/bitget-agent-cli@3.0.0",
        clock=clock,
    )
    ledger.append(EventKind.GENESIS, genesis)
    prices = _prices()
    builder = BookBuilder(starting_equity=Decimal(100_000), policy=POLICY_V1)
    fills: list[Fill] = []
    marks: list[MarkPoint] = []
    for hour in range(HOURS + 1):
        at = START + hour * HOUR
        for n, planned in enumerate(p for p in PLAN if p.hour == hour):
            clock.set(max(clock.now(), at - HOUR + timedelta(minutes=55 + n)))
            price = prices[planned.symbol][hour]
            qty = Decimal(planned.qty)
            raw = json.dumps({"symbol": planned.symbol, "price": str(price)}).encode()
            ref = blobs.put(raw, "application/json")
            exec_id = f"x{len(fills)}"
            fill = Fill(
                exec_id=exec_id,
                venue_order_id=f"o{len(fills)}",
                client_oid=None,
                symbol=planned.symbol,
                side=planned.side,
                exec_price=price,
                exec_qty=qty,
                exec_value=price * qty,
                fee_paid=(price * qty * Decimal("0.0006")).quantize(Decimal("0.00000001")),
                fee_coin="USDT",
                trade_scope="taker",
                trade_side=planned.trade_side,  # type: ignore[arg-type]
                exec_pnl=None,
                executed_at=at - HOUR + timedelta(minutes=planned.minute),
                venue=FillVenue.BITGET_DEMO,
                blob=ref,
            )
            ledger.append(EventKind.FILL, fill, blobs=(ref,))
            builder.apply_fill(fill, decision_id=planned.decision_id, purpose=planned.purpose)
            fills.append(fill)
            if exec_id == "x3":  # a reconciliation sweep that sees the same fill again
                ledger.append(EventKind.FILL, fill, blobs=(ref,))
        clock.set(max(clock.now(), at))
        held = builder.positions()
        mark_prices = {s: prices[s][hour] for s in held}
        equity = builder.equity(mark_prices)
        gross = sum((abs(p.qty) * mark_prices[s] for s, p in held.items()), Decimal(0))
        net = sum((p.qty * mark_prices[s] for s, p in held.items()), Decimal(0))
        point = MarkPoint(
            at=at,
            equity_book=equity,
            equity_venue=None,
            equity_live_mirror=None,
            gross_weight=float(gross / equity),
            net_weight=float(net / equity),
            positions=tuple(
                PositionMark(
                    symbol=s,
                    qty=p.qty,
                    demo_mark=mark_prices[s],
                    live_mark=None,
                    demo_index=None,
                    unrealized_demo=p.qty * (mark_prices[s] - p.avg_entry),
                    unrealized_live=None,
                )
                for s, p in held.items()
            ),
        )
        ledger.append(EventKind.MARK, point)
        marks.append(point)
    trades = builder.closed_trades()
    public = root / "public"
    _export(public, ledger_path, root / "var" / "blobs", marks, trades, fills)
    return Record(public=public, fills=fills, marks=marks, trades=trades, ledger_path=ledger_path)


def _export(
    public: Path,
    ledger_path: Path,
    blob_root: Path,
    marks: list[MarkPoint],
    trades: tuple[ClosedTrade, ...],
    fills: list[Fill],
) -> None:
    """What the site export must write for the record to recompute (scripts/recompute.py)."""
    public.mkdir(parents=True)
    shutil.copyfile(ledger_path, public / "ledger.jsonl")
    (public / "blobs").mkdir()
    for blob in blob_root.iterdir():
        if not blob.name.startswith("."):
            shutil.copyfile(blob, public / "blobs" / blob.name)
    with (public / "equity_hourly.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["at", "equity_book", "gross_weight", "net_weight"])
        for m in marks:
            writer.writerow(
                [m.model_dump(mode="json")["at"], str(m.equity_book), m.gross_weight, m.net_weight]
            )
    with (public / "trades.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        columns = list(ClosedTrade.model_fields)
        writer.writerow(columns)
        for t in trades:
            row = t.model_dump(mode="json")
            row["decision_ids"] = ";".join(t.decision_ids)
            writer.writerow([row[c] for c in columns])
    metrics = book_metrics(marks, trades, fills)
    (public / "metrics.json").write_text(
        json.dumps(metrics.model_dump(mode="json")), encoding="utf-8"
    )
    sim = simulator({BTC: flat_candles(BTC, START, HOURS, "80000")})
    spec = ArmSpec(
        arm_id="baseline_flat",
        kind=ArmKind.BASELINE,
        title="Flat",
        description="d",
        provenance="tests",
        uses_llm=False,
        guards=tuple(g for g in GuardId if g in VENUE_GUARDS),
    )
    flat = sim.run(spec, [], start=START, until=START + 6 * HOUR)
    later = START + 7 * HOUR
    btc_spec = spec.model_copy(update={"arm_id": "baseline_btc"})
    btc = sim.run(
        btc_spec,
        [(START + 2 * HOUR, {BTC: 0.05}, inputs({BTC: "80000"}, at=START + 2 * HOUR))],
        until=later,
        ci_resamples=0,
    )
    arms = [flat, btc, book_arm(marks, trades, fills)]
    (public / "arms.json").write_text(
        json.dumps([a.model_dump(mode="json") for a in arms]), encoding="utf-8"
    )


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> Record:
    return _write_record(tmp_path_factory.mktemp("record"))


@pytest.fixture
def record(generated: Record, tmp_path: Path) -> Path:
    """A private copy of the published directory, free to tamper with."""
    copy = tmp_path / "public"
    shutil.copytree(generated.public, copy)
    return copy


def run(public: Path) -> tuple[bool, dict[str, tuple[bool, str]]]:
    report = recompute.verify(public)
    return report.passed, {name: (ok, detail) for name, ok, detail in report.rows}


# ------------------------------------------------------------------------------------------------
# The whole record recomputes
# ------------------------------------------------------------------------------------------------


def test_the_generated_record_recomputes(record: Path) -> None:
    passed, rows = run(record)
    assert passed, rows
    assert set(rows) == {
        "ledger chain",
        "genesis",
        "blobs",
        "equity_hourly.csv",
        "trades.csv",
        "marks from fills",
        "metrics.json",
        "arms.json",
    }
    assert "1 repeated fill events ignored" in rows["trades.csv"][1]
    assert "49 marks rebuilt from the fills" in rows["marks from fills"][1]


def test_a_judge_runs_it_as_a_script(record: Path) -> None:
    done = subprocess.run(  # noqa: S603 - the Python interpreter on a fixed path
        [sys.executable, str(SCRIPT), str(record)],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "all checks agree" in done.stdout
    for name in ("ledger chain", "metrics.json", "trades.csv"):
        assert name in done.stdout


def test_usage_errors_exit_2(tmp_path: Path) -> None:
    assert recompute.main([]) == 2
    assert recompute.main([str(tmp_path / "missing")]) == 2


def test_the_record_has_what_it_should(generated: Record) -> None:
    """The generated record exercises the hard cases: a flip, a partial reduction, a late stop fill
    folded in venue-time order, and a repeated fill."""
    reasons = [t.exit_reason for t in generated.trades]
    assert "flip" in reasons
    assert "stop_filled" in reasons or "venue_initiated" in reasons
    assert len(generated.trades) >= 5


# ------------------------------------------------------------------------------------------------
# Agreement with the ledger module and this package
# ------------------------------------------------------------------------------------------------


def test_event_hash_agrees_with_the_ledger_module(generated: Record) -> None:
    lines = generated.ledger_path.read_bytes().splitlines()
    ledger = HashChainLedger(generated.ledger_path, mode=RunMode.PAPER, clock=ManualClock(START))
    events = list(ledger.events())
    assert len(events) == len(lines)
    for line, event in zip(lines, events, strict=True):
        parsed = json.loads(line)
        assert recompute.event_hash(parsed) == event.hash
        assert recompute.canonical(parsed) == line == canonical_json(event)
        assert event.hash == event_hash(
            seq=event.seq,
            ts=event.ts,
            kind=event.kind,
            mode=event.mode,
            payload=event.payload,
            blobs=event.blobs,
            prev_hash=event.prev_hash,
        )


@pytest.mark.parametrize("seed", range(20))
def test_canonical_form_agrees_with_hashing(seed: int) -> None:
    rng = random.Random(seed)  # noqa: S311 - seeded, reproducible test data

    def value(depth: int) -> Any:
        kind = rng.randrange(7 if depth < 3 else 5)
        if kind == 0:
            return rng.choice([1e-7, -0.0, 0.1, 1e16, 123456789.125, -2.5e-300])
        if kind == 1:
            return rng.randrange(-(10**12), 10**12)
        if kind == 2:
            return rng.choice(["", "é", 'a"b\\c', " ", "日本", "\n\t"])
        if kind == 3:
            return rng.choice([True, False, None])
        if kind == 4:
            return str(Decimal(rng.randrange(1, 10**9)) / Decimal(10 ** rng.randrange(0, 9)))
        if kind == 5:
            return [value(depth + 1) for _ in range(rng.randrange(4))]
        return {f"k{rng.randrange(50)}": value(depth + 1) for _ in range(rng.randrange(4))}

    for _ in range(50):
        thing = value(0)
        ours = canonical_json(thing)
        assert recompute.canonical(json.loads(ours)) == ours


def test_trades_rebuilt_from_fills_agree_with_the_book(generated: Record) -> None:
    lines = (generated.public / "ledger.jsonl").read_bytes().splitlines()
    events = [json.loads(line) for line in lines]
    fills, repeats = recompute.fills_of(events)
    assert repeats == 1
    rebuilt = recompute.rebuild_trades(fills)
    assert len(rebuilt) == len(generated.trades)
    for mine, theirs in zip(rebuilt, generated.trades, strict=True):
        assert mine["symbol"] == theirs.symbol
        assert mine["direction"] == theirs.direction
        assert (mine["opened_at"], mine["closed_at"]) == (theirs.opened_at, theirs.closed_at)
        for name in ("entry_avg", "exit_avg", "max_abs_qty", "gross_pnl", "fees", "net_pnl"):
            assert mine[name] == getattr(theirs, name), name


@pytest.mark.parametrize("seed", range(8))
def test_metrics_agree_with_this_package(seed: int, monkeypatch: pytest.MonkeyPatch) -> None:
    rng = random.Random(seed)  # noqa: S311 - seeded, reproducible test data
    n = rng.randrange(2, 120)
    equity = [10_000.0]
    for _ in range(n):
        equity.append(equity[-1] * (1 + rng.gauss(0.0001, 0.002)))
    if seed == 0:
        equity = [10_000.0] * 30  # flat: every ratio undefined
    net = [Decimal(repr(round(rng.gauss(0, 5), 4))) for _ in range(rng.randrange(0, 12))]
    marks = [
        ArmMark(at=START + i * HOUR, equity=e, gross_weight=0.0, net_weight=0.0)
        for i, e in enumerate(equity)
    ]
    trades = [_trade(x) for x in net]
    monkeypatch.setattr(recompute, "BOOT_RESAMPLES", 400)
    ours = metric_set("a", marks, trades, traded_notional=1234.5, fees=6.7, ci_resamples=400)
    theirs = recompute.recompute_metrics(
        equity, net, notional=1234.5, fees=6.7, with_intervals=None
    )
    recompute.compare_metrics(ours.model_dump(mode="json"), theirs, exact_ci_keys=True)
    assert ours.n_hours == theirs["n_hours"]
    for name, (lo, hi) in ours.ci90.items():
        assert (lo, hi) == pytest.approx(theirs["ci90"][name], rel=1e-12)


def _trade(net: Decimal) -> ClosedTrade:
    return ClosedTrade(
        symbol=BTC,
        opened_at=START,
        closed_at=START + HOUR,
        direction=1,
        entry_avg=Decimal(1),
        exit_avg=Decimal(1),
        max_abs_qty=Decimal(1),
        gross_pnl=net,
        fees=Decimal(0),
        net_pnl=net,
        decision_ids=(),
        exit_reason="model_close",
    )


# ------------------------------------------------------------------------------------------------
# Every edit is found
# ------------------------------------------------------------------------------------------------


def _edit_csv(path: Path, row: int, column: str, change: Callable[[str], str]) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    rows[row][column] = change(rows[row][column])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _lines(public: Path) -> list[bytes]:
    return (public / "ledger.jsonl").read_bytes().splitlines()


def _write_lines(public: Path, lines: list[bytes]) -> None:
    (public / "ledger.jsonl").write_bytes(b"".join(line + b"\n" for line in lines))


def _failed(public: Path) -> set[str]:
    passed, rows = run(public)
    assert not passed
    return {name for name, (ok, _) in rows.items() if not ok}


def test_an_edited_equity_row_fails(record: Path) -> None:
    _edit_csv(record / "equity_hourly.csv", 10, "equity_book", lambda v: str(Decimal(v) + 1))
    assert "equity_hourly.csv" in _failed(record)


def test_a_dropped_equity_row_fails(record: Path) -> None:
    path = record / "equity_hourly.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    assert "equity_hourly.csv" in _failed(record)


def test_an_edited_ledger_line_fails(record: Path) -> None:
    lines = _lines(record)
    index = next(i for i, line in enumerate(lines) if b'"kind":"mark"' in line)
    event = json.loads(lines[index])
    event["payload"]["equity_book"] = str(Decimal(event["payload"]["equity_book"]) + 100)
    lines[index] = canonical_json(event)  # still canonical, so only the hash can catch it
    _write_lines(record, lines)
    failed = _failed(record)
    assert "ledger chain" in failed


def test_a_respelt_ledger_line_fails(record: Path) -> None:
    lines = _lines(record)
    lines[1] = lines[1].replace(b'{"blobs"', b'{ "blobs"', 1)
    _write_lines(record, lines)
    assert "ledger chain" in _failed(record)


def test_a_boolean_sequence_number_is_refused(record: Path) -> None:
    """``true == 1`` in Python; the ledger never writes a boolean where a number belongs."""
    events = [json.loads(line) for line in _lines(record)]
    events[1]["seq"] = True
    previous = "0" * 64
    for event in events:
        event["prev_hash"] = previous
        event["hash"] = recompute.event_hash(event)
        previous = event["hash"]
    _write_lines(record, [canonical_json(e) for e in events])
    assert "ledger chain" in _failed(record)


def test_a_rehashed_forgery_is_still_caught_by_the_genesis(record: Path) -> None:
    """Someone who edits the pre-registered policy and re-hashes every line keeps a valid chain;
    the genesis policy hash no longer matches the policy it carries."""
    events = [json.loads(line) for line in _lines(record)]
    events[0]["payload"]["policy"]["per_name_max"] = 0.5
    previous = "0" * 64
    for event in events:
        event["prev_hash"] = previous
        event["hash"] = recompute.event_hash(event)
        previous = event["hash"]
    _write_lines(record, [canonical_json(e) for e in events])
    failed = _failed(record)
    assert failed == {"genesis"}


def test_a_truncated_ledger_fails(record: Path) -> None:
    _write_lines(record, _lines(record)[:-3])
    assert "equity_hourly.csv" in _failed(record)


def test_an_edited_trade_fails(record: Path) -> None:
    _edit_csv(record / "trades.csv", 0, "net_pnl", lambda v: str(Decimal(v) + Decimal("0.01")))
    assert "trades.csv" in _failed(record)


def test_a_hidden_losing_trade_fails(record: Path) -> None:
    path = record / "trades.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    losers = [i for i, r in enumerate(rows) if Decimal(r["net_pnl"]) < 0]
    assert losers
    kept = [r for i, r in enumerate(rows) if i != losers[0]]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(kept)
    assert "trades.csv" in _failed(record)


def _edit_json(path: Path, change: Callable[[Any], None]) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_an_edited_metric_fails(record: Path) -> None:
    _edit_json(record / "metrics.json", lambda d: d.update(max_drawdown=d["max_drawdown"] / 2))
    assert _failed(record) == {"metrics.json"}


def test_an_edited_interval_fails(record: Path) -> None:
    _edit_json(record / "metrics.json", lambda d: d["ci90"]["sharpe_ann"].__setitem__(1, 99.0))
    assert _failed(record) == {"metrics.json"}


def test_a_dropped_interval_fails(record: Path) -> None:
    _edit_json(record / "metrics.json", lambda d: d["ci90"].pop("total_return"))
    assert _failed(record) == {"metrics.json"}


def test_a_missing_metrics_file_fails(record: Path) -> None:
    (record / "metrics.json").unlink()
    assert _failed(record) == {"metrics.json"}


def test_a_tampered_blob_fails(record: Path) -> None:
    blob = next((record / "blobs").iterdir())
    blob.write_bytes(blob.read_bytes() + b" ")
    assert _failed(record) == {"blobs"}


def test_a_missing_blob_fails(record: Path) -> None:
    next((record / "blobs").iterdir()).unlink()
    assert _failed(record) == {"blobs"}


def test_an_edited_arm_fails(record: Path) -> None:
    def change(arms: list[dict[str, Any]]) -> None:
        arms[1]["metrics"]["total_return"] = 0.5

    _edit_json(record / "arms.json", change)
    assert _failed(record) == {"arms.json"}


def test_a_script_run_on_a_tampered_record_exits_1(record: Path) -> None:
    _edit_csv(record / "trades.csv", 0, "fees", lambda v: "0")
    done = subprocess.run(  # noqa: S603 - the Python interpreter on a fixed path
        [sys.executable, str(SCRIPT), str(record)],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert done.returncode == 1
    assert "FAIL" in done.stdout
    assert "MISMATCH" in done.stdout


def test_the_script_imports_nothing_from_this_project() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "sentiment_agent" not in source.split('"""', 2)[2]
    allowed = {
        "csv", "hashlib", "itertools", "json", "math", "random", "sys", "collections.abc",
        "dataclasses", "datetime", "decimal", "pathlib", "typing",
    }  # fmt: skip
    imported = {
        line.split()[1] for line in source.splitlines() if line.startswith(("import ", "from "))
    }
    assert imported <= allowed, imported - allowed


# ------------------------------------------------------------------------------------------------
# The marks are rebuilt from the fills (review findings)
# ------------------------------------------------------------------------------------------------


def _rehash(events: list[dict[str, Any]]) -> list[bytes]:
    previous = "0" * 64
    for seq, event in enumerate(events):
        event["seq"] = seq
        event["prev_hash"] = previous
        event["hash"] = recompute.event_hash(event)
        previous = event["hash"]
    return [canonical_json(e) for e in events]


def _republish_marks(record: Path, events: list[dict[str, Any]], generated: Record) -> None:
    """Rewrite equity_hourly.csv and metrics.json from the (edited) mark events, exactly as an
    exporter would, so only the rebuild from the fills can tell."""
    marks = [MarkPoint.model_validate(e["payload"]) for e in events if e["kind"] == "mark"]
    with (record / "equity_hourly.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["at", "equity_book", "gross_weight", "net_weight"])
        for m in marks:
            writer.writerow(
                [m.model_dump(mode="json")["at"], str(m.equity_book), m.gross_weight, m.net_weight]
            )
    metrics = book_metrics(marks, generated.trades, generated.fills)
    (record / "metrics.json").write_text(
        json.dumps(metrics.model_dump(mode="json")), encoding="utf-8"
    )


def test_an_inflated_mark_rehashed_and_republished_is_caught(
    record: Path, generated: Record
) -> None:
    """+500 USDT on every MARK after the 30th, the chain re-hashed from seq 1 (genesis kept) and
    the CSV and metrics regenerated: chain, genesis, CSV and metrics all agree with each other,
    and only rebuilding equity from the fills finds the forgery."""
    events = [json.loads(line) for line in _lines(record)]
    seen = 0
    for event in events:
        if event["kind"] != "mark":
            continue
        seen += 1
        if seen > 30:
            payload = event["payload"]
            payload["equity_book"] = str(Decimal(payload["equity_book"]) + 500)
    _write_lines(record, _rehash(events))
    _republish_marks(record, events, generated)
    passed, rows = run(record)
    assert not passed
    failed = {name for name, (ok, _) in rows.items() if not ok}
    assert "marks from fills" in failed
    assert "equity_book" in rows["marks from fills"][1]
    assert rows["ledger chain"][0]
    assert rows["genesis"][0]
    assert rows["equity_hourly.csv"][0]


def test_a_misvalued_position_is_caught(record: Path, generated: Record) -> None:
    events = [json.loads(line) for line in _lines(record)]
    target = next(e for e in events if e["kind"] == "mark" and e["payload"]["positions"])
    row = target["payload"]["positions"][0]
    row["unrealized_demo"] = str(Decimal(row["unrealized_demo"]) + 1)
    _write_lines(record, _rehash(events))
    _republish_marks(record, events, generated)
    passed, rows = run(record)
    assert not passed
    assert "unrealized" in rows["marks from fills"][1]


def test_a_fill_before_the_first_mark_fails(record: Path) -> None:
    """A record whose first MARK comes after a fill would drop that fill's cost from every
    metric (review finding on metrics.py)."""
    events = [json.loads(line) for line in _lines(record)]
    first_mark = next(i for i, e in enumerate(events) if e["kind"] == "mark")
    first_fill = next(i for i, e in enumerate(events) if e["kind"] == "fill")
    fill = events.pop(first_fill)
    fill["ts"] = events[first_mark]["ts"]  # logged at the same instant, so time still runs on
    events.insert(first_mark, fill)  # the first fill now comes before the first mark
    _write_lines(record, _rehash(events))
    passed, rows = run(record)
    assert not passed
    assert "after the first fill" in rows["marks from fills"][1]


def test_the_first_mark_must_be_the_logged_starting_equity(record: Path) -> None:
    events = [json.loads(line) for line in _lines(record)]
    ts = events[0]["ts"]
    reconciliation = {
        "seq": 0,
        "ts": ts,
        "kind": "reconciliation",
        "mode": events[0]["mode"],
        "payload": {
            "at": ts,
            "orders_checked": 0,
            "new_fill_ids": [],
            "resolved_unknown": [],
            "discrepancies": [],
            "account": {
                "at": ts,
                "equity_usdt": "99000",
                "available_usdt": "99000",
                "blob": None,
            },
            "fills_read": True,
        },
        "blobs": [],
        "prev_hash": "",
        "hash": "",
    }
    events.insert(1, reconciliation)
    _write_lines(record, _rehash(events))
    passed, rows = run(record)
    assert not passed
    assert "starting equity 99000" in rows["marks from fills"][1]


def _fill_event(index: int, **payload: Any) -> dict[str, Any]:
    base = {
        "venue_order_id": f"o{index}",
        "client_oid": None,
        "fee_coin": "USDT",
        "trade_scope": "taker",
        "exec_pnl": None,
        "venue": "bitget_demo",
        "blob": None,
    }
    return {"kind": "fill", "payload": {**base, **payload}}


def test_ties_break_by_the_venue_trade_id_as_the_book_does() -> None:
    """Two symbols close at the same instant, logged out of exec_id order (10 before 9). The book
    orders by (time, close first, len(exec_id), exec_id), so BBB (9) closes first. Ordering by
    ledger arrival would put AAA first and fail a genuine record (review finding)."""
    at = "2026-09-23T05:00:00Z"
    later = "2026-09-23T06:00:00Z"
    rows = [
        ("1", "AAAUSDT", "buy", "1", "100", "open", at),
        ("2", "BBBUSDT", "buy", "1", "50", "open", at),
        ("10", "AAAUSDT", "sell", "1", "101", "close", later),
        ("9", "BBBUSDT", "sell", "1", "51", "close", later),
    ]
    events = [
        _fill_event(
            i,
            exec_id=exec_id,
            symbol=symbol,
            side=side,
            exec_qty=qty,
            exec_price=price,
            exec_value=str(Decimal(qty) * Decimal(price)),
            fee_paid="0.01",
            trade_side=trade_side,
            executed_at=executed_at,
        )
        for i, (exec_id, symbol, side, qty, price, trade_side, executed_at) in enumerate(rows)
    ]
    fills, _ = recompute.fills_of(events)
    rebuilt = recompute.rebuild_trades(fills)

    builder = BookBuilder(starting_equity=Decimal(1000), policy=POLICY_V1)
    for event in events:
        builder.apply_fill(Fill.model_validate(event["payload"]), decision_id=None, purpose=None)
    book = builder.closed_trades()
    assert [t.symbol for t in book] == ["BBBUSDT", "AAAUSDT"]
    assert [t["symbol"] for t in rebuilt] == [t.symbol for t in book]


def test_the_script_and_the_package_agree_on_a_mostly_flat_book() -> None:
    """The bands of a mostly flat book (review finding on bootstrap.py): both implementations keep
    the Sharpe band, and state the same share of undefined resamples for the rest."""
    rng = random.Random(11)  # noqa: S311 - seeded, reproducible test data
    r = [0.0] * 50 + [rng.gauss(0.0003, 0.0004) for _ in range(22)]
    rng.shuffle(r)
    bands, shares = block_bootstrap_bands(r, CI_STATS)
    script_bands, script_shares = recompute.intervals(r)
    assert "sharpe_ann" in bands
    assert set(script_bands) == set(bands)
    for name, (lo, hi) in bands.items():
        assert script_bands[name][0] == pytest.approx(lo, rel=1e-9, abs=1e-12)
        assert script_bands[name][1] == pytest.approx(hi, rel=1e-9, abs=1e-12)
    assert script_shares == pytest.approx(shares)
