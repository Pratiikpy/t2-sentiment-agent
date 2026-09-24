"""Recompute the published record from its own files, with the Python standard library alone.

    python scripts/recompute.py public/

Nothing here imports this project, and nothing here touches the network: a judge with any Python
3.11 can run it on a copy of ``public/`` and check every number the record states. It prints one
line per check and exits 0 when everything agrees, 1 on any mismatch or missing file, and 2 on a
usage error.

What it reads from the directory, and what each file must be:

``ledger.jsonl``
    The paper ledger, byte for byte as ``ledger/chain.py`` wrote it: one JSON object per line with
    exactly the keys ``seq ts kind mode payload blobs prev_hash hash``. Checked: every line is in
    canonical form (``json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)``,
    UTF-8), so a respelt value is an edit even when it means the same; ``seq`` counts from 0;
    ``prev_hash`` is the previous line's ``hash`` (64 zeros at seq 0); ``hash`` is
    ``sha256(canonical({seq, ts, kind, mode, payload, blobs, prev_hash}))``; one mode
    throughout; ``ts`` never decreases; a ``genesis`` only at seq 0, and in a ``paper`` ledger seq 0
    must be one; the genesis ``policy_hash`` is the SHA-256 of the canonical form of the policy it
    carries.

``blobs/<sha256>``
    Every blob the ledger references (each entry of an event's ``blobs`` and every
    ``{"sha256", "media_type", "size"}`` object inside a payload) must be here, hash to its name and
    have the stated size.

``equity_hourly.csv``
    A header row with at least ``at`` and ``equity_book``, then one row per ``mark`` event of the
    ledger, in ledger order, with the same ``at`` (``2026-09-23T13:00:00Z``) and the same
    ``equity_book`` (a decimal). Marks are on the hour and strictly increasing.

    **Every mark is rebuilt from the fills, not taken on trust.** The starting equity is the
    venue's account equity as first logged before any fill (a *passed* ``environment_proof``'s
    account, or a ``reconciliation``'s), the book's own rule (``book/projection.py``). The first
    mark must come before the first fill and carry exactly that equity and no position (with no
    logged account, as in a dry run, the first mark's equity is the start and it must still
    precede every fill). Then, for each ``mark`` event, every fill logged *before it* is folded
    by the trade rules below, and the mark must agree: the same open symbols with the same
    signed quantities, each ``unrealized_demo = qty * (demo_mark - avg_entry)``, and ``equity_book
    = start + realized - fees + sum(unrealized_demo)``, within ``1e-12`` relative. So a bug or an
    edit in how a mark was valued fails here even when the chain was re-hashed around it.

``trades.csv``
    A header row with at least ``symbol opened_at closed_at direction entry_avg exit_avg
    max_abs_qty gross_pnl fees net_pnl``, then one row per closed trade in the order the trades
    closed. Rebuilt independently from the ledger's ``fill`` events by the book's own rules
    (``book/book.py``): per symbol, fills in venue-time order (``executed_at``, a ``close`` fill
    before an ``open`` one at the same instant, then the venue's trade id, shorter first and then
    as text, so a numeric id compares as a number; the ledger's arrival order never decides);
    trades listed in the order of their closing fill under the same key; average-cost positions; a
    reducing fill realizes ``(price - avg_entry) * qty * direction``; a fill larger than the
    position it reduces flips it, with its fee split by quantity; a trade runs flat to flat; the
    fees of every fill in it are in ``fees``; ``net_pnl = gross_pnl - fees``; ``entry_avg`` and
    ``exit_avg`` are the quantity-weighted prices of the fills that opened and closed it. Decimal
    arithmetic at 28 digits, half-even, as the book. A repeated ``exec_id`` with identical content
    is counted once; with different content it is a failure. Fees must be in USDT.

``metrics.json``
    One object: the governed book's metric set (``arm_id`` ``ours_governed``), recomputed from
    ``equity_hourly.csv`` (``E = equity_book``), ``trades.csv`` (``net_pnl``) and the ledger's fills
    (turnover numerator ``sum |exec_value|``, ``fees_paid = sum fee_paid``). Every field must
    agree, including each 90% interval in ``ci90``.

``arms.json`` (optional)
    A JSON array of arm results (``spec``, ``marks``, ``trades``, ``metrics``). Each arm's metrics
    are recomputed from its own marks and trades; ``turnover`` and ``fees_paid`` need the arm's
    fills, which are not published, and are not checked. An interval absent from an arm's ``ci90``
    (the coin-flip seeds carry none) is not a claim and is not checked; one present must agree.

The metric definitions are the pre-registered ones (``src/sentiment_agent/policy.py`` ``METRICS``),
written out again here rather than imported:

* ``r_t = E_t / E_{t-1} - 1``; ``mean = fsum / n``; ``pstdev`` is 0 for a constant series, else
  ``sqrt(fsum((x - mean)^2) / n)``.
* ``sharpe_ann = mean / pstdev * sqrt(8760)``; ``sharpe_se_ann = sqrt((1 + S_h^2 / 2) / n) *
  sqrt(8760)`` with ``S_h = mean / pstdev``; ``sortino_ann = mean / sqrt(fsum(min(r, 0)^2) / n) *
  sqrt(8760)``; each ``null`` where its denominator is zero or there is no return.
* ``max_drawdown = min(E_t / running max - 1)``; ``total_return = E_last / E_first - 1``;
  ``win_rate = share of trades with net_pnl > 0`` (``null`` with none); ``turnover = notional /
  mean(E)``.
* ``ci90``: moving-block bootstrap of ``r``: block ``b`` = the smallest integer with ``b^3 >= n``;
  ``random.Random(20260924)``; 10,000 resamples, each ``ceil(n / b)`` block starts
  ``randrange(n - b + 1)`` drawn in order, blocks concatenated and cut to ``n``; statistics
  ``total_return = prod(1 + r) - 1``, ``sharpe_ann`` (``0`` on a resample whose every return is
  exactly zero), ``sortino_ann`` and the drawdown of the path ``1, 1 * (1 + r_1), ...``; bounds
  ``sorted(v)[int(q * (m - 1))]`` at ``q = (1 - 0.90) / 2`` and ``1 - q`` over the resamples where
  the statistic is defined; the share of resamples where it is not must equal
  ``ci90_undefined_share`` (non-zero shares only); a statistic undefined on every resample has no
  interval.

Numbers are compared with a relative tolerance of 1e-9 (floats) and 1e-12 (decimals): the two
implementations follow one specification, and the tolerance only absorbs summation order.
"""

import csv
import hashlib
import itertools
import json
import math
import random
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import ROUND_HALF_EVEN, Context, Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any

ZERO_HASH = "0" * 64
HEX = frozenset("0123456789abcdef")
LEDGER_KEYS = frozenset({"seq", "ts", "kind", "mode", "payload", "blobs", "prev_hash", "hash"})
HASHED_KEYS = ("seq", "ts", "kind", "mode", "payload", "blobs", "prev_hash")
TRADE_COLUMNS = (
    "symbol",
    "opened_at",
    "closed_at",
    "direction",
    "entry_avg",
    "exit_avg",
    "max_abs_qty",
    "gross_pnl",
    "fees",
    "net_pnl",
)
METRIC_FLOATS = (
    "total_return",
    "sharpe_ann",
    "sharpe_se_ann",
    "sortino_ann",
    "max_drawdown",
    "win_rate",
    "turnover",
    "fees_paid",
)
CI_NAMES = ("total_return", "sharpe_ann", "sortino_ann", "max_drawdown")
LABEL = "descriptive, not inferential"
BOOK_ARM_ID = "ours_governed"
BOOT_SEED = 20260924
BOOT_RESAMPLES = 10_000
BOOT_LEVEL = 0.90
YEAR_HOURS = 8760
FLOAT_TOL = 1e-9
DECIMAL_TOL = Decimal("1e-12")
BOOK_CONTEXT = Context(prec=28, rounding=ROUND_HALF_EVEN)


class MismatchError(Exception):
    """A published file disagrees with what it can be recomputed from."""


@dataclass
class Report:
    rows: list[tuple[str, bool, str]] = field(default_factory=list)

    def ok(self, check: str, detail: str) -> None:
        self.rows.append((check, True, detail))

    def fail(self, check: str, detail: str) -> None:
        self.rows.append((check, False, detail))

    @property
    def passed(self) -> bool:
        return all(ok for _, ok, _ in self.rows)

    def table(self) -> str:
        width = max([len("check"), *(len(c) for c, _, _ in self.rows)])
        lines = [f"{'check'.ljust(width)}  result  detail"]
        for check, ok, detail in self.rows:
            lines.append(f"{check.ljust(width)}  {'ok' if ok else 'FAIL':6}  {detail}")
        verdict = "all checks agree" if self.passed else "MISMATCH: the record does not recompute"
        lines.append(verdict)
        return "\n".join(lines)


# ================================================================================================
# Canonical form and the chain
# ================================================================================================


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def event_hash(event: dict[str, Any]) -> str:
    """``sha256(canonical({seq, ts, kind, mode, payload, blobs, prev_hash}))`` of a parsed line."""
    return sha256_hex(canonical({key: event[key] for key in HASHED_KEYS}))


def parse_time(text: Any) -> datetime:
    if not isinstance(text, str):
        raise MismatchError(f"timestamp {text!r} is not a string")
    try:
        value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MismatchError(f"timestamp {text!r} is not ISO 8601") from exc
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise MismatchError(f"timestamp {text!r} is not UTC")
    return value


def dec(text: Any, what: str) -> Decimal:
    if not isinstance(text, (str, int)) or isinstance(text, bool):
        raise MismatchError(f"{what} {text!r} is not a decimal")
    try:
        value = Decimal(str(text).strip())
    except InvalidOperation as exc:
        raise MismatchError(f"{what} {text!r} is not a decimal") from exc
    if not value.is_finite():
        raise MismatchError(f"{what} {text!r} is not finite")
    return value


def is_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in HEX for c in value)


def check_types(event: dict[str, Any], where: str) -> None:
    """Every field has the JSON type the ledger writes (a ``true`` is not a sequence number)."""
    seq = event["seq"]
    if type(seq) is not int or seq < 0:
        raise MismatchError(f"{where}: seq {seq!r} is not a non-negative integer")
    for key in ("ts", "kind", "mode"):
        if not isinstance(event[key], str):
            raise MismatchError(f"{where}: {key} is not a string")
    if not isinstance(event["payload"], dict):
        raise MismatchError(f"{where}: payload is not an object")
    blobs = event["blobs"]
    if not isinstance(blobs, list) or not all(
        isinstance(b, dict)
        and set(b) == {"sha256", "media_type", "size"}
        and is_digest(b["sha256"])
        and type(b["size"]) is int
        for b in blobs
    ):
        raise MismatchError(f"{where}: blobs are not blob references")
    for key in ("prev_hash", "hash"):
        if not is_digest(event[key]):
            raise MismatchError(f"{where}: {key} is not a 64-character lowercase hex digest")


def read_ledger(path: Path) -> list[dict[str, Any]]:
    """Every line, verified as a chain (module docstring). Raises :class:`MismatchError`."""
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise MismatchError("the last line is torn (no trailing newline)")
    events: list[dict[str, Any]] = []
    previous_hash = ZERO_HASH
    previous_ts: datetime | None = None
    mode: str | None = None
    for number, line in enumerate(raw.splitlines()):
        where = f"line {number + 1}"
        try:
            event = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MismatchError(f"{where} is not JSON: {exc}") from exc
        if not isinstance(event, dict) or set(event) != LEDGER_KEYS:
            raise MismatchError(f"{where} does not carry exactly the ledger fields")
        check_types(event, where)
        if canonical(event) != line:
            raise MismatchError(f"{where} is not in canonical form (edited after it was written)")
        if event["seq"] != number:
            raise MismatchError(f"{where} has seq {event['seq']!r}, expected {number}")
        if event["prev_hash"] != previous_hash:
            raise MismatchError(f"{where} does not link to the line before it")
        if event_hash(event) != event["hash"]:
            raise MismatchError(f"{where}: the hash does not match the content (seq {number})")
        if mode is None:
            mode = event["mode"]
        elif event["mode"] != mode:
            raise MismatchError(f"{where} is a {event['mode']!r} event in a {mode!r} ledger")
        ts = parse_time(event["ts"])
        if previous_ts is not None and ts < previous_ts:
            raise MismatchError(f"{where}: time runs backwards")
        if event["kind"] == "genesis" and number != 0:
            raise MismatchError(f"{where}: a genesis after seq 0")
        previous_hash = event["hash"]
        previous_ts = ts
        events.append(event)
    return events


def check_genesis(events: Sequence[dict[str, Any]]) -> str:
    if not events:
        return "empty ledger"
    first = events[0]
    if first["kind"] != "genesis":
        if first["mode"] == "paper":
            raise MismatchError("the paper ledger does not start with its genesis")
        return f"no genesis ({first['mode']} ledger)"
    payload = first["payload"]
    if not isinstance(payload, dict) or "policy" not in payload:
        raise MismatchError("the genesis carries no policy")
    recomputed = sha256_hex(canonical(payload["policy"]))
    if recomputed != payload.get("policy_hash"):
        raise MismatchError("the genesis policy_hash is not the hash of the policy it carries")
    return f"genesis policy_hash {recomputed[:12]}... matches its policy"


def blob_refs(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if set(value) == {"sha256", "media_type", "size"}:
            yield value
            return
        for item in value.values():
            yield from blob_refs(item)
    elif isinstance(value, list):
        for item in value:
            yield from blob_refs(item)


def check_blobs(root: Path, events: Sequence[dict[str, Any]]) -> str:
    wanted: dict[str, int] = {}
    for event in events:
        for ref in (*event["blobs"], *blob_refs(event["payload"])):
            sha, size = ref["sha256"], ref["size"]
            if wanted.setdefault(sha, size) != size:
                raise MismatchError(f"blob {sha[:12]}... is referenced with two different sizes")
    for sha, size in sorted(wanted.items()):
        path = root / "blobs" / sha
        if not path.is_file():
            raise MismatchError(f"blob {sha[:12]}... is referenced but not published")
        data = path.read_bytes()
        if sha256_hex(data) != sha:
            raise MismatchError(f"blob {sha[:12]}... does not hash to its name")
        if len(data) != size:
            raise MismatchError(f"blob {sha[:12]}... is {len(data)} bytes, the ledger says {size}")
    return f"{len(wanted)} referenced, all present and intact"


# ================================================================================================
# CSV files
# ================================================================================================


def read_csv(path: Path, required: Sequence[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise MismatchError(f"{path.name} lacks the columns {', '.join(missing)}")
        return list(reader)


def check_equity(
    rows: Sequence[dict[str, str]], events: Sequence[dict[str, Any]]
) -> list[tuple[datetime, Decimal]]:
    marks = [e["payload"] for e in events if e["kind"] == "mark"]
    if len(rows) != len(marks):
        raise MismatchError(f"{len(rows)} rows, but the ledger holds {len(marks)} mark events")
    out: list[tuple[datetime, Decimal]] = []
    previous: datetime | None = None
    for index, (row, mark) in enumerate(zip(rows, marks, strict=True)):
        at = parse_time(row["at"])
        equity = dec(row["equity_book"], "equity_book")
        if at != parse_time(mark["at"]) or equity != dec(mark["equity_book"], "equity_book"):
            raise MismatchError(f"row {index + 1} ({row['at']}) differs from the ledger's mark")
        if (at.minute, at.second, at.microsecond) != (0, 0, 0):
            raise MismatchError(f"row {index + 1} ({row['at']}) is not on the hour")
        if previous is not None and at <= previous:
            raise MismatchError(f"row {index + 1} ({row['at']}) does not come after the row before")
        previous = at
        out.append((at, equity))
    return out


# ================================================================================================
# Trades, rebuilt from the fills
# ================================================================================================


@dataclass
class OpenTrade:
    direction: int
    qty: Decimal
    avg: Decimal
    opened_at: datetime
    realized: Decimal
    fees: Decimal
    entry_qty: Decimal
    entry_value: Decimal
    exit_qty: Decimal = Decimal(0)
    exit_value: Decimal = Decimal(0)
    max_abs_qty: Decimal = Decimal(0)


@dataclass(frozen=True)
class FillRow:
    exec_id: str
    symbol: str
    direction: int
    qty: Decimal
    price: Decimal
    value: Decimal
    fee: Decimal
    closing_first: int
    at: datetime
    ledger_index: int
    """Index of the event that first logged this fill: the marks after it include it."""

    @property
    def key(self) -> tuple[datetime, int, int, str]:
        """The book's order (``book/book.py`` ``_Record.key``): venue time, a closing fill first,
        then the venue's trade id, shorter first. Never the ledger's arrival order."""
        return (self.at, self.closing_first, len(self.exec_id), self.exec_id)


def fills_of(events: Sequence[dict[str, Any]]) -> tuple[list[FillRow], int]:
    """Unique fills in ledger order, and how many identical repeats were ignored."""
    seen: dict[str, dict[str, Any]] = {}
    rows: list[FillRow] = []
    repeats = 0
    for index, event in enumerate(events):
        if event["kind"] != "fill":
            continue
        payload = event["payload"]
        exec_id = payload["exec_id"]
        if exec_id in seen:
            if seen[exec_id] != payload:
                raise MismatchError(f"fill {exec_id} appears twice with different content")
            repeats += 1
            continue
        seen[exec_id] = payload
        if str(payload.get("fee_coin", "")).strip().upper() != "USDT":
            raise MismatchError(f"fill {exec_id} paid its fee in {payload.get('fee_coin')!r}")
        side = payload["side"]
        if side not in ("buy", "sell"):
            raise MismatchError(f"fill {exec_id} has side {side!r}")
        qty = dec(payload["exec_qty"], "exec_qty")
        price = dec(payload["exec_price"], "exec_price")
        if qty <= 0 or price <= 0:
            raise MismatchError(f"fill {exec_id} has a non-positive price or quantity")
        rows.append(
            FillRow(
                exec_id=exec_id,
                symbol=payload["symbol"],
                direction=1 if side == "buy" else -1,
                qty=qty,
                price=price,
                value=dec(payload["exec_value"], "exec_value"),
                fee=dec(payload["fee_paid"], "fee_paid"),
                closing_first=0 if payload.get("trade_side") == "close" else 1,
                at=parse_time(payload["executed_at"]),
                ledger_index=index,
            )
        )
    return rows, repeats


@dataclass
class Folded:
    """The book after a set of fills: closed trades (with their closing keys), open trades, and
    the realized P&L and fees of every fill."""

    closed: list[tuple[tuple[datetime, int, int, str], dict[str, Any]]]
    open: dict[str, OpenTrade]
    realized: Decimal
    fees: Decimal


def fold(fills: Sequence[FillRow]) -> Folded:
    """Every fill, by the book's rules (module docstring), per symbol in the book's order."""
    by_symbol: dict[str, list[FillRow]] = {}
    for fill in fills:
        by_symbol.setdefault(fill.symbol, []).append(fill)
    closed: list[tuple[tuple[datetime, int, int, str], dict[str, Any]]] = []
    still_open: dict[str, OpenTrade] = {}
    realized = Decimal(0)
    fees = Decimal(0)
    with localcontext(BOOK_CONTEXT):
        for symbol, rows in by_symbol.items():
            rows = sorted(rows, key=lambda f: f.key)
            trade: OpenTrade | None = None
            for fill in rows:
                fees += fill.fee
                remaining = fill.qty
                fee_left = fill.fee
                if trade is not None and trade.direction != fill.direction:
                    closing = min(fill.qty, abs(trade.qty))
                    close_fee = fill.fee if closing == fill.qty else fill.fee * closing / fill.qty
                    pnl = (fill.price - trade.avg) * closing * trade.direction
                    trade.realized += pnl
                    realized += pnl
                    trade.fees += close_fee
                    trade.exit_qty += closing
                    trade.exit_value += closing * fill.price
                    trade.qty -= closing * trade.direction
                    remaining = fill.qty - closing
                    fee_left = fill.fee - close_fee
                    if trade.qty == 0:
                        closed.append(
                            (
                                fill.key,
                                {
                                    "symbol": symbol,
                                    "opened_at": trade.opened_at,
                                    "closed_at": fill.at,
                                    "direction": trade.direction,
                                    "entry_avg": trade.entry_value / trade.entry_qty,
                                    "exit_avg": trade.exit_value / trade.exit_qty,
                                    "max_abs_qty": trade.max_abs_qty,
                                    "gross_pnl": trade.realized,
                                    "fees": trade.fees,
                                    "net_pnl": trade.realized - trade.fees,
                                },
                            )
                        )
                        trade = None
                if remaining <= 0:
                    continue
                if trade is None:
                    trade = OpenTrade(
                        direction=fill.direction,
                        qty=remaining * fill.direction,
                        avg=fill.price,
                        opened_at=fill.at,
                        realized=Decimal(0),
                        fees=fee_left,
                        entry_qty=remaining,
                        entry_value=remaining * fill.price,
                        max_abs_qty=remaining,
                    )
                    continue
                held = abs(trade.qty)
                trade.avg = (trade.avg * held + fill.price * remaining) / (held + remaining)
                trade.qty += remaining * fill.direction
                trade.fees += fee_left
                trade.entry_qty += remaining
                trade.entry_value += remaining * fill.price
                trade.max_abs_qty = max(trade.max_abs_qty, abs(trade.qty))
            if trade is not None:
                still_open[symbol] = trade
    closed.sort(key=lambda item: item[0])
    return Folded(closed=closed, open=still_open, realized=realized, fees=fees)


def rebuild_trades(fills: Sequence[FillRow]) -> list[dict[str, Any]]:
    """Closed trades by the book's rules (module docstring), in the order they closed."""
    return [trade for _, trade in fold(fills).closed]


# ================================================================================================
# Marks, rebuilt from the fills
# ================================================================================================


def logged_start(events: Sequence[dict[str, Any]], first_fill: int | None) -> Decimal | None:
    """The venue's account equity as first logged before any fill (``book/projection.py``): a
    passed environment proof's account, or a reconciliation's. ``None`` when the log has none."""
    for index, event in enumerate(events):
        if first_fill is not None and index >= first_fill:
            return None
        payload = event["payload"]
        if event["kind"] == "environment_proof" and payload.get("passed") is not True:
            continue
        if event["kind"] not in ("environment_proof", "reconciliation"):
            continue
        account = payload.get("account")
        if not isinstance(account, dict) or account.get("equity_usdt") is None:
            continue
        equity = dec(account["equity_usdt"], "account equity")
        if equity > 0:
            return equity
    return None


def check_marks_rebuilt(events: Sequence[dict[str, Any]], fills: Sequence[FillRow]) -> str:
    """Every mark event recomputed from the fills logged before it (module docstring)."""
    marks = [(i, e["payload"]) for i, e in enumerate(events) if e["kind"] == "mark"]
    if not marks:
        if fills:
            raise MismatchError("the ledger holds fills but no mark")
        return "no marks yet"
    first_fill = min((f.ledger_index for f in fills), default=None)
    first_index, first = marks[0]
    if first_fill is not None and first_index > first_fill:
        raise MismatchError(
            "the first mark comes after the first fill: the return series would start after the "
            "book was built and drop its first costs"
        )
    if first.get("positions"):
        raise MismatchError("the first mark carries positions; it must be the flat start")
    start = logged_start(events, first_fill)
    first_equity = dec(first["equity_book"], "equity_book")
    if start is None:
        start = first_equity
        source = "the first mark (no account equity logged)"
    else:
        source = "the logged account equity"
    if not same_decimal(first_equity, start):
        raise MismatchError(
            f"the first mark's equity {first_equity} is not the starting equity {start} ({source})"
        )
    with localcontext(BOOK_CONTEXT):
        for index, payload in marks:
            where = f"mark {payload.get('at')}"
            book = fold([f for f in fills if f.ledger_index < index])
            held = {s: t for s, t in book.open.items() if t.qty != 0}
            rows = payload.get("positions") or []
            if not isinstance(rows, list):
                raise MismatchError(f"{where}: positions is not a list")
            listed = {str(r["symbol"]): r for r in rows}
            if len(listed) != len(rows):
                raise MismatchError(f"{where}: a symbol is listed twice")
            if set(listed) != set(held):
                raise MismatchError(
                    f"{where}: positions {sorted(listed)} differ from the fills' {sorted(held)}"
                )
            unrealized_total = Decimal(0)
            for symbol, row in listed.items():
                trade = held[symbol]
                qty = dec(row["qty"], "qty")
                if not same_decimal(qty, trade.qty):
                    raise MismatchError(f"{where}: {symbol} qty {qty}, the fills hold {trade.qty}")
                mark = dec(row["demo_mark"], "demo_mark")
                if mark <= 0:
                    raise MismatchError(f"{where}: {symbol} has a non-positive Demo mark")
                expected = trade.qty * (mark - trade.avg)
                stated = dec(row["unrealized_demo"], "unrealized_demo")
                if not same_decimal(stated, expected):
                    raise MismatchError(
                        f"{where}: {symbol} unrealized {stated}, recomputed {expected}"
                    )
                unrealized_total += expected
            equity = start + book.realized - book.fees + unrealized_total
            stated_equity = dec(payload["equity_book"], "equity_book")
            if not same_decimal(stated_equity, equity):
                raise MismatchError(
                    f"{where}: equity_book {stated_equity}, rebuilt from the fills {equity}"
                )
    return f"{len(marks)} marks rebuilt from the fills, starting at {start} ({source})"


def same_decimal(a: Decimal, b: Decimal) -> bool:
    return abs(a - b) <= DECIMAL_TOL * max(Decimal(1), abs(a), abs(b))


def check_trades(
    rows: Sequence[dict[str, str]], rebuilt: Sequence[dict[str, Any]]
) -> list[Decimal]:
    if len(rows) != len(rebuilt):
        raise MismatchError(f"{len(rows)} rows, but the fills close {len(rebuilt)} trades")
    net: list[Decimal] = []
    for index, (row, trade) in enumerate(zip(rows, rebuilt, strict=True)):
        where = f"trade {index + 1} ({row.get('symbol')})"
        if row["symbol"] != trade["symbol"]:
            raise MismatchError(f"{where}: symbol differs from the fills ({trade['symbol']})")
        if int(row["direction"]) != trade["direction"]:
            raise MismatchError(f"{where}: direction differs from the fills")
        for column in ("opened_at", "closed_at"):
            if parse_time(row[column]) != trade[column]:
                raise MismatchError(f"{where}: {column} differs from the fills")
        for column in ("entry_avg", "exit_avg", "max_abs_qty", "gross_pnl", "fees", "net_pnl"):
            if not same_decimal(dec(row[column], column), trade[column]):
                raise MismatchError(
                    f"{where}: {column} {row[column]} differs from the fills ({trade[column]})"
                )
        net.append(dec(row["net_pnl"], "net_pnl"))
    return net


# ================================================================================================
# Metrics, written out again
# ================================================================================================


def f_mean(x: Sequence[float]) -> float:
    return math.fsum(x) / len(x)


def f_pstdev(x: Sequence[float]) -> float:
    if max(x) == min(x):
        return 0.0
    centre = f_mean(x)
    return math.sqrt(math.fsum((v - centre) * (v - centre) for v in x) / len(x))


def returns_of(equity: Sequence[float]) -> list[float]:
    out = []
    for i in range(1, len(equity)):
        if equity[i - 1] <= 0:
            raise MismatchError("a non-positive equity leaves a return undefined")
        out.append(equity[i] / equity[i - 1] - 1.0)
    return out


def s_sharpe(r: Sequence[float]) -> float | None:
    if not r:
        return None
    sd = f_pstdev(r)
    return None if sd == 0.0 else f_mean(r) / sd * math.sqrt(YEAR_HOURS)


def s_sharpe_se(r: Sequence[float]) -> float | None:
    if not r:
        return None
    sd = f_pstdev(r)
    if sd == 0.0:
        return None
    hourly = f_mean(r) / sd
    return math.sqrt((1.0 + hourly * hourly / 2.0) / len(r)) * math.sqrt(YEAR_HOURS)


def s_sortino(r: Sequence[float]) -> float | None:
    if not r:
        return None
    down = math.sqrt(math.fsum(v * v for v in r if v < 0.0) / len(r))
    return None if down == 0.0 else f_mean(r) / down * math.sqrt(YEAR_HOURS)


def s_drawdown(equity: Sequence[float]) -> float:
    worst, top = 0.0, -math.inf
    for value in equity:
        top = max(top, value)
        if top <= 0:
            raise MismatchError("a non-positive equity peak leaves a drawdown undefined")
        worst = min(worst, value / top - 1.0)
    return worst


def path_total(r: Sequence[float]) -> float:
    level = 1.0
    for value in r:
        level = level * (1.0 + value)
    return level - 1.0


def path_drawdown(r: Sequence[float]) -> float:
    level, top, worst = 1.0, 1.0, 0.0
    for value in r:
        level = level * (1.0 + value)
        top = max(top, level)
        worst = min(worst, level / top - 1.0)
    return worst


def path_sharpe(r: Sequence[float]) -> float | None:
    """The Sharpe ratio of a resample; ``0`` when every return is exactly zero."""
    if r and max(r) == 0.0 and min(r) == 0.0:
        return 0.0
    return s_sharpe(r)


BOOT_STATS: dict[str, Callable[[Sequence[float]], float | None]] = {
    "total_return": path_total,
    "sharpe_ann": path_sharpe,
    "sortino_ann": s_sortino,
    "max_drawdown": path_drawdown,
}


def cube_block(n: int) -> int:
    b = 1
    while b * b * b < n:
        b += 1
    return b


def intervals(r: Sequence[float]) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    """The four intervals over the resamples where each statistic is defined, and the share of
    resamples where it is not (non-zero shares only)."""
    n = len(r)
    if n < 2:
        return {}, {}
    b = cube_block(n)
    rng = random.Random(BOOT_SEED)  # noqa: S311 - a reproducible resampler, not a secret
    blocks = -(-n // b)
    samples: dict[str, list[float]] = {name: [] for name in CI_NAMES}
    undefined: dict[str, int] = dict.fromkeys(CI_NAMES, 0)
    for _ in range(BOOT_RESAMPLES):
        starts = [rng.randrange(n - b + 1) for _ in range(blocks)]
        path = [r[s + k] for s in starts for k in range(b)][:n]
        for name in CI_NAMES:
            value = BOOT_STATS[name](path)
            if value is None or not math.isfinite(value):
                undefined[name] += 1
            else:
                samples[name].append(value)
    q = (1.0 - BOOT_LEVEL) / 2.0
    out: dict[str, tuple[float, float]] = {}
    shares: dict[str, float] = {}
    for name in CI_NAMES:
        if undefined[name]:
            shares[name] = undefined[name] / BOOT_RESAMPLES
        v = sorted(samples[name])
        if v:
            out[name] = (v[int(q * (len(v) - 1))], v[int((1.0 - q) * (len(v) - 1))])
    return out, shares


def recompute_metrics(
    equity: Sequence[float],
    net_pnl: Sequence[Decimal],
    *,
    notional: float | None,
    fees: float | None,
    with_intervals: Iterable[str] | None,
) -> dict[str, Any]:
    """Every metric from an equity series and trade results. ``with_intervals=None`` computes all
    four intervals; otherwise only the named ones."""
    r = returns_of(equity)
    out: dict[str, Any] = {
        "n_hours": len(r),
        "n_closed_trades": len(net_pnl),
        "total_return": equity[-1] / equity[0] - 1.0 if equity else 0.0,
        "sharpe_ann": s_sharpe(r),
        "sharpe_se_ann": s_sharpe_se(r),
        "sortino_ann": s_sortino(r),
        "max_drawdown": s_drawdown(equity),
        "win_rate": sum(1 for x in net_pnl if x > 0) / len(net_pnl) if net_pnl else None,
        "label": LABEL,
    }
    if notional is not None:
        mean_equity = f_mean(equity) if equity else None
        if mean_equity is None:
            out["turnover"] = 0.0
        elif mean_equity <= 0:
            raise MismatchError("mean equity is not positive")
        else:
            out["turnover"] = notional / mean_equity
    if fees is not None:
        out["fees_paid"] = fees
    wanted = set(CI_NAMES) if with_intervals is None else set(with_intervals)
    bands, shares = intervals(r) if wanted else ({}, {})
    out["ci90"] = {k: v for k, v in bands.items() if k in wanted}
    out["ci90_undefined_share"] = {k: v for k, v in shares.items() if k in wanted}
    return out


def same_float(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(float(a), float(b), rel_tol=FLOAT_TOL, abs_tol=1e-12)


def compare_metrics(
    published: dict[str, Any], recomputed: dict[str, Any], *, exact_ci_keys: bool
) -> None:
    for key in ("n_hours", "n_closed_trades"):
        if published.get(key) != recomputed[key]:
            raise MismatchError(f"{key} is {published.get(key)!r}, recomputed {recomputed[key]!r}")
    for key in METRIC_FLOATS:
        if key not in recomputed:
            continue
        if key not in published:
            raise MismatchError(f"{key} is missing")
        if not same_float(published[key], recomputed[key]):
            raise MismatchError(f"{key} is {published[key]!r}, recomputed {recomputed[key]!r}")
    if published.get("label") != LABEL:
        raise MismatchError(f"label is {published.get('label')!r}, not {LABEL!r}")
    stated = published.get("ci90")
    if not isinstance(stated, dict):
        raise MismatchError("ci90 is missing")
    computed = recomputed["ci90"]
    if exact_ci_keys and set(stated) != set(computed):
        raise MismatchError(
            f"ci90 names {sorted(stated)}, recomputed {sorted(computed)} "
            "(an interval is stated that cannot be computed, or one is missing)"
        )
    for name, bounds in stated.items():
        if name not in computed:
            raise MismatchError(f"ci90 {name} is stated but undefined on the data")
        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
            or not all(same_float(a, b) for a, b in zip(bounds, computed[name], strict=True))
        ):
            raise MismatchError(f"ci90 {name} is {bounds!r}, recomputed {list(computed[name])!r}")
    shares = published.get("ci90_undefined_share", {})
    if not isinstance(shares, dict):
        raise MismatchError("ci90_undefined_share is not an object")
    expected = recomputed["ci90_undefined_share"]
    checked = set(expected) if exact_ci_keys else set(stated)
    for name in sorted(checked | (set(shares) if exact_ci_keys else set())):
        if not same_float(shares.get(name, 0.0), expected.get(name, 0.0)):
            raise MismatchError(
                f"ci90_undefined_share {name} is {shares.get(name, 0.0)!r}, recomputed "
                f"{expected.get(name, 0.0)!r}"
            )


def check_arms(path: Path) -> str:
    arms = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(arms, list):
        raise MismatchError("arms.json is not a JSON array of arm results")
    for arm in arms:
        arm_id = arm["spec"]["arm_id"]
        try:
            equity = [float(m["equity"]) for m in arm["marks"]]
            stamps = [parse_time(m["at"]) for m in arm["marks"]]
            if any(later <= earlier for earlier, later in itertools.pairwise(stamps)):
                raise MismatchError("marks do not strictly increase")
            net = [dec(t["net_pnl"], "net_pnl") for t in arm["trades"]]
            stated = arm["metrics"]
            recomputed = recompute_metrics(
                equity,
                net,
                notional=None,
                fees=None,
                with_intervals=list(stated.get("ci90", {})),
            )
            if stated.get("arm_id") != arm_id:
                raise MismatchError(f"metrics name arm {stated.get('arm_id')!r}")
            compare_metrics(stated, recomputed, exact_ci_keys=False)
        except MismatchError as exc:
            raise MismatchError(f"arm {arm_id}: {exc}") from exc
    return f"{len(arms)} arms recompute (turnover and fees need unpublished fills)"


# ================================================================================================
# Main
# ================================================================================================


def verify(root: Path) -> Report:
    report = Report()

    def step(name: str, action: Callable[[], str]) -> bool:
        try:
            report.ok(name, action())
            return True
        except MismatchError as exc:
            report.fail(name, str(exc))
        except (OSError, KeyError, TypeError, ValueError) as exc:
            report.fail(name, f"{type(exc).__name__}: {exc}")
        return False

    events: list[dict[str, Any]] = []
    marks: list[tuple[datetime, Decimal]] = []
    fills: list[FillRow] = []
    net: list[Decimal] = []

    def ledger() -> str:
        events.extend(read_ledger(root / "ledger.jsonl"))
        head = events[-1]["hash"][:12] + "..." if events else "none"
        mode = events[0]["mode"] if events else "-"
        return f"{len(events)} events chain intact, head {head}, mode {mode}"

    chain_ok = step("ledger chain", ledger)
    if chain_ok:
        step("genesis", lambda: check_genesis(events))
        step("blobs", lambda: check_blobs(root, events))

        def equity() -> str:
            rows = read_csv(root / "equity_hourly.csv", ("at", "equity_book"))
            marks.extend(check_equity(rows, events))
            return f"{len(marks)} hourly marks match the ledger's mark events"

        def trades() -> str:
            found, repeats = fills_of(events)
            fills.extend(found)
            rows = read_csv(root / "trades.csv", TRADE_COLUMNS)
            net.extend(check_trades(rows, rebuild_trades(found)))
            note = f"; {repeats} repeated fill events ignored" if repeats else ""
            return f"{len(net)} closed trades rebuilt from {len(found)} fills{note}"

        equity_ok = step("equity_hourly.csv", equity)
        trades_ok = step("trades.csv", trades)
        if trades_ok:
            rebuilt_ok = step("marks from fills", lambda: check_marks_rebuilt(events, fills))
            equity_ok = equity_ok and rebuilt_ok
        else:
            report.fail("marks from fills", "not rebuilt: the fills did not verify")
            equity_ok = False

        def metrics() -> str:
            if not (equity_ok and trades_ok):
                raise MismatchError("not recomputed: its inputs did not verify")
            published = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
            if not isinstance(published, dict):
                raise MismatchError("metrics.json is not an object")
            if published.get("arm_id") != BOOK_ARM_ID:
                raise MismatchError(f"metrics.json is for {published.get('arm_id')!r}")
            notional = sum((abs(f.value) for f in fills), Decimal(0))
            paid = sum((f.fee for f in fills), Decimal(0))
            recomputed = recompute_metrics(
                [float(e) for _, e in marks],
                net,
                notional=float(notional),
                fees=float(paid),
                with_intervals=None,
            )
            compare_metrics(published, recomputed, exact_ci_keys=True)
            sharpe = recomputed["sharpe_ann"]
            shown = "undefined" if sharpe is None else f"{sharpe:.2f}"
            return (
                f"every field agrees: Sharpe {shown}, max drawdown "
                f"{recomputed['max_drawdown']:.4%}, {recomputed['n_hours']} hours"
            )

        step("metrics.json", metrics)
    if (root / "arms.json").exists():
        step("arms.json", lambda: check_arms(root / "arms.json"))
    return report


def main(argv: Sequence[str]) -> int:
    if len(argv) != 1:
        print("usage: python scripts/recompute.py <public directory>", file=sys.stderr)
        return 2
    root = Path(argv[0])
    if not root.is_dir():
        print(f"{root} is not a directory", file=sys.stderr)
        return 2
    report = verify(root)
    print(report.table())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
