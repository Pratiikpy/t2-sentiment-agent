"""Bitget Agent Hub transport: every paper order is sent by Bitget's own CLI, ``bgc``.

The CLI is ``@bitget-ai/bitget-agent-cli@3.0.0`` with ``@bitget-ai/bitget-agent-sdk@3.0.0``, both
pinned in ``tools/agent-hub/package-lock.json`` and run as
``node tools/agent-hub/node_modules/@bitget-ai/bitget-agent-cli/lib/index.js``. This project never
signs a request itself; the SDK does (``agent-sdk/src/client/rest-client.ts:266-294``).

**Where orders go is decided by one constant.** Every argv this transport builds ends with
``--paper-trading`` (:data:`~sentiment_agent.execution.environment.PAPER_FLAG`); no parameter
removes it, and :meth:`BgcTransport._call` refuses any argv without it or with ``--read-only``. The
only non-paper argv in the project is the environment proof's live-negative read
(:mod:`sentiment_agent.execution.environment`).

**Credentials reach ``bgc`` only through a child environment built from scratch** (the Demo values
plus ``PATH`` and ``SYSTEMROOT``), never on the argv, in a log or in a blob. A dry-run preview gets
no credential at all: the SDK returns the preview before its credential gate
(``tools/safety.ts:86-88`` against ``rest-client.ts:152-165``), confirmed by running it
(``tests/fixtures/execution/dryrun_*.json``).

**Writes are never retried below us.** The SDK retries a POST that carries a ``clientOid`` after a
network error (``utils/retry.ts:43-55``); a retry that lands after the first attempt succeeded would
come back as a duplicate-clientOid refusal for an order that exists. Writes therefore run with
``BITGET_MAX_RETRIES=0``: a network failure on a send becomes
:class:`~sentiment_agent.types.VenueUnknown` and is resolved by reading the order back by its
``clientOid``, never by sending again. Reads keep the SDK's two retries.

**How answers are read.** ``bgc`` prints JSON on stdout with exit 0, or a structured error on stderr
with exit 1 (``agent-cli/src/index.ts:235-243``). Order, fill, position, strategy-order and account
fields follow Bitget's UTA documentation (legacy-docs/uta, fetched 2026-09-24; examples recorded in
``tests/fixtures/execution/``). The order and fill shapes are documented and parsed; the position,
strategy-order and account shapes are documented too but not yet confirmed by a Demo response, so
only the fields needed are parsed and the full answer always travels in the blob (DESIGN.md §20).
Two reading rules differ from the SDK's own helpers, deliberately:

* Fills and order history are paged with the ``cursor`` the venue returns at the top level of each
  page (documented), not with the SDK's ``fetchAll``, which takes the cursor from an id field of
  the last row and stops at 5 pages (``tools/paginate.ts:144-158, 173-174``). A walk that would
  exceed :data:`MAX_PAGES` raises instead of truncating.
* Fees are summed as absolute values. Every order here is a taker order (guard G8), so every fee is
  a cost whichever sign the venue reports; Bitget's documented fill example is positive
  (``fee "0.6417006"`` on a 1,069.50 fill), its position ``openFeeTotal`` negative. The raw lines
  are kept in the blob until the first Demo fill pins the convention.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Final, Protocol

from sentiment_agent.execution.environment import (
    CATEGORY,
    CLI_ENTRY,
    DRY_RUN_FLAG,
    HOLD_MODES,
    PAPER_FLAG,
    READ_ONLY_FLAG,
    BgcFailure,
    DemoCredentials,
    EnvironmentRefused,
    account_from_assets,
    base_child_env,
    failure_of,
    result_blob,
    result_mentions_environment_mismatch,
)
from sentiment_agent.types import (
    CLIENT_OID_HEX,
    CLIENT_OID_PREFIX,
    AccountSnapshot,
    ApprovedOrder,
    BlobRef,
    BlobStore,
    Clock,
    DryRunPreview,
    EnvironmentProof,
    FeeLine,
    Fill,
    FillVenue,
    OrderIntent,
    RunMode,
    Side,
    StopSync,
    VenueAck,
    VenueOrder,
    VenueOrderStatus,
    VenuePosition,
    VenueRejection,
    VenueStopOrder,
    VenueUnknown,
)

BGC_PACKAGE: Final = "@bitget-ai/bitget-agent-cli@3.0.0"

HOLD_ONE_WAY: Final = "one_way_mode"
HOLD_HEDGE: Final = "hedge_mode"
STOP_TRIGGER: Final = "mark"
"""Stops trigger on the mark price (guard G4; ``Policy.stop_trigger`` admits no other value)."""

PLACE_ORDER_PATH: Final = "/api/v3/trade/place-order"
READ_TIMEOUT_S: Final = 120.0
WRITE_TIMEOUT_S: Final = 45.0
REQUEST_TIMEOUT_MS: Final = "15000"
READ_RETRIES: Final = "2"
WRITE_RETRIES: Final = "0"
PAGE_LIMIT: Final = 100
MAX_PAGES: Final = 200
QUERY_WINDOW: Final = timedelta(days=30) - timedelta(seconds=1)
"""History and fills accept at most 30 days per query (legacy-docs/uta/trade/Get-Order-Fills)."""

_SYMBOL = re.compile(r"^[A-Z0-9]{2,40}$")
_VENUE_ID = re.compile(r"^[0-9A-Za-z_.:/][0-9A-Za-z_.:/-]{0,63}$")
_CLIENT_OID = re.compile(rf"^{CLIENT_OID_PREFIX}[0-9a-f]{{{CLIENT_OID_HEX}}}$")
_NOT_FOUND = re.compile(r"not\s+(?:be\s+)?found|does\s+not\s+exist|no\s+such\s+order", re.I)
_DUPLICATE_OID = re.compile(r"duplicate.*client|client.*(?:oid|order).*(?:exist|duplicate)", re.I)


# --- running bgc ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BgcResult:
    """One ``bgc`` run. ``stdout``/``stderr`` are the parsed JSON, ``{"text": ...}`` when the CLI
    printed plain text (its own argument errors), or ``None`` when empty."""

    exit_code: int
    stdout: dict[str, Any] | None
    stderr: dict[str, Any] | None
    duration_ms: int


class BgcRunner(Protocol):
    def __call__(
        self, args: Sequence[str], *, env: Mapping[str, str], timeout_s: float
    ) -> BgcResult: ...


class BgcUnavailableError(RuntimeError):
    """Node or the pinned CLI is not installed (``npm ci --ignore-scripts`` in tools/agent-hub)."""


class BgcTimeoutError(TimeoutError):
    """``bgc`` did not finish in time. For a send, the outcome is unknown, never "rejected"."""


class TransportRefusedError(RuntimeError):
    """The transport refused locally. Nothing was sent to the venue."""


class DryRunOnlyError(TransportRefusedError):
    """A send or a private read was asked of a transport built for dry-run previews only."""


class PreviewMismatchError(TransportRefusedError):
    """``bgc --dry-run`` would send something other than the approved intent."""


class VenueParseError(ValueError):
    """A venue row lacked a field this project needs, or carried a value it cannot interpret."""


class VenueCallError(RuntimeError):
    """A venue call failed. ``failure`` is ``bgc``'s structured error; ``blob`` the evidence."""

    def __init__(
        self,
        message: str,
        *,
        failure: BgcFailure | None = None,
        blob: BlobRef | None = None,
        outcome_unknown: bool = False,
    ) -> None:
        super().__init__(message)
        self.failure = failure
        self.blob = blob
        self.outcome_unknown = outcome_unknown


class VenueReadError(VenueCallError):
    """A read failed or could not be interpreted. Nothing about the venue may be assumed from it."""


class VenueWriteError(VenueCallError):
    """A stop placement or cancellation failed; ``outcome_unknown`` when it timed out."""


def parse_output(raw: bytes) -> dict[str, Any] | None:
    """Parse one stream of ``bgc`` output (tolerating a Node warning printed before the JSON)."""
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return None
    start = text.find("{")
    for candidate in (text, text[start:] if start > 0 else ""):
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return {"text": text[:8000]}


def _check_args(args: Sequence[str]) -> None:
    for arg in args:
        if not isinstance(arg, str) or any(ch in arg for ch in "\x00\r\n"):
            raise TransportRefusedError(f"refusing a malformed bgc argument: {arg!r}")


class SubprocessBgcRunner:
    """Runs the pinned CLI with Node, with exactly the environment the caller supplies."""

    def __init__(self, agent_hub_dir: Path, *, node: str | None = None) -> None:
        self._dir = agent_hub_dir.resolve()
        self._entry = self._dir / CLI_ENTRY
        if not self._entry.is_file():
            raise BgcUnavailableError(
                f"{self._entry} is missing; run `npm ci --ignore-scripts` in tools/agent-hub"
            )
        resolved = node or shutil.which("node")
        if not resolved:
            raise BgcUnavailableError("node (>= 20) is not on PATH")
        self._node = resolved

    @property
    def entry(self) -> Path:
        return self._entry

    def __call__(
        self, args: Sequence[str], *, env: Mapping[str, str], timeout_s: float
    ) -> BgcResult:
        _check_args(args)
        argv = [self._node, str(self._entry), *args]
        started = time.monotonic()
        try:
            proc = subprocess.run(
                argv,
                env=dict(env),
                cwd=self._dir,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            verb = " ".join(args[:3])
            raise BgcTimeoutError(f"bgc {verb} did not finish within {timeout_s:.0f} s") from None
        return BgcResult(
            exit_code=proc.returncode,
            stdout=parse_output(proc.stdout),
            stderr=parse_output(proc.stderr),
            duration_ms=int((time.monotonic() - started) * 1000),
        )


# --- argv -----------------------------------------------------------------------------------


def fmt_decimal(value: Decimal) -> str:
    """A positive decimal as plain text, no exponent, no trailing zeros (``Decimal("100")`` is
    ``"100"``, ``Decimal("0.050")`` is ``"0.05"``)."""
    if not value.is_finite() or value <= 0:
        raise ValueError(f"expected a positive finite decimal, got {value}")
    return format(value.normalize(), "f")


def _require_symbol(symbol: str) -> str:
    if not _SYMBOL.fullmatch(symbol):
        raise ValueError(f"not a venue symbol: {symbol!r}")
    return symbol


def _require_venue_id(venue_id: str) -> str:
    if not _VENUE_ID.fullmatch(venue_id):
        raise ValueError(f"not a venue order id: {venue_id!r}")
    return venue_id


def _require_hold_mode(hold_mode: str) -> str:
    if hold_mode not in HOLD_MODES:
        raise ValueError(f"unknown hold mode {hold_mode!r}; expected one of {sorted(HOLD_MODES)}")
    return hold_mode


def position_side(side: Side, *, adds_exposure: bool) -> str:
    """The position book an order acts on: buying opens a long or closes a short."""
    buys = side is Side.BUY
    return "long" if buys == adds_exposure else "short"


def would_send(intent: OrderIntent, *, hold_mode: str) -> dict[str, str]:
    """The request body ``bgc`` must send for ``intent``; the dry-run preview is checked against it.

    One-way mode marks reducing legs ``reduceOnly: yes``. Hedge mode names the position book with
    ``posSide`` instead, because Bitget applies ``reduceOnly`` in one-way mode only
    (legacy-docs/uta/trade/Place-Order, fetched 2026-09-24). Exposure-adding legs carry their stop
    (G4), triggered on mark and executed at market.
    """
    _require_hold_mode(hold_mode)
    payload = {
        "category": intent.category.value,
        "symbol": _require_symbol(intent.symbol),
        "side": intent.side.value,
        "orderType": intent.order_type,
        "qty": fmt_decimal(intent.qty),
        "clientOid": intent.client_oid,
    }
    if hold_mode == HOLD_HEDGE:
        payload["posSide"] = position_side(intent.side, adds_exposure=intent.purpose.adds_exposure)
    elif intent.reduce_only:
        payload["reduceOnly"] = "yes"
    if intent.stop_loss_price is not None:
        payload["stopLoss"] = fmt_decimal(intent.stop_loss_price)
        payload["slTriggerBy"] = STOP_TRIGGER
        payload["slOrderType"] = "market"
    return payload


def _paper(*parts: str) -> list[str]:
    return [*parts, PAPER_FLAG]


def build_place_args(intent: OrderIntent, *, hold_mode: str, dry_run: bool) -> list[str]:
    """``order --action place ...``; ``--paper-trading`` always, ``--dry-run`` for a preview."""
    args = ["order", "--action", "place"]
    for key, value in would_send(intent, hold_mode=hold_mode).items():
        args += [f"--{key}", value]
    args.append(PAPER_FLAG)
    if dry_run:
        args.append(DRY_RUN_FLAG)
    return args


def detail_args(client_oid: str) -> list[str]:
    if not _CLIENT_OID.fullmatch(client_oid):
        raise ValueError(f"not one of this project's clientOids: {client_oid!r}")
    return _paper("order", "--action", "detail", "--clientOid", client_oid, "--view", "full")


def detail_by_order_id_args(order_id: str) -> list[str]:
    return _paper(
        "order", "--action", "detail", "--orderId", _require_venue_id(order_id), "--view", "full"
    )


def _window_args(action: str, start_ms: str, end_ms: str, cursor: str | None) -> list[str]:
    parts = [
        "order",
        "--action",
        action,
        "--category",
        CATEGORY,
        "--startTime",
        start_ms,
        "--endTime",
        end_ms,
        "--limit",
        str(PAGE_LIMIT),
    ]
    if cursor is not None:
        parts += ["--cursor", _require_venue_id(cursor)]
    parts += ["--view", "full"]
    return _paper(*parts)


def fills_args(start_ms: str, end_ms: str, cursor: str | None = None) -> list[str]:
    return _window_args("fills", start_ms, end_ms, cursor)


def history_args(start_ms: str, end_ms: str, cursor: str | None = None) -> list[str]:
    return _window_args("history", start_ms, end_ms, cursor)


def positions_args() -> list[str]:
    return _paper("position", "--action", "info", "--category", CATEGORY, "--view", "full")


def stop_orders_args() -> list[str]:
    return _paper(
        "strategy_order",
        "--action",
        "open",
        "--category",
        CATEGORY,
        "--type",
        "tpsl",
        "--view",
        "full",
    )


def account_args() -> list[str]:
    """The account's assets as one operation (see :meth:`BgcTransport.account`)."""
    return _paper("raw", "--operationId", "getAccountAssets")


def place_stop_args(
    *,
    symbol: str,
    pos_side: str,
    qty: Decimal,
    stop_price: Decimal,
    client_oid: str,
    dry_run: bool = False,
) -> list[str]:
    """A full-position stop-loss (``type tpsl``, ``tpslMode full``), triggered on mark, at market.

    ``posSide`` and ``qty`` are sent although the documentation marks both optional for a full
    stop, because the SDK catalog marks them required for ``placeStrategyOrder``. Whether one-way
    mode accepts ``posSide`` here is NOT VERIFIED until the plumbing test (DESIGN.md §20).
    """
    if pos_side not in ("long", "short"):
        raise ValueError(f"pos_side must be long or short, got {pos_side!r}")
    if not _CLIENT_OID.fullmatch(client_oid):
        raise ValueError(f"not one of this project's clientOids: {client_oid!r}")
    args = [
        "strategy_order",
        "--action",
        "place",
        "--category",
        CATEGORY,
        "--symbol",
        _require_symbol(symbol),
        "--type",
        "tpsl",
        "--tpslMode",
        "full",
        "--posSide",
        pos_side,
        "--qty",
        fmt_decimal(qty),
        "--stopLoss",
        fmt_decimal(stop_price),
        "--slTriggerBy",
        STOP_TRIGGER,
        "--slOrderType",
        "market",
        "--clientOid",
        client_oid,
        PAPER_FLAG,
    ]
    if dry_run:
        args.append(DRY_RUN_FLAG)
    return args


def cancel_stop_args(venue_id: str) -> list[str]:
    return _paper("strategy_order", "--action", "cancel", "--orderId", _require_venue_id(venue_id))


# --- parsing venue rows ---------------------------------------------------------------------


def _dig(payload: Any, *path: str) -> Any:
    node = payload
    for key in path:
        if not isinstance(node, Mapping):
            return None
        node = node.get(key)
    return node


def _rows(data: Any) -> list[Mapping[str, Any]]:
    """Rows of a list answer: ``data`` itself when a list, else ``data["list"]``."""
    rows = data if isinstance(data, list) else _dig(data, "list")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def _text(row: Mapping[str, Any], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_text(row: Mapping[str, Any], key: str) -> str:
    value = _text(row, key)
    if value is None:
        raise VenueParseError(f"venue row has no {key}")
    return value


def _decimal(value: Any) -> Decimal | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        parsed = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None
    return parsed if parsed.is_finite() else None


def _required_decimal(row: Mapping[str, Any], key: str) -> Decimal:
    value = _decimal(row.get(key))
    if value is None:
        raise VenueParseError(f"venue row has no numeric {key} ({row.get(key)!r})")
    return value


def ms_to_utc(value: Any) -> datetime:
    """A Unix-millisecond timestamp (string or number) as UTC."""
    try:
        ms = int(str(value).strip())
    except (TypeError, ValueError):
        raise VenueParseError(f"not a millisecond timestamp: {value!r}") from None
    return datetime.fromtimestamp(ms / 1000, tz=UTC)


def utc_to_ms(at: datetime) -> str:
    if at.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return str(int(at.timestamp() * 1000))


def _side(row: Mapping[str, Any]) -> Side:
    try:
        return Side(_required_text(row, "side").lower())
    except ValueError as exc:
        raise VenueParseError(str(exc)) from None


def fee_lines(row: Mapping[str, Any]) -> tuple[FeeLine, ...]:
    detail = row.get("feeDetail")
    if not isinstance(detail, list):
        return ()
    lines: list[FeeLine] = []
    for item in detail:
        if not isinstance(item, Mapping):
            continue
        raw = _decimal(item.get("fee"))
        if raw is None:
            continue
        lines.append(FeeLine(coin=str(item.get("feeCoin") or ""), raw=raw))
    return tuple(lines)


def parse_venue_order(row: Mapping[str, Any], *, blob: BlobRef | None) -> VenueOrder:
    """One row of ``order-info`` or ``history-orders`` (legacy-docs/uta/trade/Get-Order-Details)."""
    try:
        status = VenueOrderStatus(_required_text(row, "orderStatus").lower())
    except ValueError as exc:
        raise VenueParseError(f"unknown orderStatus: {exc}") from None
    avg = _decimal(row.get("avgPrice"))
    reduce_only = {"yes": True, "no": False}.get((_text(row, "reduceOnly") or "").lower())
    return VenueOrder(
        venue_order_id=_required_text(row, "orderId"),
        client_oid=_text(row, "clientOid"),
        symbol=_required_text(row, "symbol"),
        side=_side(row),
        order_type=_text(row, "orderType") or "",
        qty=_required_decimal(row, "qty"),
        cum_exec_qty=_decimal(row.get("cumExecQty")) or Decimal(0),
        cum_exec_value=_decimal(row.get("cumExecValue")) or Decimal(0),
        avg_price=avg if avg is not None and avg > 0 else None,
        status=status,
        reduce_only=reduce_only,
        delegate_type=_text(row, "delegateType"),
        cancel_reason=_text(row, "cancelReason"),
        fees=fee_lines(row),
        created_at=ms_to_utc(row.get("createdTime")),
        updated_at=ms_to_utc(row.get("updatedTime") or row.get("createdTime")),
        blob=blob,
    )


def parse_fill(
    row: Mapping[str, Any], *, blob: BlobRef | None, venue: FillVenue = FillVenue.BITGET_DEMO
) -> Fill:
    """One row of ``/api/v3/trade/fills`` (legacy-docs/uta/trade/Get-Order-Fills)."""
    lines = fee_lines(row)
    scope = (_text(row, "tradeScope") or "").lower()
    trade_side = (_text(row, "tradeSide") or "").lower()
    try:
        return Fill(
            exec_id=_required_text(row, "execId"),
            venue_order_id=_required_text(row, "orderId"),
            client_oid=_text(row, "clientOid"),
            symbol=_required_text(row, "symbol"),
            side=_side(row),
            exec_price=_required_decimal(row, "execPrice"),
            exec_qty=_required_decimal(row, "execQty"),
            exec_value=_required_decimal(row, "execValue"),
            fee_paid=sum((abs(line.raw) for line in lines), Decimal(0)),
            fee_coin=lines[0].coin if lines and lines[0].coin else "USDT",
            trade_scope="taker" if scope == "taker" else "maker" if scope == "maker" else None,
            trade_side="open"
            if trade_side == "open"
            else "close"
            if trade_side == "close"
            else None,
            exec_pnl=_decimal(row.get("execPnl")),
            executed_at=ms_to_utc(row.get("createdTime")),
            venue=venue,
            blob=blob,
        )
    except VenueParseError:
        raise
    except ValueError as exc:
        raise VenueParseError(f"fill row rejected by the contract: {exc}") from None


def parse_position(row: Mapping[str, Any], *, blob: BlobRef | None) -> VenuePosition | None:
    """One row of ``current-position`` (legacy-docs/uta/trade/Get-Position); ``None`` when flat.

    ``total`` is the unsigned size and ``posSide`` its direction. A non-zero row without a
    direction cannot be booked and is refused rather than guessed.
    """
    total = _decimal(row.get("total"))
    if total is None:
        total = _decimal(row.get("available"))
    if total is None:
        raise VenueParseError("position row has no total")
    if total == 0:
        return None
    direction = (_text(row, "posSide") or "").lower()
    if direction not in ("long", "short"):
        raise VenueParseError(f"position row has no usable posSide ({row.get('posSide')!r})")
    avg = _decimal(row.get("avgPrice"))
    return VenuePosition(
        symbol=_required_text(row, "symbol"),
        qty=abs(total) if direction == "long" else -abs(total),
        avg_price=avg if avg is not None and avg > 0 else None,
        blob=blob,
    )


_CLOSED_STRATEGY_STATUSES: Final = frozenset({"success", "failed", "cancelled"})


def parse_stop_order(row: Mapping[str, Any], *, blob: BlobRef | None) -> VenueStopOrder | None:
    """One row of ``unfilled-strategy-orders`` (legacy-docs/uta/strategy/...). ``None`` when the
    row is not an open USDT-futures strategy order."""
    category = _text(row, "category")
    if category is not None and category != CATEGORY:
        return None
    if (_text(row, "status") or "").lower() in _CLOSED_STRATEGY_STATUSES:
        return None
    stop = _decimal(row.get("stopLoss"))
    return VenueStopOrder(
        symbol=_required_text(row, "symbol"),
        venue_id=_text(row, "orderId"),
        stop_price=stop if stop is not None and stop > 0 else None,
        blob=blob,
    )


def is_duplicate_client_oid(failure: BgcFailure) -> bool:
    return bool(_DUPLICATE_OID.search(failure.message)) or failure.code == "25212"


def is_order_not_found(failure: BgcFailure) -> bool:
    return failure.code == "25204" or (
        failure.type == "BitgetApiError" and bool(_NOT_FOUND.search(failure.message))
    )


# --- the transport --------------------------------------------------------------------------


class BgcTransport:
    """:class:`~sentiment_agent.types.VenueTransport` over ``bgc --paper-trading`` (UTA Demo).

    Built one of two ways:

    * ``dry_run_only=True`` (DRYRUN mode): no credential may be given. Only :meth:`preview` works;
      every send and private read raises :class:`DryRunOnlyError`.
    * ``dry_run_only=False`` (PAPER mode): Demo credentials and a *passed*
      :class:`~sentiment_agent.types.EnvironmentProof` for the same key are required, or
      construction raises :class:`EnvironmentRefused`.

    Any answer carrying 40099 raises :class:`EnvironmentRefused`, whatever the call.
    """

    def __init__(
        self,
        *,
        runner: BgcRunner,
        clock: Clock,
        blobs: BlobStore,
        credentials: DemoCredentials | None,
        proof: EnvironmentProof | None,
        dry_run_only: bool,
    ) -> None:
        self._runner = runner
        self._clock = clock
        self._blobs = blobs
        self._dry_run_only = dry_run_only
        if dry_run_only:
            if credentials is not None:
                raise TransportRefusedError("a dry-run transport never holds a credential")
            hold = proof.hold_mode if proof is not None else None
            self._hold_mode = hold if hold in HOLD_MODES and hold is not None else HOLD_ONE_WAY
            self._credentials: DemoCredentials | None = None
            return
        if credentials is None:
            raise EnvironmentRefused("paper trading needs the Demo credentials (.secrets/demo.env)")
        if proof is None or not proof.passed or proof.mode is not RunMode.PAPER:
            raise EnvironmentRefused(
                "no passed environment proof: `t2sa prove-env` must pass before any paper order"
            )
        fingerprint = proof.detail.get("key_fingerprint")
        if fingerprint != credentials.fingerprint:
            raise EnvironmentRefused("the environment proof was made for a different key")
        if proof.hold_mode not in HOLD_MODES or proof.hold_mode is None:
            raise EnvironmentRefused(f"the proof recorded no usable hold mode ({proof.hold_mode})")
        self._hold_mode = proof.hold_mode
        self._credentials = credentials

    # --- properties -------------------------------------------------------------------------

    @property
    def venue(self) -> FillVenue:
        return FillVenue.BITGET_DEMO

    @property
    def hold_mode(self) -> str:
        return self._hold_mode

    @property
    def dry_run_only(self) -> bool:
        return self._dry_run_only

    # --- plumbing ---------------------------------------------------------------------------

    def _env(self, *, credentialed: bool, write: bool) -> dict[str, str]:
        if credentialed:
            if self._credentials is None:
                raise DryRunOnlyError("this transport holds no credential: dry-run previews only")
            env = self._credentials.child_env()
        else:
            env = base_child_env()
        env["BITGET_TIMEOUT_MS"] = REQUEST_TIMEOUT_MS
        env["BITGET_MAX_RETRIES"] = WRITE_RETRIES if write else READ_RETRIES
        return env

    def _call(
        self, args: Sequence[str], *, credentialed: bool, write: bool, timeout_s: float
    ) -> tuple[BgcResult, BlobRef]:
        if args.count(PAPER_FLAG) != 1 or READ_ONLY_FLAG in args:
            raise TransportRefusedError(
                "every transport call carries --paper-trading and never --read-only"
            )
        env = self._env(credentialed=credentialed, write=write)
        at = self._clock.now().isoformat()
        result = self._runner(args, env=env, timeout_s=timeout_s)
        blob = result_blob(self._blobs, args=args, result=result, note="bgc", at=at)
        if result_mentions_environment_mismatch(result):
            raise EnvironmentRefused(
                f"`bgc {' '.join(args[:3])}` answered 40099 (exchange environment is incorrect): "
                "the key is not a Demo key",
                evidence=blob,
            )
        return result, blob

    def _read(self, args: Sequence[str], what: str) -> tuple[Any, BlobRef]:
        if self._dry_run_only:
            raise DryRunOnlyError(f"{what}: a dry-run transport makes no private read")
        try:
            result, blob = self._call(
                args, credentialed=True, write=False, timeout_s=READ_TIMEOUT_S
            )
        except BgcTimeoutError as exc:
            raise VenueReadError(f"{what} timed out: {exc}", outcome_unknown=True) from None
        failure = failure_of(result)
        if failure is not None:
            raise VenueReadError(
                f"{what} failed: {failure.type} {failure.code}: {failure.message}",
                failure=failure,
                blob=blob,
            )
        return _dig(result.stdout, "data"), blob

    # --- orders -----------------------------------------------------------------------------

    def preview(self, intent: OrderIntent) -> DryRunPreview:
        """``bgc ... --paper-trading --dry-run`` with no credential; checked against the intent."""
        args = build_place_args(intent, hold_mode=self._hold_mode, dry_run=True)
        result, blob = self._call(args, credentialed=False, write=False, timeout_s=READ_TIMEOUT_S)
        failure = failure_of(result)
        if failure is not None:
            raise TransportRefusedError(
                f"dry-run preview failed: {failure.type}: {failure.message}"
            )
        data = _dig(result.stdout, "data")
        if not isinstance(data, Mapping) or data.get("dryRun") is not True:
            raise TransportRefusedError("bgc did not answer with a dry-run preview")
        wanted = would_send(intent, hold_mode=self._hold_mode)
        got = data.get("wouldSend")
        if got != wanted or data.get("path") != PLACE_ORDER_PATH:
            raise PreviewMismatchError(
                f"bgc would send {got!r} to {data.get('path')!r}, the approval covers {wanted!r}"
            )
        return DryRunPreview(
            client_oid=intent.client_oid,
            operation_id=str(data.get("operationId") or ""),
            method=str(data.get("method") or ""),
            path=str(data.get("path") or ""),
            would_send=dict(got),
            argv=tuple(args),
            captured_at=self._clock.now(),
            blob=blob,
        )

    def place(self, order: ApprovedOrder) -> VenueAck | VenueRejection | VenueUnknown:
        """Send one approved order. Never retried here; a timeout is UNKNOWN, not rejected."""
        if self._dry_run_only:
            raise DryRunOnlyError("a dry-run transport never sends an order")
        if not order.verify():
            raise TransportRefusedError("the approval does not match its intent; nothing was sent")
        intent = order.intent
        args = build_place_args(intent, hold_mode=self._hold_mode, dry_run=False)
        try:
            result, blob = self._call(
                args, credentialed=True, write=True, timeout_s=WRITE_TIMEOUT_S
            )
        except BgcTimeoutError as exc:
            return VenueUnknown(client_oid=intent.client_oid, at=self._clock.now(), reason=str(exc))
        now = self._clock.now()
        failure = failure_of(result)
        if failure is None:
            data = _dig(result.stdout, "data")
            order_id = _dig(data, "orderId")
            if isinstance(order_id, (str, int)) and str(order_id).strip():
                return VenueAck(
                    client_oid=intent.client_oid,
                    venue_order_id=str(order_id).strip(),
                    acked_at=now,
                    blob=blob,
                )
            if isinstance(data, Mapping) and data.get("confirmationRequired"):
                return VenueRejection(
                    client_oid=intent.client_oid,
                    code=None,
                    message="bgc asked for confirmation; placeOrder is not graded high-risk",
                    category="local",
                    retryable=False,
                    at=now,
                    blob=blob,
                )
            return VenueUnknown(
                client_oid=intent.client_oid, at=now, reason="acknowledged without an orderId"
            )
        if failure.type == "NetworkError" or failure.category == "network":
            return VenueUnknown(
                client_oid=intent.client_oid,
                at=now,
                reason=f"{failure.type} {failure.code}: {failure.message}",
            )
        if is_duplicate_client_oid(failure):
            return VenueUnknown(
                client_oid=intent.client_oid,
                at=now,
                reason=f"the venue already knows this clientOid ({failure.message})",
            )
        return VenueRejection(
            client_oid=intent.client_oid,
            code=failure.code,
            message=failure.message,
            category="local" if failure.local else (failure.category or failure.type),
            retryable=failure.retryable,
            at=now,
            blob=blob,
        )

    def order(self, *, client_oid: str) -> VenueOrder | None:
        """The venue's record of one of our orders; ``None`` only when the venue has none."""
        if self._dry_run_only:
            raise DryRunOnlyError("a dry-run transport makes no private read")
        try:
            result, blob = self._call(
                detail_args(client_oid), credentialed=True, write=False, timeout_s=READ_TIMEOUT_S
            )
        except BgcTimeoutError as exc:
            raise VenueReadError(
                f"order detail {client_oid} timed out: {exc}", outcome_unknown=True
            ) from None
        failure = failure_of(result)
        if failure is not None:
            if is_order_not_found(failure):
                return None
            raise VenueReadError(
                f"order detail {client_oid} failed: {failure.type} {failure.code}: "
                f"{failure.message}",
                failure=failure,
                blob=blob,
            )
        data = _dig(result.stdout, "data")
        if not isinstance(data, Mapping) or not _text(data, "orderId"):
            return None
        try:
            return parse_venue_order(data, blob=blob)
        except VenueParseError as exc:
            raise VenueReadError(f"order detail {client_oid}: {exc}", blob=blob) from None

    def _paged(self, action: str, since: datetime, until: datetime) -> list[tuple[Any, BlobRef]]:
        rows: list[tuple[Any, BlobRef]] = []
        start = since
        while start < until:
            end = min(until, start + QUERY_WINDOW)
            cursor: str | None = None
            for _page in range(MAX_PAGES):
                builder = fills_args if action == "fills" else history_args
                data, blob = self._read(
                    builder(utc_to_ms(start), utc_to_ms(end), cursor), f"{action} page"
                )
                page = _rows(data)
                rows.extend((row, blob) for row in page)
                next_cursor = _dig(data, "cursor")
                if (
                    len(page) < PAGE_LIMIT
                    or next_cursor in (None, "")
                    or str(next_cursor) == cursor
                ):
                    break
                cursor = str(next_cursor)
            else:
                raise VenueReadError(f"{action} did not end within {MAX_PAGES} pages")
            start = end
        return rows

    def fills(self, *, since: datetime, until: datetime) -> list[Fill]:
        """Every fill in ``[since, until]``, de-duplicated by ``execId``, oldest first."""
        found: dict[str, Fill] = {}
        for row, blob in self._paged("fills", since, until):
            try:
                fill = parse_fill(row, blob=blob)
            except VenueParseError as exc:
                raise VenueReadError(f"unreadable fill row: {exc}", blob=blob) from None
            found.setdefault(fill.exec_id, fill)
        return sorted(found.values(), key=lambda f: (f.executed_at, f.exec_id))

    def history(self, *, since: datetime, until: datetime) -> list[VenueOrder]:
        """Order history in ``[since, until]`` (the daily full sweep, DESIGN.md §11.5)."""
        found: dict[str, VenueOrder] = {}
        for row, blob in self._paged("history", since, until):
            try:
                order = parse_venue_order(row, blob=blob)
            except VenueParseError as exc:
                raise VenueReadError(f"unreadable order-history row: {exc}", blob=blob) from None
            found.setdefault(order.venue_order_id, order)
        return sorted(found.values(), key=lambda o: (o.created_at, o.venue_order_id))

    def positions(self) -> list[VenuePosition]:
        data, blob = self._read(positions_args(), "positions")
        positions: list[VenuePosition] = []
        for row in _rows(data):
            try:
                position = parse_position(row, blob=blob)
            except VenueParseError as exc:
                raise VenueReadError(f"unreadable position row: {exc}", blob=blob) from None
            if position is not None:
                positions.append(position)
        return positions

    def stop_orders(self) -> list[VenueStopOrder]:
        data, blob = self._read(stop_orders_args(), "stop orders")
        stops: list[VenueStopOrder] = []
        for row in _rows(data):
            try:
                stop = parse_stop_order(row, blob=blob)
            except VenueParseError as exc:
                raise VenueReadError(f"unreadable stop-order row: {exc}", blob=blob) from None
            if stop is not None:
                stops.append(stop)
        return stops

    def account(self) -> AccountSnapshot:
        """Account equity and available USDT, from ``GET /api/v3/account/assets``.

        Read as one operation (``raw getAccountAssets``) rather than through ``account_overview``,
        which DESIGN.md §11.1 names: the composite reports a failed section as a message with exit
        code 0 (``account-overview.ts:92-115``), so an unreadable account would look like an
        account with no equity instead of raising :class:`VenueReadError`, and it would make four
        requests where one is needed. The environment proof still reads the whole account through
        ``account_overview`` once.
        """
        data, blob = self._read(account_args(), "account")
        if not isinstance(data, Mapping):
            raise VenueReadError("account assets came back empty", blob=blob)
        return account_from_assets(data, at=self._clock.now(), blob=blob)

    # --- stops ------------------------------------------------------------------------------

    def place_stop(
        self, *, symbol: str, pos_side: str, qty: Decimal, stop_price: Decimal, client_oid: str
    ) -> StopSync:
        """Place a full-position stop. Raises :class:`VenueWriteError` if the venue refuses it."""
        if self._dry_run_only:
            raise DryRunOnlyError("a dry-run transport never places a stop")
        args = place_stop_args(
            symbol=symbol, pos_side=pos_side, qty=qty, stop_price=stop_price, client_oid=client_oid
        )
        try:
            result, blob = self._call(
                args, credentialed=True, write=True, timeout_s=WRITE_TIMEOUT_S
            )
        except BgcTimeoutError as exc:
            raise VenueWriteError(
                f"stop placement for {symbol} timed out: {exc}", outcome_unknown=True
            ) from None
        failure = failure_of(result)
        if failure is not None:
            raise VenueWriteError(
                f"stop placement for {symbol} refused: {failure.type} {failure.code}: "
                f"{failure.message}",
                failure=failure,
                blob=blob,
                outcome_unknown=failure.transient,
            )
        venue_id = _dig(result.stdout, "data", "orderId")
        return StopSync(
            symbol=symbol,
            action="placed",
            stop_price=stop_price,
            venue_id=str(venue_id) if venue_id not in (None, "") else None,
            at=self._clock.now(),
            blob=blob,
        )

    def cancel_stop(self, *, symbol: str, venue_id: str) -> StopSync:
        """Cancel one strategy order. Raises :class:`VenueWriteError` if the venue refuses it."""
        if self._dry_run_only:
            raise DryRunOnlyError("a dry-run transport never cancels a stop")
        args = cancel_stop_args(venue_id)
        try:
            result, blob = self._call(
                args, credentialed=True, write=True, timeout_s=WRITE_TIMEOUT_S
            )
        except BgcTimeoutError as exc:
            raise VenueWriteError(
                f"stop cancellation {venue_id} timed out: {exc}", outcome_unknown=True
            ) from None
        failure = failure_of(result)
        if failure is not None:
            raise VenueWriteError(
                f"stop cancellation {venue_id} refused: {failure.type} {failure.code}: "
                f"{failure.message}",
                failure=failure,
                blob=blob,
                outcome_unknown=failure.transient,
            )
        return StopSync(
            symbol=symbol,
            action="cancelled",
            stop_price=None,
            venue_id=venue_id,
            at=self._clock.now(),
            blob=blob,
        )


__all__ = [
    "BGC_PACKAGE",
    "HOLD_HEDGE",
    "HOLD_ONE_WAY",
    "MAX_PAGES",
    "PAGE_LIMIT",
    "PLACE_ORDER_PATH",
    "STOP_TRIGGER",
    "BgcResult",
    "BgcRunner",
    "BgcTimeoutError",
    "BgcTransport",
    "BgcUnavailableError",
    "DryRunOnlyError",
    "PreviewMismatchError",
    "SubprocessBgcRunner",
    "TransportRefusedError",
    "VenueCallError",
    "VenueParseError",
    "VenueReadError",
    "VenueWriteError",
    "account_args",
    "build_place_args",
    "cancel_stop_args",
    "detail_args",
    "detail_by_order_id_args",
    "fee_lines",
    "fills_args",
    "fmt_decimal",
    "history_args",
    "is_duplicate_client_oid",
    "is_order_not_found",
    "ms_to_utc",
    "parse_fill",
    "parse_output",
    "parse_position",
    "parse_stop_order",
    "parse_venue_order",
    "place_stop_args",
    "position_side",
    "positions_args",
    "stop_orders_args",
    "utc_to_ms",
    "would_send",
]
