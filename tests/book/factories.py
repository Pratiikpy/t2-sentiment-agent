"""Builders shared by the book tests: fills, intents, marks and an in-memory hash-chained ledger.

The ledger here is a test double for ``types.LedgerReader`` (the real one is ``ledger/chain.py``):
it writes events exactly as the contract specifies (typed payload dumped to JSON, full-length
hash chain) so the projection reads the same shapes it will read from disk.
"""

import itertools
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from sentiment_agent.hashing import ZERO_HASH, content_hash
from sentiment_agent.policy import POLICY_V1
from sentiment_agent.types import (
    EVENT_PAYLOADS,
    AccountSnapshot,
    EnvironmentProof,
    EventKind,
    Fill,
    FillVenue,
    Genesis,
    LedgerEvent,
    MarkPoint,
    Model,
    OrderIntent,
    OrderPurpose,
    RunMode,
    Side,
    client_oid_of,
)

T0 = datetime(2026, 9, 23, 13, 0, tzinfo=UTC)
"""Wednesday, 13:00 UTC."""
H = timedelta(hours=1)
FEE_RATE = Decimal("0.0006")
START = Decimal("10000")

_exec_ids = itertools.count(1)


def d(value: str | int) -> Decimal:
    return Decimal(str(value))


def make_fill(
    side: Side | str,
    qty: str | Decimal,
    price: str | Decimal,
    *,
    symbol: str = "NVDAUSDT",
    at: datetime = T0,
    fee: str | Decimal | None = None,
    exec_id: str | None = None,
    client_oid: str | None = None,
    venue_order_id: str | None = None,
    trade_side: str | None = None,
    venue: FillVenue = FillVenue.SIMULATED,
    fee_coin: str = "USDT",
) -> Fill:
    q = Decimal(str(qty))
    p = Decimal(str(price))
    n = next(_exec_ids)
    return Fill.model_validate(
        {
            "exec_id": exec_id or f"exec-{n:06d}",
            "venue_order_id": venue_order_id or f"order-{n:06d}",
            "client_oid": client_oid,
            "symbol": symbol,
            "side": Side(side),
            "exec_price": p,
            "exec_qty": q,
            "exec_value": q * p,
            "fee_paid": q * p * FEE_RATE if fee is None else Decimal(str(fee)),
            "fee_coin": fee_coin,
            "trade_scope": "taker",
            "trade_side": trade_side,
            "exec_pnl": None,
            "executed_at": at,
            "venue": venue,
        }
    )


def make_intent(
    *,
    symbol: str = "NVDAUSDT",
    side: Side = Side.BUY,
    qty: str = "2",
    price: str = "100",
    purpose: OrderPurpose = OrderPurpose.OPEN,
    ruling_id: str = "ruling-1",
    decision_id: str | None = "d1",
    seed: str = "",
    stop: str | None = None,
) -> OrderIntent:
    adds = purpose.adds_exposure
    ref = Decimal(price)
    stop_price: Decimal | None = None
    if adds:
        stop_price = (
            Decimal(stop)
            if stop is not None
            else ref * Decimal("0.96")
            if side is Side.BUY
            else ref * Decimal("1.04")
        )
    intent_id = content_hash({"r": ruling_id, "s": symbol, "side": side, "q": qty, "seed": seed})
    q = Decimal(qty)
    return OrderIntent(
        intent_id=intent_id,
        ruling_id=ruling_id,
        decision_id=decision_id,
        symbol=symbol,
        side=side,
        qty=q,
        reduce_only=not adds,
        purpose=purpose,
        reference_price=ref,
        notional=q * ref,
        expected_fee=q * ref * FEE_RATE,
        stop_loss_price=stop_price,
        client_oid=client_oid_of(intent_id),
    )


def make_mark(
    at: datetime,
    equity_book: str | Decimal,
    *,
    mirror: str | Decimal | None = None,
) -> MarkPoint:
    return MarkPoint(
        at=at,
        equity_book=Decimal(str(equity_book)),
        equity_venue=None,
        equity_live_mirror=None if mirror is None else Decimal(str(mirror)),
        gross_weight=0.0,
        net_weight=0.0,
        positions=(),
    )


def make_genesis(mode: RunMode = RunMode.PAPER, at: datetime = T0 - H) -> Genesis:
    return Genesis(
        project="t2-sentiment-agent",
        contract_version="1.0.0",
        created_at=at,
        mode=mode,
        policy=POLICY_V1,
        policy_hash=POLICY_V1.content_hash(),
        prompt_hashes={},
        universe=POLICY_V1.symbols,
        metric_definitions=POLICY_V1.metrics,
        code_commit="test",
        dependency_lock_hashes={},
        qwen_model="qwen3.8-max",
        bgc_package="@bitget-ai/bitget-agent-cli@3.0.0",
        expected_envelope=POLICY_V1.expected_envelope,
        statement="test genesis",
    )


def make_proof(
    equity: str | None = "10000", *, passed: bool = True, at: datetime = T0 - H
) -> EnvironmentProof:
    account = (
        None
        if equity is None
        else AccountSnapshot(at=at, equity_usdt=Decimal(equity), available_usdt=None, blob=None)
    )
    return EnvironmentProof(
        checked_at=at,
        mode=RunMode.PAPER,
        credentials_file=".secrets/demo.env",
        key_declared_demo=True,
        bgc_package="@bitget-ai/bitget-agent-cli@3.0.0",
        paptrading_header_confirmed=True,
        demo_read_ok=True,
        demo_read_code=None,
        live_read_rejected=passed,
        live_read_code=None,
        hold_mode="one_way_mode",
        account=account,
        passed=passed and account is not None,
        reasons=() if passed else ("live read was not rejected",),
    )


class MemoryLedger:
    """An in-memory ``LedgerReader`` whose events are built exactly as the chain builds them."""

    def __init__(self, mode: RunMode = RunMode.PAPER, start: datetime = T0 - 2 * H) -> None:
        self.mode = mode
        self._events: list[LedgerEvent] = []
        self._ts = start

    def append(
        self,
        kind: EventKind,
        payload: Model,
        *,
        mode: RunMode | None = None,
        ts: datetime | None = None,
    ) -> LedgerEvent:
        if not isinstance(payload, EVENT_PAYLOADS[kind]):
            raise TypeError(f"{kind} carries {EVENT_PAYLOADS[kind].__name__}")
        self._ts = ts if ts is not None else self._ts + timedelta(seconds=1)
        seq = len(self._events)
        prev = self._events[-1].hash if self._events else ZERO_HASH
        body: Mapping[str, object] = {
            "seq": seq,
            "ts": self._ts,
            "kind": kind,
            "mode": mode or self.mode,
            "payload": payload.model_dump(mode="json"),
            "blobs": [],
            "prev_hash": prev,
        }
        event = LedgerEvent(
            seq=seq,
            ts=self._ts,
            kind=kind,
            mode=mode or self.mode,
            payload=payload.model_dump(mode="json"),
            blobs=(),
            prev_hash=prev,
            hash=content_hash(body),
        )
        self._events.append(event)
        return event

    def events(self, kinds: frozenset[EventKind] | None = None) -> Iterator[LedgerEvent]:
        for event in self._events:
            if kinds is None or event.kind in kinds:
                yield event

    def head(self) -> LedgerEvent | None:
        return self._events[-1] if self._events else None

    def prefix(self, n: int) -> "MemoryLedger":
        """A ledger holding only the first ``n`` events."""
        out = MemoryLedger(self.mode)
        out._events = self._events[:n]
        return out

    def __len__(self) -> int:
        return len(self._events)
