"""Builders for contract objects that tests in every module need.

Tests may mint an :class:`ApprovedOrder` here so the execution module can be tested before, and
independently of, the kernel. Production code may not: the source-scan test in
``tests/contract/test_contract.py`` fails if the mint token is named in any project file outside
``tests/`` other than ``types.py`` and ``kernel/approval.py``.
"""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, TypeVar, cast

from sentiment_agent.hashing import content_hash
from sentiment_agent.types import (
    _MINT_TOKEN,
    CLIENT_OID_HEX,
    CLIENT_OID_PREFIX,
    Activation,
    ApprovedOrder,
    BookState,
    Model,
    OrderIntent,
    OrderPurpose,
    PriceSource,
    Quote,
    Side,
)

T0 = datetime(2026, 9, 23, 13, 0, tzinfo=UTC)

M = TypeVar("M", bound=Model)


def unvalidated_copy(model: M, **update: Any) -> M:
    """A copy with ``update`` applied and **no validation**, for testing a module's own defences
    against a record the contract would refuse (``Model.model_copy`` re-validates; this does not).
    Tests only: ``model_construct`` is banned from ``src/``."""
    return cast(M, type(model).model_construct(**{**dict(model), **update}))


def client_oid_for(seed: str) -> str:
    """The clientOid of an intent whose ``intent_id`` is ``content_hash(seed)``."""
    return CLIENT_OID_PREFIX + content_hash(seed)[:CLIENT_OID_HEX]


def make_intent(
    *,
    symbol: str = "NVDAUSDT",
    side: Side = Side.BUY,
    qty: str = "1.00",
    purpose: OrderPurpose = OrderPurpose.OPEN,
    ruling_id: str = "ruling-test",
    price: str = "222.84",
) -> OrderIntent:
    adds = purpose.adds_exposure
    ref = Decimal(price)
    q = Decimal(qty)
    stop = (ref * Decimal("0.96") if side is Side.BUY else ref * Decimal("1.04")) if adds else None
    core = f"{ruling_id}|{symbol}|{side}|{qty}|{purpose}"
    return OrderIntent(
        intent_id=content_hash(core),
        ruling_id=ruling_id,
        decision_id="decision-test",
        symbol=symbol,
        side=side,
        qty=q,
        reduce_only=not adds,
        purpose=purpose,
        reference_price=ref,
        notional=q * ref,
        expected_fee=q * ref * Decimal("0.0006"),
        stop_loss_price=stop,
        client_oid=client_oid_for(core),
    )


def mint_for_test(intent: OrderIntent, *, ruling_id: str | None = None) -> ApprovedOrder:
    """Mint as the kernel would. ``ruling_id`` overrides the intent's own, to test the refusal."""
    return ApprovedOrder(
        intent, intent.ruling_id if ruling_id is None else ruling_id, _token=_MINT_TOKEN
    )


def make_quote(
    symbol: str = "NVDAUSDT",
    *,
    source: PriceSource = PriceSource.DEMO,
    last: str = "222.81",
    mark: str = "222.84",
    index: str = "222.8431",
    bid: str = "222.82",
    ask: str = "222.88",
    at: datetime = T0,
) -> Quote:
    return Quote(
        symbol=symbol,
        source=source,
        ts=at,
        fetched_at=at,
        last=Decimal(last),
        mark=Decimal(mark),
        index=Decimal(index),
        bid=Decimal(bid),
        ask=Decimal(ask),
        funding_rate=Decimal("0"),
        open_interest=None,
        turnover_24h=None,
        price_change_24h=None,
    )


def empty_book(equity: str = "10000", at: datetime = T0) -> BookState:
    e = Decimal(equity)
    return BookState(
        as_of=at,
        mark_source=PriceSource.DEMO,
        starting_equity=e,
        equity=e,
        peak_equity=e,
        day_open_equity=e,
        positions={},
        marks={},
        fees_today=Decimal("0"),
        fees_total=Decimal("0"),
        realized_total=Decimal("0"),
        rebalances_today={},
        consecutive_losses=0,
        activation=Activation.ACTIVE,
    )
