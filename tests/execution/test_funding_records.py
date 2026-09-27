"""Funding records read from the venue (run2-d6): the parse and the transport's paging."""

from decimal import Decimal
from typing import Any

import pytest

from helpers import T0
from sentiment_agent.execution.bgc import (
    FUNDING_RECORD_TYPES,
    VenueParseError,
    parse_funding,
    utc_to_ms,
)


def row(
    record_type: str, amount: str, record_id: str = "91", symbol: str = "NVDAUSDT"
) -> dict[str, Any]:
    return {
        "category": "USDT-FUTURES",
        "id": record_id,
        "symbol": symbol,
        "coin": "USDT",
        "type": record_type,
        "amount": amount,
        "balance": "10000",
        "ts": utc_to_ms(T0),
    }


@pytest.mark.parametrize(
    ("record_type", "amount", "expected"),
    [
        ("CONTRACT_MAIN_SETTLE_FEE_USER_IN", "0.75", "0.75"),
        ("CONTRACT_MAIN_SETTLE_FEE_USER_OUT", "0.75", "-0.75"),
        # the sign is the type's, whatever sign the row's amount carries
        ("CONTRACT_MAIN_SETTLE_FEE_USER_OUT", "-0.75", "-0.75"),
        ("RWA_CONTRACT_MAIN_SETTLE_FEE_USER_IN", "2", "2"),
    ],
)
def test_a_funding_record_is_signed_by_its_type(
    record_type: str, amount: str, expected: str
) -> None:
    settlement = parse_funding(row(record_type, amount), blob=None)
    assert settlement is not None
    assert settlement.amount == Decimal(expected)
    assert settlement.settled_at == T0


@pytest.mark.parametrize(
    "record_type",
    ["ORDER_DEALT_IN", "TRANSFER_IN", "RWA_CONTRACT_MAIN_SETTLE_FEE_SYSTEM_IN"],
)
def test_other_records_are_not_funding(record_type: str) -> None:
    assert record_type not in FUNDING_RECORD_TYPES
    assert parse_funding(row(record_type, "5"), blob=None) is None


def test_a_funding_record_without_an_amount_is_refused() -> None:
    bad = row("CONTRACT_MAIN_SETTLE_FEE_USER_IN", "")
    with pytest.raises(VenueParseError):
        parse_funding(bad, blob=None)
