"""The book's capital is the USDT it can margin with, not every coin the account holds.

The first real Demo read (2026-09-24) showed seventeen gifted demo coins worth 3.78M USDT beside
50,000 USDT; valuing them all made the starting equity 3.78M and every BTC move an equity gap.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from sentiment_agent.execution.environment import account_from_assets

AT = datetime(2026, 9, 24, 16, 0, tzinfo=UTC)


def test_other_coins_do_not_count_toward_the_books_equity() -> None:
    data = {
        "usdtEquity": "3782464.95",
        "assets": [
            {"coin": "BTC", "equity": "5", "available": "5"},
            {"coin": "USDT", "equity": "50000", "available": "49981.99"},
            {"coin": "UNI", "equity": "50000", "available": "50000"},
        ],
    }
    account = account_from_assets(data, at=AT, blob=None)
    assert account.equity_usdt == Decimal("50000")
    assert account.available_usdt == Decimal("49981.99")


def test_no_usdt_row_means_no_equity_rather_than_a_guess() -> None:
    account = account_from_assets({"usdtEquity": "100", "assets": []}, at=AT, blob=None)
    assert account.equity_usdt is None
