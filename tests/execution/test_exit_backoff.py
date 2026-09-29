"""Pacing a reduce-only order the venue refuses as venue-symbol-unavailable (run3-d1).

The sequence these tests replay is run 2's own record (``public/orders.json``, read 2026-09-29):
72 refusals of the same shape for METAUSDT between 2026-09-28T09:16:43Z and 15:11:33Z, 2 more for
MSTRUSDT in the same window, then both filled once the venue accepted the identical request again.
"""

from datetime import UTC, datetime, timedelta

import pytest

from helpers import T0, client_oid_for, make_intent
from sentiment_agent.execution.environment import VENUE_SYMBOL_UNAVAILABLE
from sentiment_agent.execution.exit_backoff import (
    BACKOFF_BASE,
    BACKOFF_CAP,
    blocked_exit_episodes,
    next_retry_at,
)
from sentiment_agent.hashing import content_hash
from sentiment_agent.types import OrderPurpose, OrderSubmitted, Side, VenueAck, VenueRejection

APPROVAL_HASH = content_hash("approval-for-test")


def _submitted(
    *,
    symbol: str,
    purpose: OrderPurpose,
    side: Side,
    ruling_id: str,
    at: datetime,
) -> OrderSubmitted:
    intent = make_intent(symbol=symbol, side=side, purpose=purpose, ruling_id=ruling_id)
    return OrderSubmitted(
        client_oid=intent.client_oid,
        intent=intent,
        approval_hash=APPROVAL_HASH,
        submitted_at=at,
        argv=("order", "--action", "place", "--symbol", symbol),
    )


def _rejected(
    submitted: OrderSubmitted, *, at: datetime, category: str | None, message: str = "refused"
) -> VenueRejection:
    return VenueRejection(
        client_oid=submitted.client_oid,
        code="400",
        message=message,
        category=category,
        retryable=False,
        at=at,
        blob=None,
    )


def _acked(submitted: OrderSubmitted, *, at: datetime) -> VenueAck:
    return VenueAck(client_oid=submitted.client_oid, venue_order_id="12345", acked_at=at, blob=None)


# --- the schedule --------------------------------------------------------------------------------


def test_next_retry_at_grows_geometrically_and_caps() -> None:
    last = T0
    waits = [next_retry_at(n, last) - last for n in range(1, 8)]
    assert waits == [
        BACKOFF_BASE,
        BACKOFF_BASE * 2,
        BACKOFF_BASE * 4,
        BACKOFF_BASE * 8,
        BACKOFF_CAP,  # base * 16 would be 80 min, capped at 60
        BACKOFF_CAP,
        BACKOFF_CAP,
    ]


def test_next_retry_at_refuses_a_non_positive_attempt() -> None:
    with pytest.raises(ValueError, match="attempt_count"):
        next_retry_at(0, T0)


def test_the_run2_schedule_bounds_attempts_over_the_actual_episode_length() -> None:
    """Run 2 sent 72 identical resends over 5h55m (09:16:43Z to 15:11:33Z). Replaying that same
    span through the schedule this project now uses must land far below 72 real attempts."""
    started = datetime(2026, 9, 28, 9, 16, 43, tzinfo=UTC)
    ended = datetime(2026, 9, 28, 15, 11, 33, tzinfo=UTC)
    attempt_count = 1
    at = started
    while at < ended:
        at = next_retry_at(attempt_count, at)
        attempt_count += 1
    assert attempt_count < 20  # 72 in the unpaced record; comfortably bounded here


# --- rebuilding episodes from the ledger's own projections ---------------------------------------


def test_blocked_exit_episodes_reproduces_the_run2_sequence() -> None:
    t0 = datetime(2026, 9, 28, 9, 16, 43, tzinfo=UTC)
    t1 = t0 + timedelta(hours=1)
    t2 = t0 + timedelta(hours=3)
    cleared_at = datetime(2026, 9, 28, 16, 6, 52, tzinfo=UTC)

    subs = [
        _submitted(
            symbol="METAUSDT",
            purpose=OrderPurpose.PROTECTIVE_EXIT,
            side=Side.SELL,
            ruling_id=f"r{i}",
            at=at,
        )
        for i, at in enumerate((t0, t1, t2, cleared_at))
    ]
    message = "HTTP 400 from Bitget: Parameter METAUSDT_UMCBL does not exist"
    rejections = [
        _rejected(subs[0], at=t0, category=VENUE_SYMBOL_UNAVAILABLE, message=message),
        _rejected(subs[1], at=t1, category=VENUE_SYMBOL_UNAVAILABLE, message=message),
        _rejected(subs[2], at=t2, category=VENUE_SYMBOL_UNAVAILABLE, message=message),
    ]
    acks = [_acked(subs[3], at=cleared_at)]

    episodes = blocked_exit_episodes(subs, rejections, acks)
    assert episodes == {}, "the ack after the rejections must clear the episode"

    # Without the ack, the episode is still open and carries the full history.
    open_episodes = blocked_exit_episodes(subs[:3], rejections, [])
    episode = open_episodes["METAUSDT"]
    assert episode.first_attempt_at == t0
    assert episode.last_attempt_at == t2
    assert episode.attempt_count == 3
    assert episode.next_retry_at == next_retry_at(3, t2)
    assert "METAUSDT_UMCBL does not exist" in episode.last_message


def test_episodes_are_independent_per_symbol() -> None:
    meta_sub = _submitted(
        symbol="METAUSDT",
        purpose=OrderPurpose.PROTECTIVE_EXIT,
        side=Side.SELL,
        ruling_id="r1",
        at=T0,
    )
    mstr_sub = _submitted(
        symbol="MSTRUSDT",
        purpose=OrderPurpose.CLOSE,
        side=Side.SELL,
        ruling_id="r2",
        at=T0,
    )
    rejections = [
        _rejected(meta_sub, at=T0, category=VENUE_SYMBOL_UNAVAILABLE),
        _rejected(mstr_sub, at=T0, category=VENUE_SYMBOL_UNAVAILABLE),
    ]
    episodes = blocked_exit_episodes([meta_sub, mstr_sub], rejections, [])
    assert set(episodes) == {"METAUSDT", "MSTRUSDT"}
    assert episodes["METAUSDT"].attempt_count == 1
    assert episodes["MSTRUSDT"].attempt_count == 1


def test_only_reduce_only_orders_can_open_an_episode() -> None:
    opening = _submitted(
        symbol="METAUSDT", purpose=OrderPurpose.OPEN, side=Side.BUY, ruling_id="r1", at=T0
    )
    rejection = _rejected(opening, at=T0, category=VENUE_SYMBOL_UNAVAILABLE)
    assert blocked_exit_episodes([opening], [rejection], []) == {}


def test_only_the_venue_symbol_unavailable_category_opens_or_extends_an_episode() -> None:
    submitted = _submitted(
        symbol="METAUSDT",
        purpose=OrderPurpose.PROTECTIVE_EXIT,
        side=Side.SELL,
        ruling_id="r1",
        at=T0,
    )
    for other_category in ("local", "unknown", "rate", None):
        rejection = _rejected(submitted, at=T0, category=other_category, message="something else")
        assert blocked_exit_episodes([submitted], [rejection], []) == {}


def test_a_rejection_with_no_matching_submission_is_ignored() -> None:
    orphan = VenueRejection(
        client_oid=client_oid_for("nothing-submitted"),
        code="400",
        message="x",
        category=VENUE_SYMBOL_UNAVAILABLE,
        retryable=False,
        at=T0,
        blob=None,
    )
    assert blocked_exit_episodes([], [orphan], []) == {}


# The planner's own send/skip gate (time, then the opportunistic instrument-status check) is
# inlined into kernel/planner.py rather than called from here (kernel/ never imports execution/;
# see this module's docstring) and is exercised directly against that real code path in
# tests/kernel/test_planner.py: test_a_blocked_exit_is_skipped_not_resent_every_cycle,
# test_a_blocked_exit_is_sent_once_its_schedule_elapses,
# test_a_stale_not_online_status_extends_the_wait_past_the_schedule.
