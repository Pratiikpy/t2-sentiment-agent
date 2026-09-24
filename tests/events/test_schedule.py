"""Heartbeat schedule and weekend phases (M5, DESIGN.md §8 and §10.3 G2)."""

from datetime import UTC, date, datetime, timedelta
from itertools import pairwise
from typing import TypeVar, cast
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from sentiment_agent.clock import ManualClock
from sentiment_agent.events.schedule import (
    NEW_YORK,
    SCHEDULE_SOURCE,
    WeekendPhase,
    heartbeat_id,
    heartbeats_between,
    iso_z,
    next_freeze,
    parse_local_time,
    us_open_utc,
    validate_schedule,
    weekend_phase,
)
from sentiment_agent.policy import POLICY_V1
from sentiment_agent.types import Model, Policy, TriggerKind, WeekendRule

M = TypeVar("M", bound=Model)


def utc(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
    microsecond: int = 0,
) -> datetime:
    return datetime(year, month, day, hour, minute, second, microsecond, tzinfo=UTC)


def bypass(model: M, **changes: object) -> M:
    """A copy that skips validation, to prove the module's own guards (tests only)."""
    return cast(M, type(model).model_construct(**{**dict(model), **changes}))


def with_triggers(**changes: object) -> Policy:
    return POLICY_V1.model_copy(update={"triggers": POLICY_V1.triggers.model_copy(update=changes)})


# --- the US open ---------------------------------------------------------------------------------


def test_us_open_is_1330_utc_in_summer_time(clock: ManualClock) -> None:
    today = clock.now().date()  # Wednesday 2026-09-23
    assert us_open_utc(today, POLICY_V1) == utc(2026, 9, 23, 13, 30)


def test_dst_switch_2026_11_01_moves_the_open_from_1330_to_1430_utc() -> None:
    assert us_open_utc(date(2026, 10, 30), POLICY_V1) == utc(2026, 10, 30, 13, 30)  # Fri, EDT
    assert us_open_utc(date(2026, 10, 31), POLICY_V1) is None  # Saturday
    assert us_open_utc(date(2026, 11, 1), POLICY_V1) is None  # Sunday: the switch itself
    assert us_open_utc(date(2026, 11, 2), POLICY_V1) == utc(2026, 11, 2, 14, 30)  # Mon, EST


def test_spring_switch_2027_03_14_moves_the_open_back_to_1330_utc() -> None:
    assert us_open_utc(date(2027, 3, 12), POLICY_V1) == utc(2027, 3, 12, 14, 30)
    assert us_open_utc(date(2027, 3, 15), POLICY_V1) == utc(2027, 3, 15, 13, 30)


def test_european_switch_does_not_move_the_us_open() -> None:
    # Europe leaves summer time a week before New York (2026-10-25); the US open must not care.
    assert us_open_utc(date(2026, 10, 26), POLICY_V1) == utc(2026, 10, 26, 13, 30)


def test_us_open_is_none_on_every_weekend_day_and_set_on_every_weekday() -> None:
    week = [date(2026, 9, 21) + timedelta(days=i) for i in range(7)]  # Monday .. Sunday
    opens = [us_open_utc(d, POLICY_V1) for d in week]
    assert all(o is not None for o in opens[:5])
    assert opens[5:] == [None, None]


def test_us_open_is_computed_through_the_iana_database() -> None:
    assert ZoneInfo("America/New_York") == NEW_YORK
    wall = datetime(2026, 11, 2, 9, 30, tzinfo=NEW_YORK)
    assert wall.tzname() == "EST"
    assert us_open_utc(date(2026, 11, 2), POLICY_V1) == wall.astimezone(UTC)


def test_us_open_follows_the_policy_local_time() -> None:
    policy = with_triggers(us_open_local="10:00")
    assert us_open_utc(date(2026, 9, 23), policy) == utc(2026, 9, 23, 14, 0)


def test_us_open_refuses_a_datetime_and_a_malformed_local_time() -> None:
    with pytest.raises(TypeError):
        us_open_utc(utc(2026, 9, 23, 13, 30), POLICY_V1)
    with pytest.raises(ValueError, match="HH:MM"):
        us_open_utc(date(2026, 9, 23), with_triggers(us_open_local="9:30am"))
    for bad in ("9:30", "24:00", "09:60", "0930", ""):
        with pytest.raises(ValueError, match="HH:MM"):
            parse_local_time(bad)


def test_a_wall_time_inside_a_spring_forward_gap_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # New York changes its clocks on Sundays, when there is no open, so the guard cannot bind
    # there. Israel changes on a Friday (2027-03-26, 02:00 -> 03:00), which exercises it.
    import sentiment_agent.events.schedule as schedule

    policy = with_triggers(us_open_local="02:30")
    assert us_open_utc(date(2027, 3, 15), policy) == utc(2027, 3, 15, 6, 30)  # exists in NY
    monkeypatch.setattr(schedule, "NEW_YORK", ZoneInfo("Asia/Jerusalem"))
    assert us_open_utc(date(2027, 3, 25), policy) == utc(2027, 3, 25, 0, 30)
    with pytest.raises(ValueError, match="does not exist"):
        us_open_utc(date(2027, 3, 26), policy)


# --- heartbeats ----------------------------------------------------------------------------------


def test_heartbeats_over_one_week() -> None:
    beats = heartbeats_between(utc(2026, 9, 21), utc(2026, 9, 28), POLICY_V1)
    opens = [t for t in beats if t.kind is TriggerKind.HEARTBEAT_US_OPEN]
    funding = [t for t in beats if t.kind is TriggerKind.HEARTBEAT_FUNDING]
    assert len(opens) == 5  # weekdays only
    assert len(funding) == 21  # 00:00 excluded on the 21st (start is open), 00:00 on the 28th in
    assert [t.fired_at for t in beats] == sorted(t.fired_at for t in beats)
    assert len({t.trigger_id for t in beats}) == len(beats)
    assert {t.fired_at.hour for t in funding} == {0, 8, 16}
    assert all(t.source == SCHEDULE_SOURCE for t in beats)
    assert funding[-1].fired_at == utc(2026, 9, 28, 0, 0)


def test_heartbeat_interval_excludes_start_and_includes_end() -> None:
    at = utc(2026, 9, 23, 13, 30)
    assert [t.fired_at for t in heartbeats_between(at, at + timedelta(minutes=1), POLICY_V1)] == []
    assert [t.fired_at for t in heartbeats_between(at - timedelta(seconds=1), at, POLICY_V1)] == [
        at
    ]
    assert heartbeats_between(at, at, POLICY_V1) == []


def test_consecutive_windows_return_each_heartbeat_exactly_once() -> None:
    edges = [utc(2026, 9, 23) + timedelta(minutes=37 * i) for i in range(200)]
    pieces = [t for a, b in pairwise(edges) for t in heartbeats_between(a, b, POLICY_V1)]
    whole = heartbeats_between(edges[0], edges[-1], POLICY_V1)
    assert [t.trigger_id for t in pieces] == [t.trigger_id for t in whole]


def test_heartbeat_fields() -> None:
    beats = heartbeats_between(utc(2026, 9, 23, 7), utc(2026, 9, 23, 14), POLICY_V1)
    assert [(t.kind, t.fired_at) for t in beats] == [
        (TriggerKind.HEARTBEAT_FUNDING, utc(2026, 9, 23, 8)),
        (TriggerKind.HEARTBEAT_US_OPEN, utc(2026, 9, 23, 13, 30)),
    ]
    funding, us_open = beats
    assert funding.symbols == ("BTCUSDT",)
    assert funding.trigger_id == heartbeat_id(TriggerKind.HEARTBEAT_FUNDING, utc(2026, 9, 23, 8))
    assert funding.trigger_id == "heartbeat_funding@2026-09-23T08:00:00Z"
    assert "BTCUSDT" not in us_open.symbols
    assert set(us_open.symbols) == {
        u.symbol for u in POLICY_V1.universe if u.asset_class.follows_us_session
    }
    assert "09:30 America/New_York (EDT)" in us_open.detail
    assert us_open.observed is None
    assert us_open.snapshot_id is None


def test_heartbeats_across_the_dst_switch() -> None:
    beats = heartbeats_between(utc(2026, 10, 30), utc(2026, 11, 3), POLICY_V1)
    opens = [t.fired_at for t in beats if t.kind is TriggerKind.HEARTBEAT_US_OPEN]
    assert opens == [utc(2026, 10, 30, 13, 30), utc(2026, 11, 2, 14, 30)]
    monday = [t for t in beats if t.kind is TriggerKind.HEARTBEAT_US_OPEN][-1]
    assert "(EST)" in monday.detail
    # Funding heartbeats are UTC and run every day, weekend included.
    funding = [t.fired_at for t in beats if t.kind is TriggerKind.HEARTBEAT_FUNDING]
    assert len(funding) == 12
    assert utc(2026, 11, 1, 8) in funding


def test_heartbeats_between_refuses_bad_intervals() -> None:
    with pytest.raises(ValueError, match="before start"):
        heartbeats_between(utc(2026, 9, 24), utc(2026, 9, 23), POLICY_V1)
    with pytest.raises(ValueError, match="timezone-aware"):
        heartbeats_between(datetime(2026, 9, 23), utc(2026, 9, 24), POLICY_V1)  # noqa: DTZ001


def test_heartbeats_accept_any_aware_zone() -> None:
    start = datetime(2026, 9, 23, 9, 29, tzinfo=NEW_YORK)
    beats = heartbeats_between(start, start + timedelta(minutes=2), POLICY_V1)
    assert [t.fired_at for t in beats] == [utc(2026, 9, 23, 13, 30)]
    assert beats[0].fired_at.tzinfo is UTC


def test_funding_hours_follow_the_policy() -> None:
    policy = with_triggers(funding_heartbeat_hours_utc=(4, 12, 20))
    beats = heartbeats_between(utc(2026, 9, 26), utc(2026, 9, 27), policy)  # a Saturday
    assert [t.fired_at.hour for t in beats] == [4, 12, 20]


def test_malformed_funding_hours_are_refused() -> None:
    with pytest.raises(ValidationError):
        type(POLICY_V1.triggers).model_validate(
            {**POLICY_V1.triggers.model_dump(), "funding_heartbeat_hours_utc": (0, 24)}
        )
    bypassed = bypass(
        POLICY_V1, triggers=bypass(POLICY_V1.triggers, funding_heartbeat_hours_utc=(0, 0, 8))
    )
    with pytest.raises(ValueError, match="distinct UTC hours"):
        heartbeats_between(utc(2026, 9, 23), utc(2026, 9, 24), bypassed)
    with pytest.raises(ValueError, match="distinct UTC hours"):
        validate_schedule(bypassed)


# --- weekend phases ------------------------------------------------------------------------------

FRIDAY = (2026, 9, 25)


@pytest.mark.parametrize(
    ("at", "phase"),
    [
        (utc(*FRIDAY, 0, 0), "open"),
        (utc(*FRIDAY, 17, 59), "open"),
        (utc(*FRIDAY, 17, 59, 59, 999999), "open"),
        (utc(*FRIDAY, 18, 0), "no_open_buffer"),
        (utc(*FRIDAY, 19, 44), "no_open_buffer"),
        (utc(*FRIDAY, 19, 44, 59, 999999), "no_open_buffer"),
        (utc(*FRIDAY, 19, 45), "preflatten"),
        (utc(*FRIDAY, 19, 59), "preflatten"),
        (utc(*FRIDAY, 19, 59, 59, 999999), "preflatten"),
        (utc(*FRIDAY, 20, 0), "frozen"),
        (utc(2026, 9, 26, 12, 0), "frozen"),  # Saturday
        (utc(2026, 9, 27, 23, 59), "frozen"),  # Sunday
        (utc(2026, 9, 27, 23, 59, 59, 999999), "frozen"),
        (utc(2026, 9, 28, 0, 0), "open"),  # Monday 00:00
        (utc(2026, 9, 28, 13, 30), "open"),
        (utc(2026, 9, 24, 20, 0), "open"),  # Thursday 20:00 is not the freeze
        (utc(2026, 9, 24, 19, 45), "open"),
    ],
)
def test_weekend_phase_boundaries(at: datetime, phase: WeekendPhase) -> None:
    assert weekend_phase(at, POLICY_V1) == phase


def test_weekend_phase_walks_the_whole_week_in_order() -> None:
    at = utc(2026, 9, 21)
    seen: list[str] = []
    while at < utc(2026, 9, 28, 0, 1):
        phase = weekend_phase(at, POLICY_V1)
        if not seen or seen[-1] != phase:
            seen.append(phase)
        at += timedelta(minutes=1)
    assert seen == ["open", "no_open_buffer", "preflatten", "frozen", "open"]


def test_weekend_phase_is_anchored_in_utc() -> None:
    # 15:59 in New York on Friday 2026-09-25 is 19:59 UTC.
    assert weekend_phase(datetime(2026, 9, 25, 15, 59, tzinfo=NEW_YORK), POLICY_V1) == "preflatten"
    # The freeze does not move with New York's clocks: same UTC boundaries either side of the DST
    # switch.
    assert weekend_phase(utc(2026, 10, 30, 19, 45), POLICY_V1) == "preflatten"
    assert weekend_phase(utc(2026, 11, 6, 19, 45), POLICY_V1) == "preflatten"
    assert weekend_phase(utc(2026, 11, 6, 20, 0), POLICY_V1) == "frozen"


def test_weekend_phase_refuses_naive_times() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        weekend_phase(datetime(2026, 9, 25, 20, 0), POLICY_V1)  # noqa: DTZ001


@pytest.mark.parametrize(
    ("at", "expected"),
    [
        (utc(2026, 9, 23, 13, 0), utc(*FRIDAY, 20, 0)),  # Wednesday
        (utc(*FRIDAY, 19, 59, 59), utc(*FRIDAY, 20, 0)),
        (utc(*FRIDAY, 20, 0), utc(2026, 10, 2, 20, 0)),  # the freeze has begun: next week's
        (utc(2026, 9, 26, 9, 0), utc(2026, 10, 2, 20, 0)),  # Saturday
        (utc(2026, 9, 28, 0, 0), utc(2026, 10, 2, 20, 0)),  # Monday 00:00
    ],
)
def test_next_freeze(at: datetime, expected: datetime) -> None:
    assert next_freeze(at, POLICY_V1) == expected
    assert next_freeze(at, POLICY_V1) > at


def test_weekend_rule_from_the_policy_is_used() -> None:
    rule = WeekendRule(
        freeze_weekday=5,  # Saturday 00:00
        freeze_hour=0,
        reopen_weekday=0,
        reopen_hour=6,  # Monday 06:00
        preflatten_minutes=30,
        no_open_buffer_hours=1.0,
        basis="test",
    )
    policy = POLICY_V1.model_copy(update={"weekend": rule})
    assert weekend_phase(utc(*FRIDAY, 22, 59), policy) == "open"
    assert weekend_phase(utc(*FRIDAY, 23, 0), policy) == "no_open_buffer"
    assert weekend_phase(utc(*FRIDAY, 23, 30), policy) == "preflatten"
    assert weekend_phase(utc(2026, 9, 26, 0, 0), policy) == "frozen"
    assert weekend_phase(utc(2026, 9, 28, 5, 59), policy) == "frozen"
    assert weekend_phase(utc(2026, 9, 28, 6, 0), policy) == "open"
    assert next_freeze(utc(2026, 9, 28, 6, 0), policy) == utc(2026, 10, 3, 0, 0)


def test_a_freeze_that_wraps_the_week_boundary() -> None:
    rule = WeekendRule(
        freeze_weekday=6,  # Sunday 22:00 to Monday 02:00
        freeze_hour=22,
        reopen_weekday=0,
        reopen_hour=2,
        preflatten_minutes=0,
        no_open_buffer_hours=0.0,
        basis="test",
    )
    policy = POLICY_V1.model_copy(update={"weekend": rule})
    assert weekend_phase(utc(2026, 9, 27, 21, 59), policy) == "open"
    assert weekend_phase(utc(2026, 9, 27, 22, 0), policy) == "frozen"
    assert weekend_phase(utc(2026, 9, 28, 1, 59), policy) == "frozen"
    assert weekend_phase(utc(2026, 9, 28, 2, 0), policy) == "open"


def test_a_degenerate_weekend_rule_is_refused() -> None:
    with pytest.raises(ValidationError):
        WeekendRule.model_validate(
            {**POLICY_V1.weekend.model_dump(), "reopen_weekday": 4, "reopen_hour": 20}
        )
    # Construction refuses these; the module refuses them too if one is smuggled past it.
    empty = bypass(POLICY_V1, weekend=bypass(POLICY_V1.weekend, reopen_weekday=4, reopen_hour=20))
    with pytest.raises(ValueError, match="different times"):
        weekend_phase(utc(*FRIDAY, 12, 0), empty)
    inverted = bypass(POLICY_V1, weekend=bypass(POLICY_V1.weekend, preflatten_minutes=180))
    with pytest.raises(ValueError, match="pre-flattening"):
        validate_schedule(inverted)


def test_iso_z() -> None:
    assert iso_z(utc(2026, 9, 23, 13, 30)) == "2026-09-23T13:30:00Z"
    assert iso_z(datetime(2026, 9, 23, 9, 30, tzinfo=NEW_YORK)) == "2026-09-23T13:30:00Z"


def test_the_kernel_and_the_schedule_agree_on_every_minute_of_the_week() -> None:
    # G2 in the kernel and the trigger engine's weekend refusal read the same rule; if the two
    # implementations ever drift, a trigger could be admitted for a leg the kernel holds frozen.
    guards = pytest.importorskip("sentiment_agent.kernel.guards")
    at = utc(2026, 9, 21)
    edges = [
        utc(*FRIDAY, h, m) - timedelta(microseconds=1) for h, m in ((18, 0), (19, 45), (20, 0))
    ]
    edges.append(utc(2026, 9, 28) - timedelta(microseconds=1))
    moments = [at + timedelta(minutes=i) for i in range(7 * 24 * 60 + 1)] + edges
    for moment in moments:
        assert guards.weekend_phase(moment, POLICY_V1.weekend) == weekend_phase(moment, POLICY_V1)
