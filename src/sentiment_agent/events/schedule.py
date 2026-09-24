"""The heartbeat schedule and the weekend phases (DESIGN.md §8 and §10.3 G2).

Two clocks decide when the agent wakes without being asked:

* **The US cash open**, 09:30 America/New_York on weekdays. New York moves between EDT (UTC-4) and
  EST (UTC-5), so the heartbeat is 13:30 UTC in summer time and 14:30 UTC in winter time. It is
  computed through :mod:`zoneinfo` with the ``tzdata`` package (Windows ships no IANA database),
  never from a fixed UTC offset: on 2026-11-01 New York leaves summer time, and the Monday open
  moves from 13:30 to 14:30 UTC.
* **Each live BTCUSDT funding settlement**, at the UTC hours in ``policy.triggers`` (00, 08, 16:
  live BTCUSDT settles every 8 hours; the runtime re-reads ``fundInterval`` at genesis). Crypto
  trades every day, so these fire on weekends too.

US market holidays are deliberately not skipped. A holiday heartbeat still wakes the model for the
crypto leg, and the stale-index detector in G1 refuses new exposure on an equity perp whose Demo
index has stopped moving, which covers holidays and any unannounced Demo freeze without a
calendar (DESIGN.md §19.3).

The weekend phases describe the G2 window, which is fixed in UTC (the envelope simulation's
``tradable()`` in ``validation/demo_venue/envelope_clean.py:28-31`` uses the same window, and the
Demo venue's weekend freeze it answers is in ``weekend_vol.json``):

========================  =========================================  ==============================
phase                     policy v1 window (UTC)                     what the kernel does
========================  =========================================  ==============================
``open``                  Monday 00:00 to Friday 18:00               nothing
``no_open_buffer``        Friday 18:00 to 19:45                      no new US-session exposure
``preflatten``            Friday 19:45 to 20:00                      US-session legs are closed
``frozen``                Friday 20:00 to Monday 00:00               US-session legs stay flat
========================  =========================================  ==============================

Each window is closed at its start and open at its end: Friday 19:45:00 is ``preflatten`` and
Monday 00:00:00 is ``open``. The phases are a property of the clock alone; which legs they bind is
the kernel's question (``AssetClass.follows_us_session``).

Every function here is pure: the same arguments give the same answer, with no clock read.
"""

import re
from datetime import UTC, date, datetime, time, timedelta
from typing import Final, Literal
from zoneinfo import ZoneInfo

from sentiment_agent.types import AssetClass, Policy, Trigger, TriggerKind, WeekendRule

NEW_YORK: Final = ZoneInfo("America/New_York")
"""The US cash market's zone. Resolved from the ``tzdata`` package where the OS has no database."""

SCHEDULE_SOURCE: Final = "schedule"
"""``Trigger.source`` of every heartbeat."""

WEEK: Final = timedelta(days=7)

WeekendPhase = Literal["open", "no_open_buffer", "preflatten", "frozen"]

_HH_MM: Final = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


# ------------------------------------------------------------------------------------------------
# Time helpers
# ------------------------------------------------------------------------------------------------


def as_utc(value: datetime, *, name: str = "time") -> datetime:
    """``value`` in UTC. A naive datetime is refused: it has no instant to convert."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware, got a naive datetime")
    return value.astimezone(UTC)


def iso_z(value: datetime) -> str:
    """ISO-8601 in UTC with a ``Z`` suffix, e.g. ``2026-09-23T13:30:00Z``."""
    return as_utc(value).isoformat().replace("+00:00", "Z")


def parse_local_time(text: str) -> time:
    """``"HH:MM"`` (24-hour) as a :class:`datetime.time`. Anything else is refused."""
    match = _HH_MM.fullmatch(text)
    if match is None:
        raise ValueError(f"expected a 24-hour HH:MM local time, got {text!r}")
    return time(int(match.group(1)), int(match.group(2)))


def funding_hours(policy: Policy) -> tuple[int, ...]:
    """The funding-heartbeat hours, sorted. Out-of-range or repeated hours are refused."""
    hours = policy.triggers.funding_heartbeat_hours_utc
    if any(not 0 <= hour <= 23 for hour in hours) or len(set(hours)) != len(hours):
        raise ValueError(f"funding heartbeat hours must be distinct UTC hours 0-23, got {hours}")
    return tuple(sorted(hours))


# ------------------------------------------------------------------------------------------------
# Heartbeats
# ------------------------------------------------------------------------------------------------


def us_open_utc(day: date, policy: Policy) -> datetime | None:
    """The US cash open on New York calendar date ``day``, in UTC; ``None`` on a weekend.

    ``policy.triggers.us_open_local`` is a New York wall-clock time. A wall time that does not exist
    on ``day`` (inside a spring-forward gap) is refused rather than silently moved; 09:30 is never
    in one, since New York changes its clocks at 02:00.
    """
    if isinstance(day, datetime):
        raise TypeError("us_open_utc takes the New York calendar date, not a datetime")
    if day.weekday() >= 5:
        return None
    wall = datetime.combine(day, parse_local_time(policy.triggers.us_open_local), tzinfo=NEW_YORK)
    instant = wall.astimezone(UTC)
    if instant.astimezone(NEW_YORK).replace(tzinfo=None) != wall.replace(tzinfo=None):
        raise ValueError(f"{wall.time():%H:%M} does not exist in New York on {day.isoformat()}")
    return instant


def heartbeat_id(kind: TriggerKind, scheduled: datetime) -> str:
    """Deterministic id of a heartbeat: its kind and scheduled instant.

    A heartbeat is the same trigger however many times the schedule is read, so a restarted
    engine recognises one it already processed (``TriggerEngine.restore``).
    """
    return f"{kind.value}@{iso_z(scheduled)}"


def _us_session_symbols(policy: Policy) -> tuple[str, ...]:
    return tuple(u.symbol for u in policy.universe if u.asset_class.follows_us_session)


def _crypto_symbols(policy: Policy) -> tuple[str, ...]:
    return tuple(u.symbol for u in policy.universe if u.asset_class is AssetClass.CRYPTO)


def _us_open_trigger(day: date, scheduled: datetime, policy: Policy) -> Trigger:
    local = scheduled.astimezone(NEW_YORK)
    return Trigger(
        trigger_id=heartbeat_id(TriggerKind.HEARTBEAT_US_OPEN, scheduled),
        kind=TriggerKind.HEARTBEAT_US_OPEN,
        fired_at=scheduled,
        symbols=_us_session_symbols(policy),
        detail=(
            f"US cash open {local:%H:%M} America/New_York ({local.tzname()}) on "
            f"{day.isoformat()}, scheduled {iso_z(scheduled)}"
        ),
        source=SCHEDULE_SOURCE,
    )


def _funding_trigger(scheduled: datetime, policy: Policy) -> Trigger:
    symbols = _crypto_symbols(policy)
    named = ", ".join(symbols) if symbols else "crypto"
    return Trigger(
        trigger_id=heartbeat_id(TriggerKind.HEARTBEAT_FUNDING, scheduled),
        kind=TriggerKind.HEARTBEAT_FUNDING,
        fired_at=scheduled,
        symbols=symbols,
        detail=f"live {named} funding settlement at {scheduled:%H:%M} UTC, scheduled "
        f"{iso_z(scheduled)}",
        source=SCHEDULE_SOURCE,
    )


def heartbeats_between(start: datetime, end: datetime, policy: Policy) -> list[Trigger]:
    """Every heartbeat scheduled in the half-open interval ``(start, end]``, in time order.

    ``start`` is excluded and ``end`` included, so consecutive calls over ``(t0, t1]``, ``(t1, t2]``
    return each heartbeat exactly once. ``fired_at`` is the scheduled instant; the engine's
    ``admit`` re-stamps it with the admission instant. ``end`` before ``start`` is refused.
    """
    lo = as_utc(start, name="start")
    hi = as_utc(end, name="end")
    if hi < lo:
        raise ValueError(f"end {iso_z(hi)} is before start {iso_z(lo)}")
    out: list[Trigger] = []
    if hi == lo:
        return out
    hours = funding_hours(policy)
    utc_day = lo.date()
    while utc_day <= hi.date():
        for hour in hours:
            scheduled = datetime(utc_day.year, utc_day.month, utc_day.day, hour, tzinfo=UTC)
            if lo < scheduled <= hi:
                out.append(_funding_trigger(scheduled, policy))
        utc_day += timedelta(days=1)
    ny_day = lo.astimezone(NEW_YORK).date()
    while ny_day <= hi.astimezone(NEW_YORK).date():
        opens = us_open_utc(ny_day, policy)
        if opens is not None and lo < opens <= hi:
            out.append(_us_open_trigger(ny_day, opens, policy))
        ny_day += timedelta(days=1)
    out.sort(key=lambda t: (t.fired_at, t.kind.value))
    return out


# ------------------------------------------------------------------------------------------------
# Weekend phases (G2)
# ------------------------------------------------------------------------------------------------


def _geometry(rule: WeekendRule) -> tuple[timedelta, timedelta, timedelta, timedelta]:
    """(freeze offset from Monday 00:00 UTC, freeze length, pre-flatten lead, no-open lead)."""
    freeze = timedelta(days=rule.freeze_weekday, hours=rule.freeze_hour)
    reopen = timedelta(days=rule.reopen_weekday, hours=rule.reopen_hour)
    length = (reopen - freeze) % WEEK
    preflatten = timedelta(minutes=rule.preflatten_minutes)
    buffer = timedelta(hours=rule.no_open_buffer_hours)
    if length == timedelta(0):
        raise ValueError("the weekend freeze must start and end at different times")
    if preflatten > buffer:
        raise ValueError("pre-flattening cannot start before new exposure is refused")
    if length + buffer >= WEEK:
        raise ValueError("the weekend freeze and its lead-in cover the whole week")
    return freeze, length, preflatten, buffer


def _last_freeze_start(at: datetime, freeze: timedelta) -> datetime:
    """The latest freeze start at or before ``at`` (``at`` in UTC)."""
    monday = (at - timedelta(days=at.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    start = monday + freeze
    return start if start <= at else start - WEEK


def weekend_phase(at: datetime, policy: Policy) -> WeekendPhase:
    """Where ``at`` falls in the G2 weekend window (see the module docstring)."""
    now = as_utc(at, name="at")
    freeze, length, preflatten, buffer = _geometry(policy.weekend)
    last = _last_freeze_start(now, freeze)
    if now < last + length:
        return "frozen"
    upcoming = last + WEEK
    if now >= upcoming - preflatten:
        return "preflatten"
    if now >= upcoming - buffer:
        return "no_open_buffer"
    return "open"


def next_freeze(at: datetime, policy: Policy) -> datetime:
    """The first freeze start strictly after ``at``.

    At the instant a freeze begins, the answer is the following week's: that freeze has already
    started, and :func:`weekend_phase` says so.
    """
    now = as_utc(at, name="at")
    freeze, _, _, _ = _geometry(policy.weekend)
    return _last_freeze_start(now, freeze) + WEEK


def validate_schedule(policy: Policy) -> None:
    """Raise ``ValueError`` if the policy's schedule cannot be computed. Used at engine start."""
    parse_local_time(policy.triggers.us_open_local)
    funding_hours(policy)
    _geometry(policy.weekend)


__all__ = [
    "NEW_YORK",
    "SCHEDULE_SOURCE",
    "WEEK",
    "WeekendPhase",
    "as_utc",
    "funding_hours",
    "heartbeat_id",
    "heartbeats_between",
    "iso_z",
    "next_freeze",
    "parse_local_time",
    "us_open_utc",
    "validate_schedule",
    "weekend_phase",
]
