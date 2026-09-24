"""Time for the replica: UTC throughout, the weekly US-session freeze, and the two heartbeats.

The sandbox allows ``datetime`` but not ``zoneinfo``, so the US cash open (09:30
America/New_York) is computed from the US daylight-saving rule in force since 2007: daylight time
from the second Sunday of March to the first Sunday of November, both changes at 02:00 local. The
open is on a weekday and the changes happen on Sundays, so the open is never inside a transition;
the primary computes the same instant with ``zoneinfo``
(``sentiment_agent.events.schedule.us_open_utc``) and the replica's tests compare the two over
several years of dates.

``weekend_phase`` is a line-for-line port of ``sentiment_agent.kernel.guards.weekend_phase``.
"""

from datetime import UTC, date, datetime, timedelta

from . import policy_v1 as policy

HOUR = timedelta(hours=1)

EASTERN_STANDARD = timedelta(hours=-5)
EASTERN_DAYLIGHT = timedelta(hours=-4)

HEARTBEAT_US_OPEN = "heartbeat_us_open"
HEARTBEAT_FUNDING = "heartbeat_funding"


def utc_now() -> datetime:
    return datetime.now(UTC)


def as_utc(value: datetime) -> datetime:
    """``value`` in UTC. A naive datetime is refused: every instant here is an instant."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("naive datetimes are refused; use timezone-aware UTC")
    return value.astimezone(UTC)


def iso_z(value: datetime) -> str:
    """ISO 8601 in UTC with a ``Z`` suffix, microseconds only when non-zero."""
    text = as_utc(value).isoformat()
    return text[: -len("+00:00")] + "Z" if text.endswith("+00:00") else text


def parse_iso(text: object) -> datetime | None:
    """A UTC instant from an ISO string (``Z`` or an offset). ``None`` for anything else."""
    if not isinstance(text, str) or not text.strip():
        return None
    raw = text.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def utc_day(at: datetime) -> str:
    """The UTC calendar date of ``at``, ``YYYY-MM-DD``."""
    return as_utc(at).date().isoformat()


def next_utc_midnight(at: datetime) -> datetime:
    moment = as_utc(at)
    return moment.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)


# ------------------------------------------------------------------------------------------------
# The weekly freeze (G2)
# ------------------------------------------------------------------------------------------------


def _freeze_starts(at: datetime) -> list[datetime]:
    week_start = (at - timedelta(days=at.weekday())).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    offset = timedelta(days=policy.WEEKEND_FREEZE_WEEKDAY, hours=policy.WEEKEND_FREEZE_HOUR)
    return [week_start + offset + timedelta(weeks=k) for k in (-1, 0, 1)]


def _freeze_length() -> timedelta:
    hours = (
        policy.WEEKEND_REOPEN_WEEKDAY * 24
        + policy.WEEKEND_REOPEN_HOUR
        - policy.WEEKEND_FREEZE_WEEKDAY * 24
        - policy.WEEKEND_FREEZE_HOUR
    ) % (7 * 24)
    return timedelta(hours=hours)


def weekend_phase(at: datetime) -> str:
    """``frozen`` inside the freeze, ``preflatten`` in the minutes before it, ``no_open_buffer`` in
    the hours before that, ``open`` otherwise. Each window includes its start and excludes its end.
    """
    moment = as_utc(at)
    starts = _freeze_starts(moment)
    length = _freeze_length()
    if length > timedelta(0) and any(s <= moment < s + length for s in starts):
        return "frozen"
    until = min(s for s in starts if s > moment) - moment
    if until <= timedelta(minutes=policy.WEEKEND_PREFLATTEN_MINUTES):
        return "preflatten"
    if until <= timedelta(hours=policy.WEEKEND_NO_OPEN_BUFFER_HOURS):
        return "no_open_buffer"
    return "open"


def hours_to_freeze(at: datetime) -> float:
    """Hours until the next freeze starts; ``0.0`` inside the freeze."""
    moment = as_utc(at)
    if weekend_phase(moment) == "frozen":
        return 0.0
    upcoming = min(s for s in _freeze_starts(moment) if s > moment)
    return (upcoming - moment).total_seconds() / 3600.0


def follows_us_session(asset_class: str | None) -> bool:
    return asset_class in policy.US_SESSION_ASSET_CLASSES


def session_open(asset_class: str | None, at: datetime) -> bool:
    """Whether the instrument's price should be moving: always for crypto, outside the freeze for
    US legs. An unknown asset class is treated as always open (the strictest reading for G1)."""
    if asset_class is None or not follows_us_session(asset_class):
        return True
    return weekend_phase(at) != "frozen"


# ------------------------------------------------------------------------------------------------
# New York and the heartbeats
# ------------------------------------------------------------------------------------------------


def _nth_sunday(year: int, month: int, n: int) -> date:
    first = date(year, month, 1)
    to_sunday = (6 - first.weekday()) % 7
    return first + timedelta(days=to_sunday + 7 * (n - 1))


def new_york_offset_on(day: date) -> timedelta:
    """UTC offset of New York wall time on the civil date ``day`` after 02:00 local."""
    starts = _nth_sunday(day.year, 3, 2)
    ends = _nth_sunday(day.year, 11, 1)
    return EASTERN_DAYLIGHT if starts <= day < ends else EASTERN_STANDARD


def us_open_utc(day: date) -> datetime | None:
    """The US cash open on New York calendar date ``day``, in UTC; ``None`` on a weekend."""
    if day.weekday() >= 5:
        return None
    hour_text, minute_text = policy.TRIGGER_US_OPEN_LOCAL.split(":")
    wall = datetime(day.year, day.month, day.day, int(hour_text), int(minute_text), tzinfo=UTC)
    return wall - new_york_offset_on(day)


def heartbeats_between(start: datetime, end: datetime) -> list[tuple[str, datetime]]:
    """Every heartbeat scheduled in ``(start, end]``, in time order, as ``(kind, instant)``."""
    lo = as_utc(start)
    hi = as_utc(end)
    if hi <= lo:
        return []
    found: list[tuple[str, datetime]] = []
    day = lo.date() - timedelta(days=1)
    last = hi.date() + timedelta(days=1)
    while day <= last:
        for hour in sorted(policy.TRIGGER_FUNDING_HEARTBEAT_HOURS_UTC):
            scheduled = datetime(day.year, day.month, day.day, hour, tzinfo=UTC)
            if lo < scheduled <= hi:
                found.append((HEARTBEAT_FUNDING, scheduled))
        opens = us_open_utc(day)
        if opens is not None and lo < opens <= hi:
            found.append((HEARTBEAT_US_OPEN, opens))
        day += timedelta(days=1)
    found.sort(key=lambda item: (item[1], item[0]))
    return found


def heartbeat_id(kind: str, scheduled: datetime) -> str:
    return f"{kind}@{iso_z(scheduled)}"
