"""Pacing a reduce-only order the venue refuses as :data:`~environment.VENUE_SYMBOL_UNAVAILABLE`.

run3-d1. What run 2's own record showed (``public/orders.json``, read against the live ledger
2026-09-29): between 2026-09-28 09:16:43Z and 15:11:33Z, every protective-exit and close order for
METAUSDT (72) and MSTRUSDT (2) was refused by Bitget Demo with HTTP 400 ``"Parameter ..._UMCBL does
not exist"``. The kernel's G10 correctly kept forcing the exit every cycle (the position was still
open; that part of the design worked exactly as built), but nothing paced the *sends*: the planner
minted a fresh ``clientOid`` from a fresh ``ruling_id`` every time, so each attempt was a new order,
logged as a new rejection, for close to seven hours, until an identical request filled at 16:06:52Z.

This module turns that record into a bounded backoff instead of removing the persistence:

* :func:`blocked_exit_episodes` reads what already exists — ``Projection.submissions``,
  ``.rejections`` and ``.acks`` — and rebuilds, per symbol, whether its most recent reduce-only
  order is still stuck behind a venue-side symbol refusal, and since when. Nothing new is written
  to the ledger to make this work: it is a pure projection, recomputed the same way
  ``App.venue_unreconciled`` already is (``runtime/wiring.py``), so a restart mid-episode picks up
  exactly where the ledger left it, with no separate persistence to get out of sync.
* :func:`next_retry_at` is the schedule: exponential, capped, seeded from the run 2 numbers above.
  Base 5 minutes, factor 2, capped at 60 minutes: a symbol stuck for the same ~5h55m run 2's was
  gets on the order of 10 real attempts instead of 72 (5 + 10 + 20 + 40 + 60x5 = 375 minutes of an
  ~355-minute episode reaches attempt 9), each one a genuine venue interaction worth recording, not
  a duplicate of the one before it.
* The gate the planner applies before it mints an exit leg (``kernel/planner.py``, ``_Planner.emit``
  — inlined there rather than called from here, because ``kernel/`` never imports ``execution/``:
  every module under ``kernel/`` imports only from ``types`` and ``hashing``, a boundary this module
  does not cross) is: the schedule first, and — opportunistically, at no extra network cost, since
  specs are already refreshed into ``KernelInputs.specs`` — the last-read ``InstrumentSpec.status``
  (``GET /api/v3/market/instruments``, the ``status`` field parsed at ``venue/public_api.py:422``;
  the operation itself is ``getInstruments``, ``agent-sdk/src/generated/catalog.ts:109``, public, no
  credential) second. The status check can only ever *extend* a wait, never shorten one: run 2's own
  ``validation/demo_venue/universe_probe.json`` shows both METAUSDT and MSTRUSDT
  ``status: "online"`` on 2026-09-24, four days before the block, so "online" is demonstrably not
  sufficient evidence the contract is reachable right now. And specs refresh only every 24 hours
  (``runtime/loop.py: SPECS_EVERY``), far coarser than the minutes-to-hours an episode like this
  lasts, so in practice this project backs off on time alone, with the status read as a free,
  occasional accelerant when it happens to be fresh and says "not online" — never as a reason to
  send early. NOT VERIFIED: whether Bitget's ``status`` field actually flips away from
  ``"online"`` during a gap like 2026-09-28's; no probe was taken *during* that window.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final

from sentiment_agent.execution.environment import VENUE_SYMBOL_UNAVAILABLE
from sentiment_agent.types import OrderSubmitted, VenueAck, VenueRejection

BACKOFF_BASE: Final = timedelta(minutes=5)
BACKOFF_FACTOR: Final = 2.0
BACKOFF_CAP: Final = timedelta(minutes=60)
"""run3-d1's schedule (module docstring): attempt *n* waits ``min(BASE * FACTOR**(n-1), CAP)`` after
the attempt before it. Never below one real attempt (``n=1`` always waits ``BACKOFF_BASE``): a
single rejection is not yet a pattern, so the first retry is still prompt."""


@dataclass(frozen=True, slots=True)
class ExitEpisode:
    """One symbol's still-open run of :data:`~environment.VENUE_SYMBOL_UNAVAILABLE` refusals on a
    reduce-only order, rebuilt from the ledger, never persisted on its own."""

    symbol: str
    first_attempt_at: datetime
    last_attempt_at: datetime
    attempt_count: int
    last_message: str
    next_retry_at: datetime

    def summary(self, *, cleared_at: datetime | None = None) -> str:
        span = (cleared_at or self.last_attempt_at) - self.first_attempt_at
        state = f"cleared at {cleared_at.isoformat()}" if cleared_at else "still blocked"
        return (
            f"{self.symbol}: exit blocked by the venue since {self.first_attempt_at.isoformat()} "
            f"({self.attempt_count} attempt(s) over {span}, "
            f"last {self.last_attempt_at.isoformat()}, {state}): {self.last_message}"
        )


def next_retry_at(attempt_count: int, last_attempt_at: datetime) -> datetime:
    """When the ``attempt_count``-th refusal may be followed by another real attempt.

    ``attempt_count`` is 1 for the first refusal of an episode. The wait grows geometrically from
    :data:`BACKOFF_BASE` and never exceeds :data:`BACKOFF_CAP`."""
    if attempt_count < 1:
        raise ValueError(f"attempt_count must be >= 1, got {attempt_count}")
    wait = min(BACKOFF_BASE * (BACKOFF_FACTOR ** (attempt_count - 1)), BACKOFF_CAP)
    return last_attempt_at + wait


def blocked_exit_episodes(
    submissions: Sequence[OrderSubmitted],
    rejections: Sequence[VenueRejection],
    acks: Sequence[VenueAck],
) -> dict[str, ExitEpisode]:
    """Every symbol whose most recent reduce-only order is still stuck behind a venue-side symbol
    refusal, keyed by symbol.

    Pure and stateless: replays ``submissions``/``rejections``/``acks`` in time order (as
    ``Projection`` already carries them) and needs nothing else, so it gives the same answer before
    and after a restart. A reduce-only order acknowledged by the venue clears its symbol's episode
    (the contract plainly exists now); a rejection of any other shape neither starts nor extends
    one, because only this specific classification is known to resolve on its own (module
    docstring) — a balance or size refusal is a different problem the approval layer already
    surfaces, and resending it on a timer would be wrong.
    """
    by_oid: dict[str, OrderSubmitted] = {s.client_oid: s for s in submissions}
    timeline: list[tuple[datetime, str, VenueRejection | VenueAck]] = []
    for rejection in rejections:
        submitted = by_oid.get(rejection.client_oid)
        if submitted is None or not submitted.intent.reduce_only:
            continue
        timeline.append((rejection.at, submitted.intent.symbol, rejection))
    for ack in acks:
        submitted = by_oid.get(ack.client_oid)
        if submitted is None or not submitted.intent.reduce_only:
            continue
        timeline.append((ack.acked_at, submitted.intent.symbol, ack))
    timeline.sort(key=lambda row: row[0])

    episodes: dict[str, ExitEpisode] = {}
    for at, symbol, event in timeline:
        if isinstance(event, VenueAck):
            episodes.pop(symbol, None)
            continue
        if event.category != VENUE_SYMBOL_UNAVAILABLE:
            continue
        prior = episodes.get(symbol)
        count = 1 if prior is None else prior.attempt_count + 1
        episodes[symbol] = ExitEpisode(
            symbol=symbol,
            first_attempt_at=at if prior is None else prior.first_attempt_at,
            last_attempt_at=at,
            attempt_count=count,
            last_message=event.message,
            next_retry_at=next_retry_at(count, at),
        )
    return episodes


__all__ = [
    "BACKOFF_BASE",
    "BACKOFF_CAP",
    "BACKOFF_FACTOR",
    "ExitEpisode",
    "blocked_exit_episodes",
    "next_retry_at",
]
