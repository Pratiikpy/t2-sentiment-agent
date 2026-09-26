"""The decision prompt: one system message and one user message per decision cycle.

The system message (``prompts/system_v1.md``, with ``prompts/output_schema_v1.md`` inside it)
states the role, the risk kernel's rules as facts the model cannot change, the spotlight instruction
and the output contract. The user message carries the triggers that woke the agent, the mandate, the
book, the session, the market mood, every instrument's positioning facts, the crowd's distinct
stories, the calendar, and which sources answered (DESIGN.md §9.2).

Three properties are built in, and each has a test:

* **Every number shown is a citable fact, and every fact is shown.** :func:`decision_facts` is the
  one map the user message's numbers come from, each written ``key = value`` with its full key,
  and it is also the reference the grounding check (guard G9) resolves the model's numbers against.
  It is the snapshot's own ``facts`` (perception's half, ``perception/features.py``), the same
  perception functions applied to the book in hand where the snapshot did not carry it, and the
  numbers only a decision cycle knows: the kernel's rules, the session clock, the triggers, the
  per-story figures in display order, the book as percentages, and how much of each budget and
  of the hold window is left. The system message's numbers are the kernel's,
  each also a ``kernel.*`` fact. So a number the model copies from its prompt resolves, and a
  number it invents does not.
* **Every piece of third-party text is spotlighted.** Crowd stories, account names, calendar
  titles, labels that are not plainly safe, and trigger details are wrapped with
  :func:`sentiment_agent.crowd.quarantine.spotlight`; the standing instruction sits in the system
  message. An item the screen withheld is never shown, not even its source.
* **The prompt is pinned.** :func:`prompt_hashes` hashes both templates and this module's source
  (line endings normalised) for the genesis record, so a change to what the model is shown after
  genesis is visible as an amendment.

Size is a design constraint: the daily token budget projects one token per prompt byte
(``llm/budget.py``), so every kilobyte here is a kilobyte less for later decisions that day. Facts
are packed several to a line and the crowd section shows the most significant stories only.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from string import Template
from typing import Final, Literal

from sentiment_agent.crowd.quarantine import (
    REDACTION,
    SPOTLIGHT_CLOSE,
    SPOTLIGHT_OPEN,
    STANDING_INSTRUCTION,
    inspect,
    spotlight,
)
from sentiment_agent.decision.grounding import extract
from sentiment_agent.hashing import sha256_hex
from sentiment_agent.perception.features import coverage_facts, crowd_facts, facts_from
from sentiment_agent.types import (
    AssetClass,
    BookState,
    ChatMessage,
    PerceptionSnapshot,
    Policy,
    Position,
    PositioningFeatures,
    ScreenedItem,
    SourceHealth,
    StoryCluster,
    Trigger,
    TriggerKind,
)

PROMPT_VERSION: Final = "prompt-v1"

PACKAGE_DIR: Final = Path(__file__).resolve().parents[1]
"""``src/sentiment_agent``: hashed paths are recorded relative to it."""
SYSTEM_TEMPLATE: Final = "decision/prompts/system_v1.md"
SCHEMA_TEMPLATE: Final = "decision/prompts/output_schema_v1.md"
HASHED_SOURCES: Final[tuple[str, ...]] = (SYSTEM_TEMPLATE, SCHEMA_TEMPLATE, "decision/prompt.py")
"""What :func:`prompt_hashes` pins: both templates, and this module, which renders the user turn."""

SIGNIFICANT_DIGITS: Final = 6
"""Every number is rendered with at most six significant digits and never in exponent notation.
Six keeps a rendered value within 0.0005% of the fact, far inside the grounding tolerance."""

FACTS_PER_LINE: Final = 3
FACT_SEPARATOR: Final = "; "

MAX_STORIES: Final = 20
"""Distinct crowd stories shown per decision, most significant first (:func:`ranked_stories`).
Stories below the cut are counted in ``crowd.*`` and not shown; their figures are not facts."""
MAX_STORY_CHARS: Final = 360
MAX_STORY_IDS: Final = 8
MAX_CALENDAR_ITEMS: Final = 20
MAX_WITHHELD_LISTED: Final = 20
MAX_LABEL_CHARS: Final = 120

HEARTBEAT_KINDS: Final[frozenset[TriggerKind]] = frozenset(
    {TriggerKind.HEARTBEAT_US_OPEN, TriggerKind.HEARTBEAT_FUNDING}
)

FEATURE_FACTS: Final[tuple[str, ...]] = (
    "demo_last",
    "live_last",
    "funding_rate_live",
    "funding_z_live",
    "open_interest_live",
    "oi_change_1h_pct",
    "oi_change_24h_pct",
    "retail_long_short_ratio",
    "top_trader_long_short_ratio",
    "taker_buy_sell_ratio",
    "price_change_24h_pct",
    "ma20_distance_atr",
    "social_mentions_24h",
    "social_velocity_per_hour",
    "demo_mark_index_gap_bps",
    "demo_live_gap_bps",
    "demo_spread_bps",
    "demo_index_move_bps_3h",
)
"""Every numeric field of :class:`~sentiment_agent.types.PositioningFeatures`, in display order.
A test fails if a numeric field is added to the contract and not listed here."""

INSTRUMENT_FACTS: Final[tuple[str, ...]] = (
    "demo_live_gap_limit_bps",
    "model_orders_today",
    "model_orders_left_today",
)
"""Per-instrument facts after the features, each keyed ``SYMBOL.name``. The order counters are
shown only for a name traded today; the message says that an absent counter means none."""

POSITION_FACTS: Final[tuple[str, ...]] = (
    "position_qty",
    "position_avg_entry",
    "position_mark",
    "position_weight_pct",
    "position_target_equivalent",
    "position_move_since_entry_pct",
    "position_unrealized_pnl",
    "position_realized_pnl",
    "position_fees_paid",
    "position_hours_held",
    "position_hours_since_increase",
    "position_hours_until_increase_allowed",
    "position_stop_price",
    "position_stop_distance_pct",
)
"""An open position's facts, each keyed ``SYMBOL.name``. Recorded numbers (qty, entry, mark, P&L,
fees, stop) are perception's; the rest depend on the decision clock and are derived here."""

BOOK_FACTS: Final[tuple[str, ...]] = (
    "book.equity",
    "book.starting_equity",
    "book.peak_equity",
    "book.day_open_equity",
    "book.day_return_pct",
    "book.drawdown_pct",
    "book.gross_weight_pct",
    "book.net_weight_pct",
    "book.risk_budget_left_pct",
    "book.open_positions",
    "book.consecutive_losses",
    "book.realized_total",
    "book.fees_today",
    "book.fees_total",
    "book.fees_today_bps",
    "book.fees_total_bps",
    "book.fee_budget_left_today_bps",
    "book.fee_budget_left_window_bps",
    "book.daily_kill_distance_pct",
    "book.reduce_only_distance_pct",
)

CROWD_FACTS: Final[tuple[str, ...]] = (
    "crowd.items",
    "crowd.withheld",
    "crowd.distinct_stories",
    "crowd.duplication_ratio",
    "crowd.coordinated_clusters",
)

STORY_FACTS: Final[tuple[str, ...]] = (
    "items",
    "distinct_sources",
    "velocity_per_hour",
    "span_minutes",
)
"""Per shown story, keyed ``story.<rank>.name`` in display order."""

FACT_LEGEND: Final[tuple[tuple[str, str], ...]] = (
    ("demo_last", "last trade price on Bitget UTA Demo, the venue your orders fill on (USDT)"),
    ("live_last", "last trade price on Bitget's live market (USDT), where the real crowd trades"),
    ("funding_rate_live", "live funding rate per settlement, a fraction; positive: longs pay"),
    (
        "funding_z_live",
        "z-score of live funding against its last 90 settlements; strongly positive means crowded "
        "longs, strongly negative crowded shorts",
    ),
    ("open_interest_live", "live open interest as Bitget reports it"),
    (
        "oi_change_1h_pct / oi_change_24h_pct",
        "change in open interest on Binance USD-M futures, the market Bitget's data services read "
        "crowd positioning from, percent",
    ),
    (
        "retail_long_short_ratio",
        "Binance USD-M accounts long per account short; above parity means more accounts are long",
    ),
    ("top_trader_long_short_ratio", "the same for Binance USD-M's top traders"),
    ("taker_buy_sell_ratio", "Binance USD-M taker buy volume per unit of taker sell volume"),
    ("price_change_24h_pct", "live price change over 24h, percent"),
    (
        "ma20_distance_atr",
        "distance of the live close from its 20-bar 1H mean in 14-bar ATR units; a large positive "
        "value is a stretched, overheating price",
    ),
    ("social_mentions_24h / social_velocity_per_hour", "distinct crowd stories naming it"),
    ("demo_mark_index_gap_bps", "gap between the Demo mark and the Demo index, bps"),
    ("demo_live_gap_bps", "gap between the Demo price and the live price, bps"),
    ("demo_live_gap_limit_bps", "the measured limit for that gap; above it the kernel exits"),
    ("demo_spread_bps", "Demo bid-ask spread, bps"),
    (
        "demo_index_move_bps_3h",
        "mean absolute 1H move of the Demo index over 3h, bps; near zero means Demo is not pricing "
        "the instrument",
    ),
    ("model_orders_today / model_orders_left_today", "your orders in this name since 00:00 UTC"),
    (
        "position_*",
        "your open position: qty (signed base units, positive long), avg_entry, mark, "
        "weight_pct (signed percent of equity), move_since_entry_pct (percent, in the position's "
        "favour), unrealized_pnl, realized_pnl and fees_paid (USDT), hours held and since the "
        "last increase, the venue stop and its distance (percent)",
    ),
    (
        "position_target_equivalent",
        "the current weight written as a target; returning it keeps the position as it is",
    ),
    (
        "position_hours_until_increase_allowed",
        "hours before an increase or a flip is allowed without a declared invalidation",
    ),
    (
        "book.*",
        "equity, P&L and fees in USDT; *_pct are percent of equity; *_bps basis points of equity; "
        "the two distance_pct values are percentage points left before the daily kill switch and "
        "before reduce-only",
    ),
    ("session.*", "hours to the weekend freeze of US-session instruments, or to their reopening"),
    ("mood.*", "Fear & Greed, from zero (extreme fear) to one hundred (extreme greed)"),
    (
        "crowd.* / story.*",
        "items, withheld items, distinct stories, copies per story and coordinated stories; per "
        "shown story, its items, distinct sources, stories per hour and span in minutes",
    ),
    ("coverage.*", "source calls made, answered, failed and not made"),
    ("trigger.*", "the value a trigger observed and the threshold it crossed"),
    ("kernel.* / reference.*", "the risk kernel's rules above, and fixed references"),
)


# ================================================================================================
# Hashing
# ================================================================================================


def _normalised_text(path: Path) -> str:
    """UTF-8 text with any BOM dropped and every line ending written ``\\n``.

    A checkout on Windows writes CRLF; hashing the raw bytes would give the same prompt two hashes.
    """
    text = path.read_bytes().decode("utf-8-sig")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def prompt_hashes() -> dict[str, str]:
    """SHA-256 of each file that decides what the model is shown, keyed by package-relative path."""
    return {
        rel: sha256_hex(_normalised_text(PACKAGE_DIR / rel).encode("utf-8"))
        for rel in HASHED_SOURCES
    }


# ================================================================================================
# Numbers and times
# ================================================================================================


def format_number(value: float) -> str:
    """At most :data:`SIGNIFICANT_DIGITS` significant digits, no exponent, no trailing zeros."""
    if not math.isfinite(value):
        raise ValueError("a non-finite value is not a fact")
    if value == 0:
        return "0"
    magnitude = math.floor(math.log10(abs(value)))
    decimals = max(0, SIGNIFICANT_DIGITS - 1 - magnitude)
    text = f"{value:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def _clean(value: float) -> float:
    """Arithmetic noise removed (``0.05 * 100`` is ``5.000000000000001``); 12 significant digits."""
    return float(f"{value:.12g}")


def _hours(delta: timedelta) -> float:
    return _clean(delta.total_seconds() / 3600)


def _ts(at: datetime) -> str:
    return at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _require_utc(now: datetime) -> None:
    if now.tzinfo is None or now.utcoffset() != timedelta(0):
        raise ValueError("now must be timezone-aware UTC")


# ================================================================================================
# The session (the weekend freeze of US-session instruments, DESIGN.md §10.3 G2)
# ================================================================================================

SessionState = Literal["open", "no_new_exposure", "preflatten", "frozen"]

_WEEKDAYS: Final = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


@dataclass(frozen=True, slots=True)
class Session:
    state: SessionState
    freeze_at: datetime
    """The start of the current freeze when frozen, otherwise the next one."""
    reopen_at: datetime


def _last_weekly(now: datetime, weekday: int, hour: int) -> datetime:
    """The most recent ``weekday`` at ``hour``:00 UTC at or before ``now``."""
    candidate = now.replace(hour=hour, minute=0, second=0, microsecond=0) - timedelta(
        days=(now.weekday() - weekday) % 7
    )
    return candidate if candidate <= now else candidate - timedelta(days=7)


def us_session(now: datetime, policy: Policy) -> Session:
    """Where ``now`` sits in the weekly freeze of US-equity and US-index legs (policy.weekend)."""
    _require_utc(now)
    rule = policy.weekend
    last_freeze = _last_weekly(now, rule.freeze_weekday, rule.freeze_hour)
    # The reopening that ends that freeze: the first reopen time after it. The policy refuses a
    # freeze and a reopening at the same weekday and hour, so this is strictly later.
    reopen = _last_weekly(last_freeze, rule.reopen_weekday, rule.reopen_hour) + timedelta(days=7)
    if last_freeze <= now < reopen:
        return Session(state="frozen", freeze_at=last_freeze, reopen_at=reopen)
    freeze = last_freeze + timedelta(days=7)
    to_freeze = freeze - now
    if to_freeze <= timedelta(minutes=rule.preflatten_minutes):
        state: SessionState = "preflatten"
    elif to_freeze <= timedelta(hours=rule.no_open_buffer_hours):
        state = "no_new_exposure"
    else:
        state = "open"
    return Session(state=state, freeze_at=freeze, reopen_at=reopen + timedelta(days=7))


_SESSION_TEXT: Final[Mapping[SessionState, str]] = {
    "open": "open",
    "no_new_exposure": "open for reductions only: new exposure is refused before the freeze",
    "preflatten": "being flattened by the kernel ahead of the weekend freeze",
    "frozen": "frozen for the weekend: the kernel holds them flat",
}


# ================================================================================================
# Facts
# ================================================================================================


def held_symbols(book: BookState, policy: Policy | None = None) -> tuple[str, ...]:
    """Symbols with a non-zero position, in universe order, then any others alphabetically."""
    held = {s for s, p in book.positions.items() if not p.is_flat}
    order = policy.symbols if policy is not None else ()
    return (*(s for s in order if s in held), *sorted(held - set(order)))


def ranked_stories(snapshot: PerceptionSnapshot, book: BookState) -> tuple[StoryCluster, ...]:
    """Crowd stories, most significant first: coordinated, then about a held name, then breadth."""
    held = set(held_symbols(book))

    def rank(c: StoryCluster) -> tuple[bool, bool, int, int, float, str]:
        return (
            not c.coordinated,
            not (held & set(c.symbols)),
            -c.distinct_sources,
            -len(c.item_ids),
            -c.last_seen.timestamp(),
            c.cluster_id,
        )

    return tuple(sorted(snapshot.crowd.clusters, key=rank))


def reference_facts(policy: Policy) -> dict[str, float]:
    """The kernel's rules and fixed references, as facts, in the units the system message uses."""
    t = policy.triggers
    b = policy.breaker
    w = policy.weekend
    values: dict[str, float] = {
        "kernel.per_name_max_pct": policy.per_name_max * 100,
        "kernel.gross_max_pct": policy.gross_max * 100,
        "kernel.risk_budget_pct": policy.mandate.risk_budget_gross * 100,
        "kernel.min_horizon_hours": policy.decision.min_horizon_hours,
        "kernel.stop_loss_pct": policy.stop_loss_pct * 100,
        "kernel.daily_kill_pct": policy.daily_kill_pct * 100,
        "kernel.max_model_orders_per_name_per_day": policy.max_rebalances_per_name_per_day,
        "kernel.min_hold_hours": policy.min_hold_hours,
        "kernel.fee_budget_daily_bps": policy.fee_budget_daily_bps,
        "kernel.fee_budget_window_bps": policy.fee_budget_window_bps,
        "kernel.max_open_spread_bps": policy.max_open_spread_bps,
        "kernel.mark_index_max_gap_pct": policy.mark_index_max_gap * 100,
        "kernel.stale_index_min_move_bps_3h": policy.stale_index_min_move_bps_3h,
        "kernel.grounding_tolerance_pct": policy.grounding_tolerance * 100,
        "kernel.weekend_no_new_exposure_hours": w.no_open_buffer_hours,
        "kernel.weekend_preflatten_minutes": w.preflatten_minutes,
        "kernel.breaker_reduce_only_drawdown_pct": b.reduce_only_drawdown * 100,
        "kernel.breaker_halt_drawdown_pct": b.halt_drawdown * 100,
        "kernel.breaker_losing_streak": b.losing_streak_reduce_only,
        "kernel.snapshot_max_age_minutes": b.snapshot_max_age_minutes,
        "kernel.quote_max_age_seconds": b.quote_max_age_seconds,
        "reference.zero": 0.0,
        "reference.long_short_parity": 1.0,
        "reference.fear_greed_neutral": 50.0,
        "reference.fear_greed_extreme_fear_at_or_below": t.fear_greed_low,
        "reference.fear_greed_extreme_greed_at_or_above": t.fear_greed_high,
        "reference.funding_z_extreme": t.funding_z_threshold,
    }
    return {key: _clean(float(value)) for key, value in values.items()}


def _finite(values: Mapping[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in values.items() if math.isfinite(v)}


def _position_facts(
    symbol: str, position: Position, book: BookState, policy: Policy, now: datetime
) -> dict[str, float]:
    """What an open position means at decision time (perception carries the recorded numbers)."""
    allowed_at = position.last_increase_at + timedelta(hours=policy.min_hold_hours)
    out: dict[str, float] = {
        f"{symbol}.position_hours_held": _hours(now - position.opened_at),
        f"{symbol}.position_hours_since_increase": _hours(now - position.last_increase_at),
        f"{symbol}.position_hours_until_increase_allowed": max(0.0, _hours(allowed_at - now)),
    }
    mark = book.marks.get(symbol)
    if mark is None or mark <= 0:
        return out
    side = 1 if position.qty > 0 else -1
    out[f"{symbol}.position_unrealized_pnl"] = _clean(
        float(position.qty * (mark - position.avg_entry))
    )
    if position.avg_entry > 0:
        out[f"{symbol}.position_move_since_entry_pct"] = _clean(
            side * float(mark / position.avg_entry - 1) * 100
        )
    if book.equity > 0:
        weight = book.weight(symbol)
        out[f"{symbol}.position_weight_pct"] = _clean(weight * 100)
        out[f"{symbol}.position_target_equivalent"] = _clean(weight / policy.mandate.per_name_max)
    if position.stop_price is not None:
        out[f"{symbol}.position_stop_distance_pct"] = _clean(
            abs(float(position.stop_price / mark - 1)) * 100
        )
    return out


def _book_facts(book: BookState, policy: Policy) -> dict[str, float]:
    """The book as percentages and budgets left (perception carries the recorded USDT figures)."""
    out: dict[str, float] = {
        "book.day_return_pct": _clean(book.day_return * 100),
        "book.drawdown_pct": _clean(book.drawdown * 100),
        "book.daily_kill_distance_pct": _clean((book.day_return + policy.daily_kill_pct) * 100),
        "book.reduce_only_distance_pct": _clean(
            (book.drawdown + policy.breaker.reduce_only_drawdown) * 100
        ),
    }
    equity = float(book.equity)
    if equity > 0:
        gross = book.gross_weight * 100
        # Both fee figures are measured against current equity, as the fee guard (G7) states them.
        fees_today_bps = float(book.fees_today) / equity * 10_000
        fees_total_bps = float(book.fees_total) / equity * 10_000
        out.update(
            {
                "book.gross_weight_pct": _clean(gross),
                "book.net_weight_pct": _clean(book.net_weight * 100),
                "book.risk_budget_left_pct": _clean(
                    max(0.0, policy.mandate.risk_budget_gross * 100 - gross)
                ),
                "book.fees_today_bps": _clean(fees_today_bps),
                "book.fees_total_bps": _clean(fees_total_bps),
                "book.fee_budget_left_today_bps": _clean(
                    max(0.0, policy.fee_budget_daily_bps - fees_today_bps)
                ),
                "book.fee_budget_left_window_bps": _clean(
                    max(0.0, policy.fee_budget_window_bps - fees_total_bps)
                ),
            }
        )
    return out


def _cycle_facts(
    snapshot: PerceptionSnapshot,
    book: BookState,
    triggers: Sequence[Trigger],
    policy: Policy,
    now: datetime,
) -> dict[str, float]:
    """Every number only a decision cycle knows. Perception's functions add the recorded ones."""
    out = reference_facts(policy)
    out.update(_book_facts(book, policy))
    session = us_session(now, policy)
    if session.state == "frozen":
        out["session.hours_to_us_reopen"] = _hours(session.reopen_at - now)
    else:
        out["session.hours_to_us_freeze"] = _hours(session.freeze_at - now)
    for number, trigger in enumerate(triggers, start=1):
        if trigger.observed is not None:
            out[f"trigger.{number}.observed"] = float(trigger.observed)
        if trigger.threshold is not None:
            out[f"trigger.{number}.threshold"] = float(trigger.threshold)
    for number, cluster in enumerate(ranked_stories(snapshot, book)[:MAX_STORIES], start=1):
        out[f"story.{number}.items"] = float(len(cluster.item_ids))
        out[f"story.{number}.distinct_sources"] = float(cluster.distinct_sources)
        if cluster.velocity_per_hour is not None:
            out[f"story.{number}.velocity_per_hour"] = _clean(cluster.velocity_per_hour)
        out[f"story.{number}.span_minutes"] = _clean(
            (cluster.last_seen - cluster.first_seen).total_seconds() / 60
        )

    for entry in policy.universe:
        out[f"{entry.symbol}.demo_live_gap_limit_bps"] = entry.demo_live_gap_p99_bps
    for symbol, used in book.rebalances_today.items():
        out[f"{symbol}.model_orders_today"] = float(used)
        out[f"{symbol}.model_orders_left_today"] = float(
            max(0, policy.max_rebalances_per_name_per_day - used)
        )
    for symbol in held_symbols(book, policy):
        out.update(_position_facts(symbol, book.positions[symbol], book, policy, now))
    return out


def decision_facts(
    snapshot: PerceptionSnapshot,
    book: BookState,
    triggers: Sequence[Trigger],
    policy: Policy,
    *,
    now: datetime,
) -> dict[str, float]:
    """Every number the user message shows, keyed as it is cited: the grounding reference.

    Three layers, each overriding the one before: the numbers only this cycle knows; perception's
    own fact functions applied to the snapshot and the book in hand (so a snapshot taken without the
    book, or a fixture, still yields the book's recorded numbers under the same keys and units);
    and the snapshot's ``facts``, which are the logged record and so win any collision.
    """
    _require_utc(now)
    facts = _finite(_cycle_facts(snapshot, book, triggers, policy, now))
    facts.update(_finite(facts_from(snapshot.features, snapshot.mood, book)))
    facts.update(_finite(crowd_facts(snapshot.crowd)))
    facts.update(_finite(coverage_facts(snapshot.source_calls)))
    facts.update(_finite(snapshot.facts))
    return facts


# ================================================================================================
# Rendering helpers
# ================================================================================================

_SAFE_LABEL_CHARS: Final = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _.:/@#&+-()',"
)


def label(text: str) -> str:
    """A short label from outside this module: as written if plainly safe, otherwise spotlighted.

    Safe means a short run of letters, digits and ordinary punctuation that states no number and in
    which the quarantine finds nothing (``crowd.quarantine.inspect``), so a label can neither carry
    an instruction nor put an unregistered figure in front of the model.
    """
    if (
        0 < len(text) <= MAX_LABEL_CHARS
        and set(text) <= _SAFE_LABEL_CHARS
        and not extract(text)
        and not inspect(text)
    ):
        return text
    return spotlight(text)


def _shown_text(prompt_text: str) -> str:
    """A screened item's ``prompt_text`` as shown: spotlighted exactly once, cut if too long.

    The screen (M3) already spotlights it. That is checked rather than trusted: text that is not
    exactly one spotlighted span is wrapped again, which also strips any marker inside it.
    """
    opening, closing = f"{SPOTLIGHT_OPEN} ", f" {SPOTLIGHT_CLOSE}"
    inner = prompt_text
    if prompt_text.startswith(opening) and prompt_text.endswith(closing):
        candidate = prompt_text[len(opening) : len(prompt_text) - len(closing)]
        if spotlight(candidate) == prompt_text:
            inner = candidate
    if len(inner) > MAX_STORY_CHARS:
        return spotlight(inner[:MAX_STORY_CHARS].rstrip()) + " (cut short)"
    return spotlight(inner)


_ASSET_CLASS_TEXT: Final[Mapping[AssetClass, str]] = {
    AssetClass.CRYPTO: "crypto, trades around the clock",
    AssetClass.US_INDEX: "US index perpetual, follows the US session",
    AssetClass.US_EQUITY: "US equity perpetual, follows the US session",
}


_ASSET_CLASS_PLURAL: Final[Mapping[AssetClass, str]] = {
    AssetClass.CRYPTO: "crypto, trades around the clock",
    AssetClass.US_INDEX: "US index perpetuals, follow the US session",
    AssetClass.US_EQUITY: "US equity perpetuals, follow the US session",
}


class _Lines:
    """The user message under construction. Fact values come only from the fact map, each written
    ``key = value``, :data:`FACTS_PER_LINE` to a line, and each at most once."""

    def __init__(self, facts: Mapping[str, float]) -> None:
        self._facts = facts
        self._lines: list[str] = []
        self.shown: set[str] = set()

    def add(self, *lines: str) -> None:
        self._lines.extend(lines)

    def blank(self) -> None:
        if self._lines and self._lines[-1] != "":
            self._lines.append("")

    def facts(self, keys: Iterable[str]) -> None:
        pairs: list[str] = []
        for key in keys:
            if key in self._facts and key not in self.shown:
                pairs.append(f"{key} = {format_number(self._facts[key])}")
                self.shown.add(key)
        for start in range(0, len(pairs), FACTS_PER_LINE):
            self._lines.append(FACT_SEPARATOR.join(pairs[start : start + FACTS_PER_LINE]))

    def text(self) -> str:
        return "\n".join(self._lines).strip() + "\n"


def _keys_with(facts: Mapping[str, float], prefix: str) -> list[str]:
    return sorted(k for k in facts if k.startswith(prefix))


# ================================================================================================
# The system message
# ================================================================================================


def _universe_text(policy: Policy) -> str:
    """The universe grouped by asset class, in policy order: one note per class, not per name."""
    groups: dict[AssetClass, list[str]] = {}
    for entry in policy.universe:
        groups.setdefault(entry.asset_class, []).append(entry.symbol)
    return "; ".join(
        f"{', '.join(symbols)} ({_ASSET_CLASS_PLURAL[asset_class]})"
        for asset_class, symbols in groups.items()
    )


def _substitutions(policy: Policy) -> dict[str, str]:
    """Template values. Every number is a ``kernel.*`` fact, in the unit the text states it in."""
    kernel = {key: format_number(value) for key, value in reference_facts(policy).items()}
    w = policy.weekend
    preflatten = w.freeze_hour * 60 - w.preflatten_minutes
    preflatten_day = (w.freeze_weekday - (1 if preflatten < 0 else 0)) % 7
    preflatten %= 24 * 60
    return {
        "per_name_max_pct": kernel["kernel.per_name_max_pct"],
        "gross_max_pct": kernel["kernel.gross_max_pct"],
        "risk_budget_pct": kernel["kernel.risk_budget_pct"],
        "min_horizon_hours": kernel["kernel.min_horizon_hours"],
        "stop_loss_pct": kernel["kernel.stop_loss_pct"],
        "daily_kill_pct": kernel["kernel.daily_kill_pct"],
        "max_orders": kernel["kernel.max_model_orders_per_name_per_day"],
        "min_hold_hours": kernel["kernel.min_hold_hours"],
        "fee_budget_daily_bps": kernel["kernel.fee_budget_daily_bps"],
        "fee_budget_window_bps": kernel["kernel.fee_budget_window_bps"],
        "max_open_spread_bps": kernel["kernel.max_open_spread_bps"],
        "mark_index_max_gap_pct": kernel["kernel.mark_index_max_gap_pct"],
        "stale_index_min_move_bps": kernel["kernel.stale_index_min_move_bps_3h"],
        "grounding_tolerance_pct": kernel["kernel.grounding_tolerance_pct"],
        "no_new_exposure_hours": kernel["kernel.weekend_no_new_exposure_hours"],
        "breaker_reduce_only_pct": kernel["kernel.breaker_reduce_only_drawdown_pct"],
        "breaker_halt_pct": kernel["kernel.breaker_halt_drawdown_pct"],
        "losing_streak": kernel["kernel.breaker_losing_streak"],
        "snapshot_max_age_minutes": kernel["kernel.snapshot_max_age_minutes"],
        "quote_max_age_seconds": kernel["kernel.quote_max_age_seconds"],
        "freeze_day": _WEEKDAYS[w.freeze_weekday],
        "freeze_time": f"{w.freeze_hour:02d}:00",
        "preflatten_day": _WEEKDAYS[preflatten_day],
        "preflatten_time": f"{preflatten // 60:02d}:{preflatten % 60:02d}",
        "reopen_day": _WEEKDAYS[w.reopen_weekday],
        "reopen_time": f"{w.reopen_hour:02d}:00",
        "universe": _universe_text(policy),
        "excluded": ", ".join(policy.excluded) if policy.excluded else "none",
        "standing_instruction": STANDING_INSTRUCTION,
        "redaction": REDACTION,
        "fact_legend": "\n".join(f"- `{name}`: {text}" for name, text in FACT_LEGEND),
    }


def render_system(policy: Policy) -> str:
    """The system message for ``policy``: role, rules, spotlight instruction, output contract."""
    values = _substitutions(policy)
    schema = Template(_normalised_text(PACKAGE_DIR / SCHEMA_TEMPLATE)).substitute(values)
    system = Template(_normalised_text(PACKAGE_DIR / SYSTEM_TEMPLATE)).substitute(
        {**values, "output_schema": schema.strip()}
    )
    return system.strip() + "\n"


# ================================================================================================
# The user message
# ================================================================================================


def _render_header(
    out: _Lines, snapshot: PerceptionSnapshot, policy: Policy, now: datetime
) -> None:
    out.add(
        "# Decision request",
        "",
        f"Decision time {_ts(now)}. Snapshot {snapshot.snapshot_id or '(unsealed)'} taken "
        f"{_ts(snapshot.taken_at)}, mode {snapshot.mode.value}, policy {policy.version}, "
        f"prompt {PROMPT_VERSION}. A fact that is absent was not measured: it is unknown, never "
        "zero.",
    )


def _render_triggers(out: _Lines, triggers: Sequence[Trigger]) -> None:
    out.blank()
    out.add("## Why you were woken", "")
    for number, trigger in enumerate(triggers, start=1):
        symbols = ", ".join(label(s) for s in trigger.symbols) if trigger.symbols else "all"
        out.add(
            f"- [{label(trigger.trigger_id)}] {trigger.kind.value}, fired {_ts(trigger.fired_at)},"
            f" instruments: {symbols}, source: {label(trigger.source)}",
            f"  detail: {spotlight(trigger.detail)}",
        )
        out.facts((f"trigger.{number}.observed", f"trigger.{number}.threshold"))


def _render_mandate(out: _Lines, policy: Policy) -> None:
    out.blank()
    out.add(
        "## Mandate",
        "",
        policy.mandate.text,
        f"Universe: {', '.join(policy.symbols)}. Never propose "
        f"{', '.join(policy.excluded) if policy.excluded else 'anything outside it'}.",
    )


def _render_book(out: _Lines, book: BookState, policy: Policy, facts: Mapping[str, float]) -> None:
    out.blank()
    out.add(
        "## Book",
        "",
        f"As of {_ts(book.as_of)}, marked at Bitget {book.mark_source.value} prices. Circuit "
        f"breaker: {book.activation.value.replace('_', '-')}.",
    )
    out.facts(BOOK_FACTS)
    out.facts(_keys_with(facts, "book."))
    held = held_symbols(book, policy)
    if held:
        sides = ", ".join(f"{s} {'long' if book.positions[s].qty > 0 else 'short'}" for s in held)
        out.add(f"Open positions: {sides}. Their position_* facts are under each instrument.")
    else:
        out.add("No open positions.")
    out.add("A name with no model_orders_today fact has had no order of yours today.")


def _render_session(out: _Lines, now: datetime, policy: Policy) -> None:
    session = us_session(now, policy)
    us_legs = [u.symbol for u in policy.universe if u.asset_class.follows_us_session]
    out.blank()
    out.add(
        "## Session",
        "",
        f"US-session instruments ({', '.join(us_legs)}) are {_SESSION_TEXT[session.state]}. "
        f"Freeze {_ts(session.freeze_at)}, reopen {_ts(session.reopen_at)}.",
    )
    out.facts(("session.hours_to_us_freeze", "session.hours_to_us_reopen"))


def _render_mood(out: _Lines, snapshot: PerceptionSnapshot, facts: Mapping[str, float]) -> None:
    mood = snapshot.mood
    out.blank()
    out.add("## Market mood", "")
    labels = []
    if mood.crypto_fear_greed_label:
        labels.append(f"crypto {label(mood.crypto_fear_greed_label)}")
    if mood.market_fear_greed_label:
        labels.append(f"US equity market {label(mood.market_fear_greed_label)}")
    agreement = {
        True: "the two crypto sources agree on the band",
        False: "the two crypto sources disagree on the band",
        None: "the two crypto sources could not be compared",
    }[mood.crypto_sources_agree]
    out.add(f"Fear & Greed: {'; '.join([*labels, agreement])}.")
    out.facts(("mood.crypto_fear_greed", "mood.market_fear_greed"))
    out.facts(_keys_with(facts, "mood."))


def _instrument_header(
    symbol: str,
    features: PositioningFeatures | None,
    book: BookState,
    policy: Policy,
    session: Session,
) -> str:
    entry = policy.entry(symbol)
    parts = [f"### {symbol}"]
    if entry is not None:
        parts.append(_ASSET_CLASS_TEXT[entry.asset_class])
        if entry.asset_class.follows_us_session:
            parts.append(f"session {session.state.replace('_', ' ')}")
    position = book.positions.get(symbol)
    if position is not None and not position.is_flat:
        parts.append(f"position {'long' if position.qty > 0 else 'short'}")
    else:
        parts.append("no position")
    if features is None:
        parts.append("positioning not measured")
    else:
        if features.coordinated_cluster:
            parts.append("named by a coordinated story")
        if features.next_earnings_at is not None:
            parts.append(f"next earnings {_ts(features.next_earnings_at)}")
    return " · ".join(parts)


def _render_instruments(
    out: _Lines,
    snapshot: PerceptionSnapshot,
    book: BookState,
    policy: Policy,
    now: datetime,
    facts: Mapping[str, float],
) -> None:
    session = us_session(now, policy)
    out.blank()
    out.add(
        "## Instruments",
        "",
        "Positioning is read from Bitget's live market (the real crowd); the venue is Bitget UTA "
        "Demo.",
    )
    for symbol in policy.symbols:
        out.blank()
        out.add(_instrument_header(symbol, snapshot.features.get(symbol), book, policy, session))
        out.facts(f"{symbol}.{name}" for name in (*FEATURE_FACTS, *INSTRUMENT_FACTS))
        out.facts(f"{symbol}.{name}" for name in POSITION_FACTS)
        out.facts(_keys_with(facts, f"{symbol}."))


def _render_crowd(
    out: _Lines,
    snapshot: PerceptionSnapshot,
    book: BookState,
    facts: Mapping[str, float],
) -> None:
    stories = ranked_stories(snapshot, book)
    out.blank()
    out.add(
        "## Crowd",
        "",
        "Distinct stories from X, Reddit and news, most significant first; copies of one story "
        "are counted once. A coordinated story was posted by several distinct sources inside a "
        "short window: evidence of promotion, not of information.",
    )
    out.facts(CROWD_FACTS)
    out.facts(_keys_with(facts, "crowd."))
    by_id: dict[str, ScreenedItem] = {}
    for screened in snapshot.text:
        by_id.setdefault(screened.item.item_id, screened)
    for number, cluster in enumerate(stories[:MAX_STORIES], start=1):
        members = [by_id[i] for i in cluster.item_ids if i in by_id]
        shown = sorted(
            (m for m in members if not m.withheld),
            key=lambda m: (m.item.published_at, m.item.item_id),
        )
        named = ", ".join(cluster.symbols) if cluster.symbols else "no universe instrument"
        channels = ", ".join(sorted({m.item.channel for m in members})) or "unknown channel"
        out.blank()
        out.add(
            f"### story.{number}: {'coordinated' if cluster.coordinated else 'organic'} "
            f"[{label(cluster.cluster_id)}] · names {named} · {channels} · "
            f"{_ts(cluster.first_seen)} to {_ts(cluster.last_seen)}"
        )
        out.facts(f"story.{number}.{name}" for name in STORY_FACTS)
        ids = ", ".join(label(i) for i in cluster.item_ids[:MAX_STORY_IDS])
        more = ", and more" if len(cluster.item_ids) > MAX_STORY_IDS else ""
        out.add(f"sources: {spotlight(', '.join(cluster.sources))}", f"item ids: {ids}{more}")
        if shown:
            out.add(_shown_text(shown[0].prompt_text))
        else:
            out.add("(no screened text of this story reached the snapshot)")
    if len(stories) > MAX_STORIES:
        out.add("", "Less significant stories are not shown.")
    withheld = sorted(
        (s for s in snapshot.text if s.withheld),
        key=lambda s: (s.item.published_at, s.item.item_id),
    )
    if withheld:
        out.blank()
        out.add(
            "### Withheld items",
            "",
            f"Each of these was replaced by {REDACTION!r}: it carried a prompt injection. Its text "
            "and its account are not shown.",
        )
        for screened in withheld[:MAX_WITHHELD_LISTED]:
            out.add(
                f"- [{label(screened.item.item_id)}] {screened.item.channel}, published "
                f"{_ts(screened.item.published_at)}"
            )


def _render_calendar(out: _Lines, snapshot: PerceptionSnapshot) -> None:
    out.blank()
    out.add("## Calendar", "")
    if not snapshot.calendar:
        out.add("No calendar items in this snapshot.")
        return
    far = datetime.max.replace(tzinfo=UTC)
    items = sorted(snapshot.calendar, key=lambda c: (c.at or far, c.symbol or "", c.title))
    for item in items[:MAX_CALENDAR_ITEMS]:
        when = _ts(item.at) if item.at is not None else "time not given"
        out.add(
            f"- {item.symbol or 'market-wide'}, {item.kind}, {when}: {spotlight(item.title)}"
            f" (source {label(item.source)})"
        )
    if len(items) > MAX_CALENDAR_ITEMS:
        out.add("Later calendar items are not shown.")


def _render_coverage(out: _Lines, snapshot: PerceptionSnapshot, facts: Mapping[str, float]) -> None:
    out.blank()
    out.add(
        "## Source coverage",
        "",
        "A source that did not answer is missing data, not a neutral reading.",
    )
    out.facts(_keys_with(facts, "coverage."))
    by_health: dict[SourceHealth, set[str]] = {}
    for call in snapshot.source_calls:
        by_health.setdefault(call.health, set()).add(label(call.source))
    for health in SourceHealth:
        if health not in by_health:
            continue
        if health is SourceHealth.OK:
            # Named only when something is wrong: the answering sources are a count, because
            # listing ~170 spotlighted names cost ~7 KB of every prompt and the daily token budget
            # projects one token per byte (llm/budget.py).
            count = len(by_health[health])
            out.add(f"- ok: {count} source{'' if count == 1 else 's'} answered (not listed)")
        else:
            out.add(f"- {health.value}: {', '.join(sorted(by_health[health]))}")


def _render_rest(out: _Lines, facts: Mapping[str, float]) -> None:
    reference = ("kernel.", "reference.")
    rest = [k for k in facts if k not in out.shown and not k.startswith(reference)]
    if rest:
        out.blank()
        out.add("## Other facts", "")
        out.facts(sorted(rest))
    out.blank()
    out.add(
        "## Reference values",
        "",
        "The risk kernel's rules, as stated in the system message, and fixed references.",
    )
    out.facts(k for k in facts if k.startswith(reference))


def _render_answer(out: _Lines, book: BookState, policy: Policy) -> None:
    held = held_symbols(book, policy)
    out.blank()
    out.add("## Your answer", "")
    if held:
        out.add(f"You hold {', '.join(held)}: each must appear in targets (zero closes it).")
    else:
        out.add(
            "You hold nothing: answer act with at least one non-zero target, or "
            "flat_with_reasons with your reasons."
        )
    out.add("Return the JSON object described in the system message, and nothing else.")


def render_user(
    snapshot: PerceptionSnapshot,
    book: BookState,
    triggers: Sequence[Trigger],
    policy: Policy,
    *,
    now: datetime,
) -> str:
    """The user message. Every fact of :func:`decision_facts` appears in it exactly once."""
    facts = decision_facts(snapshot, book, triggers, policy, now=now)
    out = _Lines(facts)
    _render_header(out, snapshot, policy, now)
    _render_triggers(out, triggers)
    _render_mandate(out, policy)
    _render_book(out, book, policy, facts)
    _render_session(out, now, policy)
    _render_mood(out, snapshot, facts)
    _render_instruments(out, snapshot, book, policy, now, facts)
    _render_crowd(out, snapshot, book, facts)
    _render_calendar(out, snapshot)
    _render_coverage(out, snapshot, facts)
    _render_rest(out, facts)
    _render_answer(out, book, policy)
    missing = set(facts) - out.shown
    if missing:  # pragma: no cover - _render_rest shows every remaining key
        raise RuntimeError(f"facts left unrendered: {sorted(missing)[:5]}")
    return out.text()


def render_messages(
    snapshot: PerceptionSnapshot,
    book: BookState,
    triggers: Sequence[Trigger],
    policy: Policy,
    *,
    now: datetime,
) -> list[ChatMessage]:
    """The two messages of one decision call: system, then user."""
    if not triggers:
        raise ValueError("a decision cycle needs at least one admitted trigger")
    if snapshot.policy_version != policy.version:
        raise ValueError(
            f"the snapshot was taken under {snapshot.policy_version!r}, "
            f"not the policy in force ({policy.version!r})"
        )
    return [
        ChatMessage(role="system", content=render_system(policy)),
        ChatMessage(role="user", content=render_user(snapshot, book, triggers, policy, now=now)),
    ]


__all__ = [
    "BOOK_FACTS",
    "CROWD_FACTS",
    "FACTS_PER_LINE",
    "FACT_LEGEND",
    "FACT_SEPARATOR",
    "FEATURE_FACTS",
    "HASHED_SOURCES",
    "HEARTBEAT_KINDS",
    "INSTRUMENT_FACTS",
    "MAX_CALENDAR_ITEMS",
    "MAX_STORIES",
    "MAX_STORY_CHARS",
    "POSITION_FACTS",
    "PROMPT_VERSION",
    "STORY_FACTS",
    "Session",
    "decision_facts",
    "format_number",
    "held_symbols",
    "label",
    "prompt_hashes",
    "ranked_stories",
    "reference_facts",
    "render_messages",
    "render_system",
    "render_user",
    "us_session",
]
