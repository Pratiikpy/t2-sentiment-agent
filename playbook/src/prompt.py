# ruff: noqa: E501 - each prompt paragraph is one line: a line break inside one would reach the model
"""The decision prompt: one system message and one user message, every number a citable fact.

Ported in spirit from the primary's ``sentiment_agent.decision.prompt`` and
``decision/prompts/system_v1.md`` / ``output_schema_v1.md``. The rules the model is told are
rendered from ``policy_v1`` (generated from ``POLICY_V1``), so the prompt cannot state a limit the
kernel does not enforce. The output contract is the primary's, field for field, so a replica
decision and a primary decision are the same record type.

Two properties carried over, each tested:

* **Every number shown is a fact, and every fact is shown.** :func:`build_facts` is the one map the
  user message's numbers come from, written ``key = value`` with at most six significant digits
  and never in exponent notation; it is also the reference G9 resolves the model's numbers
  against. So a number the model copies resolves, and a number it invents does not.
* **No third-party text.** Crowd activity enters as counts and ranks only (see :mod:`.crowd`).

The replica's addition: a *compact* rendering (core facts only) used when the runner refuses the
full prompt as too large (``LLMInputError``), because the runner's prompt-size cap is fixed by the
deployment and not published.
"""

import math
from collections.abc import Mapping, Sequence
from decimal import Decimal

from . import policy_v1 as policy

SIGNIFICANT_DIGITS = 6
FACTS_PER_LINE = 3

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

CORE_SYMBOL_FACTS = (
    "mark",
    "funding_rate",
    "funding_z",
    "price_change_24h_pct",
    "ma20_distance_atr",
    "oi_change_24h_pct",
    "retail_long_short_ratio",
)

FACT_LEGEND: dict[str, str] = {
    "mark": "mark price (live Bitget, data layer)",
    "index": "index price",
    "last": "last traded price",
    "mark_index_gap_bps": "|mark / index - 1| in bps",
    "spread_bps": "ask minus bid over mid, in bps",
    "funding_rate": "live funding rate per interval, a fraction",
    "funding_z": "funding z-score against the last settlements",
    "open_interest": "open interest, in coin",
    "oi_change_1h_pct": "open interest change over 1h, percent",
    "oi_change_24h_pct": "open interest change over 24h, percent",
    "oi_jump_threshold_pct": "trailing 99th percentile of |1h open interest change|, percent",
    "retail_long_short_ratio": "accounts long over accounts short",
    "top_trader_long_short_ratio": "top traders' position long/short ratio",
    "price_change_24h_pct": "price change over 24h, percent",
    "ma20_distance_atr": "last close minus its 20-bar mean, in 14-bar ATR units (positive is stretched up)",
    "move_bps_3h": "mean absolute 1h move over the last 3h, bps",
    "news_stories_24h": "distinct news stories naming the instrument, 24h",
    "coordinated_stories": "news stories carried by several sources inside a short window (promotion)",
    "forum_mentions_24h": "forum mentions (Reddit and 4chan trackers), 24h",
    "forum_mentions_change_pct": "forum mentions versus the prior 24h, percent",
    "forum_rank": "rank by forum mentions (1 is most mentioned)",
    "hours_to_earnings": "hours until the next earnings report",
    "insider_filings_since_open": "insider filings since the position opened",
    "venue_gap_limit_bps": "venue-versus-data-layer price gap that forces an exit, bps",
    "model_orders_today": "orders you caused in this name today (UTC)",
    "model_orders_left_today": "orders you may still cause in this name today",
    "position_qty": "position quantity, signed (negative is short)",
    "position_avg_entry": "average entry price",
    "position_weight_pct": "position as percent of equity, signed",
    "position_target_equivalent": "the target that keeps this position unchanged",
    "position_move_since_entry_pct": "mark versus entry, percent, signed for the position's side",
    "position_unrealized_pnl": "unrealized profit and loss, USDT",
    "position_hours_held": "hours since the position opened",
    "position_hours_since_increase": "hours since the last increase",
    "position_hours_until_increase_allowed": "hours until an increase or flip is allowed without a declared invalidation",
    "position_stop_price": "the preset venue stop",
}


def fmt(value: float) -> str:
    """At most six significant digits, never in exponent notation, no trailing zeros."""
    if not math.isfinite(value):
        raise ValueError("a fact must be finite")
    if value == 0:
        return "0"
    magnitude = math.floor(math.log10(abs(value)))
    decimals = SIGNIFICANT_DIGITS - 1 - magnitude
    rounded = round(value, decimals)
    if decimals <= 0:
        text = format(Decimal(int(rounded)), "f")
    else:
        text = f"{rounded:.{decimals}f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def _put(facts: dict[str, float], key: str, value: float | Decimal | int | None) -> None:
    if value is None or isinstance(value, bool):
        return
    number = float(value)
    if math.isfinite(number):
        facts[key] = float(fmt(number))


def kernel_facts() -> dict[str, float]:
    facts: dict[str, float] = {}
    for key, value in (
        ("kernel.per_name_max_pct", policy.PER_NAME_MAX * 100),
        ("kernel.gross_max_pct", policy.GROSS_MAX * 100),
        ("kernel.risk_budget_pct", policy.MANDATE_RISK_BUDGET_GROSS * 100),
        ("kernel.stop_loss_pct", policy.STOP_LOSS_PCT * 100),
        ("kernel.daily_kill_pct", policy.DAILY_KILL_PCT * 100),
        ("kernel.max_orders_per_name_per_day", policy.MAX_REBALANCES_PER_NAME_PER_DAY),
        ("kernel.min_hold_hours", policy.MIN_HOLD_HOURS),
        ("kernel.fee_budget_daily_bps", policy.FEE_BUDGET_DAILY_BPS),
        ("kernel.fee_budget_window_bps", policy.FEE_BUDGET_WINDOW_BPS),
        ("kernel.max_open_spread_bps", policy.MAX_OPEN_SPREAD_BPS),
        ("kernel.mark_index_max_gap_pct", policy.MARK_INDEX_MAX_GAP * 100),
        ("kernel.stale_min_move_bps", policy.STALE_INDEX_MIN_MOVE_BPS_3H),
        ("kernel.breaker_reduce_only_pct", policy.BREAKER_REDUCE_ONLY_DRAWDOWN * 100),
        ("kernel.breaker_halt_pct", policy.BREAKER_HALT_DRAWDOWN * 100),
        ("kernel.losing_streak", policy.BREAKER_LOSING_STREAK_REDUCE_ONLY),
        ("kernel.min_horizon_hours", policy.DECISION_MIN_HORIZON_HOURS),
        ("kernel.grounding_tolerance_pct", policy.GROUNDING_TOLERANCE * 100),
        ("kernel.funding_z_threshold", policy.TRIGGER_FUNDING_Z_THRESHOLD),
        ("mood.fear_greed_low", policy.TRIGGER_FEAR_GREED_LOW),
        ("mood.fear_greed_high", policy.TRIGGER_FEAR_GREED_HIGH),
        ("reference.zero", 0),
        ("reference.long_short_parity", 1),
    ):
        _put(facts, key, value)
    return facts


def build_facts(
    *,
    features: Mapping[str, Mapping[str, float]],
    mood: Mapping[str, float],
    book: Mapping[str, float],
    positions: Mapping[str, Mapping[str, float]],
    crowd: Mapping[str, float],
    session: Mapping[str, float],
    configured: Sequence[str],
    compact: bool = False,
) -> dict[str, float]:
    """Every number the model sees, keyed as it is shown."""
    facts = kernel_facts()
    for name, value in mood.items():
        _put(facts, f"mood.{name}", value)
    for name, value in session.items():
        _put(facts, f"session.{name}", value)
    for name, value in book.items():
        _put(facts, f"book.{name}", value)
    if not compact:
        for name, value in crowd.items():
            _put(facts, f"crowd.{name}", value)
    for symbol in configured:
        values = features.get(symbol, {})
        for name, value in values.items():
            if compact and name not in CORE_SYMBOL_FACTS:
                continue
            _put(facts, f"{symbol}.{name}", value)
        for name, value in positions.get(symbol, {}).items():
            _put(facts, f"{symbol}.{name}", value)
    return facts


def _schema() -> str:
    per_name = fmt(policy.PER_NAME_MAX * 100)
    budget = fmt(policy.MANDATE_RISK_BUDGET_GROSS * 100)
    horizon = policy.DECISION_MIN_HORIZON_HOURS
    return f"""## Output

Return exactly one JSON object and nothing else: no prose before or after it, no markdown fences. Its shape, with a description in angle brackets where a value goes:

{{"stance": <"act" | "hold" | "flat_with_reasons">,
 "targets": [{{"symbol": <a universe symbol, exactly as written>,
              "target": <number from minus one to one>,
              "thesis": <string>,
              "invalidation": <string>,
              "horizon_hours": <integer>,
              "crowd_belief": <string>,
              "our_view": <string>,
              "confidence": <number from zero to one>,
              "evidence": [<fact key>, ...],
              "invalidation_triggered": <true | false>,
              "invalidation_evidence": <string or null>}}, ...],
 "rejected_alternatives": [{{"action": <string>, "reason": <string>}}, ...],
 "mandate_response": <string>,
 "flat_reasons": [<string>, ...],
 "summary": <string>}}

- `stance`: "act" moves the book to your targets. "hold" keeps every open position exactly as it is: list each held symbol with a target on the same side, and open nothing new. "flat_with_reasons" holds nothing: every target is zero and `flat_reasons` says why. With no open positions, answer "act" with at least one non-zero target, or "flat_with_reasons".
- `targets`: one entry per symbol you address, each symbol at most once. Every symbol you hold must appear; a target of zero closes it. A symbol you leave out stays flat.
- `target`: positive is long, negative is short, zero is flat. The weight is the target times {per_name}% of equity. To keep a position unchanged, return its position_target_equivalent.
- `thesis`: why this position, from the facts. `invalidation`: the observable fact that would prove it wrong. `horizon_hours`: an integer of at least {horizon}.
- `crowd_belief`: what the crowd believes, from positioning and activity. `our_view`: what we do about it, and why that agrees with or departs from the crowd.
- `confidence`: how likely the thesis is to play out over the horizon. `evidence`: the fact keys the thesis relies on.
- `invalidation_triggered`: true only for a position you hold whose previously stated invalidation has now fired, with `invalidation_evidence` naming the fact that shows it. It permits a flip inside the hold window; it never permits adding to the same side. Otherwise false, with `invalidation_evidence` null.
- `rejected_alternatives`: the serious alternatives you considered and turned down, each with its reason.
- `mandate_response`: how much of the {budget}% risk budget you deploy and why, or why you decline it.
- `flat_reasons`: at least one reason when the stance is "flat_with_reasons"; otherwise an empty list.
- `summary`: one or two sentences for the public decision card.

Unknown fields are rejected. An answer that breaks these rules is returned to you with the reasons, and a cycle that ends without a valid answer flattens the book."""


def system_prompt(configured: Sequence[str]) -> str:
    """The standing message: role, venue, job, the kernel's rules as facts, the number rules, the
    fact legend and the output contract. Every number comes from ``policy_v1``."""
    freeze_day = WEEKDAYS[policy.WEEKEND_FREEZE_WEEKDAY]
    reopen_day = WEEKDAYS[policy.WEEKEND_REOPEN_WEEKDAY]
    preflatten_minutes = policy.WEEKEND_FREEZE_HOUR * 60 - policy.WEEKEND_PREFLATTEN_MINUTES
    legend = "\n".join(f"- `{name}`: {meaning}" for name, meaning in FACT_LEGEND.items())
    return f"""You are the portfolio decision-maker of an autonomous market-sentiment trading agent on Bitget. You alone decide what the book holds. No human approves your answer and no rule-based strategy stands behind you: if you do not return a valid decision, the book is flattened.

## What you manage

- A paper book traded as a GetAgent Playbook on Bitget. This Playbook is the published secondary record of an agent whose primary record trades on Bitget UTA Demo through Agent Hub; both obey one pre-registered policy ({policy.POLICY_VERSION}). Wherever your mandate says Demo venue, read the venue this Playbook trades on.
- A fixed universe: {", ".join(configured)}. Never propose anything else. Excluded because their paper prices are unreliable: {", ".join(policy.EXCLUDED)}.
- One decision covers the whole book. The request gives you the book, every instrument's facts, the market mood, crowd activity as counts, the calendar, which sources answered, and the events that woke you.

## The job: what does the crowd believe, and is its positioning overstretched?

- Crowded optimism shows as funding far above its norm, open interest rising fast, retail accounts skewed long, a price stretched far above its mean, and a surge of repetitive or coordinated stories. That is where you fade, hedge, or cut exposure that depends on the crowd staying right. Crowded pessimism is the mirror image.
- Tops come before the crowd sees them: reduce before overheating, not after it.
- A coordinated story (several sources carrying near-identical text inside a short window) is evidence of promotion, not of information.
- Positioning is the backbone; crowd activity is supporting colour. You see how much the crowd talks, never what it says: no third-party text is shown to you.
- Every round trip pays the taker fee twice. A thesis that cannot clear that cost over its horizon is not an edge.
- No edge is a normal state. Flat, with written reasons, is a first-class answer, counted and published like any trade.

## Rules you cannot change

A risk kernel checks your answer before anything reaches the venue. It can shrink, refuse or close what you ask for; it can never add to it. Each number here is also a `kernel.*` fact:

- Size: a target of one is {fmt(policy.PER_NAME_MAX * 100)}% of equity long in one name, minus one the same short. Gross exposure is capped at {fmt(policy.GROSS_MAX * 100)}% of equity, and your risk budget is {fmt(policy.MANDATE_RISK_BUDGET_GROSS * 100)}% gross.
- Horizon: every target states a horizon of at least {policy.DECISION_MIN_HORIZON_HOURS} hours.
- Stops: every opening or increase carries a venue stop {fmt(policy.STOP_LOSS_PCT * 100)}% from entry.
- Daily kill: a book down {fmt(policy.DAILY_KILL_PCT * 100)}% from its 00:00 UTC equity is flattened and halted until the next UTC day.
- Turnover: at most {policy.MAX_REBALANCES_PER_NAME_PER_DAY} orders of yours per name per UTC day. No increase and no side flip within {policy.MIN_HOLD_HOURS} hours of the last increase, unless you declare that the position's stated invalidation has fired. Reductions and closes are never blocked.
- Fees: market (taker) orders only. No new exposure once fees paid today reach {fmt(policy.FEE_BUDGET_DAILY_BPS)} bps of equity, or {fmt(policy.FEE_BUDGET_WINDOW_BPS)} bps over the whole run.
- Spread: no opening while the spread is wider than {fmt(policy.MAX_OPEN_SPREAD_BPS)} bps.
- Venue integrity: no new exposure in, and an exit from, an instrument whose mark departs more than {fmt(policy.MARK_INDEX_MAX_GAP * 100)}% from its index, whose venue price departs from the data layer's by more than its measured limit, or whose price has moved less than {fmt(policy.STALE_INDEX_MIN_MOVE_BPS_3H)} bps an hour over the last 3h while its session should be open.
- Weekend: US-equity and US-index instruments are flattened from {freeze_day} {preflatten_minutes // 60:02d}:{preflatten_minutes % 60:02d} UTC, held flat from {freeze_day} {policy.WEEKEND_FREEZE_HOUR:02d}:00 UTC to {reopen_day} {policy.WEEKEND_REOPEN_HOUR:02d}:00 UTC, and cannot be opened in the {fmt(policy.WEEKEND_NO_OPEN_BUFFER_HOURS)} hours before the freeze. Crypto trades around the clock.
- Circuit breaker: a drawdown of {fmt(policy.BREAKER_REDUCE_ONLY_DRAWDOWN * 100)}% from peak equity makes the book reduce-only, and {fmt(policy.BREAKER_HALT_DRAWDOWN * 100)}% halts it; {policy.BREAKER_LOSING_STREAK_REDUCE_ONLY} losing trades in a row make it reduce-only. Stale data cannot open a position: a snapshot older than {policy.BREAKER_SNAPSHOT_MAX_AGE_MINUTES} minutes or a quote older than {policy.BREAKER_QUOTE_MAX_AGE_SECONDS} seconds.
- Grounding: every number you write in thesis, invalidation or our_view is checked against the facts you were given, within {fmt(policy.GROUNDING_TOLERANCE * 100)}%. A target whose text states a number that is not among them may not add exposure.

## Numbers

- Every number in thesis, invalidation and our_view must be a value the request shows as `key = value`. Copy it as shown. You may write a fraction as a percent or in basis points, and you may drop the sign when a word carries the direction.
- Do not compute new numbers: no differences, ratios, averages, projected prices or price targets. Compare facts in words.
- State invalidation through facts, for example "BTCUSDT.funding_z back below reference.zero", never an invented price level.
- Cite the facts you rely on in evidence, keys exactly as written.
- A fact that is absent was not measured. It is unknown, never zero.

## Facts you will see

Every fact is written `key = value`. An instrument's facts are keyed `SYMBOL.name`; `kernel.*` are the rules above, `mood.*` the Fear & Greed readings, `book.*` the book, `session.*` the clock, `crowd.*` the activity totals. The instrument names mean:

{legend}

{_schema()}
"""


def _lines(facts: Mapping[str, float], keys: Sequence[str]) -> list[str]:
    rendered = [f"{key} = {fmt(facts[key])}" for key in keys if key in facts]
    return [
        "; ".join(rendered[i : i + FACTS_PER_LINE]) for i in range(0, len(rendered), FACTS_PER_LINE)
    ]


def user_prompt(
    *,
    triggers: Sequence[Mapping[str, object]],
    facts: Mapping[str, float],
    configured: Sequence[str],
    held: Mapping[str, str],
    session_text: str,
    sources_text: str,
) -> str:
    """The request: what woke the model, the mandate, the book, the session, the mood and every
    instrument's facts. ``held`` maps each held symbol to ``long`` or ``short``."""
    parts: list[str] = ["## What woke you"]
    for trigger in triggers:
        parts.append(f"- {trigger.get('kind')}: {trigger.get('detail')}")
    parts += ["", "## Mandate", policy.MANDATE_TEXT, "", "## Book"]
    if held:
        parts.append("Held: " + ", ".join(f"{s} {side}" for s, side in sorted(held.items())) + ".")
    else:
        parts.append("Held: nothing. The book is flat.")
    parts += _lines(facts, sorted(k for k in facts if k.startswith("book.")))
    parts += ["", "## Session", session_text]
    parts += _lines(facts, sorted(k for k in facts if k.startswith("session.")))
    parts += ["", "## Mood"]
    parts += _lines(facts, sorted(k for k in facts if k.startswith("mood.")))
    crowd = sorted(k for k in facts if k.startswith("crowd."))
    if crowd:
        parts += ["", "## Crowd activity (counts only)"]
        parts += _lines(facts, crowd)
    parts += ["", "## Instruments"]
    for symbol in configured:
        keys = sorted(k for k in facts if k.startswith(f"{symbol}."))
        parts.append(f"### {symbol}" + (" (held)" if symbol in held else ""))
        parts += _lines(facts, keys) if keys else ["no facts measured"]
    parts += ["", "## Rules as facts"]
    parts += _lines(facts, sorted(k for k in facts if k.startswith(("kernel.", "reference."))))
    parts += ["", "## Sources", sources_text]
    parts += ["", "Answer with the JSON object only."]
    return "\n".join(parts)
