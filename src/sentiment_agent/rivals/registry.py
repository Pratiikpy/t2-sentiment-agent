"""The rival roster: every agent that competes with ours for the Market Sentiment sub-theme.

The handbook's Market Sentiment Agent turns what the crowd says and does into positions: "FOMO
detection -> contrarian hedge; sentiment top detection; reduce before overheating" (handbook:233).
A rival here is an agent that does exactly that, social, forum or news sentiment into positions,
and not a narrower tool that only scores text. Each one is run on the same recorded
:class:`~sentiment_agent.types.PerceptionSnapshot` sequence as our agent and marked by the same
:class:`~sentiment_agent.analysis.armsim.ArmSimulator`, under the venue guards only
(:data:`~sentiment_agent.types.VENUE_GUARDS`), so a comparison isolates who decided rather than
which rules applied. ``RIVALS.md`` records, per rival, what was read, its licence, what was taken
and the measured result, win or loss.

The roster
----------
Open-source rivals, each the strongest published form of its approach:

* :data:`LEXICON_FOLLOW`, :data:`LEXICON_FADE`: a lexicon sentiment trader on VADER's scoring rules
  (``keyword_arm.py``), trading with the crowd's tone and against it.
* :data:`FINBERT_FOLLOW`, :data:`FINBERT_FADE`: the same trading rule on ProsusAI/finBERT's
  sentence scores (``finbert_arm.py``).
* :data:`TRADINGAGENTS`: TauricResearch/TradingAgents' sentiment analyst and trader chain on Qwen
  (``tradingagents_arm.py``).

Season-2 entries with runnable code that turn sentiment into positions, found in the sweep of
2026-09-24 and rebuilt here (described by what they do, not by who built them; ``RIVALS.md``):

* :data:`SEASON2_CONFLUENCE` (:class:`FearGreedConfluenceArm`): Fear & Greed extremes confirmed by
  the long/short account split, contrarian. The entry ships no licence file, so it is rebuilt from
  its described behaviour; no code is copied.
* :data:`SEASON2_FUSION` (:class:`SentimentFusionArm`): a confidence-weighted signal fusion with a
  regime router, restricted to its sentiment and news signals. MIT, ported with attribution.
* :data:`SEASON2_QWEN_HEADLINES` (:class:`QwenHeadlineTraderArm`): Qwen reads headline sentiment
  and price momentum and trades a long-only book. MIT, ported with attribution.

Rules every rival obeys
-----------------------
These live here, once, so that no arm can be advantaged by its own plumbing.

* **What a rival reads.** The raw text of every item in the snapshot, including items our
  quarantine withheld: none of the rivals has a quarantine upstream, and the comparison is of the
  rivals as they are. (The red team, M14, measures what that costs them.) Only text published at or
  before the snapshot instant is read, and a text names the universe symbols its collector tagged
  plus those :func:`~sentiment_agent.crowd.novelty.symbols_mentioned` finds in it.
* **Its own book.** A rival is shown its own positions, rebuilt from its own previous targets by
  the harness, never ours.
* **What a returned map means.** The whole book: a symbol absent from it is flat. An arm that
  keeps a position returns its current weight.
* **Size.** Weights are fractions of equity, at most the policy's per-name cap each, and scaled
  down together when their gross exceeds the mandate's risk budget (:func:`cap_weights`). The
  kernel's G3 applies the same caps again in the simulator.
"""

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Final, Literal, Protocol

from sentiment_agent.crowd.novelty import symbols_mentioned
from sentiment_agent.decision.contract import extract_json_object
from sentiment_agent.llm.budget import projected_tokens
from sentiment_agent.llm.client import QwenError
from sentiment_agent.types import (
    VENUE_GUARDS,
    ArmKind,
    ArmSpec,
    AssetClass,
    BookState,
    ChatMessage,
    ChatModel,
    Completion,
    GuardId,
    PerceptionSnapshot,
    Policy,
    TextChannel,
    TextItem,
    Thinking,
)

# ================================================================================================
# The protocol every rival satisfies
# ================================================================================================


class RivalArm(Protocol):
    """An agent that turns one snapshot, and its own book, into target weights.

    ``targets`` returns the whole book as signed fractions of equity; a symbol it omits is flat.
    """

    spec: ArmSpec

    def targets(self, snapshot: PerceptionSnapshot, book: BookState) -> dict[str, float]: ...


# ================================================================================================
# The roster
# ================================================================================================

LEXICON_FOLLOW: Final = "rival_lexicon_follow"
LEXICON_FADE: Final = "rival_lexicon_fade"
FINBERT_FOLLOW: Final = "rival_finbert_follow"
FINBERT_FADE: Final = "rival_finbert_fade"
TRADINGAGENTS: Final = "rival_tradingagents_sentiment_trader"
SEASON2_CONFLUENCE: Final = "rival_s2_fear_greed_confluence"
SEASON2_FUSION: Final = "rival_s2_sentiment_fusion"
SEASON2_QWEN_HEADLINES: Final = "rival_s2_qwen_headline_trader"

RIVAL_GUARDS: Final[tuple[GuardId, ...]] = tuple(sorted(VENUE_GUARDS))
"""Every rival runs under the venue and mandate guards and nothing that judges the decision-maker
(no turnover, fee-budget, grounding or breaker guard), exactly as the baselines do."""

VADER_PROVENANCE: Final = " ".join(
    (
        "cjhutto/vaderSentiment (MIT, Copyright (c) 2016 C.J. Hutto) at commit",
        "44fc044cd877310ee8278a0eadf34bcd50d41d06: scoring rules",
        "vaderSentiment/vaderSentiment.py:26-509 ported, general-word valences copied from",
        "vaderSentiment/vader_lexicon.txt; market vocabulary added in rivals/keyword_arm.py",
    )
)
FINBERT_PROVENANCE: Final = " ".join(
    (
        "ProsusAI/finBERT (Apache-2.0) at commit 44995e0c5870c4ab37a189d756550654ae87cdf0:",
        "sentence score P(positive) - P(negative) as finbert/finbert.py:625, 64-token sentences",
        "as finbert/finbert.py:30 and :613; model weights ProsusAI/finbert revision",
        "4556d13015211d73dccd3fdd39d39232506f3e43 loaded through transformers at evaluation",
        "time (optional [rivals] extra), never vendored",
    )
)
TRADINGAGENTS_PROVENANCE: Final = (
    "TauricResearch/TradingAgents (Apache-2.0) at commit be952b8eccb49720509af544c6675233bc1f10d0: "
    "sentiment analyst prompt tradingagents/agents/analysts/sentiment_analyst.py:88-192, trader "
    "prompt tradingagents/agents/trader/trader.py:46-74, output schemas "
    "tradingagents/agents/schemas.py:137-204 and :285-368, adapted and attributed in "
    "rivals/tradingagents_arm.py; licence text in rivals/RIVALS.md"
)
SEASON2_CONFLUENCE_PROVENANCE: Final = " ".join(
    (
        "Season-2 Track 2 entry A (public repository, commit",
        "bb4804bff99e01e545a61e0f0ffe63b42117cc44), strategy.js:22-37, :77-80, :410-470,",
        ":551-565 and :667-671. No licence file: rebuilt from the described behaviour in",
        "rivals/registry.py,",
        "no code copied",
    )
)
SEASON2_FUSION_PROVENANCE: Final = " ".join(
    (
        "Season-2 Track 2 entry B (MIT, Copyright (c) 2026 its author; public repository, commit",
        "ee3b4d65dc116f5c86c8fe972549056d20e16e64): backend/src/signals/mcpSignals.ts:54-97 and",
        ":173-211, backend/src/signals/signalFusion.ts:9-83,",
        "backend/src/agents/strategyRouter.ts:8-150, backend/src/agents/agentCycle.ts:62-154,",
        "ported in rivals/registry.py",
    )
)
SEASON2_QWEN_HEADLINES_PROVENANCE: Final = (
    "Season-2 entry C (MIT; public repository, commit b1fa7f7b6ac8f1ef2c68ef7592a073abbaaa5d33): "
    "lib/agent/prompt.ts:18-87, lib/data/sentiment.ts:13-65, lib/llm/qwen.ts:56-104, "
    "lib/portfolio/ledger.ts:45-110, lib/agent/engine.ts:74-77, lib/config.ts:61-68, ported in "
    "rivals/registry.py"
)


def _spec(arm_id: str, title: str, description: str, provenance: str, *, llm: bool) -> ArmSpec:
    return ArmSpec(
        arm_id=arm_id,
        kind=ArmKind.RIVAL,
        title=title,
        description=description,
        provenance=provenance,
        uses_llm=llm,
        guards=RIVAL_GUARDS,
    )


_SPECS: Final[tuple[ArmSpec, ...]] = (
    _spec(
        LEXICON_FOLLOW,
        "Lexicon sentiment trader (with the crowd)",
        "Scores every post and headline naming a symbol with VADER's rules on a market-aware "
        "lexicon, and goes long a symbol the crowd is positive on and short one it is negative on, "
        "sized by the mean score.",
        VADER_PROVENANCE,
        llm=False,
    ),
    _spec(
        LEXICON_FADE,
        "Lexicon sentiment trader (against the crowd)",
        "The same lexicon scores, traded contrarian: short what the crowd is euphoric about, long "
        "what it despairs of, the handbook's 'FOMO detection -> contrarian hedge' done with words.",
        VADER_PROVENANCE,
        llm=False,
    ),
    _spec(
        FINBERT_FOLLOW,
        "finBERT-weighted trader (with the crowd)",
        "Scores every sentence of every post and headline naming a symbol with finBERT "
        "(P(positive) - P(negative)) and trades the mean score with the crowd.",
        FINBERT_PROVENANCE,
        llm=False,
    ),
    _spec(
        FINBERT_FADE,
        "finBERT-weighted trader (against the crowd)",
        "The same finBERT scores, traded contrarian.",
        FINBERT_PROVENANCE,
        llm=False,
    ),
    _spec(
        TRADINGAGENTS,
        "TradingAgents sentiment analyst and trader (Qwen)",
        "Per symbol, TradingAgents' sentiment analyst reads the news, X and Reddit text and "
        "writes a sentiment report; its trader turns the report into Buy, Hold or Sell with a "
        "size. Run on the same Qwen model as our agent.",
        TRADINGAGENTS_PROVENANCE,
        llm=True,
    ),
    _spec(
        SEASON2_CONFLUENCE,
        "Season-2 entry A: Fear & Greed with long/short confluence",
        "Buys when the crypto Fear & Greed index is below 25 and fewer than 45% of accounts are "
        "long; sells when it is above 75 and more than 65% are long; otherwise holds. BTC and "
        "TSLA, 5% of balance a trade capped at $50, 3 trades a day per asset.",
        SEASON2_CONFLUENCE_PROVENANCE,
        llm=False,
    ),
    _spec(
        SEASON2_FUSION,
        "Season-2 entry B: sentiment and news signal fusion",
        "Fuses a Fear & Greed, long/short and taker-ratio sentiment signal with a keyword news "
        "signal, confidence-weighted; a bullish or bearish regime opens a 1-2.5% BTC position, an "
        "uncertain one closes it. Its technical, macro and on-chain signals need data the snapshot "
        "does not carry and are absent, which its own fusion allows for.",
        SEASON2_FUSION_PROVENANCE,
        llm=False,
    ),
    _spec(
        SEASON2_QWEN_HEADLINES,
        "Season-2 entry C: Qwen headline-sentiment trader",
        "Qwen reads per-symbol headline sentiment, 24h momentum and the latest headlines each "
        "decision and returns Buy, Sell or Hold with a size; confident calls are applied to a "
        "long-only book with its own caps, rescaled to the same per-name budget as every arm.",
        SEASON2_QWEN_HEADLINES_PROVENANCE,
        llm=True,
    ),
)


def registry() -> tuple[ArmSpec, ...]:
    """Every rival arm's spec, in the order the published comparison lists them."""
    return _SPECS


def spec_for(arm_id: str) -> ArmSpec:
    for spec in _SPECS:
        if spec.arm_id == arm_id:
            return spec
    raise KeyError(f"no rival arm {arm_id!r} in the registry")


# ================================================================================================
# Shared rules: what a rival reads
# ================================================================================================

SOCIAL_CHANNELS: Final[frozenset[TextChannel]] = frozenset({"x", "reddit", "news"})
"""The channels a sentiment rival reads. Filings (Form 4 rows) are records, not opinion."""


def item_symbols(item: TextItem, universe: Sequence[str]) -> tuple[str, ...]:
    """The universe symbols a text names: those its collector tagged and those the text mentions,
    in universe order."""
    tagged = set(item.symbols)
    found = set(symbols_mentioned(item.text, universe))
    return tuple(s for s in universe if s in tagged or s in found)


def texts_by_symbol(
    snapshot: PerceptionSnapshot,
    *,
    channels: frozenset[TextChannel] = SOCIAL_CHANNELS,
    since: datetime | None = None,
) -> dict[str, tuple[TextItem, ...]]:
    """Every text in ``snapshot`` on ``channels``, filed under each universe symbol it names, most
    recent first. Withheld items are included (module docstring); items published after the
    snapshot instant, or before ``since``, are not. An item appears once however often it was
    collected."""
    seen: set[str] = set()
    by_symbol: dict[str, list[TextItem]] = {}
    for screened in snapshot.text:
        item = screened.item
        if item.item_id in seen or item.channel not in channels:
            continue
        if item.published_at > snapshot.taken_at:
            continue
        if since is not None and item.published_at < since:
            continue
        seen.add(item.item_id)
        for symbol in item_symbols(item, snapshot.universe):
            by_symbol.setdefault(symbol, []).append(item)
    return {
        s: tuple(sorted(items, key=lambda i: (i.published_at, i.item_id), reverse=True))
        for s, items in by_symbol.items()
    }


def recent_texts(
    snapshot: PerceptionSnapshot,
    *,
    channels: frozenset[TextChannel],
    limit: int,
) -> tuple[TextItem, ...]:
    """The ``limit`` most recent texts on ``channels`` published by the snapshot instant."""
    items = {
        s.item.item_id: s.item
        for s in snapshot.text
        if s.item.channel in channels and s.item.published_at <= snapshot.taken_at
    }
    ordered = sorted(items.values(), key=lambda i: (i.published_at, i.item_id), reverse=True)
    return tuple(ordered[:limit])


def first_line(text: str) -> str:
    """A headline: the first non-blank line of a text, whitespace collapsed."""
    for line in text.splitlines():
        if line.strip():
            return " ".join(line.split())
    return ""


# ================================================================================================
# Shared rules: size
# ================================================================================================


def cap_weights(weights: Mapping[str, float], policy: Policy) -> dict[str, float]:
    """``weights`` inside the mandate: each at most the per-name cap, and all scaled down together
    when their gross exceeds the risk budget. Zero weights are dropped.

    Raises ``ValueError`` for a non-finite weight or a symbol outside the policy universe: an arm
    that produces either is broken, and a comparison must not quietly repair it.
    """
    universe = set(policy.symbols)
    cap = policy.per_name_max
    capped: dict[str, float] = {}
    for symbol, weight in weights.items():
        if not math.isfinite(weight):
            raise ValueError(f"{symbol}: non-finite weight {weight}")
        if symbol not in universe:
            raise ValueError(f"{symbol} is not in the policy universe")
        clipped = max(-cap, min(cap, weight))
        if clipped != 0.0:
            capped[symbol] = clipped
    gross = sum(abs(w) for w in capped.values())
    budget = policy.mandate.risk_budget_gross
    if gross > budget:
        scale = budget / gross
        capped = {s: w * scale for s, w in capped.items()}
    return capped


def current_weights(book: BookState) -> dict[str, float]:
    """The book's non-zero weights."""
    weights = {s: book.weight(s) for s in book.positions}
    return {s: w for s, w in weights.items() if w != 0.0}


Direction = Literal["follow", "fade"]


@dataclass(frozen=True, slots=True)
class TextSentimentRule:
    """How a text-scoring rival turns per-symbol scores into weights. Pre-registered, not tuned.

    A symbol with fewer than ``min_items`` scored texts keeps its current weight: too little was
    said to change a position, and a quiet feed is not a reason to close one. With enough texts,
    a mean score inside ``neutral_band`` is flat (VADER's own neutral band is |compound| < 0.05,
    vaderSentiment README "About the Scoring"), and beyond it the weight is the per-name cap times
    ``mean / full_size_at``, clipped, with the crowd (``follow``) or against it (``fade``).
    """

    direction: Direction
    min_items: int = 3
    neutral_band: float = 0.05
    full_size_at: float = 0.5

    def __post_init__(self) -> None:
        if self.direction not in ("follow", "fade"):
            raise ValueError(f"direction must be 'follow' or 'fade', got {self.direction!r}")
        if self.min_items < 1:
            raise ValueError("min_items must be at least 1")
        if not 0.0 <= self.neutral_band < self.full_size_at <= 1.0:
            raise ValueError("need 0 <= neutral_band < full_size_at <= 1")


def weights_from_scores(
    scores: Mapping[str, Sequence[float]],
    *,
    book: BookState,
    rule: TextSentimentRule,
    policy: Policy,
) -> dict[str, float]:
    """Targets for every universe symbol from per-symbol text scores in [-1, 1] (see
    :class:`TextSentimentRule`), inside the mandate."""
    held = current_weights(book)
    sign = 1.0 if rule.direction == "follow" else -1.0
    targets: dict[str, float] = {}
    for symbol in policy.symbols:
        values = [v for v in scores.get(symbol, ()) if math.isfinite(v)]
        if len(values) < rule.min_items:
            if symbol in held:
                targets[symbol] = held[symbol]
            continue
        mean = sum(values) / len(values)
        if abs(mean) < rule.neutral_band:
            continue
        strength = max(-1.0, min(1.0, mean / rule.full_size_at))
        targets[symbol] = sign * policy.per_name_max * strength
    return cap_weights(targets, policy)


# ================================================================================================
# Shared rules: asking a model
# ================================================================================================


@dataclass(frozen=True, slots=True)
class RivalCall:
    """One model call a rival made: what for, and whether it produced something usable."""

    arm_id: str
    snapshot_id: str
    symbol: str | None
    stage: str
    ok: bool
    detail: str
    total_tokens: int


@dataclass
class CallLog:
    """Every call an LLM rival made, for the published record of its failures and spend."""

    calls: list[RivalCall] = field(default_factory=list)

    @property
    def failures(self) -> int:
        return sum(1 for c in self.calls if not c.ok)

    @property
    def tokens(self) -> int:
        return sum(c.total_tokens for c in self.calls)


def ask(
    model: ChatModel,
    messages: Sequence[ChatMessage],
    *,
    json_mode: bool,
    max_tokens: int,
    thinking: Thinking,
    temperature: float,
    seed: int | None,
) -> tuple[Completion | None, str]:
    """One call. A transport failure (timeout, refused connection, bad status) is returned as
    ``(None, reason)`` so the rival can hold, as its own error path does; a budget refusal is not
    caught, because a run that has run out of approved spend must stop rather than publish a rival
    that silently stopped deciding. A truncated completion is returned with ``"length"`` noted."""
    try:
        completion = model.complete(
            messages,
            json_mode=json_mode,
            max_tokens=max_tokens,
            thinking=thinking,
            temperature=temperature,
            seed=seed,
        )
    except QwenError as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if completion.finish_reason == "length":
        return completion, "truncated at max_tokens"
    return completion, ""


def load_json_object(text: str) -> dict[str, Any] | None:
    """The JSON object in a completion: the whole text, else the outermost balanced object in it.
    ``None`` when there is none."""
    for candidate in (text.strip(), extract_json_object(text)):
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, RecursionError):
            continue
        if isinstance(value, dict):
            return value
    return None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _js_round(value: float) -> int:
    """JavaScript ``Math.round``: halves round toward +infinity (Python's ``round`` is banker's)."""
    return math.floor(value + 0.5)


def _finite(value: Any) -> float | None:
    """A JSON number as a finite float, else ``None`` (booleans are not numbers)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


# ================================================================================================
# Season-2 entry A: Fear & Greed with long/short confluence (rebuilt from behaviour)
# ================================================================================================


class FearGreedConfluenceArm:
    """Season-2 entry A, rebuilt from its described behaviour (it ships no licence file).

    What the entry does (``strategy.js`` at the commit in :data:`SEASON2_CONFLUENCE_PROVENANCE`):

    * Primary signal, the crypto Fear & Greed index: above 75 is SELL, below 25 is BUY, otherwise
      HOLD (``strategy.js:22-26``, ``:417``).
    * Secondary signal, the asset's long/short account split: more than 65% of accounts long is
      SELL, fewer than 45% is BUY, otherwise NEUTRAL (``:27-30``, ``:422``).
    * A trade only when both agree; anything else holds (``:424-469``).
    * Each trade is 5% of the account balance and at most $50 (``:78-80``, ``:551-565``): on any
      balance above $1,000 the dollar cap binds, so a trade is ``50 / equity`` of equity. At most
      3 trades per asset per UTC day (``:77``, ``:667-671``).
    * Two assets: BTCUSDT on spot, so a SELL can only sell BTC that is held, and TSLAUSDT on
      USDT-futures, where a SELL opens a short (``:47-72``).
    * The entry itself only previews its orders (``mode: "DRY_RUN"``, ``:787``); the arm executes
      the decisions it would have placed.

    What this rebuild reads in their place: the Fear & Greed value the snapshot carries
    (``mood.crypto_fear_greed``, from Bitget's own services; the entry reads alternative.me) and
    the live retail long/short account ratio ``r`` of the features, as a long share ``r / (1 + r)``
    (the entry reads the same Bitget ``futures-long-short`` series). The equity perps have no such
    series (DESIGN.md §6.3); the entry then falls back to a cached value or a balanced default
    (``:298-311``), which is NEUTRAL, so TSLA holds, as it would there. Each trade moves the target
    by its size in equity (the account scale is the book's equity), never beyond the policy's
    per-name limit, and the daily quota counts the arm's own rebalances today
    (``book.rebalances_today``).
    """

    GREED_ABOVE: Final = 75
    FEAR_BELOW: Final = 25
    CROWDED_LONG_ABOVE: Final = 0.65
    CROWDED_SHORT_BELOW: Final = 0.45
    TRADE_FRACTION: Final = 0.05
    MAX_TRADE_USD: Final = 50.0
    MAX_TRADES_PER_DAY: Final = 3
    ASSETS: Final[tuple[str, ...]] = ("BTCUSDT", "TSLAUSDT")
    SPOT_ASSETS: Final[frozenset[str]] = frozenset({"BTCUSDT"})

    def __init__(self, *, policy: Policy) -> None:
        self.spec = spec_for(SEASON2_CONFLUENCE)
        self._policy = policy

    @staticmethod
    def sentiment_signal(fear_greed: int) -> Literal["BUY", "SELL", "HOLD"]:
        if fear_greed > FearGreedConfluenceArm.GREED_ABOVE:
            return "SELL"
        if fear_greed < FearGreedConfluenceArm.FEAR_BELOW:
            return "BUY"
        return "HOLD"

    @staticmethod
    def positioning_signal(long_share: float | None) -> Literal["BUY", "SELL", "NEUTRAL"]:
        if long_share is None:
            return "NEUTRAL"
        if long_share > FearGreedConfluenceArm.CROWDED_LONG_ABOVE:
            return "SELL"
        if long_share < FearGreedConfluenceArm.CROWDED_SHORT_BELOW:
            return "BUY"
        return "NEUTRAL"

    @classmethod
    def trade_fraction(cls, equity: float) -> float:
        """One trade as a fraction of equity: 5% of the balance, at most $50 (``:551-565``)."""
        if not math.isfinite(equity) or equity <= 0:
            return 0.0
        return min(cls.TRADE_FRACTION * equity, cls.MAX_TRADE_USD) / equity

    @staticmethod
    def long_share(ratio: float | None) -> float | None:
        """The share of accounts long, from a long/short account ratio."""
        if ratio is None or not math.isfinite(ratio) or ratio <= 0:
            return None
        return ratio / (1.0 + ratio)

    def targets(self, snapshot: PerceptionSnapshot, book: BookState) -> dict[str, float]:
        held = current_weights(book)
        targets = dict(held)
        fear_greed = snapshot.mood.crypto_fear_greed
        if fear_greed is None:
            return cap_weights(targets, self._policy)
        size = min(self.trade_fraction(float(book.equity)), self._policy.per_name_max)
        for symbol in self.ASSETS:
            if symbol not in snapshot.universe:
                continue
            features = snapshot.features.get(symbol)
            share = self.long_share(None if features is None else features.retail_long_short_ratio)
            sentiment = self.sentiment_signal(fear_greed)
            positioning = self.positioning_signal(share)
            if sentiment == "HOLD" or positioning != sentiment:
                continue
            if book.rebalances_today.get(symbol, 0) >= self.MAX_TRADES_PER_DAY:
                continue
            current = held.get(symbol, 0.0)
            if sentiment == "BUY":
                target = min(current + size, self._policy.per_name_max)
            else:
                floor = 0.0 if symbol in self.SPOT_ASSETS else -self._policy.per_name_max
                target = max(current - size, floor)
            targets[symbol] = target
        return cap_weights(targets, self._policy)


# ================================================================================================
# Season-2 entry B: sentiment and news signal fusion (MIT, ported)
# ================================================================================================

# Ported from Season-2 entry B (MIT License, Copyright (c) 2026 its author; see
# SEASON2_FUSION_PROVENANCE for the repository commit and RIVALS.md for the licence text).

FUSION_POSITIVE_WORDS: Final[tuple[str, ...]] = (
    "surge",
    "rally",
    "bullish",
    "soar",
    "gain",
    "breakout",
    "record high",
    "adoption",
    "approval",
    "upgrade",
    "rebound",
)
"""``mcpSignals.ts:54-56``, matched as substrings of the lower-cased headline, as there."""
FUSION_NEGATIVE_WORDS: Final[tuple[str, ...]] = (
    "crash",
    "plunge",
    "bearish",
    "selloff",
    "sell-off",
    "hack",
    "ban",
    "lawsuit",
    "collapse",
    "fraud",
    "downgrade",
    "liquidation",
)
"""``mcpSignals.ts:57-59``."""

FUSION_SIGNAL_WEIGHTS: Final[Mapping[str, float]] = {
    "technical": 0.3,
    "macro": 0.25,
    "sentiment": 0.2,
    "onchain": 0.15,
    "news": 0.1,
}
"""``signalFusion.ts:9-15``. Only ``sentiment`` and ``news`` can be computed from a snapshot."""

RegimeName = Literal["bullish_trend", "bearish_trend", "ranging", "uncertain"]


@dataclass(frozen=True, slots=True)
class FusionSignal:
    kind: str
    score: float
    confidence: float


@dataclass(frozen=True, slots=True)
class FusionRegime:
    regime: RegimeName
    confidence: int
    fused_score: float


def fusion_sentiment_signal(
    fear_greed: int | None, long_short: float | None, taker: float | None
) -> FusionSignal:
    """``mcpSignals.ts:62-97``: Fear & Greed read with the crowd, the long/short ratio against it,
    the taker ratio with it. A missing input takes the entry's own default (50, 1, 1); with every
    input missing the call failed there, which scores 0 at confidence 0.3 (``:92-94``)."""
    if fear_greed is None and long_short is None and taker is None:
        return FusionSignal("sentiment", 0.0, 0.3)
    fg = 50.0 if fear_greed is None else float(fear_greed)
    ls = 1.0 if long_short is None else long_short
    tk = 1.0 if taker is None else taker
    fg_score = _clamp((fg - 50) / 50, -1, 1)
    ls_score = _clamp((1 - ls) * 0.6, -1, 1)
    taker_score = _clamp((tk - 1) * 1.5, -1, 1)
    score = _clamp(fg_score * 0.5 + ls_score * 0.25 + taker_score * 0.25, -1, 1)
    return FusionSignal("sentiment", score, 0.7)


def fusion_news_signal(titles: Sequence[str]) -> FusionSignal:
    """``mcpSignals.ts:173-211``: headlines containing any positive word count +1, any negative
    word -1 (a headline can count both); the score is their balance, the confidence grows with
    the evidence. No headlines scores 0 at confidence 0.3."""
    pos = neg = 0
    for title in titles:
        lowered = title.lower()
        if any(w in lowered for w in FUSION_POSITIVE_WORDS):
            pos += 1
        if any(w in lowered for w in FUSION_NEGATIVE_WORDS):
            neg += 1
    total = pos + neg
    score = _clamp((pos - neg) / max(total, 1), -1, 1) if total > 0 else 0.0
    confidence = _clamp(0.4 + abs(score) * 0.3 + min(total, 5) * 0.02, 0.4, 0.7) if titles else 0.3
    return FusionSignal("news", score, confidence)


def fuse_signals(signals: Sequence[FusionSignal]) -> FusionRegime:
    """``signalFusion.ts:36-83``: confidence-weighted fusion with missing signals' weight
    redistributed, a regime from the fused score, and a confidence percentage."""
    if not signals:
        return FusionRegime("uncertain", 30, 0.0)
    numerator = denominator = 0.0
    for s in signals:
        weight = FUSION_SIGNAL_WEIGHTS.get(s.kind, 0.0)
        numerator += s.score * s.confidence * weight
        denominator += s.confidence * weight
    fused = _clamp(numerator / denominator if denominator != 0 else 0.0, -1, 1)
    overall = sum(s.confidence * FUSION_SIGNAL_WEIGHTS.get(s.kind, 0.0) for s in signals)
    regime: RegimeName
    if fused > 0.2:
        regime = "bullish_trend"
    elif fused < -0.2:
        regime = "bearish_trend"
    else:
        agreeing_or_flat = sum(
            1
            for s in signals
            if (s.score > 0 and fused > 0) or (s.score < 0 and fused < 0) or abs(s.score) < 0.1
        )
        regime = "ranging" if agreeing_or_flat >= 3 and abs(fused) >= 0.05 else "uncertain"
    agreeing = sum(1 for s in signals if (s.score > 0 and fused > 0) or (s.score < 0 and fused < 0))
    bonus = 15 if agreeing >= 5 else 8 if agreeing >= 4 else 0
    confidence = min(97, _js_round(overall * 100 + bonus + abs(fused) * 20))
    return FusionRegime(regime, confidence, fused)


def fusion_position_size(confidence: int, max_position: float) -> float:
    """``strategyRouter.ts:31-34`` and ``:137-138``."""
    if confidence < 50:
        size = 0.01
    elif confidence <= 70:
        size = 0.015
    elif confidence <= 85:
        size = 0.02
    else:
        size = min(0.025, max_position)
    return min(size, max_position)


class SentimentFusionArm:
    """Season-2 entry B, restricted to what a snapshot can feed it (module docstring).

    Each decision, as ``agentCycle.ts:62-154`` runs it on its one symbol (``BTCUSDT``, ``:66``):

    1. **Exits first** (``:104-106``): a held position whose move from entry reaches the strategy's
       stop (2.5%) or take-profit (6%) is closed (``strategyRouter.ts:64-65``). The entry's trailing
       stop (``executionEngine.ts:99-``) needs the path between decisions, which recorded snapshots
       do not carry; the simulator's 4% venue stop applies to every arm.
    2. **Regime** from the fused sentiment and news signals. The news signal reads the ten most
       recent news items containing "bitcoin", as the entry queries its feed
       (``mcpSignals.ts:176``).
    3. **Route** (``strategyRouter.ts:26-135``): bullish opens a long when flat and holds one;
       bearish closes a long, holds a short, or opens a short when flat; ranging with no technical
       signal holds (the band position defaults to the middle, ``:82-96``); uncertain closes.
       Entries only when flat (``agentCycle.ts:142``), sized by confidence, at most 2% of equity
       (``MAX_POSITION_SIZE_PCT`` default, ``:16``).

    4. **Risk manager** (``riskManager.ts:55-101``), the parts a snapshot can drive: a new entry is
       halved when the news score moved by more than 0.4 since the previous decision (``:79-89``,
       with the previous score starting at 0 as ``agentCycle.ts:25`` starts it), and nothing but a
       capital-protection close executes once 10 trades were made today (``:71-77``). Its 10%
       drawdown halt and 5% daily-loss halt need the arm's own equity, which the harness does not
       show an arm; the venue guards' 1.5% daily kill is the stricter of the two in any case.

    The arm keeps the previous news score between calls, so it must be asked in time order, as the
    harness asks it.
    """

    SYMBOL: Final = "BTCUSDT"
    MAX_POSITION: Final = 0.02
    STOP_LOSS: Final = 0.025
    TAKE_PROFIT: Final = 0.06
    NEWS_KEYWORD: Final = "bitcoin"
    NEWS_LIMIT: Final = 10
    NEWS_SPIKE: Final = 0.4
    MAX_TRADES_PER_DAY: Final = 10

    def __init__(self, *, policy: Policy) -> None:
        self.spec = spec_for(SEASON2_FUSION)
        self._policy = policy
        self._last_news = 0.0

    def signals(self, snapshot: PerceptionSnapshot) -> tuple[FusionSignal, FusionSignal]:
        """The sentiment and news signals a snapshot can feed the entry."""
        features = snapshot.features.get(self.SYMBOL)
        sentiment = fusion_sentiment_signal(
            snapshot.mood.crypto_fear_greed,
            None if features is None else features.retail_long_short_ratio,
            None if features is None else features.taker_buy_sell_ratio,
        )
        news_items = [
            i
            for i in recent_texts(snapshot, channels=frozenset({"news"}), limit=len(snapshot.text))
            if self.NEWS_KEYWORD in i.text.lower()
        ][: self.NEWS_LIMIT]
        return sentiment, fusion_news_signal([first_line(i.text) for i in news_items])

    def regime(self, snapshot: PerceptionSnapshot) -> FusionRegime:
        return fuse_signals(self.signals(snapshot))

    def targets(self, snapshot: PerceptionSnapshot, book: BookState) -> dict[str, float]:
        held = current_weights(book)
        current = held.get(self.SYMBOL, 0.0)
        position = book.positions.get(self.SYMBOL)
        mark = book.marks.get(self.SYMBOL)
        if current != 0.0 and position is not None and mark is not None and position.avg_entry > 0:
            move = float(mark / position.avg_entry - 1) * (1.0 if current > 0 else -1.0)
            if move <= -self.STOP_LOSS or move >= self.TAKE_PROFIT:
                current = 0.0
        sentiment, news = self.signals(snapshot)
        reading = fuse_signals([sentiment, news])
        news_shift = abs(news.score - self._last_news)
        self._last_news = news.score
        size = fusion_position_size(reading.confidence, self.MAX_POSITION)
        target = current
        if reading.regime == "bullish_trend":
            if current == 0.0:
                target = size
        elif reading.regime == "bearish_trend":
            if current > 0:
                target = 0.0
            elif current == 0.0:
                target = -size
        elif reading.regime == "uncertain":
            target = 0.0
        if reading.regime != "uncertain":
            if sum(book.rebalances_today.values()) >= self.MAX_TRADES_PER_DAY:
                target = current
            elif current == 0.0 and target != 0.0 and news_shift > self.NEWS_SPIKE:
                target *= 0.5
        targets = {s: w for s, w in held.items() if s != self.SYMBOL}
        if self.SYMBOL in snapshot.universe and target != 0.0:
            targets[self.SYMBOL] = target
        return cap_weights(targets, self._policy)


# ================================================================================================
# Season-2 entry C: Qwen headline-sentiment trader (MIT, ported)
# ================================================================================================

# Ported from Season-2 entry C (MIT License; see SEASON2_QWEN_HEADLINES_PROVENANCE for the
# repository commit and RIVALS.md for the licence text).

HEADLINE_POSITIVE_WORDS: Final[tuple[str, ...]] = (
    "beats",
    "surge",
    "surges",
    "rally",
    "upgrade",
    "upgraded",
    "record",
    "strong",
    "demand",
    "raise",
    "raises",
    "tops",
    "optimism",
    "gains",
    "outperform",
    "expands",
    "accelerates",
    "boosts",
)
"""``lib/data/sentiment.ts:13-17``, matched as substrings of the lower-cased headline."""
HEADLINE_NEGATIVE_WORDS: Final[tuple[str, ...]] = (
    "miss",
    "falls",
    "fall",
    "cut",
    "cuts",
    "downgrade",
    "probe",
    "lawsuit",
    "warns",
    "slump",
    "weak",
    "recall",
    "recalls",
    "delay",
    "delays",
    "layoffs",
    "selloff",
    "concerns",
    "halts",
    "pressure",
)
"""``lib/data/sentiment.ts:18-22``."""

HEADLINE_WATCHLIST: Final[tuple[str, ...]] = (
    "AAPLUSDT",
    "TSLAUSDT",
    "NVDAUSDT",
    "SP500USDT",
    "COINUSDT",
)
"""The entry's watchlist (``lib/config.ts:27-34``: AAPL, TSLA, NVDA, MSFT, SPY, COIN) as the
policy universe has it: SPY is the S&P 500 perp, and MSFT has no Demo perp."""

HEADLINE_STRATEGY_TONE: Final = (
    "You balance growth and risk. Take measured positions when signals align, trim into strength, "
    "and add on constructive pullbacks."
)
"""The entry's default ("balanced") strategy, ``lib/config.ts:61-68``."""
HEADLINE_MAX_POSITION_PCT: Final = 35.0
HEADLINE_MAX_TRADE_PCT: Final = 20.0
HEADLINE_MIN_CONFIDENCE: Final = 0.5
HEADLINE_MIN_TRADE_USD: Final = 1.0
"""``lib/portfolio/ledger.ts:6``: a buy below $1 is dust and skipped."""
HEADLINE_TEMPERATURE: Final = 0.7
"""``lib/llm/qwen.ts:100``."""
HEADLINE_MAX_TOKENS: Final = 1500
"""``lib/llm/qwen.ts:103``."""
HEADLINE_LIMIT: Final = 6
"""Headlines shown to the model, ``lib/agent/prompt.ts:63``."""
HEADLINE_MAX_CHARS: Final = 200
"""A headline is cut to this many characters and to :data:`HEADLINE_MAX_BYTES` UTF-8 bytes, so the
token bound of a call can be computed before it is made (the entry's feed titles are one line)."""
HEADLINE_MAX_BYTES: Final = 400


def headline_polarity(title: str) -> int:
    """``lib/data/sentiment.ts:24-30``."""
    lowered = title.lower()
    return sum(1 for w in HEADLINE_POSITIVE_WORDS if w in lowered) - sum(
        1 for w in HEADLINE_NEGATIVE_WORDS if w in lowered
    )


def headline_sentiment(titles: Sequence[str], change_pct_24h: float) -> tuple[float, str]:
    """``lib/data/sentiment.ts:44-62``: 0.55 of the mean headline polarity plus 0.45 of the 24h move
    normalised at 4.5%, rounded to cents, labelled beyond +-0.15."""
    news = _clamp(sum(headline_polarity(t) for t in titles) / len(titles), -1, 1) if titles else 0.0
    momentum = _clamp(change_pct_24h / 4.5, -1, 1)
    score = _js_round(_clamp(0.55 * news + 0.45 * momentum, -1, 1) * 100) / 100
    label = "bullish" if score > 0.15 else "bearish" if score < -0.15 else "neutral"
    return score, label


def _truncate(text: str, *, chars: int, max_bytes: int) -> str:
    out = text[:chars]
    while len(out.encode("utf-8")) > max_bytes:
        out = out[:-1]
    return out


class QwenHeadlineTraderArm:
    """Season-2 entry C: one Qwen call per decision over the whole watchlist.

    The prompt is the entry's (``lib/agent/prompt.ts:18-80``) with the tickers renamed to the
    policy's perps and its prices and headlines taken from the snapshot: the live last price and
    24h change per symbol, the headline sentiment its own scorer computes, and the six most recent
    news headlines with the symbols they name. The book is shown as the entry keeps it: cash,
    equity, and long positions at their average cost, all in the entry's own units.

    The answer is parsed as the entry parses it (``lib/llm/qwen.ts:56-74``: unknown tickers
    dropped, size and confidence clamped) and applied as its ledger applies it (``engine.ts:74-77``,
    ``ledger.ts:45-110``): only BUY and SELL with confidence at least 0.5, on a ticker with a
    price; a BUY spends up to ``sizePct`` of cash, at most 20% of cash and never beyond 35% of
    equity in one name, and not at all below $1; a SELL sells ``sizePct`` of the holding. The book
    is long-only.

    **Scale.** The entry's caps are its own (35% of equity per name); every arm here trades inside
    the same per-name budget, so its weights are rescaled by ``per_name_max / 0.35``: its largest
    permitted position is ours. Its relative sizing is kept exactly.

    **Where it departs.** An unparseable answer falls back to a rule-based mock in the entry
    (``qwen.ts:59``); here it holds and is recorded as a failure, because a mock is not the LLM
    rival deciding. The entry sends no reasoning tier; the gateway default is NOT VERIFIED, so the
    call uses ``Thinking.LOW`` and says so. The prompt's realized P&L line reads ``$0.00``: the
    arm is shown its own positions, rebuilt from its targets, and those carry no realized history.
    """

    SEED: Final = 20260924

    def __init__(
        self, *, model: ChatModel, policy: Policy, thinking: Thinking = Thinking.LOW
    ) -> None:
        self.spec = spec_for(SEASON2_QWEN_HEADLINES)
        self._model = model
        self._policy = policy
        self._thinking = thinking
        self.log = CallLog()
        self._scale = policy.per_name_max / (HEADLINE_MAX_POSITION_PCT / 100)

    @property
    def watchlist(self) -> tuple[str, ...]:
        return tuple(s for s in HEADLINE_WATCHLIST if s in self._policy.symbols)

    def system_prompt(self) -> str:
        """``lib/agent/prompt.ts:18-36``, verbatim apart from the ticker list."""
        universe = ", ".join(self.watchlist)
        return "\n".join(
            [
                "You are Nocturne, an autonomous trading agent operating 24/7 on tokenized US "
                "equities.",
                "You manage a paper-trading portfolio. Each tick you receive market signals and "
                "must return trading decisions as STRICT JSON.",
                "",
                f"Tradable universe (use ONLY these tickers): {universe}.",
                "",
                f"Strategy Balanced: {HEADLINE_STRATEGY_TONE}",
                f"Risk limits: at most {HEADLINE_MAX_TRADE_PCT:g}% of cash per BUY; keep any "
                f"single position under {HEADLINE_MAX_POSITION_PCT:g}% of total equity.",
                "",
                "For each ticker choose BUY, SELL, or HOLD:",
                "- sizePct: BUY = percent of available cash to deploy (0-100); SELL = percent of "
                "the held position to sell (0-100); HOLD = 0.",
                "- confidence: 0-1.",
                "- rationale: one or two sentences citing the concrete signals (price move, "
                "sentiment, a specific headline).",
                "",
                "Respond with ONLY a JSON object of exactly this shape and nothing else:",
                '{"marketView": string, "decisions": [{"ticker": string, "action": '
                '"BUY"|"SELL"|"HOLD", "sizePct": number, "confidence": number, "rationale": '
                "string}]}",
            ]
        )

    def native_book(self, book: BookState) -> tuple[dict[str, float], float]:
        """The entry's own view of the book: long weights in its units, and cash as a fraction."""
        native = {
            s: w / self._scale
            for s, w in current_weights(book).items()
            if w > 0 and s in self.watchlist
        }
        return native, max(0.0, 1.0 - sum(native.values()))

    def user_prompt(self, snapshot: PerceptionSnapshot, book: BookState) -> str:
        """``lib/agent/prompt.ts:38-80``, with the snapshot's numbers."""
        native, cash_fraction = self.native_book(book)
        equity = float(book.equity)
        texts = texts_by_symbol(snapshot, channels=frozenset({"news"}))
        lines = []
        for symbol in self.watchlist:
            features = snapshot.features.get(symbol)
            price = None if features is None else features.live_last
            change = 0.0
            if features is not None and features.price_change_24h_pct is not None:
                change = features.price_change_24h_pct
            titles = [
                _truncate(
                    first_line(i.text), chars=HEADLINE_MAX_CHARS, max_bytes=HEADLINE_MAX_BYTES
                )
                for i in texts.get(symbol, ())
            ]
            score, label = headline_sentiment(titles, change)
            price_text = "n/a" if price is None else f"${price:.2f}"
            chg = f"{'+' if change >= 0 else ''}{change:.1f}%"
            lines.append(
                f"{symbol:<6} {price_text:>9}  24h {chg:>6}  sentiment {label}({score:.2f})"
            )
        held_lines = []
        for s in sorted(native):
            position, mark = book.positions.get(s), book.marks.get(s)
            if position is None or mark is None or mark <= 0:
                continue
            qty = native[s] * equity / float(mark)
            held_lines.append(f"{s} {qty:.4f}@{float(position.avg_entry):.2f}")
        positions = ", ".join(held_lines)
        headlines = []
        for item in recent_texts(snapshot, channels=frozenset({"news"}), limit=HEADLINE_LIMIT):
            title = _truncate(
                first_line(item.text), chars=HEADLINE_MAX_CHARS, max_bytes=HEADLINE_MAX_BYTES
            )
            named = ", ".join(item_symbols(item, self.watchlist))
            headlines.append(f"- {title} [{named}]")
        return "\n".join(
            [
                f"Tick timestamp: {snapshot.taken_at.strftime('%Y-%m-%dT%H:%M:%S.000Z')}",
                f"Cash: ${cash_fraction * equity:,.2f} | Equity: ${equity:,.2f} | Realized P&L: "
                "$0.00",
                f"Positions: {positions or 'none'}",
                "",
                "Signals:",
                *lines,
                "",
                "Recent headlines:",
                *headlines,
                "",
                "Return your decisions as JSON now.",
            ]
        )

    def messages(self, snapshot: PerceptionSnapshot, book: BookState) -> list[ChatMessage]:
        return [
            ChatMessage(role="system", content=self.system_prompt()),
            ChatMessage(role="user", content=self.user_prompt(snapshot, book)),
        ]

    def parse(self, content: str) -> list[tuple[str, str, float, float]] | None:
        """``(ticker, action, sizePct, confidence)`` per valid decision, or ``None`` when the answer
        has the wrong shape or names no watchlist ticker (``qwen.ts:56-74``)."""
        obj = load_json_object(content)
        if obj is None or not isinstance(obj.get("decisions"), list):
            return None
        valid = set(self.watchlist)
        decisions: list[tuple[str, str, float, float]] = []
        for raw in obj["decisions"]:
            if not isinstance(raw, dict):
                return None
            ticker, action = raw.get("ticker"), raw.get("action")
            size, confidence = _finite(raw.get("sizePct")), _finite(raw.get("confidence"))
            if not isinstance(ticker, str) or action not in ("BUY", "SELL", "HOLD"):
                return None
            if size is None or confidence is None:
                return None
            if ticker not in valid:
                continue
            decisions.append((ticker, action, _clamp(size, 0, 100), _clamp(confidence, 0, 1)))
        return decisions or None

    def apply(
        self,
        decisions: Sequence[tuple[str, str, float, float]],
        book: BookState,
        *,
        priced: frozenset[str],
    ) -> dict[str, float]:
        """``engine.ts:74-77`` then ``ledger.ts:45-110`` on the entry's own units; back to ours.

        ``priced`` are the tickers with a price this decision: a call on one without a price is
        skipped, as the ledger skips it (``ledger.ts:57-58``), and a buy of less than $1 is dust
        (``:6``, ``:69``)."""
        native, cash = self.native_book(book)
        max_position = HEADLINE_MAX_POSITION_PCT / 100
        equity = float(book.equity)
        for ticker, action, size_pct, confidence in decisions:
            if action == "HOLD" or confidence < HEADLINE_MIN_CONFIDENCE or ticker not in priced:
                continue
            held = native.get(ticker, 0.0)
            if action == "BUY":
                by_trade = cash * min(size_pct, HEADLINE_MAX_TRADE_PCT) / 100
                desired = cash * size_pct / 100
                room = max(0.0, max_position - held)
                spend = min(desired, by_trade, room, cash)
                if spend * equity < HEADLINE_MIN_TRADE_USD:
                    continue
                native[ticker] = held + spend
                cash -= spend
            elif held > 0:
                sold = held if size_pct >= 100 else held * size_pct / 100
                native[ticker] = held - sold
                cash += sold
        targets = {s: w for s, w in current_weights(book).items() if s not in self.watchlist}
        targets.update({s: w * self._scale for s, w in native.items() if w > 0})
        return cap_weights(targets, self._policy)

    def targets(self, snapshot: PerceptionSnapshot, book: BookState) -> dict[str, float]:
        completion, problem = ask(
            self._model,
            self.messages(snapshot, book),
            json_mode=True,
            max_tokens=HEADLINE_MAX_TOKENS,
            thinking=self._thinking,
            temperature=HEADLINE_TEMPERATURE,
            seed=self.SEED,
        )
        tokens = 0 if completion is None else completion.usage.total_tokens
        decisions = None if completion is None else self.parse(completion.content)
        ok = decisions is not None
        detail = problem or ("" if ok else "answer had no usable decisions")
        self.log.calls.append(
            RivalCall(self.spec.arm_id, snapshot.snapshot_id, None, "decide", ok, detail, tokens)
        )
        if decisions is None:
            return cap_weights(current_weights(book), self._policy)
        priced = frozenset(
            s
            for s in self.watchlist
            if (f := snapshot.features.get(s)) is not None
            and f.live_last is not None
            and f.live_last > 0
        )
        return self.apply(decisions, book, priced=priced)

    def max_tokens_per_snapshot(self) -> int:
        """An upper bound on one decision's tokens, the way the budget bounds a call
        (:func:`~sentiment_agent.llm.budget.projected_tokens`): the prompt's UTF-8 bytes at their
        largest plus the completion cap."""
        widest_line = f"{'X' * 12:<6} ${'9' * 12}.99  24h {'-999.9%':>6}  sentiment bearish(-1.00)"
        widest_headline = "- " + "X" * HEADLINE_MAX_BYTES + " [" + ", ".join(self.watchlist) + "]"
        positions = ", ".join(f"{s} {'9' * 16}.9999@{'9' * 12}.99" for s in self.watchlist)
        user = "\n".join(
            [
                "Tick timestamp: 2026-01-01T00:00:00.000Z",
                f"Cash: ${'9' * 20}.99 | Equity: ${'9' * 20}.99 | Realized P&L: $0.00",
                f"Positions: {positions}",
                "",
                "Signals:",
                *([widest_line] * len(self.watchlist)),
                "",
                "Recent headlines:",
                *([widest_headline] * HEADLINE_LIMIT),
                "",
                "Return your decisions as JSON now.",
            ]
        )
        messages = [
            ChatMessage(role="system", content=self.system_prompt()),
            ChatMessage(role="user", content=user),
        ]
        return projected_tokens(messages, HEADLINE_MAX_TOKENS)


# ================================================================================================
# Helpers shared with the LLM arms
# ================================================================================================

_PERCENT = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")


def percent_in(text: str | None) -> float | None:
    """The first percentage written in ``text`` as a fraction (``"5% of portfolio"`` -> 0.05)."""
    if not text:
        return None
    match = _PERCENT.search(text)
    if match is None:
        return None
    value = float(match.group(1)) / 100
    return value if math.isfinite(value) else None


def asset_type(symbol: str, policy: Policy) -> Literal["crypto", "stock"]:
    entry = policy.entry(symbol)
    return "crypto" if entry is not None and entry.asset_class is AssetClass.CRYPTO else "stock"


def seven_days_before(at: datetime) -> datetime:
    return at - timedelta(days=7)


__all__ = [
    "FINBERT_FADE",
    "FINBERT_FOLLOW",
    "FUSION_NEGATIVE_WORDS",
    "FUSION_POSITIVE_WORDS",
    "FUSION_SIGNAL_WEIGHTS",
    "HEADLINE_NEGATIVE_WORDS",
    "HEADLINE_POSITIVE_WORDS",
    "HEADLINE_WATCHLIST",
    "LEXICON_FADE",
    "LEXICON_FOLLOW",
    "RIVAL_GUARDS",
    "SEASON2_CONFLUENCE",
    "SEASON2_FUSION",
    "SEASON2_QWEN_HEADLINES",
    "SOCIAL_CHANNELS",
    "TRADINGAGENTS",
    "CallLog",
    "Direction",
    "FearGreedConfluenceArm",
    "FusionRegime",
    "FusionSignal",
    "QwenHeadlineTraderArm",
    "RivalArm",
    "RivalCall",
    "SentimentFusionArm",
    "TextSentimentRule",
    "ask",
    "asset_type",
    "cap_weights",
    "current_weights",
    "first_line",
    "fuse_signals",
    "fusion_news_signal",
    "fusion_position_size",
    "fusion_sentiment_signal",
    "headline_polarity",
    "headline_sentiment",
    "item_symbols",
    "load_json_object",
    "percent_in",
    "recent_texts",
    "registry",
    "seven_days_before",
    "spec_for",
    "texts_by_symbol",
    "weights_from_scores",
]
