# This file contains prompt text and output schemas adapted from TradingAgents
# (https://github.com/TauricResearch/TradingAgents, commit be952b8eccb49720509af544c6675233bc1f10d0,
# by Yijia Xiao, Edward Sun, Di Luo and Wei Wang), licensed under the Apache License, Version 2.0.
# You may not use this file except in compliance with that License; a copy is reproduced in
# src/sentiment_agent/rivals/RIVALS.md and is at http://www.apache.org/licenses/LICENSE-2.0.
#
# Taken from:
#   tradingagents/agents/analysts/sentiment_analyst.py:47-48, :64-76, :88-99, :140-192
#   tradingagents/agents/trader/trader.py:46-74
#   tradingagents/agents/schemas.py:137-204 (TraderProposal), :285-368 (SentimentReport)
#   tradingagents/agents/utils/structured.py:36-39 (NO_EXTERNAL_TOOLS), :59-89 (fallback)
#   tradingagents/agents/utils/agent_utils.py:136-183 (instrument context)
#   tradingagents/agents/utils/rating.py:35-66 (rating extraction)
#   tradingagents/dataflows/stocktwits.py:107-138, reddit.py:74, :264-334, yfinance_news.py:95-117
#   tradingagents/graph/propagation.py:35, tradingagents/graph/setup.py:77, :86
#
# CHANGES MADE (Apache-2.0 section 4(b)): the data blocks are filled from this project's recorded
# snapshot instead of live Yahoo Finance, StockTwits and Reddit fetches, and X posts stand in for
# StockTwits, which has no labelled equivalent here; the structured-output schemas are stated in
# the prompt because the transport offers JSON mode, not a schema binding; the analyst's report is
# handed to the trader in place of the Research Manager's plan; texts, source names and the report
# are truncated to fixed byte bounds; the Sell action on a perpetual opens a short; temperature 0
# and the LOW reasoning tier are set where upstream leaves both to the provider. Each change is
# described in the module docstring below.
"""TradingAgents' sentiment analyst and trader, as one rival, on the same Qwen model as our agent.

TradingAgents (TauricResearch, Apache-2.0) is the most-cited open multi-agent LLM trading framework.
Its sentiment analyst (the renamed social-media analyst, ``sentiment_analyst.py:1-25``) reads a
ticker's news, retail social posts and Reddit threads from the past seven days and writes a
structured sentiment report; its trader turns a plan into Buy, Hold or Sell with levels and a size.
Chained, the two are a market-sentiment trading agent in exactly the handbook's sense, which is why
this is the LLM rival for the sub-theme.

How the chain runs here, per symbol, per snapshot
-------------------------------------------------
1. **Analyst.** System prompt as ``sentiment_analyst.py:88-99`` (today's date, the instrument
   context, the no-tools rule) followed by the analyst instructions and data blocks of ``:140-192``;
   the user message is the ticker, as the graph's first message is (``propagation.py:35``). The
   blocks hold the snapshot's texts naming the symbol from the seven days before the snapshot
   (``:47-48``, ``:64``): news (up to 20, ``default_config.py:122``), X posts (up to 30, the
   StockTwits limit at ``:74``) and Reddit posts (up to 5 per subreddit and 3 subreddits,
   ``reddit.py:74``, ``:267``), each formatted as its upstream fetcher formats it.
2. **Trader.** System prompt as ``trader.py:46-63`` without a market report (the chain has no market
   analyst, so the grounding sentence is omitted exactly as upstream omits it); user message as
   ``:65-72`` with the analyst's rendered report as the investment plan.
3. **Weight.** Buy is long and Sell is short (a perpetual: upstream's Sell on a cash equity exits,
   here it opens a short, as a Sell order does on this venue); the size is the first percentage in
   ``position_sizing`` (``schemas.py:171-174``: "e.g. '5% of portfolio'"), capped at the per-name
   limit, or the per-name limit when none is given. Hold keeps the position.

Both calls use the quick-thinking model upstream gives them (``setup.py:77``, ``:86``). Upstream
leaves the temperature and the reasoning effort to the provider's defaults
(``default_config.py:93``, ``:99``); the gateway's defaults for Qwen are NOT VERIFIED, so this
harness sets them: temperature 0, for a reproducible record, and ``Thinking.LOW``, the tier that
matches a quick-thinking role.

Output handling follows ``structured.py:59-89``: the structured call first (here JSON mode with the
schema described in the prompt, since the transport offers ``json_object`` rather than a schema
binding), and on any failure one retry as free text. A free-text analyst answer is passed on as the
report, as upstream passes it on; a free-text trader answer is read for its action by
``rating.py``'s two-pass heuristic (a labelled action, then the first standalone action word),
narrowed to the trader's three actions. An answer with no recognisable action is REVIEW
(``signal_processing.py:32-39``): not tradeable, so the position is held.

Cost bounds
-----------
Every piece of text in a prompt is truncated to a fixed number of characters and UTF-8 bytes, and
the report handed to the trader to :data:`REPORT_MAX_BYTES`, so :meth:`max_tokens_per_snapshot` is a
true upper bound the budget can check before a run (bytes bound tokens for Qwen's byte-level BPE,
``llm/budget.py``). At most :data:`MAX_SYMBOLS` symbols are evaluated per snapshot: held symbols
first, then those with the most texts; the rest keep their weight.
"""

import math
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final, Literal

from sentiment_agent.crowd.novelty import aliases_for
from sentiment_agent.llm.budget import projected_tokens
from sentiment_agent.rivals.registry import (
    TRADINGAGENTS,
    CallLog,
    RivalCall,
    ask,
    asset_type,
    cap_weights,
    current_weights,
    first_line,
    load_json_object,
    percent_in,
    seven_days_before,
    spec_for,
    texts_by_symbol,
)
from sentiment_agent.types import (
    BookState,
    ChatMessage,
    ChatModel,
    PerceptionSnapshot,
    Policy,
    TextItem,
    Thinking,
)

MAX_SYMBOLS: Final = 3
NEWS_LIMIT: Final = 20
X_LIMIT: Final = 30
REDDIT_PER_SUBREDDIT: Final = 5
REDDIT_SUBREDDITS: Final = 3
POST_CHARS: Final = 280
"""``stocktwits.py:116-117``."""
TITLE_CHARS: Final = 300
BODY_CHARS: Final = 240
"""``reddit.py:322-323``."""
SOURCE_CHARS: Final = 40
TEXT_MAX_BYTES: Final = 600
"""Every truncated text is also cut to this many UTF-8 bytes, so a prompt's size is bounded."""
SOURCE_MAX_BYTES: Final = 160
REPORT_MAX_BYTES: Final = 12_000
ANALYST_MAX_TOKENS: Final = 2_048
TRADER_MAX_TOKENS: Final = 768
ATTEMPTS_PER_STAGE: Final = 2
"""The structured call and one free-text retry (``structured.py:73-89``)."""

NO_EXTERNAL_TOOLS: Final = (
    "Use only the evidence provided in this prompt. Do not call external tools "
    "or search the web; if something is missing, say so explicitly."
)
"""``structured.py:36-39``."""

SENTIMENT_BANDS: Final[tuple[str, ...]] = (
    "Bullish",
    "Mildly Bullish",
    "Neutral",
    "Mixed",
    "Mildly Bearish",
    "Bearish",
)
"""``schemas.py:285-297``."""

TraderAction = Literal["Buy", "Hold", "Sell"]
_ACTIONS: Final[tuple[TraderAction, ...]] = ("Buy", "Hold", "Sell")

ANALYST_OUTPUT: Final = """

## Output format

Respond with one JSON object and nothing else, with exactly these keys:

- "overall_band": Overall sentiment direction. Exactly one of: Bullish / Mildly Bullish / Neutral / Mixed / Mildly Bearish / Bearish. Use Mixed when sources point in clearly different directions. Use Neutral only when all sources are genuinely silent or non-committal.
- "overall_score": Numeric sentiment intensity on a 0–10 scale. 0 = maximally bearish, 5 = neutral, 10 = maximally bullish. Guideline for consistency with overall_band: Bullish ~6.5–10, Mildly Bullish ~5.5–6.4, Neutral/Mixed ~4.5–5.5, Mildly Bearish ~3.5–4.4, Bearish ~0–3.4. Only the 0–10 bounds are enforced.
- "confidence": Confidence in the assessment based on data quality and sample size. Use 'low' when one or more sources returned a placeholder or fewer than 5 data points; 'medium' when data is present but sparse; 'high' when all three sources returned substantive data.
- "narrative": Full sentiment report covering, in order: (1) source-by-source breakdown with specific evidence (cite message counts, ratios, notable posts); (2) cross-source divergences and alignments; (3) dominant narrative themes; (4) catalysts and risks surfaced by the data; (5) a markdown table summarising key sentiment signals, their direction, source, and supporting evidence. Keep it informative and substantive: develop each section thoroughly with concrete evidence so every point adds new signal for the trader."""  # noqa: E501
"""The field descriptions of ``SentimentReport`` (``schemas.py:311-352``), stated in the prompt."""

TRADER_OUTPUT: Final = """

Respond with one JSON object and nothing else, with exactly these keys:

- "action": The transaction direction. Exactly one of Buy / Hold / Sell.
- "reasoning": The case for this action, anchored in the analysts' reports and the research plan. Two to four sentences.
- "entry_price": Optional entry price target as an absolute number in the instrument's quote currency (e.g. 189.5), never a percentage or a range. Use null if you cannot state a specific level.
- "stop_loss": Optional stop-loss as an absolute price in the instrument's quote currency (e.g. 172.0), never a percentage. Convert a percentage distance to the price level it implies, or use null.
- "position_sizing": Optional sizing guidance, e.g. '5% of portfolio'. Use null if none."""  # noqa: E501
"""The field descriptions of ``TraderProposal`` (``schemas.py:146-174``), stated in the prompt."""


# ================================================================================================
# Text handling
# ================================================================================================


def clip(text: str, *, chars: int, max_bytes: int) -> str:
    """``text`` cut to ``chars`` characters (with an ellipsis, as the fetchers cut) and then to
    ``max_bytes`` UTF-8 bytes."""
    out = text if len(text) <= chars else text[:chars] + "…"
    while len(out.encode("utf-8")) > max_bytes:
        out = out[:-1]
    return out


def display_ticker(symbol: str, policy: Policy) -> str:
    """The ticker a TradingAgents user would type: ``NVDA`` for the NVDA perp, ``BTC-USD`` for
    Bitcoin (a Yahoo pair, as its crypto support expects), ``SPX`` and ``NDX`` for the indices."""
    base = aliases_for(symbol).cashtags[0]
    return f"{base}-USD" if asset_type(symbol, policy) == "crypto" else base


def instrument_context(ticker: str, kind: Literal["crypto", "stock"]) -> str:
    """``agent_utils.py:136-183`` without a resolved identity (``:186-201`` falls back to this when
    none was resolved)."""
    is_crypto = kind == "crypto"
    label = "asset" if is_crypto else "instrument"
    context = (
        f"The {label} to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`, `-USD`)."
    )
    if is_crypto:
        context += (
            " Treat it as a crypto asset rather than a company, and do not "
            "assume company fundamentals are available."
        )
    return context


def _ymd(at: datetime) -> str:
    return at.astimezone(UTC).strftime("%Y-%m-%d")


def news_block(ticker: str, items: Sequence[TextItem], start: datetime, end: datetime) -> str:
    """``yfinance_news.py:95-117``: a title line with its publisher, then the summary."""
    if not items:
        return f"No news found for {ticker} between {_ymd(start)} and {_ymd(end)}"
    body = ""
    for item in items[:NEWS_LIMIT]:
        title = clip(first_line(item.text), chars=TITLE_CHARS, max_bytes=TEXT_MAX_BYTES)
        source = clip(item.source, chars=SOURCE_CHARS, max_bytes=SOURCE_MAX_BYTES)
        body += f"### {title} (source: {source})\n"
        rest = " ".join(item.text.split())[len(first_line(item.text)) :].strip()
        if rest:
            body += clip(rest, chars=POST_CHARS, max_bytes=TEXT_MAX_BYTES) + "\n"
        body += "\n"
    return f"## {ticker} News, from {_ymd(start)} to {_ymd(end)}:\n\n{body}"


def x_block(ticker: str, items: Sequence[TextItem], start: datetime, end: datetime) -> str:
    """``stocktwits.py:107-138`` with X posts: X has no user-labelled sentiment, so every post is
    ``no-label`` and the summary line says so."""
    if not items:
        return f"<no X posts for ${ticker.upper()} within {_ymd(start)}..{_ymd(end)}>"
    lines = []
    for item in items[:X_LIMIT]:
        created = item.published_at.strftime("%Y-%m-%dT%H:%M:%SZ")
        user = clip(item.source, chars=SOURCE_CHARS, max_bytes=SOURCE_MAX_BYTES)
        body = clip(
            item.text.replace("\n", " ").strip(), chars=POST_CHARS, max_bytes=TEXT_MAX_BYTES
        )
        lines.append(f"[{created} · @{user} · no-label] {body}")
    total = len(lines)
    summary = (
        f"Bullish: 0 (0%) · Bearish: 0 (0%) · Unlabeled: {total} · "
        f"Total: {total} most-recent messages"
    )
    return summary + "\n\n" + "\n".join(lines)


def reddit_block(ticker: str, items: Sequence[TextItem]) -> str:
    """``reddit.py:264-334``: posts grouped by subreddit, a title line with its date, then a body
    excerpt."""
    by_sub: dict[str, list[TextItem]] = defaultdict(list)
    for item in items:
        sub = clip(item.source.removeprefix("r/"), chars=SOURCE_CHARS, max_bytes=SOURCE_MAX_BYTES)
        if sub not in by_sub and len(by_sub) >= REDDIT_SUBREDDITS:
            continue
        if len(by_sub[sub]) < REDDIT_PER_SUBREDDIT:
            by_sub[sub].append(item)
    if not by_sub:
        return f"<no Reddit posts found mentioning {ticker.upper()} in the past 7 days>"
    blocks = []
    for sub, posts in by_sub.items():
        lines = [f"r/{sub} — {len(posts)} recent posts mentioning {ticker.upper()}:"]
        for post in posts:
            title = clip(first_line(post.text), chars=TITLE_CHARS, max_bytes=TEXT_MAX_BYTES)
            rest = " ".join(post.text.split())[len(first_line(post.text)) :].strip()
            line = f"  [{_ymd(post.published_at)}] {title}"
            if rest:
                excerpt = clip(rest, chars=BODY_CHARS, max_bytes=TEXT_MAX_BYTES)
                line += f"\n    body excerpt: {excerpt}"
            lines.append(line)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def analyst_instructions(
    *, ticker: str, start: datetime, end: datetime, news: str, x_posts: str, reddit: str
) -> str:
    """``sentiment_analyst.py:140-192``, with X posts in the StockTwits block (module docstring)."""
    return f"""You are a financial market sentiment analyst. Your task is to produce a comprehensive sentiment report for {ticker} covering the period from {_ymd(start)} to {_ymd(end)}, drawing on three complementary data sources that have already been collected for you.

## Data sources (pre-fetched, in this prompt)

### News headlines — past 7 days
Institutional framing. Fact-driven, slower-moving signal.

<start_of_news>
{news}
<end_of_news>

### X posts — retail-trader social platform searched by cashtag
Fast-moving signal. X carries no user-labeled sentiment tag, so every post is marked no-label; read the message body.

<start_of_x_posts>
{x_posts}
<end_of_x_posts>

### Reddit posts — past 7 days
Community discussion. Engagement signal via upvote score and comment count where present. Subreddit character matters (r/wallstreetbets is often contrarian/exuberant; r/stocks more measured; r/investing longer-term).

<start_of_reddit>
{reddit}
<end_of_reddit>

## How to analyze this data (best practices)

1. **Read the balance of bullish and bearish X posts as a leading retail-sentiment signal.** A 70/30 bullish/bearish split is moderately bullish; ≥90/10 may indicate over-extension and contrarian risk; 50/50 is uncertainty. Sample size matters — base rates on the actual message count, not percentages alone.

2. **Look for cross-source divergences.** If news framing is bearish but X is overwhelmingly bullish, that mismatch is itself a signal — it can mean retail is leaning into a thesis the news flow hasn't caught up to (or vice versa, that retail is chasing while institutions are cautious).

3. **Weight Reddit posts by engagement.** A 400-upvote / 200-comment thread reflects community attention; a 3-upvote post is noise. Read the body excerpts for context — the title alone often misleads.

4. **Distinguish opinion from event.** A news headline ("Nvidia announces $500M Corning deal") is an event; an X post ("buying NVDA, this is going to moon") is opinion. Both are inputs but should be weighted differently in your conclusions.

5. **Identify recurring narrative themes.** What topic keeps coming up across sources? That's the dominant narrative driving current sentiment.

6. **Be honest about data limits.** If X returned only a handful of messages, or one or more sources returned a placeholder, the sentiment read is less robust — flag this explicitly in the `confidence` field and the narrative. If the sources are silent on a given subreddit, say so.

7. **Identify catalysts and risks** that emerge across sources — news of upcoming earnings, product launches, competitive threats, macro headlines, etc.

8. **Past sentiment is not predictive.** Frame your conclusions as signal for the trader to weigh alongside fundamentals and technicals, not as a price call."""  # noqa: E501


def analyst_messages(
    *, ticker: str, kind: Literal["crypto", "stock"], end: datetime, blocks: Mapping[str, str]
) -> list[ChatMessage]:
    start = seven_days_before(end)
    system = (
        "You are a helpful AI assistant, collaborating with other assistants."
        " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or"
        " deliverable, prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so"
        " the team knows to stop."
        f" Today's date is {_ymd(end)}; treat it as 'now' for all analysis."
        f" {instrument_context(ticker, kind)}"
        " "
        + NO_EXTERNAL_TOOLS
        + "\n"
        + analyst_instructions(
            ticker=ticker,
            start=start,
            end=end,
            news=blocks["news"],
            x_posts=blocks["x"],
            reddit=blocks["reddit"],
        )
        + ANALYST_OUTPUT
    )
    return [ChatMessage(role="system", content=system), ChatMessage(role="user", content=ticker)]


def trader_messages(
    *, ticker: str, kind: Literal["crypto", "stock"], plan: str
) -> list[ChatMessage]:
    """``trader.py:46-74`` with no market report."""
    system = (
        "You are a trading agent analyzing market data to make investment decisions. "
        "Based on your analysis, provide a specific recommendation to buy, sell, or hold. "
        "State entry price and stop-loss as absolute price levels in the "
        "instrument's quote currency (for example 189.5), never a percentage "
        "or a range; convert a percentage distance to the price level it "
        "implies, or omit the field if you cannot state a number. " + NO_EXTERNAL_TOOLS
    ) + TRADER_OUTPUT
    user = (
        f"Here is the research team's investment plan for {ticker}. "
        f"{instrument_context(ticker, kind)}\n\n"
        f"Proposed Investment Plan:\n{plan}\n\n"
        "Make an informed, strategic trading decision."
    )
    return [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)]


# ================================================================================================
# Output parsing
# ================================================================================================


@dataclass(frozen=True, slots=True)
class SentimentReport:
    overall_band: str
    overall_score: float
    confidence: Literal["low", "medium", "high"]
    narrative: str

    def render(self) -> str:
        """``render_sentiment_report``, ``schemas.py:355-368``."""
        return "\n".join(
            [
                f"**Overall Sentiment:** **{self.overall_band}** "
                f"(Score: {self.overall_score:.1f}/10)",
                f"**Confidence:** {self.confidence.capitalize()}",
                "",
                self.narrative,
            ]
        )


@dataclass(frozen=True, slots=True)
class TraderProposal:
    action: TraderAction
    reasoning: str
    entry_price: float | None
    stop_loss: float | None
    position_sizing: str | None


def _optional_price(value: Any) -> float | None:
    """``schemas.py:176-179``: a nullish or unreadable price is omitted rather than an error."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.replace(",", "").strip())
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


def parse_sentiment_report(content: str) -> SentimentReport | None:
    obj = load_json_object(content)
    if obj is None:
        return None
    band, score, confidence, narrative = (
        obj.get("overall_band"),
        obj.get("overall_score"),
        obj.get("confidence"),
        obj.get("narrative"),
    )
    bands = {b.lower(): b for b in SENTIMENT_BANDS}
    if not isinstance(band, str) or band.strip().lower() not in bands:
        return None
    if isinstance(score, bool) or not isinstance(score, int | float):
        return None
    if not math.isfinite(float(score)) or not 0.0 <= float(score) <= 10.0:
        return None
    if confidence not in ("low", "medium", "high") or not isinstance(narrative, str):
        return None
    return SentimentReport(bands[band.strip().lower()], float(score), confidence, narrative)


def parse_trader_proposal(content: str) -> TraderProposal | None:
    obj = load_json_object(content)
    if obj is None:
        return None
    action = obj.get("action")
    if not isinstance(action, str) or action.strip().capitalize() not in _ACTIONS:
        return None
    reasoning = obj.get("reasoning")
    sizing = obj.get("position_sizing")
    return TraderProposal(
        action=_as_action(action.strip().capitalize()),
        reasoning=reasoning if isinstance(reasoning, str) else "",
        entry_price=_optional_price(obj.get("entry_price")),
        stop_loss=_optional_price(obj.get("stop_loss")),
        position_sizing=sizing if isinstance(sizing, str) else None,
    )


def _as_action(word: str) -> TraderAction:
    for action in _ACTIONS:
        if action == word:
            return action
    raise ValueError(f"not a trader action: {word!r}")


_FINAL_PROPOSAL: Final = re.compile(r"FINAL TRANSACTION PROPOSAL:\s*\**\s*(BUY|HOLD|SELL)", re.I)
_ACTION_LABEL: Final = re.compile(r"action.*?[:\-][\s*]*(\w+)", re.I)
_ACTION_WORD: Final = re.compile(r"\b(Buy|Hold|Sell)\b", re.I)


def action_from_text(text: str) -> TraderAction | None:
    """The trader's action in free text: the ``FINAL TRANSACTION PROPOSAL`` line
    (``schemas.py:200-203``), else a labelled action, else the first standalone action word
    (``rating.py:45-66``, narrowed to Buy / Hold / Sell). ``None`` is REVIEW."""
    if not text:
        return None
    norm = unicodedata.normalize("NFKC", text)
    final = _FINAL_PROPOSAL.search(norm)
    if final:
        return _as_action(final.group(1).capitalize())
    for line in norm.splitlines():
        m = _ACTION_LABEL.search(line)
        if m and m.group(1).capitalize() in _ACTIONS:
            return _as_action(m.group(1).capitalize())
    word = _ACTION_WORD.search(norm)
    return _as_action(word.group(1).capitalize()) if word else None


def _report_for_trader(text: str) -> str:
    return clip(text, chars=REPORT_MAX_BYTES, max_bytes=REPORT_MAX_BYTES)


# ================================================================================================
# The arm
# ================================================================================================


@dataclass(frozen=True, slots=True)
class SymbolDecision:
    """What the chain concluded for one symbol at one snapshot."""

    symbol: str
    report: str | None
    action: TraderAction | None
    """``None`` is REVIEW: no usable action, so the position is held."""
    size: float | None
    weight: float


class TradingAgentsSocialArm:
    """TradingAgents' sentiment analyst and trader on a :class:`~sentiment_agent.types.ChatModel`
    (module docstring)."""

    SEED: Final = 20260924

    def __init__(
        self,
        *,
        model: ChatModel,
        policy: Policy,
        max_symbols: int = MAX_SYMBOLS,
        thinking: Thinking = Thinking.LOW,
    ) -> None:
        if max_symbols < 1:
            raise ValueError("max_symbols must be at least 1")
        self.spec = spec_for(TRADINGAGENTS)
        self._model = model
        self._policy = policy
        self._max_symbols = max_symbols
        self._thinking = thinking
        self.log = CallLog()
        self.decisions: list[SymbolDecision] = []

    # --- which symbols ------------------------------------------------------------------------

    def candidates(
        self, snapshot: PerceptionSnapshot, book: BookState
    ) -> tuple[list[str], dict[str, tuple[TextItem, ...]]]:
        """The symbols to run the chain on (held first, largest first, then those with the most
        texts in the window, then universe order, up to ``max_symbols``), and the window's texts
        per symbol."""
        texts = texts_by_symbol(snapshot, since=seven_days_before(snapshot.taken_at))
        held = current_weights(book)
        order = {s: i for i, s in enumerate(self._policy.symbols)}
        pool = [s for s in self._policy.symbols if s in held or texts.get(s)]
        pool.sort(
            key=lambda s: (s not in held, -abs(held.get(s, 0.0)), -len(texts.get(s, ())), order[s])
        )
        return pool[: self._max_symbols], texts

    # --- one symbol ---------------------------------------------------------------------------

    def _call(
        self, messages: Sequence[ChatMessage], *, json_mode: bool, max_tokens: int
    ) -> tuple[str | None, str, int]:
        completion, problem = ask(
            self._model,
            messages,
            json_mode=json_mode,
            max_tokens=max_tokens,
            thinking=self._thinking,
            temperature=0.0,
            seed=self.SEED,
        )
        if completion is None:
            return None, problem, 0
        return completion.content, problem, completion.usage.total_tokens

    def _record(
        self, snapshot_id: str, symbol: str, stage: str, ok: bool, detail: str, tokens: int
    ) -> None:
        self.log.calls.append(
            RivalCall(self.spec.arm_id, snapshot_id, symbol, stage, ok, detail, tokens)
        )

    def analyst_report(
        self, snapshot_id: str, symbol: str, messages: Sequence[ChatMessage]
    ) -> str | None:
        content, problem, tokens = self._call(
            messages, json_mode=True, max_tokens=ANALYST_MAX_TOKENS
        )
        report = None if content is None else parse_sentiment_report(content)
        self._record(
            snapshot_id,
            symbol,
            "analyst",
            report is not None,
            problem or ("" if report else "not a SentimentReport"),
            tokens,
        )
        if report is not None:
            return _report_for_trader(report.render())
        content, problem, tokens = self._call(
            messages, json_mode=False, max_tokens=ANALYST_MAX_TOKENS
        )
        ok = bool(content and content.strip())
        self._record(
            snapshot_id, symbol, "analyst_freetext", ok, problem or ("" if ok else "empty"), tokens
        )
        return _report_for_trader(content) if content and content.strip() else None

    def trader_action(
        self, snapshot_id: str, symbol: str, messages: Sequence[ChatMessage]
    ) -> tuple[TraderAction | None, str | None]:
        content, problem, tokens = self._call(
            messages, json_mode=True, max_tokens=TRADER_MAX_TOKENS
        )
        proposal = None if content is None else parse_trader_proposal(content)
        self._record(
            snapshot_id,
            symbol,
            "trader",
            proposal is not None,
            problem or ("" if proposal else "not a TraderProposal"),
            tokens,
        )
        if proposal is not None:
            return proposal.action, proposal.position_sizing
        content, problem, tokens = self._call(
            messages, json_mode=False, max_tokens=TRADER_MAX_TOKENS
        )
        action = None if content is None else action_from_text(content)
        self._record(
            snapshot_id,
            symbol,
            "trader_freetext",
            action is not None,
            problem or ("" if action else "REVIEW: no recognisable action"),
            tokens,
        )
        return action, None

    def size_of(self, position_sizing: str | None) -> float:
        cap = self._policy.per_name_max
        fraction = percent_in(position_sizing)
        if fraction is None:
            return cap
        return max(0.0, min(cap, fraction))

    def blocks(self, ticker: str, items: Sequence[TextItem], end: datetime) -> dict[str, str]:
        start = seven_days_before(end)
        news = [i for i in items if i.channel == "news"]
        posts = [i for i in items if i.channel == "x"]
        reddit = [i for i in items if i.channel == "reddit"]
        return {
            "news": news_block(ticker, news, start, end),
            "x": x_block(ticker, posts, start, end),
            "reddit": reddit_block(ticker, reddit),
        }

    def decide_symbol(
        self,
        snapshot: PerceptionSnapshot,
        symbol: str,
        items: Sequence[TextItem],
        current: float,
    ) -> SymbolDecision:
        ticker = display_ticker(symbol, self._policy)
        kind = asset_type(symbol, self._policy)
        messages = analyst_messages(
            ticker=ticker,
            kind=kind,
            end=snapshot.taken_at,
            blocks=self.blocks(ticker, items, snapshot.taken_at),
        )
        report = self.analyst_report(snapshot.snapshot_id, symbol, messages)
        if report is None:
            return SymbolDecision(symbol, None, None, None, current)
        action, sizing = self.trader_action(
            snapshot.snapshot_id, symbol, trader_messages(ticker=ticker, kind=kind, plan=report)
        )
        if action is None or action == "Hold":
            return SymbolDecision(symbol, report, action, None, current)
        size = self.size_of(sizing)
        weight = size if action == "Buy" else -size
        return SymbolDecision(symbol, report, action, size, weight)

    # --- the book -----------------------------------------------------------------------------

    def targets(self, snapshot: PerceptionSnapshot, book: BookState) -> dict[str, float]:
        chosen, texts = self.candidates(snapshot, book)
        held = current_weights(book)
        targets = dict(held)
        for symbol in chosen:
            decision = self.decide_symbol(
                snapshot, symbol, texts.get(symbol, ()), held.get(symbol, 0.0)
            )
            self.decisions.append(decision)
            targets[symbol] = decision.weight
        return cap_weights(targets, self._policy)

    # --- cost ---------------------------------------------------------------------------------

    def max_tokens_per_snapshot(self) -> int:
        """An upper bound on one snapshot's tokens: every chosen symbol's analyst and trader call
        with its free-text retry, each at its prompt's largest possible size in UTF-8 bytes plus
        its completion cap (the bound :func:`~sentiment_agent.llm.budget.projected_tokens` uses),
        for the universe symbol whose prompts are largest."""
        per_symbol = max(
            self._analyst_bound(symbol) + self._trader_bound(symbol)
            for symbol in self._policy.symbols
        )
        return self._max_symbols * ATTEMPTS_PER_STAGE * per_symbol

    def _analyst_bound(self, symbol: str) -> int:
        ticker, kind = display_ticker(symbol, self._policy), asset_type(symbol, self._policy)
        end = datetime(2026, 12, 31, 23, 59, tzinfo=UTC)
        widest = "\U0001d54f" * (TEXT_MAX_BYTES // 4)
        source = "\U0001d54f" * (SOURCE_MAX_BYTES // 4)

        def item(channel: Literal["x", "reddit", "news"], n: int, sub: str) -> TextItem:
            return TextItem(
                item_id=f"{channel}-{n}",
                channel=channel,
                source=sub,
                url=None,
                published_at=end,
                fetched_at=end,
                text=f"{widest}\n{widest}",
            )

        items = [
            *(item("news", n, source) for n in range(NEWS_LIMIT)),
            *(item("x", n, source) for n in range(X_LIMIT)),
            *(
                item("reddit", k * REDDIT_PER_SUBREDDIT + n, str(k) + source[1:])
                for k in range(REDDIT_SUBREDDITS)
                for n in range(REDDIT_PER_SUBREDDIT)
            ),
        ]
        messages = analyst_messages(
            ticker=ticker, kind=kind, end=end, blocks=self.blocks(ticker, items, end)
        )
        return projected_tokens(messages, ANALYST_MAX_TOKENS)

    def _trader_bound(self, symbol: str) -> int:
        ticker, kind = display_ticker(symbol, self._policy), asset_type(symbol, self._policy)
        plan = "\U0001d54f" * (REPORT_MAX_BYTES // 4)
        messages = trader_messages(ticker=ticker, kind=kind, plan=plan)
        return projected_tokens(messages, TRADER_MAX_TOKENS)


__all__ = [
    "ANALYST_MAX_TOKENS",
    "ANALYST_OUTPUT",
    "MAX_SYMBOLS",
    "NO_EXTERNAL_TOOLS",
    "REPORT_MAX_BYTES",
    "SENTIMENT_BANDS",
    "TRADER_MAX_TOKENS",
    "TRADER_OUTPUT",
    "SentimentReport",
    "SymbolDecision",
    "TraderAction",
    "TraderProposal",
    "TradingAgentsSocialArm",
    "action_from_text",
    "analyst_instructions",
    "analyst_messages",
    "clip",
    "display_ticker",
    "instrument_context",
    "news_block",
    "parse_sentiment_report",
    "parse_trader_proposal",
    "reddit_block",
    "trader_messages",
    "x_block",
]
