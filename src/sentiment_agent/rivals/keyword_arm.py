# Portions of this file are adapted from VADER (vaderSentiment), https://github.com/cjhutto/vaderSentiment
# at commit 44fc044cd877310ee8278a0eadf34bcd50d41d06: the scoring rules of
# vaderSentiment/vaderSentiment.py (constants :26-31, negations :33-41, boosters :48-70, negated()
# :87-101, normalize() :104-115, allcap_differential() :118-132, scalar_inc_dec() :135-152, token
# handling :155-198, polarity_scores() and its helpers :239-509) and the valences of the general
# English words in GENERAL_VALENCES, copied from vaderSentiment/vader_lexicon.txt. VADER's licence:
#
#   The MIT License (MIT)
#
#   Copyright (c) 2016 C.J. Hutto
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy of this software
#   and associated documentation files (the "Software"), to deal in the Software without
#   restriction, including without limitation the rights to use, copy, modify, merge, publish,
#   distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
#   Software is furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all copies or
#   substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
#   BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#   DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""The lexicon sentiment trader: VADER's rules on a market-aware lexicon, traded per symbol.

Why VADER, and why not VADER alone
----------------------------------
VADER (Hutto and Gilbert, ICWSM 2014) is the standard rule-based scorer for social-media text: a
valence lexicon plus rules for negation ("not good"), degree words ("very", "slightly"), capitals
("GREAT"), contrast ("but") and exclamation. It is the strongest open lexicon method for short
posts, which is why it is the rival here rather than a hand-rolled word count. But it is a
general-English lexicon, and the words a market crowd uses to say what it believes are missing
from it: *bullish, bearish, rally, surge, plunge, upgrade, downgrade, beats, moon, rekt, selloff*
are all absent from ``vader_lexicon.txt`` (checked at the commit above). A VADER trader would
read "$NVDA extremely bullish, calls printing" as nearly neutral. So the rival uses VADER's rules,
VADER's own valences for the general words below (:data:`GENERAL_VALENCES`, copied exactly), and
a market vocabulary on the same -4..+4 scale (:data:`MARKET_VALENCES`, written for this rival
before any result was seen, never tuned). The market words are only words VADER lacks; a test
enforces that the two tables never disagree.

What was ported, and what changed
---------------------------------
Ported: the tokenisation (split on whitespace, strip punctuation from words longer than two
characters), the "no" rules, capital emphasis (+0.733 when some but not all tokens are capitals),
boosters and dampeners within three words with distance decay (0.95, 0.9), negation within three
words (x -0.74, with "never so/this" and "without doubt"), the "least" rule, the "but" rule
(x0.5 before, x1.5 after), exclamation and question-mark emphasis, and the normalisation
``s / sqrt(s^2 + 15)`` rounded to four places.

Changed, and why:

* The "but" rule is applied by position. Upstream re-finds each value with ``list.index``
  (``vaderSentiment.py:339-352``), so two words with the same valence on either side of "but" are
  both treated as the first one; by position is what the rule describes.
* Emoji are scored directly, by the valence the market vocabulary gives the few that carry a market
  view (rocket, chart up, chart down, skull...), instead of VADER's emoji-to-description step,
  whose descriptions ("rocket") are not in its lexicon anyway.
* The special idioms ("the bomb", "kiss of death") are not ported: they are general-English idioms
  with no market reading. The multi-word dampeners in the same function ("kind of", "sort of",
  ``vaderSentiment.py:386-390``) are, since they are degree words, not idioms.

The trading rule
----------------
Every text naming a symbol within the last 24 hours is scored; the per-symbol mean becomes a
weight under :class:`~sentiment_agent.rivals.registry.TextSentimentRule` (at least 3 texts, flat
inside +-0.05, full size at +-0.5), with the crowd for
:data:`~sentiment_agent.rivals.registry.LEXICON_FOLLOW` and against it for
:data:`~sentiment_agent.rivals.registry.LEXICON_FADE`. Each post counts once,
copies included: a lexicon trader does not know what a coordinated campaign is, and the red team
measures what that costs it.
"""

import math
import re
import string
from collections.abc import Mapping, Sequence
from datetime import timedelta
from typing import Final

from sentiment_agent.rivals.registry import (
    LEXICON_FADE,
    LEXICON_FOLLOW,
    Direction,
    TextSentimentRule,
    spec_for,
    texts_by_symbol,
    weights_from_scores,
)
from sentiment_agent.types import BookState, PerceptionSnapshot, Policy

# --- VADER constants (vaderSentiment.py:26-31) ---------------------------------------------------

B_INCR: Final = 0.293
B_DECR: Final = -0.293
C_INCR: Final = 0.733
N_SCALAR: Final = -0.74
NORMALIZE_ALPHA: Final = 15
EXCLAMATION_STEP: Final = 0.292
EXCLAMATION_MAX: Final = 4
QUESTION_STEP: Final = 0.18
QUESTION_MAX_AMPLIFIER: Final = 0.96

NEGATE: Final[frozenset[str]] = frozenset(
    {
        "aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
        "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
        "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither",
        "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
        "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere",
        "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent",
        "oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't",
        "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite",
    }
)  # fmt: skip
"""``vaderSentiment.py:33-41``."""

BOOSTERS: Final[Mapping[str, float]] = {
    **dict.fromkeys(
        (
            "absolutely", "amazingly", "awfully", "completely", "considerable", "considerably",
            "decidedly", "deeply", "effing", "enormous", "enormously", "entirely", "especially",
            "exceptional", "exceptionally", "extreme", "extremely", "fabulously", "flipping",
            "flippin", "frackin", "fracking", "fricking", "frickin", "frigging", "friggin", "fully",
            "fuckin", "fucking", "fuggin", "fugging", "greatly", "hella", "highly", "hugely",
            "incredible", "incredibly", "intensely", "major", "majorly", "more", "most",
            "particularly", "purely", "quite", "really", "remarkably", "so", "substantially",
            "thoroughly", "total", "totally", "tremendous", "tremendously", "uber", "unbelievably",
            "unusually", "utter", "utterly", "very",
        ),
        B_INCR,
    ),
    **dict.fromkeys(
        (
            "almost", "barely", "hardly", "just enough", "kind of", "kinda", "kindof", "kind-of",
            "less", "little", "marginal", "marginally", "occasional", "occasionally", "partly",
            "scarce", "scarcely", "slight", "slightly", "somewhat", "sort of", "sorta", "sortof",
            "sort-of",
        ),
        B_DECR,
    ),
}  # fmt: skip
"""``vaderSentiment.py:48-70``."""

# --- The lexicon -----------------------------------------------------------------------------

GENERAL_VALENCES: Final[Mapping[str, float]] = {
    "good": 1.9, "great": 3.1, "excellent": 2.7, "amazing": 2.8, "awesome": 3.1, "best": 3.2,
    "better": 1.9, "love": 3.2, "win": 2.8, "wins": 2.7, "winning": 2.4, "winner": 2.8,
    "winners": 2.1, "gain": 2.4, "gains": 1.4, "profit": 1.9, "profits": 1.9, "profitable": 1.9,
    "strong": 2.3, "stronger": 1.6, "strength": 2.2, "growth": 1.6, "happy": 2.7, "glad": 2.0,
    "hope": 1.9, "hopeful": 2.3, "optimistic": 1.3, "optimism": 2.5, "confident": 2.2,
    "confidence": 2.3, "positive": 2.6, "opportunity": 1.8, "opportunities": 1.6, "safe": 1.9,
    "rich": 2.6, "wealthy": 1.5, "nice": 1.8, "solid": 0.6, "huge": 1.3, "wow": 2.8, "yay": 2.4,
    "lol": 1.8, "lmao": 2.9, "exciting": 2.2, "excited": 1.4, "brilliant": 2.8, "fantastic": 2.6,
    "smart": 1.7, "success": 2.7, "successful": 2.8, "successfully": 2.2, "boost": 1.7,
    "boosted": 1.5, "boosts": 1.3, "benefit": 2.0, "benefits": 1.6, "improve": 1.9,
    "improved": 2.1, "improvement": 2.0, "impressive": 2.3, "outstanding": 3.0, "superb": 3.1,
    "perfect": 2.7, "easy": 1.9, "fun": 2.3, "thrilled": 1.9, "bad": -2.5, "worse": -2.1,
    "worst": -3.1, "terrible": -2.1, "awful": -2.0, "horrible": -2.5, "weak": -1.9,
    "weakness": -1.8, "fail": -2.5, "failed": -2.3, "failure": -2.3, "fails": -1.8,
    "failing": -2.3, "loss": -1.3, "losses": -1.7, "lose": -1.7, "losing": -1.6, "loser": -2.4,
    "losers": -2.4, "crash": -1.7, "dump": -1.6, "dumping": -1.3, "dumped": -1.7, "scam": -2.7,
    "scams": -2.8, "fraud": -2.8, "hacked": -1.7, "bankrupt": -2.6, "lawsuit": -0.9, "fear": -2.2,
    "fears": -1.8, "panic": -2.3, "greed": -1.7, "pessimistic": -1.5, "risk": -1.1, "risks": -1.1,
    "risky": -0.8, "warning": -1.4, "worried": -1.2, "worry": -1.9, "worries": -1.8,
    "disappointed": -2.1, "disappointing": -2.2, "disappointment": -2.3, "collapse": -2.2,
    "collapsed": -1.1, "recession": -1.8, "fud": -1.1, "sucks": -1.5, "debt": -1.5, "delay": -1.3,
    "delayed": -0.9, "fired": -2.6, "cut": -1.1, "cuts": -1.2, "miss": -0.6, "missed": -1.2,
    "misses": -0.9, "trap": -1.3, "dead": -3.3, "negative": -2.7, "doubt": -1.5, "doubts": -1.2,
    "uncertain": -1.2, "uncertainty": -1.4, "scared": -1.9, "nervous": -1.1, "disaster": -3.1,
    "catastrophe": -3.4, "crisis": -3.1, "threat": -2.4, "danger": -2.4, "dangerous": -2.1,
    "upset": -1.6, "angry": -2.3, "sad": -2.1, "stupid": -2.4, "dumb": -2.3, "poor": -2.1,
    "broke": -1.8, "destroyed": -2.2, "killed": -3.5, "trouble": -1.7, "problem": -1.7,
    "problems": -1.7, "struggle": -1.3, "struggling": -1.8, "drop": -1.1, "falling": -0.6,
    "hurt": -2.4, "hurts": -2.1, "ugly": -2.3, "useless": -1.8, "worthless": -1.9, "no": -1.2,
    "yes": 1.7,
}  # fmt: skip
"""VADER's own valences (``vader_lexicon.txt``, mean human rating on -4..+4) for the general words
market chatter uses most. Copied exactly; ``tests/rivals/test_keyword_arm.py`` pins a sample."""

MARKET_VALENCES: Final[Mapping[str, float]] = {
    # Views and stances.
    "bullish": 2.5, "bearish": -2.5, "bull": 1.5, "bulls": 1.5, "bear": -1.5, "bears": -1.5,
    "buy": 1.5, "buying": 1.5, "bought": 1.0, "long": 0.8, "calls": 1.0, "hodl": 1.5,
    "accumulate": 1.2, "accumulating": 1.2, "undervalued": 1.8,
    "sell": -1.5, "selling": -1.5, "sold": -1.0, "short": -0.8, "shorts": -0.8, "shorting": -1.5,
    "puts": -1.0, "overvalued": -1.8, "overpriced": -1.8, "bubble": -1.5, "bagholder": -2.0,
    "bagholders": -2.0,
    # Price action.
    "moon": 2.5, "mooning": 2.8, "moonshot": 2.0, "rocket": 2.0, "pump": 1.2, "pumping": 1.5,
    "pumped": 1.2, "rally": 2.0, "rallies": 2.0, "rallied": 2.0, "rallying": 2.0, "surge": 2.0,
    "surges": 2.0, "surged": 2.0, "surging": 2.2, "soar": 2.3, "soars": 2.3, "soared": 2.3,
    "soaring": 2.3, "breakout": 1.8, "rebound": 1.5, "rebounds": 1.5, "rebounded": 1.5,
    "recover": 1.3, "recovering": 1.3, "recovery": 1.5, "uptrend": 1.8, "ath": 2.0, "green": 1.2,
    "squeeze": 1.0, "record": 0.8,
    "plunge": -2.3, "plunges": -2.3, "plunged": -2.3, "plunging": -2.3, "plummet": -2.5,
    "plummets": -2.5, "plummeted": -2.5, "slump": -2.0, "slumps": -2.0, "slumped": -2.0,
    "tank": -1.8, "tanks": -1.8, "tanked": -1.8, "tanking": -2.0, "crashed": -2.0,
    "crashing": -2.2, "selloff": -2.0, "sell-off": -2.0, "downtrend": -1.8, "decline": -1.5,
    "declines": -1.5, "declined": -1.5, "declining": -1.5, "dropped": -1.2, "drops": -1.1,
    "fell": -1.2, "dip": -0.5, "correction": -1.2, "red": -1.0, "bleed": -1.8, "bleeding": -2.0,
    "capitulation": -2.0, "liquidation": -1.8, "liquidations": -1.8, "liquidated": -2.0,
    "rekt": -2.5,
    # Company and market news.
    "beat": 1.6, "beats": 1.6, "upgrade": 1.8, "upgraded": 1.8, "upgrades": 1.8,
    "outperform": 1.8, "outperforms": 1.8, "outperformed": 1.8, "adoption": 1.2, "approval": 1.5,
    "approved": 1.5,
    "downgrade": -1.8, "downgraded": -1.8, "downgrades": -1.8, "underperform": -1.8,
    "underperforms": -1.8, "layoffs": -2.0, "bankruptcy": -3.0, "probe": -1.2,
    "investigation": -1.2, "recall": -1.2, "halt": -1.5, "halted": -1.5, "halts": -1.5,
    "hack": -2.0, "exploit": -2.0, "rug": -2.5, "rugged": -2.8, "rugpull": -3.0,
    "concerned": -1.2, "trash": -2.0, "garbage": -2.0, "rip": -1.0,
    # Emoji that carry a market view.
    "\U0001f680": 2.0, "\U0001f4c8": 1.5, "\U0001f48e": 1.2, "\U0001f525": 1.0, "\U0001f402": 1.5,
    "\U0001f4b0": 1.5, "\U0001f4c9": -1.5, "\U0001f480": -1.5, "\U0001fa78": -1.8,
    "\U0001f43b": -1.5, "\U0001f53b": -1.2,
}  # fmt: skip
"""Market vocabulary VADER lacks, on its scale, written for this rival and never tuned. The emoji
are rocket, chart up, gem, fire, ox, money bag (positive) and chart down, skull, drop of blood,
bear, red triangle down (negative)."""

LEXICON: Final[Mapping[str, float]] = {**GENERAL_VALENCES, **MARKET_VALENCES}

_EMOJI: Final = re.compile(
    "(" + "|".join(re.escape(k) for k in MARKET_VALENCES if not k.isascii()) + ")"
)


# --- The scorer ------------------------------------------------------------------------------


def _strip_punc_if_word(token: str) -> str:
    """``vaderSentiment.py:178-188``: strip leading and trailing punctuation unless what is left is
    two characters or fewer (then it was likely an emoticon)."""
    stripped = token.strip(string.punctuation)
    return token if len(stripped) <= 2 else stripped


def tokens_of(text: str) -> list[str]:
    """Words and emoticons as VADER sees them, with market emoji split out as their own tokens."""
    spaced = _EMOJI.sub(r" \1 ", text)
    return [_strip_punc_if_word(t) for t in spaced.split()]


def _negated(word: str) -> bool:
    """``vaderSentiment.py:87-101`` for one word."""
    lowered = word.lower()
    return lowered in NEGATE or "n't" in lowered


def normalize(score: float, alpha: float = NORMALIZE_ALPHA) -> float:
    """``vaderSentiment.py:104-115``."""
    norm = score / math.sqrt(score * score + alpha)
    return max(-1.0, min(1.0, norm))


def _allcap_differential(words: Sequence[str]) -> bool:
    """``vaderSentiment.py:118-132``: some, but not all, tokens are in capitals."""
    allcaps = sum(1 for w in words if w.isupper())
    return 0 < len(words) - allcaps < len(words)


def _scalar_inc_dec(word: str, valence: float, is_cap_diff: bool) -> float:
    """``vaderSentiment.py:135-152``."""
    scalar = BOOSTERS.get(word.lower(), 0.0)
    if scalar == 0.0:
        return 0.0
    if valence < 0:
        scalar *= -1
    if word.isupper() and is_cap_diff:
        scalar += C_INCR if valence > 0 else -C_INCR
    return scalar


def _negation_check(valence: float, lowered: Sequence[str], start_i: int, i: int) -> float:
    """``vaderSentiment.py:408-433``."""
    if start_i == 0:
        if _negated(lowered[i - 1]):
            valence *= N_SCALAR
    elif start_i == 1:
        if lowered[i - 2] == "never" and lowered[i - 1] in ("so", "this"):
            valence *= 1.25
        elif lowered[i - 2] == "without" and lowered[i - 1] == "doubt":
            pass
        elif _negated(lowered[i - 2]):
            valence *= N_SCALAR
    elif start_i == 2:
        if (lowered[i - 3] == "never" and lowered[i - 2] in ("so", "this")) or lowered[i - 1] in (
            "so",
            "this",
        ):
            valence *= 1.25
        elif lowered[i - 3] == "without" and "doubt" in (lowered[i - 2], lowered[i - 1]):
            pass
        elif _negated(lowered[i - 3]):
            valence *= N_SCALAR
    return valence


def _least_check(valence: float, lowered: Sequence[str], i: int) -> float:
    """``vaderSentiment.py:327-337``."""
    if i > 1 and lowered[i - 1] not in LEXICON and lowered[i - 1] == "least":
        if lowered[i - 2] not in ("at", "very"):
            valence *= N_SCALAR
    elif i > 0 and lowered[i - 1] not in LEXICON and lowered[i - 1] == "least":
        valence *= N_SCALAR
    return valence


def _valence(tokens: Sequence[str], lowered: Sequence[str], i: int, is_cap_diff: bool) -> float:
    """``vaderSentiment.py:284-325``, with the multi-word dampeners but not the special idioms of
    ``:355-392`` (module docstring)."""
    item, word = tokens[i], lowered[i]
    base = LEXICON.get(word)
    if base is None:
        return 0.0
    valence = base
    last = len(tokens) - 1
    if word == "no" and i != last and lowered[i + 1] in LEXICON:
        valence = 0.0
    if (
        (i > 0 and lowered[i - 1] == "no")
        or (i > 1 and lowered[i - 2] == "no")
        or (i > 2 and lowered[i - 3] == "no" and lowered[i - 1] in ("or", "nor"))
    ):
        valence = base * N_SCALAR
    if item.isupper() and is_cap_diff:
        valence += C_INCR if valence > 0 else -C_INCR
    for start_i in range(3):
        if i > start_i and lowered[i - (start_i + 1)] not in LEXICON:
            s = _scalar_inc_dec(tokens[i - (start_i + 1)], valence, is_cap_diff)
            if start_i == 1 and s != 0:
                s *= 0.95
            if start_i == 2 and s != 0:
                s *= 0.9
            valence += s
            valence = _negation_check(valence, lowered, start_i, i)
            if start_i == 2:
                valence = _multiword_boosters(valence, lowered, i)
    return _least_check(valence, lowered, i)


def _multiword_boosters(valence: float, lowered: Sequence[str], i: int) -> float:
    """``vaderSentiment.py:386-390``: a booster or dampener of two or three words ("kind of",
    "sort of", "just enough") in the three words before ``i``. Called only with ``i >= 3``."""
    three_two_one = f"{lowered[i - 3]} {lowered[i - 2]} {lowered[i - 1]}"
    three_two = f"{lowered[i - 3]} {lowered[i - 2]}"
    two_one = f"{lowered[i - 2]} {lowered[i - 1]}"
    for n_gram in (three_two_one, three_two, two_one):
        if n_gram in BOOSTERS:
            valence += BOOSTERS[n_gram]
    return valence


def _but_check(lowered: Sequence[str], sentiments: list[float]) -> list[float]:
    """``vaderSentiment.py:339-352``, applied by position (module docstring)."""
    if "but" not in lowered:
        return sentiments
    bi = lowered.index("but")
    return [s * 0.5 if si < bi else s * 1.5 if si > bi else s for si, s in enumerate(sentiments)]


def _punctuation_emphasis(text: str) -> float:
    """``vaderSentiment.py:435-466``."""
    exclamation = min(text.count("!"), EXCLAMATION_MAX) * EXCLAMATION_STEP
    questions = text.count("?")
    question = 0.0
    if questions > 1:
        question = questions * QUESTION_STEP if questions <= 3 else QUESTION_MAX_AMPLIFIER
    return exclamation + question


def compound(text: str) -> float:
    """VADER's compound score of ``text`` in [-1, 1] on :data:`LEXICON`, rounded to four places as
    ``polarity_scores`` returns it (``vaderSentiment.py:239-264``, ``:482-509``)."""
    tokens = tokens_of(text)
    if not tokens:
        return 0.0
    lowered = [t.lower() for t in tokens]
    is_cap_diff = _allcap_differential(tokens)
    sentiments: list[float] = []
    for i, word in enumerate(lowered):
        if word in BOOSTERS:
            sentiments.append(0.0)
            continue
        if i < len(lowered) - 1 and word == "kind" and lowered[i + 1] == "of":
            sentiments.append(0.0)
            continue
        sentiments.append(_valence(tokens, lowered, i, is_cap_diff))
    sentiments = _but_check(lowered, sentiments)
    total = sum(sentiments)
    emphasis = _punctuation_emphasis(text)
    if total > 0:
        total += emphasis
    elif total < 0:
        total -= emphasis
    return round(normalize(total), 4)


# --- The arm ---------------------------------------------------------------------------------

LOOKBACK: Final = timedelta(hours=24)


class KeywordSentimentArm:
    """The lexicon sentiment trader (module docstring). ``direction`` picks the registered arm:
    ``"follow"`` is :data:`~sentiment_agent.rivals.registry.LEXICON_FOLLOW`, ``"fade"`` is
    :data:`~sentiment_agent.rivals.registry.LEXICON_FADE`."""

    def __init__(
        self,
        *,
        policy: Policy,
        direction: Direction = "follow",
        rule: TextSentimentRule | None = None,
        lookback: timedelta = LOOKBACK,
    ) -> None:
        self._rule = rule if rule is not None else TextSentimentRule(direction=direction)
        if self._rule.direction != direction:
            raise ValueError("the rule's direction must be the arm's")
        if lookback <= timedelta(0):
            raise ValueError("lookback must be positive")
        self.spec = spec_for(LEXICON_FOLLOW if direction == "follow" else LEXICON_FADE)
        self._policy = policy
        self._lookback = lookback

    @staticmethod
    def score(text: str) -> float:
        return compound(text)

    def scores(self, snapshot: PerceptionSnapshot) -> dict[str, list[float]]:
        """Every scored text per symbol, most recent first."""
        texts = texts_by_symbol(snapshot, since=snapshot.taken_at - self._lookback)
        return {s: [compound(i.text) for i in items] for s, items in texts.items()}

    def targets(self, snapshot: PerceptionSnapshot, book: BookState) -> dict[str, float]:
        return weights_from_scores(
            self.scores(snapshot), book=book, rule=self._rule, policy=self._policy
        )


__all__ = [
    "BOOSTERS",
    "GENERAL_VALENCES",
    "LEXICON",
    "LOOKBACK",
    "MARKET_VALENCES",
    "NEGATE",
    "KeywordSentimentArm",
    "compound",
    "normalize",
    "tokens_of",
]
