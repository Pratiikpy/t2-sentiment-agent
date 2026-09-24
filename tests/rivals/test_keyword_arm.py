"""The lexicon trader: VADER's rules reproduced exactly, the market vocabulary, the trading rule."""

from datetime import timedelta

import pytest

from rivals.rbuild import BTC, NVDA, T0, TSLA, book, item, snapshot
from sentiment_agent.policy import POLICY_V1
from sentiment_agent.rivals.keyword_arm import (
    BOOSTERS,
    GENERAL_VALENCES,
    LEXICON,
    MARKET_VALENCES,
    NEGATE,
    KeywordSentimentArm,
    compound,
    normalize,
    tokens_of,
)
from sentiment_agent.rivals.registry import LEXICON_FADE, LEXICON_FOLLOW, TextSentimentRule

# Compound scores produced by upstream VADER itself (cjhutto/vaderSentiment at commit
# 44fc044cd877310ee8278a0eadf34bcd50d41d06, SentimentIntensityAnalyzer().polarity_scores), on
# sentences whose only lexicon words are in GENERAL_VALENCES. The port must reproduce them exactly:
# they exercise negation, "never so", "without doubt", "no", "nor", "least", "but", capitals, a
# booster and the multi-word dampeners.
UPSTREAM_VADER = [
    ("This is not good", -0.3412),
    ("Good earnings but guidance was terrible", -0.4939),
    ("VERY bad news for $COIN", -0.6732),
    ("The results were only kind of good.", 0.3832),
    ("At least it isnt a horrible quarter.", 0.431),
    ("never so good", 0.5777),
    ("without doubt good", 0.6136),
    ("no good", -0.3412),
    ("nor bad", 0.431),
    ("it was sort of bad news", -0.5849),
]


@pytest.mark.parametrize(("text", "expected"), UPSTREAM_VADER)
def test_the_port_reproduces_upstream_vader(text: str, expected: float) -> None:
    assert compound(text) == pytest.approx(expected, abs=1e-4)


def test_general_valences_are_vaders_own() -> None:
    # A sample checked against vaderSentiment/vader_lexicon.txt at the commit above.
    assert GENERAL_VALENCES["good"] == 1.9
    assert GENERAL_VALENCES["great"] == 3.1
    assert GENERAL_VALENCES["crash"] == -1.7
    assert GENERAL_VALENCES["greed"] == -1.7
    assert GENERAL_VALENCES["scam"] == -2.7
    assert GENERAL_VALENCES["solid"] == 0.6
    assert len(NEGATE) == 59
    assert BOOSTERS["very"] == pytest.approx(0.293)
    assert BOOSTERS["kind of"] == pytest.approx(-0.293)


def test_the_market_vocabulary_only_adds_words_vader_lacks() -> None:
    assert not set(GENERAL_VALENCES) & set(MARKET_VALENCES)
    assert LEXICON == {**GENERAL_VALENCES, **MARKET_VALENCES}
    assert all(-4.0 <= v <= 4.0 for v in LEXICON.values())


def test_normalize_is_vaders() -> None:
    assert normalize(0.0) == 0.0
    assert normalize(1.9) == pytest.approx(1.9 / (1.9**2 + 15) ** 0.5)
    assert normalize(1e9) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("text", "sign"),
    [
        ("$NVDA extremely bullish, calls printing", 1),
        ("BTC to the moon!!! \U0001f680\U0001f680", 1),
        ("NVDA beats estimates and upgraded by analysts", 1),
        ("TSLA plunges after the recall, bagholders rekt", -1),
        ("Massive selloff, liquidations everywhere \U0001f4c9", -1),
        ("I am not bullish on $COIN at all", -1),
        ("The company is headquartered in Austin.", 0),
    ],
)
def test_market_text_gets_the_sign_a_trader_would_give_it(text: str, sign: int) -> None:
    score = compound(text)
    if sign == 0:
        assert score == 0.0
    else:
        assert score * sign > 0.3


def test_emoji_are_split_into_their_own_tokens() -> None:
    assert tokens_of("NVDA\U0001f680\U0001f680 now") == ["NVDA", "\U0001f680", "\U0001f680", "now"]
    assert tokens_of("") == []
    assert compound("") == 0.0


BULLISH = ["$NVDA extremely bullish", "$NVDA rally, calls printing", "$NVDA beats, great quarter"]


def test_the_follow_arm_goes_long_what_the_crowd_loves_and_the_fade_arm_short() -> None:
    snap = snapshot([item(t) for t in BULLISH])
    follow = KeywordSentimentArm(policy=POLICY_V1)
    fade = KeywordSentimentArm(policy=POLICY_V1, direction="fade")
    assert follow.spec.arm_id == LEXICON_FOLLOW
    assert fade.spec.arm_id == LEXICON_FADE
    assert follow.targets(snap, book()) == {NVDA: pytest.approx(0.05)}
    assert fade.targets(snap, book()) == {NVDA: pytest.approx(-0.05)}
    assert all(s > 0 for s in follow.scores(snap)[NVDA])


def test_the_arm_needs_three_texts_holds_otherwise_and_ignores_stale_text() -> None:
    arm = KeywordSentimentArm(policy=POLICY_V1)
    two = snapshot([item(t) for t in BULLISH[:2]])
    assert arm.targets(two, book()) == {}
    assert arm.targets(two, book(weights={NVDA: -0.02})) == {NVDA: pytest.approx(-0.02)}
    stale = snapshot([item(t, at=T0 - timedelta(hours=25)) for t in BULLISH])
    assert arm.targets(stale, book()) == {}


def test_neutral_chatter_flattens_and_every_named_symbol_is_scored() -> None:
    arm = KeywordSentimentArm(policy=POLICY_V1)
    neutral = snapshot(
        [item("$TSLA earnings call on Tuesday"), item("$TSLA volume today"), item("$TSLA chart")]
    )
    assert arm.targets(neutral, book(weights={TSLA: 0.03})) == {}
    bitcoin = snapshot(
        [
            item("bitcoin crashing, rekt"),
            item("bitcoin bleeding again"),
            item("bearish on bitcoin, selling"),
        ]
    )
    targets = arm.targets(bitcoin, book())
    assert targets[BTC] < 0


def test_a_rule_that_disagrees_with_the_arms_direction_is_refused() -> None:
    with pytest.raises(ValueError, match="direction"):
        KeywordSentimentArm(policy=POLICY_V1, rule=TextSentimentRule("fade"))
    with pytest.raises(ValueError, match="lookback"):
        KeywordSentimentArm(policy=POLICY_V1, lookback=timedelta(0))
