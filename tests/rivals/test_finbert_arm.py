"""The finBERT trader: sentence handling and the trading rule with a stand-in scorer (always), and
the real ProsusAI/finbert on a tiny labelled set (only when the ``[rivals]`` extra and the pinned
weights are available offline; nothing is ever downloaded in a test)."""

from collections.abc import Iterator, Sequence

import pytest

from rivals.rbuild import NVDA, TSLA, book, item, snapshot
from sentiment_agent.policy import POLICY_V1
from sentiment_agent.rivals.finbert_arm import (
    FINBERT_REVISION,
    FinbertArm,
    FinbertScorer,
    FinbertUnavailable,
    finbert_installed,
    split_sentences,
)
from sentiment_agent.rivals.registry import FINBERT_FADE, FINBERT_FOLLOW


class KeywordScorer:
    """A stand-in for finBERT: +0.8 for a sentence containing "up", -0.8 for "down", else 0."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, sentences: Sequence[str]) -> list[float]:
        self.calls.append(list(sentences))
        return [0.8 if "up" in s else -0.8 if "down" in s else 0.0 for s in sentences]


def test_sentences_split_on_ends_and_line_breaks() -> None:
    assert split_sentences("Sales up. Margins down!  Guidance?\nNew line") == [
        "Sales up.",
        "Margins down!",
        "Guidance?",
        "New line",
    ]
    assert split_sentences("  \n ") == []
    assert split_sentences("v2.0 ships today") == ["v2.0 ships today"]


def test_an_item_scores_the_mean_of_its_sentences_in_one_batched_call() -> None:
    scorer = KeywordScorer()
    arm = FinbertArm(policy=POLICY_V1, scorer=scorer)
    assert arm.score_texts(["$NVDA up. flat.", "down", "   "]) == [
        pytest.approx(0.4),
        pytest.approx(-0.8),
        None,
    ]
    assert scorer.calls == [["$NVDA up.", "flat.", "down"]]


def test_the_arm_trades_the_mean_score_with_or_against_the_crowd() -> None:
    snap = snapshot([item(f"$NVDA going up {n}") for n in range(3)])
    follow = FinbertArm(policy=POLICY_V1, scorer=KeywordScorer())
    fade = FinbertArm(policy=POLICY_V1, direction="fade", scorer=KeywordScorer())
    assert follow.spec.arm_id == FINBERT_FOLLOW
    assert fade.spec.arm_id == FINBERT_FADE
    assert follow.targets(snap, book()) == {NVDA: pytest.approx(0.05)}
    assert fade.targets(snap, book()) == {NVDA: pytest.approx(-0.05)}


def test_each_text_is_scored_once_even_when_it_names_two_symbols() -> None:
    scorer = KeywordScorer()
    arm = FinbertArm(policy=POLICY_V1, scorer=scorer)
    snap = snapshot([item(f"$NVDA and $TSLA down {n}") for n in range(3)])
    assert arm.targets(snap, book()) == {NVDA: pytest.approx(-0.05), TSLA: pytest.approx(-0.05)}
    assert len(scorer.calls) == 1
    assert len(scorer.calls[0]) == 3


def test_a_scorer_that_loses_sentences_is_refused() -> None:
    arm = FinbertArm(policy=POLICY_V1, scorer=lambda sentences: [0.0])
    with pytest.raises(ValueError, match="scores for"):
        arm.score_texts(["one.", "two."])


def test_without_the_extra_the_scorer_says_what_to_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sentiment_agent.rivals.finbert_arm.finbert_installed", lambda: False)
    with pytest.raises(FinbertUnavailable, match=r"\[rivals\] extra"):
        FinbertScorer()(["anything"])
    assert FinbertScorer()([]) == []


# --- the real model -------------------------------------------------------------------------

# A tiny labelled set in the style of the Financial PhraseBank (written for this test; the
# PhraseBank itself is CC BY-NC-SA and is not vendored). Labels are the obvious reading.
LABELLED = [
    ("Operating profit rose to EUR 13.1 mn from EUR 8.7 mn in the corresponding period.", 1),
    ("Net sales increased by 25% and the company raised its full-year guidance.", 1),
    ("The company reported a net loss and cut its full-year guidance.", -1),
    ("Sales fell sharply and the company announced layoffs.", -1),
    ("The company is headquartered in Helsinki.", 0),
    ("The annual general meeting will be held on 15 April.", 0),
]


@pytest.fixture(scope="module")
def real_scorer() -> Iterator[FinbertScorer]:
    if not finbert_installed():
        pytest.skip("the optional [rivals] extra (transformers, torch) is not installed")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("HF_HUB_OFFLINE", "1")
        patch.setenv("TRANSFORMERS_OFFLINE", "1")
        scorer = FinbertScorer(local_files_only=True)
        try:
            scorer(["warm up"])
        except FinbertUnavailable as exc:
            pytest.skip(f"ProsusAI/finbert@{FINBERT_REVISION} is not available offline: {exc}")
        yield scorer


def test_finbert_scores_a_labelled_set_with_the_right_signs(real_scorer: FinbertScorer) -> None:
    scores = real_scorer([text for text, _ in LABELLED])
    for (text, label), score in zip(LABELLED, scores, strict=True):
        assert -1.0 <= score <= 1.0
        if label > 0:
            assert score > 0.5, text
        elif label < 0:
            assert score < -0.5, text
        else:
            assert abs(score) < 0.5, text


def test_the_real_finbert_arm_trades_a_snapshot(real_scorer: FinbertScorer) -> None:
    arm = FinbertArm(policy=POLICY_V1, scorer=real_scorer)
    bad = [
        item("$TSLA reported a net loss and cut its guidance."),
        item("$TSLA sales fell sharply and it announced layoffs."),
        item("$TSLA shares plunged after the profit warning."),
    ]
    targets = arm.targets(snapshot(bad), book())
    assert targets[TSLA] < 0
