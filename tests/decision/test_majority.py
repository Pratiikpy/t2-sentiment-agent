"""``decision/majority.py`` in isolation: the side classifier, the 2-of-3 side vote, and the
median weight rule, tested directly against :class:`~sentiment_agent.types.LlmDecision` objects
rather than through a full :class:`~sentiment_agent.decision.agent.DecisionAgent` cycle (that
integration is ``tests/decision/test_agent.py::TestMajorityVote``).
"""

from decision.support import decision, held_book, target
from sentiment_agent.decision.majority import (
    MAJORITY_REPEATS,
    WEIGHT_RULE,
    _majority_side,
    _side,
    majority_weights,
)
from sentiment_agent.policy import POLICY_V1
from sentiment_agent.types import LlmDecision


def _decision(stance: str, targets: list[dict[str, object]]) -> LlmDecision:
    return LlmDecision.model_validate(decision(stance, targets))


# ================================================================================================
# _side / _majority_side
# ================================================================================================


def test_side_classifies_by_sign() -> None:
    assert _side(0.6) == "long"
    assert _side(-0.6) == "short"
    assert _side(0.0) == "flat"


def test_majority_side_needs_at_least_two_agreeing() -> None:
    assert _majority_side(["long"]) is None
    assert _majority_side([]) is None
    assert _majority_side(["long", "long"]) == "long"
    assert _majority_side(["long", "short"]) is None
    assert _majority_side(["long", "long", "short"]) == "long"
    assert _majority_side(["long", "short", "flat"]) is None, "a genuine 3-way split has no winner"


# ================================================================================================
# majority_weights
# ================================================================================================


def test_majority_weights_uses_the_median_of_the_agreeing_observations() -> None:
    """3 act answers agree NVDAUSDT is short, with slightly different sizes: the weight is the
    middle value, not their average -- a median cannot be pulled toward one outlier repeat."""
    original = _decision("act", [target("NVDAUSDT", -0.4)])
    repeat1 = _decision("act", [target("NVDAUSDT", -0.6)])
    repeat2 = _decision("act", [target("NVDAUSDT", -1.0)])  # the outlier
    weights = majority_weights(original, [original, repeat1, repeat2], POLICY_V1, held_book())
    per_name = POLICY_V1.mandate.per_name_max
    assert weights == {"NVDAUSDT": -0.6 * per_name}


def test_majority_weights_confirms_only_symbols_with_2_of_3_side_agreement() -> None:
    original = _decision("act", [target("NVDAUSDT", -0.4), target("BTCUSDT", 0.5)])
    # Agrees on NVDAUSDT's side but flips BTCUSDT's.
    repeat = _decision("act", [target("NVDAUSDT", -0.6), target("BTCUSDT", -0.5)])
    weights = majority_weights(original, [original, repeat], POLICY_V1, held_book())
    per_name = POLICY_V1.mandate.per_name_max
    assert weights["NVDAUSDT"] == -0.5 * per_name  # median of -0.4, -0.6
    # BTCUSDT has no side agreement (only 2 observations, and they disagree): falls back to the
    # book's current weight, never the model's own unconfirmed number.
    assert weights["BTCUSDT"] == held_book().weight("BTCUSDT")


def test_majority_weights_falls_back_to_the_current_weight_when_unconfirmed() -> None:
    """A symbol the original addressed that only one agreeing observation also addresses (no
    2-of-3) keeps its current book weight, exactly like an unconfirmed side flip."""
    book = held_book()
    original = _decision("act", [target("NVDAUSDT", 0.0), target("BTCUSDT", 0.0)])
    # Agrees on both closes, but is the only one of the two to also propose a new TSLAUSDT short.
    repeat = _decision(
        "act", [target("NVDAUSDT", 0.0), target("BTCUSDT", 0.0), target("TSLAUSDT", -0.5)]
    )
    weights = majority_weights(original, [original, repeat], POLICY_V1, book)
    # TSLAUSDT is not in original.symbols, so it is never in the output at all.
    assert set(weights) == {"NVDAUSDT", "BTCUSDT"}
    assert weights == {"NVDAUSDT": 0.0, "BTCUSDT": 0.0}


def test_majority_weights_on_a_single_agreeing_observation_always_falls_back() -> None:
    """act_count could in principle be evaluated with only the original itself in ``agreeing``
    (every repeat failed or disagreed): nothing can reach 2-of-3 agreement, so every symbol the
    original addressed keeps its current weight."""
    book = held_book()
    original = _decision("act", [target("NVDAUSDT", -0.6), target("BTCUSDT", 0.5)])
    weights = majority_weights(original, [original], POLICY_V1, book)
    assert weights == {
        "NVDAUSDT": book.weight("NVDAUSDT"),
        "BTCUSDT": book.weight("BTCUSDT"),
    }


def test_weight_rule_and_majority_repeats_are_stated_plainly() -> None:
    assert MAJORITY_REPEATS == 2
    assert "median" in WEIGHT_RULE
    assert "2" in WEIGHT_RULE
