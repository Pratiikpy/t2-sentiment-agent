"""run3-d2: majority-of-3 cross-check on a stance ``act`` decision.

G7 (``argus/data/t2_llm_variance.json``, ``Activity/21_T2_QUANT_SEARCH.md`` row G7, 2026-09-29):
the same accepted request, at temperature 0 with a fixed seed, resent through a fresh Qwen client.
Run 2's two no-act cycles reproduced exactly, or differed only in an all-zero target list with no
economic content. Its one act cycle (``dec-189d8234…``, closing MSTRUSDT and METAUSDT) did not:
repeat 1 flipped stance to ``hold`` and kept both positions open near their prior weight; repeat 2
kept stance ``act`` and closed the same two names, but also opened TSLAUSDT and COINUSDT shorts
neither the original nor repeat 1 proposed at all, and confidence on the repeated names swung by up
to 0.27. Repeating the no-act cycles cost real tokens for no evidence of variance that would move
the record; the one place disagreement actually appeared was the one act cycle. Sampling every
cycle would roughly triple the daily spend for no benefit on the stable majority (that no-act
cycles already are); sampling only stance ``act`` cycles pays for the check only where the evidence
says it is needed.

The rule, exactly as :data:`WEIGHT_RULE` states it on every
:class:`~sentiment_agent.types.MajorityVote`:

1. The first answer must be stance ``act`` (``decision/agent.py`` checks this before ever calling
   into this module — a ``hold`` or ``flat_with_reasons`` first answer is never re-asked).
2. The identical request (same messages, model, thinking tier, temperature, seed) is sent up to 2
   more times through :func:`~sentiment_agent.decision.contract.obtain_decision`, stopping early if
   a repeat's own call is refused for the day's exhausted budget (asking again would only be
   refused again).
3. **Stance majority**: at least 2 of the (up to 3) decided answers must have stance ``act``, or
   the cycle is recorded as :data:`~sentiment_agent.types.LlmOutcome.DISAGREED` — a no-act, never a
   service outage (``runtime/loop.py``), with every observation obtained kept in the ledger.
4. **Per-instrument majority**, only once the stance majority is reached, computed only from the
   act-stance observations (a dissenting ``hold`` or ``flat_with_reasons`` answer's own targets are
   not pooled with the act answers' targets — they answer a different question): for each symbol
   the *original* (first) answer addressed, its side (long / short / flat, by the sign of the
   model's own ``target``) must agree across at least 2 of the act-stance observations that
   addressed that symbol. Not enough of them address it, or they disagree, and the symbol keeps its
   **current book weight** — no change — rather than the model's own unconfirmed number. A symbol
   only a dissenting act-stance repeat proposed, that the original never addressed at all (like
   run 2's TSLAUSDT and COINUSDT), is never in the original's symbol set and is therefore never
   acted on, because the record's kept :class:`~sentiment_agent.types.LlmDecision` — for the audit
   trail, and Track 2's judged "decision explainability" — is the *original* answer, unchanged, and
   only the symbols it addressed are ever in play.
5. Where a symbol's side is confirmed, its weight is the **median** of the agreeing observations'
   ``target`` (not the mean): a median cannot be pulled toward one outlier repeat the way an
   average would, and it always equals a value at least one real answer actually gave.

Cost per stance ``act`` cycle: up to 2 extra full decision calls (each itself bounded by
``policy.decision.max_attempts`` on invalid/truncated answers, exactly like the first). Measured on
G7's own numbers, one act cycle's accepted request billed 18.7k-30.0k tokens; 2 more of the same
shape roughly triples that one cycle's cost. This is bounded automatically by the existing daily
cap (``llm/budget.py``): a repeat that would cross it is refused before it is sent, which this
module treats as a failed repeat (rule 3), never as a reason to skip the budget check or overspend.
"""

from collections import Counter
from collections.abc import Sequence
from statistics import median
from typing import Final

from sentiment_agent.types import (
    BookState,
    LlmCallRecord,
    LlmDecision,
    LlmOutcome,
    MajorityObservation,
    MajorityVote,
    Policy,
    Stance,
)

MAJORITY_REPEATS: Final = 2
"""How many extra identical requests follow a stance ``act`` first answer (module docstring)."""

WEIGHT_RULE: Final = (
    "act only where >= 2 of the (up to 3) answers agree: first on the overall stance (>= 2 say "
    "'act'), then per instrument on the side (by the sign of the model's own target), counting "
    "only the act-stance answers. A symbol the original addressed without 2-of-3 side agreement "
    "keeps its current book weight, never the model's own unconfirmed number. Where confirmed, "
    "the weight is the median of the agreeing observations' target, not the mean."
)


def _side(target: float) -> str:
    if target > 0:
        return "long"
    if target < 0:
        return "short"
    return "flat"


def _majority_side(sides: Sequence[str]) -> str | None:
    """The side at least 2 of ``sides`` share, or ``None`` when there is no such side."""
    if len(sides) < 2:
        return None
    side, count = Counter(sides).most_common(1)[0]
    return side if count >= 2 else None


def majority_weights(
    original: LlmDecision, agreeing: Sequence[LlmDecision], policy: Policy, book: BookState
) -> dict[str, float]:
    """``proposed_weights``-shaped, but confirmed per symbol against ``agreeing`` (the act-stance
    observations; ``original`` is always one of them). See the module docstring for the rule.
    """
    per_name = policy.mandate.per_name_max
    weights: dict[str, float] = {}
    for symbol in original.symbols:
        values = [t.target for d in agreeing for t in d.targets if t.symbol == symbol]
        winner = _majority_side([_side(v) for v in values])
        if winner is None:
            weights[symbol] = book.weight(symbol)
            continue
        agreeing_values = [v for v in values if _side(v) == winner]
        weights[symbol] = float(f"{median(agreeing_values) * per_name:.12g}")
    return weights


def build_majority_vote(
    observations: Sequence[tuple[LlmDecision | None, LlmCallRecord]],
) -> MajorityVote:
    """The vote over ``observations`` (1 to 3, in the order asked; ``observations[0]`` must be a
    decided act answer). A failed repeat (``decision is None``) casts no vote either way; it only
    shrinks the sample, exactly like a repeat that came back but disagreed."""
    decided = [d for d, _ in observations if d is not None]
    act_count = sum(1 for d in decided if d.stance is Stance.ACT)
    agreed = act_count >= 2
    reason: str | None = None
    if not agreed:
        failed = sum(
            1 for d, c in observations if d is None and c.outcome is not LlmOutcome.DECIDED
        )
        stances = ", ".join(d.stance.value for d in decided)
        parts = [f"the model did not agree with itself: {len(decided)} of 3 answers decided"]
        if stances:
            parts.append(f"stances {stances}")
        if failed:
            parts.append(f"{failed} repeat(s) failed and could not be compared")
        reason = "; ".join(parts)
    return MajorityVote(
        observations=tuple(MajorityObservation(call=c, decision=d) for d, c in observations),
        act_count=act_count,
        agreed=agreed,
        weight_rule=WEIGHT_RULE,
        reason=reason,
    )


__all__ = [
    "MAJORITY_REPEATS",
    "WEIGHT_RULE",
    "build_majority_vote",
    "majority_weights",
]
