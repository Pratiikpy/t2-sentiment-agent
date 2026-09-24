"""The decision agent: one portfolio-level Qwen decision per admitted trigger set (DESIGN.md §9).

:meth:`DecisionAgent.decide` renders the prompt, obtains a validated decision through the contract,
grounds every number in each target's thesis, invalidation and view against the facts the model was
shown, and returns a :class:`~sentiment_agent.types.DecisionRecord` for the ledger. It never raises
for a failure of the model service: a cycle without a valid decision is a record with that outcome
and no decision, and the kernel answers it by flattening the book (``llm_outage``). No deterministic
rule ever opens a position in the model's place (handbook:224).

Reasoning tier: a cycle woken by a heartbeat (the US open, a funding settlement) reasons at the
policy's heartbeat tier (FULL); a cycle woken only by events reasons at the event tier (LOW), which
keeps the daily spend inside the cap the owner approves (DESIGN.md §19.6).
"""

from collections.abc import Sequence
from typing import Final

from sentiment_agent.decision.contract import obtain_decision
from sentiment_agent.decision.grounding import ground_decision
from sentiment_agent.decision.prompt import (
    HEARTBEAT_KINDS,
    decision_facts,
    held_symbols,
    render_messages,
)
from sentiment_agent.hashing import content_hash
from sentiment_agent.types import (
    BlobStore,
    BookState,
    ChatModel,
    Clock,
    DecisionRecord,
    GroundingReport,
    LlmDecision,
    PerceptionSnapshot,
    Policy,
    Stance,
    Thinking,
    Trigger,
)

DECISION_ID_PREFIX: Final = "dec-"
DECISION_ID_HEX: Final = 32


def proposed_weights(decision: LlmDecision, policy: Policy, book: BookState) -> dict[str, float]:
    """The weight the model asks for, per addressed symbol: ``target * per_name_max``.

    One exception, so that "hold" means hold: under stance ``hold`` a held symbol's proposal is its
    current weight exactly, not its target re-multiplied. A target the model copied back from
    ``position_target_equivalent`` is rounded to six digits, and re-multiplying it would ask the
    kernel for a sliver of a trade the model never wanted. When the current weight cannot be
    measured (no mark in the book), the target is used as written.
    """
    per_name = policy.mandate.per_name_max
    held = set(held_symbols(book))
    weights: dict[str, float] = {}
    for target in decision.targets:
        symbol = target.symbol
        if (
            decision.stance is Stance.HOLD
            and symbol in held
            and symbol in book.marks
            and book.equity > 0
        ):
            weights[symbol] = book.weight(symbol)
        else:
            weights[symbol] = float(f"{target.target * per_name:.12g}")
    return weights


class DecisionAgent:
    """Qwen as the primary decision-maker, behind the JSON contract and the grounding check."""

    def __init__(self, *, model: ChatModel, policy: Policy, blobs: BlobStore, clock: Clock) -> None:
        self._model = model
        self._policy = policy
        self._blobs = blobs
        self._clock = clock

    def thinking_for(self, triggers: Sequence[Trigger]) -> Thinking:
        """The heartbeat tier if any trigger is a heartbeat, otherwise the event tier."""
        rule = self._policy.decision
        if any(t.kind in HEARTBEAT_KINDS for t in triggers):
            return rule.thinking_heartbeat
        return rule.thinking_event

    def decide(
        self, snapshot: PerceptionSnapshot, book: BookState, triggers: Sequence[Trigger]
    ) -> DecisionRecord:
        """One decision cycle. Raises only for a caller's mistake (no trigger, a snapshot taken
        under another policy) or a defect, never for the model being unavailable or wrong."""
        policy = self._policy
        now = self._clock.now()
        messages = render_messages(snapshot, book, triggers, policy, now=now)
        thinking = self.thinking_for(triggers)
        decision, call = obtain_decision(
            self._model,
            messages,
            thinking=thinking,
            policy=policy,
            book=book,
            snapshot=snapshot,
            blobs=self._blobs,
        )
        weights: dict[str, float] = {}
        grounding: dict[str, GroundingReport] = {}
        if decision is not None:
            weights = proposed_weights(decision, policy, book)
            facts = decision_facts(snapshot, book, triggers, policy, now=now)
            for symbol, weight in weights.items():
                # The model's own requested size is citable: "a 3% short" quotes its answer.
                facts[f"{symbol}.proposed_weight_pct"] = float(f"{weight * 100:.12g}")
            grounding = ground_decision(decision, facts, policy.grounding_tolerance)
        trigger_ids = tuple(t.trigger_id for t in triggers)
        decision_id = (
            DECISION_ID_PREFIX
            + content_hash(
                {
                    "decided_at": now,
                    "snapshot_id": snapshot.snapshot_id,
                    "trigger_ids": trigger_ids,
                    "policy_version": policy.version,
                    "prompt_hash": call.prompt_hash,
                    "outcome": call.outcome,
                    "decision": decision,
                }
            )[:DECISION_ID_HEX]
        )
        return DecisionRecord(
            decision_id=decision_id,
            decided_at=now,
            trigger_ids=trigger_ids,
            snapshot_id=snapshot.snapshot_id,
            book_before=book,
            mandate=policy.mandate,
            policy_version=policy.version,
            call=call,
            outcome=call.outcome,
            decision=decision,
            grounding=grounding,
            proposed_weights=weights,
        )


__all__ = ["DECISION_ID_HEX", "DECISION_ID_PREFIX", "DecisionAgent", "proposed_weights"]
