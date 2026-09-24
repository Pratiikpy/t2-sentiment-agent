"""The decision agent end to end: prompt, contract, grounding, and the record the ledger receives.

Every model is M6's :class:`~sentiment_agent.llm.fakes.ScriptedChatModel` (or a replay of what one
said). The agent never raises for the model failing: each outage is a record with its outcome, and
the record's own validator (``types.DecisionRecord``) checks what an outage may carry.
"""

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from decision.support import (
    NOW,
    MemoryBlobStore,
    book_state,
    build_snapshot,
    completion,
    decision,
    funding_event,
    heartbeat,
    held_book,
    target,
)
from sentiment_agent.clock import ManualClock
from sentiment_agent.decision.agent import DecisionAgent, proposed_weights
from sentiment_agent.decision.contract import replay_calls
from sentiment_agent.decision.prompt import format_number
from sentiment_agent.llm.budget import DailyTokenBudget
from sentiment_agent.llm.client import QwenTimeout, QwenTransportError
from sentiment_agent.llm.fakes import RecordedChatModel, ScriptedChatModel
from sentiment_agent.policy import POLICY_V1
from sentiment_agent.types import (
    BookState,
    ChatModel,
    DecisionRecord,
    LlmDecision,
    LlmOutcome,
    PerceptionSnapshot,
    Thinking,
    Trigger,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "decision" / "fabricated_figures.json"


def _agent(model: ChatModel, blobs: MemoryBlobStore | None = None) -> DecisionAgent:
    return DecisionAgent(
        model=model, policy=POLICY_V1, blobs=blobs or MemoryBlobStore(), clock=ManualClock(NOW)
    )


def _decide(
    model: ChatModel,
    *,
    book: BookState | None = None,
    triggers: tuple[Trigger, ...] = (funding_event(),),
    snapshot: PerceptionSnapshot | None = None,
    blobs: MemoryBlobStore | None = None,
) -> DecisionRecord:
    book = book if book is not None else book_state()
    snap = snapshot if snapshot is not None else build_snapshot(book=book)
    return _agent(model, blobs).decide(snap, book, triggers)


def _validated(body: dict[str, Any]) -> LlmDecision:
    return LlmDecision.model_validate(body)


# ================================================================================================
# proposed_weights
# ================================================================================================


class TestProposedWeights:
    def test_act_is_target_times_the_per_name_cap_without_float_noise(self) -> None:
        body = decision("act", [target("NVDAUSDT", -0.6), target("BTCUSDT", 0.3)])
        weights = proposed_weights(_validated(body), POLICY_V1, book_state())
        assert weights == {"NVDAUSDT": -0.03, "BTCUSDT": 0.015}

    def test_hold_keeps_the_current_weight_exactly(self) -> None:
        book = held_book()
        body = decision("hold", [target("NVDAUSDT", -0.44568), target("BTCUSDT", 0.998118)])
        weights = proposed_weights(_validated(body), POLICY_V1, book)
        assert weights == {"NVDAUSDT": book.weight("NVDAUSDT"), "BTCUSDT": book.weight("BTCUSDT")}

    def test_hold_without_a_mark_uses_the_target(self) -> None:
        book = held_book()
        book = book.model_copy(update={"marks": {"BTCUSDT": book.marks["BTCUSDT"]}})
        body = decision("hold", [target("NVDAUSDT", -0.4), target("BTCUSDT", 0.998118)])
        weights = proposed_weights(_validated(body), POLICY_V1, book)
        assert weights["NVDAUSDT"] == -0.02
        assert weights["BTCUSDT"] == book.weight("BTCUSDT")

    def test_flat_is_zero_for_every_addressed_symbol(self) -> None:
        body = decision(
            "flat_with_reasons",
            [target("NVDAUSDT", 0.0), target("BTCUSDT", 0.0)],
            flat_reasons=["Unwound."],
        )
        assert proposed_weights(_validated(body), POLICY_V1, held_book()) == {
            "NVDAUSDT": 0.0,
            "BTCUSDT": 0.0,
        }


# ================================================================================================
# decide: a decision
# ================================================================================================


class TestDecided:
    def test_a_valid_act_becomes_a_complete_record(self) -> None:
        book = book_state()
        snapshot = build_snapshot(book=book)
        stretch = snapshot.features["NVDAUSDT"].ma20_distance_atr
        assert stretch is not None
        body = decision(
            "act",
            [
                target(
                    "NVDAUSDT",
                    -0.6,
                    thesis=f"NVDAUSDT.ma20_distance_atr at {format_number(stretch)} with a "
                    "coordinated story naming it.",
                    our_view="Short at target -0.6, a 3% position, confidence 55%.",
                )
            ],
        )
        model = ScriptedChatModel([completion(body)])
        record = _decide(model, book=book, snapshot=snapshot)
        assert record.outcome is LlmOutcome.DECIDED
        assert record.decision is not None
        assert record.decided_at == NOW
        assert record.snapshot_id == snapshot.snapshot_id
        assert record.trigger_ids == (funding_event().trigger_id,)
        assert record.book_before == book
        assert record.mandate == POLICY_V1.mandate
        assert record.policy_version == POLICY_V1.version
        assert record.proposed_weights == {"NVDAUSDT": -0.03}
        assert set(record.grounding) == {"NVDAUSDT"}
        report = record.grounding["NVDAUSDT"]
        assert report.grounded, [(f.raw, f.context) for f in report.unresolved]
        assert {f.source for f in report.figures} >= {
            "NVDAUSDT.ma20_distance_atr",
            "NVDAUSDT.proposed_weight_pct",
            "NVDAUSDT.proposed_confidence",
        }
        assert record.decision_id.startswith("dec-")
        assert len(record.decision_id) == 4 + 32

    def test_the_record_is_deterministic(self) -> None:
        body = decision("act", [target("NVDAUSDT", -0.6)])
        first = _decide(ScriptedChatModel([completion(body)]))
        second = _decide(ScriptedChatModel([completion(body)]))
        assert first == second
        assert first.decision_id == second.decision_id

    def test_a_different_answer_gets_a_different_id(self) -> None:
        a = _decide(ScriptedChatModel([completion(decision("act", [target("NVDAUSDT", -0.6)]))]))
        b = _decide(ScriptedChatModel([completion(decision("act", [target("NVDAUSDT", -0.5)]))]))
        assert a.decision_id != b.decision_id

    def test_flat_with_reasons_on_an_empty_book(self) -> None:
        body = decision("flat_with_reasons", flat_reasons=["No crowding worth fading."])
        record = _decide(ScriptedChatModel([completion(body)]))
        assert record.outcome is LlmOutcome.DECIDED
        assert record.proposed_weights == {}
        assert record.grounding == {}

    def test_hold_on_a_held_book_proposes_no_change(self) -> None:
        book = held_book()
        body = decision(
            "hold",
            [
                target("NVDAUSDT", -0.44568, thesis="position_target_equivalent -0.44568 kept."),
                target("BTCUSDT", 0.998118),
            ],
        )
        record = _decide(ScriptedChatModel([completion(body)]), book=book)
        assert record.proposed_weights == {
            "NVDAUSDT": book.weight("NVDAUSDT"),
            "BTCUSDT": book.weight("BTCUSDT"),
        }
        assert record.grounding["NVDAUSDT"].grounded

    def test_an_invalid_answer_then_a_valid_one(self) -> None:
        wrong = decision("act", [target("NVDAUSDT", -0.6, horizon=6)])
        model = ScriptedChatModel(
            [completion(wrong), completion(decision("act", [target("NVDAUSDT", -0.6)]))]
        )
        record = _decide(model)
        assert record.outcome is LlmOutcome.DECIDED
        assert record.call.attempts == 2
        assert "horizon_hours is 6" in model.requests[1].messages[-1].content

    def test_a_truncated_answer_then_a_larger_cap(self) -> None:
        body = decision("act", [target("NVDAUSDT", -0.6)])
        model = ScriptedChatModel([completion("{", finish_reason="length"), completion(body)])
        record = _decide(model, triggers=(funding_event(),))
        assert record.outcome is LlmOutcome.DECIDED
        assert [r.max_tokens for r in model.requests] == [4096, 8192]


class TestReasoningTier:
    def test_a_heartbeat_reasons_fully(self) -> None:
        model = ScriptedChatModel([completion(decision("act", [target("NVDAUSDT", -0.6)]))])
        record = _decide(model, triggers=(heartbeat(), funding_event()))
        assert record.call.thinking is Thinking.FULL
        assert model.requests[0].thinking is Thinking.FULL
        assert model.requests[0].max_tokens == POLICY_V1.decision.max_completion_tokens

    def test_an_event_reasons_low(self) -> None:
        model = ScriptedChatModel([completion(decision("act", [target("NVDAUSDT", -0.6)]))])
        record = _decide(model, triggers=(funding_event(),))
        assert record.call.thinking is Thinking.LOW
        assert model.requests[0].thinking is Thinking.LOW


# ================================================================================================
# decide: outages, never raised
# ================================================================================================


class TestOutage:
    def test_budget_exhausted_is_a_record_not_an_exception(self) -> None:
        budget = DailyTokenBudget(1_000, ManualClock(NOW))
        model = ScriptedChatModel(
            [completion(decision("act", [target("NVDAUSDT", -0.6)]))], budget=budget
        )
        record = _decide(model)
        assert record.outcome is LlmOutcome.BUDGET_EXHAUSTED
        assert record.decision is None
        assert record.proposed_weights == {}
        assert record.grounding == {}
        assert record.call.attempts == 1

    def test_timeout_is_a_record(self) -> None:
        record = _decide(ScriptedChatModel([QwenTimeout("no answer inside the deadline")]))
        assert record.outcome is LlmOutcome.TIMEOUT
        assert record.decision is None
        assert not record.call.usage.reported

    def test_transport_error_is_a_record(self) -> None:
        record = _decide(ScriptedChatModel([QwenTransportError("HTTP 502")]))
        assert record.outcome is LlmOutcome.TRANSPORT_ERROR

    def test_three_invalid_answers_are_a_record(self) -> None:
        record = _decide(ScriptedChatModel([completion("nope")] * 3))
        assert record.outcome is LlmOutcome.INVALID_RESPONSE
        assert record.decision is None
        assert record.call.error is not None

    def test_an_unaddressed_held_symbol_three_times_is_a_record(self) -> None:
        body = decision("act", [target("NVDAUSDT", -0.6)])
        record = _decide(ScriptedChatModel([completion(body)] * 3), book=held_book())
        assert record.outcome is LlmOutcome.INVALID_RESPONSE
        assert "you hold BTCUSDT" in (record.call.error or "")

    def test_outage_records_differ_by_outcome(self) -> None:
        timeout = _decide(ScriptedChatModel([QwenTimeout("late")]))
        transport = _decide(ScriptedChatModel([QwenTransportError("502")]))
        assert timeout.decision_id != transport.decision_id


class TestCallerMistakes:
    def test_no_trigger_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one admitted trigger"):
            _decide(ScriptedChatModel([]), triggers=())

    def test_a_snapshot_under_another_policy_is_refused(self) -> None:
        snapshot = build_snapshot().model_copy(update={"policy_version": "policy-v0"})
        with pytest.raises(ValueError, match="taken under"):
            _decide(ScriptedChatModel([]), snapshot=snapshot)


# ================================================================================================
# Grounding in the record, with the fabricated-figure set re-run at the decision level
# ================================================================================================


class TestGroundingInTheRecord:
    def test_a_fabricated_figure_leaves_the_target_ungrounded(self) -> None:
        snapshot = build_snapshot()
        last = snapshot.features["NVDAUSDT"].demo_last
        funding_z = snapshot.features["BTCUSDT"].funding_z_live
        assert last is not None
        assert funding_z is not None
        body = decision(
            "act",
            [
                target(
                    "NVDAUSDT",
                    -0.6,
                    thesis=f"Demo last {format_number(last)} against a fair value of 187.5.",
                ),
                target("BTCUSDT", 0.4, thesis=f"Funding z at {format_number(funding_z)}, crowded."),
            ],
        )
        record = _decide(ScriptedChatModel([completion(body)]), snapshot=snapshot)
        nvda = record.grounding["NVDAUSDT"]
        assert not nvda.grounded
        assert [f.raw for f in nvda.unresolved] == ["187.5"]
        assert record.grounding["BTCUSDT"].grounded

    def test_every_recorded_fabricated_price_is_ungrounded_in_a_real_decision(self) -> None:
        cases = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]
        targets = []
        for case in cases:
            if case["symbol"] not in POLICY_V1.symbols:
                continue
            targets.append(
                (
                    case["symbol"],
                    f"Entry near {case['fabricated_entry_price']} on {case['symbol']}.",
                )
            )
        assert len(targets) == 8
        for symbol, thesis in targets:
            body = decision("act", [target(symbol, -0.4, thesis=thesis)])
            record = _decide(ScriptedChatModel([completion(body)]))
            assert not record.grounding[symbol].grounded, thesis

    def test_a_real_price_grounds_in_a_real_decision(self) -> None:
        snapshot = build_snapshot()
        price = snapshot.features["AAPLUSDT"].demo_last
        assert price is not None
        body = decision("act", [target("AAPLUSDT", 0.4, thesis=f"Entry near {price} on AAPLUSDT.")])
        record = _decide(ScriptedChatModel([completion(body)]), snapshot=snapshot)
        assert record.grounding["AAPLUSDT"].grounded


# ================================================================================================
# Replay: the cycle re-runs from what the model said
# ================================================================================================


def test_replay_reproduces_the_record() -> None:
    body = decision("act", [target("NVDAUSDT", -0.6)])
    blobs = MemoryBlobStore()
    wrong = decision("act", [target("NVDAUSDT", -0.6, horizon=6)])
    record = _decide(ScriptedChatModel([completion(wrong), completion(body)]), blobs=blobs)
    replayed = _decide(RecordedChatModel(replay_calls(record.call, blobs)))
    assert replayed == record


def test_the_record_round_trips_through_json() -> None:
    record = _decide(ScriptedChatModel([completion(decision("act", [target("NVDAUSDT", -0.6)]))]))
    assert DecisionRecord.model_validate_json(record.model_dump_json()) == record
    assert record.book_before.equity == Decimal("10000")
