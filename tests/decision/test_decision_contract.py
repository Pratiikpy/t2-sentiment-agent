"""The decision contract: parsing, validation with specific complaints, retries, outcomes, replay.

Every model here is M6's :class:`~sentiment_agent.llm.fakes.ScriptedChatModel`, which validates its
arguments exactly as the live client does. No test reaches Qwen.
"""

import json
from collections.abc import Sequence
from datetime import timedelta
from decimal import Decimal
from typing import Any

import pytest

from decision.support import (
    NOW,
    MemoryBlobStore,
    book_state,
    build_snapshot,
    completion,
    decision,
    held_book,
    position,
    target,
)
from sentiment_agent.clock import ManualClock
from sentiment_agent.decision.contract import (
    ATTEMPT_MEDIA_TYPE,
    DECISION_SEED,
    INITIAL_COMPLETION_TOKENS,
    REQUEST_MEDIA_TYPE,
    DecisionInvalid,
    classify_failure,
    complaint_message,
    extract_json_object,
    is_truncated,
    obtain_decision,
    parse_decision,
    replay_calls,
)
from sentiment_agent.decision.prompt import PROMPT_VERSION
from sentiment_agent.llm.budget import BudgetExhausted, DailyTokenBudget
from sentiment_agent.llm.client import (
    ChatRequest,
    QwenError,
    QwenTimeout,
    QwenTransportError,
    encode_payload,
    prompt_hash,
)
from sentiment_agent.llm.fakes import (
    RecordedChatModel,
    ReplayMismatch,
    ScriptedChatModel,
    ScriptExhausted,
)
from sentiment_agent.policy import POLICY_V1
from sentiment_agent.types import (
    BookState,
    ChatMessage,
    ChatModel,
    Completion,
    LlmOutcome,
    Stance,
    Thinking,
)

MESSAGES = (
    ChatMessage(role="system", content="You decide."),
    ChatMessage(role="user", content="Decide now."),
)


def _parse(body: dict[str, Any] | str, book: BookState | None = None) -> None:
    book = book if book is not None else book_state()
    content = body if isinstance(body, str) else json.dumps(body)
    parse_decision(content, book=book, snapshot=build_snapshot(book=book), policy=POLICY_V1)


def _complaints(body: dict[str, Any] | str, book: BookState | None = None) -> str:
    with pytest.raises(DecisionInvalid) as caught:
        _parse(body, book)
    return " | ".join(caught.value.complaints)


def _obtain(
    model: ChatModel,
    *,
    thinking: Thinking = Thinking.LOW,
    book: BookState | None = None,
    blobs: MemoryBlobStore | None = None,
) -> tuple[Any, Any, MemoryBlobStore]:
    book = book if book is not None else book_state()
    store = blobs if blobs is not None else MemoryBlobStore()
    decided, call = obtain_decision(
        model,
        MESSAGES,
        thinking=thinking,
        policy=POLICY_V1,
        book=book,
        snapshot=build_snapshot(book=book),
        blobs=store,
    )
    return decided, call, store


VALID_ACT = decision("act", [target("NVDAUSDT", -0.6)])
VALID_FLAT = decision("flat_with_reasons", flat_reasons=["No crowding worth fading."])


# ================================================================================================
# JSON
# ================================================================================================


class TestExtractJsonObject:
    def test_an_object_wrapped_in_prose(self) -> None:
        text = 'Here is my decision: {"a": {"b": 1}} Let me know.'
        assert extract_json_object(text) == '{"a": {"b": 1}}'

    def test_fences_are_removed(self) -> None:
        assert extract_json_object('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_braces_and_escaped_quotes_inside_strings(self) -> None:
        text = 'x {"thesis": "a } brace and a \\" quote {", "n": 2} y'
        assert json.loads(extract_json_object(text)) == {
            "thesis": 'a } brace and a " quote {',
            "n": 2,
        }

    def test_no_balanced_object_returns_the_text(self) -> None:
        assert extract_json_object('prose {"a": 1') == 'prose {"a": 1'
        assert extract_json_object("no json at all") == "no json at all"


# ================================================================================================
# parse_decision: accepted
# ================================================================================================


class TestAccepted:
    def test_act(self) -> None:
        _parse(VALID_ACT)

    def test_act_in_prose_and_fences(self) -> None:
        _parse("Sure.\n```json\n" + json.dumps(VALID_ACT) + "\n```\nDone.")

    def test_flat_with_reasons_on_an_empty_book(self) -> None:
        _parse(VALID_FLAT)

    def test_hold_on_a_held_book(self) -> None:
        body = decision("hold", [target("NVDAUSDT", -0.44568), target("BTCUSDT", 0.998118)])
        _parse(body, held_book())

    def test_close_everything_with_flat_with_reasons(self) -> None:
        body = decision(
            "flat_with_reasons",
            [target("NVDAUSDT", 0.0), target("BTCUSDT", 0.0)],
            flat_reasons=["Crowding has unwound."],
        )
        _parse(body, held_book())

    def test_a_fired_invalidation_may_flip(self) -> None:
        body = decision(
            "act",
            [
                target(
                    "NVDAUSDT",
                    0.4,
                    invalidation_triggered=True,
                    invalidation_evidence="NVDAUSDT.funding_z_live back below reference.zero",
                ),
                target("BTCUSDT", 0.998118),
            ],
        )
        _parse(body, held_book())


# ================================================================================================
# parse_decision: refused, each with a complaint the model can act on
# ================================================================================================


class TestRefused:
    def test_not_json(self) -> None:
        assert "not valid JSON" in _complaints("I think we should go long NVDA.")

    def test_empty(self) -> None:
        assert "empty" in _complaints("   ")

    def test_an_array_is_not_dug_into(self) -> None:
        assert "one JSON object, not a JSON list" in _complaints(json.dumps([VALID_ACT, VALID_ACT]))

    def test_an_unknown_field(self) -> None:
        assert "notes is not a field of the answer; remove it" in _complaints(
            {**VALID_ACT, "notes": "x"}
        )

    def test_a_missing_field(self) -> None:
        body = {k: v for k, v in VALID_ACT.items() if k != "summary"}
        assert "summary is required and missing" in _complaints(body)

    def test_a_string_where_a_number_belongs_is_not_coerced(self) -> None:
        body = decision("act", [{**target("NVDAUSDT", -0.6), "target": "-0.6"}])
        assert "targets[0] (NVDAUSDT).target" in _complaints(body)

    def test_a_target_out_of_range(self) -> None:
        assert "targets[0] (NVDAUSDT).target" in _complaints(
            decision("act", [target("NVDAUSDT", -1.5)])
        )

    def test_a_duplicate_symbol(self) -> None:
        body = decision("act", [target("NVDAUSDT", -0.6), target("NVDAUSDT", 0.2)])
        assert "more than once" in _complaints(body)

    def test_an_unaddressed_held_symbol(self) -> None:
        body = decision("act", [target("NVDAUSDT", -0.6)])
        assert "you hold BTCUSDT and did not address it" in _complaints(body, held_book())

    def test_a_symbol_outside_the_universe(self) -> None:
        assert "'FOOUSDT' is not in the universe" in _complaints(
            decision("act", [target("FOOUSDT", 0.5)])
        )

    def test_an_excluded_symbol(self) -> None:
        assert "ETHUSDT is excluded by the policy" in _complaints(
            decision("act", [target("ETHUSDT", 0.5)])
        )

    def test_a_horizon_below_the_mandate(self) -> None:
        body = decision("act", [target("NVDAUSDT", -0.6, horizon=12)])
        assert "horizon_hours is 12; the mandate requires at least 24" in _complaints(body)

    def test_flat_without_reasons(self) -> None:
        assert "at least one written reason" in _complaints(decision("flat_with_reasons"))

    def test_flat_with_a_nonzero_target(self) -> None:
        body = decision("flat_with_reasons", [target("NVDAUSDT", -0.2)], flat_reasons=["x"])
        assert "cannot carry a non-zero target" in _complaints(body)

    def test_hold_on_an_empty_book(self) -> None:
        body = decision("hold", [target("NVDAUSDT", 0.0)])
        assert "stance 'hold' with no open positions" in _complaints(body)

    def test_act_that_opens_nothing_on_an_empty_book(self) -> None:
        body = decision("act", [target("NVDAUSDT", 0.0)])
        assert "must open something" in _complaints(body)

    def test_hold_that_opens_a_new_name(self) -> None:
        body = decision(
            "hold",
            [target("NVDAUSDT", -0.44), target("BTCUSDT", 0.99), target("TSLAUSDT", 0.3)],
        )
        assert "TSLAUSDT is not held" in _complaints(body, held_book())

    def test_hold_that_turns_a_side(self) -> None:
        body = decision("hold", [target("NVDAUSDT", 0.4), target("BTCUSDT", 0.99)])
        assert "keeps NVDAUSDT short" in _complaints(body, held_book())

    def test_a_declared_invalidation_needs_a_held_position(self) -> None:
        body = decision(
            "act",
            [
                target(
                    "TSLAUSDT",
                    0.3,
                    invalidation_triggered=True,
                    invalidation_evidence="TSLAUSDT.funding_z_live",
                )
            ],
        )
        assert "TSLAUSDT: invalidation_triggered applies only to a position you hold" in (
            _complaints(body)
        )

    def test_a_declared_invalidation_cannot_add_to_the_same_side(self) -> None:
        body = decision(
            "act",
            [
                target(
                    "NVDAUSDT",
                    -0.9,
                    invalidation_triggered=True,
                    invalidation_evidence="NVDAUSDT.funding_z_live",
                ),
                target("BTCUSDT", 0.99),
            ],
        )
        assert "cannot justify adding to the same side" in _complaints(body, held_book())

    def test_a_declared_invalidation_must_say_what_fired(self) -> None:
        body = decision("act", [target("NVDAUSDT", 0.4, invalidation_triggered=True)])
        assert "must state the evidence" in _complaints(body, held_book())

    def test_every_complaint_is_reported_at_once(self) -> None:
        body = decision(
            "act",
            [target("FOOUSDT", 0.5), target("NVDAUSDT", -0.6, horizon=6)],
        )
        complaints = _complaints(body, held_book())
        assert "'FOOUSDT' is not in the universe" in complaints
        assert "horizon_hours is 6" in complaints
        assert "you hold BTCUSDT" in complaints

    def test_decision_invalid_needs_a_complaint(self) -> None:
        with pytest.raises(ValueError, match="at least one complaint"):
            DecisionInvalid([])


def test_complaint_message_lists_each_complaint_and_asks_for_the_whole_object() -> None:
    text = complaint_message(["first thing", "second thing"])
    assert text.splitlines() == [
        "Your previous answer could not be accepted:",
        "- first thing",
        "- second thing",
        "Return the complete corrected JSON object, and nothing else.",
    ]


# ================================================================================================
# obtain_decision
# ================================================================================================


class TestObtain:
    def test_a_valid_answer_is_decided_first_time(self) -> None:
        model = ScriptedChatModel([completion(VALID_ACT)])
        decided, call, store = _obtain(model, thinking=Thinking.LOW)
        assert decided is not None
        assert decided.stance is Stance.ACT
        assert call.outcome is LlmOutcome.DECIDED
        assert call.attempts == 1
        assert call.error is None
        assert call.prompt_version == PROMPT_VERSION
        assert call.prompt_hash == prompt_hash(MESSAGES)
        assert call.usage.reported
        (request,) = model.requests
        assert request.messages == MESSAGES
        assert request.json_mode
        assert request.thinking is Thinking.LOW
        assert request.max_tokens == INITIAL_COMPLETION_TOKENS[Thinking.LOW]
        assert request.temperature == POLICY_V1.decision.temperature
        assert request.seed == DECISION_SEED
        assert call.request_blob is not None
        assert call.request_blob.media_type == REQUEST_MEDIA_TYPE
        assert store.get(call.request_blob.sha256) == encode_payload(
            request.payload(model.model_name)
        )

    def test_full_reasoning_starts_at_the_ceiling(self) -> None:
        model = ScriptedChatModel([completion(VALID_ACT)])
        _obtain(model, thinking=Thinking.FULL)
        assert model.requests[0].max_tokens == POLICY_V1.decision.max_completion_tokens

    def test_invalid_then_valid_feeds_the_complaint_back(self) -> None:
        wrong = decision("act", [target("NVDAUSDT", -0.6, horizon=6)])
        model = ScriptedChatModel([completion(wrong), completion(VALID_ACT)])
        decided, call, _ = _obtain(model)
        assert decided is not None
        assert call.outcome is LlmOutcome.DECIDED
        assert call.attempts == 2
        retry = model.requests[1].messages
        assert retry[: len(MESSAGES)] == MESSAGES
        assert retry[-2] == ChatMessage(role="assistant", content=completion(wrong).content)
        assert retry[-1].role == "user"
        assert "horizon_hours is 6; the mandate requires at least 24" in retry[-1].content
        assert call.usage.completion_tokens == (
            completion(wrong).usage.completion_tokens
            + completion(VALID_ACT).usage.completion_tokens
        )

    def test_only_the_last_answer_is_fed_back(self) -> None:
        first = decision("act", [target("NVDAUSDT", -0.6, horizon=6)])
        second = decision("act", [target("FOOUSDT", 0.1)])
        model = ScriptedChatModel([completion(first), completion(second), completion(VALID_ACT)])
        decided, call, _ = _obtain(model)
        assert decided is not None
        assert call.attempts == 3
        third = model.requests[2].messages
        assert len(third) == len(MESSAGES) + 2
        assert "FOOUSDT" in third[-1].content
        assert "horizon_hours is 6" not in third[-1].content

    def test_three_invalid_answers_end_the_cycle(self) -> None:
        wrong = completion("not json")
        model = ScriptedChatModel([wrong, wrong, wrong])
        decided, call, _ = _obtain(model)
        assert decided is None
        assert call.outcome is LlmOutcome.INVALID_RESPONSE
        assert call.attempts == POLICY_V1.decision.max_attempts
        assert call.error is not None
        assert call.error.count("not valid JSON") == 3

    def test_a_truncated_answer_is_retried_from_the_original_prompt_with_a_larger_cap(self) -> None:
        cut = completion('{"stance": "act", "targets": [{"symbol": "NVD', finish_reason="length")
        model = ScriptedChatModel([cut, completion(VALID_ACT)])
        decided, call, _ = _obtain(model, thinking=Thinking.LOW)
        assert decided is not None
        assert call.outcome is LlmOutcome.DECIDED
        assert [r.max_tokens for r in model.requests] == [4096, 8192]
        assert model.requests[1].messages == MESSAGES

    def test_reasoning_that_uses_the_whole_cap_counts_as_truncated(self) -> None:
        spent = completion("", reasoning="thinking at length", finish_reason="stop")
        assert is_truncated(spent)
        model = ScriptedChatModel([spent, completion(VALID_ACT)])
        decided, _, _ = _obtain(model, thinking=Thinking.LOW)
        assert decided is not None
        assert model.requests[1].max_tokens == 8192

    def test_truncated_at_the_ceiling_is_not_retried_or_repaired(self) -> None:
        cut = completion('{"stance": "act"', finish_reason="length")
        model = ScriptedChatModel([cut, completion(VALID_ACT)])
        decided, call, _ = _obtain(model, thinking=Thinking.FULL)
        assert decided is None
        assert call.outcome is LlmOutcome.TRUNCATED
        assert call.attempts == 1
        assert model.remaining == 1
        assert call.error is not None
        assert "not repaired" in call.error

    def test_truncation_that_runs_out_of_attempts_is_truncated(self) -> None:
        cut = completion("{", finish_reason="length")
        model = ScriptedChatModel([completion("nope"), completion("nope"), cut])
        decided, call, _ = _obtain(model, thinking=Thinking.LOW)
        assert decided is None
        assert call.outcome is LlmOutcome.TRUNCATED


class TestModelServiceFailures:
    def test_budget_exhausted(self) -> None:
        budget = DailyTokenBudget(100, ManualClock(NOW))
        model = ScriptedChatModel([completion(VALID_ACT)], budget=budget)
        decided, call, _ = _obtain(model)
        assert decided is None
        assert call.outcome is LlmOutcome.BUDGET_EXHAUSTED
        assert call.attempts == 1
        assert call.usage.reported
        assert call.usage.total_tokens == 0
        assert call.error is not None
        assert "BudgetExhausted" in call.error

    def test_timeout(self) -> None:
        model = ScriptedChatModel([QwenTimeout("no answer inside 600 s")])
        decided, call, _ = _obtain(model)
        assert decided is None
        assert call.outcome is LlmOutcome.TIMEOUT
        assert not call.usage.reported

    def test_transport_error(self) -> None:
        model = ScriptedChatModel([QwenTransportError("HTTP 503 after 4 attempts")])
        _, call, _ = _obtain(model)
        assert call.outcome is LlmOutcome.TRANSPORT_ERROR

    def test_a_failure_after_an_invalid_answer_takes_its_own_outcome(self) -> None:
        model = ScriptedChatModel([completion("nope"), QwenTimeout("late")])
        decided, call, _ = _obtain(model)
        assert decided is None
        assert call.outcome is LlmOutcome.TIMEOUT
        assert call.attempts == 2
        assert not call.usage.reported
        assert call.error is not None
        assert "not valid JSON" in call.error
        assert "QwenTimeout" in call.error

    @pytest.mark.parametrize(
        ("error", "outcome"),
        [
            (BudgetExhausted("cap"), LlmOutcome.BUDGET_EXHAUSTED),
            (QwenTimeout("late"), LlmOutcome.TIMEOUT),
            (TimeoutError("socket"), LlmOutcome.TIMEOUT),
            (QwenTransportError("503"), LlmOutcome.TRANSPORT_ERROR),
            (QwenError("other"), LlmOutcome.TRANSPORT_ERROR),
            (ConnectionResetError("reset"), LlmOutcome.TRANSPORT_ERROR),
            (ValueError("a defect"), None),
            (ReplayMismatch("no recording"), None),
            (ScriptExhausted("under-scripted"), None),
        ],
    )
    def test_classify_failure(self, error: BaseException, outcome: LlmOutcome | None) -> None:
        assert classify_failure(error) is outcome

    @pytest.mark.parametrize(
        "error", [ScriptExhausted("under-scripted"), ReplayMismatch("gone"), ValueError("bug")]
    )
    def test_anything_that_is_not_the_model_service_failing_is_raised(
        self, error: Exception
    ) -> None:
        model = ScriptedChatModel([error])
        with pytest.raises(type(error)):
            _obtain(model)


# ================================================================================================
# Evidence and replay
# ================================================================================================


def _attempts(call: Any, store: MemoryBlobStore) -> list[dict[str, Any]]:
    return [
        json.loads(store.get(ref.sha256))
        for ref in call.response_blobs
        if ref.media_type == ATTEMPT_MEDIA_TYPE
    ]


def test_every_attempt_is_logged_with_its_request_answer_and_verdict() -> None:
    wrong = completion("nope")
    model = ScriptedChatModel([wrong, completion(VALID_ACT)])
    _, call, store = _obtain(model)
    attempts = _attempts(call, store)
    assert [a["attempt"] for a in attempts] == [1, 2]
    assert [a["verdict"] for a in attempts] == ["invalid", "accepted"]
    assert attempts[0]["complaints"]
    assert Completion.model_validate(attempts[0]["completion"]) == wrong
    request = ChatRequest.from_payload(attempts[1]["request"])
    assert request == model.requests[1]
    assert attempts[1]["prompt_hash"] == request.prompt_hash
    assert attempts[0]["wire"] == []


def test_a_failed_attempt_is_logged_with_its_error() -> None:
    model = ScriptedChatModel([QwenTimeout("late")])
    _, call, store = _obtain(model)
    (attempt,) = _attempts(call, store)
    assert attempt["verdict"] == "failed"
    assert attempt["completion"] is None
    assert attempt["error"] == {"type": "QwenTimeout", "message": "late"}


def test_replay_reproduces_the_cycle_from_the_logged_attempts() -> None:
    cut = completion("{", finish_reason="length")
    wrong = completion(decision("act", [target("NVDAUSDT", -0.6, horizon=6)]))
    model = ScriptedChatModel([cut, wrong, completion(VALID_ACT)])
    decided, call, store = _obtain(model, thinking=Thinking.LOW)
    recordings = replay_calls(call, store)
    assert len(recordings) == 3
    replayed, replay_call, _ = _obtain(RecordedChatModel(recordings), thinking=Thinking.LOW)
    assert replayed == decided
    assert replay_call.outcome is call.outcome
    assert replay_call.attempts == call.attempts
    assert replay_call.prompt_hash == call.prompt_hash


def test_replay_refuses_a_changed_blob() -> None:
    model = ScriptedChatModel([completion(VALID_ACT)])
    _, call, store = _obtain(model)
    attempt_ref = next(r for r in call.response_blobs if r.media_type == ATTEMPT_MEDIA_TYPE)
    store.data[attempt_ref.sha256] = b'{"tampered": true}'
    with pytest.raises(ValueError, match="does not match its reference"):
        replay_calls(call, store)


def test_a_replay_with_a_different_prompt_is_not_an_outage() -> None:
    model = ScriptedChatModel([completion(VALID_ACT)])
    _, call, store = _obtain(model)
    other: Sequence[ChatMessage] = (*MESSAGES[:1], ChatMessage(role="user", content="Other."))
    with pytest.raises(ReplayMismatch):
        obtain_decision(
            RecordedChatModel(replay_calls(call, store)),
            other,
            thinking=Thinking.LOW,
            policy=POLICY_V1,
            book=book_state(),
            snapshot=build_snapshot(),
            blobs=MemoryBlobStore(),
        )


def test_a_declared_invalidation_may_cut_the_same_side() -> None:
    """Cutting a position whose invalidation fired is consistent; only adding to it is refused."""
    fresh = position("NVDAUSDT", "-1", "225.10", opened=NOW - timedelta(hours=2))
    book = book_state(positions={"NVDAUSDT": fresh}, marks={"NVDAUSDT": Decimal("222.84")})
    body = decision(
        "act",
        [
            target(
                "NVDAUSDT",
                -0.1,
                invalidation_triggered=True,
                invalidation_evidence="NVDAUSDT.funding_z_live",
            )
        ],
    )
    _parse(body, book)
