"""The stand-ins behave deterministically, validate like the real client, and a recorded exchange
replays to the completion the live parser produced."""

import json
from collections.abc import Iterator, Mapping
from datetime import timedelta
from typing import Any

import pytest

from sentiment_agent.clock import ManualClock
from sentiment_agent.hashing import sha256_hex
from sentiment_agent.llm import client
from sentiment_agent.llm.budget import BudgetExhausted, DailyTokenBudget, projected_tokens
from sentiment_agent.llm.client import (
    COMPLETION_MEDIA_TYPE,
    ChatRequest,
    QwenChatModel,
    QwenCredentials,
    QwenError,
    QwenTimeout,
    QwenTransportError,
    prompt_hash,
)
from sentiment_agent.llm.fakes import (
    FailingChatModel,
    RecordedCall,
    RecordedChatModel,
    ReplayMismatch,
    ScriptedChatModel,
    ScriptExhausted,
    completion_from_json,
)
from sentiment_agent.types import BlobRef, ChatMessage, ChatModel, Completion, Thinking

PROMPT = (
    ChatMessage(role="system", content="Decide. Answer in JSON."),
    ChatMessage(role="user", content="NVDAUSDT funding z 2.4; the crowd is long."),
)
OTHER = (ChatMessage(role="user", content="BTCUSDT funding z -1.1."),)
HOLD = {"stance": "hold", "summary": "nothing changed", "targets": []}
FLAT = {"stance": "flat_with_reasons", "flat_reasons": ["no edge"], "targets": []}


def call(model: ChatModel, messages: tuple[ChatMessage, ...] = PROMPT, **kw: Any) -> Completion:
    args: dict[str, Any] = {"json_mode": True, "max_tokens": 1024, "thinking": Thinking.LOW}
    args.update(kw)
    return model.complete(messages, **args)


# ================================================================================================
# completion_from_json
# ================================================================================================


def test_completion_from_json_is_a_finished_deterministic_answer() -> None:
    first = completion_from_json(HOLD, reasoning="funding is rich")
    again = completion_from_json(dict(reversed(list(HOLD.items()))), reasoning="funding is rich")
    assert first == again
    assert json.loads(first.content) == HOLD
    assert first.content == '{"stance":"hold","summary":"nothing changed","targets":[]}'
    assert first.reasoning == "funding is rich"
    assert "funding" not in first.content
    assert first.finish_reason == "stop"
    assert first.raw_id.startswith("fake-")
    assert first.usage.reported is True
    content_bytes = len(first.content.encode())
    assert first.usage.completion_tokens == content_bytes + len(b"funding is rich")
    assert first.usage.reasoning_tokens == len(b"funding is rich")
    assert first.usage.total_tokens == first.usage.completion_tokens


def test_different_answers_have_different_ids() -> None:
    assert completion_from_json(HOLD).raw_id != completion_from_json(FLAT).raw_id
    assert completion_from_json(HOLD).raw_id != completion_from_json(HOLD, reasoning="x").raw_id


def test_completion_from_json_refuses_non_finite_numbers() -> None:
    with pytest.raises(ValueError):  # noqa: PT011 - raised by canonical JSON
        completion_from_json({"confidence": float("nan")})


# ================================================================================================
# ScriptedChatModel
# ================================================================================================


def test_scripted_answers_in_order_and_keeps_every_request() -> None:
    model = ScriptedChatModel([completion_from_json(HOLD), completion_from_json(FLAT)])
    assert model.remaining == 2
    assert json.loads(call(model).content) == HOLD
    assert json.loads(call(model, OTHER, thinking=Thinking.FULL, seed=4).content) == FLAT
    assert model.remaining == 0
    first, second = model.requests
    assert first == ChatRequest(
        messages=PROMPT, json_mode=True, max_tokens=1024, thinking=Thinking.LOW
    )
    assert (second.messages, second.thinking, second.seed) == (OTHER, Thinking.FULL, 4)
    assert second.streamed is True


def test_scripted_raises_what_it_was_given_then_continues() -> None:
    model = ScriptedChatModel([QwenTimeout("slow"), completion_from_json(HOLD)])
    with pytest.raises(QwenTimeout, match="slow"):
        call(model)
    assert json.loads(call(model).content) == HOLD


def test_an_empty_script_fails_the_test_loudly() -> None:
    model = ScriptedChatModel([completion_from_json(HOLD)])
    call(model)
    with pytest.raises(ScriptExhausted, match="call 2"):
        call(model)
    assert not isinstance(ScriptExhausted(), QwenError)


def test_push_appends_to_the_script() -> None:
    model = ScriptedChatModel()
    model.push(completion_from_json(HOLD), QwenTransportError("down"))
    assert json.loads(call(model).content) == HOLD
    with pytest.raises(QwenTransportError):
        call(model)


def test_scripted_validates_arguments_like_the_live_client() -> None:
    model = ScriptedChatModel([completion_from_json(HOLD)])
    with pytest.raises(ValueError):  # noqa: PT011 - pydantic names the field
        call(model, max_tokens=0)
    assert model.remaining == 1


def test_scripted_charges_a_budget_like_the_live_client(clock: ManualClock) -> None:
    budget = DailyTokenBudget(1_000_000, clock)
    answer = completion_from_json(HOLD)
    model = ScriptedChatModel([answer, QwenTransportError("dropped")], budget=budget)
    call(model)
    with pytest.raises(QwenTransportError):
        call(model)
    state = budget.state()
    assert (state.calls, state.spent_tokens, state.unreported_calls) == (
        2,
        answer.usage.total_tokens,
        1,
    )


def test_scripted_refuses_over_budget_without_consuming_the_script(clock: ManualClock) -> None:
    budget = DailyTokenBudget(projected_tokens(PROMPT, 1024) - 1, clock)
    model = ScriptedChatModel([completion_from_json(HOLD)], budget=budget)
    with pytest.raises(BudgetExhausted):
        call(model)
    assert model.remaining == 1
    assert len(model.requests) == 1
    assert budget.state().calls == 0


def test_a_scripted_budget_refusal_is_not_charged(clock: ManualClock) -> None:
    budget = DailyTokenBudget(1_000_000, clock)
    model = ScriptedChatModel([BudgetExhausted("cap reached")], budget=budget)
    with pytest.raises(BudgetExhausted):
        call(model)
    assert budget.state().calls == 0


def test_the_same_script_gives_the_same_run() -> None:
    def run() -> list[str]:
        model = ScriptedChatModel([completion_from_json(HOLD), completion_from_json(FLAT)])
        return [call(model).model_dump_json(), call(model, OTHER).model_dump_json()]

    assert run() == run()


# ================================================================================================
# RecordedChatModel
# ================================================================================================


def test_recorded_replays_by_prompt_hash_in_recorded_order() -> None:
    hold, flat, other = (completion_from_json(o) for o in (HOLD, FLAT, {"stance": "act"}))
    model = RecordedChatModel(
        [
            RecordedCall.of(PROMPT, hold),
            RecordedCall.of(OTHER, other),
            RecordedCall.of(PROMPT, flat),
        ]
    )
    assert model.remaining == 3
    assert call(model, OTHER) == other
    assert call(model) == hold
    assert call(model) == flat
    assert model.remaining == 0
    assert [r.prompt_hash for r in model.requests] == [
        prompt_hash(OTHER),
        prompt_hash(PROMPT),
        prompt_hash(PROMPT),
    ]


def test_an_unrecorded_prompt_is_a_replay_failure_not_an_outage() -> None:
    model = RecordedChatModel([RecordedCall.of(PROMPT, completion_from_json(HOLD))])
    changed = (PROMPT[0], ChatMessage(role="user", content=PROMPT[1].content + " "))
    with pytest.raises(ReplayMismatch, match="not the one that was logged") as caught:
        call(model, changed)
    assert not isinstance(caught.value, QwenError)
    assert model.remaining == 1
    call(model)
    with pytest.raises(ReplayMismatch):
        call(model)


def test_recorded_parameters_must_match_too() -> None:
    request = ChatRequest(messages=PROMPT, json_mode=True, max_tokens=1024, thinking=Thinking.LOW)
    model = RecordedChatModel([RecordedCall.of(PROMPT, completion_from_json(HOLD), request)])
    with pytest.raises(ReplayMismatch, match="thinking") as caught:
        call(model, thinking=Thinking.FULL)
    assert "max_tokens" not in str(caught.value)
    with pytest.raises(ReplayMismatch, match="max_tokens"):
        call(model, max_tokens=4096)
    assert model.remaining == 1
    assert json.loads(call(model).content) == HOLD


def test_a_recording_must_hash_its_own_request() -> None:
    request = ChatRequest(messages=OTHER, json_mode=True, max_tokens=1024, thinking=Thinking.LOW)
    with pytest.raises(ValueError, match="prompt_hash does not match"):
        RecordedCall(
            prompt_hash=prompt_hash(PROMPT), completion=completion_from_json(HOLD), request=request
        )


# --- replay from stored exchanges ---------------------------------------------------------------


class MemoryBlobs:
    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}

    def put(self, data: bytes, media_type: str) -> BlobRef:
        self.data[sha256_hex(data)] = data
        return BlobRef(sha256=sha256_hex(data), media_type=media_type, size=len(data))

    def get(self, sha256: str) -> bytes:
        return self.data[sha256]


def sse_answer(obj: Mapping[str, Any], reasoning: str) -> list[bytes]:
    text = json.dumps(obj)

    def line(payload: Mapping[str, Any]) -> bytes:
        return b"data: " + json.dumps(payload).encode() + b"\n\n"

    usage = {
        "prompt_tokens": 700,
        "completion_tokens": 900,
        "total_tokens": 1600,
        "completion_tokens_details": {"reasoning_tokens": 850},
    }
    return [
        line(
            {
                "id": "chatcmpl-rec",
                "choices": [{"index": 0, "delta": {"reasoning_content": reasoning}}],
            }
        ),
        line({"id": "chatcmpl-rec", "choices": [{"index": 0, "delta": {"content": text[:10]}}]}),
        line({"id": "chatcmpl-rec", "choices": [{"index": 0, "delta": {"content": text[10:]}}]}),
        line(
            {"id": "chatcmpl-rec", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        ),
        line({"id": "chatcmpl-rec", "choices": [], "usage": usage}),
        b"data: [DONE]\n\n",
    ]


def json_answer(obj: Mapping[str, Any]) -> list[bytes]:
    body = {
        "id": "chatcmpl-low",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"content": json.dumps(obj), "reasoning_content": "brief"},
            }
        ],
        "usage": {"prompt_tokens": 70, "completion_tokens": 30, "total_tokens": 100},
    }
    return [json.dumps(body).encode()]


def live_record(
    clock: ManualClock, replies: list[tuple[int, list[bytes]]]
) -> tuple[QwenChatModel, MemoryBlobs, list[Completion], list[tuple[BlobRef, BlobRef]]]:
    """Run the real client over a fake HTTP layer and keep what it stored."""
    queue = list(replies)

    def http(
        url: str, headers: Mapping[str, str], body: bytes, timeout: float
    ) -> tuple[int, Iterator[bytes]]:
        status, chunks = queue.pop(0)
        clock.advance(timedelta(seconds=3))
        return status, iter(chunks)

    blobs = MemoryBlobs()
    model = QwenChatModel(
        credentials=QwenCredentials(api_key="sk-replay-test-key-000000"),
        budget=DailyTokenBudget(1_000_000, clock),
        clock=clock,
        blobs=blobs,
        http=http,
    )
    got = [
        call(model, PROMPT, thinking=Thinking.FULL, max_tokens=8192),
    ]
    pairs = [_pair(model)]
    got.append(call(model, OTHER, thinking=Thinking.LOW))
    pairs.append(_pair(model))
    return model, blobs, got, pairs


def _pair(model: QwenChatModel) -> tuple[BlobRef, BlobRef]:
    answered = [e for e in model.last_exchanges if e.error is None][-1]
    assert answered.request is not None
    assert answered.response is not None
    return answered.request, answered.response


def test_a_recorded_exchange_replays_to_what_the_live_call_returned(
    clock: ManualClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(client, "_sleep", lambda seconds: None)
    replies = [
        (503, [b"busy"]),
        (200, sse_answer(HOLD, "the crowd is long")),
        (200, json_answer(FLAT)),
    ]
    _, blobs, live, pairs = live_record(clock, replies)
    replay = RecordedChatModel.from_exchanges(blobs, pairs)
    assert replay.model_name == "qwen3.8-max"
    assert replay.remaining == 2

    first = call(replay, PROMPT, thinking=Thinking.FULL, max_tokens=8192)
    second = call(replay, OTHER, thinking=Thinking.LOW)
    for replayed, original in ((first, live[0]), (second, live[1])):
        assert replayed.model_dump(exclude={"latency_ms"}) == original.model_dump(
            exclude={"latency_ms"}
        )
    assert json.loads(first.content) == HOLD
    assert first.reasoning == "the crowd is long"
    assert first.usage.total_tokens == 1600
    assert json.loads(second.content) == FLAT


def test_a_replay_checks_the_recorded_parameters(
    clock: ManualClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(client, "_sleep", lambda seconds: None)
    _, blobs, _, pairs = live_record(
        clock, [(200, sse_answer(HOLD, "r")), (200, json_answer(FLAT))]
    )
    replay = RecordedChatModel.from_exchanges(blobs, pairs)
    with pytest.raises(ReplayMismatch, match="thinking"):
        call(replay, PROMPT, thinking=Thinking.LOW, max_tokens=8192)


def test_a_changed_blob_cannot_be_replayed(
    clock: ManualClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(client, "_sleep", lambda seconds: None)
    _, blobs, _, pairs = live_record(
        clock, [(200, sse_answer(HOLD, "r")), (200, json_answer(FLAT))]
    )
    response = pairs[0][1]
    blobs.data[response.sha256] = blobs.data[response.sha256].replace(b"hold", b"sell")
    with pytest.raises(ValueError, match="does not match its reference"):
        RecordedChatModel.from_exchanges(blobs, pairs)


def test_a_stored_completion_and_a_bare_message_list_replay_too() -> None:
    blobs = MemoryBlobs()
    answer = completion_from_json(HOLD, reasoning="kept apart")
    request_ref = blobs.put(
        json.dumps([m.model_dump() for m in PROMPT]).encode(), "application/json"
    )
    response_ref = blobs.put(answer.model_dump_json().encode(), COMPLETION_MEDIA_TYPE)
    replay = RecordedChatModel.from_exchanges(blobs, [(request_ref, response_ref)])
    assert call(replay, thinking=Thinking.OFF) == answer


def test_an_unusable_recorded_response_is_refused() -> None:
    blobs = MemoryBlobs()
    request = ChatRequest(messages=PROMPT, json_mode=True, max_tokens=512, thinking=Thinking.FULL)
    request_ref = blobs.put(json.dumps(request.payload("qwen3.8-max")).encode(), "application/json")
    cut = b'data: {"choices": [{"index": 0, "delta": {"content": "{\\"stan"}}]}\n\n'
    response_ref = blobs.put(cut, "text/event-stream")
    with pytest.raises(ValueError, match=r"unusable.*finish_reason"):
        RecordedChatModel.from_exchanges(blobs, [(request_ref, response_ref)])


def test_recordings_that_name_several_models_need_an_explicit_name() -> None:
    blobs = MemoryBlobs()
    pairs = []
    for model_name in ("qwen3.8-max", "qwen3.8-plus"):
        request = ChatRequest(
            messages=PROMPT, json_mode=True, max_tokens=512, thinking=Thinking.LOW
        )
        request_ref = blobs.put(
            json.dumps(request.payload(model_name)).encode(), "application/json"
        )
        response_ref = blobs.put(
            completion_from_json(HOLD).model_dump_json().encode(), COMPLETION_MEDIA_TYPE
        )
        pairs.append((request_ref, response_ref))
    with pytest.raises(ValueError, match="several models"):
        RecordedChatModel.from_exchanges(blobs, pairs)
    assert RecordedChatModel.from_exchanges(blobs, pairs, model_name="x").model_name == "x"


# ================================================================================================
# FailingChatModel
# ================================================================================================


@pytest.mark.parametrize("error", [QwenTimeout, QwenTransportError, QwenError, BudgetExhausted])
def test_failing_raises_a_fresh_chosen_error_every_time(
    error: type[QwenError] | type[BudgetExhausted],
) -> None:
    model = FailingChatModel(error, message="gateway down")
    with pytest.raises(error, match="gateway down") as first:
        call(model)
    with pytest.raises(error) as second:
        call(model)
    assert first.value is not second.value
    assert len(model.requests) == 2


def test_failing_defaults_to_a_transport_error_and_still_validates() -> None:
    model = FailingChatModel()
    with pytest.raises(QwenTransportError):
        call(model)
    with pytest.raises(ValueError):  # noqa: PT011 - pydantic names the field
        call(model, temperature=9.0)
    assert len(model.requests) == 1


def test_every_stand_in_is_a_chat_model() -> None:
    models: list[ChatModel] = [
        ScriptedChatModel(model_name="scripted"),
        RecordedChatModel([], model_name="recorded"),
        FailingChatModel(model_name="failing"),
    ]
    assert [m.model_name for m in models] == ["scripted", "recorded", "failing"]
