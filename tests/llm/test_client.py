"""QwenChatModel over a fake HTTP layer. No test here can reach Qwen: the conftest blocks every
non-loopback socket, and every model below is given a scripted ``http`` callable.

Wire fixtures:

* :data:`ARGUS_RECORDED_RESPONSE` is the unstreamed response ARGUS recorded from the live endpoint
  on 2026-09-12 (``argus/tests/test_qwen.py`` ``REAL_RESPONSE``), byte-for-byte in content.
* The streamed fixtures are built by :func:`sse_stream` in the shape ARGUS verified live and
  documented in ``argus/src/argus/llm/qwen.py`` ``_parse_stream``: ``data:`` lines carrying
  ``chat.completion.chunk`` objects with ``choices[0].delta.content`` /
  ``choices[0].delta.reasoning_content`` increments, a final chunk with ``choices: []`` and the
  usage block, then ``data: [DONE]``. They are constructed, not a byte recording (no raw stream was
  kept by ARGUS). The completion and reasoning counts in :data:`FULL_USAGE` are ARGUS's measured
  FULL figures (4,264 and 4,043); the prompt count is illustrative.
"""

import copy
import email.message
import http.client
import io
import json
import pickle
import ssl
import traceback
import urllib.error
import urllib.request
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from sentiment_agent.clock import ManualClock
from sentiment_agent.hashing import sha256_hex
from sentiment_agent.llm import client
from sentiment_agent.llm.budget import BudgetExhausted, DailyTokenBudget, projected_tokens
from sentiment_agent.llm.client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    ERROR_RESPONSE_MEDIA_TYPE,
    JSON_RESPONSE_MEDIA_TYPE,
    REQUEST_MEDIA_TYPE,
    STREAM_RESPONSE_MEDIA_TYPE,
    ChatRequest,
    QwenChatModel,
    QwenCredentials,
    QwenError,
    QwenTimeout,
    QwenTransportError,
    load_qwen_env,
    parse_response,
    urllib_post_stream,
)
from sentiment_agent.types import BlobRef, ChatMessage, ChatModel, Completion, Thinking

KEY = "sk-t2sa-TEST-KEY-7f3c1d9e0b2a4c6d8e1f"
"""A distinctive stand-in key, so a leak anywhere is found by substring search."""

MESSAGES = (
    ChatMessage(role="system", content="You are the decision-maker. Answer in JSON."),
    ChatMessage(role="user", content="Funding z on NVDAUSDT is 2.4 and the crowd is long."),
)


# ================================================================================================
# Fakes: the HTTP layer and a blob store
# ================================================================================================


class MemoryBlobs:
    """Content-addressed, in memory. Satisfies ``types.BlobStore``."""

    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}
        self.refs: list[BlobRef] = []

    def put(self, data: bytes, media_type: str) -> BlobRef:
        digest = sha256_hex(data)
        self.data[digest] = data
        ref = BlobRef(sha256=digest, media_type=media_type, size=len(data))
        self.refs.append(ref)
        return ref

    def get(self, sha256: str) -> bytes:
        return self.data[sha256]


@dataclass(frozen=True)
class Sent:
    url: str
    headers: dict[str, str]
    body: bytes
    timeout: float

    @property
    def payload(self) -> dict[str, Any]:
        loaded: dict[str, Any] = json.loads(self.body)
        return loaded


Reply = tuple[int, list[bytes]] | BaseException | Callable[[], tuple[int, Iterator[bytes]]]


class FakeHttp:
    """Answers each request with the next scripted reply and records what was sent."""

    def __init__(self, *replies: Reply) -> None:
        self.replies: deque[Reply] = deque(replies)
        self.sent: list[Sent] = []
        self.closed = 0

    def __call__(
        self, url: str, headers: Mapping[str, str], body: bytes, timeout: float
    ) -> tuple[int, Iterator[bytes]]:
        self.sent.append(Sent(url, dict(headers), body, timeout))
        reply = self.replies.popleft()
        if isinstance(reply, BaseException):
            raise reply
        if callable(reply):
            return reply()
        status, chunks = reply
        return status, self._chunks(chunks)

    def _chunks(self, chunks: list[bytes]) -> Iterator[bytes]:
        try:
            yield from chunks
        finally:
            self.closed += 1


# ================================================================================================
# Wire fixtures
# ================================================================================================

ARGUS_RECORDED_RESPONSE: dict[str, Any] = {
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": "OK",
                "reasoning_content": "We need to respond to user. Need final exactly OK.",
                "role": "assistant",
            },
        }
    ],
    "created": 1789196289,
    "id": "chatcmpl-c58829d6",
    "model": "qwen3.8-max",
    "object": "chat.completion",
    "usage": {
        "completion_tokens": 26,
        "completion_tokens_details": {"reasoning_tokens": 23, "text_tokens": 26},
        "prompt_tokens": 66,
        "prompt_tokens_details": {"cached_tokens": 0, "text_tokens": 66},
        "total_tokens": 92,
    },
}

STREAM_ID = "chatcmpl-5d1e8a7b"
FULL_USAGE: dict[str, Any] = {
    "prompt_tokens": 1843,
    "completion_tokens": 4264,
    "total_tokens": 6107,
    "completion_tokens_details": {"reasoning_tokens": 4043, "text_tokens": 4264},
    "prompt_tokens_details": {"cached_tokens": 128, "text_tokens": 1843},
}
ANSWER = '{"stance": "hold", "summary": "funding is rich; the crowd is long \\"everything\\""}'
REASONING = "Funding z is 2.4, above the 2.0 band. The crowd is long; I hold and watch."


def sse(obj: Mapping[str, Any]) -> bytes:
    return b"data: " + json.dumps(obj).encode("utf-8") + b"\n\n"


def chunk(
    delta: Mapping[str, Any] | None = None, *, finish: str | None = None, usage: Any = None
) -> bytes:
    return sse(
        {
            "id": STREAM_ID,
            "object": "chat.completion.chunk",
            "created": 1789196289,
            "model": "qwen3.8-max",
            "choices": [{"index": 0, "delta": dict(delta or {}), "finish_reason": finish}],
            "usage": usage,
        }
    )


def usage_chunk(usage: Mapping[str, Any]) -> bytes:
    return sse(
        {
            "id": STREAM_ID,
            "object": "chat.completion.chunk",
            "created": 1789196289,
            "model": "qwen3.8-max",
            "choices": [],
            "usage": dict(usage),
        }
    )


DONE = b"data: [DONE]\n\n"


def pieces(text: str, size: int = 7) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def sse_stream(
    *,
    answer: str = ANSWER,
    reasoning: str = REASONING,
    finish: str | None = "stop",
    usage: Mapping[str, Any] | None = FULL_USAGE,
    done: bool = True,
) -> list[bytes]:
    out = [chunk({"role": "assistant", "content": "", "reasoning_content": ""})]
    out += [chunk({"reasoning_content": p}) for p in pieces(reasoning)]
    out += [chunk({"content": p}) for p in pieces(answer)]
    if finish is not None:
        out.append(chunk({}, finish=finish))
    if usage is not None:
        out.append(usage_chunk(usage))
    if done:
        out.append(DONE)
    return out


def json_body(obj: Mapping[str, Any]) -> list[bytes]:
    return [json.dumps(obj).encode("utf-8")]


# ================================================================================================
# Fixtures
# ================================================================================================


@pytest.fixture(autouse=True)
def sleeps(monkeypatch: pytest.MonkeyPatch, clock: ManualClock) -> list[float]:
    """No test sleeps: backoff advances the manual clock and is recorded."""
    recorded: list[float] = []

    def fake_sleep(seconds: float) -> None:
        recorded.append(seconds)
        clock.advance(timedelta(seconds=seconds))

    monkeypatch.setattr(client, "_sleep", fake_sleep)
    return recorded


@pytest.fixture
def creds() -> QwenCredentials:
    return QwenCredentials(api_key=KEY)


@pytest.fixture
def budget(clock: ManualClock) -> DailyTokenBudget:
    return DailyTokenBudget(150_000, clock)


@pytest.fixture
def blobs() -> MemoryBlobs:
    return MemoryBlobs()


def make_model(
    http: FakeHttp,
    creds: QwenCredentials,
    budget: DailyTokenBudget,
    clock: ManualClock,
    blobs: MemoryBlobs | None = None,
    **kwargs: Any,
) -> QwenChatModel:
    return QwenChatModel(
        credentials=creds, budget=budget, clock=clock, blobs=blobs, http=http, **kwargs
    )


def ask(
    model: QwenChatModel, thinking: Thinking = Thinking.LOW, *, max_tokens: int = 2048
) -> Completion:
    return model.complete(MESSAGES, json_mode=True, max_tokens=max_tokens, thinking=thinking)


# ================================================================================================
# Credentials and the env file
# ================================================================================================


def write_env(root: Path, text: str) -> None:
    (root / ".secrets").mkdir(parents=True, exist_ok=True)
    (root / ".secrets" / "qwen.env").write_text(text, encoding="utf-8")


def test_load_reads_the_project_env_file(workdir: Path) -> None:
    write_env(
        workdir,
        "# Qwen for t2-sentiment-agent\n"
        f'export BITGET_QWEN_API_KEY="{KEY}"\n'
        "BITGET_QWEN_BASE_URL=https://hackathon.bitgetops.com/v1/   # the gateway\n"
        "BITGET_QWEN_MODEL='qwen3.8-max'\n"
        "UNRELATED=ignored\n",
    )
    loaded = load_qwen_env(workdir)
    assert loaded.base_url == "https://hackathon.bitgetops.com/v1"
    assert loaded.model == "qwen3.8-max"
    assert loaded.authorization_value() == f"Bearer {KEY}"


def test_load_defaults_the_gateway_and_model(workdir: Path) -> None:
    write_env(workdir, f"BITGET_QWEN_API_KEY={KEY}\n")
    loaded = load_qwen_env(workdir)
    assert loaded.base_url == DEFAULT_BASE_URL
    assert loaded.model == DEFAULT_MODEL


def test_load_never_reads_the_process_environment(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BITGET_QWEN_API_KEY", KEY)
    with pytest.raises(QwenError, match=r"\.secrets/qwen\.env was not found"):
        load_qwen_env(workdir)


def test_load_needs_the_key(workdir: Path) -> None:
    write_env(workdir, "BITGET_QWEN_BASE_URL=https://hackathon.bitgetops.com/v1\n")
    with pytest.raises(QwenError, match="BITGET_QWEN_API_KEY is missing"):
        load_qwen_env(workdir)


def test_a_malformed_line_is_named_by_number_never_quoted(workdir: Path) -> None:
    write_env(workdir, f"# comment\n{KEY}\n")
    with pytest.raises(QwenError) as caught:
        load_qwen_env(workdir)
    assert "line 2" in str(caught.value)
    assert KEY not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize(
    ("text", "match"),
    [
        (f'BITGET_QWEN_API_KEY="{KEY}\n', "unterminated quote"),
        (f"BITGET_QWEN_API_KEY={KEY}\nBITGET_QWEN_API_KEY={KEY}\n", "more than once"),
        (f"BITGET_QWEN_API_KEY={KEY}\nBITGET_QWEN_BASE_URL=http://gateway\n", "https"),
        ("BITGET_QWEN_API_KEY=has space inside\n", "printable ASCII"),
    ],
)
def test_load_refuses_bad_files(workdir: Path, text: str, match: str) -> None:
    write_env(workdir, text)
    with pytest.raises(QwenError, match=match) as caught:
        load_qwen_env(workdir)
    assert KEY not in "".join(traceback.format_exception(caught.value))
    assert "has space inside" not in str(caught.value)


def test_credentials_never_show_the_key(creds: QwenCredentials) -> None:
    assert KEY not in repr(creds)
    assert KEY not in str(creds)
    assert KEY not in f"{creds}"
    assert "redacted" in repr(creds)
    assert not hasattr(creds, "__dict__")
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(creds)
    with pytest.raises(TypeError, match="cannot be pickled"):
        copy.copy(creds)


@pytest.mark.parametrize(
    "base_url",
    [
        "http://hackathon.bitgetops.com/v1",
        "https://user:pw@hackathon.bitgetops.com/v1",
        "https://hackathon.bitgetops.com/v1?key=1",
        "https:///v1",
    ],
)
def test_credentials_refuse_unsafe_base_urls(base_url: str) -> None:
    with pytest.raises(ValueError, match="base URL"):
        QwenCredentials(api_key=KEY, base_url=base_url)


def test_credentials_redact(creds: QwenCredentials) -> None:
    assert creds.redact(f"bad key {KEY}!") == "bad key [REDACTED]!"
    assert creds.redact_bytes(f"x{KEY}x".encode()) == b"x[REDACTED]x"


# ================================================================================================
# The request on the wire
# ================================================================================================


@pytest.mark.parametrize(
    ("thinking", "expected", "absent", "streamed"),
    [
        (Thinking.OFF, {"enable_thinking": False}, ("reasoning_effort", "stream"), False),
        (Thinking.LOW, {"reasoning_effort": "low"}, ("enable_thinking", "stream"), False),
        (Thinking.FULL, {"stream": True}, ("enable_thinking", "reasoning_effort"), True),
    ],
)
def test_thinking_is_spelled_as_measured_and_full_streams(
    thinking: Thinking,
    expected: dict[str, Any],
    absent: tuple[str, ...],
    streamed: bool,
    creds: QwenCredentials,
    budget: DailyTokenBudget,
    clock: ManualClock,
) -> None:
    reply = (200, sse_stream()) if streamed else (200, json_body(ARGUS_RECORDED_RESPONSE))
    http = FakeHttp(reply)
    ask(make_model(http, creds, budget, clock), thinking)
    payload = http.sent[0].payload
    for name, value in expected.items():
        assert payload[name] == value
        assert type(payload[name]) is type(value)
    for name in absent:
        assert name not in payload


def test_request_body_headers_and_url(
    creds: QwenCredentials, budget: DailyTokenBudget, clock: ManualClock
) -> None:
    http = FakeHttp((200, json_body(ARGUS_RECORDED_RESPONSE)))
    model = make_model(http, creds, budget, clock)
    model.complete(
        MESSAGES, json_mode=True, max_tokens=900, thinking=Thinking.LOW, temperature=0.2, seed=7
    )
    sent = http.sent[0]
    assert sent.url == "https://hackathon.bitgetops.com/v1/chat/completions"
    assert sent.headers == {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
    assert list(sent.payload) == [
        "model",
        "messages",
        "temperature",
        "max_tokens",
        "reasoning_effort",
        "response_format",
        "seed",
    ]
    assert sent.payload["model"] == "qwen3.8-max"
    assert sent.payload["messages"] == [m.model_dump() for m in MESSAGES]
    assert sent.payload["temperature"] == 0.2
    assert sent.payload["max_tokens"] == 900
    assert sent.payload["response_format"] == {"type": "json_object"}
    assert sent.payload["seed"] == 7
    assert KEY not in sent.body.decode()


def test_plain_mode_and_no_seed_send_neither(
    creds: QwenCredentials, budget: DailyTokenBudget, clock: ManualClock
) -> None:
    http = FakeHttp((200, json_body(ARGUS_RECORDED_RESPONSE)))
    make_model(http, creds, budget, clock).complete(
        MESSAGES, json_mode=False, max_tokens=64, thinking=Thinking.OFF
    )
    assert "response_format" not in http.sent[0].payload
    assert "seed" not in http.sent[0].payload


@pytest.mark.parametrize("thinking", list(Thinking))
def test_request_round_trips_through_its_payload(thinking: Thinking) -> None:
    request = ChatRequest(
        messages=MESSAGES, json_mode=True, max_tokens=512, thinking=thinking, seed=3
    )
    assert ChatRequest.from_payload(request.payload(DEFAULT_MODEL)) == request


@pytest.mark.parametrize(
    "mutation",
    [
        {"stream": True},  # a LOW request is never streamed
        {"enable_thinking": False},  # LOW and OFF at once
        {"reasoning_effort": "high"},
        {"response_format": {"type": "text"}},
    ],
)
def test_from_payload_refuses_what_this_client_never_sends(mutation: dict[str, Any]) -> None:
    payload = ChatRequest(
        messages=MESSAGES, json_mode=True, max_tokens=512, thinking=Thinking.LOW
    ).payload(DEFAULT_MODEL)
    payload.update(mutation)
    with pytest.raises(ValueError):  # noqa: PT011 - each mutation has its own message
        ChatRequest.from_payload(payload)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_tokens": 0},
        {"temperature": 2.5},
        {"temperature": float("nan")},
        {"messages": ()},
    ],
)
def test_invalid_arguments_fail_before_anything_is_sent(
    kwargs: dict[str, Any],
    creds: QwenCredentials,
    budget: DailyTokenBudget,
    clock: ManualClock,
    blobs: MemoryBlobs,
) -> None:
    http = FakeHttp()
    model = make_model(http, creds, budget, clock, blobs)
    call: dict[str, Any] = {"json_mode": True, "max_tokens": 128, "thinking": Thinking.LOW}
    messages = kwargs.pop("messages", MESSAGES)
    call.update(kwargs)
    with pytest.raises(ValueError):  # noqa: PT011 - pydantic names the field
        model.complete(messages, **call)
    assert http.sent == []
    assert blobs.refs == []
    assert budget.state().calls == 0


# ================================================================================================
# Streamed responses (FULL)
# ================================================================================================


def test_full_stream_is_assembled_with_usage_from_the_final_chunk(
    creds: QwenCredentials, budget: DailyTokenBudget, clock: ManualClock
) -> None:
    def slow() -> tuple[int, Iterator[bytes]]:
        def body() -> Iterator[bytes]:
            for part in sse_stream():
                clock.advance(timedelta(seconds=2))
                yield part

        return 200, body()

    model = make_model(FakeHttp(slow), creds, budget, clock)
    got = ask(model, Thinking.FULL)
    assert got.content == ANSWER
    assert json.loads(got.content)["stance"] == "hold"
    assert got.reasoning == REASONING
    assert got.finish_reason == "stop"
    assert got.raw_id == STREAM_ID
    assert got.usage.reported is True
    assert (got.usage.prompt_tokens, got.usage.completion_tokens) == (1843, 4264)
    assert (got.usage.reasoning_tokens, got.usage.total_tokens) == (4043, 6107)
    assert got.usage.cached_tokens == 128
    assert got.latency_ms == 2000 * len(sse_stream())
    assert budget.state().spent_tokens == 6107


@pytest.mark.parametrize("size", [1, 2, 5, 13, 64, 100_000])
def test_stream_parsing_does_not_depend_on_chunk_boundaries(
    size: int, creds: QwenCredentials, budget: DailyTokenBudget, clock: ManualClock
) -> None:
    whole = b"".join(sse_stream())
    reference = parse_response(whole, streamed=True)
    http = FakeHttp((200, [whole[i : i + size] for i in range(0, len(whole), size)]))
    got = ask(make_model(http, creds, budget, clock), Thinking.FULL)
    assert got == reference
    assert (got.content, got.reasoning) == (ANSWER, REASONING)


def test_stream_framing_lines_are_ignored() -> None:
    parts = sse_stream()
    framed = [
        b": keep-alive\n\n",
        b"event: message\r\n",
        b"id: 1\r\n",
        b"retry: 3000\r\n",
        *(p.replace(b"\n", b"\r\n") for p in parts),
    ]
    got = parse_response(b"".join(framed), streamed=True)
    assert (got.content, got.reasoning) == (ANSWER, REASONING)


def test_the_last_usage_block_wins() -> None:
    early = {"prompt_tokens": 1843, "completion_tokens": 10, "total_tokens": 1853}
    parts = sse_stream()
    parts.insert(2, chunk({"reasoning_content": ""}, usage=early))
    got = parse_response(b"".join(parts), streamed=True)
    assert got.usage.total_tokens == 6107


def test_a_stream_without_usage_is_unreported_never_free(
    creds: QwenCredentials, budget: DailyTokenBudget, clock: ManualClock
) -> None:
    http = FakeHttp((200, sse_stream(usage=None)))
    got = ask(make_model(http, creds, budget, clock), Thinking.FULL)
    assert got.usage.reported is False
    state = budget.state()
    assert (state.calls, state.unreported_calls, state.spent_tokens) == (1, 1, 0)


def test_a_finished_stream_without_done_is_accepted(
    creds: QwenCredentials, budget: DailyTokenBudget, clock: ManualClock
) -> None:
    http = FakeHttp((200, sse_stream(done=False)))
    got = ask(make_model(http, creds, budget, clock), Thinking.FULL)
    assert got.content == ANSWER
    assert got.usage.reported is True


def test_a_cut_stream_is_never_accepted_and_is_retried(
    creds: QwenCredentials, budget: DailyTokenBudget, clock: ManualClock, sleeps: list[float]
) -> None:
    cut = sse_stream(finish=None, usage=None, done=False)
    http = FakeHttp((200, cut), (200, sse_stream()))
    model = make_model(http, creds, budget, clock)
    got = ask(model, Thinking.FULL)
    assert got.content == ANSWER
    assert len(http.sent) == 2
    assert sleeps == [1.0]
    first = model.last_exchanges[0]
    assert first.error is not None
    assert "finish_reason" in first.error
    assert budget.state().unreported_calls == 1


def test_a_cut_stream_on_every_attempt_fails(
    creds: QwenCredentials, budget: DailyTokenBudget, clock: ManualClock
) -> None:
    cut = sse_stream(finish=None, usage=None, done=False)
    http = FakeHttp(*[(200, list(cut)) for _ in range(4)])
    with pytest.raises(QwenTransportError, match="finish_reason") as caught:
        ask(make_model(http, creds, budget, clock), Thinking.FULL)
    assert caught.value.attempts == 4


@pytest.mark.parametrize(
    "bad",
    [
        b"data: {not json}\n\n",
        b"data: [1, 2]\n\n",
        b'data: {"choices": {"index": 0}}\n\n',
        b'data: {"choices": [{"index": 0, "delta": {"content": 5}}]}\n\n',
        b"data: \xff\xfe\n\n",
    ],
)
def test_a_corrupt_data_line_fails_the_attempt(
    bad: bytes, creds: QwenCredentials, budget: DailyTokenBudget, clock: ManualClock
) -> None:
    parts = sse_stream()
    parts.insert(3, bad)
    http = FakeHttp((200, parts), (200, sse_stream()))
    model = make_model(http, creds, budget, clock)
    assert ask(model, Thinking.FULL).content == ANSWER
    assert len(http.sent) == 2
    assert model.last_exchanges[0].error is not None


def test_an_error_event_in_the_stream_is_retried_and_redacted(
    creds: QwenCredentials, budget: DailyTokenBudget, clock: ManualClock
) -> None:
    error = sse({"error": {"code": "overloaded", "message": f"upstream busy for {KEY}"}})
    http = FakeHttp((200, [*sse_stream()[:3], error]), (200, sse_stream()))
    model = make_model(http, creds, budget, clock)
    assert ask(model, Thinking.FULL).content == ANSWER
    first_error = model.last_exchanges[0].error
    assert first_error is not None
    assert "overloaded" in first_error
    assert KEY not in first_error


def test_reading_stops_at_done(
    creds: QwenCredentials, budget: DailyTokenBudget, clock: ManualClock
) -> None:
    http = FakeHttp((200, [*sse_stream(), b"data: {garbage after the end}\n\n"]))
    assert ask(make_model(http, creds, budget, clock), Thinking.FULL).content == ANSWER
    assert http.closed == 1


def test_only_the_first_choice_is_the_answer() -> None:
    parts = sse_stream()
    parts.insert(
        2,
        sse({"id": STREAM_ID, "choices": [{"index": 1, "delta": {"content": "SECOND"}}]}),
    )
    assert "SECOND" not in parse_response(b"".join(parts), streamed=True).content


# ================================================================================================
# Unstreamed responses (LOW, OFF)
# ================================================================================================


def test_the_recorded_argus_response_parses(
    creds: QwenCredentials, budget: DailyTokenBudget, clock: ManualClock
) -> None:
    http = FakeHttp((200, json_body(ARGUS_RECORDED_RESPONSE)))
    got = ask(make_model(http, creds, budget, clock), Thinking.LOW)
    assert got.content == "OK"
    assert got.reasoning == "We need to respond to user. Need final exactly OK."
    assert "We need" not in got.content
    assert got.finish_reason == "stop"
    assert got.raw_id == "chatcmpl-c58829d6"
    assert got.usage.model_dump() == {
        "prompt_tokens": 66,
        "completion_tokens": 26,
        "reasoning_tokens": 23,
        "total_tokens": 92,
        "cached_tokens": 0,
        "reported": True,
    }
    assert budget.state().spent_tokens == 92


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {"prompt_tokens": 66, "completion_tokens": 26},
        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        {"prompt_tokens": "66", "completion_tokens": 26, "total_tokens": 92},
        {"prompt_tokens": -1, "completion_tokens": 26, "total_tokens": 92},
        {"prompt_tokens": True, "completion_tokens": 26, "total_tokens": 92},
    ],
)
def test_missing_or_partial_usage_is_unreported(
    usage: Any, creds: QwenCredentials, budget: DailyTokenBudget, clock: ManualClock
) -> None:
    body = copy.deepcopy(ARGUS_RECORDED_RESPONSE)
    if usage is None:
        del body["usage"]
    else:
        body["usage"] = usage
    got = ask(make_model(FakeHttp((200, json_body(body))), creds, budget, clock))
    assert got.usage.reported is False
    assert got.usage.total_tokens == 0
    state = budget.state()
    assert (state.spent_tokens, state.unreported_calls) == (0, 1)


def test_usage_without_detail_blocks_reads_zero_reasoning() -> None:
    body = copy.deepcopy(ARGUS_RECORDED_RESPONSE)
    body["usage"] = {"completion_tokens": 40, "prompt_tokens": 100, "total_tokens": 140}
    got = parse_response(json.dumps(body).encode(), streamed=False)
    assert got.usage.reported is True
    assert (got.usage.reasoning_tokens, got.usage.cached_tokens) == (0, 0)


def test_null_content_becomes_empty_text() -> None:
    body = copy.deepcopy(ARGUS_RECORDED_RESPONSE)
    body["choices"][0]["message"]["content"] = None
    body["choices"][0]["message"]["reasoning_content"] = None
    got = parse_response(json.dumps(body).encode(), streamed=False)
    assert (got.content, got.reasoning) == ("", "")


def test_length_finish_is_passed_through_for_the_caller_to_judge() -> None:
    body = copy.deepcopy(ARGUS_RECORDED_RESPONSE)
    body["choices"][0]["finish_reason"] = "length"
    assert parse_response(json.dumps(body).encode(), streamed=False).finish_reason == "length"


@pytest.mark.parametrize(
    "body",
    [
        b"<html>502 Bad Gateway</html>",
        b"[]",
        b'{"choices": []}',
        b'{"choices": [{"message": "text"}]}',
        json.dumps(
            {"choices": [{"index": 0, "message": {"content": "OK"}, "finish_reason": None}]}
        ).encode(),
        b'{"error": {"code": "InternalError", "message": "try later"}}',
    ],
)
def test_an_unusable_200_body_is_retried(
    body: bytes, creds: QwenCredentials, budget: DailyTokenBudget, clock: ManualClock
) -> None:
    http = FakeHttp((200, [body]), (200, json_body(ARGUS_RECORDED_RESPONSE)))
    model = make_model(http, creds, budget, clock)
    assert ask(model).content == "OK"
    assert len(http.sent) == 2
    with pytest.raises(QwenTransportError):
        parse_response(body, streamed=False)


# ================================================================================================
# Retry policy
# ================================================================================================


def test_retryable_statuses_back_off_then_succeed(
    creds: QwenCredentials, budget: DailyTokenBudget, clock: ManualClock, sleeps: list[float]
) -> None:
    http = FakeHttp(
        (429, [b'{"error": "rate limited"}']),
        (503, [b"<html>busy</html>"]),
        (408, [b""]),
        (200, json_body(ARGUS_RECORDED_RESPONSE)),
    )
    model = make_model(http, creds, budget, clock)
    assert ask(model).content == "OK"
    assert len(http.sent) == 4
    assert sleeps == [1.0, 2.0, 4.0]
    assert [e.status for e in model.last_exchanges] == [429, 503, 408, 200]
    state = budget.state()
    assert (state.calls, state.unreported_calls, state.spent_tokens) == (4, 3, 92)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 413, 422, 302])
def test_other_statuses_fail_fast(
    status: int,
    creds: QwenCredentials,
    budget: DailyTokenBudget,
    clock: ManualClock,
    sleeps: list[float],
) -> None:
    http = FakeHttp((status, [b'{"error": {"message": "bad request"}}']))
    with pytest.raises(QwenTransportError, match=f"HTTP {status}") as caught:
        ask(make_model(http, creds, budget, clock))
    assert caught.value.status == status
    assert caught.value.attempts == 1
    assert len(http.sent) == 1
    assert sleeps == []


def test_retries_are_bounded(
    creds: QwenCredentials, budget: DailyTokenBudget, clock: ManualClock, sleeps: list[float]
) -> None:
    http = FakeHttp(*[(502, [b"bad gateway"]) for _ in range(4)])
    with pytest.raises(QwenTransportError, match="after 4 attempts") as caught:
        ask(make_model(http, creds, budget, clock))
    assert caught.value.attempts == 4
    assert caught.value.status == 502
    assert sleeps == [1.0, 2.0, 4.0]
    assert budget.state().unreported_calls == 4


def test_zero_retries_means_one_attempt(
    creds: QwenCredentials, budget: DailyTokenBudget, clock: ManualClock
) -> None:
    http = FakeHttp((500, [b"oops"]))
    with pytest.raises(QwenTransportError):
        ask(make_model(http, creds, budget, clock, max_transport_retries=0))
    assert len(http.sent) == 1


@pytest.mark.parametrize(
    "error",
    [
        ConnectionResetError("reset by peer"),
        urllib.error.URLError(ConnectionRefusedError("refused")),
        http.client.IncompleteRead(b"partial"),
        http.client.RemoteDisconnected("closed"),
    ],
)
def test_connection_failures_are_retried(
    error: BaseException, creds: QwenCredentials, budget: DailyTokenBudget, clock: ManualClock
) -> None:
    http_fake = FakeHttp(error, (200, json_body(ARGUS_RECORDED_RESPONSE)))
    model = make_model(http_fake, creds, budget, clock)
    assert ask(model).content == "OK"
    assert model.last_exchanges[0].status is None
    assert model.last_exchanges[0].error is not None
    assert "transport failure" in model.last_exchanges[0].error


@pytest.mark.parametrize(
    "error",
    [
        ssl.SSLCertVerificationError("certificate verify failed: self-signed certificate"),
        urllib.error.URLError(ssl.SSLCertVerificationError("certificate verify failed")),
    ],
)
def test_an_unverifiable_certificate_fails_fast(
    error: BaseException,
    creds: QwenCredentials,
    budget: DailyTokenBudget,
    clock: ManualClock,
    sleeps: list[float],
) -> None:
    http_fake = FakeHttp(error, (200, json_body(ARGUS_RECORDED_RESPONSE)))
    with pytest.raises(QwenTransportError, match="certificate verify failed") as caught:
        ask(make_model(http_fake, creds, budget, clock))
    assert caught.value.attempts == 1
    assert len(http_fake.sent) == 1
    assert sleeps == []


def test_a_failure_while_reading_is_retried(
    creds: QwenCredentials, budget: DailyTokenBudget, clock: ManualClock
) -> None:
    def breaks() -> tuple[int, Iterator[bytes]]:
        def body() -> Iterator[bytes]:
            yield sse_stream()[0]
            raise ConnectionResetError("reset mid-stream")

        return 200, body()

    http = FakeHttp(breaks, (200, sse_stream()))
    model = make_model(http, creds, budget, clock)
    assert ask(model, Thinking.FULL).content == ANSWER
    assert model.last_exchanges[0].status == 200


def test_backoff_never_outlives_the_call_deadline(
    creds: QwenCredentials, budget: DailyTokenBudget, clock: ManualClock, sleeps: list[float]
) -> None:
    http = FakeHttp((503, [b"busy"]), (503, [b"busy"]), (200, json_body(ARGUS_RECORDED_RESPONSE)))
    model = make_model(http, creds, budget, clock, timeout_s=2.5)
    with pytest.raises(QwenTransportError, match="too little for the 2s backoff"):
        ask(model)
    assert sleeps == [1.0]
    assert len(http.sent) == 2


# ================================================================================================
# Timeouts
# ================================================================================================


@pytest.mark.parametrize(
    "error",
    [TimeoutError("timed out"), urllib.error.URLError(TimeoutError("connect timed out"))],
)
def test_a_socket_timeout_ends_the_call_without_retry(
    error: BaseException,
    creds: QwenCredentials,
    budget: DailyTokenBudget,
    clock: ManualClock,
    sleeps: list[float],
) -> None:
    http = FakeHttp(error, (200, json_body(ARGUS_RECORDED_RESPONSE)))
    with pytest.raises(QwenTimeout, match="600s call deadline") as caught:
        ask(make_model(http, creds, budget, clock))
    assert caught.value.attempts == 1
    assert len(http.sent) == 1
    assert sleeps == []
    assert budget.state().unreported_calls == 1


def test_a_slow_stream_hits_the_deadline_and_keeps_what_arrived(
    creds: QwenCredentials, budget: DailyTokenBudget, clock: ManualClock, blobs: MemoryBlobs
) -> None:
    def crawl() -> tuple[int, Iterator[bytes]]:
        def body() -> Iterator[bytes]:
            for part in sse_stream():
                clock.advance(timedelta(seconds=250))
                yield part

        return 200, body()

    http = FakeHttp(crawl, (200, sse_stream()))
    model = make_model(http, creds, budget, clock, blobs)
    with pytest.raises(QwenTimeout) as caught:
        ask(model, Thinking.FULL)
    assert caught.value.status == 200
    assert len(http.sent) == 1
    exchange = model.last_exchanges[0]
    assert exchange.response is not None
    partial = blobs.get(exchange.response.sha256)
    assert partial == b"".join(sse_stream()[:3])
    assert exchange.response.media_type == STREAM_RESPONSE_MEDIA_TYPE


def test_the_socket_timeout_is_what_remains_of_the_deadline(
    creds: QwenCredentials, budget: DailyTokenBudget, clock: ManualClock
) -> None:
    def slow_error() -> tuple[int, Iterator[bytes]]:
        clock.advance(timedelta(seconds=30))
        return 503, iter([b"busy"])

    http = FakeHttp(slow_error, (200, json_body(ARGUS_RECORDED_RESPONSE)))
    ask(make_model(http, creds, budget, clock, timeout_s=120))
    assert http.sent[0].timeout == pytest.approx(120)
    assert http.sent[1].timeout == pytest.approx(120 - 30 - 1)


def test_an_overslept_backoff_sends_nothing_more(
    creds: QwenCredentials,
    budget: DailyTokenBudget,
    clock: ManualClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def oversleep(seconds: float) -> None:
        clock.advance(timedelta(seconds=seconds + 100))

    monkeypatch.setattr(client, "_sleep", oversleep)
    http = FakeHttp((503, [b"busy"]), (200, json_body(ARGUS_RECORDED_RESPONSE)))
    with pytest.raises(QwenTimeout) as caught:
        ask(make_model(http, creds, budget, clock, timeout_s=60))
    assert caught.value.attempts == 1
    assert len(http.sent) == 1
    assert budget.state().calls == 1


def test_constructor_refuses_nonsense(
    creds: QwenCredentials, budget: DailyTokenBudget, clock: ManualClock
) -> None:
    for timeout in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="timeout_s"):
            QwenChatModel(credentials=creds, budget=budget, clock=clock, timeout_s=timeout)
    with pytest.raises(ValueError, match="max_transport_retries"):
        QwenChatModel(credentials=creds, budget=budget, clock=clock, max_transport_retries=-1)


# ================================================================================================
# The daily budget
# ================================================================================================


def test_the_budget_is_checked_before_anything_is_sent(
    creds: QwenCredentials, clock: ManualClock, blobs: MemoryBlobs
) -> None:
    tight = DailyTokenBudget(1000, clock)
    http = FakeHttp((200, json_body(ARGUS_RECORDED_RESPONSE)))
    model = make_model(http, creds, tight, clock, blobs)
    with pytest.raises(BudgetExhausted):
        ask(model, max_tokens=2048)
    assert http.sent == []
    assert blobs.refs == []
    assert model.last_exchanges == ()
    assert tight.state().calls == 0


def test_the_projection_is_prompt_bytes_plus_the_completion_cap(
    creds: QwenCredentials, clock: ManualClock
) -> None:
    projected = projected_tokens(MESSAGES, 512)
    exact = DailyTokenBudget(projected, clock)
    http = FakeHttp((200, json_body(ARGUS_RECORDED_RESPONSE)))
    ask(make_model(http, creds, exact, clock), max_tokens=512)
    assert len(http.sent) == 1

    short = DailyTokenBudget(projected - 1, clock)
    with pytest.raises(BudgetExhausted):
        ask(make_model(FakeHttp(), creds, short, clock), max_tokens=512)


def test_the_budget_is_rechecked_before_a_retry(creds: QwenCredentials, clock: ManualClock) -> None:
    # The first attempt is billed (a usage block arrives) but its body is unusable; the spend
    # it reported leaves too little for a second attempt, so the retry is refused unsent.
    billed_but_broken = {
        "usage": {"prompt_tokens": 90, "completion_tokens": 10, "total_tokens": 100}
    }
    projected = projected_tokens(MESSAGES, 64)
    budget = DailyTokenBudget(projected + 50, clock)
    http = FakeHttp((200, json_body(billed_but_broken)), (200, json_body(ARGUS_RECORDED_RESPONSE)))
    model = make_model(http, creds, budget, clock)
    with pytest.raises(BudgetExhausted):
        ask(model, max_tokens=64)
    assert len(http.sent) == 1
    assert budget.state().spent_tokens == 100
    assert model.last_exchanges[0].usage.total_tokens == 100


# ================================================================================================
# The key never leaks; the evidence is kept
# ================================================================================================


def test_the_key_never_reaches_an_error_a_repr_or_a_blob(
    creds: QwenCredentials, budget: DailyTokenBudget, clock: ManualClock, blobs: MemoryBlobs
) -> None:
    echo = json.dumps({"error": {"message": f"Incorrect API key provided: {KEY}"}}).encode()
    http = FakeHttp((401, [echo]))
    model = make_model(http, creds, budget, clock, blobs)
    with pytest.raises(QwenTransportError) as caught:
        ask(model)
    assert http.sent[0].headers["Authorization"] == f"Bearer {KEY}"  # used where it belongs
    rendered = "".join(traceback.format_exception(caught.value))
    assert "Incorrect API key" in rendered
    assert KEY not in rendered
    assert KEY not in repr(caught.value.args)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert KEY not in repr(model)
    assert KEY not in repr(model.last_exchanges)
    for data in blobs.data.values():
        assert KEY.encode() not in data
    response = model.last_exchanges[0].response
    assert response is not None
    assert b"[REDACTED]" in blobs.get(response.sha256)


def test_a_transport_exception_carrying_the_key_is_redacted(
    creds: QwenCredentials, budget: DailyTokenBudget, clock: ManualClock
) -> None:
    http = FakeHttp(*[OSError(f"proxy said {KEY}") for _ in range(4)])
    with pytest.raises(QwenTransportError) as caught:
        ask(make_model(http, creds, budget, clock))
    rendered = "".join(traceback.format_exception(caught.value))
    assert "proxy said [REDACTED]" in rendered
    assert KEY not in rendered


def test_every_attempt_is_kept_as_evidence(
    creds: QwenCredentials, budget: DailyTokenBudget, clock: ManualClock, blobs: MemoryBlobs
) -> None:
    http = FakeHttp((503, [b"<html>busy</html>"]), (200, sse_stream()))
    model = make_model(http, creds, budget, clock, blobs)
    ask(model, Thinking.FULL)
    first, second = model.last_exchanges
    assert (first.attempt, first.status, second.attempt, second.status) == (1, 503, 2, 200)
    assert first.error is not None
    assert "HTTP 503" in first.error
    assert "busy" in first.error
    assert second.error is None
    assert first.request == second.request
    assert first.request is not None
    assert first.request.media_type == REQUEST_MEDIA_TYPE
    assert blobs.get(first.request.sha256) == http.sent[0].body
    assert first.response is not None
    assert second.response is not None
    assert first.response.media_type == ERROR_RESPONSE_MEDIA_TYPE
    assert second.response.media_type == STREAM_RESPONSE_MEDIA_TYPE
    assert blobs.get(second.response.sha256) == b"".join(sse_stream())
    assert second.usage.total_tokens == 6107


def test_an_unstreamed_answer_is_stored_as_json(
    creds: QwenCredentials, budget: DailyTokenBudget, clock: ManualClock, blobs: MemoryBlobs
) -> None:
    model = make_model(
        FakeHttp((200, json_body(ARGUS_RECORDED_RESPONSE))), creds, budget, clock, blobs
    )
    ask(model)
    response = model.last_exchanges[0].response
    assert response is not None
    assert response.media_type == JSON_RESPONSE_MEDIA_TYPE
    assert json.loads(blobs.get(response.sha256)) == ARGUS_RECORDED_RESPONSE


def test_last_exchanges_belong_to_the_latest_call(
    creds: QwenCredentials, budget: DailyTokenBudget, clock: ManualClock
) -> None:
    http = FakeHttp(
        (500, [b"x"]),
        (200, json_body(ARGUS_RECORDED_RESPONSE)),
        (200, json_body(ARGUS_RECORDED_RESPONSE)),
    )
    model = make_model(http, creds, budget, clock)
    ask(model)
    assert len(model.last_exchanges) == 2
    ask(model)
    assert len(model.last_exchanges) == 1


def test_it_is_a_chat_model(
    creds: QwenCredentials, budget: DailyTokenBudget, clock: ManualClock
) -> None:
    model: ChatModel = QwenChatModel(credentials=creds, budget=budget, clock=clock)
    assert model.model_name == "qwen3.8-max"
    assert "hackathon.bitgetops.com" in repr(model)


# ================================================================================================
# The production transport, with urlopen replaced (no socket is opened)
# ================================================================================================


class FakeResponse:
    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self._body = io.BytesIO(body)
        self.closed = False

    def readline(self) -> bytes:
        return self._body.readline()

    def close(self) -> None:
        self.closed = True


def test_urllib_transport_refuses_plain_http() -> None:
    with pytest.raises(ValueError, match="https"):
        urllib_post_stream("http://hackathon.bitgetops.com/v1/chat/completions", {}, b"{}", 5.0)


def test_urllib_transport_streams_lines_and_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    stream = b"".join(sse_stream())
    response = FakeResponse(200, stream)

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeResponse:
        seen["url"] = request.full_url
        seen["method"] = request.get_method()
        seen["auth"] = request.get_header("Authorization")
        seen["body"] = request.data
        seen["timeout"] = timeout
        return response

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    status, lines = urllib_post_stream(
        "https://hackathon.bitgetops.com/v1/chat/completions",
        {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
        b'{"model":"qwen3.8-max"}',
        42.0,
    )
    received = list(lines)
    assert status == 200
    assert b"".join(received) == stream
    assert all(line.endswith(b"\n") for line in received)
    assert response.closed
    assert seen == {
        "url": "https://hackathon.bitgetops.com/v1/chat/completions",
        "method": "POST",
        "auth": f"Bearer {KEY}",
        "body": b'{"model":"qwen3.8-max"}',
        "timeout": 42.0,
    }


def test_urllib_transport_returns_error_statuses_with_their_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = b'{"error": {"message": "rate limited"}}'

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeResponse:
        raise urllib.error.HTTPError(
            request.full_url, 429, "Too Many Requests", email.message.Message(), io.BytesIO(body)
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    status, lines = urllib_post_stream(
        "https://hackathon.bitgetops.com/v1/chat/completions", {}, b"{}", 5.0
    )
    assert status == 429
    assert b"".join(lines) == body
