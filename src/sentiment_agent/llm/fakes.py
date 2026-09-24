"""Stand-ins for Qwen, so no test and no keyless replay ever reaches the endpoint.

All three satisfy :class:`~sentiment_agent.types.ChatModel` and validate their arguments exactly as
:class:`~sentiment_agent.llm.client.QwenChatModel` does (both build a
:class:`~sentiment_agent.llm.client.ChatRequest`), so a call that would be refused live is refused
here too, and every request is kept in ``.requests`` for a test to inspect.

* :class:`ScriptedChatModel` answers from a queue: completions to return and exceptions to raise,
  in order. Given a :class:`~sentiment_agent.llm.budget.DailyTokenBudget` it checks and charges it
  the way the live client does.
* :class:`RecordedChatModel` replays logged completions keyed by the prompt hash. ``t2sa replay``
  builds one from the ledger's blobs with :meth:`RecordedChatModel.from_exchanges`, parsing the
  stored response bytes with the same parser the live call used.
* :class:`FailingChatModel` raises a chosen error on every call: the outage path.
"""

import json
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pydantic import model_validator

from sentiment_agent.hashing import canonical_json, sha256_hex
from sentiment_agent.llm.budget import BudgetExhausted, DailyTokenBudget, projected_tokens
from sentiment_agent.llm.client import (
    COMPLETION_MEDIA_TYPE,
    DEFAULT_MODEL,
    STREAM_RESPONSE_MEDIA_TYPE,
    UNREPORTED_USAGE,
    ChatRequest,
    QwenError,
    QwenTransportError,
    parse_response,
    prompt_hash,
)
from sentiment_agent.types import (
    BlobRef,
    BlobStore,
    ChatMessage,
    Completion,
    LlmUsage,
    Model,
    Sha256Hex,
    Thinking,
)


class ScriptExhausted(AssertionError):  # noqa: N818 - reads as the fact a failing test reports
    """A scripted model was asked more often than its script allows: the test under-scripted."""


class ReplayMismatch(LookupError):  # noqa: N818 - reads as the fact a failing replay reports
    """A replayed call has no recording, or differs from the call that was recorded.

    Deliberately not a :class:`~sentiment_agent.llm.client.QwenError`: a replay that cannot find its
    recording has failed to reproduce the cycle, and must not be mistaken for a model outage."""


def _request(
    messages: Sequence[ChatMessage],
    *,
    json_mode: bool,
    max_tokens: int,
    thinking: Thinking,
    temperature: float,
    seed: int | None,
) -> ChatRequest:
    return ChatRequest(
        messages=tuple(messages),
        json_mode=json_mode,
        max_tokens=max_tokens,
        thinking=thinking,
        temperature=temperature,
        seed=seed,
    )


def completion_from_json(obj: Mapping[str, Any], *, reasoning: str = "") -> Completion:
    """A finished completion whose content is ``obj`` as JSON. Deterministic.

    The content is canonical JSON (sorted keys, no whitespace), so equal objects give equal
    completions. The usage is synthetic and marked reported, one token per UTF-8 byte of content
    and reasoning, so a budget charged with it moves by a known, repeatable amount; it is not a
    claim about what Qwen would bill.
    """
    content = canonical_json(dict(obj)).decode("utf-8")
    content_bytes = len(content.encode("utf-8"))
    reasoning_bytes = len(reasoning.encode("utf-8"))
    return Completion(
        content=content,
        reasoning=reasoning,
        usage=LlmUsage(
            prompt_tokens=0,
            completion_tokens=content_bytes + reasoning_bytes,
            reasoning_tokens=reasoning_bytes,
            total_tokens=content_bytes + reasoning_bytes,
            reported=True,
        ),
        finish_reason="stop",
        raw_id="fake-" + sha256_hex(f"{content}\x00{reasoning}".encode())[:24],
        latency_ms=0,
    )


class ScriptedChatModel:
    """Returns queued completions and raises queued exceptions, in order.

    ``requests`` holds every call made, including one the budget refused. An exhausted script
    raises :class:`ScriptExhausted`. With a budget, each call is checked against it first (a
    refusal raises :class:`~sentiment_agent.llm.budget.BudgetExhausted` and consumes nothing) and
    charged after: a returned completion with its usage, a scripted
    :class:`~sentiment_agent.llm.client.QwenError` as an unreported call, which is what the live
    client charges for a call that failed after it was sent.
    """

    def __init__(
        self,
        script: Iterable[Completion | Exception] = (),
        *,
        model_name: str = DEFAULT_MODEL,
        budget: DailyTokenBudget | None = None,
    ) -> None:
        self._queue: deque[Completion | Exception] = deque(script)
        self._model_name = model_name
        self._budget = budget
        self.requests: list[ChatRequest] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def remaining(self) -> int:
        return len(self._queue)

    def push(self, *items: Completion | Exception) -> None:
        self._queue.extend(items)

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_mode: bool,
        max_tokens: int,
        thinking: Thinking,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> Completion:
        request = _request(
            messages,
            json_mode=json_mode,
            max_tokens=max_tokens,
            thinking=thinking,
            temperature=temperature,
            seed=seed,
        )
        self.requests.append(request)
        if self._budget is not None:
            self._budget.check(projected_tokens(request.messages, request.max_tokens))
        if not self._queue:
            raise ScriptExhausted(
                f"the script is empty: call {len(self.requests)} has no scripted response"
            )
        item = self._queue.popleft()
        if isinstance(item, Exception):
            if self._budget is not None and isinstance(item, QwenError):
                self._budget.record(UNREPORTED_USAGE)
            raise item
        if self._budget is not None:
            self._budget.record(item.usage)
        return item


class RecordedCall(Model):
    """One logged completion, filed under the hash of the prompt that produced it.

    With ``request`` present, a replay must also match its parameters (reasoning tier, JSON mode,
    completion cap, temperature, seed), not only its messages.
    """

    prompt_hash: Sha256Hex
    completion: Completion
    request: ChatRequest | None = None

    @model_validator(mode="after")
    def _hash_matches_request(self) -> "RecordedCall":
        if self.request is not None and self.request.prompt_hash != self.prompt_hash:
            raise ValueError("prompt_hash does not match the recorded request's messages")
        return self

    @classmethod
    def of(
        cls,
        messages: Sequence[ChatMessage],
        completion: Completion,
        request: ChatRequest | None = None,
    ) -> "RecordedCall":
        return cls(prompt_hash=prompt_hash(messages), completion=completion, request=request)


_MATCHED_FIELDS = ("json_mode", "max_tokens", "thinking", "temperature", "seed")


def _read_blob(blobs: BlobStore, ref: BlobRef, what: str) -> bytes:
    data = blobs.get(ref.sha256)
    if sha256_hex(data) != ref.sha256 or len(data) != ref.size:
        raise ValueError(f"the {what} blob {ref.sha256[:12]} does not match its reference")
    return data


class RecordedChatModel:
    """Replays logged completions by prompt hash, in the order they were recorded.

    Several recordings under one hash (a truncated answer retried with a larger cap, say) are
    served first-in, first-out. A call whose prompt was never recorded, or whose parameters differ
    from the recording's, raises :class:`ReplayMismatch` and consumes nothing.
    """

    def __init__(
        self, recordings: Iterable[RecordedCall], *, model_name: str = DEFAULT_MODEL
    ) -> None:
        self._queues: dict[str, deque[RecordedCall]] = {}
        for recording in recordings:
            self._queues.setdefault(recording.prompt_hash, deque()).append(recording)
        self._model_name = model_name
        self.requests: list[ChatRequest] = []

    @classmethod
    def from_exchanges(
        cls,
        blobs: BlobStore,
        exchanges: Iterable[tuple[BlobRef, BlobRef]],
        *,
        model_name: str | None = None,
    ) -> "RecordedChatModel":
        """Build a replay from stored ``(request, response)`` blob pairs.

        The request blob is the JSON body the client sent (a bare message list is accepted too,
        and then only the messages are matched). The response blob is the raw body Qwen sent, parsed
        by :func:`~sentiment_agent.llm.client.parse_response` exactly as the live call parsed it,
        or a serialised :class:`~sentiment_agent.types.Completion`
        (:data:`~sentiment_agent.llm.client.COMPLETION_MEDIA_TYPE`). Every blob is checked against
        its reference first, so a replay cannot run on a changed record. The model name is the one
        the requests carried, unless ``model_name`` is given.
        """
        recordings: list[RecordedCall] = []
        models: set[str] = set()
        for number, (request_ref, response_ref) in enumerate(exchanges, start=1):
            payload: object = _json_blob(_read_blob(blobs, request_ref, "request"), number)
            request: ChatRequest | None
            if isinstance(payload, list):
                request = None
                messages = tuple(ChatMessage.model_validate(m) for m in payload)
            elif isinstance(payload, dict):
                request = ChatRequest.from_payload(payload)
                messages = request.messages
                if isinstance(payload.get("model"), str):
                    models.add(payload["model"])
            else:
                raise ValueError(f"exchange {number}: the request blob is not a chat request")
            data = _read_blob(blobs, response_ref, "response")
            if response_ref.media_type == COMPLETION_MEDIA_TYPE:
                completion = Completion.model_validate_json(data)
            else:
                streamed = (
                    request.streamed
                    if request is not None
                    else response_ref.media_type == STREAM_RESPONSE_MEDIA_TYPE
                )
                try:
                    completion = parse_response(data, streamed=streamed)
                except QwenTransportError as exc:
                    raise ValueError(
                        f"exchange {number}: the recorded response is unusable: {exc}"
                    ) from exc
            recordings.append(RecordedCall.of(messages, completion, request))
        if model_name is None:
            if len(models) > 1:
                raise ValueError(f"the recorded requests name several models: {sorted(models)}")
            model_name = models.pop() if models else DEFAULT_MODEL
        return cls(recordings, model_name=model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def remaining(self) -> int:
        return sum(len(queue) for queue in self._queues.values())

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_mode: bool,
        max_tokens: int,
        thinking: Thinking,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> Completion:
        request = _request(
            messages,
            json_mode=json_mode,
            max_tokens=max_tokens,
            thinking=thinking,
            temperature=temperature,
            seed=seed,
        )
        self.requests.append(request)
        key = request.prompt_hash
        queue = self._queues.get(key)
        if not queue:
            raise ReplayMismatch(
                f"call {len(self.requests)}: no recorded completion is left for prompt "
                f"{key[:12]}; the replayed prompt is not the one that was logged"
            )
        recorded = queue[0].request
        if recorded is not None:
            differs = [f for f in _MATCHED_FIELDS if getattr(recorded, f) != getattr(request, f)]
            if differs:
                detail = ", ".join(
                    f"{f} {getattr(request, f)!r} != recorded {getattr(recorded, f)!r}"
                    for f in differs
                )
                raise ReplayMismatch(f"call {len(self.requests)}: {detail}")
        return queue.popleft().completion


def _json_blob(data: bytes, number: int) -> object:
    try:
        return json.loads(data.decode("utf-8"))
    except ValueError as exc:
        raise ValueError(f"exchange {number}: the request blob is not JSON") from exc


class FailingChatModel:
    """Raises ``error(message)`` on every call, a fresh exception each time: the outage path.

    ``error`` is a :class:`~sentiment_agent.llm.client.QwenError` subclass (the outcomes
    ``TIMEOUT`` and ``TRANSPORT_ERROR``) or :class:`~sentiment_agent.llm.budget.BudgetExhausted`
    (``BUDGET_EXHAUSTED``). Arguments are still validated, and every call is kept in ``requests``.
    """

    def __init__(
        self,
        error: type[QwenError] | type[BudgetExhausted] = QwenTransportError,
        *,
        message: str = "scripted failure: the model is unavailable",
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        self._error = error
        self._message = message
        self._model_name = model_name
        self.requests: list[ChatRequest] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_mode: bool,
        max_tokens: int,
        thinking: Thinking,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> Completion:
        self.requests.append(
            _request(
                messages,
                json_mode=json_mode,
                max_tokens=max_tokens,
                thinking=thinking,
                temperature=temperature,
                seed=seed,
            )
        )
        raise self._error(self._message)


__all__ = [
    "FailingChatModel",
    "RecordedCall",
    "RecordedChatModel",
    "ReplayMismatch",
    "ScriptExhausted",
    "ScriptedChatModel",
    "completion_from_json",
]
