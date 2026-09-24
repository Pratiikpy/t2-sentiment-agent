"""Qwen ``qwen3.8-max`` over the Bitget hackathon gateway. The only code that ever holds the key.

Ported from ARGUS ``src/argus/llm/qwen.py`` (``QwenClient``, ``_parse``, ``_parse_stream``), MIT,
same author, at commit ``3dec6baf9dfa7be37c7b452e26a9b139df252f75``, sha256
``dcc4ba1077a63ca2dca4ee7ece18e566b90b29be2a3a5bfd1c6a987c2678daa9``. The ARGUS repository is
private, so its licence and every pin are recorded in ``third_party/argus/PROVENANCE.md``. Copied,
never imported (the independence rule in ``NOTICE.md``).

Wire facts, each measured by ARGUS against the live endpoint on 2026-09-12 (DESIGN §3.9) and kept
exactly as measured, including the request headers:

* OpenAI-compatible ``POST {base}/chat/completions`` with ``Authorization: Bearer`` and
  ``Content-Type: application/json``; ``response_format: {"type": "json_object"}`` works.
* ``reasoning_content`` arrives apart from ``content`` and is kept apart: scratch work is logged,
  never parsed as the answer.
* Reasoning tiers are spelled ``enable_thinking: false`` (OFF) and ``reasoning_effort: "low"``
  (LOW); FULL sends neither, because the endpoint reasons by default. Measured completion cost on
  one prompt: OFF 116, LOW 828, FULL 4,264 tokens.
* The gateway closes an unstreamed request at about 120 s, and a FULL call took 112 s, so FULL is
  always streamed. A stream is ``data:`` lines of ``chat.completion.chunk`` objects whose
  ``choices[0].delta`` carries ``content`` and ``reasoning_content`` increments; the final chunk
  carries ``choices: []`` and the authoritative ``usage``; the stream ends with ``data: [DONE]``.

What changed from ARGUS, and why:

* **The key comes from one file, never the environment.** ``<project>/.secrets/qwen.env`` is read by
  :func:`load_qwen_env`; the process environment is not consulted, so no inherited variable can
  silently put a different key (or a live Bitget key) in this seat.
* **An incomplete answer is never accepted as a complete one.** ARGUS defaulted a stream with no
  ``finish_reason`` to ``"stop"`` and skipped data lines that were not JSON. A stream the gateway
  cut, or a chunk that arrived corrupted, would then read as a finished answer with a hole in it.
  Here both are transport errors and are retried.
* **Absent usage is unreported on every path.** ARGUS marked it on the streamed path only; its
  unstreamed parser reported a missing ``usage`` as a measured zero.
* **One deadline per call.** ``timeout_s`` bounds the whole :meth:`QwenChatModel.complete`, retries
  and backoff included, so a decision cycle can never wait longer than the policy's call timeout.
  A timeout is not retried: the time it would need is the time that has already run out.
* **Every exchange is evidence.** The exact request body and the raw response bytes (the whole
  stream, or an error page) are stored as blobs, and each attempt is listed in
  :attr:`QwenChatModel.last_exchanges`, so a decision can be replayed from what Qwen actually sent.
* **No answer cache.** ARGUS cached identical requests. Here an identical prompt means an identical
  snapshot, which the trigger engine already de-duplicates, and a cache would make a logged
  completion unexplained by any exchange.
* **Tool calling is not wired.** The decision contract is one JSON object; nothing asks for tools.

The key is held by :class:`QwenCredentials` and leaves it only as the ``Authorization`` header
value. It is scrubbed from every error message and every stored blob, it is absent from every repr,
and no exception raised here chains an exception that could carry it.
"""

import http.client
import json
import math
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Final, Literal, NoReturn, Protocol

from pydantic import Field, model_validator

from sentiment_agent.hashing import content_hash
from sentiment_agent.llm.budget import DailyTokenBudget, projected_tokens
from sentiment_agent.types import (
    BlobRef,
    BlobStore,
    ChatMessage,
    Clock,
    Completion,
    LlmUsage,
    Model,
    Thinking,
    UtcDatetime,
)

QWEN_ENV_FILE: Final = ".secrets/qwen.env"
ENV_API_KEY: Final = "BITGET_QWEN_API_KEY"
ENV_BASE_URL: Final = "BITGET_QWEN_BASE_URL"
ENV_MODEL: Final = "BITGET_QWEN_MODEL"
DEFAULT_BASE_URL: Final = "https://hackathon.bitgetops.com/v1"
DEFAULT_MODEL: Final = "qwen3.8-max"
CHAT_PATH: Final = "/chat/completions"

REQUEST_MEDIA_TYPE: Final = "application/json"
"""The exact request body sent (the key travels in a header, never in the body)."""
JSON_RESPONSE_MEDIA_TYPE: Final = "application/json"
STREAM_RESPONSE_MEDIA_TYPE: Final = "text/event-stream"
ERROR_RESPONSE_MEDIA_TYPE: Final = "application/octet-stream"
"""A non-2xx body, stored as received: the gateway's error pages are not always JSON."""
COMPLETION_MEDIA_TYPE: Final = "application/vnd.t2sa.completion+json"
"""A :class:`~sentiment_agent.types.Completion` serialised by ``model_dump_json``; replayable."""

RETRYABLE_STATUSES: Final = frozenset({408, 429})
"""Retried with backoff, as is every 5xx. Every other non-2xx status fails fast."""
BACKOFF_BASE_S: Final = 1.0
"""Backoff before retry *n* is ``BACKOFF_BASE_S * 2 ** (n - 1)``: 1 s, 2 s, 4 s (ARGUS)."""
MAX_RESPONSE_BYTES: Final = 16 * 1024 * 1024
"""A FULL stream measured ~4.3k completion tokens; 16 MiB is orders of magnitude above that."""
ERROR_DETAIL_CHARS: Final = 300
REDACTED: Final = "[REDACTED]"

_KEY_SHAPE = re.compile(r"[!-~]+")
_MODEL_SHAPE = re.compile(r"[A-Za-z0-9._:/-]+")
_ENV_LINE = re.compile(r"(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)")


# ================================================================================================
# Errors
# ================================================================================================


class QwenError(RuntimeError):
    """A call to Qwen failed. The message never carries the key, even when echoing the server."""

    def __init__(self, message: str, *, status: int | None = None, attempts: int = 0) -> None:
        super().__init__(message)
        self.status = status
        """The HTTP status of the last attempt, when one arrived."""
        self.attempts = attempts
        """How many requests were sent before giving up."""


class QwenTimeout(QwenError):  # noqa: N818 - the name is the module contract (DESIGN M6)
    """No complete answer inside the call deadline. The decision outcome is ``TIMEOUT``."""


class QwenTransportError(QwenError):
    """The endpoint could not be reached, refused the request, or sent an unusable response.

    Covers the retryable failures once retries are spent (408, 429, 5xx, a dropped connection, a
    cut or corrupted stream) and the ones that fail fast (every other non-2xx status). The decision
    outcome is ``TRANSPORT_ERROR`` either way.
    """


# ================================================================================================
# Credentials
# ================================================================================================


def _validate_base_url(url: str) -> str:
    candidate = url.strip()
    parts = urllib.parse.urlsplit(candidate)
    if parts.scheme != "https" or not parts.hostname:
        raise ValueError("the Qwen base URL must be an https URL with a host")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("the Qwen base URL may not carry credentials, a query or a fragment")
    return candidate.rstrip("/")


class QwenCredentials:
    """The Qwen key, the gateway URL and the model name. The key is never shown.

    ``repr`` and ``str`` redact it, there is no ``__dict__`` for ``vars()`` to print, and pickling
    and copying are refused so the key cannot ride along into a serialised object by accident. It
    leaves this object only through :meth:`authorization_value`.
    """

    __slots__ = ("_api_key", "_base_url", "_model")

    def __init__(
        self, *, api_key: str, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL
    ) -> None:
        if not isinstance(api_key, str) or not _KEY_SHAPE.fullmatch(api_key):
            # The value is deliberately not echoed: a malformed key is still a key.
            raise ValueError("the Qwen API key must be non-empty printable ASCII with no spaces")
        if not isinstance(model, str) or not _MODEL_SHAPE.fullmatch(model):
            raise ValueError("the Qwen model name must be a non-empty identifier with no spaces")
        self._api_key = api_key
        self._base_url = _validate_base_url(base_url)
        self._model = model

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def model(self) -> str:
        return self._model

    def authorization_value(self) -> str:
        """The ``Authorization`` header value. The one place the key leaves this object."""
        return f"Bearer {self._api_key}"

    def redact(self, text: str) -> str:
        """``text`` with every occurrence of the key replaced by ``[REDACTED]``."""
        return text.replace(self._api_key, REDACTED)

    def redact_bytes(self, data: bytes) -> bytes:
        return data.replace(self._api_key.encode("ascii"), REDACTED.encode("ascii"))

    def __repr__(self) -> str:
        return (
            f"QwenCredentials(base_url={self._base_url!r}, model={self._model!r}, "
            "api_key=<redacted>)"
        )

    __str__ = __repr__

    def __reduce_ex__(self, protocol: object) -> NoReturn:
        raise TypeError("QwenCredentials cannot be pickled or copied: it holds the Qwen key")


def _parse_env(text: str) -> dict[str, str]:
    """``NAME=value`` lines; ``#`` comments, ``export`` and matching quotes allowed.

    Errors name the line number and the variable, never the line's content: the malformed line may
    be the one holding the key.
    """
    values: dict[str, str] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _ENV_LINE.fullmatch(line)
        if match is None:
            raise QwenError(f"{QWEN_ENV_FILE} line {number} is not of the form NAME=value")
        name, value = match.group(1), match.group(2).strip()
        if value[:1] in {"'", '"'}:
            if len(value) < 2 or value[-1] != value[0]:
                raise QwenError(f"{QWEN_ENV_FILE} line {number} has an unterminated quote")
            value = value[1:-1]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        if name in values:
            raise QwenError(f"{QWEN_ENV_FILE} sets {name} more than once")
        values[name] = value
    return values


def load_qwen_env(project_root: Path) -> QwenCredentials:
    """Read ``<project_root>/.secrets/qwen.env``. The only source of the Qwen key.

    ``BITGET_QWEN_API_KEY`` is required. ``BITGET_QWEN_BASE_URL`` defaults to the hackathon gateway
    and ``BITGET_QWEN_MODEL`` to ``qwen3.8-max``. Other variables in the file are ignored. The
    process environment is never read.
    """
    path = Path(project_root).joinpath(*QWEN_ENV_FILE.split("/"))
    if not path.is_file():
        raise QwenError(
            f"{QWEN_ENV_FILE} was not found under {Path(project_root)}. The owner creates it with "
            f"{ENV_API_KEY}=<key>; until then the agent runs with a recorded or scripted model."
        )
    text: str | None
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        # Not chained and not echoed: a decode error quotes the offending bytes.
        text = None
    if text is None:
        raise QwenError(f"{QWEN_ENV_FILE} could not be read as UTF-8 text")
    values = _parse_env(text)
    api_key = values.get(ENV_API_KEY, "")
    if not api_key:
        raise QwenError(f"{ENV_API_KEY} is missing or empty in {QWEN_ENV_FILE}")
    problem: str | None = None
    try:
        return QwenCredentials(
            api_key=api_key,
            base_url=values.get(ENV_BASE_URL) or DEFAULT_BASE_URL,
            model=values.get(ENV_MODEL) or DEFAULT_MODEL,
        )
    except ValueError as exc:
        problem = str(exc)
    raise QwenError(f"{QWEN_ENV_FILE}: {problem}")


# ================================================================================================
# The request, and how it is spelled on the wire
# ================================================================================================


def prompt_hash(messages: Sequence[ChatMessage]) -> str:
    """SHA-256 of the canonical JSON of the messages: the key a recorded completion replays by."""
    return content_hash(tuple(messages))


def thinking_fields(thinking: Thinking) -> dict[str, Any]:
    """The request fields that select a reasoning tier, spelled as measured (see module doc).

    ``extra_body`` nesting does not work on this endpoint (ARGUS measured reasoning left fully on at
    2,457 tokens), so the fields are top-level.
    """
    if thinking is Thinking.OFF:
        return {"enable_thinking": False}
    if thinking is Thinking.LOW:
        return {"reasoning_effort": "low"}
    return {}


def _thinking_from_payload(payload: Mapping[str, Any]) -> Thinking:
    has_off = "enable_thinking" in payload
    has_low = "reasoning_effort" in payload
    if has_off and has_low:
        raise ValueError("a payload sets both enable_thinking and reasoning_effort")
    if has_off:
        if payload["enable_thinking"] is not False:
            raise ValueError("enable_thinking is only ever sent as false")
        return Thinking.OFF
    if has_low:
        if payload["reasoning_effort"] != "low":
            raise ValueError("reasoning_effort is only ever sent as 'low'")
        return Thinking.LOW
    return Thinking.FULL


class ChatRequest(Model):
    """One call as the caller asked for it. Invalid arguments fail here, before anything is sent."""

    messages: tuple[ChatMessage, ...] = Field(min_length=1)
    json_mode: bool
    max_tokens: int = Field(ge=1)
    thinking: Thinking
    temperature: float = Field(default=0.0, ge=0, le=2, allow_inf_nan=False)
    seed: int | None = None

    @property
    def streamed(self) -> bool:
        """FULL is always streamed (the gateway closes unstreamed requests at ~120 s)."""
        return self.thinking is Thinking.FULL

    @property
    def prompt_hash(self) -> str:
        return prompt_hash(self.messages)

    def payload(self, model: str) -> dict[str, Any]:
        """The JSON body, in the field order ARGUS verified."""
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in self.messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        body.update(thinking_fields(self.thinking))
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.seed is not None:
            body["seed"] = self.seed
        if self.streamed:
            body["stream"] = True
        return body

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ChatRequest":
        """Invert :meth:`payload`. Refuses a body this client would not have sent."""
        thinking = _thinking_from_payload(payload)
        if (payload.get("stream") is True) != (thinking is Thinking.FULL):
            raise ValueError("a FULL request is streamed and no other tier is")
        response_format = payload.get("response_format")
        if response_format not in (None, {"type": "json_object"}):
            raise ValueError("response_format is only ever json_object")
        return cls(
            messages=tuple(ChatMessage.model_validate(m) for m in payload.get("messages", ())),
            json_mode=response_format is not None,
            max_tokens=payload.get("max_tokens", 0),
            thinking=thinking,
            temperature=payload.get("temperature", 0.0),
            seed=payload.get("seed"),
        )


def encode_payload(payload: Mapping[str, Any]) -> bytes:
    """The exact bytes sent and stored: UTF-8, compact, in insertion order."""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


# ================================================================================================
# Response parsing
# ================================================================================================

UNREPORTED_USAGE: Final = LlmUsage(
    prompt_tokens=0, completion_tokens=0, reasoning_tokens=0, total_tokens=0, reported=False
)
"""The usage of a call the endpoint did not bill in writing: unmeasured, not free."""


class _BadResponseError(Exception):
    """A response that cannot be used as an answer. Internal; surfaces as a transport error."""

    def __init__(self, message: str, usage: LlmUsage = UNREPORTED_USAGE) -> None:
        super().__init__(message)
        self.message = message
        self.usage = usage


def _count(block: object, name: str) -> int | None:
    if not isinstance(block, Mapping):
        return None
    value = block.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def usage_from_block(block: object) -> LlmUsage:
    """Read an OpenAI-shaped ``usage`` block.

    Reported only when ``prompt_tokens``, ``completion_tokens`` and a non-zero ``total_tokens`` are
    all present as non-negative integers. Anything less is unreported rather than completed with
    zeros: a zero written in for a missing field is a measurement nobody made. The detail blocks
    are optional (ARGUS's tool-call response carried none), and an absent detail reads as 0.
    """
    if not isinstance(block, Mapping):
        return UNREPORTED_USAGE
    prompt = _count(block, "prompt_tokens")
    completion = _count(block, "completion_tokens")
    total = _count(block, "total_tokens")
    if prompt is None or completion is None or total is None or total == 0:
        return UNREPORTED_USAGE
    reasoning = _count(block.get("completion_tokens_details"), "reasoning_tokens") or 0
    cached = _count(block.get("prompt_tokens_details"), "cached_tokens") or 0
    return LlmUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        reasoning_tokens=reasoning,
        total_tokens=total,
        cached_tokens=cached,
        reported=True,
    )


def _error_detail(value: object) -> str:
    if isinstance(value, Mapping):
        message = value.get("message")
        code = value.get("code")
        return f"{code}: {message}" if code is not None else str(message)
    return str(value)


class _StreamAssembler:
    """Server-Sent Events in, one :class:`Completion` out. Fed arbitrary byte chunks.

    Only ``data:`` lines carry events; comments (``:``), ``event:``, ``id:``, ``retry:`` and blank
    lines are framing. Each data line holds one JSON chunk (the shape ARGUS verified). A data line
    that is not a JSON object, or is not UTF-8, is corruption and fails the attempt: skipping it
    could drop a piece of the answer without anyone knowing.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._content: list[str] = []
        self._reasoning: list[str] = []
        self._finish: str | None = None
        self._raw_id = ""
        self._usage: object = None
        self.done = False
        """``data: [DONE]`` arrived."""

    @property
    def usage(self) -> LlmUsage:
        return usage_from_block(self._usage)

    def feed(self, data: bytes) -> None:
        if self.done:
            return
        self._buffer += data
        while not self.done:
            newline = self._buffer.find(b"\n")
            if newline < 0:
                return
            line = bytes(self._buffer[:newline])
            del self._buffer[: newline + 1]
            self._line(line)

    def close(self) -> None:
        """End of body: a final line without a trailing newline is still a line."""
        if not self.done and self._buffer.strip():
            line = bytes(self._buffer)
            self._buffer.clear()
            self._line(line)

    def result(self, latency_ms: int) -> Completion:
        self.close()
        if self._finish is None:
            raise _BadResponseError(
                "the stream ended before the model finished (no finish_reason); a cut answer is "
                "not accepted",
                self.usage,
            )
        return Completion(
            content="".join(self._content),
            reasoning="".join(self._reasoning),
            usage=self.usage,
            finish_reason=self._finish,
            raw_id=self._raw_id,
            latency_ms=latency_ms,
        )

    def _line(self, raw: bytes) -> None:
        try:
            text = raw.decode("utf-8").rstrip("\r")
        except UnicodeDecodeError:
            text = None
        if text is None:
            raise _BadResponseError("the stream carried a line that is not UTF-8", self.usage)
        if not text.startswith("data:"):
            return
        body = text[len("data:") :].strip()
        if body == "[DONE]":
            self.done = True
            return
        try:
            chunk = json.loads(body)
        except ValueError:
            chunk = None
        if not isinstance(chunk, dict):
            raise _BadResponseError(
                "the stream carried a data line that is not a JSON object", self.usage
            )
        if chunk.get("error"):
            raise _BadResponseError(
                f"the stream carried an error: {_error_detail(chunk['error'])}", self.usage
            )
        raw_id = chunk.get("id")
        if isinstance(raw_id, str) and raw_id:
            self._raw_id = raw_id
        if chunk.get("usage"):
            # The last usage block wins: the final chunk carries the authoritative totals.
            self._usage = chunk["usage"]
        choices = chunk.get("choices") or []
        if not isinstance(choices, list):
            raise _BadResponseError(
                "the stream carried a chunk whose choices is not a list", self.usage
            )
        for choice in choices:
            self._choice(choice)

    def _choice(self, choice: object) -> None:
        if not isinstance(choice, dict):
            raise _BadResponseError("the stream carried a choice that is not an object", self.usage)
        if choice.get("index", 0) != 0:
            return  # n=1 is requested; a second choice would not be the answer
        delta = choice.get("delta") or {}
        if not isinstance(delta, dict):
            raise _BadResponseError("the stream carried a delta that is not an object", self.usage)
        for field, sink in (("content", self._content), ("reasoning_content", self._reasoning)):
            piece = delta.get(field)
            if piece is None:
                continue
            if not isinstance(piece, str):
                raise _BadResponseError(f"the stream carried a non-text {field} delta", self.usage)
            sink.append(piece)
        finish = choice.get("finish_reason")
        if isinstance(finish, str) and finish:
            self._finish = finish


def _parse_body(data: bytes, latency_ms: int) -> Completion:
    """An unstreamed ``chat.completion`` body."""
    try:
        obj = json.loads(data.decode("utf-8"))
    except ValueError:  # UnicodeDecodeError is a ValueError
        obj = None
    if not isinstance(obj, dict):
        raise _BadResponseError("the response body is not a JSON object")
    usage = usage_from_block(obj.get("usage"))
    if obj.get("error"):
        raise _BadResponseError(
            f"the response carried an error: {_error_detail(obj['error'])}", usage
        )
    choices = obj.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise _BadResponseError("unexpected response shape: no choices", usage)
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise _BadResponseError("unexpected response shape: no message", usage)
    content = message.get("content")
    reasoning = message.get("reasoning_content")
    if not isinstance(content, str | None) or not isinstance(reasoning, str | None):
        raise _BadResponseError("unexpected response shape: non-text content", usage)
    finish = choice.get("finish_reason")
    if not isinstance(finish, str) or not finish:
        raise _BadResponseError("the response has no finish_reason; it may be incomplete", usage)
    raw_id = obj.get("id")
    return Completion(
        content=content or "",
        reasoning=reasoning or "",
        usage=usage,
        finish_reason=finish,
        raw_id=raw_id if isinstance(raw_id, str) else "",
        latency_ms=latency_ms,
    )


def parse_response(data: bytes, *, streamed: bool, latency_ms: int = 0) -> Completion:
    """Parse a stored response body exactly as the live call parsed it (used by replay).

    Raises :class:`QwenTransportError` when the body is not a complete answer.
    """
    try:
        if streamed:
            assembler = _StreamAssembler()
            assembler.feed(data)
            return assembler.result(latency_ms)
        return _parse_body(data, latency_ms)
    except _BadResponseError as exc:
        problem = exc.message
    raise QwenTransportError(problem)


# ================================================================================================
# HTTP
# ================================================================================================

HttpPostStream = Callable[[str, Mapping[str, str], bytes, float], tuple[int, Iterator[bytes]]]
"""``(url, headers, body, socket_timeout_s) -> (status, body chunks)``.

Returns the status of any HTTP response, error statuses included, with the body as an iterator of
byte chunks of any size. Raises ``TimeoutError`` when a socket read times out, and ``OSError`` or
``http.client.HTTPException`` when the connection fails. The iterator is closed by the caller."""


class _LineSource(Protocol):
    def readline(self) -> bytes: ...

    def close(self) -> None: ...


def _lines(source: _LineSource) -> Iterator[bytes]:
    try:
        while line := source.readline():
            yield line
    finally:
        source.close()


def urllib_post_stream(
    url: str, headers: Mapping[str, str], body: bytes, timeout: float
) -> tuple[int, Iterator[bytes]]:
    """The production transport: standard-library ``urllib``, https only, one line per chunk."""
    if urllib.parse.urlsplit(url).scheme != "https":
        raise ValueError("the Qwen transport only speaks https")
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")  # noqa: S310 - https checked above
    try:
        response = urllib.request.urlopen(request, timeout=timeout)  # noqa: S310 - https checked above
    except urllib.error.HTTPError as exc:
        return exc.code, _lines(exc)
    status: int = response.status
    return status, _lines(response)


def _sleep(seconds: float) -> None:
    """Backoff wait. A module function so tests can replace it and never sleep."""
    time.sleep(seconds)


# ================================================================================================
# The model
# ================================================================================================


class QwenExchange(Model):
    """One request sent to Qwen and what came back. :attr:`QwenChatModel.last_exchanges`."""

    attempt: int = Field(ge=1)
    sent_at: UtcDatetime
    status: int | None
    """HTTP status; ``None`` when no response arrived."""
    request: BlobRef | None
    response: BlobRef | None
    """The raw body received (partial, if the attempt failed mid-stream)."""
    usage: LlmUsage
    latency_ms: int = Field(ge=0)
    error: str | None = None
    """Why this attempt did not produce the answer; ``None`` for the one that did."""

    @model_validator(mode="after")
    def _answered_or_explained(self) -> "QwenExchange":
        if self.error is None and self.status is None:
            raise ValueError("an exchange without a response must say why")
        return self


@dataclass(frozen=True)
class _Failure:
    kind: Literal["timeout", "transport"]
    message: str
    status: int | None
    retryable: bool


def _elapsed_ms(start: datetime, end: datetime) -> int:
    return max(0, int((end - start) / timedelta(milliseconds=1)))


def _close(chunks: Iterator[bytes] | None) -> None:
    close = getattr(chunks, "close", None)
    if callable(close):
        try:
            close()
        except (OSError, http.client.HTTPException):
            # The body has been read or abandoned; a failing close changes neither.
            return


class QwenChatModel:
    """:class:`~sentiment_agent.types.ChatModel` backed by the live Qwen endpoint.

    Per call: validate the arguments, then for each attempt check the daily budget, send, read the
    whole body under the call deadline, store both sides as blobs, and charge the budget with what
    the endpoint reported (or an unreported call when it reported nothing). 408, 429, 5xx, dropped
    connections and cut or corrupted streams are retried ``max_transport_retries`` times with
    exponential backoff; every other non-2xx status fails at once; a timeout ends the call.

    ``Completion.latency_ms`` is the whole call as the decision cycle waited for it, retries and
    backoff included; each attempt's own latency is in :attr:`last_exchanges`.
    """

    def __init__(
        self,
        *,
        credentials: QwenCredentials,
        budget: DailyTokenBudget,
        clock: Clock,
        blobs: BlobStore | None = None,
        timeout_s: float = 600.0,
        max_transport_retries: int = 3,
        http: HttpPostStream = urllib_post_stream,
    ) -> None:
        if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be a positive number of seconds")
        if isinstance(max_transport_retries, bool) or max_transport_retries < 0:
            raise ValueError("max_transport_retries must be a non-negative integer")
        self._credentials = credentials
        self._budget = budget
        self._clock = clock
        self._blobs = blobs
        self._timeout_s = float(timeout_s)
        self._max_retries = max_transport_retries
        self._http = http
        self._url = credentials.base_url + CHAT_PATH
        self._last: tuple[QwenExchange, ...] = ()

    def __repr__(self) -> str:
        return (
            f"QwenChatModel(model={self.model_name!r}, url={self._url!r}, "
            f"timeout_s={self._timeout_s:g}, max_transport_retries={self._max_retries})"
        )

    @property
    def model_name(self) -> str:
        return self._credentials.model

    @property
    def last_exchanges(self) -> tuple[QwenExchange, ...]:
        """Every attempt of the most recent :meth:`complete`, successful or not, in order."""
        return self._last

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
        request = ChatRequest(
            messages=tuple(messages),
            json_mode=json_mode,
            max_tokens=max_tokens,
            thinking=thinking,
            temperature=temperature,
            seed=seed,
        )
        body = encode_payload(request.payload(self.model_name))
        projected = projected_tokens(request.messages, request.max_tokens)
        started = self._clock.now()
        deadline = started + timedelta(seconds=self._timeout_s)
        exchanges: list[QwenExchange] = []
        self._last = ()
        try:
            attempt = 0
            while True:
                attempt += 1
                if self._clock.now() >= deadline:
                    # Only reachable on a retry: nothing is sent, so nothing is charged.
                    raise QwenTimeout(self._timeout_message(attempt), attempts=attempt - 1)
                self._budget.check(projected)
                exchange, result = self._attempt(
                    attempt, body, streamed=request.streamed, started=started, deadline=deadline
                )
                exchanges.append(exchange)
                self._budget.record(exchange.usage)
                if isinstance(result, Completion):
                    return result
                self._give_up_or_wait(result, attempt, deadline)
        finally:
            self._last = tuple(exchanges)

    # --- internals ---------------------------------------------------------------------------

    def _give_up_or_wait(self, failure: _Failure, attempt: int, deadline: datetime) -> None:
        if failure.kind == "timeout":
            raise QwenTimeout(failure.message, status=failure.status, attempts=attempt)
        if not failure.retryable:
            raise QwenTransportError(failure.message, status=failure.status, attempts=attempt)
        if attempt > self._max_retries:
            raise QwenTransportError(
                f"Qwen failed after {attempt} attempts; last: {failure.message}",
                status=failure.status,
                attempts=attempt,
            )
        delay = BACKOFF_BASE_S * 2 ** (attempt - 1)
        remaining = (deadline - self._clock.now()).total_seconds()
        if delay >= remaining:
            raise QwenTransportError(
                f"Qwen failed on attempt {attempt} and {max(remaining, 0.0):.1f}s of the "
                f"{self._timeout_s:g}s call deadline is too little for the {delay:g}s backoff; "
                f"last: {failure.message}",
                status=failure.status,
                attempts=attempt,
            )
        _sleep(delay)

    def _store(self, data: bytes, media_type: str) -> BlobRef | None:
        if self._blobs is None:
            return None
        return self._blobs.put(self._credentials.redact_bytes(data), media_type)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": self._credentials.authorization_value(),
            "Content-Type": "application/json",
        }

    def _attempt(
        self, attempt: int, body: bytes, *, streamed: bool, started: datetime, deadline: datetime
    ) -> tuple[QwenExchange, Completion | _Failure]:
        request_ref = self._store(body, REQUEST_MEDIA_TYPE)
        sent_at = self._clock.now()
        remaining = (deadline - sent_at).total_seconds()
        status: int | None = None
        raw = bytearray()
        failure: _Failure | None = None
        assembler = _StreamAssembler() if streamed else None
        chunks: Iterator[bytes] | None = None
        try:
            # The loop in `complete` has already checked the deadline; the floor only keeps a
            # clock that ticked past it in between from asking the socket for a zero timeout,
            # which would mean non-blocking rather than "no time".
            status, chunks = self._http(self._url, self._headers(), body, max(remaining, 0.001))
            success = 200 <= status < 300
            for chunk in chunks:
                raw += chunk
                if len(raw) > MAX_RESPONSE_BYTES:
                    failure = self._failure(
                        "transport",
                        f"the response exceeded {MAX_RESPONSE_BYTES} bytes",
                        status,
                        retryable=True,
                    )
                    break
                if assembler is not None and success:
                    assembler.feed(chunk)
                    if assembler.done:
                        break
                if self._clock.now() >= deadline:
                    failure = self._timeout_failure(attempt, status)
                    break
        except TimeoutError:
            failure = self._timeout_failure(attempt, status)
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                failure = self._timeout_failure(attempt, status)
            else:
                failure = self._transport_failure(exc, status)
        except (OSError, http.client.HTTPException) as exc:
            failure = self._transport_failure(exc, status)
        except _BadResponseError as exc:
            failure = self._failure("transport", exc.message, status, retryable=True)
        finally:
            _close(chunks)

        finished = self._clock.now()
        response_ref = None
        if status is not None:
            if not 200 <= status < 300:
                media = ERROR_RESPONSE_MEDIA_TYPE
            else:
                media = STREAM_RESPONSE_MEDIA_TYPE if streamed else JSON_RESPONSE_MEDIA_TYPE
            response_ref = self._store(bytes(raw), media)

        result: Completion | _Failure
        usage = assembler.usage if assembler is not None else UNREPORTED_USAGE
        if failure is not None:
            result = failure
        elif status is not None and not 200 <= status < 300:
            retryable = status in RETRYABLE_STATUSES or status >= 500
            result = self._failure(
                "transport",
                f"HTTP {status}: {self._detail(bytes(raw))}",
                status,
                retryable=retryable,
            )
            usage = UNREPORTED_USAGE
        else:
            try:
                latency = _elapsed_ms(started, finished)
                result = (
                    assembler.result(latency)
                    if assembler is not None
                    else _parse_body(bytes(raw), latency)
                )
                usage = result.usage
            except _BadResponseError as exc:
                result = self._failure("transport", exc.message, status, retryable=True)
                usage = exc.usage

        exchange = QwenExchange(
            attempt=attempt,
            sent_at=sent_at,
            status=status,
            request=request_ref,
            response=response_ref,
            usage=usage,
            latency_ms=_elapsed_ms(sent_at, finished),
            error=None if isinstance(result, Completion) else result.message,
        )
        return exchange, result

    def _timeout_message(self, attempt: int) -> str:
        return (
            f"Qwen gave no complete answer within the {self._timeout_s:g}s call deadline "
            f"(attempt {attempt})"
        )

    def _timeout_failure(self, attempt: int, status: int | None) -> _Failure:
        return self._failure("timeout", self._timeout_message(attempt), status, retryable=False)

    def _transport_failure(self, exc: BaseException, status: int | None) -> _Failure:
        """A connection-level failure. Retried, except a TLS certificate that does not verify:
        that will not fix itself in a few seconds, and it is the one failure that can mean the
        request (and its key) was about to reach someone other than the gateway."""
        reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
        untrusted = isinstance(reason, ssl.SSLCertVerificationError)
        return self._failure(
            "transport",
            f"transport failure: {type(exc).__name__}: {exc}",
            status,
            retryable=not untrusted,
        )

    def _failure(
        self,
        kind: Literal["timeout", "transport"],
        message: str,
        status: int | None,
        *,
        retryable: bool,
    ) -> _Failure:
        """Every failure message passes here: the key is scrubbed from it *before* it is cut, so
        neither the whole key nor a fragment of it can reach an exception, an exchange or a log.
        Server text (an error page, an in-stream error event) is quoted only through this."""
        text = " ".join(self._credentials.redact(message).split())
        return _Failure(kind, text[: 2 * ERROR_DETAIL_CHARS], status, retryable)

    def _detail(self, raw: bytes) -> str:
        """A short, key-free excerpt of an error body. Redacted before it is cut, so a cut can
        never leave a fragment of the key behind."""
        text = " ".join(self._credentials.redact(raw.decode("utf-8", errors="replace")).split())
        return text[:ERROR_DETAIL_CHARS] or "(empty body)"


__all__ = [
    "BACKOFF_BASE_S",
    "CHAT_PATH",
    "COMPLETION_MEDIA_TYPE",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "ENV_API_KEY",
    "ENV_BASE_URL",
    "ENV_MODEL",
    "ERROR_RESPONSE_MEDIA_TYPE",
    "JSON_RESPONSE_MEDIA_TYPE",
    "MAX_RESPONSE_BYTES",
    "QWEN_ENV_FILE",
    "REQUEST_MEDIA_TYPE",
    "RETRYABLE_STATUSES",
    "STREAM_RESPONSE_MEDIA_TYPE",
    "UNREPORTED_USAGE",
    "ChatRequest",
    "HttpPostStream",
    "QwenChatModel",
    "QwenCredentials",
    "QwenError",
    "QwenExchange",
    "QwenTimeout",
    "QwenTransportError",
    "encode_payload",
    "load_qwen_env",
    "parse_response",
    "prompt_hash",
    "thinking_fields",
    "urllib_post_stream",
    "usage_from_block",
]
