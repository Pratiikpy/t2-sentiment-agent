"""The decision contract: from a completion to a validated decision.

:func:`obtain_decision` makes at most ``policy.decision.max_attempts`` calls. Each answer is taken
through three gates, and a failure at any of them is either retried with the specific complaint fed
back to the model, or ends the cycle with an outcome the kernel acts on (DESIGN.md §9.3, §9.5):

1. **Truncation**, read from ``finish_reason`` on the wire and never inferred from a parse error.
   A cut-off answer is retried from the original prompt with a larger completion cap, up to the
   policy's ceiling. It is never repaired: a plausible invented ending to a trading thesis reads as
   reasoning the model never did.
2. **JSON and schema.** The object is parsed strictly against the contract model: unknown fields,
   missing fields, strings where numbers belong and out-of-range values are all refused, and each
   refusal is named in the complaint.
3. **The book and the universe** (:func:`contract_complaints`): every held symbol addressed, no
   symbol outside the universe or on the excluded list, every horizon at least the mandate's, the
   stance consistent with the targets, and a declared invalidation only on a position that is held
   and never used to add to its side.

A failure of the model service itself (the daily budget, a timeout, a transport error) ends the
cycle at once with that outcome; the client below has already retried what can be retried. Nothing
else is caught: a replay that cannot find its recording, or a defect in this code, raises.

Ported in part from ARGUS ``argus/src/argus/llm/qwen.py`` (``complete_json``, ``_strip_fences``,
``extract_json_object``), MIT, Copyright (c) 2026 Pratiikpy, the same author as this project, at
commit ``3dec6baf9dfa7be37c7b452e26a9b139df252f75``, sha256
``dcc4ba1077a63ca2dca4ee7ece18e566b90b29be2a3a5bfd1c6a987c2678daa9`` (licence and pins:
``third_party/argus/PROVENANCE.md``); the ARGUS tree is not imported. Kept: truncation read from
``finish_reason`` (or an empty answer after reasoning), a retry from the original prompt at three
times the cap, JSON extraction by string-aware brace matching as a fallback only, and complaint-fed
retries carrying only the last answer. Changed: the schema is the typed contract model in strict
mode instead of a list of required keys, every complaint is collected rather than the first one, and
every attempt is logged as a blob.
"""

import json
import time
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Final

from pydantic import BaseModel, ValidationError

from sentiment_agent.decision.prompt import PROMPT_VERSION, held_symbols
from sentiment_agent.hashing import canonical_json, sha256_hex
from sentiment_agent.llm.budget import BudgetExhausted
from sentiment_agent.llm.client import (
    ChatRequest,
    QwenError,
    QwenTimeout,
    encode_payload,
    prompt_hash,
)
from sentiment_agent.llm.fakes import RecordedCall
from sentiment_agent.types import (
    BlobRef,
    BlobStore,
    BookState,
    ChatMessage,
    ChatModel,
    Completion,
    LlmCallRecord,
    LlmDecision,
    LlmOutcome,
    LlmUsage,
    PerceptionSnapshot,
    Policy,
    Stance,
    Thinking,
)

INITIAL_COMPLETION_TOKENS: Final[Mapping[Thinking, int]] = MappingProxyType(
    {Thinking.OFF: 2048, Thinking.LOW: 4096, Thinking.FULL: 8192}
)
"""First completion cap per reasoning tier, clamped to ``policy.decision.max_completion_tokens``.

Basis: ARGUS measured 116 (OFF), 828 (LOW) and 4,264 (FULL) completion tokens, reasoning included,
on a one-instrument decision (``llm/qwen.py``). A portfolio answer carries a thesis per target, so
each tier starts at about twice its measured need, and FULL starts at the ceiling: a truncated FULL
answer wastes a whole reasoning pass, which costs more than a generous cap."""

TRUNCATION_GROWTH: Final = 3
"""Cap multiplier after a truncated answer (ARGUS ``complete_json``)."""

DECISION_SEED: Final = 20260924
"""Sent on every call, so a provider that honours seeds answers an identical prompt identically."""

ATTEMPT_MEDIA_TYPE: Final = "application/vnd.t2sa.decision-attempt+json"
"""One attempt of a decision call: the request as sent, the completion or the error, the verdict,
and the client's wire exchanges when it keeps them."""

REQUEST_MEDIA_TYPE: Final = "application/json"
"""The first request, byte for byte the body the live client sends (``encode_payload``)."""

ERROR_CHARS: Final = 2000


class DecisionInvalid(ValueError):  # noqa: N818 - the name is the module contract (DESIGN M7)
    """A complete answer that breaks the contract. ``complaints`` are fed back to the model."""

    def __init__(self, complaints: Sequence[str]) -> None:
        if not complaints:
            raise ValueError("DecisionInvalid needs at least one complaint")
        self.complaints: tuple[str, ...] = tuple(complaints)
        super().__init__("; ".join(self.complaints))


# ================================================================================================
# JSON
# ================================================================================================


def _strip_fences(text: str) -> str:
    """Remove markdown fences a model sometimes wraps JSON in despite JSON mode (ARGUS)."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1] if "\n" in stripped else stripped
        stripped = stripped.rsplit("```", 1)[0]
    return stripped.strip()


def extract_json_object(text: str) -> str:
    """The outermost balanced ``{...}`` in ``text``, after any fences are removed (ARGUS).

    For a schema wrapped in prose ("Here is my decision: {...}"). Brace counting rather than a
    regex, string-aware so a ``}`` inside a thesis does not end the object, with escapes honoured.
    Returns the fence-stripped text unchanged when no balanced object is found, so the caller still
    reports the real parse error.
    """
    stripped = _strip_fences(text)
    start = stripped.find("{")
    if start == -1:
        return stripped
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : index + 1]
    return stripped


def _load_object(content: str) -> tuple[str, dict[str, Any]]:
    """The JSON text of the one object in ``content``, and the object. Valid JSON is tried first;
    extraction is a fallback for an object wrapped in prose, never a way into a larger structure."""
    text = _strip_fences(content)
    if not text:
        raise DecisionInvalid(["the answer was empty: return the JSON object"])
    try:
        parsed: object = json.loads(text)
    except json.JSONDecodeError as outer:
        extracted = extract_json_object(text)
        if extracted == text:
            raise DecisionInvalid(
                [f"the answer is not valid JSON ({outer.msg} at line {outer.lineno})"]
            ) from None
        try:
            parsed = json.loads(extracted)
        except json.JSONDecodeError as inner:
            raise DecisionInvalid(
                [f"the answer is not valid JSON even around its object ({inner.msg})"]
            ) from None
        text = extracted
    if not isinstance(parsed, dict):
        raise DecisionInvalid(
            [f"the answer must be one JSON object, not a JSON {type(parsed).__name__}"]
        )
    return text, parsed


def _where(loc: Sequence[int | str], parsed: Mapping[str, Any]) -> str:
    """``targets[1] (NVDAUSDT).horizon_hours`` for a pydantic error location."""
    parts: list[str] = []
    for index, part in enumerate(loc):
        if isinstance(part, int):
            name = f"{parts.pop() if parts else ''}[{part}]"
            if index == 1 and loc[0] == "targets":
                targets = parsed.get("targets")
                if isinstance(targets, list) and part < len(targets):
                    entry = targets[part]
                    if isinstance(entry, dict) and isinstance(entry.get("symbol"), str):
                        name += f" ({entry['symbol']})"
            parts.append(name)
        else:
            parts.append(part)
    return ".".join(parts) or "the answer"


def _schema_complaints(error: ValidationError, parsed: Mapping[str, Any]) -> list[str]:
    complaints: list[str] = []
    for detail in error.errors(include_url=False):
        where = _where(tuple(detail["loc"]), parsed)
        kind = detail["type"]
        message = str(detail["msg"]).removeprefix("Value error, ")
        if kind == "extra_forbidden":
            complaints.append(f"{where} is not a field of the answer; remove it")
        elif kind == "missing":
            complaints.append(f"{where} is required and missing")
        else:
            complaints.append(f"{where}: {message}")
    return complaints


# ================================================================================================
# The contract against the book and the universe
# ================================================================================================


def contract_complaints(
    decision: LlmDecision, *, book: BookState, snapshot: PerceptionSnapshot, policy: Policy
) -> list[str]:
    """Everything wrong with a schema-valid decision, given the book and the policy."""
    complaints: list[str] = []
    universe = set(policy.symbols)
    excluded = set(policy.excluded)
    for target in decision.targets:
        if target.symbol in excluded:
            complaints.append(
                f"{target.symbol} is excluded by the policy (its Demo prices are unreliable); "
                "remove it"
            )
        elif target.symbol not in universe:
            complaints.append(
                f"{target.symbol!r} is not in the universe; use one of {', '.join(policy.symbols)}"
            )
        minimum = policy.decision.min_horizon_hours
        if target.horizon_hours < minimum:
            complaints.append(
                f"{target.symbol}: horizon_hours is {target.horizon_hours}; the mandate requires "
                f"at least {minimum}"
            )

    held = held_symbols(book, policy)
    unaddressed = [s for s in held if s not in decision.symbols]
    if unaddressed:
        complaints.append(
            f"you hold {', '.join(unaddressed)} and did not address "
            f"{'it' if len(unaddressed) == 1 else 'them'}: every held symbol needs a target "
            "(zero closes it)"
        )

    if not held and decision.stance is Stance.HOLD:
        complaints.append(
            "stance 'hold' with no open positions keeps an empty book without written reasons: "
            "answer 'act' with at least one non-zero target, or 'flat_with_reasons'"
        )
    if not held and decision.stance is Stance.ACT and all(t.target == 0 for t in decision.targets):
        complaints.append(
            "stance 'act' with no open positions must open something: give at least one non-zero "
            "target, or answer 'flat_with_reasons' with your reasons"
        )
    if decision.stance is Stance.HOLD:
        for target in decision.targets:
            position = book.positions.get(target.symbol)
            if position is None or position.is_flat:
                if target.target != 0:
                    complaints.append(
                        f"stance 'hold' opens nothing new: {target.symbol} is not held, so its "
                        "target must be zero, or use stance 'act'"
                    )
            elif target.target == 0 or (target.target > 0) != (position.qty > 0):
                side = "long" if position.qty > 0 else "short"
                complaints.append(
                    f"stance 'hold' keeps {target.symbol} {side}: give it a target on the same "
                    "side (its position_target_equivalent), or use stance 'act' to change it"
                )

    for target in decision.targets:
        if not target.invalidation_triggered:
            continue
        position = book.positions.get(target.symbol)
        if position is None or position.is_flat:
            complaints.append(
                f"{target.symbol}: invalidation_triggered applies only to a position you hold, "
                "and you hold none; set it to false"
            )
            continue
        same_side = target.target != 0 and (target.target > 0) == (position.qty > 0)
        current = abs(book.weight(target.symbol)) if target.symbol in book.marks else None
        proposed = abs(target.target) * policy.mandate.per_name_max
        if same_side and current is not None and proposed > current + 1e-9:
            complaints.append(
                f"{target.symbol}: a fired invalidation cannot justify adding to the same side; "
                "cut, close or flip the position, or set invalidation_triggered to false"
            )
    return complaints


def parse_decision(
    content: str, *, book: BookState, snapshot: PerceptionSnapshot, policy: Policy
) -> LlmDecision:
    """A validated decision, or :class:`DecisionInvalid` carrying every complaint found."""
    text, parsed = _load_object(content)
    try:
        decision = LlmDecision.model_validate_json(text, strict=True)
    except ValidationError as error:
        raise DecisionInvalid(_schema_complaints(error, parsed)) from None
    complaints = contract_complaints(decision, book=book, snapshot=snapshot, policy=policy)
    if complaints:
        raise DecisionInvalid(complaints)
    return decision


def complaint_message(complaints: Sequence[str]) -> str:
    """The user turn that returns an invalid answer to the model."""
    lines = ["Your previous answer could not be accepted:"]
    lines.extend(f"- {complaint}" for complaint in complaints)
    lines.append("Return the complete corrected JSON object, and nothing else.")
    return "\n".join(lines)


# ================================================================================================
# Obtaining a decision
# ================================================================================================


def classify_failure(error: BaseException) -> LlmOutcome | None:
    """The outcome for a failure of the model service, or ``None`` for anything else.

    Budget exhaustion, a timeout and a transport failure are the model being unavailable, which the
    kernel answers by flattening. Anything else (a replay that finds no recording, an invalid
    argument, a defect) is not an outage and must not be recorded as one.
    """
    if isinstance(error, BudgetExhausted):
        return LlmOutcome.BUDGET_EXHAUSTED
    if isinstance(error, (QwenTimeout, TimeoutError)):
        return LlmOutcome.TIMEOUT
    if isinstance(error, (QwenError, OSError)):
        return LlmOutcome.TRANSPORT_ERROR
    return None


def is_truncated(completion: Completion) -> bool:
    """Cut off at the cap, or reasoning consumed the cap before any answer (ARGUS)."""
    return completion.finish_reason == "length" or (
        not completion.content.strip() and bool(completion.reasoning.strip())
    )


def _sum_usage(usages: Sequence[LlmUsage], *, unmeasured: bool) -> LlmUsage:
    return LlmUsage(
        prompt_tokens=sum(u.prompt_tokens for u in usages),
        completion_tokens=sum(u.completion_tokens for u in usages),
        reasoning_tokens=sum(u.reasoning_tokens for u in usages),
        total_tokens=sum(u.total_tokens for u in usages),
        cached_tokens=sum(u.cached_tokens for u in usages),
        reported=all(u.reported for u in usages) and not unmeasured,
    )


def _wire(model: ChatModel) -> tuple[list[dict[str, Any]], list[BlobRef]]:
    """The live client's own record of the last call, when it keeps one (``last_exchanges``)."""
    exchanges = getattr(model, "last_exchanges", ())
    dumps: list[dict[str, Any]] = []
    responses: list[BlobRef] = []
    for exchange in exchanges if isinstance(exchanges, tuple) else ():
        if isinstance(exchange, BaseModel):
            dumps.append(exchange.model_dump(mode="json"))
            response = getattr(exchange, "response", None)
            if isinstance(response, BlobRef):
                responses.append(response)
    return dumps, responses


def _truncate(text: str) -> str:
    return text if len(text) <= ERROR_CHARS else text[: ERROR_CHARS - 1] + "…"


def obtain_decision(
    model: ChatModel,
    messages: Sequence[ChatMessage],
    *,
    thinking: Thinking,
    policy: Policy,
    book: BookState,
    snapshot: PerceptionSnapshot,
    blobs: BlobStore,
) -> tuple[LlmDecision | None, LlmCallRecord]:
    """One decision call with complaint-fed retries. Never raises for a model-service failure."""
    initial = tuple(messages)
    rule = policy.decision
    ceiling = rule.max_completion_tokens
    cap = min(INITIAL_COMPLETION_TOKENS[thinking], ceiling)
    convo: tuple[ChatMessage, ...] = initial

    def request_for(conversation: Sequence[ChatMessage], max_tokens: int) -> ChatRequest:
        return ChatRequest(
            messages=tuple(conversation),
            json_mode=True,
            max_tokens=max_tokens,
            thinking=thinking,
            temperature=rule.temperature,
            seed=DECISION_SEED,
        )

    first = request_for(initial, cap)
    request_blob = blobs.put(encode_payload(first.payload(model.model_name)), REQUEST_MEDIA_TYPE)
    response_blobs: list[BlobRef] = []
    usages: list[LlmUsage] = []
    failures: list[str] = []
    latency_ms = 0
    unmeasured = False
    outcome = LlmOutcome.INVALID_RESPONSE
    decision: LlmDecision | None = None
    attempts = 0

    for attempt in range(1, rule.max_attempts + 1):
        attempts = attempt
        request = request_for(convo, cap)
        record: dict[str, Any] = {
            "attempt": attempt,
            "request": request.payload(model.model_name),
            "prompt_hash": request.prompt_hash,
            "completion": None,
            "complaints": [],
            "error": None,
        }
        started = time.perf_counter()
        try:
            completion = model.complete(
                request.messages,
                json_mode=request.json_mode,
                max_tokens=request.max_tokens,
                thinking=request.thinking,
                temperature=request.temperature,
                seed=request.seed,
            )
        except Exception as error:
            failed = classify_failure(error)
            if failed is None:
                raise
            latency_ms += int((time.perf_counter() - started) * 1000)
            outcome = failed
            unmeasured = unmeasured or failed is not LlmOutcome.BUDGET_EXHAUSTED
            failures.append(f"attempt {attempt}: {type(error).__name__}: {error}")
            record.update(
                verdict="failed",
                error={"type": type(error).__name__, "message": _truncate(str(error))},
            )
            _store_attempt(record, model, blobs, response_blobs)
            break

        usages.append(completion.usage)
        latency_ms += completion.latency_ms
        record["completion"] = completion.model_dump(mode="json")

        if is_truncated(completion):
            outcome = LlmOutcome.TRUNCATED
            reason = (
                "the answer was cut off at the cap"
                if completion.finish_reason == "length"
                else "reasoning used the whole cap before any answer"
            )
            record["verdict"] = "truncated"
            if cap >= ceiling:
                failures.append(
                    f"attempt {attempt}: {reason} at the {ceiling}-token ceiling; a truncated "
                    "decision is not repaired"
                )
                _store_attempt(record, model, blobs, response_blobs)
                break
            grown = min(cap * TRUNCATION_GROWTH, ceiling)
            failures.append(f"attempt {attempt}: {reason} ({cap} tokens); retrying at {grown}")
            _store_attempt(record, model, blobs, response_blobs)
            cap = grown
            convo = initial  # a truncated answer is never fed back: it would only grow the prompt
            continue

        try:
            decision = parse_decision(
                completion.content, book=book, snapshot=snapshot, policy=policy
            )
        except DecisionInvalid as invalid:
            outcome = LlmOutcome.INVALID_RESPONSE
            record.update(verdict="invalid", complaints=list(invalid.complaints))
            failures.append(f"attempt {attempt}: invalid: {invalid}")
            _store_attempt(record, model, blobs, response_blobs)
            convo = (
                *initial,
                ChatMessage(role="assistant", content=completion.content),
                ChatMessage(role="user", content=complaint_message(invalid.complaints)),
            )
            continue

        outcome = LlmOutcome.DECIDED
        record["verdict"] = "accepted"
        _store_attempt(record, model, blobs, response_blobs)
        break

    call = LlmCallRecord(
        model=model.model_name,
        thinking=thinking,
        prompt_version=PROMPT_VERSION,
        prompt_hash=prompt_hash(initial),
        request_blob=request_blob,
        response_blobs=tuple(response_blobs),
        attempts=attempts,
        usage=_sum_usage(usages, unmeasured=unmeasured),
        latency_ms=max(0, latency_ms),
        outcome=outcome,
        error=None if outcome is LlmOutcome.DECIDED else _truncate(" | ".join(failures)),
    )
    return (decision if outcome is LlmOutcome.DECIDED else None), call


def _store_attempt(
    record: dict[str, Any], model: ChatModel, blobs: BlobStore, response_blobs: list[BlobRef]
) -> None:
    wire, raw_responses = _wire(model)
    record["wire"] = wire
    response_blobs.append(blobs.put(canonical_json(record), ATTEMPT_MEDIA_TYPE))
    response_blobs.extend(raw_responses)


# ================================================================================================
# Replay
# ================================================================================================


def replay_calls(call: LlmCallRecord, blobs: BlobStore) -> tuple[RecordedCall, ...]:
    """Every answered attempt of a logged call, ready for
    :class:`~sentiment_agent.llm.fakes.RecordedChatModel`, so ``t2sa replay`` re-runs the cycle
    from what the model actually said. Each blob is checked against its reference first."""
    recorded: list[RecordedCall] = []
    for ref in call.response_blobs:
        if ref.media_type != ATTEMPT_MEDIA_TYPE:
            continue
        data = blobs.get(ref.sha256)
        if sha256_hex(data) != ref.sha256 or len(data) != ref.size:
            raise ValueError(f"the attempt blob {ref.sha256[:12]} does not match its reference")
        attempt = json.loads(data.decode("utf-8"))
        if attempt.get("completion") is None:
            continue
        request = ChatRequest.from_payload(attempt["request"])
        completion = Completion.model_validate(attempt["completion"])
        recorded.append(RecordedCall.of(request.messages, completion, request))
    return tuple(recorded)


__all__ = [
    "ATTEMPT_MEDIA_TYPE",
    "DECISION_SEED",
    "INITIAL_COMPLETION_TOKENS",
    "REQUEST_MEDIA_TYPE",
    "TRUNCATION_GROWTH",
    "DecisionInvalid",
    "classify_failure",
    "complaint_message",
    "contract_complaints",
    "extract_json_object",
    "is_truncated",
    "obtain_decision",
    "parse_decision",
    "replay_calls",
]
