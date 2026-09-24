"""The decision: one bounded model call with complaint-fed retries, and the contract it must meet.

A port of the primary's ``sentiment_agent.decision.contract`` (ARGUS ``llm/qwen.py`` lineage, MIT,
same author) onto ``getagent.llm``, the only model access the Playbook sandbox allows
(``references/sdk/llm/catalog.md``: ``llm.chat(messages, *, system, max_tokens, temperature)``, one
runner-managed model, fixed call, prompt, output and timeout budgets, typed ``RuntimeError``
subclasses). The model is the runner's, not the primary's ``qwen3.8-max``; the record names both.

Each answer passes the primary's three gates, and a failure is retried with the specific complaint
fed back, at most ``DECISION_MAX_ATTEMPTS`` times:

1. **Truncation**, from ``finish_reason == "length"`` (or an empty answer), retried from the
   original prompt with three times the completion cap up to the policy ceiling; never repaired.
2. **JSON and schema**, strictly: unknown and missing fields, wrong types and out-of-range values
   are each named in the complaint. Integers are not accepted as booleans, nor floats as integers.
3. **The book and the universe**: every held symbol addressed, nothing outside this subscription's
   symbols or on the excluded list, every horizon at least the mandate's, the stance consistent
   with the targets, and a declared invalidation only on a held position and never used to add to
   its side.

A model-service failure ends the cycle with an outcome the kernel answers by flattening the book
(DESIGN.md §9.5). Two replica-specific outcomes: ``deferred`` when the run had no time left to
call at all (the triggers carry to the next run; nothing is flattened), and a single compact retry
when the runner refuses the full prompt as malformed or too large (``LLMInputError``).
"""

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from . import policy_v1 as policy

STANCES = ("act", "hold", "flat_with_reasons")
TOP_REQUIRED = ("stance", "targets", "rejected_alternatives", "mandate_response", "summary")
TOP_OPTIONAL = ("flat_reasons",)
TARGET_REQUIRED = (
    "symbol",
    "target",
    "thesis",
    "invalidation",
    "horizon_hours",
    "crowd_belief",
    "our_view",
    "confidence",
)
TARGET_OPTIONAL = ("evidence", "invalidation_triggered", "invalidation_evidence")
NON_EMPTY_TARGET_TEXT = ("thesis", "invalidation", "crowd_belief", "our_view")

DECIDED = "decided"
INVALID = "invalid_response"
TRUNCATED = "truncated"
TIMEOUT = "timeout"
TRANSPORT = "transport_error"
BUDGET = "budget_exhausted"
UNAVAILABLE = "unavailable"
DEFERRED = "deferred"

ERROR_CHARS = 2000


class DecisionInvalid(ValueError):  # noqa: N818 - the primary's name for the same contract
    def __init__(self, complaints: Sequence[str]) -> None:
        if not complaints:
            raise ValueError("DecisionInvalid needs at least one complaint")
        self.complaints = tuple(complaints)
        super().__init__("; ".join(self.complaints))


# ------------------------------------------------------------------------------------------------
# JSON
# ------------------------------------------------------------------------------------------------


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1] if "\n" in stripped else stripped
        stripped = stripped.rsplit("```", 1)[0]
    return stripped.strip()


def extract_json_object(text: str) -> str:
    """The outermost balanced ``{...}``, string-aware (the primary's fallback for prose-wrapped
    answers). Returns the fence-stripped text when none is found."""
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


def load_object(content: str) -> dict[str, object]:
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
    if not isinstance(parsed, dict):
        raise DecisionInvalid(
            [f"the answer must be one JSON object, not a JSON {type(parsed).__name__}"]
        )
    return parsed


# ------------------------------------------------------------------------------------------------
# Schema (the primary's LlmDecision / TargetProposal / RejectedAlternative, strict)
# ------------------------------------------------------------------------------------------------


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def as_number(value: object) -> float:
    """``value`` as a float when it is a JSON number, NaN otherwise (never raised on)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float("nan")
    return float(value)


def as_list(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def targets_of(decision: Mapping[str, object]) -> list[dict[str, object]]:
    """The target objects of a (validated) decision."""
    return [t for t in as_list(decision.get("targets")) if isinstance(t, dict)]


def _where(index: int, entry: object) -> str:
    symbol = entry.get("symbol") if isinstance(entry, dict) else None
    return f"targets[{index}]" + (f" ({symbol})" if isinstance(symbol, str) else "")


def schema_complaints(parsed: Mapping[str, object]) -> list[str]:
    complaints: list[str] = []
    for key in parsed:
        if key not in TOP_REQUIRED and key not in TOP_OPTIONAL:
            complaints.append(f"{key} is not a field of the answer; remove it")
    for key in TOP_REQUIRED:
        if key not in parsed:
            complaints.append(f"{key} is required and missing")
    stance = parsed.get("stance")
    if "stance" in parsed and stance not in STANCES:
        complaints.append(f"stance: must be one of {', '.join(STANCES)}")
    for key in ("mandate_response", "summary"):
        if key in parsed and (not isinstance(parsed[key], str) or not parsed[key]):
            complaints.append(f"{key}: must be a non-empty string")
    flat = parsed.get("flat_reasons", [])
    if not isinstance(flat, list) or not all(isinstance(r, str) for r in flat):
        complaints.append("flat_reasons: must be a list of strings")
    alternatives = parsed.get("rejected_alternatives")
    if "rejected_alternatives" in parsed:
        if not isinstance(alternatives, list):
            complaints.append("rejected_alternatives: must be a list")
        else:
            for i, alt in enumerate(alternatives):
                if not isinstance(alt, dict):
                    complaints.append(f"rejected_alternatives[{i}]: must be an object")
                    continue
                for key in alt:
                    if key not in ("action", "reason"):
                        complaints.append(
                            f"rejected_alternatives[{i}].{key} is not a field of the answer; "
                            "remove it"
                        )
                for key in ("action", "reason"):
                    if key not in alt:
                        complaints.append(
                            f"rejected_alternatives[{i}].{key} is required and missing"
                        )
                    elif not isinstance(alt[key], str):
                        complaints.append(f"rejected_alternatives[{i}].{key}: must be a string")
    targets = parsed.get("targets")
    if "targets" in parsed and not isinstance(targets, list):
        complaints.append("targets: must be a list")
        targets = []
    for i, entry in enumerate(targets if isinstance(targets, list) else []):
        where = _where(i, entry)
        if not isinstance(entry, dict):
            complaints.append(f"{where}: must be an object")
            continue
        for key in entry:
            if key not in TARGET_REQUIRED and key not in TARGET_OPTIONAL:
                complaints.append(f"{where}.{key} is not a field of the answer; remove it")
        for key in TARGET_REQUIRED:
            if key not in entry:
                complaints.append(f"{where}.{key} is required and missing")
        if "symbol" in entry and not isinstance(entry["symbol"], str):
            complaints.append(f"{where}.symbol: must be a string")
        for key in NON_EMPTY_TARGET_TEXT:
            if key in entry and (not isinstance(entry[key], str) or not entry[key]):
                complaints.append(f"{where}.{key}: must be a non-empty string")
        target = entry.get("target")
        if "target" in entry and not (_is_number(target) and -1.0 <= as_number(target) <= 1.0):
            complaints.append(f"{where}.target: must be a number from -1 to 1")
        confidence = entry.get("confidence")
        if "confidence" in entry and not (
            _is_number(confidence) and 0.0 <= as_number(confidence) <= 1.0
        ):
            complaints.append(f"{where}.confidence: must be a number from 0 to 1")
        horizon = entry.get("horizon_hours")
        if "horizon_hours" in entry and not (
            isinstance(horizon, int) and not isinstance(horizon, bool) and horizon >= 1
        ):
            complaints.append(f"{where}.horizon_hours: must be an integer of at least 1")
        evidence = entry.get("evidence", [])
        if not isinstance(evidence, list) or not all(isinstance(e, str) for e in evidence):
            complaints.append(f"{where}.evidence: must be a list of strings")
        fired = entry.get("invalidation_triggered", False)
        if not isinstance(fired, bool):
            complaints.append(f"{where}.invalidation_triggered: must be true or false")
        proof = entry.get("invalidation_evidence")
        if proof is not None and not isinstance(proof, str):
            complaints.append(f"{where}.invalidation_evidence: must be a string or null")
        if fired is True and not (isinstance(proof, str) and proof.strip()):
            complaints.append(
                f"{where}: a declared invalidation must state the evidence that it fired"
            )
    if complaints:
        return complaints
    symbols = [str(t["symbol"]) for t in targets] if isinstance(targets, list) else []
    if len(symbols) != len(set(symbols)):
        complaints.append("a symbol appears more than once in targets")
    if stance == "flat_with_reasons":
        if not any(isinstance(r, str) and r.strip() for r in as_list(flat)):
            complaints.append("flat_with_reasons needs at least one written reason")
        if any(as_number(t.get("target")) != 0 for t in targets_of(parsed)):
            complaints.append("flat_with_reasons cannot carry a non-zero target")
    elif not symbols:
        complaints.append(
            f"stance {stance!r} must address at least one symbol; an empty book kept empty is "
            "flat_with_reasons, with the reasons written"
        )
    return complaints


def normalise(parsed: Mapping[str, object]) -> dict[str, object]:
    """The schema-valid answer with every optional field filled, as the primary's model holds it."""
    targets: list[dict[str, object]] = []
    for entry in targets_of(parsed):
        targets.append(
            {
                "symbol": entry["symbol"],
                "target": as_number(entry["target"]),
                "thesis": entry["thesis"],
                "invalidation": entry["invalidation"],
                "horizon_hours": entry["horizon_hours"],
                "crowd_belief": entry["crowd_belief"],
                "our_view": entry["our_view"],
                "confidence": as_number(entry["confidence"]),
                "evidence": [str(e) for e in as_list(entry.get("evidence", []))],
                "invalidation_triggered": bool(entry.get("invalidation_triggered", False)),
                "invalidation_evidence": entry.get("invalidation_evidence"),
            }
        )
    return {
        "stance": parsed["stance"],
        "targets": targets,
        "rejected_alternatives": [
            dict(a) for a in as_list(parsed.get("rejected_alternatives")) if isinstance(a, dict)
        ],
        "mandate_response": parsed["mandate_response"],
        "flat_reasons": [str(r) for r in as_list(parsed.get("flat_reasons"))],
        "summary": parsed["summary"],
    }


# ------------------------------------------------------------------------------------------------
# The contract against the book and the universe (contract_complaints in the primary)
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BookFacts:
    """What the contract needs to know about the book."""

    held: Mapping[str, int]
    """Held symbol -> +1 long / -1 short."""
    weights: Mapping[str, float]
    """Held symbol -> current signed weight (when measurable)."""
    configured: tuple[str, ...]


def contract_complaints(decision: Mapping[str, object], book: BookFacts) -> list[str]:
    complaints: list[str] = []
    targets = targets_of(decision)
    stance = decision["stance"]
    universe = set(book.configured)
    minimum = policy.DECISION_MIN_HORIZON_HOURS
    for target in targets:
        symbol = str(target["symbol"])
        if symbol in policy.EXCLUDED:
            complaints.append(
                f"{symbol} is excluded by the policy (its Demo prices are unreliable); remove it"
            )
        elif symbol not in universe:
            complaints.append(
                f"{symbol!r} is not in the universe; use one of {', '.join(book.configured)}"
            )
        horizon = target["horizon_hours"]
        if isinstance(horizon, int) and horizon < minimum:
            complaints.append(
                f"{symbol}: horizon_hours is {horizon}; the mandate requires at least {minimum}"
            )
    addressed = {str(t["symbol"]) for t in targets}
    held = sorted(book.held)
    unaddressed = [s for s in held if s not in addressed]
    if unaddressed:
        complaints.append(
            f"you hold {', '.join(unaddressed)} and did not address "
            f"{'it' if len(unaddressed) == 1 else 'them'}: every held symbol needs a target "
            "(zero closes it)"
        )
    if not held and stance == "hold":
        complaints.append(
            "stance 'hold' with no open positions keeps an empty book without written reasons: "
            "answer 'act' with at least one non-zero target, or 'flat_with_reasons'"
        )
    if not held and stance == "act" and all(as_number(t["target"]) == 0 for t in targets):
        complaints.append(
            "stance 'act' with no open positions must open something: give at least one non-zero "
            "target, or answer 'flat_with_reasons' with your reasons"
        )
    if stance == "hold":
        for target in targets:
            symbol = str(target["symbol"])
            value = as_number(target["target"])
            side = book.held.get(symbol)
            if side is None:
                if value != 0:
                    complaints.append(
                        f"stance 'hold' opens nothing new: {symbol} is not held, so its "
                        "target must be zero, or use stance 'act'"
                    )
            elif value == 0 or (value > 0) != (side > 0):
                word = "long" if side > 0 else "short"
                complaints.append(
                    f"stance 'hold' keeps {symbol} {word}: give it a target on the same side (its "
                    "position_target_equivalent), or use stance 'act' to change it"
                )
    for target in targets:
        if not target["invalidation_triggered"]:
            continue
        symbol = str(target["symbol"])
        side = book.held.get(symbol)
        if side is None:
            complaints.append(
                f"{symbol}: invalidation_triggered applies only to a position you hold, and you "
                "hold none; set it to false"
            )
            continue
        value = as_number(target["target"])
        same_side = value != 0 and (value > 0) == (side > 0)
        current = book.weights.get(symbol)
        proposed = abs(value) * policy.MANDATE_PER_NAME_MAX
        if same_side and current is not None and proposed > abs(current) + 1e-9:
            complaints.append(
                f"{symbol}: a fired invalidation cannot justify adding to the same side; cut, "
                "close or flip the position, or set invalidation_triggered to false"
            )
    return complaints


def parse_decision(content: str, book: BookFacts) -> dict[str, object]:
    parsed = load_object(content)
    complaints = schema_complaints(parsed)
    if complaints:
        raise DecisionInvalid(complaints)
    decision = normalise(parsed)
    complaints = contract_complaints(decision, book)
    if complaints:
        raise DecisionInvalid(complaints)
    return decision


def complaint_message(complaints: Sequence[str]) -> str:
    lines = ["Your previous answer could not be accepted:"]
    lines.extend(f"- {complaint}" for complaint in complaints)
    lines.append("Return the complete corrected JSON object, and nothing else.")
    return "\n".join(lines)


def proposed_weights(decision: Mapping[str, object], book: BookFacts) -> dict[str, float]:
    """``target * per_name_max`` per addressed symbol; under ``hold`` a held symbol's proposal
    is its current weight exactly (``decision/agent.py proposed_weights``)."""
    weights: dict[str, float] = {}
    for target in targets_of(decision):
        symbol = str(target["symbol"])
        if decision["stance"] == "hold" and symbol in book.held and symbol in book.weights:
            weights[symbol] = book.weights[symbol]
        else:
            weights[symbol] = float(
                f"{as_number(target['target']) * policy.MANDATE_PER_NAME_MAX:.12g}"
            )
    return weights


# ------------------------------------------------------------------------------------------------
# The call
# ------------------------------------------------------------------------------------------------


@dataclass
class CallRecord:
    model: str | None = None
    attempts: int = 0
    outcome: str = INVALID
    usage: dict[str, int] = field(default_factory=dict)
    compact: bool = False
    finish_reasons: list[str] = field(default_factory=list)
    request_ids: list[str] = field(default_factory=list)
    complaints: list[list[str]] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)
    error: str | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "model": self.model,
            "primary_model": policy.DECISION_PRIMARY_MODEL,
            "attempts": self.attempts,
            "outcome": self.outcome,
            "usage": dict(self.usage),
            "compact_prompt": self.compact,
            "finish_reasons": list(self.finish_reasons),
            "request_ids": list(self.request_ids),
            "complaints": [list(c) for c in self.complaints],
            "answers": list(self.answers),
            "error": self.error,
        }


def classify_failure(error: BaseException) -> str:
    """The outcome for a model-service failure, from the typed errors ``getagent.llm`` raises."""
    name = type(error).__name__
    if name == "LLMBudgetExceededError":
        return BUDGET
    if name in ("LLMRuntimeUnavailableError", "LLMConfigurationError"):
        return UNAVAILABLE
    if isinstance(error, TimeoutError) or "Timeout" in name:
        return TIMEOUT
    return TRANSPORT


def _usage(result: object) -> dict[str, int]:
    raw = getattr(result, "usage", None)
    out: dict[str, int] = {}
    if isinstance(raw, Mapping):
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = raw.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                out[key] = value
    return out


def obtain_decision(
    *,
    chat: Callable[..., object],
    available: bool,
    system: str,
    user: str,
    compact_user: Callable[[], tuple[str, str]],
    book: BookFacts,
    time_left: Callable[[], float],
    min_seconds_per_attempt: float = 30.0,
) -> tuple[dict[str, object] | None, CallRecord]:
    """One decision with complaint-fed retries. Never raises for a model-service failure.

    ``chat`` is ``getagent.llm.chat``; ``compact_user`` renders the compact prompt pair on demand;
    ``time_left`` is the seconds the run may still spend on the model.
    """
    record = CallRecord()
    if not available:
        record.outcome = UNAVAILABLE
        record.error = "getagent.llm is not available in this run (runtime_profile not injected)"
        return None, record
    if time_left() < min_seconds_per_attempt:
        record.outcome = DEFERRED
        record.error = (
            "no time left in this run for a model call; the triggers carry to the next run"
        )
        return None, record
    system_text, first_user = system, user
    initial: list[dict[str, str]] = [{"role": "user", "content": first_user}]
    convo = list(initial)
    cap = min(policy.DECISION_INITIAL_COMPLETION_TOKENS, policy.DECISION_MAX_COMPLETION_TOKENS)
    failures: list[str] = []
    for attempt in range(1, policy.DECISION_MAX_ATTEMPTS + 1):
        if attempt > 1 and time_left() < min_seconds_per_attempt:
            failures.append(f"attempt {attempt}: not made, the run is out of time")
            break
        record.attempts = attempt
        try:
            result = chat(
                convo,
                system=system_text,
                max_tokens=cap,
                temperature=policy.DECISION_TEMPERATURE,
            )
        except Exception as error:
            if type(error).__name__ == "LLMInputError" and not record.compact:
                record.compact = True
                system_text, first_user = compact_user()
                initial = [{"role": "user", "content": first_user}]
                convo = list(initial)
                failures.append(
                    f"attempt {attempt}: the runner refused the prompt ({error}); compact retry"
                )
                record.outcome = TRANSPORT
                continue
            record.outcome = classify_failure(error)
            failures.append(f"attempt {attempt}: {type(error).__name__}: {error}")
            break
        content = str(getattr(result, "content", "") or "")
        finish = str(getattr(result, "finish_reason", "") or "")
        record.model = str(getattr(result, "model", "") or "") or record.model
        request_id = getattr(result, "request_id", None)
        if request_id:
            record.request_ids.append(str(request_id))
        record.finish_reasons.append(finish)
        record.answers.append(content[: ERROR_CHARS * 4])
        for key, value in _usage(result).items():
            record.usage[key] = record.usage.get(key, 0) + value
        if finish == "length" or not content.strip():
            record.outcome = TRUNCATED
            reason = (
                "the answer was cut off at the cap"
                if finish == "length"
                else "the answer was empty"
            )
            if cap >= policy.DECISION_MAX_COMPLETION_TOKENS:
                failures.append(f"attempt {attempt}: {reason} at the ceiling; not repaired")
                break
            grown = min(
                cap * policy.DECISION_TRUNCATION_GROWTH, policy.DECISION_MAX_COMPLETION_TOKENS
            )
            failures.append(f"attempt {attempt}: {reason} ({cap} tokens); retrying at {grown}")
            cap = grown
            convo = list(initial)
            continue
        try:
            decision = parse_decision(content, book)
        except DecisionInvalid as invalid:
            record.outcome = INVALID
            record.complaints.append(list(invalid.complaints))
            failures.append(f"attempt {attempt}: invalid: {invalid}")
            convo = [
                *initial,
                {"role": "assistant", "content": content},
                {"role": "user", "content": complaint_message(invalid.complaints)},
            ]
            continue
        record.outcome = DECIDED
        return decision, record
    record.error = " | ".join(failures)[:ERROR_CHARS] or None
    return None, record
