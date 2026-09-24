"""The daily Qwen token cap: checked before every call, charged after every call.

Ported from ARGUS ``src/argus/llm/qwen.py`` (``TokenBudget``), MIT, same author, at commit
``3dec6baf9dfa7be37c7b452e26a9b139df252f75``, sha256
``dcc4ba1077a63ca2dca4ee7ece18e566b90b29be2a3a5bfd1c6a987c2678daa9``. The ARGUS repository is
private, so its licence and every pin are recorded in ``third_party/argus/PROVENANCE.md``. Copied,
never imported (the independence rule in ``NOTICE.md``).

What was kept from ARGUS:

* The check happens **before** the call and refuses it; a budget that notices after spending is not
  a budget.
* A response without a usage block is **unmeasured, never free**: it adds nothing to the spend and
  increments a visible count of unreported calls, because inventing a number here would put a
  fabricated figure into the one value that guards a finite key.

What changed, and why:

* **Per UTC day, not per session.** The policy cap (``policy.decision.daily_token_cap``) is a daily
  limit, and a crash-and-restart must not hand the agent a fresh allowance. The state is logged as
  :class:`~sentiment_agent.types.BudgetState` events and rebuilt with :meth:`restore`.
* **Exactly the cap is allowed.** ARGUS refused at ``spent + projected >= limit``; a cap is the
  largest amount that may be spent, so this refuses only when the call could take spend *past* it.
* **The projection is a bound, not an estimate.** :func:`projected_tokens` counts the prompt at one
  token per UTF-8 byte (a byte-level BPE tokenizer cannot produce more tokens than bytes) plus the
  completion cap, so a refused call is one that genuinely could have crossed the cap. ARGUS
  projected ``max_tokens`` alone, which lets a large prompt carry the spend past the limit.
"""

import math
import threading
from collections.abc import Iterable, Sequence
from datetime import UTC
from typing import Final

from sentiment_agent.types import BudgetState, ChatMessage, Clock, LlmUsage, Policy

PER_MESSAGE_OVERHEAD_TOKENS: Final = 16
"""Upper bound on the chat-template tokens wrapped around one message.

Qwen's published chat template frames a message as ``<|im_start|>{role}\\n{content}<|im_end|>\\n``:
two special tokens, the role (at most nine bytes, ``assistant``) and two newlines, so at most 13
tokens under a byte-level BPE. Rounded up to 16. The template of ``qwen3.8-max`` itself is served
behind the gateway and was not read: NOT VERIFIED beyond the published Qwen templates."""

REQUEST_OVERHEAD_TOKENS: Final = 128
"""Upper bound on prompt tokens the request carries outside the message texts.

Measured, not assumed, that there is some: ARGUS's smallest live probe, a six-word prompt, billed
66 prompt tokens (``argus/src/argus/llm/qwen.py`` module docstring and ``tests/test_qwen.py``
``REAL_RESPONSE``, 2026-09-12). Six words and one message frame account for under 20 of those, so
the gateway adds roughly 50 of its own (a default system turn, the assistant priming turn, the
empty thinking block the Qwen3 template inserts when thinking is off). 128 is that observation with
more than 2x headroom, which also covers an instruction ``json_object`` mode may attach. The exact
server-side additions are NOT VERIFIED; the reported ``prompt_tokens`` of every live call is the
check, and a call whose reported prompt exceeds its bound would show the constant is too small."""


def projected_tokens(messages: Sequence[ChatMessage], max_tokens: int) -> int:
    """The most tokens one call with these messages and this completion cap can bill.

    Every published Qwen tokenizer is byte-level BPE: each byte has a token, and merges only reduce
    the count, so a text never tokenizes to more tokens than it has UTF-8 bytes. Counting bytes
    therefore bounds the prompt from above, loose by the text's bytes-per-token ratio (not measured
    on this model; the reported ``prompt_tokens`` of a real call shows it), and ``max_tokens``
    bounds the completion, reasoning included: ARGUS observed ``finish_reason: "length"`` with the
    whole cap spent on ``reasoning_content`` (``qwen.py`` ``complete_json``), so the cap covers
    both. The price of a true bound is headroom: near the end of a day a call is refused while its
    real cost might still have fitted. That is the direction a cap on a finite key must fail in.
    """
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    prompt_bound = sum(
        len(message.content.encode("utf-8")) + PER_MESSAGE_OVERHEAD_TOKENS for message in messages
    )
    return prompt_bound + REQUEST_OVERHEAD_TOKENS + max_tokens


MEASURED_PROMPT_BYTES: Final = 40_784
"""The initial prompt of the recorded dry-run decision (snapshot ``d19225c4…``, 2026-09-24): system
11,898 bytes and user 28,886 bytes as rendered by the current ``decision/prompt.py``
(``tests/fixtures/decision/recorded_prompt.json``). The cap arithmetic and the heartbeat reserve
start from it until a live decision has been logged; after that the reserve uses the larger of it
and the newest logged request."""

PROMPT_HEADROOM: Final = 1.25
"""Allowance for a snapshot richer than the recorded one (more stories, more calendar items, a held
book). The prompt's own byte bound is enforced separately (``tests/decision/test_prompt.py``)."""

RETRY_GROWTH_BYTES: Final = 8_192
"""What a complaint-fed retry adds to the conversation: the rejected answer and the complaint
message (``decision/contract.py``; a truncated answer is never fed back). The recorded answer is
~2.2 kB of JSON; 8 kB allows an answer near the largest the contract accepts. NOT VERIFIED as a
strict bound: the budget's own check at call time projects the real conversation regardless."""


def decision_bound(prompt_bytes: int, policy: Policy) -> int:
    """The most one decision can be charged under the projection: every attempt at the completion
    ceiling (a LOW call that truncates grows to it), every retry carrying
    :data:`RETRY_GROWTH_BYTES` more, two messages framed each, plus the request overhead."""
    if prompt_bytes < 0:
        raise ValueError("prompt_bytes cannot be negative")
    rule = policy.decision
    first = prompt_bytes + 2 * PER_MESSAGE_OVERHEAD_TOKENS + REQUEST_OVERHEAD_TOKENS
    first += rule.max_completion_tokens
    retry = first + RETRY_GROWTH_BYTES + 2 * PER_MESSAGE_OVERHEAD_TOKENS
    return first + (rule.max_attempts - 1) * retry


def heartbeats_per_day(policy: Policy) -> int:
    """Scheduled heartbeats on a weekday: each funding settlement, and the US open."""
    return len(policy.triggers.funding_heartbeat_hours_utc) + 1


def worst_case_day(prompt_bytes: int, policy: Policy) -> int:
    """The projection of the policy's busiest day: every heartbeat and every event decision the
    trigger rules admit, each taking every attempt at the completion ceiling, on a prompt of
    ``prompt_bytes`` grown by :data:`PROMPT_HEADROOM`. Owner requests are the owner's own spend
    and are not included. ``policy.decision.daily_token_cap`` must be at least this for
    ``MEASURED_PROMPT_BYTES`` (``tests/llm/test_budget.py``), so the cap cannot refuse a scheduled
    decision and turn itself into a model-outage flatten."""
    grown = math.ceil(prompt_bytes * PROMPT_HEADROOM)
    decisions = heartbeats_per_day(policy) + policy.triggers.max_event_decisions_per_day
    return decisions * decision_bound(grown, policy)


class BudgetExhausted(RuntimeError):  # noqa: N818 - the name is the module contract (DESIGN M6)
    """The daily cap would be crossed. Raised before the call, so nothing was sent or spent."""

    def __init__(
        self,
        message: str,
        *,
        state: BudgetState | None = None,
        projected: int | None = None,
    ) -> None:
        super().__init__(message)
        self.state = state
        self.projected = projected


def utc_day(clock: Clock) -> str:
    """Today's UTC date as ``YYYY-MM-DD`` (the key a :class:`BudgetState` is filed under)."""
    return clock.now().astimezone(UTC).date().isoformat()


class DailyTokenBudget:
    """A hard daily ceiling on Qwen tokens, reset at 00:00 UTC.

    ``spent_tokens`` is the sum of what the endpoint reported, and nothing else. A call the endpoint
    did not bill in writing (no usage block, a dropped stream, a timeout) is counted in
    ``unreported_calls`` so a reader sees the spend beside the number of calls it does not include.

    Spend is filed under the UTC day on which it was recorded. A clock that steps backwards across
    midnight does not reopen an earlier day: the counters stay with the later day, so the cap can
    only be reached sooner, never later.
    """

    def __init__(self, cap_tokens: int, clock: Clock) -> None:
        if isinstance(cap_tokens, bool) or not isinstance(cap_tokens, int) or cap_tokens <= 0:
            raise ValueError("cap_tokens must be a positive integer")
        self._cap = cap_tokens
        self._clock = clock
        self._lock = threading.Lock()
        self._day = utc_day(clock)
        self._spent = 0
        self._calls = 0
        self._unreported = 0

    @property
    def cap_tokens(self) -> int:
        return self._cap

    def check(self, projected: int) -> None:
        """Refuse a call that could carry today's spend past the cap.

        ``projected`` is the most the call can bill (see :func:`projected_tokens`). A call that fits
        exactly is allowed.
        """
        if isinstance(projected, bool) or not isinstance(projected, int) or projected < 0:
            raise ValueError("projected must be a non-negative integer")
        with self._lock:
            self._roll()
            if self._spent + projected > self._cap:
                state = self._snapshot()
                raise BudgetExhausted(
                    f"daily Qwen token cap reached for {state.day}: {state.spent_tokens} of "
                    f"{state.cap_tokens} spent, this call could bill up to {projected} "
                    f"({state.remaining} remain, {state.unreported_calls} calls unreported). "
                    "The cap resets at 00:00 UTC; raising it is the owner's decision.",
                    state=state,
                    projected=projected,
                )

    def record(self, usage: LlmUsage) -> BudgetState:
        """Charge one call. An unreported usage adds no tokens and is counted as unreported."""
        with self._lock:
            self._roll()
            self._calls += 1
            if usage.reported:
                self._spent += usage.total_tokens
            else:
                self._unreported += 1
            return self._snapshot()

    def state(self) -> BudgetState:
        with self._lock:
            self._roll()
            return self._snapshot()

    def restore(self, states: Iterable[BudgetState]) -> None:
        """Rebuild today's counters from logged states after a restart.

        States from earlier days are history and are ignored. Each counter takes the largest value
        any of today's states (or the live counters) carry, so a replay in any order, or one state
        logged twice, can never lower the spend. The cap stays the one this budget was built with:
        a logged state records the cap in force then, and changing the cap is an amendment, not a
        side effect of a restart. A state dated after today means the clock is behind the ledger,
        which would otherwise reopen spend already made, so it is refused.
        """
        with self._lock:
            self._roll()
            for state in states:
                if state.day > self._day:
                    raise ValueError(
                        f"a budget state is dated {state.day}, after today ({self._day}): the "
                        "clock is behind the ledger"
                    )
                if state.day != self._day:
                    continue
                self._spent = max(self._spent, state.spent_tokens)
                self._calls = max(self._calls, state.calls)
                self._unreported = max(self._unreported, state.unreported_calls)

    # --- internals ---------------------------------------------------------------------------

    def _roll(self) -> None:
        today = utc_day(self._clock)
        if today > self._day:
            self._day = today
            self._spent = 0
            self._calls = 0
            self._unreported = 0

    def _snapshot(self) -> BudgetState:
        return BudgetState(
            day=self._day,
            cap_tokens=self._cap,
            spent_tokens=self._spent,
            calls=self._calls,
            unreported_calls=self._unreported,
        )


__all__ = [
    "MEASURED_PROMPT_BYTES",
    "PER_MESSAGE_OVERHEAD_TOKENS",
    "PROMPT_HEADROOM",
    "REQUEST_OVERHEAD_TOKENS",
    "RETRY_GROWTH_BYTES",
    "BudgetExhausted",
    "DailyTokenBudget",
    "decision_bound",
    "heartbeats_per_day",
    "projected_tokens",
    "utc_day",
    "worst_case_day",
]
