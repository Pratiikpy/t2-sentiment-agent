"""The daily token cap: refused before a call, charged after, reset at 00:00 UTC, rebuilt after a
restart, and never lowered by a missing usage block."""

import json
import math
import threading
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from sentiment_agent.clock import ManualClock
from sentiment_agent.llm.budget import (
    MEASURED_PROMPT_BYTES,
    PER_MESSAGE_OVERHEAD_TOKENS,
    PROMPT_HEADROOM,
    REQUEST_OVERHEAD_TOKENS,
    RETRY_GROWTH_BYTES,
    BudgetExhausted,
    DailyTokenBudget,
    decision_bound,
    heartbeats_per_day,
    projected_tokens,
    utc_day,
    worst_case_day,
)
from sentiment_agent.policy import POLICY_V1
from sentiment_agent.types import BudgetState, ChatMessage, LlmUsage


def usage(total: int, *, reported: bool = True) -> LlmUsage:
    if not reported:
        return LlmUsage(
            prompt_tokens=0, completion_tokens=0, reasoning_tokens=0, total_tokens=0, reported=False
        )
    return LlmUsage(
        prompt_tokens=total // 2,
        completion_tokens=total - total // 2,
        reasoning_tokens=0,
        total_tokens=total,
        reported=True,
    )


def state(day: str, spent: int, calls: int, unreported: int = 0, cap: int = 1000) -> BudgetState:
    return BudgetState(
        day=day, cap_tokens=cap, spent_tokens=spent, calls=calls, unreported_calls=unreported
    )


@pytest.mark.parametrize("cap", [0, -1, True, 1.5])
def test_the_cap_must_be_a_positive_integer(cap: object, clock: ManualClock) -> None:
    with pytest.raises(ValueError, match="cap_tokens"):
        DailyTokenBudget(cap, clock)  # type: ignore[arg-type]


def test_a_fresh_budget_is_empty_and_dated_today(clock: ManualClock) -> None:
    budget = DailyTokenBudget(150_000, clock)
    assert budget.state() == state("2026-09-23", 0, 0, cap=150_000)
    assert budget.state().remaining == 150_000
    assert budget.cap_tokens == 150_000


def test_a_call_that_fits_exactly_is_allowed_one_more_token_is_not(clock: ManualClock) -> None:
    budget = DailyTokenBudget(1000, clock)
    budget.record(usage(600))
    budget.check(400)
    with pytest.raises(BudgetExhausted) as caught:
        budget.check(401)
    error = caught.value
    assert error.projected == 401
    assert error.state == state("2026-09-23", 600, 1)
    assert "600 of 1000 spent" in str(error)
    assert "up to 401" in str(error)
    assert "00:00 UTC" in str(error)


@pytest.mark.parametrize("projected", [-1, True, 2.0])
def test_a_projection_must_be_a_non_negative_integer(projected: object, clock: ManualClock) -> None:
    with pytest.raises(ValueError, match="projected"):
        DailyTokenBudget(1000, clock).check(projected)  # type: ignore[arg-type]


def test_a_refused_check_changes_nothing(clock: ManualClock) -> None:
    budget = DailyTokenBudget(100, clock)
    with pytest.raises(BudgetExhausted):
        budget.check(101)
    assert budget.state() == state("2026-09-23", 0, 0, cap=100)


def test_record_adds_the_reported_total(clock: ManualClock) -> None:
    budget = DailyTokenBudget(10_000, clock)
    after = budget.record(
        LlmUsage(
            prompt_tokens=66,
            completion_tokens=26,
            reasoning_tokens=23,
            total_tokens=92,
            cached_tokens=10,
            reported=True,
        )
    )
    assert after == state("2026-09-23", 92, 1, cap=10_000)


def test_an_unreported_call_is_counted_and_never_free(clock: ManualClock) -> None:
    budget = DailyTokenBudget(1000, clock)
    budget.record(usage(300))
    after = budget.record(usage(0, reported=False))
    assert (after.spent_tokens, after.calls, after.unreported_calls) == (300, 2, 1)


def test_the_day_turns_at_utc_midnight(clock: ManualClock) -> None:
    clock.set(datetime(2026, 9, 23, 23, 59, 59, 999_000, tzinfo=UTC))
    budget = DailyTokenBudget(1000, clock)
    budget.record(usage(1000))
    with pytest.raises(BudgetExhausted):
        budget.check(1)
    clock.advance(timedelta(milliseconds=1))
    budget.check(1000)
    assert budget.state() == state("2026-09-24", 0, 0)


def test_the_day_is_the_utc_date_whatever_zone_the_clock_reports() -> None:
    class IstClock:
        def now(self) -> datetime:
            # 00:30 on the 24th in India is still 19:00 on the 23rd in UTC.
            return datetime(2026, 9, 24, 0, 30, tzinfo=timezone(timedelta(hours=5, minutes=30)))

    assert utc_day(IstClock()) == "2026-09-23"
    assert DailyTokenBudget(10, IstClock()).state().day == "2026-09-23"


def test_restore_rebuilds_today_and_ignores_history(clock: ManualClock) -> None:
    budget = DailyTokenBudget(1000, clock)
    budget.restore(
        [
            state("2026-09-22", 990, 12, 1),
            state("2026-09-23", 200, 2),
            state("2026-09-23", 450, 4, 1),
        ]
    )
    assert budget.state() == state("2026-09-23", 450, 4, 1)
    with pytest.raises(BudgetExhausted):
        budget.check(551)


def test_restore_is_order_independent_and_idempotent(clock: ManualClock) -> None:
    logged = [state("2026-09-23", 200, 2), state("2026-09-23", 450, 4, 1)]
    forward, backward, twice = (DailyTokenBudget(1000, clock) for _ in range(3))
    forward.restore(logged)
    backward.restore(reversed(logged))
    twice.restore(logged + logged)
    assert forward.state() == backward.state() == twice.state()


def test_restore_never_lowers_what_this_process_has_counted(clock: ManualClock) -> None:
    budget = DailyTokenBudget(1000, clock)
    budget.record(usage(700))
    budget.restore([state("2026-09-23", 100, 1)])
    assert budget.state().spent_tokens == 700


def test_restore_keeps_the_configured_cap(clock: ManualClock) -> None:
    budget = DailyTokenBudget(1000, clock)
    budget.restore([state("2026-09-23", 400, 3, cap=150_000)])
    after = budget.state()
    assert (after.cap_tokens, after.spent_tokens) == (1000, 400)


def test_restore_refuses_a_state_from_the_future(clock: ManualClock) -> None:
    budget = DailyTokenBudget(1000, clock)
    with pytest.raises(ValueError, match="clock is behind the ledger"):
        budget.restore([state("2026-09-24", 10, 1)])
    assert budget.state().spent_tokens == 0


def test_records_from_many_threads_are_all_counted(clock: ManualClock) -> None:
    budget = DailyTokenBudget(10_000_000, clock)

    def work() -> None:
        for _ in range(500):
            budget.record(usage(3))

    threads = [threading.Thread(target=work) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    after = budget.state()
    assert (after.calls, after.spent_tokens) == (4000, 12_000)


def test_the_projection_bounds_the_prompt_by_its_bytes() -> None:
    messages = [
        ChatMessage(role="system", content="Answer in JSON."),
        ChatMessage(role="user", content="Funding z 2.4 — the crowd is long ¥"),
    ]
    content_bytes = sum(len(m.content.encode("utf-8")) for m in messages)
    assert content_bytes > sum(len(m.content) for m in messages)  # the dash and yen are multibyte
    expected = content_bytes + 2 * PER_MESSAGE_OVERHEAD_TOKENS + REQUEST_OVERHEAD_TOKENS + 4096
    assert projected_tokens(messages, 4096) == expected


def test_the_projection_grows_with_the_prompt_and_the_cap() -> None:
    short = [ChatMessage(role="user", content="x")]
    long = [ChatMessage(role="user", content="x" * 1000)]
    assert projected_tokens(long, 100) - projected_tokens(short, 100) == 999
    assert projected_tokens(short, 200) - projected_tokens(short, 100) == 100


@pytest.mark.parametrize("max_tokens", [0, -5, True])
def test_the_projection_needs_a_positive_completion_cap(max_tokens: int) -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        projected_tokens([ChatMessage(role="user", content="x")], max_tokens)


def test_the_prompt_bound_covers_the_measured_gateway_overhead() -> None:
    """ARGUS's smallest live probe, a six-word prompt, billed 66 prompt tokens (2026-09-12). The
    prompt part of the bound for a six-word message must not be below what was actually billed."""
    six_words = [ChatMessage(role="user", content="Reply with the single word OK")]
    assert projected_tokens(six_words, 1) - 1 >= 66


# --- the cap against the policy's busiest day -----------------------------------------------------

RECORDED_PROMPT = (
    Path(__file__).resolve().parents[1] / "fixtures" / "decision" / "recorded_prompt.json"
)


def _recorded_messages() -> list[ChatMessage]:
    data = json.loads(RECORDED_PROMPT.read_text(encoding="utf-8"))
    return [ChatMessage(role=m["role"], content=m["content"]) for m in data["messages"]]


def test_the_measured_prompt_is_the_recorded_one() -> None:
    messages = _recorded_messages()
    assert [m.role for m in messages] == ["system", "user"]
    assert sum(len(m.content.encode("utf-8")) for m in messages) == MEASURED_PROMPT_BYTES


def test_one_decision_bound_covers_every_attempt_the_contract_can_make() -> None:
    """Each attempt at the completion ceiling, each retry carrying the rejected answer and the
    complaint, projected exactly as the budget projects a call."""
    rule = POLICY_V1.decision
    messages = _recorded_messages()
    first = projected_tokens(messages, rule.max_completion_tokens)
    retry_convo = [
        *messages,
        ChatMessage(role="assistant", content="x" * (RETRY_GROWTH_BYTES // 2)),
        ChatMessage(role="user", content="y" * (RETRY_GROWTH_BYTES // 2)),
    ]
    retry = projected_tokens(retry_convo, rule.max_completion_tokens)
    assert decision_bound(MEASURED_PROMPT_BYTES, POLICY_V1) == first + (rule.max_attempts - 1) * (
        retry
    )


def test_the_daily_cap_carries_the_busiest_day_the_policy_allows() -> None:
    """4 heartbeats and 8 event decisions, every one taking all three attempts at the ceiling, on
    the recorded prompt grown by the headroom. Below this the cap would refuse a scheduled call,
    and a refused call is answered by the kernel's outage flatten (review finding)."""
    assert heartbeats_per_day(POLICY_V1) == 4
    day = worst_case_day(MEASURED_PROMPT_BYTES, POLICY_V1)
    assert day == 2_333_328  # the arithmetic written into policy.decision.basis
    assert day <= POLICY_V1.decision.daily_token_cap
    assert "2,333,328" in POLICY_V1.decision.basis


def test_the_cap_is_replayed_call_by_call_without_a_refusal() -> None:
    """The budget itself, driven through the worst day: each call checked with its projection and
    charged the whole projection (the most it could bill). Nothing is refused."""
    clock = ManualClock(datetime(2026, 9, 23, 0, 30, tzinfo=UTC))
    budget = DailyTokenBudget(POLICY_V1.decision.daily_token_cap, clock)
    rule = POLICY_V1.decision
    grown = math.ceil(MEASURED_PROMPT_BYTES * PROMPT_HEADROOM)
    first = grown + 2 * PER_MESSAGE_OVERHEAD_TOKENS + REQUEST_OVERHEAD_TOKENS
    first += rule.max_completion_tokens
    retry = first + RETRY_GROWTH_BYTES + 2 * PER_MESSAGE_OVERHEAD_TOKENS
    decisions = heartbeats_per_day(POLICY_V1) + POLICY_V1.triggers.max_event_decisions_per_day
    for _ in range(decisions):
        for attempt in range(rule.max_attempts):
            cost = first if attempt == 0 else retry
            budget.check(cost)
            budget.record(
                LlmUsage(
                    prompt_tokens=cost - rule.max_completion_tokens,
                    completion_tokens=rule.max_completion_tokens,
                    reasoning_tokens=0,
                    total_tokens=cost,
                    reported=True,
                )
            )
    assert budget.state().spent_tokens == decisions * decision_bound(grown, POLICY_V1)
    assert budget.state().remaining >= 0
