"""The page says only what happened (readiness audit, 2026-09-25): no "orders were sent" banner over
zero orders, no "posted on X" over a draft, no "what Qwen decided" over a scripted stand-in."""

import pytest

from sentiment_agent.site.cards import decided_by
from sentiment_agent.site.render import mode_banner
from sentiment_agent.types import LlmCallRecord, LlmOutcome, LlmUsage, Thinking

HASH = "a" * 64


def _call(tokens: int, latency_ms: int, *, reported: bool = True) -> LlmCallRecord:
    usage = LlmUsage(
        prompt_tokens=tokens,
        completion_tokens=0,
        reasoning_tokens=0,
        total_tokens=tokens,
        reported=reported,
    )
    return LlmCallRecord(
        model="qwen3.8-max",
        thinking=Thinking.LOW,
        prompt_version="v1",
        prompt_hash=HASH,
        request_blob=None,
        response_blobs=(),
        attempts=1,
        usage=usage,
        latency_ms=latency_ms,
        outcome=LlmOutcome.DECIDED,
    )


class TestTheBanner:
    def test_a_paper_log_with_no_order_does_not_say_orders_were_sent(self) -> None:
        text = mode_banner("paper", 0)
        assert "No order has been sent yet" in text
        assert "orders were sent" not in text

    def test_a_paper_log_with_orders_counts_them(self) -> None:
        assert "3 order(s) sent through Bitget Agent Hub" in mode_banner("paper", 3)


class TestWhoDecided:
    def test_a_metered_call_is_the_model(self) -> None:
        assert decided_by(_call(12_345, 8_100)) == "qwen3.8-max (12,345 tokens, 8.1 s)"

    def test_a_call_with_no_usage_and_no_time_is_the_scripted_stand_in(self) -> None:
        assert decided_by(_call(0, 0)).startswith("a scripted stand-in")

    @pytest.mark.parametrize(("tokens", "latency"), [(0, 950), (400, 0)])
    def test_either_sign_of_a_real_call_keeps_the_model_name(
        self, tokens: int, latency: int
    ) -> None:
        assert decided_by(_call(tokens, latency)).startswith("qwen3.8-max")

    def test_unreported_usage_is_said(self) -> None:
        assert "tokens not reported" in decided_by(_call(0, 900, reported=False))
