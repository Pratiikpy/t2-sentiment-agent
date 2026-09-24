"""TradingAgents' sentiment analyst and trader on a scripted model: the prompts are the adapted
upstream text, the attribution travels with them, and the answers become weights inside the caps."""

import inspect
from datetime import timedelta
from typing import Any

import pytest

from rivals.rbuild import BTC, META, NVDA, T0, TSLA, book, item, snapshot, text_completion
from sentiment_agent.clock import ManualClock
from sentiment_agent.llm.budget import BudgetExhausted, DailyTokenBudget, projected_tokens
from sentiment_agent.llm.client import QwenTransportError
from sentiment_agent.llm.fakes import ScriptedChatModel, ScriptExhausted, completion_from_json
from sentiment_agent.policy import POLICY_V1
from sentiment_agent.rivals import tradingagents_arm
from sentiment_agent.rivals.registry import TRADINGAGENTS
from sentiment_agent.rivals.tradingagents_arm import (
    NO_EXTERNAL_TOOLS,
    TradingAgentsSocialArm,
    action_from_text,
    display_ticker,
    instrument_context,
    parse_sentiment_report,
    parse_trader_proposal,
)
from sentiment_agent.types import Completion, Thinking


def report(band: str = "Bullish", score: float = 7.5) -> Completion:
    return completion_from_json(
        {
            "overall_band": band,
            "overall_score": score,
            "confidence": "medium",
            "narrative": "X is enthusiastic about the quarter; news is constructive.",
        }
    )


def proposal(action: str, sizing: str | None = "3% of portfolio") -> Completion:
    body: dict[str, Any] = {
        "action": action,
        "reasoning": "Sentiment is constructive and broad.",
        "entry_price": 200.5,
        "stop_loss": "190",
        "position_sizing": sizing,
    }
    return completion_from_json(body)


def nvda_snapshot() -> Any:
    return snapshot(
        [
            item("$NVDA calls printing, AI demand is insane", source="trader_joe"),
            item("NVDA earnings beat, guidance raised", channel="news", source="Reuters"),
            item(
                "NVDA to 300?\nThe data center numbers were huge.",
                channel="reddit",
                source="r/wallstreetbets",
            ),
        ]
    )


def test_the_chain_turns_a_bullish_report_and_a_buy_into_a_sized_long() -> None:
    model = ScriptedChatModel([report(), proposal("Buy")])
    arm = TradingAgentsSocialArm(model=model, policy=POLICY_V1)
    assert arm.spec.arm_id == TRADINGAGENTS
    assert arm.targets(nvda_snapshot(), book()) == {NVDA: pytest.approx(0.03)}
    assert [c.stage for c in arm.log.calls] == ["analyst", "trader"]
    assert arm.log.failures == 0
    for request in model.requests:
        assert request.json_mode
        assert request.thinking is Thinking.LOW
        assert request.temperature == 0.0


def test_the_analyst_prompt_is_the_adapted_upstream_text() -> None:
    model = ScriptedChatModel([report(), proposal("Hold")])
    TradingAgentsSocialArm(model=model, policy=POLICY_V1).targets(nvda_snapshot(), book())
    system, user = model.requests[0].messages
    assert user.role == "user"
    assert user.content == "NVDA"
    text = system.content
    assert text.startswith(
        "You are a helpful AI assistant, collaborating with other assistants. If you or any other "
        "assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable"
    )
    assert "Today's date is 2026-09-23; treat it as 'now' for all analysis." in text
    assert instrument_context("NVDA", "stock") in text
    assert NO_EXTERNAL_TOOLS in text
    assert "You are a financial market sentiment analyst." in text
    assert "covering the period from 2026-09-16 to 2026-09-23" in text
    assert "## How to analyze this data (best practices)" in text
    assert "8. **Past sentiment is not predictive.**" in text
    assert "### NVDA earnings beat, guidance raised (source: Reuters)" in text
    assert "@trader_joe · no-label] $NVDA calls printing, AI demand is insane" in text
    assert "Unlabeled: 1 · Total: 1 most-recent messages" in text
    assert "r/wallstreetbets — 1 recent posts mentioning NVDA:" in text
    assert "body excerpt: The data center numbers were huge." in text
    assert '"overall_band"' in text
    assert "JSON" in text


def test_the_trader_prompt_is_the_adapted_upstream_text_with_the_report_as_the_plan() -> None:
    model = ScriptedChatModel([report(), proposal("Hold")])
    TradingAgentsSocialArm(model=model, policy=POLICY_V1).targets(nvda_snapshot(), book())
    system, user = model.requests[1].messages
    assert system.content.startswith(
        "You are a trading agent analyzing market data to make investment decisions. Based on "
        "your analysis, provide a specific recommendation to buy, sell, or hold. State entry price"
    )
    assert "Ground concrete price levels" not in system.content  # no market report upstream
    assert '"position_sizing"' in system.content
    assert user.content.startswith("Here is the research team's investment plan for NVDA.")
    assert "Proposed Investment Plan:\n**Overall Sentiment:** **Bullish** (Score: 7.5/10)" in (
        user.content
    )
    assert user.content.endswith("Make an informed, strategic trading decision.")


def test_the_attribution_and_the_changes_travel_with_the_adapted_text() -> None:
    source = inspect.getsource(tradingagents_arm)
    assert "https://github.com/TauricResearch/TradingAgents" in source
    assert "Apache License, Version 2.0" in source
    assert "CHANGES MADE (Apache-2.0 section 4(b))" in source
    assert "sentiment_analyst.py:88-99" in source
    assert "trader.py:46-74" in source


@pytest.mark.parametrize(
    ("action", "sizing", "expected"),
    [
        ("Sell", "2% of portfolio", -0.02),
        ("Buy", "10% of portfolio", 0.05),
        ("Buy", None, 0.05),
        ("Buy", "a starter position", 0.05),
        ("buy", "0%", 0.0),
    ],
)
def test_actions_and_sizes_become_weights_inside_the_cap(
    action: str, sizing: str | None, expected: float
) -> None:
    model = ScriptedChatModel([report(), proposal(action, sizing)])
    targets = TradingAgentsSocialArm(model=model, policy=POLICY_V1).targets(nvda_snapshot(), book())
    assert targets.get(NVDA, 0.0) == pytest.approx(expected)


def test_hold_keeps_the_position_and_other_holdings_are_untouched() -> None:
    model = ScriptedChatModel([report(), proposal("Hold")])
    arm = TradingAgentsSocialArm(model=model, policy=POLICY_V1, max_symbols=1)
    held = book(weights={NVDA: 0.02, META: -0.01})
    assert arm.targets(nvda_snapshot(), held) == {
        NVDA: pytest.approx(0.02),
        META: pytest.approx(-0.01),
    }


def test_free_text_answers_are_used_the_way_upstream_falls_back() -> None:
    model = ScriptedChatModel(
        [
            text_completion("not json at all"),
            text_completion("Sentiment is bearish across every source."),
            text_completion("still prose"),
            text_completion("After weighing it all: FINAL TRANSACTION PROPOSAL: **SELL**"),
        ]
    )
    arm = TradingAgentsSocialArm(model=model, policy=POLICY_V1)
    assert arm.targets(nvda_snapshot(), book()) == {NVDA: pytest.approx(-0.05)}
    assert [c.stage for c in arm.log.calls] == [
        "analyst",
        "analyst_freetext",
        "trader",
        "trader_freetext",
    ]
    assert not model.requests[1].json_mode
    assert not model.requests[3].json_mode
    assert "Sentiment is bearish across every source." in model.requests[2].messages[1].content


def test_an_answer_with_no_action_is_review_and_holds() -> None:
    model = ScriptedChatModel(
        [report(), text_completion("hmm"), text_completion("I cannot decide today.")]
    )
    arm = TradingAgentsSocialArm(model=model, policy=POLICY_V1)
    assert arm.targets(nvda_snapshot(), book(weights={NVDA: 0.01})) == {NVDA: pytest.approx(0.01)}
    assert arm.decisions[-1].action is None
    assert arm.log.calls[-1].detail == "REVIEW: no recognisable action"


def test_a_transport_failure_holds_and_is_recorded_but_a_budget_refusal_stops_the_run() -> None:
    failing = ScriptedChatModel([QwenTransportError("gateway 502"), QwenTransportError("again")])
    arm = TradingAgentsSocialArm(model=failing, policy=POLICY_V1)
    assert arm.targets(nvda_snapshot(), book(weights={NVDA: 0.01})) == {NVDA: pytest.approx(0.01)}
    assert arm.log.failures == 2
    budget = DailyTokenBudget(100, ManualClock(T0))
    broke = TradingAgentsSocialArm(
        model=ScriptedChatModel([report()], budget=budget), policy=POLICY_V1
    )
    with pytest.raises(BudgetExhausted):
        broke.targets(nvda_snapshot(), book())


def test_held_symbols_come_first_then_the_most_discussed_up_to_the_limit() -> None:
    texts = [
        *(item(f"$TSLA post {n}") for n in range(3)),
        *(item(f"$NVDA post {n}") for n in range(2)),
        item("$META post"),
        item("bitcoin post", at=T0 - timedelta(days=8)),  # outside the seven-day window
    ]
    snap = snapshot(texts)
    model = ScriptedChatModel([report(), proposal("Hold")] * 3)
    arm = TradingAgentsSocialArm(model=model, policy=POLICY_V1, max_symbols=3)
    arm.targets(snap, book(weights={BTC: 0.01}))
    assert [d.symbol for d in arm.decisions] == [BTC, TSLA, NVDA]
    assert model.remaining == 0
    with pytest.raises(ScriptExhausted):
        arm.targets(snap, book(weights={BTC: 0.01}))


def test_display_tickers_are_what_a_tradingagents_user_types() -> None:
    assert display_ticker(NVDA, POLICY_V1) == "NVDA"
    assert display_ticker(BTC, POLICY_V1) == "BTC-USD"
    assert display_ticker("SP500USDT", POLICY_V1) == "SPX"
    assert "Treat it as a crypto asset" in instrument_context("BTC-USD", "crypto")


def test_parsers_accept_the_schemas_and_refuse_everything_else() -> None:
    parsed = parse_sentiment_report(report("mildly bearish", 4.0).content)
    assert parsed is not None
    assert parsed.overall_band == "Mildly Bearish"
    assert "**Overall Sentiment:** **Mildly Bearish** (Score: 4.0/10)" in parsed.render()
    assert parse_sentiment_report(report("Euphoric").content) is None
    assert parse_sentiment_report(report("Bullish", 11.0).content) is None
    trader = parse_trader_proposal(proposal("SELL").content)
    assert trader is not None
    assert trader.action == "Sell"
    assert trader.stop_loss == 190.0
    assert parse_trader_proposal('{"action": "Short"}') is None
    assert action_from_text("**Action**: Buy") == "Buy"
    assert action_from_text("I would hold here.") == "Hold"
    assert action_from_text("Buyers and holders everywhere") is None
    assert action_from_text("") is None


def test_the_token_bound_covers_every_call_the_chain_makes() -> None:
    long_posts = [item("$NVDA " + "\U0001d54f" * 400, source="x" * 90) for _ in range(40)]
    news = [
        item("NVDA " + "headline " * 80 + "\n" + "summary " * 200, channel="news")
        for _ in range(30)
    ]
    model = ScriptedChatModel(
        [text_completion("no"), text_completion("x" * 20_000), text_completion("no")] * 1
        + [text_completion("FINAL TRANSACTION PROPOSAL: **BUY**")]
    )
    arm = TradingAgentsSocialArm(model=model, policy=POLICY_V1, max_symbols=1)
    arm.targets(snapshot([*long_posts, *news]), book())
    spent = sum(projected_tokens(r.messages, r.max_tokens) for r in model.requests)
    assert len(model.requests) == 4
    assert spent <= arm.max_tokens_per_snapshot()
    assert (
        TradingAgentsSocialArm(
            model=model, policy=POLICY_V1, max_symbols=3
        ).max_tokens_per_snapshot()
        == 3 * arm.max_tokens_per_snapshot()
    )


def test_a_symbol_limit_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="max_symbols"):
        TradingAgentsSocialArm(model=ScriptedChatModel(), policy=POLICY_V1, max_symbols=0)


def test_tsla_is_a_stock_and_bitcoin_a_crypto_asset_in_the_prompt() -> None:
    model = ScriptedChatModel([report(), proposal("Hold"), report(), proposal("Hold")])
    arm = TradingAgentsSocialArm(model=model, policy=POLICY_V1)
    arm.targets(snapshot([item("$TSLA up"), item("bitcoin up")]), book())
    systems = [r.messages[0].content for r in model.requests[::2]]
    assert any(instrument_context("BTC-USD", "crypto") in s for s in systems)
    assert any(instrument_context("TSLA", "stock") in s for s in systems)
    assert TSLA in {d.symbol for d in arm.decisions}
