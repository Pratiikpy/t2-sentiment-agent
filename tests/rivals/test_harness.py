"""The harness: every arm on the same snapshots, shown its own book, marked on the same hours by the
production simulator; and the token estimate the owner approves before an LLM arm runs."""

import math
from dataclasses import dataclass, field
from datetime import timedelta

import pytest

from rivals.rbuild import (
    BTC,
    HOUR,
    META,
    NVDA,
    PRICES,
    T0,
    TSLA,
    book,
    candles,
    features,
    item,
    simulator,
    snapshot,
)
from sentiment_agent.llm.fakes import ScriptedChatModel
from sentiment_agent.policy import POLICY_V1
from sentiment_agent.rivals.harness import (
    ShadowBook,
    arm_schedule,
    estimate_qwen_tokens,
    run_rivals,
)
from sentiment_agent.rivals.keyword_arm import KeywordSentimentArm
from sentiment_agent.rivals.registry import (
    FearGreedConfluenceArm,
    QwenHeadlineTraderArm,
    RivalArm,
    spec_for,
)
from sentiment_agent.rivals.tradingagents_arm import TradingAgentsSocialArm
from sentiment_agent.types import (
    ArmKind,
    ArmSpec,
    BookState,
    Candle,
    PerceptionSnapshot,
)

BULLISH_NVDA = [
    "$NVDA extremely bullish",
    "$NVDA rally, calls printing",
    "$NVDA beats, great quarter",
]


@dataclass
class ScriptedArm:
    """Returns scripted targets and records the books it was shown."""

    script: list[dict[str, float]]
    spec: ArmSpec = field(default_factory=lambda: spec_for("rival_lexicon_follow"))
    shown: list[BookState] = field(default_factory=list)

    def targets(self, snapshot: PerceptionSnapshot, book: BookState) -> dict[str, float]:
        self.shown.append(book)
        return dict(self.script[len(self.shown) - 1])


def _llm_spec() -> ArmSpec:
    return ArmSpec(
        arm_id="rival_unbounded",
        kind=ArmKind.RIVAL,
        title="t",
        description="d",
        provenance="p",
        uses_llm=True,
        guards=(),
    )


# --- the token estimate ------------------------------------------------------------------------


def test_the_estimate_is_each_llm_arms_bound_times_the_snapshots() -> None:
    model = ScriptedChatModel()
    tradingagents = TradingAgentsSocialArm(model=model, policy=POLICY_V1)
    headlines = QwenHeadlineTraderArm(model=model, policy=POLICY_V1)
    lexicon = KeywordSentimentArm(policy=POLICY_V1)
    arms: list[RivalArm] = [lexicon, tradingagents, headlines]
    per_snapshot = tradingagents.max_tokens_per_snapshot() + headlines.max_tokens_per_snapshot()
    assert per_snapshot > 0
    assert estimate_qwen_tokens(arms, 40) == 40 * per_snapshot
    assert estimate_qwen_tokens(arms, 0) == 0
    assert estimate_qwen_tokens([lexicon], 1_000) == 0


def test_an_llm_arm_that_cannot_bound_its_spend_cannot_be_estimated() -> None:
    unbounded = ScriptedArm(script=[], spec=_llm_spec())
    with pytest.raises(TypeError, match="cannot bound"):
        estimate_qwen_tokens([unbounded], 3)
    with pytest.raises(ValueError, match="non-negative"):
        estimate_qwen_tokens([], -1)


# --- the book an arm is shown --------------------------------------------------------------------


def _snaps(n: int, *, texts: list[str] | None = None) -> list[PerceptionSnapshot]:
    return [
        snapshot(
            [item(t, at=T0 + HOUR * i - timedelta(minutes=5)) for t in (texts or [])],
            at=T0 + HOUR * i,
            name=f"s{i}",
        )
        for i in range(n)
    ]


def test_an_arm_sees_its_own_positions_never_ours() -> None:
    arm = ScriptedArm(script=[{NVDA: 0.05}, {NVDA: 0.03, TSLA: -0.02}, {}])
    snaps = _snaps(3)
    ours = [book(weights={META: 0.05}, at=s.taken_at) for s in snaps]
    schedule = arm_schedule(arm, snaps, ours, [None] * 3, POLICY_V1)  # type: ignore[list-item]
    first, second, third = arm.shown
    assert first.positions == {}
    assert set(second.positions) == {NVDA}
    assert second.weight(NVDA) == pytest.approx(0.05)
    assert second.rebalances_today == {NVDA: 1}
    assert set(third.positions) == {NVDA, TSLA}
    assert third.weight(TSLA) == pytest.approx(-0.02)
    assert third.rebalances_today == {NVDA: 2, TSLA: 1}
    assert META not in first.positions
    assert META not in third.positions
    assert [entry[1] for entry in schedule] == [
        {NVDA: 0.05},
        {NVDA: 0.03, TSLA: -0.02},
        {NVDA: 0.0, TSLA: 0.0},  # explicit zeros: the kernel rules an unnamed holding as a hold
    ]


def test_the_shadow_entry_follows_where_the_arm_opened_and_added() -> None:
    shadow = ShadowBook()
    at0 = snapshot(at=T0, prices={NVDA: "200"})
    shadow.update(at0, book(), {NVDA: 0.02})
    at1 = snapshot(at=T0 + HOUR, prices={NVDA: "220"})
    shadow.update(at1, book(at=T0 + HOUR), {NVDA: 0.04})
    view = shadow.view(snapshot(at=T0 + 2 * HOUR, prices={NVDA: "210"}), book())
    assert float(view.positions[NVDA].avg_entry) == pytest.approx(210.0)  # (0.02*200+0.02*220)/0.04
    assert view.weight(NVDA) == pytest.approx(0.04)
    shadow.update(at1, book(), {NVDA: -0.01})  # a flip opens afresh at the current mark
    entry = shadow.holdings[NVDA].entry
    assert entry is not None
    assert float(entry) == pytest.approx(220.0)


def test_a_non_finite_weight_from_an_arm_stops_the_run() -> None:
    arm = ScriptedArm(script=[{NVDA: math.nan}])
    with pytest.raises(ValueError, match="non-finite"):
        arm_schedule(arm, _snaps(1), [book()], [None], POLICY_V1)  # type: ignore[list-item]


# --- the run -----------------------------------------------------------------------------------


def _marks(hours: int) -> dict[str, list[Candle]]:
    return {s: candles(s, T0, [PRICES[s]] * hours) for s in (BTC, NVDA, TSLA)}


def test_every_arm_gets_one_result_on_the_same_hours() -> None:
    sim = simulator(_marks(6))
    snaps = _snaps(3, texts=BULLISH_NVDA)
    books = [book(at=s.taken_at) for s in snaps]
    follow = KeywordSentimentArm(policy=POLICY_V1)
    fade = KeywordSentimentArm(policy=POLICY_V1, direction="fade")
    confluence = FearGreedConfluenceArm(policy=POLICY_V1)
    results = run_rivals([follow, fade, confluence], snaps, books, sim)
    assert [r.spec for r in results] == [follow.spec, fade.spec, confluence.spec]
    grids = [[m.at for m in r.marks] for r in results]
    assert grids[0] == grids[1] == grids[2]
    assert grids[0][0] == T0
    assert grids[0][-1] == T0 + 6 * HOUR
    assert results[0].marks[-1].net_weight == pytest.approx(0.05, rel=0.05)
    assert results[1].marks[-1].net_weight == pytest.approx(-0.05, rel=0.05)
    assert results[2].marks[-1].gross_weight == 0.0  # no Fear & Greed reading: it never traded
    for result in results:
        assert result.metrics.arm_id == result.spec.arm_id
        assert result.metrics.n_hours == len(result.marks) - 1


def test_the_confluence_entry_trades_when_its_two_signals_agree() -> None:
    sim = simulator(_marks(4))
    snaps = [
        snapshot(at=T0, fear_greed=15, feats={BTC: features(BTC, long_short=0.5)}, name="a"),
        snapshot(at=T0 + HOUR, fear_greed=50, feats={BTC: features(BTC, long_short=0.5)}, name="b"),
    ]
    (result,) = run_rivals(
        [FearGreedConfluenceArm(policy=POLICY_V1)], snaps, [book(), book(at=T0 + HOUR)], sim
    )
    # $50 of a 100,000 USDT account: the entry's own cap on a trade.
    assert result.marks[-1].net_weight == pytest.approx(0.0005, rel=0.05)


@pytest.mark.parametrize(
    ("snaps", "books", "match"),
    [
        (_snaps(2), [book()], "one book per snapshot"),
        ([], [], "no snapshots"),
        ([snapshot(at=T0, name="x"), snapshot(at=T0, name="y")], [book(), book()], "increasing"),
    ],
)
def test_malformed_runs_are_refused(
    snaps: list[PerceptionSnapshot], books: list[BookState], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        run_rivals([KeywordSentimentArm(policy=POLICY_V1)], snaps, books, simulator(_marks(3)))


def test_snapshots_past_the_last_mark_are_never_shown_to_an_arm() -> None:
    arm = ScriptedArm(script=[{NVDA: 0.05}, {NVDA: 0.0}, {NVDA: 0.0}])
    snaps = [snapshot(at=T0 + HOUR * i, name=f"t{i}") for i in (0, 2, 5)]
    books = [book(at=s.taken_at) for s in snaps]
    (result,) = run_rivals([arm], snaps, books, simulator(_marks(3)))  # marks end at T0 + 3h
    assert len(arm.shown) == 2
    assert result.marks[-1].at == T0 + 3 * HOUR
    late = [snapshot(at=T0 + 4 * HOUR, name="late")]
    with pytest.raises(ValueError, match="before the last Demo mark"):
        run_rivals([ScriptedArm(script=[{}])], late, [book()], simulator(_marks(3)))


def test_two_arms_with_one_id_are_refused() -> None:
    arms = [KeywordSentimentArm(policy=POLICY_V1), KeywordSentimentArm(policy=POLICY_V1)]
    with pytest.raises(ValueError, match="arm_id"):
        run_rivals(arms, _snaps(1), [book()], simulator(_marks(3)))
    with pytest.raises(ValueError, match="no rival arms"):
        run_rivals([], _snaps(1), [book()], simulator(_marks(3)))
