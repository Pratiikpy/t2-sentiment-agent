"""G3's net and cluster caps (policy v2, declared change run2-a3).

Run 2's agent, replayed on run 1's triggers, proposed books 100% short in 14 of 14 decisions, up to
20% net, and no guard limited them (validation/run2/act_rate.json). Policy v2 caps the book's net
weight at 10% and the crypto-beta names (MSTR, COIN, HOOD, CRCL) at 7.5% gross together."""

import pytest

from helpers import T0
from kernel.kbuild import UNIVERSE, book, breaker_state, decision, inputs
from sentiment_agent.clock import ManualClock
from sentiment_agent.kernel.guards import allocate_caps
from sentiment_agent.kernel.kernel import RiskKernel
from sentiment_agent.policy import POLICY_V1, POLICY_V2
from sentiment_agent.types import Activation, GuardId, KernelRuling, Policy

GOOGL, HOOD, MSTR, TSLA, COIN, NVDA = (
    "GOOGLUSDT",
    "HOODUSDT",
    "MSTRUSDT",
    "TSLAUSDT",
    "COINUSDT",
    "NVDAUSDT",
)


def _rule(proposed: dict[str, float], policy: Policy = POLICY_V2) -> KernelRuling:
    kernel = RiskKernel(policy, ManualClock(T0))
    return kernel.rule(
        proposed=proposed,
        book=book(at=T0),
        inputs=inputs(UNIVERSE, at=T0, snapshot_at=T0),
        context=decision(UNIVERSE),
        breaker=breaker_state(Activation.ACTIVE, (), at=T0),
    )


def _approved(ruling: KernelRuling) -> dict[str, float]:
    return {ir.symbol: ir.approved_weight for ir in ruling.instruments if ir.approved_weight}


def test_the_replayed_all_short_book_is_cut_to_the_net_cap() -> None:
    # Run 2's agent at 12:05 UTC on 25 Sep: four names 5% short each, 20% net short.
    ruling = _rule({TSLA: -0.05, HOOD: -0.05, MSTR: -0.05, GOOGL: -0.05})
    approved = _approved(ruling)
    assert sum(approved.values()) == pytest.approx(-0.10, abs=1e-6)
    assert all(w < 0 for w in approved.values())
    assert ruling.changed_by_kernel
    reasons = [g.reason for g in ruling.book_rulings if g.guard is GuardId.G3_SIZE]
    assert any("net cap" in r for r in reasons)


def test_a_hedged_book_is_not_touched_by_the_net_cap() -> None:
    ruling = _rule({NVDA: 0.05, GOOGL: -0.05, TSLA: 0.04, "AAPLUSDT": -0.04})
    assert not ruling.changed_by_kernel


def test_the_crypto_beta_cluster_holds_one_and_a_half_names() -> None:
    ruling = _rule({MSTR: 0.05, COIN: 0.05, NVDA: -0.05})
    approved = _approved(ruling)
    assert abs(approved[MSTR]) + abs(approved[COIN]) == pytest.approx(0.075, abs=1e-6)


def test_policy_v1_has_neither_cap() -> None:
    ruling = _rule({TSLA: -0.05, HOOD: -0.05, MSTR: -0.05, GOOGL: -0.05}, POLICY_V1)
    assert not ruling.changed_by_kernel
    assert "net_max" not in POLICY_V1.model_dump()
    assert "cluster_caps" not in POLICY_V1.model_dump()


def test_the_net_cap_keeps_what_is_held_and_scales_the_new() -> None:
    # 0.08 short held in A; B and C ask for 0.05 short each: the short side may be 0.10.
    ceilings, caps = allocate_caps(
        {"A": (0.08, 0.08, -1), "B": (0.05, 0.0, -1), "C": (0.05, 0.0, -1)}, POLICY_V2
    )
    assert ceilings["A"][0] == pytest.approx(0.08)
    assert ceilings["B"][0] == pytest.approx(0.01)
    assert ceilings["C"][0] == pytest.approx(0.01)
    assert caps
    assert caps[0].allocation.binds
