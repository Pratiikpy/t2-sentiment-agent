"""A stop the Demo-live drift cannot reach, at the same loss per name (policy v2, run2-a7).

G4 places every stop on the Demo mark. The Demo price of HOOD, SNDK, MSTR, CRCL and SP500 departs
from live by 267 to 383 bps at the 99th percentile, so under policy v1 a 4% stop could be taken
out by the venue's own drift. Policy v2 sets each name's stop at the wider of 4% and twice its
measured gap, and shrinks its size cap in proportion."""

from decimal import Decimal

import pytest

from helpers import T0
from kernel.kbuild import UNIVERSE, book, breaker_state, decision, inputs
from kernel.kbuild import spec as kspec
from sentiment_agent.clock import ManualClock
from sentiment_agent.kernel.kernel import RiskKernel
from sentiment_agent.kernel.planner import stop_price
from sentiment_agent.policy import POLICY_V1, POLICY_V2, STOP_GAP_MULTIPLE
from sentiment_agent.types import Activation, GuardId, KernelRuling, Side

HOOD, NVDA = "HOODUSDT", "NVDAUSDT"


def _rule(proposed: dict[str, float]) -> KernelRuling:
    return RiskKernel(POLICY_V2, ManualClock(T0)).rule(
        proposed=proposed,
        book=book(at=T0),
        inputs=inputs(UNIVERSE, at=T0, snapshot_at=T0),
        context=decision(UNIVERSE),
        breaker=breaker_state(Activation.ACTIVE, (), at=T0),
    )


@pytest.mark.parametrize("entry", POLICY_V2.universe, ids=lambda e: e.symbol)
def test_every_name_loses_the_same_at_its_stop(entry: object) -> None:
    symbol = entry.symbol  # type: ignore[attr-defined]
    gap = entry.demo_live_gap_p99_bps / 10_000  # type: ignore[attr-defined]
    stop = POLICY_V2.stop_for(symbol)
    assert stop >= STOP_GAP_MULTIPLE * gap - 1e-12
    assert stop >= POLICY_V2.stop_loss_pct
    assert POLICY_V2.name_cap(symbol) * stop == pytest.approx(
        POLICY_V2.per_name_max * POLICY_V2.stop_loss_pct
    )


def test_the_widened_names_are_the_ones_whose_drift_reaches_a_4_percent_stop() -> None:
    widened = sorted(
        e.symbol
        for e in POLICY_V2.universe
        if POLICY_V2.stop_for(e.symbol) > POLICY_V2.stop_loss_pct
    )
    assert widened == ["CRCLUSDT", "HOODUSDT", "MSTRUSDT", "SNDKUSDT", "SP500USDT"]
    assert POLICY_V2.stop_for(HOOD) == pytest.approx(0.0766)
    assert POLICY_V2.name_cap(HOOD) == pytest.approx(0.05 * 0.04 / 0.0766)


def test_policy_v1_is_untouched() -> None:
    """Run 1's record is pre-registered under v1: no field, no stop and no cap may move."""
    assert POLICY_V1.stop_gap_multiple is None
    assert "stop_gap_multiple" not in POLICY_V1.model_dump()
    assert POLICY_V1.stop_for(HOOD) == POLICY_V1.stop_loss_pct
    assert POLICY_V1.name_cap(HOOD) == POLICY_V1.per_name_max


def test_the_placed_stop_uses_the_names_distance() -> None:
    spec = kspec(HOOD, price_step="0.01")
    long_stop = stop_price(Decimal("100"), Side.BUY, POLICY_V2, spec, HOOD)
    short_stop = stop_price(Decimal("100"), Side.SELL, POLICY_V2, spec, HOOD)
    assert long_stop == Decimal("92.34")
    assert short_stop == Decimal("107.66")
    # without a name, the policy's default distance, as before
    assert stop_price(Decimal("100"), Side.BUY, POLICY_V2, spec) == Decimal("96.00")


def test_g3_trims_a_full_target_in_a_wide_stop_name_to_its_cap() -> None:
    ruling = _rule({HOOD: 0.05, NVDA: 0.05})
    approved = {ir.symbol: ir.approved_weight for ir in ruling.instruments}
    assert approved[HOOD] == pytest.approx(POLICY_V2.name_cap(HOOD), rel=1e-6)
    assert approved[NVDA] == pytest.approx(0.05)
    hood = next(ir for ir in ruling.instruments if ir.symbol == HOOD)
    g3 = next(g for g in hood.rulings if g.guard is GuardId.G3_SIZE)
    assert "7.66% stop" in g3.reason
