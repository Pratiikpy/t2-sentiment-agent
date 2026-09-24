"""Policy v1 against the evidence it cites.

Every number in ``policy.py`` that answers a measurement, and every figure a guard's basis quotes,
is read back here from ``validation/demo_venue/``. If a number is edited without the measurement,
or the evidence is re-run and moves, this fails and says which.
"""

import json
import math
import re
from pathlib import Path
from typing import Any

import pytest

from sentiment_agent.policy import EXCLUDED, EXPECTED_ENVELOPE, POLICY_V1, UNIVERSE
from sentiment_agent.types import AssetClass, GuardId

EVIDENCE = Path(__file__).resolve().parents[2] / "validation" / "demo_venue"


def _load(name: str) -> Any:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def _basis(guard: GuardId) -> str:
    return next(b.basis for b in POLICY_V1.guard_bases if b.guard is guard)


INTEGRITY = _load("demo_integrity.json")
TRACKING = _load("fade_demo.json")["tracking"]
PROBE = _load("universe_probe.json")["instruments"]
ENVELOPE = _load("envelope_clean.json")["gross25_daily"]
WEEKEND = _load("weekend_vol.json")


@pytest.mark.parametrize("entry", UNIVERSE, ids=lambda e: e.symbol)
def test_gap_threshold_is_the_measured_p99(entry: Any) -> None:
    if entry.asset_class is AssetClass.US_EQUITY:
        row = TRACKING[entry.symbol]
        assert entry.demo_live_gap_p99_bps == row["close_gap_p99_bps"]
        assert row["mark_index_over_3pct"] == 0
    else:
        row = INTEGRITY[entry.symbol]
        assert entry.demo_live_gap_p99_bps == row["gap_p99_bps"]
        assert row["over3pct_last30d"] == 0


def test_excluded_instruments_failed_the_integrity_check() -> None:
    quoted = {"ETHUSDT": 4, "SOLUSDT": 8, "DOGEUSDT": 11, "XRPUSDT": 252}
    assert set(EXCLUDED) == set(quoted)
    for symbol, excursions in quoted.items():
        assert INTEGRITY[symbol]["over3pct_last30d"] == excursions
    assert round(max(INTEGRITY[s]["gap_p99_bps"] for s in EXCLUDED)) == 19096


def test_universe_is_listed_online_on_demo_with_the_quoted_fees_and_limits() -> None:
    assert set(PROBE) == set(POLICY_V1.symbols)
    for entry in UNIVERSE:
        demo = PROBE[entry.symbol]["demo"]
        assert demo["code"] == "00000"
        assert demo["status"] == "online"
        assert demo["taker"] == "0.0006"
        expected_type = "crypto" if entry.asset_class is AssetClass.CRYPTO else "stock"
        assert demo["symbolType"] == expected_type
    nvda = PROBE["NVDAUSDT"]["demo"]
    assert (nvda["maxMkt"], nvda["minQty"]) == ("60", "0.01")
    assert "maxMarketOrderQty 60, minOrderQty 0.01" in _basis(GuardId.G11_ELIGIBILITY)


def _spread_bps(row: dict[str, str]) -> float:
    bid, ask = float(row["bid"]), float(row["ask"])
    return (ask - bid) / ((ask + bid) / 2) * 10_000


def test_spread_bound_sits_above_every_measured_demo_spread() -> None:
    spreads = [_spread_bps(PROBE[s]["demo"]) for s in POLICY_V1.symbols]
    assert max(spreads) < POLICY_V1.max_open_spread_bps
    quoted = f"{min(spreads):.1f}-{max(spreads):.1f} bps"
    assert quoted == "0.1-6.6 bps"
    assert quoted in _basis(GuardId.G8_TAKER_ONLY)


def test_fee_budget_arithmetic_matches_its_basis() -> None:
    taker = 0.0006
    round_trips = ENVELOPE["closed_trades_median"]
    per_window_bps = round_trips * POLICY_V1.per_name_max * 2 * taker * 10_000
    assert round(per_window_bps, 1) == 6.6
    assert "6.6 bps" in _basis(GuardId.G7_FEE_BUDGET)
    assert POLICY_V1.fee_budget_window_bps == pytest.approx(3 * per_window_bps, rel=0.05)


def test_breaker_trips_past_the_worst_no_edge_drawdown() -> None:
    worst_pct = ENVELOPE["mdd_pct"]["worst"]
    assert worst_pct == -2.48
    assert POLICY_V1.breaker.reduce_only_drawdown * 100 > abs(worst_pct)
    assert "-2.48%" in _basis(GuardId.G10_BREAKER)
    assert "-2.48%" in _basis(GuardId.G3_SIZE)


def test_expected_envelope_is_the_measured_one() -> None:
    e = ENVELOPE
    assert EXPECTED_ENVELOPE["return_on_equity_bps"] == (
        f"median {e['ret_bps']['median']}, p05 {e['ret_bps']['p05']}, p95 +{e['ret_bps']['p95']}"
    )
    assert EXPECTED_ENVELOPE["max_drawdown_pct"] == (
        f"median {e['mdd_pct']['median']:.2f}, p95 {e['mdd_pct']['p95_worst']:.2f}, "
        f"worst {e['mdd_pct']['worst']:.2f}"
    )
    assert EXPECTED_ENVELOPE["sharpe_ann"] == (
        f"median {e['sharpe_ann']['median']}, p05 {e['sharpe_ann']['p05']}, "
        f"p95 +{e['sharpe_ann']['p95']}"
    )
    assert EXPECTED_ENVELOPE["win_rate"] == (
        f"median {e['win_rate']['median']:.2f}, p05 {e['win_rate']['p05']:.2f}, "
        f"p95 {e['win_rate']['p95']:.2f}"
    )
    assert EXPECTED_ENVELOPE["closed_trades"] == f"median {int(e['closed_trades_median'])}"
    assert f"{e['windows']:,} windows" in EXPECTED_ENVELOPE["source"]
    assert POLICY_V1.expected_envelope == EXPECTED_ENVELOPE


def test_venue_integrity_basis_quotes_real_excursions() -> None:
    basis = _basis(GuardId.G1_VENUE_INTEGRITY)
    assert ["06-29 00", 65.6] in INTEGRITY["BTCUSDT"]["worst"]
    assert "65.6% from its index, 2026-06-29 00:00 UTC" in basis
    weekend = [v["weekend_absret_bps"]["demo_index"] for v in WEEKEND.values()]
    weekday = [v["weekday_absret_bps"]["demo_index"] for v in WEEKEND.values()]
    quoted = (
        f"{min(weekend):.1f}-{max(weekend):.1f} bps/h vs "
        f"{round(min(weekday))}-{round(max(weekday))} bps/h"
    )
    assert quoted in basis


def test_weekend_basis_quotes_real_weekend_moves() -> None:
    basis = _basis(GuardId.G2_WEEKEND_FREEZE)
    lines = (EVIDENCE / "fade_demo2.txt").read_text(encoding="utf-8").splitlines()
    rows = {line.split(" | ")[0]: line for line in lines if " | " in line}
    assert "2026-09-18 demo_last=5 mark=1 live=204 " in rows["MSTRUSDT"]
    assert "MSTR +204 bps live vs +5 bps Demo" in basis
    assert "2026-09-11 demo_last=80 mark=82 live=17 " in rows["AAPLUSDT"]
    assert "AAPL +80 bps Demo vs +17 bps live" in basis
    weekend_live = [v["weekend_absret_bps"]["live"] for v in WEEKEND.values()]
    assert f"{min(weekend_live):.1f}-{max(weekend_live):.1f} bps live" in basis


def test_sharpe_standard_error_note_holds() -> None:
    se_at_72 = math.sqrt(1 / 72) * math.sqrt(8760)
    assert round(se_at_72) == 11
    note = next(m.notes for m in POLICY_V1.metrics if m.name == "sharpe_se_ann")
    assert "about 11 at n = 72" in note


def test_every_evidence_file_a_basis_cites_exists() -> None:
    texts = [b.basis for b in POLICY_V1.guard_bases]
    texts += [u.basis for u in POLICY_V1.universe]
    texts += [POLICY_V1.weekend.basis, POLICY_V1.breaker.basis, EXPECTED_ENVELOPE["source"]]
    cited = {m for t in texts for m in re.findall(r"\b[a-z_0-9]+\.(?:json|txt)\b", t)}
    assert cited, "no evidence file is cited"
    missing = sorted(c for c in cited if not (EVIDENCE / c).is_file())
    assert missing == []
