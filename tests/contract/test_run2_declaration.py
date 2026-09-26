"""Run 2's pre-registration: exactly what changed against run 1, and nothing else (run2.py).

Run 1's record must keep verifying with this code, so policy v1 and every 1.0.0 record serialise
byte for byte as they were written. Run 2's genesis declares its predecessor and every change, and
the declared figures are the ones the replay evidence in ``validation/run2/`` holds.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from helpers import T0
from sentiment_agent.clock import ManualClock
from sentiment_agent.decision.prompt import FACT_LEGEND, prompt_hashes
from sentiment_agent.ledger.genesis import GenesisError, build_genesis
from sentiment_agent.policy import ACTIVE_POLICY, POLICY_V1, POLICY_V2, RUN2_STEPS
from sentiment_agent.run2 import (
    ACT_RATE_EVIDENCE,
    DECLARED_CHANGES,
    OUTAGE_EVIDENCE,
    PREDECESSOR,
    REPLAY_EVIDENCE,
    RUN1_CODE_COMMIT,
    RUN1_GENESIS_HASH,
    RUN1_PROMPT_HASHES,
    RUN2_PROMPT_PY_HASH,
    declaration_for,
)
from sentiment_agent.types import (
    FUNDING_Z_CRYPTO_ONLY,
    AssetClass,
    DeclaredChange,
    Genesis,
    GuardId,
    RunMode,
    TriggerRule,
)

ROOT = Path(__file__).resolve().parents[2]
RUN1_POLICY_HASH = "55ff779b5d6f2c55082be1b4db6c9fde53865213cade58232f64f372ad73aeaf"
"""The ``policy_hash`` in run 1's genesis (public/ledger.jsonl seq 0, 2026-09-24 17:06 UTC)."""
COMMIT = "a" * 40
LOCKS = {"uv.lock": "b" * 64}


def evidence() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((ROOT / REPLAY_EVIDENCE).read_text(encoding="utf-8"))
    return loaded


def genesis_of(policy: Any, **extra: Any) -> Genesis:
    return build_genesis(
        policy=policy,
        prompt_hashes=prompt_hashes(),
        mode=RunMode.PAPER,
        code_commit=COMMIT,
        lock_hashes=LOCKS,
        bgc_package="@bitget-ai/bitget-agent-cli@3.0.0",
        clock=ManualClock(T0),
        **extra,
    )


# --- run 1 stays exactly as it was ---------------------------------------------------------------


def test_policy_v1_still_hashes_to_run_1s_genesis() -> None:
    assert POLICY_V1.content_hash() == RUN1_POLICY_HASH
    assert PREDECESSOR.policy_hash == RUN1_POLICY_HASH
    assert "funding_z_asset_classes" not in POLICY_V1.triggers.model_dump(mode="json")
    assert POLICY_V1.triggers.funding_z_asset_classes == FUNDING_Z_CRYPTO_ONLY


def test_a_genesis_without_a_predecessor_is_written_in_the_1_0_0_form() -> None:
    dumped = genesis_of(POLICY_V1).model_dump(mode="json")
    assert "predecessor" not in dumped
    assert "declared_changes" not in dumped
    assert Genesis.model_validate(dumped).model_dump(mode="json") == dumped


def test_the_prompts_are_run_1s_but_for_the_declared_legend_fix() -> None:
    now = prompt_hashes()
    assert set(now) == set(RUN1_PROMPT_HASHES)
    changed = sorted(k for k in now if now[k] != RUN1_PROMPT_HASHES[k])
    assert changed == ["decision/prompt.py"]
    assert now["decision/prompt.py"] == RUN2_PROMPT_PY_HASH
    fix = next(c for c in DECLARED_CHANGES if c.change_id == "run2-d4")
    assert fix.kind == "prompt_fix"
    assert fix.files == ("src/sentiment_agent/decision/prompt.py",)
    assert RUN1_PROMPT_HASHES["decision/prompt.py"] in fix.evidence["decision/prompt.py"]
    assert RUN2_PROMPT_PY_HASH in fix.evidence["decision/prompt.py"]


def test_the_legend_names_the_market_of_every_positioning_figure() -> None:
    """run2-d4: the figures Bitget's data services read from Binance are called Binance's."""
    legend = dict(FACT_LEGEND)
    for key in (
        "oi_change_1h_pct / oi_change_24h_pct",
        "retail_long_short_ratio",
        "top_trader_long_short_ratio",
        "taker_buy_sell_ratio",
    ):
        assert "Binance USD-M" in legend[key], key
    assert not any("Bitget's top traders" in text for text in legend.values())
    assert "Bitget" in legend["live_last"]


# --- policy v2 is v1 with six declared amendments ------------------------------------------------

AMENDED: dict[str, dict[str, set[str]]] = {
    "run2-a1": {"version": set(), "triggers": {"funding_z_asset_classes", "basis"}},
    "run2-a2": {"triggers": {"funding_abs_min", "basis"}},
    "run2-a3": {"net_max": set(), "cluster_caps": set(), "guard_bases": {"G3_SIZE"}},
    "run2-a4": {"scoring_window": set(), "guard_bases": {"G2_WEEKEND_FREEZE"}},
    "run2-a5": {
        "breaker": {"losing_streak_cooloff_hours", "basis"},
        "guard_bases": {"G10_BREAKER"},
    },
    "run2-a6": {"decision": {"outage_flatten_after", "basis"}, "guard_bases": {"G10_BREAKER"}},
}
"""What each amendment may touch: top-level policy keys, and inside them the keys (or, for
``guard_bases``, the guards) it changes. Anything else changing is an undeclared change."""


def _changed(before: dict[str, Any], after: dict[str, Any]) -> dict[str, set[str]]:
    diff: dict[str, set[str]] = {}
    for key in sorted(set(before) | set(after)):
        old, new = before.get(key), after.get(key)
        if old == new:
            continue
        if key == "guard_bases":
            olds = {g["guard"]: g for g in old or []}
            diff[key] = {GuardId(g["guard"]).name for g in new or [] if olds.get(g["guard"]) != g}
        elif isinstance(old, dict) and isinstance(new, dict):
            diff[key] = {k for k in set(old) | set(new) if old.get(k) != new.get(k)}
        else:
            diff[key] = set()
    return diff


def test_each_amendment_changes_only_what_it_declares() -> None:
    before = POLICY_V1
    assert [change_id for change_id, _ in RUN2_STEPS] == list(AMENDED)
    for change_id, after in RUN2_STEPS:
        diff = _changed(before.model_dump(mode="json"), after.model_dump(mode="json"))
        assert diff == AMENDED[change_id], change_id
        for part in ("triggers", "breaker", "decision"):
            if "basis" in diff.get(part, set()):
                old, new = getattr(before, part).basis, getattr(after, part).basis
                assert new.startswith(old), (change_id, part)
                assert change_id in new[len(old) :], (change_id, part)
        before = after
    assert before is POLICY_V2
    assert ACTIVE_POLICY is POLICY_V2


def test_policy_v2_is_the_declared_rules() -> None:
    assert POLICY_V2.triggers.funding_z_asset_classes == tuple(AssetClass)
    assert POLICY_V2.triggers.funding_abs_min == 0.00075
    assert POLICY_V2.net_max == 0.10
    ((cluster),) = POLICY_V2.cluster_caps
    assert (cluster.cap, set(cluster.symbols)) == (
        0.075,
        {"MSTRUSDT", "COINUSDT", "HOODUSDT", "CRCLUSDT"},
    )
    window = POLICY_V2.scoring_window
    assert window is not None
    assert (window.start.isoformat(), window.end.isoformat()) == (
        "2026-09-28T00:00:00+00:00",
        "2026-10-01T00:00:00+00:00",
    )
    assert POLICY_V2.breaker.losing_streak_cooloff_hours == 24
    assert POLICY_V2.decision.outage_flatten_after == 3
    # The rule a reader sees is the rule the kernel applies.
    rules = {g.guard: g.rule for g in POLICY_V2.guard_bases}
    assert "10% net" in rules[GuardId.G3_SIZE]
    assert "7.5%" in rules[GuardId.G3_SIZE]
    assert "2026-10-01 00:00 UTC" in rules[GuardId.G2_WEEKEND_FREEZE]
    assert "until 24h after the last loss" in rules[GuardId.G10_BREAKER]
    assert "three failed decisions in a row flatten" in rules[GuardId.G10_BREAKER]
    assert "a model outage flattens the book" not in rules[GuardId.G10_BREAKER]


def test_the_funding_scope_must_name_distinct_classes() -> None:
    base = POLICY_V1.triggers.model_dump()
    for bad in ((), (AssetClass.CRYPTO, AssetClass.CRYPTO)):
        with pytest.raises(ValueError, match="distinct asset classes"):
            TriggerRule.model_validate({**base, "funding_z_asset_classes": bad})


# --- the declaration -----------------------------------------------------------------------------


def test_run_2s_genesis_declares_its_predecessor_and_every_change() -> None:
    predecessor, changes = declaration_for(POLICY_V2)
    assert predecessor == PREDECESSOR
    assert predecessor.genesis_hash == RUN1_GENESIS_HASH
    assert predecessor.code_commit == RUN1_CODE_COMMIT
    genesis = genesis_of(POLICY_V2, predecessor=predecessor, declared_changes=changes)
    assert [c.change_id for c in genesis.declared_changes] == [
        "run2-d1",
        "run2-d2",
        "run2-d3",
        "run2-d4",
        *AMENDED,
    ]
    amendments = [c for c in genesis.declared_changes if c.kind == "policy_amendment"]
    assert [c.change_id for c in amendments] == list(AMENDED)
    hashes = [RUN1_POLICY_HASH, *(p.content_hash() for _, p in RUN2_STEPS)]
    for i, amendment in enumerate(amendments):
        assert (amendment.previous_policy_hash, amendment.new_policy_hash) == (
            hashes[i],
            hashes[i + 1],
        )
    assert amendments[-1].new_policy_hash == genesis.policy_hash == POLICY_V2.content_hash()
    dumped = genesis.model_dump(mode="json")
    assert dumped["predecessor"]["genesis_hash"] == RUN1_GENESIS_HASH
    assert Genesis.model_validate(dumped).model_dump(mode="json") == dumped
    assert declaration_for(POLICY_V1) == (None, ())


def test_a_changed_policy_without_a_declared_amendment_is_refused() -> None:
    code_only = tuple(c for c in DECLARED_CHANGES if c.kind != "policy_amendment")
    with pytest.raises(GenesisError, match="policy amendment is declared"):
        genesis_of(POLICY_V2, predecessor=PREDECESSOR, declared_changes=code_only)
    with pytest.raises(GenesisError, match="predecessor"):
        genesis_of(POLICY_V2, declared_changes=DECLARED_CHANGES)
    amendments = tuple(c for c in DECLARED_CHANGES if c.kind == "policy_amendment")
    wrong = amendments[-1].model_copy(update={"new_policy_hash": "c" * 64})
    with pytest.raises(GenesisError, match="install the policy"):
        genesis_of(
            POLICY_V2,
            predecessor=PREDECESSOR,
            declared_changes=(*code_only, *amendments[:-1], wrong),
        )
    with pytest.raises(GenesisError, match="must chain"):
        genesis_of(
            POLICY_V2,
            predecessor=PREDECESSOR,
            declared_changes=(*code_only, amendments[0], *amendments[2:]),
        )


def test_a_declared_change_is_well_formed() -> None:
    base: dict[str, Any] = {
        "change_id": "x",
        "title": "t",
        "detail": "d",
        "files": (),
    }
    with pytest.raises(ValueError, match="names both policy hashes"):
        DeclaredChange(**base, kind="policy_amendment")
    with pytest.raises(ValueError, match="names both policy hashes"):
        DeclaredChange(**base, kind="code_fix", new_policy_hash="d" * 64)
    with pytest.raises(ValueError, match="must change the policy"):
        DeclaredChange(
            **base, kind="policy_amendment", previous_policy_hash="d" * 64, new_policy_hash="d" * 64
        )


def test_code_changes_leave_the_policy_and_prompts_alone() -> None:
    for change in DECLARED_CHANGES:
        if change.kind in ("code_fix", "observability"):
            assert "policy, the prompt files and every guard limit are unchanged" in change.detail
            assert "src/sentiment_agent/policy.py" not in change.files
            assert not any(f.startswith("src/sentiment_agent/decision/") for f in change.files)
        for path in change.files:
            assert (ROOT / path).is_file(), path


# --- the declared figures are the evidence's -----------------------------------------------------


def test_the_evidence_is_run_1s_replay_under_both_policies() -> None:
    doc = evidence()
    assert doc["input"]["genesis_hash"] == RUN1_GENESIS_HASH
    assert doc["input"]["code_commit"] == RUN1_CODE_COMMIT
    assert doc["input"]["genesis_policy_hash"] == RUN1_POLICY_HASH
    assert doc["input"]["ledger_head_seq"] == 363
    assert doc["policy_v1"]["policy_hash"] == POLICY_V1.content_hash()
    assert doc["run2_a1"]["policy_hash"] == dict(RUN2_STEPS)["run2-a1"].content_hash()
    assert doc["policy_v2"]["policy_hash"] == POLICY_V2.content_hash()
    for name in ("policy_v1", "run2_a1", "policy_v2"):
        assert doc[name]["snapshots"] == 226
        # The replay reproduces the decisions run 1 actually took on heartbeats.
        assert doc[name]["heartbeat_decisions"] == doc["as_run"]["decisions"] == 2
    assert doc["policy_v1"]["event_decisions"] == 0


def _declared(change_id: str) -> DeclaredChange:
    return next(c for c in DECLARED_CHANGES if c.change_id == change_id)


def test_the_declared_counts_match_the_evidence() -> None:
    doc = evidence()
    a1 = doc["run2_a1"]
    assert a1["event_decisions"] == 15
    assert a1["event_decisions_within_worst_case_budget"] == 15
    assert a1["event_decisions_by_day"] == {"2026-09-24": 7, "2026-09-25": 8}
    assert a1["refused_by_kind_and_reason"]["funding_zscore:daily_cap"] == 3
    assert a1["hours"] == pytest.approx(19.86, abs=0.01)
    extremes = doc["funding_extremes"]["snapshots_beyond_by_symbol"]
    assert sum(extremes.values()) == 436
    assert "BTCUSDT" not in extremes
    amendment = _declared("run2-a1")
    quoted = amendment.evidence["funding_extremes"]
    basis = dict(RUN2_STEPS)["run2-a1"].triggers.basis
    for symbol, count in extremes.items():
        assert f"{symbol.removesuffix('USDT')} {count}" in quoted
        assert f"{symbol.removesuffix('USDT')} {count}" in basis
    assert "436" in quoted
    assert "436" in basis
    assert amendment.evidence["replayed_event_decisions"].startswith("15 in 19.9 hours")
    riders = _declared("run2-d1")
    assert doc["policy_v1"]["admitted_by_kind"]["coordinated_cluster"] == 22
    assert riders.evidence["run1_full_snapshot_triggers_never_evaluated"].startswith("22 ")


def test_the_level_floor_is_declared_with_run_1s_funding_profile() -> None:
    doc = evidence()
    level = doc["funding_level"]
    assert level["instrument_snapshots"] == 3150
    assert level["beyond_threshold"] == 436
    assert level["beyond_threshold_share"] == pytest.approx(0.1384)
    assert level["abs_rate_zero_share"] == pytest.approx(0.5114)
    assert level["abs_rate_median"] == 0.0
    assert level["abs_rate_p95"] == pytest.approx(0.000451)
    assert level["beyond_with_level_share_by_floor"] == {
        "0.0005": pytest.approx(0.0346),
        "0.00075": pytest.approx(0.0137),
        "0.0008": pytest.approx(0.0105),
    }
    with_level = doc["funding_extremes"]["snapshots_beyond_with_level_by_symbol"]
    assert doc["funding_extremes"]["level_floor"] == POLICY_V2.triggers.funding_abs_min
    assert with_level == {"MSTRUSDT": 29, "SNDKUSDT": 14}
    v2 = doc["policy_v2"]
    assert v2["event_decisions"] == v2["event_decisions_within_worst_case_budget"] == 3
    assert [t["at"][:16] for t in v2["event_decision_times"]] == [
        "2026-09-25T00:18",
        "2026-09-25T01:28",
        "2026-09-25T12:31",
    ]
    a2 = _declared("run2-a2")
    assert "beyond +-2 in 13.8%" in a2.evidence["level_profile"]
    assert "zero in 51.1%" in a2.evidence["level_profile"]
    assert "95th percentile 4.5 bp" in a2.evidence["level_profile"]
    assert "3.5%, 1.4%, 1.0%" in a2.evidence["level_profile"]
    assert "3,150" in a2.evidence["level_profile"]
    assert a2.evidence["funding_extremes_with_level"] == "43 of the 436 (MSTR 29, SNDK 14)"
    assert a2.evidence["replayed_event_decisions"].startswith("3 in 19.9 hours")
    assert "13.8%" in dict(RUN2_STEPS)["run2-a2"].triggers.basis


def test_the_book_caps_are_declared_with_the_agents_own_books() -> None:
    """run2-a3's figures are the act-rate replay's, which ran under run2-a1 exactly."""
    doc: dict[str, Any] = json.loads((ROOT / ACT_RATE_EVIDENCE).read_text(encoding="utf-8"))
    assert doc["input"]["policy_hash"] == dict(RUN2_STEPS)["run2-a1"].content_hash()
    assert doc["label"].startswith("SIMULATED")
    acted = [r for r in doc["decisions"] if r["stance"] == "act"]
    assert (len(acted), doc["event_decisions"]) == (14, 15)
    assert all(w < 0 for r in acted for w in r["proposed_weights"].values())
    assert not any(r.get("kernel_changed") for r in acted)
    nets = {r["at"][:16]: sum(r["proposed_weights"].values()) for r in acted}
    worst = min(nets, key=lambda at: nets[at])
    assert (worst, round(nets[worst], 4)) == ("2026-09-25T12:05", -0.2)
    cluster = {"MSTRUSDT", "COINUSDT", "HOODUSDT", "CRCLUSDT"}
    beta = {
        r["at"][:16]: sum(abs(w) for s, w in r["proposed_weights"].items() if s in cluster)
        for r in acted
    }
    assert max(beta.values()) == pytest.approx(0.15)
    assert beta["2026-09-25T00:13"] == pytest.approx(0.15)
    a3 = _declared("run2-a3")
    assert "acted in 14 of 15 decisions" in a3.evidence["one_sided_books"]
    assert "20% net short" in a3.evidence["one_sided_books"]
    assert "2026-09-25 12:05 UTC" in a3.evidence["one_sided_books"]
    assert "(15%), 2026-09-25 00:13" in a3.evidence["crypto_beta"]
    assert "14 of the 15" in _declared("run2-a2").evidence["agent_on_the_unfloored_triggers"]


def test_the_upstream_fallback_is_declared_with_run_1s_outage() -> None:
    """run2-d3's figures are the outage recomputed from run 1's ledger, not typed by hand."""
    doc: dict[str, Any] = json.loads((ROOT / OUTAGE_EVIDENCE).read_text(encoding="utf-8"))
    assert doc["input"]["genesis_hash"] == RUN1_GENESIS_HASH
    assert doc["input"]["ledger_head_seq"] == 546
    since = doc["since_first_data_mcp_failure"]
    assert since["from"].startswith("2026-09-25T08:33")
    assert since["calls"]["bitget_mcp_server"]["asked"] == 1356
    assert since["calls"]["bitget_mcp_server"]["answered"] == 144
    assert doc["whole_run"]["calls"]["bitget_signal_mcp"]["asked"] == 939
    assert doc["whole_run"]["calls"]["bitget_signal_mcp"]["answered"] == 0
    missing = since["snapshots"] - since["readings_present"]["crypto_fear_greed"]
    assert (missing, since["snapshots"]) == (146, 164)
    assert (
        since["snapshots"] - since["readings_present"]["BTCUSDT"]["retail_long_short_ratio"] == 146
    )
    fallback = next(c for c in DECLARED_CHANGES if c.change_id == "run2-d3")
    assert fallback.kind == "code_fix"
    assert "144 of 1356" in fallback.evidence["bitget_mcp_server"]
    assert "0 of 939" in fallback.evidence["bitget_signal"]
    assert "146 of 164" in fallback.evidence["readings_missing"]
