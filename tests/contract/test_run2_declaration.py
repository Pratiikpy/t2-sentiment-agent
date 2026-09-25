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
from sentiment_agent.decision.prompt import prompt_hashes
from sentiment_agent.ledger.genesis import GenesisError, build_genesis
from sentiment_agent.policy import ACTIVE_POLICY, POLICY_V1, POLICY_V2
from sentiment_agent.run2 import (
    DECLARED_CHANGES,
    OUTAGE_EVIDENCE,
    PREDECESSOR,
    REPLAY_EVIDENCE,
    RUN1_CODE_COMMIT,
    RUN1_GENESIS_HASH,
    RUN1_PROMPT_HASHES,
    declaration_for,
)
from sentiment_agent.types import (
    FUNDING_Z_CRYPTO_ONLY,
    AssetClass,
    DeclaredChange,
    Genesis,
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


def test_the_prompts_are_run_1s() -> None:
    assert prompt_hashes() == RUN1_PROMPT_HASHES


# --- policy v2 is v1 with one change -------------------------------------------------------------


def test_policy_v2_differs_from_v1_only_in_the_funding_scope() -> None:
    v1 = POLICY_V1.model_dump(mode="json")
    v2 = POLICY_V2.model_dump(mode="json")
    assert sorted(k for k in v1 if v1[k] != v2[k]) == ["triggers", "version"]
    t1, t2 = v1["triggers"], v2["triggers"]
    assert sorted(k for k in t2 if t1.get(k) != t2[k]) == ["basis", "funding_z_asset_classes"]
    assert t2["basis"].startswith(t1["basis"])
    assert POLICY_V2.triggers.funding_z_asset_classes == tuple(AssetClass)
    assert ACTIVE_POLICY is POLICY_V2


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
        "run2-a1",
    ]
    (amendment,) = [c for c in genesis.declared_changes if c.kind == "policy_amendment"]
    assert amendment.previous_policy_hash == RUN1_POLICY_HASH
    assert amendment.new_policy_hash == genesis.policy_hash == POLICY_V2.content_hash()
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
    wrong = DECLARED_CHANGES[-1].model_copy(update={"new_policy_hash": "c" * 64})
    with pytest.raises(GenesisError, match="install the policy"):
        genesis_of(POLICY_V2, predecessor=PREDECESSOR, declared_changes=(*code_only, wrong))


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
        if change.kind != "policy_amendment":
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
    assert doc["policy_v2"]["policy_hash"] == POLICY_V2.content_hash()
    for name in ("policy_v1", "policy_v2"):
        assert doc[name]["snapshots"] == 226
        # The replay reproduces the decisions run 1 actually took on heartbeats.
        assert doc[name]["heartbeat_decisions"] == doc["as_run"]["decisions"] == 2
    assert doc["policy_v1"]["event_decisions"] == 0


def test_the_declared_counts_match_the_evidence() -> None:
    doc = evidence()
    v2 = doc["policy_v2"]
    assert v2["event_decisions"] == 15
    assert v2["event_decisions_within_worst_case_budget"] == 15
    assert v2["event_decisions_by_day"] == {"2026-09-24": 7, "2026-09-25": 8}
    assert v2["refused_by_kind_and_reason"]["funding_zscore:daily_cap"] == 3
    extremes = doc["funding_extremes"]["snapshots_beyond_by_symbol"]
    assert sum(extremes.values()) == 436
    assert "BTCUSDT" not in extremes
    amendment = next(c for c in DECLARED_CHANGES if c.change_id == "run2-a1")
    quoted = amendment.evidence["funding_extremes"]
    for symbol, count in extremes.items():
        assert f"{symbol.removesuffix('USDT')} {count}" in quoted
        assert f"{symbol.removesuffix('USDT')} {count}" in POLICY_V2.triggers.basis
    assert "436" in quoted
    assert "436" in POLICY_V2.triggers.basis
    assert amendment.evidence["replayed_event_decisions"].startswith("15 in 19.9 hours")
    assert v2["hours"] == pytest.approx(19.86, abs=0.01)
    riders = next(c for c in DECLARED_CHANGES if c.change_id == "run2-d1")
    assert doc["policy_v1"]["admitted_by_kind"]["coordinated_cluster"] == 22
    assert riders.evidence["run1_full_snapshot_triggers_never_evaluated"].startswith("22 ")


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
