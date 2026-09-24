"""The contract's own invariants. Every other module relies on these holding."""

import os
import re
import socket
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from helpers import T0, empty_book, make_intent, mint_for_test
from sentiment_agent.hashing import ZERO_HASH, canonical_json, content_hash
from sentiment_agent.policy import POLICY_V1
from sentiment_agent.types import (
    EVENT_PAYLOADS,
    PLAN_GUARDS,
    Activation,
    ApprovedOrder,
    AssetClass,
    DecisionRecord,
    EnvironmentProof,
    EventKind,
    GuardId,
    InstrumentRuling,
    KernelRuling,
    LedgerEvent,
    LlmCallRecord,
    LlmDecision,
    LlmOutcome,
    LlmUsage,
    OrderPurpose,
    ProtectiveReason,
    RunMode,
    Side,
    Stance,
    TargetProposal,
    Thinking,
    Trigger,
    TriggerKind,
    parse_payload,
)

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "sentiment_agent"


# --- policy ---------------------------------------------------------------------------------


def test_policy_universe_is_the_fourteen_measured_instruments() -> None:
    assert len(POLICY_V1.universe) == 14
    assert set(POLICY_V1.symbols) >= {"BTCUSDT", "SP500USDT", "NDX100USDT", "NVDAUSDT"}
    assert not set(POLICY_V1.symbols) & set(POLICY_V1.excluded)
    crypto = [u for u in POLICY_V1.universe if u.asset_class is AssetClass.CRYPTO]
    assert [u.symbol for u in crypto] == ["BTCUSDT"]


def test_every_guard_has_a_written_basis() -> None:
    assert {b.guard for b in POLICY_V1.guard_bases} == set(GuardId)
    assert all(len(b.basis) > 40 for b in POLICY_V1.guard_bases)
    assert len(PLAN_GUARDS) == 8


def test_policy_hash_is_stable() -> None:
    assert POLICY_V1.content_hash() == POLICY_V1.content_hash()
    rebuilt = type(POLICY_V1).model_validate(POLICY_V1.model_dump(mode="json"))
    assert rebuilt.content_hash() == POLICY_V1.content_hash()


# --- hashing --------------------------------------------------------------------------------


def test_canonical_json_is_order_independent_and_exact() -> None:
    assert canonical_json({"b": 1, "a": Decimal("0.10")}) == b'{"a":"0.10","b":1}'
    assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})
    with pytest.raises(ValueError, match="Out of range"):
        canonical_json({"x": float("nan")})
    assert len(ZERO_HASH) == 64


def test_naive_datetimes_are_refused() -> None:
    with pytest.raises(ValidationError):
        Trigger(
            trigger_id="t",
            kind=TriggerKind.OWNER_MANUAL,
            fired_at=datetime(2026, 9, 23, 13, 0),  # noqa: DTZ001 - deliberately naive
            symbols=(),
            detail="x",
            source="test",
        )


# --- kernel invariants ----------------------------------------------------------------------


def _ruling(current: float, proposed: float | None, approved: float) -> InstrumentRuling:
    # A binding guard is named so that a refusal here can only come from the only-reduce check.
    return InstrumentRuling(
        symbol="NVDAUSDT",
        current_weight=current,
        proposed_weight=proposed,
        approved_weight=approved,
        binding_guard=GuardId.G3_SIZE,
        rulings=(),
    )


@pytest.mark.parametrize(
    ("current", "proposed", "approved"),
    [
        (0.0, 0.05, 0.05),
        (0.0, 0.05, 0.02),
        (0.03, 0.05, 0.03),
        (0.03, -0.05, 0.0),
        (0.03, 0.01, 0.01),
        (0.03, None, 0.0),
        (-0.04, None, -0.02),
    ],
)
def test_only_reduce_accepts_shrinking(
    current: float, proposed: float | None, approved: float
) -> None:
    _ruling(current, proposed, approved)


@pytest.mark.parametrize(
    ("current", "proposed", "approved"),
    [
        (0.0, 0.02, 0.05),
        (0.0, 0.05, -0.01),
        (0.03, 0.0, 0.03),
        (0.03, None, 0.04),
        (-0.04, None, 0.01),
    ],
)
def test_only_reduce_refuses_adding_or_turning(
    current: float, proposed: float | None, approved: float
) -> None:
    with pytest.raises(ValidationError):
        _ruling(current, proposed, approved)


def test_ruling_answers_a_decision_or_a_protective_reason_not_both() -> None:
    base: dict[str, object] = {
        "ruling_id": "r",
        "at": T0,
        "activation_before": Activation.ACTIVE,
        "activation_after": Activation.ACTIVE,
        "book_rulings": (),
        "instruments": (),
        "guards_applied": (),
    }
    KernelRuling.model_validate({**base, "decision_id": "d", "protective_reason": None})
    KernelRuling.model_validate(
        {**base, "decision_id": None, "protective_reason": ProtectiveReason.DAILY_KILL}
    )
    with pytest.raises(ValidationError):
        KernelRuling.model_validate(
            {**base, "decision_id": "d", "protective_reason": ProtectiveReason.DAILY_KILL}
        )
    with pytest.raises(ValidationError):
        KernelRuling.model_validate({**base, "decision_id": None, "protective_reason": None})


# --- orders and approval --------------------------------------------------------------------


def test_intent_coherence() -> None:
    make_intent(purpose=OrderPurpose.OPEN)
    make_intent(purpose=OrderPurpose.CLOSE, side=Side.SELL)
    good = make_intent(purpose=OrderPurpose.OPEN)
    with pytest.raises(ValidationError):
        type(good).model_validate({**good.model_dump(), "stop_loss_price": None})
    with pytest.raises(ValidationError):
        type(good).model_validate({**good.model_dump(), "reduce_only": True})
    with pytest.raises(ValidationError):
        type(good).model_validate({**good.model_dump(), "client_oid": "random-uuid"})


def test_approved_order_cannot_be_forged() -> None:
    intent = make_intent()
    with pytest.raises(PermissionError):
        ApprovedOrder(intent, intent.ruling_id, _token=object())
    approved = mint_for_test(intent)
    assert approved.verify()
    assert approved.intent.client_oid.startswith("sa")


_SCAN_SKIP = frozenset(
    {"tests", ".git", ".venv", "node_modules", "__pycache__", "var", "cache", ".secrets"}
    | {".mypy_cache", ".ruff_cache", ".pytest_cache"}
)


def _project_files(*suffixes: str) -> list[Path]:
    """Every project file outside tests/, caches, dependencies, secrets and working state."""
    found: list[Path] = []
    for folder, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in _SCAN_SKIP]
        found += [Path(folder) / f for f in files if Path(f).suffix in suffixes]
    return found


def test_mint_token_is_referenced_only_by_the_kernel_minter() -> None:
    # "_MINT" rather than the full name, so a split string does not slip past the scan.
    allowed = {SRC / "types.py", SRC / "kernel" / "approval.py"}
    scanned = _project_files(".py")
    assert SRC / "types.py" in scanned
    offenders = [p for p in scanned if p not in allowed and "_MINT" in p.read_text("utf-8")]
    assert offenders == []


# --- decisions ------------------------------------------------------------------------------


def _target(symbol: str = "NVDAUSDT", target: float = 0.5) -> TargetProposal:
    return TargetProposal(
        symbol=symbol,
        target=target,
        thesis="Funding at 0.01% is below its 90-settlement mean",
        invalidation="Close above the prior high",
        horizon_hours=24,
        crowd_belief="Retail is long",
        our_view="Fade the crowd",
        confidence=0.55,
    )


def test_flat_with_reasons_needs_reasons_and_zero_targets() -> None:
    with pytest.raises(ValidationError):
        LlmDecision(
            stance=Stance.FLAT_WITH_REASONS,
            targets=(),
            rejected_alternatives=(),
            mandate_response="declined",
            summary="flat",
        )
    with pytest.raises(ValidationError):
        LlmDecision(
            stance=Stance.FLAT_WITH_REASONS,
            targets=(_target(),),
            rejected_alternatives=(),
            mandate_response="declined",
            flat_reasons=("no edge",),
            summary="flat",
        )
    with pytest.raises(ValidationError):
        LlmDecision(
            stance=Stance.ACT,
            targets=(_target(), _target()),
            rejected_alternatives=(),
            mandate_response="deployed",
            summary="dup",
        )


def test_decision_record_outcome_consistency() -> None:
    usage = LlmUsage(
        prompt_tokens=0, completion_tokens=0, reasoning_tokens=0, total_tokens=0, reported=False
    )
    call = LlmCallRecord(
        model="qwen3.8-max",
        thinking=Thinking.LOW,
        prompt_version="v1",
        prompt_hash=content_hash("p"),
        request_blob=None,
        response_blobs=(),
        attempts=3,
        usage=usage,
        latency_ms=0,
        outcome=LlmOutcome.TIMEOUT,
    )
    common: dict[str, object] = {
        "decision_id": "d",
        "decided_at": T0,
        "trigger_ids": ("t",),
        "snapshot_id": "s",
        "book_before": empty_book(),
        "mandate": POLICY_V1.mandate,
        "policy_version": POLICY_V1.version,
        "call": call,
        "grounding": {},
    }
    DecisionRecord.model_validate(
        {**common, "outcome": LlmOutcome.TIMEOUT, "decision": None, "proposed_weights": {}}
    )
    with pytest.raises(ValidationError):
        DecisionRecord.model_validate(
            {
                **common,
                "outcome": LlmOutcome.TIMEOUT,
                "decision": None,
                "proposed_weights": {"X": 0.01},
            }
        )


def test_environment_proof_cannot_pass_unearned() -> None:
    with pytest.raises(ValidationError):
        EnvironmentProof(
            checked_at=T0,
            mode=RunMode.PAPER,
            credentials_file=".secrets/demo.env",
            key_declared_demo=True,
            bgc_package="@bitget-ai/bitget-agent-cli@3.0.0",
            paptrading_header_confirmed=True,
            demo_read_ok=False,
            demo_read_code="40099",
            live_read_rejected=True,
            live_read_code="40006",
            hold_mode=None,
            account=None,
            passed=True,
            reasons=(),
        )


# --- ledger payload registry ----------------------------------------------------------------


def test_every_event_kind_has_a_payload_model() -> None:
    assert set(EVENT_PAYLOADS) == set(EventKind)


def test_payload_round_trip() -> None:
    trig = Trigger(
        trigger_id="t1",
        kind=TriggerKind.HEARTBEAT_US_OPEN,
        fired_at=T0 + timedelta(minutes=30),
        symbols=("NVDAUSDT",),
        detail="US open",
        source="schedule",
    )
    event = LedgerEvent(
        seq=1,
        ts=T0,
        kind=EventKind.TRIGGER,
        mode=RunMode.SIMULATED,
        payload=trig.model_dump(mode="json"),
        prev_hash=ZERO_HASH,
        hash=content_hash("x"),
    )
    assert parse_payload(event) == trig


# --- test-harness guarantees ----------------------------------------------------------------


def test_network_is_blocked() -> None:
    with pytest.raises(RuntimeError, match="network access is forbidden"):
        socket.create_connection(("api.bitget.com", 443), timeout=1)


def test_no_bitget_credentials_are_visible() -> None:
    assert not [k for k in os.environ if k.startswith("BITGET_")]


def test_live_key_file_is_never_referenced_in_source() -> None:
    # The owner's live key is a bitget.env under the parent workspace's .secrets/. No file of this
    # project (code, scripts, docs, config, evidence) may name it, with either path separator.
    pattern = re.compile(r"\.secrets[\\/]+bitget\.env", re.IGNORECASE)
    scanned = _project_files(".py", ".md", ".toml", ".json", ".txt", ".cfg", ".ini", ".yaml")
    assert SRC / "types.py" in scanned
    offenders = [p for p in scanned if pattern.search(p.read_text("utf-8", errors="replace"))]
    assert offenders == []
