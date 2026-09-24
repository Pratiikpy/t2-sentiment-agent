"""Contract invariants beyond the headline ones in ``test_contract.py``.

Each test names a way a record could contradict itself or the safety rules, and shows the contract
refuses to construct it.
"""

import copy
import json
import math
import os
import pickle
import platform
import socket
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from helpers import (
    T0,
    client_oid_for,
    empty_book,
    make_intent,
    make_quote,
    mint_for_test,
    unvalidated_copy,
)
from sentiment_agent.clock import ManualClock
from sentiment_agent.hashing import canonical_json, content_hash
from sentiment_agent.policy import POLICY_V1
from sentiment_agent.types import (
    EVENT_PAYLOADS,
    AccountSnapshot,
    Activation,
    Amendment,
    ApprovedOrder,
    BookState,
    BreakerRule,
    Category,
    CrowdReport,
    DecisionEvent,
    DecisionRecord,
    EnvironmentProof,
    EventKind,
    Genesis,
    GroundingReport,
    GuardBasis,
    GuardId,
    GuardRuling,
    GuardStatus,
    InstrumentRuling,
    InstrumentSpec,
    KernelInputs,
    KernelRuling,
    LlmCallRecord,
    LlmDecision,
    LlmOutcome,
    LlmUsage,
    MarketMood,
    OrderIntent,
    OrderPlan,
    OrderPurpose,
    PerceptionSnapshot,
    Policy,
    Position,
    PositioningFeatures,
    PriceSource,
    ProtectiveReason,
    Quote,
    RulingContext,
    RunMode,
    Side,
    Stance,
    TargetProposal,
    Thinking,
    TriggerRule,
    WeekendRule,
    client_oid_of,
)

NVDA = "NVDAUSDT"
BTC = "BTCUSDT"


# --- non-finite numbers ---------------------------------------------------------------------


def test_nan_cannot_slip_past_the_only_reduce_check() -> None:
    # NaN > 0 and NaN < 0 are both false: without the finite rule this short reference would
    # accept a NaN approval as "same side, not larger".
    with pytest.raises(ValidationError):
        InstrumentRuling(
            symbol=NVDA,
            current_weight=-0.04,
            proposed_weight=None,
            approved_weight=math.nan,
            binding_guard=GuardId.G3_SIZE,
            rulings=(),
        )


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_floats_are_refused_at_any_depth(bad: float) -> None:
    with pytest.raises(ValidationError):
        _features(funding_z_live=bad)
    with pytest.raises(ValidationError):
        _snapshot(facts={"NVDAUSDT.funding_z_live": bad})
    with pytest.raises(ValidationError):
        GuardRuling(
            guard=GuardId.G8_TAKER_ONLY,
            symbol=NVDA,
            status=GuardStatus.PASSED,
            reason="r",
            basis="b",
            inputs={"spread_bps": bad},
        )


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity", "sNaN"])
def test_non_finite_decimals_are_refused(bad: str) -> None:
    with pytest.raises(ValidationError):
        make_quote(last=bad)


# --- canonical hashing ----------------------------------------------------------------------


def test_bare_datetimes_hash_as_they_do_inside_a_model() -> None:
    quote = make_quote(at=T0 + timedelta(microseconds=123))
    dumped = quote.model_dump(mode="json")
    assert canonical_json({"ts": quote.ts}) == canonical_json({"ts": dumped["ts"]})
    assert canonical_json(T0) == b'"2026-09-23T13:00:00Z"'
    plus_two = T0.astimezone(timezone(timedelta(hours=2)))
    assert content_hash(plus_two) == content_hash(T0)
    assert canonical_json(date(2026, 9, 23)) == b'"2026-09-23"'


def test_hashing_refuses_what_has_no_canonical_form() -> None:
    with pytest.raises(ValueError, match="naive"):
        canonical_json({"at": datetime(2026, 9, 23, 13, 0)})  # noqa: DTZ001 - deliberately naive
    with pytest.raises(TypeError, match="sets"):
        canonical_json({"guards": frozenset({GuardId.G1_VENUE_INTEGRITY})})
    with pytest.raises(ValueError, match="same string"):
        canonical_json({1: "a", "1": "b"})
    with pytest.raises(ValueError, match="Out of range"):
        canonical_json([math.inf])
    assert canonical_json({GuardId.G3_SIZE: 1}) == b'{"G3_size":1}'


def test_model_hash_survives_the_ledger_round_trip() -> None:
    plan = _plan()
    line = plan.model_dump_json()
    revived = OrderPlan.model_validate(json.loads(line))
    assert revived == plan
    assert revived.content_hash() == plan.content_hash()
    assert content_hash(json.loads(line)) == plan.content_hash()


def test_model_copy_with_an_update_is_validated() -> None:
    kept = _inst(0.02, binding=GuardId.G3_SIZE)
    with pytest.raises(ValidationError, match="only reduce"):
        kept.model_copy(update={"approved_weight": 0.08})
    proof = _proof(passed=False, demo_read_code="40099")
    with pytest.raises(ValidationError, match="cannot pass"):
        proof.model_copy(update={"passed": True})
    assert kept.model_copy(update={"approved_weight": 0.01}).approved_weight == 0.01
    assert kept.model_copy() == kept
    assert kept.model_copy(deep=True) == kept
    # Tests that exercise a module's own defences build the refused record explicitly.
    forged = unvalidated_copy(kept, approved_weight=0.08)
    assert forged.approved_weight == 0.08
    with pytest.raises(ValidationError):
        InstrumentRuling.model_validate(dict(forged))


def test_validation_is_never_skipped_in_source() -> None:
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "src" / "sentiment_agent"
    offenders = [p for p in src.rglob("*.py") if "model_construct(" in p.read_text("utf-8")]
    assert offenders == []


# --- the approval capability -----------------------------------------------------------------


def test_approved_order_is_sealed() -> None:
    approved = mint_for_test(make_intent())
    other = make_intent(qty="2.00")
    with pytest.raises(AttributeError):
        approved._intent = other
    with pytest.raises(AttributeError):
        del approved._approval_hash
    with pytest.raises(TypeError):
        pickle.dumps(approved)
    with pytest.raises(TypeError):
        copy.copy(approved)
    with pytest.raises(TypeError):
        copy.deepcopy(approved)
    assert approved.verify()
    expected = content_hash({"intent": approved.intent, "ruling_id": approved.ruling_id})
    assert approved.approval_hash == expected


def test_approval_is_bound_to_its_ruling() -> None:
    intent = make_intent(ruling_id="ruling-a")
    with pytest.raises(ValueError, match="different ruling"):
        mint_for_test(intent, ruling_id="ruling-b")
    with pytest.raises(TypeError, match="wraps an OrderIntent"):
        mint_for_test(intent.model_dump(), ruling_id="ruling-a")  # type: ignore[arg-type]
    with pytest.raises(PermissionError):
        ApprovedOrder(intent, "ruling-a", _token=None)


# --- order intents and plans ------------------------------------------------------------------


def _intent_fields(**overrides: Any) -> dict[str, Any]:
    fields = make_intent().model_dump()
    fields.update(overrides)
    return fields


def test_client_oid_derives_from_the_intent_id() -> None:
    intent = make_intent()
    assert intent.client_oid == client_oid_of(intent.intent_id)
    with pytest.raises(ValidationError, match="first 30 hex"):
        OrderIntent.model_validate(_intent_fields(client_oid=client_oid_for("another intent")))


@pytest.mark.parametrize(
    ("side", "stop"),
    [(Side.BUY, "222.84"), (Side.BUY, "230"), (Side.SELL, "222.84"), (Side.SELL, "210")],
)
def test_opening_stop_must_sit_on_the_losing_side(side: Side, stop: str) -> None:
    with pytest.raises(ValidationError, match="stop"):
        OrderIntent.model_validate(_intent_fields(side=side, stop_loss_price=Decimal(stop)))


def test_opening_short_with_stop_above_is_accepted() -> None:
    intent = make_intent(side=Side.SELL, purpose=OrderPurpose.OPEN)
    assert intent.stop_loss_price is not None
    assert intent.stop_loss_price > intent.reference_price


def test_reducing_leg_carries_no_stop() -> None:
    close = make_intent(side=Side.SELL, purpose=OrderPurpose.CLOSE)
    with pytest.raises(ValidationError, match="no stop"):
        OrderIntent.model_validate({**close.model_dump(), "stop_loss_price": Decimal("200")})


def test_split_index_in_range_and_positive_stop() -> None:
    with pytest.raises(ValidationError, match="split_index"):
        OrderIntent.model_validate(_intent_fields(split_index=1, split_count=1))
    with pytest.raises(ValidationError):
        OrderIntent.model_validate(_intent_fields(stop_loss_price=Decimal("0")))


def _plan(*intents: OrderIntent) -> OrderPlan:
    return OrderPlan(
        plan_id="plan-1",
        ruling_id="ruling-test",
        created_at=T0,
        intents=intents or (make_intent(), make_intent(symbol=BTC, qty="0.0010", price="83169.7")),
        skipped=(),
    )


def test_plan_is_one_ruling_with_distinct_client_oids() -> None:
    _plan()
    with pytest.raises(ValidationError, match="plan's ruling"):
        _plan(make_intent(), make_intent(symbol=BTC, ruling_id="ruling-other"))
    with pytest.raises(ValidationError, match="share a clientOid"):
        _plan(make_intent(), make_intent())


# --- kernel rulings ---------------------------------------------------------------------------


def _guard(
    guard: GuardId,
    status: GuardStatus,
    *,
    symbol: str | None = NVDA,
    ceiling: float | None = None,
    exit_: bool = False,
) -> GuardRuling:
    return GuardRuling(
        guard=guard,
        symbol=symbol,
        status=status,
        ceiling_abs_weight=ceiling,
        forces_exit=exit_,
        reason="test",
        basis="test",
    )


def _inst(
    approved: float,
    *,
    proposed: float | None = 0.05,
    current: float = 0.0,
    binding: GuardId | None = None,
    rulings: tuple[GuardRuling, ...] = (),
    symbol: str = NVDA,
) -> InstrumentRuling:
    return InstrumentRuling(
        symbol=symbol,
        current_weight=current,
        proposed_weight=proposed,
        approved_weight=approved,
        binding_guard=binding,
        rulings=rulings,
    )


def _kernel(
    *instruments: InstrumentRuling,
    book: tuple[GuardRuling, ...] = (),
    applied: tuple[GuardId, ...] = tuple(GuardId),
    decision_id: str | None = "d",
    protective: ProtectiveReason | None = None,
) -> KernelRuling:
    return KernelRuling(
        ruling_id="r",
        at=T0,
        decision_id=decision_id,
        protective_reason=protective,
        activation_before=Activation.ACTIVE,
        activation_after=Activation.ACTIVE,
        book_rulings=book,
        instruments=instruments,
        guards_applied=applied,
    )


def test_a_change_names_its_guard() -> None:
    with pytest.raises(ValidationError, match="without naming a guard"):
        _inst(0.02)
    _inst(0.05)


def test_instrument_ruling_cannot_contradict_its_guards() -> None:
    g3 = _guard(GuardId.G3_SIZE, GuardStatus.FIRED, ceiling=0.02)
    _inst(0.02, binding=GuardId.G3_SIZE, rulings=(g3,))
    with pytest.raises(ValidationError, match="ceiling"):
        _inst(0.03, binding=GuardId.G3_SIZE, rulings=(g3,))
    g1 = _guard(GuardId.G1_VENUE_INTEGRITY, GuardStatus.FIRED, exit_=True)
    with pytest.raises(ValidationError, match="forces an exit"):
        _inst(0.01, binding=GuardId.G1_VENUE_INTEGRITY, rulings=(g1,))
    other = _guard(GuardId.G3_SIZE, GuardStatus.PASSED, symbol=BTC)
    with pytest.raises(ValidationError, match="ruling for"):
        _inst(0.05, rulings=(other,))


def test_only_a_fired_guard_forces_an_exit() -> None:
    with pytest.raises(ValidationError, match="FIRED"):
        _guard(GuardId.G5_DAILY_KILL, GuardStatus.PASSED, symbol=None, exit_=True)


def test_binding_guard_must_show_a_binding_ruling() -> None:
    passed = _guard(GuardId.G3_SIZE, GuardStatus.PASSED)
    inst = _inst(0.02, binding=GuardId.G3_SIZE, rulings=(passed,))
    with pytest.raises(ValidationError, match="no FIRED or NOT_EVALUATED"):
        _kernel(inst)
    missing_input = _guard(GuardId.G3_SIZE, GuardStatus.NOT_EVALUATED, ceiling=0.02)
    _kernel(_inst(0.02, binding=GuardId.G3_SIZE, rulings=(missing_input,)))


def test_book_level_rulings_bind_every_instrument() -> None:
    kill = _guard(GuardId.G5_DAILY_KILL, GuardStatus.FIRED, symbol=None, exit_=True)
    flat = _inst(0.0, proposed=None, current=0.03, binding=GuardId.G5_DAILY_KILL)
    _kernel(flat, book=(kill,), decision_id=None, protective=ProtectiveReason.DAILY_KILL)
    kept = _inst(-0.01, proposed=None, current=-0.03, binding=GuardId.G5_DAILY_KILL, symbol=BTC)
    with pytest.raises(ValidationError, match="forces an exit"):
        _kernel(flat, kept, book=(kill,), decision_id=None, protective=ProtectiveReason.DAILY_KILL)
    named = _guard(GuardId.G5_DAILY_KILL, GuardStatus.FIRED, symbol=NVDA, exit_=True)
    with pytest.raises(ValidationError, match="book-level"):
        _kernel(flat, book=(named,), decision_id=None, protective=ProtectiveReason.DAILY_KILL)


def test_kernel_ruling_structure() -> None:
    fired = _guard(GuardId.G3_SIZE, GuardStatus.FIRED, ceiling=0.02)
    inst = _inst(0.02, binding=GuardId.G3_SIZE, rulings=(fired,))
    with pytest.raises(ValidationError, match="no model proposal"):
        _kernel(inst, decision_id=None, protective=ProtectiveReason.BREAKER)
    with pytest.raises(ValidationError, match="more than once in a ruling"):
        _kernel(inst, inst)
    with pytest.raises(ValidationError, match="more than once in guards_applied"):
        _kernel(inst, applied=(GuardId.G3_SIZE, GuardId.G3_SIZE))
    with pytest.raises(ValidationError, match="not applied"):
        _kernel(inst, applied=(GuardId.G1_VENUE_INTEGRITY,))
    ruling = _kernel(inst, applied=(GuardId.G3_SIZE,))
    assert ruling.changed_by_kernel
    assert ruling.instrument(NVDA) == inst
    assert ruling.instrument(BTC) is None


def test_ruling_context_answers_one_source() -> None:
    RulingContext(decision_id="d", protective_reason=None)
    RulingContext(decision_id=None, protective_reason=ProtectiveReason.LLM_OUTAGE)
    with pytest.raises(ValidationError):
        RulingContext(decision_id=None, protective_reason=None)
    with pytest.raises(ValidationError):
        RulingContext(decision_id="d", protective_reason=ProtectiveReason.LLM_OUTAGE)


# --- decisions --------------------------------------------------------------------------------


def _target(symbol: str = NVDA, target: float = 0.5, **extra: Any) -> TargetProposal:
    return TargetProposal(
        symbol=symbol,
        target=target,
        thesis="Funding z of 2.4 against a crowd that is long",
        invalidation="Funding z back under 1",
        horizon_hours=24,
        crowd_belief="Retail is long",
        our_view="Fade the crowd",
        confidence=0.55,
        **extra,
    )


def _decision(
    stance: Stance, *targets: TargetProposal, reasons: tuple[str, ...] = ()
) -> LlmDecision:
    return LlmDecision(
        stance=stance,
        targets=targets,
        rejected_alternatives=(),
        mandate_response="deployed" if targets else "declined",
        flat_reasons=reasons,
        summary="s",
    )


def test_every_stance_is_an_answer_in_writing() -> None:
    _decision(Stance.ACT, _target())
    _decision(Stance.HOLD, _target())
    _decision(Stance.FLAT_WITH_REASONS, reasons=("no crowd extreme",))
    for stance in (Stance.ACT, Stance.HOLD):
        with pytest.raises(ValidationError, match="at least one symbol"):
            _decision(stance)
    with pytest.raises(ValidationError, match="written reason"):
        _decision(Stance.FLAT_WITH_REASONS, reasons=("   ",))


def test_declared_invalidation_needs_evidence() -> None:
    with pytest.raises(ValidationError, match="evidence"):
        _target(invalidation_triggered=True)
    with pytest.raises(ValidationError, match="evidence"):
        _target(invalidation_triggered=True, invalidation_evidence=" ")
    _target(invalidation_triggered=True, invalidation_evidence="Funding z fell to 0.4")


def _usage() -> LlmUsage:
    return LlmUsage(
        prompt_tokens=100,
        completion_tokens=50,
        reasoning_tokens=20,
        total_tokens=150,
        reported=True,
    )


def _call(outcome: LlmOutcome, attempts: int = 1) -> LlmCallRecord:
    return LlmCallRecord(
        model="qwen3.8-max",
        thinking=Thinking.LOW,
        prompt_version="prompt-v1",
        prompt_hash=content_hash("prompt"),
        request_blob=None,
        response_blobs=(),
        attempts=attempts,
        usage=_usage(),
        latency_ms=10,
        outcome=outcome,
    )


def _record(**overrides: Any) -> DecisionRecord:
    decision = _decision(Stance.ACT, _target(), _target(BTC, -0.5))
    fields: dict[str, Any] = {
        "decision_id": "d",
        "decided_at": T0,
        "trigger_ids": ("t",),
        "snapshot_id": "s",
        "book_before": empty_book(),
        "mandate": POLICY_V1.mandate,
        "policy_version": POLICY_V1.version,
        "call": _call(LlmOutcome.DECIDED),
        "outcome": LlmOutcome.DECIDED,
        "decision": decision,
        "grounding": {NVDA: GroundingReport(figures=())},
        "proposed_weights": {NVDA: 0.025, BTC: -0.025},
    }
    fields.update(overrides)
    return DecisionRecord.model_validate(fields)


def test_decision_record_is_self_consistent() -> None:
    record = _record()
    revived = DecisionRecord.model_validate(json.loads(record.model_dump_json()))
    assert (
        DecisionEvent(record=revived).content_hash() == DecisionEvent(record=record).content_hash()
    )
    with pytest.raises(ValidationError, match="call's outcome"):
        _record(call=_call(LlmOutcome.TIMEOUT))
    with pytest.raises(ValidationError, match="exactly the addressed"):
        _record(proposed_weights={NVDA: 0.025})
    with pytest.raises(ValidationError, match="did not address"):
        _record(grounding={"TSLAUSDT": GroundingReport(figures=())})
    with pytest.raises(ValidationError, match="no grounding"):
        _record(
            call=_call(LlmOutcome.INVALID_RESPONSE, attempts=3),
            outcome=LlmOutcome.INVALID_RESPONSE,
            decision=None,
            proposed_weights={},
        )
    with pytest.raises(ValidationError, match="at least one attempt"):
        _call(LlmOutcome.DECIDED, attempts=0)
    _call(LlmOutcome.BUDGET_EXHAUSTED, attempts=0)


# --- environment proof ------------------------------------------------------------------------


def _proof(**overrides: Any) -> EnvironmentProof:
    fields: dict[str, Any] = {
        "checked_at": T0,
        "mode": RunMode.PAPER,
        "credentials_file": ".secrets/demo.env",
        "key_declared_demo": True,
        "bgc_package": "@bitget-ai/bitget-agent-cli@3.0.0",
        "paptrading_header_confirmed": True,
        "demo_read_ok": True,
        "demo_read_code": "00000",
        "live_read_rejected": True,
        "live_read_code": "40037",
        "hold_mode": "one_way",
        "account": AccountSnapshot(at=T0, equity_usdt=None, available_usdt=None, blob=None),
        "passed": True,
        "reasons": (),
    }
    fields.update(overrides)
    return EnvironmentProof.model_validate(fields)


def test_environment_proof_passes_only_when_earned() -> None:
    assert _proof().passed
    assert _proof(credentials_file=".secrets\\demo.env").passed
    unearned: list[dict[str, Any]] = [
        {"mode": RunMode.DRYRUN},
        {"credentials_file": "C:/Users/someone/elsewhere/.secrets/demo.env"},
        {"credentials_file": ".secrets/live.env"},
        {"key_declared_demo": False},
        {"paptrading_header_confirmed": False},
        {"demo_read_ok": False},
        {"demo_read_code": "40099"},
        {"account": None},
        {"live_read_rejected": False},
    ]
    for change in unearned:
        with pytest.raises(ValidationError, match="cannot pass"):
            _proof(**change)
        assert not _proof(**change, passed=False).passed


# --- live/Demo separation and keyed maps -------------------------------------------------------


def _features(symbol: str = NVDA, **overrides: Any) -> PositioningFeatures:
    fields: dict[str, Any] = dict.fromkeys(PositioningFeatures.model_fields)
    fields.update(
        symbol=symbol,
        asset_class="us_equity",
        coordinated_cluster=False,
        social_mentions_24h=None,
        next_earnings_at=None,
    )
    fields.update(overrides)
    return PositioningFeatures.model_validate(fields)


def _snapshot(**overrides: Any) -> PerceptionSnapshot:
    fields: dict[str, Any] = {
        "snapshot_id": "",
        "taken_at": T0,
        "mode": RunMode.SIMULATED,
        "policy_version": POLICY_V1.version,
        "universe": POLICY_V1.symbols,
        "demo_quotes": {NVDA: make_quote()},
        "live_quotes": {NVDA: make_quote(source=PriceSource.LIVE)},
        "features": {NVDA: _features()},
        "mood": MarketMood(),
        "crowd": CrowdReport(
            items=0, withheld=0, distinct_stories=0, duplication_ratio=0.0, clusters=(), mentions={}
        ),
        "text": (),
        "calendar": (),
        "source_calls": (),
        "facts": {"NVDAUSDT.demo_last": 222.81},
    }
    fields.update(overrides)
    return PerceptionSnapshot.model_validate(fields)


def test_snapshot_never_mixes_demo_and_live() -> None:
    snap = _snapshot()
    assert snap.content_hash() == _snapshot().content_hash()
    with pytest.raises(ValidationError, match="live quote"):
        _snapshot(demo_quotes={NVDA: make_quote(source=PriceSource.LIVE)})
    with pytest.raises(ValidationError, match="demo quote"):
        _snapshot(live_quotes={NVDA: make_quote()})
    with pytest.raises(ValidationError, match="holds a record for"):
        _snapshot(demo_quotes={BTC: make_quote()})
    with pytest.raises(ValidationError, match="holds a record for"):
        _snapshot(features={BTC: _features()})
    with pytest.raises(ValidationError):
        _snapshot(facts={"x": math.nan})


def _spec(source: PriceSource = PriceSource.DEMO) -> InstrumentSpec:
    return InstrumentSpec(
        symbol=NVDA,
        category=Category.USDT_FUTURES,
        source=source,
        base_coin="NVDA",
        quote_coin="USDT",
        status="online",
        min_order_qty=Decimal("0.01"),
        qty_step=Decimal("0.01"),
        price_step=Decimal("0.01"),
        min_order_amount=Decimal("5"),
        max_market_order_qty=Decimal("60"),
        max_order_qty=Decimal("240"),
        taker_fee_rate=Decimal("0.0006"),
        maker_fee_rate=Decimal("0.0002"),
        max_leverage=25,
        fund_interval_hours=None,
        fetched_at=T0,
    )


def _inputs(**overrides: Any) -> KernelInputs:
    fields: dict[str, Any] = {
        "at": T0,
        "demo_quotes": {NVDA: make_quote()},
        "live_quotes": {NVDA: make_quote(source=PriceSource.LIVE)},
        "specs": {NVDA: _spec()},
        "demo_index_move_bps_3h": {NVDA: 20.0},
        "snapshot_id": "s",
        "snapshot_taken_at": T0,
    }
    fields.update(overrides)
    return KernelInputs.model_validate(fields)


def test_kernel_inputs_are_keyed_separated_and_demo_limited() -> None:
    _inputs()
    with pytest.raises(ValidationError, match="live quote"):
        _inputs(demo_quotes={NVDA: make_quote(source=PriceSource.LIVE)})
    with pytest.raises(ValidationError, match="Demo venue"):
        _inputs(specs={NVDA: _spec(PriceSource.LIVE)})
    with pytest.raises(ValidationError, match="holds a record for"):
        _inputs(specs={BTC: _spec()})


def test_book_positions_are_keyed_by_their_symbol() -> None:
    pos = Position(
        symbol=NVDA,
        qty=Decimal("1"),
        avg_entry=Decimal("222.84"),
        opened_at=T0,
        last_increase_at=T0,
        realized_pnl=Decimal("0"),
        fees_paid=Decimal("0.13"),
        stop_price=Decimal("213.93"),
        stop_venue_id=None,
        last_decision_id="d",
    )
    base = empty_book().model_dump()
    book = BookState.model_validate(
        {**base, "positions": {NVDA: pos}, "marks": {NVDA: Decimal("222.84")}}
    )
    assert book.weight(NVDA) == pytest.approx(222.84 / 10_000)
    with pytest.raises(ValidationError, match="holds a record for"):
        BookState.model_validate({**base, "positions": {BTC: pos}})


# --- genesis, amendments and policy coherence --------------------------------------------------


def _genesis(**overrides: Any) -> Genesis:
    fields: dict[str, Any] = {
        "project": "t2-sentiment-agent",
        "contract_version": "1.0.0",
        "created_at": T0,
        "mode": RunMode.PAPER,
        "policy": POLICY_V1,
        "policy_hash": POLICY_V1.content_hash(),
        "prompt_hashes": {"system_v1.md": content_hash("system")},
        "universe": POLICY_V1.symbols,
        "metric_definitions": POLICY_V1.metrics,
        "code_commit": "0" * 40,
        "dependency_lock_hashes": {},
        "qwen_model": "qwen3.8-max",
        "bgc_package": "@bitget-ai/bitget-agent-cli@3.0.0",
        "expected_envelope": POLICY_V1.expected_envelope,
        "statement": "pre-registration",
    }
    fields.update(overrides)
    return Genesis.model_validate(fields)


def test_genesis_pre_registers_the_policy_it_carries() -> None:
    genesis = _genesis()
    revived = Genesis.model_validate(json.loads(genesis.model_dump_json()))
    assert revived.content_hash() == genesis.content_hash()
    with pytest.raises(ValidationError, match="policy_hash"):
        _genesis(policy_hash=content_hash("another policy"))
    with pytest.raises(ValidationError, match="universe"):
        _genesis(universe=(BTC,))


def test_amendment_changes_the_policy_and_names_its_hash() -> None:
    tighter = POLICY_V1.model_copy(update={"version": "policy-v1.1", "max_open_spread_bps": 15.0})
    Amendment(
        amendment_id="a1",
        at=T0,
        reason="spread bound",
        previous_policy_hash=POLICY_V1.content_hash(),
        new_policy_hash=tighter.content_hash(),
        new_policy=tighter,
        owner_confirmed=True,
    )
    with pytest.raises(ValidationError, match="must change"):
        Amendment(
            amendment_id="a1",
            at=T0,
            reason="no-op",
            previous_policy_hash=POLICY_V1.content_hash(),
            new_policy_hash=POLICY_V1.content_hash(),
            new_policy=POLICY_V1,
            owner_confirmed=True,
        )
    with pytest.raises(ValidationError, match="hash of new_policy"):
        Amendment(
            amendment_id="a1",
            at=T0,
            reason="wrong hash",
            previous_policy_hash=POLICY_V1.content_hash(),
            new_policy_hash=content_hash("x"),
            new_policy=tighter,
            owner_confirmed=True,
        )


def _policy(**overrides: Any) -> Policy:
    return Policy.model_validate({**POLICY_V1.model_dump(), **overrides})


def test_policy_coherence_rules() -> None:
    _policy()
    breaker = POLICY_V1.breaker.model_dump()
    with pytest.raises(ValidationError, match="before halt"):
        BreakerRule.model_validate({**breaker, "reduce_only_drawdown": 0.05})
    triggers = POLICY_V1.triggers.model_dump()
    with pytest.raises(ValidationError, match="fear band"):
        TriggerRule.model_validate({**triggers, "fear_greed_low": 80})
    with pytest.raises(ValidationError, match="HH:MM"):
        TriggerRule.model_validate({**triggers, "us_open_local": "9:30"})
    with pytest.raises(ValidationError, match="distinct UTC hours"):
        TriggerRule.model_validate({**triggers, "funding_heartbeat_hours_utc": (0, 8, 8)})
    weekend = POLICY_V1.weekend.model_dump()
    with pytest.raises(ValidationError, match="different times"):
        WeekendRule.model_validate({**weekend, "reopen_weekday": 4, "reopen_hour": 20})
    with pytest.raises(ValidationError, match="pre-flattening"):
        WeekendRule.model_validate({**weekend, "preflatten_minutes": 180})
    bases = [*POLICY_V1.guard_bases, GuardBasis(guard=GuardId.G3_SIZE, rule="r", basis="b")]
    with pytest.raises(ValidationError, match="exactly one"):
        _policy(guard_bases=bases)
    with pytest.raises(ValidationError, match="horizon"):
        _policy(mandate={**POLICY_V1.mandate.model_dump(), "min_horizon_hours": 12})
    with pytest.raises(ValidationError, match="defined twice"):
        _policy(metrics=(*POLICY_V1.metrics, POLICY_V1.metrics[0]))


def test_payload_registry_is_read_only() -> None:
    with pytest.raises(TypeError):
        EVENT_PAYLOADS[EventKind.NOTE] = Quote  # type: ignore[index]
    assert EVENT_PAYLOADS[EventKind.DECISION] is DecisionEvent


# --- clock ------------------------------------------------------------------------------------


def test_manual_clock_is_utc_and_monotonic() -> None:
    clock = ManualClock(T0)
    assert clock.advance(timedelta(minutes=5)) == T0 + timedelta(minutes=5)
    clock.set(T0 + timedelta(hours=1))
    assert clock.now().tzinfo is UTC
    with pytest.raises(ValueError, match="backwards"):
        clock.set(T0)
    with pytest.raises(ValueError, match="backwards"):
        clock.advance(timedelta(seconds=-1))
    with pytest.raises(ValueError, match="UTC"):
        ManualClock(datetime(2026, 9, 23, 13, 0))  # noqa: DTZ001 - deliberately naive
    with pytest.raises(ValueError, match="UTC"):
        ManualClock(T0.astimezone(timezone(timedelta(hours=-4))))


# --- the test harness itself ------------------------------------------------------------------


def test_connect_ex_is_blocked_too() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock, pytest.raises(RuntimeError):
        sock.connect_ex(("api.bitget.com", 443))


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        (["node", "lib/index.js", "--paper-trading", "account_overview"], {}),
        (["curl", "https://api.bitget.com"], {}),
        ("bgc --version", {"shell": True}),
        ([sys.executable, "-c", "pass"], {"shell": True}),
    ],
)
def test_child_processes_other_than_python_are_blocked(args: Any, kwargs: dict[str, Any]) -> None:
    with pytest.raises(RuntimeError, match="forbidden in tests"):
        subprocess.run(args, check=False, **kwargs)  # noqa: S603 - refused before it runs


def test_os_system_is_blocked() -> None:
    with pytest.raises(RuntimeError, match="forbidden in tests"):
        os.system("echo hi")  # noqa: S605, S607 - refused before it runs


def test_platform_queries_still_work() -> None:
    # platform asks Windows for its version through the shell command "ver"; that one is allowed.
    platform._uname_cache = None  # type: ignore[attr-defined]
    assert platform.platform()
    assert platform.system()


def test_python_child_process_is_allowed() -> None:
    out = subprocess.run(
        [sys.executable, "-c", "print(6 * 7)"], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "42"
