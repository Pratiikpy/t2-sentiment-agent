"""RiskKernel.rule and RiskKernel.protective: minimum-of-ceilings, reductions, calendar, kill,
outage, and the record each ruling leaves."""

import math
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta

import pytest

from helpers import T0
from kernel.kbuild import (
    BTC,
    MSTR,
    NVDA,
    SPX,
    TSLA,
    UNGROUNDED,
    UNIVERSE,
    book,
    breaker_state,
    decision,
    demo_quote,
    inputs,
    kernel,
    position,
    protective_context,
    qty_for_weight,
)
from sentiment_agent.kernel.kernel import (
    PROTECTIVE_GUARDS,
    KernelError,
    ruling_hash,
)
from sentiment_agent.types import (
    ALL_GUARDS,
    VENUE_GUARDS,
    Activation,
    BookState,
    GroundingReport,
    GuardId,
    GuardStatus,
    InstrumentRuling,
    KernelInputs,
    KernelRuling,
    ProtectiveReason,
)

FRIDAY = datetime(2026, 9, 25, tzinfo=UTC)
SATURDAY = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)


def _inst(ruling: KernelRuling, symbol: str) -> InstrumentRuling:
    found = ruling.instrument(symbol)
    assert found is not None, symbol
    return found


def _status(ir: InstrumentRuling, guard: GuardId) -> GuardStatus:
    return next(g.status for g in ir.rulings if g.guard is guard)


def _rule(
    proposed: dict[str, float] | None,
    *,
    b: BookState | None = None,
    inp: KernelInputs | None = None,
    at: datetime = T0,
    activation: Activation = Activation.ACTIVE,
    trips: tuple[str, ...] = (),
    guards: frozenset[GuardId] = ALL_GUARDS,
    grounding: Mapping[str, GroundingReport] | None = None,
) -> KernelRuling:
    k, _ = kernel(at)
    return k.rule(
        proposed=proposed,
        book=b if b is not None else book(at=at),
        inputs=inp if inp is not None else inputs(UNIVERSE, at=at, snapshot_at=at),
        context=decision(UNIVERSE, grounding=grounding),
        breaker=breaker_state(activation, trips, at=at),
        guards=guards,
    )


# --- within every limit -------------------------------------------------------------------------


def test_a_proposal_within_every_limit_is_approved_as_asked() -> None:
    ruling = _rule({NVDA: 0.04, BTC: -0.03, SPX: 0.02})
    for ir in ruling.instruments:
        assert ir.approved_weight == ir.proposed_weight
        assert ir.binding_guard is None
        assert all(g.status is not GuardStatus.FIRED for g in ir.rulings)
    assert ruling.guards_applied == tuple(GuardId)
    assert ruling.decision_id == "d1"
    assert ruling.protective_reason is None
    assert not ruling.changed_by_kernel
    assert ruling.activation_before is ruling.activation_after is Activation.ACTIVE
    assert {g.guard for g in ruling.book_rulings} == {
        GuardId.G3_SIZE,
        GuardId.G5_DAILY_KILL,
        GuardId.G7_FEE_BUDGET,
        GuardId.G10_BREAKER,
    }


def test_every_applied_guard_is_evaluated_on_every_instrument() -> None:
    ruling = _rule({NVDA: 0.04, BTC: 0.0})
    for ir in ruling.instruments:
        assert [g.guard for g in ir.rulings] == list(GuardId)


# --- minimum of ceilings ------------------------------------------------------------------------


def test_the_tightest_ceiling_binds_and_the_others_are_named() -> None:
    held = position(NVDA, qty_for_weight("0.02", NVDA))
    b = book(positions=[held], fees_today="8", fees_total="8")  # G7: 8 bps >= 7 bps today
    wide = demo_quote(NVDA, bid="222.0", ask="223.0")  # G8: ~45 bps spread
    inp = inputs(demo={s: demo_quote(s) if s != NVDA else wide for s in (BTC, NVDA, SPX)})
    ruling = _rule({NVDA: 0.08}, b=b, inp=inp)
    ir = _inst(ruling, NVDA)
    assert ir.approved_weight == pytest.approx(0.02)
    assert ir.approved_weight == ir.current_weight
    # G7 and G8 both hold the leg at what is held; G7 is first in the canonical order. G3 (5%) is
    # also below the 8% request and is named, although it is not the tightest.
    assert ir.binding_guard is GuardId.G7_FEE_BUDGET
    binding = next(g for g in ir.rulings if g.guard is GuardId.G7_FEE_BUDGET)
    assert "also below" in binding.reason
    assert "G8_taker_only" in binding.reason
    assert "G3_size at 5.0000%" in binding.reason
    assert _status(ir, GuardId.G8_TAKER_ONLY) is GuardStatus.FIRED
    assert _status(ir, GuardId.G3_SIZE) is GuardStatus.FIRED


def test_a_looser_guard_never_hides_a_tighter_one() -> None:
    # ARGUS's first-to-bind chain let a looser cap return first while the breaker said trade
    # nothing (desk.py 1097-1105). Here G3 would allow 5%; the breaker in reduce-only allows only
    # what is held, and it binds.
    held = position(NVDA, qty_for_weight("0.01", NVDA))
    ruling = _rule({NVDA: 0.09}, b=book(positions=[held]), activation=Activation.REDUCE_ONLY)
    ir = _inst(ruling, NVDA)
    assert ir.binding_guard is GuardId.G10_BREAKER
    assert ir.approved_weight == pytest.approx(0.01)


def test_a_forced_exit_beats_every_ceiling() -> None:
    held = position(NVDA, qty_for_weight("0.03", NVDA))
    broken = demo_quote(NVDA, mark="240", index="222.84")
    inp = inputs(demo={NVDA: broken, BTC: demo_quote(BTC), SPX: demo_quote(SPX)})
    ruling = _rule({NVDA: 0.09}, b=book(positions=[held]), inp=inp)
    ir = _inst(ruling, NVDA)
    assert ir.approved_weight == 0.0
    assert ir.binding_guard is GuardId.G1_VENUE_INTEGRITY
    exit_ruling = next(g for g in ir.rulings if g.guard is GuardId.G1_VENUE_INTEGRITY)
    assert exit_ruling.forces_exit
    assert "G3_size at 5.0000%" in exit_ruling.reason


def test_a_missing_input_binds_as_not_evaluated() -> None:
    inp = inputs(index_moves={BTC: 30.0, NVDA: None, SPX: 30.0})
    ruling = _rule({NVDA: 0.03}, inp=inp)
    ir = _inst(ruling, NVDA)
    assert ir.approved_weight == 0.0
    assert ir.binding_guard is GuardId.G1_VENUE_INTEGRITY
    assert _status(ir, GuardId.G1_VENUE_INTEGRITY) is GuardStatus.NOT_EVALUATED


def test_ungrounded_targets_cannot_add_exposure() -> None:
    ruling = _rule({NVDA: 0.03, BTC: 0.02}, grounding={NVDA: UNGROUNDED})
    assert _inst(ruling, NVDA).approved_weight == 0.0
    assert _inst(ruling, NVDA).binding_guard is GuardId.G9_GROUNDING
    btc = _inst(ruling, BTC)
    assert btc.approved_weight == 0.0  # no report at all: NOT_EVALUATED, fail-closed
    assert _status(btc, GuardId.G9_GROUNDING) is GuardStatus.NOT_EVALUATED


# --- size and gross -----------------------------------------------------------------------------


def test_per_name_cap() -> None:
    ruling = _rule({NVDA: 0.2, BTC: -0.07})
    assert _inst(ruling, NVDA).approved_weight == pytest.approx(0.05)
    assert _inst(ruling, BTC).approved_weight == pytest.approx(-0.05)
    assert _inst(ruling, BTC).binding_guard is GuardId.G3_SIZE


def test_gross_cap_keeps_holdings_and_shares_what_is_left() -> None:
    held_names = (BTC, NVDA, SPX, MSTR)
    held = [position(s, qty_for_weight("0.05", s)) for s in held_names]
    new = (TSLA, "AAPLUSDT", "METAUSDT")
    # 20% held and asked to stay; three new names at 5% each share the 5% that is left.
    ruling = _rule(
        {**dict.fromkeys(held_names, 0.05), **dict.fromkeys(new, 0.05)}, b=book(positions=held)
    )
    for s in held_names:
        ir = _inst(ruling, s)
        assert ir.approved_weight == pytest.approx(ir.current_weight)
        assert ir.binding_guard is None
    for s in new:
        ir = _inst(ruling, s)
        assert ir.approved_weight == pytest.approx(0.05 / 3)
        assert ir.binding_guard is GuardId.G3_SIZE
    assert sum(abs(i.approved_weight) for i in ruling.instruments) == pytest.approx(0.25)
    book_g3 = next(g for g in ruling.book_rulings if g.guard is GuardId.G3_SIZE)
    assert book_g3.status is GuardStatus.FIRED
    assert "scaled" in book_g3.reason


def test_gross_exactly_at_the_cap_does_not_bind() -> None:
    held = [position(s, qty_for_weight("0.05", s)) for s in (BTC, NVDA, SPX, MSTR)]
    ruling = _rule(
        {BTC: 0.05, NVDA: 0.05, SPX: 0.05, MSTR: 0.05, TSLA: 0.05}, b=book(positions=held)
    )
    assert _inst(ruling, TSLA).approved_weight == pytest.approx(0.05)
    book_g3 = next(g for g in ruling.book_rulings if g.guard is GuardId.G3_SIZE)
    assert book_g3.status is GuardStatus.PASSED


def test_drifted_holdings_are_trimmed_to_the_caps() -> None:
    names = (BTC, NVDA, SPX, MSTR, TSLA, "AAPLUSDT")
    held = [position(s, qty_for_weight("0.06", s)) for s in names]
    ruling = _rule(None, b=book(positions=held))
    # Each 6% holding is trimmed to 5% per name; six of them are then scaled into the 25% gross.
    for ir in ruling.instruments:
        assert ir.approved_weight == pytest.approx(0.25 / 6)
    assert sum(abs(i.approved_weight) for i in ruling.instruments) == pytest.approx(0.25)


# --- reductions are never blocked ---------------------------------------------------------------


@pytest.mark.parametrize("target", [0.01, 0.0, -0.02])
def test_reductions_pass_every_refusal_at_once(target: float) -> None:
    held = position(NVDA, qty_for_weight("0.04", NVDA), last_increase_at=T0 - timedelta(hours=1))
    b = book(positions=[held], fees_today="50", fees_total="50", rebalances={NVDA: 5})
    ruling = _rule({NVDA: target}, b=b, activation=Activation.REDUCE_ONLY, grounding={})
    ir = _inst(ruling, NVDA)
    # A flip keeps only its reducing half: the close.
    assert ir.approved_weight == (target if target >= 0 else 0.0)
    for guard in (GuardId.G6_TURNOVER, GuardId.G7_FEE_BUDGET):
        ruled = next(g for g in ir.rulings if g.guard is guard)
        assert ruled.ceiling_abs_weight is not None
        assert ruled.ceiling_abs_weight >= abs(ir.approved_weight)
        if target >= 0:
            assert ruled.status is GuardStatus.PASSED


def test_a_held_symbol_the_proposal_omits_is_ruled_as_a_hold() -> None:
    held = position(NVDA, qty_for_weight("0.03", NVDA))
    ruling = _rule({BTC: 0.02}, b=book(positions=[held]))
    ir = _inst(ruling, NVDA)
    assert ir.proposed_weight is None
    assert ir.approved_weight == ir.current_weight


# --- the calendar -------------------------------------------------------------------------------


def test_weekend_preflatten_exits_us_legs_and_leaves_crypto() -> None:
    at = FRIDAY + timedelta(hours=19, minutes=50)
    held = [
        position(NVDA, qty_for_weight("0.03", NVDA)),
        position(BTC, qty_for_weight("0.03", BTC)),
    ]
    ruling = _rule({NVDA: 0.03, BTC: 0.03, SPX: 0.02}, b=book(positions=held, at=at), at=at)
    assert _inst(ruling, NVDA).approved_weight == 0.0
    assert _inst(ruling, NVDA).binding_guard is GuardId.G2_WEEKEND_FREEZE
    assert _inst(ruling, SPX).approved_weight == 0.0
    assert _inst(ruling, BTC).approved_weight == pytest.approx(0.03)


def test_no_new_us_exposure_in_the_buffer_but_holdings_stay() -> None:
    at = FRIDAY + timedelta(hours=18, minutes=30)
    held = [position(NVDA, qty_for_weight("0.03", NVDA))]
    ruling = _rule({NVDA: 0.05, SPX: 0.02, BTC: 0.02}, b=book(positions=held, at=at), at=at)
    assert _inst(ruling, NVDA).approved_weight == pytest.approx(0.03)
    assert _inst(ruling, SPX).approved_weight == 0.0
    assert _inst(ruling, BTC).approved_weight == pytest.approx(0.02)


def test_saturday_us_legs_are_flat() -> None:
    held = [position(NVDA, qty_for_weight("0.03", NVDA))]
    ruling = _rule({NVDA: 0.03, BTC: 0.02}, b=book(positions=held, at=SATURDAY), at=SATURDAY)
    assert _inst(ruling, NVDA).approved_weight == 0.0
    assert _inst(ruling, BTC).approved_weight == pytest.approx(0.02)


# --- the daily kill and the breaker -------------------------------------------------------------


def _killed_book() -> BookState:
    held = [
        position(NVDA, qty_for_weight("0.03", NVDA)),
        position(BTC, qty_for_weight("-0.02", BTC)),
    ]
    return book(positions=held, equity="9850", day_open="10000", peak="10000")


def test_the_daily_kill_flattens_everything_in_a_decision_ruling() -> None:
    ruling = _rule({NVDA: 0.05, BTC: -0.02, SPX: 0.01}, b=_killed_book())
    for ir in ruling.instruments:
        assert ir.approved_weight == 0.0
    assert _inst(ruling, NVDA).binding_guard is GuardId.G5_DAILY_KILL
    assert ruling.activation_after is Activation.HALTED
    kill = next(g for g in ruling.book_rulings if g.guard is GuardId.G5_DAILY_KILL)
    assert kill.forces_exit


def test_the_daily_kill_flattens_everything_between_decisions() -> None:
    k, _ = kernel()
    ruling = k.protective(book=_killed_book(), inputs=inputs(), breaker=breaker_state())
    assert ruling is not None
    assert ruling.protective_reason is ProtectiveReason.DAILY_KILL
    assert ruling.decision_id is None
    assert ruling.guards_applied == tuple(g for g in GuardId if g in PROTECTIVE_GUARDS)
    assert all(ir.approved_weight == 0.0 for ir in ruling.instruments)
    assert all(ir.proposed_weight is None for ir in ruling.instruments)


def test_a_model_outage_flattens_the_book() -> None:
    held = [
        position(NVDA, qty_for_weight("0.03", NVDA)),
        position(BTC, qty_for_weight("0.01", BTC)),
    ]
    k, _ = kernel()
    ruling = k.protective(
        book=book(positions=held), inputs=inputs(), breaker=breaker_state(), llm_outage=True
    )
    assert ruling is not None
    assert ruling.protective_reason is ProtectiveReason.LLM_OUTAGE
    assert all(ir.approved_weight == 0.0 for ir in ruling.instruments)
    assert all(ir.binding_guard is GuardId.G10_BREAKER for ir in ruling.instruments)
    assert ruling.activation_after is Activation.HALTED


def test_a_halted_breaker_flattens_and_is_filed_as_breaker() -> None:
    held = [position(NVDA, qty_for_weight("0.03", NVDA))]
    k, _ = kernel()
    ruling = k.protective(
        book=book(positions=held),
        inputs=inputs(),
        breaker=breaker_state(Activation.HALTED, ("drawdown_halt",)),
    )
    assert ruling is not None
    assert ruling.protective_reason is ProtectiveReason.BREAKER
    assert ruling.activation_before is Activation.HALTED


def test_protective_venue_integrity_exit_replays_the_demo_btc_excursion() -> None:
    # Demo BTCUSDT's mark ran 5.4% from its index on 2026-09-23 (DESIGN.md §14.7).
    held = [
        position(BTC, qty_for_weight("0.03", BTC)),
        position(NVDA, qty_for_weight("0.02", NVDA)),
    ]
    bad = demo_quote(BTC, mark="87711.9", index="83218.1")
    inp = inputs(demo={BTC: bad, NVDA: demo_quote(NVDA), SPX: demo_quote(SPX)})
    k, _ = kernel()
    ruling = k.protective(book=book(positions=held), inputs=inp, breaker=breaker_state())
    assert ruling is not None
    assert ruling.protective_reason is ProtectiveReason.VENUE_INTEGRITY
    assert _inst(ruling, BTC).approved_weight == 0.0
    assert _inst(ruling, NVDA).approved_weight == pytest.approx(_inst(ruling, NVDA).current_weight)


def test_protective_weekend_preflatten_and_the_broadest_cause_wins() -> None:
    at = FRIDAY + timedelta(hours=19, minutes=46)
    held = [
        position(NVDA, qty_for_weight("0.03", NVDA)),
        position(BTC, qty_for_weight("0.02", BTC)),
    ]
    k, _ = kernel(at)
    fresh = inputs(at=at, snapshot_at=at)
    ruling = k.protective(book=book(positions=held, at=at), inputs=fresh, breaker=breaker_state())
    assert ruling is not None
    assert ruling.protective_reason is ProtectiveReason.WEEKEND_FREEZE
    assert _inst(ruling, NVDA).approved_weight == 0.0
    assert _inst(ruling, BTC).approved_weight == pytest.approx(_inst(ruling, BTC).current_weight)
    killed = book(positions=held, at=at, equity="9800", day_open="10000")
    ruling = k.protective(book=killed, inputs=fresh, breaker=breaker_state())
    assert ruling is not None
    assert ruling.protective_reason is ProtectiveReason.DAILY_KILL


def test_protective_has_nothing_to_do_on_a_healthy_or_empty_book() -> None:
    k, _ = kernel()
    assert k.protective(book=book(), inputs=inputs(), breaker=breaker_state()) is None
    held = [position(NVDA, qty_for_weight("0.03", NVDA))]
    assert k.protective(book=book(positions=held), inputs=inputs(), breaker=breaker_state()) is None
    # Reduce-only and a stale index refuse increases; a hold has none to refuse.
    ruling = k.protective(
        book=book(positions=held),
        inputs=inputs(index_moves={NVDA: 0.5}),
        breaker=breaker_state(Activation.REDUCE_ONLY, ("losing_streak",)),
    )
    assert ruling is None


# --- staleness ----------------------------------------------------------------------------------


def test_a_stale_quote_cannot_open_but_can_close() -> None:
    held = position(BTC, qty_for_weight("0.03", BTC))
    old = {s: demo_quote(s, at=T0 - timedelta(minutes=5)) for s in (BTC, NVDA, SPX)}
    ruling = _rule({NVDA: 0.03, BTC: 0.0}, b=book(positions=[held]), inp=inputs(demo=old))
    assert _inst(ruling, NVDA).approved_weight == 0.0
    assert _inst(ruling, NVDA).binding_guard is GuardId.G10_BREAKER
    assert _inst(ruling, BTC).approved_weight == 0.0
    assert _inst(ruling, BTC).binding_guard is None


def test_the_kernel_measures_staleness_against_its_own_clock() -> None:
    # Inputs assembled at T0, ruled on ten minutes later: every quote is ten minutes old.
    ruling = _rule({NVDA: 0.03}, inp=inputs(), at=T0 + timedelta(minutes=10))
    assert _inst(ruling, NVDA).approved_weight == 0.0
    assert ruling.at == T0 + timedelta(minutes=10)


# --- guard subsets ------------------------------------------------------------------------------


def test_venue_guards_rule_without_the_decision_maker_guards() -> None:
    ruling = _rule({NVDA: 0.03}, guards=VENUE_GUARDS, grounding={NVDA: UNGROUNDED})
    ir = _inst(ruling, NVDA)
    assert ir.approved_weight == pytest.approx(0.03)
    assert {g.guard for g in ir.rulings} == set(VENUE_GUARDS)
    assert set(ruling.guards_applied) == set(VENUE_GUARDS)
    assert {g.guard for g in ruling.book_rulings} == {GuardId.G3_SIZE, GuardId.G5_DAILY_KILL}


def test_no_guards_approves_the_proposal_untouched() -> None:
    ruling = _rule({NVDA: 0.4}, guards=frozenset())
    assert _inst(ruling, NVDA).approved_weight == 0.4
    assert ruling.book_rulings == ()


# --- refusals to rule ---------------------------------------------------------------------------


def test_a_proposal_needs_a_decision() -> None:
    k, _ = kernel()
    with pytest.raises(KernelError, match="decision id"):
        k.rule(
            proposed={NVDA: 0.03},
            book=book(),
            inputs=inputs(),
            context=protective_context(),
            breaker=breaker_state(),
        )


def test_a_protective_context_without_a_proposal_only_holds() -> None:
    held = [position(NVDA, qty_for_weight("0.03", NVDA))]
    k, _ = kernel()
    ruling = k.rule(
        proposed=None,
        book=book(positions=held),
        inputs=inputs(),
        context=protective_context(ProtectiveReason.BREAKER),
        breaker=breaker_state(),
    )
    ir = _inst(ruling, NVDA)
    assert ir.approved_weight == ir.current_weight
    assert ruling.protective_reason is ProtectiveReason.BREAKER


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_proposals_are_refused(bad: float) -> None:
    with pytest.raises(KernelError, match="non-finite"):
        _rule({NVDA: bad})


# --- the record ---------------------------------------------------------------------------------


def test_the_ruling_id_is_the_hash_of_its_content_and_is_deterministic() -> None:
    first = _rule({NVDA: 0.04, BTC: 0.2})
    second = _rule({NVDA: 0.04, BTC: 0.2})
    assert first.ruling_id == second.ruling_id == ruling_hash(first)
    assert len(first.ruling_id) == 64
    assert _rule({NVDA: 0.04, BTC: 0.19}).ruling_id != first.ruling_id
    assert _rule({NVDA: 0.04, BTC: 0.2}, at=T0 + timedelta(seconds=1)).ruling_id != first.ruling_id


def test_the_ruling_round_trips_through_json() -> None:
    ruling = _rule({NVDA: 0.2, BTC: -0.03}, grounding={NVDA: UNGROUNDED, BTC: UNGROUNDED})
    again = KernelRuling.model_validate_json(ruling.model_dump_json())
    assert again == ruling
    assert ruling_hash(again) == ruling.ruling_id


def test_activation_after_records_what_the_book_demands() -> None:
    ruling = _rule({NVDA: 0.03}, b=book(equity="9700", peak="10000"))
    assert ruling.activation_before is Activation.ACTIVE
    assert ruling.activation_after is Activation.REDUCE_ONLY
    assert _inst(ruling, NVDA).approved_weight == 0.0
