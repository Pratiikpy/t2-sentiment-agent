"""Each guard at, just inside and just outside its threshold, and with each input missing.

Thresholds come from ``POLICY_V1``; numbers are chosen so the Decimal arithmetic lands exactly on
the threshold, so "at" really is at it.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from helpers import T0
from kernel.kbuild import (
    BTC,
    GROUNDED,
    NVDA,
    SPX,
    UNGROUNDED,
    book,
    breaker_state,
    demo_quote,
    flat_quote,
    live_quote,
    position,
    spec,
)
from sentiment_agent.kernel.guards import (
    Leg,
    allocate_gross,
    g1_venue_integrity,
    g2_weekend_freeze,
    g3_size,
    g4_stop,
    g5_daily_kill,
    g6_turnover,
    g7_fee_budget,
    g8_taker_only,
    g9_grounding,
    g10_breaker,
    g11_eligibility,
    session_open,
    weekend_phase,
)
from sentiment_agent.policy import POLICY_V1
from sentiment_agent.types import (
    Activation,
    AssetClass,
    GuardId,
    GuardRuling,
    GuardStatus,
    Quote,
)

P = POLICY_V1
OPEN_LONG = Leg(symbol=NVDA, current=0.0, proposed=0.03)
INCREASE = Leg(symbol=NVDA, current=0.02, proposed=0.04)
REDUCE = Leg(symbol=NVDA, current=0.04, proposed=0.01)
CLOSE = Leg(symbol=NVDA, current=0.04, proposed=0.0)
FLIP = Leg(symbol=NVDA, current=0.03, proposed=-0.02)
HOLD = Leg(symbol=NVDA, current=0.03)
INERT = Leg(symbol=NVDA, current=0.0, proposed=0.0)


def _fired_increase(r: GuardRuling, leg: Leg) -> None:
    assert r.status is GuardStatus.FIRED
    assert not r.forces_exit
    assert r.ceiling_abs_weight == leg.hold_ceiling


def _exit(r: GuardRuling) -> None:
    assert r.status is GuardStatus.FIRED
    assert r.forces_exit
    assert r.ceiling_abs_weight == 0.0


def _passed(r: GuardRuling) -> None:
    assert r.status is GuardStatus.PASSED, r.reason
    assert not r.forces_exit


# --- the leg ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("leg", "hold", "adds"),
    [
        (OPEN_LONG, 0.0, True),
        (INCREASE, 0.02, True),
        (REDUCE, 0.04, False),
        (CLOSE, 0.0, False),
        (FLIP, 0.0, True),
        (HOLD, 0.03, False),
        (INERT, 0.0, False),
        (Leg(symbol=NVDA, current=-0.03, proposed=-0.05), 0.03, True),
    ],
)
def test_leg_hold_ceiling_and_whether_it_adds(leg: Leg, hold: float, adds: bool) -> None:
    assert leg.hold_ceiling == hold
    assert leg.adds_exposure is adds


# --- G1 venue integrity -------------------------------------------------------------------------


def _g1(
    leg: Leg = OPEN_LONG,
    *,
    demo: Quote | None = None,
    live: Quote | None = None,
    move: float | None = 30.0,
    at: datetime = T0,
    symbol: str = NVDA,
) -> GuardRuling:
    return g1_venue_integrity(
        leg,
        entry=P.entry(symbol),
        demo=demo if demo is not None else flat_quote(symbol, "100"),
        live=live if live is not None else live_quote(symbol, last="100"),
        index_move_bps_3h=move,
        at=at,
        policy=P,
    )


@pytest.mark.parametrize(
    ("mark", "exits"), [("103", False), ("102.99", False), ("103.01", True), ("96.99", True)]
)
def test_g1_mark_index_gap_at_and_around_three_percent(mark: str, exits: bool) -> None:
    demo = demo_quote(NVDA, mark=mark, index="100", last="100", bid="99.99", ask="100.01")
    r = _g1(demo=demo)
    if exits:
        _exit(r)
        assert "from its index" in r.reason
    else:
        _passed(r)


def test_g1_index_not_positive_is_an_exit_with_finite_inputs() -> None:
    demo = demo_quote(NVDA, mark="100", index="0", last="100", bid="99.99", ask="100.01")
    r = _g1(demo=demo)
    _exit(r)
    assert r.inputs["mark_index_gap"] == "inf"


@pytest.mark.parametrize(
    ("demo_last", "exits"), [("10008.3", False), ("10008.2", False), ("10008.4", True)]
)
def test_g1_demo_live_gap_at_and_around_the_p99(demo_last: str, exits: bool) -> None:
    # BTCUSDT's measured p99 is 8.3 bps: 10008.3 against 10000 is exactly 8.3 bps.
    demo = demo_quote(BTC, mark="10000", index="10000", last=demo_last, bid="9999", ask="10001")
    r = _g1(
        Leg(symbol=BTC, current=0.0, proposed=0.02),
        demo=demo,
        live=live_quote(BTC, last="10000"),
        symbol=BTC,
    )
    if exits:
        _exit(r)
        assert "p99" in r.reason
    else:
        _passed(r)


@pytest.mark.parametrize(("move", "refuses"), [(2.0, False), (2.01, False), (1.99, True)])
def test_g1_stale_index_at_and_around_two_bps(move: float, refuses: bool) -> None:
    r = _g1(move=move)
    if refuses:
        _fired_increase(r, OPEN_LONG)
        assert "stale" in r.reason
    else:
        _passed(r)


def test_g1_stale_index_never_blocks_a_reduction() -> None:
    r = _g1(REDUCE, move=0.1)
    _passed(r)
    assert r.ceiling_abs_weight == REDUCE.hold_ceiling


def test_g1_stale_check_waits_for_the_session() -> None:
    saturday = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)
    r = _g1(HOLD, move=0.1, at=saturday)
    assert r.inputs["session_open"] is False
    _passed(r)


@pytest.mark.parametrize("gone", ["demo", "live", "move"])
def test_g1_each_missing_input_fails_closed(gone: str) -> None:
    r = g1_venue_integrity(
        OPEN_LONG,
        entry=P.entry(NVDA),
        demo=None if gone == "demo" else flat_quote(NVDA, "100"),
        live=None if gone == "live" else live_quote(NVDA, last="100"),
        index_move_bps_3h=None if gone == "move" else 30.0,
        at=T0,
        policy=P,
    )
    assert r.status is GuardStatus.NOT_EVALUATED
    assert r.ceiling_abs_weight == 0.0


def test_g1_missing_input_still_exits_on_a_measured_breach() -> None:
    demo = demo_quote(NVDA, mark="110", index="100", last="100", bid="99.99", ask="100.01")
    r = g1_venue_integrity(
        HOLD, entry=P.entry(NVDA), demo=demo, live=None, index_move_bps_3h=None, at=T0, policy=P
    )
    _exit(r)


def test_g1_on_an_inert_leg_reports_the_breach_without_firing() -> None:
    demo = demo_quote(NVDA, mark="110", index="100", last="100", bid="99.99", ask="100.01")
    r = _g1(INERT, demo=demo)
    assert r.status is GuardStatus.PASSED
    assert "nothing to exit" in r.reason


# --- G2 weekend freeze --------------------------------------------------------------------------

FRIDAY = datetime(2026, 9, 25, tzinfo=UTC)
MONDAY = datetime(2026, 9, 28, tzinfo=UTC)


@pytest.mark.parametrize(
    ("at", "phase"),
    [
        (FRIDAY + timedelta(hours=17, minutes=59, seconds=59), "open"),
        (FRIDAY + timedelta(hours=18), "no_open_buffer"),
        (FRIDAY + timedelta(hours=19, minutes=44, seconds=59), "no_open_buffer"),
        (FRIDAY + timedelta(hours=19, minutes=45), "preflatten"),
        (FRIDAY + timedelta(hours=19, minutes=59, seconds=59), "preflatten"),
        (FRIDAY + timedelta(hours=20), "frozen"),
        (FRIDAY + timedelta(days=1, hours=12), "frozen"),
        (MONDAY - timedelta(microseconds=1), "frozen"),
        (MONDAY, "open"),
        (datetime(2026, 9, 23, 13, 0, tzinfo=UTC), "open"),
    ],
)
def test_weekend_phase_boundaries(at: datetime, phase: str) -> None:
    assert weekend_phase(at, P.weekend) == phase


def test_session_open_follows_the_freeze_for_us_legs_only() -> None:
    saturday = FRIDAY + timedelta(days=1)
    assert session_open(AssetClass.CRYPTO, saturday, P.weekend)
    assert not session_open(AssetClass.US_EQUITY, saturday, P.weekend)
    assert session_open(AssetClass.US_INDEX, T0, P.weekend)
    assert session_open(None, saturday, P.weekend)


@pytest.mark.parametrize(
    ("at", "leg", "effect"),
    [
        (FRIDAY + timedelta(hours=17), OPEN_LONG, "pass"),
        (FRIDAY + timedelta(hours=18, minutes=30), OPEN_LONG, "refuse"),
        (FRIDAY + timedelta(hours=18, minutes=30), HOLD, "pass"),
        (FRIDAY + timedelta(hours=19, minutes=50), HOLD, "exit"),
        (FRIDAY + timedelta(days=1), HOLD, "exit"),
        (FRIDAY + timedelta(days=1), OPEN_LONG, "exit"),
        (MONDAY, OPEN_LONG, "pass"),
    ],
)
def test_g2_by_phase(at: datetime, leg: Leg, effect: str) -> None:
    r = g2_weekend_freeze(leg, asset_class=AssetClass.US_EQUITY, at=at, policy=P)
    if effect == "exit":
        _exit(r)
    elif effect == "refuse":
        _fired_increase(r, leg)
    else:
        _passed(r)


def test_g2_does_not_apply_to_crypto_and_fails_closed_without_an_asset_class() -> None:
    saturday = FRIDAY + timedelta(days=1)
    r = g2_weekend_freeze(OPEN_LONG, asset_class=AssetClass.CRYPTO, at=saturday, policy=P)
    assert r.status is GuardStatus.NOT_APPLICABLE
    r = g2_weekend_freeze(OPEN_LONG, asset_class=None, at=T0, policy=P)
    assert r.status is GuardStatus.NOT_EVALUATED
    assert r.ceiling_abs_weight == 0.0


# --- G3 size ------------------------------------------------------------------------------------


@pytest.mark.parametrize(("proposed", "fires"), [(0.05, False), (0.0499, False), (0.0501, True)])
def test_g3_per_name_cap(proposed: float, fires: bool) -> None:
    r = g3_size(Leg(symbol=NVDA, current=0.0, proposed=proposed), gross_ceiling=None, policy=P)
    assert r.ceiling_abs_weight == 0.05
    assert r.status is (GuardStatus.FIRED if fires else GuardStatus.PASSED)


def test_g3_gross_share_binds_when_tighter() -> None:
    r = g3_size(OPEN_LONG, gross_ceiling=0.01, policy=P, gross_note="shared")
    assert r.status is GuardStatus.FIRED
    assert r.ceiling_abs_weight == 0.01
    assert "gross" in r.reason
    assert "shared" in r.reason


def test_allocate_gross_within_the_cap_sets_no_ceilings() -> None:
    a = allocate_gross({"A": (0.05, 0.0), "B": (0.05, 0.05)}, 0.25)
    assert not a.binds
    assert a.requested == pytest.approx(0.10)


def test_allocate_gross_keeps_holdings_and_scales_new_exposure() -> None:
    # 0.20 already held; three new 0.05 requests share the remaining 0.05.
    candidates = {"H1": (0.10, 0.10), "H2": (0.10, 0.10), "A": (0.05, 0.0), "B": (0.05, 0.0)}
    candidates["C"] = (0.05, 0.0)
    a = allocate_gross(candidates, 0.25)
    assert a.binds
    assert a.ceilings["H1"] == pytest.approx(0.10)
    assert a.ceilings["A"] == pytest.approx(0.05 / 3)
    assert sum(a.ceilings.values()) == pytest.approx(0.25)
    assert "scaled" in a.note()


def test_allocate_gross_scales_holdings_when_they_alone_exceed_the_cap() -> None:
    a = allocate_gross({"A": (0.15, 0.15), "B": (0.15, 0.15), "C": (0.02, 0.0)}, 0.25)
    assert a.ceilings["A"] == pytest.approx(0.125)
    assert a.ceilings["C"] == 0.0
    assert sum(a.ceilings.values()) == pytest.approx(0.25)
    assert "nothing new admitted" in a.note()


# --- G4 stop ------------------------------------------------------------------------------------


def test_g4_places_a_stop_four_percent_from_the_entry_on_mark() -> None:
    r = g4_stop(OPEN_LONG, spec=spec(NVDA), demo=flat_quote(NVDA, "100", spread="0.02"), policy=P)
    _passed(r)
    assert r.inputs["entry"] == "100.01"
    assert r.inputs["stop_price"] == "96.01"
    assert r.inputs["trigger"] == "mark"
    short = Leg(symbol=NVDA, current=0.0, proposed=-0.03)
    r = g4_stop(short, spec=spec(NVDA), demo=flat_quote(NVDA, "100", spread="0.02"), policy=P)
    assert r.inputs["entry"] == "99.99"
    assert r.inputs["stop_price"] == "103.98"


def test_g4_does_not_apply_to_legs_that_add_nothing() -> None:
    for leg in (REDUCE, CLOSE, HOLD):
        r = g4_stop(leg, spec=None, demo=None, policy=P)
        assert r.status is GuardStatus.NOT_APPLICABLE


@pytest.mark.parametrize("gone", ["spec", "demo"])
def test_g4_missing_input_fails_closed(gone: str) -> None:
    r = g4_stop(
        INCREASE,
        spec=None if gone == "spec" else spec(NVDA),
        demo=None if gone == "demo" else flat_quote(NVDA, "100"),
        policy=P,
    )
    assert r.status is GuardStatus.NOT_EVALUATED
    assert r.ceiling_abs_weight == INCREASE.hold_ceiling


def test_g4_refuses_when_the_grid_is_too_coarse_for_any_stop() -> None:
    coarse = spec(NVDA, price_step="10")
    r = g4_stop(OPEN_LONG, spec=coarse, demo=flat_quote(NVDA, "12"), policy=P)
    _fired_increase(r, OPEN_LONG)
    assert "no valid stop" in r.reason


def test_g4_refuses_a_stop_that_would_trigger_on_the_fill() -> None:
    # The ask is 5% above the mark: a stop 4% below the ask sits above the mark it triggers on.
    demo = demo_quote(NVDA, mark="100", index="100", bid="99.9", ask="105", last="100")
    r = g4_stop(OPEN_LONG, spec=spec(NVDA), demo=demo, policy=P)
    _fired_increase(r, OPEN_LONG)
    assert "losing side of the Demo mark" in r.reason


# --- G5 daily kill ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("equity", "fires"), [("9850", True), ("9851", False), ("9849", True), ("10100", False)]
)
def test_g5_at_and_around_the_kill(equity: str, fires: bool) -> None:
    b = book(equity=equity, day_open="10000", peak="10100")
    r = g5_daily_kill(HOLD, book=b, policy=P)
    book_level = g5_daily_kill(None, book=b, policy=P)
    assert book_level.symbol is None
    if fires:
        _exit(r)
        _exit(book_level)
    else:
        _passed(r)
        _passed(book_level)


def test_g5_without_a_day_open_equity_fails_closed() -> None:
    b = book(day_open="0")
    r = g5_daily_kill(OPEN_LONG, book=b, policy=P)
    assert r.status is GuardStatus.NOT_EVALUATED
    assert g5_daily_kill(None, book=b, policy=P).status is GuardStatus.NOT_EVALUATED


# --- G6 turnover --------------------------------------------------------------------------------


@pytest.mark.parametrize(("count", "fires"), [(0, False), (1, False), (2, True), (3, True)])
def test_g6_daily_order_count(count: int, fires: bool) -> None:
    r = g6_turnover(
        OPEN_LONG,
        position=None,
        rebalances_today=count,
        invalidation_declared=False,
        at=T0,
        policy=P,
    )
    if fires:
        _fired_increase(r, OPEN_LONG)
    else:
        _passed(r)


@pytest.mark.parametrize(
    ("held_for", "fires"),
    [
        (timedelta(hours=24), False),
        (timedelta(hours=24, seconds=1), False),
        (timedelta(hours=23, minutes=59, seconds=59), True),
    ],
)
def test_g6_minimum_hold(held_for: timedelta, fires: bool) -> None:
    pos = position(NVDA, "1", last_increase_at=T0 - held_for)
    r = g6_turnover(
        INCREASE, position=pos, rebalances_today=0, invalidation_declared=False, at=T0, policy=P
    )
    if fires:
        _fired_increase(r, INCREASE)
    else:
        _passed(r)


def test_g6_declared_invalidation_lifts_the_hold_but_not_the_count() -> None:
    pos = position(NVDA, "1", last_increase_at=T0 - timedelta(hours=1))
    r = g6_turnover(
        INCREASE, position=pos, rebalances_today=1, invalidation_declared=True, at=T0, policy=P
    )
    _passed(r)
    assert "lifted" in r.reason
    r = g6_turnover(
        INCREASE, position=pos, rebalances_today=2, invalidation_declared=True, at=T0, policy=P
    )
    _fired_increase(r, INCREASE)


def test_g6_blocks_a_flip_inside_the_hold_to_a_close() -> None:
    pos = position(NVDA, "1", last_increase_at=T0 - timedelta(hours=2))
    r = g6_turnover(
        FLIP, position=pos, rebalances_today=0, invalidation_declared=False, at=T0, policy=P
    )
    _fired_increase(r, FLIP)
    assert r.ceiling_abs_weight == 0.0


@pytest.mark.parametrize("leg", [REDUCE, CLOSE, HOLD])
def test_g6_never_blocks_a_reduction(leg: Leg) -> None:
    pos = position(NVDA, "1", last_increase_at=T0 - timedelta(minutes=5))
    r = g6_turnover(
        leg, position=pos, rebalances_today=9, invalidation_declared=False, at=T0, policy=P
    )
    assert r.status is GuardStatus.PASSED
    assert r.ceiling_abs_weight is not None
    assert r.ceiling_abs_weight >= abs(leg.reference)


# --- G7 fee budget ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fees_today", "fires"), [("6.99", False), ("7", True), ("7.01", True), ("0", False)]
)
def test_g7_daily_budget(fees_today: str, fires: bool) -> None:
    b = book(fees_today=fees_today, fees_total=fees_today)
    r = g7_fee_budget(OPEN_LONG, book=b, policy=P)
    book_level = g7_fee_budget(None, book=b, policy=P)
    if fires:
        _fired_increase(r, OPEN_LONG)
        assert book_level.status is GuardStatus.FIRED
        assert not book_level.forces_exit
    else:
        _passed(r)
        _passed(book_level)


@pytest.mark.parametrize(("fees_total", "fires"), [("19.99", False), ("20", True)])
def test_g7_window_budget(fees_total: str, fires: bool) -> None:
    r = g7_fee_budget(OPEN_LONG, book=book(fees_total=fees_total), policy=P)
    assert r.status is (GuardStatus.FIRED if fires else GuardStatus.PASSED)


@pytest.mark.parametrize("leg", [REDUCE, CLOSE, HOLD, FLIP])
def test_g7_never_blocks_a_reduction(leg: Leg) -> None:
    r = g7_fee_budget(leg, book=book(fees_today="50", fees_total="50"), policy=P)
    assert r.ceiling_abs_weight == leg.hold_ceiling
    if leg is FLIP:
        assert r.status is GuardStatus.FIRED  # the open half of a flip is new exposure
    else:
        assert r.status is GuardStatus.PASSED


def test_g7_without_equity_fails_closed() -> None:
    r = g7_fee_budget(OPEN_LONG, book=book(day_open="0"), policy=P)
    assert r.status is GuardStatus.NOT_EVALUATED


# --- G8 taker only ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bid", "ask", "fires"),
    [("99.9", "100.1", False), ("99.95", "100.05", False), ("99.8999", "100.1001", True)],
)
def test_g8_spread_at_and_around_twenty_bps(bid: str, ask: str, fires: bool) -> None:
    demo = demo_quote(NVDA, mark="100", index="100", last="100", bid=bid, ask=ask)
    r = g8_taker_only(OPEN_LONG, demo=demo, policy=P)
    assert r.inputs["order_type"] == "market"
    if fires:
        _fired_increase(r, OPEN_LONG)
    else:
        _passed(r)


@pytest.mark.parametrize(("bid", "ask"), [("0", "100"), ("100.1", "100"), ("100", "0")])
def test_g8_refuses_a_one_sided_or_crossed_book(bid: str, ask: str) -> None:
    demo = demo_quote(NVDA, mark="100", index="100", last="100", bid=bid, ask=ask)
    _fired_increase(g8_taker_only(OPEN_LONG, demo=demo, policy=P), OPEN_LONG)
    _passed(g8_taker_only(REDUCE, demo=demo, policy=P))


def test_g8_missing_quote_fails_closed() -> None:
    assert g8_taker_only(OPEN_LONG, demo=None, policy=P).status is GuardStatus.NOT_EVALUATED


# --- G9 grounding -------------------------------------------------------------------------------


def test_g9_grounded_passes_and_ungrounded_refuses() -> None:
    _passed(g9_grounding(OPEN_LONG, report=GROUNDED, policy=P))
    r = g9_grounding(OPEN_LONG, report=UNGROUNDED, policy=P)
    _fired_increase(r, OPEN_LONG)
    assert "'7.3%'" in r.reason


def test_g9_missing_report_fails_closed_and_reductions_are_not_its_concern() -> None:
    r = g9_grounding(INCREASE, report=None, policy=P)
    assert r.status is GuardStatus.NOT_EVALUATED
    assert r.ceiling_abs_weight == INCREASE.hold_ceiling
    for leg in (REDUCE, CLOSE, HOLD):
        assert g9_grounding(leg, report=UNGROUNDED, policy=P).status is GuardStatus.NOT_APPLICABLE


# --- G10 breaker --------------------------------------------------------------------------------


def _g10(
    leg: Leg | None = OPEN_LONG,
    *,
    activation: Activation = Activation.ACTIVE,
    equity: str = "10000",
    peak: str = "10000",
    losses: int = 0,
    outage: bool = False,
    demo: Quote | None = None,
    snapshot_at: datetime | None = T0,
    unreconciled: tuple[str, ...] = (),
) -> GuardRuling:
    return g10_breaker(
        leg,
        breaker=breaker_state(activation),
        book=book(equity=equity, peak=peak, losses=losses),
        llm_outage=outage,
        demo=demo if demo is not None else flat_quote(NVDA, "100"),
        snapshot_taken_at=snapshot_at,
        at=T0,
        policy=P,
        venue_unreconciled=unreconciled,
    )


def test_g10_follows_the_breaker_state() -> None:
    _passed(_g10())
    _fired_increase(_g10(activation=Activation.REDUCE_ONLY), OPEN_LONG)
    _passed(_g10(REDUCE, activation=Activation.REDUCE_ONLY))
    _exit(_g10(HOLD, activation=Activation.HALTED))
    assert _g10(None, activation=Activation.REDUCE_ONLY).status is GuardStatus.FIRED
    _exit(_g10(None, activation=Activation.HALTED))


def test_g10_refuses_every_increase_while_the_book_is_not_reconciled_to_the_venue() -> None:
    why = ("missing_fill: fills since 2026-09-25T13:30:00+00:00 unreadable: timeout",)
    refused = _g10(unreconciled=why)
    _fired_increase(refused, OPEN_LONG)
    assert "not reconciled to the venue" in refused.reason
    assert "missing_fill" in refused.reason
    _passed(_g10(REDUCE, unreconciled=why))  # a reduction is never blocked
    assert _g10(None, unreconciled=why).status is GuardStatus.FIRED
    assert "venue_unreconciled" not in _g10().inputs, "a reconciled book leaves no trace"


@pytest.mark.parametrize(
    ("equity", "effect"),
    [
        ("9751", "pass"),
        ("9750", "refuse"),
        ("9601", "refuse"),
        ("9600", "exit"),
        ("9599", "exit"),
    ],
)
def test_g10_recomputes_the_drawdown_ladder_from_the_book(equity: str, effect: str) -> None:
    r = _g10(HOLD if effect == "exit" else OPEN_LONG, equity=equity, peak="10000")
    if effect == "exit":
        _exit(r)
    elif effect == "refuse":
        _fired_increase(r, OPEN_LONG)
        assert "drawdown_reduce_only" in r.reason
    else:
        _passed(r)


@pytest.mark.parametrize(("losses", "fires"), [(3, False), (4, True), (5, True)])
def test_g10_losing_streak(losses: int, fires: bool) -> None:
    r = _g10(losses=losses)
    assert r.status is (GuardStatus.FIRED if fires else GuardStatus.PASSED)


def test_g10_model_outage_flattens() -> None:
    _exit(_g10(HOLD, outage=True))
    _exit(_g10(None, outage=True))


@pytest.mark.parametrize(
    ("age", "fires"),
    [
        (timedelta(minutes=15), False),
        (timedelta(minutes=14, seconds=59), False),
        (timedelta(minutes=15, seconds=1), True),
        (-timedelta(minutes=16), True),
    ],
)
def test_g10_snapshot_staleness(age: timedelta, fires: bool) -> None:
    r = _g10(snapshot_at=T0 - age)
    if fires:
        _fired_increase(r, OPEN_LONG)
        assert "snapshot" in r.reason
    else:
        _passed(r)


@pytest.mark.parametrize(
    ("age", "fires"),
    [
        (timedelta(seconds=120), False),
        (timedelta(seconds=119), False),
        (timedelta(seconds=121), True),
    ],
)
def test_g10_quote_staleness(age: timedelta, fires: bool) -> None:
    r = _g10(demo=flat_quote(NVDA, "100", at=T0 - age))
    if fires:
        _fired_increase(r, OPEN_LONG)
        assert "quote" in r.reason
    else:
        _passed(r)


def test_g10_quote_age_uses_the_older_of_venue_and_fetch_time() -> None:
    demo = demo_quote(NVDA, at=T0 - timedelta(seconds=200), fetched_at=T0)
    _fired_increase(_g10(demo=demo), OPEN_LONG)


@pytest.mark.parametrize("gone", ["snapshot", "quote"])
def test_g10_missing_freshness_input_fails_closed(gone: str) -> None:
    r = g10_breaker(
        OPEN_LONG,
        breaker=breaker_state(),
        book=book(),
        llm_outage=False,
        demo=None if gone == "quote" else flat_quote(NVDA, "100"),
        snapshot_taken_at=None if gone == "snapshot" else T0,
        at=T0,
        policy=P,
    )
    assert r.status is GuardStatus.NOT_EVALUATED
    assert r.ceiling_abs_weight == 0.0


# --- G11 eligibility ----------------------------------------------------------------------------


def _g11(
    leg: Leg = OPEN_LONG,
    *,
    held: str = "0",
    price: str | None = "100",
    equity: str = "10000",
    candidate: float | None = None,
    status: str = "online",
    min_qty: str | None = None,
    min_amount: str | None = None,
) -> GuardRuling:
    return g11_eligibility(
        leg,
        policy=P,
        spec=spec(NVDA, status=status, min_qty=min_qty, min_amount=min_amount),
        position_qty=Decimal(held),
        price=None if price is None else Decimal(price),
        equity=Decimal(equity),
        candidate_abs_weight=candidate,
    )


@pytest.mark.parametrize("symbol", ["XRPUSDT", "ETHUSDT", "PEPEUSDT"])
def test_g11_exits_anything_outside_the_universe(symbol: str) -> None:
    leg = Leg(symbol=symbol, current=0.02)
    r = g11_eligibility(
        leg, policy=P, spec=None, position_qty=Decimal(1), price=None, equity=Decimal(10_000)
    )
    _exit(r)
    assert ("excluded" in r.reason) == (symbol != "PEPEUSDT")


def test_g11_offline_refuses_increases_and_keeps_holdings() -> None:
    _fired_increase(_g11(status="halt"), OPEN_LONG)
    _passed(_g11(HOLD, status="halt"))


def test_g11_missing_spec_fails_closed() -> None:
    r = g11_eligibility(
        OPEN_LONG,
        policy=P,
        spec=None,
        position_qty=Decimal(0),
        price=Decimal(100),
        equity=Decimal(10_000),
    )
    assert r.status is GuardStatus.NOT_EVALUATED


def test_g11_missing_price_fails_closed_for_an_increase() -> None:
    assert _g11(price=None).status is GuardStatus.NOT_EVALUATED


@pytest.mark.parametrize(
    ("weight", "fires"),
    [(0.0005, False), (0.00051, False), (0.00049, True)],
)
def test_g11_min_order_amount_after_rounding(weight: float, fires: bool) -> None:
    # 10,000 equity at 100: weight 0.0005 is 0.05 NVDA = 5 USDT, exactly minOrderAmount.
    r = _g11(Leg(symbol=NVDA, current=0.0, proposed=weight))
    assert r.status is (GuardStatus.FIRED if fires else GuardStatus.PASSED)
    if fires:
        assert "minOrderAmount" in r.reason


@pytest.mark.parametrize(("weight", "fires"), [(0.001, False), (0.00099, True)])
def test_g11_min_order_qty_after_rounding(weight: float, fires: bool) -> None:
    r = _g11(Leg(symbol=NVDA, current=0.0, proposed=weight), min_qty="0.10", min_amount="1")
    assert r.status is (GuardStatus.FIRED if fires else GuardStatus.PASSED)
    if fires:
        assert "minOrderQty" in r.reason


def test_g11_sizes_an_increase_by_what_it_adds_to_the_position() -> None:
    # Holding 3.00 NVDA at 100 on 10,000 is 3%; 3.004% adds 0.004 NVDA, which rounds to nothing.
    r = _g11(Leg(symbol=NVDA, current=0.03, proposed=0.03004), held="3.00")
    assert r.status is GuardStatus.FIRED
    assert "rounds to nothing" in r.reason


def test_g11_uses_the_candidate_the_other_guards_left() -> None:
    assert _g11(Leg(symbol=NVDA, current=0.0, proposed=0.05)).status is GuardStatus.PASSED
    r = _g11(Leg(symbol=NVDA, current=0.0, proposed=0.05), candidate=0.0001)
    assert r.status is GuardStatus.FIRED


def test_g11_reports_the_split_above_the_market_order_cap() -> None:
    # 5% of 1,000,000 at 100 is 500 NVDA; the Demo cap is 60 per market order: 9 legs.
    r = _g11(Leg(symbol=NVDA, current=0.0, proposed=0.05), equity="1000000")
    _passed(r)
    assert r.inputs["split_legs"] == 9


def test_every_ruling_carries_its_basis_and_symbol() -> None:
    r = _g11()
    assert r.guard is GuardId.G11_ELIGIBILITY
    assert r.symbol == NVDA
    assert "universe_probe.json" in r.basis
    assert _g10(None).symbol is None
    assert SPX in P.symbols
