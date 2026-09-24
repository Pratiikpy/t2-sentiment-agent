"""M17: the replica's ported logic agrees with the primary's modules on the same inputs.

The generated constants cannot drift (test_playbook_package.py). The logic written by hand against
them could, so each port is run beside the module it was ported from:

* the kernel: a seeded sweep of random books, quotes, breaker states, grounding reports and
  proposals through ``RiskKernel.rule`` and the replica's ``kernel.rule`` (and the protective
  rulings), comparing every approved weight, binding guard and guard status;
* the breaker: random assessment sequences through ``Breaker.assess`` and ``breaker.assess``;
* grounding, the decision contract, the feature formulas, the calendar (weekend phases, the US
  open through daylight saving, the heartbeats), symbol mentions and canonical JSON.

A difference found here is a defect in the port, not in the primary.
"""

import json
import math
import random
import sys
from collections.abc import Iterator, Mapping
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from types import ModuleType
from typing import Any

import pytest

from decision.support import build_snapshot, held_book
from helpers import T0
from kernel.kbuild import GROUNDED, LIMITS, PRICES, UNGROUNDED, UNIVERSE, spec
from playbook.pbuild import ROOT, build_package, load_package
from sentiment_agent.clock import ManualClock
from sentiment_agent.crowd.novelty import symbols_mentioned
from sentiment_agent.decision import contract, grounding
from sentiment_agent.decision.agent import proposed_weights
from sentiment_agent.events import schedule, triggers
from sentiment_agent.hashing import canonical_json
from sentiment_agent.kernel import guards
from sentiment_agent.kernel.breaker import Breaker
from sentiment_agent.kernel.kernel import RiskKernel
from sentiment_agent.perception import features
from sentiment_agent.policy import POLICY_V1
from sentiment_agent.types import (
    Activation,
    BookState,
    BreakerState,
    Candle,
    FundingPoint,
    GroundingReport,
    GuardId,
    KernelInputs,
    LlmDecision,
    Position,
    PriceSource,
    Quote,
    RulingContext,
)


@pytest.fixture(scope="module")
def replica(tmp_path_factory: pytest.TempPathFactory) -> Iterator[ModuleType]:
    """The built package, imported. The ported modules import nothing from ``getagent``; the
    package's SDK-facing modules do, so empty stand-in modules satisfy the import. No SDK call is
    made in this file."""
    package = build_package(tmp_path_factory.mktemp("parity"))
    patch = pytest.MonkeyPatch()
    stand_in = ModuleType("getagent")
    for name in ("data", "llm", "runtime", "trade"):
        setattr(stand_in, name, ModuleType(f"getagent.{name}"))
    patch.setitem(sys.modules, "getagent", stand_in)
    yield load_package(package, patch)
    patch.undo()


# ================================================================================================
# The kernel
# ================================================================================================

MOMENTS = (
    datetime(2026, 9, 23, 13, 0, tzinfo=UTC),  # Wednesday, open
    datetime(2026, 9, 25, 18, 30, tzinfo=UTC),  # Friday, inside the no-open buffer
    datetime(2026, 9, 25, 19, 50, tzinfo=UTC),  # Friday, pre-flatten
    datetime(2026, 9, 26, 12, 0, tzinfo=UTC),  # Saturday, frozen
    datetime(2026, 9, 28, 0, 5, tzinfo=UTC),  # Monday, just reopened
)


def _scenario(rng: random.Random) -> dict[str, Any]:
    at = rng.choice(MOMENTS) + timedelta(minutes=rng.randint(0, 9))
    symbols = rng.sample(UNIVERSE, rng.randint(2, 6))
    # A small book makes the venue minimums bind (G11); a normal one exercises everything else.
    equity = Decimal(rng.choice(["10000", "10000", "10000", "150"]))

    def scale(lo: float, hi: float) -> Decimal:
        return Decimal(str(round(rng.uniform(lo, hi), 6)))

    quotes: dict[str, dict[str, Decimal]] = {}
    moves: dict[str, float | None] = {}
    fresh: dict[str, datetime] = {}
    for s in symbols:
        mark = Decimal(PRICES[s][0])
        index_move = rng.choice([Decimal(0), Decimal("0.001"), Decimal("-0.001"), Decimal("0.035")])
        spread = Decimal(rng.choice(["1", "5", "25"])) / Decimal(10_000) * mark
        venue_last = mark * scale(0.9995, 1.0005)
        p99 = next(u.demo_live_gap_p99_bps for u in POLICY_V1.universe if u.symbol == s)
        gap = Decimal(str(rng.choice([0.0, p99 * 0.5, p99 * 1.5]))) / Decimal(10_000)
        data_last = venue_last * (1 + gap * rng.choice([1, -1]))
        quotes[s] = {
            "mark": mark,
            "index": mark * (1 + index_move),
            "bid": mark - spread / 2,
            "ask": mark + spread / 2,
            "venue_last": venue_last,
            "data_last": data_last,
        }
        moves[s] = rng.choice([None, 0.5, 30.0])
        fresh[s] = at - timedelta(seconds=rng.choice([0, 30, 200]))
    positions: dict[str, tuple[Decimal, Decimal, datetime]] = {}
    for s in symbols:
        if rng.random() < 0.5:
            weight = Decimal(str(round(rng.uniform(0.005, 0.06), 5))) * rng.choice([1, -1])
            mark = quotes[s]["mark"]
            positions[s] = (
                weight * equity / mark,
                mark * scale(0.97, 1.03),
                at - timedelta(hours=rng.uniform(0.5, 48)),
            )
    proposals: dict[str, float] = {}
    for s in symbols:
        if s in positions or rng.random() < 0.7:
            proposals[s] = rng.choice([0.0, round(rng.uniform(-0.07, 0.07), 6)])
    activation = rng.choice(
        [Activation.ACTIVE, Activation.ACTIVE, Activation.REDUCE_ONLY, Activation.HALTED]
    )
    trips = {
        Activation.ACTIVE: (),
        Activation.REDUCE_ONLY: ("losing_streak",),
        Activation.HALTED: ("llm_outage",),
    }
    return {
        "at": at,
        "symbols": symbols,
        "equity": equity,
        "peak": equity * scale(1.0, 1.06),
        "day_open": equity * scale(0.975, 1.02),
        "starting": equity * scale(0.98, 1.03),
        "fees_today": equity * scale(0, 0.001),
        "fees_extra": equity * scale(0, 0.0025),
        "losses": rng.randint(0, 5),
        "quotes": quotes,
        "moves": moves,
        "fresh": fresh,
        "positions": positions,
        "rebalances": {s: rng.randint(0, 3) for s in symbols if rng.random() < 0.3},
        "activation": activation,
        "trips": trips[activation],
        "grounding": {s: rng.choice([GROUNDED, GROUNDED, UNGROUNDED, None]) for s in symbols},
        "invalidation": {s: rng.random() < 0.2 for s in symbols},
        "proposals": proposals,
        "snapshot_at": at - timedelta(minutes=rng.choice([0, 0, 0, 20])),
    }


def _primary_quote(
    symbol: str, q: Mapping[str, Decimal], source: PriceSource, ts: datetime
) -> Quote:
    last = q["venue_last"] if source is PriceSource.DEMO else q["data_last"]
    return Quote(
        symbol=symbol,
        source=source,
        ts=ts,
        fetched_at=ts,
        last=last,
        mark=q["mark"],
        index=q["index"],
        bid=q["bid"],
        ask=q["ask"],
        funding_rate=None,
        open_interest=None,
        turnover_24h=None,
        price_change_24h=None,
    )


def _primary(sc: Mapping[str, Any]) -> tuple[BookState, KernelInputs, BreakerState]:
    at: datetime = sc["at"]
    positions = {
        s: Position(
            symbol=s,
            qty=qty,
            avg_entry=entry,
            opened_at=increased - timedelta(hours=1),
            last_increase_at=increased,
            realized_pnl=Decimal(0),
            fees_paid=Decimal(0),
            stop_price=None,
            stop_venue_id=None,
            last_decision_id="d0",
        )
        for s, (qty, entry, increased) in sc["positions"].items()
    }
    book = BookState(
        as_of=at,
        mark_source=PriceSource.DEMO,
        starting_equity=sc["starting"],
        equity=sc["equity"],
        peak_equity=sc["peak"],
        day_open_equity=sc["day_open"],
        positions=positions,
        marks={s: sc["quotes"][s]["mark"] for s in positions},
        fees_today=sc["fees_today"],
        fees_total=sc["fees_today"] + sc["fees_extra"],
        realized_total=Decimal(0),
        rebalances_today=dict(sc["rebalances"]),
        consecutive_losses=sc["losses"],
        activation=Activation.ACTIVE,
    )
    symbols = sc["symbols"]
    inputs = KernelInputs(
        at=at,
        demo_quotes={
            s: _primary_quote(s, sc["quotes"][s], PriceSource.DEMO, sc["fresh"][s]) for s in symbols
        },
        live_quotes={
            s: _primary_quote(s, sc["quotes"][s], PriceSource.LIVE, sc["fresh"][s]) for s in symbols
        },
        specs={s: spec(s, at=at) for s in symbols},
        demo_index_move_bps_3h=dict(sc["moves"]),
        snapshot_id="snap",
        snapshot_taken_at=sc["snapshot_at"],
    )
    breaker = BreakerState(
        activation=sc["activation"], since=at - timedelta(hours=2), trips=sc["trips"]
    )
    return book, inputs, breaker


def _replica(
    rkernel: ModuleType, rbreaker: ModuleType, sc: Mapping[str, Any]
) -> tuple[Any, Any, Any]:
    at: datetime = sc["at"]
    book = rkernel.BookView(
        equity=sc["equity"],
        starting_equity=sc["starting"],
        peak_equity=sc["peak"],
        day_open_equity=sc["day_open"],
        fees_today=sc["fees_today"],
        fees_total=sc["fees_today"] + sc["fees_extra"],
        consecutive_losses=sc["losses"],
        positions={
            s: rkernel.PositionView(qty=qty, avg_entry=entry, last_increase_at=increased)
            for s, (qty, entry, increased) in sc["positions"].items()
        },
        rebalances_today=dict(sc["rebalances"]),
    )
    symbols = sc["symbols"]
    quotes = {
        s: rkernel.Quote(
            symbol=s,
            mark=q["mark"],
            index=q["index"],
            last=q["data_last"],
            bid=q["bid"],
            ask=q["ask"],
            ts=sc["fresh"][s],
            fetched_at=sc["fresh"][s],
        )
        for s, q in sc["quotes"].items()
    }
    limits = {
        s: rkernel.Limits(
            min_qty=Decimal(LIMITS[s][0]),
            qty_step=Decimal(LIMITS[s][1]),
            min_amount=Decimal(LIMITS[s][3]),
            price_step=Decimal(LIMITS[s][2]),
        )
        for s in symbols
    }
    inputs = rkernel.Inputs(
        at=at,
        quotes=quotes,
        move_bps_3h=dict(sc["moves"]),
        limits=limits,
        snapshot_taken_at=sc["snapshot_at"],
        configured=tuple(UNIVERSE),
        venue_last={s: sc["quotes"][s]["venue_last"] for s in symbols},
        venue_is_data_layer=False,
    )
    breaker = rbreaker.BreakerState(
        activation=sc["activation"].value, since=at - timedelta(hours=2), trips=sc["trips"]
    )
    return book, inputs, breaker


def _statuses(rulings: Any) -> dict[str, str]:
    return {
        (r.guard.value if hasattr(r.guard, "value") else r.guard): (
            r.status.value if hasattr(r.status, "value") else r.status
        )
        for r in rulings
    }


def test_kernel_rulings_match_the_primary_on_a_seeded_sweep(replica: ModuleType) -> None:
    rkernel, rbreaker = replica.kernel, replica.breaker
    rng = random.Random(20260924)  # noqa: S311 - a reproducible sweep, not a secret
    compared = changed = 0
    bound: set[GuardId] = set()
    for _ in range(1200):
        sc = _scenario(rng)
        p_book, p_inputs, p_breaker = _primary(sc)
        r_book, r_inputs, r_breaker = _replica(rkernel, rbreaker, sc)
        grounding_map: dict[str, GroundingReport] = {
            s: g for s, g in sc["grounding"].items() if g is not None
        }
        primary = RiskKernel(POLICY_V1, ManualClock(sc["at"])).rule(
            proposed=sc["proposals"],
            book=p_book,
            inputs=p_inputs,
            context=RulingContext(
                decision_id="d1",
                protective_reason=None,
                grounding=grounding_map,
                invalidation_fired=sc["invalidation"],
            ),
            breaker=p_breaker,
        )
        ours = rkernel.rule(
            proposed=sc["proposals"],
            decision_ref="d1",
            book=r_book,
            inputs=r_inputs,
            breaker=r_breaker,
            grounding=grounding_map,
            invalidation_fired=sc["invalidation"],
        )
        assert ours.activation_after == primary.activation_after.value
        assert _statuses(ours.book_rulings) == _statuses(primary.book_rulings)
        mine = {i.symbol: i for i in ours.instruments}
        assert set(mine) == {i.symbol for i in primary.instruments}
        for inst in primary.instruments:
            other = mine[inst.symbol]
            compared += 1
            changed += inst.changed_by_kernel
            context = (sc["at"], inst.symbol, inst.proposed_weight, inst.current_weight)
            assert math.isclose(other.approved, inst.approved_weight, abs_tol=1e-12), context
            assert other.current == pytest.approx(inst.current_weight, abs=1e-12), context
            wanted = None if inst.binding_guard is None else inst.binding_guard.value
            assert other.binding_guard == wanted, context
            if inst.binding_guard is not None:
                bound.add(inst.binding_guard)
            assert _statuses(other.rulings) == _statuses(inst.rulings), context
    assert compared > 2000
    assert changed > 300, "the sweep must exercise the guards, not only pass-throughs"
    assert bound >= set(GuardId) - {GuardId.G4_STOP}, (
        f"guards never binding: {set(GuardId) - bound}"
    )


def test_protective_rulings_match_the_primary(replica: ModuleType) -> None:
    rkernel, rbreaker = replica.kernel, replica.breaker
    rng = random.Random(7)  # noqa: S311 - a reproducible sweep, not a secret
    acted = 0
    for _ in range(600):
        sc = _scenario(rng)
        p_book, p_inputs, p_breaker = _primary(sc)
        r_book, r_inputs, r_breaker = _replica(rkernel, rbreaker, sc)
        outage = rng.random() < 0.15
        primary = RiskKernel(POLICY_V1, ManualClock(sc["at"])).protective(
            book=p_book, inputs=p_inputs, breaker=p_breaker, llm_outage=outage
        )
        ours = rkernel.protective(
            book=r_book, inputs=r_inputs, breaker=r_breaker, llm_outage=outage
        )
        assert (ours is None) == (primary is None), sc["at"]
        if primary is None or ours is None:
            continue
        acted += 1
        assert primary.protective_reason is not None
        assert ours.protective_reason == primary.protective_reason.value
        assert ours.approved() == pytest.approx(
            {i.symbol: i.approved_weight for i in primary.instruments}, abs=1e-12
        )
    assert acted > 50


def test_breaker_assessments_match_the_primary(replica: ModuleType) -> None:
    rbreaker = replica.breaker
    rng = random.Random(11)  # noqa: S311 - a reproducible sweep, not a secret
    for _ in range(80):
        at = T0
        ours = rbreaker.BreakerState(activation="active", since=at)
        clock = ManualClock(at)
        primary = Breaker(POLICY_V1, clock)
        for _ in range(12):
            at = at + timedelta(minutes=rng.choice([15, 60, 600]))
            clock.set(at)
            equity = Decimal(str(round(rng.uniform(9500, 10200), 2)))
            peak = max(equity, Decimal(str(round(rng.uniform(9800, 10300), 2))))
            day_open = Decimal(str(round(rng.uniform(9700, 10150), 2)))
            losses = rng.randint(0, 5)
            outage = rng.random() < 0.15
            decided = not outage and rng.random() < 0.4
            snapshot_at = at - timedelta(minutes=rng.choice([0, 0, 20]))
            book = BookState(
                as_of=at,
                mark_source=PriceSource.DEMO,
                starting_equity=Decimal(10000),
                equity=equity,
                peak_equity=peak,
                day_open_equity=day_open,
                positions={},
                marks={},
                fees_today=Decimal(0),
                fees_total=Decimal(0),
                realized_total=Decimal(0),
                rebalances_today={},
                consecutive_losses=losses,
                activation=Activation.ACTIVE,
            )
            inputs = KernelInputs(
                at=at,
                demo_quotes={},
                live_quotes={},
                specs={},
                demo_index_move_bps_3h={},
                snapshot_id="s",
                snapshot_taken_at=snapshot_at,
            )
            p_state, _ = primary.assess(
                book, inputs=inputs, llm_outage=outage, decision_id="d" if decided else None
            )
            ours, _ = rbreaker.assess(
                ours,
                now=at,
                equity=float(equity),
                drawdown=float(equity / peak - 1),
                day_return=float(equity / day_open - 1),
                consecutive_losses=losses,
                llm_outage=outage,
                snapshot_taken_at=snapshot_at,
                valid_decision=decided,
            )
            assert ours.activation == p_state.activation.value, at
            assert ours.trips == p_state.trips, at
            assert ours.halted_until == p_state.halted_until, at


# ================================================================================================
# Grounding
# ================================================================================================

FACTS: dict[str, float] = {
    "BTCUSDT.mark": 83176.5,
    "BTCUSDT.funding_rate": 0.000125,
    "BTCUSDT.funding_z": 2.41,
    "BTCUSDT.oi_change_24h_pct": 7.84,
    "BTCUSDT.open_interest": 5_200_000_000.0,
    "BTCUSDT.retail_long_short_ratio": 1.87,
    "BTCUSDT.move_bps_3h": 31.2,
    "NVDAUSDT.mark": 222.84,
    "NVDAUSDT.ma20_distance_atr": -1.73,
    "NVDAUSDT.price_change_24h_pct": -3.2,
    "NVDAUSDT.spread_bps": 3.6,
    "book.drawdown_pct": -0.42,
    "kernel.per_name_max_pct": 5.0,
    "kernel.stop_loss_pct": 4.0,
    "mood.crypto_fear_greed": 78.0,
    "reference.zero": 0.0,
}

TEXTS = (
    "Funding at 0.0125% with a z of 2.41 and open interest up 7.84% in 24h.",
    "BTC funding 1.25 bps per interval, retail long/short 1.87, OI 5.2B.",
    "Mark 83176.5 against a 20-bar mean; stop 4% below entry; per-name cap 5%.",
    "NVDA fell 3.2% on 2026-09-23 at 13:30 UTC, S&P 500 and Nasdaq-100 flat.",
    "Fabricated: funding 0.0131%, z 2.9, open interest +9.1%, price 83900.",
    "Form 4 filed by an insider, 8-K on Sep 23; item x:1839001 and story-3f2a1c.",
    "The 3rd attempt over 14 bars and a 24-hour window; year 2026 is not a claim, $2030 is.",
    "Spread 3.6 bps, drawdown -0.42%, fear and greed 78, move 31.2 bps over 3h.",
    "Sign matters: -83176.5 is not the mark; −3.2% uses a Unicode minus and resolves.",
    "Fullwidth digits: ８３１７６.５ and a ratio of 1.9 that is close but not written precisely.",
    "",
)


def test_grounding_matches_the_primary_on_a_corpus(replica: ModuleType) -> None:
    rgrounding = replica.grounding
    tolerance = POLICY_V1.grounding_tolerance
    for text in TEXTS:
        primary = grounding.check(text, facts=FACTS, tolerance=tolerance)
        ours = rgrounding.check(text, facts=FACTS, tolerance=tolerance)
        assert [
            (f.raw, f.value, f.unit, f.context, f.resolved, f.source, f.known_value)
            for f in ours.figures
        ] == [
            (f.raw, f.value, f.unit, f.context, f.resolved, f.source, f.known_value)
            for f in primary.figures
        ], text
    assert any(
        not f.resolved for f in rgrounding.check(TEXTS[4], facts=FACTS, tolerance=tolerance).figures
    )


def test_grounding_per_target_matches_ground_decision(replica: ModuleType) -> None:
    rgrounding = replica.grounding
    raw = {
        "stance": "act",
        "targets": [
            {
                "symbol": "BTCUSDT",
                "target": -0.6,
                "thesis": "Funding z 2.41 and OI up 7.84%; NVDAUSDT down 3.2% too.",
                "invalidation": "BTCUSDT.funding_z back below reference.zero",
                "horizon_hours": 48,
                "crowd_belief": "moon at 120000",
                "our_view": "fade with 60% of a full size, confidence 55%",
                "confidence": 0.55,
                "evidence": ["BTCUSDT.funding_z"],
                "invalidation_triggered": False,
                "invalidation_evidence": None,
            },
            {
                "symbol": "NVDAUSDT",
                "target": 0.2,
                "thesis": "Stretched down 1.73 ATR; price 230 is invented.",
                "invalidation": "spread above 3.6 bps",
                "horizon_hours": 24,
                "crowd_belief": "capitulation",
                "our_view": "small long",
                "confidence": 0.4,
                "evidence": [],
                "invalidation_triggered": False,
                "invalidation_evidence": None,
            },
        ],
        "rejected_alternatives": [],
        "mandate_response": "deploy a little",
        "flat_reasons": [],
        "summary": "fade and buy",
    }
    decision = LlmDecision.model_validate(raw)
    primary = grounding.ground_decision(decision, FACTS, POLICY_V1.grounding_tolerance)
    ours = rgrounding.ground_targets(raw["targets"], FACTS, POLICY_V1.grounding_tolerance)
    assert set(ours) == set(primary)
    for symbol, report in primary.items():
        assert [(f.raw, f.resolved, f.source, f.context) for f in ours[symbol].figures] == [
            (f.raw, f.resolved, f.source, f.context) for f in report.figures
        ], symbol


# ================================================================================================
# The decision contract
# ================================================================================================


def _answer(**update: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "stance": "act",
        "targets": [
            {
                "symbol": "NVDAUSDT",
                "target": -0.3,
                "thesis": "crowded",
                "invalidation": "funding normalises",
                "horizon_hours": 48,
                "crowd_belief": "up only",
                "our_view": "fade",
                "confidence": 0.5,
            },
            {
                "symbol": "BTCUSDT",
                "target": 0.8,
                "thesis": "hold",
                "invalidation": "x",
                "horizon_hours": 24,
                "crowd_belief": "fear",
                "our_view": "keep",
                "confidence": 0.6,
            },
        ],
        "rejected_alternatives": [{"action": "flat", "reason": "edge"}],
        "mandate_response": "some",
        "summary": "s",
    }
    body.update(update)
    return body


def _variants() -> list[tuple[str, str]]:
    base = _answer()
    cases: list[tuple[str, Any]] = [
        ("valid", base),
        ("missing held symbol", _answer(targets=[base["targets"][0]])),
        (
            "unknown symbol",
            _answer(targets=[*base["targets"], {**base["targets"][0], "symbol": "PEPEUSDT"}]),
        ),
        (
            "excluded symbol",
            _answer(targets=[*base["targets"], {**base["targets"][0], "symbol": "ETHUSDT"}]),
        ),
        (
            "short horizon",
            _answer(targets=[{**base["targets"][0], "horizon_hours": 12}, base["targets"][1]]),
        ),
        ("duplicate symbol", _answer(targets=[*base["targets"], base["targets"][0]])),
        ("flat with a target", _answer(stance="flat_with_reasons", flat_reasons=["none"])),
        (
            "flat without reasons",
            _answer(
                stance="flat_with_reasons", targets=[{**t, "target": 0} for t in base["targets"]]
            ),
        ),
        (
            "flat with reasons",
            _answer(
                stance="flat_with_reasons",
                flat_reasons=["no edge"],
                targets=[{**t, "target": 0} for t in base["targets"]],
            ),
        ),
        (
            "flat with blank reason",
            _answer(
                stance="flat_with_reasons",
                flat_reasons=["  "],
                targets=[{**t, "target": 0} for t in base["targets"]],
            ),
        ),
        (
            "hold flips a side",
            _answer(
                stance="hold", targets=[{**base["targets"][0], "target": 0.3}, base["targets"][1]]
            ),
        ),
        ("hold keeps sides", _answer(stance="hold")),
        (
            "hold opens new",
            _answer(
                stance="hold",
                targets=[*base["targets"], {**base["targets"][0], "symbol": "TSLAUSDT"}],
            ),
        ),
        (
            "invalidation adds",
            _answer(
                targets=[
                    base["targets"][0],
                    {
                        **base["targets"][1],
                        "target": 1.0,
                        "invalidation_triggered": True,
                        "invalidation_evidence": "fired",
                    },
                ]
            ),
        ),
        (
            "invalidation flips",
            _answer(
                targets=[
                    {
                        **base["targets"][0],
                        "target": 0.4,
                        "invalidation_triggered": True,
                        "invalidation_evidence": "fired",
                    },
                    base["targets"][1],
                ]
            ),
        ),
        (
            "invalidation not held",
            _answer(
                targets=[
                    *base["targets"],
                    {
                        **base["targets"][0],
                        "symbol": "TSLAUSDT",
                        "invalidation_triggered": True,
                        "invalidation_evidence": "x",
                    },
                ]
            ),
        ),
        (
            "invalidation without evidence",
            _answer(
                targets=[{**base["targets"][0], "invalidation_triggered": True}, base["targets"][1]]
            ),
        ),
        ("extra field", {**base, "note": "hi"}),
        (
            "extra target field",
            _answer(targets=[{**base["targets"][0], "size": 1}, base["targets"][1]]),
        ),
        ("missing field", {k: v for k, v in base.items() if k != "summary"}),
        (
            "string target",
            _answer(targets=[{**base["targets"][0], "target": "-0.3"}, base["targets"][1]]),
        ),
        (
            "float horizon",
            _answer(targets=[{**base["targets"][0], "horizon_hours": 48.0}, base["targets"][1]]),
        ),
        (
            "integer target",
            _answer(targets=[{**base["targets"][0], "target": -1}, base["targets"][1]]),
        ),
        (
            "target out of range",
            _answer(targets=[{**base["targets"][0], "target": -1.5}, base["targets"][1]]),
        ),
        (
            "confidence out of range",
            _answer(targets=[{**base["targets"][0], "confidence": 1.2}, base["targets"][1]]),
        ),
        (
            "bool as number",
            _answer(targets=[{**base["targets"][0], "confidence": True}, base["targets"][1]]),
        ),
        (
            "blank thesis",
            _answer(targets=[{**base["targets"][0], "thesis": ""}, base["targets"][1]]),
        ),
        (
            "whitespace thesis",
            _answer(targets=[{**base["targets"][0], "thesis": " "}, base["targets"][1]]),
        ),
        ("unknown stance", _answer(stance="buy")),
        ("act all zero", _answer(targets=[{**t, "target": 0} for t in base["targets"]])),
    ]
    rendered = [(name, json.dumps(body)) for name, body in cases]
    rendered += [
        ("fenced", "```json\n" + json.dumps(base) + "\n```"),
        ("prose wrapped", "Here is my decision: " + json.dumps(base) + " Thanks."),
        ("not an object", "[1, 2]"),
        ("empty", "   "),
        ("broken", '{"stance": "act", '),
    ]
    return rendered


def test_decision_contract_matches_the_primary(replica: ModuleType) -> None:
    rdecision = replica.decision
    book = held_book()
    snapshot = build_snapshot(book=book)
    held = {s: (1 if p.qty > 0 else -1) for s, p in book.positions.items() if not p.is_flat}
    facts = rdecision.BookFacts(
        held=held,
        weights={s: book.weight(s) for s in held},
        configured=POLICY_V1.symbols,
    )
    for name, content in _variants():
        try:
            primary = contract.parse_decision(
                content, book=book, snapshot=snapshot, policy=POLICY_V1
            )
            primary_complaints: tuple[str, ...] | None = None
        except contract.DecisionInvalid as invalid:
            primary = None
            primary_complaints = invalid.complaints
        try:
            ours = rdecision.parse_decision(content, facts)
            our_complaints: tuple[str, ...] | None = None
        except rdecision.DecisionInvalid as invalid:
            ours = None
            our_complaints = invalid.complaints
        assert (ours is None) == (primary is None), (name, primary_complaints, our_complaints)
        if primary is not None and ours is not None:
            assert ours == json.loads(primary.model_dump_json()), name
            proposal = rdecision.proposed_weights(ours, facts)
            assert proposal == pytest.approx(proposed_weights(primary, POLICY_V1, book)), name
        elif (
            primary_complaints is not None
            and our_complaints is not None
            and name
            in (
                "missing held symbol",
                "unknown symbol",
                "excluded symbol",
                "short horizon",
                "hold flips a side",
                "hold opens new",
                "invalidation adds",
                "invalidation not held",
                "act all zero",
            )
        ):
            assert set(our_complaints) == set(primary_complaints), name


# ================================================================================================
# Features, calendar, mentions, canonical JSON
# ================================================================================================


def _candles(
    rng: random.Random, count: int, *, gap_at: int | None = None
) -> list[tuple[datetime, float, float, float, float]]:
    start = datetime(2026, 9, 20, tzinfo=UTC)
    out = []
    price = 100.0
    t = start
    for i in range(count):
        t = t + timedelta(hours=2 if gap_at == i else 1)
        o = price
        price = max(1.0, price * (1 + rng.uniform(-0.01, 0.01)))
        high = max(o, price) * (1 + rng.uniform(0, 0.004))
        low = min(o, price) * (1 - rng.uniform(0, 0.004))
        out.append((t, round(o, 4), round(high, 4), round(low, 4), round(price, 4)))
    return out


def test_feature_formulas_match_the_primary(replica: ModuleType) -> None:
    rperception = replica.perception
    rng = random.Random(3)  # noqa: S311 - a reproducible sweep, not a secret
    for count in (5, 14, 15, 20, 30):
        for gap in (None, count - 2):
            rows = _candles(rng, count, gap_at=gap)
            ours = [rperception.Candle(t, o, h, low, c) for t, o, h, low, c in rows]
            index = [
                Candle(
                    symbol="X",
                    source=PriceSource.DEMO,
                    kind="index",
                    interval="1H",
                    open_time=t,
                    open=Decimal(str(o)),
                    high=Decimal(str(h)),
                    low=Decimal(str(low)),
                    close=Decimal(str(c)),
                    volume=None,
                )
                for t, o, h, low, c in rows
            ]
            assert rperception.move_bps_3h(ours) == pytest.approx(
                features.index_move_bps_3h(index), rel=1e-9
            )
            live = [c.model_copy(update={"kind": "market"}) for c in index]
            wanted = features.ma_distance_atr(live)
            got = rperception.ma_distance_atr(ours)
            assert (got is None) == (wanted is None)
            if got is not None and wanted is not None:
                assert got == pytest.approx(wanted, rel=1e-9)
    lookback = POLICY_V1.triggers.funding_z_lookback_settlements
    for n in (lookback - 1, lookback, lookback + 30):
        history = [
            (
                datetime(2026, 8, 1, tzinfo=UTC) + timedelta(hours=8 * i),
                rng.uniform(-0.0003, 0.0004),
            )
            for i in range(n)
        ]
        current = rng.uniform(-0.0005, 0.0008)
        points = [
            FundingPoint(symbol="BTCUSDT", source=PriceSource.LIVE, ts=t, rate=Decimal(str(r)))
            for t, r in history
        ]
        rounded = [(t, float(Decimal(str(r)))) for t, r in history]
        wanted_z = features.funding_z(points, current, lookback)
        got_z = rperception.funding_z(rounded, current)
        assert (got_z is None) == (wanted_z is None), n
        if got_z is not None and wanted_z is not None:
            assert got_z == pytest.approx(wanted_z, rel=1e-9)
    now = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    series = [
        (now - timedelta(hours=h, minutes=rng.choice([0, 0, 2, 9])), rng.uniform(49_000, 51_000))
        for h in range(0, 800)
    ]
    series.sort()
    for hours in (1, 24):
        assert rperception.oi_change_pct(series, hours=hours, at=now) == features.oi_change_pct(
            series, hours=hours, at=now
        )
    assert rperception.hourly_oi_changes_pct(series) == pytest.approx(
        triggers.hourly_oi_changes_pct(series)
    )
    threshold = rperception.oi_jump_threshold(series)
    frozen = triggers.oi_jump_thresholds(
        {"BTCUSDT": series[:-1]}, quantile=POLICY_V1.triggers.oi_jump_quantile
    )
    assert threshold == pytest.approx(frozen["BTCUSDT"])
    for value in range(0, 101):
        assert rperception.fear_greed_band(float(value)) == features.fear_greed_band(
            value, POLICY_V1
        )
        assert replica.triggers.band_of(float(value)) == triggers.band_of(value, POLICY_V1)


def test_calendar_matches_the_primary(replica: ModuleType) -> None:
    rsessions = replica.sessions
    rng = random.Random(5)  # noqa: S311 - a reproducible sweep, not a secret
    start = datetime(2026, 1, 1, tzinfo=UTC)
    for _ in range(4000):
        at = start + timedelta(minutes=rng.randint(0, 60 * 24 * 365 * 3))
        assert rsessions.weekend_phase(at) == guards.weekend_phase(at, POLICY_V1.weekend), at
    day = date(2026, 1, 1)
    while day < date(2031, 1, 1):
        assert rsessions.us_open_utc(day) == schedule.us_open_utc(day, POLICY_V1), day
        day += timedelta(days=1)
    for _ in range(300):
        lo = start + timedelta(minutes=rng.randint(0, 60 * 24 * 365 * 2))
        hi = lo + timedelta(minutes=rng.randint(1, 60 * 24 * 3))
        ours = [(kind, at) for kind, at in rsessions.heartbeats_between(lo, hi)]
        wanted = [
            (t.kind.value, t.fired_at) for t in schedule.heartbeats_between(lo, hi, POLICY_V1)
        ]
        assert ours == wanted, (lo, hi)


def test_symbol_mentions_match_the_primary(replica: ModuleType) -> None:
    rmentions = replica.mentions
    corpus = json.loads(
        (ROOT / "tests" / "fixtures" / "crowd" / "live_headlines.json").read_text("utf-8")
    )
    texts = [str(h.get("title", h)) if isinstance(h, dict) else str(h) for h in corpus["headlines"]]
    texts += [
        "$NVDA to the moon, nvidia beats; Meta-analysis says nothing about META",
        "BACK IN THE HOOD with $hood and Robinhood; THIS COIN IS GOING UP; Coinbase lists",
        "S&P 500 and s & p 500, Nasdaq-100, QQQ, SPY but not spy; Apple vs apple; Amazon",
        "BTC/USDT perp, NVDA-USDT, TSLAUSDT.P, microstrategy, Circle Internet, sandisk",
        "zero​width $NV​DA and fullwidth ＄ＮＶＤＡ",
    ]
    universe = POLICY_V1.symbols
    for text in texts:
        assert rmentions.symbols_mentioned(text, universe) == symbols_mentioned(text, universe), (
            text
        )


def test_canonical_json_is_the_primary_form(replica: ModuleType) -> None:
    rcanonical = replica.canonical
    rng = random.Random(9)  # noqa: S311 - a reproducible sweep, not a secret
    for _ in range(300):
        value = {
            "a": [rng.random(), rng.randint(-5, 5), None, True, "é→中"],
            "d": Decimal(str(round(rng.uniform(-1000, 1000), rng.randint(0, 6)))),
            "t": datetime(2026, 9, 24, rng.randint(0, 23), rng.randint(0, 59), tzinfo=UTC),
            "n": {"z": (1, 2), "b": {"x": "y"}},
        }
        assert rcanonical.canonical(value).encode("utf-8") == canonical_json(value)
    with pytest.raises(ValueError, match="NaN"):
        rcanonical.canonical({"x": float("nan")})
