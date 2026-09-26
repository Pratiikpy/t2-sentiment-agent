# The minimum-of-ceilings rule is ported from ARGUS ``src/argus/agents/desk.py``
# ``ConstitutionPolicy.rule`` (lines 1093-1410; MIT, same author) at commit
# 3dec6baf9dfa7be37c7b452e26a9b139df252f75, sha256
# 86fc3430b06e3fb232eb32987fa195318a739f3b297127b4142cef6dd57b7ca8. The ARGUS repository is
# private, so its licence and every pin are recorded in third_party/argus/PROVENANCE.md.
# Taken: every gate contributes a ceiling and the binding one is the minimum of all of them, never
# the first encountered; the binding gate's reason names every other ceiling that was computed; the
# result is re-checked against every ceiling rather than only the one that bound. Changed: ceilings
# are on |weight| per instrument, not on an order's unit quantity (ARGUS found four of its gates
# comparing units against dollars, desk.py 834-848; a weight has no unit to confuse); a missing
# input is a fail-closed ceiling, not a skipped gate; and a whole book is ruled at once so the
# gross cap can be shared across instruments.
"""The risk kernel: the model proposes, the kernel can only reduce.

**The invariant** (DESIGN.md §10.1). For every instrument, with ``reference`` the proposed weight
(or the current weight when nothing was proposed): the approved weight has the sign of the reference
or is zero, and its magnitude is at most the reference's. ``InstrumentRuling`` refuses construction
otherwise, and a seeded sweep in ``tests/kernel/test_invariants.py`` drives tens of thousands of
random books through this module to show it is never violated.

**Evaluate all, bind the minimum** (DESIGN.md §10.2). Every applied guard is evaluated on every
instrument; none short-circuits. Each contributes a ceiling on ``|weight|`` (a forced exit is a
ceiling of zero), and the approved weight is the reference clipped to the smallest. The binding
guard's reason names every other guard that was also below the request, so a looser rule can never
hide a tighter one. Three passes make this well defined when two guards depend on the others:

1. every guard except G3 and G11's sizing, per instrument;
2. G3: the per-name cap, and the gross cap shared across instruments by :func:`allocate_gross` on
   what pass 1 left;
3. G11's venue-minimum check on the weight passes 1 and 2 left, since whether an increase is large
   enough to be an order depends on how large it ended up.

**No decision, no increase.** A proposal may only be ruled with a model decision id. Without one
(``protective``), the reference is the current weight, so nothing can add exposure by construction,
not by a guard's good behaviour.

**Time** is the kernel's clock: weekend phases, the minimum hold and every staleness check are
measured at ``clock.now()``, and the ruling is stamped with it. Replays and simulations drive a
``ManualClock``; a ruling is a pure function of its arguments and that instant, and its id is the
hash of its content.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Final

from sentiment_agent.hashing import content_hash
from sentiment_agent.kernel.breaker import book_conditions, most_severe
from sentiment_agent.kernel.guards import (
    EPS,
    BookCap,
    GrossAllocation,
    Leg,
    allocate_caps,
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
)
from sentiment_agent.kernel.planner import current_weight, planning_price
from sentiment_agent.types import (
    ALL_GUARDS,
    Activation,
    BookState,
    BreakerState,
    Clock,
    GroundingReport,
    GuardId,
    GuardRuling,
    GuardStatus,
    InstrumentRuling,
    KernelInputs,
    KernelRuling,
    Policy,
    ProtectiveReason,
    RulingContext,
)

GUARD_ORDER: Final[tuple[GuardId, ...]] = tuple(GuardId)
"""Canonical order: rulings are listed, and ties between equal ceilings broken, in this order."""

PROTECTIVE_GUARDS: Final[frozenset[GuardId]] = frozenset(
    {
        GuardId.G1_VENUE_INTEGRITY,
        GuardId.G2_WEEKEND_FREEZE,
        GuardId.G5_DAILY_KILL,
        GuardId.G10_BREAKER,
    }
)
"""The guards the 60-second protective loop applies (DESIGN.md §10.5): kill, weekend, venue
integrity, breaker and outage. They are the only ones that can force an exit of a held position;
the rest only refuse increases, which a hold never is."""

_PROTECTIVE_PRIORITY: Final[tuple[ProtectiveReason, ...]] = (
    ProtectiveReason.DAILY_KILL,
    ProtectiveReason.LLM_OUTAGE,
    ProtectiveReason.BREAKER,
    ProtectiveReason.VENUE_INTEGRITY,
    ProtectiveReason.WINDOW_END,
    ProtectiveReason.WEEKEND_FREEZE,
)
"""When several protective causes act at once, the ruling is filed under the broadest: book-wide
causes before per-instrument ones."""


class KernelError(ValueError):
    """The kernel was asked to rule on something it must not rule on."""


def ruling_hash(ruling: KernelRuling) -> str:
    """``content_hash`` of the ruling with its ``ruling_id`` set to ``""``. The id of a kernel-built
    ruling is this hash, so a ruling changed after issue no longer matches its id (the approval
    minter checks). Hashing the model rather than a dump of it keeps serialisation in
    pydantic-core: a ruling carries a basis and inputs for every guard on every instrument."""
    blank = ruling if ruling.ruling_id == "" else ruling.model_copy(update={"ruling_id": ""})
    return content_hash(blank)


def _effective_ceiling(g: GuardRuling) -> float | None:
    if g.forces_exit:
        return 0.0
    return g.ceiling_abs_weight


@dataclass(frozen=True, slots=True)
class _Scene:
    """Everything one ruling reads, gathered once."""

    now: datetime
    book: BookState
    inputs: KernelInputs
    grounding: Mapping[str, GroundingReport]
    invalidation_fired: Mapping[str, bool]
    breaker: BreakerState
    applied: frozenset[GuardId]
    llm_outage: bool


class RiskKernel:
    """Rules on model proposals (:meth:`rule`) and on the book between decisions
    (:meth:`protective`). Holds no state of its own: the breaker's state is passed in."""

    def __init__(self, policy: Policy, clock: Clock) -> None:
        self._policy = policy
        self._clock = clock

    @property
    def policy(self) -> Policy:
        return self._policy

    # --------------------------------------------------------------------------------------------
    # Public entry points
    # --------------------------------------------------------------------------------------------

    def rule(
        self,
        *,
        proposed: Mapping[str, float] | None,
        book: BookState,
        inputs: KernelInputs,
        context: RulingContext,
        breaker: BreakerState,
        guards: frozenset[GuardId] = ALL_GUARDS,
    ) -> KernelRuling:
        """Rule on a model decision's proposed weights.

        Every symbol proposed, and every symbol held, gets an ``InstrumentRuling``. A held symbol
        the proposal does not name is ruled as a hold (reference = current weight); the decision
        contract makes every held symbol addressed, so this arises only for malformed callers.
        ``guards`` narrows which guards apply (baseline and rival arms use ``VENUE_GUARDS``).

        Raises :class:`KernelError` for a proposal without a decision id, a context that names both
        or neither of a decision and a protective reason, or a non-finite proposed weight.
        """
        if (context.decision_id is None) == (context.protective_reason is None):
            raise KernelError("a ruling answers either a model decision or a protective reason")
        if proposed is not None and context.decision_id is None:
            raise KernelError("a proposal can only be ruled under a model decision id")
        if proposed is not None:
            bad = sorted(s for s, w in proposed.items() if not math.isfinite(w))
            if bad:
                raise KernelError(f"non-finite proposed weight for {', '.join(bad)}")
        scene = _Scene(
            now=self._clock.now(),
            book=book,
            inputs=inputs,
            grounding=context.grounding,
            invalidation_fired=context.invalidation_fired,
            breaker=breaker,
            applied=frozenset(guards),
            llm_outage=False,
        )
        symbols = sorted(set(proposed or {}) | _held(book))
        legs = [
            Leg(
                symbol=s,
                current=current_weight(s, book, inputs),
                proposed=None if proposed is None else proposed.get(s),
            )
            for s in symbols
        ]
        book_rulings, instruments = self._evaluate(legs, scene)
        return self._seal(
            scene,
            decision_id=context.decision_id,
            protective_reason=context.protective_reason,
            book_rulings=book_rulings,
            instruments=instruments,
        )

    def protective(
        self,
        *,
        book: BookState,
        inputs: KernelInputs,
        breaker: BreakerState,
        llm_outage: bool = False,
    ) -> KernelRuling | None:
        """The 60-second check between decisions: daily kill, weekend pre-flatten and freeze, venue
        integrity exits, breaker halts and model outage. Every held position is ruled as a hold, so
        the only possible outcome is an exit. Returns ``None`` when nothing needs to change."""
        held = sorted(_held(book))
        if not held:
            return None
        scene = _Scene(
            now=self._clock.now(),
            book=book,
            inputs=inputs,
            grounding={},
            invalidation_fired={},
            breaker=breaker,
            applied=PROTECTIVE_GUARDS,
            llm_outage=llm_outage,
        )
        legs = [Leg(symbol=s, current=current_weight(s, book, inputs)) for s in held]
        book_rulings, instruments = self._evaluate(legs, scene)
        acting = [i for i in instruments if i.changed_by_kernel]
        if not acting:
            return None
        return self._seal(
            scene,
            decision_id=None,
            protective_reason=_protective_reason(acting, llm_outage=llm_outage),
            book_rulings=book_rulings,
            instruments=instruments,
        )

    # --------------------------------------------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------------------------------------------

    def _evaluate(
        self, legs: Sequence[Leg], scene: _Scene
    ) -> tuple[tuple[GuardRuling, ...], tuple[InstrumentRuling, ...]]:
        policy = self._policy
        applied = scene.applied
        # Pass 1: every guard but G3 and G11's sizing.
        first: dict[str, dict[GuardId, GuardRuling]] = {
            leg.symbol: self._first_pass(leg, scene) for leg in legs
        }
        # Pass 2: G3, per name and gross, on what pass 1 left.
        candidates: dict[str, float] = {}
        for leg in legs:
            ceilings = [
                c for g in first[leg.symbol].values() if (c := _effective_ceiling(g)) is not None
            ]
            if GuardId.G3_SIZE in applied:
                ceilings.append(policy.per_name_max)
            candidates[leg.symbol] = min([abs(leg.reference), *ceilings])
        allocation: GrossAllocation | None = None
        if GuardId.G3_SIZE in applied:
            allocation = allocate_gross(
                {leg.symbol: (candidates[leg.symbol], leg.hold_ceiling) for leg in legs},
                policy.gross_max,
            )
        # Pass 2b: the net and cluster caps (run2-a3) on what the gross cap left.
        book_caps: dict[str, tuple[float, str]] = {}
        evaluated_caps: list[BookCap] = []
        if allocation is not None and (policy.net_max < 1.0 or policy.cluster_caps):
            after_gross = {
                leg.symbol: (
                    min(
                        candidates[leg.symbol],
                        allocation.ceilings.get(leg.symbol, candidates[leg.symbol]),
                    ),
                    leg.hold_ceiling,
                    1 if leg.reference > 0 else -1 if leg.reference < 0 else 0,
                )
                for leg in legs
            }
            book_caps, evaluated_caps = allocate_caps(after_gross, policy)
        # Pass 3: G11's venue minimums on the weight that is left, then bind.
        instruments: list[InstrumentRuling] = []
        for leg in legs:
            rulings = dict(first[leg.symbol])
            candidate = candidates[leg.symbol]
            if allocation is not None:
                gross_ceiling = allocation.ceilings.get(leg.symbol)
                rulings[GuardId.G3_SIZE] = g3_size(
                    leg,
                    gross_ceiling=gross_ceiling,
                    policy=policy,
                    gross_note=allocation.note() if allocation.binds else "",
                    book_cap=book_caps.get(leg.symbol),
                )
                if gross_ceiling is not None:
                    candidate = min(candidate, gross_ceiling)
                if leg.symbol in book_caps:
                    candidate = min(candidate, book_caps[leg.symbol][0])
            if GuardId.G11_ELIGIBILITY in applied:
                rulings[GuardId.G11_ELIGIBILITY] = self._g11(leg, scene, candidate)
            instruments.append(_bind(leg, [rulings[g] for g in GUARD_ORDER if g in rulings]))
        return self._book_rulings(scene, allocation, evaluated_caps), tuple(instruments)

    def _first_pass(self, leg: Leg, scene: _Scene) -> dict[GuardId, GuardRuling]:
        policy = self._policy
        applied = scene.applied
        book, inputs, now = scene.book, scene.inputs, scene.now
        symbol = leg.symbol
        entry = policy.entry(symbol)
        demo = inputs.demo_quotes.get(symbol)
        out: dict[GuardId, GuardRuling] = {}
        if GuardId.G1_VENUE_INTEGRITY in applied:
            out[GuardId.G1_VENUE_INTEGRITY] = g1_venue_integrity(
                leg,
                entry=entry,
                demo=demo,
                live=inputs.live_quotes.get(symbol),
                index_move_bps_3h=inputs.demo_index_move_bps_3h.get(symbol),
                at=now,
                policy=policy,
            )
        if GuardId.G2_WEEKEND_FREEZE in applied:
            out[GuardId.G2_WEEKEND_FREEZE] = g2_weekend_freeze(
                leg, asset_class=entry.asset_class if entry else None, at=now, policy=policy
            )
        if GuardId.G4_STOP in applied:
            out[GuardId.G4_STOP] = g4_stop(
                leg, spec=inputs.specs.get(symbol), demo=demo, policy=policy
            )
        if GuardId.G5_DAILY_KILL in applied:
            out[GuardId.G5_DAILY_KILL] = g5_daily_kill(leg, book=book, policy=policy)
        if GuardId.G6_TURNOVER in applied:
            out[GuardId.G6_TURNOVER] = g6_turnover(
                leg,
                position=book.positions.get(symbol),
                rebalances_today=book.rebalances_today.get(symbol, 0),
                invalidation_declared=scene.invalidation_fired.get(symbol, False),
                at=now,
                policy=policy,
            )
        if GuardId.G7_FEE_BUDGET in applied:
            out[GuardId.G7_FEE_BUDGET] = g7_fee_budget(leg, book=book, policy=policy)
        if GuardId.G8_TAKER_ONLY in applied:
            out[GuardId.G8_TAKER_ONLY] = g8_taker_only(leg, demo=demo, policy=policy)
        if GuardId.G9_GROUNDING in applied:
            out[GuardId.G9_GROUNDING] = g9_grounding(
                leg, report=scene.grounding.get(symbol), policy=policy
            )
        if GuardId.G10_BREAKER in applied:
            out[GuardId.G10_BREAKER] = g10_breaker(
                leg,
                breaker=scene.breaker,
                book=book,
                llm_outage=scene.llm_outage,
                demo=demo,
                snapshot_taken_at=inputs.snapshot_taken_at,
                at=now,
                policy=policy,
                venue_unreconciled=inputs.venue_unreconciled,
            )
        if GuardId.G11_ELIGIBILITY in applied:
            # Eligibility only (universe, online, spec): a candidate of zero means there is no
            # increase to size yet. The sizing check runs in pass 3 on the weight every other guard
            # leaves; sizing here would feed G11's own ceiling into the candidate it then sizes,
            # and the final ruling would never see the increase it refused.
            out[GuardId.G11_ELIGIBILITY] = self._g11(leg, scene, 0.0)
        return out

    def _g11(self, leg: Leg, scene: _Scene, candidate: float | None) -> GuardRuling:
        book, inputs = scene.book, scene.inputs
        position = book.positions.get(leg.symbol)
        return g11_eligibility(
            leg,
            policy=self._policy,
            spec=inputs.specs.get(leg.symbol),
            position_qty=position.qty if position is not None else Decimal(0),
            price=planning_price(leg.symbol, book, inputs),
            equity=book.equity,
            candidate_abs_weight=candidate,
        )

    def _book_rulings(
        self,
        scene: _Scene,
        allocation: GrossAllocation | None,
        book_caps: Sequence[BookCap] = (),
    ) -> tuple[GuardRuling, ...]:
        policy, book, applied = self._policy, scene.book, scene.applied
        out: list[GuardRuling] = []
        if allocation is not None:
            out.append(
                GuardRuling(
                    guard=GuardId.G3_SIZE,
                    symbol=None,
                    status=GuardStatus.FIRED if allocation.binds else GuardStatus.PASSED,
                    ceiling_abs_weight=policy.gross_max,
                    reason=allocation.note(),
                    basis=next(b for b in policy.guard_bases if b.guard is GuardId.G3_SIZE).basis,
                    inputs={
                        "gross_max": allocation.cap,
                        "gross_requested": allocation.requested,
                        "gross_held": allocation.held,
                        "gross_new": allocation.new,
                    },
                )
            )
        g3_basis = next(b for b in policy.guard_bases if b.guard is GuardId.G3_SIZE).basis
        for cap in book_caps:
            out.append(
                GuardRuling(
                    guard=GuardId.G3_SIZE,
                    symbol=None,
                    status=GuardStatus.FIRED if cap.allocation.binds else GuardStatus.PASSED,
                    ceiling_abs_weight=cap.allocation.cap,
                    reason=f"{cap.label}: "
                    + cap.allocation.note().replace(
                        "gross", "side" if "net" in cap.label else "cluster gross"
                    ),
                    basis=g3_basis,
                    inputs={
                        "cap": cap.allocation.cap,
                        "requested": cap.allocation.requested,
                        "held": cap.allocation.held,
                        "new": cap.allocation.new,
                    },
                )
            )
        if GuardId.G5_DAILY_KILL in applied:
            out.append(g5_daily_kill(None, book=book, policy=policy))
        if GuardId.G7_FEE_BUDGET in applied:
            out.append(g7_fee_budget(None, book=book, policy=policy))
        if GuardId.G10_BREAKER in applied:
            out.append(
                g10_breaker(
                    None,
                    breaker=scene.breaker,
                    book=book,
                    llm_outage=scene.llm_outage,
                    demo=None,
                    snapshot_taken_at=scene.inputs.snapshot_taken_at,
                    at=scene.now,
                    policy=policy,
                    venue_unreconciled=scene.inputs.venue_unreconciled,
                )
            )
        return tuple(out)

    def _seal(
        self,
        scene: _Scene,
        *,
        decision_id: str | None,
        protective_reason: ProtectiveReason | None,
        book_rulings: tuple[GuardRuling, ...],
        instruments: tuple[InstrumentRuling, ...],
    ) -> KernelRuling:
        body = KernelRuling(
            ruling_id="",
            at=scene.now,
            decision_id=decision_id,
            protective_reason=protective_reason,
            activation_before=scene.breaker.activation,
            activation_after=self._activation_after(scene, book_rulings),
            book_rulings=book_rulings,
            instruments=instruments,
            guards_applied=tuple(g for g in GUARD_ORDER if g in scene.applied),
        )
        return body.model_copy(update={"ruling_id": ruling_hash(body)})

    def _activation_after(self, scene: _Scene, book_rulings: Iterable[GuardRuling]) -> Activation:
        """The activation this ruling acted under: the breaker's state, escalated by what the book
        demands now (G10) and by a daily kill (G5). The :class:`Breaker` owns persistence; this
        records what the ruling itself enforced."""
        after = scene.breaker.activation
        if GuardId.G10_BREAKER in scene.applied:
            demanded, _ = book_conditions(
                scene.book,
                llm_outage=scene.llm_outage,
                policy=self._policy,
                include_daily_kill=False,
            )
            after = most_severe(after, demanded)
        if any(g.guard is GuardId.G5_DAILY_KILL and g.forces_exit for g in book_rulings):
            after = Activation.HALTED
        return after


def _held(book: BookState) -> set[str]:
    return {s for s, p in book.positions.items() if not p.is_flat}


def _bind(leg: Leg, rulings: Sequence[GuardRuling]) -> InstrumentRuling:
    """Clip the reference to the smallest ceiling and name the guard that bound."""
    reference = leg.reference
    requested = abs(reference)
    below: list[tuple[tuple[float, bool, int], float, GuardRuling]] = []
    for g in rulings:
        ceiling = _effective_ceiling(g)
        if ceiling is not None and ceiling < requested - EPS:
            below.append(((ceiling, not g.forces_exit, GUARD_ORDER.index(g.guard)), ceiling, g))
    below.sort(key=lambda item: item[0])
    approved = reference
    binding: GuardId | None = None
    final = tuple(rulings)
    if below:
        _, ceiling, winner = below[0]
        approved = math.copysign(ceiling, reference) if ceiling > 0 else 0.0
        binding = winner.guard
        if len(below) > 1:
            trail = "; ".join(
                f"{g.guard.value} {'exit' if g.forces_exit else f'at {c:.4%}'}"
                for _, c, g in below[1:]
            )
            winner = winner.model_copy(update={"reason": f"{winner.reason} | also below: {trail}"})
        final = tuple(winner if g.guard is winner.guard else g for g in rulings)
    # Re-checked against every ceiling, not merely the one that bound (ARGUS desk.py 1386-1400).
    breached = [
        g.guard.value
        for g in final
        if (c := _effective_ceiling(g)) is not None and abs(approved) > c + EPS
    ]
    if breached:  # pragma: no cover - unreachable while the minimum is taken correctly
        raise KernelError(f"{leg.symbol}: approved {approved} breaches {', '.join(breached)}")
    return InstrumentRuling(
        symbol=leg.symbol,
        current_weight=leg.current,
        proposed_weight=leg.proposed,
        approved_weight=approved,
        binding_guard=binding,
        rulings=final,
    )


def _protective_reason(acting: Sequence[InstrumentRuling], *, llm_outage: bool) -> ProtectiveReason:
    """File the ruling under the broadest cause that forced any exit (all of them are recorded in
    the per-instrument rulings either way)."""
    causes: set[ProtectiveReason] = set()
    for ir in acting:
        for g in ir.rulings:
            if not (g.forces_exit and g.status is GuardStatus.FIRED):
                continue
            if g.guard is GuardId.G5_DAILY_KILL:
                causes.add(ProtectiveReason.DAILY_KILL)
            elif g.guard is GuardId.G10_BREAKER:
                causes.add(ProtectiveReason.LLM_OUTAGE if llm_outage else ProtectiveReason.BREAKER)
            elif g.guard is GuardId.G1_VENUE_INTEGRITY:
                causes.add(ProtectiveReason.VENUE_INTEGRITY)
            elif g.guard is GuardId.G2_WEEKEND_FREEZE:
                causes.add(
                    ProtectiveReason.WINDOW_END
                    if "scoring_window_end" in g.inputs
                    else ProtectiveReason.WEEKEND_FREEZE
                )
    for reason in _PROTECTIVE_PRIORITY:
        if reason in causes:
            return reason
    raise KernelError("a protective ruling changed an instrument without a forced exit")


__all__ = [
    "GUARD_ORDER",
    "PROTECTIVE_GUARDS",
    "KernelError",
    "RiskKernel",
    "ruling_hash",
]
