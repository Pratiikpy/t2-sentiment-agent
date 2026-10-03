"""The governed/ungoverned twin: what the risk kernel did to the model's drafts, and what it cost.

Track 2 is judged on "risk control layer effectiveness" (handbook:248). The honest way to show it is
to run the model's pre-kernel drafts as if nothing governed them, on the same Demo marks and the
same cost model, and set that beside the governed book (win plan Move 5b, DESIGN.md §14.2).

**The two arms** come from the same :class:`~sentiment_agent.analysis.armsim.ArmSimulator`, so the
only difference between them is the kernel:

* :func:`ungoverned_arm`: every decided draft (``DecisionRecord.proposed_weights``) at the instant
  it was ruled, with no guard at all: no stop, no weekend freeze, no size cap, no kill. A decision
  that produced no draft (a model outage) changes nothing, because nothing but the kernel ever acts
  without the model, and there is no kernel here.
* :func:`governed_replica`: the same drafts under every guard, with each decision's grounding
  reports and declared invalidations, so G9 and G6 rule as they did live. An outage is replayed as
  what the kernel does with one: every position closed. Compare this, not the live book, with the
  ungoverned arm: both are simulated identically, so the gap is the kernel's and not the
  difference between a simulation and a venue. (The live book is set beside the replica elsewhere,
  as a check of the simulator.)

**The report** (:func:`twin_report`), from the decisions, the kernel's rulings on them and the two
arms. Only decided drafts that have a logged ruling count (the newest decision may still be
mid-cycle when a report is taken). With a *proposed leg* being one symbol of one draft:

* ``n_decisions``: decided drafts with a ruling.
* ``n_interventions`` and ``interventions``: proposed legs the kernel changed
  (``InstrumentRuling.changed_by_kernel``), one :class:`TwinIntervention` each, naming the guard
  that bound.
* ``intervention_rate``: the share of decisions in which the kernel changed at least one leg.
* ``risk_violation_rate_ungoverned``: the share of proposed legs that broke at least one rule, i.e.
  at least one guard **FIRED** with a limit below the size asked for. A guard that bound only
  because an input was missing (``NOT_EVALUATED``, fail-closed) is an intervention but not a
  violation by the model; it is not counted here.
* ``violations_prevented`` per intervention: every guard that fired below the requested size, in
  guard order, not only the one that bound.
* ``pnl_ungoverned`` and ``pnl_governed`` per intervention: what the leg earned at the size asked
  for and at the size approved, **from the decision to the next decided draft** (or the end of the
  record), in fractions of equity, net of the same cost model the arms pay: ``w * R - c * |w -
  w_held|``, with ``R`` the symbol's Demo mark return over the window and ``c`` one side's cost
  (taker fee plus half the Demo spread). Prices are the first hourly Demo mark at or after each
  end of the window, so no price the model could not have seen enters. This is a marginal
  attribution: each intervention is priced holding the rest of the book as it was, which is what
  makes the per-intervention figures add up; the arms' own paths carry every interaction (stops,
  kills, the gross cap), and their drawdowns are reported beside them.
* ``prevented_loss`` = ``sum(max(0, pnl_governed - pnl_ungoverned))``; ``forgone_gain`` =
  ``sum(max(0, pnl_ungoverned - pnl_governed))``. Both are reported whatever they show.
* ``max_drawdown_governed`` / ``_ungoverned``: from the two arms' metrics.
* ``human_takeovers``: owner-triggered decisions and amendments, counted by the caller from the
  ledger and published with the rest.

Per-intervention P&L needs the Demo mark path and the cost model, so :func:`twin_report` takes the
simulator the arms were run on (``sim=``); a report with interventions and no simulator is refused
rather than priced at zero.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Final

from sentiment_agent.analysis.armsim import ArmSimulator
from sentiment_agent.analysis.bootstrap import DEFAULT_RESAMPLES
from sentiment_agent.kernel.kernel import GUARD_ORDER
from sentiment_agent.types import (
    ALL_GUARDS,
    ArmKind,
    ArmResult,
    ArmSpec,
    DecisionRecord,
    GuardAttribution,
    GuardId,
    GuardRuling,
    GuardStatus,
    InstrumentRuling,
    KernelInputs,
    KernelRuling,
    LlmOutcome,
    RulingContext,
    TwinIntervention,
    TwinReport,
)

EPS: Final = 1e-12

UNGOVERNED_SPEC: Final = ArmSpec(
    arm_id="twin_ungoverned",
    kind=ArmKind.TWIN_UNGOVERNED,
    title="The agent, ungoverned",
    description="Every draft Qwen wrote, sent as written: no risk kernel, no stops, no weekend "
    "freeze, no size cap, no kill. Same Demo marks and cost model as the governed replica.",
    provenance="src/sentiment_agent/analysis/twin.py (this repository, MIT)",
    uses_llm=True,
    guards=(),
)

GOVERNED_REPLICA_SPEC: Final = ArmSpec(
    arm_id="twin_governed_replica",
    kind=ArmKind.OURS_GOVERNED,
    title="The agent, governed (replica)",
    description="The same drafts through the full risk kernel, simulated exactly like the "
    "ungoverned arm, so the difference between the two is the kernel alone.",
    provenance="src/sentiment_agent/analysis/twin.py (this repository, MIT)",
    uses_llm=True,
    guards=tuple(g for g in GUARD_ORDER if g in ALL_GUARDS),
)


def _aligned(
    decisions: Sequence[DecisionRecord], inputs: Sequence[KernelInputs]
) -> list[tuple[DecisionRecord, KernelInputs]]:
    if len(decisions) != len(inputs):
        raise ValueError("one set of kernel inputs is needed per decision")
    pairs = list(zip(decisions, inputs, strict=True))
    for record, given in pairs:
        if given.snapshot_id is not None and given.snapshot_id != record.snapshot_id:
            raise ValueError(
                f"decision {record.decision_id} was taken on snapshot {record.snapshot_id}, but "
                f"its kernel inputs are from {given.snapshot_id}"
            )
    pairs.sort(key=lambda pair: pair[1].at)
    return pairs


def ungoverned_arm(
    sim: ArmSimulator,
    decisions: Sequence[DecisionRecord],
    inputs: Sequence[KernelInputs],
    *,
    start: datetime | None = None,
    until: datetime | None = None,
    ci_resamples: int = DEFAULT_RESAMPLES,
) -> ArmResult:
    """The model's drafts with no guard. ``inputs[i]`` is what decision ``i`` was ruled with; each
    draft is placed at ``inputs[i].at``, the instant the kernel ruled it."""
    schedule = []
    contexts = []
    for record, given in _aligned(decisions, inputs):
        if record.outcome is not LlmOutcome.DECIDED:
            continue
        schedule.append((given.at, dict(record.proposed_weights), given))
        contexts.append(RulingContext(decision_id=record.decision_id, protective_reason=None))
    return sim.run(
        UNGOVERNED_SPEC,
        schedule,
        contexts=contexts,
        start=start,
        until=until,
        ci_resamples=ci_resamples,
    )


def governed_replica(
    sim: ArmSimulator,
    decisions: Sequence[DecisionRecord],
    inputs: Sequence[KernelInputs],
    *,
    start: datetime | None = None,
    until: datetime | None = None,
    ci_resamples: int = DEFAULT_RESAMPLES,
    spec: ArmSpec = GOVERNED_REPLICA_SPEC,
) -> ArmResult:
    """The same drafts under every guard, simulated like :func:`ungoverned_arm` (module docstring).
    An outage is replayed as a proposal of zero for every universe symbol: the kernel's flatten.
    ``spec`` carries the guards in force; :func:`guard_lifted_arm` passes every guard but one."""
    schedule = []
    contexts = []
    for record, given in _aligned(decisions, inputs):
        if record.decision is None:
            schedule.append((given.at, dict.fromkeys(sim.policy.symbols, 0.0), given))
            contexts.append(RulingContext(decision_id=record.decision_id, protective_reason=None))
            continue
        schedule.append((given.at, dict(record.proposed_weights), given))
        contexts.append(
            RulingContext(
                decision_id=record.decision_id,
                protective_reason=None,
                grounding=dict(record.grounding),
                invalidation_fired={
                    t.symbol: t.invalidation_triggered for t in record.decision.targets
                },
            )
        )
    return sim.run(
        spec,
        schedule,
        contexts=contexts,
        start=start,
        until=until,
        ci_resamples=ci_resamples,
    )


def lifted_spec(guard: GuardId) -> ArmSpec:
    """The governed replica's spec with ``guard`` lifted and every other guard in force."""
    if guard not in GOVERNED_REPLICA_SPEC.guards:
        raise ValueError(f"{guard.value} is not a guard of the governed replica")
    return ArmSpec(
        arm_id=f"twin_lifted_{guard.value}",
        kind=ArmKind.OURS_GOVERNED,
        title=f"The agent, governed, {guard.value} lifted",
        description=f"The governed replica's drafts under every guard but {guard.value}, "
        "simulated exactly like the replica, so the difference is that one guard.",
        provenance="src/sentiment_agent/analysis/twin.py (this repository, MIT)",
        uses_llm=True,
        guards=tuple(g for g in GOVERNED_REPLICA_SPEC.guards if g is not guard),
    )


def guard_lifted_arm(
    sim: ArmSimulator,
    decisions: Sequence[DecisionRecord],
    inputs: Sequence[KernelInputs],
    guard: GuardId,
    *,
    start: datetime | None = None,
    until: datetime | None = None,
    ci_resamples: int = DEFAULT_RESAMPLES,
) -> ArmResult:
    """The governed replica with one guard lifted (beyond tare's all-or-nothing A/B, S2 field
    research, ``research/s2-field/projects/68_tare.md``): the whole-path effect of that guard,
    stops, kills and the gross cap included, which the marginal per-intervention figures leave
    out."""
    return governed_replica(
        sim,
        decisions,
        inputs,
        start=start,
        until=until,
        ci_resamples=ci_resamples,
        spec=lifted_spec(guard),
    )


def guard_attribution(
    report: TwinReport, lifted: Mapping[GuardId, ArmResult] | None = None
) -> tuple[GuardAttribution, ...]:
    """Per guard that bound at least once, in guard order: how many interventions it bound and
    what they prevented and cost (marginal, as in :func:`twin_report`), and, when ``lifted`` has
    its arm, the replica's return, drawdown and Sharpe with that guard lifted."""
    lifted = lifted or {}
    out: list[GuardAttribution] = []
    for guard in GUARD_ORDER:
        own = [i for i in report.interventions if i.guard is guard]
        if not own:
            continue
        arm = lifted.get(guard)
        out.append(
            GuardAttribution(
                guard=guard,
                n_interventions=len(own),
                prevented_loss=sum(max(0.0, i.pnl_governed - i.pnl_ungoverned) for i in own),
                forgone_gain=sum(max(0.0, i.pnl_ungoverned - i.pnl_governed) for i in own),
                lifted_total_return=None if arm is None else arm.metrics.total_return,
                lifted_max_drawdown=None if arm is None else arm.metrics.max_drawdown,
                lifted_sharpe_ann=None if arm is None else arm.metrics.sharpe_ann,
            )
        )
    return tuple(out)


def _effective_ceiling(ruling: GuardRuling) -> float | None:
    return 0.0 if ruling.forces_exit else ruling.ceiling_abs_weight


def violated_guards(
    inst: InstrumentRuling, book_rulings: Sequence[GuardRuling]
) -> tuple[GuardId, ...]:
    """Guards that FIRED with a limit below the size the draft asked for, in guard order."""
    requested = abs(inst.reference)
    fired = set()
    for ruling in (*inst.rulings, *book_rulings):
        if ruling.status is not GuardStatus.FIRED:
            continue
        ceiling = _effective_ceiling(ruling)
        if ceiling is not None and ceiling < requested - EPS:
            fired.add(ruling.guard)
    return tuple(g for g in GUARD_ORDER if g in fired)


def _window_return(sim: ArmSimulator, symbol: str, start: datetime, end: datetime) -> float:
    """The Demo mark return from the first hourly mark at or after ``start`` to the first at or
    after ``end`` (the last available mark when the record ends first)."""
    entry = sim.price_after(symbol, start)
    if entry is None:
        return 0.0
    exit_ = sim.price_after(symbol, end)
    if exit_ is None:
        last = sim.last_close(symbol)
        if last is None:
            return 0.0
        exit_ = last[1]
    return float(exit_ / entry - 1)


def twin_report(
    decisions: Sequence[DecisionRecord],
    rulings: Sequence[KernelRuling],
    governed: ArmResult,
    ungoverned: ArmResult,
    *,
    human_takeovers: int,
    sim: ArmSimulator | None = None,
) -> TwinReport:
    """The twin report (module docstring). ``sim`` is the simulator the arms ran on; it prices each
    intervention and is required whenever there is one."""
    if human_takeovers < 0:
        raise ValueError("human takeovers cannot be negative")
    by_decision: dict[str, KernelRuling] = {}
    for ruling in rulings:
        if ruling.decision_id is not None:
            if ruling.decision_id in by_decision:
                raise ValueError(f"decision {ruling.decision_id} has two kernel rulings")
            by_decision[ruling.decision_id] = ruling
    drafts = sorted(
        (
            (record, by_decision[record.decision_id])
            for record in decisions
            if record.outcome is LlmOutcome.DECIDED and record.decision_id in by_decision
        ),
        key=lambda pair: pair[1].at,
    )
    ends = [m.at for arm in (governed, ungoverned) for m in arm.marks[-1:]]
    record_end = max(ends) if ends else None

    interventions: list[TwinIntervention] = []
    decisions_changed = 0
    legs = 0
    violating_legs = 0
    for index, (record, ruling) in enumerate(drafts):
        window_end = drafts[index + 1][1].at if index + 1 < len(drafts) else record_end
        changed_here = False
        for inst in ruling.instruments:
            if inst.proposed_weight is None:
                continue  # a held symbol the draft did not address: not part of the draft
            legs += 1
            violations = violated_guards(inst, ruling.book_rulings)
            if violations:
                violating_legs += 1
            if not inst.changed_by_kernel or inst.binding_guard is None:
                continue
            changed_here = True
            if sim is None:
                raise ValueError(
                    "twin_report needs the simulator the arms ran on (sim=) to price the "
                    "kernel's interventions; refusing to report them at zero"
                )
            ret = 0.0
            if window_end is not None and window_end > ruling.at:
                ret = _window_return(sim, inst.symbol, ruling.at, window_end)
            cost = sim.cost_rate(inst.symbol)
            held = inst.current_weight
            asked = inst.proposed_weight
            approved = inst.approved_weight
            interventions.append(
                TwinIntervention(
                    decision_id=record.decision_id,
                    ruling_id=ruling.ruling_id,
                    symbol=inst.symbol,
                    guard=inst.binding_guard,
                    proposed_weight=asked,
                    approved_weight=approved,
                    pnl_ungoverned=asked * ret - cost * abs(asked - held),
                    pnl_governed=approved * ret - cost * abs(approved - held),
                    violations_prevented=violations,
                )
            )
        if changed_here:
            decisions_changed += 1

    return TwinReport(
        n_decisions=len(drafts),
        n_interventions=len(interventions),
        intervention_rate=decisions_changed / len(drafts) if drafts else 0.0,
        prevented_loss=sum(max(0.0, i.pnl_governed - i.pnl_ungoverned) for i in interventions),
        forgone_gain=sum(max(0.0, i.pnl_ungoverned - i.pnl_governed) for i in interventions),
        risk_violation_rate_ungoverned=violating_legs / legs if legs else 0.0,
        max_drawdown_governed=governed.metrics.max_drawdown,
        max_drawdown_ungoverned=ungoverned.metrics.max_drawdown,
        human_takeovers=human_takeovers,
        interventions=tuple(interventions),
        governed_total_return=governed.metrics.total_return,
    )


def guard_counts(report: TwinReport) -> Mapping[GuardId, int]:
    """How many interventions each guard bound (for the guard funnel on the demo page)."""
    counts: dict[GuardId, int] = {}
    for item in report.interventions:
        counts[item.guard] = counts.get(item.guard, 0) + 1
    return {g: counts[g] for g in GUARD_ORDER if g in counts}


__all__ = [
    "GOVERNED_REPLICA_SPEC",
    "UNGOVERNED_SPEC",
    "governed_replica",
    "guard_attribution",
    "guard_counts",
    "guard_lifted_arm",
    "lifted_spec",
    "twin_report",
    "ungoverned_arm",
    "violated_guards",
]
