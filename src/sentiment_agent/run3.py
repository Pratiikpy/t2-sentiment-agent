"""Run 3's first declared change, queued the same way run 2 declared its own (``run2.py``).

Run 2 is still the live paper run this was written against (branched from its own `run2-prep`
commit; nothing in ``t2-run2`` was touched to produce this). Its genesis, predecessor and policy
chain are not repeated or guessed at here: a full run 3 declaration needs run 2's own close window
and final genesis hash, which do not exist yet while run 2 is still running. What is declared here
is the one fix this branch carries, in the same :class:`~sentiment_agent.types.DeclaredChange` shape
run 2 used, ready to be folded into run 3's full ``declared_changes`` tuple (alongside a
:class:`~sentiment_agent.types.PredecessorRun` naming run 2) once run 2 actually closes and run 3 is
prepared to start.

``run3-d1`` (code fix)
    Run 2's own ledger (``public/orders.json``, read 2026-09-29) recorded 74 rejections of the same
    shape — HTTP 400 ``"Parameter <SYMBOL>_UMCBL does not exist"`` — for two reduce-only orders: 72
    for METAUSDT between 2026-09-28T09:16:43Z and 15:11:33Z, and 2 for MSTRUSDT in the same window.
    Bitget's SDK classifies this as its catch-all ``"unknown"`` category, so nothing distinguished a
    venue-side symbol unavailability (transient: the identical request filled at 16:06:52Z the same
    day) from a permanent rejection or a credential failure, and G10 correctly kept forcing the exit
    on every 60-second protective check while the position stayed open — but with no pacing on the
    *send*, the planner re-minted a fresh ``clientOid`` from a fresh ``ruling_id`` every cycle, so
    the exit was resent, and refused, roughly once every five minutes for close to seven hours.
    From ``run3-d1``: the refusal is classified precisely
    (``execution/environment.py`` ``is_venue_symbol_unavailable``, ``VENUE_SYMBOL_UNAVAILABLE``,
    tagged onto ``VenueRejection.category`` in ``execution/bgc.py``); a symbol whose most recent
    reduce-only order was refused this way is paced on an exponential schedule (base 5 minutes,
    factor 2, capped at 60 minutes — ``execution/exit_backoff.py``, rebuilt from the ledger's own
    submissions/rejections/acks the same way ``App.venue_unreconciled`` already is, so a restart
    mid-episode picks up exactly where the ledger left it); the planner skips re-emitting the exit
    leg between scheduled attempts (a ``SkippedLeg``, not a sent-and-refused order —
    ``kernel/planner.py``); G10 refuses any *new* exposure on that one symbol for as long as its
    episode stays open, while never blocking the reduction trying to get it flat
    (``KernelInputs.exit_backoff_until``, ``kernel/guards.py``); and the health beat
    (``var/health/<mode>.json``, rewritten every tick and read by ``t2sa status`` and any external
    watchdog) carries a live "EXIT BLOCKED" line for as long as the episode is open, with one
    ``NOTE`` ledger event when it starts and one when it clears — not one rejection per attempt
    (``runtime/loop.py``). No hedge path exists in this project to fall back on while an exit is
    stuck (the only "hedge" in the codebase is Bitget's account-level ``hedge_mode``/``posSide``
    setting, which this project runs in ``one_way_mode``, not a cross-instrument hedge execution
    mechanism); none was invented. The Bitget toolkit's own contract-status read
    (``getInstruments``, ``agent-sdk/src/generated/catalog.ts:109``, public, no credential; already
    parsed into ``InstrumentSpec.status`` at ``venue/public_api.py:422`` and already refreshed into
    ``KernelInputs.specs``) is read opportunistically once the schedule elapses, as a free
    additional gate that can only extend a wait, never justify sending early — its 24-hour refresh
    cadence (``runtime/loop.py: SPECS_EVERY``) is too coarse to detect a gap of this length on its
    own, so the schedule is what actually bounds the resends.

``run3-d2`` (code fix)
    G7 (``argus/data/t2_llm_variance.json``, ``Activity/21_T2_QUANT_SEARCH.md`` row G7,
    2026-09-29): the exact accepted request behind three of run 2's own decisions was resent
    through a fresh Qwen client at temperature 0 with the same fixed seed. The two no-act
    (``flat_with_reasons``/``hold``) decisions reproduced exactly, or differed only in an all-zero
    target list with no economic content. The one ``act`` decision (``dec-189d8234...``, closing
    MSTRUSDT and METAUSDT) did not: repeat 1 flipped stance to ``hold`` and kept both positions
    open near their prior weight; repeat 2 kept stance ``act`` and closed the same two names, but
    also opened TSLAUSDT and COINUSDT shorts neither the original nor repeat 1 proposed at all,
    and confidence on the repeated names swung by up to 0.27. From ``run3-d2``: when the model's
    first answer for a cycle is stance ``act``, the identical request is asked twice more before
    it is trusted (``decision/majority.py``, ``decision/agent.py``); the cycle acts only where at
    least 2 of the (up to 3) answers agree -- first on the stance itself, then per instrument on
    the side, with the weight the median of the agreeing answers' target (the exact rule is
    carried on every vote as ``MajorityVote.weight_rule``, ``types.py``). A first answer that is
    not ``act`` is never re-asked, matching what G7 actually measured (no variance worth the extra
    spend on those). When the three do not reach a stance majority -- genuine disagreement, or too
    few repeats obtained because one failed -- the cycle is recorded as ``LlmOutcome.DISAGREED``:
    a no-act, with every observation obtained kept in the ledger (``MajorityVote``, ``types.py``),
    and explicitly never a model-service outage, so it never counts toward run2-a6's
    flatten-after-three (the primary call decided in full this cycle, which is what proves the
    service, not the confirmation, is what came up short -- ``runtime/loop.py``). The book still
    rules under a real decision id when disagreed, exactly as a genuine ``hold`` would: every
    guard (the breaker, the weekend freeze, the daily kill, G6's turnover lock) applies in full,
    and nothing new is proposed. A downstream bug this surfaced and fixed in the same commit:
    ``t2sa replay``'s ``_answering_ruling`` (``runtime/cli.py``) matched a decision's own ruling
    only when ``outcome is DECIDED``, which would have silently failed to find the ruling for
    every DISAGREED cycle; it now matches on the ruling's own ``decision_id`` first, which is
    correct for both a decided and a disagreed cycle alike.
"""

from typing import Final

from sentiment_agent.types import DeclaredChange

RUN3_D1: Final = DeclaredChange(
    change_id="run3-d1",
    kind="code_fix",
    title=(
        "Pace a reduce-only order the venue refuses as a symbol unavailability, instead of "
        "resending it every cycle"
    ),
    detail=(
        "Run 2's ledger recorded 74 rejections of one shape (HTTP 400 'Parameter <SYMBOL>_UMCBL "
        "does not exist') for two reduce-only orders -- 72 for METAUSDT between 2026-09-28 "
        "09:16:43Z and 15:11:33Z, 2 for MSTRUSDT in the same window -- resent roughly every five "
        "minutes with no back-off, until an identical request filled at 16:06:52Z the same day. "
        "The refusal is now classified as a venue-side symbol unavailability, distinct from a "
        "credential failure and from a permanent rejection "
        "(execution/environment.py: is_venue_symbol_unavailable, VENUE_SYMBOL_UNAVAILABLE); a "
        "symbol whose exit was refused this way is paced on an exponential schedule capped at 60 "
        "minutes and rebuilt fresh from the ledger on every read, so a restart never forgets an "
        "open episode (execution/exit_backoff.py); the planner skips re-emitting the exit leg "
        "between scheduled attempts as a SkippedLeg rather than sending and being refused again "
        "(kernel/planner.py); the kernel's G10 refuses new exposure on that one symbol for as long "
        "as its episode stays open, without ever blocking the reduction trying to get it flat "
        "(kernel/guards.py, KernelInputs.exit_backoff_until); and the health beat every tick, plus "
        "one NOTE ledger event when the episode starts and one when it clears, carry the block to "
        "an operator instead of it being visible only as a flood of near-identical rejections "
        "(runtime/loop.py). No hedge path exists in this project and none was added; the fix is "
        "pacing and visibility, not a workaround for the position staying open."
    ),
    files=(
        "src/sentiment_agent/execution/environment.py",
        "src/sentiment_agent/execution/bgc.py",
        "src/sentiment_agent/execution/exit_backoff.py",
        "src/sentiment_agent/types.py",
        "src/sentiment_agent/kernel/guards.py",
        "src/sentiment_agent/kernel/kernel.py",
        "src/sentiment_agent/kernel/planner.py",
        "src/sentiment_agent/runtime/wiring.py",
        "src/sentiment_agent/runtime/loop.py",
    ),
    evidence={
        "run2_rejections": "74 (METAUSDT 72, MSTRUSDT 2), all HTTP 400 'Parameter ..._UMCBL does "
        "not exist', 2026-09-28T09:16:43Z to 15:11:33Z; the identical request filled at "
        "2026-09-28T16:06:52Z (METAUSDT) and 16:06:54Z (MSTRUSDT)",
        "bounded_attempts": "replaying the same 5h55m span through the run3-d1 schedule "
        "(execution/exit_backoff.py: 5 min base, factor 2, 60 min cap) takes under 20 real "
        "attempts, not 72 (tests/execution/test_exit_backoff.py: "
        "test_the_run2_schedule_bounds_attempts_over_the_actual_episode_length)",
        "source": "run 2's live ledger, public/orders.json, read 2026-09-29 (read-only; no order "
        "was placed and no file in t2-run2 was written to produce this declaration)",
    },
)

RUN3_D2: Final = DeclaredChange(
    change_id="run3-d2",
    kind="code_fix",
    title="Majority-of-3 cross-check on a stance='act' decision before it is trusted",
    detail=(
        "G7 resent the exact accepted request behind three of run 2's own decisions through a "
        "fresh Qwen client at temperature 0 with the same fixed seed. The two no-act decisions "
        "reproduced exactly, or differed only in an all-zero target list with no economic "
        "content; the one act decision did not -- repeat 1 flipped stance to hold and kept both "
        "positions open, repeat 2 kept stance act and closed the same two names but also opened "
        "two new shorts (TSLAUSDT, COINUSDT) neither other answer proposed at all, and confidence "
        "on the repeated names swung by up to 0.27. From run3-d2: a stance='act' first answer is "
        "asked twice more before it is trusted (decision/majority.py, decision/agent.py); the "
        "cycle acts only where at least 2 of the (up to 3) answers agree, first on the stance "
        "itself, then per instrument on the side, weighted by the median of the agreeing answers' "
        "target (MajorityVote.weight_rule states the exact rule on every vote, types.py). A first "
        "answer that is not act is never re-asked, matching what G7 measured -- no variance worth "
        "the extra spend on those. No majority reached (genuine disagreement, or a failed repeat "
        "leaving too few to compare) is recorded as LlmOutcome.DISAGREED: a no-act, every "
        "observation kept in the ledger, and never a model-service outage, so it never counts "
        "toward run2-a6's flatten-after-three -- the primary call decided in full this cycle, "
        "which is what proves the service is up. The book still rules under a real decision id "
        "when disagreed, exactly as a genuine hold would, with every guard applying in full and "
        "nothing new proposed. Two downstream bugs this surfaced and fixed in the same commit: "
        "t2sa replay's _answering_ruling (runtime/cli.py) matched a decision's own ruling only "
        "when outcome is DECIDED, which would have silently failed to find the ruling for every "
        "DISAGREED cycle -- it now matches on the ruling's own decision_id first, correct for "
        "both a decided and a disagreed cycle; and every cost report that read "
        "DecisionRecord.call.usage alone (redteam/harness.py's spend total, site/export.py's "
        "decisions.json token column) would have silently undercounted a majority-voted act "
        "cycle's real cost by up to 2/3, since call is only the first of up to 3 calls once a "
        "vote runs -- both now read the new DecisionRecord.total_llm_tokens, which sums every "
        "observation's call (types.py), and decisions.json also publishes the vote itself "
        "(observations, act_count, agreed, reason, weight_rule) for Track 2's judged decision "
        "explainability. Replaying a majority-voted cycle needed its own fix too: the 2 repeats "
        "share the original's byte-identical request, so one prompt-hash-keyed RecordedChatModel "
        "cannot serve all 3 in order; ReplayModel (runtime/cli.py) now holds one per observation, "
        "advancing to the next once the current is exhausted or raises its own recorded failure."
    ),
    files=(
        "src/sentiment_agent/types.py",
        "src/sentiment_agent/decision/majority.py",
        "src/sentiment_agent/decision/agent.py",
        "src/sentiment_agent/decision/contract.py",
        "src/sentiment_agent/runtime/loop.py",
        "src/sentiment_agent/runtime/cli.py",
        "src/sentiment_agent/redteam/harness.py",
        "src/sentiment_agent/site/export.py",
    ),
    evidence={
        "act_cycle_variance": "the one act decision tested (dec-189d8234..., closing METAUSDT and "
        "MSTRUSDT) did not reproduce: repeat 1 stance hold, both positions kept open; repeat 2 "
        "stance act, closed the same two names (METAUSDT, MSTRUSDT) but also opened TSLAUSDT and "
        "COINUSDT shorts neither other answer proposed; confidence spread up to 0.27 on the "
        "repeated names",
        "no_act_cycle_stability": "the 2 no-act decisions tested reproduced exactly or differed "
        "only in an all-zero target list with no economic content",
        "cost_per_act_cycle": "up to 2 extra full decision calls (each itself bounded by "
        "policy.decision.max_attempts, exactly like the first); on G7's own measured 18.7k-30.0k "
        "tokens for one such call, roughly triples that one cycle's cost -- bounded automatically "
        "by the existing daily cap, which refuses a repeat that would cross it rather than "
        "overspend",
        "source": "argus/data/t2_llm_variance.json (generated_at 2026-09-29T10:42:39Z) and "
        "Activity/21_T2_QUANT_SEARCH.md row G7, both read-only; no order was placed and no file "
        "in t2-run2 was written to produce this declaration",
    },
)

DECLARED_CHANGES: Final = (RUN3_D1, RUN3_D2)

__all__ = ["DECLARED_CHANGES", "RUN3_D1", "RUN3_D2"]
