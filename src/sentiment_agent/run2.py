"""Run 2's declaration: what changed against run 1, written into run 2's genesis before any order.

Run 1 is the paper run pre-registered at 2026-09-24 17:06 UTC (genesis event
``ebbf607a…53a5``, code commit ``3059fcf``, policy v1). It is left exactly as it ran: its record
still verifies with this code (policy v1 hashes as it did, and every 1.0.0 record reads unchanged,
``types.CONTRACT_VERSION``). Run 2 is a new ledger with a new genesis, and that genesis carries
:data:`PREDECESSOR` and :data:`DECLARED_CHANGES`, so a reader can see from the pre-registration
itself, and not from a README, what differs and why:

``run2-d1`` (code fix)
    Trigger kinds that read crowd text or the calendar (coordinated clusters, earnings, filings)
    are evaluated on the full snapshot. In run 1 only light snapshots, which do not read those
    sources, reached the trigger engine, so those kinds could never fire.
``run2-d2`` (observability)
    Feed-health alarms: a source that errors, times out or comes back hollow is a ``feed_health``
    ledger event, on the decision card and on the public page, and every trigger kind it blinds is
    named with its cause.
``run2-d3`` (code fix)
    When both Bitget data services leave a reading empty, the upstream they wrap is read directly
    (Binance futures positioning, alternative.me Fear & Greed, publishers' RSS), on its own
    labelled surface; see ``sources/upstream.py``.
``run2-a1`` to ``run2-a6`` (policy amendments, v1 to v2, ``policy.RUN2_AMENDMENTS``)
    a1: ``funding_zscore`` is evaluated for every universe instrument, not BTCUSDT alone.
    a2: it also needs the live rate at 7.5 bp or more, since half of run 1's live rates were zero.
    a3: G3 caps the book's net weight at 10% and the crypto-beta names at 7.5% together.
    a4: the record is scored over a pre-registered window, and at its end G2 closes every leg.
    a5: the losing-streak trip lapses 24 hours after the last losing close.
    a6: a model outage flattens the book on the third failed decision in a row, not the first.
    Each is its own hashed policy, and the genesis declares them as a chain from v1 to v2.

Nothing else changes: every other guard limit, the fee and edge bar, the mandate, the system prompt
and its no-edge stance, and the prompt files (whose hashes the genesis carries and a test holds
equal to run 1's). The replayed counts in the evidence are recomputed by
``scripts/replay_triggers.py`` from run 1's published ledger, cut at seq 363 (226 snapshots,
2026-09-24 17:06 to 2026-09-25 12:58 UTC), and kept in ``validation/run2/run1_trigger_replay.json``;
the agent's own behaviour on that market is ``validation/run2/act_rate.json``, by
``scripts/act_rate.py``.
"""

from typing import Final

from sentiment_agent.policy import POLICY_V1, POLICY_V2, RUN2_STEPS
from sentiment_agent.types import DeclaredChange, Policy, PredecessorRun

RUN1_GENESIS_HASH: Final = "ebbf607a8fe199bccc0f612ededf2a6902e4884a38d21fbfea498dff7da453a5"
RUN1_CODE_COMMIT: Final = "3059fcf473927093af35d9af4c999c6d135b331a"
RUN1_PROMPT_HASHES: Final[dict[str, str]] = {
    "decision/prompt.py": "15f825ced789b7fb9c809685fd67546b466de45c3b8cf9846695e839b8715e0d",
    "decision/prompts/output_schema_v1.md": (
        "9675a8b18c2ee0a1371d95ae13b9c7dda2284473f687376a38f4a78768aa373a"
    ),
    "decision/prompts/system_v1.md": (
        "67773d85dbebbd0e6b0db5dba48b558ac97e231aa1e37db39b23201da095007e"
    ),
}
"""Run 1's genesis ``prompt_hashes``; run 2 pre-registers the same files unchanged."""

REPLAY_EVIDENCE: Final = "validation/run2/run1_trigger_replay.json"
OUTAGE_EVIDENCE: Final = "validation/run2/run1_feed_outage.json"
OUTAGE_COMMAND: Final = (
    "python scripts/feed_outage.py <run 1 public/> --until-seq 546 --out " + OUTAGE_EVIDENCE
)
ACT_RATE_EVIDENCE: Final = "validation/run2/act_rate.json"
ACT_RATE_COMMAND: Final = (
    "python scripts/act_rate.py <run 1 public/> --until-seq 363 --policy run2-a1 --out "
    + ACT_RATE_EVIDENCE
)
REPLAY_COMMAND: Final = (
    "python scripts/replay_triggers.py <run 1 public/> --until-seq 363 --out " + REPLAY_EVIDENCE
)

PREDECESSOR: Final = PredecessorRun(
    genesis_hash=RUN1_GENESIS_HASH,
    code_commit=RUN1_CODE_COMMIT,
    policy_hash=POLICY_V1.content_hash(),
    policy_version=POLICY_V1.version,
    window="2026-09-24T17:06Z to 2026-09-27T17:06Z (72 hours)",
)

_HASHES: Final = (POLICY_V1.content_hash(), *(p.content_hash() for _, p in RUN2_STEPS))
"""Policy v1's hash, then the hash after each amendment in turn; the last is policy v2's."""

_REPLAY: Final = REPLAY_EVIDENCE + ", by " + REPLAY_COMMAND
_TRIGGERS: Final = ("src/sentiment_agent/policy.py", "src/sentiment_agent/events/triggers.py")

_AMENDMENTS: Final[tuple[tuple[str, str, str, tuple[str, ...], dict[str, str]], ...]] = (
    (
        "run2-a1",
        "funding_zscore covers every universe instrument",
        "Policy v1 evaluated the live funding z-score trigger for BTCUSDT only. From run2-a1 it "
        "is evaluated for all 14 instruments with the same +-2 threshold, 90-settlement lookback, "
        "240-minute cooldown per instrument, 8 event decisions per UTC day, and the same weekend "
        "refusal of events on US-session legs while G2 holds them flat.",
        _TRIGGERS,
        {
            "funding_extremes": "live funding z beyond +-2 in 436 equity/index "
            "instrument-snapshots of 226 (HOOD 88, NVDA 84, MSTR 58, GOOGL 49, AMZN 46, SNDK 44, "
            "TSLA 30, COIN 23, NDX100 14); BTCUSDT 0",
            "replayed_event_decisions": "15 in 19.9 hours on run 1's 226 snapshots under this "
            "step alone (7 on 2026-09-24, 8 on 2026-09-25, the daily cap refusing 3 more); 15 "
            "also within the token budget at every decision's worst-case bound; run 1 itself: 0",
            "source": _REPLAY + " (policy_v1, run2_a1)",
        },
    ),
    (
        "run2-a2",
        "funding_zscore also needs a funding level of 7.5 bp",
        "The z-score alone fires on a series that sits at zero: an equity perp's live rate was "
        "exactly zero in half of run 1's instrument-snapshots, so a single tick off zero scored "
        "as an extreme and the trigger fired in 13.8% of them, where a +-2 bound on a normal "
        "series fires in 4.6%. From run2-a2 the live rate must also be at least 0.075% per "
        "settlement interval in absolute value. The threshold, lookback, cooldown and cap are "
        "unchanged.",
        _TRIGGERS,
        {
            "level_profile": "3,150 instrument-snapshots with a live z-score and rate: beyond "
            "+-2 in 13.8%; the absolute rate zero in 51.1%, median 0, 95th percentile 4.5 bp; "
            "beyond +-2 with a level of 5, 7.5 and 8 bp: 3.5%, 1.4%, 1.0%",
            "funding_extremes_with_level": "43 of the 436 (MSTR 29, SNDK 14)",
            "replayed_event_decisions": "3 in 19.9 hours (2026-09-25 00:18 MSTR, 01:28 SNDK, "
            "12:31 MSTR), against 15 under run2-a1 alone",
            "agent_on_the_unfloored_triggers": "run 2's agent acted in 14 of the 15 decisions "
            "run2-a1 alone would have woken, every one of them short",
            "source": _REPLAY
            + " (funding_level, funding_extremes, policy_v2); "
            + ACT_RATE_EVIDENCE,
        },
    ),
    (
        "run2-a3",
        "G3 caps the book's net weight at 10% and the crypto-beta names at 7.5%",
        "Policy v1 capped each name at 5% and the book at 25% gross, and nothing capped one "
        "direction or one exposure spread across names. From run2-a3 G3 also caps the net weight "
        "(long minus short) at 10% of equity and MSTR, COIN, HOOD and CRCL at 7.5% gross "
        "together. What is already held keeps its weight; new exposure is scaled pro rata "
        "into what the caps leave.",
        (
            "src/sentiment_agent/policy.py",
            "src/sentiment_agent/kernel/guards.py",
            "src/sentiment_agent/kernel/kernel.py",
            "src/sentiment_agent/types.py",
        ),
        {
            "one_sided_books": "run 2's agent, replayed on run 1's market under run2-a1, acted "
            "in 14 of 15 decisions and every book it proposed was short; the largest was 20% "
            "net short (GOOGL, HOOD, MSTR, TSLA at 5% each, 2026-09-25 12:05 UTC); no guard of "
            "policy v1 changed any of them",
            "crypto_beta": "COIN, HOOD and MSTR 5% short each at once (15%), 2026-09-25 00:13 "
            "UTC; MSTR moved 1.85x BTC over 30 days of hourly candles (correlation +0.83)",
            "source": ACT_RATE_EVIDENCE + ", by " + ACT_RATE_COMMAND,
        },
    ),
    (
        "run2-a4",
        "A pre-registered scoring window, closed by G2 at its end",
        "Run 2 is scored from 2026-09-28 00:00 UTC to 2026-10-01 00:00 UTC, and the window is in "
        "the hashed policy. At its end G2 closes every leg of every asset class and refuses new "
        "exposure, the loop stops deciding, and the published metrics count only marks, fills and "
        "trades inside it.",
        (
            "src/sentiment_agent/policy.py",
            "src/sentiment_agent/kernel/guards.py",
            "src/sentiment_agent/kernel/kernel.py",
            "src/sentiment_agent/runtime/loop.py",
            "src/sentiment_agent/analysis/metrics.py",
            "src/sentiment_agent/site/export.py",
            "src/sentiment_agent/types.py",
        ),
        {
            "run1": "no scored span in run 1's genesis; a trade still open at the end of a record "
            "has no result, so win rate and trade count depend on when the record is read",
        },
    ),
    (
        "run2-a5",
        "The losing-streak trip lapses 24 hours after the last losing close",
        "Under policy v1, four losing trades in a row put the book in reduce-only until a "
        "winning trade closed. Reduce-only refuses every exposure-adding order, so a book that "
        "was flat could never open, never close a winner, and never leave reduce-only: the trip "
        "was absorbing. From run2-a5 it lapses 24 hours after the last losing close. The "
        "drawdown trips, which clear on equity, are unchanged.",
        (
            "src/sentiment_agent/policy.py",
            "src/sentiment_agent/kernel/breaker.py",
            "src/sentiment_agent/book/book.py",
            "src/sentiment_agent/types.py",
        ),
        {
            "mechanism": "kernel/breaker.py: consecutive_losses counts closed trades back from "
            "the last winner, and in reduce-only G10 refuses every opening, so on a flat book "
            "the count cannot fall",
        },
    ),
    (
        "run2-a6",
        "A model outage flattens the book on the third failed decision in a row",
        "Under policy v1 one failed decision (a timeout, a malformed answer, a refused call) "
        "flattened every open leg at market. From run2-a6 the first and second failed decisions "
        "in a row hold the book under the venue stops G4 placed with each leg and open nothing; "
        "the third flattens it as before. A decision that succeeds resets the count.",
        (
            "src/sentiment_agent/policy.py",
            "src/sentiment_agent/runtime/loop.py",
            "src/sentiment_agent/types.py",
        ),
        {
            "cost": "a flatten is a taker trade on every open leg at 6 bp each (Demo "
            "takerFeeRate 0.0006, universe_probe.json), paid on a gateway timeout that says "
            "nothing about the thesis",
        },
    ),
)

_UNCHANGED: Final = (
    "The policy, the prompt files and every guard limit are unchanged by this change."
)

DECLARED_CHANGES: Final[tuple[DeclaredChange, ...]] = (
    DeclaredChange(
        change_id="run2-d1",
        kind="code_fix",
        title="Crowd and calendar trigger kinds are evaluated on the full snapshot",
        detail="Run 1 evaluated triggers only on light snapshots (every 5 minutes), which by "
        "design do not read crowd text or the calendar (DESIGN.md §7), and never on the full "
        "snapshot built for each decision, so coordinated_cluster, earnings_event and "
        "filing_event could not fire. Run 2 keeps the pre-registered cadences (crowd text and"
        " the calendar are read on full snapshots only) and evaluates those kinds on every "
        "full snapshot: when a candidate set will start a decision, the full snapshot is "
        "taken first and its crowd and calendar events are admitted in the same batch, under "
        "the same cooldowns and daily cap, so they join that decision instead of waking "
        "another. Light snapshots evaluate only Fear & Greed, funding and open interest, so "
        "no condition is emitted twice. " + _UNCHANGED,
        files=(
            "src/sentiment_agent/events/triggers.py",
            "src/sentiment_agent/runtime/loop.py",
        ),
        evidence={
            "run1_full_snapshot_triggers_never_evaluated": "22 coordinated_cluster triggers on "
            "run 1's 2 decision snapshots (admitted riders under run 2's wiring; 1 more refused "
            "by cooldown); 0 extra decisions",
            "source": REPLAY_EVIDENCE + " (policy_v1)",
        },
    ),
    DeclaredChange(
        change_id="run2-d2",
        kind="observability",
        title="Feed-health alarms, and a named reason for every blind trigger kind",
        detail="A source that errors, times out or answers hollow raises an alarm with the start "
        "of its streak; recovery clears it. Each change is a feed_health ledger event (and every "
        "decision snapshot's report is logged), carried on the decision card, shown on the public "
        "page and in the health file, and every trigger kind it leaves unable to fire is named "
        "per instrument with its cause (a failing feed, no usable data, the light-snapshot "
        "cadence, or no frozen threshold). Run 1 recorded these failures only inside snapshots: "
        "sentiment_index.current hollow in 226 of 226, bitget-mcp-server do_query answering 503 "
        "on every call from 2026-09-25 10:14 UTC (and a session-expiry 404 at 08:33), and no "
        "upcoming earnings date in the calendar. " + _UNCHANGED,
        files=(
            "src/sentiment_agent/perception/feeds.py",
            "src/sentiment_agent/runtime/loop.py",
            "src/sentiment_agent/site/export.py",
            "src/sentiment_agent/site/render.py",
            "src/sentiment_agent/site/cards.py",
        ),
        evidence={
            "sentiment_index.current": "hollow in 226 of 226 run-1 snapshots",
            "do_query": "error (service reported 503) on every call from 2026-09-25T10:14:31Z",
            "calendar": "equity_calendar answered; no upcoming report date for any equity",
            "source": "run 1 public/ledger.jsonl through seq 363",
        },
    ),
    DeclaredChange(
        change_id="run2-d3",
        kind="code_fix",
        title="The upstreams Bitget's data services wrap, read directly when both fail",
        detail="Crowd positioning (retail and top-trader long/short, taker buy/sell, open "
        "interest, funding) is read from bitget-mcp-server, then bitget-signal for a field it "
        "left empty; crypto Fear & Greed from both; news from bitget-signal. When both leave a "
        "reading empty, run 2 reads the source they themselves wrap: Binance USD-M futures "
        "(futures/data/*, fapi/v1/fundingRate; open interest in contracts, the unit "
        "bitget-mcp-server returns), alternative.me for Fear & Greed, and four publishers' RSS "
        "for news. Every such call is recorded on its own surface (upstream_direct) with the "
        "upstream named, after the Bitget calls, which are still made and recorded, so their "
        "failure stays a feed-health alarm and the toolkit count still counts only what Bitget "
        "answered. A series is taken whole from one source. " + _UNCHANGED,
        files=(
            "src/sentiment_agent/sources/upstream.py",
            "src/sentiment_agent/sources/toolkit.py",
            "src/sentiment_agent/runtime/wiring.py",
            "src/sentiment_agent/perception/feeds.py",
            "src/sentiment_agent/site/coverage.py",
            "src/sentiment_agent/types.py",
        ),
        evidence={
            "bitget_mcp_server": "answered 144 of 1356 calls from its first failure "
            "(2026-09-25T08:33Z) to seq 546",
            "bitget_signal": "answered 0 of 939 calls over the whole run",
            "readings_missing": "crypto Fear & Greed and BTCUSDT long/short missing from 146 "
            "of 164 snapshots since that failure",
            "source": OUTAGE_EVIDENCE + ", by " + OUTAGE_COMMAND,
        },
    ),
    *(
        DeclaredChange(
            change_id=change_id,
            kind="policy_amendment",
            title=title,
            detail=detail,
            files=files,
            previous_policy_hash=_HASHES[i],
            new_policy_hash=_HASHES[i + 1],
            evidence=evidence,
        )
        for i, (change_id, title, detail, files, evidence) in enumerate(_AMENDMENTS)
    ),
)


def declaration_for(policy: Policy) -> tuple[PredecessorRun | None, tuple[DeclaredChange, ...]]:
    """What a genesis of ``policy`` declares: run 2's changes for policy v2, nothing otherwise
    (a rehearsal or a test policy follows no run)."""
    if policy.content_hash() == POLICY_V2.content_hash():
        return PREDECESSOR, DECLARED_CHANGES
    return None, ()


__all__ = [
    "ACT_RATE_COMMAND",
    "ACT_RATE_EVIDENCE",
    "DECLARED_CHANGES",
    "PREDECESSOR",
    "REPLAY_COMMAND",
    "REPLAY_EVIDENCE",
    "RUN1_CODE_COMMIT",
    "RUN1_GENESIS_HASH",
    "RUN1_PROMPT_HASHES",
    "declaration_for",
]
