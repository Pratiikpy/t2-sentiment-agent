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
``run2-a1`` (policy amendment, v1 to v2)
    ``funding_zscore`` is evaluated for every universe instrument, not BTCUSDT alone, with the same
    threshold, lookback, cooldown, daily cap and weekend refusal.

Nothing else changes: every guard limit, the fee and edge bar, the mandate, the decision rule, the
system prompt and its no-edge stance, and the prompt files (whose hashes the genesis carries and a
test holds equal to run 1's). The replayed counts in the evidence are recomputed by
``scripts/replay_triggers.py`` from run 1's published ledger, cut at seq 363 (226 snapshots,
2026-09-24 17:06 to 2026-09-25 12:58 UTC), and kept in ``validation/run2/run1_trigger_replay.json``.
"""

from typing import Final

from sentiment_agent.policy import POLICY_V1, POLICY_V2
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
    DeclaredChange(
        change_id="run2-a1",
        kind="policy_amendment",
        title="funding_zscore covers every universe instrument (policy v1 to v2)",
        detail="Policy v1 evaluated the live funding z-score trigger for BTCUSDT only. Policy v2 "
        "evaluates it for all 14 instruments with the same +-2 threshold, 90-settlement "
        "lookback, 240-minute cooldown per instrument, 8 event decisions per UTC day, and the "
        "same weekend refusal of events on US-session legs while G2 holds them flat. No other "
        "policy value changes: every guard limit, the fee and edge bar, the mandate and the "
        "decision rule are v1's.",
        files=("src/sentiment_agent/policy.py", "src/sentiment_agent/events/triggers.py"),
        previous_policy_hash=POLICY_V1.content_hash(),
        new_policy_hash=POLICY_V2.content_hash(),
        evidence={
            "funding_extremes": "live funding z beyond +-2 in 436 equity/index "
            "instrument-snapshots of 226 (HOOD 88, NVDA 84, MSTR 58, GOOGL 49, AMZN 46, SNDK 44, "
            "TSLA 30, COIN 23, NDX100 14); BTCUSDT 0",
            "replayed_event_decisions": "15 in 19.9 hours on run 1's 226 snapshots (7 on "
            "2026-09-24, 8 on 2026-09-25, the daily cap refusing 3 more); 15 also within the "
            "token budget at every decision's worst-case bound; run 1 itself: 0",
            "source": REPLAY_EVIDENCE + " (policy_v2), by " + REPLAY_COMMAND,
        },
    ),
)


def declaration_for(policy: Policy) -> tuple[PredecessorRun | None, tuple[DeclaredChange, ...]]:
    """What a genesis of ``policy`` declares: run 2's changes for policy v2, nothing otherwise
    (a rehearsal or a test policy follows no run)."""
    if policy.content_hash() == POLICY_V2.content_hash():
        return PREDECESSOR, DECLARED_CHANGES
    return None, ()


__all__ = [
    "DECLARED_CHANGES",
    "PREDECESSOR",
    "REPLAY_COMMAND",
    "REPLAY_EVIDENCE",
    "RUN1_CODE_COMMIT",
    "RUN1_GENESIS_HASH",
    "RUN1_PROMPT_HASHES",
    "declaration_for",
]
