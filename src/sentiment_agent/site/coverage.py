"""The Bitget toolkit coverage matrix: every surface the agent touches, and every one it does not.

One :class:`~sentiment_agent.types.ToolkitUse` row per surface entry (win plan Move 21, DESIGN.md
§14.8): what it is used for, which judged line of the handbook it serves, the modules that read it,
its last health, and where on the demo page a judge sees it. Surfaces that were looked at and are
not used are rows too, each with the reason, because "we use Bitget's toolkit in depth" is only
checkable against the list of what is left out.

Three sources of rows, and one rule for each:

* **Declared** (:func:`declared_uses`): written here, from the code that makes the calls. The public
  v3 market endpoints are ``venue/public_api.py``'s paths; every Agent Hub argv is one that
  ``execution/bgc.py`` or ``execution/environment.py`` builds; the two MCP services' rows are
  ``sources/toolkit.py``'s own registry of what it reads (``USED``) and of the tools it measures and
  does not use (``SIGNAL_TOOL_PROBES``), imported rather than copied so the two lists cannot drift;
  the crowd channels are ``crowd/adapters.py``'s two CLIs. ``tests/site/test_coverage.py`` scans the
  source tree and fails when a surface, an endpoint or a ``bgc`` verb and action is called
  somewhere and not declared here.
* **Measured** (:func:`coverage_matrix`): the live probe (``sources.toolkit.probe_all``, run by
  ``t2sa probe-toolkit``) supplies the health of every MCP row and one row per catalog entry of
  bitget-mcp-server the agent does not read, each with its measured answer and its reason. The
  declared purpose, judged line and placement always win over the probe's; the probe contributes
  the measurement.
* **Observed in the log** (:func:`ledger_health`): the public-API and crowd rows take the health of
  the matching calls in the newest logged snapshot, and the Agent Hub rows take it from the order,
  fill, stop, reconciliation and environment-proof events that only a working call could have
  produced. A SIMULATED ledger made no Agent Hub call at all, and its rows say so instead of showing
  a health they do not have.

A row with no ``used_in`` is a surface the agent does not use, and its ``notes`` start with
``not used``. The Agent Hub verbs are the 16 intent verbs and two meta tools of the installed SDK
(``@bitget-ai/bitget-agent-sdk@3.0.0``, ``lib/index.js`` ``COMPOSITE_TOOL_NAMES``, read on
2026-09-24; the Agent Hub README still says 14).
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final

from sentiment_agent.book.projection import Projection
from sentiment_agent.sources.toolkit import SIGNAL_TOOL_PROBES, USED, aggregate_health
from sentiment_agent.types import (
    EnvironmentProof,
    OrderStateChange,
    RunMode,
    SourceCall,
    SourceHealth,
    ToolkitProbe,
    ToolkitSurface,
    ToolkitUse,
    VenueAck,
    VenueRejection,
    VenueUnknown,
)

# ------------------------------------------------------------------------------------------------
# Judged lines and page placements (handbook Track 2, line 248; DESIGN.md §1)
# ------------------------------------------------------------------------------------------------

LINE_EXECUTION: Final = (
    "Agent architecture quality / the Track 2 execution layer the handbook names: Agent Hub with "
    "--paper-trading produces the paper-trading log (handbook:422, :444)"
)
LINE_RISK: Final = "risk control layer effectiveness: an input the kernel rules on"
LINE_EXPLAIN: Final = "decision explainability: shown to the model and on the decision card"
LINE_QUANT: Final = "paper trading Sharpe, max drawdown, win rate: the hourly equity marks"
LINE_SAFETY: Final = (
    "hard safety rule: orders only ever reach Bitget's Demo environment (DESIGN.md §2, §11.2)"
)
LINE_PERCEPTION: Final = (
    "Agent architecture quality: the Track 2 perception layer the handbook names "
    "(bitget-signal + bitget-mcp-server, handbook:396)"
)

AT_CARD_ORDERS: Final = "decision card: orders (dry-run payload, clientOid, venue orderId, fills)"
AT_CARD_POSITIONING: Final = "decision card: positioning table"
AT_CARD_KERNEL: Final = "decision card: kernel ruling, per guard"
AT_CARD_TEXT: Final = "decision card: text shown to the model"
AT_EQUITY: Final = "demo page: equity against every arm"
AT_MIRROR: Final = "demo page: live-marked mirror"
AT_ENVIRONMENT: Final = "demo page: environment proof"
AT_MATRIX: Final = "demo page: toolkit coverage matrix"

NOT_USED: Final = "not used"

_SNAPSHOT: Final = ("perception.snapshot",)


@dataclass(frozen=True, slots=True)
class _Declared:
    surface: ToolkitSurface
    entry: str
    purpose: str
    judged_line: str
    used_in: tuple[str, ...]
    visible_at: str
    notes: str = ""
    log_sources: tuple[str, ...] = ()
    """Prefixes of ``SourceCall.source`` in a logged snapshot that are calls of this entry."""


def _unused(surface: ToolkitSurface, entry: str, purpose: str, reason: str) -> _Declared:
    return _Declared(
        surface=surface,
        entry=entry,
        purpose=purpose,
        judged_line="",
        used_in=(),
        visible_at=AT_MATRIX,
        notes=f"{NOT_USED}: {reason}",
    )


# ------------------------------------------------------------------------------------------------
# Bitget public v3 market API (venue/public_api.py), keyless
# ------------------------------------------------------------------------------------------------

_PUBLIC = ToolkitSurface.PUBLIC_MARKET_API
_DEMO_HEADER = "Demo, header paptrading: 1"

PUBLIC_ROWS: Final[tuple[_Declared, ...]] = (
    _Declared(
        _PUBLIC,
        f"GET /api/v3/market/tickers ({_DEMO_HEADER})",
        "Demo last, mark, index, bid and ask: the venue the orders go to. Marks the book every "
        "hour, fills the simulated venue, and feeds the venue-integrity (G1), spread (G8) and "
        "staleness (G10) checks",
        LINE_RISK,
        (*_SNAPSHOT, "book.marks.mark_point", "kernel.guards", "execution.simulated"),
        AT_CARD_KERNEL,
        log_sources=("public_v3.tickers[demo]", "tickers.demo:"),
    ),
    _Declared(
        _PUBLIC,
        "GET /api/v3/market/tickers (live)",
        "Live last, mark, funding and open interest: the real crowd's positioning (DESIGN.md "
        "§6.1), the Demo-live gap G1 checks, and the live price of every open position in the "
        "hourly mirror",
        LINE_EXPLAIN,
        (*_SNAPSHOT, "perception.features.build_features", "kernel.guards", "book.marks"),
        AT_CARD_POSITIONING,
        log_sources=("public_v3.tickers[live]", "tickers.live:"),
    ),
    _Declared(
        _PUBLIC,
        f"GET /api/v3/market/instruments ({_DEMO_HEADER})",
        "Demo order limits and fees (minOrderQty, quantityMultiplier, priceMultiplier, "
        "minOrderAmount, maxMarketOrderQty, takerFeeRate): eligibility (G11), rounding, splits "
        "and the expected fee of every order",
        LINE_RISK,
        ("runtime.wiring", "kernel.guards", "kernel.planner"),
        AT_CARD_ORDERS,
        log_sources=("instruments.demo:",),
    ),
    _Declared(
        _PUBLIC,
        "GET /api/v3/market/history-candles type=market interval=1H (live)",
        "Live hourly closes: the overheating feature (distance from the 20-bar mean in ATR "
        "units) and the prices the live mirror and the weekend counterfactual are marked at",
        LINE_EXPLAIN,
        ("perception.features.ma_distance_atr", "analysis.mirror"),
        AT_MIRROR,
        log_sources=("public_v3.history_candles[live,market", "history-candles.live:"),
    ),
    _Declared(
        _PUBLIC,
        f"GET /api/v3/market/history-candles type=index interval=1H ({_DEMO_HEADER})",
        "Demo index closes: the stale-index detector that refuses new exposure when the Demo "
        "index stops moving while its session should be open (G1)",
        LINE_RISK,
        ("perception.features.index_move_bps_3h", "kernel.guards"),
        AT_CARD_KERNEL,
        log_sources=("public_v3.history_candles[demo,index", "history-candles.demo:"),
    ),
    _Declared(
        _PUBLIC,
        f"GET /api/v3/market/history-candles type=mark interval=1H ({_DEMO_HEADER})",
        "Demo hourly marks: the prices the twin, the fixed-rule baselines and the rival arms "
        "are simulated and marked at, so every arm is read on the book's own venue",
        LINE_QUANT,
        ("analysis.armsim",),
        AT_EQUITY,
    ),
    _Declared(
        _PUBLIC,
        "GET /api/v3/market/history-fund-rate (live)",
        "Live funding history: the funding z-score over the last 90 settlements, shown to the "
        "model and the funding event trigger",
        LINE_EXPLAIN,
        ("perception.features.funding_z", "events.triggers"),
        AT_CARD_POSITIONING,
        log_sources=("public_v3.history_fund_rate[live]", "history-fund-rate.live:"),
    ),
    _unused(
        _PUBLIC,
        "GET /api/v3/market/instruments (live)",
        "live order limits",
        "orders go to Demo, so Demo's limits and fees are the ones that bind; the live limits "
        "were measured once for the design (validation/demo_venue/universe_probe.json)",
    ),
    _unused(
        _PUBLIC,
        f"GET /api/v3/market/history-fund-rate ({_DEMO_HEADER})",
        "Demo funding history",
        "Demo funding is a sandbox setting, not a crowd: SBTCSPERP fixed at -0.10% on 270 of 270 "
        "settlements and BTCPERP at the +30 bps cap (DESIGN.md §6.1)",
    ),
)

# ------------------------------------------------------------------------------------------------
# Agent Hub: bgc, the pinned Bitget CLI (execution/bgc.py, execution/environment.py)
# ------------------------------------------------------------------------------------------------

_BGC = ToolkitSurface.AGENT_HUB_BGC
_PAPER = "--paper-trading"
_TRANSPORT = ("execution.bgc.BgcTransport",)

BGC_ENTRY_PREVIEW: Final = "order --action place --dry-run --paper-trading"
BGC_ENTRY_PLACE: Final = "order --action place --paper-trading"
BGC_ENTRY_DETAIL: Final = "order --action detail --paper-trading"
BGC_ENTRY_FILLS: Final = "order --action fills --paper-trading"
BGC_ENTRY_HISTORY: Final = "order --action history --paper-trading"
BGC_ENTRY_POSITION: Final = "position --action info --paper-trading"
BGC_ENTRY_STOPS_OPEN: Final = "strategy_order --action open --paper-trading"
BGC_ENTRY_STOP_PLACE: Final = "strategy_order --action place --paper-trading"
BGC_ENTRY_STOP_CANCEL: Final = "strategy_order --action cancel --paper-trading"
BGC_ENTRY_ACCOUNT: Final = "account_overview --paper-trading"
BGC_ENTRY_ASSETS_DEMO: Final = "raw --operationId getAccountAssets --paper-trading"
BGC_ENTRY_ASSETS_LIVE: Final = "raw --operationId getAccountAssets --read-only"

BGC_USED_ROWS: Final[tuple[_Declared, ...]] = (
    _Declared(
        _BGC,
        BGC_ENTRY_PREVIEW,
        "the dry-run preview of every order: Agent Hub builds the exact request (wouldSend) with "
        "no network call, the preview is logged, and the order is sent only if it matches the "
        "approved intent",
        LINE_EXECUTION,
        ("execution.executor.Executor", *_TRANSPORT),
        AT_CARD_ORDERS,
    ),
    _Declared(
        _BGC,
        BGC_ENTRY_PLACE,
        "every order: market (taker) only, with a preset stop on every exposure-adding leg, "
        "reduce-only on every reducing leg, clientOid derived from the approved intent",
        LINE_EXECUTION,
        ("execution.executor.Executor", *_TRANSPORT),
        AT_CARD_ORDERS,
    ),
    _Declared(
        _BGC,
        BGC_ENTRY_DETAIL,
        "the order read back by clientOid (or orderId) until it is terminal; resolves an "
        "UNKNOWN send without ever resending it",
        LINE_EXECUTION,
        ("execution.executor.Executor", "execution.reconcile.Reconciler", *_TRANSPORT),
        AT_CARD_ORDERS,
    ),
    _Declared(
        _BGC,
        BGC_ENTRY_FILLS,
        "fills: execution price, quantity, fee and realised P&L as the venue reports them; the "
        "book is built from these alone",
        LINE_QUANT,
        ("execution.executor.Executor", "execution.reconcile.Reconciler", *_TRANSPORT),
        AT_CARD_ORDERS,
    ),
    _Declared(
        _BGC,
        BGC_ENTRY_HISTORY,
        "the daily full order history, so an order the agent did not see acknowledged is still "
        "found and reconciled",
        LINE_EXECUTION,
        ("execution.reconcile.Reconciler", *_TRANSPORT),
        AT_CARD_ORDERS,
    ),
    _Declared(
        _BGC,
        BGC_ENTRY_POSITION,
        "the venue's own positions, compared with the book built from fills; every difference "
        "is a logged discrepancy",
        LINE_RISK,
        ("execution.reconcile.Reconciler", *_TRANSPORT),
        AT_CARD_ORDERS,
    ),
    _Declared(
        _BGC,
        BGC_ENTRY_STOPS_OPEN,
        "the venue's open stop orders (type tpsl), so every position has exactly one stop (G4)",
        LINE_RISK,
        ("execution.stops.StopManager", "execution.reconcile.Reconciler", *_TRANSPORT),
        AT_CARD_KERNEL,
    ),
    _Declared(
        _BGC,
        BGC_ENTRY_STOP_PLACE,
        "a full-position stop, triggered on the Demo mark, placed or replaced after a fill "
        "changes a position (G4: a stop on the venue survives a crashed agent)",
        LINE_RISK,
        ("execution.stops.StopManager", *_TRANSPORT),
        AT_CARD_KERNEL,
    ),
    _Declared(
        _BGC,
        BGC_ENTRY_STOP_CANCEL,
        "cancels an orphaned or superseded stop, only after its replacement is in place",
        LINE_RISK,
        ("execution.stops.StopManager", *_TRANSPORT),
        AT_CARD_KERNEL,
    ),
    _Declared(
        _BGC,
        BGC_ENTRY_ACCOUNT,
        "the Demo account in one read: the environment proof's Demo-positive check, the hold "
        "mode, the starting equity and the venue equity published beside the book's",
        LINE_SAFETY,
        ("execution.environment.prove_environment", "execution.reconcile.Reconciler"),
        AT_ENVIRONMENT,
    ),
    _Declared(
        _BGC,
        BGC_ENTRY_ASSETS_DEMO,
        "the same account read as one operation, so a refusal keeps its error code: a 40099 "
        "here means the key is not a Demo key, and the agent exits",
        LINE_SAFETY,
        ("execution.environment.prove_environment",),
        AT_ENVIRONMENT,
    ),
    _Declared(
        _BGC,
        BGC_ENTRY_ASSETS_LIVE,
        "the environment proof's live-negative read: the Demo key must be refused by the live "
        "environment. The only argv without --paper-trading, and --read-only makes bgc refuse "
        "any write before the network (agent-sdk safety.ts:90-95)",
        LINE_SAFETY,
        ("execution.environment.prove_environment",),
        AT_ENVIRONMENT,
    ),
    _Declared(
        _BGC,
        "flag --paper-trading",
        "routes every call to Bitget's Demo environment (sends paptrading: 1 on private calls, "
        "agent-sdk rest-client.ts:274-280); the argv builder appends it as a constant and the "
        "environment proof re-reads the installed SDK source to confirm it",
        LINE_SAFETY,
        ("execution.bgc", "execution.environment.confirm_paptrading_header"),
        AT_ENVIRONMENT,
    ),
)

BGC_UNUSED_ROWS: Final[tuple[_Declared, ...]] = (
    _unused(
        _BGC,
        "market",
        "Bitget market data through the CLI",
        "the same public v3 endpoints are read directly over HTTPS, keyless, with every raw "
        "response stored as a blob; a child process per read would add nothing the log needs",
    ),
    _unused(
        _BGC,
        "account_config",
        "leverage, margin and hold-mode changes",
        "the agent reads the hold mode (account_overview) and never changes account settings; "
        "sizing is by notional against equity, not by leverage",
    ),
    _unused(
        _BGC,
        "transfer_funds",
        "internal transfers",
        "a paper book moves no funds",
    ),
    _unused(
        _BGC,
        "deposit",
        "deposit addresses and records",
        "a paper book moves no funds",
    ),
    _unused(
        _BGC,
        "withdraw",
        "withdrawals",
        "a paper book moves no funds, and the Demo key is created without the Withdraw "
        "permission (DESIGN.md §21)",
    ),
    _unused(
        _BGC,
        "funds_records",
        "funding and transfer records",
        "the book is built from fills; the hourly venue equity (account_overview) captures any "
        "funding credit as a published gap",
    ),
    _unused(_BGC, "repayment", "margin loan repayment", "no margin loan is taken"),
    _unused(_BGC, "subaccount", "sub-account management", "one Demo account, no sub-accounts"),
    _unused(_BGC, "broker", "broker operations", "not a broker integration"),
    _unused(_BGC, "loan", "crypto loans", "no loan is taken"),
    _unused(_BGC, "inst_loan", "institutional loans", "no loan is taken"),
    _unused(_BGC, "tax", "tax records", "outside the trading loop"),
    _unused(
        _BGC,
        "discover",
        "the CLI's live catalog of operations",
        "every argv is fixed in execution/bgc.py and checked against recorded dry-run output of "
        "the pinned 3.0.0 CLI; nothing is discovered at run time",
    ),
    _unused(
        _BGC,
        "bitget-agent-mcp (the same SDK over MCP)",
        "Agent Hub as an MCP server",
        "the CLI's argv is logged verbatim beside each order and pinned by the lockfile; the MCP "
        "server drives the same SDK operations with nothing to add for an unattended loop",
    ),
    _unused(
        _BGC,
        "Agentic account (OAuth, bitget-agent-mcp)",
        "the live Agentic sub-account",
        "Track 2 runs on a Demo API key; the Agentic account trades live funds (win plan §3.6, "
        "the owner's decision alone)",
    ),
)

# ------------------------------------------------------------------------------------------------
# bitget-signal and bitget-mcp-server (sources/toolkit.py)
# ------------------------------------------------------------------------------------------------

SIGNAL_SKILL: Final[Mapping[str, str]] = {
    "sentiment_index": "sentiment-analyst",
    "derivatives_sentiment": "sentiment-analyst (also market-intel, news-briefing)",
    "news_feed": "news-briefing (also market-intel)",
    "social_trending": "news-briefing",
    "tradfi_news": "news-briefing, macro-analyst",
    "crypto_market": "market-intel",
    "defi_analytics": "market-intel",
    "dex_market": "market-intel",
    "network_status": "market-intel",
    "global_assets": "macro-analyst",
    "macro_indicators": "macro-analyst",
    "rates_yields": "macro-analyst",
    "cross_asset": "macro-analyst",
    "cn_market": "macro-analyst",
    "global_data": "macro-analyst",
    "technical_analysis": "technical-analysis",
    "backtest": "technical-analysis",
}
"""Which of bitget-signal's five Skills documents each tool (``bitget-signal/skills/*/SKILL.md``,
tool names counted 2026-09-24). ``crypto_price`` and ``crypto_derivatives`` are named by none."""

CATALOG_PLACEHOLDER: Final = "catalog: every entry the agent does not read"


def _skill_note(entry: str) -> str:
    tool = entry.split(".", 1)[0]
    skill = SIGNAL_SKILL.get(tool)
    return f"Skill: {skill}" if skill else "Skill: none of the five names this tool"


def _mcp_rows() -> tuple[_Declared, ...]:
    rows: list[_Declared] = []
    for (surface, source), declared in USED.items():
        notes = _skill_note(source) if surface is ToolkitSurface.SIGNAL_MCP else ""
        rows.append(
            _Declared(
                surface=surface,
                entry=source,
                purpose=declared.purpose,
                judged_line=declared.judged_line,
                used_in=declared.used_in,
                visible_at=declared.visible_at,
                notes=notes,
                log_sources=(source,),
            )
        )
    rows.append(
        _Declared(
            ToolkitSurface.DATA_MCP,
            "guide",
            "the catalog of every bitget-mcp-server entry, read by the toolkit probe so every "
            "entry the agent does not read is measured and listed with its reason",
            LINE_PERCEPTION,
            ("sources.toolkit.probe_all",),
            AT_MATRIX,
        )
    )
    for tool, arguments, reason in SIGNAL_TOOL_PROBES:
        action = arguments.get("action")
        entry = f"{tool}.{action}" if isinstance(action, str) else tool
        text = reason.removeprefix(f"{NOT_USED}: ")
        rows.append(
            _Declared(
                surface=ToolkitSurface.SIGNAL_MCP,
                entry=entry,
                purpose="measured, not used",
                judged_line="",
                used_in=(),
                visible_at=AT_MATRIX,
                notes=f"{NOT_USED}: {text}; {_skill_note(entry)}",
            )
        )
    rows.append(
        _unused(
            ToolkitSurface.DATA_MCP,
            CATALOG_PLACEHOLDER,
            "the rest of the bitget-mcp-server catalog",
            "not a pre-registered input of policy v1. The toolkit probe (t2sa probe-toolkit) "
            "measures every such entry once and lists each with its answer and its own reason; "
            "this row stands in until a probe has run",
        )
    )
    return tuple(rows)


# ------------------------------------------------------------------------------------------------
# GetAgent Playbook and the crowd channels
# ------------------------------------------------------------------------------------------------

OTHER_ROWS: Final[tuple[_Declared, ...]] = (
    _unused(
        ToolkitSurface.GETAGENT_PLAYBOOK,
        "Playbook replica (runtime_profile llm_bounded, official_evidence_kind paper)",
        "the same policy run as a Bitget-hosted Playbook, whose Sharpe, drawdown and win rate "
        "Bitget computes itself (win plan Move 16)",
        "conditional (DESIGN.md M17): it needs llm_bounded enabled on the owner's GetAgent "
        "account and a Playbook key; until then no Playbook runs and none is claimed",
    ),
    _Declared(
        ToolkitSurface.CROWD_X,
        "twitter-cli search <cashtag> --type Latest --exclude retweets",
        "posts naming each universe instrument, screened by quarantine and clustered for "
        "coordination before the model sees them; an optional layer (positioning is the "
        "backbone, win plan §3.2)",
        LINE_EXPLAIN,
        ("crowd.adapters.XCollector", "crowd.quarantine", "crowd.novelty", *_SNAPSHOT),
        AT_CARD_TEXT,
        notes="not Bitget: listed because it feeds the same decision",
        log_sources=("twitter-cli:search:",),
    ),
    _Declared(
        ToolkitSurface.CROWD_REDDIT,
        "rdt-cli search <alias> --sort new --json",
        "Reddit posts naming each universe instrument, screened and clustered like X; one "
        "account is one source however many subreddits it posts in",
        LINE_EXPLAIN,
        ("crowd.adapters.RedditCollector", "crowd.quarantine", "crowd.novelty", *_SNAPSHOT),
        AT_CARD_TEXT,
        notes="not Bitget: listed because it feeds the same decision",
        log_sources=("rdt-cli:search:",),
    ),
)

SURFACE_ORDER: Final[tuple[ToolkitSurface, ...]] = (
    ToolkitSurface.AGENT_HUB_BGC,
    ToolkitSurface.PUBLIC_MARKET_API,
    ToolkitSurface.SIGNAL_MCP,
    ToolkitSurface.DATA_MCP,
    ToolkitSurface.GETAGENT_PLAYBOOK,
    ToolkitSurface.CROWD_X,
    ToolkitSurface.CROWD_REDDIT,
)


def _declared() -> tuple[_Declared, ...]:
    return (*BGC_USED_ROWS, *BGC_UNUSED_ROWS, *PUBLIC_ROWS, *_mcp_rows(), *OTHER_ROWS)


def _use(row: _Declared) -> ToolkitUse:
    return ToolkitUse(
        surface=row.surface,
        entry=row.entry,
        purpose=row.purpose,
        judged_line=row.judged_line,
        used_in=row.used_in,
        last_health=None,
        last_checked_at=None,
        visible_at=row.visible_at,
        notes=row.notes,
    )


def _ordered(rows: Iterable[ToolkitUse]) -> tuple[ToolkitUse, ...]:
    rank = {surface: i for i, surface in enumerate(SURFACE_ORDER)}
    indexed = list(enumerate(rows))
    indexed.sort(key=lambda item: (rank[item[1].surface], 0 if item[1].used_in else 1, item[0]))
    return tuple(row for _, row in indexed)


def is_used(row: ToolkitUse) -> bool:
    """A row the agent reads or calls (it names the modules that do)."""
    return bool(row.used_in)


def declared_uses() -> tuple[ToolkitUse, ...]:
    """Every surface entry the agent uses, and every one it deliberately does not, with no health
    (nothing has been measured). Ordered by surface, used rows first."""
    rows = [_use(row) for row in _declared()]
    keys = [(r.surface, r.entry) for r in rows]
    if len(keys) != len(set(keys)):  # pragma: no cover - a defect in the tables above
        raise ValueError("a toolkit entry is declared twice")
    return _ordered(rows)


def coverage_matrix(probe: ToolkitProbe | None) -> tuple[ToolkitUse, ...]:
    """The declared rows with the probe's measurements, plus a row for everything the probe measured
    that is not declared (the unused catalog entries, each with its reason).

    The probe supplies ``last_health``, ``last_checked_at`` and its measured detail (appended to the
    declared notes); the declared purpose, judged line, modules and placement are kept. Without a
    probe the declared rows are returned unmeasured.
    """
    rows = list(declared_uses())
    if probe is None:
        return tuple(rows)
    measured = {(r.surface, r.entry): r for r in probe.rows}
    out: list[ToolkitUse] = []
    for row in rows:
        found = measured.pop((row.surface, row.entry), None)
        if found is None:
            out.append(row)
            continue
        notes = "; ".join(part for part in (row.notes, found.notes) if part)
        out.append(
            row.model_copy(
                update={
                    "last_health": found.last_health,
                    "last_checked_at": found.last_checked_at or probe.at,
                    "notes": notes,
                }
            )
        )
    catalog_measured = any(
        r.surface is ToolkitSurface.DATA_MCP and not r.used_in and r.entry != "guide"
        for r in measured.values()
    )
    if catalog_measured:
        out = [
            r
            for r in out
            if not (r.surface is ToolkitSurface.DATA_MCP and r.entry == CATALOG_PLACEHOLDER)
        ]
    for extra in measured.values():
        if extra.used_in:
            out.append(extra)
        else:
            notes = (
                extra.notes if extra.notes.startswith(NOT_USED) else f"{NOT_USED}: {extra.notes}"
            )
            out.append(extra.model_copy(update={"notes": notes}))
    return _ordered(out)


# ------------------------------------------------------------------------------------------------
# Health observed in the ledger
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Observed:
    health: SourceHealth
    at: datetime
    detail: str


def _newer(row: ToolkitUse, seen: _Observed | None) -> ToolkitUse:
    if seen is None:
        return row
    if row.last_checked_at is not None and row.last_checked_at > seen.at:
        return row
    notes = "; ".join(part for part in (row.notes, f"in the log: {seen.detail}") if part)
    return row.model_copy(
        update={"last_health": seen.health, "last_checked_at": seen.at, "notes": notes}
    )


def _from_calls(
    calls: Sequence[SourceCall], prefixes: Sequence[str], at: datetime
) -> _Observed | None:
    matching = [c for c in calls if any(c.source.startswith(p) for p in prefixes)]
    if not matching:
        return None
    health, detail = aggregate_health(matching)
    return _Observed(health, at, f"newest snapshot, {detail}")


def _bgc_observations(projection: Projection) -> dict[str, _Observed]:
    seen: dict[str, _Observed] = {}
    previews = projection.previews
    if previews:
        last = previews[-1]
        seen[BGC_ENTRY_PREVIEW] = _Observed(
            SourceHealth.OK, last.captured_at, f"{len(previews)} dry-run preview(s) logged"
        )
    answers: list[VenueAck | VenueRejection | VenueUnknown] = [
        p
        for p in projection.order_events
        if isinstance(p, VenueAck | VenueRejection | VenueUnknown)
    ]
    if answers:
        last_answer = answers[-1]
        acks = sum(1 for a in answers if isinstance(a, VenueAck))
        rejected = sum(1 for a in answers if isinstance(a, VenueRejection))
        unknown = sum(1 for a in answers if isinstance(a, VenueUnknown))
        if isinstance(last_answer, VenueAck):
            health, at = SourceHealth.OK, last_answer.acked_at
        elif isinstance(last_answer, VenueRejection):
            health, at = SourceHealth.ERROR, last_answer.at
        else:
            health, at = SourceHealth.TIMEOUT, last_answer.at
        seen[BGC_ENTRY_PLACE] = _Observed(
            health, at, f"{acks} acknowledged, {rejected} rejected, {unknown} unknown"
        )
    reads = [
        c
        for c in projection.order_events
        if isinstance(c, OrderStateChange) and c.reason.startswith("venue orderStatus")
    ]
    if reads:
        seen[BGC_ENTRY_DETAIL] = _Observed(
            SourceHealth.OK, reads[-1].at, f"{len(reads)} order state(s) read back from the venue"
        )
    fills = projection.fills
    if fills:
        seen[BGC_ENTRY_FILLS] = _Observed(
            SourceHealth.OK, max(f.executed_at for f in fills), f"{len(fills)} fill(s) read"
        )
    reports = projection.reconciliations
    if reports:
        latest = reports[-1]
        detail = f"{len(reports)} reconciliation(s); latest found {len(latest.discrepancies)} "
        detail += "discrepancy(ies)"
        for entry in (BGC_ENTRY_POSITION, BGC_ENTRY_STOPS_OPEN, BGC_ENTRY_HISTORY):
            seen[entry] = _Observed(SourceHealth.OK, latest.at, detail)
    syncs = projection.stop_syncs
    placed = [s for s in syncs if s.action in ("placed", "replaced")]
    if placed:
        seen[BGC_ENTRY_STOP_PLACE] = _Observed(
            SourceHealth.OK, placed[-1].at, f"{len(placed)} stop(s) placed or replaced"
        )
    cancelled = [s for s in syncs if s.action == "cancelled"]
    if cancelled:
        seen[BGC_ENTRY_STOP_CANCEL] = _Observed(
            SourceHealth.OK, cancelled[-1].at, f"{len(cancelled)} stop(s) cancelled"
        )
    proofs = projection.environment_proofs
    if proofs:
        proof: EnvironmentProof = proofs[-1]
        demo = SourceHealth.OK if proof.demo_read_ok else SourceHealth.ERROR
        code = f" (code {proof.demo_read_code})" if proof.demo_read_code else ""
        seen[BGC_ENTRY_ACCOUNT] = _Observed(
            demo, proof.checked_at, f"environment proof: Demo read ok={proof.demo_read_ok}{code}"
        )
        seen[BGC_ENTRY_ASSETS_DEMO] = seen[BGC_ENTRY_ACCOUNT]
        live = SourceHealth.OK if proof.live_read_rejected else SourceHealth.ERROR
        live_code = f" (code {proof.live_read_code})" if proof.live_read_code else ""
        seen[BGC_ENTRY_ASSETS_LIVE] = _Observed(
            live,
            proof.checked_at,
            f"environment proof: the live environment refused the Demo key="
            f"{proof.live_read_rejected}{live_code} (a refusal is the expected answer)",
        )
        seen["flag --paper-trading"] = _Observed(
            SourceHealth.OK if proof.paptrading_header_confirmed else SourceHealth.ERROR,
            proof.checked_at,
            f"installed SDK sends paptrading: 1 = {proof.paptrading_header_confirmed}",
        )
    return seen


def ledger_health(rows: Sequence[ToolkitUse], projection: Projection) -> tuple[ToolkitUse, ...]:
    """``rows`` with the health the ledger itself shows (module docstring).

    A row keeps its own health when that measurement is newer than what the log shows. In a
    SIMULATED ledger no Agent Hub call was made, so those rows keep their health and say why the log
    has none.
    """
    snapshots = projection.snapshots
    newest = snapshots[-1] if snapshots else None
    declared = {(d.surface, d.entry): d for d in _declared()}
    bgc_seen: dict[str, _Observed] = {}
    simulated = projection.mode is RunMode.SIMULATED
    if not simulated:
        bgc_seen = _bgc_observations(projection)
    out: list[ToolkitUse] = []
    for row in rows:
        if row.surface is ToolkitSurface.AGENT_HUB_BGC:
            if simulated and row.used_in:
                note = "SIMULATED ledger: no Agent Hub call was made"
                out.append(
                    row.model_copy(update={"notes": "; ".join(p for p in (row.notes, note) if p)})
                )
            else:
                out.append(_newer(row, bgc_seen.get(row.entry)))
            continue
        spec = declared.get((row.surface, row.entry))
        if newest is None or spec is None or not spec.log_sources:
            out.append(row)
            continue
        out.append(_newer(row, _from_calls(newest.source_calls, spec.log_sources, newest.taken_at)))
    return tuple(out)


def coverage_counts(rows: Sequence[ToolkitUse]) -> dict[str, dict[str, int]]:
    """Per surface: rows used, rows not used, and used rows whose last health is OK."""
    counts: dict[str, dict[str, int]] = {}
    for surface in SURFACE_ORDER:
        mine = [r for r in rows if r.surface is surface]
        counts[surface.value] = {
            "used": sum(1 for r in mine if is_used(r)),
            "not_used": sum(1 for r in mine if not is_used(r)),
            "used_ok": sum(1 for r in mine if is_used(r) and r.last_health is SourceHealth.OK),
            "used_unmeasured": sum(1 for r in mine if is_used(r) and r.last_health is None),
        }
    return counts


__all__ = [
    "BGC_ENTRY_ACCOUNT",
    "BGC_ENTRY_ASSETS_DEMO",
    "BGC_ENTRY_ASSETS_LIVE",
    "BGC_ENTRY_DETAIL",
    "BGC_ENTRY_FILLS",
    "BGC_ENTRY_HISTORY",
    "BGC_ENTRY_PLACE",
    "BGC_ENTRY_POSITION",
    "BGC_ENTRY_PREVIEW",
    "BGC_ENTRY_STOPS_OPEN",
    "BGC_ENTRY_STOP_CANCEL",
    "BGC_ENTRY_STOP_PLACE",
    "CATALOG_PLACEHOLDER",
    "NOT_USED",
    "SIGNAL_SKILL",
    "SURFACE_ORDER",
    "coverage_counts",
    "coverage_matrix",
    "declared_uses",
    "is_used",
    "ledger_health",
]
