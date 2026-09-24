"""The Bitget toolkit coverage matrix: nothing the source calls is left out, nothing left out is
unexplained, and the health shown is the health measured."""

import html
import json
import re
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from decision.support import build_snapshot
from execution.fakes import FakeMarket
from helpers import T0, make_intent
from sentiment_agent.book.projection import Projection
from sentiment_agent.clock import ManualClock
from sentiment_agent.execution import bgc, environment
from sentiment_agent.execution.simulated import SimulatedVenue
from sentiment_agent.ledger.chain import HashChainLedger, ledger_path
from sentiment_agent.policy import POLICY_V1
from sentiment_agent.site.coverage import (
    BGC_ENTRY_ACCOUNT,
    BGC_ENTRY_ASSETS_LIVE,
    BGC_ENTRY_PLACE,
    BGC_ENTRY_PREVIEW,
    CATALOG_PLACEHOLDER,
    NOT_USED,
    coverage_counts,
    coverage_matrix,
    declared_uses,
    ledger_health,
)
from sentiment_agent.sources.toolkit import SIGNAL_TOOL_PROBES, USED
from sentiment_agent.types import (
    AccountSnapshot,
    EnvironmentProof,
    EventKind,
    Fill,
    FillVenue,
    RunMode,
    Side,
    SnapshotEvent,
    SourceCall,
    SourceHealth,
    ToolkitProbe,
    ToolkitSurface,
    ToolkitUse,
    VenueAck,
)
from sentiment_agent.venue import public_api
from site_world import Site

SRC = Path(__file__).resolve().parents[2] / "src" / "sentiment_agent"
INSTALLED_SDK = (
    Path(__file__).resolve().parents[2]
    / "tools/agent-hub/node_modules/@bitget-ai/bitget-agent-sdk/lib/index.js"
)

AGENT_HUB_VERBS = (
    "market",
    "order",
    "position",
    "strategy_order",
    "account_overview",
    "account_config",
    "repayment",
    "transfer_funds",
    "deposit",
    "withdraw",
    "funds_records",
    "subaccount",
    "broker",
    "loan",
    "inst_loan",
    "tax",
)
"""``COMPOSITE_TOOL_NAMES`` of ``@bitget-ai/bitget-agent-sdk@3.0.0`` (``lib/index.js``), the pinned
release; the meta tools ``discover`` and ``raw`` come on top."""


def rows() -> tuple[ToolkitUse, ...]:
    return declared_uses()


def used(surface: ToolkitSurface) -> list[ToolkitUse]:
    return [r for r in rows() if r.surface is surface and r.used_in]


def source_files() -> list[Path]:
    return [p for p in SRC.rglob("*.py") if "site" not in p.relative_to(SRC).parts]


# ------------------------------------------------------------------------------------------------
# Source scan: everything called is declared
# ------------------------------------------------------------------------------------------------


def test_every_surface_the_source_touches_has_a_used_row() -> None:
    referenced: set[str] = set()
    for path in source_files():
        if path.name == "types.py":
            continue
        referenced |= set(re.findall(r"ToolkitSurface\.([A-Z_]+)", path.read_text("utf-8")))
    assert referenced, "the scan found no surface at all"
    for name in sorted(referenced):
        surface = ToolkitSurface[name]
        assert used(surface), f"{surface.value} is called in src but has no used row"


def test_every_public_endpoint_is_declared_used() -> None:
    paths = {
        value
        for name, value in vars(public_api).items()
        if name.startswith("PATH_") and isinstance(value, str)
    }
    text = public_api.__file__ and Path(public_api.__file__).read_text("utf-8")
    assert paths == set(re.findall(r'"(/api/v3/market/[a-z-]+)"', text or ""))
    entries = [r.entry for r in used(ToolkitSurface.PUBLIC_MARKET_API)]
    for path in sorted(paths):
        assert any(path in entry for entry in entries), f"{path} has no used row"


def _declared_verbs() -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for row in rows():
        if row.surface is not ToolkitSurface.AGENT_HUB_BGC or not row.used_in:
            continue
        words = row.entry.split()
        if len(words) >= 3 and words[1] in ("--action", "--operationId"):
            out.add((words[0], words[2]))
        elif words:
            out.add((words[0], ""))
    return out


def _argv_verb(argv: list[str] | tuple[str, ...]) -> tuple[str, str]:
    if len(argv) >= 3 and argv[1] in ("--action", "--operationId"):
        return argv[0], argv[2]
    return argv[0], ""


def test_every_bgc_argv_the_source_builds_is_declared_used() -> None:
    intent = make_intent()
    built = [
        bgc.build_place_args(intent, hold_mode="one_way_mode", dry_run=True),
        bgc.build_place_args(intent, hold_mode="one_way_mode", dry_run=False),
        bgc.detail_args(intent.client_oid),
        bgc.detail_by_order_id_args("123456"),
        bgc.fills_args("1", "2"),
        bgc.history_args("1", "2"),
        bgc.positions_args(),
        bgc.stop_orders_args(),
        bgc.account_args(),
        bgc.place_stop_args(
            symbol="NVDAUSDT",
            pos_side="short",
            qty=Decimal("1"),
            stop_price=Decimal("230"),
            client_oid=intent.client_oid,
        ),
        bgc.cancel_stop_args("123456"),
        list(environment.DEMO_OVERVIEW_ARGS),
        list(environment.DEMO_ASSETS_ARGS),
        list(environment.LIVE_NEGATIVE_ARGS),
    ]
    declared = _declared_verbs()
    for argv in built:
        assert _argv_verb(argv) in declared, f"{' '.join(argv[:3])} is built but not declared"
    assert "--paper-trading" in bgc.build_place_args(intent, hold_mode="one_way_mode", dry_run=True)


def test_every_bgc_verb_written_in_the_source_is_declared_used() -> None:
    """A text scan, so an argv builder added later cannot slip past the list above."""
    found: set[tuple[str, str]] = set()
    for path in source_files():
        text = path.read_text("utf-8")
        found |= set(re.findall(r'"([a-z_]+)",\s*"--action",\s*"([a-z_]+)"', text))
        found |= {("order", a) for a in re.findall(r'_window_args\(\s*"([a-z_]+)"', text)}
        found |= set(re.findall(r'"(raw)",\s*"--operationId",\s*"([A-Za-z]+)"', text))
        found |= {(v, "") for v in re.findall(r'_paper\(\s*"(account_overview)"', text)}
    assert ("order", "place") in found
    assert ("raw", "getAccountAssets") in found
    declared = _declared_verbs()
    missing = sorted(found - declared)
    assert not missing, f"bgc calls in src with no used row: {missing}"


def test_every_agent_hub_intent_verb_is_used_or_explained() -> None:
    if INSTALLED_SDK.is_file():
        text = INSTALLED_SDK.read_text("utf-8")
        match = re.search(r"COMPOSITE_TOOL_NAMES = \[(.*?)\]", text, re.S)
        assert match is not None
        assert tuple(re.findall(r'"([a-z_]+)"', match.group(1))) == AGENT_HUB_VERBS
    bgc_rows = [r for r in rows() if r.surface is ToolkitSurface.AGENT_HUB_BGC]
    first_words = {r.entry.split()[0] for r in bgc_rows}
    for verb in (*AGENT_HUB_VERBS, "discover", "raw"):
        assert verb in first_words, f"Agent Hub verb {verb} is neither used nor explained"


def test_every_mcp_source_the_toolkit_reads_is_declared_used() -> None:
    declared = {(r.surface, r.entry) for r in rows() if r.used_in}
    for key in USED:
        assert key in declared
    unused = {r.entry for r in rows() if r.surface is ToolkitSurface.SIGNAL_MCP and not r.used_in}
    for tool, arguments, _ in SIGNAL_TOOL_PROBES:
        action = arguments.get("action")
        assert (f"{tool}.{action}" if isinstance(action, str) else tool) in unused


def test_both_crowd_clis_are_declared_used() -> None:
    from sentiment_agent.crowd.adapters import RedditCollector, XCollector

    for surface, program in (
        (ToolkitSurface.CROWD_X, "twitter"),
        (ToolkitSurface.CROWD_REDDIT, "rdt"),
    ):
        assert any(r.entry.startswith(f"{program}") for r in used(surface))
    assert XCollector.surface is ToolkitSurface.CROWD_X
    assert RedditCollector.surface is ToolkitSurface.CROWD_REDDIT


def test_every_surface_has_a_row_and_every_unused_row_has_a_reason() -> None:
    assert {r.surface for r in rows()} == set(ToolkitSurface)
    keys = [(r.surface, r.entry) for r in rows()]
    assert len(keys) == len(set(keys))
    for row in rows():
        if row.used_in:
            assert row.purpose, row.entry
            assert row.judged_line, row.entry
            assert row.visible_at, row.entry
        else:
            assert row.notes.startswith(f"{NOT_USED}: "), row.entry
            assert len(row.notes) > len(NOT_USED) + 12, f"{row.entry}: a reason, not a label"
        assert row.last_health is None
        assert row.last_checked_at is None


def test_the_playbook_is_listed_as_conditional_not_claimed() -> None:
    playbook = [r for r in rows() if r.surface is ToolkitSurface.GETAGENT_PLAYBOOK]
    assert len(playbook) == 1
    assert not playbook[0].used_in
    assert "llm_bounded" in playbook[0].notes


# ------------------------------------------------------------------------------------------------
# The probe's measurements
# ------------------------------------------------------------------------------------------------


def probe_row(
    entry: str, surface: ToolkitSurface, *, health: SourceHealth, used: bool
) -> ToolkitUse:
    return ToolkitUse(
        surface=surface,
        entry=entry,
        purpose="probe says" if used else "measured, not used",
        judged_line="",
        used_in=("sources.toolkit",) if used else (),
        last_health=health,
        last_checked_at=T0,
        visible_at="toolkit coverage matrix",
        notes="1 call(s): ok 1" if used else "fundamentals; measured ok (1 rows)",
    )


def test_without_a_probe_the_matrix_is_the_declared_rows() -> None:
    assert coverage_matrix(None) == declared_uses()


def test_the_probe_supplies_health_and_the_declaration_keeps_its_meaning() -> None:
    probe = ToolkitProbe(
        at=T0,
        rows=(
            probe_row(
                "sentiment_index.current",
                ToolkitSurface.SIGNAL_MCP,
                health=SourceHealth.OK,
                used=True,
            ),
            probe_row(
                "do_query:equity_profile",
                ToolkitSurface.DATA_MCP,
                health=SourceHealth.OK,
                used=False,
            ),
            probe_row(
                "do_query:etf_info", ToolkitSurface.DATA_MCP, health=SourceHealth.ERROR, used=False
            ),
        ),
    )
    matrix = coverage_matrix(probe)
    index = {(r.surface, r.entry): r for r in matrix}
    fear_greed = index[(ToolkitSurface.SIGNAL_MCP, "sentiment_index.current")]
    assert fear_greed.last_health is SourceHealth.OK
    assert fear_greed.last_checked_at == T0
    assert fear_greed.purpose != "probe says", "the declared purpose wins"
    assert "Skill: sentiment-analyst" in fear_greed.notes
    assert "ok 1" in fear_greed.notes
    for entry in ("do_query:equity_profile", "do_query:etf_info"):
        row = index[(ToolkitSurface.DATA_MCP, entry)]
        assert not row.used_in
        assert row.notes.startswith(NOT_USED)
    assert (ToolkitSurface.DATA_MCP, CATALOG_PLACEHOLDER) not in index, "measured rows replace it"
    assert len(matrix) == len(declared_uses()) - 1 + 2


def test_counts_per_surface() -> None:
    counts = coverage_counts(declared_uses())
    assert counts["agent_hub_bgc"]["used"] == len(used(ToolkitSurface.AGENT_HUB_BGC))
    assert counts["agent_hub_bgc"]["not_used"] >= 14
    assert all(c["used_ok"] == 0 for c in counts.values()), "nothing measured yet"


# ------------------------------------------------------------------------------------------------
# Health observed in the ledger
# ------------------------------------------------------------------------------------------------


def test_a_simulated_ledger_says_agent_hub_was_not_called(site: Site) -> None:
    matrix = ledger_health(declared_uses(), site.world.projection())
    for row in matrix:
        if row.surface is ToolkitSurface.AGENT_HUB_BGC and row.used_in:
            assert "SIMULATED" in row.notes
            assert row.last_health is None
    index = {(r.surface, r.entry): r for r in matrix}
    fear_greed = index[(ToolkitSurface.SIGNAL_MCP, "sentiment_index.current")]
    assert fear_greed.last_health is SourceHealth.OK
    market = index[(ToolkitSurface.DATA_MCP, "do_query:sentiment_market_fear_greed")]
    assert market.last_health is SourceHealth.ERROR, "the snapshot's failed call is shown as failed"


def test_public_api_health_comes_from_the_real_snapshot_labels(tmp_path: Path) -> None:
    clock = ManualClock(T0)
    ledger = HashChainLedger(
        ledger_path(tmp_path, RunMode.SIMULATED), mode=RunMode.SIMULATED, clock=clock
    )
    base = build_snapshot()

    def call(source: str, health: SourceHealth) -> SourceCall:
        return base.source_calls[0].model_copy(update={"source": source, "health": health})

    calls = (
        call("public_v3.tickers[demo]", SourceHealth.OK),
        call("tickers.demo:NVDAUSDT", SourceHealth.OK),
        call("public_v3.tickers[live]", SourceHealth.TIMEOUT),
        call("public_v3.history_fund_rate[live]:BTCUSDT", SourceHealth.OK),
    )
    ledger.append(
        EventKind.SNAPSHOT, SnapshotEvent(snapshot=base.model_copy(update={"source_calls": calls}))
    )
    matrix = ledger_health(declared_uses(), Projection.from_ledger(ledger, POLICY_V1))
    index = {r.entry: r for r in matrix if r.surface is ToolkitSurface.PUBLIC_MARKET_API}
    assert (
        index["GET /api/v3/market/tickers (Demo, header paptrading: 1)"].last_health
        is SourceHealth.OK
    )
    live = index["GET /api/v3/market/tickers (live)"]
    assert live.last_health is SourceHealth.TIMEOUT, "a failure is never shown as green"
    assert index["GET /api/v3/market/history-fund-rate (live)"].last_health is SourceHealth.OK
    assert index["GET /api/v3/market/instruments (live)"].last_health is None


def passed_proof() -> EnvironmentProof:
    return EnvironmentProof(
        checked_at=T0,
        mode=RunMode.PAPER,
        credentials_file=".secrets/demo.env",
        key_declared_demo=True,
        bgc_package="@bitget-ai/bitget-agent-cli@3.0.0",
        paptrading_header_confirmed=True,
        demo_read_ok=True,
        demo_read_code=None,
        live_read_rejected=True,
        live_read_code="40006",
        hold_mode="one_way_mode",
        account=AccountSnapshot(
            at=T0, equity_usdt=Decimal("10000"), available_usdt=Decimal("10000"), blob=None
        ),
        passed=True,
        reasons=(),
    )


def test_a_paper_ledger_shows_agent_hub_health_from_its_own_events(tmp_path: Path) -> None:
    clock = ManualClock(T0)
    ledger = HashChainLedger(ledger_path(tmp_path, RunMode.PAPER), mode=RunMode.PAPER, clock=clock)
    ledger.append(EventKind.ENVIRONMENT_PROOF, passed_proof())
    intent = make_intent()
    preview = SimulatedVenue(market=FakeMarket(), clock=clock, starting_equity=Decimal(1)).preview(
        intent
    )
    clock.advance(timedelta(minutes=1))
    ledger.append(EventKind.ORDER_PREVIEW, preview)
    ledger.append(
        EventKind.ORDER_ACK,
        VenueAck(
            client_oid=intent.client_oid, venue_order_id="1234", acked_at=clock.now(), blob=None
        ),
    )
    ledger.append(
        EventKind.FILL,
        Fill(
            exec_id="e1",
            venue_order_id="1234",
            client_oid=intent.client_oid,
            symbol=intent.symbol,
            side=Side.BUY,
            exec_price=Decimal("222.8"),
            exec_qty=Decimal("1"),
            exec_value=Decimal("222.8"),
            fee_paid=Decimal("0.13"),
            fee_coin="USDT",
            trade_scope="taker",
            trade_side="open",
            exec_pnl=None,
            executed_at=clock.now(),
            venue=FillVenue.BITGET_DEMO,
        ),
    )
    matrix = ledger_health(declared_uses(), Projection.from_ledger(ledger, POLICY_V1))
    index = {r.entry: r for r in matrix if r.surface is ToolkitSurface.AGENT_HUB_BGC}
    assert index[BGC_ENTRY_PREVIEW].last_health is SourceHealth.OK
    assert index[BGC_ENTRY_PLACE].last_health is SourceHealth.OK
    assert "1 acknowledged" in index[BGC_ENTRY_PLACE].notes
    assert index[BGC_ENTRY_ACCOUNT].last_health is SourceHealth.OK
    live = index[BGC_ENTRY_ASSETS_LIVE]
    assert live.last_health is SourceHealth.OK
    assert "40006" in live.notes
    assert index["flag --paper-trading"].last_health is SourceHealth.OK
    assert index["order --action history --paper-trading"].last_health is None, "not exercised"


def test_the_matrix_is_published_whole(site: Site) -> None:
    doc = json.loads((site.public / "toolkit.json").read_text("utf-8"))
    published = {(r["surface"], r["entry"]) for r in doc["rows"]}
    for row in coverage_matrix(None):
        if row.entry == CATALOG_PLACEHOLDER:
            continue
        assert (row.surface.value, row.entry) in published
    assert set(doc["counts"]) == {s.value for s in ToolkitSurface}


@pytest.mark.parametrize("surface", list(ToolkitSurface))
def test_each_surface_is_on_the_page(site: Site, surface: ToolkitSurface) -> None:
    page = (site.public / "index.html").read_text("utf-8")
    for row in declared_uses():
        if row.surface is surface and row.entry != CATALOG_PLACEHOLDER:
            assert html.escape(row.entry, quote=True) in page, row.entry
