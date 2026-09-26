"""The static demo page: ``public/index.html`` and a page per decision card, from the export alone.

``render_site`` reads only what ``export_public`` wrote into the folder and writes HTML beside it.
Every number on a page is a number in a published JSON or CSV file next to it, so a judge who
doubts the page can open the file, and ``scripts/recompute.py`` can check the file.

What a judge sees, top to bottom (win plan Move 5, DESIGN.md §14.7):

* the run's mode, with a banner whenever the record is not the scored PAPER log;
* the headline numbers against the pre-registered no-edge envelope;
* the **event -> decision -> execution** timeline: every card, what woke the agent, what it decided,
  what the kernel did to it, what was sent and filled, each linked to its card;
* equity against every arm on one chart (the coin-flip seeds as a 5-95% band), with the table;
* the guard funnel: what the model asked for, what each guard did to it, what reached the venue;
* the governed/ungoverned twin, the live-marked mirror, the red team;
* the Bitget toolkit coverage matrix, used and not used, with health;
* the environment proof, the genesis hash and its OpenTimestamps record;
* a replay of a recorded venue-integrity refusal, labelled as a replay;
* how to verify all of it.

**Static and self-contained.** One HTML file per page with the CSS and a small theme script inlined
from ``site/templates``; no font, script, image or stylesheet is fetched, and a
``Content-Security-Policy`` meta tag forbids any request the page might otherwise make. Every link
is relative. Charts are inline SVG drawn here. Light and dark themes follow the reader's system
setting, with a toggle; the layout holds at 400 px.

**Untrusted text.** Crowd posts, news headlines and the model's own words are escaped wherever they
appear; a URL inside a post is shown as text, never as a link. The rendered pages go through the
same secret and local-path scan as the export before they replace anything already published.
"""

import html
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Final

from sentiment_agent.site.cards import CARD_ID
from sentiment_agent.site.coverage import SURFACE_ORDER
from sentiment_agent.site.export import ExportRefused, StagedWrite, prune
from sentiment_agent.types import (
    ArmKind,
    ArmResult,
    BlindTrigger,
    DecisionCard,
    FeedAlarm,
    FeedHealthReport,
    GuardRuling,
    GuardStatus,
    KernelRuling,
    RedTeamReport,
    ScreenedItem,
    SourceHealth,
    ToolkitSurface,
    ToolkitUse,
    TwinReport,
)

TEMPLATES: Final = Path(__file__).resolve().parent / "templates"

CSP: Final = (
    "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
    "img-src data:; base-uri 'none'; form-action 'none'"
)
"""No request of any kind: inline style and script only, data: images only."""

PROJECT_LINE: Final = (
    "A Market Sentiment Agent for Bitget AI Base Camp S2, Track 2. Qwen decides; a risk kernel "
    "that can only reduce gates every order; Bitget's Agent Hub places paper orders on UTA Demo."
)

SECTIONS: Final[tuple[tuple[str, str], ...]] = (
    ("overview", "Overview"),
    ("timeline", "Event → decision → execution"),
    ("equity", "Equity vs every arm"),
    ("kernel", "Guard funnel"),
    ("twin", "Governed vs ungoverned"),
    ("mirror", "Live mirror"),
    ("redteam", "Red team"),
    ("toolkit", "Bitget toolkit"),
    ("feeds", "Feed health"),
    ("proof", "Proof"),
    ("replay", "Replay"),
    ("verify", "Verify"),
)

SURFACE_TITLES: Final[Mapping[ToolkitSurface, str]] = {
    ToolkitSurface.AGENT_HUB_BGC: "Agent Hub (bgc, pinned CLI)",
    ToolkitSurface.PUBLIC_MARKET_API: "Bitget public v3 market API",
    ToolkitSurface.SIGNAL_MCP: "bitget-signal (MCP)",
    ToolkitSurface.DATA_MCP: "bitget-mcp-server",
    ToolkitSurface.GETAGENT_PLAYBOOK: "GetAgent Playbook",
    ToolkitSurface.CROWD_X: "X (not Bitget)",
    ToolkitSurface.CROWD_REDDIT: "Reddit (not Bitget)",
}

HEALTH_CLASS: Final[Mapping[SourceHealth, str]] = {
    SourceHealth.OK: "good",
    SourceHealth.EMPTY: "warn",
    SourceHealth.HOLLOW: "warn",
    SourceHealth.ERROR: "bad",
    SourceHealth.TIMEOUT: "bad",
    SourceHealth.DISABLED: "",
}

STATUS_CLASS: Final[Mapping[GuardStatus, str]] = {
    GuardStatus.PASSED: "good",
    GuardStatus.FIRED: "bad",
    GuardStatus.NOT_EVALUATED: "warn",
    GuardStatus.NOT_APPLICABLE: "",
}

LINE_CLASSES: Final = ("s-1", "s-2", "s-3", "s-4", "s-5", "s-6", "s-7")

REQUIRED: Final = (
    "summary.json",
    "decisions.json",
    "orders.json",
    "funnel.json",
    "arms.json",
    "arms_summary.json",
    "twin.json",
    "mirror.json",
    "redteam.json",
    "toolkit.json",
    "feeds.json",
    "replay.json",
    "genesis.json",
    "environment.json",
    "metrics.json",
    "cards/index.json",
)


class RenderError(RuntimeError):
    """The folder does not hold a complete export to render."""


# ================================================================================================
# Small formatting helpers
# ================================================================================================


def h(value: object) -> str:
    """Escaped text for HTML content and attribute values."""
    return html.escape(str(value), quote=True)


def _parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def when(value: str | datetime | None) -> str:
    if value is None:
        return "—"
    at = _parse_time(value) if isinstance(value, str) else value
    return at.strftime("%Y-%m-%d %H:%M UTC")


def pct(value: float | None, digits: int = 2, *, signed: bool = True) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value * 100:+.{digits}f}%" if signed else f"{value * 100:.{digits}f}%"


def number(value: float | None, digits: int = 2) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def weight(value: float | None) -> str:
    """A signed weight; an exact zero is flat."""
    if value == 0:
        return "0 (flat)"
    return pct(value, 2)


def money(value: str | Decimal | None, digits: int = 2, *, signed: bool = True) -> str:
    if value is None:
        return "—"
    try:
        amount = Decimal(str(value))
    except InvalidOperation:
        return str(value)
    return f"{amount:+,.{digits}f} USDT" if signed else f"{amount:,.{digits}f} USDT"


CI_LABELS: Final[Mapping[str, str]] = {
    "total_return": "Total return",
    "max_drawdown": "Max drawdown",
    "sharpe_ann": "Sharpe (annualised)",
    "sortino_ann": "Sortino (annualised)",
}


def short(text: str, limit: int = 220) -> str:
    clean = " ".join(text.split())
    return clean if len(clean) <= limit else clean[: limit - 1].rstrip() + "…"


def badge(text: str, css: str = "") -> str:
    cls = f"badge {css}".strip()
    return f'<span class="{h(cls)}">{h(text)}</span>'


def chip(text: str) -> str:
    return f'<span class="chip">{h(text)}</span>'


def table(
    headers: Sequence[str],
    rows: Iterable[Sequence[str]],
    *,
    numeric: Iterable[int] = (),
    wide: bool = False,
) -> str:
    """A scrollable table. Cells are pre-rendered HTML; headers are text. A ``wide`` table keeps a
    readable minimum width and scrolls inside its frame on a narrow screen instead of squeezing."""
    right = set(numeric)
    css = ' class="wide"' if wide else ""
    num = ' class="num"'
    head = "".join(
        f'<th scope="col"{num if i in right else ""}>{h(t)}</th>' for i, t in enumerate(headers)
    )
    body = []
    for row in rows:
        cells = "".join(f"<td{num if i in right else ''}>{cell}</td>" for i, cell in enumerate(row))
        body.append(f"<tr>{cells}</tr>")
    if not body:
        colspan = len(headers)
        body.append(f'<tr><td colspan="{colspan}" class="muted">none</td></tr>')
    return (
        f'<div class="table-wrap"><table{css}><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def facts(pairs: Iterable[tuple[str, str]]) -> str:
    items = "".join(f"<dt>{h(k)}</dt><dd>{v}</dd>" for k, v in pairs)
    return f'<dl class="facts">{items}</dl>'


def kpi(label: str, value: str, note: str = "") -> str:
    extra = f'<div class="note">{note}</div>' if note else ""
    return (
        f'<div class="kpi"><div class="label">{h(label)}</div>'
        f'<div class="value">{value}</div>{extra}</div>'
    )


def section(anchor: str, title: str, lede: str, body: str) -> str:
    lede_html = f'<p class="lede">{lede}</p>' if lede else ""
    return (
        f'<section class="block" id="{h(anchor)}" aria-labelledby="h-{h(anchor)}">'
        f'<h2 id="h-{h(anchor)}">{h(title)}</h2>{lede_html}{body}</section>'
    )


def json_block(value: Any) -> str:
    return f"<pre>{h(json.dumps(value, indent=1, ensure_ascii=False, sort_keys=True))}</pre>"


# ================================================================================================
# Charts (inline SVG)
# ================================================================================================


@dataclass(frozen=True, slots=True)
class Series:
    label: str
    points: tuple[tuple[datetime, float], ...]
    css: str
    ours: bool = False
    dashed: bool = False


@dataclass(frozen=True, slots=True)
class Band:
    label: str
    points: tuple[tuple[datetime, float, float], ...]


def _nice_step(span: float, target: int = 5) -> float:
    raw = span / max(target, 1)
    magnitude = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
    for factor in (1.0, 2.0, 2.5, 5.0, 10.0):
        if raw <= factor * magnitude:
            return factor * magnitude
    return 10.0 * magnitude


def line_chart(
    series: Sequence[Series],
    bands: Sequence[Band],
    *,
    label: str,
    value_format: str = "pct",
    ours_on_top: bool = True,
) -> str:
    """A responsive SVG line chart of return paths (fractions), with an optional band."""
    stamps = [t for s in series for t, _ in s.points]
    stamps.extend(t for band in bands for t, _, _ in band.points)
    values = [v for s in series for _, v in s.points]
    values.extend(v for band in bands for _, lo, hi in band.points for v in (lo, hi))
    if len(set(stamps)) < 2 or not values:
        return '<p class="small muted">Not enough hourly marks to draw yet.</p>'
    width, height = 720.0, 260.0
    left, right, top, bottom = 58.0, 14.0, 12.0, 30.0
    t0, t1 = min(stamps), max(stamps)
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        lo, hi = lo - 0.001, hi + 0.001
    pad = (hi - lo) * 0.08
    lo, hi = lo - pad, hi + pad
    span = (t1 - t0).total_seconds()

    def x(t: datetime) -> float:
        return left + (t - t0).total_seconds() / span * (width - left - right)

    def y(v: float) -> float:
        return top + (hi - v) / (hi - lo) * (height - top - bottom)

    parts = [
        f'<svg class="chart" viewBox="0 0 {width:.0f} {height:.0f}" role="img" '
        f'aria-label="{h(label)}" preserveAspectRatio="xMidYMid meet">'
    ]
    step = _nice_step(hi - lo)
    tick = math.ceil(lo / step) * step
    while tick <= hi:
        yy = y(tick)
        text = pct(tick, 2) if value_format == "pct" else number(tick, 2)
        parts.append(
            f'<line class="grid" x1="{left:.1f}" x2="{width - right:.1f}" y1="{yy:.1f}" '
            f'y2="{yy:.1f}"/><text class="axis-label" x="{left - 6:.1f}" y="{yy + 3.5:.1f}" '
            f'text-anchor="end">{h(text)}</text>'
        )
        tick += step
    count = 4
    for i in range(count + 1):
        at = t0 + (t1 - t0) * (i / count)
        xx = x(at)
        anchor = "start" if i == 0 else "end" if i == count else "middle"
        parts.append(
            f'<text class="axis-label" x="{xx:.1f}" y="{height - 8:.1f}" '
            f'text-anchor="{anchor}">{h(at.strftime("%m-%d %H:%M"))}</text>'
        )
    for band in bands:
        if len(band.points) < 2:
            continue
        upper = " ".join(f"{x(t):.1f},{y(hi_):.1f}" for t, _, hi_ in band.points)
        lower = " ".join(f"{x(t):.1f},{y(lo_):.1f}" for t, lo_, _ in reversed(band.points))
        parts.append(f'<polygon class="band" points="{upper} {lower}"/>')
    ordered = sorted(series, key=lambda s: s.ours) if ours_on_top else list(series)
    for s in ordered:
        if len(s.points) < 2:
            continue
        coords = " ".join(f"{x(t):.1f},{y(v):.1f}" for t, v in s.points)
        cls = f"line {s.css}" + (" ours" if s.ours else "") + (" dashed" if s.dashed else "")
        parts.append(
            f'<polyline class="{h(cls)}" points="{coords}"><title>{h(s.label)}</title></polyline>'
        )
    parts.append("</svg>")
    legend = "".join(
        f'<li><span class="swatch {h(s.css)}"></span>{h(s.label)}</li>' for s in series
    )
    for band in bands:
        legend += f'<li><span class="swatch band"></span>{h(band.label)}</li>'
    return f'<div class="chart-wrap">{"".join(parts)}</div><ul class="legend">{legend}</ul>'


def bars(rows: Sequence[tuple[str, float, str, str]]) -> str:
    """Horizontal bars: ``(label, value, shown value, css)``, scaled to the largest."""
    top = max((abs(v) for _, v, _, _ in rows), default=0.0) or 1.0
    out = []
    for label, value, shown, css in rows:
        width = max(0.0, min(100.0, abs(value) / top * 100))
        out.append(
            f'<div class="bar-row"><span>{h(label)}</span><div class="bar-track">'
            f'<div class="bar-fill {h(css)}" style="width:{width:.1f}%"></div></div>'
            f'<span class="num">{h(shown)}</span></div>'
        )
    return f'<div class="bars">{"".join(out)}</div>'


# ================================================================================================
# Loading the export
# ================================================================================================


@dataclass(frozen=True, slots=True)
class Export:
    summary: dict[str, Any]
    decisions: dict[str, Any]
    orders: dict[str, Any]
    funnel: dict[str, Any]
    arms: tuple[ArmResult, ...]
    arms_summary: dict[str, Any]
    twin: TwinReport | None
    mirror: dict[str, Any]
    redteam: RedTeamReport | None
    toolkit: tuple[ToolkitUse, ...]
    toolkit_counts: dict[str, Any]
    feeds: dict[str, Any]
    replay: dict[str, Any]
    genesis: dict[str, Any]
    environment: dict[str, Any]
    cards: tuple[DecisionCard, ...]


def _load(public: Path, name: str) -> Any:
    path = public / name
    if not path.is_file():
        raise RenderError(f"{name} is missing from the export; run the export first")
    return json.loads(path.read_text(encoding="utf-8"))


def load_export(public: Path) -> Export:
    """Every file the page is drawn from, parsed and, where the contract has a type, validated."""
    for name in REQUIRED:
        if not (public / name).is_file():
            raise RenderError(f"{name} is missing from the export; run the export first")
    twin_doc = _load(public, "twin.json")
    red_doc = _load(public, "redteam.json")
    toolkit_doc = _load(public, "toolkit.json")
    cards: list[DecisionCard] = []
    for entry in _load(public, "cards/index.json"):
        card_id = str(entry["card_id"])
        if not CARD_ID.fullmatch(card_id):
            raise RenderError(f"card id {card_id!r} is not a safe file name")
        cards.append(DecisionCard.model_validate(_load(public, f"cards/{card_id}.json")))
    return Export(
        summary=_load(public, "summary.json"),
        decisions=_load(public, "decisions.json"),
        orders=_load(public, "orders.json"),
        funnel=_load(public, "funnel.json"),
        arms=tuple(ArmResult.model_validate(a) for a in _load(public, "arms.json")),
        arms_summary=_load(public, "arms_summary.json"),
        twin=TwinReport.model_validate(twin_doc["report"])
        if twin_doc.get("status") == "computed"
        else None,
        mirror=_load(public, "mirror.json"),
        redteam=RedTeamReport.model_validate(red_doc["report"])
        if red_doc.get("status") == "computed"
        else None,
        toolkit=tuple(ToolkitUse.model_validate(r) for r in toolkit_doc["rows"]),
        toolkit_counts=toolkit_doc.get("counts", {}),
        feeds=_load(public, "feeds.json"),
        replay=_load(public, "replay.json"),
        genesis=_load(public, "genesis.json"),
        environment=_load(public, "environment.json"),
        cards=tuple(cards),
    )


# ================================================================================================
# Page frame
# ================================================================================================


def _asset(name: str) -> str:
    return (TEMPLATES / name).read_text(encoding="utf-8")


def page(title: str, body: str, *, nav: bool, mode: str | None) -> str:
    css = _asset("site.css")
    script = _asset("site.js")
    mode_badge = badge((mode or "empty").upper(), "accent" if mode == "paper" else "warn")
    nav_html = ""
    if nav:
        links = "".join(f'<a href="#{h(a)}">{h(t)}</a>' for a, t in SECTIONS)
        nav_html = (
            f'<nav class="sections" aria-label="Sections"><div class="wrap">{links}</div></nav>'
        )
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<meta http-equiv="Content-Security-Policy" content="{h(CSP)}">'
        '<meta name="referrer" content="no-referrer">'
        '<meta name="color-scheme" content="light dark">'
        f"<title>{h(title)}</title><style>{css}</style><script>{script}</script></head>"
        '<body><header class="top"><div class="wrap"><div class="brand">'
        f'<h1>t2-sentiment-agent</h1><p>{h(PROJECT_LINE)}</p></div><div class="meta">'
        f'{mode_badge}<button type="button" class="theme-toggle" id="theme-toggle" hidden>'
        "Theme</button></div></div></header>"
        f'{nav_html}<main><div class="wrap">{body}</div></main>'
        '<footer><div class="wrap">Every figure on this page is read from a published file in this '
        "folder and can be recomputed with <code>python scripts/recompute.py public/</code>. "
        "Static page: no request leaves it.</div></footer></body></html>\n"
    )


def mode_banner(mode: str | None, orders_sent: int = 0) -> str:
    if mode == "paper":
        if orders_sent == 0:
            return (
                '<div class="banner good"><strong>PAPER:</strong> the scored paper-trading log. '
                "No order has been sent yet: every decision so far stayed flat or was cut. An "
                "order, when one is placed, goes through Bitget Agent Hub with --paper-trading to "
                "Bitget's UTA Demo environment.</div>"
            )
        return (
            f'<div class="banner good"><strong>PAPER:</strong> {orders_sent} order(s) sent '
            "through Bitget Agent Hub with --paper-trading to Bitget's UTA Demo environment. This "
            "is the scored paper-trading log.</div>"
        )
    if mode == "simulated":
        return (
            '<div class="banner warn"><strong>SIMULATED record, not the scored paper log.</strong> '
            "Fills were simulated against keyless Bitget Demo quotes; nothing was sent to Bitget. "
            "The paper run writes its own ledger once the Demo key exists.</div>"
        )
    if mode == "dryrun":
        return (
            '<div class="banner warn"><strong>DRY RUN record, not the scored paper log.</strong> '
            "Every order was built by Bitget Agent Hub with --dry-run and none was sent.</div>"
        )
    return (
        '<div class="banner warn"><strong>Empty record:</strong> the ledger holds no events.</div>'
    )


# ================================================================================================
# The index page
# ================================================================================================


def _overview(ex: Export) -> str:
    s = ex.summary
    m = s["metrics"]
    counts = s["counts"]
    envelope = s.get("expected_envelope", {})
    se = m.get("sharpe_se_ann")
    sharpe = number(m.get("sharpe_ann"))
    if se is not None and m.get("sharpe_ann") is not None:
        sharpe += f' <span class="small">±{number(se, 1)}</span>'
    trades = m.get("n_closed_trades", 0)
    d = ex.decisions["counts"]
    tiles = [
        kpi(
            "Return",
            h(pct(m.get("total_return"), 3)),
            h(f"no-edge median {_env(envelope, 'return_on_equity_bps', ' bps')}"),
        ),
        kpi(
            "Max drawdown",
            h(pct(m.get("max_drawdown"), 3, signed=False)),
            h(f"no-edge median {_env(envelope, 'max_drawdown_pct', '%')}"),
        ),
        kpi("Sharpe (annualised)", sharpe, h(f"{m.get('n_hours', 0)} hourly returns; descriptive")),
        kpi("Win rate", h(pct(m.get("win_rate"), 1, signed=False)), h(f"{trades} closed trade(s)")),
        kpi(
            "Decisions",
            h(str(d["decisions"])),
            h(
                f"{d['act']} act, {d['hold']} hold, {d['flat_with_reasons']} flat with reasons, "
                f"{sum(n for k, n in d['outcomes'].items() if k != 'decided')} outage(s)"
            ),
        ),
        kpi(
            "Orders",
            h(f"{counts['orders_sent']} sent"),
            h(f"{counts['orders_planned']} planned, {counts['fills']} fill(s)"),
        ),
        kpi(
            "Kernel",
            h(f"{counts['kernel_changed_decisions']} cut"),
            h(f"decisions it reduced; {counts['protective_rulings']} protective ruling(s)"),
        ),
        kpi(
            "Fees paid",
            h(money(str(m.get("fees_paid", 0)), signed=False)),
            h(f"turnover {number(m.get('turnover'), 3)}x"),
        ),
    ]
    ci = m.get("ci90") or {}
    ci_rows = [
        [h(CI_LABELS.get(name, name)), h(_fmt_ci(name, bounds))]
        for name, bounds in sorted(ci.items())
    ]
    ledger = s["ledger"]
    window = s.get("scoring_window")
    scored = (
        ""
        if not window
        else '<p class="small">Return, drawdown, Sharpe, win rate and fees are scored over the '
        + h(
            f"pre-registered window {window['start'][:16].replace('T', ' ')} to "
            f"{window['end'][:16].replace('T', ' ')} UTC"
        )
        + " (in the hashed policy). Decisions, orders and the equity and trade files cover the "
        "whole record.</p>"
    )
    body = (
        mode_banner(s.get("mode"), int(counts.get("orders_sent", 0)))
        + f'<div class="kpis" style="margin-top:14px">{"".join(tiles)}</div>'
        + scored
        + "<h3>90% block-bootstrap intervals (descriptive, not inferential)</h3>"
        + table(["metric", "interval"], ci_rows)
        + '<p class="small">Three days of hourly marks cannot separate skill from luck: pure '
        "noise gives an annualised Sharpe standard error of about 11. The envelope is the "
        "pre-registered result of a no-edge book on the same venue "
        "(validation/demo_venue/envelope_clean.json, hashed into the genesis).</p>"
        + facts(
            [
                ("Ledger", h(f"{ledger['events']} events, mode {s.get('mode')}")),
                ("Head hash", f'<span class="hash">{h(ledger["head_hash"])}</span>'),
                ("Exported", h(when(s["generated_at"]))),
                ("Policy", h(s.get("policy_version", ""))),
            ]
        )
    )
    return section(
        "overview",
        "Overview",
        "The paper book's numbers, computed from the hourly marks and closed trades in the ledger, "
        "beside what a book with no edge produced on the same venue.",
        body,
    )


def _env(envelope: Mapping[str, str], key: str, unit: str) -> str:
    text = envelope.get(key, "")
    match = re.search(r"median ([+-]?[0-9.]+)", text)
    return f"{match.group(1)}{unit}" if match else "n/a"


def _fmt_ci(name: str, bounds: Sequence[float]) -> str:
    lo, hi = bounds[0], bounds[1]
    if name in ("total_return", "max_drawdown"):
        return f"{pct(lo, 3)} to {pct(hi, 3)}"
    return f"{number(lo)} to {number(hi)}"


def _trigger_chips(card: DecisionCard) -> str:
    if card.decision_id is None and card.kernel is not None:
        reason = card.kernel.protective_reason.value if card.kernel.protective_reason else "?"
        return badge("protective", "warn") + " " + chip(reason.replace("_", " "))
    if not card.triggers:
        return '<span class="muted">no trigger recorded</span>'
    return "".join(
        chip(f"{t.kind.value.replace('_', ' ')}" + (f" {','.join(t.symbols)}" if t.symbols else ""))
        for t in card.triggers
    )


def _decision_cell(card: DecisionCard) -> str:
    if card.decision_id is None:
        return '<span class="muted">No model decision: the kernel acted on its own clock.</span>'
    if card.decision is None:
        outcome = card.outcome.value if card.outcome else "unknown"
        return (
            badge(f"outage: {outcome.replace('_', ' ')}", "bad")
            + " No valid decision; no deterministic rule trades in its place."
        )
    stance = card.decision.stance.value.replace("_", " ")
    css = {"act": "accent", "hold": "", "flat with reasons": "warn"}.get(stance, "")
    return badge(stance, css) + " " + h(short(card.decision.summary, 200))


def _kernel_cell(card: DecisionCard) -> str:
    ruling = card.kernel
    if ruling is None:
        return '<div class="muted">no ruling</div>'
    changed = [i for i in ruling.instruments if i.changed_by_kernel]
    fired = sorted(
        {
            g.guard.value
            for g in (*ruling.book_rulings, *(r for i in ruling.instruments for r in i.rulings))
            if g.status is GuardStatus.FIRED
        }
    )
    if not changed:
        head = (
            badge("approved as asked", "good")
            if ruling.instruments
            else badge("nothing to rule on")
        )
    else:
        head = " ".join(
            f"{h(i.symbol)} {h(weight(i.reference))} → {h(weight(i.approved_weight))} "
            f"{chip(i.binding_guard.value if i.binding_guard else '?')}"
            for i in changed
        )
    tail = f'<div class="small muted">fired: {h(", ".join(fired))}</div>' if fired else ""
    return f"<div>{head}{tail}</div>"


def _execution_cell(card: DecisionCard) -> str:
    if not card.orders:
        return '<div class="muted">no order</div>'
    rows = []
    for order in card.orders:
        venue = f" · {h(order.venue_order_id)}" if order.venue_order_id else ""
        rows.append(
            f"<div>{h(order.side.value.upper())} {h(order.qty)} {h(order.symbol)} "
            f"{badge(order.state.value, 'good' if order.state.value == 'filled' else '')}"
            f'<span class="small">{venue}</span></div>'
        )
    return "".join(rows)


def _card_pnl(card: DecisionCard) -> Decimal | None:
    values = [o.net_pnl for o in card.orders if o.net_pnl is not None]
    return sum(values, Decimal(0)) if values else None


def _step(tag: str, body: str) -> str:
    return f'<div class="step"><div class="tag">{h(tag)}</div><div class="body">{body}</div></div>'


def _timeline(ex: Export) -> str:
    if not ex.cards:
        body = '<p class="muted">No decision yet.</p>'
    else:
        items = []
        for card in ex.cards:
            pnl = _card_pnl(card)
            pnl_text = f" · realised net {h(money(pnl))}" if pnl is not None else ""
            items.append(
                f'<li><div class="when">{h(when(card.at))}</div><div class="flow">'
                + _step("Event", _trigger_chips(card))
                + _step("Decision", _decision_cell(card))
                + _step("Kernel → execution", _kernel_cell(card) + _execution_cell(card))
                + '</div><div class="open">'
                + f'<a href="cards/{h(card.card_id)}.html">Open the card</a>'
                + f'<span class="small">{pnl_text}</span></div></li>'
            )
        body = f'<ol class="timeline">{"".join(items)}</ol>'
    return section(
        "timeline",
        "Event → decision → execution",
        "Every decision, abstention, model outage and protective ruling, in order. Each card shows "
        "what woke the agent, what it saw, what Qwen decided and why, what every guard ruled, the "
        "dry-run payload, the clientOid, the venue orderId, the fills, and the ledger rows that "
        "prove each line.",
        body,
    )


def _family(arm_id: str) -> str:
    return re.sub(r"_s\d+$", "", arm_id)


def _return_path(arm: ArmResult) -> tuple[tuple[datetime, float], ...]:
    marks = arm.marks
    if not marks or marks[0].equity <= 0:
        return ()
    base = marks[0].equity
    return tuple((m.at, m.equity / base - 1.0) for m in marks)


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[int(q * (len(ordered) - 1))]


def _median(values: Sequence[float | None]) -> float | None:
    present = sorted(v for v in values if v is not None)
    if not present:
        return None
    mid = len(present) // 2
    return present[mid] if len(present) % 2 else (present[mid - 1] + present[mid]) / 2


def _family_row(family: str, members: Sequence[ArmResult]) -> list[str]:
    """One table row for a family of seeded arms: the median of each metric across seeds."""
    m = [a.metrics for a in members]
    title = f"{family.replace('_', ' ')}: median of {len(members)} seeds"
    return [
        h(title),
        h(members[0].spec.kind.value.replace("_", " ")),
        h(pct(_median([x.total_return for x in m]), 3)),
        h(number(_median([x.sharpe_ann for x in m]))),
        h(pct(_median([x.max_drawdown for x in m]), 3, signed=False)),
        h(pct(_median([x.win_rate for x in m]), 1, signed=False)),
        h(number(_median([float(x.n_closed_trades) for x in m]), 1)),
    ]


def _equity(ex: Export) -> str:
    families: dict[str, list[ArmResult]] = defaultdict(list)
    for arm in ex.arms:
        families[_family(arm.spec.arm_id)].append(arm)
    series: list[Series] = []
    bands: list[Band] = []
    palette = iter(LINE_CLASSES * 4)
    for family, members in families.items():
        if len(members) > 1:
            paths = [dict(_return_path(a)) for a in members]
            stamps = sorted(set.intersection(*(set(p) for p in paths))) if paths else []
            points = tuple(
                (
                    t,
                    _percentile([p[t] for p in paths], 0.05),
                    _percentile([p[t] for p in paths], 0.95),
                )
                for t in stamps
            )
            kind = members[0].spec.kind.value.replace("_", " ")
            bands.append(Band(f"{kind}: {family}, {len(members)} seeds, 5-95%", points))
            median = tuple((t, _percentile([p[t] for p in paths], 0.5)) for t in stamps)
            series.append(
                Series(
                    f"{family}: median of {len(members)} seeds", median, next(palette), dashed=True
                )
            )
            continue
        arm = members[0]
        ours = arm.spec.kind is ArmKind.OURS_GOVERNED and arm.spec.arm_id == "ours_governed"
        css = "s-ours" if ours else next(palette)
        dashed = arm.spec.kind in (
            ArmKind.TWIN_UNGOVERNED,
            ArmKind.MIRROR_LIVE,
            ArmKind.WEEKEND_COUNTERFACTUAL,
        )
        series.append(Series(arm.spec.title, _return_path(arm), css, ours=ours, dashed=dashed))
    chart = line_chart(
        series, bands, label="Return path of every arm, as a fraction of its starting equity"
    )
    rows = []
    shown_families: set[str] = set()
    for row in ex.arms_summary.get("arms", []):
        family = row["family"]
        members = families.get(family, [])
        if len(members) > 1:
            if family in shown_families:
                continue
            shown_families.add(family)
            rows.append(_family_row(family, members))
            continue
        met = row["metrics"]
        rows.append(
            [
                h(row["title"]) + (" " + badge("LLM", "accent") if row["uses_llm"] else ""),
                h(row["kind"].replace("_", " ")),
                h(pct(met.get("total_return"), 3)),
                h(number(met.get("sharpe_ann"))),
                h(pct(met.get("max_drawdown"), 3, signed=False)),
                h(pct(met.get("win_rate"), 1, signed=False)),
                h(str(met.get("n_closed_trades"))),
            ]
        )
    flip = ex.arms_summary.get("coin_flip")
    flip_html = ""
    if flip:
        tr = flip["total_return"]
        share = flip.get("share_below_ours_total_return")
        ranked = flip.get("ranked_arm_id", "")
        who = (
            "The governed replica (costed exactly as the seeds are)"
            if ranked == "twin_governed_replica"
            else "The governed book"
        )
        flip_html = (
            f"<p>Coin-flip null ({h(flip['seeds'])} seeds on the same timestamps): total return "
            f"5th percentile {h(pct(tr.get('p05'), 3))}, median {h(pct(tr.get('median'), 3))}, "
            f"95th {h(pct(tr.get('p95'), 3))}. {h(who)} beat "
            f"{h(pct(share, 1, signed=False) if share is not None else 'n/a')} of seeds on return. "
            "Descriptive, not inferential.</p>"
        )
    gap = ex.arms_summary.get("replica_vs_live")
    if gap:
        slip = gap.get("slippage") or {}
        flip_html += (
            "<p>Simulator check: the live Demo book returned "
            f"{h(pct(gap.get('live_total_return'), 3))} against the replica's "
            f"{h(pct(gap.get('replica_total_return'), 3))}; over {h(slip.get('n', 0))} of our "
            "fills the venue filled a median "
            f"{h(number(slip.get('median_bps')))} bps worse than the cost model assumed "
            f"(95th percentile {h(number(slip.get('p95_bps')))} bps; positive is worse). "
            "Baselines are ranked against the replica, never the live book.</p>"
        )
    body = (
        chart
        + table(
            ["arm", "kind", "return", "Sharpe", "max DD", "win rate", "trades"],
            rows,
            numeric=(2, 3, 4, 5, 6),
            wide=True,
        )
        + flip_html
        + '<p class="small">Baseline and rival arms run on the same decision timestamps and the '
        "same recorded snapshots, marked on the same Demo prices with the same costs, under the "
        "venue guards only, so the comparison isolates who decided. Full series and every arm's "
        'provenance: <a href="arms.json">arms.json</a>, '
        '<a href="arms_summary.json">arms_summary.json</a>.</p>'
    )
    return section(
        "equity",
        "Equity vs every arm",
        "The governed book (bold) against flat, BTC held at our gross, a fixed-rule crowd fade "
        "with no language model, rival sentiment agents, the ungoverned twin, the live mirror, "
        "and the coin-flip null as a band.",
        body,
    )


def _funnel(ex: Export) -> str:
    f = ex.funnel
    legs = bars(
        [
            ("legs the model proposed", f["legs_proposed"], str(f["legs_proposed"]), ""),
            (
                "... asking to add exposure",
                f["legs_adding_exposure"],
                str(f["legs_adding_exposure"]),
                "",
            ),
            ("approved in full", f["approved_in_full"], str(f["approved_in_full"]), "good"),
            ("cut by the kernel", f["cut"], str(f["cut"]), "warn"),
            ("refused by the kernel", f["refused"], str(f["refused"]), "bad"),
        ]
    )
    orders = f["orders"]
    order_bars = bars(
        [
            ("orders planned", orders["planned"], str(orders["planned"]), ""),
            ("previewed (dry run)", orders["previewed"], str(orders["previewed"]), ""),
            ("sent", orders["sent"], str(orders["sent"]), ""),
            ("acknowledged", orders["acknowledged"], str(orders["acknowledged"]), ""),
            ("filled", orders["filled"], str(orders["filled"]), "good"),
            (
                "rejected / denied / unknown",
                orders["rejected"] + orders["denied"] + orders["unknown"],
                str(orders["rejected"] + orders["denied"] + orders["unknown"]),
                "bad",
            ),
        ]
    )
    guard_rows = []
    for guard, c in f["guards"].items():
        guard_rows.append(
            [
                h(guard.replace("_", " ")),
                h(str(c.get("binding", 0))),
                h(str(c.get("fired", 0))),
                h(str(c.get("not_evaluated", 0))),
                h(str(c.get("passed", 0))),
                h(str(c.get("not_applicable", 0))),
            ]
        )
    prot = f["protective"]
    prot_rows = [[h(k.replace("_", " ")), h(str(v))] for k, v in prot["by_reason"].items()]
    body = (
        '<div class="grid-2"><div><h3>What the model asked for</h3>'
        + legs
        + f'<p class="small">Exposure asked {h(pct(f["exposure_asked"]))}, approved '
        f"{h(pct(f['exposure_approved']))} (sum of absolute weight increases).</p></div>"
        + "<div><h3>What reached the venue from model decisions</h3>"
        + order_bars
        + "</div></div><h3>Every guard, every ruling</h3>"
        + table(
            ["guard", "binding", "fired", "not evaluated", "passed", "n/a"],
            guard_rows,
            numeric=(1, 2, 3, 4, 5),
        )
        + '<p class="small">"Binding" counts the legs whose final weight that guard set. Every '
        "guard is evaluated on every ruling (none short-circuits); a missing input fails closed "
        "for any increase. The kernel can only shrink, refuse or close.</p>"
        + f"<h3>Protective rulings ({h(prot['rulings'])})</h3>"
        + table(["reason", "rulings"], prot_rows, numeric=(1,))
    )
    return section(
        "kernel",
        "Guard funnel",
        "What the risk kernel did to what Qwen asked for, guard by guard, and what reached Bitget.",
        body,
    )


def _twin(ex: Export) -> str:
    report = ex.twin
    if report is None:
        body = (
            '<p class="muted">The twin report was not computed for this export. It replays the '
            "model's drafts without the kernel on the same marks and costs (analysis/twin.py).</p>"
        )
    else:
        tiles = "".join(
            [
                kpi(
                    "Intervention rate",
                    h(pct(report.intervention_rate, 1, signed=False)),
                    h(f"{report.n_interventions} of {report.n_decisions} decisions"),
                ),
                kpi("Prevented loss", h(pct(report.prevented_loss, 3)), "equity fraction"),
                kpi("Forgone gain", h(pct(report.forgone_gain, 3)), "equity fraction"),
                kpi(
                    "Ungoverned violation rate",
                    h(pct(report.risk_violation_rate_ungoverned, 1, signed=False)),
                    "drafts that broke a guard",
                ),
                kpi(
                    "Max DD governed",
                    h(pct(report.max_drawdown_governed, 3)),
                    h(f"ungoverned {pct(report.max_drawdown_ungoverned, 3)}"),
                ),
                kpi(
                    "Human takeovers",
                    h(str(report.human_takeovers)),
                    "owner triggers and amendments",
                ),
            ]
        )
        rows = [
            [
                f'<span class="mono">{h(i.decision_id)}</span>',
                h(i.symbol),
                h(i.guard.value),
                h(weight(i.proposed_weight)),
                h(weight(i.approved_weight)),
                h(pct(i.pnl_ungoverned, 3)),
                h(pct(i.pnl_governed, 3)),
            ]
            for i in report.interventions
        ]
        body = f'<div class="kpis">{tiles}</div>' + table(
            ["decision", "symbol", "guard", "asked", "approved", "P&L ungoverned", "P&L governed"],
            rows,
            numeric=(3, 4, 5, 6),
            wide=True,
        )
    return section(
        "twin",
        "Governed vs ungoverned",
        "The model's draft before the kernel, simulated on the same Demo marks and costs: what "
        "the kernel's interventions prevented and what they cost.",
        body,
    )


def _mirror(ex: Export) -> str:
    rows = ex.mirror.get("series", [])
    demo: list[tuple[datetime, float]] = []
    live: list[tuple[datetime, float]] = []
    base: float | None = None
    for row in rows:
        book = float(row["equity_book"])
        base = base or book
        at = _parse_time(row["at"])
        demo.append((at, book / base - 1.0))
        if row.get("equity_live_mirror") is not None:
            live.append((at, float(row["equity_live_mirror"]) / base - 1.0))
    chart = line_chart(
        [
            Series("book at the Demo mark", tuple(demo), "s-ours", ours=True),
            Series("same positions at live prices", tuple(live), "s-4", dashed=True),
        ],
        (),
        label="Demo-marked book against the same positions marked at live prices",
        ours_on_top=False,
    )
    gaps = [r["gap_bps"] for r in rows if r.get("gap_bps") is not None]
    arm_rows = []
    for doc in ex.mirror.get("arms", []):
        arm = ArmResult.model_validate(doc)
        arm_rows.append(
            [
                h(arm.spec.title),
                h(pct(arm.metrics.total_return, 3)),
                h(pct(arm.metrics.max_drawdown, 3, signed=False)),
                h(str(arm.metrics.n_hours)),
            ]
        )
    gap_text = (
        f"Largest hourly gap between the two: {h(number(max(gaps, key=abs), 1))} bps."
        if gaps
        else "No open position has been marked at both prices yet."
    )
    body = (
        chart
        + f'<p class="small">{gap_text} {h(ex.mirror.get("note", ""))}</p>'
        + table(["arm", "return", "max DD", "hours"], arm_rows, numeric=(1, 2, 3))
        + '<p class="small">The weekend counterfactual holds, at live prices, exactly the '
        "equity legs the weekend-freeze guard took away while UTA Demo stops pricing them (Friday "
        "20:00 to Monday 00:00 UTC): where S2's 7×24 theme shows up honestly on this venue.</p>"
    )
    return section(
        "mirror",
        "Live mirror",
        "The same book marked at Bitget's live prices beside the Demo marks, so a reader can see "
        "the Demo result is not a sandbox artefact.",
        body,
    )


def _redteam(ex: Export) -> str:
    report = ex.redteam
    if report is None:
        body = (
            '<p class="muted">The red team has not been run for this export. It injects the '
            "HeyArka and AgentDojo attack strings and a coordinated pump into recorded snapshots "
            "and grades every arm against its clean decision (redteam/).</p>"
        )
    else:
        titles = {a.arm_id: a.title for a in report.arms}
        rate_bars = bars(
            [
                (
                    titles.get(arm, arm),
                    rate,
                    pct(rate, 1, signed=False),
                    "bad" if rate > 0 else "good",
                )
                for arm, rate in sorted(report.hijack_rate.items())
            ]
        )
        stopped: Counter[tuple[str, str]] = Counter(
            (o.arm_id, o.stopped_by) for o in report.outcomes
        )
        rows = [
            [h(titles.get(arm, arm)), h(by), h(str(n))] for (arm, by), n in sorted(stopped.items())
        ]
        sample = (
            f"on {h(len(report.snapshots_attacked))} of {h(report.snapshots_recorded)} recorded "
            f"snapshots, spread evenly over the record, "
            if report.snapshots_attacked and report.snapshots_recorded is not None
            else ""
        )
        body = (
            f"<p>{h(len(report.vectors))} attack vectors {sample}"
            f"{h(len(report.outcomes))} paired outcomes, run {h(when(report.run_at))}; "
            f"Qwen tokens spent {h(report.qwen_tokens_spent)}.</p>"
            "<h3>Hijack rate by arm</h3>"
            + rate_bars
            + "<h3>What stopped each attack</h3>"
            + table(["arm", "stopped by", "outcomes"], rows, numeric=(2,))
        )
    return section(
        "redteam",
        "Red team of the sentiment input",
        "A sentiment agent reads text that attackers write. Published whatever the grade.",
        body,
    )


def _health_badge(health: SourceHealth | None) -> str:
    if health is None:
        return badge("not measured")
    return badge(health.value, HEALTH_CLASS.get(health, ""))


def _toolkit(ex: Export) -> str:
    by_surface: dict[ToolkitSurface, list[ToolkitUse]] = defaultdict(list)
    for row in ex.toolkit:
        by_surface[row.surface].append(row)
    summary_rows = []
    blocks = []
    for surface in SURFACE_ORDER:
        rows = by_surface.get(surface, [])
        if not rows:
            continue
        used = [r for r in rows if r.used_in]
        unused = [r for r in rows if not r.used_in]
        ok = sum(1 for r in used if r.last_health is SourceHealth.OK)
        summary_rows.append(
            [
                h(SURFACE_TITLES.get(surface, surface.value)),
                h(str(len(used))),
                h(str(ok)),
                h(str(len(unused))),
            ]
        )
        used_rows = [
            [
                f'<span class="mono">{h(r.entry)}</span>',
                f'<div>{h(r.purpose)}</div><div class="small muted">{h(r.judged_line)}</div>',
                _health_badge(r.last_health)
                + (
                    f'<div class="small muted">{h(when(r.last_checked_at))}</div>'
                    if r.last_checked_at
                    else ""
                ),
                h(r.visible_at),
                f'<span class="small">{h(short(r.notes, 260))}</span>',
            ]
            for r in used
        ]
        unused_rows = [
            [
                f'<span class="mono">{h(r.entry)}</span>',
                h(short(r.notes, 320)),
                _health_badge(r.last_health),
            ]
            for r in unused
        ]
        inner = table(
            ["entry", "used for / judged line", "health", "visible at", "notes"],
            used_rows,
            wide=True,
        )
        if unused_rows:
            inner += f"<h4>Not used ({len(unused_rows)}), each with its reason</h4>" + table(
                ["entry", "reason", "measured"], unused_rows, wide=True
            )
        title = h(SURFACE_TITLES.get(surface, surface.value))
        opened = " open" if surface is ToolkitSurface.AGENT_HUB_BGC else ""
        blocks.append(
            f"<details{opened}><summary>{title}: {len(used)} used, {len(unused)} not used"
            f"</summary>{inner}</details>"
        )
    body = table(
        ["surface", "used", "used and OK now", "not used, with reason"],
        summary_rows,
        numeric=(1, 2, 3),
    ) + "".join(blocks)
    return section(
        "toolkit",
        "Bitget toolkit coverage",
        "Every Bitget surface the agent touches: what it is used for, which judged line it serves, "
        "its last health (from the live probe or the ledger), and where it shows. Surfaces that "
        "were measured and are not used are listed with the reason.",
        body,
    )


CAUSE_TEXT: Final[Mapping[str, str]] = {
    "feed_failed": "a source failed",
    "no_data": "sources answered, nothing usable",
    "cadence": "not read on light snapshots",
    "no_threshold": "no frozen threshold",
}


def _alarm_rows(alarms: Sequence[FeedAlarm]) -> list[list[str]]:
    return [
        [
            f'<span class="mono">{h(a.feed)}</span>',
            _health_badge(a.health),
            h(when(a.since)),
            h(str(a.snapshots)),
            f'<span class="small">{h(short(a.error or "", 240))}</span>',
        ]
        for a in alarms
    ]


def _blind_rows(blind: Sequence[BlindTrigger]) -> list[list[str]]:
    return [
        [
            h(b.kind.value.replace("_", " ")),
            h(", ".join(b.symbols) or "all"),
            badge(CAUSE_TEXT.get(b.cause, b.cause), "bad" if b.cause == "feed_failed" else ""),
            f'<span class="small">{h(short(b.reason, 320))}</span>',
        ]
        for b in blind
    ]


def feed_health_html(report: FeedHealthReport, *, cadence: bool = True) -> str:
    """The alarms and blind trigger kinds of one report (the page and each decision card)."""
    alarms = (
        table(
            ["source", "health", "failing since", "snapshots", "last error"],
            _alarm_rows(report.alarms),
            numeric=(3,),
            wide=True,
        )
        if report.alarms
        else '<p class="small">No source is failing.</p>'
    )
    shown = [b for b in report.blind if cadence or b.cause != "cadence"]
    blind = (
        table(["trigger kind", "instruments", "cause", "why"], _blind_rows(shown), wide=True)
        if shown
        else '<p class="small">Every trigger kind could be evaluated.</p>'
    )
    return f"<h4>Failing sources</h4>{alarms}<h4>Trigger kinds that could not fire</h4>{blind}"


def _feeds(ex: Export) -> str:
    doc = ex.feeds
    lede = (
        "Every source the agent reads that is failing now (an error, a timeout or a hollow "
        "answer), since when, and every trigger kind a snapshot could not evaluate, with the "
        "cause. A failing source degrades a snapshot and never stops the agent; here it is "
        "visible instead of silent."
    )
    if doc.get("status") != "logged" or doc.get("latest") is None:
        body = f'<p class="muted">{h(doc.get("note") or "No feed-health report yet.")}</p>'
        return section("feeds", "Feed health", lede, body)
    latest = FeedHealthReport.model_validate(doc["latest"])
    counts = doc.get("counts", {})
    head = facts(
        [
            ("As of", h(when(latest.at)) + (" (light snapshot)" if latest.light else "")),
            ("Failing now", h(str(counts.get("open", len(latest.alarms))))),
            (
                "Alarms raised / cleared",
                h(f"{counts.get('raised', 0)} / {counts.get('cleared', 0)}"),
            ),
            ("Reports logged", h(str(counts.get("reports", 0)))),
        ]
    )
    history = doc.get("history", [])
    history_html = (
        table(
            ["seq", "at", "raised", "cleared"],
            [
                [
                    h(str(row["seq"])),
                    h(when(row["at"])),
                    f'<span class="mono small">{h(", ".join(row["raised"]) or "—")}</span>',
                    f'<span class="mono small">{h(", ".join(row["cleared"]) or "—")}</span>',
                ]
                for row in reversed(history)
            ],
            wide=True,
        )
        if history
        else '<p class="small">No alarm has been raised.</p>'
    )
    body = (
        head
        + feed_health_html(latest)
        + f"<details><summary>Alarm history ({len(history)})</summary>{history_html}</details>"
    )
    return section("feeds", "Feed health", lede, body)


def _x_post_block(text: str, posted: dict[str, Any] | None) -> str:
    """The pre-registration post: a draft until the owner records it (``t2sa x-posted``), then
    the post with the time its own id carries and where that falls against the first order."""
    if not posted:
        return (
            "<h4>Drafted for X, to be posted before the first order (not yet recorded as posted)"
            f'</h4><blockquote class="text">{h(text)}</blockquote>'
        )
    first = posted.get("first_order_at")
    if first is None:
        order = "; no order has been sent yet"
    elif posted.get("before_first_order"):
        order = f", before the first order ({h(when(first))})"
    else:
        order = f", after the first order ({h(when(first))})"
    link = f'<a href="{h(posted["url"])}">{h(posted["url"])}</a>'
    return (
        f"<h4>Posted on X {h(when(posted['posted_at']))}{order}</h4>"
        f'<blockquote class="text">{h(text)}</blockquote>'
        f'<p class="small">{link}. '
        "The time is the one the post's own id encodes; the owner recorded the post in the "
        f"ledger at seq {int(posted['recorded_seq'])}.</p>"
    )


def _declared_changes(genesis: Mapping[str, Any]) -> str:
    """Run 2's declared changes against the run it follows, straight from the genesis payload."""
    changes = genesis.get("declared_changes") or []
    predecessor = genesis.get("predecessor")
    if predecessor is None:
        return ""
    rows = [
        [
            f'<span class="mono">{h(c["change_id"])}</span>',
            h(c["kind"].replace("_", " ")),
            f"<div>{h(c['title'])}</div>"
            f'<div class="small muted">{h(short(c["detail"], 600))}</div>',
            '<span class="small">'
            + h("; ".join(f"{k}: {v}" for k, v in c["evidence"].items()))
            + "</span>",
        ]
        for c in changes
    ]
    return (
        "<h4>Declared changes against the run it follows</h4>"
        + facts(
            [
                (
                    "Follows",
                    f'<span class="hash">{h(predecessor["genesis_hash"])}</span> '
                    + h(f"({predecessor['policy_version']}, {predecessor['window']})"),
                ),
                ("Its code commit", f'<span class="hash">{h(predecessor["code_commit"])}</span>'),
            ]
        )
        + table(["change", "kind", "what and why", "evidence"], rows, wide=True)
    )


def _check(ok: bool | None, label: str, detail: str = "") -> tuple[str, str]:
    mark = badge("yes", "good") if ok else badge("no", "bad") if ok is False else badge("n/a")
    return label, mark + (f' <span class="small">{h(detail)}</span>' if detail else "")


def _anchor_line(record: Mapping[str, Any]) -> str:
    status = str(record["status"])
    proof = record.get("ots_blob")
    link = f'<a href="blobs/{h(proof["sha256"])}">.ots proof</a>' if proof else "no proof stored"
    detail = h(short(str(record["detail"]), 160))
    return (
        f"<div>{badge(status, 'good' if status == 'upgraded' else 'warn')} "
        f'{h(when(record["submitted_at"]))} {link} <span class="small muted">{detail}</span></div>'
    )


def _proof(ex: Export) -> str:
    env = ex.environment
    proofs = env.get("proofs", [])
    if proofs:
        latest = proofs[-1]["proof"]
        env_html = facts(
            [
                ("Checked", h(when(latest["checked_at"]))),
                _check(
                    latest["key_declared_demo"],
                    "Key file declares Demo",
                    latest["credentials_file"],
                ),
                _check(
                    latest["paptrading_header_confirmed"],
                    "Installed SDK sends paptrading: 1",
                    latest["bgc_package"],
                ),
                _check(
                    latest["demo_read_ok"],
                    "Demo environment accepts the key",
                    f"code {latest['demo_read_code']}" if latest["demo_read_code"] else "",
                ),
                _check(
                    latest["live_read_rejected"],
                    "Live environment refuses the key",
                    f"code {latest['live_read_code']}" if latest["live_read_code"] else "",
                ),
                _check(latest["account"] is not None, "Demo account read"),
                ("Hold mode", h(latest.get("hold_mode") or "—")),
                (
                    "Verdict",
                    badge("PASSED", "good") if latest["passed"] else badge("FAILED", "bad"),
                ),
            ]
        )
        if latest["reasons"]:
            env_html += "<ul>" + "".join(f"<li>{h(r)}</li>" for r in latest["reasons"]) + "</ul>"
    else:
        env_html = (
            f'<p class="muted">{h(env.get("note") or "No environment proof in this ledger.")}</p>'
        )
    g = ex.genesis
    if g.get("present"):
        genesis = g["genesis"]
        anchors = g.get("genesis_anchors", [])
        anchor_html = "".join(_anchor_line(a["record"]) for a in anchors) or (
            '<span class="muted">not stamped yet</span>'
        )
        genesis_html = facts(
            [
                ("Genesis hash", f'<span class="hash">{h(g["hash"])}</span>'),
                ("Created", h(when(genesis["created_at"]))),
                ("Mode", h(genesis["mode"])),
                ("Policy hash", f'<span class="hash">{h(genesis["policy_hash"])}</span>'),
                ("Code commit", f'<span class="hash">{h(genesis["code_commit"])}</span>'),
                ("Agent Hub CLI", h(genesis["bgc_package"])),
                ("Model", h(genesis["qwen_model"])),
                ("OpenTimestamps", anchor_html),
            ]
        )
        genesis_html += _declared_changes(genesis)
        if g.get("x_post_text"):
            genesis_html += _x_post_block(g["x_post_text"], g.get("x_post"))
        genesis_html += (
            "<details><summary>Pre-registered statement and expected envelope</summary>"
            f"<p>{h(genesis['statement'])}</p>"
            + json_block(genesis["expected_envelope"])
            + "</details>"
        )
    else:
        genesis_html = f'<p class="muted">{h(g.get("note", "No genesis."))}</p>'
    amendments = g.get("amendments", [])
    amend_html = (
        table(
            ["seq", "at", "reason", "owner confirmed"],
            [
                [
                    h(str(a["seq"])),
                    h(when(a["ts"])),
                    h(a["amendment"]["reason"]),
                    h(str(a["amendment"]["owner_confirmed"])),
                ]
                for a in amendments
            ],
        )
        if amendments
        else '<p class="small">No amendment: the pre-registered policy has not changed.</p>'
    )
    ledger = ex.summary["ledger"]
    kinds = ", ".join(f"{k} {v}" for k, v in ledger.get("by_kind", {}).items())
    body = (
        '<div class="grid-2"><div><h3>Environment proof (Demo only)</h3>'
        + env_html
        + "</div><div><h3>Pre-registration</h3>"
        + genesis_html
        + "</div></div><h3>Amendments</h3>"
        + amend_html
        + "<h3>The ledger</h3>"
        + facts(
            [
                ("Events", h(f"{ledger['events']} ({kinds})")),
                ("Head hash", f'<span class="hash">{h(ledger["head_hash"])}</span>'),
                (
                    "First / last",
                    h(f"{when(ledger.get('first_ts'))} / {when(ledger.get('last_ts'))}"),
                ),
                (
                    "File",
                    '<a href="ledger.jsonl">ledger.jsonl</a> with its head anchor '
                    '<a href="ledger.jsonl.head">ledger.jsonl.head</a>',
                ),
            ]
        )
    )
    return section(
        "proof",
        "Proof",
        "The key is proven to be a Demo key before the first order; the policy, prompts and code "
        "are hashed into a genesis that is timestamped and posted before trading; every event is "
        "hash-chained.",
        body,
    )


def _guard_table(rulings: Sequence[GuardRuling], *, basis: bool = True) -> str:
    rows = []
    for r in rulings:
        ceiling = (
            "exit"
            if r.forces_exit
            else (
                pct(r.ceiling_abs_weight, 2, signed=False)
                if r.ceiling_abs_weight is not None
                else "—"
            )
        )
        detail = ""
        if basis:
            detail = (
                '<details class="inline"><summary>basis and inputs</summary>'
                f'<p class="small">{h(r.basis)}</p>' + json_block(r.inputs) + "</details>"
            )
        rows.append(
            [
                h(r.guard.value.replace("_", " ")),
                badge(r.status.value.replace("_", " "), STATUS_CLASS.get(r.status, "")),
                h(ceiling),
                f'<div class="small">{h(r.reason)}</div>{detail}',
            ]
        )
    return table(["guard", "status", "ceiling", "reason"], rows, wide=True)


def _request_line(request: Mapping[str, Any]) -> str:
    query = "&".join(f"{k}={v}" for k, v in request["params"].items())
    headers = ", ".join(f"{k}: {v}" for k, v in request["headers"].items())
    extra = f" [{headers}]" if headers else ""
    line = f"GET {request['path']}?{query}{extra}"
    return f'<span class="mono">{h(line)}</span>'


def _replay(ex: Export) -> str:
    r = ex.replay
    inputs = r["inputs"]
    rulings = "".join(
        f"<h4>{h(item['leg'])}</h4>" + _guard_table([GuardRuling.model_validate(item["ruling"])])
        for item in r["rulings"]
    )
    requests = table(
        ["series", "request", "fetched"],
        [[h(q["series"]), _request_line(q), h(when(q["fetched_at"]))] for q in r["requests"]],
    )
    body = (
        f'<div class="banner warn"><strong>{h(r["label"])}</strong></div>'
        + f'<p style="margin-top:12px">{h(r["what"])}</p>'
        + facts(
            [
                ("Instrument", h(f"{r['instrument']} ({r['category']})")),
                ("Hour", h(when(r["hour"]))),
                ("Demo mark (hourly extreme)", h(inputs["demo_mark_used"])),
                ("Demo index (close)", h(inputs["demo_index_close"])),
                (
                    "Mark vs index",
                    h(
                        f"{inputs['mark_index_gap_pct']:.2f}% (limit {inputs['limit_pct']:.0f}%); "
                        f"{inputs['mark_index_gap_pct_against_index_high']:.2f}% even against "
                        "the index's own high"
                    ),
                ),
                (
                    "Demo last (range in the hour)",
                    h(
                        f"{inputs['demo_last']} "
                        f"({inputs['demo_last_low']} to {inputs['demo_last_high']})"
                    ),
                ),
                ("Live last", h(inputs["live_last"])),
            ]
        )
        + "<h3>What G1 (venue integrity) rules on the recorded print</h3>"
        + rulings
        + f'<p class="small">{h(r["note"])}</p>'
        + f"<details><summary>The recorded requests ({h(r['recording'])})</summary>"
        + f"{requests}</details>"
    )
    return section(
        "replay",
        "Replay: a recorded venue-integrity refusal",
        "The venue-integrity guard exists because UTA Demo prints flash marks. This is the guard "
        "run on one it printed.",
        body,
    )


def _verify(ex: Export) -> str:
    files = [
        ("ledger.jsonl", "the hash-chained ledger, byte for byte"),
        ("verify.md", "how to check everything below"),
        ("metrics.json", "the book's metric set"),
        ("equity_hourly.csv", "one row per hourly mark"),
        ("trades.csv", "closed trades, flat to flat"),
        ("orders.json", "every order: clientOid, venue orderId, fills"),
        ("decisions.json", "every decision and abstention"),
        ("funnel.json", "the guard funnel"),
        ("arms.json", "every arm's marks, trades and metrics"),
        ("twin.json", "the twin report"),
        ("mirror.json", "the live-marked mirror"),
        ("redteam.json", "the red-team report"),
        ("toolkit.json", "the coverage matrix"),
        ("genesis.json", "the pre-registration"),
        ("environment.json", "the environment proofs"),
        ("replay.json", "the venue-integrity replay"),
        ("cards/index.json", "every decision card"),
    ]
    rows = [[f'<a href="{h(name)}">{h(name)}</a>', h(text)] for name, text in files]
    body = (
        "<ol>"
        "<li>Recompute every number with the standard library alone: "
        "<code>python scripts/recompute.py public/</code>. It re-walks the hash chain, re-hashes "
        "every blob, rebuilds every closed trade from the fills and recomputes every metric and "
        "interval.</li>"
        "<li>Check the orders exist on Bitget Demo (read-only): "
        "<code>python scripts/verify_orders.py --public public/orders.json</code>.</li>"
        "<li>Replay any decision without a key: "
        "<code>t2sa replay --public public/ --decision &lt;decision_id&gt;</code>.</li>"
        "<li>Every card lists the ledger rows and the blobs that prove it; "
        "<code>sha256(blobs/&lt;hash&gt;)</code> equals the file name.</li>"
        "</ol>" + table(["file", "what it is"], rows)
    )
    return section("verify", "Verify", "Nothing here needs to be taken on trust.", body)


def render_index(ex: Export) -> str:
    body = "".join(
        [
            _overview(ex),
            _timeline(ex),
            _equity(ex),
            _funnel(ex),
            _twin(ex),
            _mirror(ex),
            _redteam(ex),
            _toolkit(ex),
            _feeds(ex),
            _proof(ex),
            _replay(ex),
            _verify(ex),
        ]
    )
    return page(
        "t2-sentiment-agent: the public record", body, nav=True, mode=ex.summary.get("mode")
    )


# ================================================================================================
# A card page
# ================================================================================================


def _text_item(item: ScreenedItem) -> str:
    source = f"{item.item.channel} · {item.item.source} · {when(item.item.published_at)}"
    detections = "".join(chip(f"{d.severity}: {d.pattern}") for d in item.detections)
    url = f'<div class="small muted">{h(item.item.url)}</div>' if item.item.url else ""
    if item.withheld:
        body = (
            f"<div>{badge('withheld', 'bad')} The model was shown only "
            f"<code>{h(item.prompt_text)}</code>.</div>"
            '<details class="inline"><summary>What quarantine withheld (shown here as text, '
            f"never to the model)</summary><div>{h(item.item.text)}</div></details>"
        )
        css = "text withheld"
    else:
        body = f"<div>{h(item.item.text)}</div>"
        css = "text"
    return (
        f'<blockquote class="{css}"><div class="small muted">{h(source)}</div>'
        f"{body}{url}{detections}</blockquote>"
    )


def _reason(ruling: KernelRuling) -> str:
    return ruling.protective_reason.value if ruling.protective_reason else "?"


def _ruling_html(ruling: KernelRuling) -> str:
    head = facts(
        [
            ("Ruling", f'<span class="hash">{h(ruling.ruling_id)}</span>'),
            ("At", h(when(ruling.at))),
            (
                "Answers",
                h(
                    f"decision {ruling.decision_id}"
                    if ruling.decision_id
                    else f"protective: {_reason(ruling)}"
                ),
            ),
            ("Breaker", h(f"{ruling.activation_before.value} → {ruling.activation_after.value}")),
            ("Guards applied", h(", ".join(g.value for g in ruling.guards_applied))),
        ]
    )
    inst_rows = [
        [
            h(i.symbol),
            h(weight(i.current_weight)),
            h(weight(i.proposed_weight) if i.proposed_weight is not None else "hold"),
            h(weight(i.approved_weight)),
            badge(i.binding_guard.value, "warn")
            if i.binding_guard and i.changed_by_kernel
            else badge("as asked", "good"),
        ]
        for i in ruling.instruments
    ]
    parts = [
        head,
        table(["instrument", "held", "asked", "approved", "set by"], inst_rows, numeric=(1, 2, 3)),
    ]
    if ruling.book_rulings:
        parts.append("<h4>Book-level guards</h4>" + _guard_table(ruling.book_rulings))
    for inst in ruling.instruments:
        parts.append(f"<h4>{h(inst.symbol)}: every guard</h4>" + _guard_table(inst.rulings))
    return "".join(parts)


def _mono(text: str) -> str:
    return f'<span class="mono">{h(text)}</span>'


def _outcome(card: DecisionCard) -> str:
    return (card.outcome.value if card.outcome else "unknown").replace("_", " ")


def render_card(card: DecisionCard, orders: Mapping[str, Mapping[str, Any]]) -> str:
    title_kind = "Decision" if card.decision_id else "Protective ruling"
    parts: list[str] = [
        '<p><a href="../index.html#timeline">← Back to the timeline</a></p>',
        section(
            "summary",
            f"{title_kind} card",
            "",
            facts(
                [
                    ("Card", f'<span class="mono">{h(card.card_id)}</span>'),
                    ("At", h(when(card.at))),
                    ("Decision id", f'<span class="mono">{h(card.decision_id or "—")}</span>'),
                    ("Ruling id", f'<span class="hash">{h(card.ruling_id or "—")}</span>'),
                    ("Outcome", h(card.outcome.value if card.outcome else "—")),
                    ("Raw JSON", f'<a href="{h(card.card_id)}.json">{h(card.card_id)}.json</a>'),
                ]
            ),
        ),
    ]
    if card.triggers:
        rows = [
            [
                h(t.kind.value.replace("_", " ")),
                h(", ".join(t.symbols) or "—"),
                h(t.detail),
                h(number(t.observed, 3) if t.observed is not None else "—"),
                h(number(t.threshold, 3) if t.threshold is not None else "—"),
                h(t.source),
            ]
            for t in card.triggers
        ]
        parts.append(
            section(
                "triggers",
                "What woke the agent",
                "",
                table(
                    ["trigger", "symbols", "detail", "observed", "threshold", "source"],
                    rows,
                    numeric=(3, 4),
                    wide=True,
                ),
            )
        )
    if card.coverage:
        rows = [
            [f'<span class="mono">{h(s)}</span>', _health_badge(v)]
            for s, v in sorted(card.coverage.items())
        ]
        counts = Counter(v for v in card.coverage.values())
        lede = ", ".join(
            f"{v.value} {n}" for v, n in sorted(counts.items(), key=lambda kv: kv[0].value)
        )
        parts.append(
            section(
                "coverage", "Sources the snapshot asked", h(lede), table(["source", "health"], rows)
            )
        )
    if card.feed_health is not None:
        feeds = card.feed_health
        parts.append(
            section(
                "feeds",
                "Feed health on this snapshot",
                h(
                    f"{len(feeds.alarms)} source(s) failing, "
                    f"{len(feeds.failure_blind())} trigger kind(s) blinded by a failure."
                ),
                feed_health_html(feeds, cadence=False),
            )
        )
    if card.shown_text:
        withheld = sum(1 for t in card.shown_text if t.withheld)
        parts.append(
            section(
                "text",
                "Text shown to the model",
                h(
                    f"{len(card.shown_text)} item(s), {withheld} withheld by quarantine. Every "
                    "item is somebody else's words: untrusted, spotlighted, never an instruction."
                ),
                "".join(_text_item(t) for t in card.shown_text),
            )
        )
    decision = card.decision
    if decision is not None:
        targets = []
        for t in decision.targets:
            report = card.grounding.get(t.symbol)
            grounding = ""
            if report is not None:
                fig_rows = [
                    [
                        h(f.raw),
                        badge("resolved", "good") if f.resolved else badge("unresolved", "bad"),
                        f'<span class="mono">{h(f.source or "—")}</span>',
                        h(number(f.known_value, 6) if f.known_value is not None else "—"),
                        f'<span class="small">{h(short(f.context, 120))}</span>',
                    ]
                    for f in report.figures
                ]
                grounding = (
                    f"<h4>Grounding: {len(report.figures)} number(s), "
                    f"{len(report.unresolved)} unresolved</h4>"
                    + table(
                        ["figure", "status", "fact", "fact value", "context"], fig_rows, wide=True
                    )
                )
            evidence = "".join(chip(e) for e in t.evidence)
            declared = badge("invalidation declared", "warn")
            invalid = (
                f"<p>{declared} {h(t.invalidation_evidence or '')}</p>"
                if t.invalidation_triggered
                else ""
            )
            targets.append(
                f'<div class="target"><header><strong>{h(t.symbol)}</strong>'
                f"{badge(f'target {t.target:+.2f}', 'accent')}"
                f'<span class="small">confidence {h(number(t.confidence, 2))} · '
                f"horizon {h(t.horizon_hours)}h</span></header>"
                f"<p><strong>Thesis.</strong> {h(t.thesis)}</p>"
                f"<p><strong>Invalidation.</strong> {h(t.invalidation)}</p>"
                '<div class="crowd"><div><div class="tag">What the crowd believes</div>'
                f"{h(t.crowd_belief)}</div>"
                f'<div><div class="tag">What we do</div>{h(t.our_view)}</div></div>'
                f"{invalid}<div>{evidence}</div>{grounding}</div>"
            )
        alternatives = table(
            ["rejected alternative", "why"],
            [[h(a.action), h(a.reason)] for a in decision.rejected_alternatives],
        )
        flat = (
            "<h4>Reasons for staying flat</h4><ul>"
            + "".join(f"<li>{h(r)}</li>" for r in decision.flat_reasons)
            + "</ul>"
            if decision.flat_reasons
            else ""
        )
        who = card.decided_by or ""
        scripted = who.startswith("a scripted stand-in")
        parts.append(
            section(
                "decision",
                "What the scripted stand-in decided" if scripted else "What Qwen decided",
                "",
                (f'<p class="small">Decided by {h(who)}.</p>' if who else "")
                + f"<p>{badge(decision.stance.value.replace('_', ' '), 'accent')} "
                f"{h(decision.summary)}</p>"
                + f"<p><strong>Mandate.</strong> {h(decision.mandate_response)}</p>"
                + flat
                + "".join(targets)
                + "<h4>Alternatives it rejected</h4>"
                + alternatives,
            )
        )
    elif card.decision_id is not None:
        parts.append(
            section(
                "decision",
                "No valid decision",
                "",
                f"<p>{badge(_outcome(card), 'bad')} The model did not return a valid decision. "
                "No deterministic rule places an order in its place; the kernel flattens the book "
                "as a protective action.</p>",
            )
        )
    if card.kernel is not None:
        parts.append(
            section(
                "kernel",
                "What the risk kernel ruled",
                "It can only shrink, refuse or close.",
                _ruling_html(card.kernel),
            )
        )
    if card.orders or card.previews:
        order_parts = []
        previews = {p.client_oid: p for p in card.previews}
        for o in card.orders:
            extra = orders.get(o.client_oid, {})
            preview = previews.get(o.client_oid)
            fill_rows = [
                [
                    f'<span class="mono">{h(f.exec_id)}</span>',
                    h(f.side.value),
                    h(f.exec_qty),
                    h(f.exec_price),
                    h(f.fee_paid),
                    h(f.trade_side or "—"),
                    h(when(f.executed_at)),
                ]
                for f in o.fills
            ]
            preview_html = (
                "<details><summary>Dry-run preview (what Agent Hub would send)</summary>"
                + f"<pre>{h(' '.join(preview.argv))}</pre>"
                + json_block(preview.would_send)
                + "</details>"
                if preview is not None
                else ""
            )
            order_parts.append(
                '<div class="target"><header><strong>'
                f"{h(o.side.value.upper())} {h(o.qty)} {h(o.symbol)}</strong>"
                f"{badge(o.state.value, 'good' if o.state.value == 'filled' else '')}"
                f'<span class="small">{h(o.purpose.value)}</span></header>'
                + facts(
                    [
                        ("clientOid", f'<span class="mono">{h(o.client_oid)}</span>'),
                        (
                            "Venue orderId",
                            _mono(o.venue_order_id or "not acknowledged"),
                        ),
                        ("Stop", h(extra.get("stop_loss_price") or "—")),
                        ("Net P&L realised", h(money(o.net_pnl) if o.net_pnl is not None else "—")),
                    ]
                )
                + preview_html
                + table(
                    ["exec id", "side", "qty", "price", "fee", "open/close", "at"],
                    fill_rows,
                    numeric=(2, 3, 4),
                    wide=True,
                )
                + "</div>"
            )
        parts.append(
            section("orders", "What was sent and what came back", "", "".join(order_parts))
        )
    blob_rows = [
        [
            f'<a class="hash" href="../blobs/{h(b.sha256)}">{h(b.sha256)}</a>',
            h(b.media_type),
            h(b.size),
        ]
        for b in card.blobs
    ]
    parts.append(
        section(
            "proof",
            "Proof",
            h(
                f"Read from ledger rows {', '.join(str(s) for s in card.ledger_seqs)} of "
                "ledger.jsonl; every blob those rows commit to is below, named by its SHA-256."
            ),
            table(["blob", "media type", "bytes"], blob_rows, numeric=(2,)),
        )
    )
    return page(f"{title_kind} card {card.card_id}", "".join(parts), nav=False, mode=None)


# ================================================================================================
# Render
# ================================================================================================


def render_site(public_dir: Path) -> list[Path]:
    """Write ``index.html`` and ``cards/<id>.html`` into ``public_dir`` from its export.

    Pages are written to a staging folder, scanned for secrets and local paths, and moved into
    place only when clean. Card pages left by an earlier render for cards that no longer exist are
    removed. Returns the paths written.
    """
    public = Path(public_dir).resolve()
    ex = load_export(public)
    orders = {str(o["client_oid"]): o for o in ex.orders.get("orders", [])}
    stage = StagedWrite(public)
    try:
        stage.write("index.html", render_index(ex).encode("utf-8"))
        for card in ex.cards:
            stage.write(f"cards/{card.card_id}.html", render_card(card, orders).encode("utf-8"))
        findings = stage.scan(extra_paths=[public.parent])
        if findings:
            raise ExportRefused(findings)
        stage.publish()
    finally:
        stage.discard()
    prune(public, "cards", set(stage.written), (".html",))
    return [public / name for name in stage.written]


__all__ = [
    "CSP",
    "Export",
    "RenderError",
    "load_export",
    "render_card",
    "render_index",
    "render_site",
]
