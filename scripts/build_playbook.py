"""Build the GetAgent Playbook replica of this agent from ``POLICY_V1``, so the two cannot drift.

    python scripts/build_playbook.py --out playbook/            # write it, print its digest
    python scripts/build_playbook.py --out playbook/ --check    # exit 1 unless a fresh build
    python scripts/build_playbook.py --hash-signals signal.json # hash a Playbook's records

The package follows the getagent skill's contract (``references/package-schema.md``,
``sandbox-runtime.md``): ``README.md``, ``manifest.yaml`` and ``src/**`` only, with
``runtime_profile: llm_bounded``, ``backtest_support: none``, ``official_evidence_kind: paper``
and a ``*/15`` cron (the platform's floor). Three files are generated here, and only these:

* ``src/policy_v1.py``: every number the replica's kernel, breaker, triggers and prompt obey,
  rendered from ``sentiment_agent.policy.POLICY_V1``; the constants the primary's perception,
  grounding, crowd and decision modules define (the feature windows, the grounding tokens, the
  novelty thresholds, the symbol aliases, the completion caps, the measured taker fee), imported
  from those modules; the canonical JSON of the whole policy and its SHA-256, which is the digest
  the primary's genesis pre-registers. Written ASCII-only.
* ``manifest.yaml``: the package contract, the universe from the policy, the subscriber's tunables.
* ``README.md``: the plain-language explanation the platform requires, with the policy's numbers.

The sandbox modules under ``playbook/src/`` are written by hand against ``policy_v1`` and copied
verbatim when building elsewhere. ``tests/playbook/`` checks that the committed package is exactly
what this script builds, that the generated constants equal ``POLICY_V1``, and that the replica's
kernel, grounding, contract and features agree with the primary's on the same inputs.

**Hashing outside the sandbox.** The sandbox has no ``hashlib``. The replica emits the canonical
JSON of each intent and decision; ``--hash-signals`` computes their SHA-256 here and confirms that
each record's canonical form is the primary's own (``sentiment_agent.hashing.canonical_json``), so
the digests are the ones the primary would compute for the same record.

**Evidence roles.** The Agent Hub Demo ledger is the primary log and this Playbook's log is the
secondary one; every record the package emits says so, and :func:`declaration` is the statement the
genesis carries, with the package's digest.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import shutil
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # a script run from a checkout, without an install
    sys.path.insert(0, str(ROOT / "src"))

from sentiment_agent.crowd import novelty  # noqa: E402
from sentiment_agent.decision import contract, grounding  # noqa: E402
from sentiment_agent.hashing import canonical_json, content_hash, sha256_hex  # noqa: E402
from sentiment_agent.kernel import planner  # noqa: E402
from sentiment_agent.perception import features  # noqa: E402
from sentiment_agent.policy import POLICY_V1  # noqa: E402
from sentiment_agent.types import AssetClass, GuardId, Policy, Thinking  # noqa: E402

PACKAGE_NAME: Final = "t2-sentiment-agent-replica"
DISPLAY_NAME: Final = "T2 Sentiment Agent (Replica)"
MANIFEST_VERSION: Final = "1.0.0"
"""The validator requires a ``version``; the server ignores it and assigns the published one."""

CRON_MINUTES: Final = 15
"""The platform's floor (``package-schema.md``: ``schedule.cron`` no more often than every 15
minutes). Every run makes the protective checks; the model is called only when a trigger is
admitted, as in the primary."""
SCHEDULE_CRON: Final = f"*/{CRON_MINUTES} * * * *"
SCHEDULE_TZ: Final = "UTC"
"""Every time in the policy is UTC; the timezone only sets the instance default."""
LEVERAGE: Final = 1
"""Exposure is set by weights of the margin budget (5% per name, 25% gross), so the replica opens
at one-times leverage: margin equals notional and never exceeds a quarter of the budget."""
MARGIN_BUDGET_DEFAULT: Final = "1000"
"""The default margin budget, USDT. At 5% per name a full-size position is 50 USDT, ten times the
venue's 5 USDT minimum order (``universe_probe.json`` ``minAmt``)."""

EVIDENCE_ROLE: Final = "secondary"
PRIMARY_LOG: Final = (
    "t2-sentiment-agent: Bitget UTA Demo orders through Agent Hub (bgc --paper-trading), "
    "hash-chained ledger published as public/ledger.jsonl, pre-registered at genesis"
)

PLAYBOOK_DIR: Final = ROOT / "playbook"
STATIC_MODULES: Final[tuple[str, ...]] = (
    "__init__.py",
    "book.py",
    "breaker.py",
    "canonical.py",
    "crowd.py",
    "cycle.py",
    "decision.py",
    "execution.py",
    "grounding.py",
    "kernel.py",
    "main.py",
    "mentions.py",
    "perception.py",
    "planner.py",
    "prompt.py",
    "sessions.py",
    "triggers.py",
)
GENERATED: Final[tuple[str, ...]] = ("README.md", "manifest.yaml", "src/policy_v1.py")
PACKAGE_FILES: Final[tuple[str, ...]] = tuple(
    sorted((*GENERATED, *(f"src/{m}" for m in STATIC_MODULES)))
)

UNIVERSE_PROBE: Final = ROOT / "validation" / "demo_venue" / "universe_probe.json"
LOCALES: Final[tuple[str, ...]] = ("en", "zh", "zh-tw", "es", "ja", "vi")


GENERATED_NOTE: Final = (
    "GENERATED by scripts/build_playbook.py from sentiment_agent.policy.POLICY_V1. Do not edit."
)


class BuildError(RuntimeError):
    """The package cannot be built as asked."""


# ================================================================================================
# src/policy_v1.py
# ================================================================================================


def _literal(value: object) -> str:
    """A Python literal for ``value``, ASCII-only, deterministic (dict keys sorted)."""
    if isinstance(value, dict):
        items = ", ".join(f"{_literal(k)}: {_literal(value[k])}" for k in sorted(value))
        return "{" + items + "}"
    if isinstance(value, tuple):
        inner = ", ".join(_literal(v) for v in value)
        return "(" + inner + ("," if len(value) == 1 else "") + ")"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return ascii(value)
    raise BuildError(f"no literal form for {type(value).__name__}")


LINE_WIDTH: Final = 96


def _chunked_string(text: str, indent: int = 0, width: int = 80) -> str:
    """A long string as adjacent literals inside parentheses: the same value, readable lines."""
    chunks = [text[i : i + width] for i in range(0, len(text), width)] or [""]
    pad = " " * (indent + 4)
    body = "\n".join(f"{pad}{chunk!a}" for chunk in chunks)
    return f"(\n{body}\n{' ' * indent})"


def _render(value: object, indent: int = 0) -> str:
    """:func:`_literal`, broken over lines when it would not fit in :data:`LINE_WIDTH`."""
    flat = _literal(value)
    if indent + len(flat) <= LINE_WIDTH:
        return flat
    if isinstance(value, str):
        return _chunked_string(value, indent)
    pad = " " * (indent + 4)
    if isinstance(value, dict):
        rows = [f"{pad}{_literal(k)}: {_render(value[k], indent + 4)}," for k in sorted(value)]
        return "{\n" + "\n".join(rows) + "\n" + " " * indent + "}"
    if isinstance(value, tuple):
        rows = [f"{pad}{_render(v, indent + 4)}," for v in value]
        return "(\n" + "\n".join(rows) + "\n" + " " * indent + ")"
    return flat


def instrument_limits() -> dict[str, tuple[str, str, str]]:
    """Per universe symbol: (minimum quantity, quantity step, minimum notional), from the live rows
    of the primary's ``universe_probe.json`` (``minQty``, ``10 ** -qtyPrec``, ``minAmt``)."""
    probe = json.loads(UNIVERSE_PROBE.read_text(encoding="utf-8"))
    rows = probe["instruments"]
    out: dict[str, tuple[str, str, str]] = {}
    for symbol in POLICY_V1.symbols:
        live = rows[symbol]["live"]
        precision = int(live["qtyPrec"])
        step = str(Decimal(1).scaleb(-precision))
        out[symbol] = (str(live["minQty"]), step, str(live["minAmt"]))
    return out


def policy_constants(policy: Policy = POLICY_V1) -> dict[str, object]:
    """Every generated name and its value, in one place. The test reads this back."""
    w, b, t, d, m = policy.weekend, policy.breaker, policy.triggers, policy.decision, policy.mandate
    aliases = {
        symbol: {
            "cashtags": tuple(a.cashtags),
            "tickers": tuple(a.tickers),
            "upper_tickers": tuple(a.upper_tickers),
            "names": tuple(a.names),
            "proper_names": tuple(a.proper_names),
        }
        for symbol, a in sorted(novelty.ALIASES.items())
    }
    minutes = 60.0
    return {
        "PACKAGE_NAME": PACKAGE_NAME,
        "POLICY_VERSION": policy.version,
        "POLICY_HASH": policy.content_hash(),
        "UNIVERSE": tuple(
            (u.symbol, u.asset_class.value, u.demo_live_gap_p99_bps) for u in policy.universe
        ),
        "US_SESSION_ASSET_CLASSES": tuple(a.value for a in AssetClass if a.follows_us_session),
        "EXCLUDED": tuple(policy.excluded),
        "PER_NAME_MAX": policy.per_name_max,
        "GROSS_MAX": policy.gross_max,
        "STOP_LOSS_PCT": policy.stop_loss_pct,
        "STOP_TRIGGER": policy.stop_trigger,
        "DAILY_KILL_PCT": policy.daily_kill_pct,
        "MAX_REBALANCES_PER_NAME_PER_DAY": policy.max_rebalances_per_name_per_day,
        "MIN_HOLD_HOURS": policy.min_hold_hours,
        "FEE_BUDGET_DAILY_BPS": policy.fee_budget_daily_bps,
        "FEE_BUDGET_WINDOW_BPS": policy.fee_budget_window_bps,
        "TAKER_ONLY": policy.taker_only,
        "MAX_OPEN_SPREAD_BPS": policy.max_open_spread_bps,
        "MARK_INDEX_MAX_GAP": policy.mark_index_max_gap,
        "STALE_INDEX_MIN_MOVE_BPS_3H": policy.stale_index_min_move_bps_3h,
        "GROUNDING_TOLERANCE": policy.grounding_tolerance,
        # Policy v2's run-2 amendments (run2-a2..a6). Unset in v1, so the replica carries the
        # unset values: 1.0 net (no cap), no clusters, no window, no cool-off, flatten at once.
        "NET_MAX": policy.net_max,
        "CLUSTER_CAPS": tuple((c.name, tuple(c.symbols), c.cap) for c in policy.cluster_caps),
        "SCORING_WINDOW": (
            None
            if policy.scoring_window is None
            else (policy.scoring_window.start.isoformat(), policy.scoring_window.end.isoformat())
        ),
        "WEEKEND_FREEZE_WEEKDAY": w.freeze_weekday,
        "WEEKEND_FREEZE_HOUR": w.freeze_hour,
        "WEEKEND_REOPEN_WEEKDAY": w.reopen_weekday,
        "WEEKEND_REOPEN_HOUR": w.reopen_hour,
        "WEEKEND_PREFLATTEN_MINUTES": w.preflatten_minutes,
        "WEEKEND_NO_OPEN_BUFFER_HOURS": w.no_open_buffer_hours,
        "BREAKER_REDUCE_ONLY_DRAWDOWN": b.reduce_only_drawdown,
        "BREAKER_HALT_DRAWDOWN": b.halt_drawdown,
        "BREAKER_LOSING_STREAK_REDUCE_ONLY": b.losing_streak_reduce_only,
        "BREAKER_SNAPSHOT_MAX_AGE_MINUTES": b.snapshot_max_age_minutes,
        "BREAKER_QUOTE_MAX_AGE_SECONDS": b.quote_max_age_seconds,
        "BREAKER_LOSING_STREAK_COOLOFF_HOURS": b.losing_streak_cooloff_hours,
        "TRIGGER_US_OPEN_LOCAL": t.us_open_local,
        "TRIGGER_FUNDING_HEARTBEAT_HOURS_UTC": tuple(t.funding_heartbeat_hours_utc),
        "TRIGGER_FEAR_GREED_LOW": t.fear_greed_low,
        "TRIGGER_FEAR_GREED_HIGH": t.fear_greed_high,
        "TRIGGER_FUNDING_Z_THRESHOLD": t.funding_z_threshold,
        "TRIGGER_FUNDING_Z_LOOKBACK_SETTLEMENTS": t.funding_z_lookback_settlements,
        "TRIGGER_FUNDING_Z_ASSET_CLASSES": tuple(a.value for a in t.funding_z_asset_classes),
        "TRIGGER_OI_JUMP_QUANTILE": t.oi_jump_quantile,
        "TRIGGER_OI_JUMP_LOOKBACK_DAYS": t.oi_jump_lookback_days,
        "TRIGGER_COORDINATED_MIN_SOURCES": t.coordinated_min_sources,
        "TRIGGER_COORDINATED_WINDOW_MINUTES": t.coordinated_window_minutes,
        "TRIGGER_EARNINGS_LOOKAHEAD_HOURS": t.earnings_lookahead_hours,
        "TRIGGER_COOLDOWN_MINUTES": t.cooldown_minutes,
        "TRIGGER_MAX_EVENT_DECISIONS_PER_DAY": t.max_event_decisions_per_day,
        "TRIGGER_FUNDING_ABS_MIN": t.funding_abs_min,
        "DECISION_PRIMARY_MODEL": d.model,
        "DECISION_MAX_ATTEMPTS": d.max_attempts,
        "DECISION_MAX_COMPLETION_TOKENS": d.max_completion_tokens,
        "DECISION_CALL_TIMEOUT_SECONDS": d.call_timeout_seconds,
        "DECISION_MIN_HORIZON_HOURS": d.min_horizon_hours,
        "DECISION_TEMPERATURE": d.temperature,
        "DECISION_DAILY_TOKEN_CAP": d.daily_token_cap,
        "DECISION_OUTAGE_FLATTEN_AFTER": d.outage_flatten_after,
        "MANDATE_RISK_BUDGET_GROSS": m.risk_budget_gross,
        "MANDATE_PER_NAME_MAX": m.per_name_max,
        "MANDATE_MIN_HORIZON_HOURS": m.min_horizon_hours,
        "MANDATE_TEXT": m.text,
        "GUARD_IDS": tuple(g.value for g in GuardId),
        "GUARD_RULES": {b_.guard.value: b_.rule for b_ in policy.guard_bases},
        "GUARD_BASES": {b_.guard.value: b_.basis for b_ in policy.guard_bases},
        # Mirrored from the primary's modules.
        "MEASURED_TAKER_FEE": str(planner.MEASURED_DEMO_TAKER_FEE),
        "FEATURE_MA_BARS": features.MA_BARS,
        "FEATURE_ATR_BARS": features.ATR_BARS,
        "FEATURE_STALE_INDEX_MOVES": features.STALE_INDEX_MOVES,
        "FEATURE_FUNDING_SD_FLOOR": features.FUNDING_SD_FLOOR,
        "FEATURE_OI_MAX_AGE_MINUTES": features.OI_MAX_AGE.total_seconds() / minutes,
        "FEATURE_OI_MATCH_TOLERANCE_MINUTES": features.OI_MATCH_TOLERANCE.total_seconds() / minutes,
        "FEATURE_FEAR_UPPER": features._FEAR_UPPER,
        "FEATURE_NEUTRAL_UPPER": features._NEUTRAL_UPPER,
        "GROUNDING_MIN_MAGNITUDE": grounding.MIN_MAGNITUDE,
        "GROUNDING_CONTEXT_WINDOW": grounding.CONTEXT_WINDOW,
        "GROUNDED_FIELDS": tuple(grounding.GROUNDED_FIELDS),
        "GROUNDING_PERCENT_TOKENS": tuple(sorted(grounding.PERCENT_TOKENS)),
        "GROUNDING_BPS_TOKENS": tuple(sorted(grounding.BPS_TOKENS)),
        "GROUNDING_FRACTION_TOKENS": tuple(sorted(grounding.FRACTION_TOKENS)),
        "NOVELTY_SHINGLE_WORDS": novelty.SHINGLE,
        "NOVELTY_DUPLICATE_AT": novelty.DUPLICATE_AT,
        "NOVELTY_MIN_COORDINATION_WORDS": novelty.MIN_COORDINATION_WORDS,
        "ALIASES": aliases,
        "DECISION_INITIAL_COMPLETION_TOKENS": contract.INITIAL_COMPLETION_TOKENS[Thinking.LOW],
        "DECISION_TRUNCATION_GROWTH": contract.TRUNCATION_GROWTH,
        # This package's own constants.
        "CRON_MINUTES": CRON_MINUTES,
        "SCHEDULE_CRON": SCHEDULE_CRON,
        "SCHEDULE_TZ": SCHEDULE_TZ,
        "LEVERAGE": LEVERAGE,
        "INSTRUMENT_LIMITS": instrument_limits(),
        "EVIDENCE_ROLE": EVIDENCE_ROLE,
        "PRIMARY_LOG": PRIMARY_LOG,
    }


_SECTIONS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("Identity", ("PACKAGE_NAME", "POLICY_VERSION", "POLICY_HASH")),
    ("Universe", ("UNIVERSE", "US_SESSION_ASSET_CLASSES", "EXCLUDED")),
    (
        "Kernel (G1-G11)",
        (
            "PER_NAME_MAX",
            "GROSS_MAX",
            "STOP_LOSS_PCT",
            "STOP_TRIGGER",
            "DAILY_KILL_PCT",
            "MAX_REBALANCES_PER_NAME_PER_DAY",
            "MIN_HOLD_HOURS",
            "FEE_BUDGET_DAILY_BPS",
            "FEE_BUDGET_WINDOW_BPS",
            "TAKER_ONLY",
            "MAX_OPEN_SPREAD_BPS",
            "MARK_INDEX_MAX_GAP",
            "STALE_INDEX_MIN_MOVE_BPS_3H",
            "GROUNDING_TOLERANCE",
            "NET_MAX",
            "CLUSTER_CAPS",
            "SCORING_WINDOW",
            "GUARD_IDS",
        ),
    ),
    (
        "Weekend freeze (G2)",
        (
            "WEEKEND_FREEZE_WEEKDAY",
            "WEEKEND_FREEZE_HOUR",
            "WEEKEND_REOPEN_WEEKDAY",
            "WEEKEND_REOPEN_HOUR",
            "WEEKEND_PREFLATTEN_MINUTES",
            "WEEKEND_NO_OPEN_BUFFER_HOURS",
        ),
    ),
    (
        "Circuit breaker (G10)",
        (
            "BREAKER_REDUCE_ONLY_DRAWDOWN",
            "BREAKER_HALT_DRAWDOWN",
            "BREAKER_LOSING_STREAK_REDUCE_ONLY",
            "BREAKER_SNAPSHOT_MAX_AGE_MINUTES",
            "BREAKER_QUOTE_MAX_AGE_SECONDS",
            "BREAKER_LOSING_STREAK_COOLOFF_HOURS",
        ),
    ),
    (
        "Triggers",
        (
            "TRIGGER_US_OPEN_LOCAL",
            "TRIGGER_FUNDING_HEARTBEAT_HOURS_UTC",
            "TRIGGER_FEAR_GREED_LOW",
            "TRIGGER_FEAR_GREED_HIGH",
            "TRIGGER_FUNDING_Z_THRESHOLD",
            "TRIGGER_FUNDING_Z_LOOKBACK_SETTLEMENTS",
            "TRIGGER_FUNDING_Z_ASSET_CLASSES",
            "TRIGGER_OI_JUMP_QUANTILE",
            "TRIGGER_OI_JUMP_LOOKBACK_DAYS",
            "TRIGGER_COORDINATED_MIN_SOURCES",
            "TRIGGER_COORDINATED_WINDOW_MINUTES",
            "TRIGGER_EARNINGS_LOOKAHEAD_HOURS",
            "TRIGGER_COOLDOWN_MINUTES",
            "TRIGGER_MAX_EVENT_DECISIONS_PER_DAY",
            "TRIGGER_FUNDING_ABS_MIN",
        ),
    ),
    (
        "Decision and mandate",
        (
            "DECISION_PRIMARY_MODEL",
            "DECISION_MAX_ATTEMPTS",
            "DECISION_MAX_COMPLETION_TOKENS",
            "DECISION_CALL_TIMEOUT_SECONDS",
            "DECISION_MIN_HORIZON_HOURS",
            "DECISION_TEMPERATURE",
            "DECISION_DAILY_TOKEN_CAP",
            "DECISION_OUTAGE_FLATTEN_AFTER",
            "MANDATE_RISK_BUDGET_GROSS",
            "MANDATE_PER_NAME_MAX",
            "MANDATE_MIN_HORIZON_HOURS",
            "MANDATE_TEXT",
        ),
    ),
    ("Measured bases, one per guard", ("GUARD_RULES", "GUARD_BASES")),
    (
        "Mirrored from the primary's modules (perception, grounding, crowd, decision, planner)",
        (
            "MEASURED_TAKER_FEE",
            "FEATURE_MA_BARS",
            "FEATURE_ATR_BARS",
            "FEATURE_STALE_INDEX_MOVES",
            "FEATURE_FUNDING_SD_FLOOR",
            "FEATURE_OI_MAX_AGE_MINUTES",
            "FEATURE_OI_MATCH_TOLERANCE_MINUTES",
            "FEATURE_FEAR_UPPER",
            "FEATURE_NEUTRAL_UPPER",
            "GROUNDING_MIN_MAGNITUDE",
            "GROUNDING_CONTEXT_WINDOW",
            "GROUNDED_FIELDS",
            "GROUNDING_PERCENT_TOKENS",
            "GROUNDING_BPS_TOKENS",
            "GROUNDING_FRACTION_TOKENS",
            "NOVELTY_SHINGLE_WORDS",
            "NOVELTY_DUPLICATE_AT",
            "NOVELTY_MIN_COORDINATION_WORDS",
            "ALIASES",
            "DECISION_INITIAL_COMPLETION_TOKENS",
            "DECISION_TRUNCATION_GROWTH",
        ),
    ),
    (
        "This package",
        (
            "CRON_MINUTES",
            "SCHEDULE_CRON",
            "SCHEDULE_TZ",
            "LEVERAGE",
            "INSTRUMENT_LIMITS",
            "EVIDENCE_ROLE",
            "PRIMARY_LOG",
        ),
    ),
)


def render_policy_module(policy: Policy = POLICY_V1) -> str:
    constants = policy_constants(policy)
    listed = [name for _, names in _SECTIONS for name in names]
    if sorted(listed) != sorted(constants) or len(listed) != len(set(listed)):
        raise BuildError("every generated constant must appear in exactly one section")
    canonical = canonical_json(policy).decode("utf-8")
    probe = json.loads(UNIVERSE_PROBE.read_text(encoding="utf-8"))
    lines = [
        f'"""{GENERATED_NOTE}',
        "",
        "Every number the replica obeys, rendered from the primary agent's pre-registered",
        "policy, so the Agent Hub agent and this Playbook cannot drift apart. POLICY_HASH is",
        "the SHA-256 of POLICY_CANONICAL_JSON (computed outside the sandbox, which has no",
        "hashlib) and is the digest the primary's genesis record pre-registers. Rebuild with:",
        "",
        "    python scripts/build_playbook.py --out playbook/",
        "",
        f"INSTRUMENT_LIMITS: {UNIVERSE_PROBE.relative_to(ROOT).as_posix()}, live rows, probed "
        f"{probe['probed_at_utc']}.",
        '"""',
        "",
        "import json",
        "",
    ]
    for title, names in _SECTIONS:
        lines += ["", f"# --- {title} " + "-" * max(4, 94 - len(title)), ""]
        for name in names:
            lines.append(f"{name} = {_render(constants[name])}")
    lines += [
        "",
        "# --- The whole policy " + "-" * 79,
        "",
        f"POLICY_CANONICAL_JSON = {_chunked_string(canonical)}",
        "",
        "POLICY = json.loads(POLICY_CANONICAL_JSON)",
        "",
    ]
    text = "\n".join(lines)
    if not text.isascii():
        raise BuildError("the generated policy module must be ASCII")
    return text


# ================================================================================================
# manifest.yaml
# ================================================================================================

DISPLAY_NAME_I18N: Final[dict[str, str]] = {
    "en": DISPLAY_NAME,
    "zh": "T2 情绪智能体（复刻版）",
    "zh-tw": "T2 情緒智能體（複刻版）",
    "es": "T2 Agente de Sentimiento (Réplica)",
    "ja": "T2 センチメントエージェント（レプリカ）",
    "vi": "T2 Tác tử Tâm lý Thị trường (Bản sao)",
}

DESCRIPTION: Final = (
    "LLM-led market-sentiment book on Bitget perpetuals under a reduce-only risk kernel, "
    "the Playbook twin of an Agent Hub demo agent"
)

DESCRIPTION_I18N: Final[dict[str, str]] = {
    "en": DESCRIPTION,
    "zh": (
        "由大模型决策、受只减不增风控内核约束的 Bitget 永续合约市场情绪组合，"
        "是 Agent Hub 模拟盘智能体的 Playbook 孪生版"
    ),
    "zh-tw": (
        "由大模型決策、受只減不增風控內核約束的 Bitget 永續合約市場情緒組合，"
        "是 Agent Hub 模擬盤智能體的 Playbook 孿生版"
    ),
    "es": (
        "Cartera de sentimiento de mercado en perpetuos de Bitget dirigida por un LLM bajo un "
        "núcleo de riesgo que solo reduce, gemela en Playbook de un agente demo de Agent Hub"
    ),
    "ja": (
        "LLMが判断し、縮小のみ可能なリスクカーネルが制約するBitget無期限先物の市場センチメント運用。"
        "Agent Hubデモエージェントのプレイブック版ツイン"
    ),
    "vi": (
        "Danh mục tâm lý thị trường trên hợp đồng vĩnh cửu Bitget do LLM dẫn dắt, dưới lõi rủi ro "
        "chỉ được giảm, bản song sinh Playbook của tác tử demo Agent Hub"
    ),
}

LONG_DESCRIPTION: Final = """\
This Playbook is the Bitget-hosted replica of a market-sentiment trading agent. Its thesis is that
crowds overreach: when funding runs far above its norm, open interest surges, retail accounts crowd
one side and price stretches far from its recent mean, the crowd's positioning is fragile, and the
aim is to fade, hedge or cut exposure before the crowd sees the turn. A language model makes every
trading decision; a risk kernel shared with the replica's twin on Bitget's demo venue can only
shrink, refuse or close what the model asks for, never add to it. It trades Bitget perpetuals on
Bitcoin, two US equity indices and a set of US large-cap and crypto-linked stocks, long or short.

It enters only when the model is woken by a scheduled heartbeat or by a crowd event, such as a
fear-and-greed extreme, a funding or open-interest dislocation, a coordinated story or an earnings
report, and then only if the kernel lets the position through. Every opening is a market order with
a stop placed on the venue at the same time, and a written thesis, invalidation and view of what
the crowd believes stands behind every position. Declining to trade, with reasons, is a normal
outcome that is recorded like any trade.

It exits when the model asks to close or reduce, when the venue stop fires, or when the kernel
forces an exit: the daily loss limit, the drawdown circuit breaker, a model outage, a broken venue
price, or the weekend freeze that flattens US legs while their prices stop updating.

Subscribers can adjust two parameters. The margin budget sets the capital the book sizes against
and the denominator of its return; raising it scales every position proportionally without changing
the risk the kernel allows. The instrument list can be narrowed; removing instruments reduces
opportunity and concentrates risk, and instruments outside the published universe cannot be added.

Risks: over short windows the results are dominated by chance, and a book with no edge loses its
fees. Gaps through the stop, fast reversals and correlated moves across instruments can produce
losses and drawdowns larger than a single stop suggests. The model can be wrong, and it sees how
much the crowd talks, never what it says. Past behaviour is no guarantee of future results, and
live performance can underperform what the design intends.
"""


def manifest_data(policy: Policy = POLICY_V1) -> dict[str, object]:
    """The manifest as data, in the order it is written. The YAML is rendered from this."""
    symbols = list(policy.symbols)
    return {
        "name": PACKAGE_NAME,
        "display_name": DISPLAY_NAME,
        "display_name_i18n": dict(DISPLAY_NAME_I18N),
        "version": MANIFEST_VERSION,
        "description": DESCRIPTION,
        "description_i18n": dict(DESCRIPTION_I18N),
        "long_description": LONG_DESCRIPTION,
        "market_type": "contract",
        "trading_symbols": symbols,
        "tags": ["sentiment", "llm", "risk-kernel", "crowd-positioning", "contract", "paper"],
        "output_kind": "trade_strategy",
        "decision_mode": "llm_assisted",
        "backtest_support": "none",
        "runtime_profile": "llm_bounded",
        "execution_mode": "signal_only",
        "follow_trade_supported": False,
        "official_evidence_kind": "paper",
        "schedule": {"cron": SCHEDULE_CRON, "tz": SCHEDULE_TZ},
        "strategy_config": {"trading_symbols": symbols, "margin_budget": MARGIN_BUDGET_DEFAULT},
        "user_config_schema": {
            "trading_symbols": {
                "type": "array",
                "item_type": "string",
                "default": symbols,
                "options": symbols,
                "min_items": 1,
                "max_items": len(symbols),
                "label": "Instruments (a subset of the pre-registered universe)",
            },
            "margin_budget": {
                "type": "string",
                "default": MARGIN_BUDGET_DEFAULT,
                "pattern": "^[0-9]+(\\.[0-9]+)?$",
                "label": "Margin budget USDT",
            },
        },
    }


def _yaml_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    raise BuildError(f"no YAML scalar for {type(value).__name__}")


def _yaml_lines(data: Mapping[str, object], indent: int = 0) -> Iterator[str]:
    pad = " " * indent
    for key, value in data.items():
        if key == "long_description" and isinstance(value, str):
            yield f"{pad}{key}: |"
            for line in value.rstrip("\n").split("\n"):
                yield f"{pad}  {line}" if line else ""
        elif isinstance(value, Mapping):
            yield f"{pad}{key}:"
            yield from _yaml_lines(value, indent + 2)
        elif isinstance(value, list):
            yield f"{pad}{key}: [" + ", ".join(_yaml_scalar(v) for v in value) + "]"
        else:
            yield f"{pad}{key}: {_yaml_scalar(value)}"


def render_manifest(policy: Policy = POLICY_V1) -> str:
    header = [
        f"# {GENERATED_NOTE}",
        f"# policy {policy.version} sha256 {policy.content_hash()}",
    ]
    return "\n".join([*header, *_yaml_lines(manifest_data(policy))]) + "\n"


# ================================================================================================
# README.md
# ================================================================================================


def _pct(value: float) -> str:
    return f"{value * 100:g}%"


def render_readme(policy: Policy = POLICY_V1) -> str:
    w, b, t = policy.weekend, policy.breaker, policy.triggers
    days = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
    universe = ", ".join(policy.symbols)
    heartbeats = ", ".join(f"{h:02d}:00" for h in t.funding_heartbeat_hours_utc)
    preflatten = w.freeze_hour * 60 - w.preflatten_minutes
    envelope = "\n".join(
        f"* {name.replace('_', ' ')}: {value}"
        for name, value in policy.expected_envelope.items()
        if name != "source"
    )
    return f"""<!-- {GENERATED_NOTE} -->
# {DISPLAY_NAME}

The Bitget-hosted replica of **t2-sentiment-agent**, a market-sentiment agent in which a language
model makes every trading decision and a risk kernel that can only reduce stands between the
decision and the venue. Both run under one pre-registered policy, `{policy.version}`, sha256
`{policy.content_hash()}`, and every number below is generated from it.

**Evidence roles.** The primary record is the Agent Hub agent's: Bitget UTA Demo orders sent by
`bgc --paper-trading`, in a hash-chained ledger. This Playbook's paper log is the **secondary**
record, published beside it whatever it shows; every signal it emits carries the policy hash and
says which record it is.

## 策略 / Strategy

Crowds overreach. Funding far above its norm, open interest rising fast, retail accounts skewed to
one side, price stretched above its mean and a burst of coordinated stories are signs that the
crowd's positioning is fragile. The model reads them for {universe} and decides, for the whole
book at once, where to fade, hedge or cut, or to stay flat with written reasons. Every position
carries a thesis, an invalidation, a horizon of at least {policy.decision.min_horizon_hours} hours,
and a statement of what the crowd believes against what we do.

## 开仓 / Entry

The model is woken by a heartbeat ({heartbeats} UTC, and the US cash open at {t.us_open_local}
New York time on weekdays) or by an event: a Fear & Greed reading entering or leaving its extremes
(at or below {t.fear_greed_low}, at or above {t.fear_greed_high}), BTCUSDT funding beyond
{t.funding_z_threshold:g} standard deviations of its last {t.funding_z_lookback_settlements}
settlements, a one-hour open-interest change beyond its trailing {t.oi_jump_quantile * 100:g}th
percentile, a story carried by {t.coordinated_min_sources} or more sources inside
{t.coordinated_window_minutes} minutes, an earnings report within {t.earnings_lookahead_hours}
hours, or a new insider filing for a held stock. One event per kind and instrument per
{t.cooldown_minutes} minutes, at most {t.max_event_decisions_per_day} event decisions a day.

An opening goes through eleven checks. A target of one is {_pct(policy.per_name_max)} of equity;
gross exposure is capped at {_pct(policy.gross_max)}. Market orders only, not while the spread is
wider than {policy.max_open_spread_bps:g} bps. Every opening carries a venue stop
{_pct(policy.stop_loss_pct)} from its expected entry. No increase within
{policy.min_hold_hours} hours of the last one unless the model declares its invalidation fired, at
most {policy.max_rebalances_per_name_per_day} model orders per name per UTC day, and none once fees
reach {policy.fee_budget_daily_bps:g} bps of equity in a day or {policy.fee_budget_window_bps:g} bps
over the run. A thesis that states a number the model was not shown (within
{_pct(policy.grounding_tolerance)}) may not add exposure.

## 平仓 / Exit

A position closes or shrinks when the model asks, when its venue stop fires, or when the kernel
forces it: the book down {_pct(policy.daily_kill_pct)} from its 00:00 UTC equity (flat and halted
until the next UTC day), a drawdown of {_pct(b.reduce_only_drawdown)} from peak (reduce-only) or
{_pct(b.halt_drawdown)} (halted), {b.losing_streak_reduce_only} losing trades in a row
(reduce-only), a model outage (flat until the next valid decision), a mark more than
{_pct(policy.mark_index_max_gap)} from its index, and the weekend freeze: US-equity and US-index
legs are flattened from {days[w.freeze_weekday]} {preflatten // 60:02d}:{preflatten % 60:02d} UTC
and held flat until {days[w.reopen_weekday]} {w.reopen_hour:02d}:00 UTC. Reductions are never
blocked.

## 风险 / Risk

Over a few days, return, Sharpe ratio and win rate are dominated by chance; a book with no edge
loses its fees. The expectation pre-registered with the policy, for a book with no edge over 72
hours at {_pct(policy.gross_max)} gross ({policy.expected_envelope.get("source", "")}):

{envelope}

Gaps through a stop, fast reversals and correlated moves can lose more than one stop suggests. The
model can be wrong. This replica shows the model how much the crowd talks, never what it says (see
below).

## Parameters

* `margin_budget` (USDT, default {MARGIN_BUDGET_DEFAULT}): the equity the book sizes against and
  the denominator of its return. Positions scale with it; the risk limits, as fractions of it, do
  not change.
* `trading_symbols`: a subset of the pre-registered universe. Narrowing it only removes
  opportunity; nothing outside the universe can be added, and a held symbol removed from the list
  is closed at the next decision.

## How the replica differs from the primary, and why

* **The model is the runner's**, through `getagent.llm` (one managed model, fixed budgets), not
  the primary's `{policy.decision.model}`. Every record names the model that answered.
* **No third-party text reaches the model.** The primary screens and spotlights crowd text with a
  quarantine that needs `unicodedata`, which the sandbox does not allow. Rather than a weaker
  screen, the replica shows only counts, ranks and coordination flags.
* **Signal-only, never follow-trade.** The replica is its own paper venue and places no order
  anywhere. Follow-trade is refused: `getagent.trade` routes orders to a subscriber's bound
  subaccount, and nothing here can prove that subaccount is paper, so the manifest does not offer
  it, a follow-trade run stops before it reads or writes anything, and the trade callback raises.
  The primary's orders go to Bitget's Demo environment only.
* **Venue integrity (G1)** compares the data layer's mark with its index against the same
  per-instrument limits; the stale-venue check reads the data layer's hourly closes, not the Demo
  index.
* **Partial reductions** are a close followed by a re-open of the remainder with a fresh stop,
  because the plan is written for the trade SDK, which documents no reduce-only partial order. The
  remainder pays the taker fee twice; the book counts it as one trade.
* **Persistence** is `.state/`. If the runner does not keep it between runs, event triggers are
  refused (their cooldown and daily cap cannot be enforced) and only heartbeats wake the model.
* **Hashes** are taken outside the sandbox, which has no `hashlib`: each signal carries the
  canonical JSON of its intents, and `python scripts/build_playbook.py --hash-signals signal.json`
  in the primary repository digests them.

## NOT VERIFIED

* That this account's deployment enables `llm_bounded`; the Playbook is built for it and does
  nothing useful without it.
* Which model the runner assigns, and its call, prompt-size, output and timeout budgets.
* Whether GetAgent paper trading fills on Bitget's Demo venue or on live prices, and whether it
  prices US legs over the weekend.
* Whether the runner hydrates `.state/` for scheduled runs.
* How the platform scores a run that emits several actionable signals (one per instrument whose
  target changed) followed by a `watch` summary; the runtime reference says only that the last
  actionable signal is treated as primary.

## Provenance

Generated by `scripts/build_playbook.py` in the t2-sentiment-agent repository. The kernel,
breaker, triggers, grounding and decision contract are ports of that repository's modules (MIT,
same author) and are tested against them on the same inputs.
"""


# ================================================================================================
# Building
# ================================================================================================


@dataclass(frozen=True)
class BuildReport:
    out: Path
    files: dict[str, str]
    """Package path -> sha256 of its bytes."""
    package_sha256: str
    policy_hash: str


def generated_files(policy: Policy = POLICY_V1) -> dict[str, str]:
    return {
        "README.md": render_readme(policy),
        "manifest.yaml": render_manifest(policy),
        "src/policy_v1.py": render_policy_module(policy),
    }


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _clean_caches(out: Path) -> None:
    for cache in sorted(out.rglob("__pycache__"), reverse=True):
        if cache.is_dir():
            shutil.rmtree(cache)


def _stray_files(out: Path) -> list[str]:
    expected = set(PACKAGE_FILES)
    return sorted(
        p.relative_to(out).as_posix()
        for p in out.rglob("*")
        if p.is_file() and p.relative_to(out).as_posix() not in expected
    )


def package_digest(files: Mapping[str, str]) -> str:
    """``content_hash`` of the ``{path: sha256}`` map: one digest for the whole package."""
    return content_hash(dict(sorted(files.items())))


def build(out: Path, *, source: Path = PLAYBOOK_DIR, policy: Policy = POLICY_V1) -> BuildReport:
    """Write the package into ``out``. The generated files are rendered; the sandbox modules are
    copied from ``source/src`` unless ``out`` is ``source``. Caches are removed; any other file in
    ``out`` is an error, because the upload archive is the whole directory."""
    out = out.resolve()
    source = source.resolve()
    missing = [m for m in STATIC_MODULES if not (source / "src" / m).is_file()]
    if missing:
        raise BuildError(f"sandbox modules missing from {source / 'src'}: {', '.join(missing)}")
    out.mkdir(parents=True, exist_ok=True)
    if out != source:
        for module in STATIC_MODULES:
            target = out / "src" / module
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / "src" / module, target)
    for relative, text in generated_files(policy).items():
        _write(out / relative, text)
    _clean_caches(out)
    stray = _stray_files(out)
    if stray:
        raise BuildError(
            f"{out} holds files that are not part of the package and would be uploaded: "
            + ", ".join(stray)
        )
    files = {rel: sha256_hex((out / rel).read_bytes()) for rel in PACKAGE_FILES}
    return BuildReport(
        out=out,
        files=files,
        package_sha256=package_digest(files),
        policy_hash=policy.content_hash(),
    )


def check(out: Path, *, source: Path = PLAYBOOK_DIR) -> list[str]:
    """Differences between ``out`` and a fresh build (empty when ``out`` is exactly that build)."""
    problems: list[str] = []
    out = out.resolve()
    with tempfile.TemporaryDirectory(prefix="t2sa-playbook-") as scratch:
        fresh = build(Path(scratch) / "package", source=source)
        present = sorted(
            p.relative_to(out).as_posix()
            for p in out.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        )
        for extra in sorted(set(present) - set(PACKAGE_FILES)):
            problems.append(f"not part of the package: {extra}")
        for rel in PACKAGE_FILES:
            target = out / rel
            if not target.is_file():
                problems.append(f"missing: {rel}")
            elif not filecmp.cmp(target, fresh.out / rel, shallow=False):
                problems.append(f"differs from a fresh build: {rel}")
    return problems


def declaration(report: BuildReport) -> dict[str, object]:
    """What the genesis states about this package: the two logs, their roles and the digests."""
    return {
        "package": PACKAGE_NAME,
        "package_sha256": report.package_sha256,
        "files": dict(sorted(report.files.items())),
        "policy_version": POLICY_V1.version,
        "policy_hash": report.policy_hash,
        "logs": {
            "primary": PRIMARY_LOG,
            "secondary": (
                f"GetAgent Playbook {PACKAGE_NAME}: paper-trading signals from its own paper "
                "book (follow-trade refused), each carrying the policy hash; published beside the "
                "primary whatever they show"
            ),
        },
        "runtime_profile": "llm_bounded",
        "backtest_support": "none",
        "official_evidence_kind": "paper",
        "schedule": {"cron": SCHEDULE_CRON, "tz": SCHEDULE_TZ},
    }


# ================================================================================================
# Hashing a Playbook's emitted records, outside the sandbox
# ================================================================================================


def _signals(path: Path) -> list[dict[str, object]]:
    """``signal.json`` as the runtime writes it: a JSON array, one object, or JSON lines."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        parsed: object = json.loads(text)
    except json.JSONDecodeError:
        parsed = [json.loads(line) for line in text.splitlines() if line.strip()]
    items = parsed if isinstance(parsed, list) else [parsed]
    return [item for item in items if isinstance(item, dict)]


def _canonical_records(meta: Mapping[str, object]) -> Iterator[tuple[str, str]]:
    intents = meta.get("intents")
    if isinstance(intents, list):
        for item in intents:
            if isinstance(item, Mapping) and isinstance(item.get("canonical"), str):
                yield "intent", str(item["canonical"])
    plans = meta.get("plans")
    if isinstance(plans, list):
        for plan in plans:
            nested = plan.get("intents") if isinstance(plan, Mapping) else None
            for item in nested if isinstance(nested, list) else []:
                if isinstance(item, Mapping) and isinstance(item.get("canonical"), str):
                    yield "intent", str(item["canonical"])
    decision = meta.get("decision")
    if isinstance(decision, Mapping) and isinstance(decision.get("decision_canonical"), str):
        yield "decision", str(decision["decision_canonical"])


def hash_signals(path: Path) -> dict[str, object]:
    """Every canonical record in a Playbook's signal output, with its SHA-256 and whether its text
    is exactly the primary's canonical form of the same JSON (so the digest is the primary's
    ``content_hash`` of that record)."""
    expected = POLICY_V1.content_hash()
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    policy_hashes: set[str] = set()
    signals = _signals(path)
    for signal in signals:
        meta = signal.get("meta")
        if not isinstance(meta, Mapping):
            continue
        if isinstance(meta.get("policy_hash"), str):
            policy_hashes.add(str(meta["policy_hash"]))
        for kind, text in _canonical_records(meta):
            digest = sha256_hex(text.encode("utf-8"))
            if digest in seen:
                continue
            seen.add(digest)
            parsed = json.loads(text)
            records.append(
                {
                    "kind": kind,
                    "sha256": digest,
                    "canonical_is_primary_form": canonical_json(parsed) == text.encode("utf-8"),
                    "run_id": parsed.get("run_id") if isinstance(parsed, dict) else None,
                    "seq": parsed.get("seq") if isinstance(parsed, dict) else None,
                    "symbol": parsed.get("symbol") if isinstance(parsed, dict) else None,
                }
            )
    return {
        "signals": len(signals),
        "policy_hash_expected": expected,
        "policy_hashes_seen": sorted(policy_hashes),
        "policy_hash_ok": policy_hashes <= {expected} and bool(policy_hashes),
        "records": records,
    }


# ================================================================================================
# CLI
# ================================================================================================


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the GetAgent Playbook replica from POLICY_V1 (see the module docstring)."
    )
    parser.add_argument("--out", type=Path, help="package directory to write (e.g. playbook/)")
    parser.add_argument(
        "--check", action="store_true", help="compare --out with a fresh build instead of writing"
    )
    parser.add_argument(
        "--hash-signals", type=Path, metavar="SIGNAL_JSON", help="hash a Playbook's signal output"
    )
    args = parser.parse_args(argv)
    if args.hash_signals is not None:
        print(json.dumps(hash_signals(args.hash_signals), indent=2, ensure_ascii=False))
        return 0
    if args.out is None:
        parser.error("--out is required unless --hash-signals is given")
    try:
        if args.check:
            problems = check(args.out)
            for problem in problems:
                print(problem)
            print("playbook is a fresh build" if not problems else "playbook has drifted")
            return 1 if problems else 0
        report = build(args.out)
    except BuildError as exc:
        print(f"build failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(declaration(report), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
