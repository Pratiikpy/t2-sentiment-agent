"""M17: the GetAgent Playbook package is generated from POLICY_V1 and cannot drift from it.

* Every constant the package carries equals the policy field it was rendered from, field by field,
  and every field of the policy's shape is covered (a new policy field fails this file until the
  generator and this test carry it).
* The package's own digest of the policy is the primary's ``content_hash`` of ``POLICY_V1``.
* The manifest carries the Move 16 contract: ``runtime_profile: llm_bounded``,
  ``backtest_support: none``, ``official_evidence_kind: paper``, a cron no faster than every 15
  minutes, and the policy's universe.
* The committed ``playbook/`` is exactly what ``scripts/build_playbook.py`` builds.
* The getagent skill's own ``validate.py`` passes on the built package (skipped when the validator
  or PyYAML, which it needs, is not installed on this machine).
"""

import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from pydantic import BaseModel

from playbook.pbuild import FIXTURES, ROOT, VALIDATOR, build_package, builder, load_module_from
from sentiment_agent.crowd import novelty
from sentiment_agent.decision import contract, grounding
from sentiment_agent.hashing import canonical_json, content_hash, sha256_hex
from sentiment_agent.kernel import planner
from sentiment_agent.perception import features
from sentiment_agent.policy import POLICY_V1
from sentiment_agent.types import (
    AssetClass,
    BreakerRule,
    DecisionRule,
    GuardId,
    Mandate,
    Policy,
    Thinking,
    TriggerRule,
    WeekendRule,
)

UPLOADABLE = re.compile(r"^(README\.md|manifest\.yaml|src/[A-Za-z0-9_]+\.py)$")
SANDBOX_IMPORTS = frozenset(
    {
        "getagent",
        "json",
        "math",
        "datetime",
        "pathlib",
        "typing",
        "dataclasses",
        "collections",
        "functools",
        "re",
        "decimal",
        "statistics",
        "itertools",
        "operator",
        "copy",
        "enum",
        "abc",
        "numbers",
        "fractions",
    }
)
"""``references/sandbox-runtime.md``, Allowed Standard Library Modules, plus ``getagent``."""


@pytest.fixture(scope="module")
def package(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_package(tmp_path_factory.mktemp("playbook"))


@pytest.fixture(scope="module")
def generated(package: Path) -> ModuleType:
    """The generated ``src/policy_v1.py``, imported on its own (it imports only ``json``)."""
    return load_module_from(package / "src" / "policy_v1.py", "policy_v1_under_test")


def _manifest(package: Path) -> dict[str, Any]:
    data: dict[str, Any] = builder().manifest_data()
    return data


# ------------------------------------------------------------------------------------------------
# Generated constants equal POLICY_V1
# ------------------------------------------------------------------------------------------------


def test_every_policy_field_is_rendered_and_equal(generated: ModuleType) -> None:
    p = POLICY_V1
    expected: dict[str, object] = {
        "POLICY_VERSION": p.version,
        "UNIVERSE": tuple(
            (u.symbol, u.asset_class.value, u.demo_live_gap_p99_bps) for u in p.universe
        ),
        "EXCLUDED": p.excluded,
        "PER_NAME_MAX": p.per_name_max,
        "GROSS_MAX": p.gross_max,
        "STOP_LOSS_PCT": p.stop_loss_pct,
        "STOP_TRIGGER": p.stop_trigger,
        "DAILY_KILL_PCT": p.daily_kill_pct,
        "MAX_REBALANCES_PER_NAME_PER_DAY": p.max_rebalances_per_name_per_day,
        "MIN_HOLD_HOURS": p.min_hold_hours,
        "FEE_BUDGET_DAILY_BPS": p.fee_budget_daily_bps,
        "FEE_BUDGET_WINDOW_BPS": p.fee_budget_window_bps,
        "TAKER_ONLY": p.taker_only,
        "MAX_OPEN_SPREAD_BPS": p.max_open_spread_bps,
        "MARK_INDEX_MAX_GAP": p.mark_index_max_gap,
        "STALE_INDEX_MIN_MOVE_BPS_3H": p.stale_index_min_move_bps_3h,
        "GROUNDING_TOLERANCE": p.grounding_tolerance,
        "NET_MAX": p.net_max,
        "CLUSTER_CAPS": tuple((c.name, tuple(c.symbols), c.cap) for c in p.cluster_caps),
        "SCORING_WINDOW": (
            None
            if p.scoring_window is None
            else (p.scoring_window.start.isoformat(), p.scoring_window.end.isoformat())
        ),
        "GUARD_RULES": {b.guard.value: b.rule for b in p.guard_bases},
        "GUARD_BASES": {b.guard.value: b.basis for b in p.guard_bases},
    }
    nested: dict[str, tuple[BaseModel, type[BaseModel], dict[str, str]]] = {
        "weekend": (
            p.weekend,
            WeekendRule,
            {
                "freeze_weekday": "WEEKEND_FREEZE_WEEKDAY",
                "freeze_hour": "WEEKEND_FREEZE_HOUR",
                "reopen_weekday": "WEEKEND_REOPEN_WEEKDAY",
                "reopen_hour": "WEEKEND_REOPEN_HOUR",
                "preflatten_minutes": "WEEKEND_PREFLATTEN_MINUTES",
                "no_open_buffer_hours": "WEEKEND_NO_OPEN_BUFFER_HOURS",
            },
        ),
        "breaker": (
            p.breaker,
            BreakerRule,
            {
                "reduce_only_drawdown": "BREAKER_REDUCE_ONLY_DRAWDOWN",
                "halt_drawdown": "BREAKER_HALT_DRAWDOWN",
                "losing_streak_reduce_only": "BREAKER_LOSING_STREAK_REDUCE_ONLY",
                "snapshot_max_age_minutes": "BREAKER_SNAPSHOT_MAX_AGE_MINUTES",
                "quote_max_age_seconds": "BREAKER_QUOTE_MAX_AGE_SECONDS",
                "losing_streak_cooloff_hours": "BREAKER_LOSING_STREAK_COOLOFF_HOURS",
            },
        ),
        "triggers": (
            p.triggers,
            TriggerRule,
            {
                "us_open_local": "TRIGGER_US_OPEN_LOCAL",
                "funding_heartbeat_hours_utc": "TRIGGER_FUNDING_HEARTBEAT_HOURS_UTC",
                "fear_greed_low": "TRIGGER_FEAR_GREED_LOW",
                "fear_greed_high": "TRIGGER_FEAR_GREED_HIGH",
                "funding_z_threshold": "TRIGGER_FUNDING_Z_THRESHOLD",
                "funding_z_lookback_settlements": "TRIGGER_FUNDING_Z_LOOKBACK_SETTLEMENTS",
                "funding_z_asset_classes": "TRIGGER_FUNDING_Z_ASSET_CLASSES",
                "oi_jump_quantile": "TRIGGER_OI_JUMP_QUANTILE",
                "oi_jump_lookback_days": "TRIGGER_OI_JUMP_LOOKBACK_DAYS",
                "coordinated_min_sources": "TRIGGER_COORDINATED_MIN_SOURCES",
                "coordinated_window_minutes": "TRIGGER_COORDINATED_WINDOW_MINUTES",
                "earnings_lookahead_hours": "TRIGGER_EARNINGS_LOOKAHEAD_HOURS",
                "cooldown_minutes": "TRIGGER_COOLDOWN_MINUTES",
                "max_event_decisions_per_day": "TRIGGER_MAX_EVENT_DECISIONS_PER_DAY",
                "funding_abs_min": "TRIGGER_FUNDING_ABS_MIN",
            },
        ),
        "decision": (
            p.decision,
            DecisionRule,
            {
                "model": "DECISION_PRIMARY_MODEL",
                "daily_token_cap": "DECISION_DAILY_TOKEN_CAP",
                "max_attempts": "DECISION_MAX_ATTEMPTS",
                "max_completion_tokens": "DECISION_MAX_COMPLETION_TOKENS",
                "call_timeout_seconds": "DECISION_CALL_TIMEOUT_SECONDS",
                "min_horizon_hours": "DECISION_MIN_HORIZON_HOURS",
                "temperature": "DECISION_TEMPERATURE",
                "outage_flatten_after": "DECISION_OUTAGE_FLATTEN_AFTER",
            },
        ),
        "mandate": (
            p.mandate,
            Mandate,
            {
                "risk_budget_gross": "MANDATE_RISK_BUDGET_GROSS",
                "per_name_max": "MANDATE_PER_NAME_MAX",
                "min_horizon_hours": "MANDATE_MIN_HORIZON_HOURS",
                "text": "MANDATE_TEXT",
            },
        ),
    }
    # The replica has no reasoning tiers or text layer: these are the only fields it does not
    # carry as a constant, and each is still inside POLICY_CANONICAL_JSON (checked below).
    not_rendered = {
        ("decision", "thinking_heartbeat"),
        ("decision", "thinking_event"),
        ("decision", "basis"),
        ("weekend", "basis"),
        ("breaker", "basis"),
        ("triggers", "basis"),
    }
    for field, (section, model, names) in nested.items():
        for sub in model.model_fields:
            if (field, sub) in not_rendered:
                continue
            assert sub in names, f"{field}.{sub} is not rendered into the package"
            rendered = getattr(generated, names[sub])
            wanted = getattr(section, sub)
            assert rendered == (tuple(wanted) if isinstance(wanted, tuple) else wanted), sub
    covered = {"universe", "excluded", "guard_bases", "metrics", "expected_envelope", "version"}
    covered |= set(nested)
    scalar = {name.lower(): name for name in expected}
    for policy_field in Policy.model_fields:
        assert policy_field in covered or policy_field in scalar, (
            f"Policy.{policy_field} is not rendered into the package"
        )
    for name, value in expected.items():
        assert getattr(generated, name) == value, name
    assert tuple(g.value for g in GuardId) == generated.GUARD_IDS
    assert (
        tuple(a.value for a in AssetClass if a.follows_us_session)
        == generated.US_SESSION_ASSET_CLASSES
    )


def test_the_package_carries_the_primary_policy_digest(generated: ModuleType) -> None:
    text: str = generated.POLICY_CANONICAL_JSON
    assert text.encode("utf-8") == canonical_json(POLICY_V1)
    assert sha256_hex(text.encode("utf-8")) == generated.POLICY_HASH == POLICY_V1.content_hash()
    assert POLICY_V1.model_dump(mode="json") == generated.POLICY
    assert Policy.model_validate(generated.POLICY) == POLICY_V1


def test_constants_mirrored_from_the_primary_modules(generated: ModuleType) -> None:
    g = generated
    assert str(planner.MEASURED_DEMO_TAKER_FEE) == g.MEASURED_TAKER_FEE
    assert (g.FEATURE_MA_BARS, g.FEATURE_ATR_BARS) == (features.MA_BARS, features.ATR_BARS)
    assert g.FEATURE_STALE_INDEX_MOVES == features.STALE_INDEX_MOVES
    assert g.FEATURE_FUNDING_SD_FLOOR == features.FUNDING_SD_FLOOR
    assert features.OI_MAX_AGE.total_seconds() == g.FEATURE_OI_MAX_AGE_MINUTES * 60
    assert features.OI_MATCH_TOLERANCE.total_seconds() == g.FEATURE_OI_MATCH_TOLERANCE_MINUTES * 60
    assert (g.FEATURE_FEAR_UPPER, g.FEATURE_NEUTRAL_UPPER) == (
        features._FEAR_UPPER,
        features._NEUTRAL_UPPER,
    )
    assert g.GROUNDING_MIN_MAGNITUDE == grounding.MIN_MAGNITUDE
    assert g.GROUNDING_CONTEXT_WINDOW == grounding.CONTEXT_WINDOW
    assert g.GROUNDED_FIELDS == grounding.GROUNDED_FIELDS
    assert set(g.GROUNDING_PERCENT_TOKENS) == grounding.PERCENT_TOKENS
    assert set(g.GROUNDING_BPS_TOKENS) == grounding.BPS_TOKENS
    assert set(g.GROUNDING_FRACTION_TOKENS) == grounding.FRACTION_TOKENS
    assert g.NOVELTY_SHINGLE_WORDS == novelty.SHINGLE
    assert g.NOVELTY_DUPLICATE_AT == novelty.DUPLICATE_AT
    assert g.NOVELTY_MIN_COORDINATION_WORDS == novelty.MIN_COORDINATION_WORDS
    assert set(g.ALIASES) == set(novelty.ALIASES) == set(POLICY_V1.symbols)
    for symbol, aliases in novelty.ALIASES.items():
        rendered = g.ALIASES[symbol]
        for name in ("cashtags", "tickers", "upper_tickers", "names", "proper_names"):
            assert rendered[name] == tuple(getattr(aliases, name)), (symbol, name)
    assert contract.INITIAL_COMPLETION_TOKENS[Thinking.LOW] == g.DECISION_INITIAL_COMPLETION_TOKENS
    assert g.DECISION_TRUNCATION_GROWTH == contract.TRUNCATION_GROWTH


def test_instrument_limits_are_the_probed_live_rows_and_match_the_authoring_gate(
    generated: ModuleType,
) -> None:
    probe = json.loads(
        (ROOT / "validation" / "demo_venue" / "universe_probe.json").read_text("utf-8")
    )
    gate = json.loads((FIXTURES / "contracts_v2_usdt_futures.json").read_text("utf-8"))
    rows = {row["symbol"]: row for row in gate["rows"]}
    assert set(generated.INSTRUMENT_LIMITS) == set(POLICY_V1.symbols)
    for symbol, (min_qty, step, min_amount) in generated.INSTRUMENT_LIMITS.items():
        live = probe["instruments"][symbol]["live"]
        assert (min_qty, min_amount) == (live["minQty"], live["minAmt"])
        assert float(step) == 10.0 ** -int(live["qtyPrec"])
        row = rows[symbol]
        assert float(row["minTradeNum"]) == float(min_qty), symbol
        assert float(row["sizeMultiplier"]) == float(step), symbol
        assert float(row["minTradeUSDT"]) == float(min_amount), symbol
        assert int(row["minLever"]) <= generated.LEVERAGE <= int(row["maxLever"]), symbol


def test_every_manifest_symbol_passed_the_authoring_tradability_gate(package: Path) -> None:
    """The getagent skill requires each symbol to be resolved against Bitget's public contract
    config before authoring; the probe is recorded in tests/fixtures/playbook."""
    gate = json.loads((FIXTURES / "contracts_v2_usdt_futures.json").read_text("utf-8"))
    assert gate["code"] == "00000"
    rows = {row["symbol"]: row for row in gate["rows"]}
    for symbol in _manifest(package)["trading_symbols"]:
        assert rows[symbol]["symbolStatus"] == "normal", symbol
        assert rows[symbol]["symbolType"] == "perpetual", symbol
        assert rows[symbol]["quoteCoin"] == "USDT", symbol


# ------------------------------------------------------------------------------------------------
# The manifest contract (win plan Move 16)
# ------------------------------------------------------------------------------------------------


def test_manifest_carries_the_move_16_contract(package: Path) -> None:
    m = _manifest(package)
    assert m["runtime_profile"] == "llm_bounded"
    assert m["backtest_support"] == "none"
    assert m["official_evidence_kind"] == "paper"
    assert m["output_kind"] == "trade_strategy"
    assert m["market_type"] == "contract"
    assert m["decision_mode"] == "llm_assisted"
    # A live-only package may not default to follow-trade; subscribers opt in (validate.py).
    assert m["execution_mode"] == "signal_only"
    assert m["follow_trade_supported"] is False, "follow-trade routing has no Demo proof"
    cron = m["schedule"]["cron"]
    minute, *rest = cron.split()
    assert len(rest) == 4
    step = re.fullmatch(r"\*/(\d+)", minute)
    assert step is not None
    assert int(step.group(1)) >= 15
    assert ZoneInfo(m["schedule"]["tz"]) is not None
    assert m["trading_symbols"] == list(POLICY_V1.symbols)
    assert m["strategy_config"]["trading_symbols"] == m["trading_symbols"]
    budget = m["strategy_config"]["margin_budget"]
    assert re.fullmatch(m["user_config_schema"]["margin_budget"]["pattern"], budget)
    schema = m["user_config_schema"]
    assert set(schema) == set(m["strategy_config"])
    assert schema["trading_symbols"]["options"] == list(POLICY_V1.symbols)
    for field in ("display_name_i18n", "description_i18n"):
        assert set(m[field]) == {"en", "zh", "zh-tw", "es", "ja", "vi"}
        assert all(isinstance(v, str) and v.strip() for v in m[field].values())


def test_manifest_yaml_is_the_manifest_data(package: Path) -> None:
    text = (package / "manifest.yaml").read_text("utf-8")
    assert 'runtime_profile: "llm_bounded"' in text
    assert 'backtest_support: "none"' in text
    assert 'official_evidence_kind: "paper"' in text
    assert 'cron: "*/15 * * * *"' in text
    assert POLICY_V1.content_hash() in text
    yaml = pytest.importorskip("yaml")
    assert yaml.safe_load(text) == _manifest(package)


def test_long_description_meets_the_validator_rules_on_its_own(package: Path) -> None:
    text: str = _manifest(package)["long_description"]
    words = len(text.split())
    assert 300 <= words <= 400, words
    assert not re.search(r"\d", text), "no number may appear in long_description"


def test_generated_policy_module_is_ascii_and_self_contained(package: Path) -> None:
    source = (package / "src" / "policy_v1.py").read_text("utf-8")
    assert source.isascii()
    tree = ast.parse(source)
    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert [ast.unparse(n) for n in imports] == ["import json"]


# ------------------------------------------------------------------------------------------------
# The committed package, and the upload contract
# ------------------------------------------------------------------------------------------------


def test_committed_playbook_is_exactly_a_fresh_build() -> None:
    assert builder().check(ROOT / "playbook") == []


def test_package_holds_only_uploadable_paths(package: Path) -> None:
    files = sorted(p.relative_to(package).as_posix() for p in package.rglob("*") if p.is_file())
    assert files == sorted(builder().PACKAGE_FILES)
    assert all(UPLOADABLE.fullmatch(f) for f in files), files


def test_sandbox_modules_respect_the_import_and_call_contract(package: Path) -> None:
    """The sandbox allowlist (no hashlib, unicodedata, os, sys, time, zoneinfo), no
    ``from __future__``, and none of the calls the upload validator rejects (``compile``,
    ``eval``, ``exec``, ``__import__``), checked here independently of the validator."""
    for path in sorted((package / "src").glob("*.py")):
        tree = ast.parse(path.read_text("utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] in SANDBOX_IMPORTS, (path.name, alias.name)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                assert node.module is not None
                assert node.module.split(".")[0] in SANDBOX_IMPORTS, (path.name, node.module)
            elif isinstance(node, ast.Call):
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                assert name not in ("compile", "eval", "exec", "__import__", "import_module"), (
                    path.name,
                    node.lineno,
                )
            elif isinstance(node, ast.keyword):
                assert node.arg != "provider", (path.name, "provider= is refused by the validator")


def test_trade_mutations_live_only_in_the_execution_module(package: Path) -> None:
    mutations = {
        "close_position",
        "open_long_market",
        "open_short_market",
        "modify_stop_loss",
        "place_order",
        "cancel_order",
    }
    for path in sorted((package / "src").glob("*.py")):
        tree = ast.parse(path.read_text("utf-8"))
        called = {
            n.func.attr
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        if path.name != "execution.py":
            assert not (called & mutations), (path.name, called & mutations)


def test_getagent_validator_passes(package: Path) -> None:
    if not VALIDATOR.is_file():
        pytest.skip(f"the getagent validator is not installed at {VALIDATOR}")
    if importlib.util.find_spec("yaml") is None:
        pytest.skip("PyYAML is not installed; validate.py cannot parse a manifest without it")
    env = {k: v for k, v in os.environ.items() if not k.startswith("BITGET_")}
    env["PYTHONUTF8"] = "1"
    result = subprocess.run(  # noqa: S603 - fixed argv: this interpreter, the validator, a path
        [sys.executable, str(VALIDATOR), str(package)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Validation PASSED" in result.stdout
    assert "WARN" not in result.stdout, result.stdout


# ------------------------------------------------------------------------------------------------
# The builder itself
# ------------------------------------------------------------------------------------------------


def test_build_is_deterministic(tmp_path: Path, package: Path) -> None:
    again = build_package(tmp_path)
    for rel in builder().PACKAGE_FILES:
        assert (again / rel).read_bytes() == (package / rel).read_bytes(), rel


def test_build_refuses_a_stray_file_and_clears_caches(tmp_path: Path) -> None:
    out = build_package(tmp_path)
    (out / "src" / "__pycache__").mkdir()
    (out / "src" / "__pycache__" / "kernel.cpython-311.pyc").write_bytes(b"cache")
    builder().build(out)
    assert not (out / "src" / "__pycache__").exists()
    (out / "notes.txt").write_text("personal notes", "utf-8")
    with pytest.raises(builder().BuildError, match=r"notes\.txt"):
        builder().build(out)


def test_check_reports_drift(tmp_path: Path) -> None:
    out = build_package(tmp_path)
    assert builder().check(out) == []
    policy = out / "src" / "policy_v1.py"
    policy.write_text(
        policy.read_text("utf-8").replace("PER_NAME_MAX = 0.05", "PER_NAME_MAX = 0.5"), "utf-8"
    )
    (out / "src" / "extra.py").write_text("x = 1\n", "utf-8")
    problems = builder().check(out)
    assert "differs from a fresh build: src/policy_v1.py" in problems
    assert "not part of the package: src/extra.py" in problems


def test_cli_build_prints_the_genesis_declaration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "cli"
    assert builder().main(["--out", str(out)]) == 0
    declared = json.loads(capsys.readouterr().out)
    files = {rel: sha256_hex((out / rel).read_bytes()) for rel in builder().PACKAGE_FILES}
    assert declared["files"] == files
    assert declared["package_sha256"] == content_hash(dict(sorted(files.items())))
    assert declared["policy_hash"] == POLICY_V1.content_hash()
    assert declared["runtime_profile"] == "llm_bounded"
    assert declared["official_evidence_kind"] == "paper"
    assert "Agent Hub" in declared["logs"]["primary"]
    assert "secondary" not in declared["logs"]["primary"]
    assert "Playbook" in declared["logs"]["secondary"]
    assert builder().main(["--out", str(out), "--check"]) == 0
    assert "fresh build" in capsys.readouterr().out
