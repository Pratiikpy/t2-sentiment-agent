"""``t2sa``: the command line. Every owner step in DESIGN.md §16, in order, and every check after.

::

    t2sa preflight                   keyless probes, bgc installed, dry-run previews, hashes
    t2sa setup                       npm ci the pinned Agent Hub CLI; create var/
    t2sa prove-env                   the environment proof (needs .secrets/demo.env)
    t2sa plumbing-test --owner-approved
                                     one minimum-size BTCUSDT order with its stop, then closed
    t2sa genesis [--mode paper]      the pre-registration, stamped; prints the X post text
    t2sa go-live [--commit SHA]      prove-env, genesis (once), then the paper loop: one command
    t2sa run --mode paper            the loop
    t2sa once | decide | reconcile   one tick | an owner-requested decision | one sweep
    t2sa export | verify | replay    publish | check the chain | re-run a decision keylessly
    t2sa status | amend | probe-toolkit
    t2sa rivals --approve-tokens N   rival sentiment agents on the recorded snapshots
    t2sa redteam --approve-tokens N  the sentiment input under attack

**Exit codes.** 0 success; 1 a check failed (a broken chain, a replay that differs, a proof that did
not pass); 2 a refused precondition or a usage error (a missing approval flag, no genesis for a
command that needs one, a module that is not installed); 3 the environment was refused (40099, a
live key, no Demo credentials:
:data:`~sentiment_agent.execution.environment.EXIT_ENVIRONMENT_REFUSED`); 4 the pre-registration
does not match (no genesis, or the loaded policy is not the one in force); 5 another instance of the
mode is running.

Approval-gated commands refuse without their flag and say what the flag approves:
``plumbing-test`` sends real Demo orders (``--owner-approved``); ``rivals`` and ``redteam`` spend
Qwen tokens, so each prints its estimate and runs only with ``--approve-tokens N`` at or above it.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Any, Final, Literal, TextIO

from sentiment_agent.analysis.armsim import ArmSimulator
from sentiment_agent.analysis.baselines import run_baselines
from sentiment_agent.analysis.metrics import book_arm
from sentiment_agent.analysis.mirror import live_mirror, weekend_counterfactual
from sentiment_agent.analysis.simcheck import SimulatorCheck, simulator_check
from sentiment_agent.analysis.twin import governed_replica, twin_report, ungoverned_arm
from sentiment_agent.book.projection import ORDER_EVENT_KINDS, Projection
from sentiment_agent.clock import ManualClock, VenueClock
from sentiment_agent.crowd.adapters import CompositeCrowd
from sentiment_agent.decision.agent import DecisionAgent
from sentiment_agent.decision.contract import ATTEMPT_MEDIA_TYPE, replay_calls
from sentiment_agent.decision.prompt import prompt_hashes
from sentiment_agent.execution.bgc import BGC_PACKAGE, BgcRunner, BgcTransport
from sentiment_agent.execution.environment import (
    EXIT_ENVIRONMENT_REFUSED,
    EnvironmentRefused,
    load_demo_credentials,
    prove_environment,
    vendor_contract,
)
from sentiment_agent.execution.executor import Executor
from sentiment_agent.execution.orders import OrderTracker
from sentiment_agent.execution.reconcile import Reconciler
from sentiment_agent.execution.stops import StopManager
from sentiment_agent.hashing import canonical_json, content_hash, sha256_hex
from sentiment_agent.kernel.breaker import Breaker
from sentiment_agent.kernel.kernel import RiskKernel
from sentiment_agent.kernel.planner import (
    client_oid,
    entry_price,
    fee_rate,
    intent_core,
    plan_orders,
    stop_price,
)
from sentiment_agent.ledger.anchor import stamp, subprocess_ots_runner
from sentiment_agent.ledger.blobs import FileBlobStore
from sentiment_agent.ledger.chain import (
    GenesisError,
    HashChainLedger,
    LedgerError,
    ledger_path,
    verify_file,
)
from sentiment_agent.ledger.genesis import (
    X_POST_NOTE_PREFIX,
    amend,
    build_genesis,
    recorded_x_post,
    x_post_text,
    x_post_url,
    x_posted_at,
)
from sentiment_agent.llm.budget import BudgetExhausted, DailyTokenBudget
from sentiment_agent.llm.client import (
    QwenChatModel,
    QwenError,
    QwenTimeout,
    QwenTransportError,
    load_qwen_env,
)
from sentiment_agent.llm.fakes import RecordedChatModel, completion_from_json
from sentiment_agent.perception.snapshot import SnapshotBuilder
from sentiment_agent.policy import ACTIVE_POLICY
from sentiment_agent.redteam.corpus import load_vectors
from sentiment_agent.redteam.harness import AgentArm, RedTeamHarness, WithoutQuarantine, summarise
from sentiment_agent.redteam.harness import estimate_qwen_tokens as redteam_estimate
from sentiment_agent.rivals.finbert_arm import FinbertArm, finbert_installed
from sentiment_agent.rivals.harness import estimate_qwen_tokens, run_rivals
from sentiment_agent.rivals.keyword_arm import KeywordSentimentArm
from sentiment_agent.rivals.registry import (
    FearGreedConfluenceArm,
    QwenHeadlineTraderArm,
    RivalArm,
    SentimentFusionArm,
)
from sentiment_agent.rivals.tradingagents_arm import TradingAgentsSocialArm
from sentiment_agent.run2 import declaration_for
from sentiment_agent.runtime.health import status_from_ledger, status_lines
from sentiment_agent.runtime.loop import RunLoop, request_decision
from sentiment_agent.runtime.wiring import (
    DEFAULT_SIMULATED_EQUITY,
    OI_THRESHOLDS_MEDIA_TYPE,
    RULING_CONTEXT_MEDIA_TYPE,
    App,
    InstanceLock,
    InstanceLocked,
    LlmChoice,
    Parts,
    Paths,
    Planner,
    ProjectingLedger,
    RefusedToStart,
    archive_ledger,
    build_app,
    crypto_symbols,
    default_data_service,
    default_toolkit,
    frozen_oi_thresholds,
    genesis_of,
    git_commit,
    lock_hashes,
    oi_history,
    open_chain,
    open_pregenesis,
    ruling_context_blob,
    scrub_note,
)
from sentiment_agent.site.coverage import coverage_matrix
from sentiment_agent.site.export import ExportError, ExportManifest, export_public
from sentiment_agent.site.render import render_site
from sentiment_agent.sources.toolkit import ToolkitFacade, probe_all
from sentiment_agent.types import (
    ALL_GUARDS,
    Amendment,
    ArmResult,
    BlobRef,
    BlobStore,
    BookState,
    BreakerState,
    Candle,
    ChatMessage,
    ChatModel,
    Clock,
    Completion,
    DecisionCard,
    DecisionEvent,
    DecisionRecord,
    EnvironmentProof,
    EventKind,
    Genesis,
    GuardId,
    InstrumentSpec,
    KernelInputs,
    KernelRuling,
    LedgerEvent,
    LlmCallRecord,
    LlmOutcome,
    MarketData,
    Note,
    OrderIntent,
    OrderPlan,
    OrderPurpose,
    PerceptionSnapshot,
    Policy,
    PriceSource,
    ProtectiveReason,
    RedTeamReport,
    RulingContext,
    RunMode,
    Side,
    SnapshotEvent,
    Thinking,
    ToolkitProbe,
    Trigger,
    TriggerKind,
    TwinReport,
)
from sentiment_agent.venue.public_api import BitgetPublicApi

EXIT_OK: Final = 0
EXIT_FAILED: Final = 1
EXIT_USAGE: Final = 2
EXIT_ENVIRONMENT: Final = EXIT_ENVIRONMENT_REFUSED
EXIT_PREREGISTRATION: Final = 4
EXIT_LOCKED: Final = 5

PROJECT_NAME: Final = "t2-sentiment-agent"
PLUMBING_SYMBOL: Final = "BTCUSDT"

CommandRunner = Callable[[Sequence[str], Path, float], tuple[int, str, str]]
"""``(argv, cwd, timeout_s) -> (exit code, stdout, stderr)`` for ``setup``'s ``npm ci``."""


class UsageError(RuntimeError):
    """A command was asked for something it refuses; exit 2 with the message."""


@dataclass
class Context:
    """What every command runs with. Tests build one with fakes; ``main`` builds the real one."""

    root: Path
    clock: Clock
    out: TextIO
    parts: Parts
    command_runner: CommandRunner

    def say(self, text: str = "") -> None:
        print(text, file=self.out, flush=True)


# ================================================================================================
# Entry points
# ================================================================================================


def main(argv: Sequence[str] | None = None) -> int:
    """``t2sa``. Returns the process exit code."""
    return run_cli(argv)


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    root: Path | None = None,
    clock: Clock | None = None,
    parts: Parts | None = None,
    out: TextIO | None = None,
    command_runner: CommandRunner | None = None,
) -> int:
    """The CLI with its collaborators injectable (tests pass fakes; nothing here reads globals)."""
    parser = build_parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    ctx = Context(
        root=(root or _discover_root(args.root)).resolve(),
        clock=clock or VenueClock(),
        out=out or sys.stdout,
        parts=parts or Parts(),
        command_runner=command_runner or _run_command,
    )
    handler: Callable[[Context, argparse.Namespace], int] = args.handler
    try:
        return handler(ctx, args)
    except UsageError as exc:
        ctx.say(f"refused: {exc}")
        return EXIT_USAGE
    except InstanceLocked as exc:
        ctx.say(f"refused: {exc}")
        return EXIT_LOCKED
    except RefusedToStart as exc:
        ctx.say(f"refused: {exc}")
        return EXIT_USAGE
    except EnvironmentRefused as exc:
        ctx.say(f"ENVIRONMENT REFUSED: {exc.reason}")
        ctx.say("No order was sent. Exit 3 (DESIGN.md §11.2).")
        return EXIT_ENVIRONMENT
    except GenesisError as exc:
        ctx.say(f"refused: {exc}")
        return EXIT_PREREGISTRATION
    except LedgerError as exc:
        ctx.say(f"LEDGER ERROR: {exc}")
        return EXIT_FAILED


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="t2sa",
        description="t2-sentiment-agent: Qwen decides, a reduce-only kernel gates, Agent Hub "
        "places paper orders on Bitget Demo.",
    )
    parser.add_argument("--root", type=Path, default=None, help="project root (default: found)")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def command(name: str, handler: Callable[[Context, argparse.Namespace], int], text: str) -> Any:
        p = sub.add_parser(name, help=text, description=text)
        p.set_defaults(handler=handler)
        return p

    def mode_arg(p: Any, default: str = "paper") -> None:
        p.add_argument(
            "--mode", choices=[m.value for m in RunMode], default=default, help="run mode"
        )

    def llm_arg(p: Any) -> None:
        p.add_argument(
            "--llm",
            choices=["live", "scripted", "recorded"],
            default=None,
            help="decision model: live Qwen (default for paper), scripted (needs --script), "
            "recorded (replays this ledger's completions)",
        )
        p.add_argument(
            "--script", type=Path, default=None, help="JSON decisions for --llm scripted"
        )

    p = command("preflight", cmd_preflight, "keyless checks before the Demo key exists")
    p.add_argument("--offline", action="store_true", help="skip every network probe")
    command("setup", cmd_setup, "install the pinned Agent Hub CLI and create var/")
    command("prove-env", cmd_prove_env, "prove the configured key is a Demo key; log the proof")
    p = command(
        "plumbing-test",
        cmd_plumbing_test,
        "owner-approved: one minimum-size BTCUSDT Demo order with its stop, then closed",
    )
    p.add_argument("--owner-approved", action="store_true", help="the owner approves the orders")
    p = command("genesis", cmd_genesis, "write the pre-registration (seq 0) and stamp it")
    mode_arg(p)
    p.add_argument("--commit", default=None, help="the git commit the agent runs from")
    p.add_argument("--no-anchor", action="store_true", help="do not stamp with OpenTimestamps")
    p = command(
        "go-live",
        cmd_go_live,
        "the paper run in one command: prove the Demo key, write the genesis once, run the loop",
    )
    p.add_argument("--commit", default=None, help="the git commit the agent runs from")
    p.add_argument("--interval", type=float, default=30.0, help="seconds between ticks")
    p.add_argument("--max-ticks", type=int, default=None, help="stop after N ticks")
    p = command("run", cmd_run, "run the loop")
    mode_arg(p)
    llm_arg(p)
    p.add_argument("--interval", type=float, default=30.0, help="seconds between ticks")
    p.add_argument("--max-ticks", type=int, default=None, help="stop after N ticks")
    p.add_argument(
        "--fresh", action="store_true", help="archive the old simulated/dryrun ledger first"
    )
    p = command("once", cmd_once, "run one tick of the loop")
    mode_arg(p)
    llm_arg(p)
    p = command("decide", cmd_decide, "an owner-requested decision (logged as an intervention)")
    mode_arg(p)
    llm_arg(p)
    p.add_argument("--reason", required=True, help="why the owner asks for a decision")
    p.add_argument("--symbols", default="", help="comma-separated universe symbols to name")
    p = command("reconcile", cmd_reconcile, "one reconciliation sweep (read-only)")
    mode_arg(p)
    p.add_argument("--full", action="store_true", help="also read the full order history")
    p = command("export", cmd_export, "publish the record to public/")
    mode_arg(p)
    p.add_argument("--out", type=Path, default=None, help="output directory (default public/)")
    p.add_argument("--light", action="store_true", help="skip the arms that need candles")
    p.add_argument(
        "--coin-flips", type=int, default=1000, help="coin-flip null arms (default 1000)"
    )
    p = command("verify", cmd_verify, "verify a ledger's hash chain and blobs")
    mode_arg(p)
    p.add_argument("--ledger", type=Path, default=None, help="a ledger file to verify")
    p.add_argument("--blobs", type=Path, default=None, help="its blob directory")
    p.add_argument("--public", type=Path, default=None, help="an exported directory")
    p = command("replay", cmd_replay, "re-run a recorded decision cycle keylessly")
    mode_arg(p)
    p.add_argument("--decision", required=True, help="the decision id (dec-...)")
    p.add_argument("--public", type=Path, default=None, help="replay from an exported directory")
    p = command("status", cmd_status, "status from the ledger and the health file")
    p.add_argument("--mode", choices=[m.value for m in RunMode], default=None)
    p = command("amend", cmd_amend, "log that the loaded policy replaces the one in force")
    mode_arg(p)
    p.add_argument("--reason", required=True, help="why the policy changes")
    p.add_argument("--owner-confirmed", action="store_true", help="the owner confirms the change")
    p = command(
        "x-posted",
        cmd_x_posted,
        "record the URL of the owner's X post of the pre-registration (the loop may be running)",
    )
    mode_arg(p)
    p.add_argument("--url", required=True, help="the post: https://x.com/<handle>/status/<id>")
    p = command("probe-toolkit", cmd_probe_toolkit, "measure every Bitget toolkit surface")
    mode_arg(p)
    p = command("rivals", cmd_rivals, "run rival sentiment agents on the recorded snapshots")
    mode_arg(p)
    p.add_argument("--approve-tokens", type=int, default=None, help="approved Qwen token spend")
    p.add_argument("--offline-only", action="store_true", help="only the rivals that call no model")
    p = command("redteam", cmd_redteam, "attack the sentiment input of recorded snapshots")
    mode_arg(p)
    p.add_argument("--approve-tokens", type=int, default=None, help="approved Qwen token spend")
    return parser


def _discover_root(explicit: Path | None) -> Path:
    """``--root``, else the nearest directory upwards holding this project's ``pyproject.toml``,
    else the source checkout this package runs from, else the working directory."""
    if explicit is not None:
        return explicit
    for base in (Path.cwd(), *Path.cwd().parents):
        if _is_project(base):
            return base
    source = Path(__file__).resolve().parents[3]
    if _is_project(source):
        return source
    return Path.cwd()


def _is_project(path: Path) -> bool:
    try:
        text = (path / "pyproject.toml").read_text(encoding="utf-8")
    except OSError:
        return False
    return f'name = "{PROJECT_NAME}"' in text


def _run_command(argv: Sequence[str], cwd: Path, timeout_s: float) -> tuple[int, str, str]:
    program = shutil.which(argv[0])
    if program is None:
        return 127, "", f"{argv[0]} is not on PATH"
    try:
        done = subprocess.run(  # noqa: S603 - a fixed argv built in this module, no shell
            [program, *argv[1:]],
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return 124, "", f"{argv[0]} did not finish within {timeout_s:g} s"
    return done.returncode, done.stdout, done.stderr


def _mode(args: argparse.Namespace) -> RunMode:
    return RunMode(args.mode)


def _llm(args: argparse.Namespace, mode: RunMode) -> LlmChoice:
    chosen: str | None = args.llm
    if chosen is None:
        return "live" if mode is RunMode.PAPER else "scripted"
    if mode is RunMode.PAPER and chosen != "live":
        raise UsageError("the paper run decides with live Qwen only (--llm live)")
    if chosen == "live":
        return "live"
    if chosen == "recorded":
        return "recorded"
    return "scripted"


def _parts_for(ctx: Context, args: argparse.Namespace, llm: LlmChoice) -> Parts:
    """The context's parts, plus the script file for a ``scripted`` model when one was given."""
    script: Path | None = getattr(args, "script", None)
    if llm != "scripted" or script is None or ctx.parts.chat_model is not None:
        return ctx.parts
    completions = [completion_from_json(obj) for obj in _read_script(script)]
    return Parts(**{**_parts_fields(ctx.parts), "scripted": completions})


def _parts_fields(parts: Parts) -> dict[str, Any]:
    return {name: getattr(parts, name) for name in Parts.__dataclass_fields__}


def _read_script(path: Path) -> list[Mapping[str, Any]]:
    """A JSON array of decision objects, or JSON Lines of them."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list) or not all(isinstance(d, dict) for d in data):
            raise UsageError(f"{path} must hold a JSON array of decision objects")
        return list(data)
    out: list[Mapping[str, Any]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise UsageError(f"{path} line {number} is not a JSON object")
        out.append(item)
    return out


def _open_app(ctx: Context, mode: RunMode, llm: LlmChoice, args: argparse.Namespace) -> App:
    if mode is RunMode.PAPER and llm == "scripted":
        raise UsageError("the paper run decides with live Qwen only")
    return build_app(ctx.root, mode, llm=llm, clock=ctx.clock, parts=_parts_for(ctx, args, llm))


# ================================================================================================
# preflight
# ================================================================================================


@dataclass
class _Check:
    name: str
    ok: bool
    critical: bool
    detail: str


def cmd_preflight(ctx: Context, args: argparse.Namespace) -> int:
    """Everything that can be checked before the Demo key exists, with nothing sent."""
    checks: list[_Check] = []
    root = ctx.root
    policy = ctx.parts.policy or ACTIVE_POLICY
    paths = Paths(root=root, mode=RunMode.DRYRUN)
    paths.ensure()

    prompts = prompt_hashes()
    locks = lock_hashes(root)
    commit = git_commit(root)
    checks.append(_Check("policy hash", True, True, f"{policy.version} {policy.content_hash()}"))
    checks.append(
        _Check(
            "prompt hashes",
            bool(prompts),
            True,
            ", ".join(f"{k} {v[:12]}" for k, v in sorted(prompts.items())),
        )
    )
    checks.append(
        _Check(
            "dependency locks",
            "tools/agent-hub/package-lock.json" in locks,
            True,
            ", ".join(f"{k} {v[:12]}" for k, v in sorted(locks.items())) or "none found",
        )
    )
    checks.append(
        _Check(
            "code commit",
            commit is not None,
            False,
            commit or "not a git checkout: `t2sa genesis` will need --commit",
        )
    )
    checks.append(_scratch_chain_check(root, ctx.clock))
    free = shutil.disk_usage(root).free
    checks.append(
        _Check(
            "free disk for the paper run",
            free >= MIN_FREE_BYTES_FOR_PAPER,
            False,
            f"{free / 1e9:.1f} GB free; `t2sa go-live` needs "
            f"{MIN_FREE_BYTES_FOR_PAPER / 1e9:.0f} GB",
        )
    )

    contract = vendor_contract(paths.agent_hub)
    checks.append(
        _Check(
            "Agent Hub CLI installed and its paper-trading contract intact",
            contract.ok,
            True,
            "; ".join(contract.findings) or f"{BGC_PACKAGE}: every vendor check holds",
        )
    )
    runner: BgcRunner | None = ctx.parts.bgc_runner
    if runner is None and contract.ok:
        try:
            from sentiment_agent.execution.bgc import SubprocessBgcRunner

            runner = SubprocessBgcRunner(paths.agent_hub)
        except RuntimeError as exc:
            checks.append(_Check("node available", False, True, str(exc)))

    if not args.offline:
        blobs = FileBlobStore(paths.blobs)
        market = ctx.parts.market or BitgetPublicApi(clock=ctx.clock, blobs=blobs)
        symbols = policy.symbols
        demo = market.quotes(PriceSource.DEMO, symbols)
        live = market.quotes(PriceSource.LIVE, symbols)
        specs = market.instruments(PriceSource.DEMO, symbols)
        for name, got in (
            ("Demo quotes", demo),
            ("live quotes", live),
            ("Demo instruments", specs),
        ):
            missing = [s for s in symbols if s not in got]
            checks.append(
                _Check(
                    f"keyless {name}",
                    not missing,
                    True,
                    f"{len(got)} of {len(symbols)}" + (f"; missing {missing}" if missing else ""),
                )
            )
        if runner is not None:
            transport = BgcTransport(
                runner=runner,
                clock=ctx.clock,
                blobs=blobs,
                credentials=None,
                proof=None,
                dry_run_only=True,
            )
            for symbol in symbols:
                checks.append(_preview_check(transport, symbol, demo, specs, policy))
        toolkit = ctx.parts.toolkit or default_toolkit(ctx.clock, blobs)
        if isinstance(toolkit, ToolkitFacade):
            probe = probe_all(toolkit, universe=symbols, clock=ctx.clock)
            used = [r for r in probe.rows if r.used_in]
            healthy = [r for r in used if r.last_health is not None and r.last_health.value == "ok"]
            checks.append(
                _Check(
                    "toolkit surfaces the agent reads",
                    bool(healthy),
                    False,
                    f"{len(healthy)} of {len(used)} answering; "
                    + ", ".join(
                        f"{r.entry}={r.last_health.value if r.last_health else '?'}" for r in used
                    ),
                )
            )
        else:
            mood, calls = toolkit.mood()
            checks.append(
                _Check(
                    "toolkit mood",
                    mood.crypto_fear_greed is not None,
                    False,
                    ", ".join(f"{c.source}={c.health.value}" for c in calls),
                )
            )
    report = {
        "at": ctx.clock.now().isoformat(),
        "checks": [c.__dict__ for c in checks],
        "passed": all(c.ok for c in checks if c.critical),
    }
    out = paths.var / "preflight"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for check in checks:
        mark = "ok  " if check.ok else ("FAIL" if check.critical else "warn")
        ctx.say(f"[{mark}] {check.name}: {check.detail}")
    ctx.say("preflight " + ("passed" if report["passed"] else "FAILED"))
    return EXIT_OK if report["passed"] else EXIT_FAILED


def _scratch_chain_check(root: Path, clock: Clock) -> _Check:
    """Write a scratch chain, verify it, tamper with one byte, and check the break is found."""
    scratch_root = root / "var" / "preflight"
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch_root, prefix="chain-") as scratch:
        path = Path(scratch) / "scratch.jsonl"
        chain = HashChainLedger(path, mode=RunMode.SIMULATED, clock=clock)
        for index in range(3):
            chain.append(
                EventKind.NOTE,
                Note(at=clock.now(), author="system", text=f"preflight scratch event {index}"),
            )
        clean = verify_file(path)
        data = bytearray(path.read_bytes())
        at = data.find(b"scratch event 1")
        data[at] = ord("S")
        tampered_path = Path(scratch) / "tampered.jsonl"
        tampered_path.write_bytes(bytes(data))
        tampered = verify_file(tampered_path)
    ok = clean.intact and not tampered.intact and tampered.first_break_at == 1
    return _Check(
        "hash chain: a scratch chain verifies and a one-byte edit is located",
        ok,
        True,
        f"clean intact={clean.intact}; edited seq 1 -> first break at {tampered.first_break_at}",
    )


def _preview_check(
    transport: BgcTransport,
    symbol: str,
    demo: Mapping[str, Any],
    specs: Mapping[str, Any],
    policy: Any,
) -> _Check:
    quote = demo.get(symbol)
    spec = specs.get(symbol)
    if quote is None or spec is None:
        return _Check(f"dry-run preview {symbol}", False, True, "no Demo quote or spec")
    try:
        intent = _minimum_open(symbol, quote, spec, policy, tag="preflight")
        preview = transport.preview(intent)
    except Exception as exc:  # reported per symbol; the preflight lists every one
        return _Check(f"dry-run preview {symbol}", False, True, f"{type(exc).__name__}: {exc}")
    return _Check(
        f"dry-run preview {symbol}",
        True,
        True,
        f"bgc would send {preview.method} {preview.path} {json.dumps(preview.would_send)}",
    )


def _minimum_open(symbol: str, quote: Any, spec: Any, policy: Any, *, tag: str) -> OrderIntent:
    """A buy of the venue's minimum size at the Demo ask, with its preset stop, for a dry-run
    preview only. It carries no decision and no ruling: it is never approved and never sent."""
    qty = _minimum_qty(spec, min(quote.bid, quote.mark))
    entry = entry_price(Side.BUY, quote, quote.mark)
    core = intent_core(
        ruling_id=f"{tag}-{content_hash({'symbol': symbol, 'at': quote.fetched_at})[:24]}",
        symbol=symbol,
        side=Side.BUY,
        qty=qty,
        purpose=OrderPurpose.OPEN,
        split_index=0,
    )
    notional = qty * entry
    return OrderIntent(
        intent_id=content_hash(core),
        ruling_id=str(core["ruling_id"]),
        decision_id=None,
        symbol=symbol,
        side=Side.BUY,
        qty=qty,
        reduce_only=False,
        purpose=OrderPurpose.OPEN,
        reference_price=entry,
        notional=notional,
        expected_fee=notional * fee_rate(spec),
        stop_loss_price=stop_price(entry, Side.BUY, policy, spec),
        client_oid=client_oid(core),
    )


def _minimum_qty(spec: Any, price: Decimal) -> Decimal:
    """The smallest quantity on the grid that clears both ``minOrderQty`` and ``minOrderAmount``
    at ``price``."""
    step: Decimal = spec.qty_step
    needed = max(spec.min_order_qty, spec.min_order_amount / price)
    steps = (needed / step).to_integral_value(rounding=ROUND_CEILING)
    qty: Decimal = steps * step
    return qty


# ================================================================================================
# setup
# ================================================================================================


def cmd_setup(ctx: Context, args: argparse.Namespace) -> int:
    """``npm ci --ignore-scripts`` in tools/agent-hub (the lockfile pins bgc 3.0.0), then the
    vendor contract check, the var/ directories and the .gitignore rules for secrets."""
    paths = Paths(root=ctx.root, mode=RunMode.PAPER)
    for mode in RunMode:
        Paths(root=ctx.root, mode=mode).ensure()
    paths.secrets.mkdir(parents=True, exist_ok=True)
    code, out, err = ctx.command_runner(
        ["npm", "ci", "--ignore-scripts", "--no-audit", "--no-fund"], paths.agent_hub, 600.0
    )
    ctx.say(f"npm ci in tools/agent-hub: exit {code}")
    tail = (err or out).strip().splitlines()[-5:]
    for line in tail:
        ctx.say(f"  {line}")
    contract = vendor_contract(paths.agent_hub)
    ctx.say(
        "vendor contract: " + ("holds" if contract.ok else "FAILS: " + "; ".join(contract.findings))
    )
    ignored = _gitignored(ctx.root)
    ctx.say(
        ".gitignore keeps .secrets/ and var/ out of the repository: " + ("yes" if ignored else "NO")
    )
    ctx.say("")
    ctx.say("What only the owner can do next (DESIGN.md §21):")
    ctx.say(
        "  1. .secrets/demo.env: BITGET_KEY_ENVIRONMENT=demo, BITGET_API_KEY, BITGET_SECRET_KEY,"
    )
    ctx.say("     BITGET_PASSPHRASE (a Demo key: Read + Trade, no Withdraw)")
    ctx.say("  2. .secrets/qwen.env: BITGET_QWEN_API_KEY (and the daily cap approval)")
    ctx.say("  then: t2sa prove-env, t2sa plumbing-test --owner-approved, t2sa genesis")
    return EXIT_OK if code == 0 and contract.ok and ignored else EXIT_FAILED


def _gitignored(root: Path) -> bool:
    try:
        text = (root / ".gitignore").read_text(encoding="utf-8")
    except OSError:
        return False
    rules = {line.strip() for line in text.splitlines()}
    return ".secrets/" in rules and "var/" in rules


# ================================================================================================
# prove-env
# ================================================================================================


def _proof_ledger(ctx: Context) -> HashChainLedger:
    """Where PAPER records go before and after genesis: the paper ledger once it has a genesis,
    the pre-genesis ledger until then (a genesis may only be seq 0)."""
    chain = open_chain(ctx.root, RunMode.PAPER, ctx.clock)
    if genesis_of(chain) is not None:
        return chain
    return open_pregenesis(ctx.root, ctx.clock)


def _runner(ctx: Context) -> BgcRunner:
    if ctx.parts.bgc_runner is not None:
        return ctx.parts.bgc_runner
    from sentiment_agent.execution.bgc import SubprocessBgcRunner

    try:
        return SubprocessBgcRunner(Paths(root=ctx.root, mode=RunMode.PAPER).agent_hub)
    except RuntimeError as exc:
        raise UsageError(f"Agent Hub is not installed: {exc}. Run `t2sa setup`") from None


def _prove(ctx: Context, chain: HashChainLedger) -> EnvironmentProof:
    blobs = FileBlobStore(Paths(root=ctx.root, mode=RunMode.PAPER).blobs)
    try:
        proof = prove_environment(ctx.root, runner=_runner(ctx), clock=ctx.clock, blobs=blobs)
    except EnvironmentRefused as refused:
        if refused.proof is not None:
            chain.append(EventKind.ENVIRONMENT_PROOF, refused.proof)
        raise
    chain.append(EventKind.ENVIRONMENT_PROOF, proof)
    return proof


def cmd_prove_env(ctx: Context, args: argparse.Namespace) -> int:
    chain = _proof_ledger(ctx)
    proof = _prove(ctx, chain)
    where = chain.path.relative_to(ctx.root).as_posix()
    ctx.say(f"environment proof logged to {where}: " + ("PASSED" if proof.passed else "NOT PASSED"))
    ctx.say(
        f"  key declared Demo: {proof.key_declared_demo}; bgc {proof.bgc_package}; SDK sends "
        f"paptrading: {proof.paptrading_header_confirmed}"
    )
    ctx.say(
        f"  Demo read ok: {proof.demo_read_ok} ({proof.demo_read_code}); live read rejected: "
        f"{proof.live_read_rejected} ({proof.live_read_code}); hold mode {proof.hold_mode}"
    )
    if proof.account is not None:
        ctx.say(f"  Demo account equity: {proof.account.equity_usdt} USDT")
    for reason in proof.reasons:
        ctx.say(f"  reason: {reason}")
    return EXIT_OK if proof.passed else EXIT_FAILED


# ================================================================================================
# plumbing-test
# ================================================================================================


def cmd_plumbing_test(ctx: Context, args: argparse.Namespace) -> int:
    """One minimum-size BTCUSDT market order with its preset stop, confirmed, then closed.

    It pins the venue's real response shapes before the scored log begins (DESIGN.md §16 step 5,
    §20). It is not a model decision and it is not scored: it runs only before the paper genesis,
    is logged to the pre-genesis ledger, and passes through the same kernel, approval minter and
    executor as every other order (grounding, G9, does not apply: there is no model text).
    """
    if not args.owner_approved:
        raise UsageError(
            "the plumbing test sends two real orders to Bitget Demo (a minimum-size BTCUSDT buy "
            "with its stop, then its close). Run it again with --owner-approved once the owner "
            "has approved it"
        )
    paper = open_chain(ctx.root, RunMode.PAPER, ctx.clock)
    if paper.head() is not None:
        raise UsageError(
            "the paper ledger has begun; the plumbing test runs only before the genesis, so its "
            "orders never enter the scored log"
        )
    paths = Paths(root=ctx.root, mode=RunMode.PAPER)
    paths.ensure()
    lock = InstanceLock(paths.lock, mode=RunMode.PAPER, clock=ctx.clock)
    lock.acquire()
    try:
        return _plumbing(ctx, paths)
    finally:
        lock.release()


def _plumbing(ctx: Context, paths: Paths) -> int:
    policy = ctx.parts.policy or ACTIVE_POLICY
    chain = open_pregenesis(ctx.root, ctx.clock)
    proof = _prove(ctx, chain)
    if not proof.passed:
        raise EnvironmentRefused(
            "the environment proof did not pass: " + "; ".join(proof.reasons), proof=proof
        )
    credentials = load_demo_credentials(ctx.root)
    blobs = FileBlobStore(paths.blobs)
    projection = Projection.from_ledger(chain, policy)
    ledger = ProjectingLedger(chain, projection)
    transport = BgcTransport(
        runner=_runner(ctx),
        clock=ctx.clock,
        blobs=blobs,
        credentials=credentials,
        proof=proof,
        dry_run_only=False,
    )
    tracker = OrderTracker()
    tracker.restore(projection.events((*ORDER_EVENT_KINDS, EventKind.FILL)))
    executor = Executor(
        transport=transport,
        ledger=ledger,
        tracker=tracker,
        clock=ctx.clock,
        poll_timeout_s=ctx.parts.poll_timeout_s,
        poll_interval_s=ctx.parts.poll_interval_s,
        sleep=ctx.parts.sleep or time.sleep,
    )
    reconciler = Reconciler(transport=transport, ledger=ledger, tracker=tracker, clock=ctx.clock)
    market = ctx.parts.market or BitgetPublicApi(clock=ctx.clock, blobs=blobs)
    toolkit = ctx.parts.toolkit or default_toolkit(ctx.clock, blobs)
    snapshots = SnapshotBuilder(
        market=market,
        toolkit=toolkit,
        crowd=ctx.parts.crowd or CompositeCrowd([]),
        policy=policy,
        clock=ctx.clock,
        mode=RunMode.PAPER,
    )
    symbol = PLUMBING_SYMBOL
    specs = {
        s: spec
        for s, spec in market.instruments(PriceSource.DEMO, [symbol]).items()
        if s == symbol and spec.source is PriceSource.DEMO
    }
    if symbol not in specs:
        raise UsageError(f"the Demo instrument spec for {symbol} is unreadable; nothing was sent")
    kernel = RiskKernel(policy, ctx.clock)
    breaker = Breaker(policy, ctx.clock)
    breaker.restore(projection.breaker_transitions)
    stops = StopManager(transport=transport, policy=policy, clock=ctx.clock, specs=specs)
    planner = Planner(policy)
    guards = frozenset(ALL_GUARDS - {GuardId.G9_GROUNDING})

    def note(text: str, author: Literal["system", "owner"] = "system") -> None:
        clean = scrub_note(text, ctx.root)
        ledger.append(EventKind.NOTE, Note(at=ctx.clock.now(), author=author, text=clean))

    note(
        f"plumbing test begins (owner-approved, pre-genesis, not scored): one minimum-size "
        f"{symbol} market buy with its preset stop, then its close. Every order passes the "
        "kernel, the approval minter and the executor; G9 does not apply (no model text)",
        author="owner",
    )
    snapshot = snapshots.build(book=None, light=True)
    ledger.append(EventKind.SNAPSHOT, SnapshotEvent(snapshot=snapshot))

    def marks_now() -> dict[str, Decimal]:
        return {s: q.mark for s, q in market.quotes(PriceSource.DEMO, [symbol]).items()}

    def leg(tag: str, weight_of: Callable[[BookState, KernelInputs], float]) -> list[str]:
        demo = market.quotes(PriceSource.DEMO, [symbol])
        live = market.quotes(PriceSource.LIVE, [symbol])
        now = ctx.clock.now()
        marks = {s: q.mark for s, q in demo.items()}
        book = projection.book(at=now, marks=marks, mark_source=PriceSource.DEMO)
        inputs = KernelInputs(
            at=now,
            demo_quotes=dict(demo),
            live_quotes=dict(live),
            specs=specs,
            demo_index_move_bps_3h={
                s: f.demo_index_move_bps_3h for s, f in snapshot.features.items()
            },
            snapshot_id=snapshot.snapshot_id,
            snapshot_taken_at=snapshot.taken_at,
            venue_unreconciled=(
                projection.reconciliations[-1].unreconciled if projection.reconciliations else ()
            ),
        )
        weight = weight_of(book, inputs)
        state, transition = breaker.assess(book, inputs=inputs, llm_outage=False)
        if transition is not None:
            ledger.append(EventKind.BREAKER_TRANSITION, transition)
        decision_id = f"plumbing-{tag}-{content_hash({'at': now, 'tag': tag})[:16]}"
        context = RulingContext(decision_id=decision_id, protective_reason=None)
        ruling = kernel.rule(
            proposed={symbol: weight},
            book=book,
            inputs=inputs,
            context=context,
            breaker=state,
            guards=guards,
        )
        blob = ruling_context_blob(
            blobs,
            kind="rule",
            book=book,
            inputs=inputs,
            breaker=state,
            context=context,
            proposed={symbol: weight},
            guards=[g.value for g in ruling.guards_applied],
            llm_outage=False,
        )
        ledger.append(EventKind.KERNEL_RULING, ruling, blobs=[blob])
        plan = planner.plan(ruling, book, inputs, now=ctx.clock.now())
        ledger.append(EventKind.ORDER_PLAN, plan)
        if not plan.intents:
            reasons = "; ".join(f"{s.symbol}: {s.reason}" for s in plan.skipped)
            note(f"plumbing {tag}: no order was planned ({reasons}); nothing sent")
            raise UsageError(f"the kernel and planner produced no {tag} order ({reasons})")
        orders = executor.execute(planner.approve(plan, ruling, book, inputs))
        after = projection.book(at=ctx.clock.now(), marks=marks, mark_source=PriceSource.DEMO)
        reconciler.run(
            book=after,
            since=now - timedelta(minutes=5),
            known_fill_ids=tracker.fill_ids(),
            full_history=False,
        )
        synced = projection.book(at=ctx.clock.now(), marks=marks, mark_source=PriceSource.DEMO)
        for sync in stops.sync(
            synced, transport.stop_orders(), venue_positions=transport.positions()
        ):
            ledger.append(EventKind.STOP_SYNC, sync, blobs=[sync.blob] if sync.blob else [])
        for error in stops.errors():
            note(f"plumbing stop {error.action} for {error.symbol} failed: {error.message}")
        return [
            f"{tag}: {o.side.value} {o.qty} {o.symbol} -> {o.state.value}, venue order "
            f"{o.venue_order_id}, {len(o.fills)} fill(s)"
            for o in orders
        ]

    def open_weight(book: BookState, inputs: KernelInputs) -> float:
        quote = inputs.demo_quotes.get(symbol)
        spec = specs[symbol]
        if quote is None:
            raise UsageError(f"no Demo quote for {symbol}; nothing was sent")
        qty = _minimum_qty(spec, min(quote.bid, quote.mark))
        # Half a grid step above the minimum: the planner rounds toward zero, onto the minimum.
        return float((qty + spec.qty_step / 2) * quote.mark / book.equity)

    lines = leg("open", open_weight)
    lines += leg("close", lambda book, inputs: 0.0)
    report = reconciler.run(
        book=projection.book(at=ctx.clock.now(), marks=marks_now(), mark_source=PriceSource.DEMO),
        since=ctx.clock.now() - timedelta(hours=1),
        known_fill_ids=tracker.fill_ids(),
        full_history=True,
    )
    verdict = "clean" if report.clean else "; ".join(d.detail for d in report.discrepancies)
    note(f"plumbing test done: {'; '.join(lines)}; final reconciliation: {verdict}")
    for line in lines:
        ctx.say(line)
    ctx.say(f"final reconciliation: {verdict}")
    ctx.say(f"logged to {chain.path.relative_to(ctx.root).as_posix()} (pre-genesis, disclosed)")
    return EXIT_OK if report.clean else EXIT_FAILED


# ================================================================================================
# genesis
# ================================================================================================


def cmd_genesis(ctx: Context, args: argparse.Namespace) -> int:
    mode = _mode(args)
    policy = ctx.parts.policy or ACTIVE_POLICY
    paths = Paths(root=ctx.root, mode=mode)
    paths.ensure()
    lock = InstanceLock(paths.lock, mode=mode, clock=ctx.clock)
    lock.acquire()
    try:
        chain = open_chain(ctx.root, mode, ctx.clock)
        head = chain.head()
        if head is not None:
            raise UsageError(
                f"the {mode.value} ledger already holds {head.seq + 1} event(s); a genesis is "
                "seq 0 and is written once"
            )
        if mode is RunMode.PAPER:
            _require_passed_pregenesis_proof(ctx)
        commit = args.commit or git_commit(ctx.root)
        if commit is None:
            raise UsageError(
                "this project is not a git checkout: pass --commit <full sha> of the code that runs"
            )
        predecessor, declared = declaration_for(policy)
        genesis = build_genesis(
            policy=policy,
            prompt_hashes=prompt_hashes(),
            mode=mode,
            code_commit=commit.lower(),
            lock_hashes=lock_hashes(ctx.root),
            bgc_package=BGC_PACKAGE,
            clock=ctx.clock,
            predecessor=predecessor,
            declared_changes=declared,
        )
        blobs = FileBlobStore(paths.blobs)
        record = _oi_record(ctx, policy, blobs)
        oi_blob = blobs.put(canonical_json(record), OI_THRESHOLDS_MEDIA_TYPE)
        event = chain.append(EventKind.GENESIS, genesis, blobs=[oi_blob])
        ctx.say(f"genesis written: {mode.value} ledger seq 0, hash {event.hash}")
        ctx.say(f"  policy {policy.version} {genesis.policy_hash}; commit {genesis.code_commit}")
        if genesis.predecessor is not None:
            ctx.say(
                f"  follows the run pre-registered at {genesis.predecessor.genesis_hash} "
                f"({genesis.predecessor.policy_version}); declared changes: "
                + ", ".join(f"{c.change_id} ({c.kind})" for c in genesis.declared_changes)
            )
        thresholds: Mapping[str, float] = record.get("thresholds_pct", {})
        disabled = record.get("disabled") or []
        ctx.say(
            "  open-interest thresholds frozen: "
            + (", ".join(f"{s} {v:.4f}%" for s, v in thresholds.items()) or "none")
            + (f"; disabled: {', '.join(disabled)}" if disabled else "")
        )
        if not args.no_anchor:
            anchored = stamp(
                event,
                runner=ctx.parts.ots_runner or subprocess_ots_runner,
                blobs=blobs,
                workdir=paths.anchors,
                clock=ctx.clock,
            )
            chain.append(EventKind.ANCHOR, anchored)
            ctx.say(f"  OpenTimestamps: {anchored.status}: {anchored.detail}")
        if mode is RunMode.PAPER:
            ctx.say("")
            ctx.say("The owner posts this on X before the first order (DESIGN.md §16 step 7):")
            ctx.say("")
            ctx.say(x_post_text(event))
        return EXIT_OK
    finally:
        lock.release()


def _require_passed_pregenesis_proof(ctx: Context) -> None:
    pre = open_pregenesis(ctx.root, ctx.clock)
    latest: EnvironmentProof | None = None
    for event in pre.events(frozenset({EventKind.ENVIRONMENT_PROOF})):
        latest = EnvironmentProof.model_validate(event.payload)
    if latest is None or not latest.passed:
        raise UsageError(
            "no passed environment proof in the pre-genesis ledger: a paper genesis is written "
            "only once the Demo key is proven (run `t2sa prove-env`)"
        )


def _oi_record(ctx: Context, policy: Policy, blobs: FileBlobStore) -> dict[str, Any]:
    """The open-interest thresholds to freeze: computed from bitget-mcp-server's trailing history,
    or (tests and rehearsals only, from code) the ones injected through :class:`Parts`."""
    now = ctx.clock.now()
    if ctx.parts.oi_thresholds is not None:
        return {
            "version": 1,
            "computed_at": now.isoformat(),
            "source": "injected by the caller (a test or rehearsal), not measured",
            "thresholds_pct": {s: float(v) for s, v in sorted(ctx.parts.oi_thresholds.items())},
            "disabled": [],
        }
    toolkit = ctx.parts.toolkit
    data = (
        toolkit.data
        if isinstance(toolkit, ToolkitFacade)
        else default_data_service(ctx.clock, blobs)
    )
    history, notes = oi_history(
        data, crypto_symbols(policy), days=policy.triggers.oi_jump_lookback_days
    )
    return frozen_oi_thresholds(history, notes, policy=policy, at=now)


# ================================================================================================
# run, once, decide, reconcile
# ================================================================================================


def _require_script(args: argparse.Namespace, llm: LlmChoice, ctx: Context) -> None:
    if (
        llm == "scripted"
        and getattr(args, "script", None) is None
        and ctx.parts.chat_model is None
        and not ctx.parts.scripted
    ):
        raise UsageError(
            "--llm scripted needs --script FILE (a JSON array of decision objects); for a keyless "
            "replay of this ledger's own completions use --llm recorded"
        )


MIN_FREE_BYTES_FOR_PAPER: Final = 5_000_000_000
"""Free disk the paper run requires before it starts. Measured on the 2026-09-24 dry run against
live data: a light snapshot (every 5 minutes) writes about 0.56 MB of ledger line and blobs and a
full decision snapshot about 1.7 MB, so roughly 0.2 GB a day. Five GB covers the three-day window
twenty times over, with room for exports and OpenTimestamps proofs."""


def cmd_go_live(ctx: Context, args: argparse.Namespace) -> int:
    """The paper run in one command, and the same command to restart it.

    In order, stopping at the first failure: both credential files are present and the Demo one
    declares itself Demo (nothing is logged or sent before this); if the paper ledger has no
    genesis yet, a fresh environment proof is logged to the pre-genesis ledger and must pass, then
    the genesis is written and stamped and the X post text printed; finally the loop runs with live
    Qwen until Ctrl+C. When the genesis already exists it is not touched: the loop resumes, and
    :func:`~sentiment_agent.runtime.wiring.build_app` proves the environment again before anything
    can be sent. The plumbing test is not part of this command: it is owner-approved and must run
    before the genesis (``t2sa plumbing-test --owner-approved``).
    """
    load_demo_credentials(ctx.root)  # EnvironmentRefused (exit 3) when missing or not Demo
    free = shutil.disk_usage(ctx.root).free
    if free < MIN_FREE_BYTES_FOR_PAPER:
        raise UsageError(
            f"only {free / 1e9:.1f} GB free on the disk holding {ctx.root.name}/; the paper run "
            f"needs at least {MIN_FREE_BYTES_FOR_PAPER / 1e9:.0f} GB (about 0.2 GB of ledger and "
            "blobs a day, measured). A full disk stops the ledger mid-run. Free space first"
        )
    if ctx.parts.chat_model is None:  # a test injects its stand-in, exactly as build_app allows
        try:
            load_qwen_env(ctx.root)
        except RuntimeError as exc:
            raise UsageError(f"the paper run decides with live Qwen: {exc}") from None
    paper = open_chain(ctx.root, RunMode.PAPER, ctx.clock)
    if genesis_of(paper) is None:
        if paper.head() is not None:
            raise UsageError(
                "the paper ledger holds events but no genesis; it cannot be pre-registered now. "
                "Stop and inspect var/ledger/paper.jsonl"
            )
        ctx.say("1/3 environment proof")
        code = cmd_prove_env(ctx, argparse.Namespace())
        if code != EXIT_OK:
            return code
        ctx.say("2/3 genesis")
        code = cmd_genesis(
            ctx, argparse.Namespace(mode=RunMode.PAPER.value, commit=args.commit, no_anchor=False)
        )
        if code != EXIT_OK:
            return code
        ctx.say("3/3 the paper loop")
    else:
        ctx.say("the paper genesis exists: resuming the paper loop (the environment is re-proven)")
    return cmd_run(
        ctx,
        argparse.Namespace(
            mode=RunMode.PAPER.value,
            llm="live",
            script=None,
            interval=args.interval,
            max_ticks=args.max_ticks,
            fresh=False,
        ),
    )


def cmd_run(ctx: Context, args: argparse.Namespace) -> int:
    mode = _mode(args)
    llm = _llm(args, mode)
    _require_script(args, llm, ctx)
    if args.fresh:
        if mode is RunMode.PAPER:
            raise UsageError("--fresh archives a rehearsal ledger; the paper ledger is never moved")
        lock = InstanceLock(Paths(root=ctx.root, mode=mode).lock, mode=mode, clock=ctx.clock)
        lock.acquire()
        try:
            archived = archive_ledger(ctx.root, mode, ctx.clock)
        finally:
            lock.release()
        if archived is not None:
            ctx.say(f"archived the old {mode.value} ledger to {archived.relative_to(ctx.root)}")
    app = _open_app(ctx, mode, llm, args)
    with app:
        loop = RunLoop(app)
        exporter = BackgroundExporter(ctx)
        loop.exporter = exporter
        stop = threading.Event()
        with _signals(stop):
            if args.max_ticks is not None:
                for index in range(max(0, args.max_ticks)):
                    if stop.is_set():
                        break
                    report = loop.tick()
                    ctx.say(f"tick {index}: {', '.join(report.did) or 'nothing due'}")
                    if index < args.max_ticks - 1:
                        stop.wait(args.interval)
            else:
                ctx.say(f"running {mode.value} (a tick every {args.interval:g} s); Ctrl+C stops")
                try:
                    loop.run_forever(stop=stop, interval_s=args.interval)
                finally:
                    _finish_export(app, exporter)
        if args.max_ticks is not None:
            _finish_export(app, exporter)
        for line in status_lines(app):
            ctx.say(line)
    return EXIT_OK


@contextlib.contextmanager
def _signals(stop: threading.Event) -> Iterator[None]:
    """SIGINT (and SIGTERM where it exists) set ``stop``; the loop ends after the current tick."""
    previous: dict[int, Any] = {}
    if threading.current_thread() is threading.main_thread():
        for name in ("SIGINT", "SIGTERM"):
            number = getattr(signal, name, None)
            if number is None:
                continue
            with contextlib.suppress(ValueError, OSError):
                previous[number] = signal.signal(number, lambda *_: stop.set())
    try:
        yield
    finally:
        for number, handler in previous.items():
            with contextlib.suppress(ValueError, OSError):
                signal.signal(number, handler)


def cmd_once(ctx: Context, args: argparse.Namespace) -> int:
    mode = _mode(args)
    llm = _llm(args, mode)
    _require_script(args, llm, ctx)
    with _open_app(ctx, mode, llm, args) as app:
        loop = RunLoop(app, probe_toolkit=False)
        exporter = BackgroundExporter(ctx)
        loop.exporter = exporter
        report = loop.tick()
        _finish_export(app, exporter)
        ctx.say(f"tick at {_iso(report.at)}: {', '.join(report.did) or 'nothing due'}")
        if loop.last_card is not None:
            _say_card(ctx, loop.last_card)
    return EXIT_OK


def cmd_decide(ctx: Context, args: argparse.Namespace) -> int:
    """An owner-requested decision. Runs it now when no loop holds the mode; otherwise leaves the
    request in the inbox for the running loop's next tick. Either way it is logged as an owner
    intervention and counted on the published record."""
    mode = _mode(args)
    llm = _llm(args, mode)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    policy = ctx.parts.policy or ACTIVE_POLICY
    unknown = [s for s in symbols if s not in policy.symbols]
    if unknown:
        raise UsageError(f"not in the universe: {', '.join(unknown)}")
    if not args.reason.strip():
        raise UsageError("an owner decision needs a written reason")
    _require_script(args, llm, ctx)
    try:
        app = _open_app(ctx, mode, llm, args)
    except InstanceLocked:
        path = request_decision(
            Paths(root=ctx.root, mode=mode).inbox, args.reason, at=ctx.clock.now(), symbols=symbols
        )
        ctx.say(
            f"a {mode.value} loop is running; the request is queued for its next tick "
            f"({path.relative_to(ctx.root).as_posix()})"
        )
        return EXIT_OK
    with app:
        loop = RunLoop(app, probe_toolkit=False)
        loop.request_owner_decision(args.reason, symbols=symbols)
        report = loop.tick()
        ctx.say(f"tick: {', '.join(report.did)}")
        if loop.last_card is None:
            ctx.say("the owner trigger was not admitted; the ledger's trigger note says why")
            return EXIT_FAILED
        _say_card(ctx, loop.last_card)
    return EXIT_OK


def cmd_reconcile(ctx: Context, args: argparse.Namespace) -> int:
    mode = _mode(args)
    # A sweep makes no model call, so it never needs the Qwen key: the recorded model stands in.
    with build_app(ctx.root, mode, llm="recorded", clock=ctx.clock, parts=ctx.parts) as app:
        report = RunLoop(app).reconcile(full=args.full)
    if report is None:
        ctx.say(f"{mode.value}: nothing was ever sent, so there is nothing to reconcile")
        return EXIT_OK
    ctx.say(
        f"reconciliation at {_iso(report.at)}: {report.orders_checked} order(s) checked, "
        f"{len(report.new_fill_ids)} new fill(s), {len(report.resolved_unknown)} unknown resolved"
    )
    for d in report.discrepancies:
        ctx.say(f"  {d.kind} {d.symbol or ''} {d.client_oid or ''}: {d.detail}")
    ctx.say("clean" if report.clean else f"{len(report.discrepancies)} discrepancy(ies)")
    return EXIT_OK if report.clean else EXIT_FAILED


def _say_card(ctx: Context, card: DecisionCard) -> None:
    decision = card.decision
    ctx.say(f"decision {card.decision_id}: {card.outcome.value if card.outcome else 'n/a'}")
    if decision is not None:
        ctx.say(f"  stance {decision.stance.value}: {decision.summary}")
        for target in decision.targets:
            ctx.say(f"  {target.symbol} target {target.target:+.2f}: {target.thesis}")
    kernel = card.kernel
    if kernel is not None:
        for inst in kernel.instruments:
            if inst.proposed_weight is None and not inst.changed_by_kernel:
                continue
            bound = f" (bound by {inst.binding_guard.value})" if inst.binding_guard else ""
            proposed = "hold" if inst.proposed_weight is None else f"{inst.proposed_weight:+.4%}"
            ctx.say(
                f"  kernel {inst.symbol}: proposed {proposed} -> approved "
                f"{inst.approved_weight:+.4%}{bound}"
            )
    for order in card.orders:
        ctx.say(
            f"  order {order.client_oid} {order.side.value} {order.qty} {order.symbol} "
            f"{order.purpose.value}: {order.state.value}, venue id {order.venue_order_id}"
        )
    if card.ledger_seqs:
        ctx.say(f"  ledger seqs {card.ledger_seqs[0]}-{card.ledger_seqs[-1]}")


def _iso(at: datetime) -> str:
    return at.strftime("%Y-%m-%dT%H:%M:%SZ")


# ================================================================================================
# verify
# ================================================================================================


def cmd_verify(ctx: Context, args: argparse.Namespace) -> int:
    targets: list[tuple[str, Path, Path | None]] = []
    if args.public is not None:
        public = args.public if args.public.is_absolute() else ctx.root / args.public
        targets.append(("public", public / "ledger.jsonl", public / "blobs"))
    elif args.ledger is not None:
        targets.append(("ledger", args.ledger, args.blobs))
    else:
        mode = _mode(args)
        blobs = ctx.root / "var" / "blobs"
        targets.append((mode.value, ledger_path(ctx.root, mode), blobs))
        if mode is RunMode.PAPER:
            pre = Paths(root=ctx.root, mode=mode).pregenesis_ledger
            if pre.exists():
                targets.append(("paper pre-genesis", pre, blobs))
    ok = True
    for label, path, blobs_root in targets:
        if not path.exists():
            ctx.say(f"{label}: {path} does not exist")
            ok = False
            continue
        result = verify_file(path, blobs_root=blobs_root)
        ctx.say(
            f"{label}: {'INTACT' if result.intact else 'BROKEN'}: {result.events} event(s), head "
            f"{result.head_hash}; {result.anchor}"
        )
        if result.first_break_at is not None:
            ctx.say(f"  first break at seq {result.first_break_at}: {result.break_reason}")
        if result.truncated:
            ctx.say("  the head anchor says events were removed or replaced")
        if result.missing_blobs:
            shown = ", ".join(b[:12] for b in result.missing_blobs[:10])
            ctx.say(f"  {len(result.missing_blobs)} referenced blob(s) missing or altered: {shown}")
        ok = ok and result.intact
    return EXIT_OK if ok else EXIT_FAILED


# ================================================================================================
# replay
# ================================================================================================


@dataclass
class ReplayResult:
    """What a keyless replay reproduced. ``None`` where the cycle logged nothing to compare."""

    decision_id: str
    decision_matches: bool
    ruling_id: str | None
    ruling_matches: bool | None
    intent_ids: tuple[str, ...]
    plan_matches: bool | None
    lines: list[str]

    @property
    def reproduced(self) -> bool:
        return (
            self.decision_matches
            and self.ruling_matches is not False
            and self.plan_matches is not False
        )


_REPLAYED_FAILURES: Final[Mapping[str, type[Exception]]] = {
    "QwenTimeout": QwenTimeout,
    "QwenTransportError": QwenTransportError,
    "QwenError": QwenError,
    "BudgetExhausted": BudgetExhausted,
    "TimeoutError": TimeoutError,
}
"""The model-service failures a logged attempt can record, re-raised by a replay of the cycle."""


class ReplayModel:
    """The model as the ledger recorded it: every logged completion, in order, then the logged
    failure when the cycle ended in one (a timeout, a transport error, the daily cap).

    A replayed outage raises the same class of error with the same message, so the contract
    classifies it into the same outcome and the cycle reproduces its decision id; a call with no
    recording left and no recorded failure is a :class:`ReplayMismatch`, as it should be.
    """

    def __init__(self, call: LlmCallRecord, blobs: BlobStore) -> None:
        self._recorded = RecordedChatModel(replay_calls(call, blobs), model_name=call.model)
        self._failure: tuple[type[Exception], str] | None = None
        for ref in call.response_blobs:
            if ref.media_type != ATTEMPT_MEDIA_TYPE:
                continue
            attempt = json.loads(blobs.get(ref.sha256).decode("utf-8"))
            error = attempt.get("error")
            if isinstance(error, dict) and attempt.get("completion") is None:
                kind = _REPLAYED_FAILURES.get(str(error.get("type")), QwenTransportError)
                self._failure = (kind, str(error.get("message", "")))

    @property
    def model_name(self) -> str:
        return self._recorded.model_name

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_mode: bool,
        max_tokens: int,
        thinking: Thinking,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> Completion:
        if self._recorded.remaining == 0 and self._failure is not None:
            kind, message = self._failure
            raise kind(message)
        return self._recorded.complete(
            messages,
            json_mode=json_mode,
            max_tokens=max_tokens,
            thinking=thinking,
            temperature=temperature,
            seed=seed,
        )


class _OverlayBlobs:
    """Reads from the record's blob store; writes stay in memory, so a replay never adds to it."""

    def __init__(self, base: FileBlobStore) -> None:
        self._base = base
        self._memory: dict[str, bytes] = {}

    def put(self, data: bytes, media_type: str) -> BlobRef:
        raw = bytes(data)
        digest = sha256_hex(raw)
        self._memory[digest] = raw
        return BlobRef(sha256=digest, media_type=media_type, size=len(raw))

    def get(self, sha256: str) -> bytes:
        if sha256 in self._memory:
            return self._memory[sha256]
        return self._base.get(sha256)


def replay_decision(
    events: Sequence[LedgerEvent], blobs: FileBlobStore, decision_id: str, *, policy: Policy
) -> ReplayResult:
    """Re-run one logged decision cycle from the log alone and compare every identity it logged.

    The snapshot, the triggers and the book the model saw are read from the ledger; the model's
    answer comes from its logged completions through :class:`RecordedChatModel`; the kernel rules
    again on the market inputs, breaker state and book it logged with its ruling; the planner plans
    again. Nothing is fetched and nothing is written to the record.
    """
    by_seq = sorted(events, key=lambda e: e.seq)
    found = _find_decision(by_seq, decision_id)
    if found is None:
        raise UsageError(f"no decision {decision_id} in this ledger")
    decision_event, record = found
    before = [e for e in by_seq if e.seq < decision_event.seq]
    policy_in_force = _policy_at(before, policy)
    snapshot = None
    triggers_by_id: dict[str, Trigger] = {}
    for event in before:
        if event.kind is EventKind.SNAPSHOT:
            snap = SnapshotEvent.model_validate(event.payload).snapshot
            if snap.snapshot_id == record.snapshot_id:
                snapshot = snap
        elif event.kind is EventKind.TRIGGER:
            trigger = Trigger.model_validate(event.payload)
            triggers_by_id[trigger.trigger_id] = trigger
    if snapshot is None:
        raise UsageError(f"the snapshot {record.snapshot_id} the decision read is not in the log")
    missing = [t for t in record.trigger_ids if t not in triggers_by_id]
    if missing:
        raise UsageError(f"triggers {missing} named by the decision are not in the log")
    triggers = [triggers_by_id[t] for t in record.trigger_ids]

    agent = DecisionAgent(
        model=ReplayModel(record.call, blobs),
        policy=policy_in_force,
        blobs=_OverlayBlobs(blobs),
        clock=ManualClock(record.decided_at),
    )
    again = agent.decide(snapshot, record.book_before, triggers)
    decision_matches = (
        again.decision_id == record.decision_id
        and again.outcome is record.outcome
        and again.decision == record.decision
        and again.proposed_weights == record.proposed_weights
        and again.grounding == record.grounding
    )
    lines = [
        f"decision {record.decision_id}: replayed {again.decision_id} "
        + ("IDENTICAL" if decision_matches else "DIFFERS")
        + f" (outcome {again.outcome.value}, prompt {again.call.prompt_hash[:16]})"
    ]
    result = ReplayResult(record.decision_id, decision_matches, None, None, (), None, lines)

    after = [e for e in by_seq if e.seq > decision_event.seq]
    ruling_event = _answering_ruling(after, record)
    if ruling_event is None:
        lines.append("no kernel ruling answered this decision (a model outage on a flat book)")
        return result
    ruling = KernelRuling.model_validate(ruling_event.payload)
    result.ruling_id = ruling.ruling_id
    context_ref = next(
        (b for b in ruling_event.blobs if b.media_type == RULING_CONTEXT_MEDIA_TYPE), None
    )
    if context_ref is None:
        lines.append(
            f"ruling {ruling.ruling_id[:16]} carries no context blob; it cannot be re-ruled"
        )
        result.ruling_matches = False
        return result
    stored = json.loads(blobs.get(context_ref.sha256).decode("utf-8"))
    book = BookState.model_validate(stored["book"])
    inputs = KernelInputs.model_validate(stored["inputs"])
    breaker_state = BreakerState.model_validate(stored["breaker"])
    kernel = RiskKernel(policy_in_force, ManualClock(ruling.at))
    ruled: KernelRuling | None
    if stored["kind"] == "rule":
        if again.decision is None:
            lines.append("the replayed cycle has no decision, but a model ruling was logged")
            result.ruling_matches = False
            return result
        context = RulingContext(
            decision_id=again.decision_id,
            protective_reason=None,
            grounding=dict(again.grounding),
            invalidation_fired={t.symbol: t.invalidation_triggered for t in again.decision.targets},
        )
        ruled = kernel.rule(
            proposed=again.proposed_weights,
            book=book,
            inputs=inputs,
            context=context,
            breaker=breaker_state,
            guards=frozenset(GuardId(g) for g in stored["guards"]),
        )
    else:
        ruled = kernel.protective(
            book=book, inputs=inputs, breaker=breaker_state, llm_outage=bool(stored["llm_outage"])
        )
    result.ruling_matches = ruled is not None and ruled.ruling_id == ruling.ruling_id
    lines.append(
        f"kernel ruling {ruling.ruling_id[:16]}: replayed "
        f"{ruled.ruling_id[:16] if ruled is not None else 'nothing'} "
        + ("IDENTICAL" if result.ruling_matches else "DIFFERS")
    )
    plan_event = next(
        (
            e
            for e in after
            if e.kind is EventKind.ORDER_PLAN and e.payload.get("ruling_id") == ruling.ruling_id
        ),
        None,
    )
    if plan_event is None or ruled is None:
        lines.append("no order plan was logged under the ruling")
        return result
    plan = OrderPlan.model_validate(plan_event.payload)
    replanned = plan_orders(ruled, book, inputs, policy_in_force, now=plan.created_at)
    logged_ids = tuple(i.intent_id for i in plan.intents)
    result.intent_ids = tuple(i.intent_id for i in replanned.intents)
    result.plan_matches = replanned.plan_id == plan.plan_id and result.intent_ids == logged_ids
    lines.append(
        f"order plan {plan.plan_id[:16]}: {len(logged_ids)} intent(s), replayed "
        + ("IDENTICAL" if result.plan_matches else "DIFFERS")
        + (": " + ", ".join(i.client_oid for i in replanned.intents) if replanned.intents else "")
    )
    return result


def _find_decision(
    events: Sequence[LedgerEvent], decision_id: str
) -> tuple[LedgerEvent, DecisionRecord] | None:
    for event in events:
        if event.kind is EventKind.DECISION:
            record = DecisionEvent.model_validate(event.payload).record
            if record.decision_id == decision_id:
                return event, record
    return None


def _policy_at(before: Sequence[LedgerEvent], loaded: Policy) -> Policy:
    """The policy in force before a decision: the genesis's, then each amendment's in turn."""
    policy = loaded
    for event in before:
        if event.kind is EventKind.GENESIS:
            policy = Genesis.model_validate(event.payload).policy
        elif event.kind is EventKind.AMENDMENT:
            policy = Amendment.model_validate(event.payload).new_policy
    return policy


def _answering_ruling(after: Sequence[LedgerEvent], record: DecisionRecord) -> LedgerEvent | None:
    """The ruling this decision's cycle logged: its own ruling when it decided; the model-outage
    flatten when it did not. Anything after the next decision belongs to another cycle."""
    for event in after:
        if event.kind is EventKind.DECISION:
            return None
        if event.kind is not EventKind.KERNEL_RULING:
            continue
        payload = event.payload
        if record.outcome is LlmOutcome.DECIDED:
            if payload.get("decision_id") == record.decision_id:
                return event
        elif payload.get("protective_reason") == ProtectiveReason.LLM_OUTAGE.value:
            return event
    return None


def cmd_replay(ctx: Context, args: argparse.Namespace) -> int:
    if args.public is not None:
        public = args.public if args.public.is_absolute() else ctx.root / args.public
        path, blobs_root = public / "ledger.jsonl", public / "blobs"
    else:
        path, blobs_root = ledger_path(ctx.root, _mode(args)), ctx.root / "var" / "blobs"
    if not path.exists():
        raise UsageError(f"{path} does not exist")
    check = verify_file(path, blobs_root=blobs_root)
    if not check.intact:
        ctx.say(f"the ledger does not verify ({check.break_reason or check.anchor}); not replayed")
        return EXIT_FAILED
    result = replay_decision(
        read_events(path),
        FileBlobStore(blobs_root),
        args.decision,
        policy=ctx.parts.policy or ACTIVE_POLICY,
    )
    for line in result.lines:
        ctx.say(line)
    ctx.say("REPRODUCED" if result.reproduced else "NOT REPRODUCED")
    return EXIT_OK if result.reproduced else EXIT_FAILED


def read_events(path: Path) -> list[LedgerEvent]:
    """Every event of a ledger file that ``verify_file`` has already checked line by line."""
    return [
        LedgerEvent.model_validate_json(line)
        for line in path.read_bytes().splitlines()
        if line.strip()
    ]


# ================================================================================================
# status, amend, probe-toolkit
# ================================================================================================


def cmd_status(ctx: Context, args: argparse.Namespace) -> int:
    policy = ctx.parts.policy or ACTIVE_POLICY
    if args.mode:
        modes = [RunMode(args.mode)]
    else:
        modes = [m for m in RunMode if ledger_path(ctx.root, m).exists()]
    if not modes:
        ctx.say("no ledger yet: run `t2sa preflight`, then `t2sa run --mode simulated`")
        return EXIT_OK
    for mode in modes:
        for line in status_from_ledger(
            ctx.root,
            mode,
            clock=ctx.clock,
            policy=policy,
            starting_equity=ctx.parts.starting_equity,
        ):
            ctx.say(line)
        if mode is RunMode.PAPER:
            pre = Paths(root=ctx.root, mode=mode).pregenesis_ledger
            if pre.exists():
                chain = HashChainLedger(pre, mode=RunMode.PAPER, clock=ctx.clock)
                proofs = list(chain.events(frozenset({EventKind.ENVIRONMENT_PROOF})))
                last = EnvironmentProof.model_validate(proofs[-1].payload) if proofs else None
                head = chain.head()
                state = "none" if last is None else ("passed" if last.passed else "NOT passed")
                ctx.say(
                    f"pre-genesis ledger: {0 if head is None else head.seq + 1} event(s); latest "
                    f"environment proof {state}"
                )
        ctx.say("")
    return EXIT_OK


def cmd_amend(ctx: Context, args: argparse.Namespace) -> int:
    if not args.owner_confirmed:
        raise UsageError(
            "an amendment changes the pre-registered policy and is published; the owner confirms "
            "it with --owner-confirmed"
        )
    mode = _mode(args)
    policy = ctx.parts.policy or ACTIVE_POLICY
    paths = Paths(root=ctx.root, mode=mode)
    paths.ensure()
    lock = InstanceLock(paths.lock, mode=mode, clock=ctx.clock)
    lock.acquire()
    try:
        chain = open_chain(ctx.root, mode, ctx.clock)
        event = amend(
            chain, new_policy=policy, reason=args.reason, owner_confirmed=True, clock=ctx.clock
        )
        anchored = stamp(
            event,
            runner=ctx.parts.ots_runner or subprocess_ots_runner,
            blobs=FileBlobStore(paths.blobs),
            workdir=paths.anchors,
            clock=ctx.clock,
        )
        chain.append(EventKind.ANCHOR, anchored)
    finally:
        lock.release()
    ctx.say(
        f"amendment logged at seq {event.seq} ({event.hash[:16]}): policy {policy.version} "
        f"{policy.content_hash()} is now in force; anchor {anchored.status}"
    )
    return EXIT_OK


X_POST_CLOCK_SKEW: Final = timedelta(minutes=2)
"""How far X's clock may lag ours before a post counts as older than the genesis it quotes."""


def cmd_x_posted(ctx: Context, args: argparse.Namespace) -> int:
    """Record the owner's X post of the genesis hash as an owner note in the ledger.

    Runs beside a live loop: it takes no instance lock and appends through the ledger's own writer
    lock, as ``t2sa decide`` does, so the post is recorded without restarting the agent. The post
    is refused when its id says it was made before the genesis existed: a post that old cannot
    carry this run's genesis hash, so it is the wrong post.
    """
    mode = _mode(args)
    try:
        url = x_post_url(args.url)
    except GenesisError as exc:
        raise UsageError(str(exc)) from None
    chain = open_chain(ctx.root, mode, ctx.clock)
    found = genesis_of(chain)
    if found is None:
        raise UsageError(f"the {mode.value} ledger has no genesis, so there is no post to record")
    genesis_event, _ = found
    posted_at = x_posted_at(url)
    if posted_at < genesis_event.ts - X_POST_CLOCK_SKEW:
        raise UsageError(
            f"that post was made {posted_at.isoformat()} (its id says so), before the genesis at "
            f"{genesis_event.ts.isoformat()}; it cannot carry this run's genesis hash"
        )
    recorded = recorded_x_post(chain.events(frozenset({EventKind.NOTE})))
    if recorded is not None and recorded[0] == url:
        ctx.say(f"already recorded at seq {recorded[1].seq}: {url}")
        return EXIT_OK
    event = chain.append(
        EventKind.NOTE, Note(at=ctx.clock.now(), author="owner", text=X_POST_NOTE_PREFIX + url)
    )
    ctx.say(
        f"recorded at seq {event.seq}: {url}, posted {posted_at.isoformat()} by its id; the next "
        "export shows it on the page"
    )
    return EXIT_OK


def cmd_probe_toolkit(ctx: Context, args: argparse.Namespace) -> int:
    mode = _mode(args)
    policy = ctx.parts.policy or ACTIVE_POLICY
    paths = Paths(root=ctx.root, mode=mode)
    paths.ensure()
    blobs = FileBlobStore(paths.blobs)
    toolkit = ctx.parts.toolkit or default_toolkit(ctx.clock, blobs)
    if not isinstance(toolkit, ToolkitFacade):
        raise UsageError("probe-toolkit measures the two Bitget MCP services; none is configured")
    probe = probe_all(toolkit, universe=policy.symbols, clock=ctx.clock)
    chain = _proof_ledger(ctx) if mode is RunMode.PAPER else open_chain(ctx.root, mode, ctx.clock)
    event = chain.append(EventKind.TOOLKIT_PROBE, probe)
    out = paths.var / "toolkit_probe.json"
    out.write_text(probe.model_dump_json(indent=2) + "\n", encoding="utf-8")
    counts: dict[str, dict[str, int]] = {}
    for row in probe.rows:
        health = row.last_health.value if row.last_health is not None else "unmeasured"
        by_health = counts.setdefault(row.surface.value, {})
        by_health[health] = by_health.get(health, 0) + 1
    for surface, by_health in sorted(counts.items()):
        ctx.say(f"{surface}: " + ", ".join(f"{h} {n}" for h, n in sorted(by_health.items())))
    ctx.say(
        f"logged at seq {event.seq} of {chain.path.relative_to(ctx.root).as_posix()}; "
        f"written to {out.relative_to(ctx.root).as_posix()}"
    )
    return EXIT_OK


# ================================================================================================
# export: the public record, and the analysis arms it carries
# ================================================================================================

EXPORT_LOCK_WAIT_S: Final = 1800.0
"""How long an export waits for the export lock: a running export or the page publisher's copy."""

COIN_FLIP_SEEDS: Final = 1000
ANALYSIS_DIR: Final = "var/analysis"


def default_out(root: Path, mode: RunMode) -> Path:
    """``public/`` is the published paper record; a rehearsal exports beside the working state,
    so a simulated or dry-run record can never overwrite the scored one."""
    if mode is RunMode.PAPER:
        return root / "public"
    return root / "var" / f"public-{mode.value}"


def analysis_path(root: Path, mode: RunMode, name: str) -> Path:
    return root.joinpath(*ANALYSIS_DIR.split("/"), mode.value, name)


@dataclass
class Analysis:
    """Arms and reports for one export, and what could not be computed and why."""

    arms: list[ArmResult]
    twin: TwinReport | None
    notes: list[str]
    sim_check: SimulatorCheck | None = None


def _projection_of(chain: HashChainLedger, policy: Policy, starting_equity: Decimal) -> Projection:
    fallback = None if chain.mode is RunMode.PAPER else starting_equity
    return Projection.from_ledger(chain, policy, starting_equity=fallback)


def _logged_context(
    event: LedgerEvent, blobs: FileBlobStore
) -> tuple[KernelInputs, dict[str, Any]] | None:
    ref = next((b for b in event.blobs if b.media_type == RULING_CONTEXT_MEDIA_TYPE), None)
    if ref is None:
        return None
    stored = json.loads(blobs.get(ref.sha256).decode("utf-8"))
    return KernelInputs.model_validate(stored["inputs"]), stored


def decisions_with_inputs(
    events: Sequence[LedgerEvent], blobs: FileBlobStore
) -> tuple[list[DecisionRecord], list[KernelInputs], list[KernelRuling]]:
    """Every decision whose cycle logged a ruling, the market inputs that ruling read, and every
    ruling that answers a decision (the twin's inputs; DESIGN.md §14.2)."""
    ordered = sorted(events, key=lambda e: e.seq)
    decisions: list[DecisionRecord] = []
    inputs: list[KernelInputs] = []
    rulings: list[KernelRuling] = []
    for index, event in enumerate(ordered):
        if event.kind is EventKind.KERNEL_RULING and event.payload.get("decision_id"):
            rulings.append(KernelRuling.model_validate(event.payload))
        if event.kind is not EventKind.DECISION:
            continue
        record = DecisionEvent.model_validate(event.payload).record
        answer = _answering_ruling(ordered[index + 1 :], record)
        if answer is None:
            continue
        logged = _logged_context(answer, blobs)
        if logged is None:
            continue
        decisions.append(record)
        inputs.append(logged[0])
    return decisions, inputs, rulings


def _simulator(
    projection: Projection,
    market: MarketData,
    policy: Policy,
    clock: Clock,
    logged_inputs: Sequence[KernelInputs],
) -> tuple[ArmSimulator, datetime, datetime] | str:
    """The simulator every comparison arm runs on, over the book's own hourly window, or why none
    can be built yet."""
    marks = projection.marks
    if len(marks) < 2:
        return "fewer than two hourly marks: there is no window to compare arms over yet"
    start, until = marks[0].at, marks[-1].at
    symbols = policy.symbols
    demo_marks: dict[str, list[Candle]] = {}
    for symbol in symbols:
        rows = market.candles(
            PriceSource.DEMO,
            symbol,
            kind="mark",
            interval="1H",
            start=start - timedelta(hours=1),
            end=until,
        )
        if rows:
            demo_marks[symbol] = rows
    if not demo_marks:
        return "no Demo mark candles could be read for the window"
    spreads: dict[str, list[float]] = {}
    for snapshot in projection.snapshots:
        for symbol, quote in snapshot.demo_quotes.items():
            spread = quote.spread_bps
            if math.isfinite(spread) and 0 <= spread < 20_000:
                spreads.setdefault(symbol, []).append(spread)
    medians = {s: statistics.median(v) for s, v in spreads.items() if v}
    specs: dict[str, InstrumentSpec] = {}
    for given in logged_inputs:
        specs.update(given.specs)
    if not specs:
        specs = {
            s: spec
            for s, spec in market.instruments(PriceSource.DEMO, symbols).items()
            if spec.source is PriceSource.DEMO and spec.symbol == s
        }
    equity = projection.starting_equity or DEFAULT_SIMULATED_EQUITY
    sim = ArmSimulator(
        kernel=RiskKernel(policy, clock),
        policy=policy,
        demo_marks=demo_marks,
        spreads_bps=medians,
        starting_equity=float(equity),
        specs=specs,
    )
    return sim, start, until


def full_analysis(
    chain: HashChainLedger,
    blobs: FileBlobStore,
    market: MarketData,
    *,
    policy: Policy,
    clock: Clock,
    starting_equity: Decimal,
    coin_flip_seeds: int = COIN_FLIP_SEEDS,
) -> Analysis:
    """The baselines, the governed/ungoverned twin and the live-marked mirror (DESIGN.md §14.2-4),
    each on the book's own snapshots, decisions and hourly window. Keyless: candles are public."""
    projection = _projection_of(chain, policy, starting_equity)
    events = list(chain.events())
    decisions, inputs, rulings = decisions_with_inputs(events, blobs)
    built = _simulator(projection, market, policy, clock, inputs)
    if isinstance(built, str):
        return Analysis(arms=[], twin=None, notes=[built])
    sim, start, until = built
    notes: list[str] = []
    live_book = book_arm(projection.marks, projection.closed_trades, projection.fills)
    arms: list[ArmResult] = []
    twin: TwinReport | None = None
    governed: ArmResult | None = None
    if decisions:
        ungoverned = ungoverned_arm(sim, decisions, inputs, start=start, until=until)
        governed = governed_replica(sim, decisions, inputs, start=start, until=until)
        takeovers = sum(1 for t in projection.triggers if t.kind is TriggerKind.OWNER_MANUAL)
        takeovers += len(projection.amendments)
        twin = twin_report(
            decisions, rulings, governed, ungoverned, human_takeovers=takeovers, sim=sim
        )
        arms.extend([ungoverned, governed])
    else:
        notes.append("no ruled decision yet: the twin is not computed")
    # The baselines are ranked against the replica, which pays the same cost model they do; the
    # live book is set beside it with the measured gap (analysis/simcheck.py).
    comparator = governed if governed is not None else live_book
    snapshots = [s for s in projection.snapshots if start - timedelta(hours=1) <= s.taken_at]
    if snapshots:
        arms.extend(run_baselines(sim, snapshots, comparator, coin_flip_seeds=coin_flip_seeds))
    else:
        notes.append("no snapshot inside the window: baselines not run")
    sim_check = simulator_check(
        projection.fills,
        projection.intent,
        sim.half_spread,
        live=live_book,
        replica=governed,
    )
    live: dict[str, list[Candle]] = {}
    for symbol in policy.symbols:
        rows = market.candles(
            PriceSource.LIVE,
            symbol,
            kind="market",
            interval="1H",
            start=start - timedelta(hours=1),
            end=until,
        )
        if rows:
            live[symbol] = rows
    equity = projection.starting_equity or starting_equity
    try:
        arms.append(live_mirror(projection.fills, live, equity, start=start, until=until))
    except ValueError as exc:
        notes.append(f"live mirror not computed: {exc}")
    try:
        arms.append(weekend_counterfactual(decisions, rulings, live, policy=policy))
    except ValueError as exc:
        notes.append(f"weekend counterfactual not computed: {exc}")
    # The rivals that call no model run every hour with the rest, on the same snapshots and the
    # same simulator; the model rivals stay in `t2sa rivals`, which needs an approved budget.
    rival_snapshots, rival_books = _decision_inputs(chain)
    if rival_snapshots:
        try:
            arms.extend(run_rivals(offline_rival_arms(policy), rival_snapshots, rival_books, sim))
        except Exception as exc:  # a rival that cannot run is said; the other arms still publish
            notes.append(f"offline rivals not computed: {type(exc).__name__}: {exc}")
    return Analysis(arms=arms, twin=twin, notes=notes, sim_check=sim_check)


def offline_rival_arms(policy: Policy) -> list[RivalArm]:
    """The rival sentiment agents that call no model (DESIGN.md §14.6): lexicon follow and fade,
    the fear-and-greed confluence trader and the fusion trader."""
    return [
        KeywordSentimentArm(policy=policy, direction="follow"),
        KeywordSentimentArm(policy=policy, direction="fade"),
        FearGreedConfluenceArm(policy=policy),
        SentimentFusionArm(policy=policy),
    ]


def _stored_rivals(root: Path, mode: RunMode) -> list[ArmResult]:
    path = analysis_path(root, mode, "rivals.json")
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [ArmResult.model_validate(a) for a in data.get("arms", [])]


def _stored_redteam(root: Path, mode: RunMode) -> RedTeamReport | None:
    path = analysis_path(root, mode, "redteam.json")
    if not path.exists():
        return None
    return RedTeamReport.model_validate_json(path.read_bytes())


def _latest_probe(chain: HashChainLedger, root: Path) -> ToolkitProbe | None:
    latest: ToolkitProbe | None = None
    for event in chain.events(frozenset({EventKind.TOOLKIT_PROBE})):
        latest = ToolkitProbe.model_validate(event.payload)
    if latest is None:
        stored = root / "var" / "toolkit_probe.json"
        if stored.exists():
            latest = ToolkitProbe.model_validate_json(stored.read_bytes())
    return latest


LIGHT_EXPORT_REASON: Final = (
    "this was a light export (t2sa export --light), which skips the arms that need candles"
)
REDTEAM_NOT_RUN_REASON: Final = (
    "no red-team run is stored for this record: t2sa redteam calls Qwen and runs only with an "
    "owner-approved token budget (--approve-tokens)"
)


def export_record(
    root: Path,
    mode: RunMode,
    *,
    clock: Clock,
    parts: Parts,
    out: Path | None = None,
    light: bool,
    coin_flips: int = COIN_FLIP_SEEDS,
) -> tuple[ExportManifest, list[str]]:
    """Publish ``mode``'s record: the export (``site/export.py``), then the static page
    (``site/render.py``). ``light`` skips the arms that need candles. Serialised by
    ``var/run/export-<mode>.lock`` so the loop's hourly export and a manual one never interleave,
    and so neither moves files into ``public/`` while ``scripts/publish_site.ps1`` is copying it
    (the publisher holds the same lock for the copy, which takes seconds; the export waits up to
    :data:`EXPORT_LOCK_WAIT_S` for it). Returns the manifest and what the analysis could not
    compute."""
    policy = parts.policy or ACTIVE_POLICY
    target = out or default_out(root, mode)
    lock = InstanceLock(
        root / "var" / "run" / f"export-{mode.value}.lock",
        mode=mode,
        clock=clock,
        busy=f"the {mode.value} export lock is held by another export or by the page publisher",
    )
    lock.acquire(wait_s=EXPORT_LOCK_WAIT_S)
    try:
        ledger = HashChainLedger(ledger_path(root, mode), mode=mode, clock=clock)
        blobs = FileBlobStore(root / "var" / "blobs")
        notes: list[str] = []
        arms: list[ArmResult] = []
        twin: TwinReport | None = None
        sim_check: SimulatorCheck | None = None
        twin_reason = LIGHT_EXPORT_REASON
        if not light:
            market = parts.market or BitgetPublicApi(clock=clock, blobs=None)
            try:
                analysis = full_analysis(
                    ledger,
                    blobs,
                    market,
                    policy=policy,
                    clock=clock,
                    starting_equity=parts.starting_equity,
                    coin_flip_seeds=coin_flips,
                )
            except (EnvironmentRefused, LedgerError):
                raise
            except Exception as exc:  # the record still publishes; the gap is said, not hidden
                twin_reason = (
                    f"the analysis failed on this export ({type(exc).__name__}: {exc}); "
                    "the next hourly export retries it"
                )[:600]
                notes.append(twin_reason)
            else:
                arms, twin, notes = analysis.arms, analysis.twin, analysis.notes
                sim_check = analysis.sim_check
                twin_reason = "; ".join(notes) or "the analysis returned no twin"
        # A stored run of `t2sa rivals` adds the model rivals; an arm this export computed afresh
        # is not replaced by an older copy of itself.
        fresh = {a.spec.arm_id for a in arms}
        arms.extend(a for a in _stored_rivals(root, mode) if a.spec.arm_id not in fresh)
        manifest = export_public(
            ledger=ledger,
            blobs=blobs,
            out=target,
            arms=arms,
            twin=twin,
            sim_check=sim_check,
            redteam=_stored_redteam(root, mode),
            toolkit=coverage_matrix(_latest_probe(ledger, root)),
            clock=clock,
            twin_reason=twin_reason,
            redteam_reason=REDTEAM_NOT_RUN_REASON,
        )
        render_site(target)
    finally:
        lock.release()
    return manifest, notes


class BackgroundExporter:
    """The loop's hourly publisher, on a worker thread.

    A full export recomputes a thousand coin-flip arms and reads candles for every symbol; done in
    the loop it would delay the 60-second protective check by minutes. So each call starts one
    export on a worker and returns at once, with any notes the previous export left (its analysis
    gaps, or its failure), which the loop logs. One export runs at a time: an hour whose previous
    export is still running is skipped and says so. The worker reads the ledger through its own
    reader and fetches candles through its own keyless client, so it never touches the loop's
    objects.

    **Every hourly export is a full one** (2026-09-26). Until then five hours in six were light,
    and a light export publishes the book alone, so run 1's page showed no twin, no baseline and no
    coin-flip distribution for five hours in every six. A full export of run 1's record at 29,600
    blobs took 157 s beside a test run, well inside the hour. When the analysis itself fails (a
    candle read, say), :func:`export_record` still publishes the record and says on the page why
    the arms are missing.
    """

    def __init__(self, ctx: Context, *, coin_flips: int = COIN_FLIP_SEEDS) -> None:
        self._ctx = ctx
        self._coin_flips = coin_flips
        self._thread: threading.Thread | None = None
        self._notes: list[str] = []
        self._lock = threading.Lock()
        self.manifests: list[ExportManifest] = []

    def _add(self, text: str) -> None:
        with self._lock:
            self._notes.append(text)

    def _take(self) -> list[str]:
        with self._lock:
            taken, self._notes = self._notes, []
        return taken

    def __call__(self, app: App) -> list[str]:
        notes = self._take()
        if self._thread is not None and self._thread.is_alive():
            notes.append("the previous export is still running; this hour's export is skipped")
            return notes
        worker = threading.Thread(
            target=self._run,
            args=(app.paths.root, app.mode, app.clock),
            name="t2sa-export",
            daemon=True,
        )
        self._thread = worker
        worker.start()
        return notes

    def _run(self, root: Path, mode: RunMode, clock: Clock) -> None:
        try:
            manifest, notes = export_record(
                root,
                mode,
                clock=clock,
                parts=self._ctx.parts,
                light=False,
                coin_flips=self._coin_flips,
            )
        except Exception as exc:  # reported through the loop's next note; trading carries on
            self._add(f"hourly export failed: {type(exc).__name__}: {exc}"[:1500])
            return
        self.manifests.append(manifest)
        for note in notes:
            self._add(f"export analysis not computed: {note}")

    def join(self, timeout_s: float = 900.0) -> list[str]:
        """Wait for a running export (at shutdown) and return what it left to log."""
        if self._thread is not None:
            self._thread.join(timeout_s)
        return self._take()


def _finish_export(app: App, exporter: BackgroundExporter) -> None:
    """Let a running export finish before the process ends, and log what it left."""
    for note in exporter.join():
        with contextlib.suppress(LedgerError):
            app.note(note[:2000])


def cmd_export(ctx: Context, args: argparse.Namespace) -> int:
    mode = _mode(args)
    if not ledger_path(ctx.root, mode).exists():
        raise UsageError(f"there is no {mode.value} ledger to export")
    out: Path | None = args.out
    if out is not None and not out.is_absolute():
        out = ctx.root / out
    try:
        manifest, notes = export_record(
            ctx.root,
            mode,
            clock=ctx.clock,
            parts=ctx.parts,
            out=out,
            light=args.light,
            coin_flips=args.coin_flips,
        )
    except ExportError as exc:
        ctx.say(f"EXPORT REFUSED: {exc}")
        return EXIT_FAILED
    target = out or default_out(ctx.root, mode)
    ctx.say(
        f"exported {len(manifest.written)} file(s) to {target}; ledger head "
        f"{manifest.ledger_head[:16]}"
    )
    for note in notes:
        ctx.say(f"  not computed: {note}")
    ctx.say(f"  check it: python scripts/recompute.py {target}")
    return EXIT_OK


# ================================================================================================
# rivals and redteam: comparison runs that may spend Qwen tokens, approved first
# ================================================================================================


def _decision_inputs(
    chain: HashChainLedger,
) -> tuple[list[PerceptionSnapshot], list[BookState]]:
    """The snapshots the agent decided on, and the book it held at each: the same inputs every
    comparison arm is given (DESIGN.md §14.5, §14.6)."""
    snapshots: dict[str, PerceptionSnapshot] = {}
    pairs: list[tuple[PerceptionSnapshot, BookState]] = []
    for event in chain.events(frozenset({EventKind.SNAPSHOT, EventKind.DECISION})):
        if event.kind is EventKind.SNAPSHOT:
            snap = SnapshotEvent.model_validate(event.payload).snapshot
            snapshots[snap.snapshot_id] = snap
            continue
        record = DecisionEvent.model_validate(event.payload).record
        seen = snapshots.get(record.snapshot_id)
        if seen is not None and (not pairs or seen.taken_at > pairs[-1][0].taken_at):
            pairs.append((seen, record.book_before))
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _approval(ctx: Context, approved: int | None, estimate: int, what: str) -> None:
    """Refuse a run whose Qwen spend is not approved at or above its estimate. A run that can
    spend nothing (the offline arms) needs no approval."""
    ctx.say(f"{what}: estimated Qwen spend up to {estimate} tokens (an upper bound)")
    if estimate == 0:
        return
    if approved is None:
        raise UsageError(
            f"{what} spends up to {estimate} Qwen tokens; the owner approves the spend with "
            f"--approve-tokens N (N at least {estimate})"
        )
    if approved < estimate:
        raise UsageError(
            f"approved {approved} tokens, below the estimate of {estimate}; nothing was run"
        )


def _qwen(ctx: Context, cap: int, clock: Clock) -> ChatModel:
    if ctx.parts.chat_model is not None:
        return ctx.parts.chat_model
    try:
        credentials = load_qwen_env(ctx.root)
    except RuntimeError as exc:
        raise UsageError(f"the LLM arms need .secrets/qwen.env: {exc}") from None
    return QwenChatModel(
        credentials=credentials,
        budget=DailyTokenBudget(max(cap, 1), ctx.clock),
        clock=clock,
        blobs=FileBlobStore(ctx.root / "var" / "blobs"),
    )


def cmd_rivals(ctx: Context, args: argparse.Namespace) -> int:
    """Rival sentiment agents on the agent's own snapshots, marked by the same simulator
    (DESIGN.md §14.6). Written to var/analysis and published by the next export."""
    mode = _mode(args)
    policy = ctx.parts.policy or ACTIVE_POLICY
    chain = open_chain(ctx.root, mode, ctx.clock)
    snapshots, books = _decision_inputs(chain)
    if not snapshots:
        raise UsageError("no decision snapshot in the ledger yet: nothing to compare rivals on")
    offline: list[RivalArm] = offline_rival_arms(policy)
    if finbert_installed():
        offline += [
            FinbertArm(policy=policy, direction="follow"),
            FinbertArm(policy=policy, direction="fade"),
        ]
    else:
        ctx.say("finBERT arms skipped: the [rivals] extra (transformers, torch) is not installed")
    # The LLM arms are built on a placeholder first, only to bound their spend before anything runs.
    probe_model: ChatModel = ctx.parts.chat_model or RecordedChatModel([])
    llm_probe: list[RivalArm] = [
        TradingAgentsSocialArm(model=probe_model, policy=policy),
        QwenHeadlineTraderArm(model=probe_model, policy=policy),
    ]
    if args.offline_only:
        llm_probe = []
        ctx.say("offline arms only: the TradingAgents and Qwen-headline rivals are not run")
    estimate = estimate_qwen_tokens([*offline, *llm_probe], len(snapshots))
    _approval(ctx, args.approve_tokens, estimate, f"rivals on {len(snapshots)} snapshot(s)")
    arms: list[RivalArm] = list(offline)
    if not args.offline_only:
        model = _qwen(ctx, args.approve_tokens or 0, ctx.clock)
        arms += [
            TradingAgentsSocialArm(model=model, policy=policy),
            QwenHeadlineTraderArm(model=model, policy=policy),
        ]
    projection = _projection_of(chain, policy, ctx.parts.starting_equity)
    market = ctx.parts.market or BitgetPublicApi(clock=ctx.clock, blobs=None)
    _, logged_inputs, _ = decisions_with_inputs(
        list(chain.events()), FileBlobStore(ctx.root / "var" / "blobs")
    )
    built = _simulator(projection, market, policy, ctx.clock, logged_inputs)
    if isinstance(built, str):
        raise UsageError(built)
    sim = built[0]
    results = run_rivals(arms, snapshots, books, sim)
    path = analysis_path(ctx.root, mode, "rivals.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "run_at": ctx.clock.now().isoformat(),
        "snapshots": len(snapshots),
        "approved_tokens": args.approve_tokens,
        "estimated_tokens": estimate,
        "arms": [json.loads(r.model_dump_json()) for r in results],
    }
    path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    for result in results:
        m = result.metrics
        ctx.say(
            f"{result.spec.arm_id}: return {m.total_return:+.4%}, max drawdown "
            f"{m.max_drawdown:+.4%}, Sharpe {m.sharpe_ann}, {m.n_closed_trades} trade(s)"
        )
    ctx.say(f"written to {path.relative_to(ctx.root).as_posix()}; `t2sa export` publishes it")
    return EXIT_OK


def cmd_redteam(ctx: Context, args: argparse.Namespace) -> int:
    """The sentiment input under attack (DESIGN.md §14.5): every vector against our agent, our
    agent without the quarantine, and the lexicon and finBERT traders, paired against the clean
    decision on the same recorded snapshot. Published whatever the grade."""
    mode = _mode(args)
    policy = ctx.parts.policy or ACTIVE_POLICY
    chain = open_chain(ctx.root, mode, ctx.clock)
    snapshots, books = _decision_inputs(chain)
    if not snapshots:
        raise UsageError("no decision snapshot in the ledger yet: nothing to attack")
    vectors = load_vectors()
    estimate = redteam_estimate(len(snapshots), len(vectors))
    _approval(ctx, args.approve_tokens, estimate, f"red team, {len(vectors)} vector(s)")
    harness_clock = ManualClock(snapshots[0].taken_at)
    model = _qwen(ctx, args.approve_tokens or 0, harness_clock)
    blobs = _OverlayBlobs(FileBlobStore(ctx.root / "var" / "blobs"))
    agent = DecisionAgent(model=model, policy=policy, blobs=blobs, clock=harness_clock)
    ours = AgentArm(agent)
    keyword = KeywordSentimentArm(policy=policy, direction="follow")
    arms: dict[str, Callable[[PerceptionSnapshot, BookState], dict[str, float]]] = {
        "ours": ours,
        "ours_no_quarantine": WithoutQuarantine(ours, policy=policy),
        "keyword_sentiment": keyword.targets,
    }
    if finbert_installed():
        arms["finbert_sentiment"] = FinbertArm(policy=policy, direction="follow").targets
    specs: dict[str, InstrumentSpec] = {}
    _, logged_inputs, _ = decisions_with_inputs(
        list(chain.events()), FileBlobStore(ctx.root / "var" / "blobs")
    )
    for given in logged_inputs:
        specs.update(given.specs)
    harness = RedTeamHarness(
        arms=arms,
        kernel=RiskKernel(policy, harness_clock),
        policy=policy,
        clock=harness_clock,
        specs=specs or None,
    )
    report = harness.run(snapshots, books, vectors)
    path = analysis_path(ctx.root, mode, "redteam.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=1) + "\n", encoding="utf-8")
    summary = summarise(report)
    for line in summary.model_dump_json(indent=1).splitlines()[:40]:
        ctx.say(line)
    ctx.say(
        f"{len(report.outcomes)} paired outcome(s), {report.qwen_tokens_spent} Qwen tokens spent; "
        f"written to {path.relative_to(ctx.root).as_posix()}"
    )
    return EXIT_OK
