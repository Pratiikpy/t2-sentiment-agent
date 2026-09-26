"""``t2sa``: every subcommand's arguments, its approval gates, and the owner's path end to end.

The PAPER path runs against a stateful fake of the Demo account behind ``bgc``
(:class:`FakeDemoVenue`): the environment proof, the owner-approved plumbing test, the genesis, a
paper decision whose orders are sent with ``--paper-trading`` and reconciled, the status, the
verification and the replay. Nothing leaves the process.
"""

import dataclasses
import io
import json
import shutil
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from runtime.support import (
    FAKE_COMMIT,
    FakeBgc,
    FakeDemoVenue,
    FakeToolkit,
    FakeWorld,
    decision,
    make_parts,
    paper_project,
    target,
)
from sentiment_agent.book.projection import Projection
from sentiment_agent.clock import ManualClock
from sentiment_agent.execution.environment import PAPER_FLAG, READ_ONLY_FLAG
from sentiment_agent.execution.simulated import SimulatedVenue
from sentiment_agent.llm.fakes import ScriptedChatModel
from sentiment_agent.policy import POLICY_V1
from sentiment_agent.runtime import cli as cli_module
from sentiment_agent.runtime.cli import (
    EXIT_ENVIRONMENT,
    EXIT_FAILED,
    EXIT_LOCKED,
    EXIT_OK,
    EXIT_PREREGISTRATION,
    EXIT_USAGE,
    build_parser,
    main,
    run_cli,
)
from sentiment_agent.runtime.wiring import Parts, build_app, open_chain, open_pregenesis
from sentiment_agent.types import (
    EnvironmentProof,
    EventKind,
    Fill,
    Genesis,
    OrderState,
    RunMode,
)

FRIDAY_1320 = datetime(2026, 9, 25, 13, 20, tzinfo=UTC)
COMMANDS = (
    "preflight",
    "setup",
    "prove-env",
    "plumbing-test",
    "genesis",
    "go-live",
    "run",
    "once",
    "decide",
    "reconcile",
    "export",
    "verify",
    "replay",
    "status",
    "amend",
    "x-posted",
    "probe-toolkit",
    "rivals",
    "redteam",
)


def cli(root: Path, clock: ManualClock, parts: Parts, *argv: str) -> tuple[int, str]:
    out = io.StringIO()
    code = run_cli(list(argv), root=root, clock=clock, parts=parts, out=out)
    return code, out.getvalue()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return paper_project(tmp_path / "project")


@pytest.fixture
def friday() -> ManualClock:
    return ManualClock(FRIDAY_1320)


# ================================================================================================
# The parser
# ================================================================================================


def test_every_subcommand_the_design_names_is_there() -> None:
    parser = build_parser()
    choices = next(a for a in parser._actions if a.dest == "command").choices
    assert choices is not None
    assert set(choices) == set(COMMANDS)


def test_main_parses_arguments_and_reports_through_its_exit_code(root: Path) -> None:
    assert main(["--root", str(root), "status"]) == EXIT_OK
    with pytest.raises(SystemExit) as caught:
        main(["--root", str(root), "no-such-command"])
    assert caught.value.code == 2


# ================================================================================================
# Approval gates
# ================================================================================================


def test_plumbing_test_refuses_without_the_owner_approval(root: Path, friday: ManualClock) -> None:
    bgc = FakeBgc()
    code, text = cli(root, friday, make_parts(friday, bgc_runner=bgc), "plumbing-test")
    assert code == EXIT_USAGE
    assert "--owner-approved" in text
    assert bgc.calls == []


def test_plumbing_test_refuses_once_the_paper_ledger_has_begun(
    root: Path, friday: ManualClock
) -> None:
    world = FakeWorld(friday)
    venue = FakeDemoVenue(world)
    parts = make_parts(friday, world=world, bgc_runner=venue, oi_thresholds={"BTCUSDT": 5.0})
    assert cli(root, friday, parts, "prove-env")[0] == EXIT_OK
    assert cli(root, friday, parts, "genesis", "--commit", FAKE_COMMIT, "--no-anchor")[0] == 0
    code, text = cli(root, friday, parts, "plumbing-test", "--owner-approved")
    assert code == EXIT_USAGE
    assert "only before the genesis" in text
    assert venue.sends() == []


@pytest.mark.parametrize("command", ["rivals", "redteam"])
def test_token_spending_commands_print_the_estimate_and_refuse_without_approval(
    tmp_path: Path, command: str
) -> None:
    clock, root, parts = _small_simulated_record(tmp_path)
    code, text = cli(root, clock, parts, command, "--mode", "simulated")
    assert code == EXIT_USAGE
    assert "estimated Qwen spend up to" in text
    assert "--approve-tokens" in text
    code, text = cli(root, clock, parts, command, "--mode", "simulated", "--approve-tokens", "1")
    assert code == EXIT_USAGE
    assert "below the estimate" in text


def test_amend_refuses_without_the_owner_confirmation(tmp_path: Path) -> None:
    clock, root, parts = _small_simulated_record(tmp_path)
    code, text = cli(root, clock, parts, "amend", "--mode", "simulated", "--reason", "tighten")
    assert code == EXIT_USAGE
    assert "--owner-confirmed" in text


def test_amend_logs_the_new_policy_and_the_old_one_no_longer_runs(tmp_path: Path) -> None:
    clock, root, parts = _small_simulated_record(tmp_path)
    tightened = POLICY_V1.model_copy(update={"daily_kill_pct": 0.01})
    new_parts = dataclasses.replace(parts, policy=tightened)
    code, text = cli(
        root,
        clock,
        new_parts,
        "amend",
        "--mode",
        "simulated",
        "--reason",
        "tighter kill",
        "--owner-confirmed",
    )
    assert code == EXIT_OK, text
    chain = open_chain(root, RunMode.SIMULATED, clock)
    kinds = [e.kind for e in chain.events()][-2:]
    assert kinds == [EventKind.AMENDMENT, EventKind.ANCHOR]
    code, text = cli(root, clock, parts, "once", "--mode", "simulated", "--llm", "recorded")
    assert code == EXIT_PREREGISTRATION
    assert "policy in force" in text


# ================================================================================================
# genesis, prove-env, preflight, setup
# ================================================================================================


def test_a_paper_genesis_needs_a_passed_proof_first(root: Path, friday: ManualClock) -> None:
    parts = make_parts(friday, bgc_runner=FakeBgc(), oi_thresholds={"BTCUSDT": 5.0})
    code, text = cli(root, friday, parts, "genesis", "--commit", FAKE_COMMIT)
    assert code == EXIT_USAGE
    assert "prove-env" in text


def test_genesis_needs_a_commit_and_is_written_once(tmp_path: Path) -> None:
    clock = ManualClock(FRIDAY_1320)
    root = tmp_path / "rehearsal"
    root.mkdir()
    parts = make_parts(clock, oi_thresholds={"BTCUSDT": 5.0})
    code, text = cli(root, clock, parts, "genesis", "--mode", "simulated")
    assert code == EXIT_USAGE
    assert "--commit" in text
    git = root / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n", "utf-8")
    (git / "refs" / "heads" / "main").write_text(FAKE_COMMIT + "\n", "utf-8")
    code, text = cli(root, clock, parts, "genesis", "--mode", "simulated")
    assert code == EXIT_OK, text
    assert FAKE_COMMIT in text
    assert "Genesis hash" not in text, "only the paper genesis is posted on X"
    code, text = cli(root, clock, parts, "genesis", "--mode", "simulated")
    assert code == EXIT_USAGE
    assert "written once" in text


def test_prove_env_logs_to_the_pre_genesis_ledger_until_the_genesis(
    root: Path, friday: ManualClock
) -> None:
    code, text = cli(root, friday, make_parts(friday, bgc_runner=FakeBgc()), "prove-env")
    assert code == EXIT_OK
    assert "PASSED" in text
    pre = open_pregenesis(root, friday)
    proofs = list(pre.events(frozenset({EventKind.ENVIRONMENT_PROOF})))
    assert len(proofs) == 1
    assert EnvironmentProof.model_validate(proofs[0].payload).passed
    assert open_chain(root, RunMode.PAPER, friday).head() is None
    failing = FakeBgc(live="loopback_live_negative_param")
    code, text = cli(root, friday, make_parts(friday, bgc_runner=failing), "prove-env")
    assert code == EXIT_FAILED
    assert "NOT PASSED" in text


def test_preflight_offline_checks_hashes_the_chain_and_the_installed_cli(
    root: Path, friday: ManualClock
) -> None:
    code, text = cli(root, friday, make_parts(friday), "preflight", "--offline")
    assert code == EXIT_OK, text
    assert "[ok  ] policy hash" in text
    assert "a one-byte edit is located" in text
    assert "paper-trading contract intact" in text
    report = json.loads((root / "var" / "preflight" / "report.json").read_text("utf-8"))
    assert report["passed"] is True


def test_preflight_previews_every_symbol_with_no_credential(
    root: Path, friday: ManualClock
) -> None:
    bgc = FakeBgc()
    parts = make_parts(friday, bgc_runner=bgc)
    code, text = cli(root, friday, parts, "preflight")
    assert code == EXIT_OK, text
    previews = [c for c in bgc.calls if c[:3] == ("order", "--action", "place")]
    assert len(previews) == len(POLICY_V1.symbols)
    assert all("--dry-run" in c and PAPER_FLAG in c for c in previews)
    assert bgc.sends() == []


def test_setup_runs_npm_ci_in_the_agent_hub_and_names_the_owner_steps(
    root: Path, friday: ManualClock
) -> None:
    (root / ".gitignore").write_text(".secrets/\nvar/\n", "utf-8")
    seen: list[tuple[tuple[str, ...], Path]] = []

    def npm(argv: Sequence[str], cwd: Path, timeout_s: float) -> tuple[int, str, str]:
        seen.append((tuple(argv), cwd))
        return 0, "added 2 packages", ""

    out = io.StringIO()
    code = run_cli(
        ["setup"], root=root, clock=friday, parts=make_parts(friday), out=out, command_runner=npm
    )
    assert code == EXIT_OK, out.getvalue()
    assert seen == [
        (("npm", "ci", "--ignore-scripts", "--no-audit", "--no-fund"), root / "tools" / "agent-hub")
    ]
    assert ".secrets/demo.env" in out.getvalue()
    assert (root / "var" / "ledger").is_dir()


# ================================================================================================
# The owner's paper path, end to end
# ================================================================================================


def test_the_owner_paper_path_plumbing_genesis_decision_status_verify_replay(
    root: Path, friday: ManualClock
) -> None:
    world = FakeWorld(friday)
    venue = FakeDemoVenue(world)
    toolkit = FakeToolkit(friday)
    base = make_parts(friday, world=world, toolkit=toolkit, bgc_runner=venue)

    code, text = cli(root, friday, base, "prove-env")
    assert code == EXIT_OK, text

    code, text = cli(root, friday, base, "plumbing-test", "--owner-approved")
    assert code == EXIT_OK, text
    sends = venue.sends()
    assert len(sends) == 2, "one minimum-size buy with its stop, then its close"
    assert all(PAPER_FLAG in argv for argv in sends)
    assert "--stopLoss" in sends[0]
    assert "--reduceOnly" in sends[1]
    assert venue.positions == {}
    pre = open_pregenesis(root, friday)
    notes = [e.payload["text"] for e in pre.events(frozenset({EventKind.NOTE}))]
    assert any("plumbing test done" in str(n) for n in notes)
    assert open_chain(root, RunMode.PAPER, friday).head() is None, "never in the scored log"

    genesis_parts = dataclasses.replace(base, oi_thresholds={"BTCUSDT": 5.0})
    code, text = cli(root, friday, genesis_parts, "genesis", "--commit", FAKE_COMMIT)
    assert code == EXIT_OK, text
    assert "#BitgetHackathon" in text

    friday.set(FRIDAY_1320 + timedelta(minutes=11))
    model = ScriptedChatModel([decision("act", [target("NVDAUSDT", 1.0), target("BTCUSDT", -1.0)])])
    run_parts = dataclasses.replace(base, chat_model=model)
    code, text = cli(root, friday, run_parts, "once", "--mode", "paper")
    assert code == EXIT_OK, text
    assert "stance act" in text
    chain = open_chain(root, RunMode.PAPER, friday)
    kinds = [e.kind for e in chain.events()]
    assert kinds[0] is EventKind.GENESIS
    assert EventKind.ENVIRONMENT_PROOF in kinds
    created = Genesis.model_validate(next(iter(chain.events())).payload).created_at
    scored = [Fill.model_validate(e.payload) for e in chain.events(frozenset({EventKind.FILL}))]
    assert scored, "the decision's orders filled"
    assert all(f.executed_at >= created for f in scored), "the plumbing fills are never scored"
    first_mark = next(e.seq for e in chain.events() if e.kind is EventKind.MARK)
    assert all(e.seq > first_mark for e in chain.events(frozenset({EventKind.FILL})))
    fills = list(chain.events(frozenset({EventKind.FILL})))
    assert {f.payload["symbol"] for f in fills} == {"NVDAUSDT", "BTCUSDT"}
    assert set(venue.positions) == {"NVDAUSDT", "BTCUSDT"}
    states = {
        e.payload["client_oid"]: e.payload["to_state"]
        for e in chain.events(frozenset({EventKind.ORDER_STATE}))
    }
    assert set(states.values()) == {OrderState.FILLED.value}
    assert all(("--paper-trading" in c) != (READ_ONLY_FLAG in c) for c in venue.calls)
    recs = list(chain.events(frozenset({EventKind.RECONCILIATION})))
    assert recs[-1].payload["discrepancies"] == []

    code, text = cli(root, friday, run_parts, "status", "--mode", "paper")
    assert code == EXIT_OK
    assert "matches the loaded policy" in text
    assert "NVDAUSDT qty" in text
    assert "latest environment proof passed" in text

    code, text = cli(root, friday, run_parts, "verify", "--mode", "paper")
    assert code == EXIT_OK, text
    assert text.count("INTACT") == 2, "the paper ledger and the pre-genesis ledger"

    decision_id = next(iter(chain.events(frozenset({EventKind.DECISION})))).payload["record"][
        "decision_id"
    ]
    code, text = cli(
        root, friday, run_parts, "replay", "--mode", "paper", "--decision", decision_id
    )
    assert code == EXIT_OK, text
    assert "REPRODUCED" in text


# ================================================================================================
# run, once, decide, reconcile, verify, status, export
# ================================================================================================


def _small_simulated_record(tmp_path: Path) -> tuple[ManualClock, Path, Parts]:
    """A simulated rehearsal with a genesis and one decision that opened two legs."""
    clock = ManualClock(FRIDAY_1320)
    root = tmp_path / "sim"
    root.mkdir()
    world = FakeWorld(clock)
    venue = SimulatedVenue(market=world, clock=clock, starting_equity=Decimal("10000"))
    script = [decision("act", [target("NVDAUSDT", 1.0), target("BTCUSDT", 1.0)])]
    parts = make_parts(
        clock, world=world, script=script, venue=venue, oi_thresholds={"BTCUSDT": 5.0}
    )
    code, text = cli(root, clock, parts, "genesis", "--mode", "simulated", "--commit", FAKE_COMMIT)
    assert code == EXIT_OK, text
    run_parts = dataclasses.replace(parts, oi_thresholds=None)
    for at in (
        FRIDAY_1320 + timedelta(minutes=5),
        FRIDAY_1320 + timedelta(minutes=11),
        FRIDAY_1320 + timedelta(minutes=41),
    ):
        clock.set(at)
        code, text = cli(
            root,
            clock,
            run_parts,
            "run",
            "--mode",
            "simulated",
            "--max-ticks",
            "1",
            "--interval",
            "0",
        )
        assert code == EXIT_OK, text
    return clock, root, dataclasses.replace(run_parts, scripted=())


def test_run_ticks_the_loop_and_prints_the_status(tmp_path: Path) -> None:
    clock, root, _parts = _small_simulated_record(tmp_path)
    chain = open_chain(root, RunMode.SIMULATED, clock)
    decisions = list(chain.events(frozenset({EventKind.DECISION})))
    assert len(decisions) == 1
    assert list(chain.events(frozenset({EventKind.FILL})))


def test_run_fresh_archives_a_rehearsal_and_refuses_the_paper_ledger(tmp_path: Path) -> None:
    clock, root, parts = _small_simulated_record(tmp_path)
    code, text = cli(
        root,
        clock,
        parts,
        "run",
        "--mode",
        "simulated",
        "--llm",
        "recorded",
        "--fresh",
        "--max-ticks",
        "1",
        "--interval",
        "0",
    )
    assert code == EXIT_OK, text
    assert "archived the old simulated ledger" in text
    archived = list((root / "var" / "ledger" / "archive").glob("simulated-*.jsonl"))
    assert len(archived) == 1
    code, text = cli(root, clock, parts, "run", "--mode", "paper", "--fresh")
    assert code == EXIT_USAGE


def test_a_scripted_model_needs_its_script(tmp_path: Path) -> None:
    clock, root, parts = _small_simulated_record(tmp_path)
    code, text = cli(root, clock, parts, "once", "--mode", "simulated", "--llm", "scripted")
    assert code == EXIT_USAGE
    assert "--script" in text
    code, text = cli(root, clock, parts, "run", "--mode", "paper", "--llm", "scripted")
    assert code == EXIT_USAGE
    assert "live Qwen only" in text


def test_a_script_file_drives_a_keyless_rehearsal(tmp_path: Path) -> None:
    clock, root, parts = _small_simulated_record(tmp_path)
    script = tmp_path / "decisions.json"
    script.write_text(
        json.dumps(
            [
                {
                    "stance": "flat_with_reasons",
                    "targets": [target("NVDAUSDT", 0.0), target("BTCUSDT", 0.0)],
                    "rejected_alternatives": [],
                    "mandate_response": "declined: nothing is stretched",
                    "flat_reasons": ["the owner asked for a review and nothing is stretched"],
                    "summary": "flat with reasons",
                }
            ]
        ),
        "utf-8",
    )
    code, text = cli(
        root,
        clock,
        parts,
        "decide",
        "--mode",
        "simulated",
        "--llm",
        "scripted",
        "--script",
        str(script),
        "--reason",
        "review the book",
        "--symbols",
        "NVDAUSDT",
    )
    assert code == EXIT_OK, text
    assert "stance flat_with_reasons" in text
    chain = open_chain(root, RunMode.SIMULATED, clock)
    owner = [
        e
        for e in chain.events(frozenset({EventKind.TRIGGER}))
        if e.payload["kind"] == "owner_manual"
    ]
    assert len(owner) == 1
    assert owner[0].payload["symbols"] == ["NVDAUSDT"]


def test_decide_queues_the_request_when_a_loop_is_running(tmp_path: Path) -> None:
    clock, root, parts = _small_simulated_record(tmp_path)
    app = build_app(root, RunMode.SIMULATED, llm="recorded", clock=clock, parts=parts)
    try:
        code, text = cli(
            root,
            clock,
            parts,
            "decide",
            "--mode",
            "simulated",
            "--llm",
            "recorded",
            "--reason",
            "look at BTC now",
            "--symbols",
            "BTCUSDT",
        )
    finally:
        app.close()
    assert code == EXIT_OK, text
    assert "queued for its next tick" in text
    queued = list((root / "var" / "inbox" / "simulated").glob("decide-*.json"))
    assert len(queued) == 1
    assert json.loads(queued[0].read_text("utf-8"))["symbols"] == ["BTCUSDT"]


def test_decide_refuses_symbols_outside_the_universe(tmp_path: Path) -> None:
    clock, root, parts = _small_simulated_record(tmp_path)
    code, text = cli(
        root,
        clock,
        parts,
        "decide",
        "--mode",
        "simulated",
        "--reason",
        "x",
        "--symbols",
        "ETHUSDT",
    )
    assert code == EXIT_USAGE
    assert "ETHUSDT" in text


def test_reconcile_reports_clean_against_the_venue(tmp_path: Path) -> None:
    clock, root, parts = _small_simulated_record(tmp_path)
    code, text = cli(root, clock, parts, "reconcile", "--mode", "simulated", "--full")
    assert code == EXIT_OK, text
    assert "clean" in text


def test_a_simulated_book_never_resumes_on_a_venue_that_never_saw_it(tmp_path: Path) -> None:
    clock, root, parts = _small_simulated_record(tmp_path)
    fresh_venue = dataclasses.replace(parts, venue=None)
    code, text = cli(root, clock, fresh_venue, "reconcile", "--mode", "simulated")
    assert code == EXIT_USAGE
    assert "does not survive a restart" in text
    assert "--fresh" in text


def test_verify_finds_a_one_byte_edit(tmp_path: Path) -> None:
    clock, root, parts = _small_simulated_record(tmp_path)
    code, text = cli(root, clock, parts, "verify", "--mode", "simulated")
    assert code == EXIT_OK
    assert "INTACT" in text
    path = root / "var" / "ledger" / "simulated.jsonl"
    data = bytearray(path.read_bytes())
    at = data.find(b'"kind":"decision"')
    data[at + 9] = ord("D")
    path.write_bytes(bytes(data))
    code, text = cli(root, clock, parts, "verify", "--mode", "simulated")
    assert code == EXIT_FAILED
    assert "BROKEN" in text
    assert "first break at seq" in text


def test_status_with_no_ledger_and_with_one(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    clock = ManualClock(FRIDAY_1320)
    code, text = cli(empty, clock, make_parts(clock), "status")
    assert code == EXIT_OK
    assert "no ledger yet" in text
    clock, root, parts = _small_simulated_record(tmp_path)
    code, text = cli(root, clock, parts, "status")
    assert code == EXIT_OK
    assert "mode simulated" in text
    assert "last decision dec-" in text
    assert "health: ticking" in text


def test_export_refuses_a_mode_without_a_ledger_and_publishes_one(tmp_path: Path) -> None:
    clock, root, parts = _small_simulated_record(tmp_path)
    code, text = cli(root, clock, parts, "export", "--mode", "dryrun")
    assert code == EXIT_USAGE
    code, text = cli(root, clock, parts, "export", "--mode", "simulated", "--light")
    assert code == EXIT_OK, text
    out = root / "var" / "public-simulated"
    assert (out / "ledger.jsonl").is_file()
    assert (out / "index.html").is_file()


def test_the_instance_lock_is_reported_as_exit_5(tmp_path: Path) -> None:
    clock, root, parts = _small_simulated_record(tmp_path)
    app = build_app(root, RunMode.SIMULATED, llm="recorded", clock=clock, parts=parts)
    try:
        code, text = cli(root, clock, parts, "once", "--mode", "simulated", "--llm", "recorded")
    finally:
        app.close()
    assert code == EXIT_LOCKED
    assert "another simulated instance" in text


def test_probe_toolkit_needs_the_bitget_services(tmp_path: Path) -> None:
    clock, root, parts = _small_simulated_record(tmp_path)
    code, text = cli(root, clock, parts, "probe-toolkit", "--mode", "simulated")
    assert code == EXIT_USAGE
    assert "Bitget MCP services" in text


# ================================================================================================
# The comparison runs, end to end
# ================================================================================================


def test_offline_rivals_need_two_hourly_marks(tmp_path: Path) -> None:
    """One tick after the genesis leaves only the anchor mark: no window to compare arms over."""
    clock = ManualClock(FRIDAY_1320)
    root = tmp_path / "one-mark"
    root.mkdir()
    world = FakeWorld(clock)
    venue = SimulatedVenue(market=world, clock=clock, starting_equity=Decimal("10000"))
    parts = make_parts(clock, world=world, venue=venue, oi_thresholds={"BTCUSDT": 5.0})
    code, text = cli(root, clock, parts, "genesis", "--mode", "simulated", "--commit", FAKE_COMMIT)
    assert code == EXIT_OK, text
    clock.set(FRIDAY_1320 + timedelta(minutes=5))
    run_parts = dataclasses.replace(parts, oi_thresholds=None)
    ran = cli(
        root,
        clock,
        run_parts,
        "run",
        "--mode",
        "simulated",
        "--llm",
        "recorded",
        "--max-ticks",
        "1",
        "--interval",
        "0",
    )
    assert ran[0] == EXIT_OK, ran[1]
    chain = open_chain(root, RunMode.SIMULATED, clock)
    marks = list(chain.events(frozenset({EventKind.MARK})))
    assert len(marks) == 1, "the anchor alone"
    projection = Projection.from_ledger(chain, POLICY_V1, starting_equity=Decimal("10000"))
    why = cli_module._simulator(projection, world, POLICY_V1, clock, [])
    assert isinstance(why, str)
    assert "fewer than two hourly marks" in why


def test_offline_rivals_run_on_the_agent_snapshots_and_are_published(tmp_path: Path) -> None:
    clock, root, parts = _small_simulated_record(tmp_path)
    clock.set(FRIDAY_1320 + timedelta(minutes=101))
    ran = cli(
        root,
        clock,
        parts,
        "run",
        "--mode",
        "simulated",
        "--llm",
        "recorded",
        "--max-ticks",
        "1",
        "--interval",
        "0",
    )
    assert ran[0] == EXIT_OK, ran[1]
    code, text = cli(root, clock, parts, "rivals", "--mode", "simulated", "--offline-only")
    assert code == EXIT_OK, text
    assert "estimated Qwen spend up to 0 tokens" in text
    record = json.loads(
        (root / "var" / "analysis" / "simulated" / "rivals.json").read_text("utf-8")
    )
    arm_ids = [a["spec"]["arm_id"] for a in record["arms"]]
    assert "rival_lexicon_follow" in arm_ids
    assert "rival_s2_fear_greed_confluence" in arm_ids
    assert record["estimated_tokens"] == 0
    code, text = cli(root, clock, parts, "export", "--mode", "simulated", "--light")
    assert code == EXIT_OK, text
    published = json.loads((root / "var" / "public-simulated" / "arms.json").read_text("utf-8"))
    assert set(arm_ids) <= {a["spec"]["arm_id"] for a in published}


def test_the_red_team_runs_every_vector_and_publishes_whatever_the_grade(tmp_path: Path) -> None:
    from sentiment_agent.llm.client import QwenTransportError
    from sentiment_agent.llm.fakes import FailingChatModel
    from sentiment_agent.redteam.corpus import load_vectors
    from sentiment_agent.redteam.harness import estimate_qwen_tokens

    clock, root, parts = _small_simulated_record(tmp_path)
    estimate = estimate_qwen_tokens(1, len(load_vectors()))
    failing = dataclasses.replace(parts, chat_model=FailingChatModel(QwenTransportError))
    code, text = cli(
        root, clock, failing, "redteam", "--mode", "simulated", "--approve-tokens", str(estimate)
    )
    assert code == EXIT_OK, text
    report = json.loads(
        (root / "var" / "analysis" / "simulated" / "redteam.json").read_text("utf-8")
    )
    assert {a["arm_id"] for a in report["arms"]} >= {"ours", "ours_no_quarantine"}
    assert len(report["vectors"]) == len(load_vectors())
    assert report["outcomes"]
    code, text = cli(root, clock, parts, "export", "--mode", "simulated", "--light")
    assert code == EXIT_OK, text
    published = json.loads((root / "var" / "public-simulated" / "redteam.json").read_text("utf-8"))
    assert published


def test_a_sampled_red_team_says_which_snapshots_it_attacked(tmp_path: Path) -> None:
    from sentiment_agent.llm.client import QwenTransportError
    from sentiment_agent.llm.fakes import FailingChatModel
    from sentiment_agent.redteam.corpus import load_vectors
    from sentiment_agent.redteam.harness import estimate_qwen_tokens

    clock, root, parts = _small_simulated_record(tmp_path)
    estimate = estimate_qwen_tokens(1, len(load_vectors()))
    failing = dataclasses.replace(parts, chat_model=FailingChatModel(QwenTransportError))
    code, text = cli(
        root,
        clock,
        failing,
        "redteam",
        "--mode",
        "simulated",
        "--approve-tokens",
        str(estimate),
        "--snapshots",
        "1",
    )
    assert code == EXIT_OK, text
    report = json.loads(
        (root / "var" / "analysis" / "simulated" / "redteam.json").read_text("utf-8")
    )
    assert report["snapshots_recorded"] >= 1
    # every recorded snapshot was attacked, so no sample is named
    if report["snapshots_recorded"] == 1:
        assert report["snapshots_attacked"] == []


def test_the_red_team_sample_spans_the_record() -> None:
    from sentiment_agent.runtime.cli import UsageError, even_sample

    assert even_sample(5, 1) == [4]
    assert even_sample(5, 2) == [0, 4]
    assert even_sample(5, 3) == [0, 2, 4]
    assert even_sample(5, 9) == [0, 1, 2, 3, 4]
    with pytest.raises(UsageError):
        even_sample(5, 0)


def test_a_comparison_replays_the_record_under_its_own_policy() -> None:
    from types import SimpleNamespace
    from typing import cast

    from sentiment_agent.policy import POLICY_V1, POLICY_V2
    from sentiment_agent.runtime.cli import UsageError, record_policy
    from sentiment_agent.types import PerceptionSnapshot

    def snaps(*versions: str) -> list[PerceptionSnapshot]:
        # only policy_version is read
        return [cast("PerceptionSnapshot", SimpleNamespace(policy_version=v)) for v in versions]

    assert record_policy(snaps("policy-v1", "policy-v1")) is POLICY_V1
    assert record_policy(snaps("policy-v2")) is POLICY_V2
    with pytest.raises(UsageError):
        record_policy(snaps("policy-v1", "policy-v2"))
    with pytest.raises(UsageError):
        record_policy(snaps("policy-v9"))


def test_probe_toolkit_measures_both_services_and_logs_the_probe(tmp_path: Path) -> None:
    from sources.test_toolkit import probe_facade

    clock, root, parts = _small_simulated_record(tmp_path)
    with_services = dataclasses.replace(parts, toolkit=probe_facade())
    code, text = cli(root, clock, with_services, "probe-toolkit", "--mode", "simulated")
    assert code == EXIT_OK, text
    assert "bitget_mcp_server:" in text
    assert "bitget_signal_mcp:" in text
    chain = open_chain(root, RunMode.SIMULATED, clock)
    assert len(list(chain.events(frozenset({EventKind.TOOLKIT_PROBE})))) == 1
    stored = json.loads((root / "var" / "toolkit_probe.json").read_text("utf-8"))
    assert stored["rows"]


# ================================================================================================
# go-live: the owner's one command
# ================================================================================================


def _free_disk(monkeypatch: pytest.MonkeyPatch, free: int) -> None:
    """Pretend the disk holding the project has ``free`` bytes left."""
    usage = shutil.disk_usage(Path.cwd())
    monkeypatch.setattr(shutil, "disk_usage", lambda _p: usage._replace(free=free))


@pytest.fixture
def roomy_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    """The test machine's free space is not under test; the refusal below is."""
    _free_disk(monkeypatch, 10**12)


def test_go_live_refuses_on_a_nearly_full_disk(
    root: Path, friday: ManualClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    _free_disk(monkeypatch, 10**8)
    venue = FakeDemoVenue(FakeWorld(friday))
    code, text = cli(root, friday, make_parts(friday, bgc_runner=venue), "go-live")
    assert code == EXIT_USAGE
    assert "GB free" in text
    assert venue.calls == []


def test_go_live_refuses_without_the_demo_key_and_logs_nothing(tmp_path: Path) -> None:
    root = tmp_path / "bare"
    root.mkdir()
    (root / "pyproject.toml").write_text('[project]\nname = "t2-sentiment-agent"\n', "utf-8")
    clock = ManualClock(FRIDAY_1320)
    bgc = FakeBgc()
    code, text = cli(root, clock, make_parts(clock, bgc_runner=bgc), "go-live", "--max-ticks", "1")
    assert code == EXIT_ENVIRONMENT, text
    assert bgc.calls == []
    assert open_chain(root, RunMode.PAPER, clock).head() is None
    assert open_pregenesis(root, clock).head() is None


@pytest.mark.usefixtures("roomy_disk")
def test_go_live_refuses_without_the_qwen_key(root: Path, friday: ManualClock) -> None:
    venue = FakeDemoVenue(FakeWorld(friday))
    code, text = cli(root, friday, make_parts(friday, bgc_runner=venue), "go-live")
    assert code == EXIT_USAGE
    assert "qwen.env" in text
    assert venue.calls == []


@pytest.mark.usefixtures("roomy_disk")
def test_go_live_proves_writes_the_genesis_once_runs_and_resumes(
    root: Path, friday: ManualClock
) -> None:
    world = FakeWorld(friday)
    venue = FakeDemoVenue(world)
    model = ScriptedChatModel([decision("flat_with_reasons", [], flat_reasons=["nothing yet"])])
    parts = make_parts(
        friday, world=world, toolkit=FakeToolkit(friday), bgc_runner=venue, chat_model=model
    )
    code, text = cli(root, friday, parts, "go-live", "--commit", FAKE_COMMIT, "--max-ticks", "1")
    assert code == EXIT_OK, text
    assert "1/3 environment proof" in text
    assert "#BitgetHackathon" in text, "the X post text is printed at the genesis"
    chain = open_chain(root, RunMode.PAPER, friday)
    events = list(chain.events())
    assert events[0].kind is EventKind.GENESIS
    assert EventKind.ENVIRONMENT_PROOF in {e.kind for e in events[1:]}, "re-proven by the run"
    pre = open_pregenesis(root, friday)
    assert [e.kind for e in pre.events(frozenset({EventKind.ENVIRONMENT_PROOF}))]
    assert venue.sends() == [], "no decision was due, so nothing was sent"

    friday.set(FRIDAY_1320 + timedelta(minutes=5))
    code, text = cli(root, friday, parts, "go-live", "--commit", FAKE_COMMIT, "--max-ticks", "1")
    assert code == EXIT_OK, text
    assert "resuming" in text
    geneses = list(open_chain(root, RunMode.PAPER, friday).events(frozenset({EventKind.GENESIS})))
    assert len(geneses) == 1


EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "scripted"


def test_the_runbook_rehearsal_runs_from_the_committed_examples(tmp_path: Path) -> None:
    """RUNBOOK section 0's command, exactly as published, on a fresh clone: the scripted decision
    lives under examples/, not the git-ignored var/, and it rules through every guard to a
    previewed order with nothing sent."""
    clock = ManualClock(FRIDAY_1320)
    root = tmp_path / "clone"
    root.mkdir()
    bgc = FakeBgc()
    parts = make_parts(clock, world=FakeWorld(clock), bgc_runner=bgc)
    code, text = cli(
        root,
        clock,
        parts,
        "decide",
        "--mode",
        "dryrun",
        "--llm",
        "scripted",
        "--script",
        str(EXAMPLES / "decision.json"),
        "--reason",
        "rehearsal",
        "--symbols",
        "BTCUSDT",
    )
    assert code == EXIT_OK, text
    assert "stance act" in text
    chain = open_chain(root, RunMode.DRYRUN, clock)
    rulings = [e.payload for e in chain.events(frozenset({EventKind.KERNEL_RULING}))]
    assert len(rulings) == 1
    btc = next(i for i in rulings[0]["instruments"] if i["symbol"] == "BTCUSDT")
    assert len({r["guard"] for r in btc["rulings"]}) == 11, "all eleven guards ruled"
    assert btc["approved_weight"] < 0, "the grounded short survives the kernel"
    assert list(chain.events(frozenset({EventKind.ORDER_PREVIEW})))
    assert bgc.sends() == []
    runbook = (Path(__file__).resolve().parents[2] / "RUNBOOK.md").read_text(encoding="utf-8")
    assert "--script examples/scripted/decision.json" in runbook
    assert "var/e2e" not in runbook


def test_the_flat_example_is_a_valid_rehearsal(tmp_path: Path) -> None:
    clock = ManualClock(FRIDAY_1320)
    root = tmp_path / "clone"
    root.mkdir()
    parts = make_parts(clock, world=FakeWorld(clock), bgc_runner=FakeBgc())
    code, text = cli(
        root,
        clock,
        parts,
        "decide",
        "--mode",
        "dryrun",
        "--llm",
        "scripted",
        "--script",
        str(EXAMPLES / "flat.json"),
        "--reason",
        "rehearsal",
    )
    assert code == EXIT_OK, text
    assert "stance flat_with_reasons" in text
