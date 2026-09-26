"""``scripts/switchover_check.py``: run 2's genesis against this code's declaration, its first
publish, and run 1 left exactly as it closed."""

import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import pytest

from sentiment_agent.clock import ManualClock
from sentiment_agent.hashing import content_hash
from sentiment_agent.ledger.chain import HashChainLedger, ledger_path
from sentiment_agent.ledger.genesis import build_genesis, write_genesis
from sentiment_agent.policy import ACTIVE_POLICY, POLICY_V1
from sentiment_agent.run2 import DECLARED_CHANGES
from sentiment_agent.types import EventKind, Note, PredecessorRun, RunMode

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "switchover_check.py"
AT = datetime(2026, 9, 28, 0, 0, tzinfo=UTC)


@pytest.fixture
def check() -> ModuleType:
    spec = importlib.util.spec_from_file_location("switchover_check", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclass resolves annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def _paper(root: Path) -> HashChainLedger:
    path = ledger_path(root, RunMode.PAPER)
    path.parent.mkdir(parents=True)
    return HashChainLedger(path, mode=RunMode.PAPER, clock=ManualClock(AT))


def _genesis(root: Path, policy: object, **extra: object) -> str:
    chain = _paper(root)
    genesis = build_genesis(
        policy=policy,  # type: ignore[arg-type]
        prompt_hashes={"prompts/system_v1.md": content_hash("prompt")},
        mode=RunMode.PAPER,
        code_commit="ab" * 20,
        lock_hashes={"uv.lock": content_hash("lock")},
        bgc_package="@bitget-ai/bitget-agent-cli@3.0.0",
        clock=ManualClock(AT),
        **extra,  # type: ignore[arg-type]
    )
    return write_genesis(chain, genesis).hash


def _two_runs(
    tmp_path: Path, check: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    run1, run2 = tmp_path / "run1", tmp_path / "run2"
    run1_hash = _genesis(run1, POLICY_V1)
    predecessor = PredecessorRun(
        genesis_hash=run1_hash,
        code_commit="cd" * 20,
        policy_hash=POLICY_V1.content_hash(),
        policy_version=POLICY_V1.version,
        window="72 hours",
    )
    monkeypatch.setattr(check, "PREDECESSOR", predecessor)
    _genesis(run2, ACTIVE_POLICY, predecessor=predecessor, declared_changes=DECLARED_CHANGES)
    return run1, run2


def test_nothing_started_is_pending_not_failed(tmp_path: Path, check: ModuleType) -> None:
    run1 = tmp_path / "run1"
    _genesis(run1, POLICY_V1)
    states = {c.name: c.state for c in check.run(run1, tmp_path / "run2", None)}
    assert states == {
        "run 2 genesis": "PENDING",
        "first hourly publish": "PENDING",
        "run 1 untouched": "PENDING",
    }


def test_a_matching_genesis_passes_its_declaration_checks(
    tmp_path: Path, check: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    run1, run2 = _two_runs(tmp_path, check, monkeypatch)
    states = {c.name: c.state for c in check.run(run1, run2, None)}
    assert states["policy in force"] == "PASS"
    assert states["predecessor"] == "PASS"
    assert states["declared changes"] == "PASS"


def test_a_genesis_naming_another_run_fails(
    tmp_path: Path, check: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    run1, run2 = _two_runs(tmp_path, check, monkeypatch)
    monkeypatch.setattr(
        check, "PREDECESSOR", check.PREDECESSOR.model_copy(update={"genesis_hash": "0" * 64})
    )
    states = {c.name: c.state for c in check.run(run1, run2, None)}
    assert states["predecessor"] == "FAIL"


def test_the_first_publish_must_be_aliased(tmp_path: Path, check: ModuleType) -> None:
    logs = tmp_path / "var" / "logs"
    logs.mkdir(parents=True)
    log = logs / "site.log"
    log.write_text("2026-09-28T01:10:05Z publish FAILED for record generated x -- 503\n", "utf-8")
    assert check.check_first_publish(tmp_path).state == "FAIL"
    log.write_text(
        log.read_text("utf-8")
        + "2026-09-28T02:12:00Z published record generated y Aliased https://x.example\n",
        "utf-8",
    )
    assert check.check_first_publish(tmp_path).state == "PASS"


def test_run_1_moved_after_it_closed_fails(tmp_path: Path, check: ModuleType) -> None:
    run1 = tmp_path / "run1"
    _genesis(run1, POLICY_V1)
    chain = HashChainLedger(
        ledger_path(run1, RunMode.PAPER), mode=RunMode.PAPER, clock=ManualClock(AT)
    )
    closed = chain.head()
    assert closed is not None
    assert check.check_run1_untouched(run1, closed.hash).state == "PASS"
    chain.append(EventKind.NOTE, Note(at=AT, author="system", text="written after the close"))
    assert check.check_run1_untouched(run1, closed.hash).state == "FAIL"
