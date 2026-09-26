"""The hourly export carries the analysis (2026-09-26): run 1's page showed the book alone for five
hours in every six, because only one export in six computed the twin, the baselines and the
coin-flip distribution. Every hourly export is now a full one; a failed analysis still publishes the
record and says why the arms are missing, and an absent red-team run says why it is absent."""

import io
import json
from pathlib import Path
from typing import Any

import pytest

from runtime.test_cli import _small_simulated_record, cli
from sentiment_agent.runtime import cli as cli_module
from sentiment_agent.runtime.cli import (
    EXIT_OK,
    LIGHT_EXPORT_REASON,
    REDTEAM_NOT_RUN_REASON,
    BackgroundExporter,
    Context,
)
from sentiment_agent.types import RunMode


def _published(root: Path, name: str) -> Any:
    return json.loads((root / "var" / "public-simulated" / name).read_text("utf-8"))


def test_a_failed_analysis_still_publishes_and_says_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock, root, parts = _small_simulated_record(tmp_path)

    def unreadable(*args: object, **kwargs: object) -> object:
        raise RuntimeError("the candle service answered 503")

    monkeypatch.setattr(cli_module, "full_analysis", unreadable)
    code, text = cli(root, clock, parts, "export", "--mode", "simulated", "--coin-flips", "5")
    assert code == EXIT_OK, text
    twin = _published(root, "twin.json")
    assert twin["status"] == "not_computed"
    assert twin["reason"].startswith(
        "the analysis failed on this export (RuntimeError: the candle service answered 503)"
    )
    assert _published(root, "summary.json")["ledger"]["head_hash"]


def test_a_light_export_says_it_skipped_the_analysis(tmp_path: Path) -> None:
    clock, root, parts = _small_simulated_record(tmp_path)
    code, text = cli(root, clock, parts, "export", "--mode", "simulated", "--light")
    assert code == EXIT_OK, text
    assert _published(root, "twin.json")["reason"] == LIGHT_EXPORT_REASON
    assert _published(root, "redteam.json")["reason"] == REDTEAM_NOT_RUN_REASON


def test_the_hourly_exporter_always_runs_the_full_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock, root, parts = _small_simulated_record(tmp_path)
    calls: list[bool] = []

    def record(*args: object, light: bool, **kwargs: object) -> tuple[object, list[str]]:
        calls.append(light)
        return object(), []

    monkeypatch.setattr(cli_module, "export_record", record)
    ctx = Context(
        root=root,
        clock=clock,
        out=io.StringIO(),
        parts=parts,
        command_runner=lambda *a: (0, "", ""),
    )
    exporter = BackgroundExporter(ctx, coin_flips=5)
    for _ in range(3):
        exporter._run(root, RunMode.SIMULATED, clock)
    assert calls == [False, False, False]


def test_a_full_export_carries_the_model_free_rivals(tmp_path: Path) -> None:
    clock, root, parts = _small_simulated_record(tmp_path)
    code, text = cli(root, clock, parts, "export", "--mode", "simulated", "--coin-flips", "5")
    assert code == EXIT_OK, text
    arm_ids = {a["spec"]["arm_id"] for a in _published(root, "arms.json")}
    assert {
        "rival_lexicon_follow",
        "rival_lexicon_fade",
        "rival_s2_fear_greed_confluence",
    } <= arm_ids
    assert _published(root, "twin.json")["status"] == "computed"


def test_a_stored_rival_run_does_not_duplicate_an_hourly_arm(tmp_path: Path) -> None:
    clock, root, parts = _small_simulated_record(tmp_path)
    code, text = cli(root, clock, parts, "rivals", "--mode", "simulated", "--offline-only")
    assert code == EXIT_OK, text
    code, text = cli(root, clock, parts, "export", "--mode", "simulated", "--coin-flips", "5")
    assert code == EXIT_OK, text
    arm_ids = [a["spec"]["arm_id"] for a in _published(root, "arms.json")]
    assert len(arm_ids) == len(set(arm_ids))
    assert "rival_lexicon_follow" in arm_ids
