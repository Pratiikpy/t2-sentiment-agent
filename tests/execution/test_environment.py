"""Demo credentials, credential isolation, the vendor contract and the environment proof."""

import os
import pickle
from collections.abc import Sequence
from pathlib import Path

import pytest

from execution.fakes import (
    DEMO_KEY,
    DEMO_PASSPHRASE,
    DEMO_SECRET,
    MemoryBlobStore,
    ScriptedRunner,
    both,
    fixture,
    fixture_result,
    has,
    make_agent_hub,
    verb,
    write_demo_env,
)
from sentiment_agent.clock import ManualClock
from sentiment_agent.execution.bgc import BGC_PACKAGE, BgcResult, BgcTimeoutError
from sentiment_agent.execution.environment import (
    CLI_PACKAGE,
    DEMO_ENV_FILE,
    LIVE_NEGATIVE_ARGS,
    PAPER_FLAG,
    PINNED_VERSION,
    READ_ONLY_FLAG,
    DemoCredentials,
    EnvironmentRefused,
    base_child_env,
    confirm_paptrading_header,
    failure_of,
    is_key_refusal,
    load_demo_credentials,
    mentions_environment_mismatch,
    prove_environment,
    vendor_contract,
)
from sentiment_agent.types import RunMode

INSTALLED_HUB = Path(__file__).resolve().parents[2] / "tools" / "agent-hub"


# --- credentials ----------------------------------------------------------------------------


def test_loads_a_declared_demo_key(workdir: Path) -> None:
    write_demo_env(workdir)
    creds = load_demo_credentials(workdir)
    env = creds.child_env()
    assert env["BITGET_API_KEY"] == DEMO_KEY
    assert env["BITGET_SECRET_KEY"] == DEMO_SECRET
    assert env["BITGET_PASSPHRASE"] == DEMO_PASSPHRASE
    assert creds.source == DEMO_ENV_FILE


def test_missing_file_is_refused(workdir: Path) -> None:
    with pytest.raises(EnvironmentRefused, match="does not exist"):
        load_demo_credentials(workdir)


@pytest.mark.parametrize("declaration", [None, "live", "", "production", "paper-ish"])
def test_a_key_not_declared_demo_is_refused(workdir: Path, declaration: str | None) -> None:
    write_demo_env(workdir, declaration=declaration)
    with pytest.raises(EnvironmentRefused, match="BITGET_KEY_ENVIRONMENT=demo"):
        load_demo_credentials(workdir)


def test_declaration_is_case_insensitive_and_quotes_and_export_are_accepted(
    workdir: Path,
) -> None:
    path = workdir / ".secrets" / "demo.env"
    path.write_text(
        "﻿# comment\n"
        "export BITGET_KEY_ENVIRONMENT='Demo'\n"
        f'BITGET_API_KEY="{DEMO_KEY}"\n'
        f"BITGET_SECRET_KEY={DEMO_SECRET}\n"
        "BITGET_PASSPHRASE=pa#ss\n",
        encoding="utf-8",
    )
    creds = load_demo_credentials(workdir)
    assert creds.child_env()["BITGET_API_KEY"] == DEMO_KEY
    assert creds.child_env()["BITGET_PASSPHRASE"] == "pa#ss"


@pytest.mark.parametrize("missing", ["key", "secret", "passphrase"])
def test_a_missing_value_is_refused_by_name(workdir: Path, missing: str) -> None:
    write_demo_env(workdir, **{missing: None})  # type: ignore[arg-type]
    with pytest.raises(EnvironmentRefused, match="missing BITGET_"):
        load_demo_credentials(workdir)


def test_malformed_and_repeated_lines_are_refused_without_echoing_values(workdir: Path) -> None:
    write_demo_env(workdir, extra="this line has no equals sign\n")
    with pytest.raises(EnvironmentRefused, match="line 6 is not KEY=VALUE") as caught:
        load_demo_credentials(workdir)
    assert DEMO_SECRET not in str(caught.value)
    write_demo_env(workdir, extra=f"BITGET_SECRET_KEY={DEMO_SECRET}\n")
    with pytest.raises(EnvironmentRefused, match="more than once") as caught:
        load_demo_credentials(workdir)
    assert DEMO_SECRET not in str(caught.value)


def test_a_credential_file_outside_the_project_root_is_refused(
    tmp_path: Path, workdir: Path
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    real = write_demo_env(outside)
    link = workdir / ".secrets" / "demo.env"
    try:
        os.symlink(real, link)
    except OSError:
        pytest.skip("this platform does not let the test create a symlink")
    with pytest.raises(EnvironmentRefused, match="outside the project root"):
        load_demo_credentials(workdir)


def test_repr_str_and_pickle_never_reveal_the_secret(workdir: Path) -> None:
    write_demo_env(workdir)
    creds = load_demo_credentials(workdir)
    for text in (repr(creds), str(creds), f"{creds}", f"{creds!r}"):
        for secret in (DEMO_KEY, DEMO_SECRET, DEMO_PASSPHRASE):
            assert secret not in text
    assert creds.fingerprint in repr(creds)
    with pytest.raises(TypeError):
        pickle.dumps(creds)


def test_whitespace_in_a_value_is_refused() -> None:
    with pytest.raises(EnvironmentRefused, match="whitespace"):
        DemoCredentials(api_key="a b", secret_key="s", passphrase="p")


def test_parent_bitget_variables_never_reach_the_child(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BITGET_API_KEY", "LIVE-KEY-FROM-THE-PARENT-SHELL")
    monkeypatch.setenv("BITGET_API_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("BITGET_MAX_RETRIES", "9")
    monkeypatch.setenv("SOME_OTHER_VAR", "x")
    write_demo_env(workdir)
    env = load_demo_credentials(workdir).child_env()
    assert env["BITGET_API_KEY"] == DEMO_KEY
    assert "BITGET_API_BASE_URL" not in env
    assert "BITGET_MAX_RETRIES" not in env
    assert "SOME_OTHER_VAR" not in env
    assert set(env) <= {
        "PATH",
        "SYSTEMROOT",
        "BITGET_API_KEY",
        "BITGET_SECRET_KEY",
        "BITGET_PASSPHRASE",
    }
    bare = base_child_env()
    assert not [k for k in bare if k.startswith("BITGET_")]
    assert "PATH" in bare


# --- the vendor contract ---------------------------------------------------------------------


def test_the_installed_agent_hub_meets_the_contract() -> None:
    if not (INSTALLED_HUB / "node_modules").is_dir():
        pytest.skip("tools/agent-hub is not installed (npm ci --ignore-scripts)")
    contract = vendor_contract(INSTALLED_HUB)
    assert contract.ok, contract.findings
    assert confirm_paptrading_header(INSTALLED_HUB)
    assert contract.facts["cli_version"] == PINNED_VERSION
    assert len(contract.facts["lockfile_sha256"]) == 64


def test_bgc_package_is_the_pinned_cli() -> None:
    assert BGC_PACKAGE == f"{CLI_PACKAGE}@{PINNED_VERSION}" == "@bitget-ai/bitget-agent-cli@3.0.0"


def test_a_fake_hub_that_meets_the_contract_passes(tmp_path: Path) -> None:
    assert confirm_paptrading_header(make_agent_hub(tmp_path))


def test_an_sdk_without_the_paper_header_fails_the_contract(tmp_path: Path) -> None:
    hub = make_agent_hub(tmp_path, header=False)
    assert not confirm_paptrading_header(hub)
    assert any("paptrading" in f for f in vendor_contract(hub).findings)


@pytest.mark.parametrize(
    "change", [{"cli_version": "3.1.0"}, {"sdk_version": "3.3.1"}, {"locked_version": "3.3.1"}]
)
def test_any_other_version_fails_the_contract(tmp_path: Path, change: dict[str, str]) -> None:
    assert not confirm_paptrading_header(make_agent_hub(tmp_path, **change))  # type: ignore[arg-type]


def test_an_empty_directory_fails_the_contract(tmp_path: Path) -> None:
    contract = vendor_contract(tmp_path)
    assert not contract.ok
    assert contract.facts["cli_version"] == "missing"


# --- reading bgc's errors ---------------------------------------------------------------------


def test_errors_arrive_with_the_http_status_as_code() -> None:
    failure = failure_of(fixture_result("loopback_live_negative_40006"))
    assert failure is not None
    assert (failure.type, failure.code) == ("BitgetApiError", "400")
    assert "Invalid ACCESS_KEY" in failure.message
    assert is_key_refusal(failure)
    assert not failure.environment_mismatch


def test_40099_is_recognised_by_its_message_where_the_code_is_lost() -> None:
    failure = failure_of(fixture_result("loopback_live_negative_40099"))
    assert failure is not None
    assert failure.code == "400"
    assert failure.environment_mismatch
    overview = fixture_result("loopback_demo_overview_40099")
    assert overview.exit_code == 0
    assert mentions_environment_mismatch(overview.stdout)
    assert not mentions_environment_mismatch(fixture_result("loopback_demo_overview_ok").stdout)
    assert mentions_environment_mismatch({"code": "40099"})


def test_a_parameter_error_is_not_a_key_refusal() -> None:
    failure = failure_of(fixture_result("loopback_live_negative_param"))
    assert failure is not None
    assert not is_key_refusal(failure)


def test_http_200_auth_codes_arrive_as_authentication_errors() -> None:
    failure = failure_of(fixture_result("loopback_detail_auth_200"))
    assert failure is not None
    assert failure.type == "AuthenticationError"
    assert failure.code is None
    assert is_key_refusal(failure)


def test_network_errors_are_transient() -> None:
    failure = failure_of(fixture_result("loopback_place_network_error"))
    assert failure is not None
    assert failure.transient
    assert not is_key_refusal(failure)


def test_plain_text_and_unreadable_errors() -> None:
    text = failure_of(
        BgcResult(exit_code=1, stdout=None, stderr={"text": "Error: x"}, duration_ms=1)
    )
    assert text is not None
    assert text.type == "CliError"
    assert text.local
    empty = failure_of(BgcResult(exit_code=1, stdout=None, stderr=None, duration_ms=1))
    assert empty is not None
    assert empty.type == "Unreadable"


# --- the proof --------------------------------------------------------------------------------


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    write_demo_env(root)
    make_agent_hub(root)
    return root


def _runner(*, overview: str, assets: str, live: str | BaseException) -> ScriptedRunner:
    runner = ScriptedRunner()
    runner.on(both(verb("account_overview"), has(PAPER_FLAG)), fixture_result(overview))
    runner.on(both(verb("raw"), has(PAPER_FLAG)), fixture_result(assets))
    live_answer = live if isinstance(live, BaseException) else fixture_result(live)
    runner.on(both(verb("raw"), has(READ_ONLY_FLAG)), live_answer)
    return runner


def _prove(
    root: Path, runner: ScriptedRunner, clock: ManualClock
) -> tuple[object, MemoryBlobStore]:
    blobs = MemoryBlobStore()
    return prove_environment(root, runner=runner, clock=clock, blobs=blobs), blobs


def test_proof_passes_when_demo_accepts_and_live_refuses(
    tmp_path: Path, clock: ManualClock
) -> None:
    root = _project(tmp_path)
    runner = _runner(
        overview="loopback_demo_overview_ok",
        assets="loopback_demo_assets_ok",
        live="loopback_live_negative_40006",
    )
    blobs = MemoryBlobStore()
    proof = prove_environment(root, runner=runner, clock=clock, blobs=blobs)
    assert proof.passed, proof.reasons
    assert proof.mode is RunMode.PAPER
    assert proof.key_declared_demo
    assert proof.paptrading_header_confirmed
    assert proof.demo_read_ok
    assert proof.demo_read_code == "00000"
    assert proof.live_read_rejected
    assert proof.live_read_code == "400"
    assert proof.hold_mode == "one_way_mode"
    assert proof.account is not None
    # The USDT row's equity, not usdtEquity (11.13921165) — see account_from_assets.
    assert str(proof.account.equity_usdt) == "6.19300826"
    assert proof.credentials_file == ".secrets/demo.env"
    assert proof.bgc_package == BGC_PACKAGE
    assert (
        proof.detail["key_fingerprint"]
        == DemoCredentials(
            api_key=DEMO_KEY, secret_key=DEMO_SECRET, passphrase=DEMO_PASSPHRASE
        ).fingerprint
    )
    assert "Invalid ACCESS_KEY" in proof.detail["live_read_message"]
    # Every recorded blob is evidence without a secret or the account's identity in it.
    assert len(blobs.data) == 3
    for data in blobs.data.values():
        for secret in (DEMO_KEY, DEMO_SECRET, DEMO_PASSPHRASE):
            assert secret.encode() not in data
        assert b"1111111111" not in data  # the documented settings answer's uid


def test_the_only_argv_without_paper_trading_is_the_read_only_live_probe(
    tmp_path: Path, clock: ManualClock
) -> None:
    runner = _runner(
        overview="loopback_demo_overview_ok",
        assets="loopback_demo_assets_ok",
        live="loopback_live_negative_40006",
    )
    prove_environment(_project(tmp_path), runner=runner, clock=clock, blobs=MemoryBlobStore())
    non_paper = [a for a in runner.argvs() if PAPER_FLAG not in a]
    assert non_paper == [LIVE_NEGATIVE_ARGS]
    assert READ_ONLY_FLAG in non_paper[0]
    assert "getAccountAssets" in non_paper[0]
    assert fixture("loopback_live_negative_40006")["requests"][0]["method"] == "GET"
    for argv in runner.argvs():
        assert (PAPER_FLAG in argv) != (READ_ONLY_FLAG in argv)
    for _args, env, _timeout in runner.calls:
        assert env["BITGET_API_KEY"] == DEMO_KEY


def test_live_40099_counts_as_the_live_venue_refusing_the_key(
    tmp_path: Path, clock: ManualClock
) -> None:
    runner = _runner(
        overview="loopback_demo_overview_ok",
        assets="loopback_demo_assets_ok",
        live="loopback_live_negative_40099",
    )
    proof = prove_environment(
        _project(tmp_path), runner=runner, clock=clock, blobs=MemoryBlobStore()
    )
    assert proof.passed
    assert proof.live_read_code == "40099"


@pytest.mark.parametrize(
    ("overview", "assets"),
    [
        ("loopback_demo_overview_40099", "loopback_demo_assets_ok"),
        ("loopback_demo_overview_ok", "loopback_demo_assets_40099"),
    ],
)
def test_40099_on_a_demo_read_refuses_with_the_failed_proof(
    tmp_path: Path, clock: ManualClock, overview: str, assets: str
) -> None:
    runner = _runner(overview=overview, assets=assets, live="loopback_live_negative_40006")
    with pytest.raises(EnvironmentRefused, match="not a Demo key") as caught:
        prove_environment(_project(tmp_path), runner=runner, clock=clock, blobs=MemoryBlobStore())
    assert caught.value.proof is not None
    assert not caught.value.proof.passed
    assert caught.value.proof.demo_read_code == "40099"
    assert caught.value.evidence is not None
    assert not [a for a in runner.argvs() if READ_ONLY_FLAG in a]


def test_a_key_the_live_venue_accepts_is_refused(tmp_path: Path, clock: ManualClock) -> None:
    runner = _runner(
        overview="loopback_demo_overview_ok",
        assets="loopback_demo_assets_ok",
        live="loopback_live_negative_accepted",
    )
    with pytest.raises(EnvironmentRefused, match="ACCEPTED") as caught:
        prove_environment(_project(tmp_path), runner=runner, clock=clock, blobs=MemoryBlobStore())
    assert caught.value.proof is not None
    assert caught.value.proof.live_read_code == "00000"
    assert not caught.value.proof.passed


@pytest.mark.parametrize(
    "live",
    [
        "loopback_live_negative_param",
        "loopback_place_network_error",
        BgcTimeoutError("did not finish"),
        OSError("node vanished"),
    ],
)
def test_an_inconclusive_live_probe_fails_closed(
    tmp_path: Path, clock: ManualClock, live: str | BaseException
) -> None:
    runner = _runner(
        overview="loopback_demo_overview_ok", assets="loopback_demo_assets_ok", live=live
    )
    proof = prove_environment(
        _project(tmp_path), runner=runner, clock=clock, blobs=MemoryBlobStore()
    )
    assert not proof.passed
    assert not proof.live_read_rejected
    assert any("inconclusive" in r for r in proof.reasons)


def test_a_failed_demo_section_fails_the_proof(tmp_path: Path, clock: ManualClock) -> None:
    runner = _runner(
        overview="local_overview_no_credential",
        assets="loopback_demo_assets_ok",
        live="loopback_live_negative_40006",
    )
    proof = prove_environment(
        _project(tmp_path), runner=runner, clock=clock, blobs=MemoryBlobStore()
    )
    assert not proof.passed
    assert not proof.demo_read_ok
    assert any("Demo assets read failed" in r for r in proof.reasons)


def test_an_unreadable_hold_mode_fails_the_proof(tmp_path: Path, clock: ManualClock) -> None:
    ok = fixture_result("loopback_demo_overview_ok")
    assert ok.stdout is not None
    stdout = dict(ok.stdout)
    data = dict(stdout["data"])
    settings = dict(data["settings"])
    settings["data"] = {**settings["data"], "holdMode": ""}
    data["settings"] = settings
    stdout["data"] = data
    runner = ScriptedRunner()
    runner.on(verb("account_overview"), BgcResult(0, stdout, None, 1))
    runner.on(both(verb("raw"), has(PAPER_FLAG)), fixture_result("loopback_demo_assets_ok"))
    runner.on(has(READ_ONLY_FLAG), fixture_result("loopback_live_negative_40006"))
    proof = prove_environment(
        _project(tmp_path), runner=runner, clock=clock, blobs=MemoryBlobStore()
    )
    assert not proof.passed
    assert proof.hold_mode is None
    assert any("hold mode" in r for r in proof.reasons)


def test_no_credential_means_no_probe(tmp_path: Path, clock: ManualClock) -> None:
    root = tmp_path / "project"
    make_agent_hub(root)
    runner = ScriptedRunner()
    proof = prove_environment(root, runner=runner, clock=clock, blobs=MemoryBlobStore())
    assert not proof.passed
    assert not proof.key_declared_demo
    assert runner.calls == []


def test_the_key_is_never_handed_to_an_unverified_cli(tmp_path: Path, clock: ManualClock) -> None:
    root = tmp_path / "project"
    write_demo_env(root)
    make_agent_hub(root, header=False)
    runner = ScriptedRunner()
    proof = prove_environment(root, runner=runner, clock=clock, blobs=MemoryBlobStore())
    assert not proof.passed
    assert not proof.paptrading_header_confirmed
    assert runner.calls == []


def _argv_set(argvs: Sequence[tuple[str, ...]]) -> set[str]:
    return {a[0] for a in argvs}


def test_the_proof_probes_three_reads_and_nothing_else(tmp_path: Path, clock: ManualClock) -> None:
    runner = _runner(
        overview="loopback_demo_overview_ok",
        assets="loopback_demo_assets_ok",
        live="loopback_live_negative_40006",
    )
    prove_environment(_project(tmp_path), runner=runner, clock=clock, blobs=MemoryBlobStore())
    assert len(runner.calls) == 3
    assert _argv_set(runner.argvs()) == {"account_overview", "raw"}
    assert not [a for a in runner.argvs() if "place" in a or "cancel" in a]


def _overview_with_failed(section: str) -> BgcResult:
    ok = fixture_result("loopback_demo_overview_ok")
    assert ok.stdout is not None
    stdout = dict(ok.stdout)
    data = dict(stdout["data"])
    data[section] = {"ok": False, "error": "HTTP 404 from Bitget: Request URL NOT FOUND"}
    stdout["data"] = data
    return BgcResult(0, stdout, None, 1)


@pytest.mark.parametrize(("section", "passes"), [("fundingAssets", True), ("positions", False)])
def test_only_the_sections_trading_needs_decide_the_proof(
    tmp_path: Path, clock: ManualClock, section: str, passes: bool
) -> None:
    runner = ScriptedRunner()
    runner.on(verb("account_overview"), _overview_with_failed(section))
    runner.on(both(verb("raw"), has(PAPER_FLAG)), fixture_result("loopback_demo_assets_ok"))
    runner.on(has(READ_ONLY_FLAG), fixture_result("loopback_live_negative_40006"))
    proof = prove_environment(
        _project(tmp_path), runner=runner, clock=clock, blobs=MemoryBlobStore()
    )
    assert proof.passed is passes
    if passes:
        assert "NOT FOUND" in proof.detail["fundingAssets_section"]
    else:
        assert any("positions read failed" in r for r in proof.reasons)
