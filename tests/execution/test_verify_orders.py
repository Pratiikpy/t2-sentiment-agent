"""scripts/verify_orders.py: every published orderId is read back from Demo, read-only."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from execution.fakes import (
    ScriptedRunner,
    fixture,
    fixture_result,
    make_agent_hub,
    ok_result,
    verb,
    write_demo_env,
)
from helpers import make_intent
from sentiment_agent.execution.bgc import BgcTimeoutError
from sentiment_agent.execution.environment import PAPER_FLAG, READ_ONLY_FLAG, EnvironmentRefused

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "verify_orders.py"
INTENT = make_intent()
VENUE_ID = "1233319323918499840"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verify_orders_under_test", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verify = _load()


def _row(**update: str) -> dict[str, Any]:
    row: dict[str, Any] = dict(
        fixture("loopback_history_doc")["result"]["stdout"]["data"]["list"][0]
    )
    row.update(
        orderId=VENUE_ID,
        clientOid=INTENT.client_oid,
        symbol="NVDAUSDT",
        side="buy",
        orderStatus="filled",
    )
    row.update(update)
    return row


def _published(**update: Any) -> dict[str, Any]:
    order = {
        "venue_order_id": VENUE_ID,
        "client_oid": INTENT.client_oid,
        "symbol": "NVDAUSDT",
        "side": "buy",
    }
    order.update(update)
    return order


def _check(runner: ScriptedRunner, order: dict[str, Any]) -> Any:
    return verify.check_order(order, runner=runner, env={})


def test_a_published_order_that_matches_is_found() -> None:
    runner = ScriptedRunner().on(verb("order", "--action", "detail"), ok_result(_row()))
    check = _check(runner, _published())
    assert check.outcome == "FOUND"
    assert "NVDAUSDT buy" in check.detail
    (argv,) = runner.argvs()
    assert argv == (
        "order",
        "--action",
        "detail",
        "--orderId",
        VENUE_ID,
        "--view",
        "full",
        PAPER_FLAG,
    )
    assert READ_ONLY_FLAG not in argv


@pytest.mark.parametrize(
    ("update", "field"),
    [
        ({"clientOid": "sa" + "0" * 30}, "clientOid"),
        ({"symbol": "TSLAUSDT"}, "symbol"),
        ({"side": "sell"}, "side"),
    ],
)
def test_a_venue_order_that_disagrees_is_a_mismatch(update: dict[str, str], field: str) -> None:
    runner = ScriptedRunner().on(verb("order"), ok_result(_row(**update)))
    check = _check(runner, _published())
    assert check.outcome == "MISMATCH"
    assert field in check.detail


def test_missing_error_and_not_sent() -> None:
    missing = ScriptedRunner().on(verb("order"), fixture_result("loopback_detail_not_found"))
    assert _check(missing, _published()).outcome == "MISSING"
    empty = ScriptedRunner().on(verb("order"), ok_result(None))
    assert _check(empty, _published()).outcome == "MISSING"
    network = ScriptedRunner().on(verb("order"), fixture_result("loopback_place_network_error"))
    assert _check(network, _published()).outcome == "ERROR"
    slow = ScriptedRunner().on(verb("order"), BgcTimeoutError("slow"))
    assert _check(slow, _published()).outcome == "ERROR"
    never = ScriptedRunner()
    assert _check(never, _published(venue_order_id=None)).outcome == "NOT_SENT"
    assert never.calls == []
    assert _check(never, _published(venue_order_id="--read-only")).outcome == "ERROR"
    assert never.calls == []


def test_a_key_that_is_not_demo_stops_the_check() -> None:
    runner = ScriptedRunner().on(verb("order"), fixture_result("loopback_place_40099"))
    with pytest.raises(EnvironmentRefused):
        _check(runner, _published())


def test_exit_codes() -> None:
    found = ScriptedRunner().on(verb("order"), ok_result(_row()))
    assert verify.run([_published()], runner=found, env={})[1] == 0
    missing = ScriptedRunner().on(verb("order"), fixture_result("loopback_detail_not_found"))
    assert verify.run([_published()], runner=missing, env={})[1] == 1
    broken = ScriptedRunner().on(verb("order"), fixture_result("loopback_place_network_error"))
    assert verify.run([_published()], runner=broken, env={})[1] == 2


def test_orders_file_formats(tmp_path: Path) -> None:
    listing = tmp_path / "list.json"
    listing.write_text(json.dumps([_published()]), encoding="utf-8")
    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(json.dumps({"orders": [_published()]}), encoding="utf-8")
    assert verify.load_orders(listing) == verify.load_orders(wrapped) == [_published()]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"rows": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="orders"):
        verify.load_orders(bad)


def _project(tmp_path: Path, *, orders: list[dict[str, Any]]) -> tuple[Path, Path]:
    root = tmp_path / "project"
    write_demo_env(root)
    make_agent_hub(root)
    public = root / "public" / "orders.json"
    public.parent.mkdir(parents=True)
    public.write_text(json.dumps(orders), encoding="utf-8")
    return root, public


def test_main_end_to_end_with_a_fake_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root, public = _project(tmp_path, orders=[_published(), _published(venue_order_id=None)])
    runner = ScriptedRunner().on(verb("order"), ok_result(_row()))
    monkeypatch.setattr(verify, "SubprocessBgcRunner", lambda _hub: runner)
    code = verify.main(["--public", str(public), "--root", str(root), "--json"])
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert [c["outcome"] for c in report["checks"]] == ["FOUND", "NOT_SENT"]
    ((_argv, env, _timeout),) = runner.calls
    assert env["BITGET_API_KEY"] == "bg_demo_fixture_key_0001"


def test_main_refuses_to_run_without_what_it_needs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, public = _project(tmp_path, orders=[_published()])
    runner = ScriptedRunner()
    monkeypatch.setattr(verify, "SubprocessBgcRunner", lambda _hub: runner)
    assert verify.main(["--public", str(root / "absent.json"), "--root", str(root)]) == 2
    (root / ".secrets" / "demo.env").unlink()
    assert verify.main(["--public", str(public), "--root", str(root)]) == 2
    write_demo_env(root)
    make_agent_hub(root, header=False)
    assert verify.main(["--public", str(public), "--root", str(root)]) == 2
    assert runner.calls == []


def test_main_exits_3_on_40099(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root, public = _project(tmp_path, orders=[_published()])
    runner = ScriptedRunner().on(verb("order"), fixture_result("loopback_place_40099"))
    monkeypatch.setattr(verify, "SubprocessBgcRunner", lambda _hub: runner)
    assert verify.main(["--public", str(public), "--root", str(root)]) == 3
    assert "not a Demo key" in capsys.readouterr().err
