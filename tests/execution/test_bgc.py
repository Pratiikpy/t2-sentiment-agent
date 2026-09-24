"""The Agent Hub transport: argv invariants, credential isolation, answers as bgc prints them."""

import itertools
import json
import sys
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from execution.fakes import (
    DEMO_KEY,
    DEMO_PASSPHRASE,
    DEMO_SECRET,
    MemoryBlobStore,
    ScriptedRunner,
    fixture,
    fixture_result,
    has,
    ok_result,
    verb,
)
from helpers import T0, make_intent, mint_for_test
from sentiment_agent.clock import ManualClock
from sentiment_agent.execution.bgc import (
    HOLD_HEDGE,
    HOLD_ONE_WAY,
    PAGE_LIMIT,
    BgcResult,
    BgcTimeoutError,
    BgcTransport,
    BgcUnavailableError,
    DryRunOnlyError,
    PreviewMismatchError,
    SubprocessBgcRunner,
    TransportRefusedError,
    VenueReadError,
    VenueWriteError,
    build_place_args,
    cancel_stop_args,
    fmt_decimal,
    parse_output,
    place_stop_args,
    position_side,
    would_send,
)
from sentiment_agent.execution.environment import (
    DRY_RUN_FLAG,
    PAPER_FLAG,
    READ_ONLY_FLAG,
    DemoCredentials,
    EnvironmentRefused,
    base_child_env,
)
from sentiment_agent.types import (
    AccountSnapshot,
    EnvironmentProof,
    OrderPurpose,
    RunMode,
    Side,
    VenueAck,
    VenueOrderStatus,
    VenueRejection,
    VenueUnknown,
)

CREDS = DemoCredentials(api_key=DEMO_KEY, secret_key=DEMO_SECRET, passphrase=DEMO_PASSPHRASE)
OPEN_BUY = make_intent()  # NVDAUSDT buy 1.00, the intent the dry-run fixtures were captured for
CLOSE_SELL = make_intent(side=Side.SELL, purpose=OrderPurpose.CLOSE)


def passed_proof(
    *, hold_mode: str = HOLD_ONE_WAY, fingerprint: str | None = None
) -> EnvironmentProof:
    return EnvironmentProof(
        checked_at=T0,
        mode=RunMode.PAPER,
        credentials_file=".secrets/demo.env",
        key_declared_demo=True,
        bgc_package="@bitget-ai/bitget-agent-cli@3.0.0",
        paptrading_header_confirmed=True,
        demo_read_ok=True,
        demo_read_code="00000",
        live_read_rejected=True,
        live_read_code="400",
        hold_mode=hold_mode,
        account=AccountSnapshot(
            at=T0, equity_usdt=Decimal("10000"), available_usdt=Decimal("10000"), blob=None
        ),
        passed=True,
        reasons=(),
        detail={"key_fingerprint": fingerprint or CREDS.fingerprint},
    )


def paper(runner: ScriptedRunner, clock: ManualClock, **kw: Any) -> BgcTransport:
    return BgcTransport(
        runner=runner,
        clock=clock,
        blobs=MemoryBlobStore(),
        credentials=kw.pop("credentials", CREDS),
        proof=kw.pop("proof", passed_proof()),
        dry_run_only=False,
    )


def dry(runner: ScriptedRunner, clock: ManualClock) -> BgcTransport:
    return BgcTransport(
        runner=runner,
        clock=clock,
        blobs=MemoryBlobStore(),
        credentials=None,
        proof=None,
        dry_run_only=True,
    )


# --- argv ------------------------------------------------------------------------------------


@pytest.mark.parametrize("hold_mode", [HOLD_ONE_WAY, HOLD_HEDGE])
@pytest.mark.parametrize("dry_run", [True, False])
@pytest.mark.parametrize(
    "intent",
    [
        OPEN_BUY,
        CLOSE_SELL,
        make_intent(symbol="BTCUSDT", side=Side.SELL, qty="0.01", price="112345.6"),
        make_intent(side=Side.BUY, purpose=OrderPurpose.REDUCE),
        make_intent(side=Side.SELL, purpose=OrderPurpose.INCREASE),
    ],
)
def test_place_argv_is_always_paper_and_never_read_only(
    hold_mode: str, dry_run: bool, intent: Any
) -> None:
    args = build_place_args(intent, hold_mode=hold_mode, dry_run=dry_run)
    assert args.count(PAPER_FLAG) == 1
    assert READ_ONLY_FLAG not in args
    assert (DRY_RUN_FLAG in args) is dry_run
    assert args[:3] == ["order", "--action", "place"]
    assert "--clientOid" in args
    assert intent.client_oid in args


@pytest.mark.parametrize(
    ("fixture_name", "intent", "hold_mode"),
    [
        ("dryrun_place_open_buy_one_way", OPEN_BUY, HOLD_ONE_WAY),
        ("dryrun_place_close_one_way", CLOSE_SELL, HOLD_ONE_WAY),
        ("dryrun_place_open_buy_hedge", OPEN_BUY, HOLD_HEDGE),
        ("dryrun_place_close_hedge", CLOSE_SELL, HOLD_HEDGE),
        (
            "dryrun_place_open_sell_one_way",
            make_intent(symbol="BTCUSDT", side=Side.SELL, qty="0.01", price="112345.6"),
            HOLD_ONE_WAY,
        ),
    ],
)
def test_what_we_expect_bgc_to_send_is_what_bgc_says_it_sends(
    fixture_name: str, intent: Any, hold_mode: str
) -> None:
    """The recorded bgc dry-run (no credential, nothing sent) builds exactly our request body."""
    recorded = fixture(fixture_name)
    assert recorded["argv"] == build_place_args(intent, hold_mode=hold_mode, dry_run=True)
    data = recorded["result"]["stdout"]["data"]
    assert data["dryRun"] is True
    assert data["operationId"] == "placeOrder"
    assert data["wouldSend"] == would_send(intent, hold_mode=hold_mode)


def test_the_body_bgc_posts_to_the_venue_is_the_preview() -> None:
    """Loopback capture: the POST body real bgc sent equals the dry-run's wouldSend."""
    for sent, previewed in (
        ("loopback_place_ack", "dryrun_place_open_buy_one_way"),
        ("loopback_place_close_ack", "dryrun_place_close_one_way"),
        ("loopback_place_stop_ack", "dryrun_place_stop"),
    ):
        posted = fixture(sent)["requests"][0]
        assert posted["method"] == "POST"
        assert posted["paptrading"] == "1"
        assert posted["signed"] is True
        assert posted["body"] == fixture(previewed)["result"]["stdout"]["data"]["wouldSend"]


def test_every_private_request_under_paper_trading_carried_the_paptrading_header() -> None:
    """Across every loopback capture: the header is present exactly when the argv is paper."""
    root = Path(__file__).resolve().parents[1] / "fixtures" / "execution"
    checked = 0
    for path in sorted(root.glob("loopback_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        paper_run = PAPER_FLAG in record["argv"]
        for request in record["requests"]:
            checked += 1
            assert request["signed"] is True
            assert (request["paptrading"] == "1") is paper_run, path.name
    assert checked >= 20


def test_reduce_only_in_one_way_and_pos_side_in_hedge() -> None:
    one_way = would_send(CLOSE_SELL, hold_mode=HOLD_ONE_WAY)
    assert one_way["reduceOnly"] == "yes"
    assert "posSide" not in one_way
    hedge = would_send(CLOSE_SELL, hold_mode=HOLD_HEDGE)
    assert hedge["posSide"] == "long"  # selling to close the long book
    assert "reduceOnly" not in hedge
    opening = would_send(OPEN_BUY, hold_mode=HOLD_ONE_WAY)
    assert opening["stopLoss"] == "213.9264"
    assert opening["slTriggerBy"] == "mark"
    assert opening["slOrderType"] == "market"
    assert "reduceOnly" not in opening
    assert "stopLoss" not in one_way


@pytest.mark.parametrize(
    ("side", "adds", "book"),
    [
        (Side.BUY, True, "long"),
        (Side.SELL, True, "short"),
        (Side.SELL, False, "long"),
        (Side.BUY, False, "short"),
    ],
)
def test_position_side(side: Side, adds: bool, book: str) -> None:
    assert position_side(side, adds_exposure=adds) == book


def test_unknown_hold_mode_and_bad_symbol_are_refused() -> None:
    with pytest.raises(ValueError, match="hold mode"):
        build_place_args(OPEN_BUY, hold_mode="both_ways", dry_run=True)
    bad = OPEN_BUY.model_copy(update={"symbol": "--read-only"})
    with pytest.raises(ValueError, match="symbol"):
        build_place_args(bad, hold_mode=HOLD_ONE_WAY, dry_run=True)


def test_stop_argv() -> None:
    args = place_stop_args(
        symbol="NVDAUSDT",
        pos_side="long",
        qty=Decimal("1.00"),
        stop_price=Decimal("213.93"),
        client_oid=OPEN_BUY.client_oid,
    )
    assert args == fixture("loopback_place_stop_ack")["argv"]
    assert cancel_stop_args("121211212122") == fixture("loopback_cancel_stop_ok")["argv"]
    with pytest.raises(ValueError, match="pos_side"):
        place_stop_args(
            symbol="NVDAUSDT",
            pos_side="both",
            qty=Decimal(1),
            stop_price=Decimal(1),
            client_oid=OPEN_BUY.client_oid,
        )
    with pytest.raises(ValueError, match="venue order id"):
        cancel_stop_args("--read-only")


@pytest.mark.parametrize(
    ("value", "text"),
    [
        ("1.00", "1"),
        ("0.050", "0.05"),
        ("100", "100"),
        ("213.9264", "213.9264"),
        ("1E+2", "100"),
        ("0.00000001", "0.00000001"),
    ],
)
def test_fmt_decimal(value: str, text: str) -> None:
    assert fmt_decimal(Decimal(value)) == text


@pytest.mark.parametrize("value", ["0", "-1", "NaN", "Infinity"])
def test_fmt_decimal_refuses_non_positive(value: str) -> None:
    with pytest.raises(ValueError, match="positive"):
        fmt_decimal(Decimal(value))


# --- every argv the transport can build --------------------------------------------------------


def _answer_everything() -> ScriptedRunner:
    runner = ScriptedRunner()
    runner.on(has(DRY_RUN_FLAG), fixture_result("dryrun_place_open_buy_one_way"))
    runner.on(verb("order", "--action", "place"), fixture_result("loopback_place_ack"))
    runner.on(verb("order", "--action", "detail"), fixture_result("loopback_order_detail_doc"))
    runner.on(verb("order", "--action", "fills"), fixture_result("loopback_fills_doc"))
    runner.on(verb("order", "--action", "history"), fixture_result("loopback_history_doc"))
    runner.on(verb("position"), fixture_result("loopback_positions_doc"))
    runner.on(
        verb("strategy_order", "--action", "open"), fixture_result("loopback_stop_orders_doc")
    )
    runner.on(
        verb("strategy_order", "--action", "place"), fixture_result("loopback_place_stop_ack")
    )
    runner.on(
        verb("strategy_order", "--action", "cancel"), fixture_result("loopback_cancel_stop_ok")
    )
    runner.on(verb("raw"), fixture_result("loopback_account_assets_doc"))
    return runner


def test_every_transport_argv_carries_paper_trading_and_never_read_only(
    clock: ManualClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BITGET_API_KEY", "LIVE-KEY-IN-THE-PARENT")
    runner = _answer_everything()
    transport = paper(runner, clock)
    transport.preview(OPEN_BUY)
    transport.place(mint_for_test(OPEN_BUY))
    transport.order(client_oid=OPEN_BUY.client_oid)
    transport.fills(since=T0 - timedelta(days=45), until=T0)
    transport.history(since=T0 - timedelta(days=1), until=T0)
    transport.positions()
    transport.stop_orders()
    transport.account()
    transport.place_stop(
        symbol="NVDAUSDT",
        pos_side="long",
        qty=Decimal(1),
        stop_price=Decimal("213.93"),
        client_oid=OPEN_BUY.client_oid,
    )
    transport.cancel_stop(symbol="NVDAUSDT", venue_id="121211212122")
    verbs_seen = {(a[0], a[2]) for a in runner.argvs()}
    assert len(verbs_seen) >= 9
    for args, env, _timeout in runner.calls:
        assert args.count(PAPER_FLAG) == 1, args
        assert READ_ONLY_FLAG not in args, args
        assert env.get("BITGET_API_KEY") in (None, DEMO_KEY)
        assert "LIVE-KEY-IN-THE-PARENT" not in env.values()
        assert not [
            k
            for k in env
            if k.startswith("BITGET_")
            and k
            not in {
                "BITGET_API_KEY",
                "BITGET_SECRET_KEY",
                "BITGET_PASSPHRASE",
                "BITGET_TIMEOUT_MS",
                "BITGET_MAX_RETRIES",
            }
        ]
        assert " ".join(args).find(DEMO_SECRET) == -1


def test_previews_get_no_credential_sends_get_the_demo_key_and_no_sdk_retry(
    clock: ManualClock,
) -> None:
    runner = _answer_everything()
    transport = paper(runner, clock)
    transport.preview(OPEN_BUY)
    transport.place(mint_for_test(OPEN_BUY))
    transport.positions()
    (preview_env, send_env, read_env) = (call[1] for call in runner.calls)
    assert "BITGET_API_KEY" not in preview_env
    assert send_env["BITGET_API_KEY"] == DEMO_KEY
    assert send_env["BITGET_MAX_RETRIES"] == "0"
    assert read_env["BITGET_MAX_RETRIES"] == "2"


def test_the_transport_refuses_to_run_an_argv_without_paper_trading(clock: ManualClock) -> None:
    transport = paper(ScriptedRunner(), clock)
    with pytest.raises(TransportRefusedError):
        transport._call(
            ["account_overview", "--read-only"], credentialed=True, write=False, timeout_s=1
        )
    with pytest.raises(TransportRefusedError):
        transport._call(["account_overview"], credentialed=True, write=False, timeout_s=1)


# --- construction -----------------------------------------------------------------------------


def test_paper_needs_credentials_and_a_passed_proof_for_the_same_key(clock: ManualClock) -> None:
    runner = ScriptedRunner()
    with pytest.raises(EnvironmentRefused, match="credentials"):
        paper(runner, clock, credentials=None)
    with pytest.raises(EnvironmentRefused, match="no passed environment proof"):
        paper(runner, clock, proof=None)
    failed = passed_proof().model_copy(update={"passed": False})
    with pytest.raises(EnvironmentRefused, match="no passed environment proof"):
        paper(runner, clock, proof=failed)
    with pytest.raises(EnvironmentRefused, match="different key"):
        paper(runner, clock, proof=passed_proof(fingerprint="0" * 12))
    no_hold = passed_proof().model_copy(update={"hold_mode": None})
    with pytest.raises(EnvironmentRefused, match="hold mode"):
        paper(runner, clock, proof=no_hold)
    assert paper(runner, clock, proof=passed_proof(hold_mode=HOLD_HEDGE)).hold_mode == HOLD_HEDGE


def test_a_dry_run_transport_never_holds_a_key_and_never_sends(clock: ManualClock) -> None:
    runner = _answer_everything()
    with pytest.raises(TransportRefusedError):
        BgcTransport(
            runner=runner,
            clock=clock,
            blobs=MemoryBlobStore(),
            credentials=CREDS,
            proof=None,
            dry_run_only=True,
        )
    transport = dry(runner, clock)
    preview = transport.preview(OPEN_BUY)
    assert preview.argv[-1] == DRY_RUN_FLAG
    with pytest.raises(DryRunOnlyError):
        transport.place(mint_for_test(OPEN_BUY))
    with pytest.raises(DryRunOnlyError):
        transport.order(client_oid=OPEN_BUY.client_oid)
    with pytest.raises(DryRunOnlyError):
        transport.positions()
    with pytest.raises(DryRunOnlyError):
        transport.place_stop(
            symbol="NVDAUSDT",
            pos_side="long",
            qty=Decimal(1),
            stop_price=Decimal(1),
            client_oid=OPEN_BUY.client_oid,
        )
    assert [a for a in runner.argvs() if DRY_RUN_FLAG not in a] == []


# --- preview ----------------------------------------------------------------------------------


def test_preview_is_checked_against_the_intent(clock: ManualClock) -> None:
    runner = ScriptedRunner().on(has(DRY_RUN_FLAG), fixture_result("dryrun_place_open_buy_one_way"))
    transport = paper(runner, clock)
    preview = transport.preview(OPEN_BUY)
    assert preview.would_send == would_send(OPEN_BUY, hold_mode=HOLD_ONE_WAY)
    assert preview.operation_id == "placeOrder"
    assert preview.path == "/api/v3/trade/place-order"
    assert preview.argv == tuple(build_place_args(OPEN_BUY, hold_mode=HOLD_ONE_WAY, dry_run=True))
    assert preview.blob is not None
    other = make_intent(qty="2.00")
    with pytest.raises(PreviewMismatchError):
        transport.preview(other)


def test_a_preview_error_refuses(clock: ManualClock) -> None:
    runner = ScriptedRunner().on(has(DRY_RUN_FLAG), fixture_result("local_paper_and_read_only"))
    with pytest.raises(TransportRefusedError, match="preview failed"):
        paper(runner, clock).preview(OPEN_BUY)
    not_a_preview = ScriptedRunner().on(has(DRY_RUN_FLAG), ok_result({"orderId": "1"}))
    with pytest.raises(TransportRefusedError, match="dry-run preview"):
        paper(not_a_preview, clock).preview(OPEN_BUY)


# --- place ------------------------------------------------------------------------------------


def _place_with(result: BgcResult | BaseException, clock: ManualClock) -> object:
    runner = ScriptedRunner().on(verb("order", "--action", "place"), result)
    return paper(runner, clock).place(mint_for_test(OPEN_BUY))


def test_ack_carries_the_venue_order_id(clock: ManualClock) -> None:
    outcome = _place_with(fixture_result("loopback_place_ack"), clock)
    assert isinstance(outcome, VenueAck)
    assert outcome.venue_order_id == "121211212122"
    assert outcome.client_oid == OPEN_BUY.client_oid
    assert outcome.blob is not None


def test_40099_on_a_send_refuses_the_environment(clock: ManualClock) -> None:
    with pytest.raises(EnvironmentRefused, match="40099"):
        _place_with(fixture_result("loopback_place_40099"), clock)


@pytest.mark.parametrize(
    "answer",
    [
        fixture_result("loopback_place_network_error"),
        fixture_result("loopback_place_503"),
        fixture_result("loopback_place_duplicate"),
        BgcTimeoutError("bgc order --action place did not finish within 45 s"),
        ok_result({"clientOid": "x"}),
    ],
)
def test_an_uncertain_send_is_unknown_never_rejected(
    clock: ManualClock, answer: BgcResult | BaseException
) -> None:
    outcome = _place_with(answer, clock)
    assert isinstance(outcome, VenueUnknown)


def test_a_throttled_send_is_a_retryable_rejection(clock: ManualClock) -> None:
    outcome = _place_with(fixture_result("loopback_place_429"), clock)
    assert isinstance(outcome, VenueRejection)
    assert outcome.retryable
    assert outcome.category == "rate"


def test_a_venue_refusal_is_a_rejection_with_its_message(clock: ManualClock) -> None:
    outcome = _place_with(fixture_result("loopback_place_rejected_balance"), clock)
    assert isinstance(outcome, VenueRejection)
    assert not outcome.retryable
    assert "Insufficient balance" in outcome.message
    assert outcome.code == "400"


def test_a_local_refusal_is_labelled_local(clock: ManualClock) -> None:
    outcome = _place_with(fixture_result("local_paper_and_read_only"), clock)
    assert isinstance(outcome, VenueRejection)
    assert outcome.category == "local"


def test_a_tampered_approval_is_refused_before_bgc_runs(clock: ManualClock) -> None:
    runner = ScriptedRunner()
    approved = mint_for_test(OPEN_BUY)
    object.__setattr__(approved, "_intent", make_intent(qty="50.00"))
    with pytest.raises(TransportRefusedError, match="approval"):
        paper(runner, clock).place(approved)
    assert runner.calls == []


# --- reads ------------------------------------------------------------------------------------


def test_order_detail_is_parsed_from_the_documented_row(clock: ManualClock) -> None:
    runner = ScriptedRunner().on(
        verb("order", "--action", "detail"), fixture_result("loopback_order_detail_doc")
    )
    order = paper(runner, clock).order(client_oid=OPEN_BUY.client_oid)
    assert order is not None
    assert order.venue_order_id == "111111111111111111"
    assert order.status is VenueOrderStatus.FILLED
    assert order.cum_exec_qty == Decimal("0.0372")
    assert order.avg_price == Decimal("2684.23")
    assert order.reduce_only is False
    assert order.fees[0].coin == "ETH"
    assert order.fees[0].raw == Decimal("0.00000744")
    assert order.created_at.year == 2024
    assert order.cancel_reason is None


def test_order_not_found_is_none_and_other_failures_raise(clock: ManualClock) -> None:
    missing = ScriptedRunner().on(verb("order"), fixture_result("loopback_detail_not_found"))
    assert paper(missing, clock).order(client_oid=OPEN_BUY.client_oid) is None
    empty = ScriptedRunner().on(verb("order"), ok_result(None))
    assert paper(empty, clock).order(client_oid=OPEN_BUY.client_oid) is None
    auth = ScriptedRunner().on(verb("order"), fixture_result("loopback_detail_auth_200"))
    with pytest.raises(VenueReadError):
        paper(auth, clock).order(client_oid=OPEN_BUY.client_oid)
    network = ScriptedRunner().on(verb("order"), fixture_result("loopback_place_network_error"))
    with pytest.raises(VenueReadError):
        paper(network, clock).order(client_oid=OPEN_BUY.client_oid)
    slow = ScriptedRunner().on(verb("order"), BgcTimeoutError("slow"))
    with pytest.raises(VenueReadError) as caught:
        paper(slow, clock).order(client_oid=OPEN_BUY.client_oid)
    assert caught.value.outcome_unknown


def test_fills_are_parsed_with_fees_as_costs(clock: ManualClock) -> None:
    runner = ScriptedRunner().on(
        verb("order", "--action", "fills"), fixture_result("loopback_fills_doc")
    )
    fills = paper(runner, clock).fills(since=T0 - timedelta(days=1), until=T0)
    assert len(fills) == 1
    fill = fills[0]
    assert fill.exec_id == "131111111111111111"
    assert fill.side is Side.SELL
    assert fill.exec_price == Decimal("106950.1")
    assert fill.fee_paid == Decimal("0.6417006")
    assert fill.trade_scope == "taker"
    assert fill.trade_side == "open"
    assert fill.exec_pnl == Decimal("-0.002")


def test_fills_with_a_negative_fee_are_still_a_cost(clock: ManualClock) -> None:
    row = fixture("loopback_fills_doc")["result"]["stdout"]["data"]["list"][0]
    negative = {**row, "feeDetail": [{"feeCoin": "USDT", "fee": "-0.6417006"}]}
    runner = ScriptedRunner().on(verb("order"), ok_result({"list": [negative], "cursor": ""}))
    fills = paper(runner, clock).fills(since=T0 - timedelta(days=1), until=T0)
    assert fills[0].fee_paid == Decimal("0.6417006")


def _fill_row(n: int) -> dict[str, Any]:
    row: dict[str, Any] = dict(fixture("loopback_fills_doc")["result"]["stdout"]["data"]["list"][0])
    row["execId"] = f"exec-{n:05d}"
    return row


def test_fills_walk_the_venue_cursor_and_split_30_day_windows(clock: ManualClock) -> None:
    pages = [
        ok_result({"list": [_fill_row(i) for i in range(PAGE_LIMIT)], "cursor": "c1"}),
        ok_result({"list": [_fill_row(PAGE_LIMIT + i) for i in range(3)], "cursor": "c2"}),
        ok_result({"list": [_fill_row(0)], "cursor": ""}),
    ]
    runner = ScriptedRunner()
    for page in pages:
        runner.on(verb("order", "--action", "fills"), page, once=True)
    fills = paper(runner, clock).fills(since=T0 - timedelta(days=45), until=T0)
    assert len(fills) == PAGE_LIMIT + 3  # exec-00000 repeated in the second window: once
    first, second, third = runner.argvs()
    assert "--cursor" not in first
    assert second[second.index("--cursor") + 1] == "c1"
    assert "--cursor" not in third  # the second 30-day window starts afresh
    start = int(first[first.index("--startTime") + 1])
    end = int(first[first.index("--endTime") + 1])
    assert end - start <= 30 * 86_400_000


def test_a_cursor_walk_that_never_ends_raises(clock: ManualClock) -> None:
    counter = itertools.count()
    runner = ScriptedRunner().on(
        verb("order"),
        lambda _args: ok_result(
            {
                "list": [_fill_row(next(counter)) for _ in range(PAGE_LIMIT)],
                "cursor": f"c{next(counter)}",
            }
        ),
    )
    with pytest.raises(VenueReadError, match="did not end"):
        paper(runner, clock).fills(since=T0 - timedelta(days=1), until=T0)


def test_an_unreadable_fill_raises_rather_than_being_skipped(clock: ManualClock) -> None:
    row = {**_fill_row(1), "execPrice": ""}
    runner = ScriptedRunner().on(verb("order"), ok_result({"list": [row]}))
    with pytest.raises(VenueReadError, match="unreadable fill"):
        paper(runner, clock).fills(since=T0 - timedelta(days=1), until=T0)


def test_history_positions_stops_and_account(clock: ManualClock) -> None:
    transport = paper(_answer_everything(), clock)
    history = transport.history(since=T0 - timedelta(days=1), until=T0)
    assert history[0].status is VenueOrderStatus.FILLED
    assert history[0].fees[0].raw == Decimal("4.2500586")
    positions = transport.positions()
    assert len(positions) == 1
    assert positions[0].symbol == "BTCUSDT"
    assert positions[0].qty == Decimal("119.2068")
    assert positions[0].avg_price == Decimal("108674")
    stops = transport.stop_orders()
    assert stops[0].stop_price == Decimal("90000")
    assert stops[0].venue_id == "111111111111111111"
    account = transport.account()
    # The book's equity is the USDT row (6.19...), not the account's usdtEquity (11.14), which
    # values every coin the account holds; see environment.account_from_assets.
    assert account.equity_usdt == Decimal("6.19300826")
    assert account.available_usdt == Decimal("6.19300826")


def test_short_positions_are_negative_and_unreadable_rows_raise(clock: ManualClock) -> None:
    row = fixture("loopback_positions_doc")["result"]["stdout"]["data"]["list"][0]
    short = {**row, "posSide": "short", "total": "2"}
    flat = {**row, "total": "0"}
    runner = ScriptedRunner().on(verb("position"), ok_result({"list": [short, flat]}))
    positions = paper(runner, clock).positions()
    assert [p.qty for p in positions] == [Decimal(-2)]
    bad = ScriptedRunner().on(verb("position"), ok_result({"list": [{**row, "posSide": ""}]}))
    with pytest.raises(VenueReadError, match="posSide"):
        paper(bad, clock).positions()


def test_closed_and_foreign_category_strategy_orders_are_not_open_stops(clock: ManualClock) -> None:
    row = fixture("loopback_stop_orders_doc")["result"]["stdout"]["data"][0]
    rows = [
        row,
        {**row, "orderId": "2", "status": "cancelled"},
        {**row, "orderId": "3", "category": "SPOT"},
        {**row, "orderId": "4", "stopLoss": ""},
    ]
    runner = ScriptedRunner().on(verb("strategy_order"), ok_result(rows))
    stops = paper(runner, clock).stop_orders()
    assert [(s.venue_id, s.stop_price) for s in stops] == [
        ("111111111111111111", Decimal("90000")),
        ("4", None),
    ]


def test_an_unreadable_account_raises_rather_than_reading_as_empty(clock: ManualClock) -> None:
    network = ScriptedRunner().on(verb("raw"), fixture_result("loopback_place_network_error"))
    with pytest.raises(VenueReadError):
        paper(network, clock).account()
    empty = ScriptedRunner().on(verb("raw"), ok_result(None))
    with pytest.raises(VenueReadError, match="empty"):
        paper(empty, clock).account()


@pytest.mark.parametrize(
    ("predicate", "answer", "call"),
    [
        (verb("raw"), "loopback_demo_assets_40099", "account"),
        (verb("order"), "loopback_place_40099", "order"),
        (verb("position"), "loopback_demo_assets_40099", "positions"),
        (verb("strategy_order"), "loopback_demo_assets_40099", "stop_orders"),
    ],
)
def test_40099_on_any_read_refuses_with_its_evidence(
    clock: ManualClock, predicate: Any, answer: str, call: str
) -> None:
    runner = ScriptedRunner().on(predicate, fixture_result(answer))
    transport = paper(runner, clock)
    read = (
        (lambda: transport.order(client_oid=OPEN_BUY.client_oid))
        if call == "order"
        else getattr(transport, call)
    )
    with pytest.raises(EnvironmentRefused) as caught:
        read()
    assert caught.value.evidence is not None


def test_stored_evidence_never_carries_the_account_identity(clock: ManualClock) -> None:
    blobs = MemoryBlobStore()
    settings = fixture("loopback_demo_overview_ok")["result"]["stdout"]["data"]["settings"]["data"]
    assert settings["uid"] == "1111111111"
    runner = ScriptedRunner().on(verb("raw"), ok_result({**settings, "usdtEquity": "1"}))
    transport = BgcTransport(
        runner=runner,
        clock=clock,
        blobs=blobs,
        credentials=CREDS,
        proof=passed_proof(),
        dry_run_only=False,
    )
    transport.account()
    (stored,) = blobs.data.values()
    record = json.loads(stored)
    assert record["stdout"]["data"]["uid"] == "<redacted>"
    assert b"1111111111" not in stored
    assert record["stdout"]["data"]["holdMode"] == "one_way_mode"


# --- stops ------------------------------------------------------------------------------------


def test_place_and_cancel_stop(clock: ManualClock) -> None:
    transport = paper(_answer_everything(), clock)
    placed = transport.place_stop(
        symbol="NVDAUSDT",
        pos_side="long",
        qty=Decimal("1.00"),
        stop_price=Decimal("213.93"),
        client_oid=OPEN_BUY.client_oid,
    )
    assert placed.action == "placed"
    assert placed.venue_id == "121211212122"
    assert placed.stop_price == Decimal("213.93")
    cancelled = transport.cancel_stop(symbol="NVDAUSDT", venue_id="121211212122")
    assert cancelled.action == "cancelled"


def test_stop_failures_raise_with_their_evidence(clock: ManualClock) -> None:
    runner = ScriptedRunner().on(
        verb("strategy_order"), fixture_result("loopback_place_rejected_balance")
    )
    transport = paper(runner, clock)
    with pytest.raises(VenueWriteError) as caught:
        transport.place_stop(
            symbol="NVDAUSDT",
            pos_side="long",
            qty=Decimal(1),
            stop_price=Decimal(200),
            client_oid=OPEN_BUY.client_oid,
        )
    assert caught.value.blob is not None
    assert not caught.value.outcome_unknown
    with pytest.raises(VenueWriteError):
        transport.cancel_stop(symbol="NVDAUSDT", venue_id="1")
    slow = paper(ScriptedRunner().on(verb("strategy_order"), BgcTimeoutError("slow")), clock)
    with pytest.raises(VenueWriteError) as timed_out:
        slow.cancel_stop(symbol="NVDAUSDT", venue_id="1")
    assert timed_out.value.outcome_unknown


# --- the runner -------------------------------------------------------------------------------


def test_parse_output() -> None:
    assert parse_output(b"") is None
    assert parse_output(b'{"a": 1}\n') == {"a": 1}
    warned = b'(node:1) ExperimentalWarning: something\n{\n  "ok": false\n}\n'
    assert parse_output(warned) == {"ok": False}
    assert parse_output(b"Error: Flag --x requires a value.") == {
        "text": "Error: Flag --x requires a value."
    }
    assert parse_output(b"[1, 2]") == {"text": "[1, 2]"}


STAND_IN = r"""import json, os, sys, time
args = sys.argv[1:]
if args[:1] == ["sleep"]:
    time.sleep(float(args[1]))
if args[:1] == ["env"]:
    print(json.dumps({"data": {k: os.environ[k] for k in sorted(os.environ)}}))
    sys.exit(0)
if args[:1] == ["fail"]:
    sys.stderr.write("(node:7) ExperimentalWarning: noise before the JSON\n")
    sys.stderr.write(json.dumps({"ok": False, "error": {"type": "BitgetApiError", "code": "400",
                     "category": "unknown", "message": "HTTP 400 from Bitget: x",
                     "retryable": False}}, indent=2))
    sys.exit(1)
print(json.dumps({"data": {"argv": args}}))
"""
"""A Python stand-in for ``node lib/index.js``: tests may start a Python interpreter and nothing
else (tests/conftest.py), so the runner's process handling is tested with this. The real CLI's
behaviour is recorded in tests/fixtures/execution by capture.py, outside the test run."""


def _stand_in_runner(tmp_path: Path) -> SubprocessBgcRunner:
    """The runner with a Python stand-in for node and the CLI (tests may start only Python)."""
    entry = tmp_path / "hub" / "node_modules" / "@bitget-ai" / "bitget-agent-cli" / "lib"
    entry.mkdir(parents=True)
    (entry / "index.js").write_text(STAND_IN, encoding="utf-8")
    return SubprocessBgcRunner(tmp_path / "hub", node=sys.executable)


def test_the_runner_passes_argv_and_parses_json(tmp_path: Path) -> None:
    runner = _stand_in_runner(tmp_path)
    result = runner(["order", "--action", "detail", PAPER_FLAG], env=base_child_env(), timeout_s=60)
    assert result.exit_code == 0
    assert result.stdout == {"data": {"argv": ["order", "--action", "detail", PAPER_FLAG]}}
    assert result.stderr is None
    assert result.duration_ms >= 0


def test_the_child_sees_exactly_the_environment_it_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BITGET_API_KEY", "LIVE-KEY-IN-THE-PARENT")
    monkeypatch.setenv("SOMETHING_ELSE", "x")
    runner = _stand_in_runner(tmp_path)
    env = CREDS.child_env()
    result = runner(["env"], env=env, timeout_s=60)
    assert result.stdout is not None
    child = {k.upper(): v for k, v in result.stdout["data"].items()}
    assert child["BITGET_API_KEY"] == DEMO_KEY
    assert "SOMETHING_ELSE" not in child
    assert not [
        k
        for k in child
        if k.startswith("BITGET_")
        and k not in {"BITGET_API_KEY", "BITGET_SECRET_KEY", "BITGET_PASSPHRASE"}
    ]


def test_the_runner_parses_a_structured_error_after_noise(tmp_path: Path) -> None:
    result = _stand_in_runner(tmp_path)(["fail"], env=base_child_env(), timeout_s=60)
    assert result.exit_code == 1
    assert result.stderr is not None
    assert result.stderr["error"]["message"] == "HTTP 400 from Bitget: x"


def test_the_runner_times_out(tmp_path: Path) -> None:
    runner = _stand_in_runner(tmp_path)
    with pytest.raises(BgcTimeoutError, match="did not finish"):
        runner(["sleep", "30"], env=base_child_env(), timeout_s=0.5)


def test_the_runner_refuses_malformed_arguments(tmp_path: Path) -> None:
    runner = _stand_in_runner(tmp_path)
    with pytest.raises(TransportRefusedError):
        runner(["order", "--symbol", "BTC" + chr(10) + "USDT"], env={}, timeout_s=5)


def test_a_missing_cli_is_unavailable(tmp_path: Path) -> None:
    with pytest.raises(BgcUnavailableError, match="npm ci"):
        SubprocessBgcRunner(tmp_path)
