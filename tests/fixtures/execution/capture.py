"""Record how the pinned ``bgc`` answers, as fixtures for ``tests/execution``. Never calls Bitget.

    python tests/fixtures/execution/capture.py

Needs ``npm ci --ignore-scripts`` in ``tools/agent-hub`` first. Every file it writes records when,
how and with which argv it was made. Three kinds of capture:

* ``dryrun_*.json``: the production argv with ``--dry-run`` and **no credential** in the child
  environment. The SDK returns the preview before its credential gate and before any request
  (``agent-sdk/src/tools/safety.ts:86-88``), so nothing leaves the machine.
* ``local_*.json``: calls ``bgc`` refuses on its own before any request (no credential; both
  ``--paper-trading`` and ``--read-only``).
* ``loopback_*.json``: the production argv plus ``--base-url http://127.0.0.1:<port>`` and a
  throwaway credential, answered by a server on the loopback interface with Bitget's documented
  response bodies (``bitget_docs.json``) or with error bodies of the shape Bitget returns (an
  unsigned private read was measured on 2026-09-24 to answer HTTP 400
  ``{"code":"40006","msg":"Invalid ACCESS_KEY"}``). Each file also records the requests the server
  received: method, path, query, whether ``paptrading: 1`` was present and whether the request was
  signed, and the JSON body of a write. Header values are never recorded.

Error messages marked ``constructed`` in ``served`` are shaped like Bitget's but their exact text is
NOT VERIFIED; the parsers are tested against them only for how ``bgc`` wraps an error.
"""

import json
import socket
import subprocess
import sys
import threading
from datetime import UTC, datetime
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

from helpers import make_intent  # noqa: E402
from sentiment_agent.execution.bgc import (  # noqa: E402
    BGC_PACKAGE,
    HOLD_HEDGE,
    HOLD_ONE_WAY,
    SubprocessBgcRunner,
    account_args,
    build_place_args,
    cancel_stop_args,
    detail_args,
    fills_args,
    history_args,
    place_stop_args,
    positions_args,
    stop_orders_args,
)
from sentiment_agent.execution.environment import (  # noqa: E402
    DEMO_ASSETS_ARGS,
    DEMO_OVERVIEW_ARGS,
    LIVE_NEGATIVE_ARGS,
    base_child_env,
)
from sentiment_agent.types import OrderPurpose, Side  # noqa: E402

AGENT_HUB = ROOT / "tools" / "agent-hub"
DOCS = json.loads((HERE / "bitget_docs.json").read_text(encoding="utf-8"))["responses"]
THROWAWAY = {
    "BITGET_API_KEY": "loopback-fixture-key",
    "BITGET_SECRET_KEY": "loopback-fixture-secret",
    "BITGET_PASSPHRASE": "loopback-fixture-pass",
}

OPEN_BUY = make_intent(symbol="NVDAUSDT", side=Side.BUY, qty="1.00", purpose=OrderPurpose.OPEN)
OPEN_SELL = make_intent(
    symbol="BTCUSDT", side=Side.SELL, qty="0.01", purpose=OrderPurpose.OPEN, price="112345.6"
)
CLOSE_SELL = make_intent(symbol="NVDAUSDT", side=Side.SELL, qty="1.00", purpose=OrderPurpose.CLOSE)
START_MS, END_MS = "1790208000000", "1790294400000"  # 2026-09-24 00:00 to 2026-09-25 00:00 UTC


def _node_version() -> str:
    out = subprocess.run(
        ["node", "--version"],  # noqa: S607 - node on PATH, as the runner resolves it
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def _doc(key: str) -> dict[str, Any]:
    entry = DOCS[key]
    return {"status": entry["http_status"], "body": entry["body"], "source": entry["source"]}


def _error(status: int, code: str, msg: str, source: str) -> dict[str, Any]:
    return {
        "status": status,
        "body": {"code": code, "msg": msg, "requestTime": 1790248948185, "data": None},
        "source": source,
    }


MEASURED_40006 = "measured 2026-09-24: unsigned GET /api/v3/account/assets, HTTP 400"
OBSERVED_40099 = "code and msg as observed by ARGUS execution/bitget_client.py:251-263"
CONSTRUCTED = "constructed: Bitget's error shape, message text NOT VERIFIED"


class _Server:
    def __init__(self) -> None:
        self.routes: dict[str, dict[str, Any]] = {}
        self.seen: list[dict[str, Any]] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self, method: str) -> None:
                split = urlsplit(self.path)
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                server.seen.append(
                    {
                        "method": method,
                        "path": split.path,
                        "query": {k: v[0] for k, v in sorted(parse_qs(split.query).items())},
                        "paptrading": self.headers.get("paptrading"),
                        "signed": self.headers.get("ACCESS-SIGN") is not None,
                        "body": json.loads(raw) if raw else None,
                    }
                )
                route = server.routes.get(f"{method} {split.path}") or {
                    "status": 404,
                    "body": {"code": "40404", "msg": "Request URL NOT FOUND"},
                }
                payload = json.dumps(route["body"]).encode("utf-8")
                self.send_response(route["status"])
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self) -> None:
                self._answer("GET")

            def do_POST(self) -> None:
                self._answer("POST")

            def log_message(self, format: str, *args: Any) -> None:
                return

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()


def _closed_port_url() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}"


def main() -> int:
    runner = SubprocessBgcRunner(AGENT_HUB)
    node = _node_version()
    stamp = datetime.now(UTC).isoformat()
    written: list[str] = []

    def save(
        name: str,
        how: str,
        argv: list[str] | tuple[str, ...],
        *,
        env: dict[str, str],
        extra: list[str] | None = None,
        served: dict[str, dict[str, Any]] | None = None,
        server: _Server | None = None,
    ) -> None:
        if server is not None:
            server.routes = dict(served or {})
            server.seen = []
        result = runner([*argv, *(extra or [])], env=env, timeout_s=90)
        record = {
            "fixture": name,
            "captured_at": stamp,
            "how": how,
            "bgc": f"{BGC_PACKAGE} with @bitget-ai/bitget-agent-sdk@3.0.0",
            "node": node,
            "argv": list(argv),
            "extra_argv": [
                "--base-url",
                "http://127.0.0.1:<port>",
            ]
            if extra
            else [],
            "served": served or {},
            "requests": server.seen if server is not None else [],
            "result": {
                "exit_code": result.exit_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
        }
        (HERE / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        written.append(name)

    bare = base_child_env()
    dry = "dry-run: production argv, no credential in the child environment, nothing sent"
    save(
        "dryrun_place_open_buy_one_way",
        dry,
        build_place_args(OPEN_BUY, hold_mode=HOLD_ONE_WAY, dry_run=True),
        env=bare,
    )
    save(
        "dryrun_place_open_sell_one_way",
        dry,
        build_place_args(OPEN_SELL, hold_mode=HOLD_ONE_WAY, dry_run=True),
        env=bare,
    )
    save(
        "dryrun_place_close_one_way",
        dry,
        build_place_args(CLOSE_SELL, hold_mode=HOLD_ONE_WAY, dry_run=True),
        env=bare,
    )
    save(
        "dryrun_place_open_buy_hedge",
        dry,
        build_place_args(OPEN_BUY, hold_mode=HOLD_HEDGE, dry_run=True),
        env=bare,
    )
    save(
        "dryrun_place_close_hedge",
        dry,
        build_place_args(CLOSE_SELL, hold_mode=HOLD_HEDGE, dry_run=True),
        env=bare,
    )
    save(
        "dryrun_place_stop",
        dry,
        place_stop_args(
            symbol="NVDAUSDT",
            pos_side="long",
            qty=Decimal("1.00"),
            stop_price=Decimal("213.93"),
            client_oid=OPEN_BUY.client_oid,
            dry_run=True,
        ),
        env=bare,
    )

    local = "refused inside bgc before any request"
    save("local_detail_no_credential", local, detail_args(OPEN_BUY.client_oid), env=bare)
    save("local_overview_no_credential", local, list(DEMO_OVERVIEW_ARGS), env=bare)
    save(
        "local_paper_and_read_only",
        local,
        ["account_overview", "--paper-trading", "--read-only"],
        env=bare,
    )

    server = _Server()
    extra = ["--base-url", server.url]
    keyed = {**bare, **THROWAWAY, "BITGET_MAX_RETRIES": "0", "BITGET_TIMEOUT_MS": "5000"}
    loop = "loopback: real bgc, throwaway credential, answered by a local server"
    overview_doc = {
        "GET /api/v3/account/assets": _doc("GET /api/v3/account/assets"),
        "GET /api/v3/account/settings": _doc("GET /api/v3/account/settings"),
        "GET /api/v3/account/funding-assets": _doc("GET /api/v3/account/funding-assets"),
        "GET /api/v3/position/current-position": _doc("GET /api/v3/position/current-position"),
    }
    env_40099 = _error(400, "40099", "exchange environment is incorrect", OBSERVED_40099)
    cases: list[tuple[str, list[str] | tuple[str, ...], dict[str, dict[str, Any]]]] = [
        (
            "loopback_order_detail_doc",
            detail_args(OPEN_BUY.client_oid),
            {"GET /api/v3/trade/order-info": _doc("GET /api/v3/trade/order-info")},
        ),
        (
            "loopback_fills_doc",
            fills_args(START_MS, END_MS),
            {"GET /api/v3/trade/fills": _doc("GET /api/v3/trade/fills")},
        ),
        (
            "loopback_history_doc",
            history_args(START_MS, END_MS),
            {"GET /api/v3/trade/history-orders": _doc("GET /api/v3/trade/history-orders")},
        ),
        (
            "loopback_positions_doc",
            positions_args(),
            {
                "GET /api/v3/position/current-position": _doc(
                    "GET /api/v3/position/current-position"
                )
            },
        ),
        (
            "loopback_stop_orders_doc",
            stop_orders_args(),
            {
                "GET /api/v3/trade/unfilled-strategy-orders": _doc(
                    "GET /api/v3/trade/unfilled-strategy-orders"
                )
            },
        ),
        (
            "loopback_account_assets_doc",
            account_args(),
            {"GET /api/v3/account/assets": _doc("GET /api/v3/account/assets")},
        ),
        ("loopback_demo_overview_ok", DEMO_OVERVIEW_ARGS, overview_doc),
        (
            "loopback_demo_assets_ok",
            DEMO_ASSETS_ARGS,
            {"GET /api/v3/account/assets": _doc("GET /api/v3/account/assets")},
        ),
        (
            "loopback_place_ack",
            build_place_args(OPEN_BUY, hold_mode=HOLD_ONE_WAY, dry_run=False),
            {"POST /api/v3/trade/place-order": _doc("POST /api/v3/trade/place-order")},
        ),
        (
            "loopback_place_close_ack",
            build_place_args(CLOSE_SELL, hold_mode=HOLD_ONE_WAY, dry_run=False),
            {"POST /api/v3/trade/place-order": _doc("POST /api/v3/trade/place-order")},
        ),
        (
            "loopback_place_stop_ack",
            place_stop_args(
                symbol="NVDAUSDT",
                pos_side="long",
                qty=Decimal("1.00"),
                stop_price=Decimal("213.93"),
                client_oid=OPEN_BUY.client_oid,
            ),
            {
                "POST /api/v3/trade/place-strategy-order": _doc(
                    "POST /api/v3/trade/place-strategy-order"
                )
            },
        ),
        (
            "loopback_cancel_stop_ok",
            cancel_stop_args("121211212122"),
            {
                "POST /api/v3/trade/cancel-strategy-order": _doc(
                    "POST /api/v3/trade/cancel-strategy-order"
                )
            },
        ),
        (
            "loopback_place_40099",
            build_place_args(OPEN_BUY, hold_mode=HOLD_ONE_WAY, dry_run=False),
            {"POST /api/v3/trade/place-order": env_40099},
        ),
        (
            "loopback_demo_overview_40099",
            DEMO_OVERVIEW_ARGS,
            dict.fromkeys(overview_doc, env_40099),
        ),
        (
            "loopback_demo_assets_40099",
            DEMO_ASSETS_ARGS,
            {"GET /api/v3/account/assets": env_40099},
        ),
        (
            "loopback_live_negative_40006",
            LIVE_NEGATIVE_ARGS,
            {
                "GET /api/v3/account/assets": _error(
                    400, "40006", "Invalid ACCESS_KEY", MEASURED_40006
                )
            },
        ),
        (
            "loopback_live_negative_40099",
            LIVE_NEGATIVE_ARGS,
            {"GET /api/v3/account/assets": env_40099},
        ),
        (
            "loopback_live_negative_accepted",
            LIVE_NEGATIVE_ARGS,
            {"GET /api/v3/account/assets": _doc("GET /api/v3/account/assets")},
        ),
        (
            "loopback_live_negative_param",
            LIVE_NEGATIVE_ARGS,
            {
                "GET /api/v3/account/assets": _error(
                    400, "25245", "Account is not in unified mode", CONSTRUCTED
                )
            },
        ),
        (
            "loopback_detail_auth_200",
            detail_args(OPEN_BUY.client_oid),
            {
                "GET /api/v3/trade/order-info": _error(
                    200, "40017", "Parameter verification failed", CONSTRUCTED
                )
            },
        ),
        (
            "loopback_detail_not_found",
            detail_args(OPEN_BUY.client_oid),
            {
                "GET /api/v3/trade/order-info": _error(
                    400, "25204", "Order does not exist", CONSTRUCTED
                )
            },
        ),
        (
            "loopback_place_rejected_balance",
            build_place_args(OPEN_BUY, hold_mode=HOLD_ONE_WAY, dry_run=False),
            {
                "POST /api/v3/trade/place-order": _error(
                    400, "25202", "Insufficient balance", CONSTRUCTED
                )
            },
        ),
        (
            "loopback_place_duplicate",
            build_place_args(OPEN_BUY, hold_mode=HOLD_ONE_WAY, dry_run=False),
            {
                "POST /api/v3/trade/place-order": _error(
                    400, "25212", "Duplicate clientOid", CONSTRUCTED
                )
            },
        ),
        (
            "loopback_place_503",
            build_place_args(OPEN_BUY, hold_mode=HOLD_ONE_WAY, dry_run=False),
            {"POST /api/v3/trade/place-order": {"status": 503, "body": {}, "source": CONSTRUCTED}},
        ),
        (
            "loopback_place_429",
            build_place_args(OPEN_BUY, hold_mode=HOLD_ONE_WAY, dry_run=False),
            {
                "POST /api/v3/trade/place-order": _error(
                    429, "429", "Too many requests", CONSTRUCTED
                )
            },
        ),
    ]
    for name, argv, served in cases:
        save(name, loop, argv, env=keyed, extra=extra, served=served, server=server)

    dead = ["--base-url", _closed_port_url()]
    save(
        "loopback_place_network_error",
        "loopback: real bgc, throwaway credential, nothing listening on the port",
        build_place_args(OPEN_BUY, hold_mode=HOLD_ONE_WAY, dry_run=False),
        env=keyed,
        extra=dead,
    )
    server.httpd.shutdown()
    for name in written:
        print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
