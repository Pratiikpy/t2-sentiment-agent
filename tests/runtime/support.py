"""A fake world for the runtime tests: Bitget market data, the two Bitget services, the crowd, the
``ots`` calendar and scripted model answers, all in memory and all driven by a :class:`ManualClock`.

Prices move only when a test moves them. Every candle series oscillates by 12 bps an hour so the
Demo index always "moves" (G1's stale-index check) and ATR is never zero. Demo and live last prices
are identical, so G1's Demo-live gap never fires unless a test skews a Demo mark on purpose.
"""

import hashlib
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from sentiment_agent.clock import ManualClock
from sentiment_agent.ledger.anchor import OTS_COMMAND
from sentiment_agent.llm.fakes import completion_from_json
from sentiment_agent.policy import POLICY_V1
from sentiment_agent.runtime.wiring import Parts
from sentiment_agent.types import (
    AssetClass,
    CalendarItem,
    Candle,
    CandleKind,
    Category,
    Completion,
    DerivativesReading,
    FundingPoint,
    InstrumentSpec,
    MoodReading,
    Policy,
    PriceSource,
    Quote,
    SourceCall,
    SourceHealth,
    TextItem,
    ToolkitSurface,
)

HOUR = timedelta(hours=1)
FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"

BASE_PRICES: Mapping[str, Decimal] = {
    "BTCUSDT": Decimal("65000"),
    "SP500USDT": Decimal("6600"),
    "NDX100USDT": Decimal("24000"),
    "MSTRUSDT": Decimal("350"),
    "HOODUSDT": Decimal("120"),
    "CRCLUSDT": Decimal("150"),
    "SNDKUSDT": Decimal("90"),
    "COINUSDT": Decimal("330"),
    "TSLAUSDT": Decimal("420"),
    "GOOGLUSDT": Decimal("250"),
    "METAUSDT": Decimal("760"),
    "NVDAUSDT": Decimal("180"),
    "AMZNUSDT": Decimal("230"),
    "AAPLUSDT": Decimal("250"),
}

_Q = Decimal("0.00000001")


def _osc(at: datetime) -> Decimal:
    """+6 bps on even hours, -6 bps on odd hours: every hourly move is 12 bps."""
    return Decimal("0.0006") if int(at.timestamp() // 3600) % 2 == 0 else Decimal("-0.0006")


class FakeWorld:
    """Keyless Bitget market data (``types.MarketData``) on prices the test sets."""

    def __init__(self, clock: ManualClock, policy: Policy = POLICY_V1) -> None:
        self.clock = clock
        self.policy = policy
        epoch = datetime(2026, 1, 1, tzinfo=UTC)
        self._history: dict[str, list[tuple[datetime, Decimal]]] = {
            s: [(epoch, BASE_PRICES[s])] for s in policy.symbols
        }
        self.mark_skew: dict[str, Decimal] = {}
        """Demo mark = price * (1 + skew): how a test makes G1's mark-index check fire."""
        self.fail_quotes = False
        self.quote_calls = 0

    # --- the test's controls ----------------------------------------------------------------

    def price(self, symbol: str, at: datetime | None = None) -> Decimal:
        when = self.clock.now() if at is None else at
        value = self._history[symbol][0][1]
        for moment, price in self._history[symbol]:
            if moment <= when:
                value = price
        return value

    def set_price(self, symbol: str, price: Decimal) -> None:
        self._history[symbol].append((self.clock.now(), price))

    def move(self, symbol: str, pct: str) -> Decimal:
        """Move ``symbol`` by ``pct`` percent (``"-6"``) now; returns the new price."""
        new = (self.price(symbol) * (1 + Decimal(pct) / 100)).quantize(Decimal("0.01"))
        self.set_price(symbol, new)
        return new

    # --- MarketData ---------------------------------------------------------------------------

    def instruments(self, source: PriceSource, symbols: Sequence[str]) -> dict[str, InstrumentSpec]:
        out: dict[str, InstrumentSpec] = {}
        for symbol in symbols:
            if symbol not in BASE_PRICES:
                continue
            crypto = symbol == "BTCUSDT"
            out[symbol] = InstrumentSpec(
                symbol=symbol,
                category=Category.USDT_FUTURES,
                source=source,
                base_coin=symbol.removesuffix("USDT"),
                quote_coin="USDT",
                status="online",
                min_order_qty=Decimal("0.0001") if crypto else Decimal("0.01"),
                qty_step=Decimal("0.0001") if crypto else Decimal("0.01"),
                price_step=Decimal("0.1") if crypto else Decimal("0.01"),
                min_order_amount=Decimal("5"),
                max_market_order_qty=Decimal("100") if crypto else Decimal("1000"),
                max_order_qty=None,
                taker_fee_rate=Decimal("0.0006"),
                maker_fee_rate=Decimal("0.0002"),
                max_leverage=25,
                fund_interval_hours=8,
                fetched_at=self.clock.now(),
            )
        return out

    def quotes(self, source: PriceSource, symbols: Sequence[str]) -> dict[str, Quote]:
        self.quote_calls += 1
        if self.fail_quotes:
            raise OSError("fake network down")
        now = self.clock.now()
        out: dict[str, Quote] = {}
        for symbol in symbols:
            if symbol not in BASE_PRICES:
                continue
            price = self.price(symbol)
            skew = self.mark_skew.get(symbol, Decimal(0)) if source is PriceSource.DEMO else 0
            out[symbol] = Quote(
                symbol=symbol,
                source=source,
                ts=now,
                fetched_at=now,
                last=price,
                mark=(price * (1 + skew)).quantize(_Q),
                index=price,
                bid=(price * Decimal("0.9999")).quantize(_Q),
                ask=(price * Decimal("1.0001")).quantize(_Q),
                funding_rate=Decimal("0.0001"),
                open_interest=Decimal("1000000"),
                turnover_24h=Decimal("50000000"),
                price_change_24h=Decimal("0.01"),
            )
        return out

    def candles(
        self,
        source: PriceSource,
        symbol: str,
        *,
        kind: CandleKind,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> list[Candle]:
        if symbol not in BASE_PRICES:
            return []
        first = start.replace(minute=0, second=0, microsecond=0)
        if first < start:
            first += HOUR
        out: list[Candle] = []
        at = first
        now = self.clock.now()
        while at <= end and at + HOUR <= now:
            close_at = at + HOUR
            base = self.price(symbol, close_at)
            close = (base * (1 + _osc(at))).quantize(_Q)
            open_ = (self.price(symbol, at) * (1 - _osc(at))).quantize(_Q)
            out.append(
                Candle(
                    symbol=symbol,
                    source=source,
                    kind=kind,
                    interval=interval,
                    open_time=at,
                    open=open_,
                    high=max(open_, close) * Decimal("1.0002"),
                    low=min(open_, close) * Decimal("0.9998"),
                    close=close,
                    volume=Decimal("1000"),
                )
            )
            at += HOUR
        return out

    def funding_history(self, symbol: str, *, limit: int) -> list[FundingPoint]:
        now = self.clock.now()
        last = now.replace(hour=(now.hour // 8) * 8, minute=0, second=0, microsecond=0)
        return [
            FundingPoint(
                symbol=symbol,
                source=PriceSource.LIVE,
                ts=last - timedelta(hours=8 * k),
                rate=Decimal("0.0001"),
            )
            for k in range(limit)
        ]


def _call(source: str, clock: ManualClock, health: SourceHealth, rows: int) -> SourceCall:
    return SourceCall(
        call_id=hashlib.sha256(f"{source}{clock.now()}".encode()).hexdigest()[:24],
        surface=ToolkitSurface.SIGNAL_MCP,
        source=source,
        health=health,
        started_at=clock.now(),
        latency_ms=5,
        rows=rows,
        blob=None,
    )


class FakeToolkit:
    """bitget-signal and bitget-mcp-server (``types.ToolkitReader``) reading a settable mood."""

    def __init__(self, clock: ManualClock) -> None:
        self.clock = clock
        self.crypto_fear_greed = 50
        self.market_fear_greed = 50
        self.news_items: list[TextItem] = []

    def mood(self) -> tuple[MoodReading, tuple[SourceCall, ...]]:
        reading = MoodReading(
            crypto_fear_greed=self.crypto_fear_greed,
            crypto_fear_greed_label="fake",
            crypto_fear_greed_source="fake:data",
            crypto_fear_greed_alt=self.crypto_fear_greed,
            crypto_fear_greed_alt_source="fake:signal",
            market_fear_greed=self.market_fear_greed,
            market_fear_greed_label="fake",
            market_fear_greed_source="fake:data",
        )
        return reading, (_call("fake.mood", self.clock, SourceHealth.OK, 1),)

    def derivatives(self, symbol: str) -> tuple[DerivativesReading, tuple[SourceCall, ...]]:
        now = self.clock.now().replace(minute=0, second=0, microsecond=0)
        history = tuple((now - HOUR * k, 1_000_000.0 + 10.0 * (48 - k)) for k in range(48, -1, -1))
        reading = DerivativesReading(
            symbol=symbol,
            retail_long_short_ratio=1.1,
            top_trader_account_ratio=1.0,
            top_trader_position_ratio=1.0,
            taker_buy_sell_ratio=1.0,
            open_interest_history=history,
            funding_rate=0.0001,
        )
        return reading, (_call(f"fake.derivatives:{symbol}", self.clock, SourceHealth.OK, 6),)

    def news(self, limit: int) -> tuple[tuple[TextItem, ...], tuple[SourceCall, ...]]:
        items = tuple(self.news_items[:limit])
        health = SourceHealth.OK if items else SourceHealth.EMPTY
        return items, (_call("fake.news", self.clock, health, len(items)),)

    def reddit_trending(self, limit: int) -> tuple[tuple[TextItem, ...], tuple[SourceCall, ...]]:
        return (), (_call("fake.reddit", self.clock, SourceHealth.EMPTY, 0),)

    def calendar(
        self, symbols: Sequence[str], *, since: datetime
    ) -> tuple[tuple[CalendarItem, ...], tuple[SourceCall, ...]]:
        return (), (_call("fake.calendar", self.clock, SourceHealth.EMPTY, 0),)


class FakeCrowd:
    """X and Reddit (``types.CrowdCollector``): nothing to read, and it says so."""

    def __init__(self, clock: ManualClock) -> None:
        self.clock = clock

    def collect(
        self, symbols: Sequence[str], *, since: datetime
    ) -> tuple[tuple[TextItem, ...], tuple[SourceCall, ...]]:
        return (), (_call("fake.crowd", self.clock, SourceHealth.EMPTY, 0),)


# ================================================================================================
# OpenTimestamps
# ================================================================================================

_OTS_MAGIC = b"\x00OpenTimestamps\x00\x00Proof\x00\xbf\x89\xe2\xe8\x84\xe8\x92\x94"
_PENDING = bytes.fromhex("83dfe30d2ef90c8e")


def _varuint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def pending_proof(digest: bytes, calendar: bytes = b"https://alice.btc.calendar.example") -> bytes:
    """A syntactically valid pending ``.ots`` proof committing to ``digest`` (python-opentimestamps
    ``core/timestamp.py`` layout; the calendar is not contacted)."""
    uri = _varuint(len(calendar)) + calendar
    return (
        _OTS_MAGIC + _varuint(1) + b"\x08" + digest + b"\x00" + _PENDING + _varuint(len(uri)) + uri
    )


class FakeOts:
    """An ``ots`` stand-in: ``stamp`` writes a pending proof, ``upgrade`` finds no block yet."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, argv: Sequence[str], cwd: Path, timeout_s: float) -> tuple[int, str, str]:
        assert argv[0] == OTS_COMMAND
        self.calls.append(tuple(argv))
        if argv[1] == "stamp":
            target = cwd / argv[-1]
            digest = hashlib.sha256(target.read_bytes()).digest()
            (cwd / f"{argv[-1]}.ots").write_bytes(pending_proof(digest))
            return 0, "", ""
        return 1, "", "Pending confirmation in Bitcoin blockchain"


# ================================================================================================
# Scripted model answers
# ================================================================================================


def target(
    symbol: str,
    value: float,
    *,
    thesis: str = "crowd positioning and the mood reading support this side",
    invalidation: str = "positioning flips against the thesis",
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "target": value,
        "thesis": thesis,
        "invalidation": invalidation,
        "horizon_hours": 48,
        "crowd_belief": "the crowd leans the other way",
        "our_view": "we take the side the positioning favours",
        "confidence": 0.55,
        "evidence": [],
        "invalidation_triggered": False,
        "invalidation_evidence": None,
    }


def decision(
    stance: str, targets: Sequence[Mapping[str, Any]], *, flat_reasons: Sequence[str] = ()
) -> Completion:
    return completion_from_json(
        {
            "stance": stance,
            "targets": list(targets),
            "rejected_alternatives": [
                {"action": "stay flat", "reason": "the mandate asks for a view"}
            ],
            "mandate_response": "deploy part of the budget where positioning is stretched",
            "flat_reasons": list(flat_reasons),
            "summary": f"scripted {stance} decision for the runtime test",
        }
    )


def flat(reason: str = "nothing is stretched enough to trade") -> Completion:
    return decision("flat_with_reasons", [], flat_reasons=[reason])


def make_parts(
    clock: ManualClock,
    *,
    world: FakeWorld | None = None,
    toolkit: FakeToolkit | None = None,
    script: Sequence[Any] = (),
    venue: Any = None,
    oi_thresholds: Mapping[str, float] | None = None,
    ots: Callable[[Sequence[str], Path, float], tuple[int, str, str]] | None = None,
    bgc_runner: Any = None,
    chat_model: Any = None,
) -> Parts:
    return Parts(
        market=world or FakeWorld(clock),
        toolkit=toolkit or FakeToolkit(clock),
        crowd=FakeCrowd(clock),
        scripted=tuple(script),
        chat_model=chat_model,
        venue=venue,
        bgc_runner=bgc_runner,
        ots_runner=ots or FakeOts(),
        sleep=lambda _s: None,
        oi_thresholds=oi_thresholds,
        poll_timeout_s=0.0,
    )


def is_crypto(symbol: str) -> bool:
    entry = POLICY_V1.entry(symbol)
    return entry is not None and entry.asset_class is AssetClass.CRYPTO


# ================================================================================================
# A Demo account behind bgc, for the PAPER tests
# ================================================================================================


def _fixture(name: str) -> dict[str, Any]:
    import json

    path = Path(__file__).resolve().parents[1] / "fixtures" / "execution" / f"{name}.json"
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _result(name: str, *, equity: str | None = None) -> Any:
    import copy

    from sentiment_agent.execution.bgc import BgcResult

    raw = copy.deepcopy(_fixture(name)["result"])
    data = (raw.get("stdout") or {}).get("data")
    if equity is not None and isinstance(data, dict):
        section = data.get("assets")
        node = section.get("data") if isinstance(section, dict) else data
        if isinstance(node, dict) and "usdtEquity" in node:
            node["usdtEquity"] = equity
            node["accountEquity"] = equity
            for row in node.get("assets") or []:
                if isinstance(row, dict) and row.get("coin") == "USDT":
                    row["equity"] = equity  # the book's equity is the USDT row
    return BgcResult(
        exit_code=raw["exit_code"], stdout=raw["stdout"], stderr=raw["stderr"], duration_ms=1
    )


def _ok(data: Any) -> Any:
    from sentiment_agent.execution.bgc import BgcResult

    return BgcResult(
        exit_code=0,
        stdout={"endpoint": "GET (test)", "requestTime": "2026-09-25T13:00:00Z", "data": data},
        stderr=None,
        duration_ms=1,
    )


def _flags(args: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    items = list(args)
    for index, arg in enumerate(items):
        following = items[index + 1] if index + 1 < len(items) else None
        if arg.startswith("--") and following is not None and not following.startswith("--"):
            out[arg[2:]] = following
    return out


class FakeBgc:
    """``bgc`` against a Demo account that proves itself (demo reads succeed, the live read is
    refused), previews every order from its own argv, holds nothing, and answers each send with
    ``send`` (40099 unless the test says otherwise). Every argv is recorded."""

    PLACE_PATH = "/api/v3/trade/place-order"

    def __init__(
        self,
        *,
        equity: str = "100000",
        demo_overview: str = "loopback_demo_overview_ok",
        live: str = "loopback_live_negative_40006",
        send: str = "loopback_place_40099",
    ) -> None:
        self.equity = equity
        self.demo_overview = demo_overview
        self.live = live
        self.send = send
        self.calls: list[tuple[str, ...]] = []
        self.envs: list[dict[str, str]] = []
        """The child environment of every call (credential values included, for the test to
        check where they went; nothing here prints or stores them)."""

    def sends(self) -> list[tuple[str, ...]]:
        place = ("order", "--action", "place")
        return [c for c in self.calls if c[:3] == place and "--dry-run" not in c]

    def __call__(self, args: Sequence[str], *, env: Mapping[str, str], timeout_s: float) -> Any:
        from sentiment_agent.execution.bgc import BgcResult

        argv = tuple(args)
        self.calls.append(argv)
        self.envs.append(dict(env))
        paper = "--paper-trading" in argv
        if argv[0] == "account_overview" and paper:
            return _result(self.demo_overview, equity=self.equity)
        if argv[:3] == ("raw", "--operationId", "getAccountAssets"):
            if paper:
                return _result("loopback_demo_assets_ok", equity=self.equity)
            return _result(self.live)
        if argv[:3] == ("order", "--action", "place"):
            flags = _flags(argv[3:])
            if "--dry-run" in argv:
                return BgcResult(
                    exit_code=0,
                    stdout={
                        "endpoint": f"POST {self.PLACE_PATH}",
                        "requestTime": "2026-09-25T13:00:00Z",
                        "data": {
                            "dryRun": True,
                            "operationId": "placeOrder",
                            "method": "POST",
                            "path": self.PLACE_PATH,
                            "riskLevel": "write",
                            "wouldSend": flags,
                        },
                    },
                    stderr=None,
                    duration_ms=1,
                )
            return _result(self.send)
        if argv[:3] in (("order", "--action", "fills"), ("order", "--action", "history")):
            return _ok({"list": [], "cursor": ""})
        if argv[:3] == ("order", "--action", "detail"):
            return _result("loopback_detail_not_found")
        if argv[:3] in (("position", "--action", "info"), ("strategy_order", "--action", "open")):
            return _ok({"list": []})
        raise AssertionError(f"unexpected bgc call: {list(argv)}")


def paper_project(root: Path) -> Path:
    """A project root with a Demo credential file and an installed Agent Hub that meets the
    vendor contract (the execution tests' own builders)."""
    from execution.fakes import make_agent_hub, write_demo_env

    write_demo_env(root)
    make_agent_hub(root)
    (root / "pyproject.toml").write_text('[project]\nname = "t2-sentiment-agent"\n', "utf-8")
    return root


class FakeDemoVenue(FakeBgc):
    """``bgc --paper-trading`` against a stateful Demo account: market orders fill at the Demo ask
    or bid of ``world``, preset stops become strategy orders, and every read (order detail, fills,
    history, positions, stop orders, account) answers from that state in the documented UTA
    shapes the transport parses. Nothing is sent anywhere."""

    def __init__(self, world: "FakeWorld", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.world = world
        self.orders: dict[str, dict[str, str]] = {}
        self.fill_rows: list[dict[str, Any]] = []
        self.positions: dict[str, tuple[Decimal, Decimal]] = {}
        self.stops: dict[str, dict[str, str]] = {}
        self._n = 0

    def _id(self, kind: str) -> str:
        self._n += 1
        return f"{kind}{self._n:010d}"

    def _ms(self) -> str:
        return str(int(self.world.clock.now().timestamp() * 1000))

    def _fill(self, flags: Mapping[str, str]) -> Any:
        symbol, side = flags["symbol"], flags["side"]
        qty = Decimal(flags["qty"])
        quote = self.world.quotes(PriceSource.DEMO, [symbol])[symbol]
        price = quote.ask if side == "buy" else quote.bid
        held, avg = self.positions.get(symbol, (Decimal(0), Decimal(0)))
        signed = qty if side == "buy" else -qty
        if flags.get("reduceOnly") == "yes" and abs(signed) > abs(held):
            signed = -held
        closing = held != 0 and (held > 0) != (signed > 0)
        new = held + signed
        if new == 0:
            self.positions.pop(symbol, None)
            for stop_id in [k for k, v in self.stops.items() if v["symbol"] == symbol]:
                del self.stops[stop_id]
        elif closing and (new > 0) == (held > 0):
            self.positions[symbol] = (new, avg)
        elif closing:
            self.positions[symbol] = (new, price)
        else:
            total = abs(held) + abs(signed)
            self.positions[symbol] = (new, (avg * abs(held) + price * abs(signed)) / total)
        order_id = self._id("7")
        stamp = self._ms()
        value = abs(signed) * price
        self.orders[flags["clientOid"]] = {
            "orderId": order_id,
            "clientOid": flags["clientOid"],
            "category": "USDT-FUTURES",
            "symbol": symbol,
            "orderType": "market",
            "side": side,
            "qty": flags["qty"],
            "cumExecQty": str(abs(signed)),
            "cumExecValue": str(value),
            "avgPrice": str(price),
            "orderStatus": "filled",
            "reduceOnly": "YES" if flags.get("reduceOnly") == "yes" else "NO",
            "delegateType": "normal",
            "cancelReason": "",
            "createdTime": stamp,
            "updatedTime": stamp,
        }
        self.fill_rows.append(
            {
                "execId": self._id("8"),
                "orderId": order_id,
                "clientOid": flags["clientOid"],
                "category": "USDT-FUTURES",
                "symbol": symbol,
                "orderType": "market",
                "side": side,
                "execPrice": str(price),
                "execQty": str(abs(signed)),
                "execValue": str(value),
                "tradeScope": "taker",
                "tradeSide": "close" if closing else "open",
                "feeDetail": [{"feeCoin": "USDT", "fee": str(value * Decimal("0.0006"))}],
                "createdTime": stamp,
                "updatedTime": stamp,
            }
        )
        if "stopLoss" in flags and symbol in self.positions:
            self.stops[self._id("9")] = {"symbol": symbol, "stopLoss": flags["stopLoss"]}
        return _ok({"orderId": order_id, "clientOid": flags["clientOid"]})

    def _window(self, rows: Sequence[Mapping[str, Any]], flags: Mapping[str, str]) -> Any:
        start, end = int(flags["startTime"]), int(flags["endTime"])
        chosen = [r for r in rows if start <= int(r["createdTime"]) <= end]
        return _ok({"list": chosen, "cursor": ""})

    def __call__(self, args: Sequence[str], *, env: Mapping[str, str], timeout_s: float) -> Any:
        argv = tuple(args)
        flags = _flags(argv[3:])
        head = argv[:3]
        self.envs.append(dict(env))
        if head == ("order", "--action", "place") and "--dry-run" not in argv:
            self.calls.append(argv)
            return self._fill(flags)
        if head == ("order", "--action", "detail"):
            self.calls.append(argv)
            row = self.orders.get(flags.get("clientOid", ""))
            return _ok(row) if row is not None else _result("loopback_detail_not_found")
        if head == ("order", "--action", "fills"):
            self.calls.append(argv)
            return self._window(self.fill_rows, flags)
        if head == ("order", "--action", "history"):
            self.calls.append(argv)
            return self._window(list(self.orders.values()), flags)
        if head == ("position", "--action", "info"):
            self.calls.append(argv)
            rows = [
                {
                    "symbol": s,
                    "total": str(abs(q)),
                    "posSide": "long" if q > 0 else "short",
                    "avgPrice": str(a),
                }
                for s, (q, a) in self.positions.items()
            ]
            return _ok({"list": rows})
        if head == ("strategy_order", "--action", "open"):
            self.calls.append(argv)
            rows = [
                {
                    "orderId": k,
                    "symbol": v["symbol"],
                    "stopLoss": v["stopLoss"],
                    "category": "USDT-FUTURES",
                    "status": "live",
                }
                for k, v in self.stops.items()
            ]
            return _ok({"list": rows})
        if head == ("strategy_order", "--action", "place"):
            self.calls.append(argv)
            stop_id = self._id("9")
            self.stops[stop_id] = {"symbol": flags["symbol"], "stopLoss": flags["stopLoss"]}
            return _ok({"orderId": stop_id, "clientOid": flags.get("clientOid", "")})
        if head == ("strategy_order", "--action", "cancel"):
            self.calls.append(argv)
            self.stops.pop(flags.get("orderId", ""), None)
            return _ok({"orderId": flags.get("orderId", "")})
        return super().__call__(args, env=env, timeout_s=timeout_s)
