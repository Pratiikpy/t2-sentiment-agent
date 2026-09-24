"""Test support for the Playbook replica (M17): the builder, a fake ``getagent`` SDK, and a loader
that imports a built package the way the runner does (``python -m src.main``: ``src`` is a package
and its modules import each other relatively).

The fake SDK implements exactly the surface the replica uses, with the signatures the getagent
skill documents (``references/sdk/data/*.md``, ``sdk/llm/catalog.md``, ``sdk/runtime/catalog.md``,
``sdk/trade/*.md``). It is a stand-in for the sandbox, never a claim about the real platform: what
the real runner does is listed as NOT VERIFIED in the package README.

Packages are always built into a temporary directory and imported from there, so the committed
``playbook/`` never collects ``__pycache__`` from a test run.
"""

import importlib
import importlib.util
import itertools
import json
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build_playbook.py"
VALIDATOR = Path.home() / ".claude" / "skills" / "getagent" / "scripts" / "validate.py"
FIXTURES = ROOT / "tests" / "fixtures" / "playbook"

_counter = itertools.count()
_BUILDER: ModuleType | None = None


def builder() -> ModuleType:
    """``scripts/build_playbook.py``, loaded once by path (scripts are not a package)."""
    global _BUILDER
    if _BUILDER is None:
        spec = importlib.util.spec_from_file_location("build_playbook_under_test", SCRIPT)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _BUILDER = module
    return _BUILDER


def build_package(directory: Path) -> Path:
    """A fresh build of the package in ``directory`` (created)."""
    out = directory / "package"
    builder().build(out)
    return out


def load_module_from(path: Path, name: str) -> ModuleType:
    """One module imported by path, without writing bytecode next to it (a ``__pycache__`` inside
    a package directory would be part of its upload archive)."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


# ================================================================================================
# The fake SDK
# ================================================================================================


class Clock:
    """A clock the fake SDK advances by ``latency`` seconds on every call it serves."""

    def __init__(self, start: datetime, latency: float = 0.0) -> None:
        self.at = start
        self.latency = latency

    def __call__(self) -> datetime:
        return self.at

    def tick(self, seconds: float | None = None) -> None:
        self.at += timedelta(seconds=self.latency if seconds is None else seconds)


@dataclass
class Result:
    rows: list[dict[str, Any]]


@dataclass
class Market:
    """What the fake data layer answers. Rows use the documented response fields."""

    marks: dict[str, dict[str, Any]] = field(default_factory=dict)
    tickers: dict[str, dict[str, Any]] = field(default_factory=dict)
    klines: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    funding: list[dict[str, Any]] = field(default_factory=list)
    open_interest: list[dict[str, Any]] = field(default_factory=list)
    long_short: list[dict[str, Any]] = field(default_factory=list)
    top_position: list[dict[str, Any]] = field(default_factory=list)
    crypto_fear_greed: list[dict[str, Any]] = field(default_factory=list)
    market_fear_greed: list[dict[str, Any]] = field(default_factory=list)
    news: list[dict[str, Any]] = field(default_factory=list)
    trending: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    earnings: list[dict[str, Any]] = field(default_factory=list)
    insider: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    failing: set[str] = field(default_factory=set)


class _Node:
    def __init__(self, **children: Any) -> None:
        for name, child in children.items():
            setattr(self, name, child)


class FakeData:
    """``getagent.data``: generated endpoint methods and ``to_records``."""

    def __init__(self, market: Market, clock: Clock) -> None:
        self.market = market
        self.clock = clock
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.crypto = _Node(
            futures=_Node(
                mark_price=self._endpoint("crypto.futures.mark_price", self._mark_price),
                ticker=self._endpoint("crypto.futures.ticker", self._ticker),
                kline=self._endpoint("crypto.futures.kline", self._kline),
                funding_rate=self._endpoint(
                    "crypto.futures.funding_rate", lambda **_: self.market.funding
                ),
                open_interest=self._endpoint(
                    "crypto.futures.open_interest", lambda **_: self.market.open_interest
                ),
                long_short_ratio=self._endpoint(
                    "crypto.futures.long_short_ratio", lambda **_: self.market.long_short
                ),
                long_short_top_position_ratio=self._endpoint(
                    "crypto.futures.long_short_top_position_ratio",
                    lambda **_: self.market.top_position,
                ),
            ),
            sentiment=_Node(
                crypto_fear_greed=self._endpoint(
                    "crypto.sentiment.crypto_fear_greed", lambda **_: self.market.crypto_fear_greed
                )
            ),
        )
        self.sentiment = _Node(
            market_fear_greed=self._endpoint(
                "sentiment.market_fear_greed", lambda **_: self.market.market_fear_greed
            ),
            news=self._endpoint("sentiment.news", lambda **_: self.market.news),
            trending=self._endpoint(
                "sentiment.trending",
                lambda **kw: self.market.trending.get(kw.get("filter", ""), []),
            ),
        )
        self.equity = _Node(
            calendar=_Node(
                earnings=self._endpoint(
                    "equity.calendar.earnings", lambda **_: self.market.earnings
                )
            ),
            ownership=_Node(
                insider_trading=self._endpoint(
                    "equity.ownership.insider_trading",
                    lambda **kw: self.market.insider.get(str(kw.get("symbol")), []),
                )
            ),
        )

    def _endpoint(
        self, name: str, answer: Callable[..., list[dict[str, Any]]]
    ) -> Callable[..., Result]:
        def call(**kwargs: Any) -> Result:
            assert "provider" not in kwargs, "the replica must never pass provider="
            self.calls.append((name, dict(kwargs)))
            self.clock.tick()
            if name in self.market.failing:
                raise RuntimeError(f"{name} failed (scripted)")
            return Result([dict(row) for row in answer(**kwargs)])

        return call

    def _mark_price(self, **kwargs: Any) -> list[dict[str, Any]]:
        symbol = kwargs.get("symbol")
        rows = [dict(row, symbol=s) for s, row in self.market.marks.items()]
        return [r for r in rows if symbol is None or r["symbol"] == symbol]

    def _ticker(self, **kwargs: Any) -> list[dict[str, Any]]:
        row = self.market.tickers.get(str(kwargs["symbol"]))
        return [] if row is None else [dict(row, symbol=kwargs["symbol"])]

    def _kline(self, **kwargs: Any) -> list[dict[str, Any]]:
        return list(self.market.klines.get(str(kwargs["symbol"]), []))

    @staticmethod
    def to_records(result: Result) -> list[dict[str, Any]]:
        return result.rows

    def called(self, name: str) -> list[dict[str, Any]]:
        return [kwargs for n, kwargs in self.calls if n == name]


@dataclass
class LLMResult:
    content: str
    model: str = "runner-model-under-test"
    request_id: str = "req"
    finish_reason: str = "stop"
    usage: dict[str, int] = field(
        default_factory=lambda: {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    )
    raw: dict[str, Any] = field(default_factory=dict)


class LLMBudgetExceededError(RuntimeError):
    pass


class LLMInputError(RuntimeError):
    pass


class LLMRuntimeUnavailableError(RuntimeError):
    pass


class FakeLLM:
    """``getagent.llm``: scripted answers (a string, an ``LLMResult`` or an exception) in order."""

    def __init__(
        self,
        answers: Sequence[Any] = (),
        *,
        available: bool = True,
        clock: Clock | None = None,
        latency: float = 0.0,
    ) -> None:
        self.answers = list(answers)
        self.available = available
        self.calls: list[dict[str, Any]] = []
        self.clock = clock
        self.latency = latency

    def is_available(self) -> bool:
        return self.available

    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResult:
        self.calls.append(
            {
                "messages": [dict(m) for m in messages],
                "system": system,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        if self.clock is not None:
            self.clock.tick(self.latency)
        if not self.answers:
            raise AssertionError("the fake model was called more often than scripted")
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        if isinstance(answer, LLMResult):
            return answer
        return LLMResult(content=str(answer))


class FakeRuntime:
    """``getagent.runtime``: context and signal output, follow-trade routing as documented."""

    def __init__(
        self,
        manifest: Mapping[str, Any],
        *,
        mode: str = "signal_only",
        evaluation: str = "live",
        run_id: str = "run-1",
    ) -> None:
        self.manifest = dict(manifest)
        self.mode = mode
        self.evaluation_mode = evaluation
        self.run_id = run_id
        self.signals: list[dict[str, Any]] = []

    def is_historical(self) -> bool:
        return self.evaluation_mode == "historical"

    def is_live(self) -> bool:
        return self.evaluation_mode == "live"

    def is_follow_trade(self) -> bool:
        return self.mode == "follow_trade"

    def is_signal_only(self) -> bool:
        return self.mode == "signal_only"

    @staticmethod
    def is_actionable_signal(action: str) -> bool:
        return action not in ("watch", "hold", "noop", "none", "")

    def emit_signal(
        self,
        action: str,
        symbol: str = "",
        confidence: float = 0.0,
        metrics: Mapping[str, Any] | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "type": "signal",
            "action": action,
            "symbol": symbol,
            "confidence": confidence,
            "metrics": dict(metrics or {}),
            "meta": dict(meta or {}),
        }
        json.dumps(payload, allow_nan=False)  # the runner writes it as JSON
        self.signals.append(payload)
        return payload

    def emit_signal_or_follow(
        self,
        *,
        action: str,
        symbol: str,
        confidence: float,
        metrics: Mapping[str, Any],
        meta: Mapping[str, Any],
        execute_trade: Callable[[], Any] | None = None,
    ) -> dict[str, Any]:
        payload = self.emit_signal(action, symbol, confidence, metrics, meta)
        payload["followed"] = False
        if (
            self.is_follow_trade()
            and self.is_actionable_signal(action)
            and execute_trade is not None
        ):
            payload["followed"] = True
            payload["trade_result"] = execute_trade()
        return payload


# --- the trade proxy ------------------------------------------------------------------------------


@dataclass
class Selection:
    symbol: str
    hold_side: str
    size: str
    leverage: int
    candidate_count: int
    raw: dict[str, Any]


@dataclass
class VenuePosition:
    qty: Decimal
    entry: Decimal


class FakeVenue:
    """A venue with hedge-mode positions, market fills at a set price, preset stops as plan
    orders. It answers through ``trade.contract`` and ``trade.helpers`` below."""

    def __init__(
        self, prices: Mapping[str, Decimal], *, clock: Clock, price_step: Decimal = Decimal("0.01")
    ) -> None:
        self.prices = dict(prices)
        self.clock = clock
        self.price_step = price_step
        self.long: dict[str, VenuePosition] = {}
        self.short: dict[str, VenuePosition] = {}
        self.plans: dict[str, list[dict[str, Any]]] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.reject: set[str] = set()
        self.ids = itertools.count(1000)

    def _record(self, name: str, **kwargs: Any) -> None:
        self.calls.append((name, kwargs))
        self.clock.tick()

    def called(self, name: str) -> list[dict[str, Any]]:
        return [kwargs for n, kwargs in self.calls if n == name]

    def _book(self, hold: str) -> dict[str, VenuePosition]:
        return self.long if hold == "long" else self.short

    def fill(self, symbol: str, hold: str, qty: Decimal, stop: str) -> dict[str, Any]:
        book = self._book(hold)
        price = self.prices[symbol]
        current = book.get(symbol)
        if current is None:
            book[symbol] = VenuePosition(qty, price)
        else:
            total = current.qty + qty
            current.entry = (current.qty * current.entry + qty * price) / total
            current.qty = total
        order_id = str(next(self.ids))
        if stop:
            self.plans.setdefault(symbol, []).append(
                {
                    "orderId": f"sl-{order_id}",
                    "planType": "loss_plan",
                    "triggerPrice": stop,
                    "holdSide": hold,
                }
            )
        return {"code": "00000", "data": {"orderId": order_id}}


def _trade_module(venue: FakeVenue) -> SimpleNamespace:
    def current_position(symbol: str = "", product_type: str = "USDT-FUTURES") -> dict[str, Any]:
        venue._record("current_position", symbol=symbol)
        rows = []
        for hold, book in (("long", venue.long), ("short", venue.short)):
            for s, p in sorted(book.items()):
                if (not symbol or s == symbol) and p.qty > 0:
                    rows.append(
                        {
                            "symbol": s,
                            "holdSide": hold,
                            "total": str(p.qty),
                            "openPriceAvg": str(p.entry),
                        }
                    )
        return {"code": "00000", "data": rows}

    def find_contract_position(
        result: Mapping[str, Any], symbol: str, hold_side: str = "", prefer_first: bool = False
    ) -> Selection | None:
        rows = [
            r
            for r in result["data"]
            if r["symbol"] == symbol and (not hold_side or r["holdSide"] == hold_side)
        ]
        if not rows:
            return None
        row = rows[0]
        return Selection(symbol, row["holdSide"], row["total"], 1, len(rows), dict(row))

    def contract_price(symbol: str, product_type: str = "USDT-FUTURES") -> Decimal:
        venue._record("contract_price", symbol=symbol)
        return venue.prices[symbol]

    def contract_rules(symbol: str, product_type: str = "USDT-FUTURES") -> SimpleNamespace:
        venue._record("contract_rules", symbol=symbol)
        return SimpleNamespace(price_step=str(venue.price_step))

    def compute_qty(
        symbol: str,
        market: str,
        budget_amount: Any,
        leverage: Any = None,
        price: Any = "",
        product_type: str = "USDT-FUTURES",
    ) -> SimpleNamespace:
        venue._record(
            "compute_qty",
            symbol=symbol,
            market=market,
            budget_amount=budget_amount,
            leverage=leverage,
        )
        notional = Decimal(str(budget_amount)) * Decimal(str(leverage or 1))
        qty = (notional / venue.prices[symbol]).quantize(Decimal("0.0001"), rounding=ROUND_DOWN)
        return SimpleNamespace(qty=str(qty))

    def resolve_contract_tpsl(
        *,
        symbol: str,
        side: str,
        leverage: Any,
        tp_trigger_price: Any = "",
        sl_trigger_price: Any = "",
        reference_price: Any = "",
        product_type: str = "USDT-FUTURES",
    ) -> SimpleNamespace:
        venue._record(
            "resolve_contract_tpsl",
            symbol=symbol,
            side=side,
            sl_trigger_price=sl_trigger_price,
            reference_price=reference_price,
        )
        stop = Decimal(str(sl_trigger_price)).quantize(venue.price_step, rounding=ROUND_HALF_UP)
        return SimpleNamespace(tp_trigger_price="", sl_trigger_price=str(stop))

    def select_sl_plan_order(
        plan_orders_result: Mapping[str, Any], symbol: str = "", prefer_first: bool = False
    ) -> SimpleNamespace:
        rows = [r for r in plan_orders_result["data"] if r["planType"] == "loss_plan"]
        if not rows:
            raise ValueError("no stop-loss plan order")
        return SimpleNamespace(order_id=rows[0]["orderId"], raw=rows[0])

    def open_market(hold: str) -> Callable[..., dict[str, Any]]:
        def send(
            symbol: str,
            qty: Any,
            leverage: Any,
            tp_trigger_price: Any = "",
            sl_trigger_price: Any = "",
        ) -> dict[str, Any]:
            venue._record(
                f"open_{hold}_market",
                symbol=symbol,
                qty=qty,
                leverage=leverage,
                sl_trigger_price=sl_trigger_price,
                tp_trigger_price=tp_trigger_price,
            )
            if symbol in venue.reject:
                return {"code": "40762", "msg": "rejected (scripted)"}
            return venue.fill(symbol, hold, Decimal(str(qty)), str(sl_trigger_price))

        return send

    def close_position(
        symbol: str, hold_side: str, product_type: str = "USDT-FUTURES"
    ) -> dict[str, Any]:
        venue._record("close_position", symbol=symbol, hold_side=hold_side)
        venue._book(hold_side).pop(symbol, None)
        venue.plans.pop(symbol, None)
        return {"code": "00000", "data": {"orderId": str(next(venue.ids))}}

    def plan_pending_orders(
        symbol: str = "", limit: Any = None, product_type: str = "USDT-FUTURES"
    ) -> dict[str, Any]:
        venue._record("plan_pending_orders", symbol=symbol)
        return {"code": "00000", "data": list(venue.plans.get(symbol, []))}

    def modify_stop_loss(
        symbol: str, order_id: str, trigger_price: Any, product_type: str = "USDT-FUTURES"
    ) -> dict[str, Any]:
        venue._record(
            "modify_stop_loss", symbol=symbol, order_id=order_id, trigger_price=trigger_price
        )
        for row in venue.plans.get(symbol, []):
            if row["orderId"] == order_id:
                row["triggerPrice"] = str(trigger_price)
        return {"code": "00000", "data": {"orderId": order_id}}

    def is_success(result: Any) -> bool:
        return isinstance(result, Mapping) and result.get("code") == "00000"

    return SimpleNamespace(
        contract=SimpleNamespace(
            current_position=current_position,
            open_long_market=open_market("long"),
            open_short_market=open_market("short"),
            close_position=close_position,
            plan_pending_orders=plan_pending_orders,
            modify_stop_loss=modify_stop_loss,
        ),
        helpers=SimpleNamespace(
            find_contract_position=find_contract_position,
            contract_price=contract_price,
            contract_rules=contract_rules,
            compute_qty=compute_qty,
            resolve_contract_tpsl=resolve_contract_tpsl,
            select_sl_plan_order=select_sl_plan_order,
        ),
        is_success=is_success,
    )


@dataclass
class Sdk:
    data: FakeData
    llm: FakeLLM
    runtime: FakeRuntime
    trade: SimpleNamespace
    venue: FakeVenue
    clock: Clock


def install_sdk(monkeypatch: pytest.MonkeyPatch, sdk: Sdk) -> None:
    """Make ``from getagent import data, llm, runtime, trade`` resolve to the fakes."""
    package = ModuleType("getagent")
    package.__path__ = []
    for name in ("data", "llm", "runtime", "trade"):
        setattr(package, name, getattr(sdk, name))
    monkeypatch.setitem(sys.modules, "getagent", package)


def load_package(package_dir: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import ``package_dir/src`` as a fresh package (the runner's ``src``) under a unique name.
    Bytecode is not written, and every module is removed from ``sys.modules`` afterwards."""
    name = f"t2sa_replica_{next(_counter)}"
    src = package_dir / "src"
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    spec = importlib.util.spec_from_file_location(
        name, src / "__init__.py", submodule_search_locations=[str(src)]
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    for child in sorted(p.stem for p in src.glob("*.py") if p.stem != "__init__"):
        submodule = importlib.import_module(f"{name}.{child}")
        monkeypatch.setitem(sys.modules, f"{name}.{child}", submodule)
        setattr(module, child, submodule)
    return module


def iter_sdk_modules(package: ModuleType) -> Iterator[str]:
    yield from (n for n in dir(package) if not n.startswith("_"))


# ================================================================================================
# Scenarios
# ================================================================================================

WEDNESDAY_HEARTBEAT = datetime(2026, 9, 23, 16, 0, 20, tzinfo=UTC)
"""A Wednesday funding heartbeat (16:00 UTC), 20 seconds into its cron run."""

PRICES: dict[str, Decimal] = {
    "BTCUSDT": Decimal("100000.0"),
    "SP500USDT": Decimal("6600.0"),
    "NDX100USDT": Decimal("24000.0"),
    "MSTRUSDT": Decimal("330.00"),
    "HOODUSDT": Decimal("120.00"),
    "CRCLUSDT": Decimal("140.00"),
    "SNDKUSDT": Decimal("75.000"),
    "COINUSDT": Decimal("330.00"),
    "TSLAUSDT": Decimal("420.00"),
    "GOOGLUSDT": Decimal("250.00"),
    "METAUSDT": Decimal("760.00"),
    "NVDAUSDT": Decimal("180.00"),
    "AMZNUSDT": Decimal("230.00"),
    "AAPLUSDT": Decimal("255.00"),
}


def klines(
    price: Decimal, now: datetime, *, bars: int = 34, step_bps: float = 20.0
) -> list[dict[str, Any]]:
    """Closed 1H bars ending the hour before ``now``, alternating moves of ``step_bps``."""
    last_open = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    rows: list[dict[str, Any]] = []
    close = float(price)
    series: list[float] = []
    for i in range(bars):
        series.append(close * (1 + (step_bps / 10_000) * (1 if i % 2 else -1) * 0.5))
    for i, value in enumerate(series):
        opened = last_open - timedelta(hours=bars - 1 - i)
        rows.append(
            {
                "time": int(opened.timestamp() * 1000),
                "open": value,
                "high": value * 1.002,
                "low": value * 0.998,
                "close": value,
                "volume": 1.0,
            }
        )
    rows[-1]["close"] = float(price)
    return rows


def market(now: datetime, *, prices: Mapping[str, Decimal] = PRICES) -> Market:
    """A calm, fresh, two-sided market for every universe symbol."""
    stamp = int(now.timestamp() * 1000)
    m = Market()
    for symbol, price in prices.items():
        tick = Decimal("0.1") if price > 1000 else Decimal("0.01")
        m.marks[symbol] = {
            "mark_price": str(price),
            "index_price": str(price),
            "last_funding_rate": "0.0001",
            "time": stamp,
        }
        m.tickers[symbol] = {
            "last": str(price),
            "bid": str(price - tick),
            "ask": str(price + tick),
            "timestamp": stamp,
            "change_percent": 0.5,
        }
        m.klines[symbol] = klines(price, now)
    m.crypto_fear_greed = [
        {
            "date": (now - timedelta(days=1)).date().isoformat(),
            "value": 55,
            "classification": "Greed",
        },
        {"date": now.date().isoformat(), "value": 58, "classification": "Greed"},
    ]
    m.market_fear_greed = [
        {"score": 50, "rating": "Neutral", "previous_close": 49, "timestamp": now.isoformat()}
    ]
    m.long_short = [{"symbol": "BTCUSDT", "long_short_ratio": 1.4, "timestamp": stamp}]
    m.top_position = [{"symbol": "BTCUSDT", "long_short_ratio": 1.1, "timestamp": stamp}]
    m.open_interest = [
        {
            "time": int(
                (now - timedelta(hours=h)).replace(minute=0, second=0, microsecond=0).timestamp()
                * 1000
            ),
            "open_interest": 50_000 + (h % 7) * 10,
        }
        for h in range(0, 30 * 24)
    ]
    return m


def manifest(
    symbols: Sequence[str] | None = None, *, margin_budget: str = "10000"
) -> dict[str, Any]:
    chosen = list(symbols or PRICES)
    return {
        "name": "t2-sentiment-agent-replica",
        "trading_symbols": chosen,
        "strategy_config": {"trading_symbols": chosen, "margin_budget": margin_budget},
    }


def make_sdk(
    now: datetime,
    *,
    answers: Sequence[Any] = (),
    mode: str = "signal_only",
    symbols: Sequence[str] | None = None,
    latency: float = 0.0,
    llm_latency: float = 0.0,
    available: bool = True,
    evaluation: str = "live",
    run_id: str = "run-1",
) -> Sdk:
    clock = Clock(now, latency)
    venue = FakeVenue(PRICES, clock=clock)
    return Sdk(
        data=FakeData(market(now), clock),
        llm=FakeLLM(answers, available=available, clock=clock, latency=llm_latency),
        runtime=FakeRuntime(manifest(symbols), mode=mode, evaluation=evaluation, run_id=run_id),
        trade=_trade_module(venue),
        venue=venue,
        clock=clock,
    )


def target(symbol: str, value: float, **update: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "symbol": symbol,
        "target": value,
        "thesis": f"{symbol} positioning is stretched against the crowd",
        "invalidation": f"{symbol} funding back below reference.zero",
        "horizon_hours": 48,
        "crowd_belief": "the crowd expects the trend to continue",
        "our_view": "fade the crowded side with a small position",
        "confidence": 0.55,
        "evidence": [f"{symbol}.funding_rate"],
        "invalidation_triggered": False,
        "invalidation_evidence": None,
    }
    row.update(update)
    return row


def answer(stance: str, targets: Sequence[Mapping[str, Any]], **update: Any) -> str:
    body: dict[str, Any] = {
        "stance": stance,
        "targets": list(targets),
        "rejected_alternatives": [{"action": "stay flat", "reason": "positioning is stretched"}],
        "mandate_response": "deploy a small part of the risk budget where positioning is crowded",
        "flat_reasons": [],
        "summary": "Fade crowded positioning with small, stopped positions.",
    }
    body.update(update)
    return json.dumps(body)


def run_live(package: ModuleType, sdk: Sdk, state_dir: Path) -> dict[str, Any]:
    env = package.cycle.Env(
        runtime=sdk.runtime, llm=sdk.llm, trade=sdk.trade, clock=sdk.clock, state_dir=state_dir
    )
    record: dict[str, Any] = package.cycle.run_live(env)
    return record


def read_state(state_dir: Path) -> dict[str, Any]:
    state: dict[str, Any] = json.loads((state_dir / "t2sa_replica_state.json").read_text("utf-8"))
    return state
