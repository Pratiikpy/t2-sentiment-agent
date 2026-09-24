"""Keyless GET fetcher for Bitget public market data, live and UTA Demo (paptrading: 1).

Read-only. No account endpoint, no order endpoint, no credential. Caches every response to disk
so the analysis can be re-run without hitting the venue again.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import requests

BASE = "https://api.bitget.com"
CACHE = Path(__file__).parent / "cache"
CACHE.mkdir(exist_ok=True)
S = requests.Session()


def get(path: str, params: dict, demo: bool = False, cache: bool = True) -> dict:
    key = hashlib.sha1((path + json.dumps(params, sort_keys=True) + str(demo)).encode()).hexdigest()
    f = CACHE / f"{key}.json"
    if cache and f.exists():
        return json.loads(f.read_text())
    headers = {"paptrading": "1"} if demo else {}
    for attempt in range(5):
        try:
            r = S.get(BASE + path, params=params, headers=headers, timeout=30)
            d = r.json()
            if d.get("code") == "00000":
                f.write_text(json.dumps(d))
                time.sleep(0.06)
                return d
            if d.get("code") in ("429", "40429"):
                time.sleep(1 + attempt)
                continue
            return d
        except Exception as e:  # noqa: BLE001
            time.sleep(1 + attempt)
            last = str(e)
    return {"code": "ERR", "msg": "retries exhausted"}


def funding_history_v3(symbol: str, category: str, demo: bool, pages: int = 12) -> list[tuple[int, float]]:
    """All funding settlements the venue returns, newest first, paginated by cursor."""
    out: list[tuple[int, float]] = []
    for page in range(1, pages + 1):
        # catalog.ts:107 documents `cursor` as a page number and limit max 200; carry.py found the
        # live API rejects limit > 100 with 40020, so 100 is used.
        p = {"symbol": symbol, "category": category, "limit": "100", "cursor": str(page)}
        d = get("/api/v3/market/history-fund-rate", p, demo=demo)
        data = d.get("data")
        rows = data.get("resultList") if isinstance(data, dict) else data
        if not rows:
            break
        for x in rows:
            out.append((int(x["fundingRateTimestamp"] if "fundingRateTimestamp" in x else x["fundingTime"]), float(x["fundingRate"])))
        if len(rows) < 100:
            break
    out = sorted(set(out))
    return out


def funding_history_v2(symbol: str, product: str, pages: int = 12) -> list[tuple[int, float]]:
    out = []
    for page in range(1, pages + 1):
        d = get("/api/v2/mix/market/history-fund-rate", {"symbol": symbol, "productType": product, "pageSize": "100", "pageNo": str(page)})
        rows = d.get("data") or []
        if not rows:
            break
        out += [(int(x["fundingTime"]), float(x["fundingRate"])) for x in rows]
        if len(rows) < 100:
            break
    return sorted(set(out))


def candles_v2(symbol: str, product: str, gran: str = "1H", n_pages: int = 20) -> list[list[float]]:
    """History candles, walking back with endTime. Returns [ts, o, h, l, c, vol] ascending."""
    out: dict[int, list[float]] = {}
    end = None
    for _ in range(n_pages):
        p = {"symbol": symbol, "productType": product, "granularity": gran, "limit": "200"}
        if end:
            p["endTime"] = str(end)
        d = get("/api/v2/mix/market/history-candles", p)
        rows = d.get("data") or []
        if not rows:
            break
        for r in rows:
            out[int(r[0])] = [int(r[0])] + [float(v) for v in r[1:6]]
        end = min(int(r[0]) for r in rows) - 1
    return [out[k] for k in sorted(out)]


def candles_v3(symbol: str, category: str, demo: bool, interval: str = "1H", n_pages: int = 20, kind: str | None = None) -> list[list[float]]:
    out: dict[int, list[float]] = {}
    end = None
    for _ in range(n_pages):
        p = {"symbol": symbol, "category": category, "interval": interval, "limit": "100"}
        if kind:
            p["type"] = kind
        if end:
            p["endTime"] = str(end)
        d = get("/api/v3/market/history-candles", p, demo=demo)
        rows = d.get("data") or []
        if not rows:
            break
        for r in rows:
            out[int(r[0])] = [int(r[0])] + [float(v) for v in r[1:6]]
        end = min(int(r[0]) for r in rows) - 1
    return [out[k] for k in sorted(out)]
