# Validation evidence

The measurements every number in `src/sentiment_agent/policy.py` answers. All of it comes from
keyless public GETs: no account endpoint, no order, no credential. Demo reads carry the
`paptrading: 1` header on public v3 market endpoints, which returns UTA Demo market data.

## `demo_venue/`

The scripts are the as-run originals from the plan that chose this agent's design, changed only so
their input paths point at this folder (`wb/` and `fetch.py` beside them). Re-running them needs
`requests` and refetches recent candles, so the window moves and the numbers will drift.

| File | Shows | Used by |
|---|---|---|
| `weekend_path.py` → `weekend_vol.json` | UTA Demo does not move the 11 US-equity perps on weekends; live does | G2, G1 stale-index rule |
| `fade_demo2.py` → `fade_demo2.txt` | Per weekend, Saturday 00:00 → 12:00 UTC moves on Demo last, Demo mark and live. Demo mostly freezes (MSTR +5 bps vs +204 live, 2026-09-19) and sometimes jumps on its own (AAPL +80 bps vs +17 live, 2026-09-12) | G2 |
| `fade_demo.py` → `fade_demo.json` | Weekend fade on live vs Demo prices; per-name Demo tracking (`tracking`: close-gap p99, mark-index max) | G1 gap thresholds for the equity perps |
| `demo_integrity.py` → `demo_integrity.json` | Demo mark-vs-index excursions over 3% and Demo-live gap p99 for crypto, SP500, NDX100 | G1 thresholds, the excluded list |
| `envelope_clean.py` → `envelope_clean.json` | What a capped book with no edge produces over 72h (17,400 windows) | G3, G5, G10, the expected-numbers envelope in the genesis |
| `envelope.py` → `envelope.json` | The same envelope with ETH/SOL/XRP/DOGE included, for comparison | Context only |
| `fetch.py` | The keyless fetcher the scripts share | — |
| `demo_survey.json` | Demo vs live funding and book survey (plan 1) | G8 spread bound, funding caveats |
| `universe_probe.json` | Demo and live `instruments` + `tickers` for the 14 universe names, 2026-09-24 10:31 UTC: Demo taker fee 0.0006, `minOrderQty`, `maxMarketOrderQty`, spreads | G7, G8, G11 |
| `wb/` | Live 1H candles for 43 US-equity perps, the input to the scripts above | — |

`tests/contract/test_policy_evidence.py` reads these files and fails if a number in `policy.py`
or a figure quoted in a guard's basis stops matching the measurement it cites.
