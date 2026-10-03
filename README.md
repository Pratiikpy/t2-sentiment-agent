# t2-sentiment-agent

A market-sentiment trading agent for Bitget AI Base Camp S2, Track 2 (Agentic Trading), sub-theme
Market Sentiment Agent.

- **Qwen decides.** `qwen3.8-max` reads live crowd positioning, Fear & Greed, news and screened
  social text, and proposes a target book with a thesis, an invalidation and "what the crowd
  believes vs what we do" for every position. "Flat, with reasons" is a valid answer.
- **A risk kernel that can only reduce** stands between the decision and the venue: eight
  pre-registered guards, each answering a measured failure of the paper venue, plus three structural
  checks. It can shrink or refuse what the model asks for; it can never add to it.
- **Bitget's own Agent Hub places every order** (`bgc --paper-trading`) on Bitget UTA Demo, each one
  previewed with `--dry-run` first and reconciled against the venue's order history.
- **One hash-chained log** records every input, decision, ruling, order and fill. Everything
  published is computed from it, and `scripts/recompute.py` lets anyone check the numbers.

## Status

**Run 2 is the scored record.** It has traded on Bitget UTA Demo since **2026-09-28 00:07 UTC**
from this code (commit `5c838fd`), over a pre-registered scoring window that closes
**2026-10-07 00:00 UTC**. Genesis hash
`fb66b6bf40e9229407cdd62266cfe180994c38eb96f18d5c8bca983323ccd3c0` pins the code, the dependency
locks, the policy and the metric definitions before the first decision, and declares, against run
1, every change and the evidence for it (`src/sentiment_agent/run2.py`; table in
[DESIGN.md](DESIGN.md)): nine code and prompt fixes (d1-d9) and eight policy amendments (a1-a8).

- Live record, republished hourly from the log: **https://t2-sentiment-agent-run2.vercel.app** —
  every decision card, the kernel's ruling per guard, the governed book against the same model
  ungoverned, a fixed-rule crowd fade, rival sentiment agents, BTC held, flat, and a 1,000-seed
  coin-flip band.
- **Run 1** (2026-09-24 17:06 UTC to 2026-09-27) stays published as it closed:
  https://t2-sentiment-agent-live.vercel.app.
- What the data services answered: Bitget's own `bitget-signal` and `bitget-mcp-server`
  `do_query` have failed on Bitget's side for most of both runs; the agent reads their upstreams
  (Binance, alternative.me) itself on a labelled surface when both are empty (run2-d3), and every
  failed call is in the log as failed.

### Verify it in 60 seconds

No install for the first two; plain Python for the third.

```bash
R=https://t2-sentiment-agent-run2.vercel.app
# 1. The scored metrics and counts, computed from the log
curl -s "$R/summary.json" | jq '{scoring_window, metrics: (.metrics | {total_return, sharpe_ann, max_drawdown, win_rate, n_closed_trades}), counts}'
# 2. The genesis hash, and the ledger heads already confirmed in a Bitcoin block (OpenTimestamps)
curl -s "$R/genesis.json" | jq '{hash, bitcoin_confirmed: [.anchors[] | select(.record.status == "upgraded") | .seq]}'
# 3. Every head of the record, signed with the operator's Ed25519 key, checked against the key
#    committed here and against the published chain (git clone this repository first)
python scripts/verify_heads.py "$R" validation/run2/signed_heads.json
```

The full check of the chain itself, every figure on the page recomputed from the log:
`t2sa verify`, `t2sa status` and `scripts/recompute.py`.

## From event to order

```
market / crowd / calendar  ->  trigger (pre-registered; cooldowns, daily cap)
        -> snapshot (every figure logged, with its source and health)
        -> Qwen decides: target book, thesis, invalidation, what the crowd believes vs our view
        -> contract (schema, universe, fact-cited invalidation) and grounding (every number
           resolves to a logged fact)
        -> risk kernel: 11 guards that can only shrink or refuse, each ruling logged per guard
        -> Agent Hub: `bgc --dry-run` preview, then the order on Demo with a preset stop
        -> reconciliation: fills, positions, stops, funding, equity against the venue
        -> hourly mark -> published record -> scripts/recompute.py
```

Each step is a ledger event; a decision card on the live record shows one decision through all of
them.

[RUNBOOK.md](RUNBOOK.md) is the operator's procedure; `t2sa go-live` is the one command that
starts or resumes the run.

## Safety model

- Orders go only to Bitget's Demo environment. Every order the transport can build carries
  `--paper-trading`; the agent proves the key is a Demo key (accepted by Demo, rejected by live)
  before the first order, and exits on error 40099.
- Credentials are read from one place, `.secrets/demo.env` in this repository (git-ignored), and
  passed only to the Agent Hub child process.
- Tests never touch the network, never start a child process other than Python, never see a
  credential and never call the model (`tests/conftest.py`).
- The risk kernel's rules are types, not conventions: a ruling that adds exposure, turns a side,
  keeps a weight above a ceiling it reports, or cuts a weight without naming the guard that did it
  cannot be constructed (`src/sentiment_agent/types.py`, `tests/contract/`).

## Run the checks

```bash
pip install -e ".[dev]"
python scripts/check_all.py            # ruff, ruff format, mypy --strict, pytest
```

## Layout

See [DESIGN.md](DESIGN.md): architecture, data flow, every guard and its measured basis, the event
triggers, the file layout and the module specifications. The measurements behind the policy are in
[`validation/`](validation/README.md).

## Licence and attributions

MIT (see [LICENSE](LICENSE)). Parts are ported from the author's ARGUS project, also MIT; third-party
material and what was taken from it are listed in [NOTICE.md](NOTICE.md).
