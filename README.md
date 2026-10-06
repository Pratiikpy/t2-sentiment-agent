# t2-sentiment-agent

**Qwen decides. A risk kernel that can only reduce stands between it and the venue. Bitget's own
Agent Hub places every order on Bitget Demo. One hash-chained log proves all of it.**

A Market Sentiment Agent for Bitget AI Base Camp S2, Track 2 (Agentic Trading). It reads what the
crowd is doing (funding, long/short skew, open interest, Fear & Greed, news and screened social
text) on 14 Bitget perpetuals, and `qwen3.8-max` decides the whole book at once: where to fade a
crowded trade, hedge, cut, or stay flat with written reasons. Every decision carries a thesis, an
invalidation and "what the crowd believes vs what we do", and every number in it must resolve to a
logged fact.

**Live record: https://t2-sentiment-agent-run2.vercel.app** — every decision card, the kernel's
ruling guard by guard, the venue order ids, and the same model ungoverned beside it.

## The record, in numbers

Run 2, Bitget UTA Demo, pre-registered window 2026-09-28 00:00 to 2026-10-07 00:00 UTC. Figures
below are the 2026-10-05 05:04 UTC export; the page republishes hourly from the log and
`scripts/recompute.py` recomputes every one.

| | |
|---|---|
| Decisions by the model | **46** — 17 act, 10 hold, 15 flat with reasons, 4 model outages (no rule traded in its place) |
| Orders sent to Bitget Demo through Agent Hub | **95**, each previewed with `--dry-run` first; 22 fills, 7 closed trades |
| Return / max drawdown | **−0.02% / −0.37%** (90% bootstrap band −0.61% to +0.64%; three days of hourly marks cannot separate skill from luck, and the page says so) |
| Sharpe (annualised) / win rate | −0.31 ± 7.8 / 28.6% on 7 trades — descriptive, not inferential |
| What the kernel did | changed **20 of 46** decisions; 72 protective rulings; 0 human takeovers |
| The same model, ungoverned | **34.1%** of its drafts broke a guard; the kernel's cuts prevented 0.039% of equity in losses and forwent 0.026% in gains |
| Fixed-rule crowd fade, no model, same clock | **−0.65%**, 69 trades |
| Coin flip, 1,000 seeds, same venue | median **−0.24%** |
| Rival sentiment agents, same snapshots, same simulator | lexicon trader −0.21% (with the crowd) and −0.09% (against it); S2 sentiment-fusion entry −0.03%; S2 Fear & Greed entry never traded |
| Same fills marked at live prices | +0.13%, largest hourly Demo-to-live gap 8.5bps: the Demo result is not a sandbox artefact |

The quantitative half of the score is flat and we say so. What the record shows is the half that
is hard to fake: a model that stands aside 15 times with reasons, a kernel that cut a third of the
drafts that would have broken a rule, nothing added by any deterministic code, and every line
reproducible from the log.

## Event → decision → execution, one card

[Open a decision card](https://t2-sentiment-agent-run2.vercel.app) from the timeline. Each shows,
in order: what woke the agent and which sources answered; the text the model saw, with quarantined
items marked; Qwen's thesis, invalidation and crowd-vs-us; every guard's ruling, with its ceiling
and the measurement it rests on; the `--dry-run` payload, the `clientOid`, the venue `orderId` and
the fills; and the ledger rows and blob hashes that prove each line.

## Why this design

- **The model is the decision-maker, and nothing else is.** Every target comes from Qwen. The
  kernel's rules are types, not conventions: a ruling that adds exposure, turns a side, or cuts a
  weight without naming the guard cannot be constructed (`types.py`, `tests/contract/`).
- **Eleven guards, each answering a measured failure of the venue.** UTA Demo freezes US-equity perps
  at weekends, its mark price has jumped 5.4% from its index, and its spreads are wider than live;
  each guard cites the measurement in `validation/` and a test fails if the policy drifts from it.
- **Pre-registered before the first order.** The policy, prompts, code commit, metric definitions
  and the no-edge envelope are hashed into a genesis posted on X before trading, with every change
  against run 1 declared (d1–d9, a1–a8). Ledger heads are Ed25519-signed and anchored to Bitcoin
  through OpenTimestamps; five are already block-confirmed.
- **Bitget's own tools, measured.** Agent Hub (`bgc`) sends every order; `bitget-signal` and
  `bitget-mcp-server` were wired in and have failed on Bitget's side for most of both runs, so the
  agent reads their upstreams on a labelled surface and logs every failed call as failed.

## Honest limits

- The return is indistinguishable from zero and the Sharpe band spans −15 to +8. Nine days is not a
  track record.
- The red team of the sentiment input (HeyArka and AgentDojo attack strings, a coordinated pump) is
  built and tested but was not run on this export; its grade is published whatever it shows.
- The genesis's own OpenTimestamps stamp failed on a client bug (recorded as failed, not hidden);
  later ledger heads are anchored.
- Both Bitget data services were down for most of the run; the agent's view of the crowd came
  mostly from the upstreams they wrap.

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
