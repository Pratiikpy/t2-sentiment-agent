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

**Run 1** has traded on Bitget Demo since 2026-09-24 17:06:23 UTC. Genesis hash
`ebbf607a8fe199bccc0f612ededf2a6902e4884a38d21fbfea498dff7da453a5`, timestamped with
OpenTimestamps, pins the code commit, the dependency locks, the policy and the metric definitions
before the first decision. Starting equity 49,999.98 USDT. No order was placed in it. The agent ticks every 30 seconds and decides on pre-registered heartbeats (the
US open, each funding settlement) and on event triggers.

**Run 2** is prepared on this branch and not started. Its genesis will declare, against run 1,
every change and the evidence for it (`src/sentiment_agent/run2.py`; table in
[DESIGN.md](DESIGN.md)): six code and prompt fixes (d1-d6, among them crowd and calendar triggers
that could never fire in run 1, a declared invalidation that must cite a fact, and funding counted
in the scored equity) and seven policy amendments (a1-a7: funding triggers on every name with a
level floor, net and crypto-beta caps, a pre-registered scoring window closed by G2, a lapsing
losing-streak trip, an outage that flattens only on the third failure, and stops set at twice each
name's Demo-live drift at the same loss per name).

- Live record, republished hourly from the log: https://t2-sentiment-agent-live.vercel.app
- Check it yourself: `t2sa verify` (hash chain and head anchor), `t2sa status`, and
  `scripts/recompute.py` for every published figure.
- What the data services actually answered in run 1 (`validation/run2/run1_feed_outage.json`,
  recomputed from its ledger): Bitget's public market API 38,617 of 38,733 calls;
  `bitget-mcp-server` 1,592 of 2,808, and from 2026-09-25 08:33 UTC only 144 of 1,356;
  `bitget-signal`'s hosted server 0 of 939, every answer an empty envelope while its upstreams
  answered directly. Run 2 reads those upstreams itself on a labelled surface when both services
  are empty (run2-d3), and every failed call is in the log as failed.

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
