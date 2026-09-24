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

**Paper run live on Bitget Demo since 2026-09-24 17:06:23 UTC.** Genesis hash
`ebbf607a8fe199bccc0f612ededf2a6902e4884a38d21fbfea498dff7da453a5`, timestamped with
OpenTimestamps, pins the code commit, the dependency locks, the policy and the metric definitions
before the first decision. Starting equity 49,999.98 USDT. The agent ticks every 30 seconds and
decides on pre-registered heartbeats (the US open, each funding settlement) and on event triggers.

- Live record, republished hourly from the log: https://t2-sentiment-agent-live.vercel.app
- Check it yourself: `t2sa verify` (hash chain and head anchor), `t2sa status`, and
  `scripts/recompute.py` for every published figure.
- Known gap, disclosed rather than patched under a live run: `bitget-signal`'s hosted server
  returns empty envelopes for `news_feed` and `reddit_trending` (reproduced through a second MCP
  client, while their upstreams answer directly). Mood and positioning come from
  `bitget-mcp-server`, which answers. Every failed call is in the log as failed.

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
