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

Pre-genesis. Every module in [DESIGN.md](DESIGN.md) §18 is built, wired and tested, and a full
cycle has run end to end in dry-run mode on live public Bitget data (perception, decision, the
kernel's eleven checks, a `bgc --paper-trading --dry-run` order preview, the ledger, the decision
card and the demo page). No paper order has been placed and no live model call has been made. The
paper log starts when the Demo API key exists: [RUNBOOK.md](RUNBOOK.md) is the owner's procedure,
and `t2sa go-live` is the one command that starts it.

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
