# RUNBOOK — starting, checking and stopping the paper run

This is the owner's procedure for the scored Track 2 log: the agent trading on Bitget UTA **Demo**
with paper orders placed through Bitget Agent Hub (`bgc --paper-trading`). Everything here runs
from the repository root, in PowerShell.

```powershell
cd <the repository root>
.venv\Scripts\Activate.ps1          # gives the `t2sa` command
```

## 0. Before the key exists (done, and repeatable at any time)

| Check | Command | State on 2026-09-24 |
|---|---|---|
| Lint, format, strict types, all tests | `python scripts/check_all.py` | passing; the skipped tests need optional extras or the keyless network (`--run-live-public`) |
| Keyless network tests | `python -m pytest -m live_public --run-live-public` | 14 passing |
| Keyless preflight: public data, Agent Hub installed and intact, a `--dry-run` order preview for all 14 symbols, the toolkit | `t2sa preflight` | passing |
| One whole cycle on live data, nothing sent | `t2sa decide --mode dryrun --llm scripted --script examples/scripted/decision.json --reason "rehearsal" --symbols BTCUSDT` | snapshot → decision → 11 guards → planned and previewed order → ledger → card |

A dry-run or simulated record exports to `var/public-dryrun/` or `var/public-simulated/`, never to
`public/`, and every event in it carries its mode, so no rehearsal can enter the scored log.

## 1. What only the owner does first

1. **Create the Demo API key.** On bitget.site, switch to Demo trading, create an API key with
   Read and Trade permission and **no Withdraw**. Write this file (it is git-ignored):

   ```
   # .secrets/demo.env
   BITGET_KEY_ENVIRONMENT=demo
   BITGET_API_KEY=...
   BITGET_SECRET_KEY=...
   BITGET_PASSPHRASE=...
   ```

   The agent refuses to start unless the file sits inside this project and declares
   `BITGET_KEY_ENVIRONMENT=demo`. It never reads any other credential file, and it strips every
   inherited `BITGET_*` variable from the environment it hands to `bgc`.

2. **Put the Qwen key in `.secrets/qwen.env`** (also git-ignored):

   ```
   BITGET_QWEN_API_KEY=...
   # optional, these are the defaults:
   BITGET_QWEN_BASE_URL=https://hackathon.bitgetops.com/v1
   BITGET_QWEN_MODEL=qwen3.8-max
   ```

   The daily cap is 2,400,000 tokens (`policy.decision.daily_token_cap`), sized so the busiest day
   the trigger rules allow fits under the budget's own byte-level projection (the arithmetic is in
   `policy.decision.basis`). Event decisions are refused (`budget`) before they could eat into the
   heartbeats still due that day. Reaching the cap anyway is a model outage: the kernel flattens
   the book, and nothing trades in the model's place. Before genesis, read the `prompt_tokens`
   of the first live calls (each decision's `BUDGET_STATE` and attempt blobs) and confirm they
   stay under the projection; if they do not, the cap is re-derived before the policy is frozen.

3. **Make the code a git commit.** The genesis records the exact commit that runs. Initialise the
   public repository and commit (`git init`, add, commit), or pass `--commit <full sha>` below.

4. **Free disk.** `go-live` refuses to start with less than 5 GB free on the project's disk. The
   run writes about 0.2 GB a day (measured on the dry run); a full disk stops the ledger mid-run.
   On 2026-09-24 the C: drive had between 0 and 2.5 GB free, so this needs doing.

5. **Keep the machine awake** for the whole window: `powercfg /change standby-timeout-ac 0`, and
   leave it on mains power. A sleeping PC is a stopped agent.

6. *Optional, before step 7 only:* **the plumbing test**, one minimum-size BTCUSDT Demo buy with
   its preset stop, then its close, to pin the venue's response shapes before the scored log
   begins. It is logged to the pre-genesis ledger and disclosed, never scored.

   ```powershell
   t2sa plumbing-test --owner-approved
   ```

## 2. Start the paper run: one command

```powershell
t2sa go-live
```

It does, in order, and stops at the first failure:

1. checks both key files and the free disk; nothing is logged or sent before this;
2. **proves the environment** and logs the proof: the key is declared Demo; the installed Agent Hub
   is the pinned 3.0.0 whose SDK sends `paptrading: 1`; a Demo account read succeeds with no error
   40099; the same key is **rejected** by the live environment (a `--read-only` read, which cannot
   write). Anything else, and any 40099, refuses with exit code 3 and sends nothing;
3. **writes the genesis** (seq 0 of `var/ledger/paper.jsonl`): policy, prompt hashes, universe,
   metric definitions, code commit, lockfile hashes, frozen open-interest thresholds, the expected
   envelope; stamps it with OpenTimestamps; **prints the X post text**;
4. runs the loop with live Qwen, a tick every 30 seconds, until Ctrl+C.

**Post the printed X post** (it carries the genesis hash) before the first decision. The first
decision comes at the next heartbeat (the US open, 13:30 UTC, or a funding settlement at 00:00,
08:00 or 16:00 UTC) or an event trigger, not at once.

The same command restarts a stopped run: with a genesis present it never writes another, it
re-proves the environment before anything can be sent, and it rebuilds the book, the breaker, the
day's token spend and every order state from the ledger.

## 3. Verify it is trading on Demo, and only Demo

| What | How |
|---|---|
| The proof passed | `t2sa status --mode paper` says "latest environment proof passed"; every proof, with each check's result, is in `public/environment.json` after an export |
| Every order carried `--paper-trading` | each `order_submitted` event in `var/ledger/paper.jsonl` holds the exact argv; the executor refuses any preview whose argv lacks the flag |
| The orders exist on the Demo account | `python scripts/verify_orders.py --public public/orders.json` (read-only: one `order detail --paper-trading` per order) |
| The Demo UI agrees | bitget.site, Demo mode: the same orders, positions and preset stops |
| The log is intact | `t2sa verify --mode paper` (hash chain, blobs, head anchor) |
| A decision reproduces | `t2sa replay --mode paper --decision dec-...` (keyless: prompt hash, ruling and order intents recomputed) |

A 40099 on any call during the run, or any sign of the live environment, stops the loop with exit
code 3 and sends nothing further.

## 4. Watch it

```powershell
t2sa status --mode paper              # from the ledger and var/health/paper.json
t2sa reconcile --mode paper           # a read-only sweep against the venue, any time
t2sa decide --reason "..."            # an owner-requested decision; logged and published as an intervention
```

The loop exports to `public/` every hour and stamps the ledger head daily.

## 5. Stop it

**Ctrl+C** in the window running `go-live`. The loop finishes its current tick, writes a final
export and exits. Open positions stay open **with their venue-side stop-loss orders**, which Bitget
executes whether or not the agent runs. Restart with `t2sa go-live`.

To close everything by hand, use the Demo UI on bitget.site; the next reconciliation records those
fills as "not planned by the agent", and the published record shows them. There is deliberately no
command that trades without a model decision.

If the process is killed instead (a crash, a closed window), nothing is lost but the step in flight:
an order sent but not yet confirmed becomes `UNKNOWN` and is resolved only by reconciliation, never
by sending it again.

## 6. What gets submitted

Everything a judge reads is computed from the paper ledger, and all of it is in `public/`:

- `public/index.html`: the demo page (event → decision → execution for every cycle, equity against
  every arm, the guard funnel, governed vs ungoverned, the live mirror, the red team, the Bitget
  toolkit coverage, the proof, the replay). Static, no external requests, light and dark, readable
  at 400 px.
- `public/ledger.jsonl`, `public/blobs/`: the paper-trading log itself, every input and output.
- `public/cards/`: one decision card per decision, with venue order ids.
- `public/orders.json`, `public/trades.csv`, `public/equity_hourly.csv`, `public/metrics.json`.
- `public/verify.md` and `scripts/recompute.py`: anyone recomputes every published number.

For the form (handbook, Track 2): the public repository, the published `public/` page, the X post
with the genesis hash, and the demo video all go in the single "Submission Materials Link" field;
tick "Apply for Demo Day". Refresh `public/` before submitting:

```powershell
t2sa export --mode paper
python scripts/recompute.py public
t2sa verify --mode paper
```

## 7. Exit codes

0 success · 1 a check failed · 2 a refused precondition or usage error · 3 the environment was
refused (40099, a live key, no Demo credentials) · 4 the pre-registration does not match (no
genesis, or the loaded policy is not the one in force) · 5 another instance of the mode is running.
