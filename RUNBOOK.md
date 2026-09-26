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

## Run 2

Run 2 is a second paper run with its own ledger and genesis, from the `run2-prep` code. What it
changes against run 1, and why, is declared in its genesis (DESIGN.md §24): the crowd and calendar
trigger kinds are evaluated on the full snapshot (`run2-d1`), a failing or hollow source is an
alarm (`run2-d2`), a reading both Bitget data services leave empty is read from the upstream they
wrap (`run2-d3`), and policy v2 extends the funding z-score trigger to every instrument
(`run2-a1`). Nothing else changes.

**Timing.** Run 1's window ends **2026-09-27 17:06 UTC** (a Sunday; its US-equity and index legs
have been flat since Friday 20:00 UTC under G2). Run 2 starts **Monday 2026-09-28 00:00 UTC**, when
the weekend freeze ends. Its 72-hour window runs to Thursday 2026-10-01 00:00 UTC. Started at
00:00, the loop's first scheduled decision is the 08:00 UTC funding heartbeat (the 00:00 one falls
before the start); an event trigger can wake it earlier.

Two roots are involved, written below as `<run 1 root>` (the checkout run 1 is running from) and
`<run 2 root>` (a checkout of the `run2-prep` commit, for example a git worktree). Run 2 never runs
from run 1's root: a root holds one paper ledger, and its genesis is written once.

### 1. Close run 1 (after 2026-09-27 17:06 UTC)

In run 1's windows, in this order: stop the watchdog (`scripts\watchdog.ps1`) first with Ctrl+C,
otherwise it restarts the agent; then Ctrl+C the `t2sa go-live` window; then stop the site
publisher after its next publish. Since 2026-09-25 run 1's publisher runs from run 2's checkout
(`<run 2 root>\scripts\publish_site.ps1 -Root <run 1 root>`), because the fixed script uploads the
record as one archive: run 1's record passed Vercel's 15,000-file limit and the old script kept
logging "published" while every upload was rejected. Since 2026-09-26 the same script also copies
`public/` under the export lock and deploys only a finished record (`site.log` says "not
published: the copy is not one finished export" when it refuses one): run 1's hourly export had
died twice mid-move while the old copy held its files open, and a mixture of the new ledger and the
old summary went up.

If BTCUSDT is still open (the only leg G2 allows over a weekend), close it by hand in the Demo UI on
bitget.site, and record the close in run 1's ledger with a read-only sweep. Run 2 must start flat:
a venue position its own ledger does not know is a `position_mismatch`, and G10 would refuse every
increase until it cleared.

```powershell
cd <run 1 root>
.venv\Scripts\Activate.ps1
t2sa reconcile --mode paper          # read-only: the manual close becomes a logged fill
t2sa export --mode paper
python scripts/recompute.py public
t2sa verify --mode paper
t2sa status --mode paper             # "0 open position(s)"
```

Run 1's `public/` and its hosted page stay as they are: they are run 1's record.

### 2. Prepare run 2's root (any time before the start)

```powershell
git worktree add <run 2 root> run2-prep     # or a fresh clone checked out at run2-prep
cd <run 2 root>
git status                                  # clean: the genesis records the commit HEAD names
uv sync --extra dev --extra anchor --frozen
.venv\Scripts\Activate.ps1
t2sa setup                                  # npm ci the pinned Agent Hub CLI into tools/agent-hub
python scripts/check_all.py                 # lint, format, strict types, all tests
t2sa preflight                              # keyless: data sources, bgc, dry-run previews, hashes
```

Put the same two key files in place as for run 1 (RUNBOOK §1): `.secrets/demo.env` (the Demo key,
`BITGET_KEY_ENVIRONMENT=demo`) and `.secrets/qwen.env`, inside `<run 2 root>`. The agent reads
credentials only from its own root. Push `run2-prep` (or merge it) so that the commit named in the
genesis is public.

Rehearse one full cycle against live data with nothing sent, and look at what the new pieces show:

```powershell
t2sa decide --mode dryrun --llm scripted --script examples/scripted/decision.json --reason "run 2 rehearsal" --symbols BTCUSDT
t2sa export --mode dryrun               # var/public-dryrun/: feeds.json, the Feed health section, the card
t2sa status --mode dryrun               # the "feeds as of ..." line
```

### 3. Start run 2 (Monday 2026-09-28 00:00 UTC)

```powershell
cd <run 2 root>
.venv\Scripts\Activate.ps1
t2sa go-live
```

It proves the environment, writes run 2's genesis and prints, beside the policy hash
(`policy-v2`), the run it follows (run 1's genesis hash) and the declared changes
`run2-d1 (code_fix), run2-d2 (observability), run2-d3 (code_fix), run2-a1 (policy_amendment)`;
then it prints the X post
with run 2's genesis hash. **Post it before the first decision.** Then record it, from any
window, while the agent runs (no restart):

```powershell
t2sa x-posted --url https://x.com/<handle>/status/<id>
```

It appends an owner note to the ledger, refuses a post whose id says it was made before the
genesis (the wrong post), and the next export shows it on the page with the time the post's own id
encodes and whether that was before the first order. Until then the page calls the text a draft.

To keep it alive unattended, start the watchdog instead of `t2sa go-live`: it runs the same command
(so it writes the genesis on its first start) and restarts it after any exit but a refusal; the X
post is then in `var/logs/run-*.log`. Start the site publisher in its own window, under run 2's own
page name:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\watchdog.ps1
powershell -ExecutionPolicy Bypass -File scripts\publish_site.ps1 -Project t2-sentiment-agent-run2
```

`-Project` gives run 2 its own hosted page, so it never overwrites run 1's. The publisher takes
`var/run/export-paper.lock` while it copies, so an hourly export waits out the copy (up to 30
minutes) instead of failing; one log line per hour says what it did: `published`, `not
redeployed` (no new record since the last deploy), or `not published` with the reason.

### 4. Check it

Everything in §3-§5 applies unchanged. In addition:

| What | How |
|---|---|
| The genesis declares run 1 and the three changes | `public/genesis.json` (`genesis.predecessor`, `genesis.declared_changes`); the Proof section of the page |
| The policy in force is v2 | `t2sa status --mode paper`: the genesis line says it matches the loaded policy |
| Failing sources, now and since when | `t2sa status --mode paper` (the "feeds as of" line); `var/health/paper.json` `detail`; `public/feeds.json`; the page's Feed health section |
| Why a trigger kind could not fire | the same report's blind kinds, and each decision card's "Feed health on this snapshot" |
| Equity funding events | `trigger` events of kind `funding_zscore` naming equity or index perps; refused ones say why (`cooldown`, `daily_cap`, `weekend_freeze`, `budget`) |

To recompute the run-1 figures the genesis declares, from run 1's published record:

```powershell
python scripts/replay_triggers.py <run 1 root>\public --until-seq 363 --out validation/run2/run1_trigger_replay.json
```
