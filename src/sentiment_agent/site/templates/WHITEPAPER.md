# t2-sentiment-agent: A Market Sentiment Agent Whose Every Order Can Be Audited

*Whitepaper. Bitget AI Base Camp, Season 2. Track 2, Agentic Trading, sub-theme Market Sentiment
Agent. Every figure is read from the published record of run 2 (export of 2026-10-05 05:04 UTC,
ledger head `6e549b63…`) or from a file in this repository, and is named where it is used.*

**Live record:** https://t2-sentiment-agent-run2.vercel.app · **Proof deck:** `/deck.html` on the
same site · **Code:** https://github.com/Pratiikpy/t2-sentiment-agent

---

## 1. Abstract

**In one line: a language model makes every trading decision; a risk kernel that can only reduce
stands between it and Bitget; and every step is in one hash-chained log that anyone can recompute.**

t2-sentiment-agent reads what the crowd is doing on 14 Bitget USDT perpetuals (BTC, the S&P 500
and Nasdaq-100 indices, and 11 US-stock perps: MSTR, HOOD, CRCL, SNDK, COIN, TSLA, GOOGL, META,
NVDA, AMZN, AAPL): funding against its 90-settlement norm, long/short skew, open interest, taker
flow, the Fear & Greed index, news and screened social text. `qwen3.8-max` decides the whole book at
once: where to fade a crowded trade, hedge, cut, or stay flat with written reasons. Every target
carries a thesis, an invalidation tied to a logged fact, a horizon of at least 24 hours and "what
the crowd believes vs what we do".

Between that decision and the venue stands a kernel of eleven guards, each answering a failure of
Bitget's Demo venue that was measured before the guard was written. The kernel can shrink, refuse
or close; it cannot add, turn a side, or change a weight without naming the guard that did it,
because the ruling type refuses to be constructed otherwise. Bitget's own Agent Hub CLI sends every
order to Bitget UTA Demo, previewed with `--dry-run` first.

Run 2 traded from 2026-09-28 00:07 UTC under a policy, prompt set, code commit and metric
definitions hashed into a genesis before the first decision. At the export: **46 model decisions**
(17 act, 10 hold, 15 flat with reasons, 4 outages where no rule traded in its place), **95 orders
sent through Agent Hub** (22 fills, 7 closed trades), **the kernel changed 20 decisions and issued
72 protective rulings, 0 human takeovers.** Return −0.02%, max drawdown −0.37%, Sharpe −0.31 with
a standard error of 7.8: indistinguishable from zero, labelled descriptive, and said so. The same
model run ungoverned on the same marks would have broken a guard on 34.1% of its drafts. A
fixed-rule crowd fade with no model lost 0.65%; the coin-flip null lost 0.24% at its median, and the
governed replica beat 76.4% of 1,000 seeds.

The product is not the return. It is a record on which every claim about an autonomous agent can
be checked: who decided, what the rules allowed, what they prevented and cost, and whether any of
it was changed after the fact.

---

## 2. What the handbook asks, and where it is answered

| Handbook, Track 2 | Answered by | Where to look |
|---|---|---|
| "The LLM is the primary trading decision-maker … autonomously place orders with risk controls" | Qwen proposes every target; the kernel only reduces; Agent Hub places the order | §4, §5, §6; every decision card |
| "Runnable Demo + a complete event → decision → execution flow" | `t2sa replay` re-runs any recorded cycle without a key and asserts the same ruling; the static page walks each cycle | §9; `/#timeline` |
| Market Sentiment Agent: "FOMO detection → contrarian hedge; sentiment top detection; reduce before overheating" | Crowding features per instrument; the model is asked for crowd-vs-us; the kernel's reductions are visible per guard | §3, §4.3 |
| "Paper trading log (actually run during competition period)" | Two runs on Bitget UTA Demo, 24 Sep → 27 Sep and 28 Sep → 7 Oct, in a hash-chained ledger | §7; `/ledger.jsonl` |
| "Paper trading Sharpe, max drawdown, win rate" | Pre-registered metric definitions, bootstrap bands, recomputed by a standard-library script | §8.1; `scripts/recompute.py` |
| "Decision explainability" | 118 decision cards: trigger, sources, text shown, thesis, every guard's ruling, dry-run payload, venue order id, fills, ledger rows | §9.1 |
| "Agent architecture quality" | Typed contracts, a kernel invariant enforced by constructors, 2,996 tests that never touch the network | §5, §11 |
| "Risk control layer effectiveness" | Governed/ungoverned twin, guard funnel, a venue stop that fired, a forced exit on an 80bps Demo-live gap | §8.2, §8.3 |
| "Incremental value over fixed-rule or Human + AI baselines" | Same-clock baselines: flat, BTC held, fixed-rule crowd fade, 1,000 coin flips, rival sentiment agents | §8.4 |
| Toolkit: `bitget-signal`, `bitget-mcp-server`, Agent Hub, `--paper-trading`, `dryRun` | All wired; Agent Hub is the only order path; the two data services measured and found down | §10 |

---

## 3. The signal: crowds overreach

A crowded position is fragile. The agent measures crowding per instrument from Bitget's own market
data and from the crowd's words:

| Feature | Source | Read |
|---|---|---|
| Funding z-score | Bitget v3 funding history, 90 settlements | how far the current rate sits from the instrument's own norm |
| Funding level | same | a 7.5bp floor, so a z-score on a near-zero rate cannot fire (policy a2) |
| Long/short account and position ratios, taker buy/sell ratio | Bitget v3, read through `bitget-signal` first, then the upstream it wraps | who is on which side, and whether size agrees with count |
| Open-interest change | Bitget v3 | a one-hour jump past the trailing 99th percentile |
| Price stretch | Bitget v3 candles | distance from the instrument's mean |
| Fear & Greed | alternative.me through `bitget-signal`, then direct | crypto-wide mood entering or leaving an extreme (≤25, ≥76) |
| Coordinated stories | RSS, X and Reddit text, clustered by 4-word shingles | a story carried by three or more sources inside two hours |
| Earnings and filings | an equity calendar and insider-filing feed | an event inside 24 hours for a held name |

Every item of text is somebody else's words. It passes a quarantine (replacement on detection,
spotlighting with delimiters, the defences AgentDojo measured) before the model sees it, and withheld
items are marked on the card. One decision card records "one withheld item carried a prompt
injection attempt; noted as market noise, not acted upon."

**What wakes the agent.** Heartbeats at 00:00, 08:00 and 16:00 UTC and at the US cash open; events
when funding passes 2σ, Fear & Greed enters an extreme, open interest jumps, a coordinated cluster
forms, or an earnings report falls inside 24 hours. One event per kind and instrument per 240
minutes; at most 8 event decisions a day. The cadences are pre-registered.

---

## 4. The decision: the model, and only the model

### 4.1 One call for the whole book

Each decision is one `qwen3.8-max` call that sees a snapshot of all 14 instruments, every figure
with its source and health, and the book as the venue reports it. The model answers in a typed
contract (`LlmDecision`): a stance (act, hold, flat with reasons), and for each target a weight, a
thesis, an invalidation, a confidence, the evidence it rests on, what the crowd believes and what
we believe. A target of 1 is 5% of equity.

### 4.2 Grounding

Every number in a thesis, invalidation or view must resolve to a fact in the snapshot within 2%. A
number that does not is deleted, and under guard G9 an ungrounded figure may not add exposure. The
invalidation must cite a fact that was shown (policy d5). Under policy d8, an opening at confidence
0.5 or below is refused by the contract itself: "even odds" is not a reason to pay fees.

### 4.3 Flat is an answer

"Flat, with reasons" is a valid output. It was the model's answer in 15 of its 42 completed
decisions. The first decision of run 2 (card `dec-c43080e5…`) is one:

> "Three names show funding z-scores beyond the extreme reference (CRCLUSDT, SNDKUSDT, METAUSDT)
> … but none has a second independent positioning measure to confirm. Coordinated social clusters
> are all promotional spam. No thesis clears the fee hurdle at even-odds confidence."

Its rejected alternatives (short CRCL, short SNDK) are on the card with the reason each was passed.

### 4.4 An outage is not a decision

Four decisions were lost to model timeouts or transport errors. No deterministic rule trades in the
model's place; under policy a6 the book is flattened on the third failed decision in a row.

---

## 5. The kernel: eleven guards that can only reduce

### 5.1 The invariant, enforced by types

For every instrument the kernel takes the model's proposed weight as the reference and returns an
approved weight that has the reference's sign or is zero, and is no larger in magnitude. A ruling
that adds exposure, turns a side, or changes a weight without naming the guard that bound it cannot
be constructed: `InstrumentRuling._only_reduce` (`types.py`) raises, and `tests/contract/` holds
the cases. The only object the executor accepts is an `ApprovedOrder`, which can be minted by one
function in `kernel/approval.py` and nowhere else (a private token; a source scan test fails if the
token is referenced anywhere else). It is immutable, unpicklable, and bound by hash to the intent and
the ruling that produced it.

Every guard is evaluated on every ruling; none short-circuits; a missing input fails closed for any
increase. Reductions are never blocked.

### 5.2 The guards and what each one answers

Every threshold cites a measurement in `validation/`, and `tests/contract/test_policy_evidence.py`
fails if the policy drifts from it.

| Guard | Rule | The measurement it answers |
|---|---|---|
| G1 venue integrity | refuse or exit when Demo's mark is >3% from its index, Demo's last is beyond the per-instrument p99 gap from live, or the index has stopped moving while the session is open | BTCUSDT Demo mark 65.6% from index on 2026-06-29; BTCPERP 5.4% on 2026-09-23; per-name p99 gaps (`demo_integrity.json`, `fade_demo.json`) |
| G2 weekend freeze | US-equity and index legs flat from Friday 20:00 to Monday 00:00 UTC, pre-flattened at 19:45 | Demo's equity perps barely move at weekends (0.1–1.5bps an hour) while live moves 5.6–29bps; MSTR +5bps on Demo against +204bps live, 2026-09-19 (`weekend_vol.json`, `fade_demo2.txt`) |
| G3 size | ≤5% a name, ≤25% gross, ≤10% net; MSTR+COIN+HOOD+CRCL ≤7.5% together | a no-edge 25%-gross book's worst drawdown was −2.48% (`envelope_clean.json`) |
| G4 stop | a venue stop on every opening: the wider of 4% and twice the p99 Demo-live gap (HOOD 7.66%, SNDK 6.96%, MSTR 6.32%) | the same gaps |
| G5 daily kill | −1.5% from the 00:00 UTC equity flattens the book until the next day | the envelope |
| G6 turnover | ≤2 model orders a name a day; no increase or flip within 24h of the last unless the model declares its invalidation fired | the author's prior study of turnover cost |
| G7 fee budget | no new exposure once fees reach 7bps a day or 20bps a window | taker fee 0.06% on all 14 names (`universe_probe.json`) |
| G8 taker only | market orders only; no opening when Demo's spread is over 20bps | Demo spreads 0.1–6.6bps in the survey; 46.6bps seen once on MSTR, and the guard fired |
| G9 grounding | an ungrounded number may not add exposure | §4.2 |
| G10 breaker | drawdown 2.5% → reduce-only, 4% → halt; four losing trades → reduce-only (lapses after 24h); a stale snapshot or quote cannot open; an unreconciled venue cannot increase | the envelope's worst case; the breaker ported from ARGUS |
| G11 eligibility | universe only, online on Demo, above `minOrderQty` and `minOrderAmount`, split above `maxMarketOrderQty` | `universe_probe.json` |

### 5.3 What the kernel did in run 2

Of 135 legs the model proposed, 50 asked to add exposure: 5 were approved in full, 45 were cut, 0
refused outright. Exposure asked +19.31%, approved +15.46%. The guards that bound: G6 turnover 24,
G11 eligibility 8, G3 size 7, G10 breaker 5, G1 venue integrity 1, G9 grounding 1. Outside
decisions, the protective loop issued 72 rulings (71 venue integrity, 1 weekend freeze).

Two rulings a judge can read in full:

- **A cut** (`dec-e69542…`, 2026-09-29 16:01 UTC). The model asked to keep three shorts and open
  TSLA and META at 2.5% each, "about 17.3% of equity gross". G3 fired at book level: "the 10% net
  cap: side 12.27% over the 10% cap: held 7.27% kept, new exposure 5.00% scaled by 0.5456". Both
  new legs went to 1.36%. The reason is the ruling.
- **A forced exit** (`dec-250d69…`, 2026-09-28 14:29 UTC). The model asked to hold META long; G1
  found Demo's last 80.0bps from live, beyond the measured p99 of 67.2bps, and set the ceiling to
  zero.

A venue stop also fired: the MSTR short opened at 159.78 was closed by Bitget's own
`pos_loss_market` order `1489909291328884736` on 2026-10-02 13:32 UTC at 169.71, the widened G4
stop doing exactly what it was set to do.

---

## 6. Execution: Bitget's Agent Hub, and nothing else

Every order goes through the pinned `@bitget-ai/bitget-agent-cli@3.0.0` (`bgc`), run as a child
process with an environment built from scratch. Every argument list carries `--paper-trading`; the
transport refuses one that does not. Each order is previewed with `--dry-run` (logged as
`ORDER_PREVIEW`, with the exact payload the CLI would send) and then sent; the venue's `orderId` is
logged against the agent's `clientOid`, which is derived from the intent hash. Openings carry a
preset stop (`strategy_order place`); reductions are `reduceOnly`.

**Environment proof before the first order.** The key file must declare Demo; the installed SDK must
send `paptrading: 1`; a Demo account read must succeed (code 00000); and a live read with the same
key must be refused (code 40099, "exchange environment is incorrect"). The proof type cannot report
`passed` unless every check passed. Two passed proofs are in the ledger (seq 2 at 00:07:47 UTC on
28 Sep, and seq 4246 after a restart on 4 Oct).

**Reconciliation.** Fills are polled within 60s of a send, swept every 15 minutes, and the full
history is read daily; fills are de-duplicated by execution id; a discrepancy is logged, never
corrected, and an unreconciled venue stops any increase (G10).

**What the venue did.** 95 orders were sent. 21 were acknowledged and filled, plus one
venue-originated stop fill. 74 were rejected by Bitget, and 72 of those are one protective exit of a
1.7-contract META long, retried every five minutes from 09:16 to 15:11 UTC on 28 Sep, each refused
with `Parameter METAUSDT_UMCBL does not exist`; the position was closed by the model's own decision
at 16:06. The retries are in the record as rejections, and the cause on Bitget's side is an open
question we name rather than hide.

---

## 7. The record: pre-registered, chained, signed, anchored

**Genesis.** Before the first decision, a genesis event (seq 0, 2026-09-28 00:07:44 UTC, hash
`fb66b6bf…`) committed the policy (`eaee6d5a…`, policy-v2), the three prompt files by hash, the code
commit (`5c838fd`), both dependency lock files by hash, the 14-symbol universe, the eight metric
definitions, the scoring window (28 Sep 00:00 to 7 Oct 00:00 UTC), and the expected envelope of a
no-edge book on this venue. The X post carrying the genesis hash was made before the first order.

**Declared changes.** Run 2 follows run 1 (genesis `ebbf607a…`, policy-v1), and declares every
difference: nine code and prompt fixes (d1–d9) and eight policy amendments (a1–a8), each with the
evidence file that motivated it and a test in `tests/contract/test_run2_declaration.py` that the
amendment changes only what it declares. A policy change that is not declared is refused at
start-up.

**Chain.** Every event is `sha256(canonical({seq, ts, kind, mode, payload, blobs, prev_hash}))`,
written by a single process under an exclusive lock, with a head sidecar. 4,540 events at the
export: 1,888 snapshots, 144 hourly marks, 95 order previews, 22 fills, 74 rejections, 72
protective actions, 671 reconciliations.

**Signatures and anchors.** Twelve ledger heads are signed with an Ed25519 key whose public half is
committed (`keys/ledger_signing.pub`); `scripts/verify_heads.py` checks them against the published
chain. Daily heads go to OpenTimestamps; six carry a Bitcoin block attestation (the first at blocks
969,072–969,098), seven are pending, and the genesis's own stamp failed on a client bug and is
recorded as failed.

---

## 8. Results

### 8.1 The scored metrics

Pre-registered definitions, recomputed from `equity_hourly.csv` and `trades.csv` by
`scripts/recompute.py` (standard library only; exit 1 on any mismatch):

| Metric | Run 2 (to 2026-10-05 05:04 UTC) | 90% block-bootstrap band |
|---|---|---|
| Total return | −0.020% | −0.605% to +0.644% |
| Max drawdown | −0.369% | −0.785% to −0.236% |
| Sharpe, annualised | −0.31 (standard error 7.8) | −15.4 to +8.5 |
| Sortino, annualised | −0.55 | −18.5 to +17.8 |
| Win rate | 28.6% on 7 closed trades | — |
| Turnover / fees | 0.31x / 9.22 USDT | — |

Pure noise gives an annualised Sharpe standard error of about 11 over this many hours. The number is
reported because the handbook asks for it, and labelled for what it is. The pre-registered envelope
of a no-edge book put the median return at −3.5bps and the median drawdown at −0.45%; the agent sat
inside it.

### 8.2 Governed against ungoverned: what the rules were worth

The model's draft before the kernel, simulated on the same Demo marks with the same cost model:

| | Governed (replica) | Ungoverned |
|---|---|---|
| Return | −0.001% | +0.046% |
| Max drawdown | −0.383% | −0.406% |
| Drafts that broke a guard | — | **34.1%** |
| Prevented loss / forgone gain | +0.039% / +0.026% of equity | |
| Human takeovers | 0 | |

The kernel cost 2.6bps of gain and saved 3.9bps of loss over nine days; more to the point, a third
of what the model would have sent unaided broke a rule it had been given.

### 8.3 The record is not a sandbox artefact

The same fills marked at Bitget's live prices return +0.131% against −0.020% on Demo marks; the
largest hourly gap between the two is 8.5bps. Our own 21 priced fills landed a median 0.4bps
*better* than the cost model assumed (95th percentile 7.3bps worse). Every baseline is ranked
against the governed replica, which is costed exactly as they are, never against the live book.

### 8.4 Same clock, same snapshots, same simulator: the baselines

| Arm | Return | Sharpe | Max DD | Trades |
|---|---|---|---|---|
| **The agent, governed** | −0.020% | −0.31 | −0.369% | 7 |
| Flat | 0 | — | 0 | 0 |
| BTC held at our gross | +0.047% | 1.95 | −0.113% | 0 |
| Crowd fade, fixed rule, no model | **−0.650%** | −15.67 | −0.650% | 69 |
| Coin flip, median of 1,000 seeds | −0.244% | −4.53 | −0.503% | 19 |
| Lexicon sentiment trader, with the crowd | −0.210% | −3.48 | −0.324% | 56 |
| Lexicon sentiment trader, against the crowd | −0.090% | −1.47 | −0.432% | 58 |
| Season-2 entry: Fear & Greed with long/short confluence | 0 (never traded) | — | 0 | 0 |
| Season-2 entry: sentiment and news signal fusion | −0.026% | −6.50 | −0.036% | 10 |

The governed replica beat 76.4% of the coin-flip seeds on return and 76.5% on Sharpe. The rule it
replaces, a funding and long/short fade with no model, lost 0.65% on 69 trades; the model's
restraint, 15 flats and 10 holds, is where its value showed. The rival agents run on the same
recorded snapshots under the same venue guards, see their own books, and read the raw text our
quarantine withheld; their text scorers were first measured on 2,388 labelled financial tweets
(`rivals/RIVALS.md`). Two further rivals (a finBERT-weighted trader and a TradingAgents-style chain on
Qwen) are built and run only with an approved token budget; they are not in this export.

---

## 9. Explainability and verification

### 9.1 The decision card

One card per decision, abstention and protective ruling (118 at the export). Each shows: what woke
the agent and which sources answered; the text the model was shown, with quarantined items marked;
the thesis, invalidation, crowd-vs-us and rejected alternatives; the grounding report per number;
every guard's ruling with its status, ceiling, reason and the measurement it rests on; the dry-run
payload; the `clientOid`; the venue `orderId`; fills and realised P&L; and the ledger sequence numbers
and blob hashes that prove each line.

Card `dec-c8701963…` (28 Sep 06:08 UTC) is the first trade: trigger `funding_zscore:MSTRUSDT`
(z −4.86); thesis on META, "funding_z_live = −11.49 is extraordinarily far below the extreme
reference of −2 … the most crowded short in the universe"; invalidation "funding_z_live moves back
above zero"; approved as asked; META buy 1.70 → venue `1488348258885812225`, filled 732.53; MSTR buy
7.92 → `1488348265357623297`, filled 157.64; closed at 16:06 for −24.04 and −9.75 USDT. A losing
trade, fully accounted for.

### 9.2 Three ways to check it without trusting us

1. `scripts/recompute.py public/` verifies the chain from `ledger.jsonl` and rebuilds every metric
   from the equity and trade files; exit 1 on any mismatch.
2. `t2sa replay --public public/ --decision <id>` re-runs a recorded cycle without a key: the logged
   snapshot, the recorded completion, the kernel, the planner, and asserts it reproduces the logged
   ruling and intent hashes.
3. `scripts/verify_heads.py <site> validation/run2/signed_heads.json` checks twelve signed heads
   against the committed public key and the published chain; `ots verify` checks the anchors.

The page also replays a recorded venue-integrity refusal (Demo BTCPERP 5.38% from its index on
2026-09-23) labelled as a replay, so a judge can see G1 fire without waiting for the venue to
misbehave.

---

## 10. Bitget's toolkit, used and measured

| Surface | Used for | Measured |
|---|---|---|
| **Agent Hub `bgc`** (pinned 3.0.0) | the only order path: `order place --dry-run` (95 previews), `order place` (21 acknowledged), `order detail` / `fills` / `history`, `position info`, `strategy_order place` / `open` / `cancel` (stops), `account_overview`, `raw getAccountAssets` for the environment proof | 14 verbs used, 13 healthy in the last probe |
| **Bitget public v3 market API** | funding, open interest, long/short, taker ratio, candles, books; Demo reads carry `paptrading: 1` | 7 endpoints used, 5 healthy |
| **`bitget-signal`** (MCP) | the first source for Fear & Greed, long/short, taker ratio, open interest, Reddit trending and news | 7 tools wired; **0 answered** in run 2 (hollow envelopes from the first snapshot); logged as failed, read from the upstreams each tool names (policy d3) |
| **`bitget-mcp-server`** | the first source for crypto futures positioning and the equity calendar and insider filings | 11 entries wired; 1 healthy (`guide`); every `do_query` answered 503 or "Session not found" from 3 Oct |
| **Agentic account (OAuth)** | not used, with the reason on the page: Track 2 runs on a Demo API key, and the Agentic account trades live funds | |
| **GetAgent Playbook** | a signal-only replica of the policy is built (`playbook/`, 18 modules, parity tests); not published, as it needs the `llm_bounded` runtime and a Playbook key | |

The coverage matrix on the page lists all 134 rows: every surface, used or not, with its last health
and where it shows. Feed health lists every failing source since when, with its last error.

---

## 11. Engineering

- 2,996 tests (93 files); none touches the network, a credential, the model or a non-Python child
  process, and `tests/conftest.py` makes each of those a failure rather than a convention.
- `mypy --strict` and `ruff` clean; `scripts/check_all.py` runs all of it.
- 90 contract tests: the only-reduce invariant, the unforgeable `ApprovedOrder`, the environment
  proof that cannot pass unearned, NaN and infinity refused at any depth, `model_construct` banned
  from source, every policy number tied to its measurement, every run-2 amendment changing only what
  it declares, and no file pointing a reader at a private repository.
- 48,389 lines of source, 47,527 of tests.
- Provenance: modules ported from the author's ARGUS project (grounding, quarantine, novelty, the
  breaker, the minimum-of-ceilings kernel, order states, the ledger lock and anchor pattern, the
  Qwen client and budget) are listed in `NOTICE.md` with the source commit; vendored attack corpora
  (HeyArka, AgentDojo) carry their MIT licences.

---

## 12. Limits, stated

- **The return is zero within noise.** −0.02% with a band of ±0.6%; seven trades; a Sharpe band from
  −15 to +8. Nine days is not a track record and the page says so on every metric.
- **Both Bitget data services were down for the whole run.** The agent's view of the crowd came from
  the upstreams they wrap, on a labelled surface. The toolkit matrix reports 0 of 7 and 1 of 11.
- **The red team is built, tested and not run on this export.** It injects the HeyArka and AgentDojo
  attack strings and a coordinated pump into recorded snapshots and grades every arm against its
  clean decision; it calls Qwen, so it runs only with an approved token budget. The page says "not
  computed" rather than showing a stale grade.
- **72 venue rejections of one protective exit.** Bitget Demo refused the same META close every
  five minutes for six hours; the model closed the position itself. The rejections are in the log.
- **The genesis's own OpenTimestamps stamp failed** on a client bug; later heads are anchored and six
  are block-confirmed.
- **The Playbook replica is built from policy-v1,** the run-1 policy, and has not been published.

---

## 13. What comes next

Run the red team under an approved budget and publish its grade; move the same policy to a Bitget
Agentic sub-account with a small live allocation and a 30-day pre-registered window; publish the
Playbook replica once the runtime it needs is enabled; and keep every change declared in the next
genesis.

---

*Reproduce: `pip install -e ".[dev]"`, `python scripts/check_all.py`; `python scripts/recompute.py
public/` against a downloaded copy of the site; `t2sa replay --public public/ --decision
dec-c8701963b5930c3551aec635d15abaa7`.*
