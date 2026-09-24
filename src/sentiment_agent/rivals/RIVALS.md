# Rivals for the Market Sentiment sub-theme

The handbook's Market Sentiment Agent turns what the crowd says and does into positions: "FOMO
detection → contrarian hedge; sentiment top detection; reduce before overheating" (handbook:233).
The rivals here are agents that do that, social, forum and news sentiment into positions, not tools
that only score text. Several are run, because each is strong at something different: a lexicon is
fast and transparent, finBERT reads financial prose, an LLM chain reasons over mixed sources, and
the Season-2 entries are what this agent is actually judged against.

Code: `rivals/registry.py` (the roster, the shared rules, the three Season-2 entries),
`keyword_arm.py`, `finbert_arm.py`, `tradingagents_arm.py`, `harness.py`.

## How the comparison is run

* **Same inputs.** Every arm decides on the same recorded `PerceptionSnapshot` sequence as our
  agent, at the same instants (`harness.run_rivals`).
* **Same marking.** Every schedule goes through the one `analysis/armsim.ArmSimulator`: the
  production kernel, planner and book, the same Demo mark candles, Demo spreads, taker fee and 4%
  venue stop, marked on the same UTC hours from the first snapshot to the last mark.
* **Same rules.** Every rival runs under `VENUE_GUARDS` only (venue integrity, weekend freeze, size,
  stop, daily kill, taker-only, eligibility): the guards that describe the venue and the mandate,
  not the decision-maker. Weights are capped at 5% per name and 25% gross before the kernel caps
  them again.
* **Its own book.** Each rival is shown its own positions, rebuilt from its own previous targets,
  never ours (`harness.ShadowBook`).
* **What it reads.** The raw text of every item in the snapshot, including items our quarantine
  withheld: no rival has a quarantine upstream, and the comparison is of the rivals as they are. The
  red team (M14) measures what that costs them.
* **Metrics.** Those pre-registered in `policy.METRICS` (DESIGN.md §15), labelled descriptive: a
  three-day window cannot separate skill from luck, and the table says so.
* **LLM spend.** The two LLM arms run offline on the recorded snapshots only after the owner
  approves the bound printed by `harness.estimate_qwen_tokens` (see "Spend").

## Results

### Head to head on the paper window: PENDING

No arm has been run on the real record, because the record does not exist yet: the paper window
starts when the owner creates the Demo key (DESIGN.md §21), and until then no `PAPER` snapshot has
been logged. Nothing below is a trading result.

| Arm | Return | Sharpe (SE) | Max DD | Win rate | Trades | vs our agent |
|---|---|---|---|---|---|---|
| `rival_lexicon_follow` | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| `rival_lexicon_fade` | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| `rival_finbert_follow` | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| `rival_finbert_fade` | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| `rival_tradingagents_sentiment_trader` | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| `rival_s2_fear_greed_confluence` | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| `rival_s2_sentiment_fusion` | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| `rival_s2_qwen_headline_trader` | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

What it takes: the logged snapshots and the Demo mark candles of the window (both in the ledger and
blob store), and, for the two LLM arms only, the owner's approval of their token bound. The six
arms without an LLM need no approval and run the moment the window has snapshots. Every row is
published whatever it shows; a rival that beats our agent is reported as a loss.

### Measured now: how well each rival reads market text

The text scorer is each text rival's input, so it was measured on a public labelled set before any
trading: `zeroshot/twitter-financial-news-sentiment` (MIT), validation split at revision
`ccbe24de388e287beb92dd393a335c376b350ac3` (sha256 of `sent_valid.csv`
`431ea8274c98b52212daa68f2fb962811bf1913c1efe51b99765c8e2311806de6`), 2,388 finance tweets
labelled bearish (347), bullish (475) and neutral (1,566). Every scorer is read exactly as the arms
read it: a score of +0.05 or more is bullish, -0.05 or less bearish, anything between neutral.
Measured 2026-09-24; the market vocabulary of the lexicon arm was frozen before this run and has
not changed since.

| Scorer | Accuracy | Macro-F1 | Polar tweets called | Right way when called | Wrong way |
|---|---|---|---|---|---|
| VADER, upstream, full lexicon | 0.494 | 0.447 | 62.5% | 75.7% | 15.2% |
| Lexicon arm (VADER rules + market vocabulary) | 0.656 | **0.558** | 52.7% | 86.6% | 7.1% |
| finBERT arm (ProsusAI/finbert) | 0.523 | 0.521 | **91.1%** | **87.7%** | 11.2% |
| Entry C headline polarity (its own scorer) | **0.701** | 0.521 | 32.1% | 86.0% | **4.5%** |
| Entry B news keywords (its own scorer) | 0.655 | 0.403 | 19.3% | 74.8% | 4.9% |

"Polar tweets called" is the share of bullish or bearish tweets the scorer took a side on; "right
way when called" is how often that side was the labelled one; "wrong way" is the share of polar
tweets scored with the opposite sign, the error that costs a trader money. Read plainly: the market
vocabulary lifts VADER on this set (macro-F1 0.447 to 0.558, wrong-way calls halved); finBERT takes
a side on far more tweets and is right slightly more often when it does, at a higher wrong-way rate;
the two Season-2 keyword scorers rarely take a wrong side because they rarely take a side at all.
No scorer is best on every column, which is why several rivals are run.

Reproduce (needs the `[rivals]` extra and the model weights):

```python
import csv
from sentiment_agent.policy import POLICY_V1
from sentiment_agent.rivals.keyword_arm import compound
from sentiment_agent.rivals.finbert_arm import FinbertArm

rows = list(csv.DictReader(open("sent_valid.csv", encoding="utf-8")))  # 0 bear, 1 bull, 2 neutral
lexicon = [compound(r["text"]) for r in rows]
finbert = FinbertArm(policy=POLICY_V1).score_texts([r["text"] for r in rows])
# band each score at +-0.05 and compare with the labels
```

### Measured now: the ports are faithful

* **VADER.** The lexicon arm reproduces upstream VADER's compound score exactly on ten sentences
  that exercise negation, "never so", "without doubt", "no", "nor", "least", "but", capitals, a
  booster and the multi-word dampeners (`tests/rivals/test_keyword_arm.py`, values produced by the
  upstream package at the commit below).
* **finBERT.** On a six-sentence labelled set in the Financial PhraseBank style, the loaded model
  scores both positive sentences above +0.5, both negative below -0.5 and both neutral inside
  +-0.5 (`tests/rivals/test_finbert_arm.py`; skipped only when the extra is not installed).
* **TradingAgents.** The prompts sent are the upstream text with the adaptations listed below; the
  tests pin them sentence by sentence (`tests/rivals/test_tradingagents_arm.py`).

## Spend

`harness.estimate_qwen_tokens(arms, n)` is the most the LLM arms can spend over `n` snapshots,
bounded the way the daily budget bounds a call (UTF-8 bytes of the prompt plus the completion cap;
`llm/budget.py`). Every text in a prompt is cut to a fixed number of characters and bytes so the
bound exists before any call is made.

| Arm | Bound per snapshot | Calls per snapshot |
|---|---|---|
| TradingAgents chain, 3 symbols | 556,230 tokens | up to 12 (analyst and trader, each with its free-text retry, per symbol) |
| Entry C, Qwen headline trader | 6,248 tokens | 1 |

The TradingAgents bound is loose by construction: it assumes every one of 65 texts per symbol is
at its byte cap in four-byte characters and that every call needs its retry. Real prompts of
ordinary English posts are a fraction of it; the ledger records actual usage per call, and a run
that reaches its approved budget stops rather than continuing without a rival. `max_symbols`
lowers the bound linearly.

## The rivals, one by one

### Lexicon sentiment trader (`rival_lexicon_follow`, `rival_lexicon_fade`)

* **Read:** `cjhutto/vaderSentiment` at commit `44fc044cd877310ee8278a0eadf34bcd50d41d06`,
  `vaderSentiment/vaderSentiment.py:26-509` (constants, negations, boosters, `negated`,
  `normalize`, `allcap_differential`, `scalar_inc_dec`, token handling, `polarity_scores`,
  `sentiment_valence`, `_least_check`, `_but_check`, `_special_idioms_check`, `_negation_check`,
  punctuation emphasis, `score_valence`), `vader_lexicon.txt`, README "About the Scoring".
* **Licence:** MIT. Licence text carried in `keyword_arm.py`.
* **Taken:** every scoring rule and constant; the valences of 162 general words, copied exactly.
* **Changed:** the "but" rule by position (upstream re-finds values with `list.index`, so equal
  valences on both sides of "but" are mishandled); market emoji scored directly; the general-English
  special idioms dropped (the multi-word dampeners kept). Added: a market vocabulary of words VADER
  lacks (*bullish, bearish, rally, plunge, upgrade, downgrade, beats, rekt, selloff, ...*), written
  before any measurement and never tuned; a test keeps it disjoint from VADER's words.
* **Trading rule:** texts naming a symbol in the last 24 hours; at least 3, flat inside +-0.05,
  full 5% at +-0.5; with the crowd (follow) or against it (fade, the handbook's contrarian hedge).
* **Result:** text-reading benchmark above. Trading: PENDING.

### finBERT-weighted trader (`rival_finbert_follow`, `rival_finbert_fade`)

* **Read:** `ProsusAI/finBERT` at commit `44995e0c5870c4ab37a189d756550654ae87cdf0`:
  `finbert/finbert.py:30` (64-token sequences), `:581-640` (`predict`: sentence split, softmax,
  `sentiment_score = P(positive) - P(negative)` at `:625`, labels at `:607-608`),
  `finbert/utils.py:212-215` (softmax); the model card of `ProsusAI/finbert`.
* **Licence:** Apache-2.0. No code or weights are vendored; the weights (revision
  `4556d13015211d73dccd3fdd39d39232506f3e43`) are loaded through `transformers` at evaluation time
  from the optional `[rivals]` extra.
* **Taken:** the sentence score, the 64-token truncation, the class order read from the model's
  `id2label`.
* **Changed:** sentences split on punctuation and line breaks instead of NLTK's `sent_tokenize`,
  to keep NLTK out of the dependencies.
* **Trading rule:** the lexicon trader's, on the mean sentence score per text.
* **Result:** text-reading benchmark above. Trading: PENDING.

### TradingAgents sentiment analyst and trader (`rival_tradingagents_sentiment_trader`)

* **Read:** `TauricResearch/TradingAgents` at commit `be952b8eccb49720509af544c6675233bc1f10d0`:
  `tradingagents/agents/analysts/sentiment_analyst.py` (whole file; the renamed social-media
  analyst, whose shim is `social_media_analyst.py:1-23`), `agents/trader/trader.py:21-90`,
  `agents/schemas.py:58-204` and `:285-368`, `agents/utils/structured.py:36-89`,
  `agents/utils/agent_utils.py:52-66` and `:136-201`, `agents/utils/rating.py:1-83`,
  `dataflows/stocktwits.py:95-138`, `dataflows/reddit.py:74` and `:264-350`,
  `dataflows/yfinance_news.py:95-117`, `graph/setup.py:50-92`, `graph/propagation.py:18-48`,
  `graph/signal_processing.py:1-40`, `default_config.py:1-125`.
* **Licence:** Apache-2.0, text reproduced below as section 4(a) requires; the changes are stated
  at the top of `tradingagents_arm.py` as section 4(b) requires. The repository has no NOTICE file.
* **Taken:** the sentiment analyst's system framing and instructions, the trader's prompt, the
  `SentimentReport` and `TraderProposal` field descriptions, the structured-then-free-text
  fallback, the REVIEW rule for an answer with no rating, the rating heuristic, the data-block
  formats, the quick-thinking tier for both roles.
* **Changed:** data from the recorded snapshot instead of live Yahoo Finance, StockTwits and Reddit;
  X posts in place of StockTwits (no user-labelled Bullish/Bearish tags, so every post is
  `no-label`); schemas stated in the prompt under JSON mode; the analyst's report is the trader's
  plan (the full graph puts a bull/bear debate and a Research Manager between them); texts cut to
  byte bounds; Sell opens a short on a perpetual; at most 3 symbols a snapshot, held first;
  temperature 0 and the LOW reasoning tier, where upstream leaves both to the provider's defaults
  (`default_config.py:93`, `:99`), whose values on this gateway are NOT VERIFIED.
* **Result:** PENDING, and needs the owner's approval of its spend.

### Season-2 entries

The sweep, 2026-09-24: GitHub repository search (queries `bitget sentiment`, `bitget hackathon
sentiment`, `bitget fomo`, `"Market Sentiment Agent"`, `"Agentic Trading" bitget`, `bitget
"fear and greed"`, pushed after 2026-08-20; READMEs of 24 candidate repositories read) and X search
(eight queries through twitter-cli, 127 distinct posts). **No Season-2 entry found declares the
Market Sentiment Agent sub-theme**; the earlier scan of 34 Track 2 repositories found none either.
The entries below turn crowd sentiment into positions and ship runnable code, so they are run as
arms. Per the plan's default (win plan Move 24), entries are described by what they do, not by who
built them; each is identified by its public commit so the record can be checked.

**Entry A, Fear & Greed with long/short confluence (`rival_s2_fear_greed_confluence`).**
A Track 2 agent on Bitget UTA: buys when the crypto Fear & Greed index is below 25 and fewer than
45% of accounts are long, sells when it is above 75 and more than 65% are long, holds otherwise;
BTC spot and the TSLA perpetual, 5% of the balance a trade capped at $50, 3 trades a day per asset.
The entry itself only previews its orders (a dry run); the arm executes the decisions it would have
placed.

* **Read:** `strategy.js` at commit `bb4804bff99e01e545a61e0f0ffe63b42117cc44`, lines 1-838: the
  decision at `:410-470`, the quota at `:667-671`, the thresholds at `:22-37`, sizing at `:77-80`
  and `:551-565`, data fallbacks at `:250-408`, the dry-run mode at `:787`; `README.md`.
* **Licence:** none (no licence file; the README says MIT without the licence text). Treated as
  unlicensed: rebuilt from the described behaviour, no code copied.
* **Differences:** reads Fear & Greed from Bitget's services instead of alternative.me; the equity
  perps have no long/short series in the snapshot, which the entry's own fallback also reads as
  neutral. The $50 cap is kept: on an account above $1,000 it makes every trade `50 / equity` of
  equity, which is how the entry trades.
* **Result:** PENDING.

**Entry B, sentiment and news signal fusion (`rival_s2_sentiment_fusion`).**
A Track 2 agent that fuses five confidence-weighted signals into a regime and routes it to a
strategy; despite its description, its trading decision is deterministic (Qwen writes stress-test
narratives). Run with the two signals a snapshot can feed it, sentiment (Fear & Greed, long/short,
taker ratio) and keyword news, which its fusion allows for by redistributing weight.

* **Read:** at commit `ee3b4d65dc116f5c86c8fe972549056d20e16e64`:
  `backend/src/signals/mcpSignals.ts:1-211`, `signalFusion.ts:1-114`,
  `backend/src/agents/strategyRouter.ts:1-151`, `agentCycle.ts:1-242`, `riskManager.ts:1-140`,
  `executionEngine.ts:73-140`, `backend/src/services/aiReasoning.ts:1-169`, `LICENSE`.
* **Licence:** MIT, notice reproduced below. Ported: word lists, weights, thresholds, formulas.
* **Differences:** its technical, macro and on-chain signals are absent; its trailing stop is not
  modelled (the snapshots carry no path between decisions). Its risk manager is ported where a
  snapshot can drive it (a new entry halved when the news score jumps by more than 0.4; no new
  trade after 10 in a day); its 10% drawdown and 5% daily-loss halts need the arm's own equity,
  which the harness does not show an arm, and the venue guards' 1.5% daily kill is stricter.
* **Result:** PENDING.

**Entry C, Qwen headline-sentiment trader (`rival_s2_qwen_headline_trader`).**
Qwen reads per-ticker headline sentiment, 24h momentum and the latest headlines and returns Buy,
Sell or Hold with a size, applied to a long-only paper book of tokenized US equities.

* **Read:** at commit `b1fa7f7b6ac8f1ef2c68ef7592a073abbaaa5d33`: `lib/agent/prompt.ts:1-87`,
  `lib/data/sentiment.ts:1-65`, `lib/llm/qwen.ts:1-195`, `lib/agent/engine.ts:1-111`,
  `lib/portfolio/ledger.ts:40-110`, `lib/config.ts:1-102`, `LICENSE`.
* **Licence:** MIT, notice reproduced below. Ported: the prompt, the headline scorer, the parser,
  the ledger rules, the balanced strategy.
* **Differences:** the policy's perps in place of its xStocks (MSFT has no Demo perp); its 35% cap
  rescaled to the shared 5% per-name budget, relative sizing kept; an unusable answer holds instead
  of falling back to its rule-based mock; `Thinking.LOW`, because the gateway's default when no tier
  is sent is NOT VERIFIED; the realized P&L line of its prompt reads `$0.00` (the arm's own book
  carries no realized history).
* **Result:** PENDING, and needs the owner's approval of its spend.

**Examined and not run, with the reason.**

| What the entry is | Why it is not an arm here |
|---|---|
| Track 2 event-driven agent on stock perps that reads funding and long/short contrarian plus a Qwen news bias | Declares the Event-Driven sub-theme; sentiment is one input of a multi-indicator diagnostic; no licence file |
| Track 2 LLM trader with confidence brakes, sentiment as one input it deliberately strips ("text can talk it out of a trade, never into one") | A governance design rather than a sentiment trader; it belongs in the red-team comparison (M14); no licence file |
| Track 3 research workbenches that read Fear & Greed or run the five bitget-signal skills | Produce research, never positions |
| A crypto cycle-top band panel with a six-part sentiment view | Produces no positions; CC BY-NC-ND |
| A tactical signal radar using paid social-data APIs | Its sentiment inputs need paid keys and are not in any snapshot |

**CryptoTrade (EMNLP 2024), considered.** A reflective LLM trader over on-chain statistics and news
for ETH, BTC and SOL (CC BY-NC-SA: it could only be rebuilt from the paper). Not run: its
distinguishing input, daily on-chain transaction statistics, is not in the snapshots, so a rebuild
would be a CryptoTrade without its on-chain analyst; its news-reading LLM path is what the
TradingAgents chain and Entry C already exercise. Adding it needs an on-chain perception source
first (bitget-signal `network_status` is the candidate) and a rebuild from the paper's description.

## Licence texts

### TradingAgents: Apache License, Version 2.0

Applies to the prompt text and schemas adapted in `src/sentiment_agent/rivals/tradingagents_arm.py`
from TauricResearch/TradingAgents (Yijia Xiao, Edward Sun, Di Luo, Wei Wang). Reproduced verbatim
from the repository's `LICENSE` at the commit above (sha256
`1eb85fc97224598dad1852b5d6483bbcf0aa8608790dcc657a5a2a761ae9c8c6`).

```text
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

### Season-2 entry B: MIT License

Applies to the logic ported into `SentimentFusionArm` and its helpers in
`src/sentiment_agent/rivals/registry.py`.

```text
MIT License

Copyright (c) 2026 adeyemib05

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Season-2 entry C: MIT License

Applies to the prompt and logic ported into `QwenHeadlineTraderArm` and its helpers in
`src/sentiment_agent/rivals/registry.py`.

```text
MIT License

Copyright (c) 2026 Sketchify

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### VADER: MIT License

Carried in full at the top of `src/sentiment_agent/rivals/keyword_arm.py`.
