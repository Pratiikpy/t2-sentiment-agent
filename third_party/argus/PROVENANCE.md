# ARGUS provenance

Several modules and test fixtures of this project are ported from ARGUS, an earlier project by the
same author. ARGUS declares the MIT licence (`pyproject.toml` line 11, `license = { text = "MIT" }`)
and carries no separate licence file; `LICENSE` in this directory is that MIT licence with the
copyright line the ported files name. Everything listed here was copied, never imported: this
project runs without the ARGUS tree.

**The ARGUS repository is private at the time of writing**, so a reader cannot open it. This record
is therefore self-contained: every upstream file is pinned by the SHA-256 of its exact bytes and by
its git blob id, and all of them by one commit, `3dec6baf9dfa7be37c7b452e26a9b139df252f75`. Each
blob id below was read from that commit's tree (`gh api repos/<owner>/<repo>/git/trees/<commit>`)
and equals `git hash-object` of the local copy that was ported, and the SHA-256 is of that same
local copy, on 2026-09-24. Anyone later given the ARGUS source can check a file with
`sha256sum <path>` or `git hash-object <path>`.

Every receiving file's header names the upstream path and cites this record; the pin is the commit
above for all of them.

## Code

| Receiving file | Upstream file (ARGUS, path in its repository) | SHA-256 | Git blob |
|---|---|---|---|
| `src/sentiment_agent/crowd/novelty.py`, `playbook/src/crowd.py` (via the primary) | `argus/src/argus/agents/novelty.py` | `d9302be5b12e04b01d5f4a8dca11c654d510905a6a843f8a1b1dbfd8a44d0ce3` | `201729ce7042388782cc87c3393291b7002b7f65` |
| `src/sentiment_agent/crowd/quarantine.py` | `argus/src/argus/agents/quarantine.py` | `7a95194aa7f1f968b1fb42daa3eeb743e4a713187e684c1d22092ce6442a05d6` | `a33351548489b0035cd1fa8a895edd6a5a1abdfd` |
| `src/sentiment_agent/decision/grounding.py`, `playbook/src/grounding.py` (via the primary) | `argus/src/argus/agents/grounding.py` | `533831eed522539bb7d590f874ffec1445020bb16b7be3bf8626db38abcf6b17` | `e5f6634d73087e0b1a3707a9ae4876dc300dd298` |
| `src/sentiment_agent/kernel/kernel.py` (minimum-of-ceilings rule) | `argus/src/argus/agents/desk.py` (`ConstitutionPolicy.rule`) | `86fc3430b06e3fb232eb32987fa195318a739f3b297127b4142cef6dd57b7ca8` | `76e887366db1cf279ba464add00fa2928f7a43dd` |
| `src/sentiment_agent/llm/client.py`, `src/sentiment_agent/llm/budget.py`, `src/sentiment_agent/decision/contract.py` | `argus/src/argus/llm/qwen.py` | `dcc4ba1077a63ca2dca4ee7ece18e566b90b29be2a3a5bfd1c6a987c2678daa9` | `5be5d66d370a9ee2018232439fe55409d7bc7be9` |
| `src/sentiment_agent/execution/orders.py` | `argus/src/argus/execution/orders.py` | `bf2e41ee9d6ffbf87ae485ad2766afb43a82ac433de63f62cfca6d8aba7cc160` | `a5b11b9bce7bc3b9b38632415c6cce3e8ef75628` |
| `src/sentiment_agent/kernel/breaker.py`, `playbook/src/breaker.py` (via the primary) | `argus/src/argus/risk/circuit.py` | `365b7ad90197a313227761d462e7a854170e7350d72957a806ee62cc8fac4d85` | `81056e41a4466c0a50066b89112b3c28e6a73f79` |
| `src/sentiment_agent/ledger/chain.py` (lock and anchor patterns) | `argus/src/argus/paper/ledger.py` | `6ec8723130b6fd5a54aab1c669985135c168a3f24f217c7bf9bfc387998284be` | `600b00f321f446eddbd2b8a42dd913198a506385` |
| `src/sentiment_agent/redteam/attacks.py` (`coordinated_pump` rephrasings) | `argus/src/argus/eval/sentiment_comparison.py` | `7d04819db39e901eb472c2035555c174ceedddc595f6f7d7445586fb6019027b` | `a3955012750e85bcb8229bebc3b9305e06342d62` |
| `src/sentiment_agent/sources/bitget_data.py`, `src/sentiment_agent/sources/mcp_http.py` (session handshake) | `argus/src/argus/market/bitget_mcp.py` | `bc8f550f718cb6a40339439ed2233505d9845317f1a94d13a702ba5ae8905721` | `984cd5f224d70d14920178a35fca4413b412c560` |
| `src/sentiment_agent/sources/mcp_http.py` (`hollow`) | `argus/src/argus/market/skills.py` | `297bde83bb872a054af392f37e8693d074f1a3818b6d385345b3c8d246e8f654` | `27ec1b4499b5613f70cce572bd859a00c49e96df` |

## Tests and fixtures

| Receiving file | Upstream file | SHA-256 | Git blob |
|---|---|---|---|
| `tests/crowd/test_novelty.py` | `argus/tests/test_novelty.py` | `7b3ca6eff48748c81572c91984e3fa47404d7506fd1b8e361cfb07d3c8c732ce` | `10dfda77469d3402577e62f76d1f642cbd1b30d4` |
| `tests/crowd/test_quarantine.py` | `argus/tests/test_quarantine.py` | `422b1ff9a2159c39e4d660dd145ca23172f25cd1d312ed1c67fd74c342f479e9` | `f009c236ec5c48da3ef649086438f32b41f26492` |
| `tests/fixtures/crowd/agentdojo_v1_corpus.json` | `argus/src/argus/eval/baselines/agentdojo_attack_corpus.json` | `083d0c93b40d1c3b750b8836033efe62aed63497891bed2a09962f0da20af1d1` | `ca1357932a71a7f55ff2711c9b5d37a8fd104b6c` |
| `tests/fixtures/crowd/agentdojo_v122_heldout_corpus.json` | `argus/src/argus/eval/baselines/agentdojo_heldout_v122_corpus.json` | `0260b0884cd061b545c399012f35b3cd5ce3f29ec891b9ec876e33bb606d70ad` | `7c123226851afc0a0fc0dfc1d756580eb6aecfdd` |
| `tests/fixtures/crowd/live_headlines.json` | `argus/tests/data/live_headlines.json` | `fd10839e36fe52c5df2bf92b97ad43ef3df6d8a7a65d4d63feae45383b1e7aff` | `08b3d5135527347285b481c68ccdc51847680e58` |
| `tests/fixtures/crowd/production_false_positives.json` | `argus/data/quarantine_production_withholdings.json` | `ec370a7c86afbff5f26e55a34f48c87276901ecd5fb35071dc46e658dca9e0f3` | `f9d28158ee53701ea291a6a48dc94468c32b074b` |
| `tests/fixtures/crowd/handwritten_sets.json` (`near_miss_prose`, `paraphrases`) | `argus/src/argus/eval/quarantine_comparison.py` | `3bc16fee564c63308473225074f91dfbcb26b698134bc544cbeec816b92e5aeb` | `22e6b33f1bb6d706da57132377fd676857bc07b2` |
| `tests/fixtures/crowd/handwritten_sets.json` (`semantic_injections`) | `argus/src/argus/eval/quarantine_generalisation.py` | `3e59a9a6de755e7c92db0de5c843e6f544dbe42d05a8e7e001fe30a7b686016c` | `a860796e2d84d904c3b3dc56a50686f9c779faf0` |
| `tests/fixtures/decision/fabricated_figures.json` | `argus/data/explainability_comparison.json` | `8233378e9ec9317522f8ee0a4ab44d8416b7f6b64e775032a1625a1054f292a8` | `44475ce688f17c28de5d689ca3569dae0baaa542` |
| (the licence declaration) | `argus/pyproject.toml` | `13977f6e85e3d92a24d717f2bc785c7f0eaa9d2b3a52376738ed384834d82697` | `9b033a693de2c268684bc44b390e9dfede5bce46` |

The JSON fixtures were re-serialised with LF line endings when they were vendored; their data is
unchanged, and each names its origin in `_vendored_from` as `ARGUS@<commit>:<path>`.

Files ARGUS only informed, with nothing copied (a measured fact, a design lesson), are named where
they are used and are not listed here.
