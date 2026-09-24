# Notices and attributions

This project is MIT-licensed (see `LICENSE`). It builds on the work below. Anything copied carries
the upstream licence text and a header in the file that received it; anything only studied is
named here with what was learned from it.

## Code ported, same author, MIT

From ARGUS, an earlier project by the same author (MIT, declared in its `pyproject.toml` line 11).
Copied, not imported: this project never imports from the ARGUS tree. The ARGUS repository is
private at the time of writing, so the record is kept here: `third_party/argus/LICENSE` is its
licence, and `third_party/argus/PROVENANCE.md` pins every upstream file by commit
(`3dec6baf9dfa7be37c7b452e26a9b139df252f75`, the same one for all), SHA-256 and git blob id. Each
receiving file starts with a header naming the upstream file, that commit and its SHA-256, and
citing the record.

| Receiving file | Source in ARGUS |
|---|---|
| `src/sentiment_agent/decision/grounding.py` | `src/argus/agents/grounding.py` |
| `src/sentiment_agent/crowd/quarantine.py` | `src/argus/agents/quarantine.py` |
| `src/sentiment_agent/crowd/novelty.py` | `src/argus/agents/novelty.py` |
| `src/sentiment_agent/kernel/breaker.py` | `src/argus/risk/circuit.py` |
| `src/sentiment_agent/kernel/kernel.py` (minimum-of-ceilings rule) | `src/argus/agents/desk.py` `ConstitutionPolicy.rule` |
| `src/sentiment_agent/execution/orders.py` | `src/argus/execution/orders.py` |
| `src/sentiment_agent/ledger/chain.py` (lock and anchor patterns) | `src/argus/paper/ledger.py` |
| `src/sentiment_agent/llm/client.py` (wire format, streaming), `llm/budget.py`, `decision/contract.py` | `src/argus/llm/qwen.py` |
| `src/sentiment_agent/redteam/attacks.py` (`coordinated_pump` rephrasings) | `src/argus/eval/sentiment_comparison.py` |
| `src/sentiment_agent/sources/bitget_data.py`, `sources/mcp_http.py` | `src/argus/market/bitget_mcp.py`, `src/argus/market/skills.py` |
| `tests/crowd/test_novelty.py`, `tests/crowd/test_quarantine.py` and the crowd and decision fixtures | ARGUS tests and evaluation data (full list in `third_party/argus/PROVENANCE.md`) |

## Third-party material vendored

| Material | Licence | Where | Use |
|---|---|---|---|
| HeyArka red-team vectors (`Jhaycrypt001/HeyArka`) | MIT | `src/sentiment_agent/redteam/corpus/` with its licence text | Attack corpus for the sentiment-input red team. No public head-to-head against its authors. |
| AgentDojo injection strings (`ethz-spylab/agentdojo`) | MIT | `src/sentiment_agent/redteam/corpus/` with its licence text | Attack corpus; also the quarantine's regression set |

## Third-party software used, not vendored

| Software | Licence | Use |
|---|---|---|
| Bitget Agent Hub CLI `@bitget-ai/bitget-agent-cli` 3.0.0 and SDK `@bitget-ai/bitget-agent-sdk` | MIT | Every paper order is sent by Bitget's own `bgc --paper-trading`, pinned in `tools/agent-hub/package-lock.json` |
| `bitget-signal` MCP service and Skills | MIT | Perception (sentiment, positioning, news); Fear & Greed bands cited in `policy.py` |
| `bitget-mcp-server` | Bitget service | Perception (Fear & Greed, derivatives ratios, earnings calendar, insider filings) |
| `ProsusAI/finBERT` | Apache-2.0 | Rival arm, loaded through `transformers` at evaluation time only |
| `TauricResearch/TradingAgents` | Apache-2.0 | Rival arm: its social-media-analyst and trader prompts, adapted and attributed in `rivals/tradingagents_arm.py` |
| OpenTimestamps client | LGPL-3.0 | Invoked as the `ots` command to timestamp the genesis and daily head hashes |

## Studied, not copied

| Source | Licence | What was taken |
|---|---|---|
| NautilusTrader `crates/model/src/enums.rs:1304-1388` | LGPL-3.0 | The order state model, reimplemented |
| CryptoTrade (EMNLP 2024) | CC BY-NC-SA 4.0 | Nothing vendored. If used as a rival arm, it is rebuilt from the paper's description |
