# Scripted decisions

Stand-in model answers for keyless rehearsals (`--llm scripted --script <file>`). Each file is a
JSON array of decision objects in the contract the system prompt states
(`src/sentiment_agent/decision/prompts/output_schema_v1.md`); the scripted model returns them in
order, one per decision call. They never reach the paper run, which decides with live Qwen only.

| File | What it rehearses |
|---|---|
| `decision.json` | One `act` decision: a small BTCUSDT short with its thesis, invalidation and evidence. Through the kernel's eleven guards, the planner and a previewed (dry-run) or simulated order. |
| `flat.json` | One `flat_with_reasons` answer, for a tick that should not trade. |

The thesis text states no figures on purpose: a number the model was not shown fails the grounding
guard (G9), so a rehearsal on any day's live data rules the same way.

```powershell
t2sa decide --mode dryrun --llm scripted --script examples/scripted/decision.json --reason "rehearsal" --symbols BTCUSDT
```
