"""Entry point: the runner executes ``python -m src.main``.

This Playbook is live-only (``backtest_support: none``, ``runtime_profile: llm_bounded``): its
decisions come from a model reading live context, which cannot be fairly replayed on history. A
historical evaluation run therefore emits one ``watch`` signal saying so and trades nothing. A live
run is one cycle of :mod:`.cycle`.

No trade mutation happens in this file: orders are sent only inside the ``execute_trade`` callback
that :func:`src.cycle.run_live` hands to ``runtime.emit_signal_or_follow``, which the runtime calls
only for a follow-trade subscription (``references/sdk/runtime/catalog.md``).
"""

from getagent import llm, runtime, trade

from . import policy_v1 as policy
from .cycle import Env, run_live


def run() -> None:
    if runtime.is_historical():
        symbols = runtime.manifest.get("trading_symbols", []) or [policy.UNIVERSE[0][0]]
        runtime.emit_signal(
            action="watch",
            symbol=str(symbols[0]),
            confidence=0.0,
            metrics={"historical_runs": 0},
            meta={
                "reason": "live-only Playbook: the model's decisions cannot be fairly replayed on "
                "history (backtest_support: none); no historical evaluation is claimed",
                "policy_hash": policy.POLICY_HASH,
                "evidence_role": policy.EVIDENCE_ROLE,
                "official_evidence_kind": "paper",
            },
        )
        return
    if not runtime.is_live():
        raise ValueError(f"unsupported evaluation_mode={runtime.evaluation_mode!r}")
    run_live(Env(runtime=runtime, llm=llm, trade=trade))


if __name__ == "__main__":
    run()
