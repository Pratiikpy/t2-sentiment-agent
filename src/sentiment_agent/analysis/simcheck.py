"""The simulator checked against the venue: how far real Demo fills sat from the cost model.

Every baseline, coin flip and twin arm is filled by :class:`~sentiment_agent.analysis.armsim.
ArmSimulator` at ``reference × (1 ± half the median Demo spread)`` plus the taker fee. The live
book is filled by the venue. So an arm is ranked against the **governed replica**, which is
simulated exactly like them (``analysis/twin.py``), never against the live book: a systematic gap
between the venue's prices and the model's would otherwise be read as skill (review finding on
``coin_flip_summary``).

The live book is still published beside the replica, and this module states the gap between the
two so the cost model is itself checked:

* **Per-fill slippage.** For each fill of one of the agent's own orders (a ``clientOid`` whose
  intent the ledger holds), the price the simulator would have used for that intent is
  ``reference_price × (1 + h)`` for a buy and ``× (1 - h)`` for a sell, ``h`` being the simulator's
  half spread for the symbol. Slippage in basis points is ``side × (exec_price / model - 1) ×
  10,000`` with ``side`` = +1 for a buy and -1 for a sell, so a positive number is a fill *worse*
  than the model assumed. Venue-originated fills (a stop firing, a liquidation) have no intent and
  are counted apart, not priced.
* **Return gap.** ``live total return - replica total return`` over their own marks.

Both are descriptive: a handful of fills cannot estimate a cost bias precisely, and the record says
how many fills the figures rest on.
"""

import math
import statistics
from collections.abc import Callable, Sequence
from decimal import Decimal
from typing import Final, Literal

from sentiment_agent.analysis.bootstrap import percentile
from sentiment_agent.types import ArmResult, Fill, Model, OrderIntent, Side

BPS: Final = Decimal(10_000)


class SlippageBand(Model):
    n: int
    mean_bps: float | None
    median_bps: float | None
    p95_bps: float | None
    worst_bps: float | None


class SimulatorCheck(Model):
    """The live book against its simulated replica (module docstring)."""

    ranked_arm_id: str
    """The arm the coin flips and baselines are ranked against: the replica when it exists."""
    fills_priced: int
    fills_unpriced: int
    """Fills with no intent in the ledger (venue stops, liquidations) or no simulator spread."""
    slippage: SlippageBand
    live_total_return: float
    replica_total_return: float | None
    return_gap: float | None
    """``live_total_return - replica_total_return``; ``None`` without a replica."""
    label: Literal["descriptive, not inferential"] = "descriptive, not inferential"


def model_price(intent: OrderIntent, half_spread: Decimal) -> Decimal:
    """The price the simulator fills ``intent`` at (``armsim._taker_price``)."""
    one = Decimal(1)
    factor = one + half_spread if intent.side is Side.BUY else one - half_spread
    return intent.reference_price * factor


def slippage_bps(fill: Fill, intent: OrderIntent, half_spread: Decimal) -> float:
    """Positive when the venue filled worse than the simulator assumed."""
    model = model_price(intent, half_spread)
    sign = Decimal(1) if fill.side is Side.BUY else Decimal(-1)
    return float(sign * (fill.exec_price / model - 1) * BPS)


def _band(values: Sequence[float]) -> SlippageBand:
    if not values:
        return SlippageBand(n=0, mean_bps=None, median_bps=None, p95_bps=None, worst_bps=None)
    ordered = sorted(values)
    return SlippageBand(
        n=len(ordered),
        mean_bps=math.fsum(ordered) / len(ordered),
        median_bps=float(statistics.median(ordered)),
        p95_bps=percentile(ordered, 0.95),
        worst_bps=ordered[-1],
    )


def simulator_check(
    fills: Sequence[Fill],
    intent_of: Callable[[str], OrderIntent | None],
    half_spread: Callable[[str], Decimal],
    *,
    live: ArmResult,
    replica: ArmResult | None,
) -> SimulatorCheck:
    """The gap between the venue and the cost model, over every fill of the live book."""
    priced: list[float] = []
    unpriced = 0
    for fill in fills:
        intent = intent_of(fill.client_oid) if fill.client_oid else None
        if intent is None or intent.symbol != fill.symbol:
            unpriced += 1
            continue
        try:
            half = half_spread(fill.symbol)
        except ValueError:
            unpriced += 1
            continue
        priced.append(slippage_bps(fill, intent, half))
    live_return = live.metrics.total_return
    replica_return = None if replica is None else replica.metrics.total_return
    return SimulatorCheck(
        ranked_arm_id=live.spec.arm_id if replica is None else replica.spec.arm_id,
        fills_priced=len(priced),
        fills_unpriced=unpriced,
        slippage=_band(priced),
        live_total_return=live_return,
        replica_total_return=replica_return,
        return_gap=None if replica_return is None else live_return - replica_return,
    )


__all__ = [
    "SimulatorCheck",
    "SlippageBand",
    "model_price",
    "simulator_check",
    "slippage_bps",
]
