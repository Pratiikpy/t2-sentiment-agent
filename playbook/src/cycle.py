"""One live run: perceive, protect, wake the model on an admitted trigger, rule, plan, execute.

The order is the primary's (DESIGN.md §4), compressed into one ``CRON_MINUTES`` run:

0. A follow-trade run is refused before anything is read or written (``execution.py``: nothing
   proves the subscriber's subaccount is paper). The record says why; no order can be sent.
1. Read ``.state/``: the book, the breaker, the trigger book, the outage flag.
2. Follow-trade only, and unreachable while step 0 refuses it: read the venue's positions and
   adopt any change the book did not make, recorded as such.
3. Perceive: marks and funding, mood, BTCUSDT positioning, crowd counts, the calendar; on a
   decision run also the 1H bars, quotes and forum counts of every configured instrument.
4. Mark the book, roll the UTC day, move the peak; assess the breaker.
5. Admit triggers. With one admitted: build the facts and the prompt, obtain a decision, ground its
   figures, assess the breaker again with the decision, and rule on its proposal with every guard.
   With none: the protective ruling (G1, G2, G5, G10) on what is held.
6. Plan the approved weights, reducing symbols first. For each symbol with legs, emit its signal
   through ``runtime.emit_signal_or_follow``; its trade callback raises if a runtime ever invokes
   it (follow-trade is refused). The replica fills its own paper book.
7. Emit one ``watch`` summary carrying the whole cycle record, and save the state.

Every record carries ``policy_hash`` (the primary's ``POLICY_V1`` digest) and declares this log the
secondary one; the primary's Agent Hub Demo ledger is primary. Hashes of the emitted canonical
intents are taken outside the sandbox (``scripts/build_playbook.py --hash-signals``).
"""

import json
import math
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from . import breaker as breakers
from . import kernel as rules
from . import policy_v1 as policy
from .book import STATE_DIR, Book, load_state, save_state
from .canonical import canonical, compact
from .decision import (
    DEFERRED,
    BookFacts,
    as_number,
    obtain_decision,
    proposed_weights,
    targets_of,
)
from .execution import FOLLOW_TRADE_REFUSAL, ProxyVenue, execute_follow, execute_virtual
from .grounding import ground_targets
from .perception import Deadline, Perception, describe_quote
from .planner import SymbolPlan, plan_ruling
from .prompt import build_facts, system_prompt, user_prompt
from .sessions import as_utc, hours_to_freeze, iso_z, utc_now, weekend_phase
from .triggers import TriggerBook

SUMMARY_MAX_CHARS = 200_000
"""The summary signal carries the whole cycle record; past this size its perception detail is
reduced to the source health list, so one oversized run cannot lose the record."""

PERCEPTION_UNTIL = 70.0
DECISION_PERCEPTION_UNTIL = 100.0
MODEL_UNTIL = 150.0
EXECUTION_UNTIL = 168.0

UNIVERSE_SYMBOLS: tuple[str, ...] = tuple(s for s, _, _ in policy.UNIVERSE)
ASSET_CLASS: dict[str, str] = {s: a for s, a, _ in policy.UNIVERSE}


@dataclass
class Env:
    runtime: Any
    llm: Any
    trade: Any
    clock: Callable[[], datetime] = utc_now
    state_dir: Path = STATE_DIR


@dataclass
class Config:
    configured: tuple[str, ...]
    margin_budget: Decimal | None
    problems: list[str] = field(default_factory=list)


def read_config(manifest: Any) -> Config:
    """The subscriber's configuration, narrowed to the policy universe. A symbol outside the policy
    universe is dropped (the kernel would exit it anyway); a missing or non-positive margin budget
    stops decisions for the run. ``manifest`` is ``runtime.manifest``, a lazy mapping read with
    ``.get`` as the SDK documents."""
    getter = getattr(manifest, "get", None)
    raw_cfg = getter("strategy_config") if callable(getter) else None
    cfg: Mapping[str, object] = raw_cfg if isinstance(raw_cfg, Mapping) else {}
    top_symbols = getter("trading_symbols") if callable(getter) else None
    raw_symbols = cfg.get("trading_symbols") or top_symbols or []
    problems: list[str] = []
    symbols: list[str] = []
    for item in raw_symbols if isinstance(raw_symbols, list) else []:
        symbol = str(item).strip().upper()
        if symbol in UNIVERSE_SYMBOLS and symbol not in symbols:
            symbols.append(symbol)
        elif symbol:
            problems.append(f"{symbol} is not in the policy universe and is ignored")
    budget: Decimal | None
    try:
        budget = Decimal(str(cfg.get("margin_budget")))
    except (InvalidOperation, ValueError):
        budget = None
    if budget is None or not budget.is_finite() or budget <= 0:
        problems.append("margin_budget is missing or not positive: no order is sized this run")
        budget = None
    ordered = tuple(s for s in UNIVERSE_SYMBOLS if s in symbols)
    return Config(configured=ordered, margin_budget=budget, problems=problems)


def _limits(price_steps: Mapping[str, Decimal]) -> dict[str, rules.Limits]:
    out: dict[str, rules.Limits] = {}
    for symbol, (min_qty, qty_step, min_amount) in policy.INSTRUMENT_LIMITS.items():
        out[symbol] = rules.Limits(
            min_qty=Decimal(min_qty),
            qty_step=Decimal(qty_step),
            min_amount=Decimal(min_amount),
            price_step=price_steps.get(symbol),
        )
    return out


def _position_facts(book: Book, symbol: str, now: datetime) -> dict[str, float]:
    position = book.positions.get(symbol)
    facts: dict[str, float] = {}
    orders = book.rebalances_today.get(symbol, 0)
    if orders:
        facts["model_orders_today"] = orders
        facts["model_orders_left_today"] = max(0, policy.MAX_REBALANCES_PER_NAME_PER_DAY - orders)
    if position is None or position.qty == 0:
        return facts
    mark = book.mark_price(symbol)
    weight = book.weight(symbol)
    facts["position_qty"] = float(position.qty)
    facts["position_avg_entry"] = float(position.avg_entry)
    if weight is not None:
        facts["position_weight_pct"] = weight * 100
        facts["position_target_equivalent"] = weight / policy.PER_NAME_MAX
    if mark is not None and position.avg_entry > 0:
        direction = 1.0 if position.qty > 0 else -1.0
        facts["position_move_since_entry_pct"] = (
            float(mark / position.avg_entry - 1) * 100 * direction
        )
    facts["position_unrealized_pnl"] = float(book.unrealized(symbol))
    facts["position_hours_held"] = (now - position.opened_at).total_seconds() / 3600
    if position.last_increase_at is not None:
        since = (now - position.last_increase_at).total_seconds() / 3600
        facts["position_hours_since_increase"] = since
        facts["position_hours_until_increase_allowed"] = max(0.0, policy.MIN_HOLD_HOURS - since)
    if position.stop is not None:
        facts["position_stop_price"] = float(position.stop)
    return facts


def _signal_number(value: object) -> float | int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, (float, Decimal)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


class Run:
    """One run's working set. Built and driven by :func:`run_live`."""

    def __init__(self, env: Env):
        self.env = env
        self.now = as_utc(env.clock())
        self.deadline = Deadline(started=self.now, clock=env.clock)
        self.run_id = str(getattr(env.runtime, "run_id", "") or f"local-{iso_z(self.now)}")
        self.follow = bool(env.runtime.is_follow_trade())
        self.mode = "follow_trade" if self.follow else "signal_only"
        self.config = read_config(env.runtime.manifest)
        self.record: dict[str, object] = {
            "record": "t2sa-replica-cycle",
            "evidence_role": policy.EVIDENCE_ROLE,
            "primary_log": policy.PRIMARY_LOG,
            "policy_version": policy.POLICY_VERSION,
            "policy_hash": policy.POLICY_HASH,
            "package": policy.PACKAGE_NAME,
            "run_id": self.run_id,
            "mode": self.mode,
            "started_at": iso_z(self.now),
            "configured": list(self.config.configured),
            "config_problems": list(self.config.problems),
        }
        self.seq = 0
        self.emitted = 0

    # --- state ------------------------------------------------------------------------------------

    def load(self) -> None:
        budget = self.config.margin_budget or Decimal(1)
        loaded = load_state(
            directory=self.env.state_dir, mode=self.mode, margin_budget=budget, now=self.now
        )
        self.book = loaded.book
        self.state_existed = loaded.existed
        extras = loaded.raw
        breaker_raw = extras.get("breaker") if loaded.book.status == "loaded" else None
        self.breaker = breakers.breaker_from_json(breaker_raw, self.now)
        if loaded.book.status == "corrupt":
            self.breaker = breakers.BreakerState(
                activation=breakers.HALTED, since=self.now, trips=(breakers.UNREADABLE_STATE,)
            )
        outage_raw = extras.get("llm_outage") if loaded.book.status == "loaded" else False
        self.llm_outage = outage_raw is True
        triggers_raw = extras.get("triggers") if loaded.book.status == "loaded" else None
        self.trigger_state: MutableMapping[str, object] = (
            dict(triggers_raw) if isinstance(triggers_raw, dict) else {}
        )
        self.record["state"] = {
            "existed": loaded.existed,
            "status": loaded.book.status,
            "problem": loaded.problem,
        }

    def save(self) -> None:
        save_state(
            self.env.state_dir,
            book=self.book,
            extra={
                "breaker": self.breaker.to_json(),
                "llm_outage": self.llm_outage,
                "triggers": dict(self.trigger_state),
                "last_run_id": self.run_id,
                "last_run_at": iso_z(self.now),
            },
        )

    # --- venue (follow-trade) ---------------------------------------------------------------------

    def reconcile(self) -> None:
        self.venue = ProxyVenue(self.env.trade) if self.follow else None
        if self.venue is None:
            return
        symbols = tuple(sorted(set(UNIVERSE_SYMBOLS) | set(self.book.positions)))
        try:
            venue_positions = self.venue.all_positions(symbols)
        except Exception as exc:  # no venue read: nothing is adopted, increases stay refused
            self.record["reconcile"] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            self.venue_positions_known = False
            return
        self.venue_positions_known = True
        holds = {s: q for s, (q, _) in venue_positions.items() if q != 0}
        if self.book.status in ("fresh", "mode_changed", "corrupt") and holds:
            self.book.lose_history(
                self.now,
                f"state {self.book.status} while the venue holds {', '.join(sorted(holds))}",
            )
            if self.book.status != "corrupt":
                self.book.status = "lost"
        changes = self.book.reconcile(venue_positions, now=self.now, symbols=symbols)
        self.record["reconcile"] = {"changes": changes}
        state = self.record.get("state")
        if isinstance(state, dict):
            state["status"] = self.book.status

    # --- perception -------------------------------------------------------------------------------

    def perceive(self) -> Perception:
        perception = Perception(
            clock=self.env.clock,
            deadline=self.deadline,
            symbols=self.config.configured or UNIVERSE_SYMBOLS,
        )
        perception.marks(until=PERCEPTION_UNTIL)
        perception.mood(until=PERCEPTION_UNTIL)
        perception.btc_positioning(until=PERCEPTION_UNTIL)
        perception.crowd(until=PERCEPTION_UNTIL)
        equities = [s for s in self.config.configured if ASSET_CLASS.get(s) == "us_equity"]
        perception.earnings(equities, until=PERCEPTION_UNTIL)
        held_equities = {
            s: p.opened_at
            for s, p in self.book.positions.items()
            if p.qty != 0 and ASSET_CLASS.get(s) == "us_equity"
        }
        perception.filings(held_equities, until=PERCEPTION_UNTIL)
        if self.follow:
            perception.tickers(
                sorted(s for s, p in self.book.positions.items() if p.qty != 0),
                until=PERCEPTION_UNTIL,
            )
        return perception

    def marks(self, perception: Perception) -> dict[str, Decimal]:
        return {
            s: q.mark
            for s, q in perception.snapshot.quotes.items()
            if q.mark is not None and q.mark > 0
        }

    # --- breaker ----------------------------------------------------------------------------------

    def assess(self, *, valid_decision: bool, snapshot_at: datetime | None) -> None:
        view = self.book.view()
        after, transition = breakers.assess(
            self.breaker,
            now=self.now,
            equity=float(view.equity),
            drawdown=view.drawdown,
            day_return=view.day_return,
            consecutive_losses=view.consecutive_losses,
            llm_outage=self.llm_outage,
            snapshot_taken_at=snapshot_at,
            valid_decision=valid_decision,
            history_lost=not view.history_known,
        )
        self.breaker = after
        if transition is not None:
            transitions = self.record.setdefault("breaker_transitions", [])
            if isinstance(transitions, list):
                transitions.append(transition.to_json())

    # --- the kernel's inputs ----------------------------------------------------------------------

    def inputs(
        self, perception: Perception, *, venue_prices: Mapping[str, Decimal]
    ) -> rules.Inputs:
        snapshot = perception.snapshot
        return rules.Inputs(
            at=self.now,
            quotes=snapshot.quotes,
            move_bps_3h=snapshot.move_bps_3h(),
            limits=self.limits,
            snapshot_taken_at=snapshot.taken_at,
            configured=self.config.configured,
            venue_last=venue_prices,
            venue_is_data_layer=not self.follow,
        )

    def venue_prices(self, symbols: Sequence[str]) -> dict[str, Decimal]:
        prices: dict[str, Decimal] = {}
        if self.venue is None:
            return prices
        for symbol in symbols:
            try:
                price = self.venue.price(symbol)
            except Exception:  # noqa: S112 - a missing venue price fails G1 closed for that symbol
                continue
            if price is not None and price > 0:
                prices[symbol] = price
        return prices

    def price_steps(self, symbols: Sequence[str]) -> dict[str, Decimal]:
        steps: dict[str, Decimal] = {}
        if self.venue is None:
            return steps
        for symbol in symbols:
            try:
                step = self.venue.price_step(symbol)
            except Exception:  # noqa: S112 - no venue grid: G4 cannot place a stop, no increase
                continue
            if step is not None:
                steps[symbol] = step
        return steps

    # --- the decision -----------------------------------------------------------------------------

    def decide(
        self, perception: Perception, admitted: Sequence[Any], triggers: TriggerBook
    ) -> tuple[rules.KernelRuling | None, dict[str, object]]:
        configured = self.config.configured
        perception.klines(configured, until=DECISION_PERCEPTION_UNTIL)
        perception.tickers(configured, until=DECISION_PERCEPTION_UNTIL)
        perception.btc_ratios(until=DECISION_PERCEPTION_UNTIL)
        perception.forums(until=DECISION_PERCEPTION_UNTIL)
        snapshot = perception.snapshot
        self.book.marks.update(self.marks(perception))
        held = self.book.held()
        weights = {s: w for s in held if (w := self.book.weight(s)) is not None}
        facts_input = self._fact_inputs(perception)
        facts = build_facts(**facts_input, configured=configured)
        compact_facts = build_facts(**facts_input, configured=configured, compact=True)
        session_text = (
            f"now {iso_z(self.now)}; weekend phase {weekend_phase(self.now)}; "
            f"breaker {self.breaker.activation}"
        )
        sources_text = (
            "; ".join(f"{s.name} {s.health}" for s in snapshot.sources if s.health != "ok")
            or "every source answered"
        )
        trigger_rows = [t.to_json() for t in admitted]
        system = system_prompt(configured)
        held_words = {s: ("long" if side > 0 else "short") for s, side in held.items()}
        user = user_prompt(
            triggers=trigger_rows,
            facts=facts,
            configured=configured,
            held=held_words,
            session_text=session_text,
            sources_text=sources_text,
        )

        def compact_prompt() -> tuple[str, str]:
            return system, user_prompt(
                triggers=trigger_rows,
                facts=compact_facts,
                configured=configured,
                held=held_words,
                session_text=session_text,
                sources_text=sources_text,
            )

        book_facts = BookFacts(held=held, weights=weights, configured=configured)
        available = False
        try:
            available = bool(self.env.llm.is_available())
        except Exception:  # an llm module that cannot say is not available
            available = False
        decision, call = obtain_decision(
            chat=self.env.llm.chat,
            available=available,
            system=system,
            user=user,
            compact_user=compact_prompt,
            book=book_facts,
            time_left=lambda: MODEL_UNTIL - self.deadline.elapsed(),
        )
        record: dict[str, object] = {
            "triggers": trigger_rows,
            "call": call.to_json(),
            "prompt": {"system_chars": len(system), "user_chars": len(user), "facts": len(facts)},
        }
        if call.outcome == DEFERRED:
            triggers.defer(admitted)
            record["deferred"] = True
            return None, record
        if decision is None:
            self.llm_outage = True
            self.assess(valid_decision=False, snapshot_at=snapshot.taken_at)
            return None, record
        self.llm_outage = False
        decision_ref = f"{self.run_id}:decision"
        record["decision_ref"] = decision_ref
        record["decision"] = decision
        record["decision_canonical"] = canonical(decision)
        grounding = ground_targets(
            targets_of(decision),
            facts if not call.compact else compact_facts,
            policy.GROUNDING_TOLERANCE,
        )
        record["grounding"] = {s: r.to_json() for s, r in grounding.items()}
        proposal = proposed_weights(decision, book_facts)
        record["proposed_weights"] = proposal
        invalidation = {
            str(t["symbol"]): bool(t["invalidation_triggered"]) for t in targets_of(decision)
        }
        self.assess(valid_decision=True, snapshot_at=snapshot.taken_at)
        adding = [s for s, w in proposal.items() if abs(w) > abs(weights.get(s, 0.0))]
        self.limits = _limits(self.price_steps(adding))
        prices = self.venue_prices(sorted(set(adding) | set(held)))
        ruling = rules.rule(
            proposed=proposal,
            decision_ref=decision_ref,
            book=self.book.view(),
            inputs=self.inputs(perception, venue_prices=prices),
            breaker=self.breaker,
            grounding=grounding,
            invalidation_fired=invalidation,
        )
        return ruling, record

    def _fact_inputs(self, perception: Perception) -> dict[str, Any]:
        snapshot = perception.snapshot
        features: dict[str, dict[str, float]] = {}
        for symbol in self.config.configured:
            values = dict(snapshot.features.get(symbol, {}))
            values.pop("ticker_change_24h_pct", None)
            values.pop("last_close_1h", None)
            quote = snapshot.quotes.get(symbol)
            if quote is not None:
                for name, value in (
                    ("mark", quote.mark),
                    ("index", quote.index),
                    ("last", quote.last),
                ):
                    if value is not None:
                        values[name] = float(value)
                gap = quote.mark_index_gap
                if gap is not None and math.isfinite(gap):
                    values["mark_index_gap_bps"] = gap * 10_000
                spread = quote.spread_bps
                if spread is not None and math.isfinite(spread):
                    values["spread_bps"] = spread
            rate = snapshot.funding_rate.get(symbol)
            if rate is not None:
                values["funding_rate"] = rate
            if symbol == "BTCUSDT" and snapshot.oi_jump_threshold_pct is not None:
                values["oi_jump_threshold_pct"] = snapshot.oi_jump_threshold_pct
            crowd = snapshot.crowd
            if crowd is not None:
                values["news_stories_24h"] = crowd.news_mentions.get(symbol, 0)
                coordinated = len(crowd.coordinated_for(symbol))
                if coordinated:
                    values["coordinated_stories"] = coordinated
                values.update(crowd.forum_mentions.get(symbol, {}))
                values.pop("forum_mentions_prior_24h", None)
            for entry in snapshot.earnings:
                if entry.symbol == symbol and entry.at >= self.now:
                    hours = (entry.at - self.now).total_seconds() / 3600
                    values["hours_to_earnings"] = min(hours, values.get("hours_to_earnings", hours))
            filings = [f for f in snapshot.filings if f.symbol == symbol]
            if filings:
                values["insider_filings_since_open"] = len(filings)
            gap_limit = next((g for s, _, g in policy.UNIVERSE if s == symbol), None)
            if self.follow and gap_limit is not None:
                values["venue_gap_limit_bps"] = gap_limit
            features[symbol] = values
        view = self.book.view()
        equity = view.equity
        gross = sum(abs(w) for s in self.book.positions if (w := self.book.weight(s)) is not None)
        net = sum(w for s in self.book.positions if (w := self.book.weight(s)) is not None)
        book_facts: dict[str, float] = {
            "equity": float(equity),
            "starting_equity": float(view.starting_equity),
            "gross_weight_pct": gross * 100,
            "net_weight_pct": net * 100,
            "risk_budget_used_pct": gross * 100,
            "drawdown_pct": view.drawdown * 100,
            "consecutive_losses": view.consecutive_losses,
        }
        if view.day_return is not None:
            book_facts["day_return_pct"] = view.day_return * 100
        if view.day_open_equity > 0:
            book_facts["fees_today_bps"] = float(view.fees_today / view.day_open_equity * 10_000)
        if view.starting_equity > 0:
            book_facts["fees_window_bps"] = float(view.fees_total / view.starting_equity * 10_000)
        crowd_facts: dict[str, float] = {}
        if snapshot.crowd is not None:
            crowd_facts = {
                "news_items": snapshot.crowd.items,
                "stories": snapshot.crowd.stories,
                "coordinated_clusters": sum(1 for c in snapshot.crowd.clusters if c.coordinated),
            }
        return {
            "features": features,
            "mood": dict(snapshot.mood),
            "book": book_facts,
            "positions": {
                s: _position_facts(self.book, s, self.now) for s in self.config.configured
            },
            "crowd": crowd_facts,
            "session": {"hours_to_weekend_freeze": hours_to_freeze(self.now)},
        }

    # --- execution --------------------------------------------------------------------------------

    def execute(
        self,
        plans: Sequence[SymbolPlan],
        perception: Perception,
        decision: Mapping[str, object] | None,
    ) -> list[dict[str, object]]:
        runtime = self.env.runtime
        confidences: dict[str, float] = {}
        if decision is not None:
            for target in targets_of(decision):
                confidences[str(target["symbol"])] = as_number(target["confidence"])
        out: list[dict[str, object]] = []
        decision_ref = None if decision is None else f"{self.run_id}:decision"
        for plan in plans:
            entry = plan.to_json(run_id=self.run_id, first_seq=self.seq, decision_ref=decision_ref)
            first_seq = self.seq
            self.seq += len(plan.legs)
            if not plan.legs:
                entry["executions"] = []
                out.append(entry)
                continue
            if not self.deadline.before(EXECUTION_UNTIL):
                entry["executions"] = []
                entry["not_sent"] = "the run is out of time; the next run re-rules this symbol"
                out.append(entry)
                continue
            executed: list[dict[str, object]] = []

            def execute_trade(
                plan: SymbolPlan = plan,
                first_seq: int = first_seq,
                executed: list[dict[str, object]] = executed,
            ) -> list[dict[str, object]]:
                if not self.follow:
                    return executed
                records = execute_follow(
                    plan,
                    book=self.book,
                    trade=self.env.trade,
                    at=lambda: as_utc(self.env.clock()),
                    first_seq=first_seq,
                    save=self.save,
                )
                executed.extend(r.to_json() for r in records)
                return executed

            metrics: dict[str, object] = {
                "target_weight": plan.ruling.approved,
                "current_weight": plan.ruling.current,
                "proposed_weight": plan.ruling.proposed,
                "legs": len(plan.legs),
            }
            meta = {
                "record": "t2sa-replica-order",
                "evidence_role": policy.EVIDENCE_ROLE,
                "policy_hash": policy.POLICY_HASH,
                "run_id": self.run_id,
                "decision_ref": decision_ref,
                "binding_guard": plan.ruling.binding_guard,
                "intents": entry["intents"],
            }
            runtime.emit_signal_or_follow(
                action=plan.signal_action,
                symbol=plan.symbol,
                confidence=confidences.get(plan.symbol, 0.0),
                metrics={k: n for k, v in metrics.items() if (n := _signal_number(v)) is not None},
                meta=compact(meta),
                execute_trade=execute_trade,
            )
            self.emitted += 1
            if not self.follow:
                records = execute_virtual(
                    plan,
                    book=self.book,
                    quote=perception.snapshot.quotes.get(plan.symbol),
                    at=self.now,
                    first_seq=first_seq,
                )
                executed.extend(r.to_json() for r in records)
                self.save()
            entry["executions"] = list(executed)
            out.append(entry)
        return out

    # --- the whole run ----------------------------------------------------------------------------

    def run(self) -> dict[str, object]:
        if self.follow:
            # Refused before any state, venue or model call: see execution.FOLLOW_TRADE_REFUSAL.
            self.record["follow_trade_refused"] = FOLLOW_TRADE_REFUSAL
            return self.record
        self.load()
        self.limits = _limits({})
        self.reconcile()
        perception = self.perceive()
        snapshot = perception.snapshot
        self.book.open_run(self.now, self.marks(perception))
        self.assess(valid_decision=False, snapshot_at=snapshot.taken_at)
        triggers = TriggerBook(self.trigger_state, self.now, stateless=not self.state_existed)
        pending = triggers.take_pending()
        candidates = triggers.due_heartbeats()
        held_since = {s: p.opened_at for s, p in self.book.positions.items() if p.qty != 0}
        candidates += triggers.events(snapshot, held_since)
        admitted, refused = triggers.admit(candidates)
        admitted = [*pending, *admitted]
        self.record["triggers"] = {
            "admitted": [t.to_json() for t in admitted],
            "refused": [{**t.to_json(), "reason": reason} for t, reason in refused],
            "stateless": triggers.stateless,
        }
        ruling: rules.KernelRuling | None = None
        decision_record: dict[str, object] | None = None
        venue_known = not self.follow or getattr(self, "venue_positions_known", False)
        can_decide = (
            venue_known and self.config.margin_budget is not None and bool(self.config.configured)
        )
        if not venue_known:
            self.record["trading_disabled"] = "the venue's positions could not be read this run"
        elif not can_decide:
            self.record["decisions_disabled"] = "no margin budget or no configured symbol"
        if admitted and can_decide:
            ruling, decision_record = self.decide(perception, admitted, triggers)
            self.record["decision"] = decision_record
        elif admitted:
            triggers.defer(admitted)
        if ruling is None and venue_known:
            prices = self.venue_prices(
                sorted(s for s, p in self.book.positions.items() if p.qty != 0)
            )
            ruling = rules.protective(
                book=self.book.view(),
                inputs=self.inputs(perception, venue_prices=prices),
                breaker=self.breaker,
                llm_outage=self.llm_outage,
            )
        plans: list[SymbolPlan] = []
        if ruling is not None:
            self.record["ruling"] = ruling.to_json()
            held_qty = {s: p.qty for s, p in self.book.positions.items() if p.qty != 0}
            entries = {s: p.avg_entry for s, p in self.book.positions.items() if p.avg_entry > 0}
            plans = plan_ruling(
                ruling,
                held=held_qty,
                entries=entries,
                equity=self.book.equity(),
                inputs=self.inputs(perception, venue_prices={}),
                limits=self.limits,
            )
            decision = decision_record.get("decision") if decision_record else None
            self.record["plans"] = self.execute(
                plans, perception, decision if isinstance(decision, Mapping) else None
            )
        self.book.open_run(self.now, self.book.marks)
        self.record["book"] = self.book.summary()
        self.record["breaker"] = self.breaker.to_json()
        self.record["llm_outage"] = self.llm_outage
        self.record["perception"] = {
            "taken_at": iso_z(snapshot.taken_at),
            "sources": snapshot.coverage(),
            "quotes": {s: describe_quote(q) for s, q in sorted(snapshot.quotes.items())},
            "mood": dict(snapshot.mood),
            "features": {s: dict(v) for s, v in sorted(snapshot.features.items())},
            "oi_jump_threshold_pct": snapshot.oi_jump_threshold_pct,
            "crowd": None
            if snapshot.crowd is None
            else {
                "items": snapshot.crowd.items,
                "stories": snapshot.crowd.stories,
                "coordinated": [
                    {
                        "cluster_id": c.cluster_id,
                        "size": c.size,
                        "sources": c.sources,
                        "symbols": list(c.symbols),
                    }
                    for c in snapshot.crowd.clusters
                    if c.coordinated
                ],
            },
        }
        self.record["finished_at"] = iso_z(as_utc(self.env.clock()))
        self.record["elapsed_seconds"] = round(self.deadline.elapsed(), 3)
        self.save()
        return self.record


def summary_metrics(record: Mapping[str, object]) -> dict[str, float | int]:
    book = record.get("book")
    metrics: dict[str, float | int] = {}
    if isinstance(book, Mapping):
        for key in (
            "equity",
            "return_pct",
            "drawdown",
            "day_return",
            "fees_total",
            "closed_trades",
            "winning_trades",
            "consecutive_losses",
        ):
            value = _signal_number(book.get(key))
            if value is not None:
                metrics[key] = value
        positions = book.get("positions")
        if isinstance(positions, Mapping):
            metrics["open_positions"] = len(positions)
            gross = 0.0
            for position in positions.values():
                weight = position.get("weight") if isinstance(position, Mapping) else None
                if isinstance(weight, (int, float)) and math.isfinite(weight):
                    gross += abs(weight)
            metrics["gross_weight"] = gross
    decision = record.get("decision")
    if isinstance(decision, Mapping):
        call = decision.get("call")
        if isinstance(call, Mapping):
            attempts = call.get("attempts")
            metrics["llm_attempts"] = attempts if isinstance(attempts, int) else 0
    plans = record.get("plans")
    planned = 0
    for plan in plans if isinstance(plans, list) else []:
        intents = plan.get("intents") if isinstance(plan, Mapping) else None
        planned += len(intents) if isinstance(intents, list) else 0
    metrics["orders_planned"] = planned
    return metrics


def run_live(env: Env) -> dict[str, object]:
    """Run once and emit the summary signal. Returns the cycle record."""
    run = Run(env)
    try:
        record = run.run()
    except Exception as exc:  # a defect must still leave a record and the state as it stood
        record = run.record
        record["error"] = f"{type(exc).__name__}: {str(exc)[:1000]}"
        try:
            run.save()
        except Exception as save_exc:  # the state directory itself failed
            record["save_error"] = f"{type(save_exc).__name__}: {str(save_exc)[:300]}"
    stance = None
    decision = record.get("decision")
    if isinstance(decision, Mapping):
        inner = decision.get("decision")
        if isinstance(inner, Mapping):
            stance = inner.get("stance")
    anchor = run.config.configured[0] if run.config.configured else UNIVERSE_SYMBOLS[0]
    meta = compact({**record, "stance": stance})
    if isinstance(meta, dict) and len(json.dumps(meta, default=str)) > SUMMARY_MAX_CHARS:
        perceived = meta.get("perception")
        sources = perceived.get("sources") if isinstance(perceived, dict) else None
        meta["perception"] = {"sources": sources, "trimmed": "over the summary size limit"}
    env.runtime.emit_signal(
        action="watch",
        symbol=anchor,
        confidence=0.0,
        metrics=summary_metrics(record),
        meta=meta,
    )
    return record
