"""Decision cards, read out of a ledger the real pipeline wrote (``tests/site_world.py``)."""

from datetime import timedelta
from decimal import Decimal

import pytest

from helpers import T0
from sentiment_agent.book.projection import Projection
from sentiment_agent.hashing import sha256_hex
from sentiment_agent.ledger.chain import referenced_blobs
from sentiment_agent.policy import POLICY_V1
from sentiment_agent.site.cards import (
    CARD_ID,
    CardError,
    build_card_bundle,
    build_cards,
    decision_card_id,
    protective_card_id,
    realized_pnl_by_fill,
)
from sentiment_agent.types import (
    BlobRef,
    DecisionCard,
    DecisionEvent,
    EventKind,
    Fill,
    FillVenue,
    GuardId,
    LedgerEvent,
    LlmOutcome,
    OrderState,
    ProtectiveReason,
    Side,
    SnapshotEvent,
    Stance,
    parse_payload,
)
from site_world import BTC, NVDA, Site


def cards_of(site: Site) -> tuple[DecisionCard, ...]:
    return build_cards(site.world.projection(), site.world.blobs)


def by_decision(site: Site) -> dict[str, DecisionCard]:
    return {c.decision_id: c for c in cards_of(site) if c.decision_id is not None}


def event_at(site: Site, seq: int) -> LedgerEvent:
    for event in site.world.ledger.events():
        if event.seq == seq:
            return event
    raise AssertionError(f"no seq {seq}")


# ------------------------------------------------------------------------------------------------
# One card per decision, abstention, outage and protective ruling
# ------------------------------------------------------------------------------------------------


def test_one_card_per_decision_and_per_protective_ruling(site: Site) -> None:
    cards = cards_of(site)
    decision_ids = [c.decision_id for c in cards if c.decision_id is not None]
    protective = [c.ruling_id for c in cards if c.decision_id is None]
    assert decision_ids == site.world.decision_ids
    assert protective == site.world.protective_ruling_ids
    assert len(cards) == len(site.world.decision_ids) + len(site.world.protective_ruling_ids)
    assert [c.at for c in cards] == sorted(c.at for c in cards)
    assert len({c.card_id for c in cards}) == len(cards)
    assert all(CARD_ID.fullmatch(c.card_id) for c in cards)


def test_a_decision_card_carries_every_required_field(site: Site) -> None:
    card = by_decision(site)[site.world.decision_ids[0]]
    decision = card.decision
    assert decision is not None
    assert card.outcome is LlmOutcome.DECIDED
    assert {t.kind.value for t in card.triggers} == {"heartbeat_funding", "funding_zscore"}
    assert card.coverage, "the health of every source call the snapshot made"
    assert card.shown_text, "the text the model was shown"
    assert any(item.withheld for item in card.shown_text), "the injection was withheld"
    for target in decision.targets:
        assert target.thesis
        assert target.invalidation
        assert target.crowd_belief
        assert target.our_view
    assert decision.rejected_alternatives
    assert set(card.grounding) == {NVDA, BTC}
    kernel = card.kernel
    assert kernel is not None
    assert kernel.decision_id == card.decision_id
    assert card.ruling_id == kernel.ruling_id
    assert all(inst.rulings for inst in kernel.instruments), "every guard's ruling, per leg"
    assert all(r.basis for inst in kernel.instruments for r in inst.rulings)
    assert len(card.orders) == 2
    assert {p.client_oid for p in card.previews} == {o.client_oid for o in card.orders}
    for preview in card.previews:
        assert "--paper-trading" in preview.argv
        assert "--dry-run" in preview.argv
        assert preview.would_send["clientOid"] == preview.client_oid
    for order in card.orders:
        assert order.venue_order_id
        assert order.venue_order_id.startswith("sim-order-")
        assert order.state is OrderState.FILLED
        assert order.fills
        assert all(f.client_oid == order.client_oid for f in order.fills)
        assert order.net_pnl is not None


def test_ledger_seqs_are_the_events_the_card_was_read_from(site: Site) -> None:
    card = by_decision(site)[site.world.decision_ids[0]]
    kinds = [event_at(site, seq).kind for seq in card.ledger_seqs]
    assert kinds.count(EventKind.DECISION) == 1
    assert kinds.count(EventKind.SNAPSHOT) == 1
    assert kinds.count(EventKind.TRIGGER) == 2
    assert kinds.count(EventKind.KERNEL_RULING) == 1
    assert kinds.count(EventKind.ORDER_PLAN) == 1
    assert kinds.count(EventKind.ORDER_PREVIEW) == 2
    assert kinds.count(EventKind.ORDER_SUBMITTED) == 2
    assert kinds.count(EventKind.ORDER_ACK) == 2
    assert kinds.count(EventKind.FILL) == 2
    decision_event = next(
        event_at(site, s) for s in card.ledger_seqs if event_at(site, s).kind is EventKind.DECISION
    )
    payload = parse_payload(decision_event)
    assert isinstance(payload, DecisionEvent)
    assert payload.record.decision_id == card.decision_id
    snapshot_event = next(
        event_at(site, s) for s in card.ledger_seqs if event_at(site, s).kind is EventKind.SNAPSHOT
    )
    snap = parse_payload(snapshot_event)
    assert isinstance(snap, SnapshotEvent)
    assert snap.snapshot.snapshot_id == payload.record.snapshot_id
    assert card.ledger_seqs == tuple(sorted(set(card.ledger_seqs)))


def test_every_card_blob_resolves_and_is_every_blob_its_events_commit_to(site: Site) -> None:
    for card in cards_of(site):
        committed: set[str] = set()
        for seq in card.ledger_seqs:
            committed |= {ref.sha256 for ref in referenced_blobs(event_at(site, seq))}
        assert {b.sha256 for b in card.blobs} == committed
        for ref in card.blobs:
            data = site.world.blobs.get(ref.sha256)
            assert sha256_hex(data) == ref.sha256
            assert len(data) == ref.size
    first = by_decision(site)[site.world.decision_ids[0]]
    media = {b.media_type for b in first.blobs}
    assert "application/json" in media, "the raw source responses behind the snapshot"
    assert len(first.blobs) >= 3, "source responses and the model's request and completion"


def test_the_kernel_cut_is_on_its_card(site: Site) -> None:
    card = by_decision(site)[site.world.decision_ids[1]]
    kernel = card.kernel
    assert kernel is not None
    assert kernel.changed_by_kernel
    inst = kernel.instrument(NVDA)
    assert inst is not None
    assert inst.binding_guard is GuardId.G6_TURNOVER
    assert abs(inst.approved_weight) < abs(inst.reference)
    assert card.orders == (), "a refused top-up sends nothing"
    assert [t.kind.value for t in card.triggers] == ["coordinated_cluster"]


def test_a_model_outage_has_its_card_and_the_flatten_has_its_own(site: Site) -> None:
    card = by_decision(site)[site.world.decision_ids[2]]
    assert card.decision is None
    assert card.outcome is LlmOutcome.TRANSPORT_ERROR
    assert card.kernel is None
    assert card.orders == ()
    assert [t.kind.value for t in card.triggers] == ["owner_manual"]
    protective = [c for c in cards_of(site) if c.decision_id is None]
    assert len(protective) == 1
    flatten = protective[0]
    assert flatten.kernel is not None
    assert flatten.kernel.protective_reason is ProtectiveReason.LLM_OUTAGE
    assert flatten.card_id == protective_card_id(flatten.kernel.ruling_id)
    assert flatten.triggers == ()
    assert flatten.coverage == {}
    assert flatten.shown_text == ()
    assert len(flatten.orders) == 1
    order = flatten.orders[0]
    assert order.symbol == NVDA
    assert order.side is Side.BUY
    assert order.state is OrderState.FILLED
    kinds = {event_at(site, s).kind for s in flatten.ledger_seqs}
    assert EventKind.PROTECTIVE_ACTION in kinds


def test_an_abstention_is_a_card_with_its_reasons(site: Site) -> None:
    card = by_decision(site)[site.world.decision_ids[3]]
    decision = card.decision
    assert decision is not None
    assert decision.stance is Stance.FLAT_WITH_REASONS
    assert decision.flat_reasons
    assert card.orders == ()
    assert card.previews == ()


def test_net_pnl_per_order_adds_up_to_the_closed_trades(site: Site) -> None:
    """The book ends flat, so every fill's realised P&L less every fee is the closed trades' net."""
    projection = site.world.projection()
    bundle = build_card_bundle(projection, site.world.blobs)
    realised = realized_pnl_by_fill(projection.fills)
    ours = sum((t.order.net_pnl or Decimal(0) for t in bundle.trails), Decimal(0))
    venue = sum(
        (realised[f.exec_id] - f.fee_paid for _, f in bundle.venue_originated_fills), Decimal(0)
    )
    trades = sum((t.net_pnl for t in projection.closed_trades), Decimal(0))
    assert projection.builder().positions() == {}
    assert ours + venue == trades
    assert [f.exec_id for _, f in bundle.venue_originated_fills] == site.world.stop_fill_ids


def test_order_trails_cover_every_planned_intent(site: Site) -> None:
    projection = site.world.projection()
    bundle = build_card_bundle(projection, site.world.blobs)
    planned = [i.client_oid for p in projection.plans for i in p.intents]
    assert [t.order.client_oid for t in bundle.trails] == planned
    card_ids = {c.card_id for c in bundle.cards}
    assert all(t.card_id in card_ids for t in bundle.trails)


# ------------------------------------------------------------------------------------------------
# Refusals: a card never asserts more than the ledger holds
# ------------------------------------------------------------------------------------------------


def projection_without(site: Site, kind: EventKind) -> Projection:
    projection = Projection(POLICY_V1)
    for event in site.world.ledger.events():
        if event.kind is not kind:
            projection.apply(event)
    return projection


def test_a_decision_whose_snapshot_is_missing_is_refused(site: Site) -> None:
    with pytest.raises(CardError, match="snapshot"):
        build_cards(projection_without(site, EventKind.SNAPSHOT), site.world.blobs)


def test_a_decision_whose_trigger_is_missing_is_refused(site: Site) -> None:
    with pytest.raises(CardError, match="trigger"):
        build_cards(projection_without(site, EventKind.TRIGGER), site.world.blobs)


class _HidingStore:
    """A blob store that has lost one blob."""

    def __init__(self, site: Site, missing: str) -> None:
        self._store = site.world.blobs
        self._missing = missing

    def put(self, data: bytes, media_type: str) -> BlobRef:
        return self._store.put(data, media_type)

    def get(self, sha256: str) -> bytes:
        if sha256 == self._missing:
            raise KeyError(sha256)
        return self._store.get(sha256)


class _AlteringStore(_HidingStore):
    def get(self, sha256: str) -> bytes:
        data = super().get(sha256)
        return data + b" " if sha256 == self._missing else data


@pytest.mark.parametrize("store", [_HidingStore, _AlteringStore])
def test_a_card_whose_proof_blob_is_missing_or_altered_is_refused(
    site: Site, store: type[_HidingStore]
) -> None:
    card = by_decision(site)[site.world.decision_ids[0]]
    target = card.blobs[0].sha256
    with pytest.raises(CardError, match=target):
        build_cards(site.world.projection(), store(site, target))


def test_card_ids_are_safe_file_names_whatever_the_ledger_says() -> None:
    assert decision_card_id("dec-0123abcd") == "dec-0123abcd"
    hostile = decision_card_id("../../etc/passwd")
    assert CARD_ID.fullmatch(hostile)
    assert hostile.startswith("dec-")
    assert protective_card_id("a" * 64) == "prot-" + "a" * 32
    odd = protective_card_id("not hex / at all")
    assert CARD_ID.fullmatch(odd)
    assert odd.startswith("prot-")


# ------------------------------------------------------------------------------------------------
# The per-fill fold, against hand-computed cases
# ------------------------------------------------------------------------------------------------


def fill(
    exec_id: str,
    side: Side,
    qty: str,
    price: str,
    minutes: int,
    *,
    trade_side: str | None = None,
    symbol: str = NVDA,
) -> Fill:
    q, p = Decimal(qty), Decimal(price)
    return Fill(
        exec_id=exec_id,
        venue_order_id=f"o-{exec_id}",
        client_oid=None,
        symbol=symbol,
        side=side,
        exec_price=p,
        exec_qty=q,
        exec_value=q * p,
        fee_paid=Decimal("0.01"),
        fee_coin="USDT",
        trade_scope="taker",
        trade_side="close" if trade_side == "close" else "open" if trade_side == "open" else None,
        exec_pnl=None,
        executed_at=T0 + timedelta(minutes=minutes),
        venue=FillVenue.SIMULATED,
    )


def test_realised_pnl_long_add_partial_close_and_flip() -> None:
    fills = [
        fill("a", Side.BUY, "1", "100", 0),
        fill("b", Side.BUY, "1", "110", 1),  # avg 105
        fill("c", Side.SELL, "0.5", "120", 2),  # +7.5
        fill("d", Side.SELL, "3", "90", 3),  # closes 1.5 at 90: -22.5, opens 1.5 short at 90
        fill("e", Side.BUY, "1.5", "80", 4),  # short closes: +15
    ]
    got = realized_pnl_by_fill(fills)
    assert got == {
        "a": Decimal(0),
        "b": Decimal(0),
        "c": Decimal("7.5"),
        "d": Decimal("-22.5"),
        "e": Decimal("15.0"),
    }


def test_realised_pnl_folds_in_venue_time_not_arrival_order() -> None:
    late_open = fill("open", Side.BUY, "2", "100", 0)
    early_close_logged_first = fill("close", Side.SELL, "2", "103", 5)
    got = realized_pnl_by_fill([early_close_logged_first, late_open])
    assert got == {"open": Decimal(0), "close": Decimal(6)}


def test_realised_pnl_closes_before_opens_at_the_same_instant_and_counts_a_fill_once() -> None:
    fills = [
        fill("buy", Side.BUY, "1", "50", 0),
        fill("reopen", Side.BUY, "1", "60", 1, trade_side="open"),
        fill("stop", Side.SELL, "1", "40", 1, trade_side="close"),
        fill("stop", Side.SELL, "1", "40", 1, trade_side="close"),
    ]
    got = realized_pnl_by_fill(fills)
    assert got == {"buy": Decimal(0), "stop": Decimal(-10), "reopen": Decimal(0)}


def test_symbols_are_folded_apart() -> None:
    fills = [
        fill("n1", Side.BUY, "1", "100", 0),
        fill("b1", Side.SELL, "1", "50", 0, symbol=BTC),
        fill("n2", Side.SELL, "1", "101", 1),
        fill("b2", Side.BUY, "1", "40", 1, symbol=BTC),
    ]
    got = realized_pnl_by_fill(fills)
    assert got["n2"] == Decimal(1)
    assert got["b2"] == Decimal(10)
