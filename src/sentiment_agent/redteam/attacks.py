# The rephrasings in coordinated_pump are ported from ARGUS
# ``src/argus/eval/sentiment_comparison.py`` ``coordinated_scenario`` (lines 157-178; MIT, Copyright
# (c) 2026 Pratiikpy, the same author as this project) at commit
# 3dec6baf9dfa7be37c7b452e26a9b139df252f75, sha256
# 7d04819db39e901eb472c2035555c174ceedddc595f6f7d7445586fb6019027b. The ARGUS repository is
# private, so its licence and every pin are recorded in third_party/argus/PROVENANCE.md. Copied,
# never imported. Taken: the five rephrasings of one claim (as written; with a
# period replaced by ", multiple sources confirm."; prefixed "BREAKING: "; suffixed " This is huge
# if true."; upper-cased), cycled across accounts. Changed: ARGUS posted two minutes apart from a
# start time; here the posts are spread evenly across a window that ends at the decision time,
# because an attacker who wants to stay under a coordination rule spreads out, and the rule under
# test counts distinct sources inside a window; and each post comes from its own account, which is
# what the rule counts.
"""Attacks on the crowd channel of a recorded snapshot.

:func:`inject` takes a logged :class:`~sentiment_agent.types.PerceptionSnapshot` and a
:class:`~sentiment_agent.types.RedTeamVector`, applies the vector's operation to the snapshot's
crowd text (:func:`sentiment_agent.redteam.corpus.mechanism` says which), and hands the attacked
text to :func:`sentiment_agent.perception.snapshot.rebuild_text`. That is the same quarantine
(``crowd/quarantine.screen``) and the same story clustering (``crowd/novelty.build_report``) a live
snapshot's text goes through, followed by the social features, the crowd facts and a fresh seal.
So the defences are exercised, never bypassed: an injection the quarantine catches arrives
withheld, and a pump the coordination rule catches arrives as one coordinated story
(DESIGN.md §14.5).

Injected posts are X posts dated at the snapshot's own time, the moment upstream's vectors date
theirs (``ctx.asOf``); their ids are derived from their content, so the same post injected twice is
the same item, and nothing in an id tells the model it is looking at an attack. Operations that edit
text already in the snapshot act on every item it holds, as HeyArka's map over ``ctx.news`` does;
where there is nothing to edit, the attacked snapshot equals the clean one and the harness records a
no-op, never a resisted attack.

:func:`clean_reference` is the other half of the pair: the recorded snapshot's own text sent
through the same rebuild, so a clean decision and an attacked decision differ in the attack and
nothing else.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Final, Literal

from sentiment_agent.hashing import content_hash
from sentiment_agent.perception.snapshot import rebuild_text
from sentiment_agent.redteam.corpus import (
    PRICE_PLACEHOLDER,
    PUMP_CLAIMS,
    Mechanism,
    Operation,
    cashtag_of,
    mechanism,
)
from sentiment_agent.types import PerceptionSnapshot, Policy, RedTeamVector, TextItem

PUMP_HANDLES: Final[tuple[str, ...]] = (
    "tapereader",
    "flowdesk",
    "chartroom",
    "deskalerts",
    "marketbits",
    "tickerwire",
    "orderflowhq",
    "macrodesk",
)
"""Account stems for pump posts: ordinary trading-account names, none of which says what it is."""

PRICE_MULTIPLIER: Final = Decimal("1.4")
"""HeyArka's plausible falsehood quotes the market price times 1.4 (``semantic-trap.ts:76``)."""

_CENT: Final = Decimal("0.01")


class VectorNotApplicable(ValueError):  # noqa: N818 - reads as the fact the harness records
    """The vector cannot be applied to this snapshot (e.g. no price to misquote)."""


@dataclass(frozen=True, slots=True)
class Injection:
    """An attacked snapshot and what the attack did to its crowd text, by item id."""

    snapshot: PerceptionSnapshot
    vector: RedTeamVector
    added: tuple[str, ...]
    modified: tuple[str, ...]
    """Items already in the snapshot whose text or date the attack changed."""
    removed: tuple[str, ...]

    @property
    def touched(self) -> tuple[str, ...]:
        """Every item that carries the attack: added or modified."""
        return (*self.added, *self.modified)


# ------------------------------------------------------------------------------------------------
# The coordinated pump
# ------------------------------------------------------------------------------------------------


def _rephrasings(claim: str) -> tuple[str, ...]:
    """ARGUS ``coordinated_scenario``'s five near-identical forms of one claim."""
    return (
        claim,
        claim.replace(".", ", multiple sources confirm."),
        f"BREAKING: {claim}",
        f"{claim} This is huge if true.",
        claim.upper() if not claim.isupper() else claim + " (repost)",
    )


def _x_id(seed: object) -> str:
    """An X-style numeric id derived from ``seed``: deterministic, and saying nothing about it."""
    return f"x:{int(content_hash(seed)[:15], 16)}"


def pump_handle(index: int) -> str:
    """The account posting the ``index``-th copy of a pump (``@tapereader_1``, ...)."""
    stem = PUMP_HANDLES[index % len(PUMP_HANDLES)]
    return f"@{stem}_{index // len(PUMP_HANDLES) + 1}"


def coordinated_pump(
    symbol: str,
    *,
    direction: Literal["long", "short"],
    n_accounts: int,
    window: timedelta,
    at: datetime,
) -> list[TextItem]:
    """``n_accounts`` X posts pushing ``symbol`` in ``direction``, each from its own account,
    spread evenly from ``at - window`` to ``at`` (one post lands exactly on each end; a single post
    lands on ``at``), with near-identical text: one claim in ARGUS's five rephrasings, cycled.

    Every post names the instrument by cashtag and carries nothing a quarantine rule looks for, so
    the only defence in its way is the coordination rule, which the policy states as "at least
    ``coordinated_min_sources`` distinct sources inside ``coordinated_window_minutes``".
    """
    if direction not in PUMP_CLAIMS:
        raise ValueError(f"direction must be 'long' or 'short', not {direction!r}")
    if isinstance(n_accounts, bool) or not isinstance(n_accounts, int) or n_accounts < 1:
        raise ValueError("a pump needs at least one account")
    if window < timedelta(0):
        raise ValueError("the pump window cannot be negative")
    if at.tzinfo is None or at.utcoffset() != timedelta(0):
        raise ValueError("the pump is dated in timezone-aware UTC")
    claim = PUMP_CLAIMS[direction].replace("{cashtag}", cashtag_of(symbol))
    variants = _rephrasings(claim)
    step = window / (n_accounts - 1) if n_accounts > 1 else timedelta(0)
    start = at - window if n_accounts > 1 else at
    posts: list[TextItem] = []
    for i in range(n_accounts):
        published = start + step * i
        source = pump_handle(i)
        text = variants[i % len(variants)]
        posts.append(
            TextItem(
                item_id=_x_id({"pump": symbol, "source": source, "at": published, "text": text}),
                channel="x",
                source=source,
                url=None,
                published_at=published,
                fetched_at=at,
                text=text,
                symbols=(),
            )
        )
    return posts


# ------------------------------------------------------------------------------------------------
# Operations
# ------------------------------------------------------------------------------------------------


def items_of(snapshot: PerceptionSnapshot) -> tuple[TextItem, ...]:
    """The crowd items a snapshot holds, withheld ones included (they are still items)."""
    return tuple(s.item for s in snapshot.text)


def _price_x1_4(snapshot: PerceptionSnapshot, symbol: str) -> str:
    quote = snapshot.live_quotes.get(symbol) or snapshot.demo_quotes.get(symbol)
    if quote is None or quote.last <= 0:
        raise VectorNotApplicable(f"the snapshot has no price for {symbol} to misquote")
    return str((quote.last * PRICE_MULTIPLIER).quantize(_CENT, rounding=ROUND_HALF_UP))


def _post(snapshot: PerceptionSnapshot, source: str, text: str, n: int) -> TextItem:
    at = snapshot.taken_at
    return TextItem(
        item_id=_x_id({"at": at, "source": source, "text": text, "n": n}),
        channel="x",
        source=source,
        url=None,
        published_at=at,
        fetched_at=at,
        text=text,
        symbols=(),
    )


def shift_years(at: datetime, years: int) -> datetime:
    """``at`` moved by whole years. 29 February in a non-leap year becomes 1 March, which is what
    JavaScript's ``setFullYear`` (the operation HeyArka uses) does."""
    try:
        return at.replace(year=at.year + years)
    except ValueError:
        return at.replace(year=at.year + years, month=3, day=1)


def _rewrite_ticker(text: str, how: Mechanism) -> str:
    table = dict(how.confusables)
    pattern = re.compile(rf"\b{re.escape(how.ticker)}\b", re.ASCII)
    return pattern.sub(lambda m: "".join(table.get(ch, ch) for ch in m.group(0)), text)


def _attacked_items(
    snapshot: PerceptionSnapshot, vector: RedTeamVector, how: Mechanism, policy: Policy
) -> list[TextItem]:
    before = list(items_of(snapshot))
    op = how.operation
    if op in (Operation.APPEND, Operation.REPLACE_ALL):
        posts: list[TextItem] = []
        for n, post in enumerate(how.posts):
            text = post.text
            if PRICE_PLACEHOLDER in text:
                text = text.replace(PRICE_PLACEHOLDER, _price_x1_4(snapshot, vector.target_symbol))
            posts.append(_post(snapshot, post.source, text, n))
        return posts if op is Operation.REPLACE_ALL else [*before, *posts]
    if op is Operation.CLEAR:
        return []
    if op is Operation.SUFFIX:
        return [i.model_copy(update={"text": i.text + how.suffix}) for i in before]
    if op is Operation.REWRITE_TICKER:
        return [i.model_copy(update={"text": _rewrite_ticker(i.text, how)}) for i in before]
    if op is Operation.RESTAMP:
        return [
            i.model_copy(update={"published_at": shift_years(i.published_at, how.years)})
            for i in before
        ]
    if op is Operation.REPLAY_OLDEST:
        if not before:
            return before
        _, oldest = min(enumerate(before), key=lambda pair: (pair[1].published_at, pair[0]))
        text = how.prefix + oldest.text
        replay = oldest.model_copy(
            update={
                "item_id": _x_id({"replay": oldest.item_id, "at": snapshot.taken_at, "text": text}),
                "text": text,
                "published_at": snapshot.taken_at,
                "fetched_at": snapshot.taken_at,
            }
        )
        return [*before, replay]
    if op is Operation.PUMP:
        if how.pump_direction is None:
            raise ValueError(f"{vector.vector_id}: a pump without a direction")
        pump = coordinated_pump(
            vector.target_symbol,
            direction=how.pump_direction,
            n_accounts=how.pump_accounts,
            window=timedelta(minutes=policy.triggers.coordinated_window_minutes),
            at=snapshot.taken_at,
        )
        return [*before, *pump]
    raise ValueError(f"unhandled operation {op}")  # pragma: no cover - Operation is exhaustive


def _check(snapshot: PerceptionSnapshot, vector: RedTeamVector, policy: Policy) -> None:
    if snapshot.policy_version != policy.version:
        raise ValueError(
            f"the snapshot was taken under {snapshot.policy_version!r}, not {policy.version!r}"
        )
    if vector.target_symbol not in policy.symbols or vector.target_symbol not in snapshot.universe:
        raise ValueError(f"{vector.vector_id} targets {vector.target_symbol}, outside the universe")


def _diff(
    before: Sequence[TextItem], after: Sequence[TextItem]
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    old = {i.item_id: i for i in before}
    new = {i.item_id: i for i in after}
    added = tuple(i for i in new if i not in old)
    removed = tuple(i for i in old if i not in new)
    modified = tuple(i for i in new if i in old and new[i] != old[i])
    return added, modified, removed


def apply_vector(
    snapshot: PerceptionSnapshot, vector: RedTeamVector, *, policy: Policy
) -> Injection:
    """:func:`inject`, keeping the record of which items the attack added, changed or removed (the
    harness reads it to say which defence met the attack)."""
    _check(snapshot, vector, policy)
    how = mechanism(vector)
    before = items_of(snapshot)
    after = _attacked_items(snapshot, vector, how, policy)
    added, modified, removed = _diff(before, after)
    attacked = rebuild_text(snapshot, after, policy=policy)
    return Injection(
        snapshot=attacked, vector=vector, added=added, modified=modified, removed=removed
    )


def inject(
    snapshot: PerceptionSnapshot, vector: RedTeamVector, *, policy: Policy
) -> PerceptionSnapshot:
    """``snapshot`` with ``vector`` applied to its crowd text, screened, clustered and resealed.

    Pure: the input is not changed, and the same arguments give the same snapshot (same id).
    Raises :class:`VectorNotApplicable` when the vector needs something the snapshot lacks, and
    ``ValueError`` for a target outside the universe or a snapshot taken under another policy.
    """
    return apply_vector(snapshot, vector, policy=policy).snapshot


def clean_reference(snapshot: PerceptionSnapshot, *, policy: Policy) -> PerceptionSnapshot:
    """The recorded snapshot's own crowd text through the same rebuild :func:`inject` uses: the
    clean half of every pair. Equal to the recorded snapshot whenever that snapshot's text was
    measured and screened by this code; different (and the difference is the rebuild's, the same on
    both halves) when it was a light snapshot whose text was never read."""
    if snapshot.policy_version != policy.version:
        raise ValueError(
            f"the snapshot was taken under {snapshot.policy_version!r}, not {policy.version!r}"
        )
    return rebuild_text(snapshot, items_of(snapshot), policy=policy)


__all__ = [
    "PRICE_MULTIPLIER",
    "PUMP_HANDLES",
    "Injection",
    "VectorNotApplicable",
    "apply_vector",
    "clean_reference",
    "coordinated_pump",
    "inject",
    "items_of",
    "pump_handle",
    "shift_years",
]
