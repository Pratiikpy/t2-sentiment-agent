"""The crowd, as numbers only: distinct stories, coordination, and forum mention counts.

**No third-party text reaches the model in the replica.** The primary shows the model screened and
spotlighted crowd text; its screen (``sentiment_agent.crowd.quarantine``) rests on ``unicodedata``
normalisation, which the Playbook sandbox does not allow, and a weaker port would be a weaker
defence against the prompt injection a sentiment agent is most exposed to (win plan Move 15). So
the replica removes the attack surface instead of screening it: headlines are read here, clustered,
and counted, and only counts, ranks and identifiers this module generates are rendered into the
prompt or into a trigger. The loss is stated in the README: the replica's model sees *how much*
the crowd is talking and *whether it is coordinated*, never *what it says*.

Clustering is the primary's (``sentiment_agent.crowd.novelty``, from ARGUS ``agents/novelty.py``,
MIT, same author): exact Jaccard over word 4-shingles of folded text with links removed, the 0.55
duplicate threshold, single-link clusters, and a cluster *coordinated* when some window of
``TRIGGER_COORDINATED_WINDOW_MINUTES`` holds at least ``TRIGGER_COORDINATED_MIN_SOURCES`` distinct
sources and the text has at least ``MIN_COORDINATION_WORDS`` words. Every constant is generated
from the primary's modules into ``policy_v1``.
"""

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from . import policy_v1 as policy
from .mentions import fold, symbol_for_ticker, symbols_mentioned

_URL = r"(?:https?://|www\.)\S+"
_WORD = r"[a-z0-9]+"


@dataclass(frozen=True)
class NewsItem:
    source: str
    published: datetime | None
    title: str
    summary: str


@dataclass(frozen=True)
class Cluster:
    cluster_id: str
    size: int
    sources: int
    first_seen: datetime | None
    coordinated: bool
    symbols: tuple[str, ...]


@dataclass(frozen=True)
class CrowdReading:
    items: int
    stories: int
    clusters: tuple[Cluster, ...]
    news_mentions: Mapping[str, int]
    forum_mentions: Mapping[str, dict[str, float]] = field(default_factory=dict)

    def coordinated_for(self, symbol: str) -> list[Cluster]:
        return [c for c in self.clusters if c.coordinated and symbol in c.symbols]


def words(text: str) -> list[str]:
    return re.findall(_WORD, re.sub(_URL, " ", fold(text), flags=re.IGNORECASE).lower())


def shingles(text: str, size: int = policy.NOVELTY_SHINGLE_WORDS) -> frozenset[str]:
    tokens = words(text)
    if len(tokens) < size:
        return frozenset({" ".join(tokens)}) if tokens else frozenset()
    return frozenset(" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class _UnionFind:
    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, i: int) -> int:
        root = i
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[i] != root:
            self._parent[i], i = root, self._parent[i]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[max(ra, rb)] = min(ra, rb)


def cluster_indices(prints: Sequence[frozenset[str]], threshold: float) -> list[list[int]]:
    """Single-link clusters of items whose shingle sets reach ``threshold`` Jaccard similarity."""
    finder = _UnionFind(len(prints))
    index: dict[str, list[int]] = {}
    for i, shingle_set in enumerate(prints):
        for shingle in shingle_set:
            index.setdefault(shingle, []).append(i)
    for i, shingle_set in enumerate(prints):
        candidates = {j for s in shingle_set for j in index[s] if j > i}
        for j in sorted(candidates):
            if jaccard(shingle_set, prints[j]) >= threshold:
                finder.union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(len(prints)):
        groups.setdefault(finder.find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])


def coordinated(items: Sequence[NewsItem]) -> bool:
    """Some window of the policy's length holds the policy's number of distinct sources."""
    if len(words(items[0].title)) < policy.NOVELTY_MIN_COORDINATION_WORDS:
        return False
    window = timedelta(minutes=policy.TRIGGER_COORDINATED_WINDOW_MINUTES)
    timed = sorted((i.published, i.source) for i in items if i.published is not None)
    for start_index, (start, _) in enumerate(timed):
        sources = {source for at, source in timed[start_index:] if at - start <= window}
        if len(sources) >= policy.TRIGGER_COORDINATED_MIN_SOURCES:
            return True
    return False


def read_crowd(
    items: Sequence[NewsItem],
    *,
    universe: Sequence[str],
    now: datetime,
    forum_rows: Sequence[Mapping[str, object]] = (),
) -> CrowdReading:
    """Cluster ``items`` into stories and count, per universe symbol, the distinct stories that
    name it in the last 24 hours. ``forum_rows`` are ``sentiment.trending`` rows (ticker,
    mentions, mentions_24h_ago, rank); only rows whose ticker maps to a universe symbol count."""
    day_ago = now - timedelta(hours=24)
    recent = [i for i in items if i.published is None or i.published >= day_ago]
    prints = [shingles(i.title) for i in recent]
    groups = cluster_indices(prints, policy.NOVELTY_DUPLICATE_AT) if recent else []
    clusters: list[Cluster] = []
    mentions: dict[str, int] = dict.fromkeys(universe, 0)
    for group in groups:
        members = [recent[i] for i in group]
        first = min((m.published for m in members if m.published is not None), default=None)
        named: set[str] = set()
        for member in members:
            named.update(symbols_mentioned(f"{member.title} {member.summary}", list(universe)))
        for symbol in named:
            mentions[symbol] = mentions.get(symbol, 0) + 1
        representative = min(members, key=lambda m: (m.published is None, m.published, m.source))
        cluster_id = "story-" + "-".join(words(representative.title)[:6])[:48]
        clusters.append(
            Cluster(
                cluster_id=cluster_id,
                size=len(members),
                sources=len({m.source for m in members}),
                first_seen=first,
                coordinated=coordinated(members),
                symbols=tuple(s for s in universe if s in named),
            )
        )
    forum: dict[str, dict[str, float]] = {}
    for row in forum_rows:
        matched = symbol_for_ticker(row.get("ticker"), list(universe))
        if matched is None:
            continue
        entry = forum.setdefault(matched, {})
        count = _number(row.get("mentions"))
        before = _number(row.get("mentions_24h_ago"))
        rank = _number(row.get("rank"))
        if count is not None:
            entry["forum_mentions_24h"] = entry.get("forum_mentions_24h", 0.0) + count
        if before is not None:
            entry["forum_mentions_prior_24h"] = entry.get("forum_mentions_prior_24h", 0.0) + before
        if rank is not None:
            best = entry.get("forum_rank")
            entry["forum_rank"] = rank if best is None else min(best, rank)
    for entry in forum.values():
        now_count = entry.get("forum_mentions_24h")
        prior = entry.get("forum_mentions_prior_24h")
        if now_count is not None and prior is not None and prior > 0:
            entry["forum_mentions_change_pct"] = (now_count / prior - 1.0) * 100.0
    return CrowdReading(
        items=len(recent),
        stories=len(groups),
        clusters=tuple(clusters),
        news_mentions=mentions,
        forum_mentions=forum,
    )


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None
