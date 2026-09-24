"""Six posts, or one post six times? Distinct stories, coordination, and who is being talked about.

A crowd reading built on raw item counts measures republication volume. Twenty accounts pasting the
same "$NVDA to 200 by Friday" inside an hour are one story, and the fact that twenty accounts did it
inside an hour is itself the finding: the shape of a promoted narrative rather than of independent
opinion. This module turns screened text into a :class:`~sentiment_agent.types.CrowdReport`: the
items grouped into distinct stories, each story's sources, speed and whether it was coordinated,
and per-symbol mentions counted once per story.

Provenance
----------
Ported from ARGUS ``argus/src/argus/agents/novelty.py`` (MIT, Copyright (c) 2026 Pratiikpy, the same
author as this project) at commit ``3dec6baf9dfa7be37c7b452e26a9b139df252f75``, sha256
``d9302be5b12e04b01d5f4a8dca11c654d510905a6a843f8a1b1dbfd8a44d0ce3``. The ARGUS repository is
private, so its licence and every pin are recorded in ``third_party/argus/PROVENANCE.md``. Copied,
never imported. Carried over unchanged: exact Jaccard over word 4-shingles (not MinHash: a decision
cycle sees hundreds of items, where the exact computation is cheap and an approximation adds false
negatives for nothing), the 0.55 duplicate threshold, single-link clustering so rewrites chain, the
earliest item as a cluster's representative, and coordination as "at least 3 distinct sources inside
2 hours" (those two numbers now come from :class:`~sentiment_agent.types.TriggerRule`, where they
are frozen at genesis).

What this port changes, and why
-------------------------------
Each change answers a defect found by running the ARGUS module; ``tests/crowd/test_novelty.py``
reproduces each one.

* **Coordination is tested on a sliding window, not on the whole cluster's span.** ARGUS called a
  cluster coordinated only when *all* of it fitted inside 2 hours, so one early copy posted a few
  hours before a burst disarmed detection of the burst: five accounts inside 30 minutes, plus one
  post 5 hours earlier, read as "syndication". Now any 2-hour window holding 3 distinct sources is
  enough.
* **Clustering is true single-link.** ARGUS added an item to the first cluster it matched and
  stopped, so an item bridging two existing clusters left them split and the result depended on
  arrival order. Here every pair above the threshold is joined (union-find over an inverted shingle
  index, which is also what keeps it fast: only pairs that share a shingle can reach 0.55).
* **Velocity is monotone in speed.** ARGUS reported ``size`` copies per hour for copies with the
  same timestamp, so three identical posts in the same second read as 3/hour while three posts
  twelve minutes apart read as 15/hour. The span is now floored at one minute
  (:data:`VELOCITY_FLOOR`), the finest resolution the platforms' timestamps are worth.
* **Shingles are computed on folded text.** Invisible characters are removed and compatibility
  forms folded (NFKC) before tokenising, so a zero-width space inside a word (``pu`` U+200B
  ``mp``) or fullwidth letters cannot make copies look distinct; links are removed, because every
  X post carries its own ``t.co`` short link and those differ between otherwise identical copies.
* **Very short texts are never "coordinated".** A text shorter than one shingle (``"$BTC"``,
  ``"gm"``) is compared by its exact word string, so three accounts posting ``"$BTC"`` inside two
  hours formed a coordinated cluster in ARGUS. That is ordinary noise at X's volume and would wake
  the model on every cycle; coordination now needs at least :data:`MIN_COORDINATION_WORDS` words.
  Such clusters are still counted as one story.
* **Withheld items are not stories.** An item the quarantine withheld is counted in ``withheld`` but
  never clustered or counted as a mention: its text never reaches the model, and a redaction marker
  is not a story about anything.

Symbol attribution (:func:`symbols_mentioned`) is new; ARGUS keyed evidence by the symbol it was
fetched for. The alias table is in :data:`ALIASES`, with the reasoning for each ambiguous name.
"""

import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final

from sentiment_agent.hashing import content_hash
from sentiment_agent.types import CrowdReport, Policy, ScreenedItem, StoryCluster

SHINGLE: Final = 4
"""Words per shingle, the conventional choice for news-length text. Shorter shingles match common
phrasing and call unrelated posts duplicates; longer ones miss a rewrite that changed a few
words."""

DUPLICATE_AT: Final = 0.55
"""Jaccard similarity at or above which two items carry the same story.

ARGUS's calibration: a copy rewritten with a new lead still shares most of its 4-word shingles,
while two different stories about one company share the company name and little else. It leans
toward calling copies copies, because the opposite error lets a thin feed look like a broad one."""

MIN_COORDINATION_WORDS: Final = SHINGLE
"""A cluster whose text has fewer words than one shingle is never marked coordinated."""

VELOCITY_FLOOR: Final = timedelta(minutes=1)
"""The shortest span a velocity is computed over. Copies stamped within the same minute are treated
as arriving over one minute, so the fastest bursts report the highest rate rather than the
lowest."""

_URL: Final = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_WORD: Final = re.compile(r"[a-z0-9]+")


def _fold(text: str) -> str:
    """Invisible characters removed, compatibility forms folded, links removed, lower case."""
    visible = "".join(c for c in text if unicodedata.category(c) not in {"Cf", "Co", "Cs"})
    return _URL.sub(" ", unicodedata.normalize("NFKC", visible)).lower()


def _words(text: str) -> list[str]:
    return _WORD.findall(_fold(text))


def shingles(text: str, *, size: int = SHINGLE) -> frozenset[str]:
    """Overlapping word n-grams of the folded text.

    A text shorter than one shingle returns its whole word string as a single element, so a one-line
    post is still comparable with its own copy rather than silently unmatched against everything. A
    text with no words returns the empty set.
    """
    if size < 1:
        raise ValueError(f"a shingle of {size} words is meaningless")
    words = _words(text)
    if not words:
        return frozenset()
    if len(words) < size:
        return frozenset([" ".join(words)])
    return frozenset(" ".join(words[i : i + size]) for i in range(len(words) - size + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Exact Jaccard similarity. Two empty sets are uncomparable, not similar: 0.0."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# --- symbol attribution -------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SymbolAliases:
    """How crowd text refers to one instrument, and how to search for it.

    Every field lists only forms that denote the instrument with high precision. Ambiguous words are
    left out on purpose: a false mention inflates a crowd reading, while a missed one only costs a
    little recall on text that the collectors fetched by cashtag anyway.
    """

    symbol: str
    cashtags: tuple[str, ...]
    """``$TICKER`` forms, matched case-insensitively (X treats cashtags that way)."""
    tickers: tuple[str, ...] = ()
    """Bare tickers that are not English words, matched in any case (``nvda``, ``NVDA``)."""
    upper_tickers: tuple[str, ...] = ()
    """Bare tickers that are also words in lower case, matched only as written in capitals."""
    names: tuple[str, ...] = ()
    """Names matched in any case (``nvidia``, ``s&p 500``)."""
    proper_names: tuple[str, ...] = ()
    """Names that are ordinary words in lower case (``Apple``), matched only as capitalised."""
    x_query: str = ""
    """The X search query (``twitter search QUERY``)."""
    reddit_query: str = ""
    """The Reddit search query (``rdt search QUERY``)."""

    @property
    def base(self) -> str:
        return self.symbol.removesuffix("USDT")


ALIASES: Final[Mapping[str, SymbolAliases]] = {
    a.symbol: a
    for a in (
        SymbolAliases(
            symbol="BTCUSDT",
            cashtags=("BTC", "XBT"),
            tickers=("BTC", "XBT"),
            names=("bitcoin",),
            x_query="$BTC",
            reddit_query="bitcoin OR BTC",
        ),
        SymbolAliases(
            symbol="SP500USDT",
            cashtags=("SPX", "SPY"),
            tickers=("SP500", "SPX"),
            # "spy" is an English word; the ETF is written SPY.
            upper_tickers=("SPY",),
            # Bare "S&P" is left out: it names S&P Global's ratings business at least as often as
            # the index ("S&P downgrades ...").
            names=("s&p 500", "s&p500", "s & p 500"),
            x_query="$SPX OR $SPY",
            reddit_query='"S&P 500" OR SPX OR SPY',
        ),
        SymbolAliases(
            symbol="NDX100USDT",
            cashtags=("NDX", "QQQ"),
            tickers=("NDX", "NDX100", "QQQ"),
            # Bare "Nasdaq" is left out: it names the exchange ("Nasdaq halts trading") and the
            # Composite as often as the Nasdaq-100.
            names=("nasdaq 100", "nasdaq-100", "nasdaq100"),
            x_query="$NDX OR $QQQ",
            reddit_query='"Nasdaq 100" OR NDX OR QQQ',
        ),
        SymbolAliases(
            symbol="MSTRUSDT",
            cashtags=("MSTR",),
            tickers=("MSTR",),
            # The company renamed itself "Strategy" in 2025; that word is far too common to count.
            names=("microstrategy",),
            x_query="$MSTR",
            reddit_query="MSTR OR MicroStrategy",
        ),
        SymbolAliases(
            symbol="HOODUSDT",
            cashtags=("HOOD",),
            # Bare HOOD is left out: all-caps retail prose uses the word ("BACK IN THE HOOD").
            names=("robinhood",),
            x_query="$HOOD",
            reddit_query="Robinhood OR HOOD",
        ),
        SymbolAliases(
            symbol="CRCLUSDT",
            cashtags=("CRCL",),
            tickers=("CRCL",),
            # "Circle" alone is an ordinary word ("circle back"); only the full company name counts.
            names=("circle internet",),
            x_query="$CRCL",
            reddit_query='CRCL OR "Circle Internet"',
        ),
        SymbolAliases(
            symbol="SNDKUSDT",
            cashtags=("SNDK",),
            tickers=("SNDK",),
            names=("sandisk",),
            x_query="$SNDK",
            reddit_query="SNDK OR Sandisk",
        ),
        SymbolAliases(
            symbol="COINUSDT",
            cashtags=("COIN",),
            # Bare COIN is left out: "THIS COIN IS GOING TO THE MOON" is about some other coin.
            names=("coinbase",),
            x_query="$COIN",
            reddit_query="Coinbase",
        ),
        SymbolAliases(
            symbol="TSLAUSDT",
            cashtags=("TSLA",),
            tickers=("TSLA",),
            names=("tesla",),
            x_query="$TSLA",
            reddit_query="TSLA OR Tesla",
        ),
        SymbolAliases(
            symbol="GOOGLUSDT",
            cashtags=("GOOGL", "GOOG"),
            tickers=("GOOGL", "GOOG"),
            # "alphabet" and "google" are ordinary words in lower case ("just google it").
            proper_names=("Alphabet", "Google"),
            x_query="$GOOGL OR $GOOG",
            reddit_query="GOOGL OR Alphabet",
        ),
        SymbolAliases(
            symbol="METAUSDT",
            cashtags=("META",),
            # Bare META is left out ("THIS IS SO META"); capitalised "Meta" is the company, except
            # as the prefix of a hyphenated word ("Meta-analysis"), which the matcher excludes.
            names=("meta platforms",),
            proper_names=("Meta",),
            x_query="$META",
            reddit_query='"Meta Platforms" OR "META stock"',
        ),
        SymbolAliases(
            symbol="NVDAUSDT",
            cashtags=("NVDA",),
            tickers=("NVDA",),
            names=("nvidia",),
            x_query="$NVDA",
            reddit_query="NVDA OR Nvidia",
        ),
        SymbolAliases(
            symbol="AMZNUSDT",
            cashtags=("AMZN",),
            tickers=("AMZN",),
            proper_names=("Amazon",),
            x_query="$AMZN",
            reddit_query="AMZN OR Amazon",
        ),
        SymbolAliases(
            symbol="AAPLUSDT",
            cashtags=("AAPL",),
            tickers=("AAPL",),
            # "apple" in lower case is the fruit.
            proper_names=("Apple",),
            x_query="$AAPL",
            reddit_query="AAPL OR Apple",
        ),
    )
}
"""Aliases for the 14 universe instruments (``policy.UNIVERSE``). A symbol outside this table is
matched by its cashtag and its capitalised base only (:func:`aliases_for`)."""


def aliases_for(symbol: str) -> SymbolAliases:
    """The alias record for ``symbol``, or a conservative default built from its base."""
    known = ALIASES.get(symbol)
    if known is not None:
        return known
    base = symbol.removesuffix("USDT")
    return SymbolAliases(
        symbol=symbol,
        cashtags=(base,),
        upper_tickers=(base,),
        x_query=f"${base}",
        reddit_query=base,
    )


_CASHTAG: Final = re.compile(r"(?<![A-Za-z0-9_$])\$([A-Za-z][A-Za-z0-9_]{0,9})(?![A-Za-z0-9_])")
_EDGE_BEFORE: Final = r"(?<![A-Za-z0-9$])"
_EDGE_AFTER: Final = r"(?![A-Za-z0-9])"


def _phrase(text: str) -> str:
    """A regex for a name, with any run of spaces in it matching any run of whitespace."""
    return r"\s+".join(re.escape(part) for part in text.split())


@dataclass(frozen=True, slots=True)
class _Matcher:
    symbol: str
    cashtags: frozenset[str]
    anycase: re.Pattern[str]
    exactcase: re.Pattern[str] | None


def _matcher(aliases: SymbolAliases) -> _Matcher:
    base = re.escape(aliases.base)
    anycase_forms = [
        # The perpetual itself: NVDAUSDT, NVDA/USDT, NVDA-USDT, NVDA_USDT, NVDAUSDT.P.
        rf"{base}[-/_]?USDT(?:\.P)?",
        *(re.escape(t) for t in aliases.tickers),
        *(_phrase(n) for n in aliases.names),
    ]
    exact_forms = [
        *(re.escape(t) for t in aliases.upper_tickers),
        # A capitalised name followed by a hyphen is a compound word ("Meta-analysis").
        *(_phrase(n) + r"(?!-\w)" for n in aliases.proper_names),
    ]
    anycase = re.compile(
        _EDGE_BEFORE + "(?:" + "|".join(anycase_forms) + ")" + _EDGE_AFTER, re.IGNORECASE
    )
    exactcase = (
        re.compile(_EDGE_BEFORE + "(?:" + "|".join(exact_forms) + ")" + _EDGE_AFTER)
        if exact_forms
        else None
    )
    return _Matcher(
        symbol=aliases.symbol,
        cashtags=frozenset(c.upper() for c in aliases.cashtags),
        anycase=anycase,
        exactcase=exactcase,
    )


_MATCHERS: dict[str, _Matcher] = {}


def _matcher_for(symbol: str) -> _Matcher:
    found = _MATCHERS.get(symbol)
    if found is None:
        found = _matcher(aliases_for(symbol))
        _MATCHERS[symbol] = found
    return found


def symbols_mentioned(text: str, universe: Sequence[str]) -> tuple[str, ...]:
    """The universe symbols ``text`` names, in universe order.

    A symbol is named by its cashtag (any case), by the perpetual's own spelling (``NVDAUSDT``,
    ``NVDA/USDT``), by an unambiguous bare ticker, or by a company or index name (see
    :data:`ALIASES` for which forms count and why some common ones do not). Links are ignored and
    the text is NFKC-folded with invisible characters removed first, so a fullwidth ``$NVDA`` and a
    zero-width space inside a ticker do not hide a mention.
    """
    visible = "".join(c for c in text if unicodedata.category(c) not in {"Cf", "Co", "Cs"})
    folded = _URL.sub(" ", unicodedata.normalize("NFKC", visible))
    cashtags = {m.group(1).upper() for m in _CASHTAG.finditer(folded)}
    named: list[str] = []
    for symbol in universe:
        if symbol in named:
            continue
        matcher = _matcher_for(symbol)
        if (
            cashtags & matcher.cashtags
            or matcher.anycase.search(folded)
            or (matcher.exactcase is not None and matcher.exactcase.search(folded))
        ):
            named.append(symbol)
    return tuple(named)


# --- clustering ---------------------------------------------------------------------------------


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
            # The earlier-indexed root wins, so a cluster's root is its earliest member.
            self._parent[max(ra, rb)] = min(ra, rb)


def _clusters(prints: Sequence[frozenset[str]], threshold: float) -> list[list[int]]:
    """Single-link groups of indices whose shingle sets reach ``threshold``, in index order.

    Exact, not approximate: two sets can only reach the threshold if they share a shingle, so the
    inverted index finds every pair that could link. An exact copy of a set already seen joins that
    item's group and is not indexed again (anything similar to it is similar to the first copy), so
    a flood of identical posts costs linear time. Measured on the build machine: a 560-item report
    (14 symbols, two channels, 20 each) in about 0.1 s, 2,000 identical copies in 0.2 s, and 2,000
    near-identical copies, the worst case, in about 1.2 s.
    """
    uf = _UnionFind(len(prints))
    index: dict[str, list[int]] = {}
    first_copy: dict[frozenset[str], int] = {}
    for i, shingle_set in enumerate(prints):
        if shingle_set:
            original = first_copy.setdefault(shingle_set, i)
            if original != i:
                uf.union(original, i)
                continue
        candidates: set[int] = set()
        for shingle in shingle_set:
            candidates.update(index.get(shingle, ()))
        for j in candidates:
            if uf.find(i) != uf.find(j) and jaccard(shingle_set, prints[j]) >= threshold:
                uf.union(i, j)
        for shingle in shingle_set:
            index.setdefault(shingle, []).append(i)
    groups: dict[int, list[int]] = {}
    for i in range(len(prints)):
        groups.setdefault(uf.find(i), []).append(i)
    return [groups[root] for root in sorted(groups)]


def _coordinated(
    stamps_and_sources: Sequence[tuple[datetime, str]], *, window: timedelta, min_sources: int
) -> bool:
    """Some window of length ``window`` (inclusive) holds ``min_sources`` distinct sources."""
    ordered = sorted(stamps_and_sources)
    in_window: Counter[str] = Counter()
    left = 0
    for stamp, source in ordered:
        in_window[source] += 1
        while stamp - ordered[left][0] > window:
            gone = ordered[left][1]
            in_window[gone] -= 1
            if in_window[gone] == 0:
                del in_window[gone]
            left += 1
        if len(in_window) >= min_sources:
            return True
    return False


def _velocity(size: int, span: timedelta) -> float | None:
    if size < 2:
        return None
    hours = max(span, VELOCITY_FLOOR).total_seconds() / 3600
    return size / hours


def build_report(
    screened: Sequence[ScreenedItem], *, universe: Sequence[str], policy: Policy
) -> CrowdReport:
    """Group screened items into distinct stories and count who each story is about.

    ``items`` and ``withheld`` count distinct item ids (the same post fetched twice is one item). An
    item id is withheld if any of its screenings was. Only items that were not withheld are
    clustered and counted as mentions. A story's representative is its earliest item's
    ``prompt_text``, the spotlighted form, because that is the only form of third-party text the
    model may ever see. ``duplication_ratio`` is clustered items per distinct story (1.0 means no
    copies), and 0.0 when there is nothing to cluster. ``mentions`` has an entry for every universe
    symbol, zero included, so "measured zero" and "not in the universe" stay different facts.
    """
    window = timedelta(minutes=policy.triggers.coordinated_window_minutes)
    min_sources = policy.triggers.coordinated_min_sources
    universe_order = tuple(dict.fromkeys(universe))
    universe_set = frozenset(universe_order)

    first: dict[str, ScreenedItem] = {}
    withheld_ids: set[str] = set()
    for entry in screened:
        item_id = entry.item.item_id
        first.setdefault(item_id, entry)
        if entry.withheld:
            withheld_ids.add(item_id)

    kept = sorted(
        (entry for item_id, entry in first.items() if item_id not in withheld_ids),
        key=lambda e: (e.item.published_at, e.item.item_id),
    )
    prints = [shingles(e.item.text) for e in kept]
    member_symbols = [
        frozenset(s for s in e.item.symbols if s in universe_set)
        | frozenset(symbols_mentioned(e.item.text, universe_order))
        for e in kept
    ]

    clusters: list[StoryCluster] = []
    for group in _clusters(prints, DUPLICATE_AT):
        members = [kept[i] for i in group]
        ids = tuple(m.item.item_id for m in members)
        stamps = [m.item.published_at for m in members]
        first_seen, last_seen = min(stamps), max(stamps)
        named: set[str] = set()
        for i in group:
            named |= member_symbols[i]
        long_enough = len(_words(members[0].item.text)) >= MIN_COORDINATION_WORDS
        clusters.append(
            StoryCluster(
                cluster_id="story-" + content_hash(sorted(ids))[:16],
                representative=members[0].prompt_text,
                item_ids=ids,
                sources=tuple(sorted({m.item.source for m in members})),
                symbols=tuple(s for s in universe_order if s in named),
                first_seen=first_seen,
                last_seen=last_seen,
                distinct_sources=len({m.item.source for m in members}),
                coordinated=long_enough
                and _coordinated(
                    [(m.item.published_at, m.item.source) for m in members],
                    window=window,
                    min_sources=min_sources,
                ),
                velocity_per_hour=_velocity(len(members), last_seen - first_seen),
            )
        )
    clusters.sort(key=lambda c: (-len(c.item_ids), c.first_seen, c.cluster_id))

    mentions = dict.fromkeys(universe_order, 0)
    for cluster in clusters:
        for symbol in cluster.symbols:
            mentions[symbol] += 1

    return CrowdReport(
        items=len(first),
        withheld=len(withheld_ids),
        distinct_stories=len(clusters),
        duplication_ratio=len(kept) / len(clusters) if clusters else 0.0,
        clusters=tuple(clusters),
        mentions=mentions,
    )


__all__ = [
    "ALIASES",
    "DUPLICATE_AT",
    "MIN_COORDINATION_WORDS",
    "SHINGLE",
    "VELOCITY_FLOOR",
    "SymbolAliases",
    "aliases_for",
    "build_report",
    "jaccard",
    "shingles",
    "symbols_mentioned",
]
