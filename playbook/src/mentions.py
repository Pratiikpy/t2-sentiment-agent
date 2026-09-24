"""Which universe instruments a piece of text names.

A port of the primary's ``sentiment_agent.crowd.novelty.symbols_mentioned`` (itself new in the
primary; the clustering it sits beside is from ARGUS ``agents/novelty.py``, MIT, same author). The
alias table is not copied by hand: ``policy_v1.ALIASES`` is generated from the primary's
``ALIASES`` by ``scripts/build_playbook.py``, so the two cannot drift.

The sandbox has no ``unicodedata``. The primary removes every invisible character (categories Cf,
Co, Cs) and folds compatibility forms (NFKC) before matching. Here the invisible characters are
removed by explicit code-point ranges covering those categories as they occur in practice (format
characters, bidi controls, the private-use area, tag characters, surrogates), and the one
compatibility fold that matters for tickers, fullwidth ASCII (U+FF01-U+FF5E), is applied directly.
Other compatibility forms (ligatures, circled letters) are not folded; a mention spelled that way
is missed, which costs recall on a count, never a trade.
"""

import re

from . import policy_v1 as policy

_URL = r"(?:https?://|www\.)\S+"
_CASHTAG = r"(?<![A-Za-z0-9_$])\$([A-Za-z][A-Za-z0-9_]{0,9})(?![A-Za-z0-9_])"
_EDGE_BEFORE = r"(?<![A-Za-z0-9$])"
_EDGE_AFTER = r"(?![A-Za-z0-9])"

_INVISIBLE_RANGES: tuple[tuple[int, int], ...] = (
    (0x00AD, 0x00AD),
    (0x0600, 0x0605),
    (0x061C, 0x061C),
    (0x06DD, 0x06DD),
    (0x070F, 0x070F),
    (0x180E, 0x180E),
    (0x200B, 0x200F),
    (0x202A, 0x202E),
    (0x2060, 0x2064),
    (0x2066, 0x206F),
    (0xD800, 0xDFFF),
    (0xE000, 0xF8FF),
    (0xFEFF, 0xFEFF),
    (0xFFF9, 0xFFFB),
    (0x110BD, 0x110BD),
    (0x1D173, 0x1D17A),
    (0xE0001, 0xE007F),
    (0xF0000, 0x10FFFF),
)


def _invisible(code: int) -> bool:
    return any(lo <= code <= hi for lo, hi in _INVISIBLE_RANGES)


def fold(text: str) -> str:
    """Invisible characters removed, fullwidth ASCII folded, the Unicode minus read as ``-``."""
    out: list[str] = []
    for char in text:
        code = ord(char)
        if _invisible(code):
            continue
        if 0xFF01 <= code <= 0xFF5E:
            out.append(chr(code - 0xFEE0))
        elif code == 0x3000:
            out.append(" ")
        elif code == 0x2212:
            out.append("-")
        else:
            out.append(char)
    return "".join(out)


def _phrase(text: str) -> str:
    return r"\s+".join(re.escape(part) for part in text.split())


_PATTERNS: dict[str, tuple[frozenset[str], str, str | None]] = {}


def _aliases(symbol: str) -> dict[str, tuple[str, ...]]:
    known = policy.ALIASES.get(symbol)
    if known is not None:
        return known
    base = symbol[: -len("USDT")] if symbol.endswith("USDT") else symbol
    return {
        "cashtags": (base,),
        "tickers": (),
        "upper_tickers": (base,),
        "names": (),
        "proper_names": (),
    }


def _patterns(symbol: str) -> tuple[frozenset[str], str, str | None]:
    found = _PATTERNS.get(symbol)
    if found is not None:
        return found
    aliases = _aliases(symbol)
    base_text = symbol[: -len("USDT")] if symbol.endswith("USDT") else symbol
    base = re.escape(base_text)
    anycase_forms = [
        rf"{base}[-/_]?USDT(?:\.P)?",
        *(re.escape(t) for t in aliases["tickers"]),
        *(_phrase(n) for n in aliases["names"]),
    ]
    exact_forms = [
        *(re.escape(t) for t in aliases["upper_tickers"]),
        *(_phrase(n) + r"(?!-\w)" for n in aliases["proper_names"]),
    ]
    anycase = _EDGE_BEFORE + "(?:" + "|".join(anycase_forms) + ")" + _EDGE_AFTER
    exactcase = (
        _EDGE_BEFORE + "(?:" + "|".join(exact_forms) + ")" + _EDGE_AFTER if exact_forms else None
    )
    built = (frozenset(c.upper() for c in aliases["cashtags"]), anycase, exactcase)
    _PATTERNS[symbol] = built
    return built


def symbols_mentioned(text: str, universe: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """The universe symbols ``text`` names, in universe order (``novelty.symbols_mentioned``)."""
    folded = re.sub(_URL, " ", fold(text), flags=re.IGNORECASE)
    cashtags = {m.group(1).upper() for m in re.finditer(_CASHTAG, folded)}
    named: list[str] = []
    for symbol in universe:
        if symbol in named:
            continue
        tags, anycase, exactcase = _patterns(symbol)
        if (
            cashtags & tags
            or re.search(anycase, folded, flags=re.IGNORECASE)
            or (exactcase is not None and re.search(exactcase, folded))
        ):
            named.append(symbol)
    return tuple(named)


def symbol_for_ticker(ticker: object, universe: tuple[str, ...] | list[str]) -> str | None:
    """The universe symbol a bare ticker (``NVDA``, ``$SPY``, ``BTC``) denotes, by its cashtags."""
    if not isinstance(ticker, str):
        return None
    name = fold(ticker).strip().lstrip("$").upper()
    if not name:
        return None
    for symbol in universe:
        tags, _, _ = _patterns(symbol)
        if name in tags or name == symbol:
            return symbol
    return None
