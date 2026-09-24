"""Numeric grounding: every figure in a thesis must resolve to a fact the model was shown.

Provenance
----------
Ported from ARGUS ``argus/src/argus/agents/grounding.py`` (MIT, Copyright (c) 2026 Pratiikpy, the
same author as this project) at commit ``3dec6baf9dfa7be37c7b452e26a9b139df252f75``, sha256
``533831eed522539bb7d590f874ffec1445020bb16b7be3bf8626db38abcf6b17``. The ARGUS repository is
private, so its licence and every pin are recorded in ``third_party/argus/PROVENANCE.md``. Copied,
never imported: this project does not depend on the ARGUS tree.

What the check is for
---------------------
A thesis is prose written by a model, and prose containing numbers is the easiest place in the whole
system for an unsupported fact to enter a trade. Given a text and the facts that were available when
it was written, this module extracts every number, tries to resolve each one, and reports what it
could not. The risk kernel reads the report: a target whose thesis, invalidation or view states a
figure that does not resolve may not add exposure (guard G9).

It does not judge whether a number is *right*, only whether it came from somewhere. That is the
weaker, checkable property: no figure in the record is unattributable.

What was kept from ARGUS
------------------------
* The number pattern, including the lookbehind that refuses digits inside identifiers (``Q3``,
  ``NVDAUSDT``, ``SP500USDT``) and the listed exclusions for years, ordinals and time-window labels
  (``"the 24h change of +4.4bps"`` holds two numbers and only one is a claim).
* Relative tolerance with an absolute floor near zero, so ``18.8`` resolves against ``18.80`` and
  ``25`` cannot pass for ``18.80``.
* The ways a model writes a rate: a fact of ``0.0203`` is quoted as ``2%`` and as ``203bps``.
* ARGUS's own tests, re-run against this module (``tests/decision/test_grounding.py``), and its
  twelve-case fabricated-figure set (three real symbols x four fabrication factors).

What this port changes, and why
-------------------------------
Each change answers a failure found by running the ARGUS logic, not by reading it:

* **A figure must match a fact to the precision it was written with.** ARGUS resolved a figure
  when it lay within 2% of any fact under any of five scalings (x1, x100, /100, x10,000, /10,000).
  Against the three facts of its own tests that is sound; against a perception snapshot it is not.
  Measured on this project's recorded snapshot (307 facts, ``tests/decision/test_grounding.py``):
  **76.6% of fabricated figures resolved by coincidence under the ARGUS rule**, and its own
  twelve-case fabricated-price set was caught 2 times in 12 instead of 12 in 12. A 2% window is far
  wider than the precision a number is written with: ``1660.677`` claims thousandths, so it is not
  a rounding of any fact 1% away from it. Here a figure resolves only when the fact, converted to
  the unit the figure was written in, equals it to within one step of its last written digit (a
  rounding or a truncation) **and** within the policy's relative tolerance. Measured on the same
  snapshot: 0.3% of fabricated figures resolve (0.1% once scoped per instrument, below), the
  twelve-case set is caught 12 in 12, and 0 of 1,091 honest copies (as shown, rounded to four or
  three significant digits, truncated, a fraction written as a percent or in bps) fail to resolve.
* **Units come from the fact's name.** A fact is a percent (a ``pct`` token), a basis-point
  figure (``bps``), a fraction (``rate``, ``gap``, ``change``, ``weight``, ...; see
  :data:`FRACTION_TOKENS`) or a level (a price, a ratio, a z-score, a count), and each pairing of a
  written unit with a fact unit has exactly one conversion factor (:data:`_CONVERSIONS`). A figure
  written with ``%`` or ``bps`` never resolves to a level: a percent is never a price.
* **The sign is part of the claim when it is written.** ``-227.49`` does not resolve against a
  price of ``227.49``. An unsigned figure may match a negative fact by magnitude, because direction
  is usually carried by a word ("down 3.2%").
* **Magnitude suffixes are read.** ``5.2B`` is five billion, not five, so open interest can be
  quoted honestly and a fabricated one cannot hide behind a scaling.
* **More non-claims are listed**, each for a failure seen in model prose: ISO dates and clock times
  (ARGUS read ``2026-09-23`` as ``-09`` and ``-23``), month-day dates, index names (``S&P 500``,
  ``Nasdaq-100``), SEC form names (``Form 4``, ``8-K``), channel-prefixed item ids (``x:1839...``),
  cluster ids and long hex digests, hyphenated window labels (``20-bar``, ``24-hour``) and window
  units the snapshot uses (bars, settlements, sessions). A year is excluded only when it is a bare
  integer from 1990 to 2039 not written as a price (``$2030`` is a claim).
* **The closest fact wins.** ARGUS resolved to the first fact within tolerance in insertion order;
  here the smallest relative error names the source, ties broken by key, so the card shows the fact
  the model most plausibly quoted.
* **Scoped per instrument** (:func:`ground_decision`). A number in NVDAUSDT's thesis resolves
  against NVDAUSDT's facts, the book-level facts, and the facts of any instrument the text names
  (matched by :func:`sentiment_agent.crowd.novelty.symbols_mentioned`), not against every
  instrument's facts.
* **Typed.** Results are the contract's :class:`~sentiment_agent.types.GroundingFigure` and
  :class:`~sentiment_agent.types.GroundingReport`; the tolerance is the policy's
  (``policy.grounding_tolerance``), passed in, not a module constant. ARGUS's ``evidence_values``
  parameter is dropped: every citable number here is already a fact.
"""

import math
import re
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Final, Literal

from sentiment_agent.crowd.novelty import symbols_mentioned
from sentiment_agent.types import GroundingFigure, GroundingReport, LlmDecision

MIN_MAGNITUDE: Final = 1e-9
"""Below this, relative tolerance is meaningless and an absolute comparison is used."""

CONTEXT_WINDOW: Final = 32
"""Characters of surrounding text kept with each figure, so a reader sees what it was said about."""

GROUNDED_FIELDS: Final[tuple[str, ...]] = ("thesis", "invalidation", "our_view")
"""The target fields whose numbers are checked. ``crowd_belief`` is deliberately not among them:
it reports what third parties claim, and a third-party number is a claim, not a fact."""

PERCENT_TOKENS: Final[frozenset[str]] = frozenset({"pct", "percent"})
BPS_TOKENS: Final[frozenset[str]] = frozenset({"bps", "bp"})
FRACTION_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "rate",
        "gap",
        "change",
        "chg",
        "return",
        "drawdown",
        "spread",
        "move",
        "weight",
        "confidence",
        "yield",
        "premium",
        "volatility",
        "tolerance",
        "fraction",
    }
)
RATE_TOKENS: Final[frozenset[str]] = PERCENT_TOKENS | BPS_TOKENS | FRACTION_TOKENS
"""A fact whose key (split on ``.`` and ``_``) carries one of these tokens is a rate: a percent, a
basis-point figure or a fraction, in that order of precedence (:func:`fact_unit`). Every other fact
is a level: a price, a ratio, a z-score, a count."""

FactUnit = Literal["percent", "bps", "fraction", "level"]
WrittenUnit = Literal["%", "bps", "", "magnitude"]

_CONVERSIONS: Final[Mapping[tuple[WrittenUnit, FactUnit], float]] = {
    ("%", "percent"): 1.0,
    ("%", "fraction"): 100.0,
    ("%", "bps"): 0.01,
    ("bps", "bps"): 1.0,
    ("bps", "percent"): 100.0,
    ("bps", "fraction"): 10_000.0,
    ("", "level"): 1.0,
    ("", "percent"): 1.0,
    ("", "bps"): 1.0,
    ("", "fraction"): 1.0,
    ("magnitude", "level"): 1.0,
}
"""The one factor that turns a fact into the unit a figure was written in. A pair that is absent
cannot match: a percent is never a price, and a figure with a magnitude suffix is a level."""

_SYMBOL_PREFIX: Final = re.compile(r"[A-Z0-9]{2,20}USDT")
"""A fact key ``SYMBOL.name`` whose prefix is a USDT perpetual belongs to that instrument."""

# --- the number itself ------------------------------------------------------------------------

_NUMBER: Final = re.compile(
    r"(?<![\w.])"  # not mid-identifier, not the tail of a dotted key
    r"(?P<num>[+-]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+))"
    r"(?:\s*(?P<unit>%|(?i:percent|pct|bps|bp|basis\s+points?)(?![A-Za-z])"
    r"|(?:[kK]|thousand|M|mn|million|B|bn|billion|tn|trillion)(?![A-Za-z])))?"
)

_MAGNITUDE: Final[Mapping[str, float]] = {
    "k": 1e3,
    "thousand": 1e3,
    "m": 1e6,
    "mn": 1e6,
    "million": 1e6,
    "b": 1e9,
    "bn": 1e9,
    "billion": 1e9,
    "tn": 1e12,
    "trillion": 1e12,
}

# --- what is not a claim about the market ----------------------------------------------------
# Listed rather than inferred, so each exclusion can be argued with individually. The first group
# masks whole spans before extraction; the second is tested per number against its trailing text.

_MONTH: Final = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"

_MASKS: Final[tuple[re.Pattern[str], ...]] = (
    # ISO dates and datetimes: 2026-09-23, 2026-09-23T13:30:00Z, 2026-09-23 13:30 UTC.
    re.compile(
        r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?"
        r"(?:Z|[+-]\d{2}:?\d{2}|\s*UTC\b)?)?"
    ),
    # Clock times: 13:30, 19:45 UTC, 09:30 ET.
    re.compile(r"(?<![\d.:])\d{1,2}:\d{2}(?::\d{2})?(?:\s*(?:UTC|GMT|ET|EST|EDT|Z)\b)?"),
    # Month-day dates: Nov 19, 19 November, Sep 23rd, 2026.
    re.compile(rf"\b{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?\b(?:,?\s+\d{{4}}\b)?"),
    re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH}(?![A-Za-z])(?:,?\s+\d{{4}}\b)?"),
    # Index names that carry a number: S&P 500, Nasdaq-100, NDX 100, Russell 2000, ...
    re.compile(
        r"(?i:S\s*&\s*P\s*-?\s*500|Nasdaq\s*-?\s*100|NDX\s*-?\s*100|Russell\s*-?\s*2000"
        r"|Dow\s*-?\s*30|FTSE\s*-?\s*100|DAX\s*-?\s*40|Nikkei\s*-?\s*225|Stoxx\s*-?\s*(?:600|50))"
    ),
    # SEC form names: Form 4, Form 8-K, 8-K, 10-Q, 13F-HR, S-1, DEF 14A.
    re.compile(r"(?i:\bForm\s+[A-Z0-9]+(?:-[A-Z0-9]+)?)"),
    re.compile(r"\b(?:8-K|10-K|10-Q|20-F|6-K|S-1|S-3|13F(?:-HR)?|13D|13G|DEF\s*14A)\b"),
    # Channel-prefixed text item ids (x:1839..., reddit:t3_1abc) and links.
    re.compile(r"\b(?:x|reddit|rdt|twitter|news|filing|item)[:][\w\-.]+"),
    re.compile(r"https?://\S+"),
    # Story cluster ids (story-<hex>) and long hex digests (snapshot, decision and ruling ids).
    re.compile(r"\bstory-[0-9a-f]+\b"),
    re.compile(r"\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{12,}\b"),
)

_YEAR: Final = re.compile(r"(?:199\d|20[0-3]\d)")
_ORDINAL: Final = re.compile(r"[+-]?\d+(?:st|nd|rd|th)\b", re.IGNORECASE)
# A time-window label names the period a quantity was measured over; it is not itself a claim.
# Single-letter units are case-sensitive on purpose: "5M" is five million, "5m" is five minutes.
_TIME_WINDOW: Final = re.compile(
    r"[+-]?\d+(?:,\d{3})*(?:\.\d+)?\s*-?\s*"
    r"(?:(?:h|H|hr|hrs|d|D|m|min|mins|w|W|wk|wks|mo|y|yr|yrs|s|sec|secs)\b"
    r"|(?i:hours?|days?|minutes?|weeks?|months?|years?|seconds?|bars?|candles?"
    r"|settlements?|sessions?|periods?|trading\s+days?)\b)"
)


def _masked_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for pattern in _MASKS:
        spans.extend(m.span() for m in pattern.finditer(text))
    return spans


def _inside(position: int, spans: Iterable[tuple[int, int]]) -> bool:
    return any(start <= position < end for start, end in spans)


def _is_excluded(text: str, start: int, raw: str, unit: str) -> bool:
    """ARGUS's per-number exclusions (years, ordinals, time windows), with the refinements above."""
    trailing = text[start + len(raw) : start + len(raw) + 24]
    joined = f"{raw}{trailing}"
    if _ORDINAL.match(joined) or _TIME_WINDOW.match(joined):
        return True
    if unit or raw[0] in "+-" or not _YEAR.fullmatch(raw):
        return False
    # A bare four-digit year, unless it is written as a price.
    return not text[max(0, start - 1) : start].endswith("$")


def _normalise(text: str) -> str:
    """NFKC (fullwidth digits become ASCII) and the Unicode minus sign read as a hyphen-minus."""
    return unicodedata.normalize("NFKC", text).replace("\u2212", "-")


def extract(text: str) -> tuple[GroundingFigure, ...]:
    """Every numeric claim in ``text``, unresolved, each with the text around it."""
    normalised = _normalise(text)
    masked = _masked_spans(normalised)
    figures: list[GroundingFigure] = []
    for match in _NUMBER.finditer(normalised):
        raw_number = match.group("num")
        start = match.start("num")
        if _inside(start, masked):
            continue
        unit_raw = match.group("unit") or ""
        if _is_excluded(normalised, start, raw_number, unit_raw):
            continue
        value = float(raw_number.replace(",", ""))
        unit = _unit_of(unit_raw)
        if unit in _MAGNITUDE:
            value *= _MAGNITUDE[unit]
            unit = ""
        begin = max(0, match.start() - CONTEXT_WINDOW)
        end = min(len(normalised), match.end() + CONTEXT_WINDOW)
        figures.append(
            GroundingFigure(
                raw=match.group(0).strip(),
                value=value,
                unit=unit,
                context=" ".join(normalised[begin:end].split()),
                resolved=False,
                source=None,
                known_value=None,
            )
        )
    return tuple(figures)


def _unit_of(raw_unit: str) -> str:
    folded = " ".join(raw_unit.split())
    if folded == "%" or folded.lower() in ("percent", "pct"):
        return "%"
    if folded.lower() in ("bps", "bp", "basis point", "basis points"):
        return "bps"
    if folded in ("M",):
        return "m"
    if folded in ("B",):
        return "b"
    return folded.lower()


# --- resolution -------------------------------------------------------------------------------


def fact_unit(key: str) -> FactUnit:
    """The unit of the fact named ``key``, from the tokens of its name (see :data:`RATE_TOKENS`)."""
    tokens = set(re.split(r"[._]", key.lower()))
    if tokens & PERCENT_TOKENS:
        return "percent"
    if tokens & BPS_TOKENS:
        return "bps"
    if tokens & FRACTION_TOKENS:
        return "fraction"
    return "level"


def is_rate_key(key: str) -> bool:
    """Whether the fact named ``key`` is a rate (a percent, a bps figure or a fraction)."""
    return fact_unit(key) != "level"


def _written(figure: GroundingFigure) -> tuple[WrittenUnit, float]:
    """How ``figure`` was written: its unit, and one step in its last written digit.

    The step is the precision the writer claimed: ``227.49`` claims hundredths, ``19`` units,
    ``10000`` ten-thousands (trailing zeros of an integer are not taken as significant), and
    ``5.2B`` a tenth of a billion.
    """
    match = _NUMBER.match(figure.raw)
    number = match.group("num") if match is not None else figure.raw
    unit_raw = (match.group("unit") or "") if match is not None else ""
    digits = number.lstrip("+-").replace(",", "")
    if "." in digits:
        step = 10.0 ** -len(digits.split(".", 1)[1])
    else:
        significant = digits.rstrip("0")
        step = 10.0 ** (len(digits) - len(significant)) if significant else 1.0
    folded = _unit_of(unit_raw)
    if folded in _MAGNITUDE:
        return "magnitude", step * _MAGNITUDE[folded]
    if folded in ("%", "bps"):
        return ("%" if folded == "%" else "bps"), step
    return "", step


def _error_if_match(written: float, fact: float, step: float, tolerance: float) -> float | None:
    """The relative error when ``written`` is ``fact`` as written, else ``None``.

    Two conditions, both required. The figure must be the fact to the precision written, to within
    one step of its last digit (a rounding or a truncation), and within the policy's relative
    ``tolerance`` (so a coarse rounding cannot stretch far: ``5`` is not ``4.6``). ARGUS used the
    relative tolerance alone, which lets any number land within 2% of *some* fact once a snapshot
    holds a few hundred of them; the precision condition is what makes a fabricated figure fail.
    """
    error = abs(written - fact)
    scale = max(abs(written), abs(fact))
    if scale < MIN_MAGNITUDE:
        return 0.0 if error < MIN_MAGNITUDE else None
    if error > step + 1e-9 * scale:
        return None
    relative = error / scale
    return relative if relative <= tolerance else None


def _signed(raw: str) -> bool:
    return raw.lstrip()[:1] in ("+", "-")


def _resolve(
    figure: GroundingFigure, facts: Mapping[str, float], tolerance: float
) -> GroundingFigure:
    written_unit, step = _written(figure)
    signed = _signed(figure.raw)
    best: tuple[float, str, float] | None = None
    for key in sorted(facts):
        fact = facts[key]
        factor = _CONVERSIONS.get((written_unit, fact_unit(key)))
        if factor is None or not math.isfinite(fact):
            continue
        converted = fact * factor
        # A written sign is part of the claim; an unsigned figure may quote a magnitude.
        candidates = (converted,) if signed or converted >= 0 else (converted, -converted)
        for candidate in candidates:
            error = _error_if_match(figure.value, candidate, step, tolerance)
            if error is not None and (best is None or error < best[0]):
                best = (error, key, fact)
    if best is None:
        return figure
    return figure.model_copy(update={"resolved": True, "source": best[1], "known_value": best[2]})


def check(text: str, *, facts: Mapping[str, float], tolerance: float) -> GroundingReport:
    """Resolve every figure in ``text`` against ``facts`` within a relative ``tolerance``."""
    if not 0 < tolerance < 1:
        raise ValueError("tolerance must be a fraction between 0 and 1")
    return GroundingReport(figures=tuple(_resolve(f, facts, tolerance) for f in extract(text)))


# --- a whole decision -------------------------------------------------------------------------


def instrument_of(key: str) -> str | None:
    """The instrument a fact key belongs to (``"NVDAUSDT.funding_z_live"`` -> ``"NVDAUSDT"``)."""
    prefix = key.split(".", 1)[0]
    return prefix if "." in key and _SYMBOL_PREFIX.fullmatch(prefix) else None


def ground_decision(
    decision: LlmDecision, facts: Mapping[str, float], tolerance: float
) -> dict[str, GroundingReport]:
    """One report per addressed symbol, over its thesis, invalidation and our_view.

    The facts a target's text may resolve against are: every fact that belongs to no instrument
    (the book, the session, the mood, the kernel's rules), the target's own instrument's facts, and
    the facts of every instrument its text names. The decision's own numbers for a target (its
    ``target`` and ``confidence``) are citable facts of that target's instrument: a model that says
    "confidence 55%" is quoting its own structured answer, which the record already holds.
    """
    instruments = sorted(
        {s for s in (instrument_of(k) for k in facts) if s is not None} | set(decision.symbols)
    )
    own: dict[str, float] = {}
    for target in decision.targets:
        own[f"{target.symbol}.proposed_target"] = target.target
        own[f"{target.symbol}.proposed_confidence"] = target.confidence
    reports: dict[str, GroundingReport] = {}
    for target in decision.targets:
        texts = {name: getattr(target, name) for name in GROUNDED_FIELDS}
        named = set(symbols_mentioned(" ".join(texts.values()), instruments))
        allowed = {target.symbol} | named
        scoped = {
            key: value
            for key, value in (*facts.items(), *own.items())
            if (owner := instrument_of(key)) is None or owner in allowed
        }
        figures: list[GroundingFigure] = []
        for name, text in texts.items():
            for figure in check(text, facts=scoped, tolerance=tolerance).figures:
                figures.append(figure.model_copy(update={"context": f"{name}: {figure.context}"}))
        reports[target.symbol] = GroundingReport(figures=tuple(figures))
    return reports


def render(report: GroundingReport) -> list[str]:
    """Plain-language lines for a decision card (ARGUS ``GroundingReport.render``)."""
    if not report.figures:
        return ["[grounding] the text states no figures"]
    if report.grounded:
        return [f"[grounding] all {len(report.figures)} figure(s) resolve to a fact"]
    names = ", ".join(sorted({f.raw for f in report.unresolved})[:6])
    return [
        f"[grounding] {len(report.unresolved)} of {len(report.figures)} figure(s) do not "
        f"resolve to anything the agent was given: {names}",
        "[grounding] an unattributable number in a thesis is the easiest place for an "
        "unsupported fact to enter the record",
    ]


__all__ = [
    "BPS_TOKENS",
    "CONTEXT_WINDOW",
    "FRACTION_TOKENS",
    "GROUNDED_FIELDS",
    "MIN_MAGNITUDE",
    "PERCENT_TOKENS",
    "RATE_TOKENS",
    "FactUnit",
    "check",
    "extract",
    "fact_unit",
    "ground_decision",
    "instrument_of",
    "is_rate_key",
    "render",
]
