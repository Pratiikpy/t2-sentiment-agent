"""Numeric grounding (guard G9): every figure in a thesis must resolve to a fact it was shown.

A port of the primary's ``sentiment_agent.decision.grounding`` (itself ported from ARGUS
``agents/grounding.py``, MIT, same author), rule for rule: the number pattern and its exclusions
(dates, clock times, index and form names, item ids, hex digests, years, ordinals, time windows),
units read from the fact's name, a match to the precision the figure was written with *and* within
the policy's relative tolerance, the sign binding when written, the closest fact winning, and the
scope per instrument (a target's text resolves against its own instrument's facts, the book-level
facts, and the facts of any instrument its text names).

Three differences, each forced by the sandbox and each covered by the replica's parity tests:

* No ``unicodedata``: the NFKC normalisation becomes :func:`src.mentions.fold` (fullwidth digits
  and signs folded, the Unicode minus read as ``-``, invisible characters removed).
* No ``re.compile``: the upload validator rejects any call named ``compile``, so patterns are kept
  as strings and passed to ``re.finditer`` and ``re.match`` (which cache compiled patterns).
* No pydantic: figures and reports are plain dataclasses with the same fields.
"""

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace

from . import policy_v1 as policy
from .mentions import fold, symbols_mentioned

_CONVERSIONS: dict[tuple[str, str], float] = {
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

_SYMBOL_PREFIX = r"[A-Z0-9]{2,20}USDT"

_NUMBER = (
    r"(?<![\w.])"
    r"(?P<num>[+-]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+))"
    r"(?:\s*(?P<unit>%|(?i:percent|pct|bps|bp|basis\s+points?)(?![A-Za-z])"
    r"|(?:[kK]|thousand|M|mn|million|B|bn|billion|tn|trillion)(?![A-Za-z])))?"
)

_MAGNITUDE: dict[str, float] = {
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

_MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"

_MASKS: tuple[str, ...] = (
    r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?"
    r"(?:Z|[+-]\d{2}:?\d{2}|\s*UTC\b)?)?",
    r"(?<![\d.:])\d{1,2}:\d{2}(?::\d{2})?(?:\s*(?:UTC|GMT|ET|EST|EDT|Z)\b)?",
    rf"\b{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?\b(?:,?\s+\d{{4}}\b)?",
    rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH}(?![A-Za-z])(?:,?\s+\d{{4}}\b)?",
    r"(?i:S\s*&\s*P\s*-?\s*500|Nasdaq\s*-?\s*100|NDX\s*-?\s*100|Russell\s*-?\s*2000"
    r"|Dow\s*-?\s*30|FTSE\s*-?\s*100|DAX\s*-?\s*40|Nikkei\s*-?\s*225|Stoxx\s*-?\s*(?:600|50))",
    r"(?i:\bForm\s+[A-Z0-9]+(?:-[A-Z0-9]+)?)",
    r"\b(?:8-K|10-K|10-Q|20-F|6-K|S-1|S-3|13F(?:-HR)?|13D|13G|DEF\s*14A)\b",
    r"\b(?:x|reddit|rdt|twitter|news|filing|item)[:][\w\-.]+",
    r"https?://\S+",
    r"\bstory-[0-9a-f]+\b",
    r"\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{12,}\b",
)

_YEAR = r"(?:199\d|20[0-3]\d)"
_ORDINAL = r"[+-]?\d+(?:st|nd|rd|th)\b"
_TIME_WINDOW = (
    r"[+-]?\d+(?:,\d{3})*(?:\.\d+)?\s*-?\s*"
    r"(?:(?:h|H|hr|hrs|d|D|m|min|mins|w|W|wk|wks|mo|y|yr|yrs|s|sec|secs)\b"
    r"|(?i:hours?|days?|minutes?|weeks?|months?|years?|seconds?|bars?|candles?"
    r"|settlements?|sessions?|periods?|trading\s+days?)\b)"
)


@dataclass(frozen=True)
class Figure:
    raw: str
    value: float
    unit: str
    context: str
    resolved: bool = False
    source: str | None = None
    known_value: float | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "raw": self.raw,
            "value": self.value if math.isfinite(self.value) else None,
            "unit": self.unit,
            "context": self.context,
            "resolved": self.resolved,
            "source": self.source,
            "known_value": self.known_value,
        }


@dataclass(frozen=True)
class Report:
    figures: tuple[Figure, ...]

    @property
    def unresolved(self) -> tuple[Figure, ...]:
        return tuple(f for f in self.figures if not f.resolved)

    @property
    def grounded(self) -> bool:
        return not self.unresolved

    @property
    def coverage(self) -> float:
        return 1.0 if not self.figures else 1 - len(self.unresolved) / len(self.figures)

    def to_json(self) -> dict[str, object]:
        return {
            "grounded": self.grounded,
            "coverage": self.coverage,
            "figures": [f.to_json() for f in self.figures],
        }


def _masked_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for pattern in _MASKS:
        spans.extend(m.span() for m in re.finditer(pattern, text))
    return spans


def _inside(position: int, spans: list[tuple[int, int]]) -> bool:
    return any(start <= position < end for start, end in spans)


def _is_excluded(text: str, start: int, raw: str, unit: str) -> bool:
    trailing = text[start + len(raw) : start + len(raw) + 24]
    joined = f"{raw}{trailing}"
    if re.match(_ORDINAL, joined, flags=re.IGNORECASE) or re.match(_TIME_WINDOW, joined):
        return True
    if unit or raw[0] in "+-" or not re.fullmatch(_YEAR, raw):
        return False
    return not text[max(0, start - 1) : start].endswith("$")


def _unit_of(raw_unit: str) -> str:
    folded = " ".join(raw_unit.split())
    if folded == "%" or folded.lower() in ("percent", "pct"):
        return "%"
    if folded.lower() in ("bps", "bp", "basis point", "basis points"):
        return "bps"
    if folded == "M":
        return "m"
    if folded == "B":
        return "b"
    return folded.lower()


def extract(text: str) -> tuple[Figure, ...]:
    normalised = fold(text)
    masked = _masked_spans(normalised)
    figures: list[Figure] = []
    for match in re.finditer(_NUMBER, normalised):
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
        begin = max(0, match.start() - policy.GROUNDING_CONTEXT_WINDOW)
        end = min(len(normalised), match.end() + policy.GROUNDING_CONTEXT_WINDOW)
        figures.append(
            Figure(
                raw=match.group(0).strip(),
                value=value,
                unit=unit,
                context=" ".join(normalised[begin:end].split()),
            )
        )
    return tuple(figures)


def fact_unit(key: str) -> str:
    tokens = set(re.split(r"[._]", key.lower()))
    if tokens & set(policy.GROUNDING_PERCENT_TOKENS):
        return "percent"
    if tokens & set(policy.GROUNDING_BPS_TOKENS):
        return "bps"
    if tokens & set(policy.GROUNDING_FRACTION_TOKENS):
        return "fraction"
    return "level"


def _written(figure: Figure) -> tuple[str, float]:
    match = re.match(_NUMBER, figure.raw)
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
        return folded, step
    return "", step


def _error_if_match(written: float, fact: float, step: float, tolerance: float) -> float | None:
    error = abs(written - fact)
    scale = max(abs(written), abs(fact))
    if scale < policy.GROUNDING_MIN_MAGNITUDE:
        return 0.0 if error < policy.GROUNDING_MIN_MAGNITUDE else None
    if error > step + 1e-9 * scale:
        return None
    relative = error / scale
    return relative if relative <= tolerance else None


def _resolve(figure: Figure, facts: Mapping[str, float], tolerance: float) -> Figure:
    written_unit, step = _written(figure)
    signed = figure.raw.lstrip()[:1] in ("+", "-")
    best: tuple[float, str, float] | None = None
    for key in sorted(facts):
        fact = facts[key]
        factor = _CONVERSIONS.get((written_unit, fact_unit(key)))
        if factor is None or not math.isfinite(fact):
            continue
        converted = fact * factor
        candidates = (converted,) if signed or converted >= 0 else (converted, -converted)
        for candidate in candidates:
            error = _error_if_match(figure.value, candidate, step, tolerance)
            if error is not None and (best is None or error < best[0]):
                best = (error, key, fact)
    if best is None:
        return figure
    return replace(figure, resolved=True, source=best[1], known_value=best[2])


def check(text: str, *, facts: Mapping[str, float], tolerance: float) -> Report:
    if not 0 < tolerance < 1:
        raise ValueError("tolerance must be a fraction between 0 and 1")
    return Report(figures=tuple(_resolve(f, facts, tolerance) for f in extract(text)))


def instrument_of(key: str) -> str | None:
    prefix = key.split(".", 1)[0]
    return prefix if "." in key and re.fullmatch(_SYMBOL_PREFIX, prefix) else None


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float("nan")
    return float(value)


def ground_targets(
    targets: list[dict[str, object]], facts: Mapping[str, float], tolerance: float
) -> dict[str, Report]:
    """One report per addressed symbol, over its thesis, invalidation and our_view
    (``ground_decision`` in the primary). ``targets`` are validated decision targets."""
    symbols = [str(t["symbol"]) for t in targets]
    instruments = sorted(
        {s for s in (instrument_of(k) for k in facts) if s is not None} | set(symbols)
    )
    own: dict[str, float] = {}
    for target in targets:
        symbol = str(target["symbol"])
        own[f"{symbol}.proposed_target"] = _as_float(target.get("target"))
        own[f"{symbol}.proposed_confidence"] = _as_float(target.get("confidence"))
    reports: dict[str, Report] = {}
    for target in targets:
        symbol = str(target["symbol"])
        texts = {name: str(target.get(name) or "") for name in policy.GROUNDED_FIELDS}
        named = set(symbols_mentioned(" ".join(texts.values()), instruments))
        allowed = {symbol} | named
        scoped = {
            key: value
            for key, value in (*facts.items(), *own.items())
            if (owner := instrument_of(key)) is None or owner in allowed
        }
        figures: list[Figure] = []
        for name, text in texts.items():
            for figure in check(text, facts=scoped, tolerance=tolerance).figures:
                figures.append(replace(figure, context=f"{name}: {figure.context}"))
        reports[symbol] = Report(figures=tuple(figures))
    return reports
