"""Numeric grounding (G9): every figure in a thesis must resolve to a fact the model was shown.

Three kinds of test, each answering a different doubt:

* **ARGUS's own tests, re-run here** against the port, so what was kept is shown to still hold.
* **The fabricated-figure set, re-run here** rather than cited: ARGUS's twelve recorded cases
  (``tests/fixtures/decision/fabricated_figures.json``), first against the facts ARGUS used, then
  against this project's full recorded snapshot, where the original rule fails.
* **Measurements on the full snapshot**: how often a fabricated figure resolves by coincidence
  (the original rule reproduced beside the port, so the change is shown and not asserted), and
  that no honest copy of a fact fails to resolve.
"""

import json
import math
import random
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from decision.support import NOW, build_snapshot, decision, funding_event, heartbeat, held_book
from decision.support import target as target_json
from sentiment_agent.decision.grounding import (
    check,
    extract,
    fact_unit,
    ground_decision,
    instrument_of,
    is_rate_key,
    render,
)
from sentiment_agent.decision.prompt import decision_facts, format_number
from sentiment_agent.policy import POLICY_V1
from sentiment_agent.types import LlmDecision

TOL = POLICY_V1.grounding_tolerance
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "decision" / "fabricated_figures.json"

# ARGUS tests/test_grounding.py, verbatim facts and thesis (its live ledger, seq 26).
FACTS = {
    "hurdle_bps": 18.80,
    "move_24h_bps": 4.4,
    "round_trip_bps": 12.0,
    "hours_to_discovery": 46.0,
}
REAL_THESIS = (
    "Weekend session with 46+ hours to genuine price discovery, no hedge menu available, and "
    "news mix is balanced with no surprise catalyst. The 24h change of +4.4bps is below the "
    "18.80bps total hurdle, making any directional bet a negative-expectancy proposition."
)


def _fixture() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return loaded


@pytest.fixture(scope="module")
def snapshot_facts() -> dict[str, float]:
    """The grounding reference of a real decision prompt: the recorded snapshot, a held book."""
    book = held_book()
    snapshot = build_snapshot(book=book)
    return decision_facts(snapshot, book, [heartbeat(), funding_event()], POLICY_V1, now=NOW)


# ================================================================================================
# The ARGUS original rule, reproduced for comparison (argus/src/argus/agents/grounding.py at
# commit 3dec6baf, MIT, same author): five scalings, 2% relative, first match wins.
# ================================================================================================

_ARGUS_NUMBER = re.compile(
    r"(?<![\w.])([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(%|bps|bp|basis points)?", re.I
)
_ARGUS_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_ARGUS_ORDINAL = re.compile(r"\b\d+(?:st|nd|rd|th)\b", re.I)
_ARGUS_WINDOW = re.compile(
    r"\d+\s*(?:h|hr|hrs|hour|hours|d|day|days|m|min|mins|minute|minutes|w|wk|week|weeks|"
    r"mo|month|months|y|yr|year|years)\b",
    re.I,
)


def argus_original_grounded(text: str, facts: Mapping[str, float]) -> bool:
    """ARGUS ``check(text, facts=...).grounded``, reproduced line for line."""
    for match in _ARGUS_NUMBER.finditer(text):
        raw, unit = match.group(1), (match.group(2) or "").lower()
        joined = f"{raw}{text[match.end(1) : match.end(1) + 12]}"
        if (
            _ARGUS_YEAR.fullmatch(raw.strip("+-"))
            or _ARGUS_ORDINAL.match(joined)
            or _ARGUS_WINDOW.match(joined)
        ):
            continue
        value = float(raw.replace(",", ""))
        unit = {"bp": "bps", "basis points": "bps"}.get(unit, unit)
        if unit == "%":
            written: tuple[float, ...] = (value, value / 100, value * 100)
        elif unit == "bps":
            written = (value, value / 10_000, value / 100)
        else:
            written = (value, value * 100, value / 100, value * 10_000, value / 10_000)

        def close(a: float, b: float) -> bool:
            scale = max(abs(a), abs(b))
            return abs(a - b) < 1e-9 if scale < 1e-9 else abs(a - b) / scale <= 0.02

        if not any(close(w, v) for v in facts.values() for w in written):
            return False
    return True


# ================================================================================================
# ARGUS's own tests, re-run against the port
# ================================================================================================


class TestAgainstTheRealRecord:
    def test_a_real_thesis_grounds_completely(self) -> None:
        report = check(REAL_THESIS, facts=FACTS, tolerance=TOL)
        assert report.grounded, [f.raw for f in report.unresolved]
        assert report.coverage == 1.0

    def test_it_finds_the_figures_that_matter(self) -> None:
        raws = {f.raw for f in extract(REAL_THESIS)}
        assert "+4.4bps" in raws
        assert "18.80bps" in raws

    def test_the_time_window_label_is_not_treated_as_a_claim(self) -> None:
        assert "24" not in {f.raw for f in extract(REAL_THESIS)}

    def test_a_thesis_with_no_figures_is_grounded_vacuously(self) -> None:
        report = check("No actionable edge exists; any drift is noise.", facts=FACTS, tolerance=TOL)
        assert report.grounded
        assert "states no figures" in render(report)[0]


class TestItCatchesAnUnsupportedNumber:
    def test_an_invented_figure_does_not_resolve(self) -> None:
        report = check("The spread is 250bps, far beyond the hurdle.", facts=FACTS, tolerance=TOL)
        assert not report.grounded
        assert report.unresolved[0].value == 250.0

    def test_the_report_names_the_offending_figure_and_says_why(self) -> None:
        lines = " ".join(render(check("Expected move is 97bps.", facts=FACTS, tolerance=TOL)))
        assert "97bps" in lines
        assert "unsupported fact to enter the record" in lines

    def test_coverage_is_a_fraction_not_a_verdict(self) -> None:
        report = check(
            "The hurdle is 18.80bps and the target is 900bps.", facts=FACTS, tolerance=TOL
        )
        assert report.coverage == pytest.approx(0.5)


class TestTolerantButBounded:
    def test_a_rounded_quote_resolves(self) -> None:
        assert check("hurdle 18.8bps", facts=FACTS, tolerance=TOL).grounded

    def test_a_materially_different_number_does_not_pass_as_rounding(self) -> None:
        assert not check("hurdle 25bps", facts=FACTS, tolerance=TOL).grounded

    def test_a_percentage_resolves_against_a_fraction(self) -> None:
        assert check("the move was 2%", facts={"move": 0.0203}, tolerance=TOL).grounded

    def test_bps_resolves_against_a_fraction(self) -> None:
        assert check("the move was 203bps", facts={"move": 0.0203}, tolerance=TOL).grounded

    def test_a_zero_valued_fact_does_not_divide_by_zero(self) -> None:
        assert check("position is 0", facts={"position": 0.0}, tolerance=TOL).grounded

    def test_the_tolerance_must_be_a_fraction(self) -> None:
        with pytest.raises(ValueError, match="tolerance"):
            check("x 1", facts={}, tolerance=0.0)
        with pytest.raises(ValueError, match="tolerance"):
            check("x 1", facts={}, tolerance=1.5)


class TestWhatIsNotAClaim:
    @pytest.mark.parametrize(
        "text",
        [
            # ARGUS's list
            "the 2026 filing season",
            "the 3rd consecutive session",
            "over the last 24h",
            "in the past 7 days",
            "a 15m candle",
            # added in the port, each seen in model prose
            "decided at 2026-09-23T13:30:00Z",
            "on 2026-09-23",
            "the open at 13:30 UTC",
            "pre-flatten from 19:45",
            "earnings on Nov 19",
            "earnings on 19 November, 2026",
            "the S&P 500 and the Nasdaq-100",
            "a Form 4 and an 8-K and a 10-Q",
            "see x:1839204411 and reddit:t3_1abc9",
            "story story-331620b13046570c",
            "snapshot 60dd0b46d96ce1b656e369c863897b98",
            "the 20-bar mean and 14-bar ATR over 90 settlements",
            "a 24-hour hold",
            "the 1H candles",
            "https://example.com/2026/09/23/report-42",
        ],
    )
    def test_these_are_not_treated_as_market_claims(self, text: str) -> None:
        assert extract(text) == (), f"{text!r} produced {[f.raw for f in extract(text)]}"

    def test_a_real_claim_beside_an_excluded_one_still_registers(self) -> None:
        assert [f.raw for f in extract("over the last 24h the move was 44bps")] == ["44bps"]

    def test_a_year_written_as_a_price_is_a_claim(self) -> None:
        assert [f.value for f in extract("entry at $2030")] == [2030.0]

    def test_numbers_inside_identifiers_and_fact_keys_are_not_claims(self) -> None:
        text = "SP500USDT and NDX100USDT and ma20 and trigger.2.observed and oi_change_1h_pct"
        assert extract(text) == ()


class TestHowAFigureIsRead:
    def test_magnitude_suffixes_scale_the_value(self) -> None:
        assert [f.value for f in extract("open interest 5.2B, then 350k, then 1.5 million")] == [
            5.2e9,
            350e3,
            1.5e6,
        ]

    def test_five_m_is_minutes_and_five_capital_m_is_millions(self) -> None:
        assert extract("a 5m candle") == ()
        assert [f.value for f in extract("volume 5M")] == [5e6]

    def test_unicode_minus_and_fullwidth_digits(self) -> None:
        figures = extract("funding z −2.4 and ５%")
        assert [(f.value, f.unit) for f in figures] == [(-2.4, ""), (5.0, "%")]

    def test_every_figure_carries_its_surrounding_text(self) -> None:
        for figure in extract(REAL_THESIS):
            assert figure.context


# ================================================================================================
# The precision rule, the units, the sign
# ================================================================================================


class TestPrecisionAndTolerance:
    """ARGUS's ablation (real value 189.4991) re-read under the precision rule."""

    @pytest.mark.parametrize("written", ["189.4991", "189.5", "189.50", "189", "190", "189.49"])
    def test_honest_roundings_and_truncations_resolve(self, written: str) -> None:
        assert check(f"entry near {written}", facts={"price": 189.4991}, tolerance=TOL).grounded

    @pytest.mark.parametrize(
        ("written", "why"),
        [
            (
                "200",
                "a rounding to hundreds is 5.5% off: the policy tolerance caps coarse rounding",
            ),
            ("193", "3.5 away while claiming units"),
            ("191.3941", "ARGUS's 'just inside' figure: 1% off while claiming ten-thousandths"),
            ("189.6", "0.1 away while claiming tenths is one step, but 189.6 is not 189.4991"),
        ],
    )
    def test_a_figure_that_is_not_the_fact_as_written_does_not_resolve(
        self, written: str, why: str
    ) -> None:
        del why
        # 189.6 is exactly one step (0.1) from 189.5 but 0.1009 from the fact: outside one step.
        assert not check(f"entry near {written}", facts={"price": 189.4991}, tolerance=TOL).grounded

    def test_both_conditions_are_load_bearing(self) -> None:
        facts = {"price": 227.49}
        assert not check("near 229.7", facts=facts, tolerance=TOL).grounded  # 1% off, claims 0.1
        assert check("near 230", facts=facts, tolerance=TOL).grounded  # 1.1% off, claims tens
        assert not check("near 240", facts=facts, tolerance=TOL).grounded  # claims tens, 5.5% off

    def test_the_closest_fact_names_the_source(self) -> None:
        report = check("level 101", facts={"a_level": 100.4, "b_level": 101.0}, tolerance=TOL)
        assert report.figures[0].source == "b_level"
        assert report.figures[0].known_value == 101.0


class TestUnits:
    @pytest.mark.parametrize(
        ("key", "unit"),
        [
            ("NVDAUSDT.oi_change_1h_pct", "percent"),
            ("book.drawdown_pct", "percent"),
            ("NVDAUSDT.demo_index_move_bps_3h", "bps"),
            ("kernel.max_open_spread_bps", "bps"),
            ("BTCUSDT.funding_rate_live", "fraction"),
            ("NVDAUSDT.proposed_confidence", "fraction"),
            ("book.gross_weight", "fraction"),
            ("BTCUSDT.funding_z_live", "level"),
            ("NVDAUSDT.demo_last", "level"),
            ("crowd.duplication_ratio", "level"),
            ("NVDAUSDT.position_target_equivalent", "level"),
        ],
    )
    def test_the_unit_comes_from_the_key(self, key: str, unit: str) -> None:
        assert fact_unit(key) == unit
        assert is_rate_key(key) is (unit != "level")

    def test_a_percent_never_resolves_to_a_price(self) -> None:
        assert not check("up 227.49%", facts={"NVDAUSDT.demo_last": 227.49}, tolerance=TOL).grounded

    def test_rates_convert_between_fraction_percent_and_bps(self) -> None:
        funding = {"BTCUSDT.funding_rate_live": 0.000048}
        for written in ("0.000048", "0.0048%", "0.48 bps", "0.48bps"):
            assert check(f"funding {written}", facts=funding, tolerance=TOL).grounded, written
        change = {"NVDAUSDT.oi_change_1h_pct": 1.186}
        for written in ("1.186%", "1.19%", "1.2%", "1.18%", "118.6 bps", "1.186"):
            assert check(f"OI {written}", facts=change, tolerance=TOL).grounded, written
        spread = {"NVDAUSDT.demo_spread_bps": 3.59034}
        for written in ("3.59 bps", "3.6bps", "0.0359%"):
            assert check(f"spread {written}", facts=spread, tolerance=TOL).grounded, written

    def test_a_magnitude_suffix_matches_a_level_only_at_its_scale(self) -> None:
        oi = {"BTCUSDT.open_interest_live": 5_212_345_678.0}
        assert check("OI of 5.2B", facts=oi, tolerance=TOL).grounded
        assert not check("OI of 5.2M", facts=oi, tolerance=TOL).grounded
        assert not check("OI of 5.2", facts=oi, tolerance=TOL).grounded


class TestSign:
    def test_a_written_sign_is_part_of_the_claim(self) -> None:
        assert not check("entry near -227.49", facts={"p": 227.49}, tolerance=TOL).grounded
        assert not check("24h change +3.2%", facts={"c_pct": -3.2}, tolerance=TOL).grounded
        assert check("24h change -3.2%", facts={"c_pct": -3.2}, tolerance=TOL).grounded

    def test_an_unsigned_figure_may_quote_a_magnitude(self) -> None:
        assert check("down 3.2% on the day", facts={"c_pct": -3.2}, tolerance=TOL).grounded


# ================================================================================================
# The fabricated-figure set, re-run here
# ================================================================================================


def _case_text(case: Mapping[str, Any], figure: float) -> str:
    template: str = _fixture()["thesis_template"]
    return template.format(figure=figure, symbol=case["symbol"])


class TestTheFabricatedFigureSet:
    def test_the_fixture_holds_the_twelve_recorded_cases(self) -> None:
        cases = _fixture()["cases"]
        assert len(cases) == 12
        assert {c["symbol"] for c in cases} == {"NVDAUSDT", "AAPLUSDT", "MSFTUSDT"}
        assert {c["factor"] for c in cases} == {7.3, 0.03, -1.0, 1000.0}

    def test_every_case_is_flagged_against_the_facts_argus_used(self) -> None:
        for case in _fixture()["cases"]:
            facts = {"current_price": case["real_current_price"]}
            text = _case_text(case, case["fabricated_entry_price"])
            report = check(text, facts=facts, tolerance=TOL)
            assert not report.grounded, text

    def test_the_positive_control_resolves(self) -> None:
        for row in _fixture()["positive_control"]:
            text = f"Entry near {row['real_price']} on {row['symbol']}."
            assert check(text, facts={"current_price": row["real_price"]}, tolerance=TOL).grounded

    def test_every_case_is_flagged_against_a_full_snapshot(
        self, snapshot_facts: dict[str, float]
    ) -> None:
        """The same twelve figures against the 300-odd facts of a real decision prompt."""
        facts = snapshot_facts
        assert len(facts) > 250
        for case in _fixture()["cases"]:
            text = _case_text(case, case["fabricated_entry_price"])
            assert not check(text, facts=facts, tolerance=TOL).grounded, text

    def test_the_original_rule_misses_most_of_them_against_a_full_snapshot(
        self, snapshot_facts: dict[str, float]
    ) -> None:
        """Why the precision rule exists: measured, the ARGUS rule caught 2 of 12 here."""
        facts = snapshot_facts
        caught = sum(
            not argus_original_grounded(_case_text(c, c["fabricated_entry_price"]), facts)
            for c in _fixture()["cases"]
        )
        assert caught <= 6

    def test_every_case_is_flagged_inside_a_decision(
        self, snapshot_facts: dict[str, float]
    ) -> None:
        facts = snapshot_facts
        for case in _fixture()["cases"]:
            symbol = case["symbol"] if case["symbol"] in POLICY_V1.symbols else "NVDAUSDT"
            body = decision(
                "act",
                [
                    target_json(
                        symbol, -0.4, thesis=_case_text(case, case["fabricated_entry_price"])
                    )
                ],
            )
            reports = ground_decision(LlmDecision.model_validate(body), facts, TOL)
            assert not reports[symbol].grounded, case


# ================================================================================================
# Measurements on the full snapshot
# ================================================================================================


def _fabricated(rng: random.Random, values: list[float]) -> str:
    """A figure between 10% and 10x of a real fact, never within 10% of it, at four to six
    significant digits: the shape of a price or level a model might invent."""
    base = abs(rng.choice(values))
    while True:
        factor = math.exp(rng.uniform(math.log(0.1), math.log(10)))
        if not 0.9 < factor < 1.1:
            break
    value = base * factor
    decimals = min(6, max(0, 4 - math.floor(math.log10(value))))
    return f"{value:.{decimals}f}"


class TestMeasuredOnTheFullSnapshot:
    def test_fabricated_figures_almost_never_resolve_by_coincidence(
        self, snapshot_facts: dict[str, float]
    ) -> None:
        facts = snapshot_facts
        values = [v for v in facts.values() if abs(v) > 1e-9]
        rng = random.Random(7)  # noqa: S311 - a reproducible sample, not a secret
        figures = [_fabricated(rng, values) for _ in range(600)]
        ours = sum(check(f"Entry near {f}.", facts=facts, tolerance=TOL).grounded for f in figures)
        scoped = 0
        for figure in figures:
            body = decision("act", [target_json("NVDAUSDT", -0.5, thesis=f"Entry near {figure}.")])
            report = ground_decision(LlmDecision.model_validate(body), facts, TOL)["NVDAUSDT"]
            scoped += all(f.resolved for f in report.figures if f.context.startswith("thesis"))
        original = sum(argus_original_grounded(f"Entry near {f}.", facts) for f in figures)
        # Measured with 3,000 figures while building this: 76.6% original, 0.3% port, 0.1% scoped.
        assert original / len(figures) > 0.5
        assert ours / len(figures) <= 0.01
        assert scoped / len(figures) <= 0.01

    def test_no_honest_copy_of_a_fact_fails_to_resolve(
        self, snapshot_facts: dict[str, float]
    ) -> None:
        facts = snapshot_facts

        def significant(value: float, digits: int) -> str:
            if value == 0:
                return "0"
            decimals = max(0, digits - 1 - math.floor(math.log10(abs(value))))
            return f"{value:.{decimals}f}"

        failures = []
        for key, value in facts.items():
            unit = fact_unit(key)
            forms = [format_number(value), significant(value, 4), significant(value, 3)]
            if unit == "fraction":
                forms += [significant(value * 100, 3) + "%", significant(value * 1e4, 3) + " bps"]
            elif unit == "percent":
                forms.append(significant(value, 3) + "%")
            elif unit == "bps":
                forms.append(significant(value, 3) + " bps")
            for form in forms:
                report = check(f"we see {form} here", facts=facts, tolerance=TOL)
                if report.figures and not report.grounded:
                    failures.append((key, value, form))
        assert not failures, failures[:10]


# ================================================================================================
# A whole decision
# ================================================================================================


class TestGroundDecision:
    def test_one_report_per_addressed_symbol_over_the_three_fields(self) -> None:
        facts = {"NVDAUSDT.demo_last": 222.78, "book.equity": 10000.0}
        body = decision(
            "act",
            [
                target_json(
                    "NVDAUSDT",
                    -0.4,
                    thesis="Demo last 222.78 while the crowd piles in.",
                    invalidation="A close back above 240.",
                    our_view="Short with equity at 10000.",
                ),
                target_json("BTCUSDT", 0.0, thesis="No view.", our_view="Flat."),
            ],
        )
        reports = ground_decision(LlmDecision.model_validate(body), facts, TOL)
        assert set(reports) == {"NVDAUSDT", "BTCUSDT"}
        nvda = reports["NVDAUSDT"]
        assert [f.raw for f in nvda.unresolved] == ["240"]
        assert [f.context.split(":")[0] for f in nvda.figures] == [
            "thesis",
            "invalidation",
            "our_view",
        ]
        assert reports["BTCUSDT"].grounded

    def test_crowd_belief_is_not_checked(self) -> None:
        body = decision("act", [target_json("NVDAUSDT", -0.4, thesis="Crowded.")])
        body["targets"][0]["crowd_belief"] = "The crowd expects 300 by Friday."
        reports = ground_decision(LlmDecision.model_validate(body), {}, TOL)
        assert reports["NVDAUSDT"].grounded

    def test_another_instruments_fact_resolves_only_when_the_text_names_it(self) -> None:
        facts = {"BTCUSDT.funding_z_live": 2.31, "NVDAUSDT.demo_last": 222.78}
        unnamed = decision("act", [target_json("NVDAUSDT", -0.4, thesis="Funding z at 2.31.")])
        named = decision(
            "act", [target_json("NVDAUSDT", -0.4, thesis="Bitcoin funding z at 2.31.")]
        )
        by_ticker = decision(
            "act", [target_json("NVDAUSDT", -0.4, thesis="BTCUSDT.funding_z_live at 2.31.")]
        )
        assert not ground_decision(LlmDecision.model_validate(unnamed), facts, TOL)[
            "NVDAUSDT"
        ].grounded
        assert ground_decision(LlmDecision.model_validate(named), facts, TOL)["NVDAUSDT"].grounded
        assert ground_decision(LlmDecision.model_validate(by_ticker), facts, TOL)[
            "NVDAUSDT"
        ].grounded

    def test_book_level_facts_are_in_every_scope(self) -> None:
        facts = {"kernel.stop_loss_pct": 4.0, "mood.crypto_fear_greed": 22.0}
        body = decision(
            "act",
            [target_json("TSLAUSDT", 0.3, thesis="Fear & Greed at 22; inside the 4% stop.")],
        )
        assert ground_decision(LlmDecision.model_validate(body), facts, TOL)["TSLAUSDT"].grounded

    def test_the_decisions_own_numbers_are_citable(self) -> None:
        body = decision(
            "act",
            [
                target_json(
                    "NVDAUSDT",
                    -0.6,
                    confidence=0.55,
                    thesis="Target -0.6 at confidence 55%.",
                    our_view="Confidence 0.55.",
                )
            ],
        )
        report = ground_decision(LlmDecision.model_validate(body), {}, TOL)["NVDAUSDT"]
        assert report.grounded, [f.raw for f in report.unresolved]
        assert {f.source for f in report.figures} == {
            "NVDAUSDT.proposed_target",
            "NVDAUSDT.proposed_confidence",
        }

    def test_instrument_of(self) -> None:
        assert instrument_of("NVDAUSDT.funding_z_live") == "NVDAUSDT"
        assert instrument_of("SP500USDT.demo_last") == "SP500USDT"
        assert instrument_of("book.equity") is None
        assert instrument_of("trigger.1.observed") is None
        assert instrument_of("NVDAUSDT") is None


def test_render_reports_a_grounded_text_and_an_ungrounded_one() -> None:
    grounded = check("hurdle 18.8bps", facts=FACTS, tolerance=TOL)
    assert "all 1 figure(s) resolve" in render(grounded)[0]
    ungrounded = check("hurdle 25bps and 97bps", facts=FACTS, tolerance=TOL)
    assert "2 of 2 figure(s)" in render(ungrounded)[0]
