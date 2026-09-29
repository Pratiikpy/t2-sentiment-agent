"""``run3-d1`` and ``run3-d2``: what this branch declares, in the shape run 2's declarations use.

Run 3 has no genesis yet (run 2 is still the live run this was branched from), so there is no
``PredecessorRun`` or policy chain to check here the way ``test_run2_declaration.py`` checks run
2's. What is checked is that each declaration is well-formed, cites real files, and states the
same figures the code it declares actually produces (``execution/exit_backoff.py``,
``decision/majority.py`` and their own test files).
"""

from datetime import UTC, datetime
from pathlib import Path

from sentiment_agent.execution.exit_backoff import next_retry_at
from sentiment_agent.run3 import DECLARED_CHANGES, RUN3_D1, RUN3_D2
from sentiment_agent.types import DeclaredChange

ROOT = Path(__file__).resolve().parents[2]


def test_run3_d1_is_a_code_fix_with_no_policy_hashes() -> None:
    assert RUN3_D1.change_id == "run3-d1"
    assert RUN3_D1.kind == "code_fix"
    assert RUN3_D1.previous_policy_hash is None
    assert RUN3_D1.new_policy_hash is None


def test_run3_d2_is_a_code_fix_with_no_policy_hashes() -> None:
    assert RUN3_D2.change_id == "run3-d2"
    assert RUN3_D2.kind == "code_fix"
    assert RUN3_D2.previous_policy_hash is None
    assert RUN3_D2.new_policy_hash is None


def test_declared_changes_is_both_in_order() -> None:
    assert DECLARED_CHANGES == (RUN3_D1, RUN3_D2)
    assert len({c.change_id for c in DECLARED_CHANGES}) == 2


def test_run3_d1_names_files_that_actually_exist() -> None:
    assert RUN3_D1.files, "a code fix names what it touched"
    for relative in RUN3_D1.files:
        assert (ROOT / relative).is_file(), f"{relative} does not exist"


def test_run3_d2_names_files_that_actually_exist() -> None:
    assert RUN3_D2.files, "a code fix names what it touched"
    for relative in RUN3_D2.files:
        assert (ROOT / relative).is_file(), f"{relative} does not exist"


def test_run3_d2_cites_the_g7_evidence_it_fixes() -> None:
    text = RUN3_D2.title + " " + RUN3_D2.detail
    for fact in ("TSLAUSDT", "COINUSDT", "0.27", "run2-a6", "DISAGREED"):
        assert fact in text
    for fact in ("MSTRUSDT", "METAUSDT", "TSLAUSDT", "COINUSDT", "dec-189d8234"):
        assert fact in RUN3_D2.evidence["act_cycle_variance"]
    assert "argus/data/t2_llm_variance.json" in RUN3_D2.evidence["source"]
    assert "row G7" in RUN3_D2.evidence["source"]


def test_run3_d2_round_trips_as_a_declared_change() -> None:
    again = DeclaredChange.model_validate(RUN3_D2.model_dump(mode="json"))
    assert again == RUN3_D2


def test_run3_d1_cites_the_run2_record_it_fixes() -> None:
    text = RUN3_D1.title + " " + RUN3_D1.detail
    for fact in ("METAUSDT", "MSTRUSDT", "2026-09-28", "does not exist"):
        assert fact in text
    assert "74" in RUN3_D1.detail
    assert "2026-09-28T09:16:43Z" in RUN3_D1.evidence["run2_rejections"]
    assert "15:11:33Z" in RUN3_D1.evidence["run2_rejections"]
    assert "public/orders.json" in RUN3_D1.evidence["source"]


def test_run3_d1s_bounded_attempts_claim_matches_the_schedule() -> None:
    """The evidence says under 20 real attempts over run 2's own ~5h55m episode; recompute it
    from the schedule ``execution/exit_backoff.py`` actually runs, so the two can never drift
    apart silently."""
    started = datetime(2026, 9, 28, 9, 16, 43, tzinfo=UTC)
    ended = datetime(2026, 9, 28, 15, 11, 33, tzinfo=UTC)
    attempt_count = 1
    at = started
    while at < ended:
        at = next_retry_at(attempt_count, at)
        attempt_count += 1
    assert attempt_count < 20
    assert "under 20" in RUN3_D1.evidence["bounded_attempts"]


def test_run3_d1_round_trips_as_a_declared_change() -> None:
    again = DeclaredChange.model_validate(RUN3_D1.model_dump(mode="json"))
    assert again == RUN3_D1
