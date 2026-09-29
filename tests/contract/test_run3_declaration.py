"""``run3-d1``: the one change this branch declares, in the shape run 2's declarations use.

Run 3 has no genesis yet (run 2 is still the live run this was branched from), so there is no
``PredecessorRun`` or policy chain to check here the way ``test_run2_declaration.py`` checks run
2's. What is checked is that the declaration itself is well-formed, cites real files, and states
the same bound the backoff schedule actually produces (``execution/exit_backoff.py``,
``tests/execution/test_exit_backoff.py``).
"""

from datetime import UTC, datetime
from pathlib import Path

from sentiment_agent.execution.exit_backoff import next_retry_at
from sentiment_agent.run3 import DECLARED_CHANGES, RUN3_D1
from sentiment_agent.types import DeclaredChange

ROOT = Path(__file__).resolve().parents[2]


def test_run3_d1_is_a_code_fix_with_no_policy_hashes() -> None:
    assert RUN3_D1.change_id == "run3-d1"
    assert RUN3_D1.kind == "code_fix"
    assert RUN3_D1.previous_policy_hash is None
    assert RUN3_D1.new_policy_hash is None
    assert DECLARED_CHANGES == (RUN3_D1,)


def test_run3_d1_names_files_that_actually_exist() -> None:
    assert RUN3_D1.files, "a code fix names what it touched"
    for relative in RUN3_D1.files:
        assert (ROOT / relative).is_file(), f"{relative} does not exist"


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
