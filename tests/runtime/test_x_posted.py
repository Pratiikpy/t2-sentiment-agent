"""``t2sa x-posted``: the owner's X post of the genesis hash is recorded in the ledger, checked
against the genesis by the time its own id encodes, and shown on the page as posted only then."""

import io
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from runtime.support import FAKE_COMMIT, make_parts
from sentiment_agent.clock import ManualClock
from sentiment_agent.ledger.chain import GenesisError
from sentiment_agent.ledger.genesis import (
    X_POST_NOTE_PREFIX,
    X_QUOTED_POST,
    recorded_x_post,
    x_post_url,
    x_posted_at,
)
from sentiment_agent.runtime.cli import EXIT_OK, EXIT_USAGE, run_cli
from sentiment_agent.runtime.wiring import open_chain
from sentiment_agent.site.export import _x_post_doc
from sentiment_agent.site.render import _x_post_block
from sentiment_agent.types import EventKind, LedgerEvent, Note, RunMode

GENESIS_AT = datetime(2026, 9, 25, 13, 20, tzinfo=UTC)
SNOWFLAKE_EPOCH_MS = 1_288_834_974_657


def status_url(at: datetime, handle: str = "owner_handle") -> str:
    """A post URL whose id encodes ``at``, built the way X builds its ids."""
    ms = int(at.timestamp() * 1000) - SNOWFLAKE_EPOCH_MS
    return f"https://x.com/{handle}/status/{(ms << 22) | 4097}"


class TestTheUrl:
    @pytest.mark.parametrize(
        "given",
        [
            "https://x.com/owner_handle/status/1900000000000000000",
            "https://twitter.com/owner_handle/status/1900000000000000000",
            "https://www.x.com/owner_handle/status/1900000000000000000/",
            "https://x.com/owner_handle/status/1900000000000000000?s=20&t=abc",
            "  https://mobile.twitter.com/owner_handle/status/1900000000000000000#m  ",
        ],
    )
    def test_every_spelling_of_a_post_is_one_canonical_url(self, given: str) -> None:
        assert x_post_url(given) == "https://x.com/owner_handle/status/1900000000000000000"

    @pytest.mark.parametrize(
        "given",
        [
            "https://x.com/owner_handle",
            "http://x.com/owner_handle/status/1900000000000000000",
            "https://x.com.evil.example/owner_handle/status/1",
            "https://x.com/owner_handle/status/abc",
            "not a url",
        ],
    )
    def test_anything_else_is_refused(self, given: str) -> None:
        with pytest.raises(GenesisError, match="not the URL of a post on X"):
            x_post_url(given)

    def test_the_time_is_read_from_the_id(self) -> None:
        assert x_posted_at(X_QUOTED_POST) == datetime(2026, 9, 17, 9, 36, 46, 26000, tzinfo=UTC)
        at = datetime(2026, 9, 28, 0, 12, 30, tzinfo=UTC)
        assert x_posted_at(status_url(at)) == at


def _genesis(tmp_path: Path) -> tuple[Path, ManualClock]:
    root = tmp_path / "rehearsal"
    git = root / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n", "utf-8")
    (git / "refs" / "heads" / "main").write_text(FAKE_COMMIT + "\n", "utf-8")
    clock = ManualClock(GENESIS_AT)
    code, text = _cli(root, clock, "genesis", "--mode", "simulated", "--no-anchor")
    assert code == EXIT_OK, text
    return root, clock


def _cli(root: Path, clock: ManualClock, *argv: str) -> tuple[int, str]:
    out = io.StringIO()
    parts = make_parts(clock, oi_thresholds={"BTCUSDT": 5.0})
    code = run_cli(list(argv), root=root, clock=clock, parts=parts, out=out)
    return code, out.getvalue()


class TestTheCommand:
    def test_a_post_is_recorded_once_as_an_owner_note(self, tmp_path: Path) -> None:
        root, clock = _genesis(tmp_path)
        clock.advance(timedelta(minutes=20))
        url = status_url(GENESIS_AT + timedelta(minutes=12))
        code, text = _cli(root, clock, "x-posted", "--mode", "simulated", "--url", url + "?s=20")
        assert code == EXIT_OK, text
        assert "recorded at seq" in text
        code, text = _cli(root, clock, "x-posted", "--mode", "simulated", "--url", url)
        assert code == EXIT_OK
        assert "already recorded" in text
        chain = open_chain(root, RunMode.SIMULATED, clock)
        notes = [Note.model_validate(e.payload) for e in chain.events(frozenset({EventKind.NOTE}))]
        owner = [n for n in notes if n.text.startswith(X_POST_NOTE_PREFIX)]
        assert [(n.author, n.text) for n in owner] == [("owner", X_POST_NOTE_PREFIX + url)]
        recorded = recorded_x_post(chain.events())
        assert recorded is not None
        assert recorded[0] == url

    def test_a_post_older_than_the_genesis_is_the_wrong_post(self, tmp_path: Path) -> None:
        root, clock = _genesis(tmp_path)
        url = status_url(GENESIS_AT - timedelta(hours=1))
        code, text = _cli(root, clock, "x-posted", "--mode", "simulated", "--url", url)
        assert code == EXIT_USAGE
        assert "before the genesis" in text

    def test_a_ledger_without_a_genesis_has_nothing_to_record(self, tmp_path: Path) -> None:
        clock = ManualClock(GENESIS_AT)
        root = tmp_path / "empty"
        root.mkdir()
        url = status_url(GENESIS_AT)
        code, text = _cli(root, clock, "x-posted", "--mode", "simulated", "--url", url)
        assert code == EXIT_USAGE
        assert "no genesis" in text

    def test_a_url_that_is_not_a_post_is_refused_before_anything_is_read(
        self, tmp_path: Path
    ) -> None:
        clock = ManualClock(GENESIS_AT)
        code, text = _cli(tmp_path, clock, "x-posted", "--url", "https://x.com/owner_handle")
        assert code == EXIT_USAGE
        assert "not the URL of a post on X" in text


class TestThePage:
    TEXT = "t2-sentiment-agent: a Market Sentiment Agent on Bitget Demo."

    def _posted(self, **order: object) -> dict[str, object]:
        return {
            "url": "https://x.com/owner_handle/status/1",
            "posted_at": "2026-09-28T00:12:30Z",
            "recorded_seq": 7,
            "recorded_at": "2026-09-28T00:20:00Z",
            **order,
        }

    def test_an_unrecorded_post_is_a_draft(self) -> None:
        html = _x_post_block(self.TEXT, None)
        assert "Drafted for X" in html
        assert "Posted on X" not in html

    def test_a_post_before_any_order_says_so(self) -> None:
        html = _x_post_block(self.TEXT, self._posted(first_order_at=None, before_first_order=None))
        assert "Posted on X 2026-09-28 00:12 UTC; no order has been sent yet" in html
        assert "seq 7" in html

    def test_a_post_after_the_first_order_is_not_called_before_it(self) -> None:
        html = _x_post_block(
            self.TEXT,
            self._posted(first_order_at="2026-09-28T00:05:00Z", before_first_order=False),
        )
        assert "after the first order (2026-09-28 00:05 UTC)" in html
        assert "before the first order" not in html


def _event(seq: int, kind: EventKind, ts: datetime, payload: dict[str, object]) -> LedgerEvent:
    return LedgerEvent.model_construct(
        seq=seq,
        ts=ts,
        kind=kind,
        mode=RunMode.PAPER,
        payload=payload,
        blobs=(),
        prev_hash="0" * 64,
        hash=f"{seq:064x}",
    )


class TestTheExport:
    POSTED = GENESIS_AT + timedelta(minutes=12)

    def _events(self, first_order: datetime | None) -> list[LedgerEvent]:
        text = X_POST_NOTE_PREFIX + status_url(self.POSTED)
        events = [
            _event(0, EventKind.GENESIS, GENESIS_AT, {}),
            _event(1, EventKind.NOTE, GENESIS_AT, {"author": "system", "text": "started"}),
            _event(
                2,
                EventKind.NOTE,
                GENESIS_AT + timedelta(minutes=30),
                {"author": "owner", "text": text},
            ),
        ]
        if first_order is not None:
            events.append(_event(3, EventKind.ORDER_SUBMITTED, first_order, {}))
        return events

    def test_an_unrecorded_post_publishes_nothing(self) -> None:
        assert _x_post_doc(self._events(None)[:2]) is None

    def test_a_post_made_before_the_first_order_says_so(self) -> None:
        doc = _x_post_doc(self._events(GENESIS_AT + timedelta(hours=1)))
        assert doc is not None
        assert doc["url"] == status_url(self.POSTED)
        assert doc["posted_at"] == "2026-09-25T13:32:00Z"
        assert doc["recorded_seq"] == 2
        assert doc["before_first_order"] is True

    def test_a_post_made_after_the_first_order_says_that_instead(self) -> None:
        doc = _x_post_doc(self._events(GENESIS_AT + timedelta(minutes=5)))
        assert doc is not None
        assert doc["before_first_order"] is False

    def test_no_order_yet_is_neither(self) -> None:
        doc = _x_post_doc(self._events(None))
        assert doc is not None
        assert doc["first_order_at"] is None
        assert doc["before_first_order"] is None

    def test_a_system_note_that_looks_like_a_post_is_not_one(self) -> None:
        text = X_POST_NOTE_PREFIX + status_url(self.POSTED)
        forged = [_event(0, EventKind.NOTE, GENESIS_AT, {"author": "system", "text": text})]
        assert _x_post_doc(forged) is None
