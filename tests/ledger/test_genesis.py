"""Genesis and amendments: the pre-registration, the policy-hash guard, and the X post."""

import json
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from sentiment_agent.clock import ManualClock
from sentiment_agent.hashing import ZERO_HASH, canonical_json, content_hash
from sentiment_agent.ledger.chain import HashChainLedger, LedgerError, event_hash, verify_file
from sentiment_agent.ledger.genesis import (
    X_HASHTAG,
    X_MAX_WEIGHTED_LENGTH,
    X_MENTION,
    X_QUOTED_POST,
    GenesisError,
    active_policy_hash,
    amend,
    build_genesis,
    require_genesis,
    write_genesis,
    x_post_text,
    x_weighted_length,
)
from sentiment_agent.policy import POLICY_V1
from sentiment_agent.types import (
    CONTRACT_VERSION,
    PROJECT_SLUG,
    Amendment,
    EventKind,
    Genesis,
    LedgerEvent,
    Note,
    Policy,
    RunMode,
    parse_payload,
)

COMMIT = "3f2a9c41d7b8e05f6a1c2d3e4f5a6b7c8d9e0f1a"
PROMPTS = {"src/sentiment_agent/decision/prompts/system_v1.md": content_hash("system prompt v1")}
LOCKS = {
    "uv.lock": content_hash("uv.lock contents"),
    "tools/agent-hub/package-lock.json": content_hash("package-lock contents"),
}
BGC = "@bitget-ai/bitget-agent-cli@3.0.0"


def _genesis(clock: ManualClock, mode: RunMode = RunMode.PAPER, **overrides: Any) -> Genesis:
    arguments: dict[str, Any] = {
        "policy": POLICY_V1,
        "prompt_hashes": PROMPTS,
        "mode": mode,
        "code_commit": COMMIT,
        "lock_hashes": LOCKS,
        "bgc_package": BGC,
        "clock": clock,
    }
    arguments.update(overrides)
    return build_genesis(**arguments)


def _paper(tmp_path: Path, clock: ManualClock) -> HashChainLedger:
    return HashChainLedger(
        tmp_path / "var" / "ledger" / "paper.jsonl", mode=RunMode.PAPER, clock=clock
    )


def _registered(tmp_path: Path, clock: ManualClock) -> tuple[HashChainLedger, LedgerEvent]:
    ledger = _paper(tmp_path, clock)
    return ledger, write_genesis(ledger, _genesis(clock))


def _tighter(version: str = "policy-v1.1", spread: float = 15.0) -> Policy:
    return POLICY_V1.model_copy(update={"version": version, "max_open_spread_bps": spread})


def _note(clock: ManualClock, text: str = "note") -> Note:
    return Note(at=clock.now(), author="system", text=text)


# --- building the genesis -------------------------------------------------------------------------


def test_the_genesis_pre_registers_everything_the_run_depends_on(clock: ManualClock) -> None:
    genesis = _genesis(clock)
    assert genesis.project == PROJECT_SLUG
    assert genesis.contract_version == CONTRACT_VERSION
    assert genesis.created_at == clock.now()
    assert genesis.mode is RunMode.PAPER
    assert genesis.policy == POLICY_V1
    assert genesis.policy_hash == POLICY_V1.content_hash()
    assert genesis.prompt_hashes == PROMPTS
    assert genesis.universe == POLICY_V1.symbols
    assert genesis.metric_definitions == POLICY_V1.metrics
    assert genesis.code_commit == COMMIT
    assert genesis.dependency_lock_hashes == LOCKS
    assert genesis.qwen_model == POLICY_V1.decision.model == "qwen3.8-max"
    assert genesis.bgc_package == BGC
    assert genesis.expected_envelope == POLICY_V1.expected_envelope
    assert "before its first paper order" in genesis.statement
    assert "amendment" in genesis.statement


def test_a_rehearsal_genesis_says_it_is_not_the_scored_log(clock: ManualClock) -> None:
    genesis = _genesis(clock, RunMode.SIMULATED, prompt_hashes={}, lock_hashes={})
    assert genesis.mode is RunMode.SIMULATED
    assert "not the scored log" in genesis.statement


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"code_commit": "3f2a9c4"}, "full git commit"),
        ({"code_commit": COMMIT.upper()}, "full git commit"),
        ({"code_commit": "0" * 40}, "full git commit"),
        ({"code_commit": COMMIT + "-dirty"}, "full git commit"),
        ({"bgc_package": "bgc"}, "pinned npm package"),
        ({"bgc_package": "@bitget-ai/bitget-agent-cli@latest"}, "pinned npm package"),
        ({"prompt_hashes": {"C:/Users/someone/system_v1.md": content_hash("x")}}, "relative"),
        ({"prompt_hashes": {"/etc/system_v1.md": content_hash("x")}}, "relative"),
        ({"prompt_hashes": {"prompts/../../system_v1.md": content_hash("x")}}, "relative"),
        ({"lock_hashes": {"uv.lock": "not-a-digest"}}, "SHA-256"),
        ({"prompt_hashes": {}}, "prompt files"),
        ({"lock_hashes": {}}, "lockfiles"),
        ({"prompt_hashes": {"system_v1.md": "short"}}, "does not validate"),
    ],
)
def test_what_is_not_a_pre_registration_is_refused(
    clock: ManualClock, overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(GenesisError, match=message):
        _genesis(clock, **overrides)


# --- writing it: seq 0, once ----------------------------------------------------------------------


def test_the_genesis_is_seq_zero(tmp_path: Path, clock: ManualClock) -> None:
    genesis = _genesis(clock)
    ledger = _paper(tmp_path, clock)
    event = write_genesis(ledger, genesis)
    assert (event.seq, event.kind, event.prev_hash, event.mode) == (
        0,
        EventKind.GENESIS,
        ZERO_HASH,
        RunMode.PAPER,
    )
    assert parse_payload(event) == genesis
    assert ledger.verify().intact


def test_the_genesis_is_written_only_once(tmp_path: Path, clock: ManualClock) -> None:
    ledger, _ = _registered(tmp_path, clock)
    clock.advance(timedelta(hours=1))
    with pytest.raises(GenesisError, match="only be seq 0"):
        write_genesis(ledger, _genesis(clock))
    with pytest.raises(GenesisError, match="only be seq 0"):
        ledger.append(EventKind.GENESIS, _genesis(clock))  # the chain enforces it too
    assert [e.seq for e in ledger.events()] == [0]


def test_the_genesis_is_written_only_first(tmp_path: Path, clock: ManualClock) -> None:
    ledger = _paper(tmp_path, clock)
    ledger.append(EventKind.NOTE, _note(clock, "written before any genesis"))
    with pytest.raises(GenesisError, match="only be seq 0"):
        write_genesis(ledger, _genesis(clock))
    with pytest.raises(GenesisError, match="only be seq 0"):
        ledger.append(EventKind.GENESIS, _genesis(clock))
    with pytest.raises(GenesisError, match="no genesis"):
        require_genesis(ledger, POLICY_V1)


def test_the_genesis_opens_only_its_own_mode(tmp_path: Path, clock: ManualClock) -> None:
    ledger = _paper(tmp_path, clock)
    rehearsal = _genesis(clock, RunMode.SIMULATED)
    with pytest.raises(GenesisError, match="cannot open the paper ledger"):
        write_genesis(ledger, rehearsal)
    with pytest.raises(GenesisError, match="pre-registers a simulated run"):
        ledger.append(EventKind.GENESIS, rehearsal)
    assert ledger.head() is None


# --- the policy-hash guard ------------------------------------------------------------------------


def test_require_genesis_needs_a_genesis(tmp_path: Path, clock: ManualClock) -> None:
    ledger = _paper(tmp_path, clock)
    with pytest.raises(GenesisError, match="no genesis"):
        require_genesis(ledger, POLICY_V1)
    with pytest.raises(GenesisError, match="no genesis"):
        active_policy_hash(ledger)


def test_require_genesis_passes_for_the_registered_policy(
    tmp_path: Path, clock: ManualClock
) -> None:
    ledger, event = _registered(tmp_path, clock)
    assert require_genesis(ledger, POLICY_V1) == parse_payload(event)
    assert active_policy_hash(ledger) == POLICY_V1.content_hash()


def test_require_genesis_fails_on_a_policy_hash_mismatch(
    tmp_path: Path, clock: ManualClock
) -> None:
    ledger, _ = _registered(tmp_path, clock)
    tighter = _tighter()
    with pytest.raises(GenesisError) as refused:
        require_genesis(ledger, tighter)
    message = str(refused.value)
    assert tighter.content_hash() in message
    assert POLICY_V1.content_hash() in message
    assert "set by the genesis" in message


def test_require_genesis_passes_after_a_logged_amendment(
    tmp_path: Path, clock: ManualClock
) -> None:
    ledger, _ = _registered(tmp_path, clock)
    tighter = _tighter()
    clock.advance(timedelta(hours=5))
    event = amend(
        ledger,
        new_policy=tighter,
        reason="the Demo spread bound was measured again",
        owner_confirmed=True,
        clock=clock,
    )
    amendment = parse_payload(event)
    assert isinstance(amendment, Amendment)
    assert event.seq == 1
    assert amendment.previous_policy_hash == POLICY_V1.content_hash()
    assert amendment.new_policy_hash == tighter.content_hash()
    assert amendment.at == clock.now()
    assert amendment.amendment_id.startswith("amendment-")

    assert require_genesis(ledger, tighter).policy_hash == POLICY_V1.content_hash()
    assert active_policy_hash(ledger) == tighter.content_hash()
    with pytest.raises(GenesisError, match="set by the amendment at seq 1"):
        require_genesis(ledger, POLICY_V1)  # the genesis policy is superseded
    assert ledger.verify().intact


def test_amendments_chain(tmp_path: Path, clock: ManualClock) -> None:
    ledger, _ = _registered(tmp_path, clock)
    ledger.append(EventKind.NOTE, _note(clock, "trading happens between amendments"))
    first, second = _tighter("policy-v1.1", 15.0), _tighter("policy-v1.2", 12.0)
    amend(ledger, new_policy=first, reason="tighter", owner_confirmed=True, clock=clock)
    amend(ledger, new_policy=second, reason="tighter again", owner_confirmed=True, clock=clock)
    assert active_policy_hash(ledger) == second.content_hash()
    amend(ledger, new_policy=POLICY_V1, reason="back to v1", owner_confirmed=True, clock=clock)
    assert active_policy_hash(ledger) == POLICY_V1.content_hash()
    require_genesis(ledger, POLICY_V1)
    assert [e.kind for e in ledger.events()] == [
        EventKind.GENESIS,
        EventKind.NOTE,
        EventKind.AMENDMENT,
        EventKind.AMENDMENT,
        EventKind.AMENDMENT,
    ]


def test_an_amendment_needs_the_owner_and_a_reason(tmp_path: Path, clock: ManualClock) -> None:
    ledger, _ = _registered(tmp_path, clock)
    with pytest.raises(GenesisError, match="owner's confirmation"):
        amend(ledger, new_policy=_tighter(), reason="tuning", owner_confirmed=False, clock=clock)
    with pytest.raises(GenesisError, match="say why"):
        amend(ledger, new_policy=_tighter(), reason="   ", owner_confirmed=True, clock=clock)
    with pytest.raises(GenesisError, match="unchanged"):
        amend(ledger, new_policy=POLICY_V1, reason="no-op", owner_confirmed=True, clock=clock)
    assert [e.seq for e in ledger.events()] == [0]


def test_an_amendment_needs_a_genesis(tmp_path: Path, clock: ManualClock) -> None:
    ledger = _paper(tmp_path, clock)
    with pytest.raises(GenesisError, match="no genesis"):
        amend(ledger, new_policy=_tighter(), reason="r", owner_confirmed=True, clock=clock)
    orphan = Amendment(
        amendment_id="orphan",
        at=clock.now(),
        reason="written straight to the chain",
        previous_policy_hash=POLICY_V1.content_hash(),
        new_policy_hash=_tighter().content_hash(),
        new_policy=_tighter(),
        owner_confirmed=True,
    )
    with pytest.raises(GenesisError, match="needs a genesis before it"):
        ledger.append(EventKind.AMENDMENT, orphan)


def test_the_chain_refuses_an_amendment_of_a_superseded_policy(
    tmp_path: Path, clock: ManualClock
) -> None:
    ledger, _ = _registered(tmp_path, clock)
    amend(ledger, new_policy=_tighter(), reason="first", owner_confirmed=True, clock=clock)
    stale = Amendment(
        amendment_id="stale",
        at=clock.now(),
        reason="written against the genesis policy by a second writer",
        previous_policy_hash=POLICY_V1.content_hash(),
        new_policy_hash=_tighter("policy-v1.9", 9.0).content_hash(),
        new_policy=_tighter("policy-v1.9", 9.0),
        owner_confirmed=True,
    )
    with pytest.raises(GenesisError, match="but the active policy is"):
        ledger.append(EventKind.AMENDMENT, stale)
    unconfirmed = stale.model_copy(
        update={"previous_policy_hash": _tighter().content_hash(), "owner_confirmed": False}
    )
    with pytest.raises(GenesisError, match="owner's confirmation"):
        ledger.append(EventKind.AMENDMENT, unconfirmed)
    assert [e.seq for e in ledger.events()] == [0, 1]


def test_a_forged_amendment_history_does_not_verify(tmp_path: Path, clock: ManualClock) -> None:
    ledger, genesis_event = _registered(tmp_path, clock)
    forged = Amendment(
        amendment_id="forged",
        at=clock.now(),
        reason="replaces a policy that was never in force",
        previous_policy_hash=content_hash("some other policy"),
        new_policy_hash=_tighter().content_hash(),
        new_policy=_tighter(),
        owner_confirmed=True,
    )
    fields: dict[str, Any] = {
        "seq": 1,
        "ts": genesis_event.ts,
        "kind": EventKind.AMENDMENT,
        "mode": RunMode.PAPER,
        "payload": forged.model_dump(mode="json"),
        "blobs": (),
        "prev_hash": genesis_event.hash,
    }
    line = canonical_json(LedgerEvent(**fields, hash=event_hash(**fields)))
    with ledger.path.open("ab") as handle:
        handle.write(line + b"\n")
    result = verify_file(ledger.path)
    assert result.first_break_at == 1
    assert result.break_reason is not None
    assert "the active policy is" in result.break_reason
    fresh = _paper(tmp_path, clock)
    with pytest.raises(LedgerError, match="the active policy is"):
        require_genesis(fresh, _tighter())


class _MemoryReader:
    """A LedgerReader over events held in memory, to hand require_genesis a tampered history."""

    def __init__(self, events: list[LedgerEvent]) -> None:
        self._events = events

    def events(self, kinds: frozenset[EventKind] | None = None) -> Iterator[LedgerEvent]:
        return iter([e for e in self._events if kinds is None or e.kind in kinds])

    def head(self) -> LedgerEvent | None:
        return self._events[-1] if self._events else None


def test_require_genesis_refuses_an_altered_genesis_event(
    tmp_path: Path, clock: ManualClock
) -> None:
    _, event = _registered(tmp_path, clock)
    payload = json.loads(json.dumps(event.payload))
    payload["statement"] = "a different promise"
    altered = event.model_copy(update={"payload": payload})
    with pytest.raises(GenesisError, match="does not hash"):
        require_genesis(_MemoryReader([altered]), POLICY_V1)
    require_genesis(_MemoryReader([event]), POLICY_V1)


# --- the X post -----------------------------------------------------------------------------------


def test_the_x_post_announces_the_genesis_hash(tmp_path: Path, clock: ManualClock) -> None:
    _, event = _registered(tmp_path, clock)
    text = x_post_text(event)
    assert event.hash in text
    assert X_HASHTAG == "#BitgetHackathon"
    assert X_HASHTAG in text
    assert X_MENTION == "@Bitget_AI"
    assert X_MENTION in text
    assert X_QUOTED_POST == "https://x.com/Bitget_AI/status/2100519318824055159"
    assert X_QUOTED_POST in text
    assert text.startswith(
        f"{PROJECT_SLUG}: a Market Sentiment Agent paper-trading on Bitget Demo."
    )
    assert "reduce-only risk kernel" in text
    assert x_weighted_length(text) <= X_MAX_WEIGHTED_LENGTH


def test_the_x_post_is_only_for_an_intact_paper_genesis(tmp_path: Path, clock: ManualClock) -> None:
    ledger, event = _registered(tmp_path, clock)
    note = ledger.append(EventKind.NOTE, _note(clock))
    with pytest.raises(GenesisError, match="announces the genesis event"):
        x_post_text(note)
    payload = json.loads(json.dumps(event.payload))
    payload["code_commit"] = "b" * 40
    with pytest.raises(GenesisError, match="does not hash"):
        x_post_text(event.model_copy(update={"payload": payload}))

    rehearsal_ledger = HashChainLedger(
        tmp_path / "var" / "ledger" / "simulated.jsonl", mode=RunMode.SIMULATED, clock=clock
    )
    rehearsal = write_genesis(rehearsal_ledger, _genesis(clock, RunMode.SIMULATED))
    with pytest.raises(GenesisError, match="only the paper run's genesis"):
        x_post_text(rehearsal)


def _with_project(tmp_path: Path, clock: ManualClock, project: str) -> LedgerEvent:
    genesis = _genesis(clock).model_copy(update={"project": project})
    ledger = HashChainLedger(tmp_path / f"{len(project)}.jsonl", mode=RunMode.PAPER, clock=clock)
    return write_genesis(ledger, genesis)


def test_the_x_post_shortens_itself_for_a_longer_name(tmp_path: Path, clock: ManualClock) -> None:
    event = _with_project(tmp_path, clock, "an-owner-chosen-name-that-runs-long")
    text = x_post_text(event)
    assert x_weighted_length(text) <= X_MAX_WEIGHTED_LENGTH
    assert "reduce-only risk kernel" not in text  # the longest description no longer fits
    for required in (event.hash, X_HASHTAG, X_MENTION, X_QUOTED_POST, "Market Sentiment Agent"):
        assert required in text
    with pytest.raises(GenesisError, match="leaves no room"):
        x_post_text(_with_project(tmp_path, clock, "n" * 200))


def test_x_weighted_length_counts_as_x_does() -> None:
    assert x_weighted_length("abc") == 3
    assert x_weighted_length("https://x.com/Bitget_AI/status/2100519318824055159") == 23
    assert x_weighted_length("see https://example.com/a/very/long/path/indeed ok") == 4 + 23 + 3
    assert x_weighted_length("市场") == 4
    assert x_weighted_length("\u2014") == 1  # an em dash sits in a light range
