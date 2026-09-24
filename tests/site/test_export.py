"""The public export: the full file set, recomputable by a stranger, and nothing secret or local in
it, ever."""

import json
import os
import re
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from sentiment_agent.ledger.chain import verify_file
from sentiment_agent.site.cards import build_cards
from sentiment_agent.site.export import (
    EQUITY_COLUMNS,
    TRADE_COLUMNS,
    ExportError,
    ExportRefused,
    export_public,
    local_literals,
    replay_venue_integrity,
    scan_bytes,
)
from sentiment_agent.types import (
    ArmKind,
    ArmResult,
    DecisionCard,
    EventKind,
    GuardStatus,
    MetricSet,
    Note,
)
from site_world import (
    RECOMPUTE,
    Copy,
    Site,
    analysis_arms,
    copy_world,
    export_world,
    minimal_ledger,
)

REQUIRED = (
    "ledger.jsonl",
    "ledger.jsonl.head",
    "genesis.json",
    "environment.json",
    "decisions.json",
    "cards/index.json",
    "orders.json",
    "funnel.json",
    "equity_hourly.csv",
    "trades.csv",
    "metrics.json",
    "arms.json",
    "arms_summary.json",
    "twin.json",
    "mirror.json",
    "redteam.json",
    "toolkit.json",
    "replay.json",
    "summary.json",
    "verify.md",
)

DRIVE_PATH = re.compile(
    rb"(?<![A-Za-z0-9])[A-Za-z]:(?:\\\\|\\|/)[^\\/\s\"'<>|:*?]{1,255}(?:\\\\|\\|/)"
)
POSIX_HOME = re.compile(rb"(?<![A-Za-z0-9_.-])/(?:Users|home)/[A-Za-z0-9._-]+/")
KEY_SHAPES = (
    re.compile(rb"(?<![A-Za-z0-9])bg_[0-9a-f]{32}"),
    re.compile(rb"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(rb"(?i)ACCESS-(?:KEY|SIGN|PASSPHRASE)"),
)


def load(site: Site, name: str) -> object:
    return json.loads((site.public / name).read_text("utf-8"))


def load_list(site: Site, name: str) -> list[Any]:
    loaded = load(site, name)
    assert isinstance(loaded, list)
    return loaded


def head_hash(site: Site) -> str:
    head = site.world.ledger.head()
    assert head is not None
    return head.hash


def files_under(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def recompute(public: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - the project's own script, run as a judge would
        [sys.executable, str(RECOMPUTE), str(public)],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )


# ------------------------------------------------------------------------------------------------
# The file set
# ------------------------------------------------------------------------------------------------


def test_the_full_file_set_is_written(site: Site) -> None:
    written = set(site.manifest.written)
    for name in REQUIRED:
        assert name in written, name
        assert (site.public / name).is_file(), name
    cards = build_cards(site.world.projection(), site.world.blobs)
    for card in cards:
        assert f"cards/{card.card_id}.json" in written
    index = load(site, "cards/index.json")
    assert isinstance(index, list)
    assert [e["card_id"] for e in index] == [c.card_id for c in cards]
    assert site.manifest.ledger_head == head_hash(site)
    assert site.manifest.generated_at == site.world.clock.now()
    assert all("\\" not in name for name in written)


def test_every_referenced_blob_is_published(site: Site) -> None:
    verification = verify_file(site.public / "ledger.jsonl", blobs_root=site.public / "blobs")
    assert verification.intact
    assert verification.missing_blobs == ()
    published = {p.name for p in (site.public / "blobs").iterdir()}
    assert {n.removeprefix("blobs/") for n in site.manifest.written if n.startswith("blobs/")} == (
        published
    )


def test_the_published_ledger_is_the_ledger_byte_for_byte(site: Site) -> None:
    assert (site.public / "ledger.jsonl").read_bytes() == site.world.ledger.path.read_bytes()
    head = json.loads((site.public / "ledger.jsonl.head").read_text("utf-8"))
    last = site.world.ledger.head()
    assert last is not None
    assert head == {
        "events": last.seq + 1,
        "head_seq": last.seq,
        "head_hash": last.hash,
        "mode": "simulated",
    }


def test_recompute_passes_on_the_export(site: Site) -> None:
    result = recompute(site.public)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "all checks agree" in result.stdout
    for check in ("ledger chain", "blobs", "equity_hourly.csv", "trades.csv", "metrics.json"):
        assert re.search(rf"^{re.escape(check)}\s+ok\b", result.stdout, re.M), check


def test_recompute_catches_an_edited_number(site: Site, tmp_path: Path) -> None:
    public = tmp_path / "public"
    for path in files_under(site.public):
        target = public / path.relative_to(site.public)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    metrics = json.loads((public / "metrics.json").read_text("utf-8"))
    metrics["max_drawdown"] = metrics["max_drawdown"] / 2
    (public / "metrics.json").write_text(json.dumps(metrics), "utf-8")
    result = recompute(public)
    assert result.returncode == 1
    assert "MISMATCH" in result.stdout


def test_csv_files_are_the_ledger_marks_and_the_book_trades(site: Site) -> None:
    lines = (site.public / "equity_hourly.csv").read_text("utf-8").splitlines()
    assert tuple(lines[0].split(",")) == EQUITY_COLUMNS
    marks = site.world.events(EventKind.MARK)
    assert len(lines) - 1 == len(marks) == 6  # the 10:00 anchor and five hourly marks
    for line, event in zip(lines[1:], marks, strict=True):
        seq, at, equity = line.split(",")[:3]
        assert int(seq) == event.seq
        assert at == event.payload["at"]
        assert equity == event.payload["equity_book"]
    trades = (site.public / "trades.csv").read_text("utf-8").splitlines()
    assert tuple(trades[0].split(",")) == TRADE_COLUMNS
    closed = site.world.projection().closed_trades
    assert len(trades) - 1 == len(closed) == 2
    reasons = [row.split(",")[10] for row in trades[1:]]
    assert reasons == [t.exit_reason for t in closed]


def test_metrics_and_arms_are_the_ledgers_book_first(site: Site) -> None:
    metrics = MetricSet.model_validate(load(site, "metrics.json"))
    assert metrics.arm_id == "ours_governed"
    assert metrics.n_closed_trades == 2
    assert metrics.n_hours == 5
    arms = [ArmResult.model_validate(a) for a in load_list(site, "arms.json")]
    assert arms[0].spec.arm_id == "ours_governed"
    assert arms[0].metrics == metrics
    assert {a.spec.arm_id for a in arms[1:]} == {a.spec.arm_id for a in analysis_arms(site.world)}
    summary = load(site, "arms_summary.json")
    assert isinstance(summary, dict)
    assert summary["coin_flip"]["seeds"] == 6
    assert summary["families"]["baseline_coin_flip"] == 6
    # No replica among these arms, so the book itself is ranked, and the record says which.
    assert summary["coin_flip"]["ranked_arm_id"] == "ours_governed"
    assert summary["replica_vs_live"] is None


def test_decisions_json_counts_every_kind_of_decision(site: Site) -> None:
    doc = load(site, "decisions.json")
    assert isinstance(doc, dict)
    counts = doc["counts"]
    assert counts["decisions"] == 4
    assert counts["act"] == 2
    assert counts["flat_with_reasons"] == 1
    assert counts["outcomes"] == {"decided": 3, "transport_error": 1}
    assert counts["changed_by_kernel"] == 1
    assert counts["owner_manual_triggers"] == 1
    first = doc["decisions"][0]
    targets = {t["symbol"]: t for t in first["targets"]}
    assert targets["NVDAUSDT"]["proposed_weight"] == pytest.approx(-0.03)
    assert targets["NVDAUSDT"]["approved_weight"] == pytest.approx(-0.03)
    second = doc["decisions"][1]
    assert second["targets"][0]["binding_guard"] == "G6_turnover"
    assert all(d["card_id"] for d in doc["decisions"])


def test_orders_json_is_what_verify_orders_reads(site: Site) -> None:
    doc = load(site, "orders.json")
    assert isinstance(doc, dict)
    orders = doc["orders"]
    assert doc["venue"] == "simulated"
    assert "not PAPER" in doc["note"]
    assert len(orders) == 3
    for order in orders:
        for field in ("venue_order_id", "client_oid", "symbol", "side"):
            assert order[field], field
        assert order["state"] == "filled"
        assert order["fills"]
        assert order["preview_argv"][-1] == "--dry-run"
    stops = doc["venue_originated_fills"]
    assert [s["fill"]["exec_id"] for s in stops] == site.world.stop_fill_ids
    assert doc["counts"]["sent"] == 3


def test_funnel_shows_what_the_kernel_did(site: Site) -> None:
    funnel = load(site, "funnel.json")
    assert isinstance(funnel, dict)
    assert funnel["legs_proposed"] == 3
    assert funnel["legs_adding_exposure"] == 3
    assert funnel["approved_in_full"] == 2
    assert funnel["cut"] == 1
    assert funnel["refused"] == 0
    assert funnel["guards"]["G6_turnover"]["binding"] == 1
    assert funnel["guards"]["G6_turnover"]["fired"] == 1
    assert funnel["orders"]["filled"] == 2
    assert funnel["protective"]["by_reason"] == {"llm_outage": 1}


def test_the_replay_is_labelled_and_the_guard_fires_on_the_recorded_print(site: Site) -> None:
    replay = load(site, "replay.json")
    assert isinstance(replay, dict)
    assert replay["label"].startswith("REPLAY")
    assert replay["hour"] == "2026-09-23T13:00:00Z"
    assert replay["inputs"]["mark_index_gap_pct"] > 5.3
    assert replay["inputs"]["mark_index_gap_pct_against_index_high"] > 3.0
    for item in replay["rulings"]:
        ruling = item["ruling"]
        assert ruling["guard"] == "G1_venue_integrity"
        assert ruling["status"] == GuardStatus.FIRED.value
        assert ruling["forces_exit"] is True
    assert replay_venue_integrity() == replay


def test_reports_are_published_or_marked_not_computed(site: Site, tmp_path: Path) -> None:
    twin = load(site, "twin.json")
    assert isinstance(twin, dict)
    assert twin["status"] == "computed"
    red = load(site, "redteam.json")
    assert isinstance(red, dict)
    assert red["report"]["hijack_rate"] == {"ours": 0.0, "keyword": 1.0}
    bare = minimal_ledger(tmp_path)
    export_public(
        ledger=bare.ledger,
        blobs=bare.blobs,
        out=tmp_path / "public",
        arms=(),
        twin=None,
        redteam=None,
        toolkit=(),
        clock=bare.clock,
    )
    doc = json.loads((tmp_path / "public" / "twin.json").read_text("utf-8"))
    assert doc["status"] == "not_computed"


def test_an_empty_book_exports_and_recomputes(tmp_path: Path) -> None:
    bare = minimal_ledger(tmp_path, "the owner started the agent")
    manifest = export_public(
        ledger=bare.ledger,
        blobs=bare.blobs,
        out=tmp_path / "public",
        arms=(),
        twin=None,
        redteam=None,
        toolkit=(),
        clock=bare.clock,
    )
    assert "ledger.jsonl" in manifest.written
    toolkit = json.loads((tmp_path / "public" / "toolkit.json").read_text("utf-8"))
    assert toolkit["rows"], "the declared matrix stands in when none is passed"
    result = recompute(tmp_path / "public")
    assert result.returncode == 0, result.stdout + result.stderr


def test_exporting_twice_is_byte_identical(site: Site, tmp_path: Path) -> None:
    again = export_world(site.world, tmp_path / "public")
    assert again.written == site.manifest.written
    for name in again.written:
        assert (tmp_path / "public" / name).read_bytes() == (site.public / name).read_bytes(), name


# ------------------------------------------------------------------------------------------------
# No secret, no local path
# ------------------------------------------------------------------------------------------------


def test_no_secret_and_no_local_path_anywhere_in_the_output(site: Site) -> None:
    literals = local_literals([site.public, site.world.root])
    home = str(Path.home()).encode()
    for path in files_under(site.public):
        data = path.read_bytes()
        name = path.relative_to(site.public).as_posix()
        assert b"BITGET_" not in data, name
        for pattern in KEY_SHAPES:
            assert not pattern.search(data), (name, pattern.pattern)
        assert not DRIVE_PATH.search(data), name
        assert not POSIX_HOME.search(data), name
        assert home not in data
        assert home.replace(b"\\", b"\\\\") not in data
        assert str(site.world.root).encode() not in data
        assert scan_bytes(name, data, literals) == [], name


PLANTED = {
    "a credential variable": "loaded BITGET_API_KEY from the environment",
    "a Bitget key shape": "key bg_" + "0123456789abcdef" * 2 + " in a log line",
    "an sk- key": "Authorization used sk-" + "A1b2C3d4" * 4,
    "a Windows path": "could not open C:\\Users\\someone\\secrets\\demo.env",
    "a home path": "traceback in /home/someone/project/agent.py",
    "a key assignment": 'config {"api_key": "Zx81Qm0pLkT"}',
}


@pytest.mark.parametrize("text", list(PLANTED.values()), ids=list(PLANTED))
def test_a_planted_secret_or_path_is_refused_and_nothing_is_published(
    tmp_path: Path, text: str
) -> None:
    bare = minimal_ledger(tmp_path, text)
    out = tmp_path / "public"
    out.mkdir()
    (out / "CNAME").write_text("kept\n", "utf-8")
    with pytest.raises(ExportRefused) as refused:
        export_public(
            ledger=bare.ledger,
            blobs=bare.blobs,
            out=out,
            arms=(),
            twin=None,
            redteam=None,
            toolkit=(),
            clock=bare.clock,
        )
    assert refused.value.findings
    assert any(f.path == "ledger.jsonl" for f in refused.value.findings)
    assert text not in str(refused.value), "a finding names the rule, never the matched text"
    assert sorted(p.name for p in out.iterdir()) == ["CNAME"]
    assert not [p for p in tmp_path.iterdir() if ".staging-" in p.name]


def test_the_value_of_a_secret_environment_variable_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = "q7Rz0pWn4Lk2"
    monkeypatch.setenv("T2SA_TEST_API_TOKEN", value)
    bare = minimal_ledger(tmp_path, f"the gateway answered for {value}")
    with pytest.raises(ExportRefused) as refused:
        export_public(
            ledger=bare.ledger,
            blobs=bare.blobs,
            out=tmp_path / "public",
            arms=(),
            twin=None,
            redteam=None,
            toolkit=(),
            clock=bare.clock,
        )
    assert any("T2SA_TEST_API_TOKEN" in f.rule for f in refused.value.findings)
    assert value not in str(refused.value)


def test_a_local_path_inside_a_blob_is_refused(tmp_path: Path) -> None:
    bare = minimal_ledger(tmp_path)
    ref = bare.blobs.put(
        json.dumps({"error": str(tmp_path / "var" / "agent.log")}).encode(), "application/json"
    )
    bare.clock.advance(timedelta(minutes=1))
    bare.ledger.append(
        EventKind.NOTE, Note(at=bare.clock.now(), author="system", text="see blob"), blobs=[ref]
    )
    with pytest.raises(ExportRefused) as refused:
        export_public(
            ledger=bare.ledger,
            blobs=bare.blobs,
            out=tmp_path / "public",
            arms=(),
            twin=None,
            redteam=None,
            toolkit=(),
            clock=bare.clock,
        )
    assert any(f.path == f"blobs/{ref.sha256}" for f in refused.value.findings)


# ------------------------------------------------------------------------------------------------
# What else the export refuses
# ------------------------------------------------------------------------------------------------


def run_export(copy: Copy, out: Path) -> None:
    export_public(
        ledger=copy.ledger,
        blobs=copy.blobs,
        out=out,
        arms=(),
        twin=None,
        redteam=None,
        toolkit=(),
        clock=copy.clock,
    )


def test_a_tampered_blob_refuses_the_export(site: Site, tmp_path: Path) -> None:
    copy = copy_world(site.world, tmp_path)
    victim = next(p for p in (tmp_path / "var" / "blobs").iterdir() if p.is_file())
    victim.write_bytes(victim.read_bytes() + b" ")
    with pytest.raises(ExportError, match="does not verify"):
        run_export(copy, tmp_path / "public")
    assert not (tmp_path / "public" / "ledger.jsonl").exists()


def test_an_edited_ledger_line_refuses_the_export(site: Site, tmp_path: Path) -> None:
    copy = copy_world(site.world, tmp_path)
    path = copy.ledger.path
    lines = path.read_bytes().split(b"\n")
    target = next(i for i, line in enumerate(lines) if b'"10000"' in line)
    edited = lines[target].replace(b'"10000"', b'"10001"', 1)
    assert edited != lines[target]
    lines[target] = edited
    path.write_bytes(b"\n".join(lines))
    with pytest.raises(ExportError):
        run_export(copy, tmp_path / "public")
    assert not (tmp_path / "public" / "ledger.jsonl").exists()


def test_an_arm_passed_as_the_book_must_be_the_ledgers_book(site: Site, tmp_path: Path) -> None:
    book = ArmResult.model_validate(load_list(site, "arms.json")[0])
    forged = book.model_copy(
        update={"metrics": book.metrics.model_copy(update={"total_return": 0.25})}
    )
    copy = copy_world(site.world, tmp_path)
    with pytest.raises(ExportError, match="ours_governed"):
        export_public(
            ledger=copy.ledger,
            blobs=copy.blobs,
            out=tmp_path / "public",
            arms=(forged,),
            twin=None,
            redteam=None,
            toolkit=(),
            clock=copy.clock,
        )
    same = export_public(
        ledger=copy.ledger,
        blobs=copy.blobs,
        out=tmp_path / "public",
        arms=(book,),
        twin=None,
        redteam=None,
        toolkit=(),
        clock=copy.clock,
    )
    arms = json.loads((tmp_path / "public" / "arms.json").read_text("utf-8"))
    assert [a["spec"]["arm_id"] for a in arms] == ["ours_governed"]
    assert "arms.json" in same.written


def test_arms_passed_twice_are_refused(site: Site, tmp_path: Path) -> None:
    arm = analysis_arms(site.world)[0]
    copy = copy_world(site.world, tmp_path)
    with pytest.raises(ExportError, match="twice"):
        export_public(
            ledger=copy.ledger,
            blobs=copy.blobs,
            out=tmp_path / "public",
            arms=(arm, arm),
            twin=None,
            redteam=None,
            toolkit=(),
            clock=copy.clock,
        )


def test_the_working_store_is_never_the_output(site: Site, tmp_path: Path) -> None:
    copy = copy_world(site.world, tmp_path)
    with pytest.raises(ExportError, match="blob store"):
        run_export(copy, tmp_path / "var")
    with pytest.raises(ExportError, match=r"working ledger's folder"):
        run_export(copy, tmp_path / "var" / "ledger")


def test_stale_files_are_pruned_and_unrelated_files_kept(site: Site, tmp_path: Path) -> None:
    copy = copy_world(site.world, tmp_path)
    out = tmp_path / "public"
    (out / "cards").mkdir(parents=True)
    (out / "blobs").mkdir()
    (out / "cards" / "dec-gone.json").write_text("{}", "utf-8")
    (out / "blobs" / ("f" * 64)).write_bytes(b"stale")
    (out / "CNAME").write_text("record.example\n", "utf-8")
    run_export(copy, out)
    assert not (out / "cards" / "dec-gone.json").exists()
    assert not (out / "blobs" / ("f" * 64)).exists()
    assert (out / "CNAME").is_file()


def test_a_simulated_book_without_an_account_read_still_exports(tmp_path: Path) -> None:
    """The unit placeholder lets the fold run; it never reaches a published number."""
    bare = minimal_ledger(tmp_path)
    run_export(bare, tmp_path / "public")
    summary = json.loads((tmp_path / "public" / "summary.json").read_text("utf-8"))
    assert summary["mode"] == "simulated"
    assert summary["scored"] is False
    assert summary["metrics"]["n_hours"] == 0


def test_every_card_json_is_a_decision_card(site: Site) -> None:
    index = load(site, "cards/index.json")
    assert isinstance(index, list)
    for entry in index:
        card = DecisionCard.model_validate(load(site, f"cards/{entry['card_id']}.json"))
        assert card.card_id == entry["card_id"]
        assert entry["kind"] == ("decision" if card.decision_id else "protective")


def test_mirror_series_is_every_mark(site: Site) -> None:
    mirror = load(site, "mirror.json")
    assert isinstance(mirror, dict)
    assert len(mirror["series"]) == 6
    assert [a["spec"]["kind"] for a in mirror["arms"]] == [ArmKind.MIRROR_LIVE.value]
    held = [row for row in mirror["series"] if row["positions"]]
    assert held
    assert all(row["equity_live_mirror"] is not None for row in held)


def test_verify_md_names_the_commands_and_no_local_path(site: Site) -> None:
    text = (site.public / "verify.md").read_text("utf-8")
    assert "python scripts/recompute.py public/" in text
    assert "scripts/verify_orders.py" in text
    assert "simulated" in text
    assert head_hash(site) in text
    assert os.sep + "Users" + os.sep not in text


# Crowd text reaches the ledger as JSON. These are real shapes from the 2026-09-24 live dry run
# (the first one refused that export) and their neighbours: none is a path.
NOT_PATHS = [
    r""""text":"today's game isn't it. It's:\n\nDefensive certainty + catalysts\n" """,
    '"text":"Here\u2019s:\\n\\nthe plan\\n"',  # a curly apostrophe written as itself
    r""""text":"Here\u2019s:\n\nthe plan\n" """,  # the same, written as a JSON escape
    r""""text":"Q: why now?\n A:\nBecause funding\n" """,
    r""""text":"ratio 3:\t1 then 4:\r\n" """,
    r""""text":"he said:\"a:\"b\"" """,
]

PATHS = [
    r""""detail":"C:\\Users\\someone\\demo.env" """,  # a Windows path as JSON writes it
    r"could not open C:\Users\someone\secrets\demo.env",  # as a plain log line writes it
    r""""p":"D:/data/ledger/paper.jsonl" """,
    r"(E:\Build\out\x)",
]


@pytest.mark.parametrize("text", NOT_PATHS)
def test_crowd_text_with_colons_and_escapes_is_not_a_drive_path(text: str) -> None:
    assert scan_bytes("ledger.jsonl", text.encode("utf-8"), ()) == []


@pytest.mark.parametrize("text", PATHS)
def test_a_real_drive_path_is_still_found(text: str) -> None:
    findings = scan_bytes("ledger.jsonl", text.encode("utf-8"), ())
    assert [f.rule for f in findings] == ["a Windows drive path"]
