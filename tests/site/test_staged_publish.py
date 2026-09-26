"""Moving a record into ``public/`` (run 1's export failure, 2026-09-26): an unchanged file is not
moved, the files go up in an order a mid-publish copy can detect, a file another process holds open
is retried rather than abandoned, and an unchanged file is still scanned where it lies."""

import os
from pathlib import Path

import pytest

from sentiment_agent.site import export
from sentiment_agent.site.export import ExportRefused, StagedWrite, replace_patiently
from site_world import Site, export_world


def _recording(monkeypatch: pytest.MonkeyPatch, out: Path) -> list[str]:
    moved: list[str] = []
    real = export.replace_patiently

    def record(source: Path, target: Path, **kwargs: object) -> None:
        moved.append(target.relative_to(out).as_posix())
        real(source, target)

    monkeypatch.setattr(export, "replace_patiently", record)
    return moved


class TestWhatMoves:
    def test_a_second_export_of_the_same_record_moves_nothing(
        self, site: Site, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = tmp_path / "public"
        first = export_world(site.world, out)
        blobs = [n for n in first.written if n.startswith("blobs/")]
        assert blobs
        moved = _recording(monkeypatch, out)
        again = export_world(site.world, out)
        assert again.written == first.written, "the manifest still names the whole record"
        assert moved == [], "the same ledger at the same clock is the same record: nothing moves"
        assert all((out / n).is_file() for n in blobs), "kept blobs are not pruned"

    def test_an_identical_file_is_not_staged(self, tmp_path: Path) -> None:
        out = tmp_path / "public"
        out.mkdir()
        (out / "same.json").write_bytes(b"{}\n")
        stage = StagedWrite(out)
        try:
            stage.write("same.json", b"{}\n")
            stage.write("new.json", b"[]\n")
            assert stage.written == ["same.json", "new.json"]
            assert stage.staged == ["new.json"]
        finally:
            stage.discard()


class TestTheOrder:
    def test_blobs_then_ledger_then_documents_then_the_summary(
        self, site: Site, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = tmp_path / "public"
        moved = _recording(monkeypatch, out)
        export_world(site.world, out)
        first_non_blob = next(i for i, n in enumerate(moved) if not n.startswith("blobs/"))
        assert all(n.startswith("blobs/") for n in moved[:first_non_blob])
        assert not [n for n in moved[first_non_blob:] if n.startswith("blobs/")]
        assert moved[first_non_blob : first_non_blob + 2] == ["ledger.jsonl", "ledger.jsonl.head"]
        assert moved[-2:] == ["summary.json", "verify.md"]
        cards = [i for i, n in enumerate(moved) if n.startswith("cards/")]
        assert cards
        assert max(cards) < moved.index("decisions.json")


class TestAFileHeldOpen:
    def test_it_is_retried_until_the_reader_lets_go(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source, target = tmp_path / "a", tmp_path / "b"
        source.write_bytes(b"new")
        target.write_bytes(b"old")
        real = os.replace
        refusals = iter(range(3))

        def held(src: Path, dst: Path) -> None:
            if next(refusals, None) is not None:
                raise PermissionError(5, "Access is denied")
            real(src, dst)

        monkeypatch.setattr(os, "replace", held)
        naps: list[float] = []
        replace_patiently(source, target, sleep=naps.append)
        assert target.read_bytes() == b"new"
        assert naps == [0.02, 0.04, 0.08]

    def test_it_gives_up_after_the_budget(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def always_held(src: Path, dst: Path) -> None:
            raise PermissionError(5, "Access is denied")

        monkeypatch.setattr(os, "replace", always_held)
        naps: list[float] = []
        with pytest.raises(PermissionError):
            replace_patiently(tmp_path / "a", tmp_path / "b", budget_s=3.0, sleep=naps.append)
        assert sum(naps) >= 3.0
        assert max(naps) == 1.0


class TestTheScanStillReadsEverything:
    def test_an_unchanged_file_is_scanned_where_it_is_published(
        self, site: Site, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = tmp_path / "public"
        export_world(site.world, out)
        genesis_hash = next(
            line.split('"hash":"')[1][:64]
            for line in (out / "ledger.jsonl").read_text("utf-8").splitlines()
            if '"hash":"' in line
        )
        # A value that became secret after the last export: the same export again stages nothing
        # (every byte is unchanged), so only the scan of the published files can find it.
        monkeypatch.setenv("T2SA_TEST_ROTATED_TOKEN", genesis_hash)
        with pytest.raises(ExportRefused) as refused:
            export_world(site.world, out)
        assert "ledger.jsonl" in {f.path for f in refused.value.findings}
