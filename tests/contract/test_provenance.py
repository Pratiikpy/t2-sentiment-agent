"""The ARGUS provenance record is self-contained and every ported file cites it consistently.

ARGUS is private, so ``third_party/argus/PROVENANCE.md`` is the only place a reader can check a
port: every header that says it was ported from ARGUS must name the one pinned commit and a
SHA-256 that the record lists, and the record must carry the MIT licence it relies on."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECORD = ROOT / "third_party" / "argus" / "PROVENANCE.md"
COMMIT = "3dec6baf9dfa7be37c7b452e26a9b139df252f75"
PRIVATE_NAME = "argus" + "-bitget"
"""The private repository's name, built so this file does not contain it."""
SHA256 = re.compile(r"\b[0-9a-f]{64}\b")
PORTED = re.compile(r"(?:ported|Ported)(?: in part)? from ARGUS")
HEADER_CHARS = 6000
"""Provenance lives in a module's header or docstring; a later mention is a citation, not a port."""


def _ported_files() -> list[Path]:
    out = []
    for base in ("src", "tests", "playbook/src"):
        for path in sorted((ROOT / base).rglob("*.py")):
            if PORTED.search(path.read_text(encoding="utf-8")[:HEADER_CHARS]):
                out.append(path)
    return out


def test_the_record_and_its_licence_exist() -> None:
    assert "MIT License" in (ROOT / "third_party" / "argus" / "LICENSE").read_text("utf-8")
    text = RECORD.read_text(encoding="utf-8")
    assert COMMIT in text
    assert "private" in text


def test_every_ported_header_cites_the_commit_the_record_and_a_listed_hash() -> None:
    record = RECORD.read_text(encoding="utf-8")
    listed = set(SHA256.findall(record))
    files = _ported_files()
    assert len(files) >= 10
    for path in files:
        text = path.read_text(encoding="utf-8")
        name = path.relative_to(ROOT).as_posix()
        if name.startswith("playbook/"):
            continue  # ports of the primary's modules, which carry the pins
        assert "third_party/argus" in text, name
        assert COMMIT[:8] in text, name
        cited = set(SHA256.findall(text[:HEADER_CHARS])) - {COMMIT}
        assert cited <= listed, (name, cited - listed)
        assert PRIVATE_NAME not in text, name


def test_no_file_points_a_reader_at_the_private_repository() -> None:
    for base in ("src", "tests", "scripts", "playbook"):
        for path in (ROOT / base).rglob("*"):
            if path.suffix in {".py", ".md", ".json", ".yaml"} and path.is_file():
                assert PRIVATE_NAME not in path.read_text(encoding="utf-8"), path
    for name in ("NOTICE.md", "README.md", "DESIGN.md", "RUNBOOK.md"):
        assert PRIVATE_NAME not in (ROOT / name).read_text(encoding="utf-8"), name
