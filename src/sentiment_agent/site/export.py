"""The public record: every file a judge reads, computed from the ledger and written to ``public/``.

``export_public`` reads one ledger and its blob store and writes (DESIGN.md §14, §18 M15):

====================  ============================================================================
``ledger.jsonl``      the ledger, byte for byte (every line is canonical, so it is re-emitted from
                      the verified events), with ``ledger.jsonl.head``, the head anchor
``blobs/<sha256>``    every blob any event commits to
``genesis.json``      the pre-registration, its hash, its OpenTimestamps records, amendments
``environment.json``  every environment proof (the Demo-only check before the first order)
``decisions.json``    every decision: outcome, stance, summary, targets against approved weights
``cards/<id>.json``   one :class:`~sentiment_agent.types.DecisionCard` per decision and protective
                      ruling (``site/cards.py``)
``orders.json``       every planned order with its preview, clientOid, venue orderId, state, fills
                      and net P&L, the skipped legs, and fills the agent did not plan
``funnel.json``       what the kernel did to what the model asked for, guard by guard
``equity_hourly.csv`` one row per hourly ``MARK`` event, in ledger order
``trades.csv``        the book's closed trades, in the order they closed
``metrics.json``      the governed book's metric set (``analysis.metrics.book_metrics``)
``arms.json``         every arm, the book first: a JSON array of arm results
``arms_summary.json`` each arm's headline metrics and the coin-flip distribution
``twin.json``         the governed/ungoverned twin report, when one was computed
``mirror.json``       the hourly Demo book against the same positions at live prices, and the
                      mirror and weekend-counterfactual arms
``redteam.json``      the red-team report, when one was run
``toolkit.json``      the Bitget toolkit coverage matrix, with the health the log shows
``feeds.json``        feed health (contract 1.1.0): the failing sources now, every alarm raised
                      and cleared, and each trigger kind a snapshot could not evaluate, with why
``replay.json``       a replay of a recorded venue-integrity refusal, labelled as a replay
``summary.json``      the headline numbers the page opens with
``verify.md``         how to check all of it
====================  ============================================================================

The format of ``ledger.jsonl``, ``blobs/``, ``equity_hourly.csv``, ``trades.csv``, ``metrics.json``
and ``arms.json`` is the one ``scripts/recompute.py`` reads and recomputes with the standard library
alone, and ``orders.json`` is what ``scripts/verify_orders.py`` checks against the Demo account.

**What the export refuses to publish** (:class:`ExportError`):

* a ledger that does not verify: a broken or truncated chain, or a referenced blob that is missing
  or altered. The last good export stays up; publishing a broken chain would claim an integrity it
  does not have;
* a card that cannot be built honestly (``site/cards.py``), a repeated fill id with two contents,
  or an arm passed in as the governed book whose numbers are not the ledger's;
* anything that looks like a secret or a local path (:class:`ExportRefused`). Every file is
  written to a staging folder beside ``out``, scanned there (:func:`scan_for_secrets`), and only
  moved into ``out`` when the scan is clean, so nothing unscanned ever lands where it is served.
  A file byte-identical to the one already published is not re-staged; it is scanned where it
  lies, by the same rules (:class:`StagedWrite`).
  The scan looks for ``BITGET_`` anywhere (no credential variable, not even its name), Bitget API
  key shapes, bearer tokens and other key assignments, private keys, the value of every
  secret-named environment variable, this machine's home, project and working directories, and
  any Windows drive path or ``/Users``/``/home`` path. It does not redact: a ledger line or a
  blob cannot be changed without breaking the chain, so a hit stops the export and names the file
  and the rule, never the matched text.

Nothing here reads a credential file, the network or the wall clock (``clock`` stamps the export).
"""

import contextlib
import csv
import getpass
import io
import json
import math
import os
import re
import secrets
import shutil
import tempfile
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Final

from sentiment_agent import __version__
from sentiment_agent.analysis.baselines import coin_flip_arms, coin_flip_summary, comparator_arm
from sentiment_agent.analysis.metrics import BOOK_ARM_ID, BOOK_SPEC, book_marks, book_metrics
from sentiment_agent.analysis.simcheck import SimulatorCheck
from sentiment_agent.book.projection import Projection, ProjectionError
from sentiment_agent.hashing import ZERO_HASH, canonical_json
from sentiment_agent.kernel.guards import Leg, g1_venue_integrity
from sentiment_agent.ledger.blobs import FileBlobStore
from sentiment_agent.ledger.chain import (
    ANCHOR_SUFFIX,
    GenesisError,
    HashChainLedger,
    LedgerError,
    referenced_blobs,
)
from sentiment_agent.ledger.genesis import recorded_x_post, x_post_text, x_posted_at
from sentiment_agent.perception.features import index_move_bps_3h
from sentiment_agent.policy import ACTIVE_POLICY, POLICY_V1
from sentiment_agent.site.cards import CardBundle, CardError, OrderTrail, build_card_bundle
from sentiment_agent.site.coverage import coverage_counts, declared_uses, ledger_health
from sentiment_agent.types import (
    PROJECT_SLUG,
    Amendment,
    AnchorRecord,
    ArmKind,
    ArmResult,
    BlobRef,
    Candle,
    CandleKind,
    Clock,
    ClosedTrade,
    DecisionEvent,
    EnvironmentProof,
    EventKind,
    FeedHealthReport,
    Fill,
    Genesis,
    GuardId,
    GuardStatus,
    KernelRuling,
    LedgerEvent,
    MarkPoint,
    Model,
    OrderState,
    Policy,
    PriceSource,
    Quote,
    RedTeamReport,
    RunMode,
    Stance,
    ToolkitUse,
    TriggerKind,
    TwinReport,
    UtcDatetime,
    parse_payload,
)
from sentiment_agent.venue.public_api import parse_candle

TEMPLATES: Final = Path(__file__).resolve().parent / "templates"
REPLAY_FIXTURE: Final = TEMPLATES / "replay_g1_btcperp_20260923.json"
REPLAY_FIXTURE_NAME: Final = "src/sentiment_agent/site/templates/replay_g1_btcperp_20260923.json"

PLACEHOLDER_EQUITY: Final = Decimal(1)
"""Given to the projection of a SIMULATED or DRYRUN ledger that records no account read. Closed
trades are a fold over fills alone and never read it; the equity series is taken from the logged
``MARK`` events, never from this number. A PAPER ledger is never given it."""

EQUITY_COLUMNS: Final = (
    "seq",
    "at",
    "equity_book",
    "equity_venue",
    "equity_live_mirror",
    "gross_weight",
    "net_weight",
)
TRADE_COLUMNS: Final = (
    "symbol",
    "opened_at",
    "closed_at",
    "direction",
    "entry_avg",
    "exit_avg",
    "max_abs_qty",
    "gross_pnl",
    "fees",
    "net_pnl",
    "exit_reason",
    "decision_ids",
)

_FLOAT_TOL: Final = 1e-9


class ExportError(RuntimeError):
    """The record cannot be published honestly; nothing was moved into ``out``."""


@dataclass(frozen=True, slots=True)
class Finding:
    """One scan hit. Names the file, the rule and the position, never the matched text."""

    path: str
    rule: str
    offset: int


class ExportRefused(ExportError):  # noqa: N818 - it names what happened: the export was refused
    """The scan found something that looks like a secret or a local path."""

    def __init__(self, findings: Sequence[Finding]) -> None:
        self.findings = tuple(findings)
        listed = "; ".join(f"{f.path} @{f.offset}: {f.rule}" for f in self.findings[:12])
        more = f" (+{len(self.findings) - 12} more)" if len(self.findings) > 12 else ""
        super().__init__(
            f"refusing to publish: {len(self.findings)} secret or local-path finding(s): "
            f"{listed}{more}"
        )


class ExportManifest(Model):
    """What one export wrote."""

    written: tuple[str, ...]
    """Every file written, as a path relative to ``out`` with forward slashes."""
    ledger_head: str
    """The hash of the last exported event; the zero hash for an empty ledger."""
    generated_at: UtcDatetime


# ================================================================================================
# The secret and local-path scan
# ================================================================================================

_SEPARATOR: Final = r"(?:\\\\|/|\\(?![nrtbfu\"\\/]))"
"""A path separator in published text: ``/``, a JSON-escaped backslash, or a single backslash that
does not begin a JSON escape sequence. Crowd text is stored as JSON, where ``it's:\\n\\nToday\\n``
is a colon and two newlines, not a drive ``s:`` with backslash separators; the first live dry-run
export (2026-09-24) was refused on exactly such a post. A raw path whose component begins with one
of those letters (``C:\\temp\\``) is not matched here; this machine's own directories are caught
literally by :func:`local_literals` in every spelling."""

_TEXT_RULES: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("a BITGET_ variable name", re.compile(r"BITGET_")),
    ("a Bitget API key shape", re.compile(r"(?<![A-Za-z0-9])bg_[0-9a-f]{32}(?![0-9a-f])")),
    ("an sk- API key shape", re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}")),
    ("a private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("a Bitget signing header", re.compile(r"(?i)\bACCESS-(?:KEY|SIGN|PASSPHRASE)\b")),
    ("a bearer token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{16,}")),
    (
        "a key or password assignment",
        re.compile(
            r"(?i)\b(?:api[_-]?key|api[_-]?secret|secret[_-]?key|access[_-]?key|passphrase|"
            r"password)\b\\?[\"']?\s*[:=]\s*\\?[\"']?[A-Za-z0-9+/_=.-]{8,}"
        ),
    ),
    (
        "a Windows drive path",
        re.compile(
            # A drive letter standing alone (not the end of a word, nor of a contraction such as
            # the "s" of "it's:"), then one path component between two separators.
            r"(?<![A-Za-z0-9_'‘’])[A-Za-z]:" + _SEPARATOR + r"[^\\/\s\"'<>|:*?]{1,255}" + _SEPARATOR
        ),
    ),
    ("a /Users or /home path", re.compile(r"(?<![A-Za-z0-9_.-])/(?:Users|home)/[A-Za-z0-9._-]+/")),
    ("a file: URL", re.compile(r"(?i)\bfile:/")),
)

_SECRET_NAME: Final = re.compile(r"(?i)(KEY|SECRET|TOKEN|PASS|PASSWORD|PASSPHRASE|CREDENTIAL|AUTH)")
_MIN_SECRET_LENGTH: Final = 8


def _path_forms(path: Path) -> set[str]:
    text = str(path)
    forms = {text, path.as_posix(), text.replace("\\", "\\\\"), text.replace("\\", "/")}
    return {f for f in forms if len(f) >= 6}


def local_literals(extra_paths: Iterable[Path] = ()) -> tuple[tuple[str, str], ...]:
    """``(label, text)`` pairs that must never appear in a published file: the value of every
    secret-named environment variable, and this machine's home, working, temporary and given
    directories in every spelling a JSON or HTML file could carry them."""
    out: list[tuple[str, str]] = []
    for name, value in os.environ.items():
        if _SECRET_NAME.search(name) and len(value.strip()) >= _MIN_SECRET_LENGTH:
            out.append((f"the value of environment variable {name}", value.strip()))
    paths: list[tuple[str, Path]] = []
    with contextlib.suppress(RuntimeError, KeyError):
        paths.append(("the home directory", Path.home()))
    with contextlib.suppress(OSError):
        paths.append(("the working directory", Path.cwd()))
    paths.append(("the temporary directory", Path(tempfile.gettempdir())))
    paths.extend(("a local directory", p) for p in extra_paths)
    for label, path in paths:
        resolved = path.resolve()
        if len(resolved.parts) < 2:
            continue
        out.extend((label, form) for form in sorted(_path_forms(resolved)))
    with contextlib.suppress(Exception):
        user = getpass.getuser()
        if len(user) >= 3:
            for sep in ("\\", "/", "\\\\"):
                out.append(("the user's home folder", f"Users{sep}{user}"))
            out.append(("the user's home folder", f"/home/{user}"))
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for label, text in out:
        if text.lower() not in seen:
            seen.add(text.lower())
            unique.append((label, text))
    return tuple(unique)


def _looks_textual(data: bytes) -> str | None:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def scan_bytes(name: str, data: bytes, literals: Sequence[tuple[str, str]]) -> list[Finding]:
    """Every finding in one file: the pattern rules on text, the literals on every file."""
    findings: list[Finding] = []
    text = _looks_textual(data)
    haystack = (text if text is not None else data.decode("latin-1")).lower()
    for label, literal in literals:
        index = haystack.find(literal.lower())
        if index >= 0:
            findings.append(Finding(name, label, index))
    if text is not None:
        for label, pattern in _TEXT_RULES:
            match = pattern.search(text)
            if match is not None:
                findings.append(Finding(name, label, match.start()))
    return findings


def scan_for_secrets(
    root: Path, names: Iterable[str], *, extra_paths: Iterable[Path] = ()
) -> list[Finding]:
    """Scan the files ``names`` (relative to ``root``). Used on the staging folder before anything
    is published, and by the render on the page it writes."""
    literals = local_literals([root, *extra_paths])
    findings: list[Finding] = []
    for name in names:
        findings.extend(scan_bytes(name, (root / name).read_bytes(), literals))
    return findings


# ================================================================================================
# Writing
# ================================================================================================


def _dumps(value: Any) -> bytes:
    return (
        json.dumps(value, indent=1, ensure_ascii=False, allow_nan=False, sort_keys=False) + "\n"
    ).encode("utf-8")


def _dump(model: Model) -> Any:
    return model.model_dump(mode="json")


def _iso(at: datetime) -> str:
    dumped = json.loads(canonical_json(at))
    if not isinstance(dumped, str):  # pragma: no cover - canonical_json writes datetimes as text
        raise ExportError("a timestamp did not serialise as text")
    return dumped


PUBLISH_LAST: Final = (
    "ledger.jsonl",
    f"ledger.jsonl{ANCHOR_SUFFIX}",
    "index.html",
    "summary.json",
    "verify.md",
)
"""Files moved after every other one, in this order (:meth:`StagedWrite.publish`)."""

REPLACE_BUDGET_S: Final = 60.0
"""How long one move into ``out`` keeps retrying a file another process holds open."""


def _publish_rank(name: str) -> tuple[int, int]:
    """The order a record is moved into place, so a reader that copies ``out`` mid-publish
    never holds a file that cites something not yet there: blobs first (the ledger commits to
    them), then the ledger and its anchor (the documents cite its events), then cards and the other
    documents, and ``summary.json`` near the end. ``summary.json`` is the commit marker: its
    ``ledger.head_hash`` equals the anchor's only once a publish has completed, which is what
    ``scripts/publish_site.ps1`` checks before it deploys a copy."""
    if name.startswith("blobs/"):
        return (0, 0)
    if name in PUBLISH_LAST[:2]:
        return (1, PUBLISH_LAST.index(name))
    if name in PUBLISH_LAST:
        return (4, PUBLISH_LAST.index(name))
    if name.startswith("cards/"):
        return (2, 0)
    return (3, 0)


def replace_patiently(
    source: Path, target: Path, *, budget_s: float = REPLACE_BUDGET_S, sleep: Any = time.sleep
) -> None:
    """``os.replace``, retried while another process holds ``target`` open.

    On Windows a rename onto a file someone is reading fails at once with ``PermissionError``
    (WinError 5) instead of waiting. The blob store's own retry (``ledger/blobs.py``) covers
    readers that hold a blob for microseconds; a copy of this folder holds each file for as long as
    the copy takes, so this backs off to a second and gives up only after ``budget_s``. Run 1's
    hourly export died this way twice (ledger notes at seq 571 and the 17:00 mark on 2026-09-25):
    the page's publisher was copying ``public/`` while the export moved files into it.
    """
    waited = 0.0
    delay = 0.02
    while True:
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if waited >= budget_s:
                raise
            sleep(delay)
            waited += delay
            delay = min(delay * 2, 1.0)


class StagedWrite:
    """Files written under a staging folder, then moved into ``out`` together.

    A file whose bytes equal the one already in ``out`` is not staged or moved: the published copy
    is the file, and it is scanned where it lies (:meth:`scan`). Run 1 re-wrote every one of its
    27,000 content-addressed blobs every hour, which made each publish minutes long and every one
    of those minutes a chance to collide with a reader. ``written`` still names every file of the
    record, so a manifest and :func:`prune` see the whole of it; ``staged`` names what moves.
    """

    def __init__(self, out: Path) -> None:
        self.out = out
        self.root = out.parent / f".{out.name}.staging-{secrets.token_hex(6)}"
        self.root.mkdir(parents=True)
        self.written: list[str] = []
        self.staged: list[str] = []

    def write(self, name: str, data: bytes) -> None:
        self.written.append(name)
        published = self.out / name
        with contextlib.suppress(OSError):
            unchanged = published.is_file() and published.stat().st_size == len(data)
            if unchanged and published.read_bytes() == data:
                return
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        self.staged.append(name)

    def json(self, name: str, value: Any) -> None:
        self.write(name, _dumps(value))

    def scan(self, *, extra_paths: Iterable[Path] = ()) -> list[Finding]:
        """Every file of the record, scanned: staged files in the staging folder, unchanged ones
        where they are published, so nothing is served that this export's rules did not read."""
        extra = [self.out, *extra_paths]
        staged = set(self.staged)
        kept = [n for n in self.written if n not in staged]
        return scan_for_secrets(self.root, self.staged, extra_paths=extra) + scan_for_secrets(
            self.out, kept, extra_paths=[self.root, *extra]
        )

    def publish(self) -> None:
        for name in sorted(self.staged, key=_publish_rank):
            source = self.root / name
            target = self.out / name
            target.parent.mkdir(parents=True, exist_ok=True)
            replace_patiently(source, target)

    def discard(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


def prune(out: Path, folder: str, keep: set[str], suffixes: tuple[str, ...]) -> None:
    """Remove files in ``out/folder`` a previous export wrote and this one did not."""
    directory = out / folder
    if not directory.is_dir():
        return
    for entry in directory.iterdir():
        rel = f"{folder}/{entry.name}"
        if entry.is_file() and rel not in keep and (not suffixes or entry.suffix in suffixes):
            entry.unlink()


# ================================================================================================
# Reading the ledger
# ================================================================================================


class _Events:
    """A :class:`~sentiment_agent.types.LedgerReader` over a fixed list of verified events."""

    def __init__(self, events: Sequence[LedgerEvent]) -> None:
        self._events = tuple(events)

    def events(self, kinds: frozenset[EventKind] | None = None) -> Iterator[LedgerEvent]:
        if kinds is None:
            return iter(self._events)
        return iter([e for e in self._events if e.kind in kinds])

    def head(self) -> LedgerEvent | None:
        return self._events[-1] if self._events else None


def _active_policy(events: Sequence[LedgerEvent]) -> Policy:
    policy = ACTIVE_POLICY
    for event in events:
        if event.kind is EventKind.GENESIS:
            policy = Genesis.model_validate(event.payload).policy
        elif event.kind is EventKind.AMENDMENT:
            policy = Amendment.model_validate(event.payload).new_policy
    return policy


def _unique_fills(fills: Sequence[Fill]) -> tuple[Fill, ...]:
    """Fills with identical repeats removed; a repeated id with other content is refused."""
    seen: dict[str, Fill] = {}
    out: list[Fill] = []
    for fill in fills:
        known = seen.get(fill.exec_id)
        if known is None:
            seen[fill.exec_id] = fill
            out.append(fill)
        elif known != fill:
            raise ExportError(f"fill {fill.exec_id} appears twice with different content")
    return tuple(out)


@dataclass(frozen=True, slots=True)
class _Record:
    events: tuple[LedgerEvent, ...]
    projection: Projection
    mode: RunMode | None
    policy: Policy
    trades: tuple[ClosedTrade, ...]
    fills: tuple[Fill, ...]
    marks: tuple[MarkPoint, ...]


def _read(ledger: HashChainLedger, blobs: FileBlobStore) -> _Record:
    try:
        events = tuple(ledger.events())
    except LedgerError as exc:
        raise ExportError(f"the ledger does not read: {exc}") from None
    verification = ledger.verify(blobs)
    if not verification.intact:
        problem = verification.break_reason or verification.anchor
        if verification.missing_blobs:
            problem = f"{len(verification.missing_blobs)} referenced blob(s) missing or altered"
        raise ExportError(f"the ledger does not verify ({problem}); nothing is published")
    if verification.events < len(events):
        raise ExportError("the ledger shrank while it was being exported")
    policy = _active_policy(events)
    mode = events[0].mode if events else None
    fallback = None if mode is RunMode.PAPER else PLACEHOLDER_EQUITY
    projection = Projection.from_ledger(_Events(events), policy, starting_equity=fallback)
    fills = _unique_fills(projection.fills)
    trades: tuple[ClosedTrade, ...] = ()
    if fills:
        try:
            trades = projection.closed_trades
        except ProjectionError as exc:
            raise ExportError(f"the book cannot be rebuilt from the ledger: {exc}") from None
    return _Record(
        events=events,
        projection=projection,
        mode=mode,
        policy=policy,
        trades=trades,
        fills=fills,
        marks=projection.marks,
    )


# ================================================================================================
# The files
# ================================================================================================


def _ledger_files(stage: StagedWrite, record: _Record, blobs: FileBlobStore) -> None:
    buffer = io.BytesIO()
    for event in record.events:
        buffer.write(canonical_json(event) + b"\n")
    stage.write("ledger.jsonl", buffer.getvalue())
    if record.events:
        head = record.events[-1]
        anchor = {
            "events": len(record.events),
            "head_seq": head.seq,
            "head_hash": head.hash,
            "mode": head.mode.value,
        }
        stage.write(
            f"ledger.jsonl{ANCHOR_SUFFIX}",
            json.dumps(anchor, sort_keys=True).encode("utf-8") + b"\n",
        )
    refs: dict[str, BlobRef] = {}
    for event in record.events:
        try:
            for ref in referenced_blobs(event):
                refs.setdefault(ref.sha256, ref)
        except LedgerError as exc:
            raise ExportError(str(exc)) from None
    for sha in sorted(refs):
        stage.write(f"blobs/{sha}", blobs.get(sha))


def _event_row(event: LedgerEvent) -> dict[str, Any]:
    return {"seq": event.seq, "ts": _iso(event.ts), "hash": event.hash}


def _genesis_doc(record: _Record) -> dict[str, Any]:
    genesis_event = next((e for e in record.events if e.kind is EventKind.GENESIS), None)
    anchors = [
        {"seq": e.seq, "record": e.payload} for e in record.events if e.kind is EventKind.ANCHOR
    ]
    amendments = [
        {**_event_row(e), "amendment": e.payload}
        for e in record.events
        if e.kind is EventKind.AMENDMENT
    ]
    if genesis_event is None:
        return {
            "present": False,
            "note": "this ledger has no genesis: it is not a pre-registered run",
            "anchors": anchors,
            "amendments": amendments,
        }
    post: str | None
    try:
        post = x_post_text(genesis_event)
    except GenesisError:
        post = None
    genesis_anchors = [
        a
        for a in anchors
        if AnchorRecord.model_validate(a["record"]).target_seq == genesis_event.seq
    ]
    return {
        "present": True,
        **_event_row(genesis_event),
        "genesis": genesis_event.payload,
        "anchors": anchors,
        "genesis_anchors": genesis_anchors,
        "amendments": amendments,
        "x_post_text": post,
        "x_post": _x_post_doc(record.events) if post is not None else None,
    }


def _x_post_doc(events: Sequence[LedgerEvent]) -> dict[str, Any] | None:
    """The owner's X post of the pre-registration, when ``t2sa x-posted`` recorded one: its URL,
    when it was made (read from the post's own id, so nobody's word is needed for the time), and
    whether that was before the first order reached the venue. None until it is recorded, and the
    page then calls the text a draft."""
    recorded = recorded_x_post(events)
    if recorded is None:
        return None
    url, note = recorded
    posted_at = x_posted_at(url)
    first_order = next((e for e in events if e.kind is EventKind.ORDER_SUBMITTED), None)
    return {
        "url": url,
        "posted_at": _iso(posted_at),
        "recorded_seq": note.seq,
        "recorded_at": _iso(note.ts),
        "first_order_at": _iso(first_order.ts) if first_order is not None else None,
        "before_first_order": posted_at < first_order.ts if first_order is not None else None,
    }


FEED_HISTORY_LIMIT: Final = 500
"""Most recent alarm raises and clears listed in ``feeds.json`` (every one is in the ledger)."""


def _feeds_doc(record: _Record) -> dict[str, Any]:
    """Feed health from the ``feed_health`` events: the latest report, the history of alarms raised
    and cleared, and the counts. A 1.0.0 ledger has none and says so rather than showing health."""
    reports = [
        (e, FeedHealthReport.model_validate(e.payload))
        for e in record.events
        if e.kind is EventKind.FEED_HEALTH
    ]
    if not reports:
        return {
            "status": "not_logged",
            "note": "this ledger has no feed_health events (it predates contract 1.1.0); the "
            "health of each source is in its snapshots' source calls",
            "latest": None,
            "history": [],
            "counts": {"reports": 0, "raised": 0, "cleared": 0, "open": 0, "blind_by_failure": 0},
        }
    latest_event, latest = reports[-1]
    history = [
        {
            "seq": e.seq,
            "at": _iso(r.at),
            "snapshot_id": r.snapshot_id,
            "raised": list(r.raised),
            "cleared": list(r.cleared),
        }
        for e, r in reports
        if r.raised or r.cleared
    ]
    return {
        "status": "logged",
        "latest_seq": latest_event.seq,
        "latest": _dump(latest),
        "history": history[-FEED_HISTORY_LIMIT:],
        "counts": {
            "reports": len(reports),
            "raised": sum(len(r.raised) for _, r in reports),
            "cleared": sum(len(r.cleared) for _, r in reports),
            "open": len(latest.alarms),
            "blind_by_failure": len(latest.failure_blind()),
        },
    }


def _environment_doc(record: _Record) -> dict[str, Any]:
    proofs = [
        {**_event_row(e), "proof": e.payload}
        for e in record.events
        if e.kind is EventKind.ENVIRONMENT_PROOF
    ]
    latest = EnvironmentProof.model_validate(proofs[-1]["proof"]) if proofs else None
    return {
        "mode": record.mode.value if record.mode else None,
        "proofs": proofs,
        "latest_passed": latest.passed if latest is not None else None,
        "note": (
            "A PAPER ledger's first order requires a passed proof. "
            + (
                "This ledger is not PAPER: it sends nothing to Bitget, so it holds no proof."
                if record.mode is not RunMode.PAPER
                else ""
            )
        ).strip(),
    }


def _approved(ruling: KernelRuling | None, symbol: str) -> dict[str, Any]:
    if ruling is None:
        return {"approved_weight": None, "binding_guard": None}
    inst = ruling.instrument(symbol)
    if inst is None:
        return {"approved_weight": None, "binding_guard": None}
    return {
        "approved_weight": inst.approved_weight,
        "binding_guard": inst.binding_guard.value if inst.binding_guard else None,
    }


def _decisions_doc(record: _Record, bundle: CardBundle) -> dict[str, Any]:
    projection = record.projection
    by_decision = {c.decision_id: c for c in bundle.cards if c.decision_id is not None}
    seq_of: dict[str, int] = {}
    for event in record.events:
        payload = parse_payload(event)
        if isinstance(payload, DecisionEvent):
            seq_of.setdefault(payload.record.decision_id, event.seq)
    rows: list[dict[str, Any]] = []
    stances: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    trigger_kinds = {t.trigger_id: t.kind for t in projection.triggers}
    for rec in projection.decisions:
        card = by_decision.get(rec.decision_id)
        ruling = card.kernel if card is not None else None
        outcomes[rec.outcome.value] += 1
        decision = rec.decision
        if decision is not None:
            stances[decision.stance.value] += 1
        targets = []
        for target in decision.targets if decision is not None else ():
            report = rec.grounding.get(target.symbol)
            targets.append(
                {
                    "symbol": target.symbol,
                    "target": target.target,
                    "proposed_weight": rec.proposed_weights.get(target.symbol),
                    **_approved(ruling, target.symbol),
                    "confidence": target.confidence,
                    "horizon_hours": target.horizon_hours,
                    "grounded": report.grounded if report is not None else None,
                }
            )
        rows.append(
            {
                "decision_id": rec.decision_id,
                "card_id": card.card_id if card is not None else None,
                "seq": seq_of.get(rec.decision_id),
                "decided_at": _iso(rec.decided_at),
                "outcome": rec.outcome.value,
                "stance": decision.stance.value if decision is not None else None,
                "summary": decision.summary if decision is not None else None,
                "mandate_response": decision.mandate_response if decision is not None else None,
                "flat_reasons": list(decision.flat_reasons) if decision is not None else [],
                "triggers": [
                    {
                        "trigger_id": t,
                        "kind": trigger_kinds[t].value if t in trigger_kinds else None,
                    }
                    for t in rec.trigger_ids
                ],
                "targets": targets,
                "changed_by_kernel": ruling.changed_by_kernel if ruling is not None else None,
                "model": rec.call.model,
                "thinking": rec.call.thinking.value,
                "attempts": rec.call.attempts,
                "tokens": rec.call.usage.total_tokens,
                "tokens_reported": rec.call.usage.reported,
                "error": rec.call.error,
            }
        )
    owner_triggers = sum(1 for t in projection.triggers if t.kind is TriggerKind.OWNER_MANUAL)
    return {
        "counts": {
            "decisions": len(rows),
            "decided": outcomes.get("decided", 0),
            "act": stances.get(Stance.ACT.value, 0),
            "hold": stances.get(Stance.HOLD.value, 0),
            "flat_with_reasons": stances.get(Stance.FLAT_WITH_REASONS.value, 0),
            "outcomes": dict(sorted(outcomes.items())),
            "changed_by_kernel": sum(1 for r in rows if r["changed_by_kernel"]),
            "owner_manual_triggers": owner_triggers,
            "amendments": len(projection.amendments),
        },
        "decisions": rows,
    }


def _trail_doc(trail: OrderTrail, venue: str | None) -> dict[str, Any]:
    order = trail.order
    intent = trail.intent
    return {
        "card_id": trail.card_id,
        "plan_id": trail.plan_id,
        "decision_id": intent.decision_id,
        "ruling_id": intent.ruling_id,
        "client_oid": order.client_oid,
        "venue_order_id": order.venue_order_id,
        "venue": venue,
        "symbol": order.symbol,
        "side": order.side.value,
        "qty": str(order.qty),
        "purpose": order.purpose.value,
        "reduce_only": intent.reduce_only,
        "reference_price": str(intent.reference_price),
        "notional": str(intent.notional),
        "expected_fee": str(intent.expected_fee),
        "stop_loss_price": str(intent.stop_loss_price) if intent.stop_loss_price else None,
        "split": f"{intent.split_index + 1}/{intent.split_count}",
        "state": order.state.value,
        "previewed": trail.preview is not None,
        "preview_argv": list(trail.preview.argv) if trail.preview is not None else None,
        "submitted_at": _iso(trail.submitted.submitted_at) if trail.submitted else None,
        "acked_at": _iso(trail.ack.acked_at) if trail.ack else None,
        "rejection": _dump(trail.rejection) if trail.rejection else None,
        "unknown": _dump(trail.unknown) if trail.unknown else None,
        "fills": [_dump(f) for f in order.fills],
        "net_pnl": str(order.net_pnl) if order.net_pnl is not None else None,
        "ledger_seqs": list(trail.seqs),
    }


def _orders_doc(record: _Record, bundle: CardBundle) -> dict[str, Any]:
    venue = (
        {
            RunMode.PAPER: "bitget_demo",
            RunMode.SIMULATED: "simulated",
            RunMode.DRYRUN: "none (dry run: previews only)",
        }.get(record.mode)
        if record.mode
        else None
    )
    orders = [_trail_doc(t, venue) for t in bundle.trails]
    skipped: list[dict[str, Any]] = []
    card_of_ruling = {c.ruling_id: c.card_id for c in bundle.cards if c.ruling_id}
    for plan in record.projection.plans:
        for leg in plan.skipped:
            skipped.append(
                {
                    "card_id": card_of_ruling.get(plan.ruling_id),
                    "plan_id": plan.plan_id,
                    "symbol": leg.symbol,
                    "wanted_delta_weight": leg.wanted_delta_weight,
                    "reason": leg.reason,
                }
            )
    states = Counter(o["state"] for o in orders)
    return {
        "venue": venue,
        "note": (
            "Every venue_order_id can be checked against the Demo account with "
            "scripts/verify_orders.py (read-only)."
            if record.mode is RunMode.PAPER
            else "This ledger is not PAPER: its order ids are not Bitget Demo orders."
        ),
        "counts": {
            "planned": len(orders),
            "previewed": sum(1 for o in orders if o["previewed"]),
            "sent": sum(1 for o in orders if o["submitted_at"]),
            "acknowledged": sum(1 for o in orders if o["acked_at"]),
            "states": dict(sorted(states.items())),
            "skipped_legs": len(skipped),
            "venue_originated_fills": len(bundle.venue_originated_fills),
        },
        "orders": orders,
        "skipped_legs": skipped,
        "venue_originated_fills": [
            {"seq": seq, "fill": _dump(fill)} for seq, fill in bundle.venue_originated_fills
        ],
    }


def _adds(current: float, reference: float) -> bool:
    return Leg(symbol="-", current=current, proposed=reference).adds_exposure


def _funnel_doc(record: _Record, bundle: CardBundle) -> dict[str, Any]:
    rulings = record.projection.rulings
    decision_rulings = [r for r in rulings if r.decision_id is not None]
    protective = [r for r in rulings if r.protective_reason is not None]
    legs = [
        (r, i) for r in decision_rulings for i in r.instruments if i.proposed_weight is not None
    ]
    adding = [(r, i) for r, i in legs if _adds(i.current_weight, i.reference)]
    full = [(r, i) for r, i in adding if not i.changed_by_kernel]
    refused = [(r, i) for r, i in adding if i.changed_by_kernel and i.approved_weight == 0]
    cut = [(r, i) for r, i in adding if i.changed_by_kernel and i.approved_weight != 0]
    guards: dict[str, dict[str, int]] = {
        g.value: {s.value: 0 for s in GuardStatus} | {"binding": 0} for g in GuardId
    }
    for ruling in decision_rulings:
        for g in ruling.book_rulings:
            guards[g.guard.value][g.status.value] += 1
        for inst in ruling.instruments:
            for g in inst.rulings:
                guards[g.guard.value][g.status.value] += 1
            if inst.binding_guard is not None and inst.changed_by_kernel:
                guards[inst.binding_guard.value]["binding"] += 1
    trails = [t for t in bundle.trails if t.intent.decision_id is not None]
    states = Counter(t.order.state for t in trails)
    reasons = Counter(r.protective_reason.value for r in protective if r.protective_reason)
    return {
        "rulings": len(decision_rulings),
        "legs_proposed": len(legs),
        "legs_adding_exposure": len(adding),
        "approved_in_full": len(full),
        "cut": len(cut),
        "refused": len(refused),
        "exposure_asked": sum(abs(i.reference) - abs(i.current_weight) for _, i in adding),
        "exposure_approved": sum(
            max(0.0, abs(i.approved_weight) - abs(i.current_weight)) for _, i in adding
        ),
        "guards": guards,
        "orders": {
            "planned": len(trails),
            "previewed": sum(1 for t in trails if t.preview is not None),
            "sent": sum(1 for t in trails if t.submitted is not None),
            "acknowledged": sum(1 for t in trails if t.ack is not None),
            "filled": states.get(OrderState.FILLED, 0),
            "rejected": states.get(OrderState.REJECTED, 0),
            "denied": states.get(OrderState.DENIED, 0),
            "unknown": states.get(OrderState.UNKNOWN, 0),
        },
        "protective": {
            "rulings": len(protective),
            "by_reason": dict(sorted(reasons.items())),
            "legs_closed": sum(1 for r in protective for i in r.instruments if i.changed_by_kernel),
        },
    }


def _equity_csv(record: _Record) -> bytes:
    handle = io.StringIO()
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(EQUITY_COLUMNS)
    for event in record.events:
        if event.kind is not EventKind.MARK:
            continue
        p = event.payload
        writer.writerow(
            [
                event.seq,
                p["at"],
                p["equity_book"],
                p["equity_venue"] if p["equity_venue"] is not None else "",
                p["equity_live_mirror"] if p["equity_live_mirror"] is not None else "",
                repr(float(p["gross_weight"])),
                repr(float(p["net_weight"])),
            ]
        )
    return handle.getvalue().encode("utf-8")


def _trades_csv(trades: Sequence[ClosedTrade]) -> bytes:
    handle = io.StringIO()
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(TRADE_COLUMNS)
    for trade in trades:
        d = _dump(trade)
        writer.writerow(
            [
                d["symbol"],
                d["opened_at"],
                d["closed_at"],
                d["direction"],
                d["entry_avg"],
                d["exit_avg"],
                d["max_abs_qty"],
                d["gross_pnl"],
                d["fees"],
                d["net_pnl"],
                d["exit_reason"],
                " ".join(d["decision_ids"]),
            ]
        )
    return handle.getvalue().encode("utf-8")


def _close(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=_FLOAT_TOL, abs_tol=1e-12)


def _check_book_arm(given: ArmResult, ours: ArmResult) -> None:
    g, o = given.metrics, ours.metrics
    same = (
        g.n_hours == o.n_hours
        and g.n_closed_trades == o.n_closed_trades
        and all(
            _close(getattr(g, name), getattr(o, name))
            for name in (
                "total_return",
                "sharpe_ann",
                "sharpe_se_ann",
                "sortino_ann",
                "max_drawdown",
                "win_rate",
                "turnover",
                "fees_paid",
            )
        )
        and len(given.marks) == len(ours.marks)
        and all(
            a.at == b.at and _close(a.equity, b.equity)
            for a, b in zip(given.marks, ours.marks, strict=True)
        )
    )
    if not same:
        raise ExportError(
            f"the arm passed as {BOOK_ARM_ID} does not match the book the ledger records; the "
            "published book is always the ledger's"
        )


def _arms(record: _Record, arms: Sequence[ArmResult]) -> tuple[ArmResult, list[ArmResult]]:
    if record.fills and not record.marks:
        raise ExportError(
            "metrics cannot be computed yet: the ledger holds fills but no hourly mark; export "
            "after the next MARK event"
        )
    try:
        metrics = book_metrics(record.marks, record.trades, record.fills)
    except ValueError as exc:
        raise ExportError(f"the book's metrics cannot be computed: {exc}") from None
    book = ArmResult(
        spec=BOOK_SPEC, marks=book_marks(record.marks), trades=record.trades, metrics=metrics
    )
    ids = [a.spec.arm_id for a in arms]
    repeated = sorted({i for i in ids if ids.count(i) > 1})
    if repeated:
        raise ExportError(f"arms passed twice: {repeated}")
    others: list[ArmResult] = []
    for arm in arms:
        if arm.spec.arm_id == BOOK_ARM_ID:
            _check_book_arm(arm, book)
        else:
            others.append(arm)
    return book, [book, *others]


def _family(arm_id: str) -> str:
    return re.sub(r"_s\d+$", "", arm_id)


def _arms_summary(
    book: ArmResult, arms: Sequence[ArmResult], sim_check: SimulatorCheck | None
) -> dict[str, Any]:
    rows = []
    for arm in arms:
        m = arm.metrics
        rows.append(
            {
                "arm_id": arm.spec.arm_id,
                "family": _family(arm.spec.arm_id),
                "kind": arm.spec.kind.value,
                "title": arm.spec.title,
                "description": arm.spec.description,
                "provenance": arm.spec.provenance,
                "uses_llm": arm.spec.uses_llm,
                "guards": [g.value for g in arm.spec.guards],
                "metrics": _dump(m),
            }
        )
    flips = coin_flip_arms(arms)
    return {
        "arms": rows,
        "families": dict(sorted(Counter(r["family"] for r in rows).items())),
        # Ranked through the replica, which is costed exactly as the seeds are (baselines.py).
        "coin_flip": _dump(coin_flip_summary(comparator_arm(book, arms), arms)) if flips else None,
        "replica_vs_live": _dump(sim_check) if sim_check is not None else None,
        "label": "descriptive, not inferential",
    }


def _mirror_doc(record: _Record, arms: Sequence[ArmResult]) -> dict[str, Any]:
    series = []
    for mark in record.marks:
        mirror = mark.equity_live_mirror
        gap = (
            float(mirror / mark.equity_book - 1) * 10_000
            if mirror is not None and mark.equity_book > 0
            else None
        )
        series.append(
            {
                "at": _iso(mark.at),
                "equity_book": str(mark.equity_book),
                "equity_live_mirror": str(mirror) if mirror is not None else None,
                "gap_bps": gap,
                "positions": [
                    {
                        "symbol": p.symbol,
                        "qty": str(p.qty),
                        "demo_mark": str(p.demo_mark),
                        "live_mark": str(p.live_mark) if p.live_mark is not None else None,
                    }
                    for p in mark.positions
                ],
            }
        )
    mirror_arms = [
        _dump(a)
        for a in arms
        if a.spec.kind in (ArmKind.MIRROR_LIVE, ArmKind.WEEKEND_COUNTERFACTUAL)
    ]
    return {
        "note": (
            "equity_live_mirror re-marks the open positions at live prices every hour and keeps "
            "the realised Demo P&L (book/marks.py); the mirror_live arm replays every fill on live "
            "prices (analysis/mirror.py)"
        ),
        "series": series,
        "arms": mirror_arms,
    }


def _report_doc(report: Model | None, what: str) -> dict[str, Any]:
    if report is None:
        return {"status": "not_computed", "reason": f"no {what} was passed to this export"}
    return {"status": "computed", "report": _dump(report)}


# ------------------------------------------------------------------------------------------------
# The venue-integrity replay
# ------------------------------------------------------------------------------------------------


def _series(doc: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    for request in doc["requests"]:
        if request["series"] == name:
            found: Mapping[str, Any] = request
            return found
    raise ExportError(f"the replay recording has no {name} series")


def replay_venue_integrity(policy: Policy = POLICY_V1) -> dict[str, Any]:
    """G1 run on a recorded Demo flash print: BTCPERP, 2026-09-23 13:00-14:00 UTC.

    The recording (``templates/replay_g1_btcperp_20260923.json``) holds Bitget's own keyless
    history-candles responses for that hour: Demo mark, Demo index and Demo market (``paptrading:
    1``) and live market. The Demo quote is built the way the measurement that found the print
    compared them (``validation/demo_venue/demo_integrity.py``): the mark at its hourly extreme
    against the index close of the same hour. G1 then rules on two legs: a proposal to open 5% long,
    and a 5% long already held. This is a replay, not an event of this agent's run, and says so.
    """
    doc = json.loads(REPLAY_FIXTURE.read_text(encoding="utf-8"))
    symbol = str(doc["instrument"])
    hour = datetime.fromisoformat(str(doc["hour"]).replace("Z", "+00:00"))

    def candles(name: str, source: PriceSource, kind: CandleKind) -> list[Candle]:
        rows = _series(doc, name)["response"]["data"]
        parsed = [
            parse_candle(row, symbol=symbol, source=source, kind=kind, interval="1H")
            for row in rows
        ]
        return sorted(parsed, key=lambda c: c.open_time)

    def at_hour(rows: Sequence[Candle]) -> Candle:
        for c in rows:
            if c.open_time == hour:
                return c
        raise ExportError(f"the replay recording has no {hour.isoformat()} candle")

    mark = at_hour(candles("demo_mark", PriceSource.DEMO, "mark"))
    index_rows = candles("demo_index", PriceSource.DEMO, "index")
    index = at_hour(index_rows)
    market = at_hour(candles("demo_market", PriceSource.DEMO, "market"))
    live = at_hour(candles("live_market", PriceSource.LIVE, "market"))
    extreme = mark.high if mark.high / index.close - 1 >= 1 - mark.low / index.close else mark.low
    demo_quote = Quote(
        symbol=symbol,
        source=PriceSource.DEMO,
        ts=hour,
        fetched_at=hour,
        last=market.close,
        mark=extreme,
        index=index.close,
        bid=market.close,
        ask=market.close,
        funding_rate=None,
        open_interest=None,
        turnover_24h=None,
        price_change_24h=None,
    )
    live_quote = demo_quote.model_copy(
        update={
            "source": PriceSource.LIVE,
            "last": live.close,
            "mark": live.close,
            "index": live.close,
            "bid": live.close,
            "ask": live.close,
        }
    )
    move = index_move_bps_3h([c for c in index_rows if c.open_time < hour])
    legs = (
        ("A proposal to open 5% long", Leg(symbol=symbol, current=0.0, proposed=0.05)),
        ("A 5% long already held", Leg(symbol=symbol, current=0.05)),
    )
    rulings = []
    for label, leg in legs:
        ruling = g1_venue_integrity(
            leg,
            entry=policy.entry(symbol),
            demo=demo_quote,
            live=live_quote,
            index_move_bps_3h=move,
            at=hour,
            policy=policy,
        )
        rulings.append({"leg": label, "ruling": _dump(ruling)})
    fetched = sorted(str(r["fetched_at"]) for r in doc["requests"])
    return {
        "label": "REPLAY of recorded Bitget data. Not an event of this agent's run.",
        "what": doc["what"],
        "instrument": symbol,
        "category": doc["category"],
        "hour": _iso(hour),
        "recording": REPLAY_FIXTURE_NAME,
        "recorded_at": fetched[-1] if fetched else None,
        "requests": [
            {
                "series": r["series"],
                "path": r["path"],
                "params": r["params"],
                "headers": r["headers"],
                "fetched_at": r["fetched_at"],
            }
            for r in doc["requests"]
        ],
        "inputs": {
            "demo_mark_high": str(mark.high),
            "demo_mark_low": str(mark.low),
            "demo_mark_used": str(extreme),
            "demo_index_close": str(index.close),
            "demo_index_high": str(index.high),
            "demo_index_low": str(index.low),
            "mark_index_gap_pct": float(abs(extreme / index.close - 1)) * 100,
            "mark_index_gap_pct_against_index_high": float(abs(extreme / index.high - 1)) * 100,
            "demo_last": str(market.close),
            "demo_last_high": str(market.high),
            "demo_last_low": str(market.low),
            "live_last": str(live.close),
            "demo_index_move_bps_3h": move,
            "limit_pct": policy.mark_index_max_gap * 100,
        },
        "rulings": rulings,
        "note": (
            f"{symbol} is a USDC-margined perpetual outside the policy universe, so G11 would "
            "refuse it too and its Demo-live p99 is not measured (G1 notes it as missing). The "
            "replay runs G1 alone on the recorded print: the mark-index check fires on its own. "
            "Even against the index's own high for the hour the gap is beyond the 3% limit."
        ),
    }


# ------------------------------------------------------------------------------------------------
# Summary and verify.md
# ------------------------------------------------------------------------------------------------


def _summary_doc(
    record: _Record,
    book: ArmResult,
    decisions: Mapping[str, Any],
    orders: Mapping[str, Any],
    funnel: Mapping[str, Any],
    bundle: CardBundle,
    generated_at: datetime,
    *,
    feeds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    events = record.events
    genesis = next((e for e in events if e.kind is EventKind.GENESIS), None)
    envelope = (
        Genesis.model_validate(genesis.payload).expected_envelope
        if genesis is not None
        else dict(record.policy.expected_envelope)
    )
    kinds = Counter(e.kind.value for e in events)
    return {
        "project": PROJECT_SLUG,
        "version": __version__,
        "generated_at": _iso(generated_at),
        "mode": record.mode.value if record.mode else None,
        "scored": record.mode is RunMode.PAPER,
        "ledger": {
            "events": len(events),
            "head_seq": events[-1].seq if events else None,
            "head_hash": events[-1].hash if events else ZERO_HASH,
            "first_ts": _iso(events[0].ts) if events else None,
            "last_ts": _iso(events[-1].ts) if events else None,
            "by_kind": dict(sorted(kinds.items())),
            "genesis_hash": genesis.hash if genesis is not None else None,
        },
        "policy_version": record.policy.version,
        "metrics": _dump(book.metrics),
        "expected_envelope": envelope,
        "counts": {
            "cards": len(bundle.cards),
            "decisions": decisions["counts"]["decisions"],
            "flat_with_reasons": decisions["counts"]["flat_with_reasons"],
            "outages": sum(n for k, n in decisions["counts"]["outcomes"].items() if k != "decided"),
            "orders_planned": orders["counts"]["planned"],
            "orders_sent": orders["counts"]["sent"],
            "fills": len(record.fills),
            "closed_trades": len(record.trades),
            "kernel_changed_decisions": decisions["counts"]["changed_by_kernel"],
            "protective_rulings": funnel["protective"]["rulings"],
            "hourly_marks": len(record.marks),
            "feed_alarms_open": (feeds or {}).get("counts", {}).get("open", 0),
            "trigger_kinds_blind_by_failure": (feeds or {})
            .get("counts", {})
            .get("blind_by_failure", 0),
        },
    }


def _verify_md(summary: Mapping[str, Any], genesis: Mapping[str, Any]) -> bytes:
    ledger = summary["ledger"]
    mode = summary["mode"] or "empty"
    lines = [
        "# Verify this record",
        "",
        f"Exported {summary['generated_at']} from the **{mode}** ledger: "
        f"{ledger['events']} events, head hash `{ledger['head_hash']}`.",
        "",
    ]
    if mode != RunMode.PAPER.value:
        lines += [
            f"This is a {mode} ledger, not the scored paper log. Its fills are not Bitget Demo "
            "fills and its order ids cannot be found on any Bitget account.",
            "",
        ]
    lines += [
        "## 1. Recompute every number (standard library only)",
        "",
        "```",
        "python scripts/recompute.py public/",
        "```",
        "",
        "It re-walks the hash chain of `ledger.jsonl` line by line, re-hashes every file in "
        "`blobs/` against its name, checks that `equity_hourly.csv` is exactly the ledger's hourly "
        "`mark` events, rebuilds every closed trade in `trades.csv` from the ledger's `fill` "
        "events, recomputes every field of `metrics.json` (Sharpe, its standard error, Sortino, "
        "max drawdown, win rate, turnover, fees and the 90% block-bootstrap intervals) and every "
        "arm in `arms.json`. Exit code 0 means every published number recomputes.",
        "",
        "## 2. The pre-registration",
        "",
    ]
    if genesis.get("present"):
        policy_hash = genesis["genesis"]["policy_hash"]
        lines += [
            f"- Genesis: seq {genesis['seq']}, hash `{genesis['hash']}`, created "
            f"{genesis['genesis']['created_at']}.",
            f"- Policy hash `{policy_hash}`: the SHA-256 of the canonical JSON of the policy it "
            "carries (recompute.py checks it).",
        ]
        for anchor in genesis.get("genesis_anchors", []):
            record = anchor["record"]
            proof = record.get("ots_blob")
            where = f"`blobs/{proof['sha256']}`" if proof else "no proof stored"
            lines.append(
                f"- OpenTimestamps: {record['status']} at {record['submitted_at']}, proof "
                f"{where}. Check it with `ots verify` against the genesis event's canonical bytes."
            )
        if genesis.get("x_post_text"):
            lines.append("- The genesis hash was posted on X by the owner before the first order.")
    else:
        lines.append("- This ledger has no genesis; it is not a pre-registered run.")
    lines += [
        "",
        "## 3. The orders exist on Bitget Demo",
        "",
        "```",
        "python scripts/verify_orders.py --public public/orders.json",
        "```",
        "",
        "One read-only `order --action detail --orderId <id> --paper-trading` per published order "
        "through Bitget's own Agent Hub CLI. It needs a Demo key for the account that traded, set "
        "up as the script's docstring describes; the owner may publish a read-only one.",
        "",
        "## 4. Replay a decision without any key",
        "",
        "```",
        "t2sa replay --public public/ --decision <decision_id>",
        "```",
        "",
        "It re-runs the logged snapshot, the recorded completion, the kernel and the planner, and "
        "checks that the ruling and order intents hash to what the ledger holds.",
        "",
        "## 5. Every card is its own evidence",
        "",
        "Each `cards/<id>.json` lists the ledger `seq` of every event it was read from and every "
        "blob those events commit to; `sha256(blobs/<hash>)` equals the file name.",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


# ================================================================================================
# The export
# ================================================================================================


def _check_out(out: Path, ledger: HashChainLedger, blobs: FileBlobStore) -> Path:
    target = out.resolve()
    if target.exists() and not target.is_dir():
        raise ExportError(f"{target.name} exists and is not a directory")
    store = blobs.root.resolve()
    working = ledger.path.resolve().parent
    if (target / "blobs").resolve() == store or target.is_relative_to(store):
        raise ExportError("out would write into the working blob store; export somewhere else")
    if target.is_relative_to(working):
        raise ExportError("out would write into the working ledger's folder; export elsewhere")
    target.mkdir(parents=True, exist_ok=True)
    return target


def export_public(
    *,
    ledger: HashChainLedger,
    blobs: FileBlobStore,
    out: Path,
    arms: Sequence[ArmResult],
    twin: TwinReport | None,
    redteam: RedTeamReport | None,
    toolkit: Sequence[ToolkitUse],
    clock: Clock,
    sim_check: SimulatorCheck | None = None,
) -> ExportManifest:
    """Write the public record for ``ledger`` into ``out`` (module docstring).

    ``arms`` are the analysis arms (baselines, rivals, twin, mirror); the governed book's own arm is
    always computed here from the ledger, and an arm passed under its id must agree with it.
    ``toolkit`` is the coverage matrix (``coverage_matrix(probe)``); when it is empty the declared
    rows are used. Raises :class:`ExportError` (nothing published) when the record cannot be
    published honestly, and :class:`ExportRefused` when the scan finds a secret or a local path.
    """
    target = _check_out(Path(out), ledger, blobs)
    generated_at = clock.now()
    record = _read(ledger, blobs)
    try:
        bundle = build_card_bundle(record.projection, blobs)
    except CardError as exc:
        raise ExportError(f"a decision card cannot be built: {exc}") from None
    book, all_arms = _arms(record, arms)

    stage = StagedWrite(target)
    try:
        _ledger_files(stage, record, blobs)
        genesis = _genesis_doc(record)
        stage.json("genesis.json", genesis)
        stage.json("environment.json", _environment_doc(record))
        decisions = _decisions_doc(record, bundle)
        stage.json("decisions.json", decisions)
        for card in bundle.cards:
            name = f"cards/{card.card_id}.json"
            stage.json(name, _dump(card))
        stage.json(
            "cards/index.json",
            [
                {
                    "card_id": c.card_id,
                    "at": _iso(c.at),
                    "decision_id": c.decision_id,
                    "ruling_id": c.ruling_id,
                    "kind": "decision" if c.decision_id else "protective",
                }
                for c in bundle.cards
            ],
        )
        orders = _orders_doc(record, bundle)
        stage.json("orders.json", orders)
        funnel = _funnel_doc(record, bundle)
        stage.json("funnel.json", funnel)
        stage.write("equity_hourly.csv", _equity_csv(record))
        stage.write("trades.csv", _trades_csv(record.trades))
        stage.json("metrics.json", _dump(book.metrics))
        stage.json("arms.json", [_dump(a) for a in all_arms])
        stage.json("arms_summary.json", _arms_summary(book, all_arms, sim_check))
        stage.json("twin.json", _report_doc(twin, "twin report"))
        stage.json("mirror.json", _mirror_doc(record, all_arms))
        stage.json("redteam.json", _report_doc(redteam, "red-team report"))
        rows = ledger_health(tuple(toolkit) or declared_uses(), record.projection)
        stage.json(
            "toolkit.json",
            {"rows": [_dump(r) for r in rows], "counts": coverage_counts(rows)},
        )
        stage.json("replay.json", replay_venue_integrity(record.policy))
        feeds = _feeds_doc(record)
        stage.json("feeds.json", feeds)
        summary = _summary_doc(
            record, book, decisions, orders, funnel, bundle, generated_at, feeds=feeds
        )
        stage.json("summary.json", summary)
        stage.write("verify.md", _verify_md(summary, genesis))

        findings = stage.scan(extra_paths=[target.parent, ledger.path.parent, blobs.root])
        if findings:
            raise ExportRefused(findings)
        stage.publish()
    finally:
        stage.discard()
    written = tuple(stage.written)
    keep = set(written)
    prune(target, "cards", keep, (".json",))
    prune(target, "blobs", keep, ())
    head = record.events[-1].hash if record.events else ZERO_HASH
    return ExportManifest(written=written, ledger_head=head, generated_at=generated_at)


def load_json(path: Path) -> Any:
    """A published JSON file, parsed (for the render and the tests)."""
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "EQUITY_COLUMNS",
    "PLACEHOLDER_EQUITY",
    "REPLAY_FIXTURE",
    "TRADE_COLUMNS",
    "ExportError",
    "ExportManifest",
    "ExportRefused",
    "Finding",
    "StagedWrite",
    "export_public",
    "load_json",
    "local_literals",
    "prune",
    "replay_venue_integrity",
    "scan_bytes",
    "scan_for_secrets",
]
