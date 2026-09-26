"""Wiring: every component of the agent, built once per run mode, its state rebuilt from the log.

:func:`build_app` is the only place that decides which concrete implementation stands behind each
protocol in ``types.py``, and so the only place the three run modes differ (DESIGN.md §5):

=========  =================================  =============================  ===================
Mode       Venue transport                    Credentials read               Ledger (var/ledger/)
=========  =================================  =============================  ===================
SIMULATED  ``SimulatedVenue`` on Demo quotes  none                           ``simulated.jsonl``
DRYRUN     ``bgc --paper-trading --dry-run``  none                           ``dryrun.jsonl``
PAPER      ``bgc --paper-trading``            ``.secrets/demo.env``, Qwen's  ``paper.jsonl``
=========  =================================  =============================  ===================

**PAPER refuses before it can send.** A PAPER app is built only when, in this order: the paper
ledger opens with an intact genesis whose policy is the policy this code loads (or the latest
owner-confirmed amendment's), the Demo credential file exists inside the project and declares
itself Demo, and a fresh environment proof (DESIGN.md §11.2) passes. The proof is logged whatever
its verdict; a failed or refused proof raises
:class:`~sentiment_agent.execution.environment.EnvironmentRefused`, which the CLI turns into exit
code 3. The genesis is checked before the proof is logged, because a genesis may only be seq 0.

**State is rebuilt from the ledger, never carried in memory** (DESIGN.md §4). The book, the breaker,
the trigger history, the day's token spend and every order state are replayed from the log before
the app is returned, and :class:`ProjectingLedger` keeps the projection equal to the log from then
on: every append, by any component, is folded into it, and events another process appended (a
toolkit probe, say) are caught up at the next append or :meth:`App.sync`.

**One instance per mode.** :class:`InstanceLock` holds an operating-system lock on
``var/run/<mode>.lock`` for the life of the app. The OS releases it when the process dies, so a
crash never leaves a lock behind that a restart would have to guess about, and a second instance is
refused while the first lives. (The ledger's own writer lock guards each append; this lock guards
the decision loop, so two loops can never interleave decisions on one book.)

**Injection.** Tests and keyless rehearsals replace components through :class:`Parts`: market
data, the two Bitget services, the crowd collectors, the chat model, the ``bgc`` and ``ots``
runners, the simulated venue and the sleeper. Nothing in :class:`Parts` can weaken a safety rule:
the PAPER checks above run on whatever is injected, and a simulated venue can only be paired with a
SIMULATED ledger (the executor refuses otherwise).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import socket
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import IO, Any, Final, Literal

from sentiment_agent.book.projection import ORDER_EVENT_KINDS, Projection
from sentiment_agent.crowd.adapters import CompositeCrowd, RedditCollector, XCollector, run_command
from sentiment_agent.decision.agent import DecisionAgent
from sentiment_agent.decision.contract import replay_calls
from sentiment_agent.events.triggers import TriggerEngine, oi_jump_thresholds
from sentiment_agent.execution.bgc import BgcRunner, BgcTransport, SubprocessBgcRunner
from sentiment_agent.execution.environment import (
    AGENT_HUB_DIR,
    EnvironmentRefused,
    load_demo_credentials,
    prove_environment,
)
from sentiment_agent.execution.executor import Executor
from sentiment_agent.execution.orders import OrderTracker
from sentiment_agent.execution.reconcile import Reconciler
from sentiment_agent.execution.simulated import SimulatedVenue
from sentiment_agent.execution.stops import StopManager
from sentiment_agent.hashing import canonical_json, sha256_hex
from sentiment_agent.kernel.approval import approve
from sentiment_agent.kernel.breaker import Breaker
from sentiment_agent.kernel.kernel import RiskKernel
from sentiment_agent.kernel.planner import plan_orders
from sentiment_agent.ledger.anchor import OtsRunner, subprocess_ots_runner
from sentiment_agent.ledger.blobs import FileBlobStore
from sentiment_agent.ledger.chain import GenesisError, HashChainLedger, LedgerError, ledger_path
from sentiment_agent.ledger.genesis import require_genesis
from sentiment_agent.llm.budget import DailyTokenBudget
from sentiment_agent.llm.client import QwenChatModel, load_qwen_env
from sentiment_agent.llm.fakes import RecordedChatModel, ScriptedChatModel
from sentiment_agent.perception.snapshot import SnapshotBuilder
from sentiment_agent.policy import ACTIVE_POLICY
from sentiment_agent.sources.bitget_data import (
    ENTRY_OPEN_INTEREST,
    EXCHANGE,
    INTERVAL,
    BitgetDataService,
)
from sentiment_agent.sources.mcp_http import (
    DATA_MCP_URL,
    SIGNAL_MCP_URL,
    StreamableHttpMcp,
    finite,
    row_time,
)
from sentiment_agent.sources.signal_skills import SignalSkills
from sentiment_agent.sources.toolkit import ToolkitFacade
from sentiment_agent.sources.upstream import UpstreamDirect
from sentiment_agent.types import (
    ApprovedOrder,
    AssetClass,
    BlobRef,
    BookState,
    ChatModel,
    Clock,
    CrowdCollector,
    EnvironmentProof,
    EventKind,
    Genesis,
    InstrumentSpec,
    KernelInputs,
    KernelRuling,
    LedgerEvent,
    MarketData,
    Model,
    Note,
    OrderPlan,
    OrderState,
    PerceptionSnapshot,
    Policy,
    PriceSource,
    Quote,
    RunMode,
    ToolkitReader,
    VenueTransport,
)
from sentiment_agent.venue.public_api import BitgetPublicApi

LlmChoice = Literal["live", "scripted", "recorded"]
"""Which chat model decides: ``live`` Qwen (needs ``.secrets/qwen.env``), a ``scripted`` stand-in
(tests and keyless rehearsals), or ``recorded`` completions replayed from this ledger's blobs."""

DEFAULT_SIMULATED_EQUITY: Final = Decimal("10000")
"""The notional USDT equity of a SIMULATED or DRYRUN book, which has no venue account to read.
Recorded in the ledger at the first start (a reconciliation's account read, or a note)."""

OI_THRESHOLDS_MEDIA_TYPE: Final = "application/vnd.t2sa.oi-thresholds+json"
"""The open-interest jump thresholds frozen at genesis, attached to the GENESIS event itself so the
genesis hash commits to them (DESIGN.md §8)."""

RULING_CONTEXT_MEDIA_TYPE: Final = "application/vnd.t2sa.ruling-context+json"
"""Everything a kernel ruling read (book, market inputs, breaker state, context, guards), attached
to its KERNEL_RULING event so ``t2sa replay`` can re-run the kernel and the planner keylessly."""

PREGENESIS_LEDGER: Final = "var/ledger/paper-pregenesis.jsonl"
"""PAPER records made before the genesis exists: environment proofs, toolkit probes and the
owner-approved plumbing test (DESIGN.md §16). Never scored, always disclosed."""

STARTUP_RECONCILE_LOOKBACK: Final = timedelta(days=1)

X_SEARCHES_IN_FLIGHT: Final = 2
REDDIT_SEARCHES_IN_FLIGHT: Final = 3
"""Crowd searches kept in flight per CLI on a full snapshot. Measured on 2026-09-24 against the live
services from this machine: one ``rdt search`` takes 12-15 s and one ``twitter search`` 2.5-9 s, so
fourteen of each run one after another made the crowd read alone take about 4.5 minutes and the
whole full snapshot about 6, most of G10's 15-minute snapshot age budget before the model had
started. Both CLIs are personal logged-in accounts, so the widths stay small: the number of
searches per snapshot is unchanged, only how many overlap."""


class RefusedToStart(RuntimeError):  # noqa: N818 - reads as the fact the CLI reports
    """A precondition of the requested mode is not met. The message says which, and the fix."""


class InstanceLocked(RefusedToStart):
    """Another live process holds this mode's instance lock."""


# ================================================================================================
# Paths
# ================================================================================================


@dataclass(frozen=True, slots=True)
class Paths:
    """Every path the runtime reads or writes, from one project root."""

    root: Path
    mode: RunMode

    @property
    def var(self) -> Path:
        return self.root / "var"

    @property
    def ledger(self) -> Path:
        return ledger_path(self.root, self.mode)

    @property
    def pregenesis_ledger(self) -> Path:
        return self.root.joinpath(*PREGENESIS_LEDGER.split("/"))

    @property
    def blobs(self) -> Path:
        return self.var / "blobs"

    @property
    def lock(self) -> Path:
        return self.var / "run" / f"{self.mode.value}.lock"

    @property
    def health(self) -> Path:
        return self.var / "health" / f"{self.mode.value}.json"

    @property
    def inbox(self) -> Path:
        return self.var / "inbox" / self.mode.value

    @property
    def anchors(self) -> Path:
        return self.var / "anchors"

    @property
    def public(self) -> Path:
        return self.root / "public"

    @property
    def agent_hub(self) -> Path:
        return self.root.joinpath(*AGENT_HUB_DIR.split("/"))

    @property
    def secrets(self) -> Path:
        return self.root / ".secrets"

    def ensure(self) -> None:
        for directory in (
            self.ledger.parent,
            self.blobs,
            self.lock.parent,
            self.health.parent,
            self.inbox,
            self.anchors,
        ):
            directory.mkdir(parents=True, exist_ok=True)


# ================================================================================================
# The single-instance lock
# ================================================================================================

_LOCK_OFFSET: Final = 1 << 16
"""The locked byte sits far past the holder record, so another process can still read who holds the
lock (Windows locks are mandatory for the region they cover).

``scripts/publish_site.ps1`` takes the same byte of ``var/run/export-<mode>.lock`` while it copies
``public/``, so the page's publisher and the hourly export never touch that folder at once."""

LOCK_POLL_S: Final = 0.5
"""How often :meth:`InstanceLock.acquire` retries a held lock while it is willing to wait."""


class InstanceLock:
    """An operating-system lock on one file, held until :meth:`release` or process death.

    ``fcntl.flock`` on POSIX, ``msvcrt.locking`` on Windows: both are released by the kernel when
    the holder dies, so a stale lock is impossible and staleness is never guessed from a clock. The
    holder record (pid, host, mode, start time) is written beside the lock for the refusal message.
    """

    def __init__(self, path: Path, *, mode: RunMode, clock: Clock, busy: str | None = None) -> None:
        self._path = path
        self._mode = mode
        self._clock = clock
        self._busy = busy
        """What the refusal says holds the lock, when it is not another decision loop."""
        self._handle: IO[bytes] | None = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def held(self) -> bool:
        return self._handle is not None

    def holder(self) -> str:
        """Who holds the lock, as its record says (best effort)."""
        try:
            text = self._path.read_bytes()[:4096].decode("utf-8", "replace").strip("\x00 \n")
        except OSError:
            return "unknown holder"
        try:
            record = json.loads(text)
        except ValueError:
            return "unknown holder"
        if not isinstance(record, dict):
            return "unknown holder"
        return (
            f"pid {record.get('pid')} on {record.get('host')}, {record.get('mode')} run started "
            f"{record.get('started_at')}"
        )

    def acquire(self, *, wait_s: float = 0.0) -> None:
        """Take the lock, or raise :class:`InstanceLocked`. With ``wait_s`` above zero it keeps
        retrying for that long first: the export waits out the publisher's copy rather than
        losing its hour."""
        if self._handle is not None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + wait_s
        while True:
            fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o644)
            handle = os.fdopen(fd, "r+b", buffering=0)
            try:
                _os_lock(handle)
                break
            except OSError:
                handle.close()
                if time.monotonic() < deadline:
                    time.sleep(LOCK_POLL_S)
                    continue
            if self._busy is not None:
                raise InstanceLocked(f"{self._busy} ({self.holder()}), after waiting {wait_s:g} s")
            raise InstanceLocked(
                f"another {self._mode.value} instance is running ({self.holder()}); one decision "
                f"loop per mode at a time. Stop it first, or use `t2sa status`"
            ) from None
        record = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "mode": self._mode.value,
            "started_at": self._clock.now().isoformat(),
        }
        handle.seek(0)
        handle.truncate(0)
        handle.write(json.dumps(record, sort_keys=True).encode("utf-8") + b"\n")
        self._handle = handle

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        with contextlib.suppress(OSError):
            _os_unlock(handle)
        handle.close()


if sys.platform == "win32":
    import msvcrt

    def _os_lock(handle: IO[bytes]) -> None:
        handle.seek(_LOCK_OFFSET)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

    def _os_unlock(handle: IO[bytes]) -> None:
        handle.seek(_LOCK_OFFSET)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _os_lock(handle: IO[bytes]) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _os_unlock(handle: IO[bytes]) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


# ================================================================================================
# The ledger, projected
# ================================================================================================


class ProjectingLedger:
    """The hash chain, with every appended event folded into the projection as it is written.

    Satisfies :class:`types.LedgerWriter` and :class:`types.LedgerReader`, so the executor and the
    reconciler write through it unchanged. Events appended by another process are applied, in order,
    before the next one of ours, so the projection always equals the file.
    """

    def __init__(self, chain: HashChainLedger, projection: Projection) -> None:
        self._chain = chain
        self._projection = projection
        self.appended: list[LedgerEvent] = []
        """Every event this object appended, in order. Cycles slice it to cite their own events."""

    @property
    def mode(self) -> RunMode:
        return self._chain.mode

    @property
    def chain(self) -> HashChainLedger:
        return self._chain

    def append(
        self, kind: EventKind, payload: Model, *, blobs: Sequence[BlobRef] = ()
    ) -> LedgerEvent:
        event = self._chain.append(kind, payload, blobs=blobs)
        self.sync(through=event)
        self.appended.append(event)
        return event

    def events(self, kinds: frozenset[EventKind] | None = None) -> Iterator[LedgerEvent]:
        return self._chain.events(kinds)

    def head(self) -> LedgerEvent | None:
        return self._chain.head()

    def sync(self, *, through: LedgerEvent | None = None) -> int:
        """Apply every event the projection has not seen. Returns how many were applied."""
        seen = self._projection.head_seq
        if through is not None and ((seen is None and through.seq == 0) or seen == through.seq - 1):
            self._projection.apply(through)
            return 1
        applied = 0
        for event in self._chain.events():
            if seen is not None and event.seq <= seen:
                continue
            self._projection.apply(event)
            applied += 1
        return applied


# ================================================================================================
# Planner and approval, as one component
# ================================================================================================


@dataclass(frozen=True, slots=True)
class Planner:
    """The planner and the approval minter the runtime calls in sequence (DESIGN.md §10.7)."""

    policy: Policy

    def plan(
        self, ruling: KernelRuling, book: BookState, inputs: KernelInputs, *, now: datetime
    ) -> OrderPlan:
        return plan_orders(ruling, book, inputs, self.policy, now=now)

    def approve(
        self, plan: OrderPlan, ruling: KernelRuling, book: BookState, inputs: KernelInputs
    ) -> tuple[ApprovedOrder, ...]:
        return approve(plan, ruling, book, inputs)


# ================================================================================================
# Injection
# ================================================================================================


@dataclass(frozen=True)
class Parts:
    """Components to use instead of the defaults. Every field is optional."""

    policy: Policy | None = None
    market: MarketData | None = None
    toolkit: ToolkitReader | None = None
    crowd: CrowdCollector | None = None
    chat_model: ChatModel | None = None
    scripted: Sequence[Any] = ()
    """Completions or exceptions for a ``scripted`` model (unused when ``chat_model`` is given)."""
    venue: SimulatedVenue | None = None
    """A simulated venue to trade on. Passing the same one to a restarted app is how a test models
    a venue that outlives our process, as a real one does."""
    bgc_runner: BgcRunner | None = None
    ots_runner: OtsRunner | None = None
    sleep: Callable[[float], None] | None = None
    starting_equity: Decimal = DEFAULT_SIMULATED_EQUITY
    oi_thresholds: Mapping[str, float] | None = None
    """Override the thresholds frozen at genesis. Only for a ledger without a genesis."""
    poll_timeout_s: float = 60.0
    poll_interval_s: float = 2.0


# ================================================================================================
# The app
# ================================================================================================


@dataclass
class App:
    """Every constructed component of one run, and the state rebuilt from its ledger."""

    mode: RunMode
    llm: LlmChoice
    policy: Policy
    paths: Paths
    clock: Clock
    chain: HashChainLedger
    ledger: ProjectingLedger
    blobs: FileBlobStore
    projection: Projection
    market: MarketData
    toolkit: ToolkitReader
    crowd: CrowdCollector
    snapshots: SnapshotBuilder
    triggers: TriggerEngine
    chat_model: ChatModel
    budget: DailyTokenBudget
    agent: DecisionAgent
    kernel: RiskKernel
    breaker: Breaker
    planner: Planner
    transport: VenueTransport
    simulated_venue: SimulatedVenue | None
    tracker: OrderTracker
    executor: Executor
    reconciler: Reconciler
    stops: StopManager | None
    genesis: Genesis | None
    genesis_event: LedgerEvent | None
    proof: EnvironmentProof | None
    specs: dict[str, InstrumentSpec]
    oi_thresholds: dict[str, float]
    lock: InstanceLock
    ots_runner: OtsRunner
    sleep: Callable[[float], None]
    started_at: datetime
    resumed_from_seq: int | None
    """The ledger head before this start appended anything (``None`` for an empty ledger)."""
    resumed_from_ts: datetime | None
    startup_notes: list[str] = field(default_factory=list)

    # --- convenience --------------------------------------------------------------------------

    @property
    def symbols(self) -> tuple[str, ...]:
        return self.policy.symbols

    def log(self, kind: EventKind, payload: Model, *, blobs: Sequence[BlobRef] = ()) -> LedgerEvent:
        return self.ledger.append(kind, payload, blobs=blobs)

    def note(self, text: str, *, author: Literal["system", "owner"] = "system") -> LedgerEvent:
        """Log a note, with every local path and credential variable name scrubbed from it
        (:func:`scrub_note`): a note is published, and a line of the ledger can never be edited."""
        clean = scrub_note(text, self.paths.root)
        return self.log(EventKind.NOTE, Note(at=self.clock.now(), author=author, text=clean))

    def sync(self) -> int:
        """Fold in events another process appended since our last append."""
        return self.ledger.sync()

    def close(self) -> None:
        self.lock.release()

    def __enter__(self) -> App:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- market facts the kernel rules on ------------------------------------------------------

    def rebuild_stops(self) -> None:
        """(Re)build the stop manager on the current instrument specs. None in DRYRUN, which
        holds no position and reads no venue stop."""
        if self.mode is RunMode.DRYRUN:
            self.stops = None
            return
        venue: BgcTransport | SimulatedVenue = (
            self.simulated_venue if self.simulated_venue is not None else _bgc(self.transport)
        )
        self.stops = StopManager(
            transport=venue, policy=self.policy, clock=self.clock, specs=self.specs
        )

    def refresh_specs(self) -> dict[str, InstrumentSpec]:
        """Re-read the Demo instrument limits (keyless). Keeps the last good set on failure."""
        try:
            fresh = self.market.instruments(PriceSource.DEMO, self.symbols)
        except Exception as exc:  # a failed keyless read must not stop the loop; G11 fails closed
            self.startup_notes.append(f"instrument specs unreadable: {type(exc).__name__}: {exc}")
            return self.specs
        kept = {
            s: spec
            for s, spec in fresh.items()
            if s in self.symbols and spec.symbol == s and spec.source is PriceSource.DEMO
        }
        if kept:
            self.specs = kept
        return self.specs

    def quotes(self) -> tuple[dict[str, Quote], dict[str, Quote]]:
        """Fresh keyless Demo and live quotes for the universe; a failed read is an empty map."""
        demo = _safe_quotes(self.market, PriceSource.DEMO, self.symbols)
        live = _safe_quotes(self.market, PriceSource.LIVE, self.symbols)
        return demo, live

    def kernel_inputs(
        self,
        *,
        at: datetime,
        demo: Mapping[str, Quote],
        live: Mapping[str, Quote],
        snapshot: PerceptionSnapshot | None,
    ) -> KernelInputs:
        """What the kernel rules on. The Demo index moves and the snapshot freshness come from the
        latest logged snapshot, so a kernel ruling between snapshots cites the one it relied on;
        whether the book is the venue's comes from the latest reconciliation
        (:meth:`venue_unreconciled`)."""
        moves: dict[str, float | None] = {}
        if snapshot is not None:
            moves = {s: f.demo_index_move_bps_3h for s, f in snapshot.features.items()}
        return KernelInputs(
            venue_unreconciled=self.venue_unreconciled(),
            at=at,
            demo_quotes={s: q for s, q in demo.items() if q.source is PriceSource.DEMO},
            live_quotes={s: q for s, q in live.items() if q.source is PriceSource.LIVE},
            specs=dict(self.specs),
            demo_index_move_bps_3h=moves,
            snapshot_id=None if snapshot is None else snapshot.snapshot_id,
            snapshot_taken_at=None if snapshot is None else snapshot.taken_at,
        )

    def venue_unreconciled(self) -> tuple[str, ...]:
        """Why the ledger's book may not be the venue's, from the latest reconciliation.

        DRYRUN never sends, so its book is the venue's by construction. Otherwise the latest sweep's
        ``missing_fill`` and ``position_mismatch`` discrepancies (including a fills or positions
        read that failed) are returned; with no sweep logged yet and an order already sent, that
        absence is itself the reason. Fail-closed: the kernel's G10 refuses every increase while
        this is
        non-empty."""
        if self.mode is RunMode.DRYRUN:
            return ()
        reports = self.projection.reconciliations
        if not reports:
            unsent = (OrderState.DENIED, OrderState.INITIALISED, None)
            if any(self.tracker.state(oid) not in unsent for oid in self.tracker.known()):
                return ("no reconciliation has been logged since orders were sent",)
            return ()
        return reports[-1].unreconciled

    def latest_snapshot(self) -> PerceptionSnapshot | None:
        snapshots = self.projection.snapshots
        return snapshots[-1] if snapshots else None

    def book(self, *, at: datetime, demo: Mapping[str, Quote]) -> BookState:
        """The book at ``at`` from the ledger, valued at the Demo marks.

        A held symbol without a fresh Demo quote is valued at the newest Demo mark the ledger holds
        for it (the latest snapshot, then the latest hourly mark), and at its entry price only when
        the ledger holds none; the valuation used is visible in ``BookState.marks``.
        """
        marks = {s: q.mark for s, q in demo.items() if q.mark > 0}
        held = self.held_symbols()
        missing = [s for s in held if s not in marks]
        if missing:
            marks.update(self._fallback_marks(missing))
        return self.projection.book(at=at, marks=marks, mark_source=PriceSource.DEMO)

    def held_symbols(self) -> tuple[str, ...]:
        if self.projection.starting_equity is None:
            return ()
        return tuple(sorted(self.projection.builder().positions()))

    def _fallback_marks(self, symbols: Sequence[str]) -> dict[str, Decimal]:
        out: dict[str, Decimal] = {}
        snapshot = self.latest_snapshot()
        marks = self.projection.marks
        positions = self.projection.builder().positions()
        for symbol in symbols:
            quote = snapshot.demo_quotes.get(symbol) if snapshot is not None else None
            if quote is not None and quote.mark > 0:
                out[symbol] = quote.mark
                continue
            found: Decimal | None = None
            for point in reversed(marks):
                row = next((p for p in point.positions if p.symbol == symbol), None)
                if row is not None and row.demo_mark > 0:
                    found = row.demo_mark
                    break
            if found is None:
                found = positions[symbol].avg_entry
            out[symbol] = found
        return out


_DRIVE_PATH: Final = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:(?:\\\\|\\|/)[^\s\"'<>|*?]*")
_USER_PATH: Final = re.compile(
    r"(?<![A-Za-z0-9_.:/-])/(?:Users|home|root|tmp|var|private|mnt)/[^\s\"'<>]*"
)
_FILE_URL: Final = re.compile(r"(?i)\bfile:/\S*")
NOTE_MAX_CHARS: Final = 4000


def scrub_note(text: str, *roots: Path) -> str:
    """``text`` made safe to publish: local paths (the project root, the home, working and
    temporary directories, any drive or user path, any ``file:`` URL) become ``<local path>``, and
    the credential-variable prefix is spelled ``BITGET-`` so a note that names a variable never
    reads as one.

    Notes carry exception text (a failed tick, a failed export, an unreadable read), and exception
    text carries paths. The export refuses to publish a record containing a local path or a
    ``BITGET_`` name (``site/export.py``), and a ledger line cannot be edited afterwards, so one
    such note would block every later export. Scrubbing here keeps what happened and drops only
    where on this machine it happened.
    """
    known: list[Path] = [*roots]
    with contextlib.suppress(RuntimeError, KeyError):
        known.append(Path.home())
    with contextlib.suppress(OSError):
        known.append(Path.cwd())
    known.append(Path(tempfile.gettempdir()))
    for base in known:
        with contextlib.suppress(OSError):
            base = base.resolve()
        if len(base.parts) < 2:
            continue
        raw = str(base)
        for form in {raw, base.as_posix(), raw.replace("\\", "\\\\")}:
            text = re.sub(re.escape(form), "<local path>", text, flags=re.IGNORECASE)
    text = _FILE_URL.sub("<local path>", text)
    text = _DRIVE_PATH.sub("<local path>", text)
    text = _USER_PATH.sub("<local path>", text)
    text = re.sub(r"BITGET_", "BITGET-", text, flags=re.IGNORECASE)
    return text[:NOTE_MAX_CHARS]


def _safe_quotes(
    market: MarketData, source: PriceSource, symbols: Sequence[str]
) -> dict[str, Quote]:
    try:
        got = market.quotes(source, symbols)
    except Exception:  # keyless reads degrade to "no quote"; every guard fails closed on a gap
        return {}
    return {s: q for s, q in got.items() if s in symbols and q.symbol == s and q.source is source}


# ================================================================================================
# Genesis extras: the frozen open-interest thresholds
# ================================================================================================


def oi_history(
    data: BitgetDataService, symbols: Sequence[str], *, days: int
) -> tuple[dict[str, list[tuple[datetime, float]]], dict[str, str]]:
    """Hourly open-interest history per symbol from bitget-mcp-server, and a note per symbol.

    The trailing ``days`` are asked for (``24 * days`` hourly rows); what the service returns is
    what the threshold is computed from, and the note says how much that was.
    """
    history: dict[str, list[tuple[datetime, float]]] = {}
    notes: dict[str, str] = {}
    for symbol in symbols:
        rows, call = data.query(
            ENTRY_OPEN_INTEREST,
            symbol=symbol,
            interval=INTERVAL,
            limit=str(24 * days),
            exchange=EXCHANGE,
        )
        series: dict[datetime, float] = {}
        for row in rows:
            moment = row_time(row)
            value = finite(row.get("open_interest"))
            if moment is not None and value is not None and value > 0:
                series[moment] = value
        history[symbol] = sorted(series.items())
        notes[symbol] = (
            f"{len(series)} hourly points from {call.source} ({call.health.value}"
            + (f": {call.error}" if call.error else "")
            + ")"
        )
    return history, notes


def frozen_oi_thresholds(
    history: Mapping[str, Sequence[tuple[datetime, float]]],
    notes: Mapping[str, str],
    *,
    policy: Policy,
    at: datetime,
) -> dict[str, Any]:
    """The genesis record of the open-interest thresholds: values, inputs and what was left out."""
    rule = policy.triggers
    since = at - timedelta(days=rule.oi_jump_lookback_days)
    trimmed = {s: [(t, v) for t, v in series if since <= t <= at] for s, series in history.items()}
    thresholds = oi_jump_thresholds(trimmed, quantile=rule.oi_jump_quantile)
    return {
        "version": 1,
        "computed_at": at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "quantile": rule.oi_jump_quantile,
        "lookback_days": rule.oi_jump_lookback_days,
        "thresholds_pct": {s: thresholds[s] for s in sorted(thresholds)},
        "points": {s: len(trimmed[s]) for s in sorted(trimmed)},
        "notes": {s: notes.get(s, "") for s in sorted(trimmed)},
        "disabled": sorted(s for s in trimmed if s not in thresholds),
        "rule": "absolute 1h open-interest change above this percent fires open_interest_jump; a "
        "symbol with fewer 1h changes than the quantile supports is disabled, never estimated",
    }


def thresholds_from_genesis(event: LedgerEvent, blobs: FileBlobStore) -> dict[str, float] | None:
    """The thresholds the genesis event committed to, or ``None`` when it carries none."""
    for ref in event.blobs:
        if ref.media_type != OI_THRESHOLDS_MEDIA_TYPE:
            continue
        data = blobs.get(ref.sha256)
        record = json.loads(data.decode("utf-8"))
        values = record.get("thresholds_pct", {}) if isinstance(record, dict) else {}
        return {str(k): float(v) for k, v in values.items()}
    return None


def crypto_symbols(policy: Policy) -> tuple[str, ...]:
    return tuple(u.symbol for u in policy.universe if u.asset_class is AssetClass.CRYPTO)


def default_data_service(clock: Clock, blobs: FileBlobStore | None) -> BitgetDataService:
    return BitgetDataService(
        StreamableHttpMcp(DATA_MCP_URL, server_label="bitget-mcp-server", clock=clock, blobs=blobs),
        clock,
    )


def default_toolkit(clock: Clock, blobs: FileBlobStore | None) -> ToolkitFacade:
    signal = SignalSkills(
        StreamableHttpMcp(SIGNAL_MCP_URL, server_label="bitget-signal", clock=clock, blobs=blobs),
        clock,
    )
    return ToolkitFacade(
        signal, default_data_service(clock, blobs), upstream=UpstreamDirect(clock, blobs=blobs)
    )


# ================================================================================================
# Building
# ================================================================================================


def open_chain(root: Path, mode: RunMode, clock: Clock) -> HashChainLedger:
    path = ledger_path(root, mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    return HashChainLedger(path, mode=mode, clock=clock)


def open_pregenesis(root: Path, clock: Clock) -> HashChainLedger:
    paths = Paths(root=root, mode=RunMode.PAPER)
    paths.pregenesis_ledger.parent.mkdir(parents=True, exist_ok=True)
    return HashChainLedger(paths.pregenesis_ledger, mode=RunMode.PAPER, clock=clock)


def genesis_of(chain: HashChainLedger) -> tuple[LedgerEvent, Genesis] | None:
    head = next(iter(chain.events(frozenset({EventKind.GENESIS}))), None)
    if head is None:
        return None
    return head, Genesis.model_validate(head.payload)


def build_app(
    root: Path,
    mode: RunMode,
    *,
    llm: LlmChoice,
    clock: Clock,
    parts: Parts | None = None,
) -> App:
    """Build every component for ``mode``, rebuild state from its ledger, and hold its lock.

    Raises :class:`InstanceLocked` when another instance of the mode runs,
    :class:`~sentiment_agent.ledger.chain.GenesisError` when a pre-registration is missing or does
    not match the loaded policy, :class:`EnvironmentRefused` when PAPER cannot prove it trades on
    Demo, and :class:`RefusedToStart` for any other unmet precondition. Nothing is sent by building.
    """
    parts = parts or Parts()
    mode = RunMode(mode)
    root = Path(root).resolve()
    paths = Paths(root=root, mode=mode)
    paths.ensure()
    policy = parts.policy or ACTIVE_POLICY
    lock = InstanceLock(paths.lock, mode=mode, clock=clock)
    lock.acquire()
    try:
        return _build(paths, mode, llm=llm, clock=clock, parts=parts, policy=policy, lock=lock)
    except BaseException:
        lock.release()
        raise


def _build(
    paths: Paths,
    mode: RunMode,
    *,
    llm: LlmChoice,
    clock: Clock,
    parts: Parts,
    policy: Policy,
    lock: InstanceLock,
) -> App:
    root = paths.root
    notes: list[str] = []
    blobs = FileBlobStore(paths.blobs)
    chain = HashChainLedger(paths.ledger, mode=mode, clock=clock)
    head = chain.head()
    resumed_seq = None if head is None else head.seq
    resumed_ts = None if head is None else head.ts

    found = genesis_of(chain)
    genesis_event: LedgerEvent | None = None
    genesis: Genesis | None = None
    if found is not None:
        genesis_event, _ = found
        genesis = require_genesis(chain, policy)
    elif mode is RunMode.PAPER:
        raise GenesisError(
            "the paper ledger has no genesis: nothing is pre-registered, so nothing may be sent. "
            "Run `t2sa prove-env`, then `t2sa genesis` (DESIGN.md §16)"
        )

    fallback_equity = None if mode is RunMode.PAPER else parts.starting_equity
    projection = Projection.from_ledger(chain, policy, starting_equity=fallback_equity)
    ledger = ProjectingLedger(chain, projection)

    # --- the venue, and the proof that PAPER trades on Demo --------------------------------------
    proof: EnvironmentProof | None = None
    simulated: SimulatedVenue | None = None
    transport: VenueTransport
    market = parts.market or BitgetPublicApi(clock=clock, blobs=blobs)
    if mode is RunMode.SIMULATED:
        simulated = parts.venue or SimulatedVenue(
            market=market, clock=clock, starting_equity=parts.starting_equity
        )
        transport = simulated
    elif mode is RunMode.DRYRUN:
        runner = parts.bgc_runner or _subprocess_runner(paths)
        transport = BgcTransport(
            runner=runner, clock=clock, blobs=blobs, credentials=None, proof=None, dry_run_only=True
        )
    else:
        runner = parts.bgc_runner or _subprocess_runner(paths)
        credentials = load_demo_credentials(root)
        try:
            proof = prove_environment(root, runner=runner, clock=clock, blobs=blobs)
        except EnvironmentRefused as refused:
            if refused.proof is not None:
                ledger.append(EventKind.ENVIRONMENT_PROOF, refused.proof)
            raise
        ledger.append(EventKind.ENVIRONMENT_PROOF, proof)
        if not proof.passed:
            raise EnvironmentRefused(
                "the environment proof did not pass: " + "; ".join(proof.reasons), proof=proof
            )
        if projection.starting_equity is None:
            raise RefusedToStart(
                "the Demo account equity could not be read (the proof's account carries none), "
                "so the paper book has no starting equity; nothing is sent until it can be read"
            )
        transport = BgcTransport(
            runner=runner,
            clock=clock,
            blobs=blobs,
            credentials=credentials,
            proof=proof,
            dry_run_only=False,
        )

    # --- perception --------------------------------------------------------------------------
    toolkit = parts.toolkit or default_toolkit(clock, blobs)
    crowd = parts.crowd or CompositeCrowd(
        [
            XCollector(runner=run_command, clock=clock, concurrency=X_SEARCHES_IN_FLIGHT),
            RedditCollector(runner=run_command, clock=clock, concurrency=REDDIT_SEARCHES_IN_FLIGHT),
        ],
        parallel=True,
    )
    snapshots = SnapshotBuilder(
        market=market,
        toolkit=toolkit,
        crowd=crowd,
        policy=policy,
        clock=clock,
        mode=mode,
        concurrent_text=True,
    )

    # --- triggers, with the thresholds frozen at genesis -------------------------------------
    thresholds: dict[str, float] = {}
    if genesis_event is not None:
        frozen = thresholds_from_genesis(genesis_event, blobs)
        if frozen is None:
            notes.append(
                "the genesis froze no open-interest thresholds: open_interest_jump never fires"
            )
        else:
            thresholds = frozen
        if parts.oi_thresholds is not None:
            raise RefusedToStart(
                "open-interest thresholds are frozen at genesis; they cannot be overridden"
            )
    elif parts.oi_thresholds is not None:
        thresholds = dict(parts.oi_thresholds)
    else:
        notes.append("no genesis: open_interest_jump has no frozen threshold and never fires")
    triggers = TriggerEngine(policy, clock, oi_thresholds=thresholds)
    triggers.restore(projection.triggers)

    # --- the model ---------------------------------------------------------------------------
    budget = DailyTokenBudget(policy.decision.daily_token_cap, clock)
    budget.restore(projection.budget_states)
    chat_model = parts.chat_model or _chat_model(
        llm,
        root=root,
        budget=budget,
        clock=clock,
        blobs=blobs,
        projection=projection,
        parts=parts,
        policy=policy,
    )
    agent = DecisionAgent(model=chat_model, policy=policy, blobs=blobs, clock=clock)

    # --- kernel, breaker, orders -------------------------------------------------------------
    kernel = RiskKernel(policy, clock)
    breaker = Breaker(policy, clock)
    breaker.restore(projection.breaker_transitions)
    tracker = OrderTracker()
    tracker.restore(projection.events((*ORDER_EVENT_KINDS, EventKind.FILL)))
    sleep = parts.sleep or time.sleep
    executor = Executor(
        transport=transport,
        ledger=ledger,
        tracker=tracker,
        clock=clock,
        poll_timeout_s=parts.poll_timeout_s,
        poll_interval_s=parts.poll_interval_s,
        sleep=sleep,
    )
    reconciler = Reconciler(transport=transport, ledger=ledger, tracker=tracker, clock=clock)

    app = App(
        mode=mode,
        llm=llm,
        policy=policy,
        paths=paths,
        clock=clock,
        chain=chain,
        ledger=ledger,
        blobs=blobs,
        projection=projection,
        market=market,
        toolkit=toolkit,
        crowd=crowd,
        snapshots=snapshots,
        triggers=triggers,
        chat_model=chat_model,
        budget=budget,
        agent=agent,
        kernel=kernel,
        breaker=breaker,
        planner=Planner(policy),
        transport=transport,
        simulated_venue=simulated,
        tracker=tracker,
        executor=executor,
        reconciler=reconciler,
        stops=None,
        genesis=genesis,
        genesis_event=genesis_event,
        proof=proof,
        specs={},
        oi_thresholds=thresholds,
        lock=lock,
        ots_runner=parts.ots_runner or subprocess_ots_runner,
        sleep=sleep,
        started_at=clock.now(),
        resumed_from_seq=resumed_seq,
        resumed_from_ts=resumed_ts,
        startup_notes=notes,
    )
    app.refresh_specs()
    app.rebuild_stops()
    _check_simulated_resume(app, parts)
    return app


def _bgc(transport: VenueTransport) -> BgcTransport:
    if not isinstance(transport, BgcTransport):  # pragma: no cover - PAPER always builds one
        raise RefusedToStart("a paper app trades through Agent Hub only")
    return transport


def _subprocess_runner(paths: Paths) -> SubprocessBgcRunner:
    try:
        return SubprocessBgcRunner(paths.agent_hub)
    except RuntimeError as exc:
        raise RefusedToStart(f"Agent Hub is not installed: {exc}. Run `t2sa setup`") from None


def _chat_model(
    llm: LlmChoice,
    *,
    root: Path,
    budget: DailyTokenBudget,
    clock: Clock,
    blobs: FileBlobStore,
    projection: Projection,
    parts: Parts,
    policy: Policy,
) -> ChatModel:
    if llm == "live":
        try:
            credentials = load_qwen_env(root)
        except RuntimeError as exc:
            raise RefusedToStart(f"live Qwen needs .secrets/qwen.env: {exc}") from None
        return QwenChatModel(
            credentials=credentials,
            budget=budget,
            clock=clock,
            blobs=blobs,
            timeout_s=float(policy.decision.call_timeout_seconds),
        )
    if llm == "scripted":
        return ScriptedChatModel(parts.scripted, model_name=policy.decision.model, budget=budget)
    recordings = [r for d in projection.decisions for r in replay_calls(d.call, blobs)]
    return RecordedChatModel(recordings, model_name=policy.decision.model)


def _check_simulated_resume(app: App, parts: Parts) -> None:
    """A simulated venue lives in this process. Resuming a SIMULATED ledger that holds positions
    or live orders on a fresh one would pair a book with a venue that never saw its fills, so it is
    refused; a test that keeps the venue passes it in :class:`Parts`."""
    if app.mode is not RunMode.SIMULATED or parts.venue is not None:
        return
    held = app.held_symbols()
    live = app.tracker.live()
    if held or live:
        raise RefusedToStart(
            f"the simulated venue does not survive a restart, and this simulated ledger holds "
            f"positions {list(held)} and live orders {live}. Start a fresh simulated ledger with "
            "`t2sa run --mode simulated --fresh` (the old one is archived, not deleted)"
        )


def archive_ledger(root: Path, mode: RunMode, clock: Clock) -> Path | None:
    """Move a SIMULATED or DRYRUN ledger (and its head anchor) aside under ``var/ledger/archive``.

    Refused for PAPER: the scored log is never moved. Returns the archived path, or ``None`` when
    there was nothing to archive.
    """
    if mode is RunMode.PAPER:
        raise RefusedToStart("the paper ledger is never archived or replaced")
    path = ledger_path(root, mode)
    if not path.exists():
        return None
    stamp = clock.now().astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    archive = path.parent / "archive"
    archive.mkdir(parents=True, exist_ok=True)
    target = archive / f"{mode.value}-{stamp}.jsonl"
    path.replace(target)
    head = path.with_name(path.name + ".head")
    if head.exists():
        head.replace(target.with_name(target.name + ".head"))
    return target


def ruling_context_blob(
    blobs: FileBlobStore,
    *,
    kind: Literal["rule", "protective"],
    book: BookState,
    inputs: KernelInputs,
    breaker: Model,
    context: Model | None,
    proposed: Mapping[str, float] | None,
    guards: Sequence[str],
    llm_outage: bool,
) -> BlobRef:
    """Store what one ruling read, so ``t2sa replay`` can rule again from it."""
    record = {
        "version": 1,
        "kind": kind,
        "book": book,
        "inputs": inputs,
        "breaker": breaker,
        "context": context,
        "proposed": None if proposed is None else dict(proposed),
        "guards": list(guards),
        "llm_outage": llm_outage,
    }
    return blobs.put(canonical_json(record), RULING_CONTEXT_MEDIA_TYPE)


def file_sha256(path: Path) -> str | None:
    try:
        return sha256_hex(path.read_bytes())
    except OSError:
        return None


LOCKFILES: Final[tuple[str, ...]] = ("tools/agent-hub/package-lock.json", "uv.lock")


def lock_hashes(root: Path) -> dict[str, str]:
    """SHA-256 of each dependency lockfile present, keyed by project-relative path."""
    out: dict[str, str] = {}
    for rel in LOCKFILES:
        digest = file_sha256(root.joinpath(*rel.split("/")))
        if digest is not None:
            out[rel] = digest
    return out


def git_commit(root: Path) -> str | None:
    """The commit ``HEAD`` points at, read from ``.git`` without running git. ``None`` when the
    project is not a checkout or the ref cannot be resolved."""
    git = root / ".git"
    if git.is_file():
        text = git.read_text(encoding="utf-8").strip()
        if not text.startswith("gitdir:"):
            return None
        git = (root / text.removeprefix("gitdir:").strip()).resolve()
    head_file = git / "HEAD"
    try:
        head = head_file.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not head.startswith("ref:"):
        return head.lower() if _is_commit(head) else None
    ref = head.removeprefix("ref:").strip()
    common = git
    commondir = git / "commondir"
    if commondir.is_file():
        common = (git / commondir.read_text(encoding="utf-8").strip()).resolve()
    for base in (git, common):
        with contextlib.suppress(OSError):
            value = (base / ref).read_text(encoding="utf-8").strip()
            if _is_commit(value):
                return value.lower()
    with contextlib.suppress(OSError):
        for line in (common / "packed-refs").read_text(encoding="utf-8").splitlines():
            parts = line.strip().split(" ")
            if len(parts) == 2 and parts[1] == ref and _is_commit(parts[0]):
                return parts[0].lower()
    return None


def _is_commit(value: str) -> bool:
    return len(value) in (40, 64) and all(c in "0123456789abcdefABCDEF" for c in value)


def utc_now_text(clock: Clock) -> str:
    return clock.now().astimezone(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "DEFAULT_SIMULATED_EQUITY",
    "LOCKFILES",
    "OI_THRESHOLDS_MEDIA_TYPE",
    "PREGENESIS_LEDGER",
    "RULING_CONTEXT_MEDIA_TYPE",
    "STARTUP_RECONCILE_LOOKBACK",
    "App",
    "InstanceLock",
    "InstanceLocked",
    "LedgerError",
    "LlmChoice",
    "Parts",
    "Paths",
    "Planner",
    "ProjectingLedger",
    "RefusedToStart",
    "archive_ledger",
    "build_app",
    "crypto_symbols",
    "default_data_service",
    "default_toolkit",
    "file_sha256",
    "frozen_oi_thresholds",
    "genesis_of",
    "git_commit",
    "lock_hashes",
    "oi_history",
    "open_chain",
    "open_pregenesis",
    "ruling_context_blob",
    "scrub_note",
    "thresholds_from_genesis",
    "utc_now_text",
]
