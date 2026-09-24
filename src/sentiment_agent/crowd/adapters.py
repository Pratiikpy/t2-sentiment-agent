"""X and Reddit crowd text, through the two command-line clients the agent-reach router uses.

The collectors shell out to ``twitter`` (twitter-cli 0.8.5, ``jackwener/twitter-cli``, Apache-2.0)
and ``rdt`` (rdt-cli 0.4.2, ``jackwener/rdt-cli``, Apache-2.0). Neither is vendored; both are run
as installed programs. Every flag and every output field below was read, not guessed:

* Command syntax from the agent-reach skill reference (``references/social.md``: ``twitter search
  "query" -n 10``, ``rdt search "query" --limit 10``, and "use ``--json`` for structured output"),
  then checked against each tool's own source: ``twitter_cli/cli.py:690-770`` (``search``:
  ``--type`` from ``SEARCH_PRODUCTS = ["Top", "Latest", "Photos", "Videos"]``, ``--lang``,
  ``--since`` as ``YYYY-MM-DD``, ``--exclude retweets``, ``--max/-n``, ``--json``) and
  ``rdt_cli/commands/search.py:74-100`` (``search``: ``--sort`` from ``SEARCH_SORT_OPTIONS``,
  ``--time`` from ``TIME_FILTERS = ["hour", "day", "week", "month", "year", "all"]``,
  ``-n/--limit``, ``--json``).
* The output envelope: both print ``{"ok": true, "schema_version": "1", "data": ...}`` on success
  and ``{"ok": false, "schema_version": "1", "error": {"code", "message"}}`` with exit status 1 on
  failure (``twitter_cli/output.py:94-115``, ``rdt_cli/commands/_common.py:90-112``). Progress and
  pagination hints go to stderr (``rdt_cli/commands/_common.py:23``), so stdout is pure JSON.
* Error codes: ``not_authenticated``, ``rate_limited``, ``not_found``, ``network_error``,
  ``query_id_error``, ``api_error`` (``twitter_cli/exceptions.py``); ``not_authenticated``,
  ``rate_limited``, ``not_found``, ``forbidden``, ``api_error`` (``rdt_cli/exceptions.py:62-75``).
* X rows (``twitter_cli/serialization.py:12-63``): ``id``, ``text``, ``author.screenName``,
  ``createdAt`` (``"Thu Sep 24 11:11:02 +0000 2026"``), ``createdAtISO`` (ISO 8601, or the raw
  string when it could not be parsed, ``timeutil.py:74-82``), ``isRetweet``. Reddit rows are
  Reddit's own listing passed through: ``data.children[].data`` with ``id``, ``title``,
  ``selftext``, ``author``, ``subreddit``, ``created_utc`` (float seconds), ``permalink``,
  ``stickied``. Both shapes were confirmed against one live read of each CLI on 2026-09-24; the
  committed fixtures reproduce the structure with synthetic content.

Two encoding facts from ARGUS ``market/evidence.py:634-657`` (same author, MIT), found by running
these same CLIs on Windows: both write UTF-8 only when told to, so the child runs with
``PYTHONIOENCODING=utf-8`` and ``PYTHONUTF8=1``, and the parent decodes stdout as UTF-8.

Failure is data, never an exception. A missing program is ``DISABLED`` (the source was not called
because it is not installed), a refusal from the service is ``ERROR`` with the service's own code,
a hung program is ``TIMEOUT``, and every symbol always gets a :class:`~types.SourceCall`. After a
failure that will repeat for every symbol (not installed, not logged in, rate limited, timed out),
the remaining symbols are not called and are recorded as ``DISABLED`` with the reason, so an
unattended window does not spend minutes hammering a service that has already said no.
"""

import json
import os
import re
import shutil
import subprocess
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final

from sentiment_agent.crowd.novelty import aliases_for, symbols_mentioned
from sentiment_agent.hashing import content_hash
from sentiment_agent.types import (
    Clock,
    SourceCall,
    SourceHealth,
    TextChannel,
    TextItem,
    ToolkitSurface,
)

CommandRunner = Callable[[Sequence[str], float], tuple[int, str, str]]
"""Runs ``argv`` with a timeout in seconds and returns ``(exit status, stdout, stderr)``.

Contract: raise :class:`FileNotFoundError` when the program is not installed and
:class:`TimeoutError` (or :class:`subprocess.TimeoutExpired`) when it does not finish in time. Any
other exception is reported as an ``ERROR`` source call. Tests pass a fake; production passes
:func:`run_command`."""

CALL_TIMEOUT_S: Final = 60.0
MAX_CONCURRENCY: Final = 8
"""The most searches one collector may keep in flight. Both services are personal logged-in
accounts, so the wiring uses far fewer (``runtime/wiring.py``)."""
"""Per search. Measured on 2026-09-24 with 20-row searches from this machine: X answered in
5.5-7.7 s and Reddit in 13-22 s (rdt-cli refreshes its cookies on each call, and itself retries
HTTP 429 and 5xx with sleeps, ``rdt_cli/transports.py:106-117``). Sixty seconds is about three
times the slowest observed call; a call that exceeds it is hung, and the collection stops calling
that CLI."""

MAX_TEXT_CHARS: Final = 1000
"""Crowd text is clipped to this length at collection, before screening, so what is screened is
exactly what may later be shown. A tweet fits whole; a long Reddit post keeps its title and the
opening of its body."""

CLOCK_SKEW: Final = timedelta(minutes=5)
"""A post stamped later than this past our own fetch time is malformed and dropped: it cannot have
been published after we read it, beyond clock disagreement between us and the platform."""

_ABORTING_CODES: Final = frozenset({"not_authenticated", "rate_limited", "forbidden"})
"""Service error codes that will repeat for every remaining symbol in the same collection.

``forbidden`` is in the set because every search here is site-wide: rdt-cli maps Reddit's HTTP 403
to it (``rdt_cli/transports.py:121-122``), which is what an anonymous session gets for search, so it
means "no access" rather than one private subreddit."""

_X_ID: Final = re.compile(r"^\d{1,25}$")
_X_HANDLE: Final = re.compile(r"^[A-Za-z0-9_]{1,15}$")
_REDDIT_ID: Final = re.compile(r"^[a-z0-9]{1,13}$")
_REDDIT_USER: Final = re.compile(r"^[A-Za-z0-9_-]{1,20}$")
_REDDIT_SUB: Final = re.compile(r"^[A-Za-z0-9_]{1,21}$")
_X_TIME_FORMAT: Final = "%a %b %d %H:%M:%S %z %Y"
_REDDIT_EMPTY_BODIES: Final = frozenset({"", "[removed]", "[deleted]"})


def _child_environment() -> dict[str, str]:
    """The parent's environment without any Bitget credential, and with UTF-8 forced on.

    The crowd CLIs need their own login state (cookies in the user profile, ``TWITTER_*``
    variables) and nothing of ours. Every ``BITGET_*`` variable, which includes the model key
    ``BITGET_QWEN_API_KEY``, is removed so a third-party program never sees one.
    """
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("BITGET_")}
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def run_command(argv: Sequence[str], timeout: float) -> tuple[int, str, str]:
    """The production :data:`CommandRunner`: no shell, no stdin, UTF-8, a hard timeout.

    The program is resolved on ``PATH`` first, so a missing program is a clean
    :class:`FileNotFoundError` on every platform (Windows resolves ``.exe`` shims this way too).
    """
    if not argv:
        raise ValueError("an empty command line")
    program = shutil.which(argv[0])
    if program is None:
        raise FileNotFoundError(f"{argv[0]!r} was not found on PATH")
    try:
        completed = subprocess.run(
            [program, *argv[1:]],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            env=_child_environment(),
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"{argv[0]} did not finish within {timeout:g} s") from exc
    return completed.returncode, completed.stdout or "", completed.stderr or ""


_REPLACEMENT: Final = chr(0xFFFD)


def _clip(text: str) -> str:
    """Strip, replace lone surrogates, and clip to :data:`MAX_TEXT_CHARS`.

    A lone surrogate is what a JSON escape for half of a surrogate pair (U+D83D, say) decodes to
    when the other half is missing. It is broken text, not a tactic, and it cannot be encoded as
    UTF-8, so a record carrying one could not be hashed into the ledger. It becomes U+FFFD, the
    replacement character.
    """
    text = "".join(_REPLACEMENT if 0xD800 <= ord(c) <= 0xDFFF else c for c in text.strip())
    if len(text) <= MAX_TEXT_CHARS:
        return text
    return text[: MAX_TEXT_CHARS - 1] + "\u2026"


def _tail(text: str, limit: int = 300) -> str:
    """The last line or so of a program's stderr, for the record."""
    text = " ".join(text.split())
    return text[-limit:]


@dataclass(frozen=True, slots=True)
class _Outcome:
    """What one search produced, before it becomes a :class:`SourceCall`."""

    health: SourceHealth
    items: tuple[TextItem, ...] = ()
    error: str | None = None
    abort: str | None = None
    """Set when the remaining symbols should not be called; the reason they are recorded with."""


_Searched = tuple[str, dict[str, str], datetime, datetime, "_Outcome"]
"""One search as the concurrent path returns it: symbol, params, started, finished, outcome."""


class _CliCollector(ABC):
    """One search per symbol through one CLI. Subclasses supply the argv and the row parser."""

    surface: ToolkitSurface
    program: str
    channel: TextChannel

    def __init__(
        self,
        *,
        runner: CommandRunner,
        clock: Clock,
        per_symbol_limit: int = 20,
        concurrency: int = 1,
    ) -> None:
        if not 1 <= per_symbol_limit <= 100:
            raise ValueError(f"per_symbol_limit must be between 1 and 100, not {per_symbol_limit}")
        if not 1 <= concurrency <= MAX_CONCURRENCY:
            raise ValueError(
                f"concurrency must be between 1 and {MAX_CONCURRENCY}, not {concurrency}"
            )
        self._runner = runner
        self._clock = clock
        self._limit = per_symbol_limit
        self._concurrency = concurrency
        self._sequence = 0

    @property
    def clock(self) -> Clock:
        return self._clock

    # -- to be supplied by each CLI ------------------------------------------------------------

    @abstractmethod
    def _argv(self, symbol: str, *, since: datetime, now: datetime) -> list[str]:
        """Supplied by each CLI."""

    @abstractmethod
    def _params(self, symbol: str, *, since: datetime, now: datetime) -> dict[str, str]:
        """Supplied by each CLI."""

    @abstractmethod
    def _rows(self, data: Any) -> list[Any] | None:
        """The row list inside ``data``, or None when ``data`` is not the documented shape."""

    @abstractmethod
    def _item(
        self, row: Any, *, symbols: Sequence[str], since: datetime, fetched_at: datetime
    ) -> TextItem | None:
        """One row as a :class:`TextItem`, or None when the row is unusable or out of scope."""

    # -- shared --------------------------------------------------------------------------------

    def collect(
        self, symbols: Sequence[str], *, since: datetime
    ) -> tuple[tuple[TextItem, ...], tuple[SourceCall, ...]]:
        """Search each symbol once. Never raises for anything the CLI or the service does.

        Items are de-duplicated by id across symbols (a post naming two symbols is found twice)
        and returned oldest first. ``since`` must be timezone-aware UTC; a naive value is a
        programming error and raises :class:`ValueError`.
        """
        if since.tzinfo is None or since.utcoffset() != timedelta(0):
            raise ValueError("since must be a timezone-aware UTC datetime")
        requested = tuple(dict.fromkeys(symbols))
        if self._concurrency > 1 and len(requested) > 1:
            return self._collect_concurrently(requested, since=since)
        found: dict[str, TextItem] = {}
        calls: list[SourceCall] = []
        abort: str | None = None
        for symbol in requested:
            now = self._clock.now()
            params = self._params(symbol, since=since, now=now)
            if abort is not None:
                calls.append(
                    self._call(
                        symbol, params, now, now, _Outcome(SourceHealth.DISABLED, error=abort)
                    )
                )
                continue
            argv = self._argv(symbol, since=since, now=now)
            params["argv"] = " ".join(argv)
            outcome = self._search(argv, symbols=requested, since=since, fetched_at=now)
            calls.append(self._call(symbol, params, now, self._clock.now(), outcome))
            for item in outcome.items:
                found.setdefault(item.item_id, item)
            abort = outcome.abort
        ordered = sorted(found.values(), key=lambda i: (i.published_at, i.item_id))
        return tuple(ordered), tuple(calls)

    def _collect_concurrently(
        self, requested: tuple[str, ...], *, since: datetime
    ) -> tuple[tuple[TextItem, ...], tuple[SourceCall, ...]]:
        """:meth:`collect` with up to ``concurrency`` searches in flight at once.

        The first symbol is searched alone, so a failure that would repeat for every symbol (not
        installed, not logged in, rate limited, timed out) still costs one call and marks the rest
        ``DISABLED``, exactly as the serial path does. The rest run on a bounded pool; a search that
        has not started when one of them reports such a failure is skipped with its reason, and a
        search already in flight keeps its real outcome. Source calls are recorded in symbol order,
        so the snapshot reads the same whichever search finished first.
        """
        stop = threading.Event()
        reason: list[str] = []
        guard = threading.Lock()

        def search(symbol: str) -> _Searched:
            now = self._clock.now()
            params = self._params(symbol, since=since, now=now)
            with guard:
                skipped = reason[0] if stop.is_set() else None
            if skipped is not None:
                return symbol, params, now, now, _Outcome(SourceHealth.DISABLED, error=skipped)
            argv = self._argv(symbol, since=since, now=now)
            params["argv"] = " ".join(argv)
            outcome = self._search(argv, symbols=requested, since=since, fetched_at=now)
            if outcome.abort is not None:
                with guard:
                    if not stop.is_set():
                        reason.append(outcome.abort)
                        stop.set()
            return symbol, params, now, self._clock.now(), outcome

        results = [search(requested[0])]
        first_abort = results[0][4].abort
        if first_abort is not None:
            for symbol in requested[1:]:
                now = self._clock.now()
                params = self._params(symbol, since=since, now=now)
                skipped = _Outcome(SourceHealth.DISABLED, error=first_abort)
                results.append((symbol, params, now, now, skipped))
        else:
            workers = min(self._concurrency, len(requested) - 1)
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=self.program) as pool:
                results.extend(pool.map(search, requested[1:]))
        found: dict[str, TextItem] = {}
        calls: list[SourceCall] = []
        for symbol, params, started, finished, outcome in results:
            calls.append(self._call(symbol, params, started, finished, outcome))
            for item in outcome.items:
                found.setdefault(item.item_id, item)
        ordered = sorted(found.values(), key=lambda i: (i.published_at, i.item_id))
        return tuple(ordered), tuple(calls)

    def _search(
        self, argv: Sequence[str], *, symbols: Sequence[str], since: datetime, fetched_at: datetime
    ) -> _Outcome:
        try:
            status, stdout, stderr = self._runner(argv, CALL_TIMEOUT_S)
        except FileNotFoundError:
            reason = f"{self.program} is not installed ({argv[0]!r} not found on PATH)"
            return _Outcome(SourceHealth.DISABLED, error=reason, abort=f"skipped: {reason}")
        except (TimeoutError, subprocess.TimeoutExpired):
            reason = f"{self.program} did not answer within {CALL_TIMEOUT_S:g} s"
            return _Outcome(SourceHealth.TIMEOUT, error=reason, abort=f"skipped: {reason}")
        except Exception as exc:  # the runner itself broke; report it, never raise
            reason = f"{self.program} could not be run: {type(exc).__name__}: {exc}"
            return _Outcome(
                SourceHealth.ERROR, error=reason[:300], abort=f"skipped: {reason}"[:300]
            )

        envelope = _json_object(stdout)
        if envelope is None:
            detail = _tail(stderr) or "no output"
            if status != 0:
                return _Outcome(SourceHealth.ERROR, error=f"exit {status}: {detail}")
            return _Outcome(
                SourceHealth.ERROR, error=f"malformed output (not a JSON object): {detail}"
            )
        if envelope.get("ok") is not True:
            code, message = _error_of(envelope)
            reason = f"{code}: {message}" if message else code
            abort = f"skipped: {reason}" if code in _ABORTING_CODES else None
            return _Outcome(SourceHealth.ERROR, error=reason[:300], abort=abort)
        if status != 0:
            return _Outcome(
                SourceHealth.ERROR, error=f"exit {status} with ok=true: {_tail(stderr)}"
            )

        rows = self._rows(envelope.get("data"))
        if rows is None:
            return _Outcome(
                SourceHealth.ERROR, error="unexpected shape: data is not the documented listing"
            )
        if not rows:
            return _Outcome(SourceHealth.EMPTY)
        items = [
            item
            for row in rows
            if (item := self._item(row, symbols=symbols, since=since, fetched_at=fetched_at))
            is not None
        ]
        if not items:
            return _Outcome(
                SourceHealth.HOLLOW,
                error=f"{len(rows)} row(s) answered, none usable (malformed, too old, "
                "or naming no requested symbol)",
            )
        return _Outcome(SourceHealth.OK, items=tuple(items))

    def _call(
        self,
        symbol: str,
        params: dict[str, str],
        started: datetime,
        finished: datetime,
        outcome: _Outcome,
    ) -> SourceCall:
        self._sequence += 1
        source = f"{self.program}:search:{symbol}"
        call_id = (
            "crowd-"
            + content_hash(
                [self.surface.value, source, params, started.isoformat(), self._sequence]
            )[:24]
        )
        return SourceCall(
            call_id=call_id,
            surface=self.surface,
            source=source,
            params=params,
            health=outcome.health,
            started_at=started,
            latency_ms=max(0, int((finished - started).total_seconds() * 1000)),
            rows=len(outcome.items),
            blob=None,
            error=outcome.error,
        )

    def _admit(self, published_at: datetime, *, since: datetime, fetched_at: datetime) -> bool:
        return since <= published_at <= fetched_at + CLOCK_SKEW


def _json_object(stdout: str) -> dict[str, Any] | None:
    """The JSON object on stdout, or None. A nesting bomb (``RecursionError``) is malformed too."""
    try:
        payload = json.loads(stdout)
    except (ValueError, TypeError, RecursionError):
        return None
    return payload if isinstance(payload, dict) else None


def _error_of(envelope: Mapping[str, Any]) -> tuple[str, str]:
    error = envelope.get("error")
    if not isinstance(error, dict):
        return "unknown_error", "ok is not true and no error object was given"
    code = error.get("code")
    message = error.get("message")
    return (
        code if isinstance(code, str) and code else "unknown_error",
        " ".join(message.split())[:250] if isinstance(message, str) else "",
    )


def _string(row: Mapping[str, Any], key: str) -> str | None:
    value = row.get(key)
    return value if isinstance(value, str) else None


class XCollector(_CliCollector):
    """Cashtag search on X through twitter-cli: latest English posts, retweets excluded.

    Retweets are excluded at the query (``--exclude retweets``) and again per row
    (``isRetweet``): a retweet copies somebody else's words verbatim, and three accounts retweeting
    one post would otherwise read as three sources carrying one story, which is virality, not
    coordination. Quoted posts keep only the quoting author's own text.
    """

    surface = ToolkitSurface.CROWD_X
    program = "twitter-cli"
    channel: TextChannel = "x"

    def _query(self, symbol: str) -> str:
        query = aliases_for(symbol).x_query
        return f"({query})" if " OR " in query else query

    def _argv(self, symbol: str, *, since: datetime, now: datetime) -> list[str]:
        return [
            "twitter",
            "search",
            self._query(symbol),
            "--type",
            "Latest",
            "--lang",
            "en",
            "--exclude",
            "retweets",
            "--since",
            since.date().isoformat(),
            "-n",
            str(self._limit),
            "--json",
        ]

    def _params(self, symbol: str, *, since: datetime, now: datetime) -> dict[str, str]:
        return {
            "symbol": symbol,
            "query": self._query(symbol),
            "since": since.isoformat(),
            "limit": str(self._limit),
        }

    def _rows(self, data: Any) -> list[Any] | None:
        return data if isinstance(data, list) else None

    def _item(
        self, row: Any, *, symbols: Sequence[str], since: datetime, fetched_at: datetime
    ) -> TextItem | None:
        if not isinstance(row, dict) or row.get("isRetweet") is True:
            return None
        tweet_id = _string(row, "id")
        text = _string(row, "text")
        author = row.get("author")
        handle = _string(author, "screenName") if isinstance(author, dict) else None
        if not tweet_id or not _X_ID.fullmatch(tweet_id) or not handle:
            return None
        if not _X_HANDLE.fullmatch(handle) or not text or not text.strip():
            return None
        published = _x_time(row)
        if published is None or not self._admit(published, since=since, fetched_at=fetched_at):
            return None
        clipped = _clip(text)
        named = symbols_mentioned(clipped, symbols)
        if not named:
            return None
        return TextItem(
            item_id=f"x:{tweet_id}",
            channel=self.channel,
            source=f"@{handle}",
            url=f"https://x.com/{handle}/status/{tweet_id}",
            published_at=published,
            fetched_at=fetched_at,
            text=clipped,
            symbols=named,
        )


def _x_time(row: Mapping[str, Any]) -> datetime | None:
    """``createdAtISO`` when it parses, else the raw ``createdAt`` in X's own format."""
    iso = _string(row, "createdAtISO")
    if iso:
        try:
            stamp = datetime.fromisoformat(iso)
        except ValueError:
            stamp = None
        if stamp is not None and stamp.tzinfo is not None:
            return stamp.astimezone(UTC)
    raw = _string(row, "createdAt")
    if raw:
        try:
            return datetime.strptime(raw, _X_TIME_FORMAT).astimezone(UTC)
        except ValueError:
            return None
    return None


class RedditCollector(_CliCollector):
    """Site-wide Reddit search through rdt-cli, newest first, bounded by a time filter.

    The source of a post is its author (``u/name``), not its subreddit: coordination is several
    independent people posting one story, so five accounts in r/wallstreetbets are five sources and
    one account cross-posting to five subreddits is one. Posts by deleted accounts are skipped
    (their source cannot be told apart), as are stickied moderator posts.
    """

    surface = ToolkitSurface.CROWD_REDDIT
    program = "rdt-cli"
    channel: TextChannel = "reddit"

    def _argv(self, symbol: str, *, since: datetime, now: datetime) -> list[str]:
        return [
            "rdt",
            "search",
            aliases_for(symbol).reddit_query,
            "--sort",
            "new",
            "--time",
            reddit_time_filter(now - since),
            "-n",
            str(self._limit),
            "--json",
        ]

    def _params(self, symbol: str, *, since: datetime, now: datetime) -> dict[str, str]:
        return {
            "symbol": symbol,
            "query": aliases_for(symbol).reddit_query,
            "since": since.isoformat(),
            "limit": str(self._limit),
            "time": reddit_time_filter(now - since),
        }

    def _rows(self, data: Any) -> list[Any] | None:
        if not isinstance(data, dict):
            return None
        listing = data.get("data")
        if not isinstance(listing, dict):
            return None
        children = listing.get("children")
        return children if isinstance(children, list) else None

    def _item(
        self, row: Any, *, symbols: Sequence[str], since: datetime, fetched_at: datetime
    ) -> TextItem | None:
        if not isinstance(row, dict) or row.get("kind") != "t3":
            return None
        post = row.get("data")
        if not isinstance(post, dict) or post.get("stickied") is True:
            return None
        post_id = _string(post, "id")
        title = _string(post, "title")
        author = _string(post, "author")
        subreddit = _string(post, "subreddit")
        created = post.get("created_utc")
        if not post_id or not _REDDIT_ID.fullmatch(post_id) or not title or not title.strip():
            return None
        if not author or not _REDDIT_USER.fullmatch(author):
            return None
        if not subreddit or not _REDDIT_SUB.fullmatch(subreddit):
            return None
        if isinstance(created, bool) or not isinstance(created, (int, float)):
            return None
        try:
            published = datetime.fromtimestamp(created, tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None
        if not self._admit(published, since=since, fetched_at=fetched_at):
            return None
        body = (_string(post, "selftext") or "").strip()
        text = title.strip() if body in _REDDIT_EMPTY_BODIES else f"{title.strip()} \u2014 {body}"
        clipped = _clip(text)
        named = symbols_mentioned(clipped, symbols)
        if not named:
            return None
        permalink = _string(post, "permalink")
        url = (
            f"https://www.reddit.com{permalink}"
            if permalink and permalink.startswith(f"/r/{subreddit}/")
            else f"https://www.reddit.com/r/{subreddit}/comments/{post_id}/"
        )
        return TextItem(
            item_id=f"reddit:{post_id}",
            channel=self.channel,
            source=f"u/{author}",
            url=url,
            published_at=published,
            fetched_at=fetched_at,
            text=clipped,
            symbols=named,
        )


def reddit_time_filter(lookback: timedelta) -> str:
    """The narrowest rdt-cli ``--time`` value that still covers ``lookback``."""
    for limit, name in (
        (timedelta(hours=1), "hour"),
        (timedelta(days=1), "day"),
        (timedelta(days=7), "week"),
        (timedelta(days=31), "month"),
        (timedelta(days=366), "year"),
    ):
        if lookback <= limit:
            return name
    return "all"


class CompositeCrowd:
    """Every configured crowd collector, merged. Satisfies :class:`types.CrowdCollector`.

    Never raises: a collector that raises anyway (a programming error, a broken clock) becomes one
    ``ERROR`` source call naming the exception, and the other collectors still run. Items are
    de-duplicated by id and returned oldest first; source calls keep collector order.
    """

    def __init__(
        self, collectors: Sequence[XCollector | RedditCollector], *, parallel: bool = False
    ) -> None:
        self._collectors = tuple(collectors)
        self._parallel = parallel

    def collect(
        self, symbols: Sequence[str], *, since: datetime
    ) -> tuple[tuple[TextItem, ...], tuple[SourceCall, ...]]:
        """Every collector's items and calls. With ``parallel`` the collectors (independent
        programs reaching independent services) run at the same time; the results are merged in
        collector order either way, so the answer does not depend on which finished first."""

        def one(
            collector: XCollector | RedditCollector,
        ) -> tuple[tuple[TextItem, ...], tuple[SourceCall, ...]]:
            try:
                return collector.collect(symbols, since=since)
            except Exception as exc:  # the contract is "never raises"; the failure is recorded
                return (), (_crashed(collector, symbols, since, exc),)

        if self._parallel and len(self._collectors) > 1:
            with ThreadPoolExecutor(
                max_workers=len(self._collectors), thread_name_prefix="crowd"
            ) as pool:
                answers = list(pool.map(one, self._collectors))
        else:
            answers = [one(c) for c in self._collectors]
        found: dict[str, TextItem] = {}
        calls: list[SourceCall] = []
        for items, made in answers:
            calls.extend(made)
            for item in items:
                found.setdefault(item.item_id, item)
        ordered = sorted(found.values(), key=lambda i: (i.published_at, i.item_id))
        return tuple(ordered), tuple(calls)


def _crashed(
    collector: XCollector | RedditCollector,
    symbols: Sequence[str],
    since: datetime,
    exc: Exception,
) -> SourceCall:
    epoch = datetime.fromtimestamp(0, tz=UTC)
    try:
        now = collector.clock.now()
    except Exception:  # a clock that raises still must not take the snapshot down
        now = epoch
    usable = isinstance(now, datetime) and now.tzinfo is not None and not now.utcoffset()
    started = now.astimezone(UTC) if usable else epoch
    params = {"symbols": ",".join(str(s) for s in symbols), "since": str(since)}
    source = f"{collector.program}:collect"
    return SourceCall(
        call_id="crowd-" + content_hash([collector.surface.value, source, params, repr(exc)])[:24],
        surface=collector.surface,
        source=source,
        params=params,
        health=SourceHealth.ERROR,
        started_at=started,
        latency_ms=0,
        rows=0,
        blob=None,
        error=f"collector raised {type(exc).__name__}: {exc}"[:300],
    )


__all__ = [
    "CALL_TIMEOUT_S",
    "CLOCK_SKEW",
    "MAX_TEXT_CHARS",
    "CommandRunner",
    "CompositeCrowd",
    "RedditCollector",
    "XCollector",
    "reddit_time_filter",
    "run_command",
]
