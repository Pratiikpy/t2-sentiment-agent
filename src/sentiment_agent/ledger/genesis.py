"""The pre-registration: what the paper run commits to before its first order, and how it changes.

Seq 0 of the scored PAPER ledger is a :class:`~sentiment_agent.types.Genesis` (DESIGN.md §12, §16):
the policy with its hash, the prompt hashes, the universe, the metric definitions, the expected
no-edge envelope, the code commit, the dependency lockfile hashes, the Agent Hub CLI package and the
Qwen model. Its event hash is stamped with OpenTimestamps (:mod:`sentiment_agent.ledger.anchor`) and
posted on X by the owner before the first order (:func:`x_post_text`).

After genesis the policy is frozen. :func:`require_genesis` is the guard the runtime calls before it
sends anything: no genesis at seq 0, or a loaded policy that is not the *active* one, and no order
goes out. The active policy is the genesis policy until an
:class:`~sentiment_agent.types.Amendment` replaces it; each amendment is a logged, owner-confirmed
event that names the policy it replaces and the one it installs (:func:`amend`). DESIGN.md §12
words the guard as "matches neither the genesis nor the latest Amendment"; it is implemented as
"matches the latest", because once an amendment exists the genesis policy is superseded, and
running it would be running a policy the log says was replaced.

The rules themselves (genesis only at seq 0, amendments confirmed and linked to the active policy)
are enforced by the chain on every append and every read (:mod:`sentiment_agent.ledger.chain`), so
no code path can write a history this module would reject.
"""

import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Final

from pydantic import ValidationError

from sentiment_agent.hashing import content_hash
from sentiment_agent.ledger.chain import (
    GenesisError,
    HashChainLedger,
    event_hash,
    policy_hash_after,
)
from sentiment_agent.types import (
    CONTRACT_VERSION,
    PROJECT_SLUG,
    Amendment,
    Clock,
    DeclaredChange,
    EventKind,
    Genesis,
    LedgerEvent,
    LedgerReader,
    Policy,
    PredecessorRun,
    RunMode,
)

X_QUOTED_POST: Final = "https://x.com/Bitget_AI/status/2100519318824055159"
"""The Bitget post every compliant submission post must quote (handbook:123, :172)."""
X_HASHTAG: Final = "#BitgetHackathon"
X_MENTION: Final = "@Bitget_AI"
X_MAX_WEIGHTED_LENGTH: Final = 280
X_URL_WEIGHT: Final = 23
_X_LIGHT_RANGES: Final = ((0, 4351), (8192, 8205), (8208, 8223), (8242, 8247))
"""Code points that weigh 1; every other weighs 2. From twitter-text ``config/v3.json``
(``maxWeightedTweetLength`` 280, ``scale`` 100, ``defaultWeight`` 200, ``transformedURLLength`` 23),
read on 2026-09-24."""
_URL: Final = re.compile(r"https?://\S+")

_COMMIT: Final = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_DIGEST: Final = re.compile(r"^[0-9a-f]{64}$")
_RELATIVE_NAME: Final = re.compile(r"^[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*$")
_NPM_PACKAGE: Final = re.compile(
    r"^(?:@[a-z0-9][a-z0-9._-]*/)?[a-z0-9][a-z0-9._-]*@\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$"
)

_PRE_REGISTRATION: Final = frozenset({EventKind.GENESIS, EventKind.AMENDMENT})

_FROZEN: Final = (
    "It fixes the policy (every guard, limit and threshold, each with its measured basis), the "
    "prompt files, the universe, the metric definitions, the expected no-edge envelope, the code "
    "commit, the dependency locks, the Agent Hub CLI package and the Qwen model. Any change after "
    "this event is a logged, owner-confirmed amendment in this ledger, and thresholds are not "
    "tuned inside the window. Results are reported against the envelope with n and a "
    "block-bootstrap interval, labelled descriptive, not inferential."
)


def _statement(mode: RunMode) -> str:
    if mode is RunMode.PAPER:
        return (
            f"Pre-registration of the {PROJECT_SLUG} paper run on Bitget UTA Demo, written before "
            f"its first paper order. {_FROZEN}"
        )
    return (
        f"Rehearsal genesis of a {mode.value} run of {PROJECT_SLUG}. No order in this ledger "
        f"reaches Bitget and it is not the scored log. {_FROZEN}"
    )


def _require_relative_names(field: str, names: Iterable[str]) -> None:
    for name in names:
        if not _RELATIVE_NAME.fullmatch(name) or ".." in name.split("/"):
            raise GenesisError(
                f"{field} key {name!r} must be a project-relative path with forward slashes; no "
                "local path is ever published"
            )


def build_genesis(
    *,
    policy: Policy,
    prompt_hashes: Mapping[str, str],
    mode: RunMode,
    code_commit: str,
    lock_hashes: Mapping[str, str],
    bgc_package: str,
    clock: Clock,
    predecessor: PredecessorRun | None = None,
    declared_changes: Sequence[DeclaredChange] = (),
) -> Genesis:
    """The pre-registration for a run of ``policy`` in ``mode``, stamped with the clock's time.

    ``prompt_hashes`` maps each prompt file (project-relative) to its SHA-256; ``lock_hashes`` maps
    each dependency lockfile to its SHA-256; ``code_commit`` is the full git commit the agent runs
    from; ``bgc_package`` is the pinned Agent Hub CLI, e.g. ``@bitget-ai/bitget-agent-cli@3.0.0``.
    A PAPER genesis must name at least one prompt and one lockfile: a pre-registration that leaves
    out what the model is told, or what code runs, pre-registers nothing.

    ``predecessor`` and ``declared_changes`` (contract 1.1.0) name the run this one follows and
    every change against it (``sentiment_agent.run2``); the genesis refuses a policy that differs
    from the predecessor's without a declared amendment that installs it.
    """
    mode = RunMode(mode)
    if not _COMMIT.fullmatch(code_commit) or set(code_commit) == {"0"}:
        raise GenesisError(
            "code_commit must be the full git commit the agent runs from (40 or 64 lowercase "
            f"hex), not {code_commit!r}"
        )
    if not _NPM_PACKAGE.fullmatch(bgc_package):
        raise GenesisError(
            f"bgc_package must be a pinned npm package, e.g. @bitget-ai/bitget-agent-cli@3.0.0, "
            f"not {bgc_package!r}"
        )
    _require_relative_names("prompt_hashes", prompt_hashes)
    _require_relative_names("lock_hashes", lock_hashes)
    for name, digest in lock_hashes.items():
        if not _DIGEST.fullmatch(digest):
            raise GenesisError(f"lock_hashes[{name!r}] must be the file's SHA-256 (64 hex)")
    if mode is RunMode.PAPER and not prompt_hashes:
        raise GenesisError("a paper genesis must pre-register the prompt files it runs")
    if mode is RunMode.PAPER and not lock_hashes:
        raise GenesisError("a paper genesis must pre-register the dependency lockfiles")
    try:
        return Genesis(
            project=PROJECT_SLUG,
            contract_version=CONTRACT_VERSION,
            created_at=clock.now(),
            mode=mode,
            policy=policy,
            policy_hash=policy.content_hash(),
            prompt_hashes=dict(prompt_hashes),
            universe=policy.symbols,
            metric_definitions=policy.metrics,
            code_commit=code_commit,
            dependency_lock_hashes=dict(lock_hashes),
            qwen_model=policy.decision.model,
            bgc_package=bgc_package,
            expected_envelope=dict(policy.expected_envelope),
            statement=_statement(mode),
            predecessor=predecessor,
            declared_changes=tuple(declared_changes),
        )
    except ValidationError as exc:
        raise GenesisError(f"the genesis does not validate: {exc.errors()[0]['msg']}") from None


def write_genesis(ledger: HashChainLedger, genesis: Genesis) -> LedgerEvent:
    """Write ``genesis`` as seq 0 of ``ledger``. Refused anywhere else, and so only once."""
    if genesis.mode is not ledger.mode:
        raise GenesisError(
            f"a {genesis.mode.value} genesis cannot open the {ledger.mode.value} ledger"
        )
    head = ledger.head()
    if head is not None:
        raise GenesisError(
            f"a genesis may only be seq 0; {ledger.path.name} already holds {head.seq + 1} "
            "event(s). The pre-registration comes first and exactly once"
        )
    return ledger.append(EventKind.GENESIS, genesis)


def _pre_registration(ledger: LedgerReader) -> list[LedgerEvent]:
    return list(ledger.events(_PRE_REGISTRATION))


def _require_intact(event: LedgerEvent) -> None:
    recomputed = event_hash(
        seq=event.seq,
        ts=event.ts,
        kind=event.kind,
        mode=event.mode,
        payload=event.payload,
        blobs=event.blobs,
        prev_hash=event.prev_hash,
    )
    if recomputed != event.hash:
        raise GenesisError(f"seq {event.seq} does not hash to its hash field; it was altered")


def _genesis_of(events: list[LedgerEvent]) -> tuple[LedgerEvent, Genesis]:
    if not events or events[0].kind is not EventKind.GENESIS:
        raise GenesisError(
            "the ledger has no genesis: nothing is pre-registered, so no order may be sent"
        )
    event = events[0]
    if event.seq != 0:
        raise GenesisError(f"the genesis is at seq {event.seq}; it must be seq 0")
    _require_intact(event)
    try:
        genesis = Genesis.model_validate(event.payload)
    except ValidationError as exc:
        raise GenesisError(
            f"the genesis payload does not validate: {exc.errors()[0]['msg']}"
        ) from None
    if genesis.mode is not event.mode:
        raise GenesisError(
            f"the genesis pre-registers a {genesis.mode.value} run but sits in a "
            f"{event.mode.value} ledger"
        )
    return event, genesis


def active_policy_hash(ledger: LedgerReader) -> str:
    """The hash of the policy in force: the genesis's, or the latest amendment's."""
    active = policy_hash_after(_pre_registration(ledger))
    if active is None:
        raise GenesisError("the ledger has no genesis, so no policy is in force")
    return active


def require_genesis(ledger: LedgerReader, policy: Policy) -> Genesis:
    """The genesis, provided ``policy`` is the one in force. The guard before every order.

    Raises :class:`GenesisError` when seq 0 is not an intact genesis, when the amendment history is
    broken, or when ``policy`` does not hash to the active policy (the genesis's until an amendment,
    then the latest amendment's).
    """
    events = _pre_registration(ledger)
    _, genesis = _genesis_of(events)
    for event in events[1:]:
        _require_intact(event)
    active = policy_hash_after(events)
    if active is None:  # pragma: no cover - _genesis_of guarantees a genesis
        raise GenesisError("the ledger has no genesis")
    loaded = policy.content_hash()
    if loaded != active:
        amendments = [e for e in events if e.kind is EventKind.AMENDMENT]
        source = f"the amendment at seq {amendments[-1].seq}" if amendments else "the genesis"
        raise GenesisError(
            f"the loaded policy {policy.version!r} hashes to {loaded}, but the policy in force is "
            f"{active}, set by {source}. Load the pre-registered policy, or log an owner-confirmed "
            "amendment first; no order is sent under an unregistered policy"
        )
    return genesis


def amend(
    ledger: HashChainLedger,
    *,
    new_policy: Policy,
    reason: str,
    owner_confirmed: bool,
    clock: Clock,
) -> LedgerEvent:
    """Log a change of policy after genesis. Needs the owner's confirmation and a written reason.

    The amendment names the policy it replaces (the one in force now) and the one it installs. The
    chain re-checks that link inside its writer lock, so two amendments written at once cannot both
    claim to replace the same policy.
    """
    if not owner_confirmed:
        raise GenesisError(
            "a policy change after genesis needs the owner's confirmation; nothing was logged"
        )
    if not reason.strip():
        raise GenesisError("an amendment must say why the policy changes")
    previous = active_policy_hash(ledger)
    new_hash = new_policy.content_hash()
    if new_hash == previous:
        raise GenesisError("the policy is unchanged; there is nothing to amend")
    at = clock.now()
    amendment = Amendment(
        amendment_id="amendment-"
        + content_hash({"previous": previous, "new": new_hash, "at": at, "reason": reason})[:16],
        at=at,
        reason=reason,
        previous_policy_hash=previous,
        new_policy_hash=new_hash,
        new_policy=new_policy,
        owner_confirmed=owner_confirmed,
    )
    return ledger.append(EventKind.AMENDMENT, amendment)


def x_weighted_length(text: str) -> int:
    """The length X counts: each URL as 23, code points in the light ranges as 1, others as 2.

    Emoji sequences are counted per code point here, which over-counts them; that errs on the safe
    side of the 280 limit.
    """
    total = 0
    position = 0
    for match in _URL.finditer(text):
        total += _plain_weight(text[position : match.start()]) + X_URL_WEIGHT
        position = match.end()
    return total + _plain_weight(text[position:])


def _plain_weight(text: str) -> int:
    return sum(
        1 if any(low <= ord(ch) <= high for low, high in _X_LIGHT_RANGES) else 2 for ch in text
    )


def x_post_text(genesis_event: LedgerEvent) -> str:
    """The owner's X post announcing the genesis hash, quoting Bitget's hackathon post.

    Carries the full genesis event hash, ``#BitgetHackathon``, ``@Bitget_AI`` and the quoted post's
    link (a link to a post is how X quotes it), and fits X's 280 weighted characters. Only an intact
    PAPER genesis is announced: a rehearsal genesis is not the pre-registration of the scored run.
    The handbook also asks for a retweet of the quoted post (handbook:123, :285); that is the
    owner's action on X, not text.
    """
    if genesis_event.kind is not EventKind.GENESIS or genesis_event.seq != 0:
        raise GenesisError("the X post announces the genesis event, seq 0 of the paper ledger")
    _require_intact(genesis_event)
    try:
        genesis = Genesis.model_validate(genesis_event.payload)
    except ValidationError as exc:
        raise GenesisError(
            f"the genesis payload does not validate: {exc.errors()[0]['msg']}"
        ) from None
    if genesis.mode is not RunMode.PAPER or genesis_event.mode is not RunMode.PAPER:
        raise GenesisError(
            f"this is a {genesis.mode.value} genesis; only the paper run's genesis is its public "
            "pre-registration"
        )
    project = genesis.project
    descriptions = (
        f"{project}: a Market Sentiment Agent paper-trading on Bitget Demo. Qwen decides, a "
        "reduce-only risk kernel gates, Agent Hub executes.",
        f"{project}: a Market Sentiment Agent paper-trading on Bitget Demo. Qwen decides.",
        f"{project}: a Market Sentiment Agent on Bitget Demo.",
    )
    for description in descriptions:
        text = (
            f"{description}\n\nGenesis hash:\n{genesis_event.hash}\n\n"
            f"{X_HASHTAG} {X_MENTION}\n{X_QUOTED_POST}"
        )
        if x_weighted_length(text) <= X_MAX_WEIGHTED_LENGTH:
            return text
    raise GenesisError(
        f"the project name {project!r} leaves no room for the post inside "
        f"{X_MAX_WEIGHTED_LENGTH} characters"
    )


X_POST_NOTE_PREFIX: Final = "the pre-registration was posted on X: "
"""How ``t2sa x-posted`` records the owner's post: an owner note, this prefix, the post's URL. The
ledger, not an environment variable, holds it, so it is timestamped, hash-chained and published
with everything else, and recording it never needs the running agent restarted."""

_X_STATUS: Final = re.compile(
    r"https://(?:www\.|mobile\.)?(?:x|twitter)\.com/([A-Za-z0-9_]{1,15})/status/([0-9]{1,20})"
)
_X_SNOWFLAKE_EPOCH_MS: Final = 1_288_834_974_657
"""X status ids are snowflakes: the bits above the lowest 22 count milliseconds since this epoch
(2010-11-04 01:42:54.657 UTC), so a post's id says when it was made without asking X. Checked on
:data:`X_QUOTED_POST`, whose id decodes to 2026-09-17 09:36:46 UTC."""


def x_post_url(text: str) -> str:
    """The canonical URL of a post on X (``https://x.com/<handle>/status/<id>``, with any query,
    fragment or trailing slash dropped), or :class:`GenesisError` when ``text`` is not one."""
    bare = text.strip().split("?", 1)[0].split("#", 1)[0].rstrip("/")
    match = _X_STATUS.fullmatch(bare)
    if match is None:
        raise GenesisError(
            f"not the URL of a post on X: {text.strip()[:120]!r}; expected "
            "https://x.com/<handle>/status/<id>"
        )
    return f"https://x.com/{match.group(1)}/status/{match.group(2)}"


def x_posted_at(url: str) -> datetime:
    """When the post at ``url`` was made, read from its id (:data:`_X_SNOWFLAKE_EPOCH_MS`)."""
    status = int(x_post_url(url).rsplit("/", 1)[1])
    return datetime.fromtimestamp(((status >> 22) + _X_SNOWFLAKE_EPOCH_MS) / 1000, tz=UTC)


def recorded_x_post(events: Iterable[LedgerEvent]) -> tuple[str, LedgerEvent] | None:
    """The X post the owner last recorded (``t2sa x-posted``), with the note that records it."""
    found: tuple[str, LedgerEvent] | None = None
    for event in events:
        if event.kind is not EventKind.NOTE or event.payload.get("author") != "owner":
            continue
        text = str(event.payload.get("text", ""))
        if not text.startswith(X_POST_NOTE_PREFIX):
            continue
        try:
            found = (x_post_url(text[len(X_POST_NOTE_PREFIX) :]), event)
        except GenesisError:
            continue
    return found


__all__ = [
    "X_HASHTAG",
    "X_MAX_WEIGHTED_LENGTH",
    "X_MENTION",
    "X_POST_NOTE_PREFIX",
    "X_QUOTED_POST",
    "GenesisError",
    "active_policy_hash",
    "amend",
    "build_genesis",
    "recorded_x_post",
    "require_genesis",
    "write_genesis",
    "x_post_text",
    "x_post_url",
    "x_posted_at",
    "x_weighted_length",
]
