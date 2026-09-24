"""Demo credentials, the child-process environment, and the proof that the key is a Demo key.

Every paper order goes through Bitget's own CLI (``bgc``, :mod:`sentiment_agent.execution.bgc`).
This module owns the three things that decide *where* those orders land:

1. **The credentials.** Read from exactly one file, ``<project root>/.secrets/demo.env``, which
   must resolve inside the project root and must declare ``BITGET_KEY_ENVIRONMENT=demo``. The
   values live only inside :class:`DemoCredentials`, whose ``repr`` is redacted, and reach ``bgc``
   only through a child environment built from scratch: inherited ``BITGET_*`` variables are never
   copied, so a live key exported in the parent shell cannot reach the CLI.
2. **The vendor contract.** Before a credential is handed to the installed CLI, the installed files
   are read and checked: the pinned versions, the SDK rule that sends ``paptrading: 1`` on every
   private call when ``--paper-trading`` is set (``agent-sdk/src/client/rest-client.ts:274-280``),
   the CLI mapping ``--paper-trading`` to that setting (``agent-cli/src/index.ts:214-221``), the
   ``--paper-trading``/``--read-only`` exclusion (``agent-sdk/src/config.ts:207-212``) and the
   dry-run early return (``agent-sdk/src/tools/safety.ts:86-88``). An upstream change to any of them
   fails the proof instead of silently routing an order live.
3. **The environment proof** (DESIGN.md §11.2), written to the ledger before the first order.

How Bitget reports errors through ``bgc`` decides how the proof reads them, so it is stated here.
Bitget answers a refused private call with HTTP 400 and a JSON body (measured 2026-09-24: an
unsigned ``GET /api/v3/account/assets`` returns HTTP 400 ``{"code":"40006","msg":"Invalid
ACCESS_KEY"}``, with or without ``paptrading: 1``). The SDK turns any non-2xx answer into a
``BitgetApiError`` whose ``code`` is the *HTTP status* and whose message is
``"HTTP 400 from Bitget: <msg>"`` (``rest-client.ts:323-331``); Bitget's own code survives only
when the HTTP status is 200 (``rest-client.ts:334-353``), and the ``account_overview`` composite
keeps only the message string of a failed section (``tools/composites/account-overview.ts:92-115``).
So error 40099 is recognised by its code where one survives and by its message
(``"exchange environment is incorrect"``, observed by ARGUS ``execution/bitget_client.py:251-263``)
everywhere else.

Two departures from the build brief, both deliberate:

* **The live-negative probe reads ``raw --operationId getAccountAssets`` rather than
  ``account_overview``.** It is the first account read ``account_overview`` fans out to, but as a
  single operation its failure reaches stderr with the SDK's error type, code and category intact,
  which is what separates "the live venue refused this key" from "the network failed". The
  composite would report both as a message string with exit code 0.
* **On the live-negative probe, 40099 counts as the live venue refusing the key**, alongside an
  authentication-class refusal. 40099 means the key belongs to the other environment; on a
  ``--paper-trading`` call that is a live key and is refused (:class:`EnvironmentRefused`), and on
  the one non-paper call it is exactly the rejection the probe looks for. ARGUS records the
  symmetry ("a live key sent with ``paptrading: 1``, or a demo key sent without it",
  ``execution/bitget_client.py:251-253``; only the first half was measured there, so the second is
  NOT VERIFIED until the first proof runs). A live key cannot produce 40099 on the live route, and
  the demo-positive probe independently refuses a live key, so this cannot let a live key pass.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, NoReturn

from sentiment_agent.hashing import canonical_json, sha256_hex
from sentiment_agent.types import (
    DEMO_CREDENTIALS_FILE,
    AccountSnapshot,
    BlobRef,
    BlobStore,
    Category,
    Clock,
    EnvironmentProof,
    RunMode,
)

if TYPE_CHECKING:
    from sentiment_agent.execution.bgc import BgcResult, BgcRunner

# --- where things are -----------------------------------------------------------------------

DEMO_ENV_FILE: Final = DEMO_CREDENTIALS_FILE
"""``.secrets/demo.env``: the only place a Bitget credential is ever read from, relative to the
project root, and the only path a passing proof may name (``types.DEMO_CREDENTIALS_FILE``)."""

KEY_ENVIRONMENT_VAR: Final = "BITGET_KEY_ENVIRONMENT"
DEMO_DECLARATION: Final = "demo"
CREDENTIAL_VARS: Final[tuple[str, str, str]] = (
    "BITGET_API_KEY",
    "BITGET_SECRET_KEY",
    "BITGET_PASSPHRASE",
)
CHILD_ENV_PASSTHROUGH: Final[tuple[str, ...]] = ("PATH", "SYSTEMROOT")
"""The only parent variables a ``bgc`` child sees. Node on Windows cannot open a socket without
``SYSTEMROOT`` (measured 2026-09-24: ``listen UNKNOWN`` with PATH alone); nothing else is needed."""

MAX_ENV_FILE_BYTES: Final = 64 * 1024

AGENT_HUB_DIR: Final = "tools/agent-hub"
CLI_PACKAGE: Final = "@bitget-ai/bitget-agent-cli"
SDK_PACKAGE: Final = "@bitget-ai/bitget-agent-sdk"
PINNED_VERSION: Final = "3.0.0"
CLI_ENTRY: Final = "node_modules/@bitget-ai/bitget-agent-cli/lib/index.js"
"""The CLI's entry file, relative to :data:`AGENT_HUB_DIR`."""

# --- the flags that decide the environment ----------------------------------------------------

PAPER_FLAG: Final = "--paper-trading"
READ_ONLY_FLAG: Final = "--read-only"
DRY_RUN_FLAG: Final = "--dry-run"
CATEGORY: Final = Category.USDT_FUTURES.value

DEMO_OVERVIEW_ARGS: Final[tuple[str, ...]] = (
    "account_overview",
    "--category",
    CATEGORY,
    "--view",
    "full",
    PAPER_FLAG,
)
"""Demo-positive: the whole Demo account in one read (assets, settings, funding, positions)."""

DEMO_ASSETS_ARGS: Final[tuple[str, ...]] = (
    "raw",
    "--operationId",
    "getAccountAssets",
    PAPER_FLAG,
)
"""Demo-positive, code-bearing: the same assets read as one operation, so a refusal keeps its
error type and code."""

LIVE_NEGATIVE_ARGS: Final[tuple[str, ...]] = (
    "raw",
    "--operationId",
    "getAccountAssets",
    READ_ONLY_FLAG,
)
"""The only argv in this project without ``--paper-trading``. ``--read-only`` makes ``bgc`` refuse
any write before the network (``safety.ts:90-95``), and ``getAccountAssets`` is a GET."""

HOLD_MODES: Final = frozenset({"one_way_mode", "hedge_mode"})

REQUIRED_SECTIONS: Final = frozenset({"assets", "settings", "positions"})
"""The ``account_overview`` sections the demo-positive probe needs: equity, hold mode and the
futures book. Its ``fundingAssets`` section (the separate funding account) plays no part in
trading USDT futures and may not exist on Demo, so a failure there is recorded, not disqualifying;
a 40099 anywhere in the answer still refuses."""

EXIT_ENVIRONMENT_REFUSED: Final = 3
"""The runtime's exit code after :class:`EnvironmentRefused`."""

PROBE_TIMEOUT_S: Final = 90.0

ENVIRONMENT_MISMATCH_CODE: Final = "40099"
_ENVIRONMENT_MISMATCH_TEXT = re.compile(r"environment\s+is\s+incorrect", re.IGNORECASE)
_KEY_REFUSAL_TEXT = re.compile(
    r"access[_ ]?key|api ?key|passphrase|signature|sign error|does not exist|not exist"
    r"|permission|whitelist|timestamp",
    re.IGNORECASE,
)
_KEY_REFUSAL_HTTP: Final = frozenset({"400", "401", "403"})
_ENV_KEY = re.compile(r"^[A-Z_][A-Z0-9_]*$")


class EnvironmentRefused(RuntimeError):  # noqa: N818 - name fixed by DESIGN.md §18
    """Positive evidence that the key or the route is not Bitget Demo. The runtime exits with 3.

    Raised for 40099 on any ``--paper-trading`` call, for a live-negative probe that succeeds (the
    key works on the live venue), for a missing or undeclared Demo credential file, and by the
    transport when asked to send without a passed proof. ``proof`` carries the failed
    :class:`~sentiment_agent.types.EnvironmentProof` when one was built, and ``evidence`` the stored
    ``bgc`` answer that showed it, so the runtime can log both before exiting.
    """

    def __init__(
        self,
        reason: str,
        *,
        proof: EnvironmentProof | None = None,
        evidence: BlobRef | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.proof = proof
        self.evidence = evidence


# --- credentials --------------------------------------------------------------------------------


def base_child_env() -> dict[str, str]:
    """The non-secret part of every ``bgc`` child environment, built from scratch.

    Only :data:`CHILD_ENV_PASSTHROUGH` is copied from the parent. Nothing named ``BITGET_*`` is ever
    copied, so no inherited key, base URL or retry setting can change where ``bgc`` sends.
    """
    env: dict[str, str] = {}
    for name in CHILD_ENV_PASSTHROUGH:
        value = os.environ.get(name)
        if value:
            env[name] = value
    return env


class DemoCredentials:
    """A Bitget Demo API key, secret and passphrase. Never printed, logged, pickled or hashed whole.

    ``repr`` shows the source file and a 12-character fingerprint of the API key (a SHA-256 prefix,
    enough to tell two keys apart in the ledger, useless to anyone holding it). The values leave
    this object only through :meth:`child_env`, which the transport hands straight to the ``bgc``
    subprocess.
    """

    __slots__ = ("_api_key", "_passphrase", "_secret_key", "_source")

    def __init__(
        self, *, api_key: str, secret_key: str, passphrase: str, source: str = DEMO_ENV_FILE
    ) -> None:
        for name, value in (
            ("api key", api_key),
            ("secret key", secret_key),
            ("passphrase", passphrase),
        ):
            if not value or any(ch.isspace() or ord(ch) < 32 for ch in value):
                raise EnvironmentRefused(f"the Demo {name} is empty or contains whitespace")
        self._api_key = api_key
        self._secret_key = secret_key
        self._passphrase = passphrase
        self._source = source

    @property
    def source(self) -> str:
        return self._source

    @property
    def fingerprint(self) -> str:
        return sha256_hex(self._api_key.encode("utf-8"))[:12]

    def child_env(self) -> dict[str, str]:
        env = base_child_env()
        env[CREDENTIAL_VARS[0]] = self._api_key
        env[CREDENTIAL_VARS[1]] = self._secret_key
        env[CREDENTIAL_VARS[2]] = self._passphrase
        return env

    def __repr__(self) -> str:
        return f"DemoCredentials(source={self._source!r}, key=<redacted {self.fingerprint}>)"

    __str__ = __repr__

    def __reduce__(self) -> NoReturn:
        raise TypeError("credentials are never serialised")


def _parse_env_file(text: str) -> dict[str, str]:
    """``KEY=VALUE`` lines, ``export`` prefixes, ``#`` comment lines, optional matching quotes.

    No expansion and no inline comments (a passphrase may contain ``#``). A malformed line or a
    repeated key refuses the file; no message ever contains a value.
    """
    values: dict[str, str] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not _ENV_KEY.fullmatch(key):
            raise EnvironmentRefused(f"{DEMO_ENV_FILE} line {number} is not KEY=VALUE")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key in values:
            raise EnvironmentRefused(f"{DEMO_ENV_FILE} sets {key} more than once")
        values[key] = value
    return values


def load_demo_credentials(project_root: Path) -> DemoCredentials:
    """Read ``<project_root>/.secrets/demo.env``, or refuse.

    Refused when: the file does not exist; it resolves (through any symlink or junction) outside the
    project root; it is not a regular file or is implausibly large; it does not declare
    ``BITGET_KEY_ENVIRONMENT=demo``; or any of the three credential values is missing.
    """
    try:
        root = project_root.resolve(strict=True)
    except OSError as exc:
        raise EnvironmentRefused(f"project root is not readable ({type(exc).__name__})") from None
    candidate = root / DEMO_ENV_FILE
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        raise EnvironmentRefused(
            f"{DEMO_ENV_FILE} does not exist; the owner creates it with a Bitget Demo key "
            "(DESIGN.md §21). Until then only the simulated and dry-run modes can run."
        ) from None
    if not resolved.is_relative_to(root):
        raise EnvironmentRefused(
            f"{DEMO_ENV_FILE} resolves outside the project root; credentials are read from inside "
            "this repository only"
        )
    if not resolved.is_file():
        raise EnvironmentRefused(f"{DEMO_ENV_FILE} is not a regular file")
    if resolved.stat().st_size > MAX_ENV_FILE_BYTES:
        raise EnvironmentRefused(f"{DEMO_ENV_FILE} is larger than {MAX_ENV_FILE_BYTES} bytes")
    values = _parse_env_file(resolved.read_text(encoding="utf-8-sig"))
    declared = values.get(KEY_ENVIRONMENT_VAR)
    if declared is None or declared.strip().lower() != DEMO_DECLARATION:
        raise EnvironmentRefused(
            f"{DEMO_ENV_FILE} must declare {KEY_ENVIRONMENT_VAR}={DEMO_DECLARATION}; a key not "
            "marked Demo is never used"
        )
    missing = [name for name in CREDENTIAL_VARS if not values.get(name)]
    if missing:
        raise EnvironmentRefused(f"{DEMO_ENV_FILE} is missing {', '.join(missing)}")
    return DemoCredentials(
        api_key=values[CREDENTIAL_VARS[0]],
        secret_key=values[CREDENTIAL_VARS[1]],
        passphrase=values[CREDENTIAL_VARS[2]],
    )


# --- reading bgc's answers --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BgcFailure:
    """The structured error ``bgc`` printed to stderr (``toToolErrorPayload``, errors.ts:106-141).

    ``code`` is the SDK's code: Bitget's own code only for HTTP-200 errors, the HTTP status
    otherwise (see the module docstring). ``type`` is ``"CliError"`` for the CLI's own plain-text
    argument errors and ``"Unreadable"`` when stderr could not be parsed at all.
    """

    type: str
    code: str | None
    category: str | None
    message: str
    retryable: bool

    @property
    def environment_mismatch(self) -> bool:
        return self.code == ENVIRONMENT_MISMATCH_CODE or bool(
            _ENVIRONMENT_MISMATCH_TEXT.search(self.message)
        )

    @property
    def transient(self) -> bool:
        """Network, timeout, throttling or a 5xx: the call may not have been processed."""
        return self.type == "NetworkError" or self.category in {"network", "rate"}

    @property
    def local(self) -> bool:
        """Refused inside ``bgc`` before any request left the machine."""
        return self.type in {"ConfigError", "ValidationError", "CliError"}

    @property
    def label(self) -> str:
        return self.code or self.type


def failure_of(result: BgcResult) -> BgcFailure | None:
    """``None`` for a successful run; otherwise the error ``bgc`` reported."""
    if result.exit_code == 0:
        return None
    stderr = result.stderr or {}
    error = stderr.get("error")
    if isinstance(error, Mapping):
        code = error.get("code")
        category = error.get("category")
        return BgcFailure(
            type=str(error.get("type") or "Unknown"),
            code=None if code in (None, "") else str(code),
            category=None if category in (None, "") else str(category),
            message=str(error.get("message") or ""),
            retryable=error.get("retryable") is True,
        )
    text = stderr.get("text")
    if isinstance(text, str) and text:
        return BgcFailure(
            type="CliError", code=None, category=None, message=text[:2000], retryable=False
        )
    return BgcFailure(
        type="Unreadable",
        code=None,
        category=None,
        message=f"bgc exited {result.exit_code} with no readable error",
        retryable=False,
    )


def mentions_environment_mismatch(payload: Any) -> bool:
    """True when any part of a ``bgc`` output carries code 40099 or its message.

    Scans recursively, because ``account_overview`` reports a failed section as a nested message
    string with exit code 0.
    """
    if isinstance(payload, Mapping):
        if str(payload.get("code", "")) == ENVIRONMENT_MISMATCH_CODE:
            return True
        return any(mentions_environment_mismatch(v) for v in payload.values())
    if isinstance(payload, (list, tuple)):
        return any(mentions_environment_mismatch(v) for v in payload)
    if isinstance(payload, str):
        return bool(_ENVIRONMENT_MISMATCH_TEXT.search(payload))
    return False


def result_mentions_environment_mismatch(result: BgcResult) -> bool:
    return mentions_environment_mismatch(result.stdout) or mentions_environment_mismatch(
        result.stderr
    )


def is_key_refusal(failure: BgcFailure) -> bool:
    """The venue answered and refused the key itself (not the network, not a parameter)."""
    if failure.type == "AuthenticationError" or failure.category == "auth":
        return True
    return (
        failure.type == "BitgetApiError"
        and failure.code in _KEY_REFUSAL_HTTP
        and bool(_KEY_REFUSAL_TEXT.search(failure.message))
    )


IDENTITY_FIELDS: Final = frozenset({"uid", "userId", "inviterId", "parentId", "ips"})
"""Account identity in Bitget's answers (``account/settings`` carries ``uid``; ``account/info``
carries ``userId``, ``inviterId``, ``parentId`` and the IP whitelist). Blobs are published with the
ledger, so these values are replaced by :data:`REDACTED` before anything is stored."""

REDACTED: Final = "<redacted>"


def redact_identity(payload: Any) -> Any:
    """``payload`` with every :data:`IDENTITY_FIELDS` value replaced, at any depth."""
    if isinstance(payload, Mapping):
        return {
            key: REDACTED
            if key in IDENTITY_FIELDS and value not in (None, "")
            else redact_identity(value)
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [redact_identity(item) for item in payload]
    return payload


def result_blob(
    blobs: BlobStore, *, args: Sequence[str], result: BgcResult | None, note: str, at: str
) -> BlobRef:
    """Store one ``bgc`` call as evidence: argv, exit code, stdout, stderr.

    Never the environment (the credentials live only there), and never the account's identity
    (:func:`redact_identity`); everything else is kept exactly as ``bgc`` printed it.
    """
    record: dict[str, Any] = {"argv": list(args), "at": at, "note": note}
    if result is not None:
        record.update(
            exit_code=result.exit_code,
            stdout=redact_identity(result.stdout),
            stderr=redact_identity(result.stderr),
            duration_ms=result.duration_ms,
        )
    return blobs.put(canonical_json(record), "application/json")


# --- the vendor contract ------------------------------------------------------------------------

_SDK_PAPER_HEADER = re.compile(
    r"if\s*\(\s*this\.config\.paperTrading\s*&&\s*config\.auth\s*===\s*\"private\"\s*\)\s*\{\s*"
    r"headers\.set\(\s*\"paptrading\"\s*,\s*\"1\"\s*\)"
)
_SDK_EXCLUSION = re.compile(r"paperTrading and readOnly are mutually exclusive")
_SDK_DRY_RUN_FIRST = re.compile(r"if\s*\(\s*dryRun\s*\)\s*\{\s*return\s+previewResult\(")
_CLI_PAPER_FLAG = re.compile(r"paperTrading:\s*globals\[\"paper-trading\"\]\s*===\s*true")
_CLI_READ_ONLY_FLAG = re.compile(r"readOnly:\s*globals\[\"read-only\"\]\s*===\s*true")
_CLI_DRY_RUN_REMAP = re.compile(r"key\s*===\s*\"dry-run\"\s*\?\s*\"dryRun\"")


@dataclass(frozen=True, slots=True)
class VendorContract:
    """What the installed Agent Hub files were found to do. ``ok`` only when every check held."""

    ok: bool
    findings: tuple[str, ...]
    facts: Mapping[str, str]


def _package(dir_: Path) -> tuple[str | None, str | None]:
    try:
        data = json.loads((dir_ / "package.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None
    if not isinstance(data, dict):
        return None, None
    name, version = data.get("name"), data.get("version")
    return (
        name if isinstance(name, str) else None,
        version if isinstance(version, str) else None,
    )


def _file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _lib_text(package_dir: Path) -> str:
    lib = package_dir / "lib"
    parts: list[str] = []
    try:
        for path in sorted(lib.glob("*.js")):
            parts.append(path.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return ""
    return "\n".join(parts)


def vendor_contract(agent_hub_dir: Path) -> VendorContract:
    """Read the installed CLI and SDK and check the behaviour this project depends on."""
    findings: list[str] = []
    facts: dict[str, str] = {}
    modules = agent_hub_dir / "node_modules"
    cli_dir = modules / "@bitget-ai" / "bitget-agent-cli"
    nested_sdk = cli_dir / "node_modules" / "@bitget-ai" / "bitget-agent-sdk"
    sdk_dir = nested_sdk if nested_sdk.is_dir() else modules / "@bitget-ai" / "bitget-agent-sdk"

    cli_name, cli_version = _package(cli_dir)
    sdk_name, sdk_version = _package(sdk_dir)
    facts["cli_version"] = cli_version or "missing"
    facts["sdk_version"] = sdk_version or "missing"
    if cli_name != CLI_PACKAGE or cli_version != PINNED_VERSION:
        findings.append(
            f"installed CLI is {cli_name}@{cli_version}, expected {CLI_PACKAGE}@{PINNED_VERSION} "
            f"(run `npm ci --ignore-scripts` in {AGENT_HUB_DIR})"
        )
    if sdk_name != SDK_PACKAGE or sdk_version != PINNED_VERSION:
        findings.append(
            f"the SDK the CLI loads is {sdk_name}@{sdk_version}, expected "
            f"{SDK_PACKAGE}@{PINNED_VERSION}"
        )

    sdk_text = _lib_text(sdk_dir)
    cli_text = _lib_text(cli_dir)
    checks = (
        (_SDK_PAPER_HEADER, sdk_text, "the SDK sends paptrading: 1 on private calls"),
        (_SDK_EXCLUSION, sdk_text, "the SDK refuses --paper-trading together with --read-only"),
        (_SDK_DRY_RUN_FIRST, sdk_text, "the SDK returns a dry-run preview before any request"),
        (_CLI_PAPER_FLAG, cli_text, "the CLI maps --paper-trading to paperTrading"),
        (_CLI_READ_ONLY_FLAG, cli_text, "the CLI maps --read-only to readOnly"),
        (_CLI_DRY_RUN_REMAP, cli_text, "the CLI maps --dry-run to dryRun"),
    )
    for pattern, text, claim in checks:
        if not pattern.search(text):
            findings.append(f"not found in the installed code: {claim}")

    for label, path in (
        ("cli_index_sha256", cli_dir / "lib" / "index.js"),
        ("sdk_index_sha256", sdk_dir / "lib" / "index.js"),
        ("lockfile_sha256", agent_hub_dir / "package-lock.json"),
    ):
        digest = _file_sha256(path)
        facts[label] = digest or "missing"
        if digest is None:
            findings.append(f"{path.name} is missing ({label})")

    try:
        lock = json.loads((agent_hub_dir / "package-lock.json").read_text(encoding="utf-8"))
        packages = lock.get("packages", {}) if isinstance(lock, dict) else {}
    except (OSError, ValueError):
        packages = {}
    for package in (CLI_PACKAGE, SDK_PACKAGE):
        entry = packages.get(f"node_modules/{package}", {})
        locked = entry.get("version") if isinstance(entry, dict) else None
        if locked != PINNED_VERSION:
            findings.append(f"package-lock.json pins {package} at {locked}, not {PINNED_VERSION}")
    return VendorContract(ok=not findings, findings=tuple(findings), facts=facts)


def confirm_paptrading_header(agent_hub_dir: Path) -> bool:
    """True when the installed Agent Hub sends ``paptrading: 1`` under ``--paper-trading``.

    The full contract (versions, lockfile, the flag mappings, the dry-run early return) is checked,
    because the header rule is only as good as the CLI that reaches it. Findings are available from
    :func:`vendor_contract`.
    """
    return vendor_contract(agent_hub_dir).ok


# --- the proof --------------------------------------------------------------------------------


def _dig(payload: Any, *path: str) -> Any:
    node = payload
    for key in path:
        if not isinstance(node, Mapping):
            return None
        node = node.get(key)
    return node


def _decimal(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return parsed if parsed.is_finite() else None


def account_from_assets(data: Any, *, at: datetime, blob: BlobRef | None) -> AccountSnapshot:
    """``GET /api/v3/account/assets`` data (legacy-docs/uta/account/Get-Account, 2026-09-24).

    **Equity is the USDT asset row's equity, not the account's ``usdtEquity``.** The first real Demo
    read (2026-09-24) showed a unified account holding seventeen gifted demo coins — BTC, UNI, SOL,
    a pre-IPO token and more — worth 3.78M USDT in total beside 50,000 USDT. ``usdtEquity`` values
    all of them, so the book's starting equity would have been 3.78M and every BTC or UNI move would
    have opened an ``equity_gap`` in reconciliation and blocked new exposure. The strategy trades
    USDT-margined perpetuals only; its capital is the USDT it can margin with, and that is what the
    book starts from and is reconciled against. ``None`` when the account has no USDT row.
    ``available`` is read from the same row.
    """
    equity: Decimal | None = None
    available: Decimal | None = None
    assets = _dig(data, "assets")
    if isinstance(assets, list):
        for row in assets:
            if isinstance(row, Mapping) and row.get("coin") == "USDT":
                equity = _decimal(row.get("equity"))
                available = _decimal(row.get("available"))
                break
    return AccountSnapshot(at=at, equity_usdt=equity, available_usdt=available, blob=blob)


def _run_probe(
    runner: BgcRunner,
    args: Sequence[str],
    env: Mapping[str, str],
    blobs: BlobStore,
    clock: Clock,
    note: str,
) -> tuple[BgcResult | None, BlobRef, str | None]:
    at = clock.now().isoformat()
    try:
        result = runner(args, env=env, timeout_s=PROBE_TIMEOUT_S)
    except (TimeoutError, OSError) as exc:
        blob = result_blob(blobs, args=args, result=None, note=f"{note}: {exc}", at=at)
        return None, blob, f"{note} did not complete ({type(exc).__name__})"
    return result, result_blob(blobs, args=args, result=result, note=note, at=at), None


def prove_environment(
    project_root: Path, *, runner: BgcRunner, clock: Clock, blobs: BlobStore
) -> EnvironmentProof:
    """Build the environment proof (DESIGN.md §11.2). Never sends a write.

    Returns a proof with ``passed=False`` when any check fails or is inconclusive (a missing file, a
    vendor finding, a network failure, an unreadable hold mode). Raises
    :class:`EnvironmentRefused`, carrying the failed proof, on positive evidence that the key is
    live: 40099 on a Demo read, or the live venue accepting the key.

    The credential is handed to ``bgc`` only after the vendor contract holds.
    """
    checked_at = clock.now()
    reasons: list[str] = []
    detail: dict[str, str] = {
        "demo_overview_argv": " ".join(DEMO_OVERVIEW_ARGS),
        "demo_assets_argv": " ".join(DEMO_ASSETS_ARGS),
        "live_negative_argv": " ".join(LIVE_NEGATIVE_ARGS),
    }
    contract = vendor_contract(project_root / AGENT_HUB_DIR)
    detail.update(contract.facts)
    reasons.extend(contract.findings)
    bgc_package = f"{CLI_PACKAGE}@{contract.facts.get('cli_version', 'missing')}"

    credentials: DemoCredentials | None
    try:
        credentials = load_demo_credentials(project_root)
    except EnvironmentRefused as exc:
        credentials = None
        reasons.append(exc.reason)
    else:
        detail["key_fingerprint"] = credentials.fingerprint

    def proof(**checks: Any) -> EnvironmentProof:
        fields: dict[str, Any] = {
            "demo_read_ok": False,
            "demo_read_code": None,
            "live_read_rejected": False,
            "live_read_code": None,
            "hold_mode": None,
            "account": None,
            "passed": False,
        }
        fields.update(checks)
        return EnvironmentProof(
            checked_at=checked_at,
            mode=RunMode.PAPER,
            credentials_file=DEMO_ENV_FILE,
            key_declared_demo=credentials is not None,
            bgc_package=bgc_package,
            paptrading_header_confirmed=contract.ok,
            reasons=tuple(reasons),
            detail=dict(detail),
            **fields,
        )

    if credentials is None or not contract.ok:
        reasons.append("no probe was run: the credential or the installed CLI failed its checks")
        return proof()

    env = credentials.child_env()

    # 1. Demo-positive: the whole account through the composite.
    overview, overview_blob, overview_err = _run_probe(
        runner, DEMO_OVERVIEW_ARGS, env, blobs, clock, "demo-positive account_overview"
    )
    detail["demo_overview_blob"] = overview_blob.sha256
    sections: Mapping[str, Any] = {}
    if overview_err is not None:
        reasons.append(overview_err)
    elif overview is not None:
        if result_mentions_environment_mismatch(overview):
            reasons.append("a Demo read answered 40099: this key is not a Demo key")
            raise EnvironmentRefused(
                reasons[-1], proof=proof(demo_read_code="40099"), evidence=overview_blob
            )
        data = _dig(overview.stdout, "data")
        if overview.exit_code == 0 and isinstance(data, Mapping):
            sections = data
        else:
            failure = failure_of(overview)
            reasons.append(
                f"demo-positive account_overview failed: {failure.message if failure else '?'}"
            )
    failed_sections = sorted(
        name
        for name, section in sections.items()
        if not (isinstance(section, Mapping) and section.get("ok") is True)
    )
    for name in failed_sections:
        message = f"Demo {name} read failed: {_dig(sections, name, 'error')}"
        if name in REQUIRED_SECTIONS:
            reasons.append(message)
        else:
            detail[f"{name}_section"] = message[:300]
    sections_ok = (
        bool(sections)
        and set(sections) >= REQUIRED_SECTIONS
        and not REQUIRED_SECTIONS & set(failed_sections)
    )
    hold_mode_raw = _dig(sections, "settings", "data", "holdMode")
    hold_mode = hold_mode_raw if isinstance(hold_mode_raw, str) and hold_mode_raw else None
    if hold_mode not in HOLD_MODES:
        reasons.append(f"hold mode unreadable or unexpected: {hold_mode_raw!r}")
    account = (
        account_from_assets(
            _dig(sections, "assets", "data"),
            at=checked_at,
            blob=overview_blob,
        )
        if sections_ok
        else None
    )
    detail["account_mode"] = str(_dig(sections, "settings", "data", "accountMode") or "")
    detail["account_equity_usdt"] = (
        str(account.equity_usdt)
        if account is not None and account.equity_usdt is not None
        else "unreadable"
    )

    # 2. Demo-positive, code-bearing.
    assets, assets_blob, assets_err = _run_probe(
        runner, DEMO_ASSETS_ARGS, env, blobs, clock, "demo-positive raw getAccountAssets"
    )
    detail["demo_assets_blob"] = assets_blob.sha256
    demo_read_code: str | None = None
    if assets_err is not None:
        reasons.append(assets_err)
    elif assets is not None:
        if result_mentions_environment_mismatch(assets):
            reasons.append("a Demo read answered 40099: this key is not a Demo key")
            raise EnvironmentRefused(
                reasons[-1],
                proof=proof(demo_read_code="40099", hold_mode=hold_mode),
                evidence=assets_blob,
            )
        failure = failure_of(assets)
        if failure is None:
            demo_read_code = "00000"
        else:
            demo_read_code = failure.label
            reasons.append(f"demo-positive getAccountAssets failed: {failure.message}")
    demo_read_ok = sections_ok and demo_read_code == "00000"

    # 3. Live-negative: the same key must be refused by the live venue.
    live, live_blob, live_err = _run_probe(
        runner, LIVE_NEGATIVE_ARGS, env, blobs, clock, "live-negative raw getAccountAssets"
    )
    detail["live_negative_blob"] = live_blob.sha256
    live_rejected = False
    live_code: str | None = None
    if live_err is not None:
        reasons.append(f"live-negative inconclusive: {live_err}")
    elif live is not None:
        failure = failure_of(live)
        if failure is None:
            reasons.append(
                "the live venue ACCEPTED this key: it is a live key, and it is never used here"
            )
            raise EnvironmentRefused(
                reasons[-1],
                proof=proof(
                    demo_read_ok=demo_read_ok,
                    demo_read_code=demo_read_code,
                    live_read_code="00000",
                    hold_mode=hold_mode,
                    account=account,
                ),
                evidence=live_blob,
            )
        detail["live_read_message"] = failure.message[:500]
        detail["live_read_type"] = failure.type
        if failure.environment_mismatch:
            live_rejected, live_code = True, ENVIRONMENT_MISMATCH_CODE
        elif is_key_refusal(failure):
            live_rejected, live_code = True, failure.label
        else:
            live_code = failure.label
            reasons.append(
                f"live-negative inconclusive: the live venue did not refuse the key itself "
                f"({failure.type} {failure.code}: {failure.message[:200]})"
            )

    passed = (
        demo_read_ok
        and account is not None
        and live_rejected
        and hold_mode in HOLD_MODES
        and demo_read_code != ENVIRONMENT_MISMATCH_CODE
    )
    return proof(
        demo_read_ok=demo_read_ok,
        demo_read_code=demo_read_code,
        live_read_rejected=live_rejected,
        live_read_code=live_code,
        hold_mode=hold_mode,
        account=account,
        passed=passed,
    )


__all__ = [
    "AGENT_HUB_DIR",
    "CATEGORY",
    "CHILD_ENV_PASSTHROUGH",
    "CLI_ENTRY",
    "CLI_PACKAGE",
    "CREDENTIAL_VARS",
    "DEMO_ASSETS_ARGS",
    "DEMO_ENV_FILE",
    "DEMO_OVERVIEW_ARGS",
    "DRY_RUN_FLAG",
    "ENVIRONMENT_MISMATCH_CODE",
    "EXIT_ENVIRONMENT_REFUSED",
    "HOLD_MODES",
    "IDENTITY_FIELDS",
    "KEY_ENVIRONMENT_VAR",
    "LIVE_NEGATIVE_ARGS",
    "PAPER_FLAG",
    "PINNED_VERSION",
    "READ_ONLY_FLAG",
    "REDACTED",
    "REQUIRED_SECTIONS",
    "SDK_PACKAGE",
    "BgcFailure",
    "DemoCredentials",
    "EnvironmentRefused",
    "VendorContract",
    "account_from_assets",
    "base_child_env",
    "confirm_paptrading_header",
    "failure_of",
    "is_key_refusal",
    "load_demo_credentials",
    "mentions_environment_mismatch",
    "prove_environment",
    "redact_identity",
    "result_blob",
    "result_mentions_environment_mismatch",
    "vendor_contract",
]
