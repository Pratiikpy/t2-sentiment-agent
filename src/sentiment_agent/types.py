"""The shared contract. Every value that crosses a module boundary is defined here, once.

Builders code against these types and nothing else, so modules can be written in parallel and still
fit. A module may define private helpers of its own; it may not invent a second shape for anything
listed here. If a shape is wrong, it is changed here, with its test, and every consumer follows.

Conventions that hold for every model:

* Frozen and ``extra="forbid"``: a field cannot be reassigned after the record is built, and a
  misspelt field is an error rather than a silently dropped value. Frozen is not deep: the contents
  of a ``dict`` field can still be mutated in place, and nothing in this project may do so, because
  a record's hash is taken from its contents.
* Money, prices and quantities are ``Decimal`` (exact, and the venue speaks strings). Features and
  statistics are ``float``.
* No ``NaN`` and no infinity, in any float or ``Decimal`` field at any depth. A non-finite number
  cannot be hashed canonically and cannot be compared honestly (``NaN > 0`` and ``NaN < 0`` are
  both false, which would let a ``NaN`` weight through the only-reduce check). An undefined
  statistic is ``None``.
* Every timestamp is timezone-aware UTC (:data:`UtcDatetime`). Naive datetimes are refused.
* A value that was not measured is ``None``, never ``0``. An empty result and an unreachable source
  are different facts (:class:`SourceHealth`).
* Weights are fractions of equity, signed: +0.05 is long 5% of equity, -0.05 is short 5%.

The protocols at the end are the seams between modules: market data, the two MCP services, crowd
text, the chat model, the ledger, and the venue transport.
"""

import copy
import enum
import itertools
import re
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import MappingProxyType
from typing import Annotated, Any, Final, Literal, Protocol, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from sentiment_agent.hashing import HASH_HEX_LENGTH, content_hash

CONTRACT_VERSION: Final = "1.1.0"
"""1.1.0 (run 2) adds the ``feed_health`` ledger event, a genesis's declared changes against the run
it follows, and the funding z-score scope of :class:`TriggerRule`. Each addition is written only
when it is set, so every 1.0.0 record (run 1's ledger included) still reads, re-serialises and
hashes exactly as it was written."""
PROJECT_SLUG: Final = "t2-sentiment-agent"


# ================================================================================================
# Primitives
# ================================================================================================


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("timestamps must be timezone-aware UTC")
    return value.astimezone(UTC)


UtcDatetime = Annotated[datetime, AfterValidator(_require_utc)]
"""A timezone-aware UTC datetime. Naive or non-UTC values are refused at construction."""

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _require_hex64(value: str) -> str:
    if not _HEX64.fullmatch(value):
        raise ValueError(f"expected a {HASH_HEX_LENGTH}-character lowercase hex digest")
    return value


Sha256Hex = Annotated[str, AfterValidator(_require_hex64)]

CLIENT_OID_PREFIX: Final = "sa"
CLIENT_OID_HEX: Final = 30
_CLIENT_OID = re.compile(rf"^{CLIENT_OID_PREFIX}[0-9a-f]{{{CLIENT_OID_HEX}}}$")


def _require_client_oid(value: str) -> str:
    if not _CLIENT_OID.fullmatch(value):
        raise ValueError(
            f"clientOid must be '{CLIENT_OID_PREFIX}' + {CLIENT_OID_HEX} hex characters "
            "(32 in total; derived from the approved intent hash, never random)"
        )
    return value


ClientOid = Annotated[str, AfterValidator(_require_client_oid)]
"""The order idempotency key: ``"sa"`` + the first 30 hex characters of the intent hash.

Deterministic on purpose: a retried or replayed intent must reach the venue under the same id so
Bitget's own ``clientOid`` de-duplication and our :class:`OrderState.UNKNOWN` reconciliation both
work. :class:`OrderIntent` refuses a ``client_oid`` that is not :func:`client_oid_of` its own
``intent_id``. Thirty-two characters keeps well inside the venue's limit (the exact UTA limit is
NOT VERIFIED; 32 is below every limit the SDK catalog documents)."""


def client_oid_of(intent_id: str) -> str:
    """The ``clientOid`` an intent with this ``intent_id`` must carry."""
    return CLIENT_OID_PREFIX + intent_id[:CLIENT_OID_HEX]


Weight = Annotated[float, Field(ge=-1.0, le=1.0, allow_inf_nan=False)]
"""Signed fraction of equity."""

Probability = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]


class Model(BaseModel):
    """Base for every contract model."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", validate_default=True, allow_inf_nan=False
    )

    def content_hash(self) -> str:
        """SHA-256 of the canonical JSON form (see :mod:`sentiment_agent.hashing`)."""
        return content_hash(self)

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        """A copy, re-validated when ``update`` changes anything.

        Pydantic's own ``model_copy(update=...)`` skips validation, which would let a copy carry
        what construction refuses (a ruling that adds exposure, a proof that passed unearned). Here
        the updated fields go through the validators like any new record. (``model_construct``
        skips validation by design and is banned from ``src/`` by a source scan.)
        """
        if not update:
            return super().model_copy(deep=deep)
        data = {name: getattr(self, name) for name in type(self).model_fields}
        data.update(update)
        if deep:
            data = copy.deepcopy(data)
        return self.model_validate(data)


# ================================================================================================
# Enumerations
# ================================================================================================


class RunMode(enum.StrEnum):
    """Where orders go. Each mode writes its own ledger file; they never mix."""

    SIMULATED = "simulated"
    """Fills are simulated against keyless Demo quotes. No Bitget credential is read."""
    DRYRUN = "dryrun"
    """Agent Hub builds every order with ``--dry-run``; nothing is sent. No credential is read."""
    PAPER = "paper"
    """Agent Hub sends orders with ``--paper-trading`` to Bitget UTA Demo. The only scored mode."""


class PriceSource(enum.StrEnum):
    DEMO = "demo"
    """Bitget UTA Demo prices: public v3 market endpoints sent with the ``paptrading: 1`` header."""
    LIVE = "live"
    """Bitget live public prices: the same endpoints without the header."""


class FillVenue(enum.StrEnum):
    BITGET_DEMO = "bitget_demo"
    SIMULATED = "simulated"


class Category(enum.StrEnum):
    USDT_FUTURES = "USDT-FUTURES"


class AssetClass(enum.StrEnum):
    CRYPTO = "crypto"
    US_EQUITY = "us_equity"
    US_INDEX = "us_index"

    @property
    def follows_us_session(self) -> bool:
        """True for the legs UTA Demo does not price on weekends (win plan §2.1)."""
        return self is not AssetClass.CRYPTO


class Side(enum.StrEnum):
    BUY = "buy"
    SELL = "sell"


class ToolkitSurface(enum.StrEnum):
    """Every Bitget surface (and the two non-Bitget crowd channels) the agent touches."""

    PUBLIC_MARKET_API = "bitget_public_market_api"
    AGENT_HUB_BGC = "agent_hub_bgc"
    SIGNAL_MCP = "bitget_signal_mcp"
    DATA_MCP = "bitget_mcp_server"
    GETAGENT_PLAYBOOK = "getagent_playbook"
    CROWD_X = "crowd_x"
    CROWD_REDDIT = "crowd_reddit"


class SourceHealth(enum.StrEnum):
    OK = "ok"
    EMPTY = "empty"
    """Answered with no rows."""
    HOLLOW = "hollow"
    """Answered with a well-formed envelope carrying no usable values."""
    ERROR = "error"
    TIMEOUT = "timeout"
    DISABLED = "disabled"
    """Deliberately not called (configuration), which is different from failing."""


class TriggerKind(enum.StrEnum):
    HEARTBEAT_US_OPEN = "heartbeat_us_open"
    HEARTBEAT_FUNDING = "heartbeat_funding"
    FEAR_GREED_EXTREME = "fear_greed_extreme"
    FUNDING_ZSCORE = "funding_zscore"
    OPEN_INTEREST_JUMP = "open_interest_jump"
    COORDINATED_CLUSTER = "coordinated_cluster"
    EARNINGS_EVENT = "earnings_event"
    FILING_EVENT = "filing_event"
    OWNER_MANUAL = "owner_manual"
    """Logged like any other trigger; counted and published as a human intervention."""


class ProtectiveReason(enum.StrEnum):
    """Why the kernel acted without a model decision. Protective actions only ever reduce."""

    STOP_FILLED = "stop_filled"
    DAILY_KILL = "daily_kill"
    WEEKEND_FREEZE = "weekend_freeze"
    VENUE_INTEGRITY = "venue_integrity"
    LLM_OUTAGE = "llm_outage"
    BREAKER = "breaker"


class Stance(enum.StrEnum):
    ACT = "act"
    """Change the book to the stated targets."""
    HOLD = "hold"
    """Keep every position as it is; every held symbol must still be addressed."""
    FLAT_WITH_REASONS = "flat_with_reasons"
    """No exposure, with written reasons. First-class, counted and published."""


class Thinking(enum.StrEnum):
    """Qwen reasoning tier (measured cost ratios in ARGUS ``llm/qwen.py``: FULL ≈ 5x LOW)."""

    OFF = "off"
    LOW = "low"
    FULL = "full"


class LlmOutcome(enum.StrEnum):
    DECIDED = "decided"
    INVALID_RESPONSE = "invalid_response"
    TRUNCATED = "truncated"
    TIMEOUT = "timeout"
    TRANSPORT_ERROR = "transport_error"
    BUDGET_EXHAUSTED = "budget_exhausted"

    @property
    def is_outage(self) -> bool:
        return self is not LlmOutcome.DECIDED


class GuardId(enum.StrEnum):
    """The eight pre-registered guards (win plan §3.2) and three structural checks."""

    G1_VENUE_INTEGRITY = "G1_venue_integrity"
    G2_WEEKEND_FREEZE = "G2_weekend_freeze"
    G3_SIZE = "G3_size"
    G4_STOP = "G4_stop"
    G5_DAILY_KILL = "G5_daily_kill"
    G6_TURNOVER = "G6_turnover"
    G7_FEE_BUDGET = "G7_fee_budget"
    G8_TAKER_ONLY = "G8_taker_only"
    G9_GROUNDING = "G9_grounding"
    G10_BREAKER = "G10_breaker"
    G11_ELIGIBILITY = "G11_eligibility"


PLAN_GUARDS: Final[frozenset[GuardId]] = frozenset(
    {
        GuardId.G1_VENUE_INTEGRITY,
        GuardId.G2_WEEKEND_FREEZE,
        GuardId.G3_SIZE,
        GuardId.G4_STOP,
        GuardId.G5_DAILY_KILL,
        GuardId.G6_TURNOVER,
        GuardId.G7_FEE_BUDGET,
        GuardId.G8_TAKER_ONLY,
    }
)
ALL_GUARDS: Final[frozenset[GuardId]] = frozenset(GuardId)
VENUE_GUARDS: Final[frozenset[GuardId]] = frozenset(
    {
        GuardId.G1_VENUE_INTEGRITY,
        GuardId.G2_WEEKEND_FREEZE,
        GuardId.G3_SIZE,
        GuardId.G4_STOP,
        GuardId.G5_DAILY_KILL,
        GuardId.G8_TAKER_ONLY,
        GuardId.G11_ELIGIBILITY,
    }
)
"""Guards that describe the venue and the mandate rather than the decision-maker. Baseline and rival
arms run under these, so an arm comparison isolates who decided, not which rules applied."""


class GuardStatus(enum.StrEnum):
    PASSED = "passed"
    FIRED = "fired"
    """The guard produced a ceiling below the proposal, or forced an exit."""
    NOT_EVALUATED = "not_evaluated"
    """An input was missing. Fail-closed: treated as FIRED for any increase in exposure."""
    NOT_APPLICABLE = "not_applicable"
    """The guard has nothing to say here (e.g. the weekend rule on a crypto leg)."""


class Activation(enum.StrEnum):
    """Circuit-breaker state. HALTED is the safe end; an unreadable state resolves to it."""

    ACTIVE = "active"
    REDUCE_ONLY = "reduce_only"
    HALTED = "halted"


class OrderState(enum.StrEnum):
    """Order lifecycle, after NautilusTrader ``crates/model/src/enums.rs:1304-1388``.

    ``EMULATED`` and ``RELEASED`` are omitted (no order emulation here). ``UNKNOWN`` is an addition:
    a timed-out request whose venue outcome is not known, resolvable only by reconciliation.
    """

    INITIALISED = "initialised"
    DENIED = "denied"
    """Refused by our own kernel; never sent."""
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    TRIGGERED = "triggered"
    PENDING_UPDATE = "pending_update"
    PENDING_CANCEL = "pending_cancel"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    REJECTED = "rejected"
    """Refused by the venue."""
    VOIDED = "voided"
    UNKNOWN = "unknown"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL_ORDER_STATES

    @property
    def is_live(self) -> bool:
        """Carries exposure the risk layer must count: open, in flight, or unknown."""
        return self in _LIVE_ORDER_STATES


_TERMINAL_ORDER_STATES: Final = frozenset(
    {
        OrderState.DENIED,
        OrderState.FILLED,
        OrderState.CANCELLED,
        OrderState.EXPIRED,
        OrderState.REJECTED,
        OrderState.VOIDED,
    }
)
_LIVE_ORDER_STATES: Final = frozenset(
    {
        OrderState.SUBMITTED,
        OrderState.ACCEPTED,
        OrderState.TRIGGERED,
        OrderState.PENDING_UPDATE,
        OrderState.PENDING_CANCEL,
        OrderState.PARTIALLY_FILLED,
        OrderState.UNKNOWN,
    }
)


class VenueOrderStatus(enum.StrEnum):
    """Bitget UTA v3 ``orderStatus`` (documented values, api-doc/uta/trade Get Order Details)."""

    LIVE = "live"
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"


class OrderPurpose(enum.StrEnum):
    OPEN = "open"
    INCREASE = "increase"
    REDUCE = "reduce"
    CLOSE = "close"
    PROTECTIVE_EXIT = "protective_exit"

    @property
    def adds_exposure(self) -> bool:
        return self in (OrderPurpose.OPEN, OrderPurpose.INCREASE)


class ArmKind(enum.StrEnum):
    OURS_GOVERNED = "ours_governed"
    TWIN_UNGOVERNED = "twin_ungoverned"
    BASELINE = "baseline"
    RIVAL = "rival"
    MIRROR_LIVE = "mirror_live"
    WEEKEND_COUNTERFACTUAL = "weekend_counterfactual"


class EventKind(enum.StrEnum):
    """Every kind of ledger event, with its payload model in :data:`EVENT_PAYLOADS`."""

    GENESIS = "genesis"
    AMENDMENT = "amendment"
    ENVIRONMENT_PROOF = "environment_proof"
    TOOLKIT_PROBE = "toolkit_probe"
    SNAPSHOT = "snapshot"
    TRIGGER = "trigger"
    DECISION = "decision"
    KERNEL_RULING = "kernel_ruling"
    ORDER_PLAN = "order_plan"
    ORDER_PREVIEW = "order_preview"
    ORDER_SUBMITTED = "order_submitted"
    ORDER_ACK = "order_ack"
    ORDER_REJECTED = "order_rejected"
    ORDER_UNKNOWN = "order_unknown"
    ORDER_STATE = "order_state"
    FILL = "fill"
    STOP_SYNC = "stop_sync"
    PROTECTIVE_ACTION = "protective_action"
    BREAKER_TRANSITION = "breaker_transition"
    RECONCILIATION = "reconciliation"
    MARK = "mark"
    BUDGET_STATE = "budget_state"
    ANCHOR = "anchor"
    HEALTH = "health"
    NOTE = "note"
    FEED_HEALTH = "feed_health"
    """Contract 1.1.0: a source started or stopped failing, or a trigger kind went blind."""


# ================================================================================================
# Blobs
# ================================================================================================


class BlobRef(Model):
    """A content-addressed raw payload (a full API response, a prompt, a completion)."""

    sha256: Sha256Hex
    media_type: str
    size: int = Field(ge=0)


# ================================================================================================
# Market data (module: venue)
# ================================================================================================

CandleKind = Literal["market", "mark", "index"]


class InstrumentSpec(Model):
    """One instrument as the venue describes it (``GET /api/v3/market/instruments``)."""

    symbol: str
    category: Category
    source: PriceSource
    base_coin: str
    quote_coin: str
    status: str
    min_order_qty: Decimal
    qty_step: Decimal
    """``quantityMultiplier``."""
    price_step: Decimal
    """``priceMultiplier``."""
    min_order_amount: Decimal
    """``minOrderAmount``, in quote coin."""
    max_market_order_qty: Decimal | None
    max_order_qty: Decimal | None
    taker_fee_rate: Decimal
    maker_fee_rate: Decimal
    max_leverage: int | None
    fund_interval_hours: int | None
    fetched_at: UtcDatetime


class Quote(Model):
    """One ticker row (``GET /api/v3/market/tickers``)."""

    symbol: str
    source: PriceSource
    ts: UtcDatetime
    """Venue timestamp of the ticker."""
    fetched_at: UtcDatetime
    last: Decimal
    mark: Decimal
    index: Decimal
    bid: Decimal
    ask: Decimal
    funding_rate: Decimal | None
    open_interest: Decimal | None
    turnover_24h: Decimal | None
    price_change_24h: Decimal | None
    """``price24hPcnt`` as a fraction."""

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / 2

    @property
    def spread_bps(self) -> float:
        mid = self.mid
        return float((self.ask - self.bid) / mid * 10_000) if mid > 0 else float("inf")

    @property
    def mark_index_gap(self) -> float:
        """``|mark / index - 1|`` as a fraction; ``inf`` when the index is not positive."""
        return float(abs(self.mark / self.index - 1)) if self.index > 0 else float("inf")


def _require_keyed(items: Mapping[str, Any], field: str) -> None:
    """Every value in a symbol-keyed map carries the symbol it is filed under."""
    for key, value in items.items():
        if value.symbol != key:
            raise ValueError(f"{field}[{key!r}] holds a record for {value.symbol!r}")


def _require_quotes(quotes: Mapping[str, Quote], source: PriceSource, field: str) -> None:
    """Demo and live quotes never share a map (DESIGN.md §6.1)."""
    _require_keyed(quotes, field)
    for key, quote in quotes.items():
        if quote.source is not source:
            raise ValueError(f"{field}[{key!r}] is a {quote.source.value} quote")


class Candle(Model):
    symbol: str
    source: PriceSource
    kind: CandleKind
    interval: str
    open_time: UtcDatetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal | None


class FundingPoint(Model):
    symbol: str
    source: PriceSource
    ts: UtcDatetime
    rate: Decimal


# ================================================================================================
# Toolkit sources (module: sources) and crowd text (module: crowd)
# ================================================================================================


class SourceCall(Model):
    """One call to one source, successful or not. Every snapshot carries all of them."""

    call_id: str
    surface: ToolkitSurface
    source: str
    """e.g. ``"sentiment_index.current"`` or ``"do_query:crypto_sentiment_crypto_fear_greed"``."""
    params: dict[str, str] = Field(default_factory=dict)
    health: SourceHealth
    started_at: UtcDatetime
    latency_ms: int = Field(ge=0)
    rows: int = Field(ge=0)
    blob: BlobRef | None
    error: str | None = None


class McpToolResult(Model):
    """What an MCP ``tools/call`` returned, before interpretation."""

    server: str
    tool: str
    is_error: bool
    structured: dict[str, Any] | None
    text: str
    raw: BlobRef | None


class MoodReading(Model):
    """Fear & Greed from every source that answered. Scales are 0-100 (pinned by fixture)."""

    crypto_fear_greed: int | None = Field(default=None, ge=0, le=100)
    crypto_fear_greed_label: str | None = None
    crypto_fear_greed_source: str | None = None
    crypto_fear_greed_alt: int | None = Field(default=None, ge=0, le=100)
    """The second crypto F&G source, kept for the agreement check."""
    crypto_fear_greed_alt_source: str | None = None
    market_fear_greed: int | None = Field(default=None, ge=0, le=100)
    market_fear_greed_label: str | None = None
    market_fear_greed_source: str | None = None


class DerivativesReading(Model):
    """Live derivatives positioning for one crypto symbol (bitget-signal, bitget-mcp-server)."""

    symbol: str
    retail_long_short_ratio: float | None
    top_trader_account_ratio: float | None
    top_trader_position_ratio: float | None
    taker_buy_sell_ratio: float | None
    open_interest_history: tuple[tuple[UtcDatetime, float], ...] = ()
    funding_rate: float | None = None


TextChannel = Literal["x", "reddit", "news", "filing"]


class TextItem(Model):
    """One piece of text somebody else wrote. Untrusted by definition."""

    item_id: str
    channel: TextChannel
    source: str
    """Outlet, subreddit or account."""
    url: str | None
    published_at: UtcDatetime
    fetched_at: UtcDatetime
    text: str
    symbols: tuple[str, ...] = ()


class Detection(Model):
    pattern: str
    severity: Literal["hostile", "flag"]
    span: str


class ScreenedItem(Model):
    """A text item after quarantine. ``prompt_text`` is the only form the model ever sees."""

    item: TextItem
    detections: tuple[Detection, ...]
    withheld: bool
    prompt_text: str
    """Spotlighted text, or the redaction marker when withheld."""


class StoryCluster(Model):
    cluster_id: str
    representative: str
    item_ids: tuple[str, ...]
    sources: tuple[str, ...]
    symbols: tuple[str, ...]
    first_seen: UtcDatetime
    last_seen: UtcDatetime
    distinct_sources: int = Field(ge=1)
    coordinated: bool
    velocity_per_hour: float | None


class CrowdReport(Model):
    items: int = Field(ge=0)
    withheld: int = Field(ge=0)
    distinct_stories: int = Field(ge=0)
    duplication_ratio: float
    clusters: tuple[StoryCluster, ...]
    mentions: dict[str, int]
    """Distinct-story mentions per universe symbol (copies counted once)."""


class CalendarItem(Model):
    symbol: str | None
    kind: Literal["earnings", "filing_8k", "form4", "macro"]
    at: UtcDatetime | None
    title: str
    source: str
    url: str | None = None


# ================================================================================================
# Perception snapshot (module: perception)
# ================================================================================================


class MarketMood(Model):
    crypto_fear_greed: int | None = Field(default=None, ge=0, le=100)
    crypto_fear_greed_label: str | None = None
    market_fear_greed: int | None = Field(default=None, ge=0, le=100)
    market_fear_greed_label: str | None = None
    crypto_sources_agree: bool | None = None
    """Both crypto F&G sources answered and sit in the same band."""


class PositioningFeatures(Model):
    """Everything the model and the fixed-rule baselines read about one instrument.

    Crowd positioning comes from **live** data (the real crowd); venue-integrity fields compare
    **Demo** with live (the paper venue). See DESIGN.md §6 for why the two are never mixed.
    """

    symbol: str
    asset_class: AssetClass
    demo_last: float | None
    live_last: float | None
    funding_rate_live: float | None
    funding_z_live: float | None
    open_interest_live: float | None
    oi_change_1h_pct: float | None
    oi_change_24h_pct: float | None
    retail_long_short_ratio: float | None
    top_trader_long_short_ratio: float | None
    taker_buy_sell_ratio: float | None
    price_change_24h_pct: float | None
    ma20_distance_atr: float | None
    """Distance of the live close from its 20-bar 1H mean, in 14-bar ATR units."""
    social_mentions_24h: int | None
    social_velocity_per_hour: float | None
    coordinated_cluster: bool
    demo_mark_index_gap_bps: float | None
    demo_live_gap_bps: float | None
    demo_spread_bps: float | None
    demo_index_move_bps_3h: float | None
    """Mean absolute 1H Demo index move over the last 3 hours (stale-index detector)."""
    next_earnings_at: UtcDatetime | None = None


class PerceptionSnapshot(Model):
    """One complete, logged picture of the world at decision time.

    Logged in full so that every baseline, rival arm and counterfactual can be recomputed from the
    log without calling anything again.
    """

    snapshot_id: str
    """``content_hash`` of the snapshot with this field set to ``""`` (see perception.snapshot)."""
    taken_at: UtcDatetime
    mode: RunMode
    policy_version: str
    universe: tuple[str, ...]
    demo_quotes: dict[str, Quote]
    live_quotes: dict[str, Quote]
    features: dict[str, PositioningFeatures]
    mood: MarketMood
    crowd: CrowdReport
    text: tuple[ScreenedItem, ...]
    calendar: tuple[CalendarItem, ...]
    source_calls: tuple[SourceCall, ...]
    facts: dict[str, float]
    """Flat ``name -> value`` map of every number the model is shown; the grounding reference."""

    @model_validator(mode="after")
    def _keyed_and_separated(self) -> "PerceptionSnapshot":
        _require_quotes(self.demo_quotes, PriceSource.DEMO, "demo_quotes")
        _require_quotes(self.live_quotes, PriceSource.LIVE, "live_quotes")
        _require_keyed(self.features, "features")
        return self

    def coverage(self) -> dict[str, SourceHealth]:
        return {call.source: call.health for call in self.source_calls}


# ================================================================================================
# Triggers (module: events)
# ================================================================================================


class Trigger(Model):
    trigger_id: str
    kind: TriggerKind
    fired_at: UtcDatetime
    symbols: tuple[str, ...]
    detail: str
    observed: float | None = None
    threshold: float | None = None
    source: str
    snapshot_id: str | None = None


# ================================================================================================
# Feed health (module: perception.feeds; contract 1.1.0)
# ================================================================================================

FAILING_HEALTH: Final[frozenset[SourceHealth]] = frozenset(
    {SourceHealth.ERROR, SourceHealth.TIMEOUT, SourceHealth.HOLLOW}
)
"""A source in one of these states was asked and gave nothing usable. ``EMPTY`` (answered, no rows)
and ``DISABLED`` (deliberately not asked) are not failures."""

BlindCause = Literal["feed_failed", "no_data", "cadence", "no_threshold"]
"""Why a trigger kind cannot fire on a snapshot: a source it reads failed or came back hollow; its
sources answered but carried nothing it can use; the snapshot does not read those sources by design
(a light snapshot, DESIGN.md §7); or no threshold was frozen for the instrument at genesis."""


class FeedAlarm(Model):
    """One source that is failing now, and for how long it has been."""

    feed: str
    """The source name exactly as the snapshot's :class:`SourceCall` records it."""
    surface: ToolkitSurface
    health: SourceHealth
    since: UtcDatetime
    """When the current failing streak began (the first failing snapshot of it)."""
    snapshots: int = Field(ge=1)
    """Consecutive snapshots that asked this source and got a failure."""
    error: str | None = None

    @model_validator(mode="after")
    def _a_failure(self) -> "FeedAlarm":
        if self.health not in FAILING_HEALTH:
            raise ValueError(f"an alarm is raised only for a failing source, not {self.health}")
        return self


class BlindTrigger(Model):
    """A trigger kind that could not fire on a snapshot, for these instruments, and why."""

    kind: TriggerKind
    symbols: tuple[str, ...]
    cause: BlindCause
    reason: str = Field(min_length=1)
    feeds: tuple[str, ...] = ()
    """The failing sources that blinded it (``feed_failed``); empty for every other cause."""

    @model_validator(mode="after")
    def _named(self) -> "BlindTrigger":
        if (self.cause == "feed_failed") != bool(self.feeds):
            raise ValueError("a feed_failed blindness names its feeds, and only that cause does")
        return self


class FeedHealthReport(Model):
    """The state of every source and trigger kind as of one snapshot.

    Logged as a ``feed_health`` event when an alarm is raised or cleared, or when the set of trigger
    kinds blinded by a failure changes; carried on every decision card.
    """

    at: UtcDatetime
    snapshot_id: str
    light: bool
    alarms: tuple[FeedAlarm, ...]
    raised: tuple[str, ...] = ()
    """Feeds failing on this snapshot that were not failing on the previous report."""
    cleared: tuple[str, ...] = ()
    """Feeds that were failing and answered on this snapshot."""
    blind: tuple[BlindTrigger, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> "FeedHealthReport":
        failing = [a.feed for a in self.alarms]
        if len(failing) != len(set(failing)):
            raise ValueError("a feed is alarmed twice")
        if not set(self.raised) <= set(failing):
            raise ValueError("a raised feed must be among the alarms")
        if set(self.cleared) & set(failing):
            raise ValueError("a cleared feed cannot still be alarmed")
        return self

    def failure_blind(self) -> tuple[BlindTrigger, ...]:
        """Blindness caused by a failing source (what an alarm is about)."""
        return tuple(b for b in self.blind if b.cause == "feed_failed")


# ================================================================================================
# Book state (module: book)
# ================================================================================================


class Position(Model):
    symbol: str
    qty: Decimal
    """Signed base quantity; positive is long."""
    avg_entry: Decimal
    opened_at: UtcDatetime
    last_increase_at: UtcDatetime
    realized_pnl: Decimal
    fees_paid: Decimal
    stop_price: Decimal | None
    stop_venue_id: str | None
    last_decision_id: str | None

    @property
    def is_flat(self) -> bool:
        return self.qty == 0


class BookState(Model):
    """The book at one instant, rebuilt from the ledger (never from memory)."""

    as_of: UtcDatetime
    mark_source: PriceSource
    starting_equity: Decimal
    equity: Decimal
    peak_equity: Decimal
    day_open_equity: Decimal
    """Equity at the most recent 00:00 UTC; the daily kill switch is measured from here."""
    positions: dict[str, Position]
    marks: dict[str, Decimal]
    fees_today: Decimal
    fees_total: Decimal
    realized_total: Decimal
    rebalances_today: dict[str, int]
    """Model-initiated orders per symbol since 00:00 UTC (protective orders excluded)."""
    consecutive_losses: int = Field(ge=0)
    activation: Activation

    @model_validator(mode="after")
    def _keyed(self) -> "BookState":
        _require_keyed(self.positions, "positions")
        return self

    def weight(self, symbol: str) -> float:
        pos = self.positions.get(symbol)
        mark = self.marks.get(symbol)
        if pos is None or pos.is_flat or mark is None or self.equity <= 0:
            return 0.0
        return float(pos.qty * mark / self.equity)

    @property
    def gross_weight(self) -> float:
        return sum(abs(self.weight(s)) for s in self.positions)

    @property
    def net_weight(self) -> float:
        return sum(self.weight(s) for s in self.positions)

    @property
    def day_return(self) -> float:
        return float(self.equity / self.day_open_equity - 1) if self.day_open_equity > 0 else 0.0

    @property
    def drawdown(self) -> float:
        """Non-positive fraction from peak."""
        return float(self.equity / self.peak_equity - 1) if self.peak_equity > 0 else 0.0


# ================================================================================================
# LLM transport (module: llm)
# ================================================================================================


class ChatMessage(Model):
    role: Literal["system", "user", "assistant"]
    content: str


class LlmUsage(Model):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    reasoning_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    cached_tokens: int = Field(default=0, ge=0)
    reported: bool
    """False when the endpoint sent no usage block: unmeasured, never free."""


class Completion(Model):
    content: str
    reasoning: str
    """Kept apart from ``content``; logged, never parsed as the answer."""
    usage: LlmUsage
    finish_reason: str
    raw_id: str
    latency_ms: int = Field(ge=0)


class BudgetState(Model):
    day: str
    """UTC date, ``YYYY-MM-DD``."""
    cap_tokens: int = Field(gt=0)
    spent_tokens: int = Field(ge=0)
    calls: int = Field(ge=0)
    unreported_calls: int = Field(ge=0)

    @property
    def remaining(self) -> int:
        return max(0, self.cap_tokens - self.spent_tokens)


# ================================================================================================
# Decision (module: decision)
# ================================================================================================


class Mandate(Model):
    """The risk budget the model must deploy or decline in writing (win plan Move 5g)."""

    risk_budget_gross: float = Field(gt=0, le=1)
    per_name_max: float = Field(gt=0, le=1)
    min_horizon_hours: int = Field(ge=1)
    text: str


class TargetProposal(Model):
    symbol: str
    target: Weight
    """In [-1, 1]; the weight is ``target * mandate.per_name_max``."""
    thesis: str = Field(min_length=1)
    invalidation: str = Field(min_length=1)
    horizon_hours: int = Field(ge=1)
    crowd_belief: str = Field(min_length=1)
    our_view: str = Field(min_length=1)
    confidence: Probability
    evidence: tuple[str, ...] = ()
    """Fact keys and text item ids the thesis relies on."""
    invalidation_triggered: bool = False
    """The model declares that a previously stated invalidation has fired for this position.

    This is the only thing that lets G6 permit an increase or a flip inside 24h of the last
    increase, so it must say what fired: ``invalidation_evidence`` is required with it."""
    invalidation_evidence: str | None = None

    @model_validator(mode="after")
    def _invalidation_has_evidence(self) -> "TargetProposal":
        if self.invalidation_triggered and not (self.invalidation_evidence or "").strip():
            raise ValueError("a declared invalidation must state the evidence that it fired")
        return self


class RejectedAlternative(Model):
    action: str
    reason: str


class LlmDecision(Model):
    """The model's answer, validated. Every currently held symbol must appear in ``targets``.

    Checked here, without the book: symbols are unique; ``flat_with_reasons`` carries at least one
    non-blank reason and no non-zero target; ``act`` and ``hold`` address at least one symbol.
    Because every held symbol must be addressed, ``hold`` with no targets would be a flat book kept
    flat without the written reasons the mandate requires, and ``act`` with no targets changes
    nothing. Checked against the book and the universe in ``decision/contract.py``: every held
    symbol addressed, no symbol outside the universe, horizon at least the mandate's.
    """

    stance: Stance
    targets: tuple[TargetProposal, ...]
    rejected_alternatives: tuple[RejectedAlternative, ...]
    mandate_response: str = Field(min_length=1)
    flat_reasons: tuple[str, ...] = ()
    summary: str = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> "LlmDecision":
        symbols = [t.symbol for t in self.targets]
        if len(symbols) != len(set(symbols)):
            raise ValueError("a symbol appears more than once in targets")
        if self.stance is Stance.FLAT_WITH_REASONS:
            if not any(r.strip() for r in self.flat_reasons):
                raise ValueError("flat_with_reasons needs at least one written reason")
            if any(t.target != 0 for t in self.targets):
                raise ValueError("flat_with_reasons cannot carry a non-zero target")
        elif not self.targets:
            raise ValueError(
                f"stance {self.stance.value!r} must address at least one symbol; "
                "an empty book kept empty is flat_with_reasons, with the reasons written"
            )
        return self

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(t.symbol for t in self.targets)


class GroundingFigure(Model):
    raw: str
    value: float
    unit: str
    context: str
    resolved: bool
    source: str | None
    known_value: float | None


class GroundingReport(Model):
    figures: tuple[GroundingFigure, ...]

    @property
    def unresolved(self) -> tuple[GroundingFigure, ...]:
        return tuple(f for f in self.figures if not f.resolved)

    @property
    def grounded(self) -> bool:
        return not self.unresolved

    @property
    def coverage(self) -> float:
        return 1.0 if not self.figures else 1 - len(self.unresolved) / len(self.figures)


class LlmCallRecord(Model):
    model: str
    thinking: Thinking
    prompt_version: str
    prompt_hash: Sha256Hex
    request_blob: BlobRef | None
    response_blobs: tuple[BlobRef, ...]
    attempts: int = Field(ge=0)
    usage: LlmUsage
    latency_ms: int = Field(ge=0)
    outcome: LlmOutcome
    error: str | None = None

    @model_validator(mode="after")
    def _decided_means_called(self) -> "LlmCallRecord":
        if self.outcome is LlmOutcome.DECIDED and self.attempts < 1:
            raise ValueError("a decided call made at least one attempt")
        return self


class DecisionRecord(Model):
    decision_id: str
    decided_at: UtcDatetime
    trigger_ids: tuple[str, ...]
    snapshot_id: str
    book_before: BookState
    mandate: Mandate
    policy_version: str
    call: LlmCallRecord
    outcome: LlmOutcome
    """Always equal to ``call.outcome``; repeated here because every consumer filters on it."""
    decision: LlmDecision | None
    grounding: dict[str, GroundingReport]
    """Per addressed symbol: the thesis, invalidation and our_view text checked against facts."""
    proposed_weights: dict[str, float]
    """One weight per addressed symbol, keyed exactly as ``decision.targets``: normally
    ``target * per_name_max`` (``decision/agent.py`` owns the rule). Empty when there is no
    decision."""

    @model_validator(mode="after")
    def _outcome_matches(self) -> "DecisionRecord":
        if self.outcome is not self.call.outcome:
            raise ValueError("the record's outcome must be the call's outcome")
        if (self.outcome is LlmOutcome.DECIDED) != (self.decision is not None):
            raise ValueError("a decision is present exactly when the outcome is DECIDED")
        if self.decision is None:
            if self.proposed_weights:
                raise ValueError("no decision means no proposed weights")
            if self.grounding:
                raise ValueError("no decision means no grounding reports")
            return self
        addressed = set(self.decision.symbols)
        if set(self.proposed_weights) != addressed:
            raise ValueError("proposed_weights must cover exactly the addressed symbols")
        if not set(self.grounding) <= addressed:
            raise ValueError("a grounding report names a symbol the decision did not address")
        return self


# ================================================================================================
# Risk kernel (module: kernel)
# ================================================================================================


class GuardRuling(Model):
    guard: GuardId
    symbol: str | None
    """``None`` for a book-level guard (daily kill, breaker, fee budget)."""
    status: GuardStatus
    ceiling_abs_weight: float | None = Field(default=None, ge=0)
    """Largest absolute weight this guard permits; ``None`` when it imposes no ceiling."""
    forces_exit: bool = False
    """The guard requires the instrument (or, at book level, every instrument) to be flat."""
    reason: str
    basis: str
    """The measured failure this guard answers, with its source (policy.GuardBasis)."""
    inputs: dict[str, float | str | bool | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _exit_is_a_firing(self) -> "GuardRuling":
        if self.forces_exit and self.status is not GuardStatus.FIRED:
            raise ValueError(f"{self.guard.value}: a guard that forces an exit has FIRED")
        return self

    @property
    def binds(self) -> bool:
        """FIRED, or NOT_EVALUATED (a missing input fails closed)."""
        return self.status in (GuardStatus.FIRED, GuardStatus.NOT_EVALUATED)


WEIGHT_EPS: Final = 1e-12
"""Tolerance for float comparisons of weights in the kernel's invariants."""


class InstrumentRuling(Model):
    """What the kernel did to one instrument. The only-reduce invariant is checked on construction.

    With ``reference = proposed_weight`` (or ``current_weight`` when there is no proposal, as in a
    protective ruling): ``approved`` has the sign of ``reference`` or is zero, and
    ``|approved| <= |reference|``. The kernel can shrink or refuse what was asked; it can never add
    exposure, keep exposure the model asked to cut, or turn a side.

    Also checked, so a ruling cannot contradict its own explanation: every guard ruling here is
    about this symbol; ``|approved|`` is within every ceiling reported here; a forced exit here
    means ``approved == 0``; and a ruling that changed the reference names its ``binding_guard``.
    Book-level rulings are checked against every instrument by :class:`KernelRuling`.
    """

    symbol: str
    current_weight: float
    proposed_weight: float | None
    approved_weight: float
    binding_guard: GuardId | None
    """The guard whose ceiling or exit produced ``approved_weight``. Required whenever the kernel
    changed the reference; it must have a binding (FIRED or NOT_EVALUATED) ruling for this symbol
    or at book level."""
    rulings: tuple[GuardRuling, ...]

    @property
    def reference(self) -> float:
        return self.current_weight if self.proposed_weight is None else self.proposed_weight

    @model_validator(mode="after")
    def _only_reduce(self) -> "InstrumentRuling":
        reference = self.reference
        if abs(self.approved_weight) > abs(reference) + WEIGHT_EPS:
            raise ValueError(
                f"{self.symbol}: approved {self.approved_weight} exceeds reference {reference}; "
                "the kernel may only reduce"
            )
        if self.approved_weight != 0 and (self.approved_weight > 0) != (reference > 0):
            raise ValueError(f"{self.symbol}: the kernel may never turn a side")
        for ruling in self.rulings:
            if ruling.symbol != self.symbol:
                raise ValueError(
                    f"{self.symbol}: carries a {ruling.guard.value} ruling for {ruling.symbol!r}"
                )
            _check_within(self.symbol, self.approved_weight, ruling)
        if self.changed_by_kernel and self.binding_guard is None:
            raise ValueError(f"{self.symbol}: the kernel changed the weight without naming a guard")
        return self

    @property
    def changed_by_kernel(self) -> bool:
        return abs(self.approved_weight - self.reference) > WEIGHT_EPS


def _check_within(symbol: str, approved: float, ruling: GuardRuling) -> None:
    if ruling.forces_exit and approved != 0:
        raise ValueError(f"{symbol}: {ruling.guard.value} forces an exit but {approved} was kept")
    ceiling = ruling.ceiling_abs_weight
    if ceiling is not None and abs(approved) > ceiling + WEIGHT_EPS:
        raise ValueError(
            f"{symbol}: approved {approved} is above the {ruling.guard.value} ceiling {ceiling}"
        )


class KernelRuling(Model):
    """One ruling over the book, answering one model decision or one protective reason.

    Checked on construction, beyond each :class:`InstrumentRuling`'s own checks: exactly one of
    ``decision_id`` and ``protective_reason``; a protective ruling carries no proposal; one entry
    per instrument; ``guards_applied`` without repeats, and every reported ruling and binding guard
    among them; book-level rulings have no symbol, and their exits and ceilings hold for every
    instrument; each binding guard has a binding ruling for its instrument or at book level.
    """

    ruling_id: str
    at: UtcDatetime
    decision_id: str | None
    protective_reason: ProtectiveReason | None
    activation_before: Activation
    activation_after: Activation
    book_rulings: tuple[GuardRuling, ...]
    instruments: tuple[InstrumentRuling, ...]
    guards_applied: tuple[GuardId, ...]

    @model_validator(mode="after")
    def _source(self) -> "KernelRuling":
        if (self.decision_id is None) == (self.protective_reason is None):
            raise ValueError("a ruling answers either a model decision or a protective reason")
        if self.protective_reason is not None and any(
            i.proposed_weight is not None for i in self.instruments
        ):
            raise ValueError("a protective ruling has no model proposal to rule on")
        symbols = [i.symbol for i in self.instruments]
        if len(symbols) != len(set(symbols)):
            raise ValueError("an instrument appears more than once in a ruling")
        applied = set(self.guards_applied)
        if len(applied) != len(self.guards_applied):
            raise ValueError("a guard appears more than once in guards_applied")
        for ruling in self.book_rulings:
            if ruling.symbol is not None:
                raise ValueError(f"book-level {ruling.guard.value} ruling names {ruling.symbol}")
        reported = [*self.book_rulings, *(r for i in self.instruments for r in i.rulings)]
        missing = {r.guard for r in reported} - applied
        if missing:
            raise ValueError(f"rulings reported for guards not applied: {sorted(missing)}")
        for inst in self.instruments:
            for ruling in self.book_rulings:
                _check_within(inst.symbol, inst.approved_weight, ruling)
            guard = inst.binding_guard
            if guard is None:
                continue
            if guard not in applied:
                raise ValueError(f"{inst.symbol}: binding guard {guard.value} was not applied")
            if not any(r.guard is guard and r.binds for r in (*inst.rulings, *self.book_rulings)):
                raise ValueError(
                    f"{inst.symbol}: binding guard {guard.value} has no FIRED or NOT_EVALUATED "
                    "ruling to show for it"
                )
        return self

    @property
    def changed_by_kernel(self) -> bool:
        return any(i.changed_by_kernel for i in self.instruments)

    def instrument(self, symbol: str) -> InstrumentRuling | None:
        for inst in self.instruments:
            if inst.symbol == symbol:
                return inst
        return None


class BreakerState(Model):
    activation: Activation
    since: UtcDatetime
    trips: tuple[str, ...]
    halted_until: UtcDatetime | None = None


class KernelInputs(Model):
    """Market facts the kernel rules on. Supplied by the caller; the kernel never fetches.

    ``specs`` are the Demo venue's instrument limits (orders go to Demo). Quote maps are keyed by
    their own symbol and hold only quotes from their own environment (DESIGN.md §6.1)."""

    at: UtcDatetime
    demo_quotes: dict[str, Quote]
    live_quotes: dict[str, Quote]
    specs: dict[str, InstrumentSpec]
    demo_index_move_bps_3h: dict[str, float | None]
    snapshot_id: str | None
    snapshot_taken_at: UtcDatetime | None
    venue_unreconciled: tuple[str, ...] = ()
    """Why the latest reconciliation could not confirm the book is the venue's (a fill it could not
    read or find, a position that differs). Non-empty refuses every increase (G10) until a sweep
    comes back without them."""

    @model_validator(mode="after")
    def _keyed_and_separated(self) -> "KernelInputs":
        _require_quotes(self.demo_quotes, PriceSource.DEMO, "demo_quotes")
        _require_quotes(self.live_quotes, PriceSource.LIVE, "live_quotes")
        _require_keyed(self.specs, "specs")
        for symbol, spec in self.specs.items():
            if spec.source is not PriceSource.DEMO:
                raise ValueError(f"specs[{symbol}] must be the Demo venue's instrument limits")
        return self


class RulingContext(Model):
    decision_id: str | None
    protective_reason: ProtectiveReason | None
    grounding: dict[str, GroundingReport] = Field(default_factory=dict)
    invalidation_fired: dict[str, bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _source(self) -> "RulingContext":
        if (self.decision_id is None) == (self.protective_reason is None):
            raise ValueError("a ruling answers either a model decision or a protective reason")
        return self


class OrderIntent(Model):
    """One order the planner derived from an approved ruling. Market orders only (G8).

    ``intent_id`` is the content hash of the intent's identity (ruling id, symbol, side, quantity,
    purpose, split index; ``kernel/planner.py``) and ``client_oid`` is :func:`client_oid_of` it, so
    a retry or a replay of the same intent reaches the venue under the same id.

    Checked on construction: an exposure-adding leg is not reduce-only and carries a venue stop on
    the losing side of its reference price (below it for a buy, above it for a sell); a reducing
    leg is reduce-only and carries no stop; the split index is in range; the ``clientOid`` derives
    from the ``intent_id``.
    """

    intent_id: Sha256Hex
    ruling_id: str
    decision_id: str | None
    symbol: str
    category: Category = Category.USDT_FUTURES
    side: Side
    qty: Decimal = Field(gt=0)
    order_type: Literal["market"] = "market"
    reduce_only: bool
    purpose: OrderPurpose
    reference_price: Decimal = Field(gt=0)
    notional: Decimal = Field(gt=0)
    expected_fee: Decimal = Field(ge=0)
    stop_loss_price: Decimal | None
    """Preset venue stop for OPEN/INCREASE legs (G4). ``None`` for reducing legs."""
    client_oid: ClientOid
    split_index: int = Field(default=0, ge=0)
    split_count: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _coherent(self) -> "OrderIntent":
        if self.purpose.adds_exposure:
            if self.reduce_only:
                raise ValueError("an exposure-adding leg cannot be reduce-only")
            stop = self.stop_loss_price
            if stop is None:
                raise ValueError("an exposure-adding leg must carry its venue stop (G4)")
            if stop <= 0:
                raise ValueError("a stop price must be positive")
            if self.side is Side.BUY and stop >= self.reference_price:
                raise ValueError("a long's stop must sit below its reference price")
            if self.side is Side.SELL and stop <= self.reference_price:
                raise ValueError("a short's stop must sit above its reference price")
        else:
            if not self.reduce_only:
                raise ValueError("a reducing leg must be reduce-only")
            if self.stop_loss_price is not None:
                raise ValueError("a reducing leg carries no stop; stops.py manages what remains")
        if self.split_index >= self.split_count:
            raise ValueError("split_index out of range")
        if self.client_oid != client_oid_of(self.intent_id):
            raise ValueError("client_oid must be 'sa' + the first 30 hex characters of intent_id")
        return self


class SkippedLeg(Model):
    symbol: str
    wanted_delta_weight: float
    reason: str


class OrderPlan(Model):
    plan_id: str
    ruling_id: str
    created_at: UtcDatetime
    intents: tuple[OrderIntent, ...]
    skipped: tuple[SkippedLeg, ...]

    @model_validator(mode="after")
    def _one_ruling_unique_ids(self) -> "OrderPlan":
        if any(i.ruling_id != self.ruling_id for i in self.intents):
            raise ValueError("every intent in a plan is planned under the plan's ruling")
        oids = [i.client_oid for i in self.intents]
        if len(oids) != len(set(oids)):
            raise ValueError("two intents in one plan share a clientOid")
        return self


_MINT_TOKEN: Final = object()
"""Held by ``kernel/approval.py`` (the sole production minter) and by tests. A source-scan test
fails the build if any other module references it."""


class ApprovedOrder:
    """An order intent the kernel approved. Constructable only by the kernel's minter.

    The executor accepts nothing else, so an order that never met the kernel cannot be expressed.
    The approval is bound to the intent's content: :attr:`approval_hash` covers the intent and the
    ruling id, and the executor recomputes it before sending (pattern: ARGUS
    ``decision/verdicts.py`` ``Authorised``, itself taken from Ballast ``enforcer.py``).

    Immutable once minted, and it cannot be pickled or copied: an approval exists only as the
    object the minter returned in this process, never as bytes that could be revived elsewhere.
    """

    __slots__ = ("_approval_hash", "_intent", "_ruling_id")
    _intent: OrderIntent
    _ruling_id: str
    _approval_hash: str

    def __init__(self, intent: OrderIntent, ruling_id: str, *, _token: object) -> None:
        if _token is not _MINT_TOKEN:
            raise PermissionError(
                "ApprovedOrder is minted only by the risk kernel (kernel/approval.py)"
            )
        if not isinstance(intent, OrderIntent):
            raise TypeError("an ApprovedOrder wraps an OrderIntent")
        if intent.ruling_id != ruling_id:
            raise ValueError("the intent was planned under a different ruling")
        object.__setattr__(self, "_intent", intent)
        object.__setattr__(self, "_ruling_id", ruling_id)
        object.__setattr__(self, "_approval_hash", approval_hash(intent, ruling_id))

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("an ApprovedOrder cannot be changed after it is minted")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("an ApprovedOrder cannot be changed after it is minted")

    def __reduce_ex__(self, protocol: object) -> Any:
        raise TypeError("an ApprovedOrder cannot be pickled or copied; ask the kernel for one")

    @property
    def intent(self) -> OrderIntent:
        return self._intent

    @property
    def ruling_id(self) -> str:
        return self._ruling_id

    @property
    def approval_hash(self) -> str:
        return self._approval_hash

    def verify(self) -> bool:
        return approval_hash(self._intent, self._ruling_id) == self._approval_hash

    def __repr__(self) -> str:
        i = self._intent
        return f"ApprovedOrder({i.symbol} {i.side} {i.qty} {i.purpose} oid={i.client_oid})"


def approval_hash(intent: OrderIntent, ruling_id: str) -> str:
    return content_hash({"intent": intent, "ruling_id": ruling_id})


# ================================================================================================
# Execution (module: execution)
# ================================================================================================


class DryRunPreview(Model):
    """What Agent Hub would send, captured with ``--dry-run`` before every real send."""

    client_oid: ClientOid
    operation_id: str
    method: str
    path: str
    would_send: dict[str, Any]
    argv: tuple[str, ...]
    """The exact ``bgc`` argv (no credentials are ever on the argv)."""
    captured_at: UtcDatetime
    blob: BlobRef | None


class OrderSubmitted(Model):
    client_oid: ClientOid
    intent: OrderIntent
    approval_hash: Sha256Hex
    submitted_at: UtcDatetime
    argv: tuple[str, ...]


class VenueAck(Model):
    kind: Literal["ack"] = "ack"
    client_oid: ClientOid
    venue_order_id: str
    acked_at: UtcDatetime
    blob: BlobRef | None


class VenueRejection(Model):
    kind: Literal["rejected"] = "rejected"
    client_oid: ClientOid
    code: str | None
    message: str
    category: str | None
    retryable: bool
    at: UtcDatetime
    blob: BlobRef | None


class VenueUnknown(Model):
    kind: Literal["unknown"] = "unknown"
    client_oid: ClientOid
    at: UtcDatetime
    reason: str


PlaceResult = Annotated[VenueAck | VenueRejection | VenueUnknown, Field(discriminator="kind")]


class FeeLine(Model):
    coin: str
    raw: Decimal
    """Exactly as the venue reported it. Sign convention pinned by the first Demo fill fixture."""


class VenueOrder(Model):
    """A row of ``/api/v3/trade/order-info`` or ``/history-orders`` (documented fields)."""

    venue_order_id: str
    client_oid: str | None
    symbol: str
    side: Side
    order_type: str
    qty: Decimal
    cum_exec_qty: Decimal
    cum_exec_value: Decimal
    avg_price: Decimal | None
    status: VenueOrderStatus
    reduce_only: bool | None
    delegate_type: str | None
    cancel_reason: str | None
    fees: tuple[FeeLine, ...]
    created_at: UtcDatetime
    updated_at: UtcDatetime
    blob: BlobRef | None


class Fill(Model):
    """A row of ``/api/v3/trade/fills`` (documented fields), or a simulated fill."""

    exec_id: str
    venue_order_id: str
    client_oid: str | None
    symbol: str
    side: Side
    exec_price: Decimal = Field(gt=0)
    exec_qty: Decimal = Field(gt=0)
    exec_value: Decimal
    fee_paid: Decimal
    """Positive means paid. Derived from ``feeDetail`` under the rule pinned by fixture."""
    fee_coin: str
    trade_scope: Literal["taker", "maker"] | None
    trade_side: Literal["open", "close"] | None
    exec_pnl: Decimal | None
    executed_at: UtcDatetime
    venue: FillVenue
    blob: BlobRef | None = None


class VenuePosition(Model):
    """Current position as the venue reports it. Only the fields pinned by fixture are parsed."""

    symbol: str
    qty: Decimal
    avg_price: Decimal | None
    blob: BlobRef | None


class VenueStopOrder(Model):
    symbol: str
    venue_id: str | None
    stop_price: Decimal | None
    blob: BlobRef | None


class AccountSnapshot(Model):
    at: UtcDatetime
    equity_usdt: Decimal | None
    available_usdt: Decimal | None
    blob: BlobRef | None


DEMO_CREDENTIALS_FILE: Final = ".secrets/demo.env"
"""The only credential path a passing :class:`EnvironmentProof` may name: relative to the project
root, so no local path reaches the published ledger (``execution/environment.py`` resolves it and
refuses anything that escapes the root)."""


class EnvironmentProof(Model):
    """Written to the ledger before the first paper order. ``passed`` gates every send.

    ``passed`` cannot be constructed ``True`` unless every check below passed (DESIGN.md §11.2):
    PAPER mode, the credentials read from :data:`DEMO_CREDENTIALS_FILE` and declared Demo, the
    installed SDK confirmed to send ``paptrading: 1``, the Demo-positive read succeeded without a
    40099 and returned the account, and the live-negative read was rejected.
    """

    checked_at: UtcDatetime
    mode: RunMode
    credentials_file: str
    """Project-relative, forward slashes, e.g. ``.secrets/demo.env``."""
    key_declared_demo: bool
    bgc_package: str
    """e.g. ``@bitget-ai/bitget-agent-cli@3.0.0``, with the lockfile hash in ``detail``."""
    paptrading_header_confirmed: bool
    """The installed SDK source sends ``paptrading: 1`` on private calls (vendor contract check)."""
    demo_read_ok: bool
    demo_read_code: str | None
    live_read_rejected: bool
    live_read_code: str | None
    hold_mode: str | None
    account: AccountSnapshot | None
    passed: bool
    reasons: tuple[str, ...]
    detail: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _passed_is_earned(self) -> "EnvironmentProof":
        earned = (
            self.mode is RunMode.PAPER
            and self.credentials_file.replace("\\", "/") == DEMO_CREDENTIALS_FILE
            and self.key_declared_demo
            and self.paptrading_header_confirmed
            and self.demo_read_ok
            and self.demo_read_code != "40099"
            and self.account is not None
            and self.live_read_rejected
        )
        if self.passed and not earned:
            raise ValueError("an environment proof cannot pass without every check passing")
        return self


class OrderStateChange(Model):
    client_oid: ClientOid
    venue_order_id: str | None
    from_state: OrderState
    to_state: OrderState
    at: UtcDatetime
    reason: str


class StopSync(Model):
    symbol: str
    action: Literal["preset", "placed", "replaced", "cancelled", "verified", "missing"]
    stop_price: Decimal | None
    venue_id: str | None
    at: UtcDatetime
    blob: BlobRef | None = None


class Discrepancy(Model):
    kind: Literal[
        "position_mismatch",
        "unknown_order",
        "missing_fill",
        "orphan_fill",
        "missing_stop",
        "orphan_stop",
        "equity_gap",
    ]
    symbol: str | None
    client_oid: str | None
    detail: str


UNRECONCILED_KINDS: Final = frozenset({"missing_fill", "position_mismatch"})
"""Discrepancies that mean the ledger's book may not be the venue's: until a sweep is free of them,
no instrument may add exposure (``kernel/guards.py`` G10)."""


class ReconciliationReport(Model):
    at: UtcDatetime
    orders_checked: int = Field(ge=0)
    new_fill_ids: tuple[str, ...]
    resolved_unknown: tuple[str, ...]
    discrepancies: tuple[Discrepancy, ...]
    account: AccountSnapshot | None
    fills_read: bool = True
    """Whether the venue's fills were read. Only a sweep that read them moves the fills window on,
    so a failed read can never leave a fill behind the window."""

    @property
    def clean(self) -> bool:
        return not self.discrepancies

    @property
    def unreconciled(self) -> tuple[str, ...]:
        """Why the book cannot be trusted to be the venue's, one line per ``missing_fill`` or
        ``position_mismatch`` discrepancy; empty when the sweep found neither."""
        return tuple(
            f"{d.kind}{'' if d.symbol is None else ' ' + d.symbol}: {d.detail}"[:300]
            for d in self.discrepancies
            if d.kind in UNRECONCILED_KINDS
        )


class ProtectiveAction(Model):
    at: UtcDatetime
    reason: ProtectiveReason
    symbols: tuple[str, ...]
    ruling_id: str
    detail: str


class BreakerTransition(Model):
    at: UtcDatetime
    from_state: Activation
    to_state: Activation
    trips: tuple[str, ...]


# ================================================================================================
# Ledger (module: ledger)
# ================================================================================================


class LedgerEvent(Model):
    """One line of the append-only, hash-chained log. Nothing is ever rewritten."""

    seq: int = Field(ge=0)
    ts: UtcDatetime
    kind: EventKind
    mode: RunMode
    payload: dict[str, Any]
    """``model_dump(mode="json")`` of the model registered for ``kind`` (:data:`EVENT_PAYLOADS`)."""
    blobs: tuple[BlobRef, ...] = ()
    prev_hash: Sha256Hex
    hash: Sha256Hex
    """``sha256(canonical({seq, ts, kind, mode, payload, blobs, prev_hash}))``."""


class MetricDefinition(Model):
    name: str
    formula: str
    unit: str
    notes: str = ""


class PredecessorRun(Model):
    """The paper run a genesis follows (contract 1.1.0): what the declared changes are against."""

    genesis_hash: Sha256Hex
    """The event hash of the earlier run's genesis, as published and posted on X."""
    code_commit: str
    policy_hash: Sha256Hex
    policy_version: str
    window: str
    """The earlier run's scored window, as its owner states it."""


ChangeKind = Literal["code_fix", "observability", "policy_amendment"]


class DeclaredChange(Model):
    """One change against the predecessor run, declared in the genesis before the first order.

    A ``policy_amendment`` names the policy it replaces and the one it installs, exactly as an
    :class:`Amendment` does inside a run; a code change names the files it touches and states that
    the policy and prompts are unchanged by it.
    """

    change_id: str = Field(min_length=1)
    kind: ChangeKind
    title: str = Field(min_length=1)
    detail: str = Field(min_length=1)
    files: tuple[str, ...]
    previous_policy_hash: Sha256Hex | None = None
    new_policy_hash: Sha256Hex | None = None
    evidence: dict[str, str] = Field(default_factory=dict)
    """Measured figures that motivated or size the change, each with where it can be recomputed."""

    @model_validator(mode="after")
    def _amendment_names_both(self) -> "DeclaredChange":
        amended = self.previous_policy_hash is not None or self.new_policy_hash is not None
        if (self.kind == "policy_amendment") != amended:
            raise ValueError("a policy amendment names both policy hashes, and only it does")
        if amended and (self.previous_policy_hash is None or self.new_policy_hash is None):
            raise ValueError("a policy amendment names both the old and the new policy hash")
        if self.previous_policy_hash is not None and (
            self.previous_policy_hash == self.new_policy_hash
        ):
            raise ValueError("an amendment must change the policy")
        return self


class Genesis(Model):
    """The pre-registration. Hashed, OpenTimestamps-anchored and posted before the first order."""

    project: str
    contract_version: str
    created_at: UtcDatetime
    mode: RunMode
    policy: "Policy"
    policy_hash: Sha256Hex
    prompt_hashes: dict[str, Sha256Hex]
    universe: tuple[str, ...]
    metric_definitions: tuple[MetricDefinition, ...]
    code_commit: str
    dependency_lock_hashes: dict[str, str]
    qwen_model: str
    bgc_package: str
    expected_envelope: dict[str, str]
    statement: str
    predecessor: PredecessorRun | None = None
    """Contract 1.1.0: the run this one follows, when there is one."""
    declared_changes: tuple[DeclaredChange, ...] = ()
    """Contract 1.1.0: every change against ``predecessor``, declared before the first order."""

    @model_serializer(mode="wrap")
    def _omit_unset_1_1(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data: dict[str, Any] = handler(self)
        if self.predecessor is None:
            data.pop("predecessor", None)
        if not self.declared_changes:
            data.pop("declared_changes", None)
        return data

    @model_validator(mode="after")
    def _hash_is_the_policy(self) -> "Genesis":
        if self.policy_hash != self.policy.content_hash():
            raise ValueError("policy_hash is not the hash of the policy it pre-registers")
        if self.universe != self.policy.symbols:
            raise ValueError("the genesis universe must be the policy's universe")
        if self.declared_changes and self.predecessor is None:
            raise ValueError("declared changes are against a predecessor run; name it")
        ids = [c.change_id for c in self.declared_changes]
        if len(ids) != len(set(ids)):
            raise ValueError("a declared change id appears twice")
        if self.predecessor is not None:
            amendments = [c for c in self.declared_changes if c.kind == "policy_amendment"]
            changed = self.policy_hash != self.predecessor.policy_hash
            if changed and not amendments:
                raise ValueError(
                    "the policy differs from the predecessor's, so a policy amendment is declared"
                )
            if amendments:
                if amendments[0].previous_policy_hash != self.predecessor.policy_hash:
                    raise ValueError("the first amendment must replace the predecessor's policy")
                if amendments[-1].new_policy_hash != self.policy_hash:
                    raise ValueError("the last amendment must install the policy pre-registered")
                for before, after in itertools.pairwise(amendments):
                    if after.previous_policy_hash != before.new_policy_hash:
                        raise ValueError("declared amendments must chain, each replacing the last")
        return self


class Amendment(Model):
    amendment_id: str
    at: UtcDatetime
    reason: str = Field(min_length=1)
    previous_policy_hash: Sha256Hex
    new_policy_hash: Sha256Hex
    new_policy: "Policy"
    owner_confirmed: bool

    @model_validator(mode="after")
    def _hash_is_the_policy(self) -> "Amendment":
        if self.new_policy_hash != self.new_policy.content_hash():
            raise ValueError("new_policy_hash is not the hash of new_policy")
        if self.new_policy_hash == self.previous_policy_hash:
            raise ValueError("an amendment must change the policy")
        return self


class AnchorRecord(Model):
    target_seq: int = Field(ge=0)
    target_hash: Sha256Hex
    submitted_at: UtcDatetime
    status: Literal["submitted", "upgraded", "failed"]
    ots_blob: BlobRef | None
    detail: str


class HealthBeat(Model):
    at: UtcDatetime
    iteration: int = Field(ge=0)
    activation: Activation
    open_positions: int = Field(ge=0)
    last_decision_at: UtcDatetime | None
    budget: BudgetState | None
    detail: str = ""


class Note(Model):
    at: UtcDatetime
    author: Literal["system", "owner"]
    text: str


class ToolkitUse(Model):
    """One row of the Bitget toolkit coverage matrix (win plan Move 21)."""

    surface: ToolkitSurface
    entry: str
    purpose: str
    judged_line: str
    used_in: tuple[str, ...]
    last_health: SourceHealth | None
    last_checked_at: UtcDatetime | None
    visible_at: str
    notes: str = ""


class ToolkitProbe(Model):
    at: UtcDatetime
    rows: tuple[ToolkitUse, ...]


# ================================================================================================
# Marks and analysis (modules: book, analysis, rivals, redteam, site)
# ================================================================================================


class PositionMark(Model):
    symbol: str
    qty: Decimal
    demo_mark: Decimal
    live_mark: Decimal | None
    demo_index: Decimal | None
    unrealized_demo: Decimal
    unrealized_live: Decimal | None


class MarkPoint(Model):
    """The book on the hour. The hourly equity series every Sharpe here is computed from."""

    at: UtcDatetime
    equity_book: Decimal
    equity_venue: Decimal | None
    """The venue's own account equity, read in PAPER mode; ``None`` elsewhere or when unread."""
    equity_live_mirror: Decimal | None
    gross_weight: float
    net_weight: float
    positions: tuple[PositionMark, ...]


class ClosedTrade(Model):
    """One round trip on one symbol: flat to flat (a flip closes one trade and opens another)."""

    symbol: str
    opened_at: UtcDatetime
    closed_at: UtcDatetime
    direction: Literal[1, -1]
    entry_avg: Decimal
    exit_avg: Decimal
    max_abs_qty: Decimal
    gross_pnl: Decimal
    fees: Decimal
    net_pnl: Decimal
    decision_ids: tuple[str, ...]
    exit_reason: str


class ArmSpec(Model):
    arm_id: str
    kind: ArmKind
    title: str
    description: str
    provenance: str
    """Code path, upstream repository and licence for anything taken from elsewhere."""
    uses_llm: bool
    guards: tuple[GuardId, ...]


class ArmMark(Model):
    at: UtcDatetime
    equity: float
    gross_weight: float
    net_weight: float


class MetricSet(Model):
    arm_id: str
    n_hours: int = Field(ge=0)
    n_closed_trades: int = Field(ge=0)
    total_return: float
    sharpe_ann: float | None
    sharpe_se_ann: float | None
    sortino_ann: float | None
    max_drawdown: float
    win_rate: float | None
    turnover: float
    fees_paid: float
    ci90: dict[str, tuple[float, float]]
    ci90_undefined_share: dict[str, float] = Field(default_factory=dict)
    """Per interval, the share of bootstrap resamples on which its statistic was undefined (only
    non-zero shares): the band in ``ci90`` rests on the rest (``analysis/bootstrap.py``)."""
    label: Literal["descriptive, not inferential"] = "descriptive, not inferential"


class ArmResult(Model):
    spec: ArmSpec
    marks: tuple[ArmMark, ...]
    trades: tuple[ClosedTrade, ...]
    metrics: MetricSet


class TwinIntervention(Model):
    decision_id: str
    ruling_id: str
    symbol: str
    guard: GuardId
    proposed_weight: float
    approved_weight: float
    pnl_ungoverned: float
    pnl_governed: float
    violations_prevented: tuple[GuardId, ...]


class TwinReport(Model):
    n_decisions: int = Field(ge=0)
    n_interventions: int = Field(ge=0)
    intervention_rate: float
    prevented_loss: float
    """Sum over interventions of ``max(0, pnl_governed - pnl_ungoverned)``, in equity fraction."""
    forgone_gain: float
    risk_violation_rate_ungoverned: float
    max_drawdown_governed: float
    max_drawdown_ungoverned: float
    human_takeovers: int = Field(ge=0)
    interventions: tuple[TwinIntervention, ...]


class CardOrder(Model):
    client_oid: ClientOid
    venue_order_id: str | None
    symbol: str
    side: Side
    qty: Decimal
    purpose: OrderPurpose
    state: OrderState
    fills: tuple[Fill, ...]
    net_pnl: Decimal | None


class DecisionCard(Model):
    """Everything a judge needs to check one decision, on one page (win plan Move 5a)."""

    card_id: str
    at: UtcDatetime
    decision_id: str | None
    ruling_id: str | None
    triggers: tuple[Trigger, ...]
    coverage: dict[str, SourceHealth]
    shown_text: tuple[ScreenedItem, ...]
    outcome: LlmOutcome | None
    decision: LlmDecision | None
    grounding: dict[str, GroundingReport]
    kernel: KernelRuling | None
    previews: tuple[DryRunPreview, ...]
    orders: tuple[CardOrder, ...]
    ledger_seqs: tuple[int, ...]
    blobs: tuple[BlobRef, ...]
    feed_health: FeedHealthReport | None = None
    """Contract 1.1.0: failing sources and blind trigger kinds on the snapshot the model read."""


class RedTeamVector(Model):
    vector_id: str
    family: str
    provenance: str
    """e.g. ``"HeyArka (MIT) vector 07"``, ``"AgentDojo (MIT) important_instructions"``."""
    payload_text: str
    attacker_goal: Literal["long", "short", "flat", "size_up"]
    target_symbol: str


class RedTeamOutcome(Model):
    vector_id: str
    arm_id: str
    snapshot_id: str
    clean_weight: float
    attacked_weight: float
    hijacked: bool
    order_would_send: bool
    stopped_by: Literal["quarantine", "novelty", "kernel", "model", "none"]
    detail: str


class RedTeamReport(Model):
    run_at: UtcDatetime
    arms: tuple[ArmSpec, ...]
    vectors: tuple[RedTeamVector, ...]
    outcomes: tuple[RedTeamOutcome, ...]
    hijack_rate: dict[str, float]
    qwen_tokens_spent: int = Field(ge=0)


# ================================================================================================
# Policy (values live in sentiment_agent.policy; the shapes live here)
# ================================================================================================


class UniverseEntry(Model):
    symbol: str
    asset_class: AssetClass
    demo_live_gap_p99_bps: float = Field(gt=0)
    basis: str


class WeekendRule(Model):
    """US-session legs are flat from ``freeze_start`` to ``reopen`` (UTC)."""

    freeze_weekday: int = Field(ge=0, le=6)
    """Monday is 0."""
    freeze_hour: int = Field(ge=0, le=23)
    reopen_weekday: int = Field(ge=0, le=6)
    reopen_hour: int = Field(ge=0, le=23)
    preflatten_minutes: int = Field(ge=0)
    no_open_buffer_hours: float = Field(ge=0)
    basis: str

    @model_validator(mode="after")
    def _a_real_window(self) -> "WeekendRule":
        if (self.freeze_weekday, self.freeze_hour) == (self.reopen_weekday, self.reopen_hour):
            raise ValueError("the weekend freeze must start and end at different times")
        if self.preflatten_minutes > self.no_open_buffer_hours * 60:
            raise ValueError("pre-flattening cannot start before new exposure is refused")
        return self


class BreakerRule(Model):
    reduce_only_drawdown: float = Field(gt=0, lt=1)
    halt_drawdown: float = Field(gt=0, lt=1)
    losing_streak_reduce_only: int = Field(ge=1)
    snapshot_max_age_minutes: int = Field(ge=1)
    quote_max_age_seconds: int = Field(ge=1)
    basis: str

    @model_validator(mode="after")
    def _ladder(self) -> "BreakerRule":
        if self.reduce_only_drawdown >= self.halt_drawdown:
            raise ValueError("reduce-only must trip before halt")
        return self


_HH_MM = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")

FUNDING_Z_CRYPTO_ONLY: Final[tuple[AssetClass, ...]] = (AssetClass.CRYPTO,)
"""Policy v1's ``funding_zscore`` scope: the crypto leg only."""


class TriggerRule(Model):
    us_open_local: str
    """``"09:30"`` America/New_York."""
    funding_heartbeat_hours_utc: tuple[int, ...]
    fear_greed_low: int = Field(ge=0, le=100)
    fear_greed_high: int = Field(ge=0, le=100)
    funding_z_threshold: float = Field(gt=0)
    funding_z_lookback_settlements: int = Field(ge=10)
    oi_jump_quantile: float = Field(gt=0.5, lt=1)
    oi_jump_lookback_days: int = Field(ge=1)
    coordinated_min_sources: int = Field(ge=2)
    coordinated_window_minutes: int = Field(ge=1)
    earnings_lookahead_hours: int = Field(ge=1)
    cooldown_minutes: int = Field(ge=0)
    max_event_decisions_per_day: int = Field(ge=0)
    basis: str
    funding_z_asset_classes: tuple[AssetClass, ...] = FUNDING_Z_CRYPTO_ONLY
    """Which instruments ``funding_zscore`` is evaluated for, by asset class. Policy v1 read the
    crypto leg only (BTCUSDT); policy v2 reads every class (contract 1.1.0). Written into the
    policy, and so into its hash, only when it differs from the v1 scope, which keeps v1's hash
    and every record written under it exactly as they were."""

    @model_serializer(mode="wrap")
    def _omit_v1_scope(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data: dict[str, Any] = handler(self)
        if self.funding_z_asset_classes == FUNDING_Z_CRYPTO_ONLY:
            data.pop("funding_z_asset_classes", None)
        return data

    @model_validator(mode="after")
    def _coherent(self) -> "TriggerRule":
        scope = self.funding_z_asset_classes
        if not scope or len(set(scope)) != len(scope):
            raise ValueError("funding_z_asset_classes must name distinct asset classes")
        if not _HH_MM.fullmatch(self.us_open_local):
            raise ValueError("us_open_local must be HH:MM")
        hours = self.funding_heartbeat_hours_utc
        if len(set(hours)) != len(hours) or any(not 0 <= h <= 23 for h in hours):
            raise ValueError("funding heartbeat hours must be distinct UTC hours 0-23")
        if self.fear_greed_low >= self.fear_greed_high:
            raise ValueError("the fear band must sit below the greed band")
        return self


class DecisionRule(Model):
    model: str
    thinking_heartbeat: Thinking
    thinking_event: Thinking
    daily_token_cap: int = Field(gt=0)
    max_attempts: int = Field(ge=1)
    max_completion_tokens: int = Field(ge=256)
    call_timeout_seconds: int = Field(ge=10)
    min_horizon_hours: int = Field(ge=1)
    temperature: float = Field(ge=0, le=2)
    basis: str


class GuardBasis(Model):
    guard: GuardId
    rule: str
    basis: str


class Policy(Model):
    """Everything the kernel, triggers and agent obey. Frozen at genesis; changed by amendment."""

    version: str
    universe: tuple[UniverseEntry, ...]
    excluded: tuple[str, ...]
    per_name_max: float = Field(gt=0, le=1)
    gross_max: float = Field(gt=0, le=1)
    stop_loss_pct: float = Field(gt=0, lt=1)
    stop_trigger: Literal["mark"]
    daily_kill_pct: float = Field(gt=0, lt=1)
    max_rebalances_per_name_per_day: int = Field(ge=1)
    min_hold_hours: int = Field(ge=0)
    fee_budget_daily_bps: float = Field(gt=0)
    fee_budget_window_bps: float = Field(gt=0)
    taker_only: Literal[True]
    max_open_spread_bps: float = Field(gt=0)
    mark_index_max_gap: float = Field(gt=0, lt=1)
    stale_index_min_move_bps_3h: float = Field(gt=0)
    grounding_tolerance: float = Field(gt=0, lt=1)
    weekend: WeekendRule
    breaker: BreakerRule
    triggers: TriggerRule
    decision: DecisionRule
    mandate: Mandate
    guard_bases: tuple[GuardBasis, ...]
    metrics: tuple[MetricDefinition, ...]
    expected_envelope: dict[str, str]

    @model_validator(mode="after")
    def _coherent(self) -> "Policy":
        symbols = [u.symbol for u in self.universe]
        if len(symbols) != len(set(symbols)):
            raise ValueError("duplicate symbol in universe")
        if set(symbols) & set(self.excluded):
            raise ValueError("a symbol is both in the universe and excluded")
        if self.per_name_max > self.gross_max:
            raise ValueError("per-name cap above the gross cap")
        if self.mandate.per_name_max != self.per_name_max:
            raise ValueError("mandate per-name cap disagrees with the policy")
        if self.mandate.risk_budget_gross > self.gross_max:
            raise ValueError("mandate risk budget above the gross cap")
        if self.mandate.min_horizon_hours != self.decision.min_horizon_hours:
            raise ValueError("mandate horizon disagrees with the decision rule")
        guards = [b.guard for b in self.guard_bases]
        if len(guards) != len(set(guards)) or set(guards) != set(GuardId):
            raise ValueError("every guard needs exactly one written, measured basis")
        names = [m.name for m in self.metrics]
        if len(names) != len(set(names)):
            raise ValueError("a metric is defined twice")
        return self

    def entry(self, symbol: str) -> UniverseEntry | None:
        for u in self.universe:
            if u.symbol == symbol:
                return u
        return None

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(u.symbol for u in self.universe)


Genesis.model_rebuild()
Amendment.model_rebuild()


# ================================================================================================
# Ledger payload registry
# ================================================================================================


class DecisionEvent(Model):
    record: DecisionRecord


class SnapshotEvent(Model):
    snapshot: PerceptionSnapshot


EVENT_PAYLOADS: Final[Mapping[EventKind, type[Model]]] = MappingProxyType(
    {
        EventKind.GENESIS: Genesis,
        EventKind.AMENDMENT: Amendment,
        EventKind.ENVIRONMENT_PROOF: EnvironmentProof,
        EventKind.TOOLKIT_PROBE: ToolkitProbe,
        EventKind.SNAPSHOT: SnapshotEvent,
        EventKind.TRIGGER: Trigger,
        EventKind.DECISION: DecisionEvent,
        EventKind.KERNEL_RULING: KernelRuling,
        EventKind.ORDER_PLAN: OrderPlan,
        EventKind.ORDER_PREVIEW: DryRunPreview,
        EventKind.ORDER_SUBMITTED: OrderSubmitted,
        EventKind.ORDER_ACK: VenueAck,
        EventKind.ORDER_REJECTED: VenueRejection,
        EventKind.ORDER_UNKNOWN: VenueUnknown,
        EventKind.ORDER_STATE: OrderStateChange,
        EventKind.FILL: Fill,
        EventKind.STOP_SYNC: StopSync,
        EventKind.PROTECTIVE_ACTION: ProtectiveAction,
        EventKind.BREAKER_TRANSITION: BreakerTransition,
        EventKind.RECONCILIATION: ReconciliationReport,
        EventKind.MARK: MarkPoint,
        EventKind.BUDGET_STATE: BudgetState,
        EventKind.ANCHOR: AnchorRecord,
        EventKind.HEALTH: HealthBeat,
        EventKind.NOTE: Note,
        EventKind.FEED_HEALTH: FeedHealthReport,
    }
)
"""Read-only: the payload model each ledger event kind must carry."""


def parse_payload(event: LedgerEvent) -> Model:
    """The typed payload of a ledger event."""
    return EVENT_PAYLOADS[event.kind].model_validate(event.payload)


# ================================================================================================
# Protocols: the seams between modules
# ================================================================================================


class Clock(Protocol):
    def now(self) -> datetime: ...


class BlobStore(Protocol):
    def put(self, data: bytes, media_type: str) -> BlobRef: ...

    def get(self, sha256: str) -> bytes: ...


class LedgerWriter(Protocol):
    @property
    def mode(self) -> RunMode: ...

    def append(
        self, kind: EventKind, payload: Model, *, blobs: Sequence[BlobRef] = ()
    ) -> LedgerEvent: ...


class LedgerReader(Protocol):
    def events(self, kinds: frozenset[EventKind] | None = None) -> Iterator[LedgerEvent]: ...

    def head(self) -> LedgerEvent | None: ...


class MarketData(Protocol):
    """Keyless Bitget public market data, live or Demo (module: venue)."""

    def instruments(
        self, source: PriceSource, symbols: Sequence[str]
    ) -> dict[str, InstrumentSpec]: ...

    def quotes(self, source: PriceSource, symbols: Sequence[str]) -> dict[str, Quote]: ...

    def candles(
        self,
        source: PriceSource,
        symbol: str,
        *,
        kind: CandleKind,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> list[Candle]: ...

    def funding_history(self, symbol: str, *, limit: int) -> list[FundingPoint]: ...


class McpCaller(Protocol):
    """One MCP server over streamable HTTP (module: sources)."""

    @property
    def server(self) -> str: ...

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> McpToolResult: ...


class ToolkitReader(Protocol):
    """bitget-signal and bitget-mcp-server, interpreted into typed readings (module: sources)."""

    def mood(self) -> tuple[MoodReading, tuple[SourceCall, ...]]: ...

    def derivatives(self, symbol: str) -> tuple[DerivativesReading, tuple[SourceCall, ...]]: ...

    def news(self, limit: int) -> tuple[tuple[TextItem, ...], tuple[SourceCall, ...]]: ...

    def reddit_trending(
        self, limit: int
    ) -> tuple[tuple[TextItem, ...], tuple[SourceCall, ...]]: ...

    def calendar(
        self, symbols: Sequence[str], *, since: datetime
    ) -> tuple[tuple[CalendarItem, ...], tuple[SourceCall, ...]]: ...


class CrowdCollector(Protocol):
    """X and Reddit text (module: crowd). Optional; absence is logged, not fatal."""

    def collect(
        self, symbols: Sequence[str], *, since: datetime
    ) -> tuple[tuple[TextItem, ...], tuple[SourceCall, ...]]: ...


class ChatModel(Protocol):
    """Qwen, or a recorded/fake stand-in (module: llm). Tests never reach a real endpoint."""

    @property
    def model_name(self) -> str: ...

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_mode: bool,
        max_tokens: int,
        thinking: Thinking,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> Completion: ...


class VenueTransport(Protocol):
    """Agent Hub ``bgc --paper-trading`` or the simulated venue (module: execution)."""

    @property
    def venue(self) -> FillVenue: ...

    def preview(self, intent: OrderIntent) -> DryRunPreview: ...

    def place(self, order: ApprovedOrder) -> VenueAck | VenueRejection | VenueUnknown: ...

    def order(self, *, client_oid: str) -> VenueOrder | None: ...

    def fills(self, *, since: datetime, until: datetime) -> list[Fill]: ...

    def positions(self) -> list[VenuePosition]: ...

    def stop_orders(self) -> list[VenueStopOrder]: ...

    def account(self) -> AccountSnapshot: ...


__all__ = [
    "ALL_GUARDS",
    "CLIENT_OID_HEX",
    "CLIENT_OID_PREFIX",
    "CONTRACT_VERSION",
    "DEMO_CREDENTIALS_FILE",
    "EVENT_PAYLOADS",
    "FAILING_HEALTH",
    "FUNDING_Z_CRYPTO_ONLY",
    "PLAN_GUARDS",
    "PROJECT_SLUG",
    "UNRECONCILED_KINDS",
    "VENUE_GUARDS",
    "WEIGHT_EPS",
    "AccountSnapshot",
    "Activation",
    "Amendment",
    "AnchorRecord",
    "ApprovedOrder",
    "ArmKind",
    "ArmMark",
    "ArmResult",
    "ArmSpec",
    "AssetClass",
    "BlindCause",
    "BlindTrigger",
    "BlobRef",
    "BlobStore",
    "BookState",
    "BreakerRule",
    "BreakerState",
    "BreakerTransition",
    "BudgetState",
    "CalendarItem",
    "Candle",
    "CandleKind",
    "CardOrder",
    "Category",
    "ChangeKind",
    "ChatMessage",
    "ChatModel",
    "ClientOid",
    "Clock",
    "ClosedTrade",
    "Completion",
    "CrowdCollector",
    "CrowdReport",
    "DecisionCard",
    "DecisionEvent",
    "DecisionRecord",
    "DecisionRule",
    "DeclaredChange",
    "DerivativesReading",
    "Detection",
    "Discrepancy",
    "DryRunPreview",
    "EnvironmentProof",
    "EventKind",
    "FeeLine",
    "FeedAlarm",
    "FeedHealthReport",
    "Fill",
    "FillVenue",
    "FundingPoint",
    "Genesis",
    "GroundingFigure",
    "GroundingReport",
    "GuardBasis",
    "GuardId",
    "GuardRuling",
    "GuardStatus",
    "HealthBeat",
    "InstrumentRuling",
    "InstrumentSpec",
    "KernelInputs",
    "KernelRuling",
    "LedgerEvent",
    "LedgerReader",
    "LedgerWriter",
    "LlmCallRecord",
    "LlmDecision",
    "LlmOutcome",
    "LlmUsage",
    "Mandate",
    "MarkPoint",
    "MarketData",
    "MarketMood",
    "McpCaller",
    "McpToolResult",
    "MetricDefinition",
    "MetricSet",
    "Model",
    "MoodReading",
    "Note",
    "OrderIntent",
    "OrderPlan",
    "OrderPurpose",
    "OrderState",
    "OrderStateChange",
    "OrderSubmitted",
    "PerceptionSnapshot",
    "PlaceResult",
    "Policy",
    "Position",
    "PositionMark",
    "PositioningFeatures",
    "PredecessorRun",
    "PriceSource",
    "Probability",
    "ProtectiveAction",
    "ProtectiveReason",
    "Quote",
    "ReconciliationReport",
    "RedTeamOutcome",
    "RedTeamReport",
    "RedTeamVector",
    "RejectedAlternative",
    "RulingContext",
    "RunMode",
    "ScreenedItem",
    "Sha256Hex",
    "Side",
    "SkippedLeg",
    "SnapshotEvent",
    "SourceCall",
    "SourceHealth",
    "Stance",
    "StopSync",
    "StoryCluster",
    "TargetProposal",
    "TextChannel",
    "TextItem",
    "Thinking",
    "ToolkitProbe",
    "ToolkitReader",
    "ToolkitSurface",
    "ToolkitUse",
    "Trigger",
    "TriggerKind",
    "TriggerRule",
    "TwinIntervention",
    "TwinReport",
    "UniverseEntry",
    "UtcDatetime",
    "VenueAck",
    "VenueOrder",
    "VenueOrderStatus",
    "VenuePosition",
    "VenueRejection",
    "VenueStopOrder",
    "VenueTransport",
    "VenueUnknown",
    "WeekendRule",
    "Weight",
    "approval_hash",
    "client_oid_of",
    "parse_payload",
]
