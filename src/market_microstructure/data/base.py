"""Source-independent normalized events. Raw payloads remain separately auditable."""

from dataclasses import asdict, dataclass
from typing import Protocol


class FeedError(ValueError):
    """Malformed message or detected continuity failure; start a new session."""


class FeedRejected(FeedError):
    """Subscription/product rejection; configuration must be corrected."""


@dataclass(frozen=True, slots=True)
class ReceiveContext:
    session_id: str
    message_index: int
    receive_time_ns: int
    receive_monotonic_ns: int


@dataclass(frozen=True, slots=True)
class LevelChange:
    side: str
    price: str
    quantity: str


@dataclass(frozen=True, slots=True)
class MarketEvent:
    kind: str
    source: str
    symbol: str
    context: ReceiveContext
    exchange_time: str | None = None
    exchange_time_ns: int | None = None
    sequence: int | None = None
    changes: tuple[LevelChange, ...] = ()
    price: str | None = None
    quantity: str | None = None
    maker_side: str | None = None
    aggressor_side: str | None = None
    trade_id: int | None = None
    last_trade_id: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class SourceAdapter(Protocol):
    source: str
    symbol: str
    endpoint: str

    metadata_url: str

    def continuity_tracker(self) -> "EventTracker": ...

    def parse_metadata(self, payload: str) -> dict: ...

    def subscription(self) -> dict: ...

    def parse(self, payload: str, context: ReceiveContext) -> MarketEvent: ...


class EventTracker(Protocol):
    def observe(self, event: MarketEvent) -> dict | None: ...
