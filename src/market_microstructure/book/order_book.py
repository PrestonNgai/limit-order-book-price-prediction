"""Exact-price, feed-independent Level 2 order-book reconstruction.

Updates replace displayed quantity. Apply a whole exchange message at once:
an intermediate state may cross even when the final book is valid.
The book assumes a single writer.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from heapq import nlargest, nsmallest
from types import MappingProxyType
from typing import TypeAlias

from .snapshot import BookSnapshot, Level, quote_values

DecimalInput: TypeAlias = Decimal | str | int
RawLevel: TypeAlias = tuple[DecimalInput, DecimalInput]


class BookError(ValueError):
    """Base error for invalid input or unavailable book state."""


class InvalidLevel(BookError):
    """A malformed side, price, quantity, or level collection."""


class InvalidBook(BookError):
    """An empty, locked, crossed, or explicitly invalidated book."""


class BookNotReady(BookError):
    """A snapshot is required before book reads or updates."""


class Side(StrEnum):
    BID = "bid"
    ASK = "ask"


class BookState(StrEnum):
    WAITING_FOR_SNAPSHOT = "waiting_for_snapshot"
    LIVE = "live"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class LevelUpdate:
    """Absolute displayed quantity; zero removes a level."""

    side: Side | str
    price: DecimalInput
    quantity: DecimalInput


def _decimal(value: DecimalInput, field: str, *, allow_zero: bool = False) -> Decimal:
    # Float conversion can change a price key.
    if isinstance(value, bool) or not isinstance(value, (Decimal, str, int)):
        raise InvalidLevel(f"{field} must be a Decimal, decimal string, or integer")
    try:
        parsed = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise InvalidLevel(f"{field} is not a decimal: {value!r}") from exc
    if not parsed.is_finite():
        raise InvalidLevel(f"{field} must be finite")
    if parsed < 0 or (parsed == 0 and not allow_zero):
        relation = "nonnegative" if allow_zero else "positive"
        raise InvalidLevel(f"{field} must be {relation}")
    return parsed


def _side(value: Side | str) -> Side:
    try:
        return Side(value)
    except (ValueError, TypeError) as exc:
        raise InvalidLevel(f"unknown book side: {value!r}; use 'bid' or 'ask'") from exc


def _depth(n: int) -> None:
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("depth must be a positive integer")


def _read_levels(levels: Iterable[RawLevel], name: str) -> dict[Decimal, Decimal]:
    result: dict[Decimal, Decimal] = {}
    try:
        for price, quantity in levels:
            px = _decimal(price, "price")
            qty = _decimal(quantity, "quantity")
            if px in result:
                raise InvalidLevel(f"duplicate {name} snapshot price: {px}")
            result[px] = qty
    except (TypeError, ValueError) as exc:
        if isinstance(exc, BookError):
            raise
        raise InvalidLevel(f"{name} snapshot requires (price, quantity) pairs") from exc
    return result


class Level2OrderBook:
    """L2 book with atomic updates and snapshot recovery.

    Invalid input blocks reads until a new snapshot arrives. The adapter is
    responsible for detecting sequence gaps and calling invalidate().
    """

    def __init__(self) -> None:
        self._bids: dict[Decimal, Decimal] = {}
        self._asks: dict[Decimal, Decimal] = {}
        self._state = BookState.WAITING_FOR_SNAPSHOT
        self._invalid_reason: str | None = None

    @property
    def state(self) -> BookState:
        return self._state

    @property
    def invalid_reason(self) -> str | None:
        return self._invalid_reason

    def reset(self) -> None:
        """Discard state after an explicit disconnect/session boundary."""
        self._bids.clear()
        self._asks.clear()
        self._state = BookState.WAITING_FOR_SNAPSHOT
        self._invalid_reason = None

    def invalidate(self, reason: str) -> None:
        """Prevent further use after a gap, malformed event, or source disconnect."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("invalidation requires a nonempty reason")
        self._state = BookState.INVALID
        self._invalid_reason = reason

    def _require_ready(self) -> None:
        if self._state == BookState.WAITING_FOR_SNAPSHOT:
            raise BookNotReady("load a complete snapshot before using the book")
        if self._state == BookState.INVALID:
            raise InvalidBook(f"book invalid: {self._invalid_reason}; load a fresh snapshot")

    @staticmethod
    def _validate_maps(bids: Mapping[Decimal, Decimal], asks: Mapping[Decimal, Decimal]) -> None:
        if not bids or not asks:
            raise InvalidBook("both sides must contain at least one positive-quantity level")
        if max(bids) >= min(asks):
            raise InvalidBook("best bid must be strictly less than best ask (locked/crossed book)")

    def validate(self) -> None:
        """Raise on unavailable state or any violated book invariant."""
        self._require_ready()
        try:
            for levels in (self._bids, self._asks):
                for price, quantity in levels.items():
                    _decimal(price, "price")
                    _decimal(quantity, "quantity")
            self._validate_maps(self._bids, self._asks)
        except BookError as exc:
            self.invalidate(str(exc))
            raise

    def load_snapshot(self, bids: Iterable[RawLevel], asks: Iterable[RawLevel]) -> None:
        """Replace both sides atomically, also recovering from INVALID.

        Duplicate prices, nonpositive quantities, empty sides and crossed/locked
        snapshots are rejected. Snapshot levels need not arrive in sorted order.
        """
        try:
            next_bids = _read_levels(bids, "bid")
            next_asks = _read_levels(asks, "ask")
            self._validate_maps(next_bids, next_asks)
        except Exception as exc:
            # A failed input iterator also invalidates the snapshot.
            self.invalidate(f"snapshot rejected: {exc}")
            raise
        self._bids, self._asks = next_bids, next_asks
        self._state = BookState.LIVE
        self._invalid_reason = None

    def apply_updates(self, updates: Iterable[LevelUpdate]) -> None:
        """Apply one message atomically with an O(changed levels) rollback journal.

        Repeated prices within a batch are applied in order; last quantity wins.
        Deleting an absent level is an idempotent no-op. No-op batches are valid.
        Any failed batch invalidates the book and restores its previous levels.
        """
        self._require_ready()
        # Consume the iterator before changing the book.
        try:
            normalized = []
            for update in updates:
                if not isinstance(update, LevelUpdate):
                    raise InvalidLevel("updates must contain LevelUpdate objects")
                normalized.append(
                    (
                        _side(update.side),
                        _decimal(update.price, "price"),
                        _decimal(update.quantity, "quantity", allow_zero=True),
                    )
                )
        except Exception as exc:
            self.invalidate(f"update rejected: {exc}")
            raise

        journal: dict[tuple[Side, Decimal], Decimal | None] = {}
        try:
            for side, price, quantity in normalized:
                levels = self._bids if side == Side.BID else self._asks
                journal.setdefault((side, price), levels.get(price))
                if quantity == 0:
                    levels.pop(price, None)
                else:
                    levels[price] = quantity
            self._validate_maps(self._bids, self._asks)
        except Exception as exc:
            for (side, price), old_quantity in journal.items():
                levels = self._bids if side == Side.BID else self._asks
                if old_quantity is None:
                    levels.pop(price, None)
                else:
                    levels[price] = old_quantity
            self.invalidate(f"update rejected: {exc}")
            raise

    def update_level(self, side: Side | str, price: DecimalInput, quantity: DecimalInput) -> None:
        """Convenience for a one-level message; use apply_updates for multi-level events."""
        self.apply_updates([LevelUpdate(side, price, quantity)])

    @property
    def bids(self) -> Mapping[Decimal, Decimal]:
        """Detached, read-only mapping. Prefer top_bids for routine inspection."""
        self._require_ready()
        return MappingProxyType(self._bids.copy())

    @property
    def asks(self) -> Mapping[Decimal, Decimal]:
        self._require_ready()
        return MappingProxyType(self._asks.copy())

    @property
    def best_bid(self) -> Decimal:
        self._require_ready()
        return max(self._bids)

    @property
    def best_ask(self) -> Decimal:
        self._require_ready()
        return min(self._asks)

    @property
    def midprice(self) -> Decimal:
        return quote_values(self.best_bid, self.best_ask)[0]

    @property
    def spread(self) -> Decimal:
        return quote_values(self.best_bid, self.best_ask)[1]

    def top_bids(self, n: int = 10) -> tuple[Level, ...]:
        self._require_ready()
        _depth(n)
        return tuple(Level(price, self._bids[price]) for price in nlargest(n, self._bids))

    def top_asks(self, n: int = 10) -> tuple[Level, ...]:
        self._require_ready()
        _depth(n)
        return tuple(Level(price, self._asks[price]) for price in nsmallest(n, self._asks))

    def snapshot(self, depth: int = 10) -> BookSnapshot:
        """Return a detached top-N view."""
        return BookSnapshot(bids=self.top_bids(depth), asks=self.top_asks(depth))
