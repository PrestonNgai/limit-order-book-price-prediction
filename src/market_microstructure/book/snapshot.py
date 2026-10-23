"""Immutable in-memory snapshots; clock sampling and Parquet belong to processing."""

from dataclasses import dataclass
from decimal import Decimal, localcontext


def quote_values(bid: Decimal, ask: Decimal) -> tuple[Decimal, Decimal]:
    """Compute mid/spread exactly, including a half-tick under low context precision."""
    digits = (
        max(bid.adjusted(), ask.adjusted())
        - min(bid.as_tuple().exponent, ask.as_tuple().exponent)
        + 3
    )
    with localcontext() as context:
        context.prec = max(28, digits)
        return (bid + ask) / 2, ask - bid


@dataclass(frozen=True, slots=True)
class Level:
    """One aggregated price level, using exact decimal values."""

    price: Decimal
    quantity: Decimal


@dataclass(frozen=True, slots=True)
class BookSnapshot:
    """Detached top-N levels: bids descending, asks ascending.

    The sampler supplies timestamps and continuity metadata.
    """

    bids: tuple[Level, ...]
    asks: tuple[Level, ...]

    @property
    def best_bid(self) -> Decimal:
        return self.bids[0].price

    @property
    def best_ask(self) -> Decimal:
        return self.asks[0].price

    @property
    def midprice(self) -> Decimal:
        return quote_values(self.best_bid, self.best_ask)[0]

    @property
    def spread(self) -> Decimal:
        return quote_values(self.best_bid, self.best_ask)[1]
