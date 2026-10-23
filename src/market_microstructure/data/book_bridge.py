"""Apply source-independent book events without venue logic."""

from market_microstructure.book import Level2OrderBook, LevelUpdate

from .base import MarketEvent


def apply_book_event(book: Level2OrderBook, event: MarketEvent) -> None:
    if event.kind == "snapshot":
        book.load_snapshot(
            [(x.price, x.quantity) for x in event.changes if x.side == "bid"],
            [(x.price, x.quantity) for x in event.changes if x.side == "ask"],
        )
    elif event.kind == "book_update":
        book.apply_updates([LevelUpdate(x.side, x.price, x.quantity) for x in event.changes])
