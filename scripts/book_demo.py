"""Deterministic offline replay. Run from any directory; no network or market data."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from market_microstructure.book import Level2OrderBook, LevelUpdate  # noqa: E402


def main() -> None:
    book = Level2OrderBook()
    book.load_snapshot(
        bids=[("100.00", "2.5"), ("99.99", "4")],
        asks=[("100.02", "3"), ("100.03", "5")],
    )
    book.apply_updates(
        [
            LevelUpdate("bid", "100.00", "0"),
            LevelUpdate("bid", "100.01", "1.25"),
            LevelUpdate("ask", "100.02", "2"),
        ]
    )
    book.validate()
    view = book.snapshot(depth=2)
    print(
        json.dumps(
            {
                "data_kind": "synthetic demonstration; not market observations",
                "state": book.state.value,
                "bids": [[str(level.price), str(level.quantity)] for level in view.bids],
                "asks": [[str(level.price), str(level.quantity)] for level in view.asks],
                "best_bid": str(book.best_bid),
                "best_ask": str(book.best_ask),
                "midprice": str(book.midprice),
                "spread": str(book.spread),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
