"""Deterministic replay against a separate reference map implementation."""

import random
from decimal import Decimal

from market_microstructure.book import Level2OrderBook, LevelUpdate

D = Decimal


def test_seeded_replay_matches_reference_for_1000_messages():
    rng = random.Random(421)
    reference = {"bid": {D("100"): D("2")}, "ask": {D("101"): D("3")}}
    book = Level2OrderBook()
    book.load_snapshot(reference["bid"].items(), reference["ask"].items())

    for _ in range(1000):
        message = []
        for _ in range(rng.randint(1, 5)):
            side = rng.choice(["bid", "ask"])
            # Retain fixed inside levels; fuzz insert/update/delete across deeper levels.
            price = D(rng.randint(90, 99) if side == "bid" else rng.randint(102, 111))
            quantity = D(rng.randint(0, 20)) / 4
            message.append(LevelUpdate(side, price, quantity))
            if quantity:
                reference[side][price] = quantity
            else:
                reference[side].pop(price, None)
        book.apply_updates(message)
        book.validate()
        assert dict(book.bids) == reference["bid"]
        assert dict(book.asks) == reference["ask"]
        assert [level.price for level in book.top_bids(10)] == sorted(
            reference["bid"], reverse=True
        )[:10]
        assert [level.price for level in book.top_asks(10)] == sorted(reference["ask"])[:10]
