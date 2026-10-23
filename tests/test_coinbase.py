import json
from dataclasses import replace
from decimal import Decimal

import pytest

from market_microstructure.book import Level2OrderBook
from market_microstructure.data.base import FeedError, FeedRejected, ReceiveContext
from market_microstructure.data.book_bridge import apply_book_event
from market_microstructure.data.coinbase import CoinbaseAdapter, TradeContinuity, timestamp_ns

CONTEXT = ReceiveContext("test", 0, 1_000_000_000, 500)


def parse(message):
    return CoinbaseAdapter().parse(json.dumps(message), CONTEXT)


def test_subscription_is_public():
    assert CoinbaseAdapter().subscription() == {
        "type": "subscribe",
        "product_ids": ["BTC-USD"],
        "channels": ["level2_batch", "matches", "heartbeat"],
    }


def test_fixtures_reconstruct_book_without_source_logic_in_book(messages):
    book = Level2OrderBook()
    snapshot = parse(messages["snapshot"])
    assert snapshot.exchange_time is None
    assert snapshot.sequence is None  # Do not invent L2 sequence or snapshot exchange time.
    apply_book_event(book, snapshot)
    update = parse(messages["update"])
    apply_book_event(book, update)
    assert update.exchange_time_ns % 1_000_000_000 == 123456789
    assert book.midprice == Decimal("100.015")
    assert book.asks[Decimal("100.02")] == Decimal("2")
    assert update.context is CONTEXT


@pytest.mark.parametrize(("maker", "aggressor"), [("sell", "buy"), ("buy", "sell")])
def test_trade_side_is_maker_not_aggressor(messages, maker, aggressor):
    event = parse({**messages["match"], "side": maker})
    assert event.kind == "trade"
    assert event.maker_side == maker
    assert event.aggressor_side == aggressor
    assert event.sequence == 80
    assert event.trade_id == 11


def test_last_match_is_reference_not_new_live_trade(messages):
    assert parse(messages["reference"]).kind == "trade_reference"


def test_controls_and_unknown_types(messages):
    assert parse(messages["subscriptions"]).kind == "subscriptions"
    assert parse({"type": "future_extension", "opaque": True}).kind == "ignored"
    with pytest.raises(FeedRejected):
        parse(messages["error"])
    with pytest.raises(FeedRejected):
        parse({"type": "subscriptions", "channels": []})


@pytest.mark.parametrize(
    "payload",
    [
        "{bad",
        "[]",
        "null",
        '{"type":3}',
        '{"type":"snapshot","type":"match"}',
        '{"type":"x","value":NaN}',
    ],
)
def test_malformed_json_fails(payload):
    with pytest.raises(FeedError):
        CoinbaseAdapter().parse(payload, CONTEXT)


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-1", "0", 1.0, None])
def test_malformed_trade_numeric_fields(messages, value):
    with pytest.raises(FeedError):
        parse({**messages["match"], "price": value})


@pytest.mark.parametrize("value", [-1, 1.5, True, "12", None])
def test_bad_trade_id(messages, value):
    with pytest.raises(FeedError):
        parse({**messages["match"], "trade_id": value})


@pytest.mark.parametrize(
    "value", ["bad", "2026-02-30T00:00:00Z", "2026-09-24T00:00:00.1234567890Z", None]
)
def test_bad_timestamps(value):
    with pytest.raises(FeedError):
        timestamp_ns(value)


def test_timestamp_ns_has_no_float_rounding():
    assert timestamp_ns("1970-01-01T00:00:01.000000001Z") == 1_000_000_001


def test_wrong_product_and_invalid_side_rejected(messages):
    with pytest.raises(FeedError):
        parse({**messages["match"], "product_id": "ETH-USD"})
    with pytest.raises(FeedError):
        parse({**messages["update"], "changes": [["bid", "100", "2"]]})


def test_snapshot_duplicates_and_zero_rejected(messages):
    for bids in ([["100", "1"], ["100.0", "2"]], [["100", "0"]], []):
        with pytest.raises(FeedError):
            parse({**messages["snapshot"], "bids": bids})


def test_update_zero_is_delete(messages):
    event = parse({**messages["update"], "changes": [["buy", "100", "0"]]})
    assert event.changes[0].quantity == "0"


def test_global_sequences_not_treated_as_contiguous_trade_ids(messages):
    tracker = TradeContinuity()
    assert tracker.observe(parse(messages["reference"]))["kind"] == "trade_baseline"
    assert (
        tracker.observe(parse(messages["match"])) is None
    )  # Exchange sequence numbers are not contiguous trade IDs.
    assert tracker.observe(parse(messages["heartbeat"])) is None


def test_gaps_duplicates_and_heartbeat_uncertainty(messages):
    tracker = TradeContinuity()
    event = parse(messages["match"])
    tracker.observe(event)
    assert tracker.observe(event)["kind"] == "duplicate_or_late_trade"
    assert tracker.observe(replace(event, trade_id=14)) == {
        "kind": "trade_gap",
        "after": 11,
        "before": 14,
    }
    assert tracker.observe(replace(parse(messages["heartbeat"]), last_trade_id=16)) == {
        "kind": "possible_trade_gap",
        "after": 14,
        "through": 16,
    }


def test_metadata_retains_verified_increments():
    adapter = CoinbaseAdapter()
    result = adapter.parse_metadata(
        json.dumps(
            {
                "id": "BTC-USD",
                "quote_increment": "0.01",
                "base_increment": "0.00000001",
                "status": "online",
            }
        )
    )
    assert result["quote_increment"] == "0.01"
    with pytest.raises(FeedError):
        adapter.parse_metadata('{"id":"ETH-USD"}')
