"""Coinbase Exchange public level2_batch, matches and heartbeat adapter.

Protocol reference checked 2026-09-24:
https://docs.cdp.coinbase.com/exchange/websocket-feed/channels
https://docs.cdp.coinbase.com/exchange/websocket-feed/overview
"""

import calendar
import json
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation

from .base import FeedError, FeedRejected, LevelChange, MarketEvent, ReceiveContext

CHANNELS = ("level2_batch", "matches", "heartbeat")
ENDPOINT = "wss://ws-feed.exchange.coinbase.com"


def timestamp_ns(value: str) -> int:
    """UTC RFC3339 seconds with up to nanosecond precision; never use float epochs."""
    if not isinstance(value, str):
        raise FeedError("exchange time must be a UTC timestamp string")
    match = re.fullmatch(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d{1,9}))?Z", value)
    if not match:
        raise FeedError(f"unsupported exchange timestamp: {value!r}")
    try:
        seconds = calendar.timegm(datetime.strptime(match[1], "%Y-%m-%dT%H:%M:%S").timetuple())
    except ValueError as exc:
        raise FeedError(f"invalid exchange time: {value!r}") from exc
    return seconds * 1_000_000_000 + int((match[2] or "").ljust(9, "0"))


def positive_decimal(value, field: str, *, zero: bool = False) -> str:
    if not isinstance(value, str):
        raise FeedError(f"{field} must be a decimal string")
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise FeedError(f"invalid {field}") from exc
    if not number.is_finite() or number < 0 or (number == 0 and not zero):
        raise FeedError(f"{field} must be finite and {'nonnegative' if zero else 'positive'}")
    return value


def integer(value, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FeedError(f"{field} must be a nonnegative integer")
    return value


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise FeedError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def invalid_constant(value):
    raise FeedError(f"nonstandard JSON constant: {value}")


class CoinbaseAdapter:
    """Stateless parsing; live collection and offline replay use the same function."""

    source = "coinbase"
    endpoint = ENDPOINT

    def __init__(self, symbol: str = "BTC-USD"):
        if not isinstance(symbol, str) or not re.fullmatch(r"[A-Z0-9]+-[A-Z0-9]+", symbol):
            raise ValueError("symbol must be an uppercase product ID, e.g. BTC-USD")
        self.symbol = symbol

    @property
    def metadata_url(self) -> str:
        return f"https://api.exchange.coinbase.com/products/{self.symbol}"

    def parse_metadata(self, payload: str) -> dict:
        try:
            value = json.loads(payload, object_pairs_hook=strict_object)
            if value["id"] != self.symbol:
                raise FeedError("metadata product mismatch")
            return {
                "symbol": self.symbol,
                "quote_increment": positive_decimal(value["quote_increment"], "quote_increment"),
                "base_increment": positive_decimal(value["base_increment"], "base_increment"),
                "status": value.get("status"),
            }
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise FeedError("invalid product metadata") from exc

    def continuity_tracker(self):
        return TradeContinuity()

    def subscription(self) -> dict:
        return {"type": "subscribe", "product_ids": [self.symbol], "channels": list(CHANNELS)}

    def parse(self, payload: str, context: ReceiveContext) -> MarketEvent:
        try:
            message = json.loads(
                payload, object_pairs_hook=strict_object, parse_constant=invalid_constant
            )
        except (json.JSONDecodeError, UnicodeError, TypeError) as exc:
            raise FeedError("invalid JSON message") from exc
        if not isinstance(message, dict) or not isinstance(message.get("type"), str):
            raise FeedError("message must be an object with a string type")
        kind = message["type"]
        if kind == "error":
            raise FeedRejected(str(message.get("message", "exchange rejected request")))
        if kind == "subscriptions":
            channels = message.get("channels")
            if not isinstance(channels, list):
                raise FeedError("subscriptions.channels must be a list")
            actual = set()
            for channel in channels:
                if not isinstance(channel, dict):
                    raise FeedError("invalid subscription channel")
                products = channel.get("product_ids")
                if not isinstance(products, list) or not all(isinstance(p, str) for p in products):
                    raise FeedError("invalid subscription product list")
                if not isinstance(channel.get("name"), str):
                    raise FeedError("invalid subscription channel name")
                if self.symbol in products:
                    actual.add(channel["name"])
            # Public batch acknowledgements may use its documented historical alias.
            if "level2_50" in actual:
                actual.add("level2_batch")
            if not set(CHANNELS).issubset(actual):
                raise FeedRejected(f"incomplete subscription acknowledgement: {sorted(actual)}")
            return MarketEvent("subscriptions", self.source, self.symbol, context)

        supported = {"snapshot", "l2update", "match", "last_match", "heartbeat"}
        if kind not in supported:
            return MarketEvent("ignored", self.source, self.symbol, context)
        if message.get("product_id") != self.symbol:
            raise FeedError(f"unexpected product: {message.get('product_id')!r}")
        exchange_time = message.get("time")
        if kind != "snapshot" and exchange_time is None:
            raise FeedError(f"{kind} requires exchange time")
        exchange_ns = timestamp_ns(exchange_time) if exchange_time is not None else None
        sequence = integer(message["sequence"], "sequence") if "sequence" in message else None
        common = {
            "source": self.source,
            "symbol": self.symbol,
            "context": context,
            "exchange_time": exchange_time,
            "exchange_time_ns": exchange_ns,
            "sequence": sequence,
        }
        try:
            if kind == "snapshot":
                levels = []
                for side, key in (("bid", "bids"), ("ask", "asks")):
                    rows = message[key]
                    if not isinstance(rows, list) or not rows:
                        raise FeedError("snapshot requires nonempty bids and asks")
                    seen = set()
                    for row in rows:
                        if not isinstance(row, list) or len(row) != 2:
                            raise FeedError("snapshot level must have price and size")
                        price = positive_decimal(row[0], "price")
                        if Decimal(price) in seen:
                            raise FeedError("duplicate snapshot price")
                        seen.add(Decimal(price))
                        levels.append(LevelChange(side, price, positive_decimal(row[1], "size")))
                return MarketEvent("snapshot", changes=tuple(levels), **common)
            if kind == "l2update":
                rows = message["changes"]
                if not isinstance(rows, list):
                    raise FeedError("changes must be a list")
                changes = []
                for row in rows:
                    if not isinstance(row, list) or len(row) != 3 or row[0] not in ("buy", "sell"):
                        raise FeedError("change must have buy/sell side, price and size")
                    changes.append(
                        LevelChange(
                            "bid" if row[0] == "buy" else "ask",
                            positive_decimal(row[1], "price"),
                            positive_decimal(row[2], "size", zero=True),
                        )
                    )
                return MarketEvent("book_update", changes=tuple(changes), **common)
            if kind in ("match", "last_match"):
                if message["side"] not in ("buy", "sell"):
                    raise FeedError("invalid maker side")
                return MarketEvent(
                    "trade" if kind == "match" else "trade_reference",
                    price=positive_decimal(message["price"], "price"),
                    quantity=positive_decimal(message["size"], "size"),
                    maker_side=message["side"],
                    aggressor_side="sell" if message["side"] == "buy" else "buy",
                    trade_id=integer(message["trade_id"], "trade_id"),
                    **common,
                )
            return MarketEvent(
                "heartbeat",
                last_trade_id=integer(message["last_trade_id"], "last_trade_id"),
                **common,
            )
        except (KeyError, TypeError) as exc:
            raise FeedError(f"malformed {kind} message") from exc


class TradeContinuity:
    """Conservative gap detection; no claim of recovery or completeness before baseline.

    Global exchange sequence values in matches are NOT contiguous trade sequences.
    A heartbeat ahead of received trades is flagged as uncertain, never filled in.
    """

    def __init__(self):
        self.last_trade_id: int | None = None

    def observe(self, event: MarketEvent) -> dict | None:
        if event.kind == "heartbeat":
            if self.last_trade_id is None:
                self.last_trade_id = event.last_trade_id
                return {"kind": "trade_baseline", "last_trade_id": self.last_trade_id}
            if event.last_trade_id > self.last_trade_id:
                previous = self.last_trade_id
                self.last_trade_id = event.last_trade_id
                return {
                    "kind": "possible_trade_gap",
                    "after": previous,
                    "through": self.last_trade_id,
                }
            return None
        if event.kind not in ("trade", "trade_reference"):
            return None
        current, previous = event.trade_id, self.last_trade_id
        if previous is None:
            self.last_trade_id = current
            return {"kind": "trade_baseline", "last_trade_id": current}
        if current <= previous:
            return {"kind": "duplicate_or_late_trade", "trade_id": current, "last_seen": previous}
        self.last_trade_id = current
        if current > previous + 1:
            return {"kind": "trade_gap", "after": previous, "before": current}
        return None
