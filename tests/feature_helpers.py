from decimal import Decimal

import pyarrow as pa

from market_microstructure.processing.config import ProcessingConfig
from market_microstructure.processing.trade_buckets import TRADE_COLUMNS, trade_schema

MS = 1_000_000


def observation_rows(n=70, *, mids=None, segment="r/s/1", offset_ms=0):
    rows = []
    for i in range(n):
        mid = Decimal(str(mids[i] if mids is not None else 100))
        row = {
            "timestamp_ns": (offset_ms + i * 100) * MS,
            "timestamp": (offset_ms + i * 100) * MS,
            "source": "coinbase",
            "symbol": "BTC-USD",
            "run_id": "r",
            "session_id": "s",
            "segment_id": segment,
            "last_message_index": i,
            "last_book_message_index": i,
            "last_book_receive_time_ns": (offset_ms + i * 100) * MS,
            "last_book_receive_monotonic_ns": (offset_ms + i * 100) * MS,
            "last_book_exchange_time_ns": None,
            "book_age_ns": 0,
            "trade_baseline_receive_time_ns": offset_ms * MS,
            "trade_coverage_uncertain": False,
            "bid_levels_available": 10,
            "ask_levels_available": 10,
            "best_bid": mid - 1,
            "best_ask": mid + 1,
            "midprice": mid,
            "spread": Decimal(2),
            "trade_count": 1 if i else 0,
            "trade_volume": Decimal(3) if i else Decimal(0),
            "trade_buy_volume": Decimal(2) if i else Decimal(0),
            "trade_sell_volume": Decimal(1) if i else Decimal(0),
            "trade_notional": Decimal(300) if i else Decimal(0),
        }
        for level in range(1, 11):
            row[f"bid_px_{level}"] = mid - level
            row[f"ask_px_{level}"] = mid + level
            row[f"bid_sz_{level}"] = Decimal(2)
            row[f"ask_sz_{level}"] = Decimal(1)
        rows.append(row)
    return rows


def observations(rows=None, **kwargs):
    return pa.Table.from_pylist(
        rows if rows is not None else observation_rows(**kwargs),
        schema=trade_schema(ProcessingConfig()),
    )


def no_trades(row):
    for name in TRADE_COLUMNS:
        row[name] = 0 if name == "trade_count" else Decimal(0)
