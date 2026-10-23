"""Explicit exact-decimal schema and bounded Parquet row groups."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .config import ProcessingConfig


def snapshot_schema(config: ProcessingConfig) -> pa.Schema:
    number = pa.decimal128(config.decimal_precision, config.decimal_scale)
    fields = [
        pa.field("timestamp_ns", pa.int64(), nullable=False),
        pa.field("timestamp", pa.timestamp("ns", tz="UTC"), nullable=False),
    ]
    for name in ("source", "symbol", "run_id", "session_id", "segment_id"):
        fields.append(pa.field(name, pa.string(), nullable=False))
    for name in (
        "last_message_index",
        "last_book_message_index",
        "last_book_receive_time_ns",
        "last_book_receive_monotonic_ns",
        "book_age_ns",
    ):
        fields.append(pa.field(name, pa.int64(), nullable=False))
    for name in ("last_book_exchange_time_ns", "trade_baseline_receive_time_ns"):
        fields.append(pa.field(name, pa.int64()))
    fields.append(pa.field("trade_coverage_uncertain", pa.bool_(), nullable=False))
    for name in ("bid_levels_available", "ask_levels_available"):
        fields.append(pa.field(name, pa.int32(), nullable=False))
    for name in ("best_bid", "best_ask", "midprice", "spread"):
        fields.append(pa.field(name, number, nullable=False))
    for side in ("bid", "ask"):
        for level in range(1, config.depth + 1):
            fields.extend(
                [
                    pa.field(f"{side}_px_{level}", number),
                    pa.field(f"{side}_sz_{level}", number),
                ]
            )
    return pa.schema(
        fields,
        metadata={
            b"schema_version": b"1",
            b"clock": b"local_receive_time; equal timestamps resolved in journal order",
            b"sampling_config": json.dumps(config.to_dict(), sort_keys=True).encode(),
        },
    )


class SnapshotSink:
    def __init__(self, path: Path, config: ProcessingConfig, *, schema=None):
        self.config = config
        self.schema = snapshot_schema(config) if schema is None else schema
        self.writer = pq.ParquetWriter(
            path,
            self.schema,
            version="2.6",
            compression=None if config.compression == "none" else config.compression,
            use_dictionary=["source", "symbol", "run_id", "session_id", "segment_id"],
        )
        self.buffer = []
        self.rows = self.max_buffered_rows = 0
        self.minimum_time_ns = self.maximum_time_ns = None
        self.closed = False

    def add(self, row):
        self.buffer.append(row)
        self.max_buffered_rows = max(self.max_buffered_rows, len(self.buffer))
        timestamp = row["timestamp_ns"]
        self.minimum_time_ns = (
            timestamp if self.minimum_time_ns is None else min(self.minimum_time_ns, timestamp)
        )
        self.maximum_time_ns = (
            timestamp if self.maximum_time_ns is None else max(self.maximum_time_ns, timestamp)
        )
        if len(self.buffer) >= self.config.batch_rows:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        table = pa.Table.from_pylist(self.buffer, schema=self.schema)
        self.writer.write_table(table, row_group_size=self.config.batch_rows)
        self.rows += len(self.buffer)
        self.buffer.clear()

    def close(self):
        if not self.closed:
            self.flush()
            self.writer.close()
            self.closed = True

    def abort(self):
        if not self.closed:
            self.writer.close()
            self.closed = True
