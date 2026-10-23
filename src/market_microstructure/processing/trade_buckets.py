"""Constant-memory sufficient trade statistics on the same receive-time grid."""

from decimal import Decimal, localcontext

import pyarrow as pa

from .parquet import SnapshotSink, snapshot_schema
from .sampling import SnapshotSampler

TRADE_COLUMNS = (
    "trade_count",
    "trade_volume",
    "trade_buy_volume",
    "trade_sell_volume",
    "trade_notional",
)


def trade_schema(config):
    fields = [pa.field("trade_count", pa.int64(), nullable=False)]
    fields += [pa.field(name, pa.decimal256(76, 36), nullable=False) for name in TRADE_COLUMNS[1:]]
    schema = snapshot_schema(config)
    return pa.schema(
        list(schema) + fields,
        metadata={
            **schema.metadata,
            b"trade_buckets": b"(previous sample, sample]; first segment bucket discarded",
        },
    )


class TradeSnapshotSink(SnapshotSink):
    def __init__(self, path, config):
        super().__init__(path, config, schema=trade_schema(config))


class TradeSnapshotSampler(SnapshotSampler):
    def __init__(self, config, emit, *args, **kwargs):
        self.downstream = emit
        self.previous_segment = None
        self.duplicate_context = None
        self._reset_bucket()
        super().__init__(config, self._emit, *args, **kwargs)

    def _reset_bucket(self):
        self.bucket = {name: Decimal(0) for name in TRADE_COLUMNS}
        self.bucket["trade_count"] = 0

    def _emit(self, row):
        # Only a full interval within the emitted segment is a usable trade bucket.
        if row["segment_id"] != self.previous_segment:
            self._reset_bucket()
        self.downstream({**row, **self.bucket})
        self.previous_segment = row["segment_id"]
        self._reset_bucket()

    def begin_session(self, run_id, session_id):
        super().begin_session(run_id, session_id)
        self.previous_segment = self.duplicate_context = None
        self._reset_bucket()

    def invalidate(self, reason, context=None):
        super().invalidate(reason, context)
        self._reset_bucket()

    def notice(self, kind, context=None, **details):
        if kind == "duplicate_or_late_trade":
            self.duplicate_context = context
            # A late trade and a duplicate cannot be distinguished safely here.
            self.trade_uncertain = True
        super().notice(kind, context, **details)

    def apply(self, event):
        super().apply(event)
        if event.kind != "trade" or event.context == self.duplicate_context:
            return
        if event.aggressor_side not in ("buy", "sell"):
            self.trade_uncertain = True
            self.notice("unknown_aggressor_side", event.context)
            return
        with localcontext() as ctx:
            ctx.prec = 78
            quantity, price = Decimal(event.quantity), Decimal(event.price)
            self.bucket["trade_count"] += 1
            self.bucket["trade_volume"] += quantity
            self.bucket[f"trade_{event.aggressor_side}_volume"] += quantity
            self.bucket["trade_notional"] += quantity * price
