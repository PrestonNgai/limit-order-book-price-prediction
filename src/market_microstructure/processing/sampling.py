"""Causal receive-time sampling, independent of source schema and file boundaries."""

from collections import Counter
from datetime import UTC, datetime

from market_microstructure.book import BookError, BookState, Level2OrderBook
from market_microstructure.data.base import MarketEvent, ReceiveContext
from market_microstructure.data.book_bridge import apply_book_event

from .config import ProcessingConfig


class SnapshotSampler:
    """Single-pass state machine. Rows at t include all input messages received <= t.

    Equal-time rows are deferred until a later timestamp or explicit end-of-input.
    No clock extrapolation is performed beyond the final raw message in a session.
    Staleness suspends sampling but retains an otherwise uninterrupted book; a later
    update begins a NEW segment. Actual continuity faults discard the book.
    """

    def __init__(
        self,
        config: ProcessingConfig,
        emit,
        emit_segment,
        emit_quality,
        *,
        source: str,
        symbol: str,
        output_date: str | None = None,
    ):
        self.config = config
        self.emit, self.emit_segment, self.emit_quality = emit, emit_segment, emit_quality
        self.source, self.symbol, self.output_date = source, symbol, output_date
        self.interval_ns = config.interval_ms * 1_000_000
        self.max_age_ns = config.max_staleness_ms * 1_000_000
        self.book = Level2OrderBook()
        self.counts = Counter()
        self.run_id = self.session_id = None
        self.last_context: ReceiveContext | None = None
        self.book_context: ReceiveContext | None = None
        self.book_exchange_time_ns = None
        self.last_exchange_time_ns = None
        self.trade_baseline_ns = None
        self.trade_uncertain = False
        self.segment = None
        self.segment_number = 0
        self.next_grid_ns = None

    def begin_session(self, run_id, session_id):
        self.finish("session_boundary")
        self.run_id, self.session_id = run_id, session_id
        self.book.reset()
        self.last_context = self.book_context = None
        self.book_exchange_time_ns = self.last_exchange_time_ns = None
        self.trade_baseline_ns = None
        self.trade_uncertain = False
        self.segment_number = 0
        self.next_grid_ns = None

    def notice(self, kind, context=None, **details):
        self.counts[kind] += 1
        self.emit_quality(
            {
                "kind": kind,
                "run_id": self.run_id,
                "session_id": self.session_id,
                "receive_time_ns": context.receive_time_ns if context else None,
                "message_index": context.message_index if context else None,
                **details,
            }
        )

    def _close_segment(self, reason):
        if self.segment is not None:
            self.emit_segment({**self.segment, "end_reason": reason})
            self.counts["segments"] += 1
            self.segment = None
        self.next_grid_ns = None

    def invalidate(self, reason, context=None):
        # Never extend the last state to the later time when a fault is discovered.
        self._close_segment(reason)
        self.book.invalidate(reason)
        self.trade_uncertain = True
        self.trade_baseline_ns = None
        self.notice(reason, context)

    def before_message(self, context: ReceiveContext) -> bool:
        previous = self.last_context
        if previous and (
            context.receive_time_ns < previous.receive_time_ns
            or context.receive_monotonic_ns < previous.receive_monotonic_ns
        ):
            self.invalidate("clock_regression", context)
            self.last_context = context
            return False
        self.advance(context.receive_time_ns - 1)
        self.last_context = context
        return True

    def apply(self, event: MarketEvent):
        if event.kind == "snapshot":
            self._close_segment("replacement_snapshot")
        if event.kind == "book_update" and self.book.state != BookState.LIVE:
            self.notice("updates_without_snapshot", event.context)
            return
        if event.kind not in ("snapshot", "book_update"):
            return
        try:
            apply_book_event(self.book, event)
        except BookError as exc:
            self.invalidate("invalid_book", event.context)
            self.notice("book_error_detail", event.context, error=str(exc))
            return
        if (
            event.exchange_time_ns is not None
            and self.last_exchange_time_ns is not None
            and event.exchange_time_ns < self.last_exchange_time_ns
        ):
            self.notice("exchange_time_regression", event.context)
        # Preserve arrival order. Never sort/backdate an event by exchange time.
        if event.exchange_time_ns is not None:
            self.last_exchange_time_ns = event.exchange_time_ns
        self.book_context = event.context
        self.book_exchange_time_ns = event.exchange_time_ns
        if self.segment is None:
            self.segment_number += 1
            self.segment = {
                "segment_id": f"{self.run_id}/{self.session_id}/{self.segment_number:06d}",
                "run_id": self.run_id,
                "session_id": self.session_id,
                "start_receive_time_ns": event.context.receive_time_ns,
                "start_message_index": event.context.message_index,
                "start_reason": event.kind,
                "first_sample_time_ns": None,
                "last_sample_time_ns": None,
                "rows": 0,
            }
            self.next_grid_ns = (
                (event.context.receive_time_ns + self.interval_ns - 1)
                // self.interval_ns
                * self.interval_ns
            )

    def advance(self, inclusive_ns):
        if self.segment is None:
            return
        limit = min(inclusive_ns, self.book_context.receive_time_ns + self.max_age_ns)
        # A long gap costs at most max_staleness/interval rows, not gap/interval.
        while self.next_grid_ns <= limit:
            timestamp = self.next_grid_ns
            self.next_grid_ns += self.interval_ns
            if self.output_date is not None:
                day = datetime.fromtimestamp(timestamp // 1_000_000_000, UTC).date().isoformat()
                if day != self.output_date:
                    self.counts["rows_outside_requested_date"] += 1
                    continue
            view = self.book.snapshot(self.config.depth)
            row = {
                "timestamp_ns": timestamp,
                "timestamp": timestamp,
                "source": self.source,
                "symbol": self.symbol,
                "run_id": self.run_id,
                "session_id": self.session_id,
                "segment_id": self.segment["segment_id"],
                "last_message_index": self.last_context.message_index,
                "last_book_message_index": self.book_context.message_index,
                "last_book_receive_time_ns": self.book_context.receive_time_ns,
                "last_book_receive_monotonic_ns": self.book_context.receive_monotonic_ns,
                "last_book_exchange_time_ns": self.book_exchange_time_ns,
                "book_age_ns": timestamp - self.book_context.receive_time_ns,
                "trade_baseline_receive_time_ns": self.trade_baseline_ns,
                "trade_coverage_uncertain": self.trade_uncertain,
                "bid_levels_available": len(view.bids),
                "ask_levels_available": len(view.asks),
                "best_bid": view.best_bid,
                "best_ask": view.best_ask,
                "midprice": view.midprice,
                "spread": view.spread,
            }
            for side, levels in (("bid", view.bids), ("ask", view.asks)):
                for index in range(self.config.depth):
                    level = levels[index] if index < len(levels) else None
                    row[f"{side}_px_{index + 1}"] = level.price if level else None
                    row[f"{side}_sz_{index + 1}"] = level.quantity if level else None
            self.emit(row)
            if self.segment["first_sample_time_ns"] is None:
                self.segment["first_sample_time_ns"] = timestamp
            self.segment["last_sample_time_ns"] = timestamp
            self.segment["rows"] += 1
            self.counts["rows"] += 1
        if inclusive_ns > self.book_context.receive_time_ns + self.max_age_ns:
            self.notice("staleness_break", self.last_context)
            self._close_segment("staleness")

    def finish(self, reason="end_of_input"):
        if self.last_context is not None:
            self.advance(self.last_context.receive_time_ns)
        self._close_segment(reason)
