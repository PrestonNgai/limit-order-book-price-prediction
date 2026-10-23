"""Replay original journal payloads; normalized annotations are never applied twice."""

from collections import Counter

from market_microstructure.data.base import FeedError, ReceiveContext, SourceAdapter

from .sampling import SnapshotSampler


class ReplayError(ValueError):
    """Input journal ordering/schema is unsafe for reproducible replay."""


def nonnegative_int(value, field):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**63:
        raise ReplayError(f"{field} must be a nonnegative signed-64-bit integer")
    return value


class JournalReplay:
    def __init__(self, adapter: SourceAdapter, sampler: SnapshotSampler):
        self.adapter, self.sampler = adapter, sampler
        self.counts = Counter()
        self.run_id = self.session_id = None
        self.seen_runs = set()
        self.seen_sessions = set()
        self.expected_index = None
        self.tracker = adapter.continuity_tracker()
        self.closed_session = False

    def _session(self, run_id, session_id):
        if run_id != self.run_id:
            if run_id in self.seen_runs:
                raise ReplayError("a completed run reappeared; group input files by run")
            self.seen_runs.add(run_id)
            self.expected_index = None
        key = run_id, session_id
        if key in self.seen_sessions:
            raise ReplayError("session reappeared after its boundary")
        self.seen_sessions.add(key)
        self.run_id, self.session_id = key
        self.closed_session = False
        self.tracker = self.adapter.continuity_tracker()
        self.sampler.begin_session(*key)

    def consume(self, record):
        self.counts["journal_records"] += 1
        if type(record.get("schema_version")) is not int or record["schema_version"] != 1:
            raise ReplayError("unsupported raw journal schema_version")
        if (
            record.get("source") != self.adapter.source
            or record.get("symbol") != self.adapter.symbol
        ):
            raise ReplayError("input source/product does not match the selected adapter")
        kind = record.get("record_type")
        if not isinstance(kind, str):
            raise ReplayError("record_type must be a string")
        run_id, session_id = record.get("run_id"), record.get("session_id")
        if not isinstance(run_id, str) or not run_id:
            raise ReplayError("missing run_id")
        if kind == "session_start":
            if not isinstance(session_id, str) or not session_id:
                raise ReplayError("session_start requires session_id")
            self._session(run_id, session_id)
            self.counts["sessions"] += 1
            return
        if kind in (
            "connection_fault",
            "book_error",
            "malformed_message",
            "subscription_rejected",
            "clock_regression",
        ):
            if (run_id, session_id) == (self.run_id, self.session_id):
                self.sampler.invalidate(f"recorded_{kind}")
            return
        if kind in ("session_end", "recorder_stop"):
            if self.run_id == run_id:
                self.sampler.finish(kind)
                self.sampler.book.invalidate(kind)
                self.closed_session = True
            return
        if kind != "raw_message":
            self.counts["annotation_records"] += 1
            return
        if not isinstance(session_id, str) or not session_id:
            raise ReplayError("raw message requires session_id")
        if (run_id, session_id) != (self.run_id, self.session_id):
            self._session(run_id, session_id)
            self.counts["sessions_without_start_marker"] += 1
        elif self.closed_session:
            raise ReplayError("raw message after closed session")
        index = nonnegative_int(record.get("message_index"), "message_index")
        context = ReceiveContext(
            session_id,
            index,
            nonnegative_int(record.get("receive_time_ns"), "receive_time_ns"),
            nonnegative_int(record.get("receive_monotonic_ns"), "receive_monotonic_ns"),
        )
        if self.expected_index is None:
            if index != 0:
                self.sampler.notice("missing_input_prefix", context, messages=index)
        elif index < self.expected_index:
            raise ReplayError("duplicate or reordered raw message index")
        elif index > self.expected_index:
            self.sampler.invalidate("missing_raw_messages", context)
            self.sampler.notice("missing_raw_count", context, messages=index - self.expected_index)
        self.expected_index = index + 1
        self.counts["raw_messages"] += 1
        if not self.sampler.before_message(context):
            # A collector also rejects this frame before parsing; require a later snapshot.
            return
        if record.get("encoding") != "utf-8" or not isinstance(record.get("payload"), str):
            self.sampler.invalidate("invalid_payload_encoding", context)
            return
        try:
            event = self.adapter.parse(record["payload"], context)
        except FeedError as exc:
            self.sampler.invalidate("malformed_payload", context)
            self.sampler.notice("parse_error_detail", context, error=str(exc))
            return
        self.counts[event.kind] += 1
        if notice := self.tracker.observe(event):
            self.sampler.notice(notice["kind"], context, details=notice)
            if notice["kind"] == "trade_baseline":
                self.sampler.trade_baseline_ns = context.receive_time_ns
            if notice["kind"] in ("trade_gap", "possible_trade_gap"):
                self.sampler.trade_uncertain = True
        self.sampler.apply(event)

    def finish(self):
        self.sampler.finish()
