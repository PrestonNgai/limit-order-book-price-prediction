"""Record raw market data with bounded buffering and reconnects."""

import asyncio
import base64
import json
import logging
import math
import random
import time
from collections import Counter
from urllib.request import Request, urlopen
from uuid import uuid4

from websockets.asyncio.client import connect
from websockets.exceptions import WebSocketException

from market_microstructure import __version__
from market_microstructure.book import BookError, BookState, Level2OrderBook
from market_microstructure.utils.config import CollectorConfig

from .base import FeedError, FeedRejected, ReceiveContext, SourceAdapter
from .book_bridge import apply_book_event
from .storage import JsonlWriter

LOG = logging.getLogger(__name__)


def backoff_seconds(attempt: int, initial: float, maximum: float, jitter: float) -> float:
    """Bounded exponential backoff; jitter argument supports deterministic tests."""
    return min(maximum, initial * 2 ** min(attempt, 30)) * (0.5 + 0.5 * jitter)


def fetch_metadata(url: str, timeout: float) -> str:
    request = Request(url, headers={"User-Agent": f"market-microstructure-research/{__version__}"})
    with urlopen(request, timeout=timeout) as response:
        data = response.read(1024 * 1024 + 1)
        if len(data) > 1024 * 1024:
            raise FeedError("product metadata exceeds 1 MiB")
        return data.decode("utf-8")


class Recorder:
    def __init__(
        self,
        adapter: SourceAdapter,
        config: CollectorConfig,
        *,
        writer: JsonlWriter | None = None,
        connector=connect,
        metadata_fetcher=fetch_metadata,
    ):
        if adapter.symbol != config.symbol:
            raise ValueError("adapter and config symbols differ")
        self.adapter, self.config = adapter, config
        self.writer = writer or JsonlWriter(
            config.output_dir,
            config.symbol,
            source=adapter.source,
            rotate_seconds=config.rotate_seconds,
            max_bytes=config.max_file_bytes,
            fsync_seconds=config.fsync_seconds,
            compress=config.compress,
        )
        self.connector, self.metadata_fetcher = connector, metadata_fetcher
        self.run_id = self.writer.run_id
        self.stop = asyncio.Event()
        self.stats = Counter()
        self.session_id = None
        self.book = Level2OrderBook()
        self._previous_receive_time = None
        self._last_report = time.monotonic()

    def _record(self, record_type, *, context: ReceiveContext | None = None, **payload):
        self.writer.write(
            {
                "schema_version": 1,
                "record_type": record_type,
                "source": self.adapter.source,
                "symbol": self.adapter.symbol,
                "run_id": self.run_id,
                "session_id": self.session_id,
                "receive_time_ns": context.receive_time_ns if context else time.time_ns(),
                "receive_monotonic_ns": (
                    context.receive_monotonic_ns if context else time.monotonic_ns()
                ),
                **payload,
            }
        )

    def _report(self, force=False):
        now = time.monotonic()
        if force or now - self._last_report >= self.config.report_seconds:
            self._last_report = now
            LOG.info("counts %s", json.dumps(dict(self.stats), sort_keys=True))

    async def _metadata(self):
        if not self.config.fetch_product_metadata:
            self._record("metadata_skipped")
            return
        try:
            payload = await asyncio.to_thread(
                self.metadata_fetcher, self.adapter.metadata_url, self.config.open_timeout
            )
        except (OSError, FeedError, UnicodeError) as exc:
            self._record("metadata_unavailable", error=str(exc))
            LOG.warning("Product metadata unavailable: %s", exc)
            return
        self._record("product_metadata_raw", url=self.adapter.metadata_url, payload=payload)
        try:
            values = self.adapter.parse_metadata(payload)
        except FeedError as exc:
            self._record("metadata_invalid", error=str(exc))
            LOG.warning("Product metadata invalid: %s", exc)
            return
        self._record("product_metadata", values=values)

    async def run(self, *, duration: float | None = None, max_messages: int | None = None):
        if duration is not None and (duration <= 0 or not math.isfinite(duration)):
            raise ValueError("duration must be positive and finite")
        if max_messages is not None and (
            isinstance(max_messages, bool) or not isinstance(max_messages, int) or max_messages <= 0
        ):
            raise ValueError("max_messages must be a positive integer")
        reason = "requested_stop"
        try:
            self._record(
                "recorder_start",
                config=self.config.to_dict(),
                endpoint=self.adapter.endpoint,
                subscription=self.adapter.subscription(),
                duration_seconds=duration,
                max_messages=max_messages,
            )
            try:
                async with asyncio.timeout(duration):
                    await self._metadata()
                    await self._loop(max_messages)
            except TimeoutError:
                reason = "duration_elapsed"
            except asyncio.CancelledError:
                reason = "cancelled"
                raise
            except Exception:
                reason = "error"
                raise
        finally:
            self.book.invalidate("recorder stopped")
            try:
                if not self.writer.failed:
                    self._record("recorder_stop", reason=reason, counts=dict(self.stats))
            finally:
                self.writer.close()
                self._report(force=True)
        return {**dict(self.stats), "run_id": self.run_id, "stop_reason": reason}

    async def _loop(self, max_messages):
        failures = 0
        while not self.stop.is_set():
            self.session_id = uuid4().hex
            self.book.reset()
            self._previous_receive_time = None
            self.stats["connections_attempted"] += 1
            self._record("session_start")
            failure = None
            try:
                async with self.connector(
                    self.adapter.endpoint,
                    open_timeout=self.config.open_timeout,
                    close_timeout=2,
                    ping_interval=20,
                    ping_timeout=20,
                    max_size=self.config.max_message_bytes,
                    max_queue=self.config.max_queue,
                ) as websocket:
                    self.stats["connections_opened"] += 1
                    self._record("connected")
                    await websocket.send(json.dumps(self.adapter.subscription()))
                    self._record("subscription_sent", subscription=self.adapter.subscription())
                    await self._receive(websocket, max_messages)
                    return
            except FeedRejected as exc:
                self.stats["rejections"] += 1
                self._record("subscription_rejected", error=str(exc))
                raise
            except (FeedError, BookError, OSError, WebSocketException, TimeoutError) as exc:
                failure = f"{type(exc).__name__}: {exc}"
                self.stats["disconnects"] += 1
                self._record("connection_fault", error=failure)
                LOG.warning("Connection fault; fresh snapshot required: %s", failure)
            finally:
                self.book.invalidate(failure or "session ended")
                if not self.writer.failed:
                    self._record("session_end", reason=failure or "requested_stop")
            if self.stop.is_set():
                return
            if self.config.max_reconnects is not None and failures >= self.config.max_reconnects:
                raise FeedError("reconnect limit exhausted")
            delay = backoff_seconds(
                failures, self.config.backoff_initial, self.config.backoff_max, random.random()
            )
            failures += 1
            self.stats["reconnects"] += 1
            self._record("reconnect_scheduled", delay_seconds=delay)
            try:
                await asyncio.wait_for(self.stop.wait(), timeout=delay)
            except TimeoutError:
                pass

    async def _receive(self, websocket, max_messages):
        started = last_message = last_heartbeat = time.monotonic()
        acknowledged = False
        tracker = self.adapter.continuity_tracker()
        trade_uncertain = False
        while not self.stop.is_set():
            now = time.monotonic()
            if now - last_message > self.config.idle_timeout:
                raise FeedError("idle receive timeout")
            if now - last_heartbeat > self.config.heartbeat_timeout:
                raise FeedError("heartbeat timeout")
            if now - started > self.config.bootstrap_timeout and (
                not acknowledged or self.book.state != BookState.LIVE
            ):
                raise FeedError("snapshot/subscription bootstrap timeout")
            try:
                frame = await asyncio.wait_for(websocket.recv(), timeout=0.25)
            except TimeoutError:
                self._report()
                continue
            context = ReceiveContext(
                self.session_id,
                self.stats["messages_received"],
                time.time_ns(),
                time.monotonic_ns(),
            )
            last_message = time.monotonic()
            self.stats["messages_received"] += 1
            # Keep malformed frames available for replay and debugging.
            if isinstance(frame, bytes):
                self._record(
                    "raw_message",
                    context=context,
                    message_index=context.message_index,
                    encoding="base64",
                    payload=base64.b64encode(frame).decode("ascii"),
                )
            else:
                self._record(
                    "raw_message",
                    context=context,
                    message_index=context.message_index,
                    encoding="utf-8",
                    payload=frame,
                )
            if (
                self._previous_receive_time is not None
                and context.receive_time_ns < self._previous_receive_time
            ):
                self.stats["clock_regressions"] += 1
                self._record(
                    "clock_regression", context=context, message_index=context.message_index
                )
                raise FeedError("local wall clock regressed")
            self._previous_receive_time = context.receive_time_ns
            try:
                if not isinstance(frame, str):
                    raise FeedError("unexpected binary message; original bytes preserved")
                event = self.adapter.parse(frame, context)
            except FeedRejected:
                raise
            except FeedError as exc:
                self.stats["malformed_messages"] += 1
                self._record(
                    "malformed_message",
                    context=context,
                    message_index=context.message_index,
                    error=str(exc),
                )
                raise
            notice = tracker.observe(event)
            if notice:
                self.stats[notice["kind"]] += 1
                if notice["kind"] in ("trade_gap", "possible_trade_gap"):
                    trade_uncertain = True
                    LOG.warning("Trade coverage uncertain: %s", notice)
                self._record(
                    "quality_notice",
                    context=context,
                    message_index=context.message_index,
                    notice=notice,
                )
            self._record(
                "normalized_event",
                context=context,
                message_index=context.message_index,
                event=event.to_dict(),
                trade_coverage_uncertain=trade_uncertain,
            )
            self.stats[event.kind] += 1
            if event.kind == "subscriptions":
                acknowledged = True
            if event.kind == "heartbeat":
                last_heartbeat = time.monotonic()
            try:
                apply_book_event(self.book, event)
            except BookError as exc:
                self.stats["book_errors"] += 1
                self._record(
                    "book_error",
                    context=context,
                    message_index=context.message_index,
                    error=str(exc),
                )
                raise
            self._report()
            if max_messages is not None and self.stats["messages_received"] >= max_messages:
                self.stop.set()
