import asyncio
import base64
import json
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from websockets.asyncio.server import serve

from market_microstructure.book import BookState
from market_microstructure.data.base import FeedError, FeedRejected
from market_microstructure.data.coinbase import CoinbaseAdapter
from market_microstructure.data.recorder import Recorder, backoff_seconds
from market_microstructure.data.storage import JsonlWriter, StorageError, iter_records
from market_microstructure.utils.config import CollectorConfig


class FakeSocket:
    def __init__(self, frames):
        self.frames = iter(frames)
        self.sent = []

    async def send(self, payload):
        self.sent.append(json.loads(payload))

    async def recv(self):
        frame = next(self.frames, None)
        if frame is None:
            await asyncio.sleep(100)
        if isinstance(frame, Exception):
            raise frame
        return frame if isinstance(frame, (str, bytes)) else json.dumps(frame)


class Connections:
    def __init__(self, sessions):
        self.sessions = iter(sessions)
        self.sockets = []
        self.attempts = 0
        self.exits = 0

    @asynccontextmanager
    async def __call__(self, endpoint, **options):
        self.attempts += 1
        session = next(self.sessions)
        if isinstance(session, Exception):
            raise session
        socket = FakeSocket(session)
        self.sockets.append(socket)
        try:
            yield socket
        finally:
            self.exits += 1


def configuration(tmp_path, **kwargs):
    return CollectorConfig(
        **{
            "output_dir": tmp_path,
            "fetch_product_metadata": False,
            "backoff_initial": 0.001,
            "backoff_max": 0.002,
            **kwargs,
        }
    )


def records(tmp_path):
    return list(iter_records(sorted(tmp_path.rglob("*.jsonl"))))


def test_full_offline_capture_preserves_original_before_normalization(tmp_path, messages):
    original = "  " + json.dumps(messages["snapshot"], indent=2) + "\n"
    connections = Connections(
        [
            [
                messages["subscriptions"],
                original,
                messages["reference"],
                messages["match"],
                messages["heartbeat"],
            ]
        ]
    )
    recorder = Recorder(CoinbaseAdapter(), configuration(tmp_path), connector=connections)
    summary = asyncio.run(recorder.run(max_messages=5))
    result = records(tmp_path)
    raw = [row for row in result if row["record_type"] == "raw_message"]
    assert raw[1]["payload"] == original
    assert [row["message_index"] for row in raw] == list(range(5))
    for row in raw:
        position = result.index(row)
        normalized = next(
            item for item in result[position + 1 :] if item["record_type"] == "normalized_event"
        )
        assert normalized["message_index"] == row["message_index"]
        assert normalized["receive_time_ns"] == row["receive_time_ns"]
    assert summary["snapshot"] == summary["trade"] == summary["trade_reference"] == 1
    assert connections.exits == 1
    assert recorder.book.state == BookState.INVALID
    assert result[-1]["record_type"] == "recorder_stop"
    assert not list(tmp_path.rglob("*.part"))


def test_malformed_message_reconnects_resubscribes_and_requires_new_snapshot(tmp_path, messages):
    connections = Connections(
        [
            [messages["subscriptions"], messages["snapshot"], "{broken"],
            [
                messages["subscriptions"],
                messages["snapshot"],
                messages["reference"],
                messages["match"],
            ],
        ]
    )
    recorder = Recorder(CoinbaseAdapter(), configuration(tmp_path), connector=connections)
    summary = asyncio.run(recorder.run(max_messages=7))
    assert summary["malformed_messages"] == 1
    assert summary["reconnects"] == 1
    assert summary["snapshot"] == 2
    assert (
        len(
            {
                row["session_id"]
                for row in records(tmp_path)
                if row["record_type"] == "session_start"
            }
        )
        == 2
    )
    assert all(socket.sent == [CoinbaseAdapter().subscription()] for socket in connections.sockets)


def test_connection_error_retries_then_collects(tmp_path, messages):
    connections = Connections(
        [OSError("unreachable"), [messages["subscriptions"], messages["snapshot"]]]
    )
    summary = asyncio.run(
        Recorder(CoinbaseAdapter(), configuration(tmp_path), connector=connections).run(
            max_messages=2
        )
    )
    assert summary["connections_attempted"] == 2
    assert summary["reconnects"] == 1


def test_update_before_snapshot_reconnects_instead_of_partial_book(tmp_path, messages):
    connections = Connections(
        [
            [messages["update"]],
            [messages["subscriptions"], messages["snapshot"]],
        ]
    )
    summary = asyncio.run(
        Recorder(CoinbaseAdapter(), configuration(tmp_path), connector=connections).run(
            max_messages=3
        )
    )
    assert summary["book_errors"] == 1
    assert summary["reconnects"] == 1


def test_crossed_book_triggers_recovery(tmp_path, messages):
    crossed = {**messages["update"], "changes": [["buy", "101", "1"]]}
    connections = Connections([[messages["snapshot"], crossed]])
    recorder = Recorder(
        CoinbaseAdapter(), configuration(tmp_path, max_reconnects=0), connector=connections
    )
    with pytest.raises(FeedError, match="reconnect limit"):
        asyncio.run(recorder.run())
    assert recorder.stats["book_errors"] == 1
    assert any(row["record_type"] == "book_error" for row in records(tmp_path))


def test_subscription_error_is_fatal_and_retained(tmp_path, messages):
    connections = Connections([[messages["error"]]])
    with pytest.raises(FeedRejected):
        asyncio.run(
            Recorder(CoinbaseAdapter(), configuration(tmp_path), connector=connections).run()
        )
    assert connections.attempts == 1
    assert any(row["record_type"] == "subscription_rejected" for row in records(tmp_path))
    assert records(tmp_path)[-1]["reason"] == "error"


def test_binary_frame_preserved_and_flagged(tmp_path):
    payload = b"\xff\x00unexpected"
    connections = Connections([[payload]])
    with pytest.raises(FeedError):
        asyncio.run(
            Recorder(
                CoinbaseAdapter(), configuration(tmp_path, max_reconnects=0), connector=connections
            ).run()
        )
    raw = next(row for row in records(tmp_path) if row["record_type"] == "raw_message")
    assert base64.b64decode(raw["payload"]) == payload


def test_writer_failure_is_fatal_not_a_reconnect(tmp_path, monkeypatch):
    connections = Connections([])
    writer = JsonlWriter(tmp_path, "BTC-USD", fsync_seconds=0)

    def fail(fd):
        raise OSError("disk full")

    recorder = Recorder(
        CoinbaseAdapter(), configuration(tmp_path), writer=writer, connector=connections
    )
    with monkeypatch.context() as patch:
        patch.setattr("market_microstructure.data.storage.os.fsync", fail)
        with pytest.raises(StorageError):
            asyncio.run(recorder.run())
    assert connections.attempts == 0
    assert list(tmp_path.rglob("*.part"))


@pytest.mark.parametrize(
    "timeout_key",
    [
        "idle_timeout",
        "heartbeat_timeout",
        "bootstrap_timeout",
    ],
)
def test_watchdog_timeout_without_traffic(tmp_path, timeout_key):
    connections = Connections([[]])
    config = configuration(tmp_path, max_reconnects=0, **{timeout_key: 0.01})
    with pytest.raises(FeedError, match="reconnect limit"):
        asyncio.run(Recorder(CoinbaseAdapter(), config, connector=connections).run())
    fault = next(row for row in records(tmp_path) if row["record_type"] == "connection_fault")
    assert "timeout" in fault["error"]


def test_duration_and_stop_event_finalize_files(tmp_path, messages):
    async def scenario():
        recorder = Recorder(
            CoinbaseAdapter(),
            configuration(tmp_path),
            connector=Connections([[messages["snapshot"]]]),
        )
        result = await recorder.run(duration=0.02)
        assert result["stop_reason"] == "duration_elapsed"

        other = Recorder(
            CoinbaseAdapter(),
            configuration(tmp_path / "stop"),
            connector=Connections([[messages["snapshot"]]]),
        )
        task = asyncio.create_task(other.run())
        await asyncio.sleep(0.01)
        other.stop.set()
        await task

    asyncio.run(scenario())
    assert not list(tmp_path.rglob("*.part"))


def test_task_cancellation_still_records_stop_and_closes(tmp_path, messages):
    async def scenario():
        recorder = Recorder(
            CoinbaseAdapter(),
            configuration(tmp_path),
            connector=Connections([[messages["snapshot"]]]),
        )
        task = asyncio.create_task(recorder.run())
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert records(tmp_path)[-1]["reason"] == "cancelled"
    assert not list(tmp_path.rglob("*.part"))


def test_trade_gap_is_visible_and_quality_remains_uncertain(tmp_path, messages):
    gap_trade = {**messages["match"], "trade_id": 14}
    summary = asyncio.run(
        Recorder(
            CoinbaseAdapter(),
            configuration(tmp_path),
            connector=Connections(
                [[messages["snapshot"], messages["reference"], gap_trade, messages["update"]]]
            ),
        ).run(max_messages=4)
    )
    assert summary["trade_gap"] == 1
    normalized = [row for row in records(tmp_path) if row["record_type"] == "normalized_event"]
    assert normalized[-1]["trade_coverage_uncertain"]
    assert any(row["record_type"] == "quality_notice" for row in records(tmp_path))


def test_metadata_fetch_recorded_separately(tmp_path, messages):
    payload = json.dumps(
        {"id": "BTC-USD", "quote_increment": "0.01", "base_increment": "0.00000001"}
    )
    asyncio.run(
        Recorder(
            CoinbaseAdapter(),
            configuration(tmp_path, fetch_product_metadata=True),
            connector=Connections([[messages["snapshot"]]]),
            metadata_fetcher=lambda url, timeout: payload,
        ).run(max_messages=1)
    )
    result = records(tmp_path)
    assert (
        next(row for row in result if row["record_type"] == "product_metadata_raw")["payload"]
        == payload
    )


def test_metadata_failure_is_explicit_not_a_guessed_tick(tmp_path, messages):
    def unavailable(url, timeout):
        raise OSError("network unavailable")

    asyncio.run(
        Recorder(
            CoinbaseAdapter(),
            configuration(tmp_path, fetch_product_metadata=True),
            connector=Connections([[messages["snapshot"]]]),
            metadata_fetcher=unavailable,
        ).run(max_messages=1)
    )
    assert any(row["record_type"] == "metadata_unavailable" for row in records(tmp_path))
    assert not any(row["record_type"] == "product_metadata" for row in records(tmp_path))


def test_real_local_websocket_transport(tmp_path, messages):
    async def scenario():
        async def handler(socket):
            subscription = json.loads(await socket.recv())
            assert subscription == CoinbaseAdapter().subscription()
            for name in ("subscriptions", "snapshot", "update", "match"):
                await socket.send(json.dumps(messages[name]))
            await socket.wait_closed()

        async with serve(handler, "127.0.0.1", 0) as server:
            adapter = CoinbaseAdapter()
            adapter.endpoint = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
            summary = await Recorder(adapter, configuration(tmp_path)).run(max_messages=4)
            assert summary["book_update"] == 1
            assert summary["trade"] == 1

    asyncio.run(scenario())


def test_backoff_bounded():
    assert backoff_seconds(0, 1, 30, 0) == 0.5
    assert backoff_seconds(1000, 1, 30, 1) == 30


def test_clock_regression_reconnects_to_new_clock_segment(tmp_path, messages, monkeypatch):
    readings = iter([100, 90, 91, 92])
    connections = Connections(
        [
            [messages["snapshot"], messages["update"]],
            [messages["subscriptions"], messages["snapshot"]],
        ]
    )
    recorder = Recorder(CoinbaseAdapter(), configuration(tmp_path), connector=connections)
    original_record = recorder._record

    def fixed_marker_clock(record_type, *, context=None, **payload):
        if context is None:
            from market_microstructure.data.base import ReceiveContext

            context = ReceiveContext(recorder.session_id, -1, 100, 100)
        original_record(record_type, context=context, **payload)

    monkeypatch.setattr(recorder, "_record", fixed_marker_clock)
    monkeypatch.setattr(
        "market_microstructure.data.recorder.time",
        SimpleNamespace(
            time_ns=lambda: next(readings),
            monotonic=time.monotonic,
            monotonic_ns=time.monotonic_ns,
        ),
    )
    summary = asyncio.run(recorder.run(max_messages=4))
    assert summary["clock_regressions"] == 1
    assert summary["snapshot"] == 2
    assert summary["reconnects"] == 1
