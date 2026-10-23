import json

import pytest

from market_microstructure.data.storage import JsonlWriter, StorageError, iter_records

EPOCH = 1_800_000_000_000_000_000


def row(index=0, timestamp=EPOCH):
    return {"receive_time_ns": timestamp, "index": index, "payload": '  { "x": 1 }\n'}


def closed_paths(root):
    return sorted(root.rglob("*.jsonl")) + sorted(root.rglob("*.jsonl.zst"))


def test_payload_preserved_and_partial_only_while_open(tmp_path):
    writer = JsonlWriter(tmp_path, "BTC-USD")
    writer.write(row())
    assert len(list(tmp_path.rglob("*.part"))) == 1
    assert not closed_paths(tmp_path)
    writer.close()
    assert list(iter_records(closed_paths(tmp_path))) == [row()]
    assert not list(tmp_path.rglob("*.part"))
    writer.close()


def test_time_size_rotation_and_order(tmp_path):
    writer = JsonlWriter(tmp_path, "BTC-USD", rotate_seconds=1, max_bytes=150)
    records = [row(0), row(1), row(2, EPOCH + 1_000_000_000)]
    for value in records:
        writer.write(value)
    writer.close()
    assert len(closed_paths(tmp_path)) == 3
    assert list(iter_records(closed_paths(tmp_path))) == records


def test_oversized_record_kept_intact(tmp_path):
    writer = JsonlWriter(tmp_path, "BTC-USD", max_bytes=5)
    writer.write(row())
    writer.close()
    assert list(iter_records(closed_paths(tmp_path))) == [row()]


def test_zstd_roundtrip_only_after_closed(tmp_path):
    writer = JsonlWriter(tmp_path, "BTC-USD", compress=True)
    writer.write(row())
    writer.close()
    assert len(list(tmp_path.rglob("*.zst"))) == 1
    assert not list(tmp_path.rglob("*.jsonl"))
    assert list(iter_records(closed_paths(tmp_path))) == [row()]


def test_fsync_failure_is_fatal_and_leaves_part(tmp_path, monkeypatch):
    writer = JsonlWriter(tmp_path, "BTC-USD", fsync_seconds=0)

    def fail(fd):
        raise OSError("simulated disk failure")

    with monkeypatch.context() as patch:
        patch.setattr("market_microstructure.data.storage.os.fsync", fail)
        with pytest.raises(StorageError, match="disk failure"):
            writer.write(row())
    with pytest.raises(StorageError):
        writer.write(row(1))
    writer.close()
    assert len(list(tmp_path.rglob("*.part"))) == 1
    assert not closed_paths(tmp_path)


def test_compression_failure_preserves_complete_original(tmp_path, monkeypatch):
    writer = JsonlWriter(tmp_path, "BTC-USD", compress=True)
    writer.write(row())

    def fail(*args, **kwargs):
        raise RuntimeError("simulated compression failure")

    monkeypatch.setattr("zstandard.ZstdCompressor", fail)
    with pytest.raises(StorageError):
        writer.close()
    assert list(iter_records(closed_paths(tmp_path))) == [row()]


def test_no_overwrite_on_reused_run_id(tmp_path):
    first = JsonlWriter(tmp_path, "BTC-USD", run_id="fixed")
    first.write(row())
    second = JsonlWriter(tmp_path, "BTC-USD", run_id="fixed")
    with pytest.raises(StorageError):
        second.write(row())
    second.close()
    first.close()
    assert list(iter_records(closed_paths(tmp_path))) == [row()]


def test_reject_partial_and_truncated_files(tmp_path):
    partial = tmp_path / "x.jsonl.part"
    partial.write_text("{}\n")
    with pytest.raises(StorageError):
        list(iter_records([partial]))
    truncated = tmp_path / "x.jsonl"
    truncated.write_text(json.dumps(row()))
    with pytest.raises(StorageError, match="truncated"):
        list(iter_records([truncated]))


@pytest.mark.parametrize("symbol", ["../BTC-USD", "BTC/USD", "btc-usd", ""])
def test_path_input_validation(tmp_path, symbol):
    with pytest.raises(ValueError):
        JsonlWriter(tmp_path, symbol)
