import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zstandard
from test_sampling import marker, raw

from market_microstructure.data.coinbase import CoinbaseAdapter
from market_microstructure.processing.build_dataset import (
    build_dataset,
    discover_inputs,
    file_hash,
    iter_journal,
)
from market_microstructure.processing.config import ProcessingConfig, load_processing_config
from market_microstructure.processing.parquet import SnapshotSink
from market_microstructure.processing.replay import ReplayError


def write_records(path, records, compressed=False):
    data = "".join(json.dumps(r) + "\n" for r in records).encode()
    path.write_bytes(zstandard.ZstdCompressor().compress(data) if compressed else data)
    return path


def capture(messages):
    return [
        marker("session_start"),
        raw(messages["snapshot"], 0, 0),
        {**marker("normalized_event"), "event": {"annotation": "not applied"}},
        raw(messages["update"], 150, 1),
        raw(messages["heartbeat"], 550, 2),
        marker("session_end", 20000),
    ]


def build(paths, destination, **config):
    return build_dataset(paths, destination, CoinbaseAdapter(), ProcessingConfig(**config))


def test_parquet_schema_exact_values_bounded_groups_manifest(tmp_path, messages):
    source = write_records(tmp_path / "raw.jsonl", capture(messages))
    dest = tmp_path / "dataset"
    manifest = build([source], dest, batch_rows=2)
    file = pq.ParquetFile(dest / "snapshots.parquet")
    assert file.metadata.num_rows == manifest["rows"] == 6
    assert file.metadata.num_row_groups == 3
    assert manifest["max_buffered_rows"] == 2
    assert all(file.metadata.row_group(i).num_rows <= 2 for i in range(3))
    assert pa.types.is_decimal128(file.schema_arrow.field("midprice").type)
    assert file.schema_arrow.field("timestamp").type == pa.timestamp("ns", tz="UTC")
    batches = list(file.iter_batches(batch_size=2))
    rows = [row for batch in batches for row in batch.to_pylist()]
    assert [r["timestamp_ns"] for r in rows] == [i * 100_000_000 for i in range(6)]
    assert str(rows[2]["midprice"]).startswith("100.015")
    assert rows[0]["bid_px_3"] is None
    assert manifest["inputs"][0]["sha256"] == file_hash(source)
    assert manifest["outputs"]["snapshots.parquet"]["sha256"] == file_hash(
        dest / "snapshots.parquet"
    )
    assert not (dest / "_INCOMPLETE").exists()
    assert json.loads((dest / "manifest.json").read_text())["status"] == "complete"
    assert not list(dest.glob("*.part"))
    file.close()


def test_input_partition_and_row_group_size_do_not_change_rows(tmp_path, messages):
    records = capture(messages)
    whole = write_records(tmp_path / "whole.jsonl", records)
    first = write_records(tmp_path / "first.jsonl", records[:2])
    second = write_records(tmp_path / "second.jsonl", records[2:])
    build([whole], tmp_path / "a", batch_rows=2)
    build([first, second], tmp_path / "b", batch_rows=3)
    a = pq.read_table(tmp_path / "a/snapshots.parquet")
    b = pq.read_table(tmp_path / "b/snapshots.parquet")
    assert a.equals(b, check_metadata=False)
    assert (tmp_path / "a/segments.jsonl").read_text() == (
        tmp_path / "b/segments.jsonl"
    ).read_text()


def test_compressed_input_matches_plaintext(tmp_path, messages):
    records = capture(messages)
    plain = write_records(tmp_path / "raw.jsonl", records)
    compressed = write_records(tmp_path / "raw.jsonl.zst", records, compressed=True)
    build([plain], tmp_path / "a")
    build([compressed], tmp_path / "b")
    assert pq.read_table(tmp_path / "a/snapshots.parquet").equals(
        pq.read_table(tmp_path / "b/snapshots.parquet")
    )


def test_failed_build_retains_incomplete_marker_no_success_manifest(tmp_path, messages):
    source = write_records(tmp_path / "raw.jsonl", capture(messages))
    # Decimal rescaling that loses fractional price precision must fail, never round.
    with pytest.raises(pa.ArrowInvalid):
        build([source], tmp_path / "bad", decimal_scale=0, batch_rows=1)
    assert (tmp_path / "bad/_INCOMPLETE").exists()
    assert (tmp_path / "bad/failure.json").exists()
    assert not (tmp_path / "bad/manifest.json").exists()


def test_output_write_failure_does_not_publish_dataset(tmp_path, messages, monkeypatch):
    source = write_records(tmp_path / "raw.jsonl", capture(messages))

    def disk_full(self, row):
        raise OSError("simulated disk full")

    monkeypatch.setattr(SnapshotSink, "add", disk_full)
    with pytest.raises(OSError, match="disk full"):
        build([source], tmp_path / "bad")
    assert (tmp_path / "bad/_INCOMPLETE").exists()
    assert not (tmp_path / "bad/manifest.json").exists()


def test_existing_dataset_and_duplicate_inputs_rejected(tmp_path, messages):
    source = write_records(tmp_path / "raw.jsonl", capture(messages))
    output = tmp_path / "out"
    build([source], output)
    digest = file_hash(output / "snapshots.parquet")
    with pytest.raises(FileExistsError):
        build([source], output)
    assert file_hash(output / "snapshots.parquet") == digest
    with pytest.raises(ReplayError, match="unique"):
        build([source, source], tmp_path / "other")


def test_empty_output_is_explicit_valid_empty_parquet(tmp_path, messages):
    source = write_records(tmp_path / "raw.jsonl", [raw(messages["update"], 100, 0)])
    manifest = build([source], tmp_path / "out")
    assert manifest["rows"] == 0
    assert manifest["sample_time_min_ns"] is None
    assert pq.read_table(tmp_path / "out/snapshots.parquet").num_rows == 0
    assert manifest["quality_counts"]["updates_without_snapshot"] == 1


@pytest.mark.parametrize(
    "contents", [b"{}", b"[]\n", b"{bad}\n", b'{"x":1,"x":2}\n', b'{"x":NaN}\n']
)
def test_truncated_or_malformed_journal_rejected(tmp_path, contents):
    path = tmp_path / "bad.jsonl"
    path.write_bytes(contents)
    with pytest.raises(ReplayError):
        list(iter_journal(path, 1000))


def test_individual_raw_record_memory_bound(tmp_path):
    path = tmp_path / "large.jsonl"
    path.write_text(json.dumps({"x": "a" * 1000}) + "\n")
    with pytest.raises(ReplayError, match="byte bound"):
        list(iter_journal(path, 100))


def test_partial_input_rejected(tmp_path):
    path = tmp_path / "raw.jsonl.part"
    path.write_text("{}\n")
    with pytest.raises(ReplayError, match="incomplete"):
        list(iter_journal(path, 100))


def test_discovery_uses_file_index_across_clock_regression(tmp_path):
    first = tmp_path / "coinbase_BTC-USD_20260925_010000_run_a_000000.jsonl"
    second = tmp_path / "coinbase_BTC-USD_20260925_000000_run_a_000001.jsonl"
    first.touch()
    second.touch()
    assert discover_inputs(tmp_path, "coinbase", "BTC-USD") == [first, second]
    duplicate = second.with_suffix(".jsonl.zst")
    duplicate.touch()
    with pytest.raises(ReplayError, match="duplicate file index"):
        discover_inputs(tmp_path, "coinbase", "BTC-USD")


def test_discovery_does_not_silently_skip_part_files(tmp_path):
    (tmp_path / "coinbase_BTC-USD_20260925_010000_run_000000.jsonl.part").touch()
    with pytest.raises(ReplayError, match="unfinished"):
        discover_inputs(tmp_path, "coinbase", "BTC-USD")


def test_changed_input_detected(tmp_path, messages, monkeypatch):
    import market_microstructure.processing.build_dataset as module

    source = write_records(tmp_path / "raw.jsonl", capture(messages))
    original = module.file_hash

    def mutate_after_hash(path):
        digest = original(path)
        if path == source:
            with path.open("a") as stream:
                stream.write(json.dumps(marker("ignored_annotation")) + "\n")
        return digest

    monkeypatch.setattr(module, "file_hash", mutate_after_hash)
    with pytest.raises(ReplayError, match="changed while processing"):
        build([source], tmp_path / "bad")


def test_metadata_retained_in_dataset(tmp_path, messages):
    metadata = {**marker("product_metadata"), "values": {"quote_increment": "0.01"}}
    source = write_records(tmp_path / "raw.jsonl", [metadata, *capture(messages)])
    build([source], tmp_path / "out")
    saved = json.loads((tmp_path / "out/source_metadata.jsonl").read_text())
    assert saved["values"]["quote_increment"] == "0.01"


def test_cli_runs_without_network(tmp_path, messages):
    source = write_records(tmp_path / "raw.jsonl", capture(messages))
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/process.py"),
            "--input",
            str(source),
            "--output",
            str(tmp_path / "out"),
            "--batch-rows",
            "2",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["rows"] == 6


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("interval_ms", 0),
        ("depth", True),
        ("max_staleness_ms", -1),
        ("batch_rows", 1.5),
        ("decimal_precision", 39),
        ("decimal_scale", -1),
        ("decimal_scale", 38),
        ("compression", "unknown"),
        ("max_record_bytes", 0),
    ],
)
def test_processing_config_rejects_invalid_settings(key, value):
    with pytest.raises(ValueError):
        ProcessingConfig(**{key: value})


def test_processing_config_paths_and_overrides(tmp_path):
    path = tmp_path / "processing.yaml"
    path.write_text("raw_dir: raw\noutput_dir: processed\ninterval_ms: 100\n")
    config, raw_dir, output_dir = load_processing_config(path, interval_ms=50)
    assert config.interval_ms == 50
    assert raw_dir == tmp_path / "raw"
    assert output_dir == tmp_path / "processed"
    path.write_text("samplng_ms: 100\n")
    with pytest.raises(ValueError, match="unknown"):
        load_processing_config(path)
