import json
import subprocess
import sys
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from feature_helpers import MS, observation_rows, observations
from test_build_dataset import write_records
from test_sampling import marker, raw

from market_microstructure.data.coinbase import CoinbaseAdapter
from market_microstructure.processing.build_dataset import build_dataset, file_hash
from market_microstructure.processing.config import ProcessingConfig
from market_microstructure.processing.feature_config import FeatureConfig, load_feature_config
from market_microstructure.processing.features import feature_table
from market_microstructure.processing.labels import label_table
from market_microstructure.processing.research_dataset import (
    build_feature_dataset,
    build_research_dataset,
    iter_research_blocks,
)

TIME = "2026-09-25T00:00:00Z"


def fixture_records(n=120):
    snapshot = {
        "type": "snapshot",
        "product_id": "BTC-USD",
        "bids": [[str(100 - k / 100), "2"] for k in range(10)],
        "asks": [[str(100.02 + k / 100), "1"] for k in range(10)],
    }
    reference = {
        "type": "last_match",
        "product_id": "BTC-USD",
        "time": TIME,
        "side": "buy",
        "price": "100",
        "size": "900",
        "trade_id": 0,
    }
    records = [
        {**marker("product_metadata"), "values": {"symbol": "BTC-USD", "quote_increment": ".01"}},
        marker("session_start"),
        raw(snapshot, 0, 0),
        raw(reference, 0, 1),
    ]
    index = 2
    for i in range(n):
        records += [
            raw(
                {
                    "type": "match",
                    "product_id": "BTC-USD",
                    "time": TIME,
                    "side": "sell" if i % 2 == 0 else "buy",
                    "price": "100",
                    "size": "0.3",
                    "trade_id": i + 1,
                },
                i * 100 + 50,
                index,
            ),
            raw(
                {
                    "type": "l2update",
                    "product_id": "BTC-USD",
                    "time": TIME,
                    "changes": [["buy", "100", str(2 + i % 3)]],
                },
                (i + 1) * 100,
                index + 1,
            ),
        ]
        index += 2
    return records + [marker("session_end")]


def pipeline(tmp_path, *, n=120, batch=17, name="output", records=None, **kwargs):
    source = write_records(
        tmp_path / (name + ".jsonl"), records if records is not None else fixture_records(n)
    )
    dest = tmp_path / name
    manifest = build_research_dataset(
        [source],
        dest,
        CoinbaseAdapter(),
        ProcessingConfig(batch_rows=batch),
        FeatureConfig(batch_rows=batch),
        **kwargs,
    )
    return dest, manifest


def assert_tables_close(a, b):
    assert a.column_names == b.column_names
    assert a.num_rows == b.num_rows
    for name in a.column_names:
        if pa.types.is_floating(a[name].type):
            np.testing.assert_allclose(
                a[name].to_numpy(), b[name].to_numpy(), rtol=1e-11, atol=1e-11, equal_nan=True
            )
        else:
            assert a[name].to_pylist() == b[name].to_pylist(), name


def test_raw_fixture_to_separate_feature_and_target_tables(tmp_path):
    dest, manifest = pipeline(tmp_path, train_end_ns=9000 * MS)
    assert manifest["status"] == "complete"
    features = pq.read_table(dest / "research/features.parquet")
    assert features.num_rows == 121
    assert len([n for n in features.column_names if n.startswith("f_")]) == 99
    assert not any(n.startswith(("y_", "label_")) for n in features.column_names)
    obs = pq.read_table(dest / "observations/snapshots.parquet")
    # last_match reference size 900 must never be treated as a live execution.
    assert obs["trade_count"].to_pylist() == [0] + [1] * 120
    assert obs["trade_buy_volume"][1].as_py() == Decimal(".3")
    assert obs["trade_sell_volume"][1].as_py() == 0
    for horizon in (100, 500, 1000, 5000):
        target = pq.read_table(dest / f"research/labels_{horizon}ms.parquet")
        assert target.num_rows == 121 - horizon // 100
        assert not any(n.startswith("f_") for n in target.column_names)
        delta = target[f"label_end_ns_{horizon}ms"].to_numpy() - target["timestamp_ns"].to_numpy()
        assert np.all(delta == horizon * MS)
    sub = json.loads((dest / "research/manifest.json").read_text())
    assert sub["max_context_rows"] <= sub["context_row_bound"]
    assert sub["threshold"]["tick_size"] == "0.01"
    for name, details in sub["outputs"].items():
        assert file_hash(dest / "research" / name) == details["sha256"]
    assert not list(dest.rglob("*.part"))
    assert not list(dest.rglob("_INCOMPLETE"))


def test_batch_and_input_partition_invariance(tmp_path):
    records = fixture_records(180)
    a, _ = pipeline(tmp_path, batch=7, name="one", records=records)
    files = [
        write_records(tmp_path / f"part{i}.jsonl", records[i : i + 13])
        for i in range(0, len(records), 13)
    ]
    b = tmp_path / "many"
    build_research_dataset(
        files, b, CoinbaseAdapter(), ProcessingConfig(batch_rows=31), FeatureConfig(batch_rows=31)
    )
    for name in ["features.parquet"] + [f"labels_{h}ms.parquet" for h in (100, 500, 1000, 5000)]:
        assert_tables_close(
            pq.read_table(a / "research" / name), pq.read_table(b / "research" / name)
        )


def test_full_batch_matches_incremental_with_segment_reset(tmp_path):
    first = observation_rows(170, mids=[100 + i % 9 for i in range(170)])
    second = observation_rows(
        90, mids=[200 + i % 5 for i in range(90)], segment="r/s/2", offset_ms=20000
    )
    all_rows = observations(first + second)
    path = tmp_path / "observations.parquet"
    pq.write_table(all_rows, path, row_group_size=19)
    config = FeatureConfig(batch_rows=11)
    stats = {"max_context_rows": 0}
    with pq.ParquetFile(path) as file:
        blocks = list(iter_research_blocks(file, config, 100, ".01", stats))
    actual = pa.concat_tables([block for block, _ in blocks])
    expected = []
    for rows in (first, second):
        obs = observations(rows)
        features, _ = feature_table(obs, config, 100)
        labels = label_table(obs, config, 100, ".01")
        expected.append(
            pa.table(
                {
                    **{n: features[n] for n in features.column_names},
                    **{n: labels[n] for n in labels.column_names},
                }
            )
        )
    assert_tables_close(actual, pa.concat_tables(expected))
    assert actual["f_mid_return_100ms"][170].as_py() is None
    assert actual["y_return_100ms"][169].as_py() is None
    assert actual["label_valid_5000ms"][169].as_py() is False
    assert stats["max_context_rows"] <= config.history_rows(100) + 22 + config.future_rows(100)


def test_trade_duplicate_is_not_double_counted_and_masks_coverage(tmp_path):
    records = fixture_records(4)
    duplicate = {
        **records[4],
        "message_index": 4,
        "receive_time_ns": 125 * MS,
        "receive_monotonic_ns": 125 * MS,
    }
    records.insert(6, duplicate)
    # Reindex originals; the trade ID in the duplicate payload is retained.
    index = 0
    for row in records:
        if row["record_type"] == "raw_message":
            row["message_index"] = index
            index += 1
    dest, _ = pipeline(tmp_path, name="duplicate", records=records)
    obs = pq.read_table(dest / "observations/snapshots.parquet")
    assert sum(obs["trade_count"].to_pylist()) == 4
    assert obs["trade_coverage_uncertain"].to_pylist()[2:] == [True] * 3
    f = pq.read_table(dest / "research/features.parquet")
    assert f["f_trade_count_100ms"][2].as_py() is None


def test_no_trade_reference_or_first_partial_segment_bucket_counted(tmp_path):
    records = fixture_records(3)
    # Move initial bootstrap off-grid. First row is a partial interval.
    for row in records:
        if row["record_type"] == "raw_message":
            row["receive_time_ns"] += 25 * MS
            row["receive_monotonic_ns"] += 25 * MS
    dest, _ = pipeline(tmp_path, name="partial", records=records)
    obs = pq.read_table(dest / "observations/snapshots.parquet")
    assert obs["timestamp_ns"][0].as_py() == 100 * MS
    assert obs["trade_count"][0].as_py() == 0
    assert obs["trade_count"].to_pylist()[1:] == [1, 1]


def test_gap_masks_trade_features_but_preserves_book_features(tmp_path):
    records = fixture_records(4)
    payload = json.loads(records[6]["payload"])
    payload["trade_id"] = 3  # Missing ID 2, followed later by an equal ID.
    records[6]["payload"] = json.dumps(payload)
    dest, _ = pipeline(tmp_path, name="gap", records=records)
    table = pq.read_table(dest / "research/features.parquet")
    assert table["f_trade_count_100ms"][2].as_py() is None
    assert table["f_midprice"][2].as_py() is not None


def test_research_cli_offline(tmp_path):
    source = write_records(tmp_path / "input.jsonl", fixture_records(3))
    root = Path(__file__).resolve().parents[1]
    destination = tmp_path / "cli"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/features.py"),
            "--input",
            str(source),
            "--output",
            str(destination),
            "--batch-rows",
            "2",
        ],
        capture_output=True,
        text=True,
        cwd=root,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["rows"]["features.parquet"] == 4


@pytest.mark.parametrize("state", ["incomplete", "hash", "tick", "existing", "empty"])
def test_failures_never_publish_research_dataset(tmp_path, state):
    source = write_records(tmp_path / "raw.jsonl", fixture_records(2) if state != "empty" else [])
    obs = tmp_path / "obs"
    build_dataset([source], obs, CoinbaseAdapter(), ProcessingConfig(), include_trade_buckets=True)
    destination = tmp_path / "research"
    kwargs = {}
    if state == "incomplete":
        (obs / "_INCOMPLETE").write_text("unsafe")
    elif state == "hash":
        with (obs / "snapshots.parquet").open("ab") as stream:
            stream.write(b"changed")
    elif state == "tick":
        kwargs["tick_size"] = ".02"
    elif state == "existing":
        destination.mkdir()
    else:
        kwargs["tick_size"] = ".01"
    with pytest.raises((ValueError, FileExistsError)):
        build_feature_dataset(obs, destination, FeatureConfig(), **kwargs)
    assert not (destination / "manifest.json").exists()
    if state == "empty":
        assert (destination / "_INCOMPLETE").exists()
        assert (destination / "failure.json").exists()


def test_missing_metadata_requires_verified_tick(tmp_path):
    records = fixture_records(2)[1:]
    source = write_records(tmp_path / "input.jsonl", records)
    obs = tmp_path / "obs"
    build_dataset([source], obs, CoinbaseAdapter(), ProcessingConfig(), include_trade_buckets=True)
    with pytest.raises(ValueError, match="tick metadata"):
        build_feature_dataset(obs, tmp_path / "no_tick", FeatureConfig())
    result = build_feature_dataset(obs, tmp_path / "explicit", FeatureConfig(), tick_size=".01")
    assert result["threshold"]["tick_origin"] == "explicit user configuration"


@pytest.mark.parametrize(
    "setting",
    [
        {"flow_windows_ms": (0,)},
        {"horizons_ms": (True,)},
        {"horizons_ms": (500, 100)},
        {"dynamics_lags_ms": (100, 100)},
        {"alignment_tolerance_ms": -1},
        {"batch_rows": 0},
        {"epsilon_ticks": "NaN"},
        {"epsilon_ticks": "0"},
        {"relative_epsilon": "-.1"},
    ],
)
def test_invalid_feature_configuration(setting):
    with pytest.raises(ValueError):
        FeatureConfig(**setting)


def test_config_loading_interval_rules_and_unknown_keys(tmp_path):
    path = tmp_path / "features.yaml"
    path.write_text("flow_windows_ms: [100, 500]\nbatch_rows: 13\n")
    assert load_feature_config(path).batch_rows == 13
    with pytest.raises(ValueError, match="multiples"):
        FeatureConfig().validate_interval(300)
    with pytest.raises(ValueError, match="tolerance"):
        replace(FeatureConfig(), alignment_tolerance_ms=100).validate_interval(100)
    path.write_text("unknown: 42\n")
    with pytest.raises(ValueError, match="unknown"):
        load_feature_config(path)
