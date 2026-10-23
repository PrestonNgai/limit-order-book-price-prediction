import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from split_helpers import refresh_manifest, research_fixture
from test_research_dataset import pipeline

from market_microstructure.processing.split_config import SplitConfig, load_split_config
from market_microstructure.processing.split_dataset import build_split_dataset, iter_model_batches
from market_microstructure.processing.splitting import make_split_plan, scan_periods

MS = 1_000_000


def test_time_split_exact_boundary_purge_and_per_horizon_eligibility(tmp_path):
    source = research_fixture(tmp_path / "source")
    out = tmp_path / "split"
    result = build_split_dataset(source, out, SplitConfig(batch_rows=9, sequence_length=3))
    assert result["method"] == "time"
    assert [result["ranges"][s]["rows"] for s in ("train", "validation", "test")] == [70, 15, 15]
    assert result["ranges"]["train"]["end_exclusive_ns"] == 7000 * MS
    expected = {100: [69, 14, 14], 500: [65, 10, 10], 1000: [60, 5, 5], 5000: [20, 0, 0]}
    for h, counts in expected.items():
        actual = [
            result["target_counts"][f"{h}ms"][s]["rows"] for s in ("train", "validation", "test")
        ]
        assert actual == counts
        for split in ("train", "validation", "test"):
            table = pq.read_table(out / split / f"labels_{h}ms.parquet")
            if table.num_rows:
                assert (
                    table[f"label_end_ns_{h}ms"].to_numpy().max()
                    < result["ranges"][split]["end_exclusive_ns"]
                )
    assert result["target_counts"]["100ms"]["train"]["purged"] == 1
    assert result["target_counts"]["500ms"]["train"]["purged"] == 5
    assert not result["three_way_evaluation_available"]["5000ms"]
    assert result["target_counts"]["100ms"]["validation"]["sequence_rows"] == 12
    assert not list(out.rglob("*.part"))


def test_prefer_whole_sessions_and_explicit_time_override(tmp_path):
    source = research_fixture(
        tmp_path / "source", periods=[(50, 0, "a"), (50, 6000, "b"), (50, 12000, "c")]
    )
    periods = scan_periods(source / "features.parquet", 100 * MS, 7)
    auto = make_split_plan(periods, SplitConfig(), 100 * MS)
    assert auto["method"] == "sessions"
    assert auto["ranges"]["train"]["end_exclusive_ns"] == 6000 * MS
    assert auto["ranges"]["validation"]["end_exclusive_ns"] == 12000 * MS
    out = tmp_path / "out"
    build_split_dataset(source, out, SplitConfig(batch_rows=11), horizons_ms=[100])
    sessions = [
        set(pq.read_table(out / s / "features.parquet")["session_id"].to_pylist())
        for s in ("train", "validation", "test")
    ]
    assert sessions == [{"a"}, {"b"}, {"c"}]
    explicit = make_split_plan(
        periods,
        SplitConfig(mode="explicit", train_end_ns=5000 * MS, validation_end_ns=11000 * MS),
        100 * MS,
    )
    assert explicit["ranges"]["train"]["end_exclusive_ns"] == 5000 * MS


def test_sequence_history_cannot_cross_gap_or_split(tmp_path):
    source = research_fixture(tmp_path / "source", periods=[(40, 0, "s"), (60, 6000, "s")])
    out = tmp_path / "out"
    result = build_split_dataset(source, out, SplitConfig(batch_rows=7, sequence_length=20))
    index = pq.read_table(out / "index.parquet").to_pylist()
    for split in ("train", "validation", "test"):
        targets = pq.read_table(out / split / "labels_100ms.parquet").to_pylist()
        for target in targets:
            if not target["sequence_eligible"]:
                assert target["sequence_start_row_id"] is None
                continue
            start, end = target["sequence_start_row_id"], target["row_id"]
            history = index[start : end + 1]
            assert len(history) == 20
            assert {r["split"] for r in history} == {split}
            assert {r["segment_id"] for r in history} == {target["segment_id"]}
            assert history[-1]["timestamp_ns"] == target["timestamp_ns"]
            assert history[-1]["timestamp_ns"] - history[0]["timestamp_ns"] == 1900 * MS
    assert index[40]["sequence_eligible"] is False
    assert index[result["ranges"]["validation"]["start_row_id"]]["sequence_eligible"] is False


def test_test_loader_is_sealed_and_integrity_checked(tmp_path):
    source = research_fixture(tmp_path / "source")
    out = tmp_path / "out"
    build_split_dataset(source, out, SplitConfig(batch_rows=13))
    with pytest.raises(PermissionError, match="sealed"):
        list(iter_model_batches(out, 100, "test"))
    # Explicit access is tested only on this declared synthetic fixture.
    batches = list(iter_model_batches(out, 100, "test", allow_test=True, batch_rows=4))
    assert sum(t.num_rows for t in batches) == 14
    for table in batches:
        assert np.isfinite(table["f_midprice"].to_numpy()).all()
    with (out / "test/features.parquet").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="hash"):
        list(iter_model_batches(out, 100, "test", allow_test=True))


def test_loader_uses_identical_frozen_transform_for_validation(tmp_path):
    source = research_fixture(tmp_path / "source", n=200)
    out = tmp_path / "out"
    result = build_split_dataset(source, out, SplitConfig(batch_rows=17))
    state = json.loads((out / result["normalizers"]["500ms"]).read_text())
    raw = pa.concat_tables(list(iter_model_batches(out, 500, "validation", transformed=False)))
    scaled = pa.concat_tables(list(iter_model_batches(out, 500, "validation")))
    i = state["feature_columns"].index("f_midprice")
    expected = (raw["f_midprice"].to_numpy() - state["impute_mean"][i]) / state["scale"][i]
    np.testing.assert_allclose(scaled["f_midprice"].to_numpy(), expected)
    sequence = list(iter_model_batches(out, 500, "validation", sequence_only=True))
    assert (
        sum(t.num_rows for t in sequence)
        == result["target_counts"]["500ms"]["validation"]["sequence_rows"]
    )


def test_raw_capture_fixture_through_split_cli(tmp_path):
    raw_pipeline, _ = pipeline(tmp_path, n=120, batch=19)
    out = tmp_path / "split"
    root = Path(__file__).resolve().parents[1]
    process = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/split.py"),
            "--input",
            str(raw_pipeline / "research"),
            "--output",
            str(out),
            "--batch-rows",
            "11",
            "--horizons-ms",
            "100",
            "500",
        ],
        capture_output=True,
        text=True,
        cwd=root,
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["test_status"] == "sealed"
    assert sum(t.num_rows for t in iter_model_batches(out, 100, "train")) > 0


def test_no_eligible_train_target_is_explicitly_unavailable(tmp_path):
    source = research_fixture(tmp_path / "source", n=10, horizons=(5000,))
    out = tmp_path / "out"
    result = build_split_dataset(source, out, SplitConfig(batch_rows=3))
    assert result["normalizers"]["5000ms"] is None
    assert not result["three_way_evaluation_available"]["5000ms"]
    with pytest.raises(ValueError, match="no eligible TRAIN"):
        list(iter_model_batches(out, 5000, "validation"))


@pytest.mark.parametrize("issue", ["duplicate", "gap", "overlap", "identity"])
def test_bad_chronology_fails_closed(tmp_path, issue):
    source = research_fixture(tmp_path / "source")
    table = pq.read_table(source / "features.parquet")
    times = table["timestamp_ns"].to_pylist()
    if issue in ("duplicate", "gap", "overlap"):
        times[30] += {"duplicate": -100 * MS, "gap": MS, "overlap": -200 * MS}[issue]
        table = table.set_column(
            table.schema.get_field_index("timestamp_ns"), "timestamp_ns", pa.array(times)
        )
    else:
        sessions = table["session_id"].to_pylist()
        sessions[30] = "other"
        table = table.set_column(
            table.schema.get_field_index("session_id"), "session_id", pa.array(sessions)
        )
    pq.write_table(table, source / "features.parquet")
    refresh_manifest(source)
    with pytest.raises(ValueError):
        build_split_dataset(source, tmp_path / "out", SplitConfig(batch_rows=7))


@pytest.mark.parametrize(
    "issue", ["target_identity", "endpoint", "source_hash", "prior_diagnostics"]
)
def test_target_and_provenance_failures_do_not_publish(tmp_path, issue):
    source = research_fixture(tmp_path / "source")
    if issue in ("target_identity", "endpoint"):
        table = pq.read_table(source / "labels_100ms.parquet")
        column = "session_id" if issue == "target_identity" else "label_end_ns_100ms"
        values = table[column].to_pylist()
        values[2] = "wrong" if issue == "target_identity" else 99999 * MS
        table = table.set_column(table.schema.get_field_index(column), column, pa.array(values))
        pq.write_table(table, source / "labels_100ms.parquet")
        refresh_manifest(source)
    elif issue == "source_hash":
        with (source / "features.parquet").open("ab") as stream:
            stream.write(b"corrupt")
    else:
        path = source / "training_diagnostics.json"
        path.write_text(
            json.dumps({"training_start_ns": 0, "training_end_exclusive_ns": 8000 * MS})
        )
        manifest = json.loads((source / "manifest.json").read_text())
        manifest["outputs"][path.name] = {}
        (source / "manifest.json").write_text(json.dumps(manifest))
        refresh_manifest(source)
    out = tmp_path / "out"
    with pytest.raises(ValueError):
        build_split_dataset(source, out, SplitConfig(batch_rows=9))
    assert not (out / "manifest.json").exists()
    if out.exists():
        assert (out / "_INCOMPLETE").exists()


def test_existing_output_and_incomplete_input_are_rejected(tmp_path):
    source = research_fixture(tmp_path / "source")
    out = tmp_path / "out"
    out.mkdir()
    (out / "sentinel").write_text("keep")
    with pytest.raises(FileExistsError):
        build_split_dataset(source, out, SplitConfig())
    assert (out / "sentinel").read_text() == "keep"
    (source / "_INCOMPLETE").write_text("unsafe")
    with pytest.raises(ValueError, match="incomplete"):
        build_split_dataset(source, tmp_path / "other", SplitConfig())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "random"},
        {"train_fraction": "0"},
        {"validation_fraction": ".4"},
        {"train_fraction": "NaN"},
        {"train_fraction": "invalid"},
        {"train_end_ns": 1},
        {"mode": "explicit", "train_end_ns": 2, "validation_end_ns": 1},
        {"sequence_length": 0},
        {"batch_rows": True},
        {"regime_quantiles": (1,)},
        {"regime_quantiles": (0.7, 0.3)},
    ],
)
def test_invalid_split_configuration(kwargs):
    with pytest.raises(ValueError):
        SplitConfig(**kwargs)


def test_session_requirement_config_file_and_empty_time_split(tmp_path):
    source = research_fixture(tmp_path / "source", periods=[(10, 0, "a"), (10, 100000, "b")])
    with pytest.raises(ValueError, match="three sessions"):
        build_split_dataset(source, tmp_path / "sessions", SplitConfig(mode="sessions"))
    with pytest.raises(ValueError, match="no feature rows"):
        build_split_dataset(source, tmp_path / "time", SplitConfig(mode="time"))
    path = tmp_path / "splits.yaml"
    path.write_text("mode: time\nsequence_length: 5\n")
    assert load_split_config(path).sequence_length == 5
    path.write_text("shuffle: true\n")
    with pytest.raises(ValueError, match="unknown"):
        load_split_config(path)
