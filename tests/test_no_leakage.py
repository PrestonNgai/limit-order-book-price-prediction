import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from split_helpers import refresh_manifest, research_fixture

from market_microstructure.processing.preprocessing import TrainPreprocessor
from market_microstructure.processing.regime_cutoffs import TrainRegimeQuantiles, apply_regimes
from market_microstructure.processing.split_config import SplitConfig
from market_microstructure.processing.split_dataset import build_split_dataset


def fitting_table(values, *, split="train", offset=0):
    count = len(next(iter(values.values())))
    data = {
        "row_id": pa.array(np.arange(offset, offset + count)),
        "timestamp_ns": pa.array(np.arange(offset, offset + count) * 100),
        "label_end_ns_100ms": pa.array(np.arange(offset + 1, offset + count + 1) * 100),
        "y_return_100ms": pa.array(np.zeros(count)),
        "y_class_100ms": pa.array(np.zeros(count, dtype=np.int8)),
        "split": pa.array([split] * count),
    }
    data.update({name: pa.array(v, type=pa.float64()) for name, v in values.items()})
    return pa.table(data)


def test_train_mean_imputation_population_scaling_and_all_missing_policy():
    train = fitting_table(
        {"f_a": [1, 3, None], "f_constant": [7, 7, 7], "f_empty": [None, None, None]}
    )
    fitter = TrainPreprocessor(["f_a", "f_constant", "f_empty"], 100, 0, 1000)
    fitter.update(train)
    state = fitter.finalize()
    assert state["impute_mean"] == [2, 7, 0]
    assert state["observed_counts"] == [2, 3, 0]
    assert state["all_missing_features"] == ["f_empty"]
    np.testing.assert_allclose(state["scale"], [np.sqrt(2 / 3), 1, 1])
    transformed = fitter.transform(train, state)
    np.testing.assert_allclose(transformed.mean(axis=0), [0, 0, 0], atol=1e-15)
    np.testing.assert_allclose(transformed[:, 0].var(), 1)
    heldout = fitting_table(
        {"f_a": [1000, None], "f_constant": [1000, None], "f_empty": [9, None]}, split="validation"
    )
    before = json.dumps(state, sort_keys=True)
    result = fitter.transform(heldout, json.loads(before))
    assert result[1].tolist() == [0, 0, 0]
    assert result[0, 2] == 9
    assert json.dumps(state, sort_keys=True) == before
    with pytest.raises(ValueError, match="frozen"):
        fitter.update(train)


@pytest.mark.parametrize("role", ["validation", "test"])
def test_preprocessing_and_regime_fitting_reject_heldout_rows(tmp_path, role):
    table = fitting_table({"f_spread": [1, 3]}, split=role)
    with pytest.raises(ValueError, match="TRAIN"):
        TrainPreprocessor(["f_spread"], 100, 0, 1000).update(table)
    q = TrainRegimeQuantiles(tmp_path / "fit.sqlite", 100, 0, 1000, [0.5])
    try:
        with pytest.raises(ValueError, match="TRAIN"):
            q.update(table)
    finally:
        q.close()


@pytest.mark.parametrize("defect", ["endpoint", "time", "null_endpoint", "duplicate"])
def test_fitter_enforces_purged_ordered_training_scope(defect):
    table = fitting_table({"f_a": [1, 2, 3]})
    if defect == "null_endpoint":
        table = table.set_column(
            table.schema.get_field_index("label_end_ns_100ms"),
            "label_end_ns_100ms",
            pa.array([100, None, 300], type=pa.int64()),
        )
    elif defect == "time":
        table = table.set_column(
            table.schema.get_field_index("timestamp_ns"), "timestamp_ns", pa.array([-1, 100, 200])
        )
    elif defect == "duplicate":
        table = table.set_column(
            table.schema.get_field_index("timestamp_ns"), "timestamp_ns", pa.array([0, 0, 200])
        )
    with pytest.raises(ValueError):
        TrainPreprocessor(["f_a"], 100, 0, 300 if defect == "endpoint" else 1000).update(table)


def test_large_constant_values_have_unit_scale_and_batch_invariant_fingerprint():
    table = fitting_table({"f_a": [84620.155] * 101, "f_missing": [None] * 101})
    states = []
    for batch in (1, 17, 101):
        fitter = TrainPreprocessor(["f_a", "f_missing"], 100, 0, 100000)
        for start in range(0, 101, batch):
            fitter.update(table.slice(start, batch))
        states.append(fitter.finalize())
    assert all(s["scale"] == [1, 1] for s in states)
    assert len({s["fit_row_sha256"] for s in states}) == 1
    for s in states:
        np.testing.assert_array_equal(TrainPreprocessor.transform(table, s), np.zeros((101, 2)))


def test_exact_train_regime_quantiles_and_ties(tmp_path):
    table = fitting_table(
        {"f_spread": [1, 3, 9, None], "f_mid_return_1000ms": [-0.4, 0.2, None, 0.1]}
    )
    q = TrainRegimeQuantiles(tmp_path / "q.sqlite", 100, 0, 1000, [0.25, 0.5, 0.75])
    q.update(table.slice(0, 2))
    q.update(table.slice(2))
    state = q.finish()
    assert state["metrics"]["spread"]["thresholds"] == [2, 3, 6]
    np.testing.assert_allclose(
        state["metrics"]["absolute_return_1000ms"]["thresholds"], [0.15, 0.2, 0.3]
    )
    assert state["metrics"]["depth_10"]["thresholds"] == [None, None, None]
    bins = apply_regimes(table, state)
    assert bins["regime_spread"].to_pylist() == [0, 1, 3, None]
    assert bins["regime_depth_10"].to_pylist() == [None] * 4
    assert not (tmp_path / "q.sqlite").exists()


def test_future_values_and_future_endpoint_labels_cannot_change_fitted_train_artifacts(tmp_path):
    source = research_fixture(tmp_path / "source", n=160)
    config = SplitConfig(batch_rows=13)
    baseline = tmp_path / "baseline"
    a = build_split_dataset(source, baseline, config)
    cutoff = a["ranges"]["train"]["end_exclusive_ns"]
    table = pq.read_table(source / "features.parquet")
    future = table["timestamp_ns"].to_numpy() >= cutoff
    for name in table.column_names:
        if name.startswith("f_"):
            values = table[name].to_numpy().copy()
            values[future] = 999999999
            table = table.set_column(
                table.schema.get_field_index(name), name, pa.array(values, mask=np.isnan(values))
            )
    pq.write_table(table, source / "features.parquet")
    for h in a["horizons_ms"]:
        table = pq.read_table(source / f"labels_{h}ms.parquet")
        # Also perturb TRAIN decisions whose future endpoint escapes TRAIN.
        future = table[f"label_end_ns_{h}ms"].to_numpy() >= cutoff
        for name, value in ((f"y_return_{h}ms", 99.0), (f"y_class_{h}ms", 1)):
            values = table[name].to_numpy().copy()
            values[future] = value
            table = table.set_column(
                table.schema.get_field_index(name), name, pa.array(values, type=table[name].type)
            )
        pq.write_table(table, source / f"labels_{h}ms.parquet")
    refresh_manifest(source)
    altered = tmp_path / "altered"
    b = build_split_dataset(source, altered, config)
    for h in a["horizons_ms"]:
        for name in (f"preprocessor_{h}ms.json", f"regime_cutoffs_{h}ms.json"):
            assert (baseline / name).read_bytes() == (altered / name).read_bytes()
    assert (baseline / "training_report.json").read_bytes() == (
        altered / "training_report.json"
    ).read_bytes()
    assert a["input_manifest"]["sha256"] != b["input_manifest"]["sha256"]


def test_partition_batch_size_does_not_change_fitted_parameters_or_membership(tmp_path):
    source = research_fixture(tmp_path / "source", n=170)
    outputs = []
    for batch in (7, 53):
        out = tmp_path / f"batch{batch}"
        build_split_dataset(source, out, SplitConfig(batch_rows=batch), horizons_ms=[100, 500])
        outputs.append(out)
    for h in (100, 500):
        states = [json.loads((out / f"preprocessor_{h}ms.json").read_text()) for out in outputs]
        assert states[0]["fit_row_sha256"] == states[1]["fit_row_sha256"]
        for name in ("impute_mean", "scale", "variance_after_mean_imputation"):
            np.testing.assert_allclose(states[0][name], states[1][name], rtol=1e-9, atol=1e-9)
        for split in ("train", "validation", "test"):
            tables = [pq.read_table(out / split / f"labels_{h}ms.parquet") for out in outputs]
            assert tables[0].equals(tables[1])
        assert (outputs[0] / f"regime_cutoffs_{h}ms.json").read_bytes() == (
            outputs[1] / f"regime_cutoffs_{h}ms.json"
        ).read_bytes()
