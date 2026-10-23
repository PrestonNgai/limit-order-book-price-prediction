import json

import numpy as np
import pyarrow.parquet as pq
import pytest
import test_baselines

from market_microstructure.evaluation.regression import run_regression_comparison
from market_microstructure.models.experiments import read_json
from market_microstructure.models.regression import (
    RegressionConfig,
    evaluate_regression_run,
    load_regression_run,
    read_partition,
    regression_metrics,
    train_regression_run,
)
from market_microstructure.processing.build_dataset import file_hash

dataset = test_baselines.dataset


def test_regression_error_units_and_undefined_metrics():
    result = regression_metrics([0.0001, -0.0002], [0, 0])
    assert result["mae_bps"] == 1.5
    assert result["rmse_bps"] == pytest.approx(np.sqrt(2.5))
    assert result["mse_skill_vs_zero"] == 0
    assert result["pearson_correlation"] is None
    assert result["directional_accuracy"] is None
    assert regression_metrics([0, 0], [0, 0])["r2"] is None
    assert regression_metrics([], [])["status"] == "unavailable"
    with pytest.raises(ValueError):
        regression_metrics([np.nan], [0])


@pytest.mark.parametrize(
    "change",
    [
        {"model": "invalid"},
        {"ridge_alpha": 0},
        {"seed": True},
        {"max_rows": 0},
        {"learning_rate": float("inf")},
        {"horizon_ms": True},
    ],
)
def test_invalid_regression_config(change):
    with pytest.raises(ValueError):
        RegressionConfig(**change)


@pytest.mark.parametrize("model", ["zero", "ridge", "xgboost"])
def test_regression_sealed_test_reload_and_evaluation(dataset, tmp_path, model):
    test_path = dataset / "test/features.parquet"
    original = test_path.read_bytes()
    test_path.write_bytes(b"TEST must not be read during regression training")
    run = tmp_path / "regression"
    cfg = RegressionConfig(model=model, rounds=8, patience=2)
    manifest = train_regression_run(dataset, run, cfg)
    assert manifest["test_accessed"] is False
    assert manifest["train_rows"] > 0
    with pytest.raises(PermissionError):
        evaluate_regression_run(run, dataset, tmp_path / "forbidden", split="test")
    frozen, _ = load_regression_run(run)
    data_manifest = read_json(dataset / "manifest.json")
    _, matrix, _ = read_partition(dataset, data_manifest, frozen.preprocessor, cfg, "validation")
    saved = pq.read_table(run / "validation/predictions.parquet")
    np.testing.assert_allclose(
        frozen.predict(matrix), saved["prediction_return"].to_numpy(), rtol=0, atol=1e-15
    )
    if model == "zero":
        assert np.all(saved["prediction_return"].to_numpy() == 0)
    test_path.write_bytes(original)
    evaluated = evaluate_regression_run(
        run, dataset, tmp_path / "final", split="test", allow_test=True
    )
    assert evaluated["test_accessed"] and evaluated["rows"] > 0
    model_path = run / "model.json"
    model_path.write_text(model_path.read_text() + " ")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_regression_run(run)


def test_ridge_fit_is_unchanged_by_validation_targets(dataset, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    cfg = RegressionConfig(model="ridge")
    train_regression_run(dataset, first, cfg)
    path = dataset / "validation/labels_500ms.parquet"
    table = pq.read_table(path)
    import pyarrow as pa

    table = table.set_column(
        table.schema.get_field_index("y_return_500ms"),
        "y_return_500ms",
        pa.array(np.full(len(table), 0.5)),
    )
    pq.write_table(table, path)
    manifest_path = dataset / "manifest.json"
    manifest = read_json(manifest_path)
    manifest["outputs"]["validation/labels_500ms.parquet"]["sha256"] = file_hash(path)
    manifest_path.write_text(json.dumps(manifest))
    train_regression_run(dataset, second, cfg)
    assert read_json(first / "model.json") == read_json(second / "model.json")


def test_regression_resource_limit(dataset, tmp_path):
    with pytest.raises(ValueError, match="resource limits"):
        train_regression_run(dataset, tmp_path / "limited", RegressionConfig(max_rows=1))


def test_regression_comparison_tracks_missing_horizons(dataset, tmp_path):
    output = tmp_path / "regression-comparison"
    result = run_regression_comparison(
        dataset, output, RegressionConfig(rounds=3, patience=1), horizons=[500, 5000]
    )
    assert result["completed_runs"] == 3
    records = read_json(output / "research.json")["records"]
    assert len(records) == 6 and sum(r["status"] == "unavailable" for r in records) == 3
    assert (output / "figures/regression.png").exists()
    with pytest.raises(ValueError, match="horizons"):
        run_regression_comparison(dataset, tmp_path / "bad", RegressionConfig(), horizons=[7])
