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

from market_microstructure.evaluation.metrics import classification_metrics
from market_microstructure.models.baselines import (
    fit_logistic,
    imbalance_sign,
    logistic_probabilities,
    majority_class,
)
from market_microstructure.models.config import BaselineConfig, load_baseline_config
from market_microstructure.models.experiments import (
    evaluate_run,
    load_run,
    read_json,
    train_run,
)
from market_microstructure.processing.build_dataset import file_hash
from market_microstructure.processing.split_config import SplitConfig
from market_microstructure.processing.split_dataset import build_split_dataset, iter_model_batches


@pytest.fixture
def dataset(tmp_path):
    # Synthetic labels exercise all classes and both imbalance directions.
    source = research_fixture(tmp_path / "research", n=180, horizons=(500, 5000))
    f = pq.read_table(source / "features.parquet")
    signal = np.sin(np.arange(len(f)) / 4)
    f = f.set_column(f.schema.get_field_index("f_obi_1"), "f_obi_1", pa.array(signal))
    pq.write_table(f, source / "features.parquet")
    for h in (500, 5000):
        path = source / f"labels_{h}ms.parquet"
        labels = pq.read_table(path)
        classes = np.where(
            signal[: len(labels)] > 0.4, 1, np.where(signal[: len(labels)] < -0.4, -1, 0)
        ).astype(np.int8)
        labels = labels.set_column(
            labels.schema.get_field_index(f"y_class_{h}ms"), f"y_class_{h}ms", pa.array(classes)
        )
        labels = labels.set_column(
            labels.schema.get_field_index(f"y_return_{h}ms"),
            f"y_return_{h}ms",
            pa.array(classes * 0.001),
        )
        pq.write_table(labels, path)
    refresh_manifest(source)
    result = tmp_path / "splits"
    build_split_dataset(source, result, SplitConfig(batch_rows=17))
    return result


@pytest.mark.parametrize(
    "changes",
    [
        {"model": "xgb"},
        {"smoothing": 0},
        {"probability_mix": 1},
        {"imbalance_deadband": -1},
        {"logistic_c": float("nan")},
        {"horizon_ms": True},
        {"max_train_rows": 0},
        {"imbalance_feature": "y_class_500ms"},
        {"seed": -1},
        {"tolerance": True},
        {"batch_rows": 1.5},
    ],
)
def test_config_rejects_bad_settings(changes):
    with pytest.raises(ValueError):
        BaselineConfig(**changes)


def test_config_unknown_key(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("future_feature: yes", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        load_baseline_config(path)


def test_majority_ties_and_raw_imbalance():
    assert majority_class([4, 4, 4]) == 0
    assert majority_class([4, 0, 4]) == -1
    config = BaselineConfig(imbalance_deadband=0.1).to_dict()
    table = pa.table({"f_obi_1": [-0.4, -0.1, 0.0, 0.1, 0.5, None]})
    assert imbalance_sign(table, config).tolist() == [-1, 0, 0, 0, 1, 0]
    with pytest.raises(ValueError, match="raw"):
        imbalance_sign(pa.table({"f_obi_1": [1.1]}), config)


@pytest.mark.parametrize("classes", [[-1], [-1, 1], [-1, 0, 1]])
def test_logistic_export_and_absent_class_probabilities(classes):
    rng = np.random.default_rng(5)
    x = rng.normal(size=(60, 3))
    y = np.resize(np.array(classes), 60)
    state, report = fit_logistic(x, y, BaselineConfig())
    reloaded = json.loads(json.dumps(state))
    probabilities = logistic_probabilities(x, reloaded)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    assert report["reload_max_abs_error"] < 1e-12
    for missing in set([-1, 0, 1]) - set(classes):
        np.testing.assert_allclose(probabilities[:, missing + 1], 1e-6 / 3)
    assert report["status"] == ("single_class_constant" if len(classes) == 1 else "converged")


def test_known_metrics_and_undefined_classes():
    # DOWN correct once; STABLE misclassified once; UP absent from truth and predictions.
    metric = classification_metrics(
        np.array([-1, 0]), np.array([-1, -1]), np.array([[0.8, 0.1, 0.1], [0.6, 0.3, 0.1]])
    )
    assert metric["accuracy"] == 0.5
    assert metric["balanced_accuracy_present_classes"] == 0.5
    assert metric["log_loss"] == pytest.approx(-np.log([0.8, 0.3]).mean())
    assert metric["per_class"]["UP"]["recall"] is None
    assert metric["per_class"]["UP"]["f1"] is None
    assert metric["per_class"]["STABLE"]["precision"] is None
    assert metric["per_class"]["DOWN"]["f1"] == pytest.approx(2 / 3)
    assert metric["macro_zero_filled_three_classes"]["f1"] == pytest.approx(2 / 9)
    assert metric["per_class"]["DOWN"]["roc_auc_ovr"] == 1
    assert metric["roc_auc_ovr_macro"] is None
    assert metric["directional_accuracy"] == 0.5
    assert metric["directional_coverage"] == 1
    assert metric["confusion_matrix_actual_rows_predicted_columns"] == [
        [1, 0, 0],
        [1, 0, 0],
        [0, 0, 0],
    ]
    only = classification_metrics([0, 0], [0, 0], [[0.1, 0.8, 0.1]] * 2)
    assert only["directional_accuracy"] is None
    assert all(v["roc_auc_ovr"] is None for v in only["per_class"].values())
    assert classification_metrics([], [], np.empty((0, 3)))["status"] == "unavailable"


@pytest.mark.parametrize(
    "target,pred,p",
    [
        ([2], [0], [[0.1, 0.8, 0.1]]),
        ([0], [0], [[0, 1, 0]]),
        ([0], [0], [[0.2, 0.8, 0.2]]),
        ([0], [0], [[float("nan"), 0.9, 0.1]]),
    ],
)
def test_invalid_metrics_rejected(target, pred, p):
    with pytest.raises(ValueError):
        classification_metrics(target, pred, p)


@pytest.mark.parametrize("model", ["majority", "imbalance", "logistic"])
def test_fit_reload_and_evaluate_identical_rows(dataset, tmp_path, model):
    run = tmp_path / model
    config = BaselineConfig(model=model, batch_rows=11)
    trained = train_run(dataset, run, config)
    frozen, manifest = load_run(run)
    assert trained["test_accessed"] is False
    assert manifest["fit"]["class_counts"] == frozen.state["train_class_counts"]
    tables = list(iter_model_batches(dataset, 500, "validation", transformed=False, batch_rows=7))
    table = pa.concat_tables(tables)
    pred, p = frozen.predict(table)
    saved = pq.read_table(run / "validation/predictions.parquet")
    np.testing.assert_array_equal(pred, saved["prediction"].to_numpy())
    np.testing.assert_allclose(
        p,
        np.column_stack([saved[n].to_numpy() for n in ("p_down", "p_stable", "p_up")]),
        rtol=1e-12,
        atol=1e-14,
    )
    result = evaluate_run(run, dataset, tmp_path / "evaluation")
    assert result["rows"] == len(saved)
    reloaded = pq.read_table(tmp_path / "evaluation/validation/predictions.parquet")
    assert saved.equals(reloaded)
    if model == "majority":
        np.testing.assert_allclose(
            frozen.state["probabilities"],
            (np.array(trained["fit"]["class_counts"]) + 1) / (trained["fit"]["rows"] + 3),
        )
    if model == "imbalance":
        # Hard predictions remain the sign rule even if conditional probabilities disagree.
        assert pred.tolist() == imbalance_sign(table, config.to_dict()).tolist()
        conditional = np.asarray(frozen.state["conditional_counts"])
        prior = (np.array(trained["fit"]["class_counts"]) + 1) / (trained["fit"]["rows"] + 3)
        np.testing.assert_allclose(
            frozen.state["conditional_probabilities"],
            (conditional + prior) / (conditional.sum(axis=1, keepdims=True) + 1),
        )
    with pytest.raises(FileExistsError):
        train_run(dataset, run, config)


def test_all_models_share_target_fingerprint(dataset, tmp_path):
    fingerprints = []
    for name in ("majority", "imbalance", "logistic"):
        train_run(dataset, tmp_path / name, BaselineConfig(model=name))
        fingerprints.append(read_json(tmp_path / name / "validation/metrics.json")["row_sha256"])
    assert len(set(fingerprints)) == 1


def test_training_never_loads_test(dataset, tmp_path, monkeypatch):
    import market_microstructure.models.experiments as module

    calls = []
    original = module.iter_model_batches

    def guard(*args, **kwargs):
        calls.append(args[2])
        assert args[2] != "test"
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "iter_model_batches", guard)
    train_run(dataset, tmp_path / "run", BaselineConfig())
    assert calls == ["train", "validation"]


@pytest.mark.parametrize("model", ["majority", "imbalance", "logistic"])
def test_future_changes_cannot_change_fitted_model(dataset, tmp_path, model):
    config = BaselineConfig(model=model)
    first, second = tmp_path / "first", tmp_path / "second"
    train_run(dataset, first, config)
    manifest_path = dataset / "manifest.json"
    manifest = read_json(manifest_path)
    for split in ("validation", "test"):
        path = dataset / split / "features.parquet"
        table = pq.read_table(path)
        for name in manifest["feature_columns"]:
            # Preserve the raw imbalance domain while changing every future feature.
            values = np.full(len(table), -0.7 if name.startswith("f_obi") else 123456.0)
            table = table.set_column(table.schema.get_field_index(name), name, pa.array(values))
        pq.write_table(table, path)
        name = f"{split}/features.parquet"
        manifest["outputs"][name]["sha256"] = file_hash(path)
        manifest["outputs"][name]["bytes"] = path.stat().st_size
        path = dataset / split / "labels_500ms.parquet"
        table = pq.read_table(path)
        table = table.set_column(
            table.schema.get_field_index("y_class_500ms"),
            "y_class_500ms",
            pa.array(np.full(len(table), -1, np.int8)),
        )
        pq.write_table(table, path)
        manifest["outputs"][f"{split}/labels_500ms.parquet"]["sha256"] = file_hash(path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    train_run(dataset, second, config)
    assert (first / "model.json").read_bytes() == (second / "model.json").read_bytes()
    assert (first / "preprocessor.json").read_bytes() == (second / "preprocessor.json").read_bytes()
    with pytest.raises(ValueError, match="differs"):
        evaluate_run(first, dataset, tmp_path / "wrong_dataset")


def test_guard_and_synthetic_explicit_test_evaluation(dataset, tmp_path):
    run = tmp_path / "run"
    train_run(dataset, run, BaselineConfig(model="majority"))
    with pytest.raises(PermissionError, match="sealed"):
        evaluate_run(run, dataset, tmp_path / "forbidden", split="test")
    assert not (tmp_path / "forbidden").exists()
    result = evaluate_run(run, dataset, tmp_path / "allowed", split="test", allow_test=True)
    assert result["test_accessed"] is True
    assert result["rows"] > 0


@pytest.mark.parametrize("artifact", ["model.json", "preprocessor.json", "config.json"])
def test_run_hash_tampering_rejected(dataset, tmp_path, artifact):
    run = tmp_path / "run"
    train_run(dataset, run, BaselineConfig(model="majority"))
    with (run / artifact).open("a", encoding="utf-8") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="hash"):
        load_run(run)


def test_limits_and_empty_validation(dataset, tmp_path):
    with pytest.raises(ValueError, match="max_train_rows"):
        train_run(dataset, tmp_path / "limit_rows", BaselineConfig(max_train_rows=1))
    with pytest.raises(ValueError, match="max_eval_rows"):
        train_run(dataset, tmp_path / "limit_eval", BaselineConfig(max_eval_rows=1))
    assert (tmp_path / "limit_eval/_INCOMPLETE").exists()
    assert not (tmp_path / "limit_eval/manifest.json").exists()
    run = tmp_path / "empty_validation"
    train_run(dataset, run, BaselineConfig(model="majority", horizon_ms=5000))
    assert read_json(run / "validation/metrics.json")["status"] == "unavailable"
    assert pq.read_table(run / "validation/predictions.parquet").num_rows == 0


def test_training_preprocessor_provenance_enforced(dataset, tmp_path):
    path = dataset / "train/features.parquet"
    table = pq.read_table(path)
    table = table.set_column(
        table.schema.get_field_index("f_spread"), "f_spread", pa.array(np.full(len(table), 99.0))
    )
    pq.write_table(table, path)
    manifest = read_json(dataset / "manifest.json")
    manifest["outputs"]["train/features.parquet"]["sha256"] = file_hash(path)
    (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        train_run(dataset, tmp_path / "run", BaselineConfig(model="majority"))


def test_nonconvergence_leaves_uncommitted_run(dataset, tmp_path):
    from sklearn.exceptions import ConvergenceWarning

    with pytest.raises(ConvergenceWarning):
        train_run(dataset, tmp_path / "failed", BaselineConfig(max_iter=1))
    assert (tmp_path / "failed/_INCOMPLETE").exists()
    assert not (tmp_path / "failed/manifest.json").exists()


def test_raw_to_split_to_fit_reload_cli(tmp_path):
    raw_result, _ = pipeline(tmp_path, n=120)
    split = tmp_path / "splits"
    build_split_dataset(raw_result / "research", split, SplitConfig(batch_rows=13))
    root = Path(__file__).resolve().parents[1]
    run = tmp_path / "cli_run"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/train.py"),
            "--dataset",
            str(split),
            "--output",
            str(run),
            "--model",
            "logistic",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert read_json(run / "fit_report.json")["status"] == "single_class_constant"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/evaluate.py"),
            "--dataset",
            str(split),
            "--run",
            str(run),
            "--output",
            str(tmp_path / "reloaded"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert pq.read_table(run / "validation/predictions.parquet").equals(
        pq.read_table(tmp_path / "reloaded/validation/predictions.parquet")
    )


def test_logistic_memory_limit_precedes_allocation(tmp_path):
    source = research_fixture(tmp_path / "large", n=600, horizons=(500,))
    split = tmp_path / "splits"
    build_split_dataset(source, split, SplitConfig())
    with pytest.raises(ValueError, match="max_fit_mb"):
        train_run(split, tmp_path / "run", BaselineConfig(max_fit_mb=1))
    assert not (tmp_path / "run").exists()
