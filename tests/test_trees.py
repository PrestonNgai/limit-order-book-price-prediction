import json
from dataclasses import replace

import numpy as np
import pyarrow as pa
import pytest
import test_baselines

from market_microstructure.models.experiments import evaluate_run, load_run, read_json, train_run
from market_microstructure.models.learned_config import LearnedConfig
from market_microstructure.models.trees import contributions, fit_tree, importance, predict_tree
from market_microstructure.processing.split_dataset import iter_model_batches

dataset = test_baselines.dataset


def fast_config(**kwargs):
    return LearnedConfig(rounds=20, patience=3, timing_warmup=1, timing_repeats=2, **kwargs)


@pytest.mark.parametrize(
    "settings",
    [
        {"model": "unknown"},
        {"max_depth": 0},
        {"patience": 0},
        {"probability_mix": 1},
        {"dropout": 1},
        {"hidden_sizes": []},
        {"hidden_sizes": [True]},
        {"learning_rate": float("nan")},
        {"explain_rows": 1001},
        {"epochs": True},
        {"seed": -1},
        {"min_delta": -1},
    ],
)
def test_learned_config_validation(settings):
    with pytest.raises(ValueError):
        LearnedConfig(**settings)


def test_tree_round_selection_importance_and_native_shap():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(90, 4)).astype(np.float32)
    y = np.where(x[:, 0] > 0.3, 1, np.where(x[:, 0] < -0.3, -1, 0)).astype(np.int8)
    config = fast_config()
    names = [f"f_{i}" for i in range(4)]
    booster, report = fit_tree(x[:60], y[:60], x[60:], y[60:], names, config)
    history = report["history"]["validation"]["mlogloss"]
    assert report["selected_rounds"] == np.argmin(history) + 1
    assert booster.num_boosted_rounds() == report["selected_rounds"]
    values = importance(booster, names)
    assert [row["feature"] for row in values["rows"]] == names
    assert values["rows"][0]["total_gain"] > 0
    explained = contributions(booster, x[60:63], names)
    assert np.asarray(explained["values"]).shape == (3, 3, 5)
    assert explained["max_additivity_error"] < 1e-5


@pytest.mark.parametrize("classes", [[0], [-1, 1], [-1, 0, 1]])
def test_tree_fixed_class_order_with_absent_train_classes(classes):
    x = np.arange(60, dtype=np.float32).reshape(20, 3) / 10
    y = np.resize(np.array(classes, dtype=np.int8), 20)
    model, report = fit_tree(x, y, x[:0], y[:0], ["f_a", "f_b", "f_c"], fast_config())
    p = predict_tree(model, x, ["f_a", "f_b", "f_c"])
    assert p.shape == (20, 3)
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-6)
    assert report["selection"] == "fixed_rounds_no_validation"
    assert report["selected_rounds"] == 20


def test_tree_run_reload_guard_and_comparison_rows(dataset, tmp_path):
    config = fast_config(explain_rows=5)
    run = tmp_path / "tree"
    result = train_run(dataset, run, config)
    frozen, _ = load_run(run)
    validation = pa.concat_tables(
        list(iter_model_batches(dataset, 500, "validation", transformed=False))
    )
    predictions, probabilities = frozen.predict(validation)
    assert len(predictions) == result["validation_rows"]
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-12)
    assert result["fit"]["reload_max_abs_error"] == 0
    assert (
        read_json(run / "shap_validation.json")["row_ids"] == validation["row_id"][:5].to_pylist()
    )
    assert (
        read_json(run / "feature_importance.json")["feature_columns"]
        == frozen.state["feature_columns"]
    )
    evaluate_run(run, dataset, tmp_path / "reload")
    saved = read_json(run / "validation/metrics.json")
    reloaded = read_json(tmp_path / "reload/validation/metrics.json")
    assert saved["row_sha256"] == reloaded["row_sha256"]
    assert saved["log_loss"] == reloaded["log_loss"]
    with pytest.raises(PermissionError):
        evaluate_run(run, dataset, tmp_path / "forbidden", split="test")
    with (run / "tree.ubj").open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        load_run(run)


def test_tree_test_changes_do_not_change_fitted_checkpoint(dataset, tmp_path, monkeypatch):
    import market_microstructure.models.learned_experiments as module

    calls = []
    original = module.iter_model_batches

    def guard(*args, **kwargs):
        calls.append(args[2])
        assert args[2] in ("train", "validation")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "iter_model_batches", guard)
    config = fast_config()
    train_run(dataset, tmp_path / "first", config)
    # Corrupt only a synthetic TEST file. It must not be opened by fitting/selection.
    (dataset / "test/features.parquet").write_bytes(b"TEST MUST REMAIN UNREAD")
    train_run(dataset, tmp_path / "second", config)
    assert (tmp_path / "first/tree.ubj").read_bytes() == (tmp_path / "second/tree.ubj").read_bytes()
    assert calls == ["train", "validation", "train", "validation"]


def test_tree_empty_validation_and_limits(dataset, tmp_path):
    config = fast_config(horizon_ms=5000)
    result = train_run(dataset, tmp_path / "empty", config)
    assert result["validation_rows"] == 0
    assert result["fit"]["selection"] == "fixed_rounds_no_validation"
    assert read_json(tmp_path / "empty/validation/metrics.json")["status"] == "unavailable"
    with pytest.raises(ValueError, match="row limit"):
        train_run(dataset, tmp_path / "limit", replace(config, max_train_rows=1))


def test_tree_validation_only_selection_has_separate_training_history():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(90, 3)).astype(np.float32)
    y = np.where(x[:, 0] > 0.2, 1, np.where(x[:, 0] < -0.2, -1, 0)).astype(np.int8)
    _, first = fit_tree(x[:60], y[:60], x[60:], y[60:], ["f_a", "f_b", "f_c"], fast_config())
    _, second = fit_tree(x[:60], y[:60], x[60:], -y[60:], ["f_a", "f_b", "f_c"], fast_config())
    a, b = first["history"]["train"]["mlogloss"], second["history"]["train"]["mlogloss"]
    np.testing.assert_array_equal(a[: min(len(a), len(b))], b[: min(len(a), len(b))])
    assert first["history"]["validation"] != second["history"]["validation"]


def test_learned_preprocessing_provenance_and_memory_guard(dataset, tmp_path):
    config = fast_config()
    manifest_path = dataset / "manifest.json"
    manifest = read_json(manifest_path)
    manifest["target_counts"]["500ms"]["train"]["rows"] = 10000
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="max_fit_mb"):
        train_run(dataset, tmp_path / "limit", replace(config, max_fit_mb=1))
