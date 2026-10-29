import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import test_baselines
import yaml
from split_helpers import refresh_manifest, research_fixture

from market_microstructure.evaluation.tuning import (
    candidates,
    expanding_folds,
    fold_preprocessing,
    run_tuning,
)
from market_microstructure.models.experiments import evaluate_run, load_run, read_json, train_run
from market_microstructure.models.tuned import FrozenTuned, TunedConfig, feature_names
from market_microstructure.processing.split_config import SplitConfig
from market_microstructure.processing.split_dataset import build_split_dataset, iter_model_batches

dataset = test_baselines.dataset


@pytest.fixture
def sessions(tmp_path):
    source = research_fixture(
        tmp_path / "source", periods=[(100, i * 20000, f"s{i}") for i in range(6)], horizons=(500,)
    )
    features = pq.read_table(source / "features.parquet")
    signal = np.sin(np.arange(len(features)) / 4)
    features = features.set_column(
        features.schema.get_field_index("f_obi_1"), "f_obi_1", pa.array(signal)
    )
    pq.write_table(features, source / "features.parquet")
    labels = pq.read_table(source / "labels_500ms.parquet")
    index = np.searchsorted(features["timestamp_ns"].to_numpy(), labels["timestamp_ns"].to_numpy())
    target = np.where(signal[index] > 0.4, 1, np.where(signal[index] < -0.4, -1, 0)).astype(np.int8)
    labels = labels.set_column(
        labels.schema.get_field_index("y_class_500ms"), "y_class_500ms", pa.array(target)
    )
    labels = labels.set_column(
        labels.schema.get_field_index("y_return_500ms"), "y_return_500ms", pa.array(target * 0.001)
    )
    pq.write_table(labels, source / "labels_500ms.parquet")
    refresh_manifest(source)
    path = tmp_path / "session-splits"
    build_split_dataset(
        source,
        path,
        SplitConfig(mode="sessions", train_fraction="0.67", validation_fraction="0.16"),
    )
    return path


@pytest.mark.parametrize("family", ["logistic", "xgboost"])
@pytest.mark.parametrize("weight", [None, "balanced"])
def test_tuned_reload_and_test_guard(dataset, tmp_path, family, weight):
    run = tmp_path / "tuned"
    test_path = dataset / "test/features.parquet"
    original = test_path.read_bytes()
    test_path.write_bytes(b"unreadable TEST during fitting")
    result = train_run(
        dataset,
        run,
        TunedConfig(family=family, feature_set="all", class_weight=weight, rounds=5, patience=2),
    )
    assert not result["test_accessed"]
    model, _ = load_run(run)
    table = pa.concat_tables(
        list(iter_model_batches(dataset, 500, "validation", transformed=False))
    )
    pred, p = model.predict(table)
    saved = pq.read_table(run / "validation/predictions.parquet")
    np.testing.assert_array_equal(pred, saved["prediction"].to_numpy())
    np.testing.assert_allclose(
        p,
        np.column_stack([saved[n].to_numpy() for n in ("p_down", "p_stable", "p_up")]),
        rtol=0,
        atol=1e-14,
    )
    with pytest.raises(PermissionError):
        evaluate_run(run, dataset, tmp_path / "forbidden", split="test")
    test_path.write_bytes(original)
    assert evaluate_run(run, dataset, tmp_path / "allowed", split="test", allow_test=True)[
        "test_accessed"
    ]
    (run / "model.json").write_text((run / "model.json").read_text() + " ")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_run(run)


def test_class_cost_probability_correction_preserves_decision():
    config = TunedConfig(class_weight="balanced", feature_set="all")
    state = {
        "config": config.to_dict(),
        "feature_columns": ["f_obi_1"],
        "prior": [0.1, 0.8, 0.1],
        "coef": [[0], [0], [0]],
        "intercept": np.log([0.4, 0.3, 0.3]).tolist(),
    }
    frozen = FrozenTuned(state, {"feature_columns": ["f_obi_1"], "impute_mean": [0], "scale": [1]})
    pred, p = frozen.predict_matrix(np.zeros((1, 1)))
    assert pred[0] == -1 and p.argmax(axis=1)[0] == 1
    expected = np.array([0.04, 0.24, 0.03]) / 0.31
    np.testing.assert_allclose(p[0], (1 - 1e-6) * expected + 1e-6 / 3)


@pytest.mark.parametrize(
    "settings",
    [
        {"family": "bad"},
        {"class_weight": "test"},
        {"feature_set": "future"},
        {"logistic_c": 0},
        {"seed": True},
        {"max_depth": True},
    ],
)
def test_tuned_config_validation(settings):
    with pytest.raises(ValueError):
        TunedConfig(**settings)


def test_features_remove_absolute_price_levels():
    names = ["f_midprice", "f_microprice", "f_trade_vwap_100ms", "f_obi_1"]
    assert feature_names(names, "stationary") == ["f_obi_1"]
    with pytest.raises(ValueError):
        feature_names(names, "compact")


def test_walk_forward_train_only_preprocessing(sessions):
    table = pa.concat_tables(list(iter_model_batches(sessions, 500, "train", transformed=False)))
    manifest = read_json(sessions / "manifest.json")
    folds = list(expanding_folds(table, 500, manifest["ranges"]["train"]["end_exclusive_ns"]))
    assert len(folds) == 3
    for _, train, validation, start, _ in folds:
        assert table.filter(train)["label_end_ns_500ms"].to_numpy().max() < start
        assert not np.any(train & validation)
    _, train, validation, start, _ = folds[0]
    before = fold_preprocessing(table, 500, train, validation, start)[2]
    values = table["f_obi_1"].to_numpy().copy()
    values[~train] += 1000
    changed = table.set_column(table.schema.get_field_index("f_obi_1"), "f_obi_1", pa.array(values))
    after = fold_preprocessing(changed, 500, train, validation, start)[2]
    assert before == after


def test_tuning_never_reads_validation_or_test(sessions, tmp_path):
    for split in ("validation", "test"):
        (sessions / split / "features.parquet").write_bytes(b"must remain unreadable")
    config = yaml.safe_load(Path("configs/tuning.yaml").read_text())
    config.update(
        feature_sets=["all"],
        logistic_c=[0.1],
        tree_depths=[2],
        rounds=3,
        patience=1,
        max_candidates=4,
    )
    path = tmp_path / "search.yaml"
    path.write_text(yaml.safe_dump(config))
    result = run_tuning(sessions, tmp_path / "search", path)
    assert result["candidate_count"] == 4 and result["fold_count"] == 3
    assert not result["test_accessed"] and not result["validation_partition_accessed"]
    scores = read_json(tmp_path / "search/candidates.json")
    assert result["selected_id"] == scores[0]["id"]
    assert all(len(r["folds"]) == 3 for r in scores)
    assert all(r["mean_balanced_accuracy"] <= scores[0]["mean_balanced_accuracy"] for r in scores)
    assert json.loads((tmp_path / "search/selected_config.json").read_text())["model"] == "tuned"


def test_bounded_grid():
    config = yaml.safe_load(Path("configs/tuning.yaml").read_text())
    assert len(candidates(config)) == 40
    config["max_candidates"] = 1
    with pytest.raises(ValueError, match="exceeds"):
        candidates(config)


def test_fold_matrix_follows_explicit_feature_order(sessions):
    table = pa.concat_tables(list(iter_model_batches(sessions, 500, "train", transformed=False)))
    manifest = read_json(sessions / "manifest.json")
    _, train, validation, start, _ = next(
        expanding_folds(table, 500, manifest["ranges"]["train"]["end_exclusive_ns"])
    )
    names = list(reversed(manifest["feature_columns"]))
    result = fold_preprocessing(table, 500, train, validation, start, names=names)
    assert result[2]["feature_columns"] == names
    from market_microstructure.processing.preprocessing import TrainPreprocessor

    expected = TrainPreprocessor.transform(table.filter(validation), result[2])
    np.testing.assert_array_equal(result[5], expected)
    normal = fold_preprocessing(table, 500, train, validation, start, names=list(reversed(names)))
    np.testing.assert_array_equal(result[5], normal[5][:, ::-1])


@pytest.mark.parametrize("family", ["logistic", "xgboost"])
def test_selected_features_do_not_read_unused_columns(dataset, tmp_path, family):
    run = tmp_path / "small"
    train_run(dataset, run, TunedConfig(family=family, feature_set="l1", rounds=3, patience=1))
    frozen, _ = load_run(run)
    table = pa.concat_tables(
        list(iter_model_batches(dataset, 500, "validation", transformed=False))
    )
    expected, p = frozen.predict(table)
    only_inputs = table.select(frozen.state["feature_columns"])
    actual, q = frozen.predict(only_inputs)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(p, q)
