import copy
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import test_baselines
import yaml

from market_microstructure.evaluation.research import run_research, validate_config
from market_microstructure.evaluation.research_analysis import (
    align_predictions,
    cluster_interval,
    confidence_analysis,
)
from market_microstructure.models.experiments import load_run, read_json
from market_microstructure.models.learned_config import LearnedConfig
from market_microstructure.models.learned_experiments import train_learned_run
from market_microstructure.processing.build_dataset import file_hash
from market_microstructure.processing.split_dataset import iter_model_batches

dataset = test_baselines.dataset


def predictions():
    return pa.table(
        {
            "target": [-1, 0, 1, 1],
            "prediction": [-1, 1, 1, 0],
            "future_return": [-0.001, 0, 0.002, 0.002],
            "p_down": [0.8, 0.1, 0.05, 0.1],
            "p_stable": [0.1, 0.2, 0.05, 0.8],
            "p_up": [0.1, 0.7, 0.9, 0.1],
        }
    )


def test_confidence_moves_coverage_and_zeros():
    rows = confidence_analysis(predictions(), [0, 0.8, 0.95])
    assert rows[0]["directional_signals"] == 3
    assert rows[0]["signal_frequency_per_eligible_observation"] == 0.75
    assert rows[0]["mean_signed_mid_move_bps"] == pytest.approx(10)
    assert rows[0]["positive_move_win_rate"] == pytest.approx(2 / 3)
    assert rows[0]["zero_move_fraction"] == pytest.approx(1 / 3)
    assert rows[1]["directional_signals"] == 2
    assert rows[1]["mean_signed_mid_move_bps"] == 15
    assert rows[2]["mean_signed_mid_move_bps"] is None
    assert rows[2]["metrics"]["status"] == "unavailable"


def test_confidence_uses_hard_prediction_probability_not_argmax():
    table = predictions().slice(0, 1)
    table = table.set_column(
        table.schema.get_field_index("prediction"), "prediction", pa.array([1])
    )
    assert confidence_analysis(table, [0.5])[0]["directional_signals"] == 0


def test_cluster_interval_preserves_session_dependence():
    # Perfect within-session dependence: resampling rows would produce a falsely narrow interval.
    values = np.repeat([0.0, 1.0], 100)
    sessions = np.repeat(["a", "b"], 100)
    result = cluster_interval(values, sessions, minimum_sessions=2, repeats=1000)
    assert result["lower_95"] == 0 and result["upper_95"] == 1
    assert result == cluster_interval(values, sessions, minimum_sessions=2, repeats=1000)
    assert cluster_interval(values, sessions)["status"] == "unavailable"
    with pytest.raises(ValueError):
        cluster_interval([np.nan], ["a"])


@pytest.mark.parametrize("model", ["xgboost", "mlp"])
def test_subset_reload_and_unused_feature_invariance(dataset, tmp_path, model):
    cfg = LearnedConfig(
        model=model, rounds=5, epochs=3, patience=2, timing_warmup=1, timing_repeats=2
    )
    names = ["f_obi_1"]
    run = tmp_path / "subset"
    train_learned_run(dataset, run, cfg, feature_columns=names)
    frozen, manifest = load_run(run)
    assert manifest["feature_columns"] == names
    assert len(frozen.preprocessor["feature_columns"]) > 1
    table = pa.concat_tables(
        list(iter_model_batches(dataset, 500, "validation", transformed=False))
    )
    _, before = frozen.predict(table)
    changed = table.set_column(
        table.schema.get_field_index("f_midprice"),
        "f_midprice",
        pa.array(np.full(len(table), -1e8)),
    )
    _, after = frozen.predict(changed)
    np.testing.assert_array_equal(before, after)
    saved = pq.read_table(run / "validation/predictions.parquet")
    np.testing.assert_array_equal(
        before, np.column_stack([saved[n].to_numpy() for n in ("p_down", "p_stable", "p_up")])
    )
    with pytest.raises(ValueError, match="subset"):
        train_learned_run(dataset, tmp_path / "bad", cfg, feature_columns=["f_obi_1", "f_obi_1"])


def config_for(dataset):
    cfg = yaml.safe_load(Path("configs/research.yaml").read_text(encoding="utf-8"))
    cfg.update(
        dataset_manifest_sha256=file_hash(dataset / "manifest.json"),
        horizons_ms=[500, 5000],
        models=["majority", "xgboost", "mlp"],
        ablation_horizons_ms=[500],
        feature_sets={"book": ["f_midprice"]},
        bootstrap_repeats=100,
    )
    cfg["model_parameters"] = {
        model: {"rounds": 3, "epochs": 2, "patience": 1, "timing_warmup": 1, "timing_repeats": 2}
        for model in ("xgboost", "mlp")
    }
    return cfg


@pytest.mark.parametrize(
    "key,value",
    [
        ("models", ["bad"]),
        ("dataset_manifest_sha256", "invalid"),
        ("dataset_manifest_sha256", 123),
        ("horizons_ms", [500, 500]),
        ("confidence_thresholds", [0.9, 0.5]),
        ("minimum_sessions", 1),
        ("max_analysis_rows", True),
        ("feature_sets", {"../bad": ["f_midprice"]}),
        ("feature_sets", {"bad": ["future_return"]}),
        ("ablation_horizons_ms", [100]),
    ],
)
def test_bad_research_config(dataset, key, value):
    config = config_for(dataset)
    config[key] = value
    with pytest.raises(ValueError):
        validate_config(config, read_json(dataset / "manifest.json"))


def test_research_regeneration_ignores_sealed_test(dataset, tmp_path):
    cfg = config_for(dataset)
    path = tmp_path / "research.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    # Fitting and diagnostics must not open TEST, even to hash it.
    for file in (dataset / "test").glob("*.parquet"):
        file.write_bytes(b"sealed and unreadable")
    first, second = tmp_path / "first", tmp_path / "second"
    for output in (first, second):
        if output == second:
            cfg["dataset_manifest_sha256"] = None
            path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        manifest = run_research(dataset, output, path)
        assert read_json(output / "resolved_config.json")["dataset_manifest_sha256"] == file_hash(
            dataset / "manifest.json"
        )
        assert manifest["completed_runs"] == 5
        assert manifest["unavailable_runs"] == 3
        assert manifest["test_accessed"] is False
        assert len(list((output / "figures").glob("*.png"))) == 9
        assert not list(output.rglob("*.md"))
        for name, info in manifest["outputs"].items():
            assert file_hash(output / name) == info["sha256"]
    a, b = read_json(first / "research.json"), read_json(second / "research.json")
    for left, right in zip(a["records"], b["records"], strict=True):
        if left["status"] == "complete":
            for field in ("metrics", "confidence", "regimes", "accuracy_uncertainty"):
                assert left[field] == right[field]
    assert a["comparisons"] == b["comparisons"]
    run = first / "runs/majority_500ms_all"
    predictions = pq.read_table(run / "validation/predictions.parquet")
    joined = pa.concat_tables(
        list(iter_model_batches(dataset, 500, "validation", transformed=False))
    )
    assert align_predictions(predictions, joined, 500).num_rows == len(predictions)
    changed = predictions.set_column(
        predictions.schema.get_field_index("future_return"),
        "future_return",
        pa.array(np.ones(len(predictions))),
    )
    with pytest.raises(ValueError, match="mismatch"):
        align_predictions(changed, joined, 500)
    wrong = copy.deepcopy(cfg)
    wrong["dataset_manifest_sha256"] = "0" * 64
    path.write_text(yaml.safe_dump(wrong), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        run_research(dataset, tmp_path / "wrong", path)
    assert not (tmp_path / "wrong").exists()
