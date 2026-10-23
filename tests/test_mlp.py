import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import test_baselines
import torch
from test_research_dataset import pipeline

from market_microstructure.models.experiments import evaluate_run, load_run, read_json, train_run
from market_microstructure.models.learned_config import LearnedConfig
from market_microstructure.models.mlp import (
    SmallMLP,
    SnapshotDataset,
    cpu_runtime,
    fit_mlp,
    load_mlp,
    predict_mlp,
    save_mlp,
)
from market_microstructure.processing.split_config import SplitConfig
from market_microstructure.processing.split_dataset import build_split_dataset, iter_model_batches

dataset = test_baselines.dataset


def fast_config(**kwargs):
    return LearnedConfig(
        model="mlp",
        epochs=8,
        patience=3,
        hidden_sizes=(16,),
        timing_warmup=1,
        timing_repeats=2,
        **kwargs,
    )


def test_tiny_training_subset_overfit_and_reload(tmp_path):
    rng = np.random.default_rng(19)
    labels = np.tile(np.array([-1, 0, 1], np.int8), 8)
    x = (np.eye(3)[labels + 1] + rng.normal(0, 0.02, (24, 3))).astype(np.float32)
    config = replace(
        fast_config(),
        epochs=120,
        learning_rate=0.03,
        dropout=0.0,
        weight_decay=0.0,
        train_batch_size=24,
    )
    model, report = fit_mlp(x, labels, x[:0], labels[:0], config)
    p = predict_mlp(model, x)
    assert (p.argmax(axis=1) - 1 == labels).all()
    assert -np.log(p[np.arange(24), labels + 1]).mean() < 0.01
    assert not model.training
    assert report["selection"] == "fixed_epochs_no_validation"
    assert report["parameter_count"] == (3 * 16 + 16) + (16 * 3 + 3)
    path = tmp_path / "weights.pt"
    save_mlp(model, path)
    np.testing.assert_array_equal(p, predict_mlp(load_mlp(path, 3, config), x))


def test_dropout_modes_and_runtime_restoration():
    with cpu_runtime(42):
        model = SmallMLP(4, (32,), 0.5)
        x = torch.ones((20, 4))
        model.train()
        first, second = model(x), model(x)
        assert not torch.equal(first, second)
        model.eval()
        assert torch.equal(model(x), model(x))
    threads = torch.get_num_threads()
    state = torch.random.get_rng_state().clone()
    numpy_state = np.random.get_state()
    predict_mlp(model, np.ones((2, 4), np.float32))
    assert torch.get_num_threads() == threads
    assert torch.equal(torch.random.get_rng_state(), state)
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert not model.training


@pytest.mark.parametrize("classes", [[0], [-1, 1], [-1, 0, 1]])
def test_mlp_three_outputs_and_seed_repeatability(classes):
    x = np.arange(36, dtype=np.float32).reshape(12, 3) / 36
    y = np.resize(np.array(classes, np.int8), 12)
    config = fast_config()
    first, a = fit_mlp(x, y, x[:3], y[:3], config)
    second, b = fit_mlp(x, y, x[:3], y[:3], config)
    assert a["history"] == b["history"]
    p = predict_mlp(first, x)
    np.testing.assert_array_equal(p, predict_mlp(second, x))
    assert p.shape == (12, 3)
    assert np.isfinite(p).all()
    losses = [row["validation_loss"] for row in a["history"]]
    assert a["selected_epoch"] == np.argmin(losses) + 1
    expected = -np.log(p[:3][np.arange(3), y[:3] + 1]).mean()
    assert expected == pytest.approx(a["best_validation_loss"], abs=2e-7)


def test_dataset_rejects_invalid_values_and_maps_labels():
    dataset = SnapshotDataset(np.ones((3, 2), np.float32), np.array([-1, 0, 1]))
    assert [dataset[i][1].item() for i in range(3)] == [0, 1, 2]
    assert dataset[0][0].dtype == torch.float32
    with pytest.raises(ValueError):
        SnapshotDataset(np.array([[np.inf]]), np.array([0]))
    with pytest.raises(ValueError):
        SnapshotDataset(np.ones((3, 2)), np.array([0]))


def test_mlp_frozen_run_evaluation_timing_and_integrity(dataset, tmp_path):
    run = tmp_path / "mlp"
    result = train_run(dataset, run, fast_config())
    frozen, _ = load_run(run)
    assert not frozen.estimator.training
    assert result["fit"]["reload_max_abs_error"] == 0
    timing = read_json(run / "inference_timing.json")
    assert timing["batch"]["rows_per_call"] == result["validation_rows"]
    assert timing["single"]["rows_per_call"] == 1
    assert timing["batch"]["median_call_ns"] > 0
    assert timing["thread_limit"] == 1
    evaluate_run(run, dataset, tmp_path / "evaluation")
    assert pq.read_table(run / "validation/predictions.parquet").equals(
        pq.read_table(tmp_path / "evaluation/validation/predictions.parquet")
    )
    with pytest.raises(PermissionError):
        evaluate_run(run, dataset, tmp_path / "forbidden", split="test")
    with (run / "weights.pt").open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="hash"):
        load_run(run)


def test_mlp_test_changes_do_not_change_weights(dataset, tmp_path, monkeypatch):
    import market_microstructure.models.learned_experiments as module

    original, calls = module.iter_model_batches, []

    def guard(*args, **kwargs):
        assert args[2] in ("train", "validation")
        calls.append(args[2])
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "iter_model_batches", guard)
    config = fast_config()
    train_run(dataset, tmp_path / "first", config)
    (dataset / "test/features.parquet").write_bytes(b"NEVER READ TEST FOR SELECTION")
    train_run(dataset, tmp_path / "second", config)
    first, _ = load_run(tmp_path / "first")
    second, _ = load_run(tmp_path / "second")
    for name, value in first.estimator.state_dict().items():
        assert torch.equal(value, second.estimator.state_dict()[name])
    assert calls == ["train", "validation", "train", "validation"]


def test_mlp_same_eligible_rows_as_baseline(dataset, tmp_path):
    from market_microstructure.models.config import BaselineConfig

    train_run(dataset, tmp_path / "baseline", BaselineConfig(model="majority"))
    train_run(dataset, tmp_path / "mlp", fast_config())
    a = read_json(tmp_path / "baseline/validation/metrics.json")
    b = read_json(tmp_path / "mlp/validation/metrics.json")
    assert a["row_sha256"] == b["row_sha256"]
    table = pa.concat_tables(
        list(iter_model_batches(dataset, 500, "validation", transformed=False))
    )
    assert a["rows"] == b["rows"] == table.num_rows


def test_mlp_no_validation_fits_fixed_epoch_budget(dataset, tmp_path):
    result = train_run(dataset, tmp_path / "empty", fast_config(horizon_ms=5000))
    assert result["fit"]["selection"] == "fixed_epochs_no_validation"
    assert result["fit"]["selected_epoch"] == 8
    assert result["validation_rows"] == 0


def test_raw_pipeline_mlp_cli(tmp_path):
    research, _ = pipeline(tmp_path, n=100)
    split = tmp_path / "split"
    build_split_dataset(research / "research", split, SplitConfig())
    root = Path(__file__).resolve().parents[1]
    config_path = tmp_path / "small.yaml"
    config_path.write_text(
        "model: mlp\nepochs: 3\ntiming_repeats: 2\ntiming_warmup: 1\n", encoding="utf-8"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/train.py"),
            "--dataset",
            str(split),
            "--output",
            str(tmp_path / "run"),
            "--config",
            str(config_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert load_run(tmp_path / "run")[1]["model"] == "mlp"
