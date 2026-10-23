import json
from dataclasses import replace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import test_baselines
import torch
from split_helpers import research_fixture

from market_microstructure.models.experiments import evaluate_run, load_run, read_json, train_run
from market_microstructure.models.mlp import cpu_runtime
from market_microstructure.models.recurrent import SmallGRU, fit_sequence_model
from market_microstructure.models.sequence_config import SequenceConfig
from market_microstructure.models.sequences import SequenceDataset, load_sequences
from market_microstructure.processing.build_dataset import file_hash
from market_microstructure.processing.split_config import SplitConfig
from market_microstructure.processing.split_dataset import build_split_dataset

dataset = test_baselines.dataset


def config(**kwargs):
    settings = dict(
        epochs=4,
        patience=2,
        gru_hidden_size=8,
        hidden_sizes=(16,),
        timing_warmup=1,
        timing_repeats=2,
    )
    settings.update(kwargs)
    return SequenceConfig(**settings)


def load(dataset, cfg=None, split="train"):
    cfg = cfg or config()
    manifest = read_json(dataset / "manifest.json")
    prep = read_json(dataset / manifest["normalizers"][f"{cfg.horizon_ms}ms"])
    return load_sequences(dataset, manifest, prep, cfg, split)


@pytest.mark.parametrize(
    "settings",
    [
        {"sequence_length": 0},
        {"sequence_length": True},
        {"gru_hidden_size": 0},
        {"gru_layers": 4},
        {"gradient_clip": float("nan")},
        {"gradient_clip": 0},
    ],
)
def test_sequence_config_rejects_invalid(settings):
    with pytest.raises(ValueError):
        config(**settings)


def test_lazy_views_end_at_t_and_future_perturbation():
    matrix = np.arange(100 * 4, dtype=np.float32).reshape(100, 4)
    view = SequenceDataset(matrix, [19, 30, 50], [0, 1, -1], 20)
    assert view.features.data_ptr() == torch.from_numpy(matrix).data_ptr()
    assert view[0][0].shape == (20, 4)
    np.testing.assert_array_equal(view[0][0].numpy(), matrix[:20])
    before = view[0][0].clone()
    matrix[20:] = -9999
    assert torch.equal(before, view[0][0])
    assert np.shares_memory(view[1][0].numpy(), matrix)
    assert view.matrix.ndim == 2  # No N x L x F expansion.
    snapshot = SequenceDataset(matrix, [19], [0], 20, snapshot=True)
    np.testing.assert_array_equal(snapshot[0][0].numpy(), matrix[19])


@pytest.mark.parametrize("length", [1, 5, 20])
def test_histories_stay_in_split_session_segment_and_cadence(tmp_path, length):
    source = research_fixture(
        tmp_path / "research",
        periods=[
            (40, 0, "a"),
            (40, 5000, "a"),
            (40, 10000, "b"),
            (40, 15000, "b"),
            (40, 20000, "c"),
            (40, 25000, "c"),
        ],
        horizons=(500,),
    )
    split = tmp_path / "splits"
    build_split_dataset(source, split, SplitConfig(sequence_length=length, batch_rows=7))
    cfg = config(sequence_length=length)
    for role in ("train", "validation"):
        view, labels, info = load(split, cfg, role)
        raw = pq.read_table(split / role / "features.parquet")
        assert info["rows"] == len(view)
        for i, endpoint in enumerate(view.endpoints):
            history = raw.slice(int(endpoint) - length + 1, length)
            assert len(set(history["segment_id"].to_pylist())) == 1
            assert len(set(history["session_id"].to_pylist())) == 1
            assert set(history["split"].to_pylist()) == {role}
            assert history["row_id"][-1].as_py() == labels["row_id"][i].as_py()
            assert history["timestamp_ns"][-1].as_py() == labels["timestamp_ns"][i].as_py()
            np.testing.assert_array_equal(
                np.diff(history["timestamp_ns"].to_numpy()), np.full(length - 1, 100_000_000)
            )
            assert labels["sequence_start_row_id"][i].as_py() == history["row_id"][0].as_py()


def test_batch_invariance_and_mask_length_mismatch(dataset):
    a, la, _ = load(dataset, replace(config(), batch_rows=3))
    b, lb, _ = load(dataset, replace(config(), batch_rows=31))
    np.testing.assert_array_equal(a.matrix, b.matrix)
    assert la.equals(lb)
    with pytest.raises(ValueError, match="sequence_length"):
        load(dataset, config(sequence_length=5))


@pytest.mark.parametrize("corruption", ["mask", "start", "gap"])
def test_corrupt_sequence_metadata_is_rejected(dataset, corruption):
    path = dataset / "validation/features.parquet"
    table = pq.read_table(path)
    column = {"mask": "sequence_eligible", "start": "sequence_start_row_id", "gap": "timestamp_ns"}[
        corruption
    ]
    values = table[column].to_pylist()
    if corruption == "mask":
        values[0] = True
    elif corruption == "start":
        values[19] = values[19] - 1
    else:
        values[1] += 1
    table = table.set_column(
        table.schema.get_field_index(column), column, pa.array(values, type=table[column].type)
    )
    pq.write_table(table, path)
    manifest = read_json(dataset / "manifest.json")
    manifest["outputs"]["validation/features.parquet"]["sha256"] = file_hash(path)
    (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="mask|start|gap"):
        load(dataset, split="validation")


def test_empty_validation_sequences_and_test_guard(dataset):
    view, labels, info = load(dataset, config(horizon_ms=5000), "validation")
    assert len(view) == len(labels) == info["rows"] == 0
    with pytest.raises(PermissionError, match="sealed"):
        load(dataset, split="test")


def test_gru_has_no_cross_sample_hidden_state():
    with cpu_runtime(4):
        model = SmallGRU(3, 8, 1, 0.0).eval()
        x = torch.randn(4, 20, 3)
        with torch.inference_mode():
            p = model(x)
            perm = torch.tensor([2, 0, 3, 1])
            np.testing.assert_allclose(model(x[perm]).numpy(), p[perm].numpy(), atol=1e-7)
            np.testing.assert_allclose(model(x[:1]).numpy(), p[:1].numpy(), atol=1e-7)


@pytest.mark.parametrize("model", ["gru", "sequence_mlp"])
def test_training_reload_eligibility_and_test_isolation(dataset, tmp_path, model):
    cfg = config(model=model)
    first, second = tmp_path / "first", tmp_path / "second"
    result = train_run(dataset, first, cfg)
    frozen, _ = load_run(first)
    view, labels, _ = load(dataset, cfg, "validation")
    predictions = frozen.predict(view.batch(0, len(view)))[1]
    assert len(predictions) == result["validation_rows"]
    assert result["fit"]["reload_max_abs_error"] == 0
    (dataset / "test/features.parquet").write_bytes(b"REAL TEST WOULD REMAIN UNREAD")
    train_run(dataset, second, cfg)
    again, _ = load_run(second)
    for name, tensor in frozen.estimator.state_dict().items():
        assert torch.equal(tensor, again.estimator.state_dict()[name])
    assert (first / "preprocessor.json").read_bytes() == (second / "preprocessor.json").read_bytes()
    evaluate_run(first, dataset, tmp_path / "reload")
    assert pq.read_table(first / "validation/predictions.parquet").equals(
        pq.read_table(tmp_path / "reload/validation/predictions.parquet")
    )
    with pytest.raises(PermissionError):
        evaluate_run(first, dataset, tmp_path / "forbidden", split="test")
    with (first / "weights.pt").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="hash"):
        load_run(first)


def test_gru_and_mlp_use_same_train_and_validation_targets(dataset, tmp_path):
    for model in ("gru", "sequence_mlp"):
        train_run(dataset, tmp_path / model, config(model=model))
    a, b = [read_json(tmp_path / m / "fit_report.json") for m in ("gru", "sequence_mlp")]
    assert a["row_sha256"] == b["row_sha256"]
    assert a["validation_row_sha256"] == b["validation_row_sha256"]
    for role in ("train", "validation"):
        assert pq.read_table(tmp_path / "gru" / f"{role}_sequence_index.parquet").equals(
            pq.read_table(tmp_path / "sequence_mlp" / f"{role}_sequence_index.parquet")
        )


def test_sequence_memory_and_row_limits(dataset, tmp_path):
    with pytest.raises(ValueError, match="memory limit"):
        train_run(dataset, tmp_path / "limit", config(max_train_rows=1))
    with pytest.raises(ValueError, match="max_fit_mb"):
        train_run(dataset, tmp_path / "memory", config(max_fit_mb=1))


def test_gru_training_requires_temporal_information():
    # Same last observation, opposite first observations: snapshot-only inputs cannot solve this.
    matrix = np.zeros((48, 2), dtype=np.float32)
    endpoints, targets = np.arange(3, 48, 4), np.tile([-1, 1], 6)
    for i, target in enumerate(targets):
        matrix[4 * i, 0] = target
    view = SequenceDataset(matrix, endpoints, targets, 4)
    empty = SequenceDataset(matrix, [], [], 4)
    cfg = replace(
        config(sequence_length=4), epochs=100, dropout=0.0, learning_rate=0.03, train_batch_size=12
    )
    model, _ = fit_sequence_model(view, empty, cfg)
    with cpu_runtime(), torch.inference_mode():
        prediction = model(torch.from_numpy(view.batch(0, len(view)))).argmax(dim=1).numpy() - 1
    np.testing.assert_array_equal(prediction, targets)


def test_empty_validation_run_and_no_training_sequences(dataset, tmp_path):
    result = train_run(dataset, tmp_path / "no_validation", config(horizon_ms=5000))
    assert result["fit"]["selection"] == "fixed_epochs_no_validation"
    assert result["validation_rows"] == 0
    source = research_fixture(tmp_path / "short", n=60, horizons=(500,))
    split = tmp_path / "long_history"
    build_split_dataset(source, split, SplitConfig(sequence_length=60))
    with pytest.raises(ValueError, match="no eligible TRAIN sequences"):
        train_run(split, tmp_path / "no_train", config(sequence_length=60))


def test_sequence_dataset_float32_overflow():
    with pytest.raises(ValueError, match="float32"):
        SequenceDataset(np.array([[1e100]]), [0], [0], 1)


def test_future_target_cannot_cross_gap(tmp_path):
    source = research_fixture(
        tmp_path / "research",
        periods=[
            (40, 0, "a"),
            (40, 5000, "a"),
            (40, 10000, "b"),
            (40, 15000, "b"),
            (40, 20000, "c"),
            (40, 25000, "c"),
        ],
        horizons=(500,),
    )
    split = tmp_path / "split"
    build_split_dataset(source, split, SplitConfig())
    features = pq.read_table(split / "validation/features.parquet")
    path = split / "validation/labels_500ms.parquet"
    labels = pq.read_table(path)
    ends = labels["label_end_ns_500ms"].to_pylist()
    ends[19] = features["timestamp_ns"][45].as_py()
    labels = labels.set_column(
        labels.schema.get_field_index("label_end_ns_500ms"), "label_end_ns_500ms", pa.array(ends)
    )
    pq.write_table(labels, path)
    manifest = read_json(split / "manifest.json")
    manifest["outputs"]["validation/labels_500ms.parquet"]["sha256"] = file_hash(path)
    (split / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="segment"):
        load(split, split="validation")


def test_explicit_test_evaluation_uses_synthetic_sequences_only(dataset, tmp_path):
    run = tmp_path / "run"
    train_run(dataset, run, config())
    result = evaluate_run(run, dataset, tmp_path / "allowed", split="test", allow_test=True)
    manifest = read_json(dataset / "manifest.json")
    assert result["test_accessed"] is True
    assert result["rows"] == manifest["target_counts"]["500ms"]["test"]["sequence_rows"]
