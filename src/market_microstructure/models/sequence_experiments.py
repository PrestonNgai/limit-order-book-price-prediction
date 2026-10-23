"""Sequence fitting/evaluation preserves the snapshot pipeline's artifact and TEST guards."""

import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from market_microstructure.evaluation.metrics import classification_metrics
from market_microstructure.processing.build_dataset import file_hash
from market_microstructure.processing.research_dataset import _write_json

from .experiments import artifact, begin, check_dataset, commit, fail, read_json, software
from .recurrent import FrozenSequence, fit_sequence_model, load_sequence_weights
from .sequence_config import SequenceConfig
from .sequences import load_sequences, sequence_plan


def check_inputs(dataset, manifest, prep, config):
    sequence_plan(dataset, manifest, config)
    train = manifest["ranges"]["train"]
    if (
        prep["feature_columns"] != manifest["feature_columns"]
        or prep["horizon_ms"] != config.horizon_ms
        or prep["fit_split"] != "train"
        or prep["train_start_ns"] != train["start_ns"]
        or prep["train_end_exclusive_ns"] != train["end_exclusive_ns"]
        or prep["fit_rows"] != manifest["target_counts"][f"{config.horizon_ms}ms"]["train"]["rows"]
    ):
        raise ValueError("sequence preprocessing does not match TRAIN")
    rows = manifest["ranges"]["train"]["rows"] + manifest["ranges"]["validation"]["rows"]
    # Base arrays plus metadata and conservative mini-batch recurrent activations.
    estimate = rows * (len(prep["feature_columns"]) * 32 + 256)
    estimate += (
        config.train_batch_size
        * config.sequence_length
        * (len(prep["feature_columns"]) + 12 * config.gru_hidden_size * config.gru_layers)
        * 4
    )
    if estimate > config.max_fit_mb * 1024**2:
        raise ValueError("sequence fit exceeds max_fit_mb input/activation estimate")
    return estimate


def score_sequences(frozen, view, labels, info, config, split, destination):
    destination.mkdir()
    n = len(view)
    prediction, probabilities = np.empty(n, dtype=np.int8), np.empty((n, 3))
    started, inference = time.perf_counter(), 0.0
    for start in range(0, n, config.train_batch_size):
        end = min(n, start + config.train_batch_size)
        clock = time.perf_counter()
        prediction[start:end], probabilities[start:end] = frozen.predict(view.batch(start, end))
        inference += time.perf_counter() - clock
    suffix = f"{config.horizon_ms}ms"
    y = labels[f"y_class_{suffix}"].to_numpy()
    report = classification_metrics(y, prediction, probabilities)
    report.update(
        split=split,
        row_sha256=info["row_sha256"],
        sequence_length=config.sequence_length,
        inference_seconds=inference,
        evaluation_seconds=time.perf_counter() - started,
        batch_rows=config.train_batch_size,
        thread_limit=1,
        timing_policy="lazy batch assembly and inference; excludes disk and base transformation",
    )
    table = pa.table(
        {
            "row_id": labels["row_id"],
            "timestamp_ns": labels["timestamp_ns"],
            "sequence_start_row_id": labels["sequence_start_row_id"],
            "label_end_ns": labels[f"label_end_ns_{suffix}"],
            "split": labels["split"],
            "target": labels[f"y_class_{suffix}"],
            "future_return": labels[f"y_return_{suffix}"],
            "prediction": pa.array(prediction),
            "p_down": probabilities[:, 0],
            "p_stable": probabilities[:, 1],
            "p_up": probabilities[:, 2],
        }
    )
    pq.write_table(table, destination / "predictions.parquet.part", compression="zstd")
    (destination / "predictions.parquet.part").rename(destination / "predictions.parquet")
    _write_json(destination / "metrics.json", report)
    return report


def timing_sequences(frozen, view, config):
    if not len(view):
        return {"status": "unavailable", "reason": "no eligible sequence targets"}
    result = {}
    for name, size in (("batch", min(len(view), config.timing_batch_size)), ("single", 1)):
        for _ in range(config.timing_warmup):
            frozen.predict(view.batch(0, size))
        elapsed = []
        for _ in range(config.timing_repeats):
            started = time.perf_counter_ns()
            frozen.predict(view.batch(0, size))
            elapsed.append(time.perf_counter_ns() - started)
        result[name] = {
            "rows_per_call": size,
            "median_call_ns": float(np.median(elapsed)),
            "p95_call_ns": float(np.quantile(elapsed, 0.95)),
        }
    return {
        "status": "complete",
        "warmup_calls": config.timing_warmup,
        "measurement_calls": config.timing_repeats,
        "sequence_length": config.sequence_length,
        "scope": "lazy window assembly + CPU inference; excludes base transform and IO; "
        "not exchange-to-order latency",
        "hardware": software(),
        **result,
    }


def train_sequence_run(dataset, destination, config):
    dataset = Path(dataset)
    manifest, digest = check_dataset(dataset)
    if config.horizon_ms not in manifest["horizons_ms"]:
        raise ValueError("unknown horizon")
    suffix = f"{config.horizon_ms}ms"
    if not manifest["target_counts"][suffix]["train"]["sequence_rows"]:
        raise ValueError("no eligible TRAIN sequences")
    prep_name = manifest["normalizers"][suffix]
    prep_path = artifact(dataset, prep_name, manifest["outputs"][prep_name]["sha256"])
    prep = read_json(prep_path)
    estimate = check_inputs(dataset, manifest, prep, config)
    destination = begin(destination)
    try:
        started = time.perf_counter()
        _write_json(destination / "config.json", config.to_dict())
        for source, name in (
            (dataset / "manifest.json", "dataset_manifest.json"),
            (prep_path, "preprocessor.json"),
            (dataset / "split_plan.json", "split_plan.json"),
        ):
            shutil.copyfile(source, destination / name)
        policy = artifact(
            dataset, "training_report.json", manifest["outputs"]["training_report.json"]["sha256"]
        )
        shutil.copyfile(policy, destination / "label_policy.json")
        train, train_labels, train_info = load_sequences(dataset, manifest, prep, config, "train")
        val, val_labels, val_info = load_sequences(dataset, manifest, prep, config, "validation")
        model, report = fit_sequence_model(train, val, config)
        state = {
            "schema_version": 1,
            "model": config.model,
            "classes": [-1, 0, 1],
            "feature_columns": prep["feature_columns"],
            "config": config.to_dict(),
        }
        _write_json(destination / "model.json", state)
        torch.save(model.state_dict(), destination / "weights.pt")
        before = FrozenSequence(state, prep, model)
        after = FrozenSequence(
            state,
            prep,
            load_sequence_weights(destination / "weights.pt", train.matrix.shape[1], config),
        )
        sample = train.batch(0, min(len(train), config.train_batch_size))
        a, b = before.predict(sample)[1], after.predict(sample)[1]
        np.testing.assert_array_equal(a, b)
        report.update(
            rows=len(train),
            validation_rows=len(val),
            row_sha256=train_info["row_sha256"],
            validation_row_sha256=val_info["row_sha256"],
            train_class_counts=np.bincount(train.targets.numpy(), minlength=3).tolist(),
            reload_max_abs_error=float(np.max(np.abs(a - b))),
            fit_seconds_including_io=time.perf_counter() - started,
            input_working_set_estimate_bytes=estimate,
        )
        _write_json(destination / "fit_report.json", report)
        _write_json(
            destination / "sequence_report.json", {"train": train_info, "validation": val_info}
        )
        # Save window endpoints so alignment can be checked without model inference.
        for role, labels in (("train", train_labels), ("validation", val_labels)):
            pq.write_table(
                labels, destination / f"{role}_sequence_index.parquet", compression="zstd"
            )
        hashes = {name: file_hash(destination / name) for name in ("model.json", "weights.pt")}
        metrics = score_sequences(
            after, val, val_labels, val_info, config, "validation", destination / "validation"
        )
        timing = timing_sequences(after, val if len(val) else train, config)
        timing["sample_split"] = "validation" if len(val) else "train"
        _write_json(destination / "inference_timing.json", timing)
        check_dataset(dataset, digest)
        for name in (
            prep_name,
            "split_plan.json",
            "train/features.parquet",
            f"train/labels_{suffix}.parquet",
            "validation/features.parquet",
            f"validation/labels_{suffix}.parquet",
        ):
            artifact(dataset, name, manifest["outputs"][name]["sha256"])
        for name, checksum in hashes.items():
            artifact(destination, name, checksum)
        return commit(
            destination,
            {
                "schema_version": 1,
                "kind": "sequence_run",
                "status": "complete",
                "created_utc": datetime.now(UTC).isoformat(),
                "run_id": destination.name,
                "model": config.model,
                "horizon_ms": config.horizon_ms,
                "sequence_length": config.sequence_length,
                "feature_columns": prep["feature_columns"],
                "ranges": manifest["ranges"],
                "dataset": {"path": str(dataset.resolve()), "manifest_sha256": digest},
                "eligibility": "frozen purged sequence mask; same GRU/MLP endpoints",
                "selection_policy": "fixed hyperparameters; validation-only checkpoint selection",
                "preprocessing_policy": "full purged TRAIN snapshot scaler, shared by both models",
                "frozen_model_hashes_before_evaluation": hashes,
                "fit": report,
                "validation_rows": metrics["rows"],
                "test_accessed": False,
                "software": software(),
            },
        )
    except Exception as exc:
        fail(destination, exc)
        raise


def load_sequence_run(run):
    run = Path(run)
    manifest = read_json(run / "manifest.json")
    if (
        (run / "_INCOMPLETE").exists()
        or manifest.get("status") != "complete"
        or manifest.get("kind") != "sequence_run"
        or manifest.get("schema_version") != 1
    ):
        raise ValueError("require a committed sequence run")
    required = {
        "model.json",
        "weights.pt",
        "config.json",
        "preprocessor.json",
        "dataset_manifest.json",
        "split_plan.json",
    }
    if required - set(manifest["outputs"]):
        raise ValueError("missing sequence artifacts")
    for name, info in manifest["outputs"].items():
        artifact(run, name, info["sha256"])
    for name, checksum in manifest["frozen_model_hashes_before_evaluation"].items():
        artifact(run, name, checksum)
    config = SequenceConfig(**read_json(run / "config.json"))
    state, prep = read_json(run / "model.json"), read_json(run / "preprocessor.json")
    if (
        state["schema_version"] != 1
        or state["classes"] != [-1, 0, 1]
        or state["config"] != config.to_dict()
        or state["model"] != config.model
        or manifest["model"] != config.model
        or manifest["horizon_ms"] != config.horizon_ms
        or manifest["sequence_length"] != config.sequence_length
        or state["feature_columns"] != prep["feature_columns"]
        or state["feature_columns"] != manifest["feature_columns"]
        or file_hash(run / "dataset_manifest.json") != manifest["dataset"]["manifest_sha256"]
    ):
        raise ValueError("inconsistent sequence run")
    check_inputs(run, read_json(run / "dataset_manifest.json"), prep, config)
    model = load_sequence_weights(run / "weights.pt", len(state["feature_columns"]), config)
    return FrozenSequence(state, prep, model), manifest


def evaluate_sequence_partition(
    dataset, manifest, frozen, config, split, destination, *, allow_test=False
):
    view, labels, info = load_sequences(
        dataset, manifest, frozen.preprocessor, config, split, allow_test=allow_test
    )
    return score_sequences(frozen, view, labels, info, config, split, destination)
