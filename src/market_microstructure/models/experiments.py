"""Fit models on TRAIN and save runs with data and checkpoint hashes."""

import hashlib
import json
import platform
import shutil
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

from market_microstructure import __version__
from market_microstructure.evaluation.metrics import classification_metrics
from market_microstructure.processing.build_dataset import _git_commit, file_hash
from market_microstructure.processing.preprocessing import TrainPreprocessor
from market_microstructure.processing.research_dataset import _write_json
from market_microstructure.processing.split_dataset import iter_model_batches

from .baselines import FrozenBaseline, fit_logistic, imbalance_sign, majority_class
from .config import BaselineConfig


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def artifact(root, name, expected):
    path = (Path(root) / name).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("artifact path escapes its directory")
    if file_hash(path) != expected:
        raise ValueError(f"artifact hash mismatch: {name}")
    return path


def software():
    return {
        "project": __version__,
        "python": platform.python_version(),
        **{
            n: version(n)
            for n in (
                "numpy",
                "pyarrow",
                "scikit-learn",
                "scipy",
                "threadpoolctl",
                "xgboost",
                "torch",
            )
        },
        "git_commit": _git_commit(),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }


def begin(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "_INCOMPLETE").write_text("Incomplete experiment\n", encoding="utf-8")
    return destination


def fail(destination, exc):
    _write_json(destination / "failure.json", {"error_type": type(exc).__name__, "error": str(exc)})


def commit(destination, manifest):
    manifest["outputs"] = {
        p.relative_to(destination).as_posix(): {"sha256": file_hash(p), "bytes": p.stat().st_size}
        for p in sorted(destination.rglob("*"))
        if p.is_file() and p.name not in ("manifest.json", "_INCOMPLETE")
    }
    _write_json(destination / "manifest.json.part", manifest)
    (destination / "manifest.json.part").rename(destination / "manifest.json")
    (destination / "_INCOMPLETE").unlink()
    return manifest


def check_dataset(dataset, expected_digest=None):
    dataset = Path(dataset)
    manifest = read_json(dataset / "manifest.json")
    if (
        (dataset / "_INCOMPLETE").exists()
        or manifest.get("status") != "complete"
        or manifest.get("kind") != "chronological_splits"
    ):
        raise ValueError("a committed chronological split dataset is required")
    digest = file_hash(dataset / "manifest.json")
    if expected_digest is not None and digest != expected_digest:
        raise ValueError("dataset manifest differs from the frozen training run")
    return manifest, digest


def row_fingerprint(digest, table, horizon):
    data = np.empty(
        table.num_rows, dtype=[("row", "<i8"), ("time", "<i8"), ("end", "<i8"), ("class", "i1")]
    )
    for field, column in (
        ("row", "row_id"),
        ("time", "timestamp_ns"),
        ("end", f"label_end_ns_{horizon}ms"),
        ("class", f"y_class_{horizon}ms"),
    ):
        data[field] = table[column].to_numpy()
    digest.update(data.tobytes())


def train_run(dataset, destination, config):
    if config.model in ("gru", "sequence_mlp"):
        from .sequence_experiments import train_sequence_run

        return train_sequence_run(dataset, destination, config)
    if config.model in ("xgboost", "mlp"):
        from .learned_experiments import train_learned_run

        return train_learned_run(dataset, destination, config)
    dataset = Path(dataset)
    manifest, digest = check_dataset(dataset)
    suffix = f"{config.horizon_ms}ms"
    if config.horizon_ms not in manifest["horizons_ms"]:
        raise ValueError("horizon is absent from the split dataset")
    expected = manifest["target_counts"][suffix]["train"]["rows"]
    if expected <= 0:
        raise ValueError("no eligible TRAIN targets")
    if expected > config.max_train_rows:
        raise ValueError("TRAIN exceeds max_train_rows; increase the explicit limit")
    names = manifest["feature_columns"]
    active_features = (
        names
        if config.model == "logistic"
        else [config.imbalance_feature]
        if config.model == "imbalance"
        else []
    )
    # Conservative four-matrix allowance; native solver workspace is additional.
    estimate = expected * (len(names) * 8 * 4 + 32)
    if config.model == "logistic" and estimate > config.max_fit_mb * 1024**2:
        raise ValueError("logistic fit exceeds max_fit_mb working-set estimate")
    state_name = manifest["normalizers"][suffix]
    if state_name is None:
        raise ValueError("missing TRAIN preprocessor")
    state_path = artifact(dataset, state_name, manifest["outputs"][state_name]["sha256"])
    preprocessor = read_json(state_path)
    bounds = manifest["ranges"]["train"]
    if (
        preprocessor["feature_columns"] != names
        or preprocessor["horizon_ms"] != config.horizon_ms
        or preprocessor["fit_rows"] != expected
        or preprocessor["train_start_ns"] != bounds["start_ns"]
        or preprocessor["train_end_exclusive_ns"] != bounds["end_exclusive_ns"]
    ):
        raise ValueError("preprocessor does not match the training partition")
    report_path = artifact(
        dataset, "training_report.json", manifest["outputs"]["training_report.json"]["sha256"]
    )
    destination = begin(destination)
    try:
        # Save settings before reading validation.
        _write_json(destination / "config.json", config.to_dict())
        shutil.copyfile(dataset / "manifest.json", destination / "dataset_manifest.json")
        shutil.copyfile(state_path, destination / "preprocessor.json")
        shutil.copyfile(report_path, destination / "label_policy.json")
        matrix = np.empty((expected, len(names))) if config.model == "logistic" else None
        targets = np.empty(expected, dtype=np.int8) if matrix is not None else None
        counts, conditional = np.zeros(3, dtype=np.int64), np.zeros((3, 3), dtype=np.int64)
        checker = TrainPreprocessor(
            names, config.horizon_ms, bounds["start_ns"], bounds["end_exclusive_ns"]
        )
        rows, fingerprint = 0, hashlib.sha256()
        started = time.perf_counter()
        for table in iter_model_batches(
            dataset, config.horizon_ms, "train", batch_rows=config.batch_rows, transformed=False
        ):
            y = table[f"y_class_{suffix}"].to_numpy()
            if not np.isin(y, [-1, 0, 1]).all() or rows + len(y) > expected:
                raise ValueError("invalid TRAIN classes or row count")
            checker.update(table)
            row_fingerprint(fingerprint, table, config.horizon_ms)
            counts += np.bincount(y.astype(int) + 1, minlength=3)
            if config.model == "imbalance":
                signs = imbalance_sign(table, config.to_dict())
                np.add.at(conditional, (signs + 1, y.astype(int) + 1), 1)
            if matrix is not None:
                matrix[rows : rows + len(y)] = TrainPreprocessor.transform(table, preprocessor)
                targets[rows : rows + len(y)] = y
            rows += len(y)
        if rows != expected:
            raise ValueError("TRAIN row count differs from split manifest")
        if checker.finalize()["fit_row_sha256"] != preprocessor["fit_row_sha256"]:
            raise ValueError("TRAIN values differ from the fitted preprocessor provenance")
        state = {
            "schema_version": 1,
            "model": config.model,
            "classes": [-1, 0, 1],
            "config": config.to_dict(),
            "feature_columns": names,
            "train_class_counts": counts.tolist(),
            "active_feature_columns": active_features,
        }
        fit = {
            "status": "fitted_rule",
            "rows": rows,
            "row_sha256": fingerprint.hexdigest(),
            "class_counts": counts.tolist(),
            "input_working_set_estimate_bytes": estimate if matrix is not None else None,
        }
        prior = (counts + config.smoothing) / (rows + 3 * config.smoothing)
        if config.model == "majority":
            state.update(majority_class=majority_class(counts), probabilities=prior.tolist())
        elif config.model == "imbalance":
            # Three sign bins. Each gets alpha total pseudocounts distributed by TRAIN prior.
            posterior = (conditional + config.smoothing * prior) / (
                conditional.sum(axis=1, keepdims=True) + config.smoothing
            )
            state.update(
                conditional_counts=conditional.tolist(),
                conditional_probabilities=posterior.tolist(),
                probability_policy="TRAIN class frequencies conditional on fixed OBI sign; "
                "unseen sign bins use smoothed TRAIN prior",
            )
        else:
            parameters, details = fit_logistic(matrix, targets, config)
            state.update(parameters)
            fit.update(details)
        fit["fit_seconds_including_train_io_and_preprocessing"] = time.perf_counter() - started
        _write_json(destination / "model.json", state)
        _write_json(destination / "fit_report.json", fit)
        frozen_digest = file_hash(destination / "model.json")
        # Evaluate the saved model, not the training object.
        frozen = FrozenBaseline(
            read_json(destination / "model.json"), read_json(destination / "preprocessor.json")
        )
        metrics = evaluate_partition(
            dataset, manifest, frozen, config, "validation", destination / "validation"
        )
        if file_hash(destination / "model.json") != frozen_digest:
            raise ValueError("model changed during validation")
        check_dataset(dataset, digest)
        for name in ("train/features.parquet", f"train/labels_{suffix}.parquet", state_name):
            artifact(dataset, name, manifest["outputs"][name]["sha256"])
        return commit(
            destination,
            {
                "schema_version": 1,
                "kind": "baseline_run",
                "status": "complete",
                "created_utc": datetime.now(UTC).isoformat(),
                "run_id": destination.name,
                "dataset": {"path": str(dataset.resolve()), "manifest_sha256": digest},
                "model": config.model,
                "horizon_ms": config.horizon_ms,
                "feature_columns": names,
                "active_feature_columns": active_features,
                "ranges": manifest["ranges"],
                "eligibility": "all purged snapshot targets; no sequence filter",
                "frozen_model_sha256_before_validation": frozen_digest,
                "fit": fit,
                "validation_rows": metrics["rows"],
                "test_accessed": False,
                "selection_policy": "fixed configuration; no validation tuning",
                "software": software(),
            },
        )
    except Exception as exc:
        fail(destination, exc)
        raise


def load_run(run):
    run = Path(run)
    manifest = read_json(run / "manifest.json")
    if manifest.get("kind") == "sequence_run":
        from .sequence_experiments import load_sequence_run

        return load_sequence_run(run)
    if manifest.get("kind") == "learned_run":
        from .learned_experiments import load_learned_run

        return load_learned_run(run)
    if (
        (run / "_INCOMPLETE").exists()
        or manifest.get("status") != "complete"
        or manifest.get("kind") != "baseline_run"
        or manifest.get("schema_version") != 1
    ):
        raise ValueError("require a committed baseline run")
    for name, info in manifest["outputs"].items():
        artifact(run, name, info["sha256"])
    state, prep = read_json(run / "model.json"), read_json(run / "preprocessor.json")
    config = BaselineConfig(**read_json(run / "config.json"))
    if (
        state["config"] != config.to_dict()
        or state["feature_columns"] != prep["feature_columns"]
        or state["model"] != config.model
        or config.horizon_ms != manifest["horizon_ms"]
        or file_hash(run / "model.json") != manifest["frozen_model_sha256_before_validation"]
        or file_hash(run / "dataset_manifest.json") != manifest["dataset"]["manifest_sha256"]
    ):
        raise ValueError("inconsistent frozen run artifacts")
    return FrozenBaseline(state, prep), manifest


def evaluate_partition(dataset, manifest, frozen, config, split, destination, *, allow_test=False):
    if config.model in ("gru", "sequence_mlp"):
        from .sequence_experiments import evaluate_sequence_partition

        return evaluate_sequence_partition(
            dataset, manifest, frozen, config, split, destination, allow_test=allow_test
        )
    if split == "test" and not allow_test:
        raise PermissionError("TEST is sealed; use explicit final-test opt-in")
    suffix = f"{config.horizon_ms}ms"
    expected = manifest["target_counts"][suffix][split]["rows"]
    if expected > config.max_eval_rows:
        raise ValueError("evaluation exceeds max_eval_rows; no rows are silently truncated")
    destination.mkdir()
    y = np.empty(expected, dtype=np.int8)
    predicted = np.empty(expected, dtype=np.int8)
    probabilities = np.empty((expected, 3))
    schema = pa.schema(
        [
            ("row_id", pa.int64()),
            ("timestamp_ns", pa.int64()),
            ("label_end_ns", pa.int64()),
            ("split", pa.string()),
            ("target", pa.int8()),
            ("future_return", pa.float64()),
            ("prediction", pa.int8()),
            ("p_down", pa.float64()),
            ("p_stable", pa.float64()),
            ("p_up", pa.float64()),
        ]
    )
    partial = destination / "predictions.parquet.part"
    rows, fingerprint, inference = 0, hashlib.sha256(), 0.0
    elapsed = time.perf_counter()
    with pq.ParquetWriter(partial, schema, compression="zstd") as writer:
        with threadpool_limits(limits=1):
            for table in iter_model_batches(
                dataset,
                config.horizon_ms,
                split,
                batch_rows=config.batch_rows,
                transformed=False,
                allow_test=allow_test,
            ):
                size = table.num_rows
                if rows + size > expected:
                    raise ValueError("evaluation row count exceeds split manifest")
                start = time.perf_counter()
                pred, p = frozen.predict(table)
                inference += time.perf_counter() - start
                labels = table[f"y_class_{suffix}"].to_numpy()
                y[rows : rows + size], predicted[rows : rows + size] = labels, pred
                probabilities[rows : rows + size] = p
                row_fingerprint(fingerprint, table, config.horizon_ms)
                writer.write_table(
                    pa.table(
                        {
                            "row_id": table["row_id"],
                            "timestamp_ns": table["timestamp_ns"],
                            "label_end_ns": table[f"label_end_ns_{suffix}"],
                            "split": table["split"],
                            "target": pa.array(labels, pa.int8()),
                            "future_return": table[f"y_return_{suffix}"],
                            "prediction": pa.array(pred, pa.int8()),
                            "p_down": p[:, 0],
                            "p_stable": p[:, 1],
                            "p_up": p[:, 2],
                        },
                        schema=schema,
                    )
                )
                rows += size
    if rows != expected:
        raise ValueError("evaluation row count differs from split manifest")
    for name in (f"{split}/features.parquet", f"{split}/labels_{suffix}.parquet"):
        artifact(dataset, name, manifest["outputs"][name]["sha256"])
    metrics = classification_metrics(y, predicted, probabilities)
    metrics.update(
        split=split,
        row_sha256=fingerprint.hexdigest(),
        inference_seconds=inference,
        batch_rows=config.batch_rows,
        thread_limit=1,
        evaluation_seconds_including_io_and_metrics=time.perf_counter() - elapsed,
        timing_policy="unwarmed aggregate batch inference including transform; "
        "excludes data loading; not a trading latency benchmark",
    )
    _write_json(destination / "metrics.json", metrics)
    partial.rename(destination / "predictions.parquet")
    return metrics


def evaluate_run(run, dataset, destination, *, split="validation", allow_test=False):
    if split not in ("validation", "test"):
        raise ValueError("evaluation split must be validation or test")
    if split == "test" and not allow_test:
        raise PermissionError("TEST is sealed; use explicit final-test opt-in")
    frozen, trained = load_run(run)
    manifest, digest = check_dataset(dataset, trained["dataset"]["manifest_sha256"])
    from .learned_config import config_from_mapping

    config = config_from_mapping(frozen.state["config"])
    destination = begin(destination)
    try:
        metrics = evaluate_partition(
            Path(dataset),
            manifest,
            frozen,
            config,
            split,
            destination / split,
            allow_test=allow_test,
        )
        check_dataset(dataset, digest)
        return commit(
            destination,
            {
                "schema_version": 1,
                "kind": "baseline_evaluation",
                "status": "complete",
                "created_utc": datetime.now(UTC).isoformat(),
                "run": {
                    "path": str(Path(run).resolve()),
                    "manifest_sha256": file_hash(Path(run) / "manifest.json"),
                },
                "dataset_manifest_sha256": digest,
                "split": split,
                "rows": metrics["rows"],
                "test_accessed": split == "test",
                "software": software(),
            },
        )
    except Exception as exc:
        fail(destination, exc)
        raise
