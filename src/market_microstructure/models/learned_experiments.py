"""Bounded learned-model training with TRAIN provenance and validation-only selection."""

import hashlib
import shutil
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from market_microstructure.processing.build_dataset import file_hash
from market_microstructure.processing.preprocessing import TrainPreprocessor
from market_microstructure.processing.research_dataset import _write_json
from market_microstructure.processing.split_dataset import iter_model_batches

from .experiments import (
    artifact,
    begin,
    check_dataset,
    commit,
    evaluate_partition,
    fail,
    read_json,
    row_fingerprint,
    software,
)
from .learned import FrozenLearned, float32_matrix
from .learned_config import LearnedConfig


def read_partition(dataset, manifest, prep, config, split):
    count = manifest["target_counts"][f"{config.horizon_ms}ms"][split]["rows"]
    matrix = np.empty((count, len(prep["feature_columns"])), dtype=np.float32)
    target = np.empty(count, dtype=np.int8)
    row_ids = np.empty(count, dtype=np.int64)
    checker = None
    if split == "train":
        bound = manifest["ranges"]["train"]
        checker = TrainPreprocessor(
            prep["feature_columns"], config.horizon_ms, bound["start_ns"], bound["end_exclusive_ns"]
        )
    offset, fingerprint, sample = 0, hashlib.sha256(), None
    for table in iter_model_batches(
        dataset, config.horizon_ms, split, batch_rows=config.batch_rows, transformed=False
    ):
        end = offset + table.num_rows
        if end > count:
            raise ValueError("partition row count exceeds manifest")
        labels = table[f"y_class_{config.horizon_ms}ms"].to_numpy()
        if not np.isin(labels, [-1, 0, 1]).all():
            raise ValueError("invalid training/validation classes")
        matrix[offset:end] = float32_matrix(table, prep)
        target[offset:end] = labels
        row_ids[offset:end] = table["row_id"].to_numpy()
        if checker:
            checker.update(table)
        row_fingerprint(fingerprint, table, config.horizon_ms)
        if sample is None:
            sample = table.slice(0, config.timing_batch_size)
        offset = end
    if offset != count:
        raise ValueError("partition row count differs from manifest")
    if checker and checker.finalize()["fit_row_sha256"] != prep["fit_row_sha256"]:
        raise ValueError("TRAIN values differ from preprocessor provenance")
    return matrix, target, row_ids, fingerprint.hexdigest(), sample


def load_estimator(run, state):
    if state["model"] == "xgboost":
        import xgboost as xgb

        estimator = xgb.Booster({"nthread": 1})
        estimator.load_model(run / "tree.ubj")
        if estimator.feature_names != state["feature_columns"]:
            raise ValueError("tree feature schema differs from frozen metadata")
        if estimator.num_boosted_rounds() != state["selected_rounds"]:
            raise ValueError("tree round count differs from selected checkpoint")
        return estimator
    from .mlp import load_mlp

    return load_mlp(
        run / "weights.pt", len(state["feature_columns"]), LearnedConfig(**state["config"])
    )


def benchmark(frozen, table, config, sample_split):
    if table is None:
        return {"status": "unavailable", "reason": "no timing observations"}
    table = table.slice(0, config.timing_batch_size)
    result = {}
    with threadpool_limits(limits=1):
        for name, batch in (("batch", table), ("single", table.slice(0, 1))):
            for _ in range(config.timing_warmup):
                frozen.predict(batch)
            elapsed = []
            for _ in range(config.timing_repeats):
                started = time.perf_counter_ns()
                frozen.predict(batch)
                elapsed.append(time.perf_counter_ns() - started)
            result[name] = {
                "rows_per_call": batch.num_rows,
                "median_call_ns": float(np.median(elapsed)),
                "p95_call_ns": float(np.quantile(elapsed, 0.95)),
                "median_per_row_ns": float(np.median(elapsed)) / batch.num_rows,
            }
    return {
        "status": "complete",
        "sample_split": sample_split,
        "device": "cpu",
        "thread_limit": 1,
        "requested_batch_size": config.timing_batch_size,
        "warmup_calls_per_shape": config.timing_warmup,
        "measured_calls_per_shape": config.timing_repeats,
        **result,
        "scope": "Arrow table -> frozen transform -> prediction, warmed in-process; "
        "excludes disk/network; not exchange-to-order latency",
    }


def train_learned_run(dataset, destination, config, *, feature_columns=None):
    dataset = Path(dataset)
    manifest, digest = check_dataset(dataset)
    suffix = f"{config.horizon_ms}ms"
    if config.horizon_ms not in manifest["horizons_ms"]:
        raise ValueError("unknown horizon")
    counts = manifest["target_counts"][suffix]
    train_count, val_count = counts["train"]["rows"], counts["validation"]["rows"]
    if train_count <= 0:
        raise ValueError("no eligible TRAIN targets")
    if train_count > config.max_train_rows or val_count > config.max_eval_rows:
        raise ValueError("partition exceeds configured row limit")
    full_names = manifest["feature_columns"]
    names = full_names if feature_columns is None else list(feature_columns)
    if not names or len(set(names)) != len(names) or any(n not in full_names for n in names):
        raise ValueError("feature selection must be a nonempty unique subset")
    indices = [full_names.index(n) for n in names]
    estimate = (train_count + val_count) * (len(full_names) * 8 * 4 + 32)
    if estimate > config.max_fit_mb * 1024**2:
        raise ValueError("learned fit exceeds max_fit_mb input working-set estimate")
    prep_name = manifest["normalizers"][suffix]
    if prep_name is None:
        raise ValueError("no TRAIN preprocessor")
    prep_path = artifact(dataset, prep_name, manifest["outputs"][prep_name]["sha256"])
    prep = read_json(prep_path)
    bounds = manifest["ranges"]["train"]
    if (
        prep["feature_columns"] != full_names
        or prep["horizon_ms"] != config.horizon_ms
        or prep["fit_rows"] != train_count
        or prep["fit_split"] != "train"
        or prep["train_start_ns"] != bounds["start_ns"]
        or prep["train_end_exclusive_ns"] != bounds["end_exclusive_ns"]
    ):
        raise ValueError("preprocessor does not match the training partition")
    policy_path = artifact(
        dataset, "training_report.json", manifest["outputs"]["training_report.json"]["sha256"]
    )
    destination = begin(destination)
    started = time.perf_counter()
    try:
        _write_json(destination / "config.json", config.to_dict())
        for source, name in (
            (dataset / "manifest.json", "dataset_manifest.json"),
            (prep_path, "preprocessor.json"),
            (policy_path, "label_policy.json"),
        ):
            shutil.copyfile(source, destination / name)
        train_x, train_y, _, train_digest, train_sample = read_partition(
            dataset, manifest, prep, config, "train"
        )
        val_x, val_y, val_ids, val_digest, val_sample = read_partition(
            dataset, manifest, prep, config, "validation"
        )
        train_x = np.ascontiguousarray(train_x[:, indices])
        val_x = np.ascontiguousarray(val_x[:, indices])
        state = {
            "schema_version": 1,
            "model": config.model,
            "classes": [-1, 0, 1],
            "feature_columns": names,
            "config": config.to_dict(),
            "train_class_counts": np.bincount(train_y + 1, minlength=3).tolist(),
        }
        optimization_started = time.perf_counter()
        with threadpool_limits(limits=1):
            if config.model == "xgboost":
                from .trees import contributions, fit_tree, importance

                estimator, details = fit_tree(train_x, train_y, val_x, val_y, names, config)
                estimator.save_model(destination / "tree.ubj")
                state["selected_rounds"] = details["selected_rounds"]
                _write_json(destination / "feature_importance.json", importance(estimator, names))
                if config.explain_rows and val_count:
                    n = min(config.explain_rows, val_count)
                    explained = contributions(estimator, val_x[:n], names)
                    explained.update(split="validation", row_ids=val_ids[:n].tolist())
                    _write_json(destination / "shap_validation.json", explained)
            else:
                from .mlp import fit_mlp, save_mlp

                estimator, details = fit_mlp(train_x, train_y, val_x, val_y, config)
                save_mlp(estimator, destination / "weights.pt")
        details["optimization_seconds"] = time.perf_counter() - optimization_started
        _write_json(destination / "model.json", state)
        before = FrozenLearned(state, prep, estimator)
        after = FrozenLearned(
            read_json(destination / "model.json"), prep, load_estimator(destination, state)
        )
        # Reload parity uses TRAIN; validation already selected the checkpoint.
        sample = train_x[: min(len(train_x), 256)]
        expected, actual = before.probabilities(sample), after.probabilities(sample)
        np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-9)
        details.update(
            rows=train_count,
            validation_rows=val_count,
            row_sha256=train_digest,
            validation_row_sha256=val_digest,
            class_counts=state["train_class_counts"],
            input_working_set_estimate_bytes=estimate,
            reload_max_abs_error=float(np.max(np.abs(actual - expected))),
            fit_seconds_including_io=time.perf_counter() - started,
        )
        _write_json(destination / "fit_report.json", details)
        frozen_files = ("model.json", "tree.ubj" if config.model == "xgboost" else "weights.pt")
        frozen_hashes = {name: file_hash(destination / name) for name in frozen_files}
        metrics = evaluate_partition(
            dataset, manifest, after, config, "validation", destination / "validation"
        )
        if metrics["row_sha256"] != val_digest:
            raise ValueError("checkpoint selection and evaluation rows differ")
        timing = benchmark(
            after,
            val_sample if val_sample is not None else train_sample,
            config,
            "validation" if val_sample is not None else "train",
        )
        timing["hardware"] = software()
        _write_json(destination / "inference_timing.json", timing)
        check_dataset(dataset, digest)
        for split in ("train", "validation"):
            for name in (f"{split}/features.parquet", f"{split}/labels_{suffix}.parquet"):
                artifact(dataset, name, manifest["outputs"][name]["sha256"])
        artifact(dataset, prep_name, manifest["outputs"][prep_name]["sha256"])
        for name, expected_hash in frozen_hashes.items():
            artifact(destination, name, expected_hash)
        return commit(
            destination,
            {
                "schema_version": 1,
                "kind": "learned_run",
                "status": "complete",
                "created_utc": datetime.now(UTC).isoformat(),
                "run_id": destination.name,
                "model": config.model,
                "horizon_ms": config.horizon_ms,
                "dataset": {"path": str(dataset.resolve()), "manifest_sha256": digest},
                "feature_columns": names,
                "active_feature_columns": names,
                "ranges": manifest["ranges"],
                "eligibility": "all purged snapshot targets; no sequence filter",
                "selection_policy": "fixed hyperparameters; validation-only checkpoint selection",
                "frozen_model_hashes_before_evaluation": frozen_hashes,
                "fit": details,
                "validation_rows": metrics["rows"],
                "test_accessed": False,
                "software": {
                    **software(),
                    config.model if config.model == "xgboost" else "torch": version(
                        "xgboost" if config.model == "xgboost" else "torch"
                    ),
                },
            },
        )
    except Exception as exc:
        fail(destination, exc)
        raise


def load_learned_run(run):
    run = Path(run)
    manifest = read_json(run / "manifest.json")
    if (
        (run / "_INCOMPLETE").exists()
        or manifest.get("status") != "complete"
        or manifest.get("kind") != "learned_run"
        or manifest.get("schema_version") != 1
    ):
        raise ValueError("require a committed learned run")
    required = {
        "model.json",
        "config.json",
        "preprocessor.json",
        "dataset_manifest.json",
        "tree.ubj" if manifest["model"] == "xgboost" else "weights.pt",
    }
    if required - set(manifest["outputs"]):
        raise ValueError("missing required learned artifacts")
    for name, info in manifest["outputs"].items():
        artifact(run, name, info["sha256"])
    for name, expected in manifest["frozen_model_hashes_before_evaluation"].items():
        artifact(run, name, expected)
    state, prep = read_json(run / "model.json"), read_json(run / "preprocessor.json")
    config = LearnedConfig(**read_json(run / "config.json"))
    if (
        state["schema_version"] != 1
        or state["classes"] != [-1, 0, 1]
        or state["model"] != manifest["model"]
        or state["model"] != config.model
        or state["config"] != config.to_dict()
        or not state["feature_columns"]
        or len(set(state["feature_columns"])) != len(state["feature_columns"])
        or any(n not in prep["feature_columns"] for n in state["feature_columns"])
        or prep["feature_columns"] != read_json(run / "dataset_manifest.json")["feature_columns"]
        or state["feature_columns"] != manifest["feature_columns"]
        or config.horizon_ms != manifest["horizon_ms"]
        or file_hash(run / "dataset_manifest.json") != manifest["dataset"]["manifest_sha256"]
    ):
        raise ValueError("inconsistent frozen learned run")
    return FrozenLearned(state, prep, load_estimator(run, state)), manifest
