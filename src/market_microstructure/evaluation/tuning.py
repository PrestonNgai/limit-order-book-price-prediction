"""Expanding-session model selection using only the declared development TRAIN period."""

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import yaml

from market_microstructure.models.baselines import (
    fit_logistic,
    imbalance_sign,
    logistic_probabilities,
    majority_class,
)
from market_microstructure.models.config import BaselineConfig
from market_microstructure.models.experiments import begin, check_dataset, commit, fail, software
from market_microstructure.models.learned_config import LearnedConfig
from market_microstructure.models.trees import fit_tree, predict_tree
from market_microstructure.models.tuned import (
    FrozenTuned,
    TunedConfig,
    feature_names,
    fit_classifier,
)
from market_microstructure.processing.preprocessing import TrainPreprocessor
from market_microstructure.processing.research_dataset import _write_json
from market_microstructure.processing.split_dataset import iter_model_batches

from .metrics import classification_metrics

FIELDS = {
    "horizon_ms",
    "seed",
    "feature_sets",
    "class_weights",
    "logistic_c",
    "tree_depths",
    "rounds",
    "patience",
    "max_candidates",
    "max_rows",
}


def candidates(config):
    if not isinstance(config, dict) or set(config) != FIELDS:
        raise ValueError("unknown or missing tuning fields")
    for key in ("horizon_ms", "rounds", "patience", "max_candidates", "max_rows"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"invalid {key}")
    if config["max_candidates"] > 64 or config["max_rows"] > 250000:
        raise ValueError("tuning exceeds bounded search limits")
    for key in ("feature_sets", "class_weights", "logistic_c", "tree_depths"):
        values = config[key]
        if not isinstance(values, list) or not values or len(values) != len(set(values)):
            raise ValueError("search lists must be nonempty and unique")
    result = []
    for subset in config["feature_sets"]:
        for weight in config["class_weights"]:
            base = TunedConfig(
                feature_set=subset,
                class_weight=weight,
                horizon_ms=config["horizon_ms"],
                seed=config["seed"],
                rounds=config["rounds"],
                patience=config["patience"],
            )
            for strength in config["logistic_c"]:
                cfg = replace(base, family="logistic", logistic_c=strength)
                key = f"logistic_{subset}_{weight or 'uniform'}_c{strength}"
                result.append((key, cfg))
            for depth in config["tree_depths"]:
                cfg = replace(base, family="xgboost", max_depth=depth)
                key = f"tree_{subset}_{weight or 'uniform'}_d{depth}"
                result.append((key, cfg))
    if len(result) > config["max_candidates"]:
        raise ValueError("candidate grid exceeds max_candidates")
    return result


def expanding_folds(table, horizon_ms, train_end_ns):
    times = table["timestamp_ns"].to_numpy()
    ends = table[f"label_end_ns_{horizon_ms}ms"].to_numpy()
    if np.any(np.diff(times) <= 0):
        raise ValueError("development rows must be chronological")
    keys = list(zip(table["run_id"].to_pylist(), table["session_id"].to_pylist(), strict=True))
    sessions = []
    for key in keys:
        if not sessions or sessions[-1] != key:
            if key in sessions:
                raise ValueError("development sessions must not reappear")
            sessions.append(key)
    if len(sessions) < 4:
        raise ValueError("walk-forward tuning requires at least four TRAIN sessions")
    for i, key in enumerate(sessions[1:], 1):
        validation = np.array([value == key for value in keys])
        start = int(times[validation].min())
        end = (
            int(times[np.array([value == sessions[i + 1] for value in keys])].min())
            if i + 1 < len(sessions)
            else train_end_ns
        )
        train = (times < start) & (ends < start)
        validation &= ends < end
        if not train.any() or not validation.any():
            raise ValueError("empty walk-forward fold after purging")
        yield i, train, validation, start, end


def fold_preprocessing(table, horizon, train_mask, validation_mask, boundary, *, names=None):
    train, validation = table.filter(train_mask), table.filter(validation_mask)
    names = (
        list(names) if names is not None else [n for n in table.column_names if n.startswith("f_")]
    )
    prep = TrainPreprocessor(names, horizon, int(table["timestamp_ns"][0].as_py()), boundary)
    prep.update(train)
    state = prep.finalize()
    x = TrainPreprocessor.transform(train, state)
    vx = TrainPreprocessor.transform(validation, state)
    y = train[f"y_class_{horizon}ms"].to_numpy()
    vy = validation[f"y_class_{horizon}ms"].to_numpy()
    if set(np.unique(y)) != {-1, 0, 1} or set(np.unique(vy)) != {-1, 0, 1}:
        raise ValueError("all three classes are required in development folds")
    return train, validation, state, x, y, vx, vy


def reference_metrics(x, y, vx, vy, validation, horizon, seed, names):
    counts = np.bincount(y + 1, minlength=3)
    p = np.tile((counts + 1) / (len(y) + 3), (len(vy), 1))
    values = {"majority": classification_metrics(vy, np.full(len(vy), majority_class(counts)), p)}
    cfg = BaselineConfig(horizon_ms=horizon, seed=seed)
    state, _ = fit_logistic(x, y, cfg)
    p = logistic_probabilities(vx, state)
    values["logistic"] = classification_metrics(vy, p.argmax(axis=1) - 1, p)
    signs = imbalance_sign(validation, cfg.to_dict())
    # Only hard predictions are compared here; confidence diagnostics use the saved runs later.
    values["imbalance"] = {
        "balanced_accuracy_present_classes": float(
            np.mean([np.mean(signs[vy == k] == k) for k in (-1, 0, 1)])
        )
    }
    tree, _ = fit_tree(x, y, vx, vy, names, LearnedConfig(horizon_ms=horizon, seed=seed))
    p = predict_tree(tree, vx, names).astype(float)
    p /= p.sum(axis=1, keepdims=True)
    p = (1 - 1e-6) * p + 1e-6 / 3
    values["xgboost"] = classification_metrics(vy, p.argmax(axis=1) - 1, p)
    return values


def run_tuning(dataset, destination, config_path):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    grid = candidates(config)
    manifest, digest = check_dataset(dataset)
    horizon = config["horizon_ms"]
    if horizon not in manifest["horizons_ms"]:
        raise ValueError("unknown tuning horizon")
    count = manifest["target_counts"][f"{horizon}ms"]["train"]["rows"]
    if count > config["max_rows"] or count * len(manifest["feature_columns"]) * 32 > 512 * 1024**2:
        raise ValueError("tuning TRAIN exceeds resource limits")
    table = pa.concat_tables(list(iter_model_batches(dataset, horizon, "train", transformed=False)))
    if len(table) != count:
        raise ValueError("development row count differs from manifest")
    names = manifest["feature_columns"]
    for _, cfg in grid:
        feature_names(names, cfg.feature_set)
    folds = list(expanding_folds(table, horizon, manifest["ranges"]["train"]["end_exclusive_ns"]))
    destination = begin(destination)
    try:
        _write_json(
            destination / "search_plan.json",
            {
                "created_utc": datetime.now(UTC).isoformat(),
                "config": config,
                "dataset_manifest_sha256": digest,
                "criterion": "mean fold balanced accuracy; "
                "tie-break mean corrected log loss, then ID",
                "candidates": [{"id": key, "config": cfg.to_dict()} for key, cfg in grid],
            },
        )
        records = {key: {"id": key, "config": cfg.to_dict(), "folds": []} for key, cfg in grid}
        fold_records = []
        for index, train_mask, val_mask, start, end in folds:
            train, validation, prep, x, y, vx, vy = fold_preprocessing(
                table, horizon, train_mask, val_mask, start, names=names
            )
            references = reference_metrics(x, y, vx, vy, validation, horizon, config["seed"], names)
            fold_records.append(
                {
                    "fold": index,
                    "train_rows": len(y),
                    "validation_rows": len(vy),
                    "validation_start_ns": start,
                    "validation_end_ns": end,
                    "maximum_train_label_end_ns": int(
                        train[f"label_end_ns_{horizon}ms"].to_numpy().max()
                    ),
                    "preprocessor_sha256": hashlib.sha256(
                        json.dumps(prep, sort_keys=True).encode()
                    ).hexdigest(),
                    "references": references,
                }
            )
            for key, cfg in grid:
                active = feature_names(names, cfg.feature_set)
                columns = [names.index(n) for n in active]
                state, estimator, details = fit_classifier(
                    x[:, columns].astype(np.float32),
                    y,
                    vx[:, columns].astype(np.float32),
                    vy,
                    active,
                    cfg,
                )
                predicted, p = FrozenTuned(state, prep, estimator).predict_matrix(
                    vx[:, columns].astype(np.float32)
                )
                metrics = classification_metrics(vy, predicted, p)
                records[key]["folds"].append({"fold": index, "metrics": metrics, "fit": details})
            print(f"Completed development fold {index}/{len(folds)}", flush=True)
            _write_json(destination / "folds.json", fold_records)
        for record in records.values():
            record["mean_balanced_accuracy"] = float(
                np.mean(
                    [f["metrics"]["balanced_accuracy_present_classes"] for f in record["folds"]]
                )
            )
            record["mean_log_loss"] = float(
                np.mean([f["metrics"]["log_loss"] for f in record["folds"]])
            )
        ordered = sorted(
            records.values(),
            key=lambda r: (-r["mean_balanced_accuracy"], r["mean_log_loss"], r["id"]),
        )
        _write_json(destination / "candidates.json", ordered)
        _write_json(destination / "selected_config.json", ordered[0]["config"])
        check_dataset(dataset, digest)
        return commit(
            destination,
            {
                "kind": "walk_forward_tuning",
                "status": "complete",
                "schema_version": 1,
                "created_utc": datetime.now(UTC).isoformat(),
                "dataset_manifest_sha256": digest,
                "test_accessed": False,
                "validation_partition_accessed": False,
                "candidate_count": len(grid),
                "software": software(),
                "source_sha256": {
                    p.relative_to(Path(__file__).parents[1]).as_posix(): hashlib.sha256(
                        p.read_bytes()
                    ).hexdigest()
                    for p in sorted(Path(__file__).parents[1].rglob("*.py"))
                },
                "fold_count": len(folds),
                "selected_id": ordered[0]["id"],
                "selected_mean_balanced_accuracy": ordered[0]["mean_balanced_accuracy"],
            },
        )
    except Exception as exc:
        fail(destination, exc)
        raise
