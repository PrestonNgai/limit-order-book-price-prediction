"""Regularized classifiers with explicit feature sets and TRAIN-derived class costs."""

import math
import shutil
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from market_microstructure.processing.build_dataset import file_hash
from market_microstructure.processing.research_dataset import _write_json

from .experiments import (
    artifact,
    begin,
    check_dataset,
    commit,
    evaluate_partition,
    fail,
    read_json,
    software,
)
from .learned import float32_matrix
from .learned_experiments import read_partition


@dataclass(frozen=True)
class TunedConfig:
    model: str = "tuned"
    family: str = "logistic"
    feature_set: str = "compact"
    class_weight: str | None = None
    logistic_c: float = 0.1
    max_depth: int = 2
    min_child_weight: float = 10.0
    reg_lambda: float = 10.0
    tree_learning_rate: float = 0.05
    rounds: int = 200
    patience: int = 15
    horizon_ms: int = 500
    seed: int = 42
    batch_rows: int = 4096
    max_train_rows: int = 250000
    max_eval_rows: int = 250000
    max_fit_mb: int = 512
    timing_batch_size: int = 64
    probability_mix: float = 1e-6

    def __post_init__(self):
        if self.model != "tuned" or self.family not in ("logistic", "xgboost"):
            raise ValueError("invalid tuned model family")
        if self.feature_set not in ("all", "l1", "deep", "compact", "stationary"):
            raise ValueError("invalid tuned feature set")
        if self.class_weight not in (None, "balanced"):
            raise ValueError("class_weight must be null or balanced")
        for name in (
            "max_depth",
            "rounds",
            "patience",
            "horizon_ms",
            "batch_rows",
            "max_train_rows",
            "max_eval_rows",
            "max_fit_mb",
            "timing_batch_size",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an unsigned 32-bit integer")
        for name in (
            "logistic_c",
            "min_child_weight",
            "reg_lambda",
            "tree_learning_rate",
            "probability_mix",
        ):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.probability_mix >= 1:
            raise ValueError("probability_mix must be below one")

    def to_dict(self):
        return asdict(self)


def feature_names(available, feature_set):
    if feature_set == "all":
        return list(available)
    l1 = ["f_relative_spread", "f_obi_1", "f_microprice_minus_mid"]
    if feature_set == "l1":
        names = l1
    elif feature_set == "deep":
        names = l1 + [f"f_obi_{n}" for n in (3, 5, 10)]
        names += [f"f_{side}_depth_{n}" for n in (3, 5, 10) for side in ("bid", "ask")]
    elif feature_set == "compact":
        names = l1 + [f"f_obi_{n}" for n in (3, 5, 10)]
        names += [f"f_trade_imbalance_{w}ms" for w in (100, 500, 1000, 5000)]
        names += [f"f_mid_return_{w}ms" for w in (100, 500, 1000)]
        names += [f"f_obi_1_change_{w}ms" for w in (100, 500, 1000)]
        names += [f"f_depth_flow_imbalance_{w}ms" for w in (100, 500, 1000, 5000)]
    elif feature_set == "stationary":
        removed = {
            "f_midprice",
            "f_microprice",
            *[f"f_trade_vwap_{w}ms" for w in (100, 500, 1000, 5000)],
        }
        names = [n for n in available if n not in removed]
    else:
        raise ValueError("unknown feature set")
    if not names or set(names) - set(available):
        raise ValueError("dataset does not contain the declared feature set")
    return names


def fit_classifier(train_x, train_y, validation_x, validation_y, names, config):
    counts = np.bincount(np.asarray(train_y, dtype=int) + 1, minlength=3)
    if np.any(counts == 0):
        raise ValueError("tuning requires all three classes in each TRAIN fold")
    prior = counts / counts.sum()
    costs = 1 / (3 * prior) if config.class_weight == "balanced" else np.ones(3)
    state = {
        "model": "tuned",
        "config": config.to_dict(),
        "feature_columns": names,
        "prior": prior.tolist(),
        "train_class_counts": counts.tolist(),
    }
    start = time.perf_counter()
    estimator = None
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        if config.family == "logistic":
            fitted = LogisticRegression(
                C=config.logistic_c,
                class_weight=config.class_weight,
                max_iter=2000,
                tol=1e-6,
                random_state=config.seed,
            )
            fitted.fit(train_x, train_y)
            state.update(coef=fitted.coef_.tolist(), intercept=fitted.intercept_.tolist())
            details = {
                "iterations": int(fitted.n_iter_.max()),
                "parameter_count": int(fitted.coef_.size + fitted.intercept_.size),
            }
        else:
            train = xgb.DMatrix(
                train_x,
                label=train_y + 1,
                weight=costs[train_y + 1],
                feature_names=names,
                nthread=1,
            )
            validation = xgb.DMatrix(
                validation_x,
                label=validation_y + 1,
                weight=costs[validation_y + 1],
                feature_names=names,
                nthread=1,
            )
            estimator = xgb.train(
                {
                    "objective": "multi:softprob",
                    "num_class": 3,
                    "eval_metric": "mlogloss",
                    "tree_method": "hist",
                    "nthread": 1,
                    "seed": config.seed,
                    "max_depth": config.max_depth,
                    "eta": config.tree_learning_rate,
                    "min_child_weight": config.min_child_weight,
                    "lambda": config.reg_lambda,
                },
                train,
                num_boost_round=config.rounds,
                evals=[(validation, "validation")],
                early_stopping_rounds=config.patience,
                verbose_eval=False,
            )
            estimator = estimator[: estimator.best_iteration + 1]
            state["selected_rounds"] = estimator.num_boosted_rounds()
            details = {"selected_rounds": state["selected_rounds"]}
    details.update(
        status="fitted",
        optimization_seconds=time.perf_counter() - start,
        rows=len(train_y),
        validation_rows=len(validation_y),
        train_class_counts=counts.tolist(),
        class_costs=costs.tolist(),
        probability_policy="inverse class-cost correction using TRAIN priors; "
        "hard decisions retain cost-sensitive argmax"
        if config.class_weight
        else "argmax",
    )
    return state, estimator, details


class FrozenTuned:
    def __init__(self, state, prep, estimator=None):
        self.state, self.preprocessor, self.estimator = state, prep, estimator
        self.config = TunedConfig(**state["config"])
        self.indices = [prep["feature_columns"].index(n) for n in state["feature_columns"]]
        self.input_preprocessor = {
            "feature_columns": state["feature_columns"],
            "impute_mean": [prep["impute_mean"][i] for i in self.indices],
            "scale": [prep["scale"][i] for i in self.indices],
        }
        self.prior = np.asarray(state["prior"], dtype=float)
        if (
            self.prior.shape != (3,)
            or np.any(self.prior <= 0)
            or not np.isfinite(self.prior).all()
            or not np.isclose(self.prior.sum(), 1)
        ):
            raise ValueError("invalid frozen TRAIN class priors")
        if self.config.family == "logistic":
            self.coef = np.asarray(state["coef"], dtype=float)
            self.intercept = np.asarray(state["intercept"], dtype=float)
            if (
                self.coef.shape != (3, len(self.indices))
                or self.intercept.shape != (3,)
                or not np.isfinite(self.coef).all()
                or not np.isfinite(self.intercept).all()
            ):
                raise ValueError("invalid frozen classifier coefficients")

    def predict_matrix(self, matrix):
        if self.config.family == "logistic":
            scores = matrix @ self.coef.T + self.intercept
            scores -= scores.max(axis=1, keepdims=True)
            q = np.exp(scores)
        else:
            q = self.estimator.predict(
                xgb.DMatrix(matrix, feature_names=self.state["feature_columns"], nthread=1)
            ).astype(float)
        q /= q.sum(axis=1, keepdims=True)
        prediction = (q.argmax(axis=1) - 1).astype(np.int8)
        p = q * self.prior if self.config.class_weight == "balanced" else q
        p /= p.sum(axis=1, keepdims=True)
        p = (1 - self.config.probability_mix) * p + self.config.probability_mix / 3
        if p.shape != (len(matrix), 3) or not np.isfinite(p).all() or np.any(p <= 0):
            raise ValueError("invalid tuned probabilities")
        return prediction, p

    def predict(self, table):
        matrix = float32_matrix(table, self.input_preprocessor)
        return self.predict_matrix(np.ascontiguousarray(matrix))


def load_estimator(run, state):
    if state["config"]["family"] == "logistic":
        return None
    estimator = xgb.Booster({"nthread": 1})
    estimator.load_model(Path(run) / "tree.ubj")
    if (
        estimator.feature_names != state["feature_columns"]
        or estimator.num_boosted_rounds() != state["selected_rounds"]
    ):
        raise ValueError("tuned tree differs from its frozen schema")
    return estimator


def load_tuned_run(run):
    run = Path(run)
    manifest = read_json(run / "manifest.json")
    if (
        (run / "_INCOMPLETE").exists()
        or manifest.get("kind") != "tuned_run"
        or manifest.get("status") != "complete"
    ):
        raise ValueError("require a committed tuned run")
    for name, info in manifest["outputs"].items():
        artifact(run, name, info["sha256"])
    state, prep = read_json(run / "model.json"), read_json(run / "preprocessor.json")
    config = TunedConfig(**read_json(run / "config.json"))
    if (
        state["config"] != config.to_dict()
        or state["feature_columns"] != manifest["feature_columns"]
        or config.horizon_ms != manifest["horizon_ms"]
        or file_hash(run / "dataset_manifest.json") != manifest["dataset"]["manifest_sha256"]
    ):
        raise ValueError("inconsistent tuned model artifacts")
    return FrozenTuned(state, prep, load_estimator(run, state)), manifest


def train_tuned_run(dataset, destination, config):
    dataset = Path(dataset)
    manifest, digest = check_dataset(dataset)
    suffix = f"{config.horizon_ms}ms"
    if config.horizon_ms not in manifest["horizons_ms"]:
        raise ValueError("unknown tuned horizon")
    counts = manifest["target_counts"][suffix]
    names = feature_names(manifest["feature_columns"], config.feature_set)
    for split, limit in (("train", config.max_train_rows), ("validation", config.max_eval_rows)):
        size = counts[split]["rows"]
        if (
            size <= 0
            or size > limit
            or size * (len(manifest["feature_columns"]) * 16 + 64) > config.max_fit_mb * 1024**2
        ):
            raise ValueError("tuned partition exceeds its resource limit or is empty")
    prep_name = manifest["normalizers"][suffix]
    prep_path = artifact(dataset, prep_name, manifest["outputs"][prep_name]["sha256"])
    prep = read_json(prep_path)
    bounds = manifest["ranges"]["train"]
    if (
        prep["feature_columns"] != manifest["feature_columns"]
        or prep["horizon_ms"] != config.horizon_ms
        or prep["fit_split"] != "train"
        or prep["fit_rows"] != counts["train"]["rows"]
        or prep["train_start_ns"] != bounds["start_ns"]
        or prep["train_end_exclusive_ns"] != bounds["end_exclusive_ns"]
    ):
        raise ValueError("tuned preprocessor does not match TRAIN")
    destination = begin(destination)
    try:
        _write_json(destination / "config.json", config.to_dict())
        shutil.copyfile(prep_path, destination / "preprocessor.json")
        shutil.copyfile(dataset / "manifest.json", destination / "dataset_manifest.json")
        x, y, _, _, _ = read_partition(dataset, manifest, prep, config, "train")
        vx, vy, _, _, _ = read_partition(dataset, manifest, prep, config, "validation")
        indices = [prep["feature_columns"].index(n) for n in names]
        state, estimator, fit = fit_classifier(x[:, indices], y, vx[:, indices], vy, names, config)
        _write_json(destination / "model.json", state)
        if estimator is not None:
            estimator.save_model(destination / "tree.ubj")
        frozen = FrozenTuned(
            read_json(destination / "model.json"), prep, load_estimator(destination, state)
        )
        expected, probabilities = FrozenTuned(state, prep, estimator).predict_matrix(vx[:, indices])
        actual, saved = frozen.predict_matrix(vx[:, indices])
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_allclose(saved, probabilities, rtol=0, atol=1e-14)
        metrics = evaluate_partition(
            dataset, manifest, frozen, config, "validation", destination / "validation"
        )
        check_dataset(dataset, digest)
        for split in ("train", "validation"):
            for name in (f"{split}/features.parquet", f"{split}/labels_{suffix}.parquet"):
                artifact(dataset, name, manifest["outputs"][name]["sha256"])
        return commit(
            destination,
            {
                "kind": "tuned_run",
                "status": "complete",
                "schema_version": 1,
                "created_utc": datetime.now(UTC).isoformat(),
                "model": "tuned",
                "horizon_ms": config.horizon_ms,
                "feature_columns": names,
                "active_feature_columns": names,
                "ranges": manifest["ranges"],
                "dataset": {"path": str(dataset.resolve()), "manifest_sha256": digest},
                "fit": fit,
                "validation_rows": metrics["rows"],
                "test_accessed": False,
                "selection_policy": "configuration selected on development walk-forward folds",
                "software": software(),
            },
        )
    except Exception as exc:
        fail(destination, exc)
        raise
