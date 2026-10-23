"""Future-return regression with frozen TRAIN transforms and explicit TEST access."""

import hashlib
import math
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from market_microstructure.processing.build_dataset import file_hash
from market_microstructure.processing.preprocessing import TrainPreprocessor
from market_microstructure.processing.research_dataset import _write_json
from market_microstructure.processing.split_dataset import iter_model_batches

from .experiments import artifact, begin, check_dataset, commit, fail, read_json, software


@dataclass(frozen=True)
class RegressionConfig:
    model: str = "ridge"
    horizon_ms: int = 500
    seed: int = 42
    ridge_alpha: float = 1.0
    rounds: int = 200
    max_depth: int = 3
    learning_rate: float = 0.05
    patience: int = 15
    batch_rows: int = 4096
    max_rows: int = 250000
    max_fit_mb: int = 512

    def __post_init__(self):
        if self.model not in ("zero", "ridge", "xgboost"):
            raise ValueError("unknown regression model")
        for name in (
            "horizon_ms",
            "rounds",
            "max_depth",
            "patience",
            "batch_rows",
            "max_rows",
            "max_fit_mb",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an unsigned 32-bit integer")
        for name in ("ridge_alpha", "learning_rate"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


def regression_metrics(target, prediction):
    y, p = np.asarray(target, dtype=float), np.asarray(prediction, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("finite aligned regression vectors are required")
    if not len(y):
        return {"status": "unavailable", "rows": 0}
    y, p = y * 10000, p * 10000
    error = p - y
    total = float(np.square(y - y.mean()).sum())
    zero_error = float(np.square(y).sum())
    directional = p != 0
    return {
        "status": "complete",
        "rows": len(y),
        "mae_bps": float(np.abs(error).mean()),
        "rmse_bps": float(np.sqrt(np.square(error).mean())),
        "mean_error_bps": float(error.mean()),
        "r2": 1 - float(np.square(error).sum()) / total if total > 0 else None,
        "mse_skill_vs_zero": 1 - float(np.square(error).sum()) / zero_error
        if zero_error > 0
        else None,
        "pearson_correlation": float(np.corrcoef(y, p)[0, 1])
        if len(y) > 1 and np.ptp(y) > 0 and np.ptp(p) > 0
        else None,
        "directional_coverage": float(directional.mean()),
        "directional_accuracy": float((np.sign(y[directional]) == np.sign(p[directional])).mean())
        if directional.any()
        else None,
        "units": "errors in basis points; predictions and targets saved as relative returns",
    }


def read_partition(dataset, manifest, prep, config, split, *, allow_test=False):
    count = manifest["target_counts"][f"{config.horizon_ms}ms"][split]["rows"]
    width = len(prep["feature_columns"])
    if count > config.max_rows or count * (width * 8 * 4 + 64) > config.max_fit_mb * 1024**2:
        raise ValueError("regression partition exceeds the configured resource limits")
    pieces, seen = [], 0
    for piece in iter_model_batches(
        dataset,
        config.horizon_ms,
        split,
        batch_rows=config.batch_rows,
        transformed=False,
        allow_test=allow_test,
    ):
        seen += len(piece)
        if seen > count:
            raise ValueError("regression rows exceed the split manifest")
        pieces.append(piece)
    if not pieces:
        raise ValueError(f"no eligible {split} regression targets")
    table = pa.concat_tables(pieces)
    if len(table) != count:
        raise ValueError("regression rows differ from the split manifest")
    if split == "train":
        bound = manifest["ranges"]["train"]
        checker = TrainPreprocessor(
            prep["feature_columns"], config.horizon_ms, bound["start_ns"], bound["end_exclusive_ns"]
        )
        for piece in pieces:
            checker.update(piece)
        if checker.finalize()["fit_row_sha256"] != prep["fit_row_sha256"]:
            raise ValueError("regression TRAIN differs from preprocessing provenance")
    matrix = TrainPreprocessor.transform(table, prep)
    target = table[f"y_return_{config.horizon_ms}ms"].to_numpy()
    if not np.isfinite(target).all():
        raise ValueError("nonfinite future-return targets")
    return table, matrix, target


class FrozenRegressor:
    def __init__(self, state, prep, estimator=None):
        self.state, self.preprocessor, self.estimator = state, prep, estimator
        if state["config"]["model"] == "ridge":
            coef = np.asarray(state["coef"], dtype=float)
            if (
                coef.shape != (len(prep["feature_columns"]),)
                or not np.isfinite(coef).all()
                or not np.isfinite(state["intercept"])
            ):
                raise ValueError("invalid frozen ridge coefficients")

    def predict(self, matrix):
        if self.state["config"]["model"] == "zero":
            result = np.zeros(len(matrix))
        elif self.state["config"]["model"] == "ridge":
            result = matrix @ np.asarray(self.state["coef"]) + self.state["intercept"]
        else:
            import xgboost as xgb

            result = self.estimator.predict(
                xgb.DMatrix(matrix, feature_names=self.preprocessor["feature_columns"])
            )
        result = np.asarray(result, dtype=float) / 10000
        if result.shape != (len(matrix),) or not np.isfinite(result).all():
            raise ValueError("invalid regression predictions")
        return result


def load_regression_run(run):
    run = Path(run)
    manifest = read_json(run / "manifest.json")
    if (
        (run / "_INCOMPLETE").exists()
        or manifest.get("kind") != "regression_run"
        or manifest.get("status") != "complete"
    ):
        raise ValueError("require a committed regression run")
    for name, info in manifest["outputs"].items():
        artifact(run, name, info["sha256"])
    state, prep = read_json(run / "model.json"), read_json(run / "preprocessor.json")
    config = RegressionConfig(**state["config"])
    if (
        state["config"] != read_json(run / "config.json")
        or prep["feature_columns"] != manifest["feature_columns"]
        or config.horizon_ms != manifest["horizon_ms"]
        or config.model != manifest["model"]
        or file_hash(run / "dataset_manifest.json") != manifest["dataset"]["manifest_sha256"]
    ):
        raise ValueError("inconsistent regression artifacts")
    return load_saved_regressor(run, state, prep), manifest


def load_saved_regressor(run, state, prep):
    config = RegressionConfig(**state["config"])
    estimator = None
    if config.model == "xgboost":
        import xgboost as xgb

        estimator = xgb.Booster({"nthread": 1})
        estimator.load_model(run / "tree.ubj")
        if (
            estimator.feature_names != prep["feature_columns"]
            or estimator.num_boosted_rounds() != state["selected_rounds"]
        ):
            raise ValueError("regression tree schema or checkpoint differs")
    return FrozenRegressor(state, prep, estimator)


def save_predictions(destination, table, target, prediction, horizon, split):
    destination.mkdir()
    metrics = regression_metrics(target, prediction)
    predictions = pa.table(
        {
            "row_id": table["row_id"],
            "timestamp_ns": table["timestamp_ns"],
            "label_end_ns": table[f"label_end_ns_{horizon}ms"],
            "split": table["split"],
            "target_return": target,
            "prediction_return": prediction,
        }
    )
    pq.write_table(predictions, destination / "predictions.parquet", compression="zstd")
    _write_json(destination / "metrics.json", {**metrics, "split": split})
    return metrics


def train_regression_run(dataset, destination, config):
    dataset = Path(dataset)
    manifest, digest = check_dataset(dataset)
    suffix = f"{config.horizon_ms}ms"
    if config.horizon_ms not in manifest["horizons_ms"]:
        raise ValueError("unknown regression horizon")
    name = manifest["normalizers"][suffix]
    if name is None:
        raise ValueError("missing regression TRAIN preprocessor")
    prep_path = artifact(dataset, name, manifest["outputs"][name]["sha256"])
    prep = read_json(prep_path)
    bound = manifest["ranges"]["train"]
    if (
        prep["horizon_ms"] != config.horizon_ms
        or prep["feature_columns"] != manifest["feature_columns"]
        or prep["fit_rows"] != manifest["target_counts"][suffix]["train"]["rows"]
        or prep["train_start_ns"] != bound["start_ns"]
        or prep["train_end_exclusive_ns"] != bound["end_exclusive_ns"]
    ):
        raise ValueError("regression preprocessor differs from TRAIN provenance")
    destination = begin(destination)
    try:
        _write_json(destination / "config.json", asdict(config))
        shutil.copyfile(prep_path, destination / "preprocessor.json")
        shutil.copyfile(dataset / "manifest.json", destination / "dataset_manifest.json")
        train_table, x, y = read_partition(dataset, manifest, prep, config, "train")
        val_table, vx, vy = read_partition(dataset, manifest, prep, config, "validation")
        state, estimator = {"config": asdict(config)}, None
        start = time.perf_counter()
        with threadpool_limits(limits=1):
            if config.model == "ridge":
                fitted = Ridge(alpha=config.ridge_alpha, solver="svd").fit(x, y * 10000)
                state.update(coef=fitted.coef_.tolist(), intercept=float(fitted.intercept_))
            elif config.model == "xgboost":
                import xgboost as xgb

                train = xgb.DMatrix(x, label=y * 10000, feature_names=prep["feature_columns"])
                validation = xgb.DMatrix(
                    vx, label=vy * 10000, feature_names=prep["feature_columns"]
                )
                estimator = xgb.train(
                    {
                        "objective": "reg:squarederror",
                        "eval_metric": "rmse",
                        "tree_method": "hist",
                        "nthread": 1,
                        "seed": config.seed,
                        "max_depth": config.max_depth,
                        "eta": config.learning_rate,
                    },
                    train,
                    num_boost_round=config.rounds,
                    evals=[(validation, "validation")],
                    early_stopping_rounds=config.patience,
                    verbose_eval=False,
                )
                estimator = estimator[: estimator.best_iteration + 1]
                state["selected_rounds"] = estimator.num_boosted_rounds()
                estimator.save_model(destination / "tree.ubj")
        elapsed = time.perf_counter() - start
        _write_json(destination / "model.json", state)
        frozen = FrozenRegressor(state, prep, estimator)
        prediction = frozen.predict(vx)
        reloaded = load_saved_regressor(destination, read_json(destination / "model.json"), prep)
        if not np.allclose(reloaded.predict(vx), prediction, rtol=0, atol=1e-15):
            raise ValueError("regression reload prediction mismatch")
        metrics = save_predictions(
            destination / "validation", val_table, vy, prediction, config.horizon_ms, "validation"
        )
        check_dataset(dataset, digest)
        for split in ("train", "validation"):
            for name in (f"{split}/features.parquet", f"{split}/labels_{suffix}.parquet"):
                artifact(dataset, name, manifest["outputs"][name]["sha256"])
        fingerprint = hashlib.sha256()
        fingerprint.update(train_table["row_id"].to_numpy().astype("<i8").tobytes())
        fingerprint.update(y.astype("<f8").tobytes())
        return commit(
            destination,
            {
                "schema_version": 1,
                "kind": "regression_run",
                "status": "complete",
                "created_utc": datetime.now(UTC).isoformat(),
                "model": config.model,
                "horizon_ms": config.horizon_ms,
                "feature_columns": prep["feature_columns"],
                "dataset": {"path": str(dataset.resolve()), "manifest_sha256": digest},
                "ranges": manifest["ranges"],
                "train_rows": len(y),
                "validation_rows": metrics["rows"],
                "train_target_sha256": fingerprint.hexdigest(),
                "fit_seconds": elapsed,
                "target_units": "relative return; fitted internally in basis points",
                "selection_policy": "fixed zero/ridge; validation early stopping for XGBoost",
                "test_accessed": False,
                "software": software(),
            },
        )
    except Exception as exc:
        fail(destination, exc)
        raise


def evaluate_regression_run(run, dataset, destination, *, split="validation", allow_test=False):
    if split not in ("validation", "test"):
        raise ValueError("regression evaluation requires validation or test")
    if split == "test" and not allow_test:
        raise PermissionError("TEST is sealed; use explicit final-test opt-in")
    frozen, trained = load_regression_run(run)
    manifest, digest = check_dataset(dataset, trained["dataset"]["manifest_sha256"])
    config = RegressionConfig(**frozen.state["config"])
    destination = begin(destination)
    try:
        table, matrix, y = read_partition(
            dataset, manifest, frozen.preprocessor, config, split, allow_test=allow_test
        )
        prediction = frozen.predict(matrix)
        metrics = save_predictions(
            destination / split, table, y, prediction, config.horizon_ms, split
        )
        check_dataset(dataset, digest)
        return commit(
            destination,
            {
                "schema_version": 1,
                "kind": "regression_evaluation",
                "status": "complete",
                "created_utc": datetime.now(UTC).isoformat(),
                "dataset_manifest_sha256": digest,
                "run_manifest_sha256": file_hash(Path(run) / "manifest.json"),
                "split": split,
                "rows": metrics["rows"],
                "test_accessed": split == "test",
            },
        )
    except Exception as exc:
        fail(destination, exc)
        raise
