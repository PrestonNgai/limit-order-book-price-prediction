"""Mean imputation and population standardization fitted only on purged TRAIN rows."""

import hashlib

import numpy as np


def require_train(table, horizon_ms, start_ns, end_ns):
    if table["split"].null_count or set(table["split"].to_pylist()) - {"train"}:
        raise ValueError("fitting is restricted to TRAIN")
    required = ["timestamp_ns", f"label_end_ns_{horizon_ms}ms", "row_id"]
    if any(table[name].null_count for name in required):
        raise ValueError("null fit timestamp or row identifier")
    times = table["timestamp_ns"].to_numpy()
    if np.any(np.diff(times) <= 0):
        raise ValueError("fit observations must be ordered and unique")
    ends = table[f"label_end_ns_{horizon_ms}ms"].to_numpy()
    if np.any((times < start_ns) | (times >= end_ns) | (ends <= times) | (ends >= end_ns)):
        raise ValueError("fit rows or future endpoints escape the TRAIN boundary")


def feature_matrix(table, names):
    if not names or len(names) != len(set(names)) or any(not n.startswith("f_") for n in names):
        raise ValueError("explicit unique f_ feature names are required")
    values = np.column_stack([table[name].to_numpy() for name in names]).astype(np.float64)
    if np.any(np.isinf(values)):
        raise ValueError("infinite feature values are not valid missing data")
    return values


class TrainPreprocessor:
    def __init__(self, feature_names, horizon_ms, start_ns, end_ns):
        self.names = list(feature_names)
        if (
            not self.names
            or len(set(self.names)) != len(self.names)
            or any(not n.startswith("f_") for n in self.names)
        ):
            raise ValueError("only explicit unique feature columns may be fitted")
        self.horizon_ms, self.start_ns, self.end_ns = horizon_ms, start_ns, end_ns
        self.count = np.zeros(len(self.names), dtype=np.int64)
        self.mean, self.m2 = np.zeros(len(self.names)), np.zeros(len(self.names))
        self.rows = 0
        self.minimum_time = self.maximum_time = self.maximum_endpoint = None
        self.digest = hashlib.sha256()
        self.state = None

    def update(self, table):
        if self.state is not None:
            raise ValueError("a frozen preprocessor cannot be refitted")
        require_train(table, self.horizon_ms, self.start_ns, self.end_ns)
        if not table.num_rows:
            return
        if self.maximum_time is not None and table["timestamp_ns"][0].as_py() <= self.maximum_time:
            raise ValueError("fit batches must increase chronologically")
        values = feature_matrix(table, self.names)
        finite = np.isfinite(values)
        count = finite.sum(axis=0)
        origin = values[finite.argmax(axis=0), np.arange(values.shape[1])]
        origin = np.where(count > 0, origin, 0)
        differences = np.where(finite, values - origin, 0).sum(axis=0)
        mean = origin + np.divide(differences, count, out=np.zeros_like(origin), where=count > 0)
        centered = np.where(finite, values - mean, 0)
        m2 = np.square(centered).sum(axis=0)
        total = self.count + count
        delta = mean - self.mean
        self.mean += np.divide(delta * count, total, out=np.zeros_like(delta), where=total > 0)
        correction = np.divide(
            self.count.astype(float) * count, total, out=np.zeros_like(delta), where=total > 0
        )
        self.m2 += m2 + np.square(delta) * correction
        self.count = total
        if np.any(~np.isfinite(self.mean)) or np.any(~np.isfinite(self.m2)):
            raise ValueError("training statistics overflow")
        times = table["timestamp_ns"].to_numpy()
        ends = table[f"label_end_ns_{self.horizon_ms}ms"].to_numpy()
        self.minimum_time = (
            int(times.min())
            if self.minimum_time is None
            else min(self.minimum_time, int(times.min()))
        )
        self.maximum_time = (
            int(times.max())
            if self.maximum_time is None
            else max(self.maximum_time, int(times.max()))
        )
        self.maximum_endpoint = (
            int(ends.max())
            if self.maximum_endpoint is None
            else max(self.maximum_endpoint, int(ends.max()))
        )
        # Stable binary row layout gives a batch-independent fingerprint of fit inputs.
        fingerprint = np.empty(
            len(values),
            dtype=[
                ("row", "<i8"),
                ("time", "<i8"),
                ("endpoint", "<i8"),
                ("target", "<f8"),
                ("class", "i1"),
                ("features", "<f8", (len(self.names),)),
            ],
        )
        fingerprint["row"] = table["row_id"].to_numpy()
        fingerprint["time"], fingerprint["endpoint"] = times, ends
        fingerprint["target"] = table[f"y_return_{self.horizon_ms}ms"].to_numpy()
        fingerprint["class"] = table[f"y_class_{self.horizon_ms}ms"].to_numpy()
        fingerprint["features"] = np.where(finite, values, np.nan)
        self.digest.update(fingerprint.tobytes())
        self.rows += len(values)

    def finalize(self):
        if self.state is not None:
            return self.state
        if not self.rows:
            raise ValueError("no eligible TRAIN rows; preprocessing cannot be fitted")
        # Mean-imputed entries contribute zero squared deviation, but count in N.
        variance = np.maximum(self.m2 / self.rows, 0)
        scale = np.sqrt(variance)
        scale[scale == 0] = 1
        self.state = {
            "schema_version": 1,
            "fit_split": "train",
            "horizon_ms": self.horizon_ms,
            "train_start_ns": self.start_ns,
            "train_end_exclusive_ns": self.end_ns,
            "fit_rows": self.rows,
            "fit_min_time_ns": self.minimum_time,
            "fit_max_time_ns": self.maximum_time,
            "fit_max_endpoint_ns": self.maximum_endpoint,
            "fit_row_sha256": self.digest.hexdigest(),
            "feature_columns": self.names,
            "observed_counts": self.count.tolist(),
            "impute_mean": self.mean.tolist(),
            "scale": scale.tolist(),
            "variance_after_mean_imputation": variance.tolist(),
            "all_missing_features": [
                n for n, c in zip(self.names, self.count, strict=True) if c == 0
            ],
            "policy": (
                "TRAIN observed means; all-missing=0; population variance after "
                "mean imputation; zero-scale=1"
            ),
        }
        return self.state

    @staticmethod
    def transform(table, state):
        names = state["feature_columns"]
        values = feature_matrix(table, names)
        mean, scale = np.array(state["impute_mean"]), np.array(state["scale"])
        if (
            mean.shape != (len(names),)
            or scale.shape != mean.shape
            or np.any(~np.isfinite(mean))
            or np.any(~np.isfinite(scale))
            or np.any(scale <= 0)
        ):
            raise ValueError("invalid frozen preprocessing parameters")
        result = (np.where(np.isnan(values), mean, values) - mean) / scale
        if np.any(~np.isfinite(result)):
            raise ValueError("transformed feature overflow")
        return result
