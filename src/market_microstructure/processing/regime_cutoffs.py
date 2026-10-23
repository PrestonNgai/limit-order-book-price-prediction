"""Exact TRAIN quantiles with a disk-backed index instead of unlimited arrays."""

import math
import sqlite3
from pathlib import Path

import numpy as np
import pyarrow as pa

from .preprocessing import require_train

METRICS = {
    "absolute_return_1000ms": "abs(f_mid_return_1000ms), a volatility proxy",
    "spread": "f_spread",
    "trade_count_1000ms": "f_trade_count_1000ms",
    "depth_10": "f_bid_depth_10 + f_ask_depth_10",
}


def regime_values(table):
    def column(name):
        return (
            table[name].to_numpy()
            if name in table.column_names
            else np.full(table.num_rows, np.nan)
        )

    return {
        "absolute_return_1000ms": np.abs(column("f_mid_return_1000ms")),
        "spread": column("f_spread"),
        "trade_count_1000ms": column("f_trade_count_1000ms"),
        "depth_10": column("f_bid_depth_10") + column("f_ask_depth_10"),
    }


class TrainRegimeQuantiles:
    def __init__(self, path, horizon_ms, start_ns, end_ns, quantiles):
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(self.path)
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA cache_size=-4096")
        self.db.execute("PRAGMA temp_store=FILE")
        self.db.execute("CREATE TABLE samples (metric TEXT NOT NULL, value REAL NOT NULL)")
        self.db.execute("CREATE INDEX ordered_samples ON samples(metric,value)")
        self.horizon_ms, self.start_ns, self.end_ns = horizon_ms, start_ns, end_ns
        self.quantiles, self.rows = tuple(quantiles), 0

    def update(self, table):
        require_train(table, self.horizon_ms, self.start_ns, self.end_ns)
        for name, values in regime_values(table).items():
            if np.any(np.isinf(values)):
                raise ValueError("infinite regime value")
            self.db.executemany(
                "INSERT INTO samples VALUES (?,?)",
                ((name, float(v)) for v in values[np.isfinite(values)]),
            )
        self.db.commit()
        self.rows += table.num_rows

    def finish(self):
        result = {
            "fit_split": "train",
            "horizon_ms": self.horizon_ms,
            "train_start_ns": self.start_ns,
            "train_end_exclusive_ns": self.end_ns,
            "fit_rows": self.rows,
            "quantiles": list(self.quantiles),
            "method": "exact linear-interpolated empirical quantiles; observed TRAIN values only",
            "metrics": {},
        }
        for name, formula in METRICS.items():
            count = self.db.execute(
                "SELECT count(*) FROM samples WHERE metric=?", (name,)
            ).fetchone()[0]
            thresholds = []
            for q in self.quantiles:
                if not count:
                    thresholds.append(None)
                    continue
                rank = (count - 1) * q
                left, right = math.floor(rank), math.ceil(rank)

                def value_at(offset, metric=name):
                    return self.db.execute(
                        "SELECT value FROM samples WHERE metric=? ORDER BY value LIMIT 1 OFFSET ?",
                        (metric, offset),
                    ).fetchone()[0]

                lo, hi = value_at(left), value_at(right)
                thresholds.append(lo + (rank - left) * (hi - lo))
            result["metrics"][name] = {
                "formula": formula,
                "observed_rows": count,
                "thresholds": thresholds,
            }
        self.close()
        self.path.unlink()  # Remove the temporary quantile index.
        return result

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None


def apply_regimes(table, state):
    output = {}
    for name, values in regime_values(table).items():
        thresholds = state["metrics"][name]["thresholds"]
        known = np.isfinite(values)
        if any(v is None for v in thresholds):
            known[:] = False
            thresholds = []
        # Ties enter the lower bin. Unknown values remain Arrow nulls.
        bins = np.searchsorted(thresholds, values, side="left").astype(np.int32)
        output["regime_" + name] = pa.array(bins, mask=~known)
    return pa.table(output)
