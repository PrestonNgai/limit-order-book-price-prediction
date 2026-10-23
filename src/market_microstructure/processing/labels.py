"""Future-only targets with exact decimal class comparisons and explicit eligibility."""

from decimal import Decimal, localcontext

import numpy as np
import pyarrow as pa

from .feature_config import positive_decimal
from .features import validate_observations


def label_table(observations, config, interval_ms, tick_size):
    config.validate_interval(interval_ms)
    validate_observations(observations, interval_ms)
    tick = positive_decimal(tick_size, "tick_size")
    times = observations["timestamp_ns"].to_numpy()
    mid = np.array(observations["midprice"].to_pylist(), dtype=object)
    if any(not isinstance(value, Decimal) or not value.is_finite() or value <= 0 for value in mid):
        raise ValueError("labels require exact, positive Decimal midprices")
    columns = {}
    with localcontext() as ctx:
        ctx.prec = 100
        threshold = (
            mid * positive_decimal(config.relative_epsilon, "relative_epsilon")
            if config.relative_epsilon is not None
            else np.full(len(mid), tick * Decimal(str(config.epsilon_ticks)), dtype=object)
        )
        for horizon in config.horizons_ms:
            offset = horizon * 1_000_000
            tolerance = config.alignment_tolerance_ms * 1_000_000
            if int(times[-1]) + offset + tolerance >= 2**63:
                raise ValueError("label timestamp overflow")
            target_time = times + offset
            indexes = np.searchsorted(times, target_time, side="left")
            safe = np.minimum(indexes, len(times) - 1)
            valid = (indexes < len(times)) & (times[safe] <= target_time + tolerance)
            delta = mid[safe] - mid
            returns = np.asarray(delta / mid, dtype=float)
            if np.any(~np.isfinite(returns)):
                raise ValueError("return overflow")
            classes = np.where(delta > threshold, 1, np.where(delta < -threshold, -1, 0)).astype(
                np.int8
            )
            suffix = f"{horizon}ms"
            columns[f"label_valid_{suffix}"] = pa.array(valid)
            columns[f"label_end_ns_{suffix}"] = pa.array(times[safe], mask=~valid)
            columns[f"y_return_{suffix}"] = pa.array(returns, mask=~valid)
            columns[f"y_class_{suffix}"] = pa.array(classes, mask=~valid)
    return pa.table(columns)


def eligible_target(table, horizon_ms):
    """Drop unavailable endpoints per horizon; never encode missing future prices as STABLE."""
    return table.filter(table[f"label_valid_{horizon_ms}ms"])


class TrainingDiagnostics:
    """Streaming descriptive statistics restricted to an explicit [start, end) period."""

    def __init__(self, config, start_ns, end_ns):
        if (
            type(start_ns) is not int
            or type(end_ns) is not int
            or not 0 <= start_ns < end_ns < 2**63
        ):
            raise ValueError("training diagnostics require 0 <= start < end < 2**63")
        self.config, self.start_ns, self.end_ns = config, start_ns, end_ns
        self.stats = {
            str(h): {
                "eligible_rows": 0,
                "class_counts": {"DOWN": 0, "STABLE": 0, "UP": 0},
                "sum_absolute_return": 0.0,
                "max_absolute_return": 0.0,
            }
            for h in config.horizons_ms
        }

    def consume(self, table):
        times = table["timestamp_ns"].to_numpy()
        for horizon in self.config.horizons_ms:
            suffix = f"{horizon}ms"
            ends = np.array(table[f"label_end_ns_{suffix}"].fill_null(2**63 - 1))
            valid = (times >= self.start_ns) & (times < self.end_ns) & (ends < self.end_ns)
            target = table[f"y_return_{suffix}"].to_numpy()[valid]
            classes = table[f"y_class_{suffix}"].to_numpy()[valid]
            if not len(target):
                continue
            stat = self.stats[str(horizon)]
            stat["eligible_rows"] += len(target)
            stat["sum_absolute_return"] += float(np.abs(target).sum())
            stat["max_absolute_return"] = max(
                stat["max_absolute_return"], float(np.abs(target).max())
            )
            for label, value in (("DOWN", -1), ("STABLE", 0), ("UP", 1)):
                stat["class_counts"][label] += int((classes == value).sum())

    def report(self):
        return {
            "training_start_ns": self.start_ns,
            "training_end_exclusive_ns": self.end_ns,
            "endpoint_policy": "both decision and label endpoint inside TRAIN",
            "threshold_selection": "fixed beforehand; diagnostics do not auto-tune epsilon",
            "horizons_ms": {
                h: {
                    "eligible_rows": s["eligible_rows"],
                    "class_counts": s["class_counts"],
                    "mean_absolute_return": (
                        s["sum_absolute_return"] / s["eligible_rows"]
                        if s["eligible_rows"]
                        else None
                    ),
                    "max_absolute_return": s["max_absolute_return"] if s["eligible_rows"] else None,
                }
                for h, s in self.stats.items()
            },
        }
