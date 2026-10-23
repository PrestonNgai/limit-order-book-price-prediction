"""Vectorized features from a bounded, regular, single-segment observation table."""

import numpy as np
import pyarrow as pa

from .feature_config import FeatureConfig

IDENTIFIERS = (
    "timestamp_ns",
    "timestamp",
    "source",
    "symbol",
    "run_id",
    "session_id",
    "segment_id",
)


def numbers(table, name):
    return table[name].cast(pa.float64()).to_numpy(zero_copy_only=False)


def safe_divide(numerator, denominator, *, zero_value=np.nan):
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    output = np.full(numerator.shape, np.nan, dtype=np.float64)
    good = np.isfinite(numerator) & np.isfinite(denominator)
    np.divide(numerator, denominator, out=output, where=good & (denominator > 0))
    output[good & (denominator == 0)] = zero_value
    return output


def lag(values, steps):
    result = np.full(len(values), np.nan)
    if steps < len(values):
        result[steps:] = values[:-steps]
    return result


def rolling_sum(values, steps):
    """Full windows only; a missing observation invalidates its whole window."""
    result = np.full(len(values), np.nan)
    if steps > len(values):
        return result
    # Prefix-sum subtraction loses small trades after a large execution.
    # Direct window sums cost O(rows * window) but avoid that cancellation.
    windows = np.lib.stride_tricks.sliding_window_view(values, steps)
    result[steps - 1 :] = windows.sum(axis=-1)
    return result


def validate_observations(table, interval_ms):
    if not table.num_rows:
        raise ValueError("observation block must not be empty")
    for name in ("source", "symbol", "run_id", "session_id", "segment_id"):
        if table[name].null_count or len(table[name].unique()) != 1:
            raise ValueError("each observation block must have one source/product/segment")
    times = table["timestamp_ns"].to_numpy()
    if np.any(np.diff(times) != interval_ms * 1_000_000):
        raise ValueError("missing, duplicate or unordered timestamp inside segment")
    if np.any(times < 0):
        raise ValueError("timestamps must be nonnegative")


def feature_arrays(table, config: FeatureConfig, interval_ms):
    config.validate_interval(interval_ms)
    validate_observations(table, interval_ms)
    result, groups = {}, {}

    def add(name, value, group):
        key = "f_" + name
        if np.any(np.isinf(value)):
            raise ValueError(f"nonfinite feature overflow: {key}")
        result[key] = value
        groups.setdefault(group, []).append(key)

    mid, spread = numbers(table, "midprice"), numbers(table, "spread")
    if np.any(~np.isfinite(mid) | (mid <= 0) | ~np.isfinite(spread) | (spread <= 0)):
        raise ValueError("midprice and spread must be finite and positive")
    add("midprice", mid, "book")
    add("spread", spread, "book")
    add("relative_spread", safe_divide(spread, mid), "book")
    depth, imbalance = {}, {}
    for level in (1, 3, 5, 10):
        for side in ("bid", "ask"):
            # Null padding propagates; partial depth is never treated as zero liquidity.
            values = np.column_stack(
                [numbers(table, f"{side}_sz_{i}") for i in range(1, level + 1)]
            )
            values[values <= 0] = np.nan
            depth[side, level] = values.sum(axis=1)
            add(f"{side}_depth_{level}", depth[side, level], "book" if level == 1 else "depth")
        b, a = depth["bid", level], depth["ask", level]
        imbalance[level] = safe_divide(b - a, b + a)
        add(f"obi_{level}", imbalance[level], "imbalance")
    micro_gap = spread * imbalance[1] / 2
    micro = mid + micro_gap
    add("microprice", micro, "microprice")
    add("microprice_minus_mid", micro_gap, "microprice")
    total_depth = depth["bid", 10] + depth["ask", 10]
    bid_delta = depth["bid", 10] - lag(depth["bid", 10], 1)
    ask_delta = depth["ask", 10] - lag(depth["ask", 10], 1)
    times = table["timestamp_ns"].to_numpy()
    baseline = table["trade_baseline_receive_time_ns"].to_pylist()
    # Avoid float conversion of nanosecond availability timestamps.
    baseline_known = np.array([value is not None for value in baseline])
    baseline_ns = np.array(
        [value if value is not None else 0 for value in baseline], dtype=np.int64
    )
    uncertain = table["trade_coverage_uncertain"].to_numpy(zero_copy_only=False)
    trade_imbalances = {}
    for window in config.flow_windows_ms:
        steps = window // interval_ms
        flows = {}
        for side, delta in (("bid", bid_delta), ("ask", ask_delta)):
            for kind, value in (
                ("added", np.maximum(delta, 0)),
                ("removed", np.maximum(-delta, 0)),
            ):
                flows[side, kind] = rolling_sum(value, steps)
                add(f"{side}_{kind}_{window}ms", flows[side, kind], "flow")
        add(
            f"depth_flow_imbalance_{window}ms",
            flows["bid", "added"]
            - flows["bid", "removed"]
            - flows["ask", "added"]
            + flows["ask", "removed"],
            "flow",
        )
        complete = (np.arange(len(times)) >= steps) & baseline_known
        complete &= baseline_ns <= times - window * 1_000_000
        complete &= rolling_sum(uncertain.astype(float), steps) == 0
        totals = {}
        for name in (
            "trade_count",
            "trade_volume",
            "trade_buy_volume",
            "trade_sell_volume",
            "trade_notional",
        ):
            totals[name] = rolling_sum(numbers(table, name), steps)
            totals[name][~complete] = np.nan
        count, volume = totals["trade_count"], totals["trade_volume"]
        buy, sell = totals["trade_buy_volume"], totals["trade_sell_volume"]
        trade_imbalance = safe_divide(buy - sell, volume, zero_value=0.0)
        trade_imbalances[window] = trade_imbalance
        vwap = safe_divide(totals["trade_notional"], volume)
        for name, value in (
            ("trade_count", count),
            ("trade_volume", volume),
            ("trade_buy_volume", buy),
            ("trade_sell_volume", sell),
            ("trade_imbalance", trade_imbalance),
            ("trade_mean_size", safe_divide(volume, count)),
            ("trade_vwap", vwap),
            ("trade_vwap_minus_mid", vwap - mid),
        ):
            add(f"{name}_{window}ms", value, "trades")
    for window in config.dynamics_lags_ms:
        steps = window // interval_ms
        old_mid = lag(mid, steps)
        add(f"mid_return_{window}ms", safe_divide(mid - old_mid, old_mid), "dynamics")
        for name, values in (
            ("microprice", micro),
            ("spread", spread),
            ("obi_1", imbalance[1]),
            ("total_depth_10", total_depth),
        ):
            add(f"{name}_change_{window}ms", values - lag(values, steps), "dynamics")
        base_window = min(config.flow_windows_ms)
        add(
            f"trade_imbalance_{base_window}ms_lag_{window}ms",
            lag(trade_imbalances[base_window], steps),
            "dynamics",
        )
    return result, groups


def feature_table(table, config, interval_ms):
    arrays, groups = feature_arrays(table, config, interval_ms)
    columns = {name: table[name] for name in IDENTIFIERS}
    columns.update(
        {
            name: pa.array(value, mask=np.isnan(value), type=pa.float64())
            for name, value in arrays.items()
        }
    )
    return pa.table(columns), groups
