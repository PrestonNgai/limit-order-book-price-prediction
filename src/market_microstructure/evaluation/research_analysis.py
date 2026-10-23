"""Validation diagnostics; no threshold fitting or TEST access."""

import numpy as np

from .metrics import classification_metrics


def arrays(table):
    return (
        table["target"].to_numpy(),
        table["prediction"].to_numpy(),
        np.column_stack([table[n].to_numpy() for n in ("p_down", "p_stable", "p_up")]),
    )


def confidence_analysis(table, thresholds):
    y, pred, p = arrays(table)
    classification_metrics(y, pred, p)
    returns = table["future_return"].to_numpy()
    if not np.isfinite(returns).all():
        raise ValueError("nonfinite forward returns")
    confidence = p[np.arange(len(y)), pred.astype(int) + 1]
    results = []
    for threshold in thresholds:
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("confidence threshold outside [0,1]")
        mask = confidence >= threshold
        directional = mask & (pred != 0)
        moves = pred[directional] * returns[directional] * 10000
        results.append(
            {
                "threshold": threshold,
                "confident_rows": int(mask.sum()),
                "metrics": classification_metrics(y[mask], pred[mask], p[mask]),
                "directional_signals": int(directional.sum()),
                "signal_frequency_per_eligible_observation": float(directional.mean())
                if len(y)
                else None,
                "mean_signed_mid_move_bps": float(moves.mean()) if len(moves) else None,
                "median_signed_mid_move_bps": float(np.median(moves)) if len(moves) else None,
                "positive_move_win_rate": float((moves > 0).mean()) if len(moves) else None,
                "zero_move_fraction": float((moves == 0).mean()) if len(moves) else None,
            }
        )
    return results


def cluster_interval(values, sessions, *, minimum_sessions=10, repeats=2000, seed=42):
    """Row-weighted mean; resample entire sessions, preserving all within-session dependence."""
    values = np.asarray(values, dtype=float)
    sessions = np.asarray(sessions)
    if values.ndim != 1 or sessions.shape != values.shape or not np.isfinite(values).all():
        raise ValueError("invalid cluster observations")
    if minimum_sessions < 2 or repeats < 100:
        raise ValueError("insufficient uncertainty configuration")
    keys = np.unique(sessions)
    base = {
        "sessions": len(keys),
        "rows": len(values),
        "minimum_sessions": minimum_sessions,
        "method": "whole-session cluster percentile bootstrap, row-weighted mean",
        "assumption": "collection sessions must be sufficiently independent; "
        "count alone cannot prove this",
    }
    if len(keys) < minimum_sessions:
        return {
            **base,
            "status": "unavailable",
            "reason": "insufficient independent collection sessions",
        }
    sums = np.array([values[sessions == k].sum() for k in keys])
    counts = np.array([(sessions == k).sum() for k in keys])
    rng = np.random.default_rng(seed)
    samples = np.empty(repeats)
    for i in range(repeats):
        selected = rng.integers(0, len(keys), len(keys))
        samples[i] = sums[selected].sum() / counts[selected].sum()
    return {
        **base,
        "status": "complete",
        "estimate": float(values.mean()),
        "lower_95": float(np.quantile(samples, 0.025)),
        "upper_95": float(np.quantile(samples, 0.975)),
        "repeats": repeats,
        "seed": seed,
    }


def align_predictions(predictions, joined, horizon, *, split="validation"):
    """Identity check precedes any regime, return or session aggregation."""
    ids = joined["row_id"].to_numpy()
    wanted = predictions["row_id"].to_numpy()
    positions = np.searchsorted(ids, wanted)
    if (
        len(np.unique(wanted)) != len(wanted)
        or np.any(np.diff(wanted) <= 0)
        or np.any(positions >= len(ids))
    ):
        raise ValueError("prediction rows are not an ordered unique subset")
    result = joined.take(positions)
    pairs = [
        ("row_id", "row_id"),
        ("timestamp_ns", "timestamp_ns"),
        ("label_end_ns", f"label_end_ns_{horizon}ms"),
        ("target", f"y_class_{horizon}ms"),
        ("future_return", f"y_return_{horizon}ms"),
    ]
    for left, right in pairs:
        if not np.array_equal(predictions[left].to_numpy(), result[right].to_numpy()):
            raise ValueError("prediction identity/target/return mismatch")
    if split not in ("validation", "test") or set(predictions["split"].to_pylist()) - {split}:
        raise ValueError(f"research requires {split} predictions")
    return result
