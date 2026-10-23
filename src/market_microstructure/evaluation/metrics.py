"""Three-class diagnostics with explicit undefined metrics and probability semantics."""

import numpy as np
from sklearn.metrics import roc_auc_score

NAMES = ("DOWN", "STABLE", "UP")


def classification_metrics(target, prediction, probabilities):
    y, pred, p = np.asarray(target), np.asarray(prediction), np.asarray(probabilities, dtype=float)
    n = len(y)
    if y.shape != (n,) or pred.shape != (n,) or p.shape != (n, 3):
        raise ValueError("metric input shapes disagree")
    if (
        not np.isin(y, [-1, 0, 1]).all()
        or not np.isin(pred, [-1, 0, 1]).all()
        or not np.isfinite(p).all()
        or np.any(p <= 0)
        or not np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-12)
    ):
        raise ValueError("invalid classes or probability matrix")
    if not n:
        return {"status": "unavailable", "rows": 0, "reason": "no eligible targets"}
    matrix = np.zeros((3, 3), dtype=np.int64)
    np.add.at(matrix, (y.astype(int) + 1, pred.astype(int) + 1), 1)
    rows, cols = matrix.sum(axis=1), matrix.sum(axis=0)
    per_class, aucs = {}, []
    for k, name in enumerate(NAMES):
        tp = int(matrix[k, k])
        support, predicted = int(rows[k]), int(cols[k])
        auc = None
        if 0 < support < n:
            auc = float(roc_auc_score(y == k - 1, p[:, k]))
        aucs.append(auc)
        per_class[name] = {
            "support": support,
            "predicted": predicted,
            "precision": tp / predicted if predicted else None,
            "recall": tp / support if support else None,
            "f1": 2 * tp / (support + predicted) if support + predicted else None,
            "roc_auc_ovr": auc,
        }
    macro = {
        key: float(np.mean([v[key] or 0.0 for v in per_class.values()]))
        for key in ("precision", "recall", "f1")
    }
    directed = pred != 0
    confidence = p[np.arange(n), pred.astype(int) + 1]
    correct = pred == y
    calibration = []
    bins = np.minimum((confidence * 10).astype(int), 9)
    ece = 0.0
    for k in range(10):
        mask = bins == k
        count = int(mask.sum())
        average = float(confidence[mask].mean()) if count else None
        accuracy = float(correct[mask].mean()) if count else None
        if count:
            ece += count / n * abs(average - accuracy)
        calibration.append(
            {
                "lower": k / 10,
                "upper": (k + 1) / 10,
                "rows": count,
                "mean_confidence": average,
                "accuracy": accuracy,
            }
        )
    onehot = np.eye(3)[y.astype(int) + 1]
    return {
        "status": "complete",
        "rows": n,
        "class_order": [-1, 0, 1],
        "confusion_matrix_actual_rows_predicted_columns": matrix.tolist(),
        "accuracy": float(correct.mean()),
        "balanced_accuracy_present_classes": float(
            np.mean([v["recall"] for v in per_class.values() if v["support"]])
        ),
        "per_class": per_class,
        "macro_zero_filled_three_classes": macro,
        "log_loss": float(-np.log(p[np.arange(n), y.astype(int) + 1]).mean()),
        "brier_multiclass_sum": float(np.square(p - onehot).sum(axis=1).mean()),
        "roc_auc_ovr_macro": float(np.mean(aucs)) if all(v is not None for v in aucs) else None,
        "directional_coverage": float(directed.mean()),
        "directional_accuracy": float(correct[directed].mean()) if directed.any() else None,
        "probability_argmax_agreement": float((pred == p.argmax(axis=1) - 1).mean()),
        "confidence_ece_10_bins": ece,
        "calibration": calibration,
        "undefined_policy": (
            "Zero-denominator per-class metrics and undefined one-vs-rest AUC are null; "
            "Macro precision/recall/F1 average all three classes, "
            "with undefined values set to zero. "
            "Balanced accuracy averages recall only over observed target classes. "
            "Directional accuracy counts STABLE outcomes as misses on directional predictions. "
            "Confidence is the probability assigned to the hard prediction, not necessarily max(p)."
        ),
    }
