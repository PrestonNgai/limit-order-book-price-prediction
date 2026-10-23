"""Transparent three-class baselines and portable JSON inference parameters."""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from market_microstructure.processing.preprocessing import TrainPreprocessor

CLASSES = np.array([-1, 0, 1], dtype=np.int8)


def majority_class(counts):
    # Break ties in favor of STABLE, then DOWN.
    return int([0, -1, 1][np.argmax(np.asarray(counts)[[1, 0, 2]])])


def imbalance_sign(table, config):
    values = table[config["imbalance_feature"]].to_numpy()
    if np.any(np.isinf(values)) or np.any(np.abs(values[np.isfinite(values)]) > 1 + 1e-12):
        raise ValueError("OBI requires raw values in [-1, 1]")
    band = config["imbalance_deadband"]
    # Missing book depth and values at the deadband edges abstain as STABLE.
    return np.where(values > band, 1, np.where(values < -band, -1, 0)).astype(np.int8)


def logistic_probabilities(matrix, state):
    classes = np.asarray(state["observed_classes"], dtype=int)
    result = np.zeros((len(matrix), 3))
    if len(classes) == 1:
        result[:, classes[0] + 1] = 1
    else:
        scores = matrix @ np.asarray(state["coef"]).T + np.asarray(state["intercept"])
        if len(classes) == 2:
            # Stable sigmoid without exponent overflow.
            z = scores[:, 0]
            positive = np.exp(-np.logaddexp(0, -z))
            probabilities = np.column_stack((1 - positive, positive))
        else:
            scores -= scores.max(axis=1, keepdims=True)
            probabilities = np.exp(scores)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
        result[:, classes + 1] = probabilities
    mix = state["config"]["probability_mix"]
    return (1 - mix) * result + mix / 3


def fit_logistic(matrix, target, config):
    classes = np.unique(target)
    state = {
        "observed_classes": classes.tolist(),
        "coef": [],
        "intercept": [],
        "config": config.to_dict(),
    }
    report = {
        "solver": "lbfgs",
        "regularization": "L2",
        "class_weight": None,
        "thread_limit": 1,
        "iterations": 0,
        "reload_max_abs_error": 0.0,
    }
    if len(classes) == 1:
        report["status"] = "single_class_constant"
        report["reason"] = "TRAIN contains only one class; no optimization was attempted"
        return state, report
    estimator = LogisticRegression(
        C=config.logistic_c,
        solver="lbfgs",
        max_iter=config.max_iter,
        tol=config.tolerance,
        random_state=config.seed,
    )
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        estimator.fit(matrix, target)
    state.update(coef=estimator.coef_.tolist(), intercept=estimator.intercept_.tolist())
    report.update(status="converged", iterations=int(estimator.n_iter_.max()))
    # Check that portable inference matches sklearn before evaluating validation.
    expected = np.zeros((len(target), 3))
    expected[:, estimator.classes_ + 1] = estimator.predict_proba(matrix)
    mix = config.probability_mix
    expected = (1 - mix) * expected + mix / 3
    actual = logistic_probabilities(matrix, state)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
    report["reload_max_abs_error"] = float(np.max(np.abs(actual - expected)))
    return state, report


class FrozenBaseline:
    """No pickle and no fitting during prediction. Input must contain raw features."""

    def __init__(self, state, preprocessor):
        if state.get("schema_version") != 1 or state.get("classes") != [-1, 0, 1]:
            raise ValueError("unsupported model schema or class order")
        self.state, self.preprocessor = state, preprocessor

    def predict(self, table):
        state, count = self.state, table.num_rows
        if state["model"] == "majority":
            probabilities = np.tile(state["probabilities"], (count, 1))
            prediction = np.full(count, state["majority_class"], dtype=np.int8)
        elif state["model"] == "imbalance":
            prediction = imbalance_sign(table, state["config"])
            probabilities = np.asarray(state["conditional_probabilities"])[prediction + 1]
        elif state["model"] == "logistic":
            matrix = TrainPreprocessor.transform(table, self.preprocessor)
            probabilities = logistic_probabilities(matrix, state)
            prediction = CLASSES[probabilities.argmax(axis=1)]
        else:
            raise ValueError("unsupported model")
        if (
            probabilities.shape != (count, 3)
            or not np.all(np.isfinite(probabilities))
            or np.any(probabilities <= 0)
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-12, rtol=0)
        ):
            raise ValueError("invalid model probability output")
        return prediction, probabilities
