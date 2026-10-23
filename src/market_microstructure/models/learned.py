"""Frozen learned-model inference using the same TRAIN-only preprocessing as baselines."""

import numpy as np

from market_microstructure.processing.preprocessing import TrainPreprocessor


def float32_matrix(table, preprocessor):
    matrix = TrainPreprocessor.transform(table, preprocessor)
    with np.errstate(over="ignore"):
        matrix = matrix.astype(np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("features overflow float32")
    return matrix


class FrozenLearned:
    def __init__(self, state, preprocessor, estimator):
        self.state, self.preprocessor, self.estimator = state, preprocessor, estimator

    def probabilities(self, matrix):
        if self.state["model"] == "xgboost":
            from .trees import predict_tree

            probabilities = predict_tree(self.estimator, matrix, self.state["feature_columns"])
        else:
            from .mlp import predict_mlp

            probabilities = predict_mlp(self.estimator, matrix)
        p = np.asarray(probabilities, dtype=np.float64)
        if p.shape != (len(matrix), 3) or not np.isfinite(p).all() or np.any(p < 0):
            raise ValueError("invalid learned-model probabilities")
        p /= p.sum(axis=1, keepdims=True)
        mix = self.state["config"]["probability_mix"]
        p = (1 - mix) * p + mix / 3
        if not np.isfinite(p).all() or np.any(p <= 0):
            raise ValueError("invalid smoothed probabilities")
        return p

    def predict(self, table):
        indices = [
            self.preprocessor["feature_columns"].index(n) for n in self.state["feature_columns"]
        ]
        matrix = float32_matrix(table, self.preprocessor)
        p = self.probabilities(np.ascontiguousarray(matrix[:, indices]))
        return (p.argmax(axis=1) - 1).astype(np.int8), p
