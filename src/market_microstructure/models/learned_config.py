"""Frozen settings for boosted trees and the small CPU MLP."""

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class LearnedConfig:
    model: str = "xgboost"
    horizon_ms: int = 500
    seed: int = 42
    batch_rows: int = 4096
    max_train_rows: int = 250000
    max_eval_rows: int = 250000
    max_fit_mb: int = 512
    probability_mix: float = 1e-6
    patience: int = 15
    min_delta: float = 0.0
    rounds: int = 200
    max_depth: int = 3
    tree_learning_rate: float = 0.05
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0
    explain_rows: int = 0
    hidden_sizes: tuple = (64, 32)
    dropout: float = 0.1
    epochs: int = 100
    train_batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    timing_warmup: int = 5
    timing_repeats: int = 30
    timing_batch_size: int = 64

    def __post_init__(self):
        if self.model not in ("xgboost", "mlp"):
            raise ValueError("unknown learned model")
        for name in (
            "horizon_ms",
            "batch_rows",
            "max_train_rows",
            "max_eval_rows",
            "max_fit_mb",
            "patience",
            "rounds",
            "max_depth",
            "epochs",
            "train_batch_size",
            "timing_warmup",
            "timing_repeats",
            "timing_batch_size",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an unsigned 32-bit integer")
        if type(self.explain_rows) is not int or not 0 <= self.explain_rows <= 1000:
            raise ValueError("explain_rows must be an integer in [0, 1000]")
        for name in (
            "probability_mix",
            "min_delta",
            "tree_learning_rate",
            "min_child_weight",
            "reg_lambda",
            "dropout",
            "learning_rate",
            "weight_decay",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("tree_learning_rate", "learning_rate", "probability_mix"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.probability_mix >= 1 or self.dropout >= 1:
            raise ValueError("probability_mix and dropout must be below one")
        if not isinstance(self.hidden_sizes, (tuple, list)) or not 1 <= len(self.hidden_sizes) <= 3:
            raise ValueError("hidden_sizes needs one to three small layers")
        if any(type(n) is not int or not 1 <= n <= 512 for n in self.hidden_sizes):
            raise ValueError("hidden layer sizes must be integers in [1, 512]")
        object.__setattr__(self, "hidden_sizes", tuple(self.hidden_sizes))

    def to_dict(self):
        result = asdict(self)
        result["hidden_sizes"] = list(self.hidden_sizes)
        return result


def config_from_mapping(data):
    if data.get("model") == "tuned":
        from .tuned import TunedConfig

        return TunedConfig(**data)
    if data.get("model") in ("gru", "sequence_mlp"):
        from .sequence_config import SequenceConfig

        return SequenceConfig(**data)
    if data.get("model") in ("xgboost", "mlp"):
        return LearnedConfig(**data)
    from .config import BaselineConfig

    return BaselineConfig(**data)


def load_model_config(path, **overrides):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("model config must be a mapping")
    data.update({k: v for k, v in overrides.items() if v is not None})
    return config_from_mapping(data)
