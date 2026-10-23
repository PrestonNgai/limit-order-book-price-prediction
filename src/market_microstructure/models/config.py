"""Frozen baseline settings; validation is never used to choose these values."""

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class BaselineConfig:
    model: str = "logistic"
    horizon_ms: int = 500
    seed: int = 42
    smoothing: float = 1.0
    imbalance_feature: str = "f_obi_1"
    imbalance_deadband: float = 0.0
    logistic_c: float = 1.0
    max_iter: int = 1000
    tolerance: float = 1e-6
    probability_mix: float = 1e-6
    batch_rows: int = 4096
    max_train_rows: int = 250000
    max_eval_rows: int = 250000
    max_fit_mb: int = 512

    def __post_init__(self):
        if self.model not in ("majority", "imbalance", "logistic"):
            raise ValueError("unknown baseline model")
        for name in (
            "horizon_ms",
            "max_iter",
            "batch_rows",
            "max_train_rows",
            "max_eval_rows",
            "max_fit_mb",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an unsigned 32-bit integer")
        for name in ("smoothing", "logistic_c", "tolerance", "probability_mix"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.probability_mix >= 1:
            raise ValueError("probability_mix must be below one")
        if (
            isinstance(self.imbalance_deadband, bool)
            or not isinstance(self.imbalance_deadband, (int, float))
            or not math.isfinite(self.imbalance_deadband)
            or not 0 <= self.imbalance_deadband <= 1
        ):
            raise ValueError("imbalance_deadband must be in [0, 1]")
        if self.imbalance_feature not in ("f_obi_1", "f_obi_3", "f_obi_5", "f_obi_10"):
            raise ValueError("imbalance_feature must be an unscaled OBI feature")

    def to_dict(self):
        return asdict(self)


def load_baseline_config(path, **overrides):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("baseline config must be a mapping")
    unknown = set(data) - set(BaselineConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown baseline settings: {sorted(unknown)}")
    data.update({k: v for k, v in overrides.items() if v is not None})
    return BaselineConfig(**data)
