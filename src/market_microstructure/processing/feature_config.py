"""Explicit windows and a price-tick threshold; no fitting on all observations."""

from dataclasses import asdict, dataclass, fields
from decimal import Decimal, InvalidOperation
from pathlib import Path

import yaml


def positive_decimal(value, name):
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be a positive decimal") from exc
    if not result.is_finite() or result <= 0:
        raise ValueError(f"{name} must be a positive decimal")
    return result


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    flow_windows_ms: tuple[int, ...] = (100, 500, 1000, 5000)
    dynamics_lags_ms: tuple[int, ...] = (100, 500, 1000, 2000, 5000)
    horizons_ms: tuple[int, ...] = (100, 500, 1000, 5000)
    alignment_tolerance_ms: int = 0
    epsilon_ticks: str = "1"
    relative_epsilon: str | None = None
    batch_rows: int = 4096

    def __post_init__(self):
        for name in ("flow_windows_ms", "dynamics_lags_ms", "horizons_ms"):
            values = tuple(getattr(self, name))
            if not values or any(type(v) is not int or v <= 0 for v in values):
                raise ValueError(f"{name} must contain positive integers")
            if list(values) != sorted(set(values)):
                raise ValueError(f"{name} must be sorted and unique")
            object.__setattr__(self, name, values)
        if type(self.batch_rows) is not int or self.batch_rows <= 0:
            raise ValueError("batch_rows must be positive")
        if type(self.alignment_tolerance_ms) is not int or self.alignment_tolerance_ms < 0:
            raise ValueError("alignment_tolerance_ms must be nonnegative")
        positive_decimal(self.epsilon_ticks, "epsilon_ticks")
        if self.relative_epsilon is not None:
            positive_decimal(self.relative_epsilon, "relative_epsilon")

    def validate_interval(self, interval_ms):
        if type(interval_ms) is not int or interval_ms <= 0:
            raise ValueError("interval_ms must be positive")
        if any(v % interval_ms for v in self.flow_windows_ms + self.dynamics_lags_ms):
            raise ValueError("feature windows and lags must be multiples of the sampling interval")
        if self.alignment_tolerance_ms >= interval_ms:
            raise ValueError("alignment tolerance must be smaller than one sampling interval")

    def history_rows(self, interval_ms):
        return (max(self.dynamics_lags_ms) + max(self.flow_windows_ms)) // interval_ms

    def future_rows(self, interval_ms):
        return (
            max(self.horizons_ms) + self.alignment_tolerance_ms + interval_ms - 1
        ) // interval_ms

    def to_dict(self):
        return asdict(self)


def load_feature_config(path: Path, **overrides):
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("feature YAML must be a mapping")
    if unknown := set(value) - {f.name for f in fields(FeatureConfig)}:
        raise ValueError(f"unknown feature keys: {sorted(unknown)}")
    value.update({k: v for k, v in overrides.items() if v is not None})
    return FeatureConfig(**value)
