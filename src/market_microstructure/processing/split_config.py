"""Frozen chronological split and sequence settings."""

from dataclasses import asdict, dataclass, fields
from decimal import Decimal, InvalidOperation
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class SplitConfig:
    mode: str = "auto"
    train_fraction: str = "0.70"
    validation_fraction: str = "0.15"
    train_end_ns: int | None = None
    validation_end_ns: int | None = None
    sequence_length: int = 20
    batch_rows: int = 4096
    regime_quantiles: tuple[float, ...] = (0.5,)

    def __post_init__(self):
        if self.mode not in ("auto", "sessions", "time", "explicit"):
            raise ValueError("mode must be auto, sessions, time or explicit")
        try:
            a, b = Decimal(str(self.train_fraction)), Decimal(str(self.validation_fraction))
        except InvalidOperation as exc:
            raise ValueError("split fractions must be decimal numbers") from exc
        if not a.is_finite() or not b.is_finite() or a <= 0 or b <= 0 or a + b >= 1:
            raise ValueError("positive train/validation fractions must sum to less than one")
        for name in ("sequence_length", "batch_rows"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        explicit = (self.train_end_ns, self.validation_end_ns)
        if self.mode == "explicit":
            if (
                any(type(v) is not int for v in explicit)
                or not 0 < explicit[0] < explicit[1] < 2**63
            ):
                raise ValueError("explicit mode requires ordered positive boundary timestamps")
        elif any(v is not None for v in explicit):
            raise ValueError("boundary timestamps require explicit mode")
        q = tuple(self.regime_quantiles)
        if (
            not q
            or list(q) != sorted(set(q))
            or any(type(v) not in (int, float) or not 0 < v < 1 for v in q)
        ):
            raise ValueError("regime quantiles must be sorted, unique and inside (0,1)")
        object.__setattr__(self, "regime_quantiles", q)

    def to_dict(self):
        return asdict(self)


def load_split_config(path: Path, **overrides):
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("split YAML must be a mapping")
    if unknown := set(value) - {f.name for f in fields(SplitConfig)}:
        raise ValueError(f"unknown split keys: {sorted(unknown)}")
    value.update({k: v for k, v in overrides.items() if v is not None})
    return SplitConfig(**value)
