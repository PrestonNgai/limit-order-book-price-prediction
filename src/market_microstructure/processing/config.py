"""Validated settings for receive-time book sampling and bounded Parquet output."""

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class ProcessingConfig:
    interval_ms: int = 100
    depth: int = 10
    max_staleness_ms: int = 1000
    batch_rows: int = 4096
    decimal_precision: int = 38
    decimal_scale: int = 18
    compression: str = "zstd"
    max_record_bytes: int = 64 * 1024 * 1024

    def __post_init__(self):
        for name in (
            "interval_ms",
            "depth",
            "max_staleness_ms",
            "batch_rows",
            "decimal_precision",
            "max_record_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.decimal_precision > 38:
            raise ValueError("decimal128 precision must be <= 38")
        if (
            isinstance(self.decimal_scale, bool)
            or not isinstance(self.decimal_scale, int)
            or not 0 <= self.decimal_scale < self.decimal_precision
        ):
            raise ValueError("decimal_scale must be an integer in [0, precision)")
        if self.compression not in ("zstd", "snappy", "none"):
            raise ValueError("compression must be zstd, snappy or none")

    def to_dict(self):
        return asdict(self)


def load_processing_config(path: Path, **overrides) -> tuple[ProcessingConfig, Path, Path]:
    path = Path(path).resolve()
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("processing YAML must be a mapping")
    allowed = {item.name for item in fields(ProcessingConfig)} | {"raw_dir", "output_dir"}
    if unknown := set(value) - allowed:
        raise ValueError(f"unknown processing keys: {sorted(unknown)}")
    raw_dir = path.parent / value.pop("raw_dir", "../data/raw")
    output_dir = path.parent / value.pop("output_dir", "../data/processed")
    value.update({key: val for key, val in overrides.items() if val is not None})
    return ProcessingConfig(**value), raw_dir.resolve(), output_dir.resolve()
