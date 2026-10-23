"""Strict collector YAML configuration; paths are resolved against the config file."""

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

from market_microstructure.data.coinbase import CoinbaseAdapter


@dataclass(frozen=True)
class CollectorConfig:
    symbol: str = "BTC-USD"
    output_dir: Path = Path("data/raw")
    rotate_seconds: int = 3600
    max_file_bytes: int = 134217728
    fsync_seconds: float = 1.0
    compress: bool = False
    max_message_bytes: int = 16777216
    max_queue: int = 32
    open_timeout: float = 10.0
    idle_timeout: float = 15.0
    heartbeat_timeout: float = 10.0
    bootstrap_timeout: float = 15.0
    backoff_initial: float = 1.0
    backoff_max: float = 30.0
    report_seconds: float = 10.0
    max_reconnects: int | None = None
    fetch_product_metadata: bool = True

    def __post_init__(self):
        CoinbaseAdapter(self.symbol)
        for name in ("rotate_seconds", "max_file_bytes", "max_message_bytes", "max_queue"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "fsync_seconds",
            "open_timeout",
            "idle_timeout",
            "heartbeat_timeout",
            "bootstrap_timeout",
            "backoff_initial",
            "backoff_max",
            "report_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(value) or value < 0 or (name != "fsync_seconds" and value == 0):
                raise ValueError(f"{name} must be finite and positive (fsync may be zero)")
        if self.backoff_max < self.backoff_initial:
            raise ValueError("backoff_max must be >= backoff_initial")
        if self.max_reconnects is not None and (
            isinstance(self.max_reconnects, bool)
            or not isinstance(self.max_reconnects, int)
            or self.max_reconnects < 0
        ):
            raise ValueError("max_reconnects must be null or a nonnegative integer")
        for name in ("compress", "fetch_product_metadata"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        object.__setattr__(self, "output_dir", Path(self.output_dir).resolve())

    def to_dict(self) -> dict:
        return {**asdict(self), "output_dir": str(self.output_dir)}


def load_collector_config(path: Path, **overrides) -> CollectorConfig:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        content = yaml.safe_load(handle)
    if not isinstance(content, dict):
        raise ValueError("collector YAML must be a mapping")
    allowed = {f.name for f in fields(CollectorConfig)}
    unknown = set(content) - allowed
    if unknown:
        raise ValueError(f"unknown collector keys: {sorted(unknown)}")
    content.update({key: value for key, value in overrides.items() if value is not None})
    # An explicit CLI path is relative to CWD; YAML paths are relative to its file.
    if overrides.get("output_dir") is not None:
        content["output_dir"] = Path(overrides["output_dir"]).resolve()
    else:
        content["output_dir"] = path.parent / content.get("output_dir", "../data/raw")
    return CollectorConfig(**content)
