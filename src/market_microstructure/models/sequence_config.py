"""GRU and matched-MLP configuration."""

import math
from dataclasses import dataclass

from .learned_config import LearnedConfig


@dataclass(frozen=True)
class SequenceConfig(LearnedConfig):
    model: str = "gru"
    sequence_length: int = 20
    gru_hidden_size: int = 32
    gru_layers: int = 1
    gradient_clip: float = 1.0

    def __post_init__(self):
        if self.model not in ("gru", "sequence_mlp"):
            raise ValueError("unknown sequence comparison model")
        common = {name: getattr(self, name) for name in LearnedConfig.__dataclass_fields__}
        common["model"] = "mlp"
        checked = LearnedConfig(**common)
        object.__setattr__(self, "hidden_sizes", checked.hidden_sizes)
        for name, upper in (("sequence_length", 1024), ("gru_hidden_size", 512), ("gru_layers", 3)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(f"{name} must be an integer in [1, {upper}]")
        if (
            isinstance(self.gradient_clip, bool)
            or not isinstance(self.gradient_clip, (float, int))
            or not math.isfinite(self.gradient_clip)
            or self.gradient_clip <= 0
        ):
            raise ValueError("gradient_clip must be finite and positive")
