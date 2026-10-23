"""Fit on TRAIN, select learned checkpoints on validation and save a frozen run."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from market_microstructure.models.experiments import train_run  # noqa: E402
from market_microstructure.models.learned_config import load_model_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="NEW versioned run directory")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--model",
        choices=["majority", "imbalance", "logistic", "xgboost", "mlp", "gru", "sequence_mlp"],
    )
    parser.add_argument("--horizon-ms", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    try:
        config = load_model_config(
            args.config
            or (
                Path(__file__).resolve().parents[1]
                / "configs"
                / (
                    f"{args.model}.yaml"
                    if args.model in ("xgboost", "mlp", "gru", "sequence_mlp")
                    else "baselines.yaml"
                )
            ),
            model=args.model,
            horizon_ms=args.horizon_ms,
            seed=args.seed,
        )
        result = train_run(args.dataset, args.output, config)
        print(
            json.dumps(
                {
                    "output": str(args.output.resolve()),
                    "model": result["model"],
                    "train_rows": result["fit"]["rows"],
                    "validation_rows": result["validation_rows"],
                    "test_accessed": result["test_accessed"],
                },
                indent=2,
            )
        )
        return 0
    except Exception as exc:
        print(f"Training failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
