"""Train or evaluate future-return regression experiments."""

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from market_microstructure.evaluation.regression import run_regression_comparison  # noqa: E402
from market_microstructure.models.regression import (  # noqa: E402
    RegressionConfig,
    evaluate_regression_run,
    train_regression_run,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "evaluate", "compare"])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/regression.yaml",
    )
    parser.add_argument("--model", choices=["zero", "ridge", "xgboost"])
    parser.add_argument("--horizon-ms", type=int)
    parser.add_argument("--horizons-ms", type=int, nargs="+", default=[100, 500, 1000, 5000])
    parser.add_argument("--run", type=Path)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.action in ("train", "compare"):
            if args.split != "validation" or args.allow_test or args.run:
                raise ValueError("training cannot access TEST or reuse a fitted run")
            settings = yaml.safe_load(args.config.read_text(encoding="utf-8"))
            settings.update(
                {
                    k: v
                    for k, v in {"model": args.model, "horizon_ms": args.horizon_ms}.items()
                    if v is not None
                }
            )
            config = RegressionConfig(**settings)
            if args.action == "compare":
                result = run_regression_comparison(
                    args.dataset, args.output, config, horizons=args.horizons_ms
                )
            else:
                result = train_regression_run(args.dataset, args.output, config)
        else:
            if args.run is None:
                raise ValueError("evaluation requires --run")
            result = evaluate_regression_run(
                args.run, args.dataset, args.output, split=args.split, allow_test=args.allow_test
            )
        print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:
        print(f"Regression failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
