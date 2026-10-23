"""Freeze a research plan or evaluate its saved models on final TEST data."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from market_microstructure.evaluation.final_test import (  # noqa: E402
    freeze_final_plan,
    run_final_test,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--research", type=Path, required=True)
    freeze.add_argument("--dataset", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--regression-runs", type=Path, nargs="*", default=[])
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--plan", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.action == "freeze":
            result = freeze_final_plan(
                args.research, args.dataset, args.output, regression_runs=args.regression_runs
            )
        else:
            result = run_final_test(args.plan, args.output, allow_test=args.allow_test)
        print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:
        print(f"Final evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
