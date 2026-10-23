"""Reload a saved model and evaluate a declared chronological split."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from market_microstructure.models.experiments import evaluate_run  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="NEW evaluation directory")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Explicit final-test access; freeze research choices first",
    )
    args = parser.parse_args()
    try:
        result = evaluate_run(
            args.run, args.dataset, args.output, split=args.split, allow_test=args.allow_test
        )
        print(
            json.dumps(
                {
                    "output": str(args.output.resolve()),
                    "split": result["split"],
                    "rows": result["rows"],
                    "test_accessed": result["test_accessed"],
                },
                indent=2,
            )
        )
        return 0
    except Exception as exc:
        print(f"Evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
