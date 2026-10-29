"""Select a classifier with expanding-session development folds; never opens TEST."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from market_microstructure.evaluation.tuning import run_tuning  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/tuning.yaml"
    )
    args = parser.parse_args()
    try:
        print(json.dumps(run_tuning(args.dataset, args.output, args.config), indent=2))
        return 0
    except Exception as exc:
        print(f"Tuning failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
