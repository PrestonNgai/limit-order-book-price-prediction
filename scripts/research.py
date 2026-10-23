"""Run validation experiments from a YAML configuration."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from market_microstructure.evaluation.research import run_research  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run_research(args.dataset, args.output, args.config)
    except Exception as exc:
        print(f"Research failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {k: result[k] for k in ("completed_runs", "unavailable_runs", "test_accessed")},
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
