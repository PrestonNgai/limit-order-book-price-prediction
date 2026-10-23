"""Create purged chronological datasets and frozen TRAIN-only preprocessing."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from market_microstructure.processing.split_config import load_split_config  # noqa: E402
from market_microstructure.processing.split_dataset import build_split_dataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="Directory containing features.parquet and labels"
    )
    parser.add_argument("--output", type=Path, required=True, help="NEW output directory")
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/splits.yaml"
    )
    parser.add_argument("--mode", choices=["auto", "sessions", "time", "explicit"])
    parser.add_argument("--train-end-ns", type=int)
    parser.add_argument("--validation-end-ns", type=int)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--batch-rows", type=int)
    parser.add_argument("--horizons-ms", type=int, nargs="+")
    args = parser.parse_args()
    try:
        config = load_split_config(
            args.config,
            mode=args.mode,
            train_end_ns=args.train_end_ns,
            validation_end_ns=args.validation_end_ns,
            sequence_length=args.sequence_length,
            batch_rows=args.batch_rows,
        )
        result = build_split_dataset(args.input, args.output, config, horizons_ms=args.horizons_ms)
        print(
            json.dumps(
                {
                    "output": str(args.output.resolve()),
                    "method": result["method"],
                    "ranges": result["ranges"],
                    "target_counts": result["target_counts"],
                    "three_way_evaluation_available": result["three_way_evaluation_available"],
                    "three_way_sequence_evaluation_available": result[
                        "three_way_sequence_evaluation_available"
                    ],
                    "test_status": "sealed",
                },
                indent=2,
            )
        )
        return 0
    except Exception as exc:
        print(f"Splitting failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
