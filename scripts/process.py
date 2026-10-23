"""Offline raw journal -> receive-time book snapshots -> bounded Parquet dataset."""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from market_microstructure.data.coinbase import CoinbaseAdapter  # noqa: E402
from market_microstructure.processing.build_dataset import (  # noqa: E402
    build_dataset,
    discover_inputs,
)
from market_microstructure.processing.config import load_processing_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTC-USD")
    parser.add_argument("--date", help="UTC date for OUTPUT rows; inputs retain warm-up history")
    parser.add_argument(
        "--config",
        type=Path,
        default=(Path(__file__).resolve().parents[1] / "configs/processing.yaml"),
    )
    parser.add_argument("--input", type=Path, nargs="+", help="closed files in run/file order")
    parser.add_argument("--raw-dir", type=Path, help="override raw discovery root")
    parser.add_argument(
        "--output", type=Path, help="NEW dataset directory; existing paths rejected"
    )
    parser.add_argument("--interval-ms", type=int)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--max-staleness-ms", type=int)
    parser.add_argument("--batch-rows", type=int)
    args = parser.parse_args()
    try:
        config, raw_dir, output_root = load_processing_config(
            args.config,
            interval_ms=args.interval_ms,
            depth=args.depth,
            max_staleness_ms=args.max_staleness_ms,
            batch_rows=args.batch_rows,
        )
        adapter = CoinbaseAdapter(args.symbol)
        paths = args.input or discover_inputs(
            args.raw_dir or raw_dir, adapter.source, adapter.symbol
        )
        destination = args.output or output_root / (
            f"{args.symbol}_{args.date or 'all'}_"
            f"{datetime.now(UTC):%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
        )
        manifest = build_dataset(paths, destination, adapter, config, output_date=args.date)
        print(
            json.dumps(
                {
                    "output": str(destination.resolve()),
                    "rows": manifest["rows"],
                    "row_groups": manifest["row_groups"],
                    "quality_counts": manifest["quality_counts"],
                },
                indent=2,
            )
        )
        return 0 if manifest["rows"] else 2
    except Exception as exc:
        print(f"Processing failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
