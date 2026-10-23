"""Replay closed raw data into causal features and separate future-target Parquet files."""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from market_microstructure.data.coinbase import CoinbaseAdapter  # noqa: E402
from market_microstructure.processing.build_dataset import discover_inputs  # noqa: E402
from market_microstructure.processing.config import load_processing_config  # noqa: E402
from market_microstructure.processing.feature_config import load_feature_config  # noqa: E402
from market_microstructure.processing.research_dataset import build_research_dataset  # noqa: E402


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTC-USD")
    parser.add_argument("--date", help="UTC output date; warm-up starts at first emitted row")
    parser.add_argument("--input", type=Path, nargs="+")
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--processing-config", type=Path, default=root / "configs/processing.yaml")
    parser.add_argument("--config", type=Path, default=root / "configs/features.yaml")
    parser.add_argument("--tick-size", help="verified quoted tick; defaults to captured metadata")
    parser.add_argument("--epsilon-ticks")
    parser.add_argument("--relative-epsilon", help="fixed relative threshold overrides tick mode")
    parser.add_argument("--batch-rows", type=int)
    parser.add_argument("--train-start-ns", type=int, default=0)
    parser.add_argument(
        "--train-end-ns", type=int, help="exclusive cutoff for TRAIN-only diagnostics"
    )
    args = parser.parse_args()
    try:
        processing, raw_dir, output_root = load_processing_config(
            args.processing_config, batch_rows=args.batch_rows
        )
        config = load_feature_config(
            args.config,
            batch_rows=args.batch_rows,
            epsilon_ticks=args.epsilon_ticks,
            relative_epsilon=args.relative_epsilon,
        )
        adapter = CoinbaseAdapter(args.symbol)
        paths = args.input or discover_inputs(
            args.raw_dir or raw_dir, adapter.source, adapter.symbol
        )
        destination = args.output or output_root / (
            f"research_{args.symbol}_{datetime.now(UTC):%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
        )
        manifest = build_research_dataset(
            paths,
            destination,
            adapter,
            processing,
            config,
            tick_size=args.tick_size,
            output_date=args.date,
            train_start_ns=args.train_start_ns,
            train_end_ns=args.train_end_ns,
        )
        print(json.dumps({"output": str(destination.resolve()), **manifest}, indent=2))
        return 0
    except Exception as exc:
        print(f"Feature processing failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
