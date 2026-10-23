"""Run public market-data collection, separately from offline research."""

import argparse
import asyncio
import json
import logging
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from market_microstructure.data.coinbase import CoinbaseAdapter  # noqa: E402
from market_microstructure.data.recorder import Recorder  # noqa: E402
from market_microstructure.utils.config import load_collector_config  # noqa: E402


async def collect(args):
    config = load_collector_config(
        args.config,
        symbol=args.symbol,
        output_dir=args.output,
        compress=args.compress,
        max_reconnects=args.max_reconnects,
    )
    recorder = Recorder(CoinbaseAdapter(config.symbol), config)
    loop = asyncio.get_running_loop()
    previous = {}

    def stop_handler(signum, frame):
        loop.call_soon_threadsafe(recorder.stop.set)

    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.signal(sig, stop_handler)
    try:
        summary = await recorder.run(duration=args.duration, max_messages=args.max_messages)
        print(json.dumps(summary, indent=2))
        return 0 if summary.get("snapshot", 0) > 0 else 1
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "collector.yaml",
    )
    parser.add_argument("--symbol", help="initially BTC-USD; other public product IDs accepted")
    parser.add_argument(
        "--output", type=Path, help="override raw path, relative to current directory"
    )
    parser.add_argument("--duration", type=float, help="bounded capture duration in seconds")
    parser.add_argument("--max-messages", type=int)
    parser.add_argument("--max-reconnects", type=int, help="zero disables reconnect attempts")
    parser.add_argument("--compress", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        return asyncio.run(collect(args))
    except KeyboardInterrupt:
        return 130
    except Exception:
        logging.exception("Collector stopped with an error; inspect raw .part files if present")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
