"""Stream original journals into sampled books; no network or model dependencies."""

import hashlib
import io
import json
import os
import platform
import re
import subprocess
from contextlib import ExitStack
from datetime import UTC, date, datetime
from importlib.metadata import version
from pathlib import Path

import pyarrow.parquet as pq

from market_microstructure import __version__
from market_microstructure.data.base import SourceAdapter

from .config import ProcessingConfig
from .parquet import SnapshotSink
from .replay import JournalReplay, ReplayError
from .sampling import SnapshotSampler


def discover_inputs(raw_dir: Path, source: str, symbol: str) -> list[Path]:
    """Group files by run, then writer index, even after clock changes."""
    raw_dir = Path(raw_dir).resolve()
    pattern = re.compile(
        rf"^{re.escape(source)}_{re.escape(symbol)}_(\d{{8}}_\d{{6}})_"
        r"([A-Za-z0-9_-]+)_(\d+)\.jsonl(?:\.zst)?$"
    )
    groups = {}
    for path in raw_dir.rglob(f"{source}_{symbol}_*.jsonl*"):
        if path.name.endswith(".part"):
            raise ReplayError(f"unfinished capture; select closed inputs explicitly: {path}")
        match = pattern.fullmatch(path.name)
        if match is None:
            raise ReplayError(f"unrecognized capture filename: {path}")
        run_id, index = match[2], int(match[3])
        groups.setdefault(run_id, []).append((index, match[1], path))
    ordered = []
    for _, group in sorted(
        groups.items(), key=lambda pair: (min(pair[1], key=lambda x: x[0])[1], pair[0])
    ):
        indexes = [x[0] for x in group]
        if len(indexes) != len(set(indexes)):
            raise ReplayError("duplicate file index (possibly JSONL and compressed copies)")
        ordered.extend(x[2] for x in sorted(group))
    if not ordered:
        raise ReplayError(f"no closed {source} {symbol} files found in {raw_dir}")
    return ordered


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ReplayError(f"duplicate journal key: {key}")
        value[key] = item
    return value


def _invalid_constant(value):
    raise ReplayError(f"nonstandard JSON constant: {value}")


def iter_journal(path: Path, max_record_bytes: int):
    """Read one bounded JSONL record at a time, including streaming decompression."""
    path = Path(path)
    if path.name.endswith(".part"):
        raise ReplayError(f"incomplete input rejected: {path}")
    with ExitStack() as stack:
        binary = stack.enter_context(path.open("rb"))
        if path.name.endswith(".zst"):
            import zstandard

            stream = stack.enter_context(zstandard.ZstdDecompressor().stream_reader(binary))
            binary = stack.enter_context(io.BufferedReader(stream))
        line_number = 0
        while line := binary.readline(max_record_bytes + 1):
            line_number += 1
            if len(line) > max_record_bytes:
                raise ReplayError(f"record exceeds byte bound: {path}:{line_number}")
            if not line.endswith(b"\n"):
                raise ReplayError(f"truncated JSONL tail: {path}:{line_number}")
            try:
                value = json.loads(
                    line, object_pairs_hook=_unique_object, parse_constant=_invalid_constant
                )
            except (ValueError, UnicodeError) as exc:
                raise ReplayError(f"invalid journal JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ReplayError(f"journal row must be an object: {path}:{line_number}")
            yield value


def _git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
            cwd=Path(__file__).resolve().parents[3],
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def build_dataset(
    paths,
    destination: Path,
    adapter: SourceAdapter,
    config: ProcessingConfig,
    *,
    output_date: str | None = None,
    include_trade_buckets: bool = False,
) -> dict:
    """Write a NEW dataset; failures retain _INCOMPLETE and diagnostics.

    File order is explicit; partitions do not reset replay. Date filters apply to
    emitted rows only, preserving earlier input needed for snapshot bootstrap.
    """
    paths = [Path(p).resolve() for p in paths]
    if not paths or len(paths) != len(set(paths)):
        raise ReplayError("provide a nonempty list of unique input paths")
    if output_date is not None:
        if date.fromisoformat(output_date).isoformat() != output_date:
            raise ValueError("output date must use YYYY-MM-DD")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    marker = destination / "_INCOMPLETE"
    marker.write_text("No committed dataset yet. Do not use these files.\n")
    sink = None
    inputs = []
    try:
        sink_type, sampler_type = SnapshotSink, SnapshotSampler
        if include_trade_buckets:
            from .trade_buckets import TradeSnapshotSampler, TradeSnapshotSink

            sink_type, sampler_type = TradeSnapshotSink, TradeSnapshotSampler
        sink = sink_type(destination / "snapshots.parquet.part", config)
        with ExitStack() as stack:
            sidecars = {
                name: stack.enter_context(
                    (destination / f"{name}.jsonl.part").open("x", encoding="utf-8", newline="\n")
                )
                for name in ("segments", "quality", "source_metadata")
            }

            def write_sidecar(name, value):
                sidecars[name].write(json.dumps(value, allow_nan=False) + "\n")

            sampler = sampler_type(
                config,
                sink.add,
                lambda row: write_sidecar("segments", row),
                lambda row: write_sidecar("quality", row),
                source=adapter.source,
                symbol=adapter.symbol,
                output_date=output_date,
            )
            replay = JournalReplay(adapter, sampler)
            for path in paths:
                before = path.stat()
                digest = file_hash(path)
                input_rows = 0
                for record in iter_journal(path, config.max_record_bytes):
                    replay.consume(record)
                    input_rows += 1
                    if record["record_type"] in (
                        "product_metadata",
                        "metadata_unavailable",
                        "metadata_invalid",
                        "metadata_skipped",
                    ):
                        write_sidecar("source_metadata", record)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ReplayError(f"input changed while processing: {path}")
                inputs.append(
                    {
                        "path": str(path),
                        "sha256": digest,
                        "bytes": before.st_size,
                        "journal_records": input_rows,
                    }
                )
            replay.finish()
            sink.close()
            for handle in sidecars.values():
                handle.flush()
                os.fsync(handle.fileno())
        parquet = pq.ParquetFile(destination / "snapshots.parquet.part")
        assert parquet.metadata.num_rows == sink.rows
        row_groups = parquet.metadata.num_row_groups
        parquet.close()
        outputs = {}
        for name in (
            "snapshots.parquet",
            "segments.jsonl",
            "quality.jsonl",
            "source_metadata.jsonl",
        ):
            partial, final = destination / (name + ".part"), destination / name
            partial.rename(final)
            outputs[name] = {"sha256": file_hash(final), "bytes": final.stat().st_size}
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "created_utc": datetime.now(UTC).isoformat(),
            "source": adapter.source,
            "symbol": adapter.symbol,
            "output_date": output_date,
            "configuration": config.to_dict(),
            "include_trade_buckets": include_trade_buckets,
            "clock_policy": {
                "sampling": "UTC epoch-aligned local receive-time grid",
                "equal_timestamps": "all messages at t included in journal order",
                "exchange_time": "retained, never used to backdate availability",
                "tail": "no rows after final raw receive time in a session",
                "staleness": "inclusive maximum age; resumption opens a new segment",
                "continuity": "faults require a fresh snapshot",
                "ordering": "run/file/message order; timestamps monotonic within each segment",
                "missing_depth": "null price AND quantity; never invented liquidity",
            },
            "inputs": inputs,
            "outputs": outputs,
            "rows": sink.rows,
            "row_groups": row_groups,
            "max_buffered_rows": sink.max_buffered_rows,
            "sample_time_min_ns": sink.minimum_time_ns,
            "sample_time_max_ns": sink.maximum_time_ns,
            "replay_counts": dict(replay.counts),
            "quality_counts": dict(sampler.counts),
            "schema": [
                {"name": field.name, "type": str(field.type), "nullable": field.nullable}
                for field in sink.schema
            ],
            "software": {
                "project": __version__,
                "python": platform.python_version(),
                "pyarrow": version("pyarrow"),
                "git_commit": _git_commit(),
            },
        }
        temporary = destination / "manifest.json.part"
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.rename(destination / "manifest.json")
        marker.unlink()
        return manifest
    except Exception as exc:
        if sink is not None:
            sink.abort()
        (destination / "failure.json").write_text(
            json.dumps({"error_type": type(exc).__name__, "error": str(exc)}, indent=2),
            encoding="utf-8",
        )
        raise
