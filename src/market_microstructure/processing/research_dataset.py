"""Bounded feature/target construction; features and future targets are separate files."""

import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from market_microstructure import __version__

from .build_dataset import _git_commit, build_dataset, file_hash
from .feature_config import FeatureConfig, positive_decimal
from .features import IDENTIFIERS, feature_table
from .labels import TrainingDiagnostics, eligible_target, label_table


def iter_research_blocks(parquet, config, interval_ms, tick_size, metrics):
    """Carry bounded history and future context, resetting at every segment boundary."""
    past = config.history_rows(interval_ms)
    future = config.future_rows(interval_ms)
    pending, history, active = None, 0, None
    seen = set()

    def drain(final):
        nonlocal pending, history
        while pending is not None and pending.num_rows > history:
            available = pending.num_rows - history
            if not final and available < config.batch_rows + future:
                break
            count = min(config.batch_rows, available)
            context = pending.slice(0, history + count + future)
            features, groups = feature_table(context, config, interval_ms)
            labels = label_table(context, config, interval_ms, tick_size)
            joined = pa.table(
                {
                    **{n: features[n] for n in features.column_names},
                    **{n: labels[n] for n in labels.column_names},
                }
            )
            yield joined.slice(history, count), groups
            end = history + count
            keep = min(past, end)
            pending = pending.slice(end - keep)
            history = keep

    for batch in parquet.iter_batches(batch_size=config.batch_rows):
        table = pa.Table.from_batches([batch])
        ids = np.asarray(table["segment_id"].to_pylist(), dtype=object)
        cuts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
        for start, end in zip(cuts[:-1], cuts[1:], strict=True):
            key = ids[start]
            part = table.slice(int(start), int(end - start))
            if key != active:
                yield from drain(True)
                if key in seen:
                    raise ValueError("a completed segment reappeared")
                seen.add(key)
                pending, history, active = None, 0, key
            pending = part if pending is None else pa.concat_tables([pending, part])
            metrics["max_context_rows"] = max(metrics["max_context_rows"], pending.num_rows)
            yield from drain(False)
    yield from drain(True)


def read_tick_size(observations, explicit=None):
    ticks = set()
    with (observations / "source_metadata.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["record_type"] == "product_metadata":
                ticks.add(positive_decimal(row["values"]["quote_increment"], "quote_increment"))
    if len(ticks) > 1:
        raise ValueError("tick size changed; process homogeneous periods separately")
    chosen = (
        positive_decimal(explicit, "tick_size") if explicit is not None else next(iter(ticks), None)
    )
    if chosen is None:
        raise ValueError("tick metadata unavailable; provide an explicitly verified tick size")
    if ticks and chosen not in ticks:
        raise ValueError("explicit tick size conflicts with captured product metadata")
    return str(chosen), "captured product metadata" if ticks else "explicit user configuration"


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def build_feature_dataset(
    observations: Path,
    destination: Path,
    config: FeatureConfig,
    *,
    tick_size=None,
    train_start_ns=0,
    train_end_ns=None,
):
    observations, destination = Path(observations).resolve(), Path(destination).resolve()
    if (observations / "_INCOMPLETE").exists():
        raise ValueError("observations are incomplete")
    manifest_path = observations / "manifest.json"
    source = json.loads(manifest_path.read_text(encoding="utf-8"))
    if source.get("status") != "complete" or not source.get("include_trade_buckets"):
        raise ValueError("require a committed observation dataset with trade buckets")
    if source["configuration"]["depth"] < 10:
        raise ValueError("features require stored depth >= 10")
    interval = source["configuration"]["interval_ms"]
    config.validate_interval(interval)
    before = {name: file_hash(observations / name) for name in source["outputs"]}
    for name, digest in before.items():
        if digest != source["outputs"][name]["sha256"]:
            raise ValueError(f"observation hash mismatch: {name}")
    source_digest = file_hash(manifest_path)
    tick, tick_origin = read_tick_size(observations, tick_size)
    diagnostics = (
        TrainingDiagnostics(config, train_start_ns, train_end_ns)
        if train_end_ns is not None
        else None
    )
    destination.mkdir(parents=True, exist_ok=False)
    marker = destination / "_INCOMPLETE"
    marker.write_text("Dataset not committed.\n")
    writers, schemas, rows, null_counts = {}, {}, {}, {}
    metrics = {
        "max_context_rows": 0,
        "context_row_bound": (
            config.history_rows(interval) + 2 * config.batch_rows + config.future_rows(interval)
        ),
    }
    groups = None

    def write(name, table):
        if name not in writers:
            schemas[name] = table.schema
            writers[name] = pq.ParquetWriter(
                destination / (name + ".part"), table.schema, version="2.6", compression="zstd"
            )
            rows[name] = 0
        writers[name].write_table(table, row_group_size=config.batch_rows)
        rows[name] += table.num_rows

    try:
        with pq.ParquetFile(observations / "snapshots.parquet") as parquet:
            for table, block_groups in iter_research_blocks(
                parquet, config, interval, tick, metrics
            ):
                groups = block_groups
                names = [n for n in table.column_names if n.startswith("f_")]
                features = table.select(list(IDENTIFIERS) + names)
                write("features.parquet", features)
                for name in names:
                    null_counts[name] = null_counts.get(name, 0) + features[name].null_count
                if diagnostics is not None:
                    diagnostics.consume(table)
                for horizon in config.horizons_ms:
                    suffix = f"{horizon}ms"
                    target = eligible_target(table, horizon).select(
                        list(IDENTIFIERS)
                        + [f"label_end_ns_{suffix}", f"y_return_{suffix}", f"y_class_{suffix}"]
                    )
                    write(f"labels_{suffix}.parquet", target)
        if groups is None:
            raise ValueError("no observations available for feature construction")
        for writer in writers.values():
            writer.close()
        writers.clear()
        if file_hash(manifest_path) != source_digest or any(
            file_hash(observations / name) != digest for name, digest in before.items()
        ):
            raise ValueError("observations changed during feature construction")
        for name in rows:
            (destination / (name + ".part")).rename(destination / name)
        catalog = {
            "groups": groups,
            "feature_columns": [name for names in groups.values() for name in names],
            "arithmetic": "float64 features; Decimal class comparisons; null undefined values",
            "flow": "net sampled top-10 displayed-depth increases/decreases, not order submissions",
            "windows": "(t-window, t], complete history inside segment required",
            "trade_side": "source adapter aggressor_side; Coinbase match.side is maker side",
            "trade_quality": (
                "unknown baseline/gap/late trade masks trade windows; references excluded"
            ),
            "depth_policy": "all requested levels required; missing depth propagates null",
            "no_trade_policy": "count/volume/imbalance zero; mean size and VWAP null",
            "normalization": "none; fitting transformations belongs to later TRAIN-only pipeline",
        }
        _write_json(destination / "feature_catalog.json", catalog)
        if diagnostics is not None:
            _write_json(destination / "training_diagnostics.json", diagnostics.report())
        names = list(rows) + ["feature_catalog.json"]
        if diagnostics is not None:
            names.append("training_diagnostics.json")
        outputs = {
            name: {
                "sha256": file_hash(destination / name),
                "bytes": (destination / name).stat().st_size,
            }
            for name in names
        }
        result = {
            "schema_version": 1,
            "kind": "features_and_labels",
            "status": "complete",
            "created_utc": datetime.now(UTC).isoformat(),
            "configuration": config.to_dict(),
            "observation_manifest": {"path": str(manifest_path), "sha256": source_digest},
            "source": source["source"],
            "symbol": source["symbol"],
            "interval_ms": interval,
            "threshold": {
                "mode": "relative_return" if config.relative_epsilon is not None else "price_ticks",
                "tick_size": tick,
                "tick_origin": tick_origin,
                "epsilon_ticks": str(config.epsilon_ticks),
                "relative_epsilon": config.relative_epsilon,
                "ties": "STABLE",
                "classes": {"DOWN": -1, "STABLE": 0, "UP": 1},
                "price_tick_return_equivalent": "epsilon_ticks * tick_size / current_midprice",
                "selection": "fixed configuration, no automatic fitting",
            },
            "label_alignment": "first observation >= t+h within tolerance; same segment only",
            "missing_endpoints": "dropped independently from each horizon target file",
            "outputs": outputs,
            "rows": rows,
            "feature_null_counts": null_counts,
            "feature_count": len(catalog["feature_columns"]),
            **metrics,
            "training_diagnostics": "explicit period only" if diagnostics else "not requested",
            "software": {
                "project": __version__,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pyarrow": pa.__version__,
                "git_commit": _git_commit(),
            },
        }
        _write_json(destination / "manifest.json.part", result)
        (destination / "manifest.json.part").rename(destination / "manifest.json")
        marker.unlink()
        return result
    except Exception as exc:
        for writer in writers.values():
            writer.close()
        _write_json(
            destination / "failure.json", {"error_type": type(exc).__name__, "error": str(exc)}
        )
        raise


def build_research_dataset(
    paths,
    destination,
    adapter,
    processing_config,
    feature_config,
    *,
    tick_size=None,
    output_date=None,
    train_start_ns=0,
    train_end_ns=None,
):
    """End-to-end raw replay -> observations -> features and horizon-specific targets."""
    if processing_config.depth < 10:
        raise ValueError("research feature pipeline requires depth >= 10")
    feature_config.validate_interval(processing_config.interval_ms)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    marker = destination / "_INCOMPLETE"
    marker.write_text("Pipeline not committed.\n")
    try:
        build_dataset(
            paths,
            destination / "observations",
            adapter,
            processing_config,
            output_date=output_date,
            include_trade_buckets=True,
        )
        research = build_feature_dataset(
            destination / "observations",
            destination / "research",
            feature_config,
            tick_size=tick_size,
            train_start_ns=train_start_ns,
            train_end_ns=train_end_ns,
        )
        result = {
            "schema_version": 1,
            "status": "complete",
            "kind": "raw_to_features_and_labels",
            "manifests": {
                name: file_hash(destination / name / "manifest.json")
                for name in ("observations", "research")
            },
            "feature_count": research["feature_count"],
            "rows": research["rows"],
        }
        _write_json(destination / "manifest.json.part", result)
        (destination / "manifest.json.part").rename(destination / "manifest.json")
        marker.unlink()
        return result
    except Exception as exc:
        _write_json(
            destination / "failure.json", {"error_type": type(exc).__name__, "error": str(exc)}
        )
        raise
