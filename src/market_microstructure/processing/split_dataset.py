"""Partition, purge, fit TRAIN artifacts, and seal held-out data without model selection."""

import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from market_microstructure import __version__

from .build_dataset import _git_commit, file_hash
from .features import IDENTIFIERS
from .joins import iter_joined
from .preprocessing import TrainPreprocessor
from .regime_cutoffs import TrainRegimeQuantiles
from .research_dataset import _write_json
from .splitting import SPLITS, make_split_plan, scan_periods, sequence_starts, split_codes

META = ("row_id", "split", "sequence_eligible", "sequence_start_row_id")


class Writers:
    def __init__(self, root, batch_rows):
        self.root, self.batch_rows = root, batch_rows
        self.handles, self.rows = {}, {}

    def write(self, name, table):
        if name not in self.handles:
            path = self.root / (name + ".part")
            path.parent.mkdir(parents=True, exist_ok=True)
            self.handles[name] = pq.ParquetWriter(
                path, table.schema, version="2.6", compression="zstd"
            )
            self.rows[name] = 0
        if table.num_rows:
            self.handles[name].write_table(table, row_group_size=self.batch_rows)
            self.rows[name] += table.num_rows

    def close(self, publish=False):
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()
        if publish:
            for name in self.rows:
                (self.root / (name + ".part")).rename(self.root / name)


def partition_features(source, destination, plan, config):
    writers = Writers(destination, config.batch_rows)
    starts, total = {}, 0
    segment_lookup = {s["segment_id"]: s for s in plan["coverage"]["segments"]}
    try:
        with pq.ParquetFile(source) as file:
            for batch in file.iter_batches(batch_size=config.batch_rows):
                table = pa.Table.from_batches([batch])
                times = table["timestamp_ns"].to_numpy()
                row_ids = np.arange(total, total + len(times), dtype=np.int64)
                codes = split_codes(times, plan)
                segment_ids = np.array(table["segment_id"].to_pylist(), dtype=object)
                segment_starts, segment_ends = (
                    np.empty(len(times), np.int64),
                    np.empty(len(times), np.int64),
                )
                for key in np.unique(segment_ids):
                    descriptor = segment_lookup[key]
                    mask = segment_ids == key
                    segment_starts[mask] = descriptor["start_row_id"]
                    segment_ends[mask] = descriptor["last_ns"]
                split_start = np.empty(len(times), np.int64)
                for code, name in enumerate(SPLITS):
                    indexes = np.flatnonzero(codes == code)
                    if len(indexes):
                        starts.setdefault(name, int(row_ids[indexes[0]]))
                        split_start[indexes] = starts[name]
                seq_start, seq_valid = sequence_starts(
                    row_ids, segment_starts, split_start, config.sequence_length
                )
                table = table.append_column("row_id", pa.array(row_ids))
                table = table.append_column(
                    "split", pa.array(np.array(SPLITS, dtype=object)[codes])
                )
                table = table.append_column("sequence_eligible", pa.array(seq_valid))
                table = table.append_column(
                    "sequence_start_row_id", pa.array(seq_start, mask=~seq_valid)
                )
                index = table.select(list(IDENTIFIERS) + list(META))
                index = index.append_column("segment_last_ns", pa.array(segment_ends))
                writers.write("index.parquet", index)
                for code, name in enumerate(SPLITS):
                    writers.write(f"{name}/features.parquet", table.filter(pa.array(codes == code)))
                total += len(times)
        if total != plan["coverage"]["rows"]:
            raise ValueError("feature row count changed while partitioning")
        for name in SPLITS:
            count = writers.rows[f"{name}/features.parquet"]
            if not count:
                raise ValueError(f"no feature rows in {name}; supply different explicit boundaries")
            plan["ranges"][name].update(
                rows=count, start_row_id=starts[name], end_row_exclusive=starts[name] + count
            )
        writers.close(publish=True)
        return writers.rows
    except Exception:
        writers.close()
        raise


def partition_targets(source, destination, horizon, plan, config, tolerance_ms):
    suffix = f"{horizon}ms"
    names = (
        list(IDENTIFIERS)
        + list(META)
        + [f"label_end_ns_{suffix}", f"y_return_{suffix}", f"y_class_{suffix}"]
    )
    writers = Writers(destination, config.batch_rows)
    counters = {s: {"before_purge": 0, "purged": 0, "rows": 0, "sequence_rows": 0} for s in SPLITS}
    with pq.ParquetFile(destination / "index.parquet") as index, pq.ParquetFile(source) as target:
        expected = list(IDENTIFIERS) + names[-3:]
        if target.schema_arrow.names != expected:
            raise ValueError("unexpected target schema")
        schema = pa.schema(
            list(index.schema_arrow)[:-1] + [target.schema_arrow.field(n) for n in names[-3:]]
        )
    empty = pa.Table.from_batches([], schema=schema)
    try:
        for split in SPLITS:
            writers.write(f"{split}/labels_{suffix}.parquet", empty)
        for table in iter_joined(destination / "index.parquet", source, config.batch_rows):
            if any(table[n].null_count for n in names[-3:]):
                raise ValueError("null future targets must have been removed")
            times = table["timestamp_ns"].to_numpy()
            ends = table[f"label_end_ns_{suffix}"].to_numpy()
            returns = table[f"y_return_{suffix}"].to_numpy()
            classes = table[f"y_class_{suffix}"].to_numpy()
            if (
                np.any(ends - times < horizon * 1_000_000)
                or np.any(ends - times > (horizon + tolerance_ms) * 1_000_000)
                or np.any(ends > table["segment_last_ns"].to_numpy())
                or np.any((ends - times) % plan["interval_ns"] != 0)
                or np.any(~np.isfinite(returns))
                or np.any(~np.isin(classes, [-1, 0, 1]))
            ):
                raise ValueError("invalid horizon, segment endpoint or target value")
            roles = np.array(table["split"].to_pylist(), dtype=object)
            for split in SPLITS:
                belongs = roles == split
                keep = belongs & (ends < plan["ranges"][split]["end_exclusive_ns"])
                counters[split]["before_purge"] += int(belongs.sum())
                counters[split]["purged"] += int((belongs & ~keep).sum())
                selected = table.filter(pa.array(keep)).select(names)
                writers.write(f"{split}/labels_{suffix}.parquet", selected)
                counters[split]["rows"] += selected.num_rows
                counters[split]["sequence_rows"] += sum(selected["sequence_eligible"].to_pylist())
        writers.close(publish=True)
        return counters, writers.rows
    except Exception:
        writers.close()
        raise


def build_split_dataset(source, destination, config, *, horizons_ms=None):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if (source / "_INCOMPLETE").exists():
        raise ValueError("input research dataset is incomplete")
    source_manifest = source / "manifest.json"
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("kind") != "features_and_labels":
        raise ValueError("a committed features-and-labels dataset is required")
    horizons = tuple(horizons_ms or manifest["configuration"]["horizons_ms"])
    available = manifest["configuration"]["horizons_ms"]
    if (
        not horizons
        or len(set(horizons)) != len(horizons)
        or any(type(h) is not int or h not in available for h in horizons)
    ):
        raise ValueError("choose unique available target horizons")
    hashes = {name: file_hash(source / name) for name in manifest["outputs"]}
    if any(hashes[name] != spec["sha256"] for name, spec in manifest["outputs"].items()):
        raise ValueError("source artifact hash mismatch")
    source_digest = file_hash(source_manifest)
    catalog = json.loads((source / "feature_catalog.json").read_text(encoding="utf-8"))
    features = catalog["feature_columns"]
    if (
        not features
        or len(features) != len(set(features))
        or any(not n.startswith("f_") for n in features)
    ):
        raise ValueError("invalid feature catalog")
    with pq.ParquetFile(source / "features.parquet") as file:
        if set(file.schema_arrow.names) != set(IDENTIFIERS) | set(features):
            raise ValueError("feature schema does not match catalog or contains target columns")
        if any(file.schema_arrow.field(n).type != pa.float64() for n in features):
            raise ValueError("expected float64 feature columns")
    interval = manifest["interval_ms"] * 1_000_000
    periods = scan_periods(source / "features.parquet", interval, config.batch_rows)
    plan = make_split_plan(periods, config, interval)
    # Prior threshold/class diagnostics must not have inspected this plan's held-out periods.
    if "training_diagnostics.json" in hashes:
        prior = json.loads((source / "training_diagnostics.json").read_text(encoding="utf-8"))
        train = plan["ranges"]["train"]
        if (
            prior["training_start_ns"] < train["start_ns"]
            or prior["training_end_exclusive_ns"] > train["end_exclusive_ns"]
        ):
            raise ValueError(
                "prior diagnostics extend beyond planned TRAIN; rebuild without those diagnostics"
            )
    destination.mkdir(parents=True, exist_ok=False)
    marker = destination / "_INCOMPLETE"
    marker.write_text("Split dataset not committed.\n", encoding="utf-8")
    try:
        rows = partition_features(source / "features.parquet", destination, plan, config)
        _write_json(destination / "split_plan.json", plan)
        _write_json(destination / "feature_catalog.json", catalog)
        counts, normalizers, train_reports, regimes = {}, {}, {}, {}
        for horizon in horizons:
            suffix = f"{horizon}ms"
            counts[suffix], written = partition_targets(
                source / f"labels_{suffix}.parquet",
                destination,
                horizon,
                plan,
                config,
                manifest["configuration"]["alignment_tolerance_ms"],
            )
            rows.update(written)
            train = plan["ranges"]["train"]
            fitter = TrainPreprocessor(
                features, horizon, train["start_ns"], train["end_exclusive_ns"]
            )
            quantiles = TrainRegimeQuantiles(
                destination / f"regimes_{suffix}.sqlite.part",
                horizon,
                train["start_ns"],
                train["end_exclusive_ns"],
                config.regime_quantiles,
            )
            balance = {"DOWN": 0, "STABLE": 0, "UP": 0}
            try:
                for table in iter_joined(
                    destination / "train/features.parquet",
                    destination / f"train/labels_{suffix}.parquet",
                    config.batch_rows,
                ):
                    fitter.update(table)
                    quantiles.update(table)
                    labels = table[f"y_class_{suffix}"].to_numpy()
                    for name, value in (("DOWN", -1), ("STABLE", 0), ("UP", 1)):
                        balance[name] += int((labels == value).sum())
                regime = quantiles.finish()
            finally:
                quantiles.close()
            regime_path = f"regime_cutoffs_{suffix}.json"
            _write_json(destination / regime_path, regime)
            regimes[suffix] = regime_path
            if fitter.rows:
                path = f"preprocessor_{suffix}.json"
                _write_json(destination / path, fitter.finalize())
                normalizers[suffix] = path
            else:
                normalizers[suffix] = None
            train_reports[suffix] = {
                "eligible_train_rows": fitter.rows,
                "class_counts": balance,
                "preprocessing_available": bool(fitter.rows),
            }
        _write_json(
            destination / "training_report.json",
            {
                "fit_scope": "purged TRAIN per horizon; held-out class distributions not computed",
                "threshold": manifest["threshold"],
                "threshold_policy": "inherited and frozen, never refitted",
                "horizons": train_reports,
            },
        )
        if file_hash(source_manifest) != source_digest or any(
            file_hash(source / n) != v for n, v in hashes.items()
        ):
            raise ValueError("source artifacts changed while splitting")
        outputs = {}
        for path in sorted(destination.rglob("*")):
            if path.is_file() and path.name != "_INCOMPLETE":
                if path.name.endswith(".part"):
                    raise ValueError("unfinished output remains")
                name = path.relative_to(destination).as_posix()
                outputs[name] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
        seal = {
            "schema_version": 1,
            "status": "sealed",
            "split": "test",
            "boundary": plan["ranges"]["test"],
            "split_plan_sha256": outputs["split_plan.json"]["sha256"],
            "threshold": manifest["threshold"],
            "files": {n: spec for n, spec in outputs.items() if n.startswith("test/")},
            "policy": (
                "no test fitting or model selection; explicit allow_test=True required by loader"
            ),
            "scope": "workflow guard and integrity hashes, not encryption or access control",
        }
        _write_json(destination / "sealed_test_manifest.json", seal)
        outputs["sealed_test_manifest.json"] = {
            "sha256": file_hash(destination / "sealed_test_manifest.json"),
            "bytes": (destination / "sealed_test_manifest.json").stat().st_size,
        }
        result = {
            "schema_version": 1,
            "kind": "chronological_splits",
            "status": "complete",
            "created_utc": datetime.now(UTC).isoformat(),
            "configuration": config.to_dict(),
            "input_manifest": {"path": str(source_manifest), "sha256": source_digest},
            "feature_columns": features,
            "horizons_ms": list(horizons),
            "method": plan["method"],
            "ranges": plan["ranges"],
            "rows": rows,
            "target_counts": counts,
            "normalizers": normalizers,
            "regime_cutoffs": regimes,
            "three_way_evaluation_available": {
                h: all(c[s]["rows"] > 0 for s in SPLITS) for h, c in counts.items()
            },
            "three_way_sequence_evaluation_available": {
                h: all(c[s]["sequence_rows"] > 0 for s in SPLITS) for h, c in counts.items()
            },
            "outputs": outputs,
            "memory_policy": (
                "bounded Arrow batches, O(features) fit state, disk-indexed exact TRAIN quantiles"
            ),
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
        _write_json(
            destination / "failure.json", {"error_type": type(exc).__name__, "error": str(exc)}
        )
        raise


def iter_model_batches(
    dataset,
    horizon_ms,
    split,
    *,
    batch_rows=4096,
    transformed=True,
    sequence_only=False,
    allow_test=False,
):
    """Guarded model-facing loader. Explicit feature schema excludes labels and metadata."""
    if split not in SPLITS:
        raise ValueError("unknown split")
    if split == "test" and not allow_test:
        raise PermissionError(
            "TEST is sealed; explicit allow_test=True is reserved for final evaluation"
        )
    dataset = Path(dataset)
    if (dataset / "_INCOMPLETE").exists():
        raise ValueError("split dataset is incomplete")
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("kind") != "chronological_splits":
        raise ValueError("require a committed split dataset")
    if type(horizon_ms) is not int or horizon_ms not in manifest["horizons_ms"]:
        raise ValueError("unknown horizon")
    suffix = f"{horizon_ms}ms"
    paths = [f"{split}/features.parquet", f"{split}/labels_{suffix}.parquet"]
    state_path = manifest["normalizers"][suffix]
    if transformed:
        if state_path is None:
            raise ValueError("no eligible TRAIN rows for this horizon")
        paths.append(state_path)
    if split == "test":
        paths.append("sealed_test_manifest.json")
    for name in paths:
        if file_hash(dataset / name) != manifest["outputs"][name]["sha256"]:
            raise ValueError(f"split artifact hash mismatch: {name}")
    state = json.loads((dataset / state_path).read_text(encoding="utf-8")) if transformed else None
    for table in iter_joined(dataset / paths[0], dataset / paths[1], batch_rows):
        if sequence_only:
            table = table.filter(table["sequence_eligible"])
        if not table.num_rows:
            continue
        if transformed:
            matrix = TrainPreprocessor.transform(table, state)
            for i, name in enumerate(state["feature_columns"]):
                table = table.set_column(
                    table.schema.get_field_index(name), name, pa.array(matrix[:, i])
                )
        yield table
