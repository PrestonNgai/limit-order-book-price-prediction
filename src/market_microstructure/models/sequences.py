"""Lazy history slices from one bounded feature matrix, using the frozen split mask."""

import hashlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from market_microstructure.processing.features import IDENTIFIERS
from market_microstructure.processing.preprocessing import TrainPreprocessor
from market_microstructure.processing.split_dataset import META, iter_model_batches

from .experiments import artifact, read_json, row_fingerprint
from .learned import float32_matrix


class SequenceDataset(Dataset):
    """Store O(N*F) base features, not O(N*L*F) overlapping windows."""

    def __init__(self, matrix, endpoints, targets, length, *, snapshot=False):
        endpoints = np.asarray(endpoints, dtype=np.int64)
        targets = np.asarray(targets)
        if type(length) is not int or length < 1 or matrix.ndim != 2:
            raise ValueError("invalid sequence matrix or length")
        if len(endpoints) != len(targets) or not np.isin(targets, [-1, 0, 1]).all():
            raise ValueError("invalid sequence targets")
        if (
            np.any(endpoints < length - 1)
            or np.any(endpoints >= len(matrix))
            or np.any(np.diff(endpoints) <= 0)
            or not np.isfinite(matrix).all()
        ):
            raise ValueError("invalid sequence endpoints or features")
        with np.errstate(over="ignore"):
            self.matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        if not np.isfinite(self.matrix).all():
            raise ValueError("sequence features overflow float32")
        self.features = torch.from_numpy(self.matrix)
        self.endpoints = endpoints
        self.targets = torch.from_numpy(targets.astype(np.int64) + 1)
        self.length, self.snapshot = length, snapshot

    def __len__(self):
        return len(self.endpoints)

    def __getitem__(self, index):
        end = int(self.endpoints[index])
        x = self.features[end] if self.snapshot else self.features[end - self.length + 1 : end + 1]
        return x, self.targets[index]

    def batch(self, start, stop):
        return torch.stack([self[i][0] for i in range(start, stop)]).numpy()


def sequence_plan(dataset, manifest, config):
    path = artifact(dataset, "split_plan.json", manifest["outputs"]["split_plan.json"]["sha256"])
    plan = read_json(path)
    if (
        plan["sequence_length"] != config.sequence_length
        or manifest["configuration"]["sequence_length"] != config.sequence_length
    ):
        raise ValueError(
            "sequence_length differs from frozen split; rebuild splits for another length"
        )
    if plan["ranges"] != manifest["ranges"] or plan["interval_ns"] <= 0:
        raise ValueError("split plan differs from manifest")
    return plan


def load_sequences(dataset, manifest, prep, config, split, *, allow_test=False):
    if split not in ("train", "validation", "test"):
        raise ValueError("invalid sequence split")
    if split == "test" and not allow_test:
        raise PermissionError("TEST is sealed; explicit final evaluation opt-in is required")
    dataset = Path(dataset)
    plan = sequence_plan(dataset, manifest, config)
    bound, length = manifest["ranges"][split], config.sequence_length
    count = bound["rows"]
    limit = config.max_train_rows if split == "train" else config.max_eval_rows
    if (
        count > limit
        or count * (len(prep["feature_columns"]) * 32 + 256) > config.max_fit_mb * 1024**2
    ):
        raise ValueError("sequence features exceed configured row/memory limit")
    if prep["feature_columns"] != manifest["feature_columns"]:
        raise ValueError("sequence preprocessor schema mismatch")
    suffix = f"{config.horizon_ms}ms"
    if config.horizon_ms not in manifest["horizons_ms"]:
        raise ValueError("unknown horizon")
    paths = [f"{split}/features.parquet", f"{split}/labels_{suffix}.parquet"]
    if split == "test":
        paths.append("sealed_test_manifest.json")
    for name in paths:
        artifact(dataset, name, manifest["outputs"][name]["sha256"])
    matrix = np.empty((count, len(prep["feature_columns"])), dtype=np.float32)
    pieces, offset = [], 0
    with pq.ParquetFile(dataset / paths[0]) as file:
        for batch in file.iter_batches(batch_size=config.batch_rows):
            table = pa.Table.from_batches([batch])
            end = offset + table.num_rows
            if end > count:
                raise ValueError("feature row count exceeds manifest")
            matrix[offset:end] = float32_matrix(table, prep)
            pieces.append(table.select(list(IDENTIFIERS) + list(META)))
            offset = end
    if offset != count:
        raise ValueError("feature count differs from manifest")
    meta = pa.concat_tables(pieces)
    if any(meta[n].null_count for n in meta.column_names if n != "sequence_start_row_id"):
        raise ValueError("null sequence identity or eligibility")
    ids = meta["row_id"].to_numpy()
    times = meta["timestamp_ns"].to_numpy()
    if (
        not np.array_equal(ids, np.arange(bound["start_row_id"], bound["end_row_exclusive"]))
        or set(meta["split"].to_pylist()) != {split}
        or np.any(np.diff(times) <= 0)
        or np.any(times < bound["start_ns"])
        or np.any(times >= bound["end_exclusive_ns"])
    ):
        raise ValueError("sequence feature rows escape split or chronological order")
    keys = list(
        zip(
            *(
                meta[n].to_pylist()
                for n in ("source", "symbol", "run_id", "session_id", "segment_id")
            ),
            strict=True,
        )
    )
    starts = np.empty(count, dtype=np.int64)
    seen, start = set(), 0
    for i, key in enumerate(keys):
        if i == 0 or key != keys[i - 1]:
            if key in seen:
                raise ValueError("sequence segment reappeared")
            seen.add(key)
            start = i
        elif times[i] - times[i - 1] != plan["interval_ns"]:
            raise ValueError("timestamp gap inside a sequence segment")
        starts[i] = start
    expected_mask = np.arange(count) - length + 1 >= starts
    if not np.array_equal(expected_mask, meta["sequence_eligible"].to_numpy()):
        raise ValueError("stored sequence mask crosses split/segment boundary")
    stored_starts = meta["sequence_start_row_id"].to_pylist()
    for i, valid in enumerate(expected_mask):
        expected = int(ids[i] - length + 1) if valid else None
        if stored_starts[i] != expected:
            raise ValueError("stored sequence start is inconsistent")
    labels, raw_count = [], 0
    checker = None
    if split == "train":
        checker = TrainPreprocessor(
            prep["feature_columns"], config.horizon_ms, bound["start_ns"], bound["end_exclusive_ns"]
        )
    for table in iter_model_batches(
        dataset,
        config.horizon_ms,
        split,
        batch_rows=config.batch_rows,
        transformed=False,
        allow_test=allow_test,
    ):
        raw_count += table.num_rows
        if checker:
            checker.update(table)
        selected = table.filter(table["sequence_eligible"])
        labels.append(
            selected.select(
                list(IDENTIFIERS)
                + list(META)
                + [f"label_end_ns_{suffix}", f"y_return_{suffix}", f"y_class_{suffix}"]
            )
        )
    counts = manifest["target_counts"][suffix][split]
    if raw_count != counts["rows"]:
        raise ValueError("target count differs from split manifest")
    if checker and checker.finalize()["fit_row_sha256"] != prep["fit_row_sha256"]:
        raise ValueError("TRAIN values differ from preprocessing provenance")
    if labels:
        labels = pa.concat_tables(labels)
    else:
        with pq.ParquetFile(dataset / paths[1]) as file:
            labels = pa.Table.from_batches([], schema=file.schema_arrow)
    if len(labels) != counts["sequence_rows"]:
        raise ValueError("sequence target count differs from manifest")
    endpoints = labels["row_id"].to_numpy() - bound["start_row_id"]
    if len(endpoints):
        if (
            np.any(endpoints < length - 1)
            or np.any(endpoints >= count)
            or not np.all(expected_mask[endpoints])
        ):
            raise ValueError("target references an ineligible history")
        for name in IDENTIFIERS:
            if not meta.take(pa.array(endpoints))[name].equals(labels[name]):
                raise ValueError("sequence target identity mismatch")
        ends = labels[f"label_end_ns_{suffix}"].to_numpy()
        if np.any(ends >= bound["end_exclusive_ns"]) or np.any(ends <= times[endpoints]):
            raise ValueError("sequence targets cross split boundary")
        future = np.searchsorted(times, ends)
        if np.any(future >= count) or np.any(times[future] != ends):
            raise ValueError("future target is not an observed endpoint")
        if any(keys[int(a)] != keys[int(b)] for a, b in zip(endpoints, future, strict=True)):
            raise ValueError("future target crosses a segment")
        if np.any(ends - times[endpoints] < config.horizon_ms * 1_000_000):
            raise ValueError("future target is earlier than its horizon")
    y = labels[f"y_class_{suffix}"].to_numpy()
    view = SequenceDataset(matrix, endpoints, y, length, snapshot=config.model == "sequence_mlp")
    digest = hashlib.sha256()
    row_fingerprint(digest, labels, config.horizon_ms)
    for name in paths:
        artifact(dataset, name, manifest["outputs"][name]["sha256"])
    return (
        view,
        labels,
        {
            "rows": len(view),
            "base_feature_rows": count,
            "sequence_length": length,
            "row_sha256": digest.hexdigest(),
            "base_matrix_bytes": matrix.nbytes,
            "window_policy": "past L observations including t; one split/segment",
        },
    )
