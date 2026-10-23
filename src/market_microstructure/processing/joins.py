"""Streaming sorted subset joins; no unconstrained in-memory feature/label merge."""

from contextlib import ExitStack

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .features import IDENTIFIERS


class TargetCursor:
    def __init__(self, parquet, batch_rows):
        self.iterator = iter(parquet.iter_batches(batch_size=batch_rows))
        self.schema = parquet.schema_arrow
        self.pending = None
        self.previous_time = None
        self._advance()

    def _advance(self):
        batch = next(self.iterator, None)
        if batch is None:
            self.pending = None
            return
        self.pending = pa.Table.from_batches([batch])
        for name in IDENTIFIERS:
            if self.pending[name].null_count:
                raise ValueError("null target identifier")
        times = self.pending["timestamp_ns"].to_numpy()
        if np.any(np.diff(times) <= 0) or (
            self.previous_time is not None and times[0] <= self.previous_time
        ):
            raise ValueError("targets must be globally ordered and unique")
        self.previous_time = int(times[-1])

    def through(self, inclusive_ns, maximum_rows):
        pieces, count = [], 0
        while self.pending is not None:
            times = self.pending["timestamp_ns"].to_numpy()
            size = int(np.searchsorted(times, inclusive_ns, side="right"))
            if not size:
                break
            pieces.append(self.pending.slice(0, size))
            count += size
            if count > maximum_rows:
                raise ValueError("more targets than observations in the matching interval")
            if size == self.pending.num_rows:
                self._advance()
            else:
                self.pending = self.pending.slice(size)
                break
        return pa.concat_tables(pieces) if pieces else pa.Table.from_batches([], schema=self.schema)


def iter_joined(features_path, targets_path, batch_rows, *, columns=None):
    with ExitStack() as stack:
        features = stack.enter_context(pq.ParquetFile(features_path))
        targets = stack.enter_context(pq.ParquetFile(targets_path))
        cursor = TargetCursor(targets, batch_rows)
        for batch in features.iter_batches(batch_size=batch_rows, columns=columns):
            table = pa.Table.from_batches([batch])
            times = table["timestamp_ns"].to_numpy()
            selected = cursor.through(int(times[-1]), table.num_rows)
            if not selected.num_rows:
                continue
            target_times = selected["timestamp_ns"].to_numpy()
            indexes = np.searchsorted(times, target_times)
            if np.any(indexes >= len(times)) or np.any(times[indexes] != target_times):
                raise ValueError("target does not match an observation")
            matched = table.take(pa.array(indexes))
            for name in IDENTIFIERS:
                if not matched[name].equals(selected[name]):
                    raise ValueError(f"target/feature identity mismatch: {name}")
            for name in selected.column_names:
                if name not in IDENTIFIERS:
                    if name in matched.column_names:
                        if not matched[name].equals(selected[name]):
                            raise ValueError(f"target metadata mismatch: {name}")
                    else:
                        matched = matched.append_column(name, selected[name])
            yield matched
        if cursor.pending is not None:
            raise ValueError("target after final observation")
