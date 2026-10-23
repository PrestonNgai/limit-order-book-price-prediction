"""Metadata-only chronological planning. Never inspect labels to choose boundaries."""

from decimal import Decimal

import numpy as np
import pyarrow.parquet as pq

from .features import IDENTIFIERS

SPLITS = ("train", "validation", "test")


def scan_periods(path, interval_ns, batch_rows):
    segments, sessions = [], []
    seen_segments, seen_sessions = set(), set()
    previous_time, product, row_id = None, None, 0
    with pq.ParquetFile(path) as file:
        for batch in file.iter_batches(batch_size=batch_rows, columns=list(IDENTIFIERS)):
            for name in IDENTIFIERS:
                if batch.column(name).null_count:
                    raise ValueError("null observation identifier")
            times = batch.column("timestamp_ns").to_numpy()
            if (
                np.any(times < 0)
                or np.any(np.diff(times) <= 0)
                or (previous_time is not None and times[0] <= previous_time)
            ):
                raise ValueError(
                    "timestamps must increase globally; overlapping or regressed periods rejected"
                )
            previous_time = int(times[-1])
            ids = {
                n: np.array(batch.column(n).to_pylist(), dtype=object)
                for n in ("source", "symbol", "run_id", "session_id", "segment_id")
            }
            products = set(zip(ids["source"], ids["symbol"], strict=True))
            if len(products) != 1 or (product is not None and product not in products):
                raise ValueError("one source/product is required")
            product = next(iter(products))
            keys = np.array([str(x) for x in ids["segment_id"]], dtype=object)
            cuts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1, len(keys)]
            for left, right in zip(cuts[:-1], cuts[1:], strict=True):
                left, right = int(left), int(right)
                session = (str(ids["run_id"][left]), str(ids["session_id"][left]))
                segment = str(keys[left])
                for name, expected in (("run_id", session[0]), ("session_id", session[1])):
                    if np.any(ids[name][left:right] != expected):
                        raise ValueError("segment identity changed within its rows")
                if np.any(np.diff(times[left:right]) != interval_ns):
                    raise ValueError("sampling gap inside an uninterrupted segment")
                if not segments or segments[-1]["segment_id"] != segment:
                    if segment in seen_segments:
                        raise ValueError("completed segment reappeared")
                    seen_segments.add(segment)
                    segments.append(
                        {
                            "segment_id": segment,
                            "run_id": session[0],
                            "session_id": session[1],
                            "start_row_id": row_id + left,
                            "start_ns": int(times[left]),
                            "rows": 0,
                        }
                    )
                else:
                    last = segments[-1]
                    if (last["run_id"], last["session_id"]) != session or int(times[left]) != last[
                        "last_ns"
                    ] + interval_ns:
                        raise ValueError("inconsistent segment continuation")
                segments[-1].update(
                    last_ns=int(times[right - 1]), rows=segments[-1]["rows"] + right - left
                )
                if not sessions or (sessions[-1]["run_id"], sessions[-1]["session_id"]) != session:
                    if session in seen_sessions:
                        raise ValueError("completed session reappeared")
                    seen_sessions.add(session)
                    sessions.append(
                        {
                            "run_id": session[0],
                            "session_id": session[1],
                            "start_row_id": row_id + left,
                            "start_ns": int(times[left]),
                            "rows": 0,
                        }
                    )
                sessions[-1].update(
                    last_ns=int(times[right - 1]), rows=sessions[-1]["rows"] + right - left
                )
            row_id += len(times)
    if row_id < 3:
        raise ValueError("at least three chronological observations are required")
    if previous_time + interval_ns >= 2**63:
        raise ValueError("coverage end timestamp overflow")
    return {
        "rows": row_id,
        "source": product[0],
        "symbol": product[1],
        "start_ns": segments[0]["start_ns"],
        "end_ns": previous_time + interval_ns,
        "segments": segments,
        "sessions": sessions,
    }


def make_split_plan(periods, config, interval_ns):
    sessions = periods["sessions"]
    mode = config.mode
    if mode == "auto":
        mode = "sessions" if len(sessions) >= 3 else "time"
    a, b = Decimal(str(config.train_fraction)), Decimal(str(config.validation_fraction))
    if mode == "sessions":
        if len(sessions) < 3:
            raise ValueError("whole-session splitting requires at least three sessions")
        total = periods["rows"]
        candidates = range(1, len(sessions) - 1)
        first = min(
            candidates, key=lambda i: (abs(Decimal(sessions[i]["start_row_id"]) - total * a), i)
        )
        second = min(
            range(first + 1, len(sessions)),
            key=lambda i: (abs(Decimal(sessions[i]["start_row_id"]) - total * (a + b)), i),
        )
        train_end, validation_end = sessions[first]["start_ns"], sessions[second]["start_ns"]
    elif mode == "time":
        duration = periods["end_ns"] - periods["start_ns"]
        train_end = periods["start_ns"] + int(duration * a)
        validation_end = periods["start_ns"] + int(duration * (a + b))
    else:
        train_end, validation_end = config.train_end_ns, config.validation_end_ns
    if not periods["start_ns"] < train_end < validation_end < periods["end_ns"]:
        raise ValueError("split boundaries must be inside observed coverage")
    boundaries = (periods["start_ns"], train_end, validation_end, periods["end_ns"])
    ranges = {
        name: {"start_ns": int(boundaries[i]), "end_exclusive_ns": int(boundaries[i + 1])}
        for i, name in enumerate(SPLITS)
    }
    return {
        "schema_version": 1,
        "method": mode,
        "interval_ns": interval_ns,
        "sequence_length": config.sequence_length,
        "configuration": config.to_dict(),
        "ranges": ranges,
        "coverage": periods,
        "session_policy": "nearest feasible cumulative row fractions; chronological whole sessions",
        "time_policy": "70/15/15 defaults over [first sample, last sample + interval)",
        "purge_policy": "label endpoint must be strictly before its split end",
        "sequence_policy": "all input rows in one split and one uninterrupted segment",
    }


def split_codes(times, plan):
    return np.searchsorted(
        [
            plan["ranges"]["train"]["end_exclusive_ns"],
            plan["ranges"]["validation"]["end_exclusive_ns"],
        ],
        times,
        side="right",
    )


def sequence_starts(row_ids, segment_starts, split_starts, length):
    starts = row_ids - length + 1
    valid = (starts >= segment_starts) & (starts >= split_starts)
    return starts, valid
