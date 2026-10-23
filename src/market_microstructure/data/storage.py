"""Bounded, rotating JSONL storage. Disk failures are fatal, never network retries."""

import json
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


class StorageError(RuntimeError):
    """Writing or finalizing a recording failed; a .part file may need inspection."""


class JsonlWriter:
    """A single writer; UTC receive-time buckets plus a soft per-file byte limit.

    Every line is flushed; fsync is periodic and always performed on clean closure.
    One oversized record is allowed intact. A crash leaves .part files visible.
    Compression is optional and synchronous on closed files; raw originals survive
    any compression failure. Storage paths are never silently overwritten.
    """

    def __init__(
        self,
        root: Path,
        symbol: str,
        *,
        source: str = "coinbase",
        rotate_seconds: int = 3600,
        max_bytes: int = 128 * 1024 * 1024,
        fsync_seconds: float = 1.0,
        compress: bool = False,
        run_id: str | None = None,
    ):
        if not re.fullmatch(r"[A-Z0-9]+-[A-Z0-9]+", symbol):
            raise ValueError("unsafe symbol")
        if not re.fullmatch(r"[a-z0-9_]+", source):
            raise ValueError("unsafe source")
        if rotate_seconds <= 0 or max_bytes <= 0 or fsync_seconds < 0:
            raise ValueError("invalid rotation/durability settings")
        self.root, self.symbol, self.source = Path(root), symbol, source
        self.rotate_seconds, self.max_bytes = rotate_seconds, max_bytes
        self.fsync_seconds, self.compress = fsync_seconds, compress
        self.run_id = run_id or uuid4().hex
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", self.run_id):
            raise ValueError("unsafe run ID")
        if compress:
            import zstandard  # noqa: F401
        self._file = None
        self._path: Path | None = None
        self._bucket = None
        self._size = 0
        self._file_index = 0
        self._last_sync = time.monotonic()
        self.failed = False
        self.closed = False
        self.records = 0
        self.files_closed = 0

    def _open(self, receive_time_ns: int):
        moment = datetime.fromtimestamp(receive_time_ns // 1_000_000_000, UTC)
        directory = self.root / moment.strftime("%Y-%m-%d")
        directory.mkdir(parents=True, exist_ok=True)
        name = (
            f"{self.source}_{self.symbol}_{moment:%Y%m%d_%H%M%S}_"
            f"{self.run_id}_{self._file_index:06d}.jsonl.part"
        )
        self._file_index += 1
        self._path = directory / name
        self._file = self._path.open("xb")
        self._bucket = receive_time_ns // (self.rotate_seconds * 1_000_000_000)
        self._size = 0

    def _finalize(self):
        if self._file is None:
            return
        handle, path = self._file, self._path
        self._file = None
        try:
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            handle.close()
        if self.failed:
            return
        complete = path.with_suffix("")
        if complete.exists():
            raise OSError(f"refusing to overwrite {complete}")
        path.rename(complete)
        self.files_closed += 1
        if self.compress:
            import zstandard

            temporary = complete.with_suffix(".jsonl.zst.part")
            destination = complete.with_suffix(".jsonl.zst")
            if destination.exists():
                raise OSError(f"refusing to overwrite {destination}")
            with complete.open("rb") as source, temporary.open("xb") as target:
                zstandard.ZstdCompressor(level=3).copy_stream(source, target)
                target.flush()
                os.fsync(target.fileno())
            temporary.rename(destination)
            complete.unlink()  # Only this writer's completed, successfully compressed file.

    def write(self, record: dict):
        if self.failed or self.closed:
            raise StorageError("writer is failed or closed")
        try:
            encoded = (json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n").encode(
                "utf-8"
            )
            received = record["receive_time_ns"]
            if isinstance(received, bool) or not isinstance(received, int) or received < 0:
                raise ValueError("record requires nonnegative integer receive_time_ns")
            bucket = received // (self.rotate_seconds * 1_000_000_000)
            if self._file is not None and (
                bucket != self._bucket or self._size + len(encoded) > self.max_bytes
            ):
                self._finalize()
            if self._file is None:
                self._open(received)
            written = self._file.write(encoded)
            if written != len(encoded):
                raise OSError("short recording write")
            self._file.flush()
            now = time.monotonic()
            if now - self._last_sync >= self.fsync_seconds:
                os.fsync(self._file.fileno())
                self._last_sync = now
            self._size += len(encoded)
            self.records += 1
        except Exception as exc:
            self.failed = True
            raise StorageError(f"raw recording failed: {exc}") from exc

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self._finalize()
        except Exception as exc:
            self.failed = True
            raise StorageError(f"recording finalization failed: {exc}") from exc


def iter_records(paths):
    """Stream explicitly supplied closed JSONL or zstd files; reject partial tails."""
    import io

    for value in paths:
        path = Path(value)
        if path.name.endswith(".part"):
            raise StorageError(f"incomplete capture must be audited before replay: {path}")
        with path.open("rb") as source:
            if path.name.endswith(".zst"):
                import zstandard

                with zstandard.ZstdDecompressor().stream_reader(source) as reader:
                    with io.TextIOWrapper(reader, encoding="utf-8") as text:
                        yield from _read_lines(text, path)
            else:
                with io.TextIOWrapper(source, encoding="utf-8") as text:
                    yield from _read_lines(text, path)


def _read_lines(stream, path):
    for number, line in enumerate(stream, 1):
        if not line.endswith("\n"):
            raise StorageError(f"truncated JSONL line: {path}:{number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise StorageError(f"invalid JSONL: {path}:{number}") from exc
        if not isinstance(value, dict):
            raise StorageError(f"non-object record: {path}:{number}")
        yield value
