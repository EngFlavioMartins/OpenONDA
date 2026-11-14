"""Durable text, JSON, NumPy archive, CSV and JSON-lines output shared by the solvers.

Whole files are replaced atomically: the content is written to a temporary
file in the destination directory, synchronized to disk and renamed over the
destination, so a reader never opens a partially written file. A failed write
leaves the old file and no temporary. :func:`atomic_path` is the one
implementation; the text, JSON, ``.npz``, HDF5 and VTK writers use it. Time series
are appended one record at a time; a record that fails to write is truncated
so the file keeps a valid prefix. After a restart, ``rewind_*`` keeps the
records up to and including the restored time.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
import csv
import errno
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .clock import time_tolerance

__all__ = [
    "InsufficientStorageError",
    "append_csv_rows",
    "append_jsonl",
    "atomic_path",
    "npz_size_bound",
    "require_free_space",
    "rewind_csv",
    "rewind_jsonl",
    "write_json_atomic",
    "write_npz_atomic",
    "write_text_atomic",
]

_DEFAULT_RESERVE_BYTES = 1 << 30


class InsufficientStorageError(OSError):
    """Raised before an atomic output cannot fit alongside its old version."""

    def __init__(self, path: str | Path, required_bytes: int, free_bytes: int) -> None:
        """Describe a failed atomic-write capacity check.

        Parameters
        ----------
        path : str or pathlib.Path
            Output path whose replacement could not be staged safely.
        required_bytes : int
            Bytes needed for the new file plus the configured reserve.
        free_bytes : int
            Bytes available when the check ran.
        """
        self.path = Path(path)
        self.required_bytes = int(required_bytes)
        self.free_bytes = int(free_bytes)
        super().__init__(
            errno.ENOSPC,
            (
                f"insufficient free space for atomic output at {self.path}: "
                f"need {self.required_bytes} bytes including reserve, "
                f"have {self.free_bytes} bytes"
            ),
            str(self.path),
        )


def require_free_space(
    path: str | Path,
    archive_bytes: int,
) -> int:
    """Verify an atomic temporary archive can fit and return current free bytes."""
    destination = Path(path).resolve()
    probe = destination if destination.exists() else destination.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = int(shutil.disk_usage(probe).free)
    required = max(0, int(archive_bytes)) + _DEFAULT_RESERVE_BYTES
    if free < required:
        raise InsufficientStorageError(destination, required, free)
    return free


@contextmanager
def atomic_path(destination: str | Path) -> Iterator[Path]:
    """Yield a temporary path that replaces ``destination`` when the block succeeds.

    The temporary file lies in the destination directory, so the final rename
    stays on one file system. On normal exit it is synchronized to disk and
    renamed over ``destination``. If the block raises, the temporary file is
    removed and ``destination`` is untouched.

    Parameters
    ----------
    destination : str or pathlib.Path
        Final file. Missing parent directories are created.

    Yields
    ------
    pathlib.Path
        An existing empty file with the destination's suffix to write into by
        path, for example with ``h5py`` or a VTK writer.
    """
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.stem}.", suffix=f".tmp{target.suffix}", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        yield temporary
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_text_atomic(path: str | Path, text: str) -> Path:
    """Atomically replace a UTF-8 text file.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination file. Missing parent directories are created.
    text : str
        Complete file content.

    Returns
    -------
    pathlib.Path
        The destination path.
    """
    destination = Path(path)
    with (
        atomic_path(destination) as temporary,
        temporary.open("w", encoding="utf-8", newline="") as stream,
    ):
        stream.write(text)
    return destination


def npz_size_bound(arrays: Mapping[str, object]) -> int:
    """Return a conservative size in bytes of the compressed archive of ``arrays``."""
    return sum(int(np.asarray(value).nbytes) + 4096 for value in arrays.values()) + (4 << 20)


def write_npz_atomic(
    path: str | Path,
    arrays: Mapping[str, object],
    *,
    check_free_space: bool = True,
) -> Path:
    """Atomically replace a compressed NumPy archive.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination ``.npz`` file. Missing parent directories are created.
    arrays : mapping of str to array_like
        Archive members.
    check_free_space : bool, default=True
        Raise :class:`InsufficientStorageError` before writing when the disk
        cannot hold the new archive next to the old one. A collective writer
        that checks the aggregate of all ranks once passes ``False``.

    Returns
    -------
    pathlib.Path
        The destination path.

    Raises
    ------
    InsufficientStorageError
        If the capacity check fails.
    """
    destination = Path(path)
    if check_free_space:
        require_free_space(destination, npz_size_bound(arrays))
    with atomic_path(destination) as temporary, temporary.open("wb") as stream:
        # NumPy accepts arbitrary member names here; its stub models one
        # reserved keyword and consequently rejects a typed mapping.
        np.savez_compressed(stream, **arrays)  # pyrefly: ignore[bad-argument-type]
    return destination


def write_json_atomic(path: str | Path, data: object, *, indent: int | None = 2) -> Path:
    """Atomically replace a JSON file with sorted keys and strict numbers.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination file. Missing parent directories are created.
    data : object
        JSON-compatible content. NumPy scalars must be converted by the
        caller; NaN and infinite floats are rejected.
    indent : int or None, default=2
        Indentation of nested members; ``None`` writes one line.

    Returns
    -------
    pathlib.Path
        The destination path.

    Raises
    ------
    TypeError
        If ``data`` contains a value JSON cannot represent.
    ValueError
        If ``data`` contains NaN or an infinite float.
    """
    text = json.dumps(data, indent=indent, sort_keys=True, allow_nan=False)
    return write_text_atomic(path, text + "\n")


def _append_text(path: Path, text: str) -> None:
    """Append text, truncating a partial record if the write fails."""
    with path.open("a", encoding="utf-8", newline="") as stream:
        start = stream.tell()
        try:
            stream.write(text)
            stream.flush()
        except OSError:
            stream.seek(start)
            stream.truncate()
            raise


def _ends_with_newline(path: Path) -> bool:
    """Return whether a non-empty file ends with a line terminator."""
    with path.open("rb") as stream:
        stream.seek(-1, os.SEEK_END)
        return stream.read(1) in (b"\n", b"\r")


def append_csv_rows(
    path: str | Path,
    columns: Sequence[str],
    rows: Iterable[Sequence[object]],
) -> Path:
    """Append rows to a CSV time series, writing the header for a new file.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination CSV file. Missing parent directories are created.
    columns : sequence of str
        Column names. They are written when the file is new or empty and
        must equal the existing header otherwise.
    rows : iterable of sequence
        Data rows, each with one value per column. Floats are written with
        Python's shortest round-trip representation.

    Returns
    -------
    pathlib.Path
        The destination path.

    Raises
    ------
    ValueError
        If the existing header differs from ``columns``, a row has the wrong
        length, or the file ends in a partial row (call :func:`rewind_csv`
        first after an interrupted run).
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    header = [str(name) for name in columns]
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    if destination.is_file() and destination.stat().st_size > 0:
        with destination.open(newline="", encoding="utf-8") as stream:
            existing = next(csv.reader(stream), [])
        if existing != header:
            raise ValueError(
                f"CSV header of {destination} is {existing}, expected {header}; "
                "write to a new file or remove the old series"
            )
        if not _ends_with_newline(destination):
            raise ValueError(f"CSV file {destination} ends in a partial row")
    else:
        writer.writerow(header)
    for row in rows:
        values = list(row)
        if len(values) != len(header):
            raise ValueError(
                f"CSV row for {destination} has {len(values)} values, expected {len(header)}"
            )
        writer.writerow(values)
    _append_text(destination, buffer.getvalue())
    return destination


def rewind_csv(path: str | Path, time: float) -> None:
    """Keep the CSV rows recorded up to and including ``time``.

    A partial last row left by an interrupted append is removed. The file is
    rewritten atomically only when a row is removed.

    Parameters
    ----------
    path : str or pathlib.Path
        CSV file with a ``time`` column. A missing or empty file is left as is.
    time : float
        Restored solution time in s.

    Raises
    ------
    ValueError
        If the file has no ``time`` column, an interior row has the wrong
        length, or a time value is not a finite number.
    """
    source = Path(path)
    if not source.is_file() or source.stat().st_size == 0:
        return
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    complete_tail = _ends_with_newline(source)
    if not rows:
        return
    header = rows[0]
    if "time" not in header:
        raise ValueError(f"CSV file {source} has no time column")
    time_column = header.index("time")
    kept = [header]
    for index, row in enumerate(rows[1:], start=1):
        last = index == len(rows) - 1
        if last and not complete_tail:
            break
        if len(row) != len(header):
            raise ValueError(f"CSV file {source} has an invalid row {index}")
        try:
            row_time = float(row[time_column])
        except ValueError as error:
            raise ValueError(f"CSV file {source} has an invalid time in row {index}") from error
        if not math.isfinite(row_time):
            raise ValueError(f"CSV file {source} has a non-finite time in row {index}")
        if row_time <= time + time_tolerance(time):
            kept.append(row)
    if len(kept) == len(rows):
        return
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\n").writerows(kept)
    write_text_atomic(source, buffer.getvalue())


def append_jsonl(path: str | Path, record: Mapping[str, object]) -> Path:
    """Append one JSON object as a line of a JSON-lines time series.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination file. Missing parent directories are created.
    record : mapping
        JSON-compatible record; it should contain a ``time`` member in s so
        :func:`rewind_jsonl` can trim it. NaN and infinite floats are rejected.

    Returns
    -------
    pathlib.Path
        The destination path.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(record), sort_keys=True, allow_nan=False) + "\n"
    _append_text(destination, line)
    return destination


def rewind_jsonl(path: str | Path, time: float) -> None:
    """Keep the JSON-lines records recorded up to and including ``time``.

    A partial last line left by an interrupted append is removed. The file is
    rewritten atomically only when a record is removed.

    Parameters
    ----------
    path : str or pathlib.Path
        JSON-lines file whose records each have a ``time`` member in s.
        A missing file is left as is.
    time : float
        Restored solution time in s.

    Raises
    ------
    ValueError
        If an interior line is not a JSON object with a finite ``time``.
    """
    source = Path(path)
    if not source.is_file():
        return
    lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
    kept = []
    for index, line in enumerate(lines):
        if index == len(lines) - 1 and not line.endswith(("\n", "\r")):
            break
        try:
            record = json.loads(line)
            row_time = float(record["time"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f"JSON-lines file {source} has an invalid line {index + 1}") from error
        if not math.isfinite(row_time):
            raise ValueError(f"JSON-lines file {source} has a non-finite time in line {index + 1}")
        if row_time <= time + time_tolerance(time):
            kept.append(line)
    if len(kept) != len(lines):
        write_text_atomic(source, "".join(kept))
