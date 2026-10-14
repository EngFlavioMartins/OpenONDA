"""Durable text, JSON, CSV and JSON-lines output shared by the solvers.

Whole files are replaced atomically: the content is written to a temporary
file in the destination directory, synchronized to disk and renamed over the
destination, so a reader never opens a partially written file. Time series
are appended one record at a time; a record that fails to write is truncated
so the file keeps a valid prefix. After a restart, ``rewind_*`` keeps the
records up to and including the restored time.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import csv
import io
import json
import math
import os
from pathlib import Path
import tempfile

__all__ = [
    "TIME_TOLERANCE",
    "append_csv_rows",
    "append_jsonl",
    "rewind_csv",
    "rewind_jsonl",
    "write_json_atomic",
    "write_text_atomic",
]

TIME_TOLERANCE = 1.0e-12
"""Absolute tolerance in s when a recorded time is compared with a restart time."""


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
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
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
        if row_time <= time + TIME_TOLERANCE:
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
        if row_time <= time + TIME_TOLERANCE:
            kept.append(line)
    if len(kept) != len(lines):
        write_text_atomic(source, "".join(kept))
