"""Read and write the solver output files used by this tutorial's figures."""

import csv
import json
from pathlib import Path

import numpy as np


def read_json(path):
    """Read one JSON object."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    """Write one JSON object."""
    write_text(path, json.dumps(data, indent=2) + "\n")


def read_csv_table(path):
    """Read named CSV columns; a leading ``# time = t`` line supplies the time."""
    with Path(path).open(encoding="utf-8") as stream:
        first_line = stream.readline()
    clock = first_line.startswith("#")
    data = np.atleast_1d(
        np.genfromtxt(
            path, delimiter=",", names=True, dtype=None, encoding="utf-8", skip_header=int(clock)
        )
    )
    table = {name: data[name] for name in data.dtype.names}
    if clock and "time" not in table:
        table["time"] = np.full(data.size, float(first_line.split("=")[1]))
    return table


def read_history_table(path):
    """Read an accepted history with one row per physical time."""
    return read_csv_table(path)


def history_window(table, start, end, *, columns):
    """Interpolate a history onto ``[start, end]``, keeping the interior samples."""
    time = np.asarray(table["time"], dtype=float)
    interior = (time > start) & (time < end)
    result = {"time": np.r_[start, time[interior], end]}
    for name in columns:
        result[name] = np.interp(result["time"], time, np.asarray(table[name], dtype=float))
    return result


def write_text(path, content, encoding="utf-8"):
    """Write a text file, creating its directory."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(content, encoding=encoding)


def write_csv_table(path, rows, *, columns):
    """Write a named CSV table, creating its directory."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)
