"""Read and write the solver output files used by this tutorial's figures."""

import csv
from pathlib import Path

import numpy as np


def read_csv_columns(path):
    """Read a numeric CSV table as a mapping from column name to values."""
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return {name: np.asarray([float(row[name]) for row in rows]) for name in rows[0]}


def write_csv_table(path, rows, *, columns):
    """Write a named CSV table, creating its directory."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)
