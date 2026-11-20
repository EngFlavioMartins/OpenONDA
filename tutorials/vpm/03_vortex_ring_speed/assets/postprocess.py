"""Table reading and figure export for the ring-speed figure."""

from __future__ import annotations

from pathlib import Path

import numpy as np


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


def write_text(path, content, encoding="utf-8"):
    """Write a text file, creating its directory."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(content, encoding=encoding)


def theme():
    """Return the repository plotting module without duplicating its settings."""
    from openonda import plotting

    return plotting


def figure_size(height_cm):
    """Return a thesis-width figure size in inches for ``height_cm``."""
    plotting = theme()
    return (plotting.MAX_FIGURE_WIDTH_CM * plotting.CM, height_cm * plotting.CM)


def save_figure(fig, path, axes, formats=("png",), *, fit_margins=True):
    """Validate and save one figure using its script-matching base name."""
    plotting = theme()
    for figure_format in formats:
        plotting.export_figure(
            fig,
            Path(path).with_suffix(f".{figure_format}"),
            figure_format=figure_format,
            close=False,
        )
