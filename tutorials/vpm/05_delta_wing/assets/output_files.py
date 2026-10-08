"""Read and write the solver output files used by this tutorial's figures."""

import json
from pathlib import Path

from defusedxml import ElementTree
import numpy as np


def read_json(path):
    """Read one JSON object."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def read_pvd_frames(path):
    """Return the ``(time, frame path)`` pairs of a VTK collection in time order."""
    path = Path(path)
    return sorted(
        (float(item.attrib["timestep"]), path.parent / item.attrib["file"])
        for item in ElementTree.parse(path).iter("DataSet")
    )
