"""Read and write the solver output files used by this tutorial's figures."""

import csv
import json
from pathlib import Path

from defusedxml import ElementTree
import numpy as np
import pyvista as pv


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


def read_history_table(path):
    """Read an accepted history with one row per physical time."""
    return read_csv_table(path)


def read_numeric_table(path, *, delimiter=","):
    """Read a headerless numeric table."""
    return np.loadtxt(path, delimiter=delimiter)


def read_pvd_frames(path):
    """Return the ``(time, frame path)`` pairs of a VTK collection in time order."""
    path = Path(path)
    return sorted(
        (float(item.attrib["timestep"]), path.parent / item.attrib["file"])
        for item in ElementTree.parse(path).iter("DataSet")
    )


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


def read_surface_frame(path):
    """Read a structured surface-sampler frame as two-dimensional arrays."""
    grid = pv.read(path)
    ni, nj, _ = grid.dimensions
    shape = (nj, ni)
    points = np.asarray(grid.points, dtype=float)
    valid = np.asarray(grid.point_data.get("vtkValidPointMask", np.ones(grid.n_points)), bool)

    def point_field(name, components):
        values = np.asarray(grid.point_data[name], dtype=float)
        return values.reshape(*shape, components) if components > 1 else values.reshape(shape)

    velocity = point_field("velocity", 3)
    data = {
        "x": points[:, 0].reshape(shape),
        "y": points[:, 1].reshape(shape),
        "z": points[:, 2].reshape(shape),
        "velocity": velocity,
        "valid": valid.reshape(shape),
        **{f"velocity_{axis}": velocity[..., index] for index, axis in enumerate("xyz")},
    }
    for name in ("vorticity", "velocity_standard_error", "vorticity_standard_error"):
        if name in grid.point_data:
            vector = point_field(name, 3)
            data[name] = vector
            data.update({f"{name}_{axis}": vector[..., index] for index, axis in enumerate("xyz")})
    for name in ("velocity_gradient_yx", "velocity_gradient_yx_standard_error"):
        if name in grid.point_data:
            data[name] = point_field(name, 1)
    for name in ("ensemble_size", "confidence_multiplier"):
        if name in grid.field_data:
            data[name] = np.asarray(grid.field_data[name])[0].item()
    return data
