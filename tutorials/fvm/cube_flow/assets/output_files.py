"""Read and write the solver output files used by this tutorial's figures."""

import csv
from pathlib import Path

from defusedxml import ElementTree
import numpy as np


def read_grouped_csv(path, group_columns):
    """Read numeric history columns grouped by their recorded entity."""
    columns = (group_columns,) if isinstance(group_columns, str) else tuple(group_columns)
    groups = {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            values = tuple(row[name] for name in columns)
            group = groups.setdefault(values[0] if len(values) == 1 else values, {})
            for name, value in row.items():
                if name not in columns:
                    group.setdefault(name, []).append(float(value))
    return {
        key: {name: np.asarray(values) for name, values in group.items()}
        for key, group in groups.items()
    }


def read_pvd_frames(path):
    """Return the ``(time, frame path)`` pairs of a VTK collection in time order."""
    path = Path(path)
    return sorted(
        (float(item.attrib["timestep"]), path.parent / item.attrib["file"])
        for item in ElementTree.parse(path).iter("DataSet")
    )


def latest_fvm_frame(solution_directory):
    """Return the latest saved FVM field frame."""
    return read_pvd_frames(Path(solution_directory) / "fvm.pvd")[-1][1]


def snapshot_vector_field(mesh, name):
    """Return a saved field with the coordinates at its cell or point association."""
    if name in mesh.cell_data:
        return mesh.cell_data[name], mesh.cell_centers().points
    return mesh.point_data[name], mesh.points
