"""Read and write the solver output files used by this tutorial's figures."""

import csv
from pathlib import Path

from defusedxml import ElementTree
import numpy as np


def read_csv_columns(path):
    """Read a numeric CSV table as a mapping from column name to values."""
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return {name: np.asarray([float(row[name]) for row in rows]) for name in rows[0]}


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


def section_polygons(mesh, *, normal, origin, field):
    """Cut cells by a plane; return each polygon with its cell value."""
    if field not in mesh.cell_data:
        mesh = mesh.point_data_to_cell_data()
    section = mesh.slice(normal=normal, origin=origin)
    polygons = []
    cursor = 0
    while cursor < len(section.faces):
        count = int(section.faces[cursor])
        polygons.append(section.points[section.faces[cursor + 1 : cursor + 1 + count], :2])
        cursor += count + 1
    return polygons, section.cell_data[field]
