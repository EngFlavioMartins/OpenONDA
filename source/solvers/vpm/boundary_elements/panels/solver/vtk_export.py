"""
VTK export for panel method — writes panel mesh with solution fields to .vtp.

Mirrors ``vlm/solver/lattice.py::save_vtk()`` in output structure:
  - Unstructured grid of triangle polygons
  - Per-cell data: doublet strength, pressure coefficient, panel force,
    LESP, TE/LE flags, group ID, area, normal

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: June 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from __future__ import annotations

from pathlib import Path
from xml.dom import minidom
import xml.etree.ElementTree as ET

import numpy as np


def panel_mesh_to_vtp(
    vertex_position: np.ndarray,
    panel_centre: np.ndarray,
    normal: np.ndarray,
    doublet_strength: np.ndarray,
    area: np.ndarray,
    pressure_coefficient: np.ndarray,
    panel_force: np.ndarray,
    group_id: np.ndarray,
    time: float,
    filepath: str | Path,
) -> str:
    n = len(vertex_position)
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    # VTK XML header
    root = ET.Element("VTKFile", type="PolyData", version="1.0")
    poly = ET.SubElement(root, "PolyData")
    poly.set("NumberOfPoints", str(n * 3))
    poly.set("NumberOfVerts", "0")
    poly.set("NumberOfLines", "0")
    poly.set("NumberOfStrips", "0")
    poly.set("NumberOfPolys", str(n))

    # Points section: each triangle contributes 3 vertex_position (unstructured)
    points_elem = ET.SubElement(poly, "Points")
    da_points = ET.SubElement(
        points_elem, "DataArray", type="Float64", NumberOfComponents="3", format="ascii"
    )
    pts_str = []
    for i in range(n):
        for j in range(3):
            pts_str.append(
                f"{vertex_position[i, j, 0]:.10e} {vertex_position[i, j, 1]:.10e} "
                f"{vertex_position[i, j, 2]:.10e}"
            )
    da_points.text = " ".join(pts_str)

    # Polys section: connectivity (3 vertex_position per triangle)
    polys_elem = ET.SubElement(poly, "Polys")
    da_conn = ET.SubElement(
        polys_elem, "DataArray", type="Int32", Name="connectivity", format="ascii"
    )
    conn = []
    da_offset = ET.SubElement(polys_elem, "DataArray", type="Int32", Name="offsets", format="ascii")
    offsets = []
    for i in range(n):
        base = i * 3
        conn.extend([base, base + 1, base + 2])
        offsets.append((i + 1) * 3)
    da_conn.text = " ".join(str(v) for v in conn)
    da_offset.text = " ".join(str(v) for v in offsets)

    # Cell data
    celldata = ET.SubElement(poly, "CellData")
    arrays = [
        ("doublet_strength", doublet_strength, "Float64", 1),
        ("pressure_coefficient", pressure_coefficient, "Float64", 1),
        ("area", area, "Float64", 1),
        ("group_id", group_id, "Int32", 1),
        ("normal", normal, "Float64", 3),
        ("panel_force", panel_force, "Float64", 3),
        ("panel_centre", panel_centre, "Float64", 3),
    ]
    for name, data, dtype, nc in arrays:
        da = ET.SubElement(
            celldata, "DataArray", type=dtype, Name=name, NumberOfComponents=str(nc), format="ascii"
        )
        if nc == 1:
            da.text = " ".join(f"{v:.10e}" for v in data[:n])
        else:
            da.text = " ".join(" ".join(f"{data[i][j]:.10e}" for j in range(nc)) for i in range(n))

    # Time
    field = ET.SubElement(celldata, "FieldData")
    ft = ET.SubElement(
        field, "DataArray", type="Float64", Name="time", NumberOfComponents="1", format="ascii"
    )
    ft.text = str(time)

    # Write pretty-printed XML
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")  # nosec B318
    with open(path, "w") as f:
        f.write(xml_str)

    return str(path)
