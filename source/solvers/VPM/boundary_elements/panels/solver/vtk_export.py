"""
VTK export for panel method — writes panel mesh with solution fields to .vtp.

Mirrors ``vlm/solver/lattice.py::save_vtk()`` in output structure:
  - Unstructured grid of triangle polygons
  - Per-cell data: circulation (strength), pressure coefficient, panel forces,
    LESP, TE/LE flags, group ID, area, normals

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
    vertices: np.ndarray,
    centers: np.ndarray,
    normals: np.ndarray,
    strengths: np.ndarray,
    areas: np.ndarray,
    cp: np.ndarray,
    panel_forces: np.ndarray,
    lesp: np.ndarray,
    is_te: np.ndarray,
    is_le: np.ndarray,
    group_ids: np.ndarray,
    flow_time: float,
    filepath: str | Path,
) -> str:
    n = len(vertices)
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

    # Points section: each triangle contributes 3 vertices (unstructured)
    points_elem = ET.SubElement(poly, "Points")
    da_points = ET.SubElement(
        points_elem, "DataArray", type="Float64", NumberOfComponents="3", format="ascii"
    )
    pts_str = []
    for i in range(n):
        for j in range(3):
            pts_str.append(
                f"{vertices[i, j, 0]:.10e} {vertices[i, j, 1]:.10e} {vertices[i, j, 2]:.10e}"
            )
    da_points.text = " ".join(pts_str)

    # Polys section: connectivity (3 vertices per triangle)
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
        ("strength", strengths, "Float64", 1),
        ("Cp", cp, "Float64", 1),
        ("area", areas, "Float64", 1),
        ("LESP", lesp, "Float64", 1),
        ("is_TE", is_te, "Int32", 1),
        ("is_LE", is_le, "Int32", 1),
        ("group_id", group_ids, "Int32", 1),
        ("normals", normals, "Float64", 3),
        ("force", panel_forces, "Float64", 3),
        ("center", centers, "Float64", 3),
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
        field, "DataArray", type="Float64", Name="TimeValue", NumberOfComponents="1", format="ascii"
    )
    ft.text = str(flow_time)

    # Write pretty-printed XML
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")  # nosec B318
    with open(path, "w") as f:
        f.write(xml_str)

    return str(path)
