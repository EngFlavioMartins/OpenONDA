"""
Vtk Export module for VPM solver.
==================
Vtk Export module for VPM solver. module.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import numpy as np
import vtk
from vtk.util import numpy_support


def export_panels_vtk(solver, filename: str, compression: bool = True):
    """
    Export panel geometry with properties to VTK PolyData format.

    Args:
        solver: VPM Solver instance with panel_solver
        filename: Output filename for VTK file
        compression: Whether to compress the VTK file (default: True)
    """
    if solver.panel_solver is None:
        return

    lattice = getattr(solver.panel_solver, "lattice", None)
    if lattice is None:
        return

    if lattice.num_panels == 0:
        return

    # Get panel data (slice to active panels)
    n_panels = lattice.num_panels
    vertices = lattice.vertices.to_numpy()[:n_panels]
    lattice.centers.to_numpy()[:n_panels]
    normals = lattice.normals.to_numpy()[:n_panels]
    areas = lattice.areas.to_numpy()[:n_panels]
    strengths = lattice.strengths.to_numpy()[:n_panels]

    # Flatten vertices: (N*3, 3)
    flat_vertices = vertices.reshape(-1, 3)

    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(flat_vertices, deep=True))

    # Create VTK cells (triangles)
    # Format: [npts, id0, id1, id2, npts, id0, id1, id2, ...]
    connectivity = np.arange(n_panels * 3, dtype=np.int64).reshape(n_panels, 3)
    cells = np.column_stack([np.full(n_panels, 3, dtype=np.int64), connectivity])
    cells_flat = cells.flatten()

    # Use numpy_to_vtk with explicit ID type
    cell_array = numpy_support.numpy_to_vtk(cells_flat, deep=True, array_type=vtk.VTK_ID_TYPE)

    polys = vtk.vtkCellArray()
    polys.SetCells(n_panels, cell_array)

    # Create polydata
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(polys)

    # Add cell data
    polydata.GetCellData().AddArray(numpy_support.numpy_to_vtk(normals, deep=True))
    polydata.GetCellData().GetArray(0).SetName("Normal")

    polydata.GetCellData().AddArray(numpy_support.numpy_to_vtk(areas, deep=True))
    polydata.GetCellData().GetArray(1).SetName("Area")

    polydata.GetCellData().AddArray(numpy_support.numpy_to_vtk(strengths, deep=True))
    polydata.GetCellData().GetArray(2).SetName("DoubletStrength")

    # Add Cp
    Cp = lattice.Cp.to_numpy()[:n_panels]
    polydata.GetCellData().AddArray(numpy_support.numpy_to_vtk(Cp, deep=True))
    polydata.GetCellData().GetArray(3).SetName("Cp")

    # Add time stamp
    time_array = numpy_support.numpy_to_vtk(np.array([solver.flow_time]), deep=True)
    time_array.SetName("TimeValue")
    polydata.GetFieldData().AddArray(time_array)

    _write_polydata_file(polydata, filename, compression)


def _write_polydata_file(polydata, filename: str, compression: bool = True):
    """
    Write VTK PolyData to file.

    Args:
        polydata: VTK PolyData object
        filename: Output filename
        compression: Whether to compress the file (default: True)
    """
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    if compression:
        writer.SetCompressorTypeToZLib()
    writer.Write()
