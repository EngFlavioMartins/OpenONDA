import os
from typing import Any

import numpy as np
import pyvista as pv
import vtk

# Suppress non-fatal VTK-m warnings (e.g., unsupported cell types in Viskores)
vtk.vtkObject.GlobalWarningDisplayOff()


class VTKExporter:
    """
    Exporters results to VTU format using PyVista.
    Supports cell-centered data (Finite Volume).
    Mappings:
        - mesh_data['points'] -> VTK Points
        - mesh_data elements -> VTK Cells (Polyhedron)
    """

    def __init__(self, mesh_data: dict[str, Any]):
        self.mesh_data = mesh_data
        self._grid = self._initialize_grid()

    def _initialize_grid(self) -> pv.UnstructuredGrid:
        """
        Constructs a PyVista UnstructuredGrid from mesh_data.
        Handles arbitrary polyhedra.
        """
        points = self.mesh_data["points"]
        faces = self.mesh_data["faces"]
        owners = self.mesh_data["owners"]
        neighbours = self.mesh_data["neighbours"]
        n_cells = self.mesh_data["n_elements"]
        n_internal = self.mesh_data["n_interior_faces"]

        # Group faces by cell
        cell_faces = [[] for _ in range(n_cells)]
        for f_idx in range(len(faces)):
            own = owners[f_idx]
            cell_faces[own].append(f_idx)
            if f_idx < n_internal:
                nei = neighbours[f_idx]
                cell_faces[nei].append(f_idx)

        # VTK Polyhedron format:
        # [num_faces, num_nodes_f1, n1, n2, ..., num_nodes_f2, n1, n2, ...]
        cells = []
        cell_types = []
        for c_idx in range(n_cells):
            f_indices = cell_faces[c_idx]
            cell_data = [len(f_indices)]
            for f_idx in f_indices:
                f_nodes = faces[f_idx]
                cell_data.append(len(f_nodes))
                cell_data.extend(f_nodes)

            cells.append(len(cell_data))
            cells.extend(cell_data)
            cell_types.append(pv.CellType.POLYHEDRON)

        grid = pv.UnstructuredGrid(cells, cell_types, points)
        return grid

    def export(
        self, filename: str, fields: dict[str, np.ndarray], interpolate_to_points: bool = True
    ):
        """
        Exports fields to a .vtu file.
        Fields should be cell-centered arrays of size n_cells.
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Update cell data
        for name, data in fields.items():
            # OpenONDA fields often include ghost cells for boundaries.
            # We take only the first n_elements (internal cells).
            if data.shape[0] > self.mesh_data["n_elements"]:
                self._grid.cell_data[name] = data[: self.mesh_data["n_elements"]]
            else:
                self._grid.cell_data[name] = data

        if interpolate_to_points:
            # This allows ParaView to offer Point-based filters and smooth gradients
            point_grid = self._grid.cell_data_to_point_data()
            point_grid.save(filename)
        else:
            self._grid.save(filename)

        return filename


class PVDManager:
    """
    Manages ParaView Data (.pvd) files for time-series visualization.
    """

    def __init__(self, filename: str):
        self.filename = filename
        self.entries = []
        if os.path.exists(filename):
            self._parse_existing()

    def _parse_existing(self):
        # Very simple parser if we need to resume
        import re

        with open(self.filename) as f:
            content = f.read()
            matches = re.findall(
                r'<DataSet timestep="(.+?)" group="" part="0" file="(.+?)"/>', content
            )
            for time, fpath in matches:
                self.entries.append((float(time), fpath))

    def add_step(self, time: float, vtu_file: str):
        # Store relative path for portability
        rel_path = os.path.relpath(vtu_file, os.path.dirname(self.filename))
        self.entries.append((time, rel_path))
        self.write()

    def write(self):
        with open(self.filename, "w") as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
            f.write("  <Collection>\n")
            for time, fpath in self.entries:
                f.write(f'    <DataSet timestep="{time}" group="" part="0" file="{fpath}"/>\n')
            f.write("  </Collection>\n")
            f.write("</VTKFile>\n")


def write_vtu(filename: str, mesh_data: dict[str, Any], fields: dict[str, np.ndarray]):
    """Convenience function for one-off export."""
    exporter = VTKExporter(mesh_data)
    exporter.export(filename, fields)
