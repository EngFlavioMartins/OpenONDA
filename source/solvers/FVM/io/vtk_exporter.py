import os
from typing import Any

import numpy as np

try:
    import pyvista as pv
    import vtk
except ImportError:  # Optional visualization dependencies.
    pv = None
    vtk = None
else:
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
        """Initialise the VTK exporter.

        Builds a :class:`pyvista.UnstructuredGrid` from the mesh data
        on construction.

        Args:
            mesh_data: Mesh dictionary (needs ``points``, ``faces``,
                       ``owners``, ``neighbours``, ``n_elements``,
                       ``n_interior_faces``).
        """
        if pv is None or vtk is None:
            raise ImportError(
                "VTK export requires the optional FVM dependencies: pip install 'OpenONDA[fvm]'"
            )
        self.mesh_data = mesh_data
        self._grid = self._initialize_grid()

    def _initialize_grid(self) -> pv.UnstructuredGrid:
        """Construct a :class:`pyvista.UnstructuredGrid` from mesh data.

        Uses native VTK hexahedra whenever explicit cell vertices are
        available.  Other meshes fall back to general polyhedra.

        Returns:
            A fully constructed :class:`pyvista.UnstructuredGrid`.
        """
        points = self.mesh_data["points"]
        cell_vertices = self.mesh_data.get("cell_vertices")
        cell_types = self.mesh_data.get("cell_type_codes")
        if (
            cell_vertices is not None
            and cell_types is not None
            and np.asarray(cell_vertices).shape == (self.mesh_data["n_elements"], 8)
            and np.all(np.asarray(cell_types) == 5)  # Gmsh's 8-node hex code
        ):
            vertices = np.asarray(cell_vertices, dtype=np.int64)
            cells = np.empty((len(vertices), 9), dtype=np.int64)
            cells[:, 0] = 8
            cells[:, 1:] = vertices
            return pv.UnstructuredGrid(
                cells.ravel(),
                np.full(len(vertices), pv.CellType.HEXAHEDRON, dtype=np.uint8),
                points,
            )

        faces = self.mesh_data["faces"]
        owners = self.mesh_data["owners"]
        neighbours = self.mesh_data["neighbours"]
        n_cells = self.mesh_data["n_elements"]

        # Group faces by cell
        cell_faces = self.mesh_data.get("cell_faces")
        cell_face_offsets = self.mesh_data.get("cell_face_offsets")
        if cell_faces is None or cell_face_offsets is None:
            from ..mesh.topology import build_cell_face_csr

            cell_faces, cell_face_offsets = build_cell_face_csr(
                owners, neighbours, n_cells, self.mesh_data["n_faces"]
            )
            self.mesh_data["cell_faces"] = cell_faces
            self.mesh_data["cell_face_offsets"] = cell_face_offsets

        # VTK Polyhedron format:
        # [num_faces, num_nodes_f1, n1, n2, ..., num_nodes_f2, n1, n2, ...]
        cells = []
        cell_types = []
        for c_idx in range(n_cells):
            f_indices = cell_faces[cell_face_offsets[c_idx] : cell_face_offsets[c_idx + 1]]
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
        self, filename: str, fields: dict[str, np.ndarray], interpolate_to_points: bool = False
    ):
        """Export fields to a ``.vtu`` file (VTK unstructured grid format).

        Fields are stored as cell data. Arrays must contain exactly the cells,
        or the cells followed by one ghost value per physical boundary face.

        Args:
            filename:              Output ``.vtu`` file path.
            fields:                Dict mapping field name → array.
            interpolate_to_points: If ``True``, converts cell
                                   data to point data via
                                   :meth:`pyvista.DataSetFilters.cell_data_to_point_data`
                                   for smoother visualisation in ParaView.  The
                                   default keeps raw finite-volume cell data;
                                   cubeFlow's comparison loaders explicitly
                                   support this representation.

        Returns:
            The output file path.
        """
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)

        # Update cell data
        for name, data in fields.items():
            values = np.asarray(data)
            n_cells = self.mesh_data["n_elements"]
            n_with_boundary = (
                n_cells + self.mesh_data["n_faces"] - self.mesh_data["n_interior_faces"]
            )
            if values.shape[0] == n_with_boundary:
                values = values[:n_cells]
            elif values.shape[0] != n_cells:
                raise ValueError(
                    f"VTK field {name!r} has {values.shape[0]} rows; expected "
                    f"{n_cells} cells or {n_with_boundary} cells plus boundary ghosts"
                )
            self._grid.cell_data[name] = values

        if interpolate_to_points:
            # This allows ParaView to offer Point-based filters and smooth gradients
            point_grid = self._grid.cell_data_to_point_data()
            point_grid.save(filename)
        else:
            self._grid.save(filename)

        return filename

    def export_cells(self, filename: str, cell_ids, fields: dict[str, np.ndarray]):
        """Write an explicitly selected cell partition without interpolation."""
        ids = np.asarray(cell_ids, dtype=np.int64)
        if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= self.mesh_data["n_elements"]):
            raise ValueError("cell_ids must be valid one-dimensional global cell indices")
        grid = self._grid.extract_cells(ids)
        for name, data in fields.items():
            values = np.asarray(data)
            if values.shape[0] != len(ids):
                raise ValueError(
                    f"Partition field {name!r} has {values.shape[0]} rows; expected {len(ids)}"
                )
            grid.cell_data[name] = values
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
        grid.save(filename)
        return filename


class PVDManager:
    """Manages ParaView Data (``.pvd``) files for time-series animation.

    A collection file that references individual ``.vtu`` snapshots at
    their respective time values, enabling time-dependent visualisation
    in ParaView.

    Args:
        filename: Output ``.pvd`` file path.
    """

    def __init__(self, filename: str):
        self.filename = filename
        self.entries: list[tuple[float, str]] = []
        if os.path.exists(filename):
            self._parse_existing()

    def _parse_existing(self):
        """Parse an existing ``.pvd`` file to resume appending.

        Uses a simple regex to extract ``timestep`` and ``file``
        attributes from each ``<DataSet>`` element.
        """
        import re

        with open(self.filename) as f:
            content = f.read()
            matches = re.findall(
                r'<DataSet timestep="(.+?)" group="" part="0" file="(.+?)"/>', content
            )
            for time, fpath in matches:
                self.entries.append((float(time), fpath))

    def add_step(self, time: float, vtu_file: str):
        """Register a time step and re-write the ``.pvd`` file.

        The *vtu_file* path is stored as relative to the ``.pvd`` file
        location for portability.

        Args:
            time:     Simulation time for this snapshot.
            vtu_file: Path to the ``.vtu`` file.
        """
        rel_path = os.path.relpath(vtu_file, os.path.dirname(self.filename))
        self.entries.append((time, rel_path))
        self.write()

    def write(self):
        """Write the ``.pvd`` collection file."""
        with open(self.filename, "w") as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
            f.write("  <Collection>\n")
            for time, fpath in self.entries:
                f.write(f'    <DataSet timestep="{time}" group="" part="0" file="{fpath}"/>\n')
            f.write("  </Collection>\n")
            f.write("</VTKFile>\n")


def write_vtu(filename: str, mesh_data: dict[str, Any], fields: dict[str, np.ndarray]):
    """Convenience function for one-off VTU export.

    Creates a :class:`VTKExporter`, attaches the given fields, and
    writes a single ``.vtu`` file.

    Args:
        filename:  Output ``.vtu`` file path.
        mesh_data: Mesh dictionary.
        fields:    Dict mapping field name → array (cell-centred).
    """
    exporter = VTKExporter(mesh_data)
    exporter.export(filename, fields)
