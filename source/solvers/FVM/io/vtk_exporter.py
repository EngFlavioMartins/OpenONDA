from html import escape, unescape
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from ..config.types import OutputSetup

try:
    import pyvista as pv
    import vtk
except ImportError:  # Optional visualization dependencies.
    pv = None
    vtk = None
else:
    # Suppress non-fatal VTK-m warnings (e.g., unsupported cell types in Viskores)
    vtk.vtkObject.GlobalWarningDisplayOff()

_pyvista: Any = pv
_vtk: Any = vtk


def atomic_write_text(path: str | Path, content: str) -> None:
    """Atomically publish a UTF-8 metadata file after durable temporary output."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


class VTKExporter:
    """
    Exporters results to VTU format using PyVista.
    Supports cell-centered data (Finite Volume).
    Mappings:
        - mesh_data['points'] -> VTK Points
        - mesh_data elements -> VTK Cells (Polyhedron)
    """

    def __init__(
        self,
        mesh_data: dict[str, Any],
        output: OutputSetup | None = None,
    ):
        """Initialise the VTK exporter.

        Builds a :class:`pyvista.UnstructuredGrid` from the mesh data
        on construction.

        Args:
            mesh_data: Mesh dictionary (needs ``points``, ``faces``,
                       ``owners``, ``neighbours``, ``n_elements``,
                       ``n_interior_faces``).
        """
        if _pyvista is None or _vtk is None:
            raise ImportError(
                "VTK export requires the optional FVM dependencies: pip install 'OpenONDA[fvm]'"
            )
        self.mesh_data = mesh_data
        self.output = output or OutputSetup()
        self._grid = self._initialize_grid()

    def _field_array(self, data: np.ndarray) -> np.ndarray:
        """Return a contiguous array matching the qualified output precision."""
        values = np.asarray(data)
        if np.issubdtype(values.dtype, np.floating):
            values = values.astype(np.float64, copy=False)
        return np.ascontiguousarray(values)

    def _write_grid(self, filename: str, grid) -> None:
        """Write one valid appended-binary VTU and publish it atomically."""
        target = Path(filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{target.stem}.",
            suffix=".tmp.vtu",
            dir=target.parent,
        )
        os.close(descriptor)
        try:
            writer = _vtk.vtkXMLUnstructuredGridWriter()
            writer.SetFileName(temporary)
            writer.SetInputData(grid)
            writer.SetDataModeToAppended()
            writer.EncodeAppendedDataOn()
            writer.SetHeaderTypeToUInt64()
            if self.output.compression == "lz4":
                writer.SetCompressorTypeToLZ4()
            elif self.output.compression == "zlib":
                writer.SetCompressorTypeToZLib()
            else:
                writer.SetCompressorTypeToNone()
            if writer.Write() != 1:
                raise OSError(f"VTK failed to write {target}")
            with open(temporary, "rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        except BaseException:
            if os.path.exists(temporary):
                os.unlink(temporary)
            raise

    def _initialize_grid(self) -> Any:
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
            return _pyvista.UnstructuredGrid(
                cells.ravel(),
                np.full(len(vertices), _pyvista.CellType.HEXAHEDRON, dtype=np.uint8),
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
            cell_types.append(_pyvista.CellType.POLYHEDRON)

        grid = _pyvista.UnstructuredGrid(cells, cell_types, points)
        return grid

    def export(
        self,
        filename: str,
        fields: dict[str, np.ndarray],
        interpolate_to_points: bool = False,
        *,
        point_fields: dict[str, np.ndarray] | None = None,
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

        self._grid.cell_data.clear()
        for name, data in fields.items():
            values = self._field_array(data)
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

        self._grid.point_data.clear()
        for name, data in (point_fields or {}).items():
            values = self._field_array(data)
            if values.shape[0] != self._grid.n_points:
                raise ValueError(
                    f"VTK point field {name!r} has {values.shape[0]} rows; "
                    f"expected {self._grid.n_points} points"
                )
            self._grid.point_data[name] = values

        if interpolate_to_points:
            # This allows ParaView to offer Point-based filters and smooth gradients
            point_grid = self._grid.cell_data_to_point_data()
            self._write_grid(filename, point_grid)
        else:
            self._write_grid(filename, self._grid)

        return filename

    def export_cells(self, filename: str, cell_ids, fields: dict[str, np.ndarray]):
        """Write an explicitly selected cell partition without interpolation."""
        ids = np.asarray(cell_ids, dtype=np.int64)
        if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= self.mesh_data["n_elements"]):
            raise ValueError("cell_ids must be valid one-dimensional global cell indices")
        self._grid.cell_data.clear()
        self._grid.point_data.clear()
        grid = self._grid.extract_cells(ids)
        for name, data in fields.items():
            values = self._field_array(data)
            if values.shape[0] != len(ids):
                raise ValueError(
                    f"Partition field {name!r} has {values.shape[0]} rows; expected {len(ids)}"
                )
            grid.cell_data[name] = values
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._write_grid(filename, grid)
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
                self.entries.append((float(time), unescape(fpath)))

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
        lines = [
            '<?xml version="1.0"?>',
            '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
            "  <Collection>",
        ]
        lines.extend(
            f'    <DataSet timestep="{time}" group="" part="0" file="{escape(fpath, quote=True)}"/>'
            for time, fpath in self.entries
        )
        lines.extend(["  </Collection>", "</VTKFile>"])
        atomic_write_text(self.filename, "\n".join(lines) + "\n")


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
