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
        self._point_operators: dict[bool, dict[str, np.ndarray]] = {}

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

    def _cell_vertex_incidence(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(cell_index, point_index)`` pairs for every cell corner."""
        cell_vertices = self.mesh_data.get("cell_vertices")
        n_cells = int(self.mesh_data["n_elements"])
        if cell_vertices is not None:
            vertices = np.asarray(cell_vertices, dtype=np.int64)
            if vertices.ndim == 2 and vertices.shape[0] == n_cells:
                cells = np.repeat(np.arange(n_cells, dtype=np.int64), vertices.shape[1])
                return cells, vertices.ravel()

        faces = self.mesh_data["faces"]
        cell_faces = self.mesh_data["cell_faces"]
        cell_face_offsets = self.mesh_data["cell_face_offsets"]
        cells_out: list[np.ndarray] = []
        points_out: list[np.ndarray] = []
        for cell in range(n_cells):
            face_ids = cell_faces[cell_face_offsets[cell] : cell_face_offsets[cell + 1]]
            corners = np.unique(np.concatenate([np.asarray(faces[f]) for f in face_ids]))
            cells_out.append(np.full(len(corners), cell, dtype=np.int64))
            points_out.append(corners.astype(np.int64))
        return np.concatenate(cells_out), np.concatenate(points_out)

    def _boundary_face_incidence(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(boundary_face_index, point_index)`` pairs for patch faces."""
        faces = self.mesh_data["faces"]
        n_interior = int(self.mesh_data["n_interior_faces"])
        n_faces = int(self.mesh_data["n_faces"])
        faces_out: list[np.ndarray] = []
        points_out: list[np.ndarray] = []
        for local, face in enumerate(range(n_interior, n_faces)):
            nodes = np.asarray(faces[face], dtype=np.int64)
            faces_out.append(np.full(len(nodes), local, dtype=np.int64))
            points_out.append(nodes)
        if not faces_out:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty
        return np.concatenate(faces_out), np.concatenate(points_out)

    @staticmethod
    def _inverse_distance(points: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Inverse point-to-centroid distance, guarded against coincidence."""
        distances = np.linalg.norm(points - targets, axis=1)
        return 1.0 / np.maximum(distances, np.finfo(np.float64).tiny)

    def _centroids(self, owner_ids: np.ndarray, node_ids: np.ndarray, count: int) -> np.ndarray:
        """Average the given nodes into one centroid per owning entity."""
        points = np.asarray(self.mesh_data["points"], dtype=np.float64)
        centroids = np.zeros((count, 3), dtype=np.float64)
        for axis in range(3):
            centroids[:, axis] = np.bincount(
                owner_ids, weights=points[node_ids, axis], minlength=count
            )
        occurrences = np.bincount(owner_ids, minlength=count)
        return centroids / np.maximum(occurrences, 1)[:, None]

    def _point_interpolation_operator(self, use_boundary: bool) -> dict[str, np.ndarray]:
        """Build a linearly exact cell/boundary → point interpolation.

        Serves the same role as OpenFOAM's ``volPointInterpolation``: each
        mesh point receives a weighted combination of the surrounding
        finite-volume values, and points lying on a boundary take their
        value from the adjacent boundary faces alone so that the applied
        boundary condition appears at the wall.  A plain
        :meth:`~pyvista.DataSetFilters.cell_data_to_point_data` averages
        interior cells only and cannot reproduce that.

        Rather than inverse-distance averaging, the weights come from a
        distance-weighted least-squares fit of a linear function through
        the surrounding values, evaluated at the point.  That reproduces
        any linear field exactly, which plain averaging does not do on a
        graded or unstructured mesh -- the bias there is what makes
        refinement transitions read as a see-saw.

        Args:
            use_boundary: Whether boundary-face values are available to
                          supply the boundary points.

        Returns:
            Cached ``point``/``source``/``weight`` arrays; the weights of
            each point already sum to one.
        """
        cached = self._point_operators.get(use_boundary)
        if cached is not None:
            return cached

        points = np.asarray(self.mesh_data["points"], dtype=np.float64)
        n_points = len(points)
        n_cells = int(self.mesh_data["n_elements"])

        cell_ids, cell_points = self._cell_vertex_incidence()
        cell_centroids = self._centroids(cell_ids, cell_points, n_cells)

        face_ids, face_points = self._boundary_face_incidence()
        n_boundary = int(self.mesh_data["n_faces"]) - int(self.mesh_data["n_interior_faces"])

        if use_boundary and len(face_ids):
            # Boundary faces join the stencil rather than replacing the
            # adjacent cells.  A linear fit is not biased by the extra
            # interior samples the way a plain average would be, and the
            # wider stencil keeps corner points full rank.
            face_centroids = self._centroids(face_ids, face_points, n_boundary)
            row = np.concatenate([cell_points, face_points])
            source = np.concatenate([cell_ids, n_cells + face_ids])
            origin = np.concatenate([cell_centroids[cell_ids], face_centroids[face_ids]])
        else:
            row = cell_points
            source = cell_ids
            origin = cell_centroids[cell_ids]

        # Distance-weighted least squares for f(x) ~ a + b.(x - p); the
        # interpolated value at the point is the constant term a.
        offset = origin - points[row]
        weight = self._inverse_distance(origin, points[row])
        basis = np.empty((len(row), 4), dtype=np.float64)
        basis[:, 0] = 1.0
        basis[:, 1:] = offset

        normal = np.zeros((n_points, 4, 4), dtype=np.float64)
        for i in range(4):
            for j in range(4):
                normal[:, i, j] = np.bincount(
                    row, weights=weight * basis[:, i] * basis[:, j], minlength=n_points
                )
        # A pseudo-inverse keeps coplanar boundary stencils well posed: the
        # wall-normal term is simply undetermined, and the constant is not.
        constant_row = np.linalg.pinv(normal, hermitian=True)[:, 0, :]
        weight = weight * np.einsum("ij,ij->i", constant_row[row], basis)

        # A rank-deficient stencil leaves the pseudo-inverse free to shrink
        # the constant term, so renormalise: reproducing a uniform field
        # exactly matters more than the linear term it may cost.
        totals = np.bincount(row, weights=weight, minlength=n_points)
        weight = weight / np.where(np.abs(totals[row]) > 1.0e-12, totals[row], 1.0)

        operator = {
            "row": row,
            "source": source,
            "weight": weight,
            "n_points": np.asarray(n_points),
        }
        self._point_operators[use_boundary] = operator
        return operator

    def _interpolate_to_points(self, values: np.ndarray, n_cells: int) -> np.ndarray:
        """Scatter cell (and boundary-face) values onto the mesh points.

        Args:
            values:  Cell values, optionally followed by one value per
                     physical boundary face.
            n_cells: Number of physical cells at the head of *values*.

        Returns:
            One interpolated value per mesh point.
        """
        operator = self._point_interpolation_operator(values.shape[0] > n_cells)
        n_points = int(operator["n_points"])
        row = operator["row"]
        source = operator["source"]
        weight = operator["weight"]

        if values.ndim == 1:
            return np.bincount(row, weights=weight * values[source], minlength=n_points)

        result = np.empty((n_points, values.shape[1]), dtype=np.float64)
        for component in range(values.shape[1]):
            result[:, component] = np.bincount(
                row, weights=weight * values[source, component], minlength=n_points
            )
        return result

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

        smooth = self.output.point_interpolation == "boundary_weighted"
        interpolated: dict[str, np.ndarray] = {}

        self._grid.cell_data.clear()
        for name, data in fields.items():
            values = self._field_array(data)
            n_cells = self.mesh_data["n_elements"]
            n_with_boundary = (
                n_cells + self.mesh_data["n_faces"] - self.mesh_data["n_interior_faces"]
            )
            if values.shape[0] != n_cells and values.shape[0] != n_with_boundary:
                raise ValueError(
                    f"VTK field {name!r} has {values.shape[0]} rows; expected "
                    f"{n_cells} cells or {n_with_boundary} cells plus boundary ghosts"
                )
            # Interpolate before discarding the ghosts: the boundary values
            # are precisely what a cell-to-point filter cannot reconstruct.
            if smooth and np.issubdtype(values.dtype, np.floating):
                interpolated[name] = self._interpolate_to_points(values, n_cells)
            if values.shape[0] == n_with_boundary:
                values = values[:n_cells]
            self._grid.cell_data[name] = values

        self._grid.point_data.clear()
        for name, values in interpolated.items():
            self._grid.point_data[name] = self._field_array(values)
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
        location for portability.  Both paths must be expressed against
        the same base: callers build them from a shared solution
        directory, so a relative *vtu_file* is resolved against the
        process working directory exactly as ``self.filename`` was.

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
