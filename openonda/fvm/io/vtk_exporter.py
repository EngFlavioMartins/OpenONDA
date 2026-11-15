"""VTK unstructured-grid output of FVM cell fields and ParaView ``.pvd`` time series."""

from typing import Any

import numpy as np
import pyvista as pv
import vtk

from openonda.core._numba import cacheable_njit as njit
from openonda.core.vtk_output import write_vtk_dataset
from openonda.core.write_precision import cast_for_write

from ..config.output import Output
from .mesh_storage import packed_faces

# Suppress non-fatal VTK-m warnings (e.g., unsupported cell types in Viskores)
vtk.vtkObject.GlobalWarningDisplayOff()


@njit(cache=True)
def _hexahedron_corners(local_faces: np.ndarray, corners: np.ndarray) -> bool:
    """Write the legacy VTK hex order of one cell from its six oriented quad faces.

    Returns false, leaving ``corners`` undefined, unless the faces join eight
    nodes into a hexahedron.
    """
    unique_nodes = np.empty(8, dtype=np.int64)
    unique_count = 0
    for local in range(6):
        for corner in range(4):
            node = local_faces[local, corner]
            seen = False
            for index in range(unique_count):
                if unique_nodes[index] == node:
                    seen = True
                    break
            if not seen:
                if unique_count == 8:
                    return False
                unique_nodes[unique_count] = node
                unique_count += 1
    if unique_count != 8:
        return False

    opposite = -1
    for local in range(1, 6):
        disjoint = True
        for corner in range(4):
            for base in range(4):
                if local_faces[local, corner] == local_faces[0, base]:
                    disjoint = False
                    break
            if not disjoint:
                break
        if disjoint:
            opposite = local
            break
    if opposite < 0:
        return False

    for corner in range(4):
        base_node = local_faces[0, 3 - corner]
        corners[corner] = base_node
        top_node = -1
        for local in range(6):
            for edge in range(4):
                first = local_faces[local, edge]
                second = local_faces[local, (edge + 1) % 4]
                if first == base_node:
                    adjacent = second
                elif second == base_node:
                    adjacent = first
                else:
                    continue
                on_base = False
                for base in range(4):
                    if adjacent == local_faces[0, base]:
                        on_base = True
                        break
                if on_base:
                    continue
                if top_node >= 0 and top_node != adjacent:
                    return False
                top_node = adjacent
        if top_node < 0:
            return False
        corners[corner + 4] = top_node
    for corner in range(4):
        top_node = corners[4 + corner]
        for other in range(corner):
            if top_node == corners[4 + other]:
                return False
        on_opposite = False
        for opposite_corner in range(4):
            if top_node == local_faces[opposite, opposite_corner]:
                on_opposite = True
                break
        if not on_opposite:
            return False
    return True


@njit(cache=True)
def _hex_vertices_from_faces(
    face_vertices: np.ndarray,
    face_offsets: np.ndarray,
    cell_face_indices: np.ndarray,
    cell_face_offset: np.ndarray,
    cell_face_reversed: np.ndarray,
    neighbours: np.ndarray,
    n_internal_faces: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover the legacy VTK hex order of every cell that is a hexahedron.

    Faces are packed as ``face_vertices`` and ``face_offsets``. The result
    holds the eight corners of each cell and a mask that is true for the cells
    with six quad faces that form a hexahedron; the corners of other cells are
    undefined. The optional reversal array is empty when orientation comes
    from neighbours.
    """
    n_cells = len(cell_face_offset) - 1
    vertices = np.empty((n_cells, 8), dtype=np.int64)
    is_hexahedron = np.zeros(n_cells, dtype=np.bool_)
    local_faces = np.empty((6, 4), dtype=np.int64)
    for cell in range(n_cells):
        start = int(cell_face_offset[cell])
        if int(cell_face_offset[cell + 1]) - start != 6:
            continue
        quads = True
        for local in range(6):
            face_id = int(cell_face_indices[start + local])
            first = int(face_offsets[face_id])
            if int(face_offsets[face_id + 1]) - first != 4:
                quads = False
                break
            reverse = (
                bool(cell_face_reversed[start + local])
                if len(cell_face_reversed)
                else face_id < n_internal_faces and int(neighbours[face_id]) == cell
            )
            for corner in range(4):
                local_faces[local, corner] = face_vertices[
                    first + (3 - corner if reverse else corner)
                ]
        if quads:
            is_hexahedron[cell] = _hexahedron_corners(local_faces, vertices[cell])
    return vertices, is_hexahedron


def mesh_cell_fields(
    mesh_data: dict[str, Any], cell_volume: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Cell geometry and meshing source information shared by mesh and time-step output.

    ``cell_size`` is the nominal octree edge before surface projection and
    wrapper insertion. ``cell_volume`` is the solver's physical volume (m³),
    and ``cell_equivalent_size`` is its cube root (m), not a maximum edge.
    Geometry is supplied by the caller so diagnostic export of a rejected
    mesh does not depend on successful geometric validation.
    """
    fields = {}
    for source_name, output_name in (
        ("cell_sizes", "cell_size"),
        ("cell_levels", "refinement_level"),
        ("boundary_layer_index", "boundary_layer_index"),
    ):
        values = mesh_data.get(source_name)
        if values is not None:
            fields[output_name] = np.asarray(values)
    if cell_volume is not None:
        fields["cell_volume"] = np.asarray(cell_volume)
        fields["cell_equivalent_size"] = np.cbrt(cell_volume)
    return fields


class VTKExporter:
    """Write FVM cell fields as VTU files, one cell of the grid per mesh cell.

    The grid is built once from the mesh. A cell with six quad faces that join
    eight nodes is a VTK hexahedron; every other cell is a polyhedron. Fields
    are stored as cell data at the output precision.
    """

    def __init__(
        self,
        mesh_data: dict[str, Any],
        output: Output | None = None,
    ) -> None:
        """Initialise the VTK exporter.

        Builds a :class:`pyvista.UnstructuredGrid` from the mesh data
        on construction.

        Args:
            mesh_data: Mesh dictionary (needs ``vertex_position``, ``faces``,
                       ``owners``, ``neighbours``, ``n_cells``,
                       ``n_interior_faces``).
        """
        self.mesh_data = mesh_data
        self.output = output or Output()
        self._grid = self._initialize_grid()

    def _set_vertex_position(self, vertex_position: np.ndarray | None) -> None:
        """Place the grid points at the snapshot's (possibly moving) mesh coordinates."""
        points = self.mesh_data["vertex_position"] if vertex_position is None else vertex_position
        if not np.array_equal(self._grid.points, points):
            self._grid.points = np.array(points, dtype=np.float64)

    def _field_array(self, data: np.ndarray) -> np.ndarray:
        """Return a contiguous array matching the qualified output precision."""
        return cast_for_write(data, self.output.precision)

    def _initialize_grid(self) -> Any:
        """Construct a :class:`pyvista.UnstructuredGrid` from mesh data.

        Uses the explicit cell vertices of an all-hexahedron mesh when the
        mesh has them. Otherwise the hexahedra are recovered from the faces,
        cell by cell, and the other cells become polyhedra.

        Returns:
            A fully constructed :class:`pyvista.UnstructuredGrid`.
        """
        points = self.mesh_data["vertex_position"]
        cell_vertex_indices = self.mesh_data.get("cell_vertex_indices")
        cell_types = self.mesh_data.get("cell_type_code")
        if (
            cell_vertex_indices is not None
            and cell_types is not None
            and np.asarray(cell_vertex_indices).shape == (self.mesh_data["n_cells"], 8)
            and np.all(np.asarray(cell_types) == 5)  # Gmsh's 8-node hex code
        ):
            vertices = np.asarray(cell_vertex_indices, dtype=np.int64)
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
        n_cells = self.mesh_data["n_cells"]
        n_interior_faces = int(self.mesh_data["n_interior_faces"])

        # Group faces by cell
        cell_face_indices = self.mesh_data.get("cell_face_indices")
        cell_face_offset = self.mesh_data.get("cell_face_offset")
        cell_face_reversed = self.mesh_data.get("cell_face_reversed")
        if cell_face_indices is None or cell_face_offset is None:
            from ..mesh.topology import build_cell_face_csr

            cell_face_indices, cell_face_offset = build_cell_face_csr(
                owners, neighbours, n_cells, self.mesh_data["n_faces"]
            )
            self.mesh_data["cell_face_indices"] = cell_face_indices
            self.mesh_data["cell_face_offset"] = cell_face_offset

        # Ordinary six-quad/eight-corner cells are native VTK hexahedra. This
        # preserves the expected hex-dominant cfMesh appearance in ParaView;
        # only genuine cut and transition cells need polyhedra.
        face_vertices, face_offsets = packed_faces(faces)
        vertices, is_hexahedron = _hex_vertices_from_faces(
            face_vertices,
            face_offsets,
            np.asarray(cell_face_indices),
            np.asarray(cell_face_offset),
            np.asarray(cell_face_reversed, dtype=np.bool_)
            if cell_face_reversed is not None
            else np.empty(0, dtype=np.bool_),
            np.asarray(neighbours),
            n_interior_faces,
        )

        # VTK Polyhedron format:
        # [n_ints, n_faces, num_nodes_f1, n1, n2, ..., num_nodes_f2, n1, n2, ...]
        # ``mesh_data`` permits coarse/fine subfaces whose hanging vertices do
        # not appear on an adjacent long face edge. Native FVM closure remains
        # exact, but VTK_POLYHEDRON requires every local edge to be split at
        # those vertices. Build a compact, cell-local conforming face stream.
        point_values = np.asarray(points, dtype=np.float64)
        records: dict[int, np.ndarray] = {}
        for cell in np.flatnonzero(~is_hexahedron):
            records[int(cell)] = self._polyhedron_record(
                int(cell),
                point_values,
                face_vertices,
                face_offsets,
                cell_face_indices,
                cell_face_offset,
                cell_face_reversed,
                neighbours,
                n_interior_faces,
            )

        lengths = np.full(n_cells, 9, dtype=np.int64)
        for cell, record in records.items():
            lengths[cell] = len(record)
        starts = np.cumsum(lengths) - lengths
        cells = np.empty(int(lengths.sum()), dtype=np.int64)
        hexahedra = np.flatnonzero(is_hexahedron)
        cells[starts[hexahedra]] = 8
        cells[starts[hexahedra, np.newaxis] + 1 + np.arange(8)] = vertices[hexahedra]
        for cell, record in records.items():
            cells[starts[cell] : starts[cell] + len(record)] = record
        cell_types = np.full(n_cells, pv.CellType.POLYHEDRON, dtype=np.uint8)
        cell_types[hexahedra] = pv.CellType.HEXAHEDRON
        return pv.UnstructuredGrid(cells, cell_types, points)

    @staticmethod
    def _polyhedron_record(
        cell: int,
        point_values: np.ndarray,
        face_vertices: np.ndarray,
        face_offsets: np.ndarray,
        cell_face_indices: np.ndarray,
        cell_face_offset: np.ndarray,
        cell_face_reversed: np.ndarray | None,
        neighbours: np.ndarray,
        n_interior_faces: int,
    ) -> np.ndarray:
        """Return the VTK polyhedron record of one cell, with every face edge conforming."""
        start = int(cell_face_offset[cell])
        stop = int(cell_face_offset[cell + 1])
        local_faces: list[np.ndarray] = []
        for local_entry, face_id in enumerate(cell_face_indices[start:stop]):
            face_id = int(face_id)
            nodes = face_vertices[face_offsets[face_id] : face_offsets[face_id + 1]].astype(
                np.int64
            )
            if cell_face_reversed is not None:
                reverse = bool(cell_face_reversed[start + local_entry])
            else:
                reverse = face_id < n_interior_faces and int(neighbours[face_id]) == cell
            if reverse:
                nodes = nodes[::-1]
            local_faces.append(nodes)

        candidate_ids = np.unique(np.concatenate(local_faces))
        candidates = point_values[candidate_ids]
        expanded_faces: list[np.ndarray] = []
        for nodes in local_faces:
            expanded: list[int] = []
            for edge, first_node in enumerate(nodes):
                second_node = int(nodes[(edge + 1) % len(nodes)])
                first_node = int(first_node)
                first_point = point_values[first_node]
                direction = point_values[second_node] - first_point
                length_squared = float(np.dot(direction, direction))
                expanded.append(first_node)
                if length_squared <= np.finfo(np.float64).tiny:
                    continue
                parameter = (candidates - first_point) @ direction / length_squared
                projection = first_point + parameter[:, None] * direction
                distance = np.linalg.norm(candidates - projection, axis=1)
                tolerance = max(1.0e-10 * np.sqrt(length_squared), 1.0e-12)
                interior = np.flatnonzero(
                    (parameter > 1.0e-10) & (parameter < 1.0 - 1.0e-10) & (distance <= tolerance)
                )
                for local_id in interior[np.argsort(parameter[interior])]:
                    expanded.append(int(candidate_ids[local_id]))
            expanded_faces.append(np.asarray(expanded, dtype=np.int64))
        edge_use: dict[tuple[int, int], int] = {}
        for nodes in expanded_faces:
            for edge, first_node in enumerate(nodes):
                first = int(first_node)
                second = int(nodes[(edge + 1) % len(nodes)])
                pair = (first, second) if first < second else (second, first)
                edge_use[pair] = edge_use.get(pair, 0) + 1
        if any(count != 2 for count in edge_use.values()):
            raise ValueError(f"Cell {cell} cannot be represented as a watertight VTK polyhedron")
        record_width = 1 + sum(1 + len(nodes) for nodes in expanded_faces)
        record = [record_width, len(expanded_faces)]
        for nodes in expanded_faces:
            record.append(len(nodes))
            record.extend(int(node) for node in nodes)
        return np.asarray(record, dtype=np.int64)

    def export(
        self,
        filename: str,
        fields: dict[str, np.ndarray],
        *,
        point_fields: dict[str, np.ndarray] | None = None,
        vertex_position: np.ndarray | None = None,
    ) -> str:
        """Export fields to a ``.vtu`` file (VTK unstructured grid format).

        Fields are stored as cell data. Arrays must contain exactly the cells,
        or the cells followed by one ghost value per physical boundary face.

        Args:
            filename:              Output ``.vtu`` file path.
            fields:                Dict mapping field name → array.
            point_fields:          Optional dict of per-point arrays.
            vertex_position:       Vertex coordinates of this snapshot; the
                                   mesh's current coordinates when omitted.

        Returns:
            The output file path.
        """
        self._set_vertex_position(vertex_position)

        self._grid.cell_data.clear()
        for name, data in fields.items():
            values = self._field_array(data)
            n_cells = self.mesh_data["n_cells"]
            n_with_boundary = (
                n_cells + self.mesh_data["n_faces"] - self.mesh_data["n_interior_faces"]
            )
            if values.shape[0] != n_cells and values.shape[0] != n_with_boundary:
                raise ValueError(
                    f"VTK field {name!r} has {values.shape[0]} rows; expected "
                    f"{n_cells} cells or {n_with_boundary} cells plus boundary ghosts"
                )
            if values.shape[0] == n_with_boundary:
                values = values[:n_cells]
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

        write_vtk_dataset(
            self._grid, filename, compression=self.output.compression, uint64_header=True
        )

        return filename
