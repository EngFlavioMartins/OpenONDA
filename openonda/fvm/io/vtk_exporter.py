"""VTK unstructured-grid output of FVM cell fields and ParaView ``.pvd`` time series."""

from array import array
from typing import Any

import numpy as np
import pyvista as pv
import vtk

from openonda.core._numba import cacheable_njit as njit
from openonda.core.vtk_output import write_vtk_dataset
from openonda.core.write_precision import cast_for_write

from ..config.output import Output

# Suppress non-fatal VTK-m warnings (e.g., unsupported cell types in Viskores)
vtk.vtkObject.GlobalWarningDisplayOff()


@njit(cache=True)
def _hex_vertices_from_faces(
    faces: np.ndarray,
    cell_face_indices: np.ndarray,
    cell_face_offset: np.ndarray,
    cell_face_reversed: np.ndarray,
    neighbours: np.ndarray,
    n_internal_faces: int,
) -> tuple[np.ndarray, bool]:
    """Recover the legacy VTK hex order from six oriented quad faces per cell.

    A failed topology check leaves the general polyhedron exporter in charge.
    The optional reversal array is empty when orientation comes from neighbours.
    """
    n_cells = len(cell_face_offset) - 1
    vertices = np.empty((n_cells, 8), dtype=np.int64)
    for cell in range(n_cells):
        start = int(cell_face_offset[cell])
        if int(cell_face_offset[cell + 1]) - start != 6:
            return vertices, False
        local_faces = np.empty((6, 4), dtype=np.int64)
        unique_nodes = np.empty(8, dtype=np.int64)
        unique_count = 0
        for local in range(6):
            face_id = int(cell_face_indices[start + local])
            reverse = (
                bool(cell_face_reversed[start + local])
                if len(cell_face_reversed)
                else face_id < n_internal_faces and int(neighbours[face_id]) == cell
            )
            for corner in range(4):
                node = int(faces[face_id, 3 - corner if reverse else corner])
                local_faces[local, corner] = node
                seen = False
                for index in range(unique_count):
                    if unique_nodes[index] == node:
                        seen = True
                        break
                if not seen:
                    if unique_count == 8:
                        return vertices, False
                    unique_nodes[unique_count] = node
                    unique_count += 1
        if unique_count != 8:
            return vertices, False

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
            return vertices, False

        for corner in range(4):
            base_node = local_faces[0, 3 - corner]
            vertices[cell, corner] = base_node
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
                        return vertices, False
                    top_node = adjacent
            if top_node < 0:
                return vertices, False
            vertices[cell, corner + 4] = top_node
        for corner in range(4):
            top_node = vertices[cell, 4 + corner]
            for other in range(corner):
                if top_node == vertices[cell, 4 + other]:
                    return vertices, False
            on_opposite = False
            for opposite_corner in range(4):
                if top_node == local_faces[opposite, opposite_corner]:
                    on_opposite = True
                    break
            if not on_opposite:
                return vertices, False
    return vertices, True


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
    """
    Exporters results to VTU format using PyVista.
    Supports cell-centred data (Finite Volume).
    Mappings:
        - mesh_data['points'] -> VTK Points
        - mesh_data elements -> VTK Cells (Polyhedron)
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

        Uses native VTK hexahedra whenever explicit cell vertices are
        available.  Other meshes fall back to general polyhedra.

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

        # cfMesh can finish with only hexahedra after discarding its original
        # cell-vertex table. Recover the same VTK ordering in compiled code,
        # rather than repeating Python face/adjacency work for every cell.
        if n_cells >= 1000 and cell_types is not None and np.all(np.asarray(cell_types) == 5):
            try:
                quad_faces = np.asarray(faces, dtype=np.int32)
            except (TypeError, ValueError):
                quad_faces = np.empty((0, 0), dtype=np.int32)
            if quad_faces.shape == (self.mesh_data["n_faces"], 4):
                vertices, valid = _hex_vertices_from_faces(
                    quad_faces,
                    np.asarray(cell_face_indices),
                    np.asarray(cell_face_offset),
                    np.asarray(cell_face_reversed, dtype=np.bool_)
                    if cell_face_reversed is not None
                    else np.empty(0, dtype=np.bool_),
                    np.asarray(neighbours),
                    int(self.mesh_data["n_interior_faces"]),
                )
                if valid:
                    cells = np.empty((n_cells, 9), dtype=np.int64)
                    cells[:, 0] = 8
                    cells[:, 1:] = vertices
                    return pv.UnstructuredGrid(
                        cells.ravel(),
                        np.full(n_cells, pv.CellType.HEXAHEDRON, dtype=np.uint8),
                        points,
                    )

        # Keep ordinary six-quad/eight-corner cells as native VTK hexahedra.
        # This preserves the expected hex-dominant cfMesh appearance in
        # ParaView; only genuine cut and transition cells need polyhedra.
        # VTK Polyhedron format:
        # [n_faces, num_nodes_f1, n1, n2, ..., num_nodes_f2, n1, n2, ...]
        # ``mesh_data`` permits coarse/fine subfaces whose hanging vertices do
        # not appear on an adjacent long face edge. Native FVM closure remains
        # exact, but VTK_POLYHEDRON requires every local edge to be split at
        # those vertices. Build a compact, cell-local conforming face stream.
        # ``array('q')`` avoids the tens of bytes per entry consumed by a
        # Python ``list[int]`` while allowing the expanded length to vary.
        stream = array("q")
        cell_types = np.empty(n_cells, dtype=np.uint8)
        point_values = np.asarray(points, dtype=np.float64)
        for cell in range(n_cells):
            start = int(cell_face_offset[cell])
            stop = int(cell_face_offset[cell + 1])
            face_ids = cell_face_indices[start:stop]
            local_faces: list[np.ndarray] = []
            for local_entry, face_id in enumerate(face_ids):
                nodes = np.asarray(faces[int(face_id)], dtype=np.int64)
                if cell_face_reversed is not None:
                    reverse = bool(cell_face_reversed[start + local_entry])
                else:
                    reverse = (
                        int(face_id) < int(self.mesh_data["n_interior_faces"])
                        and int(neighbours[int(face_id)]) == cell
                    )
                if reverse:
                    nodes = nodes[::-1]
                local_faces.append(nodes)

            candidate_ids = np.unique(np.concatenate(local_faces))

            if (
                len(local_faces) == 6
                and len(candidate_ids) == 8
                and all(len(nodes) == 4 for nodes in local_faces)
            ):
                base = local_faces[0]
                base_set = set(map(int, base))
                opposite = next(
                    (nodes for nodes in local_faces[1:] if base_set.isdisjoint(map(int, nodes))),
                    None,
                )
                adjacency: dict[int, set[int]] = {int(node): set() for node in candidate_ids}
                for nodes in local_faces:
                    for edge, first_node in enumerate(nodes):
                        first = int(first_node)
                        second = int(nodes[(edge + 1) % len(nodes)])
                        adjacency[first].add(second)
                        adjacency[second].add(first)
                vtk_base = base[::-1]
                vtk_top = [
                    next(
                        (adjacent for adjacent in adjacency[int(node)] if adjacent not in base_set),
                        -1,
                    )
                    for node in vtk_base
                ]
                if (
                    opposite is not None
                    and -1 not in vtk_top
                    and len(set(vtk_top)) == 4
                    and set(vtk_top) == set(map(int, opposite))
                ):
                    stream.append(8)
                    stream.extend(map(int, vtk_base))
                    stream.extend(vtk_top)
                    cell_types[cell] = pv.CellType.HEXAHEDRON
                    continue

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
                        (parameter > 1.0e-10)
                        & (parameter < 1.0 - 1.0e-10)
                        & (distance <= tolerance)
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
                raise ValueError(
                    f"Cell {cell} cannot be represented as a watertight VTK polyhedron"
                )
            record_width = 1 + sum(1 + len(nodes) for nodes in expanded_faces)
            stream.append(record_width)
            stream.append(len(expanded_faces))
            for nodes in expanded_faces:
                stream.append(len(nodes))
                stream.extend(map(int, nodes))
            cell_types[cell] = pv.CellType.POLYHEDRON
        cells = np.frombuffer(stream, dtype=np.int64)

        grid = pv.UnstructuredGrid(cells, cell_types, points)
        return grid

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
