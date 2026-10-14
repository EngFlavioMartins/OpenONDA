"""Topology, geometry, and quality validation for FVM meshes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, TypedDict, overload

from numba import types
from numba.typed import Dict
import numpy as np

from openonda.core._numba import cacheable_njit as njit

from .data import GeometryData, MeshData
from .topology import build_cell_face_csr, pack_face_nodes

if TYPE_CHECKING:
    import pyvista as pv

    from ..config.mesh_quality import MeshQualityConfig


class MeshQualityReport(TypedDict):
    """Mesh-quality extrema from :func:`validate_geometry`; angles in degrees."""

    min_volume: float
    max_volume: float
    min_face_area: float
    max_non_orthogonality_deg: float
    p99_non_orthogonality_deg: float
    mean_non_orthogonality_deg: float
    out_of_bounds_interpolation_weights: int
    max_skewness: float
    p99_internal_face_skewness: float
    max_skewness_face: int
    max_aspect_ratio: float
    min_owner_face_pyramid_cosine: float
    min_neighbour_face_pyramid_cosine: float
    inverted_owner_face_pyramids: int
    inverted_neighbour_face_pyramids: int
    max_adjacent_volume_scale_ratio: float
    max_adjacent_cell_size_ratio: float | None
    max_lsq_condition: float | None
    rank_deficient_lsq_cells: int
    svd_lsq_cells: int


_EDGE_KEY_TYPE = types.UniTuple(types.int64, 2)


class MeshValidationError(ValueError):
    """Raised when a mesh fails topology, geometry, or quality validation.

    The validation functions in this module raise this exception (through
    the :func:`_require` helper) when the mesh data cannot safely enter the
    finite-volume solver.  The error message includes the specific
    validation that failed and the offending values where applicable.
    """


def _require(condition: bool | np.bool_, message: str) -> None:
    if not condition:
        raise MeshValidationError(message)


def validate_topology(mesh_data: MeshData) -> dict[str, int]:
    """Validate the backend-neutral face-based mesh topology.

    The function deliberately rejects ambiguous patch layouts and malformed
    connectivity.  Importers should normalize their data before this boundary.
    """
    required = {
        "vertex_position",
        "faces",
        "owners",
        "neighbours",
        "boundary",
        "n_cells",
        "n_faces",
        "n_interior_faces",
    }
    missing = sorted(required - set(mesh_data))
    _require(not missing, f"Mesh is missing required keys: {missing}")

    points = np.asarray(mesh_data["vertex_position"])
    owners = np.asarray(mesh_data["owners"])
    neighbours = np.asarray(mesh_data["neighbours"])
    faces = mesh_data["faces"]
    n_cells = int(mesh_data["n_cells"])
    n_faces = int(mesh_data["n_faces"])
    n_internal = int(mesh_data["n_interior_faces"])

    _require(points.ndim == 2 and points.shape[1] == 3, "Mesh points must have shape (n, 3)")
    _require(np.all(np.isfinite(points)), "Mesh points contain non-finite coordinates")
    _require(n_cells > 0, "Mesh must contain at least one cell")
    _require(0 <= n_internal <= n_faces, "Invalid interior/total face counts")
    _require(len(faces) == n_faces, "Face list length does not match n_faces")
    _require(owners.shape == (n_faces,), "Owner array length does not match n_faces")
    _require(
        neighbours.shape == (n_internal,),
        "Neighbour array length does not match n_interior_faces",
    )
    _require(np.all((owners >= 0) & (owners < n_cells)), "Owner index outside cell range")
    _require(
        np.all((neighbours >= 0) & (neighbours < n_cells)),
        "Neighbour index outside cell range",
    )
    _require(
        np.all(owners[:n_internal] != neighbours),
        "An interior face connects a cell to itself",
    )

    n_points = points.shape[0]
    try:
        face_nodes = np.asarray(faces)
    except ValueError:
        face_nodes = np.asarray(faces, dtype=object)
    if face_nodes.ndim == 2 and np.issubdtype(face_nodes.dtype, np.integer):
        _require(face_nodes.shape[1] >= 3, "Mesh faces must have at least three nodes")
        invalid = np.flatnonzero((face_nodes < 0) | (face_nodes >= n_points))
        _require(not invalid.size, "Mesh contains a face-node index outside the point range")
        repeated = np.any(np.diff(np.sort(face_nodes, axis=1), axis=1) == 0, axis=1)
        duplicate_faces = np.flatnonzero(repeated)
        if duplicate_faces.size:
            raise MeshValidationError(f"Face {int(duplicate_faces[0])} repeats a node")
    else:
        for i, face in enumerate(faces):
            nodes = np.asarray(face)
            _require(nodes.ndim == 1 and len(nodes) >= 3, f"Face {i} has fewer than three nodes")
            _require(len(np.unique(nodes)) == len(nodes), f"Face {i} repeats a node")
            _require(
                np.all((nodes >= 0) & (nodes < n_points)),
                f"Face {i} contains a point index outside [0, {n_points})",
            )

    expected_start = n_internal
    seen_names = set()
    for patch in mesh_data["boundary"]:
        name = str(patch.get("name", ""))
        _require(
            bool(name) and name not in seen_names, f"Invalid or duplicate boundary name {name!r}"
        )
        seen_names.add(name)
        start = int(patch.get("start_face", -1))
        count = int(patch.get("n_faces", -1))
        _require(count >= 0, f"Boundary {name!r} has a negative face count")
        _require(
            start == expected_start,
            f"Boundary {name!r} starts at {start}; expected contiguous start {expected_start}",
        )
        expected_start += count
    _require(
        expected_start == n_faces,
        f"Boundary patches cover through face {expected_start}, expected {n_faces}",
    )

    edge_report = validate_cell_edge_incidence(mesh_data)

    metadata_shapes = {
        "source_point_ids": n_points,
        "source_cell_id": n_cells,
        "cell_type_code": n_cells,
        "cell_families": n_cells,
        "cell_order": n_cells,
        "boundary_layer_index": n_cells,
        "global_cell_id": n_cells,
        "global_face_id": n_faces,
    }
    # Read-only view for checking optional per-entity arrays by name.
    mesh_arrays: Mapping[str, object] = mesh_data
    for key, expected in metadata_shapes.items():
        if key in mesh_arrays:
            _require(
                np.asarray(mesh_arrays[key]).shape == (expected,),
                f"Mesh metadata {key!r} must have shape ({expected},)",
            )
    for key in ("global_cell_id", "global_face_id"):
        if key in mesh_arrays:
            values = np.asarray(mesh_arrays[key])
            _require(len(np.unique(values)) == len(values), f"Mesh metadata {key!r} is not unique")

    return {
        "n_points": n_points,
        "n_cells": n_cells,
        "n_faces": n_faces,
        "n_internal_faces": n_internal,
        "n_boundary_patches": len(mesh_data["boundary"]),
        **edge_report,
    }


@njit(cache=True)
def _cell_edge_incidence_kernel(
    nodes: np.ndarray, offsets: np.ndarray, cell_faces: np.ndarray, cell_offsets: np.ndarray
) -> tuple[int, int, int]:
    bad_cells = 0
    bad_edges = 0
    first_bad_cell = -1
    for cell in range(len(cell_offsets) - 1):
        incidences = Dict.empty(_EDGE_KEY_TYPE, types.int64)
        for entry in range(cell_offsets[cell], cell_offsets[cell + 1]):
            face = cell_faces[entry]
            first, stop = offsets[face], offsets[face + 1]
            for position in range(first, stop):
                a = nodes[position]
                b = nodes[first if position + 1 == stop else position + 1]
                edge = (min(a, b), max(a, b))
                incidences[edge] = incidences.get(edge, 0) + 1
        invalid = 0
        for count in incidences.values():
            if count != 2:
                invalid += 1
        if invalid:
            bad_cells += 1
            bad_edges += invalid
            if first_bad_cell < 0:
                first_bad_cell = cell
    return bad_cells, bad_edges, first_bad_cell


def validate_cell_edge_incidence(mesh_data: MeshData) -> dict[str, int]:
    """Require every finite-volume cell polygon edge to occur twice.

    Area-vector closure is necessary but not sufficient for a conforming
    polyhedral cell: a coarse face can omit the midpoint vertices introduced
    by a neighbouring fine face while its signed area vectors still cancel.
    Counting undirected polygon edges catches that topological hole before
    geometry or wall treatment can mask it.
    """
    n_cells = int(mesh_data["n_cells"])
    n_internal = int(mesh_data["n_interior_faces"])
    faces = mesh_data["faces"]
    owners = np.asarray(mesh_data["owners"], dtype=np.int64)
    neighbours = np.asarray(mesh_data["neighbours"], dtype=np.int64)
    nodes, offsets = pack_face_nodes(faces)
    cell_faces, cell_offsets = build_cell_face_csr(
        owners, neighbours[:n_internal], n_cells, len(faces)
    )
    bad_cells, bad_edges, first_bad_cell = _cell_edge_incidence_kernel(
        nodes, offsets, cell_faces, cell_offsets
    )

    _require(
        bad_cells == 0,
        "Mesh contains cells with non-closed polygon edges: "
        f"{bad_cells} cells, {bad_edges} unmatched/overused edges; "
        f"first cell {first_bad_cell}",
    )
    return {
        "cells_with_unclosed_edges": bad_cells,
        "unclosed_cell_edges": bad_edges,
    }


def validate_geometry(mesh_data: MeshData, geo_data: GeometryData) -> MeshQualityReport:
    """Validate geometry and return mesh-quality extrema."""
    n_cells = mesh_data["n_cells"]
    n_faces = mesh_data["n_faces"]
    n_internal = mesh_data["n_interior_faces"]

    volumes = np.asarray(geo_data["cell_volume"])
    areas = np.asarray(geo_data["face_area"])
    sf = np.asarray(geo_data["face_area_vector"])
    cf = np.asarray(geo_data["cell_connection_vector"])
    weights = np.asarray(geo_data["face_interpolation_weight"])

    _require(volumes.shape == (n_cells,), "Cell-volume array has the wrong shape")
    _require(areas.shape == (n_faces,), "Face-area array has the wrong shape")
    owners = mesh_data["owners"]
    neighbours = mesh_data["neighbours"]
    cell_centre = np.asarray(geo_data["cell_centre"])
    face_centre = np.asarray(geo_data["face_centre"])
    # Boundary faces and core/layer interfaces are part of the solver's
    # finite-volume geometry.  They must use the same orientation and
    # owner-pyramid checks as internal faces; excluding them hides exactly the
    # wall-distance failures that can make boundary diffusion singular.
    quality_face_mask = np.ones(n_faces, dtype=bool)

    # Accumulate quality extrema and sums in blocks to bound temporary
    # storage alongside rank zero's global geometry.
    _require(np.all(np.isfinite(volumes)), "Mesh cell volumes contain non-finite values")
    _require(np.all(np.isfinite(cell_centre)), "Mesh cell cell_centre contain non-finite values")
    _require(np.all(volumes > 0.0), "Mesh contains non-positive cell volumes")
    min_distance = np.full(n_cells, np.inf, dtype=np.float64)
    max_distance = np.zeros(n_cells, dtype=np.float64)
    max_non_orthogonality = 0.0
    sum_non_orthogonality = 0.0
    non_orthogonality_values: list[np.ndarray] = []
    quality_face_count = 0
    max_skewness = 0.0
    max_skewness_face = -1
    skewness_values: list[np.ndarray] = []
    min_owner_pyramid_cosine = 1.0
    min_neighbour_pyramid_cosine = 1.0
    inverted_owner_pyramids = 0
    inverted_neighbour_pyramids = 0
    max_adjacent_volume_scale_ratio = 1.0
    max_adjacent_cell_size_ratio = 1.0
    cell_sizes = np.asarray(mesh_data.get("cell_sizes", ()), dtype=np.float64)
    has_cell_sizes = cell_sizes.shape == (n_cells,)
    out_of_bounds_weights = 0
    chunk_size = 250_000
    for start in range(0, n_faces, chunk_size):
        stop = min(start + chunk_size, n_faces)
        face_slice = slice(start, stop)
        areas_block = areas[face_slice]
        sf_block = sf[face_slice]
        cf_block = cf[face_slice]
        weights_block = weights[face_slice]
        face_centres_block = face_centre[face_slice]
        _require(
            np.all(np.isfinite(areas_block))
            and np.all(np.isfinite(sf_block))
            and np.all(np.isfinite(cf_block))
            and np.all(np.isfinite(weights_block))
            and np.all(np.isfinite(face_centres_block)),
            "Mesh face geometry contains non-finite values",
        )
        _require(np.all(areas_block > 0.0), "Mesh contains zero-area faces")
        magnitude = np.linalg.norm(cf_block, axis=1)
        _require(np.all(magnitude > 0.0), "Mesh contains zero cell-to-cell/face distances")
        orientation = np.einsum("ij,ij->i", sf_block, cf_block)
        quality_faces = quality_face_mask[face_slice]
        _require(
            np.all(orientation[quality_faces] > 0.0),
            "Face orientation is inconsistent with owner-neighbour/boundary direction",
        )
        cosine = np.clip(orientation / (areas_block * magnitude), -1.0, 1.0)
        non_orthogonality = np.degrees(np.arccos(cosine))
        quality_non_orthogonality = non_orthogonality[quality_faces]
        max_non_orthogonality = max(
            max_non_orthogonality,
            float(np.max(quality_non_orthogonality, initial=0.0)),
        )
        sum_non_orthogonality += float(np.sum(quality_non_orthogonality))
        quality_face_count += int(quality_non_orthogonality.size)
        if quality_non_orthogonality.size:
            non_orthogonality_values.append(quality_non_orthogonality)

        owner_block = owners[face_slice]
        owner_vectors = face_centres_block - cell_centre[owner_block]
        owner_distance = np.linalg.norm(owner_vectors, axis=1)
        owner_pyramids = np.einsum("ij,ij->i", sf_block, owner_vectors)
        bad_owner = np.flatnonzero((owner_pyramids <= 0.0) & quality_faces)
        inverted_owner_pyramids += int(len(bad_owner))
        owner_cosines = owner_pyramids / np.maximum(
            areas_block * owner_distance,
            np.finfo(np.float64).tiny,
        )
        owner_cosines = owner_cosines[quality_faces]
        min_owner_pyramid_cosine = min(
            min_owner_pyramid_cosine,
            float(np.min(owner_cosines, initial=1.0)),
        )
        np.minimum.at(min_distance, owner_block, owner_distance)
        np.maximum.at(max_distance, owner_block, owner_distance)

        internal_stop = min(stop, n_internal)
        if start < internal_stop:
            count = internal_stop - start
            local = slice(0, count)
            neighbour_block = neighbours[start:internal_stop]
            owner_internal = owner_block[local]
            weight_internal = weights_block[local]
            neighbour_vectors = cell_centre[neighbour_block] - face_centres_block[local]
            neighbour_distance = np.linalg.norm(neighbour_vectors, axis=1)
            neighbour_pyramids = np.einsum(
                "ij,ij->i",
                sf_block[local],
                neighbour_vectors,
            )
            quality_internal = quality_faces[:count]
            bad_neighbour = np.flatnonzero((neighbour_pyramids <= 0.0) & quality_internal)
            inverted_neighbour_pyramids += int(len(bad_neighbour))
            neighbour_cosines = neighbour_pyramids / np.maximum(
                areas_block[local] * neighbour_distance,
                np.finfo(np.float64).tiny,
            )
            neighbour_cosines = neighbour_cosines[quality_internal]
            min_neighbour_pyramid_cosine = min(
                min_neighbour_pyramid_cosine,
                float(np.min(neighbour_cosines, initial=1.0)),
            )
            out_of_bounds_weights += int(
                np.count_nonzero((weight_internal < 0.0) | (weight_internal > 1.0))
            )
            interpolation = (1.0 - weight_internal[:, np.newaxis]) * cell_centre[
                owner_internal
            ] + weight_internal[:, np.newaxis] * cell_centre[neighbour_block]
            centre_distance = np.linalg.norm(
                cell_centre[neighbour_block] - cell_centre[owner_internal], axis=1
            )
            skewness = np.linalg.norm(face_centres_block[local] - interpolation, axis=1) / (
                centre_distance + 1e-30
            )
            skewness = skewness[quality_internal]
            local_maximum = float(np.max(skewness, initial=0.0))
            if skewness.size:
                skewness_values.append(skewness)
            if local_maximum > max_skewness:
                max_skewness = local_maximum
                max_skewness_face = start + int(np.argmax(skewness))
            owner_scale = np.cbrt(volumes[owner_internal])
            neighbour_scale = np.cbrt(volumes[neighbour_block])
            volume_scale_ratio = np.maximum(owner_scale, neighbour_scale) / np.maximum(
                np.minimum(owner_scale, neighbour_scale),
                np.finfo(np.float64).tiny,
            )
            max_adjacent_volume_scale_ratio = max(
                max_adjacent_volume_scale_ratio,
                float(np.max(volume_scale_ratio, initial=1.0)),
            )
            if has_cell_sizes:
                nominal_ratio = np.maximum(
                    cell_sizes[owner_internal],
                    cell_sizes[neighbour_block],
                ) / np.maximum(
                    np.minimum(cell_sizes[owner_internal], cell_sizes[neighbour_block]),
                    np.finfo(np.float64).tiny,
                )
                max_adjacent_cell_size_ratio = max(
                    max_adjacent_cell_size_ratio,
                    float(np.max(nominal_ratio, initial=1.0)),
                )
            np.minimum.at(min_distance, neighbour_block, neighbour_distance)
            np.maximum.at(max_distance, neighbour_block, neighbour_distance)

    _require(np.all(np.isfinite(min_distance)), "Mesh cell has no adjacent face")
    _require(
        inverted_owner_pyramids == 0,
        f"Mesh contains non-positive owner face pyramids: {inverted_owner_pyramids} faces",
    )
    _require(
        inverted_neighbour_pyramids == 0,
        f"Mesh contains non-positive neighbour face pyramids: {inverted_neighbour_pyramids} faces",
    )
    max_aspect_ratio = 0.0
    for start in range(0, n_cells, chunk_size):
        stop = min(start + chunk_size, n_cells)
        ratios = max_distance[start:stop] / np.maximum(min_distance[start:stop], 1e-30)
        max_aspect_ratio = max(max_aspect_ratio, float(np.max(ratios, initial=0.0)))

    # Partitioned meshes store owned cells first and halo cells afterward.
    # Halo-only LSQ rows are communication placeholders on some ranks and are
    # not solved as local control volumes; including them can report enormous
    # artificial condition numbers or deficient ranks for an otherwise valid
    # global mesh.  Every owned cell is checked exactly once across the MPI
    # communicator.
    parallel = mesh_data.get("_parallel_context")
    n_owned = parallel.n_owned if parallel is not None and parallel.is_partitioned else None
    n_quality_cells = n_owned if n_owned is not None else n_cells
    lsq_condition = np.asarray(geo_data.get("lsq_condition", []), dtype=np.float64)[
        :n_quality_cells
    ]
    finite_lsq_condition = lsq_condition[np.isfinite(lsq_condition)]

    return {
        "min_volume": float(np.min(volumes)),
        "max_volume": float(np.max(volumes)),
        "min_face_area": float(np.min(areas)),
        "max_non_orthogonality_deg": max_non_orthogonality,
        "p99_non_orthogonality_deg": float(
            np.quantile(np.concatenate(non_orthogonality_values), 0.99)
            if non_orthogonality_values
            else 0.0
        ),
        "mean_non_orthogonality_deg": sum_non_orthogonality / max(quality_face_count, 1),
        "out_of_bounds_interpolation_weights": out_of_bounds_weights,
        "max_skewness": max_skewness,
        "p99_internal_face_skewness": float(
            np.quantile(np.concatenate(skewness_values), 0.99) if skewness_values else 0.0
        ),
        "max_skewness_face": max_skewness_face,
        "max_aspect_ratio": max_aspect_ratio,
        "min_owner_face_pyramid_cosine": min_owner_pyramid_cosine,
        "min_neighbour_face_pyramid_cosine": min_neighbour_pyramid_cosine,
        "inverted_owner_face_pyramids": inverted_owner_pyramids,
        "inverted_neighbour_face_pyramids": inverted_neighbour_pyramids,
        "max_adjacent_volume_scale_ratio": max_adjacent_volume_scale_ratio,
        "max_adjacent_cell_size_ratio": (max_adjacent_cell_size_ratio if has_cell_sizes else None),
        "max_lsq_condition": (
            float(np.max(finite_lsq_condition)) if finite_lsq_condition.size else None
        ),
        "rank_deficient_lsq_cells": int(
            np.count_nonzero(np.asarray(geo_data.get("lsq_rank", []))[:n_quality_cells] < 3)
        ),
        "svd_lsq_cells": int(
            np.count_nonzero(
                np.asarray(geo_data.get("lsq_solver_method", []))[:n_quality_cells] == "svd"
            )
        ),
    }


def validate_cell_area_closure(
    mesh_data: MeshData,
    geo_data: GeometryData,
    *,
    tolerance: float = 1.0e-10,
) -> dict[str, float]:
    """Validate the normalized divergence-theorem area-vector closure.

    Face vectors point out of the owner.  Internal faces therefore contribute
    with the opposite sign to the neighbour; a closed finite-volume cell must
    sum to zero independently of its polygon valence.
    """
    n_cells = int(mesh_data["n_cells"])
    n_internal = int(mesh_data["n_interior_faces"])
    owners = np.asarray(mesh_data["owners"], dtype=np.int64)
    neighbours = np.asarray(mesh_data["neighbours"], dtype=np.int64)
    vectors = np.asarray(geo_data["face_area_vector"], dtype=np.float64)
    areas = np.linalg.norm(vectors, axis=1)
    closure = np.zeros((n_cells, 3), dtype=np.float64)
    np.add.at(closure, owners, vectors)
    np.add.at(closure, neighbours, -vectors[:n_internal])
    denominator = np.zeros(n_cells, dtype=np.float64)
    np.add.at(denominator, owners, areas)
    np.add.at(denominator, neighbours, areas[:n_internal])
    normalized = np.linalg.norm(closure, axis=1) / np.maximum(
        denominator, np.finfo(np.float64).tiny
    )
    maximum = float(np.max(normalized, initial=0.0))
    _require(
        maximum <= tolerance,
        f"Cell area-vector closure error {maximum:.6g} exceeds tolerance {tolerance:.6g}",
    )
    return {"max_normalized_area_closure_error": maximum}


def validate_single_fluid_component(mesh_data: MeshData) -> dict[str, int]:
    """Require all solver cells to belong to one internal-face component."""
    n_cells = int(mesh_data["n_cells"])
    n_internal = int(mesh_data["n_interior_faces"])
    owners = np.asarray(mesh_data["owners"], dtype=np.int64)[:n_internal]
    neighbours = np.asarray(mesh_data["neighbours"], dtype=np.int64)
    adjacency: list[list[int]] = [[] for _ in range(n_cells)]
    for owner, neighbour in zip(owners, neighbours, strict=True):
        adjacency[int(owner)].append(int(neighbour))
        adjacency[int(neighbour)].append(int(owner))
    visited = np.zeros(n_cells, dtype=bool)
    stack = [0] if n_cells else []
    while stack:
        cell = stack.pop()
        if visited[cell]:
            continue
        visited[cell] = True
        stack.extend(neighbour for neighbour in adjacency[cell] if not visited[neighbour])
    missing = np.flatnonzero(~visited)
    _require(
        not len(missing),
        f"Fluid mesh has {len(missing) + 1} disconnected cell components; "
        f"first unconnected cell {int(missing[0]) if len(missing) else -1}",
    )
    return {"fluid_components": 1, "fluid_cells": n_cells}


def validate_vtk_cell_intersections(
    dataset: pv.DataSet, *, maximum_intersections: int = 0
) -> dict[str, int]:
    """Reject intersecting VTK cells and report other validator classifications.

    VTK's validity state is a bit mask.  Concave cut cells can legitimately be
    reported as non-convex, so this check rejects the two geometric
    intersection bits while the native face-pyramid checks above establish
    finite-volume orientation and star-shapedness about the cell centres.
    ``maximum_intersections`` is used only for validated no-regression
    comparisons; production validation retains the default of zero.
    """
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy  # pyrefly: ignore[missing-import]
    except ImportError as exc:  # pragma: no cover - FVM dependencies include VTK.
        raise MeshValidationError("VTK is required for cell-intersection validation") from exc

    validator = vtk.vtkCellValidator()
    validator.SetInputData(dataset)
    validator.Update()
    state_array = validator.GetOutput().GetCellData().GetArray("ValidityState")
    if state_array is None:
        raise MeshValidationError("VTK cell validator did not return ValidityState")
    states = np.asarray(vtk_to_numpy(state_array), dtype=np.int64)
    intersecting = np.flatnonzero(states & 0b000110)
    if maximum_intersections < 0:
        raise MeshValidationError("maximum_intersections must not be negative")
    if intersecting.size > maximum_intersections:
        examples = ", ".join(map(str, intersecting[:8]))
        raise MeshValidationError(
            f"Mesh contains {len(intersecting)} intersecting VTK cells; first ids: {examples}"
        )
    return {
        "cell_count": int(len(states)),
        "intersecting_cells": int(len(intersecting)),
        "nonconvex_cells": int(np.count_nonzero(states & 0b010000)),
        "incorrectly_oriented_vtk_cells": int(np.count_nonzero(states & 0b100000)),
    }


def cells_incident_to_points(mesh_data: MeshData, point_ids: list[int] | np.ndarray) -> np.ndarray:
    """Find every cell whose geometry can change when these points move."""
    point_ids = np.asarray(point_ids, dtype=np.int64)
    if not len(point_ids):
        return np.empty(0, dtype=np.int32)
    if int(point_ids.min()) < 0 or int(point_ids.max()) >= int(mesh_data["n_points"]):
        raise ValueError("Point selection contains an id outside the mesh")
    point_mask = np.zeros(int(mesh_data["n_points"]), dtype=np.bool_)
    point_mask[point_ids] = True
    faces = mesh_data["faces"]
    try:
        face_nodes = np.asarray(faces, dtype=np.int32)
    except (TypeError, ValueError):
        face_nodes = None
    if face_nodes is not None and face_nodes.ndim == 2:
        touched_faces = np.any(point_mask[face_nodes], axis=1)
    else:
        touched_faces = np.fromiter(
            (np.any(point_mask[np.asarray(face, dtype=np.int32)]) for face in faces),
            dtype=np.bool_,
            count=int(mesh_data["n_faces"]),
        )
    owners = np.asarray(mesh_data["owners"], dtype=np.int32)
    neighbours = np.asarray(mesh_data["neighbours"], dtype=np.int32)
    n_internal = int(mesh_data["n_interior_faces"])
    return np.unique(
        np.concatenate((owners[touched_faces], neighbours[touched_faces[:n_internal]]))
    )


@overload
def extract_cell_subset_mesh(
    mesh_data: MeshData, cell_ids: np.ndarray, *, return_point_ids: Literal[False] = False
) -> MeshData: ...


@overload
def extract_cell_subset_mesh(
    mesh_data: MeshData, cell_ids: np.ndarray, *, return_point_ids: Literal[True]
) -> tuple[MeshData, np.ndarray]: ...


def extract_cell_subset_mesh(
    mesh_data: MeshData, cell_ids: np.ndarray, *, return_point_ids: bool = False
) -> MeshData | tuple[MeshData, np.ndarray]:
    """Return a closed native mesh containing only the selected cells.

    Faces against unselected cells become temporary boundary faces.  Only
    points referenced by the selected cells are retained, keeping both VTK
    conversion and the curved-face compaction pass local to the surface band.
    With ``return_point_ids``, also return the original ids of those points so
    callers can update the subset's coordinates after moving mesh vertices.
    """
    selected_ids = np.unique(np.asarray(cell_ids, dtype=np.int32))
    n_cells = int(mesh_data["n_cells"])
    if not len(selected_ids):
        raise ValueError("Cell subset must contain at least one cell")
    if int(selected_ids[0]) < 0 or int(selected_ids[-1]) >= n_cells:
        raise ValueError("Cell subset contains an id outside the mesh")
    selected = np.zeros(n_cells, dtype=bool)
    selected[selected_ids] = True
    cell_map = np.full(n_cells, -1, dtype=np.int32)
    cell_map[selected_ids] = np.arange(len(selected_ids), dtype=np.int32)
    faces = mesh_data["faces"]
    owners = np.asarray(mesh_data["owners"], dtype=np.int32)
    neighbours = np.asarray(mesh_data["neighbours"], dtype=np.int32)
    n_internal = int(mesh_data["n_interior_faces"])

    owner_selected = selected[owners[:n_internal]]
    neighbour_selected = selected[neighbours]
    retained_internal = np.flatnonzero(owner_selected & neighbour_selected)
    exposed_internal = np.flatnonzero(owner_selected ^ neighbour_selected)
    retained_boundary = n_internal + np.flatnonzero(selected[owners[n_internal:]])

    subset_faces = [
        np.asarray(faces[int(face_id)], dtype=np.int32).copy() for face_id in retained_internal
    ]
    subset_owners = [int(cell_map[owners[int(face_id)]]) for face_id in retained_internal]
    subset_neighbours = [int(cell_map[neighbours[int(face_id)]]) for face_id in retained_internal]
    boundary_faces: list[np.ndarray] = []
    boundary_owners: list[int] = []
    for face_id_value in exposed_internal:
        face_id = int(face_id_value)
        if owner_selected[face_id]:
            boundary_faces.append(np.asarray(faces[face_id], dtype=np.int32).copy())
            boundary_owners.append(int(cell_map[owners[face_id]]))
        else:
            boundary_faces.append(np.asarray(faces[face_id], dtype=np.int32)[::-1].copy())
            boundary_owners.append(int(cell_map[neighbours[face_id]]))
    for face_id_value in retained_boundary:
        face_id = int(face_id_value)
        boundary_faces.append(np.asarray(faces[face_id], dtype=np.int32).copy())
        boundary_owners.append(int(cell_map[owners[face_id]]))

    subset_faces.extend(boundary_faces)
    subset_owners.extend(boundary_owners)
    used_points = np.unique(np.concatenate(subset_faces))
    point_map = np.full(int(mesh_data["n_points"]), -1, dtype=np.int32)
    point_map[used_points] = np.arange(len(used_points), dtype=np.int32)
    subset_faces = [point_map[face] for face in subset_faces]
    widths = {len(face) for face in subset_faces}
    result: MeshData = {
        "vertex_position": np.ascontiguousarray(
            np.asarray(mesh_data["vertex_position"], dtype=np.float64)[used_points]
        ),
        "faces": (
            np.ascontiguousarray(subset_faces, dtype=np.int32) if len(widths) == 1 else subset_faces
        ),
        "owners": np.ascontiguousarray(subset_owners, dtype=np.int32),
        "neighbours": np.ascontiguousarray(subset_neighbours, dtype=np.int32),
        "boundary": [
            {
                "name": "validation_boundary",
                "start_face": len(retained_internal),
                "n_faces": len(boundary_faces),
                "type": "patch",
            }
        ],
        "n_cells": len(selected_ids),
        "n_faces": len(subset_faces),
        "n_interior_faces": len(retained_internal),
        "n_points": len(used_points),
    }
    if "cell_vertex_indices" in mesh_data:
        result["cell_vertex_indices"] = np.ascontiguousarray(
            point_map[np.asarray(mesh_data["cell_vertex_indices"], dtype=np.int32)[selected_ids]],
            dtype=np.int32,
        )
    if "cell_type_code" in mesh_data:
        result["cell_type_code"] = np.ascontiguousarray(
            np.asarray(mesh_data["cell_type_code"])[selected_ids]
        )
    return (result, used_points) if return_point_ids else result


def enforce_quality_thresholds(report: MeshQualityReport, mesh_config: MeshQualityConfig) -> None:
    """Reject a mesh that exceeds any explicitly configured quality limit."""
    checks = (
        (
            "max_non_orthogonality_deg",
            report["max_non_orthogonality_deg"],
            mesh_config.max_non_orthogonality_deg,
        ),
        ("max_skewness", report["max_skewness"], mesh_config.max_skewness),
        ("max_aspect_ratio", report["max_aspect_ratio"], mesh_config.max_aspect_ratio),
        ("max_lsq_condition", report["max_lsq_condition"], mesh_config.max_lsq_condition),
    )
    violations = []
    for metric, measured, limit in checks:
        if limit is None:
            continue
        if not np.isfinite(limit) or float(limit) <= 0.0:
            raise MeshValidationError(f"Configured mesh quality limit {metric} must be > 0")
        if measured is not None and measured > limit:
            violations.append(f"{metric}={measured:.6g} exceeds configured limit {limit:.6g}")
    if violations:
        raise MeshValidationError("Mesh quality rejection: " + "; ".join(violations))


def validate_wall_vertex_conformance(
    mesh_data: MeshData,
    surface_triangles: np.ndarray,
    wall_patch_name: str,
) -> dict[str, float | int]:
    """Require every vertex of a wall patch to lie on the input surface.

    A positive cell volume is not evidence that a curved wall was recovered:
    a displaced Cartesian staircase can have entirely valid volumes.  This
    check measures the actual wall vertices against the immutable triangulated
    surface and rejects any partial or silently abandoned projection.

    Parameters
    ----------
    mesh_data:
        Native face-based mesh data containing ``wall_patch_name``.
    surface_triangles:
        Input STL triangles, shape ``(n, 3, 3)``.
    wall_patch_name:
        Boundary patch whose vertices define the wall. Vertices must lie
        within ``1e-8`` times the larger of the geometry span and one model
        unit.
    """
    from .surface_classification import SurfaceIndex

    points = np.asarray(mesh_data["vertex_position"], dtype=np.float64)
    patch = next(
        (item for item in mesh_data["boundary"] if item["name"] == wall_patch_name),
        None,
    )
    if patch is None:
        raise MeshValidationError(f"Wall patch {wall_patch_name!r} is missing")
    start = int(patch["start_face"])
    stop = start + int(patch["n_faces"])
    faces = mesh_data["faces"][start:stop]
    if not len(faces):
        raise MeshValidationError(f"Wall patch {wall_patch_name!r} has no faces")
    wall_vertex_ids = np.unique(
        np.concatenate([np.asarray(face, dtype=np.int64) for face in faces])
    )
    surface_triangles = np.asarray(surface_triangles, dtype=np.float64)
    scale = max(float(np.ptp(surface_triangles, axis=(0, 1)).max()), 1.0)
    limit = 1.0e-8 * scale
    index = SurfaceIndex.build(surface_triangles)
    _nearest, distances, _triangle_ids = index.nearest_points(points[wall_vertex_ids])
    maximum = float(distances.max(initial=0.0))
    if maximum > limit:
        worst = int(wall_vertex_ids[int(np.argmax(distances))])
        raise MeshValidationError(
            f"Wall patch {wall_patch_name!r} is not conformal to the input surface: "
            f"maximum vertex distance {maximum:.6g} exceeds tolerance {limit:.6g} "
            f"(worst vertex {worst})"
        )
    return {
        "vertex_count": int(len(wall_vertex_ids)),
        "max_vertex_distance": maximum,
        "mean_vertex_distance": float(distances.mean() if len(distances) else 0.0),
        "tolerance": limit,
    }


def validate_no_fluid_cell_centres_inside_surface(
    cell_centres: np.ndarray,
    surface_triangles: np.ndarray,
) -> dict[str, int]:
    """Reject fluid-cell centres classified strictly inside a closed surface.

    This is intentionally a point-classification check rather than a proxy
    based on the surface bounding box.  It catches cells left inside a curved
    body when a staircase or incomplete recovery stage is returned.
    """
    from .surface_classification import SurfaceIndex

    values = np.asarray(cell_centres, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
        raise MeshValidationError("cell_centres must have shape (n, 3) and contain finite values")
    index = SurfaceIndex.build(np.asarray(surface_triangles, dtype=np.float64))
    inside = index.is_inside(values)
    bad = np.flatnonzero(inside)
    if bad.size:
        raise MeshValidationError(
            f"{bad.size} fluid-cell centres are inside the input surface; "
            f"first cell ids {bad[:10].tolist()}"
        )
    return {"cell_count": int(len(values)), "inside_count": 0}
