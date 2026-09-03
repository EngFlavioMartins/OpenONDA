"""Topology, geometry, and quality validation for FVM meshes."""

from __future__ import annotations

import numpy as np


class MeshValidationError(ValueError):
    """Raised when a mesh fails topology, geometry, or quality validation.

    The validation functions in this module raise this exception (through
    the :func:`_require` helper) when the mesh data cannot safely enter the
    finite-volume solver.  The error message includes the specific
    validation that failed and the offending values where applicable.
    """


def _require(condition, message):
    if not condition:
        raise MeshValidationError(message)


def validate_topology(mesh_data):
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
        _require(name and name not in seen_names, f"Invalid or duplicate boundary name {name!r}")
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
    for key, expected in metadata_shapes.items():
        if key in mesh_data:
            _require(
                np.asarray(mesh_data[key]).shape == (expected,),
                f"Mesh metadata {key!r} must have shape ({expected},)",
            )
    for key in ("global_cell_id", "global_face_id"):
        if key in mesh_data:
            values = np.asarray(mesh_data[key])
            _require(len(np.unique(values)) == len(values), f"Mesh metadata {key!r} is not unique")

    return {
        "n_points": n_points,
        "n_cells": n_cells,
        "n_faces": n_faces,
        "n_internal_faces": n_internal,
        "n_boundary_patches": len(mesh_data["boundary"]),
    }


def validate_geometry(mesh_data, geo_data):
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

    # Mesh validation used to allocate all quality vectors for all faces at
    # once, briefly adding several hundred megabytes to rank zero's global
    # geometry.  Accumulate extrema and sums in blocks instead.
    _require(np.all(np.isfinite(volumes)), "Mesh cell volumes contain non-finite values")
    _require(np.all(np.isfinite(cell_centre)), "Mesh cell cell_centre contain non-finite values")
    _require(np.all(volumes > 0.0), "Mesh contains non-positive cell volumes")
    min_distance = np.full(n_cells, np.inf, dtype=np.float64)
    max_distance = np.zeros(n_cells, dtype=np.float64)
    max_non_orthogonality = 0.0
    sum_non_orthogonality = 0.0
    max_skewness = 0.0
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
        _require(
            np.all(orientation > 0.0),
            "Face orientation is inconsistent with owner-neighbour/boundary direction",
        )
        cosine = np.clip(orientation / (areas_block * magnitude), -1.0, 1.0)
        non_orthogonality = np.degrees(np.arccos(cosine))
        max_non_orthogonality = max(
            max_non_orthogonality, float(np.max(non_orthogonality, initial=0.0))
        )
        sum_non_orthogonality += float(np.sum(non_orthogonality))

        owner_block = owners[face_slice]
        owner_distance = np.linalg.norm(face_centres_block - cell_centre[owner_block], axis=1)
        np.minimum.at(min_distance, owner_block, owner_distance)
        np.maximum.at(max_distance, owner_block, owner_distance)

        internal_stop = min(stop, n_internal)
        if start < internal_stop:
            count = internal_stop - start
            local = slice(0, count)
            neighbour_block = neighbours[start:internal_stop]
            owner_internal = owner_block[local]
            weight_internal = weights_block[local]
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
            max_skewness = max(max_skewness, float(np.max(skewness, initial=0.0)))
            neighbour_distance = np.linalg.norm(
                face_centres_block[local] - cell_centre[neighbour_block], axis=1
            )
            np.minimum.at(min_distance, neighbour_block, neighbour_distance)
            np.maximum.at(max_distance, neighbour_block, neighbour_distance)

    _require(np.all(np.isfinite(min_distance)), "Mesh cell has no adjacent face")
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
    n_quality_cells = (
        int(parallel.n_owned)
        if parallel is not None and parallel.is_partitioned
        else n_cells
    )
    lsq_condition = np.asarray(
        geo_data.get("lsq_condition", []), dtype=np.float64
    )[:n_quality_cells]
    finite_lsq_condition = lsq_condition[np.isfinite(lsq_condition)]

    return {
        "min_volume": float(np.min(volumes)),
        "max_volume": float(np.max(volumes)),
        "min_face_area": float(np.min(areas)),
        "max_non_orthogonality_deg": max_non_orthogonality,
        "mean_non_orthogonality_deg": sum_non_orthogonality / max(n_faces, 1),
        "out_of_bounds_interpolation_weights": out_of_bounds_weights,
        "max_skewness": max_skewness,
        "max_aspect_ratio": max_aspect_ratio,
        "max_lsq_condition": (
            float(np.max(finite_lsq_condition)) if finite_lsq_condition.size else None
        ),
        "rank_deficient_lsq_cells": int(
            np.count_nonzero(
                np.asarray(geo_data.get("lsq_rank", []))[:n_quality_cells] < 3
            )
        ),
        "svd_lsq_cells": int(
            np.count_nonzero(
                np.asarray(geo_data.get("lsq_solver_method", []))[:n_quality_cells]
                == "svd"
            )
        ),
    }


def validate_mesh(mesh_data, geo_data=None):
    """Validate topology and, when supplied, geometry; return one report."""
    report = validate_topology(mesh_data)
    if geo_data is not None:
        report.update(validate_geometry(mesh_data, geo_data))
    return report


def enforce_quality_thresholds(report, mesh_config) -> None:
    """Reject a mesh that exceeds any explicitly configured quality limit."""
    checks = (
        ("max_non_orthogonality_deg", mesh_config.max_non_orthogonality_deg),
        ("max_skewness", mesh_config.max_skewness),
        ("max_aspect_ratio", mesh_config.max_aspect_ratio),
        ("max_lsq_condition", mesh_config.max_lsq_condition),
    )
    violations = []
    for metric, limit in checks:
        if limit is None:
            continue
        if not np.isfinite(limit) or float(limit) <= 0.0:
            raise MeshValidationError(f"Configured mesh quality limit {metric} must be > 0")
        measured = report.get(metric)
        if measured is not None and measured > limit:
            violations.append(f"{metric}={measured:.6g} exceeds configured limit {limit:.6g}")
    if violations:
        raise MeshValidationError("Mesh quality rejection: " + "; ".join(violations))


def validate_wall_vertex_conformance(
    mesh_data,
    surface_triangles,
    wall_patch_name: str,
    *,
    tolerance: float | None = None,
) -> dict[str, float | int]:
    """Require every vertex of a wall patch to lie on the input surface.

    A positive cell volume is not evidence that a curved wall was recovered:
    a displaced Cartesian staircase can have entirely valid volumes.  This
    gate measures the actual wall vertices against the immutable triangulated
    surface and rejects any partial or silently abandoned projection.

    Parameters
    ----------
    mesh_data:
        Native face-based mesh data containing ``wall_patch_name``.
    surface_triangles:
        Input STL triangles, shape ``(n, 3, 3)``.
    wall_patch_name:
        Boundary patch whose vertices define the wall.
    tolerance:
        Absolute distance tolerance.  When omitted, use ``1e-8`` times the
        larger of the geometry span and one model unit.
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
    limit = 1.0e-8 * scale if tolerance is None else float(tolerance)
    if not np.isfinite(limit) or limit <= 0.0:
        raise MeshValidationError("wall conformance tolerance must be finite and positive")
    index = SurfaceIndex.build(surface_triangles)
    distances = np.asarray(
        [index.nearest_point(points[int(vertex_id)])[1] for vertex_id in wall_vertex_ids],
        dtype=np.float64,
    )
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
    cell_centres,
    surface_triangles,
) -> dict[str, int]:
    """Reject fluid-cell centres classified strictly inside a closed surface.

    This is intentionally a point-classification gate rather than a proxy
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


def validate_no_fluid_solid_overlap(mesh_data, solid_bounds, *, tolerance: float = 1.0e-10) -> None:
    """Reject any ordinary fluid cell whose volume overlaps the solid AABB.

    Cells derive their axis-aligned bounds from their *actual vertices*, never
    from a cell-centre test, so a cell whose centre lies outside the body but
    whose volume crosses a body face is caught here.  Touching the solid with
    zero volume is valid (wall-adjacent fluid).

    Args:
        mesh_data: A mesh dictionary with ``points`` and either
            ``cell_vertex_indices`` (hex cells) or the face-based
            ``faces``/``cell_face_indices`` topology.
        solid_bounds: Axis-aligned body bounds ``(xmin, xmax, ymin, ymax,
            zmin, zmax)``.
        tolerance: Positive-volume overlap below this magnitude is treated as
            zero (roundoff).

    Raises:
        MeshValidationError: When any fluid cell overlaps the solid with
            positive volume.
    """
    solid_bounds = np.asarray(solid_bounds, dtype=np.float64)
    if solid_bounds.shape != (6,):
        raise MeshValidationError("solid_bounds must contain six coordinates")
    solid_min = solid_bounds[::2]
    solid_max = solid_bounds[1::2]
    points = np.asarray(mesh_data["vertex_position"], dtype=np.float64)
    n_cells = int(mesh_data["n_cells"])

    cell_vertex_indices = mesh_data.get("cell_vertex_indices")
    if cell_vertex_indices is not None and np.asarray(cell_vertex_indices).ndim == 2:
        vertices = points[np.asarray(cell_vertex_indices, dtype=np.int64)]
        if vertices.shape[0] != n_cells:
            raise MeshValidationError("cell_vertex_indices does not match n_elements")
        cell_min = vertices.min(axis=1)
        cell_max = vertices.max(axis=1)
    else:
        from .topology import build_cell_face_csr

        faces = np.asarray(mesh_data["faces"])
        cell_face_indices, cell_face_offset = build_cell_face_csr(
            mesh_data["owners"], mesh_data["neighbours"], n_cells, mesh_data["n_faces"]
        )
        cell_face_indices = np.asarray(cell_face_indices, dtype=np.int64)
        cell_face_offset = np.asarray(cell_face_offset, dtype=np.int64)
        counts = np.diff(cell_face_offset)
        if counts.max() == counts.min():
            block = faces[cell_face_indices.reshape(-1)]
            cell_min = block.reshape(n_cells, counts[0], -1)[:, :, 0].min(axis=1)
            cell_max = block.reshape(n_cells, counts[0], -1)[:, :, 0].max(axis=1)
        else:
            cell_min = np.empty((n_cells, 3), dtype=np.float64)
            cell_max = np.empty((n_cells, 3), dtype=np.float64)
            for cell in range(n_cells):
                cell_pts = points[
                    np.unique(
                        faces[
                            cell_face_indices[cell_face_offset[cell] : cell_face_offset[cell + 1]]
                        ]
                    )
                ]
                cell_min[cell] = cell_pts.min(axis=0)
                cell_max[cell] = cell_pts.max(axis=0)

    overlap = np.maximum(
        0.0,
        np.minimum(cell_max, solid_max) - np.maximum(cell_min, solid_min),
    )
    overlap_volume = overlap[:, 0] * overlap[:, 1] * overlap[:, 2]
    bad = np.flatnonzero(overlap_volume > tolerance)
    if len(bad):
        ids = bad[:10].tolist()
        raise MeshValidationError(
            f"{len(bad)} fluid cells overlap the solid with positive volume "
            f"(max {overlap_volume[bad].max():.6g}); first global ids {ids}. "
            "Body preservation requires every body plane to be an exact Cartesian "
            "lattice plane."
        )


def _polygon_area(vertices: np.ndarray) -> float:
    centre = vertices.mean(axis=0)
    area = 0.0
    n = len(vertices)
    for i in range(n):
        a = vertices[i] - centre
        b = vertices[(i + 1) % n] - centre
        area += 0.5 * float(np.linalg.norm(np.cross(a, b)))
    return area


def _clip_polygon_to_axis_plane(
    polygon: list[np.ndarray], axis: int, bound: float, keep_greater: bool
) -> list[np.ndarray]:
    """Sutherland--Hodgman clip of a 3-D polygon by one axis plane."""
    if not polygon:
        return []

    def inside(point: np.ndarray) -> bool:
        return bool(point[axis] >= bound if keep_greater else point[axis] <= bound)

    output: list[np.ndarray] = []
    previous = polygon[-1]
    previous_inside = inside(previous)
    for current in polygon:
        current_inside = inside(current)
        if current_inside != previous_inside:
            denominator = current[axis] - previous[axis]
            fraction = 0.0 if denominator == 0.0 else (bound - previous[axis]) / denominator
            output.append(previous + fraction * (current - previous))
        if current_inside:
            output.append(current)
        previous = current
        previous_inside = current_inside
    return output


def _triangles_area_inside_bounds(triangles: np.ndarray, bounds) -> float:
    """Area of a triangulated surface after clipping it to an AABB."""
    total = 0.0
    for triangle in triangles:
        polygon = [vertex.copy() for vertex in triangle]
        for axis in range(3):
            polygon = _clip_polygon_to_axis_plane(polygon, axis, float(bounds[2 * axis]), True)
            polygon = _clip_polygon_to_axis_plane(polygon, axis, float(bounds[2 * axis + 1]), False)
        if len(polygon) < 3:
            continue
        origin = polygon[0]
        for index in range(1, len(polygon) - 1):
            total += 0.5 * float(
                np.linalg.norm(np.cross(polygon[index] - origin, polygon[index + 1] - origin))
            )
    return total


def validate_curved_wall_conformance(
    mesh_data,
    surface_triangles,
    wall_patch_name: str,
    *,
    area_tolerance: float = 0.35,
    surface_clip_bounds=None,
) -> None:
    """Sanity gate for a general (curved) conformal mesh.

    Rejects any cell whose volume is non-positive after surface conformance,
    and rejects a wall patch whose total area deviates from the input STL's
    own surface area by more than ``area_tolerance``. If a closed surface
    crosses the domain boundary, ``surface_clip_bounds`` limits the reference
    area to the portion inside the solved domain.
    """
    points = np.asarray(mesh_data["vertex_position"], dtype=np.float64)
    cell_vertex_indices = mesh_data.get("cell_vertex_indices")
    if cell_vertex_indices is not None:
        from .cartesian.octree import _hex_volume

        vertices = points[np.asarray(cell_vertex_indices, dtype=np.int64)]
        bad = [i for i in range(len(vertices)) if _hex_volume(vertices[i]) <= 0.0]
        if bad:
            raise MeshValidationError(
                f"{len(bad)} cells have non-positive volume after curved-surface "
                f"conformance; first ids {bad[:10]}"
            )

    validate_wall_vertex_conformance(
        mesh_data,
        surface_triangles,
        wall_patch_name,
    )

    faces = mesh_data["faces"]
    (wall,) = [patch for patch in mesh_data["boundary"] if patch["name"] == wall_patch_name]
    first, count = int(wall["start_face"]), int(wall["n_faces"])
    mesh_area = sum(_polygon_area(points[face]) for face in faces[first : first + count])

    surface_triangles = np.asarray(surface_triangles, dtype=np.float64)
    if surface_clip_bounds is None:
        stl_area = 0.5 * float(
            np.sum(
                np.linalg.norm(
                    np.cross(
                        surface_triangles[:, 1] - surface_triangles[:, 0],
                        surface_triangles[:, 2] - surface_triangles[:, 0],
                    ),
                    axis=1,
                )
            )
        )
    else:
        stl_area = _triangles_area_inside_bounds(surface_triangles, surface_clip_bounds)
    relative_error = abs(mesh_area - stl_area) / stl_area if stl_area > 0.0 else float("inf")
    if relative_error > area_tolerance:
        raise MeshValidationError(
            f"Conformal wall area {mesh_area:.6g} deviates from the STL surface area "
            f"{stl_area:.6g} by {relative_error:.1%}, exceeding the {area_tolerance:.0%} tolerance"
        )
