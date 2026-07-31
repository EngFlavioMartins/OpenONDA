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
        "points",
        "faces",
        "owners",
        "neighbours",
        "boundary",
        "n_elements",
        "n_faces",
        "n_interior_faces",
    }
    missing = sorted(required - set(mesh_data))
    _require(not missing, f"Mesh is missing required keys: {missing}")

    points = np.asarray(mesh_data["points"])
    owners = np.asarray(mesh_data["owners"])
    neighbours = np.asarray(mesh_data["neighbours"])
    faces = mesh_data["faces"]
    n_cells = int(mesh_data["n_elements"])
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
        start = int(patch.get("startFace", -1))
        count = int(patch.get("nFaces", -1))
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
        "source_cell_ids": n_cells,
        "cell_type_codes": n_cells,
        "cell_families": n_cells,
        "cell_orders": n_cells,
        "global_cell_ids": n_cells,
        "global_face_ids": n_faces,
    }
    for key, expected in metadata_shapes.items():
        if key in mesh_data:
            _require(
                np.asarray(mesh_data[key]).shape == (expected,),
                f"Mesh metadata {key!r} must have shape ({expected},)",
            )
    for key in ("global_cell_ids", "global_face_ids"):
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
    n_cells = mesh_data["n_elements"]
    n_faces = mesh_data["n_faces"]
    n_internal = mesh_data["n_interior_faces"]

    volumes = np.asarray(geo_data["element_volumes"])
    areas = np.asarray(geo_data["face_areas"])
    sf = np.asarray(geo_data["face_sf"])
    cf = np.asarray(geo_data["face_cf_vector"])
    weights = np.asarray(geo_data["face_weights"])

    _require(volumes.shape == (n_cells,), "Cell-volume array has the wrong shape")
    _require(areas.shape == (n_faces,), "Face-area array has the wrong shape")
    for name, values in (
        ("cell volumes", volumes),
        ("face areas", areas),
        ("face area vectors", sf),
        ("cell/face vectors", cf),
        ("interpolation weights", weights),
    ):
        _require(np.all(np.isfinite(values)), f"Mesh {name} contain non-finite values")
    _require(np.all(volumes > 0.0), "Mesh contains non-positive cell volumes")
    _require(np.all(areas > 0.0), "Mesh contains zero-area faces")

    mag_cf = np.linalg.norm(cf, axis=1)
    _require(np.all(mag_cf > 0.0), "Mesh contains zero cell-to-cell/face distances")
    orientation = np.sum(sf * cf, axis=1)
    _require(
        np.all(orientation > 0.0),
        "Face orientation is inconsistent with owner-neighbour/boundary direction",
    )

    cosine = np.clip(orientation / (areas * mag_cf), -1.0, 1.0)
    non_orthogonality = np.degrees(np.arccos(cosine))
    internal_weights = weights[:n_internal]
    out_of_bounds_weights = int(
        np.count_nonzero((internal_weights < 0.0) | (internal_weights > 1.0))
    )
    owners = mesh_data["owners"]
    neighbours = mesh_data["neighbours"]
    centroids = np.asarray(geo_data["element_centroids"])
    face_centroids = np.asarray(geo_data["face_centroids"])
    interpolation_points = (1.0 - internal_weights[:, np.newaxis]) * centroids[
        owners[:n_internal]
    ] + internal_weights[:, np.newaxis] * centroids[neighbours]
    centre_distance = np.linalg.norm(centroids[neighbours] - centroids[owners[:n_internal]], axis=1)
    skewness = np.linalg.norm(face_centroids[:n_internal] - interpolation_points, axis=1) / (
        centre_distance + 1e-30
    )

    # Reduce owner and neighbour face distances directly into cell extrema.
    # This avoids one Python list object per cell during large-mesh validation.
    minimum_distance = np.full(n_cells, np.inf, dtype=np.float64)
    maximum_distance = np.zeros(n_cells, dtype=np.float64)
    owner_distance = np.linalg.norm(face_centroids - centroids[owners], axis=1)
    np.minimum.at(minimum_distance, owners, owner_distance)
    np.maximum.at(maximum_distance, owners, owner_distance)
    if n_internal:
        neighbour_distance = np.linalg.norm(
            face_centroids[:n_internal] - centroids[neighbours], axis=1
        )
        np.minimum.at(minimum_distance, neighbours, neighbour_distance)
        np.maximum.at(maximum_distance, neighbours, neighbour_distance)
    _require(np.all(np.isfinite(minimum_distance)), "Mesh cell has no adjacent face")
    aspect_ratio = maximum_distance / np.maximum(minimum_distance, 1e-30)

    lsq_condition = np.asarray(geo_data.get("lsq_condition", []), dtype=np.float64)
    finite_lsq_condition = lsq_condition[np.isfinite(lsq_condition)]

    return {
        "min_volume": float(np.min(volumes)),
        "max_volume": float(np.max(volumes)),
        "min_face_area": float(np.min(areas)),
        "max_non_orthogonality_deg": float(np.max(non_orthogonality)),
        "mean_non_orthogonality_deg": float(np.mean(non_orthogonality)),
        "out_of_bounds_interpolation_weights": out_of_bounds_weights,
        "max_skewness": float(np.max(skewness)) if skewness.size else 0.0,
        "max_aspect_ratio": float(np.max(aspect_ratio)),
        "max_lsq_condition": (
            float(np.max(finite_lsq_condition)) if finite_lsq_condition.size else None
        ),
        "rank_deficient_lsq_cells": int(
            np.count_nonzero(np.asarray(geo_data.get("lsq_rank", [])) < 3)
        ),
        "svd_lsq_cells": int(
            np.count_nonzero(np.asarray(geo_data.get("lsq_solver_method", [])) == "svd")
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
