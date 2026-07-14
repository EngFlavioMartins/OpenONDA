"""Topology, geometry, and quality validation for FVM meshes."""

from __future__ import annotations

import numpy as np


class MeshValidationError(ValueError):
    """Raised when a mesh cannot safely enter the finite-volume solver."""


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

    return {
        "min_volume": float(np.min(volumes)),
        "max_volume": float(np.max(volumes)),
        "min_face_area": float(np.min(areas)),
        "max_non_orthogonality_deg": float(np.max(non_orthogonality)),
        "mean_non_orthogonality_deg": float(np.mean(non_orthogonality)),
        "out_of_bounds_interpolation_weights": out_of_bounds_weights,
    }


def validate_mesh(mesh_data, geo_data=None):
    """Validate topology and, when supplied, geometry; return one report."""
    report = validate_topology(mesh_data)
    if geo_data is not None:
        report.update(validate_geometry(mesh_data, geo_data))
    return report
