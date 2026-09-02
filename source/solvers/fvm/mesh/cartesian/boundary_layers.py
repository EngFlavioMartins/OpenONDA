# SPDX-License-Identifier: GPL-3.0-or-later
"""Planar patch-layer construction for the native Cartesian adapter.

Curved/non-planar boundary layers are deliberately rejected by
``CartesianMesher`` until a surface-first layer and transition-shell
algorithm is available.  This module therefore only serves exact planar
patches where the Cartesian interface topology is already authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import compute_mesh_geometry
from ..surface_classification import SurfaceIndex
from .config import BoundaryLayers


@dataclass(frozen=True, slots=True)
class LayerDiagnostics:
    """Summary of selected wall-normal layers."""

    patches: tuple[str, ...]
    requested_layers: int
    first_cell_height: float
    growth_ratio: float


def _area_vector(points: np.ndarray) -> np.ndarray:
    centre = points.mean(axis=0)
    result = np.zeros(3, dtype=np.float64)
    for index in range(len(points)):
        result += 0.5 * np.cross(
            points[index] - centre,
            points[(index + 1) % len(points)] - centre,
        )
    return result


def _oriented(face: np.ndarray, points: np.ndarray, centre: np.ndarray) -> np.ndarray:
    values = np.asarray(face, dtype=np.int32)
    if np.dot(_area_vector(points[values]), points[values].mean(axis=0) - centre) < 0.0:
        return values[::-1].copy()
    return values


def build_patch_layers(
    mesh_data: dict,
    surface_index: SurfaceIndex,
    layer: BoundaryLayers,
    wall_patch_name: str,
    interface_patch_name: str,
) -> dict:
    """Build hexahedral wall-normal layers over one exact planar wall patch.

    The input patch is the outer interface of the Cartesian core.  Its points
    are connected with a monotone geometric progression.  The interface
    points are deliberately retained as an explicit matching set so the
    native stitcher can make the core/layer face internal.  Curved surfaces
    must be rejected by the caller; mapping a Cartesian staircase here is not
    a conformal curved-surface algorithm.
    """
    layer_patch = next(
        (patch for patch in mesh_data["boundary"] if patch["name"] == wall_patch_name),
        None,
    )
    if layer_patch is None or int(layer_patch["n_faces"]) == 0:
        raise ValueError(f"Boundary-layer patch {wall_patch_name!r} has no faces")
    faces = np.asarray(mesh_data["faces"], dtype=np.int32)
    points = np.asarray(mesh_data["vertex_position"], dtype=np.float64)
    start = int(layer_patch["start_face"])
    stop = start + int(layer_patch["n_faces"])
    wall_faces = faces[start:stop]
    wall_points = np.unique(wall_faces)
    point_lookup = {int(point_id): index for index, point_id in enumerate(wall_points)}
    outer = points[wall_points]
    inner = np.empty_like(outer)
    for index, point in enumerate(outer):
        inner[index], _ = surface_index.nearest_point(point)
    path = np.linalg.norm(outer - inner, axis=1)
    if np.any(path <= np.finfo(np.float64).eps):
        raise ValueError(
            f"Boundary-layer patch {wall_patch_name!r} has zero wall-normal recovery distance"
        )

    cumulative = np.asarray(layer.layer_heights, dtype=np.float64).cumsum()
    fractions = np.concatenate(([0.0], cumulative / cumulative[-1]))
    local_faces = np.asarray(
        [[point_lookup[int(point)] for point in face] for face in wall_faces], dtype=np.int32
    )

    def face_area(face: np.ndarray) -> float:
        return float(np.linalg.norm(_area_vector(inner[face])))

    # Independent closest-point projection is ambiguous at a sharp junction:
    # several Cartesian corners may legitimately map to one triangulated
    # vertex.  A local tangent plane through the closest point of each face
    # centre retains every quad's parameterisation and avoids folded layers.
    candidate_sum: dict[int, np.ndarray] = {}
    candidate_count: dict[int, int] = {}
    for face in local_faces:
        outer_face = outer[face]
        normal = _area_vector(outer_face)
        normal_length = float(np.linalg.norm(normal))
        if normal_length <= np.finfo(np.float64).eps:
            raise ValueError(f"Boundary-layer patch {wall_patch_name!r} contains a zero-area face")
        normal /= normal_length
        centre = outer_face.mean(axis=0)
        target, _ = surface_index.nearest_point(centre)
        offset = float(np.dot(target - centre, normal))
        candidates = outer_face + offset * normal
        for point_id, candidate in zip(face, candidates, strict=True):
            value = int(point_id)
            candidate_sum[value] = candidate_sum.get(value, np.zeros(3)) + candidate
            candidate_count[value] = candidate_count.get(value, 0) + 1
    for point_id, total in candidate_sum.items():
        inner[point_id] = total / candidate_count[point_id]
    if any(face_area(face) <= 1.0e-14 for face in local_faces):
        raise ValueError(
            f"Boundary-layer patch {wall_patch_name!r} could not recover a non-degenerate wall"
        )
    path = np.linalg.norm(outer - inner, axis=1)
    radial = inner[None, :, :] + fractions[:, None, None] * (outer - inner)[None, :, :]

    layer_cells: list[np.ndarray] = []
    for face in local_faces:
        for radial_index in range(layer.layers):
            cell = np.asarray(
                (
                    radial_index * len(wall_points) + face[0],
                    radial_index * len(wall_points) + face[1],
                    radial_index * len(wall_points) + face[2],
                    radial_index * len(wall_points) + face[3],
                    (radial_index + 1) * len(wall_points) + face[0],
                    (radial_index + 1) * len(wall_points) + face[1],
                    (radial_index + 1) * len(wall_points) + face[2],
                    (radial_index + 1) * len(wall_points) + face[3],
                ),
                dtype=np.int32,
            )
            layer_cells.append(cell)

    layer_cells_array = np.asarray(layer_cells, dtype=np.int32)
    local_points = np.ascontiguousarray(radial.reshape(-1, 3))
    cell_centres = local_points[layer_cells_array].mean(axis=1)
    records: dict[tuple[int, ...], list[tuple[np.ndarray, int, str]]] = {}
    for cell_id, cell in enumerate(layer_cells_array):
        radial_index = cell_id % layer.layers
        candidates = (
            (cell[:4], "wall" if radial_index == 0 else "internal"),
            (cell[4:], "interface" if radial_index == layer.layers - 1 else "internal"),
            (cell[[0, 1, 5, 4]], "side"),
            (cell[[1, 2, 6, 5]], "side"),
            (cell[[2, 3, 7, 6]], "side"),
            (cell[[3, 0, 4, 7]], "side"),
        )
        for candidate, role in candidates:
            oriented = _oriented(candidate, local_points, cell_centres[cell_id])
            key = tuple(sorted(int(value) for value in oriented))
            records.setdefault(key, []).append((oriented, cell_id, role))

    internal_faces: list[np.ndarray] = []
    internal_owners: list[int] = []
    internal_neighbours: list[int] = []
    boundary_by_name: dict[str, list[np.ndarray]] = {
        wall_patch_name: [],
        interface_patch_name: [],
        "layer_termination": [],
    }
    boundary_owners: dict[str, list[int]] = {name: [] for name in boundary_by_name}
    for entries in records.values():
        if len(entries) == 2:
            first, second = entries
            face = first[0]
            direction = cell_centres[second[1]] - cell_centres[first[1]]
            if np.dot(_area_vector(local_points[face]), direction) < 0.0:
                face = face[::-1].copy()
            internal_faces.append(face)
            internal_owners.append(first[1])
            internal_neighbours.append(second[1])
        elif len(entries) == 1:
            face, owner, role = entries[0]
            name = (
                wall_patch_name
                if role == "wall"
                else interface_patch_name
                if role == "interface"
                else "layer_termination"
            )
            boundary_by_name[name].append(face)
            boundary_owners[name].append(owner)
        else:
            raise ValueError(
                f"Boundary-layer patch {wall_patch_name!r} has a non-manifold layer face"
            )

    face_blocks = [np.asarray(internal_faces, dtype=np.int32).reshape(-1, 4)]
    owner_blocks = [np.asarray(internal_owners, dtype=np.int32)]
    boundaries = []
    start_face = len(internal_faces)
    for name in (wall_patch_name, interface_patch_name, "layer_termination"):
        block = np.asarray(boundary_by_name[name], dtype=np.int32).reshape(-1, 4)
        owners = np.asarray(boundary_owners[name], dtype=np.int32)
        if not len(block):
            continue
        face_blocks.append(block)
        owner_blocks.append(owners)
        boundaries.append(
            {
                "name": name,
                "start_face": start_face,
                "n_faces": len(block),
                "type": "wall" if name == wall_patch_name else "patch",
            }
        )
        start_face += len(block)
    result = {
        "vertex_position": local_points,
        "faces": np.ascontiguousarray(np.vstack(face_blocks), dtype=np.int32),
        "owners": np.ascontiguousarray(np.concatenate(owner_blocks), dtype=np.int32),
        "neighbours": np.asarray(internal_neighbours, dtype=np.int32),
        "boundary": boundaries,
        "n_cells": len(layer_cells_array),
        "n_faces": start_face,
        "n_interior_faces": len(internal_faces),
        "n_points": len(local_points),
        "cell_vertex_indices": layer_cells_array,
        "cell_type_code": np.full(len(layer_cells_array), 5, dtype=np.int32),
        "cell_levels": np.zeros(len(layer_cells_array), dtype=np.int8),
        "cell_sizes": np.asarray(
            [layer.layer_heights[index % layer.layers] for index in range(len(layer_cells_array))],
            dtype=np.float32,
        ),
        "boundary_layer_index": np.asarray(
            [index % layer.layers for index in range(len(layer_cells_array))], dtype=np.int16
        ),
        "interface_point_ids": np.arange(
            layer.layers * len(wall_points),
            (layer.layers + 1) * len(wall_points),
            dtype=np.int32,
        ),
        "mesh_generation": {
            "method": "patch_normal_layers",
            "patch": wall_patch_name,
            "layers": layer.layers,
            "first_cell_height": layer.first_cell_height,
            "growth_ratio": layer.growth_ratio,
            "requested_layer_heights": layer.layer_heights,
            "wall_normal_distance_min": float(np.min(path)),
            "wall_normal_distance_max": float(np.max(path)),
        },
    }
    for _ in range(3):
        geometry = compute_mesh_geometry(result, compute_lsq=False)
        owners = np.asarray(result["owners"], dtype=np.int64)
        area_vectors = np.asarray(geometry["face_area_vector"])
        face_centres = np.asarray(geometry["face_centre"])
        cell_centres = np.asarray(geometry["cell_centre"])
        directions = np.empty_like(area_vectors)
        internal_count = int(result["n_interior_faces"])
        directions[:internal_count] = (
            cell_centres[np.asarray(result["neighbours"], dtype=np.int64)]
            - cell_centres[owners[:internal_count]]
        )
        directions[internal_count:] = (
            face_centres[internal_count:] - cell_centres[owners[internal_count:]]
        )
        reverse = np.einsum("ij,ij->i", area_vectors, directions) < 0.0
        if not np.any(reverse):
            break
        result["faces"][reverse] = result["faces"][reverse, ::-1]
    return result


__all__ = ["LayerDiagnostics", "build_patch_layers"]
