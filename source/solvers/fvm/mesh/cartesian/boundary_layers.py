# SPDX-License-Identifier: GPL-3.0-or-later
"""Planar patch-layer construction for the native Cartesian adapter.

Curved/non-planar boundary layers are deliberately rejected by
``CartesianMesher`` until a surface-first layer and transition-shell
algorithm is available.  This module therefore only serves exact planar
patches where the Cartesian interface topology is already authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..geometry import compute_mesh_geometry
from ..surface_classification import SurfaceIndex
from .config import BoundaryLayers
from .surface_recovery import _merge_coplanar_surface_fragments


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


@dataclass(frozen=True, slots=True)
class LayerSurface:
    """Original and cumulative offset triangles for one selected patch."""

    patch: str
    distances: np.ndarray
    triangles: tuple[np.ndarray, ...]

    @property
    def outer_triangles(self) -> np.ndarray:
        """Triangulated interface between the layer block and Cartesian core."""
        return self.triangles[-1]


def _optimise_coplanar_quad_diagonals(
    triangles: np.ndarray,
    feature_angle_degrees: float,
) -> np.ndarray:
    """Flip planar quad diagonals to align centroids across smooth edges.

    STL exporters commonly split every surface quad with the same diagonal.
    In thin extruded layers that pattern places adjacent prism centroids at
    opposite ends of a long shared edge and produces a nearly non-orthogonal
    internal face.  Flipping a diagonal inside a planar convex quad preserves
    the exact input surface.  The deterministic local objective below reduces
    centroid displacement *along* neighboring smooth edges.

    Only triangle pairs for which both triangles have exactly one coplanar
    neighbour are considered.  This recognises isolated quad splits without
    rewriting general planar triangulations such as cap fans.
    """
    source = np.ascontiguousarray(triangles, dtype=np.float64)
    scale = max(float(np.ptp(source, axis=(0, 1)).max()), 1.0)
    tolerance = max(1.0e-11 * scale, np.finfo(np.float64).eps * scale * 64.0)
    flat = source.reshape(-1, 3)
    keys = np.rint(flat / tolerance).astype(np.int64)
    _unique, first, inverse = np.unique(
        keys, axis=0, return_index=True, return_inverse=True
    )
    points = flat[first]
    faces = inverse.reshape(-1, 3)
    raw_normals = np.cross(source[:, 1] - source[:, 0], source[:, 2] - source[:, 0])
    normal_lengths = np.linalg.norm(raw_normals, axis=1)
    normals = raw_normals / normal_lengths[:, None]
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face_id, face in enumerate(faces):
        for first_id, second_id in ((0, 1), (1, 2), (2, 0)):
            edge = tuple(sorted((int(face[first_id]), int(face[second_id]))))
            edge_faces.setdefault(edge, []).append(face_id)
    coplanar_neighbours = np.zeros(len(faces), dtype=np.int16)
    coplanar_edges: list[tuple[tuple[int, int], int, int]] = []
    for edge, adjacent in edge_faces.items():
        if len(adjacent) != 2:
            continue
        first_face, second_face = adjacent
        if float(np.dot(normals[first_face], normals[second_face])) < 1.0 - 1.0e-10:
            continue
        plane_error = float(
            np.max(
                np.abs(
                    (source[second_face] - source[first_face, 0])
                    @ normals[first_face]
                )
            )
        )
        if plane_error > tolerance:
            continue
        coplanar_neighbours[first_face] += 1
        coplanar_neighbours[second_face] += 1
        coplanar_edges.append((edge, first_face, second_face))

    panels: list[dict[str, Any]] = []
    occupied: set[int] = set()
    for diagonal, first_face, second_face in coplanar_edges:
        if (
            coplanar_neighbours[first_face] != 1
            or coplanar_neighbours[second_face] != 1
            or first_face in occupied
            or second_face in occupied
        ):
            continue
        opposite_first = next(
            int(vertex) for vertex in faces[first_face] if int(vertex) not in diagonal
        )
        opposite_second = next(
            int(vertex) for vertex in faces[second_face] if int(vertex) not in diagonal
        )
        if opposite_first == opposite_second:
            continue
        target_normal = normals[first_face]
        alternative = np.asarray(
            (
                (opposite_first, opposite_second, diagonal[0]),
                (opposite_first, diagonal[1], opposite_second),
            ),
            dtype=np.int64,
        )
        for triangle_id in range(2):
            coordinates = points[alternative[triangle_id]]
            normal = np.cross(
                coordinates[1] - coordinates[0], coordinates[2] - coordinates[0]
            )
            if float(np.dot(normal, target_normal)) < 0.0:
                alternative[triangle_id, 1:] = alternative[triangle_id, :0:-1]
        options = (
            np.asarray((faces[first_face], faces[second_face]), dtype=np.int64),
            alternative,
        )
        boundary_centres: list[dict[tuple[int, int], np.ndarray]] = []
        for option in options:
            option_edges: dict[tuple[int, int], list[int]] = {}
            for local_face, face in enumerate(option):
                for first_id, second_id in ((0, 1), (1, 2), (2, 0)):
                    edge = tuple(sorted((int(face[first_id]), int(face[second_id]))))
                    option_edges.setdefault(edge, []).append(local_face)
            boundary_centres.append(
                {
                    edge: points[option[local_faces[0]]].mean(axis=0)
                    for edge, local_faces in option_edges.items()
                    if len(local_faces) == 1
                }
            )
        panels.append(
            {
                "faces": (first_face, second_face),
                "normal": target_normal,
                "options": options,
                "boundary_centres": tuple(boundary_centres),
            }
        )
        occupied.update((first_face, second_face))
    if len(panels) < 2:
        return source

    edge_panels: dict[tuple[int, int], list[int]] = {}
    for panel_id, panel in enumerate(panels):
        for edge in panel["boundary_centres"][0]:
            edge_panels.setdefault(edge, []).append(panel_id)
    smooth_cosine = float(np.cos(np.deg2rad(feature_angle_degrees)))
    connections: list[tuple[int, int, tuple[int, int]]] = []
    for edge, adjacent in edge_panels.items():
        if len(adjacent) != 2:
            continue
        first_panel, second_panel = adjacent
        if (
            float(np.dot(panels[first_panel]["normal"], panels[second_panel]["normal"]))
            >= smooth_cosine
        ):
            connections.append((first_panel, second_panel, edge))
    if not connections:
        return source

    incident: list[list[tuple[int, int, tuple[int, int]]]] = [
        [] for _ in panels
    ]
    for connection in connections:
        incident[connection[0]].append(connection)
        incident[connection[1]].append(connection)
    choices = np.zeros(len(panels), dtype=np.int8)

    def edge_cost(
        first_panel: int,
        first_choice: int,
        second_panel: int,
        second_choice: int,
        edge: tuple[int, int],
    ) -> float:
        direction = points[edge[1]] - points[edge[0]]
        direction /= np.linalg.norm(direction)
        first_centre = panels[first_panel]["boundary_centres"][first_choice][edge]
        second_centre = panels[second_panel]["boundary_centres"][second_choice][edge]
        return abs(float(np.dot(second_centre - first_centre, direction)))

    for _iteration in range(max(2, 2 * len(panels))):
        changed = False
        for panel_id in range(len(panels)):
            current = 0.0
            flipped = 0.0
            for first_panel, second_panel, edge in incident[panel_id]:
                other = second_panel if first_panel == panel_id else first_panel
                current += edge_cost(
                    panel_id, int(choices[panel_id]), other, int(choices[other]), edge
                )
                flipped += edge_cost(
                    panel_id, 1 - int(choices[panel_id]), other, int(choices[other]), edge
                )
            if flipped + tolerance < current:
                choices[panel_id] = 1 - choices[panel_id]
                changed = True
        if not changed:
            break

    optimised_faces = faces.copy()
    for panel_id, choice in enumerate(choices):
        if not choice:
            continue
        face_ids = panels[panel_id]["faces"]
        optimised_faces[np.asarray(face_ids)] = panels[panel_id]["options"][1]
    return np.ascontiguousarray(points[optimised_faces], dtype=np.float64)


def _refine_surface_triangles(
    triangles: np.ndarray,
    maximum_edge_length: float,
    *,
    maximum_iterations: int = 12,
) -> np.ndarray:
    """Conformingly bisect long STL edges while retaining source facets."""
    current = np.ascontiguousarray(triangles, dtype=np.float64)
    scale = max(float(np.ptp(current, axis=(0, 1)).max()), 1.0)
    tolerance = max(1.0e-11 * scale, np.finfo(np.float64).eps * scale * 64.0)
    for _iteration in range(maximum_iterations):
        flat = current.reshape(-1, 3)
        keys = np.rint(flat / tolerance).astype(np.int64)
        _unique, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
        vertices = [point.copy() for point in flat[first]]
        faces = inverse.reshape(-1, 3)
        edge_lengths: dict[tuple[int, int], float] = {}
        for face in faces:
            for first_id, second_id in ((0, 1), (1, 2), (2, 0)):
                a = int(face[first_id])
                b = int(face[second_id])
                edge = (min(a, b), max(a, b))
                if edge not in edge_lengths:
                    edge_lengths[edge] = float(
                        np.linalg.norm(vertices[edge[1]] - vertices[edge[0]])
                    )
        marked = {
            edge for edge, length in edge_lengths.items() if length > maximum_edge_length
        }
        if not marked:
            return current
        midpoint: dict[tuple[int, int], int] = {}
        for edge in sorted(marked):
            midpoint[edge] = len(vertices)
            vertices.append(0.5 * (vertices[edge[0]] + vertices[edge[1]]))
        refined: list[tuple[int, int, int]] = []
        for a_raw, b_raw, c_raw in faces:
            a, b, c = int(a_raw), int(b_raw), int(c_raw)
            ab = midpoint.get((min(a, b), max(a, b)))
            bc = midpoint.get((min(b, c), max(b, c)))
            ca = midpoint.get((min(c, a), max(c, a)))
            count = sum(value is not None for value in (ab, bc, ca))
            if count == 0:
                refined.append((a, b, c))
            elif count == 1 and ab is not None:
                refined.extend(((a, ab, c), (ab, b, c)))
            elif count == 1 and bc is not None:
                refined.extend(((b, bc, a), (bc, c, a)))
            elif count == 1 and ca is not None:
                refined.extend(((c, ca, b), (ca, a, b)))
            elif count == 2 and ab is not None and bc is not None:
                refined.extend(((b, bc, ab), (a, ab, c), (ab, bc, c)))
            elif count == 2 and bc is not None and ca is not None:
                refined.extend(((c, ca, bc), (b, bc, a), (bc, ca, a)))
            elif count == 2 and ca is not None and ab is not None:
                refined.extend(((a, ab, ca), (c, ca, b), (ca, ab, b)))
            elif ab is not None and bc is not None and ca is not None:
                refined.extend(
                    ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))
                )
            else:
                raise RuntimeError("Unhandled conforming surface-refinement pattern")
        vertex_array = np.asarray(vertices, dtype=np.float64)
        current = np.ascontiguousarray(vertex_array[np.asarray(refined, dtype=np.int64)])
    raise ValueError(
        "Surface subdivision did not reach the requested edge length within "
        f"{maximum_iterations} iterations"
    )


def build_layer_surface(
    triangles: np.ndarray,
    patch: str,
    layer: BoundaryLayers,
    *,
    feature_angle_degrees: float = 45.0,
    target_edge_length: float | None = None,
) -> LayerSurface:
    """Create feature-aware cumulative offsets from an arbitrary STL patch.

    Corner normals are smoothed only across face pairs below the feature
    angle. Sharp edges therefore terminate cleanly instead of being rounded
    by unrelated face normals.
    """
    source = _optimise_coplanar_quad_diagonals(
        np.ascontiguousarray(triangles, dtype=np.float64),
        feature_angle_degrees,
    )
    if target_edge_length is not None:
        if not np.isfinite(target_edge_length) or target_edge_length <= 0.0:
            raise ValueError("target_edge_length must be finite and positive")
        source = _refine_surface_triangles(source, target_edge_length)
    scale = max(float(np.ptp(source, axis=(0, 1)).max()), 1.0)
    tolerance = max(1.0e-11 * scale, np.finfo(np.float64).eps * scale * 64.0)
    flat = source.reshape(-1, 3)
    keys = np.rint(flat / tolerance).astype(np.int64)
    _unique_keys, first, inverse = np.unique(
        keys, axis=0, return_index=True, return_inverse=True
    )
    vertices = flat[first]
    face_vertices = inverse.reshape(-1, 3)
    raw_normals = np.cross(source[:, 1] - source[:, 0], source[:, 2] - source[:, 0])
    lengths = np.linalg.norm(raw_normals, axis=1)
    if np.any(lengths <= tolerance**2):
        raise ValueError(f"Boundary-layer patch {patch!r} contains a degenerate triangle")
    signed_volume = float(
        np.einsum("ij,ij->i", source[:, 0], np.cross(source[:, 1], source[:, 2])).sum()
        / 6.0
    )
    outward = raw_normals / lengths[:, None]
    if signed_volume < 0.0:
        outward *= -1.0
    incident: list[list[int]] = [[] for _ in range(len(vertices))]
    for face_id, vertex_ids in enumerate(face_vertices):
        for vertex_id in vertex_ids:
            incident[int(vertex_id)].append(face_id)
    cosine = float(np.cos(np.deg2rad(feature_angle_degrees)))
    vertex_offsets = np.empty_like(vertices)
    for vertex_id, adjacent_faces in enumerate(incident):
        candidates = np.asarray(adjacent_faces, dtype=np.int64)
        candidate_normals = outward[candidates]
        pairwise_alignment = candidate_normals @ candidate_normals.T
        if float(np.min(pairwise_alignment)) >= cosine:
            # Smooth vertices follow one area-weighted unit normal.
            weighted = (candidate_normals * lengths[candidates, None]).sum(axis=0)
            norm = float(np.linalg.norm(weighted))
            if norm <= tolerance:
                raise ValueError(
                    f"Boundary-layer patch {patch!r} has an undefined smooth normal"
                )
            vertex_offsets[vertex_id] = weighted / norm
            continue

        # A sharp edge or corner still needs one shared offset point.  The
        # miter constraints n_i . d = 1 preserve the requested normal height
        # on every incident feature face and avoid cracks between separately
        # extruded triangles.  The minimum-norm solution is deterministic for
        # rank-two edges as well as full-rank corners.
        weights = np.sqrt(lengths[candidates])
        matrix = candidate_normals * weights[:, None]
        right_hand_side = weights
        offset, _residuals, rank, _singular_values = np.linalg.lstsq(
            matrix, right_hand_side, rcond=None
        )
        normal_heights = candidate_normals @ offset
        if (
            rank == 0
            or not np.all(np.isfinite(offset))
            or np.min(normal_heights) <= 0.0
            or np.max(np.abs(normal_heights - 1.0)) > 0.15
        ):
            raise ValueError(
                f"Boundary-layer patch {patch!r} has incompatible normals at "
                f"surface vertex {vertex_id}"
            )
        vertex_offsets[vertex_id] = offset
    corner_normals = vertex_offsets[face_vertices]

    distances = np.concatenate(
        (np.asarray([0.0]), np.cumsum(np.asarray(layer.layer_heights, dtype=np.float64)))
    )
    levels = tuple(
        np.ascontiguousarray(source + distance * corner_normals)
        for distance in distances
    )
    for level, current in enumerate(levels):
        current_lengths = np.linalg.norm(
            np.cross(current[:, 1] - current[:, 0], current[:, 2] - current[:, 0]), axis=1
        )
        if np.any(current_lengths <= tolerance**2):
            raise ValueError(
                f"Boundary-layer patch {patch!r} collapses at cumulative level {level}"
            )
    distances.setflags(write=False)
    for values in levels:
        values.setflags(write=False)
    return LayerSurface(patch=patch, distances=distances, triangles=levels)


def _barycentric_coordinates(points: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    a, b, c = triangle
    v0 = b - a
    v1 = c - a
    v2 = points - a
    d00 = float(np.dot(v0, v0))
    d01 = float(np.dot(v0, v1))
    d11 = float(np.dot(v1, v1))
    denominator = d00 * d11 - d01 * d01
    if abs(denominator) <= np.finfo(np.float64).eps:
        raise ValueError("Offset layer surface contains a degenerate triangle")
    d20 = np.einsum("ij,j->i", v2, v0)
    d21 = np.einsum("ij,j->i", v2, v1)
    v = (d11 * d20 - d01 * d21) / denominator
    w = (d00 * d21 - d01 * d20) / denominator
    return np.column_stack((1.0 - v - w, v, w))


def _source_triangle_for_face(
    face_points: np.ndarray,
    surface: LayerSurface,
    index: SurfaceIndex,
    tolerance: float,
) -> tuple[int, np.ndarray]:
    candidates = index.candidate_triangles(
        face_points.min(axis=0) - tolerance,
        face_points.max(axis=0) + tolerance,
    )
    best: tuple[float, int, np.ndarray] | None = None
    for triangle_id in candidates:
        triangle = surface.outer_triangles[int(triangle_id)]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        normal /= np.linalg.norm(normal)
        plane_error = float(np.max(np.abs((face_points - triangle[0]) @ normal)))
        barycentric = _barycentric_coordinates(face_points, triangle)
        range_error = float(
            max(
                0.0,
                -float(barycentric.min()),
                float(barycentric.max()) - 1.0,
            )
        )
        score = plane_error + range_error
        if best is None or score < best[0]:
            best = (score, int(triangle_id), barycentric)
    if best is None or best[0] > 100.0 * tolerance:
        raise ValueError("Layer/core interface face cannot be mapped to one offset STL triangle")
    return best[1], best[2]


def _map_offset_point_to_source(
    point: np.ndarray,
    surface: LayerSurface,
    index: SurfaceIndex,
    tolerance: float,
    preferred_normal: np.ndarray,
) -> np.ndarray:
    candidates = index.candidate_triangles(point - tolerance, point + tolerance)
    best: tuple[float, float, int, np.ndarray] | None = None
    for triangle_id in candidates:
        triangle = surface.outer_triangles[int(triangle_id)]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        normal /= np.linalg.norm(normal)
        alignment_error = 1.0 - abs(float(np.dot(normal, preferred_normal)))
        plane_error = abs(float(np.dot(point - triangle[0], normal)))
        barycentric = _barycentric_coordinates(point[None, :], triangle)[0]
        range_error = max(
            0.0,
            -float(barycentric.min()),
            float(barycentric.max()) - 1.0,
        )
        geometry_error = plane_error + range_error
        if geometry_error > 100.0 * tolerance:
            continue
        candidate = (alignment_error, geometry_error, int(triangle_id), barycentric)
        if best is None or candidate[:3] < best[:3]:
            best = candidate
    if best is None:
        raise ValueError("Layer/core interface vertex cannot be mapped to the offset STL")
    return best[3] @ surface.triangles[0][best[2]]


class _CoordinateRegistry:
    def __init__(self, points: np.ndarray, tolerance: float) -> None:
        self.tolerance = tolerance
        self.points = [np.asarray(point, dtype=np.float64).copy() for point in points]
        self.lookup = {
            tuple(np.rint(point / tolerance).astype(np.int64)): point_id
            for point_id, point in enumerate(points)
        }

    def ids(self, coordinates: np.ndarray) -> np.ndarray:
        ids = np.empty(len(coordinates), dtype=np.int32)
        for local_id, point in enumerate(coordinates):
            key = tuple(np.rint(point / self.tolerance).astype(np.int64))
            point_id = self.lookup.get(key)
            if point_id is None:
                point_id = len(self.points)
                self.lookup[key] = point_id
                self.points.append(np.asarray(point, dtype=np.float64).copy())
            ids[local_id] = point_id
        return ids


def insert_surface_layers(
    core: dict[str, Any],
    layer_surfaces: tuple[LayerSurface, ...],
    layer_specs: tuple[BoundaryLayers, ...],
    domain_bounds: tuple[float, float, float, float, float, float],
    domain_patch_names: tuple[str, str, str, str, str, str],
) -> dict[str, Any]:
    """Split recovered wall columns into native polygonal boundary layers."""
    if len(layer_surfaces) != len(layer_specs):
        raise ValueError("Layer surfaces and controls must have equal length")
    core_points = np.asarray(core["vertex_position"], dtype=np.float64)
    scale = max(float(np.ptp(core_points, axis=0).max()), 1.0)
    tolerance = max(2.0e-10 * scale, np.finfo(np.float64).eps * scale * 128.0)
    registry = _CoordinateRegistry(core_points, tolerance)
    surface_by_patch = {surface.patch: surface for surface in layer_surfaces}
    spec_by_patch = {
        patch: spec
        for spec in layer_specs
        for patch in spec.patches
        if patch in surface_by_patch
    }
    if set(surface_by_patch) != set(spec_by_patch):
        raise ValueError("Every offset layer surface must have one boundary-layer control")
    indices = {
        patch: SurfaceIndex.build(surface.outer_triangles)
        for patch, surface in surface_by_patch.items()
    }
    original_indices = {
        patch: SurfaceIndex.build(surface.triangles[0])
        for patch, surface in surface_by_patch.items()
    }

    interface_records: dict[tuple[int, ...], tuple[np.ndarray, int]] = {}
    layer_cells: list[list[tuple[np.ndarray, str]]] = []
    layer_cell_index: list[int] = []
    layer_cell_size: list[float] = []
    layer_group_keys: list[tuple[str, int, tuple[int, int, int, int], int]] = []
    measured_height_min = {
        patch: np.full(spec_by_patch[patch].layers, np.inf, dtype=np.float64)
        for patch in surface_by_patch
    }
    measured_height_max = {
        patch: np.zeros(spec_by_patch[patch].layers, dtype=np.float64)
        for patch in surface_by_patch
    }
    for patch in core["boundary"]:
        patch_name = str(patch["name"])
        if patch_name not in surface_by_patch:
            continue
        start = int(patch["start_face"])
        stop = start + int(patch["n_faces"])
        surface = surface_by_patch[patch_name]
        spec = spec_by_patch[patch_name]
        for face_id in range(start, stop):
            core_owner = int(core["owners"][face_id])
            outer_face = np.asarray(core["faces"][face_id], dtype=np.int32)
            outer_points = core_points[outer_face]
            facet_normal = _area_vector(outer_points)
            facet_normal /= np.linalg.norm(facet_normal)
            inner_points = np.asarray(
                [
                    _map_offset_point_to_source(
                        point,
                        surface,
                        indices[patch_name],
                        tolerance,
                        facet_normal,
                    )
                    for point in outer_points
                ],
                dtype=np.float64,
            )
            facet_plane = float(np.dot(facet_normal, outer_points[0]))
            facet_quantum = 1.0e-8
            facet_group = tuple(
                np.rint(
                    np.concatenate((facet_normal, np.asarray([facet_plane])))
                    / facet_quantum
                ).astype(np.int64)
            )
            inner_keys = np.rint(inner_points / tolerance).astype(np.int64)
            _unique_inner, first_inner = np.unique(
                inner_keys, axis=0, return_index=True
            )
            if len(first_inner) != len(inner_points):
                keep = np.sort(first_inner)
                if len(keep) < 3:
                    raise ValueError(
                        f"Boundary-layer patch {patch_name!r} collapses an interface polygon"
                    )
                outer_face = outer_face[keep]
                outer_points = outer_points[keep]
                inner_points = inner_points[keep]
            paths = outer_points - inner_points
            path_lengths = np.linalg.norm(paths, axis=1)
            if np.any(path_lengths <= tolerance):
                raise ValueError(
                    f"Boundary-layer patch {patch_name!r} has a collapsed wall-normal column"
                )
            fractions = surface.distances / surface.distances[-1]
            normal_path_lengths = np.abs(paths @ facet_normal)
            if np.any(normal_path_lengths <= tolerance):
                raise ValueError(
                    f"Boundary-layer patch {patch_name!r} has a tangential layer column"
                )
            height_samples = (
                np.diff(fractions)[:, None] * normal_path_lengths[None, :]
            )
            measured_height_min[patch_name] = np.minimum(
                measured_height_min[patch_name], height_samples.min(axis=1)
            )
            measured_height_max[patch_name] = np.maximum(
                measured_height_max[patch_name], height_samples.max(axis=1)
            )
            rings = [
                registry.ids(inner_points + fraction * paths)
                for fraction in fractions
            ]
            for layer_index in range(spec.layers):
                lower = rings[layer_index]
                upper = rings[layer_index + 1]
                entries: list[tuple[np.ndarray, str]] = [
                    (
                        lower[::-1].copy(),
                        "wall" if layer_index == 0 else "internal",
                    ),
                    (
                        upper.copy(),
                        "interface" if layer_index == spec.layers - 1 else "internal",
                    ),
                ]
                for edge in range(len(lower)):
                    following = (edge + 1) % len(lower)
                    entries.append(
                        (
                            np.asarray(
                                (lower[edge], lower[following], upper[following], upper[edge]),
                                dtype=np.int32,
                            ),
                            "side",
                        )
                    )
                layer_cells.append(entries)
                layer_group_keys.append(
                    (patch_name, core_owner, facet_group, layer_index)
                )
                layer_cell_index.append(layer_index)
                layer_cell_size.append(spec.layer_heights[layer_index])
            signature = tuple(sorted(map(int, rings[-1])))
            if signature in interface_records:
                raise ValueError("Layer/core interface contains a duplicate face")
            interface_records[signature] = (outer_face, core_owner)

    if not layer_cells:
        raise ValueError("Boundary-layer controls selected no recovered wall faces")

    # Co-planar STL triangles frequently describe one logical surface facet.
    # Merge only columns that are *actually connected by an identical face*.
    # Grouping by owner/facet alone can accidentally join disconnected clipped
    # fragments into a non-convex cell, which destroys skewness even though the
    # face count looks attractive.
    parent = list(range(len(layer_cells)))

    def find(cell: int) -> int:
        while parent[cell] != cell:
            parent[cell] = parent[parent[cell]]
            cell = parent[cell]
        return cell

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    connected_faces: dict[tuple[int, ...], list[int]] = {}
    for local_cell, entries in enumerate(layer_cells):
        for face, _role in entries:
            signature = tuple(sorted(map(int, face)))
            connected_faces.setdefault(signature, []).append(local_cell)
    for cells in connected_faces.values():
        unique_cells = tuple(dict.fromkeys(cells))
        if len(unique_cells) != 2:
            continue
        first, second = unique_cells
        if layer_group_keys[first] == layer_group_keys[second]:
            union(first, second)

    grouped_cells: list[list[tuple[np.ndarray, str]]] = []
    grouped_indices: list[int] = []
    grouped_sizes: list[float] = []
    grouped_ids: dict[int, int] = {}
    for local_cell, entries in enumerate(layer_cells):
        root = find(local_cell)
        grouped_id = grouped_ids.get(root)
        if grouped_id is None:
            grouped_id = len(grouped_cells)
            grouped_ids[root] = grouped_id
            grouped_cells.append([])
            grouped_indices.append(layer_cell_index[local_cell])
            grouped_sizes.append(layer_cell_size[local_cell])
        grouped_cells[grouped_id].extend(entries)
    layer_cells = grouped_cells
    layer_cell_index = grouped_indices
    layer_cell_size = grouped_sizes

    # Present each agglomerated polyhedron with one polygon per connected
    # coplanar wall/radial surface.  Retaining the original STL subfaces here
    # gives a valid volume but makes a tiny subface appear severely skewed
    # relative to the centroid of the whole cell.
    layer_points = np.asarray(registry.points, dtype=np.float64)
    for local_cell, entries in enumerate(layer_cells):
        retained = [(face, role) for face, role in entries if role not in {"wall", "internal"}]
        for role in ("wall", "internal"):
            polygons = [
                layer_points[np.asarray(face, dtype=np.int64)]
                for face, entry_role in entries
                if entry_role == role
            ]
            if not polygons:
                continue
            retained.extend(
                (registry.ids(polygon), role)
                for polygon in _merge_coplanar_surface_fragments(polygons, tolerance)
            )
        layer_cells[local_cell] = retained

    layer_offset = int(core["n_cells"])
    records: dict[tuple[int, ...], list[tuple[np.ndarray, int, str]]] = {}
    for local_cell, entries in enumerate(layer_cells):
        for face, role in entries:
            signature = tuple(sorted(map(int, face)))
            records.setdefault(signature, []).append((face, local_cell, role))

    layer_internal_faces: list[np.ndarray] = []
    layer_internal_owners: list[int] = []
    layer_internal_neighbours: list[int] = []
    wall_faces: dict[str, list[np.ndarray]] = {patch: [] for patch in surface_by_patch}
    wall_owners: dict[str, list[int]] = {patch: [] for patch in surface_by_patch}
    side_faces: dict[str, list[np.ndarray]] = {}
    side_owners: dict[str, list[int]] = {}
    interface_neighbours: dict[tuple[int, ...], int] = {}
    bounds = np.asarray(domain_bounds, dtype=np.float64)
    all_points = np.asarray(registry.points, dtype=np.float64)

    def side_patch(face: np.ndarray) -> str:
        coordinates = all_points[np.asarray(face, dtype=np.int64)]
        for side, patch_name in enumerate(domain_patch_names):
            axis = side // 2
            value = bounds[side]
            if np.all(np.abs(coordinates[:, axis] - value) <= tolerance):
                return patch_name
        return "layer_termination"

    for signature, entries in records.items():
        entry_cells = {entry[1] for entry in entries}
        if len(entry_cells) == 1 and len(entries) > 1:
            # STL-fragment and Cartesian-clip subdivision faces internal to
            # one agglomerated cut-cell column disappear from its boundary.
            continue
        if len(entries) == 2:
            first, second = entries
            layer_internal_faces.append(first[0])
            layer_internal_owners.append(first[1] + layer_offset)
            layer_internal_neighbours.append(second[1] + layer_offset)
            continue
        if len(entries) != 1:
            raise ValueError("Boundary-layer extrusion produced a non-manifold face")
        face, local_cell, role = entries[0]
        owner = local_cell + layer_offset
        if role == "interface":
            if signature in interface_neighbours:
                raise ValueError("Boundary-layer interface contains a duplicate column")
            interface_neighbours[signature] = owner
        elif role == "wall":
            centre = all_points[np.asarray(face, dtype=np.int64)].mean(axis=0)
            patch_name = min(
                surface_by_patch,
                key=lambda name: original_indices[name].nearest_point(centre)[1],
            )
            wall_faces[patch_name].append(face)
            wall_owners[patch_name].append(owner)
        else:
            patch_name = side_patch(face)
            side_faces.setdefault(patch_name, []).append(face)
            side_owners.setdefault(patch_name, []).append(owner)

    if set(interface_records) != set(interface_neighbours):
        missing_core = len(set(interface_neighbours) - set(interface_records))
        missing_layer = len(set(interface_records) - set(interface_neighbours))
        raise ValueError(
            "Layer/core interface is not conformal: "
            f"unmatched_core={missing_core}, unmatched_layer={missing_layer}"
        )

    core_internal = int(core["n_interior_faces"])
    combined_faces: list[np.ndarray] = [
        np.asarray(face, dtype=np.int32) for face in core["faces"][:core_internal]
    ]
    combined_owners = list(map(int, np.asarray(core["owners"][:core_internal])))
    combined_neighbours = list(map(int, np.asarray(core["neighbours"])))
    combined_faces.extend(layer_internal_faces)
    combined_owners.extend(layer_internal_owners)
    combined_neighbours.extend(layer_internal_neighbours)
    for signature, (face, owner) in interface_records.items():
        combined_faces.append(np.asarray(face, dtype=np.int32))
        combined_owners.append(owner)
        combined_neighbours.append(interface_neighbours[signature])
    n_internal = len(combined_faces)

    patch_order: list[str] = []
    patch_faces: dict[str, list[np.ndarray]] = {}
    patch_owners: dict[str, list[int]] = {}
    patch_types: dict[str, str] = {}

    def ensure(name: str, patch_type: str) -> None:
        if name not in patch_faces:
            patch_order.append(name)
            patch_faces[name] = []
            patch_owners[name] = []
            patch_types[name] = patch_type

    for patch in core["boundary"]:
        name = str(patch["name"])
        if name in surface_by_patch:
            continue
        ensure(name, str(patch.get("type", "patch")))
        start = int(patch["start_face"])
        stop = start + int(patch["n_faces"])
        patch_faces[name].extend(
            np.asarray(face, dtype=np.int32) for face in core["faces"][start:stop]
        )
        patch_owners[name].extend(map(int, np.asarray(core["owners"][start:stop])))
    for name in surface_by_patch:
        ensure(name, "wall")
        patch_faces[name].extend(wall_faces[name])
        patch_owners[name].extend(wall_owners[name])
    for name, faces in side_faces.items():
        ensure(name, "wall" if name == "layer_termination" else "patch")
        patch_faces[name].extend(faces)
        patch_owners[name].extend(side_owners[name])

    boundary: list[dict[str, Any]] = []
    start_face = n_internal
    for name in patch_order:
        combined_faces.extend(patch_faces[name])
        combined_owners.extend(patch_owners[name])
        boundary.append(
            {
                "name": name,
                "start_face": start_face,
                "n_faces": len(patch_faces[name]),
                "type": patch_types[name],
            }
        )
        start_face += len(patch_faces[name])

    widths = {len(face) for face in combined_faces}
    result = {
        "vertex_position": np.ascontiguousarray(registry.points, dtype=np.float64),
        "faces": (
            np.ascontiguousarray(combined_faces, dtype=np.int32)
            if len(widths) == 1
            else combined_faces
        ),
        "owners": np.ascontiguousarray(combined_owners, dtype=np.int32),
        "neighbours": np.ascontiguousarray(combined_neighbours, dtype=np.int32),
        "boundary": boundary,
        "n_cells": int(core["n_cells"]) + len(layer_cells),
        "n_faces": len(combined_faces),
        "n_interior_faces": n_internal,
        "n_points": len(registry.points),
        "cell_levels": np.concatenate(
            (
                np.asarray(core["cell_levels"], dtype=np.int8),
                np.full(len(layer_cells), np.max(core["cell_levels"]), dtype=np.int8),
            )
        ),
        "cell_sizes": np.concatenate(
            (
                np.asarray(core["cell_sizes"], dtype=np.float32),
                np.asarray(layer_cell_size, dtype=np.float32),
            )
        ),
        "boundary_layer_index": np.concatenate(
            (
                np.full(int(core["n_cells"]), -1, dtype=np.int16),
                np.asarray(layer_cell_index, dtype=np.int16),
            )
        ),
        "mesh_generation": dict(core["mesh_generation"]),
    }
    result["mesh_generation"]["boundary_layers"] = {
        "method": "recovered_surface_cell_splitting",
        "patches": tuple(surface_by_patch),
        "layer_cells": len(layer_cells),
        "layer_cells_by_index": tuple(
            int(np.count_nonzero(np.asarray(layer_cell_index) == layer_index))
            for layer_index in range(max(layer_cell_index) + 1)
        ),
        "termination_faces": len(side_faces.get("layer_termination", ())),
        "core_interface_faces": len(interface_records),
        "measurements": {
            patch: {
                "requested_layers": spec_by_patch[patch].layers,
                "requested_first_cell_height": spec_by_patch[patch].first_cell_height,
                "requested_growth_ratio": spec_by_patch[patch].growth_ratio,
                "requested_heights": spec_by_patch[patch].layer_heights,
                "measured_height_min": tuple(map(float, measured_height_min[patch])),
                "measured_height_max": tuple(map(float, measured_height_max[patch])),
            }
            for patch in surface_by_patch
        },
    }
    orientation_history: list[int] = []
    for _iteration in range(8):
        geometry = compute_mesh_geometry(result, compute_lsq=False)
        area = np.asarray(geometry["face_area_vector"])
        face_centre = np.asarray(geometry["face_centre"])
        cell_centre = np.asarray(geometry["cell_centre"])
        direction = np.empty_like(area)
        direction[:n_internal] = (
            cell_centre[result["neighbours"]] - cell_centre[result["owners"][:n_internal]]
        )
        direction[n_internal:] = (
            face_centre[n_internal:] - cell_centre[result["owners"][n_internal:]]
        )
        reverse = np.flatnonzero(np.einsum("ij,ij->i", area, direction) < 0.0)
        orientation_history.append(len(reverse))
        if not len(reverse):
            break
        for face_id in reverse:
            combined_faces[int(face_id)] = combined_faces[int(face_id)][::-1].copy()
        result["faces"] = (
            np.ascontiguousarray(combined_faces, dtype=np.int32)
            if len(widths) == 1
            else combined_faces
        )
    else:
        last_reverse = reverse
        closure = np.zeros((result["n_cells"], 3), dtype=np.float64)
        np.add.at(closure, result["owners"], area)
        np.add.at(
            closure,
            result["neighbours"],
            -area[:n_internal],
        )
        layer_internal_start = core_internal
        interface_start = core_internal + len(layer_internal_faces)
        reverse_patches: dict[str, int] = {
            "core_internal": int(np.count_nonzero(last_reverse < layer_internal_start)),
            "layer_internal": int(
                np.count_nonzero(
                    (last_reverse >= layer_internal_start) & (last_reverse < interface_start)
                )
            ),
            "core_layer_interface": int(
                np.count_nonzero(
                    (last_reverse >= interface_start) & (last_reverse < n_internal)
                )
            ),
        }
        for patch in boundary:
            start = int(patch["start_face"])
            stop = start + int(patch["n_faces"])
            reverse_patches[str(patch["name"])] = int(
                np.count_nonzero((last_reverse >= start) & (last_reverse < stop))
            )
        entities = [
            {
                "face": int(face_id),
                "owner": int(result["owners"][face_id]),
                "neighbour": (
                    int(result["neighbours"][face_id])
                    if face_id < n_internal
                    else None
                ),
                "orientation_dot": float(
                    np.dot(area[face_id], direction[face_id])
                ),
                "face_centre": tuple(map(float, face_centre[face_id])),
                "owner_centre": tuple(
                    map(float, cell_centre[int(result["owners"][face_id])])
                ),
                "neighbour_centre": (
                    tuple(
                        map(
                            float,
                            cell_centre[int(result["neighbours"][face_id])],
                        )
                    )
                    if face_id < n_internal
                    else None
                ),
                "owner_volume": float(
                    geometry["cell_volume"][int(result["owners"][face_id])]
                ),
                "neighbour_volume": (
                    float(
                        geometry["cell_volume"][
                            int(result["neighbours"][face_id])
                        ]
                    )
                    if face_id < n_internal
                    else None
                ),
                "owner_closure": tuple(
                    map(float, closure[int(result["owners"][face_id])])
                ),
                "neighbour_closure": (
                    tuple(
                        map(
                            float,
                            closure[int(result["neighbours"][face_id])],
                        )
                    )
                    if face_id < n_internal
                    else None
                ),
            }
            for face_id in last_reverse[:16]
        ]
        raise ValueError(
            "Boundary-layer face orientation did not converge: "
            f"reversed_per_iteration={orientation_history}, last_by_patch={reverse_patches}, "
            f"entities={entities}"
        )
    return result


__all__ = [
    "LayerDiagnostics",
    "LayerSurface",
    "build_layer_surface",
    "build_patch_layers",
    "insert_surface_layers",
]
