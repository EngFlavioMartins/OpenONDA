# SPDX-License-Identifier: GPL-3.0-or-later
"""Exact Cartesian box sides for FVM–VPM coupling interfaces.

The cfMesh-style workflow treats the outer box like any other surface: it
removes the octree leaves touching it, projects, adds a wrapper layer and
optimises. That leaves thin, skewed cells and rounded box edges at the outer
faces. Boundary closures that extrapolate across such cells (the coupling's
mixed velocity condition is one) can become unstable there, and the
particle-mesh interpolation of the coupling assumes a planar interface.

For a box with coupling patches, :class:`CartesianMesher` therefore builds every
*exact* side (each box side that no surface crosses) as follows:

1. The octree root is aligned with the anchor lattice and the workflow runs on
   a domain enlarged by :data:`MARGIN_CELLS` background cells on the exact sides.
   Its box faces, projection, wrapper layer and optimisation stay outside the
   requested box.
2. :func:`restore_exact_zone` moves every octree-template point back to its
   template position within :data:`EXACT_LAYERS` background cells of an exact
   plane, and blends to the optimised position over :data:`BLEND_LAYERS` further
   cells. Template points keep their indices through the workflow; wrapper points
   are appended after them.
3. :func:`clip_to_box` removes the cells outside the requested box. The exposed
   faces lie on lattice planes and join the patch of their box side.
4. :func:`validate_exact_sides` checks the result and raises if it is not
   coupling-grade: planar sides and axis-aligned box cells in the first two
   layers, apart from the junction with a side that a surface crosses.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...mesh.data import MeshData
from ..topology import build_cell_face_csr

MARGIN_CELLS = 3
"""Background cells added beyond each exact side for the meshing workflow."""

EXACT_LAYERS = 2
"""Cell layers inside each exact side restored to the octree template."""

BLEND_LAYERS = 4
"""Further cell layers blending from the template to the optimised positions."""

SURFACE_CLEARANCE_CELLS = EXACT_LAYERS + BLEND_LAYERS + 2
"""Required distance, in background cells, between a surface and an exact side."""

CROSSED_TAPER_CELLS = (1, 3)
"""Distance range, in background cells from a crossed side, over which the template
restoration tapers in; closer, the wrapper geometry of the crossed side is kept."""

JUNCTION_CELLS = CROSSED_TAPER_CELLS[1] + 1
"""Exact-side cells within this many background cells of a crossed side form the
junction, which is only required to be planar."""

_SIDE_NAMES = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")


def side_name(side: int) -> str:
    """Return ``xmin`` ... ``zmax`` for side index 0 ... 5."""
    return _SIDE_NAMES[side]


def crossed_sides(bounds: np.ndarray, surface_bounds: list[np.ndarray]) -> tuple[int, ...]:
    """Return the box sides that a surface's bounding box reaches or crosses."""
    crossed = []
    for side in range(6):
        axis, upper = divmod(side, 2)
        plane = bounds[side]
        for sb in surface_bounds:
            if (upper and sb[2 * axis + 1] >= plane) or (not upper and sb[2 * axis] <= plane):
                crossed.append(side)
                break
    return tuple(crossed)


def exact_side_layout(
    bounds: np.ndarray,
    coupling_sides: tuple[int, ...],
    surface_bounds: list[np.ndarray],
    background_cell_size: float,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return ``(exact_sides, crossed_sides)`` or raise for an invalid coupling box.

    Coupling sides must not be crossed by a surface and need a clearance of
    :data:`SURFACE_CLEARANCE_CELLS` background cells to every surface.
    """
    crossed = crossed_sides(bounds, surface_bounds)
    bad = [side_name(side) for side in coupling_sides if side in crossed]
    if bad:
        raise ValueError(
            f"Coupling box sides {bad} are crossed by a surface. A coupling interface "
            "must lie in the fluid; move the surface or the box face."
        )
    clearance = SURFACE_CLEARANCE_CELLS * background_cell_size
    exact = tuple(side for side in range(6) if side not in crossed)
    for side in exact:
        axis, upper = divmod(side, 2)
        plane = bounds[side]
        for sb in surface_bounds:
            gap = plane - sb[2 * axis + 1] if upper else sb[2 * axis] - plane
            if gap < clearance:
                raise ValueError(
                    f"A surface lies {gap:.6g} m from the exact box side {side_name(side)}; "
                    f"coupling meshes need at least {SURFACE_CLEARANCE_CELLS} background "
                    f"cells ({clearance:.6g} m). Enlarge the box."
                )
    return exact, crossed


def workflow_bounds(
    bounds: np.ndarray, exact_sides: tuple[int, ...], background_cell_size: float
) -> np.ndarray:
    """Enlarge ``bounds`` by :data:`MARGIN_CELLS` background cells on ``exact_sides``."""
    result = np.asarray(bounds, dtype=np.float64).copy()
    for side in exact_sides:
        sign = 1.0 if side % 2 else -1.0
        result[side] += sign * MARGIN_CELLS * background_cell_size
    return result


def _inner_distance(points: np.ndarray, bounds: np.ndarray, sides: tuple[int, ...]) -> np.ndarray:
    """Signed distance inside the box to the nearest of ``sides`` (negative outside)."""
    distance = np.full(len(points), np.inf)
    for side in sides:
        axis, upper = divmod(side, 2)
        value = bounds[side] - points[:, axis] if upper else points[:, axis] - bounds[side]
        distance = np.minimum(distance, value)
    return distance


def _smoothstep(values: np.ndarray) -> np.ndarray:
    t = np.clip(values, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def restore_exact_zone(
    mesh_data: MeshData,
    template_points: np.ndarray,
    bounds: np.ndarray,
    exact_sides: tuple[int, ...],
    crossed: tuple[int, ...],
    background_cell_size: float,
) -> dict[str, float]:
    """Return points near the exact sides to their octree-template coordinates.

    The weight is one within :data:`EXACT_LAYERS` cells of an exact plane and
    outside the box, and decreases smoothly to zero over :data:`BLEND_LAYERS`
    cells. It applies to the coordinates along the axes of the exact sides.
    Along the axis of a side that a surface crosses, it tapers to zero towards
    that side (:data:`CROSSED_TAPER_CELLS`), keeping the crossed side's wrapper
    geometry. Wrapper boundary points, which have no template position, take
    the exact-axis coordinates of the template point inward of them in their
    wrapper column, so that exact planes stay planar up to the crossed sides.
    """
    points = np.asarray(mesh_data["vertex_position"], dtype=np.float64).copy()
    n_template = len(template_points)
    if n_template > len(points):
        raise RuntimeError("Mesh workflow removed octree-template points")
    h = float(background_cell_size)
    template = np.asarray(template_points, dtype=np.float64)
    depth = _inner_distance(template, bounds, exact_sides)
    weight = 1.0 - _smoothstep((depth - EXACT_LAYERS * h) / (BLEND_LAYERS * h))
    axis_weight = np.repeat(weight[:, None], 3, axis=1)
    start, stop = CROSSED_TAPER_CELLS
    for side in crossed:
        axis = side // 2
        crossed_depth = _inner_distance(template, bounds, (side,))
        axis_weight[:, axis] *= _smoothstep((crossed_depth - start * h) / ((stop - start) * h))
    current = points[:n_template]
    moved = np.abs(current - template).max(axis=1)
    points[:n_template] = axis_weight * template + (1.0 - axis_weight) * current

    # Wrapper boundary points follow their column's template point.
    columns = mesh_data.get("_cfmesh_boundary_column_inner") or {}
    column_points = 0
    for column_map in columns.values():
        if not column_map:
            continue
        outer = np.fromiter(column_map.keys(), dtype=np.int64)
        inner = np.fromiter(column_map.values(), dtype=np.int64)
        keep = (outer >= n_template) & (inner < n_template)
        outer, inner = outer[keep], inner[keep]
        column_points += len(outer)
        w = axis_weight[inner].copy()
        for side in crossed:
            # A point on a crossed side stays on its plane.
            axis = side // 2
            on_plane = np.abs(points[outer, axis] - bounds[side]) <= 0.5 * h
            w[on_plane, axis] = 0.0
        points[outer] = w * template[inner] + (1.0 - w) * points[outer]
    mesh_data["vertex_position"] = points
    zone = weight > 0.0
    return {
        "restored_points": int(np.count_nonzero(weight >= 1.0)),
        "blended_points": int(np.count_nonzero(zone & (weight < 1.0))),
        "wrapper_column_points": int(column_points),
        "max_restored_displacement": float(moved[zone].max()) if np.any(zone) else 0.0,
    }


def _face_list(mesh_data: MeshData) -> list[np.ndarray] | np.ndarray:
    faces = mesh_data["faces"]
    try:
        array = np.asarray(faces, dtype=np.int32)
    except (TypeError, ValueError):
        return [np.asarray(face, dtype=np.int32) for face in faces]
    return array if array.ndim == 2 else [np.asarray(face, dtype=np.int32) for face in faces]


def _cell_centres(mesh_data: MeshData, faces, points: np.ndarray) -> np.ndarray:
    """Mean of each cell's face centroids (classification only)."""
    n_cells = int(mesh_data["n_cells"])
    n_internal = int(mesh_data["n_interior_faces"])
    owners = np.asarray(mesh_data["owners"], dtype=np.int64)
    neighbours = np.asarray(mesh_data["neighbours"], dtype=np.int64)
    if isinstance(faces, np.ndarray):
        centroid = points[faces].mean(axis=1)
    else:
        centroid = np.stack([points[face].mean(axis=0) for face in faces])
    total = np.zeros((n_cells, 3))
    count = np.zeros(n_cells)
    np.add.at(total, owners, centroid)
    np.add.at(count, owners, 1.0)
    np.add.at(total, neighbours, centroid[:n_internal])
    np.add.at(count, neighbours, 1.0)
    return total / count[:, None]


def clip_to_box(
    mesh_data: MeshData,
    bounds: np.ndarray,
    exact_sides: tuple[int, ...],
    side_patch_names: tuple[str, ...],
    background_cell_size: float,
) -> MeshData:
    """Remove the cells outside ``bounds`` on ``exact_sides`` and name the exposed faces.

    Exposed faces must lie on an exact plane. Their vertices are set exactly on
    it (removing round-off and the residual offset of wrapper points at
    junctions with crossed sides) and the faces join the side's patch. Patch
    order, interior-face order and per-cell metadata are preserved.
    """
    points = np.asarray(mesh_data["vertex_position"], dtype=np.float64).copy()
    faces = _face_list(mesh_data)
    n_faces = int(mesh_data["n_faces"])
    n_cells = int(mesh_data["n_cells"])
    n_internal = int(mesh_data["n_interior_faces"])
    owners = np.asarray(mesh_data["owners"], dtype=np.int64)
    neighbours = np.asarray(mesh_data["neighbours"], dtype=np.int64)
    centres = _cell_centres(mesh_data, faces, points)
    keep = np.ones(n_cells, dtype=bool)
    for side in exact_sides:
        axis, upper = divmod(side, 2)
        keep &= centres[:, axis] < bounds[side] if upper else centres[:, axis] > bounds[side]
    if not np.any(keep):
        raise ValueError("Exact coupling sides removed every cell")
    cell_map = np.full(n_cells, -1, dtype=np.int64)
    cell_map[keep] = np.arange(int(keep.sum()))

    own_keep = keep[owners[:n_internal]]
    nei_keep = keep[neighbours]
    interior = np.flatnonzero(own_keep & nei_keep)
    exposed = np.flatnonzero(own_keep ^ nei_keep)
    tolerance = 0.25 * background_cell_size

    def face_nodes(face_id: int) -> np.ndarray:
        return np.asarray(faces[face_id], dtype=np.int64)

    exposed_side = np.empty(len(exposed), dtype=np.int64)
    for k, face_id in enumerate(exposed):
        nodes = face_nodes(int(face_id))
        coords = points[nodes]
        matches = []
        for side in exact_sides:
            axis = side // 2
            if np.all(np.abs(coords[:, axis] - bounds[side]) <= tolerance):
                matches.append(side)
        if len(matches) != 1:
            centre = coords.mean(axis=0)
            raise ValueError(
                "Exact coupling side clip exposed a face that is not on one box plane "
                f"(face centre {np.array2string(centre, precision=6)}); a surface or "
                "refinement transition is too close to the box."
            )
        exposed_side[k] = matches[0]

    names = [str(patch["name"]) for patch in mesh_data["boundary"]]
    new_faces: list[np.ndarray] = [face_nodes(int(f)) for f in interior]
    new_owners = list(cell_map[owners[interior]])
    new_neighbours = list(cell_map[neighbours[interior]])
    boundary: list[dict[str, Any]] = []
    side_faces: dict[str, list[tuple[np.ndarray, int]]] = {}
    for k, face_id in enumerate(exposed):
        face_id = int(face_id)
        nodes = face_nodes(face_id)
        if own_keep[face_id]:
            owner = int(cell_map[owners[face_id]])
        else:
            nodes = nodes[::-1].copy()
            owner = int(cell_map[neighbours[face_id]])
        name = side_patch_names[int(exposed_side[k])]
        side_faces.setdefault(name, []).append((nodes, owner))
        axis = int(exposed_side[k]) // 2
        points[nodes, axis] = bounds[int(exposed_side[k])]
    for name in names + [n for n in side_faces if n not in names]:
        start = len(new_faces)
        source = next((p for p in mesh_data["boundary"] if str(p["name"]) == name), None)
        if source is not None:
            first = int(source["start_face"])
            for face_id in range(first, first + int(source["n_faces"])):
                if keep[owners[face_id]]:
                    new_faces.append(face_nodes(face_id))
                    new_owners.append(int(cell_map[owners[face_id]]))
        for nodes, owner in side_faces.get(name, ()):
            new_faces.append(nodes)
            new_owners.append(owner)
        patch = dict(source or {"type": "patch"})
        patch.update({"name": name, "start_face": start, "n_faces": len(new_faces) - start})
        if patch["n_faces"] or source is not None:
            boundary.append(patch)

    used = np.unique(np.concatenate(new_faces))
    point_map = np.full(len(points), -1, dtype=np.int64)
    point_map[used] = np.arange(len(used))
    mapped = [point_map[face].astype(np.int32) for face in new_faces]
    widths = {len(face) for face in mapped}
    result: MeshData = {
        key: value
        for key, value in mesh_data.items()
        if key
        not in {
            "vertex_position",
            "faces",
            "owners",
            "neighbours",
            "boundary",
            "n_cells",
            "n_faces",
            "n_interior_faces",
            "n_points",
            "cell_face_indices",
            "cell_face_offset",
            "cell_face_reversed",
            "_cfmesh_cell_face_order",
            "_cfmesh_octree_leaves",
            "_cfmesh_boundary_column_inner",
        }
    }
    result.update(
        {
            "vertex_position": np.ascontiguousarray(points[used]),
            "faces": np.ascontiguousarray(mapped, dtype=np.int32) if len(widths) == 1 else mapped,
            "owners": np.ascontiguousarray(new_owners, dtype=np.int32),
            "neighbours": np.ascontiguousarray(new_neighbours, dtype=np.int32),
            "boundary": boundary,
            "n_cells": int(keep.sum()),
            "n_faces": len(mapped),
            "n_interior_faces": len(interior),
            "n_points": len(used),
        }
    )
    for key, value in mesh_data.items():
        if isinstance(value, np.ndarray) and value.shape[:1] == (n_cells,) and key in result:
            if key == "cell_vertex_indices":
                result[key] = np.ascontiguousarray(point_map[value[keep]].astype(value.dtype))
            else:
                result[key] = np.ascontiguousarray(value[keep])
    if "cell_face_indices" in mesh_data:
        indices, offsets = build_cell_face_csr(
            result["owners"], result["neighbours"], result["n_cells"], result["n_faces"]
        )
        result["cell_face_indices"] = indices
        result["cell_face_offset"] = offsets
    if n_faces != len(owners):
        raise RuntimeError("Mesh face and owner counts differ")
    return result


def _cell_boxes(mesh_data: MeshData, faces, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_cells = int(mesh_data["n_cells"])
    n_internal = int(mesh_data["n_interior_faces"])
    owners = np.asarray(mesh_data["owners"], dtype=np.int64)
    neighbours = np.asarray(mesh_data["neighbours"], dtype=np.int64)
    if isinstance(faces, np.ndarray):
        lo = points[faces].min(axis=1)
        hi = points[faces].max(axis=1)
    else:
        lo = np.stack([points[f].min(axis=0) for f in faces])
        hi = np.stack([points[f].max(axis=0) for f in faces])
    cell_lo = np.full((n_cells, 3), np.inf)
    cell_hi = np.full((n_cells, 3), -np.inf)
    np.minimum.at(cell_lo, owners, lo)
    np.maximum.at(cell_hi, owners, hi)
    np.minimum.at(cell_lo, neighbours, lo[:n_internal])
    np.maximum.at(cell_hi, neighbours, hi[:n_internal])
    return cell_lo, cell_hi


def validate_exact_sides(
    mesh_data: MeshData,
    geometry: dict[str, np.ndarray],
    bounds: np.ndarray,
    exact_sides: tuple[int, ...],
    crossed: tuple[int, ...],
    side_patch_names: tuple[str, ...],
    background_cell_size: float,
) -> dict[str, Any]:
    """Check coupling-grade geometry of the exact sides and return the metrics.

    Every face of an exact side must lie on its plane with the outward axis
    normal. The owner cells and their face neighbours (the first two layers)
    must be axis-aligned boxes, so that the owner-to-face vector is normal to
    the face. Faces at the junction with a crossed side are only
    required to be planar when they lie within :data:`JUNCTION_CELLS` background
    cells of a crossed side. Raises ``ValueError`` on violation.
    """
    points = np.asarray(mesh_data["vertex_position"], dtype=np.float64)
    faces = _face_list(mesh_data)
    owners = np.asarray(mesh_data["owners"], dtype=np.int64)
    neighbours = np.asarray(mesh_data["neighbours"], dtype=np.int64)
    n_internal = int(mesh_data["n_interior_faces"])
    h = float(background_cell_size)
    cell_lo, cell_hi = _cell_boxes(mesh_data, faces, points)
    volume = np.asarray(geometry["cell_volume"], dtype=np.float64)
    box_volume = np.prod(cell_hi - cell_lo, axis=1)
    is_box = np.abs(box_volume - volume) <= 1.0e-9 * box_volume
    centres = np.asarray(geometry["cell_centre"], dtype=np.float64)
    area_vector = np.asarray(geometry["face_area_vector"], dtype=np.float64)
    connection = np.asarray(geometry["cell_connection_vector"], dtype=np.float64)

    # Face neighbours of each cell, for the second layer.
    report: dict[str, Any] = {"sides": {}}
    names = [str(patch["name"]) for patch in mesh_data["boundary"]]
    for side in exact_sides:
        axis, upper = divmod(side, 2)
        plane = bounds[side]
        name = side_patch_names[side]
        patch = mesh_data["boundary"][names.index(name)]
        start, count = int(patch["start_face"]), int(patch["n_faces"])
        face_ids = np.arange(start, start + count)
        plane_tolerance = 1.0e-9 * max(1.0, abs(plane))
        if isinstance(faces, np.ndarray):
            deviation = np.abs(points[faces[face_ids], axis] - plane).max(axis=1)
        else:
            deviation = np.array(
                [np.abs(points[faces[f], axis] - plane).max() for f in face_ids], dtype=np.float64
            )
        on_side = face_ids[deviation <= plane_tolerance]
        if not len(on_side):
            raise ValueError(f"Exact box side {side_name(side)} has no faces")
        normal = area_vector[on_side] / np.linalg.norm(area_vector[on_side], axis=1)[:, None]
        expected = np.zeros(3)
        expected[axis] = 1.0 if upper else -1.0
        normal_error = float(np.max(np.linalg.norm(normal - expected, axis=1)))
        area = float(np.sum(np.linalg.norm(area_vector[on_side], axis=1)))
        lo_extent = np.delete(np.arange(3), axis)
        footprint = float(np.prod([bounds[2 * a + 1] - bounds[2 * a] for a in lo_extent]))
        owner_cells = owners[on_side]
        near_crossed = np.zeros(len(on_side), dtype=bool)
        if crossed:
            depth = _inner_distance(centres[owner_cells], bounds, crossed)
            near_crossed = depth < JUNCTION_CELLS * h
        strict_owners = owner_cells[~near_crossed]
        strict = np.zeros(len(volume), dtype=bool)
        strict[strict_owners] = True
        interior_owners = owners[:n_internal]
        layer2_mask = np.zeros(len(volume), dtype=bool)
        layer2_mask[neighbours[strict[interior_owners]]] = True
        layer2_mask[interior_owners[strict[neighbours]]] = True
        layer2_cells = np.flatnonzero(layer2_mask & ~strict)
        if len(layer2_cells) and crossed:
            depth2 = _inner_distance(centres[layer2_cells], bounds, crossed)
            layer2_cells = layer2_cells[depth2 >= JUNCTION_CELLS * h]
        r = connection[on_side][~near_crossed]
        n_strict = normal[~near_crossed]
        d = np.einsum("ij,ij->i", r, n_strict)
        tangential = np.linalg.norm(r - d[:, None] * n_strict, axis=1)
        skew = float(np.max(tangential / d)) if len(d) else 0.0
        bad_owner = strict_owners[~is_box[strict_owners]]
        bad_layer2 = layer2_cells[~is_box[layer2_cells]] if len(layer2_cells) else layer2_cells
        side_report = {
            "faces": int(len(on_side)),
            "area_fraction": area / footprint,
            "max_normal_error": normal_error,
            "max_tangential_offset_ratio": skew,
            "non_box_first_layer_cells": int(len(bad_owner)),
            "non_box_second_layer_cells": int(len(bad_layer2)),
            "junction_faces_near_crossed_sides": int(np.count_nonzero(near_crossed)),
        }
        report["sides"][side_name(side)] = side_report
        problems = []
        if abs(side_report["area_fraction"] - 1.0) > 1.0e-9 and not crossed:
            problems.append(f"covers {side_report['area_fraction']:.9f} of the box face")
        if normal_error > 1.0e-9:
            problems.append(f"face normals deviate by {normal_error:.3e}")
        if skew > 1.0e-9:
            problems.append(f"owner-to-face offset ratio {skew:.3e}")
        if len(bad_owner):
            c = centres[bad_owner[0]]
            problems.append(f"{len(bad_owner)} distorted first-layer cells (e.g. at {c})")
        if len(bad_layer2):
            c = centres[bad_layer2[0]]
            problems.append(f"{len(bad_layer2)} distorted second-layer cells (e.g. at {c})")
        if problems:
            raise ValueError(
                f"Coupling box side {side_name(side)} ({name!r}) is not coupling-grade: "
                + "; ".join(problems)
            )
    return report


__all__ = [
    "BLEND_LAYERS",
    "EXACT_LAYERS",
    "MARGIN_CELLS",
    "SURFACE_CLEARANCE_CELLS",
    "clip_to_box",
    "exact_side_layout",
    "restore_exact_zone",
    "validate_exact_sides",
    "workflow_bounds",
]
