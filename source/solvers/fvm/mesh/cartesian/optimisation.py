# SPDX-License-Identifier: GPL-3.0-or-later
"""Typed quality-optimisation diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..geometry import compute_mesh_geometry
from ..surface_classification import SurfaceIndex


@dataclass(frozen=True, slots=True)
class OptimisationDiagnostics:
    """Quality values measured after a mesh stage."""

    max_non_orthogonality_deg: float | None = None
    max_skewness: float | None = None
    max_aspect_ratio: float | None = None

    @classmethod
    def from_quality(cls, quality: dict) -> OptimisationDiagnostics:
        """Adapt authoritative validation metrics into a typed stage result."""
        return cls(
            max_non_orthogonality_deg=_optional_float(quality, "max_non_orthogonality_deg"),
            max_skewness=_optional_float(quality, "max_skewness"),
            max_aspect_ratio=_optional_float(quality, "max_aspect_ratio"),
        )

    def as_dict(self) -> dict[str, float | None]:
        """Return a serialisable diagnostics snapshot."""
        return {
            "max_non_orthogonality_deg": self.max_non_orthogonality_deg,
            "max_skewness": self.max_skewness,
            "max_aspect_ratio": self.max_aspect_ratio,
        }


def _optional_float(values: dict, key: str) -> float | None:
    value = values.get(key)
    return None if value is None else float(value)


def agglomerate_small_cut_cells(
    mesh_data: dict[str, Any],
    wall_patches: tuple[str, ...],
    *,
    minimum_volume_fraction: float = 0.02,
    surface_indices: tuple[SurfaceIndex, ...] = (),
) -> dict[str, Any]:
    """Merge undersized recovered cut cells into their strongest neighbour.

    Cartesian surface recovery can leave a valid but arbitrarily small fluid
    sliver when an STL facet passes close to a lattice corner.  Keeping that
    sliver is harmful to both conditioning and non-orthogonality.  This pass
    agglomerates only wall-adjacent cells below a fraction of their source
    Cartesian volume; regular octree and refinement-transition cells are
    untouched.
    """
    if not 0.0 < minimum_volume_fraction < 0.5:
        raise ValueError("minimum_volume_fraction must lie between zero and 0.5")
    geometry = compute_mesh_geometry(mesh_data, compute_lsq=False)
    volumes = np.asarray(geometry["cell_volume"], dtype=np.float64)
    sizes = np.asarray(mesh_data["cell_sizes"], dtype=np.float64)
    if len(sizes) != len(volumes):
        raise ValueError("cell_sizes must cover every cell before cut-cell optimisation")
    wall_cells: set[int] = set()
    for patch in mesh_data["boundary"]:
        if str(patch["name"]) not in wall_patches:
            continue
        start = int(patch["start_face"])
        stop = start + int(patch["n_faces"])
        wall_cells.update(map(int, np.asarray(mesh_data["owners"])[start:stop]))
    fractions = volumes / np.maximum(sizes**3, np.finfo(np.float64).tiny)
    inside_surface = np.zeros(len(volumes), dtype=bool)
    for index in surface_indices:
        inside_surface |= index.is_inside(np.asarray(geometry["cell_centre"]))
    candidates = np.asarray(
        sorted(
            cell
            for cell in wall_cells
            if fractions[cell] < minimum_volume_fraction or inside_surface[cell]
        ),
        dtype=np.int64,
    )
    if not len(candidates):
        mesh_data.setdefault("mesh_generation", {})["cut_cell_agglomeration"] = {
            "merged_cells": 0,
            "minimum_volume_fraction": minimum_volume_fraction,
        }
        return mesh_data

    n_cells = int(mesh_data["n_cells"])
    n_internal = int(mesh_data["n_interior_faces"])
    owners = np.asarray(mesh_data["owners"], dtype=np.int64)
    neighbours = np.asarray(mesh_data["neighbours"], dtype=np.int64)
    face_areas = np.asarray(geometry["face_area"], dtype=np.float64)
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n_cells)]
    for face_id in range(n_internal):
        owner = int(owners[face_id])
        neighbour = int(neighbours[face_id])
        area = float(face_areas[face_id])
        adjacency[owner].append((neighbour, area))
        adjacency[neighbour].append((owner, area))

    parent = np.arange(n_cells, dtype=np.int64)

    def root(cell: int) -> int:
        current = cell
        while parent[current] != current:
            parent[current] = parent[parent[current]]
            current = int(parent[current])
        return current

    candidate_set = set(map(int, candidates))
    merged = 0
    for cell in sorted(map(int, candidates), key=lambda value: (volumes[value], value)):
        source = root(cell)
        choices = [
            (neighbour not in candidate_set, volumes[neighbour], area, -neighbour, neighbour)
            for neighbour, area in adjacency[cell]
            if root(neighbour) != source
        ]
        if not choices:
            raise ValueError(f"Small cut cell {cell} has no fluid neighbour for agglomeration")
        target = root(max(choices)[-1])
        if target == source:
            continue
        parent[source] = target
        merged += 1
    for cell in range(n_cells):
        parent[cell] = root(cell)

    representatives = np.unique(parent)
    new_id = np.full(n_cells, -1, dtype=np.int32)
    for index, representative in enumerate(representatives):
        new_id[parent == representative] = index
    mapped_owners = new_id[owners]
    mapped_neighbours = new_id[neighbours]

    kept_internal = mapped_owners[:n_internal] != mapped_neighbours
    internal_faces = [
        np.asarray(mesh_data["faces"][face_id], dtype=np.int32)
        for face_id in np.flatnonzero(kept_internal)
    ]
    internal_owners = mapped_owners[:n_internal][kept_internal]
    internal_neighbours = mapped_neighbours[kept_internal]
    face_blocks = list(internal_faces)
    owner_blocks = list(map(int, internal_owners))
    boundary: list[dict[str, Any]] = []
    start_face = len(face_blocks)
    for patch in mesh_data["boundary"]:
        start = int(patch["start_face"])
        stop = start + int(patch["n_faces"])
        faces = [np.asarray(face, dtype=np.int32) for face in mesh_data["faces"][start:stop]]
        patch_owners = mapped_owners[start:stop]
        face_blocks.extend(faces)
        owner_blocks.extend(map(int, patch_owners))
        boundary.append(
            {
                "name": str(patch["name"]),
                "start_face": start_face,
                "n_faces": len(faces),
                "type": str(patch.get("type", "patch")),
            }
        )
        start_face += len(faces)

    widths = {len(face) for face in face_blocks}
    mesh_data["faces"] = (
        np.ascontiguousarray(face_blocks, dtype=np.int32) if len(widths) == 1 else face_blocks
    )
    mesh_data["owners"] = np.ascontiguousarray(owner_blocks, dtype=np.int32)
    mesh_data["neighbours"] = np.ascontiguousarray(internal_neighbours, dtype=np.int32)
    mesh_data["boundary"] = boundary
    mesh_data["n_cells"] = len(representatives)
    mesh_data["n_faces"] = len(face_blocks)
    mesh_data["n_interior_faces"] = len(internal_faces)
    levels = np.asarray(mesh_data["cell_levels"])
    new_levels = np.empty(len(representatives), dtype=levels.dtype)
    new_sizes = np.empty(len(representatives), dtype=np.asarray(mesh_data["cell_sizes"]).dtype)
    for index, representative in enumerate(representatives):
        members = np.flatnonzero(parent == representative)
        new_levels[index] = np.max(levels[members])
        new_sizes[index] = np.min(sizes[members])
    mesh_data["cell_levels"] = np.ascontiguousarray(new_levels)
    mesh_data["cell_sizes"] = np.ascontiguousarray(new_sizes)
    mesh_data.pop("cell_face_indices", None)
    mesh_data.pop("cell_face_offset", None)
    mesh_data.pop("cell_vertex_indices", None)
    mesh_data.pop("cell_type_code", None)
    mesh_data.setdefault("mesh_generation", {})["cut_cell_agglomeration"] = {
        "merged_cells": merged,
        "minimum_volume_fraction": minimum_volume_fraction,
        "minimum_fraction_before": float(fractions[candidates].min()),
        "inside_centres_before": int(np.count_nonzero(inside_surface[candidates])),
    }
    oriented_faces = [np.asarray(face, dtype=np.int32) for face in mesh_data["faces"]]
    for _iteration in range(4):
        geometry = compute_mesh_geometry(mesh_data, compute_lsq=False)
        area = np.asarray(geometry["face_area_vector"])
        face_centre = np.asarray(geometry["face_centre"])
        cell_centre = np.asarray(geometry["cell_centre"])
        direction = np.empty_like(area)
        direction[: mesh_data["n_interior_faces"]] = (
            cell_centre[mesh_data["neighbours"]]
            - cell_centre[mesh_data["owners"][: mesh_data["n_interior_faces"]]]
        )
        direction[mesh_data["n_interior_faces"] :] = (
            face_centre[mesh_data["n_interior_faces"] :]
            - cell_centre[mesh_data["owners"][mesh_data["n_interior_faces"] :]]
        )
        reverse = np.flatnonzero(np.einsum("ij,ij->i", area, direction) < 0.0)
        if not len(reverse):
            break
        for face_id in reverse:
            oriented_faces[int(face_id)] = oriented_faces[int(face_id)][::-1].copy()
        widths = {len(face) for face in oriented_faces}
        mesh_data["faces"] = (
            np.ascontiguousarray(oriented_faces, dtype=np.int32)
            if len(widths) == 1
            else oriented_faces
        )
        mesh_data.pop("cell_face_indices", None)
        mesh_data.pop("cell_face_offset", None)
    else:
        raise ValueError("Agglomerated cut-cell face orientation did not converge")
    return mesh_data


def agglomerate_small_layer_columns(
    mesh_data: dict[str, Any],
    target_surface_size: float,
    *,
    minimum_area_fraction: float = 0.005,
) -> dict[str, Any]:
    """Tangentially merge tiny layer columns consistently through all bands."""
    if not np.isfinite(target_surface_size) or target_surface_size <= 0.0:
        raise ValueError("target_surface_size must be finite and positive")
    if not 0.0 < minimum_area_fraction < 0.5:
        raise ValueError("minimum_area_fraction must lie between zero and 0.5")
    labels = np.asarray(mesh_data.get("boundary_layer_index"), dtype=np.int16)
    if labels.shape != (int(mesh_data["n_cells"]),):
        raise ValueError("boundary_layer_index must cover every layered mesh cell")
    geometry = compute_mesh_geometry(mesh_data, compute_lsq=False)
    n_internal = int(mesh_data["n_interior_faces"])
    owners = np.asarray(mesh_data["owners"], dtype=np.int64)
    neighbours = np.asarray(mesh_data["neighbours"], dtype=np.int64)
    n_cells = int(mesh_data["n_cells"])
    column_parent = np.arange(n_cells, dtype=np.int64)

    def column_root(cell: int) -> int:
        while column_parent[cell] != cell:
            column_parent[cell] = column_parent[column_parent[cell]]
            cell = int(column_parent[cell])
        return cell

    def join_column(first: int, second: int) -> None:
        first_root = column_root(first)
        second_root = column_root(second)
        if first_root != second_root:
            column_parent[second_root] = first_root

    for face_id in range(n_internal):
        owner = int(owners[face_id])
        neighbour = int(neighbours[face_id])
        if labels[owner] >= 0 and labels[neighbour] >= 0 and labels[owner] != labels[neighbour]:
            join_column(owner, neighbour)
    columns: dict[int, dict[int, int]] = {}
    for cell in np.flatnonzero(labels >= 0):
        columns.setdefault(column_root(int(cell)), {})[int(labels[cell])] = int(cell)
    number_of_layers = int(labels.max(initial=-1)) + 1
    complete_columns = {
        root: cells
        for root, cells in columns.items()
        if len(cells) == number_of_layers
    }
    volumes = np.asarray(geometry["cell_volume"], dtype=np.float64)
    heights = np.asarray(mesh_data["cell_sizes"], dtype=np.float64)
    column_area = {
        root: volumes[cells[0]] / max(heights[cells[0]], np.finfo(np.float64).tiny)
        for root, cells in complete_columns.items()
    }
    area_fraction = {
        root: area / target_surface_size**2 for root, area in column_area.items()
    }
    candidates = sorted(
        (root for root, fraction in area_fraction.items() if fraction < minimum_area_fraction),
        key=lambda root: (area_fraction[root], root),
    )
    face_areas = np.asarray(geometry["face_area"], dtype=np.float64)
    adjacency: dict[int, dict[int, float]] = {root: {} for root in complete_columns}
    for face_id in range(n_internal):
        owner = int(owners[face_id])
        neighbour = int(neighbours[face_id])
        if labels[owner] < 0 or labels[owner] != labels[neighbour]:
            continue
        first_root = column_root(owner)
        second_root = column_root(neighbour)
        if first_root == second_root or first_root not in adjacency or second_root not in adjacency:
            continue
        area = float(face_areas[face_id])
        adjacency[first_root][second_root] = adjacency[first_root].get(second_root, 0.0) + area
        adjacency[second_root][first_root] = adjacency[second_root].get(first_root, 0.0) + area

    parent = np.arange(n_cells, dtype=np.int64)
    used_columns: set[int] = set()
    merged_columns = 0
    candidate_set = set(candidates)
    for source_root in candidates:
        if source_root in used_columns:
            continue
        choices = [
            (target not in candidate_set, shared_area, column_area[target], -target, target)
            for target, shared_area in adjacency[source_root].items()
            if target not in used_columns
        ]
        if not choices:
            continue
        target_root = max(choices)[-1]
        for layer_index in range(number_of_layers):
            parent[complete_columns[source_root][layer_index]] = complete_columns[target_root][
                layer_index
            ]
        used_columns.update((source_root, target_root))
        merged_columns += 1
    if not merged_columns:
        mesh_data.setdefault("mesh_generation", {})["layer_column_agglomeration"] = {
            "merged_columns": 0,
            "minimum_area_fraction": minimum_area_fraction,
            "candidate_columns": len(candidates),
        }
        return mesh_data
    representatives = np.unique(parent)
    new_id = np.full(n_cells, -1, dtype=np.int32)
    for index, representative in enumerate(representatives):
        new_id[parent == representative] = index
    mapped_owners = new_id[owners]
    mapped_neighbours = new_id[neighbours]
    kept_internal = mapped_owners[:n_internal] != mapped_neighbours
    face_blocks = [
        np.asarray(mesh_data["faces"][face_id], dtype=np.int32)
        for face_id in np.flatnonzero(kept_internal)
    ]
    owner_blocks = list(map(int, mapped_owners[:n_internal][kept_internal]))
    internal_neighbours = mapped_neighbours[kept_internal]
    boundary: list[dict[str, Any]] = []
    start_face = len(face_blocks)
    for patch in mesh_data["boundary"]:
        start = int(patch["start_face"])
        stop = start + int(patch["n_faces"])
        faces = [np.asarray(face, dtype=np.int32) for face in mesh_data["faces"][start:stop]]
        face_blocks.extend(faces)
        owner_blocks.extend(map(int, mapped_owners[start:stop]))
        boundary.append(
            {
                "name": str(patch["name"]),
                "start_face": start_face,
                "n_faces": len(faces),
                "type": str(patch.get("type", "patch")),
            }
        )
        start_face += len(faces)
    widths = {len(face) for face in face_blocks}
    mesh_data["faces"] = (
        np.ascontiguousarray(face_blocks, dtype=np.int32) if len(widths) == 1 else face_blocks
    )
    mesh_data["owners"] = np.ascontiguousarray(owner_blocks, dtype=np.int32)
    mesh_data["neighbours"] = np.ascontiguousarray(internal_neighbours, dtype=np.int32)
    mesh_data["boundary"] = boundary
    mesh_data["n_cells"] = len(representatives)
    mesh_data["n_faces"] = len(face_blocks)
    mesh_data["n_interior_faces"] = int(np.count_nonzero(kept_internal))
    levels = np.asarray(mesh_data["cell_levels"])
    sizes = np.asarray(mesh_data["cell_sizes"])
    new_levels = np.empty(len(representatives), dtype=levels.dtype)
    new_sizes = np.empty(len(representatives), dtype=sizes.dtype)
    new_labels = np.empty(len(representatives), dtype=labels.dtype)
    for index, representative in enumerate(representatives):
        members = np.flatnonzero(parent == representative)
        new_levels[index] = np.max(levels[members])
        new_sizes[index] = np.min(sizes[members])
        member_labels = np.unique(labels[members])
        if len(member_labels) != 1:
            raise ValueError("Layer agglomeration crossed radial bands")
        new_labels[index] = member_labels[0]
    mesh_data["cell_levels"] = np.ascontiguousarray(new_levels)
    mesh_data["cell_sizes"] = np.ascontiguousarray(new_sizes)
    mesh_data["boundary_layer_index"] = np.ascontiguousarray(new_labels)
    mesh_data.pop("cell_face_indices", None)
    mesh_data.pop("cell_face_offset", None)
    mesh_data.pop("cell_vertex_indices", None)
    mesh_data.pop("cell_type_code", None)
    mesh_data.setdefault("mesh_generation", {})["layer_column_agglomeration"] = {
        "merged_columns": merged_columns,
        "merged_cells": merged_columns * number_of_layers,
        "minimum_area_fraction": minimum_area_fraction,
        "candidate_columns": len(candidates),
        "minimum_fraction_before": float(
            min((area_fraction[root] for root in candidates), default=1.0)
        ),
    }
    oriented_faces = [np.asarray(face, dtype=np.int32) for face in mesh_data["faces"]]
    for _iteration in range(4):
        updated = compute_mesh_geometry(mesh_data, compute_lsq=False)
        area = np.asarray(updated["face_area_vector"])
        face_centre = np.asarray(updated["face_centre"])
        cell_centre = np.asarray(updated["cell_centre"])
        n_internal = int(mesh_data["n_interior_faces"])
        direction = np.empty_like(area)
        direction[:n_internal] = (
            cell_centre[mesh_data["neighbours"]] - cell_centre[mesh_data["owners"][:n_internal]]
        )
        direction[n_internal:] = (
            face_centre[n_internal:] - cell_centre[mesh_data["owners"][n_internal:]]
        )
        reverse = np.flatnonzero(np.einsum("ij,ij->i", area, direction) < 0.0)
        if not len(reverse):
            break
        for face_id in reverse:
            oriented_faces[int(face_id)] = oriented_faces[int(face_id)][::-1].copy()
        mesh_data["faces"] = (
            np.ascontiguousarray(oriented_faces, dtype=np.int32)
            if len(widths) == 1
            else oriented_faces
        )
        mesh_data.pop("cell_face_indices", None)
        mesh_data.pop("cell_face_offset", None)
    else:
        raise ValueError("Agglomerated layer-cell face orientation did not converge")
    return mesh_data


__all__ = [
    "OptimisationDiagnostics",
    "agglomerate_small_cut_cells",
    "agglomerate_small_layer_columns",
]
