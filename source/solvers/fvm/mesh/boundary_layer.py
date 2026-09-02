"""Generic boundary-layer metadata and native layer/core stitching.

The public native mesher builds patch-normal layers in
``mesh/cartesian/boundary_layers.py``.  This module contains only the small
generic metadata object retained by the legacy general-body compatibility
mesher and the topology stitcher shared by both paths.  It intentionally has
no geometry recognition, coordinate-axis assumptions, or O-grid construction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class BoundaryLayerSpec:
    """Generic wall-normal layer controls for the legacy compatibility path."""

    first_cell_height: float
    layers: int
    growth_ratio: float
    transition_layers: int

    def validate(self) -> None:
        """Reject non-physical layer counts, sizes, and growth controls."""
        if not math.isfinite(self.first_cell_height) or self.first_cell_height <= 0.0:
            raise ValueError("first_cell_height must be finite and positive")
        if int(self.layers) != self.layers or self.layers < 1:
            raise ValueError("layers must be a positive integer")
        if not math.isfinite(self.growth_ratio) or self.growth_ratio < 1.0:
            raise ValueError("growth_ratio must be finite and at least one")
        if int(self.transition_layers) != self.transition_layers or self.transition_layers < 1:
            raise ValueError("transition_layers must be a positive integer")

    @property
    def thickness(self) -> float:
        """Total requested thickness of the constant-growth wall layers."""
        if math.isclose(self.growth_ratio, 1.0):
            return self.layers * self.first_cell_height
        return (
            self.first_cell_height
            * (self.growth_ratio**self.layers - 1.0)
            / (self.growth_ratio - 1.0)
        )

    @property
    def layer_heights(self) -> tuple[float, ...]:
        """Individual wall-normal cell heights, from the wall outwards."""
        return tuple(
            self.first_cell_height * self.growth_ratio**layer for layer in range(self.layers)
        )

    @property
    def cumulative_heights(self) -> tuple[float, ...]:
        """Cumulative extrusion distances expected by advancing-layer meshers."""
        return tuple(np.cumsum(np.asarray(self.layer_heights, dtype=np.float64)).tolist())


def _patch_rows(mesh: dict, patch_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return faces and owners for one named boundary patch."""
    patch = next(item for item in mesh["boundary"] if item["name"] == patch_name)
    start = int(patch["start_face"])
    stop = start + int(patch["n_faces"])
    return np.asarray(mesh["faces"])[start:stop], np.asarray(mesh["owners"])[start:stop]


def _morton_order(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Return a deterministic spatial cell order for compatibility partitioning."""
    centres = np.empty((len(cells), 3), dtype=np.float64)
    for start in range(0, len(cells), 50_000):
        stop = min(start + 50_000, len(cells))
        centres[start:stop] = points[cells[start:stop]].mean(axis=1)
    lower = centres.min(axis=0)
    extent = np.maximum(centres.max(axis=0) - lower, 1.0e-30)
    coordinates = np.clip(np.floor((centres - lower) / extent * 1023.0), 0, 1023).astype(np.uint64)

    def split(value: np.ndarray) -> np.ndarray:
        value = value & np.uint64(0x3FF)
        value = (value | value << np.uint64(16)) & np.uint64(0x30000FF)
        value = (value | value << np.uint64(8)) & np.uint64(0x300F00F)
        value = (value | value << np.uint64(4)) & np.uint64(0x30C30C3)
        value = (value | value << np.uint64(2)) & np.uint64(0x9249249)
        return value

    code = (
        split(coordinates[:, 0])
        | split(coordinates[:, 1]) << np.uint64(1)
        | split(coordinates[:, 2]) << np.uint64(2)
    )
    return np.argsort(code, kind="stable")


def _reorder_cells_for_partitioning(mesh: dict) -> None:
    """Renumber cells spatially while preserving face and patch ordering."""
    cells = np.asarray(mesh["cell_vertex_indices"], dtype=np.int32)
    order = _morton_order(np.asarray(mesh["vertex_position"]), cells)
    new_id = np.empty(len(order), dtype=np.int32)
    new_id[order] = np.arange(len(order), dtype=np.int32)
    mesh["owners"] = np.ascontiguousarray(new_id[np.asarray(mesh["owners"])], dtype=np.int32)
    mesh["neighbours"] = np.ascontiguousarray(
        new_id[np.asarray(mesh["neighbours"])], dtype=np.int32
    )
    for name in (
        "cell_vertex_indices",
        "cell_type_code",
        "cell_levels",
        "cell_sizes",
        "boundary_layer_index",
    ):
        mesh[name] = np.ascontiguousarray(np.asarray(mesh[name])[order])
    mesh["mesh_generation"]["cell_order"] = "morton"


def _orient_native_faces(mesh: dict) -> None:
    """Orient every stitched face from its owner toward its neighbour/patch."""
    points = np.asarray(mesh["vertex_position"], dtype=np.float64)
    faces = np.asarray(mesh["faces"])
    cells = np.asarray(mesh["cell_vertex_indices"], dtype=np.int32)
    centres = points[cells].mean(axis=1)
    n_internal = int(mesh["n_interior_faces"])
    owners = np.asarray(mesh["owners"], dtype=np.int64)
    neighbours = np.asarray(mesh["neighbours"], dtype=np.int64)
    for face_id, face in enumerate(faces):
        coordinates = points[np.asarray(face, dtype=np.int64)]
        face_centre = coordinates.mean(axis=0)
        area = np.zeros(3, dtype=np.float64)
        for index in range(len(coordinates)):
            area += 0.5 * np.cross(
                coordinates[index] - face_centre,
                coordinates[(index + 1) % len(coordinates)] - face_centre,
            )
        owner = int(owners[face_id])
        direction = (
            centres[int(neighbours[face_id])] - centres[owner]
            if face_id < n_internal
            else face_centre - centres[owner]
        )
        if float(np.dot(area, direction)) < 0.0:
            faces[face_id] = face[::-1]


def stitch_boundary_layer(outer: dict, layer: dict, interface_patch_name: str) -> dict:
    """Join a generic layer block to its exactly matching Cartesian hole."""
    outer_points = np.asarray(outer["vertex_position"], dtype=np.float64)
    layer_points = np.asarray(layer["vertex_position"], dtype=np.float64)
    scale = max(float(np.ptp(outer_points, axis=0).max()), 1.0)
    tolerance = 1.0e-10 * scale

    interface_ids = np.asarray(layer.pop("interface_point_ids"), dtype=np.int64)
    interface_set = np.zeros(len(layer_points), dtype=bool)
    interface_set[interface_ids] = True
    outer_key_to_id = {
        tuple(np.rint(point / tolerance).astype(np.int64)): index
        for index, point in enumerate(outer_points)
    }
    layer_point_map = np.empty(len(layer_points), dtype=np.int32)
    for point_id in interface_ids:
        key = tuple(np.rint(layer_points[point_id] / tolerance).astype(np.int64))
        if key not in outer_key_to_id:
            raise ValueError("Boundary-layer interface point is absent from Cartesian mesh")
        layer_point_map[point_id] = outer_key_to_id[key]
    new_ids = np.flatnonzero(~interface_set)
    layer_point_map[new_ids] = np.arange(
        len(outer_points), len(outer_points) + len(new_ids), dtype=np.int32
    )
    points = np.vstack((outer_points, layer_points[new_ids]))
    layer_faces = layer_point_map[np.asarray(layer["faces"], dtype=np.int32)]
    layer_cells = layer_point_map[np.asarray(layer["cell_vertex_indices"], dtype=np.int32)]

    outer_interface_faces, outer_interface_owners = _patch_rows(outer, interface_patch_name)
    layer_interface_faces, layer_interface_owners = _patch_rows(
        {**layer, "faces": layer_faces}, interface_patch_name
    )
    layer_by_signature = {
        tuple(sorted(map(int, face))): (face, int(owner))
        for face, owner in zip(layer_interface_faces, layer_interface_owners, strict=True)
    }
    if len(layer_by_signature) != len(layer_interface_faces):
        raise ValueError("Boundary-layer interface contains duplicate faces")
    matched_layer_owners = np.empty(len(outer_interface_faces), dtype=np.int32)
    for index, face in enumerate(outer_interface_faces):
        match = layer_by_signature.pop(tuple(sorted(map(int, face))), None)
        if match is None:
            raise ValueError("Cartesian and boundary-layer interface faces do not match")
        matched_layer_owners[index] = match[1]
    if layer_by_signature:
        raise ValueError("Boundary-layer interface has unmatched faces")

    outer_n_internal = int(outer["n_interior_faces"])
    layer_n_internal = int(layer["n_interior_faces"])
    cell_offset = int(outer["n_cells"])
    internal_faces = np.vstack(
        (
            np.asarray(outer["faces"][:outer_n_internal], dtype=np.int32),
            layer_faces[:layer_n_internal],
            np.asarray(outer_interface_faces, dtype=np.int32),
        )
    )
    internal_owners = np.concatenate(
        (
            np.asarray(outer["owners"][:outer_n_internal], dtype=np.int32),
            np.asarray(layer["owners"][:layer_n_internal], dtype=np.int32) + cell_offset,
            np.asarray(outer_interface_owners, dtype=np.int32),
        )
    )
    internal_neighbours = np.concatenate(
        (
            np.asarray(outer["neighbours"], dtype=np.int32),
            np.asarray(layer["neighbours"], dtype=np.int32) + cell_offset,
            matched_layer_owners + cell_offset,
        )
    )

    patch_order = []
    patch_data: dict[str, dict[str, list[np.ndarray] | str]] = {}

    def collect(mesh: dict, faces: np.ndarray, owner_offset: int) -> None:
        for patch in mesh["boundary"]:
            name = str(patch["name"])
            if name == interface_patch_name:
                continue
            if name not in patch_data:
                patch_order.append(name)
                patch_data[name] = {
                    "type": str(patch.get("type", "patch")),
                    "faces": [],
                    "owners": [],
                }
            start = int(patch["start_face"])
            stop = start + int(patch["n_faces"])
            face_blocks = patch_data[name]["faces"]
            owner_blocks = patch_data[name]["owners"]
            if not isinstance(face_blocks, list) or not isinstance(owner_blocks, list):
                raise RuntimeError("Boundary patch storage is internally inconsistent")
            face_blocks.append(np.asarray(faces[start:stop], dtype=np.int32))
            owner_blocks.append(
                np.asarray(mesh["owners"][start:stop], dtype=np.int32) + owner_offset
            )

    collect(outer, np.asarray(outer["faces"]), 0)
    collect(layer, layer_faces, cell_offset)
    face_blocks = [internal_faces]
    owner_blocks = [internal_owners]
    boundary = []
    start_face = len(internal_faces)
    for name in patch_order:
        entry = patch_data[name]
        entry_faces = entry["faces"]
        entry_owners = entry["owners"]
        if not isinstance(entry_faces, list) or not isinstance(entry_owners, list):
            raise RuntimeError("Boundary patch storage is internally inconsistent")
        faces = np.vstack(entry_faces)
        owners = np.concatenate(entry_owners)
        face_blocks.append(faces)
        owner_blocks.append(owners)
        boundary.append(
            {
                "name": name,
                "start_face": start_face,
                "n_faces": len(faces),
                "type": entry["type"],
            }
        )
        start_face += len(faces)

    cells = np.vstack(
        (
            np.asarray(outer["cell_vertex_indices"], dtype=np.int32),
            layer_cells,
        )
    )
    n_cells = len(cells)
    generation = dict(outer["mesh_generation"])
    generation["method"] = "cartesian_with_patch_normal_layers"
    generation["boundary_layer"] = layer["mesh_generation"]
    result = {
        "vertex_position": np.ascontiguousarray(points),
        "faces": np.ascontiguousarray(np.vstack(face_blocks), dtype=np.int32),
        "owners": np.ascontiguousarray(np.concatenate(owner_blocks), dtype=np.int32),
        "neighbours": np.ascontiguousarray(internal_neighbours, dtype=np.int32),
        "boundary": boundary,
        "n_cells": n_cells,
        "n_faces": start_face,
        "n_interior_faces": len(internal_faces),
        "n_points": len(points),
        "cell_vertex_indices": np.ascontiguousarray(cells),
        "cell_type_code": np.full(n_cells, 5, dtype=np.int32),
        "cell_levels": np.concatenate(
            (
                np.asarray(outer["cell_levels"], dtype=np.int8),
                np.full(int(layer["n_cells"]), np.max(outer["cell_levels"]), dtype=np.int8),
            )
        ),
        "cell_sizes": np.concatenate(
            (
                np.asarray(outer["cell_sizes"], dtype=np.float32),
                np.asarray(layer["cell_sizes"], dtype=np.float32),
            )
        ),
        "boundary_layer_index": np.concatenate(
            (
                np.full(int(outer["n_cells"]), -1, dtype=np.int16),
                np.asarray(layer["boundary_layer_index"], dtype=np.int16),
            )
        ),
        "mesh_generation": generation,
    }
    _orient_native_faces(result)
    _reorder_cells_for_partitioning(result)
    return result


__all__ = ["BoundaryLayerSpec", "stitch_boundary_layer"]
