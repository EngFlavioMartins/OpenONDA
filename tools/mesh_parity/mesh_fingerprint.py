"""Numbering-independent topology fingerprints for polyhedral meshes."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from .openfoam_poly_mesh import PolyMesh


def _histogram(values: np.ndarray | list[int]) -> dict[str, int]:
    counts = Counter(int(value) for value in values)
    return {str(key): counts[key] for key in sorted(counts)}


def cell_face_counts(mesh: PolyMesh) -> np.ndarray:
    """Return the number of incident faces for every cell."""
    result = np.bincount(mesh.owner, minlength=mesh.n_cells).astype(np.int64, copy=False)
    if mesh.n_internal_faces:
        result += np.bincount(mesh.neighbour, minlength=mesh.n_cells)
    return result


def internal_face_pairs(mesh: PolyMesh) -> np.ndarray:
    """Return sorted, de-duplicated cell-pair edges of the dual graph."""
    if not mesh.n_internal_faces:
        return np.empty((0, 2), dtype=np.int64)
    pairs = np.column_stack((mesh.owner[: mesh.n_internal_faces], mesh.neighbour))
    pairs.sort(axis=1)
    return np.unique(pairs, axis=0)


def cell_neighbour_counts(mesh: PolyMesh) -> np.ndarray:
    """Return distinct adjacent-cell counts, robust to duplicate internal faces."""
    pairs = internal_face_pairs(mesh)
    if not len(pairs):
        return np.zeros(mesh.n_cells, dtype=np.int64)
    return np.bincount(pairs.ravel(), minlength=mesh.n_cells).astype(np.int64, copy=False)


def _connected_components(n_cells: int, pairs: np.ndarray) -> int:
    """Count components of the cell dual graph with a compact union-find."""
    if n_cells == 0:
        return 0
    parent = np.arange(n_cells, dtype=np.int64)
    size = np.ones(n_cells, dtype=np.int64)

    def find(item: int) -> int:
        root = item
        while parent[root] != root:
            root = int(parent[root])
        while parent[item] != item:
            next_item = int(parent[item])
            parent[item] = root
            item = next_item
        return root

    components = n_cells
    for first, second in pairs:
        root_first = find(int(first))
        root_second = find(int(second))
        if root_first == root_second:
            continue
        if size[root_first] < size[root_second]:
            root_first, root_second = root_second, root_first
        parent[root_second] = root_first
        size[root_first] += size[root_second]
        components -= 1
    return components


@dataclass(frozen=True, slots=True)
class MeshFingerprint:
    """The mandatory Level-A parity invariants from the completion plan."""

    n_cells: int
    n_faces: int
    n_internal_faces: int
    n_boundary_faces: int
    n_points: int
    patches: dict[str, dict[str, int | str]]
    face_vertex_histogram: dict[str, int]
    cell_face_histogram: dict[str, int]
    cell_neighbour_histogram: dict[str, int]
    connected_components: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready canonical representation."""
        return {
            "n_cells": self.n_cells,
            "n_faces": self.n_faces,
            "n_internal_faces": self.n_internal_faces,
            "n_boundary_faces": self.n_boundary_faces,
            "n_points": self.n_points,
            "patches": self.patches,
            "face_vertex_histogram": self.face_vertex_histogram,
            "cell_face_histogram": self.cell_face_histogram,
            "cell_neighbour_histogram": self.cell_neighbour_histogram,
            "connected_components": self.connected_components,
        }


def fingerprint_mesh(mesh: PolyMesh) -> MeshFingerprint:
    """Build an exact, numbering-independent topology fingerprint."""
    pairs = internal_face_pairs(mesh)
    patches = {
        patch.name: {"type": patch.type, "n_faces": patch.n_faces}
        for patch in sorted(mesh.boundary, key=lambda item: item.name)
    }
    return MeshFingerprint(
        n_cells=mesh.n_cells,
        n_faces=mesh.n_faces,
        n_internal_faces=mesh.n_internal_faces,
        n_boundary_faces=mesh.n_boundary_faces,
        n_points=len(mesh.points),
        patches=patches,
        face_vertex_histogram=_histogram([len(face) for face in mesh.faces]),
        cell_face_histogram=_histogram(cell_face_counts(mesh)),
        cell_neighbour_histogram=_histogram(cell_neighbour_counts(mesh)),
        connected_components=_connected_components(mesh.n_cells, pairs),
    )


def fingerprint_differences(
    reference: MeshFingerprint, candidate: MeshFingerprint
) -> dict[str, dict[str, Any]]:
    """Return all Level-A disagreements, keyed by invariant name."""
    reference_dict = reference.to_dict()
    candidate_dict = candidate.to_dict()
    return {
        key: {"cfmesh": reference_dict[key], "openonda": candidate_dict[key]}
        for key in reference_dict
        if reference_dict[key] != candidate_dict[key]
    }


__all__ = [
    "MeshFingerprint",
    "cell_face_counts",
    "cell_neighbour_counts",
    "fingerprint_differences",
    "fingerprint_mesh",
    "internal_face_pairs",
]
