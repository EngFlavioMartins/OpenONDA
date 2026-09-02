# SPDX-License-Identifier: GPL-3.0-or-later
"""Validated surface loading and deterministic topology diagnostics."""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from ..triangulated_surface import TriangulatedSurface


def _quantized_triangles(surface: TriangulatedSurface) -> np.ndarray:
    scale = max(float(np.ptp(surface.triangles, axis=(0, 1)).max()), 1.0)
    tolerance = scale * 1.0e-9
    quantized = np.rint(surface.triangles.reshape(-1, 3) / tolerance).astype(np.int64)
    _, ids = np.unique(quantized, axis=0, return_inverse=True)
    return ids.reshape(-1, 3)


def _edge_records(triangle_ids: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int, int]]]:
    records: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for triangle_id, triangle in enumerate(triangle_ids):
        for start, end in zip(triangle, np.roll(triangle, -1), strict=True):
            key = (min(int(start), int(end)), max(int(start), int(end)))
            records[key].append((triangle_id, int(start), int(end)))
    return records


def _signed_component_volumes(triangles: np.ndarray, triangle_ids: np.ndarray) -> list[float]:
    records = _edge_records(triangle_ids)
    neighbours: dict[int, set[int]] = defaultdict(set)
    for values in records.values():
        if len(values) == 2:
            left, right = values
            neighbours[left[0]].add(right[0])
            neighbours[right[0]].add(left[0])
    components: list[list[int]] = []
    unseen = set(range(len(triangles)))
    while unseen:
        root = min(unseen)
        queue = deque((root,))
        unseen.remove(root)
        component = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbour in sorted(neighbours[current]):
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    queue.append(neighbour)
        components.append(component)
    volumes = []
    for component in components:
        values = triangles[np.asarray(component, dtype=np.int64)]
        volumes.append(
            float(
                np.einsum("ij,ij->i", values[:, 0], np.cross(values[:, 1], values[:, 2])).sum()
                / 6.0
            )
        )
    return volumes


def validate_surface_orientation(surface: TriangulatedSurface) -> None:
    """Reject inconsistent edge winding and disconnected mixed orientation."""
    triangle_ids = _quantized_triangles(surface)
    records = _edge_records(triangle_ids)
    for edge, values in sorted(records.items()):
        if len(values) != 2:
            raise ValueError(f"Surface edge {edge} has {len(values)} incident triangles")
        first, second = values
        if first[1:] == second[1:]:
            raise ValueError(f"Surface edge {edge} has inconsistent triangle orientation")
    volumes = _signed_component_volumes(surface.triangles, triangle_ids)
    nonzero = [volume for volume in volumes if abs(volume) > np.finfo(np.float64).eps]
    if not nonzero:
        raise ValueError("Surface components have zero enclosed volume")
    signs = {volume > 0.0 for volume in nonzero}
    if len(signs) > 1:
        raise ValueError("Disconnected surface components have inconsistent orientation")


def load_surface(path: str | Path) -> TriangulatedSurface:
    """Load an STL, validate its winding, and preserve the exact source hash."""
    surface = TriangulatedSurface.from_stl(path)
    validate_surface_orientation(surface)
    return surface


__all__ = ["load_surface", "validate_surface_orientation"]
