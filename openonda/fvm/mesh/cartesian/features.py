# SPDX-License-Identifier: GPL-3.0-or-later
"""Deterministic feature classification primitives."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True, slots=True)
class SurfaceFeature:
    """A sharp edge represented by its endpoints and adjacent face IDs."""

    start: tuple[float, float, float]
    end: tuple[float, float, float]
    face_ids: tuple[int, int]
    angle: float


@dataclass(frozen=True, slots=True)
class FeatureSet:
    """Classified sharp edges and topological corners."""

    edges: tuple[SurfaceFeature, ...]
    corners: tuple[tuple[float, float, float], ...]


def classify_features(triangles: np.ndarray, angle_degrees: float) -> FeatureSet:
    """Classify manifold edges whose adjacent face angle exceeds a threshold."""
    triangles = np.asarray(triangles, dtype=np.float64)
    angle_degrees = float(angle_degrees)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError("triangles must have shape (n, 3, 3)")
    if not math.isfinite(angle_degrees) or not 0.0 < angle_degrees < 180.0:
        raise ValueError("angle_degrees must be strictly between 0 and 180")

    scale = max(float(np.ptp(triangles, axis=(0, 1)).max()), 1.0)
    tolerance = scale * 1.0e-9
    quantized = np.rint(triangles.reshape(-1, 3) / tolerance).astype(np.int64)
    _, ids = np.unique(quantized, axis=0, return_inverse=True)
    ids = ids.reshape(-1, 3)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    normals = normals / lengths[:, None]
    edge_faces: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for face_id, face in enumerate(ids):
        for start, end in zip(face, np.roll(face, -1), strict=True):
            key = (min(int(start), int(end)), max(int(start), int(end)))
            edge_faces.setdefault(key, []).append((face_id, int(start), int(end)))

    features: list[SurfaceFeature] = []
    corner_ids: set[int] = set()
    vertices = np.asarray(triangles).reshape(-1, 3)
    unique_vertices = np.zeros((int(ids.max()) + 1, 3), dtype=np.float64)
    for vertex_id in range(len(unique_vertices)):
        unique_vertices[vertex_id] = vertices[np.flatnonzero(ids.reshape(-1) == vertex_id)[0]]
    for edge, records in sorted(edge_faces.items()):
        if len(records) != 2:
            continue
        left, right = records
        cosine = float(np.clip(np.dot(normals[left[0]], normals[right[0]]), -1.0, 1.0))
        angle = math.degrees(math.acos(cosine))
        if angle >= angle_degrees:
            features.append(
                SurfaceFeature(
                    tuple(unique_vertices[edge[0]].tolist()),
                    tuple(unique_vertices[edge[1]].tolist()),
                    (left[0], right[0]),
                    angle,
                )
            )
            corner_ids.update(edge)
    return FeatureSet(
        tuple(features),
        tuple(tuple(point.tolist()) for point in unique_vertices[sorted(corner_ids)]),
    )


__all__ = ["FeatureSet", "SurfaceFeature", "classify_features"]
