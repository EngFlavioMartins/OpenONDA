"""Cached interpolation of the FVM velocity trace."""

from __future__ import annotations

from collections import OrderedDict
import hashlib

import numpy as np


class CachedVelocityTrace:
    """Smooth cell-centred Taylor interpolation with cached neighbour stencils."""

    def __init__(self, cell_centres: np.ndarray, tree, neighbours: int = 4):
        self.cell_centres = np.asarray(cell_centres, dtype=np.float64).reshape(-1, 3)
        self.tree = tree
        self.neighbours = min(max(int(neighbours), 1), len(self.cell_centres))
        self._cache: OrderedDict[bytes, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    @staticmethod
    def _key(points: np.ndarray) -> bytes:
        array = np.ascontiguousarray(points, dtype=np.float64)
        digest = hashlib.blake2b(array.tobytes(), digest_size=16)
        return digest.digest()

    def _stencil(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        key = self._key(points)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        distance, indices = self.tree.query(points, k=self.neighbours, workers=-1)
        distance = np.asarray(distance, dtype=np.float64).reshape(len(points), self.neighbours)
        indices = np.asarray(indices, dtype=np.int32).reshape(len(points), self.neighbours)

        weights = 1.0 / np.maximum(distance, 1.0e-12) ** 2
        exact = distance[:, 0] <= 1.0e-12
        if exact.any():
            weights[exact] = 0.0
            weights[exact, 0] = 1.0
        weights /= weights.sum(axis=1, keepdims=True)
        result = indices, weights.astype(np.float32)

        self._cache[key] = result
        self._cache.move_to_end(key)
        while len(self._cache) > 6:
            self._cache.popitem(last=False)
        return result

    def sample(
        self,
        points: np.ndarray,
        velocity: np.ndarray,
        gradient: np.ndarray,
        chunk_size: int = 100_000,
    ) -> np.ndarray:
        points = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)
        velocity = np.asarray(velocity, dtype=np.float64).reshape(-1, 3)
        gradient = np.asarray(gradient, dtype=np.float64).reshape(-1, 3, 3)
        indices, weights = self._stencil(points)
        sampled = np.empty((len(points), 3), dtype=np.float64)

        for start in range(0, len(points), chunk_size):
            stop = min(start + chunk_size, len(points))
            local_indices = indices[start:stop]
            delta = points[start:stop, None, :] - self.cell_centres[local_indices]
            reconstructed = velocity[local_indices] + np.einsum(
                "mki,mkij->mkj", delta, gradient[local_indices], optimize=True
            )
            sampled[start:stop] = np.einsum(
                "mk,mkj->mj", weights[start:stop], reconstructed, optimize=True
            )
        return sampled


__all__ = ["CachedVelocityTrace"]
