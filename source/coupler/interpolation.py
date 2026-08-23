"""Local Taylor interpolation of cell-centred FVM velocity."""

from __future__ import annotations

from collections import OrderedDict
import hashlib

import numpy as np


class FVMVelocityInterpolator:
    """Second-order local Taylor interpolation on arbitrary cell centres.

    Exact donor gradients reproduce affine fields exactly. Smooth fields are
    second-order accurate when the supplied FVM gradients are consistent.
    """

    def __init__(self, cell_centre: np.ndarray, tree, neighbour_count: int = 4):
        self.cell_centre = np.asarray(cell_centre, dtype=np.float64).reshape(-1, 3)
        self.tree = tree
        self.neighbour_count = min(max(int(neighbour_count), 1), len(self.cell_centre))
        self._cache: OrderedDict[bytes, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    @staticmethod
    def _key(evaluation_position: np.ndarray) -> bytes:
        array = np.ascontiguousarray(evaluation_position, dtype=np.float64)
        digest = hashlib.blake2b(array.tobytes(), digest_size=16)
        return digest.digest()

    def _stencil(self, evaluation_position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        key = self._key(evaluation_position)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        distance, indices = self.tree.query(evaluation_position, k=self.neighbour_count, workers=-1)
        distance = np.asarray(distance, dtype=np.float64).reshape(
            len(evaluation_position), self.neighbour_count
        )
        indices = np.asarray(indices, dtype=np.int32).reshape(
            len(evaluation_position), self.neighbour_count
        )

        weights = 1.0 / np.maximum(distance, 1.0e-12) ** 2
        exact = distance[:, 0] <= 1.0e-12
        if exact.any():
            weights[exact] = 0.0
            weights[exact, 0] = 1.0
        weights /= weights.sum(axis=1, keepdims=True)
        result = indices, weights

        self._cache[key] = result
        self._cache.move_to_end(key)
        while len(self._cache) > 6:
            self._cache.popitem(last=False)
        return result

    def sample(
        self,
        evaluation_position: np.ndarray,
        velocity: np.ndarray,
        gradient: np.ndarray,
        chunk_size: int = 100_000,
    ) -> np.ndarray:
        evaluation_position = np.ascontiguousarray(evaluation_position, dtype=np.float64).reshape(
            -1, 3
        )
        velocity = np.asarray(velocity, dtype=np.float64).reshape(-1, 3)
        gradient = np.asarray(gradient, dtype=np.float64).reshape(-1, 3, 3)
        indices, weights = self._stencil(evaluation_position)
        sampled = np.empty((len(evaluation_position), 3), dtype=np.float64)

        for start in range(0, len(evaluation_position), chunk_size):
            stop = min(start + chunk_size, len(evaluation_position))
            local_indices = indices[start:stop]
            delta = evaluation_position[start:stop, None, :] - self.cell_centre[local_indices]
            reconstructed = velocity[local_indices] + np.einsum(
                "mki,mkij->mkj", delta, gradient[local_indices], optimize=True
            )
            sampled[start:stop] = np.einsum(
                "mk,mkj->mj", weights[start:stop], reconstructed, optimize=True
            )
        return sampled


__all__ = ["FVMVelocityInterpolator"]
