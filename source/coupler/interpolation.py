"""Local Taylor interpolation of cell-centred FVM velocity."""

from __future__ import annotations

from collections import OrderedDict
import hashlib

import numpy as np
from scipy.spatial import cKDTree


class FVMVelocityInterpolator:
    """Second-order local Taylor interpolation on arbitrary cell centres.

    Exact donor gradients reproduce affine fields exactly. Smooth fields are
    second-order accurate when the supplied FVM gradients are consistent. The
    coupler constructs this helper internally for velocity traces needed by
    buffered renewal.

    Parameters
    ----------
    cell_centre : ndarray, shape (M, 3)
        Cartesian FVM donor-cell centres in m. Values are converted to a
        ``float64`` view/copy as required by NumPy.
    tree : scipy.spatial.cKDTree
        Search tree built from exactly ``cell_centre`` in the same order.
    neighbour_count : int, default=4
        Minimum number of nearest donors in the inverse-distance blend. It is
        clamped to ``[1, M]``. All donors tied at the last distance are included
        so cell ordering does not select a preferred spatial direction.

    Attributes
    ----------
    cell_centre : ndarray, shape (M, 3)
        Normalized donor coordinates in m.
    neighbour_count : int
        Minimum stencil size after clamping; distance ties can increase it.

    Notes
    -----
    Stencil indices/weights are cached for the six most recent exact target
    arrays. Cached arrays are internal and never returned for mutation.
    """

    def __init__(
        self,
        cell_centre: np.ndarray,
        tree: cKDTree,
        neighbour_count: int = 4,
    ) -> None:
        """Create an interpolator over one donor geometry and search index.

        Parameters
        ----------
        cell_centre : ndarray, shape (M, 3)
            FVM cell-centre coordinates in metres. The array is normalized to
            float64 and retained as the donor row ordering.
        tree : scipy.spatial.cKDTree
            Search tree built from the same ``M`` rows, in the same order.
        neighbour_count : int, default=4
            Minimum number of nearest donor cells used for each query, clipped
            to ``[1, M]``. The last equal-distance shell is included in full.

        Notes
        -----
        The tree and donor geometry are retained rather than rebuilt. Do not
        reuse this object after changing the FVM mesh because cached indices
        would refer to the previous cell ordering.
        """
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
        """Return nearest-donor indices and normalized inverse-square weights.

        Parameters
        ----------
        evaluation_position : ndarray, shape (N, 3)
            Cartesian query points in metres.

        Returns
        -------
        tuple of ndarray
            ``(indices, weights)`` with shapes ``(N, K)`` and ``(N, K)``;
            ``K`` includes the widest tied donor shell in this target array.
            Shorter stencils have zero-weight padding. Each weight row sums
            to one. These internal cache entries must not be modified.
        """
        key = self._key(evaluation_position)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        # Ask for one additional donor to detect a truncated distance shell.
        # Cartesian cells frequently tie: choosing an arbitrary subset then
        # changes reconstructed curvature under cell reordering/reflection.
        count = self.neighbour_count
        query_count = min(count + 1, len(self.cell_centre))
        distance, indices = self.tree.query(evaluation_position, k=query_count, workers=-1)
        distance = np.asarray(distance, dtype=np.float64).reshape(
            len(evaluation_position), query_count
        )
        indices = np.asarray(indices, dtype=np.int32).reshape(len(evaluation_position), query_count)

        tied_rows = np.empty(0, dtype=np.int64)
        if query_count > count:
            cutoff = distance[:, count - 1]
            tolerance = 64.0 * np.finfo(float).eps * np.maximum(1.0, cutoff)
            tied_rows = np.flatnonzero(
                (distance[:, count] <= cutoff + tolerance) & (distance[:, 0] > 1.0e-12)
            )
        if len(tied_rows):
            neighbours = self.tree.query_ball_point(
                evaluation_position[tied_rows],
                r=cutoff[tied_rows] + tolerance[tied_rows],
                workers=-1,
            )
            width = max(count, max(len(rows) for rows in neighbours))
            complete_distance = np.full((len(evaluation_position), width), np.inf)
            complete_indices = np.zeros((len(evaluation_position), width), dtype=np.int32)
            complete_distance[:, :count] = distance[:, :count]
            complete_indices[:, :count] = indices[:, :count]
            for row, donors in zip(tied_rows, neighbours, strict=True):
                donors = np.asarray(donors, dtype=np.int32)
                complete_indices[row, : len(donors)] = donors
                complete_distance[row, : len(donors)] = np.linalg.norm(
                    evaluation_position[row] - self.cell_centre[donors], axis=1
                )
            distance, indices = complete_distance, complete_indices
        else:
            distance, indices = distance[:, :count], indices[:, :count]

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
        """Reconstruct velocity at arbitrary points from values and gradients.

        Parameters
        ----------
        evaluation_position : ndarray, shape (N, 3)
            Cartesian target positions in m.
        velocity : ndarray, shape (M, 3)
            Donor cell-centred velocity in m/s.
        gradient : ndarray, shape (M, 3, 3)
            Donor velocity gradient in 1/s with
            ``gradient[i, j] = d(velocity_j)/d(x_i)``.
        chunk_size : int, default=100000
            Positive target rows reconstructed per temporary batch.

        Returns
        -------
        ndarray, shape (N, 3)
            New ``float64`` target-velocity array in m/s.

        Notes
        -----
        For donor ``k`` the Taylor value is
        ``u_k + (x_target - x_k) dot grad(u)_k``. Neighbour values are then
        inverse-square-distance blended. Inputs are not modified.

        Raises
        ------
        ValueError
            If the velocity or gradient donor count is inconsistent with the
            cell-centre count.
        """
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

    def sample_cell_field(
        self,
        evaluation_position: np.ndarray,
        field: np.ndarray,
        chunk_size: int = 100_000,
    ) -> np.ndarray:
        """Interpolate a cell-centred vector field without donor gradients.

        Parameters
        ----------
        evaluation_position : ndarray, shape (N, 3)
            Cartesian target positions in m.
        field : ndarray, shape (M, 3)
            Donor vector values. Units are preserved in the output.
        chunk_size : int, default=100000
            Positive target rows evaluated per batch.

        Returns
        -------
        ndarray, shape (N, 3)
            New ``float64`` interpolated field in the donor field's units.

        Notes
        -----
        Constant fields are reproduced exactly. The donor value is taken as it
        stands, so a field the FVM already differentiated is not differentiated
        a second time on the coupling lattice. Inputs are not modified.
        """
        evaluation_position = np.ascontiguousarray(evaluation_position, dtype=np.float64).reshape(
            -1, 3
        )
        field = np.asarray(field, dtype=np.float64).reshape(-1, 3)
        indices, weights = self._stencil(evaluation_position)
        sampled = np.empty((len(evaluation_position), 3), dtype=np.float64)

        for start in range(0, len(evaluation_position), chunk_size):
            stop = min(start + chunk_size, len(evaluation_position))
            sampled[start:stop] = np.einsum(
                "mk,mkj->mj",
                weights[start:stop],
                field[indices[start:stop]],
                optimize=True,
            )
        return sampled


__all__ = ["FVMVelocityInterpolator"]
