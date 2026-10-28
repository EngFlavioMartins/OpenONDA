"""Reconstruction of the FVM velocity from cell values and gradients on an interpolation stencil."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib

import numpy as np
from scipy.spatial import cKDTree

from openonda.core._numba import cacheable_njit as njit

_INTERPOLATION_BATCH_SIZE = 100_000


@njit(cache=True, fastmath=False)
def _sample_velocity(position, cell_centre, velocity, gradient, indices, weights, sampled):
    """Blend the Taylor values of the stencil cells; a uniform velocity is kept exactly."""
    for row in range(len(position)):
        for component in range(3):
            reference = velocity[indices[row, 0], component]
            value = 0.0
            for neighbour in range(indices.shape[1]):
                weight = weights[row, neighbour]
                if weight == 0.0:
                    continue
                cell = indices[row, neighbour]
                increment = 0.0
                for axis in range(3):
                    increment += (position[row, axis] - cell_centre[cell, axis]) * gradient[
                        cell, axis, component
                    ]
                value += weight * (velocity[cell, component] - reference + increment)
            sampled[row, component] = reference + value


@dataclass(frozen=True)
class InterpolationStencil:
    """Fixed interpolation stencil ``𝒮(x)`` of a fixed set of target points.

    The stencil geometry is computed once; each evaluation uses the current FVM
    fields, including those of every coupling iteration. Build it with
    :meth:`FVMVelocityInterpolator.prepare` once the mesh is final.
    """

    position: np.ndarray
    cell_centre: np.ndarray
    indices: np.ndarray
    weights: np.ndarray

    def sample(self, velocity: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """Reconstruct the current FVM velocity without a geometry search.

        Parameters
        ----------
        velocity : ndarray, shape (M, 3)
            Current cell-centred FVM velocity in m/s.
        gradient : ndarray, shape (M, 3, 3)
            Current FVM velocity gradient, indexed as d(u_j)/d(x_i), in 1/s.

        Returns
        -------
        ndarray, shape (N, 3)
            Independent reconstructed velocities in m/s.
        """
        velocity = np.asarray(velocity, dtype=np.float64).reshape(-1, 3)
        gradient = np.asarray(gradient, dtype=np.float64).reshape(-1, 3, 3)
        if len(velocity) != len(self.cell_centre) or len(gradient) != len(self.cell_centre):
            raise ValueError("Velocity and gradient must have one row per FVM cell")
        sampled = np.empty((len(self.position), 3), dtype=np.float64)
        _sample_velocity(
            self.position,
            self.cell_centre,
            velocity,
            gradient,
            self.indices,
            self.weights,
            sampled,
        )
        return sampled


class FVMVelocityInterpolator:
    """Second-order local Taylor interpolation on arbitrary cell centres.

    Exact cell gradients reproduce affine fields exactly. Smooth fields are
    second-order accurate when the supplied FVM gradients are consistent. The
    vorticity transfer uses it for the reconstructed FVM velocity ``ũ^FVM`` on the
    faces of the lattice cubes.

    Parameters
    ----------
    cell_centre : ndarray, shape (M, 3)
        Cartesian FVM cell centres in m. Values are converted to a
        ``float64`` view/copy as required by NumPy.
    tree : scipy.spatial.cKDTree
        Search tree built from exactly ``cell_centre`` in the same order.
    neighbour_count : int, default=4
        Minimum number of nearest stencil cells in the inverse-distance blend. It is
        clamped to ``[1, M]``. All cells tied at the last distance are included
        so cell ordering does not select a preferred spatial direction.

    Attributes
    ----------
    cell_centre : ndarray, shape (M, 3)
        Normalized cell coordinates in m.
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
        *,
        solid_boundary=None,
    ) -> None:
        """Create an interpolator over one cell geometry and search index.

        Parameters
        ----------
        cell_centre : ndarray, shape (M, 3)
            FVM cell-centre coordinates in metres. The array is normalized to
            float64 and retained as the cell row ordering.
        tree : scipy.spatial.cKDTree
            Search tree built from the same ``M`` rows, in the same order.
        neighbour_count : int, default=4
            Minimum number of nearest stencil cells for each query, clipped
            to ``[1, M]``. The last equal-distance shell is included in full.

        Notes
        -----
        The tree and cell geometry are retained rather than rebuilt. Do not
        reuse this object after changing the FVM mesh because cached indices
        would refer to the previous cell ordering.
        """
        self.cell_centre = np.asarray(cell_centre, dtype=np.float64).reshape(-1, 3)
        self.tree = tree
        self._cell_centroid = self.cell_centre.mean(axis=0)
        relative = self.cell_centre - self._cell_centroid
        _, singular_values, basis = np.linalg.svd(relative, full_matrices=False)
        tolerance = max(relative.shape) * np.finfo(float).eps * singular_values.max(initial=0)
        resolved = basis[singular_values > tolerance]
        self._distance_projection = None if len(resolved) == 3 else resolved.T @ resolved
        self.neighbour_count = min(max(int(neighbour_count), 1), len(self.cell_centre))
        self.solid_boundary = solid_boundary
        self._fluid_cells = (
            None if solid_boundary is None else ~solid_boundary.contains(self.cell_centre)
        )
        self._cache: OrderedDict[bytes, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def _distance_position(self, position):
        """Measure cell distances in directions resolved by their geometry.

        A displacement normal to a plane of cell centres cannot change the
        weights. Reconstruction still uses the original 3D position and all
        supplied gradient components.
        """
        if self._distance_projection is None:
            return position
        return self._cell_centroid + (position - self._cell_centroid) @ self._distance_projection

    def _stencil_through_fluid(self, positions):
        """Find complete distance shells of fluid cells connected to each target without
        crossing the wall."""
        search_positions = self._distance_position(positions)
        rows = np.arange(len(positions))
        selected = [None] * len(rows)
        count = self.neighbour_count
        query_count = min(max(2 * count, 8), len(self.cell_centre))
        while len(rows):
            distance, cells = self.tree.query(search_positions[rows], k=query_count, workers=-1)
            distance = np.asarray(distance).reshape(len(rows), query_count)
            cells = np.asarray(cells).reshape(len(rows), query_count)
            targets = np.broadcast_to(positions[rows, None, :], (*cells.shape, 3))
            reachable = self._fluid_cells[cells].copy()
            reachable[reachable] &= ~self.solid_boundary.segments_cross_wall(
                self.cell_centre[cells[reachable]],
                targets[reachable],
            )
            pending = []
            for local, row in enumerate(rows):
                valid = np.flatnonzero(reachable[local])
                complete = query_count == len(self.cell_centre)
                if not len(valid):
                    if complete:
                        raise RuntimeError(
                            "the interpolation stencil has no fluid cell reachable without "
                            "crossing the wall"
                        )
                    pending.append(row)
                    continue
                stencil_radius = distance[local, valid[min(count, len(valid)) - 1]]
                tolerance = 64 * np.finfo(float).eps * max(1.0, stencil_radius)
                if not complete and (
                    len(valid) < count or distance[local, -1] <= stencil_radius + tolerance
                ):
                    pending.append(row)
                    continue
                valid = valid[distance[local, valid] <= stencil_radius + tolerance]
                selected[row] = (cells[local, valid], distance[local, valid])
            rows = np.asarray(pending, dtype=np.int64)
            query_count = min(2 * query_count, len(self.cell_centre))
        width = max((len(item[0]) for item in selected), default=count)
        indices = np.zeros((len(positions), width), dtype=np.int32)
        distances = np.full(indices.shape, np.inf)
        for row, (cells, distance) in enumerate(selected):
            indices[row, : len(cells)] = cells
            distances[row, : len(cells)] = distance
        return distances, indices

    def prepare(self, evaluation_position: np.ndarray) -> InterpolationStencil:
        """Fix a stencil independently of the six-entry query cache.

        Parameters
        ----------
        evaluation_position : ndarray, shape (N, 3)
            Fixed target coordinates in m; copied to prevent caller mutation.

        Returns
        -------
        InterpolationStencil
            Read-only target geometry and interpolation weights. Discard this
            stencil when the cell geometry or ordering changes. Field values are
            never cached.
        """
        position = np.array(evaluation_position, dtype=np.float64, copy=True).reshape(-1, 3)
        indices, weights = self._stencil(position)
        position.flags.writeable = False
        indices.flags.writeable = False
        weights.flags.writeable = False
        return InterpolationStencil(position, self.cell_centre, indices, weights)

    @staticmethod
    def _key(evaluation_position: np.ndarray) -> bytes:
        array = np.ascontiguousarray(evaluation_position, dtype=np.float64)
        digest = hashlib.blake2b(array.tobytes(), digest_size=16)
        return digest.digest()

    def _stencil(self, evaluation_position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return nearest-cell indices and normalized inverse-square weights.

        Parameters
        ----------
        evaluation_position : ndarray, shape (N, 3)
            Cartesian query points in metres.

        Returns
        -------
        tuple of ndarray
            ``(indices, weights)`` with shapes ``(N, K)`` and ``(N, K)``;
            ``K`` includes the widest tied cell shell in this target array.
            Shorter stencils have zero-weight padding. Each weight row sums
            to one. These internal cache entries must not be modified.
        """
        key = self._key(evaluation_position)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        if self.solid_boundary is not None:
            distance, indices = self._stencil_through_fluid(evaluation_position)
            weights = self._normalized_weights(distance)
            return self._cache_stencil(key, indices, weights)

        # Ask for one additional cell to detect a truncated distance shell.
        # Cartesian cells frequently tie: choosing an arbitrary subset then
        # changes reconstructed curvature under cell reordering/reflection.
        count = self.neighbour_count
        query_count = min(count + 1, len(self.cell_centre))
        search_position = self._distance_position(evaluation_position)
        distance, indices = self.tree.query(search_position, k=query_count, workers=-1)
        distance = np.asarray(distance, dtype=np.float64).reshape(
            len(evaluation_position), query_count
        )
        indices = np.asarray(indices, dtype=np.int32).reshape(len(evaluation_position), query_count)

        stencil_radius = distance[:, count - 1]
        tolerance = 64.0 * np.finfo(float).eps * np.maximum(1.0, stencil_radius)
        tied_rows = np.empty(0, dtype=np.int64)
        if query_count > count:
            tied_rows = np.flatnonzero(
                (distance[:, count] <= stencil_radius + tolerance) & (distance[:, 0] > 1.0e-12)
            )
        if len(tied_rows):
            neighbours = self.tree.query_ball_point(
                search_position[tied_rows],
                r=stencil_radius[tied_rows] + tolerance[tied_rows],
                workers=-1,
            )
            width = max(count, max(len(rows) for rows in neighbours))
            complete_distance = np.full((len(evaluation_position), width), np.inf)
            complete_indices = np.zeros((len(evaluation_position), width), dtype=np.int32)
            complete_distance[:, :count] = distance[:, :count]
            complete_indices[:, :count] = indices[:, :count]
            for row, cells in zip(tied_rows, neighbours, strict=True):
                cells = np.asarray(cells, dtype=np.int32)
                complete_indices[row, : len(cells)] = cells
                complete_distance[row, : len(cells)] = np.linalg.norm(
                    search_position[row] - self.cell_centre[cells], axis=1
                )
            distance, indices = complete_distance, complete_indices
        else:
            distance, indices = distance[:, :count], indices[:, :count]

        weights = self._normalized_weights(distance)
        return self._cache_stencil(key, indices, weights)

    @staticmethod
    def _normalized_weights(distance):
        weights = 1.0 / np.maximum(distance, 1.0e-12) ** 2
        exact = distance[:, 0] <= 1.0e-12
        if exact.any():
            weights[exact] = 0.0
            weights[exact, 0] = 1.0
        weights /= weights.sum(axis=1, keepdims=True)
        return weights

    def _cache_stencil(self, key, indices, weights):
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
    ) -> np.ndarray:
        """Reconstruct velocity at arbitrary points from values and gradients.

        Parameters
        ----------
        evaluation_position : ndarray, shape (N, 3)
            Cartesian target positions in m.
        velocity : ndarray, shape (M, 3)
            Cell-centred FVM velocity in m/s.
        gradient : ndarray, shape (M, 3, 3)
            FVM velocity gradient in 1/s with
            ``gradient[i, j] = d(velocity_j)/d(x_i)``.
        Returns
        -------
        ndarray, shape (N, 3)
            New ``float64`` target-velocity array in m/s.

        Notes
        -----
        For cell ``k`` the Taylor value is
        ``u_k + (x_target - x_k) dot grad(u)_k``. Neighbour values are then
        inverse-square-distance blended. Inputs are not modified.

        Raises
        ------
        ValueError
            If the velocity or gradient cell count is inconsistent with the
            cell-centre count.
        """
        evaluation_position = np.ascontiguousarray(evaluation_position, dtype=np.float64).reshape(
            -1, 3
        )
        velocity = np.asarray(velocity, dtype=np.float64).reshape(-1, 3)
        gradient = np.asarray(gradient, dtype=np.float64).reshape(-1, 3, 3)
        if len(velocity) != len(self.cell_centre) or len(gradient) != len(self.cell_centre):
            raise ValueError("Velocity and gradient must have one row per FVM cell")
        indices, weights = self._stencil(evaluation_position)
        sampled = np.empty((len(evaluation_position), 3), dtype=np.float64)

        for start in range(0, len(evaluation_position), _INTERPOLATION_BATCH_SIZE):
            stop = min(start + _INTERPOLATION_BATCH_SIZE, len(evaluation_position))
            _sample_velocity(
                evaluation_position[start:stop],
                self.cell_centre,
                velocity,
                gradient,
                indices[start:stop],
                weights[start:stop],
                sampled[start:stop],
            )
        return sampled


__all__ = ["FVMVelocityInterpolator", "InterpolationStencil"]
