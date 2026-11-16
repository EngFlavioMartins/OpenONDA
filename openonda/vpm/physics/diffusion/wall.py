"""Solid-body exclusion on the diffusion lattice.

:class:`BodyWall` holds the body classifier the coupler installs: the strict-interior
test, the segment-visibility test behind the wall, and the bounds the queries are limited
to. From them it fills the lattice's body mask and wall-link fields (reusing exactly
validated host geometry between steps), folds slip-slab ghost coordinates back into the
slab, and restores the M4' deposits of wall-adjacent particles on fluid-connected nodes
(:meth:`BodyWall.compute_m4_wall_corrections`).
"""

from __future__ import annotations

from collections.abc import Callable
import inspect
from typing import TYPE_CHECKING

import numpy as np
import taichi as ti

from openonda.core.grid_connectivity import connected_grid_components
from openonda.core.m4_kernel import m4_prime

from .body_geometry import BodyGeometryKey, BodyGridGeometryCache, ImmutableBodyGeometryQueries

if TYPE_CHECKING:
    from .lattice import DiffusionLattice


class BodyWall:
    """The body of a diffusion lattice: classifier callbacks, cached masks and wall links."""

    def __init__(self, lattice: DiffusionLattice) -> None:
        self.lattice = lattice
        # Body-aware diffusion is enabled by the first configure_body_classifier call.
        self.active: bool = False
        self.classifier: Callable[[np.ndarray], np.ndarray] | None = None
        self.segment_classifier: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None
        self.geometry_revision: object | None = None
        self.mask_cache_key: tuple | None = None
        self.mask_host: np.ndarray | None = None
        self.links_host: np.ndarray | None = None
        self.query_bounds: np.ndarray | None = None
        self.geometry_queries: ImmutableBodyGeometryQueries | None = None
        self.grid_geometry_cache: BodyGridGeometryCache | None = None

    def configure_body_classifier(
        self,
        contains_interior: Callable[[np.ndarray], np.ndarray],
        *,
        revision: object,
        query_bounds: np.ndarray | None = None,
        segments_cross_wall: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        geometry_queries: ImmutableBodyGeometryQueries | None = None,
    ) -> None:
        """Use the transfer body's strict-interior classifier on the GBD lattice.

        ``revision`` identifies an immutable body configuration. Changing it
        invalidates the resident device mask. Queries are bounded in size and
        only repeated if the lattice origin, spacing, shape, or body changes.
        """
        if not callable(contains_interior):
            raise TypeError("body classifier must be callable")
        if revision is None:
            raise ValueError("body geometry revision must be supplied")
        self.classifier = contains_interior
        self.segment_classifier = segments_cross_wall
        self.geometry_queries = geometry_queries
        self.grid_geometry_cache = None
        self.geometry_revision = revision
        if query_bounds is not None:
            bounds = np.asarray(query_bounds, dtype=np.float64).reshape(6)
            if not np.all(np.isfinite(bounds)) or np.any(bounds[1::2] <= bounds[::2]):
                raise ValueError("body query bounds must be finite and increasing")
            self.query_bounds = bounds.copy()
        else:
            self.query_bounds = None
        self.active = True
        self.mask_cache_key = None
        self.mask_host = None

    def _geometry_key(self) -> BodyGeometryKey | None:
        queries = self.geometry_queries
        if type(queries) is not ImmutableBodyGeometryQueries:
            return None
        if any(
            name in vars(self) or inspect.getattr_static(type(self), name, None) is not method
            for name, method in _BODY_GRID_GEOMETRY_METHODS.items()
        ):
            return None
        key = queries.key(self.classifier, self.segment_classifier, self.geometry_revision)
        if key is None:
            return None
        slab = self.lattice.slip_slab_bounds
        slab_key = None if slab is None else np.asarray(slab, dtype=np.float64).tobytes()
        return key, ("none",) if slab_key is None else ("slab", slab_key)

    def prepare_mask(
        self, grid_min: np.ndarray, particle_spacing: float, nx: int, ny: int, nz: int
    ) -> None:
        """Populate masks/links, reusing only exactly validated host geometry."""
        if not self.active or self.lattice.body_mask_grid is None:
            return
        try:
            geometry_key = self._geometry_key()
        except BaseException:
            self.mask_cache_key = None
            self.mask_host = self.links_host = None
            raise
        slab = self.lattice.slip_slab_bounds
        g = np.asarray(grid_min, dtype=np.float32).reshape(3)
        key = (
            id(self.lattice.body_mask_grid),
            self.geometry_revision,
            tuple(float(value) for value in g),
            float(np.float32(particle_spacing)),
            int(nx),
            int(ny),
            int(nz),
            tuple(float(value) for value in slab) if slab is not None else None,
            float(particle_spacing),
            geometry_key,
            id(self.lattice.body_link_grid),
            g.tobytes(),
        )
        invalid_queries = self.geometry_queries is not None and geometry_key is None
        if not invalid_queries and key == self.mask_cache_key:
            return
        # Never leave stale residency checked after a partially written mask
        # or link grid. Host evidence is independent of device allocation.
        self.mask_cache_key = None
        self.mask_host = self.links_host = None
        try:
            if geometry_key is not None:
                if self.grid_geometry_cache is None:
                    self.grid_geometry_cache = BodyGridGeometryCache()
                cache = self.grid_geometry_cache
                required_bytes = 5 * int(nx) * int(ny) * int(nz) + 12 * (nx + ny + nz)
                if required_bytes <= cache.max_bytes:
                    mask, links = cache.prepare(
                        g,
                        particle_spacing,
                        (nx, ny, nz),
                        contains=self.interior_at_particles,
                        segments_cross_wall=self.segments_cross_wall
                        if self.segment_classifier is not None
                        else None,
                        geometry_key=geometry_key,
                    )
                    if self.segment_classifier is not None and self.lattice.body_link_grid is None:
                        self.lattice.body_link_grid = ti.field(
                            dtype=ti.i32, shape=self.lattice.body_mask_grid.shape
                        )
                    for field, values in (
                        (self.lattice.body_mask_grid, mask),
                        (self.lattice.body_link_grid, links),
                    ):
                        if field is None:
                            continue
                        buffer = self.lattice.grid_transfer_buffer("scalar", field, "upload")
                        flat = values.reshape(-1)
                        for start in range(0, len(flat), self.lattice.TRANSFER_CHUNK):
                            count = min(self.lattice.TRANSFER_CHUNK, len(flat) - start)
                            buffer[:count] = flat[start : start + count]
                            self.lattice.upload_scalar_chunk_kernel(
                                field, buffer, start, count, ny, nz
                            )
                    ti.sync()
                    self.mask_host, self.links_host = mask, links
                    self.mask_cache_key = (*key[:-2], id(self.lattice.body_link_grid), key[-1])
                    return
            self._prepare_mask_fresh(g, particle_spacing, nx, ny, nz)
            ti.sync()
            self.mask_cache_key = (
                None if invalid_queries else (*key[:-2], id(self.lattice.body_link_grid), key[-1])
            )
        except BaseException:
            self.mask_cache_key = None
            self.mask_host = self.links_host = None
            raise

    def _prepare_mask_fresh(
        self, g: np.ndarray, particle_spacing: float, nx: int, ny: int, nz: int
    ) -> None:
        """Original predicate path for unqualified/custom geometry queries."""
        if self.lattice.body_link_grid is not None:
            self.lattice.body_link_grid.fill(0)
        total = int(nx) * int(ny) * int(nz)
        host_mask = np.empty(total, dtype=bool)
        buffer = self.lattice.grid_transfer_buffer("scalar", self.lattice.body_mask_grid, "upload")
        for start in range(0, total, self.lattice.TRANSFER_CHUNK):
            stop = min(start + self.lattice.TRANSFER_CHUNK, total)
            linear = np.arange(start, stop, dtype=np.int64)
            points = np.column_stack(
                (
                    g[0] + (linear // (ny * nz)) * particle_spacing,
                    g[1] + ((linear // nz) % ny) * particle_spacing,
                    g[2] + (linear % nz) * particle_spacing,
                )
            )
            try:
                values = self.interior_at_particles(points.astype(np.float32).astype(np.float64))
            except RuntimeError as error:
                if "one flag per particle" not in str(error):
                    raise
                raise RuntimeError("body classifier must return one flag per GBD node") from error
            if len(values) != stop - start:
                raise RuntimeError("body classifier must return one flag per GBD node")
            host_mask[start:stop] = values
            buffer[: stop - start] = values
            self.lattice.upload_scalar_chunk_kernel(
                self.lattice.body_mask_grid, buffer, start, stop - start, ny, nz
            )
        ti.sync()
        self.mask_host = host_mask.reshape(nx, ny, nz)
        self._prepare_links(g, particle_spacing, nx, ny, nz)

    def segments_cross_wall(self, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """Test visibility on the physical wall, including reflected slab paths."""
        starts = np.asarray(starts, dtype=np.float64).reshape(-1, 3)
        ends = np.asarray(ends, dtype=np.float64).reshape(-1, 3)
        classifier = self.segment_classifier
        if classifier is None:
            return np.zeros(len(starts), dtype=bool)

        def query(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            result = np.asarray(classifier(a, b), dtype=bool).reshape(-1)
            if len(result) != len(a):
                raise RuntimeError("body segment classifier must return one flag per segment")
            return result

        slab = self.lattice.slip_slab_bounds
        if slab is None or len(starts) == 0:
            return query(starts, ends)
        lower, upper = slab
        length = upper - lower
        # Splitting at mirror planes preserves the kink of the folded path.
        zmin = min(starts[:, 2].min(), ends[:, 2].min())
        zmax = max(starts[:, 2].max(), ends[:, 2].max())
        planes = lower + length * np.arange(
            int(np.floor((zmin - lower) / length)),
            int(np.ceil((zmax - lower) / length)) + 1,
        )
        delta = ends - starts
        fractions = np.ones((len(starts), len(planes) + 2))
        fractions[:, 0] = 0.0
        moving = np.abs(delta[:, 2]) > 0.0
        fractions[moving, 1:-1] = np.clip(
            (planes[None, :] - starts[moving, 2, None]) / delta[moving, 2, None],
            0,
            1,
        )
        fractions.sort(axis=1)
        result = np.zeros(len(starts), dtype=bool)
        for index in range(fractions.shape[1] - 1):
            rows = (fractions[:, index + 1] > fractions[:, index]) & ~result
            if not np.any(rows):
                continue
            a = starts[rows] + fractions[rows, index, None] * delta[rows]
            b = starts[rows] + fractions[rows, index + 1, None] * delta[rows]
            result[rows] |= query(self.fold_slab_exterior_z(a), self.fold_slab_exterior_z(b))
        return result

    def _prepare_links(self, origin: np.ndarray, spacing: float, nx: int, ny: int, nz: int) -> None:
        """Cache surface-crossing links, including walls with no solid grid node."""
        if self.segment_classifier is None:
            return
        if self.lattice.body_link_grid is None:
            self.lattice.body_link_grid = ti.field(
                dtype=ti.i32, shape=self.lattice.body_mask_grid.shape
            )
        mask = self.mask_host
        if mask is None:
            raise RuntimeError("the body mask must be prepared before its wall links")
        buffer = self.lattice.grid_transfer_buffer("scalar", self.lattice.body_link_grid, "upload")
        total = nx * ny * nz
        host_links = np.zeros(total, dtype=np.int32)
        dimensions = np.array([nx, ny, nz])
        for start in range(0, total, self.lattice.TRANSFER_CHUNK):
            stop = min(start + self.lattice.TRANSFER_CHUNK, total)
            linear = np.arange(start, stop, dtype=np.int64)
            indices = np.column_stack((linear // (ny * nz), (linear // nz) % ny, linear % nz))
            points = np.asarray(origin) + spacing * indices
            values = np.zeros(len(indices), dtype=np.int32)
            for axis in range(3):
                next_indices = indices.copy()
                next_indices[:, axis] = np.minimum(indices[:, axis] + 1, dimensions[axis] - 1)
                rows = (
                    ~mask[tuple(indices.T)]
                    & ~mask[tuple(next_indices.T)]
                    & (indices[:, axis] + 1 < dimensions[axis])
                )
                ends = points[rows].copy()
                ends[:, axis] += spacing
                values[rows] |= (
                    self.segments_cross_wall(points[rows], ends).astype(np.int32) << axis
                )
            buffer[: len(values)] = values
            host_links[start:stop] = values
            self.lattice.upload_scalar_chunk_kernel(
                self.lattice.body_link_grid, buffer, start, len(values), ny, nz
            )
        self.links_host = host_links.reshape(nx, ny, nz)

    def fold_slab_exterior_z(self, positions: np.ndarray) -> np.ndarray:
        """Map ghost z to its physical mirror, leaving in-slab coordinates exact."""
        points = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        slab = self.lattice.slip_slab_bounds
        if slab is None or len(points) == 0:
            return points
        lower, upper = map(float, slab)
        length = upper - lower
        if length <= 0.0:
            raise ValueError("slip-slab planes must be increasing")
        exterior = (points[:, 2] < lower) | (points[:, 2] > upper)
        if not np.any(exterior):
            return points
        folded = points.copy()
        phase = np.mod(points[exterior, 2] - lower, 2.0 * length)
        folded[exterior, 2] = lower + length - np.abs(phase - length)
        return folded

    def interior_at_particles(self, positions: np.ndarray) -> np.ndarray:
        """Classify strict solid interior before a remesh can erase input."""
        points = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        if not self.active:
            return np.zeros(len(points), dtype=bool)
        points = self.fold_slab_exterior_z(points)
        if self.classifier is not None:
            result = np.asarray(self.classifier(points), dtype=bool).reshape(-1)
            if len(result) != len(points):
                raise RuntimeError("body classifier must return one flag per particle")
            return result
        return np.zeros(len(points), dtype=bool)

    def _wall_moment_correction(
        self,
        point: np.ndarray,
        indices: np.ndarray,
        weights: np.ndarray,
        fluid: np.ndarray,
        origin: np.ndarray,
        spacing: float,
        shape: tuple[int, int, int],
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Find bounded conservative weights on fluid-connected support.

        A cut stencil can need more than its original four nodes per axis
        to represent a particle between the wall and the nearest grid node.
        Expand locally before declaring the geometry under-resolved.
        """
        fractional = (point - origin) / spacing
        target = np.array([1.0, 0.0, 0.0, 0.0])
        for radius in range(1, 5):
            if radius > 1:
                offsets = np.stack(
                    np.meshgrid(*([np.arange(-radius, radius + 2)] * 3), indexing="ij"), axis=-1
                )
                indices = np.floor(fractional).astype(np.int64) + offsets.reshape(-1, 3)
                indices = indices[np.all((indices >= 0) & (indices < np.asarray(shape)), axis=1)]
                nodes = origin + spacing * indices
                fluid = ~self.interior_at_particles(nodes)
                fluid[fluid] &= ~self.segments_cross_wall(
                    np.broadcast_to(point, nodes[fluid].shape),
                    nodes[fluid],
                )
                weights = np.prod(m4_prime(fractional - indices), axis=1)
            relative = (origin + spacing * indices[fluid] - point) / spacing
            constraints = np.vstack((np.ones(fluid.sum()), relative.T))
            gram = constraints @ constraints.T
            if np.linalg.cond(gram) > 1e10:
                continue
            defect = target - constraints @ weights[fluid]
            correction = constraints.T @ np.linalg.solve(gram, defect)
            corrected = weights[fluid] + correction
            if not np.all(np.isfinite(corrected)) or np.sum(np.abs(corrected)) > 2.0:
                continue
            if np.max(np.abs(constraints @ corrected - target)) <= 1e-10:
                return indices[fluid], correction, radius
        raise RuntimeError(
            "GBD wall transfer has insufficient fluid-connected support for bounded moment "
            "conservation. Refine the particle spacing near the wall or narrow gap."
        )

    def _wall_strength_fallback(
        self,
        point: np.ndarray,
        indices: np.ndarray,
        weights: np.ndarray,
        fluid: np.ndarray,
        origin: np.ndarray,
        spacing: float,
        shape: tuple[int, int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Deposit a particle's whole strength on fluid-connected nodes, without its first moments.

        Used where :meth:`_wall_moment_correction` finds no bounded support, which happens at
        a wall thinner than the lattice spacing (a sharp trailing edge). The fluid deposits are
        rescaled to unit sum when they carry at least a quarter of the strength; otherwise the
        strength moves to the nearest fluid-connected node. The returned values are corrections
        to the M4' deposits already on ``indices[fluid]``.
        """
        own = weights[fluid]
        total = float(np.sum(own))
        if fluid.any() and total >= 0.25:
            return indices[fluid], own * (1.0 / total - 1.0)
        fractional = (point - origin) / spacing
        base = np.floor(fractional).astype(np.int64)
        for radius in range(1, 5):
            offsets = np.stack(
                np.meshgrid(*([np.arange(-radius, radius + 2)] * 3), indexing="ij"), axis=-1
            ).reshape(-1, 3)
            candidates = base + offsets
            candidates = candidates[
                np.all((candidates >= 0) & (candidates < np.asarray(shape)), axis=1)
            ]
            nodes = origin + spacing * candidates
            open_nodes = ~self.interior_at_particles(nodes)
            open_nodes[open_nodes] &= ~self.segments_cross_wall(
                np.broadcast_to(point, nodes[open_nodes].shape),
                nodes[open_nodes],
            )
            if open_nodes.any():
                reachable = candidates[open_nodes]
                nearest = reachable[
                    np.argmin(np.sum((origin + spacing * reachable - point) ** 2, axis=1))
                ]
                return (
                    np.vstack((indices[fluid], nearest[None, :])),
                    np.concatenate((-own, [1.0])),
                )
        raise RuntimeError(
            "GBD wall transfer found no fluid-connected lattice node within four spacings of a "
            "particle at the wall."
        )

    def compute_m4_wall_corrections(
        self,
        positions: np.ndarray,
        strengths: np.ndarray,
        origin: np.ndarray,
        spacing: float,
        shape: tuple[int, int, int],
        wall_fallback_limit: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
        """Restore each wall-adjacent M4 deposit on local fluid support.

        Signed M4 weights are retained. A minimum-norm correction on fluid
        nodes enforces each source's zeroth and three first spatial moments.
        Failed/under-resolved support is rejected, never globally compensated,
        unless ``wall_fallback_limit`` is positive: a particle without bounded
        support then moves its whole strength to fluid-connected nodes, and the
        step fails only when those particles carry more than that fraction of
        the total strength.
        """
        points = np.asarray(positions, dtype=np.float64)
        gamma = np.asarray(strengths, dtype=np.float64)
        bounds = self.query_bounds
        if bounds is None:
            candidates = np.arange(len(points))
        else:
            pad = 2.0 * float(spacing)
            candidates = np.flatnonzero(
                np.all((points >= bounds[::2] - pad) & (points <= bounds[1::2] + pad), axis=1)
            )
        offsets = np.stack(
            np.meshgrid(np.arange(-1, 3), np.arange(-1, 3), np.arange(-1, 3), indexing="ij"),
            axis=-1,
        ).reshape(-1, 3)
        node_chunks = []
        value_chunks = []
        touched = 0
        excluded_l1 = 0.0
        correction_l1 = 0.0
        expanded_support = 0
        fallback_particles = 0
        fallback_l1 = 0.0
        fallback_limit = float(wall_fallback_limit)
        grid_min = np.asarray(origin, dtype=np.float64)
        dimensions = np.asarray(shape)
        cached_mask = self.mask_host
        cached_key = self.mask_cache_key
        if (
            cached_mask is None
            or cached_key is None
            or cached_mask.shape != shape
            or cached_key[2]
            != tuple(float(value) for value in np.asarray(origin, dtype=np.float32))
            or cached_key[3] != float(np.float32(spacing))
        ):
            cached_mask = None
        for start in range(0, len(candidates), 512):
            particle_ids = candidates[start : start + 512]
            fractions = (points[particle_ids] - grid_min) / float(spacing)
            index_batch = np.floor(fractions).astype(np.int64)[:, None, :] + offsets
            valid_batch = np.all((index_batch >= 0) & (index_batch < dimensions), axis=2)
            safe_indices = np.clip(index_batch, 0, dimensions - 1)
            if cached_mask is not None:
                solid_batch = (
                    cached_mask[safe_indices[:, :, 0], safe_indices[:, :, 1], safe_indices[:, :, 2]]
                    & valid_batch
                )
            else:
                node_batch = grid_min + index_batch * float(spacing)
                solid_batch = (
                    self.interior_at_particles(node_batch.reshape(-1, 3)).reshape(
                        len(particle_ids), 64
                    )
                    & valid_batch
                )
            unavailable_batch = solid_batch.copy()
            if self.segment_classifier is not None:
                node_batch = grid_min + index_batch * float(spacing)
                sources = np.broadcast_to(points[particle_ids, None, :], node_batch.shape)
                fluid_side_candidates = valid_batch & ~solid_batch
                unavailable_batch[fluid_side_candidates] |= self.segments_cross_wall(
                    sources[fluid_side_candidates],
                    node_batch[fluid_side_candidates],
                )
            for row in np.flatnonzero(np.any(unavailable_batch, axis=1)):
                particle = particle_ids[row]
                indices = index_batch[row]
                if not np.all(valid_batch[row]):
                    raise RuntimeError("GBD M4 support leaves the diffusion lattice near a solid")
                solid = solid_batch[row]
                touched += 1
                weights = np.prod(m4_prime(fractions[row] - indices), axis=1)
                fluid = ~unavailable_batch[row]
                try:
                    correction_indices, correction, radius = self._wall_moment_correction(
                        points[particle],
                        indices,
                        weights,
                        fluid,
                        grid_min,
                        float(spacing),
                        shape,
                    )
                except RuntimeError:
                    if fallback_limit <= 0.0:
                        raise
                    correction_indices, correction = self._wall_strength_fallback(
                        points[particle],
                        indices,
                        weights,
                        fluid,
                        grid_min,
                        float(spacing),
                        shape,
                    )
                    radius = 0
                    fallback_particles += 1
                    fallback_l1 += float(np.linalg.norm(gamma[particle]))
                expanded_support += radius > 1
                node_chunks.append(correction_indices.astype(np.int32))
                value_chunks.append((correction[:, None] * gamma[particle]).astype(np.float32))
                # The initial device scatter included fluid nodes hidden
                # behind a wall. Remove those deposits before redistributing
                # on this particle's fluid side of the wall.
                occluded = unavailable_batch[row] & ~solid
                if np.any(occluded):
                    node_chunks.append(indices[occluded].astype(np.int32))
                    value_chunks.append(
                        (-weights[occluded, None] * gamma[particle]).astype(np.float32)
                    )
                magnitude = float(np.linalg.norm(gamma[particle]))
                excluded_l1 += float(np.sum(np.abs(weights[unavailable_batch[row]]))) * magnitude
                correction_l1 += float(np.sum(np.abs(correction))) * magnitude
        fallback_fraction = 0.0
        if fallback_particles:
            fallback_fraction = fallback_l1 / max(
                float(np.sum(np.linalg.norm(gamma, axis=1))), 1e-300
            )
            if fallback_fraction > fallback_limit:
                raise RuntimeError(
                    f"GBD wall transfer without bounded moment support for {fallback_particles} "
                    f"particles carrying {fallback_fraction:.3e} of the total vortex strength, "
                    f"above the limit {fallback_limit:.3e}. Refine the particle spacing near the "
                    "wall or narrow gap."
                )
        diagnostics = {
            "wall_adjacent_particles": touched,
            "excluded_signed_weight_l1": excluded_l1,
            "fluid_correction_l1": correction_l1,
            "expanded_support_particles": expanded_support,
            "fallback_particles": fallback_particles,
            "fallback_strength_fraction": fallback_fraction,
        }
        if not node_chunks:
            return np.empty((0, 3), dtype=np.int32), np.empty((0, 3), dtype=np.float32), diagnostics
        return np.concatenate(node_chunks), np.concatenate(value_chunks), diagnostics

    def correction_labels(self, magnitude: np.ndarray, groups: np.ndarray) -> np.ndarray:
        """Keep the moment correction within grid support connected through the fluid.

        Zero-strength nodes are excluded: a correction cannot jump across a
        wall by following an empty route around its far edge. Existing source
        group boundaries are retained as well.
        """
        return connected_grid_components(magnitude, groups, self.links_host)


_BODY_GRID_GEOMETRY_METHODS = {
    name: inspect.getattr_static(BodyWall, name)
    for name in (
        "_geometry_key",
        "prepare_mask",
        "_prepare_mask_fresh",
        "_prepare_links",
        "interior_at_particles",
        "segments_cross_wall",
        "fold_slab_exterior_z",
    )
}
