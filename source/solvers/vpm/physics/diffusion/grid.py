"""Shared grid-based diffusion machinery for the VPM solver.

The :class:`_GridDiffusionMixin` owns the stateful grid structure and the
kernels that both the DVH and GBD viscous-diffusion schemes share: grid
allocation and bounds, body masking, M4' scattering, Laplacian stepping,
particle regeneration, and the transfer buffers between the Taichi grid and
host arrays.  The two schemes both drive this mixin through their own
``grid_based_diffusion`` / ``gbd_diffusion`` entry points, which are kept here
with the machinery because they share its grid state.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from dataclasses import dataclass
import logging
import math

from numba import njit
import numpy as np
import taichi as ti

from ...config.constants import _DVH_BETA, MAX_N_PARTICLES
from ...io.logging import Logging

_logger = logging.getLogger("vpm")

_GRID_TRANSFER_CHUNK = 65536
# M4' performs 64 atomic grid deposits per particle.  One all-particle Vulkan
# dispatch can exceed integrated-GPU watchdog limits at production counts, so
# accumulate the same grid in bounded particle batches.
# Each particle performs 64 atomic deposits, making a nominal 32k batch more
# than two million contended grid updates.  Keep dispatches below the i915
# Vulkan fence-watchdog limit observed in the production coupled-cube case.
_M4_SCATTER_BATCH_SIZE = 4096

# The nine moment constraints need 27 coefficients per survivor when written
# as one dense tensor.  Production clouds can contain O(10^6) survivors, so
# form that tensor only in bounded host chunks while accumulating the fixed
# 9x9 normal matrix and applying its correction.
_GBD_MOMENT_CHUNK_SIZE = 65536
_GBD_MOMENT_RCOND = 1.0e-12
_GBD_MOMENT_CONDITION_LIMIT = 1.0e12
_GBD_MOMENT_RESIDUAL_LIMIT = 1.0e-5
# A deficient support normally needs at most three replacements.  Limit the
# cap-full search to a small weakest-node frontier so each candidate is judged
# with fixed 9x9 algebra instead of an O(N_retained) trial cloud.
_GBD_MOMENT_REPLACEMENT_POOL_SIZE = 32
# A recovery whose L1 change is comparable with the complete diffused field is
# no longer a small pruning closure.  Stop before mutating the production grid
# instead of accepting a numerically closed but physically amplified cloud.
_GBD_MOMENT_CORRECTION_FRACTION_LIMIT = 0.5

# Radius assigned to freshly regenerated particles: σ = _REGEN_RADIUS_RATIO * particle_spacing
_REGEN_RADIUS_RATIO = 2.5


@dataclass(frozen=True)
class _NearestNodeMapping:
    """One nearest-grid-node classification shared by regenerated fields."""

    valid: np.ndarray
    linear_index: np.ndarray
    vortex_strength_weight: np.ndarray


def _nearest_node_mapping(
    pos_np: np.ndarray,
    vortex_strength: np.ndarray,
    grid_min_np: np.ndarray,
    particle_spacing: float,
    nx: int,
    ny: int,
    nz: int,
) -> _NearestNodeMapping:
    """Map particles to valid nearest nodes once for every scalar/ID scatter."""
    indices = np.rint(
        (np.asarray(pos_np) - np.asarray(grid_min_np)) / float(particle_spacing)
    ).astype(np.intp)
    valid = (
        (indices[:, 0] >= 0)
        & (indices[:, 0] < nx)
        & (indices[:, 1] >= 0)
        & (indices[:, 1] < ny)
        & (indices[:, 2] >= 0)
        & (indices[:, 2] < nz)
    )
    selected = indices[valid]
    linear_index = selected[:, 0] * (ny * nz) + selected[:, 1] * nz + selected[:, 2]
    vortex_strength_weight = np.abs(np.asarray(vortex_strength)[valid]).sum(axis=1)
    return _NearestNodeMapping(valid, linear_index, vortex_strength_weight)


def _gbd_moment_constraint_rows(scaled_position: np.ndarray) -> np.ndarray:
    """Return one bounded chunk of the nine GBD moment constraints."""
    radius_squared = np.einsum(
        "ij,ij->i",
        scaled_position,
        scaled_position,
    )
    rows = np.zeros((len(scaled_position), 9, 3), dtype=np.float64)
    rows[:, :3, :] = np.eye(3)
    rows[:, 3, 1] = -scaled_position[:, 2]
    rows[:, 3, 2] = scaled_position[:, 1]
    rows[:, 4, 0] = scaled_position[:, 2]
    rows[:, 4, 2] = -scaled_position[:, 0]
    rows[:, 5, 0] = -scaled_position[:, 1]
    rows[:, 5, 1] = scaled_position[:, 0]
    rows[:, 6:, :] = (
        np.einsum("ij,ik->ijk", scaled_position, scaled_position)
        - radius_squared[:, None, None] * np.eye(3)
    ) / 3.0
    return rows


def _gbd_moment_gram(
    position: np.ndarray,
    vortex_strength: np.ndarray,
    particle_spacing: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Build the bounded-memory weighted moment Gram matrix."""
    weight_magnitude = np.linalg.norm(vortex_strength, axis=1)
    weight_sum = float(weight_magnitude.sum(dtype=np.float64))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        raise RuntimeError("GBD moment support has no finite retained strength")
    weights = weight_magnitude / weight_sum
    length_scale = max(
        float(
            np.sqrt(
                np.sum(
                    weights
                    * np.einsum(
                        "ij,ij->i",
                        position,
                        position,
                    )
                )
            )
        ),
        float(particle_spacing),
    )
    gram = np.zeros((9, 9), dtype=np.float64)
    for start in range(0, len(position), _GBD_MOMENT_CHUNK_SIZE):
        stop = min(start + _GBD_MOMENT_CHUNK_SIZE, len(position))
        rows = _gbd_moment_constraint_rows(position[start:stop] / length_scale)
        gram += np.einsum(
            "i,ijk,ilk->jl",
            weights[start:stop],
            rows,
            rows,
        )
    return gram, length_scale, weights


def _gbd_moment_gram_quality(gram: np.ndarray) -> tuple[int, float]:
    """Return numerical rank and condition under the production closure limits."""
    if not np.all(np.isfinite(gram)):
        return 0, float("inf")
    singular_values = np.linalg.svd(gram, compute_uv=False)
    if len(singular_values) == 0 or singular_values[0] <= 0.0:
        return 0, float("inf")
    cutoff = _GBD_MOMENT_RCOND * singular_values[0]
    rank = int(np.count_nonzero(singular_values > cutoff))
    condition = (
        float("inf")
        if singular_values[-1] <= 0.0
        else float(singular_values[0] / singular_values[-1])
    )
    return rank, condition


def _gbd_moment_support_quality(
    position: np.ndarray,
    vortex_strength: np.ndarray,
    particle_spacing: float,
) -> tuple[int, float]:
    """Return closure rank/conditioning for one retained support cloud."""
    if len(position) < 4:
        return 0, float("inf")
    gram, _length_scale, _weights = _gbd_moment_gram(
        position,
        vortex_strength,
        particle_spacing,
    )
    return _gbd_moment_gram_quality(gram)


def _m4_prime_1d(r: np.ndarray) -> np.ndarray:
    """Vectorized 1D M4' (Monaghan 1985) interpolation kernel.

    Support: [-2, 2].  Satisfies partition of unity and reproduces
    quadratic polynomials exactly → O(particle_spacing⁴) remeshing error (3rd-order
    convergence), compared to CIC's O(particle_spacing²).

    The kernel has negative lobes in (1, 2), which is essential for
    anti-aliasing and high-order accuracy.
    """
    q = np.abs(r)
    w = np.zeros_like(q)
    m1 = q <= 1.0
    m2 = (q > 1.0) & (q <= 2.0)
    w[m1] = 1.0 - 2.5 * q[m1] ** 2 + 1.5 * q[m1] ** 3
    w[m2] = 0.5 * (2.0 - q[m2]) ** 2 * (1.0 - q[m2])
    return w


@njit(cache=True, fastmath=False)
def _dvh_scatter_numba(
    pos: np.ndarray,
    vortex_strength: np.ndarray,
    widths: np.ndarray,
    grid_out: np.ndarray,
    gmin: np.ndarray,
    particle_spacing: float,
    R_d: float,
    R_d_sq: float,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """DVH heat-kernel scatter (Durante et al. 2024, Eqs. 17-19), JIT-compiled.

    For each particle ``p`` at ``pos[p]`` with vortex strength ``vortex_strength[p]`` and
    Gaussian width ``widths[p]`` (= β·R_d²·q_p), spread its vortex strength to the
    grid nodes within the diffusive radius R_d using the exact heat-kernel
    weight ``exp(-r²/width_p)``.  Shepard normalisation (the per-particle
    ``/w_sum``) conserves each particle's total vortex strength exactly; the node
    contributions accumulate in ``grid_out`` (shape ``(nx, ny, nz, 3)``, f64).

    This is the exact f64 algorithm of the former NumPy ``for j in range(N)``
    loop — same formulas, same accumulation order — compiled with Numba so the
    49k-particle scatter that cost ≈5 min in pure Python runs in a fraction of
    a second.  ``grid_out`` must be zeroed by the caller.
    """
    N = pos.shape[0]
    for p in range(N):
        px = pos[p, 0]
        py = pos[p, 1]
        pz = pos[p, 2]
        width = widths[p]

        # Index bounds of the bounding box within R_d of particle p.
        i_lo = max(0, int(np.floor((px - R_d - gmin[0]) / particle_spacing)))
        i_hi = min(nx - 1, int(np.ceil((px + R_d - gmin[0]) / particle_spacing)))
        j_lo = max(0, int(np.floor((py - R_d - gmin[1]) / particle_spacing)))
        j_hi = min(ny - 1, int(np.ceil((py + R_d - gmin[1]) / particle_spacing)))
        k_lo = max(0, int(np.floor((pz - R_d - gmin[2]) / particle_spacing)))
        k_hi = min(nz - 1, int(np.ceil((pz + R_d - gmin[2]) / particle_spacing)))

        if i_lo > i_hi or j_lo > j_hi or k_lo > k_hi:
            continue

        # Pass 1 — Shepard denominator over support nodes within R_d.
        w_sum = 0.0
        for ii in range(i_lo, i_hi + 1):
            dx = (gmin[0] + ii * particle_spacing) - px
            dx2 = dx * dx
            for jj in range(j_lo, j_hi + 1):
                dy = (gmin[1] + jj * particle_spacing) - py
                dxy2 = dx2 + dy * dy
                for kk in range(k_lo, k_hi + 1):
                    dz = (gmin[2] + kk * particle_spacing) - pz
                    r2 = dxy2 + dz * dz
                    if r2 <= R_d_sq:
                        w_sum += np.exp(-r2 / width)

        if w_sum < 1e-300:
            continue

        # Pass 2 — deposit Shepard-normalised vortex strength (exact Γ per particle).
        cx = vortex_strength[p, 0] / w_sum
        cy = vortex_strength[p, 1] / w_sum
        cz = vortex_strength[p, 2] / w_sum
        for ii in range(i_lo, i_hi + 1):
            dx = (gmin[0] + ii * particle_spacing) - px
            dx2 = dx * dx
            for jj in range(j_lo, j_hi + 1):
                dy = (gmin[1] + jj * particle_spacing) - py
                dxy2 = dx2 + dy * dy
                for kk in range(k_lo, k_hi + 1):
                    dz = (gmin[2] + kk * particle_spacing) - pz
                    r2 = dxy2 + dz * dz
                    if r2 <= R_d_sq:
                        w = np.exp(-r2 / width)
                        grid_out[ii, jj, kk, 0] += w * cx
                        grid_out[ii, jj, kk, 1] += w * cy
                        grid_out[ii, jj, kk, 2] += w * cz


@ti.func
def _m4_prime_1d_ti(q: ti.f32) -> ti.f32:
    """Taichi 1D M4' kernel (Monaghan 1985).  Mirrors _m4_prime_1d."""
    r = ti.abs(q)
    w = 0.0
    if r <= 1.0:
        w = 1.0 - 2.5 * r * r + 1.5 * r * r * r
    elif r <= 2.0:
        w = 0.5 * (2.0 - r) * (2.0 - r) * (1.0 - r)
    return w


@ti.data_oriented
class _GridDiffusionMixin:
    """
    Mixin providing grid-based diffusion kernels and grid management.

    This mixin is shared between DiffusionPhysics and PhysicsEngine to
    ensure consistent grid-based viscous diffusion (DVH).
    """

    # Headroom factor: allocate this much larger than first observed grid
    # so slow vortex-cloud growth never triggers a reallocation.
    _ALLOC_HEADROOM = 1.5

    # How many times the DVH grid is allowed to re-allocate when the
    # particle cloud outgrows the initial allocation.
    _MAX_GRID_REALLOCS = 5

    # Headroom applied when *re*-allocating (larger than the initial
    # headroom to avoid repeated re-allocations as the wake extends).
    _REALLOC_HEADROOM = 2.0

    # Fallback ceiling for grid pre-allocation when the device pool is unknown
    # (CPU, Metal).  On CUDA/Vulkan the runtime pool published by the backend is
    # the authority instead; see _grid_prealloc_budget_bytes.
    _MAX_PREALLOC_BYTES: int = 1 << 30

    # Share of the device pool the diffusion workspace may claim.  The rest is
    # for particles, the treecode, evaluation fields and ndarray staging.
    _GRID_POOL_SHARE: float = 0.45

    # Thresholding is a population-control operation, not a physical sink.
    # Restore the discarded nodes onto the retained cloud while preserving the
    # diffused field's circulation and impulse moments.  The long-running cube
    # coupling method relied on this closure before it was removed during the
    # later GBD simplification.
    conserve_pruned_moments: bool = True

    def _init_grid_diffusion(self):
        """Initialize grid-based diffusion state."""
        self._grid_realloc_count: int = 0
        self._last_gbd_diffusion_substeps: int = 1
        self._last_gbd_moment_recovery = self._empty_gbd_moment_recovery()

        # Core radius assigned to regenerated particles (σ = ratio·particle_spacing).
        self.core_radius_ratio: float = _REGEN_RADIUS_RATIO

        # Maximum grid dimensions from VPM domain (set by configure_max_grid_extent).
        self._max_grid_dims: tuple[int, int, int] | None = None
        self._grid_domain_bounds: np.ndarray | None = None

        # Fixed grid origin when the domain is pre-allocated.
        self._fixed_grid_min: np.ndarray | None = None

        # Lazily allocated grid fields (never freed after first allocation).
        self._grid_a: ti.template() | None = None
        self._grid_b: ti.template() | None = None
        self._body_mask_grid: ti.template() | None = None  # 1=inside solid, 0=fluid
        self._effective_viscosity_grid: ti.template() | None = None
        self._grid_shape: tuple[int, int, int] | None = None

        # Ping-pong flag: True → _grid_a is the current/source field.
        self._ping: bool = True

        # Body-aware diffusion settings (optional; enabled when body STL is configured).
        self._body_mask_active: bool = False
        self._body_box_bounds: np.ndarray | None = None

        # Maximum number of grid cells per spatial dimension.
        self._max_cells_per_dimension: int = 2000
        self._require_fixed_grid_allocation: bool = False

    def require_fixed_grid_allocation(self, enabled: bool = True) -> None:
        """Require grid-based diffusion to pre-allocate one fixed grid.

        GPU backends use this because replacing Taichi fields leaks device memory
        and recompiles template kernels.
        """
        self._require_fixed_grid_allocation = bool(enabled)

    @property
    def _current_grid(self):
        """Field holding the current vorticity (source)."""
        return self._grid_a if self._ping else self._grid_b

    @property
    def last_gbd_diffusion_substeps(self) -> int:
        """Exact explicit-Laplacian stage count used by the last GBD operation."""
        return self._last_gbd_diffusion_substeps

    @staticmethod
    def _empty_gbd_moment_recovery() -> dict[str, bool | int | float]:
        return {
            "applied": False,
            "nonzero_node_count": 0,
            "retained_node_count": 0,
            "pruned_node_count": 0,
            "support_augmented_node_count": 0,
            "correction_fraction": 0.0,
            "normalized_vortex_strength_residual": 0.0,
            "normalized_linear_impulse_residual": 0.0,
            "normalized_angular_impulse_residual": 0.0,
        }

    @property
    def last_gbd_moment_recovery(self) -> dict[str, bool | int | float]:
        """Return a copy of the most recent GBD pruning-closure diagnostic."""
        return dict(
            getattr(
                self,
                "_last_gbd_moment_recovery",
                self._empty_gbd_moment_recovery(),
            )
        )

    @property
    def _other_grid(self):
        """Field that will receive the updated vorticity (destination)."""
        return self._grid_b if self._ping else self._grid_a

    # ---- Internal: grid management ----

    def _compute_grid_bounds(
        self,
        pos: np.ndarray,
        particle_spacing: float,
        padding: float,
        half_cell_offset: bool = True,
    ) -> tuple[np.ndarray, tuple[int, int, int]]:
        """Grid origin (float32) and integer dimensions from particle cloud.

        Robust against NaN position and stray outliers:

        1. Non-finite position are filtered before computing the bounding box.
        2. Each axis is capped at ``_max_cells_per_dimension`` cells. If the raw
           extent exceeds the cap, the domain is centred on the cloud and
           clamped, so a single stray particle cannot inflate the grid into
           an OOM allocation.

        Domain anchoring was removed (Feb 2026) because with the budget
        threshold the positive feedback loop it guarded against is no
        longer a concern, while its never-shrink property made it
        catastrophically sensitive to even one outlier.
        """
        # -- Filter non-finite position ----------------------------------
        finite_mask = np.isfinite(pos).all(axis=1)
        if not finite_mask.all():
            n_bad = int((~finite_mask).sum())
            Logging.warning(f"component=diffusion_grid filtered_nonfinite_positions={n_bad}")
            pos = pos[finite_mask]
            if len(pos) == 0:
                # Everything was NaN — return minimal grid
                lo = np.zeros(3, dtype=np.float32)
                return lo, (5, 5, 5)

        margin = padding * particle_spacing
        lo = pos.min(axis=0) - margin
        hi = pos.max(axis=0) + margin

        # -- Cap maximum grid extent per axis -----------------------------
        max_extent = self._max_cells_per_dimension * particle_spacing
        for d in range(3):
            span = hi[d] - lo[d]
            if span > max_extent:
                centre = 0.5 * (lo[d] + hi[d])
                lo[d] = centre - 0.5 * max_extent
                hi[d] = centre + 0.5 * max_extent
                Logging.warning(
                    f"component=diffusion_grid axis={d} requested_extent_m={span:.1f} "
                    f"extent_limit_m={max_extent:.1f} "
                    f"cell_limit={self._max_cells_per_dimension} "
                    "status=clamped"
                )

        nx = max(5, int(np.ceil((hi[0] - lo[0]) / particle_spacing)) + 1)
        ny = max(5, int(np.ceil((hi[1] - lo[1]) / particle_spacing)) + 1)
        nz = max(5, int(np.ceil((hi[2] - lo[2]) / particle_spacing)) + 1)

        # Half-cell offset to avoid particle-on-node coincidence (M4 aliasing).
        if half_cell_offset:
            lo = lo - 0.5 * particle_spacing

        return lo.astype(np.float32), (nx, ny, nz)

    def _lattice_aligned_bounds(
        self, pos: np.ndarray, particle_spacing: float, padding: float
    ) -> tuple[np.ndarray, tuple[int, int, int]]:
        """Active sub-box covering the cloud, with the origin on the fixed lattice.

        Returns ``(grid_min, (nx, ny, nz))``.  ``grid_min`` is offset from
        ``_fixed_grid_min`` by a whole number of cells, so every node keeps the
        position it would have on the full pre-allocated grid; only the extent
        shrinks to the occupied region.  Clamped to stay inside the allocation.
        """
        anchor = np.asarray(self._fixed_grid_min, dtype=np.float64).reshape(3)
        cap = np.asarray(self._max_grid_dims, dtype=np.int64)
        finite = np.isfinite(pos).all(axis=1)
        if self._grid_domain_bounds is not None:
            bounds = self._grid_domain_bounds
            finite &= (
                (pos[:, 0] >= bounds[0])
                & (pos[:, 0] <= bounds[1])
                & (pos[:, 1] >= bounds[2])
                & (pos[:, 1] <= bounds[3])
                & (pos[:, 2] >= bounds[4])
                & (pos[:, 2] <= bounds[5])
            )
        pts = pos[finite]
        if len(pts) == 0:
            return anchor.astype(np.float32), (5, 5, 5)

        margin = float(padding) * float(particle_spacing)
        lo = pts.min(axis=0) - margin
        hi = pts.max(axis=0) + margin

        first = np.floor((lo - anchor) / particle_spacing).astype(np.int64)
        last = np.ceil((hi - anchor) / particle_spacing).astype(np.int64)
        first = np.clip(first, 0, np.maximum(cap - 5, 0))
        last = np.clip(last, first + 4, cap - 1)
        grid_min = anchor + first * particle_spacing
        ext = last - first + 1
        return grid_min.astype(np.float32), (int(ext[0]), int(ext[1]), int(ext[2]))

    @staticmethod
    def _device_pool_bytes() -> int | None:
        """Taichi device memory pool in bytes, or None when self-managed."""
        from ...config import constants as constants_module

        return getattr(constants_module, "TAICHI_POOL_BYTES", None)

    def _grid_prealloc_budget_bytes(self) -> int:
        """Bytes the diffusion workspace may claim."""
        return self._MAX_PREALLOC_BYTES

    def _warn_if_grid_crowds_device_pool(self, total_bytes: int, nx: int, ny: int, nz: int) -> None:
        pool = self._device_pool_bytes()
        if not pool or total_bytes <= pool * self._GRID_POOL_SHARE:
            return
        Logging.warning(
            f"component=diffusion_grid shape={nx}x{ny}x{nz} "
            f"memory_mib={total_bytes / (1 << 20):.0f} "
            f"device_pool_pct={100.0 * total_bytes / pool:.0f} "
            f"device_pool_mib={pool / (1 << 20):.0f}"
        )

    def _ensure_grid_capacity(self, nx: int, ny: int, nz: int) -> tuple[int, int, int]:
        """Allocate ping-pong Taichi grid fields and return effective (nx, ny, nz).

        If ``configure_max_grid_extent`` was called, the grid is pre-allocated
        to the full VPM domain and no re-allocation is ever necessary.

        Otherwise, fields are allocated on the first call with headroom.  If
        the cloud outgrows the allocation, the grid is re-allocated (capped at
        ``_MAX_GRID_REALLOCS``).

        In either case, requested dimensions are clamped to
        ``_max_grid_dims`` (the VPM domain ceiling) when available.
        """
        # -- Clamp request to VPM-domain ceiling --------------------------
        cap = self._max_grid_dims
        if cap is not None:
            nx = min(nx, cap[0])
            ny = min(ny, cap[1])
            nz = min(nz, cap[2])
        elif self._require_fixed_grid_allocation:
            raise RuntimeError(
                "GPU DVH/GBD requires domain_bounds so the diffusion grid "
                "can be pre-allocated once. Refusing grow-on-demand grid allocation "
                "because Taichi 1.7.x retains replaced Vulkan fields until ti.reset()."
            )

        if self._grid_a is not None:
            alloc = self._grid_shape
            if nx <= alloc[0] and ny <= alloc[1] and nz <= alloc[2]:
                return nx, ny, nz

            if self._require_fixed_grid_allocation:
                raise RuntimeError(
                    "GPU DVH/GBD grid request exceeds the fixed pre-allocated "
                    f"grid {alloc}: requested {(nx, ny, nz)}. Increase "
                    "domain_bounds/padding or use CUDA/CPU for this run."
                )

            if self._grid_realloc_count >= self._MAX_GRID_REALLOCS:
                clamped = (min(nx, alloc[0]), min(ny, alloc[1]), min(nz, alloc[2]))
                Logging.warning(
                    f"component=diffusion_grid reallocations={self._grid_realloc_count} "
                    f"limit={self._MAX_GRID_REALLOCS} status=clamped shape={clamped} "
                    f"requested_shape={(nx, ny, nz)}"
                )
                return clamped

            # Re-allocate:
            self._grid_realloc_count += 1
            if cap is not None:
                alloc_nx, alloc_ny, alloc_nz = cap
            else:
                rh = self._REALLOC_HEADROOM
                alloc_nx = int(nx * rh) if nx > alloc[0] else alloc[0]
                alloc_ny = int(ny * rh) if ny > alloc[1] else alloc[1]
                alloc_nz = int(nz * rh) if nz > alloc[2] else alloc[2]
            Logging.warning(
                f"component=diffusion_grid reallocation={self._grid_realloc_count} "
                f"limit={self._MAX_GRID_REALLOCS} old_shape={alloc} "
                f"new_shape={(alloc_nx, alloc_ny, alloc_nz)} retained_device_fields=true"
            )
            self._grid_a = ti.Vector.field(3, dtype=ti.f32, shape=(alloc_nx, alloc_ny, alloc_nz))
            self._grid_b = ti.Vector.field(3, dtype=ti.f32, shape=(alloc_nx, alloc_ny, alloc_nz))
            self._body_mask_grid = ti.field(dtype=ti.i32, shape=(alloc_nx, alloc_ny, alloc_nz))
            self._effective_viscosity_grid = ti.field(
                dtype=ti.f32, shape=(alloc_nx, alloc_ny, alloc_nz)
            )
            self._grid_shape = (alloc_nx, alloc_ny, alloc_nz)
            self._ping = True
            return nx, ny, nz

        alloc_nx = int(nx * self._ALLOC_HEADROOM)
        alloc_ny = int(ny * self._ALLOC_HEADROOM)
        alloc_nz = int(nz * self._ALLOC_HEADROOM)

        _logger.debug(
            "diffusion grid allocating: requested shape %dx%dx%d, "
            "allocated shape %dx%dx%d, headroom %.1f",
            nx,
            ny,
            nz,
            alloc_nx,
            alloc_ny,
            alloc_nz,
            self._ALLOC_HEADROOM,
        )

        self._grid_a = ti.Vector.field(3, dtype=ti.f32, shape=(alloc_nx, alloc_ny, alloc_nz))
        self._grid_b = ti.Vector.field(3, dtype=ti.f32, shape=(alloc_nx, alloc_ny, alloc_nz))
        self._body_mask_grid = ti.field(dtype=ti.i32, shape=(alloc_nx, alloc_ny, alloc_nz))
        self._effective_viscosity_grid = ti.field(
            dtype=ti.f32, shape=(alloc_nx, alloc_ny, alloc_nz)
        )
        self._grid_shape = (alloc_nx, alloc_ny, alloc_nz)
        self._ping = True
        return nx, ny, nz

    def configure_max_grid_extent(
        self,
        domain_bounds: list[float],
        particle_spacing: float,
        padding: float = 3.0,
    ) -> None:
        """Set maximum grid-diffusion dimensions from VPM domain bounds.

        When the VPM domain is known, this method computes the grid that
        would cover the full domain (with padding) and either:

        * **Pre-allocates** the grid immediately if the memory cost is
          within ``_MAX_PREALLOC_BYTES`` (default 1 GB).  This avoids all
          re-allocation during the simulation.
        * **Stores the max dims** as a ceiling for future re-allocations
          if the full grid would exceed the memory budget.

        Parameters
        ----------
        domain_bounds : list[float]
            ``[xmin, xmax, ymin, ymax, zmin, zmax]`` of the VPM domain.
        particle_spacing : float
            Particle / grid spacing [m].
        padding : float
            Margin in multiples of *particle_spacing* added to each side.  Default 3.0.
        """
        import math

        margin = padding * particle_spacing
        nx = max(
            5, math.ceil((domain_bounds[1] - domain_bounds[0] + 2 * margin) / particle_spacing) + 1
        )
        ny = max(
            5, math.ceil((domain_bounds[3] - domain_bounds[2] + 2 * margin) / particle_spacing) + 1
        )
        nz = max(
            5, math.ceil((domain_bounds[5] - domain_bounds[4] + 2 * margin) / particle_spacing) + 1
        )
        self._max_grid_dims = (nx, ny, nz)
        self._grid_domain_bounds = np.asarray(domain_bounds, dtype=np.float64)

        # Store a fixed grid origin derived from the domain bounds.
        self._fixed_grid_min = np.array(
            [domain_bounds[0] - margin, domain_bounds[2] - margin, domain_bounds[4] - margin],
            dtype=np.float32,
        )

        # Memory estimate: 2 vector fields (3×f32), mask i32, effective_viscosity f32.
        bytes_per_node = 2 * 12 + 4 + 4  # = 32 bytes
        total_bytes = nx * ny * nz * bytes_per_node
        total_mb = total_bytes / (1 << 20)

        budget = self._grid_prealloc_budget_bytes()
        self._warn_if_grid_crowds_device_pool(total_bytes, nx, ny, nz)
        if total_bytes <= budget and self._grid_a is None:
            Logging.record(
                "diffusion grid preallocated",
                ("shape", f"{nx}x{ny}x{nz}"),
                ("memory", f"{total_mb:.0f}", "MiB"),
            )
            self._grid_a = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny, nz))
            self._grid_b = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny, nz))
            self._body_mask_grid = ti.field(dtype=ti.i32, shape=(nx, ny, nz))
            self._effective_viscosity_grid = ti.field(dtype=ti.f32, shape=(nx, ny, nz))
            self._grid_shape = (nx, ny, nz)
            self._ping = True
        else:
            if self._require_fixed_grid_allocation and self._grid_a is None:
                pool = self._device_pool_bytes()
                pool_txt = (
                    f"{pool / (1 << 20):.0f} MB device pool" if pool else "unknown device pool"
                )
                raise MemoryError(
                    "GPU DVH/GBD requires a fixed pre-allocated grid.\n"
                    f"  requested grid : {nx}x{ny}x{nz} = {nx * ny * nz:,} nodes\n"
                    f"  grid memory    : {total_mb:.0f} MB "
                    f"({bytes_per_node} B/node)\n"
                    f"  budget         : {budget / (1 << 20):.0f} MB "
                    f"({self._GRID_POOL_SHARE:.0%} of {pool_txt})\n"
                    "Reduce domain_bounds, coarsen the diffusion particle_spacing, or use CUDA/CPU."
                )
            allocation = "retained" if self._grid_a is not None else "deferred"
            reason = "existing_allocation" if self._grid_a is not None else "budget_exceeded"
            Logging.record(
                f"diffusion grid {allocation}",
                ("reason", reason.replace("_", " ")),
                ("shape, max", f"{nx}x{ny}x{nz}"),
                ("memory", f"{total_mb:.0f}", "MiB"),
            )

    def configure_grid_lattice_anchor(self, anchor, particle_spacing: float) -> None:
        """Align the fixed diffusion-grid phase with a coupled FVM lattice.

        Only the origin phase changes; the pre-allocated dimensions remain
        unchanged.  This avoids alternating between half-cell-shifted GBD and
        handoff lattices every coupling window.
        """
        if self._fixed_grid_min is None:
            return
        a = np.asarray(anchor, dtype=np.float64).reshape(3)
        particle_spacing = float(particle_spacing)
        if (
            not np.all(np.isfinite(a))
            or not np.isfinite(particle_spacing)
            or particle_spacing <= 0.0
        ):
            raise ValueError("grid lattice anchor and spacing must be finite")
        origin = np.asarray(self._fixed_grid_min, dtype=np.float64)
        origin = a + np.floor((origin - a) / particle_spacing) * particle_spacing
        self._fixed_grid_min = origin.astype(np.float32)

    def configure_body_mask(self, body_stl: str | None) -> None:
        """Configure optional body masking for DVH diffusion (not yet implemented).

        Parameters
        ----------
        body_stl : Optional[str]
            STL path to a closed body surface (reserved for future use).
        """
        if self._body_box_bounds is None:
            self._body_mask_active = False

    def configure_body_box(self, bounds) -> None:
        """Configure an exact axis-aligned solid mask for grid diffusion.

        The mask is rebuilt on the active diffusion lattice, so it remains
        correct for both fixed-domain and dynamically sized grids.  The GBD
        Laplacian treats solid neighbours as the fluid centre value (zero
        normal diffusive flux) and never regenerates particles at solid nodes.
        """
        b = np.asarray(bounds, dtype=np.float32).reshape(-1)
        if b.shape != (6,) or not np.all(np.isfinite(b)):
            raise ValueError("body box must contain six finite bounds")
        if np.any(b[1::2] <= b[::2]):
            raise ValueError("body box upper bounds must exceed lower bounds")
        self._body_box_bounds = b.copy()
        self._body_mask_active = True

    def _prepare_body_mask_current_grid(
        self, grid_min: np.ndarray, particle_spacing: float, nx: int, ny: int, nz: int
    ) -> None:
        """Populate the active grid's solid-node mask."""
        if (
            not self._body_mask_active
            or self._body_box_bounds is None
            or self._body_mask_grid is None
        ):
            return
        g = np.asarray(grid_min, dtype=np.float32).reshape(3)
        b = self._body_box_bounds
        self._fill_box_body_mask_kernel(
            self._body_mask_grid,
            float(g[0]),
            float(g[1]),
            float(g[2]),
            float(particle_spacing),
            float(b[0]),
            float(b[1]),
            float(b[2]),
            float(b[3]),
            float(b[4]),
            float(b[5]),
            nx,
            ny,
            nz,
        )

    # ---- Grid-diffusion orchestration ----

    def _apply_body_mask_current_grid(self, nx: int, ny: int, nz: int) -> None:
        """Zero vorticity inside masked (solid) cells on the active grid."""
        if not self._body_mask_active or self._body_mask_grid is None:
            return
        self._apply_body_mask_kernel(self._current_grid, self._body_mask_grid, nx, ny, nz)

    def _scatter_id_field(
        self,
        pos_np: np.ndarray,
        vortex_strength: np.ndarray,
        ids_np,
        grid_min_np: np.ndarray,
        particle_spacing: float,
        nx: int,
        ny: int,
        nz: int,
        default_id: int = 0,
        propagate_to: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        mapping: _NearestNodeMapping | None = None,
    ) -> np.ndarray:
        """Scatter an integer ID field to grid nodes (|Γ|-weighted nearest-node).

        Each grid node receives the ID of the dominant (by |Γ|-weight) particle
        whose nearest node it is.  Nodes listed in ``propagate_to`` that received
        no particle inherit the ID of their nearest populated node; all other
        empty nodes retain ``default_id``.

        Used for both ``zone_id`` (default 3) and ``group_id`` (default 0).
        """
        winner_grid = np.full((nx, ny, nz), default_id, dtype=np.int32)
        if ids_np is None:
            return winner_grid

        if mapping is None:
            mapping = _nearest_node_mapping(
                pos_np, vortex_strength, grid_min_np, particle_spacing, nx, ny, nz
            )
        w = mapping.vortex_strength_weight
        lin = mapping.linear_index
        ids = ids_np[mapping.valid]
        if len(ids) == 0:
            return winner_grid
        weight_flat = np.zeros(nx * ny * nz, dtype=np.float64)
        winner_flat = winner_grid.ravel()
        if np.all(ids == ids[0]):
            populated = np.unique(lin)
            winner_flat[populated] = ids[0]
            weight_flat[populated] = 1.0
            if propagate_to is not None:
                query = np.column_stack(propagate_to)
                winner_grid[tuple(query.T)] = ids[0]
                return winner_grid
        else:
            for id_val in np.unique(ids):
                mask_val = ids == id_val
                temp = np.zeros(nx * ny * nz, dtype=np.float64)
                np.add.at(temp, lin[mask_val], w[mask_val])
                better = temp > weight_flat
                winner_flat[better] = id_val
                weight_flat[better] = temp[better]

        if propagate_to is not None and np.any(weight_flat > 0.0):
            from scipy.spatial import cKDTree

            query = np.column_stack(propagate_to)
            query_linear = query[:, 0] * (ny * nz) + query[:, 1] * nz + query[:, 2]
            empty_query = weight_flat[query_linear] == 0.0
            if np.any(empty_query):
                populated_linear = np.flatnonzero(weight_flat > 0.0)
                populated = np.column_stack(np.unravel_index(populated_linear, (nx, ny, nz)))
                _, nearest = cKDTree(populated, compact_nodes=False).query(query[empty_query])
                empty_nodes = query[empty_query]
                winner_grid[tuple(empty_nodes.T)] = winner_flat[populated_linear[nearest]]
        return winner_grid

    def _scatter_zone_ids(
        self,
        pos_np: np.ndarray,
        vortex_strength: np.ndarray,
        zone_id_np,
        grid_min_np: np.ndarray,
        particle_spacing: float,
        nx: int,
        ny: int,
        nz: int,
        mapping: _NearestNodeMapping | None = None,
    ) -> np.ndarray:
        """Scatter zone IDs — thin wrapper around _scatter_id_field (default_id=3)."""
        return self._scatter_id_field(
            pos_np,
            vortex_strength,
            zone_id_np,
            grid_min_np,
            particle_spacing,
            nx,
            ny,
            nz,
            default_id=3,
            mapping=mapping,
        )

    def _scatter_scalar_weighted(
        self,
        pos_np: np.ndarray,
        vortex_strength: np.ndarray,
        scalar_np: np.ndarray,
        grid_min_np: np.ndarray,
        particle_spacing: float,
        nx: int,
        ny: int,
        nz: int,
        mapping: _NearestNodeMapping | None = None,
    ) -> np.ndarray:
        """Scatter a per-particle scalar to grid nodes (|Γ|-weighted average).

        Each grid node receives the vortex strength-magnitude-weighted average of
        the scalar over particles whose nearest node it is — mirroring
        ``_scatter_id_field``'s nearest-node winner scheme, but averaging a
        scalar instead of taking the dominant ID.  Nodes with no contributing
        particle get 0.0.

        Used to carry ν_t (Bug B) and ν_eff (Bug A) from the pre-regen
        particle cloud onto the diffusion grid so that regenerated particles
        and the variable-coefficient Laplacian inherit a per-node value.
        """
        out = np.zeros((nx, ny, nz), dtype=np.float32)
        if scalar_np is None:
            return out

        if mapping is None:
            mapping = _nearest_node_mapping(
                pos_np, vortex_strength, grid_min_np, particle_spacing, nx, ny, nz
            )
        w = mapping.vortex_strength_weight
        lin = mapping.linear_index
        s = np.ascontiguousarray(scalar_np[mapping.valid], dtype=np.float64)

        weight_flat = np.zeros(nx * ny * nz, dtype=np.float64)
        accum_flat = np.zeros(nx * ny * nz, dtype=np.float64)
        np.add.at(weight_flat, lin, w)
        np.add.at(accum_flat, lin, w * s)

        nz_mask = weight_flat > 0.0
        out_flat = out.ravel()
        out_flat[nz_mask] = (accum_flat[nz_mask] / weight_flat[nz_mask]).astype(np.float32)
        return out

    def _select_diffusion_threshold(
        self,
        vortex_strength_magnitude: np.ndarray,
        regen_threshold_mode: str,
        regen_threshold: float,
        max_vortex_strength_magnitude: float,
        vortex_strength_post_diffusion: float,
    ) -> float:
        """Return one cloud-wide magnitude threshold for regenerated nodes."""
        if regen_threshold_mode == "budget":
            vortex_strength_magnitude_sum = vortex_strength_post_diffusion
            if vortex_strength_magnitude_sum > 1e-30:
                flat = vortex_strength_magnitude.ravel()
                order = np.argsort(-flat)
                # The active diffusion grid can contain several hundred
                # thousand f32 nodes.  Accumulating that budget in f32 loses
                # enough low-magnitude tail strength to violate a nominal
                # 1e-4 retention target (observed 3.06e-4 loss in the
                # Lamb--Oseen DVH benchmark).  Selection is host-side already,
                # so use f64 accounting without changing the GPU field data.
                cumsum = np.cumsum(flat[order], dtype=np.float64)
                target = (1.0 - regen_threshold) * vortex_strength_magnitude_sum
                cutoff = min(int(np.searchsorted(cumsum, target)), len(order) - 1)
                threshold = float(flat[order[cutoff]])
            else:
                threshold = 1e-10
        elif regen_threshold_mode == "relative_max":
            threshold = regen_threshold * max_vortex_strength_magnitude
        elif regen_threshold_mode == "absolute":
            threshold = regen_threshold
        else:
            raise ValueError(
                f"Unknown regen_threshold_mode: {regen_threshold_mode!r}. Must be "
                "'budget', 'relative_max', or 'absolute'."
            )
        return max(threshold, 1e-10)

    @staticmethod
    def _regeneration_cap(particles, n_before: int, max_nodes: int | None) -> int:
        """Return the declared global particle limit for regeneration."""
        del n_before
        capacity = int(getattr(particles, "capacity", 0) or MAX_N_PARTICLES)
        cap = capacity
        if max_nodes is not None:
            cap = min(cap, int(max_nodes))
        return max(int(cap), 1)

    @staticmethod
    def _cap_surviving_nodes(
        vortex_strength_magnitude: np.ndarray,
        ix: np.ndarray,
        iy: np.ndarray,
        iz: np.ndarray,
        cap: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
        """Apply a cloud-wide strongest-node cap to regenerated particles."""
        n_survivors = len(ix)
        if cap <= 0:
            raise ValueError("Diffusion regeneration cap must be positive.")
        if n_survivors <= cap:
            threshold = float(vortex_strength_magnitude[ix, iy, iz].min()) if n_survivors else 0.0
            return ix, iy, iz, threshold, n_survivors

        values = vortex_strength_magnitude[ix, iy, iz]
        keep = np.argsort(-values, kind="stable")[:cap]
        ix_keep = ix[keep]
        iy_keep = iy[keep]
        iz_keep = iz[keep]
        threshold = float(vortex_strength_magnitude[ix_keep, iy_keep, iz_keep].min())
        return ix_keep, iy_keep, iz_keep, threshold, n_survivors

    @staticmethod
    def _augment_moment_recovery_support(
        grid_np: np.ndarray,
        vortex_strength_magnitude: np.ndarray,
        ix: np.ndarray,
        iy: np.ndarray,
        iz: np.ndarray,
        grid_min_np: np.ndarray,
        particle_spacing: float,
        cap: int,
        labels: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, bool]:
        """Introduce the strongest pruned nodes needed for a safe moment basis.

        The threshold/cap selection remains the physical population-control
        decision.  This method only augments that set when its geometry cannot
        carry the nine historical recovery constraints.  It first fills unused
        capacity, then (only when the cap is full) exchanges the weakest
        retained node for the strongest discarded candidate that improves the
        moment basis.  It never exceeds the cap.

        The final boolean selects historical per-group recovery.  If any group
        was sparse in the original threshold result, recovery remains global,
        exactly matching the former sparse-group fallback.
        """
        shape = vortex_strength_magnitude.shape
        flat_grid = grid_np.reshape(-1, 3)
        flat_magnitude = vortex_strength_magnitude.reshape(-1)
        retained_flat = np.ravel_multi_index((ix, iy, iz), shape).astype(np.int64)
        nonzero_flat = np.flatnonzero(flat_magnitude > 0.0)
        if len(retained_flat) >= len(nonzero_flat):
            return ix, iy, iz, 0, False

        retained_marker = np.zeros(flat_magnitude.shape, dtype=bool)
        retained_marker[retained_flat] = True
        candidate_flat = nonzero_flat[~retained_marker[nonzero_flat]]

        flat_labels = None if labels is None else np.asarray(labels).reshape(-1)
        preserve_groups = False
        unique_labels: np.ndarray | None = None
        if flat_labels is not None:
            unique_labels = np.unique(flat_labels[nonzero_flat])
            retained_labels = flat_labels[retained_flat]
            preserve_groups = len(unique_labels) > 1 and all(
                np.count_nonzero(retained_labels == label) >= 4 for label in unique_labels
            )

        selected = retained_flat.tolist()
        selected_set = set(selected)
        introduced: set[int] = set()
        retired: set[int] = set()

        def support_geometry(selected_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            sx, sy, sz = np.unravel_index(selected_flat, shape)
            position = np.stack(
                [
                    grid_min_np[0] + sx * particle_spacing,
                    grid_min_np[1] + sy * particle_spacing,
                    grid_min_np[2] + sz * particle_spacing,
                ],
                axis=1,
            ).astype(np.float64)
            strength = flat_grid[selected_flat].astype(np.float64)
            return position, strength

        def support_quality(selected_flat: np.ndarray) -> tuple[int, float]:
            position, strength = support_geometry(selected_flat)
            return _gbd_moment_support_quality(
                position,
                strength,
                particle_spacing,
            )

        def quality_improves(
            trial_rank: int,
            trial_condition: float,
            rank: int,
            condition: float,
        ) -> bool:
            if trial_rank != rank:
                return trial_rank > rank
            if not np.isfinite(trial_condition):
                return False
            return not np.isfinite(condition) or trial_condition < condition * (1.0 - 1.0e-10)

        def augment_scope(label: int | None) -> None:
            def scope_selection(*, with_indices: bool = False) -> np.ndarray:
                selected_array = np.asarray(selected, dtype=np.int64)
                if label is None or flat_labels is None:
                    selection = np.arange(len(selected_array), dtype=np.int64)
                else:
                    selection = np.flatnonzero(flat_labels[selected_array] == label)
                return selection if with_indices else selected_array[selection]

            scoped_candidates = candidate_flat
            if label is not None and flat_labels is not None:
                scoped_candidates = scoped_candidates[flat_labels[scoped_candidates] == label]
            # Per-group closure does not require a basis for a group from which
            # no node was pruned.
            if len(scoped_candidates) == 0:
                return

            rank, condition = support_quality(scope_selection())
            if rank == 9 and condition <= _GBD_MOMENT_CONDITION_LIMIT:
                return

            candidate_order = np.argsort(
                -flat_magnitude[scoped_candidates],
                kind="stable",
            )
            scoped_candidates = scoped_candidates[candidate_order]

            for candidate in scoped_candidates:
                if len(selected) >= cap:
                    break
                candidate_int = int(candidate)
                if candidate_int in selected_set:
                    continue
                selected.append(candidate_int)
                selected_set.add(candidate_int)
                introduced.add(candidate_int)
                rank, condition = support_quality(scope_selection())
                if rank == 9 and condition <= _GBD_MOMENT_CONDITION_LIMIT:
                    return
            else:
                # Every nonzero node in this scope is now retained, so there is
                # no discarded strength for its moment basis to redistribute.
                return

            # The cap is full but its support can still be geometrically
            # deficient (for example, four nearly collinear strongest nodes).
            # Keep the strongest selected nodes whenever possible: examine a
            # bounded weakest-node frontier and accept the first, strongest
            # discarded candidate that gives a strict rank/condition gain.
            while len(selected) >= cap:
                scoped_flat = scope_selection()
                if len(scoped_flat) < 4:
                    break
                position, strength = support_geometry(scoped_flat)
                gram, length_scale, _weights = _gbd_moment_gram(
                    position,
                    strength,
                    particle_spacing,
                )
                rank, condition = _gbd_moment_gram_quality(gram)
                if rank == 9 and condition <= _GBD_MOMENT_CONDITION_LIMIT:
                    return
                raw_gram = gram * float(np.linalg.norm(strength, axis=1).sum(dtype=np.float64))

                scope_indices = scope_selection(with_indices=True)
                weakest_order = np.argsort(
                    flat_magnitude[scoped_flat],
                    kind="stable",
                )[:_GBD_MOMENT_REPLACEMENT_POOL_SIZE]
                replacement_indices = scope_indices[weakest_order]
                replacement_flat = scoped_flat[weakest_order]
                replacement_rows = _gbd_moment_constraint_rows(
                    position[weakest_order] / length_scale
                )
                replacement_contributions = np.einsum(
                    "i,ijk,ilk->ijl",
                    np.linalg.norm(strength[weakest_order], axis=1),
                    replacement_rows,
                    replacement_rows,
                )

                accepted = False
                for candidate in scoped_candidates:
                    candidate_int = int(candidate)
                    if candidate_int in selected_set or candidate_int in retired:
                        continue
                    cx, cy, cz = np.unravel_index(candidate_int, shape)
                    candidate_position = np.asarray(
                        [
                            grid_min_np[0] + cx * particle_spacing,
                            grid_min_np[1] + cy * particle_spacing,
                            grid_min_np[2] + cz * particle_spacing,
                        ],
                        dtype=np.float64,
                    )
                    candidate_rows = _gbd_moment_constraint_rows(
                        candidate_position[None, :] / length_scale
                    )[0]
                    candidate_contribution = flat_magnitude[candidate_int] * (
                        candidate_rows @ candidate_rows.T
                    )

                    improving_replacements: list[int] = []
                    for local_index, removed_contribution in enumerate(replacement_contributions):
                        trial_gram = raw_gram - removed_contribution + candidate_contribution
                        trial_rank, trial_condition = _gbd_moment_gram_quality(trial_gram)
                        if not quality_improves(
                            trial_rank,
                            trial_condition,
                            rank,
                            condition,
                        ):
                            continue
                        improving_replacements.append(local_index)
                    if not improving_replacements:
                        continue

                    # ``weakest_order`` is ascending, so retain every stronger
                    # threshold/cap winner unless replacing a weaker node does
                    # not produce a genuine full-support improvement after the
                    # exact length-scale normalization is recomputed.
                    for local_index in improving_replacements:
                        replacement_index = int(replacement_indices[local_index])
                        removed_flat = int(replacement_flat[local_index])
                        selected[replacement_index] = candidate_int
                        trial_rank, trial_condition = support_quality(scope_selection())
                        if not quality_improves(
                            trial_rank,
                            trial_condition,
                            rank,
                            condition,
                        ):
                            selected[replacement_index] = removed_flat
                            continue

                        selected_set.remove(removed_flat)
                        selected_set.add(candidate_int)
                        retired.add(removed_flat)
                        introduced.discard(removed_flat)
                        introduced.add(candidate_int)
                        rank, condition = trial_rank, trial_condition
                        accepted = True
                        break
                    if accepted:
                        break
                if not accepted:
                    break
                if rank == 9 and condition <= _GBD_MOMENT_CONDITION_LIMIT:
                    return
            scope_name = "the complete cloud" if label is None else f"group {label!r}"
            raise RuntimeError(
                "GBD moment recovery cannot form full-rank support within the "
                f"regeneration cap for {scope_name}: rank={rank}/9, "
                f"condition={condition:.6e}, retained={len(selected):,}, cap={cap:,}"
            )

        if preserve_groups:
            assert unique_labels is not None
            for label in unique_labels:
                augment_scope(int(label))
        else:
            augment_scope(None)

        augmented_flat = np.asarray(selected, dtype=np.int64)
        augmented_ix, augmented_iy, augmented_iz = np.unravel_index(
            augmented_flat,
            shape,
        )
        return (
            np.asarray(augmented_ix),
            np.asarray(augmented_iy),
            np.asarray(augmented_iz),
            len(introduced),
            preserve_groups,
        )

    @staticmethod
    def _redistribute_pruned_moments(
        grid_np: np.ndarray,
        vortex_strength_magnitude: np.ndarray,
        ix: np.ndarray,
        iy: np.ndarray,
        iz: np.ndarray,
        grid_min_np: np.ndarray,
        particle_spacing: float,
        labels: np.ndarray | None = None,
        diagnostics: dict[str, bool | int | float] | None = None,
    ) -> np.ndarray:
        """Coarsen pruned nodes locally while preserving diffusive moments.

        The retained cloud preserves total vortex strength, linear impulse, and
        angular impulse of the complete post-diffusion grid.  A nearest-node
        coarsening first keeps discarded tail strength close to its physical
        location; a weighted minimum-norm correction then closes the nine
        moment constraints.  Discarded nodes alone are queried against the
        retained-node tree, and the constraint algebra is evaluated in bounded
        chunks so memory use does not grow by 27 float64 values per survivor.

        The returned values have already been rounded to the storage dtype and
        validated.  Singular geometry, excessive correction, or a failed
        post-cast moment closure raises before the production grid is mutated.
        """
        retained_vortex_strength = grid_np[ix, iy, iz].astype(np.float64)
        retained_position = np.stack(
            [
                grid_min_np[0] + ix * particle_spacing,
                grid_min_np[1] + iy * particle_spacing,
                grid_min_np[2] + iz * particle_spacing,
            ],
            axis=1,
        ).astype(np.float64)

        discarded_mask = vortex_strength_magnitude > 0.0
        discarded_mask[ix, iy, iz] = False
        discarded_i, discarded_j, discarded_k = np.where(discarded_mask)
        if len(discarded_i) == 0:
            if diagnostics is not None:
                diagnostics.update(
                    correction_fraction=0.0,
                    normalized_vortex_strength_residual=0.0,
                    normalized_linear_impulse_residual=0.0,
                    normalized_angular_impulse_residual=0.0,
                )
            return retained_vortex_strength.astype(grid_np.dtype, copy=False)
        discarded_vortex_strength = grid_np[
            discarded_i,
            discarded_j,
            discarded_k,
        ].astype(np.float64)
        discarded_position = np.stack(
            [
                grid_min_np[0] + discarded_i * particle_spacing,
                grid_min_np[1] + discarded_j * particle_spacing,
                grid_min_np[2] + discarded_k * particle_spacing,
            ],
            axis=1,
        ).astype(np.float64)

        def moment_state(
            position: np.ndarray,
            vortex_strength: np.ndarray,
        ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[float, float, float]]:
            linear_terms = np.cross(position, vortex_strength)
            angular_terms = np.cross(position, linear_terms) / 3.0
            moments = (
                vortex_strength.sum(axis=0, dtype=np.float64),
                linear_terms.sum(axis=0, dtype=np.float64),
                angular_terms.sum(axis=0, dtype=np.float64),
            )
            scales = (
                float(np.linalg.norm(vortex_strength, axis=1).sum(dtype=np.float64)),
                float(np.linalg.norm(linear_terms, axis=1).sum(dtype=np.float64)),
                float(np.linalg.norm(angular_terms, axis=1).sum(dtype=np.float64)),
            )
            return moments, scales

        def validate(
            survivor_position: np.ndarray,
            raw_survivor_strength: np.ndarray,
            removed_position: np.ndarray,
            removed_strength: np.ndarray,
            corrected: np.ndarray,
            *,
            scope: str,
            record_diagnostics: bool = False,
        ) -> np.ndarray:
            stored = corrected.astype(grid_np.dtype, copy=False)
            stored_float64 = stored.astype(np.float64, copy=False)
            if not np.all(np.isfinite(stored_float64)):
                raise RuntimeError(f"GBD moment recovery produced non-finite strengths for {scope}")

            survivor_moments, survivor_scales = moment_state(
                survivor_position,
                raw_survivor_strength,
            )
            removed_moments, removed_scales = moment_state(
                removed_position,
                removed_strength,
            )
            target_moments = tuple(
                survivor + removed
                for survivor, removed in zip(
                    survivor_moments,
                    removed_moments,
                    strict=True,
                )
            )
            target_scales = tuple(
                survivor + removed
                for survivor, removed in zip(
                    survivor_scales,
                    removed_scales,
                    strict=True,
                )
            )
            actual_moments, _actual_scales = moment_state(
                survivor_position,
                stored_float64,
            )

            complete_strength_scale = max(target_scales[0], np.finfo(np.float64).tiny)
            position_scale = max(
                float(
                    np.sqrt(
                        np.einsum(
                            "ij,ij->i",
                            survivor_position,
                            survivor_position,
                        ).mean()
                    )
                ),
                float(particle_spacing),
            )
            normalization = (
                complete_strength_scale,
                max(target_scales[1], complete_strength_scale * position_scale),
                max(target_scales[2], complete_strength_scale * position_scale**2),
            )
            relative_residual = np.asarray(
                [
                    np.linalg.norm(target - actual) / scale
                    for target, actual, scale in zip(
                        target_moments,
                        actual_moments,
                        normalization,
                        strict=True,
                    )
                ],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(relative_residual)) or np.any(
                relative_residual > _GBD_MOMENT_RESIDUAL_LIMIT
            ):
                raise RuntimeError(
                    "GBD moment recovery failed its post-cast closure for "
                    f"{scope}: normalized residuals "
                    f"gamma={relative_residual[0]:.6e}, "
                    f"linear={relative_residual[1]:.6e}, "
                    f"angular={relative_residual[2]:.6e}, "
                    f"limit={_GBD_MOMENT_RESIDUAL_LIMIT:.6e}"
                )

            correction_fraction = float(
                np.linalg.norm(stored_float64 - raw_survivor_strength, axis=1).sum(dtype=np.float64)
                / complete_strength_scale
            )
            if (
                not np.isfinite(correction_fraction)
                or correction_fraction > _GBD_MOMENT_CORRECTION_FRACTION_LIMIT
            ):
                raise RuntimeError(
                    "GBD moment recovery requires an excessive strength correction for "
                    f"{scope}: fraction={correction_fraction:.6e}, "
                    f"limit={_GBD_MOMENT_CORRECTION_FRACTION_LIMIT:.6e}"
                )
            if record_diagnostics and diagnostics is not None:
                diagnostics.update(
                    correction_fraction=correction_fraction,
                    normalized_vortex_strength_residual=float(relative_residual[0]),
                    normalized_linear_impulse_residual=float(relative_residual[1]),
                    normalized_angular_impulse_residual=float(relative_residual[2]),
                )
            return stored

        def redistribute(
            survivor_position: np.ndarray,
            survivor_vortex_strength: np.ndarray,
            removed_position: np.ndarray,
            removed_vortex_strength: np.ndarray,
            *,
            scope: str,
            record_diagnostics: bool = False,
        ) -> np.ndarray:
            weight_magnitude = np.linalg.norm(survivor_vortex_strength, axis=1)
            weight_sum = float(weight_magnitude.sum())
            if not np.isfinite(weight_sum) or weight_sum <= 0.0:
                raise RuntimeError(
                    f"GBD moment recovery has no finite retained strength for {scope}"
                )
            if len(removed_position) == 0:
                return survivor_vortex_strength.astype(grid_np.dtype, copy=False)

            from scipy.spatial import cKDTree

            nearest = cKDTree(survivor_position, compact_nodes=False).query(
                removed_position,
                k=1,
                workers=-1,
            )[1]
            corrected = survivor_vortex_strength.copy()
            np.add.at(corrected, nearest, removed_vortex_strength)

            survivor_moments, _survivor_scales = moment_state(
                survivor_position,
                survivor_vortex_strength,
            )
            removed_moments, _removed_scales = moment_state(
                removed_position,
                removed_vortex_strength,
            )
            corrected_moments, _corrected_scales = moment_state(
                survivor_position,
                corrected,
            )
            residuals = tuple(
                survivor + removed - current
                for survivor, removed, current in zip(
                    survivor_moments,
                    removed_moments,
                    corrected_moments,
                    strict=True,
                )
            )
            if not any(np.any(np.abs(residual) > 0.0) for residual in residuals):
                return validate(
                    survivor_position,
                    survivor_vortex_strength,
                    removed_position,
                    removed_vortex_strength,
                    corrected,
                    scope=scope,
                    record_diagnostics=record_diagnostics,
                )
            if len(survivor_position) < 4:
                raise RuntimeError(
                    "GBD moment recovery requires at least four retained nodes "
                    f"for {scope}; got {len(survivor_position)}"
                )

            gram, length_scale, weights = _gbd_moment_gram(
                survivor_position,
                survivor_vortex_strength,
                particle_spacing,
            )
            right_hand_side = np.concatenate(
                (
                    residuals[0],
                    residuals[1] / length_scale,
                    residuals[2] / length_scale**2,
                )
            )
            if not np.all(np.isfinite(gram)) or not np.all(np.isfinite(right_hand_side)):
                raise RuntimeError(f"GBD moment recovery formed non-finite constraints for {scope}")

            multipliers, _residual, rank, singular_values = np.linalg.lstsq(
                gram,
                right_hand_side,
                rcond=_GBD_MOMENT_RCOND,
            )
            condition = (
                float("inf")
                if len(singular_values) == 0 or singular_values[-1] <= 0.0
                else float(singular_values[0] / singular_values[-1])
            )
            if rank < 9 or not np.isfinite(condition) or condition > _GBD_MOMENT_CONDITION_LIMIT:
                raise RuntimeError(
                    "GBD moment recovery constraints are rank-deficient or ill-conditioned "
                    f"for {scope}: rank={rank}/9, condition={condition:.6e}, "
                    f"limit={_GBD_MOMENT_CONDITION_LIMIT:.6e}"
                )
            if not np.all(np.isfinite(multipliers)):
                raise RuntimeError(
                    f"GBD moment recovery produced non-finite multipliers for {scope}"
                )

            recovered = corrected.copy()
            for start in range(0, len(survivor_position), _GBD_MOMENT_CHUNK_SIZE):
                stop = min(start + _GBD_MOMENT_CHUNK_SIZE, len(survivor_position))
                rows = _gbd_moment_constraint_rows(survivor_position[start:stop] / length_scale)
                recovered[start:stop] += np.einsum(
                    "i,ijk,j->ik",
                    weights[start:stop],
                    rows,
                    multipliers,
                )
            return validate(
                survivor_position,
                survivor_vortex_strength,
                removed_position,
                removed_vortex_strength,
                recovered,
                scope=scope,
                record_diagnostics=record_diagnostics,
            )

        if labels is None:
            return redistribute(
                retained_position,
                retained_vortex_strength,
                discarded_position,
                discarded_vortex_strength,
                scope="the complete cloud",
                record_diagnostics=True,
            )

        survivor_labels = labels[ix, iy, iz]
        discarded_labels = labels[discarded_i, discarded_j, discarded_k]
        unique_labels = np.union1d(survivor_labels, discarded_labels)
        if len(unique_labels) == 1:
            return redistribute(
                retained_position,
                retained_vortex_strength,
                discarded_position,
                discarded_vortex_strength,
                scope=f"group {unique_labels[0]!r}",
                record_diagnostics=True,
            )
        corrected = np.zeros_like(retained_vortex_strength)
        for label in unique_labels:
            survivor_selection = survivor_labels == label
            discarded_selection = discarded_labels == label
            if np.count_nonzero(survivor_selection) < 4:
                return redistribute(
                    retained_position,
                    retained_vortex_strength,
                    discarded_position,
                    discarded_vortex_strength,
                    scope=f"the complete cloud (sparse group {label!r})",
                    record_diagnostics=True,
                )
            corrected[survivor_selection] = redistribute(
                retained_position[survivor_selection],
                retained_vortex_strength[survivor_selection],
                discarded_position[discarded_selection],
                discarded_vortex_strength[discarded_selection],
                scope=f"group {label!r}",
            )
        return validate(
            retained_position,
            retained_vortex_strength,
            discarded_position,
            discarded_vortex_strength,
            corrected,
            scope="the complete cloud after per-group recovery",
            record_diagnostics=True,
        )

    def _build_diffusion_particle_arrays(
        self,
        ix: np.ndarray,
        iy: np.ndarray,
        iz: np.ndarray,
        grid_np: np.ndarray,
        grid_min_np: np.ndarray,
        particle_spacing: float,
        kinematic_viscosity: float,
        time_step_size: float,
        particles,
        N: int,
        zone_winner_grid: np.ndarray,
        group_winner_grid: np.ndarray,
        eddy_viscosity_grid: np.ndarray | None = None,
    ) -> dict:
        """Assemble the new-particle dict from the diffused grid (3-D)."""
        M = len(ix)
        # 3-D cell particle_volume: particle_spacing³
        vol = float(particle_spacing) ** 3
        r = float(self.core_radius_ratio) * float(particle_spacing)
        new_pos = np.stack(
            [
                grid_min_np[0] + ix * particle_spacing,
                grid_min_np[1] + iy * particle_spacing,
                grid_min_np[2] + iz * particle_spacing,
            ],
            axis=1,
        ).astype(np.float32)
        # Carry the pre-regen turbulent viscosity forward: reconstructs ν_eff = ν + ν_t.
        if eddy_viscosity_grid is not None:
            eddy_viscosity = eddy_viscosity_grid[ix, iy, iz].astype(np.float32)
        else:
            eddy_viscosity = np.zeros(M, dtype=np.float32)
        return {
            "position": new_pos,
            "vortex_strength": grid_np[ix, iy, iz].astype(np.float32),
            "velocity": np.zeros((M, 3), dtype=np.float32),
            "particle_volume": np.full(M, vol, dtype=np.float32),
            "core_radius": np.full(M, r, dtype=np.float32),
            "kinematic_viscosity": np.full(M, kinematic_viscosity, dtype=np.float32),
            "eddy_viscosity": eddy_viscosity,
            "zone_id": zone_winner_grid[ix, iy, iz].astype(np.int32),
            "group_id": group_winner_grid[ix, iy, iz].astype(np.int32),
        }

    @staticmethod
    def _explicit_diffusion_substep_count(
        max_diffusivity: float,
        time_step_size: float,
        particle_spacing: float,
    ) -> tuple[int, float]:
        """Return stable forward-Euler stages for the 3-D 7-point Laplacian."""
        values = np.asarray([max_diffusivity, time_step_size, particle_spacing], dtype=np.float64)
        if not np.isfinite(values).all():
            raise FloatingPointError(
                "GBD diffusion requires finite diffusivity, time step, and grid spacing."
            )
        if max_diffusivity < 0.0:
            raise ValueError("GBD diffusion requires non-negative effective viscosity.")
        if time_step_size < 0.0:
            raise ValueError("GBD diffusion requires a non-negative time step.")
        if particle_spacing <= 0.0:
            raise ValueError("GBD diffusion requires a positive particle spacing.")

        max_diffusion_number = float(max_diffusivity * time_step_size / particle_spacing**2)
        if not np.isfinite(max_diffusion_number):
            raise FloatingPointError("GBD diffusion produced a non-finite Fourier number.")
        if max_diffusion_number == 0.0:
            return 1, 0.0

        # The most negative eigenvalue of the dimensionless 3-D 7-point
        # Laplacian is -12.  Requiring alpha < 1/12 keeps every modal
        # amplification in (0, 1], so diffusion cannot reverse a grid mode.
        scaled = 12.0 * max_diffusion_number
        if not np.isfinite(scaled):
            raise FloatingPointError("GBD diffusion requires an unrepresentable stage count.")
        n_diffusion_substeps = max(1, math.floor(scaled) + 1)

        # Kernels run in f32 in production.  Guard against a nominally safe
        # host-side value rounding back onto or above the stability boundary.
        diffusivity_f32 = np.float32(max_diffusivity)
        spacing_f32 = np.float32(particle_spacing)
        spacing_sq_f32 = np.float32(spacing_f32 * spacing_f32)
        if not np.isfinite(diffusivity_f32) or not np.isfinite(spacing_sq_f32):
            raise FloatingPointError("GBD diffusion inputs exceed the production precision.")
        if spacing_sq_f32 == 0.0:
            raise FloatingPointError("GBD grid spacing underflows the production precision.")
        while True:
            substep_f32 = np.float32(time_step_size / n_diffusion_substeps)
            variable_alpha_f32 = np.float32(
                np.float32(diffusivity_f32 * substep_f32) / spacing_sq_f32
            )
            scalar_alpha_f32 = np.float32(
                max_diffusivity * (time_step_size / n_diffusion_substeps) / particle_spacing**2
            )
            if max(float(variable_alpha_f32), float(scalar_alpha_f32)) < 1.0 / 12.0:
                break
            n_diffusion_substeps += 1
        return n_diffusion_substeps, max_diffusion_number

    @classmethod
    def gbd_diffusion_substep_count(
        cls,
        max_diffusivity: float,
        time_step_size: float,
        particle_spacing: float,
    ) -> int:
        """Return the production GBD stage count for the supplied runtime state."""
        return cls._explicit_diffusion_substep_count(
            max_diffusivity,
            time_step_size,
            particle_spacing,
        )[0]

    def _advance_gbd_laplacian(
        self,
        *,
        nx: int,
        ny: int,
        nz: int,
        time_step_size: float,
        particle_spacing: float,
        kinematic_viscosity: float,
        effective_viscosity_grid: np.ndarray | None,
    ) -> tuple[int, float]:
        """Advance the frozen GBD grid operator with stable explicit stages."""
        if effective_viscosity_grid is None:
            max_diffusivity = float(kinematic_viscosity)
        else:
            max_diffusivity = (
                float(np.max(effective_viscosity_grid)) if effective_viscosity_grid.size else 0.0
            )

        n_diffusion_substeps, max_diffusion_number = self._explicit_diffusion_substep_count(
            max_diffusivity,
            time_step_size,
            particle_spacing,
        )
        diffusion_substep_size = float(time_step_size) / n_diffusion_substeps
        max_substep_diffusion_number = max_diffusion_number / n_diffusion_substeps

        if n_diffusion_substeps > 1:
            Logging.record(
                "gaussian blob diffusion substepping",
                ("substeps", f"{n_diffusion_substeps:,}"),
                ("diffusion number, max", f"{max_diffusion_number:.6f}"),
                ("diffusion number, max per substep", f"{max_substep_diffusion_number:.6f}"),
                ("substep size", f"{diffusion_substep_size:.6e}", "s"),
            )

        self._zero_grid_kernel(self._other_grid, nx, ny, nz)
        if effective_viscosity_grid is not None:
            self._upload_active_scalar_grid(
                self._effective_viscosity_grid, effective_viscosity_grid, nx, ny, nz
            )

        for _ in range(n_diffusion_substeps):
            if effective_viscosity_grid is None:
                diffusion_number = float(
                    kinematic_viscosity * diffusion_substep_size / particle_spacing**2
                )
                self._laplacian_step_gpu_kernel(
                    self._current_grid,
                    self._other_grid,
                    self._body_mask_grid,
                    diffusion_number,
                    nx,
                    ny,
                    nz,
                )
            else:
                self._laplacian_step_variable_gpu_kernel(
                    self._current_grid,
                    self._other_grid,
                    self._effective_viscosity_grid,
                    self._body_mask_grid,
                    diffusion_substep_size,
                    particle_spacing,
                    nx,
                    ny,
                    nz,
                )
            self._ping = not self._ping

        return n_diffusion_substeps, max_diffusion_number

    def _gbd_diffusion_impl(
        self,
        particles,
        time_step_size: float,
        particle_spacing: float,
        kinematic_viscosity: float,
        domain_padding: float = 3.0,
        regen_threshold: float = 0.01,
        regen_threshold_mode: str = "budget",
        effective_viscosity: np.ndarray | None = None,
        max_nodes: int | None = None,
    ) -> dict[str, np.ndarray] | None:
        """GBD diffusion + particle regeneration (Cottet & Koumoutsakos 2000).

        Algorithm
        ---------
        1. M4' scatter: particle vortex strength → grid (GPU Taichi kernel).
        2. Explicit Laplacian: stable forward-Euler substeps of kinematic_viscosity∇²ω
           (GPU Taichi kernel).
           When ``effective_viscosity`` (per-particle ν+ν_t) is given, the Laplacian uses a
           per-node coefficient α_node = ν_eff·dt/particle_spacing² instead of the
           scalar molecular α = ν·dt/particle_spacing² — so the Smagorinsky SGS model acts in
           GBD runs just as it does in DVH/CS/RWM.
        3. Threshold pruning: discard weak grid nodes  (CPU NumPy — small).
        4. Spawn new particles at surviving nodes       (CPU NumPy).

        Steps 1-2 run entirely on GPU with no intermediate CPU↔GPU grid
        transfers.  Only position are read to CPU for grid-bounds computation
        (cached), and the diffused grid is read back once for pruning.

        If the macro-step exceeds the 3-D explicit limit, only the Laplacian is
        substepped. Scatter, thresholding, and regeneration still occur once.
        """
        # One is also the correct reach for a skipped/zero-diffusivity GBD
        # operation. A completed Laplacian below replaces this with the exact
        # runtime stage count used by the current regeneration.
        self._last_gbd_diffusion_substeps = 1
        self._last_gbd_moment_recovery = self._empty_gbd_moment_recovery()
        N = particles.n_particles_total
        if N == 0:
            return None
        self._ping = True

        try:
            zone_id_np = particles.zone_id_cpu().copy()
        except (AttributeError, Exception):
            zone_id_np = None
        group_id_np = particles.group_id_cpu()[:N].copy()

        pos_np = particles.position_cpu()
        vortex_strength = particles.vortex_strength_cpu()

        # -- LES: per-particle ν_t to carry through regen  -------------
        # The scattered ν_t is inherited by regenerated particles so that ν_t
        # survives the rebuild and reaches the backup (LES recomputes it
        # next step anyway, but carrying it keeps the backuped field
        # faithful).
        eddy_viscosity_np = particles.eddy_viscosity_cpu()

        # -- Grid setup --------------------------------------------------------
        # Use a fixed grid origin when the domain was pre-configured, to avoid
        # the asymmetric flat-end artefact (see _fixed_grid_min docstring).
        if self._fixed_grid_min is not None and self._max_grid_dims is not None:
            grid_min_np, (nx, ny, nz) = self._lattice_aligned_bounds(
                pos_np, particle_spacing, domain_padding
            )
            nx, ny, nz = self._ensure_grid_capacity(nx, ny, nz)
        else:
            grid_min_np, (nx, ny, nz) = self._compute_grid_bounds(
                pos_np,
                particle_spacing,
                domain_padding,
                half_cell_offset=False,
            )
            nx, ny, nz = self._ensure_grid_capacity(nx, ny, nz)
        node_mapping = _nearest_node_mapping(
            pos_np, vortex_strength, grid_min_np, particle_spacing, nx, ny, nz
        )

        # -- M4' scatter (GPU) -------------------------------------------------
        self._zero_grid_kernel(self._current_grid, nx, ny, nz)
        gmin = grid_min_np.astype(float)
        for start in range(0, N, _M4_SCATTER_BATCH_SIZE):
            count = min(_M4_SCATTER_BATCH_SIZE, N - start)
            self._m4_scatter_gpu_kernel(
                particles.position,
                particles.vortex_strength,
                self._current_grid,
                gmin[0],
                gmin[1],
                gmin[2],
                float(particle_spacing),
                nx,
                ny,
                nz,
                start,
                count,
            )
            ti.sync()
        self._prepare_body_mask_current_grid(grid_min_np, particle_spacing, nx, ny, nz)
        self._apply_body_mask_current_grid(nx, ny, nz)

        # -- Explicit Laplacian diffusion (GPU) --------------------------------
        # When ν_eff (ν+ν_t) is supplied, use a per-node coefficient
        # α_node = ν_eff·dt/particle_spacing² so the SGS eddy viscosity acts in GBD runs.
        # Otherwise fall back to the scalar molecular α = ν·dt/particle_spacing².
        effective_viscosity_grid_np = None
        if effective_viscosity is not None:
            effective_viscosity_np = np.ascontiguousarray(effective_viscosity[:N], dtype=np.float32)
            # Clip negatives (defensive: Smagorinsky guards already, but the
            # grid scatter can introduce tiny excursions via round-off).
            np.clip(effective_viscosity_np, 0.0, None, out=effective_viscosity_np)
            effective_viscosity_grid_np = self._scatter_scalar_weighted(
                pos_np,
                vortex_strength,
                effective_viscosity_np,
                grid_min_np,
                particle_spacing,
                nx,
                ny,
                nz,
                mapping=node_mapping,
            )
        self._last_gbd_diffusion_substeps, _max_diffusion_number = self._advance_gbd_laplacian(
            nx=nx,
            ny=ny,
            nz=nz,
            time_step_size=float(time_step_size),
            particle_spacing=float(particle_spacing),
            kinematic_viscosity=float(kinematic_viscosity),
            effective_viscosity_grid=effective_viscosity_grid_np,
        )

        # -- Body mask (GPU, optional) -----------------------------------------
        self._apply_body_mask_current_grid(nx, ny, nz)

        # -- ID-field scatters (CPU, small cost) -------------------------------
        zone_winner_grid = self._scatter_zone_ids(
            pos_np,
            vortex_strength,
            zone_id_np,
            grid_min_np,
            particle_spacing,
            nx,
            ny,
            nz,
            mapping=node_mapping,
        )
        # The |Γ|-weighted ν_t average is inherited by regenerated particles.
        # particles inherit the pre-regen turbulent viscosity.
        eddy_viscosity_grid = self._scatter_scalar_weighted(
            pos_np,
            vortex_strength,
            eddy_viscosity_np,
            grid_min_np,
            particle_spacing,
            nx,
            ny,
            nz,
            mapping=node_mapping,
        )

        # -- Threshold pruning (CPU — read diffused grid back once) ------------
        grid_np = self._download_active_vec_grid(self._current_grid, nx, ny, nz)

        vortex_strength_magnitude = np.linalg.norm(grid_np, axis=-1)
        max_vortex_strength_magnitude = float(vortex_strength_magnitude.max())

        if max_vortex_strength_magnitude < 1e-30:
            Logging.warning(
                "component=GBD status=skipped reason=empty_scattered_grid particles_unchanged=true"
            )
            self._ping = True
            return None

        vortex_strength_total = float(vortex_strength_magnitude.sum(dtype=np.float64))
        threshold = self._select_diffusion_threshold(
            vortex_strength_magnitude,
            regen_threshold_mode,
            regen_threshold,
            max_vortex_strength_magnitude,
            vortex_strength_total,
        )
        ix, iy, iz = np.where(vortex_strength_magnitude >= threshold)
        if len(ix) == 0:
            Logging.warning(
                f"component=GBD status=skipped reason=no_nodes_above_threshold "
                f"threshold={threshold:.2e} particles_unchanged=true"
            )
            self._ping = True
            return None
        group_winner_grid = self._scatter_id_field(
            pos_np,
            vortex_strength,
            group_id_np,
            grid_min_np,
            particle_spacing,
            nx,
            ny,
            nz,
            default_id=0,
            propagate_to=np.where(vortex_strength_magnitude > 0.0),
            mapping=node_mapping,
        )
        threshold_retained = (
            float(vortex_strength_magnitude[ix, iy, iz].sum(dtype=np.float64))
            / vortex_strength_total
        )
        Logging.record(
            "gaussian blob diffusion regeneration",
            ("threshold", f"{threshold:.3e}"),
            ("nodes retained", f"{len(ix):,}"),
            ("strength fraction retained", f"{threshold_retained:.6f}"),
        )

        # -- Particle-count cap ------------------------------------------------
        cap = self._regeneration_cap(particles, N, max_nodes)
        if len(ix) > cap:
            survivor_abs = float(vortex_strength_magnitude[ix, iy, iz].sum(dtype=np.float64))
            ix, iy, iz, threshold, old_count = self._cap_surviving_nodes(
                vortex_strength_magnitude,
                ix,
                iy,
                iz,
                cap,
            )
            retained = (
                float(vortex_strength_magnitude[ix, iy, iz].sum(dtype=np.float64)) / survivor_abs
            )
            retained_total = retained * threshold_retained
            Logging.record(
                "gaussian blob diffusion population cap",
                ("cap", f"{cap:,}"),
                ("nodes, before", f"{old_count:,}"),
                ("nodes, after", f"{len(ix):,}"),
                ("strength fraction, candidates", f"{retained:.6f}"),
                ("strength fraction, net", f"{retained_total:.6f}"),
                ("threshold", f"{threshold:.3e}"),
            )

        nonzero_node_count = int(np.count_nonzero(vortex_strength_magnitude > 0.0))
        support_augmented_node_count = 0
        preserve_group_recovery = False
        if self.conserve_pruned_moments and len(ix) < nonzero_node_count:
            retained_before_support = len(ix)
            (
                ix,
                iy,
                iz,
                support_augmented_node_count,
                preserve_group_recovery,
            ) = self._augment_moment_recovery_support(
                grid_np,
                vortex_strength_magnitude,
                ix,
                iy,
                iz,
                grid_min_np,
                particle_spacing,
                cap,
                labels=group_winner_grid,
            )
            if support_augmented_node_count:
                Logging.record(
                    "gaussian blob diffusion moment support",
                    ("nodes, threshold/cap result", f"{retained_before_support:,}"),
                    ("nodes, support added", f"{support_augmented_node_count:,}"),
                    ("nodes, final retained", f"{len(ix):,}"),
                    ("regeneration cap", f"{cap:,}"),
                )
        recovery_diagnostics = self._empty_gbd_moment_recovery()
        recovery_diagnostics.update(
            nonzero_node_count=nonzero_node_count,
            retained_node_count=int(len(ix)),
            pruned_node_count=int(nonzero_node_count - len(ix)),
            support_augmented_node_count=int(support_augmented_node_count),
        )
        self._last_gbd_moment_recovery = recovery_diagnostics
        if self.conserve_pruned_moments and len(ix) < nonzero_node_count:
            raw_retained = grid_np[ix, iy, iz].astype(np.float64)
            closure_diagnostics: dict[str, bool | int | float] = {}
            corrected_retained = self._redistribute_pruned_moments(
                grid_np,
                vortex_strength_magnitude,
                ix,
                iy,
                iz,
                grid_min_np,
                particle_spacing,
                labels=(group_winner_grid if preserve_group_recovery else None),
                diagnostics=closure_diagnostics,
            )
            correction_l1 = float(
                np.linalg.norm(corrected_retained.astype(np.float64) - raw_retained, axis=1).sum(
                    dtype=np.float64
                )
            )
            complete_net = grid_np.sum(axis=(0, 1, 2), dtype=np.float64)
            raw_net = raw_retained.sum(axis=0, dtype=np.float64)
            corrected_net = corrected_retained.sum(axis=0, dtype=np.float64)
            net_scale = vortex_strength_total + 1.0e-30
            Logging.record(
                "gaussian blob diffusion moment recovery",
                ("correction strength fraction", f"{correction_l1 / net_scale:.6e}"),
                (
                    "net residual, before",
                    f"{np.linalg.norm(raw_net - complete_net) / net_scale:.6e}",
                ),
                (
                    "net residual, after",
                    f"{np.linalg.norm(corrected_net - complete_net) / net_scale:.6e}",
                ),
            )
            grid_np[ix, iy, iz] = corrected_retained
            recovery_diagnostics.update(closure_diagnostics)
            recovery_diagnostics["applied"] = True
            self._last_gbd_moment_recovery = recovery_diagnostics

        result = self._build_diffusion_particle_arrays(
            ix,
            iy,
            iz,
            grid_np,
            grid_min_np,
            particle_spacing,
            kinematic_viscosity,
            time_step_size,
            particles,
            N,
            zone_winner_grid,
            group_winner_grid,
            eddy_viscosity_grid=eddy_viscosity_grid,
        )
        self._ping = True
        return result

    def gbd_diffusion(
        self,
        particles,
        time_step_size: float,
        particle_spacing: float,
        kinematic_viscosity: float,
        domain_padding: float = 3.0,
        regen_threshold: float = 0.01,
        regen_threshold_mode: str = "budget",
        effective_viscosity: np.ndarray | None = None,
        max_nodes: int | None = None,
    ) -> dict[str, np.ndarray] | None:
        """GBD (Cottet & Koumoutsakos 2000) diffusion step with particle regeneration.

        CIC scatter → explicit 7-point Laplacian → threshold pruning → spawn.

        When ``effective_viscosity`` (per-particle ν+ν_t) is given, the Laplacian uses a
        per-node coefficient so the SGS eddy viscosity acts; otherwise
        the molecular ν is used.  Regenerated particles carry the pre-regen ν_t
        forward.
        """
        return self._gbd_diffusion_impl(
            particles,
            time_step_size,
            particle_spacing,
            kinematic_viscosity,
            domain_padding,
            regen_threshold,
            regen_threshold_mode,
            effective_viscosity=effective_viscosity,
            max_nodes=max_nodes,
        )

    def _dvh_scatter_vortex_strength(
        self,
        pos_np: np.ndarray,
        vortex_strength: np.ndarray,
        grid_min_np: np.ndarray,
        particle_spacing: float,
        kinematic_viscosity: float,
        time_step_size: float,
        nx: int,
        ny: int,
        nz: int,
        rd_ratio: float = 4.0,
        effective_viscosity_np: np.ndarray | None = None,
        q_max: float = 4.0,
    ) -> None:
        """DVH heat-kernel scatter (Durante et al. 2024, Section 2.3, Eqs. 17-19).

        For each particle j at y_j with vortex strength α_j, spread its vortex strength
        to grid nodes x_i within the diffusive radius R_d = rd_ratio * particle_spacing using
        the exact Green's function of the heat equation:

            w_ij = exp(-|x_i - y_j|² / (4 kinematic_viscosity Δt))

        Shepard normalization (Eq. 18) enforces exact per-particle Γ conservation:

            α'_ij = α_j · w_ij / Σ_{i∈P_j} w_ij

        The vortex strength at each node is the sum over all contributing particles
        (Eq. 19):

            α_i = Σ_{j∈B_i} α'_ij

        Parameters
        ----------
        pos_np : (N, 3) float array   Particle position.
        vortex_strength : (N, 3) float array  Particle vortex strength [m³/s].
        grid_min_np : (3,) float      Grid origin (minimum corner position).
        particle_spacing : float                     Grid spacing [m].
        kinematic_viscosity : float                    Kinematic viscosity [m²/s].
        time_step_size : float                    Time step [s] (unused — diffusive width set by R_d and β).
        nx, ny, nz : int              Active grid extents.
        rd_ratio : float              R_d / particle_spacing compact-support radius ratio.
                                      Default 4.0 (optimal, Durante 2024 Sec. 4.2).
        effective_viscosity_np : (N,) float array or None
                                      Per-particle effective viscosity (e.g.
                                      kinematic_viscosity + eddy_viscosity from an LES model).  When given,
                                      each particle's Gaussian width is scaled
                                      by q_j = effective_viscosity_j/kinematic_viscosity — the exact split-step
                                      heat kernel for that particle's viscosity.
        q_max : float                 Cap on the width ratio q_j.  The compact
                                      support stays at R_d, so a wide Gaussian
                                      is truncated: its tail beyond R_d carries
                                      ~exp(−1/(β·q)) of the weight (≈4 % at
                                      q = 4, β = 0.077).  Shepard normalization
                                      keeps Γ conserved, but the *shape* error
                                      grows with q — hence the cap.
        """
        R_d = rd_ratio * particle_spacing
        # Durante 2024, Eq. 15: β = 4nu·Δt_d / R_d² ≈ 0.077.
        # The diffusive timestep Δt_d is NOT the advection step; it is derived
        # from β so that the Gaussian is calibrated to spread meaningfully
        # across all ~270 nodes within R_d.  Using the advection Δt_a here
        # would make 4nu·Δt_a << particle_spacing² → exp(-particle_spacing²/(4nu·Δt_a)) ≈ 0 → no diffusion.
        diffusion_variance = _DVH_BETA * R_d * R_d  # = β·R_d² (≡ 4ν·Δt_d)
        R_d_sq = R_d * R_d
        N = len(pos_np)

        # Always leave the grid in a fully-defined state (zeros where no
        # particle deposits), so callers can read it back without a prior fill.
        self._zero_grid_kernel(self._current_grid, nx, ny, nz)
        if N == 0:
            return

        # Per-particle Gaussian width β·R_d²·q_j.  q_j = ν_eff_j/ν scales the
        # heat-kernel width to that particle's effective viscosity (the
        # mechanism by which an LES sub-grid ν_t acts in DVH), clipped at q_max
        # so the compact support stays at R_d.
        widths = np.full(N, diffusion_variance, dtype=np.float64)
        if effective_viscosity_np is not None and kinematic_viscosity > 0.0:
            q = np.clip(
                np.asarray(effective_viscosity_np, dtype=np.float64) / kinematic_viscosity,
                1.0,
                q_max,
            )
            widths *= q
            n_clipped = int(
                np.count_nonzero(np.asarray(effective_viscosity_np) / kinematic_viscosity > q_max)
            )
            if n_clipped > 0:
                Logging.record(
                    "discrete vortex heat method width cap",
                    ("particles clipped", f"{n_clipped:,}"),
                    ("particles", f"{N:,}"),
                    ("q, max", f"{q_max:.1f}"),
                )

        # Numba-compiled heat-kernel scatter.  This is the exact f64 algorithm
        # of the former ``for j in range(N)`` Python loop (same formulas, same
        # accumulation order → bit-identical conservation), but JIT-compiled.
        # That serial Python loop dominated DVH cost (≈5 min at 49k particles);
        # the compiled loop runs in a fraction of a second.
        grid_out = np.zeros((nx, ny, nz, 3), dtype=np.float64)
        _dvh_scatter_numba(
            np.ascontiguousarray(pos_np, dtype=np.float64),
            np.ascontiguousarray(vortex_strength, dtype=np.float64),
            widths,
            grid_out,
            np.ascontiguousarray(grid_min_np, dtype=np.float64),
            float(particle_spacing),
            float(R_d),
            float(R_d_sq),
            int(nx),
            int(ny),
            int(nz),
        )

        # Upload result to the Taichi grid field (f32, like the rest of the
        # grid-diffusion pipeline).
        self._upload_active_vec_grid(self._current_grid, grid_out, nx, ny, nz)

    def _grid_based_diffusion_impl(
        self,
        particles,
        time_step_size: float,
        particle_spacing: float,
        kinematic_viscosity: float,
        domain_padding: float = 3.0,
        regen_threshold: float = 0.01,
        regen_threshold_mode: str = "budget",
        rd_ratio: float = 4.0,
        effective_viscosity: np.ndarray | None = None,
        max_nodes: int | None = None,
    ) -> dict[str, np.ndarray] | None:
        """DVH diffusion + particle regeneration (Durante et al. 2024).

        Algorithm
        ---------
        1. DVH scatter: each particle's vortex strength is spread to grid nodes
           within R_d = rd_ratio * particle_spacing using the exact heat-kernel Gaussian.
           Shepard normalization ensures exact per-particle Γ conservation.
        2. Threshold pruning: discard nodes whose |Γ| is below the threshold.
        3. Spawn new particles at surviving grid nodes.

        No finite-difference solve or CFL constraint is involved — diffusion
        is encoded directly in the Gaussian scatter weights.
        """
        N = particles.n_particles_total
        if N == 0 or time_step_size == 0.0:
            return None

        try:
            zone_id_np = particles.zone_id_cpu().copy()
        except (AttributeError, Exception):
            zone_id_np = None
        group_id_np = particles.group_id_cpu()[:N].copy()

        pos_np = particles.position_cpu()
        vortex_strength = particles.vortex_strength_cpu()

        # -- LES: per-particle ν_t to carry through regen (Bug B) -------------
        eddy_viscosity_np = particles.eddy_viscosity_cpu()

        # -- Grid setup --------------------------------------------------------
        if self._fixed_grid_min is not None and self._max_grid_dims is not None:
            grid_min_np, (nx, ny, nz) = self._lattice_aligned_bounds(
                pos_np, particle_spacing, domain_padding
            )
            nx, ny, nz = self._ensure_grid_capacity(nx, ny, nz)
        else:
            grid_min_np, (nx, ny, nz) = self._compute_grid_bounds(
                pos_np,
                particle_spacing,
                domain_padding,
                half_cell_offset=False,
            )
            nx, ny, nz = self._ensure_grid_capacity(nx, ny, nz)
        node_mapping = _nearest_node_mapping(
            pos_np, vortex_strength, grid_min_np, particle_spacing, nx, ny, nz
        )

        # -- DVH heat-kernel scatter (Durante 2024, Eqs. 17-19) ---------------
        # (the scatter zeroes the grid internally before depositing)
        self._dvh_scatter_vortex_strength(
            pos_np,
            vortex_strength,
            grid_min_np,
            particle_spacing,
            kinematic_viscosity,
            time_step_size,
            nx,
            ny,
            nz,
            rd_ratio,
            effective_viscosity_np=effective_viscosity,
        )
        self._prepare_body_mask_current_grid(grid_min_np, particle_spacing, nx, ny, nz)
        self._apply_body_mask_current_grid(nx, ny, nz)

        # -- ID-field scatters (nearest-node, |Γ|-weighted) -------------------
        zone_winner_grid = self._scatter_zone_ids(
            pos_np,
            vortex_strength,
            zone_id_np,
            grid_min_np,
            particle_spacing,
            nx,
            ny,
            nz,
            mapping=node_mapping,
        )
        # ν_t scatter (Bug B): |Γ|-weighted average onto the grid so regenerated
        # particles inherit the pre-regen turbulent viscosity.
        eddy_viscosity_grid = self._scatter_scalar_weighted(
            pos_np,
            vortex_strength,
            eddy_viscosity_np,
            grid_min_np,
            particle_spacing,
            nx,
            ny,
            nz,
            mapping=node_mapping,
        )

        # -- Threshold pruning -------------------------------------------------
        grid_np = self._download_active_vec_grid(self._current_grid, nx, ny, nz)
        vortex_strength_magnitude = np.linalg.norm(grid_np, axis=-1)
        max_vortex_strength_magnitude = float(vortex_strength_magnitude.max())

        if max_vortex_strength_magnitude < 1e-30:
            Logging.warning(
                "component=DVH status=skipped reason=empty_scattered_grid particles_unchanged=true"
            )
            return None

        vortex_strength_total = float(vortex_strength_magnitude.sum(dtype=np.float64))
        threshold = self._select_diffusion_threshold(
            vortex_strength_magnitude,
            regen_threshold_mode,
            regen_threshold,
            max_vortex_strength_magnitude,
            vortex_strength_total,
        )
        ix, iy, iz = np.where(vortex_strength_magnitude >= threshold)
        if len(ix) == 0:
            Logging.warning(
                f"component=DVH status=skipped reason=no_nodes_above_threshold "
                f"threshold={threshold:.2e} particles_unchanged=true"
            )
            return None
        group_winner_grid = self._scatter_id_field(
            pos_np,
            vortex_strength,
            group_id_np,
            grid_min_np,
            particle_spacing,
            nx,
            ny,
            nz,
            default_id=0,
            propagate_to=np.where(vortex_strength_magnitude > 0.0),
            mapping=node_mapping,
        )
        threshold_retained = (
            float(vortex_strength_magnitude[ix, iy, iz].sum(dtype=np.float64))
            / vortex_strength_total
        )
        Logging.record(
            "discrete vortex heat method regeneration",
            ("threshold", f"{threshold:.3e}"),
            ("nodes retained", f"{len(ix):,}"),
            ("strength fraction retained", f"{threshold_retained:.6f}"),
        )

        # -- Particle-count cap ------------------------------------------------
        cap = self._regeneration_cap(particles, N, max_nodes)
        if len(ix) > cap:
            survivor_abs = float(vortex_strength_magnitude[ix, iy, iz].sum(dtype=np.float64))
            ix, iy, iz, threshold, old_count = self._cap_surviving_nodes(
                vortex_strength_magnitude,
                ix,
                iy,
                iz,
                cap,
            )
            retained = (
                float(vortex_strength_magnitude[ix, iy, iz].sum(dtype=np.float64)) / survivor_abs
            )
            Logging.record(
                "discrete vortex heat method population cap",
                ("cap", f"{cap:,}"),
                ("nodes, before", f"{old_count:,}"),
                ("nodes, after", f"{len(ix):,}"),
                ("strength fraction, candidates", f"{retained:.6f}"),
                ("threshold", f"{threshold:.3e}"),
            )

        return self._build_diffusion_particle_arrays(
            ix,
            iy,
            iz,
            grid_np,
            grid_min_np,
            particle_spacing,
            kinematic_viscosity,
            time_step_size,
            particles,
            N,
            zone_winner_grid,
            group_winner_grid,
            eddy_viscosity_grid=eddy_viscosity_grid,
        )

    def grid_based_diffusion(
        self,
        particles,
        time_step_size: float,
        particle_spacing: float,
        kinematic_viscosity: float,
        domain_padding: float = 3.0,
        regen_threshold: float = 0.01,
        regen_threshold_mode: str = "budget",
        rd_ratio: float = 4.0,
        effective_viscosity: np.ndarray | None = None,
        max_nodes: int | None = None,
    ) -> dict[str, np.ndarray] | None:
        """DVH (Durante 2024) diffusion step with particle regeneration.

        Spreads each particle's vortex strength to nearby grid nodes via the exact
        Gaussian heat-kernel Green's function, then replaces all particles with
        surviving grid nodes.  No finite-difference solve or CFL constraint.

        With ``effective_viscosity`` (per-particle effective viscosity, e.g. kinematic_viscosity + eddy_viscosity from
        an LES model), each particle's heat-kernel width is scaled by
        effective_viscosity/kinematic_viscosity — the exact per-particle split-step Green's function.
        """
        return self._grid_based_diffusion_impl(
            particles,
            time_step_size,
            particle_spacing,
            kinematic_viscosity,
            domain_padding,
            regen_threshold,
            regen_threshold_mode,
            rd_ratio=rd_ratio,
            effective_viscosity=effective_viscosity,
            max_nodes=max_nodes,
        )

    # ---- Taichi Kernels ----

    @ti.kernel
    def _m4_scatter_gpu_kernel(
        self,
        position: ti.template(),
        vortex_strength: ti.template(),
        grid: ti.template(),
        gmin_x: ti.f32,
        gmin_y: ti.f32,
        gmin_z: ti.f32,
        particle_spacing: ti.f32,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
        start_particle: ti.i32,
        count: ti.i32,
    ):
        """GPU M4' remeshing scatter: each particle deposits to 4³=64 grid nodes.

        Mirrors ``_cic_scatter_vortex_strength`` but runs entirely on GPU — no CPU↔GPU
        data transfer for the scatter itself.  Atomic adds ensure correctness
        when multiple particles contribute to the same grid node.

        Accuracy: same O(particle_spacing⁴) remeshing error as the CPU M4' scatter.
        """
        for local_particle in range(count):
            p = start_particle + local_particle
            pos = position[p]
            particle_vortex_strength = vortex_strength[p]

            # Fractional grid coordinates
            fx = (pos[0] - gmin_x) / particle_spacing
            fy = (pos[1] - gmin_y) / particle_spacing
            fz = (pos[2] - gmin_z) / particle_spacing

            # Base index (floor)
            ix0 = int(ti.floor(fx))
            iy0 = int(ti.floor(fy))
            iz0 = int(ti.floor(fz))

            # M4' 4³ stencil: offsets -1, 0, 1, 2 relative to base index
            for di in ti.static(range(-1, 3)):
                rx = ti.abs(fx - ti.cast(ix0 + di, ti.f32))
                wx = _m4_prime_1d_ti(rx)
                for dj in ti.static(range(-1, 3)):
                    ry = ti.abs(fy - ti.cast(iy0 + dj, ti.f32))
                    wy = _m4_prime_1d_ti(ry)
                    for dk in ti.static(range(-1, 3)):
                        rz = ti.abs(fz - ti.cast(iz0 + dk, ti.f32))
                        wz = _m4_prime_1d_ti(rz)
                        w = wx * wy * wz

                        ii = ix0 + di
                        jj = iy0 + dj
                        kk = iz0 + dk

                        if 0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz:
                            ti.atomic_add(grid[ii, jj, kk][0], w * particle_vortex_strength[0])
                            ti.atomic_add(grid[ii, jj, kk][1], w * particle_vortex_strength[1])
                            ti.atomic_add(grid[ii, jj, kk][2], w * particle_vortex_strength[2])

    @ti.kernel
    def _laplacian_step_gpu_kernel(
        self,
        src: ti.template(),
        dst: ti.template(),
        body_mask: ti.template(),
        alpha: ti.f32,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ):
        """GPU 7-point explicit Laplacian diffusion step with Neumann BC.

        Computes dst = src + α·∇²src where α = kinematic_viscosity·dt/particle_spacing².  Boundary nodes use
        index clamping (Neumann / zero-gradient), which is equivalent to the
        ``np.pad(mode='edge')`` used in the CPU path.

        Only the active sub-particle_volume ``[0, nx) × [0, ny) × [0, nz)`` is updated;
        cells beyond (due to headroom allocation) are left at zero.
        """
        for i, j, k in ti.ndrange(nx, ny, nz):
            if body_mask[i, j, k] != 0:
                dst[i, j, k] = ti.Vector.zero(ti.f32, 3)
                continue
            # Neumann BC: clamp neighbour indices to the active domain
            im = ti.max(i - 1, 0)
            ip = ti.min(i + 1, nx - 1)
            jm = ti.max(j - 1, 0)
            jp = ti.min(j + 1, ny - 1)
            km = ti.max(k - 1, 0)
            kp = ti.min(k + 1, nz - 1)

            centre = src[i, j, k]
            xp = centre if body_mask[ip, j, k] != 0 else src[ip, j, k]
            xm = centre if body_mask[im, j, k] != 0 else src[im, j, k]
            yp = centre if body_mask[i, jp, k] != 0 else src[i, jp, k]
            ym = centre if body_mask[i, jm, k] != 0 else src[i, jm, k]
            zp = centre if body_mask[i, j, kp] != 0 else src[i, j, kp]
            zm = centre if body_mask[i, j, km] != 0 else src[i, j, km]
            laplacian = xp + xm + yp + ym + zp + zm - 6.0 * centre
            dst[i, j, k] = centre + alpha * laplacian

    @ti.kernel
    def _laplacian_step_variable_gpu_kernel(
        self,
        src: ti.template(),
        dst: ti.template(),
        effective_viscosity_grid: ti.template(),
        body_mask: ti.template(),
        time_step_size: ti.f32,
        particle_spacing: ti.f32,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ):
        """GPU 7-point explicit Laplacian with a per-node ν_eff.

        Computes dst = src + α_node·∇²src where
        α_node = ν_eff_grid[i,j,k]·dt/particle_spacing²  (ν + ν_t from the SGS model).

        The ∇ν·∇ω cross-term of the full ∇·(ν_eff∇ω) operator is neglected
        (standard approximation for explicit grid VPM+LES).  Neumann BC and
        active-sub-particle_volume behaviour match ``_laplacian_step_gpu_kernel``.
        """
        h_sq = particle_spacing * particle_spacing
        for i, j, k in ti.ndrange(nx, ny, nz):
            if body_mask[i, j, k] != 0:
                dst[i, j, k] = ti.Vector.zero(ti.f32, 3)
                continue
            im = ti.max(i - 1, 0)
            ip = ti.min(i + 1, nx - 1)
            jm = ti.max(j - 1, 0)
            jp = ti.min(j + 1, ny - 1)
            km = ti.max(k - 1, 0)
            kp = ti.min(k + 1, nz - 1)

            centre = src[i, j, k]
            xp = centre if body_mask[ip, j, k] != 0 else src[ip, j, k]
            xm = centre if body_mask[im, j, k] != 0 else src[im, j, k]
            yp = centre if body_mask[i, jp, k] != 0 else src[i, jp, k]
            ym = centre if body_mask[i, jm, k] != 0 else src[i, jm, k]
            zp = centre if body_mask[i, j, kp] != 0 else src[i, j, kp]
            zm = centre if body_mask[i, j, km] != 0 else src[i, j, km]
            laplacian = xp + xm + yp + ym + zp + zm - 6.0 * centre
            alpha_node = effective_viscosity_grid[i, j, k] * time_step_size / h_sq
            dst[i, j, k] = src[i, j, k] + alpha_node * laplacian

    @ti.kernel
    def _fill_box_body_mask_kernel(
        self,
        body_mask: ti.template(),
        gmin_x: ti.f32,
        gmin_y: ti.f32,
        gmin_z: ti.f32,
        particle_spacing: ti.f32,
        xmin: ti.f32,
        xmax: ti.f32,
        ymin: ti.f32,
        ymax: ti.f32,
        zmin: ti.f32,
        zmax: ti.f32,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ):
        """Mark open-interior box nodes as solid; surface nodes stay fluid."""
        for i, j, k in ti.ndrange(nx, ny, nz):
            x = gmin_x + ti.cast(i, ti.f32) * particle_spacing
            y = gmin_y + ti.cast(j, ti.f32) * particle_spacing
            z = gmin_z + ti.cast(k, ti.f32) * particle_spacing
            inside = xmin < x and x < xmax and ymin < y and y < ymax and zmin < z and z < zmax
            body_mask[i, j, k] = 1 if inside else 0

    @ti.kernel
    def _zero_grid_kernel(
        self,
        field: ti.template(),
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ):
        """Zero the active sub-particle_volume of a vec3 grid field."""
        for i, j, k in ti.ndrange(nx, ny, nz):
            field[i, j, k] = ti.Vector.zero(ti.f32, 3)

    @ti.kernel
    def _download_vec_chunk_kernel(
        self,
        src: ti.template(),
        dst: ti.types.ndarray(),
        offset: ti.i32,
        count: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ):
        """Copy active-box nodes [offset, offset+count) into a flat host chunk."""
        for t in range(count):
            idx = offset + t
            i = idx // (ny * nz)
            rem = idx - i * ny * nz
            j = rem // nz
            k = rem - j * nz
            v = src[i, j, k]
            dst[t, 0] = v[0]
            dst[t, 1] = v[1]
            dst[t, 2] = v[2]

    @ti.kernel
    def _upload_vec_chunk_kernel(
        self,
        dst: ti.template(),
        src: ti.types.ndarray(),
        offset: ti.i32,
        count: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ):
        """Write a flat host chunk into active-box nodes [offset, offset+count)."""
        for t in range(count):
            idx = offset + t
            i = idx // (ny * nz)
            rem = idx - i * ny * nz
            j = rem // nz
            k = rem - j * nz
            dst[i, j, k] = ti.Vector([src[t, 0], src[t, 1], src[t, 2]])

    @ti.kernel
    def _upload_scalar_chunk_kernel(
        self,
        dst: ti.template(),
        src: ti.types.ndarray(),
        offset: ti.i32,
        count: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ):
        """Write a flat host chunk into active-box nodes of a scalar grid."""
        for t in range(count):
            idx = offset + t
            i = idx // (ny * nz)
            rem = idx - i * ny * nz
            j = rem // nz
            k = rem - j * nz
            dst[i, j, k] = src[t]

    def _grid_transfer_buffer(self, family: str, field, direction: str) -> np.ndarray:
        """Return a fixed staging array unique to a grid field and direction."""
        if not hasattr(self, "_grid_vec_chunks"):
            self._grid_vec_chunks = {}
            self._grid_scalar_chunks = {}
        key = (direction, id(field))
        if family == "vector":
            buffers = self._grid_vec_chunks
            shape = (_GRID_TRANSFER_CHUNK, 3)
        elif family == "scalar":
            buffers = self._grid_scalar_chunks
            shape = (_GRID_TRANSFER_CHUNK,)
        else:
            raise ValueError(f"Unknown grid transfer buffer family {family!r}")
        if key not in buffers:
            buffers[key] = np.zeros(shape, dtype=np.float32)
        return buffers[key]

    def _download_active_vec_grid(self, src, nx: int, ny: int, nz: int) -> np.ndarray:
        """Return the active box of a vec3 grid as an (nx, ny, nz, 3) host array."""
        buf = self._grid_transfer_buffer("vector", src, "download")
        total = int(nx) * int(ny) * int(nz)
        out = np.empty((total, 3), dtype=np.float32)
        for offset in range(0, total, _GRID_TRANSFER_CHUNK):
            count = min(_GRID_TRANSFER_CHUNK, total - offset)
            self._download_vec_chunk_kernel(src, buf, offset, count, ny, nz)
            ti.sync()
            out[offset : offset + count] = buf[:count]
        return out.reshape(nx, ny, nz, 3)

    def _upload_active_vec_grid(self, dst, values: np.ndarray, nx: int, ny: int, nz: int) -> None:
        """Write an (nx, ny, nz, 3) host array into the active box of a vec3 grid."""
        buf = self._grid_transfer_buffer("vector", dst, "upload")
        flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1, 3)
        total = int(nx) * int(ny) * int(nz)
        for offset in range(0, total, _GRID_TRANSFER_CHUNK):
            count = min(_GRID_TRANSFER_CHUNK, total - offset)
            buf[:count] = flat[offset : offset + count]
            self._upload_vec_chunk_kernel(dst, buf, offset, count, ny, nz)
            ti.sync()

    def _upload_active_scalar_grid(
        self, dst, values: np.ndarray, nx: int, ny: int, nz: int
    ) -> None:
        """Write an (nx, ny, nz) host array into the active box of a scalar grid."""
        buf = self._grid_transfer_buffer("scalar", dst, "upload")
        flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
        total = int(nx) * int(ny) * int(nz)
        for offset in range(0, total, _GRID_TRANSFER_CHUNK):
            count = min(_GRID_TRANSFER_CHUNK, total - offset)
            buf[:count] = flat[offset : offset + count]
            self._upload_scalar_chunk_kernel(dst, buf, offset, count, ny, nz)
            ti.sync()

    @ti.kernel
    def _grid_norm_kernel(
        self,
        field: ti.template(),
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ) -> ti.f32:
        """GPU-parallel L2 norm² reduction over the active sub-particle_volume."""
        result = 0.0
        for i, j, k in ti.ndrange(nx, ny, nz):
            v = field[i, j, k]
            result += v[0] ** 2 + v[1] ** 2 + v[2] ** 2
        return result

    @ti.kernel
    def _apply_body_mask_kernel(
        self,
        grid: ti.template(),
        body_mask: ti.template(),
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ):
        """Zero vorticity in solid cells for active domain extents."""
        for i, j, k in ti.ndrange(nx, ny, nz):
            if body_mask[i, j, k] != 0:
                grid[i, j, k] = ti.Vector.zero(ti.f32, 3)
