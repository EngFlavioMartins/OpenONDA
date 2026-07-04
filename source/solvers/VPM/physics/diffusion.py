"""
Diffusion Physics Module for VPM Solver.
========================================
Handles viscous diffusion schemes for vortex particle methods.

This module provides:
- DVH (Grid-Based Diffusion / DVH) - Diffusion + particle regeneration, O(N)
- CSM (Core Spreading Method) - Local, O(N)
- RWM (Random Walk Method) - Stochastic, O(N)
- Volume update from velocity divergence

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import logging

import numpy as np
import taichi as ti
from numba import njit

from ..config.constants import MAX_PARTICLES
from .base import PhysicsBase

_logger = logging.getLogger("vpm")

# Radius assigned to freshly regenerated particles: σ = _REGEN_RADIUS_RATIO * h
_REGEN_RADIUS_RATIO = 2.5

# DVH truncation-error parameter β ≈ 0.077 (Durante et al. 2024, Eq. 14-15).
# Controls the Gaussian width: 4nu·Δt_d = β·R_d².
_DVH_BETA = 0.077

def _m4_prime_1d(r: np.ndarray) -> np.ndarray:
    """Vectorized 1D M4' (Monaghan 1985) interpolation kernel.

    Support: [-2, 2].  Satisfies partition of unity and reproduces
    quadratic polynomials exactly → O(h⁴) remeshing error (3rd-order
    convergence), compared to CIC's O(h²).

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
    circ: np.ndarray,
    widths: np.ndarray,
    grid_out: np.ndarray,
    gmin: np.ndarray,
    h: float,
    R_d: float,
    R_d_sq: float,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """DVH heat-kernel scatter (Durante et al. 2024, Eqs. 17-19), JIT-compiled.

    For each particle ``p`` at ``pos[p]`` with circulation ``circ[p]`` and
    Gaussian width ``widths[p]`` (= β·R_d²·q_p), spread its circulation to the
    grid nodes within the diffusive radius R_d using the exact heat-kernel
    weight ``exp(-r²/width_p)``.  Shepard normalisation (the per-particle
    ``/w_sum``) conserves each particle's total circulation exactly; the node
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
        i_lo = max(0, int(np.floor((px - R_d - gmin[0]) / h)))
        i_hi = min(nx - 1, int(np.ceil((px + R_d - gmin[0]) / h)))
        j_lo = max(0, int(np.floor((py - R_d - gmin[1]) / h)))
        j_hi = min(ny - 1, int(np.ceil((py + R_d - gmin[1]) / h)))
        k_lo = max(0, int(np.floor((pz - R_d - gmin[2]) / h)))
        k_hi = min(nz - 1, int(np.ceil((pz + R_d - gmin[2]) / h)))

        if i_lo > i_hi or j_lo > j_hi or k_lo > k_hi:
            continue

        # Pass 1 — Shepard denominator over support nodes within R_d.
        w_sum = 0.0
        for ii in range(i_lo, i_hi + 1):
            dx = (gmin[0] + ii * h) - px
            dx2 = dx * dx
            for jj in range(j_lo, j_hi + 1):
                dy = (gmin[1] + jj * h) - py
                dxy2 = dx2 + dy * dy
                for kk in range(k_lo, k_hi + 1):
                    dz = (gmin[2] + kk * h) - pz
                    r2 = dxy2 + dz * dz
                    if r2 <= R_d_sq:
                        w_sum += np.exp(-r2 / width)

        if w_sum < 1e-300:
            continue

        # Pass 2 — deposit Shepard-normalised circulation (exact Γ per particle).
        cx = circ[p, 0] / w_sum
        cy = circ[p, 1] / w_sum
        cz = circ[p, 2] / w_sum
        for ii in range(i_lo, i_hi + 1):
            dx = (gmin[0] + ii * h) - px
            dx2 = dx * dx
            for jj in range(j_lo, j_hi + 1):
                dy = (gmin[1] + jj * h) - py
                dxy2 = dx2 + dy * dy
                for kk in range(k_lo, k_hi + 1):
                    dz = (gmin[2] + kk * h) - pz
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

    # Maximum GPU memory (bytes) for DVH grid pre-allocation.  If the
    # full VPM domain grid exceeds this, the grid starts small and grows.
    _MAX_PREALLOC_BYTES: int = 1 << 30  # 1 GB

    # Conservative prune: redistribute the circulation of pruned (sub-threshold
    # / count-capped) grid nodes onto the survivors so the regeneration step
    # preserves the 0th moment (total circulation) and 1st moment (linear
    # impulse) exactly.
    conserve_pruned_moments: bool = True

    def _init_grid_diffusion(self):
        """Initialize grid-based diffusion state."""
        self._grid_realloc_count: int = 0

        # Core radius assigned to regenerated particles (σ = ratio·h).
        self.regen_radius_ratio: float = _REGEN_RADIUS_RATIO

        # Maximum grid dimensions from VPM domain (set by configure_max_grid_extent).
        self._max_grid_dims: tuple[int, int, int] | None = None

        # Fixed grid origin when the domain is pre-allocated.  When set, the
        # scatter grid uses this origin every step instead of re-computing it
        # from pos.min – padding.  This prevents the grid from drifting in the
        # direction of the minimum-position particle.
        self._fixed_grid_min: np.ndarray | None = None

        # Lazily allocated grid fields (never freed after first allocation).
        self._grid_a: ti.template() | None = None
        self._grid_b: ti.template() | None = None
        self._body_mask_grid: ti.template() | None = None  # 1=inside solid, 0=fluid
        # Per-node effective viscosity (ν+ν_t) for the variable-coefficient
        # GBD Laplacian (Bug A).  Lazily allocated alongside _body_mask_grid.
        self._nu_eff_grid: ti.template() | None = None
        self._grid_shape: tuple[int, int, int] | None = None

        # Ping-pong flag: True → _grid_a is the current/source field.
        self._ping: bool = True

        # Body-aware diffusion settings (optional; enabled when body STL is configured).
        self._body_mask_active: bool = False

        # Maximum number of grid cells per spatial dimension.
        self._MAX_CELLS_PER_DIM: int = 2000

        # Vulkan/Taichi 1.7.x does not release replaced grid fields until
        # ti.reset().  When enabled, DVH/GBD must use a fixed pre-allocated
        # domain grid and any later growth is treated as a configuration error.
        self._require_fixed_grid_allocation: bool = False

    def require_fixed_grid_allocation(self, enabled: bool = True) -> None:
        """Require grid-based diffusion to pre-allocate one fixed grid.

        This is used for Vulkan backends, where replacing Taichi fields during a
        long DVH/GBD run causes device memory to grow monotonically.
        """
        self._require_fixed_grid_allocation = bool(enabled)

    @property
    def _current_grid(self):
        """Field holding the current vorticity (source)."""
        return self._grid_a if self._ping else self._grid_b

    @property
    def _other_grid(self):
        """Field that will receive the updated vorticity (destination)."""
        return self._grid_b if self._ping else self._grid_a

    # ---- Internal: grid management ----

    def _compute_grid_bounds(
        self,
        pos: np.ndarray,
        h: float,
        padding: float,
        half_cell_offset: bool = True,
    ) -> tuple[np.ndarray, tuple[int, int, int]]:
        """Grid origin (float32) and integer dimensions from particle cloud.

        Robust against NaN positions and stray outliers:

        1. Non-finite positions are filtered before computing the bounding box.
        2. Each axis is capped at ``_MAX_CELLS_PER_DIM`` cells.  If the raw
           extent exceeds the cap, the domain is centred on the cloud and
           clamped, so a single stray particle cannot inflate the grid into
           an OOM allocation.

        Domain anchoring was removed (Feb 2026) because with the budget
        threshold the positive feedback loop it guarded against is no
        longer a concern, while its never-shrink property made it
        catastrophically sensitive to even one outlier.
        """
        # -- Filter non-finite positions ----------------------------------
        finite_mask = np.isfinite(pos).all(axis=1)
        if not finite_mask.all():
            n_bad = int((~finite_mask).sum())
            _logger.warning("DVH: filtered %d non-finite particle positions.", n_bad)
            pos = pos[finite_mask]
            if len(pos) == 0:
                # Everything was NaN — return minimal grid
                lo = np.zeros(3, dtype=np.float32)
                return lo, (5, 5, 5)

        margin = padding * h
        lo = pos.min(axis=0) - margin
        hi = pos.max(axis=0) + margin

        # -- Cap maximum grid extent per axis -----------------------------
        max_extent = self._MAX_CELLS_PER_DIM * h
        for d in range(3):
            span = hi[d] - lo[d]
            if span > max_extent:
                center = 0.5 * (lo[d] + hi[d])
                lo[d] = center - 0.5 * max_extent
                hi[d] = center + 0.5 * max_extent
                _logger.warning(
                    "DVH: axis %d extent %.1f m > cap %.1f m (%d cells); "
                    "clamping to cap.  Stray particles outside will be ignored.",
                    d,
                    span,
                    max_extent,
                    self._MAX_CELLS_PER_DIM,
                )

        nx = max(5, int(np.ceil((hi[0] - lo[0]) / h)) + 1)
        ny = max(5, int(np.ceil((hi[1] - lo[1]) / h)) + 1)
        nz = max(5, int(np.ceil((hi[2] - lo[2]) / h)) + 1)

        # Half-cell offset to avoid particle-on-node coincidence (M4 aliasing).
        if half_cell_offset:
            lo = lo - 0.5 * h

        return lo.astype(np.float32), (nx, ny, nz)

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
                "Vulkan DVH/GBD requires vpm_domain_bounds so the diffusion grid "
                "can be pre-allocated once. Refusing grow-on-demand grid allocation "
                "because Taichi 1.7.x retains replaced Vulkan fields until ti.reset()."
            )

        if self._grid_a is not None:
            alloc = self._grid_shape
            if nx <= alloc[0] and ny <= alloc[1] and nz <= alloc[2]:
                return nx, ny, nz

            if self._require_fixed_grid_allocation:
                raise RuntimeError(
                    "Vulkan DVH/GBD grid request exceeds the fixed pre-allocated "
                    f"grid {alloc}: requested {(nx, ny, nz)}. Increase "
                    "vpm_domain_bounds/padding or use CUDA/CPU for this run."
                )

            if self._grid_realloc_count >= self._MAX_GRID_REALLOCS:
                clamped = (min(nx, alloc[0]), min(ny, alloc[1]), min(nz, alloc[2]))
                _logger.warning(
                    "DVH: grid re-allocation budget exhausted (%d/%d). Clamping to %s (needed %s).",
                    self._grid_realloc_count,
                    self._MAX_GRID_REALLOCS,
                    clamped,
                    (nx, ny, nz),
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
            _logger.warning(
                "DVH: re-allocating grid (%d/%d): %s → (%d, %d, %d) (old fields leaked on GPU).",
                self._grid_realloc_count,
                self._MAX_GRID_REALLOCS,
                alloc,
                alloc_nx,
                alloc_ny,
                alloc_nz,
            )
            self._grid_a = ti.Vector.field(3, dtype=ti.f32, shape=(alloc_nx, alloc_ny, alloc_nz))
            self._grid_b = ti.Vector.field(3, dtype=ti.f32, shape=(alloc_nx, alloc_ny, alloc_nz))
            self._body_mask_grid = ti.field(dtype=ti.i32, shape=(alloc_nx, alloc_ny, alloc_nz))
            self._nu_eff_grid = ti.field(dtype=ti.f32, shape=(alloc_nx, alloc_ny, alloc_nz))
            self._grid_shape = (alloc_nx, alloc_ny, alloc_nz)
            self._ping = True
            return nx, ny, nz

        alloc_nx = int(nx * self._ALLOC_HEADROOM)
        alloc_ny = int(ny * self._ALLOC_HEADROOM)
        alloc_nz = int(nz * self._ALLOC_HEADROOM)

        _logger.debug(
            "DVH: allocating grid fields (%d×%d×%d) → (%d×%d×%d) with %.1f× headroom.",
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
        self._nu_eff_grid = ti.field(dtype=ti.f32, shape=(alloc_nx, alloc_ny, alloc_nz))
        self._grid_shape = (alloc_nx, alloc_ny, alloc_nz)
        self._ping = True
        return nx, ny, nz

    def configure_max_grid_extent(
        self,
        domain_bounds: list[float],
        h: float,
        padding: float = 3.0,
    ) -> None:
        """Set the maximum DVH grid dimensions from VPM domain bounds.

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
        h : float
            Particle / grid spacing [m].
        padding : float
            Margin in multiples of *h* added to each side.  Default 3.0.
        """
        import math

        margin = padding * h
        nx = max(5, math.ceil((domain_bounds[1] - domain_bounds[0] + 2 * margin) / h) + 1)
        ny = max(5, math.ceil((domain_bounds[3] - domain_bounds[2] + 2 * margin) / h) + 1)
        nz = max(5, math.ceil((domain_bounds[5] - domain_bounds[4] + 2 * margin) / h) + 1)
        self._max_grid_dims = (nx, ny, nz)

        # Store a fixed grid origin derived from the domain bounds.
        self._fixed_grid_min = np.array(
            [domain_bounds[0] - margin, domain_bounds[2] - margin, domain_bounds[4] - margin],
            dtype=np.float32,
        )

        # Memory estimate: 2 vector fields (3×f32), mask i32, nu_eff f32.
        bytes_per_node = 2 * 12 + 4 + 4  # = 32 bytes
        total_bytes = nx * ny * nz * bytes_per_node
        total_mb = total_bytes / (1 << 20)

        if total_bytes <= self._MAX_PREALLOC_BYTES and self._grid_a is None:
            _logger.info(
                "DVH: pre-allocating grid to VPM domain size (%d×%d×%d, %.0f MB).",
                nx,
                ny,
                nz,
                total_mb,
            )
            self._grid_a = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny, nz))
            self._grid_b = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny, nz))
            self._body_mask_grid = ti.field(dtype=ti.i32, shape=(nx, ny, nz))
            self._nu_eff_grid = ti.field(dtype=ti.f32, shape=(nx, ny, nz))
            self._grid_shape = (nx, ny, nz)
            self._ping = True
        else:
            if self._require_fixed_grid_allocation and self._grid_a is None:
                limit_mb = self._MAX_PREALLOC_BYTES / (1 << 20)
                raise MemoryError(
                    "Vulkan DVH/GBD requires a fixed pre-allocated grid, but "
                    f"the configured VPM domain grid is {nx}x{ny}x{nz} "
                    f"({total_mb:.0f} MB estimate), above the {limit_mb:.0f} MB "
                    "pre-allocation limit. Reduce vpm_domain_bounds/padding, "
                    "increase h, or use CUDA/CPU."
                )
            _logger.info(
                "DVH: VPM domain grid (%d×%d×%d) = %.0f MB — %s. "
                "Grid will grow on demand up to this cap.",
                nx,
                ny,
                nz,
                total_mb,
                "already allocated" if self._grid_a is not None else "exceeds pre-alloc budget",
            )

    def configure_body_mask(self, body_stl: str | None) -> None:
        """Configure optional body masking for DVH diffusion (not yet implemented).

        Parameters
        ----------
        body_stl : Optional[str]
            STL path to a closed body surface (reserved for future use).
        """
        self._body_mask_active = False

    # ---- DVH: DVH orchestration ----

    def _apply_body_mask_current_grid(self, nx: int, ny: int, nz: int) -> None:
        """Zero vorticity inside masked (solid) cells on the active grid."""
        if not self._body_mask_active or self._body_mask_grid is None:
            return
        self._apply_body_mask_kernel(self._current_grid, self._body_mask_grid, nx, ny, nz)

    def _scatter_id_field(
        self,
        pos_np: np.ndarray,
        circ_np: np.ndarray,
        ids_np,
        grid_min_np: np.ndarray,
        h: float,
        nx: int,
        ny: int,
        nz: int,
        default_id: int = 0,
    ) -> np.ndarray:
        """Scatter an integer ID field to grid nodes (|Γ|-weighted nearest-node).

        Each grid node receives the ID of the dominant (by |Γ|-weight) particle
        whose nearest node it is.  Nodes with no contributing particle get
        ``default_id``.

        Used for both ``zone_id`` (default 3) and ``group_id`` (default 0).
        """
        winner_grid = np.full((nx, ny, nz), default_id, dtype=np.int32)
        if ids_np is None:
            return winner_grid

        fx = (pos_np[:, 0] - grid_min_np[0]) / h
        fy = (pos_np[:, 1] - grid_min_np[1]) / h
        fz = (pos_np[:, 2] - grid_min_np[2]) / h
        ix_arr = np.round(fx).astype(np.intp)
        iy_arr = np.round(fy).astype(np.intp)
        iz_arr = np.round(fz).astype(np.intp)
        valid = (
            (ix_arr >= 0)
            & (ix_arr < nx)
            & (iy_arr >= 0)
            & (iy_arr < ny)
            & (iz_arr >= 0)
            & (iz_arr < nz)
        )
        w = np.abs(circ_np[valid]).sum(axis=1)
        lin = ix_arr[valid] * (ny * nz) + iy_arr[valid] * nz + iz_arr[valid]
        ids = ids_np[valid]
        weight_flat = np.zeros(nx * ny * nz, dtype=np.float64)
        winner_flat = winner_grid.ravel()
        for id_val in np.unique(ids):
            mask_val = ids == id_val
            temp = np.zeros(nx * ny * nz, dtype=np.float64)
            np.add.at(temp, lin[mask_val], w[mask_val])
            better = temp > weight_flat
            winner_flat[better] = id_val
            weight_flat[better] = temp[better]
        return winner_grid

    def _scatter_zone_ids(
        self,
        pos_np: np.ndarray,
        circ_np: np.ndarray,
        zone_id_np,
        grid_min_np: np.ndarray,
        h: float,
        nx: int,
        ny: int,
        nz: int,
    ) -> np.ndarray:
        """Scatter zone IDs — thin wrapper around _scatter_id_field (default_id=3)."""
        return self._scatter_id_field(
            pos_np, circ_np, zone_id_np, grid_min_np, h, nx, ny, nz, default_id=3
        )

    def _scatter_scalar_weighted(
        self,
        pos_np: np.ndarray,
        circ_np: np.ndarray,
        scalar_np: np.ndarray,
        grid_min_np: np.ndarray,
        h: float,
        nx: int,
        ny: int,
        nz: int,
    ) -> np.ndarray:
        """Scatter a per-particle scalar to grid nodes (|Γ|-weighted average).

        Each grid node receives the circulation-magnitude-weighted average of
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

        fx = (pos_np[:, 0] - grid_min_np[0]) / h
        fy = (pos_np[:, 1] - grid_min_np[1]) / h
        fz = (pos_np[:, 2] - grid_min_np[2]) / h
        ix_arr = np.round(fx).astype(np.intp)
        iy_arr = np.round(fy).astype(np.intp)
        iz_arr = np.round(fz).astype(np.intp)
        valid = (
            (ix_arr >= 0)
            & (ix_arr < nx)
            & (iy_arr >= 0)
            & (iy_arr < ny)
            & (iz_arr >= 0)
            & (iz_arr < nz)
        )
        w = np.abs(circ_np[valid]).sum(axis=1)
        lin = ix_arr[valid] * (ny * nz) + iy_arr[valid] * nz + iz_arr[valid]
        s = np.ascontiguousarray(scalar_np[valid], dtype=np.float64)

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
        circ_mag: np.ndarray,
        regen_threshold_mode: str,
        regen_threshold: float,
        max_circ: float,
        gamma_post_diffusion: float,
    ) -> float:
        """Determine the magnitude threshold for grid-node survival.

        Supports three modes:
        * ``'budget'``       — keep the top-(1-threshold) fraction of total |Γ| sum.
        * ``'relative_max'`` — keep nodes above threshold × global max|Γ|.
        * ``'absolute'``     — keep nodes above the absolute circulation value
                               ``regen_threshold`` (units [m³/s]).  This is the
                               preferred mode for controlling particle count when
                               the dynamic range of circulation spans many orders
                               of magnitude (e.g. coupled FVM-VPM simulations).
        """
        if regen_threshold_mode == "budget":
            circ_sum = gamma_post_diffusion
            if circ_sum > 1e-30:
                flat = circ_mag.ravel()
                order = np.argsort(-flat)
                cumsum = np.cumsum(flat[order])
                target = (1.0 - regen_threshold) * circ_sum
                cutoff = min(int(np.searchsorted(cumsum, target)), len(order) - 1)
                threshold = float(flat[order[cutoff]])
            else:
                threshold = 1e-10
        elif regen_threshold_mode == "relative_max":
            threshold = regen_threshold * max_circ
        elif regen_threshold_mode == "absolute":
            threshold = regen_threshold
        else:
            raise ValueError(
                f"Unknown regen_threshold_mode: {regen_threshold_mode!r}. "
                f"Must be 'budget', 'relative_max', or 'absolute'."
            )
        return max(threshold, 1e-10)

    @staticmethod
    def _cap_surviving_nodes(
        circ_mag: np.ndarray,
        ix: np.ndarray,
        iy: np.ndarray,
        iz: np.ndarray,
        cap: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
        """Keep exactly the strongest ``cap`` surviving grid nodes.

        Raising a magnitude threshold can still overshoot the requested cap
        when many nodes share the cutoff value.  Selecting by the already
        surviving linear indices makes the particle-count budget exact.
        """
        n_survivors = len(ix)
        if cap <= 0:
            raise ValueError("Diffusion regeneration cap must be positive.")
        if n_survivors <= cap:
            threshold = float(circ_mag[ix, iy, iz].min()) if n_survivors else 0.0
            return ix, iy, iz, threshold, n_survivors

        values = circ_mag[ix, iy, iz]
        keep_local = np.argpartition(-values, cap - 1)[:cap]
        keep_local = keep_local[np.argsort(-values[keep_local])]
        ix_keep = ix[keep_local]
        iy_keep = iy[keep_local]
        iz_keep = iz[keep_local]
        threshold = float(circ_mag[ix_keep, iy_keep, iz_keep].min())
        return ix_keep, iy_keep, iz_keep, threshold, n_survivors

    @staticmethod
    def _redistribute_pruned_moments(
        grid_np: np.ndarray,
        circ_mag: np.ndarray,
        ix: np.ndarray,
        iy: np.ndarray,
        iz: np.ndarray,
        grid_min_np: np.ndarray,
        h: float,
    ) -> np.ndarray:
        """Conservatively redistribute pruned circulation onto surviving nodes.

        The diffused grid conserves total circulation (Neumann Laplacian), but
        keeping only the survivor nodes ``(ix,iy,iz)`` discards the circulation of
        every other non-empty node.  This restores it on the survivors with the
        weighted affine correction

            δΓ_i = w_i · (a + b × r_i),   w_i = |Γ_i| / Σ|Γ|,  r_i = x_i − x̄,

        with ``a = ΔG`` and ``b = M⁻¹ (ΔL − x̄ × ΔG)`` where ΔG, ΔL are the total
        circulation and linear-impulse deficits and M = Σ w_i(|r_i|²I − r_ir_iᵀ).
        This conserves the 0th moment (Σ Γ) and 1st moment (Σ x×Γ) exactly; the
        2nd moment is intentionally not forced (not a remeshing invariant).

        Returns the corrected survivor circulations (n_keep, 3).  Falls back to
        0th-moment-only (b=0) if the survivor inertia tensor is ill-conditioned
        (few / coplanar survivors).
        """
        Gk = grid_np[ix, iy, iz].astype(np.float64)  # (Mk, 3) survivor circ
        Xk = np.stack(
            [grid_min_np[0] + ix * h, grid_min_np[1] + iy * h, grid_min_np[2] + iz * h],
            axis=1,
        ).astype(np.float64)
        wmag = np.linalg.norm(Gk, axis=1)
        wsum = float(wmag.sum())
        if wsum <= 0.0 or len(ix) < 4:
            return Gk.astype(np.float32)

        # Deficit = (all non-empty nodes) - (survivors), computed via full-grid
        # moments minus survivor moments
        nzi, nzj, nzk = np.where(circ_mag > 0.0)
        Gall = grid_np[nzi, nzj, nzk].astype(np.float64)
        Xall = np.stack(
            [grid_min_np[0] + nzi * h, grid_min_np[1] + nzj * h, grid_min_np[2] + nzk * h],
            axis=1,
        ).astype(np.float64)
        dG = Gall.sum(axis=0) - Gk.sum(axis=0)
        dL = np.cross(Xall, Gall).sum(axis=0) - np.cross(Xk, Gk).sum(axis=0)
        if not (np.any(np.abs(dG) > 0) or np.any(np.abs(dL) > 0)):
            return Gk.astype(np.float32)  # nothing pruned → exact no-op

        w = wmag / wsum
        xbar = (w[:, None] * Xk).sum(axis=0)
        r = Xk - xbar
        a = dG
        r2 = np.einsum("ij,ij->i", r, r)
        M = np.einsum("i,jk->jk", w * r2, np.eye(3)) - np.einsum("i,ij,ik->jk", w, r, r)
        rhs = dL - np.cross(xbar, a)
        try:
            # zeros for degenerate survivor geometry → 0th-moment correction only
            b = np.linalg.solve(M, rhs) if np.linalg.cond(M) < 1e12 else np.zeros(3)
        except np.linalg.LinAlgError:
            b = np.zeros(3)
        dGamma = w[:, None] * (a[None, :] + np.cross(np.broadcast_to(b, r.shape), r))
        return (Gk + dGamma).astype(np.float32)

    def _build_diffusion_particle_arrays(
        self,
        ix: np.ndarray,
        iy: np.ndarray,
        iz: np.ndarray,
        grid_np: np.ndarray,
        grid_min_np: np.ndarray,
        h: float,
        nu: float,
        dt: float,
        particles,
        N: int,
        zone_winner_grid: np.ndarray,
        group_winner_grid: np.ndarray,
        nu_t_grid: np.ndarray | None = None,
    ) -> dict:
        """Assemble the new-particle dict from the diffused grid (3-D)."""
        M = len(ix)
        # 3-D cell volume: h³
        vol = float(h) ** 3
        r = float(self.regen_radius_ratio) * float(h)
        new_pos = np.stack(
            [
                grid_min_np[0] + ix * h,
                grid_min_np[1] + iy * h,
                grid_min_np[2] + iz * h,
            ],
            axis=1,
        ).astype(np.float32)
        # Carry the pre-regen turbulent viscosity forward: reconstructs ν_eff = ν + ν_t.
        if nu_t_grid is not None:
            viscosity_turbulent = nu_t_grid[ix, iy, iz].astype(np.float32)
        else:
            viscosity_turbulent = np.zeros(M, dtype=np.float32)
        return {
            "position": new_pos,
            "circulation": grid_np[ix, iy, iz].astype(np.float32),
            "velocity": np.zeros((M, 3), dtype=np.float32),
            "volume": np.full(M, vol, dtype=np.float32),
            "radius": np.full(M, r, dtype=np.float32),
            "viscosity": np.full(M, nu, dtype=np.float32),
            "viscosity_turbulent": viscosity_turbulent,
            "zone_id": zone_winner_grid[ix, iy, iz].astype(np.int32),
            "group_id": group_winner_grid[ix, iy, iz].astype(np.int32),
        }

    def _gbd_diffusion_impl(
        self,
        particles,
        dt: float,
        h: float,
        nu: float,
        domain_padding: float = 3.0,
        regen_threshold: float = 0.01,
        regen_threshold_mode: str = "budget",
        nu_eff: np.ndarray | None = None,
        max_nodes: int | None = None,
    ) -> dict[str, np.ndarray] | None:
        """GBD diffusion + particle regeneration (Cottet & Koumoutsakos 2000).

        Algorithm
        ---------
        1. M4' scatter: particle circulation → grid  (GPU Taichi kernel).
        2. Explicit Laplacian: forward-Euler nu∇²ω    (GPU Taichi kernel).
           When ``nu_eff`` (per-particle ν+ν_t) is given, the Laplacian uses a
           per-node coefficient α_node = ν_eff·dt/h² (Bug A) instead of the
           scalar molecular α = ν·dt/h² — so the Smagorinsky SGS model acts in
           GBD runs just as it does in DVH/CS/RWM.
        3. Threshold pruning: discard weak grid nodes  (CPU NumPy — small).
        4. Spawn new particles at surviving nodes       (CPU NumPy).

        Steps 1-2 run entirely on GPU with no intermediate CPU↔GPU grid
        transfers.  Only positions are read to CPU for grid-bounds computation
        (cached), and the diffused grid is read back once for pruning.

        The CFL stability condition dt ≤ h²/(6·ν_eff_max) must be satisfied
        externally; a warning is logged if α_node exceeds 1/6.
        """
        N = particles.number_of_particles
        if N == 0 or dt == 0.0:
            return None

        try:
            zone_id_np = particles.zone_id_cpu().copy()
        except (AttributeError, Exception):
            zone_id_np = None
        group_id_np = particles.group_id_cpu()[:N].copy()

        pos_np = particles.position_cpu()
        circ_np = particles.circulation_cpu()

        # -- LES: per-particle ν_t to carry through regen  -------------
        # The scattered ν_t is inherited by regenerated particles so that ν_t
        # survives the rebuild and reaches backup (LES recomputes it next step
        # anyway, but carrying it keeps the backed-up field faithful).
        nu_t_np = particles.viscosity_turbulent_cpu()

        # -- Grid setup --------------------------------------------------------
        # Use a fixed grid origin when the domain was pre-configured, to avoid
        # the asymmetric flat-end artefact (see _fixed_grid_min docstring).
        if self._fixed_grid_min is not None and self._max_grid_dims is not None:
            grid_min_np = self._fixed_grid_min.copy()
            nx, ny, nz = self._ensure_grid_capacity(*self._max_grid_dims)
        else:
            grid_min_np, (nx, ny, nz) = self._compute_grid_bounds(
                pos_np,
                h,
                domain_padding,
                half_cell_offset=False,
            )
            nx, ny, nz = self._ensure_grid_capacity(nx, ny, nz)

        # -- M4' scatter (GPU) -------------------------------------------------
        self._current_grid.fill(0.0)
        gmin = grid_min_np.astype(float)
        self._m4_scatter_gpu_kernel(
            particles.position,
            particles.circulation,
            self._current_grid,
            gmin[0],
            gmin[1],
            gmin[2],
            float(h),
            nx,
            ny,
            nz,
            N,
        )

        # -- Explicit Laplacian diffusion (GPU) --------------------------------
        # Bug A: when ν_eff (ν+ν_t) is supplied, use a per-node coefficient
        # α_node = ν_eff·dt/h² so the SGS eddy viscosity acts in GBD runs.
        # Otherwise fall back to the scalar molecular α = ν·dt/h².
        self._other_grid.fill(0.0)
        if nu_eff is not None:
            nu_eff_np = np.ascontiguousarray(nu_eff[:N], dtype=np.float32)
            # Clip negatives (defensive: Smagorinsky guards already, but the
            # grid scatter can introduce tiny excursions via round-off).
            np.clip(nu_eff_np, 0.0, None, out=nu_eff_np)
            nu_eff_grid_np = self._scatter_scalar_weighted(
                pos_np, circ_np, nu_eff_np, grid_min_np, h, nx, ny, nz
            )
            alpha_max = float(nu_eff_grid_np.max()) * dt / (h * h) if nu_eff_grid_np.size else 0.0
            if alpha_max > 1.0 / 6.0:
                _logger.warning(
                    "[GBD] LES α_node_max=%.3f > 1/6 (ν_eff_max/ν=%.1f, dt/h²=%.3e) — "
                    "explicit Laplacian may go unstable; reduce dt.",
                    alpha_max,
                    float(nu_eff_grid_np.max()) / nu if nu > 0 else 0.0,
                    dt / (h * h),
                )
            # The field is allocated with headroom (_grid_shape ≥ (nx,ny,nz)); pad
            # the scattered array to the full shape before upload, exactly as the
            # regen scatter does (the Laplacian kernel only reads [:nx,:ny,:nz]).
            if nu_eff_grid_np.shape != self._grid_shape:
                _full = np.zeros(self._grid_shape, dtype=nu_eff_grid_np.dtype)
                _full[:nx, :ny, :nz] = nu_eff_grid_np
                nu_eff_grid_np = _full
            self._nu_eff_grid.from_numpy(nu_eff_grid_np)
            self._laplacian_step_variable_gpu_kernel(
                self._current_grid,
                self._other_grid,
                self._nu_eff_grid,
                float(dt),
                float(h),
                nx,
                ny,
                nz,
            )
        else:
            alpha = float(nu * dt / (h * h))
            self._laplacian_step_gpu_kernel(
                self._current_grid,
                self._other_grid,
                alpha,
                nx,
                ny,
                nz,
            )
        self._ping = not self._ping  # flip so _current_grid points to diffused field

        # -- Body mask (GPU, optional) -----------------------------------------
        self._apply_body_mask_current_grid(nx, ny, nz)

        # -- ID-field scatters (CPU, small cost) -------------------------------
        zone_winner_grid = self._scatter_zone_ids(
            pos_np, circ_np, zone_id_np, grid_min_np, h, nx, ny, nz
        )
        group_winner_grid = self._scatter_id_field(
            pos_np, circ_np, group_id_np, grid_min_np, h, nx, ny, nz, default_id=0
        )
        # ν_t scatter (Bug B): |Γ|-weighted average onto the grid so regenerated
        # particles inherit the pre-regen turbulent viscosity.
        nu_t_grid = self._scatter_scalar_weighted(
            pos_np, circ_np, nu_t_np, grid_min_np, h, nx, ny, nz
        )

        # -- Threshold pruning (CPU — read diffused grid back once) ------------
        grid_np = self._current_grid.to_numpy()[:nx, :ny, :nz, :]

        circ_mag = np.linalg.norm(grid_np, axis=-1)
        max_circ = float(circ_mag.max())

        if max_circ < 1e-30:
            _logger.warning("[GBD] Scattered grid is empty — keeping original particles.")
            return None

        gamma_total = float(circ_mag.sum())
        threshold = self._select_diffusion_threshold(
            circ_mag, regen_threshold_mode, regen_threshold, max_circ, gamma_total
        )
        ix, iy, iz = np.where(circ_mag >= threshold)
        if len(ix) == 0:
            _logger.warning("[GBD] No nodes above threshold %.2e — keeping originals.", threshold)
            return None

        # -- Particle-count cap ------------------------------------------------
        # Never exceed MAX_PARTICLES - 10k to avoid Taichi field resize/crash.
        # The top surviving nodes (by |Γ|) contain essentially all circulation;
        # the dropped tail nodes contribute little but can otherwise trigger
        # large Taichi external-array uploads during particle replacement.
        cap = min(max(int(3.0 * N), N + 50_000), MAX_PARTICLES - 10_000)
        if max_nodes is not None:
            cap = min(cap, int(max_nodes))
        if len(ix) > cap:
            ix, iy, iz, threshold, old_count = self._cap_surviving_nodes(
                circ_mag, ix, iy, iz, cap
            )
            _logger.info(
                "[GBD] Capped surviving nodes: %d → %d (threshold raised to %.2e).",
                old_count,
                len(ix),
                threshold,
            )

        _logger.info(
            "[GBD] %d particles → %d grid nodes (h=%.3e, threshold=%.2e).",
            N,
            len(ix),
            h,
            threshold,
        )

        # Conservative prune: restore the pruned nodes' circulation/impulse on
        # the survivors so total Γ and linear impulse are preserved (else the
        # threshold silently deletes circulation — non-physical wake decay).
        if self.conserve_pruned_moments:
            grid_np[ix, iy, iz] = self._redistribute_pruned_moments(
                grid_np, circ_mag, ix, iy, iz, grid_min_np, h
            )

        return self._build_diffusion_particle_arrays(
            ix,
            iy,
            iz,
            grid_np,
            grid_min_np,
            h,
            nu,
            dt,
            particles,
            N,
            zone_winner_grid,
            group_winner_grid,
            nu_t_grid=nu_t_grid,
        )

    def gbd_diffusion(
        self,
        particles,
        dt: float,
        h: float,
        nu: float,
        domain_padding: float = 3.0,
        regen_threshold: float = 0.01,
        regen_threshold_mode: str = "budget",
        nu_eff: np.ndarray | None = None,
        max_nodes: int | None = None,
    ) -> dict[str, np.ndarray] | None:
        """GBD (Cottet & Koumoutsakos 2000) diffusion step with particle regeneration.

        CIC scatter → explicit 7-point Laplacian → threshold pruning → spawn.

        When ``nu_eff`` (per-particle ν+ν_t) is given, the Laplacian uses a
        per-node coefficient so the SGS eddy viscosity acts (Bug A); otherwise
        the molecular ν is used.  Regenerated particles carry the pre-regen ν_t
        forward (Bug B).
        """
        return self._gbd_diffusion_impl(
            particles,
            dt,
            h,
            nu,
            domain_padding,
            regen_threshold,
            regen_threshold_mode,
            nu_eff=nu_eff,
            max_nodes=max_nodes,
        )

    def _dvh_scatter_circ(
        self,
        pos_np: np.ndarray,
        circ_np: np.ndarray,
        grid_min_np: np.ndarray,
        h: float,
        nu: float,
        dt: float,
        nx: int,
        ny: int,
        nz: int,
        rd_ratio: float = 4.0,
        nu_eff_np: np.ndarray | None = None,
        q_max: float = 4.0,
    ) -> None:
        """DVH heat-kernel scatter (Durante et al. 2024, Section 2.3, Eqs. 17-19).

        For each particle j at y_j with circulation α_j, spread its circulation
        to grid nodes x_i within the diffusive radius R_d = rd_ratio * h using
        the exact Green's function of the heat equation:

            w_ij = exp(-|x_i - y_j|² / (4 nu Δt))

        Shepard normalization (Eq. 18) enforces exact per-particle Γ conservation:

            α'_ij = α_j · w_ij / Σ_{i∈P_j} w_ij

        The circulation at each node is the sum over all contributing particles
        (Eq. 19):

            α_i = Σ_{j∈B_i} α'_ij

        Parameters
        ----------
        pos_np : (N, 3) float array   Particle positions.
        circ_np : (N, 3) float array  Particle circulations [m²/s = vol × ω].
        grid_min_np : (3,) float      Grid origin (minimum corner position).
        h : float                     Grid spacing [m].
        nu : float                    Kinematic viscosity [m²/s].
        dt : float                    Time step [s] (unused — diffusive width set by R_d and β).
        nx, ny, nz : int              Active grid extents.
        rd_ratio : float              R_d / h compact-support radius ratio.
                                      Default 4.0 (optimal, Durante 2024 Sec. 4.2).
        nu_eff_np : (N,) float array or None
                                      Per-particle effective viscosity (e.g.
                                      nu + nu_t from an LES model).  When given,
                                      each particle's Gaussian width is scaled
                                      by q_j = nu_eff_j/nu — the exact split-step
                                      heat kernel for that particle's viscosity.
        q_max : float                 Cap on the width ratio q_j.  The compact
                                      support stays at R_d, so a wide Gaussian
                                      is truncated: its tail beyond R_d carries
                                      ~exp(−1/(β·q)) of the weight (≈4 % at
                                      q = 4, β = 0.077).  Shepard normalization
                                      keeps Γ conserved, but the *shape* error
                                      grows with q — hence the cap.
        """
        R_d = rd_ratio * h
        # Durante 2024, Eq. 15: β = 4nu·Δt_d / R_d² ≈ 0.077.
        # The diffusive timestep Δt_d is NOT the advection step; it is derived
        # from β so that the Gaussian is calibrated to spread meaningfully
        # across all ~270 nodes within R_d.  Using the advection Δt_a here
        # would make 4nu·Δt_a << h² → exp(-h²/(4nu·Δt_a)) ≈ 0 → no diffusion.
        four_nu_dt = _DVH_BETA * R_d * R_d  # = β·R_d² (≡ 4nu·Δt_d)
        R_d_sq = R_d * R_d
        N = len(pos_np)

        # Always leave the grid in a fully-defined state (zeros where no
        # particle deposits), so callers can read it back without a prior fill.
        self._current_grid.fill(0.0)
        if N == 0:
            return

        # Per-particle Gaussian width β·R_d²·q_j.  q_j = ν_eff_j/ν scales the
        # heat-kernel width to that particle's effective viscosity (the
        # mechanism by which an LES sub-grid ν_t acts in DVH), clipped at q_max
        # so the compact support stays at R_d.
        widths = np.full(N, four_nu_dt, dtype=np.float64)
        if nu_eff_np is not None and nu > 0.0:
            q = np.clip(np.asarray(nu_eff_np, dtype=np.float64) / nu, 1.0, q_max)
            widths *= q
            n_clipped = int(np.count_nonzero(np.asarray(nu_eff_np) / nu > q_max))
            if n_clipped > 0:
                _logger.info(
                    "[DVH] %d/%d particles hit the nu_eff/nu width cap q_max=%.1f.",
                    n_clipped,
                    N,
                    q_max,
                )

        # Numba-compiled heat-kernel scatter.  This is the exact f64 algorithm
        # of the former ``for j in range(N)`` Python loop (same formulas, same
        # accumulation order → bit-identical conservation), but JIT-compiled.
        # That serial Python loop dominated DVH cost (≈5 min at 49k particles);
        # the compiled loop runs in a fraction of a second.
        grid_out = np.zeros((nx, ny, nz, 3), dtype=np.float64)
        _dvh_scatter_numba(
            np.ascontiguousarray(pos_np, dtype=np.float64),
            np.ascontiguousarray(circ_np, dtype=np.float64),
            widths,
            grid_out,
            np.ascontiguousarray(grid_min_np, dtype=np.float64),
            float(h),
            float(R_d),
            float(R_d_sq),
            int(nx),
            int(ny),
            int(nz),
        )

        # Upload result to the Taichi grid field (f32, like the rest of the
        # grid-diffusion pipeline).
        buf = np.zeros(self._grid_shape + (3,), dtype=np.float32)
        buf[:nx, :ny, :nz, :] = grid_out.astype(np.float32)
        self._current_grid.from_numpy(buf)

    def _grid_based_diffusion_impl(
        self,
        particles,
        dt: float,
        h: float,
        nu: float,
        domain_padding: float = 3.0,
        regen_threshold: float = 0.01,
        regen_threshold_mode: str = "budget",
        rd_ratio: float = 4.0,
        nu_eff: np.ndarray | None = None,
        max_nodes: int | None = None,
    ) -> dict[str, np.ndarray] | None:
        """DVH diffusion + particle regeneration (Durante et al. 2024).

        Algorithm
        ---------
        1. DVH scatter: each particle's circulation is spread to grid nodes
           within R_d = rd_ratio * h using the exact heat-kernel Gaussian.
           Shepard normalization ensures exact per-particle Γ conservation.
        2. Threshold pruning: discard nodes whose |Γ| is below the threshold.
        3. Spawn new particles at surviving grid nodes.

        No finite-difference solve or CFL constraint is involved — diffusion
        is encoded directly in the Gaussian scatter weights.
        """
        N = particles.number_of_particles
        if N == 0 or dt == 0.0:
            return None

        try:
            zone_id_np = particles.zone_id_cpu().copy()
        except (AttributeError, Exception):
            zone_id_np = None
        group_id_np = particles.group_id_cpu()[:N].copy()

        pos_np = particles.position_cpu()
        circ_np = particles.circulation_cpu()

        # -- LES: per-particle ν_t to carry through regen (Bug B) -------------
        nu_t_np = particles.viscosity_turbulent_cpu()

        # -- Grid setup --------------------------------------------------------
        # Use a fixed grid origin when the domain was pre-configured, to avoid
        # the asymmetric flat-end artefact (see _fixed_grid_min docstring).
        if self._fixed_grid_min is not None and self._max_grid_dims is not None:
            grid_min_np = self._fixed_grid_min.copy()
            nx, ny, nz = self._ensure_grid_capacity(*self._max_grid_dims)
        else:
            grid_min_np, (nx, ny, nz) = self._compute_grid_bounds(
                pos_np,
                h,
                domain_padding,
                half_cell_offset=False,
            )
            nx, ny, nz = self._ensure_grid_capacity(nx, ny, nz)

        # -- DVH heat-kernel scatter (Durante 2024, Eqs. 17-19) ---------------
        # (the scatter zeroes the grid internally before depositing)
        self._dvh_scatter_circ(
            pos_np, circ_np, grid_min_np, h, nu, dt, nx, ny, nz, rd_ratio, nu_eff_np=nu_eff
        )
        self._apply_body_mask_current_grid(nx, ny, nz)

        # -- ID-field scatters (nearest-node, |Γ|-weighted) -------------------
        zone_winner_grid = self._scatter_zone_ids(
            pos_np, circ_np, zone_id_np, grid_min_np, h, nx, ny, nz
        )
        group_winner_grid = self._scatter_id_field(
            pos_np, circ_np, group_id_np, grid_min_np, h, nx, ny, nz, default_id=0
        )
        # ν_t scatter (Bug B): |Γ|-weighted average onto the grid so regenerated
        # particles inherit the pre-regen turbulent viscosity.
        nu_t_grid = self._scatter_scalar_weighted(
            pos_np, circ_np, nu_t_np, grid_min_np, h, nx, ny, nz
        )

        # -- Threshold pruning -------------------------------------------------
        grid_np = self._current_grid.to_numpy()[:nx, :ny, :nz, :]
        circ_mag = np.linalg.norm(grid_np, axis=-1)
        max_circ = float(circ_mag.max())

        if max_circ < 1e-30:
            _logger.warning("[DVH] Scattered grid is empty — keeping original particles.")
            return None

        gamma_total = float(circ_mag.sum())
        threshold = self._select_diffusion_threshold(
            circ_mag, regen_threshold_mode, regen_threshold, max_circ, gamma_total
        )
        ix, iy, iz = np.where(circ_mag >= threshold)
        if len(ix) == 0:
            _logger.warning("[DVH] No nodes above threshold %.2e — keeping originals.", threshold)
            return None

        # -- Particle-count cap ------------------------------------------------
        # Never exceed MAX_PARTICLES - 10k to avoid Taichi field resize/crash.
        # An explicit max_nodes (e.g. ViscousConfig.dvh_max_nodes) additionally
        # bounds the budget-mode halo growth: keeping only the strongest
        # max_nodes nodes is a budget-by-count — the dropped tail carries a
        # negligible circulation fraction while N (and cost) stays bounded.
        cap = min(max(int(3.0 * N), N + 50_000), MAX_PARTICLES - 10_000)
        if max_nodes is not None:
            cap = min(cap, int(max_nodes))
        if len(ix) > cap:
            ix, iy, iz, threshold, old_count = self._cap_surviving_nodes(
                circ_mag, ix, iy, iz, cap
            )
            _logger.info(
                "[DVH] Capped surviving nodes: %d → %d (threshold raised to %.2e).",
                old_count,
                len(ix),
                threshold,
            )

        _logger.info(
            "[DVH] %d particles → %d grid nodes (R_d/h=%.1f, threshold=%.2e).",
            N,
            len(ix),
            rd_ratio,
            threshold,
        )

        # Conservative prune (see GBD path): restore pruned circulation/impulse.
        if self.conserve_pruned_moments:
            grid_np[ix, iy, iz] = self._redistribute_pruned_moments(
                grid_np, circ_mag, ix, iy, iz, grid_min_np, h
            )

        return self._build_diffusion_particle_arrays(
            ix,
            iy,
            iz,
            grid_np,
            grid_min_np,
            h,
            nu,
            dt,
            particles,
            N,
            zone_winner_grid,
            group_winner_grid,
            nu_t_grid=nu_t_grid,
        )

    def grid_based_diffusion(
        self,
        particles,
        dt: float,
        h: float,
        nu: float,
        domain_padding: float = 3.0,
        regen_threshold: float = 0.01,
        regen_threshold_mode: str = "budget",
        rd_ratio: float = 4.0,
        nu_eff: np.ndarray | None = None,
        max_nodes: int | None = None,
    ) -> dict[str, np.ndarray] | None:
        """DVH (Durante 2024) diffusion step with particle regeneration.

        Spreads each particle's circulation to nearby grid nodes via the exact
        Gaussian heat-kernel Green's function, then replaces all particles with
        surviving grid nodes.  No finite-difference solve or CFL constraint.

        With ``nu_eff`` (per-particle effective viscosity, e.g. nu + nu_t from
        an LES model), each particle's heat-kernel width is scaled by
        nu_eff/nu — the exact per-particle split-step Green's function.
        """
        return self._grid_based_diffusion_impl(
            particles,
            dt,
            h,
            nu,
            domain_padding,
            regen_threshold,
            regen_threshold_mode,
            rd_ratio,
            nu_eff=nu_eff,
            max_nodes=max_nodes,
        )

    # ---- Taichi Kernels ----

    @ti.kernel
    def _m4_scatter_gpu_kernel(
        self,
        positions: ti.template(),
        circulations: ti.template(),
        grid: ti.template(),
        gmin_x: ti.f32,
        gmin_y: ti.f32,
        gmin_z: ti.f32,
        h: ti.f32,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
        N: ti.i32,
    ):
        """GPU M4' remeshing scatter: each particle deposits to 4³=64 grid nodes.

        Mirrors ``_cic_scatter_circ`` but runs entirely on GPU — no CPU↔GPU
        data transfer for the scatter itself.  Atomic adds ensure correctness
        when multiple particles contribute to the same grid node.

        Accuracy: same O(h⁴) remeshing error as the CPU M4' scatter.
        """
        for p in range(N):
            pos = positions[p]
            circ = circulations[p]

            # Fractional grid coordinates
            fx = (pos[0] - gmin_x) / h
            fy = (pos[1] - gmin_y) / h
            fz = (pos[2] - gmin_z) / h

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
                            ti.atomic_add(grid[ii, jj, kk][0], w * circ[0])
                            ti.atomic_add(grid[ii, jj, kk][1], w * circ[1])
                            ti.atomic_add(grid[ii, jj, kk][2], w * circ[2])

    @ti.kernel
    def _laplacian_step_gpu_kernel(
        self,
        src: ti.template(),
        dst: ti.template(),
        alpha: ti.f32,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ):
        """GPU 7-point explicit Laplacian diffusion step with Neumann BC.

        Computes dst = src + α·∇²src where α = nu·dt/h².  Boundary nodes use
        index clamping (Neumann / zero-gradient), which is equivalent to the
        ``np.pad(mode='edge')`` used in the CPU path.

        Only the active sub-volume ``[0, nx) × [0, ny) × [0, nz)`` is updated;
        cells beyond (due to headroom allocation) are left at zero.
        """
        for i, j, k in ti.ndrange(nx, ny, nz):
            # Neumann BC: clamp neighbour indices to the active domain
            im = ti.max(i - 1, 0)
            ip = ti.min(i + 1, nx - 1)
            jm = ti.max(j - 1, 0)
            jp = ti.min(j + 1, ny - 1)
            km = ti.max(k - 1, 0)
            kp = ti.min(k + 1, nz - 1)

            laplacian = (
                src[ip, j, k]
                + src[im, j, k]
                + src[i, jp, k]
                + src[i, jm, k]
                + src[i, j, kp]
                + src[i, j, km]
                - 6.0 * src[i, j, k]
            )
            dst[i, j, k] = src[i, j, k] + alpha * laplacian

    @ti.kernel
    def _laplacian_step_variable_gpu_kernel(
        self,
        src: ti.template(),
        dst: ti.template(),
        nu_eff_grid: ti.template(),
        dt: ti.f32,
        h: ti.f32,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ):
        """GPU 7-point explicit Laplacian with a per-node ν_eff (Bug A).

        Computes dst = src + α_node·∇²src where
        α_node = ν_eff_grid[i,j,k]·dt/h²  (ν + ν_t from the SGS model).

        The ∇ν·∇ω cross-term of the full ∇·(ν_eff∇ω) operator is neglected
        (standard approximation for explicit grid VPM+LES).  Neumann BC and
        active-sub-volume behaviour match ``_laplacian_step_gpu_kernel``.
        """
        h_sq = h * h
        for i, j, k in ti.ndrange(nx, ny, nz):
            im = ti.max(i - 1, 0)
            ip = ti.min(i + 1, nx - 1)
            jm = ti.max(j - 1, 0)
            jp = ti.min(j + 1, ny - 1)
            km = ti.max(k - 1, 0)
            kp = ti.min(k + 1, nz - 1)

            laplacian = (
                src[ip, j, k]
                + src[im, j, k]
                + src[i, jp, k]
                + src[i, jm, k]
                + src[i, j, kp]
                + src[i, j, km]
                - 6.0 * src[i, j, k]
            )
            alpha_node = nu_eff_grid[i, j, k] * dt / h_sq
            dst[i, j, k] = src[i, j, k] + alpha_node * laplacian

    @ti.kernel
    def _grid_norm_kernel(
        self,
        field: ti.template(),
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ) -> ti.f32:
        """GPU-parallel L2 norm² reduction over the active sub-volume."""
        result = 0.0
        for i, j, k in field:
            if i < nx and j < ny and k < nz:
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
        for i, j, k in grid:
            if i < nx and j < ny and k < nz and body_mask[i, j, k] != 0:
                grid[i, j, k] = ti.Vector([0.0, 0.0, 0.0])

@ti.data_oriented
class DiffusionPhysics(PhysicsBase, _GridDiffusionMixin):
    """
    Diffusion physics handler for viscous effects.

    Implements multiple viscous diffusion schemes with different
    conservation properties and computational costs.
    """

    def __init__(
        self,
        particles_kernel: str = "GAUSSIAN",
        max_particles: int = MAX_PARTICLES,
        accumulator_dtype: ti.types = ti.f32,
    ):
        """Initialize diffusion physics module."""
        super().__init__(particles_kernel, max_particles, accumulator_dtype)
        self._init_grid_diffusion()

    # CORE SPREADING METHOD (CSM)

    def core_spreading_diffusion(self, particles, dt: float):
        """
        Apply viscous diffusion using Core Spreading Method.

        CSM models diffusion by expanding the particle core radius.  For the
        Gaussian kernel:
            sigma^2(t) = sigma^2(0) + 4*nu*t

        which is equivalent to convolution with a Gaussian diffusion kernel.

        Advantages:
        - O(N) computational cost
        - Simple and stable
        - No particle interactions needed

        Disadvantages:
        - Requires periodic remeshing to prevent excessive core overlap
        - Less accurate than direct-interaction methods for non-uniform distributions

        Args:
            particles: Particle container
            dt: Time step size [s]
        """
        N = len(particles)
        if N == 0 or dt <= 0.0:
            return

        self._resize_temp_fields(N)
        self._zero_temp_fields()

        # Update radii using the kernel-specific diffusivity constant.
        self.update_radius_csm_kernel(particles.radius, particles.viscosity_effective, dt, N)

    # RANDOM WALK METHOD (RWM)

    def random_walk_method_diffusion(self, particles, dt: float):
        """
        Apply viscous diffusion using Random Walk Method.

        RWM models diffusion by adding Gaussian random displacements:
            Δx = η * sqrt(2nu*dt)

        where η is a normally distributed random vector.

        This is a stochastic method that converges to the exact solution
        in the limit of many particles and small time steps.

        Advantages:
        - O(N) computational cost
        - Simple implementation
        - Works for any particle distribution

        Disadvantages:
        - Statistical noise (requires ensemble averaging)
        - May need very small time steps for accuracy

        Args:
            particles: Particle container
            dt: Time step size [s]
        """
        N = len(particles)
        if N == 0 or dt <= 0.0:
            return

        self._resize_temp_fields(N)
        self._zero_temp_fields()

        # Add random displacement: x_new = x + η*sqrt(2nu*dt)
        self.update_position_rwm_kernel(particles.position, particles.viscosity_effective, dt, N)

    # VOLUME UPDATE FROM DIVERGENCE

    def update_volumes(self, particles, dt: float):
        """
        Update particle volumes from velocity divergence.

        For compressible or variable-density flows:
            dV/dt = V * ∇·u

        The radius is updated consistently:
            r = (3V / 4π)^(1/3)

        Args:
            particles: Particle container
            dt: Time step size [s]
        """
        N = len(particles)
        if N == 0 or dt == 0.0:
            return

        self._resize_temp_fields(N)
        self._zero_temp_fields()

        # Update volume: V_new = V * (1 + dt * div(u))
        self.update_volume_divergence_kernel(
            particles.volume, particles.radius, particles.velocity_gradient, dt, N
        )
