"""Discrete vortex heat (DVH) diffusion with particle regeneration (Durante et al. 2024).

Each particle's vortex strength is spread to the lattice nodes within a diffusive radius
with the exact heat-kernel Gaussian, Shepard-normalised so that every particle keeps its
strength; the nodes that carry enough strength become the new particles. The scatter is
host Numba code in float64.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from openonda.core._numba import cacheable_njit as njit
from openonda.core.slip_slab_images import add_slab_images

from ...config.constants import DVH_BETA
from .regeneration import (
    build_particle_arrays,
    nearest_node_mapping,
    read_diffusion_sources,
    select_regenerated_nodes,
)

if TYPE_CHECKING:
    from ...particles.container import Particles
    from .lattice import DiffusionLattice


def uniform_dvh_viscosity(
    effective_viscosity: np.ndarray | None, kinematic_viscosity: float, count: int
) -> np.ndarray:
    """Return the per-particle effective viscosity of a DVH step, rejecting what DVH cannot use.

    The heat kernel needs one finite, non-negative viscosity for the whole cloud, to within
    float32 rounding; variable viscosity needs GBD. Without a field the molecular viscosity
    is used for every particle.
    """
    if effective_viscosity is None:
        viscosity = np.full(count, float(kinematic_viscosity), dtype=np.float64)
    else:
        viscosity = np.asarray(effective_viscosity, dtype=np.float64).reshape(-1)
        if viscosity.shape != (count,):
            raise ValueError("DVH effective viscosity must contain one value per particle")
    if not np.all(np.isfinite(viscosity)) or np.any(viscosity < 0.0):
        raise ValueError("DVH effective viscosity must be finite and non-negative")
    scale = max(float(np.max(np.abs(viscosity))), float(np.finfo(np.float32).tiny))
    if float(np.ptp(viscosity)) > 8.0 * np.finfo(np.float32).eps * scale:
        raise ValueError(
            "DVH requires spatially uniform effective viscosity; "
            "use GBD for variable-viscosity diffusion"
        )
    return viscosity


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
    Gaussian width ``widths[p]`` (= 4·nu_eff·Δt), spread its vortex strength to the
    grid nodes within the diffusive radius R_d using the exact heat-kernel
    weight ``exp(-r²/width_p)``.  Shepard normalisation (the per-particle
    ``/w_sum``) conserves each particle's total vortex strength exactly; the node
    contributions accumulate in ``grid_out`` (shape ``(nx, ny, nz, 3)``, f64).

    The loop is compiled with Numba, so a 49k-particle scatter runs in a fraction of a
    second.  ``grid_out`` must be zeroed by the caller.
    """
    N = pos.shape[0]
    for p in range(N):
        px = pos[p, 0]
        py = pos[p, 1]
        pz = pos[p, 2]
        width = widths[p]

        # Zero viscosity is the no-spread limit.  Avoid division by zero and
        # preserve the particle strength at its nearest active node.
        if not np.isfinite(width) or width <= 0.0:
            ii = int(np.rint((px - gmin[0]) / particle_spacing))
            jj = int(np.rint((py - gmin[1]) / particle_spacing))
            kk = int(np.rint((pz - gmin[2]) / particle_spacing))
            if 0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz:
                grid_out[ii, jj, kk, 0] += vortex_strength[p, 0]
                grid_out[ii, jj, kk, 1] += vortex_strength[p, 1]
                grid_out[ii, jj, kk, 2] += vortex_strength[p, 2]
            continue

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


def dvh_scatter_vortex_strength(
    lattice: DiffusionLattice,
    pos_np: np.ndarray,
    vortex_strength: np.ndarray,
    grid_min_np: np.ndarray,
    particle_spacing: float,
    viscosity_per_particle: np.ndarray,
    time_step_size: float,
    nx: int,
    ny: int,
    nz: int,
    rd_ratio: float = 4.0,
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
    viscosity_per_particle : (N,) float64 array  Uniform effective viscosity [m²/s], as
                                  returned by :func:`uniform_dvh_viscosity`.
    time_step_size : float                    Matched physical diffusion interval [s].
    nx, ny, nz : int              Active grid extents.
    rd_ratio : float              R_d / particle_spacing compact-support radius ratio.
                                  Default 4.0 (optimal, Durante 2024 Sec. 4.2).
    """
    N = len(pos_np)

    if N:
        resolved_width = DVH_BETA * (float(rd_ratio) * float(particle_spacing)) ** 2
        positive_viscosity = viscosity_per_particle > 0.0
        resolution_tolerance = (
            64.0
            * np.finfo(np.float32).eps
            * max(
                resolved_width,
                float(np.max(4.0 * viscosity_per_particle * float(time_step_size))),
            )
        )
        if np.any(
            positive_viscosity
            & (
                4.0 * viscosity_per_particle * float(time_step_size) + resolution_tolerance
                < resolved_width
            )
        ):
            raise ValueError(
                "DVH diffusion interval is unresolved on the configured lattice; "
                "accumulate accepted steps or increase time_step_size/viscosity"
            )

    # Always leave the grid in a fully-defined state (zeros where no
    # particle deposits), so callers can read it back without a prior fill.
    lattice.zero_grid_kernel(lattice.current_grid, nx, ny, nz)
    if N == 0:
        return

    widths = 4.0 * np.maximum(viscosity_per_particle, 0.0) * float(time_step_size)
    widths[~np.isfinite(widths)] = 0.0

    # Keep the configured support radius as a floor, but enlarge it when
    # the requested physical interval would otherwise truncate the heat
    # kernel.  R_d is a support and cost choice, not the variance; the
    # factor 3.6 keeps the truncated tail near beta ≈ 0.077.
    base_radius = float(rd_ratio * particle_spacing)
    max_heat_radius = 3.6 * float(np.sqrt(np.max(widths))) if N else 0.0
    R_d = max(base_radius, max_heat_radius)
    R_d_sq = R_d * R_d

    # Numba-compiled heat-kernel scatter in f64.
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
    slab = lattice.slip_slab_bounds
    if slab is not None:
        # The heat kernel and its Shepard sum are unchanged by a slab
        # reflection, so the physical deposits fix every image deposit.
        grid_out = add_slab_images(grid_out, float(grid_min_np[2]), particle_spacing, slab)

    # Upload result to the Taichi grid field (f32, like the rest of the
    # grid-diffusion sequence).
    lattice.upload_active_vec_grid(lattice.current_grid, grid_out, nx, ny, nz)


def dvh_diffusion(
    lattice: DiffusionLattice,
    particles: Particles,
    time_step_size: float,
    particle_spacing: float,
    kinematic_viscosity: float,
    domain_padding: float = 3.0,
    regen_threshold: float = 0.01,
    regen_threshold_mode: str = "budget",
    rd_ratio: float = 4.0,
    effective_viscosity: np.ndarray | None = None,
) -> dict[str, np.ndarray] | None:
    """DVH diffusion + particle regeneration (Durante et al. 2024).

    Spreads each particle's vortex strength to nearby grid nodes via the
    Gaussian heat-kernel Green's function, then replaces all particles with
    surviving grid nodes. No finite-difference solve is used.

    The operator is narrower than a general variable-coefficient diffusion
    operator: the effective viscosity must be spatially uniform, the
    accepted interval must be resolved by the configured lattice, and the
    incoming core radius must be finite, positive, and uniform. Smaller
    accepted intervals are accumulated by the solver before this remap is
    called. The incoming core radius is preserved on regenerated particles
    so remapping is not mistaken for physical diffusion.

    Algorithm
    ---------
    1. DVH scatter: each particle's vortex strength is spread to grid nodes
       within R_d = rd_ratio * particle_spacing using the exact heat-kernel Gaussian.
       Shepard normalization ensures exact per-particle Γ conservation.
    2. Weak-strength removal: discard nodes whose |Γ| is below the threshold.
    3. Spawn new particles at surviving grid nodes.

    No finite-difference solve is involved — diffusion is encoded directly
    in the Gaussian scatter weights. The finite-lattice scatter, support
    truncation, and optional weak-strength removal are a remap approximation, not an
    exact continuum heat solve. The remap requires a
    uniform effective viscosity and an accepted interval resolved by the
    configured lattice; callers must accumulate smaller intervals first.
    """
    N = particles.n_particles_total
    if N == 0 or time_step_size == 0.0:
        return None

    slab_bounds = lattice.slip_slab_bounds
    sources = read_diffusion_sources(particles, "DVH", slab_bounds)
    pos_np = sources.position
    vortex_strength = sources.vortex_strength
    if slab_bounds is not None:
        lattice.require_fixed_slab_extent("DVH")

    dvh_viscosity_np = uniform_dvh_viscosity(effective_viscosity, kinematic_viscosity, N)
    # A zero-width heat transfer is the identity operator. Return before
    # projecting onto the diffusion lattice; nearest-node remapping would
    # otherwise move off-lattice particles even though no diffusion is
    # physically active.
    if not np.any(dvh_viscosity_np > 0.0):
        return None
    support_radius = max(
        float(rd_ratio) * float(particle_spacing),
        3.6 * float(np.sqrt(np.max(4.0 * dvh_viscosity_np * float(time_step_size)))),
    )

    # -- Grid setup --------------------------------------------------------
    heat_padding = max(float(domain_padding), support_radius / float(particle_spacing) + 1.0)
    grid_min_np, (nx, ny, nz) = lattice.fit_active_box(
        pos_np, particle_spacing, heat_padding, include_halo_sources=True
    )
    grid_max_np = grid_min_np + np.asarray((nx - 1, ny - 1, nz - 1), dtype=np.float64) * float(
        particle_spacing
    )
    finite_position = np.isfinite(pos_np).all(axis=1)
    if np.any(finite_position):
        selected_position = pos_np[finite_position]
        if np.any(selected_position.min(axis=0) - support_radius < grid_min_np) or np.any(
            selected_position.max(axis=0) + support_radius > grid_max_np
        ):
            raise RuntimeError(
                "DVH heat support exceeds the allocated diffusion grid; "
                "increase domain padding/bounds or reduce the diffusion interval"
                + (
                    " (slip-slab images need the complete support beyond both planes)"
                    if slab_bounds is not None
                    else ""
                )
            )
    node_mapping = nearest_node_mapping(
        pos_np, vortex_strength, grid_min_np, particle_spacing, nx, ny, nz
    )

    # The heat-kernel transfer assumes a uniform incoming blob
    # basis. Preserve that basis through the finite-lattice remap instead
    # of replacing every source with the unrelated default 2.5h core.
    core_radius_np = particles.core_radius_cpu(use_cache=False)[:N].astype(np.float64)
    if not np.all(np.isfinite(core_radius_np)) or np.any(core_radius_np <= 0.0):
        raise ValueError("DVH requires finite, positive incoming core radii")
    core_radius_scale = max(float(np.max(np.abs(core_radius_np))), float(np.finfo(np.float32).tiny))
    core_radius_tolerance = 8.0 * np.finfo(np.float32).eps * core_radius_scale
    if float(np.ptp(core_radius_np)) > core_radius_tolerance:
        raise ValueError(
            "DVH currently requires a uniform incoming core radius; "
            "use GBD for mixed-core diffusion"
        )
    regenerated_core_radius = float(np.mean(core_radius_np))

    # -- DVH heat-kernel scatter (Durante 2024, Eqs. 17-19) ---------------
    # (the scatter zeroes the grid internally before depositing)
    dvh_scatter_vortex_strength(
        lattice,
        pos_np,
        vortex_strength,
        grid_min_np,
        particle_spacing,
        dvh_viscosity_np,
        time_step_size,
        nx,
        ny,
        nz,
        rd_ratio,
    )
    lattice.wall.prepare_mask(grid_min_np, particle_spacing, nx, ny, nz)
    lattice.apply_body_mask(nx, ny, nz)

    # -- Weak-strength removal ---------------------------------------------
    grid_np = lattice.download_active_vec_grid(lattice.current_grid, nx, ny, nz)
    regenerated = select_regenerated_nodes(
        lattice.logger,
        "DVH",
        particles,
        sources,
        grid_np,
        grid_min_np,
        particle_spacing,
        node_mapping,
        regen_threshold,
        regen_threshold_mode,
        slab_bounds,
    )
    if regenerated is None:
        return None

    return build_particle_arrays(
        regenerated.ix,
        regenerated.iy,
        regenerated.iz,
        regenerated.grid,
        grid_min_np,
        particle_spacing,
        kinematic_viscosity,
        regenerated_core_radius,
        regenerated.zone_winner_grid,
        regenerated.group_winner_grid,
        eddy_viscosity_grid=regenerated.eddy_viscosity_grid,
        slab_bounds=slab_bounds,
    )
