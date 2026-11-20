"""Regeneration of particles from a diffused grid, shared by DVH and GBD.

After a scheme has diffused the vortex strength on the lattice, the grid nodes whose
strength magnitude reaches a cloud-wide threshold become the new particles. This module
reads the old cloud's fields, maps them to nearest nodes, chooses the retained nodes,
and builds the new particle arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from openonda.core.slip_slab_images import physical_slab_share

from ...config.constants import MAX_N_PARTICLES

if TYPE_CHECKING:
    from openonda.core.log_style import StepLog

    from ...particles.container import Particles


@dataclass(frozen=True)
class NearestNodeMapping:
    """One nearest-grid-node classification shared by regenerated fields."""

    valid: np.ndarray
    linear_index: np.ndarray
    vortex_strength_weight: np.ndarray


@dataclass(frozen=True)
class DiffusionSources:
    """Host copies of the particle fields one grid diffusion step reads."""

    position: np.ndarray
    vortex_strength: np.ndarray
    zone_id: np.ndarray
    group_id: np.ndarray
    eddy_viscosity: np.ndarray


@dataclass(frozen=True)
class RegeneratedNodes:
    """The grid nodes that become the new particles, with the fields they inherit."""

    grid: np.ndarray
    magnitude: np.ndarray
    magnitude_total: float
    ix: np.ndarray
    iy: np.ndarray
    iz: np.ndarray
    zone_winner_grid: np.ndarray
    group_winner_grid: np.ndarray
    eddy_viscosity_grid: np.ndarray
    capacity: int


_REGENERATION_TITLES = {
    "GBD": "gaussian blob diffusion regeneration",
    "DVH": "discrete vortex heat method regeneration",
}


def nearest_node_mapping(
    pos_np: np.ndarray,
    vortex_strength: np.ndarray,
    grid_min_np: np.ndarray,
    particle_spacing: float,
    nx: int,
    ny: int,
    nz: int,
) -> NearestNodeMapping:
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
    return NearestNodeMapping(valid, linear_index, vortex_strength_weight)


def scatter_id_field(
    pos_np: np.ndarray,
    vortex_strength: np.ndarray,
    ids_np: np.ndarray,
    grid_min_np: np.ndarray,
    particle_spacing: float,
    nx: int,
    ny: int,
    nz: int,
    default_id: int = 0,
    propagate_to: tuple[np.ndarray, ...] | None = None,
    mapping: NearestNodeMapping | None = None,
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
        mapping = nearest_node_mapping(
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
            temp = np.bincount(lin[mask_val], weights=w[mask_val], minlength=nx * ny * nz)
            better = temp > weight_flat
            winner_flat[better] = id_val
            weight_flat[better] = temp[better]

    if propagate_to is not None and np.any(weight_flat > 0.0):
        from scipy.spatial import cKDTree  # pyrefly: ignore[missing-module-attribute]

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


def scatter_scalar_weighted(
    pos_np: np.ndarray,
    vortex_strength: np.ndarray,
    scalar_np: np.ndarray,
    grid_min_np: np.ndarray,
    particle_spacing: float,
    nx: int,
    ny: int,
    nz: int,
    mapping: NearestNodeMapping | None = None,
    fill_empty: bool = False,
    fill_radius: int | None = None,
    fill_where: np.ndarray | None = None,
) -> np.ndarray:
    """Scatter a per-particle scalar to grid nodes (|Γ|-weighted average).

    Each grid node receives the vortex strength-magnitude-weighted average of
    the scalar over particles whose nearest node it is — mirroring
    ``_scatter_id_field``'s nearest-node winner scheme, but averaging a
    scalar instead of taking the dominant ID.  Nodes with no contributing
    particle get 0.0 unless ``fill_empty`` is true, in which case the
    nearest populated node's value is propagated.

    ``fill_radius`` limits the nearest-node search to empty nodes within that
    many cells (Chebyshev distance) of a populated node, and ``fill_where``, a
    boolean grid, to the empty nodes it marks; every other node takes the
    smallest populated value. A caller sets one when no value outside that set
    is read: no vorticity reaches it during the step, so it never enters a flux
    or a regenerated particle, and the grid maximum is unchanged.

    Used to carry ν_t and ν_eff from the pre-regen
    particle cloud onto the diffusion grid so that regenerated particles
    and the variable-coefficient Laplacian inherit a per-node value.
    """
    out = np.zeros((nx, ny, nz), dtype=np.float32)
    if scalar_np is None:
        return out

    if mapping is None:
        mapping = nearest_node_mapping(
            pos_np, vortex_strength, grid_min_np, particle_spacing, nx, ny, nz
        )
    w = mapping.vortex_strength_weight
    lin = mapping.linear_index
    s = np.ascontiguousarray(scalar_np[mapping.valid], dtype=np.float64)

    # Uniform viscosity (including zero eddy viscosity) needs no nearest-
    # neighbour search over the empty grid. Preserve the zero-weight and
    # out-of-domain behavior of the general weighted scatter.
    populated = w > 0.0
    if len(s) and np.all(s == s[0]):
        if np.any(populated):
            if fill_empty:
                out.fill(s[0])
            else:
                out.ravel()[np.unique(lin[populated])] = s[0]
        return out

    weight_flat = np.bincount(lin, weights=w, minlength=nx * ny * nz)
    accum_flat = np.bincount(lin, weights=w * s, minlength=nx * ny * nz)

    nz_mask = weight_flat > 0.0
    out_flat = out.ravel()
    out_flat[nz_mask] = (accum_flat[nz_mask] / weight_flat[nz_mask]).astype(np.float32)
    if fill_empty and np.any(nz_mask) and np.any(~nz_mask):
        from scipy.spatial import cKDTree  # pyrefly: ignore[missing-module-attribute]

        populated_linear = np.flatnonzero(nz_mask)
        if fill_radius is None and fill_where is None:
            empty_linear = np.flatnonzero(~nz_mask)
        else:
            out_flat[~nz_mask] = out_flat[populated_linear].min()
            empty_linear = (
                np.flatnonzero(fill_where.ravel() & ~nz_mask)
                if fill_where is not None
                else _empty_nodes_near(nz_mask.reshape(nx, ny, nz), fill_radius)
            )
        populated = np.column_stack(np.unravel_index(populated_linear, (nx, ny, nz)))
        empty = np.column_stack(np.unravel_index(empty_linear, (nx, ny, nz)))
        _, nearest = cKDTree(populated, compact_nodes=False).query(empty)
        out_flat[empty_linear] = out_flat[populated_linear[nearest]]
    return out


def _empty_nodes_near(populated: np.ndarray, radius: int) -> np.ndarray:
    """Linear indices of the empty nodes within ``radius`` cells (Chebyshev) of a populated one."""
    from scipy.ndimage import maximum_filter

    shape = populated.shape
    occupied = np.nonzero(populated)
    lower = [max(int(index.min()) - radius, 0) for index in occupied]
    upper = [
        min(int(index.max()) + radius + 1, size)
        for index, size in zip(occupied, shape, strict=True)
    ]
    box = tuple(slice(low, high) for low, high in zip(lower, upper, strict=True))
    local = populated[box]
    near = maximum_filter(local.view(np.uint8), size=2 * radius + 1, mode="constant") > 0
    ix, iy, iz = np.nonzero(near & ~local)
    return np.ravel_multi_index((ix + lower[0], iy + lower[1], iz + lower[2]), shape)


def read_diffusion_sources(
    particles: Particles,
    scheme: str,
    slab_bounds: tuple[float, float] | None,
) -> DiffusionSources:
    """Copy the particle fields a diffusion step reads; reject a particle outside the slab."""
    count = particles.n_particles_total
    sources = DiffusionSources(
        position=particles.position_cpu(),
        vortex_strength=particles.vortex_strength_cpu(),
        zone_id=particles.zone_id_cpu().copy(),
        group_id=particles.group_id_cpu()[:count].copy(),
        eddy_viscosity=particles.eddy_viscosity_cpu(),
    )
    if slab_bounds is not None:
        z_min, z_max = slab_bounds
        outside = (sources.position[:, 2] < z_min - 1e-7) | (sources.position[:, 2] > z_max + 1e-7)
        if np.any(outside):
            raise RuntimeError(f"{scheme} received a physical particle outside the slip slab")
    return sources


def select_regenerated_nodes(
    logger: StepLog,
    scheme: str,
    particles: Particles,
    sources: DiffusionSources,
    grid_np: np.ndarray,
    grid_min_np: np.ndarray,
    particle_spacing: float,
    node_mapping: NearestNodeMapping,
    regen_threshold: float,
    regen_threshold_mode: str,
    slab_bounds: tuple[float, float] | None,
    fill_support_only: bool = False,
) -> RegeneratedNodes | None:
    """Choose the diffused grid nodes that become particles, or None when nothing is left.

    ``grid_np`` is the diffused strength grid just downloaded; with a slip slab it is
    reduced to its physical share here. Zone and group labels and the eddy viscosity
    of the old cloud are carried onto the grid. A node is kept when its strength
    magnitude reaches the cloud-wide threshold; the count must fit the particle capacity.
    ``fill_support_only`` searches eddy-viscosity values only for the nodes that hold
    diffused strength, the only nodes a regenerated particle can come from.
    """
    nx, ny, nz = grid_np.shape[:3]
    # Nodes no particle reaches belong to zone 3, the background zone.
    zone_winner_grid = scatter_id_field(
        sources.position,
        sources.vortex_strength,
        sources.zone_id,
        grid_min_np,
        particle_spacing,
        nx,
        ny,
        nz,
        default_id=3,
        mapping=node_mapping,
    )
    if slab_bounds is not None:
        # Keep the physical share only: image nodes beyond the planes are
        # dropped, and a node on a plane, counted twice, keeps one half.
        grid_np *= physical_slab_share(nz, float(grid_min_np[2]), particle_spacing, slab_bounds)[
            None, None, :, None
        ]
    vortex_strength_magnitude = np.linalg.norm(grid_np, axis=-1)
    max_vortex_strength_magnitude = float(vortex_strength_magnitude.max())
    # The |Γ|-weighted ν_t average is inherited by the regenerated particles.
    # Empty nodes take the nearest populated value so variable diffusion does not
    # create zero-viscosity islands.
    eddy_viscosity_grid = scatter_scalar_weighted(
        sources.position,
        sources.vortex_strength,
        sources.eddy_viscosity,
        grid_min_np,
        particle_spacing,
        nx,
        ny,
        nz,
        mapping=node_mapping,
        fill_empty=True,
        fill_where=vortex_strength_magnitude > 0.0 if fill_support_only else None,
    )

    if max_vortex_strength_magnitude < 1e-30:
        logger.warning(
            f"{scheme} skipped: the scattered grid is empty; particle state is unchanged"
        )
        return None

    vortex_strength_total = float(vortex_strength_magnitude.sum(dtype=np.float64))
    threshold = _select_diffusion_threshold(
        vortex_strength_magnitude,
        regen_threshold_mode,
        regen_threshold,
        max_vortex_strength_magnitude,
        vortex_strength_total,
    )
    ix, iy, iz = np.where(vortex_strength_magnitude >= threshold)
    if len(ix) == 0:
        logger.warning(
            f"{scheme} skipped: no grid nodes reach the strength threshold "
            f"{threshold:.2e} m^3/s; particle state is unchanged"
        )
        return None
    group_winner_grid = scatter_id_field(
        sources.position,
        sources.vortex_strength,
        sources.group_id,
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
        float(vortex_strength_magnitude[ix, iy, iz].sum(dtype=np.float64)) / vortex_strength_total
    )
    logger.section(
        _REGENERATION_TITLES[scheme],
        ("threshold", f"{threshold:.3e}"),
        ("nodes retained", f"{len(ix):,}"),
        ("strength fraction retained", f"{threshold_retained:.6f}"),
    )

    capacity = regeneration_capacity(particles)
    _require_regeneration_capacity(len(ix), capacity, scheme.lower())
    return RegeneratedNodes(
        grid=grid_np,
        magnitude=vortex_strength_magnitude,
        magnitude_total=vortex_strength_total,
        ix=ix,
        iy=iy,
        iz=iz,
        zone_winner_grid=zone_winner_grid,
        group_winner_grid=group_winner_grid,
        eddy_viscosity_grid=eddy_viscosity_grid,
        capacity=capacity,
    )


def _select_diffusion_threshold(
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


def regeneration_capacity(particles: Particles) -> int:
    """Return the declared global particle limit for regeneration."""
    capacity = int(getattr(particles, "capacity", 0) or MAX_N_PARTICLES)
    return max(capacity, 1)


def _require_regeneration_capacity(n_survivors: int, capacity: int, scheme: str) -> None:
    """Reject capacity overflow before discarding any threshold-retained nodes."""
    if n_survivors > capacity:
        raise RuntimeError(
            f"{scheme} regeneration needs {n_survivors} particles, exceeding "
            f"max_n_particles={capacity}; increase the declared particle capacity"
        )


def build_particle_arrays(
    ix: np.ndarray,
    iy: np.ndarray,
    iz: np.ndarray,
    grid_np: np.ndarray,
    grid_min_np: np.ndarray,
    particle_spacing: float,
    kinematic_viscosity: float,
    core_radius: float,
    zone_winner_grid: np.ndarray,
    group_winner_grid: np.ndarray,
    eddy_viscosity_grid: np.ndarray | None = None,
    slab_bounds: tuple[float, float] | None = None,
) -> dict:
    """Assemble the new-particle dict from the diffused grid (3-D)."""
    M = len(ix)
    # 3-D cell particle_volume: particle_spacing³
    vol = float(particle_spacing) ** 3
    r = float(core_radius)
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
    volumes = np.full(M, vol, dtype=np.float32)
    if slab_bounds is not None:
        # A node on a slip plane keeps half its cell inside the slab.
        share = physical_slab_share(
            int(np.max(iz, initial=0)) + 1, float(grid_min_np[2]), particle_spacing, slab_bounds
        )
        volumes *= share[iz].astype(np.float32)
    return {
        "position": new_pos,
        "vortex_strength": grid_np[ix, iy, iz].astype(np.float32),
        "velocity": np.zeros((M, 3), dtype=np.float32),
        "particle_volume": volumes,
        "core_radius": np.full(M, r, dtype=np.float32),
        "kinematic_viscosity": np.full(M, kinematic_viscosity, dtype=np.float32),
        "eddy_viscosity": eddy_viscosity,
        "zone_id": zone_winner_grid[ix, iy, iz].astype(np.int32),
        "group_id": group_winner_grid[ix, iy, iz].astype(np.int32),
    }
