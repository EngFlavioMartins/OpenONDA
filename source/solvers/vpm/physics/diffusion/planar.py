"""Conservative XY M4-prime remeshing and molecular grid diffusion.

Strength keeps its volume-integrated units, but particle volume is h² times
the represented span. There is no z stencil, z diffusion, or end loss.
"""

from __future__ import annotations

import numpy as np

from source._numba import cacheable_njit as njit


@njit(cache=True)
def _weight(q):
    """Return the dimensionless M4-prime weight with support |q|<2."""
    q = abs(q)
    if q < 1.0:
        return 1.0 - 2.5 * q * q + 1.5 * q * q * q
    if q < 2.0:
        return 0.5 * (1.0 - q) * (2.0 - q) ** 2
    return 0.0


@njit(cache=True)
def scatter_planar(position, strength, origin, h, nx, ny):
    """Scatter z strengths onto a padded XY lattice with a 4-by-4 M4-prime stencil.

    Parameters
    ----------
    position, strength : numpy.ndarray, shape (N, 3)
        Source centres in m and stored strengths Gamma in m³/s. Only XY
        coordinates and Gamma_z are used; the caller validates planar inputs.
    origin : array_like, length >=2
        Lower XY lattice node in m.
    h : float
        Positive lattice spacing in m.
    nx, ny : int
        Node counts, including at least two padding nodes around donors.

    Returns
    -------
    numpy.ndarray, shape (nx, ny, 3)
        New float64 array of deposited strengths in m³/s. The first two
        components are zero. Input arrays are read-only.

    Raises
    ------
    ValueError
        If any nonzero interpolation weight lies outside the padded lattice.
    """
    grid = np.zeros((nx, ny, 3), dtype=np.float64)
    for p in range(len(position)):
        x = (position[p, 0] - origin[0]) / h
        y = (position[p, 1] - origin[1]) / h
        ix, iy = int(np.floor(x)) - 1, int(np.floor(y)) - 1
        for a in range(4):
            for b in range(4):
                w = _weight(x - ix - a) * _weight(y - iy - b)
                if w != 0.0:
                    if ix + a < 0 or ix + a >= nx or iy + b < 0 or iy + b >= ny:
                        raise ValueError("Planar scatter stencil exceeds padded lattice")
                    grid[ix + a, iy + b, 2] += w * strength[p, 2]
    return grid


def diffuse_planar_grid(grid, nu, dt, h, solid=None):
    """Advance an XY strength grid with the molecular five-point Laplacian.

    Parameters
    ----------
    grid : array_like, shape (nx, ny, 3)
        Grid strengths in m³/s. The input is copied and remains unchanged.
    nu : float
        Nonnegative kinematic viscosity in m²/s.
    dt : float
        Nonnegative total diffusion interval in s.
    h : float
        Positive XY spacing in m.
    solid : numpy.ndarray of bool, shape (nx, ny), optional
        Absorbing solid-node mask. The FVM owns wall vorticity generation.

    Returns
    -------
    field : numpy.ndarray, shape (nx, ny, 3)
        New float64 array of diffused strengths in m³/s.
    count : int
        Forward-Euler substeps, at least one, with nu*dt/(count*h²)<=0.12.

    Raises
    ------
    ValueError
        If any scale is nonfinite or violates the stated sign constraint.

    Notes
    -----
    The padded exterior is zero and solid nodes are zeroed after each
    substep. There is no z stencil or z-end loss.
    """
    if nu < 0 or dt < 0 or h <= 0 or not np.isfinite([nu, dt, h]).all():
        raise ValueError("Invalid planar diffusion scales")
    count = max(1, int(np.ceil(nu * dt / (0.12 * h * h))))
    alpha = nu * dt / (count * h * h)
    field = np.asarray(grid, dtype=np.float64).copy()
    if solid is not None:
        field[solid] = 0
    for _ in range(count):
        padded = np.pad(field, ((1, 1), (1, 1), (0, 0)))
        field = field + alpha * (
            padded[2:, 1:-1] + padded[:-2, 1:-1] + padded[1:-1, 2:] + padded[1:-1, :-2] - 4 * field
        )
        if solid is not None:
            field[solid] = 0
    return field, count


def planar_gbd(
    particles, config, induction, *, dt, anchor, core_radius_ratio, max_particles, solid_at=None
):
    """Remesh, diffuse and recover planar particle moments on an XY grid.

    Parameters
    ----------
    particles : Particles
        Active particle state; host reads synchronize device fields.
    config : ViscousConfig
        GBD spacing (m), viscosity (m²/s), padding, pruning mode and capacity.
    induction : PlanarInduction
        Supplies represented span and source plane, both in m.
    dt : float
        Diffusion interval in s.
    anchor : array_like, shape (3,), or None
        Fixed lattice phase in m; None uses the origin in XY.
    core_radius_ratio : float
        Dimensionless output core radius divided by grid spacing.
    max_particles : int
        Maximum replacement population, also limited by config.gbd_max_nodes.
    solid_at : callable or None
        Maps world points (N,3) in m to a boolean solid mask.

    Returns
    -------
    replacement : dict[str, numpy.ndarray] or None
        Detached particle fields: position/core_radius in m, vortex_strength
        in m³/s, particle_volume=h²*span in m³, viscosity in m²/s and integer
        zone/group IDs. Floating fields use particle storage dtype. None
        means the input cloud is empty; particle replacement is caller-owned.
    substeps : int
        Number of explicit molecular-diffusion substeps.

    Raises
    ------
    ValueError
        If transverse source strength or diffusion scales violate the model.
    RuntimeError
        If pruning removes all support, capacity is exceeded, or retained
        support cannot recover circulation and first/second XY moments.

    Side Effects
    ------------
    Allocates host grids and updates induction.last_gbd_moment_recovery.
    Particle fields are read only; this function does not commit a new state.
    """
    count = particles.n_particles_total
    if count == 0:
        return None, 1
    position = np.asarray(particles.position_cpu()[:count], dtype=np.float64)
    strength = np.asarray(particles.vortex_strength_cpu()[:count], dtype=np.float64)
    if np.max(np.abs(strength[:, :2]), initial=0) > 1e-8 * max(1.0, np.max(np.abs(strength[:, 2]))):
        raise ValueError("Planar GBD cannot evolve transverse vortex strength")
    h = config.gbd_grid_spacing
    nu = config.kinematic_viscosity
    if nu is None:
        nu = float(particles.kinematic_viscosity_cpu()[:count].mean())
    nsub = max(1, int(np.ceil(nu * dt / (0.12 * h * h))))
    # A finite explicit stencil travels at most one node per substep. Padding
    # retains the full remeshing+diffusion support and its exact moments.
    padding = max(int(np.ceil(config.gbd_domain_padding)), 2 + nsub)
    anchor = np.zeros(3) if anchor is None else np.asarray(anchor)
    lower = anchor[:2] + (np.floor((position[:, :2].min(0) - anchor[:2]) / h) - padding) * h
    upper = anchor[:2] + (np.ceil((position[:, :2].max(0) - anchor[:2]) / h) + padding) * h
    shape = np.rint((upper - lower) / h).astype(int) + 1
    grid = scatter_planar(position, strength, lower, h, *shape)
    xy = np.stack(
        np.meshgrid(
            lower[0] + h * np.arange(shape[0]), lower[1] + h * np.arange(shape[1]), indexing="ij"
        ),
        -1,
    )
    points = np.column_stack((xy.reshape(-1, 2), np.full(int(np.prod(shape)), induction.plane_z)))
    solid = None if solid_at is None else np.asarray(solid_at(points)).reshape(tuple(shape))
    grid, nsub = diffuse_planar_grid(grid, nu, dt, h, solid)
    values = grid.reshape(-1, 3)
    magnitude = np.abs(values[:, 2])
    if config.gbd_threshold_mode == "absolute":
        threshold = config.gbd_threshold
    elif config.gbd_threshold_mode == "relative_max":
        threshold = config.gbd_threshold * magnitude.max(initial=0)
    else:
        order = np.argsort(magnitude)
        discarded = np.cumsum(magnitude[order]) <= config.gbd_threshold * magnitude.sum()
        threshold = magnitude[order[discarded][-1]] if np.any(discarded) else 0.0
    keep = magnitude > threshold
    cap = min(max_particles, config.gbd_max_nodes or max_particles)
    if keep.sum() > cap:
        raise RuntimeError(
            f"Planar GBD needs {keep.sum()} particles, exceeding capacity {cap}; increase capacity or demonstrate coarser-grid convergence"
        )
    if not np.any(keep):
        raise RuntimeError("Planar GBD threshold removed the entire wake")
    # Preserve signed circulation, impulse, and all quadratic moments after
    # pruning. Scaling positions and solving a small Gram system avoids a
    # dense N x N solve. Report inability instead of silently losing moments.
    centre = points[keep, :2].mean(0)
    scale = max(h, np.ptp(points[keep, :2], axis=0).max())
    q = (points[:, :2] - centre) / scale
    rows = np.array(
        [np.ones(len(q)), q[:, 0], q[:, 1], q[:, 0] ** 2, q[:, 0] * q[:, 1], q[:, 1] ** 2]
    )
    target = rows @ values[:, 2]
    basis = rows[:, keep]
    retained = values[keep].copy()
    correction = np.linalg.lstsq(basis @ basis.T, target - basis @ retained[:, 2], rcond=1e-12)[0]
    correction_values = basis.T @ correction
    retained[:, 2] += correction_values
    residual = np.linalg.norm(basis @ retained[:, 2] - target)
    if residual > 1e-9 * max(1.0, magnitude.sum()):
        raise RuntimeError("Planar GBD has insufficient retained support to preserve moments")
    from source.coupler.stable_renewal import vortex_invariants

    before = vortex_invariants(points, values)
    after = vortex_invariants(points[keep], retained)
    strength_scale = max(float(magnitude.sum()), np.finfo(float).tiny)
    length = max(h, float(np.max(np.linalg.norm(points, axis=1))))
    induction.last_gbd_moment_recovery = {
        "applied": True,
        "nonzero_node_count": int(np.count_nonzero(magnitude)),
        "retained_node_count": int(keep.sum()),
        "pruned_node_count": int(np.count_nonzero((magnitude > 0) & ~keep)),
        "support_augmented_node_count": 0,
        "correction_fraction": float(np.abs(correction_values).sum() / strength_scale),
        "normalized_vortex_strength_residual": float(
            np.linalg.norm(after.total_vortex_strength - before.total_vortex_strength)
            / strength_scale
        ),
        "normalized_linear_impulse_residual": float(
            np.linalg.norm(after.linear_impulse - before.linear_impulse) / (strength_scale * length)
        ),
        "normalized_angular_impulse_residual": float(
            np.linalg.norm(after.angular_impulse - before.angular_impulse)
            / (strength_scale * length * length)
        ),
    }
    n = int(keep.sum())
    dtype = particles.position_cpu().dtype
    return {
        "position": points[keep].astype(dtype),
        "vortex_strength": retained.astype(dtype),
        "core_radius": np.full(n, core_radius_ratio * h, dtype=dtype),
        "particle_volume": np.full(n, h * h * induction.planar_span, dtype=dtype),
        "kinematic_viscosity": np.full(n, nu, dtype=dtype),
        "zone_id": np.zeros(n, dtype=np.int32),
        "group_id": np.zeros(n, dtype=np.int32),
    }, nsub
