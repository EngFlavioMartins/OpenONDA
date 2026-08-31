"""Deterministic, divergence-free initial wake perturbation for the cylinder.

The controlled-seed experiment (``OPENONDA_SEED_AMPLITUDE``) needs exactly the
same analytic perturbation applied to the initial FVM velocity field of both
the coupled hybrid and the fully meshed reference, so that any remaining onset
offset can be attributed to the coupling rather than to different forcing.

The perturbation is written from a 2-D streamfunction in the (x, y) plane,

    psi = eps * U_inf * D * exp(-((x - x_s)^2 + y^2) / a^2)
          * (1 - R^2 / r^2)^2 * g(z),

with

    u'_x = dpsi/dy,   u'_y = -dpsi/dx,   u'_z = 0,

which is divergence-free by construction.  The radial factor and its first
derivative vanish at the cylinder wall, preserving the no-slip initial state.
``g(z)`` is a smooth spanwise taper that is one through the central portion of
the cylinder and vanishes toward the domain edges, so the seed excites the
primary Karman mode without perturbing a spanwise boundary.

The caller normalises the sampled field so that ``max|u'| == eps * U_inf``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def streamfunction_perturbation(
    points: np.ndarray,
    *,
    epsilon: float,
    freestream_speed: float = 1.0,
    diameter: float = 1.0,
    streamfunction_centre_x: float = 0.75,
    gaussian_half_width: float = 0.4,
    z_taper_begin: float = 3.0,
    z_taper_end: float = 4.75,
    wall_radius: float | None = None,
) -> np.ndarray:
    """Return the raw (not normalised) perturbation velocity at ``points``.

    Args:
        points: Cell-centre coordinates ``(N, 3)``.
        epsilon: Dimensionless seed amplitude (multiplies ``freestream_speed * diameter``).
        freestream_speed: Freestream speed.
        diameter: Cylinder diameter (the streamfunction length scale).
        streamfunction_centre_x: Streamfunction centre ``x/D`` (downstream of the cylinder).
        gaussian_half_width: Gaussian half-width relative to the cylinder diameter.
        z_taper_begin: ``|z|`` at which the spanwise taper starts (D units).
        z_taper_end: ``|z|`` at which the spanwise taper reaches zero (D units).
        wall_radius: Cylinder radius.  When supplied, the streamfunction and
            its first derivative vanish at the no-slip wall.

    Returns:
        Array ``(N, 3)`` with ``u'_z = 0``.  Not normalised: the peak value is
        approximately ``epsilon * freestream_speed`` only after :func:`normalise_seed`.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    taper_weight = np.ones_like(z)
    z_abs = np.abs(z)
    taper = (z_abs > z_taper_begin) & (z_abs < z_taper_end)
    taper_weight[taper] = 0.5 * (
        1.0 + np.cos(np.pi * (z_abs[taper] - z_taper_begin) / (z_taper_end - z_taper_begin))
    )
    taper_weight[z_abs >= z_taper_end] = 0.0

    radial_distance_squared = (x - streamfunction_centre_x) ** 2 + y**2
    envelope = np.exp(-radial_distance_squared / gaussian_half_width**2) * taper_weight
    envelope_x = -2.0 * (x - streamfunction_centre_x) / gaussian_half_width**2 * envelope
    envelope_y = -2.0 * y / gaussian_half_width**2 * envelope

    wall_shape = np.ones_like(x)
    wall_shape_x = np.zeros_like(x)
    wall_shape_y = np.zeros_like(y)
    if wall_radius is not None:
        radius_squared = x * x + y * y
        if np.any(radius_squared <= wall_radius * wall_radius):
            raise ValueError("Seed points must lie outside the cylinder wall")
        ratio = wall_radius * wall_radius / radius_squared
        one_minus_ratio = 1.0 - ratio
        wall_shape = one_minus_ratio**2
        factor = 4.0 * wall_radius * wall_radius * one_minus_ratio / radius_squared**2
        wall_shape_x = factor * x
        wall_shape_y = factor * y

    perturbation_velocity = np.empty_like(points)
    perturbation_velocity[:, 0] = envelope_y * wall_shape + envelope * wall_shape_y
    perturbation_velocity[:, 1] = -(envelope_x * wall_shape + envelope * wall_shape_x)
    perturbation_velocity[:, 2] = 0.0
    return epsilon * freestream_speed * diameter * perturbation_velocity


def normalise_seed(
    perturbation: np.ndarray, *, epsilon: float, freestream_speed: float = 1.0
) -> np.ndarray:
    """Scale a raw perturbation to ``epsilon * freestream_speed`` exactly."""
    perturb = np.asarray(perturbation, dtype=np.float64)
    peak = float(np.linalg.norm(perturb, axis=1).max())
    if not np.isfinite(peak) or peak <= 0.0:
        raise RuntimeError("seed perturbation field is identically zero or non-finite")
    return perturb * (epsilon * freestream_speed / peak)


def build_seed_velocity(
    cell_centre: np.ndarray,
    *,
    base_velocity,
    epsilon: float,
    freestream_speed: float = 1.0,
    diameter: float = 1.0,
) -> np.ndarray:
    """Return the seeded velocity with the requested perturbation magnitude.

    ``base_velocity`` is broadcast to ``(N, 3)``.  When ``epsilon <= 0`` the
    unperturbed base field is returned unchanged (no seed applied).
    """
    n = cell_centre.shape[0]
    base = np.broadcast_to(np.asarray(base_velocity, dtype=np.float64), (n, 3)).copy()
    if epsilon is None or float(epsilon) <= 0.0:
        return base
    raw = streamfunction_perturbation(
        cell_centre,
        epsilon=float(epsilon),
        freestream_speed=freestream_speed,
        diameter=diameter,
    )
    seed = normalise_seed(raw, epsilon=float(epsilon), freestream_speed=freestream_speed)
    return base + seed


def build_cylinder_initial_state(
    cell_centre: np.ndarray,
    *,
    freestream_velocity,
    diameter: float,
    seed_amplitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a divergence-free no-slip-compatible cylinder start.

    The streamfunction ``psi = U y (1 - a^2/r^2)^2`` gives zero radial and
    tangential velocity at ``r=a`` and approaches uniform flow smoothly.  It
    avoids the singular pressure impulse produced by initializing uniform flow
    through a no-slip cylinder.  A Bernoulli pressure estimate and the shared
    antisymmetric wake seed complete the deterministic initial state.
    """
    points = np.asarray(cell_centre, dtype=np.float64).reshape(-1, 3)
    freestream = np.asarray(freestream_velocity, dtype=np.float64)
    if freestream.shape != (3,) or not np.allclose(freestream[1:], 0.0):
        raise ValueError("Cylinder initial state requires an x-directed freestream")
    radius = 0.5 * float(diameter)
    x = points[:, 0]
    y = points[:, 1]
    radial_squared = x * x + y * y
    if np.any(radial_squared <= radius * radius):
        raise ValueError("Cylinder initial state received a cell centre inside the solid")
    radial = np.sqrt(radial_squared)
    ratio = radius * radius / radial_squared
    shape = (1.0 - ratio) ** 2
    shape_derivative = 4.0 * radius * radius * (1.0 - ratio) / radial**3

    velocity = np.zeros_like(points)
    velocity[:, 0] = freestream[0] * (shape + y * y * shape_derivative / radial)
    velocity[:, 1] = -freestream[0] * x * y * shape_derivative / radial
    if seed_amplitude > 0.0:
        raw_seed = streamfunction_perturbation(
            points,
            epsilon=seed_amplitude,
            freestream_speed=abs(float(freestream[0])),
            diameter=diameter,
            wall_radius=radius,
        )
        velocity += normalise_seed(
            raw_seed,
            epsilon=seed_amplitude,
            freestream_speed=abs(float(freestream[0])),
        )
    pressure = 0.5 * (float(np.dot(freestream, freestream)) - np.einsum(
        "ij,ij->i", velocity, velocity
    ))
    return velocity, pressure


def _affine_interpolate_2d(
    source_xy: np.ndarray,
    source_values: np.ndarray,
    target_xy: np.ndarray,
    *,
    k: int = 12,
) -> np.ndarray:
    """Interpolate a cell-centred 2-D field with bounded affine-exact weights."""
    from scipy.spatial import cKDTree

    source_xy = np.asarray(source_xy, dtype=np.float64)
    target_xy = np.asarray(target_xy, dtype=np.float64)
    values = np.asarray(source_values, dtype=np.float64)
    neighbours = min(int(k), len(source_xy))
    distance, indices = cKDTree(source_xy).query(target_xy, k=neighbours)
    if neighbours == 1:
        distance = distance[:, np.newaxis]
        indices = indices[:, np.newaxis]
    inverse_distance = 1.0 / (distance + 1.0e-12) ** 2
    inverse_distance /= inverse_distance.sum(axis=1, keepdims=True)

    offsets = source_xy[indices] - target_xy[:, np.newaxis, :]
    scale = np.max(np.abs(offsets), axis=1)
    scale[scale < 1.0e-12] = 1.0
    design = np.concatenate(
        (np.ones((*offsets.shape[:2], 1)), offsets / scale[:, np.newaxis, :]), axis=2
    )
    normal = np.einsum("nki,nk,nkj->nij", design, inverse_distance, design)
    intercept = np.linalg.pinv(normal, rcond=1.0e-12)[:, :, 0]
    weights = inverse_distance * np.einsum("nki,ni->nk", design, intercept)
    weight_sum = weights.sum(axis=1, keepdims=True)
    valid = (
        np.all(np.isfinite(weights), axis=1)
        & (np.abs(weight_sum[:, 0]) > 1.0e-12)
        & (np.sum(np.abs(weights), axis=1) < 10.0)
    )
    weights[valid] /= weight_sum[valid]
    weights[~valid] = inverse_distance[~valid]
    exact = np.any(distance <= 1.0e-12, axis=1)
    if np.any(exact):
        weights[exact] = 0.0
        nearest = np.argmin(distance[exact], axis=1)
        weights[np.flatnonzero(exact), nearest] = 1.0

    sampled = values[indices]
    if values.ndim == 1:
        return np.sum(weights * sampled, axis=1)
    return np.sum(weights[:, :, np.newaxis] * sampled, axis=1)


def build_transferred_initial_state(
    cell_centre: np.ndarray,
    source_field: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Prolong a saturated extruded reference field onto another x-y mesh.

    The source is averaged over its span before interpolation. This removes
    roundoff-scale three-dimensional contamination and gives every target slab
    exactly the same x-y state, as required for the nominally two-dimensional
    Re=150 cylinder benchmark.
    """
    import pyvista as pv

    source_path = Path(source_field).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    grid = pv.read(source_path)
    centres = np.asarray(grid.cell_centers().points, dtype=np.float64)
    velocity = np.asarray(grid.cell_data["velocity"], dtype=np.float64)
    pressure = np.asarray(grid.cell_data["kinematic_pressure"], dtype=np.float64)
    if "vtkGhostType" in grid.cell_data:
        owned = np.asarray(grid.cell_data["vtkGhostType"]) == 0
        centres = centres[owned]
        velocity = velocity[owned]
        pressure = pressure[owned]

    rounded_xy = np.round(centres[:, :2], decimals=12)
    source_xy, inverse, counts = np.unique(
        rounded_xy, axis=0, return_inverse=True, return_counts=True
    )
    mean_velocity = np.zeros((len(source_xy), 3), dtype=np.float64)
    mean_pressure = np.zeros(len(source_xy), dtype=np.float64)
    np.add.at(mean_velocity, inverse, velocity)
    np.add.at(mean_pressure, inverse, pressure)
    mean_velocity /= counts[:, np.newaxis]
    mean_pressure /= counts
    mean_velocity[:, 2] = 0.0

    targets = np.asarray(cell_centre, dtype=np.float64).reshape(-1, 3)
    target_xy = np.round(targets[:, :2], decimals=12)
    transferred_velocity = _affine_interpolate_2d(source_xy, mean_velocity, target_xy)
    transferred_pressure = _affine_interpolate_2d(source_xy, mean_pressure, target_xy)
    transferred_velocity[:, 2] = 0.0
    if not np.all(np.isfinite(transferred_velocity)) or not np.all(
        np.isfinite(transferred_pressure)
    ):
        raise RuntimeError("Transferred cylinder initial state contains non-finite values")
    return transferred_velocity, transferred_pressure
