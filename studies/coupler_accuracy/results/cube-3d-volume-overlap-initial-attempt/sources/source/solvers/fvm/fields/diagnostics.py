#!/usr/bin/env python3
"""
Diagnostic Field Computations for OpenONDA FVM Solver.
=====================================================

Implements functions for computing:
- Courant Number (CFL)
- Vorticity
- y+ for wall boundaries
"""

import numpy as np

from source._numba import cacheable_njit as njit

from . import gradients


def compute_courant_number(velocity, volumetric_face_flux, time_step_size, mesh_data, geo_data):
    """Compute the cell Courant number from oriented face fluxes.

    Parameters
    ----------
    velocity : numpy.ndarray
        Cell/face velocity in m/s. Accepted for a common diagnostic signature;
        this implementation uses the supplied face flux directly.
    volumetric_face_flux : numpy.ndarray
        Face flux ``U · Sf``, shape ``(n_faces,)``, in m³/s. Interior entries
        are owner-to-neighbour oriented; boundary entries point outward.
    time_step_size : float
        Physical time step in seconds.
    mesh_data, geo_data : dict
        Native connectivity and geometry, including owner/neighbour indices and
        cell volumes in m³.

    Returns
    -------
    numpy.ndarray
        Non-negative cell values, shape ``(n_cells,)``. The convention is
        ``Co = 0.5 * dt * sum_faces(abs(phi)) / V``.

    Raises
    ------
    ValueError
        If any cell volume is non-finite or non-positive.
    """
    n_cells = mesh_data["n_cells"]
    n_interior = mesh_data["n_interior_faces"]
    owners = mesh_data["owners"]
    neighbours = mesh_data["neighbours"]
    cell_volume = np.asarray(geo_data["cell_volume"], dtype=np.float64)
    if not np.all(np.isfinite(cell_volume)) or np.any(cell_volume <= 0.0):
        raise ValueError("element cell_volume must be finite and positive")

    # Absolute flux
    absolute_volumetric_face_flux = np.abs(volumetric_face_flux)

    courant_number = np.zeros(n_cells)

    # Interior faces contribution
    np.add.at(courant_number, owners[:n_interior], absolute_volumetric_face_flux[:n_interior])
    np.add.at(courant_number, neighbours[:n_interior], absolute_volumetric_face_flux[:n_interior])

    # Boundary faces contribution
    np.add.at(courant_number, owners[n_interior:], absolute_volumetric_face_flux[n_interior:])

    # Final scaling
    courant_number = 0.5 * time_step_size * courant_number / cell_volume

    return courant_number


def compute_continuity_error(
    volumetric_face_flux: np.ndarray,
    mesh_data: dict,
    geo_data: dict,
) -> np.ndarray:
    """Compute the signed net volumetric flux for every finite-volume cell.

    For a discretely divergence-free (incompressible) solution this net face
    flux is ~0 in every cell.  Returned unnormalised so callers can form both
    the global mass imbalance Σ|residual| and the local divergence
    max|residual / V|.

    Parameters
    ----------
    volumetric_face_flux : numpy.ndarray, shape (n_faces,)
        Oriented face flux ``velocity · Sf`` in m³/s. Interior-face orientation
        is owner-to-neighbour; boundary normals point out of their owner cell.
    mesh_data : dict
        Mesh mapping containing ``n_cells``, ``n_interior_faces``, ``owners``,
        and ``neighbours``.
    geo_data : dict
        Accepted for diagnostic-call signature compatibility; not read.

    Returns
    -------
    numpy.ndarray, shape (n_cells,)
        Newly allocated signed net outward flux per cell in m³/s.
    """
    n_cells = mesh_data["n_cells"]
    n_interior = mesh_data["n_interior_faces"]
    owners = mesh_data["owners"]
    neighbours = mesh_data["neighbours"]

    div = np.zeros(n_cells)
    np.add.at(div, owners[:n_interior], volumetric_face_flux[:n_interior])
    np.add.at(div, neighbours[:n_interior], -volumetric_face_flux[:n_interior])
    np.add.at(div, owners[n_interior:], volumetric_face_flux[n_interior:])
    return div


def compute_kinetic_energy(
    velocity: np.ndarray,
    geo_data: dict,
    density: float | np.ndarray = 1.0,
) -> float:
    """Integrate ``0.5 * density * |velocity|²`` over interior cells.

    Parameters
    ----------
    velocity : numpy.ndarray, shape (n_cells[, + n_boundary], 3)
        Cell-centred velocity in m/s. Any appended boundary/ghost rows are
        ignored.
    geo_data : dict
        Geometry mapping whose ``cell_volume`` has shape ``(n_cells,)`` in m³.
    density : float or numpy.ndarray, default=1.0
        Positive density in kg/m³, scalar or shape ``(n_cells,)``.

    Returns
    -------
    float
        Volume-integrated kinetic energy in joules.

    Raises
    ------
    ValueError
        If density has the wrong shape or contains non-positive/non-finite data.
    """
    cell_volume = np.asarray(geo_data["cell_volume"], dtype=np.float64)
    velocity = np.asarray(velocity[: len(cell_volume)], dtype=np.float64)
    density = np.asarray(density, dtype=np.float64)
    if density.ndim == 0:
        density = np.full(len(cell_volume), float(density))
    if (
        density.shape != cell_volume.shape
        or np.any(density <= 0.0)
        or not np.all(np.isfinite(density))
    ):
        raise ValueError(f"density must be finite and positive with shape {cell_volume.shape}")
    return 0.5 * float(np.sum(density * cell_volume * np.sum(velocity * velocity, axis=1)))


def compute_enstrophy(
    velocity: np.ndarray,
    mesh_data: dict,
    geo_data: dict,
) -> float:
    """Integrate ``0.5 * |curl(velocity)|²`` over the FVM domain.

    Parameters
    ----------
    velocity : numpy.ndarray, shape (n_cells_with_ghosts, 3)
        Cell-centred and reconstructed boundary velocity in m/s.
    mesh_data, geo_data : dict
        Mesh connectivity and geometry accepted by the configured gradient
        reconstruction. ``cell_volume`` is in m³.

    Returns
    -------
    float
        Domain enstrophy integral in m³/s². The reconstructed vorticity has
        units 1/s and only interior-cell rows are integrated.
    """
    vorticity = compute_vorticity(velocity, mesh_data, geo_data)
    cell_volume = np.asarray(geo_data["cell_volume"], dtype=np.float64)
    return 0.5 * float(np.sum(cell_volume * np.sum(vorticity * vorticity, axis=1)))


def vorticity_from_gradient(velocity_gradient, n_cells: int | None = None):
    """Return curl of velocity from a reconstructed Jacobian.

    Parameters
    ----------
    velocity_gradient : numpy.ndarray
        Jacobian with shape ``(n, 3, 3)`` and convention
        ``gradient[cell, direction, component] = dU_component/dx_direction``.
    n_cells : int or None, default=None
        Number of leading entries to convert. ``None`` converts all rows.

    Returns
    -------
    numpy.ndarray
        Vorticity with shape ``(n_cells, 3)`` in 1/s.

    Raises
    ------
    ValueError
        If the input does not have a 3×3 trailing shape.
    """
    gradient = np.asarray(velocity_gradient, dtype=np.float64)
    if gradient.ndim != 3 or gradient.shape[1:] != (3, 3):
        raise ValueError("Velocity gradient must have shape (n, 3, 3)")
    n = gradient.shape[0] if n_cells is None else int(n_cells)
    vorticity = np.empty((n, 3), dtype=np.float64)
    vorticity[:, 0] = gradient[:n, 1, 2] - gradient[:n, 2, 1]
    vorticity[:, 1] = gradient[:n, 2, 0] - gradient[:n, 0, 2]
    vorticity[:, 2] = gradient[:n, 0, 1] - gradient[:n, 1, 0]
    return vorticity


@njit(cache=True)
def _enstrophy_from_gradient_kernel(gradient, cell_volume, n_cells):
    total = 0.0
    for cell in range(n_cells):
        vorticity_x = gradient[cell, 1, 2] - gradient[cell, 2, 1]
        vorticity_y = gradient[cell, 2, 0] - gradient[cell, 0, 2]
        vorticity_z = gradient[cell, 0, 1] - gradient[cell, 1, 0]
        total += cell_volume[cell] * (
            vorticity_x * vorticity_x + vorticity_y * vorticity_y + vorticity_z * vorticity_z
        )
    return 0.5 * total


def enstrophy_from_gradient(velocity_gradient, cell_volume, n_cells: int | None = None) -> float:
    """Integrate enstrophy directly from cell velocity gradients.

    Parameters
    ----------
    velocity_gradient : numpy.ndarray
        Jacobians, shape ``(n, 3, 3)``, in 1/s.
    cell_volume : numpy.ndarray
        Cell volumes, shape ``(n_cells,)``, in m³.
    n_cells : int or None, default=None
        Leading integration range; ``None`` uses every gradient row.

    Returns
    -------
    float
        ``0.5 * sum(V * |curl(U)|²)`` in m³/s².

    Raises
    ------
    ValueError
        If shapes or the requested range are incompatible.
    """
    gradient = np.asarray(velocity_gradient, dtype=np.float64)
    cell_volume = np.asarray(cell_volume, dtype=np.float64)
    if gradient.ndim != 3 or gradient.shape[1:] != (3, 3):
        raise ValueError("Velocity gradient must have shape (n, 3, 3)")
    n = gradient.shape[0] if n_cells is None else int(n_cells)
    if not 0 <= n <= gradient.shape[0] or cell_volume.shape[0] < n:
        raise ValueError("Enstrophy integration range exceeds gradient or volume storage")
    return float(_enstrophy_from_gradient_kernel(gradient, cell_volume, n))


def compute_vorticity(velocity, mesh_data, geo_data, *, gradient=None):
    """Reconstruct velocity gradients and return cell-centred vorticity.

    Parameters
    ----------
    velocity : numpy.ndarray
        Cell/ghost velocity values with shape ``(n_total, 3)`` in m/s.
    mesh_data, geo_data : dict
        Topology and geometry used by the configured Gauss/LSQ gradient scheme.
    gradient : numpy.ndarray or None, default=None
        Optional precomputed gradient with shape ``(n_total, 3, 3)`` in 1/s;
        supplying it avoids a second reconstruction.

    Returns
    -------
    numpy.ndarray
        Interior-cell vorticity, shape ``(n_cells, 3)``, in 1/s.
    """
    # velocity_gradient[i, j, k] is d(velocity_k)/dx_j. Solvers commonly need this same
    # expensive reconstruction for wall loads and VTK in one time state, so
    # accepting a supplied gradient prevents duplicate full-domain work.
    if gradient is None:
        _grad_fn = gradients._resolve_gradient_fn(geo_data)
        gradient = _grad_fn(velocity, mesh_data, geo_data)
    return vorticity_from_gradient(gradient, mesh_data["n_cells"])


def _normalize_patch_names(patch_names):
    """Normalize *patch_names* into a list of strings.

    Accepts ``None`` (returns ``None``), a comma-separated string, or an
    iterable of strings.

    Args:
        patch_names: Patch name(s) to normalise.  May be ``None``, a
            comma-separated ``str``, or an iterable of ``str``.

    Returns:
        list[str] | None: Normalized list of patch names, or ``None`` when
        the input is ``None``.
    """
    if patch_names is None:
        return None
    if isinstance(patch_names, str):
        return [
            kinematic_pressure.strip()
            for kinematic_pressure in patch_names.split(",")
            if kinematic_pressure.strip()
        ]
    return list(patch_names)


def _should_compute_yplus(boundary: dict, patch_names: list | None) -> bool:
    """Determine whether y+ should be computed for a given boundary.

    When *patch_names* is provided the boundary is selected by name.
    Otherwise the boundary is selected only when its mesh type is ``"wall"``.

    Args:
        boundary: Boundary dictionary.  Must contain key ``"name"``, and
            may contain ``"velocity_type"`` and ``"type"``.
        patch_names: Explicit list of patch names to select, or ``None``
            for auto-detection.

    Returns:
        ``True`` if y+ should be computed for this boundary, ``False``
        otherwise.
    """
    name = boundary["name"]
    if patch_names is not None:
        return name in patch_names
    return boundary.get("type") == "wall"


def _compute_face_viscous_forces(
    velocity,
    velocity_gradient,
    owners_idx,
    boundary_idx,
    n_vec,
    mag_Sf,
    wall_distance,
    dynamic_viscosity,
    nf,
):
    """Compute viscous traction forces on boundary faces.

    Corrects the reconstructed owner-cell gradient so that its face-normal
    derivative matches the boundary diffusion operator,

    ``snGrad(velocity) = (velocity_boundary - velocity_owner) / wall_distance``.

    The resulting face stress uses the incompressible convention
    ``dynamic_viscosity * dev(twoSymm(grad(velocity)))``. Returning the stress traction here keeps
    force diagnostics consistent with the actual fixed-value wall flux rather
    than sampling an uncorrected cell-centred gradient half a cell away.

    Args:
        velocity: Velocity field including boundary-face values.
        velocity_gradient: Velocity gradient field ``(n_elements, 3, 3)``, or
            ``None`` (returns zero forces).
        owners_idx: Indices into *velocity_gradient* for the owner cells of the
            boundary faces ``(nf,)``.
        boundary_idx: Indices into *velocity* for the boundary-face values.
        n_vec: Unit face normal vectors ``(nf, 3)``.
        mag_Sf: Face area magnitudes ``(nf,)``.
        wall_distance: Owner-cell-centre to face distance normal to the face.
        dynamic_viscosity: Dynamic viscosity — scalar ``float`` or per-element array
            ``(n_elements,)``.
        nf: Number of boundary faces (``int``).

    Returns:
        ndarray: Viscous force per face ``(nf, 3)``.
    """
    if velocity_gradient is None:
        return np.zeros((nf, 3))

    grad_face = np.asarray(velocity_gradient[owners_idx], dtype=np.float64).copy()
    distance = np.asarray(wall_distance, dtype=np.float64)
    if distance.shape != (nf,) or np.any(~np.isfinite(distance)) or np.any(distance <= 0.0):
        raise ValueError("Boundary wall distances must be finite and positive")

    # velocity_gradient[d, c] = d(cell_velocity)/d(x_d).  Replace only its normal projection,
    # retaining the reconstructed tangential derivatives.
    sn_grad = (np.asarray(velocity[boundary_idx]) - np.asarray(velocity[owners_idx])) / distance[
        :, None
    ]
    reconstructed_sn_grad = np.einsum("fi,fij->fj", n_vec, grad_face)
    grad_face += n_vec[:, :, None] * (sn_grad - reconstructed_sn_grad)[:, None, :]

    two_symm = grad_face + np.transpose(grad_face, (0, 2, 1))
    divergence = np.trace(grad_face, axis1=1, axis2=2)
    dev_two_symm = two_symm.copy()
    diagonal = np.arange(3)
    dev_two_symm[:, diagonal, diagonal] -= (2.0 / 3.0) * divergence[:, None]

    t_faces = np.einsum("fij,fj->fi", dev_two_symm, n_vec)
    mu_values = np.asarray(dynamic_viscosity, dtype=np.float64)
    mu_face = float(mu_values.item()) if mu_values.ndim == 0 else mu_values[owners_idx, None]
    t_faces = t_faces * mu_face
    return t_faces * mag_Sf[:, np.newaxis]


def _compute_force_coefficients(
    total_force, moment, reference_velocity, reference_area, density, reference_length=None
):
    """Compute force coefficients and the z-axis pitching-moment coefficient.

    Coefficients are normalised by the dynamic pressure
    ``q = 0.5 * density * reference_velocity**2``.

    Args:
        total_force: Total force vector ``(3,)``.
        reference_velocity: Reference velocity magnitude.
        reference_area: Reference area.  If zero or ``None`` all coefficients
            are set to ``0.0``.
        density: Fluid density.
        moment: Integrated moment vector about the requested centre.
        reference_length: Reference length for pitching moment (optional).

    Returns:
        dict: Dictionary with canonical force and moment coefficient keys.
    """
    q = 0.5 * density * reference_velocity**2
    force_coefficients = (
        np.asarray(total_force, dtype=np.float64) / (q * reference_area)
        if reference_area
        else np.zeros(3)
    )
    result = {
        "drag_coefficient": float(force_coefficients[0]),
        "lift_coefficient": float(force_coefficients[1]),
        "side_force_coefficient": float(force_coefficients[2]),
        "force_coefficient_x": float(force_coefficients[0]),
        "force_coefficient_y": float(force_coefficients[1]),
        "force_coefficient_z": float(force_coefficients[2]),
    }
    if reference_length and reference_length > 0 and reference_area:
        result["pitching_moment_coefficient"] = float(
            moment[2] / (q * reference_area * reference_length)
        )
    return result


def compute_y_plus(
    velocity, kinematic_viscosity, mesh_data, geo_data, boundaries, patch_names=None
):
    """Compute wall-unit ``y+`` statistics for selected boundary patches.

    Parameters
    ----------
    velocity : numpy.ndarray
        Cell-centred velocities, shape ``(n_cells_with_ghosts, 3)``, in m/s.
    kinematic_viscosity : float or numpy.ndarray
        Positive viscosity in m²/s, scalar or at least ``(n_cells,)`` values.
    mesh_data, geo_data : dict
        Connectivity and geometry including owner indices, face areas, and wall
        distances in m.
    boundaries : sequence[dict]
        Patch records containing ``name``, ``type``, and face ranges.
    patch_names : sequence[str] or str or None, default=None
        Explicit patch selection. ``None`` auto-selects patches whose mesh type
        is ``"wall"``.

    Returns
    -------
    dict[str, dict[str, float]]
        Per-patch ``min``, ``max``, ``avg``, and ``n_faces`` statistics.

    Raises
    ------
    ValueError
        If viscosity, wall distances, or face areas are invalid.
    """
    owners = mesh_data["owners"]

    patch_names = _normalize_patch_names(patch_names)
    if patch_names is not None and len(patch_names) == 0:
        return {}

    kinematic_viscosity_values = np.asarray(kinematic_viscosity, dtype=np.float64)
    if kinematic_viscosity_values.ndim == 0:
        if not np.isfinite(kinematic_viscosity_values) or kinematic_viscosity_values <= 0.0:
            raise ValueError("kinematic_viscosity must be finite and positive")
        cell_kinematic_viscosity = None
        scalar_kinematic_viscosity = float(kinematic_viscosity_values)
    else:
        n_cells = mesh_data["n_cells"]
        if kinematic_viscosity_values.ndim != 1 or len(kinematic_viscosity_values) < n_cells:
            raise ValueError(
                f"kinematic_viscosity must be scalar or contain at least {n_cells} cell values"
            )
        cell_kinematic_viscosity = kinematic_viscosity_values[:n_cells]
        if not np.all(np.isfinite(cell_kinematic_viscosity)) or np.any(
            cell_kinematic_viscosity <= 0.0
        ):
            raise ValueError("kinematic_viscosity must contain finite positive values")
        scalar_kinematic_viscosity = None

    y_plus_stats = {}

    for boundary in boundaries:
        if not _should_compute_yplus(boundary, patch_names):
            continue

        name = boundary["name"]
        start = boundary["start_face"]
        nf = boundary["n_faces"]
        idx = np.arange(start, start + nf)
        own = owners[idx]

        # 1. Wall distance: cell centre to face centre
        # wall_distance is usually pre-computed in geo_data for FVM
        if "wall_distance" in geo_data:
            d = geo_data["wall_distance"][idx]
        else:
            # Fallback: CF vector projection
            cf_vec = geo_data["cell_connection_vector"][idx]
            d = np.linalg.norm(cf_vec, axis=1)
        if not np.all(np.isfinite(d)) or np.any(d <= 0.0):
            raise ValueError(f"wall distance must be finite and positive on patch {name!r}")

        # 2. Velocity at cell centre (tangential to wall)
        cell_velocity = velocity[own]
        sf = geo_data["face_area_vector"][idx]
        mag_sf = np.linalg.norm(sf, axis=1)
        if not np.all(np.isfinite(mag_sf)) or np.any(mag_sf <= 0.0):
            raise ValueError(f"face areas must be finite and positive on patch {name!r}")
        n_vec = sf / mag_sf[:, np.newaxis]

        # Normal velocity: normal_velocity = (cell_velocity · normal) * normal
        normal_velocity_magnitude = np.sum(cell_velocity * n_vec, axis=1)
        normal_velocity = normal_velocity_magnitude[:, np.newaxis] * n_vec

        # Tangential velocity: tangential_velocity = cell_velocity - normal_velocity
        tangential_velocity = cell_velocity - normal_velocity
        tangential_velocity_magnitude = np.linalg.norm(tangential_velocity, axis=1)

        # 3. Wall shear stress (assuming linear profile:
        # d(tangential_velocity)/dn = tangential_velocity / wall_distance).
        # wall_shear_stress = kinematic_viscosity * density * tangential_velocity / wall_distance
        # friction_velocity = sqrt(wall_shear_stress / density)
        wall_kinematic_viscosity = (
            scalar_kinematic_viscosity
            if cell_kinematic_viscosity is None
            else cell_kinematic_viscosity[own]
        )
        friction_velocity = np.sqrt(wall_kinematic_viscosity * tangential_velocity_magnitude / d)

        # 4. y+ = friction_velocity * d / kinematic_viscosity
        y_plus = friction_velocity * d / wall_kinematic_viscosity

        y_plus_stats[name] = {
            "min": float(np.min(y_plus)),
            "max": float(np.max(y_plus)),
            "avg": float(np.mean(y_plus)),
            "n_faces": nf,
        }

    return y_plus_stats


def compute_surface_face_loads(
    velocity,
    kinematic_pressure,
    dynamic_viscosity,
    density,
    mesh_data,
    geo_data,
    boundaries,
    patch_names=None,
    gradient=None,
):
    """Return discrete pressure and viscous loads on selected boundary faces.

    The arrays use the same face values and normal-gradient reconstruction as
    :func:`compute_surface_forces`.  ``pressure_force`` and
    ``viscous_force`` are forces exerted on the solid, so their sum can be
    integrated directly into a force coefficient or compared face-for-face
    between cell-identical meshes.

    Parameters
    ----------
    velocity : numpy.ndarray
        Cell/ghost velocities, shape ``(n_total, 3)``, in m/s.
    kinematic_pressure : numpy.ndarray
        Cell/ghost ``p/rho`` values, shape ``(n_total,)``, in m²/s².
    dynamic_viscosity : float or numpy.ndarray
        Dynamic viscosity in Pa·s, scalar or one value per cell.
    density : float
        Fluid density in kg/m³.
    mesh_data, geo_data : dict
        Mesh topology/geometry used for owners, normals, areas, and wall
        distances.
    boundaries : sequence[dict]
        Boundary patch records.
    patch_names : sequence[str] or None, default=None
        Selected patch names; ``None`` selects wall patches.
    gradient : numpy.ndarray or None, default=None
        Optional velocity gradient, shape ``(n_total, 3, 3)``, in 1/s.

    Returns
    -------
    dict[str, dict[str, numpy.ndarray]]
        Per-face centers, areas, normals, pressure values, pressure/viscous
        forces, and wall-shear vectors.
    """
    from .gradients import _resolve_gradient_fn as _resolve_grad

    n_cells = mesh_data["n_cells"]
    n_interior = mesh_data["n_interior_faces"]
    owners = mesh_data["owners"]
    rho_value = float(np.asarray(density))
    if not np.isfinite(rho_value) or rho_value <= 0.0:
        raise ValueError("Density must be a finite positive scalar")

    if patch_names is None:
        patch_names = [b["name"] for b in boundaries if b.get("type") == "wall"]

    mu_values = np.asarray(dynamic_viscosity)
    if not np.all(np.isfinite(mu_values)) or np.any(mu_values < 0.0):
        raise ValueError("Dynamic viscosity must be finite and non-negative")
    if mu_values.ndim > 0 and mu_values.shape != (n_cells,):
        raise ValueError(
            f"Dynamic viscosity must be scalar or have shape ({n_cells},), got {mu_values.shape}"
        )
    velocity_gradient = None
    if not np.all(mu_values == 0.0):
        velocity_gradient = (
            gradient
            if gradient is not None
            else _resolve_grad(geo_data)(velocity, mesh_data, geo_data)
        )
        if velocity_gradient.ndim == 3:
            velocity_gradient = velocity_gradient[:n_cells]

    results = {}
    for boundary in boundaries:
        name = boundary["name"]
        if name not in patch_names:
            continue
        start = int(boundary["start_face"])
        nf = int(boundary["n_faces"])
        face_idx = np.arange(start, start + nf)
        owners_idx = owners[face_idx]
        boundary_idx = n_cells + (face_idx - n_interior)
        sf = np.asarray(geo_data["face_area_vector"])[face_idx]
        area = np.linalg.norm(sf, axis=1)
        normal = sf / (area[:, None] + 1e-30)
        kinematic_pressure_face = np.asarray(kinematic_pressure, dtype=np.float64)[boundary_idx]
        pressure_force = rho_value * kinematic_pressure_face[:, None] * sf
        viscous_force = -_compute_face_viscous_forces(
            velocity,
            velocity_gradient,
            owners_idx,
            boundary_idx,
            normal,
            area,
            geo_data["wall_distance"][face_idx],
            dynamic_viscosity,
            nf,
        )
        traction = viscous_force / (area[:, None] + 1e-30)
        wall_shear = traction - np.einsum("fi,fi->f", traction, normal)[:, None] * normal
        results[name] = {
            "face_centre": np.asarray(geo_data["face_centre"])[face_idx].copy(),
            "face_area": area,
            "normal": normal,
            "kinematic_pressure": kinematic_pressure_face,
            "pressure_force": pressure_force,
            "viscous_force": viscous_force,
            "wall_shear": wall_shear,
        }
    return results


def compute_surface_forces(
    velocity,
    kinematic_pressure,
    dynamic_viscosity,
    density,
    mesh_data,
    geo_data,
    boundaries,
    patch_names=None,
    reference_velocity=None,
    reference_area=None,
    reference_length=None,
    moment_centre=None,
    gradient=None,
):
    """Integrate pressure and viscous forces over selected surface patches.

    Parameters
    ----------
    velocity : numpy.ndarray
        Cell/ghost velocities, shape ``(n_total, 3)``, in m/s.
    kinematic_pressure : numpy.ndarray
        Cell/ghost kinematic pressure, shape ``(n_total,)``, in m²/s².
    dynamic_viscosity : float or numpy.ndarray
        Dynamic viscosity in Pa·s, scalar or one value per cell.
    density : float
        Fluid density in kg/m³.
    mesh_data, geo_data : dict
        Topology/geometry consumed by :func:`compute_surface_face_loads`.
    boundaries : sequence[dict]
        Boundary patch records.
    patch_names : sequence[str] or None, default=None
        Patches to integrate; ``None`` selects all wall patches.
    reference_velocity, reference_area : float or None
        Dynamic-pressure reference values used for coefficients.
    reference_length : float or None, default=None
        Length in m for the z-axis pitching-moment coefficient.
    moment_centre : numpy.ndarray or None, default=None
        Moment origin, shape ``(3,)``, in m.
    gradient : numpy.ndarray or None, default=None
        Optional velocity gradient, shape ``(n_total, 3, 3)``, in 1/s.

    Returns
    -------
    dict[str, dict[str, object]]
        Per-patch integrated pressure/viscous/total force vectors in N,
        moment vectors in N·m, face counts, and optional coefficients.
    """
    face_loads = compute_surface_face_loads(
        velocity,
        kinematic_pressure,
        dynamic_viscosity,
        density,
        mesh_data,
        geo_data,
        boundaries,
        patch_names=patch_names,
        gradient=gradient,
    )
    results = {}
    for name, loads in face_loads.items():
        Fp_faces = loads["pressure_force"]
        Fv_faces = loads["viscous_force"]
        Fp = np.sum(Fp_faces, axis=0)
        Fv = np.sum(Fv_faces, axis=0)
        total_force = Fp + Fv
        centre = np.zeros(3) if moment_centre is None else np.asarray(moment_centre, dtype=float)
        if centre.shape != (3,):
            raise ValueError("moment_centre must contain exactly three coordinates")
        arm = loads["face_centre"] - centre
        moment = np.sum(np.cross(arm, Fp_faces + Fv_faces), axis=0)
        has_refs = (
            reference_velocity is not None and reference_area is not None and density is not None
        )
        coeffs = (
            _compute_force_coefficients(
                total_force, moment, reference_velocity, reference_area, density, reference_length
            )
            if has_refs
            else {}
        )
        results[name] = {
            "pressure_force": Fp,
            "viscous_force": Fv,
            "total_force": total_force,
            "moment": moment,
            "coeffs": coeffs,
            "n_faces": len(loads["face_area"]),
        }

    return results


def merge_partition_forces(parts):
    """Merge non-overlapping per-patch force fragments from MPI ranks.

    Parameters
    ----------
    parts : iterable[dict]
        Rank-local results matching :func:`compute_surface_forces`.

    Returns
    -------
    dict
        Force/moment vectors and summed face counts keyed by patch name.
    """
    merged = {}
    for rank_forces in parts:
        for name, values in rank_forces.items():
            target = merged.setdefault(
                name,
                {
                    "pressure_force": np.zeros(3),
                    "viscous_force": np.zeros(3),
                    "total_force": np.zeros(3),
                    "moment": np.zeros(3),
                    "coeffs": {},
                    "n_faces": 0,
                },
            )
            for key in ("pressure_force", "viscous_force", "total_force", "moment"):
                target[key] += np.asarray(values[key], dtype=np.float64)
            for key, value in values["coeffs"].items():
                target["coeffs"][key] = target["coeffs"].get(key, 0.0) + float(value)
            target["n_faces"] += int(values["n_faces"])
    return merged


def merge_partition_yplus(parts):
    """Combine per-patch y+ extrema and face-weighted means across ranks.

    Parameters
    ----------
    parts : iterable[dict]
        Rank-local statistics matching :func:`compute_y_plus`.

    Returns
    -------
    dict
        Global minimum, maximum, face-weighted average, and face count per
        patch.
    """
    merged = {}
    for rank_stats in parts:
        for name, values in rank_stats.items():
            count = int(values["n_faces"])
            target = merged.setdefault(
                name,
                {"min": np.inf, "max": -np.inf, "weighted": 0.0, "n_faces": 0},
            )
            target["min"] = min(target["min"], float(values["min"]))
            target["max"] = max(target["max"], float(values["max"]))
            target["weighted"] += float(values["avg"]) * count
            target["n_faces"] += count
    return {
        name: {
            "min": values["min"],
            "max": values["max"],
            "avg": values["weighted"] / values["n_faces"],
            "n_faces": values["n_faces"],
        }
        for name, values in merged.items()
    }
