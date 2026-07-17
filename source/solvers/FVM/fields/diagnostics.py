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

from . import gradients


def compute_courant_number(U, phi, dt, mesh_data, geo_data):
    """
    Compute Courant number field.
    Co = 0.5 * dt * sum(|phi_f|) / V_c

    Args:
        U: Velocity field (not strictly needed if phi is provided)
        phi: Face mass flux (rho*U.Sf)
        dt: Time step
        mesh_data: Mesh connectivity
        geo_data: Geometric data

    Returns:
        Co: Courant number field (n_elements)
    """
    n_elements = mesh_data["n_elements"]
    n_interior = mesh_data["n_interior_faces"]
    owners = mesh_data["owners"]
    neighbours = mesh_data["neighbours"]
    volumes = geo_data["element_volumes"]

    # Absolute flux
    abs_phi = np.abs(phi)

    # Initialize Co field
    Co = np.zeros(n_elements)

    # Interior faces contribution
    np.add.at(Co, owners[:n_interior], abs_phi[:n_interior])
    np.add.at(Co, neighbours[:n_interior], abs_phi[:n_interior])

    # Boundary faces contribution
    np.add.at(Co, owners[n_interior:], abs_phi[n_interior:])

    # Final scaling
    Co = 0.5 * dt * Co / (volumes + 1e-12)

    return Co


def compute_continuity_error(phi, mesh_data, geo_data):
    """Per-cell continuity residual ∮ U·dS = Σ_f (±φ_f) [m³/s].

    For a discretely divergence-free (incompressible) solution this net face
    flux is ~0 in every cell.  Returned unnormalised so callers can form both
    the global mass imbalance Σ|residual| and the local divergence
    max|residual / V|.

    Args:
        phi: Face volumetric/mass flux (U·Sf), length n_faces.
        mesh_data: Mesh connectivity.
        geo_data: Geometric data (unused; kept for signature parity).

    Returns:
        np.ndarray: net flux per cell (n_elements,).
    """
    n_elements = mesh_data["n_elements"]
    n_interior = mesh_data["n_interior_faces"]
    owners = mesh_data["owners"]
    neighbours = mesh_data["neighbours"]

    div = np.zeros(n_elements)
    np.add.at(div, owners[:n_interior], phi[:n_interior])
    np.add.at(div, neighbours[:n_interior], -phi[:n_interior])
    np.add.at(div, owners[n_interior:], phi[n_interior:])
    return div


def compute_kinetic_energy(U, geo_data, density=1.0):
    """Return volume-integrated kinetic energy for the interior cells."""
    volumes = np.asarray(geo_data["element_volumes"], dtype=np.float64)
    velocity = np.asarray(U[: len(volumes)], dtype=np.float64)
    rho = np.asarray(density, dtype=np.float64)
    if rho.ndim == 0:
        rho = np.full(len(volumes), float(rho))
    if rho.shape != volumes.shape or np.any(rho <= 0.0) or not np.all(np.isfinite(rho)):
        raise ValueError(f"density must be finite and positive with shape {volumes.shape}")
    return 0.5 * float(np.sum(rho * volumes * np.sum(velocity * velocity, axis=1)))


def compute_enstrophy(U, mesh_data, geo_data):
    """Return ``0.5 ∫ |curl(U)|² dV`` over the interior cells."""
    vorticity = compute_vorticity(U, mesh_data, geo_data)
    volumes = np.asarray(geo_data["element_volumes"], dtype=np.float64)
    return 0.5 * float(np.sum(volumes * np.sum(vorticity * vorticity, axis=1)))


def compute_vorticity(U, mesh_data, geo_data):
    """
    Compute vorticity field: w = curl(U)

    Args:
        U: Velocity field (N, 3)
        mesh_data: Mesh connectivity
        geo_data: Geometric data

    Returns:
        vorticity: Vorticity field (n_elements, 3)
    """
    # 1. Compute velocity gradient grad(U)
    # grad_U shape: (n_total, 3, 3) where grad_U[i, j, k] is dU_k/dx_j
    _grad_fn = gradients._resolve_gradient_fn(geo_data)
    grad_U = _grad_fn(U, mesh_data, geo_data)

    n_elements = mesh_data["n_elements"]

    # 2. Extract components
    # grad_U[:, 0, 0] = dUx/dx
    # grad_U[:, 1, 0] = dUx/dy
    # grad_U[:, 2, 0] = dUx/dz
    # grad_U[:, 0, 1] = dUy/dx
    # grad_U[:, 1, 1] = dUy/dy
    # grad_U[:, 2, 1] = dUy/dz
    # grad_U[:, 0, 2] = dUz/dx
    # grad_U[:, 1, 2] = dUz/dy
    # grad_U[:, 2, 2] = dUz/dz

    vorticity = np.zeros((n_elements, 3))

    # wx = dUz/dy - dUy/dz
    vorticity[:, 0] = grad_U[:n_elements, 1, 2] - grad_U[:n_elements, 2, 1]
    # wy = dUx/dz - dUz/dx
    vorticity[:, 1] = grad_U[:n_elements, 2, 0] - grad_U[:n_elements, 0, 2]
    # wz = dUy/dx - dUx/dy
    vorticity[:, 2] = grad_U[:n_elements, 0, 1] - grad_U[:n_elements, 1, 0]

    return vorticity


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
        return [p.strip() for p in patch_names.split(",") if p.strip()]
    return list(patch_names)


def _should_compute_yplus(boundary: dict, patch_names: list | None) -> bool:
    """Determine whether y+ should be computed for a given boundary.

    When *patch_names* is provided the boundary is selected by name.
    Otherwise the boundary is selected only when its mesh type is ``"wall"``.

    Args:
        boundary: Boundary dictionary.  Must contain key ``"name"``, and
            may contain ``"bc_type_U"`` and ``"type"``.
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
    U,
    gradU,
    owners_idx,
    boundary_idx,
    n_vec,
    mag_Sf,
    wall_dist,
    mu,
    nf,
):
    """Compute viscous traction forces on boundary faces.

    Corrects the reconstructed owner-cell gradient so that its face-normal
    derivative matches the boundary diffusion operator,

    ``snGrad(U) = (U_boundary - U_owner) / wall_dist``.

    The resulting face stress is the incompressible OpenFOAM convention
    ``mu * dev(twoSymm(grad(U)))``.  Returning the stress traction here keeps
    force diagnostics consistent with the actual fixed-value wall flux rather
    than sampling an uncorrected cell-centred gradient half a cell away.

    Args:
        U: Velocity field including boundary-face values.
        gradU: Velocity gradient field ``(n_elements, 3, 3)``, or
            ``None`` (returns zero forces).
        owners_idx: Indices into *gradU* for the owner cells of the
            boundary faces ``(nf,)``.
        boundary_idx: Indices into *U* for the boundary-face values.
        n_vec: Unit face normal vectors ``(nf, 3)``.
        mag_Sf: Face area magnitudes ``(nf,)``.
        wall_dist: Owner-centroid to face distance normal to the face.
        mu: Dynamic viscosity — scalar ``float`` or per-element array
            ``(n_elements,)``.
        nf: Number of boundary faces (``int``).

    Returns:
        ndarray: Viscous force per face ``(nf, 3)``.
    """
    if gradU is None:
        return np.zeros((nf, 3))

    grad_face = np.asarray(gradU[owners_idx], dtype=np.float64).copy()
    distance = np.asarray(wall_dist, dtype=np.float64)
    if distance.shape != (nf,) or np.any(~np.isfinite(distance)) or np.any(distance <= 0.0):
        raise ValueError("Boundary wall distances must be finite and positive")

    # gradU[d, c] = d(U_c)/d(x_d).  Replace only its normal projection,
    # retaining the reconstructed tangential derivatives.
    sn_grad = (np.asarray(U[boundary_idx]) - np.asarray(U[owners_idx])) / distance[:, None]
    reconstructed_sn_grad = np.einsum("fi,fij->fj", n_vec, grad_face)
    grad_face += n_vec[:, :, None] * (sn_grad - reconstructed_sn_grad)[:, None, :]

    two_symm = grad_face + np.transpose(grad_face, (0, 2, 1))
    divergence = np.trace(grad_face, axis1=1, axis2=2)
    dev_two_symm = two_symm.copy()
    diagonal = np.arange(3)
    dev_two_symm[:, diagonal, diagonal] -= (2.0 / 3.0) * divergence[:, None]

    t_faces = np.einsum("fij,fj->fi", dev_two_symm, n_vec)
    mu_face = float(mu) if np.isscalar(mu) else np.asarray(mu)[owners_idx, None]
    t_faces = t_faces * mu_face
    return t_faces * mag_Sf[:, np.newaxis]


def _compute_force_coefficients(Ftot, moment, ref_U, ref_area, rho, ref_length=None):
    """Compute force coefficients and the z-axis pitching-moment coefficient.

    Coefficients are normalised by the dynamic pressure
    ``q = 0.5 * rho * ref_U**2``.

    Args:
        Ftot: Total force vector ``(3,)``.
        ref_U: Reference velocity magnitude.
        ref_area: Reference area.  If zero or ``None`` all coefficients
            are set to ``0.0``.
        rho: Fluid density.
        moment: Integrated moment vector about the requested centre.
        ref_length: Reference length for pitching moment (optional).

    Returns:
        dict: Dictionary with keys ``"Cd"``, ``"Cl"``, ``"Cz"``, and
        optionally ``"Cm"`` (pitching moment) when *ref_length* is
        positive.
    """
    q = 0.5 * rho * ref_U**2
    cd = float(Ftot[0] / (q * ref_area)) if ref_area else 0.0
    cl = float(Ftot[1] / (q * ref_area)) if ref_area else 0.0
    cz = float(Ftot[2] / (q * ref_area)) if ref_area else 0.0
    result = {"Cd": cd, "Cl": cl, "Cz": cz}
    if ref_length and ref_length > 0 and ref_area:
        result["Cm"] = float(moment[2] / (q * ref_area * ref_length))
    return result


def compute_y_plus(U, nu, mesh_data, geo_data, boundaries, patch_names=None):
    """
    Compute y+ for wall boundaries and return statistics.

    Args:
        U: Velocity field
        nu: Kinematic viscosity (scalar or field)
        mesh_data: Mesh connectivity
        geo_data: Geometric data
        boundaries: Boundary list
        patch_names: Optional list of patch names to compute y+ for. If None,
                     the function auto-selects wall patches (same as previous behavior).

    Returns:
        y_plus_stats: Dictionary mapping selected boundary names to {min, max, avg}
    """
    owners = mesh_data["owners"]

    patch_names = _normalize_patch_names(patch_names)
    if patch_names is not None and len(patch_names) == 0:
        return {}

    # Ensure nu is reachable
    nu_val = nu if isinstance(nu, float | int) else np.mean(nu)

    y_plus_stats = {}

    for boundary in boundaries:
        if not _should_compute_yplus(boundary, patch_names):
            continue

        name = boundary["name"]
        start = boundary["startFace"]
        nf = boundary["nFaces"]
        idx = np.arange(start, start + nf)
        own = owners[idx]

        # 1. Wall distance: cell center to face center
        # wall_dist is usually pre-computed in geo_data for FVM
        if "wall_dist" in geo_data:
            d = geo_data["wall_dist"][idx]
        else:
            # Fallback: CF vector projection
            cf_vec = geo_data["face_cf_vector"][idx]
            d = np.linalg.norm(cf_vec, axis=1)

        # 2. Velocity at cell center (tangential to wall)
        U_c = U[own]
        sf = geo_data["face_sf"][idx]
        mag_sf = np.linalg.norm(sf, axis=1)
        n_vec = sf / (mag_sf[:, np.newaxis] + 1e-30)

        # Normal velocity: Un = (U . n) * n
        U_n_mag = np.sum(U_c * n_vec, axis=1)
        U_n = U_n_mag[:, np.newaxis] * n_vec

        # Tangential velocity: Ut = U - Un
        U_t = U_c - U_n
        U_t_mag = np.linalg.norm(U_t, axis=1)

        # 3. Wall Shear Stress (Assuming linear profile: du/dn = Ut/d)
        # tau_w = nu * rho * (Ut/d)
        # u_tau = sqrt(tau_w / rho) = sqrt(nu * Ut / d)
        u_tau = np.sqrt(nu_val * U_t_mag / (d + 1e-12))

        # 4. y+ = u_tau * d / nu
        y_plus = u_tau * d / (nu_val + 1e-12)

        y_plus_stats[name] = {
            "min": np.min(y_plus),
            "max": np.max(y_plus),
            "avg": np.mean(y_plus),
            "nFaces": nf,
        }

    return y_plus_stats


def compute_surface_forces(
    U,
    p,
    mu,
    rho,
    mesh_data,
    geo_data,
    boundaries,
    patch_names=None,
    ref_U=None,
    ref_area=None,
    ref_length=None,
    moment_centre=None,
):
    """
    Compute surface forces (pressure + viscous) on boundary patches.

    Args:
        U: Velocity field (n_elements + n_boundary, 3)
        p: Kinematic pressure field (n_elements + n_boundary,)
        mu: Dynamic viscosity (scalar or array)
        rho: Density (scalar)
        mesh_data: Mesh connectivity
        geo_data: Geometric data
        boundaries: List of boundary patch dicts
        patch_names: list of patch names to compute (default: all wall patches)
        ref_U: reference velocity for coefficient calculation
        ref_area: reference area for coefficient calculation

    Returns:
        dict: mapping patch name -> {force: [Fx,Fy,Fz], Cd: , Cl: }
    """
    import numpy as _np

    from .gradients import _resolve_gradient_fn as _resolve_grad

    _grad = _resolve_grad(geo_data)

    n_elements = mesh_data["n_elements"]
    n_interior = mesh_data["n_interior_faces"]
    owners = mesh_data["owners"]

    rho_value = float(np.asarray(rho))
    if not np.isfinite(rho_value) or rho_value <= 0.0:
        raise ValueError("Density must be a finite positive scalar")

    if patch_names is None:
        patch_names = [
            b["name"]
            for b in boundaries
            if (b.get("type") == "wall" or "wall" in b["name"].lower())
        ]

    gradU = None
    mu_values = _np.asarray(mu)
    if not _np.all(_np.isfinite(mu_values)) or _np.any(mu_values < 0.0):
        raise ValueError("Dynamic viscosity must be finite and non-negative")
    if mu_values.ndim > 0 and mu_values.shape != (n_elements,):
        raise ValueError(
            f"Dynamic viscosity must be scalar or have shape ({n_elements},), got {mu_values.shape}"
        )
    mu_is_zero = _np.all(mu_values == 0.0)

    if not mu_is_zero:
        gradU = _grad(U, mesh_data, geo_data)
        if gradU.ndim == 3:
            gradU = gradU[:n_elements]

    results = {}
    for b in boundaries:
        name = b["name"]
        if name not in patch_names:
            continue
        start = b["startFace"]
        nf = b["nFaces"]
        face_idx = _np.arange(start, start + nf)
        owners_idx = owners[face_idx]
        boundary_idx = n_elements + (face_idx - n_interior)
        Sf = geo_data["face_sf"][face_idx]
        mag_Sf = _np.linalg.norm(Sf, axis=1)
        n_vec = Sf / (mag_Sf[:, _np.newaxis] + 1e-30)
        p_face = _np.asarray(p)[boundary_idx]
        # ``p`` is kinematic pressure, so rho is required to recover force.
        Fp_faces = rho_value * p_face[:, _np.newaxis] * Sf
        Fv_faces = -_compute_face_viscous_forces(
            U,
            gradU,
            owners_idx,
            boundary_idx,
            n_vec,
            mag_Sf,
            geo_data["wall_dist"][face_idx],
            mu,
            nf,
        )
        Fp = _np.sum(Fp_faces, axis=0)
        Fv = _np.sum(Fv_faces, axis=0)
        Ftot = Fp + Fv
        centre = _np.zeros(3) if moment_centre is None else _np.asarray(moment_centre, dtype=float)
        if centre.shape != (3,):
            raise ValueError("moment_centre must contain exactly three coordinates")
        arm = geo_data["face_centroids"][face_idx] - centre
        moment = _np.sum(_np.cross(arm, Fp_faces + Fv_faces), axis=0)
        has_refs = ref_U is not None and ref_area is not None and rho is not None
        coeffs = (
            _compute_force_coefficients(Ftot, moment, ref_U, ref_area, rho, ref_length)
            if has_refs
            else {}
        )
        results[name] = {
            "Fp": Fp,
            "Fv": Fv,
            "Ftot": Ftot,
            "Mtot": moment,
            "coeffs": coeffs,
            "nFaces": nf,
        }

    return results


def merge_partition_forces(parts):
    """Sum non-overlapping patch-force fragments from all MPI ranks."""
    merged = {}
    for rank_forces in parts:
        for name, values in rank_forces.items():
            target = merged.setdefault(
                name,
                {
                    "Fp": np.zeros(3),
                    "Fv": np.zeros(3),
                    "Ftot": np.zeros(3),
                    "Mtot": np.zeros(3),
                    "coeffs": {},
                    "nFaces": 0,
                },
            )
            for key in ("Fp", "Fv", "Ftot", "Mtot"):
                target[key] += np.asarray(values[key], dtype=np.float64)
            for key, value in values["coeffs"].items():
                target["coeffs"][key] = target["coeffs"].get(key, 0.0) + float(value)
            target["nFaces"] += int(values["nFaces"])
    return merged


def merge_partition_yplus(parts):
    """Combine per-patch extrema and face-weighted means from MPI ranks."""
    merged = {}
    for rank_stats in parts:
        for name, values in rank_stats.items():
            count = int(values["nFaces"])
            target = merged.setdefault(
                name,
                {"min": np.inf, "max": -np.inf, "weighted": 0.0, "nFaces": 0},
            )
            target["min"] = min(target["min"], float(values["min"]))
            target["max"] = max(target["max"], float(values["max"]))
            target["weighted"] += float(values["avg"]) * count
            target["nFaces"] += count
    return {
        name: {
            "min": values["min"],
            "max": values["max"],
            "avg": values["weighted"] / values["nFaces"],
            "nFaces": values["nFaces"],
        }
        for name, values in merged.items()
    }
