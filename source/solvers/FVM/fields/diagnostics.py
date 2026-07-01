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
    Otherwise the boundary is selected if its type is ``"wall"``, its
    velocity boundary condition is ``"fixedValue"``, or its name contains
    ``"wall"``.

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
    bc_type_u = boundary.get("bc_type_U")
    return bc_type_u == "fixedValue" or boundary.get("type") == "wall" or "wall" in name.lower()


def _compute_face_viscous_forces(gradU, owners_idx, n_vec, mag_Sf, mu, nf):
    """Compute viscous traction forces on boundary faces.

    Constructs the symmetric gradient tensor from the velocity gradient at
    the owner cell, projects it onto the face normal, multiplies by
    viscosity and face area magnitude.

    Args:
        gradU: Velocity gradient field ``(n_elements, 3, 3)``, or
            ``None`` (returns zero forces).
        owners_idx: Indices into *gradU* for the owner cells of the
            boundary faces ``(nf,)``.
        n_vec: Unit face normal vectors ``(nf, 3)``.
        mag_Sf: Face area magnitudes ``(nf,)``.
        mu: Dynamic viscosity — scalar ``float`` or per-element array
            ``(n_elements,)``.
        nf: Number of boundary faces (``int``).

    Returns:
        ndarray: Viscous force per face ``(nf, 3)``.
    """
    if gradU is None:
        return np.zeros((nf, 3))
    grad_owner = gradU[owners_idx]
    sym_grad = grad_owner + np.transpose(grad_owner, (0, 2, 1))
    t_faces = np.einsum("fij,fj->fi", sym_grad, n_vec)
    t_faces = t_faces * mu if np.isscalar(mu) else t_faces * mu[owners_idx][:, np.newaxis]
    return t_faces * mag_Sf[:, np.newaxis]


def _compute_force_coefficients(Ftot, ref_U, ref_area, rho, ref_length=None, moment_centre=None):
    """Compute drag, lift, side-force, and pitching moment coefficients.

    Coefficients are normalised by the dynamic pressure
    ``q = 0.5 * rho * ref_U**2``.

    Args:
        Ftot: Total force vector ``(3,)``.
        ref_U: Reference velocity magnitude.
        ref_area: Reference area.  If zero or ``None`` all coefficients
            are set to ``0.0``.
        rho: Fluid density.
        ref_length: Reference length for pitching moment (optional).
            Ignored when ``None`` or ``<= 0``.
        moment_centre: Moment centre (unused in the current simplified
            implementation).

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
    # Pitching moment Cm about moment_centre (simplified: moment arm from origin)
    if ref_length and ref_length > 0:
        # Simplified: use Fz as proxy for pitching moment on symmetric bodies
        cm = float(-Ftot[2] / (q * ref_area * ref_length))  # nose-up positive
        result["Cm"] = cm
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
    nu_val = nu if isinstance(nu, (float, int)) else np.mean(nu)

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
    U, p, mu, rho, mesh_data, geo_data, boundaries, patch_names=None, ref_U=None, ref_area=None,
    ref_length=None, moment_centre=None
):
    """
    Compute surface forces (pressure + viscous) on boundary patches.

    Args:
        U: Velocity field (n_elements + n_boundary, 3)
        p: Pressure field (n_elements + n_boundary,)
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
    owners = mesh_data["owners"]

    if patch_names is None:
        patch_names = [
            b["name"]
            for b in boundaries
            if (b.get("type") == "wall" or "wall" in b["name"].lower())
        ]

    gradU = None
    try:
        mu_is_zero = _np.allclose(mu, 0.0)
    except Exception:
        mu_is_zero = False

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
        Sf = geo_data["face_sf"][face_idx]
        mag_Sf = _np.linalg.norm(Sf, axis=1)
        n_vec = Sf / (mag_Sf[:, _np.newaxis] + 1e-30)
        p_owner = p[owners_idx]
        Fp = _np.sum(p_owner[:, _np.newaxis] * Sf, axis=0)
        Fv = -_np.sum(_compute_face_viscous_forces(gradU, owners_idx, n_vec, mag_Sf, mu, nf), axis=0)
        Ftot = Fp + Fv
        has_refs = ref_U is not None and ref_area is not None and rho is not None
        coeffs = _compute_force_coefficients(Ftot, ref_U, ref_area, rho, ref_length, moment_centre) if has_refs else {}
        results[name] = {"Fp": Fp, "Fv": Fv, "Ftot": Ftot, "coeffs": coeffs, "nFaces": nf}

    return results
