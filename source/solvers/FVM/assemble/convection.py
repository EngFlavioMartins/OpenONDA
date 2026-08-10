#!/usr/bin/env python3
"""
Convection Term Assembly for OpenONDA FVM Solver

Implements convection term discretization:
- Upwind scheme (first-order, stable)
- Central differencing (second-order, may oscillate)
- Deferred correction approach

Converted from uFVM cfdAssembleConvectionTerm.m
"""

import numpy as np

from ..schemes.boundaries import BOUNDARIES, BoundaryStrategy
from ..schemes.limiters import apply_limiter, is_limited_scheme


def _convection_boundary_strategy(boundary_patch):
    """Resolve velocity/scalar boundary behavior without an implicit fallback."""
    type_u = boundary_patch.get("bc_type_U")
    if type_u is not None:
        return BOUNDARIES.strategy(type_u, "U", "convection")
    bc_type = boundary_patch.get("bc_type")
    return BOUNDARIES.strategy(bc_type, "scalar", "convection")


def assemble_convection_term_upwind(phi, mdot, mesh_data, *, include_total_flux=True):
    """
    Assemble convection term using upwind scheme.

    Upwind: φ_f = φ_upwind (first-order, stable)

    Args:
        phi: Field values (n_elements + n_boundary,)
        mdot: Mass flow rate through faces (n_faces,)
        mesh_data: Mesh connectivity

    Returns:
        dict: Flux coefficients
    """

    n_interior_faces = mesh_data["n_interior_faces"]
    owners = mesh_data["owners"][:n_interior_faces]
    neighbours = mesh_data["neighbours"][:n_interior_faces]

    mdot_i = mdot[:n_interior_faces]

    # Upwind scheme: use upstream value
    # If mdot > 0: flow from owner to neighbor, use owner value
    # If mdot < 0: flow from neighbor to owner, use neighbor value

    flux_cf = np.maximum(mdot_i, 0.0)  # Owner contribution
    flux_ff = np.minimum(mdot_i, 0.0)  # Neighbor contribution

    # No explicit correction for upwind
    flux_vf = np.zeros_like(mdot_i)

    # Total flux
    result = {"flux_cf": flux_cf, "flux_ff": flux_ff, "flux_vf": flux_vf}
    if include_total_flux:
        result["flux_tf"] = flux_cf * phi[owners] + flux_ff * phi[neighbours]
    return result


def assemble_convection_term_central(phi, mdot, mesh_data, geo_data, *, include_total_flux=True):
    """
    Assemble convection term using central differencing.

    Central: φ_f = w*φ_neighbor + (1-w)*φ_owner (second-order)

    Args:
        phi: Field values
        mdot: Mass flow rate through faces
        mesh_data: Mesh connectivity
        geo_data: Geometric data (for weights)

    Returns:
        dict: Flux coefficients
    """

    n_interior_faces = mesh_data["n_interior_faces"]
    owners = mesh_data["owners"][:n_interior_faces]
    neighbours = mesh_data["neighbours"][:n_interior_faces]

    mdot_i = mdot[:n_interior_faces]
    weights = geo_data["face_weights"][:n_interior_faces]

    # Central differencing coefficients
    flux_cf = (1 - weights) * mdot_i
    flux_ff = weights * mdot_i

    # No explicit correction for pure central
    flux_vf = np.zeros_like(mdot_i)

    # Total flux
    result = {"flux_cf": flux_cf, "flux_ff": flux_ff, "flux_vf": flux_vf}
    if include_total_flux:
        result["flux_tf"] = flux_cf * phi[owners] + flux_ff * phi[neighbours]
    return result


def assemble_convection_term_deferred_correction(
    phi, mdot, mesh_data, geo_data, *, include_total_flux=True
):
    """
    Assemble convection term using deferred correction.

    Approach:
    1. Use upwind for implicit part (stable)
    2. Add correction to central as explicit term (accurate)

    This gives stability of upwind with accuracy approaching central.

    Args:
        phi: Field values
        mdot: Mass flow rate
        mesh_data: Mesh connectivity
        geo_data: Geometric data

    Returns:
        dict: Flux coefficients
    """

    n_interior_faces = mesh_data["n_interior_faces"]
    owners = mesh_data["owners"][:n_interior_faces]
    neighbours = mesh_data["neighbours"][:n_interior_faces]

    mdot_i = mdot[:n_interior_faces]
    weights = geo_data["face_weights"][:n_interior_faces]

    # Upwind part (implicit)
    flux_cf_upwind = np.maximum(mdot_i, 0.0)
    flux_ff_upwind = np.minimum(mdot_i, 0.0)

    # Central part
    flux_cf_central = (1 - weights) * mdot_i
    flux_ff_central = weights * mdot_i

    # Deferred correction: explicit term = central - upwind
    flux_vf = (flux_cf_central - flux_cf_upwind) * phi[owners] + (
        flux_ff_central - flux_ff_upwind
    ) * phi[neighbours]

    # Use upwind for implicit coefficients
    flux_cf = flux_cf_upwind
    flux_ff = flux_ff_upwind

    # Total flux
    result = {"flux_cf": flux_cf, "flux_ff": flux_ff, "flux_vf": flux_vf}
    if include_total_flux:
        result["flux_tf"] = flux_cf * phi[owners] + flux_ff * phi[neighbours] + flux_vf
    return result


def _tvd_face_psi(phi, mdot_i, grad_phi, owners, neighbours, cf_vector, limiter):
    """Per-face standard TVD limiter ψ ∈ [0, 2].

    r = 2 (d · ∇φ_upwind) / (φ_N − φ_P) − 1,  d = c_N − c_C (owner→neighbour).
    Extrema (φ_N ≈ φ_P) are handled by saturating r so the limiter → upwind.
    """
    if grad_phi is None:
        raise ValueError("Limited convection schemes require grad_phi (cell gradient).")
    if grad_phi.ndim == 3 and grad_phi.shape[2] == 1:
        grad_phi = grad_phi.squeeze(-1)
    if not np.all(np.isfinite(phi)) or not np.all(np.isfinite(grad_phi)):
        raise FloatingPointError("TVD convection received a non-finite field or gradient")

    phi_p = phi[owners]
    phi_n = phi[neighbours]
    gradf = phi_n - phi_p

    grad_cp = np.sum(grad_phi[owners] * cf_vector, axis=1)
    grad_cn = np.sum(grad_phi[neighbours] * cf_vector, axis=1)
    grad_cf = np.where(mdot_i >= 0.0, grad_cp, grad_cn)

    small = np.abs(gradf) < 1e-30 * np.maximum(np.abs(grad_cf), 1.0)
    r = np.full_like(gradf, -1.0, dtype=np.float64)
    np.divide(2.0 * grad_cf, gradf, out=r, where=~small)
    r[~small] -= 1.0
    # At an extremum (gradf→0) force upwind (r large negative → ψ=0) unless the
    # upwind gradient agrees in sign (smooth plateau → keep some blend).
    r = np.where(small, np.where(np.sign(grad_cf) == np.sign(gradf), 1000.0, -1.0), r)
    return apply_limiter(limiter, r)


def assemble_convection_term_limited(
    phi,
    mdot,
    mesh_data,
    geo_data,
    grad_phi,
    limiter,
    psi=None,
    *,
    include_total_flux=True,
):
    """High-resolution TVD convection in deferred-correction form.

    Implicit part = upwind (bounded, diagonally dominant); explicit correction =
    ``ψ · mdot · (φ_linear − φ_upwind)`` where ψ comes from the TVD limiter (or a
    constant ``psi`` for blended schemes such as LUST).  ψ = 1 ⇒ pure central,
    ψ = 0 ⇒ upwind.
    """
    n_interior_faces = mesh_data["n_interior_faces"]
    owners = mesh_data["owners"][:n_interior_faces]
    neighbours = mesh_data["neighbours"][:n_interior_faces]
    mdot_i = mdot[:n_interior_faces]
    weights = geo_data["face_weights"][:n_interior_faces]
    cf_vector = geo_data["face_cf_vector"][:n_interior_faces]

    if psi is None:
        psi = _tvd_face_psi(phi, mdot_i, grad_phi, owners, neighbours, cf_vector, limiter)
    else:
        psi = np.full_like(mdot_i, float(psi))

    # Implicit upwind coefficients.
    flux_cf = np.maximum(mdot_i, 0.0)
    flux_ff = np.minimum(mdot_i, 0.0)

    # Deferred high-resolution correction: ψ·mdot·(φ_linear − φ_upwind).
    phi_upwind = np.where(mdot_i >= 0.0, phi[owners], phi[neighbours])
    phi_linear = weights * phi[neighbours] + (1.0 - weights) * phi[owners]
    flux_vf = psi * mdot_i * (phi_linear - phi_upwind)

    result = {"flux_cf": flux_cf, "flux_ff": flux_ff, "flux_vf": flux_vf}
    if include_total_flux:
        result["flux_tf"] = flux_cf * phi[owners] + flux_ff * phi[neighbours] + flux_vf
    return result


def assemble_convection_term_boundary(
    phi,
    mdot,
    boundary_patch,
    mesh_data,
    geo_data=None,
    scheme="upwind",
    grad_phi=None,
    *,
    include_total_flux=True,
):
    """
    Assemble convection term for boundary faces (1st-order upwind).

    Physical boundary values follow their mathematical condition directly.
    Cyclic faces use the selected interior scheme and a true paired-cell column.

    Args:
        phi: Field values including boundary elements
        mdot: Mass flow rate
        boundary_patch: Boundary patch info
        mesh_data: Mesh connectivity

    Returns:
        dict: Flux coefficients for boundary
    """

    n_interior_faces = mesh_data["n_interior_faces"]
    n_elements = mesh_data["n_elements"]

    start_face = boundary_patch["startFace"]
    n_faces = boundary_patch["nFaces"]
    end_face = start_face + n_faces

    b_face_indices = np.arange(start_face, end_face)
    owners_b = mesh_data["owners"][b_face_indices]

    mdot_b = mdot[b_face_indices]

    strategy = _convection_boundary_strategy(boundary_patch)
    if strategy is BoundaryStrategy.CYCLIC:
        if geo_data is None:
            raise ValueError("Cyclic convection requires geometric interpolation data")
        neighbours_b = mesh_data["boundary_neighbours"][b_face_indices]
        if np.any(neighbours_b < 0):
            raise ValueError("Cyclic boundary faces are missing paired owner cells")
        weights = geo_data["face_weights"][b_face_indices]
        scheme_name = str(scheme)
        flux_cf_upwind = np.maximum(mdot_b, 0.0)
        flux_ff_upwind = np.minimum(mdot_b, 0.0)
        if scheme_name == "upwind":
            flux_cf = flux_cf_upwind
            flux_ff = flux_ff_upwind
            flux_vf = np.zeros_like(mdot_b)
        elif scheme_name in {"central", "linear"}:
            flux_cf = (1.0 - weights) * mdot_b
            flux_ff = weights * mdot_b
            flux_vf = np.zeros_like(mdot_b)
        else:
            phi_upwind = np.where(mdot_b >= 0.0, phi[owners_b], phi[neighbours_b])
            phi_linear = (1.0 - weights) * phi[owners_b] + weights * phi[neighbours_b]
            if scheme_name == "deferred":
                psi = np.ones_like(mdot_b)
            elif scheme_name.lower() == "lust":
                psi = np.full_like(mdot_b, 0.75)
            elif scheme_name.lower() == "linearupwind":
                # The translated-stencil gradient correction is not built for
                # cyclic pairs; a central correction is the closest 2nd-order
                # target on these faces.
                psi = np.ones_like(mdot_b)
            elif is_limited_scheme(scheme_name):
                psi = _tvd_face_psi(
                    phi,
                    mdot_b,
                    grad_phi,
                    owners_b,
                    neighbours_b,
                    geo_data["face_cf_vector"][b_face_indices],
                    scheme_name,
                )
            else:
                raise ValueError(f"Unknown scheme: {scheme}")
            flux_cf = flux_cf_upwind
            flux_ff = flux_ff_upwind
            flux_vf = psi * mdot_b * (phi_linear - phi_upwind)
        result = {
            "flux_cf": flux_cf,
            "flux_ff": flux_ff,
            "flux_vf": flux_vf,
            "face_indices": b_face_indices,
        }
        if include_total_flux:
            result["flux_tf"] = flux_cf * phi[owners_b] + flux_ff * phi[neighbours_b] + flux_vf
        return result

    # Boundary element indices
    b_elem_start = start_face - n_interior_faces
    b_elem_indices = np.arange(n_elements + b_elem_start, n_elements + b_elem_start + n_faces)

    if strategy in (
        BoundaryStrategy.FIXED_VALUE,
        BoundaryStrategy.NO_SLIP,
    ):
        # A Dirichlet face value is authoritative for either flow direction.
        flux_cf = np.zeros_like(mdot_b)
        flux_ff_val = mdot_b
    else:
        # Extrapolating conditions use the owner on outflow and the boundary
        # ghost value on reverse flow (the inletOutlet contract).
        flux_cf = np.maximum(mdot_b, 0.0)
        flux_ff_val = np.minimum(mdot_b, 0.0)

    # Set flux_ff to 0 for matrix assembly (no neighbor column for boundary)
    flux_ff = np.zeros_like(mdot_b)

    # Explicit correction: contribution from known boundary value
    # Equation: flux_cf * phi_c + flux_ff_val * phi_b = source
    # Matrix: A[c,c] * phi_c = ... - flux_ff_val * phi_b
    # RHS assembly does: b -= flux_vf
    # So we need flux_vf = flux_ff_val * phi_b

    phi_c = phi[owners_b]
    phi_b = phi[b_elem_indices]

    flux_vf = flux_ff_val * phi_b

    result = {
        "flux_cf": flux_cf,
        "flux_ff": flux_ff,
        "flux_vf": flux_vf,
        "face_indices": b_face_indices,
    }
    if include_total_flux:
        result["flux_tf"] = flux_cf * phi_c + flux_vf
    return result


def assemble_convection_term_gradient_upwind(
    phi,
    mdot,
    mesh_data,
    geo_data,
    grad_phi,
    linear_blend,
    *,
    include_total_flux=True,
):
    """Linear-upwind/LUST interpolation in deferred-correction form.

    The face target is ``linear_blend * φ_linear + (1 - linear_blend) *
    φ_linearUpwind`` with ``φ_linearUpwind = φ_up + (∇φ)_up · (x_f - x_up)``
    — a second-order upwind-biased value; ``linear_blend = 0.75`` gives the
    standard LUST blend.

    The implicit part stays first-order upwind (bounded, diagonally
    dominant); the explicit correction carries the difference to the target.
    Blending toward first-order upwind instead (a constant ψ on the
    central-vs-upwind correction) adds ``(1-blend)·|U|h/2`` of numerical
    viscosity, which in a bluff-body wake is several times the physical ν
    and suppresses vortex shedding entirely.
    """
    if grad_phi is None:
        raise ValueError("linearUpwind/LUST convection requires grad_phi (cell gradient)")
    if grad_phi.ndim == 3 and grad_phi.shape[2] == 1:
        grad_phi = grad_phi.squeeze(-1)

    n_interior_faces = mesh_data["n_interior_faces"]
    owners_all = mesh_data["owners"]
    neighbours_all = mesh_data["neighbours"]
    flux_cf = np.empty(n_interior_faces, dtype=np.float64)
    flux_ff = np.empty(n_interior_faces, dtype=np.float64)
    flux_vf = np.empty(n_interior_faces, dtype=np.float64)
    flux_tf = np.empty(n_interior_faces, dtype=np.float64) if include_total_flux else None

    chunk_size = 250_000
    for start in range(0, n_interior_faces, chunk_size):
        stop = min(start + chunk_size, n_interior_faces)
        face_slice = slice(start, stop)
        owners = owners_all[face_slice]
        neighbours = neighbours_all[face_slice]
        mdot_i = mdot[face_slice]
        weights = geo_data["face_weights"][face_slice]
        upwind = np.where(mdot_i >= 0.0, owners, neighbours)
        to_face = geo_data["face_centroids"][face_slice] - geo_data["element_centroids"][upwind]
        phi_upwind = phi[upwind]
        phi_linear_upwind = phi_upwind + np.sum(grad_phi[upwind] * to_face, axis=1)
        phi_linear = weights * phi[neighbours] + (1.0 - weights) * phi[owners]
        phi_target = linear_blend * phi_linear + (1.0 - linear_blend) * phi_linear_upwind
        cf = np.maximum(mdot_i, 0.0)
        ff = np.minimum(mdot_i, 0.0)
        vf = mdot_i * (phi_target - phi_upwind)
        flux_cf[face_slice] = cf
        flux_ff[face_slice] = ff
        flux_vf[face_slice] = vf
        if flux_tf is not None:
            flux_tf[face_slice] = cf * phi[owners] + ff * phi[neighbours] + vf
    result = {"flux_cf": flux_cf, "flux_ff": flux_ff, "flux_vf": flux_vf}
    if flux_tf is not None:
        result["flux_tf"] = flux_tf
    return result


def assemble_convection_term(
    phi,
    mdot,
    mesh_data,
    geo_data,
    boundaries,
    scheme="deferred",
    grad_phi=None,
    *,
    include_total_flux=True,
):
    """
    Assemble complete convection term.

    Args:
        phi: Field values
        mdot: Mass flow rate through faces
        mesh_data: Mesh connectivity
        geo_data: Geometric data
        boundaries: Boundary patch list
        scheme: Convection scheme. First-order: ``'upwind'``.  Second-order
            (unbounded, energy-conserving): ``'central'`` / ``'linear'``,
            ``'deferred'`` (= central via deferred correction).  Blended:
            ``'LUST'`` (0.75 linear + 0.25 upwind).  Bounded high-resolution
            TVD (require ``grad_phi``): ``'limitedLinear'``, ``'vanLeer'``,
            ``'MUSCL'``, ``'minmod'``, ``'superbee'``.
        grad_phi: Cell gradient of ``phi`` (n_total, 3), required by the TVD
            schemes; ignored by the others.

    Returns:
        dict: Complete flux data
    """

    n_faces = mesh_data["n_faces"]
    n_interior = mesh_data["n_interior_faces"]

    # Initialize
    flux_cf = np.zeros(n_faces)
    flux_ff = np.zeros(n_faces)
    flux_vf = np.zeros(n_faces)
    flux_tf = np.zeros(n_faces) if include_total_flux else None

    # Interior faces
    s = str(scheme)
    if s == "upwind":
        interior_fluxes = assemble_convection_term_upwind(
            phi, mdot, mesh_data, include_total_flux=include_total_flux
        )
    elif s in ("central", "linear"):
        interior_fluxes = assemble_convection_term_central(
            phi, mdot, mesh_data, geo_data, include_total_flux=include_total_flux
        )
    elif s == "deferred":
        interior_fluxes = assemble_convection_term_deferred_correction(
            phi, mdot, mesh_data, geo_data, include_total_flux=include_total_flux
        )
    elif s in ("LUST", "lust"):
        # LUST: 0.75 linear + 0.25 linearUpwind.
        # (second-order, gradient-corrected — NOT first-order upwind).
        interior_fluxes = assemble_convection_term_gradient_upwind(
            phi,
            mdot,
            mesh_data,
            geo_data,
            grad_phi,
            linear_blend=0.75,
            include_total_flux=include_total_flux,
        )
    elif s in ("linearUpwind", "linearupwind"):
        # Linear-upwind: second-order upwind-biased interpolation.
        interior_fluxes = assemble_convection_term_gradient_upwind(
            phi,
            mdot,
            mesh_data,
            geo_data,
            grad_phi,
            linear_blend=0.0,
            include_total_flux=include_total_flux,
        )
    elif is_limited_scheme(s):
        interior_fluxes = assemble_convection_term_limited(
            phi,
            mdot,
            mesh_data,
            geo_data,
            grad_phi,
            limiter=s,
            include_total_flux=include_total_flux,
        )
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    flux_cf[:n_interior] = interior_fluxes["flux_cf"]
    flux_ff[:n_interior] = interior_fluxes["flux_ff"]
    flux_vf[:n_interior] = interior_fluxes["flux_vf"]
    if flux_tf is not None:
        flux_tf[:n_interior] = interior_fluxes["flux_tf"]

    # Boundary faces
    for boundary in boundaries:
        strategy = _convection_boundary_strategy(boundary)
        if strategy in (
            BoundaryStrategy.EMPTY,
            BoundaryStrategy.SLIP,
            BoundaryStrategy.SYMMETRY,
        ):
            # No convective flux through an impermeable plane.
            continue

        b_fluxes = assemble_convection_term_boundary(
            phi,
            mdot,
            boundary,
            mesh_data,
            geo_data,
            scheme=scheme,
            grad_phi=grad_phi,
            include_total_flux=include_total_flux,
        )

        indices = b_fluxes["face_indices"]
        flux_cf[indices] = b_fluxes["flux_cf"]
        flux_ff[indices] = b_fluxes["flux_ff"]
        flux_vf[indices] = b_fluxes["flux_vf"]
        if flux_tf is not None:
            flux_tf[indices] = b_fluxes["flux_tf"]

    result = {"flux_cf": flux_cf, "flux_ff": flux_ff, "flux_vf": flux_vf}
    if flux_tf is not None:
        result["flux_tf"] = flux_tf
    return result


def compute_volumetric_face_flux(velocity, mesh_data, geo_data):
    """Compute volumetric flux through faces: ``phi = U · Sf``.

    Density is deliberately absent; callers that need mass flux must multiply
    ``phi`` by face-interpolated density.

    Args:
        velocity: Cell and boundary-ghost velocity [m/s], shape
            ``(n_cells_with_ghosts, 3)``.
        mesh_data: Mesh connectivity
        geo_data: Geometric data

    Returns:
        numpy.ndarray: Volumetric face flux [m³/s], shape ``(n_faces,)``;
        positive from owner to neighbour on interior faces.
    """

    n_faces = mesh_data["n_faces"]
    n_interior = mesh_data["n_interior_faces"]

    owners = mesh_data["owners"]
    neighbours = mesh_data["neighbours"]
    face_sf = geo_data["face_sf"]
    face_weights = geo_data["face_weights"]

    phi = np.zeros(n_faces)

    # Interpolate/dot in blocks instead of retaining a face-vector field.
    chunk_size = 250_000
    for start in range(0, n_interior, chunk_size):
        stop = min(start + chunk_size, n_interior)
        face_slice = slice(start, stop)
        w = face_weights[face_slice, np.newaxis]
        u_face = w * velocity[neighbours[face_slice]] + (1.0 - w) * velocity[owners[face_slice]]
        phi[face_slice] = np.sum(u_face * face_sf[face_slice], axis=1)

    # Boundary faces: use boundary velocity
    n_elements = mesh_data["n_elements"]

    # Vectorized boundary processing
    b_face_indices = np.arange(n_interior, n_faces)
    b_elem_indices = n_elements + (b_face_indices - n_interior)

    u_face_b = velocity[b_elem_indices]
    phi[n_interior:] = np.sum(u_face_b * face_sf[n_interior:], axis=1)

    boundary_neighbours = np.asarray(
        mesh_data.get("boundary_neighbours", np.full(n_faces, -1, dtype=np.int32))
    )
    coupled = np.flatnonzero(boundary_neighbours >= 0)
    if coupled.size:
        weights_b = face_weights[coupled, np.newaxis]
        u_face_coupled = (
            weights_b * velocity[boundary_neighbours[coupled]]
            + (1.0 - weights_b) * velocity[owners[coupled]]
        )
        phi[coupled] = np.sum(u_face_coupled * face_sf[coupled], axis=1)

    return phi
