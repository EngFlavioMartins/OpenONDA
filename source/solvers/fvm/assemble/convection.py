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
    type_u = boundary_patch.get("velocity_type")
    if type_u is not None:
        return BOUNDARIES.strategy(type_u, "velocity", "convection")
    boundary_condition_type = boundary_patch.get("boundary_condition_type")
    return BOUNDARIES.strategy(boundary_condition_type, "scalar", "convection")


def assemble_convection_term_upwind(
    scalar_field, advective_face_flux, mesh_data, *, include_total_flux=True
):
    """
    Assemble convection term using upwind scheme.

    Upwind: φ_f = φ_upwind (first-order, stable)

    Args:
        scalar_field: Field values (n_elements + n_boundary,)
        advective_face_flux: Mass flow rate through faces (n_faces,)
        mesh_data: Mesh connectivity

    Returns:
        dict: Flux coefficients
    """

    n_interior_faces = mesh_data["n_interior_faces"]
    owners = mesh_data["owners"][:n_interior_faces]
    neighbours = mesh_data["neighbours"][:n_interior_faces]

    interior_advective_face_flux = advective_face_flux[:n_interior_faces]

    # Upwind scheme: use upstream value
    # If advective_face_flux > 0: flow from owner to neighbour, use owner value
    # If advective_face_flux < 0: flow from neighbour to owner, use neighbour value

    flux_cf = np.maximum(interior_advective_face_flux, 0.0)  # Owner contribution
    flux_ff = np.minimum(interior_advective_face_flux, 0.0)  # Neighbour contribution

    # No explicit correction for upwind
    flux_vf = np.zeros_like(interior_advective_face_flux)

    # Total flux
    result = {"flux_cf": flux_cf, "flux_ff": flux_ff, "flux_vf": flux_vf}
    if include_total_flux:
        result["flux_tf"] = flux_cf * scalar_field[owners] + flux_ff * scalar_field[neighbours]
    return result


def assemble_convection_term_central(
    scalar_field, advective_face_flux, mesh_data, geo_data, *, include_total_flux=True
):
    """
    Assemble convection term using central differencing.

    Central: φ_f = w*φ_neighbour + (1-w)*φ_owner (second-order)

    Args:
        scalar_field: Field values
        advective_face_flux: Mass flow rate through faces
        mesh_data: Mesh connectivity
        geo_data: Geometric data (for weights)

    Returns:
        dict: Flux coefficients
    """

    n_interior_faces = mesh_data["n_interior_faces"]
    owners = mesh_data["owners"][:n_interior_faces]
    neighbours = mesh_data["neighbours"][:n_interior_faces]

    interior_advective_face_flux = advective_face_flux[:n_interior_faces]
    weights = geo_data["face_interpolation_weight"][:n_interior_faces]

    # Central differencing coefficients
    flux_cf = (1 - weights) * interior_advective_face_flux
    flux_ff = weights * interior_advective_face_flux

    # No explicit correction for pure central
    flux_vf = np.zeros_like(interior_advective_face_flux)

    # Total flux
    result = {"flux_cf": flux_cf, "flux_ff": flux_ff, "flux_vf": flux_vf}
    if include_total_flux:
        result["flux_tf"] = flux_cf * scalar_field[owners] + flux_ff * scalar_field[neighbours]
    return result


def assemble_convection_term_deferred_correction(
    scalar_field, advective_face_flux, mesh_data, geo_data, *, include_total_flux=True
):
    """
    Assemble convection term using deferred correction.

    Approach:
    1. Use upwind for implicit part (stable)
    2. Add correction to central as explicit term (accurate)

    This gives stability of upwind with accuracy approaching central.

    Args:
        scalar_field: Field values
        advective_face_flux: Mass flow rate
        mesh_data: Mesh connectivity
        geo_data: Geometric data

    Returns:
        dict: Flux coefficients
    """

    n_interior_faces = mesh_data["n_interior_faces"]
    owners = mesh_data["owners"][:n_interior_faces]
    neighbours = mesh_data["neighbours"][:n_interior_faces]

    interior_advective_face_flux = advective_face_flux[:n_interior_faces]
    weights = geo_data["face_interpolation_weight"][:n_interior_faces]

    # Upwind part (implicit)
    flux_cf_upwind = np.maximum(interior_advective_face_flux, 0.0)
    flux_ff_upwind = np.minimum(interior_advective_face_flux, 0.0)

    # Central part
    flux_cf_central = (1 - weights) * interior_advective_face_flux
    flux_ff_central = weights * interior_advective_face_flux

    # Deferred correction: explicit term = central - upwind
    flux_vf = (flux_cf_central - flux_cf_upwind) * scalar_field[owners] + (
        flux_ff_central - flux_ff_upwind
    ) * scalar_field[neighbours]

    # Use upwind for implicit coefficients
    flux_cf = flux_cf_upwind
    flux_ff = flux_ff_upwind

    # Total flux
    result = {"flux_cf": flux_cf, "flux_ff": flux_ff, "flux_vf": flux_vf}
    if include_total_flux:
        result["flux_tf"] = (
            flux_cf * scalar_field[owners] + flux_ff * scalar_field[neighbours] + flux_vf
        )
    return result


def _tvd_face_psi(
    scalar_field,
    interior_advective_face_flux,
    scalar_field_gradient,
    owners,
    neighbours,
    cf_vector,
    limiter,
):
    """Per-face standard TVD limiter ψ ∈ [0, 2].

    r = 2 (d · ∇φ_upwind) / (φ_N − φ_P) − 1,  d = c_N − c_C (owner→neighbour).
    Extrema (φ_N ≈ φ_P) are handled by saturating r so the limiter → upwind.
    """
    if scalar_field_gradient is None:
        raise ValueError(
            "Limited convection schemes require scalar_field_gradient (cell gradient)."
        )
    if scalar_field_gradient.ndim == 3 and scalar_field_gradient.shape[2] == 1:
        scalar_field_gradient = scalar_field_gradient.squeeze(-1)
    if not np.all(np.isfinite(scalar_field)) or not np.all(np.isfinite(scalar_field_gradient)):
        raise FloatingPointError("TVD convection received a non-finite field or gradient")

    scalar_field_p = scalar_field[owners]
    scalar_field_n = scalar_field[neighbours]
    gradf = scalar_field_n - scalar_field_p

    grad_cp = np.sum(scalar_field_gradient[owners] * cf_vector, axis=1)
    grad_cn = np.sum(scalar_field_gradient[neighbours] * cf_vector, axis=1)
    grad_cf = np.where(interior_advective_face_flux >= 0.0, grad_cp, grad_cn)

    small = np.abs(gradf) < 1e-30 * np.maximum(np.abs(grad_cf), 1.0)
    r = np.full_like(gradf, -1.0, dtype=np.float64)
    np.divide(2.0 * grad_cf, gradf, out=r, where=~small)
    r[~small] -= 1.0
    # At an extremum (gradf→0) force upwind (r large negative → ψ=0) unless the
    # upwind gradient agrees in sign (smooth plateau → keep some blend).
    r = np.where(small, np.where(np.sign(grad_cf) == np.sign(gradf), 1000.0, -1.0), r)
    return apply_limiter(limiter, r)


def assemble_convection_term_limited(
    scalar_field,
    advective_face_flux,
    mesh_data,
    geo_data,
    scalar_field_gradient,
    limiter,
    psi=None,
    *,
    include_total_flux=True,
):
    """High-resolution TVD convection in deferred-correction form.

    Implicit part = upwind (bounded, diagonally dominant); explicit correction =
    ``ψ · advective_face_flux · (φ_linear − φ_upwind)`` where ψ comes from the TVD limiter (or a
    constant ``psi`` for blended schemes such as LUST).  ψ = 1 ⇒ pure central,
    ψ = 0 ⇒ upwind.
    """
    n_interior_faces = mesh_data["n_interior_faces"]
    owners = mesh_data["owners"][:n_interior_faces]
    neighbours = mesh_data["neighbours"][:n_interior_faces]
    interior_advective_face_flux = advective_face_flux[:n_interior_faces]
    weights = geo_data["face_interpolation_weight"][:n_interior_faces]
    cf_vector = geo_data["cell_connection_vector"][:n_interior_faces]

    if psi is None:
        psi = _tvd_face_psi(
            scalar_field,
            interior_advective_face_flux,
            scalar_field_gradient,
            owners,
            neighbours,
            cf_vector,
            limiter,
        )
    else:
        psi = np.full_like(interior_advective_face_flux, float(psi))

    # Implicit upwind coefficients.
    flux_cf = np.maximum(interior_advective_face_flux, 0.0)
    flux_ff = np.minimum(interior_advective_face_flux, 0.0)

    # Deferred high-resolution correction: ψ·advective_face_flux·(φ_linear − φ_upwind).
    scalar_field_upwind = np.where(
        interior_advective_face_flux >= 0.0, scalar_field[owners], scalar_field[neighbours]
    )
    scalar_field_linear = (
        weights * scalar_field[neighbours] + (1.0 - weights) * scalar_field[owners]
    )
    flux_vf = psi * interior_advective_face_flux * (scalar_field_linear - scalar_field_upwind)

    result = {"flux_cf": flux_cf, "flux_ff": flux_ff, "flux_vf": flux_vf}
    if include_total_flux:
        result["flux_tf"] = (
            flux_cf * scalar_field[owners] + flux_ff * scalar_field[neighbours] + flux_vf
        )
    return result


def assemble_convection_term_boundary(
    scalar_field,
    advective_face_flux,
    boundary_patch,
    mesh_data,
    geo_data=None,
    scheme="upwind",
    scalar_field_gradient=None,
    *,
    include_total_flux=True,
):
    """
    Assemble convection term for boundary faces (1st-order upwind).

    Physical boundary values follow their mathematical condition directly.
    Cyclic faces use the selected interior scheme and a true paired-cell column.

    Args:
        scalar_field: Field values including boundary elements
        advective_face_flux: Mass flow rate
        boundary_patch: Boundary patch info
        mesh_data: Mesh connectivity

    Returns:
        dict: Flux coefficients for boundary
    """

    n_interior_faces = mesh_data["n_interior_faces"]
    n_cells = mesh_data["n_cells"]

    start_face = boundary_patch["start_face"]
    n_faces = boundary_patch["n_faces"]
    end_face = start_face + n_faces

    b_face_indices = np.arange(start_face, end_face)
    owners_b = mesh_data["owners"][b_face_indices]

    boundary_advective_face_flux = advective_face_flux[b_face_indices]

    strategy = _convection_boundary_strategy(boundary_patch)
    if strategy is BoundaryStrategy.CYCLIC:
        if geo_data is None:
            raise ValueError("Cyclic convection requires geometric interpolation data")
        neighbours_b = mesh_data["boundary_neighbour_cell"][b_face_indices]
        if np.any(neighbours_b < 0):
            raise ValueError("Cyclic boundary faces are missing paired owner cells")
        weights = geo_data["face_interpolation_weight"][b_face_indices]
        scheme_name = str(scheme)
        flux_cf_upwind = np.maximum(boundary_advective_face_flux, 0.0)
        flux_ff_upwind = np.minimum(boundary_advective_face_flux, 0.0)
        if scheme_name == "upwind":
            flux_cf = flux_cf_upwind
            flux_ff = flux_ff_upwind
            flux_vf = np.zeros_like(boundary_advective_face_flux)
        elif scheme_name in {"central", "linear"}:
            flux_cf = (1.0 - weights) * boundary_advective_face_flux
            flux_ff = weights * boundary_advective_face_flux
            flux_vf = np.zeros_like(boundary_advective_face_flux)
        else:
            scalar_field_upwind = np.where(
                boundary_advective_face_flux >= 0.0,
                scalar_field[owners_b],
                scalar_field[neighbours_b],
            )
            scalar_field_linear = (1.0 - weights) * scalar_field[owners_b] + weights * scalar_field[
                neighbours_b
            ]
            if scheme_name == "deferred":
                psi = np.ones_like(boundary_advective_face_flux)
            elif scheme_name.lower() == "lust":
                psi = np.full_like(boundary_advective_face_flux, 0.75)
            elif scheme_name.lower() == "linearupwind":
                # The translated-stencil gradient correction is not built for
                # cyclic pairs; a central correction is the closest 2nd-order
                # target on these faces.
                psi = np.ones_like(boundary_advective_face_flux)
            elif is_limited_scheme(scheme_name):
                psi = _tvd_face_psi(
                    scalar_field,
                    boundary_advective_face_flux,
                    scalar_field_gradient,
                    owners_b,
                    neighbours_b,
                    geo_data["cell_connection_vector"][b_face_indices],
                    scheme_name,
                )
            else:
                raise ValueError(f"Unknown scheme: {scheme}")
            flux_cf = flux_cf_upwind
            flux_ff = flux_ff_upwind
            flux_vf = (
                psi * boundary_advective_face_flux * (scalar_field_linear - scalar_field_upwind)
            )
        result = {
            "flux_cf": flux_cf,
            "flux_ff": flux_ff,
            "flux_vf": flux_vf,
            "face_indices": b_face_indices,
        }
        if include_total_flux:
            result["flux_tf"] = (
                flux_cf * scalar_field[owners_b] + flux_ff * scalar_field[neighbours_b] + flux_vf
            )
        return result

    # Boundary element indices
    b_elem_start = start_face - n_interior_faces
    b_elem_indices = np.arange(n_cells + b_elem_start, n_cells + b_elem_start + n_faces)

    if strategy in (
        BoundaryStrategy.FIXED_VALUE,
        BoundaryStrategy.NO_SLIP,
        BoundaryStrategy.NORMAL_VALUE_TANGENTIAL_GRADIENT,
    ):
        # A Dirichlet face value is authoritative for either flow direction.
        flux_cf = np.zeros_like(boundary_advective_face_flux)
        flux_ff_val = boundary_advective_face_flux
    else:
        # Extrapolating conditions use the owner on outflow and the boundary
        # ghost value on reverse flow (the inletOutlet contract).
        flux_cf = np.maximum(boundary_advective_face_flux, 0.0)
        flux_ff_val = np.minimum(boundary_advective_face_flux, 0.0)

    # Set flux_ff to 0 for matrix assembly (no neighbour column for boundary)
    flux_ff = np.zeros_like(boundary_advective_face_flux)

    # Explicit correction: contribution from known boundary value
    # Equation: flux_cf * scalar_field_c + flux_ff_val * scalar_field_b = source
    # Matrix: A[c,c] * scalar_field_c = ... - flux_ff_val * scalar_field_b
    # RHS assembly does: b -= flux_vf
    # So we need flux_vf = flux_ff_val * scalar_field_b

    scalar_field_c = scalar_field[owners_b]
    scalar_field_b = scalar_field[b_elem_indices]

    flux_vf = flux_ff_val * scalar_field_b

    result = {
        "flux_cf": flux_cf,
        "flux_ff": flux_ff,
        "flux_vf": flux_vf,
        "face_indices": b_face_indices,
    }
    if include_total_flux:
        result["flux_tf"] = flux_cf * scalar_field_c + flux_vf
    return result


def assemble_convection_term_gradient_upwind(
    scalar_field,
    advective_face_flux,
    mesh_data,
    geo_data,
    scalar_field_gradient,
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
    central-vs-upwind correction) adds ``(1-blend)·|velocity|h/2`` of numerical
    viscosity, which in a bluff-body wake is several times the physical ν
    and suppresses vortex shedding entirely.
    """
    if scalar_field_gradient is None:
        raise ValueError(
            "linearUpwind/LUST convection requires scalar_field_gradient (cell gradient)"
        )
    if scalar_field_gradient.ndim == 3 and scalar_field_gradient.shape[2] == 1:
        scalar_field_gradient = scalar_field_gradient.squeeze(-1)

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
        interior_advective_face_flux = advective_face_flux[face_slice]
        weights = geo_data["face_interpolation_weight"][face_slice]
        upwind = np.where(interior_advective_face_flux >= 0.0, owners, neighbours)
        to_face = geo_data["face_centre"][face_slice] - geo_data["cell_centre"][upwind]
        scalar_field_upwind = scalar_field[upwind]
        scalar_field_linear_upwind = scalar_field_upwind + np.sum(
            scalar_field_gradient[upwind] * to_face, axis=1
        )
        scalar_field_linear = (
            weights * scalar_field[neighbours] + (1.0 - weights) * scalar_field[owners]
        )
        scalar_field_target = (
            linear_blend * scalar_field_linear + (1.0 - linear_blend) * scalar_field_linear_upwind
        )
        cf = np.maximum(interior_advective_face_flux, 0.0)
        ff = np.minimum(interior_advective_face_flux, 0.0)
        vf = interior_advective_face_flux * (scalar_field_target - scalar_field_upwind)
        flux_cf[face_slice] = cf
        flux_ff[face_slice] = ff
        flux_vf[face_slice] = vf
        if flux_tf is not None:
            flux_tf[face_slice] = cf * scalar_field[owners] + ff * scalar_field[neighbours] + vf
    result = {"flux_cf": flux_cf, "flux_ff": flux_ff, "flux_vf": flux_vf}
    if flux_tf is not None:
        result["flux_tf"] = flux_tf
    return result


def assemble_convection_term(
    scalar_field,
    advective_face_flux,
    mesh_data,
    geo_data,
    boundaries,
    scheme="deferred",
    scalar_field_gradient=None,
    *,
    include_total_flux=True,
):
    """
    Assemble complete convection term.

    Args:
        scalar_field: Field values
        advective_face_flux: Mass flow rate through faces
        mesh_data: Mesh connectivity
        geo_data: Geometric data
        boundaries: Boundary patch list
        scheme: Convection scheme. First-order: ``'upwind'``.  Second-order
            (unbounded, energy-conserving): ``'central'`` / ``'linear'``,
            ``'deferred'`` (= central via deferred correction).  Blended:
            ``'LUST'`` (0.75 linear + 0.25 upwind).  Bounded high-resolution
            TVD (require ``scalar_field_gradient``): ``'limitedLinear'``, ``'vanLeer'``,
            ``'MUSCL'``, ``'minmod'``, ``'superbee'``.
        scalar_field_gradient: Cell gradient of ``scalar_field`` (n_total, 3), required by the TVD
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
            scalar_field, advective_face_flux, mesh_data, include_total_flux=include_total_flux
        )
    elif s in ("central", "linear"):
        interior_fluxes = assemble_convection_term_central(
            scalar_field,
            advective_face_flux,
            mesh_data,
            geo_data,
            include_total_flux=include_total_flux,
        )
    elif s == "deferred":
        interior_fluxes = assemble_convection_term_deferred_correction(
            scalar_field,
            advective_face_flux,
            mesh_data,
            geo_data,
            include_total_flux=include_total_flux,
        )
    elif s in ("LUST", "lust"):
        # LUST: 0.75 linear + 0.25 linearUpwind.
        # (second-order, gradient-corrected — NOT first-order upwind).
        interior_fluxes = assemble_convection_term_gradient_upwind(
            scalar_field,
            advective_face_flux,
            mesh_data,
            geo_data,
            scalar_field_gradient,
            linear_blend=0.75,
            include_total_flux=include_total_flux,
        )
    elif s in ("linearUpwind", "linearupwind"):
        # Linear-upwind: second-order upwind-biased interpolation.
        interior_fluxes = assemble_convection_term_gradient_upwind(
            scalar_field,
            advective_face_flux,
            mesh_data,
            geo_data,
            scalar_field_gradient,
            linear_blend=0.0,
            include_total_flux=include_total_flux,
        )
    elif is_limited_scheme(s):
        interior_fluxes = assemble_convection_term_limited(
            scalar_field,
            advective_face_flux,
            mesh_data,
            geo_data,
            scalar_field_gradient,
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
            scalar_field,
            advective_face_flux,
            boundary,
            mesh_data,
            geo_data,
            scheme=scheme,
            scalar_field_gradient=scalar_field_gradient,
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
    """Compute volumetric face flux through faces: ``volumetric_face_flux = velocity · Sf``.

    Density is deliberately absent; callers that need mass flux must multiply
    ``volumetric_face_flux`` by face-interpolated density.

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
    face_area_vector = geo_data["face_area_vector"]
    face_interpolation_weight = geo_data["face_interpolation_weight"]

    volumetric_face_flux = np.zeros(n_faces)

    # Interpolate/dot in blocks instead of retaining a face-vector field.
    chunk_size = 250_000
    for start in range(0, n_interior, chunk_size):
        stop = min(start + chunk_size, n_interior)
        face_slice = slice(start, stop)
        w = face_interpolation_weight[face_slice, np.newaxis]
        u_face = w * velocity[neighbours[face_slice]] + (1.0 - w) * velocity[owners[face_slice]]
        volumetric_face_flux[face_slice] = np.sum(u_face * face_area_vector[face_slice], axis=1)

    # Boundary faces: use boundary velocity
    n_cells = mesh_data["n_cells"]

    # Vectorized boundary processing
    b_face_indices = np.arange(n_interior, n_faces)
    b_elem_indices = n_cells + (b_face_indices - n_interior)

    u_face_b = velocity[b_elem_indices]
    volumetric_face_flux[n_interior:] = np.sum(u_face_b * face_area_vector[n_interior:], axis=1)

    boundary_neighbour_cell = np.asarray(
        mesh_data.get("boundary_neighbour_cell", np.full(n_faces, -1, dtype=np.int32))
    )
    coupled = np.flatnonzero(boundary_neighbour_cell >= 0)
    if coupled.size:
        weights_b = face_interpolation_weight[coupled, np.newaxis]
        u_face_coupled = (
            weights_b * velocity[boundary_neighbour_cell[coupled]]
            + (1.0 - weights_b) * velocity[owners[coupled]]
        )
        volumetric_face_flux[coupled] = np.sum(u_face_coupled * face_area_vector[coupled], axis=1)

    return volumetric_face_flux
