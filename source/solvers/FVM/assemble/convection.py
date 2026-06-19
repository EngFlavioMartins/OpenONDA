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


def assemble_convection_term_upwind(phi, mdot, mesh_data):
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
    flux_tf = flux_cf * phi[owners] + flux_ff * phi[neighbours]

    return {"flux_cf": flux_cf, "flux_ff": flux_ff, "flux_vf": flux_vf, "flux_tf": flux_tf}


def assemble_convection_term_central(phi, mdot, mesh_data, geo_data):
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
    flux_tf = flux_cf * phi[owners] + flux_ff * phi[neighbours]

    return {"flux_cf": flux_cf, "flux_ff": flux_ff, "flux_vf": flux_vf, "flux_tf": flux_tf}


def assemble_convection_term_deferred_correction(phi, mdot, mesh_data, geo_data):
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
    flux_tf = flux_cf * phi[owners] + flux_ff * phi[neighbours] + flux_vf

    return {"flux_cf": flux_cf, "flux_ff": flux_ff, "flux_vf": flux_vf, "flux_tf": flux_tf}


def assemble_convection_term_boundary(phi, mdot, boundary_patch, mesh_data):
    """
    Assemble convection term for boundary faces.

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

    # Boundary element indices
    b_elem_start = start_face - n_interior_faces
    b_elem_indices = np.arange(n_elements + b_elem_start, n_elements + b_elem_start + n_faces)

    # For boundary: use upwind
    # Outflow (mdot > 0): use owner value
    # Inflow (mdot < 0): use boundary value
    flux_cf = np.maximum(mdot_b, 0.0)

    # Neighbor contribution (inflow)
    # This is a known value, so it goes to RHS
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

    flux_tf = flux_cf * phi_c + flux_vf

    return {
        "flux_cf": flux_cf,
        "flux_ff": flux_ff,
        "flux_vf": flux_vf,
        "flux_tf": flux_tf,
        "face_indices": b_face_indices,
    }


def assemble_convection_term(phi, mdot, mesh_data, geo_data, boundaries, scheme="deferred"):
    """
    Assemble complete convection term.

    Args:
        phi: Field values
        mdot: Mass flow rate through faces
        mesh_data: Mesh connectivity
        geo_data: Geometric data
        boundaries: Boundary patch list
        scheme: 'upwind', 'central', or 'deferred'

    Returns:
        dict: Complete flux data
    """

    n_faces = mesh_data["n_faces"]
    n_interior = mesh_data["n_interior_faces"]

    # Initialize
    flux_cf = np.zeros(n_faces)
    flux_ff = np.zeros(n_faces)
    flux_vf = np.zeros(n_faces)
    flux_tf = np.zeros(n_faces)

    # Interior faces
    if scheme == "upwind":
        interior_fluxes = assemble_convection_term_upwind(phi, mdot, mesh_data)
    elif scheme == "central":
        interior_fluxes = assemble_convection_term_central(phi, mdot, mesh_data, geo_data)
    elif scheme == "deferred":
        interior_fluxes = assemble_convection_term_deferred_correction(
            phi, mdot, mesh_data, geo_data
        )
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    flux_cf[:n_interior] = interior_fluxes["flux_cf"]
    flux_ff[:n_interior] = interior_fluxes["flux_ff"]
    flux_vf[:n_interior] = interior_fluxes["flux_vf"]
    flux_tf[:n_interior] = interior_fluxes["flux_tf"]

    # Boundary faces
    for boundary in boundaries:
        if boundary.get("bc_type_U") == "empty" or boundary.get("type") == "empty":
            continue

        b_fluxes = assemble_convection_term_boundary(phi, mdot, boundary, mesh_data)

        indices = b_fluxes["face_indices"]
        flux_cf[indices] = b_fluxes["flux_cf"]
        flux_ff[indices] = b_fluxes["flux_ff"]
        flux_vf[indices] = b_fluxes["flux_vf"]
        flux_tf[indices] = b_fluxes["flux_tf"]

    return {"flux_cf": flux_cf, "flux_ff": flux_ff, "flux_vf": flux_vf, "flux_tf": flux_tf}


def compute_mass_flow_rate(velocity, mesh_data, geo_data):
    """
    Compute mass flow rate through faces: mdot = rho * U · Sf

    For incompressible flow: mdot = U · Sf

    Args:
        velocity: Velocity field (n_elements + n_boundary, 3)
        mesh_data: Mesh connectivity
        geo_data: Geometric data

    Returns:
        numpy.ndarray: Mass flow rate (n_faces,)
    """

    n_faces = mesh_data["n_faces"]
    n_interior = mesh_data["n_interior_faces"]

    owners = mesh_data["owners"]
    neighbours = mesh_data["neighbours"]
    face_sf = geo_data["face_sf"]
    face_weights = geo_data["face_weights"]

    mdot = np.zeros(n_faces)

    # Interior faces: interpolate velocity to face
    # Interior faces: interpolate velocity to face
    # u_face = w * U[nei] + (1-w) * U[own]
    u_face = (
        face_weights[:n_interior, np.newaxis] * velocity[neighbours[:n_interior]]
        + (1.0 - face_weights[:n_interior, np.newaxis]) * velocity[owners[:n_interior]]
    )

    # mdot = U_face · Sf
    # Dot product along axis 1 (N, 3) * (N, 3) -> (N,)
    mdot[:n_interior] = np.sum(u_face * face_sf[:n_interior], axis=1)

    # Boundary faces: use boundary velocity
    n_elements = mesh_data["n_elements"]

    # Vectorized boundary processing
    b_face_indices = np.arange(n_interior, n_faces)
    b_elem_indices = n_elements + (b_face_indices - n_interior)

    u_face_b = velocity[b_elem_indices]
    mdot[n_interior:] = np.sum(u_face_b * face_sf[n_interior:], axis=1)

    return mdot
