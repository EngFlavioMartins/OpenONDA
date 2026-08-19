#!/usr/bin/env python3
"""
Time Integration for OpenONDA FVM Solver

Implements transient term discretization:
- Euler implicit (first-order, unconditionally stable)
- Euler explicit (first-order, conditionally stable)

Converted from uFVM cfdAssembleTransientTermEuler.m
"""

import numpy as np


def assemble_transient_term_euler_implicit(phi_old, time_step_size, rho, geo_data):
    """
    Assemble transient term using Euler implicit scheme.

    ∂(ρφ)/∂t ≈ (ρφ^{n+1} - ρφ^n) / Δt

    Implicit: φ^{n+1} appears in coefficient matrix (unconditionally stable)

    Args:
        phi_old: Previous cell values [units depend on field], shape
            ``(n_cells,)``.
        time_step_size: Positive time-step size [s].
        rho: Positive density [kg/m³], scalar or shape ``(n_cells,)``.
        geo_data: Geometry dictionary containing ``element_volumes`` [m³].

    Returns:
        dict: Transient term contribution
            - ac: Diagonal coefficients (n_elements,)
            - bc: RHS contribution (n_elements,)
    """

    volumes = geo_data["element_volumes"]

    # Coefficient: ρV/Δt
    ac = rho * volumes / time_step_size

    # RHS: (ρV/Δt) * φ_old
    bc = ac * phi_old

    return {"ac": ac, "bc": bc}


def advance_euler_explicit(phi_old, spatial_matrix, spatial_rhs, time_step_size, rho, volumes):
    """Advance one scalar-transport step with forward Euler.

    The steady finite-volume operators use ``A_spatial @ phi = b_spatial``.
    Their instantaneous residual is therefore ``b_spatial - A_spatial @ phi``
    and forward Euler advances

    ``phi_new = phi_old + dt * residual / (rho * volume)``.

    Explicit Euler cannot be represented as the diagonal/RHS pair returned by
    :func:`assemble_transient_term_euler_implicit`; doing so would leave the
    new field absent from the discrete time derivative.

    Args:
        phi_old: Cell-centred scalar at the old time [units depend on field],
            shape ``(n_cells,)``.
        spatial_matrix: Assembled steady spatial operator ``A_spatial``.
        spatial_rhs: Assembled steady right-hand side ``b_spatial``, shape
            ``(n_cells,)``.
        time_step_size: Positive time-step size [s].
        rho: Positive density [kg/m³], scalar or shape ``(n_cells,)``.
        volumes: Positive cell volumes [m³], shape ``(n_cells,)``.

    Returns:
        numpy.ndarray: Cell-centred scalar at the new time, shape
        ``(n_cells,)``.
    """
    phi_old = np.asarray(phi_old, dtype=np.float64)
    spatial_rhs = np.asarray(spatial_rhs, dtype=np.float64)
    volumes = np.asarray(volumes, dtype=np.float64)
    density = np.asarray(rho, dtype=np.float64)

    if phi_old.ndim != 1:
        raise ValueError("phi_old must be one-dimensional")
    if spatial_rhs.shape != phi_old.shape or volumes.shape != phi_old.shape:
        raise ValueError("spatial_rhs and volumes must have the same shape as phi_old")
    if density.ndim == 0:
        density = np.full(phi_old.shape, float(density), dtype=np.float64)
    if density.shape != phi_old.shape:
        raise ValueError("rho must be scalar or have the same shape as phi_old")
    if not np.isfinite(time_step_size) or time_step_size <= 0.0:
        raise ValueError("time_step_size must be finite and positive")
    if not np.all(np.isfinite(density)) or np.any(density <= 0.0):
        raise ValueError("rho must contain finite positive values")
    if not np.all(np.isfinite(volumes)) or np.any(volumes <= 0.0):
        raise ValueError("volumes must contain finite positive values")

    residual = spatial_rhs - np.asarray(spatial_matrix @ phi_old, dtype=np.float64)
    return phi_old + time_step_size * residual / (density * volumes)
