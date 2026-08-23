#!/usr/bin/env python3
"""
Time Integration for OpenONDA FVM Solver

Implements transient term discretization:
- Euler implicit (first-order, unconditionally stable)
- Euler explicit (first-order, conditionally stable)

Converted from uFVM cfdAssembleTransientTermEuler.m
"""

import numpy as np


def assemble_transient_term_euler_implicit(scalar_field_old, time_step_size, density, geo_data):
    """
    Assemble transient term using Euler implicit scheme.

    ∂(ρφ)/∂t ≈ (ρφ^{n+1} - ρφ^n) / Δt

    Implicit: φ^{n+1} appears in coefficient matrix (unconditionally stable)

    Args:
        scalar_field_old: Previous cell values [units depend on field], shape
            ``(n_cells,)``.
        time_step_size: Positive time-step size [s].
        density: Positive density [kg/m³], scalar or shape ``(n_cells,)``.
        geo_data: Geometry dictionary containing ``cell_volume`` [m³].

    Returns:
        dict: Transient term contribution
            - ac: Diagonal coefficients (n_elements,)
            - bc: RHS contribution (n_elements,)
    """

    cell_volume = geo_data["cell_volume"]

    # Coefficient: ρV/Δt
    ac = density * cell_volume / time_step_size

    # RHS: (ρV/Δt) * φ_old
    bc = ac * scalar_field_old

    return {"ac": ac, "bc": bc}


def advance_euler_explicit(
    scalar_field_old, spatial_matrix, spatial_rhs, time_step_size, density, cell_volume
):
    """Advance one scalar-transport step with forward Euler.

    The steady finite-volume operators use ``A_spatial @ scalar_field = b_spatial``.
    Their instantaneous residual is therefore
    ``b_spatial - A_spatial @ scalar_field``
    and forward Euler advances

    ``scalar_field_new = scalar_field_old + time_step_size * residual /
    (density * volume)``.

    Explicit Euler cannot be represented as the diagonal/RHS pair returned by
    :func:`assemble_transient_term_euler_implicit`; doing so would leave the
    new field absent from the discrete time derivative.

    Args:
        scalar_field_old: Cell-centred scalar at the old time [units depend on field],
            shape ``(n_cells,)``.
        spatial_matrix: Assembled steady spatial operator ``A_spatial``.
        spatial_rhs: Assembled steady right-hand side ``b_spatial``, shape
            ``(n_cells,)``.
        time_step_size: Positive time-step size [s].
        density: Positive density [kg/m³], scalar or shape ``(n_cells,)``.
        cell_volume: Positive cell cell_volume [m³], shape ``(n_cells,)``.

    Returns:
        numpy.ndarray: Cell-centred scalar at the new time, shape
        ``(n_cells,)``.
    """
    scalar_field_old = np.asarray(scalar_field_old, dtype=np.float64)
    spatial_rhs = np.asarray(spatial_rhs, dtype=np.float64)
    cell_volume = np.asarray(cell_volume, dtype=np.float64)
    density = np.asarray(density, dtype=np.float64)

    if scalar_field_old.ndim != 1:
        raise ValueError("scalar_field_old must be one-dimensional")
    if spatial_rhs.shape != scalar_field_old.shape or cell_volume.shape != scalar_field_old.shape:
        raise ValueError("spatial_rhs and cell_volume must have the same shape as scalar_field_old")
    if density.ndim == 0:
        density = np.full(scalar_field_old.shape, float(density), dtype=np.float64)
    if density.shape != scalar_field_old.shape:
        raise ValueError("density must be scalar or have the same shape as scalar_field_old")
    if not np.isfinite(time_step_size) or time_step_size <= 0.0:
        raise ValueError("time_step_size must be finite and positive")
    if not np.all(np.isfinite(density)) or np.any(density <= 0.0):
        raise ValueError("density must contain finite positive values")
    if not np.all(np.isfinite(cell_volume)) or np.any(cell_volume <= 0.0):
        raise ValueError("cell_volume must contain finite positive values")

    residual = spatial_rhs - np.asarray(spatial_matrix @ scalar_field_old, dtype=np.float64)
    return scalar_field_old + time_step_size * residual / (density * cell_volume)
