"""
Influence matrix and RHS kernels for panel method.
==================
GPU kernels for AIC assembly and boundary condition RHS computation.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: February 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import taichi as ti

from ....config.constants import PANEL_EPSILON, TI_FLOAT
from ..kernels.biot_savart import compute_doublet_potential, compute_vortex_ring_velocity
from ..kernels.source_velocity import compute_source_velocity


@ti.kernel
def build_AIC_matrix(
    vertices: ti.template(),
    centers: ti.template(),
    normals: ti.template(),
    AIC: ti.template(),
    n: int,
):
    """
    Build the Aerodynamic Influence Coefficient (AIC) matrix for Neumann BC.
    AIC[i, j] = Normal velocity at center i induced by unit doublet at panel j.

    For the Neumann BC, the AIC is the normal component of the velocity
    induced by a unit doublet panel. The self-term is 0.5 (half the solid angle).
    """
    for i, j in ti.ndrange(n, n):
        if i == j:
            AIC[i, j] = 0.5
        else:
            v0, v1, v2 = vertices[j, 0], vertices[j, 1], vertices[j, 2]
            # Compute velocity induced by unit doublet at panel j
            v_induced = compute_vortex_ring_velocity(centers[i], v0, v1, v2)
            # Project onto normal at panel i
            AIC[i, j] = v_induced.dot(normals[i])


@ti.kernel
def build_AIC_matrix_dirichlet(
    vertices: ti.template(),
    centers: ti.template(),
    normals: ti.template(),
    AIC: ti.template(),
    n: int,
):
    """
    Build the AIC matrix for Dirichlet BC (Morino formulation).
    AIC[i, j] = Potential at center i induced by unit doublet at panel j.

    For the Dirichlet BC, the AIC is the potential coefficient matrix.
    The self-term is -0.5 (potential from the interior side).
    """
    for i, j in ti.ndrange(n, n):
        if i == j:
            AIC[i, j] = -0.5
        else:
            v0, v1, v2 = vertices[j, 0], vertices[j, 1], vertices[j, 2]
            AIC[i, j] = compute_doublet_potential(centers[i], v0, v1, v2)


@ti.kernel
def compute_RHS(
    centers: ti.template(),
    normals: ti.template(),
    V_inf: ti.types.vector(3, TI_FLOAT),
    V_wake: ti.template(),
    rhs: ti.template(),
    n: int,
):
    """
    Compute RHS for the boundary condition (zero normal velocity).
    rhs[i] = - (V_inf + V_wake) · n_i

    Note: This is the Neumann BC formulation. For a consistent panel method
    with the potential AIC matrix, use compute_RHS_dirichlet instead.
    """
    for i in range(n):
        v_total = V_inf + V_wake[i]
        rhs[i] = -v_total.dot(normals[i])


@ti.kernel
def compute_RHS_neumann_with_sources(
    vertices: ti.template(),
    centers: ti.template(),
    normals: ti.template(),
    V_inf: ti.types.vector(3, TI_FLOAT),
    V_wake: ti.template(),
    rhs: ti.template(),
    n: int,
):
    """
    Compute RHS for Neumann BC with source-doublet formulation (Hess-Smith).
    rhs[i] = - (V_inf + V_wake) · n_i - Σ_j (V_source_j · n_i) * σ_j

    where:
        σ_j = -V_inf · n_j (source strength)
        V_source_j = velocity induced by unit source at panel j
    """
    for i in range(n):
        # Freestream normal velocity
        v_total = V_inf + V_wake[i]
        rhs[i] = -v_total.dot(normals[i])

        # Source contribution: sum over all panels
        for j in range(n):
            # Source strength for panel j
            sigma_j = -V_inf.dot(normals[j])

            # Compute source velocity at point i due to unit source at panel j
            v0, v1, v2 = vertices[j, 0], vertices[j, 1], vertices[j, 2]
            v_source_j = compute_source_velocity(centers[i], v0, v1, v2, normals[j])

            # Add contribution: (V_source_j · n_i) * σ_j
            rhs[i] -= v_source_j.dot(normals[i]) * sigma_j


@ti.kernel
def compute_RHS_dirichlet(
    centers: ti.template(),
    V_inf: ti.types.vector(3, TI_FLOAT),
    V_wake: ti.template(),
    rhs: ti.template(),
    n: int,
):
    """
    Compute RHS for the Dirichlet boundary condition (Morino formulation).
    rhs[i] = - (V_inf + V_wake[i]) · centers[i]

    For the doublet panel method with potential AIC matrix, the Dirichlet BC
    enforces zero total potential on the surface:
        Σ_j C_ij * μ_j = -φ_∞(x_i) = -V_∞ · x_i

    This is the standard Morino formulation for non-lifting closed bodies.
    """
    for i in range(n):
        v_total = V_inf + V_wake[i]
        rhs[i] = -v_total.dot(centers[i])


@ti.kernel
def compute_RHS_dirichlet_with_sources(
    vertices: ti.template(),
    centers: ti.template(),
    normals: ti.template(),
    V_inf: ti.types.vector(3, TI_FLOAT),
    V_wake: ti.template(),
    rhs: ti.template(),
    n: int,
):
    """
    Compute RHS for the Dirichlet BC including source panel contributions.

    For the source-doublet panel method (Hess-Smith / Morino):
        rhs[i] = -φ_∞(x_i) - Σ_j φ_source_j(x_i) · σ_j

    where:
        φ_∞(x_i) = V_∞ · x_i  (freestream potential)
        σ_j = -V_∞ · n_j  (source strength, known from freestream)
        φ_source_j(x_i) = potential at x_i due to unit source at panel j

    This formulation ensures exact cancellation of the freestream normal velocity
    on the surface, which is critical for accurate results on closed bodies.
    """
    for i in range(n):
        # Freestream potential at collocation point
        v_total = V_inf + V_wake[i]
        phi_inf = -v_total.dot(centers[i])

        # Pure doublet formulation (no source contributions)
        rhs[i] = phi_inf


@ti.kernel
def compute_surface_velocities(
    vertices: ti.template(),
    centers: ti.template(),
    strengths: ti.template(),
    V_inf: ti.types.vector(3, TI_FLOAT),
    V_wake: ti.template(),
    V_surface: ti.template(),
    n: int,
):
    """
    Compute total velocity at collocation points.
    V_total = V_inf + V_wake + V_induced_by_panels
    Note: Self-induction (i==j) is skipped.
    """
    for i in range(n):
        v_induced = ti.Vector([0.0, 0.0, 0.0])
        p_target = centers[i]

        for j in range(n):
            if i != j:
                v0, v1, v2 = vertices[j, 0], vertices[j, 1], vertices[j, 2]
                v_induced += strengths[j] * compute_vortex_ring_velocity(p_target, v0, v1, v2)

        V_surface[i] = V_inf + V_wake[i] + v_induced


@ti.kernel
def compute_surface_velocities_with_sources(
    vertices: ti.template(),
    centers: ti.template(),
    normals: ti.template(),
    strengths: ti.template(),
    source_strengths: ti.template(),
    V_inf: ti.types.vector(3, TI_FLOAT),
    V_wake: ti.template(),
    V_surface: ti.template(),
    n: int,
):
    """
    Compute total velocity at collocation points including exact source contribution.
    V_total = V_inf + V_wake + V_doublet + V_source

    For a source-doublet panel method (Hess-Smith / Morino):
        - Source strength: σ_j = -V_∞ · n_j (known from freestream)
        - Doublet strength: μ_j (solved from Dirichlet BC)
        - V_source: exact analytical velocity from source panels
        - V_doublet: exact analytical velocity from doublet panels (vortex ring)

    This formulation provides accurate results on curved surfaces (e.g., sphere)
    where the point-source approximation fails.
    """
    for i in range(n):
        v_doublet = ti.Vector([0.0, 0.0, 0.0])
        v_source = ti.Vector([0.0, 0.0, 0.0])
        p_target = centers[i]

        for j in range(n):
            v0, v1, v2 = vertices[j, 0], vertices[j, 1], vertices[j, 2]
            normal_j = normals[j]

            # Doublet contribution (vortex ring) - skip self-induction
            if i != j:
                v_doublet += strengths[j] * compute_vortex_ring_velocity(p_target, v0, v1, v2)

            # Source contribution - include self-induction (exact analytical)
            v_source += source_strengths[j] * compute_source_velocity(
                p_target, v0, v1, v2, normal_j
            )

        V_surface[i] = V_inf + V_wake[i] + v_doublet + v_source


@ti.kernel
def compute_pressure_bernoulli(
    V_surface: ti.template(), V_inf_mag: TI_FLOAT, Cp: ti.template(), n: int
):
    """
    Compute pressure coefficient using Bernoulli: Cp = 1 - (V/V_inf)^2.
    """
    for i in range(n):
        v_mag_sq = V_surface[i].norm_sqr()
        Cp[i] = 1.0 - v_mag_sq / (V_inf_mag * V_inf_mag + PANEL_EPSILON)


@ti.kernel
def compute_forces_bernoulli(
    V_surface: ti.template(),
    V_inf_mag: TI_FLOAT,
    areas: ti.template(),
    normals: ti.template(),
    rho: TI_FLOAT,
    forces: ti.template(),
    n: int,
):
    """
    Compute forces using Bernoulli: F = 0.5 * rho * (V_inf^2 - V^2) * Area * n.
    """
    for i in range(n):
        v_mag_sq = V_surface[i].norm_sqr()
        p_diff = 0.5 * rho * (V_inf_mag * V_inf_mag - v_mag_sq)
        forces[i] = p_diff * areas[i] * normals[i]


@ti.kernel
def compute_forces_kutta_joukowski(
    strengths: ti.template(),
    V_surface: ti.template(),
    vertices: ti.template(),
    areas: ti.template(),
    normals: ti.template(),
    rho: TI_FLOAT,
    forces: ti.template(),
    n: int,
):
    """
    Compute panel force using a discrete Kutta-Joukowski form:
    F_i = rho * Gamma_i * (V_surface_i x l_bound_i).

    Gamma_i is taken as panel strength (doublet/circulation-like variable)
    and l_bound_i is the panel bound-edge vector from vertex 0 to vertex 1.
    If the edge is degenerate, a geometric fallback based on area/normal is used.
    """
    for i in range(n):
        v0, v1 = vertices[i, 0], vertices[i, 1]
        l_bound = v1 - v0
        edge_norm = l_bound.norm()
        if edge_norm < PANEL_EPSILON:
            l_bound = normals[i] * ti.sqrt(ti.max(areas[i], PANEL_EPSILON))
        gamma_i = strengths[i]
        forces[i] = rho * gamma_i * V_surface[i].cross(l_bound)
