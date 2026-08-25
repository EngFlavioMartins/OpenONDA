"""
Influence matrix and RHS kernels for panel method.
==================
GPU kernels for aerodynamic_influence_coefficient assembly and boundary condition RHS computation.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: February 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import taichi as ti

from ....config.constants import PANEL_EPSILON
from ..kernels.biot_savart import compute_doublet_potential, compute_vortex_ring_velocity
from ..kernels.source_velocity import compute_source_velocity


@ti.kernel
def build_source_aerodynamic_influence_coefficient_matrix(
    vertex_position: ti.template(),
    panel_centre: ti.template(),
    normal: ti.template(),
    aerodynamic_influence_coefficient: ti.template(),
    n: int,
):
    for i, j in ti.ndrange(n, n):
        if i == j:
            # Seed the literal with a lattice scalar so f64 panel fields are
            # not silently evaluated through default_fp=f32.
            diagonal = normal[i][0] * 0.0 + 0.5
            aerodynamic_influence_coefficient[i, j] = diagonal
        else:
            v0, v1, v2 = vertex_position[j, 0], vertex_position[j, 1], vertex_position[j, 2]
            velocity = compute_source_velocity(panel_centre[i], v0, v1, v2, normal[j])
            aerodynamic_influence_coefficient[i, j] = velocity.dot(normal[i])


@ti.kernel
def build_dirichlet_aerodynamic_influence_coefficient_matrix(
    vertex_position: ti.template(),
    panel_centre: ti.template(),
    normal: ti.template(),
    aerodynamic_influence_coefficient: ti.template(),
    n: int,
):
    """
    Build the aerodynamic_influence_coefficient matrix for Dirichlet BC (Morino formulation).
    aerodynamic_influence_coefficient[i, j] = Potential at center i induced by unit doublet at panel j.

    For the Dirichlet BC, the aerodynamic_influence_coefficient is the potential coefficient matrix.
    The self-term is -0.5 (potential from the interior side).
    """
    for i, j in ti.ndrange(n, n):
        if i == j:
            diagonal = normal[i][0] * 0.0 - 0.5
            aerodynamic_influence_coefficient[i, j] = diagonal
        else:
            v0, v1, v2 = vertex_position[j, 0], vertex_position[j, 1], vertex_position[j, 2]
            aerodynamic_influence_coefficient[i, j] = compute_doublet_potential(
                panel_centre[i], v0, v1, v2
            )


@ti.kernel
def compute_right_hand_side(
    normal: ti.template(),
    body_velocity: ti.template(),
    freestream_velocity: ti.template(),
    incident_velocity: ti.template(),
    right_hand_side: ti.template(),
    n: int,
):
    """
    Compute the NEUMANN RHS for body-relative impermeability.

    ``incident_velocity`` is the VPM/external velocity only; rigid-body
    velocity is a distinct field.  The equation is
    ``A sigma = -(U_inf + u_incident - u_body) . n``.
    """
    for i in range(n):
        relative_incident_velocity = freestream_velocity + incident_velocity[i] - body_velocity[i]
        right_hand_side[i] = -relative_incident_velocity.dot(normal[i])


@ti.kernel
def compute_neumann_right_hand_side_with_sources(
    vertex_position: ti.template(),
    panel_centre: ti.template(),
    normal: ti.template(),
    freestream_velocity: ti.template(),
    wake_velocity: ti.template(),
    right_hand_side: ti.template(),
    n: int,
):
    """
    Compute RHS for Neumann BC with source-doublet formulation (Hess-Smith).
    right_hand_side[i] = - (freestream_velocity + wake_velocity) · n_i - Σ_j (V_source_j · n_i) * σ_j

    where:
        σ_j = -freestream_velocity · n_j (source strength)
        V_source_j = velocity induced by unit source at panel j
    """
    for i in range(n):
        # Freestream normal velocity
        v_total = freestream_velocity + wake_velocity[i]
        right_hand_side[i] = -v_total.dot(normal[i])

        # Source contribution: sum over all panels
        for j in range(n):
            # Source strength for panel j
            sigma_j = -freestream_velocity.dot(normal[j])

            # Compute source velocity at point i due to unit source at panel j
            v0, v1, v2 = vertex_position[j, 0], vertex_position[j, 1], vertex_position[j, 2]
            v_source_j = compute_source_velocity(panel_centre[i], v0, v1, v2, normal[j])

            # Add contribution: (V_source_j · n_i) * σ_j
            right_hand_side[i] -= v_source_j.dot(normal[i]) * sigma_j


@ti.kernel
def compute_dirichlet_right_hand_side(
    panel_centre: ti.template(),
    freestream_velocity: ti.template(),
    wake_velocity: ti.template(),
    right_hand_side: ti.template(),
    n: int,
):
    """
    Compute RHS for the Dirichlet boundary condition (Morino formulation).
    right_hand_side[i] = - (freestream_velocity + wake_velocity[i]) · panel_centre[i]

    For the doublet panel method with potential aerodynamic_influence_coefficient matrix, the Dirichlet BC
    enforces zero total potential on the surface:
        Σ_j C_ij * μ_j = -φ_∞(x_i) = -V_∞ · x_i

    This is the standard Morino formulation for non-lifting closed bodies.
    """
    for i in range(n):
        v_total = freestream_velocity + wake_velocity[i]
        right_hand_side[i] = -v_total.dot(panel_centre[i])


@ti.kernel
def compute_dirichlet_right_hand_side_with_sources(
    vertex_position: ti.template(),
    panel_centre: ti.template(),
    normal: ti.template(),
    freestream_velocity: ti.template(),
    wake_velocity: ti.template(),
    right_hand_side: ti.template(),
    n: int,
):
    """
    Compute RHS for the Dirichlet BC including source panel contributions.

    For the source-doublet panel method (Hess-Smith / Morino):
        right_hand_side[i] = -φ_∞(x_i) - Σ_j φ_source_j(x_i) · σ_j

    where:
        φ_∞(x_i) = V_∞ · x_i  (freestream potential)
        σ_j = -V_∞ · n_j  (source strength, known from freestream)
        φ_source_j(x_i) = potential at x_i due to unit source at panel j

    This formulation ensures exact cancellation of the freestream normal velocity
    on the surface, which is critical for accurate results on closed bodies.
    """
    for i in range(n):
        # Freestream potential at collocation point
        v_total = freestream_velocity + wake_velocity[i]
        phi_inf = -v_total.dot(panel_centre[i])

        # Pure doublet formulation (no source contributions)
        right_hand_side[i] = phi_inf


@ti.kernel
def compute_surface_velocity(
    vertex_position: ti.template(),
    panel_centre: ti.template(),
    doublet_strength: ti.template(),
    freestream_velocity: ti.template(),
    wake_velocity: ti.template(),
    surface_velocity: ti.template(),
    n: int,
):
    """
    Compute total velocity at collocation points.
    V_total = freestream_velocity + wake_velocity + V_induced_by_panels
    Note: Self-induction (i==j) is skipped.
    """
    for i in range(n):
        v_induced = panel_centre[i] * 0.0
        p_target = panel_centre[i]

        for j in range(n):
            if i != j:
                v0, v1, v2 = vertex_position[j, 0], vertex_position[j, 1], vertex_position[j, 2]
                v_induced += doublet_strength[j] * compute_vortex_ring_velocity(
                    p_target, v0, v1, v2
                )

        surface_velocity[i] = freestream_velocity + wake_velocity[i] + v_induced


@ti.kernel
def compute_surface_velocity_with_sources(
    vertex_position: ti.template(),
    panel_centre: ti.template(),
    normal: ti.template(),
    doublet_strength: ti.template(),
    source_strength: ti.template(),
    freestream_velocity: ti.template(),
    wake_velocity: ti.template(),
    surface_velocity: ti.template(),
    n: int,
):
    """
    Compute total velocity at collocation points including exact source contribution.
    V_total = freestream_velocity + wake_velocity + V_doublet + V_source

    For a source-doublet panel method (Hess-Smith / Morino):
        - Source strength: σ_j = -V_∞ · n_j (known from freestream)
        - Doublet strength: μ_j (solved from Dirichlet BC)
        - V_source: exact analytical velocity from source panels
        - V_doublet: exact analytical velocity from doublet panels (vortex ring)

    This formulation provides accurate results on curved surfaces (e.g., sphere)
    where the point-source approximation fails.
    """
    for i in range(n):
        # Multiplying a lattice vector by zero derives the accumulator dtype
        # from the panel field instead of Taichi's global default_fp.
        v_doublet = normal[i] * 0.0
        v_source = normal[i] * 0.0
        p_target = panel_centre[i]

        for j in range(n):
            v0, v1, v2 = vertex_position[j, 0], vertex_position[j, 1], vertex_position[j, 2]
            normal_j = normal[j]

            # Doublet contribution (vortex ring) - skip self-induction
            if i != j:
                v_doublet += doublet_strength[j] * compute_vortex_ring_velocity(
                    p_target, v0, v1, v2
                )

            if i == j:
                # Exterior limit of a constant source panel at its own
                # collocation point: sigma/2 along the outward normal. The
                # general kernel must not be used here — at the panel's own
                # centroid the solid-angle term evaluates atan2(0, negative),
                # which sits exactly on the branch cut, so the sign of the
                # self-term follows floating-point noise. This is the same
                # +0.5 the influence matrix assembles on its diagonal, so the
                # two paths now agree by construction and the solved
                # no-penetration condition is reproduced exactly here.
                v_source += source_strength[j] * 0.5 * normal_j
            else:
                v_source += source_strength[j] * compute_source_velocity(
                    p_target, v0, v1, v2, normal_j
                )

        surface_velocity[i] = freestream_velocity + wake_velocity[i] + v_doublet + v_source


@ti.kernel
def compute_relative_surface_velocity(
    surface_velocity_absolute: ti.template(),
    body_velocity: ti.template(),
    surface_velocity_relative: ti.template(),
    n: int,
):
    """Subtract rigid-body velocity from the absolute surface flow field."""
    for i in range(n):
        surface_velocity_relative[i] = surface_velocity_absolute[i] - body_velocity[i]


@ti.kernel
def compute_pressure_bernoulli(
    surface_velocity: ti.template(),
    freestream_speed: ti.f64,
    pressure_coefficient: ti.template(),
    n: int,
):
    """
    Compute pressure coefficient using Bernoulli: pressure_coefficient = 1 - (V/freestream_velocity)^2.
    """
    for i in range(n):
        v_mag_sq = surface_velocity[i].norm_sqr()
        pressure_coefficient[i] = 1.0 - v_mag_sq / (
            freestream_speed * freestream_speed + PANEL_EPSILON
        )


@ti.kernel
def compute_forces_bernoulli(
    surface_velocity: ti.template(),
    freestream_speed: ti.f64,
    area: ti.template(),
    normal: ti.template(),
    density: ti.f64,
    forces: ti.template(),
    n: int,
):
    """
    Compute forces using Bernoulli: F = 0.5 * density * (freestream_velocity^2 - V^2) * Area * n.
    """
    for i in range(n):
        v_mag_sq = surface_velocity[i].norm_sqr()
        pressure_difference = 0.5 * density * (freestream_speed * freestream_speed - v_mag_sq)
        forces[i] = pressure_difference * area[i] * normal[i]


@ti.kernel
def compute_forces_kutta_joukowski(
    doublet_strength: ti.template(),
    surface_velocity: ti.template(),
    vertex_position: ti.template(),
    area: ti.template(),
    normal: ti.template(),
    density: ti.f64,
    forces: ti.template(),
    n: int,
):
    """
    Compute panel force using a discrete Kutta-Joukowski form:
    F_i = density * circulation_i * (V_surface_i x bound_vortex_leg_i).

    vortex_strength_i is taken as panel strength (doublet/circulation-like variable)
    and l_bound_i is the panel bound-edge vector from vertex 0 to vertex 1.
    If the edge is degenerate, a geometric fallback based on area/normal is used.
    """
    for i in range(n):
        v0, v1 = vertex_position[i, 0], vertex_position[i, 1]
        l_bound = v1 - v0
        edge_norm = l_bound.norm()
        if edge_norm < PANEL_EPSILON:
            l_bound = normal[i] * ti.sqrt(ti.max(area[i], PANEL_EPSILON))
        panel_doublet_strength = doublet_strength[i]
        forces[i] = density * panel_doublet_strength * surface_velocity[i].cross(l_bound)
