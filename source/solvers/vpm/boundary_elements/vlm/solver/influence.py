"""
VLM influence kernels: aerodynamic_influence_coefficient assembly, RHS construction, and induced-velocity and
pressure-coefficient evaluation.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import taichi as ti

from ....config.constants import VLM_EPSILON, VLM_SMALL_VELOCITY
from ..kernels.biot_savart import bound_vortex_velocity, horseshoe_velocity, vortex_ring_velocity


@ti.kernel
def compute_aerodynamic_influence_coefficient_matrix(
    collocation_point: ti.template(),
    vortex_point_position: ti.template(),
    panel_corner_position: ti.template(),
    normal: ti.template(),
    is_trailing_edge: ti.template(),
    trailing_edge_index: ti.template(),
    aerodynamic_influence_coefficient: ti.template(),
    n_panels: ti.i32,
    epsilon: float,
    coupled_mode: ti.i32,
    wake_offset: ti.types.vector(3, float),
):
    """
    Compute aerodynamic influence coefficient matrix.

    aerodynamic_influence_coefficient[i,j] = normal_i · velocity_at_i_from_horseshoe_j (if not coupled_mode)
    aerodynamic_influence_coefficient[i,j] = normal_i · velocity_at_i_from_bound_only_j  (if coupled_mode)

    This is the core VLM computation. Each entry represents the normal
    component of velocity induced at collocation_point point i by a unit
    circulation vortex on panel j.

    In 'coupled_mode', the bound horseshoe is closed by a short near-wake panel
    rather than semi-infinite trailing legs: each trailing-edge panel's trailing
    legs are extended downstream by 'wake_offset' (= relative-wind · dt, the
    one-step convection length).  This fills the otherwise-empty gap between the
    trailing edge and the first free wake particle (which sits one convection
    length downstream), so the bound solve "sees" its own near wake (the implicit
    near-wake panel of the canonical UVLM-VPM coupling).  The free VPM particles
    continue the wake from TE + wake_offset onward, so there is no double-counting.
    With wake_offset = 0 this reduces to the bound-only horseshoe.

    Args:
        collocation_point: Collocation points (N x 3)
        vortex_point_position: Horseshoe vertices (N x 4 x 3)
                       [v1=far_left, v2=bound_left, v3=bound_right, v4=far_right]
        panel_corner_position: Panel corner positions (N x 4 x 3)
        normal: Panel normal (N x 3)
        aerodynamic_influence_coefficient: Output matrix (N x N)
        n_panels: Number of panels
        epsilon: Regularization parameter
        coupled_mode: If 1, use bound-only induction (coupled VLM-VPM)
    """
    for i, j in ti.ndrange(n_panels, n_panels):
        # Get horseshoe vertices
        v1 = vortex_point_position[j, ti.i32(0)]
        v2 = vortex_point_position[j, ti.i32(1)]
        v3 = vortex_point_position[j, ti.i32(2)]
        v4 = vortex_point_position[j, ti.i32(3)]

        vel = ti.Vector([0.0, 0.0, 0.0])

        if coupled_mode == 1:
            # Internal bound-horseshoe: TE_left -> bound_left -> bound_right -> TE_right
            # This captures the on-body 3D induction (downwash) while letting
            # VPM particles handle the wake behind the trailing edge.
            #
            # The internal trailing legs must run from the bound vortex all the way
            # to the WING trailing edge (the downstream edge of the strip's TE
            # panel), NOT merely to this panel's own downstream edge. Using the
            # panel's own edge truncates each chordwise contribution so the
            # streamwise (trailing) vorticity along a spanwise station never
            # accumulates to the full bound circulation that the VPM wake then
            # convects downstream. That truncation destroys the finite-wing
            # downwash and yields an almost-constant spanwise loading that fails
            # to taper to zero at the tips. Reading the TE panel_corner_position of the strip's
            # trailing-edge panel (trailing_edge_index[j]) extends the legs correctly; for a
            # TE panel trailing_edge_index[j] == j, so its own behaviour is unchanged.
            te = trailing_edge_index[j]
            v_te_L = panel_corner_position[te, ti.i32(3)]
            v_te_R = panel_corner_position[te, ti.i32(2)]

            # 1. Left internal leg (TE to Bound)
            vel += bound_vortex_velocity(collocation_point[i], v_te_L, v2, 1.0, epsilon)
            # 2. Bound leg
            vel += bound_vortex_velocity(collocation_point[i], v2, v3, 1.0, epsilon)
            # 3. Right internal leg (Bound to TE)
            vel += bound_vortex_velocity(collocation_point[i], v3, v_te_R, 1.0, epsilon)
            # 4. Near-wake panel: extend the trailing legs one convection length
            #    downstream so the filament free ends meet the first wake particle
            #    at TE + wake_offset (implicit near-wake; only for TE panels).
            if is_trailing_edge[j] == 1:
                vel += bound_vortex_velocity(
                    collocation_point[i], v_te_R, v_te_R + wake_offset, 1.0, epsilon
                )
                vel += bound_vortex_velocity(
                    collocation_point[i], v_te_L + wake_offset, v_te_L, 1.0, epsilon
                )
        else:
            # FULL semi-infinite horseshoe: used for standalone VLM
            vel = horseshoe_velocity(collocation_point[i], v1, v2, v3, v4, 1.0, epsilon)

        # Normal component (downwash factor)
        aerodynamic_influence_coefficient[i, j] = vel.dot(normal[i])


@ti.kernel
def compute_right_hand_side(
    collocation_point: ti.template(),
    normal: ti.template(),
    right_hand_side: ti.template(),
    n_panels: ti.i32,
    freestream_velocity_x: float,
    freestream_velocity_y: float,
    freestream_velocity_z: float,
):
    """
    Compute right-hand side of VLM system.

    RHS[i] = -normal_i · freestream_velocity

    This enforces the boundary condition that total normal velocity
    (freestream + induced) must be zero at each collocation_point point.

    Args:
        collocation_point: Collocation points (N x 3)
        normal: Panel normal (N x 3)
        right_hand_side: Output RHS vector (N,)
        n_panels: Number of panels
        freestream_velocity_x, freestream_velocity_y, freestream_velocity_z: Freestream velocity components
    """
    freestream_velocity = ti.Vector(
        [freestream_velocity_x, freestream_velocity_y, freestream_velocity_z]
    )

    for i in range(n_panels):
        # Negative because we move freestream_velocity term to RHS
        right_hand_side[i] = -normal[i].dot(freestream_velocity)


@ti.kernel
def compute_coupled_right_hand_side(
    collocation_point: ti.template(),
    normal: ti.template(),
    external_velocity: ti.template(),
    V_kinematic: ti.template(),
    right_hand_side: ti.template(),
    n_panels: ti.i32,
):
    """
    Compute RHS with generic external velocity field and surface motion.

    RHS[i] = -normal[i] · (external_velocity[i] - V_kinematic[i])

    Boundary condition: (external_velocity + V_induced - V_kinematic) · n = 0
    Therefore: V_induced · n = - (external_velocity - V_kinematic) · n

    Args:
        collocation_point: Collocation points (N x 3)
        normal: Panel normal (N x 3)
        external_velocity: Total external velocity (background + particles) at collocation_point (N x 3)
        V_kinematic: Surface velocity at collocation_point (N x 3)
        right_hand_side: Output RHS vector (N,)
        n_panels: Number of panels
    """
    for i in range(n_panels):
        # Relative flow velocity seen by the surface (excluding induced)
        V_kin = ti.Vector(
            [V_kinematic[i, ti.i32(0)], V_kinematic[i, ti.i32(1)], V_kinematic[i, ti.i32(2)]]
        )
        V_rel_inflow = external_velocity[i] - V_kin
        right_hand_side[i] = -normal[i].dot(V_rel_inflow)


@ti.kernel
def compute_induced_velocities(
    n_panels: ti.i32,
    collocation_point: ti.template(),
    vortex_point_position: ti.template(),
    circulation: ti.template(),
    velocity: ti.template(),
    external_velocity: ti.template(),
):
    """
    Compute total velocity including external field.

    V_total = external_velocity + V_induced

    Args:
        n_panels: Number of panels
        collocation_point: Collocation points (N x 3)
        vortex_point_position: Horseshoe vertices (N x 4 x 3)
        circulation: Circulation distribution (N,)
        velocity: Output velocity field (N x 3)
        external_velocity: External velocity field (N x 3)
    """
    for i in range(n_panels):
        vel_induced = ti.Vector([0.0, 0.0, 0.0])

        for j in range(n_panels):
            v1 = vortex_point_position[j, ti.i32(0)]
            v2 = vortex_point_position[j, ti.i32(1)]
            v3 = vortex_point_position[j, ti.i32(2)]
            v4 = vortex_point_position[j, ti.i32(3)]

            vel_induced += vortex_ring_velocity(
                collocation_point[i], v1, v2, v3, v4, circulation[j], VLM_EPSILON
            )

        # Total velocity
        velocity[i] = external_velocity[i] + vel_induced


@ti.kernel
def add_induced_velocity_at_targets(
    target_position: ti.template(),
    target_velocity: ti.template(),
    vortex_point_position: ti.template(),
    circulation: ti.template(),
    n_targets: ti.i32,
    n_panels: ti.i32,
):
    """Accumulate the solved VLM field at arbitrary temporary VPM targets."""
    for i in range(n_targets):
        induced = ti.Vector([0.0, 0.0, 0.0])
        for j in range(n_panels):
            induced += vortex_ring_velocity(
                target_position[i],
                vortex_point_position[j, ti.i32(0)],
                vortex_point_position[j, ti.i32(1)],
                vortex_point_position[j, ti.i32(2)],
                vortex_point_position[j, ti.i32(3)],
                circulation[j],
                VLM_EPSILON,
            )
        target_velocity[i] += induced


@ti.kernel
def add_induced_velocity_and_gradient_at_targets(
    target_position: ti.template(),
    target_velocity: ti.template(),
    target_gradient: ti.template(),
    vortex_point_position: ti.template(),
    circulation: ti.template(),
    n_targets: ti.i32,
    n_panels: ti.i32,
    finite_difference_step: ti.f32,
):
    """Accumulate VLM velocity and its stage-consistent target Jacobian.

    The ring kernel is the authoritative VLM velocity operator.  Centered
    differences reuse that same operator at temporary stage positions, so the
    stretching field cannot drift from the velocity field used for advection.
    ``target_gradient[i,j]`` is ``∂u_i/∂x_j``.
    """
    for i in range(n_targets):
        target = target_position[i]
        # Derive temporaries from the destination fields so f64 VPM
        # accumulators do not silently force this contribution through f32.
        induced = target * 0.0
        for panel in range(n_panels):
            induced += vortex_ring_velocity(
                target,
                vortex_point_position[panel, ti.i32(0)],
                vortex_point_position[panel, ti.i32(1)],
                vortex_point_position[panel, ti.i32(2)],
                vortex_point_position[panel, ti.i32(3)],
                circulation[panel],
                VLM_EPSILON,
            )
        gradient = target_gradient[i] * 0.0
        for column in ti.static(range(3)):
            offset = target * 0.0
            offset[column] = finite_difference_step
            plus = target * 0.0
            minus = target * 0.0
            for panel in range(n_panels):
                v1 = vortex_point_position[panel, ti.i32(0)]
                v2 = vortex_point_position[panel, ti.i32(1)]
                v3 = vortex_point_position[panel, ti.i32(2)]
                v4 = vortex_point_position[panel, ti.i32(3)]
                plus += vortex_ring_velocity(
                    target + offset, v1, v2, v3, v4, circulation[panel], VLM_EPSILON
                )
                minus += vortex_ring_velocity(
                    target - offset, v1, v2, v3, v4, circulation[panel], VLM_EPSILON
                )
            derivative = (plus - minus) / (2.0 * finite_difference_step)
            for row in ti.static(range(3)):
                gradient[row, column] = derivative[row]
        target_velocity[i] += induced
        target_gradient[i] = gradient


@ti.func
def _bound_panel_pair_velocity(
    i: int,
    j: int,
    target: ti.template(),
    vortex_point_position: ti.template(),
    panel_corner_position: ti.template(),
    trailing_edge_index: ti.template(),
    g: float,
    coupled_mode: ti.i32,
) -> ti.math.vec3:
    """Compute velocity contribution from panel j at evaluation point on panel i."""
    v2 = vortex_point_position[j, ti.i32(1)]
    v3 = vortex_point_position[j, ti.i32(2)]
    vel = ti.Vector([0.0, 0.0, 0.0])
    # Coupled internal trailing legs run to the WING trailing edge (the TE panel of
    # j's chordwise strip), matching compute_aerodynamic_influence_coefficient_matrix. See note there.
    te = trailing_edge_index[j]
    if i == j:
        v_te_L = panel_corner_position[te, ti.i32(3)]
        v_te_R = panel_corner_position[te, ti.i32(2)]
        if coupled_mode == 1:
            vel += bound_vortex_velocity(target, v_te_L, v2, g, VLM_EPSILON)
            vel += bound_vortex_velocity(target, v3, v_te_R, g, VLM_EPSILON)
        else:
            vel += bound_vortex_velocity(
                target, vortex_point_position[j, ti.i32(0)], v2, g, VLM_EPSILON
            )
            vel += bound_vortex_velocity(
                target, v3, vortex_point_position[j, ti.i32(3)], g, VLM_EPSILON
            )
    else:
        if coupled_mode == 1:
            v_te_L = panel_corner_position[te, ti.i32(3)]
            v_te_R = panel_corner_position[te, ti.i32(2)]
            vel += bound_vortex_velocity(target, v_te_L, v2, g, VLM_EPSILON)
            vel += bound_vortex_velocity(target, v2, v3, g, VLM_EPSILON)
            vel += bound_vortex_velocity(target, v3, v_te_R, g, VLM_EPSILON)
        else:
            vel += horseshoe_velocity(
                target,
                vortex_point_position[j, ti.i32(0)],
                v2,
                v3,
                vortex_point_position[j, ti.i32(3)],
                g,
                VLM_EPSILON,
            )
    return vel


@ti.kernel
def apply_circulation_smoothing(
    circulation: ti.template(),
    circulation_old: ti.template(),
    smoothed_circulation: ti.template(),
    n: ti.i32,
):
    for i in range(n):
        smoothed_circulation[i] = 0.5 * (circulation[i] + circulation_old[i])


@ti.kernel
def compute_induced_velocities_at_bound(
    n_panels: ti.i32,
    bound_vortex_midpoint: ti.template(),
    vortex_point_position: ti.template(),
    panel_corner_position: ti.template(),
    trailing_edge_index: ti.template(),
    circulation: ti.template(),
    velocity: ti.template(),
    external_velocity: ti.template(),
    coupled_mode: ti.i32,
):
    """
    Compute total velocity AT BOUND VORTEX MIDPOINTS.

    This is used for Kutta-Joukowski force calculation: F = density * circulation * (V x l)

    Physics notes:
    - Excludes self-induced velocity from the bound leg itself (singular).
    - In standalone mode: includes induction from others + local horseshoe side-legs.
    - In coupled_mode: includes induction from others (bound-only) + VPM wake.

    Args:
        n_panels: Number of panels
        bound_vortex_midpoint: Midpoints of bound vortices (N x 3)
        vortex_point_position: Horseshoe vertices (N x 4 x 3)
        circulation: Circulation distribution (N,)
        velocity: Output velocity field (N x 3)
        external_velocity: External velocity field at bound midpoints (N x 3)
        coupled_mode: If 1, use bound-only induction consistent with UVLM
    """
    for i in range(n_panels):
        vel_induced = ti.Vector([0.0, 0.0, 0.0])
        target = bound_vortex_midpoint[i]

        for j in range(n_panels):
            vel_induced += _bound_panel_pair_velocity(
                i,
                j,
                target,
                vortex_point_position,
                panel_corner_position,
                trailing_edge_index,
                circulation[j],
                coupled_mode,
            )

        # Total velocity = External + Induced
        velocity[i] = external_velocity[i] + vel_induced


@ti.kernel
def compute_pressure_coefficients(
    velocity: ti.template(),
    pressure_coefficient: ti.template(),
    n_panels: ti.i32,
    freestream_velocity_mag_sq: float,
):
    """
    Compute pressure coefficient at each panel.

    pressure_coefficient = 1 - (|V|/|freestream_velocity|)²

    Args:
        velocity: Velocity at collocation_point points (N x 3)
        pressure_coefficient: Output pressure coefficient (N,)
        n_panels: Number of panels
        freestream_velocity_mag_sq: Magnitude squared of freestream velocity
    """
    for i in range(n_panels):
        v_mag_sq = velocity[i].dot(velocity[i])

        if freestream_velocity_mag_sq > VLM_SMALL_VELOCITY * VLM_SMALL_VELOCITY:
            pressure_coefficient[i] = 1.0 - v_mag_sq / freestream_velocity_mag_sq
        else:
            pressure_coefficient[i] = 0.0


@ti.kernel
def compute_panel_force_coupled(
    velocity: ti.template(),
    vortex_point_position: ti.template(),
    circulation: ti.template(),
    circulation_old: ti.template(),
    V_kinematic: ti.template(),
    forces: ti.template(),
    n_panels: ti.i32,
    density: float,
    smooth_kj: ti.i32,
):
    """
    Compute forces using Kutta-Joukowski theorem on bound vortices.

    F = density * circulation * (V_rel x l)

    Sign convention (confirmed by sign-check instrumentation):
      - Bound leg l = V3 - V2 points in +Y (root→tip)
      - RHS = -n·(V_ext - V_kin): for a plate at positive AoA, RHS < 0
      - aerodynamic_influence_coefficient diagonal < 0  →  γ = RHS/aerodynamic_influence_coefficient > 0  (positive for positive AoA)
      - Joukowski: F = ρ Γ (V × l); with γ > 0, V≈+X, l≈+Y: (+X)×(+Y)=+Z ✓

    'velocity' input must be the fluid velocity evaluated at the
    BOUND VORTEX MIDPOINT (not collocation_point point), excluding self-induction.

    'smooth_kj': when 1, applies a 2-step running average γ_eff = 0.5*(γ_n + γ_{n-1})
    before the force integral.  This exactly cancels the 2Δt oscillation mode
    that the explicit VPM-VLM coupling introduces in γ, while leaving the
    time-mean (and therefore steady-state CL) unchanged.
    """
    for i in range(n_panels):
        # Bound leg (spanwise vortex)
        v2 = vortex_point_position[i, ti.i32(1)]
        v3 = vortex_point_position[i, ti.i32(2)]
        bound_leg = v3 - v2
        bound_length = bound_leg.norm()

        force = ti.Vector([0.0, 0.0, 0.0])

        if bound_length > VLM_SMALL_VELOCITY:
            # 2-step running average removes the 2Δt oscillation mode
            # introduced by the explicit VPM-VLM coupling (ForceConfig.kj_smoothing).
            g = 0.5 * (circulation[i] + circulation_old[i]) if smooth_kj == 1 else circulation[i]

            # Relative velocity at bound vortex
            V_kin = ti.Vector(
                [V_kinematic[i, ti.i32(0)], V_kinematic[i, ti.i32(1)], V_kinematic[i, ti.i32(2)]]
            )
            V_rel = velocity[i] - V_kin

            force = density * g * V_rel.cross(bound_leg)

        forces[i] = force
