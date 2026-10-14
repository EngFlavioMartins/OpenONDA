"""
VLM influence kernels: aerodynamic_influence_coefficient assembly, RHS construction, and induced-velocity and
pressure-coefficient evaluation.
"""

import taichi as ti

from ....config.constants import VLM_EPSILON, VLM_SMALL_VELOCITY
from ....physics.induction.stretching import stretching_rate
from ..kernels.biot_savart import (
    bound_vortex_velocity,
    horseshoe_velocity,
    regularized_segment_velocity_and_gradient,
)


@ti.kernel
def compute_aerodynamic_influence_coefficient_matrix(
    collocation_point: ti.template(),  # type: ignore[invalid-annotation]
    vortex_point_position: ti.template(),  # type: ignore[invalid-annotation]
    panel_corner_position: ti.template(),  # type: ignore[invalid-annotation]
    normal: ti.template(),  # type: ignore[invalid-annotation]
    is_trailing_edge: ti.template(),  # type: ignore[invalid-annotation]
    trailing_edge_index: ti.template(),  # type: ignore[invalid-annotation]
    aerodynamic_influence_coefficient: ti.template(),  # type: ignore[invalid-annotation]
    n_panels: ti.i32,
    epsilon: float,
    coupled_mode: ti.i32,
):
    """
    Compute aerodynamic influence coefficient matrix.

    aerodynamic_influence_coefficient[i,j] = normal_i · velocity_at_i_from_horseshoe_j (if not coupled_mode)
    aerodynamic_influence_coefficient[i,j] = normal_i · velocity_at_i_from_bound_only_j  (if coupled_mode)

    This is the core VLM computation. Each entry represents the normal
    component of velocity induced at collocation_point point i by a unit
    circulation vortex on panel j.

    In coupled mode, each chordwise horseshoe ends at its strip's trailing
    edge. The actual newborn VPM row is added separately with its native blob
    kernel, positions and core radii. The transported old particle wake supplies
    the remaining influence through the right-hand side.

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
        else:
            # FULL semi-infinite horseshoe: used for standalone VLM
            vel = horseshoe_velocity(collocation_point[i], v1, v2, v3, v4, 1.0, epsilon)

        # Normal component (downwash factor)
        aerodynamic_influence_coefficient[i, j] = vel.dot(normal[i])


@ti.kernel
def compute_coupled_right_hand_side(
    collocation_point: ti.template(),  # type: ignore[invalid-annotation]
    normal: ti.template(),  # type: ignore[invalid-annotation]
    external_velocity: ti.template(),  # type: ignore[invalid-annotation]
    V_kinematic: ti.template(),  # type: ignore[invalid-annotation]
    right_hand_side: ti.template(),  # type: ignore[invalid-annotation]
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


@ti.func
def panel_velocity_and_gradient(
    target,
    panel,
    vortex_points,
    corners,
    trailing_edge_index,
    circulation,
    coupled_mode,
    core_radius,
):
    """Bound horseshoe in coupled mode; full horseshoe for standalone VLM.

    The free VPM particles own the downstream wake. Do not add far trailing
    legs or a fictitious ring closure to their velocity field.
    """
    left = vortex_points[panel, 0]
    right = vortex_points[panel, 3]
    if coupled_mode == 1:
        te = trailing_edge_index[panel]
        left = corners[te, 3]
        right = corners[te, 2]
    v1, g1 = regularized_segment_velocity_and_gradient(
        target,
        left,
        vortex_points[panel, 1],
        circulation,
        core_radius,
    )
    v2, g2 = regularized_segment_velocity_and_gradient(
        target,
        vortex_points[panel, 1],
        vortex_points[panel, 2],
        circulation,
        core_radius,
    )
    v3, g3 = regularized_segment_velocity_and_gradient(
        target,
        vortex_points[panel, 2],
        right,
        circulation,
        core_radius,
    )
    return v1 + v2 + v3, g1 + g2 + g3


@ti.kernel
def compute_induced_velocities(
    n_panels: ti.i32,
    collocation_point: ti.template(),  # type: ignore[invalid-annotation]
    vortex_point_position: ti.template(),  # type: ignore[invalid-annotation]
    panel_corner_position: ti.template(),  # type: ignore[invalid-annotation]
    trailing_edge_index: ti.template(),  # type: ignore[invalid-annotation]
    circulation: ti.template(),  # type: ignore[invalid-annotation]
    velocity: ti.template(),  # type: ignore[invalid-annotation]
    external_velocity: ti.template(),  # type: ignore[invalid-annotation]
    coupled_mode: ti.i32,
):
    """Evaluate the same bound field used for wake advection at collocation points."""
    for i in range(n_panels):
        induced = collocation_point[i] * 0.0
        for j in range(n_panels):
            value, _ = panel_velocity_and_gradient(
                collocation_point[i],
                j,
                vortex_point_position,
                panel_corner_position,
                trailing_edge_index,
                circulation[j],
                coupled_mode,
                VLM_EPSILON,
            )
            induced += value
        velocity[i] = external_velocity[i] + induced


@ti.kernel
def add_induced_velocity_at_targets(
    target_position: ti.template(),  # type: ignore[invalid-annotation]
    target_velocity: ti.template(),  # type: ignore[invalid-annotation]
    vortex_point_position: ti.template(),  # type: ignore[invalid-annotation]
    panel_corner_position: ti.template(),  # type: ignore[invalid-annotation]
    trailing_edge_index: ti.template(),  # type: ignore[invalid-annotation]
    circulation: ti.template(),  # type: ignore[invalid-annotation]
    n_targets: ti.i32,
    n_panels: ti.i32,
    coupled_mode: ti.i32,
):
    """Accumulate bound velocity at point probes, without a particle-volume filter."""
    for i in range(n_targets):
        induced = target_position[i] * 0.0
        for panel in range(n_panels):
            value, _ = panel_velocity_and_gradient(
                target_position[i],
                panel,
                vortex_point_position,
                panel_corner_position,
                trailing_edge_index,
                circulation[panel],
                coupled_mode,
                VLM_EPSILON,
            )
            induced += value
        target_velocity[i] += induced


@ti.kernel
def add_induced_velocity_and_gradient_at_targets(
    target_position: ti.template(),  # type: ignore[invalid-annotation]
    target_velocity: ti.template(),  # type: ignore[invalid-annotation]
    target_gradient: ti.template(),  # type: ignore[invalid-annotation]
    target_core_radius: ti.template(),  # type: ignore[invalid-annotation]
    vortex_point_position: ti.template(),  # type: ignore[invalid-annotation]
    panel_corner_position: ti.template(),  # type: ignore[invalid-annotation]
    trailing_edge_index: ti.template(),  # type: ignore[invalid-annotation]
    circulation: ti.template(),  # type: ignore[invalid-annotation]
    n_targets: ti.i32,
    n_panels: ti.i32,
    coupled_mode: ti.i32,
):
    """Add particle-volume-filtered velocity and write its analytic Jacobian.

    A particle represents a finite vortex volume. Filter filament induction
    with that target particle's current core radius; using an almost singular
    point derivative at a newly shed filament endpoint is not resolved by the
    particle discretization. The gradient destination is provider-owned scratch
    storage; its caller adds it to the other stage contributions.
    """
    for i in range(n_targets):
        target = target_position[i]
        induced = target * 0.0
        gradient = target.outer_product(target) * 0.0
        radius = ti.max(target_core_radius[i], VLM_EPSILON)
        for panel in range(n_panels):
            value, jacobian = panel_velocity_and_gradient(
                target,
                panel,
                vortex_point_position,
                panel_corner_position,
                trailing_edge_index,
                circulation[panel],
                coupled_mode,
                radius,
            )
            induced += value
            gradient += jacobian
        target_velocity[i] += induced
        target_gradient[i] = gradient


@ti.kernel
def add_stage_rates_with_bound_exchange(
    position: ti.template(),  # type: ignore[invalid-annotation]
    strength: ti.template(),  # type: ignore[invalid-annotation]
    radius: ti.template(),  # type: ignore[invalid-annotation]
    velocity: ti.template(),  # type: ignore[invalid-annotation]
    rate: ti.template(),  # type: ignore[invalid-annotation]
    gradient_out: ti.template(),  # type: ignore[invalid-annotation]
    vortex_points: ti.template(),  # type: ignore[invalid-annotation]
    corners: ti.template(),  # type: ignore[invalid-annotation]
    trailing_edge_index: ti.template(),  # type: ignore[invalid-annotation]
    is_trailing_edge: ti.template(),  # type: ignore[invalid-annotation]
    circulation: ti.template(),  # type: ignore[invalid-annotation]
    stage_exchange: ti.template(),  # type: ignore[invalid-annotation]
    count: ti.i32,
    n_panels: ti.i32,
    coupled_mode: ti.i32,
    mode: ti.i32,
    weighted_dt: float,
    has_gradient: ti.template(),  # type: ignore[invalid-annotation]
):
    """Evaluate bound induction and retain its opposite strip strength exchange.

    Chordwise panels are contiguous within each strip. Accumulate their reaction
    before one atomic update at the trailing edge. Accumulate into a zeroed
    stage field so small f32 increments are not rounded against the much larger
    old bound vector. A diagnostic evaluation passes zero weight.
    """
    for i in range(count):
        target = position[i]
        induced = ti.Vector.zero(velocity.dtype, 3)
        gradient = ti.Matrix.zero(rate.dtype, 3, 3)
        strip_rate = ti.Vector.zero(rate.dtype, 3)
        core = ti.max(radius[i], VLM_EPSILON)
        for panel in range(n_panels):
            value, jacobian = panel_velocity_and_gradient(
                target,
                panel,
                vortex_points,
                corners,
                trailing_edge_index,
                circulation[panel],
                coupled_mode,
                core,
            )
            induced += value
            gradient += jacobian
            if weighted_dt != 0.0:
                strip_rate += stretching_rate(jacobian, strength[i], mode)
                if is_trailing_edge[panel] == 1:
                    for axis in ti.static(range(3)):
                        ti.atomic_add(stage_exchange[panel][axis], -strip_rate[axis])
                    strip_rate = ti.Vector.zero(rate.dtype, 3)
        velocity[i] += induced
        rate[i] += stretching_rate(gradient, strength[i], mode)
        if ti.static(has_gradient):
            gradient_out[i] += gradient


@ti.kernel
def accumulate_bound_transport(
    transported_bound: ti.template(),  # type: ignore[invalid-annotation]
    stage_exchange: ti.template(),  # type: ignore[invalid-annotation]
    weighted_dt: float,
    count: ti.i32,
):
    """Add the completed strip rate once, with its RK weight and step size."""
    for i in range(count):
        transported_bound[i] += weighted_dt * stage_exchange[i]


@ti.kernel
def initialize_bound_transport(
    corners: ti.template(),  # type: ignore[invalid-annotation]
    circulation: ti.template(),  # type: ignore[invalid-annotation]
    is_trailing_edge: ti.template(),  # type: ignore[invalid-annotation]
    transported_bound: ti.template(),  # type: ignore[invalid-annotation]
    count: ti.i32,
):
    """Start each strip's transport from its accepted closing-line vector."""
    for i in range(count):
        transported_bound[i] = ti.Vector.zero(transported_bound.dtype, 3)
        if is_trailing_edge[i] == 1:
            transported_bound[i] = circulation[i] * (corners[i, 2] - corners[i, 3])


@ti.func
def _bound_panel_pair_velocity(
    i: int,
    j: int,
    target: ti.template(),  # type: ignore[invalid-annotation]
    vortex_point_position: ti.template(),  # type: ignore[invalid-annotation]
    panel_corner_position: ti.template(),  # type: ignore[invalid-annotation]
    trailing_edge_index: ti.template(),  # type: ignore[invalid-annotation]
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
def compute_induced_velocities_at_bound(
    n_panels: ti.i32,
    bound_vortex_midpoint: ti.template(),  # type: ignore[invalid-annotation]
    vortex_point_position: ti.template(),  # type: ignore[invalid-annotation]
    panel_corner_position: ti.template(),  # type: ignore[invalid-annotation]
    trailing_edge_index: ti.template(),  # type: ignore[invalid-annotation]
    circulation: ti.template(),  # type: ignore[invalid-annotation]
    velocity: ti.template(),  # type: ignore[invalid-annotation]
    external_velocity: ti.template(),  # type: ignore[invalid-annotation]
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
    velocity: ti.template(),  # type: ignore[invalid-annotation]
    pressure_coefficient: ti.template(),  # type: ignore[invalid-annotation]
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
    velocity: ti.template(),  # type: ignore[invalid-annotation]
    vortex_point_position: ti.template(),  # type: ignore[invalid-annotation]
    circulation: ti.template(),  # type: ignore[invalid-annotation]
    V_kinematic: ti.template(),  # type: ignore[invalid-annotation]
    forces: ti.template(),  # type: ignore[invalid-annotation]
    n_panels: ti.i32,
    density: float,
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

    """
    for i in range(n_panels):
        # Bound leg (spanwise vortex)
        v2 = vortex_point_position[i, ti.i32(1)]
        v3 = vortex_point_position[i, ti.i32(2)]
        bound_leg = v3 - v2
        bound_length = bound_leg.norm()

        force = ti.Vector([0.0, 0.0, 0.0])

        if bound_length > VLM_SMALL_VELOCITY:
            g = circulation[i]

            # Relative velocity at bound vortex
            V_kin = ti.Vector(
                [V_kinematic[i, ti.i32(0)], V_kinematic[i, ti.i32(1)], V_kinematic[i, ti.i32(2)]]
            )
            V_rel = velocity[i] - V_kin

            force = density * g * V_rel.cross(bound_leg)

        forces[i] = force
