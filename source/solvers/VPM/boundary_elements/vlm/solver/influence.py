"""
VLM influence kernels: AIC assembly, RHS construction, and induced-velocity and
pressure-coefficient evaluation.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import taichi as ti

from ....config.constants import VLM_EPSILON, VLM_SMALL_VELOCITY
from ..kernels.biot_savart import bound_vortex_velocity, horseshoe_velocity, vortex_ring_velocity


@ti.kernel
def compute_AIC_matrix(
    collocation: ti.template(),
    vortex_points: ti.template(),
    corners: ti.template(),
    normals: ti.template(),
    is_TE_panel: ti.template(),
    te_index: ti.template(),
    AIC: ti.template(),
    num_panels: ti.i32,
    epsilon: float,
    coupled_mode: ti.i32,
    wake_offset: ti.types.vector(3, float),
):
    """
    Compute aerodynamic influence coefficient matrix.

    AIC[i,j] = normal_i · velocity_at_i_from_horseshoe_j (if not coupled_mode)
    AIC[i,j] = normal_i · velocity_at_i_from_bound_only_j  (if coupled_mode)

    This is the core VLM computation. Each entry represents the normal
    component of velocity induced at collocation point i by a unit
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
        collocation: Collocation points (N x 3)
        vortex_points: Horseshoe vertices (N x 4 x 3)
                       [v1=far_left, v2=bound_left, v3=bound_right, v4=far_right]
        corners: Panel corners (N x 4 x 3) - Kept for API compatibility
        normals: Panel normals (N x 3)
        AIC: Output matrix (N x N)
        num_panels: Number of panels
        epsilon: Regularization parameter
        coupled_mode: If 1, use bound-only induction (coupled VLM-VPM)
    """
    for i, j in ti.ndrange(num_panels, num_panels):
        # Get horseshoe vertices
        v1 = vortex_points[j, ti.i32(0)]
        v2 = vortex_points[j, ti.i32(1)]
        v3 = vortex_points[j, ti.i32(2)]
        v4 = vortex_points[j, ti.i32(3)]

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
            # to taper to zero at the tips. Reading the TE corners of the strip's
            # trailing-edge panel (te_index[j]) extends the legs correctly; for a
            # TE panel te_index[j] == j, so its own behaviour is unchanged.
            te = te_index[j]
            v_te_L = corners[te, ti.i32(3)]
            v_te_R = corners[te, ti.i32(2)]

            # 1. Left internal leg (TE to Bound)
            vel += bound_vortex_velocity(collocation[i], v_te_L, v2, 1.0, epsilon)
            # 2. Bound leg
            vel += bound_vortex_velocity(collocation[i], v2, v3, 1.0, epsilon)
            # 3. Right internal leg (Bound to TE)
            vel += bound_vortex_velocity(collocation[i], v3, v_te_R, 1.0, epsilon)
            # 4. Near-wake panel: extend the trailing legs one convection length
            #    downstream so the filament free ends meet the first wake particle
            #    at TE + wake_offset (implicit near-wake; only for TE panels).
            if is_TE_panel[j] == 1:
                vel += bound_vortex_velocity(
                    collocation[i], v_te_R, v_te_R + wake_offset, 1.0, epsilon
                )
                vel += bound_vortex_velocity(
                    collocation[i], v_te_L + wake_offset, v_te_L, 1.0, epsilon
                )
        else:
            # FULL semi-infinite horseshoe: used for standalone VLM
            vel = horseshoe_velocity(collocation[i], v1, v2, v3, v4, 1.0, epsilon)

        # Normal component (downwash factor)
        AIC[i, j] = vel.dot(normals[i])


@ti.kernel
def compute_RHS(
    collocation: ti.template(),
    normals: ti.template(),
    rhs: ti.template(),
    num_panels: ti.i32,
    V_inf_x: float,
    V_inf_y: float,
    V_inf_z: float,
):
    """
    Compute right-hand side of VLM system.

    RHS[i] = -normal_i · V_inf

    This enforces the boundary condition that total normal velocity
    (freestream + induced) must be zero at each collocation point.

    Args:
        collocation: Collocation points (N x 3)
        normals: Panel normals (N x 3)
        rhs: Output RHS vector (N,)
        num_panels: Number of panels
        V_inf_x, V_inf_y, V_inf_z: Freestream velocity components
    """
    V_inf = ti.Vector([V_inf_x, V_inf_y, V_inf_z])

    for i in range(num_panels):
        # Negative because we move V_inf term to RHS
        rhs[i] = -normals[i].dot(V_inf)


@ti.kernel
def compute_RHS_coupled(
    collocation: ti.template(),
    normals: ti.template(),
    V_external: ti.template(),
    V_kinematic: ti.template(),
    rhs: ti.template(),
    num_panels: ti.i32,
):
    """
    Compute RHS with generic external velocity field and surface motion.

    RHS[i] = -normal[i] · (V_external[i] - V_kinematic[i])

    Boundary condition: (V_external + V_induced - V_kinematic) · n = 0
    Therefore: V_induced · n = - (V_external - V_kinematic) · n

    Args:
        collocation: Collocation points (N x 3)
        normals: Panel normals (N x 3)
        V_external: Total external velocity (background + particles) at collocation (N x 3)
        V_kinematic: Surface velocity at collocation (N x 3)
        rhs: Output RHS vector (N,)
        num_panels: Number of panels
    """
    for i in range(num_panels):
        # Relative flow velocity seen by the surface (excluding induced)
        V_kin = ti.Vector(
            [V_kinematic[i, ti.i32(0)], V_kinematic[i, ti.i32(1)], V_kinematic[i, ti.i32(2)]]
        )
        V_rel_inflow = V_external[i] - V_kin
        rhs[i] = -normals[i].dot(V_rel_inflow)


@ti.kernel
def compute_induced_velocities(
    num_panels: ti.i32,
    collocation: ti.template(),
    vortex_points: ti.template(),
    gamma: ti.template(),
    velocity: ti.template(),
    V_external: ti.template(),
):
    """
    Compute total velocity including external field.

    V_total = V_external + V_induced

    Args:
        num_panels: Number of panels
        collocation: Collocation points (N x 3)
        vortex_points: Horseshoe vertices (N x 4 x 3)
        gamma: Circulation distribution (N,)
        velocity: Output velocity field (N x 3)
        V_external: External velocity field (N x 3)
    """
    for i in range(num_panels):
        vel_induced = ti.Vector([0.0, 0.0, 0.0])

        for j in range(num_panels):
            v1 = vortex_points[j, ti.i32(0)]
            v2 = vortex_points[j, ti.i32(1)]
            v3 = vortex_points[j, ti.i32(2)]
            v4 = vortex_points[j, ti.i32(3)]

            vel_induced += vortex_ring_velocity(
                collocation[i], v1, v2, v3, v4, gamma[j], VLM_EPSILON
            )

        # Total velocity
        velocity[i] = V_external[i] + vel_induced


@ti.func
def _bound_panel_pair_velocity(
    i: int,
    j: int,
    target: ti.template(),
    vortex_points: ti.template(),
    corners: ti.template(),
    te_index: ti.template(),
    g: float,
    coupled_mode: ti.i32,
) -> ti.math.vec3:
    """Compute velocity contribution from panel j at evaluation point on panel i."""
    v2 = vortex_points[j, ti.i32(1)]
    v3 = vortex_points[j, ti.i32(2)]
    vel = ti.Vector([0.0, 0.0, 0.0])
    # Coupled internal trailing legs run to the WING trailing edge (the TE panel of
    # j's chordwise strip), matching compute_AIC_matrix. See note there.
    te = te_index[j]
    if i == j:
        v_te_L = corners[te, ti.i32(3)]
        v_te_R = corners[te, ti.i32(2)]
        if coupled_mode == 1:
            vel += bound_vortex_velocity(target, v_te_L, v2, g, VLM_EPSILON)
            vel += bound_vortex_velocity(target, v3, v_te_R, g, VLM_EPSILON)
        else:
            vel += bound_vortex_velocity(target, vortex_points[j, ti.i32(0)], v2, g, VLM_EPSILON)
            vel += bound_vortex_velocity(target, v3, vortex_points[j, ti.i32(3)], g, VLM_EPSILON)
    else:
        if coupled_mode == 1:
            v_te_L = corners[te, ti.i32(3)]
            v_te_R = corners[te, ti.i32(2)]
            vel += bound_vortex_velocity(target, v_te_L, v2, g, VLM_EPSILON)
            vel += bound_vortex_velocity(target, v2, v3, g, VLM_EPSILON)
            vel += bound_vortex_velocity(target, v3, v_te_R, g, VLM_EPSILON)
        else:
            vel += horseshoe_velocity(
                target,
                vortex_points[j, ti.i32(0)],
                v2,
                v3,
                vortex_points[j, ti.i32(3)],
                g,
                VLM_EPSILON,
            )
    return vel


@ti.kernel
def apply_gamma_smooth(
    gamma: ti.template(),
    gamma_old: ti.template(),
    gamma_smooth: ti.template(),
    n: ti.i32,
):
    for i in range(n):
        gamma_smooth[i] = 0.5 * (gamma[i] + gamma_old[i])


@ti.kernel
def compute_induced_velocities_at_bound(
    num_panels: ti.i32,
    bound_midpoints: ti.template(),
    vortex_points: ti.template(),
    corners: ti.template(),
    te_index: ti.template(),
    gamma: ti.template(),
    velocity: ti.template(),
    V_external: ti.template(),
    coupled_mode: ti.i32,
):
    """
    Compute total velocity AT BOUND VORTEX MIDPOINTS.

    This is used for Kutta-Joukowski force calculation: F = density * Gamma * (V x l)

    Physics notes:
    - Excludes self-induced velocity from the bound leg itself (singular).
    - In standalone mode: includes induction from others + local horseshoe side-legs.
    - In coupled_mode: includes induction from others (bound-only) + VPM wake.

    Args:
        num_panels: Number of panels
        bound_midpoints: Midpoints of bound vortices (N x 3)
        vortex_points: Horseshoe vertices (N x 4 x 3)
        gamma: Circulation distribution (N,)
        velocity: Output velocity field (N x 3)
        V_external: External velocity field at bound midpoints (N x 3)
        coupled_mode: If 1, use bound-only induction consistent with UVLM
    """
    for i in range(num_panels):
        vel_induced = ti.Vector([0.0, 0.0, 0.0])
        target = bound_midpoints[i]

        for j in range(num_panels):
            vel_induced += _bound_panel_pair_velocity(
                i, j, target, vortex_points, corners, te_index, gamma[j], coupled_mode
            )

        # Total velocity = External + Induced
        velocity[i] = V_external[i] + vel_induced


@ti.kernel
def compute_pressure_coefficients(
    velocity: ti.template(), Cp: ti.template(), num_panels: ti.i32, V_inf_mag_sq: float
):
    """
    Compute pressure coefficient at each panel.

    Cp = 1 - (|V|/|V_inf|)²

    Args:
        velocity: Velocity at collocation points (N x 3)
        Cp: Output pressure coefficient (N,)
        num_panels: Number of panels
        V_inf_mag_sq: Magnitude squared of freestream velocity
    """
    for i in range(num_panels):
        v_mag_sq = velocity[i].dot(velocity[i])

        if V_inf_mag_sq > VLM_SMALL_VELOCITY * VLM_SMALL_VELOCITY:
            Cp[i] = 1.0 - v_mag_sq / V_inf_mag_sq
        else:
            Cp[i] = 0.0


@ti.kernel
def compute_panel_forces_coupled(
    velocity: ti.template(),
    vortex_points: ti.template(),
    gamma: ti.template(),
    gamma_old: ti.template(),
    V_kinematic: ti.template(),
    forces: ti.template(),
    num_panels: ti.i32,
    density: float,
    V_ref_mag: float,
    smooth_kj: ti.i32,
):
    """
    Compute forces using Kutta-Joukowski theorem on bound vortices.

    F = density * Gamma * (V_rel x l)

    Sign convention (confirmed by sign-check instrumentation):
      - Bound leg l = V3 - V2 points in +Y (root→tip)
      - RHS = -n·(V_ext - V_kin): for a plate at positive AoA, RHS < 0
      - AIC diagonal < 0  →  γ = RHS/AIC > 0  (positive for positive AoA)
      - Joukowski: F = ρ Γ (V × l); with γ > 0, V≈+X, l≈+Y: (+X)×(+Y)=+Z ✓

    'velocity' input must be the fluid velocity evaluated at the
    BOUND VORTEX MIDPOINT (not collocation point), excluding self-induction.

    'smooth_kj': when 1, applies a 2-step running average γ_eff = 0.5*(γ_n + γ_{n-1})
    before the force integral.  This exactly cancels the 2Δt oscillation mode
    that the explicit VPM-VLM coupling introduces in γ, while leaving the
    time-mean (and therefore steady-state CL) unchanged.
    """
    for i in range(num_panels):
        # Bound leg (spanwise vortex)
        v2 = vortex_points[i, ti.i32(1)]
        v3 = vortex_points[i, ti.i32(2)]
        bound_leg = v3 - v2
        bound_length = bound_leg.norm()

        force = ti.Vector([0.0, 0.0, 0.0])

        if bound_length > VLM_SMALL_VELOCITY:
            # 2-step running average removes the 2Δt oscillation mode
            # introduced by the explicit VPM-VLM coupling (ForceConfig.kj_smoothing).
            g = 0.5 * (gamma[i] + gamma_old[i]) if smooth_kj == 1 else gamma[i]

            # Relative velocity at bound vortex
            V_kin = ti.Vector(
                [V_kinematic[i, ti.i32(0)], V_kinematic[i, ti.i32(1)], V_kinematic[i, ti.i32(2)]]
            )
            V_rel = velocity[i] - V_kin

            force = density * g * V_rel.cross(bound_leg)

        forces[i] = force


@ti.kernel
def compute_panel_forces_impulse_coupled(
    velocity: ti.template(),
    vortex_points: ti.template(),
    gamma: ti.template(),
    gamma_old: ti.template(),
    gamma_old2: ti.template(),
    cumulative_gamma: ti.template(),
    cumulative_gamma_old: ti.template(),
    cumulative_gamma_old2: ti.template(),
    V_kinematic: ti.template(),
    forces: ti.template(),
    areas: ti.template(),
    normals: ti.template(),
    panel_is_mirrored: ti.template(),
    forces_kj: ti.template(),
    forces_unsteady: ti.template(),
    num_panels: ti.i32,
    density: float,
    dt: float,
    vlm_step: ti.i32,
    startup_steps: ti.i32,
    max_order: ti.i32,
    smooth_kj: ti.i32,
):
    """
    Compute forces using Unified method (KJ + Unsteady Bernoulli term).

    F = F_KJ + density * (d(cumulative_Gamma)/dt) * Area * n

    'F_KJ' uses the local horseshoe strength (Gamma).
    'density * d(mu)/dt * Area' uses the local doublet strength (cumulative_Gamma).

    Note on mirror sign correction:
        The VLM horseshoe convention assigns NEGATIVE circulation to mirrored
        panels (bound vortex direction flips). This automatically cancels in
        the KJ term (gamma and bound_leg both flip sign), but the impulse
        term  rho * d(mu)/dt * A * n  depends on the PHYSICAL doublet
        strength, which is always positive. We therefore negate d(mu)/dt for
        mirrored panels so the added-mass force is correctly additive.

    Time differentiation strategy:
        The VPM-VLM explicit coupling introduces a numerical 2Δt oscillation
        in per-panel circulation because each newly shed wake particle
        perturbs the induced velocity at collocation points. BDF1 amplifies
        this mode by 2/Δt; BDF2 is even worse (4/Δt).

        For order=1 (default), the kernel uses a **smoothed backward
        difference** once sufficient history is available (vlm_step >= 2):

            dμ/dt = (μ_n − μ_{n−2}) / (2 Δt)

        This is mathematically equivalent to running-average BDF1 and
        *completely* filters the 2Δt mode while remaining 1st-order accurate.

        Order=2 retains the standard BDF2 stencil for cases where the wake
        is well resolved and 2nd-order temporal accuracy is preferred.

    Args:
        max_order: Maximum BDF order to use (1 or 2).
                   Order 1 = smoothed backward difference (default).
                   Order 2 = BDF1 startup then BDF2.
        smooth_kj: When 1, applies 2-step running average to the KJ base term
                   to cancel the 2Δt coupling oscillation in gamma.
    """
    for i in range(num_panels):
        # 1. Quasi-steady Kutta-Joukowski term
        v2 = vortex_points[i, ti.i32(1)]
        v3 = vortex_points[i, ti.i32(2)]
        bound_leg = v3 - v2
        bound_length = bound_leg.norm()

        f_kj = ti.Vector([0.0, 0.0, 0.0])

        if bound_length > VLM_SMALL_VELOCITY:
            V_kin = ti.Vector(
                [V_kinematic[i, ti.i32(0)], V_kinematic[i, ti.i32(1)], V_kinematic[i, ti.i32(2)]]
            )
            V_rel = velocity[i] - V_kin
            # 2-step running average of γ removes the 2Δt oscillation mode from the
            # KJ base term. The unsteady dμ/dt term already uses smoothed BDF1
            # (μ_n − μ_{n-2})/(2Δt) which independently cancels the same mode.
            g_eff = 0.5 * (gamma[i] + gamma_old[i]) if smooth_kj == 1 else gamma[i]
            f_kj = density * g_eff * V_rel.cross(bound_leg)  # V×l (Joukowski: F=ρΓ(V×l))

        # 2. Unsteady term (Added Mass): + density * (d(cumulative_Gamma)/dt) * Area * n
        dmu_dt = 0.0
        mu = cumulative_gamma[i]
        mu_old = cumulative_gamma_old[i]
        mu_old2 = cumulative_gamma_old2[i]

        # Time-derivative differentiation scheme selection:
        #
        #   vlm_step <= startup_steps  → BDF1  (μ_n − μ_{n−1}) / Δt
        #       History too short for wider stencils AND mu_old2 is still
        #       zero (initial condition), which would produce ~half the
        #       true derivative for a monotone ramp at step 1.
        #
        #   vlm_step > startup_steps, order 1
        #       → Smoothed backward diff  (μ_n − μ_{n−2}) / (2Δt)
        #         Equivalent to running-average BDF1; cancels the 2Δt
        #         oscillation mode introduced by the explicit VPM coupling.
        #
        #   vlm_step > startup_steps, order 2
        #       → BDF2  (3μ_n − 4μ_{n−1} + μ_{n−2}) / (2Δt)
        #         2nd-order accurate; only safe once wake is well-resolved.
        if vlm_step <= startup_steps:
            # Startup: BDF1 until mu_old2 holds two real history values.
            dmu_dt = (mu - mu_old) / dt
        elif max_order >= 2:
            # BDF2: 2nd-order accurate, assumes clean (non-oscillatory) data
            dmu_dt = (3.0 * mu - 4.0 * mu_old + mu_old2) / (2.0 * dt)
        else:
            # Smoothed backward difference: (μ_n − μ_{n−2}) / (2Δt)
            # Equivalent to running-average BDF1; eliminates 2Δt mode.
            dmu_dt = (mu - mu_old2) / (2.0 * dt)

        # Correct for VLM mirror sign convention:
        # Mirrored panels carry negative gamma/mu, but the physical doublet
        # strength (and its time derivative) is positive on both halves.
        mirror_sign = 1.0 - 2.0 * ti.cast(panel_is_mirrored[i], float)
        dmu_dt = dmu_dt * mirror_sign

        f_unsteady = density * dmu_dt * areas[i] * normals[i]

        forces[i] = f_kj + f_unsteady
        forces_kj[i] = f_kj
        forces_unsteady[i] = f_unsteady
