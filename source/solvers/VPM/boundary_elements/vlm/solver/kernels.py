"""
Taichi kernels for VLM geometry updates and wake shedding.
"""

import taichi as ti

# -- Weak-freestream wake-convection heuristics ------------------------------
# When the freestream is weak relative to the blade/section kinematic speed
# (e.g. a hovering or low-advance-ratio rotor), newly shed TE particles would
# otherwise be deposited in (or convected back through) the rotor disk plane.
# These factors add a normal-direction convection component so the wake leaves
# the disk. They are geometry-stabilisation heuristics, NOT a physical model;
# they only engage when |V_ext| is small compared with |V_kin|, and have no
# effect on a well-resolved forward-flight wake. Tuned empirically against the
# momentum-source rotor tutorial.
WAKE_WEAK_FREESTREAM_RATIO = 0.3  # |V_ext| < ratio·|V_kin| ⇒ freestream too weak
WAKE_NORMAL_KICK_FRACTION = 0.5  # normal kick magnitude = fraction·|V_kin|
WAKE_MIN_AXIAL_RATIO = 0.15  # ensure axial convection ≥ ratio·|V_kin|


@ti.kernel
def update_geometry_translating_kernel(
    dX: ti.types.vector(3, float),
    start_idx: ti.i32,
    end_idx: ti.i32,
    # Lattice fields
    corners: ti.template(),
    vortex_points: ti.template(),
    collocation: ti.template(),
    bound_midpoints: ti.template(),
):
    """
    Update VLM geometry for pure translation on a subset of panels.
    Adds dX to all geometry points.

    Note: Unrolled loop to avoid Taichi/Vulkan compilation issues with nested loops.
    """
    for i in range(start_idx, end_idx):
        # Update corners (4 points per panel) - explicit unrolling
        corners[i, 0] += dX
        corners[i, 1] += dX
        corners[i, 2] += dX
        corners[i, 3] += dX

        # Update vortex points (4 points per panel) - explicit unrolling
        vortex_points[i, 0] += dX
        vortex_points[i, 1] += dX
        vortex_points[i, 2] += dX
        vortex_points[i, 3] += dX

        # Update centers
        collocation[i] += dX
        bound_midpoints[i] += dX


@ti.kernel
def update_geometry_rotating_kernel(
    origin: ti.types.vector(3, float),
    R_flat: ti.types.vector(9, float),  # 3x3 rotation matrix flattened
    dX: ti.types.vector(3, float),
    start_idx: ti.i32,
    end_idx: ti.i32,
    # Lattice fields
    corners: ti.template(),
    vortex_points: ti.template(),
    collocation: ti.template(),
    bound_midpoints: ti.template(),
    normals: ti.template(),
):
    """
    Update VLM geometry for rotation + translation on a subset of panels.
    Points are rotated about 'origin' then translated by dX.
    origin_new = origin + dX is handled by caller if needed for next step.

    New pos = R @ (old_pos - origin) + origin + dX
    """
    # Reconstruct matrix
    R = ti.Matrix(
        [
            [R_flat[0], R_flat[1], R_flat[2]],
            [R_flat[3], R_flat[4], R_flat[5]],
            [R_flat[6], R_flat[7], R_flat[8]],
        ]
    )

    for i in range(start_idx, end_idx):
        # Update corners
        for j in ti.static(range(4)):
            p = corners[i, j] - origin
            corners[i, j] = (R @ p) + origin + dX

            p_v = vortex_points[i, j] - origin
            vortex_points[i, j] = (R @ p_v) + origin + dX

        # Update centers
        p_sub = collocation[i] - origin
        collocation[i] = (R @ p_sub) + origin + dX

        p_b = bound_midpoints[i] - origin
        bound_midpoints[i] = (R @ p_b) + origin + dX

        # Update normals (rotate only)
        n = normals[i]
        normals[i] = R @ n


@ti.func
def compute_shedding_velocity(
    X: ti.types.vector(3, float), V_inf: ti.types.vector(3, float)
) -> ti.types.vector(3, float):
    return V_inf


@ti.func
def _v_conv_compute(
    V_convection: ti.types.vector(3, float),
    use_local_velocity: ti.i32,
    V_kin_local: ti.types.vector(3, float),
    V_ext: ti.types.vector(3, float),
    n: ti.types.vector(3, float),
) -> ti.types.vector(3, float):
    """Return wake-particle convection velocity for one TE panel."""
    V_conv = V_convection
    if use_local_velocity == 1:
        V_kin_mag = V_kin_local.norm()
        if V_kin_mag > 1e-10:
            V_conv = V_ext - V_kin_local
            V_ext_mag = V_ext.norm()
            if V_ext_mag < WAKE_WEAK_FREESTREAM_RATIO * V_kin_mag:
                V_conv = -V_kin_local + n * (WAKE_NORMAL_KICK_FRACTION * V_kin_mag)
            else:
                V_axial = n * n.dot(V_conv)
                if V_axial.norm() < WAKE_MIN_AXIAL_RATIO * V_kin_mag:
                    V_conv = V_conv + n * (WAKE_MIN_AXIAL_RATIO * V_kin_mag)
    return V_conv


@ti.func
def _v_kin_for_sizing(
    V_conv: ti.types.vector(3, float),
    use_local_velocity: ti.i32,
    V_kin_local: ti.types.vector(3, float),
) -> float:
    """Characteristic kinematic speed used for particle sizing."""
    speed = V_conv.norm()
    if use_local_velocity == 1:
        kin_speed = V_kin_local.norm()
        if kin_speed > speed:
            speed = kin_speed
    return speed


@ti.func
def _shed_left_particle(
    i: ti.i32,
    left_idx: ti.i32,
    Gamma: float,
    TeL: ti.types.vector(3, float),
    V_part: ti.types.vector(3, float),
    l_te: float,
    V_unit: ti.types.vector(3, float),
    sigma: float,
    vol: float,
    shedding_threshold: float,
    panel_is_mirrored: ti.template(),
    cumulative_circulation: ti.template(),
    panel_group_id: ti.template(),
    wake_positions: ti.template(),
    wake_velocities: ti.template(),
    wake_strengths: ti.template(),
    wake_radii: ti.template(),
    wake_volumes: ti.template(),
    wake_group_ids: ti.template(),
    num_wake: ti.template(),
):
    """Atomically append a left-edge trailing-vortex wake particle."""
    is_at_symmetry_root = 0
    if left_idx != -1 and panel_is_mirrored[i] != panel_is_mirrored[left_idx]:
        is_at_symmetry_root = 1
    if is_at_symmetry_root == 0:
        dGamma_left = Gamma
        is_tip = 1
        if left_idx != -1:
            dGamma_left = Gamma - cumulative_circulation[left_idx]
            is_tip = 0
        shed_left = 1
        if is_tip == 0 and ti.abs(dGamma_left) < shedding_threshold:
            shed_left = 0
        if shed_left == 1:
            idx = ti.atomic_add(num_wake[None], 1)
            if idx < wake_positions.shape[0]:
                wake_positions[idx] = TeL
                wake_velocities[idx] = V_part
                wake_strengths[idx] = -dGamma_left * l_te * V_unit
                wake_radii[idx] = sigma
                wake_volumes[idx] = vol
                wake_group_ids[idx] = panel_group_id[i]


@ti.func
def _shed_right_particle(
    right_idx: ti.i32,
    Gamma: float,
    TeR: ti.types.vector(3, float),
    V_part: ti.types.vector(3, float),
    l_te: float,
    V_unit: ti.types.vector(3, float),
    sigma: float,
    vol: float,
    group_id: ti.i32,
    wake_positions: ti.template(),
    wake_velocities: ti.template(),
    wake_strengths: ti.template(),
    wake_radii: ti.template(),
    wake_volumes: ti.template(),
    wake_group_ids: ti.template(),
    num_wake: ti.template(),
):
    """Atomically append a right-tip trailing-vortex wake particle."""
    if right_idx == -1:
        idx = ti.atomic_add(num_wake[None], 1)
        if idx < wake_positions.shape[0]:
            wake_positions[idx] = TeR
            wake_velocities[idx] = V_part
            wake_strengths[idx] = Gamma * l_te * V_unit
            wake_radii[idx] = sigma
            wake_volumes[idx] = vol
            wake_group_ids[idx] = group_id


@ti.kernel
def shed_wake_particles_kernel(
    num_panels: ti.i32,
    dt: float,
    V_convection: ti.types.vector(3, float),
    V_particle: ti.types.vector(3, float),
    sigma_factor: float,
    shedding_threshold: float,
    # Lattice topology and state
    cumulative_circulation: ti.template(),
    cumulative_circulation_old: ti.template(),
    corners: ti.template(),
    neighbor_indices: ti.template(),
    is_TE_panel: ti.template(),
    panel_is_mirrored: ti.template(),
    panel_group_id: ti.template(),
    kinematic_velocity: ti.template(),
    external_velocity: ti.template(),  # VPM-induced velocity at each panel
    normals: ti.template(),  # Panel normals for axial kick
    use_local_velocity: ti.i32,
    # Outputs
    wake_positions: ti.template(),
    wake_velocities: ti.template(),
    wake_strengths: ti.template(),
    wake_radii: ti.template(),
    wake_volumes: ti.template(),
    wake_group_ids: ti.template(),
    num_wake: ti.template(),
):
    """
    Shed wake particles from trailing edges using CUMULATIVE circulation.

    Convection modes:
    - Global (use_local_velocity=0): All particles use V_convection (forward flight)
    - Local  (use_local_velocity=1): Each TE panel uses a physically-derived
      convection velocity = V_external - V_kinematic (flow velocity in the lab
      frame at the trailing edge). This includes the VPM-induced downwash,
      preventing particles from being placed in the rotor disk plane.
      If external velocity is small (first few steps), a normal-direction
      kick is added to push particles out of the disk plane.

    Strength = dΓ_cumulative * l_te * (trailing direction unit vector)
    """
    for i in range(num_panels):
        if is_TE_panel[i] == 1:
            Gamma = cumulative_circulation[i]
            TeL = corners[i, 3]
            TeR = corners[i, 2]
            V_kin_local = ti.Vector(
                [kinematic_velocity[i, 0], kinematic_velocity[i, 1], kinematic_velocity[i, 2]]
            )
            V_conv = _v_conv_compute(
                V_convection, use_local_velocity, V_kin_local, external_velocity[i], normals[i]
            )
            V_mag = V_conv.norm()
            if V_mag < 1e-12:
                continue
            V_unit = V_conv / V_mag
            V_kin_for_sizing = _v_kin_for_sizing(V_conv, use_local_velocity, V_kin_local)
            span_vec = TeR - TeL
            span_mag = span_vec.norm()
            l_te = V_kin_for_sizing * dt
            # Trailing-vortex particle core tracks the LOCAL spanwise edge spacing
            # (span_mag), floored at one streamwise step (l_te) to keep the shed sheet
            # connected. The previous `sigma_factor * V * dt` floor over-sized the core on
            # fine-spanwise meshes (e.g. geometric tip clustering, where span_mag drops
            # below sigma_factor*V*dt): adjacent trailing particles then over-overlapped and
            # smeared the spanwise dΓ/dy gradient, under-resolving the tip downwash and
            # producing a spurious ~5% bound-circulation overshoot one-to-two panels inboard
            # of the tip. Sizing the core to span_mag gives a uniform overlap (~1) across the
            # span and removes the artifact without changing the integrated load.
            sigma_trailing = ti.max(l_te, span_mag)
            sigma_transverse = ti.max(sigma_factor * V_kin_for_sizing * dt, span_mag / 3.0)
            if sigma_transverse < 1e-10:
                sigma_transverse = span_mag
            vol = 3.14159 * (span_mag / 2.0) ** 2 * V_kin_for_sizing * dt
            V_part = V_particle
            if use_local_velocity == 1:
                V_part = V_conv
            _shed_left_particle(
                i,
                neighbor_indices[i, 0],
                Gamma,
                TeL,
                V_part,
                l_te,
                V_unit,
                sigma_trailing,
                vol,
                shedding_threshold,
                panel_is_mirrored,
                cumulative_circulation,
                panel_group_id,
                wake_positions,
                wake_velocities,
                wake_strengths,
                wake_radii,
                wake_volumes,
                wake_group_ids,
                num_wake,
            )
            _shed_right_particle(
                neighbor_indices[i, 1],
                Gamma,
                TeR,
                V_part,
                l_te,
                V_unit,
                sigma_trailing,
                vol,
                panel_group_id[i],
                wake_positions,
                wake_velocities,
                wake_strengths,
                wake_radii,
                wake_volumes,
                wake_group_ids,
                num_wake,
            )
            delta_Gamma = cumulative_circulation[i] - cumulative_circulation_old[i]
            idx = ti.atomic_add(num_wake[None], 1)
            if idx < wake_positions.shape[0]:
                wake_positions[idx] = 0.5 * (TeL + TeR)
                wake_velocities[idx] = V_part
                wake_strengths[idx] = -delta_Gamma * span_mag * span_vec / span_mag
                wake_radii[idx] = sigma_transverse
                wake_volumes[idx] = vol
                wake_group_ids[idx] = panel_group_id[i]
