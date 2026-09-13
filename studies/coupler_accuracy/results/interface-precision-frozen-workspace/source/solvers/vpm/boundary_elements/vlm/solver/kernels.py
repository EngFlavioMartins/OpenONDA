"""
Taichi kernels for VLM geometry updates and wake shedding.
"""

import taichi as ti


@ti.kernel
def update_geometry_translating_kernel(
    dX: ti.types.vector(3, float),
    start_idx: ti.i32,
    end_idx: ti.i32,
    # Lattice fields
    panel_corner_position: ti.template(),
    vortex_point_position: ti.template(),
    collocation_point: ti.template(),
    bound_vortex_midpoint: ti.template(),
):
    """
    Update VLM geometry for pure translation on a subset of panels.
    Adds dX to all geometry points.

    Note: Unrolled loop to avoid Taichi/Vulkan compilation issues with nested loops.
    """
    for i in range(start_idx, end_idx):
        # Update panel_corner_position (4 points per panel) - explicit unrolling
        panel_corner_position[i, 0] += dX
        panel_corner_position[i, 1] += dX
        panel_corner_position[i, 2] += dX
        panel_corner_position[i, 3] += dX

        # Update vortex points (4 points per panel) - explicit unrolling
        vortex_point_position[i, 0] += dX
        vortex_point_position[i, 1] += dX
        vortex_point_position[i, 2] += dX
        vortex_point_position[i, 3] += dX

        # Update centers
        collocation_point[i] += dX
        bound_vortex_midpoint[i] += dX


@ti.kernel
def update_geometry_rotating_kernel(
    origin: ti.types.vector(3, float),
    R_flat: ti.types.vector(9, float),  # 3x3 rotation matrix flattened
    dX: ti.types.vector(3, float),
    start_idx: ti.i32,
    end_idx: ti.i32,
    # Lattice fields
    panel_corner_position: ti.template(),
    vortex_point_position: ti.template(),
    collocation_point: ti.template(),
    bound_vortex_midpoint: ti.template(),
    normal: ti.template(),
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
        # Update panel_corner_position
        for j in ti.static(range(4)):
            p = panel_corner_position[i, j] - origin
            panel_corner_position[i, j] = (R @ p) + origin + dX

            p_v = vortex_point_position[i, j] - origin
            vortex_point_position[i, j] = (R @ p_v) + origin + dX

        # Update centers
        p_sub = collocation_point[i] - origin
        collocation_point[i] = (R @ p_sub) + origin + dX

        p_b = bound_vortex_midpoint[i] - origin
        bound_vortex_midpoint[i] = (R @ p_b) + origin + dX

        # Update normal (rotate only)
        n = normal[i]
        normal[i] = R @ n


@ti.func
def _append_wake_particle(
    position,
    velocity,
    strength,
    radius,
    volume,
    group,
    wake_position: ti.template(),
    wake_velocity: ti.template(),
    wake_vortex_strength: ti.template(),
    wake_core_radius: ti.template(),
    wake_volume: ti.template(),
    wake_group_id: ti.template(),
    n_wake: ti.template(),
):
    """Reserve one wake slot atomically; retain the count on capacity overflow.

    The caller checks that count before transferring any truncated wake.
    Strength is the integrated vorticity vector (m³/s), not circulation alone.
    """
    index = ti.atomic_add(n_wake[None], 1)
    if index < wake_position.shape[0]:
        wake_position[index] = position
        wake_velocity[index] = velocity
        wake_vortex_strength[index] = strength
        wake_core_radius[index] = radius
        wake_volume[index] = volume
        wake_group_id[index] = group


@ti.kernel
def shed_wake_particles_kernel(
    n_panels: ti.i32,
    sigma_factor: float,
    core_overlap: float,
    shedding_threshold: float,
    relative_cancellation_tolerance: float,
    cumulative_circulation: ti.template(),
    cumulative_circulation_old: ti.template(),
    panel_corner_position: ti.template(),
    neighbor_indices: ti.template(),
    is_trailing_edge: ti.template(),
    is_mirrored: ti.template(),
    group_id: ti.template(),
    wake_offset: ti.template(),
    trailing_edge_velocity: ti.template(),
    wake_position: ti.template(),
    wake_velocity: ti.template(),
    wake_vortex_strength: ti.template(),
    wake_core_radius: ti.template(),
    wake_volume: ti.template(),
    wake_group_id: ti.template(),
    n_wake: ti.template(),
    transported_bound: ti.template(),
    use_bound_transport: ti.template(),
):
    """Deposit the completed row as trailing midpoints and a transverse far-edge element.

    Offsets run from the new body edge to the old edge convected with local fluid
    velocity. The same offsets define the implicit circulation system. Each
    interior trailing filament is shared by adjacent strips and emitted once.
    """
    for i in range(n_panels):
        if is_trailing_edge[i] == 1:
            gamma = cumulative_circulation[i]
            left = panel_corner_position[i, 3]
            right = panel_corner_position[i, 2]
            dl, dr = wake_offset[i, 0], wake_offset[i, 1]
            ul, ur = trailing_edge_velocity[i, 0], trailing_edge_velocity[i, 1]
            span = (right - left).norm()
            length = 0.5 * (dl.norm() + dr.norm())
            if span > 1e-12 and length > 1e-12:
                left_radius = ti.max(dl.norm(), span)
                right_radius = ti.max(dr.norm(), span)
                transverse_radius = ti.max(sigma_factor * length, span / 3)
                if core_overlap > 0:
                    left_radius *= core_overlap
                    right_radius *= core_overlap
                    transverse_radius = core_overlap * ti.max(length, span)
                volume = 3.141592653589793 * (span / 2) ** 2 * length
                left_index = neighbor_indices[i, 0]
                right_index = neighbor_indices[i, 1]
                delta_left = gamma
                left_threshold = shedding_threshold
                shared_root = 0
                owns_left = 1
                if left_index != -1:
                    shared_root = ti.cast(is_mirrored[i] != is_mirrored[left_index], ti.i32)
                    if shared_root == 1:
                        # Both mirrored halves point root-to-tip, so their
                        # signed circulations add on the common root edge.
                        delta_left += cumulative_circulation[left_index]
                        left_threshold = ti.max(
                            left_threshold,
                            relative_cancellation_tolerance
                            * (ti.abs(gamma) + ti.abs(cumulative_circulation[left_index])),
                        )
                        owns_left = ti.cast(i < left_index, ti.i32)
                    else:
                        delta_left -= cumulative_circulation[left_index]
                if owns_left == 1 and ti.abs(delta_left) > left_threshold:
                    _append_wake_particle(
                        left + 0.5 * dl,
                        ul,
                        -delta_left * dl,
                        left_radius,
                        volume,
                        group_id[i],
                        wake_position,
                        wake_velocity,
                        wake_vortex_strength,
                        wake_core_radius,
                        wake_volume,
                        wake_group_id,
                        n_wake,
                    )
                if right_index == -1 and ti.abs(gamma) > shedding_threshold:
                    _append_wake_particle(
                        right + 0.5 * dr,
                        ur,
                        gamma * dr,
                        right_radius,
                        volume,
                        group_id[i],
                        wake_position,
                        wake_velocity,
                        wake_vortex_strength,
                        wake_core_radius,
                        wake_volume,
                        wake_group_id,
                        n_wake,
                    )
                far_left, far_right = left + dl, right + dr
                far_span = far_right - far_left
                transverse_strength = (cumulative_circulation_old[i] - gamma) * far_span
                if ti.static(use_bound_transport):
                    transverse_strength = transported_bound[i] - gamma * far_span
                # Retain the circulation threshold's units (m²/s), including
                # exchange vectors that are not parallel to the far edge.
                if transverse_strength.norm() > shedding_threshold * far_span.norm():
                    _append_wake_particle(
                        0.5 * (far_left + far_right),
                        0.5 * (ul + ur),
                        transverse_strength,
                        transverse_radius,
                        volume,
                        group_id[i],
                        wake_position,
                        wake_velocity,
                        wake_vortex_strength,
                        wake_core_radius,
                        wake_volume,
                        wake_group_id,
                        n_wake,
                    )
