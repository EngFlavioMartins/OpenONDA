"""VLM wake: the shedding kernel, the near-wake row and its implicit coupling to the solve."""

from typing import TYPE_CHECKING, TypeAlias

import numpy as np
import taichi as ti

if TYPE_CHECKING:
    from ..particles.container import Particles
    from ..physics.base import PhysicsBase
    from .solver import VLMSolver

# One near-wake source: (position, core radius, strip-circulation
# coefficients, constant vortex strength).
NearWakeSource: TypeAlias = tuple[np.ndarray, float, np.ndarray, np.ndarray]


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
    core_overlap: float,
    relative_cancellation_tolerance: float,
    cumulative_circulation: ti.template(),
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
                left_radius = core_overlap * ti.max(dl.norm(), span)
                right_radius = core_overlap * ti.max(dr.norm(), span)
                transverse_radius = core_overlap * ti.max(length, span)
                volume = 3.141592653589793 * (span / 2) ** 2 * length
                left_index = neighbor_indices[i, 0]
                right_index = neighbor_indices[i, 1]
                delta_left = gamma
                left_threshold = 0.0
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
                if right_index == -1 and gamma != 0.0:
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
                transverse_strength = transported_bound[i] - gamma * far_span
                if transverse_strength.norm() > 0.0:
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


def compute_cumulative_circulation(vlm: "VLMSolver") -> None:
    """
    Compute cumulative circulation for all panels using segment information.

    This replaces the Taichi kernel version which relies on neighbor_indices
    that may be incorrectly set for tapered/swept wings.

    For each spanwise station, the cumulative circulation is the sum of all
    chordwise panel circulations. This is what gets shed as trailing vortices.
    """
    n_panels = vlm.lattice.n_panels
    circulation = vlm.lattice.circulation.to_numpy()[:n_panels]
    circulation_cumulative = np.zeros(n_panels, dtype=np.float64)

    for block in vlm.lattice.segment_blocks:
        n_chordwise_panels, n_spanwise_panels = block["nc"], block["ns"]
        panels = slice(block["start"], block["start"] + n_chordwise_panels * n_spanwise_panels)
        circulation_cumulative[panels] = np.cumsum(
            circulation[panels].reshape((n_spanwise_panels, n_chordwise_panels)),
            axis=1,
        ).ravel()

    # Upload to Taichi field for kernel access
    dtype_np = np.float32 if vlm.lattice.dtype == ti.f32 else np.float64
    circulation_full = np.zeros(vlm.lattice.max_n_panels, dtype=dtype_np)
    circulation_full[:n_panels] = circulation_cumulative
    vlm.lattice.cumulative_circulation.from_numpy(circulation_full)


def prepare_near_wake(
    vlm: "VLMSolver", time_step_size: float, physics: "PhysicsBase", particles: "Particles"
) -> None:
    """Construct one row from previous TE positions convected to the new clock."""
    n = vlm.lattice.n_panels
    corners = vlm.lattice.panel_corner_position.to_numpy()[:n]
    old = corners if vlm._previous_panel_corners is None else vlm._previous_panel_corners
    indices = np.flatnonzero(vlm.lattice.is_trailing_edge.to_numpy()[:n])
    points = corners[indices][:, [3, 2]].reshape(-1, 3)
    target_velocity = physics.compute_target_velocity(particles, points, include_freestream=True)
    fluid = np.zeros_like(points) if target_velocity is None else target_velocity
    fluid = fluid.reshape(-1, 2, 3)
    offset = vlm.lattice.wake_offset.to_numpy()
    velocity = vlm.lattice.trailing_edge_velocity.to_numpy()
    offset[indices] = old[indices][:, [3, 2]] - corners[indices][:, [3, 2]] + fluid * time_step_size
    velocity[indices] = fluid
    vlm.lattice.wake_offset.from_numpy(offset)
    vlm.lattice.trailing_edge_velocity.from_numpy(velocity)


def shed_wake_particles(vlm: "VLMSolver") -> None:
    """Discretize the completed near-wake row at edge midpoints and its far edge."""
    vlm.lattice.reset_wake_buffer()
    compute_cumulative_circulation(vlm)
    # Antisymmetric signed root circulations cancel to solver roundoff.
    # Do not turn frame-dependent last-bit noise into extra wake particles.
    dtype = np.float32 if vlm.lattice.dtype == ti.f32 else np.float64
    shed_wake_particles_kernel(
        vlm.lattice.n_panels,
        vlm.wake_core_overlap,
        32.0 * np.finfo(dtype).eps,
        vlm.lattice.cumulative_circulation,
        vlm.lattice.panel_corner_position,
        vlm.lattice.neighbor_indices,
        vlm.lattice.is_trailing_edge,
        vlm.lattice.is_mirrored,
        vlm.lattice.group_id,
        vlm.lattice.wake_offset,
        vlm.lattice.trailing_edge_velocity,
        vlm.lattice.wake_position,
        vlm.lattice.wake_velocity,
        vlm.lattice.wake_vortex_strength,
        vlm.lattice.wake_core_radius,
        vlm.lattice.wake_volume,
        vlm.lattice.wake_group_id,
        vlm.lattice.n_wake_particles,
        vlm._transported_bound,
    )

    n_particles_shed = vlm.lattice.n_wake_particles[None]
    wake_buffer_capacity = vlm.lattice.wake_position.shape[0]
    if n_particles_shed > wake_buffer_capacity:
        raise RuntimeError(
            f"VLM wake buffer overflow: {n_particles_shed} particles exceed "
            f"capacity {wake_buffer_capacity}"
        )


def trailing_edge_strip_map(vlm: "VLMSolver") -> tuple[list[int], np.ndarray]:
    """Map every panel to its trailing-edge strip."""
    trailing_edge_indices: list[int] = []
    strip_index_by_panel = np.zeros(vlm.lattice.n_panels, dtype=np.intp)
    for block in vlm.lattice.segment_blocks:
        n_chordwise_panels = block["nc"]
        for spanwise_index in range(block["ns"]):
            first_panel = block["start"] + spanwise_index * n_chordwise_panels
            strip_index_by_panel[first_panel : first_panel + n_chordwise_panels] = len(
                trailing_edge_indices
            )
            trailing_edge_indices.append(first_panel + n_chordwise_panels - 1)
    return trailing_edge_indices, strip_index_by_panel


def _near_wake_row(
    vlm: "VLMSolver", corners: np.ndarray, offsets: np.ndarray, old_bound: np.ndarray
) -> tuple[list[NearWakeSource], np.ndarray, int]:
    """Return affine native-row sources in cumulative strip circulation.

    Each source is (position, radius, strip-vector coefficients, constant
    strength). Strength units are m³/s; circulation coefficients have
    length units. A strip's old closing vector is kept in the inertial
    frame, including transported reaction.
    """
    edges, strip_by_panel = trailing_edge_strip_map(vlm)
    n = vlm.lattice.n_panels
    neighbors = vlm.lattice.neighbor_indices.to_numpy()[:n]
    mirrored = vlm.lattice.is_mirrored.to_numpy()[:n]
    sources: list[NearWakeSource] = []
    for strip, panel in enumerate(edges):
        left, right = corners[panel, 3], corners[panel, 2]
        dl, dr = offsets[panel]
        span = np.linalg.norm(right - left)
        length = 0.5 * (np.linalg.norm(dl) + np.linalg.norm(dr))
        if span <= 1e-12 or length <= 1e-12:
            continue
        left_radius = float(vlm.wake_core_overlap * max(np.linalg.norm(dl), span))
        right_radius = float(vlm.wake_core_overlap * max(np.linalg.norm(dr), span))
        transverse_radius = float(vlm.wake_core_overlap * max(length, span))
        left_index, right_index = neighbors[panel, :2]
        shared_root = left_index != -1 and mirrored[panel] != mirrored[left_index]
        if not shared_root or panel < left_index:
            coefficients = np.zeros((len(edges), 3))
            coefficients[strip] -= dl
            if left_index != -1:
                coefficients[strip_by_panel[left_index]] += -dl if shared_root else dl
            sources.append((left + 0.5 * dl, left_radius, coefficients, np.zeros(3)))
        if right_index == -1:
            coefficients = np.zeros((len(edges), 3))
            coefficients[strip] = dr
            sources.append((right + 0.5 * dr, right_radius, coefficients, np.zeros(3)))
        far_left, far_right = left + dl, right + dr
        coefficients = np.zeros((len(edges), 3))
        coefficients[strip] = far_left - far_right
        sources.append(
            (
                0.5 * (far_left + far_right),
                transverse_radius,
                coefficients,
                old_bound[panel],
            )
        )
    return sources, strip_by_panel, len(edges)


def _near_wake_row_influence(
    vlm: "VLMSolver",
    sources: list[NearWakeSource],
    targets: np.ndarray,
    normals: np.ndarray,
    n_strips: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project an affine native row onto all receiving collocation normals."""
    matrix, old_velocity = np.zeros((len(targets), n_strips)), np.zeros(len(targets))
    for position, radius, coefficients, constant in sources:
        displacement = targets - position
        for strip in np.flatnonzero(np.any(coefficients != 0, axis=1)):
            velocity = vlm._wake_kernel.velocity_pair(
                displacement, coefficients[strip], radius, radius
            )
            matrix[:, strip] += np.einsum("ij,ij->i", velocity, normals)
        if np.any(constant):
            velocity = vlm._wake_kernel.velocity_pair(displacement, constant, radius, radius)
            old_velocity += np.einsum("ij,ij->i", velocity, normals)
    return matrix, old_velocity


def _near_wake_particle_influence(vlm: "VLMSolver") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the accepted row matrix, old-strength RHS and panel/strip map."""
    lattice, n = vlm.lattice, vlm.lattice.n_panels
    corners = lattice.panel_corner_position.to_numpy()[:n]
    offsets = lattice.wake_offset.to_numpy()[:n]
    old_bound = vlm._transported_bound.to_numpy()[:n]
    sources, strip_map, n_strips = _near_wake_row(vlm, corners, offsets, old_bound)
    matrix, old_velocity = _near_wake_row_influence(
        vlm,
        sources,
        lattice.collocation_point.to_numpy()[:n],
        lattice.normal.to_numpy()[:n],
        n_strips,
    )
    return matrix, old_velocity, strip_map


def augment_near_wake_particles(vlm: "VLMSolver") -> None:
    """Include the actual newborn particle row in the circulation solve."""
    matrix, old_velocity, strip_by_panel = _near_wake_particle_influence(vlm)
    n = vlm.lattice.n_panels
    influence = vlm.lattice.aerodynamic_influence_coefficient.to_numpy()
    influence[:n, :n] += matrix[:, strip_by_panel]
    vlm.lattice.aerodynamic_influence_coefficient.from_numpy(influence)
    rhs = vlm.lattice.right_hand_side.to_numpy()
    rhs[:n] -= old_velocity
    vlm.lattice.right_hand_side.from_numpy(rhs)
