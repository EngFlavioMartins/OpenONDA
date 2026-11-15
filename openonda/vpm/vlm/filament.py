"""
VLM filament kernels: Rosenhead segment and horseshoe velocity, influence-matrix
assembly, right-hand side, induced-velocity and bound-transport kernels.
"""

import warnings

import numpy as np
import scipy.linalg
import taichi as ti

from ..config.constants import VLM_SMALL_VELOCITY
from ..physics.induction.stretching import stretching_rate


@ti.func
def regularized_segment_velocity_and_gradient(target, pa, pb, circulation, core_radius):
    """Integrate a Rosenhead-regularized filament and its target Jacobian.

    ``core_radius`` is a length, not an area. The formula is the exact line
    integral of ``Gamma cross r / (4*pi*(r*r + core_radius**2)**1.5)``.
    The Jacobian uses ``gradient[i,j] = d velocity[i] / d target[j]``.
    It remains finite on the filament and needs no finite-difference probes.
    """
    r1 = target - pa
    r2 = target - pb
    segment = pb - pa
    cross = segment.cross(r1)
    radius_sq = core_radius * core_radius
    d1 = ti.sqrt(r1.dot(r1) + radius_sq)
    d2 = ti.sqrt(r2.dot(r2) + radius_sq)
    denominator = cross.dot(cross) + radius_sq * segment.dot(segment)
    velocity = target * 0.0
    gradient = target.outer_product(target) * 0.0
    if denominator > 0.0 and d1 > 0.0 and d2 > 0.0:
        projection = segment.dot(r1 / d1 - r2 / d2)
        factor = circulation / (4.0 * 3.141592653589793)
        velocity = factor * projection * cross / denominator
        projection_gradient = (
            segment / d1
            - r1 * segment.dot(r1) / (d1 * d1 * d1)
            - segment / d2
            + r2 * segment.dot(r2) / (d2 * d2 * d2)
        )
        for column in ti.static(range(3)):
            unit = target * 0.0
            unit[column] = 1.0
            cross_gradient = segment.cross(unit)
            derivative = factor * (
                (projection_gradient[column] * cross + projection * cross_gradient) / denominator
                - projection
                * cross
                * (2.0 * cross.dot(cross_gradient))
                / (denominator * denominator)
            )
            for row in ti.static(range(3)):
                gradient[row, column] = derivative[row]
    return velocity, gradient


@ti.func
def bound_vortex_velocity(target, pa, pb, circulation: float, epsilon: float):
    """Finite-filament velocity with a Rosenhead core of length ``epsilon``.

    Use the same dimensionally consistent line integral as the particle-stage
    field. Its denominator contains ``epsilon**2 * segment_length**2``;
    adding a length squared directly to a cross-product magnitude squared
    would change the model when geometry units or scale change.
    """
    velocity, _ = regularized_segment_velocity_and_gradient(target, pa, pb, circulation, epsilon)
    return velocity


@ti.func
def horseshoe_velocity(target, v1, v2, v3, v4, circulation: float, epsilon: float):
    """
    Compute velocity induced by a horseshoe vortex at target point.

    Horseshoe consists of three segments:
    1. Left trailing leg: v1 (far downstream) → v2 (bound left)
    2. Bound leg: v2 → v3 (bound right)
    3. Right trailing leg: v3 → v4 (far downstream)

    The far points v1/v4 lie downstream along the prescribed trailing
    directions, so the trailing legs run v1→v2 and v3→v4.
    """
    # Sum contributions from all three legs
    vel_left = bound_vortex_velocity(target, v1, v2, circulation, epsilon)  # Left trailing
    vel_bound = bound_vortex_velocity(target, v2, v3, circulation, epsilon)  # Bound
    vel_right = bound_vortex_velocity(target, v3, v4, circulation, epsilon)  # Right trailing

    vel = vel_left + vel_bound + vel_right

    return vel


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
    core_radius: float,
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
                core_radius,
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
    core_radius: float,
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
                core_radius,
            )
            induced += value
        target_velocity[i] += induced


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
    core_radius: float,
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
        core = ti.max(radius[i], core_radius)
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
    core_radius: float,
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
            vel += bound_vortex_velocity(target, v_te_L, v2, g, core_radius)
            vel += bound_vortex_velocity(target, v3, v_te_R, g, core_radius)
        else:
            vel += bound_vortex_velocity(
                target, vortex_point_position[j, ti.i32(0)], v2, g, core_radius
            )
            vel += bound_vortex_velocity(
                target, v3, vortex_point_position[j, ti.i32(3)], g, core_radius
            )
    else:
        if coupled_mode == 1:
            v_te_L = panel_corner_position[te, ti.i32(3)]
            v_te_R = panel_corner_position[te, ti.i32(2)]
            vel += bound_vortex_velocity(target, v_te_L, v2, g, core_radius)
            vel += bound_vortex_velocity(target, v2, v3, g, core_radius)
            vel += bound_vortex_velocity(target, v3, v_te_R, g, core_radius)
        else:
            vel += horseshoe_velocity(
                target,
                vortex_point_position[j, ti.i32(0)],
                v2,
                v3,
                vortex_point_position[j, ti.i32(3)],
                g,
                core_radius,
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
    core_radius: float,
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
        core_radius: Filament core length of the lattice
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
                core_radius,
            )

        # Total velocity = External + Induced
        velocity[i] = external_velocity[i] + vel_induced


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


class CirculationSolver:
    """Solve ``influence @ circulation = right_hand_side`` with SciPy LU factors.

    Factorization costs O(N³); right-hand sides of an identical matrix reuse
    the factors. A singular or non-finite system raises before either the
    cache or the output circulation is changed.
    """

    def __init__(self) -> None:
        """Start with an empty matrix and factor cache."""
        self._matrix: np.ndarray | None = None
        self._factorization: tuple[np.ndarray, np.ndarray] | None = None

    def solve(
        self, influence: ti.Field, right_hand_side: ti.Field, circulation: ti.Field, n_panels: int
    ) -> None:
        """Solve the active ``n_panels`` system in the precision of the fields.

        Parameters
        ----------
        influence : Taichi field, shape (capacity, capacity)
            Dense influence matrix; entries map panel circulation to the
            collocation-point normal velocity.
        right_hand_side : Taichi field, shape (capacity,)
            Boundary-condition vector.
        circulation : Taichi field, shape (capacity,)
            Solution storage; the active prefix is overwritten and the rest
            set to zero.
        n_panels : int
            Active system size.

        Raises
        ------
        numpy.linalg.LinAlgError
            If the influence matrix is singular or the solution is not finite.
        """
        matrix = influence.to_numpy()[:n_panels, :n_panels]
        rhs = right_hand_side.to_numpy()[:n_panels]
        if self._matrix is None or not np.array_equal(matrix, self._matrix):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", scipy.linalg.LinAlgWarning)
                    factorization = scipy.linalg.lu_factor(matrix)
            except scipy.linalg.LinAlgWarning as error:
                raise np.linalg.LinAlgError("VLM influence matrix is singular") from error
            self._matrix, self._factorization = matrix, factorization
        solution = scipy.linalg.lu_solve(self._factorization, rhs)
        if not np.all(np.isfinite(solution)):
            raise np.linalg.LinAlgError("VLM circulation solve produced non-finite values")
        full = np.zeros(circulation.shape[0], dtype=matrix.dtype)
        full[:n_panels] = solution
        circulation.from_numpy(full)
