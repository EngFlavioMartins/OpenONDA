"""VLM loads: pressure-time loads on the panels and the Kutta-Joukowski force evaluator."""

from typing import TYPE_CHECKING

import numpy as np
import taichi as ti

from openonda.core.kinematics import RigidMotion

from .filament import (
    compute_induced_velocities,
    compute_induced_velocities_at_bound,
    compute_panel_force_coupled,
)
from .wake import compute_cumulative_circulation

if TYPE_CHECKING:
    from .solver import VLMSolver


@ti.func
def _triangle_load(a, b, c, pressure, reference):
    """Return oriented triangle pressure force and moment about reference (SI)."""
    force = 0.5 * pressure * (b - a).cross(c - a)
    moment = ((a + b + c) / 3.0 - reference).cross(force)
    return force, moment


@ti.kernel
def add_unsteady_pressure_loads(
    corners: ti.template(),
    vortex: ti.template(),
    circulation: ti.template(),
    old_circulation: ti.template(),
    cumulative: ti.template(),
    old_cumulative: ti.template(),
    forces: ti.template(),
    unsteady_forces: ti.template(),
    moment_correction: ti.template(),
    pressure_jump_coefficient: ti.template(),
    n_panels: ti.i32,
    density_over_dt: float,
    inverse_dynamic_pressure: float,
):
    """Add rho*d(potential jump)/dt with exact piecewise-planar moments.

    The bound line divides each physical panel: its fore portion carries the
    upstream cumulative jump; its aft portion also includes the local horseshoe
    increment. Signed circulation and oriented triangles handle both halves of
    a mirrored surface without absolute values or a preferred world axis.
    """
    for i in range(n_panels):
        a, b = vortex[i, 1], vortex[i, 2]
        p, q = corners[i, 0], corners[i, 1]
        r, s = corners[i, 2], corners[i, 3]
        reference = 0.5 * (a + b)
        after = density_over_dt * (cumulative[i] - old_cumulative[i])
        before = after - density_over_dt * (circulation[i] - old_circulation[i])
        f1, m1 = _triangle_load(p, a, b, before, reference)
        f2, m2 = _triangle_load(p, b, q, before, reference)
        f3, m3 = _triangle_load(a, s, r, after, reference)
        f4, m4 = _triangle_load(a, r, b, after, reference)
        unsteady_forces[i] = f1 + f2 + f3 + f4
        moment_correction[i] = m1 + m2 + m3 + m4
        forces[i] += unsteady_forces[i]
        area_vector = 0.5 * (r - p).cross(q - s)
        area_squared = area_vector.dot(area_vector)
        pressure_jump_coefficient[i] = 0.0
        if area_squared > 0.0:
            pressure_jump_coefficient[i] = (
                unsteady_forces[i].dot(area_vector) / area_squared * inverse_dynamic_pressure
            )


def lift_direction(reference_direction: np.ndarray) -> np.ndarray:
    """Return the unit wind-axis lift direction for the unit ``reference_direction``.

    It is the vertical (z) direction with its component along the reference
    direction removed; a reference direction along z leaves the vertical itself.
    """
    vertical_direction = np.array([0.0, 0.0, 1.0])
    direction = (
        vertical_direction - np.dot(vertical_direction, reference_direction) * reference_direction
    )
    magnitude = np.linalg.norm(direction)
    return direction / magnitude if magnitude > 1e-10 else vertical_direction


def resolve_reference_velocity(
    vlm_solver: "VLMSolver",
    explicit: np.ndarray | None = None,
    background: np.ndarray | None = None,
) -> np.ndarray:
    """Return the velocity of the flow relative to the surfaces [m/s].

    Force coefficients and wind axes refer to it. In order, it is ``explicit``;
    the freestream declared in the VLM configuration; ``background``, the flow
    velocity of the caller, when it is not zero; and otherwise the flow minus
    the representative velocity of the moving surfaces. Without ``background``
    the flow is the mean velocity at the collocation points; with it, the
    surfaces' velocity is subtracted from zero.

    Raises
    ------
    ValueError
        If nothing is declared and the surfaces do not move relative to a
        still flow, so that the reference velocity is zero.
    """
    if explicit is not None:
        return np.asarray(explicit, dtype=float)
    freestream = vlm_solver.freestream_velocity
    if freestream is not None and np.linalg.norm(freestream) > 1e-10:
        return freestream
    if background is not None and np.linalg.norm(background) > 1e-10:
        return np.array(background)
    kinematic_velocity = vlm_solver._get_active_kinematic_velocity()
    if background is not None:
        resolved = -kinematic_velocity
    else:
        n_panels = vlm_solver.lattice.n_panels
        mean_flow = (
            np.mean(vlm_solver.lattice.external_velocity.to_numpy()[:n_panels], axis=0)
            if n_panels > 0
            else np.zeros(3)
        )
        resolved = mean_flow - kinematic_velocity
    if np.linalg.norm(resolved) <= 1e-10:
        raise ValueError(
            "The VLM reference velocity is zero: no reference velocity was given, "
            "VLMConfig.freestream_velocity is not set, and the surfaces do not move "
            "relative to a still flow. Set VLMConfig.freestream_velocity or pass a reference velocity."
        )
    return resolved


def kutta_joukowski_forces(
    vlm_solver: "VLMSolver", density: float, reference_speed: float | None
) -> dict[str, np.ndarray | float | str]:
    """Return bound-leg KJ loads plus the configured unsteady pressure term."""
    if not vlm_solver._solved:
        raise RuntimeError("Force evaluation requires a solved VLM lattice")
    reference = resolve_reference_velocity(vlm_solver)
    if reference_speed is not None:
        reference = reference / np.linalg.norm(reference) * reference_speed
    forces = vlm_solver.compute_forces(density, reference_velocity=reference)
    return {
        "unsteady_pressure": vlm_solver.force.unsteady,
        "force": np.array([forces["force_x"], forces["force_y"], forces["force_z"]]),
        **forces,
    }


def panel_kinematic_velocity(
    kinematics: RigidMotion,
    rotation_centre: np.ndarray,
    time: float,
    collocation_point: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> np.ndarray:
    """Return the rigid-body velocity ``V + Omega x (x - c)`` of a panel slice."""
    angular_velocity = kinematics.get_angular_velocity(time)
    translational_velocity = kinematics.get_velocity(time)
    radial_position = collocation_point[start_idx:end_idx] - rotation_centre
    return translational_velocity + np.cross(angular_velocity, radial_position)


def compute_postprocess(
    vlm: "VLMSolver",
    external_velocity: np.ndarray,
    reference_velocity: np.ndarray,
    density: float,
    time_step_size: float | None = None,
    coupled: bool = False,
    bound_external_velocity: np.ndarray | None = None,
) -> None:
    """
    Compute derived quantities (velocity, pressures, forces).

    Args:
       external_velocity: External velocity (N, 3).
       reference_velocity: Reference velocity vector [ux, uy, uz] (m/s).
       density: Fluid density
       time_step_size: Time step size
       coupled: Whether in coupled mode (bound only aerodynamic_influence_coefficient)
    """
    if vlm.force.unsteady and (
        time_step_size is None or not np.isfinite(time_step_size) or time_step_size <= 0
    ):
        raise ValueError("Unsteady VLM pressure requires a finite positive time_step_size")
    vlm._force_density = density
    vlm.lattice.force_density = float(density)
    vlm.lattice.set_external_velocity(external_velocity)

    reference_velocity_magnitude = float(np.linalg.norm(reference_velocity))
    if reference_velocity_magnitude < 1e-10:
        reference_velocity_magnitude = 1.0

    # 0. Keep cumulative circulation current for wake diagnostics.
    compute_cumulative_circulation(vlm)

    # 1. Compute velocity at collocation points.
    compute_induced_velocities(
        vlm.lattice.n_panels,
        vlm.lattice.collocation_point,
        vlm.lattice.vortex_point_position,
        vlm.lattice.panel_corner_position,
        vlm.lattice.trailing_edge_index,
        vlm.lattice.circulation,
        vlm.lattice.velocity,
        vlm.lattice.external_velocity,
        int(coupled),
        vlm.lattice.filament_core_radius,
    )
    velocity = vlm.lattice.velocity.to_numpy()[: vlm.lattice.n_panels]
    kinematic_velocity = vlm.lattice.get_kinematic_velocity()
    relative_velocity = np.zeros((vlm.max_n_panels, 3), dtype=vlm.lattice.np_dtype)
    relative_velocity[: vlm.lattice.n_panels] = velocity - kinematic_velocity
    vlm.lattice.relative_velocity.from_numpy(relative_velocity)

    # 2. Compute velocity at BOUND VORTEX midpoints (for Forces)
    incident_at_bound = (
        external_velocity if bound_external_velocity is None else bound_external_velocity
    )
    padded = np.zeros((vlm.max_n_panels, 3), dtype=vlm.lattice.np_dtype)
    padded[: vlm.lattice.n_panels] = incident_at_bound
    vlm.lattice.bound_external_velocity.from_numpy(padded)
    bound_kinematics = np.zeros_like(padded)
    points = vlm.lattice.bound_vortex_midpoint.to_numpy()[: vlm.lattice.n_panels]
    for name, (_, motion) in vlm.surfaces.items():
        panels = vlm._surface_panels[name]
        bound_kinematics[panels] = panel_kinematic_velocity(
            motion,
            vlm.surface_rotation_centre(name),
            vlm._current_time or 0.0,
            points,
            panels.start,
            panels.stop,
        )
    vlm.lattice.bound_kinematic_velocity.from_numpy(bound_kinematics)
    vlm.lattice.reference_speed = float(reference_velocity_magnitude)
    compute_induced_velocities_at_bound(
        vlm.lattice.n_panels,
        vlm.lattice.bound_vortex_midpoint,
        vlm.lattice.vortex_point_position,
        vlm.lattice.panel_corner_position,
        vlm.lattice.trailing_edge_index,
        vlm.lattice.circulation,
        vlm.lattice.bound_vortex_velocity,
        vlm.lattice.bound_external_velocity,
        1 if coupled else 0,
        vlm.lattice.filament_core_radius,
    )
    bound_velocity = vlm.lattice.bound_vortex_velocity.to_numpy()[: vlm.lattice.n_panels]
    bound_relative_velocity = np.zeros((vlm.max_n_panels, 3), dtype=vlm.lattice.np_dtype)
    bound_relative_velocity[: vlm.lattice.n_panels] = (
        bound_velocity - bound_kinematics[: vlm.lattice.n_panels]
    )
    vlm.lattice.bound_relative_velocity.from_numpy(bound_relative_velocity)

    compute_panel_force_coupled(
        vlm.lattice.bound_vortex_velocity,
        vlm.lattice.vortex_point_position,
        vlm.lattice.circulation,
        vlm.lattice.bound_kinematic_velocity,
        vlm.lattice.panel_force,
        vlm.lattice.n_panels,
        density,
    )
    vlm.lattice.unsteady_panel_force.fill(0.0)
    vlm.lattice.panel_moment_correction.fill(0.0)
    vlm.lattice.unsteady_pressure_jump_coefficient.fill(0.0)
    if vlm.force.unsteady and time_step_size is not None:
        add_unsteady_pressure_loads(
            vlm.lattice.panel_corner_position,
            vlm.lattice.vortex_point_position,
            vlm.lattice.circulation,
            vlm.lattice.circulation_old,
            vlm.lattice.cumulative_circulation,
            vlm.lattice.cumulative_circulation_old,
            vlm.lattice.panel_force,
            vlm.lattice.unsteady_panel_force,
            vlm.lattice.panel_moment_correction,
            vlm.lattice.unsteady_pressure_jump_coefficient,
            vlm.lattice.n_panels,
            density / time_step_size,
            2.0 / (density * reference_velocity_magnitude**2),
        )


def _decompose_wind_axes(
    total_force: np.ndarray,
    reference_velocity_magnitude: float,
    reference_velocity: np.ndarray,
) -> tuple[float, float, float]:
    """Decompose total force into lift, drag, side-force in wind axes."""
    if reference_velocity_magnitude > 1e-10:
        reference_direction = reference_velocity / reference_velocity_magnitude
        lift_axis = lift_direction(reference_direction)
        side_force_direction = np.cross(reference_direction, lift_axis)
        return (
            float(np.dot(total_force, lift_axis)),
            float(np.dot(total_force, reference_direction)),
            float(np.dot(total_force, side_force_direction)),
        )
    force_x, force_y, force_z = total_force
    return -float(force_z), float(force_x), float(force_y)


def _compute_force_moments(
    vlm: "VLMSolver",
    panel_force: np.ndarray,
    panel_moment_correction: np.ndarray,
) -> tuple[tuple[float, float, float], tuple[float, float, float], np.ndarray]:
    """Compute moments about reference center and quarter-chord."""
    bound_vortex_midpoint = vlm.lattice.bound_vortex_midpoint.to_numpy()[: vlm.lattice.n_panels]
    # A single surface carries its placed centroid; several surfaces share
    # the fixed placed centroid of the whole lattice.
    reference_point = vlm._placed_centroid
    if len(vlm.surfaces) == 1:
        pose = next(iter(vlm.surface_poses.values()))
        reference_point = pose.apply(reference_point[np.newaxis])[0]
    total_moment = np.sum(
        np.cross(bound_vortex_midpoint - reference_point, panel_force) + panel_moment_correction,
        axis=0,
    )
    # Derive the quarter-chord point from the current physical strips, so
    # prescribed placement, pitch and translation all transform it with the wing.
    corners = vlm.lattice.panel_corner_position.to_numpy()[: vlm.lattice.n_panels]
    leading = vlm.lattice.is_leading_edge.to_numpy()[: vlm.lattice.n_panels] == 1
    trailing = vlm.lattice.trailing_edge_index.to_numpy()[: vlm.lattice.n_panels][leading]
    leading_midpoint = 0.5 * (corners[leading, 0] + corners[leading, 1])
    trailing_midpoint = 0.5 * (corners[trailing, 2] + corners[trailing, 3])
    strip_chord = np.linalg.norm(trailing_midpoint - leading_midpoint, axis=1)
    strip_span = np.linalg.norm(corners[leading, 1] - corners[leading, 0], axis=1)
    quarter_chord_reference_point = np.average(
        0.75 * leading_midpoint + 0.25 * trailing_midpoint,
        axis=0,
        weights=strip_chord * strip_span,
    )
    quarter_chord_total_moment = np.sum(
        np.cross(bound_vortex_midpoint - quarter_chord_reference_point, panel_force)
        + panel_moment_correction,
        axis=0,
    )
    return (
        tuple(total_moment),
        tuple(quarter_chord_total_moment),
        reference_point,
    )


def compute_forces(
    vlm: "VLMSolver",
    density: float,
    reference_velocity: np.ndarray | None = None,
    reference_area: float | None = None,
    reference_chord: float | None = None,
    reference_span: float | None = None,
) -> dict[str, float | np.ndarray]:
    """
    Compute integrated aerodynamic forces and moments.

    lift_coefficient, drag_coefficient, side_force_coefficient are normalized by the dynamic pressure and reference area:
    lift_coefficient = L / (0.5 * rho * reference_area * reference_velocity²)

    Args:
       density: Fluid density (kg/m³)
       reference_velocity: Reference velocity vector [ux, uy, uz] (for coefficients and L/D axes).
              If None, auto-computed from freestream_velocity or kinematics.
       reference_area: Reference area (m²). If None, uses aircraft defaults.
       reference_chord: Reference chord (m). If None, uses aircraft defaults.
       reference_span: Reference span (m). If None, uses aircraft defaults.

    Returns:
        Dictionary with force components and coefficients
    """
    if not vlm._solved:
        raise RuntimeError("Must solve system before computing forces")
    panel_force = vlm.lattice.get_forces() * (density / vlm._force_density)
    total_force = np.sum(panel_force, axis=0)
    force_x, force_y, force_z = total_force
    reference_velocity = resolve_reference_velocity(vlm, reference_velocity)
    reference_velocity_magnitude = float(np.linalg.norm(reference_velocity))
    lift, drag, side_force = _decompose_wind_axes(
        total_force, reference_velocity_magnitude, reference_velocity
    )
    (moment_x, moment_y, moment_z), quarter_chord_moment, reference_point = _compute_force_moments(
        vlm,
        panel_force,
        vlm.lattice.panel_moment_correction.to_numpy()[: vlm.lattice.n_panels]
        * (density / vlm._force_density),
    )
    dynamic_pressure = 0.5 * density * reference_velocity_magnitude**2
    if reference_area is None:
        reference_area = float(vlm.aircraft.refs.get("area", 1.0))
    if reference_span is None:
        reference_span = float(vlm.aircraft.refs.get("span", 1.0))
    if reference_chord is None:
        reference_chord = float(vlm.aircraft.refs.get("chord", 1.0))
    force_normalization = dynamic_pressure * reference_area
    chord_moment_normalization = force_normalization * reference_chord
    span_moment_normalization = force_normalization * reference_span
    normalized = dynamic_pressure > 1e-10 and force_normalization > 1e-10

    def coefficient(value: float, normalization: float) -> float:
        return value / normalization if normalized else 0.0

    result = {
        "force_x": force_x,
        "force_y": force_y,
        "force_z": force_z,
        "lift": lift,
        "drag": drag,
        "side_force": side_force,
        "moment_x": moment_x,
        "moment_y": moment_y,
        "moment_z": moment_z,
        "force_coefficient_x": coefficient(force_x, force_normalization),
        "force_coefficient_y": coefficient(force_y, force_normalization),
        "force_coefficient_z": coefficient(force_z, force_normalization),
        "lift_coefficient": coefficient(lift, force_normalization),
        "drag_coefficient": coefficient(drag, force_normalization),
        "side_force_coefficient": coefficient(side_force, force_normalization),
        "rolling_moment_coefficient": coefficient(moment_x, span_moment_normalization),
        "pitching_moment_coefficient": coefficient(moment_y, chord_moment_normalization),
        "yawing_moment_coefficient": coefficient(moment_z, span_moment_normalization),
        "rolling_moment_coefficient_quarter_chord": coefficient(
            quarter_chord_moment[0], span_moment_normalization
        ),
        "pitching_moment_coefficient_quarter_chord": coefficient(
            quarter_chord_moment[1], chord_moment_normalization
        ),
        "yawing_moment_coefficient_quarter_chord": coefficient(
            quarter_chord_moment[2], span_moment_normalization
        ),
        "dynamic_pressure": dynamic_pressure,
        "reference_area": reference_area,
        "reference_point": reference_point,
        "reference_chord": reference_chord,
        "reference_span": reference_span,
    }
    unsteady = vlm.lattice.unsteady_panel_force.to_numpy()[: vlm.lattice.n_panels].sum(axis=0)
    unsteady *= density / vlm._force_density
    result.update(
        {
            f"unsteady_force_{axis}": float(value)
            for axis, value in zip("xyz", unsteady, strict=True)
        }
    )
    return result


def compute_total_bound_vortex_strength(vlm: "VLMSolver") -> np.ndarray:
    """
    Compute the integrated vector strength of the bound VLM field.

    A coupled horseshoe contains all three on-wing legs, from its strip's
    left trailing edge to its right trailing edge. Their vectors telescope
    to that endpoint difference, including sweep, taper and twist. The
    standalone diagnostic retains the quarter-chord bound-leg convention.

    Returns:
        np.ndarray: Integrated bound vortex-strength vector [m³/s].

    """
    if not vlm._solved:
        return np.zeros(3)

    n_panels = vlm.lattice.n_panels
    circulation = vlm.lattice.circulation.to_numpy()[:n_panels]
    if vlm._coupled_mode:
        trailing_edges = vlm.lattice.trailing_edge_index.to_numpy()[:n_panels]
        corners = vlm.lattice.panel_corner_position.to_numpy()[trailing_edges]
        l_vec = corners[:, 2] - corners[:, 3]
    else:
        vortex_pts = vlm.lattice.vortex_point_position.to_numpy()[:n_panels]
        l_vec = vortex_pts[:, 2] - vortex_pts[:, 1]

    net_vortex_strength = np.sum(circulation[:, np.newaxis] * l_vec, axis=0)

    return net_vortex_strength


def compute_bound_linear_impulse(vlm: "VLMSolver") -> np.ndarray:
    """Integrate half of ``x cross omega`` over the finite bound field [m⁴/s].

    Coupled mode includes the three on-wing legs, ending at the actual
    trailing edge. Standalone mode uses the same quarter-chord-only
    convention as the bound-strength diagnostic. Density is applied by
    the caller; the bound contribution alone is origin-dependent.
    """
    if not vlm._solved:
        return np.zeros(3)
    n = vlm.lattice.n_panels
    gamma = vlm.lattice.circulation.to_numpy()[:n].astype(np.float64)
    vortex = vlm.lattice.vortex_point_position.to_numpy()[:n].astype(np.float64)
    crosses = np.cross(vortex[:, 1], vortex[:, 2])
    if vlm._coupled_mode:
        trailing = vlm.lattice.trailing_edge_index.to_numpy()[:n]
        corners = vlm.lattice.panel_corner_position.to_numpy()[trailing].astype(np.float64)
        crosses += np.cross(corners[:, 3], vortex[:, 1])
        crosses += np.cross(vortex[:, 2], corners[:, 2])
    return 0.5 * np.sum(gamma[:, None] * crosses, axis=0)


def _compute_one_surface_forces(
    vlm: "VLMSolver",
    surface_name: str,
    panel_force: np.ndarray,
    reference_direction: np.ndarray,
    force_normalization: float,
    panel_moment_correction: np.ndarray,
) -> dict[str, float]:
    """Compute forces, moment about the current pivot, and fluid-on-body power."""
    panels = vlm._surface_panels[surface_name]
    _, motion = vlm.surfaces[surface_name]
    time = vlm._current_time if vlm._current_time is not None else 0.0
    centre = vlm.surface_rotation_centre(surface_name)
    positions = vlm.lattice.bound_vortex_midpoint.to_numpy()[panels]
    forces = panel_force[panels]
    panel_centers = vlm.lattice.panel_corner_position.to_numpy()[panels].mean(axis=1)
    areas = vlm.lattice.area.to_numpy()[panels]
    centroid = np.average(panel_centers, axis=0, weights=areas)
    surface_force = forces.sum(axis=0)
    moment = (np.cross(positions - centre, forces) + panel_moment_correction[panels]).sum(axis=0)
    translation_velocity = motion.get_velocity(time)
    angular_velocity = motion.get_angular_velocity(time)
    rotational_power = float(moment @ angular_velocity)
    translational_power = float(surface_force @ translation_velocity)
    lift, drag, _ = _decompose_wind_axes(surface_force, 1.0, reference_direction)
    result = {
        "lift": lift,
        "drag": drag,
        "lift_coefficient": lift / force_normalization if force_normalization > 1e-10 else 0.0,
        "drag_coefficient": drag / force_normalization if force_normalization > 1e-10 else 0.0,
        "panel_count": panels.stop - panels.start,
        "rotational_power": rotational_power,
        "translational_power": translational_power,
        "power": rotational_power + translational_power,
    }
    for label, vector in (
        ("force", surface_force),
        ("moment", moment),
        ("rotation_centre", centre),
        ("translation_velocity", translation_velocity),
        ("angular_velocity", angular_velocity),
        ("position", vlm.surface_poses[surface_name].translation),
        ("centroid", centroid),
    ):
        result.update(
            {f"{label}_{axis}": float(value) for axis, value in zip("xyz", vector, strict=True)}
        )
    return result


def compute_per_surface_forces(
    vlm: "VLMSolver",
    density: float,
    reference_velocity: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    """
    Compute forces for each individual surface.

    All coefficients are normalized by dynamic pressure and the aircraft
    reference area: lift_coefficient = L / (0.5 * rho * S_ref * reference_velocity²)

    Args:
        density: Fluid density (kg/m³)
        reference_velocity: Reference velocity vector [ux, uy, uz]

    Returns:
        Dictionary mapping surface name to force dictionary
    """
    if not vlm._solved:
        return {}
    panel_force = vlm.lattice.get_forces() * (density / vlm._force_density)
    reference_velocity = resolve_reference_velocity(vlm, reference_velocity)
    reference_velocity_magnitude = float(np.linalg.norm(reference_velocity))
    if reference_velocity_magnitude <= 1e-10:
        raise ValueError("Per-surface forces need a nonzero reference velocity")
    dynamic_pressure = 0.5 * density * reference_velocity_magnitude**2
    force_normalization = dynamic_pressure * float(vlm.aircraft.refs.get("area", 1.0))
    reference_direction = reference_velocity / reference_velocity_magnitude
    moment_correction = vlm.lattice.panel_moment_correction.to_numpy()[: vlm.lattice.n_panels]
    moment_correction *= density / vlm._force_density
    return {
        name: _compute_one_surface_forces(
            vlm,
            name,
            panel_force,
            reference_direction,
            force_normalization,
            moment_correction,
        )
        for name in vlm.surfaces
    }
