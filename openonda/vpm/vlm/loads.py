"""VLM loads: pressure-time loads on the panels and the Kutta-Joukowski force evaluator."""

from typing import TYPE_CHECKING

import numpy as np
import taichi as ti

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
    corners: ti.template(),  # type: ignore[invalid-annotation]
    vortex: ti.template(),  # type: ignore[invalid-annotation]
    circulation: ti.template(),  # type: ignore[invalid-annotation]
    old_circulation: ti.template(),  # type: ignore[invalid-annotation]
    cumulative: ti.template(),  # type: ignore[invalid-annotation]
    old_cumulative: ti.template(),  # type: ignore[invalid-annotation]
    forces: ti.template(),  # type: ignore[invalid-annotation]
    unsteady_forces: ti.template(),  # type: ignore[invalid-annotation]
    moment_correction: ti.template(),  # type: ignore[invalid-annotation]
    pressure_jump_coefficient: ti.template(),  # type: ignore[invalid-annotation]
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
