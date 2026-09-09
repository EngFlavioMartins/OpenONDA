"""Pressure-time loads on the physical panels of a horseshoe lattice."""

import taichi as ti


@ti.func
def _triangle_load(a, b, c, pressure, reference):
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
