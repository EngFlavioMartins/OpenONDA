"""Constant triangular source-panel velocity."""

import taichi as ti

from ....config.constants import PANEL_EPSILON


@ti.func
def compute_source_velocity(
    p,
    v0,
    v1,
    v2,
    normal,
):
    """Return the velocity induced by a unit-strength source triangle."""
    r0 = v0 - p
    r1 = v1 - p
    r2 = v2 - p
    m0 = ti.max(r0.norm(), PANEL_EPSILON)
    m1 = ti.max(r1.norm(), PANEL_EPSILON)
    m2 = ti.max(r2.norm(), PANEL_EPSILON)

    det = r0.dot(r1.cross(r2))
    den = m0 * m1 * m2 + r0.dot(r1) * m2 + r1.dot(r2) * m0 + r2.dot(r0) * m1
    omega = 2.0 * ti.atan2(det, den)
    vel = -omega * normal

    edge0 = v1 - v0
    edge1 = v2 - v1
    edge2 = v0 - v2
    length0 = ti.max(edge0.norm(), PANEL_EPSILON)
    length1 = ti.max(edge1.norm(), PANEL_EPSILON)
    length2 = ti.max(edge2.norm(), PANEL_EPSILON)

    log0 = ti.log(
        ti.max(
            (m0 + m1 + length0) / ti.max(m0 + m1 - length0, PANEL_EPSILON),
            PANEL_EPSILON,
        )
    )
    log1 = ti.log(
        ti.max(
            (m1 + m2 + length1) / ti.max(m1 + m2 - length1, PANEL_EPSILON),
            PANEL_EPSILON,
        )
    )
    log2 = ti.log(
        ti.max(
            (m2 + m0 + length2) / ti.max(m2 + m0 - length2, PANEL_EPSILON),
            PANEL_EPSILON,
        )
    )

    vel += edge0.cross(normal) * (log0 / length0)
    vel += edge1.cross(normal) * (log1 / length1)
    vel += edge2.cross(normal) * (log2 / length2)
    return vel / (4.0 * 3.141592653589793)
