"""
Source Panel Velocity Kernel
=============================
Exact analytical formula for the velocity induced by a constant-strength
source triangle at a point P.

Reference: Katz & Plotkin, "Low-Speed Aerodynamics", Eq. 10.23

Author: Flavio A. C. Martins, OpenONDA Team
Date: June 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import taichi as ti

from ....config.constants import PANEL_EPSILON


@ti.func
def compute_source_velocity(
    p: ti.types.vector(3, float),
    v0: ti.types.vector(3, float),
    v1: ti.types.vector(3, float),
    v2: ti.types.vector(3, float),
    normal: ti.types.vector(3, float),
) -> ti.types.vector(3, float):
    """
    Exact analytical velocity induced by a constant-strength source triangle at point P.

    The triangle vertices are (v0, v1, v2) with unit normal vector.
    Source strength σ = 1.0 (unit strength).

    For a point not on the panel:
        V = (1 / 4π) · [ Ω · n + Σ_edges (tᵢ × rᵢ) · ln_term ]

    For a point on the panel (h ≈ 0):
        V = (1/2) · n  (self-induced velocity)

    Parameters
    ----------
    p : ti.types.vector(3, float)
        Evaluation point
    v0, v1, v2 : ti.types.vector(3, float)
        Triangle vertices (counter-clockwise when viewed from outside)
    normal : ti.types.vector(3, float)
        Unit normal vector of the triangle

    Returns
    -------
    ti.types.vector(3, float)
        Velocity at point P due to unit-strength source panel
    """
    # Check if point is on the panel plane
    r_p0 = p - v0
    h = r_p0.dot(normal)

    # Initialize velocity
    vel = ti.Vector([0.0, 0.0, 0.0])

    # For a point on the panel, return self-induced velocity
    # Use a more relaxed tolerance (1e-6) to handle floating-point precision issues
    if ti.abs(h) < 1e-6:
        vel = 0.5 * normal
    else:
        # Compute solid angle Ω subtended by triangle at P
        r0 = v0 - p
        r1 = v1 - p
        r2 = v2 - p

        m0 = r0.norm()
        m1 = r1.norm()
        m2 = r2.norm()

        # Clamp to avoid division by zero
        m0 = ti.max(m0, PANEL_EPSILON)
        m1 = ti.max(m1, PANEL_EPSILON)
        m2 = ti.max(m2, PANEL_EPSILON)

        # Solid angle formula
        det = r0.dot(r1.cross(r2))
        den = m0 * m1 * m2 + r0.dot(r1) * m2 + r1.dot(r2) * m0 + r2.dot(r0) * m1
        omega = 2.0 * ti.atan2(det, den)

        # Initialize velocity with normal component
        vel = omega * normal

        # Edge 0: v0 -> v1
        edge0 = v1 - v0
        L0 = edge0.norm()
        L0 = ti.max(L0, PANEL_EPSILON)
        t0 = edge0 / L0  # unit tangent
        ln_term0 = ti.log(ti.max((m1 + m0 + L0) / (m1 + m0 - L0 + PANEL_EPSILON), PANEL_EPSILON))
        # Cross product: t0 × r0 (r0 is vector from P to v0)
        vel += (t0.cross(r0)) * ln_term0

        # Edge 1: v1 -> v2
        edge1 = v2 - v1
        L1 = edge1.norm()
        L1 = ti.max(L1, PANEL_EPSILON)
        t1 = edge1 / L1
        ln_term1 = ti.log(ti.max((m2 + m1 + L1) / (m2 + m1 - L1 + PANEL_EPSILON), PANEL_EPSILON))
        vel += (t1.cross(r1)) * ln_term1

        # Edge 2: v2 -> v0
        edge2 = v0 - v2
        L2 = edge2.norm()
        L2 = ti.max(L2, PANEL_EPSILON)
        t2 = edge2 / L2
        ln_term2 = ti.log(ti.max((m0 + m2 + L2) / (m0 + m2 - L2 + PANEL_EPSILON), PANEL_EPSILON))
        vel += (t2.cross(r2)) * ln_term2

        # Multiply by 1/(4π)
        vel = vel / (4.0 * 3.141592653589793)

    return vel
