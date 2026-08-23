"""
Source Panel Potential Kernel
==============================
Exact analytical formula for the potential induced by a constant-strength
source triangle at a point P.

Reference: Katz & Plotkin, "Low-Speed Aerodynamics", Eq. 10.22

Author: Flavio A. C. Martins, OpenONDA Team
Date: June 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import taichi as ti

from ....config.constants import PANEL_EPSILON


@ti.func
def compute_source_potential(
    p: ti.types.vector(3, float),
    v0: ti.types.vector(3, float),
    v1: ti.types.vector(3, float),
    v2: ti.types.vector(3, float),
    normal: ti.types.vector(3, float),
) -> float:
    """
    Exact analytical potential induced by a constant-strength source triangle at point P.

    The triangle vertex_position are (v0, v1, v2) with unit normal vector.
    Source strength σ = 1.0 (unit strength).

    Formula (Katz & Plotkin, Eq. 10.22):
        φ = -(1 / 4π) · [ h · Ω + Σ_edges (r₀ · ln_term - L) ]

    where:
        h = perpendicular distance from P to panel plane
        Ω = solid angle subtended by triangle at P
        L = edge length
        r₀, r₁ = distances from P to edge endpoints
        ln_term = ln((r₁ + r₀ + L) / (r₁ + r₀ - L))

    Parameters
    ----------
    p : ti.types.vector(3, float)
        Evaluation point
    v0, v1, v2 : ti.types.vector(3, float)
        Triangle vertex_position (counter-clockwise when viewed from outside)
    normal : ti.types.vector(3, float)
        Unit normal vector of the triangle

    Returns
    -------
    float
        Potential at point P due to unit-strength source panel
    """
    # Compute perpendicular distance h from P to panel plane
    # Panel plane passes through v0 with normal vector
    r_p0 = p - v0
    h = r_p0.dot(normal)

    # Compute solid angle Ω subtended by triangle at P
    # Using the formula from Katz & Plotkin, Eq. 10.21
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

    # Solid angle formula:
    # tan(Ω/2) = (r0 · (r1 × r2)) / (m0·m1·m2 + (r0·r1)·m2 + (r1·r2)·m0 + (r2·r0)·m1)
    det = r0.dot(r1.cross(r2))
    den = m0 * m1 * m2 + r0.dot(r1) * m2 + r1.dot(r2) * m0 + r2.dot(r0) * m1

    # Use atan2 for numerical stability
    omega = 2.0 * ti.atan2(det, den)

    # Compute edge contributions
    # Edge 0: v0 -> v1
    edge0 = v1 - v0
    L0 = edge0.norm()
    L0 = ti.max(L0, PANEL_EPSILON)
    ln_term0 = ti.log(ti.max((m1 + m0 + L0) / (m1 + m0 - L0 + PANEL_EPSILON), PANEL_EPSILON))
    edge_contrib0 = m0 * ln_term0

    # Edge 1: v1 -> v2
    edge1 = v2 - v1
    L1 = edge1.norm()
    L1 = ti.max(L1, PANEL_EPSILON)
    ln_term1 = ti.log(ti.max((m2 + m1 + L1) / (m2 + m1 - L1 + PANEL_EPSILON), PANEL_EPSILON))
    edge_contrib1 = m1 * ln_term1

    # Edge 2: v2 -> v0
    edge2 = v0 - v2
    L2 = edge2.norm()
    L2 = ti.max(L2, PANEL_EPSILON)
    ln_term2 = ti.log(ti.max((m0 + m2 + L2) / (m0 + m2 - L2 + PANEL_EPSILON), PANEL_EPSILON))
    edge_contrib2 = m2 * ln_term2

    # Total potential
    phi = -(1.0 / (4.0 * 3.141592653589793)) * (
        h * omega + edge_contrib0 + edge_contrib1 + edge_contrib2
    )

    return phi
