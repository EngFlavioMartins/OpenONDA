"""
Biot-Savart kernels library for panel method.
==================
GPU functions (@ti.func) for induced velocity by vortex rings and bound segments.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: February 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import taichi as ti

from ....config.constants import PANEL_EPSILON


@ti.func
def compute_vortex_ring_velocity(
    p: ti.types.vector(3, float),
    v0: ti.types.vector(3, float),
    v1: ti.types.vector(3, float),
    v2: ti.types.vector(3, float),
) -> ti.types.vector(3, float):
    """
    Induced velocity at point P from a triangular vortex ring (v0, v1, v2).
    Doublet strength = 1.0. Velocity = Σ (Biot-Savart segment).
    """
    vel = ti.Vector([0.0, 0.0, 0.0])
    vel += biot_savart_segment(p, v0, v1)
    vel += biot_savart_segment(p, v1, v2)
    vel += biot_savart_segment(p, v2, v0)
    return vel


@ti.func
def biot_savart_segment(
    p: ti.types.vector(3, float), v0: ti.types.vector(3, float), v1: ti.types.vector(3, float)
) -> ti.types.vector(3, float):
    """Biot-Savart induced velocity by a single straight segment [v0, v1]. Strength = 1.0."""
    r1 = p - v0
    r2 = p - v1
    r1xr2 = r1.cross(r2)

    # Clamp distances to prevent division-by-zero near vertices
    dist1 = ti.max(r1.norm(), PANEL_EPSILON)
    dist2 = ti.max(r2.norm(), PANEL_EPSILON)
    r1_dot_r2 = r1.dot(r2)

    # Regularized Biot-Savart: V = (1/4π) · (r1 × r2) · (1/dist1 + 1/dist2) / (dist1·dist2 + r1·r2 + ε)
    # Both numerator and denominator are regularized following Krasny/Rosenhead form.
    K = (
        (1.0 / (4.0 * 3.141592653589793))
        * (1.0 / dist1 + 1.0 / dist2)
        * (1.0 / (dist1 * dist2 + r1_dot_r2 + PANEL_EPSILON))
    )

    return r1xr2 * K


@ti.func
def compute_doublet_potential(
    p: ti.types.vector(3, float),
    v0: ti.types.vector(3, float),
    v1: ti.types.vector(3, float),
    v2: ti.types.vector(3, float),
) -> ti.template():
    """Induced potential phi at point P by a triangular doublet element (v0, v1, v2). Strength = 1.0."""
    # phi = - (1/4π) · SolidAngle
    r1 = v0 - p
    r2 = v1 - p
    r3 = v2 - p
    m1 = r1.norm()
    m2 = r2.norm()
    m3 = r3.norm()
    det = r1.dot(r2.cross(r3))
    den = m1 * m2 * m3 + (r1.dot(r2)) * m3 + (r2.dot(r3)) * m1 + (r3.dot(r1)) * m2
    omega = 2.0 * ti.atan2(det, den)
    return -omega / (4.0 * 3.141592653589793)
