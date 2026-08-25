"""
Induced-velocity kernel for panel method.
==================
GPU kernel (@ti.kernel) that computes induced velocity from all panel
vortex segments onto arbitrary query points using the Biot-Savart law.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: February 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import taichi as ti

from ....config.constants import PANEL_EPSILON
from .source_velocity import compute_source_velocity


@ti.func
def _segment_velocity(p, a, b):
    r1 = p - a
    r2 = p - b
    r1xr2 = r1.cross(r2)
    d1 = r1.norm()
    d2 = r2.norm()
    denom = d1 * d2 + r1.dot(r2) + PANEL_EPSILON
    coeff = (
        (1.0 / (4.0 * 3.141592653589793))
        * (1.0 / (d1 + PANEL_EPSILON) + 1.0 / (d2 + PANEL_EPSILON))
        * (1.0 / denom)
    )
    return -coeff * r1xr2


@ti.kernel
def compute_induced_velocity_kernel(
    vertex_position: ti.types.ndarray(ndim=3),
    doublet_strength: ti.types.ndarray(ndim=1),
    points: ti.types.ndarray(ndim=2),
    velocity: ti.types.ndarray(ndim=2),
):
    n_queries = points.shape[0]
    n_panels = doublet_strength.shape[0]

    for i in range(n_queries):
        p = ti.Vector([points[i, 0], points[i, 1], points[i, 2]])
        v_total = p * 0.0

        for j in range(n_panels):
            v0 = ti.Vector(
                [vertex_position[j, 0, 0], vertex_position[j, 0, 1], vertex_position[j, 0, 2]]
            )
            v1 = ti.Vector(
                [vertex_position[j, 1, 0], vertex_position[j, 1, 1], vertex_position[j, 1, 2]]
            )
            v2 = ti.Vector(
                [vertex_position[j, 2, 0], vertex_position[j, 2, 1], vertex_position[j, 2, 2]]
            )
            vortex_strength = doublet_strength[j]

            v_total += vortex_strength * (
                _segment_velocity(p, v0, v1)
                + _segment_velocity(p, v1, v2)
                + _segment_velocity(p, v2, v0)
            )

        velocity[i, 0] = v_total[0]
        velocity[i, 1] = v_total[1]
        velocity[i, 2] = v_total[2]


@ti.kernel
def compute_source_induced_velocity_kernel(
    vertex_position: ti.types.ndarray(ndim=3),
    normal: ti.types.ndarray(ndim=2),
    doublet_strength: ti.types.ndarray(ndim=1),
    points: ti.types.ndarray(ndim=2),
    velocity: ti.types.ndarray(ndim=2),
):
    for i in range(points.shape[0]):
        point = ti.Vector([points[i, 0], points[i, 1], points[i, 2]])
        value = point * 0.0
        for j in range(doublet_strength.shape[0]):
            v0 = ti.Vector(
                [vertex_position[j, 0, 0], vertex_position[j, 0, 1], vertex_position[j, 0, 2]]
            )
            v1 = ti.Vector(
                [vertex_position[j, 1, 0], vertex_position[j, 1, 1], vertex_position[j, 1, 2]]
            )
            v2 = ti.Vector(
                [vertex_position[j, 2, 0], vertex_position[j, 2, 1], vertex_position[j, 2, 2]]
            )
            panel_normal = ti.Vector([normal[j, 0], normal[j, 1], normal[j, 2]])
            value += doublet_strength[j] * compute_source_velocity(point, v0, v1, v2, panel_normal)
        velocity[i, 0] = value[0]
        velocity[i, 1] = value[1]
        velocity[i, 2] = value[2]


# The kernels below take the panel lattice and the particle buffers as Taichi
# fields rather than numpy arrays, so a coupled step never copies panel
# geometry or particle state across the host boundary. The ndarray kernels
# above remain for host-side queries at arbitrary probe points.


@ti.kernel
def accumulate_source_panel_velocity_on_field(
    vertex_position: ti.template(),
    normal: ti.template(),
    source_strength: ti.template(),
    target_position: ti.template(),
    target_velocity: ti.template(),
    n_panels: ti.i32,
    n_targets: ti.i32,
):
    """Add constant-source panel velocity to ``target_velocity`` in place."""
    for i in range(n_targets):
        point = target_position[i]
        value = normal[0] * 0.0
        for j in range(n_panels):
            value += source_strength[j] * compute_source_velocity(
                point,
                vertex_position[j, 0],
                vertex_position[j, 1],
                vertex_position[j, 2],
                normal[j],
            )
        target_velocity[i] += value


@ti.kernel
def accumulate_doublet_panel_velocity_on_field(
    vertex_position: ti.template(),
    doublet_strength: ti.template(),
    target_position: ti.template(),
    target_velocity: ti.template(),
    n_panels: ti.i32,
    n_targets: ti.i32,
):
    """Add vortex-ring (doublet) panel velocity to ``target_velocity`` in place."""
    for i in range(n_targets):
        point = target_position[i]
        value = vertex_position[0, 0] * 0.0
        for j in range(n_panels):
            v0 = vertex_position[j, 0]
            v1 = vertex_position[j, 1]
            v2 = vertex_position[j, 2]
            value += doublet_strength[j] * (
                _segment_velocity(point, v0, v1)
                + _segment_velocity(point, v1, v2)
                + _segment_velocity(point, v2, v0)
            )
        target_velocity[i] += value
