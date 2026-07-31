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
    vertices: ti.types.ndarray(ndim=3),
    strengths: ti.types.ndarray(ndim=1),
    points: ti.types.ndarray(ndim=2),
    velocity: ti.types.ndarray(ndim=2),
):
    n_queries = points.shape[0]
    n_panels = strengths.shape[0]

    for i in range(n_queries):
        p = ti.Vector([points[i, 0], points[i, 1], points[i, 2]])
        v_total = ti.Vector([0.0, 0.0, 0.0])

        for j in range(n_panels):
            v0 = ti.Vector([vertices[j, 0, 0], vertices[j, 0, 1], vertices[j, 0, 2]])
            v1 = ti.Vector([vertices[j, 1, 0], vertices[j, 1, 1], vertices[j, 1, 2]])
            v2 = ti.Vector([vertices[j, 2, 0], vertices[j, 2, 1], vertices[j, 2, 2]])
            gamma = strengths[j]

            v_total += gamma * (
                _segment_velocity(p, v0, v1)
                + _segment_velocity(p, v1, v2)
                + _segment_velocity(p, v2, v0)
            )

        velocity[i, 0] = v_total[0]
        velocity[i, 1] = v_total[1]
        velocity[i, 2] = v_total[2]


@ti.kernel
def compute_source_induced_velocity_kernel(
    vertices: ti.types.ndarray(ndim=3),
    normals: ti.types.ndarray(ndim=2),
    strengths: ti.types.ndarray(ndim=1),
    points: ti.types.ndarray(ndim=2),
    velocity: ti.types.ndarray(ndim=2),
):
    for i in range(points.shape[0]):
        point = ti.Vector([points[i, 0], points[i, 1], points[i, 2]])
        value = ti.Vector([0.0, 0.0, 0.0])
        for j in range(strengths.shape[0]):
            v0 = ti.Vector([vertices[j, 0, 0], vertices[j, 0, 1], vertices[j, 0, 2]])
            v1 = ti.Vector([vertices[j, 1, 0], vertices[j, 1, 1], vertices[j, 1, 2]])
            v2 = ti.Vector([vertices[j, 2, 0], vertices[j, 2, 1], vertices[j, 2, 2]])
            normal = ti.Vector([normals[j, 0], normals[j, 1], normals[j, 2]])
            value += strengths[j] * compute_source_velocity(point, v0, v1, v2, normal)
        velocity[i, 0] = value[0]
        velocity[i, 1] = value[1]
        velocity[i, 2] = value[2]
