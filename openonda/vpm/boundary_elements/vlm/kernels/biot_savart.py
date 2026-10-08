"""
Biot-Savart @ti.func kernels for the VLM: bound, semi-infinite, horseshoe, and
vortex-ring induced velocity.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import taichi as ti

# Import VLM constants from centralized config
from ....config.constants import VLM_EPSILON

# Local aliases used in @ti.func kernels
EPSILON = VLM_EPSILON


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
