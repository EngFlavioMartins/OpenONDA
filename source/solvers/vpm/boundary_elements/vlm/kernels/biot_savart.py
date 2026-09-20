"""
Biot-Savart @ti.func kernels for the VLM: bound, semi-infinite, horseshoe, and
vortex-ring induced velocity.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import taichi as ti

# Import VLM constants from centralized config
from ....config.constants import VLM_CUTOFF, VLM_EPSILON

# Local aliases used in @ti.func kernels
EPSILON = VLM_EPSILON
CUTOFF = VLM_CUTOFF


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
def semi_infinite_vortex_velocity(target, p, d, circulation: float, epsilon: float):
    """
    Compute velocity induced by a semi-infinite vortex.

    The vortex starts at point p and extends to infinity in direction d.
    Based on FLOWPanel.jl U_semiinfinite_vortex implementation.

    The vortex is split into:
    1. A bound section from p to projection point p0
    2. A semi-infinite section from p0 to infinity

    Args:
        target: Point where velocity is evaluated
        p: Starting point of semi-infinite vortex
        d: Unit direction vector (must be normalized)
        circulation: Circulation strength
        epsilon: Regularization parameter

    Returns:
        Velocity vector at target
    """
    vel = ti.Vector([0.0, 0.0, 0.0])

    # Project target onto the semi-infinite line: p0 = p + [(target-p)·d]d
    xmp = target - p
    xmpdotd = xmp.dot(d)
    p0 = p + xmpdotd * d

    # Bound Vortex Section (from p to p0)
    p0mp = p0 - p
    if p0mp.dot(p0mp) > epsilon * epsilon:  # Check there is a bound section
        vel = vel + bound_vortex_velocity(target, p, p0, circulation, epsilon)

    # Semi-Infinite Vortex Section (from p0 to infinity)
    # Velocity = (Γ/4π) * (d × h) / (|h|² + ε²)
    # where h = target - p0
    h = target - p0
    hsqr = h.dot(h)

    if hsqr > CUTOFF * CUTOFF:
        # n = d × h (perpendicular to both d and h)
        n = d.cross(h)
        denom = hsqr + epsilon * epsilon
        factor = circulation / (4.0 * 3.14159265359 * denom)
        vel = vel + factor * n

    return vel


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


@ti.func
def vortex_ring_velocity(target, v1, v2, v3, v4, circulation: float, epsilon: float):
    """
    Compute velocity induced by a closed vortex ring (doublé) at target point.

    Ring consists of four segments:
    1. v1 → v2
    2. v2 → v3
    3. v3 → v4
    4. v4 → v1
    """
    v12 = bound_vortex_velocity(target, v1, v2, circulation, epsilon)
    v23 = bound_vortex_velocity(target, v2, v3, circulation, epsilon)
    v34 = bound_vortex_velocity(target, v3, v4, circulation, epsilon)
    v41 = bound_vortex_velocity(target, v4, v1, circulation, epsilon)

    return v12 + v23 + v34 + v41


@ti.func
def horseshoe_semi_infinite_velocity(target, v2, v3, da, db, circulation: float, epsilon: float):
    """
    Compute velocity from horseshoe with semi-infinite trailing legs.

    This is the standard VLM formulation where trailing legs extend to
    infinity in the freestream direction.  The canonical orientation is the
    L → ∞ limit of the finite horseshoe (see horseshoe_velocity):

        left  leg:  infinity -> v2      (semi-infinite from v2 along +da, -circulation)
        bound:      v2 -> v3            (+circulation)
        right leg:  v3 -> infinity      (semi-infinite from v3 along +db, +circulation)

    i.e. a filament that runs v1→v2 / v3→v4 with +circulation collapses onto
    semi_infinite(v2, da, -circulation) / semi_infinite(v3, db, +circulation) as the far
    points v1 = v2 + da·∞, v4 = v3 + db·∞ go downstream.  This is certified
    numerically in tests/vpm/test_semi_infinite_horseshoe.py by comparing
    against the finite horseshoe at growing L/c.

    Args:
        target: Point where velocity is evaluated
        v2: Bound leg left endpoint
        v3: Bound leg right endpoint
        da: Downstream unit direction for left trailing leg
        db: Downstream unit direction for right trailing leg
        circulation: Circulation strength
        epsilon: Regularization parameter

    Returns:
        Velocity vector at target
    """
    # Left semi-infinite trailing leg: infinity -> v2 (reverse of v2 -> +inf in da)
    vel_left = semi_infinite_vortex_velocity(target, v2, da, -circulation, epsilon)

    # Bound leg (from v2 to v3)
    vel_bound = bound_vortex_velocity(target, v2, v3, circulation, epsilon)

    # Right semi-infinite trailing leg: v3 -> infinity
    vel_right = semi_infinite_vortex_velocity(target, v3, db, circulation, epsilon)

    vel = vel_left + vel_bound + vel_right

    return vel


@ti.func
def vortex_ring_tri_velocity(
    target: ti.types.vector(3, float),
    v0: ti.types.vector(3, float),
    v1: ti.types.vector(3, float),
    v2: ti.types.vector(3, float),
    circulation: float,
    epsilon: float,
) -> ti.types.vector(3, float):
    """
    Compute velocity induced by a triangular vortex ring panel.

    Biot-Savart integration for three edges with regularization.
    Used for comparison or alternative formulations.

    Args:
        target: Point where velocity is evaluated
        v0, v1, v2: Triangle vertices
        circulation: Circulation strength
        epsilon: Regularization parameter

    Returns:
        Velocity vector at target
    """
    # Three edges of the triangle
    vel_01 = bound_vortex_velocity(target, v0, v1, circulation, epsilon)
    vel_12 = bound_vortex_velocity(target, v1, v2, circulation, epsilon)
    vel_20 = bound_vortex_velocity(target, v2, v0, circulation, epsilon)

    vel = vel_01 + vel_12 + vel_20

    return vel
