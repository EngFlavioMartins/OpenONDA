"""
Biot-Savart @ti.func kernels for the VLM: bound, semi-infinite, horseshoe, and
vortex-ring induced velocities.

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
def bound_vortex_velocity(target, pa, pb, gamma: float, epsilon: float):
    """
    Compute velocity induced by a bound vortex filament from pa to pb.

    Uses regularized Biot-Savart law:
    V = (Γ/4π) * (r1×r2) * [(r1/|r1| - r2/|r2|) · r12] / (|r1×r2|² + ε²)

    Args:
        target: Point where velocity is evaluated
        pa: Starting point of vortex filament
        pb: Ending point of vortex filament
        gamma: Circulation strength
        epsilon: Regularization parameter

    Returns:
        Velocity vector at target
    """
    vel = ti.Vector([0.0, 0.0, 0.0])

    # Vectors from endpoints to target
    r1 = target - pa
    r2 = target - pb
    r12 = pb - pa

    # Cross product r1 × r2
    cross = r1.cross(r2)
    cross_mag_sq = cross.dot(cross)

    # Check if target is too close to the vortex line
    # Use conditional assignment instead of early return
    if cross_mag_sq > CUTOFF * CUTOFF:
        # Floor the endpoint distances at epsilon to avoid 0/0 when the target
        # coincides with a filament endpoint.  Use max() (a guard that only
        # activates at the true singularity) rather than the previous
        # `|r| + epsilon`, which biased the whole near field. The physical
        # core regularization lives solely in the denominator term below
        # (Krasny/Rosenhead form: |r1×r2|² + ε²).
        r1_mag = ti.max(r1.norm(), epsilon)
        r2_mag = ti.max(r2.norm(), epsilon)

        # Biot-Savart kernel: r12 · (r1/|r1| - r2/|r2|)
        r12_dot_hat = r12.dot(r1 / r1_mag - r2 / r2_mag)

        # Regularized denominator
        denom = cross_mag_sq + epsilon * epsilon

        # Velocity contribution
        factor = gamma * r12_dot_hat / (4.0 * 3.14159265359 * denom)
        vel = factor * cross

    return vel


@ti.func
def semi_infinite_vortex_velocity(target, p, d, gamma: float, epsilon: float):
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
        gamma: Circulation strength
        epsilon: Regularization parameter

    Returns:
        Velocity vector at target
    """
    vel = ti.Vector([0.0, 0.0, 0.0])

    # Project target onto the semi-infinite line: p0 = p + [(target-p)·d]d
    xmp = target - p
    xmpdotd = xmp.dot(d)
    p0 = p + xmpdotd * d

    # ---- Bound Vortex Section (from p to p0) ----
    p0mp = p0 - p
    if p0mp.dot(p0mp) > epsilon * epsilon:  # Check there is a bound section
        vel = vel + bound_vortex_velocity(target, p, p0, gamma, epsilon)

    # ---- Semi-Infinite Vortex Section (from p0 to infinity) ----
    # Velocity = (Γ/4π) * (d × h) / (|h|² + ε²)
    # where h = target - p0
    h = target - p0
    hsqr = h.dot(h)

    if hsqr > CUTOFF * CUTOFF:
        # n = d × h (perpendicular to both d and h)
        n = d.cross(h)
        denom = hsqr + epsilon * epsilon
        factor = gamma / (4.0 * 3.14159265359 * denom)
        vel = vel + factor * n

    return vel


@ti.func
def horseshoe_velocity(target, v1, v2, v3, v4, gamma: float, epsilon: float):
    """
    Compute velocity induced by a horseshoe vortex at target point.

    Horseshoe consists of three segments:
    1. Left trailing leg: v1 (far upstream) → v2 (bound left)
    2. Bound leg: v2 → v3 (bound right)
    3. Right trailing leg: v3 → v4 (far upstream)
    """
    # Sum contributions from all three legs
    vel_left = bound_vortex_velocity(target, v1, v2, gamma, epsilon)  # Left trailing
    vel_bound = bound_vortex_velocity(target, v2, v3, gamma, epsilon)  # Bound
    vel_right = bound_vortex_velocity(target, v3, v4, gamma, epsilon)  # Right trailing

    vel = vel_left + vel_bound + vel_right

    return vel


@ti.func
def vortex_ring_velocity(target, v1, v2, v3, v4, gamma: float, epsilon: float):
    """
    Compute velocity induced by a closed vortex ring (doublé) at target point.

    Ring consists of four segments:
    1. v1 → v2
    2. v2 → v3
    3. v3 → v4
    4. v4 → v1
    """
    v12 = bound_vortex_velocity(target, v1, v2, gamma, epsilon)
    v23 = bound_vortex_velocity(target, v2, v3, gamma, epsilon)
    v34 = bound_vortex_velocity(target, v3, v4, gamma, epsilon)
    v41 = bound_vortex_velocity(target, v4, v1, gamma, epsilon)

    return v12 + v23 + v34 + v41


@ti.func
def horseshoe_semi_infinite_velocity(target, v2, v3, da, db, gamma: float, epsilon: float):
    """
    Compute velocity from horseshoe with semi-infinite trailing legs.

    This is the standard VLM formulation where trailing legs extend to
    infinity in the freestream direction.

    Args:
        target: Point where velocity is evaluated
        v2: Bound leg left endpoint
        v3: Bound leg right endpoint
        da: Direction vector for left trailing leg (normalized)
        db: Direction vector for right trailing leg (normalized)
        gamma: Circulation strength
        epsilon: Regularization parameter

    Returns:
        Velocity vector at target
    """
    # Left semi-infinite trailing leg (from v2 extending in direction -da)
    vel_left = semi_infinite_vortex_velocity(target, v2, -da, gamma, epsilon)

    # Bound leg (from v2 to v3)
    vel_bound = bound_vortex_velocity(target, v2, v3, gamma, epsilon)

    # Right semi-infinite trailing leg (from v3 extending in direction -db)
    vel_right = semi_infinite_vortex_velocity(target, v3, -db, gamma, epsilon)

    vel = vel_left + vel_bound + vel_right

    return vel


@ti.func
def vortex_ring_tri_velocity(
    target: ti.types.vector(3, float),
    v0: ti.types.vector(3, float),
    v1: ti.types.vector(3, float),
    v2: ti.types.vector(3, float),
    gamma: float,
    epsilon: float,
) -> ti.types.vector(3, float):
    """
    Compute velocity induced by a triangular vortex ring panel.

    Biot-Savart integration for three edges with regularization.
    Used for comparison or alternative formulations.

    Args:
        target: Point where velocity is evaluated
        v0, v1, v2: Triangle vertices
        gamma: Circulation strength
        epsilon: Regularization parameter

    Returns:
        Velocity vector at target
    """
    # Three edges of the triangle
    vel_01 = bound_vortex_velocity(target, v0, v1, gamma, epsilon)
    vel_12 = bound_vortex_velocity(target, v1, v2, gamma, epsilon)
    vel_20 = bound_vortex_velocity(target, v2, v0, gamma, epsilon)

    vel = vel_01 + vel_12 + vel_20

    return vel
