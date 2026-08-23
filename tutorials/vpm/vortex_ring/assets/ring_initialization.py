"""Research initial conditions for the vortex-ring stability tutorial."""

from __future__ import annotations

import numpy as np


def initialize_single_mode_toroidal_ring(
    position: np.ndarray,
    particle_volume: np.ndarray,
    core_radius: np.ndarray,
    *,
    kinematic_viscosity: float,
    ring_radius: float,
    tube_circulation: float,
    ring_core_radius: float,
    amplitude: float,
    mode: int,
    seed: int = 42,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    """Displace an unperturbed toroidal cloud by one solenoidal bending mode."""
    if amplitude < 0.0:
        raise ValueError("amplitude must be non-negative")
    if mode < 1:
        raise ValueError("mode must be positive")
    if kinematic_viscosity <= 0.0:
        raise ValueError("kinematic_viscosity must be positive")

    position = np.asarray(position, dtype=float).copy()
    particle_volume = np.asarray(particle_volume, dtype=float).copy()
    radius = np.asarray(core_radius, dtype=float).copy()
    theta = np.arctan2(position[:, 2], position[:, 1])
    rho_unperturbed = np.hypot(position[:, 1], position[:, 2])
    phase = float(2.0 * np.pi * np.random.RandomState(seed).rand())
    argument = mode * theta + phase
    displacement = ring_radius * amplitude * np.cos(argument)
    rho = rho_unperturbed + displacement
    if np.any(rho <= 0.0):
        raise ValueError("single-mode displacement moved particles through the ring axis")
    cosine = np.cos(theta)
    sine = np.sin(theta)
    position[:, 1] = rho * cosine
    position[:, 2] = rho * sine
    # Preserve the toroidal quadrature Jacobian rho*dtheta under the shift.
    particle_volume *= rho / rho_unperturbed

    represented_core_sq = ring_core_radius**2 - float(np.mean(radius)) ** 2
    if represented_core_sq <= 0.0:
        raise ValueError("particle radius must be smaller than the physical core radius")
    centreline_radius = ring_radius + displacement
    centreline_slope = -ring_radius * amplitude * mode * np.sin(argument)
    core_dist_sq = (rho - centreline_radius) ** 2 + position[:, 0] ** 2
    omega_magnitude = (
        tube_circulation
        / (np.pi * represented_core_sq)
        * np.exp(-core_dist_sq / represented_core_sq)
    )
    omega_radial = omega_magnitude * centreline_slope / rho
    vorticity = np.zeros_like(position)
    vorticity[:, 1] = -omega_magnitude * sine + omega_radial * cosine
    vorticity[:, 2] = omega_magnitude * cosine + omega_radial * sine
    vortex_strength = vorticity * particle_volume[:, None]

    tangent = np.column_stack((np.zeros_like(theta), -sine, cosine))
    represented_circulation = np.sum(np.einsum("ij,ij->i", vortex_strength, tangent) / rho) / (
        2.0 * np.pi
    )
    if abs(represented_circulation) <= np.finfo(float).tiny:
        raise ValueError("cannot normalize a ring with zero represented circulation")
    vortex_strength *= tube_circulation / represented_circulation
    velocity = np.zeros_like(position)
    particle_kinematic_viscosity = np.full(len(position), kinematic_viscosity)
    return (
        position,
        particle_volume,
        radius,
        velocity,
        particle_kinematic_viscosity,
        vortex_strength,
        phase,
    )
