"""Velocity and velocity gradient of the complete VPM flow at arbitrary points.

The coupler evaluates the flow at the faces of the FVM domain and the samplers
at the points of a line or plane. Both need the particles, the freestream and
the velocity of an attached vortex lattice. Particle induction goes through the
physics engine; the lattice adds its velocity at the accepted time and, where a
gradient is wanted, its centred difference. Positions are in m, velocities in
m/s and gradients ``J[i, j] = du_i/dx_j`` in 1/s.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..particles.container import Particles
    from ..vlm.solver import VLMSolver
    from .engine import PhysicsEngine


def compute_velocity_at_points(
    physics: PhysicsEngine,
    particles: Particles,
    vlm_solver: VLMSolver | None,
    time: float,
    evaluation_position: np.ndarray,
    include_freestream: bool = True,
    zone_mask: np.ndarray | None = None,
    include_body: bool = True,
) -> np.ndarray:
    """Evaluate complete velocity at arbitrary target points.

    Parameters
    ----------
    physics, particles, vlm_solver, time
        Physics engine, particle cloud and optional vortex lattice of the
        solver, and its accepted time in s.
    evaluation_position : numpy.ndarray
        Target coordinates, shape ``(N, 3)``, in m.
    include_freestream : bool, default=True
        Include the uniform background velocity in the particle induction.
    zone_mask : numpy.ndarray or None, default=None
        Optional boolean/int mask selecting source particles. Its length
        must match the active particle count.
    include_body : bool, default=True
        Include the velocity of the attached vortex lattice.

    Returns
    -------
    numpy.ndarray
        Velocity vectors, shape ``(N, 3)``, in m/s.
    """
    velocity = physics.compute_target_velocity(
        particles,
        evaluation_position,
        include_freestream=include_freestream,
        zone_mask=zone_mask,
    )
    return _add_lattice_velocity(
        physics, vlm_solver, time, evaluation_position, velocity, include_body=include_body
    )


def compute_velocity_and_gradient_at_points(
    physics: PhysicsEngine,
    particles: Particles,
    vlm_solver: VLMSolver | None,
    time: float,
    evaluation_position: np.ndarray,
    *,
    particle_spacing: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate body-complete target velocity and Jacobian together.

    Treecode runs build and traverse the particle hierarchy once, then add
    the attached-VLM velocity and Jacobian terms.

    Parameters
    ----------
    physics, particles, vlm_solver, time
        Physics engine, particle cloud and optional vortex lattice of the
        solver, and its accepted time in s.
    evaluation_position : numpy.ndarray
        Target coordinates, shape ``(N, 3)``, in m.
    particle_spacing : float
        Coupling spacing in m for centered differences of the attached-VLM term.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Complete velocity ``(N, 3)`` in m/s and Jacobian ``(N, 3, 3)`` in
        1/s.

    Raises
    ------
    RuntimeError
        If the complete gradient is not finite.
    """
    points = np.asarray(evaluation_position, dtype=np.float64).reshape(-1, 3)
    # One fused particle evaluation: the velocity and the Jacobian use the same
    # approximation, which a direct Jacobian beside a treecode velocity would not.
    velocity, gradient = physics.compute_target_velocity_and_gradients_consistent(
        particles, points, include_freestream=True
    )
    complete_velocity = _add_lattice_velocity(
        physics, vlm_solver, time, points, velocity, include_body=True
    )
    complete_gradient = np.asarray(gradient, dtype=np.float64).reshape(-1, 3, 3).copy()
    if len(points) == 0 or not _lattice_solved(vlm_solver):
        return complete_velocity, complete_gradient

    step = _difference_step(particle_spacing)
    for axis in range(3):
        offset = np.zeros(3, dtype=np.float64)
        offset[axis] = step
        plus = _lattice_velocity(physics, vlm_solver, time, points + offset)
        minus = _lattice_velocity(physics, vlm_solver, time, points - offset)
        complete_gradient[:, :, axis] += (plus - minus) / (2.0 * step)
    if not np.all(np.isfinite(complete_gradient)):
        raise RuntimeError("Complete VPM target-gradient evaluation returned non-finite data")
    return complete_velocity, complete_gradient


def compute_velocity_and_tangential_normal_gradient_at_points(
    physics: PhysicsEngine,
    particles: Particles,
    vlm_solver: VLMSolver | None,
    time: float,
    evaluation_position: np.ndarray,
    normal: np.ndarray,
    *,
    particle_spacing: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the velocity and the normal gradient of the tangential velocity at points.

    The mixed FVM boundary condition does not consume the full nine-component
    Jacobian. Particle induction is still evaluated by the configured fused
    target operation. The attached-VLM term uses
    only two centred samples along each face normal instead of three
    coordinate-direction pairs.

    Parameters
    ----------
    physics, particles, vlm_solver, time
        Physics engine, particle cloud and optional vortex lattice of the
        solver, and its accepted time in s.
    evaluation_position, normal : numpy.ndarray
        Face points and normals, both shape ``(N, 3)``. Positions are in m;
        normals are dimensionless and must be finite/non-zero.
    particle_spacing : float
        Centered-difference spacing in m for the attached-VLM term.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Complete velocity ``(N, 3)`` in m/s and tangential part of
        ``J @ n`` with shape ``(N, 3)`` in m/s².

    Raises
    ------
    ValueError
        If normal and point shapes differ or a normal is zero/non-finite.
    """
    points = np.asarray(evaluation_position, dtype=np.float64).reshape(-1, 3)
    face_normals = np.asarray(normal, dtype=np.float64).reshape(-1, 3)
    if face_normals.shape != points.shape:
        raise ValueError("normal count does not match target position")
    normal_magnitude = np.linalg.norm(face_normals, axis=1)
    if np.any(~np.isfinite(face_normals)) or np.any(normal_magnitude <= 0.0):
        raise ValueError("target normal must be finite and non-zero")
    unit_normals = face_normals / normal_magnitude[:, None]

    velocity, gradient = physics.compute_target_velocity_and_gradients_consistent(
        particles, points, include_freestream=True
    )

    complete_velocity = _add_lattice_velocity(
        physics, vlm_solver, time, points, velocity, include_body=True
    )
    normal_velocity_gradient = np.einsum(
        "fij,fj->fi", np.asarray(gradient, dtype=np.float64).reshape(-1, 3, 3), unit_normals
    )
    if _lattice_solved(vlm_solver):
        step = _difference_step(particle_spacing)
        plus = _lattice_velocity(physics, vlm_solver, time, points + step * unit_normals)
        minus = _lattice_velocity(physics, vlm_solver, time, points - step * unit_normals)
        normal_velocity_gradient += (plus - minus) / (2.0 * step)
    tangential = (
        normal_velocity_gradient
        - np.einsum("fi,fi->f", normal_velocity_gradient, unit_normals)[:, None] * unit_normals
    )
    if not np.all(np.isfinite(complete_velocity)) or not np.all(np.isfinite(tangential)):
        raise RuntimeError("Mixed VPM target evaluation returned non-finite data")
    return np.asarray(complete_velocity, dtype=np.float64), tangential


def _lattice_solved(vlm_solver: VLMSolver | None) -> bool:
    """Return whether an attached vortex lattice has a solved circulation."""
    return vlm_solver is not None and vlm_solver._solved


def _difference_step(particle_spacing: float) -> float:
    """Return the centred-difference step in m for the lattice velocity."""
    return max(1.0e-6, 1.0e-3 * float(particle_spacing))


def _add_lattice_velocity(
    physics: PhysicsEngine,
    vlm_solver: VLMSolver | None,
    time: float,
    evaluation_position: np.ndarray,
    particle_velocity: np.ndarray,
    *,
    include_body: bool,
) -> np.ndarray:
    """Add the attached-VLM velocity to the particle induction."""
    points = np.asarray(evaluation_position, dtype=np.float64).reshape(-1, 3)
    velocity = np.array(particle_velocity, dtype=physics.np_dtype, copy=True).reshape(-1, 3)
    if len(velocity) != len(points):
        raise ValueError("target velocity and position counts must match")

    if include_body and _lattice_solved(vlm_solver):
        for start, stop in physics._target_batch_slices(len(points)):
            count = stop - start
            positions = physics.target_position
            velocities = physics.target_velocity
            physics.upload_vector_array(points[start:stop], positions, count)
            physics.upload_vector_array(velocity[start:stop], velocities, count)
            vlm_solver.add_stage_velocity(positions, velocities, count, time)
            velocity[start:stop] = physics.extract_target_velocity(count)

    return velocity


def _lattice_velocity(
    physics: PhysicsEngine,
    vlm_solver: VLMSolver | None,
    time: float,
    evaluation_position: np.ndarray,
) -> np.ndarray:
    """Return only the attached-VLM velocity at the target points."""
    points = np.asarray(evaluation_position, dtype=np.float64).reshape(-1, 3)
    return _add_lattice_velocity(
        physics,
        vlm_solver,
        time,
        points,
        np.zeros((len(points), 3), dtype=physics.np_dtype),
        include_body=True,
    )
