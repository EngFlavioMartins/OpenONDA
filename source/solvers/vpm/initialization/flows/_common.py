"""Shared geometry helpers for canonical flow initialization."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ._shared import ParticleCoreCompensation


def vector3(value: Sequence[float], name: str) -> np.ndarray:
    """Validate and copy a finite three-component vector.

    Parameters
    ----------
    value : sequence of float
        Candidate Cartesian vector.  Units depend on the caller.
    name : str
        Label included in the validation error.

    Returns
    -------
    ndarray, shape (3,)
        Floating-point copy of ``value``.

    Raises
    ------
    ValueError
        If the value is not finite or does not have exactly three components.
    """
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain three finite values")
    return result


def unit_vector(value: Sequence[float], name: str) -> np.ndarray:
    """Return a normalized finite three-vector.

    Parameters
    ----------
    value : sequence of float
        Candidate vector, with units inherited from the caller.
    name : str
        Label used in validation errors.

    Returns
    -------
    ndarray, shape (3,)
        Unit-length dimensionless vector.

    Raises
    ------
    ValueError
        If ``value`` is not a finite non-zero 3-vector.
    """
    result = vector3(value, name)
    magnitude = float(np.linalg.norm(result))
    if magnitude <= np.finfo(float).eps:
        raise ValueError(f"{name} must be non-zero")
    return result / magnitude


def transverse_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Construct two orthonormal vectors transverse to a unit axis.

    Parameters
    ----------
    axis : ndarray, shape (3,)
        Non-zero axis; callers normally pass a unit vector.

    Returns
    -------
    first, second : ndarray, shape (3,)
        Right-handed transverse basis vectors.  Both are dimensionless.
    """
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(axis, reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    first = np.cross(axis, reference)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    return first, second


def represented_core_radius_squared(
    physical_core_radius: float,
    particle_core_radius: np.ndarray,
    *,
    compensation: ParticleCoreCompensation | None,
) -> float:
    """Compute the physical squared core radius left after compensation.

    Parameters
    ----------
    physical_core_radius : float
        Target physical vortex core radius in metres.
    particle_core_radius : ndarray
        Active particle core radii in metres.
    compensation : ParticleCoreCompensation or None
        Optional diffusion compensation model.  Its kernel diffusivity is
        dimensionless in the stored convention.

    Returns
    -------
    float
        Represented squared radius in m².

    Raises
    ------
    ValueError
        If the physical radius is invalid or compensation would make the
        represented radius non-positive.
    """
    physical_core_radius = float(physical_core_radius)
    if not np.isfinite(physical_core_radius) or physical_core_radius <= 0.0:
        raise ValueError("vortex_core_radius must be finite and positive")
    represented = physical_core_radius**2
    if compensation is not None:
        mean_particle_core_radius = float(np.mean(particle_core_radius))
        represented -= 4.0 * mean_particle_core_radius**2 / compensation.kernel_diffusivity
    if represented <= 0.0:
        raise ValueError("particle core radius must be smaller than the physical vortex core")
    return represented


def validate_viscosity(kinematic_viscosity: float) -> float:
    """Validate a non-negative kinematic viscosity in m²/s."""
    value = float(kinematic_viscosity)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("kinematic_viscosity must be finite and non-negative")
    return value
