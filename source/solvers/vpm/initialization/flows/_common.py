"""Shared geometry helpers for canonical flow initialization."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ._shared import ParticleCoreCompensation


def vector3(value: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain three finite values")
    return result


def unit_vector(value: Sequence[float], name: str) -> np.ndarray:
    result = vector3(value, name)
    magnitude = float(np.linalg.norm(result))
    if magnitude <= np.finfo(float).eps:
        raise ValueError(f"{name} must be non-zero")
    return result / magnitude


def transverse_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    value = float(kinematic_viscosity)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("kinematic_viscosity must be finite and non-negative")
    return value
