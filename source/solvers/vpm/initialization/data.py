"""Validated array contracts for VPM particle initialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _vector_field(value: np.ndarray, name: str, count: int | None = None) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    if count is not None and len(array) != count:
        raise ValueError(f"{name} must contain {count} particles")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _scalar_field(value: np.ndarray, name: str, count: int) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=float)
    if array.shape != (count,):
        raise ValueError(f"{name} must have shape ({count},)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _integer_field(value: np.ndarray | None, name: str, count: int) -> np.ndarray | None:
    if value is None:
        return None
    array = np.ascontiguousarray(value, dtype=np.int32)
    if array.shape != (count,):
        raise ValueError(f"{name} must have shape ({count},)")
    return array


@dataclass(frozen=True, slots=True)
class ParticleDistribution:
    """Particle geometry and quadrature, without an attributed flow field."""

    position: np.ndarray
    core_radius: np.ndarray
    particle_volume: np.ndarray
    spacing: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.spacing) or self.spacing <= 0.0:
            raise ValueError("spacing must be finite and positive")
        position = _vector_field(self.position, "position")
        count = len(position)
        core_radius = _scalar_field(self.core_radius, "core_radius", count)
        particle_volume = _scalar_field(self.particle_volume, "particle_volume", count)
        if np.any(core_radius <= 0.0):
            raise ValueError("core_radius must be positive")
        if np.any(particle_volume <= 0.0):
            raise ValueError("particle_volume must be positive")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "core_radius", core_radius)
        object.__setattr__(self, "particle_volume", particle_volume)
        object.__setattr__(self, "spacing", float(self.spacing))

    def __len__(self) -> int:
        return len(self.position)

    @property
    def core_radius_ratio(self) -> float:
        """Return the mean particle ``sigma/h`` ratio."""
        return float(np.mean(self.core_radius) / self.spacing)

    def select(self, selection: np.ndarray) -> ParticleDistribution:
        """Return a distribution containing only the selected particles."""
        return ParticleDistribution(
            position=self.position[selection],
            core_radius=self.core_radius[selection],
            particle_volume=self.particle_volume[selection],
            spacing=self.spacing,
        )


@dataclass(frozen=True, slots=True)
class VortexParticleDistribution:
    """Complete, solver-ready vortex-particle initialization arrays."""

    position: np.ndarray
    velocity: np.ndarray
    vortex_strength: np.ndarray
    core_radius: np.ndarray
    particle_volume: np.ndarray
    kinematic_viscosity: np.ndarray
    spacing: float
    group_id: np.ndarray | None = None
    zone_id: np.ndarray | None = None

    def __post_init__(self) -> None:
        geometry = ParticleDistribution(
            position=self.position,
            core_radius=self.core_radius,
            particle_volume=self.particle_volume,
            spacing=self.spacing,
        )
        count = len(geometry)
        velocity = _vector_field(self.velocity, "velocity", count)
        vortex_strength = _vector_field(self.vortex_strength, "vortex_strength", count)
        kinematic_viscosity = _scalar_field(self.kinematic_viscosity, "kinematic_viscosity", count)
        if np.any(kinematic_viscosity < 0.0):
            raise ValueError("kinematic_viscosity must be non-negative")
        object.__setattr__(self, "position", geometry.position)
        object.__setattr__(self, "core_radius", geometry.core_radius)
        object.__setattr__(self, "particle_volume", geometry.particle_volume)
        object.__setattr__(self, "spacing", geometry.spacing)
        object.__setattr__(self, "velocity", velocity)
        object.__setattr__(self, "vortex_strength", vortex_strength)
        object.__setattr__(self, "kinematic_viscosity", kinematic_viscosity)
        object.__setattr__(self, "group_id", _integer_field(self.group_id, "group_id", count))
        object.__setattr__(self, "zone_id", _integer_field(self.zone_id, "zone_id", count))

    def __len__(self) -> int:
        return len(self.position)

    @property
    def distribution(self) -> ParticleDistribution:
        """Return the geometry-only view of this initialization."""
        return ParticleDistribution(
            position=self.position,
            core_radius=self.core_radius,
            particle_volume=self.particle_volume,
            spacing=self.spacing,
        )

    def select(self, selection: np.ndarray) -> VortexParticleDistribution:
        """Return a solver-ready subset while keeping every field aligned."""
        return VortexParticleDistribution(
            position=self.position[selection],
            velocity=self.velocity[selection],
            vortex_strength=self.vortex_strength[selection],
            core_radius=self.core_radius[selection],
            particle_volume=self.particle_volume[selection],
            kinematic_viscosity=self.kinematic_viscosity[selection],
            spacing=self.spacing,
            group_id=None if self.group_id is None else self.group_id[selection],
            zone_id=None if self.zone_id is None else self.zone_id[selection],
        )

    def solver_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments accepted by ``VPMSolver.add_vortex_particles``."""
        values: dict[str, Any] = {
            "position": self.position,
            "velocity": self.velocity,
            "vortex_strength": self.vortex_strength,
            "core_radius": self.core_radius,
            "particle_volume": self.particle_volume,
            "kinematic_viscosity": self.kinematic_viscosity,
        }
        if self.group_id is not None:
            values["group_id"] = self.group_id
        if self.zone_id is not None:
            values["zone_id"] = self.zone_id
        return values


def _attributed_distribution(
    distribution: ParticleDistribution,
    *,
    velocity: np.ndarray,
    vortex_strength: np.ndarray,
    kinematic_viscosity: float | np.ndarray,
    group_id: np.ndarray | None = None,
    zone_id: np.ndarray | None = None,
) -> VortexParticleDistribution:
    """Combine particle geometry with attributed flow fields."""
    viscosity = np.asarray(kinematic_viscosity, dtype=float)
    if viscosity.ndim == 0:
        viscosity = np.full(len(distribution), float(viscosity))
    return VortexParticleDistribution(
        position=distribution.position,
        velocity=velocity,
        vortex_strength=vortex_strength,
        core_radius=distribution.core_radius,
        particle_volume=distribution.particle_volume,
        kinematic_viscosity=viscosity,
        spacing=distribution.spacing,
        group_id=group_id,
        zone_id=zone_id,
    )
