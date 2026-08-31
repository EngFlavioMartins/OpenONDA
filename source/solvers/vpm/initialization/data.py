"""Typed, immutable array contracts used by VPM initial conditions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int32]
BoolArray: TypeAlias = NDArray[np.bool_]


def _immutable_float(value: object, name: str, shape: tuple[int, ...]) -> FloatArray:
    """Copy a finite floating array with an exact shape."""
    array = np.array(value, dtype=np.float64, copy=True, order="C")
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _immutable_vector_field(value: object, name: str, count: int | None = None) -> FloatArray:
    """Copy a finite ``(N, 3)`` field."""
    array = np.array(value, dtype=np.float64, copy=True, order="C")
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    if count is not None and len(array) != count:
        raise ValueError(f"{name} must contain {count} particles")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _immutable_scalar_field(value: object, name: str, count: int) -> FloatArray:
    """Copy a finite ``(N,)`` scalar field."""
    return _immutable_float(value, name, (count,))


def _immutable_integer_field(value: object | None, name: str, count: int) -> IntArray | None:
    """Copy an integral ID field without silently truncating values."""
    if value is None:
        return None
    source = np.asarray(value)
    if source.shape != (count,):
        raise ValueError(f"{name} must have shape ({count},)")
    if not np.issubdtype(source.dtype, np.integer):
        if not np.issubdtype(source.dtype, np.floating) or not np.all(np.isfinite(source)):
            raise ValueError(f"{name} must contain integral values")
        if not np.all(source == np.floor(source)):
            raise ValueError(f"{name} must contain integral values")
    info = np.iinfo(np.int32)
    if np.any(source < info.min) or np.any(source > info.max):
        raise ValueError(f"{name} values must fit in int32")
    array = np.array(source, dtype=np.int32, copy=True, order="C")
    array.setflags(write=False)
    return array


def _selection_indices(selection: object, count: int) -> NDArray[np.intp] | BoolArray:
    """Validate a boolean mask or bounded integral index array."""
    array = np.asarray(selection)
    if array.dtype == np.bool_:
        if array.shape != (count,):
            raise ValueError(f"selection mask must have shape ({count},)")
        return array
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("selection must be a boolean mask or integral indices")
    if array.ndim != 1 or np.any(array < 0) or np.any(array >= count):
        raise ValueError("selection indices must be one-dimensional and in bounds")
    return np.asarray(array, dtype=np.intp)


@dataclass(frozen=True, slots=True)
class ParticleDistribution:
    """Immutable particle geometry and quadrature.

    Positions have shape ``(N, 3)`` in length units; radii have shape ``(N,)``
    in length units; volumes have shape ``(N,)`` in cubic length units.
    ``spacing`` is the nominal transverse particle spacing and is positive.
    """

    position: FloatArray
    core_radius: FloatArray
    particle_volume: FloatArray
    spacing: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.spacing) or self.spacing <= 0.0:
            raise ValueError("spacing must be finite and positive")
        position = _immutable_vector_field(self.position, "position")
        count = len(position)
        core_radius = _immutable_scalar_field(self.core_radius, "core_radius", count)
        particle_volume = _immutable_scalar_field(self.particle_volume, "particle_volume", count)
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
        """Return the mean particle ``sigma / h`` ratio."""
        return float(np.mean(self.core_radius) / self.spacing)

    def select(self, selection: object) -> ParticleDistribution:
        """Return an immutable, validated subset of this distribution."""
        indices = _selection_indices(selection, len(self))
        return ParticleDistribution(
            position=self.position[indices],
            core_radius=self.core_radius[indices],
            particle_volume=self.particle_volume[indices],
            spacing=self.spacing,
        )


@dataclass(frozen=True, slots=True)
class VortexParticleSet:
    """Immutable solver-ready vortex-particle arrays.

    Vector fields use shape ``(N, 3)``; scalar and ID fields use ``(N,)``.
    Arrays are copied and marked read-only, so a frozen container cannot be
    mutated through an input-array alias.
    """

    position: FloatArray
    velocity: FloatArray
    vortex_strength: FloatArray
    core_radius: FloatArray
    particle_volume: FloatArray
    kinematic_viscosity: FloatArray
    spacing: float
    group_id: IntArray | None = None
    zone_id: IntArray | None = None

    def __post_init__(self) -> None:
        geometry = ParticleDistribution(
            position=self.position,
            core_radius=self.core_radius,
            particle_volume=self.particle_volume,
            spacing=self.spacing,
        )
        count = len(geometry)
        velocity = _immutable_vector_field(self.velocity, "velocity", count)
        vortex_strength = _immutable_vector_field(self.vortex_strength, "vortex_strength", count)
        kinematic_viscosity = _immutable_scalar_field(
            self.kinematic_viscosity, "kinematic_viscosity", count
        )
        if np.any(kinematic_viscosity < 0.0):
            raise ValueError("kinematic_viscosity must be non-negative")
        object.__setattr__(self, "position", geometry.position)
        object.__setattr__(self, "core_radius", geometry.core_radius)
        object.__setattr__(self, "particle_volume", geometry.particle_volume)
        object.__setattr__(self, "spacing", geometry.spacing)
        object.__setattr__(self, "velocity", velocity)
        object.__setattr__(self, "vortex_strength", vortex_strength)
        object.__setattr__(self, "kinematic_viscosity", kinematic_viscosity)
        object.__setattr__(
            self, "group_id", _immutable_integer_field(self.group_id, "group_id", count)
        )
        object.__setattr__(
            self, "zone_id", _immutable_integer_field(self.zone_id, "zone_id", count)
        )

    def __len__(self) -> int:
        return len(self.position)

    @property
    def distribution(self) -> ParticleDistribution:
        """Return the geometry-only immutable view."""
        return ParticleDistribution(
            position=self.position,
            core_radius=self.core_radius,
            particle_volume=self.particle_volume,
            spacing=self.spacing,
        )

    def select(self, selection: object) -> VortexParticleSet:
        """Return a subset while keeping all particle fields aligned."""
        indices = _selection_indices(selection, len(self))
        return VortexParticleSet(
            position=self.position[indices],
            velocity=self.velocity[indices],
            vortex_strength=self.vortex_strength[indices],
            core_radius=self.core_radius[indices],
            particle_volume=self.particle_volume[indices],
            kinematic_viscosity=self.kinematic_viscosity[indices],
            spacing=self.spacing,
            group_id=None if self.group_id is None else self.group_id[indices],
            zone_id=None if self.zone_id is None else self.zone_id[indices],
        )


def attributed_particle_set(
    distribution: ParticleDistribution,
    *,
    velocity: object,
    vortex_strength: object,
    kinematic_viscosity: float | FloatArray,
    group_id: object | None = None,
    zone_id: object | None = None,
) -> VortexParticleSet:
    """Combine immutable geometry with attributed flow fields."""
    viscosity = np.asarray(kinematic_viscosity, dtype=np.float64)
    if viscosity.ndim == 0:
        viscosity = np.full(len(distribution), float(viscosity), dtype=np.float64)
    return VortexParticleSet(
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
