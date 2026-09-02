# SPDX-License-Identifier: GPL-3.0-or-later
"""Immutable declarative inputs for the native Cartesian mesher."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np

from ..triangulated_surface import SurfaceBounds, TriangulatedSurface
from .surface import load_surface

Bounds: TypeAlias = SurfaceBounds
Point: TypeAlias = tuple[float, float, float]


def _refinement_attribute(refinement: Refinement, attribute: str) -> Any:
    """Read a common declarative field without widening the base class contract."""
    return getattr(refinement, attribute)


def _finite_positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _nonempty(value: str, name: str) -> str:
    value = str(value).strip()
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _bounds(value: Bounds, name: str) -> Bounds:
    if len(value) != 6:
        raise ValueError(f"{name} must contain six coordinates")
    result = tuple(float(coordinate) for coordinate in value)
    if not all(math.isfinite(coordinate) for coordinate in result):
        raise ValueError(f"{name} must contain only finite coordinates")
    if not all(result[2 * axis] < result[2 * axis + 1] for axis in range(3)):
        raise ValueError(f"{name} must have positive extent along every axis")
    return result  # type: ignore[return-value]


def _point(value: Point, name: str) -> Point:
    if len(value) != 3:
        raise ValueError(f"{name} must contain three coordinates")
    result = tuple(float(coordinate) for coordinate in value)
    if not all(math.isfinite(coordinate) for coordinate in result):
        raise ValueError(f"{name} must contain only finite coordinates")
    return result  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class BoxPatches:
    """Names assigned to the six faces of an axis-aligned domain box."""

    xmin: str
    xmax: str
    ymin: str
    ymax: str
    zmin: str
    zmax: str

    def __post_init__(self) -> None:
        for name in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), f"patches.{name}"))

    def as_tuple(self) -> tuple[str, str, str, str, str, str]:
        """Return names in ``xmin, xmax, ymin, ymax, zmin, zmax`` order."""
        return (self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax)


@dataclass(frozen=True, slots=True)
class BoxDomain:
    """Outer Cartesian domain and its configured boundary-patch names."""

    bounds: Bounds
    patches: BoxPatches

    def __post_init__(self) -> None:
        object.__setattr__(self, "bounds", _bounds(self.bounds, "bounds"))
        if not isinstance(self.patches, BoxPatches):
            raise TypeError("patches must be a BoxPatches instance")


@dataclass(frozen=True, slots=True)
class STLSurface:
    """A validated triangulated surface and the patch name it supplies."""

    path: Path | str
    patch: str
    _surface: TriangulatedSurface = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        resolved_path = Path(self.path).expanduser().resolve()
        object.__setattr__(self, "path", resolved_path)
        object.__setattr__(self, "patch", _nonempty(self.patch, "patch"))
        object.__setattr__(self, "_surface", load_surface(resolved_path))

    @property
    def triangles(self) -> np.ndarray:
        """Read-only ``(n, 3, 3)`` triangle coordinates."""
        return self._surface.triangles

    @property
    def surface_data(self) -> TriangulatedSurface:
        """Validated immutable surface model used by the meshing stages."""
        return self._surface

    @property
    def bounds(self) -> Bounds:
        """Axis-aligned bounds of the supplied surface."""
        return self._surface.bounds

    @property
    def sha256(self) -> str:
        """SHA-256 hash of the exact input STL bytes."""
        return self._surface.sha256

    @property
    def kind(self) -> str:
        """Surface classification retained for diagnostics only."""
        return self._surface.kind


class Refinement(ABC):
    """Base protocol for a physical size request."""

    @abstractmethod
    def contains(self, points: np.ndarray) -> np.ndarray:
        """Return a boolean mask for points controlled by this request."""
        return np.empty(0, dtype=bool)

    def intersects_box(self, lower: np.ndarray, upper: np.ndarray) -> bool:
        """Return a conservative intersection test for one axis-aligned box."""
        bounds = np.asarray(cast(Any, self).bounds, dtype=np.float64)
        return bool(
            np.all(np.asarray(lower, dtype=np.float64) < bounds[1::2])
            and np.all(np.asarray(upper, dtype=np.float64) > bounds[::2])
        )


@dataclass(frozen=True, slots=True)
class BoxRefinement(Refinement):
    """Request an upper cell size inside an axis-aligned box."""

    name: str
    bounds: Bounds
    cell_size: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty(self.name, "name"))
        object.__setattr__(self, "bounds", _bounds(self.bounds, f"{self.name}.bounds"))
        object.__setattr__(
            self, "cell_size", _finite_positive(self.cell_size, f"{self.name}.cell_size")
        )

    def contains(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        lower = np.asarray(self.bounds[::2])
        upper = np.asarray(self.bounds[1::2])
        return np.all((values >= lower) & (values <= upper), axis=1)


@dataclass(frozen=True, slots=True)
class SphereRefinement(Refinement):
    """Request an upper cell size inside a radius around a centre."""

    name: str
    centre: Point
    radius: float
    cell_size: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty(self.name, "name"))
        object.__setattr__(self, "centre", _point(self.centre, f"{self.name}.centre"))
        object.__setattr__(self, "radius", _finite_positive(self.radius, f"{self.name}.radius"))
        object.__setattr__(
            self, "cell_size", _finite_positive(self.cell_size, f"{self.name}.cell_size")
        )

    def contains(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        centre = np.asarray(self.centre, dtype=np.float64)
        return np.einsum("ij,ij->i", values - centre, values - centre) <= self.radius**2

    def intersects_box(self, lower: np.ndarray, upper: np.ndarray) -> bool:
        """Use the exact closest-point test for a radial/AABB pair."""
        centre = np.asarray(self.centre, dtype=np.float64)
        closest = np.minimum(np.maximum(centre, lower), upper)
        return bool(np.dot(closest - centre, closest - centre) <= self.radius**2)

    @property
    def bounds(self) -> Bounds:
        """Conservative axis-aligned bounds used by the Cartesian grid."""
        centre = np.asarray(self.centre)
        radius = np.full(3, self.radius)
        return (
            float(centre[0] - radius[0]),
            float(centre[0] + radius[0]),
            float(centre[1] - radius[1]),
            float(centre[1] + radius[1]),
            float(centre[2] - radius[2]),
            float(centre[2] + radius[2]),
        )


@dataclass(frozen=True, slots=True)
class ConeRefinement(Refinement):
    """Request an upper cell size inside a finite axis-aligned cone volume."""

    name: str
    centre: Point
    axis: Point
    radius: float
    height: float
    cell_size: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty(self.name, "name"))
        object.__setattr__(self, "centre", _point(self.centre, f"{self.name}.centre"))
        axis = np.asarray(_point(self.axis, f"{self.name}.axis"), dtype=np.float64)
        length = float(np.linalg.norm(axis))
        if not math.isfinite(length) or length <= 0.0:
            raise ValueError(f"{self.name}.axis must be non-zero")
        object.__setattr__(self, "axis", tuple((axis / length).tolist()))
        object.__setattr__(self, "radius", _finite_positive(self.radius, f"{self.name}.radius"))
        object.__setattr__(self, "height", _finite_positive(self.height, f"{self.name}.height"))
        object.__setattr__(
            self, "cell_size", _finite_positive(self.cell_size, f"{self.name}.cell_size")
        )

    def contains(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        centre = np.asarray(self.centre, dtype=np.float64)
        axis = np.asarray(self.axis, dtype=np.float64)
        axial = np.einsum("ij,j->i", values - centre, axis)
        radial = values - centre - axial[:, None] * axis
        allowed_radius = self.radius * np.maximum(0.0, 1.0 - axial / self.height)
        return (
            (axial >= 0.0)
            & (axial <= self.height)
            & (np.einsum("ij,ij->i", radial, radial) <= allowed_radius**2)
        )

    @property
    def bounds(self) -> Bounds:
        """Conservative axis-aligned bounds enclosing the finite cone."""
        centre = np.asarray(self.centre)
        tip = centre + self.height * np.asarray(self.axis)
        lower = np.minimum(centre, tip) - self.radius
        upper = np.maximum(centre, tip) + self.radius
        return tuple(float(value) for pair in zip(lower, upper, strict=True) for value in pair)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class LineRefinement(Refinement):
    """Request an upper cell size in a radius around a line segment."""

    name: str
    start: Point
    end: Point
    cell_size: float
    radius: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty(self.name, "name"))
        start = np.asarray(_point(self.start, f"{self.name}.start"), dtype=np.float64)
        end = np.asarray(_point(self.end, f"{self.name}.end"), dtype=np.float64)
        if np.array_equal(start, end):
            raise ValueError(f"{self.name}.start and end must differ")
        object.__setattr__(self, "start", tuple(start.tolist()))
        object.__setattr__(self, "end", tuple(end.tolist()))
        object.__setattr__(
            self, "cell_size", _finite_positive(self.cell_size, f"{self.name}.cell_size")
        )
        if self.radius is not None:
            object.__setattr__(self, "radius", _finite_positive(self.radius, f"{self.name}.radius"))

    def contains(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        start = np.asarray(self.start, dtype=np.float64)
        delta = np.asarray(self.end, dtype=np.float64) - start
        fraction = np.einsum("ij,j->i", values - start, delta) / np.dot(delta, delta)
        fraction = np.clip(fraction, 0.0, 1.0)
        closest = start + fraction[:, None] * delta
        width = self.radius if self.radius is not None else 0.5 * self.cell_size
        return np.einsum("ij,ij->i", values - closest, values - closest) <= width**2

    @property
    def bounds(self) -> Bounds:
        """Conservative axis-aligned bounds enclosing the line tube."""
        width = self.radius if self.radius is not None else 0.5 * self.cell_size
        lower = np.minimum(self.start, self.end) - width
        upper = np.maximum(self.start, self.end) + width
        return tuple(float(value) for pair in zip(lower, upper, strict=True) for value in pair)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class FeatureRefinement:
    """Requested sizing around surface features above an included angle."""

    angle: float
    cell_size: float

    def __post_init__(self) -> None:
        angle = float(self.angle)
        if not math.isfinite(angle) or not 0.0 < angle < 180.0:
            raise ValueError("angle must be finite and strictly between 0 and 180 degrees")
        object.__setattr__(self, "angle", angle)
        object.__setattr__(self, "cell_size", _finite_positive(self.cell_size, "cell_size"))


@dataclass(frozen=True, slots=True)
class BoundaryLayers:
    """Wall-normal layer request for one or more configured patches."""

    patches: tuple[str, ...]
    layers: int
    first_cell_height: float
    growth_ratio: float

    def __post_init__(self) -> None:
        patches = tuple(_nonempty(patch, "patch") for patch in self.patches)
        if not patches or len(set(patches)) != len(patches):
            raise ValueError("patches must contain one or more unique names")
        if int(self.layers) != self.layers or self.layers < 1:
            raise ValueError("layers must be a positive integer")
        growth_ratio = float(self.growth_ratio)
        if not math.isfinite(growth_ratio) or growth_ratio < 1.0:
            raise ValueError("growth_ratio must be finite and at least one")
        object.__setattr__(self, "patches", patches)
        object.__setattr__(self, "layers", int(self.layers))
        object.__setattr__(
            self, "first_cell_height", _finite_positive(self.first_cell_height, "first_cell_height")
        )
        object.__setattr__(self, "growth_ratio", growth_ratio)

    @property
    def layer_heights(self) -> tuple[float, ...]:
        """Requested layer-cell heights from the wall outwards."""
        return tuple(
            self.first_cell_height * self.growth_ratio**index for index in range(self.layers)
        )


class SizeField(ABC):
    """General interface for evaluating requested physical cell sizes."""

    @abstractmethod
    def requested_size(self, points: np.ndarray) -> np.ndarray:
        """Return one requested upper size for every point."""


@dataclass(frozen=True, slots=True)
class CompositeSizeField(SizeField):
    """Combine background and refinement requests by taking the smallest size."""

    background_size: float
    refinements: tuple[Refinement, ...] = ()
    minimum_size: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "refinements", tuple(self.refinements))
        object.__setattr__(
            self, "background_size", _finite_positive(self.background_size, "background_size")
        )
        names = [str(_refinement_attribute(refinement, "name")) for refinement in self.refinements]
        if len(set(names)) != len(names):
            raise ValueError("refinement names must be unique")
        if self.minimum_size is not None:
            minimum = _finite_positive(self.minimum_size, "minimum_size")
            if minimum > self.background_size:
                raise ValueError("minimum_size must not exceed background_size")
            object.__setattr__(self, "minimum_size", minimum)

    def requested_size(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("points must have shape (n, 3)")
        result = np.full(len(values), self.background_size, dtype=np.float64)
        for refinement in self.refinements:
            mask = refinement.contains(values)
            if np.any(mask):
                result[mask] = np.minimum(
                    result[mask], float(_refinement_attribute(refinement, "cell_size"))
                )
        if self.minimum_size is not None:
            result = np.maximum(result, self.minimum_size)
        return result

    @property
    def requested_sizes(self) -> tuple[float, ...]:
        """All distinct requested levels, including background and safety limit."""
        values = [
            self.background_size,
            *(
                float(_refinement_attribute(refinement, "cell_size"))
                for refinement in self.refinements
            ),
        ]
        if self.minimum_size is not None:
            values.append(self.minimum_size)
        return tuple(sorted(set(values), reverse=True))


__all__ = [
    "Bounds",
    "BoundaryLayers",
    "BoxDomain",
    "BoxPatches",
    "BoxRefinement",
    "CompositeSizeField",
    "ConeRefinement",
    "FeatureRefinement",
    "LineRefinement",
    "Refinement",
    "SizeField",
    "SphereRefinement",
    "STLSurface",
]
