# SPDX-License-Identifier: GPL-3.0-or-later
"""Immutable declarative inputs for the native Cartesian mesher."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import TypeAlias

import numpy as np

from ..triangulated_surface import SurfaceBounds, TriangulatedSurface
from .surface import load_surface

Bounds: TypeAlias = SurfaceBounds
Point: TypeAlias = tuple[float, float, float]


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


@dataclass(frozen=True, slots=True)
class BoxPatches:
    """Names assigned to the six faces of an axis-aligned domain box.

    The attributes are ordered as ``xmin, xmax, ymin, ymax, zmin, zmax``.
    Names are metadata used to create contiguous native boundary ranges; they
    do not themselves select a velocity or pressure boundary condition.

    Parameters
    ----------
    xmin, xmax, ymin, ymax, zmin, zmax : str
        Non-empty, case-sensitive patch names for the corresponding outward
        domain face. Names need not encode a boundary-condition type.
    coupling : tuple[str, ...], default=()
        Patch names that form an FVM–VPM coupling interface. Every box face
        carrying one of these names is meshed as an exact Cartesian plane with
        undistorted, axis-aligned cells at its faces, edges and corners (see
        :class:`~openonda.fvm.mesh.cartesian.mesher.CartesianMesher`). With any
        coupling patch, the other box faces that no surface crosses are meshed
        the same way.

    Examples
    --------
    >>> BoxPatches.coupled("numericalBoundary").coupling
    ('numericalBoundary',)
    """

    xmin: str
    xmax: str
    ymin: str
    ymax: str
    zmin: str
    zmax: str
    coupling: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), f"patches.{name}"))
        if isinstance(self.coupling, str):
            raise TypeError("patches.coupling must be a tuple of patch names, not one string")
        coupling = tuple(_nonempty(name, "patches.coupling") for name in self.coupling)
        if len(set(coupling)) != len(coupling):
            raise ValueError("patches.coupling contains a duplicate patch name")
        unknown = sorted(set(coupling) - set(self.as_tuple()))
        if unknown:
            raise ValueError(f"patches.coupling names no box face: {unknown}")
        object.__setattr__(self, "coupling", coupling)

    @classmethod
    def coupled(cls, name: str = "numericalBoundary") -> BoxPatches:
        """Return six faces named ``name`` that together form one coupling interface."""
        return cls(name, name, name, name, name, name, coupling=(name,))

    def as_tuple(self) -> tuple[str, str, str, str, str, str]:
        """Return names in ``xmin, xmax, ymin, ymax, zmin, zmax`` order."""
        return (self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax)

    def coupling_sides(self) -> tuple[int, ...]:
        """Return the side indices (``xmin`` = 0 ... ``zmax`` = 5) of coupling faces."""
        return tuple(side for side, name in enumerate(self.as_tuple()) if name in self.coupling)


@dataclass(frozen=True, slots=True)
class BoxDomain:
    """Finite Cartesian domain and its six outer patch names.

    ``bounds`` is ``(xmin, xmax, ymin, ymax, zmin, zmax)`` in metres. The
    domain is immutable after validation; mesher stages use these bounds for
    surface containment, outer-plane construction, and patch naming.

    Parameters
    ----------
    bounds : tuple[float, float, float, float, float, float]
        Strictly increasing bounds ``(xmin, xmax, ymin, ymax, zmin, zmax)`` in m.
    patches : BoxPatches
        Names assigned to the six corresponding outward faces.
    """

    bounds: Bounds
    patches: BoxPatches

    def __post_init__(self) -> None:
        object.__setattr__(self, "bounds", _bounds(self.bounds, "bounds"))
        if not isinstance(self.patches, BoxPatches):
            raise TypeError("patches must be a BoxPatches instance")


@dataclass(frozen=True, slots=True)
class STLSurface:
    """Validated triangulated surface assigned to one mesh patch.

    Parameters
    ----------
    path : str or pathlib.Path
        STL path. It is expanded/resolved and loaded during construction.
    patch : str
        Boundary patch name supplied to the generated mesh.
    allow_open : bool, default=False
        Allow an open/non-watertight surface when the downstream workflow can
        still classify it. Closed surfaces are required for solid carving.

    Notes
    -----
    Triangle coordinates are retained as a validated immutable ``(n, 3, 3)``
    array in metres. Construction performs file I/O and records a SHA-256
    source information hash; it raises before meshing if the surface is invalid.
    """

    path: Path | str
    patch: str
    allow_open: bool = False
    _surface: TriangulatedSurface = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        resolved_path = Path(self.path).expanduser().resolve()
        object.__setattr__(self, "path", resolved_path)
        object.__setattr__(self, "patch", _nonempty(self.patch, "patch"))
        if not isinstance(self.allow_open, bool):
            raise TypeError("allow_open must be a bool")
        object.__setattr__(
            self, "_surface", load_surface(resolved_path, allow_open=self.allow_open)
        )

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


@dataclass(frozen=True, slots=True)
class PatchRefinement:
    """Request a local surface cell size on one exact named patch.

    ``cell_size`` is an upper target in metres. The patch must exist in the
    domain/surface patch namespace before mesher execution. Without a
    ``CartesianMesher.cell_size_anchor``, cfMesh selects
    ``max_cell_size / 2**level <= cell_size``. With an anchor, compatible
    power-of-two requests are exact. Final surface and wrapper geometry can
    differ from the nominal octree size.

    Parameters
    ----------
    patch : str
        Exact case-sensitive name of an outer or embedded-surface patch.
    cell_size : float
        Positive requested maximum edge length in m.
    """

    patch: str
    cell_size: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "patch", _nonempty(self.patch, "patch"))
        object.__setattr__(self, "cell_size", _finite_positive(self.cell_size, "cell_size"))


@dataclass(frozen=True, slots=True)
class BoxRefinement:
    """Request an upper cell size inside an axis-aligned box in metres.

    ``bounds`` uses ``(xmin, xmax, ymin, ymax, zmin, zmax)``.
    Without a ``CartesianMesher.cell_size_anchor``, cfMesh selects
    ``max_cell_size / 2**level < cell_size`` (with its native floating-point
    tolerance). Equality then adds a level. With an anchor, compatible
    power-of-two requests are exact and equality retains the requested level.
    Intersecting leaves and the 2:1 neighbour transition can extend refinement
    past the box; overlapping controls select the finest applicable level.

    Parameters
    ----------
    name : str
        Unique diagnostic name for this refinement request.
    bounds : tuple[float, float, float, float, float, float]
        Closed physical region in m.
    cell_size : float
        Positive requested maximum edge length in m.
    """

    name: str
    bounds: Bounds
    cell_size: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty(self.name, "name"))
        object.__setattr__(self, "bounds", _bounds(self.bounds, f"{self.name}.bounds"))
        object.__setattr__(
            self, "cell_size", _finite_positive(self.cell_size, f"{self.name}.cell_size")
        )


@dataclass(frozen=True, slots=True)
class FeatureRefinement:
    """Requested sizing around surface edges above an included angle.

    ``angle`` is in degrees and ``cell_size`` is in metres. Only detected
    surface features above the angle threshold receive the refinement.

    Parameters
    ----------
    angle : float
        Included-angle threshold strictly between 0 and 180 degrees.
    cell_size : float
        Positive requested maximum size near qualifying edges, in m.
    """

    angle: float
    cell_size: float

    def __post_init__(self) -> None:
        angle = float(self.angle)
        if not math.isfinite(angle) or not 0.0 < angle < 180.0:
            raise ValueError("angle must be finite and strictly between 0 and 180 degrees")
        object.__setattr__(self, "angle", angle)
        object.__setattr__(self, "cell_size", _finite_positive(self.cell_size, "cell_size"))


__all__ = [
    "Bounds",
    "BoxDomain",
    "BoxPatches",
    "BoxRefinement",
    "FeatureRefinement",
    "PatchRefinement",
    "STLSurface",
]
