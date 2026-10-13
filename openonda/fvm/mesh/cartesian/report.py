# SPDX-License-Identifier: GPL-3.0-or-later
"""Immutable reports emitted by Cartesian-mesher builds."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ..data import MeshData


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class SizeReport:
    """Requested and nominal octree size for one named meshing control.

    Effective sizes describe each control before overlapping requests,
    neighbour balancing, surface projection, and wrapper insertion. They do
    not promise identical physical edge lengths or volumes in the final cells.

    Parameters
    ----------
    name : str
        Diagnostic control name, such as ``"background"`` or ``"patch:wall"``.
    requested, effective : float
        Requested upper size and realized power-of-two size in m. The effective
        value is no larger than the request.
    level : int
        Additional refinement level relative to ``max_cell_size``. Per-cell
        ``cell_levels`` instead count from the Cartesian root box.
    """

    name: str
    requested: float
    effective: float
    level: int

    def as_dict(self) -> dict[str, Any]:
        """Return a serialisable copy of this size record."""
        return {
            "name": self.name,
            "requested": self.requested,
            "effective": self.effective,
            "level": self.level,
        }


@dataclass(frozen=True, slots=True)
class RecoveryDiagnostics:
    """Wall-point counts from the template surface projection."""

    attempted: int
    accepted: int
    rejected: int
    partial_accepted: int = 0

    @classmethod
    def from_mesh(cls, mesh_data: MeshData) -> RecoveryDiagnostics:
        """Read the projection counts recorded in ``mesh_generation``."""
        projection = mesh_data.get("mesh_generation", {}).get("surface_projection", {})
        attempted = int(projection.get("attempted_points", 0))
        accepted = int(projection.get("accepted_points", 0))
        partial = int(projection.get("partial_accepted_points", 0))
        rejected = max(0, attempted - accepted)
        return cls(attempted, accepted, rejected, partial)

    def as_dict(self) -> dict[str, int]:
        """Return a serialisable diagnostics snapshot."""
        return {
            "attempted": self.attempted,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "partial_accepted": self.partial_accepted,
        }


@dataclass(frozen=True, slots=True)
class OptimisationDiagnostics:
    """Quality values measured after a mesh stage."""

    max_non_orthogonality_deg: float | None = None
    max_skewness: float | None = None
    max_aspect_ratio: float | None = None

    @classmethod
    def from_quality(cls, quality: dict) -> OptimisationDiagnostics:
        """Adapt mesh-quality measurements into a typed stage result."""
        return cls(
            max_non_orthogonality_deg=_optional_float(quality, "max_non_orthogonality_deg"),
            max_skewness=_optional_float(quality, "max_skewness"),
            max_aspect_ratio=_optional_float(quality, "max_aspect_ratio"),
        )

    def as_dict(self) -> dict[str, float | None]:
        """Return a serialisable diagnostics snapshot."""
        return {
            "max_non_orthogonality_deg": self.max_non_orthogonality_deg,
            "max_skewness": self.max_skewness,
            "max_aspect_ratio": self.max_aspect_ratio,
        }


def _optional_float(values: dict, key: str) -> float | None:
    value = values.get(key)
    return None if value is None else float(value)


@dataclass(frozen=True, slots=True)
class GenerationReport:
    """Immutable source information and diagnostics for one successful mesh build.

    Parameters
    ----------
    method : str
        Meshing workflow identifier.
    sizes : tuple[SizeReport, ...]
        Requested/effective sizing decisions in m.
    boundary_patches : tuple[str, ...]
        Contiguous native boundary-patch names in output order.
    surface_hashes : tuple[str, ...]
        SHA-256 digests of the exact input surface files.
    diagnostics : collections.abc.Mapping
        Nested quality/topology measurements. Construction recursively freezes
        mappings and sequences so callers cannot mutate report state.

    Notes
    -----
    :meth:`as_dict` returns a detached JSON-compatible copy suitable for run
    metadata_files; changing that copy does not affect this report.
    """

    method: str
    sizes: tuple[SizeReport, ...]
    boundary_patches: tuple[str, ...]
    surface_hashes: tuple[str, ...]
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sizes", tuple(self.sizes))
        object.__setattr__(self, "boundary_patches", tuple(self.boundary_patches))
        object.__setattr__(self, "surface_hashes", tuple(self.surface_hashes))
        object.__setattr__(self, "diagnostics", _freeze(self.diagnostics))

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible report snapshot."""
        return {
            "method": self.method,
            "sizes": [size.as_dict() for size in self.sizes],
            "boundary_patches": list(self.boundary_patches),
            "surface_hashes": list(self.surface_hashes),
            "diagnostics": _thaw(self.diagnostics),
        }


__all__ = ["GenerationReport", "OptimisationDiagnostics", "RecoveryDiagnostics", "SizeReport"]
