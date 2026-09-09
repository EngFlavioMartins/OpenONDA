# SPDX-License-Identifier: GPL-3.0-or-later
"""Immutable reports emitted by Cartesian-mesher builds."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


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
    """Requested and realizable dyadic size for one named meshing control.

    Parameters
    ----------
    name : str
        Diagnostic control name, such as ``"background"`` or ``"patch:wall"``.
    requested, effective : float
        Requested upper size and realized power-of-two size in m. The effective
        value is no larger than the request.
    level : int
        Non-negative refinement level relative to the Cartesian root box.
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
class GenerationReport:
    """Immutable provenance and diagnostics for one successful mesh build.

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
    manifests; changing that copy does not affect this report.
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


__all__ = ["GenerationReport", "SizeReport"]
