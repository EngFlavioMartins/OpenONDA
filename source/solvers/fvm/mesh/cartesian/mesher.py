# SPDX-License-Identifier: GPL-3.0-or-later
"""Public orchestration object for the native Cartesian mesher."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, cast

import numpy as np

from ..adaptive_cartesian import AdaptiveCartesianMesher as _LegacyMesher
from ..adaptive_cartesian import BoxRefinement as _LegacyBoxRefinement
from ..boundary_layer import stitch_boundary_layer
from ..geometry import compute_mesh_geometry
from ..surface_classification import SurfaceIndex
from ..triangulated_surface import TriangulatedSurface
from ..validation import validate_geometry, validate_topology
from .boundary_layers import build_patch_layers
from .config import (
    BoundaryLayers,
    Bounds,
    BoxDomain,
    CompositeSizeField,
    FeatureRefinement,
    Refinement,
    STLSurface,
    _refinement_attribute,
)
from .features import classify_features
from .native_mesh import require_native_mesh
from .optimisation import OptimisationDiagnostics
from .report import GenerationReport, SizeReport
from .surface_recovery import RecoveryDiagnostics


def _dyadic_size(background: float, requested: float) -> tuple[float, int]:
    if not math.isfinite(requested) or requested <= 0.0:
        raise ValueError("requested cell sizes must be finite and positive")
    if requested >= background:
        return background, 0
    level = max(0, int(math.ceil(math.log2(background / requested) - 1.0e-12)))
    return background / (2**level), level


def _combined_surface(surfaces: tuple[STLSurface, ...]) -> TriangulatedSurface:
    """Create one immutable geometric authority for multi-surface extraction."""
    if len(surfaces) == 1:
        return surfaces[0].surface_data
    triangles = np.ascontiguousarray(np.concatenate([surface.triangles for surface in surfaces]))
    triangles.setflags(write=False)
    lower = triangles.min(axis=(0, 1))
    upper = triangles.max(axis=(0, 1))
    bounds = tuple(float(value) for axis in range(3) for value in (lower[axis], upper[axis]))
    return TriangulatedSurface(
        path=Path("<combined-surface>"),
        triangles=triangles,
        bounds=bounds,  # type: ignore[arg-type]
        sha256=hashlib.sha256(triangles.tobytes()).hexdigest(),
        kind="multi_box" if all(surface.kind == "box" for surface in surfaces) else "general",
    )


def _rename_boundary_patches(
    mesh_data: dict[str, Any],
    domain: BoxDomain,
    surfaces: tuple[STLSurface, ...],
    source_wall_name: str,
) -> None:
    """Apply declarative patch names while preserving contiguous face ranges."""
    source_faces = mesh_data["faces"]
    source_owners = np.asarray(mesh_data["owners"])
    n_internal = int(mesh_data["n_interior_faces"])
    internal_faces = list(source_faces[:n_internal])
    internal_owners = list(source_owners[:n_internal])
    source_blocks = {
        patch["name"]: (
            int(patch["start_face"]),
            int(patch["n_faces"]),
            patch.get("type", "patch"),
        )
        for patch in mesh_data["boundary"]
    }
    logical_sources = ("inlet", "outlet", "ymin", "ymax", "zmin", "zmax")
    logical_targets = domain.patches.as_tuple()
    grouped: dict[str, list[tuple[str, str]]] = {}
    order: list[str] = []
    for source, target in zip(logical_sources, logical_targets, strict=True):
        if target not in grouped:
            grouped[target] = []
            order.append(target)
        grouped[target].append((source, "patch"))
    if source_wall_name not in source_blocks:
        raise RuntimeError(f"Native extraction did not produce source patch {source_wall_name!r}")
    first, count, _source_type = source_blocks[source_wall_name]
    source_wall_faces = source_faces[first : first + count]
    source_wall_owners = source_owners[first : first + count]
    wall_faces: dict[str, list[Any]] = {surface.patch: [] for surface in surfaces}
    wall_owners: dict[str, list[Any]] = {surface.patch: [] for surface in surfaces}
    indices = [SurfaceIndex.build(surface.triangles) for surface in surfaces]
    points = np.asarray(mesh_data["vertex_position"], dtype=np.float64)
    for face, owner in zip(source_wall_faces, source_wall_owners, strict=True):
        centre = points[np.asarray(face, dtype=np.int64)].mean(axis=0)
        distances = [index.nearest_point(centre)[1] for index in indices]
        target = surfaces[int(np.argmin(distances))].patch
        wall_faces[target].append(face)
        wall_owners[target].append(owner)
    for surface in surfaces:
        if not wall_faces[surface.patch]:
            raise RuntimeError(
                f"Native extraction did not produce wall faces for surface patch {surface.patch!r}"
            )
        grouped[surface.patch] = []
        order.append(surface.patch)

    faces = internal_faces
    owners = internal_owners
    boundaries: list[dict[str, Any]] = []
    start = n_internal
    for target in order:
        target_faces = []
        target_owners = []
        target_type = "patch"
        for source, block_type in grouped[target]:
            if source not in source_blocks:
                raise RuntimeError(f"Native extraction did not produce source patch {source!r}")
            first, count, source_type = source_blocks[source]
            target_faces.extend(source_faces[first : first + count])
            target_owners.extend(source_owners[first : first + count])
            target_type = "wall" if block_type == "wall" or source_type == "wall" else target_type
        if target in wall_faces:
            target_faces.extend(wall_faces[target])
            target_owners.extend(wall_owners[target])
            target_type = "wall"
        faces.extend(target_faces)
        owners.extend(target_owners)
        boundaries.append(
            {"name": target, "start_face": start, "n_faces": len(target_faces), "type": target_type}
        )
        start += len(target_faces)
    mesh_data["faces"] = faces
    mesh_data["owners"] = np.asarray(owners, dtype=np.int32)
    mesh_data["boundary"] = boundaries
    mesh_data["n_faces"] = len(faces)


def _quality_snapshot(mesh_data: dict[str, Any]) -> dict[str, Any]:
    """Compute the authoritative native geometry/quality evidence."""
    topology = validate_topology(mesh_data)
    geometry = compute_mesh_geometry(mesh_data, compute_lsq=False)
    quality: dict[str, Any] = dict(validate_geometry(mesh_data, geometry))
    face_areas = np.asarray(geometry["face_area"])
    quality["patch_areas"] = {
        str(patch["name"]): float(
            face_areas[
                int(patch["start_face"]) : int(patch["start_face"]) + int(patch["n_faces"])
            ].sum()
        )
        for patch in mesh_data["boundary"]
    }
    return {**topology, **quality}


def _surface_distance_snapshot(
    mesh_data: dict[str, Any], surfaces: tuple[STLSurface, ...]
) -> dict[str, dict[str, float]]:
    """Measure configured wall vertices against their authoritative surfaces."""
    points = np.asarray(mesh_data["vertex_position"], dtype=np.float64)
    result: dict[str, dict[str, float]] = {}
    for surface in surfaces:
        patch = next(item for item in mesh_data["boundary"] if item["name"] == surface.patch)
        start = int(patch["start_face"])
        stop = start + int(patch["n_faces"])
        vertex_ids = np.unique(
            np.concatenate(
                [np.asarray(face, dtype=np.int64) for face in mesh_data["faces"][start:stop]]
            )
        )
        index = SurfaceIndex.build(surface.triangles)
        distances = np.asarray(
            [index.nearest_point(points[int(vertex_id)])[1] for vertex_id in vertex_ids],
            dtype=np.float64,
        )
        result[surface.patch] = {
            "max": float(distances.max(initial=0.0)),
            "mean": float(distances.mean() if len(distances) else 0.0),
        }
    return result


class CartesianMesher:
    """Build an OpenONDA-native Cartesian mesh from declarative inputs.

    The serial Cartesian core is assembled through typed surface, size-field,
    extraction, recovery, layer, and reporting stages. All accepted controls
    are geometry-independent; unsupported topology or invalid configuration
    raises a diagnostic error instead of being silently ignored.
    """

    def __init__(
        self,
        *,
        domain: BoxDomain,
        surfaces: tuple[STLSurface, ...],
        max_cell_size: float,
        boundary_cell_size: float | None = None,
        min_cell_size: float | None = None,
        refinements: tuple[Refinement, ...] = (),
        features: FeatureRefinement | None = None,
        boundary_layers: tuple[BoundaryLayers, ...] = (),
        surface_may_cross_domain_boundary: bool = False,
    ) -> None:
        if not isinstance(domain, BoxDomain):
            raise TypeError("domain must be a BoxDomain instance")
        if not surfaces:
            raise ValueError("surfaces must contain at least one STLSurface")
        if any(not isinstance(surface, STLSurface) for surface in surfaces):
            raise TypeError("surfaces must contain only STLSurface instances")
        max_cell_size = float(max_cell_size)
        if not math.isfinite(max_cell_size) or max_cell_size <= 0.0:
            raise ValueError("max_cell_size must be finite and positive")
        if boundary_cell_size is None:
            boundary_cell_size = max_cell_size
        boundary_cell_size = float(boundary_cell_size)
        if not math.isfinite(boundary_cell_size) or boundary_cell_size <= 0.0:
            raise ValueError("boundary_cell_size must be finite and positive")
        if boundary_cell_size > max_cell_size:
            raise ValueError("boundary_cell_size must not exceed max_cell_size")
        if not isinstance(surface_may_cross_domain_boundary, bool):
            raise TypeError("surface_may_cross_domain_boundary must be a bool")
        if min_cell_size is not None:
            min_cell_size = float(min_cell_size)
            if not math.isfinite(min_cell_size) or min_cell_size <= 0.0:
                raise ValueError("min_cell_size must be finite and positive")
            if min_cell_size > boundary_cell_size:
                raise ValueError("min_cell_size must not exceed boundary_cell_size")
        surfaces = tuple(surfaces)
        patches = tuple(surface.patch for surface in surfaces)
        if len(set(patches)) != len(patches):
            raise ValueError("surface patch names must be unique")
        if set(patches) & set(domain.patches.as_tuple()):
            raise ValueError("surface and outer-domain patch names must be distinct")
        for surface in surfaces:
            if surface_may_cross_domain_boundary:
                intersects = all(
                    domain.bounds[2 * axis] < surface.bounds[2 * axis + 1]
                    and surface.bounds[2 * axis] < domain.bounds[2 * axis + 1]
                    for axis in range(3)
                )
                if not intersects:
                    raise ValueError(f"surface {surface.patch!r} does not overlap domain")
            elif not all(
                domain.bounds[2 * axis] < surface.bounds[2 * axis]
                and surface.bounds[2 * axis + 1] < domain.bounds[2 * axis + 1]
                for axis in range(3)
            ):
                raise ValueError(f"surface {surface.patch!r} must lie strictly inside domain")
        refinements = tuple(refinements)
        if any(not isinstance(refinement, Refinement) for refinement in refinements):
            raise TypeError("refinements must contain only typed refinement objects")
        if features is not None and not isinstance(features, FeatureRefinement):
            raise TypeError("features must be a FeatureRefinement instance or None")
        boundary_layers = tuple(boundary_layers)
        if any(not isinstance(layer, BoundaryLayers) for layer in boundary_layers):
            raise TypeError("boundary_layers must contain only BoundaryLayers instances")
        known_patches = set(patches)
        for refinement in refinements:
            bounds = tuple(float(value) for value in _refinement_attribute(refinement, "bounds"))
            if not all(
                domain.bounds[2 * axis] <= bounds[2 * axis]
                and bounds[2 * axis + 1] <= domain.bounds[2 * axis + 1]
                for axis in range(3)
            ):
                raise ValueError(
                    f"refinement {str(_refinement_attribute(refinement, 'name'))!r} "
                    "must lie inside domain"
                )
        for layer in boundary_layers:
            unknown = sorted(set(layer.patches) - known_patches)
            if unknown:
                raise ValueError(f"boundary layer refers to unknown surface patches: {unknown}")

        self.domain = domain
        self.surfaces = surfaces
        self.max_cell_size = max_cell_size
        self.boundary_cell_size = boundary_cell_size
        self.min_cell_size = min_cell_size
        self.refinements = refinements
        self.features = features
        self.boundary_layers = boundary_layers
        self.surface_may_cross_domain_boundary = surface_may_cross_domain_boundary
        self.size_field = CompositeSizeField(
            background_size=max_cell_size,
            refinements=refinements,
            minimum_size=min_cell_size,
        )
        self._report: GenerationReport | None = None
        self._feature_sets = (
            tuple(
                classify_features(surface.triangles, features.angle).edges for surface in surfaces
            )
            if features is not None
            else ()
        )

    @property
    def report(self) -> GenerationReport | None:
        """Immutable generation report, available after a successful build."""
        return self._report

    def effective_cell_size(self, requested: float) -> float:
        """Return the dyadic size selected for a requested upper size."""
        effective, _level = _dyadic_size(self.max_cell_size, requested)
        if self.min_cell_size is not None:
            effective = max(effective, self.min_cell_size)
        return effective

    def _legacy_refinements(self) -> tuple[_LegacyBoxRefinement, ...]:
        return tuple(
            _LegacyBoxRefinement(
                cast(
                    Bounds,
                    tuple(float(value) for value in _refinement_attribute(refinement, "bounds")),
                ),
                self.effective_cell_size(float(_refinement_attribute(refinement, "cell_size"))),
                str(_refinement_attribute(refinement, "name")),
            )
            for refinement in self.refinements
        )

    def _legacy_mesher(self) -> _LegacyMesher:
        surface = _combined_surface(self.surfaces)
        surface_size = self.effective_cell_size(self.boundary_cell_size)
        if self.features is not None:
            surface_size = self.effective_cell_size(min(surface_size, self.features.cell_size))
        exact_surface_components = (
            tuple(surface.bounds for surface in self.surfaces)
            if (
                not self.boundary_layers
                and len(self.surfaces) > 1
                and all(surface.kind == "box" for surface in self.surfaces)
            )
            else ()
        )
        layer_thickness = max(
            (sum(layer.layer_heights) for layer in self.boundary_layers),
            default=0.0,
        )
        return _LegacyMesher(
            domain=self.domain.bounds,
            max_cell_size=self.max_cell_size,
            surface_data=surface,
            exact_surface_components=exact_surface_components,
            surface_exclusion_distance=layer_thickness,
            skip_surface_recovery=bool(self.boundary_layers),
            surface_may_cross_domain_boundary=self.surface_may_cross_domain_boundary,
            wall_patch_name="__cartesian_surface__",
            surface_cell_size=surface_size,
            refinements=self._legacy_refinements(),
        )

    def _apply_boundary_layers(self, mesh_data: dict[str, Any]) -> dict[str, Any]:
        """Extrude each selected wall patch and stitch it to the core."""
        result = mesh_data
        for layer in self.boundary_layers:
            for patch_name in layer.patches:
                surface = next(surface for surface in self.surfaces if surface.patch == patch_name)
                interface_name = f"__cartesian_layer_interface_{patch_name}__"
                layer_mesh = build_patch_layers(
                    result,
                    SurfaceIndex.build(surface.triangles),
                    layer,
                    patch_name,
                    interface_name,
                )
                for patch in result["boundary"]:
                    if patch["name"] == patch_name:
                        patch["name"] = interface_name
                        patch["type"] = "patch"
                        break
                else:
                    raise ValueError(f"Boundary-layer patch {patch_name!r} is missing")
                result = stitch_boundary_layer(result, layer_mesh, interface_name)
        return result

    def build(self) -> dict[str, Any]:
        """Build, validate, name, and return native face-based mesh data."""
        mesher = self._legacy_mesher()
        mesh_data = require_native_mesh(mesher.build())
        _rename_boundary_patches(
            mesh_data,
            self.domain,
            self.surfaces,
            "__cartesian_surface__",
        )
        if self.boundary_layers:
            mesh_data = self._apply_boundary_layers(mesh_data)
        quality = _quality_snapshot(mesh_data)
        quality["surface_distance"] = _surface_distance_snapshot(mesh_data, self.surfaces)
        recovery = RecoveryDiagnostics.from_mesh(mesh_data)
        optimisation = OptimisationDiagnostics.from_quality(quality)
        requested_sizes = [("background", self.max_cell_size)]
        requested_sizes.append(("boundary", self.boundary_cell_size))
        if self.min_cell_size is not None:
            requested_sizes.append(("minimum", self.min_cell_size))
        requested_sizes.extend(
            (
                str(_refinement_attribute(refinement, "name")),
                float(_refinement_attribute(refinement, "cell_size")),
            )
            for refinement in self.refinements
        )
        if self.features is not None:
            requested_sizes.append(("features", self.features.cell_size))
        sizes = tuple(
            SizeReport(name, requested, *(_dyadic_size(self.max_cell_size, requested)))
            for name, requested in requested_sizes
        )
        diagnostics = {
            "quality": quality,
            "recovery": recovery.as_dict(),
            "optimisation": optimisation.as_dict(),
            "surface_count": len(self.surfaces),
            "surface_patches": tuple(surface.patch for surface in self.surfaces),
            "refinement_count": len(self.refinements),
            "feature_control": self.features is not None,
            "feature_edge_counts": tuple(len(edges) for edges in self._feature_sets),
            "layer_control": bool(self.boundary_layers),
        }
        self._report = GenerationReport(
            method="cartesian.adapter",
            sizes=sizes,
            boundary_patches=tuple(patch["name"] for patch in mesh_data["boundary"]),
            surface_hashes=tuple(surface.sha256 for surface in self.surfaces),
            diagnostics=diagnostics,
        )
        generation = mesh_data.setdefault("mesh_generation", {})
        generation.update(
            {
                "method": "cartesian.adapter",
                "requested_sizes": [size.as_dict() for size in sizes],
                "cartesian_report": self._report.as_dict(),
            }
        )
        return mesh_data

    def __call__(self) -> dict[str, Any]:
        """Make the mesher directly callable for solver factory integration."""
        return self.build()


__all__ = ["CartesianMesher"]
