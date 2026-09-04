# SPDX-License-Identifier: GPL-3.0-or-later
"""Public orchestration object for the native Cartesian mesher."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, cast

import numpy as np

from ..geometry import compute_mesh_geometry
from ..surface_classification import SurfaceIndex
from ..triangulated_surface import TriangulatedSurface
from ..validation import (
    extract_cell_subset_mesh,
    validate_geometry,
    validate_no_fluid_cell_centres_inside_surface,
    validate_topology,
    validate_wall_vertex_conformance,
)
from .boundary_layers import (
    LayerSurface,
    build_layer_surface,
    insert_default_surface_layer,
    insert_surface_layers,
)
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
from .octree import (
    CartesianOctree,
    _compact_conformed_topology,
    _conform_wall_to_surface,
    _prepare_wall_topology,
    _remove_non_mappable_surface_cells,
)
from .optimisation import (
    OptimisationDiagnostics,
    agglomerate_small_cut_cells,
    agglomerate_small_layer_columns,
)
from .report import GenerationReport, SizeReport
from .surface_recovery import RecoveryDiagnostics, recover_cut_cells


class _EffectiveRefinement:
    """Carry a typed refinement into the octree without changing its shape."""

    def __init__(self, source: Refinement, cell_size: float) -> None:
        self.source = source
        self.name = str(_refinement_attribute(source, "name"))
        self.bounds = cast(
            Bounds,
            tuple(float(value) for value in _refinement_attribute(source, "bounds")),
        )
        self.cell_size = float(cell_size)

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Delegate point containment to the immutable typed control."""
        return self.source.contains(points)

    def intersects_box(self, lower: np.ndarray, upper: np.ndarray) -> bool:
        """Delegate conservative cell intersection to the typed control."""
        return self.source.intersects_box(lower, upper)


def _dyadic_size(background: float, requested: float) -> tuple[float, int]:
    if not math.isfinite(requested) or requested <= 0.0:
        raise ValueError("requested cell sizes must be finite and positive")
    if requested >= background:
        return background, 0
    level = max(0, int(math.ceil(math.log2(background / requested) - 1.0e-12)))
    return background / (2**level), level


def _combined_surface(
    surfaces: tuple[STLSurface, ...],
    triangle_sets: tuple[np.ndarray, ...] | None = None,
) -> TriangulatedSurface:
    """Create one immutable geometric authority for multi-surface extraction."""
    if triangle_sets is None and len(surfaces) == 1:
        return surfaces[0].surface_data
    authority = triangle_sets or tuple(surface.triangles for surface in surfaces)
    triangles = np.ascontiguousarray(np.concatenate(authority))
    triangles.setflags(write=False)
    lower = triangles.min(axis=(0, 1))
    upper = triangles.max(axis=(0, 1))
    bounds = tuple(float(value) for axis in range(3) for value in (lower[axis], upper[axis]))
    return TriangulatedSurface(
        path=Path("<combined-surface>"),
        triangles=triangles,
        bounds=bounds,  # type: ignore[arg-type]
        sha256=hashlib.sha256(triangles.tobytes()).hexdigest(),
        kind=(
            "multi_box"
            if triangle_sets is None and all(surface.kind == "box" for surface in surfaces)
            else "general"
        ),
    )


def _rename_boundary_patches(
    mesh_data: dict[str, Any],
    domain: BoxDomain,
    surfaces: tuple[STLSurface, ...],
    source_wall_name: str,
    *,
    surface_triangles: tuple[np.ndarray, ...] | None = None,
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
    authority = surface_triangles or tuple(surface.triangles for surface in surfaces)
    indices = [SurfaceIndex.build(triangles) for triangles in authority]
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
    mesh_data.pop("cell_face_indices", None)
    mesh_data.pop("cell_face_offset", None)


def _quality_snapshot(mesh_data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute the authoritative native geometry/quality evidence."""
    topology = validate_topology(mesh_data)
    geometry = compute_mesh_geometry(mesh_data, compute_lsq=False)
    quality: dict[str, Any] = dict(validate_geometry(mesh_data, geometry))
    skew_face = int(quality.get("max_skewness_face", -1))
    if skew_face >= 0:
        owner = int(mesh_data["owners"][skew_face])
        neighbour = (
            int(mesh_data["neighbours"][skew_face])
            if skew_face < int(mesh_data["n_interior_faces"])
            else None
        )
        layer_index = np.asarray(mesh_data.get("boundary_layer_index", ()), dtype=np.int64)
        quality["max_skewness_context"] = {
            "face": skew_face,
            "face_centre": tuple(map(float, geometry["face_centre"][skew_face])),
            "face_points": tuple(
                tuple(map(float, point))
                for point in np.asarray(mesh_data["vertex_position"])[
                    np.asarray(mesh_data["faces"][skew_face], dtype=np.int64)
                ]
            ),
            "vertices": len(mesh_data["faces"][skew_face]),
            "owner": owner,
            "owner_centre": tuple(map(float, geometry["cell_centre"][owner])),
            "owner_layer": int(layer_index[owner]) if len(layer_index) else None,
            "neighbour": neighbour,
            "neighbour_centre": (
                tuple(map(float, geometry["cell_centre"][neighbour]))
                if neighbour is not None
                else None
            ),
            "neighbour_layer": (
                int(layer_index[neighbour]) if neighbour is not None and len(layer_index) else None
            ),
        }
    face_areas = np.asarray(geometry["face_area"])
    quality["patch_areas"] = {
        str(patch["name"]): float(
            face_areas[
                int(patch["start_face"]) : int(patch["start_face"]) + int(patch["n_faces"])
            ].sum()
        )
        for patch in mesh_data["boundary"]
    }
    return {**topology, **quality}, geometry


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


def _cell_type_counts(mesh_data: dict[str, Any]) -> dict[str, int]:
    """Classify untouched hexahedra versus general polyhedral control volumes."""
    n_cells = int(mesh_data["n_cells"])
    n_internal = int(mesh_data["n_interior_faces"])
    owners = np.asarray(mesh_data["owners"], dtype=np.int64)
    neighbours = np.asarray(mesh_data["neighbours"], dtype=np.int64)
    face_count = np.zeros(n_cells, dtype=np.int16)
    non_quad_count = np.zeros(n_cells, dtype=np.int16)
    np.add.at(face_count, owners, 1)
    np.add.at(face_count, neighbours, 1)
    non_quad = np.fromiter(
        (len(face) != 4 for face in mesh_data["faces"]),
        dtype=bool,
        count=int(mesh_data["n_faces"]),
    )
    np.add.at(non_quad_count, owners[non_quad], 1)
    internal_non_quad = non_quad[:n_internal]
    np.add.at(non_quad_count, neighbours[internal_non_quad], 1)
    hexahedra = (face_count == 6) & (non_quad_count == 0)
    labels = np.asarray(mesh_data.get("boundary_layer_index", ()), dtype=np.int16)
    return {
        "hexahedra": int(np.count_nonzero(hexahedra)),
        "polyhedra": int(n_cells - np.count_nonzero(hexahedra)),
        "boundary_layer_cells": int(
            np.count_nonzero(labels >= 0) if labels.shape == (n_cells,) else 0
        ),
    }


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

    def _octree_refinements(self) -> tuple[_EffectiveRefinement, ...]:
        return tuple(
            _EffectiveRefinement(
                refinement,
                self.effective_cell_size(float(_refinement_attribute(refinement, "cell_size"))),
            )
            for refinement in self.refinements
        )

    def _background_mesher(
        self, surface_triangles: tuple[np.ndarray, ...] | None = None
    ) -> CartesianOctree:
        surface = _combined_surface(self.surfaces, surface_triangles)
        surface_size = self.effective_cell_size(self.boundary_cell_size)
        if self.features is not None:
            surface_size = self.effective_cell_size(min(surface_size, self.features.cell_size))
        exact_surface_components = (
            tuple(surface.bounds for surface in self.surfaces)
            if (
                surface_triangles is None
                and not self.boundary_layers
                and len(self.surfaces) > 1
                and all(surface.kind == "box" for surface in self.surfaces)
            )
            else ()
        )
        return CartesianOctree(
            domain=self.domain.bounds,
            max_cell_size=self.max_cell_size,
            surface_data=surface,
            exact_surface_components=exact_surface_components,
            surface_exclusion_distance=0.0,
            # General STL recovery is the separate transactional cut-cell
            # stage below; the octree constructs only the background topology.
            surface_may_cross_domain_boundary=self.surface_may_cross_domain_boundary,
            wall_patch_name="__cartesian_surface__",
            surface_cell_size=surface_size,
            refinements=cast(Any, self._octree_refinements()),
        )

    def build(self) -> dict[str, Any]:
        """Build, validate, name, and return native face-based mesh data."""
        layer_surfaces: list[LayerSurface] = []
        layer_specs: list[BoundaryLayers] = []
        authority = [surface.triangles for surface in self.surfaces]
        if self.boundary_layers:
            feature_angle = self.features.angle if self.features is not None else 45.0
            for layer in self.boundary_layers:
                for patch_name in layer.patches:
                    surface_id = next(
                        index
                        for index, surface in enumerate(self.surfaces)
                        if surface.patch == patch_name
                    )
                    geometry = build_layer_surface(
                        self.surfaces[surface_id].triangles,
                        patch_name,
                        layer,
                        feature_angle_degrees=feature_angle,
                    )
                    authority[surface_id] = geometry.outer_triangles
                    layer_surfaces.append(geometry)
                    layer_specs.append(layer)
        authority_tuple = tuple(authority)
        mesher = self._background_mesher(authority_tuple if self.boundary_layers else None)
        mesh_data = require_native_mesh(mesher.build())
        feature_angle = self.features.angle if self.features is not None else 45.0

        def has_active_sharp_edges(surface: STLSurface) -> bool:
            bounds = self.domain.bounds
            features = classify_features(surface.triangles, feature_angle)
            return any(
                all(
                    max(edge.start[axis], edge.end[axis]) >= bounds[2 * axis]
                    and min(edge.start[axis], edge.end[axis]) <= bounds[2 * axis + 1]
                    for axis in range(3)
                )
                for edge in features.edges
            )

        has_sharp_surface = any(
            surface.kind != "box" and has_active_sharp_edges(surface) for surface in self.surfaces
        )
        extracted_wall_faces = int(
            next(
                (
                    patch["n_faces"]
                    for patch in mesh_data["boundary"]
                    if patch["name"] == "__cartesian_surface__"
                ),
                0,
            )
        )
        used_cut_recovery = bool(
            self.boundary_layers or has_sharp_surface or extracted_wall_faces == 0
        )
        if any(surface.kind != "box" for surface in self.surfaces):
            if used_cut_recovery:
                mesh_data = recover_cut_cells(
                    mesh_data,
                    tuple(SurfaceIndex.build(triangles) for triangles in authority_tuple),
                    "__cartesian_surface__",
                )
            else:
                combined_triangles = np.concatenate(authority_tuple, axis=0)
                surface_index = SurfaceIndex.build(combined_triangles)
                from ...io.vtk_exporter import VTKExporter

                try:
                    from vtk import vtkCellValidator  # pyrefly: ignore[missing-import]
                    from vtk.util.numpy_support import (  # pyrefly: ignore[missing-import]
                        vtk_to_numpy,  # pyrefly: ignore[missing-import]
                    )
                except ImportError as exc:  # pragma: no cover - VTK is an FVM dependency.
                    raise ValueError("VTK is required for surface-topology preparation") from exc
                # Removing a non-mappable cell exposes its retained neighbour as
                # the next surface cell.  Continue to the geometric fixed point;
                # the long-cylinder/domain intersection needs 17 passes at D/12.
                for _surface_iteration in range(64):
                    _prepare_wall_topology(mesh_data, "__cartesian_surface__")
                    _conform_wall_to_surface(
                        mesh_data,
                        surface_index,
                        "__cartesian_surface__",
                        fixed_bounds=self.domain.bounds,
                    )
                    mesh_data.pop("cell_face_indices", None)
                    mesh_data.pop("cell_face_offset", None)
                    validation_view = dict(mesh_data)
                    _compact_conformed_topology(validation_view)
                    raw_wall_patch = next(
                        patch
                        for patch in mesh_data["boundary"]
                        if patch["name"] == "__cartesian_surface__"
                    )
                    raw_wall_start = int(raw_wall_patch["start_face"])
                    raw_wall_stop = raw_wall_start + int(raw_wall_patch["n_faces"])
                    raw_wall_points = np.unique(
                        np.concatenate(
                            [
                                np.asarray(face, dtype=np.int32)
                                for face in mesh_data["faces"][raw_wall_start:raw_wall_stop]
                            ]
                        )
                    )
                    raw_cell_vertices = np.asarray(mesh_data["cell_vertex_indices"], dtype=np.int32)
                    validation_cell_ids = np.flatnonzero(
                        np.any(np.isin(raw_cell_vertices, raw_wall_points), axis=1)
                    )
                    validator = vtkCellValidator()
                    validator.SetInputData(
                        VTKExporter(
                            extract_cell_subset_mesh(validation_view, validation_cell_ids)
                        )._grid
                    )
                    validator.Update()
                    state_array = validator.GetOutput().GetCellData().GetArray("ValidityState")
                    if state_array is None:
                        raise ValueError("VTK cell validator did not return ValidityState")
                    states = np.asarray(vtk_to_numpy(state_array), dtype=np.int64)
                    non_mappable_cells = set(
                        map(int, validation_cell_ids[np.flatnonzero(states & 0b000110)])
                    )
                    prepared_geometry = compute_mesh_geometry(validation_view, compute_lsq=False)
                    non_mappable_cells.update(
                        map(
                            int,
                            np.flatnonzero(np.asarray(prepared_geometry["cell_volume"]) <= 0.0),
                        )
                    )
                    non_mappable_cells.update(
                        map(
                            int,
                            validation_cell_ids[
                                np.flatnonzero(
                                    surface_index.is_inside(
                                        prepared_geometry["cell_centre"][validation_cell_ids]
                                    )
                                )
                            ],
                        )
                    )
                    prepared_area = np.asarray(prepared_geometry["face_area_vector"])
                    prepared_direction = np.asarray(prepared_geometry["cell_connection_vector"])
                    prepared_face_centre = np.asarray(prepared_geometry["face_centre"])
                    prepared_cell_centre = np.asarray(prepared_geometry["cell_centre"])
                    prepared_owners = np.asarray(validation_view["owners"], dtype=np.int32)
                    prepared_neighbours = np.asarray(validation_view["neighbours"], dtype=np.int32)
                    owner_pyramid = np.einsum(
                        "ij,ij->i",
                        prepared_area,
                        prepared_face_centre - prepared_cell_centre[prepared_owners],
                    )
                    neighbour_pyramid = np.ones(len(prepared_area), dtype=np.float64)
                    prepared_internal = int(validation_view["n_interior_faces"])
                    neighbour_pyramid[:prepared_internal] = np.einsum(
                        "ij,ij->i",
                        prepared_area[:prepared_internal],
                        prepared_cell_centre[prepared_neighbours]
                        - prepared_face_centre[:prepared_internal],
                    )
                    invalid_face_mask = (
                        np.einsum("ij,ij->i", prepared_area, prepared_direction) <= 0.0
                    )
                    if self.surface_may_cross_domain_boundary:
                        invalid_face_mask |= (owner_pyramid <= 0.0) | (neighbour_pyramid <= 0.0)
                    invalid_faces = np.flatnonzero(invalid_face_mask)
                    wall_patch = next(
                        patch
                        for patch in mesh_data["boundary"]
                        if patch["name"] == "__cartesian_surface__"
                    )
                    wall_start = int(wall_patch["start_face"])
                    wall_stop = wall_start + int(wall_patch["n_faces"])
                    wall_owners = set(
                        map(int, np.asarray(mesh_data["owners"])[wall_start:wall_stop])
                    )
                    prepared_faces = [
                        np.asarray(face, dtype=np.int32) for face in validation_view["faces"]
                    ]
                    wall_points = set(
                        map(
                            int,
                            np.unique(np.concatenate(prepared_faces[wall_start:wall_stop])),
                        )
                    )
                    cell_points: list[set[int]] = [set() for _ in range(int(mesh_data["n_cells"]))]
                    for prepared_face_id, prepared_face in enumerate(prepared_faces):
                        prepared_owner = int(mesh_data["owners"][prepared_face_id])
                        cell_points[prepared_owner].update(map(int, prepared_face))
                        if prepared_face_id < int(mesh_data["n_interior_faces"]):
                            prepared_neighbour = int(mesh_data["neighbours"][prepared_face_id])
                            cell_points[prepared_neighbour].update(map(int, prepared_face))
                    n_prepared_internal = int(mesh_data["n_interior_faces"])
                    for face_id_value in invalid_faces:
                        face_id = int(face_id_value)
                        candidates = [int(mesh_data["owners"][face_id])]
                        if face_id < n_prepared_internal:
                            candidates.append(int(mesh_data["neighbours"][face_id]))
                        surface_candidates = [
                            cell
                            for cell in candidates
                            if cell in wall_owners or bool(cell_points[cell] & wall_points)
                        ]
                        non_mappable_cells.update(
                            surface_candidates if surface_candidates else candidates
                        )
                    non_mappable = np.asarray(sorted(non_mappable_cells), dtype=np.int32)
                    if not len(non_mappable):
                        break
                    _remove_non_mappable_surface_cells(
                        mesh_data,
                        non_mappable,
                        "__cartesian_surface__",
                    )
                else:
                    raise ValueError("Surface non-mappable-cell removal did not converge")
                _compact_conformed_topology(mesh_data)
        _rename_boundary_patches(
            mesh_data,
            self.domain,
            self.surfaces,
            "__cartesian_surface__",
            surface_triangles=authority_tuple,
        )
        if used_cut_recovery:
            mesh_data = agglomerate_small_cut_cells(
                mesh_data,
                tuple(surface.patch for surface in self.surfaces),
                surface_indices=tuple(
                    SurfaceIndex.build(triangles) for triangles in authority_tuple
                ),
            )
        if self.boundary_layers:
            mesh_data = insert_surface_layers(
                mesh_data,
                tuple(layer_surfaces),
                tuple(layer_specs),
                self.domain.bounds,
                self.domain.patches.as_tuple(),
            )
            if any(surface.prefer_vectorized_mapping for surface in layer_surfaces):
                mesh_data = agglomerate_small_layer_columns(
                    mesh_data,
                    self.effective_cell_size(self.boundary_cell_size),
                )
        else:
            curved_surfaces = tuple(surface for surface in self.surfaces if surface.kind != "box")
            smooth_curved_patches = tuple(
                surface.patch
                for surface in curved_surfaces
                if not used_cut_recovery and not has_active_sharp_edges(surface)
            )
            sharp_curved_patches = tuple(
                surface.patch
                for surface in curved_surfaces
                if surface.patch not in smooth_curved_patches
            )
            if smooth_curved_patches:
                mesh_data = insert_default_surface_layer(
                    mesh_data,
                    smooth_curved_patches,
                    self.domain.bounds,
                    self.domain.patches.as_tuple(),
                    SurfaceIndex.build(np.concatenate(authority_tuple, axis=0)),
                )
            if sharp_curved_patches:
                mesh_data.setdefault("mesh_generation", {})["sharp_surface_wrapper"] = {
                    "patches": sharp_curved_patches,
                    "method": "snapped_castellated_surface_cells",
                    "reason": (
                        "surface_domain_intersection_uses_snapped_cells"
                        if self.surface_may_cross_domain_boundary
                        else "sharp_edge_corner_columns_require_partitioned_patch_topology"
                    ),
                }
        quality, geometry = _quality_snapshot(mesh_data)
        surface_conformance: dict[str, dict[str, float | int]] = {}
        for surface in self.surfaces:
            if surface.kind == "box":
                continue
            wall = validate_wall_vertex_conformance(
                mesh_data,
                surface.triangles,
                surface.patch,
            )
            centres = validate_no_fluid_cell_centres_inside_surface(
                geometry["cell_centre"],
                surface.triangles,
            )
            surface_conformance[surface.patch] = {**wall, **centres}
        quality["surface_conformance"] = surface_conformance
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
            "surface_may_cross_domain_boundary": self.surface_may_cross_domain_boundary,
            "refinement_count": len(self.refinements),
            "feature_control": self.features is not None,
            "feature_edge_counts": tuple(len(edges) for edges in self._feature_sets),
            "layer_control": bool(self.boundary_layers),
            "cell_types": _cell_type_counts(mesh_data),
        }
        self._report = GenerationReport(
            method="cartesian",
            sizes=sizes,
            boundary_patches=tuple(patch["name"] for patch in mesh_data["boundary"]),
            surface_hashes=tuple(surface.sha256 for surface in self.surfaces),
            diagnostics=diagnostics,
        )
        generation = mesh_data.setdefault("mesh_generation", {})
        generation.update(
            {
                "method": "cartesian",
                "requested_sizes": [size.as_dict() for size in sizes],
                "cartesian_report": self._report.as_dict(),
                "cell_types": diagnostics["cell_types"],
            }
        )
        return mesh_data

    def __call__(self) -> dict[str, Any]:
        """Make the mesher directly callable for solver factory integration."""
        return self.build()


__all__ = ["CartesianMesher"]
