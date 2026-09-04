"""Native adaptive octree construction for the finite-volume solver.

This module provides triangulated-surface input, dyadic local refinement, 2:1
transition bands, fluid-region selection, and direct construction of the
solver's face-based ``mesh_data`` dictionary. It neither reads nor writes an
external solver case and has no external-mesher runtime dependency.

Algorithmic lineage and credit
------------------------------
The workflow is inspired by the open-source cfMesh Cartesian mesher by Dr.
Franjo Juretic and Creative Fields, Ltd. cfMesh creates an octree background
and extracts a predominantly hexahedral/polyhedral mesh.  This is an
independent Python implementation of the octree/extraction stages; it does
not copy cfMesh source code. Generic patch-normal boundary layers are handled
by the typed mesher pipeline.

Both cfMesh and OpenONDA are distributed under GPL-3.0-or-later.  See
``CFMESH_ATTRIBUTION.md`` beside this file for upstream links and the precise
scope of the adaptation.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from math import gcd, lcm
from pathlib import Path
from typing import cast

import numpy as np

from ..surface_classification import SurfaceIndex
from ..triangulated_surface import SurfaceBounds as Bounds
from ..triangulated_surface import TriangulatedSurface
from ..validation import validate_no_fluid_solid_overlap

OUTER_PATCH_NAMES = ("inlet", "outlet", "ymin", "ymax", "zmin", "zmax")


@dataclass(frozen=True)
class BoxRefinement:
    """Request a cell size inside an axis-aligned box.

    The effective size is the first dyadic level not larger than
    ``cell_size``.  For example, a 0.04 background and a 0.0125 request yield
    0.01 cells, matching cfMesh's octree-level behaviour.
    """

    bounds: Bounds
    cell_size: float
    name: str = "refinement"


@dataclass(frozen=True)
class _IntegerBox:
    x0: int
    x1: int
    y0: int
    y1: int
    z0: int
    z1: int

    def expanded(self, amount: int, limits: tuple[int, int, int]) -> _IntegerBox:
        return _IntegerBox(
            max(0, self.x0 - amount),
            min(limits[0], self.x1 + amount),
            max(0, self.y0 - amount),
            min(limits[1], self.y1 + amount),
            max(0, self.z0 - amount),
            min(limits[2], self.z1 + amount),
        )

    def overlaps(self, x0: int, x1: int, y0: int, y1: int, z0: int, z1: int) -> bool:
        return (
            x0 < self.x1
            and x1 > self.x0
            and y0 < self.y1
            and y1 > self.y0
            and z0 < self.z1
            and z1 > self.z0
        )

    def contains(self, x0: int, x1: int, y0: int, y1: int, z0: int, z1: int) -> bool:
        return (
            x0 >= self.x0
            and x1 <= self.x1
            and y0 >= self.y0
            and y1 <= self.y1
            and z0 >= self.z0
            and z1 <= self.z1
        )


@dataclass(frozen=True)
class _CompositeIntegerBox:
    """A set of disjoint lattice-aligned solid boxes."""

    components: tuple[_IntegerBox, ...]

    def expanded(self, amount: int, limits: tuple[int, int, int]) -> _CompositeIntegerBox:
        return _CompositeIntegerBox(
            tuple(component.expanded(amount, limits) for component in self.components)
        )

    def overlaps(self, x0: int, x1: int, y0: int, y1: int, z0: int, z1: int) -> bool:
        return any(component.overlaps(x0, x1, y0, y1, z0, z1) for component in self.components)

    def contains(self, x0: int, x1: int, y0: int, y1: int, z0: int, z1: int) -> bool:
        return any(component.contains(x0, x1, y0, y1, z0, z1) for component in self.components)


_SolidRegion = _IntegerBox | _CompositeIntegerBox


class _WorldRefinementRegion:
    """Evaluate a typed refinement against world-space Cartesian cells."""

    def __init__(self, refinement, h_min: float, origin: tuple[float, float, float]) -> None:
        self.refinement = refinement
        self.h_min = float(h_min)
        self.origin = np.asarray(origin, dtype=np.float64)
        indices = []
        for axis in range(3):
            lower = (refinement.bounds[2 * axis] - self.origin[axis]) / self.h_min
            upper = (refinement.bounds[2 * axis + 1] - self.origin[axis]) / self.h_min
            indices.extend((math.floor(lower), math.ceil(upper)))
        self._integer_bounds = _IntegerBox(*indices)

    def expanded(self, amount: int, limits: tuple[int, int, int]) -> _IntegerBox:
        """Return an integer transition band around the primitive bounds."""
        return self._integer_bounds.expanded(amount, limits)

    def overlaps(self, x0: int, x1: int, y0: int, y1: int, z0: int, z1: int) -> bool:
        """Test the current Cartesian cell against the primitive volume."""
        lower = self.origin + self.h_min * np.asarray((x0, y0, z0), dtype=np.float64)
        upper = self.origin + self.h_min * np.asarray((x1, y1, z1), dtype=np.float64)
        return bool(self.refinement.intersects_box(lower, upper))


_RefinementRegion = _SolidRegion | _WorldRefinementRegion


class _SurfaceSolid:
    """Real curved-surface classifier, duck-typed to ``_IntegerBox``'s
    ``contains``/``overlaps`` interface so the octree traversal in
    ``_build_leaves`` needs no branching between the box and general paths.
    """

    def __init__(
        self,
        index: SurfaceIndex,
        h_min: float,
        origin: tuple[float, float, float],
        exclusion_distance: float = 0.0,
    ) -> None:
        self.index = index
        self.h_min = h_min
        self.origin = np.asarray(origin, dtype=np.float64)
        self.exclusion_distance = exclusion_distance

    def _world_bounds(
        self, x0: int, x1: int, y0: int, y1: int, z0: int, z1: int
    ) -> tuple[np.ndarray, np.ndarray]:
        lo = self.origin + np.array([x0, y0, z0], dtype=np.float64) * self.h_min
        hi = self.origin + np.array([x1, y1, z1], dtype=np.float64) * self.h_min
        return lo, hi

    def contains(self, x0: int, x1: int, y0: int, y1: int, z0: int, z1: int) -> bool:
        lo, hi = self._world_bounds(x0, x1, y0, y1, z0, z1)
        # Classify a general-surface leaf by its centre, as cfMesh-style
        # Cartesian extraction does before mapping the retained fluid-side
        # boundary onto the STL. Keeping every intersected leaf -- including
        # leaves whose centres are solid -- creates a layer of nominal fluid
        # cells inside curved bodies. At lattice-aligned caps it can also
        # collapse adjacent interior faces when those cells are projected.
        # ``overlaps`` still drives full surface refinement; this predicate
        # decides only which side of the interface owns the leaf.
        # Keep every surface-intersected finest cell for the transactional
        # cut-cell stage, including cells whose centre lies just inside the
        # solid.  Centre-only retention loses complete wall sectors whenever
        # the intersected leaf on one side of a facet happens to own the solid
        # centre (the axis sectors of a round body expose this immediately).
        if self.index.box_intersects_surface(lo, hi):
            return False
        centre = 0.5 * (lo + hi)
        if bool(self.index.is_inside(centre[None, :])[0]):
            return True
        # Intersected cells on the fluid side are retained.  The staged native
        # recovery transaction clips them against the exact STL triangles;
        # deleting them here recreates the staircase that recovery is meant to
        # replace.  Thin bodies without any centre-inside cell are therefore
        # represented by their cut-cell band instead of disappearing.
        return bool(
            self.exclusion_distance > 0.0
            and self.index.nearest_point(centre)[1] <= self.exclusion_distance
        )

    def overlaps(self, x0: int, x1: int, y0: int, y1: int, z0: int, z1: int) -> bool:
        lo, hi = self._world_bounds(x0, x1, y0, y1, z0, z1)
        return self.index.box_intersects_surface(lo, hi)


def _validate_bounds(bounds: Bounds, name: str) -> None:
    if len(bounds) != 6 or not all(math.isfinite(float(value)) for value in bounds):
        raise ValueError(f"{name} must contain six finite coordinates")
    if not all(bounds[2 * axis] < bounds[2 * axis + 1] for axis in range(3)):
        raise ValueError(f"{name} must have positive extent along every axis: {bounds}")


def _dyadic_level(background: float, requested: float) -> int:
    if not math.isfinite(requested) or requested <= 0.0:
        raise ValueError(f"Requested cell size must be finite and positive, got {requested}")
    if requested >= background:
        return 0
    ratio = background / requested
    return max(0, int(math.ceil(math.log2(ratio) - 1.0e-12)))


def _fitted_background_size(
    domain: Bounds,
    requested: float,
    *,
    axes: tuple[int, ...] = (0, 1, 2),
) -> float:
    """Return the largest spacing no larger than ``requested`` that tiles selected axes."""
    if not axes or any(axis not in (0, 1, 2) for axis in axes):
        raise ValueError("axes must contain one or more of 0, 1, and 2")
    extents = [
        Fraction(str(domain[2 * axis + 1] - domain[2 * axis])).limit_denominator(1_000_000)
        for axis in axes
    ]
    denominator = 1
    for extent in extents:
        denominator = lcm(denominator, extent.denominator)
    common_numerator = 0
    for extent in extents:
        scaled = extent.numerator * (denominator // extent.denominator)
        common_numerator = gcd(common_numerator, scaled)
    common_extent = Fraction(common_numerator, denominator)
    requested_fraction = Fraction(str(requested)).limit_denominator(1_000_000)
    count = math.ceil(common_extent / requested_fraction)
    return float(common_extent / max(1, count))


@dataclass(frozen=True)
class _ResolvedCartesianLattice:
    """Geometry-first lattice resolution for a preserved axis-aligned body.

    The input body coordinates are immutable.  The finest Cartesian spacing is
    chosen so that every body plane is an exact lattice plane, and the outer
    domain is padded outward (never inward, never onto the body) until the
    resulting lattice tiles it.
    """

    requested_domain: Bounds
    effective_domain: Bounds
    padding_per_face: tuple[float, float, float, float, float, float]
    background_cell_size: float
    finest_cell_size: float
    max_level: int
    body_bounds: Bounds
    body_lattice_indices: tuple[int, int, int, int, int, int]


def _integer_quotient(value: Fraction) -> int | None:
    """Return the integer value of an exact rational, or ``None`` if not integral."""
    return value.numerator if value.denominator == 1 else None


def _as_fraction(value: float) -> Fraction:
    # No denominator limiting: the body coordinates are geometry authority and
    # must round-trip exactly.  ``str`` on a float yields the shortest decimal
    # that reproduces it, so the fraction is exact by construction.
    return Fraction(str(value))


def _resolve_preserved_lattice(
    requested_domain: Bounds,
    background: float,
    requested_level: int,
    body_bounds: Bounds,
    *,
    max_extra_levels: int = 12,
) -> _ResolvedCartesianLattice:
    """Resolve a Cartesian lattice that preserves the body exactly.

    The body is the authority: its six faces become exact lattice planes at
    the finest spacing ``h``.  ``h`` is never coarser than the requested finest
    spacing.  If the requested dyadic spacing already divides every body
    extent, it is kept unchanged.  Otherwise a body-determined common spacing
    ``h = D[a0]/n`` (with ``n`` a common multiple of the per-axis extent
    ratios) is chosen, and the background cell size is derived from it so the
    requested refinement levels are preserved.  The outer domain is then padded
    outward until the background lattice tiles it.

    Raises
    ------
    ValueError
        If no compatible isotropic spacing at or finer than the requested one
        can represent the body exactly (the body is never modified).
    """
    background_fraction = _as_fraction(background)
    h_requested = background_fraction / (2**requested_level)
    bounds_min = [_as_fraction(body_bounds[2 * axis]) for axis in range(3)]
    bounds_max = [_as_fraction(body_bounds[2 * axis + 1]) for axis in range(3)]
    extents = [bounds_max[axis] - bounds_min[axis] for axis in range(3)]

    if all(_integer_quotient(extent / h_requested) is not None for extent in extents):
        finest = h_requested
        level = requested_level
        background_final = background_fraction
    else:
        # Choose the coarsest common spacing h = D[a0]/n <= h_requested.  For a
        # single isotropic spacing to divide every axis, ``n`` must be a
        # common multiple of the denominators of every D[a]/D[a0] ratio.
        best: Fraction | None = None
        for reference in range(3):
            denominator_lcm = 1
            valid = True
            for axis in range(3):
                if axis == reference:
                    continue
                ratio = extents[axis] / extents[reference]
                if ratio.denominator > 1_000_000:
                    valid = False
                    break
                denominator_lcm = lcm(denominator_lcm, ratio.denominator)
            if not valid:
                continue
            min_n = math.ceil(extents[reference] / h_requested)
            n = int(math.ceil(min_n / denominator_lcm)) * denominator_lcm
            candidate = extents[reference] / n
            if candidate < h_requested / (2**max_extra_levels):
                continue
            if best is None or candidate > best:
                best = candidate
        if best is None:
            counts = ", ".join(f"{float(extent / h_requested):.9g}" for extent in extents)
            raise ValueError(
                "The body width cannot be represented exactly at any compatible "
                f"Cartesian spacing. Requested finest spacing {float(h_requested):.9g} m "
                f"gives non-integer cell counts per axis: [{counts}]. The body was NOT "
                "modified. Choose a spacing that divides the body extents, or use a "
                "body whose extents are commensurable."
            )
        finest = best
        level = requested_level
        background_final = finest * (2**level)

    base_width = 2**level
    effective: list[float] = []
    padding: list[float] = []
    indices: list[int] = []
    for axis in range(3):
        body_min = bounds_min[axis]
        body_max = bounds_max[axis]
        requested_min = _as_fraction(requested_domain[2 * axis])
        requested_max = _as_fraction(requested_domain[2 * axis + 1])

        # Anchor the lattice on the body: body_min maps to fine index zero.
        low = math.floor((requested_min - body_min) / finest)
        high = math.ceil((requested_max - body_min) / finest)
        remainder = low % base_width
        if remainder:
            low -= remainder
        count = high - low
        remainder = count % base_width
        if remainder:
            high += base_width - remainder

        effective_min = body_min + low * finest
        effective_max = body_min + high * finest
        effective.extend((float(effective_min), float(effective_max)))
        padding.extend((float(requested_min - effective_min), float(effective_max - requested_max)))

        cell_count = _integer_quotient((body_max - body_min) / finest)
        if cell_count is None:
            raise RuntimeError("Resolved lattice does not divide the body extent")
        indices.extend((-low, -low + cell_count))

    return _ResolvedCartesianLattice(
        requested_domain=cast(Bounds, tuple(float(value) for value in requested_domain)),
        effective_domain=cast(Bounds, tuple(effective)),
        padding_per_face=cast(Bounds, tuple(padding)),
        background_cell_size=float(background_final),
        finest_cell_size=float(finest),
        max_level=level,
        body_bounds=cast(Bounds, tuple(float(value) for value in body_bounds)),
        body_lattice_indices=cast(tuple[int, int, int, int, int, int], tuple(indices)),
    )


def _validate_wall_on_surface(
    mesh_data: dict, surface_bounds: Bounds, wall_patch_name: str
) -> None:
    """Raise unless the wall patch vertex bounds coincide with the STL bounds."""
    (wall,) = [patch for patch in mesh_data["boundary"] if patch["name"] == wall_patch_name]
    first = int(wall["start_face"])
    faces = np.asarray(mesh_data["faces"])[first : first + int(wall["n_faces"])]
    vertices = np.asarray(mesh_data["vertex_position"])[np.unique(faces)]
    lower = vertices.min(axis=0)
    upper = vertices.max(axis=0)
    expected_lower = np.asarray(surface_bounds[::2])
    expected_upper = np.asarray(surface_bounds[1::2])
    scale = float(np.abs(expected_upper).max())
    # STL stores coordinates as float32, so the triangle bounds and the double
    # lattice may differ by ~1e-7 even when the wall is exactly conformal.
    tolerance = 1.0e-6 * max(1.0, scale)
    if not np.allclose(lower, expected_lower, rtol=0.0, atol=tolerance) or not np.allclose(
        upper, expected_upper, rtol=0.0, atol=tolerance
    ):
        raise ValueError(
            "Wall patch does not lie exactly on the STL surface: wall bounds "
            f"{lower.tolist()} .. {upper.tolist()} vs STL "
            f"{expected_lower.tolist()} .. {expected_upper.tolist()}"
        )


# Outward-oriented quad faces of a hexahedron in this codebase's fixed
# corner order: 0..3 the z0 face (x0y0,x1y0,x1y1,x0y1), 4..7 the z1 face.
_HEX_FACES = (
    (0, 3, 2, 1),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (3, 7, 6, 2),
    (0, 4, 7, 3),
    (1, 2, 6, 5),
)
_HEX_FACE_ARRAY = np.asarray(_HEX_FACES, dtype=np.intp)


def _hex_volumes(corners: np.ndarray) -> np.ndarray:
    """Vectorised divergence-theorem volumes for one or more hexahedra."""
    values = np.asarray(corners, dtype=np.float64)
    single = values.ndim == 2
    if single:
        values = values[None, ...]
    cell_centres = values.mean(axis=1)
    face_points = values[:, _HEX_FACE_ARRAY]
    face_centres = face_points.mean(axis=2)
    offsets = face_points - face_centres[:, :, None, :]
    area_vectors = 0.5 * np.cross(offsets, np.roll(offsets, -1, axis=2)).sum(axis=2)
    volumes = (
        np.einsum(
            "nfi,nfi->n",
            area_vectors,
            face_centres - cell_centres[:, None, :],
        )
        / 3.0
    )
    return volumes[0] if single else volumes


def _hex_volume(corners: np.ndarray) -> float:
    """Divergence-theorem volume of one hexahedron, matching ``geometry.py``'s
    fan-triangulated face formula so a positive result here means the solver's
    own volume computation will also be positive."""
    return float(_hex_volumes(corners))


def _conform_wall_to_surface(
    mesh_data: dict,
    surface_index: SurfaceIndex,
    wall_patch_name: str,
    *,
    min_volume_ratio: float = 0.15,
    fixed_bounds: Bounds | None = None,
) -> None:
    """Snap wall-patch corner points onto the true curved surface.

    Every extracted wall-adjacent point is projected onto the nearest point of
    the triangulated surface. Centre-based solid classification places the
    initial Cartesian wall within half a cell on either side of the surface,
    so restricting projection to points already inside the solid would retain
    the outside half of the staircase. A snap is
    rejected, leaving the point on its original Cartesian lattice position,
    if it would drive any cell referencing that point to a non-positive
    volume or shrink it by more than ``min_volume_ratio`` of its original
    volume. The bounded interpolation is recorded as partial recovery, so a
    caller can distinguish it from a fully mapped wall.
    """
    points = mesh_data["vertex_position"]
    mesh_scale = max(float(np.ptp(points, axis=0).max()), 1.0)
    point_quantum = 1.0e-12 * mesh_scale
    cell_vertex_indices = mesh_data.get("cell_vertex_indices")
    if cell_vertex_indices is None:
        return
    (wall,) = [patch for patch in mesh_data["boundary"] if patch["name"] == wall_patch_name]
    faces = [np.asarray(face, dtype=np.int32) for face in mesh_data["faces"]]
    first, count = int(wall["start_face"]), int(wall["n_faces"])
    wall_point_ids = np.unique(np.concatenate(faces[first : first + count]))

    # cfMesh's meshSurfaceMapper::preMapVertices performs three shrinking
    # Laplace passes before the final nearest-surface projection.  The pass is
    # essential at Cartesian corners: mapping two orthogonal staircase edges
    # directly to a curved surface can reverse one of the incident faces.
    boundary_start = int(mesh_data["n_interior_faces"])
    point_boundary_faces: dict[int, list[tuple[int, int]]] = {
        int(point_id): [] for point_id in wall_point_ids
    }
    initial_face_distances: dict[tuple[int, int], float] = {}
    for face_id in range(boundary_start, len(faces)):
        face = faces[face_id]
        face_centre = points[face].mean(axis=0)
        for position, point_id_value in enumerate(face):
            point_id = int(point_id_value)
            if point_id not in point_boundary_faces:
                continue
            point_boundary_faces[point_id].append((face_id, position))
            offset = points[point_id] - face_centre
            initial_face_distances[(face_id, position)] = max(
                float(np.dot(offset, offset)), np.finfo(np.float64).tiny
            )

    pre_map_face_ids = sorted(
        {face_id for entries in point_boundary_faces.values() for face_id, _position in entries}
    )

    def nearest_many(values: np.ndarray) -> list[tuple[np.ndarray, float]]:
        if len(surface_index.triangles) < 500:
            return [surface_index.nearest_point(value) for value in values]
        nearest, distances, _triangle_ids = surface_index.nearest_points(values)
        return list(zip(nearest, map(float, distances), strict=True))

    for _iteration in range(3):
        face_centres = {
            face_id: points[faces[face_id]].mean(axis=0) for face_id in pre_map_face_ids
        }
        weighted_centres = np.empty((len(wall_point_ids), 3), dtype=np.float64)
        for local_index, point_id_value in enumerate(wall_point_ids):
            point_id = int(point_id_value)
            point = points[point_id]
            weighted_centre = np.zeros(3, dtype=np.float64)
            weight_sum = 0.0
            for face_id, position in point_boundary_faces[point_id]:
                face_centre = face_centres[face_id]
                offset = point - face_centre
                weight = max(
                    float(np.dot(offset, offset)) / initial_face_distances[(face_id, position)],
                    np.finfo(np.float64).tiny,
                )
                weighted_centre += weight * face_centre
                weight_sum += weight
            weighted_centres[local_index] = weighted_centre / weight_sum
        mapped_points = nearest_many(weighted_centres)
        updates: dict[int, np.ndarray] = {}
        for local_index, point_id_value in enumerate(wall_point_ids):
            point_id = int(point_id_value)
            point = points[point_id]
            mapped, _distance = mapped_points[local_index]
            candidate = point + 0.5 * (mapped - point)
            if fixed_bounds is not None:
                for side, value in enumerate(fixed_bounds):
                    axis = side // 2
                    if abs(float(point[axis]) - value) <= point_quantum:
                        candidate[axis] = value
            updates[point_id] = candidate
        for point_id, candidate in updates.items():
            points[point_id] = candidate

    point_face_ids: dict[int, list[int]] = {}
    for face_id, face in enumerate(faces):
        for point_id in face:
            point_face_ids.setdefault(int(point_id), []).append(face_id)

    point_cells: dict[int, list[int]] = {}
    for cell_id, corners in enumerate(cell_vertex_indices):
        for corner in corners:
            point_cells.setdefault(int(corner), []).append(cell_id)

    accepted = 0
    partial_accepted = 0
    rejected_nonpositive = 0
    rejected_volume_ratio = 0
    rejected_face_collapse = 0
    maximum_requested_displacement = 0.0
    maximum_rejected_displacement = 0.0
    rejection_examples: list[dict] = []
    final_targets = nearest_many(points[wall_point_ids].copy())
    for local_index, point_id in enumerate(wall_point_ids):
        affected = point_cells.get(int(point_id), [])
        if not affected:
            continue
        affected_array = np.asarray(affected, dtype=np.intp)
        before = _hex_volumes(points[cell_vertex_indices[affected_array]])
        if float(np.min(before)) <= 0.0:
            continue
        original = points[point_id].copy()
        target, _distance = final_targets[local_index]
        if fixed_bounds is not None:
            for side, value in enumerate(fixed_bounds):
                axis = side // 2
                if abs(float(original[axis]) - value) <= point_quantum:
                    target[axis] = value
        requested_displacement = float(np.linalg.norm(target - original))
        maximum_requested_displacement = max(maximum_requested_displacement, requested_displacement)

        def check_candidate(
            candidate: np.ndarray,
            point_id: int = int(point_id),
            affected: list[int] = affected,
            before: np.ndarray = before,
            affected_array: np.ndarray = affected_array,
        ) -> tuple[float, float, bool]:
            points[point_id] = candidate
            after = _hex_volumes(points[cell_vertex_indices[affected_array]])
            minimum_after = float(np.min(after))
            minimum_ratio = float(np.min(after / before))
            face_invalid = False
            for face_id in point_face_ids[int(point_id)]:
                coordinates = points[faces[face_id]]
                distinct = np.unique(np.rint(coordinates / point_quantum), axis=0)
                if len(distinct) < 3:
                    face_invalid = True
                    break
                centre = coordinates.mean(axis=0)
                area_vector = np.zeros(3, dtype=np.float64)
                for index in range(len(coordinates)):
                    area_vector += 0.5 * np.cross(
                        coordinates[index] - centre,
                        coordinates[(index + 1) % len(coordinates)] - centre,
                    )
                if np.linalg.norm(area_vector) <= np.finfo(np.float64).eps * 64.0:
                    face_invalid = True
                    break
            return minimum_after, minimum_ratio, face_invalid

        minimum_after, minimum_ratio, face_invalid = check_candidate(target)
        if minimum_after <= 0.0 or minimum_ratio < min_volume_ratio or face_invalid:
            low = 0.0
            high = 1.0
            for _ in range(32):
                fraction = 0.5 * (low + high)
                candidate = original + fraction * (target - original)
                trial_after, trial_ratio, trial_invalid = check_candidate(candidate)
                if trial_after > 0.0 and trial_ratio >= min_volume_ratio and not trial_invalid:
                    low = fraction
                else:
                    high = fraction
            if low > 1.0e-6:
                points[point_id] = original + low * (target - original)
                accepted += 1
                partial_accepted += 1
                continue
            points[point_id] = original
            if minimum_after <= 0.0:
                rejected_nonpositive += 1
                reason = "nonpositive_volume"
            elif minimum_ratio < min_volume_ratio:
                rejected_volume_ratio += 1
                reason = "volume_ratio"
            else:
                rejected_face_collapse += 1
                reason = "face_collapse"
            maximum_rejected_displacement = max(
                maximum_rejected_displacement, requested_displacement
            )
            if len(rejection_examples) < 16:
                rejection_examples.append(
                    {
                        "point_id": int(point_id),
                        "reason": reason,
                        "original": original.tolist(),
                        "target": target.tolist(),
                        "requested_displacement": requested_displacement,
                        "minimum_after_volume": float(minimum_after),
                        "minimum_volume_ratio": float(minimum_ratio),
                    }
                )
        else:
            accepted += 1

    mesh_data["mesh_generation"]["surface_projection"] = {
        "pre_map_iterations": 3,
        "attempted_points": int(len(wall_point_ids)),
        "accepted_points": accepted,
        "partial_accepted_points": partial_accepted,
        "rejected_nonpositive_volume": rejected_nonpositive,
        "rejected_volume_ratio": rejected_volume_ratio,
        "rejected_face_collapse": rejected_face_collapse,
        "maximum_requested_displacement": maximum_requested_displacement,
        "maximum_rejected_displacement": maximum_rejected_displacement,
        "rejection_examples": rejection_examples,
    }


def _prepare_wall_topology(mesh_data: dict, wall_patch_name: str) -> None:
    """Enforce cfMesh's one-surface-face-per-boundary-cell invariant.

    The Cartesian extractor can expose two or more edge-connected staircase
    faces of one cell to a curved solid.  cfMesh performs a surface-preparation
    stage before mapping; treating those faces independently makes their
    nearest-point projections overlap.  Replace each connected group by its
    oriented outline polygon, retaining the cell and all of its other faces.
    """
    patches = list(mesh_data["boundary"])
    matching = [patch for patch in patches if patch["name"] == wall_patch_name]
    if len(matching) != 1:
        raise ValueError(f"Expected exactly one wall patch named {wall_patch_name!r}")
    wall = matching[0]
    faces = [np.asarray(face, dtype=np.int32) for face in mesh_data["faces"]]
    owners = np.asarray(mesh_data["owners"], dtype=np.int32)
    n_internal = int(mesh_data["n_interior_faces"])
    wall_start = int(wall["start_face"])
    wall_stop = wall_start + int(wall["n_faces"])

    grouped: dict[int, list[np.ndarray]] = {}
    for face_id in range(wall_start, wall_stop):
        grouped.setdefault(int(owners[face_id]), []).append(faces[face_id])

    def merged_outline(group: list[np.ndarray], owner: int) -> np.ndarray:
        if len(group) == 1:
            return group[0].copy()
        directed: list[tuple[int, int, tuple[int, int]]] = []
        counts: dict[tuple[int, int], int] = {}
        for face in group:
            for first, second in zip(face, np.roll(face, -1), strict=True):
                a, b = int(first), int(second)
                edge = (a, b) if a < b else (b, a)
                counts[edge] = counts.get(edge, 0) + 1
                directed.append((a, b, edge))
        outline_edges = [(a, b) for a, b, edge in directed if counts[edge] == 1]
        if not outline_edges:
            raise ValueError(f"Surface faces of cell {owner} have no boundary outline")
        following: dict[int, int] = {}
        for first, second in outline_edges:
            if first in following:
                raise ValueError(f"Surface faces of cell {owner} do not form one mappable outline")
            following[first] = second
        outline = [outline_edges[0][0]]
        while len(outline) < len(outline_edges):
            next_point = following.get(outline[-1])
            if next_point is None or next_point in outline:
                raise ValueError(f"Surface faces of cell {owner} are not one edge-connected group")
            outline.append(next_point)
        if following.get(outline[-1]) != outline[0]:
            raise ValueError(f"Surface outline of cell {owner} is open")
        return np.asarray(outline, dtype=np.int32)

    merged_faces = [merged_outline(group, owner) for owner, group in grouped.items()]
    merged_owners = list(grouped)
    rebuilt_faces = faces[:n_internal]
    rebuilt_owners = list(map(int, owners[:n_internal]))
    rebuilt_boundary: list[dict] = []
    start_face = n_internal
    for patch in patches:
        name = str(patch["name"])
        if name == wall_patch_name:
            patch_faces = merged_faces
            patch_owners = merged_owners
        else:
            patch_start = int(patch["start_face"])
            patch_stop = patch_start + int(patch["n_faces"])
            patch_faces = faces[patch_start:patch_stop]
            patch_owners = list(map(int, owners[patch_start:patch_stop]))
        rebuilt_faces.extend(patch_faces)
        rebuilt_owners.extend(patch_owners)
        rebuilt_boundary.append(
            {
                **patch,
                "start_face": start_face,
                "n_faces": len(patch_faces),
            }
        )
        start_face += len(patch_faces)

    widths = {len(face) for face in rebuilt_faces}
    mesh_data["faces"] = (
        np.ascontiguousarray(rebuilt_faces, dtype=np.int32) if len(widths) == 1 else rebuilt_faces
    )
    mesh_data["owners"] = np.ascontiguousarray(rebuilt_owners, dtype=np.int32)
    mesh_data["boundary"] = rebuilt_boundary
    mesh_data["n_faces"] = len(rebuilt_faces)
    mesh_data.pop("cell_face_indices", None)
    mesh_data.pop("cell_face_offset", None)
    mesh_data.setdefault("mesh_generation", {})["surface_preparation"] = {
        "original_faces": wall_stop - wall_start,
        "prepared_faces": len(merged_faces),
        "merged_faces": (wall_stop - wall_start) - len(merged_faces),
    }


def _remove_non_mappable_surface_cells(
    mesh_data: dict, cell_ids: np.ndarray, wall_patch_name: str
) -> None:
    """Remove tangled surface cells and expose their fluid neighbours.

    This is the serial native equivalent of cfMesh's
    ``checkNonMappableCellConnections::removeCells``.  Faces between retained
    and removed cells become wall faces; faces whose two cells are removed
    disappear.  The remaining cell numbering and per-cell metadata stay
    contiguous.
    """
    n_cells = int(mesh_data["n_cells"])
    remove = np.zeros(n_cells, dtype=bool)
    remove[np.asarray(cell_ids, dtype=np.int64)] = True
    if not np.any(remove):
        return
    keep = ~remove
    cell_map = np.full(n_cells, -1, dtype=np.int32)
    cell_map[keep] = np.arange(np.count_nonzero(keep), dtype=np.int32)
    faces = [np.asarray(face, dtype=np.int32) for face in mesh_data["faces"]]
    owners = np.asarray(mesh_data["owners"], dtype=np.int32)
    neighbours = np.asarray(mesh_data["neighbours"], dtype=np.int32)
    n_internal = int(mesh_data["n_interior_faces"])

    internal_faces: list[np.ndarray] = []
    internal_owners: list[int] = []
    internal_neighbours: list[int] = []
    exposed_faces: list[np.ndarray] = []
    exposed_owners: list[int] = []
    for face_id in range(n_internal):
        owner = int(owners[face_id])
        neighbour = int(neighbours[face_id])
        owner_kept = bool(keep[owner])
        neighbour_kept = bool(keep[neighbour])
        if owner_kept and neighbour_kept:
            internal_faces.append(faces[face_id].copy())
            internal_owners.append(int(cell_map[owner]))
            internal_neighbours.append(int(cell_map[neighbour]))
        elif owner_kept:
            exposed_faces.append(faces[face_id].copy())
            exposed_owners.append(int(cell_map[owner]))
        elif neighbour_kept:
            exposed_faces.append(faces[face_id][::-1].copy())
            exposed_owners.append(int(cell_map[neighbour]))

    boundary_faces: dict[str, list[np.ndarray]] = {}
    boundary_owners: dict[str, list[int]] = {}
    patch_order: list[str] = []
    patch_records: dict[str, dict] = {}
    for patch in mesh_data["boundary"]:
        name = str(patch["name"])
        patch_order.append(name)
        patch_records[name] = dict(patch)
        boundary_faces[name] = []
        boundary_owners[name] = []
        start = int(patch["start_face"])
        stop = start + int(patch["n_faces"])
        for face_id in range(start, stop):
            owner = int(owners[face_id])
            if keep[owner]:
                boundary_faces[name].append(faces[face_id].copy())
                boundary_owners[name].append(int(cell_map[owner]))
    if wall_patch_name not in boundary_faces:
        raise ValueError(f"Cannot expose removed cells onto missing patch {wall_patch_name!r}")
    boundary_faces[wall_patch_name].extend(exposed_faces)
    boundary_owners[wall_patch_name].extend(exposed_owners)

    rebuilt_faces = internal_faces.copy()
    rebuilt_owners = internal_owners.copy()
    rebuilt_boundary: list[dict] = []
    start_face = len(internal_faces)
    for name in patch_order:
        rebuilt_faces.extend(boundary_faces[name])
        rebuilt_owners.extend(boundary_owners[name])
        rebuilt_boundary.append(
            {
                **patch_records[name],
                "start_face": start_face,
                "n_faces": len(boundary_faces[name]),
            }
        )
        start_face += len(boundary_faces[name])

    widths = {len(face) for face in rebuilt_faces}
    mesh_data["faces"] = (
        np.ascontiguousarray(rebuilt_faces, dtype=np.int32) if len(widths) == 1 else rebuilt_faces
    )
    mesh_data["owners"] = np.ascontiguousarray(rebuilt_owners, dtype=np.int32)
    mesh_data["neighbours"] = np.ascontiguousarray(internal_neighbours, dtype=np.int32)
    mesh_data["boundary"] = rebuilt_boundary
    mesh_data["n_cells"] = int(np.count_nonzero(keep))
    mesh_data["n_faces"] = len(rebuilt_faces)
    mesh_data["n_interior_faces"] = len(internal_faces)
    for name in ("cell_levels", "cell_sizes", "cell_type_code", "cell_vertex_indices"):
        if name in mesh_data:
            mesh_data[name] = np.ascontiguousarray(np.asarray(mesh_data[name])[keep])
    mesh_data.pop("cell_face_indices", None)
    mesh_data.pop("cell_face_offset", None)
    history = mesh_data.setdefault("mesh_generation", {}).setdefault(
        "non_mappable_cell_removal", []
    )
    history.append(
        {
            "removed_cells": int(np.count_nonzero(remove)),
            "exposed_wall_faces": len(exposed_faces),
        }
    )


def _compact_conformed_topology(mesh_data: dict, *, tolerance: float = 1.0e-12) -> None:
    """Merge coincident projected points and turn collapsed quads into polygons.

    Mapping a Cartesian wall to a curved feature can make two corners of one
    wall quad coincide (notably where a cylindrical side meets an exactly
    aligned cap). The physical face is then a valid triangle, but retaining a
    four-node row gives a zero area vector or a self-touching polygon. Merge
    geometrically coincident point ids, remove repeated nodes from each face,
    and switch the conformed mesh to the solver's native variable-face
    polyhedron representation.
    """
    points = np.asarray(mesh_data["vertex_position"], dtype=np.float64)
    scale = max(float(np.ptp(points, axis=0).max()), 1.0)
    quantum = max(tolerance * scale, np.finfo(np.float64).eps * scale * 32.0)
    quantized = np.rint(points / quantum).astype(np.int64)
    _, first, inverse = np.unique(quantized, axis=0, return_index=True, return_inverse=True)
    unique_points = points[first]

    cell_corners = np.asarray(mesh_data["cell_vertex_indices"], dtype=np.int64)
    remapped_cell_corners = inverse[cell_corners]
    provisional_cell_centres = unique_points[remapped_cell_corners].mean(axis=1)
    n_internal = int(mesh_data["n_interior_faces"])
    owners = np.asarray(mesh_data["owners"], dtype=np.int64)
    neighbours = np.asarray(mesh_data["neighbours"], dtype=np.int64)

    compact_faces: list[np.ndarray] = []
    faces_by_width: dict[int, list[int]] = {}
    for face_id, face in enumerate(mesh_data["faces"]):
        remapped = inverse[np.asarray(face, dtype=np.int64)]
        compact = list(dict.fromkeys(map(int, remapped)))
        if len(compact) < 3:
            raise ValueError(
                f"Curved-surface projection collapsed face {face_id} to {len(compact)} "
                f"unique nodes: original={np.asarray(face).tolist()}, "
                f"remapped={remapped.tolist()}"
            )
        compact_faces.append(np.asarray(compact, dtype=np.int32))
        faces_by_width.setdefault(len(compact), []).append(face_id)

    # Winding checks dominate this pass on production meshes.  Evaluate equal-
    # width polygons in bounded vectorised batches instead of issuing four
    # ``np.cross`` calls per face from Python.
    for face_ids in faces_by_width.values():
        for chunk_start in range(0, len(face_ids), 100_000):
            chunk = np.asarray(face_ids[chunk_start : chunk_start + 100_000], dtype=np.intp)
            connectivity = np.stack([compact_faces[int(face_id)] for face_id in chunk])
            coordinates = unique_points[connectivity]
            face_centres = coordinates.mean(axis=1)
            offsets = coordinates - face_centres[:, None, :]
            area_vectors = 0.5 * np.cross(offsets, np.roll(offsets, -1, axis=1)).sum(axis=1)
            directions = np.empty_like(area_vectors)
            internal_mask = chunk < n_internal
            internal_faces = chunk[internal_mask]
            directions[internal_mask] = (
                provisional_cell_centres[neighbours[internal_faces]]
                - provisional_cell_centres[owners[internal_faces]]
            )
            boundary_faces = chunk[~internal_mask]
            directions[~internal_mask] = (
                face_centres[~internal_mask] - provisional_cell_centres[owners[boundary_faces]]
            )
            reverse = chunk[np.einsum("ij,ij->i", area_vectors, directions) < 0.0]
            for face_id in reverse:
                compact_faces[int(face_id)] = compact_faces[int(face_id)][::-1].copy()

    widths = {len(face) for face in compact_faces}
    mesh_data["faces"] = (
        np.ascontiguousarray(compact_faces, dtype=np.int32) if len(widths) == 1 else compact_faces
    )
    mesh_data["vertex_position"] = np.ascontiguousarray(unique_points)
    mesh_data["n_points"] = len(unique_points)
    # Projected boundary cells are polyhedra, not axis-aligned VTK hexes.
    mesh_data.pop("cell_vertex_indices", None)
    mesh_data.pop("cell_type_code", None)


class CartesianOctree:
    """Build an adaptive, body-fitted octree mesh directly in memory.

    Parameters
    ----------
    domain:
        Axis-aligned fluid-domain bounds ``(xmin, xmax, ymin, ymax, zmin,
        zmax)``.
    max_cell_size:
        Background cell size.  Every domain extent must be an integer multiple
        of this value, which keeps the outer coupling lattice exact.
    surface_file:
        Location of the watertight STL surface that defines the solid.
    wall_patch_name:
        Boundary-patch name assigned to faces extracted from ``surface_file``.
        It must be supplied together with the surface location.
    surface_cell_size:
        Requested size in the first fluid-cell layer around the STL surface.
        Refinement coarsens outwards in automatically generated 2:1 bands.
        refinements:
            Additional typed volume-refinement regions.
    merge_outer_patch:
        Merge the six outer sides into one patch (the coupled FVM--VPM case),
        or leave the conventional inlet/outlet/ymin/ymax/zmin/zmax patches.
    preserve_outer_patches:
        Outer sides excluded from ``merge_outer_patch``; useful for cyclic,
        slip, or empty spanwise boundaries in a coupled case.
        surface_may_cross_domain_boundary:
            Permit a closed general surface to extend outside the mesh domain.
            Outer patches retain precedence where the domain clips the solid.
    preserve_body_geometry:
        Guarantee that the input body coordinates are immutable.  The body
        faces become exact Cartesian lattice planes, the outer domain is padded
        outward as needed, and no ordinary fluid cell may overlap the solid
        with positive volume.  This is the only supported mode; the legacy
        body-snapping path was removed.

    Notes
    -----
    Coarse cells next to finer cells are represented as valid polyhedra with
        multiple coplanar subfaces. Geometrically the untouched cells remain
        axis-aligned hexahedra. This
    is the same useful finite-volume topology as the nine-face transition
    body-fitted polyhedra, without an external intermediary.

    ``effective_domain`` exposes the resolved outer domain; it equals
    ``requested_domain`` when no outer padding was required.  Any coupling
    region or sampler derived from the outer boundary must use the effective
    domain.
    """

    def __init__(
        self,
        domain: Bounds,
        max_cell_size: float,
        *,
        surface_file: str | Path | None = None,
        surface_data: TriangulatedSurface | None = None,
        exact_surface_components: tuple[Bounds, ...] = (),
        surface_exclusion_distance: float = 0.0,
        wall_patch_name: str | None = None,
        surface_cell_size: float | None = None,
        refinements: tuple[BoxRefinement, ...] = (),
        merge_outer_patch: str | None = None,
        preserve_outer_patches: tuple[str, ...] = (),
        surface_may_cross_domain_boundary: bool = False,
        include_cell_vertex_indices: bool = True,
        preserve_body_geometry: bool = True,
    ) -> None:
        _validate_bounds(domain, "domain")
        if not math.isfinite(max_cell_size) or max_cell_size <= 0.0:
            raise ValueError("max_cell_size must be finite and positive")
        if surface_file is not None and surface_data is not None:
            raise ValueError("surface_file and surface_data are mutually exclusive")
        if exact_surface_components and surface_data is None:
            raise ValueError("exact_surface_components requires surface_data")
        if not math.isfinite(surface_exclusion_distance) or surface_exclusion_distance < 0.0:
            raise ValueError("surface_exclusion_distance must be finite and non-negative")
        if (surface_file is None and surface_data is None) != (wall_patch_name is None):
            raise ValueError("surface_file and wall_patch_name must be supplied together")
        if surface_file is None and surface_data is None and surface_cell_size is not None:
            raise ValueError("surface_cell_size requires surface_file")
        if wall_patch_name is not None and not wall_patch_name.strip():
            raise ValueError("wall_patch_name must not be empty")

        requested_domain: Bounds = cast(Bounds, tuple(float(value) for value in domain))
        surface = (
            surface_data
            if surface_data is not None
            else TriangulatedSurface.from_stl(surface_file)
            if surface_file is not None
            else None
        )
        if surface is not None and not surface_may_cross_domain_boundary:
            surface_bounds = surface.bounds
            if not all(
                requested_domain[2 * axis]
                < surface_bounds[2 * axis]
                < surface_bounds[2 * axis + 1]
                < requested_domain[2 * axis + 1]
                for axis in range(3)
            ):
                raise ValueError("STL surface must lie strictly inside domain")
        elif surface is not None:
            if surface.kind != "general":
                raise ValueError(
                    "surface_may_cross_domain_boundary requires a general curved surface"
                )
            if not all(
                requested_domain[2 * axis] < surface.bounds[2 * axis + 1]
                and surface.bounds[2 * axis] < requested_domain[2 * axis + 1]
                for axis in range(3)
            ):
                raise ValueError("STL surface does not overlap the requested domain")
        if wall_patch_name is not None and merge_outer_patch == wall_patch_name:
            raise ValueError("Outer and wall patch names must differ")
        unknown_preserved = sorted(set(preserve_outer_patches) - set(OUTER_PATCH_NAMES))
        if unknown_preserved:
            raise ValueError(f"Unknown preserved outer patches: {unknown_preserved}")
        if preserve_outer_patches and merge_outer_patch is None:
            raise ValueError("preserve_outer_patches requires merge_outer_patch")
        for refinement in refinements:
            _validate_bounds(refinement.bounds, f"{refinement.name}.bounds")
            if not all(
                requested_domain[2 * axis]
                <= refinement.bounds[2 * axis]
                < refinement.bounds[2 * axis + 1]
                <= requested_domain[2 * axis + 1]
                for axis in range(3)
            ):
                raise ValueError(f"{refinement.name} must lie inside domain")

        fitted_max_cell_size = _fitted_background_size(
            requested_domain,
            max_cell_size,
        )
        requested_sizes = [refinement.cell_size for refinement in refinements]
        if surface_cell_size is not None:
            requested_sizes.append(surface_cell_size)
        requested_level = max(
            (_dyadic_level(fitted_max_cell_size, size) for size in requested_sizes),
            default=0,
        )

        if surface is not None:
            if not preserve_body_geometry:
                raise ValueError(
                    "preserve_body_geometry=False is no longer supported: the legacy "
                    "body-snapping path was removed and the input body is always "
                    "preserved exactly."
                )
            if surface.kind == "box" and surface_exclusion_distance <= 0.0:
                resolved = _resolve_preserved_lattice(
                    requested_domain, fitted_max_cell_size, requested_level, surface.bounds
                )
                self._resolved_lattice = resolved
                self.domain: Bounds = resolved.effective_domain
                self._surface_index = None
            else:
                # A general (curved) surface imposes no lattice-plane-alignment
                # constraint: the background/finest spacing is fitted exactly as
                # in the no-surface case, and the boundary is conformed to the
                # true triangulated surface after the Cartesian lattice is built
                # (see ``_conform_wall_to_surface``), not by choosing spacing.
                resolved = None
                self._resolved_lattice = None
                self.domain = requested_domain
                self._surface_index = SurfaceIndex.build(surface.triangles)
        else:
            self._resolved_lattice = None
            resolved = None
            self.domain = requested_domain
            self._surface_index = None

        self._requested_domain = requested_domain
        self._effective_domain = (
            resolved.effective_domain if resolved is not None else requested_domain
        )
        self.padding_per_face = (
            resolved.padding_per_face if resolved is not None else (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        )
        self.requested_max_cell_size = float(max_cell_size)
        self.max_cell_size = (
            resolved.background_cell_size if resolved is not None else fitted_max_cell_size
        )
        self._resolved_max_level = resolved.max_level if resolved is not None else requested_level
        self._resolved_h_min = (
            resolved.finest_cell_size
            if resolved is not None
            else fitted_max_cell_size / (2**requested_level)
        )
        self._body_lattice_indices = resolved.body_lattice_indices if resolved is not None else None
        self.surface = surface
        self.surface_file = str(surface.path) if surface is not None else None
        self.surface_bounds = surface.bounds if surface is not None else None
        self.surface_cell_size = float(surface_cell_size) if surface_cell_size is not None else None
        self.refinements = tuple(refinements)
        self.wall_patch_name = wall_patch_name or ""
        self.merge_outer_patch = merge_outer_patch
        self.preserve_outer_patches = tuple(dict.fromkeys(preserve_outer_patches))
        self.surface_may_cross_domain_boundary = bool(surface_may_cross_domain_boundary)
        self.include_cell_vertex_indices = include_cell_vertex_indices
        self.preserve_body_geometry = preserve_body_geometry
        self.exact_surface_components = tuple(exact_surface_components)
        self.surface_exclusion_distance = float(surface_exclusion_distance)

    def effective_cell_size(self, requested: float) -> float:
        """Return the dyadic size generated for a requested size."""
        return self.max_cell_size / (2 ** _dyadic_level(self.max_cell_size, requested))

    @property
    def effective_domain(self) -> Bounds:
        """Resolved outer domain; equals ``requested_domain`` when no padding was needed."""
        return self._effective_domain

    @property
    def requested_domain(self) -> Bounds:
        """The outer domain requested by the caller (before body-driven padding)."""
        return self._requested_domain

    @property
    def padding(self) -> tuple[float, float, float, float, float, float]:
        """Outward padding per outer face ``(xmin, xmax, ymin, ymax, zmin, zmax)``."""
        return self.padding_per_face

    def _base_counts(self) -> tuple[int, int, int]:
        counts = []
        for axis in range(3):
            extent = self.domain[2 * axis + 1] - self.domain[2 * axis]
            cell_size = self.max_cell_size
            count = int(round(extent / cell_size))
            if count < 1 or not math.isclose(
                count * cell_size,
                extent,
                rel_tol=1.0e-10,
                abs_tol=1.0e-12,
            ):
                raise RuntimeError("Fitted Cartesian background size does not tile the domain")
            counts.append(count)
        return tuple(counts)  # type: ignore[return-value]

    def _integer_box(self, bounds: Bounds, h_min: float) -> _IntegerBox:
        indices = []
        for axis in range(3):
            origin = self.domain[2 * axis]
            for side in range(2):
                value = (bounds[2 * axis + side] - origin) / h_min
                rounded = int(round(value))
                if math.isclose(value, rounded, rel_tol=1.0e-9, abs_tol=1.0e-8):
                    index = rounded
                elif side == 0:
                    index = math.floor(value)
                else:
                    index = math.ceil(value)
                indices.append(index)
        return _IntegerBox(*indices)

    def _refinement_regions(
        self,
        h_min: float,
        max_level: int,
        limits: tuple[int, int, int],
        body_region: _SolidRegion | None,
    ) -> tuple[tuple[_RefinementRegion, int], ...]:
        regions: list[tuple[_RefinementRegion, int]] = []

        def add_balanced(box: _RefinementRegion, target_level: int, first_padding: int = 0) -> None:
            current = box.expanded(first_padding, limits) if first_padding else box
            regions.append((current, target_level))
            for level in range(target_level - 1, 0, -1):
                # One cell at the next-coarser level gives an explicit 2:1
                # buffer before the following level is allowed.
                current = current.expanded(2 ** (max_level - level), limits)
                regions.append((current, level))

        if body_region is not None:
            # The body needs a one-fine-cell shell around it so the leaf grid
            # is fully refined wherever the true surface can be: for a
            # preserved box, ``body_region`` is the exact body; for a general
            # curved surface, it is the body's AABB, which over-refines a
            # little near its corners but never under-refines near the body.
            add_balanced(body_region, max_level, first_padding=1)

        for refinement in self.refinements:
            level = _dyadic_level(self.max_cell_size, refinement.cell_size)
            if level:
                if hasattr(refinement, "intersects_box"):
                    region = _WorldRefinementRegion(
                        refinement,
                        h_min,
                        (self.domain[0], self.domain[2], self.domain[4]),
                    )
                else:
                    region = self._integer_box(refinement.bounds, h_min)
                add_balanced(region, level)

        # Higher levels take precedence in overlap regions.
        regions.sort(key=lambda item: item[1], reverse=True)
        return tuple(regions)

    def _build_leaves(
        self,
        base_counts: tuple[int, int, int],
        max_level: int,
        solid: _SolidRegion | _SurfaceSolid | None,
        regions: tuple[tuple[_RefinementRegion, int], ...],
    ) -> np.ndarray:
        base_width = 2**max_level
        # A Python tuple/list representation costs several times the 20 bytes
        # of actual octree data per cell.  Grow one packed numeric buffer so a
        # multi-million-cell wake mesh does not spend gigabytes on objects
        # before topology construction even begins.
        capacity = max(1024, math.prod(base_counts))
        leaves = np.empty((capacity, 5), dtype=np.int32)
        n_leaves = 0

        def append_leaf(x0: int, y0: int, z0: int, width: int, level: int) -> None:
            nonlocal capacity, leaves, n_leaves
            if n_leaves == capacity:
                new_capacity = int(capacity * 1.5) + 1
                grown = np.empty((new_capacity, 5), dtype=np.int32)
                grown[:n_leaves] = leaves[:n_leaves]
                leaves = grown
                capacity = new_capacity
            leaves[n_leaves] = (x0, y0, z0, width, level)
            n_leaves += 1

        def visit(x0: int, y0: int, z0: int, width: int, level: int) -> None:
            x1, y1, z1 = x0 + width, y0 + width, z0 + width
            if solid is not None and solid.contains(x0, x1, y0, y1, z0, z1):
                return

            target = level
            for region, region_level in regions:
                if region_level <= target:
                    break
                if region.overlaps(x0, x1, y0, y1, z0, z1):
                    target = region_level
                    break

            if level < target:
                child = width // 2
                for dz in (0, child):
                    for dy in (0, child):
                        for dx in (0, child):
                            visit(x0 + dx, y0 + dy, z0 + dz, child, level + 1)
                return

            if (
                solid is not None
                and isinstance(solid, _IntegerBox | _CompositeIntegerBox)
                and solid.overlaps(x0, x1, y0, y1, z0, z1)
            ):
                # Exact interval/AABB classification: a leaf may touch the body
                # with zero volume (wall-adjacent fluid) but must never overlap
                # it with positive volume.  Wholly-inside cells are removed
                # above by ``solid.contains``.  Any remaining positive-volume
                # overlap means a body plane is not an exact leaf lattice
                # plane, which preserved mode must reject rather than silently
                # keep or remove on a cell-centre test.
                components = (
                    solid.components if isinstance(solid, _CompositeIntegerBox) else (solid,)
                )
                if any(
                    min(x1, component.x1) > max(x0, component.x0)
                    and min(y1, component.y1) > max(y0, component.y0)
                    and min(z1, component.z1) > max(z0, component.z0)
                    for component in components
                ):
                    raise ValueError(
                        "STL surface cuts a leaf cell with positive volume; the body "
                        "was not modified. The preserved-body contract requires every "
                        "body face to be an exact Cartesian lattice plane, so a "
                        "surface_cell_size (or a body position) that aligns the surface "
                        "with the finest level must be used"
                    )
            # A general curved-surface leaf flagged boundary by ``solid.overlaps``
            # above (the ``_SurfaceSolid`` branch) is kept and conformed to the
            # true surface afterwards by ``_conform_wall_to_surface``.
            append_leaf(x0, y0, z0, width, level)

        nx, ny, nz = base_counts
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    visit(i * base_width, j * base_width, k * base_width, base_width, 0)

        if not n_leaves:
            raise ValueError("Meshing configuration removed every fluid cell")
        return leaves[:n_leaves].copy()

    @staticmethod
    def _face_codes(
        axis: int,
        coordinate: int,
        a0: int,
        a1: int,
        b0: int,
        b1: int,
        strides: tuple[int, int],
        positive: bool,
    ) -> tuple[int, int, int, int]:
        """Encode one axis-aligned face ring with outward orientation."""
        sx, sy = strides

        def code(x: int, y: int, z: int) -> int:
            return x + sx * (y + sy * z)

        if axis == 0:
            ring = (
                code(coordinate, a0, b0),
                code(coordinate, a1, b0),
                code(coordinate, a1, b1),
                code(coordinate, a0, b1),
            )
        elif axis == 1:
            ring = (
                code(a0, coordinate, b0),
                code(a0, coordinate, b1),
                code(a1, coordinate, b1),
                code(a1, coordinate, b0),
            )
        else:
            ring = (
                code(a0, b0, coordinate),
                code(a1, b0, coordinate),
                code(a1, b1, coordinate),
                code(a0, b1, coordinate),
            )
        return ring if positive else (ring[3], ring[2], ring[1], ring[0])

    def _extract_topology(
        self,
        leaves: np.ndarray,
        max_level: int,
        limits: tuple[int, int, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], np.ndarray]:
        n_cells = len(leaves)
        nx, ny, nz = limits
        max_point_code = (nx + 1) * (ny + 1) * (nz + 1) - 1
        code_dtype = np.int32 if max_point_code <= np.iinfo(np.int32).max else np.int64
        point_strides = (nx + 1, ny + 1)
        cell_stride_x, cell_stride_y = nx, ny

        level_maps: list[dict[int, int]] = [{} for _ in range(max_level + 1)]
        for cell_id, (x0, y0, z0, _width, level) in enumerate(leaves):
            key = int(x0) + cell_stride_x * (int(y0) + cell_stride_y * int(z0))
            level_maps[int(level)][key] = cell_id

        levels = np.asarray(leaves[:, 4], dtype=np.int8)

        def find_cell(x: int, y: int, z: int) -> int:
            if x < 0 or y < 0 or z < 0 or x >= nx or y >= ny or z >= nz:
                return -1
            for level in range(max_level, -1, -1):
                width = 2 ** (max_level - level)
                ox, oy, oz = (x // width) * width, (y // width) * width, (z // width) * width
                key = ox + cell_stride_x * (oy + cell_stride_y * oz)
                found = level_maps[level].get(key)
                if found is not None:
                    return found
            return -1

        # A conforming Cartesian volume has about three internal faces per
        # cell.  The allowance covers 2:1 transition subfaces without building
        # millions of Python tuples.
        capacity = max(16, int(math.ceil(3.35 * n_cells)))
        interior_codes = np.empty((capacity, 4), dtype=code_dtype)
        interior_owners = np.empty(capacity, dtype=np.int32)
        interior_neighbours = np.empty(capacity, dtype=np.int32)
        n_interior = 0

        boundary_names = (
            (self.merge_outer_patch, *self.preserve_outer_patches)
            if self.merge_outer_patch
            else OUTER_PATCH_NAMES
        )
        patch_codes: dict[str, list[tuple[int, int, int, int]]] = {
            name: [] for name in boundary_names
        }
        patch_owners: dict[str, list[int]] = {name: [] for name in boundary_names}
        if self.surface is not None:
            patch_codes[self.wall_patch_name] = []
            patch_owners[self.wall_patch_name] = []

        def grow() -> None:
            nonlocal capacity, interior_codes, interior_owners, interior_neighbours
            new_capacity = int(capacity * 1.35) + 1
            new_codes = np.empty((new_capacity, 4), dtype=code_dtype)
            new_codes[:capacity] = interior_codes
            interior_codes = new_codes
            new_owners = np.empty(new_capacity, dtype=np.int32)
            new_owners[:capacity] = interior_owners
            interior_owners = new_owners
            new_neighbours = np.empty(new_capacity, dtype=np.int32)
            new_neighbours[:capacity] = interior_neighbours
            interior_neighbours = new_neighbours
            capacity = new_capacity

        def emit_interior(
            owner: int,
            neighbour: int,
            axis: int,
            coordinate: int,
            a0: int,
            a1: int,
            b0: int,
            b1: int,
        ) -> None:
            nonlocal n_interior
            if n_interior == capacity:
                grow()
            interior_codes[n_interior] = self._face_codes(
                axis, coordinate, a0, a1, b0, b1, point_strides, True
            )
            interior_owners[n_interior] = owner
            interior_neighbours[n_interior] = neighbour
            n_interior += 1

        def patch_for(axis: int, positive: bool, coordinate: int) -> str:
            limit = limits[axis] if positive else 0
            if coordinate == limit:
                outer_name = OUTER_PATCH_NAMES[2 * axis + int(positive)]
                if outer_name in self.preserve_outer_patches:
                    return outer_name
                if self.merge_outer_patch:
                    return self.merge_outer_patch
                return outer_name
            if self.surface is None:
                raise RuntimeError("Adaptive mesh contains an unexplained internal boundary")
            return self.wall_patch_name

        def samples(lo: int, width: int) -> tuple[int, ...]:
            if width == 1:
                return (lo,)
            return (lo + width // 4, lo + (3 * width) // 4)

        for owner, (x0v, y0v, z0v, widthv, levelv) in enumerate(leaves):
            x0, y0, z0, width, level = map(int, (x0v, y0v, z0v, widthv, levelv))
            x1, y1, z1 = x0 + width, y0 + width, z0 + width
            origins = (x0, y0, z0)
            ends = (x1, y1, z1)
            tangential = ((y0, y1, z0, z1), (x0, x1, z0, z1), (x0, x1, y0, y1))

            for axis in range(3):
                a0, a1, b0, b1 = tangential[axis]
                query = [x0, y0, z0]
                query[axis] = ends[axis]
                neighbour_ids: set[int] = set()
                for a in samples(a0, width):
                    for b in samples(b0, width):
                        if axis == 0:
                            candidate = find_cell(query[0], a, b)
                        elif axis == 1:
                            candidate = find_cell(a, query[1], b)
                        else:
                            candidate = find_cell(a, b, query[2])
                        if candidate >= 0:
                            neighbour_ids.add(candidate)

                if neighbour_ids:
                    for neighbour in sorted(neighbour_ids):
                        nx0, ny0, nz0, nw, neighbour_level = map(int, leaves[neighbour])
                        if abs(level - neighbour_level) > 1:
                            raise RuntimeError(
                                "Refinement transition is not 2:1 balanced between cells "
                                f"{owner} (level {level}) and {neighbour} "
                                f"(level {neighbour_level})"
                            )
                        if axis == 0:
                            emit_interior(
                                owner,
                                neighbour,
                                axis,
                                x1,
                                max(y0, ny0),
                                min(y1, ny0 + nw),
                                max(z0, nz0),
                                min(z1, nz0 + nw),
                            )
                        elif axis == 1:
                            emit_interior(
                                owner,
                                neighbour,
                                axis,
                                y1,
                                max(x0, nx0),
                                min(x1, nx0 + nw),
                                max(z0, nz0),
                                min(z1, nz0 + nw),
                            )
                        else:
                            emit_interior(
                                owner,
                                neighbour,
                                axis,
                                z1,
                                max(x0, nx0),
                                min(x1, nx0 + nw),
                                max(y0, ny0),
                                min(y1, ny0 + nw),
                            )
                else:
                    name = patch_for(axis, True, ends[axis])
                    patch_codes[name].append(
                        self._face_codes(axis, ends[axis], a0, a1, b0, b1, point_strides, True)
                    )
                    patch_owners[name].append(owner)

                # Negative faces are needed only at a physical boundary;
                # interior faces are emitted once by the cell on the low side.
                query[axis] = origins[axis] - 1
                negative_neighbours: set[int] = set()
                for a in samples(a0, width):
                    for b in samples(b0, width):
                        if axis == 0:
                            candidate = find_cell(query[0], a, b)
                        elif axis == 1:
                            candidate = find_cell(a, query[1], b)
                        else:
                            candidate = find_cell(a, b, query[2])
                        if candidate >= 0:
                            negative_neighbours.add(candidate)
                if not negative_neighbours:
                    name = patch_for(axis, False, origins[axis])
                    patch_codes[name].append(
                        self._face_codes(axis, origins[axis], a0, a1, b0, b1, point_strides, False)
                    )
                    patch_owners[name].append(owner)

        face_blocks = [interior_codes[:n_interior]]
        owner_blocks = [interior_owners[:n_interior]]
        boundary: list[dict] = []
        start = n_interior
        ordered_names = list(boundary_names)
        if self.surface is not None:
            ordered_names.append(self.wall_patch_name)
        for name in ordered_names:
            codes = np.asarray(patch_codes[name], dtype=code_dtype).reshape(-1, 4)
            owners = np.asarray(patch_owners[name], dtype=np.int32)
            face_blocks.append(codes)
            owner_blocks.append(owners)
            boundary.append(
                {
                    "name": name,
                    "start_face": start,
                    "n_faces": len(codes),
                    "type": "wall" if name == self.wall_patch_name else "patch",
                }
            )
            start += len(codes)

        encoded_faces = np.ascontiguousarray(np.vstack(face_blocks), dtype=code_dtype)
        owners = np.ascontiguousarray(np.concatenate(owner_blocks), dtype=np.int32)
        neighbour_array = np.ascontiguousarray(interior_neighbours[:n_interior], dtype=np.int32)
        return encoded_faces, owners, neighbour_array, boundary, levels

    def build(self) -> dict:
        """Generate and return a solver-native ``mesh_data`` dictionary.

        In preserved-body mode the body planes are exact lattice planes by
        construction, and the returned mesh is verified before returning: no
        ordinary fluid cell may overlap the solid with positive volume, and the
        wall patch must lie exactly on the input STL bounds.
        """
        base_counts = self._base_counts()
        requested_sizes = [refinement.cell_size for refinement in self.refinements]
        if self.surface_cell_size is not None:
            requested_sizes.append(self.surface_cell_size)
        max_level = max(
            (_dyadic_level(self.max_cell_size, size) for size in requested_sizes),
            default=0,
        )
        if self.surface_bounds is not None:
            max_level = max(max_level, self._resolved_max_level)
        h_min = self.max_cell_size / (2**max_level)
        base_width = 2**max_level
        limits = tuple(count * base_width for count in base_counts)
        body_region: _SolidRegion | None
        if self.exact_surface_components:
            solid = _CompositeIntegerBox(
                tuple(self._integer_box(bounds, h_min) for bounds in self.exact_surface_components)
            )
            body_region = solid
        elif (
            self.surface_bounds is not None
            and self.surface is not None
            and self.surface.kind == "box"
        ):
            if self._body_lattice_indices is None:
                raise RuntimeError("Internal lattice resolution missing for preserved body")
            scale = 2 ** (max_level - self._resolved_max_level)
            solid = _IntegerBox(*(index * scale for index in self._body_lattice_indices))
            body_region = solid
        elif self._surface_index is not None:
            if self.surface_bounds is None:
                raise RuntimeError("Surface bounds are missing for the curved body")
            origin = (self.domain[0], self.domain[2], self.domain[4])
            solid = _SurfaceSolid(
                self._surface_index,
                h_min,
                origin,
                self.surface_exclusion_distance,
            )
            body_bounds = self.surface_bounds
            if self.surface_exclusion_distance > 0.0:
                distance = self.surface_exclusion_distance
                body_bounds = tuple(
                    float(value + (distance if index % 2 else -distance))
                    for index, value in enumerate(self.surface_bounds)
                )
            body_region = self._integer_box(cast(Bounds, body_bounds), h_min)
        else:
            solid = None
            body_region = None
        regions = self._refinement_regions(h_min, max_level, limits, body_region)
        leaves = self._build_leaves(base_counts, max_level, solid, regions)
        encoded_faces, owners, neighbours, boundary, levels = self._extract_topology(
            leaves, max_level, limits
        )

        point_codes = np.unique(encoded_faces)
        if len(point_codes) > np.iinfo(np.int32).max:
            raise MemoryError("Adaptive mesh has too many points for int32 face connectivity")
        faces = np.empty(encoded_faces.shape, dtype=np.int32)
        for start in range(0, len(encoded_faces), 250_000):
            stop = min(start + 250_000, len(encoded_faces))
            faces[start:stop] = np.searchsorted(point_codes, encoded_faces[start:stop]).astype(
                np.int32
            )
        del encoded_faces

        nx, ny, _nz = limits
        sx, sy = nx + 1, ny + 1
        px = point_codes % sx
        yz = point_codes // sx
        py = yz % sy
        pz = yz // sy
        points = np.column_stack((px, py, pz)).astype(np.float64)
        points *= h_min
        points += np.asarray(self.domain[::2], dtype=np.float64)

        mesh_data = {
            "vertex_position": np.ascontiguousarray(points),
            "faces": faces,
            "owners": owners,
            "neighbours": neighbours,
            "boundary": boundary,
            "n_cells": len(leaves),
            "n_faces": len(faces),
            "n_interior_faces": len(neighbours),
            "n_points": len(points),
            "cell_levels": levels,
            "cell_sizes": np.asarray(self.max_cell_size / np.power(2.0, levels), dtype=np.float32),
            "mesh_generation": {
                "method": "cartesian_octree",
                "max_cell_size": self.max_cell_size,
                "requested_max_cell_size": self.requested_max_cell_size,
                "finest_cell_size": h_min,
                "max_level": max_level,
                "base_counts": base_counts,
                "requested_domain": self.requested_domain,
                "effective_domain": self.effective_domain,
                "padding_per_face": self.padding_per_face,
                "preserve_body_geometry": self.preserve_body_geometry,
                "surface_file": self.surface_file,
                "surface_sha256": self.surface.sha256 if self.surface is not None else None,
                "surface_bounds": self.surface_bounds,
                "surface_triangle_count": (
                    len(self.surface.triangles) if self.surface is not None else 0
                ),
                "wall_patch_name": self.wall_patch_name or None,
                "surface_may_cross_domain_boundary": self.surface_may_cross_domain_boundary,
                "preserve_outer_patches": self.preserve_outer_patches,
                "attribution": "Inspired by cfMesh cartesianMesh (Franjo Juretic / Creative Fields)",
            },
        }

        if self.include_cell_vertex_indices:
            # All geometrical cells are cuboids even where a coarse face is
            # topologically split.  Native hex vertices keep VTK output compact.
            cell_vertex_indices = np.empty((len(leaves), 8), dtype=np.int32)
            for start in range(0, len(leaves), 250_000):
                stop = min(start + 250_000, len(leaves))
                block = leaves[start:stop]
                x0, y0, z0, width = (block[:, index].astype(np.int64) for index in range(4))
                x1, y1, z1 = x0 + width, y0 + width, z0 + width

                def encode(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
                    return x + sx * (y + sy * z)

                codes = np.column_stack(
                    (
                        encode(x0, y0, z0),
                        encode(x1, y0, z0),
                        encode(x1, y1, z0),
                        encode(x0, y1, z0),
                        encode(x0, y0, z1),
                        encode(x1, y0, z1),
                        encode(x1, y1, z1),
                        encode(x0, y1, z1),
                    )
                )
                indices = np.searchsorted(point_codes, codes)
                if np.any(indices >= len(point_codes)) or not np.array_equal(
                    point_codes[indices], codes
                ):
                    raise RuntimeError("A Cartesian cell corner is absent from the face points")
                cell_vertex_indices[start:stop] = indices.astype(np.int32)
            mesh_data["cell_vertex_indices"] = cell_vertex_indices
            mesh_data["cell_type_code"] = np.full(len(leaves), 5, dtype=np.int32)

        if (
            self.surface_bounds is not None
            and self.surface is not None
            and self.surface.kind == "box"
        ):
            validate_no_fluid_solid_overlap(mesh_data, self.surface_bounds)
            _validate_wall_on_surface(mesh_data, self.surface_bounds, self.wall_patch_name)
        elif self.exact_surface_components:
            for bounds in self.exact_surface_components:
                validate_no_fluid_solid_overlap(mesh_data, bounds)
        return mesh_data

    def __call__(self) -> dict:
        """Allow the mesher itself to be passed to ``create_fvm_solver``."""
        return self.build()


def build_cartesian_background(
    domain: Bounds,
    max_cell_size: float,
    **kwargs,
) -> dict:
    """Functional wrapper around :class:`CartesianOctree`."""
    return CartesianOctree(domain, max_cell_size, **kwargs).build()


__all__ = [
    "CartesianOctree",
    "Bounds",
    "BoxRefinement",
    "build_cartesian_background",
]
