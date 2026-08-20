"""Native adaptive Cartesian meshes for the finite-volume solver.

This module implements the part of the cfMesh ``cartesianMesh`` workflow that
OpenONDA's axis-aligned verification cases need: triangulated-surface input,
dyadic local refinement, 2:1 transition bands, removal of an axis-aligned
solid, and direct construction of the solver's face-based ``mesh_data``
dictionary. It does not read or write an external solver case and does not
require an external mesher.

Algorithmic lineage and credit
------------------------------
The workflow is inspired by the open-source cfMesh Cartesian mesher by Dr.
Franjo Juretic and Creative Fields, Ltd.  cfMesh creates an octree template,
extracts a predominantly hexahedral/polyhedral mesh, maps it to the input
surface, and optionally creates boundary layers.  This is an independent
Python implementation of the axis-aligned template/extraction stages; it does
not copy cfMesh source code and does not yet implement arbitrary surface
mapping or boundary-layer extrusion.

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

from .surface_classification import SurfaceIndex
from .triangulated_surface import SurfaceBounds as Bounds
from .triangulated_surface import TriangulatedSurface
from .validation import (
    validate_curved_wall_conformance,
    validate_no_fluid_solid_overlap,
)

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


class _SurfaceSolid:
    """Real curved-surface classifier, duck-typed to ``_IntegerBox``'s
    ``contains``/``overlaps`` interface so the octree traversal in
    ``_build_leaves`` needs no branching between the box and general paths.
    """

    def __init__(
        self, index: SurfaceIndex, h_min: float, origin: tuple[float, float, float]
    ) -> None:
        self.index = index
        self.h_min = h_min
        self.origin = np.asarray(origin, dtype=np.float64)

    def _world_bounds(
        self, x0: int, x1: int, y0: int, y1: int, z0: int, z1: int
    ) -> tuple[np.ndarray, np.ndarray]:
        lo = self.origin + np.array([x0, y0, z0], dtype=np.float64) * self.h_min
        hi = self.origin + np.array([x1, y1, z1], dtype=np.float64) * self.h_min
        return lo, hi

    def contains(self, x0: int, x1: int, y0: int, y1: int, z0: int, z1: int) -> bool:
        lo, hi = self._world_bounds(x0, x1, y0, y1, z0, z1)
        if self.index.box_intersects_surface(lo, hi):
            return False
        centre = 0.5 * (lo + hi)
        return bool(self.index.is_inside(centre[None, :])[0])

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


def _fitted_background_size(domain: Bounds, requested: float) -> float:
    """Return the largest isotropic spacing no larger than ``requested`` that tiles ``domain``."""
    extents = [
        Fraction(str(domain[2 * axis + 1] - domain[2 * axis])).limit_denominator(1_000_000)
        for axis in range(3)
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
    maximum_level: int
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
            minimum_n = math.ceil(extents[reference] / h_requested)
            n = int(math.ceil(minimum_n / denominator_lcm)) * denominator_lcm
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
        maximum_level=level,
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
    vertices = np.asarray(mesh_data["points"])[np.unique(faces)]
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


def _hex_volume(corners: np.ndarray) -> float:
    """Divergence-theorem volume of one hexahedron, matching ``geometry.py``'s
    fan-triangulated face formula so a positive result here means the solver's
    own volume computation will also be positive."""
    centre = corners.mean(axis=0)
    volume = 0.0
    for face in _HEX_FACES:
        pts = corners[list(face)]
        face_centre = pts.mean(axis=0)
        sf = np.zeros(3)
        for i in range(4):
            a, b = pts[i], pts[(i + 1) % 4]
            sf += 0.5 * np.cross(a - face_centre, b - face_centre)
        volume += float(np.dot(sf, face_centre - centre)) / 3.0
    return volume


def _conform_wall_to_surface(
    mesh_data: dict,
    surface_index: SurfaceIndex,
    wall_patch_name: str,
    *,
    min_volume_ratio: float = 0.15,
) -> None:
    """Snap wall-patch corner points onto the true curved surface.

    Every wall-adjacent point that is still (numerically) inside the solid is
    projected onto the nearest point of the triangulated surface.  A snap is
    rejected, leaving the point on its original Cartesian lattice position,
    if it would drive any cell referencing that point to a non-positive
    volume or shrink it by more than ``min_volume_ratio`` of its original
    volume -- cfMesh's real robustness fallback: partial conformance (a small
    stairstep at that one corner) beats an invalid mesh.
    """
    points = mesh_data["points"]
    cell_vertices = mesh_data.get("cell_vertices")
    if cell_vertices is None:
        return
    (wall,) = [patch for patch in mesh_data["boundary"] if patch["name"] == wall_patch_name]
    faces = np.asarray(mesh_data["faces"])
    first, count = int(wall["start_face"]), int(wall["n_faces"])
    wall_point_ids = np.unique(faces[first : first + count])

    point_cells: dict[int, list[int]] = {}
    for cell_id, corners in enumerate(cell_vertices):
        for corner in corners:
            point_cells.setdefault(int(corner), []).append(cell_id)

    inside = surface_index.is_inside(points[wall_point_ids])
    for point_id, is_inside_solid in zip(wall_point_ids, inside, strict=True):
        if not is_inside_solid:
            continue
        affected = point_cells.get(int(point_id), [])
        if not affected:
            continue
        before = [_hex_volume(points[cell_vertices[c]]) for c in affected]
        if min(before) <= 0.0:
            continue
        target, _distance = surface_index.nearest_point(points[point_id])
        original = points[point_id].copy()
        points[point_id] = target
        after = [_hex_volume(points[cell_vertices[c]]) for c in affected]
        if (
            min(after) <= 0.0
            or min(a / b for a, b in zip(after, before, strict=True)) < min_volume_ratio
        ):
            points[point_id] = original


class AdaptiveCartesianMesher:
    """Build an adaptive, body-fitted Cartesian mesh directly in memory.

    Parameters
    ----------
    domain:
        Axis-aligned fluid-domain bounds ``(xmin, xmax, ymin, ymax, zmin,
        zmax)``.
    max_cell_size:
        Background cell size.  Every domain extent must be an integer multiple
        of this value, which keeps the outer coupling lattice exact.
    surface_file:
        Location of the watertight STL surface that defines the solid. The
        current Cartesian subset supports closed axis-aligned solids.
    wall_patch_name:
        Boundary-patch name assigned to faces extracted from ``surface_file``.
        It must be supplied together with the surface location.
    surface_cell_size:
        Requested size in the first fluid-cell layer around the STL surface.
        Refinement coarsens outwards in automatically generated 2:1 bands.
    refinements:
        Additional box refinements, such as a resolved wake region.
    merge_outer_patch:
        Merge the six outer sides into one patch (the coupled FVM--VPM case),
        or leave the conventional inlet/outlet/ymin/ymax/zmin/zmax patches.
    preserve_body_geometry:
        Guarantee that the input body coordinates are immutable.  The body
        faces become exact Cartesian lattice planes, the outer domain is padded
        outward as needed, and no ordinary fluid cell may overlap the solid
        with positive volume.  This is the only supported mode; the legacy
        body-snapping path was removed.

    Notes
    -----
    Coarse cells next to finer cells are represented as valid polyhedra with
    multiple coplanar subfaces.  Geometrically all cells remain cuboids.  This
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
        wall_patch_name: str | None = None,
        surface_cell_size: float | None = None,
        refinements: tuple[BoxRefinement, ...] = (),
        merge_outer_patch: str | None = None,
        include_cell_vertices: bool = True,
        preserve_body_geometry: bool = True,
    ) -> None:
        _validate_bounds(domain, "domain")
        if not math.isfinite(max_cell_size) or max_cell_size <= 0.0:
            raise ValueError("max_cell_size must be finite and positive")
        if (surface_file is None) != (wall_patch_name is None):
            raise ValueError("surface_file and wall_patch_name must be supplied together")
        if surface_file is None and surface_cell_size is not None:
            raise ValueError("surface_cell_size requires surface_file")
        if wall_patch_name is not None and not wall_patch_name.strip():
            raise ValueError("wall_patch_name must not be empty")

        requested_domain: Bounds = cast(Bounds, tuple(float(value) for value in domain))
        surface = TriangulatedSurface.from_stl(surface_file) if surface_file is not None else None
        if surface is not None:
            surface_bounds = surface.bounds
            if not all(
                requested_domain[2 * axis]
                < surface_bounds[2 * axis]
                < surface_bounds[2 * axis + 1]
                < requested_domain[2 * axis + 1]
                for axis in range(3)
            ):
                raise ValueError("STL surface must lie strictly inside domain")
        if wall_patch_name is not None and merge_outer_patch == wall_patch_name:
            raise ValueError("Outer and wall patch names must differ")
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

        fitted_max_cell_size = _fitted_background_size(requested_domain, max_cell_size)
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
            if surface.kind == "box":
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
        self._resolved_max_level = (
            resolved.maximum_level if resolved is not None else requested_level
        )
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
        self.include_cell_vertices = include_cell_vertices
        self.preserve_body_geometry = preserve_body_geometry

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
            count = int(round(extent / self.max_cell_size))
            if count < 1 or not math.isclose(
                count * self.max_cell_size,
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
        body_region: _IntegerBox | None,
    ) -> tuple[tuple[_IntegerBox, int], ...]:
        regions: list[tuple[_IntegerBox, int]] = []

        def add_balanced(box: _IntegerBox, target_level: int, first_padding: int = 0) -> None:
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
                add_balanced(self._integer_box(refinement.bounds, h_min), level)

        # Higher levels take precedence in overlap regions.
        regions.sort(key=lambda item: item[1], reverse=True)
        return tuple(regions)

    def _build_leaves(
        self,
        base_counts: tuple[int, int, int],
        max_level: int,
        solid: _IntegerBox | None,
        regions: tuple[tuple[_IntegerBox, int], ...],
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
                and isinstance(solid, _IntegerBox)
                and solid.overlaps(x0, x1, y0, y1, z0, z1)
            ):
                # Exact interval/AABB classification: a leaf may touch the body
                # with zero volume (wall-adjacent fluid) but must never overlap
                # it with positive volume.  Wholly-inside cells are removed
                # above by ``solid.contains``.  Any remaining positive-volume
                # overlap means a body plane is not an exact leaf lattice
                # plane, which preserved mode must reject rather than silently
                # keep or remove on a centroid test.
                overlap_x = min(x1, solid.x1) - max(x0, solid.x0)
                overlap_y = min(y1, solid.y1) - max(y0, solid.y0)
                overlap_z = min(z1, solid.z1) - max(z0, solid.z0)
                if overlap_x > 0 and overlap_y > 0 and overlap_z > 0:
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
        """Encoded point ring with normal along ``axis`` and sign."""
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

        boundary_names = (self.merge_outer_patch,) if self.merge_outer_patch else OUTER_PATCH_NAMES
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
                if self.merge_outer_patch:
                    return self.merge_outer_patch
                return OUTER_PATCH_NAMES[2 * axis + int(positive)]
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
        body_region: _IntegerBox | None
        if (
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
            solid = _SurfaceSolid(self._surface_index, h_min, self.domain[::2])
            body_region = self._integer_box(self.surface_bounds, h_min)
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
            "points": np.ascontiguousarray(points),
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
                "method": "adaptive_cartesian",
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
                "attribution": "Inspired by cfMesh cartesianMesh (Franjo Juretic / Creative Fields)",
            },
        }

        if self.include_cell_vertices:
            # All geometrical cells are cuboids even where a coarse face is
            # topologically split.  Native hex vertices keep VTK output compact.
            cell_vertices = np.empty((len(leaves), 8), dtype=np.int32)
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
                cell_vertices[start:stop] = indices.astype(np.int32)
            mesh_data["cell_vertices"] = cell_vertices
            mesh_data["cell_type_codes"] = np.full(len(leaves), 5, dtype=np.int32)

        if self.surface_bounds is not None and self.surface.kind == "box":
            validate_no_fluid_solid_overlap(mesh_data, self.surface_bounds)
            _validate_wall_on_surface(mesh_data, self.surface_bounds, self.wall_patch_name)
        elif self._surface_index is not None:
            _conform_wall_to_surface(mesh_data, self._surface_index, self.wall_patch_name)
            validate_curved_wall_conformance(
                mesh_data, self.surface.triangles, self.wall_patch_name
            )

        return mesh_data

    def __call__(self) -> dict:
        """Allow the mesher itself to be passed to ``setup_fvm_solver``."""
        return self.build()


def adaptive_cartesian_mesh(
    domain: Bounds,
    max_cell_size: float,
    **kwargs,
) -> dict:
    """Functional wrapper around :class:`AdaptiveCartesianMesher`."""
    return AdaptiveCartesianMesher(domain, max_cell_size, **kwargs).build()


__all__ = ["AdaptiveCartesianMesher", "Bounds", "BoxRefinement", "adaptive_cartesian_mesh"]
